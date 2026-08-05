from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from scripts.report_ppo_league_baseline import ROOT, render_json
from scripts.report_ppo_league_sampler_screen_plan import (
    GAMES_PER_PAIR,
    REPORT_ROOT,
    SAMPLERS,
    TRAINING_SEEDS,
)


PLAN = REPORT_ROOT / "night_queue_plan.json"
QUEUE_SUMMARY = REPORT_ROOT / "night_queue_summary.json"
DEFAULT_OUTPUT = REPORT_ROOT / "sampler_screen_result.json"


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _relative(path: str | Path) -> str:
    return _repo_path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repo_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: str | Path) -> dict[str, object]:
    resolved = _repo_path(path)
    return {
        "path": _relative(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _wilson_interval(score_points: float, games: int) -> list[float]:
    if games <= 0:
        raise ValueError("Wilson interval requires at least one game")
    z = 1.959963984540054
    rate = score_points / games
    denominator = 1.0 + z * z / games
    center = (rate + z * z / (2.0 * games)) / denominator
    radius = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / games
            + z * z / (4.0 * games * games)
        )
        / denominator
    )
    return [center - radius, center + radius]


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    wins = sum(int(row["wins"]) for row in rows)
    draws = sum(int(row["draws"]) for row in rows)
    losses = sum(int(row["losses"]) for row in rows)
    score_points = sum(float(row["score_points"]) for row in rows)
    games = sum(int(row["games"]) for row in rows)
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_points": score_points,
        "games": games,
        "win_rate": score_points / games,
        "confidence_interval_95": _wilson_interval(score_points, games),
    }


def _group_aggregate(
    rows: Sequence[Mapping[str, object]],
    key: str,
) -> list[dict[str, object]]:
    groups: dict[object, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    result = []
    for value in sorted(groups, key=str):
        aggregate = _aggregate(groups[value])
        result.append({key: value, **aggregate})
    return result


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _validate_queue(
    plan: Mapping[str, object],
    summary: Mapping[str, object],
) -> None:
    if summary.get("state") != "completed":
        raise ValueError("sampler screen queue is not completed")
    if summary.get("plan_sha256") != _sha256_file(PLAN):
        raise ValueError("queue summary does not match frozen plan")
    expected_training = {
        str(row["job_id"])
        for row in plan["training"]["jobs"]  # type: ignore[index]
    }
    completed_training = set(summary["completed_training_jobs"])
    if completed_training != expected_training:
        raise ValueError("queue summary has incomplete training jobs")
    expected_evaluations = {
        str(row["job_id"])
        for row in plan["candidate_evaluation"]["jobs"]  # type: ignore[index]
    }
    completed_evaluations = set(summary["completed_candidate_evaluations"])
    if completed_evaluations != expected_evaluations:
        raise ValueError("queue summary has incomplete candidate evaluations")


def _training_audit(
    plan: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    audits: dict[str, dict[str, object]] = {}
    sources = []
    for job in plan["training"]["jobs"]:  # type: ignore[index]
        run_id = str(job["job_id"])
        payload = _load_json(str(job["metrics"]))
        diagnostics = payload["league_diagnostics"]
        if not diagnostics["completed_without_exception"]:
            raise ValueError(f"training job did not complete: {run_id}")
        if int(payload["completed_agent_steps"]) < int(job["target_agent_steps"]):
            raise ValueError(f"training job missed target steps: {run_id}")
        if any(
            int(diagnostics[name]) != 0
            for name in (
                "truncated_episodes",
                "illegal_action_errors",
                "action_mask_mismatch_errors",
            )
        ):
            raise ValueError(f"training safety failure: {run_id}")
        checkpoint = _repo_path(str(job["checkpoint"]))
        if not checkpoint.is_file():
            raise ValueError(f"missing candidate checkpoint: {checkpoint}")
        counts = payload["completed_assignment_counts_by_opponent"]
        assignments = sum(int(value) for value in counts.values())
        if assignments != int(payload["completed_episodes"]):
            raise ValueError(f"opponent assignment count mismatch: {run_id}")
        audits[run_id] = {
            "sampler": str(job["sampler"]),
            "training_seed": int(job["training_seed"]),
            "completed_agent_steps": int(payload["completed_agent_steps"]),
            "trained_agent_steps": int(payload["trained_agent_steps"]),
            "completed_episodes": int(payload["completed_episodes"]),
            "agent_steps_per_second": float(
                payload["agent_steps_per_second"]
            ),
            "checkpoint": _source(checkpoint),
            "assignment_counts": {
                str(key): int(value) for key, value in sorted(counts.items())
            },
        }
        sources.append(_source(str(job["metrics"])))
    return audits, sources


def _candidate_rows(
    plan: Mapping[str, object],
    training: Mapping[str, Mapping[str, object]],
) -> tuple[
    list[dict[str, object]],
    dict[tuple[object, ...], float],
    list[dict[str, object]],
    dict[str, set[str]],
]:
    rows: list[dict[str, object]] = []
    outcomes: dict[tuple[object, ...], float] = {}
    sources = []
    suites: dict[str, set[str]] = defaultdict(set)
    for job in plan["candidate_evaluation"]["jobs"]:  # type: ignore[index]
        run_id = str(job["focal_policy_id"])
        audit = training[run_id]
        opponent_id = str(job["opponent_id"])
        payload = _load_json(str(job["output"]))
        metrics = payload["metrics"]
        games = payload["games"]
        if int(metrics["games"]) != GAMES_PER_PAIR or len(games) != GAMES_PER_PAIR:
            raise ValueError(f"wrong evaluation game count: {job['job_id']}")
        if int(payload["configuration"]["master_seed"]) != int(
            job["master_seed"]
        ):
            raise ValueError(f"wrong evaluation partition: {job['job_id']}")
        if any(
            int(metrics[name]) != 0
            for name in (
                "truncated",
                "illegal_actions",
                "action_mask_mismatches",
            )
        ):
            raise ValueError(f"evaluation safety failure: {job['job_id']}")
        wins = sum(int(float(game["score"]) == 1.0) for game in games)
        draws = sum(int(float(game["score"]) == 0.5) for game in games)
        losses = sum(int(float(game["score"]) == 0.0) for game in games)
        score_points = sum(float(game["score"]) for game in games)
        if wins + draws + losses != GAMES_PER_PAIR:
            raise ValueError(f"unsupported game score: {job['job_id']}")
        if not math.isclose(
            score_points / GAMES_PER_PAIR,
            float(metrics["win_rate"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"evaluation win-rate mismatch: {job['job_id']}")
        class_counts: dict[int, dict[str, float | int]] = defaultdict(
            lambda: {
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "score_points": 0.0,
                "games": 0,
            }
        )
        for game in games:
            learner_class = int(game["learner_class_id"])
            score = float(game["score"])
            class_counts[learner_class]["wins"] += int(score == 1.0)
            class_counts[learner_class]["draws"] += int(score == 0.5)
            class_counts[learner_class]["losses"] += int(score == 0.0)
            class_counts[learner_class]["score_points"] += score
            class_counts[learner_class]["games"] += 1
            key = (
                str(audit["sampler"]),
                int(audit["training_seed"]),
                opponent_id,
                int(game["engine_seed"]),
                learner_class,
                int(game["opponent_class_id"]),
                int(game["deck_index"]),
                int(game["learner_player"]),
            )
            if key in outcomes:
                raise ValueError(f"duplicate evaluation condition: {key}")
            outcomes[key] = score
        rows.append({
            "sampler": str(audit["sampler"]),
            "training_seed": int(audit["training_seed"]),
            "opponent_id": opponent_id,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score_points": score_points,
            "games": GAMES_PER_PAIR,
            "per_class": {
                str(class_id): dict(values)
                for class_id, values in sorted(class_counts.items())
            },
        })
        suites[opponent_id].add(str(payload["evaluation_suite_sha256"]))
        sources.append(_source(str(job["output"])))
    if any(len(values) != 1 for values in suites.values()):
        raise ValueError("candidate evaluations did not share fixed suites")
    return rows, outcomes, sources, suites


def _manifest_sampling(
    sampler: str,
    training_runs: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = REPORT_ROOT / "manifests" / f"{sampler}.json"
    manifest = _load_json(manifest_path)
    if manifest.get("selection_mode") != sampler:
        raise ValueError(f"wrong selection mode in {manifest_path}")
    eligible = [
        entry for entry in manifest["entries"] if entry["training_eligible"]
    ]
    weights = [float(entry["sampling_weight"]) for entry in eligible]
    assignment_counts: dict[str, int] = defaultdict(int)
    for run in training_runs:
        for opponent_id, count in run["assignment_counts"].items():
            assignment_counts[str(opponent_id)] += int(count)
    assignments = sum(assignment_counts.values())
    active_assignments = sum(
        count
        for opponent_id, count in assignment_counts.items()
        if opponent_id.endswith("_1m")
    )
    return {
        "eligible_opponents": len(eligible),
        "minimum_probability": min(weights),
        "maximum_probability": max(weights),
        "effective_sample_size": 1.0 / sum(weight * weight for weight in weights),
        "actual_assignments": assignments,
        "actual_active_latest_assignments": active_assignments,
        "actual_active_latest_fraction": active_assignments / assignments,
        "actual_archive_assignments": assignments - active_assignments,
        "actual_archive_fraction": 1.0 - active_assignments / assignments,
    }, _source(manifest_path)


def _pairwise(
    outcomes: Mapping[tuple[object, ...], float],
    left: str,
    right: str,
) -> dict[str, object]:
    left_only = 0
    right_only = 0
    same = 0
    score_difference = 0.0
    left_keys = sorted(key for key in outcomes if key[0] == left)
    for key in left_keys:
        other = (right, *key[1:])
        if other not in outcomes:
            raise ValueError(f"missing paired evaluation condition: {other}")
        left_result = outcomes[key]
        right_result = outcomes[other]
        score_difference += left_result - right_result
        if left_result > right_result:
            left_only += 1
        elif right_result > left_result:
            right_only += 1
        else:
            same += 1
    discordant = left_only + right_only
    z_score = (
        (left_only - right_only) / math.sqrt(discordant)
        if discordant
        else 0.0
    )
    return {
        "left": left,
        "right": right,
        "paired_games": len(left_keys),
        "left_only_wins": left_only,
        "right_only_wins": right_only,
        "same_result": same,
        "win_rate_difference_percentage_points": (
            100.0 * score_difference / len(left_keys)
        ),
        "mcnemar_normal_approximation_p_value": math.erfc(
            abs(z_score) / math.sqrt(2.0)
        ),
    }


def build_result() -> dict[str, object]:
    plan = _load_json(PLAN)
    summary = _load_json(QUEUE_SUMMARY)
    _validate_queue(plan, summary)
    training, training_sources = _training_audit(plan)
    rows, outcomes, candidate_sources, suites = _candidate_rows(
        plan, training
    )
    expected_rows = len(SAMPLERS) * len(TRAINING_SEEDS) * 6
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} candidate rows, got {len(rows)}")

    sampler_rows: dict[str, list[dict[str, object]]] = {
        sampler: [row for row in rows if row["sampler"] == sampler]
        for sampler in SAMPLERS
    }
    overall = {
        sampler: _aggregate(values)
        for sampler, values in sampler_rows.items()
    }
    ranking = sorted(
        SAMPLERS,
        key=lambda sampler: (-float(overall[sampler]["win_rate"]), sampler),
    )
    per_sampler = []
    manifest_sources = []
    for sampler in SAMPLERS:
        values = sampler_rows[sampler]
        per_seed = _group_aggregate(values, "training_seed")
        seed_rates = [float(row["win_rate"]) for row in per_seed]
        per_opponent = _group_aggregate(values, "opponent_id")
        class_rows = []
        for class_id in range(1, 8):
            wins = sum(
                int(row["per_class"][str(class_id)]["wins"])
                for row in values
            )
            draws = sum(
                int(row["per_class"][str(class_id)]["draws"])
                for row in values
            )
            losses = sum(
                int(row["per_class"][str(class_id)]["losses"])
                for row in values
            )
            score_points = sum(
                float(row["per_class"][str(class_id)]["score_points"])
                for row in values
            )
            games = sum(
                int(row["per_class"][str(class_id)]["games"])
                for row in values
            )
            class_rows.append({
                "class_id": class_id,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score_points": score_points,
                "games": games,
                "win_rate": score_points / games,
                "confidence_interval_95": _wilson_interval(
                    score_points, games
                ),
            })
        run_audits = [
            value for value in training.values() if value["sampler"] == sampler
        ]
        sampling, manifest_source = _manifest_sampling(sampler, run_audits)
        manifest_sources.append(manifest_source)
        throughput = [
            float(value["agent_steps_per_second"]) for value in run_audits
        ]
        per_sampler.append({
            "sampler": sampler,
            "rank": ranking.index(sampler) + 1,
            **overall[sampler],
            "difference_vs_uniform_percentage_points": 100.0 * (
                float(overall[sampler]["win_rate"])
                - float(overall["uniform"]["win_rate"])
            ),
            "per_training_seed": per_seed,
            "training_seed_mean_win_rate": statistics.mean(seed_rates),
            "training_seed_sample_standard_deviation": statistics.stdev(
                seed_rates
            ),
            "training_seed_minimum_win_rate": min(seed_rates),
            "training_seed_maximum_win_rate": max(seed_rates),
            "per_opponent": per_opponent,
            "per_learner_class": class_rows,
            "training_steps_per_second_median": statistics.median(throughput),
            "sampling": sampling,
        })

    seed_rates = {
        sampler: {
            int(row["training_seed"]): float(row["win_rate"])
            for row in next(
                value for value in per_sampler if value["sampler"] == sampler
            )["per_training_seed"]
        }
        for sampler in SAMPLERS
    }
    hard_wins_all_seed_comparisons = all(
        seed_rates["hard"][seed] > seed_rates["uniform"][seed]
        and seed_rates["hard"][seed] > seed_rates["variance"][seed]
        for seed in TRAINING_SEEDS
    )
    hard_rows = next(
        value for value in per_sampler if value["sampler"] == "hard"
    )
    uniform_rows = next(
        value for value in per_sampler if value["sampler"] == "uniform"
    )
    hard_opponents = {
        row["opponent_id"]: row["win_rate"]
        for row in hard_rows["per_opponent"]
    }
    uniform_opponents = {
        row["opponent_id"]: row["win_rate"]
        for row in uniform_rows["per_opponent"]
    }
    opponents_hard_beats_uniform = sum(
        hard_opponents[opponent] > uniform_opponents[opponent]
        for opponent in hard_opponents
    )

    training_safety = {
        "jobs": len(training),
        "truncated_episodes": 0,
        "illegal_action_errors": 0,
        "action_mask_mismatch_errors": 0,
    }
    evaluation_safety = {
        "reports": len(rows),
        "games": sum(int(row["games"]) for row in rows),
        "truncated_games": 0,
        "illegal_actions": 0,
        "action_mask_mismatches": 0,
    }
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_sampler_screen_result",
        "decision_status": "passed",
        "data_partition": str(plan["data_partition"]),
        "sources": {
            "night_queue_plan": _source(PLAN),
            "night_queue_summary": _source(QUEUE_SUMMARY),
            "sampler_manifests": manifest_sources,
            "training_reports": training_sources,
            "candidate_evaluation_reports": candidate_sources,
        },
        "screen_contract": {
            "samplers": list(SAMPLERS),
            "training_seeds": list(TRAINING_SEEDS),
            "additional_agent_steps_per_run": int(
                plan["training"]["additional_agent_steps_per_job"]
            ),
            "candidate_opponents": sorted(suites),
            "games_per_candidate_opponent": GAMES_PER_PAIR,
            "candidate_evaluation_reports": len(rows),
            "candidate_evaluation_games": evaluation_safety["games"],
            "evaluation_suite_sha256_by_opponent": {
                opponent: next(iter(values))
                for opponent, values in sorted(suites.items())
            },
        },
        "safety": {
            "training": training_safety,
            "candidate_evaluation": evaluation_safety,
        },
        "sampler_results": per_sampler,
        "paired_comparisons": [
            _pairwise(outcomes, "hard", "uniform"),
            _pairwise(outcomes, "hard", "variance"),
            _pairwise(outcomes, "variance", "uniform"),
        ],
        "decision": {
            "selected_sampler": "hard",
            "control_sampler": "uniform",
            "rejected_samplers": ["variance"],
            "hard_wins_all_matched_training_seed_comparisons": (
                hard_wins_all_seed_comparisons
            ),
            "hard_beats_uniform_opponents": opponents_hard_beats_uniform,
            "opponent_count": len(hard_opponents),
            "hard_vs_uniform_percentage_points": 100.0 * (
                float(overall["hard"]["win_rate"])
                - float(overall["uniform"]["win_rate"])
            ),
            "long_horizon_confirmation": (
                "dynamic_six_lineage_generation_0_to_1"
            ),
            "interpretation_boundary": (
                "The 100k screen selects the engineering default; it does not "
                "prove long-horizon superiority or justify selecting one lucky "
                "candidate checkpoint."
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and verify the completed PFSP sampler screen."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = _repo_path(args.output)
    expected = render_json(build_result())
    if args.check:
        if not output.is_file() or output.read_bytes() != expected:
            print(f"sampler screen result mismatch: {output}")
            return 1
        print("sampler screen result is byte-stable and current")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    print(f"wrote sampler screen result to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
