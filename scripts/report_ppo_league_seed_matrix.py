from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from swb.rl.versioning import stable_json_sha256

from scripts.report_ppo_league_baseline import render_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = Path("data/reports/ppo_7x7_seed_matrix_20260804")
OUTPUT_DIRECTORY = Path("data/reports/league_training")
CHECKPOINT_REGISTRY = OUTPUT_DIRECTORY / "checkpoint_registry.json"
BASELINE_MANIFEST = OUTPUT_DIRECTORY / "baseline_manifest.json"
EVALUATION_PROTOCOL = OUTPUT_DIRECTORY / "evaluation_protocol.json"
ONE_M_SEEDS = tuple(range(20260903, 20260909))
THREE_M_SEEDS = (20260831, 20260901, 20260902)
SIGNIFICANT_CYCLE_THRESHOLD = 0.55


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


def _load_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(_repo_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {_repo_path(path)}")
    return payload


def wilson_interval(
    successes: float,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> list[float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    spread = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [max(0.0, center - spread), min(1.0, center + spread)]


def _model_id(seed: int, tier: str) -> str:
    return f"seed_{seed}_{tier}"


def _parse_pair_summary(entry: Mapping[str, object]) -> tuple[int, str, int, str]:
    return (
        int(entry["learner_seed"]),
        str(entry["learner_tier"]),
        int(entry["opponent_seed"]),
        str(entry["opponent_tier"]),
    )


def _expected_pair_ids() -> set[str]:
    expected = set()
    for left_index, left in enumerate(ONE_M_SEEDS):
        for right in ONE_M_SEEDS[left_index + 1 :]:
            expected.add(f"1m_{left}_vs_1m_{right}")
    for anchor in THREE_M_SEEDS:
        for candidate in ONE_M_SEEDS:
            expected.add(f"3m_{anchor}_vs_1m_{candidate}")
    return expected


def find_cycles(
    score_by_edge: Mapping[tuple[str, str], float],
    model_ids: Sequence[str],
    *,
    threshold: float,
) -> list[dict[str, object]]:
    cycles: list[dict[str, object]] = []
    ordered = tuple(sorted(model_ids))
    for first_index, first in enumerate(ordered):
        for second_index in range(first_index + 1, len(ordered)):
            second = ordered[second_index]
            for third_index in range(second_index + 1, len(ordered)):
                third = ordered[third_index]
                orientations = (
                    (first, second, third),
                    (first, third, second),
                )
                for cycle in orientations:
                    edges = tuple(zip(cycle, (*cycle[1:], cycle[0])))
                    scores = [score_by_edge.get(edge) for edge in edges]
                    if all(score is not None and score > threshold for score in scores):
                        cycles.append({
                            "models": list(cycle),
                            "edges": [
                                {
                                    "winner": winner,
                                    "loser": loser,
                                    "score": score_by_edge[(winner, loser)],
                                }
                                for winner, loser in edges
                            ],
                        })
    return cycles


def _aggregate_games(games: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not games:
        raise ValueError("cannot aggregate an empty game sequence")
    scores = [float(game["score"]) for game in games]
    trials = len(scores)
    successes = sum(scores)
    side_scores: dict[str, list[float]] = defaultdict(list)
    for game, score in zip(games, scores):
        side_scores[str(game["learner_player"])].append(score)
    return {
        "games": trials,
        "wins": sum(score == 1.0 for score in scores),
        "draws": sum(score == 0.5 for score in scores),
        "losses": sum(score == 0.0 for score in scores),
        "score_rate": successes / trials,
        "confidence_interval_95": wilson_interval(successes, trials),
        "side_score_rates": {
            side: sum(values) / len(values)
            for side, values in sorted(side_scores.items())
        },
        "average_turn": sum(int(game["turn"]) for game in games) / trials,
        "average_agent_steps": (
            sum(int(game["agent_steps"]) for game in games) / trials
        ),
        "terminated": sum(bool(game["terminated"]) for game in games),
        "truncated": sum(bool(game["truncated"]) for game in games),
        "illegal_actions": sum(int(game["illegal_actions"]) for game in games),
        "action_mask_mismatches": sum(
            int(game["action_mask_mismatches"]) for game in games
        ),
    }


def _reverse_game(game: Mapping[str, object]) -> dict[str, object]:
    reversed_game = dict(game)
    reversed_game.update({
        "score": 1.0 - float(game["score"]),
        "learner_player": 1 - int(game["learner_player"]),
        "learner_class_id": int(game["opponent_class_id"]),
        "learner_class_name": game["opponent_class_name"],
        "opponent_class_id": int(game["learner_class_id"]),
        "opponent_class_name": game["learner_class_name"],
    })
    return reversed_game


def build_seed_payoff_matrix(
    *,
    source_directory: str | Path = SOURCE_DIRECTORY,
    checkpoint_registry_path: str | Path = CHECKPOINT_REGISTRY,
) -> tuple[dict[str, object], dict[str, object]]:
    source = _repo_path(source_directory)
    launcher = _load_json(source / "launcher_status.json")
    summary = _load_json(source / "summary.json")
    registry = _load_json(checkpoint_registry_path)
    registry_by_seed = {
        int(entry["seed"]): entry for entry in registry["entries"]
    }

    expected_ids = _expected_pair_ids()
    summary_pairs = summary.get("pairs")
    if not isinstance(summary_pairs, list):
        raise ValueError("source summary has no pairs list")
    summary_by_id: dict[str, Mapping[str, object]] = {}
    duplicate_ids = []
    for pair in summary_pairs:
        pair_id = str(pair["pair_id"])
        if pair_id in summary_by_id:
            duplicate_ids.append(pair_id)
        summary_by_id[pair_id] = pair
    observed_ids = set(summary_by_id)
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    if launcher.get("state") != "completed":
        raise ValueError("source evaluation launcher is not completed")
    if missing_ids or unexpected_ids or duplicate_ids:
        raise ValueError(
            "source matrix topology mismatch: "
            f"missing={missing_ids}, unexpected={unexpected_ids}, "
            f"duplicates={duplicate_ids}"
        )

    candidate_ids = [_model_id(seed, "1m") for seed in ONE_M_SEEDS]
    anchor_ids = [_model_id(seed, "3m") for seed in THREE_M_SEEDS]
    model_ids = [*candidate_ids, *anchor_ids]
    size = len(model_ids)
    model_index = {model_id: index for index, model_id in enumerate(model_ids)}
    score_matrix: list[list[float | None]] = [
        [None for _ in range(size)] for _ in range(size)
    ]
    payoff_matrix: list[list[float | None]] = [
        [None for _ in range(size)] for _ in range(size)
    ]
    for index in range(size):
        score_matrix[index][index] = 0.5
        payoff_matrix[index][index] = 0.0

    pair_records = []
    all_games: list[Mapping[str, object]] = []
    candidate_games_by_model: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    anchor_games_by_model: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    class_games_internal: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    class_games_all: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    class_games_internal_both_sides: dict[
        int, list[Mapping[str, object]]
    ] = defaultdict(list)
    class_cells_internal_both_sides: dict[
        tuple[int, int], list[Mapping[str, object]]
    ] = defaultdict(list)
    score_by_edge: dict[tuple[str, str], float] = {}
    close_pairs = []
    source_reports = []
    checkpoint_mismatches = []
    version_signatures = set()

    for pair_id in sorted(expected_ids):
        summary_pair = summary_by_id[pair_id]
        learner_seed, learner_tier, opponent_seed, opponent_tier = (
            _parse_pair_summary(summary_pair)
        )
        learner_id = _model_id(learner_seed, learner_tier)
        opponent_id = _model_id(opponent_seed, opponent_tier)
        report_path = source / f"{pair_id}.json"
        report = _load_json(report_path)
        configuration = report["configuration"]
        metrics = report["metrics"]
        games = report["games"]
        if not isinstance(games, list) or len(games) != 196:
            raise ValueError(f"{pair_id} does not contain exactly 196 games")
        if len({int(game["engine_seed"]) for game in games}) != len(games):
            raise ValueError(f"{pair_id} has duplicate engine seeds")
        if len(metrics["per_matchup"]) != 49:
            raise ValueError(f"{pair_id} does not contain 49 class cells")
        if any(int(cell["games"]) != 4 for cell in metrics["per_matchup"].values()):
            raise ValueError(f"{pair_id} class cells are not four games each")

        learner_expected = registry_by_seed[learner_seed]["checkpoint"]["sha256"]
        opponent_expected = registry_by_seed[opponent_seed]["checkpoint"]["sha256"]
        learner_actual = report["checkpoint"]["sha256"]
        opponent_actual = configuration["opponent_checkpoint_sha256"]
        if learner_expected != learner_actual or opponent_expected != opponent_actual:
            checkpoint_mismatches.append({
                "pair_id": pair_id,
                "learner_expected": learner_expected,
                "learner_actual": learner_actual,
                "opponent_expected": opponent_expected,
                "opponent_actual": opponent_actual,
            })
        version_signatures.add(stable_json_sha256(report["versions"]))

        aggregate = _aggregate_games(games)
        score = float(aggregate["score_rate"])
        payoff = 2.0 * score - 1.0
        left = model_index[learner_id]
        right = model_index[opponent_id]
        if score_matrix[left][right] is not None:
            raise ValueError(f"duplicate matrix edge for {pair_id}")
        score_matrix[left][right] = score
        score_matrix[right][left] = 1.0 - score
        payoff_matrix[left][right] = payoff
        payoff_matrix[right][left] = -payoff
        score_by_edge[(learner_id, opponent_id)] = score
        score_by_edge[(opponent_id, learner_id)] = 1.0 - score

        interval = [float(value) for value in metrics["confidence_interval_95"]]
        close = interval[0] <= 0.5 <= interval[1]
        if close:
            close_pairs.append(pair_id)
        cross_rule = learner_tier != opponent_tier
        pair_records.append({
            "pair_id": pair_id,
            "group": summary_pair["group"],
            "learner_model": learner_id,
            "opponent_model": opponent_id,
            **aggregate,
            "reported_win_rate": metrics["win_rate"],
            "reported_confidence_interval_95": interval,
            "relative_elo": metrics["elo_relative"],
            "confidence_interval_includes_50_percent": close,
            "ordering_claim": (
                "no_forced_ordering" if close else "direction_supported_at_95_ci"
            ),
            "comparison_scope": (
                "cross_rule_historical_anchor_not_steps_ablation"
                if cross_rule
                else "same_rule_same_budget_seed_comparison"
            ),
            "class_cells": metrics["per_matchup"],
            "report": {
                "path": _relative(report_path),
                "bytes": report_path.stat().st_size,
                "sha256": _sha256_file(report_path),
            },
            "checkpoint_sha256": {
                "learner": learner_actual,
                "opponent": opponent_actual,
            },
        })
        source_reports.append(pair_records[-1]["report"])
        all_games.extend(games)
        for game in games:
            class_games_all[int(game["learner_class_id"])].append(game)
            if not cross_rule:
                class_games_internal[int(game["learner_class_id"])].append(game)
        if not cross_rule:
            candidate_games_by_model[learner_id].extend(games)
            candidate_games_by_model[opponent_id].extend(
                _reverse_game(game) for game in games
            )
            for game in games:
                reverse = _reverse_game(game)
                class_games_internal_both_sides[
                    int(game["learner_class_id"])
                ].append(game)
                class_games_internal_both_sides[
                    int(reverse["learner_class_id"])
                ].append(reverse)
                class_cells_internal_both_sides[(
                    int(game["learner_class_id"]),
                    int(game["opponent_class_id"]),
                )].append(game)
                class_cells_internal_both_sides[(
                    int(reverse["learner_class_id"]),
                    int(reverse["opponent_class_id"]),
                )].append(reverse)
        else:
            anchor_games_by_model[learner_id].extend(games)

    if checkpoint_mismatches:
        raise ValueError(f"checkpoint mismatches: {checkpoint_mismatches}")
    if len(version_signatures) != 1:
        raise ValueError("source reports do not share one experiment version")

    for row_index in range(size):
        for column_index in range(size):
            forward = payoff_matrix[row_index][column_index]
            reverse = payoff_matrix[column_index][row_index]
            if forward is None or reverse is None:
                if forward is not None or reverse is not None:
                    raise AssertionError("matrix missingness is not symmetric")
            elif abs(forward + reverse) > 1e-12:
                raise AssertionError("payoff matrix is not antisymmetric")

    point_cycles = find_cycles(score_by_edge, candidate_ids, threshold=0.5)
    significant_cycles = find_cycles(
        score_by_edge,
        candidate_ids,
        threshold=SIGNIFICANT_CYCLE_THRESHOLD,
    )
    pair_by_models = {
        frozenset((record["learner_model"], record["opponent_model"])): record
        for record in pair_records
    }
    for collection in (point_cycles, significant_cycles):
        for cycle in collection:
            for edge in cycle["edges"]:
                record = pair_by_models[frozenset((edge["winner"], edge["loser"]))]
                edge["pair_id"] = record["pair_id"]
                interval = record["reported_confidence_interval_95"]
                if record["learner_model"] != edge["winner"]:
                    interval = [1.0 - interval[1], 1.0 - interval[0]]
                edge["confidence_interval_95"] = interval

    missing_anchor_edges = []
    for left_index, left in enumerate(anchor_ids):
        for right in anchor_ids[left_index + 1 :]:
            missing_anchor_edges.append([left, right])

    report = {
        "schema_version": 1,
        "report_kind": "ppo_league_seed_payoff_matrix",
        "source_contract": {
            "baseline_manifest": {
                "path": _relative(BASELINE_MANIFEST),
                "sha256": _sha256_file(BASELINE_MANIFEST),
            },
            "evaluation_protocol": {
                "path": _relative(EVALUATION_PROTOCOL),
                "sha256": _sha256_file(EVALUATION_PROTOCOL),
            },
            "checkpoint_registry": {
                "path": _relative(checkpoint_registry_path),
                "sha256": _sha256_file(checkpoint_registry_path),
            },
            "launcher_status": {
                "path": _relative(source / "launcher_status.json"),
                "sha256": _sha256_file(source / "launcher_status.json"),
            },
            "source_summary": {
                "path": _relative(source / "summary.json"),
                "sha256": _sha256_file(source / "summary.json"),
            },
            "source_reports": source_reports,
            "shared_experiment_versions_sha256": next(iter(version_signatures)),
        },
        "audit": {
            "launcher_state": launcher["state"],
            "expected_pairs": 33,
            "observed_pairs": len(pair_records),
            "expected_games": 6468,
            "observed_games": len(all_games),
            "missing_pair_ids": missing_ids,
            "unexpected_pair_ids": unexpected_ids,
            "duplicate_pair_ids": duplicate_ids,
            "checkpoint_mismatches": checkpoint_mismatches,
            "terminated": sum(bool(game["terminated"]) for game in all_games),
            "truncated": sum(bool(game["truncated"]) for game in all_games),
            "illegal_actions": sum(int(game["illegal_actions"]) for game in all_games),
            "action_mask_mismatches": sum(
                int(game["action_mask_mismatches"]) for game in all_games
            ),
            "draws": sum(float(game["score"]) == 0.5 for game in all_games),
            "complete_for_preregistered_33_pair_topology": True,
        },
        "models": {
            "candidate_ids": candidate_ids,
            "historical_anchor_ids": anchor_ids,
            "all_ids": model_ids,
        },
        "pairwise_results": pair_records,
        "payoff_matrix": {
            "definition": "payoff=2*score_rate-1; rows are focal models",
            "model_ids": model_ids,
            "score_matrix": score_matrix,
            "antisymmetric_payoff_matrix": payoff_matrix,
            "observed_undirected_edges": 33,
            "unobserved_anchor_vs_anchor_edges": missing_anchor_edges,
            "candidate_submatrix_complete": True,
            "antisymmetry_tolerance": 1e-12,
        },
        "aggregates": {
            "all_games": _aggregate_games(all_games),
            "candidate_vs_candidate_by_model": {
                model_id: _aggregate_games(candidate_games_by_model[model_id])
                for model_id in candidate_ids
            },
            "anchor_vs_candidate_by_anchor": {
                model_id: _aggregate_games(anchor_games_by_model[model_id])
                for model_id in anchor_ids
            },
            "candidate_internal_by_learner_class": {
                str(class_id): _aggregate_games(class_games_internal[class_id])
                for class_id in sorted(class_games_internal)
            },
            "candidate_internal_by_class_both_model_sides": {
                str(class_id): _aggregate_games(
                    class_games_internal_both_sides[class_id]
                )
                for class_id in sorted(class_games_internal_both_sides)
            },
            "candidate_internal_class_cells_both_model_sides": {
                f"{focal}_vs_{opponent}": _aggregate_games(games)
                for (focal, opponent), games in sorted(
                    class_cells_internal_both_sides.items()
                )
            },
            "all_pairs_by_learner_class": {
                str(class_id): _aggregate_games(class_games_all[class_id])
                for class_id in sorted(class_games_all)
            },
        },
        "cycles": {
            "point_estimate_threshold_exclusive": 0.5,
            "point_estimate_cycles": point_cycles,
            "preregistered_threshold_exclusive": SIGNIFICANT_CYCLE_THRESHOLD,
            "preregistered_cycles": significant_cycles,
            "significant_cycle_detected": bool(significant_cycles),
        },
        "close_pair_policy": {
            "confidence_interval_includes_50_percent": close_pairs,
            "required_980_game_confirmations": [],
            "reason": (
                "Generation 0 retains all six candidates and all three anchors; "
                "no 196-game close result is used to include, exclude, or order a "
                "model. Any future pool-removal decision based on a close pair "
                "requires at least 980 games first."
            ),
        },
        "interpretation_limits": [
            "Pairs whose 95% CI includes 50% are not force-ranked.",
            "Old 3M versus new 1M is a cross-rule historical-anchor comparison, not a steps ablation.",
            "The three old-anchor-versus-old-anchor edges were outside the preregistered 33-pair topology.",
        ],
    }
    close_plan = {
        "schema_version": 1,
        "report_kind": "ppo_league_close_pair_confirmation_plan",
        "source_payoff_matrix_sha256": stable_json_sha256(report),
        "screened_close_pair_ids": close_pairs,
        "required_confirmation_pair_ids": [],
        "minimum_games_if_required": 980,
        "decision": "retain_all_no_confirmation_needed_for_generation_0",
        "future_removal_requires_confirmation": True,
    }
    return report, close_plan


def _percentage(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(report: Mapping[str, object]) -> bytes:
    audit = report["audit"]
    lines = [
        "# League Seed Payoff Matrix Audit",
        "",
        "## Audit result",
        "",
        f"- Planned topology: {audit['observed_pairs']}/{audit['expected_pairs']} pairs.",
        f"- Games: {audit['observed_games']}/{audit['expected_games']}.",
        f"- Terminated/truncated: {audit['terminated']}/{audit['truncated']}.",
        f"- Illegal actions / mask mismatches: {audit['illegal_actions']} / {audit['action_mask_mismatches']}.",
        f"- Draws: {audit['draws']}.",
        "",
        "The six 1M models form a complete same-rule candidate matrix. The three",
        "3M models are cross-rule historical anchors; their results against 1M",
        "models must not be interpreted as a pure training-step ablation.",
        "",
        "## Pair results",
        "",
        "| Pair | Score | 95% CI | Relative Elo | Side 0 / Side 1 | Avg turn | Avg steps | Interpretation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pair in report["pairwise_results"]:
        interval = pair["reported_confidence_interval_95"]
        side = pair["side_score_rates"]
        lines.append(
            "| {pair} | {score} | {lower}–{upper} | {elo:+.1f} | "
            "{side0} / {side1} | {turn:.2f} | {steps:.2f} | {claim} |".format(
                pair=pair["pair_id"],
                score=_percentage(pair["score_rate"]),
                lower=_percentage(interval[0]),
                upper=_percentage(interval[1]),
                elo=pair["relative_elo"],
                side0=_percentage(side["0"]),
                side1=_percentage(side["1"]),
                turn=pair["average_turn"],
                steps=pair["average_agent_steps"],
                claim=(
                    "CI overlaps 50%; no forced order"
                    if pair["confidence_interval_includes_50_percent"]
                    else "direction supported by 95% CI"
                ),
            )
        )

    candidates = report["models"]["candidate_ids"]
    all_ids = report["payoff_matrix"]["model_ids"]
    score_matrix = report["payoff_matrix"]["score_matrix"]
    model_index = {model_id: index for index, model_id in enumerate(all_ids)}
    lines.extend([
        "",
        "## Six-candidate score matrix",
        "",
        "Rows are focal models; diagonal is 50%.",
        "",
        "| Model | " + " | ".join(candidates) + " |",
        "| --- | " + " | ".join("---:" for _ in candidates) + " |",
    ])
    for row_model in candidates:
        row = score_matrix[model_index[row_model]]
        lines.append(
            "| "
            + row_model
            + " | "
            + " | ".join(
                _percentage(row[model_index[column_model]])
                for column_model in candidates
            )
            + " |"
        )

    lines.extend([
        "",
        "## Candidate class aggregate",
        "",
        "Each game is counted once from each model side, so class strength is not",
        "tied to which checkpoint happened to be the report learner.",
        "",
        "| Class ID | Games | Score | 95% CI |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for class_id, aggregate in report["aggregates"][
        "candidate_internal_by_class_both_model_sides"
    ].items():
        interval = aggregate["confidence_interval_95"]
        lines.append(
            f"| {class_id} | {aggregate['games']} | "
            f"{_percentage(aggregate['score_rate'])} | "
            f"{_percentage(interval[0])}–{_percentage(interval[1])} |"
        )

    lines.extend([
        "",
        "## Cycle and close-pair decision",
        "",
        f"- Point-estimate cycles (>50% each edge): {len(report['cycles']['point_estimate_cycles'])}.",
        f"- Preregistered strong cycles (>55% each edge): {len(report['cycles']['preregistered_cycles'])}.",
        f"- Close pairs whose 95% CI includes 50%: {len(report['close_pair_policy']['confidence_interval_includes_50_percent'])}.",
        "- Generation 0 retains every candidate and anchor, so none of the close",
        "  196-game screens is used to alter pool membership. A future removal",
        "  decision requires a 980-game confirmation first.",
        "",
    ])
    return ("\n".join(lines)).encode("utf-8")


def build_outputs() -> dict[str, bytes]:
    report, close_plan = build_seed_payoff_matrix()
    return {
        "seed_payoff_matrix.json": render_json(report),
        "seed_payoff_matrix.md": render_markdown(report),
        "close_pair_confirmations/plan.json": render_json(close_plan),
    }


def write_outputs(outputs: Mapping[str, bytes], output_directory: str | Path) -> None:
    output = _repo_path(output_directory)
    for name, payload in outputs.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def check_outputs(outputs: Mapping[str, bytes], output_directory: str | Path) -> list[str]:
    output = _repo_path(output_directory)
    return [
        name
        for name, payload in outputs.items()
        if not (output / name).is_file() or (output / name).read_bytes() != payload
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the completed League seed matrix.")
    parser.add_argument("--output-directory", default=str(OUTPUT_DIRECTORY))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_outputs()
    if args.check:
        mismatches = check_outputs(outputs, args.output_directory)
        if mismatches:
            print("seed payoff matrix mismatch: " + ", ".join(mismatches))
            return 1
        print("seed payoff matrix is byte-stable and current")
        return 0
    write_outputs(outputs, args.output_directory)
    print(f"wrote League seed payoff matrix to {_repo_path(args.output_directory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
