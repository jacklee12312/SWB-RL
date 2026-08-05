from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from scripts.report_ppo_league_baseline import ROOT, render_json
from scripts.report_ppo_league_generation import _selection_audit
from swb.rl.pfsp import PayoffEstimate, compute_pfsp_distribution
from swb.rl.versioning import stable_json_sha256


REPORT_ROOT = Path("data/reports/league_training")
SAMPLER_ROOT = REPORT_ROOT / "sampler_screen_20260804"
GENERATION_ZERO_MANIFEST = REPORT_ROOT / "generation_000_manifest.json"
EVALUATION_PROTOCOL = REPORT_ROOT / "evaluation_protocol.json"
SAMPLER_RESULT = SAMPLER_ROOT / "sampler_screen_result.json"
CONTRACT_OUTPUT = REPORT_ROOT / "evolving_league_contract.json"
SCHEDULE_OUTPUT = REPORT_ROOT / "generation_schedule.json"
ACTIVE_MATRIX_OUTPUT = (
    REPORT_ROOT / "generation_000_active_payoff_matrix.json"
)
LINEAGE_MANIFEST_DIRECTORY = (
    REPORT_ROOT / "generation_000_lineage_manifests"
)
ARCHIVE_SELECTION_OUTPUT = REPORT_ROOT / "archive_selection_report.json"
GENERATION_QUEUE_SCHEMA_OUTPUT = REPORT_ROOT / "generation_queue_schema.json"

ACTIVE_FRACTION = 0.70
ARCHIVE_FRACTION = 0.30
GENERATION_AGENT_STEPS = 250_000
CHECKPOINT_INTERVAL_AGENT_STEPS = 100_000
ARCHIVE_AUDIT_INTERVAL_AGENT_STEPS = 500_000
MAX_TARGET_GENERATION = 8
ACTIVE_PAIR_GAMES = 196
GENERATION_ZERO_PAYOFF_MASTER_SEED = 20261001
ACTIVE_SEEDS = tuple(range(20260903, 20260909))


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


def _json(path: str | Path) -> dict[str, object]:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _entry_map(
    manifest: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    return {
        str(entry["opponent_id"]): entry
        for entry in manifest["entries"]  # type: ignore[index]
    }


def _active_ids() -> tuple[str, ...]:
    return tuple(f"seed_{seed}_1m" for seed in ACTIVE_SEEDS)


def _active_pair_report(left_id: str, right_id: str) -> Path:
    filename = f"{left_id}__vs__{right_id}.json"
    if left_id == "seed_20260903_1m":
        return (
            REPORT_ROOT / "generation_000_payoff_evaluations" / filename
        )
    return SAMPLER_ROOT / "generation_000_active_matrix" / filename


def _archive_report(active_id: str, archive_id: str) -> Path:
    filename = f"{active_id}__vs__{archive_id}.json"
    if active_id == "seed_20260903_1m":
        return (
            REPORT_ROOT / "generation_000_payoff_evaluations" / filename
        )
    return SAMPLER_ROOT / "archive_baseline" / filename


def _validate_evaluation(
    path: str | Path,
    *,
    learner_entry: Mapping[str, object],
    opponent_entry: Mapping[str, object],
    master_seed: int,
    games: int,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = _json(path)
    configuration = payload.get("configuration")
    metrics = payload.get("metrics")
    rows = payload.get("games")
    if not isinstance(configuration, dict) or not isinstance(metrics, dict):
        raise ValueError(f"evaluation report lacks configuration/metrics: {path}")
    if not isinstance(rows, list) or len(rows) != games:
        raise ValueError(f"evaluation report has wrong game count: {path}")
    if int(configuration.get("master_seed", -1)) != master_seed:
        raise ValueError(f"evaluation report uses wrong tuning seed: {path}")
    if int(configuration.get("seed_count", -1)) * 98 != games:
        raise ValueError(f"evaluation report has wrong seed_count: {path}")
    if configuration.get("full_matchup_matrix") is not True:
        raise ValueError(f"evaluation report lacks full 7x7 matrix: {path}")
    if configuration.get("match_setup") != "official":
        raise ValueError(f"evaluation report uses wrong match setup: {path}")
    if payload.get("checkpoint", {}).get("sha256") != learner_entry.get(
        "checkpoint_sha256"
    ):
        raise ValueError(f"evaluation learner checkpoint mismatch: {path}")
    if configuration.get("opponent_checkpoint_sha256") != opponent_entry.get(
        "checkpoint_sha256"
    ):
        raise ValueError(f"evaluation opponent checkpoint mismatch: {path}")
    for name in ("truncated", "illegal_actions", "action_mask_mismatches"):
        if int(metrics.get(name, -1)) != 0:
            raise ValueError(f"evaluation safety failure {name}: {path}")
    if int(metrics.get("terminated", -1)) != games:
        raise ValueError(f"evaluation did not terminate every game: {path}")
    scores = [float(row["score"]) for row in rows]
    if any(score not in (0.0, 0.5, 1.0) for score in scores):
        raise ValueError(f"evaluation contains unsupported score: {path}")
    score_rate = sum(scores) / games
    if not math.isclose(
        score_rate,
        float(metrics["win_rate"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"evaluation score-rate mismatch: {path}")
    interval = [float(value) for value in metrics["confidence_interval_95"]]
    return {
        "games": games,
        "wins": sum(score == 1.0 for score in scores),
        "draws": sum(score == 0.5 for score in scores),
        "losses": sum(score == 0.0 for score in scores),
        "score_rate": score_rate,
        "confidence_interval_95": interval,
        "side_win_rates": metrics["side_win_rates"],
        "average_turn": float(metrics["average_turn"]),
        "average_agent_steps": float(metrics["average_agent_steps"]),
    }, _source(path)


def build_evolving_contract() -> dict[str, object]:
    generation = _json(GENERATION_ZERO_MANIFEST)
    protocol = _json(EVALUATION_PROTOCOL)
    sampler = _json(SAMPLER_RESULT)
    entries = _entry_map(generation)
    lineages = []
    for seed, active_id in zip(ACTIVE_SEEDS, _active_ids()):
        entry = entries[active_id]
        lineages.append({
            "lineage_id": f"lineage_{seed}",
            "policy_seed": seed,
            "generation_000_parent": {
                "opponent_id": active_id,
                "checkpoint_path": entry["checkpoint_path"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "agent_steps": int(entry["training_steps"]),
            },
        })
    return {
        "schema_version": 1,
        "report_kind": "ppo_evolving_league_contract",
        "immutable": True,
        "sources": {
            "generation_000_manifest": _source(GENERATION_ZERO_MANIFEST),
            "evaluation_protocol": _source(EVALUATION_PROTOCOL),
            "sampler_screen_result": _source(SAMPLER_RESULT),
        },
        "selected_sampler": sampler["decision"]["selected_sampler"],
        "lineages": lineages,
        "roles": {
            "active_latest": (
                "six frozen checkpoints from the immediately preceding generation"
            ),
            "historical_archive": (
                "read-only historical checkpoints retained for diversity and forgetting audits"
            ),
            "evaluation_anchor": (
                "zero-training-weight checkpoints used only for validation"
            ),
        },
        "generation_barrier": {
            "source_manifest_immutable": True,
            "all_lineages_read_same_source_generation": True,
            "same_generation_visibility": False,
            "publish_requires_all_six_lineages": True,
            "failed_lineage_policy": "resume_original_checkpoint_until_complete",
            "atomic_publish": True,
        },
        "cadence": {
            "generation_agent_steps": GENERATION_AGENT_STEPS,
            "checkpoint_interval_agent_steps": (
                CHECKPOINT_INTERVAL_AGENT_STEPS
            ),
            "archive_audit_interval_agent_steps": (
                ARCHIVE_AUDIT_INTERVAL_AGENT_STEPS
            ),
            "payoff_refresh": "generation_boundary_only",
        },
        "opponent_distribution": {
            "active_latest_fraction": ACTIVE_FRACTION,
            "historical_archive_fraction": ARCHIVE_FRACTION,
            "active_sampler": "hard",
            "active_epsilon_floor": 0.02,
            "active_maximum_probability": 0.35,
            "active_hard_alpha": 1.0,
            "archive_sampler": "deterministic_uniform",
            "maximum_positive_opponents": 32,
        },
        "frozen_training_semantics": {
            "sparse_terminal_reward": True,
            "observation": "v4.1",
            "policy_architecture": "entity_action_v1",
            "class_schedule": "full_7x7_both_positions",
            "rules_database_action_contract": generation["contract"],
        },
        "seed_isolation": protocol["seed_partitions"],
        "gates": {
            "safety": (
                "zero illegal actions, mask mismatches, abnormal truncations, "
                "generation leakage, NaN/Inf, and residual workers"
            ),
            "generation_001": (
                "at least four of six lineage validation scores >= 0.48 and "
                "the minimum lineage validation score >= 0.48"
            ),
            "plateau": (
                "stop after two consecutive generation gains below 0.03 in "
                "mean fixed-parent validation score unless the registered "
                "population proxy improves by at least 10 percent"
            ),
        },
    }


def build_generation_schedule() -> dict[str, object]:
    contract = build_evolving_contract()
    tuning_seeds = contract["seed_isolation"][
        "pfsp_tuning_match_master_seeds"
    ]
    transitions = []
    for target_generation in range(1, MAX_TARGET_GENERATION + 1):
        source_generation = target_generation - 1
        lineages = []
        for row in contract["lineages"]:
            initial = int(row["generation_000_parent"]["agent_steps"])
            lineages.append({
                "lineage_id": row["lineage_id"],
                "policy_seed": row["policy_seed"],
                "target_agent_steps": (
                    initial + target_generation * GENERATION_AGENT_STEPS
                ),
                "additional_agent_steps_from_parent": GENERATION_AGENT_STEPS,
            })
        transitions.append({
            "source_generation": source_generation,
            "target_generation": target_generation,
            "nominal_millions": 1.0 + 0.25 * target_generation,
            "payoff_master_seed": int(tuning_seeds[target_generation + 1]),
            "lineages": lineages,
            "archive_audit_due": target_generation % 2 == 0,
            "paths": {
                "directory": (
                    "data/reports/league_training/generations/"
                    f"generation_{target_generation:03d}"
                ),
                "population_manifest": (
                    "data/reports/league_training/generations/"
                    f"generation_{target_generation:03d}/population_manifest.json"
                ),
                "active_payoff": (
                    "data/reports/league_training/generations/"
                    f"generation_{target_generation:03d}/active_payoff.json"
                ),
                "training_report": (
                    "data/reports/league_training/generations/"
                    f"generation_{target_generation:03d}/training_report.json"
                ),
            },
        })
    return {
        "schema_version": 1,
        "report_kind": "ppo_evolving_league_generation_schedule",
        "immutable": True,
        "sources": {
            "evolving_league_contract_payload_sha256": stable_json_sha256(
                contract
            ),
            "evaluation_protocol": _source(EVALUATION_PROTOCOL),
        },
        "generation_zero": {
            "nominal_millions": 1.0,
            "lineages": contract["lineages"],
        },
        "transitions": transitions,
        "budget": {
            "lineages": 6,
            "transitions": MAX_TARGET_GENERATION,
            "additional_agent_steps_per_transition": (
                6 * GENERATION_AGENT_STEPS
            ),
            "total_additional_agent_steps_to_generation_008": (
                6 * GENERATION_AGENT_STEPS * MAX_TARGET_GENERATION
            ),
            "active_payoff_unique_pairs_per_generation": 15,
            "active_payoff_games_per_generation": 15 * ACTIVE_PAIR_GAMES,
        },
        "stop_is_allowed_before_generation_008": True,
    }


def build_generation_queue_schema() -> dict[str, object]:
    return {
        "schema_version": 1,
        "report_kind": "ppo_evolving_league_generation_queue_schema",
        "states": [
            "running",
            "completed",
            "stopped_by_gate",
            "failed",
        ],
        "job_lifecycle": {
            "pending": "listed in pending_training_jobs",
            "running": "identified by active_stage and active_job",
            "completed": "listed in completed_training_jobs",
            "failed": "queue state is failed and error/traceback are populated",
        },
        "required_fields": [
            "runner_pid",
            "mode",
            "state",
            "active_stage",
            "active_job",
            "pending_training_jobs",
            "completed_training_jobs",
            "completed_generations",
            "stop_reason",
            "error",
            "updated_at_epoch_seconds",
        ],
        "resume_contract": {
            "completed_metrics_reuse": (
                "requires completed steps, safety diagnostics, lineage manifest "
                "hash, checkpoint existence, and checkpoint step validation"
            ),
            "partial_checkpoint": (
                "resume the same generation checkpoint without replacing its "
                "already-frozen opponent pool"
            ),
            "generation_boundary": (
                "replace the opponent pool only when resuming the immutable "
                "parent checkpoint into a new generation"
            ),
        },
    }


def build_generation_zero_active_matrix() -> dict[str, object]:
    generation = _json(GENERATION_ZERO_MANIFEST)
    entries = _entry_map(generation)
    active_ids = _active_ids()
    matrix = {
        left: {right: (0.5 if left == right else None) for right in active_ids}
        for left in active_ids
    }
    intervals = {
        left: {
            right: ([0.5, 0.5] if left == right else None)
            for right in active_ids
        }
        for left in active_ids
    }
    pairs = []
    sources = []
    for left_index, left in enumerate(active_ids):
        for right in active_ids[left_index + 1 :]:
            path = _active_pair_report(left, right)
            result, source = _validate_evaluation(
                path,
                learner_entry=entries[left],
                opponent_entry=entries[right],
                master_seed=GENERATION_ZERO_PAYOFF_MASTER_SEED,
                games=ACTIVE_PAIR_GAMES,
            )
            score = float(result["score_rate"])
            interval = result["confidence_interval_95"]
            matrix[left][right] = score
            matrix[right][left] = 1.0 - score
            intervals[left][right] = interval
            intervals[right][left] = [1.0 - interval[1], 1.0 - interval[0]]
            pairs.append({
                "left": left,
                "right": right,
                **result,
                "report": source,
            })
            sources.append(source)
    for left in active_ids:
        for right in active_ids:
            if matrix[left][right] is None:
                raise ValueError(f"active payoff matrix is incomplete: {left}/{right}")
            if not math.isclose(
                float(matrix[left][right]) + float(matrix[right][left]),
                1.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"active payoff matrix is not antisymmetric: {left}/{right}")
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_active_payoff_matrix",
        "immutable": True,
        "data_partition": "pfsp_tuning",
        "generation": 0,
        "source_generation_manifest": _source(GENERATION_ZERO_MANIFEST),
        "master_seed": GENERATION_ZERO_PAYOFF_MASTER_SEED,
        "active_policy_ids": list(active_ids),
        "games_per_unique_pair": ACTIVE_PAIR_GAMES,
        "unique_pair_count": len(pairs),
        "total_games": len(pairs) * ACTIVE_PAIR_GAMES,
        "score_matrix": matrix,
        "confidence_interval_95_matrix": intervals,
        "pairs": pairs,
        "audit": {
            "complete": True,
            "antisymmetric": True,
            "illegal_actions": 0,
            "action_mask_mismatches": 0,
            "truncated": 0,
            "report_sources": sources,
        },
    }


def build_archive_selection_report() -> dict[str, object]:
    generation = _json(GENERATION_ZERO_MANIFEST)
    entries = _entry_map(generation)
    active_ids = _active_ids()
    archive_ids = tuple(sorted(
        opponent_id
        for opponent_id, entry in entries.items()
        if entry["role"] == "self_history"
    ))
    if len(archive_ids) != 18:
        raise ValueError("Generation 0 archive requires exactly 18 histories")
    baselines = []
    sources = []
    for active_id in active_ids:
        for archive_id in archive_ids:
            path = _archive_report(active_id, archive_id)
            result, source = _validate_evaluation(
                path,
                learner_entry=entries[active_id],
                opponent_entry=entries[archive_id],
                master_seed=GENERATION_ZERO_PAYOFF_MASTER_SEED,
                games=ACTIVE_PAIR_GAMES,
            )
            baselines.append({
                "active_policy_id": active_id,
                "archive_policy_id": archive_id,
                **result,
                "report": source,
            })
            sources.append(source)
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_archive_selection_report",
        "immutable": True,
        "generation": 0,
        "selection_policy": (
            "all 18 unique Generation 0 histories fit below the 26-entry "
            "archive cap and are retained in deterministic opponent_id order"
        ),
        "maximum_positive_archive_entries": 26,
        "selected_archive_ids": list(archive_ids),
        "zero_weight_archive_ids": [],
        "conditional_sampling_probability": {
            archive_id: 1.0 / len(archive_ids) for archive_id in archive_ids
        },
        "active_archive_baselines": baselines,
        "audit": {
            "active_policy_count": len(active_ids),
            "archive_policy_count": len(archive_ids),
            "pair_count": len(baselines),
            "total_games": len(baselines) * ACTIVE_PAIR_GAMES,
            "illegal_actions": 0,
            "action_mask_mismatches": 0,
            "truncated": 0,
            "report_sources": sources,
        },
    }


def build_generation_zero_lineage_manifests() -> dict[str, dict[str, object]]:
    generation = _json(GENERATION_ZERO_MANIFEST)
    active_matrix = build_generation_zero_active_matrix()
    archive_report = build_archive_selection_report()
    entries = _entry_map(generation)
    active_ids = _active_ids()
    archive_ids = tuple(archive_report["selected_archive_ids"])
    anchor_ids = tuple(sorted(
        opponent_id
        for opponent_id, entry in entries.items()
        if entry["role"] == "anchor_only"
    ))
    manifests = {}
    for seed, learner_id in zip(ACTIVE_SEEDS, active_ids):
        estimates = tuple(
            PayoffEstimate(
                opponent_id=opponent_id,
                checkpoint_sha256=str(entries[opponent_id]["checkpoint_sha256"]),
                games=ACTIVE_PAIR_GAMES,
                score_rate=float(active_matrix["score_matrix"][learner_id][opponent_id]),
                confidence_interval_95=tuple(
                    float(value)
                    for value in active_matrix[
                        "confidence_interval_95_matrix"
                    ][learner_id][opponent_id]
                ),
            )
            for opponent_id in active_ids
        )
        distribution = compute_pfsp_distribution(
            estimates,
            sampler="hard",
            epsilon_floor=0.02,
            maximum_probability=0.35,
            hard_alpha=1.0,
        )
        output_entries = []
        for opponent_id in active_ids:
            entry = dict(entries[opponent_id])
            entry["role"] = "active_latest"
            entry["training_eligible"] = True
            entry["sampling_weight"] = (
                ACTIVE_FRACTION * distribution.probabilities[opponent_id]
            )
            output_entries.append(entry)
        for opponent_id in archive_ids:
            entry = dict(entries[opponent_id])
            entry["role"] = "historical_archive"
            entry["training_eligible"] = True
            entry["sampling_weight"] = ARCHIVE_FRACTION / len(archive_ids)
            output_entries.append(entry)
        for opponent_id in anchor_ids:
            entry = dict(entries[opponent_id])
            entry["role"] = "evaluation_anchor"
            entry["training_eligible"] = False
            entry["sampling_weight"] = 0.0
            output_entries.append(entry)
        output_entries.sort(key=lambda row: str(row["opponent_id"]))
        active_total = sum(
            float(row["sampling_weight"])
            for row in output_entries
            if row["role"] == "active_latest"
        )
        archive_total = sum(
            float(row["sampling_weight"])
            for row in output_entries
            if row["role"] == "historical_archive"
        )
        if not math.isclose(active_total, ACTIVE_FRACTION, abs_tol=1e-12):
            raise ValueError(f"active mass mismatch for {learner_id}")
        if not math.isclose(archive_total, ARCHIVE_FRACTION, abs_tol=1e-12):
            raise ValueError(f"archive mass mismatch for {learner_id}")
        manifests[learner_id] = {
            "schema_version": 1,
            "report_kind": "ppo_league_generation_manifest",
            "immutable": True,
            "path_base": "repository_root",
            "generation": 0,
            "selection_mode": "hard",
            "learner": {
                "lineage_id": f"lineage_{seed}",
                "policy_seed": seed,
                "parent_opponent_id": learner_id,
                "parent_checkpoint_sha256": entries[learner_id][
                    "checkpoint_sha256"
                ],
            },
            "contract": generation["contract"],
            "sources": {
                "generation_000_manifest": _source(GENERATION_ZERO_MANIFEST),
                "active_payoff_matrix": {
                    "path": _relative(ACTIVE_MATRIX_OUTPUT),
                    "payload_sha256": stable_json_sha256(active_matrix),
                },
                "archive_selection_report": {
                    "path": _relative(ARCHIVE_SELECTION_OUTPUT),
                    "payload_sha256": stable_json_sha256(archive_report),
                },
                "evolving_league_contract": {
                    "path": _relative(CONTRACT_OUTPUT),
                    "payload_sha256": stable_json_sha256(
                        build_evolving_contract()
                    ),
                },
            },
            "two_stage_sampling": {
                "active_latest_mass": active_total,
                "historical_archive_mass": archive_total,
                "active_latest_distribution": distribution.report(),
                "historical_archive_distribution": "uniform",
            },
            "entries": output_entries,
            "selection_audit": _selection_audit(output_entries),
            "summary": {
                "entry_count": len(output_entries),
                "positive_entry_count": sum(
                    float(row["sampling_weight"]) > 0 for row in output_entries
                ),
                "active_latest_count": len(active_ids),
                "historical_archive_count": len(archive_ids),
                "evaluation_anchor_count": len(anchor_ids),
                "sampling_weight_total": sum(
                    float(row["sampling_weight"]) for row in output_entries
                ),
            },
        }
    return manifests


def _artifacts() -> dict[Path, bytes]:
    active_matrix = build_generation_zero_active_matrix()
    archive_report = build_archive_selection_report()
    artifacts = {
        CONTRACT_OUTPUT: render_json(build_evolving_contract()),
        SCHEDULE_OUTPUT: render_json(build_generation_schedule()),
        ACTIVE_MATRIX_OUTPUT: render_json(active_matrix),
        ARCHIVE_SELECTION_OUTPUT: render_json(archive_report),
        GENERATION_QUEUE_SCHEMA_OUTPUT: render_json(
            build_generation_queue_schema()
        ),
    }
    for learner_id, manifest in build_generation_zero_lineage_manifests().items():
        artifacts[
            LINEAGE_MANIFEST_DIRECTORY / f"{learner_id}.json"
        ] = render_json(manifest)
    return artifacts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the six-lineage evolving League contract, "
            "Generation 0 active payoff, archive report, and learner manifests."
        )
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mismatches = []
    for path, expected in _artifacts().items():
        output = _repo_path(path)
        if args.check:
            if not output.is_file() or output.read_bytes() != expected:
                mismatches.append(str(output))
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(expected)
        print(f"wrote {output}")
    if mismatches:
        print("evolving League artifacts mismatch: " + ", ".join(mismatches))
        return 1
    if args.check:
        print("evolving League artifacts are byte-stable and current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
