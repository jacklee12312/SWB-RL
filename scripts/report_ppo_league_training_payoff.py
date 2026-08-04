from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from scripts.report_ppo_league_baseline import ROOT, render_json
from scripts.report_ppo_league_seed_matrix import wilson_interval
from swb.rl.versioning import stable_json_sha256


REPORT_DIRECTORY = Path("data/reports/league_training")
GENERATION_MANIFEST = REPORT_DIRECTORY / "generation_000_manifest.json"
EVALUATION_PROTOCOL = REPORT_DIRECTORY / "evaluation_protocol.json"
SOURCE_DIRECTORY = REPORT_DIRECTORY / "generation_000_payoff_evaluations"
DEFAULT_OUTPUT = REPORT_DIRECTORY / "generation_000_training_payoff_snapshot.json"
DEFAULT_PLAN_OUTPUT = REPORT_DIRECTORY / "generation_000_payoff_evaluation_plan.json"
FOCAL_POLICY_IDS = ("seed_20260903_1m",)
MATCH_MASTER_SEEDS = (20261001,)
SEED_COUNT = 2
GAMES_PER_PAIR = 196


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
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _report_path(
    directory: str | Path,
    focal_policy_id: str,
    opponent_id: str,
) -> Path:
    return _repo_path(directory) / f"{focal_policy_id}__vs__{opponent_id}.json"


def _validate_pair_report(
    report: Mapping[str, object],
    *,
    focal: Mapping[str, object],
    opponent: Mapping[str, object],
    match_master_seed: int,
    seed_count: int,
    games_per_pair: int,
    expected_versions: Mapping[str, object],
) -> dict[str, object]:
    configuration = report.get("configuration")
    metrics = report.get("metrics")
    checkpoint = report.get("checkpoint")
    games = report.get("games")
    if not all(isinstance(value, Mapping) for value in (
        configuration,
        metrics,
        checkpoint,
    )) or not isinstance(games, list):
        raise ValueError("training payoff pair report is missing required fields")
    assert isinstance(configuration, Mapping)
    assert isinstance(metrics, Mapping)
    assert isinstance(checkpoint, Mapping)
    if checkpoint.get("sha256") != focal["checkpoint_sha256"]:
        raise ValueError("training payoff focal checkpoint hash mismatch")
    if (
        configuration.get("opponent_checkpoint_sha256")
        != opponent["checkpoint_sha256"]
    ):
        raise ValueError("training payoff opponent checkpoint hash mismatch")
    expected_configuration = {
        "master_seed": match_master_seed,
        "seed_count": seed_count,
        "max_agent_steps": 512,
        "full_matchup_matrix": True,
        "match_setup": "official",
        "opponent_kind": "historical",
    }
    mismatches = {
        key: {"expected": value, "actual": configuration.get(key)}
        for key, value in expected_configuration.items()
        if configuration.get(key) != value
    }
    if list(configuration.get("class_ids", [])) != list(range(1, 8)):
        mismatches["class_ids"] = {
            "expected": list(range(1, 8)),
            "actual": configuration.get("class_ids"),
        }
    if mismatches:
        raise ValueError(f"training payoff evaluation config mismatch: {mismatches}")
    if report.get("versions") != expected_versions:
        raise ValueError("training payoff experiment versions mismatch")
    if int(metrics.get("games", -1)) != games_per_pair or len(games) != games_per_pair:
        raise ValueError("training payoff report has wrong game count")
    safety = {
        "terminated": int(metrics.get("terminated", -1)),
        "truncated": int(metrics.get("truncated", -1)),
        "illegal_actions": int(metrics.get("illegal_actions", -1)),
        "action_mask_mismatches": int(
            metrics.get("action_mask_mismatches", -1)
        ),
    }
    if safety != {
        "terminated": games_per_pair,
        "truncated": 0,
        "illegal_actions": 0,
        "action_mask_mismatches": 0,
    }:
        raise ValueError(f"training payoff report failed safety: {safety}")
    scores = [float(game["score"]) for game in games]
    if any(score not in (0.0, 0.5, 1.0) for score in scores):
        raise ValueError("training payoff report has invalid game score")
    return {
        "games": len(scores),
        "score_total": sum(scores),
        "score_rate": sum(scores) / len(scores),
        "confidence_interval_95": list(wilson_interval(sum(scores), len(scores))),
        **safety,
    }


def build_training_payoff_plan(
    *,
    generation_manifest_path: str | Path = GENERATION_MANIFEST,
    evaluation_protocol_path: str | Path = EVALUATION_PROTOCOL,
    focal_policy_ids: Sequence[str] = FOCAL_POLICY_IDS,
    match_master_seeds: Sequence[int] = MATCH_MASTER_SEEDS,
    seed_count: int = SEED_COUNT,
    games_per_pair: int = GAMES_PER_PAIR,
) -> dict[str, object]:
    generation = _load_json(generation_manifest_path)
    protocol = _load_json(evaluation_protocol_path)
    partitions = protocol["seed_partitions"]
    tuning_seeds = set(partitions["pfsp_tuning_match_master_seeds"])
    final_seeds = set(partitions["final_evaluation_match_master_seeds"])
    if (
        not match_master_seeds
        or set(match_master_seeds) - tuning_seeds
        or set(match_master_seeds) & final_seeds
    ):
        raise ValueError("training payoff plan violates the evaluator seed partition")
    entries = {
        str(entry["opponent_id"]): entry
        for entry in generation["entries"]
    }
    trainable = [
        entry
        for entry in generation["entries"]
        if bool(entry["training_eligible"])
    ]
    missing_focals = sorted(set(focal_policy_ids) - set(entries))
    if missing_focals:
        raise ValueError(f"training payoff focal policies are missing: {missing_focals}")
    pair_count = len(focal_policy_ids) * len(trainable)
    manifest_path = _repo_path(generation_manifest_path)
    protocol_path = _repo_path(evaluation_protocol_path)
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_training_payoff_evaluation_plan",
        "status": "preregistered_not_started",
        "source_generation": int(generation["generation"]),
        "target_generation": int(generation["generation"]) + 1,
        "data_partition": "pfsp_tuning",
        "sources": {
            "generation_manifest": {
                "path": _relative(manifest_path),
                "sha256": _sha256_file(manifest_path),
                "payload_sha256": stable_json_sha256(generation),
            },
            "evaluation_protocol": {
                "path": _relative(protocol_path),
                "sha256": _sha256_file(protocol_path),
                "payload_sha256": stable_json_sha256(protocol),
            },
        },
        "focal_policies": [
            {
                "opponent_id": focal_policy_id,
                "checkpoint_path": entries[focal_policy_id]["checkpoint_path"],
                "checkpoint_sha256": entries[focal_policy_id][
                    "checkpoint_sha256"
                ],
                "selection_reason": (
                    "highest six-candidate internal aggregate score and "
                    "non-zero Generation 0 Nash support"
                    if focal_policy_id == "seed_20260903_1m"
                    else "explicitly preregistered focal policy"
                ),
            }
            for focal_policy_id in focal_policy_ids
        ],
        "opponents": [
            {
                "opponent_id": entry["opponent_id"],
                "checkpoint_path": entry["checkpoint_path"],
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "model_generation": entry["generation"],
                "role": entry.get("role"),
            }
            for entry in sorted(trainable, key=lambda row: row["opponent_id"])
        ],
        "evaluation": {
            "match_master_seeds": list(match_master_seeds),
            "registered_tuning_match_master_seeds": list(
                partitions["pfsp_tuning_match_master_seeds"]
            ),
            "final_evaluation_match_master_seeds_used": [],
            "seed_count": seed_count,
            "games_per_pair": games_per_pair,
            "class_ids": list(range(1, 8)),
            "full_matchup_matrix": True,
            "both_player_positions": True,
            "max_agent_steps": 512,
            "match_setup": "official",
            "device": "cuda",
        },
        "expected": {
            "pair_count": pair_count,
            "game_count": pair_count * games_per_pair,
            "trainable_opponent_count": len(trainable),
            "focal_policy_count": len(focal_policy_ids),
            "illegal_actions": 0,
            "action_mask_mismatches": 0,
            "truncated": 0,
        },
        "outputs": {
            "raw_report_directory": _relative(SOURCE_DIRECTORY),
            "training_payoff_snapshot": _relative(DEFAULT_OUTPUT),
        },
    }


def build_training_payoff_snapshot(
    *,
    generation_manifest_path: str | Path = GENERATION_MANIFEST,
    evaluation_protocol_path: str | Path = EVALUATION_PROTOCOL,
    source_directory: str | Path = SOURCE_DIRECTORY,
    focal_policy_ids: Sequence[str] = FOCAL_POLICY_IDS,
    match_master_seeds: Sequence[int] = MATCH_MASTER_SEEDS,
    seed_count: int = SEED_COUNT,
    games_per_pair: int = GAMES_PER_PAIR,
) -> dict[str, object]:
    generation = _load_json(generation_manifest_path)
    protocol = _load_json(evaluation_protocol_path)
    partitions = protocol["seed_partitions"]
    tuning_seeds = set(partitions["pfsp_tuning_match_master_seeds"])
    final_seeds = set(partitions["final_evaluation_match_master_seeds"])
    used_seeds = set(match_master_seeds)
    if not match_master_seeds or used_seeds - tuning_seeds or used_seeds & final_seeds:
        raise ValueError("training payoff evaluator seed partition violation")
    if len(used_seeds) != len(match_master_seeds):
        raise ValueError("training payoff match master seeds must be unique")
    if len(match_master_seeds) != 1:
        raise ValueError("first PFSP payoff snapshot uses one registered master seed")
    if seed_count != 2 or games_per_pair != 196:
        raise ValueError("first PFSP payoff snapshot is preregistered at 196 games")

    entries = {
        str(entry["opponent_id"]): entry
        for entry in generation["entries"]
    }
    trainable = {
        opponent_id: entry
        for opponent_id, entry in entries.items()
        if bool(entry["training_eligible"])
    }
    if len(set(focal_policy_ids)) != len(focal_policy_ids):
        raise ValueError("training payoff focal policy IDs must be unique")
    missing_focals = sorted(set(focal_policy_ids) - set(trainable))
    if missing_focals:
        raise ValueError(f"training payoff focal policies are missing: {missing_focals}")

    expected_versions = generation["contract"]["experiment_versions"]
    source_reports = []
    opponents = []
    total_games = 0
    for opponent_id, opponent in sorted(trainable.items()):
        focal_results = []
        for focal_policy_id in focal_policy_ids:
            path = _report_path(
                source_directory,
                focal_policy_id,
                opponent_id,
            )
            if not path.is_file():
                raise ValueError(f"missing training payoff report: {path}")
            report = _load_json(path)
            result = _validate_pair_report(
                report,
                focal=entries[focal_policy_id],
                opponent=opponent,
                match_master_seed=int(match_master_seeds[0]),
                seed_count=seed_count,
                games_per_pair=games_per_pair,
                expected_versions=expected_versions,
            )
            result["focal_policy_id"] = focal_policy_id
            result["report"] = {
                "path": _relative(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            focal_results.append(result)
            source_reports.append(result["report"])
            total_games += int(result["games"])
        aggregate_games = sum(int(row["games"]) for row in focal_results)
        aggregate_score = sum(float(row["score_total"]) for row in focal_results)
        opponents.append({
            "opponent_id": opponent_id,
            "checkpoint_sha256": opponent["checkpoint_sha256"],
            "games": aggregate_games,
            "score_rate": aggregate_score / aggregate_games,
            "confidence_interval_95": list(wilson_interval(
                aggregate_score,
                aggregate_games,
            )),
            "per_focal_policy": focal_results,
        })

    manifest_path = _repo_path(generation_manifest_path)
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_training_payoff_snapshot",
        "immutable": True,
        "source_generation": int(generation["generation"]),
        "target_generation": int(generation["generation"]) + 1,
        "source_generation_manifest": {
            "path": _relative(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "payload_sha256": stable_json_sha256(generation),
        },
        "evaluator": {
            "data_partition": "pfsp_tuning",
            "payoff_update_boundary": "generation_end",
            "match_master_seeds": list(match_master_seeds),
            "seed_count": seed_count,
            "games_per_focal_opponent_pair": games_per_pair,
            "class_ids": list(range(1, 8)),
            "full_matchup_matrix": True,
            "both_player_positions": True,
            "max_agent_steps": 512,
            "match_setup": "official",
            "final_evaluation_seed_count_used": 0,
        },
        "focal_policy_ids": list(focal_policy_ids),
        "focal_policies": [
            {
                "opponent_id": focal_policy_id,
                "checkpoint_path": entries[focal_policy_id]["checkpoint_path"],
                "checkpoint_sha256": entries[focal_policy_id][
                    "checkpoint_sha256"
                ],
            }
            for focal_policy_id in focal_policy_ids
        ],
        "aggregation": "equal_weight_mean_across_focal_policies",
        "opponents": opponents,
        "audit": {
            "expected_opponents": len(trainable),
            "observed_opponents": len(opponents),
            "expected_reports": len(trainable) * len(focal_policy_ids),
            "observed_reports": len(source_reports),
            "total_games": total_games,
            "terminated": total_games,
            "truncated": 0,
            "illegal_actions": 0,
            "action_mask_mismatches": 0,
            "source_reports": source_reports,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Generation 0 PFSP training-only payoff reports."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan_only:
        payload = render_json(build_training_payoff_plan())
        output = _repo_path(args.plan_output)
        if args.check:
            if not output.is_file() or output.read_bytes() != payload:
                print(f"training payoff plan mismatch: {output}")
                return 1
            print("training payoff plan is byte-stable and current")
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        print(f"wrote training payoff plan to {output}")
        return 0
    payload = render_json(build_training_payoff_snapshot())
    output = _repo_path(args.output)
    if args.check:
        if not output.is_file() or output.read_bytes() != payload:
            print(f"training payoff snapshot mismatch: {output}")
            return 1
        print("training payoff snapshot is byte-stable and current")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(f"wrote training payoff snapshot to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
