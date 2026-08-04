from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from swb.db.repository import CardRepository
from swb.engine.environment import ShadowverseEnv
from swb.rl.class_schedule import (
    ALL_CLASS_IDS,
    CLASS_SCHEDULE_VERSION,
    class_pair_for_episode,
)
from swb.rl.fixed_decks import (
    fixed_training_deck_names,
    get_fixed_training_deck,
)
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.versioning import (
    ExperimentVersions,
    action_layout_manifest,
    observation_schema_manifest,
    stable_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = Path("data/reports/league_training")
BASELINE_GIT_COMMIT = "884249b3320e029b1652948a2c1ae6a3d609682d"
BASELINE_BRANCH = "main"
LEAGUE_BRANCH = "feature/league-core"
PRESERVED_USER_CHANGES = ("docs/roadmap.md",)
EXTRA_PP_RULE_COMMIT = "22d8d76806df6ee67ad91761286c4884e3872e03"

NEW_RULE_SEEDS = tuple(range(20260903, 20260909))
LEGACY_ANCHOR_SEEDS = (20260831, 20260901, 20260902)

CHECKPOINT_ROOT = Path("data/checkpoints/ppo_7x7_scaling_20260801")
NEW_REPORT_ROOT = Path(
    "data/reports/ppo_7x7_scaling_1m_expansion_20260803"
)
LEGACY_REPORT_ROOT = Path("data/reports/ppo_7x7_scaling_3m_20260802")

CONTRACT_PATHS = (
    "artifacts.database.file_sha256",
    "artifacts.rulebook.sha256",
    "interfaces.observation.sha256",
    "interfaces.action.sha256",
    "checkpoint_registry_sha256",
)


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


def render_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported manifest value: {type(value).__name__}")


def _file_manifest(path: str | Path) -> dict[str, object]:
    resolved = _repo_path(path)
    return {
        "path": _relative(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _checkpoint_spec(seed: int) -> tuple[str, Path, Path, str, str]:
    if seed in NEW_RULE_SEEDS:
        return (
            "new_rule_1m",
            CHECKPOINT_ROOT / f"seed_{seed}" / "final.pt",
            NEW_REPORT_ROOT / f"seed_{seed}.json",
            "refundable_until_base_pp_is_exceeded",
            EXTRA_PP_RULE_COMMIT,
        )
    if seed in LEGACY_ANCHOR_SEEDS:
        return (
            "legacy_rule_3m_anchor",
            CHECKPOINT_ROOT / f"seed_{seed}" / "final_3m.pt",
            LEGACY_REPORT_ROOT / f"seed_{seed}.json",
            "consumed_immediately_on_activation",
            f"before:{EXTRA_PP_RULE_COMMIT}",
        )
    raise ValueError(f"unregistered checkpoint seed {seed}")


def _model_signature(model_state: Mapping[str, Any]) -> dict[str, object]:
    tensors = []
    parameter_count = 0
    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model_state[{name!r}] is not a tensor")
        shape = list(tensor.shape)
        parameter_count += math.prod(shape)
        tensors.append({
            "name": name,
            "shape": shape,
            "dtype": str(tensor.dtype),
        })
    return {
        "parameter_count": parameter_count,
        "tensor_count": len(tensors),
        "structure_sha256": stable_json_sha256(tensors),
    }


def build_checkpoint_registry() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for seed in (*NEW_RULE_SEEDS, *LEGACY_ANCHOR_SEEDS):
        tier, checkpoint_path, report_path, extra_pp, semantics_version = (
            _checkpoint_spec(seed)
        )
        checkpoint = _repo_path(checkpoint_path)
        source_report = _repo_path(report_path)
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise TypeError(f"checkpoint payload is not a mapping: {checkpoint}")
        trainer = payload.get("trainer")
        versions = payload.get("versions")
        model_state = payload.get("model_state")
        if not isinstance(trainer, dict) or not isinstance(versions, dict):
            raise ValueError(f"checkpoint metadata is incomplete: {checkpoint}")
        if not isinstance(model_state, Mapping):
            raise ValueError(f"checkpoint model_state is missing: {checkpoint}")
        source_payload = json.loads(source_report.read_text(encoding="utf-8"))
        source_steps = int(source_payload["completed_agent_steps"])
        checkpoint_steps = int(trainer["agent_steps"])
        if source_steps != checkpoint_steps:
            raise ValueError(
                f"checkpoint/report step mismatch for seed {seed}: "
                f"{checkpoint_steps} != {source_steps}"
            )
        entries.append({
            "checkpoint_id": f"seed_{seed}_{tier}",
            "seed": seed,
            "tier": tier,
            "role": "training_candidate" if seed in NEW_RULE_SEEDS else "anchor_only",
            "checkpoint": _file_manifest(checkpoint),
            "source_report": _file_manifest(source_report),
            "training": {
                "agent_steps": checkpoint_steps,
                "completed_episodes": int(trainer["completed_episodes"]),
                "update_count": int(trainer["update_count"]),
                "master_seed": int(trainer["master_seed"]),
                "config": _json_safe(trainer["config"]),
            },
            "versions": _json_safe(versions),
            "policy": _model_signature(model_state),
            "rules_semantics": {
                "extra_pp": extra_pp,
                "extra_pp_semantics_version": semantics_version,
                "comparison_scope": (
                    "strict_same_rule_candidate"
                    if seed in NEW_RULE_SEEDS
                    else "cross_rule_historical_anchor_not_steps_ablation"
                ),
            },
        })
        del payload

    version_signatures = {
        stable_json_sha256(entry["versions"]) for entry in entries
    }
    policy_signatures = {
        str(entry["policy"]["structure_sha256"]) for entry in entries
    }
    if len(version_signatures) != 1 or len(policy_signatures) != 1:
        raise ValueError(
            "registered checkpoints do not share one interface/policy structure"
        )
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_checkpoint_registry",
        "entries": entries,
        "summary": {
            "checkpoint_count": len(entries),
            "new_rule_candidate_count": len(NEW_RULE_SEEDS),
            "legacy_anchor_count": len(LEGACY_ANCHOR_SEEDS),
            "shared_versions_sha256": next(iter(version_signatures)),
            "shared_policy_structure_sha256": next(iter(policy_signatures)),
        },
    }


def build_evaluation_protocol() -> dict[str, object]:
    training_seeds = list(NEW_RULE_SEEDS)
    tuning_seeds = list(range(20261001, 20261011))
    final_seeds = list(range(20262001, 20262021))
    if set(training_seeds) & set(tuning_seeds):
        raise AssertionError("training and tuning seeds overlap")
    if set(training_seeds) & set(final_seeds):
        raise AssertionError("training and final seeds overlap")
    if set(tuning_seeds) & set(final_seeds):
        raise AssertionError("tuning and final seeds overlap")
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_evaluation_protocol",
        "seed_partitions": {
            "training_model_seeds": training_seeds,
            "pfsp_tuning_match_master_seeds": tuning_seeds,
            "final_evaluation_match_master_seeds": final_seeds,
            "exploratory_matrix_seed_already_used": 20260804,
            "final_seeds_allowed_for_pfsp": False,
            "partition_rule": (
                "Final evaluation seeds are held out from opponent selection, "
                "PFSP payoff updates, hyperparameter tuning, and early stopping."
            ),
        },
        "evaluation_scale": {
            "screening_games_per_model_pair": 196,
            "confirmation_public_match_seeds": 10,
            "confirmation_class_cells": 49,
            "confirmation_games_per_model_pair": 980,
            "both_player_positions_required": True,
        },
        "primary_metrics": [
            {
                "name": "frozen_anchor_paired_mean_win_rate",
                "uncertainty": "paired_bootstrap_95_ci",
            },
            {
                "name": "meta_strategy_worst_expected_payoff",
                "companion": "best_response_exploitability_proxy",
            },
            {"name": "class_matrix_worst_cell"},
            {"name": "class_matrix_p10_cell"},
            {
                "name": "tactical_replay_preference",
                "components": ["top1", "topk", "preference_margin"],
            },
            {
                "name": "safety",
                "requirements": {
                    "illegal_actions": 0,
                    "action_mask_mismatches": 0,
                    "truncation_rate": 0.0,
                },
            },
        ],
        "diagnostic_metrics_not_standalone_success": [
            "agent_steps_per_second",
            "gpu_utilization",
            "gpu_memory",
            "cpu_utilization",
            "ram_utilization",
            "episode_agent_steps",
            "episode_turns",
            "policy_entropy",
            "value_loss",
            "policy_loss",
            "clip_fraction",
            "approx_kl",
            "explained_variance",
            "grad_norm",
            "opponent_selection_distribution",
        ],
        "reporting": {
            "win_rate_interval": "wilson_95_ci",
            "multi_seed_aggregates": [
                "per_seed",
                "median",
                "iqm",
                "paired_bootstrap_95_ci",
                "performance_profile",
            ],
            "elo_is_primary": False,
            "all_runs_included": True,
        },
    }


def _make_interface_manifests(
    snapshot: WorkerAssetsSnapshot,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    assets = snapshot.load()
    recipe = get_fixed_training_deck(sorted(fixed_training_deck_names())[0])
    deck = recipe.build(assets.catalog)
    env = ShadowverseEnv(
        deck,
        deck,
        class_a=recipe.class_id,
        class_b=recipe.class_id,
        seed=1,
        rulebook=assets.rulebook,
        card_resolver=assets.catalog.resolve,
        observation_version="v4.1",
        card_vocabulary=assets.catalog.card_vocabulary,
        max_game_turns=200,
        max_agent_steps=256,
        training_mode=True,
        match_setup="official",
    )
    observation = observation_schema_manifest(env)
    action = action_layout_manifest(env)
    versions = ExperimentVersions.capture(
        env,
        assets.catalog,
        rulebook_sha256=assets.rulebook_sha256,
    ).to_dict()
    return observation, action, versions


def build_baseline_manifest(
    checkpoint_registry: Mapping[str, object],
    evaluation_protocol: Mapping[str, object],
) -> dict[str, object]:
    database = _repo_path("data/cards.sqlite3")
    repository = CardRepository(database)
    snapshot = WorkerAssetsSnapshot.build(repository)
    catalog = snapshot.catalog
    observation, action, versions = _make_interface_manifests(snapshot)
    decks = [
        get_fixed_training_deck(name).manifest()
        for name in sorted(fixed_training_deck_names())
    ]
    pairs = [
        list(class_pair_for_episode(ALL_CLASS_IDS, episode_id))
        for episode_id in range(len(ALL_CLASS_IDS) ** 2)
    ]
    schedule = {
        "class_schedule_version": CLASS_SCHEDULE_VERSION,
        "class_ids": list(ALL_CLASS_IDS),
        "cycle_length": len(pairs),
        "ordered_pairs": pairs,
        "deck_source": "seeded_exact_catalog_sampling_per_class",
        "fixed_decks_are_evaluation_and_audit_decks": True,
    }
    source_snapshot = _json_safe(repository.source_snapshot())
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_baseline_manifest",
        "source_control": {
            "baseline_commit": BASELINE_GIT_COMMIT,
            "baseline_branch": BASELINE_BRANCH,
            "league_branch": LEAGUE_BRANCH,
            "preserved_user_changes_excluded_from_commits": list(
                PRESERVED_USER_CHANGES
            ),
        },
        "artifacts": {
            "database": {
                "path": _relative(database),
                "bytes": database.stat().st_size,
                "file_sha256": _sha256_file(database),
                "source_snapshot": source_snapshot,
            },
            "rulebook": {
                "directory": _relative(ShadowverseEnv.DEFAULT_RULE_DIRECTORY),
                "sha256": snapshot.rulebook_sha256,
            },
            "coverage_report": _file_manifest(
                "data/reports/rule_coverage.json"
            ),
            "catalog": {
                "sha256": catalog.catalog_sha256,
                "card_vocabulary_sha256": catalog.card_vocabulary_sha256,
                "training_pool_sha256": catalog.training_pool_sha256,
                "exact_collectible_count": len(catalog.exact_collectible_ids),
                "excluded_collectible_count": len(
                    catalog.excluded_collectible_ids
                ),
            },
            "fixed_decks": {
                "count": len(decks),
                "manifest_sha256": stable_json_sha256(decks),
                "decks": decks,
            },
            "class_schedule": {
                **schedule,
                "sha256": stable_json_sha256(schedule),
            },
        },
        "interfaces": {
            "observation": {
                "version": observation["version"],
                "sha256": stable_json_sha256(observation),
                "manifest": observation,
            },
            "action": {
                "version": action["version"],
                "sha256": stable_json_sha256(action),
                "manifest": action,
            },
            "experiment_versions": versions,
            "seed_derivation": {
                "version": versions["seed_derivation_version"],
                "algorithm": "stable-json-sha256-first-64-bits",
            },
        },
        "policy_architecture": {
            "name": "entity_action_v1",
            "model_parameters": checkpoint_registry["entries"][0]["policy"][
                "parameter_count"
            ],
            "structure_sha256": checkpoint_registry["summary"][
                "shared_policy_structure_sha256"
            ],
            "frozen_config": checkpoint_registry["entries"][0]["training"][
                "config"
            ],
        },
        "checkpoint_registry_sha256": stable_json_sha256(checkpoint_registry),
        "evaluation_protocol_sha256": stable_json_sha256(evaluation_protocol),
        "contract_invalidation_fields": list(CONTRACT_PATHS),
    }


def _lookup(payload: Mapping[str, object], dotted_path: str) -> object:
    current: object = payload
    for key in dotted_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def contract_differences(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> list[dict[str, object]]:
    return [
        {
            "field": path,
            "expected": _lookup(expected, path),
            "actual": _lookup(actual, path),
        }
        for path in CONTRACT_PATHS
        if _lookup(expected, path) != _lookup(actual, path)
    ]


def build_reports() -> dict[str, dict[str, object]]:
    checkpoint_registry = build_checkpoint_registry()
    evaluation_protocol = build_evaluation_protocol()
    baseline_manifest = build_baseline_manifest(
        checkpoint_registry,
        evaluation_protocol,
    )
    return {
        "baseline_manifest.json": baseline_manifest,
        "evaluation_protocol.json": evaluation_protocol,
        "checkpoint_registry.json": checkpoint_registry,
    }


def write_reports(
    reports: Mapping[str, Mapping[str, object]],
    output_directory: str | Path,
) -> None:
    output = _repo_path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in reports.items():
        (output / name).write_bytes(render_json(payload))


def check_reports(
    reports: Mapping[str, Mapping[str, object]],
    output_directory: str | Path,
) -> list[str]:
    output = _repo_path(output_directory)
    mismatches = []
    for name, payload in reports.items():
        path = output / name
        if not path.is_file() or path.read_bytes() != render_json(payload):
            mismatches.append(name)
    return mismatches


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the League stage 3.0 experiment contract."
    )
    parser.add_argument(
        "--output-directory",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify saved reports instead of writing them",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reports = build_reports()
    if args.check:
        mismatches = check_reports(reports, args.output_directory)
        if mismatches:
            print("league baseline mismatch: " + ", ".join(mismatches))
            return 1
        print("league baseline contract is byte-stable and current")
        return 0
    write_reports(reports, args.output_directory)
    print(
        "wrote League baseline contract to "
        f"{_repo_path(args.output_directory)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
