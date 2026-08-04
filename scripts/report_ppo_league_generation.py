from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import torch

from scripts.report_ppo_league_baseline import (
    EXTRA_PP_RULE_COMMIT,
    NEW_RULE_SEEDS,
    ROOT,
    render_json,
)
from swb.rl.class_schedule import ALL_CLASS_IDS, class_pair_for_episode
from swb.rl.seeding import derive_seed
from swb.rl.versioning import stable_json_sha256


REPORT_DIRECTORY = Path("data/reports/league_training")
DEFAULT_OUTPUT = REPORT_DIRECTORY / "generation_000_manifest.json"
CHECKPOINT_REGISTRY = REPORT_DIRECTORY / "checkpoint_registry.json"
BASELINE_MANIFEST = REPORT_DIRECTORY / "baseline_manifest.json"
META_GAME_REPORT = REPORT_DIRECTORY / "meta_game.json"
CHECKPOINT_ROOT = Path("data/checkpoints/ppo_7x7_scaling_20260801")
SELECTION_AUDIT_SEED = 20261001
SELECTION_AUDIT_EPISODES = 4096


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


def _json_file_source(path: str | Path) -> dict[str, object]:
    resolved = _repo_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return {
        "path": _relative(resolved),
        "file_sha256": _sha256_file(resolved),
        "payload_sha256": stable_json_sha256(payload),
    }


def _model_values_sha256(
    model_state: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in model_state.items():
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_entry(
    path: str | Path,
    *,
    opponent_id: str,
    role: str,
    rules_version: str,
    training_eligible: bool,
) -> dict[str, object]:
    checkpoint = _repo_path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer = payload["trainer"]
    versions = payload["versions"]
    model_state = payload["model_state"]
    entry = {
        "opponent_id": opponent_id,
        "checkpoint_path": _relative(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "model_values_sha256": _model_values_sha256(model_state),
        "policy_seed": int(trainer["master_seed"]),
        "training_steps": int(trainer["agent_steps"]),
        "generation": 0,
        "role": role,
        "rules_version": rules_version,
        "policy_architecture": str(trainer["config"]["policy_architecture"]),
        "versions_sha256": stable_json_sha256(versions),
        "training_eligible": training_eligible,
        "sampling_weight": 1.0 if training_eligible else 0.0,
    }
    del payload
    return entry


def _selection_audit(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    eligible = [entry for entry in entries if entry["training_eligible"]]
    ids = [str(entry["opponent_id"]) for entry in eligible]
    weights = [float(entry["sampling_weight"]) for entry in eligible]
    counts: Counter[str] = Counter()
    class_pair_positions = set()
    positions = set()
    for episode_id in range(SELECTION_AUDIT_EPISODES):
        learner_player = episode_id % 2
        positions.add(learner_player)
        class_pair_positions.add((
            *class_pair_for_episode(ALL_CLASS_IDS, episode_id),
            learner_player,
        ))
        rng = random.Random(derive_seed(
            SELECTION_AUDIT_SEED,
            "opponent",
            episode_id,
            learner_player,
        ))
        counts[rng.choices(ids, weights=weights, k=1)[0]] += 1
    missing = sorted(set(ids) - set(counts))
    if missing:
        raise ValueError(
            "fixed selection audit did not hit trainable entries: "
            + ", ".join(missing)
        )
    if len(class_pair_positions) != len(ALL_CLASS_IDS) ** 2 * 2:
        raise ValueError("fixed selection audit lacks 7x7/both-position coverage")
    return {
        "master_seed": SELECTION_AUDIT_SEED,
        "episode_ids": [0, SELECTION_AUDIT_EPISODES - 1],
        "episode_count": SELECTION_AUDIT_EPISODES,
        "learner_player_rule": "episode_id_mod_2",
        "selection_count_by_opponent": dict(sorted(counts.items())),
        "all_trainable_entries_hit": True,
        "reference_entries_hit": 0,
        "learner_positions": sorted(positions),
        "class_pair_position_coverage": len(class_pair_positions),
        "required_class_pair_position_coverage": (
            len(ALL_CLASS_IDS) ** 2 * 2
        ),
    }


def build_generation_manifest() -> dict[str, object]:
    registry_path = _repo_path(CHECKPOINT_REGISTRY)
    baseline_path = _repo_path(BASELINE_MANIFEST)
    meta_path = _repo_path(META_GAME_REPORT)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    meta_game = json.loads(meta_path.read_text(encoding="utf-8"))
    registry_by_seed = {
        int(entry["seed"]): entry for entry in registry["entries"]
    }
    disposition_by_model = {
        str(row["model_id"]): row
        for row in meta_game["population_selection_evidence"]
    }

    entries: list[dict[str, object]] = []
    new_rules_version = (
        f"{EXTRA_PP_RULE_COMMIT}:"
        "refundable_until_base_pp_is_exceeded"
    )
    for seed in NEW_RULE_SEEDS:
        registry_entry = registry_by_seed[seed]
        model_id = f"seed_{seed}_1m"
        if disposition_by_model[model_id]["generation_0_disposition"] != (
            "include_candidate"
        ):
            raise ValueError(f"meta-game excludes required candidate {model_id}")
        history_paths = sorted(
            _repo_path(
                CHECKPOINT_ROOT / f"seed_{seed}" / "final_history"
            ).glob("step_*.pt")
        )
        if len(history_paths) != 3:
            raise ValueError(
                f"seed {seed} requires exactly three representative histories"
            )
        for history_path in history_paths:
            entry = _checkpoint_entry(
                history_path,
                opponent_id=f"seed_{seed}_{history_path.stem}",
                role="self_history",
                rules_version=new_rules_version,
                training_eligible=True,
            )
            entries.append(entry)
        final_path = registry_entry["checkpoint"]["path"]
        entries.append(_checkpoint_entry(
            final_path,
            opponent_id=model_id,
            role="candidate_final",
            rules_version=new_rules_version,
            training_eligible=True,
        ))

    for seed, registry_entry in sorted(registry_by_seed.items()):
        if seed in NEW_RULE_SEEDS:
            continue
        model_id = f"seed_{seed}_3m"
        if disposition_by_model[model_id]["generation_0_disposition"] != (
            "include_anchor_only"
        ):
            raise ValueError(f"meta-game lacks anchor disposition for {model_id}")
        entries.append(_checkpoint_entry(
            registry_entry["checkpoint"]["path"],
            opponent_id=model_id,
            role="anchor_only",
            rules_version=(
                f"before:{EXTRA_PP_RULE_COMMIT}:"
                "consumed_immediately_on_activation"
            ),
            training_eligible=False,
        ))

    entries.sort(key=lambda entry: str(entry["opponent_id"]))
    seen_model_values: dict[str, str] = {}
    duplicates = []
    for entry in entries:
        model_hash = str(entry["model_values_sha256"])
        previous = seen_model_values.get(model_hash)
        if previous is not None:
            duplicates.append({
                "retained": previous,
                "duplicate": entry["opponent_id"],
                "model_values_sha256": model_hash,
            })
        else:
            seen_model_values[model_hash] = str(entry["opponent_id"])
    if duplicates:
        raise ValueError(
            "Generation 0 source checkpoints contain duplicate model values: "
            f"{duplicates}"
        )

    versions = registry["entries"][0]["versions"]
    contract = {
        "experiment_versions": versions,
        "policy_architecture": baseline["policy_architecture"]["name"],
        "model_structure_sha256": registry["summary"][
            "shared_policy_structure_sha256"
        ],
        "catalog_sha256": versions["catalog_sha256"],
        "rulebook_sha256": versions["rulebook_sha256"],
        "observation_schema_sha256": versions[
            "observation_schema_sha256"
        ],
        "action_layout_sha256": versions["action_layout_sha256"],
        "database_file_sha256": baseline["artifacts"]["database"][
            "file_sha256"
        ],
        "checkpoint_registry_payload_sha256": stable_json_sha256(registry),
    }
    audit = _selection_audit(entries)
    trainable_count = sum(bool(entry["training_eligible"]) for entry in entries)
    anchor_count = len(entries) - trainable_count
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_generation_manifest",
        "immutable": True,
        "path_base": "repository_root",
        "generation": 0,
        "selection_mode": "uniform",
        "contract": contract,
        "sources": {
            "baseline_manifest": _json_file_source(BASELINE_MANIFEST),
            "checkpoint_registry": _json_file_source(CHECKPOINT_REGISTRY),
            "meta_game": _json_file_source(META_GAME_REPORT),
        },
        "deduplication": {
            "key": "model_values_sha256",
            "duplicate_groups": duplicates,
            "unique_model_count": len(seen_model_values),
        },
        "entries": entries,
        "selection_audit": audit,
        "summary": {
            "entry_count": len(entries),
            "trainable_entry_count": trainable_count,
            "candidate_final_count": sum(
                entry["role"] == "candidate_final" for entry in entries
            ),
            "self_history_count": sum(
                entry["role"] == "self_history" for entry in entries
            ),
            "anchor_only_count": anchor_count,
            "trainable_raw_sampling_weight_total": sum(
                float(entry["sampling_weight"]) for entry in entries
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the immutable League Generation 0 pool."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_generation_manifest()
    output = _repo_path(args.output)
    expected = render_json(payload)
    if args.check:
        if not output.is_file() or output.read_bytes() != expected:
            print(f"Generation 0 manifest mismatch: {output}")
            return 1
        print("Generation 0 manifest is byte-stable and current")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    print(f"wrote Generation 0 manifest to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
