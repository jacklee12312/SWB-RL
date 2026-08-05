from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from scripts.report_ppo_league_baseline import ROOT, render_json


REPORT_ROOT = Path(
    "data/reports/league_training/sampler_screen_20260804"
)
DEFAULT_OUTPUT = REPORT_ROOT / "night_queue_plan.json"
GENERATION_MANIFEST = Path(
    "data/reports/league_training/generation_000_manifest.json"
)
PAYOFF_SNAPSHOT = Path(
    "data/reports/league_training/generation_000_training_payoff_snapshot.json"
)
EVALUATION_PROTOCOL = Path(
    "data/reports/league_training/evaluation_protocol.json"
)
BASE_CHECKPOINT = Path(
    "data/checkpoints/ppo_7x7_scaling_20260801/seed_20260903/final.pt"
)
SAMPLERS = ("uniform", "variance", "hard")
TRAINING_SEEDS = (20261101, 20261102, 20261103)
TRAINING_ADDITIONAL_STEPS = 100_000
TRAINING_EVALUATION_MASTER_SEED = 20261002
PAYOFF_MASTER_SEED = 20261001
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


def _source(path: str | Path) -> dict[str, object]:
    return {
        "path": _relative(path),
        "sha256": _sha256_file(path),
        "bytes": _repo_path(path).stat().st_size,
    }


def _entry_by_id(
    generation: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    return {
        str(entry["opponent_id"]): entry
        for entry in generation["entries"]  # type: ignore[index]
    }


def _evaluation_job(
    *,
    job_id: str,
    focal_id: str,
    focal_checkpoint: str,
    focal_sha256: str | None,
    opponent: Mapping[str, object],
    output_directory: Path,
    master_seed: int,
) -> dict[str, object]:
    return {
        "job_id": job_id,
        "focal_policy_id": focal_id,
        "focal_checkpoint": focal_checkpoint,
        "focal_checkpoint_sha256": focal_sha256,
        "opponent_id": str(opponent["opponent_id"]),
        "opponent_checkpoint": str(opponent["checkpoint_path"]),
        "opponent_checkpoint_sha256": str(opponent["checkpoint_sha256"]),
        "master_seed": master_seed,
        "seed_count": SEED_COUNT,
        "games": GAMES_PER_PAIR,
        "output": _relative(output_directory / f"{job_id}.json"),
        "log": _relative(output_directory / f"{job_id}.log"),
    }


def build_plan() -> dict[str, object]:
    generation_path = _repo_path(GENERATION_MANIFEST)
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    protocol_path = _repo_path(EVALUATION_PROTOCOL)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    tuning_seeds = set(
        protocol["seed_partitions"]["pfsp_tuning_match_master_seeds"]
    )
    final_seeds = set(
        protocol["seed_partitions"]["final_evaluation_match_master_seeds"]
    )
    used_match_seeds = {
        TRAINING_EVALUATION_MASTER_SEED,
        PAYOFF_MASTER_SEED,
    }
    if not used_match_seeds <= tuning_seeds or used_match_seeds & final_seeds:
        raise ValueError("night queue match seeds violate tuning partition")

    entries = _entry_by_id(generation)
    active = [
        entries[f"seed_{seed}_1m"]
        for seed in range(20260903, 20260909)
    ]
    archive = sorted(
        (
            entry
            for entry in entries.values()
            if entry["role"] == "self_history"
            and bool(entry["training_eligible"])
        ),
        key=lambda row: str(row["opponent_id"]),
    )
    if len(active) != 6 or len(archive) != 18:
        raise ValueError("Generation 0 requires six active and 18 archive models")

    base_payload = torch.load(
        _repo_path(BASE_CHECKPOINT), map_location="cpu", weights_only=False
    )
    base_steps = int(base_payload["trainer"]["agent_steps"])
    base_seed = int(base_payload["trainer"]["master_seed"])
    del base_payload
    if base_seed != 20260903:
        raise ValueError("sampler screen base checkpoint must be seed 20260903")
    target_steps = base_steps + TRAINING_ADDITIONAL_STEPS

    training_jobs = []
    for sampler in SAMPLERS:
        manifest = REPORT_ROOT / "manifests" / f"{sampler}.json"
        manifest_payload = json.loads(
            _repo_path(manifest).read_text(encoding="utf-8")
        )
        if manifest_payload.get("selection_mode") != sampler:
            raise ValueError(f"wrong sampler manifest for {sampler}")
        for training_seed in TRAINING_SEEDS:
            run_id = f"{sampler}__train_seed_{training_seed}"
            run_root = Path("data/checkpoints/league_sampler_screen_20260804") / sampler / f"train_seed_{training_seed}"
            training_jobs.append({
                "job_id": run_id,
                "sampler": sampler,
                "training_seed": training_seed,
                "parent_checkpoint": _relative(BASE_CHECKPOINT),
                "parent_checkpoint_sha256": _sha256_file(BASE_CHECKPOINT),
                "parent_agent_steps": base_steps,
                "target_agent_steps": target_steps,
                "additional_agent_steps": TRAINING_ADDITIONAL_STEPS,
                "opponent_manifest": _relative(manifest),
                "opponent_manifest_sha256": _sha256_file(manifest),
                "checkpoint": _relative(run_root / "final_100k.pt"),
                "metrics": _relative(
                    REPORT_ROOT / "training" / f"{run_id}.json"
                ),
                "log": _relative(
                    REPORT_ROOT / "training" / f"{run_id}.log"
                ),
            })

    candidate_evaluations = []
    for training in training_jobs:
        for opponent in active:
            job_id = (
                f"{training['job_id']}__vs__{opponent['opponent_id']}"
            )
            candidate_evaluations.append(_evaluation_job(
                job_id=job_id,
                focal_id=str(training["job_id"]),
                focal_checkpoint=str(training["checkpoint"]),
                focal_sha256=None,
                opponent=opponent,
                output_directory=REPORT_ROOT / "candidate_evaluations",
                master_seed=TRAINING_EVALUATION_MASTER_SEED,
            ))

    existing_active_pairs = []
    missing_active_pairs = []
    for left, right in itertools.combinations(active, 2):
        left_id = str(left["opponent_id"])
        right_id = str(right["opponent_id"])
        job_id = f"{left_id}__vs__{right_id}"
        if left_id == "seed_20260903_1m":
            report = (
                Path("data/reports/league_training/")
                / "generation_000_payoff_evaluations"
                / f"{job_id}.json"
            )
            existing_active_pairs.append({
                "job_id": job_id,
                "report": _source(report),
            })
        else:
            missing_active_pairs.append(_evaluation_job(
                job_id=job_id,
                focal_id=left_id,
                focal_checkpoint=str(left["checkpoint_path"]),
                focal_sha256=str(left["checkpoint_sha256"]),
                opponent=right,
                output_directory=REPORT_ROOT / "generation_000_active_matrix",
                master_seed=PAYOFF_MASTER_SEED,
            ))

    archive_jobs = []
    for focal in active[1:]:
        focal_id = str(focal["opponent_id"])
        for opponent in archive:
            job_id = f"{focal_id}__vs__{opponent['opponent_id']}"
            archive_jobs.append(_evaluation_job(
                job_id=job_id,
                focal_id=focal_id,
                focal_checkpoint=str(focal["checkpoint_path"]),
                focal_sha256=str(focal["checkpoint_sha256"]),
                opponent=opponent,
                output_directory=REPORT_ROOT / "archive_baseline",
                master_seed=PAYOFF_MASTER_SEED,
            ))

    evaluation_count = (
        len(candidate_evaluations)
        + len(missing_active_pairs)
        + len(archive_jobs)
    )
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_sampler_screen_night_queue_plan",
        "immutable": True,
        "data_partition": "pfsp_tuning",
        "sources": {
            "generation_manifest": _source(GENERATION_MANIFEST),
            "payoff_snapshot": _source(PAYOFF_SNAPSHOT),
            "evaluation_protocol": _source(EVALUATION_PROTOCOL),
            "base_checkpoint": _source(BASE_CHECKPOINT),
        },
        "training": {
            "samplers": list(SAMPLERS),
            "training_seeds": list(TRAINING_SEEDS),
            "additional_agent_steps_per_job": TRAINING_ADDITIONAL_STEPS,
            "total_additional_agent_steps": (
                len(training_jobs) * TRAINING_ADDITIONAL_STEPS
            ),
            "runtime": {
                "device": "cuda",
                "rollout_workers": 7,
                "rollout_worker_torch_threads": 2,
                "central_inference_batch_wait_ms": 1.0,
                "opponent_model_cache_size": 7,
                "opponent_model_cache_max_mib": 512,
                "opponent_batching_mode": "episode_seed_clustered",
                "checkpoint_interval_agent_steps": 50_000,
            },
            "jobs": training_jobs,
        },
        "candidate_evaluation": {
            "master_seed": TRAINING_EVALUATION_MASTER_SEED,
            "seed_count": SEED_COUNT,
            "games_per_pair": GAMES_PER_PAIR,
            "jobs": candidate_evaluations,
        },
        "generation_000_active_matrix": {
            "master_seed": PAYOFF_MASTER_SEED,
            "seed_count": SEED_COUNT,
            "games_per_pair": GAMES_PER_PAIR,
            "existing_pairs": existing_active_pairs,
            "jobs": missing_active_pairs,
        },
        "archive_baseline": {
            "master_seed": PAYOFF_MASTER_SEED,
            "seed_count": SEED_COUNT,
            "games_per_pair": GAMES_PER_PAIR,
            "focal_policy_ids": [
                str(entry["opponent_id"]) for entry in active[1:]
            ],
            "archive_opponent_ids": [
                str(entry["opponent_id"]) for entry in archive
            ],
            "jobs": archive_jobs,
        },
        "summary": {
            "training_job_count": len(training_jobs),
            "candidate_evaluation_pair_count": len(candidate_evaluations),
            "existing_active_pair_count": len(existing_active_pairs),
            "missing_active_pair_count": len(missing_active_pairs),
            "archive_baseline_pair_count": len(archive_jobs),
            "queued_evaluation_pair_count": evaluation_count,
            "queued_evaluation_game_count": (
                evaluation_count * GAMES_PER_PAIR
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the frozen overnight sampler screen plan."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = _repo_path(args.output)
    expected = render_json(build_plan())
    if args.check:
        if not output.is_file() or output.read_bytes() != expected:
            print(f"sampler screen plan mismatch: {output}")
            return 1
        print("sampler screen night queue plan is byte-stable and current")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    print(f"wrote sampler screen night queue plan to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
