from __future__ import annotations

import argparse
import atexit
import json
import time
from pathlib import Path

import torch

from swb.db.repository import CardRepository
from swb.rl.checkpoint import load_checkpoint, save_checkpoint_atomic
from swb.rl.ppo import PPOConfig, PPOTrainer
from swb.rl.runtime import WorkerAssetsSnapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the shared recurrent masked PPO SWB baseline"
    )
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--total-agent-steps", type=int, default=10_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--rollout-workers", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=256)
    parser.add_argument("--master-seed", type=int, default=20260721)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opponent-current-weight", type=float, default=0.6)
    parser.add_argument("--opponent-random-weight", type=float, default=0.2)
    parser.add_argument("--opponent-fixed-weight", type=float, default=0.2)
    parser.add_argument("--opponent-historical-weight", type=float, default=0.5)
    parser.add_argument("--opponent-max-history", type=int, default=4)
    parser.add_argument(
        "--opponent-snapshot-interval-steps", type=int, default=2_500
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/checkpoints/ppo_smoke.pt"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/reports/ppo_smoke_training.json"),
    )
    args = parser.parse_args()
    if args.total_agent_steps <= 0 or args.rollout_steps <= 0:
        parser.error("total-agent-steps and rollout-steps must be positive")
    if args.rollout_workers <= 0:
        parser.error("rollout-workers must be positive")
    if args.resume is None and args.rollout_workers > 1 and (
        args.opponent_current_weight,
        args.opponent_random_weight,
        args.opponent_fixed_weight,
        args.opponent_historical_weight,
    ) != (1.0, 0.0, 0.0, 0.0):
        parser.error(
            "multiprocess rollout currently requires current-policy self-play "
            "weights: 1, 0, 0, 0"
        )

    snapshot = WorkerAssetsSnapshot.build(CardRepository(args.database))
    if args.resume is not None:
        trainer = load_checkpoint(args.resume, snapshot, device=args.device)
    else:
        trainer = PPOTrainer(
            snapshot,
            master_seed=args.master_seed,
            config=PPOConfig(
                rollout_steps=args.rollout_steps,
                max_agent_steps_per_episode=args.max_episode_steps,
                opponent_current_weight=args.opponent_current_weight,
                opponent_random_weight=args.opponent_random_weight,
                opponent_fixed_weight=args.opponent_fixed_weight,
                opponent_historical_weight=args.opponent_historical_weight,
                opponent_max_history=args.opponent_max_history,
                opponent_snapshot_interval_steps=(
                    args.opponent_snapshot_interval_steps
                ),
                rollout_workers=args.rollout_workers,
            ),
            device=args.device,
        )
    if args.total_agent_steps <= trainer.agent_steps:
        parser.error(
            f"target {args.total_agent_steps} must exceed checkpoint progress "
            f"{trainer.agent_steps}"
        )
    atexit.register(trainer.close)

    started = time.perf_counter()
    starting_agent_steps = trainer.agent_steps
    metrics = []
    historical_snapshots = []
    history_directory = (
        args.checkpoint.parent / f"{args.checkpoint.stem}_history"
    )
    while trainer.agent_steps < args.total_agent_steps:
        records, bootstrap, _ = trainer.collect_rollout()
        update_metrics = trainer.update(records, bootstrap)
        metrics.append(update_metrics)
        print(json.dumps(update_metrics, sort_keys=True))
        if trainer.opponent_pool.snapshot_due(trainer.agent_steps):
            history_path = history_directory / (
                f"step_{trainer.agent_steps:012d}.pt"
            )
            previous_paths = {
                entry.checkpoint_path
                for entry in trainer.opponent_pool.entries
                if entry.kind == "historical"
            }
            save_checkpoint_atomic(history_path, trainer)
            entry = trainer.opponent_pool.register_snapshot(
                history_path, agent_steps=trainer.agent_steps
            )
            historical_snapshots.append(entry.checkpoint_path)
            retained_paths = {
                candidate.checkpoint_path
                for candidate in trainer.opponent_pool.entries
                if candidate.kind == "historical"
            }
            history_root = history_directory.resolve()
            for stale in previous_paths - retained_paths:
                stale_path = Path(stale).resolve()
                if (
                    stale_path.parent == history_root
                    and stale_path.name.startswith("step_")
                    and stale_path.suffix == ".pt"
                    and stale_path.exists()
                ):
                    stale_path.unlink()
    elapsed = time.perf_counter() - started
    save_checkpoint_atomic(args.checkpoint, trainer)
    report = {
        "schema_version": 1,
        "purpose": "CPU smoke only; not a policy-strength conclusion",
        "torch_version": torch.__version__,
        "device": str(trainer.device),
        "master_seed": trainer.master_seed,
        "requested_agent_steps": args.total_agent_steps,
        "completed_agent_steps": trainer.agent_steps,
        "completed_episodes": trainer.completed_episodes,
        "updates": trainer.update_count,
        "elapsed_seconds": elapsed,
        "agent_steps_per_second": (
            (trainer.agent_steps - starting_agent_steps) / max(elapsed, 1e-12)
        ),
        "checkpoint": str(args.checkpoint),
        "resumed_from": None if args.resume is None else str(args.resume),
        "hyperparameters": trainer.hyperparameters(),
        "final_metrics": metrics[-1],
        "opponent_pool": trainer.opponent_pool.state_dict(),
        "opponent_assignments": list(trainer.opponent_assignments),
        "historical_snapshots_created": historical_snapshots,
        "versions": {
            "catalog_sha256": snapshot.catalog.catalog_sha256,
            "card_vocabulary_sha256": snapshot.catalog.card_vocabulary_sha256,
            "rulebook_sha256": snapshot.rulebook_sha256,
        },
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    trainer.close()
    atexit.unregister(trainer.close)


if __name__ == "__main__":
    main()
