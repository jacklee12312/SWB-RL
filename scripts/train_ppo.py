from __future__ import annotations

import argparse
import atexit
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from swb.db.repository import CardRepository
from swb.engine.environment import MATCH_SETUP_OFFICIAL, MATCH_SETUP_VALUES
from swb.rl.checkpoint import load_checkpoint, save_checkpoint_atomic
from swb.rl.class_schedule import ALL_CLASS_IDS, CLASS_SCHEDULE_VERSION
from swb.rl.deck_schedule import DECK_MATCHUP_SCHEDULE_VERSION
from swb.rl.fixed_decks import (
    fixed_training_deck_names,
    get_fixed_training_deck,
)
from swb.rl.ppo import PPOConfig, PPOTrainer
from swb.rl.policy import (
    ENTITY_ACTION_POLICY_ARCHITECTURE,
    POLICY_ARCHITECTURES,
)
from swb.rl.profiling import training_timing_report
from swb.rl.runtime import WorkerAssetsSnapshot


RUNTIME_OVERRIDE_FIELDS = (
    "rollout_workers",
    "rollout_worker_torch_threads",
    "central_inference_batch_wait_seconds",
)


def _resume_runtime_config(
    config: PPOConfig,
    *,
    rollout_workers: int,
    rollout_worker_threads: int,
    central_inference_batch_wait_ms: float,
) -> PPOConfig:
    return replace(
        config,
        rollout_workers=rollout_workers,
        rollout_worker_torch_threads=rollout_worker_threads,
        central_inference_batch_wait_seconds=(
            central_inference_batch_wait_ms / 1000.0
        ),
    )


def _runtime_override_report(
    before: PPOConfig,
    after: PPOConfig,
) -> dict[str, dict[str, object]]:
    before_values = asdict(before)
    after_values = asdict(after)
    return {
        field: {
            "before": before_values[field],
            "after": after_values[field],
        }
        for field in RUNTIME_OVERRIDE_FIELDS
        if before_values[field] != after_values[field]
    }


def _periodic_checkpoint_due(
    *,
    last_checkpoint_steps: int,
    current_steps: int,
    interval_steps: int,
) -> bool:
    return (
        interval_steps > 0
        and current_steps - last_checkpoint_steps >= interval_steps
    )


def _periodic_checkpoint_path(
    checkpoint: Path,
    agent_steps: int,
) -> Path:
    return (
        checkpoint.parent
        / f"{checkpoint.stem}_checkpoints"
        / f"step_{agent_steps:012d}.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the shared recurrent masked PPO SWB baseline"
    )
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--total-agent-steps", type=int, default=10_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--rollout-workers", type=int, default=1)
    parser.add_argument("--rollout-worker-threads", type=int, default=2)
    parser.add_argument(
        "--central-inference-batch-wait-ms",
        type=float,
        default=0.5,
    )
    parser.add_argument("--max-episode-steps", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--minibatch-sequences", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument(
        "--policy-architecture",
        choices=sorted(POLICY_ARCHITECTURES),
        default=ENTITY_ACTION_POLICY_ARCHITECTURE,
    )
    parser.add_argument(
        "--observation-version",
        choices=("v3", "v4", "v4.1"),
        default="v4.1",
        help=(
            "v4.1 is the structured-token schema for new training; "
            "v4 and v3 are retained for checkpoint compatibility"
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--card-embedding-dim", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--master-seed", type=int, default=20260721)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="ordered playable class IDs used by the deterministic matchup cycle",
    )
    parser.add_argument(
        "--training-deck",
        choices=fixed_training_deck_names(),
        help="use one named fixed deck for mirrored self-play training",
    )
    parser.add_argument(
        "--opponent-decks",
        nargs="+",
        choices=fixed_training_deck_names(),
        help=(
            "train only the --training-deck side while deterministically "
            "cycling these fixed opponent decks and both player positions"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--match-setup",
        choices=sorted(MATCH_SETUP_VALUES),
        default=MATCH_SETUP_OFFICIAL,
        help="official enables seeded random first player and interactive mulligan",
    )
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
        "--resume-runtime-overrides",
        action="store_true",
        help=(
            "when resuming, explicitly apply only --rollout-workers, "
            "--rollout-worker-threads, and "
            "--central-inference-batch-wait-ms to the loaded config"
        ),
    )
    parser.add_argument(
        "--checkpoint-interval-agent-steps",
        type=int,
        default=0,
        help=(
            "atomically refresh --checkpoint after this many additional "
            "agent steps; zero saves only at normal completion"
        ),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("data/reports/ppo_smoke_training.json"),
    )
    args = parser.parse_args()
    positive_dimensions = (
        args.total_agent_steps,
        args.rollout_steps,
        args.sequence_length,
        args.minibatch_sequences,
        args.update_epochs,
        args.hidden_size,
        args.card_embedding_dim,
        args.model_dim,
        args.transformer_layers,
        args.attention_heads,
        args.feedforward_dim,
    )
    if any(value <= 0 for value in positive_dimensions):
        parser.error("training step counts and model dimensions must be positive")
    if args.rollout_workers <= 0:
        parser.error("rollout-workers must be positive")
    if args.rollout_worker_threads <= 0:
        parser.error("rollout-worker-threads must be positive")
    if args.central_inference_batch_wait_ms < 0:
        parser.error("central-inference-batch-wait-ms must be non-negative")
    if args.checkpoint_interval_agent_steps < 0:
        parser.error(
            "checkpoint-interval-agent-steps must be non-negative"
        )
    if args.resume_runtime_overrides and args.resume is None:
        parser.error("--resume-runtime-overrides requires --resume")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if args.entropy_coefficient < 0:
        parser.error("entropy-coefficient must be non-negative")
    if not 0 < args.clip_ratio < 1:
        parser.error("clip-ratio must be between zero and one")
    if args.resume is None and args.rollout_workers > 1 and (
        args.opponent_current_weight <= 0
        or args.opponent_random_weight > 0
        or args.opponent_fixed_weight > 0
    ):
        parser.error(
            "multiprocess rollout requires positive current weight, zero "
            "random/fixed weights, and optionally historical weight"
        )
    if args.resume is not None and (
        args.training_deck is not None
        or args.opponent_decks is not None
    ):
        parser.error(
            "--training-deck/--opponent-decks cannot be combined with "
            "--resume; the checkpoint owns its deck schedule"
        )
    if args.opponent_decks is not None and args.training_deck is None:
        parser.error("--opponent-decks requires --training-deck")
    training_class_ids = (
        tuple(ALL_CLASS_IDS)
        if args.classes is None
        else tuple(args.classes)
    )
    if args.training_deck is not None:
        fixed_deck = get_fixed_training_deck(args.training_deck)
        if (
            args.classes is not None
            and training_class_ids != (fixed_deck.class_id,)
        ):
            parser.error(
                f"--training-deck {args.training_deck} requires "
                f"--classes {fixed_deck.class_id}"
            )
        training_class_ids = (fixed_deck.class_id,)

    snapshot = WorkerAssetsSnapshot.build(CardRepository(args.database))
    if args.resume is not None:
        trainer = load_checkpoint(args.resume, snapshot, device=args.device)
        resume_config = trainer.config
        if args.resume_runtime_overrides:
            trainer.config = _resume_runtime_config(
                trainer.config,
                rollout_workers=args.rollout_workers,
                rollout_worker_threads=args.rollout_worker_threads,
                central_inference_batch_wait_ms=(
                    args.central_inference_batch_wait_ms
                ),
            )
        runtime_overrides = _runtime_override_report(
            resume_config,
            trainer.config,
        )
    else:
        trainer = PPOTrainer(
            snapshot,
            master_seed=args.master_seed,
            config=PPOConfig(
                rollout_steps=args.rollout_steps,
                sequence_length=args.sequence_length,
                minibatch_sequences=args.minibatch_sequences,
                update_epochs=args.update_epochs,
                hidden_size=args.hidden_size,
                card_embedding_dim=args.card_embedding_dim,
                policy_architecture=args.policy_architecture,
                observation_version=args.observation_version,
                model_dim=args.model_dim,
                transformer_layers=args.transformer_layers,
                attention_heads=args.attention_heads,
                feedforward_dim=args.feedforward_dim,
                learning_rate=args.learning_rate,
                entropy_coefficient=args.entropy_coefficient,
                clip_ratio=args.clip_ratio,
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
                rollout_worker_torch_threads=args.rollout_worker_threads,
                central_inference_batch_wait_seconds=(
                    args.central_inference_batch_wait_ms / 1000.0
                ),
                training_class_ids=training_class_ids,
                training_deck=args.training_deck,
                opponent_decks=(
                    ()
                    if args.opponent_decks is None
                    else tuple(args.opponent_decks)
                ),
                match_setup=args.match_setup,
            ),
            device=args.device,
        )
        runtime_overrides = {}
    if args.total_agent_steps <= trainer.agent_steps:
        parser.error(
            f"target {args.total_agent_steps} must exceed checkpoint progress "
            f"{trainer.agent_steps}"
        )
    atexit.register(trainer.close)

    started = time.perf_counter()
    starting_agent_steps = trainer.agent_steps
    metrics = []
    collect_timing_samples = []
    update_timing_samples = []
    historical_snapshots = []
    periodic_checkpoints = []
    last_checkpoint_steps = trainer.agent_steps
    history_directory = (
        args.checkpoint.parent / f"{args.checkpoint.stem}_history"
    )
    while trainer.agent_steps < args.total_agent_steps:
        records, bootstrap, _ = trainer.collect_rollout()
        update_metrics = trainer.update(records, bootstrap)
        metrics.append(update_metrics)
        collect_timing_samples.append(dict(trainer.last_collect_timing))
        update_timing_samples.append(dict(trainer.last_update_timing))
        progress = {
            **update_metrics,
            "timing": {
                "collect": trainer.last_collect_timing,
                "update": trainer.last_update_timing,
            },
        }
        print(json.dumps(progress, sort_keys=True), flush=True)
        if _periodic_checkpoint_due(
            last_checkpoint_steps=last_checkpoint_steps,
            current_steps=trainer.agent_steps,
            interval_steps=args.checkpoint_interval_agent_steps,
        ):
            periodic_path = _periodic_checkpoint_path(
                args.checkpoint,
                trainer.agent_steps,
            )
            save_checkpoint_atomic(periodic_path, trainer)
            last_checkpoint_steps = trainer.agent_steps
            periodic_checkpoints.append({
                "agent_steps": trainer.agent_steps,
                "path": str(periodic_path),
            })
            print(json.dumps({
                "periodic_checkpoint": str(periodic_path),
                "agent_steps": trainer.agent_steps,
            }, sort_keys=True), flush=True)
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
        "purpose": (
            "fixed-policy PPO training; policy strength requires a separate "
            "held-out mirrored evaluation"
        ),
        "torch_version": torch.__version__,
        "device": str(trainer.device),
        "policy_architecture": trainer.model.architecture,
        "model_parameters": sum(
            parameter.numel() for parameter in trainer.model.parameters()
        ),
        "master_seed": trainer.master_seed,
        "requested_agent_steps": args.total_agent_steps,
        "starting_agent_steps": starting_agent_steps,
        "trained_agent_steps": trainer.agent_steps - starting_agent_steps,
        "completed_agent_steps": trainer.agent_steps,
        "completed_episodes": trainer.completed_episodes,
        "updates": trainer.update_count,
        "elapsed_seconds": elapsed,
        "agent_steps_per_second": (
            (trainer.agent_steps - starting_agent_steps) / max(elapsed, 1e-12)
        ),
        "checkpoint": str(args.checkpoint),
        "resumed_from": None if args.resume is None else str(args.resume),
        "resume_runtime_overrides": runtime_overrides,
        "checkpoint_interval_agent_steps": (
            args.checkpoint_interval_agent_steps
        ),
        "periodic_checkpoint_steps": [
            checkpoint["agent_steps"]
            for checkpoint in periodic_checkpoints
        ],
        "periodic_checkpoints": periodic_checkpoints,
        "hyperparameters": trainer.hyperparameters(),
        "training_class_ids": list(trainer.config.training_class_ids),
        "training_deck": (
            None
            if trainer.fixed_training_deck is None
            else trainer.fixed_training_deck.manifest()
        ),
        "opponent_decks": [
            deck.manifest()
            for deck in trainer.fixed_opponent_decks
        ],
        "matchup_statistics": dict(trainer.matchup_statistics),
        "final_metrics": metrics[-1],
        "timing": training_timing_report(
            collect_timing_samples,
            update_timing_samples,
        ),
        "opponent_pool": trainer.opponent_pool.state_dict(),
        "opponent_assignments": list(trainer.opponent_assignments),
        "historical_snapshots_created": historical_snapshots,
        "versions": {
            "observation_version": trainer.config.observation_version,
            "catalog_sha256": snapshot.catalog.catalog_sha256,
            "card_vocabulary_sha256": snapshot.catalog.card_vocabulary_sha256,
            "rulebook_sha256": snapshot.rulebook_sha256,
            "class_schedule_version": CLASS_SCHEDULE_VERSION,
            "deck_matchup_schedule_version": (
                DECK_MATCHUP_SCHEDULE_VERSION
            ),
            "match_setup": trainer.config.match_setup,
            "training_deck_sha256": (
                None
                if trainer.fixed_training_deck is None
                else trainer.fixed_training_deck.sha256
            ),
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
