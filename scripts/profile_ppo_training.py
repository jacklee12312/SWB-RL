from __future__ import annotations

import argparse
import atexit
import json
import os
import platform
import time
from dataclasses import replace
from pathlib import Path

import torch

from swb.db.repository import CardRepository
from swb.rl.checkpoint import load_checkpoint
from swb.rl.profiling import training_timing_report
from swb.rl.runtime import WorkerAssetsSnapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile PPO rollout and learner stages without writing a checkpoint"
        )
    )
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--additional-agent-steps", type=int, default=100_000)
    parser.add_argument("--exclude-warmup-updates", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-worker-threads", type=int)
    parser.add_argument(
        "--central-inference-batch-wait-ms",
        type=float,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/ppo_training_profile.json"),
    )
    args = parser.parse_args()
    if args.additional_agent_steps <= 0:
        parser.error("--additional-agent-steps must be positive")
    if args.exclude_warmup_updates < 0:
        parser.error("--exclude-warmup-updates must be non-negative")
    if (
        args.rollout_worker_threads is not None
        and args.rollout_worker_threads <= 0
    ):
        parser.error("--rollout-worker-threads must be positive")
    if (
        args.central_inference_batch_wait_ms is not None
        and args.central_inference_batch_wait_ms < 0
    ):
        parser.error(
            "--central-inference-batch-wait-ms must be non-negative"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    snapshot = WorkerAssetsSnapshot.build(CardRepository(args.database))
    checkpoint_stat = args.checkpoint.stat()
    trainer = load_checkpoint(args.checkpoint, snapshot, device=args.device)
    runtime_overrides = {}
    if args.rollout_worker_threads is not None:
        runtime_overrides["rollout_worker_torch_threads"] = (
            args.rollout_worker_threads
        )
    if args.central_inference_batch_wait_ms is not None:
        runtime_overrides["central_inference_batch_wait_seconds"] = (
            args.central_inference_batch_wait_ms / 1000.0
        )
    if runtime_overrides:
        trainer.config = replace(trainer.config, **runtime_overrides)
    atexit.register(trainer.close)
    starting_agent_steps = trainer.agent_steps
    starting_episodes = trainer.completed_episodes
    target_agent_steps = starting_agent_steps + args.additional_agent_steps
    iterations: list[dict[str, object]] = []
    collect_samples: list[dict[str, float]] = []
    update_samples: list[dict[str, float]] = []
    started = time.perf_counter()
    while trainer.agent_steps < target_agent_steps:
        iteration_started = time.perf_counter()
        before_steps = trainer.agent_steps
        before_episodes = trainer.completed_episodes
        records, bootstrap, _ = trainer.collect_rollout()
        metrics = trainer.update(records, bootstrap)
        collect_timing = dict(trainer.last_collect_timing)
        update_timing = dict(trainer.last_update_timing)
        collect_samples.append(collect_timing)
        update_samples.append(update_timing)
        iteration = {
            "index": len(iterations),
            "agent_steps": trainer.agent_steps - before_steps,
            "episodes": trainer.completed_episodes - before_episodes,
            "elapsed_seconds": time.perf_counter() - iteration_started,
            "collect": collect_timing,
            "update": update_timing,
            "metrics": metrics,
        }
        iterations.append(iteration)
        print(json.dumps({
            "profile_update": len(iterations),
            "completed_agent_steps": trainer.agent_steps - starting_agent_steps,
            "target_additional_agent_steps": args.additional_agent_steps,
            "rollout_seconds": collect_timing["collect_total_seconds"],
            "update_seconds": update_timing["update_total_seconds"],
        }, sort_keys=True), flush=True)
    elapsed = time.perf_counter() - started
    warmup = min(args.exclude_warmup_updates, len(iterations))
    steady_collect = collect_samples[warmup:]
    steady_update = update_samples[warmup:]
    checkpoint_after = args.checkpoint.stat()
    report = {
        "schema_version": 1,
        "purpose": (
            "PPO pipeline performance diagnosis; the loaded checkpoint is "
            "updated only in memory and is never saved"
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_unchanged": (
            checkpoint_stat.st_size == checkpoint_after.st_size
            and checkpoint_stat.st_mtime_ns == checkpoint_after.st_mtime_ns
        ),
        "device": str(trainer.device),
        "runtime_rollout_configuration": {
            "rollout_workers": trainer.config.rollout_workers,
            "worker_torch_threads": (
                trainer.config.rollout_worker_torch_threads
            ),
            "central_inference_batch_wait_seconds": (
                trainer.config.central_inference_batch_wait_seconds
            ),
        },
        "hardware": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "gpu": (
                torch.cuda.get_device_name(trainer.device)
                if trainer.device.type == "cuda"
                else None
            ),
        },
        "start": {
            "agent_steps": starting_agent_steps,
            "completed_episodes": starting_episodes,
        },
        "result": {
            "requested_additional_agent_steps": args.additional_agent_steps,
            "completed_additional_agent_steps": (
                trainer.agent_steps - starting_agent_steps
            ),
            "completed_additional_episodes": (
                trainer.completed_episodes - starting_episodes
            ),
            "updates": len(iterations),
            "elapsed_seconds": elapsed,
            "agent_steps_per_second": (
                (trainer.agent_steps - starting_agent_steps)
                / max(elapsed, 1e-12)
            ),
        },
        "all_updates": training_timing_report(
            collect_samples,
            update_samples,
        ),
        "steady_state": {
            "excluded_warmup_updates": warmup,
            **training_timing_report(steady_collect, steady_update),
        },
        "iterations": iterations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "profile_report": str(args.output),
        "agent_steps_per_second": report["result"]["agent_steps_per_second"],
        "checkpoint_unchanged": report["checkpoint_unchanged"],
    }, ensure_ascii=False, sort_keys=True))
    trainer.close()
    atexit.unregister(trainer.close)


if __name__ == "__main__":
    main()
