from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import torch

from scripts.training_speed_baseline import SystemMonitor, _system_manifest
from swb.db.repository import CardRepository
from swb.rl.checkpoint import load_checkpoint
from swb.rl.profiling import training_timing_report
from swb.rl.runtime import WorkerAssetsSnapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _system_monitor_summary(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not samples:
        return {
            "sample_count": 0,
            "passed": False,
        }
    pagefile_used = [
        int(sample["pagefile_used_bytes"]) for sample in samples
    ]
    pagefile_sin = [
        int(sample["pagefile_sin_bytes"]) for sample in samples
    ]
    pagefile_sout = [
        int(sample["pagefile_sout_bytes"]) for sample in samples
    ]
    ram_used = [int(sample["ram_used_bytes"]) for sample in samples]
    cpu_total = [
        float(sample["cpu_total_percent"]) for sample in samples
    ]
    gpu_samples = [
        sample["gpu"]
        for sample in samples
        if isinstance(sample.get("gpu"), Mapping)
    ]
    gpu_memory = [
        float(sample["memory_used_mib"])
        for sample in gpu_samples
        if sample.get("memory_used_mib") is not None
    ]
    return {
        "sample_count": len(samples),
        "elapsed_seconds": float(samples[-1]["elapsed_seconds"]),
        "cpu_total_median_percent": statistics.median(cpu_total),
        "cpu_single_core_peak_percent": max(
            float(value)
            for sample in samples
            for value in sample["cpu_per_core_percent"]
        ),
        "ram_used_first_bytes": ram_used[0],
        "ram_used_last_bytes": ram_used[-1],
        "ram_used_peak_bytes": max(ram_used),
        "pagefile_used_first_bytes": pagefile_used[0],
        "pagefile_used_last_bytes": pagefile_used[-1],
        "pagefile_used_peak_bytes": max(pagefile_used),
        "pagefile_used_change_bytes": (
            pagefile_used[-1] - pagefile_used[0]
        ),
        "pagefile_sin_change_bytes": pagefile_sin[-1] - pagefile_sin[0],
        "pagefile_sout_change_bytes": (
            pagefile_sout[-1] - pagefile_sout[0]
        ),
        "gpu_memory_peak_mib": (
            max(gpu_memory) if gpu_memory else None
        ),
        "no_page_in_or_page_out": (
            pagefile_sin[-1] == pagefile_sin[0]
            and pagefile_sout[-1] == pagefile_sout[0]
        ),
        "passed": True,
    }


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
    parser.add_argument("--rollout-workers", type=int)
    parser.add_argument("--rollout-worker-threads", type=int)
    parser.add_argument(
        "--central-inference-batch-wait-ms",
        type=float,
    )
    parser.add_argument(
        "--profile-ipc-timing",
        action="store_true",
        help=(
            "send policy requests through the instrumented serialized "
            "envelope and report request serialization/send/wait timing"
        ),
    )
    parser.add_argument(
        "--profile-central-timing",
        action="store_true",
        help=(
            "synchronize and split central queue, batching, transfer, model, "
            "distribution, sampling, and result-dispatch timing"
        ),
    )
    parser.add_argument(
        "--profile-learner-timing",
        action="store_true",
        help=(
            "record learner padding, H2D, CUDA component events, effective "
            "tokens, and explicit/implicit synchronization timing"
        ),
    )
    parser.add_argument(
        "--monitor-system",
        action="store_true",
        help=(
            "sample CPU, RAM, pagefile I/O, and NVIDIA GPU state while "
            "the profile runs"
        ),
    )
    parser.add_argument(
        "--monitor-interval-seconds",
        type=float,
        default=0.5,
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
    if args.monitor_interval_seconds <= 0:
        parser.error("--monitor-interval-seconds must be positive")
    if args.rollout_workers is not None and args.rollout_workers <= 0:
        parser.error("--rollout-workers must be positive")
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
    checkpoint_sha256 = _sha256(args.checkpoint)
    trainer = load_checkpoint(args.checkpoint, snapshot, device=args.device)
    runtime_overrides = {}
    if args.rollout_workers is not None:
        runtime_overrides["rollout_workers"] = args.rollout_workers
    if args.rollout_worker_threads is not None:
        runtime_overrides["rollout_worker_torch_threads"] = (
            args.rollout_worker_threads
        )
    if args.central_inference_batch_wait_ms is not None:
        runtime_overrides["central_inference_batch_wait_seconds"] = (
            args.central_inference_batch_wait_ms / 1000.0
        )
    if args.profile_ipc_timing:
        runtime_overrides["profile_ipc_timing"] = True
    if args.profile_central_timing:
        runtime_overrides["profile_central_timing"] = True
    if args.profile_learner_timing:
        runtime_overrides["profile_learner_timing"] = True
    if runtime_overrides:
        trainer.config = replace(trainer.config, **runtime_overrides)
    atexit.register(trainer.close)
    starting_agent_steps = trainer.agent_steps
    starting_episodes = trainer.completed_episodes
    target_agent_steps = starting_agent_steps + args.additional_agent_steps
    iterations: list[dict[str, object]] = []
    collect_samples: list[dict[str, float]] = []
    update_samples: list[dict[str, float]] = []
    system_monitor = (
        SystemMonitor(args.monitor_interval_seconds)
        if args.monitor_system
        else None
    )
    system_before = _system_manifest() if args.monitor_system else None
    if system_monitor is not None:
        system_monitor.start()
    started = time.perf_counter()
    try:
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
                "completed_agent_steps": (
                    trainer.agent_steps - starting_agent_steps
                ),
                "target_additional_agent_steps": (
                    args.additional_agent_steps
                ),
                "rollout_seconds": (
                    collect_timing["collect_total_seconds"]
                ),
                "update_seconds": (
                    update_timing["update_total_seconds"]
                ),
            }, sort_keys=True), flush=True)
    finally:
        if system_monitor is not None:
            system_monitor.stop()
    elapsed = time.perf_counter() - started
    warmup = min(args.exclude_warmup_updates, len(iterations))
    steady_collect = collect_samples[warmup:]
    steady_update = update_samples[warmup:]
    checkpoint_after = args.checkpoint.stat()
    checkpoint_sha256_after = _sha256(args.checkpoint)
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
            and checkpoint_sha256 == checkpoint_sha256_after
        ),
        "checkpoint_sha256_before": checkpoint_sha256,
        "checkpoint_sha256_after": checkpoint_sha256_after,
        "device": str(trainer.device),
        "runtime_rollout_configuration": {
            "rollout_workers": trainer.config.rollout_workers,
            "worker_torch_threads": (
                trainer.config.rollout_worker_torch_threads
            ),
            "central_inference_batch_wait_seconds": (
                trainer.config.central_inference_batch_wait_seconds
            ),
            "profile_ipc_timing": trainer.config.profile_ipc_timing,
            "profile_central_timing": (
                trainer.config.profile_central_timing
            ),
            "profile_learner_timing": (
                trainer.config.profile_learner_timing
            ),
        },
        "ipc_timing_methodology": {
            "enabled": trainer.config.profile_ipc_timing,
            "request_serialization": (
                "ForkingPickler serialization of the exact policy request "
                "tuple before queue submission"
            ),
            "request_send": (
                "elapsed from serialized-envelope queue submission timestamp "
                "until the central process dequeues the envelope"
            ),
            "response_wait": (
                "elapsed from central request dequeue until the worker "
                "receives the matching policy action"
            ),
            "aggregation": (
                "seconds are summed across concurrent workers; per-request "
                "means and accounted fraction are the comparable diagnostics"
            ),
            "normal_training_path": (
                "unchanged unless --profile-ipc-timing is enabled"
            ),
        },
        "central_timing_methodology": {
            "enabled": trainer.config.profile_central_timing,
            "queue_to_batch": (
                "per-request elapsed from central queue dequeue to the "
                "timestamp at which its inference batch is closed"
            ),
            "cpu_input": (
                "separate NumPy stack allocation, CPU tensor views/dtype "
                "conversion, copied bytes, and hidden-state concatenation"
            ),
            "cuda_components": (
                "CUDA events split H2D, Transformer, GRU, policy/value heads, "
                "masked distribution, and D2H; diagnostic synchronization is "
                "enabled only by --profile-central-timing"
            ),
            "gpu_busy_vs_worker_wait": (
                "GPU busy is the sum of profiled CUDA-event stages; worker "
                "wait is central blocking for worker messages plus configured "
                "batch-formation wait. The ratio is diagnostic, not system "
                "GPU utilization."
            ),
        },
        "learner_timing_methodology": {
            "enabled": trainer.config.profile_learner_timing,
            "trajectory_and_padding": (
                "advantages/returns, recurrent chunks, NumPy padding, CPU "
                "tensor construction, effective tokens, and padding slots "
                "are measured separately"
            ),
            "cuda_components": (
                "CUDA events split H2D, model forward, PPO loss, backward, "
                "gradient clipping, and optimizer work without adding a "
                "synchronize between those components"
            ),
            "synchronization": (
                "existing explicit torch.cuda.synchronize calls and implicit "
                "host waits from finite checks, parameter validation, and "
                "metric extraction are reported separately; event time is "
                "not added to synchronization wait"
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
        "system_monitor": {
            "enabled": args.monitor_system,
            "interval_seconds": args.monitor_interval_seconds,
            "system_before": system_before,
            "system_after": (
                _system_manifest() if args.monitor_system else None
            ),
            "summary": (
                _system_monitor_summary(system_monitor.samples)
                if system_monitor is not None
                else None
            ),
            "samples": (
                system_monitor.samples
                if system_monitor is not None
                else []
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
