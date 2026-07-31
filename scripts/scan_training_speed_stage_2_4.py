from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from scripts.training_speed_baseline import (
    ROOT,
    SystemMonitor,
    _system_manifest,
)


DEFAULT_CHECKPOINT = Path(
    "data/checkpoints/training_speed/frozen_v4_1_seed_20260801_500k.pt"
)
DEFAULT_BASELINE = Path(
    "data/reports/training_speed/baseline_summary.json"
)
DEFAULT_RUN_ROOT = Path(
    "data/reports/training_speed/stage_2_4_runs"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/stage_2_4_scan.json"
)
BASELINE_WORKERS = 4
BASELINE_THREADS = 2
BASELINE_WAIT_MS = 0.5
WAIT_SCAN_MS = (0.0, 0.1, 0.25, 0.5, 1.0)
WORKER_SCAN = (2, 3, 4, 5, 6)
THREAD_SCAN = (1, 2, 4)
FORMAL_RUNS = 3
WARMUP_UPDATES = 2
MEASURED_AGENT_STEPS = 6_144
MINIMUM_STEADY_UPDATES = 3
WINNER_RELATIVE_GAIN = 0.05


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, object]:
    resolved = _repo_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _repo_path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _configuration(
    config_id: str,
    dimension: str,
    *,
    workers: int = BASELINE_WORKERS,
    threads: int = BASELINE_THREADS,
    wait_ms: float = BASELINE_WAIT_MS,
) -> dict[str, object]:
    return {
        "id": config_id,
        "dimension": dimension,
        "rollout_workers": workers,
        "worker_torch_threads": threads,
        "central_inference_batch_wait_ms": wait_ms,
    }


def primary_configurations() -> list[dict[str, object]]:
    configs = [_configuration("baseline", "baseline")]
    configs.extend(
        _configuration(
            f"wait_{str(wait_ms).replace('.', '_')}_ms",
            "batch_wait",
            wait_ms=wait_ms,
        )
        for wait_ms in WAIT_SCAN_MS
        if wait_ms != BASELINE_WAIT_MS
    )
    configs.extend(
        _configuration(
            f"workers_{workers}",
            "rollout_workers",
            workers=workers,
        )
        for workers in WORKER_SCAN
        if workers != BASELINE_WORKERS
    )
    configs.extend(
        _configuration(
            f"threads_{threads}",
            "worker_threads",
            threads=threads,
        )
        for threads in THREAD_SCAN
        if threads != BASELINE_THREADS
    )
    return configs


def configuration_by_id(config_id: str) -> dict[str, object]:
    configs = {
        str(config["id"]): config
        for config in primary_configurations()
    }
    try:
        return configs[config_id]
    except KeyError as error:
        raise ValueError(f"unknown stage 2.4 configuration: {config_id}") from error


def _field(
    row: Mapping[str, object],
    name: str,
) -> float:
    collect = row.get("collect")
    if not isinstance(collect, Mapping):
        return 0.0
    return float(collect.get(name, 0.0))


def _weighted_episode_lengths(
    rows: Sequence[Mapping[str, object]],
) -> list[float]:
    lengths: list[float] = []
    prefix = "worker_episode_steps_"
    suffix = "_count"
    for row in rows:
        collect = row.get("collect")
        if not isinstance(collect, Mapping):
            continue
        for name, value in collect.items():
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            middle = name[len(prefix):-len(suffix)]
            if not middle.isdigit():
                continue
            lengths.extend([float(middle)] * int(float(value)))
    return lengths


def _batch_histogram(
    rows: Sequence[Mapping[str, object]],
) -> Counter[int]:
    histogram: Counter[int] = Counter()
    prefix = "central_batch_size_"
    suffix = "_count"
    for row in rows:
        collect = row.get("collect")
        if not isinstance(collect, Mapping):
            continue
        for name, value in collect.items():
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            middle = name[len(prefix):-len(suffix)]
            if middle.isdigit():
                histogram[int(middle)] += int(float(value))
    return histogram


def _histogram_percentile(
    histogram: Mapping[int, int],
    fraction: float,
) -> float:
    count = sum(histogram.values())
    if count == 0:
        return 0.0
    target = max(1, math.ceil(fraction * count))
    seen = 0
    for value, frequency in sorted(histogram.items()):
        seen += frequency
        if seen >= target:
            return float(value)
    raise AssertionError("non-empty histogram did not reach percentile")


def _system_summary(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    stable_samples = list(samples[1:] if len(samples) > 1 else samples)
    gpu_samples = [
        sample["gpu"]
        for sample in stable_samples
        if isinstance(sample.get("gpu"), Mapping)
        and "utilization_percent" in sample["gpu"]
    ]
    gpu_utilization = [
        float(sample["utilization_percent"])
        for sample in gpu_samples
    ]
    return {
        "sample_count": len(stable_samples),
        "cpu_total_median_percent": (
            statistics.median(
                float(sample["cpu_total_percent"])
                for sample in stable_samples
            )
            if stable_samples else None
        ),
        "cpu_single_core_peak_percent": (
            max(
                max(float(value) for value in sample["cpu_per_core_percent"])
                for sample in stable_samples
                if sample.get("cpu_per_core_percent")
            )
            if stable_samples else None
        ),
        "ram_used_peak_bytes": (
            max(int(sample["ram_used_bytes"]) for sample in stable_samples)
            if stable_samples else None
        ),
        "pagefile_used_peak_bytes": (
            max(int(sample["pagefile_used_bytes"]) for sample in stable_samples)
            if stable_samples else None
        ),
        "gpu_sample_count": len(gpu_samples),
        "gpu_utilization_median_percent": (
            statistics.median(gpu_utilization)
            if gpu_utilization else None
        ),
        "gpu_utilization_p95_percent": (
            _nearest_rank(gpu_utilization, 0.95)
            if gpu_utilization else None
        ),
        "gpu_idle_sample_fraction_at_or_below_5_percent": (
            sum(value <= 5.0 for value in gpu_utilization)
            / len(gpu_utilization)
            if gpu_utilization else None
        ),
        "gpu_memory_peak_mib": (
            max(float(sample["memory_used_mib"]) for sample in gpu_samples)
            if gpu_samples else None
        ),
    }


def compact_profile(
    profile: Mapping[str, object],
    *,
    measured_agent_steps: int,
    warmup_updates: int,
    system_samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    iterations = profile.get("iterations")
    if not isinstance(iterations, list):
        raise ValueError("profile iterations must be a list")
    if len(iterations) < warmup_updates + MINIMUM_STEADY_UPDATES:
        raise ValueError(
            "formal run requires at least three steady updates after warmup"
        )
    steady_rows = iterations[warmup_updates:]
    if not all(isinstance(row, Mapping) for row in steady_rows):
        raise ValueError("profile iteration must be an object")
    typed_rows = list(steady_rows)
    steps = sum(int(row["agent_steps"]) for row in typed_rows)
    elapsed = sum(float(row["elapsed_seconds"]) for row in typed_rows)
    if steps < measured_agent_steps:
        raise ValueError(
            f"steady measurement has {steps} steps, expected "
            f"{measured_agent_steps}"
        )

    requests = sum(_field(row, "central_inference_requests") for row in typed_rows)
    batches = sum(_field(row, "central_inference_batches") for row in typed_rows)
    capacity = sum(
        _field(row, "central_batch_capacity_slots") for row in typed_rows
    )
    empty_slots = sum(
        _field(row, "central_batch_empty_slots") for row in typed_rows
    )
    batch_histogram = _batch_histogram(typed_rows)
    episode_lengths = _weighted_episode_lengths(typed_rows)
    collect_seconds = [
        _field(row, "collect_total_seconds") for row in typed_rows
    ]
    update_seconds = [
        float(row["update"]["update_total_seconds"])
        for row in typed_rows
    ]
    per_update_speeds = [
        int(row["agent_steps"]) / max(float(row["elapsed_seconds"]), 1e-12)
        for row in typed_rows
    ]
    return {
        "excluded_warmup_updates": warmup_updates,
        "steady_update_count": len(typed_rows),
        "agent_steps": steps,
        "elapsed_seconds": elapsed,
        "agent_steps_per_second": steps / elapsed,
        "per_update_steps_per_second": per_update_speeds,
        "collect_p95_seconds": _nearest_rank(collect_seconds, 0.95),
        "update_p95_seconds": _nearest_rank(update_seconds, 0.95),
        "batching": {
            "inference_requests": requests,
            "inference_batches": batches,
            "mean_batch_size": requests / max(batches, 1.0),
            "p50_batch_size": _histogram_percentile(batch_histogram, 0.50),
            "p95_batch_size": _histogram_percentile(batch_histogram, 0.95),
            "empty_slot_fraction": empty_slots / max(capacity, 1.0),
            "histogram": {
                str(size): count
                for size, count in sorted(batch_histogram.items())
            },
            "configured_wait_total_seconds": sum(
                _field(row, "central_batch_wait_seconds")
                for row in typed_rows
            ),
            "worker_message_wait_total_seconds": sum(
                _field(row, "central_worker_message_wait_seconds")
                for row in typed_rows
            ),
        },
        "episode_length": {
            "sample_count": len(episode_lengths),
            "mean": (
                statistics.fmean(episode_lengths)
                if episode_lengths else 0.0
            ),
            "p50": _nearest_rank(episode_lengths, 0.50),
            "p95": _nearest_rank(episode_lengths, 0.95),
            "maximum": max(episode_lengths, default=0.0),
            "long_episode_count": sum(
                _field(row, "worker_long_episode_count")
                for row in typed_rows
            ),
            "truncated_episode_count": sum(
                _field(row, "worker_truncated_episode_count")
                for row in typed_rows
            ),
        },
        "observation": {
            "decision_construction_seconds": sum(
                _field(
                    row,
                    "worker_decision_observation_construction_seconds",
                )
                for row in typed_rows
            ),
            "step_construction_seconds": sum(
                _field(row, "worker_step_observation_construction_seconds")
                for row in typed_rows
            ),
            "bootstrap_construction_seconds": sum(
                _field(
                    row,
                    "worker_bootstrap_observation_construction_seconds",
                )
                for row in typed_rows
            ),
            "total_construction_seconds": sum(
                _field(row, "worker_observation_construction_seconds")
                for row in typed_rows
            ),
        },
        "system": _system_summary(system_samples),
        "abnormal_exit_count": 0,
    }


def _run_path(
    run_root: Path,
    config_id: str,
    run_index: int,
) -> Path:
    return run_root / f"{config_id}_run_{run_index}.json"


def run_configuration(
    *,
    config: Mapping[str, object],
    run_index: int,
    checkpoint: Path,
    output: Path,
    measured_agent_steps: int,
    warmup_updates: int,
    monitor_interval_seconds: float,
    force: bool,
) -> dict[str, object]:
    output = _repo_path(output)
    if output.exists() and not force:
        return _json(output)
    checkpoint = _repo_path(checkpoint)
    checkpoint_stat = checkpoint.stat()
    checkpoint_hash = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rollout_steps = int(payload["trainer"]["config"]["rollout_steps"])
    requested_steps = measured_agent_steps + warmup_updates * rollout_steps
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(f".{output.stem}.raw.json")
    log_path = output.with_suffix(".log")
    command = [
        sys.executable,
        "-m",
        "scripts.profile_ppo_training",
        "--checkpoint",
        str(checkpoint),
        "--additional-agent-steps",
        str(requested_steps),
        "--exclude-warmup-updates",
        str(warmup_updates),
        "--device",
        "cuda",
        "--rollout-workers",
        str(config["rollout_workers"]),
        "--rollout-worker-threads",
        str(config["worker_torch_threads"]),
        "--central-inference-batch-wait-ms",
        str(config["central_inference_batch_wait_ms"]),
        "--output",
        str(raw_output),
    ]
    system_before = _system_manifest()
    monitor = SystemMonitor(monitor_interval_seconds)
    monitor.start()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    finally:
        monitor.stop()
    if result.returncode != 0:
        raise RuntimeError(
            f"profile failed with exit code {result.returncode}; "
            f"see {log_path}"
        )
    profile = _json(raw_output)
    raw_output.unlink()
    checkpoint_after = checkpoint.stat()
    unchanged = (
        checkpoint_stat.st_size == checkpoint_after.st_size
        and checkpoint_stat.st_mtime_ns == checkpoint_after.st_mtime_ns
        and checkpoint_hash == _sha256(checkpoint)
        and bool(profile["checkpoint_unchanged"])
    )
    if not unchanged:
        raise RuntimeError("fixed performance checkpoint changed during scan")
    runtime = profile["runtime_rollout_configuration"]
    expected_runtime = {
        "rollout_workers": int(config["rollout_workers"]),
        "worker_torch_threads": int(config["worker_torch_threads"]),
        "central_inference_batch_wait_seconds": (
            float(config["central_inference_batch_wait_ms"]) / 1000.0
        ),
    }
    for name, expected in expected_runtime.items():
        if runtime[name] != expected:
            raise ValueError(
                f"runtime {name} was {runtime[name]!r}, expected {expected!r}"
            )
    compact = {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_4_run",
        "checklist_section": "2.4A",
        "configuration": dict(config),
        "run_index": run_index,
        "started_utc": started_utc,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_unchanged": unchanged,
        "runtime_rollout_configuration": runtime,
        "profiling_switches_disabled": all(
            not bool(runtime[name])
            for name in (
                "profile_ipc_timing",
                "profile_central_timing",
                "profile_learner_timing",
            )
        ),
        "system_before": system_before,
        "system_after": _system_manifest(),
        "monitor_interval_seconds": monitor_interval_seconds,
        "system_samples": monitor.samples,
        "measurement": compact_profile(
            profile,
            measured_agent_steps=measured_agent_steps,
            warmup_updates=warmup_updates,
            system_samples=monitor.samples,
        ),
        "profile_result": profile["result"],
        "profile_command": command,
        "profile_log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
    }
    output.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return compact


def _median_metric(
    reports: Sequence[Mapping[str, object]],
    *path: str,
) -> float:
    values = []
    for report in reports:
        value: object = report
        for name in path:
            if not isinstance(value, Mapping):
                raise ValueError(f"metric path {'/'.join(path)} is malformed")
            value = value[name]
        values.append(float(value))
    return statistics.median(values)


def _coverage(
    configs: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    rows = list(configs)
    waits = sorted({
        float(config["central_inference_batch_wait_ms"])
        for config in rows
        if config["dimension"] in {"baseline", "batch_wait"}
    })
    workers = sorted({
        int(config["rollout_workers"])
        for config in rows
        if config["dimension"] in {"baseline", "rollout_workers"}
    })
    threads = sorted({
        int(config["worker_torch_threads"])
        for config in rows
        if config["dimension"] in {"baseline", "worker_threads"}
    })
    return {
        "batch_wait_ms": waits,
        "rollout_workers": workers,
        "worker_torch_threads": threads,
        "passed": (
            waits == list(WAIT_SCAN_MS)
            and workers == list(WORKER_SCAN)
            and threads == list(THREAD_SCAN)
        ),
    }


def summarize_scan(
    reports: Sequence[Mapping[str, object]],
    baseline: Mapping[str, object],
    *,
    source_paths: Sequence[str],
    source_sha256: Sequence[str],
) -> dict[str, object]:
    expected_configs = {
        str(config["id"]): config
        for config in primary_configurations()
    }
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for report in reports:
        config = report["configuration"]
        if not isinstance(config, Mapping):
            raise ValueError("run configuration must be an object")
        config_id = str(config["id"])
        if config_id not in expected_configs:
            raise ValueError(f"unexpected stage 2.4 config: {config_id}")
        if dict(config) != expected_configs[config_id]:
            raise ValueError(f"configuration drift for {config_id}")
        grouped.setdefault(config_id, []).append(report)
    if set(grouped) != set(expected_configs):
        missing = sorted(set(expected_configs) - set(grouped))
        raise ValueError(f"stage 2.4 scan is missing configs: {missing}")

    baseline_v4 = baseline["observations"]["v4.1"]
    baseline_speed = baseline_v4["agent_steps_per_second"]
    baseline_median = float(baseline_speed["median"])
    baseline_relative_range = (
        float(baseline_speed["range"]) / max(baseline_median, 1e-12)
    )
    summaries: dict[str, dict[str, object]] = {}
    fixed_hashes = set()
    all_monitored = True
    no_abnormal_exits = True
    for config_id, rows in grouped.items():
        if len(rows) != FORMAL_RUNS:
            raise ValueError(f"{config_id} requires exactly three runs")
        run_indexes = sorted(int(row["run_index"]) for row in rows)
        if run_indexes != list(range(1, FORMAL_RUNS + 1)):
            raise ValueError(f"{config_id} run indexes must be 1, 2, 3")
        speeds = [
            float(row["measurement"]["agent_steps_per_second"])
            for row in rows
        ]
        median_speed = statistics.median(speeds)
        relative_gain = (
            median_speed - baseline_median
        ) / max(baseline_median, 1e-12)
        above_variability = relative_gain > baseline_relative_range
        stable_winner = (
            relative_gain >= WINNER_RELATIVE_GAIN
            and above_variability
        )
        fixed_hashes.update(str(row["checkpoint_sha256"]) for row in rows)
        all_monitored = all_monitored and all(
            int(row["measurement"]["system"]["sample_count"]) > 0
            and int(row["measurement"]["system"]["gpu_sample_count"]) > 0
            for row in rows
        )
        no_abnormal_exits = no_abnormal_exits and all(
            int(row["measurement"]["abnormal_exit_count"]) == 0
            for row in rows
        )
        summaries[config_id] = {
            "configuration": expected_configs[config_id],
            "run_count": len(rows),
            "agent_steps_per_second": {
                "runs": speeds,
                "median": median_speed,
                "minimum": min(speeds),
                "maximum": max(speeds),
                "range": max(speeds) - min(speeds),
                "relative_gain_vs_frozen_baseline": relative_gain,
                "above_frozen_baseline_three_run_variability": (
                    above_variability
                ),
                "stable_five_percent_winner": stable_winner,
            },
            "collect_p95_seconds_median": _median_metric(
                rows, "measurement", "collect_p95_seconds"
            ),
            "update_p95_seconds_median": _median_metric(
                rows, "measurement", "update_p95_seconds"
            ),
            "batching": {
                name: _median_metric(rows, "measurement", "batching", name)
                for name in (
                    "mean_batch_size",
                    "p50_batch_size",
                    "p95_batch_size",
                    "empty_slot_fraction",
                    "configured_wait_total_seconds",
                    "worker_message_wait_total_seconds",
                )
            },
            "episode_length": {
                name: _median_metric(rows, "measurement", "episode_length", name)
                for name in (
                    "mean",
                    "p50",
                    "p95",
                    "maximum",
                    "long_episode_count",
                    "truncated_episode_count",
                )
            },
            "system": {
                name: _median_metric(rows, "measurement", "system", name)
                for name in (
                    "cpu_total_median_percent",
                    "cpu_single_core_peak_percent",
                    "ram_used_peak_bytes",
                    "pagefile_used_peak_bytes",
                    "gpu_utilization_median_percent",
                    "gpu_utilization_p95_percent",
                    "gpu_idle_sample_fraction_at_or_below_5_percent",
                    "gpu_memory_peak_mib",
                )
            },
        }

    ranked = sorted(
        summaries,
        key=lambda config_id: summaries[config_id][
            "agent_steps_per_second"
        ]["median"],
        reverse=True,
    )
    stable_winners = [
        config_id
        for config_id in ranked
        if config_id != "baseline"
        and summaries[config_id]["agent_steps_per_second"][
            "stable_five_percent_winner"
        ]
    ]
    wait_rows = [
        summaries[str(config["id"])]
        for config in expected_configs.values()
        if config["dimension"] in {"baseline", "batch_wait"}
    ]
    max_wait_batch = max(
        float(row["batching"]["mean_batch_size"]) for row in wait_rows
    )
    baseline_scan_batch = float(
        summaries["baseline"]["batching"]["mean_batch_size"]
    )
    batch_growth = (
        max_wait_batch - baseline_scan_batch
    ) / max(baseline_scan_batch, 1e-12)
    wait_conclusion = (
        "request_arrival_insufficient"
        if batch_growth < WINNER_RELATIVE_GAIN
        else "batch_window_materially_limits_batch_formation"
    )
    coverage = _coverage(expected_configs.values())
    passed = (
        bool(coverage["passed"])
        and len(fixed_hashes) == 1
        and all_monitored
        and no_abnormal_exits
    )
    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_4_scan",
        "checklist_section": "2.4",
        "methodology": {
            "fixed_checkpoint": str(DEFAULT_CHECKPOINT).replace("\\", "/"),
            "profiling_switches": "disabled",
            "formal_runs_per_configuration": FORMAL_RUNS,
            "warmup_updates_excluded": WARMUP_UPDATES,
            "minimum_steady_updates_per_run": MINIMUM_STEADY_UPDATES,
            "minimum_measured_agent_steps_per_run": MEASURED_AGENT_STEPS,
            "single_variable_primary_scan": True,
            "baseline_median_agent_steps_per_second": baseline_median,
            "baseline_three_run_relative_range": baseline_relative_range,
            "stable_winner_threshold": WINNER_RELATIVE_GAIN,
        },
        "coverage": coverage,
        "configurations": summaries,
        "ranking": ranked,
        "diagnosis": {
            "batch_wait_scan_mean_batch_growth_vs_scan_baseline": batch_growth,
            "batch_formation_limit": wait_conclusion,
            "best_configuration": ranked[0],
            "stable_five_percent_winners": stable_winners,
            "stage_2_4_b_gate": {
                "enter": bool(stable_winners),
                "reason": (
                    "at least one primary candidate exceeded both the five "
                    "percent threshold and frozen-baseline three-run variability"
                    if stable_winners
                    else "no primary candidate cleared the five percent gate"
                ),
            },
        },
        "integrity": {
            "checkpoint_sha256": sorted(fixed_hashes),
            "all_runs_monitored": all_monitored,
            "no_abnormal_exits": no_abnormal_exits,
            "source_paths": list(source_paths),
            "source_sha256": list(source_sha256),
        },
        "passed": passed,
    }


def _all_run_paths(run_root: Path) -> list[Path]:
    return [
        _run_path(run_root, str(config["id"]), run_index)
        for config in primary_configurations()
        for run_index in range(1, FORMAL_RUNS + 1)
    ]


def _write_summary(
    *,
    run_root: Path,
    baseline_path: Path,
    output: Path,
) -> dict[str, object]:
    paths = _all_run_paths(run_root)
    reports = [_json(path) for path in paths]
    summary = summarize_scan(
        reports,
        _json(baseline_path),
        source_paths=[
            str(path).replace("\\", "/")
            for path in paths
        ],
        source_sha256=[_sha256(path) for path in paths],
    )
    resolved = _repo_path(output)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run and summarize the checklist 2.4 central-inference "
            "batching scan"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--config-id",
        choices=tuple(str(row["id"]) for row in primary_configurations()),
        required=True,
    )
    run.add_argument("--run-index", type=int, choices=(1, 2, 3), required=True)
    scan = subparsers.add_parser("scan")
    summarize = subparsers.add_parser("summarize")
    for command in (run, scan):
        command.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
        command.add_argument(
            "--run-root",
            type=Path,
            default=DEFAULT_RUN_ROOT,
        )
        command.add_argument(
            "--measured-agent-steps",
            type=int,
            default=MEASURED_AGENT_STEPS,
        )
        command.add_argument(
            "--monitor-interval-seconds",
            type=float,
            default=2.0,
        )
        command.add_argument("--force", action="store_true")
    for command in (scan, summarize):
        command.add_argument(
            "--baseline",
            type=Path,
            default=DEFAULT_BASELINE,
        )
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.command in {"run", "scan"}:
        if args.measured_agent_steps < MEASURED_AGENT_STEPS:
            parser.error(
                f"--measured-agent-steps must be at least "
                f"{MEASURED_AGENT_STEPS}"
            )
        if args.monitor_interval_seconds <= 0:
            parser.error("--monitor-interval-seconds must be positive")

    if args.command == "run":
        config = configuration_by_id(args.config_id)
        path = _run_path(args.run_root, args.config_id, args.run_index)
        report = run_configuration(
            config=config,
            run_index=args.run_index,
            checkpoint=args.checkpoint,
            output=path,
            measured_agent_steps=args.measured_agent_steps,
            warmup_updates=WARMUP_UPDATES,
            monitor_interval_seconds=args.monitor_interval_seconds,
            force=args.force,
        )
        print(json.dumps({
            "output": str(path).replace("\\", "/"),
            "agent_steps_per_second": report["measurement"][
                "agent_steps_per_second"
            ],
        }, sort_keys=True))
        return

    if args.command == "scan":
        for config in primary_configurations():
            for run_index in range(1, FORMAL_RUNS + 1):
                path = _run_path(
                    args.run_root,
                    str(config["id"]),
                    run_index,
                )
                report = run_configuration(
                    config=config,
                    run_index=run_index,
                    checkpoint=args.checkpoint,
                    output=path,
                    measured_agent_steps=args.measured_agent_steps,
                    warmup_updates=WARMUP_UPDATES,
                    monitor_interval_seconds=args.monitor_interval_seconds,
                    force=args.force,
                )
                print(json.dumps({
                    "completed": str(path).replace("\\", "/"),
                    "agent_steps_per_second": report["measurement"][
                        "agent_steps_per_second"
                    ],
                }, sort_keys=True), flush=True)

    summary = _write_summary(
        run_root=args.run_root,
        baseline_path=args.baseline,
        output=args.output,
    )
    print(json.dumps({
        "output": str(args.output).replace("\\", "/"),
        "passed": summary["passed"],
        "best_configuration": summary["diagnosis"]["best_configuration"],
        "enter_stage_2_4_b": summary["diagnosis"][
            "stage_2_4_b_gate"
        ]["enter"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
