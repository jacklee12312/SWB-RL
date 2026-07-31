from __future__ import annotations

import argparse
import hashlib
import json
import locale
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence

import psutil
import torch

from swb.db.repository import CardRepository
from swb.rl.checkpoint import build_checkpoint, load_checkpoint
from swb.rl.ppo import PPOConfig, PPOTrainer
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.versioning import ExperimentVersions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_REPORT_ROOT = Path("data/reports/training_speed")
DEFAULT_CHECKPOINT_ROOT = Path("data/checkpoints/training_speed")
SOURCE_CHECKPOINTS = {
    "v4.1": Path(
        "data/checkpoints/observation_nightly_20260729/final/"
        "v4_1_seed_20260801_500k.pt"
    ),
    "v3.6": Path(
        "data/checkpoints/observation_nightly_20260729/final/"
        "v3_6_seed_20260801_500k.pt"
    ),
}
FROZEN_CHECKPOINTS = {
    "v4.1": DEFAULT_CHECKPOINT_ROOT / "frozen_v4_1_seed_20260801_500k.pt",
    "v3.6": DEFAULT_CHECKPOINT_ROOT / "frozen_v3_6_seed_20260801_500k.pt",
}
ALLOWED_VERSION_MIGRATIONS = frozenset({
    "catalog_sha256",
    "training_pool_sha256",
})
STAGE_1_FREEZE_COMMIT = "fae33c2"
MEASURED_AGENT_STEPS = 100_000
WARMUP_UPDATES = 2


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def version_differences(
    source: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, tuple[object, object]]:
    keys = set(source) | set(current)
    return {
        key: (source.get(key), current.get(key))
        for key in sorted(keys)
        if source.get(key) != current.get(key)
    }


def _current_versions(
    snapshot: WorkerAssetsSnapshot,
    payload: Mapping[str, object],
) -> tuple[dict[str, object], PPOTrainer]:
    trainer_state = payload["trainer"]
    assert isinstance(trainer_state, Mapping)
    config = PPOConfig(**dict(trainer_state["config"]))
    trainer = PPOTrainer(
        snapshot,
        master_seed=int(trainer_state["master_seed"]),
        config=config,
        device="cuda",
    )
    assert trainer.env is not None
    versions = ExperimentVersions.capture(
        trainer.env,
        snapshot.catalog,
        rulebook_sha256=snapshot.rulebook_sha256,
    ).to_dict()
    return versions, trainer


def prepare_checkpoint(
    *,
    source: Path,
    destination: Path,
    snapshot: WorkerAssetsSnapshot,
) -> dict[str, object]:
    source = _repo_path(source)
    destination = _repo_path(destination)
    source_stat = source.stat()
    source_hash = _sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if int(payload.get("checkpoint_schema_version", -1)) != 2:
        raise ValueError("only schema-v2 checkpoints can be frozen")
    current_versions, trainer = _current_versions(snapshot, payload)
    try:
        source_versions = dict(payload["versions"])
        differences = version_differences(source_versions, current_versions)
        if set(differences) != set(ALLOWED_VERSION_MIGRATIONS):
            raise ValueError(
                "performance checkpoint has unsafe compatibility differences: "
                f"{sorted(differences)}"
            )
        excluded = set(snapshot.catalog.excluded_collectible_ids)
        deck_ids = set()
        if trainer.fixed_training_deck is not None:
            deck_ids.update(trainer.fixed_training_deck.card_ids)
        for deck in trainer.fixed_opponent_decks:
            deck_ids.update(deck.card_ids)
        environment = payload["environment"]
        for ids in environment["deck_card_ids"]:
            deck_ids.update(int(card_id) for card_id in ids)
        overlap = sorted(excluded & deck_ids)
        if overlap:
            raise ValueError(
                "excluded Catalog cards occur in the fixed benchmark decks: "
                f"{overlap}"
            )

        trainer.model.load_state_dict(payload["model_state"])
        trainer.optimizer.load_state_dict(payload["optimizer_state"])
        frozen_payload = build_checkpoint(trainer)
        frozen_payload["performance_baseline_freeze"] = {
            "stage_1_freeze_commit": STAGE_1_FREEZE_COMMIT,
            "source_checkpoint": str(source.relative_to(ROOT)),
            "source_sha256": source_hash,
            "source_agent_steps": int(payload["trainer"]["agent_steps"]),
            "source_completed_episodes": int(
                payload["trainer"]["completed_episodes"]
            ),
            "benchmark_episode_schedule_reset": True,
            "migrated_version_fields": {
                key: {"from": old, "to": new}
                for key, (old, new) in differences.items()
            },
            "excluded_collectible_ids": sorted(excluded),
            "excluded_cards_in_fixed_decks": overlap,
            "purpose": (
                "Performance measurement only. The original checkpoint is "
                "unchanged; strict general checkpoint compatibility remains "
                "unchanged."
            ),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                torch.save(frozen_payload, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
    finally:
        trainer.close()

    if (
        source.stat().st_size != source_stat.st_size
        or source.stat().st_mtime_ns != source_stat.st_mtime_ns
        or _sha256(source) != source_hash
    ):
        raise RuntimeError("source checkpoint changed during freeze")
    loaded = load_checkpoint(
        destination,
        snapshot,
        device="cpu",
        restore_rng_state=False,
    )
    try:
        loaded_config = loaded.hyperparameters()
    finally:
        loaded.close()
    return {
        "source": str(source.relative_to(ROOT)),
        "source_sha256": source_hash,
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "destination": str(destination.relative_to(ROOT)),
        "destination_sha256": _sha256(destination),
        "destination_size": destination.stat().st_size,
        "destination_mtime_ns": destination.stat().st_mtime_ns,
        "checkpoint_structure_load_verified_on_cpu": True,
        "checkpoint_cuda_rng_restore_verified": False,
        "migrated_version_fields": sorted(ALLOWED_VERSION_MIGRATIONS),
        "excluded_collectible_ids": sorted(
            snapshot.catalog.excluded_collectible_ids
        ),
        "excluded_cards_in_fixed_decks": [],
        "configuration": loaded_config,
    }


def _run_text(*command: str) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _nvidia_sample() -> dict[str, object]:
    output = _run_text(
        "nvidia-smi",
        "--query-gpu=name,driver_version,utilization.gpu,memory.used,"
        "memory.total,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    )
    if not output:
        return {}
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    if len(values) != 7:
        return {}
    return {
        "name": values[0],
        "driver_version": values[1],
        "utilization_percent": float(values[2]),
        "memory_used_mib": float(values[3]),
        "memory_total_mib": float(values[4]),
        "power_watts": float(values[5]),
        "power_limit_watts": float(values[6]),
    }


def _background_training_processes() -> list[dict[str, object]]:
    rows = []
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            command = " ".join(process.info.get("cmdline") or ())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        lowered = command.lower()
        if (
            "train_ppo" in lowered
            or "profile_ppo_training" in lowered
            or "nightly_observation" in lowered
        ):
            rows.append({
                "pid": process.info["pid"],
                "name": process.info.get("name"),
                "command": command,
            })
    return rows


class SystemMonitor:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        psutil.cpu_percent(interval=None, percpu=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, 2 * self.interval_seconds))

    def _run(self) -> None:
        started = time.perf_counter()
        while not self._stop.is_set():
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            self.samples.append({
                "elapsed_seconds": time.perf_counter() - started,
                "cpu_total_percent": (
                    statistics.fmean(per_core) if per_core else 0.0
                ),
                "cpu_per_core_percent": per_core,
                "ram_used_bytes": memory.used,
                "ram_available_bytes": memory.available,
                "pagefile_used_bytes": swap.used,
                "pagefile_percent": swap.percent,
                "pagefile_sin_bytes": swap.sin,
                "pagefile_sout_bytes": swap.sout,
                "gpu": _nvidia_sample(),
            })
            self._stop.wait(self.interval_seconds)


def _system_manifest() -> dict[str, object]:
    memory = psutil.virtual_memory()
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "windows_version": platform.version(),
        "cpu": platform.processor(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "ram_total_bytes": memory.total,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_initial": _nvidia_sample(),
        "power_scheme": _run_text("powercfg", "/GETACTIVESCHEME"),
        "background_training_processes": _background_training_processes(),
    }


def _steady_measurement(
    profile: Mapping[str, object],
) -> dict[str, object]:
    steady_iterations = profile["iterations"][WARMUP_UPDATES:]
    steps = sum(int(row["agent_steps"]) for row in steady_iterations)
    elapsed = sum(float(row["elapsed_seconds"]) for row in steady_iterations)
    if steps < MEASURED_AGENT_STEPS:
        raise ValueError(
            f"steady measurement has {steps} steps, expected at least "
            f"{MEASURED_AGENT_STEPS}"
        )
    return {
        "excluded_warmup_updates": WARMUP_UPDATES,
        "agent_steps": steps,
        "elapsed_seconds": elapsed,
        "agent_steps_per_second": steps / elapsed,
        "update_count": len(steady_iterations),
    }


def run_baseline(
    *,
    observation: str,
    run_index: int,
    checkpoint: Path,
    output: Path,
    monitor_interval_seconds: float,
) -> dict[str, object]:
    checkpoint = _repo_path(checkpoint)
    output = _repo_path(output)
    checkpoint_stat = checkpoint.stat()
    checkpoint_hash = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rollout_steps = int(payload["trainer"]["config"]["rollout_steps"])
    requested_steps = (
        MEASURED_AGENT_STEPS + WARMUP_UPDATES * rollout_steps
    )
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
        str(WARMUP_UPDATES),
        "--device",
        "cuda",
        "--rollout-worker-threads",
        "2",
        "--central-inference-batch-wait-ms",
        "0.5",
        "--output",
        str(raw_output),
    ]
    system_before = _system_manifest()
    monitor = SystemMonitor(monitor_interval_seconds)
    monitor.start()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    monitor.stop()
    if result.returncode != 0:
        raise RuntimeError(
            f"profile failed with exit code {result.returncode}; see {log_path}"
        )
    profile = _json(raw_output)
    raw_output.unlink()
    checkpoint_after = checkpoint.stat()
    checkpoint_unchanged = (
        checkpoint_stat.st_size == checkpoint_after.st_size
        and checkpoint_stat.st_mtime_ns == checkpoint_after.st_mtime_ns
        and checkpoint_hash == _sha256(checkpoint)
    )
    if not checkpoint_unchanged or not profile["checkpoint_unchanged"]:
        raise RuntimeError("fixed performance checkpoint changed during profile")
    profile["baseline"] = {
        "schema_version": 1,
        "stage_1_freeze_commit": STAGE_1_FREEZE_COMMIT,
        "observation": observation,
        "run_index": run_index,
        "started_utc": started_utc,
        "classification": "A-PROFILE-001",
        "measured_agent_steps_minimum": MEASURED_AGENT_STEPS,
        "warmup_updates": WARMUP_UPDATES,
        "fixed_checkpoint_sha256": checkpoint_hash,
        "fixed_checkpoint_size": checkpoint_stat.st_size,
        "fixed_checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "fixed_checkpoint_unchanged": checkpoint_unchanged,
        "system_before": system_before,
        "system_after": _system_manifest(),
        "monitor_interval_seconds": monitor_interval_seconds,
        "system_samples": monitor.samples,
        "steady_measurement": _steady_measurement(profile),
        "profile_command": command,
        "profile_log": str(log_path.relative_to(ROOT)),
    }
    output.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize_baselines(paths: Sequence[Path]) -> dict[str, object]:
    reports = [_json(_repo_path(path)) for path in paths]
    by_observation: dict[str, list[dict[str, object]]] = {}
    for report in reports:
        baseline = report["baseline"]
        by_observation.setdefault(
            str(baseline["observation"]), []
        ).append(report)
    if set(by_observation) != {"v3.6", "v4.1"}:
        raise ValueError(
            "v3.6 and v4.1 each require exactly three baseline runs"
        )
    observations = {}
    for observation, rows in sorted(by_observation.items()):
        if len(rows) != 3:
            raise ValueError(
                f"{observation} requires exactly three baseline runs"
            )
        speeds = [
            float(row["baseline"]["steady_measurement"][
                "agent_steps_per_second"
            ])
            for row in rows
        ]
        collect_samples = [
            float(iteration["collect"]["collect_total_seconds"])
            for row in rows
            for iteration in row["iterations"][WARMUP_UPDATES:]
        ]
        update_samples = [
            float(iteration["update"]["update_total_seconds"])
            for row in rows
            for iteration in row["iterations"][WARMUP_UPDATES:]
        ]
        system_samples = [
            sample
            for row in rows
            for sample in row["baseline"]["system_samples"]
        ]
        gpu_samples = [
            sample["gpu"]
            for sample in system_samples
            if sample.get("gpu")
        ]
        observations[observation] = {
            "run_count": 3,
            "agent_steps_per_second": {
                "runs": speeds,
                "median": statistics.median(speeds),
                "minimum": min(speeds),
                "maximum": max(speeds),
                "range": max(speeds) - min(speeds),
            },
            "stage_time_seconds": {
                "collect_p95": _percentile(collect_samples, 0.95),
                "update_p95": _percentile(update_samples, 0.95),
            },
            "system_monitor": {
                "sample_count": len(system_samples),
                "cpu_total_median_percent": statistics.median(
                    float(sample["cpu_total_percent"])
                    for sample in system_samples
                ),
                "ram_used_peak_bytes": max(
                    int(sample["ram_used_bytes"])
                    for sample in system_samples
                ),
                "pagefile_used_peak_bytes": max(
                    int(sample["pagefile_used_bytes"])
                    for sample in system_samples
                ),
                "gpu_utilization_median_percent": (
                    statistics.median(
                        float(sample["utilization_percent"])
                        for sample in gpu_samples
                    )
                    if gpu_samples else None
                ),
                "gpu_memory_peak_mib": (
                    max(
                        float(sample["memory_used_mib"])
                        for sample in gpu_samples
                    )
                    if gpu_samples else None
                ),
                "gpu_power_p95_watts": (
                    _percentile(
                        [
                            float(sample["power_watts"])
                            for sample in gpu_samples
                        ],
                        0.95,
                    )
                    if gpu_samples else None
                ),
            },
            "checkpoint_sha256": sorted({
                row["baseline"]["fixed_checkpoint_sha256"]
                for row in rows
            }),
        }
    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_baseline_summary",
        "stage_1_freeze_commit": STAGE_1_FREEZE_COMMIT,
        "comparison_policy": (
            "v3.6 and v4.1 are reported independently. Different Observation "
            "widths are not interpreted as implementation regressions."
        ),
        "observations": observations,
        "input_reports": [str(path) for path in paths],
        "passed": set(observations) == {"v3.6", "v4.1"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, run and summarize the frozen PPO speed baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    run = subparsers.add_parser("run")
    run.add_argument("--observation", choices=("v3.6", "v4.1"), required=True)
    run.add_argument("--run-index", type=int, choices=(1, 2, 3), required=True)
    run.add_argument("--monitor-interval-seconds", type=float, default=2.0)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_ROOT / "baseline_summary.json",
    )
    args = parser.parse_args()

    if args.command == "prepare":
        snapshot = WorkerAssetsSnapshot.build(
            CardRepository(_repo_path(args.database))
        )
        report = {
            "schema_version": 1,
            "stage_1_freeze_commit": STAGE_1_FREEZE_COMMIT,
            "checkpoints": {
                observation: prepare_checkpoint(
                    source=SOURCE_CHECKPOINTS[observation],
                    destination=FROZEN_CHECKPOINTS[observation],
                    snapshot=snapshot,
                )
                for observation in ("v4.1", "v3.6")
            },
        }
        output = _repo_path(
            DEFAULT_REPORT_ROOT / "baseline_configuration.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return

    if args.command == "run":
        output = (
            DEFAULT_REPORT_ROOT
            / f"baseline_run_{args.observation.replace('.', '_')}_"
            f"{args.run_index}.json"
        )
        report = run_baseline(
            observation=args.observation,
            run_index=args.run_index,
            checkpoint=FROZEN_CHECKPOINTS[args.observation],
            output=output,
            monitor_interval_seconds=args.monitor_interval_seconds,
        )
        print(json.dumps({
            "output": str(output),
            "steady_measurement": report["baseline"]["steady_measurement"],
        }, sort_keys=True))
        return

    paths = [
        DEFAULT_REPORT_ROOT / f"baseline_run_{observation}_{run_index}.json"
        for observation in ("v4_1", "v3_6")
        for run_index in (1, 2, 3)
    ]
    report = summarize_baselines(paths)
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
