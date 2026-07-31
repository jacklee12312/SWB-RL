from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from scripts.scan_training_speed_stage_2_4 import (
    DEFAULT_BASELINE,
    DEFAULT_CHECKPOINT,
    FORMAL_RUNS,
    MEASURED_AGENT_STEPS,
    WARMUP_UPDATES,
    WINNER_RELATIVE_GAIN,
    _json,
    _median_metric,
    _repo_path,
    _sha256,
    run_configuration,
)


DEFAULT_RUN_ROOT = Path(
    "data/reports/training_speed/stage_2_4_b_interaction_runs"
)
DEFAULT_PRIMARY_SCAN = Path(
    "data/reports/training_speed/stage_2_4_scan.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/stage_2_4_b_interactions.json"
)
INTERACTION_CONFIGURATIONS = (
    {
        "id": "workers_5_wait_1_0_ms",
        "dimension": "stable_winner_interaction",
        "rollout_workers": 5,
        "worker_torch_threads": 2,
        "central_inference_batch_wait_ms": 1.0,
        "constituents": ("workers_5", "wait_1_0_ms"),
    },
    {
        "id": "workers_6_wait_1_0_ms",
        "dimension": "stable_winner_interaction",
        "rollout_workers": 6,
        "worker_torch_threads": 2,
        "central_inference_batch_wait_ms": 1.0,
        "constituents": ("workers_6", "wait_1_0_ms"),
    },
)


def interaction_configurations() -> list[dict[str, object]]:
    return [
        {
            **config,
            "constituents": list(config["constituents"]),
        }
        for config in INTERACTION_CONFIGURATIONS
    ]


def _run_path(
    run_root: Path,
    config_id: str,
    run_index: int,
) -> Path:
    return run_root / f"{config_id}_run_{run_index}.json"


def summarize_interactions(
    reports: Sequence[Mapping[str, object]],
    primary_scan: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    source_paths: Sequence[str],
    source_sha256: Sequence[str],
) -> dict[str, object]:
    expected = {
        str(config["id"]): config
        for config in interaction_configurations()
    }
    stable_primary = set(
        primary_scan["diagnosis"]["stable_five_percent_winners"]
    )
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for report in reports:
        config = report["configuration"]
        if not isinstance(config, Mapping):
            raise ValueError("interaction configuration must be an object")
        config_id = str(config["id"])
        if config_id not in expected:
            raise ValueError(f"unexpected interaction config: {config_id}")
        if dict(config) != expected[config_id]:
            raise ValueError(f"configuration drift for {config_id}")
        constituents = set(config["constituents"])
        if not constituents.issubset(stable_primary):
            raise ValueError(
                f"{config_id} contains a non-winning constituent"
            )
        grouped.setdefault(config_id, []).append(report)
    if set(grouped) != set(expected):
        raise ValueError("interaction evidence is incomplete")

    baseline_speed = baseline["observations"]["v4.1"][
        "agent_steps_per_second"
    ]
    baseline_median = float(baseline_speed["median"])
    baseline_relative_range = (
        float(baseline_speed["range"]) / max(baseline_median, 1e-12)
    )
    primary_configs = primary_scan["configurations"]
    summaries: dict[str, dict[str, object]] = {}
    checkpoint_hashes = set()
    all_monitored = True
    no_abnormal_exits = True
    for config_id, rows in grouped.items():
        if len(rows) != FORMAL_RUNS:
            raise ValueError(f"{config_id} requires exactly three runs")
        if sorted(int(row["run_index"]) for row in rows) != [1, 2, 3]:
            raise ValueError(f"{config_id} run indexes must be 1, 2, 3")
        speeds = [
            float(row["measurement"]["agent_steps_per_second"])
            for row in rows
        ]
        median_speed = statistics.median(speeds)
        constituents = expected[config_id]["constituents"]
        constituent_speeds = {
            constituent: float(
                primary_configs[constituent][
                    "agent_steps_per_second"
                ]["median"]
            )
            for constituent in constituents
        }
        strongest_constituent = max(constituent_speeds.values())
        gain_vs_baseline = (
            median_speed - baseline_median
        ) / max(baseline_median, 1e-12)
        gain_vs_strongest_constituent = (
            median_speed - strongest_constituent
        ) / max(strongest_constituent, 1e-12)
        stable_vs_baseline = (
            gain_vs_baseline >= WINNER_RELATIVE_GAIN
            and gain_vs_baseline > baseline_relative_range
        )
        checkpoint_hashes.update(
            str(row["checkpoint_sha256"]) for row in rows
        )
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
            "configuration": expected[config_id],
            "run_count": len(rows),
            "agent_steps_per_second": {
                "runs": speeds,
                "median": median_speed,
                "minimum": min(speeds),
                "maximum": max(speeds),
                "range": max(speeds) - min(speeds),
                "relative_gain_vs_frozen_baseline": gain_vs_baseline,
                "relative_gain_vs_strongest_constituent": (
                    gain_vs_strongest_constituent
                ),
                "stable_five_percent_winner_vs_baseline": (
                    stable_vs_baseline
                ),
            },
            "constituent_median_steps_per_second": constituent_speeds,
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
                name: _median_metric(
                    rows, "measurement", "episode_length", name
                )
                for name in ("mean", "p50", "p95", "maximum")
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

    ranking = sorted(
        summaries,
        key=lambda config_id: summaries[config_id][
            "agent_steps_per_second"
        ]["median"],
        reverse=True,
    )
    best = ranking[0]
    best_payload = summaries[best]["agent_steps_per_second"]
    passed = (
        len(checkpoint_hashes) == 1
        and all_monitored
        and no_abnormal_exits
        and bool(best_payload["stable_five_percent_winner_vs_baseline"])
    )
    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_4_b_interactions",
        "checklist_section": "2.4B",
        "methodology": {
            "formal_runs_per_configuration": FORMAL_RUNS,
            "warmup_updates_excluded": WARMUP_UPDATES,
            "minimum_measured_agent_steps_per_run": MEASURED_AGENT_STEPS,
            "only_cross_dimension_stable_winners_combined": True,
            "full_cartesian_product": False,
            "baseline_median_agent_steps_per_second": baseline_median,
        },
        "configurations": summaries,
        "ranking": ranking,
        "decision": {
            "adopted_runtime_configuration": summaries[best][
                "configuration"
            ],
            "median_agent_steps_per_second": best_payload["median"],
            "relative_gain_vs_frozen_baseline": best_payload[
                "relative_gain_vs_frozen_baseline"
            ],
            "relative_gain_vs_strongest_constituent": best_payload[
                "relative_gain_vs_strongest_constituent"
            ],
            "passed": passed,
        },
        "integrity": {
            "checkpoint_sha256": sorted(checkpoint_hashes),
            "all_runs_monitored": all_monitored,
            "no_abnormal_exits": no_abnormal_exits,
            "source_paths": list(source_paths),
            "source_sha256": list(source_sha256),
        },
        "passed": passed,
    }


def _all_paths(run_root: Path) -> list[Path]:
    return [
        _run_path(run_root, str(config["id"]), run_index)
        for config in interaction_configurations()
        for run_index in range(1, FORMAL_RUNS + 1)
    ]


def _write_summary(
    *,
    run_root: Path,
    primary_scan_path: Path,
    baseline_path: Path,
    output: Path,
) -> dict[str, object]:
    paths = _all_paths(run_root)
    report = summarize_interactions(
        [_json(path) for path in paths],
        _json(primary_scan_path),
        _json(baseline_path),
        source_paths=[str(path).replace("\\", "/") for path in paths],
        source_sha256=[_sha256(path) for path in paths],
    )
    resolved = _repo_path(output)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify stage 2.4B stable-winner interactions"
    )
    parser.add_argument(
        "command",
        choices=("scan", "summarize"),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--primary-scan",
        type=Path,
        default=DEFAULT_PRIMARY_SCAN,
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--measured-agent-steps",
        type=int,
        default=MEASURED_AGENT_STEPS,
    )
    parser.add_argument("--monitor-interval-seconds", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.measured_agent_steps < MEASURED_AGENT_STEPS:
        parser.error(
            f"--measured-agent-steps must be at least "
            f"{MEASURED_AGENT_STEPS}"
        )
    if args.monitor_interval_seconds <= 0:
        parser.error("--monitor-interval-seconds must be positive")

    if args.command == "scan":
        for config in interaction_configurations():
            for run_index in range(1, FORMAL_RUNS + 1):
                path = _run_path(
                    args.run_root,
                    str(config["id"]),
                    run_index,
                )
                run = run_configuration(
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
                    "agent_steps_per_second": run["measurement"][
                        "agent_steps_per_second"
                    ],
                }, sort_keys=True), flush=True)

    report = _write_summary(
        run_root=args.run_root,
        primary_scan_path=args.primary_scan,
        baseline_path=args.baseline,
        output=args.output,
    )
    print(json.dumps({
        "output": str(args.output).replace("\\", "/"),
        "passed": report["passed"],
        "adopted_runtime_configuration": report["decision"][
            "adopted_runtime_configuration"
        ]["id"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
