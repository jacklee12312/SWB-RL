from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from scripts.scan_training_speed_stage_2_4 import (
    DEFAULT_CHECKPOINT,
    FORMAL_RUNS,
    MEASURED_AGENT_STEPS,
    WARMUP_UPDATES,
    _json,
    _repo_path,
    _sha256,
    run_configuration,
)


CONFIGURATION = {
    "id": "b_batched_learner_001",
    "dimension": "learner",
    "rollout_workers": 6,
    "worker_torch_threads": 2,
    "central_inference_batch_wait_ms": 1.0,
}
DEFAULT_RUN_ROOT = Path(
    "data/reports/training_speed/"
    "stage_2_7_b_batched_learner_001_runs"
)
DEFAULT_LEARNING = Path(
    "data/reports/training_speed/"
    "stage_2_7_b_batched_learner_001_learning.json"
)
DEFAULT_COMPARISON = Path(
    "data/reports/training_speed/stage_2_5_a_obs_001.json"
)
DEFAULT_FROZEN_BASELINE = Path(
    "data/reports/training_speed/baseline_summary.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/"
    "stage_2_7_b_batched_learner_001_end_to_end.json"
)


def _run_path(run_root: Path, run_index: int) -> Path:
    return run_root / f"b_batched_learner_001_run_{run_index}.json"


def _source(path: Path) -> dict[str, object]:
    return {
        "path": str(path).replace("\\", "/"),
        "sha256": _sha256(path),
    }


def build_report(
    runs: Sequence[Mapping[str, object]],
    learning: Mapping[str, object],
    comparison: Mapping[str, object],
    frozen_baseline: Mapping[str, object],
    *,
    sources: Mapping[str, object],
) -> dict[str, object]:
    if len(runs) != FORMAL_RUNS:
        raise ValueError("B-BATCHED-LEARNER-001 requires three runs")
    if sorted(int(run["run_index"]) for run in runs) != [1, 2, 3]:
        raise ValueError("run indexes must be 1, 2, 3")
    for run in runs:
        if run["configuration"] != CONFIGURATION:
            raise ValueError("runtime configuration drifted")

    speeds = [
        float(run["measurement"]["agent_steps_per_second"])
        for run in runs
    ]
    comparison_speeds = [
        float(value)
        for value in comparison["end_to_end"][
            "runs_agent_steps_per_second"
        ]
    ]
    median_speed = statistics.median(speeds)
    comparison_median = statistics.median(comparison_speeds)
    comparison_relative_range = (
        max(comparison_speeds) - min(comparison_speeds)
    ) / comparison_median
    relative_gain = (
        median_speed - comparison_median
    ) / comparison_median
    frozen_v4_1 = frozen_baseline["observations"]["v4.1"][
        "agent_steps_per_second"
    ]
    frozen_median = float(frozen_v4_1["median"])
    frozen_relative_gain = (
        median_speed - frozen_median
    ) / frozen_median

    all_step_gates = all(
        int(run["measurement"]["agent_steps"])
        >= MEASURED_AGENT_STEPS
        for run in runs
    )
    all_steady_updates = all(
        int(run["measurement"]["steady_update_count"]) >= 3
        for run in runs
    )
    all_profiling_disabled = all(
        bool(run["profiling_switches_disabled"]) for run in runs
    )
    no_abnormal_exits = all(
        int(run["measurement"]["abnormal_exit_count"]) == 0
        for run in runs
    )
    no_truncations = all(
        int(
            run["measurement"]["episode_length"][
                "truncated_episode_count"
            ]
        )
        == 0
        for run in runs
    )
    all_monitored = all(
        int(run["measurement"]["system"]["sample_count"]) > 0
        and int(run["measurement"]["system"]["gpu_sample_count"]) > 0
        for run in runs
    )
    checkpoint_unchanged = all(
        bool(run["checkpoint_unchanged"]) for run in runs
    )
    learning_passed = (
        bool(learning["passed"])
        and bool(learning["decision"]["advance_to_end_to_end"])
        and bool(
            learning["summary"]["learning_non_degradation_passed"]
        )
    )
    gain_is_clear = (
        relative_gain > 0.0
        and relative_gain > comparison_relative_range
    )
    integrity_passed = all((
        all_step_gates,
        all_steady_updates,
        all_profiling_disabled,
        no_abnormal_exits,
        no_truncations,
        all_monitored,
        checkpoint_unchanged,
    ))
    adopted = learning_passed and integrity_passed and gain_is_clear

    return {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_7_"
            "b_batched_learner_001_end_to_end"
        ),
        "checklist_section": "2.7",
        "candidate": "B-BATCHED-LEARNER-001",
        "classification": "B",
        "configuration": CONFIGURATION,
        "end_to_end": {
            "runs_agent_steps_per_second": speeds,
            "median_agent_steps_per_second": median_speed,
            "comparison_runs_agent_steps_per_second": (
                comparison_speeds
            ),
            "comparison_median_agent_steps_per_second": (
                comparison_median
            ),
            "relative_gain": relative_gain,
            "comparison_three_run_relative_range": (
                comparison_relative_range
            ),
            "frozen_v4_1_runs_agent_steps_per_second": list(
                frozen_v4_1["runs"]
            ),
            "frozen_v4_1_median_agent_steps_per_second": (
                frozen_median
            ),
            "frozen_v4_1_relative_gain": frozen_relative_gain,
            "collect_p95_seconds_median": statistics.median(
                float(run["measurement"]["collect_p95_seconds"])
                for run in runs
            ),
            "update_p95_seconds_median": statistics.median(
                float(run["measurement"]["update_p95_seconds"])
                for run in runs
            ),
        },
        "runtime": {
            "batch_mean_median": statistics.median(
                float(run["measurement"]["batching"]["mean_batch_size"])
                for run in runs
            ),
            "batch_p95_median": statistics.median(
                float(run["measurement"]["batching"]["p95_batch_size"])
                for run in runs
            ),
            "episode_p95_steps_median": statistics.median(
                float(run["measurement"]["episode_length"]["p95"])
                for run in runs
            ),
            "episode_maximum_steps": max(
                float(run["measurement"]["episode_length"]["maximum"])
                for run in runs
            ),
            "cpu_total_median_percent": statistics.median(
                float(
                    run["measurement"]["system"][
                        "cpu_total_median_percent"
                    ]
                )
                for run in runs
            ),
            "gpu_utilization_median_percent": statistics.median(
                float(
                    run["measurement"]["system"][
                        "gpu_utilization_median_percent"
                    ]
                )
                for run in runs
            ),
            "gpu_memory_peak_mib": max(
                float(
                    run["measurement"]["system"][
                        "gpu_memory_peak_mib"
                    ]
                )
                for run in runs
            ),
            "ram_used_peak_bytes": max(
                int(
                    run["measurement"]["system"][
                        "ram_used_peak_bytes"
                    ]
                )
                for run in runs
            ),
            "pagefile_used_peak_bytes": max(
                int(
                    run["measurement"]["system"][
                        "pagefile_used_peak_bytes"
                    ]
                )
                for run in runs
            ),
        },
        "decision": {
            "adopt": adopted,
            "default_enabled": adopted,
            "learning_gate_passed": learning_passed,
            "gain_exceeds_comparison_run_variability": (
                gain_is_clear
            ),
            "reason": (
                "three_seed_learning_and_end_to_end_gates_passed"
                if adopted
                else "learning_integrity_or_speed_gate_failed"
            ),
        },
        "integrity": {
            "all_runs_meet_step_gate": all_step_gates,
            "all_runs_have_three_steady_updates": all_steady_updates,
            "all_optional_profiling_disabled": (
                all_profiling_disabled
            ),
            "no_abnormal_exits": no_abnormal_exits,
            "no_truncations": no_truncations,
            "all_runs_monitored": all_monitored,
            "checkpoint_unchanged": checkpoint_unchanged,
            "checkpoint_sha256": sorted({
                str(run["checkpoint_sha256"]) for run in runs
            }),
        },
        "sources": dict(sources),
        "passed": learning_passed and integrity_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run and summarize the B-BATCHED-LEARNER-001 "
            "end-to-end gate"
        )
    )
    parser.add_argument("command", choices=("scan", "summarize"))
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--run-root", type=Path, default=DEFAULT_RUN_ROOT
    )
    parser.add_argument(
        "--learning", type=Path, default=DEFAULT_LEARNING
    )
    parser.add_argument(
        "--comparison", type=Path, default=DEFAULT_COMPARISON
    )
    parser.add_argument(
        "--frozen-baseline",
        type=Path,
        default=DEFAULT_FROZEN_BASELINE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.command == "scan":
        for run_index in range(1, FORMAL_RUNS + 1):
            path = _run_path(args.run_root, run_index)
            run = run_configuration(
                config=CONFIGURATION,
                run_index=run_index,
                checkpoint=args.checkpoint,
                output=path,
                measured_agent_steps=MEASURED_AGENT_STEPS,
                warmup_updates=WARMUP_UPDATES,
                monitor_interval_seconds=2.0,
                force=False,
            )
            print(json.dumps({
                "completed": str(path).replace("\\", "/"),
                "agent_steps_per_second": run["measurement"][
                    "agent_steps_per_second"
                ],
            }, sort_keys=True), flush=True)

    run_paths = [
        _run_path(args.run_root, run_index)
        for run_index in range(1, FORMAL_RUNS + 1)
    ]
    sources = {
        "learning": _source(args.learning),
        "comparison": _source(args.comparison),
        "frozen_baseline": _source(args.frozen_baseline),
        "runs": [_source(path) for path in run_paths],
    }
    report = build_report(
        [_json(path) for path in run_paths],
        _json(args.learning),
        _json(args.comparison),
        _json(args.frozen_baseline),
        sources=sources,
    )
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output).replace("\\", "/"),
        "passed": report["passed"],
        "adopt": report["decision"]["adopt"],
        "median_agent_steps_per_second": report["end_to_end"][
            "median_agent_steps_per_second"
        ],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
