from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping

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


DEFAULT_RUN_ROOT = Path(
    "data/reports/training_speed/stage_2_5_a_obs_001_runs"
)
DEFAULT_STAGE_2_2 = Path(
    "data/reports/training_speed/stage_2_2_profiling_disabled_smoke.json"
)
DEFAULT_STAGE_2_4 = Path(
    "data/reports/training_speed/stage_2_4_b_interactions.json"
)
DEFAULT_ENVIRONMENT = Path(
    "data/reports/training_speed/stage_2_5_environment_benchmark.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/stage_2_5_a_obs_001.json"
)
CONFIGURATION = {
    "id": "a_obs_001",
    "dimension": "implementation_candidate",
    "rollout_workers": 6,
    "worker_torch_threads": 2,
    "central_inference_batch_wait_ms": 1.0,
}
MATERIALITY_THRESHOLD = 0.05
REMAINING_CANDIDATES = [
    "precompute_static_card_fields",
    "write_fixed_tokens_into_contiguous_arrays",
    "preallocate_observation_buffers",
    "remove_policy_unused_debug_fields_from_ipc",
    "remove_unnecessary_wide_dtypes_and_implicit_conversions",
]


def _run_path(run_root: Path, run_index: int) -> Path:
    return run_root / f"a_obs_001_run_{run_index}.json"


def _before_field(before: Mapping[str, object], name: str) -> float:
    return float(
        before["steady_state"]["collect"]["fields"][name]["total"]
    )


def build_report(
    runs: list[Mapping[str, object]],
    before: Mapping[str, object],
    stage_2_4: Mapping[str, object],
    environment: Mapping[str, object],
    *,
    sources: Mapping[str, object],
) -> dict[str, object]:
    if len(runs) != FORMAL_RUNS:
        raise ValueError("A-OBS-001 requires exactly three runs")
    if sorted(int(run["run_index"]) for run in runs) != [1, 2, 3]:
        raise ValueError("A-OBS-001 run indexes must be 1, 2, 3")
    for run in runs:
        if run["configuration"] != CONFIGURATION:
            raise ValueError("A-OBS-001 runtime configuration drifted")
        measurement = run["measurement"]
        observation = measurement["observation"]
        if int(measurement["agent_steps"]) < MEASURED_AGENT_STEPS:
            raise ValueError("A-OBS-001 run is shorter than the measurement gate")
        if int(measurement["steady_update_count"]) < 3:
            raise ValueError("A-OBS-001 run has fewer than three steady updates")
        if not bool(run["profiling_switches_disabled"]):
            raise ValueError("A-OBS-001 run enabled optional profiling")
        if int(measurement["abnormal_exit_count"]) != 0:
            raise ValueError("A-OBS-001 run recorded an abnormal exit")
        if float(observation["decision_construction_seconds"]) != 0.0:
            raise ValueError(
                "A-OBS-001 still constructs an observation at decision time"
            )
        expected_total = (
            float(observation["step_construction_seconds"])
            + float(observation["bootstrap_construction_seconds"])
        )
        if abs(
            float(observation["total_construction_seconds"]) - expected_total
        ) > 1e-9:
            raise ValueError("A-OBS-001 observation timing components drifted")
    speeds = [
        float(run["measurement"]["agent_steps_per_second"])
        for run in runs
    ]
    median_speed = statistics.median(speeds)
    comparison_speed = float(
        stage_2_4["decision"]["median_agent_steps_per_second"]
    )
    relative_gain = (
        median_speed - comparison_speed
    ) / max(comparison_speed, 1e-12)
    baseline_range = float(
        stage_2_4["configurations"]["workers_6_wait_1_0_ms"][
            "agent_steps_per_second"
        ]["range"]
    )
    baseline_relative_range = baseline_range / comparison_speed

    before_decision = _before_field(
        before, "worker_decision_observation_construction_seconds"
    )
    before_step = _before_field(
        before, "worker_step_observation_construction_seconds"
    )
    before_bootstrap = _before_field(
        before, "worker_bootstrap_observation_construction_seconds"
    )
    before_total = _before_field(
        before, "worker_observation_construction_seconds"
    )
    before_steps = _before_field(before, "worker_agent_steps")
    pipeline_wall = float(
        before["steady_state"]["pipeline_wall_time"]["measured_seconds"]
    )
    worker_count = int(
        before["runtime_rollout_configuration"]["rollout_workers"]
    )
    observation_concurrency_normalized_wall_fraction = before_total / (
        worker_count * pipeline_wall
    )
    observation_serial_worker_time_fraction = before_total / pipeline_wall
    after_decision = [
        float(
            run["measurement"]["observation"][
                "decision_construction_seconds"
            ]
        )
        for run in runs
    ]
    after_total = [
        float(
            run["measurement"]["observation"][
                "total_construction_seconds"
            ]
        )
        for run in runs
    ]
    trajectory_gate = (
        all(value == 0.0 for value in after_decision)
        and all(bool(run["checkpoint_unchanged"]) for run in runs)
    )
    remaining_same_source_cost_profiled = all(
        value > 0.0 for value in after_total
    )
    continue_gate = (
        observation_concurrency_normalized_wall_fraction
        >= MATERIALITY_THRESHOLD
        or (
            relative_gain > baseline_relative_range
            and relative_gain > 0.0
            and remaining_same_source_cost_profiled
        )
    )
    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_5_a_obs_001",
        "checklist_section": "2.5",
        "candidate": "A-OBS-001",
        "configuration": CONFIGURATION,
        "minimal_reproduction": {
            "before_decision_observation_construction_seconds": (
                before_decision
            ),
            "before_decision_milliseconds_per_agent_step": (
                1000.0 * before_decision / before_steps
            ),
            "before_step_observation_construction_seconds": before_step,
            "before_bootstrap_observation_construction_seconds": (
                before_bootstrap
            ),
            "before_total_observation_construction_seconds": before_total,
            "after_decision_observation_construction_seconds": after_decision,
            "after_total_observation_construction_seconds": after_total,
            "call_trace": [
                "env.reset returns first observation",
                "policy consumes cached current observation",
                "env.step returns next observation",
                "next policy decision consumes StepResult.observation",
            ],
        },
        "environment_microbenchmark": {
            "observation_cached_per_second": float(
                environment["rates"]["observe_cached_per_second"]
            ),
            "observation_cold_per_second": float(
                environment["rates"]["observe_cold_per_second"]
            ),
            "observation_cache_speedup": float(
                environment["rates"]["observe_cache_speedup"]
            ),
            "full_environment_steps_per_second": float(
                environment["rates"]["step_per_second"]
            ),
            "passed": bool(environment["thresholds_passed"]),
        },
        "end_to_end": {
            "runs_agent_steps_per_second": speeds,
            "median_agent_steps_per_second": median_speed,
            "stage_2_4_comparison_median_agent_steps_per_second": (
                comparison_speed
            ),
            "relative_gain": relative_gain,
            "comparison_three_run_relative_range": baseline_relative_range,
        },
        "decision_gate": {
            "observation_concurrency_normalized_pipeline_wall_fraction": (
                observation_concurrency_normalized_wall_fraction
            ),
            "observation_serial_worker_time_sum_over_pipeline_wall": (
                observation_serial_worker_time_fraction
            ),
            "rollout_worker_count_for_normalization": worker_count,
            "materiality_threshold": MATERIALITY_THRESHOLD,
            "gain_exceeds_run_variability": (
                relative_gain > baseline_relative_range
            ),
            "remaining_same_source_cost_profiled": (
                remaining_same_source_cost_profiled
            ),
            "continue_remaining_stage_2_5_candidates": continue_gate,
            "remaining_candidate_disposition": (
                "continue"
                if continue_gate
                else "closed_below_materiality_and_variability_gate"
            ),
            "remaining_candidates": REMAINING_CANDIDATES,
        },
        "equivalence": {
            "decision_duplicate_removed": trajectory_gate,
            "fixed_seed_full_trajectory_contract_test": (
                "tests.test_ppo.PPOTrainerTests."
                "test_seeded_central_policy_rollout_is_reproducible"
            ),
            "observation_mask_hidden_logprob_value_generation_covered": True,
            "checkpoint_unchanged": all(
                bool(run["checkpoint_unchanged"]) for run in runs
            ),
        },
        "integrity": {
            "all_runs_meet_step_gate": all(
                int(run["measurement"]["agent_steps"])
                >= MEASURED_AGENT_STEPS
                for run in runs
            ),
            "all_runs_have_three_steady_updates": all(
                int(run["measurement"]["steady_update_count"]) >= 3
                for run in runs
            ),
            "all_optional_profiling_disabled": all(
                bool(run["profiling_switches_disabled"]) for run in runs
            ),
            "no_abnormal_exits": all(
                int(run["measurement"]["abnormal_exit_count"]) == 0
                for run in runs
            ),
            "checkpoint_sha256": sorted({
                str(run["checkpoint_sha256"]) for run in runs
            }),
        },
        "sources": dict(sources),
        "passed": (
            trajectory_gate
            and bool(environment["thresholds_passed"])
            and len(speeds) == FORMAL_RUNS
            and all(
                int(run["measurement"]["agent_steps"])
                >= MEASURED_AGENT_STEPS
                for run in runs
            )
            and all(
                int(run["measurement"]["abnormal_exit_count"]) == 0
                for run in runs
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and summarize checklist 2.5 A-OBS-001"
    )
    parser.add_argument("command", choices=("scan", "summarize"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--before", type=Path, default=DEFAULT_STAGE_2_2)
    parser.add_argument("--stage-2-4", type=Path, default=DEFAULT_STAGE_2_4)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENVIRONMENT)
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
    paths = {
        "before": args.before,
        "stage_2_4": args.stage_2_4,
        "environment": args.environment,
    }
    run_paths = [
        _run_path(args.run_root, run_index)
        for run_index in range(1, FORMAL_RUNS + 1)
    ]
    report = build_report(
        [_json(path) for path in run_paths],
        _json(args.before),
        _json(args.stage_2_4),
        _json(args.environment),
        sources={
            **{
                name: {
                    "path": str(path).replace("\\", "/"),
                    "sha256": _sha256(path),
                }
                for name, path in paths.items()
            },
            "runs": [
                {
                    "path": str(path).replace("\\", "/"),
                    "sha256": _sha256(path),
                }
                for path in run_paths
            ],
        },
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
        "continue": report["decision_gate"][
            "continue_remaining_stage_2_5_candidates"
        ],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
