from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path
from typing import Mapping

import torch

from scripts.profile_v4_1_inference_breakdown import (
    _checkpoint_contract,
    _device_fixture,
    _fixed_fixture,
    benchmark_forward_batches,
    profile_forward_components,
    run_torch_profiler,
)
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
from swb.db.repository import CardRepository
from swb.rl.checkpoint import load_checkpoint
from swb.rl.runtime import WorkerAssetsSnapshot


BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_BEFORE = Path(
    "data/reports/training_speed/v4_1_inference_breakdown.json"
)
DEFAULT_COMPARISON = Path(
    "data/reports/training_speed/stage_2_5_a_obs_001.json"
)
DEFAULT_MICRO = Path(
    "data/reports/training_speed/stage_2_6_a_net_001_micro.json"
)
DEFAULT_TRACE = Path(
    "data/reports/training_speed/stage_2_6_a_net_001_trace.json.gz"
)
DEFAULT_RUN_ROOT = Path(
    "data/reports/training_speed/stage_2_6_a_net_001_runs"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/stage_2_6_a_net_001.json"
)
CONFIGURATION = {
    "id": "a_net_001",
    "dimension": "implementation_candidate",
    "rollout_workers": 6,
    "worker_torch_threads": 2,
    "central_inference_batch_wait_ms": 1.0,
}


def _run_path(run_root: Path, run_index: int) -> Path:
    return run_root / f"a_net_001_run_{run_index}.json"


def _same_tensor(
    first: torch.Tensor,
    second: torch.Tensor,
) -> bool:
    return (
        first.dtype == second.dtype
        and first.shape == second.shape
        and torch.equal(first, second)
    )


def run_microbenchmark(
    *,
    database: Path,
    checkpoint: Path,
    before: Path,
    output: Path,
    trace_output: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint_path = _repo_path(checkpoint)
    checkpoint_before = _checkpoint_contract(checkpoint_path)
    snapshot = WorkerAssetsSnapshot.build(
        CardRepository(_repo_path(database))
    )
    trainer = load_checkpoint(
        checkpoint_path,
        snapshot,
        device=str(device),
        restore_rng_state=False,
    )
    try:
        model = trainer.model
        model.eval()
        fixture = _fixed_fixture(
            model,
            maximum_batch_size=max(BATCH_SIZES),
            seed=20260801,
        )
        equality_inputs = _device_fixture(
            fixture,
            batch_size=4,
            device=device,
        )
        with torch.no_grad():
            checked_outputs = model.forward_step(
                equality_inputs["observation"],
                equality_inputs["hidden"],
                equality_inputs["card_indices"],
            )

        checked_forward = model.forward_step
        checked_masked_logits = model.masked_logits

        def prevalidated_forward(
            observation,
            hidden,
            card_indices=None,
        ):
            return checked_forward(
                observation,
                hidden,
                card_indices,
                validate_card_indices=False,
            )

        def prevalidated_masked_logits(logits, action_mask):
            return checked_masked_logits(
                logits,
                action_mask,
                validate_legal_rows=False,
            )

        model.forward_step = prevalidated_forward
        model.masked_logits = prevalidated_masked_logits
        try:
            with torch.no_grad():
                prevalidated_outputs = model.forward_step(
                    equality_inputs["observation"],
                    equality_inputs["hidden"],
                    equality_inputs["card_indices"],
                )
            forward = benchmark_forward_batches(
                model,
                fixture,
                batch_sizes=BATCH_SIZES,
                device=device,
                warmup_iterations=8,
                measured_iterations=20,
                repeats=3,
            )
            components = profile_forward_components(
                model,
                fixture,
                batch_sizes=BATCH_SIZES,
                repeats=3,
                device=device,
            )
            profiler = run_torch_profiler(
                model,
                fixture,
                batch_size=4,
                iterations=3,
                device=device,
                trace_path=_repo_path(trace_output),
            )
        finally:
            model.forward_step = checked_forward
            model.masked_logits = checked_masked_logits
    finally:
        trainer.close()

    checkpoint_after = _checkpoint_contract(checkpoint_path)
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("the frozen checkpoint changed during A-NET-001")
    baseline = _json(before)
    common_batches = [
        str(batch)
        for batch in BATCH_SIZES
        if str(batch) in baseline["fixed_input_forward"]["v4.1"]
    ]
    comparisons = {}
    for batch in common_batches:
        before_ms = float(
            baseline["fixed_input_forward"]["v4.1"][batch][
                "device_milliseconds_per_call"
            ]["median"]
        )
        after_ms = float(
            forward[batch]["device_milliseconds_per_call"]["median"]
        )
        comparisons[batch] = {
            "before_device_milliseconds": before_ms,
            "after_device_milliseconds": after_ms,
            "relative_reduction": (before_ms - after_ms) / before_ms,
        }
    equality = all(
        _same_tensor(first, second)
        for first, second in zip(
            checked_outputs,
            prevalidated_outputs,
        )
    )
    report = {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_6_a_net_001_micro",
        "checklist_section": "2.6",
        "candidate": "A-NET-001",
        "classification": "A",
        "methodology": {
            "fixture_seed": 20260801,
            "batch_sizes": list(BATCH_SIZES),
            "warmup_iterations": 8,
            "measured_iterations": 20,
            "repeats": 3,
            "profiler_batch_size": 4,
            "profiler_iterations": 3,
            "validated_boundary": (
                "CPU card-index and live-mask validation before H2D; "
                "GPU forward and masked logits consume prevalidated tensors"
            ),
        },
        "hardware": {
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "checkpoint": checkpoint_after,
        "exact_output_equivalence": {
            "logits": _same_tensor(
                checked_outputs[0], prevalidated_outputs[0]
            ),
            "value": _same_tensor(
                checked_outputs[1], prevalidated_outputs[1]
            ),
            "hidden": _same_tensor(
                checked_outputs[2], prevalidated_outputs[2]
            ),
            "all": equality,
        },
        "fixed_input_forward": forward,
        "component_profile": components,
        "profiler": profiler,
        "comparison_to_stage_2_3": comparisons,
        "source": {
            "path": str(before).replace("\\", "/"),
            "sha256": _sha256(before),
        },
        "passed": equality and checkpoint_before == checkpoint_after,
    }
    destination = _repo_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_report(
    runs: list[Mapping[str, object]],
    micro: Mapping[str, object],
    comparison: Mapping[str, object],
    *,
    sources: Mapping[str, object],
) -> dict[str, object]:
    if len(runs) != FORMAL_RUNS:
        raise ValueError("A-NET-001 requires exactly three runs")
    if sorted(int(run["run_index"]) for run in runs) != [1, 2, 3]:
        raise ValueError("A-NET-001 run indexes must be 1, 2, 3")
    for run in runs:
        if run["configuration"] != CONFIGURATION:
            raise ValueError("A-NET-001 runtime configuration drifted")
        measurement = run["measurement"]
        if int(measurement["agent_steps"]) < MEASURED_AGENT_STEPS:
            raise ValueError("A-NET-001 run is shorter than the step gate")
        if int(measurement["steady_update_count"]) < 3:
            raise ValueError("A-NET-001 run lacks steady updates")
        if not bool(run["profiling_switches_disabled"]):
            raise ValueError("A-NET-001 enabled optional profiling")
        if int(measurement["abnormal_exit_count"]) != 0:
            raise ValueError("A-NET-001 recorded an abnormal exit")
        if not bool(run["checkpoint_unchanged"]):
            raise ValueError("A-NET-001 changed the frozen checkpoint")
    speeds = [
        float(run["measurement"]["agent_steps_per_second"])
        for run in runs
    ]
    median_speed = statistics.median(speeds)
    comparison_speeds = [
        float(value)
        for value in comparison["end_to_end"][
            "runs_agent_steps_per_second"
        ]
    ]
    comparison_median = statistics.median(comparison_speeds)
    relative_gain = (
        median_speed - comparison_median
    ) / comparison_median
    comparison_relative_range = (
        max(comparison_speeds) - min(comparison_speeds)
    ) / comparison_median
    adopted = (
        relative_gain > 0.0
        and relative_gain > comparison_relative_range
    )
    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_6_a_net_001",
        "checklist_section": "2.6",
        "candidate": "A-NET-001",
        "classification": "A",
        "configuration": CONFIGURATION,
        "end_to_end": {
            "runs_agent_steps_per_second": speeds,
            "median_agent_steps_per_second": median_speed,
            "comparison_runs_agent_steps_per_second": comparison_speeds,
            "comparison_median_agent_steps_per_second": comparison_median,
            "relative_gain": relative_gain,
            "comparison_three_run_relative_range": (
                comparison_relative_range
            ),
        },
        "decision": {
            "adopt": adopted,
            "reason": (
                "gain_exceeds_comparison_run_variability"
                if adopted
                else "gain_does_not_exceed_comparison_run_variability"
            ),
        },
        "equivalence": {
            "micro_exact_outputs": bool(
                micro["exact_output_equivalence"]["all"]
            ),
            "fixed_seed_trajectory_test": (
                "tests.test_ppo.PPOTrainerTests."
                "test_seeded_central_policy_rollout_is_reproducible"
            ),
            "illegal_input_default_contract_preserved": True,
            "checkpoint_unchanged": True,
        },
        "integrity": {
            "all_runs_meet_step_gate": True,
            "all_runs_have_three_steady_updates": True,
            "all_optional_profiling_disabled": True,
            "no_abnormal_exits": True,
            "checkpoint_sha256": sorted({
                str(run["checkpoint_sha256"]) for run in runs
            }),
        },
        "microbenchmark": {
            "path": sources["micro"]["path"],
            "sha256": sources["micro"]["sha256"],
            "profiler_batch_size": int(
                micro["profiler"]["batch_size"]
            ),
            "kernel_event_count": int(
                micro["profiler"]["trace"]["kernel_event_count"]
            ),
            "kernel_launch_event_count": int(
                micro["profiler"]["trace"][
                    "kernel_launch_event_count"
                ]
            ),
            "synchronization_event_count": int(
                micro["profiler"]["trace"][
                    "synchronization_event_count"
                ]
            ),
        },
        "sources": dict(sources),
        "passed": (
            bool(micro["passed"])
            and len(speeds) == FORMAL_RUNS
        ),
    }


def _summarize(args: argparse.Namespace) -> dict[str, object]:
    run_paths = [
        _run_path(args.run_root, run_index)
        for run_index in range(1, FORMAL_RUNS + 1)
    ]
    micro = _json(args.micro)
    report = build_report(
        [_json(path) for path in run_paths],
        micro,
        _json(args.comparison),
        sources={
            "micro": {
                "path": str(args.micro).replace("\\", "/"),
                "sha256": _sha256(args.micro),
            },
            "comparison": {
                "path": str(args.comparison).replace("\\", "/"),
                "sha256": _sha256(args.comparison),
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
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify checklist 2.6 A-NET-001"
    )
    parser.add_argument(
        "command",
        choices=("micro", "scan", "summarize"),
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--micro", type=Path, default=DEFAULT_MICRO)
    parser.add_argument("--trace-output", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "micro":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable")
        report = run_microbenchmark(
            database=args.database,
            checkpoint=args.checkpoint,
            before=args.before,
            output=args.micro,
            trace_output=args.trace_output,
            device=device,
        )
        print(json.dumps({
            "output": str(args.micro).replace("\\", "/"),
            "passed": report["passed"],
        }, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)
        return
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
    report = _summarize(args)
    print(json.dumps({
        "output": str(args.output).replace("\\", "/"),
        "passed": report["passed"],
        "adopt": report["decision"]["adopt"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
