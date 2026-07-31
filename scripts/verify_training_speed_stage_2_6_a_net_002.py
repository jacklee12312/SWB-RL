from __future__ import annotations

import argparse
import hashlib
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
from swb.engine import observation_v4_1 as v4_1
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
DEFAULT_REFERENCE = Path(
    "data/reports/training_speed/stage_2_6_a_net_002_reference.json"
)
DEFAULT_MICRO = Path(
    "data/reports/training_speed/stage_2_6_a_net_002_micro.json"
)
DEFAULT_TRACE = Path(
    "data/reports/training_speed/stage_2_6_a_net_002_trace.json.gz"
)
DEFAULT_RUN_ROOT = Path(
    "data/reports/training_speed/stage_2_6_a_net_002_runs"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/stage_2_6_a_net_002.json"
)
CONFIGURATION = {
    "id": "a_net_002",
    "dimension": "implementation_candidate",
    "rollout_workers": 6,
    "worker_torch_threads": 2,
    "central_inference_batch_wait_ms": 1.0,
}
STATIC_BUFFERS = {
    "_v41_semantic_byte_positions": list(range(4)),
    "_v41_player_relations": [1, 2],
    "_v41_leader_positions": list(range(v4_1.LEADER_AREA_SLOTS)),
    "_v41_zone_positions": list(range(v4_1.ZONE_GROUPS)),
    "_v41_history_positions": list(range(v4_1.HISTORY_LENGTH)),
    "_v41_record_positions": list(
        range(v4_1.HISTORY_RECORDS_PER_GROUP)
    ),
    "_v41_record_groups": list(range(v4_1.RECORD_GROUPS)),
}


def _run_path(run_root: Path, run_index: int) -> Path:
    return run_root / f"a_net_002_run_{run_index}.json"


def _tensor_contract(value: torch.Tensor) -> dict[str, object]:
    cpu = value.detach().cpu().contiguous()
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
    }


def _write_json(path: Path, report: Mapping[str, object]) -> None:
    destination = _repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_model(
    *,
    database: Path,
    checkpoint: Path,
    device: torch.device,
):
    snapshot = WorkerAssetsSnapshot.build(
        CardRepository(_repo_path(database))
    )
    return load_checkpoint(
        _repo_path(checkpoint),
        snapshot,
        device=str(device),
        restore_rng_state=False,
    )


def write_reference(
    *,
    database: Path,
    checkpoint: Path,
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint_path = _repo_path(checkpoint)
    checkpoint_before = _checkpoint_contract(checkpoint_path)
    trainer = _load_model(
        database=database,
        checkpoint=checkpoint,
        device=device,
    )
    try:
        model = trainer.model
        model.eval()
        fixture = _fixed_fixture(
            model,
            maximum_batch_size=max(BATCH_SIZES),
            seed=20260801,
        )
        inputs = _device_fixture(
            fixture,
            batch_size=4,
            device=device,
        )
        with torch.no_grad():
            outputs = model.forward_step(
                inputs["observation"],
                inputs["hidden"],
                inputs["card_indices"],
            )
    finally:
        trainer.close()
    checkpoint_after = _checkpoint_contract(checkpoint_path)
    report = {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_6_a_net_002_reference"
        ),
        "candidate": "A-NET-002",
        "implementation": "dynamic_positions_before_candidate",
        "fixture_seed": 20260801,
        "batch_size": 4,
        "outputs": {
            name: _tensor_contract(value)
            for name, value in zip(
                ("logits", "value", "hidden"),
                outputs,
            )
        },
        "checkpoint": checkpoint_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
        "passed": checkpoint_before == checkpoint_after,
    }
    _write_json(output, report)
    return report


def run_microbenchmark(
    *,
    database: Path,
    checkpoint: Path,
    before: Path,
    reference: Path,
    output: Path,
    trace_output: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint_path = _repo_path(checkpoint)
    checkpoint_before = _checkpoint_contract(checkpoint_path)
    reference_report = _json(reference)
    trainer = _load_model(
        database=database,
        checkpoint=checkpoint,
        device=device,
    )
    try:
        model = trainer.model
        model.eval()
        fixture = _fixed_fixture(
            model,
            maximum_batch_size=max(BATCH_SIZES),
            seed=20260801,
        )
        inputs = _device_fixture(
            fixture,
            batch_size=4,
            device=device,
        )
        with torch.no_grad():
            outputs = model.forward_step(
                inputs["observation"],
                inputs["hidden"],
                inputs["card_indices"],
            )
        output_contracts = {
            name: _tensor_contract(value)
            for name, value in zip(
                ("logits", "value", "hidden"),
                outputs,
            )
        }
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
        buffer_contract = {}
        for name, expected in STATIC_BUFFERS.items():
            value = getattr(model, name)
            buffer_contract[name] = {
                "dtype": str(value.dtype),
                "device": str(value.device),
                "persistent": (
                    name not in model._non_persistent_buffers_set
                ),
                "values": value.detach().cpu().tolist(),
                "expected": expected,
                "matches": (
                    value.dtype == torch.long
                    and value.detach().cpu().tolist() == expected
                    and name in model._non_persistent_buffers_set
                ),
            }
    finally:
        trainer.close()

    checkpoint_after = _checkpoint_contract(checkpoint_path)
    baseline = _json(before)
    comparisons = {}
    for batch in BATCH_SIZES:
        key = str(batch)
        if key not in baseline["fixed_input_forward"]["v4.1"]:
            continue
        before_ms = float(
            baseline["fixed_input_forward"]["v4.1"][key][
                "device_milliseconds_per_call"
            ]["median"]
        )
        after_ms = float(
            forward[key]["device_milliseconds_per_call"]["median"]
        )
        comparisons[key] = {
            "before_device_milliseconds": before_ms,
            "after_device_milliseconds": after_ms,
            "relative_reduction": (before_ms - after_ms) / before_ms,
        }
    equivalence = {
        name: output_contracts[name] == expected
        for name, expected in reference_report["outputs"].items()
    }
    report = {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_6_a_net_002_micro"
        ),
        "checklist_section": "2.6",
        "candidate": "A-NET-002",
        "classification": "A",
        "methodology": {
            "fixture_seed": 20260801,
            "batch_sizes": list(BATCH_SIZES),
            "warmup_iterations": 8,
            "measured_iterations": 20,
            "repeats": 3,
            "profiler_batch_size": 4,
            "profiler_iterations": 3,
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
            **equivalence,
            "all": all(equivalence.values()),
        },
        "static_buffers": buffer_contract,
        "fixed_input_forward": forward,
        "component_profile": components,
        "profiler": profiler,
        "comparison_to_stage_2_3": comparisons,
        "sources": {
            "reference": {
                "path": str(reference).replace("\\", "/"),
                "sha256": _sha256(reference),
            },
            "before": {
                "path": str(before).replace("\\", "/"),
                "sha256": _sha256(before),
            },
        },
        "passed": (
            all(equivalence.values())
            and all(
                contract["matches"]
                for contract in buffer_contract.values()
            )
            and checkpoint_before == checkpoint_after
        ),
    }
    _write_json(output, report)
    return report


def build_report(
    runs: list[Mapping[str, object]],
    micro: Mapping[str, object],
    comparison: Mapping[str, object],
    *,
    sources: Mapping[str, object],
) -> dict[str, object]:
    if len(runs) != FORMAL_RUNS:
        raise ValueError("A-NET-002 requires exactly three runs")
    if sorted(int(run["run_index"]) for run in runs) != [1, 2, 3]:
        raise ValueError("A-NET-002 run indexes must be 1, 2, 3")
    for run in runs:
        if run["configuration"] != CONFIGURATION:
            raise ValueError("A-NET-002 runtime configuration drifted")
        measurement = run["measurement"]
        if int(measurement["agent_steps"]) < MEASURED_AGENT_STEPS:
            raise ValueError("A-NET-002 run is shorter than the step gate")
        if int(measurement["steady_update_count"]) < 3:
            raise ValueError("A-NET-002 run lacks steady updates")
        if not bool(run["profiling_switches_disabled"]):
            raise ValueError("A-NET-002 enabled optional profiling")
        if int(measurement["abnormal_exit_count"]) != 0:
            raise ValueError("A-NET-002 recorded an abnormal exit")
        if not bool(run["checkpoint_unchanged"]):
            raise ValueError("A-NET-002 changed the frozen checkpoint")
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
        "report_kind": "swb_training_speed_stage_2_6_a_net_002",
        "checklist_section": "2.6",
        "candidate": "A-NET-002",
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
            "all_buffers_non_persistent": all(
                not bool(contract["persistent"])
                for contract in micro["static_buffers"].values()
            ),
            "fixed_seed_trajectory_test": (
                "tests.test_ppo.PPOTrainerTests."
                "test_seeded_central_policy_rollout_is_reproducible"
            ),
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
        "passed": bool(micro["passed"]) and len(speeds) == FORMAL_RUNS,
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
    _write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify checklist 2.6 A-NET-002"
    )
    parser.add_argument(
        "command",
        choices=("reference", "micro", "scan", "summarize"),
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--micro", type=Path, default=DEFAULT_MICRO)
    parser.add_argument("--trace-output", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.command == "reference":
        report = write_reference(
            database=args.database,
            checkpoint=args.checkpoint,
            output=args.reference,
            device=device,
        )
        print(json.dumps({
            "output": str(args.reference).replace("\\", "/"),
            "passed": report["passed"],
        }, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)
        return
    if args.command == "micro":
        report = run_microbenchmark(
            database=args.database,
            checkpoint=args.checkpoint,
            before=args.before,
            reference=args.reference,
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
