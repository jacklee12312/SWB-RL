from __future__ import annotations

import hashlib
import json
import platform
import statistics
from pathlib import Path
from typing import Callable, Mapping

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
    FORMAL_RUNS,
    MEASURED_AGENT_STEPS,
    _json,
    _repo_path,
    _sha256,
)
from swb.db.repository import CardRepository
from swb.rl.checkpoint import load_checkpoint
from swb.rl.runtime import WorkerAssetsSnapshot


BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)


def run_path(
    run_root: Path,
    candidate_slug: str,
    run_index: int,
) -> Path:
    return run_root / f"{candidate_slug}_run_{run_index}.json"


def tensor_contract(value: torch.Tensor) -> dict[str, object]:
    cpu = value.detach().cpu().contiguous()
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
    }


def write_json(path: Path, report: Mapping[str, object]) -> None:
    destination = _repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_model(
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
    candidate: str,
    implementation: str,
    database: Path,
    checkpoint: Path,
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    checkpoint_path = _repo_path(checkpoint)
    checkpoint_before = _checkpoint_contract(checkpoint_path)
    trainer = load_model(
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
            "swb_training_speed_stage_2_6_candidate_reference"
        ),
        "candidate": candidate,
        "implementation": implementation,
        "fixture_seed": 20260801,
        "batch_size": 4,
        "outputs": {
            name: tensor_contract(value)
            for name, value in zip(
                ("logits", "value", "hidden"),
                outputs,
            )
        },
        "checkpoint": checkpoint_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
        "passed": checkpoint_before == checkpoint_after,
    }
    write_json(output, report)
    return report


def run_microbenchmark(
    *,
    candidate: str,
    database: Path,
    checkpoint: Path,
    before: Path,
    reference: Path,
    output: Path,
    trace_output: Path,
    device: torch.device,
    candidate_contract: (
        Callable[[torch.nn.Module], tuple[Mapping[str, object], bool]]
        | None
    ) = None,
) -> dict[str, object]:
    checkpoint_path = _repo_path(checkpoint)
    checkpoint_before = _checkpoint_contract(checkpoint_path)
    reference_report = _json(reference)
    trainer = load_model(
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
            name: tensor_contract(value)
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
        contract, contract_passed = (
            candidate_contract(model)
            if candidate_contract is not None
            else ({}, True)
        )
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
        "report_kind": "swb_training_speed_stage_2_6_candidate_micro",
        "checklist_section": "2.6",
        "candidate": candidate,
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
        "candidate_contract": dict(contract),
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
            and contract_passed
            and checkpoint_before == checkpoint_after
        ),
    }
    write_json(output, report)
    return report


def build_report(
    *,
    candidate: str,
    configuration: Mapping[str, object],
    runs: list[Mapping[str, object]],
    micro: Mapping[str, object],
    comparison: Mapping[str, object],
    sources: Mapping[str, object],
    equivalence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if len(runs) != FORMAL_RUNS:
        raise ValueError(f"{candidate} requires exactly three runs")
    if sorted(int(run["run_index"]) for run in runs) != [1, 2, 3]:
        raise ValueError(f"{candidate} run indexes must be 1, 2, 3")
    for run in runs:
        if run["configuration"] != configuration:
            raise ValueError(f"{candidate} runtime configuration drifted")
        measurement = run["measurement"]
        if int(measurement["agent_steps"]) < MEASURED_AGENT_STEPS:
            raise ValueError(f"{candidate} run is shorter than the step gate")
        if int(measurement["steady_update_count"]) < 3:
            raise ValueError(f"{candidate} run lacks steady updates")
        if not bool(run["profiling_switches_disabled"]):
            raise ValueError(f"{candidate} enabled optional profiling")
        if int(measurement["abnormal_exit_count"]) != 0:
            raise ValueError(f"{candidate} recorded an abnormal exit")
        if not bool(run["checkpoint_unchanged"]):
            raise ValueError(f"{candidate} changed the frozen checkpoint")
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
        "report_kind": "swb_training_speed_stage_2_6_candidate",
        "checklist_section": "2.6",
        "candidate": candidate,
        "classification": "A",
        "configuration": dict(configuration),
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
            "checkpoint_unchanged": True,
            **dict(equivalence or {}),
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
