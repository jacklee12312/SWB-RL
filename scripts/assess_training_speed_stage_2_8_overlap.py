from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = Path(
    "data/reports/training_speed/stage_2_8_overlap_profile.json"
)
DEFAULT_ADOPTED = Path(
    "data/reports/training_speed/"
    "stage_2_7_b_batched_learner_001_end_to_end.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/stage_2_8_overlap_gate.json"
)
PROFILED_SOURCE_COMMIT = (
    "a75af5c8b84d050e5ee51544110f6422d1a0d501"
)
FROZEN_CHECKPOINT_SHA256 = (
    "4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd"
)
MATERIALITY_THRESHOLD = 0.05


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, object]:
    value = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field_total(
    fields: Mapping[str, object],
    name: str,
) -> float:
    value = fields[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a summary object")
    return float(value["total"])


def _fraction(seconds: float, wall_seconds: float) -> float:
    return seconds / max(wall_seconds, 1e-12)


def build_report(
    profile: Mapping[str, object],
    adopted: Mapping[str, object],
    *,
    sources: Mapping[str, object],
) -> dict[str, object]:
    runtime = profile["runtime_rollout_configuration"]
    steady = profile["steady_state"]
    pipeline = steady["pipeline_wall_time"]
    collect = steady["collect"]
    update = steady["update"]
    collect_fields = collect["fields"]
    update_fields = update["fields"]
    pipeline_wall = float(pipeline["measured_seconds"])
    rollout_wall = float(pipeline["rollout_seconds"])
    update_wall = float(pipeline["update_seconds"])

    rollout_prepare = _field_total(
        collect_fields,
        "central_batch_prepare_to_device_seconds",
    )
    rollout_cpu_assembly = sum(
        _field_total(collect_fields, name)
        for name in (
            "central_cpu_input_assembly_seconds",
            "central_cpu_tensor_construction_seconds",
            "central_hidden_state_assembly_seconds",
        )
    )
    rollout_h2d = _field_total(
        collect_fields,
        "central_host_to_device_seconds",
    )
    profiled_batches = _field_total(
        collect_fields,
        "central_profiled_batches",
    )
    rollout_pipeline_fraction = _fraction(
        rollout_prepare,
        pipeline_wall,
    )

    learner_padding = _field_total(
        update_fields,
        "learner_padding_and_numpy_seconds",
    )
    learner_cpu_tensor = _field_total(
        update_fields,
        "learner_cpu_tensor_construction_seconds",
    )
    learner_h2d = _field_total(
        update_fields,
        "learner_host_to_device_seconds",
    )
    learner_minibatches = _field_total(
        update_fields,
        "learner_profiled_minibatches",
    )
    learner_prepare = (
        learner_padding + learner_cpu_tensor + learner_h2d
    )
    learner_pipeline_fraction = _fraction(
        learner_prepare,
        pipeline_wall,
    )

    worker_message_wait = _field_total(
        collect_fields,
        "central_worker_message_wait_seconds",
    )
    batch_wait = _field_total(
        collect_fields,
        "central_batch_wait_seconds",
    )
    idle_wait = worker_message_wait + batch_wait
    idle_wait_fraction = _fraction(idle_wait, pipeline_wall)

    configuration_passed = (
        int(runtime["rollout_workers"]) == 6
        and int(runtime["worker_torch_threads"]) == 2
        and float(
            runtime["central_inference_batch_wait_seconds"]
        ) == 0.001
        and bool(runtime["profile_ipc_timing"])
        and bool(runtime["profile_central_timing"])
        and bool(runtime["profile_learner_timing"])
    )
    checkpoint_passed = (
        bool(profile["checkpoint_unchanged"])
        and str(profile["checkpoint_sha256_before"])
        == FROZEN_CHECKPOINT_SHA256
        and str(profile["checkpoint_sha256_after"])
        == FROZEN_CHECKPOINT_SHA256
    )
    adopted_source_passed = (
        bool(adopted["passed"])
        and bool(adopted["decision"]["adopt"])
        and bool(adopted["decision"]["default_enabled"])
    )
    profile_coverage_passed = (
        int(steady["sample_count"]) >= 3
        and int(steady["excluded_warmup_updates"]) >= 1
        and bool(
            steady["stage_breakdown"]["collect"][
                "passed_90_percent"
            ]
        )
        and bool(
            steady["stage_breakdown"]["update"][
                "passed_90_percent"
            ]
        )
    )

    rollout_advance = (
        rollout_pipeline_fraction >= MATERIALITY_THRESHOLD
    )
    learner_advance = (
        learner_pipeline_fraction >= MATERIALITY_THRESHOLD
    )
    synchronous_advance = rollout_advance or learner_advance
    passed = (
        configuration_passed
        and checkpoint_passed
        and adopted_source_passed
        and profile_coverage_passed
        and not synchronous_advance
    )

    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_8_overlap_gate",
        "checklist_section": "2.8",
        "profiled_source_commit": PROFILED_SOURCE_COMMIT,
        "methodology": {
            "diagnostic_profile_only": True,
            "materiality_threshold_fraction_of_pipeline_wall": (
                MATERIALITY_THRESHOLD
            ),
            "advance_rule": (
                "Implement synchronous overlap only when independently "
                "schedulable CPU preparation, H2D, or a pipeline hole "
                "accounts for at least 5% of steady pipeline wall time "
                "and can theoretically overlap CUDA work."
            ),
            "utilization_alone_is_not_evidence": True,
        },
        "configuration": {
            "rollout_workers": int(runtime["rollout_workers"]),
            "worker_torch_threads": int(
                runtime["worker_torch_threads"]
            ),
            "central_inference_batch_wait_seconds": float(
                runtime["central_inference_batch_wait_seconds"]
            ),
            "steady_state_samples": int(steady["sample_count"]),
            "warmup_updates_excluded": int(
                steady["excluded_warmup_updates"]
            ),
            "all_component_profilers_enabled": (
                bool(runtime["profile_ipc_timing"])
                and bool(runtime["profile_central_timing"])
                and bool(runtime["profile_learner_timing"])
            ),
        },
        "pipeline": {
            "measured_seconds": pipeline_wall,
            "rollout_seconds": rollout_wall,
            "update_seconds": update_wall,
            "agent_steps_per_second_is_profiled_diagnostic_only": float(
                profile["result"]["agent_steps_per_second"]
            ),
        },
        "a_overlap_001": {
            "class": "A",
            "rollout_same_generation": {
                "grouped_prepare_and_h2d_seconds": rollout_prepare,
                "grouped_prepare_and_h2d_fraction_of_pipeline_wall": (
                    rollout_pipeline_fraction
                ),
                "grouped_prepare_and_h2d_fraction_of_rollout_wall": (
                    _fraction(rollout_prepare, rollout_wall)
                ),
                "diagnostic_components": {
                    "cpu_assembly_seconds": rollout_cpu_assembly,
                    "host_to_device_seconds": rollout_h2d,
                    "profiled_batches": profiled_batches,
                    "host_to_device_milliseconds_per_batch": (
                        1000.0
                        * rollout_h2d
                        / max(profiled_batches, 1.0)
                    ),
                },
                "theoretically_overlap_capable_with_an_independent_batch": (
                    True
                ),
                "meets_materiality_threshold": rollout_advance,
                "advance_to_implementation": rollout_advance,
            },
            "learner_next_minibatch": {
                "preparation_and_h2d_seconds": learner_prepare,
                "preparation_and_h2d_fraction_of_pipeline_wall": (
                    learner_pipeline_fraction
                ),
                "preparation_and_h2d_fraction_of_update_wall": (
                    _fraction(learner_prepare, update_wall)
                ),
                "components": {
                    "padding_and_numpy_seconds": learner_padding,
                    "cpu_tensor_construction_seconds": (
                        learner_cpu_tensor
                    ),
                    "host_to_device_seconds": learner_h2d,
                    "profiled_minibatches": learner_minibatches,
                    "preparation_and_h2d_milliseconds_per_minibatch": (
                        1000.0
                        * learner_prepare
                        / max(learner_minibatches, 1.0)
                    ),
                },
                "theoretically_overlap_capable_with_current_cuda": True,
                "meets_materiality_threshold": learner_advance,
                "advance_to_implementation": learner_advance,
            },
            "pipeline_holes": {
                "worker_message_wait_seconds": worker_message_wait,
                "batch_formation_wait_seconds": batch_wait,
                "total_seconds": idle_wait,
                "fraction_of_pipeline_wall": idle_wait_fraction,
                "meets_size_threshold": (
                    idle_wait_fraction >= MATERIALITY_THRESHOLD
                ),
                "independently_schedulable_cuda_work_available": False,
                "advance_to_overlap_implementation": False,
                "causality": [
                    (
                        "Worker-message wait begins only when the central "
                        "request queue has no ready message."
                    ),
                    (
                        "A worker cannot emit its next observation until "
                        "it receives the current action and advances the "
                        "environment."
                    ),
                    (
                        "The multiprocessing queue already buffers other "
                        "workers' requests while the central process runs "
                        "CUDA; consuming it in another thread does not "
                        "create an independent inference batch."
                    ),
                    (
                        "The configured 1 ms batch-formation wait is an "
                        "intentional batching tradeoff; launching the "
                        "partial batch removes the wait rather than "
                        "overlapping it with another ready CUDA batch."
                    ),
                ],
            },
            "semantic_contract_if_reopened": [
                "worker weight version remains fixed for the rollout",
                "request and response ordering remains deterministic",
                "policy RNG consumption remains unchanged",
                "hidden state remains attached to its episode and player",
                "PPO generation boundaries remain unchanged",
            ],
            "decision": {
                "advance_to_implementation": synchronous_advance,
                "disposition": (
                    "advance_to_implementation"
                    if synchronous_advance
                    else "closed_below_materiality_or_not_overlapable"
                ),
                "synchronous_default_unchanged": True,
            },
        },
        "c_async_001": {
            "class": "C",
            "current_synchronous_contract": {
                "policy_generation_lag_updates": 0,
                "generation_checked_on_worker_messages": True,
                "behavior_policy_log_probability_stored_per_record": True,
                "ppo_update_starts_after_rollout_completion": True,
            },
            "required_if_reopened": {
                "trajectory_policy_generation": True,
                "maximum_policy_lag": True,
                "behavior_policy_log_probability": True,
                "ppo_ratio_and_clip_boundary_justification": True,
                "three_seed_learning_curves": True,
                "fixed_match_evaluation": True,
            },
            "decision": {
                "advance_in_stage_2_8": False,
                "disposition": "deferred_separate_algorithm_experiment",
                "reason": (
                    "Actor/learner asynchrony changes the on-policy "
                    "boundary and is not a synchronous overlap fallback."
                ),
                "synchronous_default_retained": True,
            },
        },
        "integrity": {
            "configuration_passed": configuration_passed,
            "checkpoint_passed": checkpoint_passed,
            "adopted_stage_2_7_source_passed": adopted_source_passed,
            "profile_coverage_passed": profile_coverage_passed,
        },
        "sources": dict(sources),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess checklist 2.8 overlap materiality gates"
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
    )
    parser.add_argument(
        "--adopted",
        type=Path,
        default=DEFAULT_ADOPTED,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()
    sources = {
        "profile": {
            "path": args.profile.as_posix(),
            "sha256": _sha256(args.profile),
        },
        "adopted_stage_2_7": {
            "path": args.adopted.as_posix(),
            "sha256": _sha256(args.adopted),
        },
    }
    report = build_report(
        _json(args.profile),
        _json(args.adopted),
        sources=sources,
    )
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": args.output.as_posix(),
        "passed": report["passed"],
        "rollout_fraction": report["a_overlap_001"][
            "rollout_same_generation"
        ]["grouped_prepare_and_h2d_fraction_of_pipeline_wall"],
        "learner_fraction": report["a_overlap_001"][
            "learner_next_minibatch"
        ]["preparation_and_h2d_fraction_of_pipeline_wall"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
