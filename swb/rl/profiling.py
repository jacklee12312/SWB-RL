from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence


COLLECT_STAGE_FIELDS = (
    "central_rollout_startup_seconds",
    "central_collection_setup_seconds",
    "episode_dispatch_seconds",
    "central_worker_message_wait_seconds",
    "central_batch_wait_seconds",
    "central_batch_prepare_to_device_seconds",
    "central_forward_seconds",
    "central_device_to_host_and_sample_seconds",
    "central_record_packaging_seconds",
    "central_response_dispatch_seconds",
    "central_bootstrap_seconds",
    "central_episode_completion_seconds",
    "central_model_restore_seconds",
    "central_collection_finalize_seconds",
    "trajectory_conversion_seconds",
)

PROFILED_UPDATE_STAGE_FIELDS = (
    "advantages_seconds",
    "sequence_batching_seconds",
    "permutation_seconds",
    "learner_padding_and_numpy_seconds",
    "learner_cpu_tensor_construction_seconds",
    "learner_host_to_device_seconds",
    "learner_forward_seconds",
    "learner_loss_seconds",
    "learner_loss_validation_seconds",
    "learner_zero_grad_seconds",
    "learner_backward_seconds",
    "learner_gradient_clip_seconds",
    "learner_grad_norm_validation_seconds",
    "learner_optimizer_seconds",
    "parameter_validation_seconds",
    "metric_extraction_seconds",
)

COARSE_UPDATE_STAGE_FIELDS = (
    "advantages_seconds",
    "sequence_batching_seconds",
    "permutation_seconds",
    "batch_prepare_to_device_seconds",
    "forward_loss_seconds",
    "backward_clip_seconds",
    "optimizer_step_seconds",
    "parameter_validation_seconds",
    "metric_extraction_seconds",
)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0,
        min(
            len(ordered) - 1,
            math.ceil(percentile * len(ordered)) - 1,
        ),
    )
    return ordered[index]


def summarize_timing_samples(
    samples: Sequence[Mapping[str, float]],
) -> dict[str, object]:
    """Summarize flat timing/counter samples without hiding individual stages."""
    if not samples:
        return {"sample_count": 0, "fields": {}}
    keys = sorted({key for sample in samples for key in sample})
    fields: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [
            float(sample[key])
            for sample in samples
            if key in sample
        ]
        fields[key] = {
            "samples": float(len(values)),
            "total": sum(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p95": _percentile(values, 0.95),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "sample_count": len(samples),
        "fields": fields,
    }


def summarize_stage_breakdown(
    samples: Sequence[Mapping[str, float]],
    *,
    stage_fields: Sequence[str],
    wall_field: str,
    step_field: str = "records",
) -> dict[str, object]:
    wall_seconds = sum(
        float(sample.get(wall_field, 0.0))
        for sample in samples
    )
    agent_steps = sum(
        float(sample.get(step_field, 0.0))
        for sample in samples
    )
    stages: dict[str, dict[str, float]] = {}
    accounted_seconds = 0.0
    for field in stage_fields:
        if not any(field in sample for sample in samples):
            continue
        values = [
            float(sample.get(field, 0.0))
            for sample in samples
        ]
        total = sum(values)
        accounted_seconds += total
        stages[field] = {
            "total_seconds": total,
            "milliseconds_per_agent_step": (
                total * 1000.0 / agent_steps if agent_steps else 0.0
            ),
            "fraction_of_stage_wall": (
                total / wall_seconds if wall_seconds else 0.0
            ),
            "median_seconds": (
                statistics.median(values) if values else 0.0
            ),
            "p95_seconds": _percentile(values, 0.95),
        }
    unattributed_seconds = max(0.0, wall_seconds - accounted_seconds)
    accounted_fraction = (
        accounted_seconds / wall_seconds if wall_seconds else 0.0
    )
    return {
        "sample_count": len(samples),
        "wall_seconds": wall_seconds,
        "agent_steps": agent_steps,
        "accounted_seconds": accounted_seconds,
        "accounted_fraction": accounted_fraction,
        "unattributed_seconds": unattributed_seconds,
        "unattributed_fraction": (
            unattributed_seconds / wall_seconds if wall_seconds else 0.0
        ),
        "passed_90_percent": (
            bool(samples) and accounted_fraction >= 0.90
        ),
        "stages": stages,
    }


def training_timing_report(
    collect_samples: Sequence[Mapping[str, float]],
    update_samples: Sequence[Mapping[str, float]],
) -> dict[str, object]:
    if len(collect_samples) != len(update_samples):
        raise ValueError("collect and update timing sample counts must match")
    collect_summary = summarize_timing_samples(collect_samples)
    update_summary = summarize_timing_samples(update_samples)
    collect_total = sum(
        float(sample.get("collect_total_seconds", 0.0))
        for sample in collect_samples
    )
    update_total = sum(
        float(sample.get("update_total_seconds", 0.0))
        for sample in update_samples
    )
    measured_total = collect_total + update_total
    collect_breakdown = summarize_stage_breakdown(
        collect_samples,
        stage_fields=COLLECT_STAGE_FIELDS,
        wall_field="collect_total_seconds",
    )
    use_profiled_update = any(
        "learner_profiled_accounted_seconds" in sample
        for sample in update_samples
    )
    update_breakdown = summarize_stage_breakdown(
        update_samples,
        stage_fields=(
            PROFILED_UPDATE_STAGE_FIELDS
            if use_profiled_update
            else COARSE_UPDATE_STAGE_FIELDS
        ),
        wall_field="update_total_seconds",
    )
    pipeline_accounted = (
        float(collect_breakdown["accounted_seconds"])
        + float(update_breakdown["accounted_seconds"])
    )
    pipeline_accounted_fraction = (
        pipeline_accounted / measured_total if measured_total else 0.0
    )
    return {
        "sample_count": len(collect_samples),
        "pipeline_wall_time": {
            "measured_seconds": measured_total,
            "rollout_seconds": collect_total,
            "update_seconds": update_total,
            "rollout_fraction": (
                collect_total / measured_total if measured_total else 0.0
            ),
            "update_fraction": (
                update_total / measured_total if measured_total else 0.0
            ),
        },
        "collect": collect_summary,
        "update": update_summary,
        "stage_breakdown": {
            "collect": collect_breakdown,
            "update": update_breakdown,
            "pipeline": {
                "wall_seconds": measured_total,
                "accounted_seconds": pipeline_accounted,
                "accounted_fraction": pipeline_accounted_fraction,
                "unattributed_seconds": max(
                    0.0,
                    measured_total - pipeline_accounted,
                ),
                "passed_90_percent": (
                    bool(collect_samples)
                    and pipeline_accounted_fraction >= 0.90
                ),
            },
        },
    }
