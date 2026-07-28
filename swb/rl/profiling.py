from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence


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
    }
