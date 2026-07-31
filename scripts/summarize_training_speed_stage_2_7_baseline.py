from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    "data/reports/training_speed/stage_2_7_learner_baseline.json"
)
DEFAULT_COMPARISON = Path(
    "data/reports/training_speed/stage_2_5_a_obs_001.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/"
    "stage_2_7_learner_baseline_summary.json"
)


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _field(
    fields: Mapping[str, object],
    name: str,
    statistic: str,
) -> float:
    value = fields[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a summary object")
    return float(value[statistic])


def build_report(
    profile: Mapping[str, object],
    comparison: Mapping[str, object],
) -> dict[str, object]:
    runtime = profile["runtime_rollout_configuration"]
    steady = profile["steady_state"]
    update = steady["update"]
    fields = update["fields"]
    pipeline = steady["pipeline_wall_time"]
    update_wall = float(pipeline["update_seconds"])
    pipeline_wall = float(pipeline["measured_seconds"])
    forward_total = _field(
        fields, "learner_forward_seconds", "total"
    )
    backward_total = _field(
        fields, "learner_backward_seconds", "total"
    )
    padding_total = _field(
        fields, "learner_padding_and_numpy_seconds", "total"
    )
    cpu_tensor_total = _field(
        fields,
        "learner_cpu_tensor_construction_seconds",
        "total",
    )
    h2d_total = _field(
        fields, "learner_host_to_device_seconds", "total"
    )
    zero_grad_total = _field(
        fields, "learner_zero_grad_seconds", "total"
    )
    clip_total = _field(
        fields, "learner_gradient_clip_seconds", "total"
    )
    optimizer_total = _field(
        fields, "learner_optimizer_seconds", "total"
    )
    comparison_speeds = [
        float(value)
        for value in comparison["end_to_end"][
            "runs_agent_steps_per_second"
        ]
    ]
    comparison_median = sorted(comparison_speeds)[1]
    variability = (
        max(comparison_speeds) - min(comparison_speeds)
    ) / comparison_median
    preparation_total = padding_total + cpu_tensor_total + h2d_total
    optimizer_group_total = (
        zero_grad_total + clip_total + optimizer_total
    )
    configuration_passed = (
        int(runtime["rollout_workers"]) == 6
        and int(runtime["worker_torch_threads"]) == 2
        and float(
            runtime["central_inference_batch_wait_seconds"]
        ) == 0.001
        and bool(runtime["profile_learner_timing"])
    )
    passed = (
        bool(profile["checkpoint_unchanged"])
        and int(update["sample_count"]) >= 3
        and configuration_passed
    )
    return {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_7_learner_baseline_summary"
        ),
        "checklist_section": "2.7",
        "configuration": {
            "rollout_workers": int(runtime["rollout_workers"]),
            "worker_torch_threads": int(
                runtime["worker_torch_threads"]
            ),
            "central_inference_batch_wait_seconds": float(
                runtime["central_inference_batch_wait_seconds"]
            ),
            "profile_learner_timing": bool(
                runtime["profile_learner_timing"]
            ),
            "steady_state_samples": int(update["sample_count"]),
            "warmup_updates_excluded": int(
                steady["excluded_warmup_updates"]
            ),
        },
        "pipeline": {
            "measured_seconds": pipeline_wall,
            "rollout_seconds": float(pipeline["rollout_seconds"]),
            "update_seconds": update_wall,
            "learner_update_fraction": (
                update_wall / max(pipeline_wall, 1e-12)
            ),
            "profiled_accounted_fraction": float(
                steady["stage_breakdown"]["pipeline"][
                    "accounted_fraction"
                ]
            ),
        },
        "learner": {
            "update_median_seconds": _field(
                fields, "update_total_seconds", "median"
            ),
            "update_p95_seconds": _field(
                fields, "update_total_seconds", "p95"
            ),
            "forward_total_seconds": forward_total,
            "forward_fraction": (
                forward_total / max(update_wall, 1e-12)
            ),
            "backward_total_seconds": backward_total,
            "backward_fraction": (
                backward_total / max(update_wall, 1e-12)
            ),
            "forward_plus_backward_fraction": (
                (forward_total + backward_total)
                / max(update_wall, 1e-12)
            ),
            "padding_slot_fraction_median": _field(
                fields, "learner_padding_fraction", "median"
            ),
            "effective_slot_fraction_median": _field(
                fields,
                "learner_effective_token_fraction",
                "median",
            ),
            "minibatch_effective_tokens_minimum": _field(
                fields,
                "learner_minibatch_effective_tokens_min",
                "minimum",
            ),
            "minibatch_effective_tokens_p95": _field(
                fields,
                "learner_minibatch_effective_tokens_p95",
                "p95",
            ),
        },
        "materiality_gates": {
            "comparison_three_run_relative_range": variability,
            "buffer_only": {
                "components": {
                    "padding_and_numpy_seconds": padding_total,
                    "cpu_tensor_construction_seconds": (
                        cpu_tensor_total
                    ),
                    "host_to_device_seconds": h2d_total,
                },
                "impossible_zero_cost_pipeline_upper_bound": (
                    preparation_total / max(pipeline_wall, 1e-12)
                ),
                "advance_to_implementation": (
                    preparation_total / max(pipeline_wall, 1e-12)
                    > variability
                ),
            },
            "optimizer_group": {
                "components": {
                    "zero_grad_seconds": zero_grad_total,
                    "gradient_clip_seconds": clip_total,
                    "optimizer_seconds": optimizer_total,
                },
                "impossible_zero_cost_pipeline_upper_bound": (
                    optimizer_group_total
                    / max(pipeline_wall, 1e-12)
                ),
                "advance_to_implementation": (
                    optimizer_group_total
                    / max(pipeline_wall, 1e-12)
                    > variability
                ),
            },
            "padded_compute": {
                "actual_padding_slot_fraction_median": _field(
                    fields, "learner_padding_fraction", "median"
                ),
                "advance_to_microbenchmark": (
                    _field(
                        fields,
                        "learner_padding_fraction",
                        "median",
                    )
                    > variability
                ),
            },
            "learner_amp": {
                "forward_plus_backward_fraction": (
                    (forward_total + backward_total)
                    / max(update_wall, 1e-12)
                ),
                "advance_to_numeric_and_microbenchmark": (
                    (forward_total + backward_total)
                    / max(update_wall, 1e-12)
                    > variability
                ),
            },
        },
        "checkpoint": {
            "path": str(profile["checkpoint"]).replace("\\", "/"),
            "sha256": profile["checkpoint_sha256_after"],
            "unchanged": bool(profile["checkpoint_unchanged"]),
        },
        "sources": {
            "profile": {
                "path": DEFAULT_INPUT.as_posix(),
                "sha256": _sha256(DEFAULT_INPUT),
            },
            "comparison": {
                "path": DEFAULT_COMPARISON.as_posix(),
                "sha256": _sha256(DEFAULT_COMPARISON),
            },
        },
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize checklist 2.7 learner baseline"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--comparison", type=Path, default=DEFAULT_COMPARISON
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(_json(args.input), _json(args.comparison))
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": args.output.as_posix(),
        "passed": report["passed"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
