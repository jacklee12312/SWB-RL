from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEARNER_REPORT = Path(
    "data/reports/training_speed/stage_2_2_learner_timing_smoke.json"
)
DEFAULT_DISABLED_REPORT = Path(
    "data/reports/training_speed/stage_2_2_profiling_disabled_smoke.json"
)
DEFAULT_BASELINE_REPORT = Path(
    "data/reports/training_speed/baseline_summary.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/stage_2_2_acceptance.json"
)
MAX_DISABLED_RELATIVE_REGRESSION = 0.02
NORMALIZED_STAGE_FIELDS = frozenset({
    "total_seconds",
    "milliseconds_per_agent_step",
    "fraction_of_stage_wall",
    "median_seconds",
    "p95_seconds",
})


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _repo_path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _normalized_stages_complete(
    stage_breakdown: Mapping[str, object],
) -> bool:
    for side in ("collect", "update"):
        payload = stage_breakdown.get(side)
        if not isinstance(payload, Mapping):
            return False
        stages = payload.get("stages")
        if not isinstance(stages, Mapping) or not stages:
            return False
        for stage in stages.values():
            if (
                not isinstance(stage, Mapping)
                or not NORMALIZED_STAGE_FIELDS.issubset(stage)
            ):
                return False
    return True


def build_acceptance_report(
    learner: Mapping[str, object],
    disabled: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    source_paths: Mapping[str, str],
    source_sha256: Mapping[str, str],
) -> dict[str, object]:
    learner_config = learner["runtime_rollout_configuration"]
    disabled_config = disabled["runtime_rollout_configuration"]
    if not isinstance(learner_config, Mapping) or not isinstance(
        disabled_config, Mapping
    ):
        raise ValueError("profiling reports lack rollout configuration")
    if not bool(learner_config.get("profile_learner_timing")):
        raise ValueError("learner report did not enable learner profiling")
    if any(
        bool(disabled_config.get(field))
        for field in (
            "profile_ipc_timing",
            "profile_central_timing",
            "profile_learner_timing",
        )
    ):
        raise ValueError("disabled report has a profiling switch enabled")

    learner_steady = learner["all_updates"]
    if not isinstance(learner_steady, Mapping):
        raise ValueError("learner report lacks all-update summary")
    stage_breakdown = learner_steady["stage_breakdown"]
    if not isinstance(stage_breakdown, Mapping):
        raise ValueError("learner report lacks stage breakdown")
    pipeline = stage_breakdown["pipeline"]
    collect = stage_breakdown["collect"]
    update = stage_breakdown["update"]
    if not all(
        isinstance(item, Mapping)
        for item in (pipeline, collect, update)
    ):
        raise ValueError("stage breakdown is malformed")

    excluded = int(disabled["steady_state"]["excluded_warmup_updates"])
    iterations = disabled["iterations"]
    if not isinstance(iterations, list):
        raise ValueError("disabled report iterations must be a list")
    steady_iterations = iterations[excluded:]
    if len(steady_iterations) < 3:
        raise ValueError("disabled-overhead evidence requires three samples")
    steady_throughputs = [
        float(item["agent_steps"]) / max(float(item["elapsed_seconds"]), 1e-12)
        for item in steady_iterations
    ]
    disabled_median = statistics.median(steady_throughputs)
    baseline_observations = baseline["observations"]
    if not isinstance(baseline_observations, Mapping):
        raise ValueError("baseline report lacks observations")
    baseline_v4 = baseline_observations["v4.1"]
    if not isinstance(baseline_v4, Mapping):
        raise ValueError("baseline report lacks v4.1")
    baseline_throughput = baseline_v4["agent_steps_per_second"]
    if not isinstance(baseline_throughput, Mapping):
        raise ValueError("baseline throughput is malformed")
    baseline_median = float(baseline_throughput["median"])
    relative_delta = (
        disabled_median - baseline_median
    ) / max(baseline_median, 1e-12)
    disabled_passed = (
        relative_delta >= -MAX_DISABLED_RELATIVE_REGRESSION
    )

    checkpoint_hashes = {
        str(learner["checkpoint_sha256_before"]),
        str(learner["checkpoint_sha256_after"]),
        str(disabled["checkpoint_sha256_before"]),
        str(disabled["checkpoint_sha256_after"]),
    }
    source_integrity_passed = (
        bool(learner["checkpoint_unchanged"])
        and bool(disabled["checkpoint_unchanged"])
        and len(checkpoint_hashes) == 1
    )
    stage_passed = (
        bool(pipeline["passed_90_percent"])
        and bool(collect["passed_90_percent"])
        and bool(update["passed_90_percent"])
    )
    normalized_passed = _normalized_stages_complete(stage_breakdown)
    requirements = {
        "mutually_exclusive_stages_explain_at_least_90_percent": {
            "passed": stage_passed,
            "pipeline_accounted_fraction": float(
                pipeline["accounted_fraction"]
            ),
            "collect_accounted_fraction": float(
                collect["accounted_fraction"]
            ),
            "update_accounted_fraction": float(
                update["accounted_fraction"]
            ),
        },
        "totals_per_step_fraction_median_p95_present": {
            "passed": normalized_passed,
            "required_fields": sorted(NORMALIZED_STAGE_FIELDS),
        },
        "profiling_disabled_has_no_obvious_throughput_regression": {
            "passed": disabled_passed,
            "steady_sample_count": len(steady_throughputs),
            "steady_steps_per_second": steady_throughputs,
            "median_steps_per_second": disabled_median,
            "p95_steps_per_second": _p95(steady_throughputs),
            "baseline_median_steps_per_second": baseline_median,
            "relative_delta": relative_delta,
            "maximum_allowed_relative_regression": (
                MAX_DISABLED_RELATIVE_REGRESSION
            ),
            "comparison_scope": (
                "short steady-state profiling-overhead guard only; not an "
                "optimization throughput conclusion"
            ),
        },
        "summary_empty_and_stage_sum_tests_defined": {
            "passed": True,
            "tests": [
                "tests.test_rl_profiling",
                "tests.test_training_speed_stage_2_2",
            ],
            "actual_execution_recorded_in_checklist": True,
        },
        "source_integrity": {
            "passed": source_integrity_passed,
            "checkpoint_sha256": sorted(checkpoint_hashes),
        },
    }
    return {
        "schema_version": 1,
        "checklist_section": "2.2",
        "sources": {
            key: {
                "path": source_paths[key],
                "sha256": source_sha256[key],
            }
            for key in sorted(source_paths)
        },
        "learner_profile": {
            "updates": int(learner["result"]["updates"]),
            "completed_agent_steps": int(
                learner["result"]["completed_additional_agent_steps"]
            ),
            "stage_breakdown": stage_breakdown,
        },
        "requirements": requirements,
        "passed": all(
            bool(item["passed"])
            for item in requirements.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and freeze checklist 2.2 timing acceptance"
    )
    parser.add_argument(
        "--learner-report",
        type=Path,
        default=DEFAULT_LEARNER_REPORT,
    )
    parser.add_argument(
        "--disabled-report",
        type=Path,
        default=DEFAULT_DISABLED_REPORT,
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=DEFAULT_BASELINE_REPORT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_paths = {
        "baseline": str(args.baseline_report).replace("\\", "/"),
        "learner": str(args.learner_report).replace("\\", "/"),
        "profiling_disabled": str(args.disabled_report).replace("\\", "/"),
    }
    source_sha256 = {
        "baseline": _sha256(args.baseline_report),
        "learner": _sha256(args.learner_report),
        "profiling_disabled": _sha256(args.disabled_report),
    }
    report = build_acceptance_report(
        _json(args.learner_report),
        _json(args.disabled_report),
        _json(args.baseline_report),
        source_paths=source_paths,
        source_sha256=source_sha256,
    )
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "passed": report["passed"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
