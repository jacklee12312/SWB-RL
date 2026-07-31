from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.scan_training_speed_stage_2_4 import (
    _json,
    _repo_path,
    _sha256,
)


DEFAULT_PROFILE = Path(
    "data/reports/training_speed/v4_1_inference_breakdown.json"
)
DEFAULT_COMPARISON = Path(
    "data/reports/training_speed/stage_2_5_a_obs_001.json"
)
DEFAULT_POLICY = Path("swb/rl/policy.py")
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/"
    "stage_2_6_a_net_004_layout_projection_gate.json"
)
TARGET_OPERATORS = (
    "aten::permute",
    "aten::contiguous",
    "aten::cat",
    "aten::mm",
)


def build_report(
    profile: dict[str, object],
    comparison: dict[str, object],
    policy_text: str,
    *,
    sources: dict[str, object],
) -> dict[str, object]:
    profiler = profile["profiler"]
    iterations = int(profiler["iterations"])
    operators = {
        str(row["name"]): row
        for row in profiler["top_operators"]
    }
    forward_ms = float(
        profile["fixed_input_forward"]["v4.1"]["4"][
            "device_milliseconds_per_call"
        ]["median"]
    )
    cat_total_us = float(
        operators["aten::cat"]["device_total_microseconds"]
    )
    mm_total_us = float(
        operators["aten::mm"]["device_total_microseconds"]
    )
    generous_device_ms = (
        cat_total_us + mm_total_us
    ) / iterations / 1000.0
    generous_fraction = generous_device_ms / forward_ms
    comparison_speeds = [
        float(value)
        for value in comparison["end_to_end"][
            "runs_agent_steps_per_second"
        ]
    ]
    comparison_median = sorted(comparison_speeds)[1]
    comparison_relative_range = (
        max(comparison_speeds) - min(comparison_speeds)
    ) / comparison_median
    source_calls = {
        "permute": len(re.findall(r"\.permute\s*\(", policy_text)),
        "contiguous": len(
            re.findall(r"\.contiguous\s*\(", policy_text)
        ),
    }
    operator_contracts = {}
    for name in TARGET_OPERATORS:
        row = operators.get(name)
        operator_contracts[name] = {
            "present_in_top_50": row is not None,
            "calls_over_profile": (
                int(row["calls"]) if row is not None else 0
            ),
            "device_total_microseconds": (
                float(row["device_total_microseconds"])
                if row is not None
                else None
            ),
        }
    closed = (
        source_calls == {"permute": 0, "contiguous": 0}
        and generous_fraction < comparison_relative_range
    )
    return {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_6_a_net_004_"
            "layout_projection_gate"
        ),
        "checklist_section": "2.6",
        "candidate": "A-NET-004",
        "classification": "A",
        "methodology": {
            "profile_batch_size": int(profiler["batch_size"]),
            "profile_iterations": iterations,
            "profile_operator_limit": len(profiler["top_operators"]),
            "upper_bound_assumption": (
                "delete every aten::cat and every aten::mm, including "
                "core projections that the candidate cannot remove"
            ),
        },
        "source_layout_scan": source_calls,
        "operators": operator_contracts,
        "upper_bound": {
            "batch_4_forward_device_milliseconds": forward_ms,
            "all_cat_and_mm_device_milliseconds_per_forward": (
                generous_device_ms
            ),
            "all_cat_and_mm_fraction_of_forward": generous_fraction,
            "comparison_three_run_relative_range": (
                comparison_relative_range
            ),
            "below_variability_gate": (
                generous_fraction < comparison_relative_range
            ),
        },
        "decision": {
            "implement": False,
            "run_end_to_end": False,
            "disposition": (
                "closed_no_actionable_layout_and_"
                "below_variability_upper_bound"
            ),
            "reason": (
                "there is no explicit permute/contiguous site, and an "
                "intentionally impossible all-cat/all-mm removal remains "
                "below adjacent three-run variability"
            ),
        },
        "sources": sources,
        "passed": closed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess checklist 2.6 A-NET-004 materiality"
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--comparison", type=Path, default=DEFAULT_COMPARISON
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    policy_path = _repo_path(args.policy)
    report = build_report(
        _json(args.profile),
        _json(args.comparison),
        policy_path.read_text(encoding="utf-8"),
        sources={
            "profile": {
                "path": str(args.profile).replace("\\", "/"),
                "sha256": _sha256(args.profile),
            },
            "comparison": {
                "path": str(args.comparison).replace("\\", "/"),
                "sha256": _sha256(args.comparison),
            },
            "policy": {
                "path": str(args.policy).replace("\\", "/"),
                "sha256": _sha256(args.policy),
            },
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
        "implement": report["decision"]["implement"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
