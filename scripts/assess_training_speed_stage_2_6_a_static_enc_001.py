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
    "stage_2_6_a_static_enc_001_gate.json"
)


def _function_body(source: str, name: str, next_name: str) -> str:
    start = source.index(f"    def {name}(")
    end = source.index(f"    def {next_name}(", start)
    return source[start:end]


def build_report(
    profile: dict[str, object],
    comparison: dict[str, object],
    policy_text: str,
    *,
    sources: dict[str, object],
) -> dict[str, object]:
    fields = profile["v4_1_component_profile"]["4"]["fields"]
    embedding_ms = float(
        fields["card_embedding_lookup_milliseconds"]["median"]
    )
    projection_ms = float(
        fields["card_projection_milliseconds"]["median"]
    )
    forward_ms = float(
        profile["fixed_input_forward"]["v4.1"]["4"][
            "device_milliseconds_per_call"
        ]["median"]
    )
    upper_bound_ms = embedding_ms + projection_ms
    upper_bound_fraction = upper_bound_ms / forward_ms
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
    body = _function_body(
        policy_text, "_forward_step_v4_1", "_gather_entities"
    )
    calls = {
        "card_embedding": len(
            re.findall(r"self\.card_embedding\s*\(", body)
        ),
        "card_projection": len(
            re.findall(r"self\.card_projection\s*\(", body)
        ),
    }
    closed = (
        calls == {"card_embedding": 1, "card_projection": 1}
        and upper_bound_fraction < comparison_relative_range
    )
    return {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_6_a_static_enc_001_gate"
        ),
        "checklist_section": "2.6",
        "candidate": "A-STATIC-ENC-001",
        "classification": "A",
        "methodology": {
            "profile_batch_size": 4,
            "upper_bound_assumption": (
                "eliminate the complete measured card embedding lookup "
                "and projection without cache lookup or invalidation cost"
            ),
            "cache_scope_if_revisited": (
                "inference only, invalidated on every policy generation"
            ),
        },
        "same_forward_source_calls": calls,
        "upper_bound": {
            "card_embedding_lookup_milliseconds": embedding_ms,
            "card_projection_milliseconds": projection_ms,
            "combined_milliseconds_per_forward": upper_bound_ms,
            "batch_4_forward_device_milliseconds": forward_ms,
            "combined_fraction_of_forward": upper_bound_fraction,
            "comparison_three_run_relative_range": (
                comparison_relative_range
            ),
            "below_variability_gate": (
                upper_bound_fraction < comparison_relative_range
            ),
        },
        "training_constraints": {
            "embedding_trainable": True,
            "projection_trainable": True,
            "cache_must_not_enter_learner_graph": True,
            "invalidation_boundary": "every PPO policy generation",
        },
        "decision": {
            "implement": False,
            "run_end_to_end": False,
            "disposition": (
                "closed_below_materiality_and_"
                "no_repeated_forward_encoding"
            ),
            "reason": (
                "the same forward has no repeated card encoding, and "
                "zero-cost elimination of the full measured work remains "
                "below adjacent three-run variability"
            ),
        },
        "sources": sources,
        "passed": closed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess checklist 2.6 A-STATIC-ENC-001"
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
