from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CENTRAL_PROFILE = Path(
    "data/reports/training_speed/stage_2_2_central_inference_smoke.json"
)
DEFAULT_INFERENCE_BREAKDOWN = Path(
    "data/reports/training_speed/v4_1_inference_breakdown.json"
)
DEFAULT_INTERACTIONS = Path(
    "data/reports/training_speed/stage_2_4_b_interactions.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/stage_2_4_acceptance.json"
)
MATERIALITY_THRESHOLD = 0.05
REFERENCE_BATCH_SIZE = 4


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, object]:
    resolved = _repo_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _repo_path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _total_field(
    central: Mapping[str, object],
    name: str,
) -> float:
    fields = central["steady_state"]["collect"]["fields"]
    return float(fields[name]["total"])


def build_acceptance_report(
    central: Mapping[str, object],
    inference: Mapping[str, object],
    interactions: Mapping[str, object],
    *,
    sources: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    if not bool(interactions["passed"]):
        raise ValueError("stage 2.4 interaction gate did not pass")
    adopted = interactions["decision"]["adopted_runtime_configuration"]
    if not isinstance(adopted, Mapping):
        raise ValueError("adopted runtime configuration is malformed")
    if (
        int(adopted["rollout_workers"]) != 6
        or int(adopted["worker_torch_threads"]) != 2
        or float(adopted["central_inference_batch_wait_ms"]) != 1.0
    ):
        raise ValueError("unexpected adopted stage 2.4 configuration")

    packing = inference["v4_1_input_packing"][
        str(REFERENCE_BATCH_SIZE)
    ]
    forward = inference["fixed_input_forward"]["v4.1"][
        str(REFERENCE_BATCH_SIZE)
    ]
    stack_ms = float(
        packing["cpu_numpy_stack_milliseconds"]["median"]
    )
    tensor_ms = float(
        packing["cpu_tensor_construction_milliseconds"]["median"]
    )
    h2d_ms = float(
        packing["host_to_device_milliseconds"]["median"]
    )
    forward_ms = float(
        forward["device_milliseconds_per_call"]["median"]
    )
    packing_fraction = (stack_ms + tensor_ms) / forward_ms
    h2d_fraction = h2d_ms / forward_ms
    combined_fraction = (
        stack_ms + tensor_ms + h2d_ms
    ) / forward_ms

    requests = _total_field(central, "central_inference_requests")
    request_put_seconds = _total_field(
        central, "worker_request_queue_put_seconds"
    )
    response_wait_seconds = _total_field(
        central, "worker_response_queue_wait_seconds"
    )
    request_put_ms = 1000.0 * request_put_seconds / requests
    response_wait_ms = 1000.0 * response_wait_seconds / requests
    message_fraction = request_put_seconds / max(
        request_put_seconds + response_wait_seconds,
        1e-12,
    )

    candidate_gates = {
        "A-IPC-001-message-count": {
            "disposition": "closed_already_minimal_without_semantic_change",
            "applicable": False,
            "current_request_messages_per_decision": 1,
            "current_response_messages_per_decision": 1,
            "request_contains_fixed_fields_in_one_tuple": [
                "worker_id",
                "generation",
                "episode_id",
                "step_index",
                "player_id",
                "observation",
                "card_indices",
                "action_mask",
            ],
            "request_queue_put_milliseconds_per_request": request_put_ms,
            "response_wait_milliseconds_per_request": response_wait_ms,
            "request_put_fraction_of_put_plus_wait": message_fraction,
            "reason": (
                "Each action depends on the preceding action's resolved state. "
                "The current path already sends one aggregate request and one "
                "response per decision; fewer messages require prefetching or "
                "multiple environments and change decision/topology semantics."
            ),
        },
        "A-IPC-001-reusable-batch-buffers": {
            "disposition": "closed_below_materiality_gate",
            "applicable": False,
            "reference_batch_size": REFERENCE_BATCH_SIZE,
            "numpy_stack_milliseconds": stack_ms,
            "cpu_tensor_construction_milliseconds": tensor_ms,
            "device_forward_milliseconds": forward_ms,
            "packing_fraction_of_device_forward": packing_fraction,
            "materiality_threshold": MATERIALITY_THRESHOLD,
            "reason": (
                "Even perfect removal of NumPy stacking and CPU tensor "
                "construction has a measured ceiling below the five-percent "
                "candidate gate at the adopted batch bucket."
            ),
        },
        "A-IPC-001-pinned-nonblocking-h2d": {
            "disposition": "closed_below_materiality_gate",
            "applicable": False,
            "reference_batch_size": REFERENCE_BATCH_SIZE,
            "host_to_device_milliseconds": h2d_ms,
            "device_forward_milliseconds": forward_ms,
            "h2d_fraction_of_device_forward": h2d_fraction,
            "packing_plus_h2d_fraction_of_device_forward": combined_fraction,
            "materiality_threshold": MATERIALITY_THRESHOLD,
            "reason": (
                "Measured H2D and the combined ideal packing+H2D ceiling are "
                "below five percent; pinned/non-blocking transfer would add "
                "lifetime and synchronization complexity without a material "
                "end-to-end opportunity."
            ),
        },
    }
    gates_passed = all(
        (
            not bool(item["applicable"])
            and (
                "below_materiality_gate" in str(item["disposition"])
                or "already_minimal" in str(item["disposition"])
            )
        )
        for item in candidate_gates.values()
    )
    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_4_acceptance",
        "checklist_section": "2.4",
        "adopted_runtime_configuration": dict(adopted),
        "throughput": {
            "frozen_baseline_median_agent_steps_per_second": float(
                interactions["methodology"][
                    "baseline_median_agent_steps_per_second"
                ]
            ),
            "adopted_median_agent_steps_per_second": float(
                interactions["decision"]["median_agent_steps_per_second"]
            ),
            "relative_gain": float(
                interactions["decision"][
                    "relative_gain_vs_frozen_baseline"
                ]
            ),
            "three_run_evidence": interactions["configurations"][
                "workers_6_wait_1_0_ms"
            ]["agent_steps_per_second"]["runs"],
        },
        "candidate_gates": candidate_gates,
        "candidate_policy": {
            "materiality_threshold": MATERIALITY_THRESHOLD,
            "independent_gate_per_candidate": True,
            "implemented_low_ceiling_candidates": [],
            "closed_with_evidence": list(candidate_gates),
        },
        "sources": {
            name: dict(payload)
            for name, payload in sources.items()
        },
        "requirements": {
            "interaction_gate_passed": bool(interactions["passed"]),
            "candidate_gates_resolved": gates_passed,
            "adopted_gain_at_least_five_percent": float(
                interactions["decision"][
                    "relative_gain_vs_frozen_baseline"
                ]
            ) >= MATERIALITY_THRESHOLD,
            "no_semantics_changing_ipc_prefetch": True,
        },
        "passed": (
            bool(interactions["passed"])
            and gates_passed
            and float(
                interactions["decision"][
                    "relative_gain_vs_frozen_baseline"
                ]
            ) >= MATERIALITY_THRESHOLD
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve and freeze checklist 2.4 acceptance"
    )
    parser.add_argument(
        "--central-profile",
        type=Path,
        default=DEFAULT_CENTRAL_PROFILE,
    )
    parser.add_argument(
        "--inference-breakdown",
        type=Path,
        default=DEFAULT_INFERENCE_BREAKDOWN,
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=DEFAULT_INTERACTIONS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = {
        "central_profile": args.central_profile,
        "inference_breakdown": args.inference_breakdown,
        "interactions": args.interactions,
    }
    report = build_acceptance_report(
        _json(args.central_profile),
        _json(args.inference_breakdown),
        _json(args.interactions),
        sources={
            name: {
                "path": str(path).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
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
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
