from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Mapping

import torch

from scripts.scan_training_speed_stage_2_4 import DEFAULT_CHECKPOINT
from scripts.training_speed_stage_2_6_candidate import (
    BATCH_SIZES,
    run_microbenchmark,
    write_json,
    write_reference,
)


DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_BEFORE = Path(
    "data/reports/training_speed/v4_1_inference_breakdown.json"
)
DEFAULT_REFERENCE = Path(
    "data/reports/training_speed/"
    "stage_2_6_a_forward_001_reference.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/"
    "stage_2_6_a_forward_001_sdpa_gate.json"
)
DEFAULT_TRACE = Path(
    "data/reports/training_speed/"
    "stage_2_6_a_forward_001_sdpa_trace.json.gz"
)


def _trace_calls(
    events: list[Mapping[str, object]],
    name: str,
) -> int:
    return sum(
        1 for event in events if str(event.get("name")) == name
    )


def _sdpa_tolerance_contract(
    model: torch.nn.Module,
    inputs: Mapping[str, torch.Tensor],
) -> tuple[dict[str, object], bool]:
    with torch.no_grad():
        sdpa_outputs = model.forward_step(
            inputs["observation"],
            inputs["hidden"],
            inputs["card_indices"],
        )
        torch.backends.mha.set_fastpath_enabled(True)
        try:
            native_outputs = model.forward_step(
                inputs["observation"],
                inputs["hidden"],
                inputs["card_indices"],
            )
        finally:
            torch.backends.mha.set_fastpath_enabled(False)
    names = ("logits", "value", "hidden")
    tensors = {}
    all_close = True
    for name, native, sdpa in zip(
        names, native_outputs, sdpa_outputs
    ):
        close = torch.allclose(
            native, sdpa, rtol=1e-5, atol=1e-6
        )
        tensors[name] = {
            "max_absolute_error": float(
                (native - sdpa).abs().max()
            ),
            "allclose_rtol_1e_5_atol_1e_6": close,
        }
        all_close = all_close and close
    native_probabilities = torch.softmax(native_outputs[0], dim=-1)
    sdpa_probabilities = torch.softmax(sdpa_outputs[0], dim=-1)
    probability_max_abs = float(
        (native_probabilities - sdpa_probabilities).abs().max()
    )
    argmax_equal = torch.equal(
        native_outputs[0].argmax(dim=-1),
        sdpa_outputs[0].argmax(dim=-1),
    )
    return {
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "tensors": tensors,
        "probability_max_absolute_error": probability_max_abs,
        "argmax_equal": argmax_equal,
    }, all_close and argmax_equal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess checklist 2.6 A-FORWARD-001 SDPA"
    )
    parser.add_argument(
        "command", choices=("reference", "micro")
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-output", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.command == "reference":
        report = write_reference(
            candidate="A-FORWARD-001",
            implementation="native_mha_fastpath",
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

    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        report = run_microbenchmark(
            candidate="A-FORWARD-001",
            database=args.database,
            checkpoint=args.checkpoint,
            before=args.before,
            reference=args.reference,
            output=args.output,
            trace_output=args.trace_output,
            device=device,
            candidate_contract=_sdpa_tolerance_contract,
            require_exact_outputs=False,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
    trace_path = Path(
        str(report["profiler"]["compressed_trace_path"])
    )
    with gzip.open(trace_path, "rt", encoding="utf-8") as source:
        trace_events = json.load(source)["traceEvents"]
    operator_calls = {
        "native_multi_head_attention": _trace_calls(
            trace_events, "aten::_native_multi_head_attention"
        ),
        "scaled_dot_product_attention": _trace_calls(
            trace_events, "aten::scaled_dot_product_attention"
        ),
        "efficient_attention": _trace_calls(
            trace_events,
            "aten::_scaled_dot_product_efficient_attention",
        ),
    }
    reductions = {
        batch: float(values["relative_reduction"])
        for batch, values in report[
            "comparison_to_stage_2_3"
        ].items()
    }
    all_batches_regress = all(
        reduction < 0.0 for reduction in reductions.values()
    )
    sdpa_selected = (
        operator_calls["scaled_dot_product_attention"] > 0
        and operator_calls["native_multi_head_attention"] == 0
    )
    report["backend"] = {
        "native_fastpath_enabled": False,
        "operator_calls": operator_calls,
        "sdpa_selected": sdpa_selected,
    }
    candidate_viability_passed = bool(report["passed"])
    report["candidate_viability_passed"] = (
        candidate_viability_passed
    )
    report["decision"] = {
        "implement": False,
        "run_end_to_end": False,
        "disposition": (
            "rejected_native_sdpa_numeric_and_micro_regression"
        ),
        "all_profiled_batches_regress": all_batches_regress,
        "reason": (
            "the verified SDPA fallback exceeds the numeric tolerance "
            "and is slower at every common batch size than the current "
            "native MHA fastpath"
        ),
    }
    report["passed"] = (
        sdpa_selected
        and all_batches_regress
        and not candidate_viability_passed
    )
    write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output).replace("\\", "/"),
        "passed": report["passed"],
        "operator_calls": operator_calls,
        "all_batches_regress": all_batches_regress,
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
