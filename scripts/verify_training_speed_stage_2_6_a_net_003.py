from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from scripts.scan_training_speed_stage_2_4 import (
    DEFAULT_CHECKPOINT,
    FORMAL_RUNS,
    MEASURED_AGENT_STEPS,
    WARMUP_UPDATES,
    _json,
    _sha256,
    run_configuration,
)
from scripts.training_speed_stage_2_6_candidate import (
    BATCH_SIZES,
    build_report as build_candidate_report,
    run_microbenchmark,
    run_path,
    write_json,
    write_reference,
)


DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_BEFORE = Path(
    "data/reports/training_speed/v4_1_inference_breakdown.json"
)
DEFAULT_COMPARISON = Path(
    "data/reports/training_speed/stage_2_5_a_obs_001.json"
)
DEFAULT_REFERENCE = Path(
    "data/reports/training_speed/stage_2_6_a_net_003_reference.json"
)
DEFAULT_MICRO = Path(
    "data/reports/training_speed/stage_2_6_a_net_003_micro.json"
)
DEFAULT_TRACE = Path(
    "data/reports/training_speed/stage_2_6_a_net_003_trace.json.gz"
)
DEFAULT_RUN_ROOT = Path(
    "data/reports/training_speed/stage_2_6_a_net_003_runs"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/stage_2_6_a_net_003.json"
)
CONFIGURATION = {
    "id": "a_net_003",
    "dimension": "implementation_candidate",
    "rollout_workers": 6,
    "worker_torch_threads": 2,
    "central_inference_batch_wait_ms": 1.0,
}


def _operator_calls(
    report: Mapping[str, object],
    name: str,
) -> int:
    return sum(
        int(row["calls"])
        for row in report["profiler"]["top_operators"]
        if row["name"] == name
    )


def build_report(
    runs: list[Mapping[str, object]],
    micro: Mapping[str, object],
    comparison: Mapping[str, object],
    *,
    sources: Mapping[str, object],
) -> dict[str, object]:
    return build_candidate_report(
        candidate="A-NET-003",
        configuration=CONFIGURATION,
        runs=runs,
        micro=micro,
        comparison=comparison,
        sources=sources,
        equivalence={
            "arbitrary_float_categorical_test": (
                "tests.test_ppo.MaskedPolicyTests."
                "test_v4_1_categorical_merge_preserves_arbitrary_float_outputs"
            ),
        },
    )


def _summarize(args: argparse.Namespace) -> dict[str, object]:
    run_paths = [
        run_path(args.run_root, "a_net_003", run_index)
        for run_index in range(1, FORMAL_RUNS + 1)
    ]
    report = build_report(
        [_json(path) for path in run_paths],
        _json(args.micro),
        _json(args.comparison),
        sources={
            "micro": {
                "path": str(args.micro).replace("\\", "/"),
                "sha256": _sha256(args.micro),
            },
            "comparison": {
                "path": str(args.comparison).replace("\\", "/"),
                "sha256": _sha256(args.comparison),
            },
            "runs": [
                {
                    "path": str(path).replace("\\", "/"),
                    "sha256": _sha256(path),
                }
                for path in run_paths
            ],
        },
    )
    write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify checklist 2.6 A-NET-003"
    )
    parser.add_argument(
        "command",
        choices=("reference", "micro", "scan", "summarize"),
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--micro", type=Path, default=DEFAULT_MICRO)
    parser.add_argument("--trace-output", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.command == "reference":
        report = write_reference(
            candidate="A-NET-003",
            implementation="unmerged_categorical_conversions",
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
    if args.command == "micro":
        report = run_microbenchmark(
            candidate="A-NET-003",
            database=args.database,
            checkpoint=args.checkpoint,
            before=args.before,
            reference=args.reference,
            output=args.micro,
            trace_output=args.trace_output,
            device=device,
        )
        report["operator_calls"] = {
            name: _operator_calls(report, name)
            for name in ("aten::round", "aten::clamp", "aten::_to_copy")
        }
        write_json(args.micro, report)
        print(json.dumps({
            "output": str(args.micro).replace("\\", "/"),
            "passed": report["passed"],
            "operator_calls": report["operator_calls"],
        }, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(1)
        return
    if args.command == "scan":
        for run_index in range(1, FORMAL_RUNS + 1):
            path = run_path(args.run_root, "a_net_003", run_index)
            run = run_configuration(
                config=CONFIGURATION,
                run_index=run_index,
                checkpoint=args.checkpoint,
                output=path,
                measured_agent_steps=MEASURED_AGENT_STEPS,
                warmup_updates=WARMUP_UPDATES,
                monitor_interval_seconds=2.0,
                force=False,
            )
            print(json.dumps({
                "completed": str(path).replace("\\", "/"),
                "agent_steps_per_second": run["measurement"][
                    "agent_steps_per_second"
                ],
            }, sort_keys=True), flush=True)
    report = _summarize(args)
    print(json.dumps({
        "output": str(args.output).replace("\\", "/"),
        "passed": report["passed"],
        "adopt": report["decision"]["adopt"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
