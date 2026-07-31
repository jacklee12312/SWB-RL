from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.scan_training_speed_stage_2_4 import (
    _json,
    _repo_path,
    _sha256,
)


DEFAULT_PROFILE = Path(
    "data/reports/training_speed/v4_1_inference_breakdown.json"
)
DEFAULT_SYNC_CANDIDATE = Path(
    "data/reports/training_speed/stage_2_6_a_net_001.json"
)
DEFAULT_RUN_ROOT = Path(
    "data/reports/training_speed/stage_2_5_a_obs_001_runs"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/"
    "stage_2_6_a_cuda_graph_001_prerequisite_gate.json"
)


def _run_paths(run_root: Path) -> list[Path]:
    return [
        run_root / f"a_obs_001_run_{run_index}.json"
        for run_index in range(1, 4)
    ]


def build_report(
    profile: dict[str, object],
    sync_candidate: dict[str, object],
    runs: list[dict[str, object]],
    *,
    sources: dict[str, object],
) -> dict[str, object]:
    current_sync_events = int(
        profile["profiler"]["trace"]["synchronization_event_count"]
    )
    candidate_sync_events = int(
        sync_candidate["microbenchmark"][
            "synchronization_event_count"
        ]
    )
    sync_candidate_adopted = bool(
        sync_candidate["decision"]["adopt"]
    )
    histograms = [
        {
            str(batch): int(count)
            for batch, count in run["measurement"]["batching"][
                "histogram"
            ].items()
        }
        for run in runs
    ]
    observed_batches = sorted({
        int(batch)
        for histogram in histograms
        for batch, count in histogram.items()
        if count > 0
    })
    means = [
        float(run["measurement"]["batching"]["mean_batch_size"])
        for run in runs
    ]
    p95s = [
        float(run["measurement"]["batching"]["p95_batch_size"])
        for run in runs
    ]
    empty_fractions = [
        float(run["measurement"]["batching"]["empty_slot_fraction"])
        for run in runs
    ]
    host_sync_handled = (
        current_sync_events == 0 or sync_candidate_adopted
    )
    one_stable_bucket = len(observed_batches) == 1
    prerequisites_met = host_sync_handled and one_stable_bucket
    return {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_6_"
            "a_cuda_graph_001_prerequisite_gate"
        ),
        "checklist_section": "2.6",
        "candidate": "A-CUDA-GRAPH-001",
        "classification": "A",
        "host_sync": {
            "current_sync_events_over_three_forwards": (
                current_sync_events
            ),
            "a_net_001_sync_events_over_three_forwards": (
                candidate_sync_events
            ),
            "a_net_001_adopted": sync_candidate_adopted,
            "handled_in_current_default": host_sync_handled,
        },
        "batch_buckets": {
            "observed_batch_sizes": observed_batches,
            "mean_batch_sizes": means,
            "p95_batch_sizes": p95s,
            "empty_slot_fractions": empty_fractions,
            "histograms": histograms,
            "one_stable_bucket": one_stable_bucket,
            "dynamic_tail_requires_eager_path": True,
        },
        "prerequisites": {
            "host_sync_handled": host_sync_handled,
            "batch_bucket_stable": one_stable_bucket,
            "all_met": prerequisites_met,
        },
        "decision": {
            "implement": False,
            "run_end_to_end": False,
            "disposition": "deferred_prerequisites_unsatisfied",
            "reason": (
                "the sync-removal candidate was rejected and reverted, "
                "while live central inference spans six batch shapes; "
                "the checklist forbids CUDA Graph work before both "
                "prerequisites are satisfied"
            ),
            "reopen_when": (
                "a default sync-free forward exists and profiling proves "
                "a materially dominant stable capture bucket"
            ),
        },
        "sources": sources,
        "passed": not prerequisites_met,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess checklist 2.6 A-CUDA-GRAPH-001"
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--sync-candidate",
        type=Path,
        default=DEFAULT_SYNC_CANDIDATE,
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_paths = _run_paths(args.run_root)
    report = build_report(
        _json(args.profile),
        _json(args.sync_candidate),
        [_json(path) for path in run_paths],
        sources={
            "profile": {
                "path": str(args.profile).replace("\\", "/"),
                "sha256": _sha256(args.profile),
            },
            "sync_candidate": {
                "path": str(args.sync_candidate).replace("\\", "/"),
                "sha256": _sha256(args.sync_candidate),
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
