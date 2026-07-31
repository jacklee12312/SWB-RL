from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKLIST = Path(
    "docs/card_bug_audit_and_training_speed_checklist.md"
)
DEFAULT_REGISTRY = Path(
    "data/reports/training_speed/candidate_registry.json"
)
DEFAULT_GATE = Path(
    "data/reports/training_speed/stage_2_8_overlap_gate.json"
)
DEFAULT_CHECKPOINT = Path(
    "data/checkpoints/training_speed/"
    "frozen_v4_1_seed_20260801_500k.pt"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/stage_2_8_acceptance.json"
)
PROFILED_SOURCE_COMMIT = (
    "a75af5c8b84d050e5ee51544110f6422d1a0d501"
)
SOURCE_PATHS = (
    Path("swb/rl/policy.py"),
    Path("swb/rl/ppo.py"),
    Path("swb/rl/vector_rollout.py"),
)
EXPECTED_CHECKPOINT_SHA256 = (
    "4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd"
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


def _source(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
    }


def _candidate(
    registry: dict[str, object],
    candidate_id: str,
) -> dict[str, object]:
    return next(
        row
        for row in registry["candidates"]
        if row["id"] == candidate_id
    )


def _source_contract(path: Path) -> dict[str, object]:
    expected = subprocess.check_output(
        [
            "git",
            "show",
            f"{PROFILED_SOURCE_COMMIT}:{path.as_posix()}",
        ],
        cwd=ROOT,
    )
    actual = subprocess.check_output(
        ["git", "show", f"HEAD:{path.as_posix()}"],
        cwd=ROOT,
    )
    return {
        "profiled_commit_sha256": hashlib.sha256(
            expected
        ).hexdigest(),
        "current_sha256": hashlib.sha256(actual).hexdigest(),
        "unchanged": expected == actual,
    }


def _unittest_result(log_path: Path) -> dict[str, object]:
    text = _path(log_path).read_text(
        encoding="utf-8",
        errors="replace",
    )
    matches = list(
        re.finditer(r"Ran ([0-9,]+) tests? in ([0-9.]+)s", text)
    )
    if not matches:
        raise ValueError("unittest log lacks a final Ran line")
    match = matches[-1]
    tail = text[match.start():]
    skipped_match = re.search(r"skipped=([0-9]+)", tail)
    passed = (
        "\nOK" in tail
        and "FAILED" not in tail
        and "Traceback (most recent call last)" not in tail
    )
    return {
        "command": (
            "E:\\anaconda\\python.exe -m unittest "
            "discover -s tests -v"
        ),
        "log": log_path.as_posix(),
        "tests_run": int(match.group(1).replace(",", "")),
        "seconds": float(match.group(2)),
        "skipped": (
            int(skipped_match.group(1)) if skipped_match else 0
        ),
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify checklist stage 2.8 acceptance"
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=DEFAULT_CHECKLIST,
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=DEFAULT_GATE,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--unittest-log", type=Path, required=True)
    parser.add_argument(
        "--compileall-passed",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    checklist_text = _path(args.checklist).read_text(
        encoding="utf-8"
    )
    section = checklist_text.split(
        "## 2.8 有条件地评估流水线重叠和策略滞后",
        1,
    )[1].split("## 2.9 每项性能候选的统一验收", 1)[0]
    unchecked = [
        line.strip()
        for line in section.splitlines()
        if line.startswith("- [ ]")
    ]
    registry = _json(args.registry)
    gate = _json(args.gate)
    dispositions = {
        candidate_id: _candidate(
            registry,
            candidate_id,
        )["disposition"]
        for candidate_id in (
            "A-OVERLAP-001",
            "C-ASYNC-001",
        )
    }
    expected_dispositions = {
        "A-OVERLAP-001": (
            "closed_below_materiality_or_not_overlapable"
        ),
        "C-ASYNC-001": (
            "deferred_separate_algorithm_experiment"
        ),
    }
    source_contracts = {
        path.as_posix(): _source_contract(path)
        for path in SOURCE_PATHS
    }
    unittest = _unittest_result(args.unittest_log)
    checkpoint_sha256 = _sha256(args.checkpoint)
    overlap = gate["a_overlap_001"]
    async_gate = gate["c_async_001"]
    gates = {
        "all_checklist_items_closed": not unchecked,
        "candidate_dispositions_match": (
            dispositions == expected_dispositions
        ),
        "overlap_gate_passed": bool(gate["passed"]),
        "rollout_overlap_below_materiality": (
            not bool(
                overlap["rollout_same_generation"][
                    "advance_to_implementation"
                ]
            )
        ),
        "learner_overlap_below_materiality": (
            not bool(
                overlap["learner_next_minibatch"][
                    "advance_to_implementation"
                ]
            )
        ),
        "pipeline_hole_rejected_by_causality": (
            bool(
                overlap["pipeline_holes"]["meets_size_threshold"]
            )
            and not bool(
                overlap["pipeline_holes"][
                    "independently_schedulable_cuda_work_available"
                ]
            )
            and not bool(
                overlap["pipeline_holes"][
                    "advance_to_overlap_implementation"
                ]
            )
        ),
        "synchronous_default_retained": (
            bool(
                overlap["decision"][
                    "synchronous_default_unchanged"
                ]
            )
            and bool(
                async_gate["decision"][
                    "synchronous_default_retained"
                ]
            )
        ),
        "async_deferred_with_contract": (
            not bool(
                async_gate["decision"]["advance_in_stage_2_8"]
            )
            and all(async_gate["required_if_reopened"].values())
        ),
        "profiled_training_sources_current": all(
            row["unchanged"] for row in source_contracts.values()
        ),
        "checkpoint_unchanged": (
            checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
        ),
        "unittest_passed": bool(unittest["passed"]),
        "compileall_passed": args.compileall_passed,
    }
    report = {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_8_acceptance",
        "checklist_section": "2.8",
        "candidate_dispositions": dispositions,
        "checklist": {
            "path": args.checklist.as_posix(),
            "unchecked_items": unchecked,
        },
        "profiled_source_commit": PROFILED_SOURCE_COMMIT,
        "source_contracts": source_contracts,
        "checkpoint": {
            "path": args.checkpoint.as_posix(),
            "sha256": checkpoint_sha256,
            "unchanged_from_frozen_contract": gates[
                "checkpoint_unchanged"
            ],
        },
        "mandatory_verification": {
            "unittest": unittest,
            "compileall": {
                "command": (
                    "E:\\anaconda\\python.exe -m "
                    "compileall -q swb scripts tests"
                ),
                "passed": args.compileall_passed,
            },
        },
        "gates": gates,
        "sources": {
            "registry": _source(args.registry),
            "overlap_gate": _source(args.gate),
        },
        "passed": all(gates.values()),
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": args.output.as_posix(),
        "passed": report["passed"],
        "gates": gates,
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
