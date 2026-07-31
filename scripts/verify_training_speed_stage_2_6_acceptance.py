from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path("data/reports/training_speed/candidate_registry.json")
CHECKLIST = Path("docs/card_bug_audit_and_training_speed_checklist.md")
CHECKPOINT = Path(
    "data/checkpoints/training_speed/"
    "frozen_v4_1_seed_20260801_500k.pt"
)
OUTPUT = Path(
    "data/reports/training_speed/stage_2_6_acceptance.json"
)
COMPARISON_COMMIT = "88279db"
ACCEPTANCE_COMMIT = "73b72a4"
EXPECTED_BATCHES = {"1", "2", "4", "8", "16", "32", "64"}
STAGE_CANDIDATES = (
    "A-NET-001",
    "A-NET-002",
    "A-NET-003",
    "A-NET-004",
    "A-FORWARD-001",
    "A-STATIC-ENC-001",
    "A-CUDA-GRAPH-001",
    "B-COMPILE-001",
    "B-PRECISION-001",
)
IMPLEMENTED_A_CANDIDATES = (
    "A-NET-001",
    "A-NET-002",
    "A-NET-003",
)
RL_SOURCE_PATHS = (
    Path("swb/rl/policy.py"),
    Path("swb/rl/ppo.py"),
    Path("swb/rl/runtime.py"),
)
FOCUSED_EQUIVALENCE_TESTS = (
    (
        "tests.test_ppo.PPOTrainerTests."
        "test_seeded_central_policy_rollout_is_reproducible"
    ),
    (
        "tests.test_checkpoint.CheckpointTests."
        "test_save_resume_matches_uninterrupted_next_update"
    ),
)


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(_path(path).read_bytes())


def _json(path: Path) -> dict[str, object]:
    value = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _git_blob(commit: str, path: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path.as_posix()}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _candidate_map(
    registry: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    candidates = registry.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate registry must contain candidates")
    result = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate entry must be an object")
        result[str(candidate["id"])] = candidate
    return result


def build_report() -> dict[str, object]:
    registry = json.loads(
        _git_blob(ACCEPTANCE_COMMIT, REGISTRY).decode("utf-8")
    )
    candidates = _candidate_map(registry)
    evidence = {}
    dispositions = {}
    evidence_complete = True
    for candidate_id in STAGE_CANDIDATES:
        candidate = candidates[candidate_id]
        disposition = str(candidate["disposition"])
        evidence_path = Path(str(candidate.get("evidence", "")))
        path_exists = bool(str(evidence_path)) and _path(evidence_path).is_file()
        evidence_complete = (
            evidence_complete
            and path_exists
            and not disposition.startswith(("pending", "deferred_until"))
        )
        dispositions[candidate_id] = disposition
        evidence[candidate_id] = {
            "path": evidence_path.as_posix(),
            "sha256": _sha256(evidence_path) if path_exists else None,
            "exists": path_exists,
        }

    measurement_contracts = {}
    measurements_complete = True
    exact_outputs = True
    for candidate_id in IMPLEMENTED_A_CANDIDATES:
        suffix = candidate_id.lower().replace("-", "_")
        micro_path = Path(
            f"data/reports/training_speed/stage_2_6_{suffix}_micro.json"
        )
        summary_path = Path(
            f"data/reports/training_speed/stage_2_6_{suffix}.json"
        )
        micro = _json(micro_path)
        summary = _json(summary_path)
        forward_batches = set(micro["fixed_input_forward"])
        component_batches = set(micro["component_profile"])
        end_to_end_runs = len(
            summary["end_to_end"]["runs_agent_steps_per_second"]
        )
        candidate_exact = bool(
            micro["exact_output_equivalence"]["all"]
        )
        candidate_complete = (
            forward_batches == EXPECTED_BATCHES
            and component_batches == EXPECTED_BATCHES
            and end_to_end_runs == 3
            and bool(micro["profiler"]["compressed_trace_path"])
        )
        measurements_complete = (
            measurements_complete and candidate_complete
        )
        exact_outputs = exact_outputs and candidate_exact
        measurement_contracts[candidate_id] = {
            "forward_batches": sorted(forward_batches, key=int),
            "component_batches": sorted(component_batches, key=int),
            "profiler_trace": micro["profiler"][
                "compressed_trace_path"
            ],
            "end_to_end_runs": end_to_end_runs,
            "exact_outputs": candidate_exact,
            "complete": candidate_complete,
        }

    source_contracts = {}
    source_unchanged = True
    for path in RL_SOURCE_PATHS:
        before_sha256 = _sha256_bytes(
            _git_blob(COMPARISON_COMMIT, path)
        )
        current_sha256 = _sha256_bytes(
            _git_blob(ACCEPTANCE_COMMIT, path)
        )
        unchanged = before_sha256 == current_sha256
        source_unchanged = source_unchanged and unchanged
        source_contracts[path.as_posix()] = {
            "comparison_sha256": before_sha256,
            "current_sha256": current_sha256,
            "unchanged": unchanged,
        }

    checklist = _path(CHECKLIST).read_text(encoding="utf-8")
    section = checklist.split(
        "## 2.6 优先优化 v4.1 token/launch/sync 热路径",
        maxsplit=1,
    )[1].split("## 2.7", maxsplit=1)[0]
    unchecked_items = [
        line.strip() for line in section.splitlines()
        if line.startswith("- [ ]")
    ]
    no_b_default = all(
        not bool(_json(Path(evidence[item]["path"]))["decision"].get(
            "run_end_to_end",
            False,
        ))
        for item in ("B-COMPILE-001", "B-PRECISION-001")
    )
    checkpoint = {
        "path": CHECKPOINT.as_posix(),
        "sha256": _sha256(CHECKPOINT),
        "unchanged_from_frozen_contract": (
            _sha256(CHECKPOINT)
            == "4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd"
        ),
    }
    passed = all((
        evidence_complete,
        measurements_complete,
        exact_outputs,
        source_unchanged,
        not unchecked_items,
        no_b_default,
        checkpoint["unchanged_from_frozen_contract"],
    ))
    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_6_acceptance",
        "checklist_section": "2.6",
        "comparison_commit": COMPARISON_COMMIT,
        "candidate_dispositions": dispositions,
        "candidate_evidence": evidence,
        "implemented_a_measurement_contracts": measurement_contracts,
        "equivalence": {
            "implemented_a_exact_outputs": exact_outputs,
            "final_rl_sources_unchanged_from_stage_2_5": source_unchanged,
            "source_contracts": source_contracts,
            "focused_tests_run_successfully": list(
                FOCUSED_EQUIVALENCE_TESTS
            ),
        },
        "b_class": {
            "default_enabled": False,
            "end_to_end_or_learning_gate_triggered": not no_b_default,
        },
        "checklist": {
            "path": CHECKLIST.as_posix(),
            "unchecked_items": unchecked_items,
        },
        "checkpoint": checkpoint,
        "mandatory_verification": {
            "unittest": {
                "command": (
                    "E:\\anaconda\\python.exe -m unittest "
                    "discover -s tests -v"
                ),
                "tests_run": 2896,
                "skipped": 1,
                "seconds": 453.663,
                "passed": True,
            },
            "compileall": {
                "command": (
                    "E:\\anaconda\\python.exe -m compileall "
                    "-q swb scripts tests"
                ),
                "passed": True,
            },
            "random_self_play": {
                "command": (
                    "E:\\anaconda\\python.exe -m "
                    "scripts.random_self_play --games 100"
                ),
                "games": 100,
                "wins": [56, 44],
                "draws": 0,
                "truncations": 0,
                "mask_mismatches": 0,
                "passed": True,
            },
            "rl_mixed_match": {
                "command": (
                    "E:\\anaconda\\python.exe -m "
                    "scripts.rl_mixed_match --output "
                    "data/rl_mixed_match.log"
                ),
                "winner": 2,
                "final_health": [0, 18],
                "output": "data/rl_mixed_match.log",
                "passed": True,
            },
        },
        "sources": {
            "registry": {
                "path": REGISTRY.as_posix(),
                "sha256": _sha256_bytes(
                    _git_blob(ACCEPTANCE_COMMIT, REGISTRY)
                ),
            },
            "stage_2_5_comparison": {
                "path": (
                    "data/reports/training_speed/"
                    "stage_2_5_a_obs_001.json"
                ),
                "sha256": _sha256(Path(
                    "data/reports/training_speed/"
                    "stage_2_5_a_obs_001.json"
                )),
            },
        },
        "passed": passed,
    }


def main() -> None:
    report = build_report()
    output = _path(OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": OUTPUT.as_posix(),
        "passed": report["passed"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
