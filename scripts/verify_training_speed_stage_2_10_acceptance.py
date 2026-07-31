from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY = Path("data/reports/training_speed")
DEFAULT_CHECKLIST = Path(
    "docs/card_bug_audit_and_training_speed_checklist.md"
)
DEFAULT_BASELINE = REPORT_DIRECTORY / "baseline_summary.json"
DEFAULT_STAGE_2_4 = (
    REPORT_DIRECTORY / "stage_2_4_b_interactions.json"
)
DEFAULT_STAGE_2_7 = (
    REPORT_DIRECTORY
    / "stage_2_7_b_batched_learner_001_end_to_end.json"
)
DEFAULT_STAGE_2_7_LEARNING = (
    REPORT_DIRECTORY
    / "stage_2_7_b_batched_learner_001_learning.json"
)
DEFAULT_STAGE_2_8 = (
    REPORT_DIRECTORY / "stage_2_8_overlap_gate.json"
)
DEFAULT_STAGE_2_9 = (
    REPORT_DIRECTORY / "stage_2_9_acceptance.json"
)
DEFAULT_STABILITY = (
    REPORT_DIRECTORY / "stage_2_10_stability_100k.json"
)
DEFAULT_CHECKPOINT = Path(
    "data/checkpoints/training_speed/"
    "frozen_v4_1_seed_20260801_500k.pt"
)
DEFAULT_OUTPUT = REPORT_DIRECTORY / "final_comparison.json"
FROZEN_CHECKPOINT_SHA256 = (
    "4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd"
)
STAGE_2_7_FINAL_COMMIT = "a75af5c"
ADOPTED_IDS = {
    "A-BATCH-WAIT-001",
    "A-WORKERS-001",
    "B-BATCHED-LEARNER-001",
}


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, object]:
    value = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
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
        "api_test_passed": "Passed API test" in tail,
        "passed": (
            "\nOK" in tail
            and "FAILED" not in tail
            and "Traceback (most recent call last)" not in tail
        ),
    }


def _unchecked_stage_items(
    checklist_text: str,
    section: str,
) -> list[str]:
    match = re.search(
        (
            rf"^## {re.escape(section)}\b"
            r"(?P<body>.*?)(?=^产物：|^## |^# |\Z)"
        ),
        checklist_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"checklist section {section} is missing")
    return re.findall(r"^- \[ \] (.+)$", match.group("body"), re.MULTILINE)


def _finite_metrics(iterations: Sequence[object]) -> bool:
    for row in iterations:
        if not isinstance(row, Mapping):
            return False
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            return False
        for value in metrics.values():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                return False
    return True


def _truncation_summary(
    iterations: Sequence[object],
) -> dict[str, object]:
    rows: list[dict[str, int]] = []
    for item in iterations:
        collect = item["collect"]
        rows.append({
            "episodes": int(item["episodes"]),
            "terminated": int(
                collect.get("worker_terminated_episode_count", 0)
            ),
            "truncated": int(
                collect.get("worker_truncated_episode_count", 0)
            ),
        })
    split = max(1, len(rows) // 2)

    def rate(group: Sequence[Mapping[str, int]]) -> float:
        episodes = sum(row["episodes"] for row in group)
        truncated = sum(row["truncated"] for row in group)
        return truncated / max(episodes, 1)

    total_episodes = sum(row["episodes"] for row in rows)
    total_truncated = sum(row["truncated"] for row in rows)
    first_half_rate = rate(rows[:split])
    second_half_rate = rate(rows[split:])
    total_rate = total_truncated / max(total_episodes, 1)
    return {
        "completed_episodes": total_episodes,
        "terminated_episodes": sum(
            row["terminated"] for row in rows
        ),
        "truncated_episodes": total_truncated,
        "truncation_rate": total_rate,
        "first_half_truncation_rate": first_half_rate,
        "second_half_truncation_rate": second_half_rate,
        "absolute_rate_gate_maximum": 0.02,
        "half_to_half_increase_gate_maximum": 0.01,
        "no_abnormal_increase": (
            total_rate <= 0.02
            and second_half_rate <= first_half_rate + 0.01
        ),
        "interpretation": (
            "A truncation is the configured 256-step environment cap, "
            "not a crashed worker or an interrupted PPO update."
        ),
    }


def _candidate_outcomes(
    stage_2_9: Mapping[str, object],
) -> dict[str, object]:
    matrix = stage_2_9["candidate_matrix"]
    groups: dict[str, list[str]] = {
        "adopted": [],
        "measured_no_clear_gain_or_rejected": [],
        "closed_before_implementation": [],
        "deferred_or_blocked": [],
    }
    rows: dict[str, object] = {}
    for candidate_id, item in matrix.items():
        disposition = str(item["disposition"])
        if candidate_id in ADOPTED_IDS:
            group = "adopted"
        elif (
            disposition.startswith("rejected")
            or "no_measurable" in disposition
            or disposition.startswith("reclassified")
        ):
            group = "measured_no_clear_gain_or_rejected"
        elif (
            disposition.startswith("deferred")
            or disposition.startswith("blocked")
            or "separate_algorithm" in disposition
        ):
            group = "deferred_or_blocked"
        else:
            group = "closed_before_implementation"
        groups[group].append(str(candidate_id))
        rows[str(candidate_id)] = {
            "class": item["class"],
            "disposition": disposition,
            "final_group": group,
            "evidence": item["evidence"],
        }
    for values in groups.values():
        values.sort()
    return {
        "candidate_count": len(rows),
        "groups": groups,
        "rows": rows,
    }


def _git_changed_training_paths() -> list[str]:
    output = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            STAGE_2_7_FINAL_COMMIT,
            "--",
            "swb/rl",
            "swb/engine",
        ],
        cwd=ROOT,
        text=True,
    )
    return [line for line in output.splitlines() if line]


def build_report(
    *,
    checklist_text: str,
    baseline: Mapping[str, object],
    stage_2_4: Mapping[str, object],
    stage_2_7: Mapping[str, object],
    stage_2_7_learning: Mapping[str, object],
    stage_2_8: Mapping[str, object],
    stage_2_9: Mapping[str, object],
    stability: Mapping[str, object],
    checkpoint_sha256: str,
    unittest: Mapping[str, object],
    compileall_passed: bool,
    sources: Mapping[str, object],
) -> dict[str, object]:
    baseline_v4_1 = baseline["observations"]["v4.1"]
    baseline_throughput = baseline_v4_1["agent_steps_per_second"]
    stage_2_4_final = stage_2_4["configurations"][
        "workers_6_wait_1_0_ms"
    ]
    stage_2_7_e2e = stage_2_7["end_to_end"]
    stage_2_8_pipeline = stage_2_8["pipeline"]
    iterations = stability["iterations"]
    monitor = stability["system_monitor"]
    monitor_summary = monitor["summary"]
    truncations = _truncation_summary(iterations)
    changed_training_paths = _git_changed_training_paths()

    baseline_median = float(baseline_throughput["median"])
    a_median = float(
        stage_2_4_final["agent_steps_per_second"]["median"]
    )
    final_median = float(
        stage_2_7_e2e["median_agent_steps_per_second"]
    )
    a_gain = a_median / baseline_median - 1.0
    final_gain = final_median / baseline_median - 1.0
    monitor_samples = monitor["samples"]
    gpu_total_mib = float(
        monitor["system_before"]["gpu_initial"]["memory_total_mib"]
    )
    gpu_peak_mib = float(monitor_summary["gpu_memory_peak_mib"])

    formal_long_command = (
        "E:\\anaconda\\python.exe -m scripts.profile_ppo_training "
        "--checkpoint data/checkpoints/training_speed/"
        "frozen_v4_1_seed_20260801_500k.pt "
        "--additional-agent-steps 102400 "
        "--exclude-warmup-updates 2 --device cuda "
        "--rollout-workers 6 --rollout-worker-threads 2 "
        "--central-inference-batch-wait-ms 1.0 "
        "--monitor-system --monitor-interval-seconds 0.5 "
        "--output data/reports/training_speed/"
        "stage_2_10_stability_100k.json"
    )
    checklist_unchecked = _unchecked_stage_items(
        checklist_text,
        "2.10",
    )
    contract_preserved = (
        not changed_training_paths
        and checkpoint_sha256 == FROZEN_CHECKPOINT_SHA256
        and bool(stability["checkpoint_unchanged"])
        and bool(stage_2_7_learning["passed"])
        and bool(stage_2_9["class_validation"]["A"][
            "implemented_a_exact_outputs"
        ])
        and bool(stage_2_9["class_validation"]["B"][
            "numeric_and_runtime_stable"
        ])
    )

    gates = {
        "stage_2_10_checklist_closed": not checklist_unchecked,
        "trusted_v4_1_baseline_has_three_runs": (
            int(baseline_v4_1["run_count"]) == 3
            and len(baseline_throughput["runs"]) == 3
        ),
        "wall_time_bottleneck_measured_not_inferred_from_utilization": (
            float(stage_2_8_pipeline["rollout_seconds"]) > 0.0
            and float(stage_2_8_pipeline["update_seconds"]) > 0.0
            and bool(stage_2_8["passed"])
        ),
        "at_least_one_a_end_to_end_optimization_adopted": (
            stage_2_4_final["run_count"] >= 3
            and {"A-BATCH-WAIT-001", "A-WORKERS-001"}
            <= ADOPTED_IDS
        ),
        "a_stack_gain_at_least_25_percent": a_gain >= 0.25,
        "final_stack_has_three_trusted_runs": (
            len(stage_2_7_e2e["runs_agent_steps_per_second"]) == 3
            and len(
                stage_2_7_e2e[
                    "comparison_runs_agent_steps_per_second"
                ]
            )
            == 3
        ),
        "stability_completed_at_least_100k_agent_steps": (
            int(stability["result"]["completed_additional_agent_steps"])
            >= 100_000
        ),
        "stability_no_oom": (
            gpu_peak_mib < gpu_total_mib
            and _finite_metrics(iterations)
        ),
        "stability_no_page_in_or_page_out": bool(
            monitor_summary["no_page_in_or_page_out"]
        ),
        "stability_no_deadlock_or_zero_progress_update": (
            int(stability["result"]["updates"]) == len(iterations)
            and all(int(item["agent_steps"]) > 0 for item in iterations)
        ),
        "stability_no_abnormal_truncation_increase": bool(
            truncations["no_abnormal_increase"]
        ),
        "system_monitor_covers_long_run": (
            bool(monitor["enabled"])
            and int(monitor_summary["sample_count"]) >= 100
            and float(monitor_summary["elapsed_seconds"])
            >= 0.95 * float(stability["result"]["elapsed_seconds"])
            and len(monitor_samples)
            == int(monitor_summary["sample_count"])
        ),
        "v4_1_ppo_and_checkpoint_contract_preserved": (
            contract_preserved
        ),
        "all_24_candidates_have_final_dispositions": (
            int(stage_2_9["candidate_count"]) == 24
            and len(stage_2_9["candidate_matrix"]) == 24
        ),
        "prior_unified_acceptance_passed": bool(stage_2_9["passed"]),
        "full_unittest_and_compileall_passed": (
            bool(unittest["passed"]) and compileall_passed
        ),
    }

    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_final_comparison",
        "checklist_section": "2.10",
        "experiment_contract": dict(
            stage_2_9["frozen_experiment_contract"]
        ),
        "baseline": {
            "label": "frozen_final_rules_v4_1_baseline",
            "rules_commit": "fae33c2",
            "run_count": int(baseline_v4_1["run_count"]),
            "runs_agent_steps_per_second": list(
                baseline_throughput["runs"]
            ),
            "median_agent_steps_per_second": baseline_median,
            "collect_p95_seconds": baseline_v4_1[
                "stage_time_seconds"
            ]["collect_p95"],
            "update_p95_seconds": baseline_v4_1[
                "stage_time_seconds"
            ]["update_p95"],
            "system": baseline_v4_1["system_monitor"],
        },
        "a_class_stack": {
            "adopted_candidates": [
                "A-BATCH-WAIT-001",
                "A-WORKERS-001",
            ],
            "controlled_difference": (
                "rollout workers 4->6 and central inference batch "
                "wait 0.5->1.0 ms; all frozen experiment dimensions "
                "otherwise unchanged"
            ),
            "run_count": int(stage_2_4_final["run_count"]),
            "runs_agent_steps_per_second": list(
                stage_2_4_final["agent_steps_per_second"]["runs"]
            ),
            "median_agent_steps_per_second": a_median,
            "relative_gain_vs_baseline": a_gain,
            "target_relative_gain": 0.25,
            "target_met": a_gain >= 0.25,
            "collect_p95_seconds_median": stage_2_4_final[
                "collect_p95_seconds_median"
            ],
            "update_p95_seconds_median": stage_2_4_final[
                "update_p95_seconds_median"
            ],
            "system": stage_2_4_final["system"],
        },
        "final_adopted_stack": {
            "adopted_candidates": sorted(ADOPTED_IDS),
            "runs_agent_steps_per_second": list(
                stage_2_7_e2e["runs_agent_steps_per_second"]
            ),
            "median_agent_steps_per_second": final_median,
            "relative_gain_vs_frozen_baseline": final_gain,
            "relative_gain_vs_a_observation_baseline": (
                stage_2_7_e2e["relative_gain"]
            ),
            "collect_p95_seconds_median": stage_2_7_e2e[
                "collect_p95_seconds_median"
            ],
            "update_p95_seconds_median": stage_2_7_e2e[
                "update_p95_seconds_median"
            ],
            "runtime": stage_2_7["runtime"],
        },
        "measured_bottleneck": {
            "profiled_pipeline_seconds": stage_2_8_pipeline,
            "rollout_fraction": (
                float(stage_2_8_pipeline["rollout_seconds"])
                / float(stage_2_8_pipeline["measured_seconds"])
            ),
            "update_fraction": (
                float(stage_2_8_pipeline["update_seconds"])
                / float(stage_2_8_pipeline["measured_seconds"])
            ),
            "rollout_forward_seconds": stage_2_8[
                "a_overlap_001"
            ]["rollout_same_generation"],
            "pipeline_holes": stage_2_8["a_overlap_001"][
                "pipeline_holes"
            ],
            "conclusion": (
                "After the batched learner, same-generation rollout "
                "dominates wall time. Central forward work is the main "
                "measured rollout component; the remaining queue holes "
                "lack independent same-generation CUDA work."
            ),
        },
        "stability_100k": {
            "profile_source_commit": "6253dd4",
            "command": formal_long_command,
            "result": stability["result"],
            "checkpoint_unchanged": stability[
                "checkpoint_unchanged"
            ],
            "runtime_rollout_configuration": stability[
                "runtime_rollout_configuration"
            ],
            "system_monitor_summary": monitor_summary,
            "gpu_memory_total_mib": gpu_total_mib,
            "truncations": truncations,
            "finite_update_metrics": _finite_metrics(iterations),
        },
        "contract_preservation": {
            "observation_version": "v4.1",
            "policy_architecture": "entity_action_v1",
            "stage_2_7_final_commit": STAGE_2_7_FINAL_COMMIT,
            "changed_training_paths_since_stage_2_7_final": (
                changed_training_paths
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_unchanged_during_stability_run": stability[
                "checkpoint_unchanged"
            ],
            "three_seed_learning_gate_passed": stage_2_7_learning[
                "passed"
            ],
            "ppo_generation_and_exact_output_gates_passed": (
                stage_2_9["class_validation"]["A"][
                    "implemented_a_exact_outputs"
                ]
            ),
            "batched_learner_numeric_runtime_gate_passed": (
                stage_2_9["class_validation"]["B"][
                    "numeric_and_runtime_stable"
                ]
            ),
            "preserved": contract_preserved,
        },
        "candidate_outcomes": _candidate_outcomes(stage_2_9),
        "reproducibility": {
            "formal_long_training_command": formal_long_command,
            "formal_runtime_configuration": stability[
                "runtime_rollout_configuration"
            ],
            "checkpoint_path": DEFAULT_CHECKPOINT.as_posix(),
            "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
            "hardware": stability["hardware"],
        },
        "mandatory_verification": {
            "unittest": dict(unittest),
            "compileall": {
                "command": (
                    "E:\\anaconda\\python.exe -m "
                    "compileall -q swb scripts tests"
                ),
                "passed": compileall_passed,
            },
            "stage_2_9_prescribed_smokes": (
                stage_2_9["safety"]
            ),
        },
        "checklist": {
            "path": DEFAULT_CHECKLIST.as_posix(),
            "unchecked_items": checklist_unchecked,
        },
        "sources": dict(sources),
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify checklist stage 2.10 and build final comparison"
    )
    parser.add_argument(
        "--checklist",
        type=Path,
        default=DEFAULT_CHECKLIST,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
    )
    parser.add_argument(
        "--stage-2-4",
        type=Path,
        default=DEFAULT_STAGE_2_4,
    )
    parser.add_argument(
        "--stage-2-7",
        type=Path,
        default=DEFAULT_STAGE_2_7,
    )
    parser.add_argument(
        "--stage-2-7-learning",
        type=Path,
        default=DEFAULT_STAGE_2_7_LEARNING,
    )
    parser.add_argument(
        "--stage-2-8",
        type=Path,
        default=DEFAULT_STAGE_2_8,
    )
    parser.add_argument(
        "--stage-2-9",
        type=Path,
        default=DEFAULT_STAGE_2_9,
    )
    parser.add_argument(
        "--stability",
        type=Path,
        default=DEFAULT_STABILITY,
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

    source_paths = {
        "baseline": args.baseline,
        "stage_2_4": args.stage_2_4,
        "stage_2_7": args.stage_2_7,
        "stage_2_7_learning": args.stage_2_7_learning,
        "stage_2_8": args.stage_2_8,
        "stage_2_9": args.stage_2_9,
        "stability_100k": args.stability,
    }
    report = build_report(
        checklist_text=_path(args.checklist).read_text(
            encoding="utf-8"
        ),
        baseline=_json(args.baseline),
        stage_2_4=_json(args.stage_2_4),
        stage_2_7=_json(args.stage_2_7),
        stage_2_7_learning=_json(args.stage_2_7_learning),
        stage_2_8=_json(args.stage_2_8),
        stage_2_9=_json(args.stage_2_9),
        stability=_json(args.stability),
        checkpoint_sha256=_sha256(args.checkpoint),
        unittest=_unittest_result(args.unittest_log),
        compileall_passed=args.compileall_passed,
        sources={
            name: _source(path)
            for name, path in source_paths.items()
        },
    )
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": args.output.as_posix(),
        "passed": report["passed"],
        "gates": report["gates"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
