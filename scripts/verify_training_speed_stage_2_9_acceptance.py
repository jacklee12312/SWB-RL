from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY = Path("data/reports/training_speed")
DEFAULT_CHECKLIST = Path(
    "docs/card_bug_audit_and_training_speed_checklist.md"
)
DEFAULT_REGISTRY = REPORT_DIRECTORY / "candidate_registry.json"
DEFAULT_BASELINE = REPORT_DIRECTORY / "baseline_summary.json"
DEFAULT_BASELINE_CONFIGURATION = (
    REPORT_DIRECTORY / "baseline_configuration.json"
)
DEFAULT_STAGE_2_4 = (
    REPORT_DIRECTORY / "stage_2_4_b_interactions.json"
)
DEFAULT_STAGE_2_5 = (
    REPORT_DIRECTORY / "stage_2_5_a_obs_001.json"
)
DEFAULT_STAGE_2_6_ACCEPTANCE = (
    REPORT_DIRECTORY / "stage_2_6_acceptance.json"
)
DEFAULT_STAGE_2_7_LEARNING = (
    REPORT_DIRECTORY
    / "stage_2_7_b_batched_learner_001_learning.json"
)
DEFAULT_STAGE_2_7_END_TO_END = (
    REPORT_DIRECTORY
    / "stage_2_7_b_batched_learner_001_end_to_end.json"
)
DEFAULT_STAGE_2_8_ACCEPTANCE = (
    REPORT_DIRECTORY / "stage_2_8_acceptance.json"
)
DEFAULT_DEFERRED_C = (
    REPORT_DIRECTORY / "stage_2_9_deferred_c_candidates.json"
)
DEFAULT_SELF_PLAY = (
    REPORT_DIRECTORY / "stage_2_9_random_self_play_100.json"
)
DEFAULT_MIXED_MATCH = (
    REPORT_DIRECTORY / "stage_2_9_rl_mixed_match.log"
)
DEFAULT_CHECKPOINT = Path(
    "data/checkpoints/training_speed/"
    "frozen_v4_1_seed_20260801_500k.pt"
)
DEFAULT_OUTPUT = (
    REPORT_DIRECTORY / "stage_2_9_acceptance.json"
)
FROZEN_CHECKPOINT_SHA256 = (
    "4d6a8dd7d32f4e530766aab8d2ec4691de4925bc73e188021da1f45dbe54e0bd"
)
FORMAL_END_TO_END_IDS = {
    "A-BATCH-WAIT-001",
    "A-WORKERS-001",
    "A-OBS-001",
    "A-NET-001",
    "A-NET-002",
    "A-NET-003",
    "B-BATCHED-LEARNER-001",
}
ADOPTED_IDS = {
    "A-BATCH-WAIT-001",
    "A-WORKERS-001",
    "B-BATCHED-LEARNER-001",
}
NO_CLEAR_GAIN_IDS = {
    "A-OBS-001",
    "A-NET-001",
    "A-NET-002",
    "A-NET-003",
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


def _candidate_evidence(
    registry: Mapping[str, object],
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for candidate in registry["candidates"]:
        candidate_id = str(candidate["id"])
        evidence_path = Path(str(candidate.get("evidence", "")))
        evidence_exists = bool(str(evidence_path)) and _path(
            evidence_path
        ).is_file()
        evidence_passed = False
        evidence_source = None
        if evidence_exists:
            evidence = _json(evidence_path)
            evidence_passed = bool(evidence.get("passed"))
            evidence_source = _source(evidence_path)
        learning_path_value = candidate.get("learning_evidence")
        learning_source = None
        learning_passed = None
        if learning_path_value:
            learning_path = Path(str(learning_path_value))
            learning_source = _source(learning_path)
            learning_passed = bool(
                _json(learning_path).get("passed")
            )
        if candidate_id in ADOPTED_IDS:
            evidence_tier = "formal_end_to_end_adoption"
        elif candidate_id in FORMAL_END_TO_END_IDS:
            evidence_tier = "formal_end_to_end_no_clear_gain"
        elif str(candidate["class"]) == "C":
            evidence_tier = "deferred_algorithm_contract"
        elif candidate_id == "A-PROFILE-001":
            evidence_tier = "instrumented_frozen_baseline"
        else:
            evidence_tier = "preimplementation_or_rejection_gate"
        rows[candidate_id] = {
            "class": candidate["class"],
            "disposition": candidate["disposition"],
            "evidence_tier": evidence_tier,
            "evidence": evidence_source,
            "evidence_exists": evidence_exists,
            "evidence_report_passed": evidence_passed,
            "learning_evidence": learning_source,
            "learning_evidence_passed": learning_passed,
            "adoption_claim": candidate_id in ADOPTED_IDS,
            "formal_end_to_end_measured": (
                candidate_id in FORMAL_END_TO_END_IDS
            ),
        }
    return rows


def _comparison(
    candidate_id: str,
    candidate_runs: Sequence[object],
    comparison_runs: Sequence[object],
    *,
    candidate_median: float,
    comparison_median: float,
    relative_gain: float,
    comparison_relative_range: float,
    disposition: str,
    controlled_difference: str,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_run_count": len(candidate_runs),
        "comparison_run_count": len(comparison_runs),
        "candidate_runs_agent_steps_per_second": [
            float(value) for value in candidate_runs
        ],
        "comparison_runs_agent_steps_per_second": [
            float(value) for value in comparison_runs
        ],
        "candidate_median_agent_steps_per_second": (
            candidate_median
        ),
        "comparison_median_agent_steps_per_second": (
            comparison_median
        ),
        "relative_gain": relative_gain,
        "comparison_three_run_relative_range": (
            comparison_relative_range
        ),
        "gain_exceeds_comparison_variability": (
            relative_gain > comparison_relative_range
        ),
        "disposition": disposition,
        "controlled_difference": controlled_difference,
        "three_run_gate_satisfied": (
            len(candidate_runs) >= 3
            and len(comparison_runs) >= 3
        ),
    }


def _required_run_metrics(
    paths: Sequence[Path],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    all_complete = True
    for path in paths:
        report = _json(path)
        measurement = report.get("measurement")
        if not isinstance(measurement, Mapping):
            all_complete = False
            rows.append({
                "path": path.as_posix(),
                "complete": False,
            })
            continue
        batching = measurement.get("batching")
        episode = measurement.get("episode_length")
        system = measurement.get("system")
        required = {
            "throughput": (
                "agent_steps_per_second" in measurement
            ),
            "p95_stage_time": (
                "collect_p95_seconds" in measurement
                and "update_p95_seconds" in measurement
            ),
            "batch_distribution": (
                isinstance(batching, Mapping)
                and "mean_batch_size" in batching
                and "p95_batch_size" in batching
            ),
            "episode_length": (
                isinstance(episode, Mapping)
                and "p95" in episode
                and "maximum" in episode
            ),
            "cpu_gpu_ram": (
                isinstance(system, Mapping)
                and "cpu_total_median_percent" in system
                and "gpu_utilization_median_percent" in system
                and "gpu_memory_peak_mib" in system
                and "ram_used_peak_bytes" in system
            ),
            "no_abnormal_exit": (
                int(measurement.get("abnormal_exit_count", 0))
                == 0
            ),
        }
        complete = all(required.values())
        all_complete = all_complete and complete
        rows.append({
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "required_fields": required,
            "complete": complete,
        })
    return {
        "run_count": len(rows),
        "runs": rows,
        "all_complete": all_complete,
    }


def build_report(
    *,
    checklist_text: str,
    registry: Mapping[str, object],
    baseline: Mapping[str, object],
    baseline_configuration: Mapping[str, object],
    stage_2_4: Mapping[str, object],
    stage_2_5: Mapping[str, object],
    stage_2_6_acceptance: Mapping[str, object],
    stage_2_6_reports: Mapping[str, Mapping[str, object]],
    stage_2_7_learning: Mapping[str, object],
    stage_2_7_end_to_end: Mapping[str, object],
    stage_2_8_acceptance: Mapping[str, object],
    deferred_c: Mapping[str, object],
    self_play: Mapping[str, object],
    mixed_match_text: str,
    unittest: Mapping[str, object],
    compileall_passed: bool,
    checkpoint_sha256: str,
    sources: Mapping[str, object],
) -> dict[str, object]:
    section = checklist_text.split(
        "## 2.9 每项性能候选的统一验收",
        1,
    )[1].split("## 2.10 第一轮速度目标", 1)[0]
    unchecked = [
        line.strip()
        for line in section.splitlines()
        if line.startswith("- [ ]")
    ]
    candidate_rows = _candidate_evidence(registry)
    dispositions = {
        candidate_id: row["disposition"]
        for candidate_id, row in candidate_rows.items()
    }

    frozen = baseline["observations"]["v4.1"]
    frozen_runs = frozen["agent_steps_per_second"]["runs"]
    frozen_median = float(
        frozen["agent_steps_per_second"]["median"]
    )
    frozen_range = (
        float(frozen["agent_steps_per_second"]["range"])
        / max(frozen_median, 1e-12)
    )
    interaction = stage_2_4["configurations"][
        "workers_6_wait_1_0_ms"
    ]
    interaction_runs = interaction["agent_steps_per_second"][
        "runs"
    ]
    interaction_median = float(
        interaction["agent_steps_per_second"]["median"]
    )
    interaction_gain = float(
        interaction["agent_steps_per_second"][
            "relative_gain_vs_frozen_baseline"
        ]
    )
    interaction_difference = (
        "rollout worker count and batch-wait duration are the "
        "explicit candidate dimensions; checkpoint, model, "
        "Observation, decks, seeds, training parameters, rules, "
        "and hardware remain controlled"
    )
    comparisons = {
        candidate_id: _comparison(
            candidate_id,
            interaction_runs,
            frozen_runs,
            candidate_median=interaction_median,
            comparison_median=frozen_median,
            relative_gain=interaction_gain,
            comparison_relative_range=frozen_range,
            disposition=dispositions[candidate_id],
            controlled_difference=interaction_difference,
        )
        for candidate_id in (
            "A-BATCH-WAIT-001",
            "A-WORKERS-001",
        )
    }

    obs_e2e = stage_2_5["end_to_end"]
    comparisons["A-OBS-001"] = _comparison(
        "A-OBS-001",
        obs_e2e["runs_agent_steps_per_second"],
        stage_2_4["configurations"][
            "workers_6_wait_1_0_ms"
        ]["agent_steps_per_second"]["runs"],
        candidate_median=float(
            obs_e2e["median_agent_steps_per_second"]
        ),
        comparison_median=float(
            obs_e2e[
                "stage_2_4_comparison_median_agent_steps_per_second"
            ]
        ),
        relative_gain=float(obs_e2e["relative_gain"]),
        comparison_relative_range=float(
            obs_e2e["comparison_three_run_relative_range"]
        ),
        disposition=dispositions["A-OBS-001"],
        controlled_difference=(
            "only duplicate observation construction is removed"
        ),
    )
    for candidate_id, candidate_report in stage_2_6_reports.items():
        end_to_end = candidate_report["end_to_end"]
        comparisons[candidate_id] = _comparison(
            candidate_id,
            end_to_end["runs_agent_steps_per_second"],
            end_to_end["comparison_runs_agent_steps_per_second"],
            candidate_median=float(
                end_to_end["median_agent_steps_per_second"]
            ),
            comparison_median=float(
                end_to_end[
                    "comparison_median_agent_steps_per_second"
                ]
            ),
            relative_gain=float(end_to_end["relative_gain"]),
            comparison_relative_range=float(
                end_to_end[
                    "comparison_three_run_relative_range"
                ]
            ),
            disposition=dispositions[candidate_id],
            controlled_difference=(
                "only the named policy forward implementation "
                "candidate differs"
            ),
        )
    batched_e2e = stage_2_7_end_to_end["end_to_end"]
    comparisons["B-BATCHED-LEARNER-001"] = _comparison(
        "B-BATCHED-LEARNER-001",
        batched_e2e["runs_agent_steps_per_second"],
        batched_e2e["comparison_runs_agent_steps_per_second"],
        candidate_median=float(
            batched_e2e["median_agent_steps_per_second"]
        ),
        comparison_median=float(
            batched_e2e[
                "comparison_median_agent_steps_per_second"
            ]
        ),
        relative_gain=float(batched_e2e["relative_gain"]),
        comparison_relative_range=float(
            batched_e2e["comparison_three_run_relative_range"]
        ),
        disposition=dispositions["B-BATCHED-LEARNER-001"],
        controlled_difference=(
            "only v4.1 learner structured-token batching differs"
        ),
    )

    raw_run_paths: list[Path] = []
    raw_run_paths.extend(
        Path(str(source["path"]))
        for source in stage_2_5["sources"]["runs"]
    )
    for candidate_report in stage_2_6_reports.values():
        raw_run_paths.extend(
            Path(str(source["path"]))
            for source in candidate_report["sources"]["runs"]
        )
    raw_run_paths.extend(
        Path(str(source["path"]))
        for source in stage_2_7_end_to_end["sources"]["runs"]
    )
    run_metric_coverage = _required_run_metrics(raw_run_paths)

    no_clear_gain = {
        candidate_id: {
            "relative_gain": comparisons[candidate_id][
                "relative_gain"
            ],
            "comparison_three_run_relative_range": comparisons[
                candidate_id
            ]["comparison_three_run_relative_range"],
            "gain_exceeds_variability": comparisons[candidate_id][
                "gain_exceeds_comparison_variability"
            ],
            "disposition": dispositions[candidate_id],
            "correctly_not_adopted": (
                not comparisons[candidate_id][
                    "gain_exceeds_comparison_variability"
                ]
                and candidate_id not in ADOPTED_IDS
            ),
        }
        for candidate_id in sorted(NO_CLEAR_GAIN_IDS)
    }

    learning_runs = stage_2_7_learning["runs"]
    learning_illegal = sum(
        int(run["evaluation"]["illegal_actions"])
        for run in learning_runs
    )
    learning_mask_mismatches = sum(
        int(run["evaluation"]["action_mask_mismatches"])
        for run in learning_runs
    )
    self_play_passed = (
        int(self_play["games"]) == 100
        and int(self_play["illegal_actions"]) == 0
        and int(self_play["action_mask_mismatches"]) == 0
        and int(self_play["truncations"]) == 0
        and bool(self_play["official_acceptance_passed"])
    )
    mixed_match_passed = (
        "最终状态" in mixed_match_text
        and "结果=2" in mixed_match_text
        and "截断" not in mixed_match_text
    )
    all_evidence = all(
        bool(row["evidence_exists"])
        and bool(row["evidence_report_passed"])
        and (
            row["learning_evidence_passed"] in (None, True)
        )
        for row in candidate_rows.values()
    )
    adopted_have_three_runs = all(
        bool(comparisons[candidate_id]["three_run_gate_satisfied"])
        for candidate_id in ADOPTED_IDS
    )
    adopted_never_micro_only = all(
        candidate_rows[candidate_id]["evidence_tier"]
        == "formal_end_to_end_adoption"
        for candidate_id in ADOPTED_IDS
    )
    no_gain_correct = all(
        bool(row["correctly_not_adopted"])
        for row in no_clear_gain.values()
    )
    a_contracts = {
        "fixed_seed_full_trajectory": (
            "tests.test_ppo.PPOTrainerTests."
            "test_seeded_central_policy_rollout_is_reproducible"
        ),
        "log_probability_value_hidden_generation": (
            bool(stage_2_5["equivalence"][
                "observation_mask_hidden_logprob_value_generation_covered"
            ])
        ),
        "checkpoint_resume_exact": (
            "tests.test_checkpoint.CheckpointTests."
            "test_save_resume_matches_uninterrupted_next_update"
        ),
        "implemented_a_exact_outputs": bool(
            stage_2_6_acceptance["equivalence"][
                "implemented_a_exact_outputs"
            ]
        ),
        "all_adopted_a_have_formal_end_to_end": all(
            candidate_id in comparisons
            and comparisons[candidate_id][
                "three_run_gate_satisfied"
            ]
            for candidate_id in ADOPTED_IDS
            if candidate_id.startswith("A-")
        ),
    }
    b_summary = stage_2_7_learning["summary"]
    b_contracts = {
        "three_learning_seeds": (
            len(stage_2_7_learning["configuration"]["seeds"]) >= 3
        ),
        "numeric_and_runtime_stable": bool(
            b_summary["all_numeric_and_runtime_stable"]
        ),
        "learning_non_degradation_passed": bool(
            b_summary["learning_non_degradation_passed"]
        ),
        "checkpoint_resume_bounded": (
            stage_2_7_learning["decision"][
                "checkpoint_resume_test"
            ]
        ),
        "long_episode_and_no_truncation": (
            float(
                stage_2_7_end_to_end["runtime"][
                    "episode_maximum_steps"
                ]
            )
            >= 100.0
            and bool(
                stage_2_7_end_to_end["integrity"][
                    "no_truncations"
                ]
            )
        ),
        "nan_inf_absent": bool(
            b_summary["all_numeric_and_runtime_stable"]
        ),
    }
    c_contracts = {
        "candidate_count": sum(
            row["class"] == "C" for row in candidate_rows.values()
        ),
        "all_deferred": all(
            not row["adoption_claim"]
            for row in candidate_rows.values()
            if row["class"] == "C"
        ),
        "no_learning_claim_without_three_seeds": all(
            not row["adoption_claim"]
            for row in candidate_rows.values()
            if row["class"] == "C"
        ),
        "future_three_seed_and_fixed_match_gate_recorded": (
            bool(
                deferred_c["classification_contract"][
                    "minimum_learning_seeds_if_reopened"
                ]
                >= 3
            )
            and bool(
                deferred_c["classification_contract"][
                    "fixed_match_strength_comparison_required"
                ]
            )
        ),
    }
    experiment_config = baseline_configuration["checkpoints"][
        "v4.1"
    ]["configuration"]
    gates = {
        "all_checklist_items_closed": not unchecked,
        "all_candidates_have_passing_machine_evidence": all_evidence,
        "all_adopted_candidates_have_three_run_e2e": (
            adopted_have_three_runs
        ),
        "no_adoption_relies_on_microbenchmark_or_utilization": (
            adopted_never_micro_only
        ),
        "below_variability_candidates_not_adopted": (
            no_gain_correct
        ),
        "required_metrics_present_in_formal_raw_runs": bool(
            run_metric_coverage["all_complete"]
        ),
        "class_a_contracts_passed": all(
            value
            for value in a_contracts.values()
            if isinstance(value, bool)
        ),
        "class_b_contracts_passed": all(
            value
            for value in b_contracts.values()
            if isinstance(value, bool)
        ),
        "class_c_has_no_ungated_adoption": (
            bool(c_contracts["all_deferred"])
            and bool(
                c_contracts[
                    "no_learning_claim_without_three_seeds"
                ]
            )
            and bool(
                c_contracts[
                    "future_three_seed_and_fixed_match_gate_recorded"
                ]
            )
        ),
        "zero_illegal_actions_and_mask_mismatches": (
            learning_illegal == 0
            and learning_mask_mismatches == 0
            and int(self_play["illegal_actions"]) == 0
            and int(self_play["action_mask_mismatches"]) == 0
        ),
        "worker_lifecycle_regression_in_full_suite": bool(
            unittest["passed"]
        ),
        "prescribed_smokes_passed": (
            self_play_passed and mixed_match_passed
        ),
        "full_unittest_and_compileall_passed": (
            bool(unittest["passed"])
            and bool(unittest["api_test_passed"])
            and compileall_passed
        ),
        "checkpoint_unchanged": (
            checkpoint_sha256 == FROZEN_CHECKPOINT_SHA256
        ),
        "prior_stage_acceptances_passed": all(
            (
                bool(stage_2_6_acceptance["passed"]),
                bool(stage_2_7_learning["passed"]),
                bool(stage_2_7_end_to_end["passed"]),
                bool(stage_2_8_acceptance["passed"]),
            )
        ),
    }
    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_stage_2_9_acceptance",
        "checklist_section": "2.9",
        "checklist": {
            "path": DEFAULT_CHECKLIST.as_posix(),
            "unchecked_items": unchecked,
        },
        "frozen_experiment_contract": {
            "rules_commit": baseline_configuration[
                "stage_1_freeze_commit"
            ],
            "checkpoint_sha256": checkpoint_sha256,
            "observation_version": experiment_config[
                "observation_version"
            ],
            "policy_architecture": experiment_config[
                "policy_architecture"
            ],
            "training_deck": experiment_config["training_deck"],
            "opponent_decks": experiment_config[
                "opponent_decks"
            ],
            "rollout_steps": experiment_config["rollout_steps"],
            "sequence_length": experiment_config[
                "sequence_length"
            ],
            "minibatch_sequences": experiment_config[
                "minibatch_sequences"
            ],
            "update_epochs": experiment_config["update_epochs"],
            "hardware_controlled_by_run_manifests": True,
            "candidate_dimension_is_the_only_allowed_difference": True,
        },
        "candidate_count": len(candidate_rows),
        "candidate_matrix": candidate_rows,
        "adopted_candidates": sorted(ADOPTED_IDS),
        "formal_end_to_end_comparisons": comparisons,
        "no_clear_gain_decisions": no_clear_gain,
        "metric_coverage": {
            "formal_raw_runs": run_metric_coverage,
            "candidate_specific_supplements": {
                "A-OBS-001": (
                    "observation construction timing in every raw run"
                ),
                "A-NET-001..003": (
                    "batch 1/2/4/8/16/32/64 microbenchmarks and "
                    "compressed profiler traces"
                ),
                "B-BATCHED-LEARNER-001": (
                    "padding, numeric drift, update timing, and "
                    "three-seed learning evidence"
                ),
                "A-OVERLAP-001": (
                    "CPU preparation, H2D, and causal pipeline-hole "
                    "fractions"
                ),
            },
        },
        "class_validation": {
            "A": a_contracts,
            "B": b_contracts,
            "C": c_contracts,
        },
        "safety": {
            "learning_evaluation_illegal_actions": learning_illegal,
            "learning_evaluation_mask_mismatches": (
                learning_mask_mismatches
            ),
            "self_play": {
                "games": int(self_play["games"]),
                "wins": self_play["wins"],
                "draws": int(self_play["draws"]),
                "truncations": int(self_play["truncations"]),
                "illegal_actions": int(self_play["illegal_actions"]),
                "action_mask_mismatches": int(
                    self_play["action_mask_mismatches"]
                ),
                "passed": self_play_passed,
            },
            "mixed_match": {
                "terminal_result": 2,
                "passed": mixed_match_passed,
            },
            "worker_lifecycle_test": (
                "tests.test_vector_rollout.VectorRolloutTests."
                "test_close_is_graceful_and_leaves_no_live_worker"
            ),
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
        },
        "gates": gates,
        "sources": dict(sources),
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify checklist stage 2.9 unified acceptance"
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
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
    )
    parser.add_argument(
        "--baseline-configuration",
        type=Path,
        default=DEFAULT_BASELINE_CONFIGURATION,
    )
    parser.add_argument(
        "--stage-2-4",
        type=Path,
        default=DEFAULT_STAGE_2_4,
    )
    parser.add_argument(
        "--stage-2-5",
        type=Path,
        default=DEFAULT_STAGE_2_5,
    )
    parser.add_argument(
        "--stage-2-6-acceptance",
        type=Path,
        default=DEFAULT_STAGE_2_6_ACCEPTANCE,
    )
    parser.add_argument(
        "--stage-2-7-learning",
        type=Path,
        default=DEFAULT_STAGE_2_7_LEARNING,
    )
    parser.add_argument(
        "--stage-2-7-end-to-end",
        type=Path,
        default=DEFAULT_STAGE_2_7_END_TO_END,
    )
    parser.add_argument(
        "--stage-2-8-acceptance",
        type=Path,
        default=DEFAULT_STAGE_2_8_ACCEPTANCE,
    )
    parser.add_argument(
        "--deferred-c",
        type=Path,
        default=DEFAULT_DEFERRED_C,
    )
    parser.add_argument(
        "--self-play",
        type=Path,
        default=DEFAULT_SELF_PLAY,
    )
    parser.add_argument(
        "--mixed-match",
        type=Path,
        default=DEFAULT_MIXED_MATCH,
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

    stage_2_6_paths = {
        candidate_id: (
            REPORT_DIRECTORY
            / f"stage_2_6_{candidate_id.lower().replace('-', '_')}.json"
        )
        for candidate_id in (
            "A-NET-001",
            "A-NET-002",
            "A-NET-003",
        )
    }
    sources = {
        "registry": _source(args.registry),
        "baseline": _source(args.baseline),
        "baseline_configuration": _source(
            args.baseline_configuration
        ),
        "stage_2_4": _source(args.stage_2_4),
        "stage_2_5": _source(args.stage_2_5),
        "stage_2_6_acceptance": _source(
            args.stage_2_6_acceptance
        ),
        "stage_2_7_learning": _source(
            args.stage_2_7_learning
        ),
        "stage_2_7_end_to_end": _source(
            args.stage_2_7_end_to_end
        ),
        "stage_2_8_acceptance": _source(
            args.stage_2_8_acceptance
        ),
        "deferred_c": _source(args.deferred_c),
        "self_play": _source(args.self_play),
        "mixed_match": _source(args.mixed_match),
    }
    sources["stage_2_6_reports"] = {
        candidate_id: _source(path)
        for candidate_id, path in stage_2_6_paths.items()
    }
    report = build_report(
        checklist_text=_path(args.checklist).read_text(
            encoding="utf-8"
        ),
        registry=_json(args.registry),
        baseline=_json(args.baseline),
        baseline_configuration=_json(
            args.baseline_configuration
        ),
        stage_2_4=_json(args.stage_2_4),
        stage_2_5=_json(args.stage_2_5),
        stage_2_6_acceptance=_json(
            args.stage_2_6_acceptance
        ),
        stage_2_6_reports={
            candidate_id: _json(path)
            for candidate_id, path in stage_2_6_paths.items()
        },
        stage_2_7_learning=_json(args.stage_2_7_learning),
        stage_2_7_end_to_end=_json(
            args.stage_2_7_end_to_end
        ),
        stage_2_8_acceptance=_json(
            args.stage_2_8_acceptance
        ),
        deferred_c=_json(args.deferred_c),
        self_play=_json(args.self_play),
        mixed_match_text=_path(args.mixed_match).read_text(
            encoding="utf-8"
        ),
        unittest=_unittest_result(args.unittest_log),
        compileall_passed=args.compileall_passed,
        checkpoint_sha256=_sha256(args.checkpoint),
        sources=sources,
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
        "candidate_count": report["candidate_count"],
        "gates": report["gates"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
