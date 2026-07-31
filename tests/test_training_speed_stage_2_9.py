from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path


class TrainingSpeedStage29AcceptanceTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REGISTRY_COMMIT = (
        "83b2d16184ea0055fc73ddbca46aa94c89ab9733"
    )
    REPORT = (
        ROOT
        / "data/reports/training_speed/stage_2_9_acceptance.json"
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            cls.REPORT.read_text(encoding="utf-8")
        )

    def test_every_candidate_has_passing_machine_evidence(
        self,
    ) -> None:
        report = self.report
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(report["candidate_count"], 24)
        self.assertEqual(
            len(report["candidate_matrix"]),
            report["candidate_count"],
        )
        for row in report["candidate_matrix"].values():
            self.assertTrue(row["evidence_exists"])
            self.assertTrue(row["evidence_report_passed"])
            self.assertIn(
                row["learning_evidence_passed"],
                (None, True),
            )

    def test_adoptions_have_three_run_e2e_not_micro_only(
        self,
    ) -> None:
        report = self.report
        self.assertEqual(
            report["adopted_candidates"],
            [
                "A-BATCH-WAIT-001",
                "A-WORKERS-001",
                "B-BATCHED-LEARNER-001",
            ],
        )
        for candidate_id in report["adopted_candidates"]:
            comparison = report[
                "formal_end_to_end_comparisons"
            ][candidate_id]
            self.assertGreaterEqual(
                comparison["candidate_run_count"],
                3,
            )
            self.assertGreaterEqual(
                comparison["comparison_run_count"],
                3,
            )
            self.assertEqual(
                report["candidate_matrix"][candidate_id][
                    "evidence_tier"
                ],
                "formal_end_to_end_adoption",
            )

    def test_below_variability_candidates_are_not_adopted(
        self,
    ) -> None:
        decisions = self.report["no_clear_gain_decisions"]
        self.assertEqual(
            set(decisions),
            {
                "A-OBS-001",
                "A-NET-001",
                "A-NET-002",
                "A-NET-003",
            },
        )
        for row in decisions.values():
            self.assertFalse(row["gain_exceeds_variability"])
            self.assertTrue(row["correctly_not_adopted"])

    def test_class_and_safety_contracts_are_closed(self) -> None:
        validation = self.report["class_validation"]
        self.assertTrue(
            validation["A"]["all_adopted_a_have_formal_end_to_end"]
        )
        self.assertTrue(
            validation["A"]["implemented_a_exact_outputs"]
        )
        self.assertTrue(
            validation["B"]["three_learning_seeds"]
        )
        self.assertTrue(
            validation["B"]["numeric_and_runtime_stable"]
        )
        self.assertTrue(
            validation["B"]["learning_non_degradation_passed"]
        )
        self.assertTrue(validation["C"]["all_deferred"])
        self.assertTrue(
            validation["C"][
                "future_three_seed_and_fixed_match_gate_recorded"
            ]
        )
        safety = self.report["safety"]
        self.assertEqual(
            safety["learning_evaluation_illegal_actions"],
            0,
        )
        self.assertEqual(
            safety["learning_evaluation_mask_mismatches"],
            0,
        )
        self.assertTrue(safety["self_play"]["passed"])
        self.assertTrue(safety["mixed_match"]["passed"])

    def test_saved_sources_are_frozen_and_current(self) -> None:
        sources = self.report["sources"]
        for name, source in sources.items():
            if name == "stage_2_6_reports":
                nested_sources = source.values()
                for nested in nested_sources:
                    content = (
                        self.ROOT / nested["path"]
                    ).read_bytes()
                    self.assertEqual(
                        hashlib.sha256(content).hexdigest(),
                        nested["sha256"],
                    )
                continue
            if name == "registry":
                content = subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{self.REGISTRY_COMMIT}:{source['path']}",
                    ],
                    cwd=self.ROOT,
                )
            else:
                content = (
                    self.ROOT / source["path"]
                ).read_bytes()
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                source["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
