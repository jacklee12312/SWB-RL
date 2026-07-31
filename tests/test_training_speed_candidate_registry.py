from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    ROOT / "data" / "reports" / "training_speed" / "candidate_registry.json"
)


class TrainingSpeedCandidateRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        cls.candidates = {
            item["id"]: item for item in cls.registry["candidates"]
        }

    def test_candidate_ids_are_unique_and_every_candidate_is_classified(self) -> None:
        raw = self.registry["candidates"]
        self.assertEqual(len(self.candidates), len(raw))
        self.assertGreaterEqual(len(raw), 12)
        self.assertEqual({item["class"] for item in raw}, {"A", "B", "C"})
        for item in raw:
            self.assertTrue(item["description"])
            self.assertTrue(item["semantics"])
            self.assertTrue(item["disposition"])

    def test_execution_order_and_evidence_requirements_are_explicit(self) -> None:
        policy = self.registry["policy"]
        self.assertEqual(policy["execution_order"], ["A", "B", "C"])
        self.assertIn("trajectory", policy["class_a_requirement"].lower())
        self.assertIn("hidden state", policy["class_a_requirement"].lower())
        self.assertIn("three-seed", policy["class_b_requirement"].lower())
        self.assertIn("three", policy["class_c_requirement"].lower())
        self.assertIn("fixed-match", policy["class_c_requirement"].lower())

    def test_semantics_changing_candidates_are_not_mislabeled_as_a(self) -> None:
        self.assertEqual(self.candidates["B-PRECISION-001"]["class"], "B")
        self.assertEqual(self.candidates["B-COMPILE-001"]["class"], "B")
        self.assertEqual(self.candidates["C-ASYNC-001"]["class"], "C")
        self.assertEqual(
            self.candidates["C-ASYNC-001"]["disposition"],
            "deferred_separate_algorithm_experiment",
        )
        self.assertEqual(
            self.candidates["C-ASYNC-001"]["evidence"],
            "data/reports/training_speed/stage_2_8_overlap_gate.json",
        )
        self.assertEqual(self.candidates["C-HYPERPARAM-001"]["class"], "C")
        self.assertEqual(self.candidates["C-MODEL-001"]["class"], "C")

    def test_performance_commits_forbid_rule_observation_and_reward_changes(self) -> None:
        forbidden = set(self.registry["policy"]["forbidden_mixed_changes"])
        self.assertEqual(
            forbidden,
            {
                "data/rules/",
                "swb/engine/",
                "observation_semantics",
                "reward_function",
            },
        )

    def test_profile_prerequisite_has_frozen_baseline_evidence(self) -> None:
        candidate = self.candidates["A-PROFILE-001"]
        self.assertEqual(
            candidate["disposition"],
            "baseline_prerequisite",
        )
        self.assertEqual(
            candidate["evidence"],
            "data/reports/training_speed/baseline_summary.json",
        )

    def test_hyperparameters_remain_an_evidenced_c_experiment(self) -> None:
        candidate = self.candidates["C-HYPERPARAM-001"]
        self.assertEqual(candidate["class"], "C")
        self.assertEqual(
            candidate["disposition"],
            "separate_algorithm_experiment",
        )
        self.assertEqual(
            candidate["evidence"],
            "data/reports/training_speed/"
            "stage_2_9_deferred_c_candidates.json",
        )

    def test_model_changes_remain_an_evidenced_c_experiment(self) -> None:
        candidate = self.candidates["C-MODEL-001"]
        self.assertEqual(candidate["class"], "C")
        self.assertEqual(
            candidate["disposition"],
            "separate_algorithm_experiment",
        )
        self.assertEqual(
            candidate["evidence"],
            "data/reports/training_speed/"
            "stage_2_9_deferred_c_candidates.json",
        )


if __name__ == "__main__":
    unittest.main()
