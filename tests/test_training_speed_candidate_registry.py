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


if __name__ == "__main__":
    unittest.main()
