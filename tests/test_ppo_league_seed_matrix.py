from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.report_ppo_league_seed_matrix import (
    OUTPUT_DIRECTORY,
    build_outputs,
    build_seed_payoff_matrix,
    find_cycles,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY = ROOT / OUTPUT_DIRECTORY


class PPOLeagueSeedMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report, cls.close_plan = build_seed_payoff_matrix()
        cls.outputs = build_outputs()

    def test_completed_topology_and_safety_audit(self) -> None:
        audit = self.report["audit"]
        self.assertEqual(audit["launcher_state"], "completed")
        self.assertEqual(audit["observed_pairs"], 33)
        self.assertEqual(audit["expected_pairs"], 33)
        self.assertEqual(audit["observed_games"], 6468)
        self.assertEqual(audit["expected_games"], 6468)
        self.assertEqual(audit["terminated"], 6468)
        self.assertEqual(audit["truncated"], 0)
        self.assertEqual(audit["illegal_actions"], 0)
        self.assertEqual(audit["action_mask_mismatches"], 0)
        self.assertEqual(audit["draws"], 2)
        for field in (
            "missing_pair_ids",
            "unexpected_pair_ids",
            "duplicate_pair_ids",
            "checkpoint_mismatches",
        ):
            self.assertEqual(audit[field], [])

    def test_every_pair_has_traceable_hashes_and_49_cells(self) -> None:
        pairs = self.report["pairwise_results"]
        self.assertEqual(len(pairs), 33)
        self.assertEqual(len({pair["pair_id"] for pair in pairs}), 33)
        for pair in pairs:
            with self.subTest(pair=pair["pair_id"]):
                self.assertEqual(pair["games"], 196)
                self.assertEqual(len(pair["class_cells"]), 49)
                self.assertEqual(
                    {cell["games"] for cell in pair["class_cells"].values()},
                    {4},
                )
                self.assertEqual(len(pair["report"]["sha256"]), 64)
                self.assertEqual(len(pair["checkpoint_sha256"]["learner"]), 64)
                self.assertEqual(len(pair["checkpoint_sha256"]["opponent"]), 64)

    def test_payoff_matrix_is_antisymmetric_with_explicit_missing_edges(self) -> None:
        matrix = self.report["payoff_matrix"]["antisymmetric_payoff_matrix"]
        self.assertEqual(len(matrix), 9)
        for row_index, row in enumerate(matrix):
            self.assertEqual(len(row), 9)
            self.assertEqual(row[row_index], 0.0)
            for column_index, value in enumerate(row):
                reverse = matrix[column_index][row_index]
                if value is None:
                    self.assertIsNone(reverse)
                else:
                    self.assertAlmostEqual(value, -reverse, places=12)
        self.assertEqual(
            len(self.report["payoff_matrix"]["unobserved_anchor_vs_anchor_edges"]),
            3,
        )
        self.assertTrue(self.report["payoff_matrix"]["candidate_submatrix_complete"])

    def test_same_rule_model_and_class_aggregates_use_both_sides(self) -> None:
        by_model = self.report["aggregates"]["candidate_vs_candidate_by_model"]
        self.assertEqual(len(by_model), 6)
        self.assertEqual({value["games"] for value in by_model.values()}, {980})

        by_class = self.report["aggregates"][
            "candidate_internal_by_class_both_model_sides"
        ]
        self.assertEqual(set(by_class), {str(value) for value in range(1, 8)})
        self.assertEqual({value["games"] for value in by_class.values()}, {840})
        cells = self.report["aggregates"][
            "candidate_internal_class_cells_both_model_sides"
        ]
        self.assertEqual(len(cells), 49)
        self.assertEqual({value["games"] for value in cells.values()}, {120})

    def test_cross_rule_pairs_are_not_labeled_as_steps_ablation(self) -> None:
        cross_rule = [
            pair
            for pair in self.report["pairwise_results"]
            if pair["group"] == "three_m_vs_one_m"
        ]
        self.assertEqual(len(cross_rule), 18)
        self.assertEqual(
            {pair["comparison_scope"] for pair in cross_rule},
            {"cross_rule_historical_anchor_not_steps_ablation"},
        )

    def test_close_pairs_are_not_force_ranked_or_used_to_remove_models(self) -> None:
        close_pairs = [
            pair
            for pair in self.report["pairwise_results"]
            if pair["confidence_interval_includes_50_percent"]
        ]
        self.assertEqual(len(close_pairs), 29)
        self.assertEqual(
            {pair["ordering_claim"] for pair in close_pairs},
            {"no_forced_ordering"},
        )
        self.assertEqual(
            self.report["close_pair_policy"]["required_980_game_confirmations"],
            [],
        )
        self.assertEqual(self.close_plan["required_confirmation_pair_ids"], [])
        self.assertTrue(self.close_plan["future_removal_requires_confirmation"])

    def test_cycle_detection_records_point_cycles_but_no_strong_cycle(self) -> None:
        cycles = self.report["cycles"]
        self.assertEqual(len(cycles["point_estimate_cycles"]), 2)
        self.assertEqual(cycles["preregistered_cycles"], [])
        self.assertFalse(cycles["significant_cycle_detected"])
        for cycle in cycles["point_estimate_cycles"]:
            self.assertEqual(len(cycle["edges"]), 3)
            for edge in cycle["edges"]:
                self.assertIn("confidence_interval_95", edge)

    def test_cycle_detector_handles_synthetic_rock_paper_scissors(self) -> None:
        edges = {
            ("rock", "scissors"): 0.6,
            ("scissors", "paper"): 0.6,
            ("paper", "rock"): 0.6,
            ("scissors", "rock"): 0.4,
            ("paper", "scissors"): 0.4,
            ("rock", "paper"): 0.4,
        }
        cycles = find_cycles(edges, ("rock", "paper", "scissors"), threshold=0.55)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(
            {(edge["winner"], edge["loser"]) for edge in cycles[0]["edges"]},
            {("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")},
        )

    def test_wilson_interval_contains_observed_rate(self) -> None:
        lower, upper = wilson_interval(114, 196)
        self.assertLess(lower, 114 / 196)
        self.assertGreater(upper, 114 / 196)

    def test_saved_outputs_are_byte_stable(self) -> None:
        for name, payload in self.outputs.items():
            with self.subTest(name=name):
                self.assertEqual((REPORT_DIRECTORY / name).read_bytes(), payload)
        saved = json.loads(
            (REPORT_DIRECTORY / "seed_payoff_matrix.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(saved, self.report)


if __name__ == "__main__":
    unittest.main()
