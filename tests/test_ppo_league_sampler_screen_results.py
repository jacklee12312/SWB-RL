from __future__ import annotations

import unittest

from scripts.report_ppo_league_baseline import render_json
from scripts.report_ppo_league_sampler_screen_results import build_result


class SamplerScreenResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = build_result()

    def test_result_selects_hard_from_complete_safe_screen(self) -> None:
        result = self.result
        self.assertEqual(result["decision_status"], "passed")
        self.assertEqual(result["decision"]["selected_sampler"], "hard")
        self.assertEqual(result["decision"]["control_sampler"], "uniform")
        self.assertEqual(
            result["decision"]["rejected_samplers"], ["variance"]
        )
        self.assertEqual(
            result["screen_contract"]["candidate_evaluation_reports"], 54
        )
        self.assertEqual(
            result["screen_contract"]["candidate_evaluation_games"], 10_584
        )
        self.assertEqual(
            result["safety"]["candidate_evaluation"]["truncated_games"], 0
        )
        self.assertEqual(
            result["safety"]["candidate_evaluation"]["illegal_actions"], 0
        )
        self.assertEqual(
            result["safety"]["candidate_evaluation"][
                "action_mask_mismatches"
            ],
            0,
        )

    def test_result_preserves_observed_sampler_totals(self) -> None:
        samplers = {
            row["sampler"]: row for row in self.result["sampler_results"]
        }
        self.assertEqual(
            (
                samplers["uniform"]["wins"],
                samplers["uniform"]["draws"],
                samplers["uniform"]["score_points"],
                samplers["uniform"]["games"],
            ),
            (1878, 1, 1878.5, 3528),
        )
        self.assertEqual(
            (samplers["variance"]["wins"], samplers["variance"]["games"]),
            (1873, 3528),
        )
        self.assertEqual(
            (samplers["hard"]["wins"], samplers["hard"]["games"]),
            (1983, 3528),
        )
        self.assertTrue(
            self.result["decision"][
                "hard_wins_all_matched_training_seed_comparisons"
            ]
        )
        self.assertEqual(
            self.result["decision"]["hard_beats_uniform_opponents"], 5
        )
        comparison = next(
            row
            for row in self.result["paired_comparisons"]
            if row["left"] == "hard" and row["right"] == "uniform"
        )
        self.assertEqual(comparison["left_only_wins"], 606)
        self.assertEqual(comparison["right_only_wins"], 502)
        self.assertEqual(comparison["same_result"], 2420)

    def test_result_render_is_byte_stable(self) -> None:
        self.assertEqual(render_json(build_result()), render_json(build_result()))


if __name__ == "__main__":
    unittest.main()
