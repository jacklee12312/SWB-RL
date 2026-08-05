from __future__ import annotations

import unittest

from scripts.report_ppo_league_baseline import render_json
from scripts.report_ppo_league_sampler_screen_plan import build_plan


class SamplerScreenPlanTests(unittest.TestCase):
    def test_plan_counts_seeds_and_outputs_are_frozen(self) -> None:
        plan = build_plan()
        self.assertTrue(plan["immutable"])
        self.assertEqual(plan["data_partition"], "pfsp_tuning")
        self.assertEqual(
            plan["training"]["training_seeds"],
            [20261101, 20261102, 20261103],
        )
        self.assertEqual(plan["summary"], {
            "training_job_count": 9,
            "candidate_evaluation_pair_count": 54,
            "existing_active_pair_count": 5,
            "missing_active_pair_count": 10,
            "archive_baseline_pair_count": 90,
            "queued_evaluation_pair_count": 154,
            "queued_evaluation_game_count": 30_184,
        })
        training_outputs = {
            row["checkpoint"] for row in plan["training"]["jobs"]
        }
        evaluation_outputs = {
            row["output"]
            for section in (
                "candidate_evaluation",
                "generation_000_active_matrix",
                "archive_baseline",
            )
            for row in plan[section]["jobs"]
        }
        self.assertEqual(len(training_outputs), 9)
        self.assertEqual(len(evaluation_outputs), 154)

    def test_plan_render_is_byte_stable(self) -> None:
        self.assertEqual(render_json(build_plan()), render_json(build_plan()))


if __name__ == "__main__":
    unittest.main()
