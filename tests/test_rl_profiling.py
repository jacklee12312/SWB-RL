from __future__ import annotations

import unittest

from swb.rl.profiling import (
    summarize_stage_breakdown,
    summarize_timing_samples,
    training_timing_report,
)


class RlProfilingTests(unittest.TestCase):
    def test_summary_preserves_totals_median_and_nearest_rank_p95(self) -> None:
        summary = summarize_timing_samples([
            {"stage_seconds": 1.0, "records": 10.0},
            {"stage_seconds": 3.0, "records": 20.0},
            {"stage_seconds": 2.0, "records": 30.0},
        ])

        self.assertEqual(summary["sample_count"], 3)
        stage = summary["fields"]["stage_seconds"]
        self.assertEqual(stage["total"], 6.0)
        self.assertEqual(stage["mean"], 2.0)
        self.assertEqual(stage["median"], 2.0)
        self.assertEqual(stage["p95"], 3.0)

    def test_training_report_calculates_pipeline_wall_fractions(self) -> None:
        report = training_timing_report(
            [
                {"collect_total_seconds": 3.0},
                {"collect_total_seconds": 1.0},
            ],
            [
                {"update_total_seconds": 1.0},
                {"update_total_seconds": 1.0},
            ],
        )

        pipeline = report["pipeline_wall_time"]
        self.assertEqual(pipeline["measured_seconds"], 6.0)
        self.assertEqual(pipeline["rollout_fraction"], 2.0 / 3.0)
        self.assertEqual(pipeline["update_fraction"], 1.0 / 3.0)

    def test_training_report_rejects_mismatched_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample counts"):
            training_timing_report(
                [{"collect_total_seconds": 1.0}],
                [],
            )

    def test_empty_stage_breakdown_is_explicit_and_safe(self) -> None:
        breakdown = summarize_stage_breakdown(
            [],
            stage_fields=("first_seconds",),
            wall_field="total_seconds",
        )

        self.assertEqual(breakdown["sample_count"], 0)
        self.assertEqual(breakdown["accounted_fraction"], 0.0)
        self.assertEqual(breakdown["stages"], {})
        self.assertFalse(breakdown["passed_90_percent"])

    def test_stage_breakdown_reports_per_step_share_and_stage_sum(
        self,
    ) -> None:
        breakdown = summarize_stage_breakdown(
            [
                {
                    "total_seconds": 10.0,
                    "records": 100.0,
                    "first_seconds": 6.0,
                    "second_seconds": 3.5,
                },
                {
                    "total_seconds": 20.0,
                    "records": 200.0,
                    "first_seconds": 12.0,
                    "second_seconds": 7.0,
                },
            ],
            stage_fields=("first_seconds", "second_seconds"),
            wall_field="total_seconds",
        )

        self.assertEqual(breakdown["accounted_seconds"], 28.5)
        self.assertEqual(breakdown["accounted_fraction"], 0.95)
        self.assertTrue(breakdown["passed_90_percent"])
        first = breakdown["stages"]["first_seconds"]
        self.assertEqual(first["milliseconds_per_agent_step"], 60.0)
        self.assertEqual(first["fraction_of_stage_wall"], 0.6)
        self.assertEqual(first["median_seconds"], 9.0)
        self.assertEqual(first["p95_seconds"], 12.0)

    def test_training_report_exposes_pipeline_accounted_fraction(self) -> None:
        report = training_timing_report(
            [{
                "collect_total_seconds": 10.0,
                "records": 100.0,
                "central_forward_seconds": 9.5,
            }],
            [{
                "update_total_seconds": 10.0,
                "records": 100.0,
                "forward_loss_seconds": 9.5,
            }],
        )

        pipeline = report["stage_breakdown"]["pipeline"]
        self.assertEqual(pipeline["accounted_fraction"], 0.95)
        self.assertTrue(pipeline["passed_90_percent"])


if __name__ == "__main__":
    unittest.main()
