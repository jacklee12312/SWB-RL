from __future__ import annotations

import unittest

from swb.rl.profiling import (
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


if __name__ == "__main__":
    unittest.main()
