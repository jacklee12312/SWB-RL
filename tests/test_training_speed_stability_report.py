from __future__ import annotations

import unittest

from scripts.report_training_speed_stability import build_report


class TrainingSpeedStabilityReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_same_checkpoint_profiles_reproduce_runtime_variability(self) -> None:
        self.assertEqual(len(self.report["checkpoint_sha256"]), 1)
        self.assertGreater(
            self.report["findings"][
                "initial_six_worker_runtime_variability_factor"
            ],
            2.0,
        )
        initial = self.report["groups"][
            "initial_six_worker_baseline"
        ]["aggregate"]
        self.assertTrue(initial["all_checkpoints_unchanged"])

    def test_final_candidate_passes_speed_memory_and_termination_gates(
        self,
    ) -> None:
        decision = self.report["decision"]
        self.assertTrue(decision["adopted"])
        self.assertEqual(decision["runtime_overrides"], {
            "rollout_workers": 7,
            "rollout_worker_torch_threads": 2,
            "central_inference_batch_wait_seconds": 0.0005,
        })
        final = self.report["groups"][
            "final_seven_worker_half_ms"
        ]["aggregate"]
        self.assertEqual(final["run_count"], 3)
        self.assertEqual(final["episodes"], 644)
        self.assertEqual(final["truncations"], 0)
        self.assertGreater(
            final["gpu_memory_minimum_headroom_mib"], 1024.0
        )
        self.assertFalse(final["any_hardware_throttle"])
        self.assertGreater(
            self.report["findings"][
                "final_improvement_over_fast_six"
            ],
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
