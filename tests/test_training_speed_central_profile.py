from __future__ import annotations

import json
import unittest
from pathlib import Path


class TrainingSpeedCentralProfileTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REPORT = (
        ROOT
        / "data/reports/training_speed/"
        "stage_2_2_central_inference_smoke.json"
    )

    def setUp(self) -> None:
        self.report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        self.collect = self.report["iterations"][0]["collect"]

    def test_saved_profile_uses_frozen_checkpoint_and_cuda_diagnostics(
        self,
    ) -> None:
        self.assertTrue(self.report["checkpoint_unchanged"])
        self.assertEqual(
            self.report["checkpoint_sha256_before"],
            self.report["checkpoint_sha256_after"],
        )
        self.assertEqual(self.report["device"], "cuda")
        self.assertTrue(
            self.report["runtime_rollout_configuration"][
                "profile_central_timing"
            ]
        )
        self.assertTrue(
            self.report["central_timing_methodology"]["enabled"]
        )
        self.assertGreater(
            self.collect["central_profiled_cuda_batches"],
            0.0,
        )

    def test_batch_histogram_explains_requests_and_empty_slots(self) -> None:
        histogram = {
            int(key.removeprefix("central_batch_size_").removesuffix(
                "_count"
            )): float(value)
            for key, value in self.collect.items()
            if (
                key.startswith("central_batch_size_")
                and key.endswith("_count")
                and key[
                    len("central_batch_size_") : -len("_count")
                ].isdigit()
            )
        }
        self.assertEqual(
            sum(histogram.values()),
            self.collect["central_inference_batches"],
        )
        self.assertEqual(
            sum(size * count for size, count in histogram.items()),
            self.collect["central_inference_requests"],
        )
        self.assertEqual(
            self.collect["central_batch_capacity_slots"]
            - self.collect["central_inference_requests"],
            self.collect["central_batch_empty_slots"],
        )
        self.assertLessEqual(
            self.collect["central_batch_size_p50"],
            self.collect["central_batch_size_p95"],
        )

    def test_model_components_and_gpu_ratios_are_self_consistent(self) -> None:
        model_components = sum(
            self.collect[name]
            for name in (
                "central_model_input_encoding_seconds",
                "central_transformer_seconds",
                "central_transformer_to_gru_seconds",
                "central_gru_seconds",
                "central_action_value_stage_seconds",
            )
        )
        self.assertAlmostEqual(
            model_components,
            self.collect["central_model_forward_seconds"],
            delta=0.02,
        )
        self.assertAlmostEqual(
            self.collect[
                "central_gpu_busy_fraction_of_busy_plus_worker_wait"
            ]
            + self.collect[
                "central_gpu_worker_wait_fraction_of_busy_plus_worker_wait"
            ],
            1.0,
        )
        for field in (
            "central_queue_to_batch_wait_seconds",
            "central_cpu_input_assembly_seconds",
            "central_host_to_device_seconds",
            "central_transformer_seconds",
            "central_gru_seconds",
            "central_policy_head_seconds",
            "central_masked_distribution_seconds",
            "central_sampling_seconds",
            "central_device_to_host_seconds",
            "central_result_distribution_seconds",
        ):
            with self.subTest(field=field):
                self.assertGreater(self.collect[field], 0.0)


if __name__ == "__main__":
    unittest.main()
