from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.training_speed_baseline import (
    ALLOWED_VERSION_MIGRATIONS,
    _nvidia_sample,
    summarize_baselines,
    version_differences,
)
from scripts.profile_ppo_training import _system_monitor_summary


class TrainingSpeedBaselineTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_version_difference_names_every_changed_contract(self) -> None:
        self.assertEqual(
            version_differences(
                {"catalog_sha256": "old", "action": "same"},
                {
                    "catalog_sha256": "new",
                    "training_pool_sha256": "new-pool",
                    "action": "same",
                },
            ),
            {
                "catalog_sha256": ("old", "new"),
                "training_pool_sha256": (None, "new-pool"),
            },
        )
        self.assertEqual(
            ALLOWED_VERSION_MIGRATIONS,
            {"catalog_sha256", "training_pool_sha256"},
        )

    def test_summary_rejects_missing_three_run_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly three"):
            summarize_baselines([])

    def test_saved_summary_has_three_100k_runs_per_observation(self) -> None:
        report = json.loads(
            (
                self.ROOT
                / "data/reports/training_speed/baseline_summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(report["passed"])
        self.assertEqual(set(report["observations"]), {"v3.6", "v4.1"})
        for observation in report["observations"].values():
            self.assertEqual(observation["run_count"], 3)
            self.assertEqual(
                len(observation["agent_steps_per_second"]["runs"]),
                3,
            )
            self.assertEqual(len(observation["checkpoint_sha256"]), 1)
            self.assertGreater(
                observation["system_monitor"]["sample_count"],
                0,
            )

    def test_every_saved_run_is_fixed_monitored_and_over_100k(self) -> None:
        for observation in ("v4_1", "v3_6"):
            for run_index in (1, 2, 3):
                path = (
                    self.ROOT
                    / "data/reports/training_speed"
                    / f"baseline_run_{observation}_{run_index}.json"
                )
                report = json.loads(path.read_text(encoding="utf-8"))
                baseline = report["baseline"]
                config = report["runtime_rollout_configuration"]
                self.assertTrue(report["checkpoint_unchanged"])
                self.assertTrue(baseline["fixed_checkpoint_unchanged"])
                self.assertGreaterEqual(
                    baseline["steady_measurement"]["agent_steps"],
                    100_000,
                )
                self.assertEqual(
                    baseline["steady_measurement"][
                        "excluded_warmup_updates"
                    ],
                    2,
                )
                self.assertEqual(config["rollout_workers"], 4)
                self.assertEqual(config["worker_torch_threads"], 2)
                self.assertEqual(
                    config["central_inference_batch_wait_seconds"],
                    0.0005,
                )
                self.assertGreater(len(baseline["system_samples"]), 0)
                first = baseline["system_samples"][0]
                self.assertEqual(
                    len(first["cpu_per_core_percent"]),
                    baseline["system_before"]["logical_cpu_count"],
                )
                self.assertIn("power_watts", first["gpu"])

    def test_profile_system_summary_reports_paging_and_memory(self) -> None:
        samples = [
            {
                "elapsed_seconds": 0.0,
                "cpu_total_percent": 10.0,
                "cpu_per_core_percent": [5.0, 15.0],
                "cpu_frequency_mhz": 3200.0,
                "ram_used_bytes": 100,
                "pagefile_used_bytes": 200,
                "pagefile_sin_bytes": 7,
                "pagefile_sout_bytes": 11,
                "gpu": {
                    "memory_used_mib": 300.0,
                    "utilization_percent": 40.0,
                    "memory_utilization_percent": 20.0,
                    "power_watts": 100.0,
                    "power_limit_watts": 320.0,
                    "temperature_celsius": 50.0,
                    "graphics_clock_mhz": 1800.0,
                    "memory_clock_mhz": 11200.0,
                    "pstate": "P2",
                    "software_power_cap_active": False,
                    "hardware_thermal_slowdown_active": False,
                    "hardware_power_brake_slowdown_active": False,
                },
            },
            {
                "elapsed_seconds": 1.0,
                "cpu_total_percent": 20.0,
                "cpu_per_core_percent": [25.0, 30.0],
                "cpu_frequency_mhz": 3400.0,
                "ram_used_bytes": 150,
                "pagefile_used_bytes": 220,
                "pagefile_sin_bytes": 7,
                "pagefile_sout_bytes": 11,
                "gpu": {
                    "memory_used_mib": 350.0,
                    "utilization_percent": 90.0,
                    "memory_utilization_percent": 70.0,
                    "power_watts": 250.0,
                    "power_limit_watts": 320.0,
                    "temperature_celsius": 62.0,
                    "graphics_clock_mhz": 2500.0,
                    "memory_clock_mhz": 11200.0,
                    "pstate": "P2",
                    "software_power_cap_active": True,
                    "hardware_thermal_slowdown_active": False,
                    "hardware_power_brake_slowdown_active": False,
                },
            },
        ]
        summary = _system_monitor_summary(samples)
        self.assertTrue(summary["passed"])
        self.assertTrue(summary["no_page_in_or_page_out"])
        self.assertEqual(summary["pagefile_used_change_bytes"], 20)
        self.assertEqual(summary["ram_used_peak_bytes"], 150)
        self.assertEqual(summary["gpu_memory_peak_mib"], 350.0)
        self.assertEqual(summary["cpu_total_median_percent"], 15.0)
        self.assertEqual(summary["cpu_single_core_peak_percent"], 30.0)
        self.assertEqual(summary["cpu_frequency_median_mhz"], 3300.0)
        self.assertEqual(summary["gpu_graphics_clock_minimum_mhz"], 1800.0)
        self.assertEqual(summary["gpu_graphics_clock_p95_mhz"], 2500.0)
        self.assertEqual(summary["gpu_pstate_sample_counts"], {"P2": 2})
        self.assertEqual(
            summary["gpu_active_clock_event_sample_counts"][
                "software_power_cap_active"
            ],
            1,
        )
        self.assertFalse(summary["gpu_any_hardware_throttle"])

    @patch("scripts.training_speed_baseline._run_text")
    def test_nvidia_sample_includes_clocks_pstate_and_throttle_reasons(
        self,
        run_text,
    ) -> None:
        run_text.return_value = (
            "RTX 4080, 591.86, P2, 91, 73, 15000, 16376, 275.5, "
            "320.0, 64, 2520, 11201, Active, Not Active, Not Active"
        )

        sample = _nvidia_sample()

        self.assertEqual(sample["pstate"], "P2")
        self.assertEqual(sample["graphics_clock_mhz"], 2520.0)
        self.assertEqual(sample["memory_clock_mhz"], 11201.0)
        self.assertTrue(sample["software_power_cap_active"])
        self.assertFalse(sample["hardware_thermal_slowdown_active"])


if __name__ == "__main__":
    unittest.main()
