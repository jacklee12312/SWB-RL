from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.scan_training_speed_stage_2_4 import (
    FORMAL_RUNS,
    MEASURED_AGENT_STEPS,
)
from scripts.verify_training_speed_stage_2_4_b import (
    interaction_configurations,
    summarize_interactions,
)


class TrainingSpeedStage24BTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REPORT = (
        ROOT
        / "data/reports/training_speed/stage_2_4_b_interactions.json"
    )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _measurement(speed: float) -> dict[str, object]:
        return {
            "agent_steps": MEASURED_AGENT_STEPS,
            "agent_steps_per_second": speed,
            "collect_p95_seconds": 18.0,
            "update_p95_seconds": 20.0,
            "batching": {
                "mean_batch_size": 3.5,
                "p50_batch_size": 4.0,
                "p95_batch_size": 6.0,
                "empty_slot_fraction": 0.4,
                "configured_wait_total_seconds": 1.0,
                "worker_message_wait_total_seconds": 1.0,
            },
            "episode_length": {
                "mean": 70.0,
                "p50": 65.0,
                "p95": 110.0,
                "maximum": 150.0,
            },
            "system": {
                "sample_count": 2,
                "gpu_sample_count": 2,
                "cpu_total_median_percent": 20.0,
                "cpu_single_core_peak_percent": 50.0,
                "ram_used_peak_bytes": 200.0,
                "pagefile_used_peak_bytes": 20.0,
                "gpu_utilization_median_percent": 40.0,
                "gpu_utilization_p95_percent": 50.0,
                "gpu_idle_sample_fraction_at_or_below_5_percent": 0.0,
                "gpu_memory_peak_mib": 200.0,
            },
            "abnormal_exit_count": 0,
        }

    @staticmethod
    def _primary_scan() -> dict[str, object]:
        return {
            "diagnosis": {
                "stable_five_percent_winners": [
                    "wait_1_0_ms",
                    "workers_5",
                    "workers_6",
                ],
            },
            "configurations": {
                "wait_1_0_ms": {
                    "agent_steps_per_second": {"median": 59.0},
                },
                "workers_5": {
                    "agent_steps_per_second": {"median": 49.0},
                },
                "workers_6": {
                    "agent_steps_per_second": {"median": 52.0},
                },
            },
        }

    def test_interactions_only_combine_cross_dimension_winners(self) -> None:
        configs = interaction_configurations()
        self.assertEqual(len(configs), 2)
        self.assertEqual(
            {config["rollout_workers"] for config in configs},
            {5, 6},
        )
        for config in configs:
            self.assertEqual(
                config["central_inference_batch_wait_ms"],
                1.0,
            )
            self.assertEqual(config["worker_torch_threads"], 2)
            self.assertIn("wait_1_0_ms", config["constituents"])
            self.assertEqual(len(config["constituents"]), 2)

    def test_summary_selects_best_three_run_interaction(self) -> None:
        reports = []
        for config in interaction_configurations():
            speed = (
                70.0
                if config["rollout_workers"] == 6
                else 66.0
            )
            for run_index in range(1, FORMAL_RUNS + 1):
                reports.append({
                    "configuration": deepcopy(config),
                    "run_index": run_index,
                    "checkpoint_sha256": "fixed",
                    "measurement": self._measurement(speed),
                })
        report = summarize_interactions(
            reports,
            self._primary_scan(),
            {
                "observations": {
                    "v4.1": {
                        "agent_steps_per_second": {
                            "median": 44.7,
                            "range": 0.2,
                        },
                    },
                },
            },
            source_paths=["synthetic"],
            source_sha256=["synthetic"],
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["decision"]["adopted_runtime_configuration"]["id"],
            "workers_6_wait_1_0_ms",
        )
        self.assertGreater(
            report["decision"]["relative_gain_vs_strongest_constituent"],
            0.0,
        )

    def test_saved_interactions_are_complete_and_integrity_checked(self) -> None:
        report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertEqual(
            set(report["configurations"]),
            {config["id"] for config in interaction_configurations()},
        )
        self.assertEqual(len(report["integrity"]["checkpoint_sha256"]), 1)
        self.assertTrue(report["integrity"]["all_runs_monitored"])
        self.assertTrue(report["integrity"]["no_abnormal_exits"])
        paths = report["integrity"]["source_paths"]
        hashes = report["integrity"]["source_sha256"]
        self.assertEqual(len(paths), 2 * FORMAL_RUNS)
        for relative, expected_hash in zip(paths, hashes):
            path = self.ROOT / relative
            run = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(self._sha256(path), expected_hash)
            self.assertTrue(run["checkpoint_unchanged"])
            self.assertTrue(run["profiling_switches_disabled"])
            self.assertGreaterEqual(
                run["measurement"]["agent_steps"],
                MEASURED_AGENT_STEPS,
            )


if __name__ == "__main__":
    unittest.main()
