from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.scan_training_speed_stage_2_4 import (
    FORMAL_RUNS,
    MEASURED_AGENT_STEPS,
    THREAD_SCAN,
    WAIT_SCAN_MS,
    WORKER_SCAN,
    compact_profile,
    primary_configurations,
    summarize_scan,
)


class TrainingSpeedStage24Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REPORT = (
        ROOT
        / "data/reports/training_speed/stage_2_4_scan.json"
    )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _iteration(index: int) -> dict[str, object]:
        return {
            "agent_steps": 2_048,
            "elapsed_seconds": 40.0 + index,
            "collect": {
                "collect_total_seconds": 25.0 + index,
                "central_inference_requests": 2_000.0,
                "central_inference_batches": 1_000.0,
                "central_batch_capacity_slots": 4_000.0,
                "central_batch_empty_slots": 2_000.0,
                "central_batch_size_1_count": 200.0,
                "central_batch_size_2_count": 600.0,
                "central_batch_size_3_count": 200.0,
                "central_batch_wait_seconds": 0.2,
                "central_worker_message_wait_seconds": 1.0,
                f"worker_episode_steps_{60 + index}_count": 2.0,
                "worker_long_episode_count": 0.0,
                "worker_truncated_episode_count": 0.0,
            },
            "update": {"update_total_seconds": 15.0},
        }

    @staticmethod
    def _system_samples() -> list[dict[str, object]]:
        return [
            {
                "cpu_total_percent": 1.0,
                "cpu_per_core_percent": [1.0, 1.0],
                "ram_used_bytes": 100,
                "pagefile_used_bytes": 10,
                "gpu": {
                    "utilization_percent": 0.0,
                    "memory_used_mib": 100.0,
                },
            },
            {
                "cpu_total_percent": 20.0,
                "cpu_per_core_percent": [10.0, 50.0],
                "ram_used_bytes": 200,
                "pagefile_used_bytes": 20,
                "gpu": {
                    "utilization_percent": 40.0,
                    "memory_used_mib": 200.0,
                },
            },
        ]

    @staticmethod
    def _measurement(speed: float) -> dict[str, object]:
        return {
            "agent_steps": MEASURED_AGENT_STEPS,
            "agent_steps_per_second": speed,
            "collect_p95_seconds": 30.0,
            "update_p95_seconds": 20.0,
            "batching": {
                "mean_batch_size": 2.0,
                "p50_batch_size": 2.0,
                "p95_batch_size": 3.0,
                "empty_slot_fraction": 0.5,
                "configured_wait_total_seconds": 0.2,
                "worker_message_wait_total_seconds": 1.0,
            },
            "episode_length": {
                "mean": 64.0,
                "p50": 64.0,
                "p95": 80.0,
                "maximum": 100.0,
                "long_episode_count": 0.0,
                "truncated_episode_count": 0.0,
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

    def test_primary_scan_is_single_variable_and_deduplicates_baseline(
        self,
    ) -> None:
        configs = primary_configurations()
        self.assertEqual(len(configs), 11)
        self.assertEqual(
            sum(config["id"] == "baseline" for config in configs),
            1,
        )
        self.assertEqual(
            {
                float(config["central_inference_batch_wait_ms"])
                for config in configs
                if config["dimension"] in {"baseline", "batch_wait"}
            },
            set(WAIT_SCAN_MS),
        )
        self.assertEqual(
            {
                int(config["rollout_workers"])
                for config in configs
                if config["dimension"] in {"baseline", "rollout_workers"}
            },
            set(WORKER_SCAN),
        )
        self.assertEqual(
            {
                int(config["worker_torch_threads"])
                for config in configs
                if config["dimension"] in {"baseline", "worker_threads"}
            },
            set(THREAD_SCAN),
        )
        for config in configs:
            changed = sum((
                config["rollout_workers"] != 4,
                config["worker_torch_threads"] != 2,
                config["central_inference_batch_wait_ms"] != 0.5,
            ))
            self.assertLessEqual(changed, 1)

    def test_compaction_excludes_warmup_and_preserves_required_metrics(
        self,
    ) -> None:
        profile = {
            "iterations": [
                self._iteration(index)
                for index in range(5)
            ],
        }
        compact = compact_profile(
            profile,
            measured_agent_steps=MEASURED_AGENT_STEPS,
            warmup_updates=2,
            system_samples=self._system_samples(),
        )
        self.assertEqual(compact["steady_update_count"], 3)
        self.assertEqual(compact["agent_steps"], MEASURED_AGENT_STEPS)
        self.assertAlmostEqual(
            compact["batching"]["mean_batch_size"],
            2.0,
        )
        self.assertEqual(compact["batching"]["p50_batch_size"], 2.0)
        self.assertEqual(compact["batching"]["p95_batch_size"], 3.0)
        self.assertEqual(compact["episode_length"]["sample_count"], 6)
        self.assertEqual(compact["system"]["sample_count"], 1)
        self.assertEqual(
            compact["system"]["gpu_idle_sample_fraction_at_or_below_5_percent"],
            0.0,
        )

    def test_summary_applies_five_percent_and_variability_gate(self) -> None:
        reports = []
        for config in primary_configurations():
            speed = 47.5 if config["id"] == "workers_6" else 45.0
            for run_index in range(1, FORMAL_RUNS + 1):
                reports.append({
                    "configuration": deepcopy(config),
                    "run_index": run_index,
                    "checkpoint_sha256": "fixed",
                    "measurement": self._measurement(speed),
                })
        baseline = {
            "observations": {
                "v4.1": {
                    "agent_steps_per_second": {
                        "median": 44.7,
                        "range": 0.2,
                    },
                },
            },
        }
        report = summarize_scan(
            reports,
            baseline,
            source_paths=["synthetic"],
            source_sha256=["synthetic"],
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["diagnosis"]["stable_five_percent_winners"],
            ["workers_6"],
        )
        self.assertTrue(
            report["diagnosis"]["stage_2_4_b_gate"]["enter"]
        )
        self.assertEqual(
            report["diagnosis"]["batch_formation_limit"],
            "request_arrival_insufficient",
        )

    def test_saved_scan_has_three_monitored_runs_per_configuration(
        self,
    ) -> None:
        report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["coverage"]["passed"])
        self.assertEqual(
            set(report["configurations"]),
            {str(config["id"]) for config in primary_configurations()},
        )
        self.assertEqual(
            len(report["integrity"]["checkpoint_sha256"]),
            1,
        )
        self.assertTrue(report["integrity"]["all_runs_monitored"])
        self.assertTrue(report["integrity"]["no_abnormal_exits"])
        for config in report["configurations"].values():
            self.assertEqual(config["run_count"], FORMAL_RUNS)
            self.assertEqual(
                len(config["agent_steps_per_second"]["runs"]),
                FORMAL_RUNS,
            )

        source_paths = report["integrity"]["source_paths"]
        source_hashes = report["integrity"]["source_sha256"]
        self.assertEqual(len(source_paths), 11 * FORMAL_RUNS)
        self.assertEqual(len(source_hashes), len(source_paths))
        for relative, expected_hash in zip(source_paths, source_hashes):
            path = self.ROOT / relative
            run = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(self._sha256(path), expected_hash)
            self.assertTrue(run["checkpoint_unchanged"])
            self.assertTrue(run["profiling_switches_disabled"])
            self.assertGreaterEqual(
                run["measurement"]["agent_steps"],
                MEASURED_AGENT_STEPS,
            )
            self.assertEqual(
                run["measurement"]["abnormal_exit_count"],
                0,
            )
            self.assertGreater(
                run["measurement"]["system"]["sample_count"],
                0,
            )
            self.assertGreater(
                run["measurement"]["system"]["gpu_sample_count"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
