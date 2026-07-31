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
from scripts.verify_training_speed_stage_2_5 import (
    CONFIGURATION,
    MATERIALITY_THRESHOLD,
    build_report,
)


class TrainingSpeedStage25Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REPORT = (
        ROOT
        / "data/reports/training_speed/stage_2_5_a_obs_001.json"
    )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _before() -> dict[str, object]:
        fields = {
            "worker_decision_observation_construction_seconds": {
                "total": 0.4,
            },
            "worker_step_observation_construction_seconds": {
                "total": 11.0,
            },
            "worker_bootstrap_observation_construction_seconds": {
                "total": 0.01,
            },
            "worker_observation_construction_seconds": {"total": 11.41},
            "worker_agent_steps": {"total": 6_400.0},
        }
        return {
            "runtime_rollout_configuration": {"rollout_workers": 4},
            "steady_state": {
                "pipeline_wall_time": {"measured_seconds": 100.0},
                "collect": {"fields": fields},
            },
        }

    @staticmethod
    def _stage_2_4() -> dict[str, object]:
        return {
            "decision": {
                "median_agent_steps_per_second": 64.0,
            },
            "configurations": {
                "workers_6_wait_1_0_ms": {
                    "agent_steps_per_second": {"range": 1.28},
                },
            },
        }

    @staticmethod
    def _environment() -> dict[str, object]:
        return {
            "rates": {
                "observe_cached_per_second": 8_000.0,
                "observe_cold_per_second": 500.0,
                "observe_cache_speedup": 16.0,
                "step_per_second": 250.0,
            },
            "thresholds_passed": True,
        }

    @staticmethod
    def _runs(speed: float) -> list[dict[str, object]]:
        return [
            {
                "configuration": deepcopy(CONFIGURATION),
                "run_index": run_index,
                "checkpoint_unchanged": True,
                "checkpoint_sha256": "fixed",
                "profiling_switches_disabled": True,
                "measurement": {
                    "agent_steps": MEASURED_AGENT_STEPS,
                    "steady_update_count": 3,
                    "agent_steps_per_second": speed,
                    "observation": {
                        "decision_construction_seconds": 0.0,
                        "step_construction_seconds": 10.0,
                        "bootstrap_construction_seconds": 0.01,
                        "total_construction_seconds": 10.01,
                    },
                    "abnormal_exit_count": 0,
                },
            }
            for run_index in range(1, FORMAL_RUNS + 1)
        ]

    def test_report_closes_followups_below_materiality_and_variability(
        self,
    ) -> None:
        report = build_report(
            self._runs(64.5),
            self._before(),
            self._stage_2_4(),
            self._environment(),
            sources={},
        )
        self.assertTrue(report["passed"])
        self.assertTrue(
            report["equivalence"]["decision_duplicate_removed"]
        )
        self.assertFalse(
            report["decision_gate"][
                "continue_remaining_stage_2_5_candidates"
            ]
        )
        self.assertLess(
            report["decision_gate"][
                "observation_concurrency_normalized_pipeline_wall_fraction"
            ],
            MATERIALITY_THRESHOLD,
        )
        self.assertEqual(
            report["decision_gate"]["remaining_candidate_disposition"],
            "closed_below_materiality_and_variability_gate",
        )
        self.assertTrue(report["integrity"]["no_abnormal_exits"])
        self.assertEqual(
            report["integrity"]["checkpoint_sha256"],
            ["fixed"],
        )

    def test_report_continues_when_observation_is_material(self) -> None:
        before = self._before()
        before["steady_state"]["collect"]["fields"][
            "worker_observation_construction_seconds"
        ]["total"] = 24.0
        report = build_report(
            self._runs(64.0),
            before,
            self._stage_2_4(),
            self._environment(),
            sources={},
        )
        self.assertTrue(
            report["decision_gate"][
                "continue_remaining_stage_2_5_candidates"
            ]
        )

    def test_saved_report_is_complete_and_integrity_checked(self) -> None:
        report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertEqual(report["candidate"], "A-OBS-001")
        self.assertEqual(
            report["minimal_reproduction"][
                "after_decision_observation_construction_seconds"
            ],
            [0.0] * FORMAL_RUNS,
        )
        self.assertTrue(
            report["equivalence"]["fixed_seed_full_trajectory_contract_test"]
        )
        self.assertTrue(report["equivalence"]["checkpoint_unchanged"])
        self.assertTrue(
            report["environment_microbenchmark"]["passed"]
        )
        self.assertTrue(report["integrity"]["all_runs_meet_step_gate"])
        self.assertTrue(
            report["integrity"]["all_optional_profiling_disabled"]
        )
        self.assertTrue(report["integrity"]["no_abnormal_exits"])
        self.assertEqual(len(report["integrity"]["checkpoint_sha256"]), 1)
        self.assertEqual(
            len(report["end_to_end"]["runs_agent_steps_per_second"]),
            FORMAL_RUNS,
        )
        for source in report["sources"].values():
            entries = source if isinstance(source, list) else [source]
            for entry in entries:
                path = self.ROOT / entry["path"]
                self.assertTrue(path.is_file())
                self.assertEqual(self._sha256(path), entry["sha256"])


if __name__ == "__main__":
    unittest.main()
