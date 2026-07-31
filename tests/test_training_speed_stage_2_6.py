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
from scripts.verify_training_speed_stage_2_6_a_net_001 import (
    BATCH_SIZES,
    CONFIGURATION,
    build_report,
)


class TrainingSpeedStage26ANet001Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    MICRO = (
        ROOT
        / "data/reports/training_speed/stage_2_6_a_net_001_micro.json"
    )
    REPORT = (
        ROOT
        / "data/reports/training_speed/stage_2_6_a_net_001.json"
    )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _runs(speed: float) -> list[dict[str, object]]:
        return [
            {
                "configuration": deepcopy(CONFIGURATION),
                "run_index": run_index,
                "checkpoint_sha256": "fixed",
                "checkpoint_unchanged": True,
                "profiling_switches_disabled": True,
                "measurement": {
                    "agent_steps": MEASURED_AGENT_STEPS,
                    "steady_update_count": 3,
                    "agent_steps_per_second": speed,
                    "abnormal_exit_count": 0,
                },
            }
            for run_index in range(1, FORMAL_RUNS + 1)
        ]

    @staticmethod
    def _micro() -> dict[str, object]:
        return {
            "passed": True,
            "exact_output_equivalence": {"all": True},
            "profiler": {
                "batch_size": 4,
                "trace": {
                    "kernel_event_count": 1_926,
                    "kernel_launch_event_count": 1_926,
                    "synchronization_event_count": 5,
                },
            },
        }

    def test_summary_rejects_gain_within_comparison_variability(
        self,
    ) -> None:
        report = build_report(
            self._runs(64.8),
            self._micro(),
            {
                "end_to_end": {
                    "runs_agent_steps_per_second": [
                        63.5,
                        64.4,
                        64.3,
                    ],
                },
            },
            sources={"micro": {"path": "micro", "sha256": "hash"}},
        )
        self.assertTrue(report["passed"])
        self.assertFalse(report["decision"]["adopt"])
        self.assertEqual(
            report["decision"]["reason"],
            "gain_does_not_exceed_comparison_run_variability",
        )

    def test_saved_micro_covers_all_batches_and_profiler(self) -> None:
        report = json.loads(self.MICRO.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["exact_output_equivalence"]["all"])
        self.assertEqual(
            set(report["fixed_input_forward"]),
            {str(batch) for batch in BATCH_SIZES},
        )
        profiler = report["profiler"]
        trace = profiler["trace"]
        self.assertEqual(profiler["batch_size"], 4)
        self.assertGreater(trace["kernel_event_count"], 0)
        self.assertGreater(trace["kernel_launch_event_count"], 0)
        self.assertGreater(trace["synchronization_event_count"], 0)
        trace_path = self.ROOT / profiler["compressed_trace_path"]
        self.assertEqual(
            self._sha256(trace_path),
            profiler["compressed_trace_sha256"],
        )

    def test_saved_end_to_end_report_is_complete(self) -> None:
        report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertFalse(report["decision"]["adopt"])
        self.assertEqual(
            len(report["end_to_end"]["runs_agent_steps_per_second"]),
            FORMAL_RUNS,
        )
        self.assertLessEqual(
            report["end_to_end"]["relative_gain"],
            report["end_to_end"][
                "comparison_three_run_relative_range"
            ],
        )
        self.assertTrue(report["equivalence"]["micro_exact_outputs"])
        self.assertTrue(report["integrity"]["no_abnormal_exits"])
        self.assertEqual(len(report["integrity"]["checkpoint_sha256"]), 1)
        for source in report["sources"].values():
            entries = source if isinstance(source, list) else [source]
            for entry in entries:
                path = self.ROOT / entry["path"]
                self.assertEqual(self._sha256(path), entry["sha256"])


if __name__ == "__main__":
    unittest.main()
