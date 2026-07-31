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
from scripts.verify_training_speed_stage_2_6_a_net_002 import (
    BATCH_SIZES as A_NET_002_BATCH_SIZES,
    CONFIGURATION as A_NET_002_CONFIGURATION,
    STATIC_BUFFERS,
    build_report as build_a_net_002_report,
)
from scripts.verify_training_speed_stage_2_6_a_net_003 import (
    CONFIGURATION as A_NET_003_CONFIGURATION,
    build_report as build_a_net_003_report,
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


class TrainingSpeedStage26ANet002Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    MICRO = (
        ROOT
        / "data/reports/training_speed/stage_2_6_a_net_002_micro.json"
    )
    REPORT = (
        ROOT
        / "data/reports/training_speed/stage_2_6_a_net_002.json"
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
                "configuration": deepcopy(A_NET_002_CONFIGURATION),
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
            "static_buffers": {
                name: {"persistent": False}
                for name in STATIC_BUFFERS
            },
            "profiler": {
                "batch_size": 4,
                "trace": {
                    "kernel_event_count": 1_902,
                    "kernel_launch_event_count": 1_902,
                    "synchronization_event_count": 8,
                },
            },
        }

    def test_summary_rejects_gain_within_comparison_variability(
        self,
    ) -> None:
        report = build_a_net_002_report(
            self._runs(64.5),
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

    def test_saved_micro_is_exact_and_buffers_are_non_persistent(
        self,
    ) -> None:
        report = json.loads(self.MICRO.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["exact_output_equivalence"]["all"])
        self.assertEqual(
            set(report["fixed_input_forward"]),
            {str(batch) for batch in A_NET_002_BATCH_SIZES},
        )
        self.assertEqual(set(report["static_buffers"]), set(STATIC_BUFFERS))
        for name, contract in report["static_buffers"].items():
            self.assertFalse(contract["persistent"], name)
            self.assertTrue(contract["matches"], name)
        trace = report["profiler"]["trace"]
        trace_path = self.ROOT / report["profiler"][
            "compressed_trace_path"
        ]
        self.assertGreater(trace["kernel_event_count"], 0)
        self.assertEqual(
            self._sha256(trace_path),
            report["profiler"]["compressed_trace_sha256"],
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
        for source in report["sources"].values():
            entries = source if isinstance(source, list) else [source]
            for entry in entries:
                path = self.ROOT / entry["path"]
                self.assertEqual(self._sha256(path), entry["sha256"])


class TrainingSpeedStage26ANet003Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    MICRO = (
        ROOT
        / "data/reports/training_speed/stage_2_6_a_net_003_micro.json"
    )
    REPORT = (
        ROOT
        / "data/reports/training_speed/stage_2_6_a_net_003.json"
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
                "configuration": deepcopy(A_NET_003_CONFIGURATION),
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
                    "kernel_event_count": 1_869,
                    "kernel_launch_event_count": 1_869,
                    "synchronization_event_count": 11,
                },
            },
        }

    def test_summary_rejects_gain_within_comparison_variability(
        self,
    ) -> None:
        report = build_a_net_003_report(
            self._runs(65.1),
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

    def test_saved_micro_is_exact_and_reduces_round_calls(self) -> None:
        report = json.loads(self.MICRO.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["exact_output_equivalence"]["all"])
        self.assertEqual(report["operator_calls"]["aten::round"], 84)
        self.assertLess(
            report["profiler"]["trace"]["kernel_event_count"],
            1_938,
        )
        trace_path = self.ROOT / report["profiler"][
            "compressed_trace_path"
        ]
        self.assertEqual(
            self._sha256(trace_path),
            report["profiler"]["compressed_trace_sha256"],
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
        for source in report["sources"].values():
            entries = source if isinstance(source, list) else [source]
            for entry in entries:
                path = self.ROOT / entry["path"]
                self.assertEqual(self._sha256(path), entry["sha256"])


class TrainingSpeedStage26ANet004Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REPORT = (
        ROOT
        / "data/reports/training_speed/"
        "stage_2_6_a_net_004_layout_projection_gate.json"
    )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def test_saved_layout_projection_gate_is_reproducible(self) -> None:
        report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertFalse(report["decision"]["implement"])
        self.assertFalse(report["decision"]["run_end_to_end"])
        self.assertEqual(
            report["source_layout_scan"],
            {"permute": 0, "contiguous": 0},
        )
        self.assertFalse(
            report["operators"]["aten::permute"]["present_in_top_50"]
        )
        self.assertFalse(
            report["operators"]["aten::contiguous"][
                "present_in_top_50"
            ]
        )
        self.assertLess(
            report["upper_bound"][
                "all_cat_and_mm_fraction_of_forward"
            ],
            report["upper_bound"][
                "comparison_three_run_relative_range"
            ],
        )
        for source in report["sources"].values():
            path = self.ROOT / source["path"]
            self.assertEqual(self._sha256(path), source["sha256"])


if __name__ == "__main__":
    unittest.main()
