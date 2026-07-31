from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.summarize_training_speed_stage_2_7_baseline import (
    build_report,
)
from scripts.verify_training_speed_stage_2_7_b_batched_learner_001 import (
    CONFIGURATION as BATCHED_CONFIGURATION,
    build_report as build_batched_end_to_end_report,
)


class TrainingSpeedStage27BaselineTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    PROFILE = (
        ROOT
        / "data/reports/training_speed/"
        "stage_2_7_learner_baseline.json"
    )
    COMPARISON = (
        ROOT
        / "data/reports/training_speed/"
        "stage_2_5_a_obs_001.json"
    )
    SUMMARY = (
        ROOT
        / "data/reports/training_speed/"
        "stage_2_7_learner_baseline_summary.json"
    )

    @staticmethod
    def _json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def test_saved_summary_is_current_and_profile_is_frozen(self) -> None:
        saved = self._json(self.SUMMARY)
        self.assertEqual(
            saved,
            build_report(
                self._json(self.PROFILE),
                self._json(self.COMPARISON),
            ),
        )
        self.assertTrue(saved["passed"])
        self.assertTrue(saved["checkpoint"]["unchanged"])
        self.assertEqual(
            saved["configuration"]["steady_state_samples"],
            3,
        )
        for source in saved["sources"].values():
            path = self.ROOT / source["path"]
            self.assertEqual(self._sha256(path), source["sha256"])

    def test_materiality_gates_separate_prep_from_padded_compute(
        self,
    ) -> None:
        report = self._json(self.SUMMARY)
        gates = report["materiality_gates"]
        self.assertFalse(
            gates["buffer_only"]["advance_to_implementation"]
        )
        self.assertFalse(
            gates["optimizer_group"]["advance_to_implementation"]
        )
        self.assertTrue(
            gates["padded_compute"]["advance_to_microbenchmark"]
        )
        self.assertTrue(
            gates["learner_amp"][
                "advance_to_numeric_and_microbenchmark"
            ]
        )
        self.assertGreater(
            report["learner"]["forward_plus_backward_fraction"],
            0.95,
        )
        self.assertGreater(
            report["learner"]["padding_slot_fraction_median"],
            0.30,
        )


class TrainingSpeedStage27APaddedCompute001Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REPORT = (
        ROOT
        / "data/reports/training_speed/"
        "stage_2_7_a_padded_compute_001_gate.json"
    )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def test_saved_gate_reclassifies_fast_but_drifting_candidate(
        self,
    ) -> None:
        report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertTrue(report["timing"]["speed_gate_passed"])
        self.assertGreater(
            report["timing"]["relative_reduction"], 0.50
        )
        self.assertFalse(
            report["numeric"]["model_after_one_update"]["allclose"]
        )
        self.assertTrue(
            report["numeric"]["optimizer_after_one_update"][
                "allclose"
            ]
        )
        self.assertFalse(report["decision"]["adopt_as_a"])
        self.assertTrue(report["decision"]["reclassify_as_b"])
        self.assertFalse(report["decision"]["run_end_to_end_as_a"])
        for source in report["sources"].values():
            path = self.ROOT / source["path"]
            self.assertEqual(self._sha256(path), source["sha256"])


class TrainingSpeedStage27BBatchedLearner001Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REPORT = (
        ROOT
        / "data/reports/training_speed/"
        "stage_2_7_b_batched_learner_001_end_to_end.json"
    )

    @staticmethod
    def _run(run_index: int, speed: float) -> dict[str, object]:
        return {
            "run_index": run_index,
            "configuration": BATCHED_CONFIGURATION,
            "profiling_switches_disabled": True,
            "checkpoint_unchanged": True,
            "checkpoint_sha256": "checkpoint",
            "measurement": {
                "agent_steps": 6_144,
                "steady_update_count": 3,
                "agent_steps_per_second": speed,
                "collect_p95_seconds": 12.0,
                "update_p95_seconds": 5.0,
                "abnormal_exit_count": 0,
                "batching": {
                    "mean_batch_size": 3.0,
                    "p95_batch_size": 6.0,
                },
                "episode_length": {
                    "p95": 120.0,
                    "maximum": 160.0,
                    "truncated_episode_count": 0,
                },
                "system": {
                    "sample_count": 10,
                    "gpu_sample_count": 10,
                    "cpu_total_median_percent": 10.0,
                    "gpu_utilization_median_percent": 50.0,
                    "gpu_memory_peak_mib": 9_000.0,
                    "ram_used_peak_bytes": 20_000_000_000,
                    "pagefile_used_peak_bytes": 21_000_000_000,
                },
            },
        }

    def test_clear_three_run_gain_adopts_after_learning_gate(
        self,
    ) -> None:
        report = build_batched_end_to_end_report(
            [
                self._run(1, 90.0),
                self._run(2, 92.0),
                self._run(3, 91.0),
            ],
            {
                "passed": True,
                "decision": {"advance_to_end_to_end": True},
                "summary": {
                    "learning_non_degradation_passed": True
                },
            },
            {
                "end_to_end": {
                    "runs_agent_steps_per_second": [
                        60.0,
                        61.0,
                        60.5,
                    ]
                }
            },
            {
                "observations": {
                    "v4.1": {
                        "agent_steps_per_second": {
                            "runs": [44.5, 44.7, 44.8],
                            "median": 44.7,
                        }
                    }
                }
            },
            sources={},
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["decision"]["adopt"])
        self.assertTrue(report["decision"]["default_enabled"])
        self.assertGreater(
            report["end_to_end"]["frozen_v4_1_relative_gain"],
            0.25,
        )

    def test_configuration_drift_is_rejected(self) -> None:
        runs = [
            self._run(1, 90.0),
            self._run(2, 91.0),
            self._run(3, 92.0),
        ]
        runs[0] = {
            **runs[0],
            "configuration": {
                **BATCHED_CONFIGURATION,
                "rollout_workers": 5,
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            "configuration drifted",
        ):
            build_batched_end_to_end_report(
                runs,
                {},
                {},
                {},
                sources={},
            )

    def test_saved_end_to_end_report_is_current_and_adopted(
        self,
    ) -> None:
        saved = json.loads(self.REPORT.read_text(encoding="utf-8"))
        sources = saved["sources"]

        def load_source(source: dict[str, object]) -> dict[str, object]:
            return json.loads(
                (self.ROOT / source["path"]).read_text(
                    encoding="utf-8"
                )
            )

        rebuilt = build_batched_end_to_end_report(
            [load_source(source) for source in sources["runs"]],
            load_source(sources["learning"]),
            load_source(sources["comparison"]),
            load_source(sources["frozen_baseline"]),
            sources=sources,
        )
        self.assertEqual(saved, rebuilt)
        self.assertTrue(saved["passed"])
        self.assertTrue(saved["decision"]["adopt"])
        self.assertTrue(saved["integrity"]["no_truncations"])
        self.assertGreater(
            saved["end_to_end"]["relative_gain"],
            saved["end_to_end"][
                "comparison_three_run_relative_range"
            ],
        )
        for name, source in sources.items():
            rows = source if name == "runs" else [source]
            for row in rows:
                self.assertEqual(
                    self._sha256(self.ROOT / row["path"]),
                    row["sha256"],
                )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


class TrainingSpeedStage27BLearnerAmp001Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REPORT = (
        ROOT
        / "data/reports/training_speed/"
        "stage_2_7_b_learner_amp_001_gate.json"
    )

    def test_saved_amp_gate_closes_below_variability_variants(
        self,
    ) -> None:
        report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        self.assertTrue(report["passed"])
        self.assertFalse(
            report["decision"]["advance_to_three_seed_learning"]
        )
        self.assertFalse(report["decision"]["run_end_to_end"])
        self.assertFalse(report["decision"]["default_enabled"])
        self.assertEqual(
            report["decision"]["advancing_variants"],
            [],
        )
        for name in ("float16", "bfloat16"):
            variant = report["variants"][name]
            self.assertTrue(variant["stable"])
            self.assertFalse(
                variant[
                    "speed_exceeds_current_three_run_variability"
                ]
            )
            self.assertFalse(
                variant["advance_to_three_seed_learning"]
            )
            self.assertTrue(
                variant["model_after_one_update_drift"]["finite"]
            )
            self.assertTrue(
                all(
                    run["grad_scaler"]["enabled"]
                    for run in variant["runs"]
                )
            )
        for source in report["sources"].values():
            path = self.ROOT / source["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, source["sha256"])


if __name__ == "__main__":
    unittest.main()
