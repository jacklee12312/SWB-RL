from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import unittest
from pathlib import Path

from scripts.assess_training_speed_stage_2_8_overlap import (
    MATERIALITY_THRESHOLD,
    build_report,
)


class TrainingSpeedStage28OverlapTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    PROFILE = (
        ROOT
        / "data/reports/training_speed/stage_2_8_overlap_profile.json"
    )
    ADOPTED = (
        ROOT
        / "data/reports/training_speed/"
        "stage_2_7_b_batched_learner_001_end_to_end.json"
    )
    REPORT = (
        ROOT
        / "data/reports/training_speed/stage_2_8_overlap_gate.json"
    )

    @staticmethod
    def _json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_saved_gate_is_current_and_closes_synchronous_overlap(
        self,
    ) -> None:
        saved = self._json(self.REPORT)
        rebuilt = build_report(
            self._json(self.PROFILE),
            self._json(self.ADOPTED),
            sources=saved["sources"],
        )
        self.assertEqual(saved, rebuilt)
        self.assertTrue(saved["passed"])
        self.assertTrue(all(saved["integrity"].values()))

        candidate = saved["a_overlap_001"]
        rollout = candidate["rollout_same_generation"]
        learner = candidate["learner_next_minibatch"]
        holes = candidate["pipeline_holes"]
        self.assertLess(
            rollout[
                "grouped_prepare_and_h2d_fraction_of_pipeline_wall"
            ],
            MATERIALITY_THRESHOLD,
        )
        self.assertLess(
            learner["preparation_and_h2d_fraction_of_pipeline_wall"],
            MATERIALITY_THRESHOLD,
        )
        self.assertGreater(
            holes["fraction_of_pipeline_wall"],
            MATERIALITY_THRESHOLD,
        )
        self.assertFalse(
            holes["independently_schedulable_cuda_work_available"]
        )
        self.assertFalse(
            candidate["decision"]["advance_to_implementation"]
        )
        self.assertTrue(
            candidate["decision"]["synchronous_default_unchanged"]
        )

        for source in saved["sources"].values():
            path = self.ROOT / source["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, source["sha256"])

    def test_material_rollout_preparation_would_advance(self) -> None:
        profile = self._json(self.PROFILE)
        pipeline_wall = float(
            profile["steady_state"]["pipeline_wall_time"][
                "measured_seconds"
            ]
        )
        profile["steady_state"]["collect"]["fields"][
            "central_batch_prepare_to_device_seconds"
        ]["total"] = pipeline_wall * MATERIALITY_THRESHOLD
        report = build_report(
            profile,
            self._json(self.ADOPTED),
            sources={},
        )
        self.assertTrue(
            report["a_overlap_001"]["rollout_same_generation"][
                "advance_to_implementation"
            ]
        )
        self.assertFalse(report["passed"])

    def test_material_pipeline_wait_does_not_bypass_causality_gate(
        self,
    ) -> None:
        profile = copy.deepcopy(self._json(self.PROFILE))
        report = build_report(
            profile,
            self._json(self.ADOPTED),
            sources={},
        )
        holes = report["a_overlap_001"]["pipeline_holes"]
        self.assertTrue(holes["meets_size_threshold"])
        self.assertFalse(holes["advance_to_overlap_implementation"])

    def test_async_is_explicitly_a_separate_algorithm_experiment(
        self,
    ) -> None:
        report = self._json(self.REPORT)
        async_gate = report["c_async_001"]
        self.assertEqual(async_gate["class"], "C")
        self.assertFalse(
            async_gate["decision"]["advance_in_stage_2_8"]
        )
        self.assertEqual(
            async_gate["decision"]["disposition"],
            "deferred_separate_algorithm_experiment",
        )
        required = async_gate["required_if_reopened"]
        self.assertTrue(required["trajectory_policy_generation"])
        self.assertTrue(required["maximum_policy_lag"])
        self.assertTrue(required["behavior_policy_log_probability"])
        self.assertTrue(required["three_seed_learning_curves"])
        self.assertTrue(required["fixed_match_evaluation"])


class TrainingSpeedStage28AcceptanceTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REGISTRY_COMMIT = (
        "8341c6de0903d46980104cf99e5654c725717138"
    )
    REPORT = (
        ROOT
        / "data/reports/training_speed/stage_2_8_acceptance.json"
    )

    def test_saved_stage_acceptance_passes_every_gate(self) -> None:
        report = json.loads(
            self.REPORT.read_text(encoding="utf-8")
        )
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(
            report["checklist"]["unchecked_items"],
            [],
        )
        self.assertGreaterEqual(
            report["mandatory_verification"]["unittest"][
                "tests_run"
            ],
            2910,
        )
        self.assertEqual(
            report["mandatory_verification"]["unittest"][
                "skipped"
            ],
            1,
        )
        self.assertTrue(
            all(
                row["unchanged"]
                for row in report["source_contracts"].values()
            )
        )
        for name, source in report["sources"].items():
            if name == "registry":
                content = subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{self.REGISTRY_COMMIT}:{source['path']}",
                    ],
                    cwd=self.ROOT,
                )
            else:
                content = (
                    self.ROOT / source["path"]
                ).read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            self.assertEqual(digest, source["sha256"])


if __name__ == "__main__":
    unittest.main()
