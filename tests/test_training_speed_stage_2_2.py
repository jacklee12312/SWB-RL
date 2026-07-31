from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from scripts.report_training_speed_stage_2_2 import (
    MAX_DISABLED_RELATIVE_REGRESSION,
    build_acceptance_report,
)


class TrainingSpeedStage22Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    @classmethod
    def setUpClass(cls) -> None:
        report_root = cls.ROOT / "data/reports/training_speed"
        cls.learner = json.loads(
            (
                report_root / "stage_2_2_learner_timing_smoke.json"
            ).read_text(encoding="utf-8")
        )
        cls.disabled = json.loads(
            (
                report_root / "stage_2_2_profiling_disabled_smoke.json"
            ).read_text(encoding="utf-8")
        )
        cls.baseline = json.loads(
            (report_root / "baseline_summary.json").read_text(
                encoding="utf-8"
            )
        )
        cls.saved = json.loads(
            (report_root / "stage_2_2_acceptance.json").read_text(
                encoding="utf-8"
            )
        )

    def build(self, learner=None, disabled=None, baseline=None):
        return build_acceptance_report(
            self.learner if learner is None else learner,
            self.disabled if disabled is None else disabled,
            self.baseline if baseline is None else baseline,
            source_paths={
                "baseline": "baseline.json",
                "learner": "learner.json",
                "profiling_disabled": "disabled.json",
            },
            source_sha256={
                "baseline": "baseline-hash",
                "learner": "learner-hash",
                "profiling_disabled": "disabled-hash",
            },
        )

    def test_saved_acceptance_passes_every_requirement(self) -> None:
        self.assertTrue(self.saved["passed"])
        self.assertTrue(all(
            item["passed"]
            for item in self.saved["requirements"].values()
        ))
        stages = self.saved["learner_profile"]["stage_breakdown"]
        self.assertGreaterEqual(
            stages["pipeline"]["accounted_fraction"],
            0.90,
        )
        self.assertGreaterEqual(
            stages["collect"]["accounted_fraction"],
            0.90,
        )
        self.assertGreaterEqual(
            stages["update"]["accounted_fraction"],
            0.90,
        )
        for source in self.saved["sources"].values():
            path = self.ROOT / source["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                source["sha256"],
            )

    def test_builder_rejects_fewer_than_three_disabled_samples(self) -> None:
        disabled = copy.deepcopy(self.disabled)
        excluded = int(
            disabled["steady_state"]["excluded_warmup_updates"]
        )
        disabled["iterations"] = disabled["iterations"][: excluded + 2]
        with self.assertRaisesRegex(ValueError, "three samples"):
            self.build(disabled=disabled)

    def test_disabled_overhead_threshold_is_enforced(self) -> None:
        disabled = copy.deepcopy(self.disabled)
        excluded = int(
            disabled["steady_state"]["excluded_warmup_updates"]
        )
        for iteration in disabled["iterations"][excluded:]:
            iteration["elapsed_seconds"] *= 2.0
        report = self.build(disabled=disabled)
        guard = report["requirements"][
            "profiling_disabled_has_no_obvious_throughput_regression"
        ]
        self.assertFalse(guard["passed"])
        self.assertLess(
            guard["relative_delta"],
            -MAX_DISABLED_RELATIVE_REGRESSION,
        )
        self.assertFalse(report["passed"])

    def test_effective_tokens_and_padding_cover_every_slot(self) -> None:
        for iteration in self.learner["iterations"]:
            update = iteration["update"]
            self.assertEqual(
                update["learner_effective_tokens"]
                + update["learner_padding_tokens"],
                update["learner_token_slots"],
            )
            self.assertAlmostEqual(
                update["learner_effective_token_fraction"]
                + update["learner_padding_fraction"],
                1.0,
            )
            self.assertEqual(
                update["learner_profiled_minibatches"],
                update["minibatches"],
            )
            cuda_components = sum(
                update[field]
                for field in (
                    "learner_host_to_device_seconds",
                    "learner_forward_seconds",
                    "learner_loss_seconds",
                    "learner_backward_seconds",
                    "learner_gradient_clip_seconds",
                    "learner_optimizer_seconds",
                )
            )
            self.assertAlmostEqual(
                cuda_components,
                update["learner_cuda_component_seconds"],
            )
            self.assertGreaterEqual(
                update["learner_total_host_synchronization_seconds"],
                update["learner_optimizer_synchronize_seconds"],
            )


if __name__ == "__main__":
    unittest.main()
