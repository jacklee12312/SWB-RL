from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.summarize_training_speed_stage_2_7_baseline import (
    build_report,
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


if __name__ == "__main__":
    unittest.main()
