from __future__ import annotations

import json
import unittest

from scripts.report_ppo_league_baseline import NEW_RULE_SEEDS, render_json
from scripts.report_ppo_league_generation import (
    DEFAULT_OUTPUT,
    ROOT,
    build_generation_manifest,
)
from swb.rl.opponents import load_external_opponent_manifest


class PPOLeagueGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.saved_path = ROOT / DEFAULT_OUTPUT
        cls.saved = json.loads(cls.saved_path.read_text(encoding="utf-8"))

    def test_saved_generation_manifest_is_byte_stable(self) -> None:
        rebuilt = build_generation_manifest()
        self.assertEqual(self.saved_path.read_bytes(), render_json(rebuilt))

    def test_generation_zero_has_required_population_and_uniform_weights(
        self,
    ) -> None:
        entries = self.saved["entries"]
        trainable = [entry for entry in entries if entry["training_eligible"]]
        anchors = [entry for entry in entries if not entry["training_eligible"]]
        self.assertEqual(len(entries), 27)
        self.assertEqual(len(trainable), 24)
        self.assertEqual(len(anchors), 3)
        self.assertEqual(
            {entry["role"] for entry in anchors},
            {"anchor_only"},
        )
        self.assertEqual(
            {entry["sampling_weight"] for entry in trainable},
            {1.0},
        )
        self.assertEqual(
            {entry["sampling_weight"] for entry in anchors},
            {0.0},
        )
        for seed in NEW_RULE_SEEDS:
            seed_entries = [
                entry
                for entry in trainable
                if entry["opponent_id"].startswith(f"seed_{seed}_")
            ]
            self.assertEqual(len(seed_entries), 4)
            self.assertEqual(
                sum(entry["role"] == "candidate_final" for entry in seed_entries),
                1,
            )
            self.assertEqual(
                sum(entry["role"] == "self_history" for entry in seed_entries),
                3,
            )
        self.assertEqual(
            len({entry["model_values_sha256"] for entry in entries}),
            len(entries),
        )

    def test_loader_validates_real_files_and_selection_audit(self) -> None:
        manifest = load_external_opponent_manifest(
            self.saved_path,
            external_weight=1.0,
            repository_root=ROOT,
        )
        self.assertEqual(len(manifest.trainable_entries), 24)
        self.assertEqual(len(manifest.reference_entries), 3)
        self.assertAlmostEqual(
            sum(entry.weight for entry in manifest.trainable_entries),
            1.0,
        )
        self.assertTrue(
            self.saved["selection_audit"]["all_trainable_entries_hit"]
        )
        self.assertEqual(
            self.saved["selection_audit"]["class_pair_position_coverage"],
            98,
        )
        self.assertEqual(
            self.saved["selection_audit"]["reference_entries_hit"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
