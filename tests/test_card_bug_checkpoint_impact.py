# -*- coding: utf-8 -*-
"""Contracts for the checklist 1.13 old-checkpoint impact report."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from swb.engine.environment import ShadowverseEnv
from swb.rl.runtime import hash_rule_directory


REPORT_PATH = Path(
    "data/reports/card_bug_audit/repros/checkpoint_impact.json"
)


class CardBugCheckpointImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    def test_report_is_read_only_and_preserves_old_models(self) -> None:
        self.assertEqual(
            self.report["scope"]["scan_mode"],
            "read-only metadata inspection",
        )
        self.assertEqual(
            self.report["scope"]["files_modified_or_deleted"],
            0,
        )
        self.assertTrue(
            self.report["policy"]["preserve_old_checkpoints"]
        )
        for checkpoint in self.report["checkpoints"]:
            self.assertEqual(
                checkpoint["preservation"],
                "keep_read_only_historical_artifact",
            )

    def test_pre_fix_models_are_not_fair_post_fix_comparators(self) -> None:
        summary = self.report["summary"]
        self.assertGreater(summary["checkpoint_count"], 0)
        self.assertEqual(
            summary["readable_count"] + summary["unreadable_count"],
            summary["checkpoint_count"],
        )
        self.assertGreater(
            summary["potentially_affected_pre_fix_count"],
            0,
        )
        for checkpoint in self.report["checkpoints"]:
            self.assertFalse(
                checkpoint[
                    "fair_strength_comparison_with_post_fix_models"
                ]
            )

    def test_report_records_engine_hash_compatibility_caveat(self) -> None:
        fix = self.report["fix"]
        self.assertEqual(
            fix["commit"],
            "b6f1d95cd2336cc86772e717e5bd09440a8f38a7",
        )
        self.assertEqual(
            fix["current_rulebook_sha256"],
            hash_rule_directory(
                ShadowverseEnv.DEFAULT_RULE_DIRECTORY
            ),
        )
        self.assertIn("Python engine semantics", fix["compatibility_caveat"])


if __name__ == "__main__":
    unittest.main()
