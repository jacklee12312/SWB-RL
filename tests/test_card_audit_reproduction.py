# -*- coding: utf-8 -*-
"""Portable replay contracts for saved card-audit failures."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.reproduce_card_audit_failure import replay_saved_failure


SOURCE_REPORT = Path(
    "data/reports/card_bug_audit/reproductions/"
    "SWB-CARD-0008-random-self-play.json"
)


class CardAuditReproductionTests(unittest.TestCase):
    def test_swb_card_0008_exact_saved_prefix_reaches_official_stats(self) -> None:
        report = replay_saved_failure(
            source_report=SOURCE_REPORT,
            database=Path("data/cards.sqlite3"),
            capture_after_action_count=107,
        )

        self.assertIsNone(report["exception"])
        self.assertEqual(report["action_count"], 107)
        self.assertEqual(report["illegal_action_indices"], [])
        capture = report["captured_state_after_action"]
        self.assertEqual(capture["command"]["command_type"], "SuperEvolve")
        self.assertIn("unit_id=69", capture["command"]["command"])
        target = next(
            entity
            for player in capture["state"]["players"]
            for entity in player["board"]
            if entity["entity_id"] == 69
        )
        self.assertEqual(target["card_id"], 10154120)
        self.assertEqual(
            (target["attack"], target["health"], target["max_health"]),
            (8, 4, 4),
        )
        self.assertTrue(target["super_evolved"])
        self.assertEqual(
            [
                (
                    modifier["attack_delta"],
                    modifier["health_delta"],
                )
                for modifier in target["stat_modifiers"]
            ],
            [(2, 0)],
        )


if __name__ == "__main__":
    unittest.main()
