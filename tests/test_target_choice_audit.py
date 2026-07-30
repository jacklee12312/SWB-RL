from __future__ import annotations

import unittest
from pathlib import Path

from scripts.report_target_choice_audit import (
    MANUAL_TARGETS,
    ROOT,
    build_report,
    render_markdown,
)
from swb.db.repository import CardRepository
from swb.engine.effects import TargetKind


class TargetChoiceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()
        cls.contracts = cls.report["contracts"]

    def test_current_database_and_rule_sources_are_scanned_dynamically(self):
        cards = CardRepository(ROOT / "data/cards.sqlite3").all_cards()
        scope = self.report["scope"]
        summary = self.report["summary"]

        self.assertEqual(scope["card_count"], len(cards))
        self.assertEqual(
            scope["collectible_card_count"],
            sum(card.is_collectible for card in cards),
        )
        self.assertEqual(
            summary["source_card_count"],
            summary["collectible_source_card_count"]
            + summary["generated_source_card_count"],
        )
        self.assertGreater(summary["training_source_card_count"], 0)
        self.assertGreater(summary["global_source_count"], 0)

    def test_synthetic_demo_rules_are_explicitly_separated(self):
        demo_rows = [
            row for row in self.report["inventory"] if row["synthetic_demo"]
        ]
        self.assertEqual(
            len(demo_rows),
            self.report["summary"]["synthetic_demo_source_count"],
        )
        self.assertGreater(len(demo_rows), 0)
        for row in demo_rows:
            self.assertTrue(row["demo_rule_file"].endswith("_demo.json"))
            self.assertTrue(row["passed"])
            self.assertTrue(row["test_evidence"])

    def test_every_production_source_has_coverage_and_existing_evidence(self):
        production_rows = [
            row
            for row in self.report["inventory"]
            if not row["synthetic_demo"]
        ]
        self.assertTrue(production_rows)
        for row in production_rows:
            with self.subTest(card_id=row["card_id"]):
                self.assertTrue(row["passed"], row["issues"])
                self.assertTrue(row["test_evidence"])
                for evidence in row["test_evidence"]:
                    self.assertTrue((ROOT / evidence).is_file(), evidence)

    def test_all_manual_target_kinds_have_domain_and_order_contracts(self):
        rows = self.contracts["candidate_domains"]
        self.assertEqual(
            [row["target"] for row in rows],
            [target.value for target in MANUAL_TARGETS],
        )
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertEqual(
            set(row["target"] for row in rows),
            {
                TargetKind.OWN_UNIT.value,
                TargetKind.ENEMY_UNIT.value,
                TargetKind.ANY_UNIT.value,
                TargetKind.OWN_UNIT_OR_LEADER.value,
                TargetKind.ENEMY_UNIT_OR_LEADER.value,
                TargetKind.ANY_UNIT_OR_LEADER.value,
                TargetKind.OWN_AMULET.value,
                TargetKind.ENEMY_AMULET.value,
                TargetKind.ANY_AMULET.value,
                TargetKind.OWN_BOARD.value,
                TargetKind.ENEMY_BOARD.value,
                TargetKind.ANY_BOARD.value,
                TargetKind.OWN_HAND.value,
                TargetKind.OWN_GRAVEYARD_CARD.value,
            },
        )

    def test_zero_one_and_multiple_candidate_contracts_pass_atomically(self):
        rows = self.contracts["candidate_cardinalities"]
        self.assertEqual(
            [row["legal_candidate_count"] for row in rows],
            [0, 1, 2],
        )
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertTrue(rows[0]["zero_candidate_atomic"])

    def test_other_target_excludes_only_the_source(self):
        row = self.contracts["source_exclusion"]
        self.assertTrue(row["passed"])
        self.assertNotIn(
            row["source_entity_id"],
            row["actual_option_ids"],
        )
        self.assertEqual(
            row["actual_option_ids"],
            row["expected_option_ids"],
        )

    def test_target_restrictions_apply_only_to_their_target_classes(self):
        rows = self.contracts["restrictions"]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["passed"] for row in rows))

    def test_multi_target_duplicate_order_and_shortage_contracts_pass(self):
        rows = self.contracts["multi_target"]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertTrue(rows[0]["duplicate_rejection_atomic"])
        self.assertEqual(rows[2]["effective_target_count"], 2)

    def test_pending_targets_are_revalidated_after_every_required_change(self):
        rows = self.contracts["stale_targets"]
        self.assertEqual(
            {row["case"] for row in rows},
            {
                "target_died",
                "target_left_play",
                "target_transformed",
                "target_changed_controller",
                "target_failed_filter",
            },
        )
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertTrue(all(row["draw_continued"] for row in rows))

    def test_source_leave_and_selected_random_all_order_contracts_pass(self):
        source_rows = self.contracts["source_leaving"]
        mixed = self.contracts["mixed_target_order"]
        self.assertTrue(all(row["passed"] for row in source_rows))
        self.assertTrue(mixed["passed"])
        self.assertTrue(mixed["selected_left_play"])
        self.assertEqual(
            mixed["survivor_health"],
            mixed["expected_survivor_health"],
        )

    def test_no_candidate_policies_and_snapshot_restore_are_deterministic(self):
        policies = self.contracts["no_candidate_policies"]
        snapshot = self.contracts["snapshot_restore"]
        self.assertTrue(all(row["passed"] for row in policies))
        self.assertTrue(snapshot["passed"])
        self.assertTrue(snapshot["pending_choice_restored"])
        self.assertTrue(snapshot["events_equal"])
        self.assertTrue(snapshot["fingerprints_equal"])

    def test_112_action_mask_decode_command_and_ui_order_are_identical(self):
        row = self.contracts["action_order"]
        self.assertTrue(row["passed"])
        self.assertEqual(row["action_size"], 112)
        self.assertEqual(
            row["decoded_option_ids"],
            row["legal_command_option_ids"],
        )
        self.assertEqual(
            len(row["ui_option_entity_ids"]),
            len(row["enabled_choice_actions"]),
        )

    def test_report_has_no_failures_and_markdown_is_reviewable(self):
        summary = self.report["summary"]
        self.assertTrue(summary["passed"], summary["failures"])
        self.assertEqual(summary["inventory_issue_count"], 0)
        self.assertEqual(summary["contract_failure_count"], 0)
        markdown = render_markdown(self.report)
        self.assertIn("Result: **PASS**", markdown)
        self.assertIn("## Manual target-domain matrix", markdown)
        self.assertIn("## Source inventory", markdown)


if __name__ == "__main__":
    unittest.main()
