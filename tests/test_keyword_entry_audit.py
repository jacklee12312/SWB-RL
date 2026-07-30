"""Contracts for the full-pool keyword provenance and entry-method audit."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.report_keyword_entry_audit import (
    DEFAULT_CLOSURE,
    DEFAULT_MARKDOWN,
    DEFAULT_OUTPUT,
    ENTRY_METHODS,
    KEYWORDS,
    ROOT,
    SOURCE_CATEGORIES,
    build_report,
    render_json,
    render_markdown,
)
from swb.db.repository import CardRepository


class KeywordEntryAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_scope_matches_the_dynamic_database_and_frozen_closure(self):
        cards = CardRepository(ROOT / "data/cards.sqlite3").all_cards()
        closure = json.loads(
            (ROOT / DEFAULT_CLOSURE).read_text(encoding="utf-8")
        )
        closure_ids = {
            int(row["card_id"])
            for row in closure["cards"]
        }
        scope = self.report["scope"]
        self.assertEqual(scope["database_card_count"], len(cards))
        self.assertEqual(
            scope["collectible_card_count"],
            sum(card.is_collectible for card in cards),
        )
        self.assertEqual(
            scope["generated_card_count"],
            sum(not card.is_collectible for card in cards),
        )
        self.assertEqual(
            scope["training_closure_card_count"],
            len(closure_ids),
        )
        self.assertTrue(scope["scope_complete"])

    def test_entry_matrix_covers_every_required_method_and_source(self):
        rows = self.report["entry_method_matrix"]
        self.assertEqual(
            [row["entry_method"] for row in rows],
            list(ENTRY_METHODS),
        )
        for row in rows:
            with self.subTest(entry_method=row["entry_method"]):
                self.assertEqual(
                    row["source_categories"],
                    list(SOURCE_CATEGORIES),
                )
                self.assertTrue(row["evidence_files_exist"])
                self.assertGreater(row["actual_contract_case_count"], 0)
                self.assertTrue(row["actual_contract_passed"])
                self.assertTrue(row["passed"])

    def test_all_runtime_keyword_state_transitions_pass(self):
        rows = self.report["state_contracts"]
        self.assertEqual(
            [row["keyword"] for row in rows],
            list(KEYWORDS),
        )
        for row in rows:
            with self.subTest(keyword=row["keyword"]):
                self.assertTrue(row["printed_state"])
                self.assertTrue(row["permanent_add"])
                self.assertTrue(row["permanent_remove"])
                self.assertTrue(row["temporary_add"])
                self.assertTrue(row["temporary_expiry"])
                self.assertTrue(row["silence_clears"])
                self.assertTrue(row["passed"])

    def test_play_summon_transform_and_zone_reset_contracts_pass(self):
        entry_rows = {
            row["keyword"]: row for row in self.report["entry_contracts"]
        }
        reset_rows = {
            row["keyword"]: row
            for row in self.report["zone_reset_contracts"]
        }
        self.assertEqual(set(entry_rows), set(KEYWORDS))
        self.assertEqual(set(reset_rows), set(KEYWORDS))
        for keyword in KEYWORDS:
            with self.subTest(keyword=keyword):
                entry = entry_rows[keyword]
                self.assertTrue(entry["normal_play"])
                self.assertTrue(entry["enhance_play"])
                self.assertTrue(entry["transform"])
                self.assertTrue(all(entry["summon_causes"].values()))
                self.assertTrue(entry["passed"])
                reset = reset_rows[keyword]
                self.assertTrue(reset["return_to_hand_resets_dynamic"])
                self.assertTrue(reset["transform_resets_dynamic"])
                self.assertTrue(reset["passed"])

    def test_zone_backed_summons_and_copies_execute_for_every_keyword(self):
        rows = self.report["special_entry_contracts"]
        self.assertEqual(
            [row["keyword"] for row in rows],
            list(KEYWORDS),
        )
        for row in rows:
            with self.subTest(keyword=row["keyword"]):
                self.assertTrue(row["summon_from_deck"])
                self.assertTrue(row["reanimate"])
                self.assertTrue(row["definition_copy"])
                self.assertTrue(row["exact_copy_preserves_dynamic"])
                self.assertTrue(
                    row["fanfare_skipped_for_non_play_entries"]
                )
                self.assertTrue(row["passed"])

    def test_manual_and_effect_evolution_trigger_scopes_are_distinct(self):
        rows = {
            row["evolution_method"]: row
            for row in self.report["evolution_contracts"]
        }
        self.assertEqual(
            set(rows),
            {
                "normal_evolution",
                "super_evolution",
                "effect_evolution",
                "effect_super_evolution",
            },
        )
        self.assertTrue(rows["normal_evolution"]["keyword_evolve_trigger"])
        self.assertFalse(
            rows["normal_evolution"]["keyword_super_evolve_trigger"]
        )
        self.assertTrue(
            rows["super_evolution"]["keyword_evolve_trigger"]
        )
        self.assertTrue(
            rows["super_evolution"]["keyword_super_evolve_trigger"]
        )
        self.assertFalse(
            rows["effect_evolution"]["keyword_evolve_trigger"]
        )
        self.assertTrue(
            rows["effect_evolution"]["self_evolved_trigger"]
        )
        self.assertFalse(
            rows["effect_super_evolution"][
                "keyword_super_evolve_trigger"
            ]
        )
        self.assertTrue(
            rows["effect_super_evolution"][
                "self_super_evolved_trigger"
            ]
        )
        self.assertTrue(all(row["passed"] for row in rows.values()))

    def test_storm_rush_commands_and_rl_masks_agree(self):
        expected = {
            "storm_leader": True,
            "storm_follower": True,
            "rush_leader": False,
            "rush_follower": True,
            "ward_blocks_leader": False,
            "ward_blocks_other_follower": False,
            "ward_is_required_target": True,
            "ambush_is_not_attack_target": False,
            "intimidate_is_not_attack_target": False,
            "ordinary_follower_remains_target": True,
            "two_attack_capacity_first": True,
            "two_attack_capacity_second": True,
            "cannot_attack_restriction": False,
            "cannot_be_targeted_blocks_play": False,
        }
        rows = self.report["attack_mask_contracts"]
        self.assertEqual(
            {row["case"]: row["expected"] for row in rows},
            expected,
        )
        for row in rows:
            with self.subTest(case=row["case"]):
                self.assertEqual(
                    row["command_legal"],
                    row["action_mask_legal"],
                )
                self.assertEqual(row["command_legal"], row["expected"])
                self.assertTrue(row["passed"])

    def test_full_pool_sources_have_exact_or_generated_evidence(self):
        self.assertTrue(self.report["cards"])
        for row in self.report["cards"]:
            with self.subTest(card_id=row["card_id"]):
                self.assertTrue(row["test_evidence"])
                self.assertEqual(row["issues"], [])
                self.assertTrue(row["passed"])
                self.assertFalse(
                    set(row["intrinsic_keywords"])
                    & set(row["conditional_keywords"])
                )

    def test_zooey_regression_remains_explicit(self):
        regression = self.report["zooey_regression"]
        self.assertEqual(regression["card_id"], 10444120)
        self.assertFalse(regression["normal_has_storm"])
        self.assertTrue(regression["enhance_10_has_storm"])
        self.assertTrue(regression["conditional_source_recorded"])
        self.assertEqual(
            regression["official_evidence"]["accessed_on"],
            "2026-07-30",
        )
        self.assertIn(
            "shadowverse-wb.com",
            regression["official_evidence"]["url"],
        )
        self.assertTrue(regression["passed"])

    def test_report_has_no_failures(self):
        self.assertEqual(self.report["summary"]["failure_count"], 0)
        self.assertEqual(self.report["summary"]["failures"], [])
        self.assertTrue(self.report["summary"]["passed"])

    def test_saved_reports_match_deterministic_generation(self):
        self.assertEqual(
            (ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"),
            render_json(self.report),
        )
        self.assertEqual(
            (ROOT / DEFAULT_MARKDOWN).read_text(encoding="utf-8"),
            render_markdown(self.report),
        )

    def test_repeated_generation_is_byte_deterministic(self):
        repeated = build_report()
        self.assertEqual(render_json(self.report), render_json(repeated))
        self.assertEqual(
            render_markdown(self.report),
            render_markdown(repeated),
        )


if __name__ == "__main__":
    unittest.main()
