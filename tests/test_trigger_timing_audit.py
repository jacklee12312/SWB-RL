"""Contracts for the full-pool trigger timing and batch audit."""

from __future__ import annotations

import json
import unittest

from scripts.report_trigger_timing_audit import (
    CHECKLIST_CONTRACTS,
    DEFAULT_CLOSURE,
    DEFAULT_MARKDOWN,
    DEFAULT_OUTPUT,
    REQUIRED_TRIGGER_CATEGORIES,
    ROOT,
    build_report,
    render_json,
    render_markdown,
)
from swb.db.repository import CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import EndTurn
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.resolution import (
    MAX_RESOLUTION_STEPS,
    GameEngine,
)
from swb.engine.state import Phase, Unit
from tests.test_last_words import card


def _timing_engine(rulebook: RuleBook) -> GameEngine:
    engine = GameEngine(
        [card(810000 + index) for index in range(40)],
        [card(820000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=17001,
        rulebook=rulebook,
    )
    engine.reset(seed=17001)
    engine.state.phase = Phase.MAIN
    engine.state.active_player = 0
    engine.players[0].board.clear()
    engine.players[1].board.clear()
    return engine


class TriggerTimingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_scope_matches_dynamic_database_rulebook_and_closure(self):
        cards = CardRepository(ROOT / "data/cards.sqlite3").all_cards()
        closure = json.loads(
            (ROOT / DEFAULT_CLOSURE).read_text(encoding="utf-8")
        )
        scope = self.report["scope"]
        self.assertEqual(scope["card_count"], len(cards))
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
            len(closure["cards"]),
        )
        self.assertGreater(scope["rule_trigger_definition_count"], 0)
        self.assertGreater(scope["listener_definition_count"], 0)
        self.assertGreater(scope["emblem_trigger_definition_count"], 0)
        self.assertGreater(scope["faith_trigger_definition_count"], 0)

    def test_required_trigger_matrix_has_sources_and_executable_evidence(self):
        rows = {
            row["category"]: row for row in self.report["trigger_matrix"]
        }
        self.assertTrue(set(REQUIRED_TRIGGER_CATEGORIES) <= set(rows))
        for category in REQUIRED_TRIGGER_CATEGORIES:
            with self.subTest(category=category):
                row = rows[category]
                self.assertGreater(row["source_card_count"], 0)
                self.assertGreater(row["source_record_count"], 0)
                self.assertEqual(
                    row["source_record_count"],
                    sum(
                        inventory_row["category_record_counts"].get(
                            category,
                            0,
                        )
                        for inventory_row in self.report["inventory"]
                        if not inventory_row["synthetic_demo"]
                    ),
                )
                self.assertTrue(row["test_evidence"])
                self.assertTrue(
                    all(item["passed"] for item in row["test_evidence"])
                )
                self.assertTrue(row["passed"])

    def test_every_production_source_has_coverage_and_test_evidence(self):
        rows = [
            row
            for row in self.report["inventory"]
            if not row["synthetic_demo"]
        ]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(card_id=row["card_id"]):
                self.assertEqual(row["issues"], [])
                self.assertTrue(row["test_evidence"])
                self.assertTrue(row["passed"])

    def test_synthetic_sources_are_isolated_to_named_demo_files(self):
        rows = [
            row
            for row in self.report["inventory"]
            if row["synthetic_demo"]
        ]
        self.assertTrue(rows)
        self.assertEqual(
            len(rows),
            self.report["summary"]["synthetic_demo_source_count"],
        )
        for row in rows:
            with self.subTest(card_id=row["card_id"]):
                self.assertTrue(row["demo_rule_files"])
                self.assertTrue(
                    all(
                        path.endswith("_demo.json")
                        for path in row["demo_rule_files"]
                    )
                )
                self.assertTrue(row["test_evidence"])
                self.assertTrue(row["passed"])

    def test_all_checklist_contracts_have_live_test_and_source_evidence(self):
        rows = {
            row["contract_id"]: row
            for row in self.report["behavior_contracts"]
        }
        self.assertEqual(
            set(rows),
            {
                definition["contract_id"]
                for definition in CHECKLIST_CONTRACTS
            },
        )
        for contract_id, row in rows.items():
            with self.subTest(contract_id=contract_id):
                self.assertTrue(row["test_evidence"])
                self.assertTrue(
                    all(item["passed"] for item in row["test_evidence"])
                )
                self.assertEqual(
                    row["missing_external_evidence_ids"],
                    [],
                )
                self.assertTrue(row["passed"])

    def test_official_evidence_distinguishes_qa_from_retained_regression(self):
        rows = {
            row["evidence_id"]: row
            for row in self.report["external_evidence"]
        }
        self.assertEqual(
            set(rows),
            {
                "SWB-TIMING-OFFICIAL-001",
                "SWB-TIMING-OFFICIAL-002",
                "SWB-TIMING-CARD-003",
            },
        )
        self.assertEqual(
            rows["SWB-TIMING-OFFICIAL-001"]["authority"],
            "official_qa",
        )
        self.assertEqual(
            rows["SWB-TIMING-OFFICIAL-002"]["authority"],
            "official_qa",
        )
        self.assertEqual(
            rows["SWB-TIMING-CARD-003"]["authority"],
            "official_card_text_plus_retained_regression",
        )
        self.assertTrue(
            all(
                row["url"].startswith("https://shadowverse-wb.com/")
                for row in rows.values()
            )
        )

    def test_queued_turn_end_source_continues_after_leaving_play(self):
        destroyer_id = 870001
        queued_source_id = 870002
        rulebook = RuleBook(
            (
                CardRule(
                    card_id=destroyer_id,
                    trigger=Trigger.TURN_END,
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DESTROY,
                            target=TargetKind.ALL_OWN_UNITS,
                        ),
                    ),
                ),
                CardRule(
                    card_id=queued_source_id,
                    trigger=Trigger.TURN_END,
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_LEADER,
                            target=TargetKind.ENEMY_LEADER,
                            amount=3,
                        ),
                    ),
                ),
            )
        )
        engine = _timing_engine(rulebook)
        destroyer = Unit.summon(
            card(destroyer_id, life=2),
            entity_id=engine.state.allocate_entity_id(),
        )
        queued_source = Unit.summon(
            card(queued_source_id, life=2),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board = [destroyer, queued_source]

        engine.apply(EndTurn(0))

        self.assertNotIn(destroyer, engine.players[0].board)
        self.assertNotIn(queued_source, engine.players[0].board)
        self.assertEqual(engine.players[1].health, 17)
        engine.assert_invariants()

    def test_loop_guard_and_explicit_unsupported_boundary_are_auditable(self):
        self.assertEqual(
            self.report["scope"]["max_resolution_steps"],
            MAX_RESOLUTION_STEPS,
        )
        self.assertGreater(MAX_RESOLUTION_STEPS, 0)
        rows = self.report["explicit_unsupported"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["mechanism"],
            "death_batch_start_emblem_trigger",
        )
        self.assertEqual(rows[0]["production_source_count"], 0)
        self.assertTrue(rows[0]["test_evidence"]["passed"])
        self.assertTrue(rows[0]["passed"])

    def test_saved_reports_match_deterministic_generation(self):
        self.assertEqual(
            (ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"),
            render_json(self.report),
        )
        self.assertEqual(
            (ROOT / DEFAULT_MARKDOWN).read_text(encoding="utf-8"),
            render_markdown(self.report),
        )
        repeated = build_report()
        self.assertEqual(render_json(self.report), render_json(repeated))
        self.assertEqual(
            render_markdown(self.report),
            render_markdown(repeated),
        )

    def test_report_has_no_failures(self):
        summary = self.report["summary"]
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(summary["failures"], [])
        self.assertTrue(summary["passed"])


if __name__ == "__main__":
    unittest.main()
