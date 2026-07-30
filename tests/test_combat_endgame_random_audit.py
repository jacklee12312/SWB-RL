"""Executable contracts for checklist section 1.9."""

from __future__ import annotations

import json
import unittest

from scripts.report_combat_endgame_random_audit import (
    CATEGORIES,
    CHECKLIST_CONTRACTS,
    DEFAULT_CLOSURE,
    DEFAULT_MARKDOWN,
    DEFAULT_OUTPUT,
    ROOT,
    build_report,
    render_json,
    render_markdown,
)
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Attack, EndTurn, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import Unit


def _card(
    card_id: int,
    *,
    attack: int = 1,
    life: int = 3,
    card_type: str = "随从",
    cost: int = 1,
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"combat-audit-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack if card_type == "随从" else None,
        life=life if card_type == "随从" else None,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def _deck(start: int) -> list[CardDefinition]:
    return [_card(start + index) for index in range(40)]


def _random_damage_engine(seed: int, *, with_targets: bool) -> GameEngine:
    spell = _card(990001, card_type="法术")
    rulebook = RuleBook((
        CardRule(
            spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.DAMAGE_UNIT,
                    TargetKind.RANDOM_ENEMY_UNIT,
                    amount=2,
                ),
            ),
        ),
    ))
    engine = GameEngine(
        _deck(810000),
        _deck(820000),
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
    )
    engine.reset(seed=seed)
    engine.players[0].mana = 10
    engine.players[0].hand[0] = spell
    if with_targets:
        engine.players[1].board = [
            Unit.summon(
                _card(991000 + index, life=5),
                entity_id=engine.state.allocate_entity_id(),
            )
            for index in range(3)
        ]
    else:
        engine.players[1].board.clear()
    return engine


def _event_projection(engine: GameEngine) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            event.type.value,
            event.player_index,
            event.source_id,
            event.target_id,
            event.amount,
            event.metadata.get("card_id"),
            event.metadata.get("damage_type"),
            event.metadata.get("winner"),
        )
        for event in engine.event_history
    )


class CombatEndgameRandomAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_scope_matches_dynamic_database_and_training_closure(self):
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

    def test_category_matrix_has_sources_and_executable_evidence(self):
        rows = {
            row["category"]: row for row in self.report["category_matrix"]
        }
        self.assertEqual(set(rows), set(CATEGORIES))
        for category in CATEGORIES:
            with self.subTest(category=category):
                row = rows[category]
                self.assertGreater(row["source_card_count"], 0)
                self.assertTrue(row["test_evidence"])
                self.assertTrue(
                    all(item["passed"] for item in row["test_evidence"])
                )
                self.assertTrue(row["passed"])

    def test_every_relevant_full_pool_source_has_permanent_evidence(self):
        rows = self.report["inventory"]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(card_id=row["card_id"]):
                self.assertEqual(row["issues"], [])
                self.assertTrue(row["test_evidence"])
                self.assertTrue(row["passed"])

    def test_all_ten_checklist_contracts_are_live(self):
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
        self.assertEqual(len(rows), 10)
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

    def test_engine_random_callsites_use_owned_rng(self):
        audit = self.report["engine_rng_audit"]
        self.assertGreater(audit["callsite_count"], 0)
        self.assertEqual(audit["violation_count"], 0)
        self.assertEqual(audit["violations"], [])
        self.assertTrue(all(row["passed"] for row in audit["callsites"]))
        self.assertTrue(audit["passed"])

    def test_random_choice_is_event_visible_and_full_replay_is_identical(self):
        first = _random_damage_engine(19001, with_targets=True)
        second = _random_damage_engine(19001, with_targets=True)

        first_transition = first.apply(PlayCard(0, 0))
        second_transition = second.apply(PlayCard(0, 0))

        first_damage = [
            event
            for event in first_transition.events
            if event.type is EventType.DAMAGE_APPLIED
        ]
        second_damage = [
            event
            for event in second_transition.events
            if event.type is EventType.DAMAGE_APPLIED
        ]
        self.assertEqual(len(first_damage), 1)
        self.assertEqual(len(second_damage), 1)
        self.assertIsNotNone(first_damage[0].target_id)
        self.assertEqual(first_damage[0].target_id, second_damage[0].target_id)
        self.assertEqual(
            first.deterministic_fingerprint(),
            second.deterministic_fingerprint(),
        )
        self.assertEqual(_event_projection(first), _event_projection(second))
        self.assertEqual(first.winner, second.winner)

    def test_no_candidate_skip_and_illegal_branch_do_not_consume_rng(self):
        engine = _random_damage_engine(19002, with_targets=False)
        before_rng = engine.random.getstate()

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.random.getstate(), before_rng)
        attacker = Unit.summon(
            _card(992001),
            entity_id=engine.state.allocate_entity_id(),
        )
        attacker.can_attack = False
        engine.players[0].board = [attacker]
        before_rng = engine.random.getstate()
        before_state = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Attack(0, attacker.entity_id, None))
        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertEqual(engine.deterministic_fingerprint(), before_state)

    def test_terminated_match_rejects_commands_without_visible_mutation(self):
        engine = _random_damage_engine(19003, with_targets=False)
        attacker = Unit.summon(
            _card(993001, attack=5),
            entity_id=engine.state.allocate_entity_id(),
        )
        attacker.can_attack = True
        attacker.summoned_this_turn = False
        engine.players[0].board = [attacker]
        engine.players[1].health = 5

        transition = engine.apply(Attack(0, attacker.entity_id, None))

        self.assertTrue(transition.terminated)
        self.assertEqual(transition.winner, 0)
        self.assertEqual(engine.legal_commands(), [])
        before_state = engine.deterministic_fingerprint()
        before_rng = engine.random.getstate()
        before_events = _event_projection(engine)
        before_logs = tuple(engine.logs)
        with self.assertRaises(IllegalCommand):
            engine.apply(EndTurn(0))
        self.assertEqual(engine.deterministic_fingerprint(), before_state)
        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertEqual(_event_projection(engine), before_events)
        self.assertEqual(tuple(engine.logs), before_logs)

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
