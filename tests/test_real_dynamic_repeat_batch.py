# -*- coding: utf-8 -*-
"""Bounded sequential repeats and exact real-card coverage."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Evolve
from swb.engine.effects import (
    EffectKind,
    EffectOperation,
    ExprType,
    HandFilter,
    MAX_REPEAT_COUNT,
    TargetKind,
    ValueExpression,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import ResolutionLoopError
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10114130, 10313310)
SOURCE_HASHES = {
    10114130: "cc87ebe3021b9bf7c92489e2a8ec381a35824cae355dc0ef697508d6475ee1af",
    10313310: "cb9fa176773cd99b2767bd4e89b74bd53c1ef370ee72eacfa2d957166524bd08",
}
PIXIE_ID = 90011110


class DynamicRepeatSchemaTests(unittest.TestCase):
    def test_repeat_and_filtered_hand_count_expression_parse(self):
        operation = _parse_operation(
            {
                "kind": "repeat",
                "amount": {
                    "type": "controller_hand_count",
                    "filter": {"card_type": "随从", "tribe_name": "妖精"},
                },
                "operations": [
                    {
                        "kind": "damage_unit",
                        "target": "random_enemy_unit",
                        "amount": 1,
                    }
                ],
            },
            "test",
            1,
        )
        self.assertIs(operation.kind, EffectKind.REPEAT)
        self.assertIs(operation.target, TargetKind.OWN_LEADER)
        self.assertIs(operation.amount_expr.type, ExprType.CONTROLLER_HAND_COUNT)
        self.assertEqual(
            operation.amount_expr.card_filter,
            HandFilter(card_type="随从", tribe_name="妖精"),
        )
        self.assertEqual(
            [(nested.kind, nested.target) for nested in operation.repeat_operations],
            [(EffectKind.DAMAGE_UNIT, TargetKind.RANDOM_ENEMY_UNIT)],
        )

    def test_repeat_schema_rejects_unbounded_or_ambiguous_payloads(self):
        invalid_payloads = (
            ({"kind": "repeat", "amount": 1}, "non-empty"),
            (
                {
                    "kind": "repeat",
                    "amount": -1,
                    "operations": [{"kind": "draw", "target": "own_leader", "amount": 1}],
                },
                "non-negative",
            ),
            (
                {
                    "kind": "repeat",
                    "amount": MAX_REPEAT_COUNT + 1,
                    "operations": [{"kind": "draw", "target": "own_leader", "amount": 1}],
                },
                "maximum",
            ),
            (
                {
                    "kind": "repeat",
                    "amount": {"type": "target_attack"},
                    "operations": [{"kind": "draw", "target": "own_leader", "amount": 1}],
                },
                "cannot depend",
            ),
            (
                {
                    "kind": "repeat",
                    "amount": 2,
                    "operations": [
                        {
                            "kind": "damage_unit",
                            "target": "enemy_unit",
                            "amount": 1,
                            "requires_target": True,
                        }
                    ],
                },
                "cannot use requires_target",
            ),
        )
        for payload, message in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, message):
                    _parse_operation(payload, "test", 1)

    def test_repeat_count_is_frozen_once_and_runtime_limit_is_diagnostic(self):
        rulebook = RuleBook.from_directory("data/rules")
        repository = CardRepository("data/cards.sqlite3")
        engine = _fresh(rulebook, repository, seed=2101)
        source = _put_unit(engine, 0, _card(99001))
        _put_hand(engine, repository.get(PIXIE_ID))
        operation = EffectOperation(
            EffectKind.REPEAT,
            TargetKind.OWN_LEADER,
            amount_expr=ValueExpression(
                ExprType.CONTROLLER_HAND_COUNT,
                card_filter=HandFilter(card_type="随从", tribe_name="妖精"),
            ),
            repeat_operations=(
                EffectOperation(
                    EffectKind.ADD_CARD,
                    TargetKind.OWN_LEADER,
                    card_id=PIXIE_ID,
                ),
            ),
        )
        engine._start_effects(source.definition, source.entity_id, (operation,))
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [PIXIE_ID, PIXIE_ID],
        )

        overflow = replace(
            operation,
            amount_expr=ValueExpression.constant(MAX_REPEAT_COUNT + 1),
        )
        with self.assertRaises(ResolutionLoopError) as raised:
            engine._start_effects(source.definition, source.entity_id, (overflow,))
        self.assertEqual(
            raised.exception.diagnostics["repeat_count"],
            MAX_REPEAT_COUNT + 1,
        )
        self.assertEqual(
            raised.exception.diagnostics["repeat_limit"],
            MAX_REPEAT_COUNT,
        )


class RealDynamicRepeatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2201):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_real_rule_shapes_are_sequential_repeat_operations(self):
        fanfare = self.rulebook.operations_for(10114130, Trigger.FANFARE)[0]
        evolve = self.rulebook.operations_for(10114130, Trigger.EVOLVE)[0]
        arrow = self.rulebook.operations_for(10313310, Trigger.PLAY)[0]
        self.assertIs(fanfare.amount_expr.type, ExprType.CONTROLLER_HAND_COUNT)
        self.assertIs(evolve.kind, EffectKind.REPEAT)
        self.assertIs(evolve.repeat_operations[0].target, TargetKind.RANDOM_ENEMY_UNIT)
        self.assertIs(arrow.amount_expr.type, ExprType.CONTROLLER_COMBO)
        self.assertEqual(arrow.repeat_operations[0].secondary_amount, -1)

    def test_amataz_fanfare_counts_only_pixie_followers_and_has_ward(self):
        engine = self.fresh(seed=5)
        pixie = self.repository.get(PIXIE_ID)
        _put_hand(engine, pixie)
        _put_hand(engine, pixie)
        _put_hand(engine, _card(99101, attack=4, life=4))
        _put_hand(
            engine,
            replace(
                pixie,
                card_id=99102,
                card_type="法术",
                attack=None,
                life=None,
            ),
        )

        source = _play(engine, self.repository, 10114130)
        definition = self.repository.get(10114130)
        self.assertEqual(
            (source.attack, source.health, source.max_health),
            (definition.attack + 2, definition.life + 2, definition.life + 2),
        )
        self.assertTrue(source.has_keyword("守护"))

    def test_amataz_evolve_reselects_and_stabilizes_between_every_hit(self):
        engine = self.fresh(seed=7)
        for _ in range(3):
            _put_hand(engine, self.repository.get(PIXIE_ID))
        source = _play(engine, self.repository, 10114130)
        enemy = _put_unit(engine, 1, _card(99201, life=3))
        engine.players[0].turns_started = engine.config.evolution_unlock_turn

        engine.apply(Evolve(0, source.entity_id))

        self.assertNotIn(enemy, engine.players[1].board)
        hits = [
            event
            for event in engine.event_history
            if event.type is EventType.DAMAGE_APPLIED
            and event.target_id == enemy.entity_id
        ]
        self.assertEqual([event.amount for event in hits], [1, 1, 1])
        self.assertIsNone(engine.state.pending_choice)

    def test_empty_random_candidate_set_is_a_noop_without_rng_consumption(self):
        engine = self.fresh(seed=11)
        source = _put_unit(engine, 0, self.repository.get(10114130))
        _put_hand(engine, self.repository.get(PIXIE_ID))
        _put_hand(engine, self.repository.get(PIXIE_ID))
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        before_rng = engine.random.getstate()

        engine.apply(Evolve(0, source.entity_id))

        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertIsNone(engine.state.pending_choice)

    def test_banishment_arrow_uses_combo_including_itself_and_repeats_health_loss(self):
        one = self.fresh(seed=13)
        survivor = _put_unit(one, 1, _card(99301, life=2))
        _play(one, self.repository, 10313310)
        self.assertEqual((survivor.health, survivor.max_health), (1, 1))

        three = self.fresh(seed=17)
        victim = _put_unit(three, 1, _card(99302, life=3))
        three.players[0].cards_played_this_turn = 2
        _play(three, self.repository, 10313310)
        self.assertNotIn(victim, three.players[1].board)
        self.assertEqual(
            [card.definition.card_id for card in three.players[1].graveyard],
            [99302],
        )

    def test_seeded_repeat_distribution_and_fingerprint_replay_are_identical(self):
        fingerprints = []
        healths = []
        for _ in range(2):
            engine = self.fresh(seed=23)
            enemies = [
                _put_unit(engine, 1, _card(99400 + index, life=10))
                for index in range(2)
            ]
            engine.players[0].cards_played_this_turn = 4
            _play(engine, self.repository, 10313310)
            healths.append(tuple(enemy.health for enemy in enemies))
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(healths[0], healths[1])
        self.assertEqual(sum(10 - health for health in healths[0]), 5)
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_rl_evolve_mask_executes_repeat_without_exposing_random_choice(self):
        deck = [_card(99500 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=29,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=29)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        source = _put_unit(env.core, 0, self.repository.get(10114130))
        _put_hand(env.core, self.repository.get(PIXIE_ID))
        _put_hand(env.core, self.repository.get(PIXIE_ID))
        enemy = _put_unit(env.core, 1, _card(99601, life=4))
        env.players[0].turns_started = env.core.config.evolution_unlock_turn
        command = Evolve(0, source.entity_id)
        action = env._encode_command(command)
        self.assertTrue(env.action_mask()[action])

        env.step(action)

        self.assertEqual(enemy.health, 2)
        self.assertIsNone(env.core.state.pending_choice)


class DynamicRepeatDatabaseAuditTests(unittest.TestCase):
    def test_database_clauses_and_absent_modes_or_references_match(self):
        expected_phrases = {
            10114130: ("number of Pixie followers", "Do this X times", "Ward"),
            10313310: ("Do this X times", "X is your", "Combo"),
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id, phrases in expected_phrases.items():
                with self.subTest(card_id=card_id):
                    rows = connection.execute(
                        "SELECT text_eng FROM skill_texts WHERE card_id=? ORDER BY position",
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(len(rows), 1)
                    normalized = re.sub(r"<[^>]+>", "", rows[0][0])
                    for phrase in phrases:
                        self.assertIn(phrase, normalized)
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM card_references WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_cards_have_exact_mapped_clause_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_dynamic_repeat_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
