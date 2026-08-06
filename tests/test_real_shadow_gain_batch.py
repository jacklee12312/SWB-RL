# -*- coding: utf-8 -*-
"""Structured shadow gain and exact Nightmare real-card coverage."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import ActivateAmulet, Evolve, PlayCard
from swb.engine.effects import (
    EffectKind,
    EffectOperation,
    ExprType,
    TargetKind,
    ValueExpression,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10152210, 10153120)
SOURCE_HASHES = {
    10152210: "dd76afee6ce963358e89bbf3ef0973c1d6de1485ad17d424bc2b2b3dff743cf0",
    10153120: "9576894dcc7a53dd32d1d50d25409f87772c4ef1d20a6fb9068f4adf924147ee",
}
GHOST_ID = 90051130


class ShadowGainSchemaTests(unittest.TestCase):
    def test_add_shadows_accepts_leaders_and_dynamic_non_target_expression(self):
        own = _parse_operation(
            {"kind": "add_shadows", "target": "own_leader", "amount": 2},
            "test",
            1,
        )
        enemy = _parse_operation(
            {
                "kind": "add_shadows",
                "target": "enemy_leader",
                "amount": {"type": "controller_shadows"},
            },
            "test",
            1,
        )
        self.assertIs(own.kind, EffectKind.ADD_SHADOWS)
        self.assertEqual(own.amount, 2)
        self.assertIs(enemy.amount_expr.type, ExprType.CONTROLLER_SHADOWS)

    def test_add_shadows_and_summon_reject_ambiguous_schema(self):
        invalid = (
            (
                {"kind": "add_shadows", "target": "own_unit", "amount": 1},
                "requires own_leader or enemy_leader",
            ),
            (
                {"kind": "add_shadows", "target": "own_leader", "amount": -1},
                "non-negative",
            ),
            (
                {"kind": "add_shadows", "target": "own_leader", "amount": True},
                "non-negative",
            ),
            (
                {
                    "kind": "add_shadows",
                    "target": "own_leader",
                    "amount": {"type": "target_health"},
                },
                "cannot depend",
            ),
            (
                {
                    "kind": "summon",
                    "target": "own_board",
                    "card_id": GHOST_ID,
                },
                "summon requires",
            ),
        )
        for payload, message in invalid:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ValueError, message):
                    _parse_operation(payload, "test", 1)

    def test_add_shadows_executes_for_both_leaders_and_emits_auditable_events(self):
        rulebook = RuleBook.from_directory("data/rules")
        repository = CardRepository("data/cards.sqlite3")
        engine = _fresh(rulebook, repository, seed=2501)
        source = _put_unit(engine, 0, _card(99701))
        engine.players[0].shadows = 2
        operations = (
            EffectOperation(
                EffectKind.ADD_SHADOWS,
                TargetKind.OWN_LEADER,
                amount_expr=ValueExpression(ExprType.CONTROLLER_SHADOWS),
            ),
            EffectOperation(
                EffectKind.ADD_SHADOWS,
                TargetKind.ENEMY_LEADER,
                amount=3,
            ),
            EffectOperation(
                EffectKind.ADD_SHADOWS,
                TargetKind.OWN_LEADER,
                amount=0,
            ),
        )

        engine._start_effects(source.definition, source.entity_id, operations)

        self.assertEqual((engine.players[0].shadows, engine.players[1].shadows), (4, 3))
        events = [
            event
            for event in engine.event_history
            if event.type is EventType.SHADOWS_CHANGED
            and event.metadata.get("source_card_id") == source.definition.card_id
        ]
        self.assertEqual([event.amount for event in events], [2, 3])
        self.assertEqual([event.metadata["target_player"] for event in events], [0, 1])

        bad = EffectOperation(
            EffectKind.ADD_SHADOWS,
            TargetKind.OWN_LEADER,
            amount=-1,
        )
        with self.assertRaisesRegex(IllegalCommand, "non-negative"):
            engine._start_effects(source.definition, source.entity_id, (bad,))


class RealShadowGainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2601):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_real_rule_shapes_include_activation_necromancy_and_repeat(self):
        garden_play = self.rulebook.operations_for(10152210, Trigger.PLAY)
        garden_activate = self.rulebook.operations_for(10152210, Trigger.ACTIVATE)
        orthrus_evolve = self.rulebook.operations_for(10153120, Trigger.EVOLVE)[0]
        self.assertEqual(garden_play[0].kind, EffectKind.ADD_SHADOWS)
        self.assertEqual(
            [operation.kind for operation in garden_activate],
            [EffectKind.DESTROY, EffectKind.SUMMON, EffectKind.SUMMON],
        )
        self.assertEqual(self.rulebook.activation_for(10152210).cost, 0)
        self.assertEqual(orthrus_evolve.kind, EffectKind.NECROMANCY)
        repeat = orthrus_evolve.necromancy_operations[0]
        self.assertEqual((repeat.kind, repeat.amount), (EffectKind.REPEAT, 2))
        self.assertEqual(repeat.repeat_operations[0].amount, 2)

    def test_graveyard_garden_gains_two_then_activation_summons_two_ghosts(self):
        engine = self.fresh(seed=3)
        source = _play(engine, self.repository, 10152210)
        self.assertEqual(engine.players[0].shadows, 2)
        gain = next(
            event
            for event in engine.event_history
            if event.type is EventType.SHADOWS_CHANGED
            and event.metadata.get("source_card_id") == 10152210
        )
        self.assertEqual(
            (gain.amount, gain.metadata["shadows_before"], gain.metadata["shadows_after"]),
            (2, 0, 2),
        )

        command = ActivateAmulet(0, source.entity_id)
        self.assertIn(command, engine.legal_commands())
        engine.apply(command)

        ghosts = [
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == GHOST_ID
        ]
        self.assertEqual(len(ghosts), 2)
        self.assertNotIn(source, engine.players[0].board)
        self.assertEqual(engine.players[0].shadows, 3)
        self.assertTrue(all(unit.origin is CardOrigin.TOKEN for unit in ghosts))
        self.assertTrue(all(unit.has_keyword("疾驰") for unit in ghosts))

    def test_graveyard_garden_respects_board_capacity_after_source_destruction(self):
        engine = self.fresh(seed=5)
        for index in range(4):
            _put_unit(engine, 0, _card(99800 + index))
        source = _play(engine, self.repository, 10152210)
        self.assertEqual(len(engine.players[0].board), engine.config.max_board)

        engine.apply(ActivateAmulet(0, source.entity_id))

        ghosts = [
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == GHOST_ID
        ]
        self.assertEqual(len(ghosts), 1)
        self.assertEqual(len(engine.players[0].board), engine.config.max_board)

    def test_orthrus_fanfare_gains_two_and_intrinsic_ward(self):
        engine = self.fresh(seed=7)
        source = _play(engine, self.repository, 10153120)
        self.assertEqual(engine.players[0].shadows, 2)
        self.assertTrue(source.has_keyword("守护"))

    def test_orthrus_insufficient_necromancy_skips_damage_and_rng(self):
        engine = self.fresh(seed=11)
        source = _play(engine, self.repository, 10153120)
        enemy = _put_unit(engine, 1, _card(99901, life=4))
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        before_rng = engine.random.getstate()

        engine.apply(Evolve(0, source.entity_id))

        self.assertEqual(engine.players[0].shadows, 2)
        self.assertEqual(enemy.health, 4)
        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertFalse(any(
            event.type is EventType.NECROMANCY_ACTIVATED
            for event in engine.event_history
        ))

    def test_orthrus_pays_once_and_hits_the_only_enemy_twice(self):
        engine = self.fresh(seed=13)
        engine.players[0].shadows = 2
        source = _play(engine, self.repository, 10153120)
        enemy = _put_unit(engine, 1, _card(99902, life=4))
        engine.players[0].turns_started = engine.config.evolution_unlock_turn

        engine.apply(Evolve(0, source.entity_id))

        self.assertEqual(engine.players[0].shadows, 0)
        self.assertNotIn(enemy, engine.players[1].board)
        activations = [
            event
            for event in engine.event_history
            if event.type is EventType.NECROMANCY_ACTIVATED
            and event.metadata.get("source_card_id") == 10153120
        ]
        self.assertEqual([event.amount for event in activations], [4])
        hits = [
            event
            for event in engine.event_history
            if event.type is EventType.DAMAGE_APPLIED
            and event.target_id == enemy.entity_id
        ]
        self.assertEqual([event.amount for event in hits], [2, 2])

    def test_orthrus_empty_candidates_still_pay_but_do_not_consume_rng(self):
        engine = self.fresh(seed=17)
        engine.players[0].shadows = 2
        source = _play(engine, self.repository, 10153120)
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        before_rng = engine.random.getstate()

        engine.apply(Evolve(0, source.entity_id))

        self.assertEqual(engine.players[0].shadows, 0)
        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertEqual(
            sum(event.type is EventType.NECROMANCY_ACTIVATED for event in engine.event_history),
            1,
        )

    def test_seeded_orthrus_distribution_and_fingerprint_replay_match(self):
        healths = []
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=19)
            engine.players[0].shadows = 2
            source = _play(engine, self.repository, 10153120)
            enemies = [
                _put_unit(engine, 1, _card(99910 + index, life=10))
                for index in range(2)
            ]
            engine.players[0].turns_started = engine.config.evolution_unlock_turn
            engine.apply(Evolve(0, source.entity_id))
            healths.append(tuple(enemy.health for enemy in enemies))
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(healths[0], healths[1])
        self.assertEqual(sum(10 - health for health in healths[0]), 4)
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_rl_masks_execute_garden_activation_and_orthrus_evolution(self):
        deck = [_card(100100 + index) for index in range(40)]
        garden_env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=23,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        garden_env.reset(seed=23)
        garden_env.players[0].hand.clear()
        garden_env.players[0].hand_entity_ids.clear()
        garden_env.players[0].board.clear()
        garden_env.players[0].mana = garden_env.players[0].max_mana = 10
        _put_hand(garden_env.core, self.repository.get(10152210))
        play = PlayCard(0, 0)
        play_action = garden_env._encode_command(play)
        self.assertTrue(garden_env.action_mask()[play_action])
        garden_env.step(play_action)
        garden = next(
            card
            for card in garden_env.players[0].board
            if card.definition.card_id == 10152210
        )
        activate = ActivateAmulet(0, garden.entity_id)
        activate_action = garden_env._encode_command(activate)
        self.assertTrue(garden_env.action_mask()[activate_action])
        garden_env.step(activate_action)
        self.assertEqual(
            sum(unit.definition.card_id == GHOST_ID for unit in garden_env.players[0].board),
            2,
        )

        orthrus_env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=29,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        orthrus_env.reset(seed=29)
        for player in orthrus_env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        source = _put_unit(orthrus_env.core, 0, self.repository.get(10153120))
        enemy = _put_unit(orthrus_env.core, 1, _card(100201, life=4))
        orthrus_env.players[0].shadows = 4
        orthrus_env.players[0].turns_started = orthrus_env.core.config.evolution_unlock_turn
        evolve = Evolve(0, source.entity_id)
        evolve_action = orthrus_env._encode_command(evolve)
        self.assertTrue(orthrus_env.action_mask()[evolve_action])
        orthrus_env.step(evolve_action)
        self.assertNotIn(enemy, orthrus_env.players[1].board)
        self.assertEqual(orthrus_env.players[0].shadows, 0)


class ShadowGainDatabaseAuditTests(unittest.TestCase):
    def test_database_text_references_and_absent_modes_match_reviewed_sources(self):
        expected_phrases = {
            10152210: ("Gain 2 shadows", "Destroy this card", "Summon 2 copies"),
            10153120: ("Gain 2 shadows", "Necromancy", "Do this 2 times"),
        }
        expected_references = {
            10152210: [(GHOST_ID,)],
            10153120: [],
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
                            "SELECT referenced_card_id FROM card_references WHERE card_id=? ORDER BY position",
                            (card_id,),
                        ).fetchall(),
                        expected_references[card_id],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_cards_have_exact_mapped_clause_evidence_and_ghost_producer(self):
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
                    ["tests/test_real_shadow_gain_batch.py"],
                )

        token_report = _build_token_audit("data/cards.sqlite3", "data/rules")
        ghost = next(card for card in token_report["cards"] if card["card_id"] == GHOST_ID)
        self.assertIn(
            {
                "source_card_id": 10152210,
                "entry_kind": "summon",
                "rule_file": "real_shadow_gain_batch.json",
                "rule_group": "rules",
            },
            ghost["authored_producers"],
        )


if __name__ == "__main__":
    unittest.main()
