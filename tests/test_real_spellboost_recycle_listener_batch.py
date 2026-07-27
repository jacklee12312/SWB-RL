# -*- coding: utf-8 -*-
"""Exact Spellboost, deck-recycle, survived-damage, and token-summon cards."""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import (
    EffectKind,
    EffectOperation,
    ExprType,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10131320,
    10831310,
    10341110,
    10551310,
    10843310,
    10752310,
)
SOURCE_HASHES = {
    10131320: "54d450622e6601e336272504847b151912ec6ddf159997339bb2c9c585df57b3",
    10831310: "54d450622e6601e336272504847b151912ec6ddf159997339bb2c9c585df57b3",
    10341110: "f75cde2c3a37578ccfb445c92df434dd2fa1b6ab8db353c528d4e9ad4cc695d0",
    10551310: "59a81e35b233335a1d58921e05266120bce5f314e058df762bf9b80fc7de7b70",
    10843310: "7ebf32a75b4cf390a494d18dd886e4d87545677291d2962fcb619d44ecb2e771",
    10752310: "7879b70f795a6921a61d9067071541bd8a0fabbe9ef938210711469c5dec5079",
}


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        candidate
        for candidate in request.options
        if candidate.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _choose_enemy_leader(engine) -> None:
    request = engine.state.pending_choice
    option = next(
        candidate
        for candidate in request.options
        if candidate.leader_player_index == 1
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _play_spell(engine, repository: CardRepository, card_id: int):
    card = _put_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, engine.players[0].hand.index(card)))
    return card


def _damage_self(engine, source, amount: int) -> None:
    engine._start_effects(
        source.definition,
        source.entity_id,
        (
            EffectOperation(
                kind=EffectKind.DAMAGE_UNIT,
                target=TargetKind.SELF,
                amount=amount,
            ),
        ),
        controller=0,
        label="测试自伤",
    )


def _load_single_operation(operation: dict) -> RuleBook:
    payload = {
        "rules": [
            {
                "card_id": 990001,
                "trigger": "play",
                "operations": [operation],
            }
        ]
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rule.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return RuleBook.from_directory(directory)


class RealSpellboostRecycleListenerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 3101):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_new_schema_validation(self):
        for card_id in (10131320, 10831310):
            with self.subTest(card_id=card_id):
                operation = self.rulebook.operations_for(
                    card_id,
                    Trigger.PLAY,
                )[0]
                self.assertIs(operation.kind, EffectKind.DAMAGE_UNIT)
                self.assertIs(operation.amount_expr.type, ExprType.ADD)
                self.assertEqual(
                    [value.type for value in operation.amount_expr.values],
                    [ExprType.CONSTANT, ExprType.SOURCE_SPELLBOOST_COUNT],
                )
                self.assertTrue(operation.requires_target)

        recycle = self.rulebook.operations_for(10551310, Trigger.PLAY)[1]
        self.assertIs(recycle.kind, EffectKind.ADD_CARD_TO_DECK)
        self.assertEqual(recycle.card_id, 10551310)
        listener = self.rulebook.listeners_for(10341110)[0]
        self.assertEqual(listener.event, EventType.FOLLOWER_DAMAGED_SURVIVED)
        self.assertEqual(listener.operations[0].deck_filter.class_id, 4)
        self.assertEqual(listener.operations[0].deck_filter.card_type, "随从")

        with self.assertRaisesRegex(ValueError, "requires card_id"):
            _load_single_operation({
                "kind": "add_card_to_deck",
                "target": "own_leader",
            })
        with self.assertRaisesRegex(ValueError, "requires target 'own_leader'"):
            _load_single_operation({
                "kind": "add_card_to_deck",
                "target": "enemy_leader",
                "card_id": 990002,
            })
        with self.assertRaisesRegex(ValueError, "must not have 'values'"):
            _load_single_operation({
                "kind": "damage_unit",
                "target": "enemy_unit",
                "amount": {
                    "type": "source_spellboost_count",
                    "values": [{"type": "constant", "value": 1}],
                },
            })

    def test_both_storm_blasts_use_frozen_spellboost_count(self):
        for card_id in (10131320, 10831310):
            with self.subTest(card_id=card_id):
                engine = self.fresh(seed=7)
                target = _put_unit(engine, 1, _card(991001, life=10))
                spell = _put_hand(engine, self.repository.get(card_id))
                spell.apply_spellboost(3)

                engine.apply(PlayCard(0, 0))
                _choose_entity(engine, target.entity_id)

                self.assertEqual(target.health, 5)
                damage = [
                    event
                    for event in engine.event_history
                    if event.type is EventType.DAMAGE_APPLIED
                    and event.target_id == target.entity_id
                ]
                self.assertEqual(damage[-1].amount, 5)

    def test_target_required_spell_is_atomic_without_enemy_follower(self):
        engine = self.fresh(seed=11)
        _put_hand(engine, self.repository.get(10131320))
        before = (
            engine.deterministic_fingerprint(),
            engine.random.getstate(),
            tuple(engine.event_history),
            tuple(engine.logs),
            engine.state.pending_choice,
        )

        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

        self.assertEqual(
            (
                engine.deterministic_fingerprint(),
                engine.random.getstate(),
                tuple(engine.event_history),
                tuple(engine.logs),
                engine.state.pending_choice,
            ),
            before,
        )

    def test_affirmation_draws_only_on_own_turn_after_surviving_damage(self):
        engine = self.fresh(seed=13)
        neutral = _card(991010, class_id=0, class_name="中立")
        dragon_spell = _card(
            991011,
            class_id=4,
            class_name="龙族",
            card_type="法术",
            attack=None,
            life=None,
        )
        dragon_follower = _card(
            991012,
            class_id=4,
            class_name="龙族",
        )
        engine.players[0].deck = [neutral, dragon_spell, dragon_follower]
        source = _put_unit(engine, 0, self.repository.get(10341110))

        _damage_self(engine, source, 1)

        self.assertEqual(source.health, 1)
        self.assertEqual(
            [card.definition.card_id for card in engine.players[0].hand],
            [dragon_follower.card_id],
        )
        self.assertEqual(
            [card.card_id for card in engine.players[0].deck],
            [neutral.card_id, dragon_spell.card_id],
        )

        opponent_turn = self.fresh(seed=17)
        opponent_turn.players[0].deck = [dragon_follower]
        source = _put_unit(
            opponent_turn,
            0,
            self.repository.get(10341110),
        )
        opponent_turn.apply(EndTurn(0))
        _damage_self(opponent_turn, source, 1)
        self.assertFalse(opponent_turn.players[0].hand)

        lethal = self.fresh(seed=19)
        lethal.players[0].deck = [dragon_follower]
        source = _put_unit(lethal, 0, self.repository.get(10341110))
        _damage_self(lethal, source, 2)
        self.assertNotIn(source, lethal.players[0].board)
        self.assertFalse(lethal.players[0].hand)

    def test_nightblossom_recycles_after_damage_with_seeded_order(self):
        outcomes = []
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=23)
            engine.players[0].deck = [_card(991020 + index) for index in range(8)]
            target = _put_unit(engine, 1, _card(991030, life=8))
            _play_spell(engine, self.repository, 10551310)
            _choose_entity(engine, target.entity_id)

            outcomes.append(tuple(card.card_id for card in engine.players[0].deck))
            fingerprints.append(engine.deterministic_fingerprint())
            self.assertEqual(target.health, 3)
            added = [
                event
                for event in engine.event_history
                if event.type is EventType.CARD_ADDED_TO_DECK
            ]
            self.assertEqual(added[-1].metadata["card_id"], 10551310)
            self.assertNotIn("position", added[-1].metadata)
            self.assertEqual(outcomes[-1].count(10551310), 1)

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_foxfire_recycles_then_draws_only_in_overflow(self):
        normal = self.fresh(seed=29)
        normal.players[0].max_mana = normal.players[0].mana = 6
        normal.players[0].deck = [_card(991040), _card(991041)]
        normal_event_count = len(normal.event_history)
        _play_spell(normal, self.repository, 10843310)
        _choose_enemy_leader(normal)
        self.assertEqual(normal.players[1].health, 19)
        self.assertFalse(normal.players[0].hand)
        self.assertEqual(len(normal.players[0].deck), 3)
        self.assertFalse(any(
            event.type is EventType.CARD_DRAWN
            for event in normal.event_history[normal_event_count:]
        ))

        overflow = self.fresh(seed=31)
        overflow.players[0].max_mana = overflow.players[0].mana = 7
        overflow.players[0].deck = [_card(991042), _card(991043)]
        overflow_event_count = len(overflow.event_history)
        _play_spell(overflow, self.repository, 10843310)
        _choose_enemy_leader(overflow)
        self.assertEqual(len(overflow.players[0].hand), 1)
        self.assertEqual(len(overflow.players[0].deck), 2)
        ordered_types = [
            event.type
            for event in overflow.event_history[overflow_event_count:]
        ]
        self.assertLess(
            ordered_types.index(EventType.CARD_ADDED_TO_DECK),
            ordered_types.index(EventType.CARD_DRAWN),
        )

    def test_youth_summons_and_evolves_each_successful_token_in_order(self):
        engine = self.fresh(seed=37)
        before_ep = engine.players[0].evolution_points

        _play_spell(engine, self.repository, 10752310)

        self.assertEqual(
            [unit.definition.card_id for unit in engine.players[0].board],
            [90051130, 90051120, 90051110],
        )
        self.assertTrue(all(unit.evolved for unit in engine.players[0].board))
        self.assertEqual(engine.players[0].evolution_points, before_ep)
        for unit in engine.players[0].board:
            self.assertEqual(unit.attack, (unit.definition.attack or 0) + 2)
            self.assertEqual(unit.max_health, (unit.definition.life or 0) + 2)

        constrained = self.fresh(seed=41)
        for index in range(4):
            _put_unit(constrained, 0, _card(991050 + index))
        _play_spell(constrained, self.repository, 10752310)
        generated = [
            unit
            for unit in constrained.players[0].board
            if unit.definition.card_id in {90051110, 90051120, 90051130}
        ]
        self.assertEqual(
            [(unit.definition.card_id, unit.evolved) for unit in generated],
            [(90051130, True)],
        )

    def test_rl_mask_matches_required_target_without_action_migration(self):
        deck = [
            _card(
                992000 + index,
                class_id=0,
                class_name="中立",
            )
            for index in range(40)
        ]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=3,
            class_b=3,
            seed=43,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        observation, _ = env.reset(seed=43)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        _put_hand(env.core, self.repository.get(10131320))
        play_action = env._encode_command(PlayCard(0, 0))
        self.assertEqual(env.ACTION_SIZE, 112)
        self.assertEqual(
            len(observation),
            ShadowverseEnv.OBSERVATION_V1_SIZE,
        )
        self.assertFalse(env.action_mask()[play_action])

        target = _put_unit(env.core, 1, _card(992100))
        self.assertTrue(env.action_mask()[play_action])
        env.step(play_action)
        choice_actions = [
            action
            for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        ]
        self.assertEqual(len(choice_actions), 1)
        env.step(choice_actions[0])
        self.assertEqual(target.health, 3)


class SpellboostRecycleListenerDatabaseAuditTests(unittest.TestCase):
    def test_database_text_and_references_match_reviewed_official_cards(self):
        expected_phrases = {
            10131320: ("X starts at 2", "On Spellboost", "deal it X damage"),
            10831310: ("X starts at 2", "On Spellboost", "deal it X damage"),
            10341110: ("takes damage but isn't destroyed", "Dragoncraft follower"),
            10551310: ("deal it 5 damage", "to your deck"),
            10843310: ("enemy leader", "to your deck", "Overflow"),
            10752310: ("Ghost", "Bat", "Skeleton", "evolve them"),
        }
        expected_references = {
            10131320: (),
            10831310: (),
            10341110: (),
            10551310: (10551310,),
            10843310: (10843310,),
            10752310: (90051130, 90051120, 90051110),
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id, phrases in expected_phrases.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        "SELECT text_eng FROM skill_texts WHERE card_id=?",
                        (card_id,),
                    ).fetchone()
                    self.assertIsNotNone(row)
                    normalized = re.sub(r"<[^>]+>", "", row[0])
                    for phrase in phrases:
                        self.assertIn(phrase, normalized)
                    references = tuple(
                        reference[0]
                        for reference in connection.execute(
                            "SELECT referenced_card_id FROM card_references "
                            "WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    )
                    self.assertEqual(references, expected_references[card_id])
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_all_six_cards_have_exact_clause_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(
                    info["clause_audit"]["status"],
                    "mapped_exact",
                )
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_spellboost_recycle_listener_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
