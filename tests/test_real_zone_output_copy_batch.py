# -*- coding: utf-8 -*-
"""Exact draw-output, hidden-copy, and post-banish copy rules."""

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
from swb.engine.commands import Choose, PlayCard
from swb.engine.effects import EffectKind, ExprType, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import CostModifier, DeckCard, HandCard
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10853310, 10541310, 10802310, 10443310, 10652310)
SOURCE_HASHES = {
    10853310: "4eb10c63bc26c7f973c195c1e0eb5717dfb01fc3510a91bde4d51f6e30a90c07",
    10541310: "0e46884a910dc2ef571c49a2e5d775648277df1089be6f835971f5d9bb6a2974",
    10802310: "707e7f47b0dbf75922e9bbe6c717c605110a63b6e0389998f4cceedbdbff1253",
    10443310: "bfd077a398120be80d2eb2c7ecc8bd67b03a0a8f568df64a67ca972dbb887220",
    10652310: "d0752b2a9da851ffef2331d6fd76f5925f7b493083e0b2cf668a6c3947426fdf",
}


def _put_hand_for(engine, owner: int, definition) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[owner].hand.append(card)
    engine.players[owner].hand_entity_ids.append(card.entity_id)
    return card


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        candidate
        for candidate in request.options
        if candidate.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _play_spell(engine, repository: CardRepository, card_id: int) -> HandCard:
    source = _put_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
    return source


def _load_operations(operations: list[dict]) -> RuleBook:
    payload = {
        "rules": [{
            "card_id": 990001,
            "trigger": "play",
            "operations": operations,
        }]
    }
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "rule.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return RuleBook.from_directory(directory)


class RealZoneOutputCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 5301):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_binding_schema_validation(self):
        flow = self.rulebook.operations_for(10853310, Trigger.PLAY)
        self.assertEqual(
            [operation.kind for operation in flow],
            [EffectKind.RETURN_TO_DECK, EffectKind.DRAW, EffectKind.CHANGE_COST],
        )
        self.assertTrue(flow[0].requires_target)
        self.assertEqual(flow[1].target_key, "drawn_cards")
        self.assertIs(flow[2].target, TargetKind.PREVIOUS_TARGET)

        verdict = self.rulebook.operations_for(10541310, Trigger.PLAY)
        self.assertEqual(verdict[0].target_key, "drawn_follower")
        self.assertIs(verdict[1].amount_expr.type, ExprType.BOUND_CARD_COST)
        self.assertEqual(verdict[1].amount_expr.binding_key, "drawn_follower")

        heroism = self.rulebook.operations_for(10802310, Trigger.PLAY)[0]
        self.assertIs(heroism.target, TargetKind.RANDOM_ENEMY_HAND)
        temptation = self.rulebook.operations_for(10652310, Trigger.PLAY)[1]
        self.assertIs(temptation.kind, EffectKind.SUMMON_COPY)

        with self.assertRaisesRegex(ValueError, "requires a non-empty binding_key"):
            _load_operations([{
                "kind": "damage_unit",
                "target": "random_enemy_unit",
                "amount": {"type": "bound_card_cost"},
            }])
        with self.assertRaisesRegex(ValueError, "was not defined"):
            _load_operations([{
                "kind": "damage_unit",
                "target": "random_enemy_unit",
                "amount": {
                    "type": "bound_card_cost",
                    "binding_key": "missing",
                },
            }])
        with self.assertRaisesRegex(ValueError, "exactly one card"):
            _load_operations([
                {
                    "kind": "draw",
                    "target": "own_leader",
                    "amount": 2,
                    "target_key": "many",
                },
                {
                    "kind": "damage_unit",
                    "target": "random_enemy_unit",
                    "amount": {
                        "type": "bound_card_cost",
                        "binding_key": "many",
                    },
                },
            ])
        with self.assertRaisesRegex(ValueError, "requires a follower"):
            _load_operations([{
                "kind": "summon_copy",
                "target": "enemy_board",
            }])

    def test_changed_flow_reduces_only_drawn_cards_after_unlock(self):
        results = []
        for unlocked in (False, True):
            engine = self.fresh(seed=17)
            if unlocked:
                engine.players[0].turns_started = (
                    engine.config.first_player_super_evolution_unlock_turn
                )
            engine.players[0].deck = [
                _card(991001, cost=4),
                _card(991002, cost=6),
                _card(991003, cost=8),
            ]
            returned = _put_hand(engine, _card(991004, cost=5))
            _play_spell(engine, self.repository, 10853310)
            _choose_entity(engine, returned.entity_id)

            self.assertEqual(len(engine.players[0].hand), 2)
            costs = tuple(
                (card.definition.cost, card.current_cost)
                for card in engine.players[0].hand
            )
            results.append(costs)
            for printed, current in costs:
                self.assertEqual(current, max(0, printed - int(unlocked)))

        self.assertNotEqual(results[0], results[1])

    def test_changed_flow_requires_another_hand_card_atomically(self):
        engine = self.fresh(seed=19)
        _put_hand(engine, self.repository.get(10853310))
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

    def test_verdict_uses_drawn_physical_cost_and_seeded_enemy_choice(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=23)
            follower = _card(991010, cost=7)
            modified = DeckCard(
                follower,
                [CostModifier(
                    modifier_id=engine._allocate_modifier_id(),
                    mode="subtract",
                    amount=2,
                    duration="permanent",
                )],
            )
            engine.players[0].deck = [
                _card(
                    991011,
                    card_type="法术",
                    attack=None,
                    life=None,
                ),
                modified,
            ]
            enemies = [
                _put_unit(engine, 1, _card(991012 + index, life=9))
                for index in range(2)
            ]

            _play_spell(engine, self.repository, 10541310)

            damaged = [enemy for enemy in enemies if enemy.health == 4]
            self.assertEqual(len(damaged), 1)
            drawn = engine.players[0].hand[0]
            self.assertEqual((drawn.card_id, drawn.current_cost), (991010, 5))
            outcomes.append(
                (damaged[0].definition.card_id, engine.deterministic_fingerprint())
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_verdict_no_match_skips_damage_and_rng_but_overdraw_keeps_cost(self):
        no_match = self.fresh(seed=29)
        no_match.players[0].deck = [_card(
            991020,
            card_type="法术",
            attack=None,
            life=None,
        )]
        enemy = _put_unit(no_match, 1, _card(991021, life=9))
        rng_before = no_match.random.getstate()
        _play_spell(no_match, self.repository, 10541310)
        self.assertEqual(enemy.health, 9)
        self.assertEqual(no_match.random.getstate(), rng_before)

        overdraw = self.fresh(seed=31)
        for index in range(overdraw.config.max_hand):
            _put_hand(overdraw, _card(991030 + index))
        overdraw.players[0].deck = [_card(991040, cost=6)]
        target = _put_unit(overdraw, 1, _card(991041, life=9))
        overdraw._start_effects(
            self.repository.get(10541310),
            None,
            self.rulebook.operations_for(10541310, Trigger.PLAY),
            controller=0,
            label="测试过抽",
        )
        self.assertEqual(target.health, 3)
        self.assertEqual(len(overdraw.players[0].hand), overdraw.config.max_hand)
        self.assertTrue(any(
            card.definition.card_id == 991040
            and card.entry_cause == "overdraw"
            for card in overdraw.players[0].graveyard
        ))

    def test_heroism_copies_seeded_hidden_enemy_card_then_draws(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=37)
            hidden = [
                _put_hand_for(engine, 1, _card(991050, cost=4)),
                _put_hand_for(engine, 1, _card(991051, cost=7)),
            ]
            engine.players[0].deck = [_card(991052, cost=2)]
            event_start = len(engine.event_history)

            _play_spell(engine, self.repository, 10802310)

            copied = next(
                card
                for card in engine.players[0].hand
                if card.card_id in {item.card_id for item in hidden}
            )
            self.assertEqual(copied.current_cost, copied.definition.cost - 1)
            self.assertIn(991052, [card.card_id for card in engine.players[0].hand])
            ordered = [
                event.type for event in engine.event_history[event_start:]
            ]
            self.assertLess(
                ordered.index(EventType.CARD_ADDED_TO_HAND),
                ordered.index(EventType.CARD_DRAWN),
            )
            copy_event = next(
                event
                for event in engine.event_history[event_start:]
                if event.type is EventType.CARD_ADDED_TO_HAND
            )
            self.assertFalse(copy_event.metadata["revealed"])
            outcomes.append((copied.card_id, engine.deterministic_fingerprint()))

        self.assertEqual(outcomes[0], outcomes[1])

        empty = self.fresh(seed=41)
        empty.players[0].deck = [_card(991053)]
        rng_before = empty.random.getstate()
        _play_spell(empty, self.repository, 10802310)
        self.assertEqual([card.card_id for card in empty.players[0].hand], [991053])
        self.assertEqual(empty.random.getstate(), rng_before)

    def test_heroism_copy_identity_is_hidden_from_opponent_observation(self):
        deck = [
            _card(993000 + index, class_id=0, class_name="中立")
            for index in range(40)
        ]
        vocabulary = tuple(
            [card.card_id for card in deck]
            + [10802310, 993100, 993101, 993102]
        )

        def resolved(seed: int):
            env = ShadowverseEnv(
                deck,
                deck,
                class_a=1,
                class_b=1,
                seed=71,
                rulebook=self.rulebook,
                card_resolver=self.repository.get,
                observation_version="v2",
                card_vocabulary=vocabulary,
                validate_invariants=True,
            )
            env.reset(seed=71)
            for player in env.players:
                player.hand.clear()
                player.hand_entity_ids.clear()
                player.board.clear()
            env.players[0].mana = env.players[0].max_mana = 10
            env.players[0].deck = [_card(993102, class_id=0, class_name="中立")]
            _put_hand(env.core, self.repository.get(10802310))
            _put_hand_for(env.core, 1, _card(993100, cost=4))
            _put_hand_for(env.core, 1, _card(993101, cost=8))
            env.core.random.seed(seed)
            env.core.apply(PlayCard(0, 0))
            copied_id = next(
                card.card_id
                for card in env.players[0].hand
                if card.card_id in {993100, 993101}
            )
            env.core.state.active_player = 1
            return copied_id, env.observation()

        first_id, first_observation = resolved(0)
        second_id, second_observation = resolved(1)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first_observation, second_observation)

    def test_absorption_banishes_before_copying_from_snapshot(self):
        engine = self.fresh(seed=43)
        target = _put_unit(engine, 1, _card(991060, cost=6))
        event_start = len(engine.event_history)
        _play_spell(engine, self.repository, 10443310)
        _choose_entity(engine, target.entity_id)

        self.assertNotIn(target, engine.players[1].board)
        copied = engine.players[0].hand[0]
        self.assertEqual(copied.card_id, target.definition.card_id)
        self.assertIs(copied.origin, CardOrigin.GENERATED)
        ordered = [event.type for event in engine.event_history[event_start:]]
        self.assertLess(
            ordered.index(EventType.CARD_BANISHED),
            ordered.index(EventType.CARD_ADDED_TO_HAND),
        )

    def test_stale_absorption_target_does_not_copy_snapshot(self):
        engine = self.fresh(seed=47)
        target = _put_unit(engine, 1, _card(991070))
        _play_spell(engine, self.repository, 10443310)
        engine.players[1].board.remove(target)

        _choose_entity(engine, target.entity_id)

        self.assertFalse(engine.players[0].hand)
        self.assertIsNone(engine.state.pending_choice)

    def test_temptation_summons_unevolved_base_copy_after_banish(self):
        engine = self.fresh(seed=53)
        definition = _card(991080, attack=3, life=7)
        target = _put_unit(engine, 1, definition)
        target.evolved = True
        target.attack = 9
        target.health = 4
        event_start = len(engine.event_history)

        _play_spell(engine, self.repository, 10652310)
        _choose_entity(engine, target.entity_id)

        copy = engine.players[0].board[0]
        self.assertEqual(copy.definition.card_id, definition.card_id)
        self.assertEqual((copy.attack, copy.health), (3, 7))
        self.assertFalse(copy.evolved)
        self.assertIs(copy.origin, CardOrigin.GENERATED)
        ordered = [event.type for event in engine.event_history[event_start:]]
        self.assertLess(
            ordered.index(EventType.CARD_BANISHED),
            ordered.index(EventType.FOLLOWER_SUMMONED),
        )
        summon_event = next(
            event
            for event in engine.event_history[event_start:]
            if event.type is EventType.FOLLOWER_SUMMONED
        )
        self.assertEqual(summon_event.metadata["via"], "copy_summon")

        full = self.fresh(seed=59)
        for index in range(full.config.max_board):
            _put_unit(full, 0, _card(991090 + index))
        enemy = _put_unit(full, 1, _card(991100))
        _play_spell(full, self.repository, 10652310)
        _choose_entity(full, enemy.entity_id)
        self.assertNotIn(enemy, full.players[1].board)
        self.assertEqual(len(full.players[0].board), full.config.max_board)

    def test_required_board_targets_are_atomic_and_rl_layout_is_stable(self):
        for card_id in (10443310, 10652310):
            with self.subTest(card_id=card_id):
                engine = self.fresh(seed=61)
                _put_hand(engine, self.repository.get(card_id))
                before = (
                    engine.deterministic_fingerprint(),
                    engine.random.getstate(),
                    tuple(engine.event_history),
                    tuple(engine.logs),
                )
                with self.assertRaises(IllegalCommand):
                    engine.apply(PlayCard(0, 0))
                self.assertEqual(
                    (
                        engine.deterministic_fingerprint(),
                        engine.random.getstate(),
                        tuple(engine.event_history),
                        tuple(engine.logs),
                    ),
                    before,
                )

        deck = [_card(992000 + index, class_id=0, class_name="中立") for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=67,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        observation, _ = env.reset(seed=67)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        source = _put_hand(env.core, self.repository.get(10443310))
        play_action = env._encode_command(PlayCard(0, 0))
        self.assertEqual((env.ACTION_SIZE, len(observation)), (111, 294))
        self.assertFalse(env.action_mask()[play_action])
        target = _put_unit(env.core, 1, _card(992100))
        self.assertTrue(env.action_mask()[play_action])
        env.step(play_action)
        self.assertEqual(
            sum(env.action_mask()[env.CHOICE_OFFSET:env.GRAVEYARD_CHOICE_OFFSET]),
            1,
        )
        env.step(next(
            action
            for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        ))
        self.assertNotIn(target, env.players[1].board)
        self.assertNotIn(source, env.players[0].hand)


class ZoneOutputCopyDatabaseAuditTests(unittest.TestCase):
    def test_database_text_matches_reviewed_official_cards(self):
        expected_phrases = {
            10853310: ("return it to deck", "Draw 2 cards", "super-evolution"),
            10541310: ("Draw a follower", "cost of the card you drew"),
            10802310: ("opponent's hand", "without revealing it", "cost by 1"),
            10443310: ("enemy card on the field", "banish it", "copy of it"),
            10652310: ("enemy follower", "banish it", "exact copy"),
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

    def test_all_five_cards_have_exact_clause_evidence(self):
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
                    ["tests/test_real_zone_output_copy_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
