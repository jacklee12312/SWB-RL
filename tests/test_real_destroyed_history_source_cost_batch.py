# -*- coding: utf-8 -*-
"""Exact destroyed-follower history copies and frozen source-cost rules."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, PlayCard
from swb.engine.effects import (
    ConditionType,
    CostChangeMode,
    EffectKind,
    EffectOperation,
    ExprType,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import CostModifier, DeathCause, HandCard
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10572310, 10871130, 10641310, 10643310, 10331110)
SOURCE_HASHES = {
    10572310: "90f38f64b6590b99d9448c98b3de1cdebbf3731bc48188fe78fddb50f8f54320",
    10871130: "a8589b276cd37c67a932fa83e59bce8e52ac1e881c3cf52be3d6ba604f696f33",
    10641310: "3cf5300d8a98aefc904c30935931fefbfef036ce7cf75ae47ec45ca57bfa4b5a",
    10643310: "b5b16fe0a41a9fc190225cf3eb339a8e6dc5fc730626f1f045e90b51d21b8955",
    10331110: "21be4108c58e1de2eb26ac4a94e07bbba30c2ff47c83a41b1b2065369162f894",
}


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        candidate
        for candidate in request.options
        if candidate.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _put_hand_for(engine, owner: int, definition) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[owner].hand.append(card)
    engine.players[owner].hand_entity_ids.append(card.entity_id)
    return card


def _set_cost(card: HandCard, value: int) -> None:
    card.cost_modifiers.append(CostModifier(990000 + value, "set", value, "permanent"))


def _discard(engine, hand_card: HandCard) -> None:
    source = _card(
        999991,
        name="测试弃牌源",
        card_type="法术",
        attack=None,
        life=None,
    )
    engine._start_effects(
        source,
        None,
        (EffectOperation(EffectKind.DISCARD, TargetKind.OWN_HAND),),
        controller=0,
        label="测试弃牌",
    )
    _choose_entity(engine, hand_card.entity_id)


def _destroy_all_own_followers(engine) -> None:
    source = _card(
        999992,
        name="测试破坏源",
        card_type="法术",
        attack=None,
        life=None,
    )
    engine._start_effects(
        source,
        None,
        (EffectOperation(EffectKind.DESTROY, TargetKind.ALL_OWN_UNITS),),
        controller=0,
        label="测试破坏",
    )
    engine._stabilize()


class DestroyedHistorySourceCostSchemaTests(unittest.TestCase):
    def test_history_copy_schema_and_source_cost_shapes(self):
        operation = _parse_operation(
            {
                "kind": "copy_destroyed_followers_to_hand",
                "target": "own_leader",
                "amount": 2,
                "distinct_card_names": True,
                "history_filter": {"card_type": "随从", "tribe_name": "创造物"},
            },
            "test",
            123,
        )
        self.assertIs(operation.kind, EffectKind.COPY_DESTROYED_FOLLOWERS_TO_HAND)
        self.assertTrue(operation.distinct_card_names)
        self.assertEqual(operation.history_filter.tribe_name, "创造物")

        damage = _parse_operation(
            {
                "kind": "damage_unit",
                "target": "all_enemy_units",
                "amount": {"type": "source_cost"},
            },
            "test",
            123,
        )
        self.assertIs(damage.amount_expr.type, ExprType.SOURCE_COST)

        generated = _parse_operation(
            {
                "kind": "add_card",
                "target": "own_leader",
                "card_id": 123,
                "amount": 2,
                "mode": "set",
                "conditions": [{"type": "source_cost_equals", "value": 4}],
            },
            "test",
            123,
        )
        self.assertIs(generated.mode, CostChangeMode.SET)
        self.assertIs(generated.conditions[0].type, ConditionType.SOURCE_COST_EQUALS)

    def test_history_and_generated_cost_schema_rejects_invalid_payloads(self):
        invalid = (
            {
                "kind": "copy_destroyed_followers_to_hand",
                "target": "enemy_leader",
                "amount": 1,
            },
            {
                "kind": "copy_destroyed_followers_to_hand",
                "target": "own_leader",
                "amount": 0,
            },
            {
                "kind": "copy_destroyed_followers_to_hand",
                "target": "own_leader",
                "amount": 1,
                "distinct_card_names": "yes",
            },
            {
                "kind": "copy_destroyed_followers_to_hand",
                "target": "own_leader",
                "amount": 1,
                "history_filter": {"card_type": "法术"},
            },
            {
                "kind": "add_card",
                "target": "own_leader",
                "card_id": 123,
                "amount": 2,
            },
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test", 123)


class RealDestroyedHistorySourceCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 5701):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_share_history_and_source_cost_primitives(self):
        tuning = self.rulebook.operations_for(10572310, Trigger.PLAY)
        self.assertEqual(
            [operation.kind for operation in tuning],
            [EffectKind.DISCARD, EffectKind.COPY_DESTROYED_FOLLOWERS_TO_HAND],
        )
        self.assertTrue(tuning[1].distinct_card_names)

        gilke = self.rulebook.operations_for(10871130, Trigger.FANFARE)[0]
        self.assertEqual(gilke.history_filter.tribe_name, "创造物")

        beheading = self.rulebook.operations_for(10643310, Trigger.PLAY)[0]
        self.assertIs(beheading.amount_expr.type, ExprType.SOURCE_COST)

    def test_revive_tuning_uses_two_distinct_destroyed_names_deterministically(self):
        def resolved(seed: int):
            engine = self.fresh(seed=seed)
            first = _card(995100, name="重复名称")
            duplicate_name = _card(995101, name="重复名称")
            second = _card(995102, name="第二名称")
            third = _card(995103, name="第三名称")
            for definition in (first, duplicate_name, second, third):
                _put_unit(engine, 0, definition)
            _destroy_all_own_followers(engine)
            victim = _put_hand(engine, _card(995110, name="弃置对象"))
            source = _put_hand(engine, self.repository.get(10572310))
            engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
            _choose_entity(engine, victim.entity_id)
            copied = [
                card
                for card in engine.players[0].hand
                if card.card_id in {995100, 995101, 995102, 995103}
            ]
            return engine, copied

        first_engine, first_copies = resolved(17)
        second_engine, second_copies = resolved(17)
        self.assertEqual(len(first_copies), 2)
        self.assertEqual(len({card.name for card in first_copies}), 2)
        self.assertEqual(
            [card.card_id for card in first_copies],
            [card.card_id for card in second_copies],
        )
        self.assertEqual(
            first_engine.deterministic_fingerprint(),
            second_engine.deterministic_fingerprint(),
        )
        hidden_events = [
            event
            for event in first_engine.event_history
            if event.type is EventType.CARD_ADDED_TO_HAND
            and event.metadata.get("copied_from_death_sequence") is not None
        ]
        self.assertEqual(len(hidden_events), 2)
        self.assertTrue(all(event.metadata["revealed"] is False for event in hidden_events))
        self.assertTrue(all(card.origin is CardOrigin.GENERATED for card in first_copies))

    def test_revive_tuning_without_discard_candidate_still_copies_and_empty_history_uses_no_rng(self):
        engine = self.fresh(seed=19)
        engine._record_destroyed_follower(0, _card(995120), DeathCause.EFFECT_DESTROY)
        engine._record_destroyed_follower(0, _card(995121), DeathCause.COMBAT)
        source = _put_hand(engine, self.repository.get(10572310))
        engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
        self.assertEqual(
            {card.card_id for card in engine.players[0].hand},
            {995120, 995121},
        )

        empty = self.fresh(seed=23)
        victim = _put_hand(empty, _card(995122))
        source = _put_hand(empty, self.repository.get(10572310))
        rng_before = empty.random.getstate()
        empty.apply(PlayCard(0, empty.players[0].hand.index(source)))
        _choose_entity(empty, victim.entity_id)
        self.assertEqual(empty.random.getstate(), rng_before)
        self.assertFalse(any(card.card_id == 995122 for card in empty.players[0].hand))

    def test_revive_tuning_full_hand_burns_only_overflow_copy(self):
        engine = self.fresh(seed=29)
        engine._record_destroyed_follower(0, _card(995130), DeathCause.COMBAT)
        engine._record_destroyed_follower(0, _card(995131), DeathCause.COMBAT)
        victim = _put_hand(engine, _card(995132))
        source = _put_hand(engine, self.repository.get(10572310))
        for index in range(engine.config.max_hand - 2):
            _put_hand_for(engine, 0, _card(995140 + index))
        engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
        _put_hand_for(engine, 0, _card(995199))
        _choose_entity(engine, victim.entity_id)
        copied_in_hand = sum(
            card.card_id in {995130, 995131}
            for card in engine.players[0].hand
        )
        copied_burned = sum(
            card.definition.card_id in {995130, 995131}
            and card.entry_cause == "hand_full"
            for card in engine.players[0].graveyard
        )
        self.assertEqual(
            (copied_in_hand, copied_burned, len(engine.players[0].hand)),
            (1, 1, engine.config.max_hand),
        )

    def test_gilke_filters_destroyed_history_to_artifact_followers(self):
        engine = self.fresh(seed=31)
        artifact = replace(
            _card(995200, name="测试创造物"),
            tribe_id=14,
            tribe_name="创造物",
        )
        ordinary = _card(995201, name="普通随从")
        _put_unit(engine, 0, artifact)
        _put_unit(engine, 0, ordinary)
        _destroy_all_own_followers(engine)
        source = _put_hand(engine, self.repository.get(10871130))
        engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
        self.assertTrue(any(card.card_id == artifact.card_id for card in engine.players[0].hand))
        self.assertFalse(any(card.card_id == ordinary.card_id for card in engine.players[0].hand))

    def test_gilke_random_result_is_hidden_from_opponent_observation(self):
        deck = [_card(995300 + index, class_id=0, class_name="中立") for index in range(40)]
        first_artifact = replace(_card(995350), tribe_id=14, tribe_name="创造物")
        second_artifact = replace(_card(995351), tribe_id=14, tribe_name="创造物")
        vocabulary = tuple([card.card_id for card in deck] + [10871130, 995350, 995351])

        def resolved(seed: int):
            env = ShadowverseEnv(
                deck,
                deck,
                class_a=7,
                class_b=7,
                seed=37,
                rulebook=self.rulebook,
                card_resolver=self.repository.get,
                observation_version="v2",
                card_vocabulary=vocabulary,
                validate_invariants=True,
            )
            env.reset(seed=37)
            for player in env.players:
                player.hand.clear()
                player.hand_entity_ids.clear()
                player.board.clear()
            env.players[0].mana = env.players[0].max_mana = 10
            env.core._record_destroyed_follower(0, first_artifact, DeathCause.COMBAT)
            env.core._record_destroyed_follower(0, second_artifact, DeathCause.COMBAT)
            _put_hand(env.core, self.repository.get(10871130))
            env.core.random.seed(seed)
            env.core.apply(PlayCard(0, 0))
            copied_id = next(
                card.card_id
                for card in env.players[0].hand
                if card.card_id in {995350, 995351}
            )
            env.core.state.active_player = 1
            return copied_id, env.observation()

        first_id, first_observation = resolved(0)
        second_id, second_observation = resolved(1)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first_observation, second_observation)

    def test_advent_discard_replacement_freezes_cost_and_play_requires_target(self):
        engine = self.fresh(seed=41)
        original = _put_hand(engine, self.repository.get(10641310))
        _discard(engine, original)
        replacement = next(card for card in engine.players[0].hand if card.card_id == 10641310)
        self.assertEqual(replacement.current_cost, 2)
        _discard(engine, replacement)
        self.assertFalse(any(card.card_id == 10641310 for card in engine.players[0].hand))

        played = self.fresh(seed=43)
        target = _put_unit(played, 0, _card(995400, attack=2, life=3))
        source = _put_hand(played, self.repository.get(10641310))
        played.apply(PlayCard(0, played.players[0].hand.index(source)))
        _choose_entity(played, target.entity_id)
        self.assertEqual((target.attack, target.health, target.max_health), (4, 5, 5))

        illegal = self.fresh(seed=47)
        _put_hand(illegal, self.repository.get(10641310))
        before = (
            illegal.deterministic_fingerprint(),
            illegal.random.getstate(),
            tuple(illegal.event_history),
            tuple(illegal.logs),
        )
        with self.assertRaises(IllegalCommand):
            illegal.apply(PlayCard(0, 0))
        self.assertEqual(
            (
                illegal.deterministic_fingerprint(),
                illegal.random.getstate(),
                tuple(illegal.event_history),
                tuple(illegal.logs),
            ),
            before,
        )

    def test_beheading_discard_chain_and_play_damage_use_physical_cost(self):
        engine = self.fresh(seed=53)
        original = _put_hand(engine, self.repository.get(10643310))
        observed_costs = []
        for expected in (7, 5, 3):
            current = next(card for card in engine.players[0].hand if card.card_id == 10643310)
            self.assertEqual(current.current_cost, expected)
            _discard(engine, current)
            observed_costs.append(
                next(
                    event.metadata["cost"]
                    for event in reversed(engine.event_history)
                    if event.type is EventType.CARD_DISCARDED
                    and event.metadata["card_id"] == 10643310
                )
            )
        self.assertEqual(observed_costs, [7, 5, 3])
        self.assertFalse(any(card.card_id == 10643310 for card in engine.players[0].hand))

        played = self.fresh(seed=59)
        enemies = [_put_unit(played, 1, _card(995500 + index, life=9)) for index in range(2)]
        source = _put_hand(played, self.repository.get(10643310))
        _set_cost(source, 5)
        played.apply(PlayCard(0, played.players[0].hand.index(source)))
        self.assertEqual([enemy.health for enemy in enemies], [4, 4])

    def test_truth_affirmer_heals_only_when_physical_play_cost_is_not_two(self):
        normal = self.fresh(seed=61)
        normal.players[0].health = 10
        source = _put_hand(normal, self.repository.get(10331110))
        normal.apply(PlayCard(0, normal.players[0].hand.index(source)))
        self.assertEqual(normal.players[0].health, 10)

        modified = self.fresh(seed=67)
        modified.players[0].health = 10
        source = _put_hand(modified, self.repository.get(10331110))
        _set_cost(source, 1)
        modified.apply(PlayCard(0, modified.players[0].hand.index(source)))
        self.assertEqual(modified.players[0].health, 13)

    def test_rl_action_layout_reuses_existing_target_choice(self):
        deck = [_card(995600 + index, class_id=0, class_name="中立") for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=4,
            class_b=4,
            seed=71,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        observation, _ = env.reset(seed=71)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        _put_hand(env.core, self.repository.get(10641310))
        play_action = env._encode_command(PlayCard(0, 0))
        self.assertEqual((env.ACTION_SIZE, len(observation)), (111, 294))
        self.assertFalse(env.action_mask()[play_action])
        target = _put_unit(env.core, 0, _card(995650, attack=1, life=2))
        self.assertTrue(env.action_mask()[play_action])
        env.step(play_action)
        choices = [
            action
            for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        ]
        self.assertEqual(len(choices), 1)
        env.step(choices[0])
        self.assertEqual((target.attack, target.health), (3, 4))


class DestroyedHistorySourceCostDatabaseAuditTests(unittest.TestCase):
    def test_database_text_references_and_modes_match_official_review(self):
        expected_phrases = {
            10572310: ("discard it", "2 random differently named", "without revealing"),
            10871130: ("Artifact follower destroyed", "without revealing"),
            10641310: ("cost is 4", "set its cost to 2", "give it +2/+2"),
            10643310: ("cost is 7", "cost is 5", "X is this card's cost"),
            10331110: ("cost isn't 2", "restore 3 defense"),
        }
        expected_references = {
            10572310: (),
            10871130: (),
            10641310: (10641310,),
            10643310: (10643310,),
            10331110: (),
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
                        row[0]
                        for row in connection.execute(
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
                    ["tests/test_real_destroyed_history_source_cost_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
