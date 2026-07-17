# -*- coding: utf-8 -*-
"""Exact Apocalypse Deck replacement and all four generated entries."""

from __future__ import annotations

import sqlite3
import unittest
from collections import Counter
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Attack, Choose, EndTurn, PlayCard
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import HandCard
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


SOURCE_ID = 10104120
SILENT_RIDER_ID = 90004110
SERVANT_ID = 90004120
FIEND_ID = 90004130
ASTAROTH_ID = 90004310
HEAL_SPELL_ID = 90011310
SPECIAL_IDS = (
    SILENT_RIDER_ID,
    SERVANT_ID,
    FIEND_ID,
    ASTAROTH_ID,
)
SOURCE_HASHES = {
    SOURCE_ID: "32b3206e40810ae7f8e6d6a6868b9fe5aa1296df7210e9ee9a543f64e1c50e40",
    SILENT_RIDER_ID: "487bbc3ead279c8faccb6c74842ddc77225535cb22afa3b651dcdc587a172eb8",
    SERVANT_ID: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    FIEND_ID: "c2a5f313310bb1348f3df1870ec50e1317bf5fe1f1780b0c7a803873e246513c",
    ASTAROTH_ID: "004694355979fc9ffee0b9ba23b11fd25d3c3ec98e03bcaa6634e222ecd3efeb",
}


def _choose_target(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _put_hand_for(engine, owner: int, definition) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.GENERATED,
    )
    engine.players[owner].hand.insert(0, card)
    engine.players[owner].hand_entity_ids.insert(0, card.entity_id)
    return card


class RealApocalypseDeckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 4120):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_leader_max_health_schema_validation(self):
        source = self.rulebook.operations_for(SOURCE_ID, Trigger.FANFARE)
        fiend = self.rulebook.operations_for(FIEND_ID, Trigger.FANFARE)
        astaroth = self.rulebook.operations_for(ASTAROTH_ID, Trigger.PLAY)

        self.assertEqual(len(source), 1)
        self.assertEqual(source[0].kind, EffectKind.REPLACE_DECK)
        self.assertEqual(source[0].target, TargetKind.OWN_LEADER)
        self.assertTrue(source[0].shuffle)
        self.assertEqual(
            Counter(source[0].card_ids),
            Counter({
                SILENT_RIDER_ID: 3,
                SERVANT_ID: 3,
                FIEND_ID: 3,
                ASTAROTH_ID: 1,
            }),
        )
        self.assertEqual(
            [(effect.kind, effect.target, effect.amount) for effect in fiend],
            [
                (EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 6),
                (EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 6),
            ],
        )
        self.assertEqual(fiend[0].target_count, 2)
        self.assertFalse(fiend[0].requires_full_target_count)
        self.assertEqual(
            (astaroth[0].kind, astaroth[0].target, astaroth[0].amount),
            (EffectKind.SET_LEADER_MAX_HEALTH, TargetKind.ENEMY_LEADER, 1),
        )

        parsed = _parse_operation(
            {
                "kind": "set_leader_max_health",
                "target": "enemy_leader",
                "amount": 7,
            },
            "test.json/operations[0]",
            SOURCE_ID,
        )
        self.assertEqual(parsed.amount, 7)
        for raw in (
            {
                "kind": "set_leader_max_health",
                "target": "enemy_unit",
                "amount": 1,
            },
            {
                "kind": "set_leader_max_health",
                "target": "enemy_leader",
                "amount": 0,
            },
            {
                "kind": "set_leader_max_health",
                "target": "enemy_leader",
                "amount": True,
            },
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(
                    raw,
                    "test.json/operations[0]",
                    SOURCE_ID,
                )

    def test_fanfare_replaces_deck_with_exact_seeded_ten_card_composition(self):
        engine = self.fresh(seed=5)

        _play(engine, self.repository, SOURCE_ID)

        actual = tuple(card.card_id for card in engine.players[0].deck)
        self.assertEqual(len(actual), 10)
        self.assertEqual(
            Counter(actual),
            Counter({
                SILENT_RIDER_ID: 3,
                SERVANT_ID: 3,
                FIEND_ID: 3,
                ASTAROTH_ID: 1,
            }),
        )
        event = next(
            event
            for event in engine.event_history
            if event.type is EventType.DECK_REPLACED
        )
        self.assertEqual(event.amount, 10)
        self.assertEqual(event.metadata["card_ids"], actual)
        self.assertTrue(event.metadata["shuffled"])
        engine.assert_invariants()

    def test_replacement_shuffle_and_full_state_are_seed_reproducible(self):
        first = self.fresh(seed=7)
        second = self.fresh(seed=7)
        third = self.fresh(seed=8)
        for engine in (first, second, third):
            _play(engine, self.repository, SOURCE_ID)

        first_order = tuple(card.card_id for card in first.players[0].deck)
        second_order = tuple(card.card_id for card in second.players[0].deck)
        third_order = tuple(card.card_id for card in third.players[0].deck)
        self.assertEqual(first_order, second_order)
        self.assertEqual(
            first.deterministic_fingerprint(),
            second.deterministic_fingerprint(),
        )
        self.assertNotEqual(first_order, third_order)

    def test_illegal_source_play_preserves_state_rng_events_and_logs(self):
        engine = self.fresh(seed=11)
        _put_hand(engine, self.repository.get(SOURCE_ID))
        engine.players[0].mana = 9
        command = PlayCard(0, 0)
        before = engine.deterministic_fingerprint()
        before_rng = engine.random.getstate()
        before_events = tuple(engine.event_history)
        before_logs = tuple(engine.logs)

        self.assertNotIn(command, engine.legal_commands())
        with self.assertRaises(IllegalCommand):
            engine.apply(command)

        self.assertEqual(engine.deterministic_fingerprint(), before)
        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertEqual(tuple(engine.event_history), before_events)
        self.assertEqual(tuple(engine.logs), before_logs)

    def test_silent_rider_has_immediate_storm_attack(self):
        engine = self.fresh(seed=13)
        rider = _play(engine, self.repository, SILENT_RIDER_ID)

        self.assertTrue(rider.has_keyword("疾驰"))
        command = Attack(0, rider.entity_id, None)
        self.assertIn(command, engine.legal_commands())
        engine.apply(command)
        self.assertEqual(engine.players[1].health, 10)

    def test_servant_is_exact_vanilla_one_cost_thirteen_thirteen(self):
        engine = self.fresh(seed=17)
        servant = _play(engine, self.repository, SERVANT_ID)

        self.assertEqual(servant.definition.cost, 1)
        self.assertEqual(
            (servant.attack, servant.health, servant.max_health),
            (13, 13, 13),
        )
        self.assertFalse(servant.definition.keywords)
        self.assertIsNone(engine.state.pending_choice)

    def test_fiend_selects_two_distinct_followers_then_damages_leader(self):
        engine = self.fresh(seed=19)
        targets = [
            _put_unit(engine, 1, _card(99100 + index, life=10))
            for index in range(2)
        ]

        _play(engine, self.repository, FIEND_ID)
        self.assertEqual(engine.state.pending_choice.target_count, 2)
        _choose_target(engine, targets[0].entity_id)
        self.assertEqual(engine.state.pending_choice.target_count, 2)
        self.assertEqual(len(engine.state.pending_choice.selected_options), 1)
        _choose_target(engine, targets[1].entity_id)

        self.assertEqual([target.health for target in targets], [4, 4])
        self.assertEqual(engine.players[1].health, 14)

    def test_fiend_uses_as_many_targets_as_possible_with_one_or_zero(self):
        one = self.fresh(seed=23)
        target = _put_unit(one, 1, _card(99201, life=10))
        _play(one, self.repository, FIEND_ID)
        self.assertEqual(one.state.pending_choice.target_count, 1)
        _choose_target(one, target.entity_id)
        self.assertEqual(target.health, 4)
        self.assertEqual(one.players[1].health, 14)

        zero = self.fresh(seed=29)
        _play(zero, self.repository, FIEND_ID)
        self.assertIsNone(zero.state.pending_choice)
        self.assertEqual(zero.players[1].health, 14)

    def test_stale_second_fiend_target_cancels_unit_damage_but_not_leader_damage(self):
        engine = self.fresh(seed=31)
        first = _put_unit(engine, 1, _card(99301, life=10))
        stale = _put_unit(engine, 1, _card(99302, life=10))
        _play(engine, self.repository, FIEND_ID)
        _choose_target(engine, first.entity_id)
        request = engine.state.pending_choice
        stale_option = next(
            option for option in request.options if option.entity_id == stale.entity_id
        )
        engine.players[1].board.remove(stale)

        engine.apply(Choose(request.player_index, stale_option.option_id))

        self.assertEqual(first.health, 10)
        self.assertEqual(engine.players[1].health, 14)
        engine.assert_invariants()

    def test_astaroth_sets_and_clamps_max_health_without_damage_then_caps_healing(self):
        engine = self.fresh(seed=37)
        engine.players[1].health = 14

        _play(engine, self.repository, ASTAROTH_ID)

        opponent = engine.players[1]
        self.assertEqual((opponent.health, opponent.max_health), (1, 1))
        change = next(
            event
            for event in engine.event_history
            if event.type is EventType.LEADER_MAX_HEALTH_CHANGED
        )
        self.assertEqual(change.amount, -19)
        self.assertEqual(change.metadata["previous_health"], 14)
        self.assertEqual(change.metadata["current_health"], 1)
        self.assertFalse(any(
            event.type is EventType.DAMAGE_APPLIED
            for event in engine.event_history
        ))

        engine.apply(EndTurn(0))
        _put_hand_for(engine, 1, self.repository.get(HEAL_SPELL_ID))
        engine.apply(PlayCard(1, 0))
        self.assertEqual((opponent.health, opponent.max_health), (1, 1))
        engine.assert_invariants()

    def test_v2_exposes_max_health_while_v1_action_layout_stays_stable(self):
        deck_a = [_card(99400 + index) for index in range(40)]
        deck_b = [_card(99500 + index) for index in range(40)]
        vocabulary = tuple(
            [card.card_id for card in (*deck_a, *deck_b)]
            + [ASTAROTH_ID]
        )
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=41,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            observation_version="v2",
            card_vocabulary=vocabulary,
            validate_invariants=True,
        )
        env.reset(seed=41)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        _put_hand(env.core, self.repository.get(ASTAROTH_ID))

        self.assertTrue(env.action_mask()[env.PLAY_OFFSET])
        result = env.step(env.PLAY_OFFSET)

        self.assertEqual(
            result.observation["leader_area"]["leader_max_healths"],
            (20, 1),
        )
        self.assertEqual(env.observation_v2_spec()["leader_max_healths"], 2)
        self.assertEqual((env.ACTION_SIZE, len(result.observation["continuous_v1"])), (111, 294))


class ApocalypseDeckDatabaseAndAuditTests(unittest.TestCase):
    def test_database_stats_and_imported_text_match_reviewed_entries(self):
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            rows = {
                row[0]: row[1:]
                for row in connection.execute(
                    "SELECT card_id, cost, attack, life FROM cards "
                    "WHERE card_id IN (10104120,90004110,90004120,90004130,90004310)"
                )
            }
            self.assertEqual(rows[SILENT_RIDER_ID], (6, 10, 10))
            self.assertEqual(rows[SERVANT_ID], (1, 13, 13))
            self.assertEqual(rows[FIEND_ID], (5, 9, 6))
            self.assertEqual(rows[ASTAROTH_ID], (10, None, None))
            texts = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT card_id, text_eng FROM skill_texts "
                    "WHERE card_id IN (10104120,90004110,90004130,90004310)"
                )
            }
            self.assertIn("Apocalypse Deck", texts[SOURCE_ID])
            self.assertIn("Storm", texts[SILENT_RIDER_ID])
            self.assertIn("Select 2 enemy followers", texts[FIEND_ID])
            self.assertIn("max defense to 1", texts[ASTAROTH_ID])

    def test_collectible_is_mapped_exact_and_all_source_hashes_match(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        source = report["classifications"][str(SOURCE_ID)]
        self.assertEqual(source["coverage"], "covered_exact")
        self.assertEqual(source["clause_audit"]["status"], "mapped_exact")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id, expected_hash in SOURCE_HASHES.items():
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    expected_hash,
                )

    def test_all_tokens_have_complete_replace_deck_producers(self):
        audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        self.assertEqual(
            audit["summary"]["categories"],
            {
                "entry_behavior_complete": 91,
                "entry_behavior_partial": 0,
                "database_only_no_entry": 0,
                "text_unclear": 0,
                "external_blocker": 0,
            },
        )
        cards = {card["card_id"]: card for card in audit["cards"]}
        expected_producer = [{
            "source_card_id": SOURCE_ID,
            "entry_kind": "replace_deck",
            "rule_file": "real_apocalypse_deck_batch.json",
            "rule_group": "rules",
        }]
        for card_id in SPECIAL_IDS:
            with self.subTest(card_id=card_id):
                self.assertEqual(
                    cards[card_id]["category"],
                    "entry_behavior_complete",
                )
                self.assertEqual(cards[card_id]["explicit_coverage"], "exact")
                self.assertEqual(
                    cards[card_id]["authored_producers"],
                    expected_producer,
                )


if __name__ == "__main__":
    unittest.main()
