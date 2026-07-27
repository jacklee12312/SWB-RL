# -*- coding: utf-8 -*-
"""Exact Mjerrabaine replacement deck, Victory Card, and token chain."""

from __future__ import annotations

import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import (
    RuleBook,
    Trigger,
    _parse_hand_filter,
    _parse_operation,
)
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard
from swb.engine.effects import EmptyDeckOutcome, EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


SOURCE_ID = 10304110
TOKEN_ID = 90004320
CARD_SET_ID = 10003
EMBLEM_ID = "mjerrabaine_great_manifest"
SOURCE_HASHES = {
    SOURCE_ID: "56d5e1874aad0ece9dfc8ec8e5aa0083ed2143ab3a660658565db7d351390bb6",
    TOKEN_ID: "66ec49c21c8e25e6493b951661ca2eac60463d34540a24c76b935050158a13da",
}


def _choose_target(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


class RealMjerrabaineDeckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")
        cls.emblem = cls.rulebook.emblem_def(EMBLEM_ID)
        cls.special_deck_ids = cls.emblem.on_gain[0].card_ids

    def fresh(self, *, seed: int = 4110):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def evolve_source(self, *, seed: int = 4110):
        engine = self.fresh(seed=seed)
        source = _play(engine, self.repository, SOURCE_ID)
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        engine.apply(Evolve(0, source.entity_id))
        return engine, source

    def test_rule_shapes_and_new_schema_validation(self):
        fanfare = self.rulebook.operations_for(SOURCE_ID, Trigger.FANFARE)
        evolve = self.rulebook.operations_for(SOURCE_ID, Trigger.EVOLVE)
        testimony = self.rulebook.operations_for(TOKEN_ID, Trigger.PLAY)

        self.assertEqual((fanfare[0].kind, fanfare[0].card_id), (EffectKind.ADD_CARD, TOKEN_ID))
        self.assertEqual((evolve[0].kind, evolve[0].emblem_id), (EffectKind.GAIN_EMBLEM, EMBLEM_ID))
        self.assertEqual(
            [(operation.kind, operation.target) for operation in self.emblem.on_gain],
            [
                (EffectKind.REPLACE_DECK, TargetKind.OWN_LEADER),
                (EffectKind.SET_EMPTY_DECK_OUTCOME, TargetKind.OWN_LEADER),
            ],
        )
        self.assertEqual(len(self.special_deck_ids), 76)
        self.assertEqual(len(set(self.special_deck_ids)), 76)
        self.assertNotIn(SOURCE_ID, self.special_deck_ids)
        self.assertTrue(self.emblem.on_gain[0].shuffle)
        self.assertIs(
            self.emblem.on_gain[1].empty_deck_outcome,
            EmptyDeckOutcome.VICTORY,
        )
        discard, draw = self.emblem.triggers[0].operations
        self.assertEqual((discard.kind, discard.target), (EffectKind.DISCARD, TargetKind.ALL_OWN_HAND))
        self.assertEqual(discard.hand_filter.exclude_card_ids, (TOKEN_ID,))
        self.assertEqual((draw.kind, draw.amount), (EffectKind.DRAW, 6))
        self.assertTrue(testimony[0].requires_target)

        parsed = _parse_operation(
            {
                "kind": "replace_deck",
                "target": "own_leader",
                "card_ids": [11, 12],
                "shuffle": False,
            },
            "test.json/operations[0]",
            SOURCE_ID,
        )
        self.assertEqual(parsed.card_ids, (11, 12))
        self.assertFalse(parsed.shuffle)
        outcome = _parse_operation(
            {
                "kind": "set_empty_deck_outcome",
                "target": "own_leader",
                "outcome": "victory",
            },
            "test.json/operations[0]",
            SOURCE_ID,
        )
        self.assertIs(outcome.empty_deck_outcome, EmptyDeckOutcome.VICTORY)
        hand_filter = _parse_hand_filter(
            {"exclude_card_ids": [TOKEN_ID]},
            source_path="test.json/hand_filter",
            card_id=SOURCE_ID,
        )
        self.assertFalse(hand_filter.matches(self.repository.get(TOKEN_ID)))
        self.assertTrue(hand_filter.matches(self.repository.get(SOURCE_ID)))

        invalid_operations = (
            {"kind": "replace_deck", "target": "own_leader", "card_ids": []},
            {"kind": "replace_deck", "target": "own_leader", "card_ids": [0]},
            {"kind": "replace_deck", "target": "own_unit", "card_ids": [11]},
            {"kind": "replace_deck", "target": "own_leader", "card_ids": [11], "shuffle": 1},
            {"kind": "draw", "target": "own_leader", "amount": 1, "card_ids": [11]},
            {"kind": "set_empty_deck_outcome", "target": "own_leader", "outcome": "draw"},
            {"kind": "set_empty_deck_outcome", "target": "own_unit", "outcome": "victory"},
        )
        for raw in invalid_operations:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test.json/operations[0]", SOURCE_ID)
        for raw_filter in (
            {"exclude_card_ids": TOKEN_ID},
            {"exclude_card_ids": [0]},
            {"exclude_card_ids": [TOKEN_ID, TOKEN_ID]},
        ):
            with self.subTest(raw_filter=raw_filter), self.assertRaises(ValueError):
                _parse_hand_filter(
                    raw_filter,
                    source_path="test.json/hand_filter",
                    card_id=SOURCE_ID,
                )

    def test_empty_deck_draw_is_official_immediate_defeat_not_fatigue(self):
        engine = self.fresh(seed=3)
        engine.players[1].deck.clear()

        result = engine.apply(EndTurn(0))

        self.assertTrue(result.terminated)
        self.assertEqual(result.winner, 0)
        self.assertEqual(engine.players[1].health, 20)
        self.assertEqual(engine.players[1].fatigue, 0)
        resolved = [
            event
            for event in engine.event_history
            if event.type is EventType.EMPTY_DECK_DRAW_RESOLVED
        ]
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].metadata["outcome"], "defeat")
        self.assertTrue(any(event.type is EventType.GAME_ENDED for event in engine.event_history))

    def test_fanfare_and_evolve_create_exact_seeded_special_deck(self):
        engine, source = self.evolve_source(seed=5)

        self.assertEqual([card.card_id for card in engine.players[0].hand], [TOKEN_ID])
        self.assertEqual([emblem.emblem_id for emblem in engine.players[0].emblems], [EMBLEM_ID])
        actual_ids = [card.card_id for card in engine.players[0].deck]
        self.assertEqual(len(actual_ids), 76)
        self.assertEqual(set(actual_ids), set(self.special_deck_ids))
        self.assertEqual(len(set(actual_ids)), 76)
        self.assertNotIn(SOURCE_ID, actual_ids)
        self.assertIs(
            engine.players[0].empty_deck_outcome,
            EmptyDeckOutcome.VICTORY,
        )
        event_types = [event.type for event in engine.event_history]
        self.assertLess(event_types.index(EventType.EMBLEM_GAINED), event_types.index(EventType.DECK_REPLACED))
        self.assertLess(event_types.index(EventType.DECK_REPLACED), event_types.index(EventType.EMPTY_DECK_OUTCOME_CHANGED))
        self.assertTrue(source.evolved)

    def test_replacement_shuffle_is_deterministic_and_ignored_gain_is_noop(self):
        first, _ = self.evolve_source(seed=7)
        second, _ = self.evolve_source(seed=7)
        third, _ = self.evolve_source(seed=8)
        first_order = tuple(card.card_id for card in first.players[0].deck)
        second_order = tuple(card.card_id for card in second.players[0].deck)
        third_order = tuple(card.card_id for card in third.players[0].deck)

        self.assertEqual(first_order, second_order)
        self.assertNotEqual(first_order, third_order)
        before_rng = first.random.getstate()
        before_fingerprint = first.deterministic_fingerprint()
        first._add_emblem_to_player(0, self.emblem, self.repository.get(SOURCE_ID))
        self.assertEqual(first.random.getstate(), before_rng)
        self.assertEqual(first.deterministic_fingerprint(), before_fingerprint)
        self.assertEqual(len(first.players[0].emblems), 1)

    def test_turn_end_discards_every_non_testimony_then_draws_six(self):
        engine, _ = self.evolve_source(seed=11)
        discarded_ids = (99101, 99102)
        for card_id in discarded_ids:
            _put_hand(engine, _card(card_id, card_type="法术", attack=None, life=None))

        engine.apply(EndTurn(0))

        hand_ids = [card.card_id for card in engine.players[0].hand]
        self.assertEqual(hand_ids.count(TOKEN_ID), 1)
        self.assertEqual(len(hand_ids), 7)
        self.assertEqual(len(engine.players[0].deck), 70)
        self.assertTrue(
            set(discarded_ids).issubset(
                {card.definition.card_id for card in engine.players[0].graveyard}
            )
        )
        discarded_events = [
            event.metadata["card_id"]
            for event in engine.event_history
            if event.type is EventType.CARD_DISCARDED
        ]
        self.assertEqual(set(discarded_events), set(discarded_ids))
        self.assertNotIn(TOKEN_ID, discarded_events)

    def test_victory_card_result_stops_turn_end_resolution_immediately(self):
        engine, _ = self.evolve_source(seed=13)
        engine.players[0].deck.clear()
        turn_before = engine.turn

        result = engine.apply(EndTurn(0))

        self.assertTrue(result.terminated)
        self.assertEqual(result.winner, 0)
        self.assertEqual(engine.current_player, 0)
        self.assertEqual(engine.turn, turn_before)
        resolved = [
            event
            for event in engine.event_history
            if event.type is EventType.EMPTY_DECK_DRAW_RESOLVED
        ]
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].metadata["outcome"], "victory")
        self.assertEqual(resolved[0].metadata["winner"], 0)

    def test_testimony_requires_target_and_illegal_play_preserves_everything(self):
        engine = self.fresh(seed=17)
        _put_hand(engine, self.repository.get(TOKEN_ID))
        before = engine.deterministic_fingerprint()
        before_rng = engine.random.getstate()
        before_events = tuple(engine.event_history)
        before_logs = tuple(engine.logs)

        self.assertNotIn(PlayCard(0, 0), engine.legal_commands())
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.deterministic_fingerprint(), before)
        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertEqual(tuple(engine.event_history), before_events)
        self.assertEqual(tuple(engine.logs), before_logs)

    def test_testimony_destroys_selected_target_and_skips_stale_target(self):
        engine = self.fresh(seed=19)
        target = _put_unit(engine, 1, _card(99201, life=9))
        _play(engine, self.repository, TOKEN_ID)
        self.assertIsNotNone(engine.state.pending_choice)
        _choose_target(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)
        self.assertTrue(
            any(
                card.definition.card_id == TOKEN_ID
                for card in engine.players[0].graveyard
            )
        )

        stale = self.fresh(seed=23)
        stale_target = _put_unit(stale, 1, _card(99202, life=9))
        _play(stale, self.repository, TOKEN_ID)
        request = stale.state.pending_choice
        option = next(option for option in request.options if option.entity_id == stale_target.entity_id)
        stale.players[1].board.remove(stale_target)

        stale.apply(Choose(request.player_index, option.option_id))

        self.assertFalse(stale.players[1].board)
        self.assertTrue(
            any(
                card.definition.card_id == TOKEN_ID
                for card in stale.players[0].graveyard
            )
        )
        stale.assert_invariants()

    def test_rl_masks_and_v2_observation_expose_public_victory_mode(self):
        deck_a = [_card(99300 + index) for index in range(40)]
        deck_b = [_card(99400 + index) for index in range(40)]
        vocabulary = tuple(
            [card.card_id for card in (*deck_a, *deck_b)] + [SOURCE_ID, TOKEN_ID]
        )
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=29,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            observation_version="v2",
            card_vocabulary=vocabulary,
            validate_invariants=True,
        )
        env.reset(seed=29)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        _put_hand(env.core, self.repository.get(TOKEN_ID))
        play_action = env.PLAY_OFFSET
        self.assertFalse(env.action_mask()[play_action])
        target = _put_unit(env.core, 1, _card(99501, life=7))
        self.assertTrue(env.action_mask()[play_action])

        env.players[0].empty_deck_outcome = EmptyDeckOutcome.VICTORY
        observation = env.observation()
        self.assertEqual(observation["leader_area"]["empty_deck_outcomes"], (1, 0))
        self.assertEqual(env.observation_v2_spec()["empty_deck_outcomes"], 2)

        result = env.step(play_action)
        self.assertFalse(result.terminated)
        request = env.core.state.pending_choice
        option = next(option for option in request.options if option.entity_id == target.entity_id)
        choice_action = env._encode_command(Choose(request.player_index, option.option_id))
        self.assertTrue(env.action_mask()[choice_action])
        env.step(choice_action)
        self.assertNotIn(target, env.players[1].board)
        self.assertEqual(env.ACTION_SIZE, 112)
        self.assertEqual(
            len(env.observation()["continuous_v1"]),
            ShadowverseEnv.OBSERVATION_V1_SIZE,
        )


class MjerrabaineDatabaseAuditTests(unittest.TestCase):
    def test_imported_set_and_source_text_match_reviewed_special_deck(self):
        rulebook = RuleBook.from_directory("data/rules")
        special_ids = rulebook.emblem_def(EMBLEM_ID).on_gain[0].card_ids
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            set_id = connection.execute(
                "SELECT card_set_id FROM cards WHERE card_id=?",
                (SOURCE_ID,),
            ).fetchone()[0]
            imported_ids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT card_id FROM cards WHERE card_set_id=? AND card_id<>? ORDER BY card_id",
                    (set_id, SOURCE_ID),
                )
            )
            self.assertEqual(set_id, CARD_SET_ID)
            self.assertEqual(len(imported_ids), 76)
            self.assertEqual(set(imported_ids), set(special_ids))
            primary = connection.execute(
                "SELECT text_eng FROM skill_texts WHERE card_id=? ORDER BY position LIMIT 1",
                (SOURCE_ID,),
            ).fetchone()[0]
            crest = connection.execute(
                "SELECT text_eng FROM alt_modes WHERE card_id=? AND mode_type='纹章'",
                (SOURCE_ID,),
            ).fetchone()[0]
            testimony = connection.execute(
                "SELECT text_eng FROM skill_texts WHERE card_id=?",
                (TOKEN_ID,),
            ).fetchone()[0]
            self.assertIn("Great Testimony", primary)
            self.assertIn("Mjerrabaine Deck", crest)
            self.assertIn("draw 6 cards", crest)
            self.assertIn("destroy it", testimony)
            self.assertEqual(
                connection.execute(
                    "SELECT referenced_card_id FROM card_references WHERE card_id=?",
                    (SOURCE_ID,),
                ).fetchall(),
                [(TOKEN_ID,)],
            )

    def test_source_and_token_are_exact_with_executable_producer(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        source = report["classifications"][str(SOURCE_ID)]
        token = report["classifications"][str(TOKEN_ID)]
        self.assertEqual(source["coverage"], "covered_exact")
        self.assertEqual(source["clause_audit"]["status"], "mapped_exact")
        for card_id, info in ((SOURCE_ID, source), (TOKEN_ID, token)):
            self.assertEqual(
                info["clause_audit"]["source_text_sha256"],
                SOURCE_HASHES[card_id],
            )

        audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        testimony = next(card for card in audit["cards"] if card["card_id"] == TOKEN_ID)
        self.assertEqual(testimony["category"], "entry_behavior_complete")
        self.assertEqual(testimony["explicit_coverage"], "exact")
        self.assertEqual(
            testimony["authored_producers"],
            [{
                "source_card_id": SOURCE_ID,
                "entry_kind": "add_card",
                "rule_file": "real_mjerrabaine_deck_batch.json",
                "rule_group": "rules",
            }],
        )


if __name__ == "__main__":
    unittest.main()
