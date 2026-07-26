# -*- coding: utf-8 -*-
"""P0 regressions for start/end-of-turn timing and trigger batching."""

from __future__ import annotations

import unittest

from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import EndTurn
from swb.engine.events import EventType
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


class TurnTimingCornerCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int) -> object:
        return _fresh(self.rulebook, self.repository, seed=seed)

    def _add_real_emblem(
        self,
        engine,
        owner: int,
        emblem_id: str,
        source_card_id: int,
    ):
        definition = self.rulebook.emblem_def(emblem_id)
        engine._add_emblem_to_player(owner, definition, source_card_id)
        return engine.players[owner].emblems[-1]

    def test_turn_start_invocation_precedes_countdown_last_words(self):
        engine = self.fresh(seed=9101)
        for card_id in range(991001, 991005):
            _put_unit(engine, 0, _card(card_id, life=5))
        pact = _play(engine, self.repository, 10163210)
        self.assertIsNotNone(pact)
        pact.countdown = 1

        player = engine.players[0]
        player.followers_evolved_this_match = 6
        player.deck.extend(
            (
                self.repository.get(10404110),
                _card(991010, name="normal draw"),
            )
        )
        event_start = len(engine.event_history)

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))

        self.assertFalse(
            any(entity.definition.card_id == 90061120 for entity in player.board)
        )
        self.assertTrue(any(card.card_id == 10404110 for card in player.hand))
        new_events = engine.event_history[event_start:]
        invoked_index = next(
            index
            for index, event in enumerate(new_events)
            if event.type is EventType.CARD_INVOKED
            and event.metadata.get("card_id") == 10404110
        )
        last_words_index = next(
            index
            for index, event in enumerate(new_events)
            if event.type is EventType.LAST_WORDS_START
            and event.source_id == pact.entity_id
        )
        self.assertLess(invoked_index, last_words_index)
        engine.assert_invariants()

    def test_until_opponent_turn_end_covers_end_triggers(self):
        engine = self.fresh(seed=9102)
        zoey = _play(engine, self.repository, 10444120, mode_id="enhance_10")
        self.assertIsNotNone(zoey)
        self._add_real_emblem(
            engine,
            1,
            "underground_bounty_hunter_balt",
            10153140,
        )
        protected = engine.players[0]
        opponent = engine.players[1]
        self.assertEqual(protected.health, 1)

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))

        self.assertEqual(protected.health, 1)
        self.assertEqual(opponent.health, 19)
        self.assertFalse(protected.leader_damage_modifiers)
        self.assertFalse(engine.terminated)
        engine.assert_invariants()

    def test_turn_end_conditions_are_snapshotted(self):
        engine = self.fresh(seed=9103)
        frog = _play(engine, self.repository, 10522120)
        self.assertIsNotNone(frog)
        _put_hand(
            engine,
            _card(
                991020,
                name="only spell at timing start",
                card_type="法术",
                attack=None,
                life=None,
            ),
        )
        self._add_real_emblem(engine, 0, "unkei_goldbloom", 10524120)
        enemy = _put_unit(engine, 1, _card(991021, life=8))

        engine.apply(EndTurn(0))

        self.assertEqual(enemy.health, 8)
        self.assertEqual(
            sum(card.card_id == 90021350 for card in engine.players[0].hand),
            1,
        )
        engine.assert_invariants()

    def test_newly_eligible_turn_end_effect_does_not_join_batch(self):
        engine = self.fresh(seed=9104)
        galleon = _play(engine, self.repository, 10464110)
        ramletta = _play(engine, self.repository, 10461120)
        self.assertIsNotNone(galleon)
        self.assertIsNotNone(ramletta)
        enemy = _put_unit(engine, 1, _card(991030, life=8))
        galleon.evolved = True
        engine.players[0].turns_started = (
            engine.config.first_player_super_evolution_unlock_turn
        )
        health_before = (galleon.health, ramletta.health, enemy.health)

        engine.apply(EndTurn(0))

        self.assertTrue(ramletta.evolved)
        self.assertEqual(galleon.health, health_before[0])
        self.assertEqual(ramletta.health, health_before[1] + 2)
        self.assertEqual(enemy.health, health_before[2])
        engine.assert_invariants()

    def test_all_turn_end_sources_precede_resulting_last_words(self):
        engine = self.fresh(seed=9105)
        _put_unit(engine, 0, self.repository.get(90073130))
        _put_unit(engine, 0, self.repository.get(90073130))
        original = _put_unit(engine, 1, self.repository.get(90051140))

        engine.apply(EndTurn(0))

        zombies = [
            unit
            for unit in engine.players[1].board
            if unit.definition.card_id == 90051140
        ]
        self.assertEqual(len(zombies), 1)
        self.assertNotEqual(zombies[0].entity_id, original.entity_id)
        self.assertTrue(zombies[0].printed_abilities_removed)
        engine.assert_invariants()

    def test_marwynn_crest_precedes_dark_dimension(self):
        engine = self.fresh(seed=9106)
        self._add_real_emblem(
            engine,
            0,
            "despair_manifest_marwynn",
            10364120,
        )
        dark_dimension = _play(engine, self.repository, 10603210)
        self.assertIsNotNone(dark_dimension)
        enemy = _put_unit(engine, 1, _card(991041, life=10))
        event_start = len(engine.event_history)

        engine.apply(EndTurn(0))

        new_events = engine.event_history[event_start:]
        marwynn_index = next(
            index
            for index, event in enumerate(new_events)
            if event.type is EventType.EMBLEM_TRIGGERED
            and event.metadata.get("emblem_id") == "despair_manifest_marwynn"
        )
        marwynn_damage_index = next(
            index
            for index, event in enumerate(new_events)
            if event.type is EventType.DAMAGE_APPLIED
            and event.amount == 1
        )
        dark_damage_index = next(
            index
            for index, event in enumerate(new_events)
            if event.type is EventType.DAMAGE_APPLIED
            and event.target_id == enemy.entity_id
            and event.amount == 2
        )
        self.assertLess(marwynn_index, dark_damage_index)
        self.assertLess(marwynn_damage_index, dark_damage_index)
        engine.assert_invariants()


if __name__ == "__main__":
    unittest.main()
