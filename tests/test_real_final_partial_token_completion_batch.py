# -*- coding: utf-8 -*-
"""Close the last two partially implemented generated-card behaviors."""

from __future__ import annotations

import unittest

from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import CardPassive, CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine
from swb.engine.state import DeathCause
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


GHOST_ID = 90051130
IMPROVED_PUPPET_ID = 90071120
SOURCE_HASHES = {
    GHOST_ID: "b1c3082346cccece90ba26f1432e9c917c430479b26a54e3fb617a8f8685b7dc",
    IMPROVED_PUPPET_ID: "59cabb19eb7ce9372deffcc55b53bf616c720e33efcd81f225a43702bc79115d",
}


def _choose_entity(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


class RealFinalPartialTokenCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 951):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def leave_engine(
        self,
        operation: EffectOperation,
        *,
        include_last_words: bool = False,
        seed: int = 953,
    ) -> tuple[GameEngine, object]:
        spell_id = 994001
        rules = [CardRule(spell_id, Trigger.PLAY, (operation,))]
        if include_last_words:
            rules.append(CardRule(
                GHOST_ID,
                Trigger.LAST_WORDS,
                (EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    amount=7,
                ),),
            ))
        engine = _fresh(
            RuleBook(
                rules=tuple(rules),
                passives=(CardPassive(GHOST_ID, "banish_on_leave", 0),),
            ),
            self.repository,
            seed=seed,
        )
        spell = _card(
            spell_id,
            cost=0,
            card_type="法术",
            attack=None,
            life=None,
        )
        return engine, spell

    def test_banish_on_leave_replaces_destroy_without_death_or_last_words(self):
        engine, spell = self.leave_engine(
            EffectOperation(
                EffectKind.DESTROY,
                TargetKind.ALL_ENEMY_UNITS,
            ),
            include_last_words=True,
            seed=3,
        )
        ghost = _put_unit(engine, 1, self.repository.get(GHOST_ID))
        ordinary = _put_unit(engine, 1, _card(994010))
        shadows_before = engine.players[1].shadows
        _put_hand(engine, spell)

        engine.apply(PlayCard(0, 0))

        self.assertNotIn(ghost, engine.players[1].board)
        self.assertIn(GHOST_ID, [card.card_id for card in engine.players[1].banished])
        self.assertNotIn(
            GHOST_ID,
            [card.definition.card_id for card in engine.players[1].graveyard],
        )
        self.assertEqual(engine.players[1].shadows, shadows_before + 1)
        self.assertEqual(
            [record.definition.card_id for record in engine.state.destroyed_followers],
            [ordinary.definition.card_id],
        )
        self.assertEqual(
            [record.card_id for record in engine.state.death_queue[-1].records],
            [ordinary.definition.card_id],
        )
        self.assertEqual(engine.players[0].health, 20)
        ghost_events = [
            event for event in engine.event_history
            if event.source_id == ghost.entity_id
        ]
        self.assertIn(EventType.CARD_BANISHED, {event.type for event in ghost_events})
        self.assertNotIn(EventType.FOLLOWER_DESTROYED, {event.type for event in ghost_events})
        leave = next(event for event in ghost_events if event.type is EventType.ENTITY_LEFT_PLAY)
        self.assertEqual(leave.metadata["cause"], DeathCause.BANISH.value)
        self.assertEqual(
            leave.metadata["replaced_leave_cause"],
            DeathCause.EFFECT_DESTROY.value,
        )

    def test_banish_on_leave_replaces_return_to_hand_and_deck_without_rng_use(self):
        cases = (
            (EffectKind.RETURN_TO_HAND, DeathCause.RETURN_TO_HAND),
            (EffectKind.RETURN_TO_DECK, DeathCause.RETURN_TO_DECK),
        )
        for index, (kind, replaced_cause) in enumerate(cases):
            with self.subTest(kind=kind):
                engine, spell = self.leave_engine(
                    EffectOperation(kind, TargetKind.ENEMY_UNIT),
                    seed=11 + index,
                )
                ghost = _put_unit(engine, 1, self.repository.get(GHOST_ID))
                _put_hand(engine, spell)
                rng_before = engine.random.getstate()

                engine.apply(PlayCard(0, 0))
                _choose_entity(engine, ghost.entity_id)

                self.assertEqual(engine.random.getstate(), rng_before)
                self.assertNotIn(ghost, engine.players[1].board)
                self.assertNotIn(GHOST_ID, [card.card_id for card in engine.players[1].hand])
                self.assertNotIn(GHOST_ID, [card.card_id for card in engine.players[1].deck])
                self.assertIn(GHOST_ID, [card.card_id for card in engine.players[1].banished])
                leave = next(
                    event for event in engine.event_history
                    if event.type is EventType.ENTITY_LEFT_PLAY
                    and event.source_id == ghost.entity_id
                )
                self.assertEqual(leave.metadata["cause"], DeathCause.BANISH.value)
                self.assertEqual(
                    leave.metadata["replaced_leave_cause"],
                    replaced_cause.value,
                )

    def test_removed_printed_abilities_disable_leave_replacement(self):
        engine, spell = self.leave_engine(
            EffectOperation(EffectKind.RETURN_TO_HAND, TargetKind.ENEMY_UNIT),
            seed=17,
        )
        ghost = _put_unit(engine, 1, self.repository.get(GHOST_ID))
        ghost.printed_abilities_removed = True
        ghost._synchronize_keyword_state()
        _put_hand(engine, spell)

        engine.apply(PlayCard(0, 0))
        _choose_entity(engine, ghost.entity_id)

        self.assertIn(GHOST_ID, [card.card_id for card in engine.players[1].hand])
        self.assertNotIn(GHOST_ID, [card.card_id for card in engine.players[1].banished])

    def test_real_ghost_producer_storm_and_own_turn_expiry_are_exact(self):
        engine = self.fresh(seed=23)
        _play(engine, self.repository, 10251310)
        ghost = next(card for card in engine.players[0].hand if card.card_id == GHOST_ID)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, engine.players[0].hand.index(ghost)))
        unit = next(unit for unit in engine.players[0].board if unit.definition.card_id == GHOST_ID)
        self.assertTrue(unit.has_keyword("疾驰"))
        shadows_before = engine.players[0].shadows

        engine.apply(EndTurn(0))

        self.assertNotIn(unit, engine.players[0].board)
        self.assertIn(GHOST_ID, [card.card_id for card in engine.players[0].banished])
        self.assertEqual(engine.players[0].shadows, shadows_before)
        self.assertNotIn(
            GHOST_ID,
            [card.definition.card_id for card in engine.players[0].graveyard],
        )

    def test_real_improved_puppet_producer_and_opponent_turn_expiry_are_exact(self):
        engine = self.fresh(seed=29)
        _play(engine, self.repository, 10171310)
        puppets = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == IMPROVED_PUPPET_ID
        ]
        self.assertEqual(len(puppets), 2)
        self.assertTrue(all(unit.has_keyword("突进") for unit in puppets))
        shadows_before = engine.players[0].shadows

        engine.apply(EndTurn(0))
        self.assertTrue(all(unit in engine.players[0].board for unit in puppets))
        engine.apply(EndTurn(1))

        self.assertTrue(all(unit not in engine.players[0].board for unit in puppets))
        self.assertEqual(engine.players[0].shadows, shadows_before + 2)
        self.assertEqual(
            sum(
                card.definition.card_id == IMPROVED_PUPPET_ID
                for card in engine.players[0].graveyard
            ),
            2,
        )

    def test_both_tokens_are_now_behavior_complete_with_real_producers(self):
        report = _build_token_audit("data/cards.sqlite3", "data/rules")
        cards = {card["card_id"]: card for card in report["cards"]}
        for card_id in (GHOST_ID, IMPROVED_PUPPET_ID):
            with self.subTest(card_id=card_id):
                info = cards[card_id]
                self.assertEqual(info["category"], "entry_behavior_complete")
                self.assertEqual(info["explicit_coverage"], "exact")
                self.assertTrue(info["authored_producers"])
                self.assertEqual(len(SOURCE_HASHES[card_id]), 64)


if __name__ == "__main__":
    unittest.main()
