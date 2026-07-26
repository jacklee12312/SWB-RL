# -*- coding: utf-8 -*-
"""P1 regressions for lethal ordering, overdraw, and timing batches."""

from __future__ import annotations

import copy
import unittest

from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Attack, EndTurn
from swb.engine.effects import (
    EffectKind,
    EffectOperation,
    TargetKind,
    TurnEndDestroyTiming,
)
from swb.engine.events import EventType
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _put_hand,
    _put_unit,
)


class TurnTimingP1CornerCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(
        self,
        *,
        seed: int,
        rulebook: RuleBook | None = None,
    ):
        return _fresh(
            self.rulebook if rulebook is None else rulebook,
            self.repository,
            seed=seed,
        )

    def _add_real_emblem(
        self,
        engine,
        owner: int,
        emblem_id: str,
        source_card_id: int,
    ):
        definition = engine.rulebook.emblem_def(emblem_id)
        engine._add_emblem_to_player(owner, definition, source_card_id)
        return engine.players[owner].emblems[-1]

    def _ready_super_evolved_attacker(self, engine, *, card_id: int):
        attacker = _put_unit(
            engine,
            0,
            _card(card_id, name="P1 super-evolved attacker", attack=10, life=10),
        )
        attacker.evolved = True
        attacker.super_evolved = True
        attacker.super_evolved_turn = engine.turn
        attacker.can_attack = True
        attacker.attacks_remaining = 1
        attacker.summoned_this_turn = False
        return attacker

    def _fill_hand(self, engine) -> None:
        while len(engine.players[0].hand) < engine.config.max_hand:
            _put_hand(
                engine,
                _card(992000 + len(engine.players[0].hand)),
            )

    def test_balt_lethal_to_both_leaders_awards_opponent(self):
        engine = self.fresh(seed=9201)
        self._add_real_emblem(
            engine,
            0,
            "underground_bounty_hunter_balt",
            10153140,
        )
        engine.players[0].health = 1
        engine.players[1].health = 1

        transition = engine.apply(EndTurn(0))

        self.assertTrue(transition.terminated)
        self.assertEqual(transition.winner, 1)
        self.assertEqual(
            (engine.players[0].health, engine.players[1].health),
            (0, 1),
        )
        damage_events = [
            event
            for event in transition.events
            if event.type is EventType.DAMAGE_APPLIED
        ]
        self.assertEqual(len(damage_events), 1)
        engine.assert_invariants()

    def test_super_evolution_lethal_stops_mimi_counter_damage(self):
        engine = self.fresh(seed=9202)
        attacker = self._ready_super_evolved_attacker(
            engine,
            card_id=992101,
        )
        mimi = _put_unit(engine, 1, self.repository.get(90054110))
        engine.players[0].health = 1
        engine.players[1].health = 1

        transition = engine.apply(
            Attack(0, attacker.entity_id, mimi.entity_id)
        )

        self.assertTrue(transition.terminated)
        self.assertEqual(transition.winner, 0)
        self.assertEqual(
            (engine.players[0].health, engine.players[1].health),
            (1, 0),
        )
        self.assertFalse(any(
            event.type is EventType.LAST_WORDS_START
            and event.source_id == mimi.entity_id
            for event in transition.events
        ))
        engine.assert_invariants()

    def test_super_evolution_lethal_stops_coco_last_words_heal(self):
        engine = self.fresh(seed=9203)
        attacker = self._ready_super_evolved_attacker(
            engine,
            card_id=992102,
        )
        coco = _put_unit(engine, 1, self.repository.get(90054120))
        engine.players[1].health = 1

        transition = engine.apply(
            Attack(0, attacker.entity_id, coco.entity_id)
        )

        self.assertTrue(transition.terminated)
        self.assertEqual(transition.winner, 0)
        self.assertEqual(engine.players[1].health, 0)
        self.assertFalse(any(
            event.type is EventType.LAST_WORDS_START
            and event.source_id == coco.entity_id
            for event in transition.events
        ))
        engine.assert_invariants()

    def test_overdraw_does_not_trigger_desperate_shrinemouse(self):
        engine = self.fresh(seed=9204)
        _put_unit(engine, 0, self.repository.get(10562120))
        enemy = _put_unit(engine, 1, _card(992201, life=8))
        self._fill_hand(engine)
        engine.players[0].deck = [_card(992202, cost=1)]
        event_start = len(engine.event_history)

        engine._draw(0, reason="P1 overdraw")
        engine._resolve_event_queue()
        engine._stabilize()

        self.assertEqual(enemy.health, 8)
        self.assertFalse(any(
            event.type is EventType.CARD_DRAWN
            for event in engine.event_history[event_start:]
        ))
        self.assertEqual(engine.players[0].graveyard[-1].entry_cause, "overdraw")
        engine.assert_invariants()

    def test_overdraw_does_not_trigger_mistbloom_emblem(self):
        engine = self.fresh(seed=9205)
        self._add_real_emblem(
            engine,
            0,
            "kukishiro_mistbloom",
            10564120,
        )
        self._fill_hand(engine)
        engine.players[0].deck = [_card(992301, cost=1)]
        event_start = len(engine.event_history)

        engine._draw(0, reason="P1 overdraw")
        engine._resolve_event_queue()
        engine._stabilize()

        new_events = engine.event_history[event_start:]
        self.assertFalse(any(
            event.type is EventType.CARD_DRAWN
            for event in new_events
        ))
        self.assertFalse(any(
            event.type is EventType.RANDOM_CHOICES_SELECTED
            for event in new_events
        ))
        self.assertFalse(engine.players[0].board)
        self.assertFalse(engine.players[1].board)
        engine.assert_invariants()

    def test_follower_summoned_mid_turn_end_waits_for_next_batch(self):
        rulebook = copy.deepcopy(self.rulebook)
        summoner_id = 992401
        summoned_id = 90031120
        rulebook._rules[(summoner_id, Trigger.TURN_END)] = (
            EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=summoned_id,
            ),
        )
        rulebook._rules[(summoned_id, Trigger.TURN_END)] = (
            EffectOperation(
                kind=EffectKind.DAMAGE_LEADER,
                target=TargetKind.ENEMY_LEADER,
                amount=1,
            ),
        )
        engine = self.fresh(seed=9206, rulebook=rulebook)
        _put_unit(engine, 0, _card(summoner_id))

        engine.apply(EndTurn(0))

        self.assertEqual(engine.players[1].health, 20)
        self.assertEqual(
            sum(
                unit.definition.card_id == summoned_id
                for unit in engine.players[0].board
            ),
            1,
        )

        engine.apply(EndTurn(1))
        engine.apply(EndTurn(0))

        self.assertEqual(engine.players[1].health, 19)
        engine.assert_invariants()

    def test_turn_end_ability_granted_mid_batch_waits_for_next_batch(self):
        rulebook = copy.deepcopy(self.rulebook)
        granter_id = 992501
        target_id = 992502
        rulebook._rules[(granter_id, Trigger.TURN_END)] = (
            EffectOperation(
                kind=EffectKind.GRANT_TURN_END_ABILITY,
                target=TargetKind.ALL_OWN_UNITS,
                exclude_source=True,
                turn_end_ability_timing=TurnEndDestroyTiming.OWNER_TURN,
                granted_operations=(
                    EffectOperation(
                        kind=EffectKind.DAMAGE_LEADER,
                        target=TargetKind.ENEMY_LEADER,
                        amount=1,
                    ),
                ),
            ),
        )
        engine = self.fresh(seed=9207, rulebook=rulebook)
        _put_unit(engine, 0, _card(granter_id))
        target = _put_unit(engine, 0, _card(target_id))

        engine.apply(EndTurn(0))

        self.assertEqual(engine.players[1].health, 20)
        self.assertEqual(len(target.granted_turn_end_abilities), 1)

        engine.apply(EndTurn(1))
        engine.apply(EndTurn(0))

        self.assertEqual(engine.players[1].health, 19)
        engine.assert_invariants()


if __name__ == "__main__":
    unittest.main()
