from __future__ import annotations

import os
import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard


def _card(card_id: int, **overrides) -> CardDefinition:
    values = {
        "card_id": card_id,
        "card_set_id": 10000,
        "class_id": 4,
        "class_name": "龙族",
        "name": f"c{card_id}",
        "cost": 1,
        "card_type": "法术",
        "attack": None,
        "life": None,
        "keywords": frozenset(),
        "support_level": "basic",
        "is_collectible": True,
    }
    values.update(overrides)
    return CardDefinition(**values)


def _engine(rulebook: RuleBook, *, resolver=None) -> GameEngine:
    deck_a = [_card(1000 + index, card_type="随从", attack=1, life=1) for index in range(40)]
    deck_b = [_card(2000 + index, card_type="随从", attack=1, life=1) for index in range(40)]
    engine = GameEngine(
        deck_a,
        deck_b,
        class_a=4,
        class_b=4,
        seed=42,
        rulebook=rulebook,
        card_resolver=resolver,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=42)
    return engine


def _insert(engine: GameEngine, definition: CardDefinition) -> None:
    card = HandCard(definition, engine.state.allocate_entity_id())
    engine.players[0].hand.insert(0, card)
    engine.players[0].hand_entity_ids.insert(0, card.entity_id)


class MaxManaPrimitiveTests(unittest.TestCase):
    def test_schema_requires_nonzero_amount_and_leader_target(self):
        operation = _parse_operation(
            {"kind": "change_max_mana", "target": "own_leader", "amount": -2},
            "test",
            1,
        )
        self.assertIs(operation.kind, EffectKind.CHANGE_MAX_MANA)
        for invalid in (
            {"kind": "change_max_mana", "target": "own_leader", "amount": 0},
            {"kind": "change_max_mana", "target": "own_unit", "amount": 1},
        ):
            with self.assertRaises(ValueError):
                _parse_operation(invalid, "test", 1)

    def test_increase_caps_at_ten_and_emits_applied_delta(self):
        source = _card(1)
        rule = CardRule(1, Trigger.PLAY, (
            EffectOperation(EffectKind.CHANGE_MAX_MANA, TargetKind.OWN_LEADER, 3),
        ))
        engine = _engine(RuleBook(rules=(rule,)))
        engine.players[0].max_mana = engine.players[0].mana = 9
        _insert(engine, source)

        engine.apply(PlayCard(0, 0))

        self.assertEqual((engine.players[0].mana, engine.players[0].max_mana), (8, 10))
        event = next(e for e in engine.event_history if e.type is EventType.MAX_MANA_CHANGED)
        self.assertEqual(event.metadata["requested_amount"], 3)
        self.assertEqual(event.metadata["applied_amount"], 1)

    def test_decrease_clamps_current_mana(self):
        source = _card(1)
        rule = CardRule(1, Trigger.PLAY, (
            EffectOperation(EffectKind.CHANGE_MAX_MANA, TargetKind.OWN_LEADER, -3),
        ))
        engine = _engine(RuleBook(rules=(rule,)))
        engine.players[0].max_mana = engine.players[0].mana = 8
        _insert(engine, source)

        engine.apply(PlayCard(0, 0))

        self.assertEqual((engine.players[0].mana, engine.players[0].max_mana), (5, 5))


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class DragonOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def test_oracle_reaches_ten_then_draws(self):
        oracle = self.repo.get(10042310)
        engine = _engine(self.rulebook, resolver=self.repo.get)
        engine.players[0].max_mana = engine.players[0].mana = 9
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        _insert(engine, oracle)
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].max_mana, 10)
        self.assertEqual(engine.players[0].mana, 6)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_oracle_below_ten_does_not_draw(self):
        oracle = self.repo.get(10042310)
        engine = _engine(self.rulebook, resolver=self.repo.get)
        engine.players[0].max_mana = engine.players[0].mana = 8
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        _insert(engine, oracle)
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].max_mana, 9)
        self.assertEqual(len(engine.players[0].deck), deck_before)

    def test_longfu_overflow_fanfare_evolves_and_then_raises_max_mana(self):
        longfu = self.repo.get(10143120)
        engine = _engine(self.rulebook, resolver=self.repo.get)
        engine.players[0].max_mana = engine.players[0].mana = 7
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        _insert(engine, longfu)
        ep_before = engine.players[0].evolution_points

        engine.apply(PlayCard(0, 0))

        unit = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10143120
        )
        self.assertTrue(unit.evolved)
        self.assertEqual(engine.players[0].max_mana, 8)
        self.assertEqual(engine.players[0].evolution_points, ep_before)
        evolution = next(
            event for event in engine.event_history
            if event.type is EventType.FOLLOWER_EVOLVED
            and event.source_id == unit.entity_id
        )
        self.assertEqual(evolution.metadata["cause"], "effect")

    def test_longfu_below_overflow_stays_unevolved(self):
        longfu = self.repo.get(10143120)
        engine = _engine(self.rulebook, resolver=self.repo.get)
        engine.players[0].max_mana = engine.players[0].mana = 6
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        _insert(engine, longfu)

        engine.apply(PlayCard(0, 0))

        unit = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10143120
        )
        self.assertFalse(unit.evolved)
        self.assertEqual(engine.players[0].max_mana, 6)


if __name__ == "__main__":
    unittest.main()
