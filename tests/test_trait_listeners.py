from __future__ import annotations

import os
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import EndTurn, Evolve, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.emblem import EventScope
from swb.engine.events import EventType, GameEvent
from swb.engine.listeners import (
    CardListenerDefinition,
    EventCardFilter,
    ListenerZone,
    SourceRelation,
)
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


def _card(card_id: int, **overrides) -> CardDefinition:
    values = {
        "card_id": card_id,
        "card_set_id": 10000,
        "class_id": 1,
        "class_name": "精灵",
        "name": f"card-{card_id}",
        "cost": 1,
        "card_type": "随从",
        "attack": 2,
        "life": 5,
        "keywords": frozenset(),
        "support_level": "basic",
        "is_collectible": True,
    }
    values.update(overrides)
    return CardDefinition(**values)


def _engine(rulebook: RuleBook | None = None) -> GameEngine:
    engine = GameEngine(
        [_card(1000 + index) for index in range(40)],
        [_card(2000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=42,
        rulebook=rulebook or RuleBook(),
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=42)
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _place(
    engine: GameEngine,
    player_index: int,
    definition: CardDefinition,
) -> Unit:
    unit = engine._summon_follower_to_board(
        player_index,
        definition,
        summon_cause="test_setup",
    )
    assert unit is not None
    return unit


def _insert_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int = 0,
) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    player = engine.players[player_index]
    player.hand.append(card)
    player.hand_entity_ids.append(card.entity_id)
    return card


def _emit_summoned(
    engine: GameEngine,
    player_index: int,
    definition: CardDefinition,
) -> Unit:
    unit = _place(engine, player_index, definition)
    engine._emit(GameEvent(
        EventType.FOLLOWER_SUMMONED,
        player_index,
        source_id=unit.entity_id,
        metadata={"source": unit, "card_id": definition.card_id},
    ))
    engine._resolve_event_queue()
    engine._stabilize()
    return unit


class TraitListenerPrimitiveTests(unittest.TestCase):
    def test_trait_filter_uses_metadata_not_display_name(self):
        listener = CardListenerDefinition(
            card_id=300,
            zone=ListenerZone.BOARD,
            event=EventType.FOLLOWER_SUMMONED,
            event_scope=EventScope.OWNER_EVENT,
            source_relation=SourceRelation.OTHER,
            event_filter=EventCardFilter(
                card_type="随从",
                tribe_id=5,
                tribe_name="妖精",
            ),
            operations=(EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                amount=1,
            ),),
        )
        engine = _engine(RuleBook(listener_defs={300: (listener,)}))
        _place(engine, 0, _card(300))

        _emit_summoned(engine, 0, _card(400, name="妖精"))
        self.assertEqual(engine.players[1].health, 20)

        _emit_summoned(
            engine,
            0,
            _card(401, name="不同名称", tribe_id=5, tribe_name="妖精"),
        )
        self.assertEqual(engine.players[1].health, 19)

    def test_trait_metadata_participates_in_engine_fingerprint(self):
        first = _engine()
        second = _engine()
        _place(first, 0, _card(300, tribe_id=5, tribe_name="妖精"))
        _place(second, 0, _card(300))

        self.assertNotEqual(
            first.deterministic_fingerprint(),
            second.deterministic_fingerprint(),
        )


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealTraitListenerBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def _real_engine(self) -> GameEngine:
        engine = GameEngine(
            [_card(1000 + index) for index in range(40)],
            [_card(2000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=self.rulebook,
            card_resolver=self.repo.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=42)
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10
        return engine

    def test_fairy_sword_bearer_reacts_to_any_fairy_trait(self):
        engine = self._real_engine()
        bearer = _place(engine, 0, self.repo.get(10311120))
        attack_before = bearer.attack

        _emit_summoned(engine, 0, self.repo.get(90011120))
        self.assertEqual(bearer.attack, attack_before + 1)

        _emit_summoned(engine, 0, _card(400, name="妖精"))
        self.assertEqual(bearer.attack, attack_before + 1)

    def test_yuni_heals_only_at_end_of_her_controllers_turn(self):
        engine = self._real_engine()
        yuni = _place(engine, 0, self.repo.get(10402110))
        ally = _place(engine, 0, _card(400))
        yuni.health -= 1
        ally.health -= 2
        engine.players[0].health = 15

        engine.apply(EndTurn(0))

        self.assertEqual(yuni.health, yuni.max_health)
        self.assertEqual(ally.health, ally.max_health - 1)
        self.assertEqual(engine.players[0].health, 16)

        ally.health -= 1
        engine.players[0].health -= 1
        ally_before = ally.health
        leader_before = engine.players[0].health
        engine.apply(EndTurn(1))

        self.assertEqual(ally.health, ally_before)
        self.assertEqual(engine.players[0].health, leader_before)

    def test_hanetsuki_artisan_fanfare_and_evolve_each_summon_a_fairy(self):
        engine = self._real_engine()
        artisan_card = self.repo.get(10511120)
        _insert_hand(engine, artisan_card)
        engine.players[0].turns_started = engine.config.evolution_unlock_turn

        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        artisan = next(
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == 10511120
        )
        self.assertEqual(engine.players[1].health, 19)
        self.assertEqual(sum(
            unit.definition.card_id == 90011110
            for unit in engine.players[0].board
            if isinstance(unit, Unit)
        ), 1)

        engine.apply(Evolve(0, artisan.entity_id))

        self.assertEqual(engine.players[1].health, 18)
        self.assertEqual(sum(
            unit.definition.card_id == 90011110
            for unit in engine.players[0].board
            if isinstance(unit, Unit)
        ), 2)

    def test_magic_student_heals_once_for_each_hand_that_enters(self):
        engine = self._real_engine()
        student_card = self.repo.get(10632110)
        _insert_hand(engine, student_card)
        engine.players[0].health = 15

        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))

        self.assertEqual(engine.players[0].health, 17)
        self.assertEqual(sum(
            unit.definition.card_id == 10631110
            for unit in engine.players[0].board
            if isinstance(unit, Unit)
        ), 2)

    def test_batch_cards_are_covered_exact(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in (10311120, 10402110, 10511120, 10632110):
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertIn("listener:board", info["reason"])


if __name__ == "__main__":
    unittest.main()
