from __future__ import annotations

import os
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import RuleBook
from swb.engine.commands import EndTurn, Evolve, PlayCard
from swb.engine.events import EventType, GameEvent
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


def _card(card_id: int, **overrides) -> CardDefinition:
    values = {
        "card_id": card_id,
        "card_set_id": 10000,
        "class_id": 2,
        "class_name": "皇家护卫",
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


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RoyalTraitListenerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def _engine(self) -> GameEngine:
        engine = GameEngine(
            [_card(1000 + index) for index in range(40)],
            [_card(2000 + index) for index in range(40)],
            class_a=2,
            class_b=2,
            seed=42,
            rulebook=self.rulebook,
            card_resolver=self.repo.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=42)
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10
        return engine

    @staticmethod
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

    @staticmethod
    def _insert_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
        card = HandCard(
            definition=definition,
            entity_id=engine.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        engine.players[0].hand.append(card)
        engine.players[0].hand_entity_ids.append(card.entity_id)
        return card

    def _play_real(self, engine: GameEngine, card_id: int) -> Unit:
        self._insert_hand(engine, self.repo.get(card_id))
        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        return next(
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == card_id
        )

    def _emit_soldier_during_current_turn(self, engine: GameEngine) -> Unit:
        soldier = self._place(engine, 0, self.repo.get(90021110))
        engine._emit(GameEvent(
            EventType.FOLLOWER_SUMMONED,
            0,
            source_id=soldier.entity_id,
            metadata={"source": soldier, "card_id": soldier.definition.card_id},
        ))
        engine._resolve_event_queue()
        engine._stabilize()
        return soldier

    def test_luminous_knight_opponent_turn_buff_expires_that_turn(self):
        engine = self._engine()
        knight = self._place(engine, 0, self.repo.get(10122110))
        base_attack = knight.attack
        engine.apply(EndTurn(0))
        self.assertEqual(engine.current_player, 1)

        self._emit_soldier_during_current_turn(engine)

        self.assertEqual(knight.attack, base_attack + 1)
        self.assertEqual(len(knight.stat_modifiers), 1)
        self.assertEqual(knight.stat_modifiers[0].expires_for_player, 1)
        engine.apply(EndTurn(1))
        self.assertEqual(knight.attack, base_attack)

    def test_luminous_knight_evolve_summons_soldier_and_triggers_self_buff(self):
        engine = self._engine()
        knight = self._play_real(engine, 10122110)
        base_attack = knight.attack
        engine.players[0].turns_started = engine.config.evolution_unlock_turn

        engine.apply(Evolve(0, knight.entity_id))

        self.assertEqual(knight.attack, base_attack + 3)
        soldiers = [
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == 90021110
        ]
        self.assertEqual(len(soldiers), 1)
        self.assertFalse(any(
            event.card_id == 10122110
            for event in engine.placeholder_ability_events
        ))

    def test_luminous_mage_summons_three_guard_soldiers(self):
        engine = self._engine()
        mage = self._play_real(engine, 10122120)
        soldiers = [
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == 90021120
        ]

        self.assertEqual(len(soldiers), 3)
        self.assertTrue(all(unit.has_guard for unit in soldiers))
        self.assertFalse(mage.has_guard)

    def test_luminous_lancer_summons_one_rush_soldier(self):
        engine = self._engine()
        lancer = self._play_real(engine, 10122130)
        soldier = next(
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == 90021110
        )

        self.assertTrue(soldier.has_keyword("突进"))
        self.assertTrue(soldier.can_attack_units)
        self.assertFalse(soldier.can_attack_leader)
        self.assertFalse(lancer.has_keyword("突进"))

    def test_amalia_buffs_only_the_four_followers_she_summons(self):
        engine = self._engine()
        amalia = self._play_real(engine, 10123140)
        soldiers = [
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == 90021120
        ]

        self.assertEqual(len(soldiers), 4)
        for soldier in soldiers:
            self.assertEqual((soldier.attack, soldier.health), (3, 2))
            self.assertTrue(soldier.has_keyword("突进"))
            self.assertTrue(soldier.has_guard)
        self.assertEqual((amalia.attack, amalia.health), (6, 6))
        self.assertFalse(amalia.has_keyword("突进"))
        self.assertFalse(amalia.has_guard)

    def test_batch_cards_are_covered_exact_without_keyword_placeholders(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in (10122110, 10122120, 10122130, 10123140):
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertIn("listener:board:follower_summoned", info["reason"])

        engine = self._engine()
        for card_id in (10122120, 10122130, 10123140):
            unit = self._place(engine, 0, self.repo.get(card_id))
            expected_absent = {
                10122120: (AbilityKeyword.WARD,),
                10122130: (AbilityKeyword.RUSH,),
                10123140: (AbilityKeyword.RUSH, AbilityKeyword.WARD),
            }[card_id]
            self.assertTrue(all(
                not unit.has_keyword(keyword.value)
                for keyword in expected_absent
            ))


if __name__ == "__main__":
    unittest.main()
