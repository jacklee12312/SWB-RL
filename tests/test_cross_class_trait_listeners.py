from __future__ import annotations

import os
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Evolve, PlayCard
from swb.engine.events import EventType, GameEvent
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


def _card(card_id: int, **overrides) -> CardDefinition:
    values = {
        "card_id": card_id,
        "card_set_id": 10000,
        "class_id": 0,
        "class_name": "中立",
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
class CrossClassTraitListenerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def _engine(self) -> GameEngine:
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

    @staticmethod
    def _insert_hand(engine: GameEngine, definition: CardDefinition) -> None:
        card = HandCard(
            definition=definition,
            entity_id=engine.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        engine.players[0].hand.append(card)
        engine.players[0].hand_entity_ids.append(card.entity_id)

    def _play(self, engine: GameEngine, card_id: int) -> Unit:
        self._insert_hand(engine, self.repo.get(card_id))
        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        return next(
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == card_id
        )

    @staticmethod
    def _emit_summoned(engine: GameEngine, definition: CardDefinition) -> Unit:
        unit = engine._summon_follower_to_board(
            0,
            definition,
            summon_cause="test_setup",
        )
        assert unit is not None
        engine._emit(GameEvent(
            EventType.FOLLOWER_SUMMONED,
            0,
            source_id=unit.entity_id,
            metadata={"source": unit, "card_id": definition.card_id},
        ))
        engine._resolve_event_queue()
        engine._stabilize()
        return unit

    def test_eral_summons_a_storm_bat_and_damages_own_leader(self):
        engine = self._engine()
        eral = self._play(engine, 10151110)
        bat = next(
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == 90051120
        )

        self.assertEqual(engine.players[0].health, 19)
        self.assertTrue(bat.has_keyword("疾驰"))
        self.assertTrue(bat.can_attack_leader)
        self.assertFalse(eral.has_keyword("疾驰"))

    def test_waterdrop_drummer_heals_for_any_ocean_trait_follower(self):
        engine = self._engine()
        engine.players[0].health = 15
        self._play(engine, 10541120)

        self.assertEqual(engine.players[0].health, 17)
        self.assertTrue(any(
            isinstance(unit, Unit) and unit.definition.card_id == 90041130
            for unit in engine.players[0].board
        ))

        engine.players[0].health = 14
        self._emit_summoned(
            engine,
            _card(400, name="other-ocean", tribe_id=17, tribe_name="海洋"),
        )
        self.assertEqual(engine.players[0].health, 16)

    def test_edge_master_evolve_summons_artifact_and_triggers_heal(self):
        engine = self._engine()
        engine.players[0].health = 15
        master = self._play(engine, 10771110)
        engine.players[0].turns_started = engine.config.evolution_unlock_turn

        engine.apply(Evolve(0, master.entity_id))

        self.assertEqual(engine.players[0].health, 16)
        artifact = next(
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == 90071150
        )
        self.assertEqual((artifact.definition.tribe_id, artifact.definition.tribe_name), (14, "创造物"))

    def test_batch_cards_are_covered_exact(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in (10151110, 10541120, 10771110):
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertIn("listener:board:follower_summoned", info["reason"])


if __name__ == "__main__":
    unittest.main()
