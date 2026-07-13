from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import EndTurn, Evolve, PlayCard
from swb.engine.resolution import GameEngine
from swb.engine.state import HandCard, Unit


def card(card_id: int, *, attack: int = 1) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=1,
        card_type="随从",
        attack=attack,
        life=2,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


class ExactRuleAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def make_engine(self) -> GameEngine:
        game = GameEngine(
            [card(1000 + index) for index in range(40)],
            [card(2000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=29,
            rulebook=self.rulebook,
            card_resolver=self.repo.get,
        )
        game.reset(seed=29)
        return game

    def test_10061110_heals_five_and_retains_ward(self):
        game = self.make_engine()
        definition = self.repo.get(10061110)
        game.players[0].hand = [definition]
        game.players[0].hand_entity_ids = []
        game.players[0].mana = 10
        game.players[0].health = 12

        game.apply(PlayCard(0, 0))

        self.assertEqual(game.players[0].health, 17)
        self.assertTrue(game.players[0].board[0].has_guard)

    def test_10161130_draws_two_and_heals_two(self):
        game = self.make_engine()
        definition = self.repo.get(10161130)
        game.players[0].hand = [definition]
        game.players[0].hand_entity_ids = []
        game.players[0].mana = 10
        game.players[0].health = 15
        deck_before = len(game.players[0].deck)

        game.apply(PlayCard(0, 0))

        self.assertEqual(game.players[0].health, 17)
        self.assertEqual(len(game.players[0].deck), deck_before - 2)

    def test_10431120_turn_end_spellboosts_all_hand_by_source_attack(self):
        game = self.make_engine()
        source = Unit.summon(
            card(10431120, attack=3),
            entity_id=game.state.allocate_entity_id(),
        )
        game.players[0].board = [source]
        game.players[0].hand = [
            HandCard(
                definition=card(900 + index),
                entity_id=game.state.allocate_entity_id(),
            )
            for index in range(2)
        ]
        game.players[0].hand_entity_ids = [
            hand.entity_id for hand in game.players[0].hand
        ]

        game.apply(EndTurn(0))

        self.assertEqual(
            [hand.spellboost_count for hand in game.players[0].hand],
            [3, 3],
        )

    def test_10431120_evolve_permanently_prevents_attacking(self):
        game = self.make_engine()
        definition = self.repo.get(10431120)
        source = Unit.summon(
            definition,
            entity_id=game.state.allocate_entity_id(),
        )
        source.summoned_this_turn = False
        source.can_attack = True
        game.players[0].board = [source]
        game.players[0].turns_started = game.config.evolution_unlock_turn

        game.apply(Evolve(0, source.entity_id))

        self.assertTrue(source.evolved)
        self.assertFalse(source.can_attack_units)
        self.assertFalse(source.can_attack_leader)


if __name__ == "__main__":
    unittest.main()
