from __future__ import annotations

import unittest
from dataclasses import replace

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard
from swb.engine.resolution import GameEngine
from swb.engine.state import DeathCause, DestroyedFollowerRecord, HandCard, Unit


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

    def test_10041130_fanfare_damages_enemy_leader_for_six(self):
        game = self.make_engine()
        game.players[0].hand = [self.repo.get(10041130)]
        game.players[0].hand_entity_ids = []
        game.players[0].mana = 10

        game.apply(PlayCard(0, 0))

        self.assertEqual(game.players[1].health, 14)

    def test_10051120_fanfare_damages_own_leader_for_one(self):
        game = self.make_engine()
        game.players[0].hand = [self.repo.get(10051120)]
        game.players[0].hand_entity_ids = []
        game.players[0].mana = 10

        game.apply(PlayCard(0, 0))

        self.assertEqual(game.players[0].health, 19)

    def test_10132320_sets_health_and_restricts_through_opponent_turn(self):
        game = self.make_engine()
        game.players[0].hand = [self.repo.get(10132320)]
        game.players[0].hand_entity_ids = []
        game.players[0].mana = 10
        target = Unit.summon(
            card(800, attack=4),
            entity_id=game.state.allocate_entity_id(),
        )
        target.health = 6
        target.max_health = 6
        target.base_health = 6
        game.players[1].board = [target]

        game.apply(PlayCard(0, 0))
        game.apply(next(
            command
            for command in game.legal_commands()
            if isinstance(command, Choose)
        ))

        self.assertEqual((target.health, target.max_health), (1, 1))
        self.assertFalse(target.can_attack_units)
        self.assertFalse(target.can_attack_leader)
        game.apply(EndTurn(0))
        self.assertFalse(target.can_attack_units)
        game.apply(EndTurn(1))
        self.assertTrue(target.can_attack_units)

    def test_10551120_fanfare_reanimates_cost_two_follower(self):
        game = self.make_engine()
        game.players[0].hand = [self.repo.get(10551120)]
        game.players[0].hand_entity_ids = []
        game.players[0].mana = 10
        buried = replace(card(801), cost=2)
        game.state.destroyed_followers.append(DestroyedFollowerRecord(
            definition=buried,
            owner=0,
            death_sequence=1,
            cause=DeathCause.EFFECT_DESTROY,
        ))

        game.apply(PlayCard(0, 0))

        self.assertTrue(any(
            unit.definition.card_id == buried.card_id
            for unit in game.players[0].board
        ))

    def test_10642310_discards_then_destroys_selected_enemy(self):
        game = self.make_engine()
        spell = self.repo.get(10642310)
        material = card(802)
        game.players[0].hand = [spell, material]
        game.players[0].hand_entity_ids = []
        game.players[0].mana = 10
        target = Unit.summon(
            card(803),
            entity_id=game.state.allocate_entity_id(),
        )
        game.players[1].board = [target]

        game.apply(PlayCard(0, 0))
        hand_choice = next(
            command
            for command in game.legal_commands()
            if isinstance(command, Choose)
        )
        game.apply(hand_choice)
        board_choice = next(
            command
            for command in game.legal_commands()
            if isinstance(command, Choose)
        )
        game.apply(board_choice)

        self.assertFalse(any(
            hand.definition.card_id == material.card_id
            for hand in game.players[0].hand
        ))
        self.assertNotIn(target, game.players[1].board)


if __name__ == "__main__":
    unittest.main()
