from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.commands import (
    Attack,
    ChoiceOption,
    ChoiceRequest,
    Choose,
    EndTurn,
    PlayCard,
)
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine
from swb.engine.state import Phase, Unit


def card(
    card_id: int,
    *,
    attack: int | None = 1,
    life: int | None = 1,
    cost: int = 1,
    card_type: str = "随从",
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"core-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack,
        life=life,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


class CoreEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        deck = [card(index) for index in range(40)]
        self.engine = GameEngine(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=3,
        )
        self.engine.reset(seed=3)

    def test_commands_produce_events_without_rl_action_numbers(self) -> None:
        self.engine.players[0].mana = 10
        transition = self.engine.apply(PlayCard(0, 0))
        self.assertEqual(
            [event.type for event in transition.events],
            [EventType.CARD_PLAYED, EventType.FOLLOWER_SUMMONED],
        )
        unit = self.engine.players[0].board[0]
        self.assertGreater(unit.entity_id, 0)

    def test_stabilization_moves_destroyed_units_to_graveyards(self) -> None:
        attacker = Unit.summon(card(100, attack=3, life=1))
        defender = Unit.summon(card(101, attack=1, life=2))
        attacker.can_attack = True
        self.engine.players[0].board = [attacker]
        self.engine.players[1].board = [defender]
        commands = self.engine.legal_commands()
        attack = next(
            command
            for command in commands
            if isinstance(command, Attack) and command.target_id is not None
        )
        transition = self.engine.apply(attack)
        self.assertEqual(self.engine.players[0].board, [])
        self.assertEqual(self.engine.players[1].board, [])
        self.assertEqual(len(self.engine.players[0].graveyard), 1)
        self.assertEqual(len(self.engine.players[1].graveyard), 1)
        self.assertEqual(
            sum(
                event.type is EventType.FOLLOWER_DESTROYED
                for event in transition.events
            ),
            2,
        )

    def test_pending_choice_blocks_other_commands(self) -> None:
        self.engine.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="选择模式",
            options=(ChoiceOption("a", "模式A"), ChoiceOption("b", "模式B")),
            continuation_id="test",
        )
        self.engine.state.phase = Phase.AWAITING_CHOICE
        legal = self.engine.legal_commands()
        self.assertEqual(legal, [Choose(0, "a"), Choose(0, "b")])
        self.engine.apply(Choose(0, "a"))
        self.assertIsNone(self.engine.state.pending_choice)
        self.assertEqual(self.engine.state.phase, Phase.MAIN)

    def test_explicit_rulebook_loads_machine_readable_effects(self) -> None:
        rulebook = RuleBook.from_directory("data/rules")
        operations = rulebook.operations_for(10041130, Trigger.FANFARE)
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].amount, 6)

        shark = card(10041130, attack=4, life=3)
        engine = GameEngine(
            [shark] * 40,
            [card(2)] * 40,
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 14)

    def test_targeted_spell_pauses_and_resumes_resolution(self) -> None:
        rulebook = RuleBook.from_directory("data/rules")
        spell = card(
            10041310,
            attack=None,
            life=None,
            card_type="法术",
        )
        engine = GameEngine(
            [spell] * 40,
            [card(2)] * 40,
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(99, attack=1, life=2))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = spell

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        self.assertEqual(len(engine.players[0].graveyard), 0)
        choice = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        engine.apply(choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(engine.players[0].graveyard[-1].card_id, spell.card_id)

    def test_countdown_amulet_expires_and_runs_last_words(self) -> None:
        rulebook = RuleBook.from_directory("data/rules")
        amulet = card(
            10161210,
            attack=None,
            life=None,
            card_type="护符",
        )
        filler = card(2)
        engine = GameEngine(
            [amulet] * 40,
            [filler] * 40,
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = amulet
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].board[0].countdown, 3)
        deck_before = len(engine.players[0].deck)

        for _ in range(6):
            engine.apply(EndTurn(engine.current_player))

        self.assertEqual(engine.players[0].board, [])
        self.assertIn(amulet, engine.players[0].graveyard)
        self.assertLessEqual(len(engine.players[0].deck), deck_before - 5)


if __name__ == "__main__":
    unittest.main()
