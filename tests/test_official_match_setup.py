from __future__ import annotations

import unittest
from dataclasses import replace

from swb.db.repository import CardDefinition
from swb.engine.commands import Choose, EndTurn, Evolve, UseExtraPP
from swb.engine.emblem import EmblemDefinition
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.faith import FaithDefinition, FaithInstance
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import Phase, Unit


def card(card_id: int) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"setup-{card_id}",
        cost=1,
        card_type="随从",
        attack=1,
        life=1,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def deck(offset: int = 0) -> list[CardDefinition]:
    return [card(offset + index + 1) for index in range(40)]


def engine(*, seed: int = 7, **config_kwargs) -> GameEngine:
    result = GameEngine(
        deck(),
        deck(100),
        class_a=1,
        class_b=1,
        seed=seed,
        config=GameConfig(validate_invariants=True, **config_kwargs),
    )
    result.reset(seed=seed)
    return result


class OfficialMulliganTests(unittest.TestCase):
    def test_four_card_mulligan_uses_all_subsets_and_excludes_same_instances(self):
        game = engine(enable_mulligan=True, starting_player=0)
        self.assertIs(game.state.phase, Phase.MULLIGAN)
        self.assertEqual(game.state.pending_choice.player_index, 0)
        self.assertEqual(len(game.legal_commands()), 16)

        original = tuple(game.players[0].hand)
        game.apply(Choose(0, "mulligan:5"))
        self.assertEqual(game.state.pending_choice.player_index, 1)
        self.assertEqual(len(game.players[0].hand), 4)
        self.assertIs(game.players[0].hand[1], original[1])
        self.assertIs(game.players[0].hand[3], original[3])
        self.assertNotIn(original[0].entity_id, game.players[0].hand_entity_ids)
        self.assertNotIn(original[2].entity_id, game.players[0].hand_entity_ids)

        game.apply(Choose(1, "mulligan:0"))
        self.assertIs(game.state.phase, Phase.MAIN)
        self.assertEqual(game.current_player, 0)
        self.assertEqual(game.state.mulligan_completed, [True, True])
        self.assertEqual((len(game.players[0].hand), len(game.players[1].hand)), (5, 4))

    def test_mulligan_can_draw_another_copy_with_the_same_name(self):
        game = engine(enable_mulligan=True, starting_player=0)
        original = game.players[0].hand[0]
        same_name_copy = replace(card(8001), name=original.name)
        game.players[0].deck[-1] = same_name_copy

        game.apply(Choose(0, "mulligan:1"))

        self.assertEqual(game.players[0].hand[0].name, original.name)
        self.assertNotEqual(game.players[0].hand[0].entity_id, original.entity_id)

    def test_mulligan_is_seed_reproducible_and_blocks_non_choice_commands(self):
        first = engine(enable_mulligan=True, starting_player=0)
        before = first.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            first.apply(EndTurn(0))
        self.assertEqual(first.deterministic_fingerprint(), before)

        second = engine(enable_mulligan=True, starting_player=0)
        for game in (first, second):
            game.apply(Choose(0, "mulligan:15"))
            game.apply(Choose(1, "mulligan:3"))
        self.assertEqual(
            first.deterministic_fingerprint(),
            second.deterministic_fingerprint(),
        )


class OfficialTurnOrderAndExtraPPTests(unittest.TestCase):
    def test_seeded_random_starting_player_is_reproducible(self):
        first = engine(seed=19, starting_player=None)
        second = engine(seed=19, starting_player=None)
        self.assertEqual(first.state.first_player, second.state.first_player)
        self.assertEqual(first.current_player, first.state.first_player)
        seen = {
            engine(seed=seed, starting_player=None).state.first_player
            for seed in range(10)
        }
        self.assertEqual(seen, {0, 1})

    def test_first_evolves_on_fifth_turn_and_second_on_fourth(self):
        game = engine(starting_player=0)
        first_unit = Unit.summon(card(9001), entity_id=game.state.allocate_entity_id())
        second_unit = Unit.summon(card(9002), entity_id=game.state.allocate_entity_id())
        game.players[0].board = [first_unit]
        game.players[1].board = [second_unit]
        game.players[0].turns_started = 4
        game.players[1].turns_started = 4

        game.state.active_player = 0
        self.assertFalse(any(isinstance(command, Evolve) for command in game.legal_commands()))
        game.players[0].turns_started = 5
        self.assertTrue(any(isinstance(command, Evolve) for command in game.legal_commands()))

        game.state.active_player = 1
        self.assertTrue(any(isinstance(command, Evolve) for command in game.legal_commands()))

    def test_second_player_extra_pp_refreshes_once_on_sixth_turn(self):
        game = engine(starting_player=0)
        game.apply(EndTurn(0))
        second = game.players[1]
        self.assertEqual((second.max_mana, second.mana), (1, 1))
        transition = game.apply(UseExtraPP(1))
        self.assertEqual((second.max_mana, second.mana), (1, 2))
        self.assertEqual(second.extra_pp_uses, 1)
        self.assertIn(EventType.EXTRA_PP_USED, [event.type for event in transition.events])
        before = game.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            game.apply(UseExtraPP(1))
        self.assertEqual(game.deterministic_fingerprint(), before)

        while second.turns_started < 6:
            game.apply(EndTurn(game.current_player))
        self.assertEqual(game.current_player, 1)
        self.assertTrue(second.extra_pp_available)
        self.assertTrue(second.extra_pp_refresh_done)
        game.apply(UseExtraPP(1))
        self.assertEqual(second.extra_pp_uses, 2)
        self.assertFalse(second.extra_pp_available)


class OfficialCapacityAndRLTests(unittest.TestCase):
    def test_faith_and_emblems_share_five_leader_area_slots(self):
        game = engine(starting_player=0)
        faith_definition = FaithDefinition("setup-faith", 9900)
        game.players[0].faiths.append(
            FaithInstance(
                definition=faith_definition,
                entity_id=game.state.allocate_entity_id(),
                controller=0,
                created_sequence=1,
            )
        )
        game.players[0]._next_faith_sequence = 2
        for index in range(5):
            definition = EmblemDefinition(f"setup-emblem-{index}", 9910 + index)
            game._add_emblem_to_player(0, definition, definition.source_card_id)
        self.assertEqual(len(game.players[0].faiths), 1)
        self.assertEqual(len(game.players[0].emblems), 4)
        game.assert_invariants()

    def test_rl_reuses_choice_slots_for_mulligan_and_appends_extra_pp_action(self):
        env = ShadowverseEnv(
            deck(),
            deck(100),
            class_a=1,
            class_b=1,
            seed=11,
            starting_player=0,
            enable_mulligan=True,
            validate_invariants=True,
        )
        observation, info = env.reset(seed=11)
        self.assertEqual((env.ACTION_SIZE, len(observation)), (112, 304))
        self.assertEqual(
            [index for index, legal in enumerate(info["action_mask"]) if legal],
            list(range(env.CHOICE_OFFSET, env.CHOICE_OFFSET + 16)),
        )
        env.step(env.CHOICE_OFFSET)
        env.step(env.CHOICE_OFFSET)
        env.step(env.END_TURN)
        self.assertTrue(env.action_mask()[env.USE_EXTRA_PP])
        result = env.step(env.USE_EXTRA_PP)
        self.assertEqual(result.info["extra_pp"][1]["uses"], 1)


if __name__ == "__main__":
    unittest.main()
