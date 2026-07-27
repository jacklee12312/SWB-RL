from __future__ import annotations

import unittest

from pettingzoo.test import api_test

from swb.db.repository import CardDefinition
from swb.engine.state import Unit
from swb.rl.aec_env import SWBAECEnv


def card(card_id: int) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=1,
        card_type="随从",
        attack=1,
        life=2,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


class AECEnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs) -> SWBAECEnv:
        return SWBAECEnv(
            [card(1 + index) for index in range(40)],
            [card(101 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            card_vocabulary=tuple((*range(1, 41), *range(101, 141))),
            seed=3,
            max_agent_steps=50,
            **kwargs,
        )

    def test_passes_pettingzoo_api_contract(self) -> None:
        api_test(self.make_env(), num_cycles=60)

    def test_agent_selection_tracks_pending_decision_player(self) -> None:
        env = self.make_env(match_setup="legacy")
        env.reset(seed=3)
        acting = env.agent_selection
        env.step(env.engine_env.END_TURN)
        self.assertNotEqual(env.agent_selection, acting)
        self.assertTrue(env.observe(env.agent_selection)["action_mask"].any())
        self.assertFalse(env.observe(acting)["action_mask"].any())

    def test_terminal_reward_and_done_state_are_per_agent(self) -> None:
        env = self.make_env(match_setup="legacy")
        env.reset(seed=3)
        attacker = Unit.summon(
            card(999),
            entity_id=env.engine_env.core.state.allocate_entity_id(),
        )
        attacker.attack = 20
        attacker.can_attack = True
        env.engine_env.players[0].board = [attacker]

        env.step(env.engine_env.ATTACK_OFFSET)

        self.assertEqual(env.rewards, {"player_0": 1.0, "player_1": -1.0})
        self.assertTrue(all(env.terminations.values()))
        self.assertFalse(any(env.truncations.values()))

    def test_truncation_has_no_synthetic_winner_or_reward(self) -> None:
        env = self.make_env(max_game_turns=1, match_setup="legacy")
        env.reset(seed=3)
        env.step(env.engine_env.END_TURN)
        self.assertTrue(all(env.truncations.values()))
        self.assertFalse(any(env.terminations.values()))
        self.assertEqual(env.rewards, {"player_0": 0.0, "player_1": 0.0})
        self.assertIsNone(env.engine_env.winner)

    def test_adapter_defaults_to_official_match_setup(self) -> None:
        env = self.make_env()
        env.reset(seed=3)
        self.assertEqual(env.engine_env.match_setup, "official")
        self.assertEqual(env.engine_env.info()["phase"], "mulligan")


if __name__ == "__main__":
    unittest.main()
