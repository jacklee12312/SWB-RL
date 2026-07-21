from __future__ import annotations

import unittest

import numpy as np

from swb.db.repository import CardDefinition
from swb.engine.environment import ShadowverseEnv


def card(card_id: int, *, cost: int = 1) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=cost,
        card_type="随从",
        attack=1,
        life=2,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


class ObservationV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.deck_a = [card(100 + index) for index in range(40)]
        self.deck_b = [card(200 + index) for index in range(40)]
        self.vocabulary = tuple(range(100, 240))

    def make_env(self, **kwargs) -> ShadowverseEnv:
        env = ShadowverseEnv(
            self.deck_a,
            self.deck_b,
            class_a=1,
            class_b=1,
            seed=42,
            observation_version="v3",
            card_vocabulary=self.vocabulary,
            **kwargs,
        )
        env.reset(seed=42)
        return env

    def test_observation_is_numpy_only_and_matches_gymnasium_space(self) -> None:
        env = self.make_env()
        observation = env.observation()
        self.assertTrue(all(isinstance(value, np.ndarray) for value in observation.values()))
        self.assertTrue(env.observation_v3_space().contains(observation))
        self.assertEqual(observation["continuous"].dtype, np.float32)
        self.assertEqual(observation["action_mask"].dtype, np.int8)

    def test_closed_decklists_hide_opponent_composition(self) -> None:
        env = self.make_env()
        observation = env.observation()
        self.assertEqual(int(observation["own_initial_deck"].sum()), 40)
        self.assertEqual(int(observation["opponent_initial_deck"].sum()), 0)

        opponent = env.players[1]
        opponent.hand[0].definition = card(239, cost=9)
        opponent.deck.reverse()
        changed = env.observation()
        for key in observation:
            np.testing.assert_array_equal(observation[key], changed[key])

    def test_open_decklist_mode_is_explicit(self) -> None:
        env = self.make_env(open_decklists=True)
        self.assertEqual(int(env.observation()["opponent_initial_deck"].sum()), 40)

    def test_non_decision_perspective_gets_no_legal_action_mask(self) -> None:
        env = self.make_env()
        other = 1 - env.decision_player
        observation = env.observation(perspective=other)
        self.assertFalse(observation["action_mask"].any())
        self.assertFalse(observation["choice_categorical"].any())
        self.assertTrue(env.observation()["action_mask"].any())


if __name__ == "__main__":
    unittest.main()
