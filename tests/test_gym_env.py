from __future__ import annotations

import unittest

from gymnasium.utils.env_checker import check_env

from swb.db.repository import CardDefinition
from swb.rl.gym_env import SWBGymEnv


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


class GymEnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs) -> SWBGymEnv:
        kwargs.setdefault("max_agent_steps", 20)
        return SWBGymEnv(
            [card(100 + index) for index in range(40)],
            [card(200 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            card_vocabulary=tuple((*range(100, 140), *range(200, 240))),
            seed=3,
            **kwargs,
        )

    def test_passes_official_gymnasium_checker(self) -> None:
        check_env(self.make_env(), skip_render_check=True)

    def test_second_player_wrapper_advances_builtin_opponent(self) -> None:
        env = self.make_env(
            learner_player=1,
            opponent_policy="fixed_first_legal",
        )
        observation, info = env.reset(seed=7)
        self.assertEqual(env.engine_env.decision_player, 1)
        self.assertTrue(observation["action_mask"].any())
        action = next(index for index, legal in enumerate(info["action_mask"]) if legal)
        _, _, terminated, truncated, _ = env.step(action)
        self.assertFalse(terminated and truncated)

    def test_bounds_mask_order_and_dead_step_are_guarded(self) -> None:
        env = self.make_env(max_agent_steps=1)
        with self.assertRaisesRegex(RuntimeError, "finished"):
            env.step(0)
        _, info = env.reset(seed=3)
        with self.assertRaisesRegex(ValueError, "outside"):
            env.step(env.engine_env.ACTION_SIZE)
        illegal = next(
            index for index, legal in enumerate(info["action_mask"]) if not legal
        )
        with self.assertRaisesRegex(ValueError, "illegal masked"):
            env.step(illegal)
        legal = next(index for index, allowed in enumerate(info["action_mask"]) if allowed)
        _, _, _, truncated, _ = env.step(legal)
        self.assertTrue(truncated)
        with self.assertRaisesRegex(RuntimeError, "finished"):
            env.step(legal)

    def test_bad_pluggable_opponent_action_is_rejected(self) -> None:
        env = self.make_env(
            learner_player=1,
            opponent_policy=lambda engine, mask: engine.ACTION_SIZE,
        )
        with self.assertRaisesRegex(RuntimeError, "opponent selected an illegal"):
            env.reset(seed=3)


if __name__ == "__main__":
    unittest.main()
