from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from swb.db.repository import CardDefinition
from swb.engine.environment import ShadowverseEnv


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


class EnvironmentCacheTests(unittest.TestCase):
    def make_env(self, **kwargs) -> ShadowverseEnv:
        deck_a = [card(100 + index) for index in range(40)]
        deck_b = [card(200 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=7,
            observation_version="v3",
            card_vocabulary=tuple(range(100, 240)),
            **kwargs,
        )
        env.reset(seed=7)
        return env

    def test_same_transition_reuses_legality_mask_and_observation(self) -> None:
        env = self.make_env()
        env.invalidate_cache(reason="test cold start")
        with patch.object(
            env._core,
            "legal_commands",
            wraps=env._core.legal_commands,
        ) as legal_commands:
            first_mask = env.action_mask()
            first_observation = env.observation()
            info = env.info()
            second_observation = env.observation()
            second_mask = env.action_mask()

        self.assertEqual(legal_commands.call_count, 1)
        self.assertEqual(first_mask, second_mask)
        self.assertEqual(first_mask, info["action_mask"])
        for key in first_observation:
            np.testing.assert_array_equal(
                first_observation[key], second_observation[key]
            )
        stats = env.cache_stats
        self.assertGreaterEqual(stats.get("action_mask_hits", 0), 2)
        self.assertGreaterEqual(stats.get("observation_hits", 0), 1)

    def test_reset_and_legal_step_advance_versions_and_invalidate(self) -> None:
        env = self.make_env()
        state_before = env.state_version
        transition_before = env.transition_version
        env.step(env.END_TURN)
        self.assertEqual(env.state_version, state_before + 1)
        self.assertGreater(env.transition_version, transition_before)

        state_after_step = env.state_version
        transition_after_step = env.transition_version
        env.reset(seed=8)
        self.assertEqual(env.state_version, state_after_step + 1)
        self.assertGreater(env.transition_version, transition_after_step)

    def test_illegal_action_does_not_advance_versions_or_drop_cache(self) -> None:
        env = self.make_env()
        mask = env.action_mask()
        illegal = next(index for index, legal in enumerate(mask) if not legal)
        versions = (env.state_version, env.transition_version)
        hits_before = env.cache_stats.get("action_mask_hits", 0)
        with self.assertRaisesRegex(ValueError, "Illegal action"):
            env.step(illegal)
        self.assertEqual((env.state_version, env.transition_version), versions)
        self.assertEqual(env.action_mask(), mask)
        self.assertGreater(env.cache_stats.get("action_mask_hits", 0), hits_before)

    def test_mutable_access_is_an_explicit_invalidation_boundary(self) -> None:
        env = self.make_env()
        env.action_mask()
        transition_before = env.transition_version
        player = env.players[env.current_player]
        player.mana = 10
        self.assertGreater(env.transition_version, transition_before)
        misses_before = env.cache_stats.get("action_mask_misses", 0)
        env.action_mask()
        self.assertGreater(env.cache_stats.get("action_mask_misses", 0), misses_before)

    def test_debug_guard_rejects_retained_reference_mutation(self) -> None:
        env = self.make_env(debug_cache_validation=True)
        core = env.core
        env.action_mask()
        core.state.active_player = 1 - core.state.active_player
        with self.assertRaisesRegex(RuntimeError, "invalidate_cache"):
            env.action_mask()

    def test_zone_histograms_are_shared_across_perspectives(self) -> None:
        env = self.make_env()
        env.invalidate_cache(reason="test histogram cold start")
        env.observation(perspective=0)
        hits_before = env.cache_stats.get("histogram_hits", 0)
        env.observation(perspective=1)
        self.assertGreaterEqual(
            env.cache_stats.get("histogram_hits", 0) - hits_before,
            4,
        )

    def test_cached_mask_matches_every_public_legal_command(self) -> None:
        env = self.make_env()
        core = env.core
        commands = core.legal_commands()
        mask = env.action_mask()
        for command in commands:
            action = env._encode_command(command)
            self.assertIsNotNone(action, command)
            self.assertTrue(mask[action], (command, action))


if __name__ == "__main__":
    unittest.main()
