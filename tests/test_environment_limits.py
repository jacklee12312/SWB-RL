from __future__ import annotations

import unittest
from unittest.mock import patch

from swb.db.repository import CardDefinition
from swb.engine.commands import ChoiceKind, ChoiceOption, ChoiceRequest
from swb.engine.environment import ShadowverseEnv
from swb.engine.state import Unit


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


class EnvironmentLimitTests(unittest.TestCase):
    def make_env(self, **kwargs) -> ShadowverseEnv:
        env = ShadowverseEnv(
            [card(1 + index) for index in range(40)],
            [card(101 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=9,
            **kwargs,
        )
        env.reset(seed=9)
        return env

    def test_agent_step_limit_truncates_without_declaring_winner(self) -> None:
        env = self.make_env(max_agent_steps=1, max_game_turns=None)
        result = env.step(env.END_TURN)
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertTrue(env.truncated)
        self.assertIsNone(env.winner)
        self.assertEqual(env.agent_steps, 1)
        self.assertFalse(any(result.info["action_mask"]))
        with self.assertRaisesRegex(ValueError, "finished environment"):
            env.step(env.END_TURN)

    def test_game_turn_limit_is_a_truncation_not_health_tiebreak(self) -> None:
        env = self.make_env(max_game_turns=1, max_agent_steps=None)
        env.players[0].health = 1
        env.players[1].health = 20
        result = env.step(env.END_TURN)
        self.assertTrue(result.truncated)
        self.assertFalse(result.terminated)
        self.assertIsNone(env.winner)

    def test_rules_terminal_result_takes_precedence_over_step_limit(self) -> None:
        env = self.make_env(max_agent_steps=1, max_game_turns=None)
        attacker = Unit.summon(card(999, attack=20))
        attacker.can_attack = True
        env.players[0].board = [attacker]
        result = env.step(env.ATTACK_OFFSET)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(env.winner, 0)

    def test_page_navigation_counts_toward_agent_step_limit(self) -> None:
        env = self.make_env(max_agent_steps=1, max_game_turns=None)
        env.core.state.pending_choice = ChoiceRequest(
            player_index=0,
            choice_kind=ChoiceKind.GRAVEYARD,
            prompt="graveyard",
            options=tuple(
                ChoiceOption(str(index), str(index)) for index in range(17)
            ),
            continuation_id="test",
        )
        result = env.step(env.GRAVEYARD_NEXT_PAGE)
        self.assertTrue(result.truncated)
        self.assertEqual(env.agent_steps, 1)

    def test_limits_validate_and_reset_cleanly(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_agent_steps"):
            self.make_env(max_agent_steps=0)
        env = self.make_env(max_agent_steps=1)
        env.step(env.END_TURN)
        env.reset(seed=10)
        self.assertFalse(env.truncated)
        self.assertEqual(env.agent_steps, 0)

    def test_action_mask_calls_legal_command_generation_once(self) -> None:
        env = self.make_env()
        with patch.object(
            env.core,
            "legal_commands",
            wraps=env.core.legal_commands,
        ) as legal_commands:
            env.action_mask()
        self.assertEqual(legal_commands.call_count, 1)

    def test_step_reuses_each_next_mask_for_observation_and_info(self) -> None:
        env = self.make_env()
        with patch.object(
            env.core,
            "legal_commands",
            wraps=env.core.legal_commands,
        ) as legal_commands:
            env.step(env.END_TURN)
        self.assertEqual(legal_commands.call_count, 2)


if __name__ == "__main__":
    unittest.main()
