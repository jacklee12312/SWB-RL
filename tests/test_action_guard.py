from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from swb.engine.commands import (
    BeginFusion,
    ChoiceKind,
    Choose,
    EndTurn,
    PlayCard,
)
from swb.rl.action_guard import FusionCancelActionGuard
from swb.simulator.service import DeterministicPPOPolicy


class _GuardEnvironment:
    MODE_PLAY_OFFSET = 10
    SUPER_EVOLVE_OFFSET = 20
    ACTION_SIZE = 21

    def __init__(self) -> None:
        self.core = SimpleNamespace(
            state=SimpleNamespace(pending_choice=None)
        )
        self.commands = {
            0: EndTurn(0),
            5: Choose(0, "fusion:cancel"),
            6: Choose(0, "hand:123"),
            10: BeginFusion(0, 100),
            11: BeginFusion(0, 101),
            12: PlayCard(0, 0, "enhance"),
        }

    def _decode_action(self, action: int):
        return self.commands[action]

    def pending_choice(self, choice_kind: ChoiceKind) -> None:
        self.core.state.pending_choice = SimpleNamespace(
            choice_kind=choice_kind
        )

    def action_mask(self):
        return _mask(0, 10, 11, 12)

    def observation(self, *, perspective: int, action_mask):
        del perspective, action_mask
        return {}


class _PolicyModel:
    def initial_state(self, batch_size: int, *, device: torch.device):
        return torch.zeros((batch_size, 1), device=device)

    def forward_step(self, vector, hidden, card_indices):
        del vector, card_indices
        logits = torch.zeros((1, _GuardEnvironment.ACTION_SIZE))
        logits[0, 10] = 10.0
        logits[0, 11] = 9.0
        logits[0, 12] = 2.0
        logits[0, 0] = 1.0
        return logits, torch.tensor([[0.25]]), hidden + 1.0

    @staticmethod
    def masked_logits(logits, action_mask):
        return logits.masked_fill(~action_mask.to(dtype=torch.bool), -torch.inf)


class _PolicyFlattener:
    @staticmethod
    def encode(observation):
        del observation
        return np.zeros(1, dtype=np.float32)

    @staticmethod
    def encode_cards(observation):
        del observation
        return np.zeros(1, dtype=np.int64)


def _mask(*actions: int) -> np.ndarray:
    result = np.zeros(_GuardEnvironment.ACTION_SIZE, dtype=np.bool_)
    result[list(actions)] = True
    return result


class FusionCancelActionGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = _GuardEnvironment()
        self.guard = FusionCancelActionGuard()

    def test_fusion_cancel_suppresses_all_immediate_begin_fusion_actions(self):
        self.env.pending_choice(ChoiceKind.FUSION)
        self.guard.record_selected_action(self.env, 0, 5)
        legal_mask = _mask(0, 10, 11, 12)

        policy_mask = self.guard.policy_mask(self.env, 0, legal_mask)

        self.assertTrue(self.guard.is_blocking(0))
        self.assertTrue(policy_mask[0])
        self.assertFalse(policy_mask[10])
        self.assertFalse(policy_mask[11])
        self.assertTrue(policy_mask[12])
        self.assertTrue(legal_mask[10])
        self.assertTrue(legal_mask[11])
        self.assertEqual(self.guard.suppressed_decisions, 1)
        self.assertEqual(self.guard.suppressed_actions, 2)

    def test_one_non_fusion_action_releases_guard_in_same_turn(self):
        self.env.pending_choice(ChoiceKind.FUSION)
        self.guard.record_selected_action(self.env, 0, 5)
        self.env.core.state.pending_choice = None
        self.guard.record_selected_action(self.env, 0, 12)

        policy_mask = self.guard.policy_mask(
            self.env, 0, _mask(0, 10, 11, 12)
        )

        self.assertFalse(self.guard.is_blocking(0))
        self.assertTrue(policy_mask[10])
        self.assertTrue(policy_mask[11])

    def test_material_selection_and_non_fusion_choice_do_not_enable_guard(self):
        self.env.pending_choice(ChoiceKind.FUSION)
        self.guard.record_selected_action(self.env, 0, 6)
        self.assertFalse(self.guard.is_blocking(0))

        self.env.pending_choice(ChoiceKind.GENERIC)
        self.guard.record_selected_action(self.env, 0, 5)
        self.assertFalse(self.guard.is_blocking(0))

    def test_guard_is_per_player_and_reset_clears_episode_state(self):
        self.env.pending_choice(ChoiceKind.FUSION)
        self.guard.record_selected_action(self.env, 0, 5)

        other_mask = self.guard.policy_mask(
            self.env, 1, _mask(0, 10, 11)
        )
        self.assertTrue(other_mask[10])
        self.guard.reset()

        self.assertFalse(self.guard.is_blocking(0))
        reset_mask = self.guard.policy_mask(
            self.env, 0, _mask(0, 10, 11)
        )
        self.assertTrue(reset_mask[10])

    def test_guard_fails_loudly_if_no_progress_action_remains(self):
        self.env.pending_choice(ChoiceKind.FUSION)
        self.guard.record_selected_action(self.env, 0, 5)

        with self.assertRaisesRegex(RuntimeError, "removed every legal action"):
            self.guard.policy_mask(self.env, 0, _mask(10, 11))

    def test_deterministic_policy_keeps_legal_audit_and_uses_guarded_mask(self):
        policy = DeterministicPPOPolicy(
            model=_PolicyModel(),
            flattener=_PolicyFlattener(),
            device=torch.device("cpu"),
            hidden=torch.zeros((1, 1)),
        )
        self.env.pending_choice(ChoiceKind.FUSION)
        policy.fusion_cancel_guard.record_selected_action(self.env, 0, 5)
        self.env.core.state.pending_choice = None

        decision = policy.decision(self.env, 0)

        self.assertEqual(decision.action, 12)
        self.assertEqual(decision.suppressed_actions, (10, 11))
        self.assertEqual(set(decision.probabilities), {0, 10, 11, 12})
        self.assertEqual(decision.probabilities[10], 0.0)
        self.assertEqual(decision.probabilities[11], 0.0)
        self.assertAlmostEqual(sum(decision.probabilities.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
