from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from swb.engine.commands import BeginFusion, ChoiceKind, Choose
from swb.engine.environment import ShadowverseEnv


@dataclass
class FusionCancelActionGuard:
    """Apply policy-only guards for provably non-progressing actions.

    The engine mask remains the source of truth for game-rule legality.  This
    guard derives a policy-only mask so human clients and the deterministic
    rules engine retain the legal ability to cancel/reconsider fusion and to
    activate refundable Extra PP manually.  Policies additionally suppress an
    immediate fusion retry and an Extra PP activation that unlocks no payment.

    Call :meth:`record_selected_action` immediately before executing an action
    that has already been validated against :meth:`policy_mask`.
    """

    _blocked_players: set[int] = field(default_factory=set)
    suppressed_decisions: int = 0
    suppressed_actions: int = 0
    extra_pp_suppressed_decisions: int = 0
    extra_pp_suppressed_actions: int = 0

    def reset(self) -> None:
        """Clear episode-local blocking state while retaining audit counters."""

        self._blocked_players.clear()

    def policy_mask(
        self,
        env: ShadowverseEnv,
        player_id: int,
        legal_mask: Sequence[bool] | np.ndarray,
    ) -> np.ndarray:
        """Return the effective sampling mask without mutating ``legal_mask``."""

        mask = np.asarray(legal_mask, dtype=np.bool_).copy()
        suppressed = 0
        if player_id in self._blocked_players:
            for raw_action in np.flatnonzero(mask):
                action = int(raw_action)
                if not env.MODE_PLAY_OFFSET <= action < env.SUPER_EVOLVE_OFFSET:
                    continue
                if isinstance(env._decode_action(action), BeginFusion):
                    mask[action] = False
                    suppressed += 1

        if suppressed:
            self.suppressed_decisions += 1
            self.suppressed_actions += suppressed

        extra_pp_action = getattr(env, "USE_EXTRA_PP", None)
        if (
            isinstance(extra_pp_action, int)
            and 0 <= extra_pp_action < mask.size
            and mask[extra_pp_action]
            and not env.core.extra_pp_unlocked_payment_commands(player_id)
        ):
            mask[extra_pp_action] = False
            self.extra_pp_suppressed_decisions += 1
            self.extra_pp_suppressed_actions += 1
        if not bool(mask.any()):
            raise RuntimeError(
                "policy action guard removed every legal action"
            )
        return mask

    def record_selected_action(
        self,
        env: ShadowverseEnv,
        player_id: int,
        action: int,
    ) -> None:
        """Update the guard for one policy-selected, engine-legal action."""

        request = env.core.state.pending_choice
        cancelled_fusion = False
        if request is not None and request.choice_kind is ChoiceKind.FUSION:
            command = env._decode_action(action)
            cancelled_fusion = (
                isinstance(command, Choose)
                and command.option_id == "fusion:cancel"
            )

        if cancelled_fusion:
            self._blocked_players.add(player_id)
        elif player_id in self._blocked_players:
            # A guarded decision cannot select BeginFusion.  Once the agent
            # performs any other action, fusion may be considered again later
            # in the same turn if the real rules still allow it.
            self._blocked_players.remove(player_id)

    def is_blocking(self, player_id: int) -> bool:
        return player_id in self._blocked_players
