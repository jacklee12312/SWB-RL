from __future__ import annotations

import random
from typing import Sequence

from swb.engine.environment import ShadowverseEnv
from swb.engine.state import Phase

MULLIGAN_POLICY_RANDOM = "random"
MULLIGAN_POLICY_CURVE = "curve"
MULLIGAN_POLICY_VALUES = frozenset({
    MULLIGAN_POLICY_RANDOM,
    MULLIGAN_POLICY_CURVE,
})


def select_baseline_action(
    env: ShadowverseEnv,
    action_mask: Sequence[bool],
    rng: random.Random,
    *,
    mulligan_policy: str = MULLIGAN_POLICY_RANDOM,
    curve_keep_cost: int = 3,
) -> int:
    """Select a legal baseline action with an optional opening-hand policy.

    The curve policy replaces opening cards whose printed cost is above the
    configured threshold. Outside mulligan both policies remain uniform random
    legal-action baselines, so they are not policy-strength claims.
    """

    if mulligan_policy not in MULLIGAN_POLICY_VALUES:
        raise ValueError(
            f"unknown mulligan policy {mulligan_policy!r}; "
            f"expected one of {sorted(MULLIGAN_POLICY_VALUES)}"
        )
    if (
        not isinstance(curve_keep_cost, int)
        or isinstance(curve_keep_cost, bool)
        or curve_keep_cost < 0
    ):
        raise ValueError("curve_keep_cost must be a non-negative integer")

    legal = [index for index, allowed in enumerate(action_mask) if allowed]
    if not legal:
        raise RuntimeError("baseline policy received an empty legal-action mask")

    if (
        mulligan_policy == MULLIGAN_POLICY_CURVE
        and env._core.state.phase is Phase.MULLIGAN
    ):
        hand = env._core.players[env.decision_player].hand
        replace_mask = sum(
            1 << index
            for index, card in enumerate(hand[: env.STARTING_HAND])
            if card.definition.cost > curve_keep_cost
        )
        action = env.CHOICE_OFFSET + replace_mask
        if action not in legal:
            raise RuntimeError(
                "curve mulligan action disagrees with the executable action mask"
            )
        return action

    return legal[rng.randrange(len(legal))]
