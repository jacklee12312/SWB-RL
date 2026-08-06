from __future__ import annotations

import random
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from swb.db.repository import CardDefinition
from swb.engine.environment import MATCH_SETUP_OFFICIAL, ShadowverseEnv


class _LegalDiscrete(spaces.Discrete):
    """Discrete space whose checker samples respect the current legal mask."""

    def __init__(self, n: int, mask_provider: Callable[[], Sequence[bool]]):
        super().__init__(n)
        self._mask_provider = mask_provider

    def sample(self, mask=None, probability=None):
        if mask is None:
            mask = np.asarray(self._mask_provider(), dtype=np.int8)
        return super().sample(mask=mask, probability=probability)


class SWBGymEnv(gym.Env):
    """One learning player against a deterministic pluggable built-in opponent."""

    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(
        self,
        deck_a: Sequence[CardDefinition],
        deck_b: Sequence[CardDefinition],
        *,
        class_a: int,
        class_b: int,
        card_vocabulary: Sequence[int],
        learner_player: int = 0,
        opponent_policy: str | Callable[[ShadowverseEnv, Sequence[bool]], int] = (
            "random_legal"
        ),
        render_mode: str | None = None,
        **environment_kwargs: Any,
    ) -> None:
        if learner_player not in (0, 1):
            raise ValueError("learner_player must be 0 or 1")
        if render_mode is not None:
            raise ValueError("SWBGymEnv does not provide a render mode")
        if (
            not callable(opponent_policy)
            and opponent_policy not in {"random_legal", "fixed_first_legal"}
        ):
            raise ValueError("unknown built-in opponent policy")
        observation_version = environment_kwargs.setdefault(
            "observation_version", "v4"
        )
        if observation_version not in {"v3", "v4", "v4.1"}:
            raise ValueError(
                "SWBGymEnv requires observation_version='v3', 'v4', "
                "or 'v4.1'"
            )
        environment_kwargs["card_vocabulary"] = card_vocabulary
        environment_kwargs.setdefault("match_setup", MATCH_SETUP_OFFICIAL)
        self.engine_env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=class_a,
            class_b=class_b,
            **environment_kwargs,
        )
        self.learner_player = learner_player
        self.opponent_policy = opponent_policy
        self.render_mode = render_mode
        if observation_version == "v4.1":
            self.observation_space = (
                self.engine_env.observation_v4_1_space()
            )
        elif observation_version == "v4":
            self.observation_space = self.engine_env.observation_v4_space()
        else:
            self.observation_space = self.engine_env.observation_v3_space()
        self.action_space = _LegalDiscrete(
            self.engine_env.ACTION_SIZE,
            self._sample_mask,
        )
        self._opponent_rng = random.Random()
        self._finished = True
        self._last_info: dict[str, object] = {}

    def _sample_mask(self) -> Sequence[bool]:
        if self._finished:
            mask = np.zeros(self.engine_env.ACTION_SIZE, dtype=np.int8)
            if self.engine_env.enable_mulligan:
                mask[
                    self.engine_env.CHOICE_OFFSET:
                    self.engine_env.CHOICE_OFFSET
                    + self.engine_env.MAX_CHOICE_OPTIONS
                ] = 1
            else:
                mask[self.engine_env.END_TURN] = 1
            return mask
        return self._last_info.get(
            "action_mask", self.engine_env.action_mask()
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        super().reset(seed=seed)
        del options
        self._opponent_rng.seed(seed)
        _, info = self.engine_env.reset(seed=seed)
        self._finished = False
        info = self._advance_opponent(info)
        self._last_info = info
        return self._learner_observation(info), dict(info)

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        if self._finished:
            raise RuntimeError("Cannot step a finished Gym environment; call reset()")
        if not isinstance(action, (int, np.integer)) or isinstance(action, bool):
            raise TypeError("action must be an integer")
        action = int(action)
        if action < 0 or action >= self.engine_env.ACTION_SIZE:
            raise ValueError(f"action {action} is outside the action space")
        if self.engine_env.decision_player != self.learner_player:
            raise RuntimeError("step called while the built-in opponent owns the decision")
        mask = self._last_info["action_mask"]
        if not mask[action]:
            raise ValueError(f"illegal masked action {action}")

        result = self.engine_env.step(action)
        info = result.info
        if not result.terminated and not result.truncated:
            info = self._advance_opponent(info)
        terminated = self.engine_env.terminated
        truncated = self.engine_env.truncated
        self._finished = terminated or truncated
        reward = (
            0.0
            if not terminated or self.engine_env.winner is None
            else (1.0 if self.engine_env.winner == self.learner_player else -1.0)
        )
        self._last_info = info
        return (
            self._learner_observation(info),
            reward,
            terminated,
            truncated,
            dict(info),
        )

    def _advance_opponent(self, info: dict[str, object]) -> dict[str, object]:
        while (
            not self.engine_env.terminated
            and not self.engine_env.truncated
            and self.engine_env.decision_player != self.learner_player
        ):
            mask = info["action_mask"]
            if callable(self.opponent_policy):
                action = int(self.opponent_policy(self.engine_env, mask))
            else:
                legal = [index for index, allowed in enumerate(mask) if allowed]
                if self.opponent_policy == "fixed_first_legal":
                    action = legal[0]
                else:
                    action = self._opponent_rng.choice(legal)
            if action < 0 or action >= self.engine_env.ACTION_SIZE or not mask[action]:
                raise RuntimeError("built-in opponent selected an illegal action")
            info = self.engine_env.step(action).info
        return info

    def _learner_observation(
        self,
        info: dict[str, object],
    ) -> dict[str, np.ndarray]:
        mask = (
            info["action_mask"]
            if not self._finished
            and self.engine_env.decision_player == self.learner_player
            else [False] * self.engine_env.ACTION_SIZE
        )
        return self.engine_env.observation(
            perspective=self.learner_player,
            action_mask=mask,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
