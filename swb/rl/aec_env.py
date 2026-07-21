from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from swb.db.repository import CardDefinition
from swb.engine.environment import ShadowverseEnv


class SWBAECEnv(AECEnv):
    """PettingZoo AEC adapter for the alternating-decision SWB environment."""

    metadata = {
        "name": "swb_aec_v0",
        "render_modes": [],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        deck_a: Sequence[CardDefinition],
        deck_b: Sequence[CardDefinition],
        *,
        class_a: int,
        class_b: int,
        card_vocabulary: Sequence[int],
        render_mode: str | None = None,
        **environment_kwargs: Any,
    ) -> None:
        if render_mode is not None:
            raise ValueError("SWBAECEnv does not currently provide a render mode")
        self.render_mode = render_mode
        if environment_kwargs.get("observation_version", "v3") != "v3":
            raise ValueError("SWBAECEnv requires observation_version='v3'")
        environment_kwargs["observation_version"] = "v3"
        environment_kwargs["card_vocabulary"] = card_vocabulary
        self.engine_env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=class_a,
            class_b=class_b,
            **environment_kwargs,
        )
        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {
            agent: index for index, agent in enumerate(self.possible_agents)
        }
        observation_space = self.engine_env.observation_v3_space()
        action_space = spaces.Discrete(self.engine_env.ACTION_SIZE)
        self.observation_spaces = {
            agent: observation_space for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: action_space for agent in self.possible_agents
        }
        self.agents: list[str] = []
        self.agent_selection = self.possible_agents[0]
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, object]] = {}
        self._current_action_mask = [False] * self.engine_env.ACTION_SIZE

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> None:
        del options
        _, base_info = self.engine_env.reset(seed=seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.agent_selection = self._decision_agent()
        self._current_action_mask = list(base_info["action_mask"])
        self.infos = {
            agent: self._agent_info(agent, base_info) for agent in self.agents
        }

    def observe(self, agent: str) -> dict[str, np.ndarray]:
        player_index = self.agent_name_mapping[agent]
        is_current = (
            agent == self.agent_selection
            and not self.terminations.get(agent, False)
            and not self.truncations.get(agent, False)
        )
        action_mask = (
            self._current_action_mask
            if is_current
            else [False] * self.engine_env.ACTION_SIZE
        )
        return self.engine_env.observation(
            perspective=player_index,
            action_mask=action_mask,
        )

    def step(self, action: int | None) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        if action is None:
            raise ValueError("A live agent must provide an action")

        self._cumulative_rewards[agent] = 0.0
        result = self.engine_env.step(int(action))
        self._current_action_mask = list(result.info["action_mask"])
        self._clear_rewards()

        if result.terminated:
            for candidate in self.agents:
                player_index = self.agent_name_mapping[candidate]
                self.rewards[candidate] = (
                    0.0
                    if self.engine_env.winner is None
                    else (1.0 if self.engine_env.winner == player_index else -1.0)
                )
                self.terminations[candidate] = True
            self.agent_selection = agent
        elif result.truncated:
            for candidate in self.agents:
                self.truncations[candidate] = True
            self.agent_selection = agent
        else:
            self.agent_selection = self._decision_agent()

        self.infos = {
            candidate: self._agent_info(candidate, result.info)
            for candidate in self.agents
        }
        self._accumulate_rewards()

    def _decision_agent(self) -> str:
        return self.possible_agents[self.engine_env.decision_player]

    def _agent_info(
        self,
        agent: str,
        base_info: dict[str, object],
    ) -> dict[str, object]:
        info = dict(base_info)
        if agent != self.agent_selection:
            info["action_mask"] = [False] * self.engine_env.ACTION_SIZE
        return info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def raw_env(*args, **kwargs) -> SWBAECEnv:
    return SWBAECEnv(*args, **kwargs)


def env(*args, **kwargs) -> SWBAECEnv:
    return raw_env(*args, **kwargs)
