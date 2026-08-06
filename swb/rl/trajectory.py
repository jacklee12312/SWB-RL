from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from swb.rl.seeding import EpisodeSeeds


TRAJECTORY_SCHEMA_VERSION = "trajectory-v1.0"


@dataclass(frozen=True)
class TrajectoryStep:
    episode_id: int
    worker_id: int
    step_index: int
    player_id: int
    observation: Mapping[str, np.ndarray]
    action: int
    action_mask: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    bootstrap_value_allowed: bool
    policy_log_prob: float
    recurrent_state_reset: bool
    opponent_id: str
    versions: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.terminated and self.truncated:
            raise ValueError("trajectory step cannot be terminated and truncated")
        if self.bootstrap_value_allowed != (self.truncated and not self.terminated):
            raise ValueError(
                "bootstrap_value_allowed must be true exactly at truncation boundaries"
            )
        if self.action_mask.dtype != np.int8:
            raise ValueError("trajectory action_mask must have dtype int8")
        if self.action < 0 or self.action >= self.action_mask.size:
            raise ValueError("trajectory action is out of range")
        if not bool(self.action_mask[self.action]):
            raise ValueError("trajectory action must be legal under its stored mask")


@dataclass(frozen=True)
class EpisodeTrajectory:
    schema_version: str
    episode_id: int
    worker_id: int
    seeds: EpisodeSeeds
    deck_card_ids: tuple[tuple[int, ...], tuple[int, ...]]
    steps: tuple[TrajectoryStep, ...]
    winner: int | None
    terminated: bool
    truncated: bool
    final_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema {self.schema_version!r}")
        if self.terminated and self.truncated:
            raise ValueError("episode cannot be terminated and truncated")
        if not self.steps:
            raise ValueError("episode trajectory must contain at least one step")
        final = self.steps[-1]
        if (final.terminated, final.truncated) != (self.terminated, self.truncated):
            raise ValueError("episode boundary flags disagree with final step")
