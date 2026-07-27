from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch
from torch import nn

from swb.engine.environment import (
    MATCH_SETUP_OFFICIAL,
    MATCH_SETUP_VALUES,
    ShadowverseEnv,
)
from swb.rl.class_schedule import class_pair_for_episode, normalize_class_ids
from swb.rl.opponents import OpponentEntry, OpponentPool
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.seeding import episode_seeds


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int = 256
    sequence_length: int = 16
    minibatch_sequences: int = 8
    update_epochs: int = 2
    hidden_size: int = 64
    card_embedding_dim: int = 16
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    max_agent_steps_per_episode: int = 256
    max_game_turns: int | None = 200
    opponent_current_weight: float = 1.0
    opponent_random_weight: float = 0.0
    opponent_fixed_weight: float = 0.0
    opponent_historical_weight: float = 0.0
    opponent_max_history: int = 8
    opponent_snapshot_interval_steps: int = 50_000
    rollout_workers: int = 1
    rollout_result_timeout_seconds: float = 120.0
    training_class_ids: tuple[int, ...] = (1,)
    match_setup: str = MATCH_SETUP_OFFICIAL

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_class_ids",
            normalize_class_ids(self.training_class_ids),
        )
        integer_fields = (
            self.rollout_steps,
            self.sequence_length,
            self.minibatch_sequences,
            self.update_epochs,
            self.hidden_size,
            self.card_embedding_dim,
            self.max_agent_steps_per_episode,
            self.opponent_max_history,
            self.opponent_snapshot_interval_steps,
            self.rollout_workers,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("PPO integer hyperparameters must be positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda are outside valid ranges")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.rollout_result_timeout_seconds <= 0:
            raise ValueError("rollout_result_timeout_seconds must be positive")
        if self.match_setup not in MATCH_SETUP_VALUES:
            raise ValueError("match_setup must be 'legacy' or 'official'")
        opponent_weights = (
            self.opponent_current_weight,
            self.opponent_random_weight,
            self.opponent_fixed_weight,
            self.opponent_historical_weight,
        )
        if any(weight < 0 for weight in opponent_weights):
            raise ValueError("opponent weights must be non-negative")
        if not any(weight > 0 for weight in opponent_weights[:3]):
            raise ValueError(
                "at least one initially available opponent must have positive weight"
            )
        if self.rollout_workers > 1 and opponent_weights != (1.0, 0.0, 0.0, 0.0):
            raise ValueError(
                "multiprocess policy rollout currently supports current-policy "
                "self-play only"
            )


class ObservationFlattener:
    CARD_INDEX_FIELDS = ("own_hand_cards", "public_board_cards")

    def __init__(
        self,
        field_names: tuple[str, ...],
        field_sizes: tuple[int, ...],
        card_field_names: tuple[str, ...],
        card_field_sizes: tuple[int, ...],
    ):
        self.field_names = field_names
        self.field_sizes = field_sizes
        self.size = sum(field_sizes)
        self.card_field_names = card_field_names
        self.card_field_sizes = card_field_sizes
        self.card_slots = sum(card_field_sizes)

    @classmethod
    def from_observation(
        cls,
        observation: Mapping[str, np.ndarray],
    ) -> ObservationFlattener:
        card_names = tuple(
            name for name in cls.CARD_INDEX_FIELDS if name in observation
        )
        names = tuple(
            name
            for name in observation
            if name != "action_mask" and name not in card_names
        )
        return cls(
            names,
            tuple(int(observation[name].size) for name in names),
            card_names,
            tuple(int(observation[name].size) for name in card_names),
        )

    def encode(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        values = []
        for name, expected_size in zip(self.field_names, self.field_sizes):
            array = np.asarray(observation[name], dtype=np.float32).reshape(-1)
            if array.size != expected_size:
                raise ValueError(
                    f"observation field {name!r} changed size: "
                    f"expected {expected_size}, got {array.size}"
                )
            if name in {
                "own_initial_deck",
                "opponent_initial_deck",
                "public_graveyards",
                "public_banished",
            }:
                array = array / 40.0
            elif name.endswith("_categorical"):
                array = array / 1024.0
            values.append(array)
        return np.concatenate(values, dtype=np.float32)

    def encode_cards(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        values = []
        for name, expected_size in zip(
            self.card_field_names, self.card_field_sizes
        ):
            array = np.asarray(observation[name], dtype=np.int64).reshape(-1)
            if array.size != expected_size:
                raise ValueError(
                    f"observation field {name!r} changed size: "
                    f"expected {expected_size}, got {array.size}"
                )
            if bool((array < 0).any()):
                raise ValueError(f"observation field {name!r} has a negative card index")
            values.append(array)
        if not values:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(values, dtype=np.int64)


class RecurrentMaskedActorCritic(nn.Module):
    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_size: int = 64,
        *,
        card_vocabulary_size: int = 0,
        card_slot_count: int = 0,
        card_embedding_dim: int = 16,
    ):
        super().__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.card_vocabulary_size = card_vocabulary_size
        self.card_slot_count = card_slot_count
        self.card_embedding_dim = card_embedding_dim
        if (card_vocabulary_size == 0) != (card_slot_count == 0):
            raise ValueError(
                "card_vocabulary_size and card_slot_count must both be zero or positive"
            )
        self.card_embedding = (
            nn.Embedding(
                card_vocabulary_size + 1,
                card_embedding_dim,
                padding_idx=0,
            )
            if card_vocabulary_size > 0
            else None
        )
        encoder_size = input_size + card_slot_count * card_embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_size, hidden_size),
            nn.Tanh(),
        )
        self.recurrent = nn.GRUCell(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def forward_step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        card_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_input = observation
        if self.card_embedding is not None:
            if card_indices is None:
                raise ValueError("card indices are required by this policy")
            if card_indices.shape[:-1] != observation.shape[:-1]:
                raise ValueError("card index batch shape must match observations")
            if card_indices.shape[-1] != self.card_slot_count:
                raise ValueError(
                    f"expected {self.card_slot_count} card slots, "
                    f"got {card_indices.shape[-1]}"
                )
            if bool((card_indices < 0).any()) or bool(
                (card_indices > self.card_vocabulary_size).any()
            ):
                raise ValueError("card index is outside the policy vocabulary")
            embedded = self.card_embedding(card_indices.to(dtype=torch.long))
            model_input = torch.cat(
                (observation, embedded.flatten(start_dim=-2)), dim=-1
            )
        encoded = self.encoder(model_input)
        next_hidden = self.recurrent(encoded, hidden)
        return self.policy_head(next_hidden), self.value_head(next_hidden).squeeze(-1), next_hidden

    @staticmethod
    def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        mask = action_mask.to(dtype=torch.bool)
        if mask.ndim != logits.ndim or mask.shape != logits.shape:
            raise ValueError("action mask shape must match policy logits")
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every live policy row must contain a legal action")
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


@dataclass
class _Record:
    episode_id: int
    player_id: int
    observation: np.ndarray
    card_indices: np.ndarray
    action_mask: np.ndarray
    action: int
    old_log_prob: float
    value: float
    reward: float
    hidden_before: np.ndarray
    trainable: bool = True
    opponent_id: str = "current"


@dataclass
class _SequenceBatch:
    observations: torch.Tensor
    card_indices: torch.Tensor
    action_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    valid: torch.Tensor
    initial_hidden: torch.Tensor


class PPOTrainer:
    """Shared-parameter recurrent masked PPO with per-player hidden state."""

    def __init__(
        self,
        snapshot: WorkerAssetsSnapshot,
        *,
        master_seed: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.snapshot = snapshot
        self.assets = snapshot.load()
        self.master_seed = master_seed
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        random.seed(master_seed)
        np.random.seed(master_seed % (2**32))
        torch.manual_seed(master_seed)
        self.torch_generator = torch.Generator(device=self.device)
        self.torch_generator.manual_seed(master_seed)
        self.next_episode_id = 0
        self.completed_episodes = 0
        self.agent_steps = 0
        self.update_count = 0
        self.opponent_pool = OpponentPool(
            master_seed,
            current_weight=self.config.opponent_current_weight,
            random_weight=self.config.opponent_random_weight,
            fixed_weight=self.config.opponent_fixed_weight,
            historical_weight=self.config.opponent_historical_weight,
            max_history=self.config.opponent_max_history,
            snapshot_interval_steps=self.config.opponent_snapshot_interval_steps,
        )
        self.learner_player = 0
        self.current_opponent: OpponentEntry | None = None
        self.opponent_assignments: list[dict[str, object]] = []
        self.opponent_rng = random.Random(master_seed)
        self.opponent_model: RecurrentMaskedActorCritic | None = None
        self.opponent_hidden: torch.Tensor | None = None
        self.env: ShadowverseEnv | None = None
        self.info: dict[str, object] | None = None
        self.current_episode_id: int | None = None
        self.hidden_by_player: dict[int, torch.Tensor] = {}
        self._policy_vector_rollout = None
        self._start_episode()
        if self.config.rollout_workers > 1:
            # The local environment is retained only as a version/schema anchor.
            # Vector workers own every sampled episode, starting at ID zero.
            self.next_episode_id = 0
            self.opponent_assignments.clear()
        assert self.env is not None and self.info is not None
        first_observation = self.env.observation(
            perspective=self.env.decision_player,
            action_mask=self.info["action_mask"],
        )
        self.flattener = ObservationFlattener.from_observation(first_observation)
        self.model = RecurrentMaskedActorCritic(
            self.flattener.size,
            self.env.ACTION_SIZE,
            self.config.hidden_size,
            card_vocabulary_size=len(self.assets.catalog.card_vocabulary),
            card_slot_count=self.flattener.card_slots,
            card_embedding_dim=self.config.card_embedding_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.hidden_by_player = {
            player: self.model.initial_state(1, device=self.device)
            for player in (0, 1)
        }

    def _load_historical_opponent(self, entry: OpponentEntry) -> None:
        from swb.rl.checkpoint import _load_payload
        from swb.rl.versioning import ExperimentVersions

        payload = _load_payload(entry.checkpoint_path)
        expected_versions = ExperimentVersions(**payload["versions"])
        actual_versions = ExperimentVersions.capture(
            self.env,
            self.assets.catalog,
            rulebook_sha256=self.snapshot.rulebook_sha256,
        )
        expected_versions.assert_compatible(actual_versions)
        config = PPOConfig(**payload["trainer"]["config"])
        if config.hidden_size != self.config.hidden_size:
            raise ValueError(
                f"historical opponent {entry.opponent_id!r} hidden size "
                f"{config.hidden_size} does not match current {self.config.hidden_size}"
            )
        if config.card_embedding_dim != self.config.card_embedding_dim:
            raise ValueError(
                f"historical opponent {entry.opponent_id!r} card embedding "
                f"dimension {config.card_embedding_dim} does not match current "
                f"{self.config.card_embedding_dim}"
            )
        model = RecurrentMaskedActorCritic(
            self.flattener.size,
            self.env.ACTION_SIZE,
            config.hidden_size,
            card_vocabulary_size=len(self.assets.catalog.card_vocabulary),
            card_slot_count=self.flattener.card_slots,
            card_embedding_dim=config.card_embedding_dim,
        ).to(self.device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        self.opponent_model = model
        self.opponent_hidden = model.initial_state(1, device=self.device)

    def _prepare_opponent(self) -> None:
        self.opponent_model = None
        self.opponent_hidden = None
        if (
            self.current_opponent is not None
            and self.current_opponent.kind == "historical"
        ):
            self._load_historical_opponent(self.current_opponent)

    def _start_episode(self) -> None:
        episode_id = self.next_episode_id
        self.next_episode_id += 1
        self.learner_player = episode_id % 2
        class_a, class_b = class_pair_for_episode(
            self.config.training_class_ids,
            episode_id,
        )
        self.current_opponent = self.opponent_pool.select(
            episode_id=episode_id,
            learner_player=self.learner_player,
        )
        self.opponent_assignments.append({
            "episode_id": episode_id,
            "learner_player": self.learner_player,
            "opponent_id": self.current_opponent.opponent_id,
            "opponent_kind": self.current_opponent.kind,
            "class_a": class_a,
            "class_b": class_b,
            "learner_class": (class_a, class_b)[self.learner_player],
            "opponent_class": (class_a, class_b)[1 - self.learner_player],
        })
        self.opponent_assignments = self.opponent_assignments[-4096:]
        seeds = episode_seeds(self.master_seed, 0, episode_id)
        deck_a = self.assets.catalog.sample_deck(
            class_a, random.Random(seeds.deck_seed_a)
        )
        deck_b = self.assets.catalog.sample_deck(
            class_b, random.Random(seeds.deck_seed_b)
        )
        self.env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=class_a,
            class_b=class_b,
            seed=seeds.engine_seed,
            rulebook=self.assets.rulebook,
            card_resolver=self.assets.catalog.resolve,
            observation_version="v3",
            card_vocabulary=self.assets.catalog.card_vocabulary,
            max_game_turns=self.config.max_game_turns,
            max_agent_steps=self.config.max_agent_steps_per_episode,
            training_mode=True,
            match_setup=self.config.match_setup,
        )
        _, self.info = self.env.reset(seed=seeds.engine_seed)
        self.current_episode_id = episode_id
        if hasattr(self, "model"):
            self.hidden_by_player = {
                player: self.model.initial_state(1, device=self.device)
                for player in (0, 1)
            }
            self._prepare_opponent()

    def _policy_action(
        self,
        player_id: int,
        vector: torch.Tensor,
        card_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[int, float, float, torch.Tensor, bool]:
        assert self.current_opponent is not None
        is_external_opponent = (
            player_id != self.learner_player
            and self.current_opponent.kind != "current"
        )
        if not is_external_opponent:
            hidden_before = self.hidden_by_player[player_id]
            with torch.no_grad():
                logits, value, next_hidden = self.model.forward_step(
                    vector, hidden_before, card_indices
                )
                masked = self.model.masked_logits(logits, mask)
                probabilities = torch.softmax(masked, dim=-1)
                action_tensor = torch.multinomial(
                    probabilities, 1, generator=self.torch_generator
                )
                log_prob = torch.log_softmax(masked, dim=-1).gather(
                    -1, action_tensor
                )
            self.hidden_by_player[player_id] = next_hidden.detach()
            return (
                int(action_tensor.item()),
                float(log_prob.item()),
                float(value.item()),
                hidden_before,
                True,
            )

        legal = torch.nonzero(mask.squeeze(0), as_tuple=False).squeeze(-1)
        if self.current_opponent.kind == "random_legal":
            action = int(legal[self.opponent_rng.randrange(legal.numel())].item())
            return action, 0.0, 0.0, self.hidden_by_player[player_id], False
        if self.current_opponent.kind == "fixed":
            return (
                int(legal[0].item()),
                0.0,
                0.0,
                self.hidden_by_player[player_id],
                False,
            )
        if self.opponent_model is None or self.opponent_hidden is None:
            raise RuntimeError("historical opponent model is not initialized")
        hidden_before = self.opponent_hidden
        with torch.no_grad():
            logits, value, next_hidden = self.opponent_model.forward_step(
                vector, hidden_before, card_indices
            )
            masked = self.opponent_model.masked_logits(logits, mask)
            probabilities = torch.softmax(masked, dim=-1)
            action_tensor = torch.multinomial(
                probabilities, 1, generator=self.torch_generator
            )
            log_prob = torch.log_softmax(masked, dim=-1).gather(-1, action_tensor)
        self.opponent_hidden = next_hidden.detach()
        return (
            int(action_tensor.item()),
            float(log_prob.item()),
            float(value.item()),
            hidden_before,
            False,
        )

    def _value_for_player(self, player_id: int) -> float:
        assert self.env is not None
        observation = self.env.observation(
            perspective=player_id,
            action_mask=[False] * self.env.ACTION_SIZE,
        )
        vector = torch.from_numpy(self.flattener.encode(observation)).to(self.device)
        card_indices = torch.from_numpy(
            self.flattener.encode_cards(observation)
        ).to(self.device)
        with torch.no_grad():
            _, value, _ = self.model.forward_step(
                vector.unsqueeze(0),
                self.hidden_by_player[player_id],
                card_indices.unsqueeze(0),
            )
        return float(value.item())

    def collect_rollout(
        self,
    ) -> tuple[list[_Record], dict[tuple[int, int], float], dict[int, str]]:
        if self.config.rollout_workers > 1:
            return self._collect_vector_rollout()
        records: list[_Record] = []
        bootstrap: dict[tuple[int, int], float] = {}
        boundaries: dict[int, str] = {}
        episode_record_indices: dict[tuple[int, int], list[int]] = {}
        while len(records) < self.config.rollout_steps:
            assert self.env is not None and self.info is not None
            assert self.current_episode_id is not None
            episode_id = self.current_episode_id
            player_id = self.env.decision_player
            observation = self.env.observation(
                perspective=player_id,
                action_mask=self.info["action_mask"],
            )
            vector_np = self.flattener.encode(observation)
            card_indices_np = self.flattener.encode_cards(observation)
            mask_np = np.asarray(self.info["action_mask"], dtype=np.bool_)
            vector = torch.from_numpy(vector_np).to(self.device).unsqueeze(0)
            card_indices = torch.from_numpy(card_indices_np).to(
                self.device
            ).unsqueeze(0)
            mask = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)
            action, log_prob, value, hidden_before, trainable = self._policy_action(
                player_id, vector, card_indices, mask
            )
            result = self.env.step(action)
            index = len(records)
            records.append(_Record(
                episode_id=episode_id,
                player_id=player_id,
                observation=vector_np,
                card_indices=card_indices_np,
                action_mask=mask_np,
                action=action,
                old_log_prob=log_prob,
                value=value,
                reward=0.0,
                hidden_before=hidden_before.squeeze(0).cpu().numpy().copy(),
                trainable=trainable,
                opponent_id=self.current_opponent.opponent_id,
            ))
            if trainable:
                episode_record_indices.setdefault((episode_id, player_id), []).append(index)
            self.info = result.info
            self.agent_steps += 1

            if result.terminated or result.truncated:
                boundary = "terminated" if result.terminated else "truncated"
                boundaries[episode_id] = boundary
                for candidate in (0, 1):
                    indices = episode_record_indices.get((episode_id, candidate), [])
                    if indices and result.terminated:
                        records[indices[-1]].reward = (
                            0.0
                            if self.env.winner is None
                            else (1.0 if self.env.winner == candidate else -1.0)
                        )
                    bootstrap[(episode_id, candidate)] = (
                        0.0
                        if result.terminated
                        else self._value_for_player(candidate)
                    )
                self.completed_episodes += 1
                self._start_episode()

        assert self.current_episode_id is not None
        if self.current_episode_id not in boundaries:
            boundaries[self.current_episode_id] = "rollout_cut"
            for candidate in (0, 1):
                bootstrap[(self.current_episode_id, candidate)] = self._value_for_player(
                    candidate
                )
        return records, bootstrap, boundaries

    def _collect_vector_rollout(
        self,
    ) -> tuple[list[_Record], dict[tuple[int, int], float], dict[int, str]]:
        from swb.rl.vector_rollout import PolicyVectorRollout, RolloutConfig

        if self._policy_vector_rollout is None:
            per_worker_steps = max(
                1,
                math.ceil(
                    self.config.rollout_steps / self.config.rollout_workers
                ),
            )
            self._policy_vector_rollout = PolicyVectorRollout(
                self.snapshot,
                RolloutConfig(
                    master_seed=self.master_seed,
                    worker_count=self.config.rollout_workers,
                    class_ids=self.config.training_class_ids,
                    max_game_turns=self.config.max_game_turns,
                    max_agent_steps=min(
                        self.config.max_agent_steps_per_episode,
                        per_worker_steps,
                    ),
                    result_timeout_seconds=(
                        self.config.rollout_result_timeout_seconds
                    ),
                    match_setup=self.config.match_setup,
                ),
            )
        records: list[_Record] = []
        bootstrap: dict[tuple[int, int], float] = {}
        boundaries: dict[int, str] = {}
        while len(records) < self.config.rollout_steps:
            episode_ids = tuple(
                range(
                    self.next_episode_id,
                    self.next_episode_id + self.config.rollout_workers,
                )
            )
            self.next_episode_id += len(episode_ids)
            episodes = self._policy_vector_rollout.collect(
                self.model, episode_ids
            )
            for episode in episodes:
                class_a, class_b = class_pair_for_episode(
                    self.config.training_class_ids,
                    episode.episode_id,
                )
                self.opponent_assignments.append({
                    "episode_id": episode.episode_id,
                    "learner_player": "both",
                    "opponent_id": "current",
                    "opponent_kind": "current",
                    "worker_id": episode.worker_id,
                    "class_a": class_a,
                    "class_b": class_b,
                    "learner_class": "both",
                    "opponent_class": "self_play",
                })
                for step in episode.records:
                    records.append(_Record(
                        episode_id=step.episode_id,
                        player_id=step.player_id,
                        observation=step.observation,
                        card_indices=step.card_indices,
                        action_mask=step.action_mask,
                        action=step.action,
                        old_log_prob=step.old_log_prob,
                        value=step.value,
                        reward=step.reward,
                        hidden_before=step.hidden_before,
                        trainable=True,
                        opponent_id="current",
                    ))
                bootstrap.update(episode.bootstrap)
                boundaries[episode.episode_id] = episode.boundary
                self.completed_episodes += 1
        self.opponent_assignments = self.opponent_assignments[-4096:]
        self.agent_steps += len(records)
        return records, bootstrap, boundaries

    def close(self) -> None:
        if self._policy_vector_rollout is not None:
            self._policy_vector_rollout.close()
            self._policy_vector_rollout = None

    def _advantages(
        self,
        records: list[_Record],
        bootstrap: dict[tuple[int, int], float],
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(records), dtype=np.float32)
        returns = np.zeros(len(records), dtype=np.float32)
        groups: dict[tuple[int, int], list[int]] = {}
        for index, record in enumerate(records):
            if record.trainable:
                groups.setdefault((record.episode_id, record.player_id), []).append(index)
        for key, indices in groups.items():
            next_value = bootstrap.get(key, 0.0)
            gae = 0.0
            for index in reversed(indices):
                record = records[index]
                delta = record.reward + self.config.gamma * next_value - record.value
                gae = delta + self.config.gamma * self.config.gae_lambda * gae
                advantages[index] = gae
                returns[index] = gae + record.value
                next_value = record.value
        trainable = np.asarray([record.trainable for record in records], dtype=np.bool_)
        if not bool(trainable.any()):
            raise ValueError("rollout contains no trainable policy transitions")
        mean = float(advantages[trainable].mean())
        std = float(advantages[trainable].std())
        advantages[trainable] = (
            advantages[trainable] - mean
        ) / max(std, 1e-8)
        return advantages, returns

    def _sequence_batches(
        self,
        records: list[_Record],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        groups: dict[tuple[int, int], list[int]] = {}
        for index, record in enumerate(records):
            if record.trainable:
                groups.setdefault((record.episode_id, record.player_id), []).append(index)
        chunks = []
        length = self.config.sequence_length
        for indices in groups.values():
            for start in range(0, len(indices), length):
                selected = indices[start : start + length]
                chunks.append({
                    "indices": np.asarray(selected, dtype=np.int64),
                    "initial_hidden": records[selected[0]].hidden_before,
                })
        return chunks

    def _collate(
        self,
        chunks: list[dict[str, np.ndarray]],
        records: list[_Record],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> _SequenceBatch:
        batch = len(chunks)
        length = self.config.sequence_length
        observations = np.zeros(
            (batch, length, self.flattener.size), dtype=np.float32
        )
        card_indices = np.zeros(
            (batch, length, self.flattener.card_slots), dtype=np.int64
        )
        masks = np.zeros((batch, length, self.env.ACTION_SIZE), dtype=np.bool_)
        actions = np.zeros((batch, length), dtype=np.int64)
        old_log_probs = np.zeros((batch, length), dtype=np.float32)
        batch_advantages = np.zeros((batch, length), dtype=np.float32)
        batch_returns = np.zeros((batch, length), dtype=np.float32)
        valid = np.zeros((batch, length), dtype=np.bool_)
        initial_hidden = np.zeros(
            (batch, self.config.hidden_size), dtype=np.float32
        )
        for row, chunk in enumerate(chunks):
            indices = chunk["indices"]
            initial_hidden[row] = chunk["initial_hidden"]
            for column, index in enumerate(indices):
                record = records[int(index)]
                observations[row, column] = record.observation
                card_indices[row, column] = record.card_indices
                masks[row, column] = record.action_mask
                actions[row, column] = record.action
                old_log_probs[row, column] = record.old_log_prob
                batch_advantages[row, column] = advantages[int(index)]
                batch_returns[row, column] = returns[int(index)]
                valid[row, column] = True

        def tensor(value, dtype=None):
            return torch.as_tensor(value, dtype=dtype, device=self.device)

        return _SequenceBatch(
            observations=tensor(observations),
            card_indices=tensor(card_indices, torch.long),
            action_masks=tensor(masks, torch.bool),
            actions=tensor(actions, torch.long),
            old_log_probs=tensor(old_log_probs),
            advantages=tensor(batch_advantages),
            returns=tensor(batch_returns),
            valid=tensor(valid, torch.bool),
            initial_hidden=tensor(initial_hidden),
        )

    def update(self, records: list[_Record], bootstrap) -> dict[str, float]:
        advantages, returns = self._advantages(records, bootstrap)
        chunks = self._sequence_batches(records, advantages, returns)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "grad_norm": 0.0}
        updates = 0
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(
                len(chunks), generator=self.torch_generator
            ).tolist()
            for start in range(0, len(chunks), self.config.minibatch_sequences):
                selected = [
                    chunks[index]
                    for index in permutation[start : start + self.config.minibatch_sequences]
                ]
                batch = self._collate(selected, records, advantages, returns)
                hidden = batch.initial_hidden
                logits_rows = []
                value_rows = []
                for timestep in range(self.config.sequence_length):
                    logits, values, hidden = self.model.forward_step(
                        batch.observations[:, timestep],
                        hidden,
                        batch.card_indices[:, timestep],
                    )
                    logits_rows.append(logits)
                    value_rows.append(values)
                logits = torch.stack(logits_rows, dim=1)
                values = torch.stack(value_rows, dim=1)
                flat_valid = batch.valid
                masked_logits = self.model.masked_logits(
                    logits[flat_valid], batch.action_masks[flat_valid]
                )
                actions = batch.actions[flat_valid]
                log_probs_all = torch.log_softmax(masked_logits, dim=-1)
                log_probs = log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                probabilities = torch.softmax(masked_logits, dim=-1)
                entropy = -(probabilities * log_probs_all).sum(dim=-1).mean()
                ratios = torch.exp(log_probs - batch.old_log_probs[flat_valid])
                advantage_values = batch.advantages[flat_valid]
                unclipped = ratios * advantage_values
                clipped = torch.clamp(
                    ratios,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantage_values
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(
                    values[flat_valid], batch.returns[flat_valid]
                )
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("non-finite PPO gradient norm")
                self.optimizer.step()
                if not all(
                    bool(torch.isfinite(parameter).all())
                    for parameter in self.model.parameters()
                ):
                    raise FloatingPointError("non-finite PPO model parameter")
                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["grad_norm"] += float(grad_norm.item())
                updates += 1
        self.update_count += 1
        for key in metrics:
            metrics[key] /= max(1, updates)
        metrics.update({
            "agent_steps": float(self.agent_steps),
            "completed_episodes": float(self.completed_episodes),
            "updates": float(self.update_count),
        })
        return metrics

    def train(self, total_agent_steps: int) -> list[dict[str, float]]:
        if total_agent_steps <= self.agent_steps:
            raise ValueError("total_agent_steps must exceed current training progress")
        metrics = []
        while self.agent_steps < total_agent_steps:
            records, bootstrap, _ = self.collect_rollout()
            metrics.append(self.update(records, bootstrap))
        return metrics

    def hyperparameters(self) -> dict[str, object]:
        return asdict(self.config)
