from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import pickle
import queue
import random
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.reduction import ForkingPickler
from typing import Any, Mapping

import numpy as np

from swb.engine.environment import (
    MATCH_SETUP_OFFICIAL,
    MATCH_SETUP_VALUES,
    ShadowverseEnv,
)
from swb.rl.class_schedule import class_pair_for_episode, normalize_class_ids
from swb.rl.deck_schedule import (
    DeckMatchup,
    deck_matchup_for_episode,
    normalize_opponent_decks,
)
from swb.rl.fixed_decks import get_fixed_training_deck
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.seeding import episode_seeds
from swb.rl.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    EpisodeTrajectory,
    TrajectoryStep,
)
from swb.rl.versioning import ExperimentVersions


@dataclass(frozen=True)
class RolloutConfig:
    master_seed: int
    worker_count: int = 1
    class_a: int = 1
    class_b: int = 1
    class_ids: tuple[int, ...] = ()
    max_game_turns: int | None = 200
    max_agent_steps: int = 2000
    open_decklists: bool = False
    start_method: str = "spawn"
    result_timeout_seconds: float = 120.0
    fail_episode_id: int | None = None
    training_deck: str | None = None
    opponent_decks: tuple[str, ...] = ()
    match_setup: str = MATCH_SETUP_OFFICIAL
    worker_torch_threads: int = 2
    central_inference_batch_wait_seconds: float = 0.0005
    profile_ipc_timing: bool = False
    profile_central_timing: bool = False
    observation_version: str = "v4"

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if self.class_a not in range(1, 8) or self.class_b not in range(1, 8):
            raise ValueError("class_a and class_b must be in 1..7")
        if self.class_ids:
            object.__setattr__(
                self,
                "class_ids",
                normalize_class_ids(self.class_ids),
            )
        if self.max_agent_steps <= 0:
            raise ValueError("max_agent_steps must be positive")
        if self.result_timeout_seconds <= 0:
            raise ValueError("result_timeout_seconds must be positive")
        if self.worker_torch_threads <= 0:
            raise ValueError("worker_torch_threads must be positive")
        if self.central_inference_batch_wait_seconds < 0:
            raise ValueError(
                "central_inference_batch_wait_seconds must be non-negative"
            )
        if self.start_method not in mp.get_all_start_methods():
            raise ValueError(f"unsupported multiprocessing start method {self.start_method!r}")
        if self.match_setup not in MATCH_SETUP_VALUES:
            raise ValueError("match_setup must be 'legacy' or 'official'")
        if self.observation_version not in {"v3", "v4", "v4.1"}:
            raise ValueError(
                "observation_version must be 'v3', 'v4', or 'v4.1'"
            )
        if self.opponent_decks:
            if self.training_deck is None:
                raise ValueError(
                    "opponent_decks requires one fixed training_deck"
                )
            object.__setattr__(
                self,
                "opponent_decks",
                normalize_opponent_decks(
                    self.training_deck,
                    self.opponent_decks,
                ),
            )
        elif self.training_deck is not None:
            fixed_deck = get_fixed_training_deck(self.training_deck)
            configured_classes = (
                self.class_ids
                if self.class_ids
                else (self.class_a, self.class_b)
            )
            if any(
                class_id != fixed_deck.class_id
                for class_id in configured_classes
            ):
                raise ValueError(
                    f"fixed training deck {self.training_deck!r} requires "
                    f"class {fixed_deck.class_id}"
                )


class RolloutWorkerError(RuntimeError):
    pass


_PROFILED_POLICY_REQUEST = "profiled_policy_inference"


def _profile_policy_request(message: tuple) -> tuple:
    serialization_started = time.perf_counter()
    payload = bytes(ForkingPickler.dumps(message))
    serialization_seconds = time.perf_counter() - serialization_started
    sent_at = time.perf_counter()
    return (
        _PROFILED_POLICY_REQUEST,
        sent_at,
        serialization_seconds,
        len(payload),
        payload,
    )


def _decode_profiled_policy_request(
    envelope: tuple,
    *,
    received_at: float,
) -> tuple:
    if len(envelope) != 5:
        raise RolloutWorkerError("invalid profiled policy request envelope")
    _, sent_at, serialization_seconds, payload_bytes, payload = envelope
    if not isinstance(payload, bytes):
        raise RolloutWorkerError(
            "profiled policy request payload must be bytes"
        )
    message = pickle.loads(payload)
    if (
        not isinstance(message, tuple)
        or not message
        or message[0] != "policy_inference"
    ):
        raise RolloutWorkerError(
            "profiled policy request decoded an unexpected message"
        )
    return (
        *message,
        {
            "received_at": float(received_at),
            "serialization_seconds": float(serialization_seconds),
            "send_seconds": max(0.0, float(received_at) - float(sent_at)),
            "payload_bytes": float(payload_bytes),
        },
    )


def _profile_model_forward_step(
    model,
    observations,
    hidden,
    card_indices,
) -> tuple[tuple[Any, Any, Any], dict[str, float]]:
    """Measure model components without changing the executed forward graph."""
    import torch

    transformer = getattr(model, "entity_encoder", None)
    encoder = transformer or getattr(model, "encoder", None)
    modules = {
        "encoder": encoder,
        "gru": getattr(model, "recurrent", None),
        "policy_head": getattr(model, "policy_head", None),
        "value_head": getattr(model, "value_head", None),
    }
    if modules["encoder"] is None or modules["gru"] is None:
        raise ValueError(
            "central component profiling requires an encoder and recurrent "
            "module"
        )
    use_cuda_events = observations.device.type == "cuda"
    markers: dict[str, Any] = {}
    handles = []

    def mark(name: str) -> None:
        if use_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            markers[name] = event
        else:
            markers[name] = time.perf_counter()

    def pre_hook(name: str):
        def hook(_module, _inputs) -> None:
            mark(f"{name}_start")

        return hook

    def post_hook(name: str):
        def hook(_module, _inputs, _output) -> None:
            mark(f"{name}_end")

        return hook

    for name, module in modules.items():
        if module is None:
            continue
        handles.append(module.register_forward_pre_hook(pre_hook(name)))
        handles.append(module.register_forward_hook(post_hook(name)))
    try:
        mark("model_start")
        outputs = model.forward_step(
            observations,
            hidden,
            card_indices,
            validate_card_indices=False,
        )
        mark("model_end")
    finally:
        for handle in handles:
            handle.remove()
    if use_cuda_events:
        torch.cuda.synchronize(observations.device)

    def elapsed(start: str, end: str) -> float:
        if start not in markers or end not in markers:
            return 0.0
        if use_cuda_events:
            return markers[start].elapsed_time(markers[end]) / 1000.0
        return float(markers[end] - markers[start])

    encoder_seconds = elapsed("encoder_start", "encoder_end")
    return outputs, {
        "central_model_input_encoding_seconds": elapsed(
            "model_start", "encoder_start"
        ),
        "central_encoder_seconds": encoder_seconds,
        "central_transformer_seconds": (
            encoder_seconds if transformer is not None else 0.0
        ),
        "central_transformer_to_gru_seconds": elapsed(
            "encoder_end", "gru_start"
        ),
        "central_gru_seconds": elapsed("gru_start", "gru_end"),
        "central_action_value_stage_seconds": elapsed(
            "gru_end", "model_end"
        ),
        "central_policy_head_seconds": elapsed(
            "policy_head_start", "policy_head_end"
        ),
        "central_value_head_seconds": elapsed(
            "value_head_start", "value_head_end"
        ),
        "central_model_forward_seconds": elapsed(
            "model_start", "model_end"
        ),
    }


@dataclass
class PolicyStep:
    episode_id: int
    worker_id: int
    player_id: int
    observation: np.ndarray
    card_indices: np.ndarray
    action_mask: np.ndarray
    action: int
    old_log_prob: float
    value: float
    reward: float
    hidden_before: np.ndarray


@dataclass(frozen=True)
class PolicyEpisode:
    episode_id: int
    worker_id: int
    records: tuple[PolicyStep, ...]
    bootstrap: Mapping[tuple[int, int], float]
    boundary: str
    winner: int | None
    matchup: Mapping[str, object] | None = None
    timing: Mapping[str, float] = field(default_factory=dict)


def _episode_matchup(
    config: RolloutConfig,
    episode_id: int,
) -> DeckMatchup | None:
    if not config.opponent_decks:
        return None
    assert config.training_deck is not None
    return deck_matchup_for_episode(
        config.training_deck,
        config.opponent_decks,
        episode_id,
    )


def _episode_classes(
    config: RolloutConfig,
    episode_id: int,
) -> tuple[int, int]:
    matchup = _episode_matchup(config, episode_id)
    if matchup is not None:
        return matchup.class_ids
    if config.class_ids:
        return class_pair_for_episode(config.class_ids, episode_id)
    return config.class_a, config.class_b


def _episode_decks(
    assets,
    config,
    seeds,
    class_a: int,
    class_b: int,
    episode_id: int,
):
    matchup = _episode_matchup(config, episode_id)
    if matchup is not None:
        recipe_a, recipe_b = matchup.decks
        return (
            recipe_a.build(assets.catalog),
            recipe_b.build(assets.catalog),
        )
    if config.training_deck is not None:
        fixed_deck = get_fixed_training_deck(config.training_deck)
        return (
            fixed_deck.build(assets.catalog),
            fixed_deck.build(assets.catalog),
        )
    return (
        assets.catalog.sample_deck(
            class_a,
            random.Random(seeds.deck_seed_a),
        ),
        assets.catalog.sample_deck(
            class_b,
            random.Random(seeds.deck_seed_b),
        ),
    )


def _run_episode(
    snapshot: WorkerAssetsSnapshot,
    assets,
    config: RolloutConfig,
    worker_id: int,
    episode_id: int,
) -> EpisodeTrajectory:
    if config.fail_episode_id == episode_id:
        raise RuntimeError(f"injected rollout failure for episode {episode_id}")
    seeds = episode_seeds(config.master_seed, worker_id, episode_id)
    class_a, class_b = _episode_classes(config, episode_id)
    deck_a, deck_b = _episode_decks(
        assets,
        config,
        seeds,
        class_a,
        class_b,
        episode_id,
    )
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=class_a,
        class_b=class_b,
        seed=seeds.engine_seed,
        rulebook=assets.rulebook,
        card_resolver=assets.catalog.resolve,
        observation_version=config.observation_version,
        card_vocabulary=assets.catalog.card_vocabulary,
        open_decklists=config.open_decklists,
        max_game_turns=config.max_game_turns,
        max_agent_steps=config.max_agent_steps,
        training_mode=True,
        match_setup=config.match_setup,
    )
    _, info = env.reset(seed=seeds.engine_seed)
    versions = ExperimentVersions.capture(
        env,
        assets.catalog,
        rulebook_sha256=snapshot.rulebook_sha256,
    ).to_dict()
    chooser = random.Random(seeds.policy_seed)
    steps: list[TrajectoryStep] = []
    seen_players: set[int] = set()
    while not env.terminated and not env.truncated:
        player_id = env.decision_player
        mask = np.asarray(info["action_mask"], dtype=np.int8)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError(
                f"episode {episode_id} has no legal action before termination"
            )
        action = int(legal[chooser.randrange(legal.size)])
        observation = env.observation(
            perspective=player_id,
            action_mask=mask,
        )
        result = env.step(action)
        steps.append(TrajectoryStep(
            episode_id=episode_id,
            worker_id=worker_id,
            step_index=len(steps),
            player_id=player_id,
            observation=observation,
            action=action,
            action_mask=mask,
            reward=float(result.reward),
            terminated=result.terminated,
            truncated=result.truncated,
            bootstrap_value_allowed=(result.truncated and not result.terminated),
            policy_log_prob=-math.log(float(legal.size)),
            recurrent_state_reset=player_id not in seen_players,
            opponent_id="random_legal",
            versions=versions,
        ))
        seen_players.add(player_id)
        info = result.info

    fingerprint = repr(env._core.deterministic_fingerprint()).encode("utf-8")
    return EpisodeTrajectory(
        schema_version=TRAJECTORY_SCHEMA_VERSION,
        episode_id=episode_id,
        worker_id=worker_id,
        seeds=seeds,
        deck_card_ids=(
            tuple(card.card_id for card in deck_a),
            tuple(card.card_id for card in deck_b),
        ),
        steps=tuple(steps),
        winner=env.winner,
        terminated=env.terminated,
        truncated=env.truncated,
        final_fingerprint_sha256=hashlib.sha256(fingerprint).hexdigest(),
    )


def _worker_main(
    snapshot: WorkerAssetsSnapshot,
    config: RolloutConfig,
    worker_id: int,
    input_queue: Any,
    output_queue: Any,
) -> None:
    try:
        assets = snapshot.load()
        while True:
            message = input_queue.get()
            if message is None:
                return
            episode_id = int(message)
            try:
                trajectory = _run_episode(
                    snapshot,
                    assets,
                    config,
                    worker_id,
                    episode_id,
                )
            except Exception as exc:
                output_queue.put((
                    "error",
                    worker_id,
                    episode_id,
                    type(exc).__name__,
                    str(exc),
                    traceback.format_exc(),
                ))
                return
            output_queue.put(("episode", episode_id, trajectory))
    except Exception as exc:
        output_queue.put((
            "error",
            worker_id,
            None,
            type(exc).__name__,
            str(exc),
            traceback.format_exc(),
        ))


def _run_central_policy_episode(
    snapshot: WorkerAssetsSnapshot,
    assets,
    config: RolloutConfig,
    worker_id: int,
    generation: int,
    episode_id: int,
    input_queue: Any,
    output_queue: Any,
    *,
    assignment_wait_seconds: float,
) -> None:

    from swb.rl.ppo import ObservationFlattener

    episode_started = time.perf_counter()
    setup_started = time.perf_counter()
    seeds = episode_seeds(config.master_seed, worker_id, episode_id)
    matchup = _episode_matchup(config, episode_id)
    class_a, class_b = _episode_classes(config, episode_id)
    deck_construction_started = time.perf_counter()
    deck_a, deck_b = _episode_decks(
        assets,
        config,
        seeds,
        class_a,
        class_b,
        episode_id,
    )
    deck_construction_seconds = (
        time.perf_counter() - deck_construction_started
    )
    environment_construction_started = time.perf_counter()
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=class_a,
        class_b=class_b,
        seed=seeds.engine_seed,
        rulebook=assets.rulebook,
        card_resolver=assets.catalog.resolve,
        observation_version=config.observation_version,
        card_vocabulary=assets.catalog.card_vocabulary,
        open_decklists=config.open_decklists,
        max_game_turns=config.max_game_turns,
        max_agent_steps=config.max_agent_steps,
        training_mode=True,
        match_setup=config.match_setup,
    )
    environment_construction_seconds = (
        time.perf_counter() - environment_construction_started
    )
    reset_started = time.perf_counter()
    observation, info = env.reset(seed=seeds.engine_seed)
    reset_seconds = time.perf_counter() - reset_started
    flattener = ObservationFlattener.from_observation(observation)
    setup_seconds = time.perf_counter() - setup_started
    observation_seconds = 0.0
    decision_observation_construction_seconds = 0.0
    step_observation_construction_seconds = 0.0
    inference_round_trip_seconds = 0.0
    ipc_request_serialization_seconds = 0.0
    ipc_response_wait_seconds = 0.0
    ipc_request_payload_bytes = 0.0
    ipc_profiled_requests = 0
    request_queue_put_seconds = 0.0
    response_queue_wait_seconds = 0.0
    engine_step_seconds = 0.0
    action_mask_seconds = 0.0
    command_decode_seconds = 0.0
    resolution_seconds = 0.0
    mulligan_seconds = 0.0
    mulligan_steps = 0
    step_index = 0
    while not env.terminated and not env.truncated:
        observation_started = time.perf_counter()
        player_id = env.decision_player
        vector_np = flattener.encode(observation)
        card_indices_np = flattener.encode_cards(observation)
        mask_np = np.asarray(info["action_mask"], dtype=np.bool_)
        observation_seconds += time.perf_counter() - observation_started

        inference_started = time.perf_counter()
        request = (
            "policy_inference",
            worker_id,
            generation,
            episode_id,
            step_index,
            player_id,
            vector_np,
            card_indices_np,
            mask_np,
        )
        if config.profile_ipc_timing:
            envelope = _profile_policy_request(request)
            ipc_request_serialization_seconds += float(envelope[2])
            ipc_request_payload_bytes += float(envelope[3])
            ipc_profiled_requests += 1
            request_put_started = time.perf_counter()
            output_queue.put(envelope)
        else:
            request_put_started = time.perf_counter()
            output_queue.put(request)
        request_queue_put_seconds += (
            time.perf_counter() - request_put_started
        )
        response_wait_started = time.perf_counter()
        response = input_queue.get()
        response_queue_wait_seconds += (
            time.perf_counter() - response_wait_started
        )
        response_received_at = time.perf_counter()
        if response is None:
            raise RuntimeError("central policy inference was interrupted")
        if (
            response[0] != "policy_action"
            or response[1] != generation
            or response[2] != episode_id
            or response[3] != step_index
        ):
            raise RuntimeError(
                f"unexpected central policy response {response[0]!r}"
            )
        if config.profile_ipc_timing:
            if len(response) != 6 or response[5] is None:
                raise RuntimeError(
                    "profiled policy response lacks request receipt timing"
                )
            ipc_response_wait_seconds += max(
                0.0,
                response_received_at - float(response[5]),
            )
        action = int(response[4])
        inference_round_trip_seconds += (
            time.perf_counter() - inference_started
        )

        engine_step_started = time.perf_counter()
        phase_before_step = str(info["phase"])
        step_timing: dict[str, float] = {}
        result = env.step(action, timing=step_timing)
        engine_step_elapsed = time.perf_counter() - engine_step_started
        engine_step_seconds += engine_step_elapsed
        if phase_before_step == "mulligan":
            mulligan_seconds += engine_step_elapsed
            mulligan_steps += 1
        action_mask_seconds += step_timing["action_mask_seconds"]
        command_decode_seconds += step_timing["command_decode_seconds"]
        resolution_seconds += step_timing["resolution_seconds"]
        step_observation_construction_seconds += step_timing[
            "observation_seconds"
        ]

        info = result.info
        observation = result.observation
        step_index += 1

    bootstrap_started = time.perf_counter()
    bootstrap_observation_construction_seconds = 0.0
    boundary = "terminated" if env.terminated else "truncated"
    bootstrap_inputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if env.truncated:
        for player_id in (0, 1):
            construction_started = time.perf_counter()
            observation = env.observation(
                perspective=player_id,
                action_mask=[False] * env.ACTION_SIZE,
            )
            bootstrap_observation_construction_seconds += (
                time.perf_counter() - construction_started
            )
            bootstrap_inputs[player_id] = (
                flattener.encode(observation),
                flattener.encode_cards(observation),
            )
    bootstrap_seconds = time.perf_counter() - bootstrap_started
    episode_total_seconds = time.perf_counter() - episode_started
    long_episode_threshold_steps = max(
        1,
        math.ceil(config.max_agent_steps * 0.75),
    )
    output_queue.put((
        "policy_episode_end",
        episode_id,
        worker_id,
        generation,
        boundary,
        env.winner,
        bootstrap_inputs,
        None if matchup is None else matchup.manifest(),
        {
            "worker_episode_total_seconds": episode_total_seconds,
            "worker_setup_seconds": setup_seconds,
            "worker_assignment_wait_seconds": assignment_wait_seconds,
            "worker_deck_construction_seconds": deck_construction_seconds,
            "worker_environment_construction_seconds": (
                environment_construction_seconds
            ),
            "worker_reset_seconds": reset_seconds,
            "worker_mulligan_seconds": mulligan_seconds,
            "worker_mulligan_steps": float(mulligan_steps),
            "worker_observation_seconds": observation_seconds,
            "worker_observation_construction_seconds": (
                decision_observation_construction_seconds
                + step_observation_construction_seconds
                + bootstrap_observation_construction_seconds
            ),
            "worker_decision_observation_construction_seconds": (
                decision_observation_construction_seconds
            ),
            "worker_step_observation_construction_seconds": (
                step_observation_construction_seconds
            ),
            "worker_bootstrap_observation_construction_seconds": (
                bootstrap_observation_construction_seconds
            ),
            "worker_inference_round_trip_seconds": (
                inference_round_trip_seconds
            ),
            "worker_ipc_request_serialization_seconds": (
                ipc_request_serialization_seconds
            ),
            "worker_ipc_response_wait_seconds": (
                ipc_response_wait_seconds
            ),
            "worker_ipc_request_payload_bytes": (
                ipc_request_payload_bytes
            ),
            "worker_ipc_profiled_requests": float(ipc_profiled_requests),
            "worker_request_queue_put_seconds": (
                request_queue_put_seconds
            ),
            "worker_response_queue_wait_seconds": (
                response_queue_wait_seconds
            ),
            "worker_engine_step_seconds": engine_step_seconds,
            "worker_action_mask_seconds": action_mask_seconds,
            "worker_command_decode_seconds": command_decode_seconds,
            "worker_resolution_seconds": resolution_seconds,
            "worker_bootstrap_seconds": bootstrap_seconds,
            "worker_agent_steps": float(step_index),
            "worker_episode_count": 1.0,
            f"worker_episode_steps_{step_index}_count": 1.0,
            "worker_long_episode_threshold_steps": float(
                long_episode_threshold_steps
            ),
            "worker_long_episode_count": float(
                step_index >= long_episode_threshold_steps
            ),
            "worker_terminated_episode_count": float(env.terminated),
            "worker_truncated_episode_count": float(env.truncated),
            "worker_torch_threads": float(config.worker_torch_threads),
        },
    ))


def _policy_worker_main(
    snapshot: WorkerAssetsSnapshot,
    config: RolloutConfig,
    worker_id: int,
    input_queue: Any,
    output_queue: Any,
) -> None:
    try:
        import torch

        torch.set_num_threads(config.worker_torch_threads)
        torch.set_num_interop_threads(1)
        assets = snapshot.load()
        while True:
            assignment_wait_started = time.perf_counter()
            message = input_queue.get()
            assignment_wait_seconds = (
                time.perf_counter() - assignment_wait_started
            )
            if message is None:
                return
            command = message[0]
            if command != "episode":
                raise RuntimeError(f"unexpected policy worker command {command!r}")
            _, generation, episode_id = message
            try:
                _run_central_policy_episode(
                    snapshot,
                    assets,
                    config,
                    worker_id,
                    int(generation),
                    int(episode_id),
                    input_queue,
                    output_queue,
                    assignment_wait_seconds=assignment_wait_seconds,
                )
            except Exception as exc:
                output_queue.put((
                    "error",
                    worker_id,
                    episode_id,
                    type(exc).__name__,
                    str(exc),
                    traceback.format_exc(),
                ))
                return
    except Exception as exc:
        output_queue.put((
            "error",
            worker_id,
            None,
            type(exc).__name__,
            str(exc),
            traceback.format_exc(),
        ))


class VectorRollout:
    """Persistent deterministic rollout workers with ordered episode IDs."""

    def __init__(
        self,
        snapshot: WorkerAssetsSnapshot,
        config: RolloutConfig,
    ) -> None:
        self.snapshot = snapshot
        self.config = config
        self._context = mp.get_context(config.start_method)
        self._input_queues: list[Any] = []
        self._output_queue: Any | None = None
        self._processes: list[mp.Process] = []
        self._next_episode_id = 0

    @property
    def processes(self) -> tuple[mp.Process, ...]:
        return tuple(self._processes)

    def start(self) -> None:
        if self._processes:
            return
        self._output_queue = self._context.Queue()
        for worker_id in range(self.config.worker_count):
            input_queue = self._context.Queue()
            process = self._context.Process(
                target=_worker_main,
                args=(
                    self.snapshot,
                    self.config,
                    worker_id,
                    input_queue,
                    self._output_queue,
                ),
                name=f"swb-rollout-{worker_id}",
            )
            process.start()
            self._input_queues.append(input_queue)
            self._processes.append(process)

    def collect(self, episode_count: int) -> tuple[EpisodeTrajectory, ...]:
        if episode_count <= 0:
            raise ValueError("episode_count must be positive")
        self.start()
        episode_ids = tuple(
            range(self._next_episode_id, self._next_episode_id + episode_count)
        )
        self._next_episode_id += episode_count
        for episode_id in episode_ids:
            worker_id = episode_id % self.config.worker_count
            self._input_queues[worker_id].put(episode_id)

        completed: dict[int, EpisodeTrajectory] = {}
        assert self._output_queue is not None
        while len(completed) < episode_count:
            try:
                message = self._output_queue.get(
                    timeout=self.config.result_timeout_seconds
                )
            except queue.Empty as exc:
                self.close()
                raise RolloutWorkerError(
                    "timed out waiting for rollout worker results"
                ) from exc
            if message[0] == "error":
                _, worker_id, episode_id, error_type, error, stack = message
                self.close()
                raise RolloutWorkerError(
                    f"worker {worker_id} failed on episode {episode_id}: "
                    f"{error_type}: {error}\n{stack}"
                )
            _, episode_id, trajectory = message
            completed[episode_id] = trajectory
        return tuple(completed[episode_id] for episode_id in episode_ids)

    def close(self, *, timeout_seconds: float = 10.0) -> None:
        for input_queue in self._input_queues:
            input_queue.put(None)
        for process in self._processes:
            process.join(timeout_seconds)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout_seconds)
        for input_queue in self._input_queues:
            input_queue.close()
        if self._output_queue is not None:
            self._output_queue.close()
        self._input_queues = []
        self._output_queue = None

    def __enter__(self) -> VectorRollout:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class PolicyVectorRollout:
    """Persistent engine workers using one central batched policy model."""

    def __init__(
        self,
        snapshot: WorkerAssetsSnapshot,
        config: RolloutConfig,
    ) -> None:
        self.snapshot = snapshot
        self.config = config
        self._context = mp.get_context(config.start_method)
        self._input_queues: list[Any] = []
        self._output_queue: Any | None = None
        self._processes: list[mp.Process] = []
        self._generation = 0
        self.last_timing: dict[str, float] = {}

    @property
    def processes(self) -> tuple[mp.Process, ...]:
        return tuple(self._processes)

    def start(self) -> None:
        if self._processes:
            return
        self._output_queue = self._context.Queue()
        for worker_id in range(self.config.worker_count):
            input_queue = self._context.Queue()
            process = self._context.Process(
                target=_policy_worker_main,
                args=(
                    self.snapshot,
                    self.config,
                    worker_id,
                    input_queue,
                    self._output_queue,
                ),
                name=f"swb-policy-rollout-{worker_id}",
            )
            process.start()
            self._input_queues.append(input_queue)
            self._processes.append(process)

    def _next_message(
        self,
        *,
        timeout_seconds: float | None = None,
        allow_timeout: bool = False,
    ):
        assert self._output_queue is not None
        try:
            message = self._output_queue.get(
                timeout=(
                    self.config.result_timeout_seconds
                    if timeout_seconds is None
                    else timeout_seconds
                )
            )
            received_at = time.perf_counter()
        except queue.Empty as exc:
            if allow_timeout:
                return None
            self.close()
            raise RolloutWorkerError(
                "timed out waiting for policy rollout worker results"
            ) from exc
        if (
            isinstance(message, tuple)
            and message
            and message[0] == _PROFILED_POLICY_REQUEST
        ):
            message = _decode_profiled_policy_request(
                message,
                received_at=received_at,
            )
        if (
            self.config.profile_central_timing
            and isinstance(message, tuple)
            and message
            and message[0] == "policy_inference"
        ):
            if len(message) == 9:
                metadata: dict[str, float] = {}
            elif len(message) == 10 and isinstance(message[9], Mapping):
                metadata = {
                    str(key): float(value)
                    for key, value in message[9].items()
                }
            else:
                raise RolloutWorkerError(
                    "central profiling received an invalid policy request"
                )
            metadata["central_received_at"] = received_at
            message = (*message[:9], metadata)
        if message[0] == "error":
            _, worker_id, episode_id, error_type, error, stack = message
            self.close()
            raise RolloutWorkerError(
                f"worker {worker_id} failed on episode {episode_id}: "
                f"{error_type}: {error}\n{stack}"
            )
        return message

    @staticmethod
    def _serve_inference_batch(
        model,
        requests: list[tuple],
        episode_records: dict[int, list[PolicyStep]],
        hidden_by_player: dict[tuple[int, int], Any],
        generators: dict[int, Any],
        *,
        profile_central_timing: bool,
        max_batch_size: int,
    ) -> dict[str, float]:
        import torch

        prepare_started = time.perf_counter()
        device = next(model.parameters()).device
        component_timing: dict[str, float] = {}
        cpu_input_started = (
            time.perf_counter() if profile_central_timing else 0.0
        )
        observations_np = np.stack([
            request[6] for request in requests
        ])
        card_indices_np = np.stack([
            request[7] for request in requests
        ])
        action_masks_np = np.stack([
            request[8] for request in requests
        ])
        if (
            card_indices_np.size
            and (
                int(card_indices_np.min()) < 0
                or int(card_indices_np.max())
                > model.card_vocabulary_size
            )
        ):
            raise RolloutWorkerError(
                "card index is outside the policy vocabulary"
            )
        if (
            action_masks_np.shape
            != (len(requests), model.action_size)
            or not bool(action_masks_np.any(axis=-1).all())
        ):
            raise RolloutWorkerError(
                "every live policy row must contain a legal action"
            )
        if profile_central_timing:
            component_timing["central_cpu_input_assembly_seconds"] = (
                time.perf_counter() - cpu_input_started
            )
            component_timing["central_cpu_input_bytes"] = float(
                observations_np.nbytes
                + card_indices_np.nbytes
                + action_masks_np.nbytes
            )
            component_timing[
                "central_cpu_repeated_stack_copy_bytes"
            ] = component_timing["central_cpu_input_bytes"]
            component_timing[
                "central_cpu_stack_copy_operations"
            ] = 3.0

            tensor_started = time.perf_counter()
            observations_cpu = torch.from_numpy(observations_np)
            card_indices_cpu = torch.from_numpy(card_indices_np).to(
                dtype=torch.long
            )
            action_masks_cpu = torch.from_numpy(action_masks_np).to(
                dtype=torch.bool
            )
            component_timing[
                "central_cpu_tensor_construction_seconds"
            ] = time.perf_counter() - tensor_started

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                h2d_start = torch.cuda.Event(enable_timing=True)
                h2d_end = torch.cuda.Event(enable_timing=True)
                h2d_start.record()
            h2d_started = time.perf_counter()
            observations = observations_cpu.to(device=device)
            card_indices = card_indices_cpu.to(device=device)
            action_masks = action_masks_cpu.to(device=device)
            if device.type == "cuda":
                h2d_end.record()
                torch.cuda.synchronize(device)
                component_timing["central_h2d_gpu_seconds"] = (
                    h2d_start.elapsed_time(h2d_end) / 1000.0
                )
            else:
                component_timing["central_h2d_gpu_seconds"] = 0.0
            component_timing["central_host_to_device_seconds"] = (
                time.perf_counter() - h2d_started
            )
        else:
            observations = torch.as_tensor(
                observations_np,
                device=device,
            )
            card_indices = torch.as_tensor(
                card_indices_np,
                dtype=torch.long,
                device=device,
            )
            action_masks = torch.as_tensor(
                action_masks_np,
                dtype=torch.bool,
                device=device,
            )
        hidden_started = time.perf_counter()
        hidden = torch.cat([
            hidden_by_player[(int(request[3]), int(request[5]))]
            for request in requests
        ])
        if profile_central_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        if profile_central_timing:
            component_timing[
                "central_hidden_state_assembly_seconds"
            ] = time.perf_counter() - hidden_started
        prepare_seconds = time.perf_counter() - prepare_started

        forward_started = time.perf_counter()
        with torch.no_grad():
            if profile_central_timing:
                (
                    (logits, values, next_hidden),
                    model_timing,
                ) = _profile_model_forward_step(
                    model,
                    observations,
                    hidden,
                    card_indices,
                )
                component_timing.update(model_timing)
                if device.type == "cuda":
                    distribution_start = torch.cuda.Event(
                        enable_timing=True
                    )
                    distribution_end = torch.cuda.Event(
                        enable_timing=True
                    )
                    distribution_start.record()
                distribution_started = time.perf_counter()
            else:
                logits, values, next_hidden = model.forward_step(
                    observations,
                    hidden,
                    card_indices,
                    validate_card_indices=False,
                )
            masked_logits = model.masked_logits(
                logits,
                action_masks,
                validate_legal_rows=False,
            )
            probabilities = torch.softmax(masked_logits, dim=-1)
            log_probs = torch.log_softmax(masked_logits, dim=-1)
            if profile_central_timing:
                if device.type == "cuda":
                    distribution_end.record()
                    torch.cuda.synchronize(device)
                    component_timing[
                        "central_masked_distribution_gpu_seconds"
                    ] = (
                        distribution_start.elapsed_time(distribution_end)
                        / 1000.0
                    )
                else:
                    component_timing[
                        "central_masked_distribution_gpu_seconds"
                    ] = 0.0
                component_timing[
                    "central_masked_distribution_seconds"
                ] = time.perf_counter() - distribution_started
        if device.type == "cuda":
            synchronize_started = time.perf_counter()
            torch.cuda.synchronize(device)
            if profile_central_timing:
                component_timing[
                    "central_post_forward_synchronize_seconds"
                ] = time.perf_counter() - synchronize_started
        forward_seconds = time.perf_counter() - forward_started

        sample_started = time.perf_counter()
        if profile_central_timing and device.type == "cuda":
            d2h_start = torch.cuda.Event(enable_timing=True)
            d2h_end = torch.cuda.Event(enable_timing=True)
            d2h_start.record()
        d2h_started = time.perf_counter()
        probabilities_cpu = probabilities.detach().cpu()
        log_probs_cpu = log_probs.detach().cpu()
        values_cpu = values.detach().cpu()
        hidden_cpu_tensor = hidden.detach().cpu()
        if profile_central_timing and device.type == "cuda":
            d2h_end.record()
            torch.cuda.synchronize(device)
            component_timing["central_d2h_gpu_seconds"] = (
                d2h_start.elapsed_time(d2h_end) / 1000.0
            )
        elif profile_central_timing:
            component_timing["central_d2h_gpu_seconds"] = 0.0
        if profile_central_timing:
            component_timing["central_device_to_host_seconds"] = (
                time.perf_counter() - d2h_started
            )
        hidden_cpu = hidden_cpu_tensor.numpy()
        sampling_started = time.perf_counter()
        actions: list[int] = []
        for row, request in enumerate(requests):
            episode_id = int(request[3])
            action_tensor = torch.multinomial(
                probabilities_cpu[row],
                1,
                generator=generators[episode_id],
            )
            actions.append(int(action_tensor.item()))
        if profile_central_timing:
            component_timing["central_sampling_seconds"] = (
                time.perf_counter() - sampling_started
            )
        sample_seconds = time.perf_counter() - sample_started

        dispatch_started = time.perf_counter()
        for row, (request, action) in enumerate(zip(requests, actions)):
            (
                _,
                worker_id,
                generation,
                episode_id,
                step_index,
                player_id,
                observation,
                cards,
                action_mask,
            ) = request[:9]
            episode_id = int(episode_id)
            player_id = int(player_id)
            step_index = int(step_index)
            if step_index != len(episode_records[episode_id]):
                raise RolloutWorkerError(
                    f"episode {episode_id} sent out-of-order policy step "
                    f"{step_index}"
                )
            if not bool(action_mask[action]):
                raise RolloutWorkerError(
                    f"central policy selected illegal action {action}"
                )
            episode_records[episode_id].append(PolicyStep(
                episode_id=episode_id,
                worker_id=int(worker_id),
                player_id=player_id,
                observation=observation,
                card_indices=cards,
                action_mask=action_mask,
                action=action,
                old_log_prob=float(log_probs_cpu[row, action].item()),
                value=float(values_cpu[row].item()),
                reward=0.0,
                hidden_before=hidden_cpu[row].copy(),
            ))
            hidden_by_player[(episode_id, player_id)] = (
                next_hidden[row : row + 1].detach()
            )
        dispatch_seconds = time.perf_counter() - dispatch_started
        timing = {
            "central_batch_prepare_to_device_seconds": prepare_seconds,
            "central_forward_seconds": forward_seconds,
            "central_device_to_host_and_sample_seconds": sample_seconds,
            "central_record_packaging_seconds": dispatch_seconds,
            "central_inference_requests": float(len(requests)),
            "central_inference_batches": 1.0,
            "central_max_batch_size": float(len(requests)),
            f"central_batch_size_{len(requests)}_count": 1.0,
            "central_batch_capacity_slots": float(max_batch_size),
            "central_batch_empty_slots": float(
                max_batch_size - len(requests)
            ),
        }
        timing.update(component_timing)
        if profile_central_timing:
            timing["central_gpu_busy_seconds"] = (
                component_timing.get("central_h2d_gpu_seconds", 0.0)
                + component_timing.get(
                    "central_model_forward_seconds", 0.0
                )
                + component_timing.get(
                    "central_masked_distribution_gpu_seconds", 0.0
                )
                + component_timing.get("central_d2h_gpu_seconds", 0.0)
            )
            timing["central_profiled_batches"] = 1.0
            timing["central_profiled_cuda_batches"] = float(
                device.type == "cuda"
            )
        ipc_timings = [
            request[9]
            for request in requests
            if (
                len(request) == 10
                and isinstance(request[9], Mapping)
                and "send_seconds" in request[9]
            )
        ]
        if ipc_timings:
            timing["worker_ipc_request_send_seconds"] = sum(
                float(metadata["send_seconds"])
                for metadata in ipc_timings
            )
        return timing

    @staticmethod
    def _bootstrap_values(
        model,
        episode_id: int,
        bootstrap_inputs: Mapping[
            int,
            tuple[np.ndarray, np.ndarray],
        ],
        hidden_by_player: Mapping[tuple[int, int], Any],
    ) -> tuple[dict[tuple[int, int], float], float]:
        import torch

        started = time.perf_counter()
        players = sorted(bootstrap_inputs)
        device = next(model.parameters()).device
        card_indices_np = np.stack([
            bootstrap_inputs[player][1]
            for player in players
        ])
        if (
            card_indices_np.size
            and (
                int(card_indices_np.min()) < 0
                or int(card_indices_np.max())
                > model.card_vocabulary_size
            )
        ):
            raise RolloutWorkerError(
                "card index is outside the policy vocabulary"
            )
        observations = torch.as_tensor(
            np.stack([
                bootstrap_inputs[player][0]
                for player in players
            ]),
            device=device,
        )
        card_indices = torch.as_tensor(
            card_indices_np,
            dtype=torch.long,
            device=device,
        )
        hidden = torch.cat([
            hidden_by_player[(episode_id, player)]
            for player in players
        ])
        with torch.no_grad():
            _, values, _ = model.forward_step(
                observations,
                hidden,
                card_indices,
                validate_card_indices=False,
            )
        values_cpu = values.detach().cpu()
        return (
            {
                (episode_id, player): float(values_cpu[row].item())
                for row, player in enumerate(players)
            },
            time.perf_counter() - started,
        )

    def collect(self, model, episode_ids: tuple[int, ...]) -> tuple[PolicyEpisode, ...]:
        if not episode_ids:
            raise ValueError("episode_ids must not be empty")
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("episode_ids must be unique")
        if len(episode_ids) > self.config.worker_count:
            raise ValueError(
                "one central-policy collection can assign at most one "
                "episode per worker"
            )
        collect_started = time.perf_counter()
        rollout_startup_started = time.perf_counter()
        self.start()
        rollout_startup_seconds = (
            time.perf_counter() - rollout_startup_started
        )
        collection_setup_started = time.perf_counter()
        self._generation += 1
        generation = self._generation
        import torch

        device = next(model.parameters()).device
        episode_records: dict[int, list[PolicyStep]] = {
            episode_id: [] for episode_id in episode_ids
        }
        hidden_by_player = {
            (episode_id, player_id): model.initial_state(1, device=device)
            for episode_id in episode_ids
            for player_id in (0, 1)
        }
        generators = {}
        for episode_id in episode_ids:
            worker_id = episode_id % self.config.worker_count
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                episode_seeds(
                    self.config.master_seed,
                    worker_id,
                    episode_id,
                ).policy_seed
            )
            generators[episode_id] = generator

        was_training = model.training
        model.eval()
        collection_setup_seconds = (
            time.perf_counter() - collection_setup_started
        )
        episode_dispatch_started = time.perf_counter()
        for episode_id in episode_ids:
            worker_id = episode_id % self.config.worker_count
            self._input_queues[worker_id].put((
                "episode", generation, episode_id
            ))
        episode_dispatch_seconds = (
            time.perf_counter() - episode_dispatch_started
        )
        episode_wait_started = time.perf_counter()
        completed: dict[int, PolicyEpisode] = {}
        pending_messages: deque[tuple] = deque()
        central_timing: dict[str, float] = {}
        batch_wait_seconds = 0.0
        worker_message_wait_seconds = 0.0
        queue_to_batch_wait_seconds = 0.0
        episode_completion_seconds = 0.0
        model_restore_seconds = 0.0
        try:
            while len(completed) < len(episode_ids):
                if pending_messages:
                    message = pending_messages.popleft()
                else:
                    message_wait_started = time.perf_counter()
                    message = self._next_message()
                    worker_message_wait_seconds += (
                        time.perf_counter() - message_wait_started
                    )
                assert message is not None
                if message[0] == "policy_inference":
                    requests = [message]
                    wait_started = time.perf_counter()
                    deadline = (
                        wait_started
                        + self.config.central_inference_batch_wait_seconds
                    )
                    while len(requests) < self.config.worker_count:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        candidate = self._next_message(
                            timeout_seconds=remaining,
                            allow_timeout=True,
                        )
                        if candidate is None:
                            break
                        if candidate[0] == "policy_inference":
                            requests.append(candidate)
                        else:
                            pending_messages.append(candidate)
                    batch_wait_seconds += time.perf_counter() - wait_started
                    batch_formed_at = time.perf_counter()
                    if self.config.profile_central_timing:
                        for request in requests:
                            if (
                                len(request) != 10
                                or not isinstance(request[9], Mapping)
                                or "central_received_at" not in request[9]
                            ):
                                raise RolloutWorkerError(
                                    "central request lacks queue timing"
                                )
                            queue_to_batch_wait_seconds += max(
                                0.0,
                                batch_formed_at
                                - float(
                                    request[9]["central_received_at"]
                                ),
                            )
                    batch_timing = self._serve_inference_batch(
                        model,
                        requests,
                        episode_records,
                        hidden_by_player,
                        generators,
                        profile_central_timing=(
                            self.config.profile_central_timing
                        ),
                        max_batch_size=self.config.worker_count,
                    )
                    dispatch_started = time.perf_counter()
                    for request in requests:
                        worker_id = int(request[1])
                        episode_id = int(request[3])
                        step_index = int(request[4])
                        action = episode_records[episode_id][-1].action
                        request_received_at = (
                            float(request[9]["received_at"])
                            if (
                                len(request) == 10
                                and isinstance(request[9], Mapping)
                                and "received_at" in request[9]
                            )
                            else None
                        )
                        response = (
                            "policy_action",
                            generation,
                            episode_id,
                            step_index,
                            action,
                        )
                        if request_received_at is not None:
                            response = (*response, request_received_at)
                        self._input_queues[worker_id].put(response)
                    batch_timing["central_response_dispatch_seconds"] = (
                        time.perf_counter() - dispatch_started
                    )
                    for key, value in batch_timing.items():
                        if key == "central_max_batch_size":
                            central_timing[key] = max(
                                central_timing.get(key, 0.0),
                                float(value),
                            )
                        else:
                            central_timing[key] = (
                                central_timing.get(key, 0.0)
                                + float(value)
                            )
                    continue
                if message[0] != "policy_episode_end":
                    self.close()
                    raise RolloutWorkerError(
                        f"unexpected policy worker message {message[0]!r}"
                    )
                episode_completion_started = time.perf_counter()
                (
                    _,
                    episode_id,
                    worker_id,
                    requested_generation,
                    boundary,
                    winner,
                    bootstrap_inputs,
                    matchup,
                    worker_timing,
                ) = message
                episode_id = int(episode_id)
                if int(requested_generation) != generation:
                    raise RolloutWorkerError(
                        "worker completed the wrong policy generation"
                    )
                records = episode_records[episode_id]
                if int(worker_timing["worker_agent_steps"]) != len(records):
                    raise RolloutWorkerError(
                        "worker and central policy record counts disagree"
                    )
                bootstrap: dict[tuple[int, int], float]
                bootstrap_seconds = 0.0
                if boundary == "terminated":
                    bootstrap = {
                        (episode_id, player_id): 0.0
                        for player_id in (0, 1)
                    }
                    for player_id in (0, 1):
                        indices = [
                            index
                            for index, record in enumerate(records)
                            if record.player_id == player_id
                        ]
                        if indices:
                            records[indices[-1]].reward = (
                                0.0
                                if winner is None
                                else (
                                    1.0
                                    if int(winner) == player_id
                                    else -1.0
                                )
                            )
                else:
                    bootstrap, bootstrap_seconds = self._bootstrap_values(
                        model,
                        episode_id,
                        bootstrap_inputs,
                        hidden_by_player,
                    )
                central_timing["central_bootstrap_seconds"] = (
                    central_timing.get("central_bootstrap_seconds", 0.0)
                    + bootstrap_seconds
                )
                completed[episode_id] = PolicyEpisode(
                    episode_id=episode_id,
                    worker_id=int(worker_id),
                    records=tuple(records),
                    bootstrap=bootstrap,
                    boundary=str(boundary),
                    winner=None if winner is None else int(winner),
                    matchup=matchup,
                    timing=worker_timing,
                )
                episode_completion_seconds += max(
                    0.0,
                    time.perf_counter()
                    - episode_completion_started
                    - bootstrap_seconds,
                )
        except Exception:
            self.close()
            raise
        finally:
            model_restore_started = time.perf_counter()
            model.train(was_training)
            model_restore_seconds = (
                time.perf_counter() - model_restore_started
            )
        episode_wait_seconds = time.perf_counter() - episode_wait_started
        collection_finalize_started = time.perf_counter()
        ordered = tuple(completed[episode_id] for episode_id in episode_ids)
        collection_finalize_seconds = (
            time.perf_counter() - collection_finalize_started
        )
        timing = {
            "central_rollout_startup_seconds": rollout_startup_seconds,
            "central_collection_setup_seconds": collection_setup_seconds,
            "episode_dispatch_seconds": episode_dispatch_seconds,
            "episode_wait_seconds": episode_wait_seconds,
            "central_batch_wait_seconds": batch_wait_seconds,
            "central_worker_message_wait_seconds": (
                worker_message_wait_seconds
            ),
            "central_queue_to_batch_wait_seconds": (
                queue_to_batch_wait_seconds
            ),
            "central_episode_completion_seconds": (
                episode_completion_seconds
            ),
            "central_model_restore_seconds": model_restore_seconds,
            "central_collection_finalize_seconds": (
                collection_finalize_seconds
            ),
            "collect_call_total_seconds": (
                time.perf_counter() - collect_started
            ),
            "policy_payload_bytes": 0.0,
            "policy_transmitted_bytes": 0.0,
            "episodes": float(len(ordered)),
        }
        timing.update(central_timing)
        if self.config.profile_central_timing:
            gpu_busy_seconds = timing.get(
                "central_gpu_busy_seconds", 0.0
            )
            gpu_waiting_for_worker_seconds = (
                worker_message_wait_seconds + batch_wait_seconds
            )
            busy_plus_wait = (
                gpu_busy_seconds + gpu_waiting_for_worker_seconds
            )
            timing.update({
                "central_gpu_waiting_for_worker_seconds": (
                    gpu_waiting_for_worker_seconds
                ),
                "central_gpu_busy_fraction_of_busy_plus_worker_wait": (
                    gpu_busy_seconds / max(busy_plus_wait, 1e-12)
                ),
                "central_gpu_worker_wait_fraction_of_busy_plus_worker_wait": (
                    gpu_waiting_for_worker_seconds
                    / max(busy_plus_wait, 1e-12)
                ),
                "central_result_distribution_seconds": (
                    timing.get(
                        "central_record_packaging_seconds", 0.0
                    )
                    + timing.get(
                        "central_response_dispatch_seconds", 0.0
                    )
                ),
            })
        for episode in ordered:
            for key, value in episode.timing.items():
                if key in {
                    "worker_torch_threads",
                    "worker_long_episode_threshold_steps",
                }:
                    timing[key] = max(
                        timing.get(key, 0.0),
                        float(value),
                    )
                else:
                    timing[key] = timing.get(key, 0.0) + float(value)
        self.last_timing = timing
        return ordered

    def close(self, *, timeout_seconds: float = 10.0) -> None:
        for input_queue in self._input_queues:
            input_queue.put(None)
        for process in self._processes:
            process.join(timeout_seconds)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout_seconds)
        for input_queue in self._input_queues:
            input_queue.close()
        if self._output_queue is not None:
            self._output_queue.close()
        self._input_queues = []
        self._output_queue = None

    def __enter__(self) -> PolicyVectorRollout:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
