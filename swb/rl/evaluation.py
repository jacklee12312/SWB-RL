from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from swb.engine.commands import ChoiceKind
from swb.engine.deck import CLASS_NAMES
from swb.engine.environment import (
    MATCH_SETUP_OFFICIAL,
    MATCH_SETUP_VALUES,
    ShadowverseEnv,
)
from swb.engine.events import EventType
from swb.rl.action_guard import FusionCancelActionGuard
from swb.rl.checkpoint import load_checkpoint
from swb.rl.class_schedule import ALL_CLASS_IDS, normalize_class_ids
from swb.rl.fixed_decks import get_fixed_training_deck
from swb.rl.policy import MaskedPolicyNetwork
from swb.rl.ppo import ObservationFlattener, PPOTrainer
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.seeding import derive_seed
from swb.rl.versioning import ExperimentVersions, stable_json_sha256


@dataclass(frozen=True)
class EvaluationConfig:
    master_seed: int = 20260721
    seed_count: int = 2
    max_agent_steps: int = 512
    opponent_kind: str = "random_legal"
    opponent_checkpoint: str | None = None
    class_ids: tuple[int, ...] = ALL_CLASS_IDS
    match_setup: str = MATCH_SETUP_OFFICIAL
    training_deck: str | None = None
    full_matchup_matrix: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "class_ids",
            normalize_class_ids(self.class_ids),
        )
        if self.seed_count <= 0 or self.max_agent_steps <= 0:
            raise ValueError("seed_count and max_agent_steps must be positive")
        if self.opponent_kind not in {
            "current",
            "historical",
            "random_legal",
            "fixed",
        }:
            raise ValueError(f"unsupported opponent kind {self.opponent_kind!r}")
        if self.opponent_kind == "historical" and not self.opponent_checkpoint:
            raise ValueError("historical evaluation requires opponent_checkpoint")
        if self.match_setup not in MATCH_SETUP_VALUES:
            raise ValueError("match_setup must be 'legacy' or 'official'")
        if self.training_deck is not None:
            fixed_deck = get_fixed_training_deck(self.training_deck)
            if self.class_ids != (fixed_deck.class_id,):
                raise ValueError(
                    f"training_deck {self.training_deck!r} requires "
                    f"class_ids=({fixed_deck.class_id},)"
                )
            if self.full_matchup_matrix:
                raise ValueError(
                    "full_matchup_matrix cannot use one fixed training deck"
                )


def _evaluation_class_matchups(
    config: EvaluationConfig,
) -> tuple[tuple[int, int], ...]:
    if config.full_matchup_matrix:
        return tuple(
            (learner_class, opponent_class)
            for learner_class in config.class_ids
            for opponent_class in config.class_ids
        )
    return tuple((class_id, class_id) for class_id in config.class_ids)


class _Policy:
    def __init__(self) -> None:
        self.fusion_cancel_guard = FusionCancelActionGuard()

    def reset(self) -> None:
        self.fusion_cancel_guard.reset()

    def action(
        self,
        env: ShadowverseEnv,
        player_id: int,
        action_mask: np.ndarray,
    ) -> int:
        raise NotImplementedError


class _RandomLegalPolicy(_Policy):
    def __init__(self, seed: int):
        super().__init__()
        self.rng = random.Random(seed)

    def action(self, env, player_id, action_mask) -> int:
        policy_mask = self.fusion_cancel_guard.policy_mask(
            env, player_id, action_mask
        )
        legal = np.flatnonzero(policy_mask)
        action = int(legal[self.rng.randrange(legal.size)])
        self.fusion_cancel_guard.record_selected_action(
            env, player_id, action
        )
        return action


class _FirstLegalPolicy(_Policy):
    def action(self, env, player_id, action_mask) -> int:
        policy_mask = self.fusion_cancel_guard.policy_mask(
            env, player_id, action_mask
        )
        action = int(np.flatnonzero(policy_mask)[0])
        self.fusion_cancel_guard.record_selected_action(
            env, player_id, action
        )
        return action


class _RecurrentPolicy(_Policy):
    def __init__(
        self,
        model: MaskedPolicyNetwork,
        flattener: ObservationFlattener,
        device: torch.device,
        observation_version: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.flattener = flattener
        self.device = device
        self.observation_version = observation_version
        self.hidden: dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        super().reset()
        self.hidden = {
            player: self.model.initial_state(1, device=self.device)
            for player in (0, 1)
        }

    def action(self, env, player_id, action_mask) -> int:
        policy_mask = self.fusion_cancel_guard.policy_mask(
            env, player_id, action_mask
        )
        observation = _policy_observation(
            env,
            observation_version=self.observation_version,
            perspective=player_id,
            action_mask=policy_mask,
        )
        vector = torch.from_numpy(self.flattener.encode(observation)).to(
            self.device
        ).unsqueeze(0)
        card_indices = torch.from_numpy(
            self.flattener.encode_cards(observation)
        ).to(self.device).unsqueeze(0)
        mask = torch.from_numpy(policy_mask).to(
            self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _, hidden = self.model.forward_step(
                vector, self.hidden[player_id], card_indices
            )
            masked = self.model.masked_logits(logits, mask)
        self.hidden[player_id] = hidden
        action = int(masked.argmax(dim=-1).item())
        self.fusion_cancel_guard.record_selected_action(
            env, player_id, action
        )
        return action


def _policy_observation(
    env: ShadowverseEnv,
    *,
    observation_version: str,
    perspective: int,
    action_mask: np.ndarray,
) -> object:
    """Encode one shared engine state in the policy's own schema.

    Cross-version evaluation must not force both frozen policies through the
    learner's observation encoder. Calling the encoder directly also avoids
    mutating the live environment's configured schema or its observation
    cache.
    """
    if observation_version == env.observation_version:
        return env.observation(
            perspective=perspective,
            action_mask=action_mask,
        )
    if observation_version == "v3":
        from swb.engine.observation_v3 import encode_observation_v3

        return encode_observation_v3(
            env,
            perspective=perspective,
            action_mask=action_mask,
            open_decklists=env.open_decklists,
        )
    if observation_version == "v4":
        from swb.engine.observation_v4 import encode_observation_v4

        return encode_observation_v4(
            env,
            perspective=perspective,
            action_mask=action_mask,
            open_decklists=env.open_decklists,
        )
    if observation_version == "v4.1":
        from swb.engine.observation_v4_1 import encode_observation_v4_1

        return encode_observation_v4_1(
            env,
            perspective=perspective,
            action_mask=action_mask,
            open_decklists=env.open_decklists,
        )
    raise ValueError(
        f"unsupported recurrent-policy observation version "
        f"{observation_version!r}"
    )


def _wilson_interval(wins: float, games: int) -> tuple[float, float]:
    if games == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = wins / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denominator
    margin = z * math.sqrt(
        (p * (1.0 - p) + z * z / (4.0 * games)) / games
    ) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def _opponent_policy(
    config: EvaluationConfig,
    trainer: PPOTrainer,
    snapshot: WorkerAssetsSnapshot,
    seed: int,
    historical_trainer: PPOTrainer | None = None,
) -> _Policy:
    if config.opponent_kind == "random_legal":
        return _RandomLegalPolicy(seed)
    if config.opponent_kind == "fixed":
        return _FirstLegalPolicy()
    if config.opponent_kind == "current":
        return _RecurrentPolicy(
            trainer.model,
            trainer.flattener,
            trainer.device,
            trainer.config.observation_version,
        )
    historical = historical_trainer
    if historical is None:
        historical = load_checkpoint(
            Path(config.opponent_checkpoint),
            snapshot,
            device=str(trainer.device),
            restore_rng_state=False,
        )
    return _RecurrentPolicy(
        historical.model,
        historical.flattener,
        historical.device,
        historical.config.observation_version,
    )


def _deck_manifest(
    *,
    class_id: int,
    deck_index: int,
    role: str,
    seed: int,
    deck,
) -> dict[str, object]:
    card_ids = tuple(card.card_id for card in deck)
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "deck_index": deck_index,
        "role": role,
        "seed": seed,
        "card_ids": list(card_ids),
        "composition_sha256": stable_json_sha256(tuple(sorted(card_ids))),
    }


def _aggregate_metrics(results: list[dict[str, object]]) -> dict[str, object]:
    games = len(results)
    score = sum(float(result["score"]) for result in results)
    win_rate = score / games
    confidence = _wilson_interval(score, games)
    clamped = min(1.0 - 1e-6, max(1e-6, win_rate))
    side_rates = {}
    for side in (0, 1):
        side_results = [
            result for result in results if result["learner_player"] == side
        ]
        side_rates[str(side)] = (
            sum(float(result["score"]) for result in side_results)
            / len(side_results)
        )
    mask_checks = sum(int(result["action_mask_checks"]) for result in results)
    mask_mismatches = sum(
        int(result["action_mask_mismatches"]) for result in results
    )
    illegal_actions = sum(int(result["illegal_actions"]) for result in results)
    suppressed_decisions = sum(
        int(result["fusion_retry_suppressed_decisions"])
        for result in results
    )
    suppressed_actions = sum(
        int(result["fusion_retry_suppressed_actions"])
        for result in results
    )
    empty_extra_pp_decisions = sum(
        int(result["empty_extra_pp_suppressed_decisions"])
        for result in results
    )
    empty_extra_pp_actions = sum(
        int(result["empty_extra_pp_suppressed_actions"])
        for result in results
    )
    return {
        "games": games,
        "win_rate": win_rate,
        "side_win_rates": side_rates,
        "confidence_interval_95": confidence,
        "elo_relative": 400.0 * math.log10(clamped / (1.0 - clamped)),
        "average_turn": sum(int(result["turn"]) for result in results) / games,
        "average_agent_steps": (
            sum(int(result["agent_steps"]) for result in results) / games
        ),
        "terminated": sum(bool(result["terminated"]) for result in results),
        "truncated": sum(bool(result["truncated"]) for result in results),
        "terminated_rate": (
            sum(bool(result["terminated"]) for result in results) / games
        ),
        "truncated_rate": (
            sum(bool(result["truncated"]) for result in results) / games
        ),
        "illegal_actions": illegal_actions,
        "illegal_action_rate": illegal_actions / max(1, mask_checks),
        "action_mask_checks": mask_checks,
        "action_mask_mismatches": mask_mismatches,
        "fusion_retry_suppressed_decisions": suppressed_decisions,
        "fusion_retry_suppressed_actions": suppressed_actions,
        "empty_extra_pp_suppressed_decisions": empty_extra_pp_decisions,
        "empty_extra_pp_suppressed_actions": empty_extra_pp_actions,
    }


def _observe_state(
    env: ShadowverseEnv,
    visited_cards: set[int],
    visited_classes: set[int],
    resource_maxima: dict[str, int],
) -> None:
    for player in env._core.players:
        visited_classes.add(player.class_id)
        visited_cards.update(card.card_id for card in player.hand)
        visited_cards.update(
            entity.definition.card_id for entity in player.board
        )
        visited_cards.update(
            card.definition.card_id for card in player.graveyard
        )
        visited_cards.update(card.card_id for card in player.banished)
        visited_cards.update(
            material.definition.card_id for material in player.fusion_materials
        )
        values = {
            "shadows": player.shadows,
            "cooperation": player.cooperation,
            "earth_sigils": player.earth_sigils,
            "faith_count": len(player.faiths),
            "faith_value": sum(faith.value for faith in player.faiths),
            "emblem_count": len(player.emblems),
            "emblem_activations": sum(
                sum(emblem.activation_counts.values())
                for emblem in player.emblems
            ),
            "fusion_material_count": len(player.fusion_materials),
            "spellboosted_hand_cards": sum(
                card.spellboost_count > 0 for card in player.hand
            ),
            "followers_evolved": player.followers_evolved_this_match,
        }
        for name, value in values.items():
            resource_maxima[name] = max(resource_maxima.get(name, 0), value)


def evaluate(
    trainer: PPOTrainer,
    snapshot: WorkerAssetsSnapshot,
    config: EvaluationConfig | None = None,
) -> dict[str, object]:
    config = config or EvaluationConfig()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state()
    generator_rng = trainer.torch_generator.get_state().clone()
    was_training = trainer.model.training
    trainer.model.eval()
    results: list[dict[str, object]] = []
    deck_manifests: list[dict[str, object]] = []
    deck_card_ids: set[int] = set()
    visited_cards: set[int] = set()
    visited_classes: set[int] = set()
    visited_mechanisms: set[str] = set()
    resource_maxima: dict[str, int] = {}
    fixed_deck = (
        None
        if config.training_deck is None
        else get_fixed_training_deck(config.training_deck)
    )
    historical_trainer = None
    opponent_checkpoint_sha256 = None
    if config.opponent_kind == "historical":
        opponent_checkpoint_sha256 = hashlib.sha256(
            Path(config.opponent_checkpoint).read_bytes()
        ).hexdigest()
    try:
        if config.opponent_kind == "historical":
            historical_trainer = load_checkpoint(
                Path(config.opponent_checkpoint),
                snapshot,
                device=str(trainer.device),
                restore_rng_state=False,
            )
            historical_trainer.model.eval()
        for class_id, opponent_class_id in _evaluation_class_matchups(
            config
        ):
            for deck_index in range(config.seed_count):
                deck_seed = (
                    derive_seed(
                        config.master_seed,
                        "evaluation_decks",
                        class_id,
                        opponent_class_id,
                        deck_index,
                    )
                    if config.full_matchup_matrix
                    else derive_seed(
                        config.master_seed,
                        "evaluation_decks",
                        class_id,
                        deck_index,
                    )
                )
                learner_deck_seed = derive_seed(deck_seed, "learner")
                opponent_deck_seed = derive_seed(deck_seed, "opponent")
                if fixed_deck is None:
                    learner_deck = snapshot.catalog.sample_deck(
                        class_id,
                        random.Random(learner_deck_seed),
                    )
                    opponent_deck = snapshot.catalog.sample_deck(
                        opponent_class_id,
                        random.Random(opponent_deck_seed),
                    )
                else:
                    learner_deck = fixed_deck.build(snapshot.catalog)
                    opponent_deck = fixed_deck.build(snapshot.catalog)
                learner_manifest = _deck_manifest(
                    class_id=class_id,
                    deck_index=deck_index,
                    role="learner",
                    seed=learner_deck_seed,
                    deck=learner_deck,
                )
                opponent_manifest = _deck_manifest(
                    class_id=opponent_class_id,
                    deck_index=deck_index,
                    role="opponent",
                    seed=opponent_deck_seed,
                    deck=opponent_deck,
                )
                deck_manifests.extend((learner_manifest, opponent_manifest))
                deck_card_ids.update(card.card_id for card in learner_deck)
                deck_card_ids.update(card.card_id for card in opponent_deck)

                for learner_player in (0, 1):
                    if learner_player == 0:
                        decks = (learner_deck, opponent_deck)
                    else:
                        decks = (opponent_deck, learner_deck)
                    engine_seed = (
                        derive_seed(
                            config.master_seed,
                            "evaluation_engine",
                            class_id,
                            opponent_class_id,
                            deck_index,
                            learner_player,
                        )
                        if config.full_matchup_matrix
                        else derive_seed(
                            config.master_seed,
                            "evaluation_engine",
                            class_id,
                            deck_index,
                            learner_player,
                        )
                    )
                    class_a, class_b = (
                        (class_id, opponent_class_id)
                        if learner_player == 0
                        else (opponent_class_id, class_id)
                    )
                    env = ShadowverseEnv(
                        decks[0],
                        decks[1],
                        class_a=class_a,
                        class_b=class_b,
                        seed=engine_seed,
                        rulebook=trainer.assets.rulebook,
                        card_resolver=trainer.assets.catalog.resolve,
                        observation_version=trainer.config.observation_version,
                        card_vocabulary=trainer.assets.catalog.card_vocabulary,
                        max_agent_steps=config.max_agent_steps,
                        training_mode=True,
                        training_event_history_limit=4096,
                        validate_invariants=True,
                        match_setup=config.match_setup,
                    )
                    _, info = env.reset(seed=engine_seed)
                    learner_policy = _RecurrentPolicy(
                        trainer.model,
                        trainer.flattener,
                        trainer.device,
                        trainer.config.observation_version,
                    )
                    opponent = _opponent_policy(
                        config,
                        trainer,
                        snapshot,
                        derive_seed(engine_seed, "opponent_policy"),
                        historical_trainer,
                    )
                    learner_policy.reset()
                    opponent.reset()
                    steps = 0
                    mask_checks = 0
                    mask_mismatches = 0
                    illegal_actions = 0
                    _observe_state(
                        env,
                        visited_cards,
                        visited_classes,
                        resource_maxima,
                    )
                    while not env.terminated and not env.truncated:
                        player = env.decision_player
                        reported_mask = np.asarray(
                            info["action_mask"], dtype=np.int8
                        )
                        executable_mask = np.asarray(
                            env.action_mask(), dtype=np.int8
                        )
                        mask_checks += 1
                        if not np.array_equal(reported_mask, executable_mask):
                            mask_mismatches += 1
                        policy = (
                            learner_policy
                            if player == learner_player
                            else opponent
                        )
                        action = policy.action(env, player, reported_mask)
                        if (
                            action < 0
                            or action >= env.ACTION_SIZE
                            or not reported_mask[action]
                        ):
                            illegal_actions += 1
                            raise RuntimeError(
                                "evaluation policy selected an illegal action"
                            )
                        transition = env.step(action)
                        info = transition.info
                        steps += 1
                        _observe_state(
                            env,
                            visited_cards,
                            visited_classes,
                            resource_maxima,
                        )
                        pending = env._core.state.pending_choice
                        if pending is not None:
                            visited_mechanisms.add(
                                f"choice:{pending.choice_kind.value}"
                            )
                    visited_mechanisms.update(
                        event.type.value for event in env._core.event_history
                    )
                    score = (
                        0.5
                        if env.winner is None
                        else (
                            1.0 if env.winner == learner_player else 0.0
                        )
                    )
                    results.append({
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "learner_class_id": class_id,
                        "learner_class_name": CLASS_NAMES[class_id],
                        "opponent_class_id": opponent_class_id,
                        "opponent_class_name": CLASS_NAMES[
                            opponent_class_id
                        ],
                        "deck_index": deck_index,
                        "learner_player": learner_player,
                        "score": score,
                        "winner": env.winner,
                        "turn": env.turn,
                        "agent_steps": steps,
                        "terminated": env.terminated,
                        "truncated": env.truncated,
                        "engine_seed": engine_seed,
                        "learner_deck_sha256": learner_manifest[
                            "composition_sha256"
                        ],
                        "opponent_deck_sha256": opponent_manifest[
                            "composition_sha256"
                        ],
                        "action_mask_checks": mask_checks,
                        "action_mask_mismatches": mask_mismatches,
                        "illegal_actions": illegal_actions,
                        "fusion_retry_suppressed_decisions": (
                            learner_policy.fusion_cancel_guard.suppressed_decisions
                            + opponent.fusion_cancel_guard.suppressed_decisions
                        ),
                        "fusion_retry_suppressed_actions": (
                            learner_policy.fusion_cancel_guard.suppressed_actions
                            + opponent.fusion_cancel_guard.suppressed_actions
                        ),
                        "empty_extra_pp_suppressed_decisions": (
                            learner_policy.fusion_cancel_guard.extra_pp_suppressed_decisions
                            + opponent.fusion_cancel_guard.extra_pp_suppressed_decisions
                        ),
                        "empty_extra_pp_suppressed_actions": (
                            learner_policy.fusion_cancel_guard.extra_pp_suppressed_actions
                            + opponent.fusion_cancel_guard.extra_pp_suppressed_actions
                        ),
                    })
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        trainer.torch_generator.set_state(generator_rng)
        trainer.model.train(was_training)
        if historical_trainer is not None:
            historical_trainer.close()

    mechanism_universe = {
        *(event_type.value for event_type in EventType),
        *(f"choice:{choice_kind.value}" for choice_kind in ChoiceKind),
    }
    visited_known_mechanisms = visited_mechanisms & mechanism_universe
    exact_ids = set(snapshot.catalog.exact_collectible_ids)
    visited_exact_ids = visited_cards & exact_ids
    sampled_exact_ids = deck_card_ids & exact_ids
    class_metrics = {}
    matchup_metrics = {}
    class_coverage = {}
    for learner_class_id, opponent_class_id in _evaluation_class_matchups(
        config
    ):
        matchup_results = [
            result
            for result in results
            if (
                result["learner_class_id"] == learner_class_id
                and result["opponent_class_id"] == opponent_class_id
            )
        ]
        matchup_metrics[
            f"{learner_class_id}_vs_{opponent_class_id}"
        ] = {
            "learner_class_id": learner_class_id,
            "learner_class_name": CLASS_NAMES[learner_class_id],
            "opponent_class_id": opponent_class_id,
            "opponent_class_name": CLASS_NAMES[opponent_class_id],
            **_aggregate_metrics(matchup_results),
        }
    for class_id in config.class_ids:
        class_results = [
            result for result in results if result["class_id"] == class_id
        ]
        class_metrics[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            **_aggregate_metrics(class_results),
        }
        eligible_ids = {
            card.card_id
            for card in snapshot.catalog.pool(class_id=class_id)
        }
        seen_ids = visited_exact_ids & eligible_ids
        sampled_ids = sampled_exact_ids & eligible_ids
        class_coverage[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "eligible_exact_count": len(eligible_ids),
            "sampled_exact_count": len(sampled_ids),
            "sampled_exact_rate": (
                len(sampled_ids) / max(1, len(eligible_ids))
            ),
            "encountered_exact_count": len(seen_ids),
            "encountered_exact_rate": (
                len(seen_ids) / max(1, len(eligible_ids))
            ),
            "unsampled_exact_card_ids": sorted(eligible_ids - sampled_ids),
            "unencountered_exact_card_ids": sorted(eligible_ids - seen_ids),
        }
    versions = ExperimentVersions.capture(
        trainer.env,
        snapshot.catalog,
        rulebook_sha256=snapshot.rulebook_sha256,
    ).to_dict()
    configuration = {
        "master_seed": config.master_seed,
        "seed_count": config.seed_count,
        "deck_pairs_per_class": config.seed_count,
        "deck_pairs_per_matchup": config.seed_count,
        "class_ids": list(config.class_ids),
        "class_names": [CLASS_NAMES[class_id] for class_id in config.class_ids],
        "mirrored_games": len(results),
        "full_matchup_matrix": config.full_matchup_matrix,
        "matchup_count": len(_evaluation_class_matchups(config)),
        "opponent_kind": config.opponent_kind,
        "opponent_checkpoint": config.opponent_checkpoint,
        "max_agent_steps": config.max_agent_steps,
        "match_setup": config.match_setup,
        "validate_invariants": True,
        "training_deck": (
            None if fixed_deck is None else fixed_deck.manifest()
        ),
        "policy_architecture": trainer.model.architecture,
        "model_parameters": sum(
            parameter.numel() for parameter in trainer.model.parameters()
        ),
        "policy_action_guard": (
            "fusion-cancel-retry-v1+empty-extra-pp-v1"
        ),
    }
    if opponent_checkpoint_sha256 is not None:
        configuration["opponent_checkpoint_sha256"] = (
            opponent_checkpoint_sha256
        )
    report = {
        "schema_version": 2,
        "purpose": (
            "fixed-seed mirrored class-matchup evaluation; not a "
            "standalone policy-strength claim"
        ),
        "configuration": configuration,
        "versions": versions,
        "metrics": {
            **_aggregate_metrics(results),
            "per_class": class_metrics,
            "per_matchup": matchup_metrics,
        },
        "coverage": {
            "card_ids": sorted(visited_cards),
            "card_count": len(visited_cards),
            "exact_card_ids": sorted(visited_exact_ids),
            "exact_card_count": len(visited_exact_ids),
            "card_coverage_rate": (
                len(visited_exact_ids) / max(1, len(exact_ids))
            ),
            "deck_exact_card_ids": sorted(sampled_exact_ids),
            "deck_exact_card_count": len(sampled_exact_ids),
            "deck_card_coverage_rate": (
                len(sampled_exact_ids) / max(1, len(exact_ids))
            ),
            "unsampled_exact_card_ids": sorted(exact_ids - sampled_exact_ids),
            "unencountered_exact_card_ids": sorted(
                exact_ids - visited_exact_ids
            ),
            "classes": sorted(visited_classes),
            "class_coverage_rate": len(visited_classes) / 7.0,
            "per_class": class_coverage,
            "mechanisms": sorted(visited_mechanisms),
            "known_mechanism_count": len(mechanism_universe),
            "mechanism_coverage_rate": (
                len(visited_known_mechanisms) / max(1, len(mechanism_universe))
            ),
            "resource_maxima": dict(sorted(resource_maxima.items())),
            "active_resources": sorted(
                name for name, value in resource_maxima.items() if value > 0
            ),
            "inactive_resources": sorted(
                name for name, value in resource_maxima.items() if value == 0
            ),
        },
        "decks": deck_manifests,
        "games": results,
    }
    report["evaluation_suite_sha256"] = stable_json_sha256({
        "configuration": configuration,
        "versions": versions,
        "decks": deck_manifests,
    })
    return report
