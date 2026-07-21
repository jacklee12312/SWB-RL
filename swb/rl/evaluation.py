from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from swb.engine.commands import ChoiceKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.rl.checkpoint import load_checkpoint
from swb.rl.ppo import ObservationFlattener, PPOTrainer, RecurrentMaskedActorCritic
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.seeding import derive_seed


@dataclass(frozen=True)
class EvaluationConfig:
    master_seed: int = 20260721
    seed_count: int = 8
    max_agent_steps: int = 512
    opponent_kind: str = "random_legal"
    opponent_checkpoint: str | None = None

    def __post_init__(self) -> None:
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


class _Policy:
    def reset(self) -> None:
        pass

    def action(
        self,
        env: ShadowverseEnv,
        player_id: int,
        action_mask: np.ndarray,
    ) -> int:
        raise NotImplementedError


class _RandomLegalPolicy(_Policy):
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def action(self, env, player_id, action_mask) -> int:
        legal = np.flatnonzero(action_mask)
        return int(legal[self.rng.randrange(legal.size)])


class _FirstLegalPolicy(_Policy):
    def action(self, env, player_id, action_mask) -> int:
        return int(np.flatnonzero(action_mask)[0])


class _RecurrentPolicy(_Policy):
    def __init__(
        self,
        model: RecurrentMaskedActorCritic,
        flattener: ObservationFlattener,
        device: torch.device,
    ) -> None:
        self.model = model
        self.flattener = flattener
        self.device = device
        self.hidden: dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self.hidden = {
            player: self.model.initial_state(1, device=self.device)
            for player in (0, 1)
        }

    def action(self, env, player_id, action_mask) -> int:
        observation = env.observation(
            perspective=player_id,
            action_mask=action_mask,
        )
        vector = torch.from_numpy(self.flattener.encode(observation)).to(
            self.device
        ).unsqueeze(0)
        card_indices = torch.from_numpy(
            self.flattener.encode_cards(observation)
        ).to(self.device).unsqueeze(0)
        mask = torch.from_numpy(action_mask.astype(np.bool_)).to(
            self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _, hidden = self.model.forward_step(
                vector, self.hidden[player_id], card_indices
            )
            masked = self.model.masked_logits(logits, mask)
        self.hidden[player_id] = hidden
        return int(masked.argmax(dim=-1).item())


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
) -> _Policy:
    if config.opponent_kind == "random_legal":
        return _RandomLegalPolicy(seed)
    if config.opponent_kind == "fixed":
        return _FirstLegalPolicy()
    if config.opponent_kind == "current":
        return _RecurrentPolicy(trainer.model, trainer.flattener, trainer.device)
    historical = load_checkpoint(
        Path(config.opponent_checkpoint),
        snapshot,
        device=str(trainer.device),
    )
    return _RecurrentPolicy(
        historical.model,
        historical.flattener,
        historical.device,
    )


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
    results = []
    visited_cards: set[int] = set()
    visited_classes: set[int] = set()
    visited_mechanisms: set[str] = set()
    mask_checks = 0
    mask_mismatches = 0
    illegal_actions = 0
    try:
        for seed_index in range(config.seed_count):
            deck_seed = derive_seed(config.master_seed, "evaluation_decks", seed_index)
            learner_deck = snapshot.catalog.sample_deck(
                1, random.Random(derive_seed(deck_seed, "learner"))
            )
            opponent_deck = snapshot.catalog.sample_deck(
                1, random.Random(derive_seed(deck_seed, "opponent"))
            )
            for learner_player in (0, 1):
                if learner_player == 0:
                    decks = (learner_deck, opponent_deck)
                else:
                    decks = (opponent_deck, learner_deck)
                engine_seed = derive_seed(
                    config.master_seed,
                    "evaluation_engine",
                    seed_index,
                    learner_player,
                )
                env = ShadowverseEnv(
                    decks[0],
                    decks[1],
                    class_a=1,
                    class_b=1,
                    seed=engine_seed,
                    rulebook=trainer.assets.rulebook,
                    card_resolver=trainer.assets.catalog.resolve,
                    observation_version="v3",
                    card_vocabulary=trainer.assets.catalog.card_vocabulary,
                    max_agent_steps=config.max_agent_steps,
                    training_mode=True,
                    training_event_history_limit=4096,
                )
                _, info = env.reset(seed=engine_seed)
                learner_policy = _RecurrentPolicy(
                    trainer.model, trainer.flattener, trainer.device
                )
                opponent = _opponent_policy(
                    config,
                    trainer,
                    snapshot,
                    derive_seed(engine_seed, "opponent_policy"),
                )
                learner_policy.reset()
                opponent.reset()
                steps = 0
                while not env.terminated and not env.truncated:
                    player = env.decision_player
                    reported_mask = np.asarray(info["action_mask"], dtype=np.int8)
                    executable_mask = np.asarray(env.action_mask(), dtype=np.int8)
                    mask_checks += 1
                    if not np.array_equal(reported_mask, executable_mask):
                        mask_mismatches += 1
                    policy = learner_policy if player == learner_player else opponent
                    action = policy.action(env, player, reported_mask)
                    if action < 0 or action >= env.ACTION_SIZE or not reported_mask[action]:
                        illegal_actions += 1
                        raise RuntimeError("evaluation policy selected an illegal action")
                    transition = env.step(action)
                    info = transition.info
                    steps += 1
                    for state_player in env._core.players:
                        visited_classes.add(state_player.class_id)
                        visited_cards.update(
                            card.card_id for card in state_player.hand
                        )
                        visited_cards.update(
                            entity.definition.card_id for entity in state_player.board
                        )
                        visited_cards.update(
                            card.definition.card_id for card in state_player.graveyard
                        )
                        visited_cards.update(
                            card.card_id for card in state_player.banished
                        )
                    pending = env._core.state.pending_choice
                    if pending is not None:
                        visited_mechanisms.add(f"choice:{pending.choice_kind.value}")
                visited_mechanisms.update(
                    event.type.value for event in env._core.event_history
                )
                score = (
                    0.5
                    if env.winner is None
                    else (1.0 if env.winner == learner_player else 0.0)
                )
                results.append({
                    "seed_index": seed_index,
                    "learner_player": learner_player,
                    "score": score,
                    "winner": env.winner,
                    "turn": env.turn,
                    "agent_steps": steps,
                    "terminated": env.terminated,
                    "truncated": env.truncated,
                })
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        trainer.torch_generator.set_state(generator_rng)
        trainer.model.train(was_training)

    games = len(results)
    score = sum(result["score"] for result in results)
    win_rate = score / games
    confidence = _wilson_interval(score, games)
    clamped = min(1.0 - 1e-6, max(1e-6, win_rate))
    elo = 400.0 * math.log10(clamped / (1.0 - clamped))
    side_rates = {
        str(side): (
            sum(
                result["score"]
                for result in results
                if result["learner_player"] == side
            )
            / config.seed_count
        )
        for side in (0, 1)
    }
    mechanism_universe = {
        *(event_type.value for event_type in EventType),
        *(f"choice:{choice_kind.value}" for choice_kind in ChoiceKind),
    }
    visited_known_mechanisms = visited_mechanisms & mechanism_universe
    return {
        "schema_version": 1,
        "purpose": "fixed-seed evaluation; not a policy-strength claim",
        "configuration": {
            "master_seed": config.master_seed,
            "seed_count": config.seed_count,
            "mirrored_games": games,
            "opponent_kind": config.opponent_kind,
            "opponent_checkpoint": config.opponent_checkpoint,
            "max_agent_steps": config.max_agent_steps,
        },
        "metrics": {
            "win_rate": win_rate,
            "side_win_rates": side_rates,
            "confidence_interval_95": confidence,
            "elo_relative": elo,
            "average_turn": sum(result["turn"] for result in results) / games,
            "average_agent_steps": (
                sum(result["agent_steps"] for result in results) / games
            ),
            "terminated": sum(result["terminated"] for result in results),
            "truncated": sum(result["truncated"] for result in results),
            "terminated_rate": (
                sum(result["terminated"] for result in results) / games
            ),
            "truncated_rate": (
                sum(result["truncated"] for result in results) / games
            ),
            "illegal_action_rate": illegal_actions / max(1, mask_checks),
            "action_mask_checks": mask_checks,
            "action_mask_mismatches": mask_mismatches,
        },
        "coverage": {
            "card_ids": sorted(visited_cards),
            "card_count": len(visited_cards),
            "card_coverage_rate": (
                len(visited_cards)
                / max(1, len(snapshot.catalog.exact_collectible_ids))
            ),
            "classes": sorted(visited_classes),
            "class_coverage_rate": len(visited_classes) / 7.0,
            "mechanisms": sorted(visited_mechanisms),
            "known_mechanism_count": len(mechanism_universe),
            "mechanism_coverage_rate": (
                len(visited_known_mechanisms) / max(1, len(mechanism_universe))
            ),
        },
        "games": results,
    }
