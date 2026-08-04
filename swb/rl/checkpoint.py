from __future__ import annotations

import os
import platform
import random
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from swb.engine.environment import MATCH_SETUP_LEGACY, ShadowverseEnv
from swb.rl.ppo import PPOConfig, PPOTrainer
from swb.rl.opponents import (
    OpponentEntry,
    OpponentEpisodeScheduler,
    OpponentPool,
)
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.versioning import ExperimentVersions


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_CONFIG_OVERRIDE_FIELDS = frozenset({
    "rollout_workers",
    "rollout_worker_torch_threads",
    "central_inference_batch_wait_seconds",
    "opponent_current_weight",
    "opponent_random_weight",
    "opponent_fixed_weight",
    "opponent_historical_weight",
    "opponent_external_manifest",
    "opponent_external_weight",
    "opponent_model_cache_size",
    "opponent_model_cache_max_bytes",
    "opponent_max_history",
    "opponent_snapshot_interval_steps",
    "opponent_batching_mode",
})
OPPONENT_POOL_CONFIG_FIELDS = frozenset({
    "opponent_current_weight",
    "opponent_random_weight",
    "opponent_fixed_weight",
    "opponent_historical_weight",
    "opponent_external_manifest",
    "opponent_external_weight",
    "opponent_max_history",
    "opponent_snapshot_interval_steps",
    "opponent_batching_mode",
})


def _git_state() -> dict[str, object]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ("git", *args),
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _versions(trainer: PPOTrainer) -> ExperimentVersions:
    assert trainer.env is not None
    return ExperimentVersions.capture(
        trainer.env,
        trainer.assets.catalog,
        rulebook_sha256=trainer.snapshot.rulebook_sha256,
    )


def build_checkpoint(trainer: PPOTrainer) -> dict[str, object]:
    assert trainer.env is not None
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "trainer": {
            "master_seed": trainer.master_seed,
            "config": asdict(trainer.config),
            "next_episode_id": trainer.next_episode_id,
            "current_episode_id": trainer.current_episode_id,
            "completed_episodes": trainer.completed_episodes,
            "agent_steps": trainer.agent_steps,
            "update_count": trainer.update_count,
            "learner_player": trainer.learner_player,
            "current_opponent": (
                None
                if trainer.current_opponent is None
                else asdict(trainer.current_opponent)
            ),
            "opponent_pool": trainer.opponent_pool.state_dict(),
            "opponent_scheduler": trainer.opponent_scheduler.state_dict(),
            "opponent_cache_metrics": trainer.opponent_cache_metrics(),
            "resume_config_overrides": getattr(
                trainer, "resume_config_overrides", {}
            ),
            "opponent_pool_replaced_on_load": getattr(
                trainer, "opponent_pool_replaced_on_load", False
            ),
            "opponent_assignments": list(trainer.opponent_assignments),
            "matchup_statistics": dict(trainer.matchup_statistics),
            "current_matchup_assignment": trainer.current_matchup_assignment,
            "current_episode_agent_steps": (
                trainer.current_episode_agent_steps
            ),
            "opponent_rng_state": trainer.opponent_rng.getstate(),
            "opponent_hidden": (
                None
                if trainer.opponent_hidden is None
                else trainer.opponent_hidden.detach().cpu()
            ),
            "hidden_by_player": {
                player: hidden.detach().cpu()
                for player, hidden in trainer.hidden_by_player.items()
            },
            "torch_generator_state": trainer.torch_generator.get_state(),
        },
        "environment": {
            "snapshot": trainer.env.snapshot(),
            "deck_card_ids": tuple(
                tuple(card.card_id for card in deck)
                for deck in trainer.env._core.deck_lists
            ),
            "classes": trainer.env._core.player_classes,
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        "versions": _versions(trainer).to_dict(),
        "experiment_manifest": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "git": _git_state(),
            "database_source": dict(trainer.assets.catalog.source_snapshot),
            "coverage_report_sha256": (
                trainer.assets.catalog.coverage_report_sha256
            ),
            "catalog_sha256": trainer.assets.catalog.catalog_sha256,
            "rulebook_sha256": trainer.snapshot.rulebook_sha256,
            "observation_version": trainer.env.observation_version,
            "action_size": trainer.env.ACTION_SIZE,
            "match_setup": trainer.config.match_setup,
            "training_deck": (
                None
                if trainer.fixed_training_deck is None
                else trainer.fixed_training_deck.manifest()
            ),
            "opponent_decks": [
                deck.manifest()
                for deck in trainer.fixed_opponent_decks
            ],
            "policy_representation": {
                "architecture": trainer.model.architecture,
                "numeric_size": trainer.flattener.size,
                "card_slots": trainer.flattener.card_slots,
                "card_vocabulary_size": len(
                    trainer.assets.catalog.card_vocabulary
                ),
                "card_embedding_dim": trainer.config.card_embedding_dim,
            },
            "opponent_pool": trainer.opponent_pool.state_dict(),
            "opponent_scheduler": trainer.opponent_scheduler.state_dict(),
            "external_opponent_manifest": (
                None
                if trainer.external_opponent_manifest is None
                else trainer.external_opponent_manifest.summary()
            ),
            "opponent_assignments": list(trainer.opponent_assignments),
        },
    }


def save_checkpoint_atomic(path: str | Path, trainer: PPOTrainer) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(build_checkpoint(trainer), temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _load_payload(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Unable to load checkpoint {path!s}: {exc}") from exc
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported checkpoint schema version: "
            f"{payload.get('checkpoint_schema_version')!r}"
        )
    return payload


def load_checkpoint(
    path: str | Path,
    snapshot: WorkerAssetsSnapshot,
    *,
    device: str = "cpu",
    restore_rng_state: bool = True,
    config_overrides: Mapping[str, object] | None = None,
    replace_opponent_pool: bool = False,
) -> PPOTrainer:
    payload = _load_payload(path)
    trainer_state = payload["trainer"]
    config_payload = dict(trainer_state["config"])
    overrides = dict(config_overrides or {})
    unsupported_overrides = sorted(
        set(overrides) - CHECKPOINT_CONFIG_OVERRIDE_FIELDS
    )
    if unsupported_overrides:
        raise ValueError(
            "unsupported checkpoint config overrides: "
            + ", ".join(unsupported_overrides)
        )
    changed_pool_fields = {
        field
        for field in OPPONENT_POOL_CONFIG_FIELDS
        if field in overrides and overrides[field] != config_payload.get(field)
    }
    if changed_pool_fields and not replace_opponent_pool:
        raise ValueError(
            "opponent-pool config overrides require replace_opponent_pool=true"
        )
    if replace_opponent_pool and not changed_pool_fields:
        raise ValueError(
            "replace_opponent_pool=true requires an opponent-pool config change"
        )
    config_payload.update(overrides)
    # Schema-v2 checkpoints created before official setup integration did not
    # record this field and must retain their historical fixed-player/no-
    # mulligan behavior when resumed.
    config_payload.setdefault("match_setup", MATCH_SETUP_LEGACY)
    checkpoint_observation = str(
        payload.get("versions", {}).get("observation_version", "")
    )
    config_payload.setdefault(
        "observation_version",
        (
            "v3"
            if checkpoint_observation.startswith("observation-v3")
            else (
                "v4.1"
                if checkpoint_observation.startswith("observation-v4.1")
                else "v4"
            )
        ),
    )
    trainer = PPOTrainer(
        snapshot,
        master_seed=int(trainer_state["master_seed"]),
        config=PPOConfig(**config_payload),
        device=device,
    )
    checkpoint_versions = ExperimentVersions(**payload["versions"])
    checkpoint_versions.assert_compatible(_versions(trainer))

    fresh_opponent_pool = trainer.opponent_pool

    trainer.model.load_state_dict(payload["model_state"])
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.next_episode_id = int(trainer_state["next_episode_id"])
    trainer.current_episode_id = int(trainer_state["current_episode_id"])
    trainer.completed_episodes = int(trainer_state["completed_episodes"])
    trainer.agent_steps = int(trainer_state["agent_steps"])
    trainer.update_count = int(trainer_state["update_count"])
    trainer.learner_player = int(trainer_state["learner_player"])
    trainer.current_opponent = (
        None
        if replace_opponent_pool or trainer_state["current_opponent"] is None
        else OpponentEntry(**trainer_state["current_opponent"])
    )
    if replace_opponent_pool:
        trainer.opponent_pool = fresh_opponent_pool
        trainer.opponent_pool.selection_count = 0
        trainer.opponent_pool.selection_counts.clear()
        trainer.opponent_pool.selection_counts_by_opponent.clear()
    else:
        trainer.opponent_pool = OpponentPool.from_state_dict(
            trainer_state["opponent_pool"],
            expected_external_manifest_sha256=(
                None
                if trainer.external_opponent_manifest is None
                else trainer.external_opponent_manifest.file_sha256
            ),
            expected_external_entries=(
                None
                if trainer.external_opponent_manifest is None
                else trainer.external_opponent_manifest.entries
            ),
        )
    trainer.opponent_scheduler = OpponentEpisodeScheduler(
        trainer.opponent_pool,
        worker_count=trainer.config.rollout_workers,
        mode=trainer.config.opponent_batching_mode,
    )
    if not replace_opponent_pool:
        trainer.opponent_scheduler.load_state_dict(
            trainer_state.get("opponent_scheduler", {
                "worker_count": trainer.config.rollout_workers,
                "mode": trainer.config.opponent_batching_mode,
                "pending": [],
            })
        )
    cache_metrics = (
        {}
        if replace_opponent_pool
        else trainer_state.get("opponent_cache_metrics", {})
    )
    trainer.opponent_assignments = (
        []
        if replace_opponent_pool
        else list(trainer_state.get("opponent_assignments", []))
    )
    trainer.resume_config_overrides = overrides
    trainer.opponent_pool_replaced_on_load = replace_opponent_pool
    trainer.matchup_statistics = dict(
        trainer_state.get("matchup_statistics", {})
    )
    for stats in trainer.matchup_statistics.values():
        if "learner_first" in stats:
            stats["learner_player_0"] = stats.pop("learner_first")
        if "learner_second" in stats:
            stats["learner_player_1"] = stats.pop("learner_second")
    trainer.current_matchup_assignment = (
        None
        if replace_opponent_pool
        else trainer_state.get("current_matchup_assignment")
    )
    trainer.current_episode_agent_steps = (
        0
        if replace_opponent_pool
        else int(trainer_state.get("current_episode_agent_steps", 0))
    )
    trainer.opponent_rng.setstate(trainer_state["opponent_rng_state"])
    trainer.hidden_by_player = {
        int(player): hidden.to(trainer.device)
        for player, hidden in trainer_state["hidden_by_player"].items()
    }
    if restore_rng_state:
        try:
            trainer.torch_generator.set_state(
                trainer_state["torch_generator_state"]
            )
        except RuntimeError as exc:
            raise ValueError(
                "checkpoint policy RNG cannot be restored on "
                f"device {trainer.device}; exact training resume requires the "
                "same device type used to create the checkpoint"
            ) from exc

    environment_state = payload["environment"]
    decks = []
    for card_ids in environment_state["deck_card_ids"]:
        definitions = []
        for card_id in card_ids:
            definition = trainer.assets.catalog.resolve(card_id)
            if definition is None:
                raise ValueError(
                    f"checkpoint deck references missing card {card_id}"
                )
            definitions.append(definition)
        decks.append(definitions)
    classes = environment_state["classes"]
    trainer.env = ShadowverseEnv(
        decks[0],
        decks[1],
        class_a=classes[0],
        class_b=classes[1],
        seed=0,
        rulebook=trainer.assets.rulebook,
        card_resolver=trainer.assets.catalog.resolve,
        observation_version=trainer.config.observation_version,
        card_vocabulary=trainer.assets.catalog.card_vocabulary,
        max_game_turns=trainer.config.max_game_turns,
        max_agent_steps=trainer.config.max_agent_steps_per_episode,
        training_mode=True,
        match_setup=trainer.config.match_setup,
    )
    trainer.env.restore(environment_state["snapshot"])
    mask = trainer.env.action_mask()
    trainer.info = trainer.env.info(action_mask=mask)
    if replace_opponent_pool and trainer.config.rollout_workers == 1:
        trainer._start_episode()
    else:
        trainer._prepare_opponent()
    if (
        not replace_opponent_pool
        and trainer.opponent_hidden is not None
        and trainer_state["opponent_hidden"] is not None
    ):
        trainer.opponent_hidden = trainer_state["opponent_hidden"].to(trainer.device)
    trainer._opponent_cache_hits = int(cache_metrics.get("hits", 0))
    trainer._opponent_cache_misses = int(cache_metrics.get("misses", 0))
    trainer._opponent_cache_evictions = int(cache_metrics.get("evictions", 0))
    trainer._opponent_cache_load_seconds = float(
        cache_metrics.get("load_seconds", 0.0)
    )
    trainer._opponent_model_switches = int(
        cache_metrics.get("model_switches", 0)
    )
    trainer._last_opponent_cache_key = cache_metrics.get("last_key")

    if restore_rng_state:
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return trainer
