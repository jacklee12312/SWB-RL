from __future__ import annotations

import os
import platform
import random
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swb.engine.environment import ShadowverseEnv
from swb.rl.ppo import PPOConfig, PPOTrainer
from swb.rl.opponents import OpponentEntry, OpponentPool
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.versioning import ExperimentVersions


CHECKPOINT_SCHEMA_VERSION = 2


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
            "opponent_assignments": list(trainer.opponent_assignments),
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
            "observation_version": "v3",
            "action_size": trainer.env.ACTION_SIZE,
            "policy_representation": {
                "numeric_size": trainer.flattener.size,
                "card_slots": trainer.flattener.card_slots,
                "card_vocabulary_size": len(
                    trainer.assets.catalog.card_vocabulary
                ),
                "card_embedding_dim": trainer.config.card_embedding_dim,
            },
            "opponent_pool": trainer.opponent_pool.state_dict(),
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
) -> PPOTrainer:
    payload = _load_payload(path)
    trainer_state = payload["trainer"]
    trainer = PPOTrainer(
        snapshot,
        master_seed=int(trainer_state["master_seed"]),
        config=PPOConfig(**trainer_state["config"]),
        device=device,
    )
    checkpoint_versions = ExperimentVersions(**payload["versions"])
    checkpoint_versions.assert_compatible(_versions(trainer))

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
        if trainer_state["current_opponent"] is None
        else OpponentEntry(**trainer_state["current_opponent"])
    )
    trainer.opponent_pool = OpponentPool.from_state_dict(
        trainer_state["opponent_pool"]
    )
    trainer.opponent_assignments = list(
        trainer_state.get("opponent_assignments", [])
    )
    trainer.opponent_rng.setstate(trainer_state["opponent_rng_state"])
    trainer.hidden_by_player = {
        int(player): hidden.to(trainer.device)
        for player, hidden in trainer_state["hidden_by_player"].items()
    }
    trainer.torch_generator.set_state(trainer_state["torch_generator_state"])

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
        observation_version="v3",
        card_vocabulary=trainer.assets.catalog.card_vocabulary,
        max_game_turns=trainer.config.max_game_turns,
        max_agent_steps=trainer.config.max_agent_steps_per_episode,
        training_mode=True,
    )
    trainer.env.restore(environment_state["snapshot"])
    mask = trainer.env.action_mask()
    trainer.info = trainer.env.info(action_mask=mask)
    trainer._prepare_opponent()
    if trainer.opponent_hidden is not None and trainer_state["opponent_hidden"] is not None:
        trainer.opponent_hidden = trainer_state["opponent_hidden"].to(trainer.device)

    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return trainer
