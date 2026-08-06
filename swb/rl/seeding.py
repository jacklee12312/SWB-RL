from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from swb.rl.versioning import SEED_DERIVATION_VERSION


def derive_seed(parent_seed: int, domain: str, *components: int) -> int:
    if not isinstance(parent_seed, int) or isinstance(parent_seed, bool):
        raise TypeError("parent_seed must be an integer")
    if not domain:
        raise ValueError("seed derivation domain must be non-empty")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in components):
        raise TypeError("seed derivation components must be integers")
    payload = json.dumps(
        {
            "version": SEED_DERIVATION_VERSION,
            "parent": parent_seed,
            "domain": domain,
            "components": components,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class EpisodeSeeds:
    master_seed: int
    worker_seed: int
    episode_seed: int
    deck_seed_a: int
    deck_seed_b: int
    engine_seed: int
    policy_seed: int
    derivation_version: int = SEED_DERIVATION_VERSION


def episode_seeds(
    master_seed: int,
    worker_id: int,
    episode_id: int,
) -> EpisodeSeeds:
    if worker_id < 0 or episode_id < 0:
        raise ValueError("worker_id and episode_id must be non-negative")
    worker = derive_seed(master_seed, "worker", worker_id)
    episode = derive_seed(worker, "episode", episode_id)
    return EpisodeSeeds(
        master_seed=master_seed,
        worker_seed=worker,
        episode_seed=episode,
        deck_seed_a=derive_seed(episode, "deck", 0),
        deck_seed_b=derive_seed(episode, "deck", 1),
        engine_seed=derive_seed(episode, "engine"),
        policy_seed=derive_seed(episode, "policy"),
    )
