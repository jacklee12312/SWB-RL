from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine.environment import MATCH_SETUP_OFFICIAL, MATCH_SETUP_VALUES
from swb.rl.fixed_decks import (
    fixed_training_deck_names,
    get_fixed_training_deck,
)
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.vector_rollout import RolloutConfig, VectorRollout


DEFAULT_OUTPUT = Path("data/reports/vector_rollout_benchmark.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic multi-process SWB rollouts"
    )
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-agent-steps", type=int, default=256)
    parser.add_argument("--master-seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--training-deck",
        choices=fixed_training_deck_names(),
        help="use one named fixed deck for both players",
    )
    parser.add_argument(
        "--match-setup",
        choices=sorted(MATCH_SETUP_VALUES),
        default=MATCH_SETUP_OFFICIAL,
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.episodes <= 0 or args.max_agent_steps <= 0:
        parser.error("workers, episodes, and max-agent-steps must be positive")

    startup = time.perf_counter()
    snapshot = WorkerAssetsSnapshot.build(CardRepository(args.database))
    snapshot_seconds = time.perf_counter() - startup
    fixed_deck = (
        None
        if args.training_deck is None
        else get_fixed_training_deck(args.training_deck)
    )
    config = RolloutConfig(
        master_seed=args.master_seed,
        worker_count=args.workers,
        class_a=1 if fixed_deck is None else fixed_deck.class_id,
        class_b=1 if fixed_deck is None else fixed_deck.class_id,
        max_agent_steps=args.max_agent_steps,
        training_deck=args.training_deck,
        match_setup=args.match_setup,
    )
    rollout_started = time.perf_counter()
    with VectorRollout(snapshot, config) as rollout:
        trajectories = rollout.collect(args.episodes)
    rollout_seconds = time.perf_counter() - rollout_started
    agent_steps = sum(len(trajectory.steps) for trajectory in trajectories)
    report = {
        "schema_version": 1,
        "machine": {
            "platform": platform.platform(),
            "python": sys.version,
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "configuration": {
            "master_seed": args.master_seed,
            "workers": args.workers,
            "episodes": args.episodes,
            "max_agent_steps": args.max_agent_steps,
            "start_method": config.start_method,
            "match_setup": config.match_setup,
            "training_deck": (
                None if fixed_deck is None else fixed_deck.manifest()
            ),
        },
        "versions": {
            "catalog_sha256": snapshot.catalog.catalog_sha256,
            "card_vocabulary_sha256": snapshot.catalog.card_vocabulary_sha256,
            "rulebook_sha256": snapshot.rulebook_sha256,
            "coverage_report_sha256": snapshot.catalog.coverage_report_sha256,
            "training_pool_sha256": snapshot.catalog.training_pool_sha256,
        },
        "results": {
            "snapshot_build_seconds": snapshot_seconds,
            "rollout_seconds": rollout_seconds,
            "episodes_per_second": args.episodes / max(rollout_seconds, 1e-12),
            "agent_steps": agent_steps,
            "agent_steps_per_second": agent_steps / max(rollout_seconds, 1e-12),
            "terminated": sum(item.terminated for item in trajectories),
            "truncated": sum(item.truncated for item in trajectories),
            "episode_ids": [item.episode_id for item in trajectories],
            "worker_ids": [item.worker_id for item in trajectories],
            "fingerprints": [
                item.final_fingerprint_sha256 for item in trajectories
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
