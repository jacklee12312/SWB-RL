from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import tracemalloc
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.environment import ShadowverseEnv
from swb.rl.catalog import TrainableCardCatalog


DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_OUTPUT = Path("data/reports/rl_environment_benchmark.json")


def _rate(iterations: int, callback) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        callback()
    elapsed = time.perf_counter() - started
    return iterations / max(elapsed, 1e-12)


def _measure_cached_and_cold(env: ShadowverseEnv, iterations: int) -> dict[str, float]:
    env.action_mask()
    env.observation()
    cached_mask = _rate(iterations, env.action_mask)
    cached_observe = _rate(iterations, env.observation)

    def cold_mask() -> None:
        env.invalidate_cache(reason="benchmark cold mask")
        env.action_mask()

    def cold_observe() -> None:
        env.invalidate_cache(reason="benchmark cold observation")
        env.observation()

    cold_mask_rate = _rate(iterations, cold_mask)
    cold_observe_rate = _rate(iterations, cold_observe)
    return {
        "mask_cached_per_second": cached_mask,
        "mask_cold_per_second": cold_mask_rate,
        "mask_cache_speedup": cached_mask / max(cold_mask_rate, 1e-12),
        "observe_cached_per_second": cached_observe,
        "observe_cold_per_second": cold_observe_rate,
        "observe_cache_speedup": cached_observe / max(cold_observe_rate, 1e-12),
    }


def _measure_steps(
    env: ShadowverseEnv,
    *,
    steps: int,
    seed: int,
) -> dict[str, float | int]:
    chooser = random.Random(seed)
    completed = 0
    episodes = 0
    _, info = env.reset(seed=seed)
    started = time.perf_counter()
    while completed < steps:
        legal = [index for index, allowed in enumerate(info["action_mask"]) if allowed]
        action = chooser.choice(legal)
        transition = env.step(action)
        completed += 1
        info = transition.info
        if transition.terminated or transition.truncated:
            episodes += 1
            _, info = env.reset(seed=seed + episodes)
    elapsed = time.perf_counter() - started
    return {
        "agent_steps": completed,
        "episodes_completed": episodes,
        "step_per_second": completed / max(elapsed, 1e-12),
    }


def _measure_snapshots(env: ShadowverseEnv, iterations: int) -> dict[str, float | int]:
    snapshot = env.snapshot()
    snapshot_rate = _rate(iterations, env.snapshot)
    clone_iterations = max(1, min(iterations, 50))
    clone_rate = _rate(clone_iterations, env.clone)
    return {
        "snapshot_per_second": snapshot_rate,
        "clone_per_second": clone_rate,
        "snapshot_payload_bytes": len(snapshot.core.payload),
    }


def run_benchmark(
    *,
    database: Path,
    iterations: int,
    steps: int,
    seed: int,
    observation_version: str = "v4.1",
) -> dict[str, object]:
    tracemalloc.start()
    startup_started = time.perf_counter()
    repository = CardRepository(database)
    catalog = TrainableCardCatalog.from_repository(repository)
    rulebook = RuleBook.from_directory(ShadowverseEnv.DEFAULT_RULE_DIRECTORY)
    deck_rng = random.Random(seed)
    deck_a = catalog.sample_deck(1, deck_rng)
    deck_b = catalog.sample_deck(1, deck_rng)
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
        card_resolver=catalog.resolve,
        observation_version=observation_version,
        card_vocabulary=catalog.card_vocabulary,
        max_agent_steps=2000,
    )
    env.reset(seed=seed)
    startup_seconds = time.perf_counter() - startup_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cache_rates = _measure_cached_and_cold(env, iterations)
    step_rates = _measure_steps(env, steps=steps, seed=seed)
    snapshot_rates = _measure_snapshots(env, iterations)
    thresholds = {
        "minimum_mask_cache_speedup": 1.5,
        "minimum_observe_cache_speedup": 1.25,
    }
    passed = (
        cache_rates["mask_cache_speedup"]
        >= thresholds["minimum_mask_cache_speedup"]
        and cache_rates["observe_cache_speedup"]
        >= thresholds["minimum_observe_cache_speedup"]
    )
    return {
        "schema_version": 1,
        "machine": {
            "platform": platform.platform(),
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "configuration": {
            "seed": seed,
            "iterations": iterations,
            "agent_steps": steps,
            "card_vocabulary_size": len(catalog.card_vocabulary),
            "exact_collectible_count": len(catalog.exact_collectible_ids),
            "observation_version": observation_version,
            "action_size": env.ACTION_SIZE,
        },
        "startup": {
            "catalog_rulebook_env_reset_seconds": startup_seconds,
            "tracemalloc_peak_bytes": peak_bytes,
        },
        "rates": {**cache_rates, **step_rates, **snapshot_rates},
        "thresholds": thresholds,
        "thresholds_passed": passed,
        "historical_reference": {
            "source": "docs/rl_architecture_audit.md pre-cache legacy-pool audit",
            "core_action_per_second": 627.0,
            "v1_environment_action_per_second": 404.0,
            "v2_observation_milliseconds": 0.98,
            "comparability": (
                "Directional only: the historical run used the legacy follower "
                "pool and is not an exact-card v3 regression baseline."
            ),
        },
        "cache_stats": env.cache_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the SWB RL environment")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--observation-version",
        choices=("v3", "v4", "v4.1"),
        default="v4.1",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.steps <= 0:
        parser.error("--iterations and --steps must be positive")

    report = run_benchmark(
        database=args.database,
        iterations=args.iterations,
        steps=args.steps,
        seed=args.seed,
        observation_version=args.observation_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["thresholds_passed"]:
        raise SystemExit("RL cache performance thresholds failed")


if __name__ == "__main__":
    main()
