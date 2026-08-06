from __future__ import annotations

import multiprocessing as mp
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from swb.db.repository import CardRepository
from swb.engine.environment import ShadowverseEnv
from swb.rl.fixed_decks import (
    OFFICIAL_QR_EVOLVE_HAVEN,
    get_fixed_training_deck,
)
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.seeding import derive_seed, episode_seeds
from swb.rl.trajectory import TRAJECTORY_SCHEMA_VERSION
from swb.rl.vector_rollout import (
    PolicyVectorRollout,
    RolloutConfig,
    RolloutWorkerError,
    VectorRollout,
)


DATABASE = Path("data/cards.sqlite3")
SPECIALIST_OPPONENT_DECKS = (
    "international_qr_forest_20260728",
    "international_qr_sword_20260728",
)


def trajectory_summary(trajectory) -> tuple:
    return (
        trajectory.episode_id,
        trajectory.worker_id,
        trajectory.seeds,
        trajectory.deck_card_ids,
        tuple(
            (
                step.player_id,
                step.action,
                step.reward,
                step.terminated,
                step.truncated,
            )
            for step in trajectory.steps
        ),
        trajectory.winner,
        trajectory.terminated,
        trajectory.truncated,
        trajectory.final_fingerprint_sha256,
    )


@unittest.skipUnless(DATABASE.exists(), "real card database is unavailable")
class VectorRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = WorkerAssetsSnapshot.build(CardRepository(DATABASE))

    def config(self, **kwargs) -> RolloutConfig:
        values = {
            "master_seed": 123456,
            "worker_count": 1,
            "max_agent_steps": 12,
            "result_timeout_seconds": 60.0,
        }
        values.update(kwargs)
        return RolloutConfig(**values)

    def collect(self, config: RolloutConfig, count: int):
        rollout = VectorRollout(self.snapshot, config)
        try:
            return rollout.collect(count)
        finally:
            rollout.close()

    def test_seed_derivation_is_domain_separated_and_stable(self) -> None:
        first = episode_seeds(123, 2, 7)
        self.assertEqual(first, episode_seeds(123, 2, 7))
        self.assertNotEqual(first.deck_seed_a, first.deck_seed_b)
        self.assertNotEqual(first.engine_seed, first.policy_seed)
        self.assertNotEqual(
            derive_seed(123, "worker", 2),
            derive_seed(123, "episode", 2),
        )

    def test_single_worker_runs_are_identical(self) -> None:
        first = self.collect(self.config(worker_count=1), 2)
        second = self.collect(self.config(worker_count=1), 2)
        self.assertEqual(
            tuple(map(trajectory_summary, first)),
            tuple(map(trajectory_summary, second)),
        )

    def test_multi_worker_runs_are_identical_and_return_episode_order(self) -> None:
        first = self.collect(self.config(worker_count=2), 4)
        second = self.collect(self.config(worker_count=2), 4)
        self.assertEqual([item.episode_id for item in first], [0, 1, 2, 3])
        self.assertEqual([item.worker_id for item in first], [0, 1, 0, 1])
        self.assertEqual(
            tuple(map(trajectory_summary, first)),
            tuple(map(trajectory_summary, second)),
        )

    def test_trajectory_schema_covers_masks_versions_and_boundaries(self) -> None:
        trajectory = self.collect(self.config(), 1)[0]
        self.assertEqual(trajectory.schema_version, TRAJECTORY_SCHEMA_VERSION)
        self.assertTrue(trajectory.truncated)
        self.assertFalse(trajectory.terminated)
        self.assertTrue(trajectory.steps[-1].bootstrap_value_allowed)
        self.assertEqual(
            {step.player_id for step in trajectory.steps},
            {0, 1},
        )
        for step in trajectory.steps:
            self.assertEqual(step.action_mask.dtype, np.int8)
            self.assertTrue(step.action_mask[step.action])
            self.assertIn("observation_schema_sha256", step.versions)
            self.assertIn("action_layout_sha256", step.versions)
            self.assertIn("catalog_sha256", step.versions)
            self.assertIn("rulebook_sha256", step.versions)
        self.assertTrue(all(
            ShadowverseEnv.CHOICE_OFFSET
            <= step.action
            < ShadowverseEnv.CHOICE_OFFSET + 16
            for step in trajectory.steps[:2]
        ))
        self.assertNotEqual(
            trajectory.steps[0].player_id,
            trajectory.steps[1].player_id,
        )

    def test_fixed_training_deck_reaches_spawn_worker_unchanged(self) -> None:
        recipe = get_fixed_training_deck(OFFICIAL_QR_EVOLVE_HAVEN)
        trajectory = self.collect(
            self.config(
                class_a=recipe.class_id,
                class_b=recipe.class_id,
                training_deck=recipe.name,
            ),
            1,
        )[0]
        expected = Counter(recipe.card_ids)
        self.assertEqual(Counter(trajectory.deck_card_ids[0]), expected)
        self.assertEqual(Counter(trajectory.deck_card_ids[1]), expected)

    def test_fixed_training_deck_rejects_wrong_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires class 6"):
            self.config(training_deck=OFFICIAL_QR_EVOLVE_HAVEN)

    def test_specialist_deck_cycle_reaches_spawn_workers(self) -> None:
        learner = get_fixed_training_deck(OFFICIAL_QR_EVOLVE_HAVEN)
        opponents = tuple(
            get_fixed_training_deck(name)
            for name in SPECIALIST_OPPONENT_DECKS
        )
        trajectories = self.collect(
            self.config(
                worker_count=2,
                class_a=learner.class_id,
                class_b=learner.class_id,
                training_deck=learner.name,
                opponent_decks=SPECIALIST_OPPONENT_DECKS,
            ),
            4,
        )
        expected = (
            (learner, opponents[0]),
            (opponents[0], learner),
            (learner, opponents[1]),
            (opponents[1], learner),
        )
        for trajectory, recipes in zip(trajectories, expected):
            with self.subTest(episode_id=trajectory.episode_id):
                for card_ids, recipe in zip(
                    trajectory.deck_card_ids,
                    recipes,
                ):
                    self.assertEqual(
                        Counter(card_ids),
                        Counter(recipe.card_ids),
                    )

    def test_worker_exception_is_propagated_and_processes_stop(self) -> None:
        rollout = VectorRollout(
            self.snapshot,
            self.config(worker_count=2, fail_episode_id=0),
        )
        with self.assertRaisesRegex(RolloutWorkerError, "injected rollout failure"):
            rollout.collect(2)
        self.assertTrue(rollout.processes)
        self.assertFalse(any(process.is_alive() for process in rollout.processes))

    def test_close_is_graceful_and_leaves_no_live_worker(self) -> None:
        rollout = VectorRollout(self.snapshot, self.config(worker_count=2))
        rollout.start()
        processes = rollout.processes
        rollout.close()
        self.assertFalse(any(process.is_alive() for process in processes))
        active_ids = {process.pid for process in mp.active_children()}
        self.assertFalse(any(process.pid in active_ids for process in processes))

if __name__ == "__main__":
    unittest.main()
