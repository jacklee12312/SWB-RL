from __future__ import annotations

import multiprocessing as mp
import unittest
from pathlib import Path

import numpy as np

from swb.db.repository import CardRepository
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.seeding import derive_seed, episode_seeds
from swb.rl.trajectory import TRAJECTORY_SCHEMA_VERSION
from swb.rl.vector_rollout import (
    RolloutConfig,
    RolloutWorkerError,
    VectorRollout,
)


DATABASE = Path("data/cards.sqlite3")


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
