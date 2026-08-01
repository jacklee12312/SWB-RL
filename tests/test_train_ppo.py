from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path

from scripts.train_ppo import (
    RUNTIME_OVERRIDE_FIELDS,
    _periodic_checkpoint_due,
    _periodic_checkpoint_path,
    _resume_runtime_config,
    _runtime_override_report,
)
from swb.rl.ppo import PPOConfig
from swb.rl.policy import ENTITY_ACTION_POLICY_ARCHITECTURE


class TrainPPOCommandTests(unittest.TestCase):
    def test_resume_runtime_override_changes_only_runtime_fields(
        self,
    ) -> None:
        before = PPOConfig(
            rollout_workers=4,
            rollout_worker_torch_threads=2,
            central_inference_batch_wait_seconds=0.0005,
            learning_rate=1e-4,
            policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
            observation_version="v4.1",
        )
        after = _resume_runtime_config(
            before,
            rollout_workers=6,
            rollout_worker_threads=2,
            central_inference_batch_wait_ms=1.0,
        )
        changed = {
            key
            for key, value in asdict(before).items()
            if asdict(after)[key] != value
        }
        self.assertEqual(
            changed,
            {
                "rollout_workers",
                "central_inference_batch_wait_seconds",
            },
        )
        self.assertEqual(after.learning_rate, 1e-4)
        self.assertEqual(after.observation_version, "v4.1")
        self.assertEqual(
            set(_runtime_override_report(before, after)),
            changed,
        )
        self.assertEqual(
            set(RUNTIME_OVERRIDE_FIELDS),
            {
                "rollout_workers",
                "rollout_worker_torch_threads",
                "central_inference_batch_wait_seconds",
            },
        )

    def test_periodic_checkpoint_gate_is_incremental(self) -> None:
        self.assertFalse(_periodic_checkpoint_due(
            last_checkpoint_steps=100,
            current_steps=1_000,
            interval_steps=0,
        ))
        self.assertFalse(_periodic_checkpoint_due(
            last_checkpoint_steps=100,
            current_steps=1_099,
            interval_steps=1_000,
        ))
        self.assertTrue(_periodic_checkpoint_due(
            last_checkpoint_steps=100,
            current_steps=1_100,
            interval_steps=1_000,
        ))

    def test_periodic_checkpoint_path_retains_each_step(self) -> None:
        checkpoint = Path("data/checkpoints/run/final.pt")
        self.assertEqual(
            _periodic_checkpoint_path(checkpoint, 123_456),
            Path(
                "data/checkpoints/run/final_checkpoints/"
                "step_000000123456.pt"
            ),
        )


if __name__ == "__main__":
    unittest.main()
