from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from swb.db.repository import CardRepository
from swb.rl.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint_atomic,
)
from swb.rl.ppo import PPOConfig, PPOTrainer
from swb.rl.runtime import WorkerAssetsSnapshot


DATABASE = Path("data/cards.sqlite3")


@unittest.skipUnless(DATABASE.exists(), "real card database is unavailable")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = WorkerAssetsSnapshot.build(CardRepository(DATABASE))

    def make_trainer(self) -> PPOTrainer:
        return PPOTrainer(
            self.snapshot,
            master_seed=2468,
            config=PPOConfig(
                rollout_steps=8,
                sequence_length=4,
                minibatch_sequences=2,
                update_epochs=1,
                hidden_size=32,
                max_agent_steps_per_episode=6,
            ),
        )

    def trained_once(self) -> PPOTrainer:
        trainer = self.make_trainer()
        records, bootstrap, _ = trainer.collect_rollout()
        trainer.update(records, bootstrap)
        return trainer

    def test_manifest_contains_full_progress_rng_versions_and_git_state(self) -> None:
        payload = build_checkpoint(self.trained_once())
        self.assertIn("model_state", payload)
        self.assertIn("optimizer_state", payload)
        self.assertIn("python", payload["rng"])
        self.assertIn("numpy", payload["rng"])
        self.assertIn("torch_cpu", payload["rng"])
        self.assertIn("next_episode_id", payload["trainer"])
        self.assertIn("current_episode_id", payload["trainer"])
        self.assertIn("hidden_by_player", payload["trainer"])
        self.assertIn("git", payload["experiment_manifest"])
        self.assertIn("opponent_pool", payload["experiment_manifest"])
        self.assertEqual(payload["experiment_manifest"]["match_setup"], "official")
        self.assertEqual(payload["trainer"]["config"]["match_setup"], "official")
        self.assertIn("observation_schema_sha256", payload["versions"])
        self.assertIn("action_layout_sha256", payload["versions"])

    def test_save_resume_matches_uninterrupted_next_update(self) -> None:
        trainer = self.trained_once()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint_atomic(path, trainer)

            direct_records, direct_bootstrap, _ = trainer.collect_rollout()
            direct_summary = [
                (record.episode_id, record.player_id, record.action, record.reward)
                for record in direct_records
            ]
            direct_metrics = trainer.update(direct_records, direct_bootstrap)
            direct_parameters = {
                key: value.detach().clone()
                for key, value in trainer.model.state_dict().items()
            }

            resumed = load_checkpoint(path, self.snapshot)
            resumed_records, resumed_bootstrap, _ = resumed.collect_rollout()
            resumed_summary = [
                (record.episode_id, record.player_id, record.action, record.reward)
                for record in resumed_records
            ]
            resumed_metrics = resumed.update(resumed_records, resumed_bootstrap)

        self.assertEqual(direct_summary, resumed_summary)
        self.assertEqual(direct_bootstrap, resumed_bootstrap)
        self.assertEqual(direct_metrics, resumed_metrics)
        self.assertEqual(trainer.next_episode_id, resumed.next_episode_id)
        self.assertEqual(trainer.current_episode_id, resumed.current_episode_id)
        for key, expected in direct_parameters.items():
            torch.testing.assert_close(expected, resumed.model.state_dict()[key])

    def test_atomic_replace_failure_preserves_existing_checkpoint(self) -> None:
        trainer = self.trained_once()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint_atomic(path, trainer)
            original = path.read_bytes()
            with patch("os.replace", side_effect=OSError("injected replace failure")):
                with self.assertRaisesRegex(OSError, "injected replace failure"):
                    save_checkpoint_atomic(path, trainer)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_incompatible_catalog_is_rejected_with_named_version(self) -> None:
        trainer = self.trained_once()
        incompatible = replace(
            self.snapshot,
            catalog=replace(
                self.snapshot.catalog,
                catalog_sha256="0" * 64,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint_atomic(path, trainer)
            with self.assertRaisesRegex(ValueError, "catalog_sha256"):
                load_checkpoint(path, incompatible)

    def test_corrupt_checkpoint_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pt"
            path.write_bytes(b"not a torch checkpoint")
            with self.assertRaisesRegex(ValueError, "Unable to load checkpoint"):
                load_checkpoint(path, self.snapshot)

    def test_legacy_pre_embedding_checkpoint_is_rejected(self) -> None:
        payload = build_checkpoint(self.trained_once())
        payload["checkpoint_schema_version"] = CHECKPOINT_SCHEMA_VERSION - 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(
                ValueError, "Unsupported checkpoint schema version"
            ):
                load_checkpoint(path, self.snapshot)


if __name__ == "__main__":
    unittest.main()
