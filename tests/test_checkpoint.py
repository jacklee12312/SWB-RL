from __future__ import annotations

import tempfile
import unittest
from collections import Counter
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
from swb.rl.fixed_decks import (
    OFFICIAL_QR_EVOLVE_HAVEN,
    get_fixed_training_deck,
)
from swb.rl.ppo import PPOConfig, PPOTrainer
from swb.rl.policy import ENTITY_ACTION_POLICY_ARCHITECTURE
from swb.rl.runtime import WorkerAssetsSnapshot


DATABASE = Path("data/cards.sqlite3")
SPECIALIST_OPPONENT_DECKS = (
    "international_qr_forest_20260728",
    "international_qr_sword_20260728",
)


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
        trainer = self.trained_once()
        payload = build_checkpoint(trainer)
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
        self.assertEqual(
            payload["experiment_manifest"]["policy_representation"][
                "architecture"
            ],
            trainer.model.architecture,
        )

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

    def test_batched_v41_learner_resume_next_update_drift_is_bounded(
        self,
    ) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=2477,
            config=PPOConfig(
                rollout_steps=8,
                rollout_workers=2,
                sequence_length=4,
                minibatch_sequences=2,
                update_epochs=1,
                hidden_size=64,
                card_embedding_dim=16,
                policy_architecture=(
                    ENTITY_ACTION_POLICY_ARCHITECTURE
                ),
                observation_version="v4.1",
                model_dim=32,
                transformer_layers=1,
                attention_heads=4,
                feedforward_dim=64,
                max_agent_steps_per_episode=8,
                training_class_ids=(6,),
                training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
            ),
        )
        resumed = None
        try:
            records, bootstrap, _ = trainer.collect_rollout()
            trainer.update(records, bootstrap)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "batched-v41.pt"
                save_checkpoint_atomic(path, trainer)
                direct_records, direct_bootstrap, _ = (
                    trainer.collect_rollout()
                )
                direct_metrics = trainer.update(
                    direct_records, direct_bootstrap
                )
                direct_parameters = {
                    key: value.detach().clone()
                    for key, value in (
                        trainer.model.state_dict().items()
                    )
                }
                resumed = load_checkpoint(path, self.snapshot)
                resumed_records, resumed_bootstrap, _ = (
                    resumed.collect_rollout()
                )
                resumed_metrics = resumed.update(
                    resumed_records, resumed_bootstrap
                )
            self.assertEqual(
                [record.action for record in direct_records],
                [record.action for record in resumed_records],
            )
            for direct_record, resumed_record in zip(
                direct_records,
                resumed_records,
                strict=True,
            ):
                self.assertAlmostEqual(
                    direct_record.old_log_prob,
                    resumed_record.old_log_prob,
                    delta=1e-7,
                )
                self.assertAlmostEqual(
                    direct_record.value,
                    resumed_record.value,
                    delta=1e-7,
                )
            for player_id, expected in direct_bootstrap.items():
                self.assertAlmostEqual(
                    expected,
                    resumed_bootstrap[player_id],
                    delta=1e-7,
                )
            self.assertEqual(direct_metrics.keys(), resumed_metrics.keys())
            for name, expected in direct_metrics.items():
                self.assertAlmostEqual(
                    expected,
                    resumed_metrics[name],
                    delta=1e-6,
                )
            max_parameter_drift = max(
                (
                    expected
                    - resumed.model.state_dict()[key]
                ).abs().max().item()
                for key, expected in direct_parameters.items()
            )
            self.assertTrue(torch.isfinite(torch.tensor(max_parameter_drift)))
            self.assertLess(max_parameter_drift, 1e-3)
        finally:
            trainer.close()
            if resumed is not None:
                resumed.close()

    def test_fixed_training_deck_is_manifested_and_resumed(self) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=2469,
            config=PPOConfig(
                rollout_steps=4,
                hidden_size=16,
                max_agent_steps_per_episode=4,
                training_class_ids=(6,),
                training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
            ),
        )
        payload = build_checkpoint(trainer)
        manifest = payload["experiment_manifest"]["training_deck"]
        recipe = get_fixed_training_deck(OFFICIAL_QR_EVOLVE_HAVEN)
        self.assertEqual(manifest["name"], recipe.name)
        self.assertEqual(manifest["sha256"], recipe.sha256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.pt"
            save_checkpoint_atomic(path, trainer)
            resumed = load_checkpoint(path, self.snapshot)
        self.assertEqual(resumed.config.training_deck, recipe.name)
        for deck in resumed.env._core.deck_lists:
            self.assertEqual(Counter(card.card_id for card in deck), Counter(recipe.card_ids))

    def test_entity_action_architecture_round_trips_through_checkpoint(self) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=2470,
            config=PPOConfig(
                rollout_steps=4,
                sequence_length=2,
                minibatch_sequences=1,
                update_epochs=1,
                hidden_size=64,
                card_embedding_dim=16,
                policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
                model_dim=32,
                transformer_layers=1,
                attention_heads=4,
                feedforward_dim=64,
                max_agent_steps_per_episode=4,
            ),
        )
        records, bootstrap, _ = trainer.collect_rollout()
        trainer.update(records, bootstrap)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entity_action.pt"
            save_checkpoint_atomic(path, trainer)
            resumed = load_checkpoint(path, self.snapshot)
        self.assertEqual(
            resumed.model.architecture,
            ENTITY_ACTION_POLICY_ARCHITECTURE,
        )
        self.assertEqual(
            resumed.model.specification(),
            trainer.model.specification(),
        )
        for name, expected in trainer.model.state_dict().items():
            torch.testing.assert_close(expected, resumed.model.state_dict()[name])

    def test_v4_1_checkpoint_round_trips_with_structured_policy(self) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=2473,
            config=PPOConfig(
                rollout_steps=2,
                sequence_length=2,
                minibatch_sequences=1,
                update_epochs=1,
                hidden_size=32,
                card_embedding_dim=8,
                policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
                observation_version="v4.1",
                model_dim=32,
                transformer_layers=1,
                attention_heads=4,
                feedforward_dim=64,
                max_agent_steps_per_episode=2,
            ),
        )
        records, bootstrap, _ = trainer.collect_rollout()
        metrics = trainer.update(records, bootstrap)
        self.assertTrue(torch.isfinite(torch.tensor(
            metrics["policy_loss"] + metrics["value_loss"]
        )))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observation_v4_1.pt"
            save_checkpoint_atomic(path, trainer)
            resumed = load_checkpoint(path, self.snapshot)
        self.assertEqual(resumed.config.observation_version, "v4.1")
        self.assertEqual(resumed.env.observation_version, "v4.1")
        self.assertEqual(resumed.model.structured_token_count, 93)
        self.assertEqual(
            resumed.model.specification(),
            trainer.model.specification(),
        )
        for name, expected in trainer.model.state_dict().items():
            torch.testing.assert_close(expected, resumed.model.state_dict()[name])

    def test_v3_checkpoint_without_config_field_keeps_legacy_observation(
        self,
    ) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=2472,
            config=PPOConfig(
                rollout_steps=4,
                hidden_size=16,
                max_agent_steps_per_episode=4,
                observation_version="v3",
            ),
        )
        payload = build_checkpoint(trainer)
        payload["trainer"]["config"].pop("observation_version")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_v3.pt"
            torch.save(payload, path)
            resumed = load_checkpoint(path, self.snapshot)
        self.assertEqual(resumed.config.observation_version, "v3")
        self.assertEqual(resumed.env.observation_version, "v3")

    def test_specialist_deck_schedule_and_statistics_round_trip(self) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=2471,
            config=PPOConfig(
                rollout_steps=16,
                rollout_workers=2,
                hidden_size=16,
                max_agent_steps_per_episode=8,
                training_class_ids=(6,),
                training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
                opponent_decks=SPECIALIST_OPPONENT_DECKS,
            ),
        )
        try:
            trainer.collect_rollout()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "specialist.pt"
                save_checkpoint_atomic(path, trainer)
                payload = build_checkpoint(trainer)
                resumed = load_checkpoint(path, self.snapshot)
            self.assertEqual(
                resumed.config.opponent_decks,
                SPECIALIST_OPPONENT_DECKS,
            )
            self.assertEqual(
                resumed.matchup_statistics,
                trainer.matchup_statistics,
            )
            self.assertTrue(all(
                "learner_player_0" in stats
                and "learner_player_1" in stats
                for stats in resumed.matchup_statistics.values()
            ))
            self.assertEqual(
                [
                    manifest["name"]
                    for manifest in payload[
                        "experiment_manifest"
                    ]["opponent_decks"]
                ],
                list(SPECIALIST_OPPONENT_DECKS),
            )
        finally:
            trainer.close()
            if "resumed" in locals():
                resumed.close()

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
