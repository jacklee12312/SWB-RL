from __future__ import annotations

import pickle
import unittest
from pathlib import Path

import torch

from swb.db.repository import CardRepository
from swb.rl.evaluation import EvaluationConfig, evaluate
from swb.rl.ppo import PPOConfig, PPOTrainer
from swb.rl.runtime import WorkerAssetsSnapshot


DATABASE = Path("data/cards.sqlite3")


@unittest.skipUnless(DATABASE.exists(), "real card database is unavailable")
class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = WorkerAssetsSnapshot.build(CardRepository(DATABASE))

    def make_trainer(self) -> PPOTrainer:
        return PPOTrainer(
            self.snapshot,
            master_seed=1234,
            config=PPOConfig(
                rollout_steps=8,
                sequence_length=4,
                hidden_size=32,
                max_agent_steps_per_episode=8,
            ),
        )

    def test_fixed_seed_mirrored_evaluation_is_reproducible(self) -> None:
        trainer = self.make_trainer()
        config = EvaluationConfig(
            master_seed=55,
            seed_count=1,
            max_agent_steps=12,
            opponent_kind="random_legal",
            class_ids=(1,),
        )
        first = evaluate(trainer, self.snapshot, config)
        second = evaluate(trainer, self.snapshot, config)
        self.assertEqual(first, second)
        self.assertIn("terminated_rate", first["metrics"])
        self.assertIn("truncated_rate", first["metrics"])
        self.assertIn("card_coverage_rate", first["coverage"])
        self.assertIn("class_coverage_rate", first["coverage"])
        self.assertIn("mechanism_coverage_rate", first["coverage"])
        self.assertEqual(first["configuration"]["mirrored_games"], 2)
        self.assertEqual(first["schema_version"], 2)
        self.assertEqual(first["configuration"]["class_ids"], [1])
        self.assertEqual(set(first["metrics"]["per_class"]), {"1"})
        self.assertEqual(len(first["decks"]), 2)
        self.assertTrue(first["evaluation_suite_sha256"])
        self.assertIn("training_pool_sha256", first["versions"])
        self.assertEqual(set(first["metrics"]["side_win_rates"]), {"0", "1"})
        self.assertEqual(first["metrics"]["illegal_action_rate"], 0.0)
        self.assertEqual(first["metrics"]["action_mask_mismatches"], 0)
        self.assertTrue(first["coverage"]["card_ids"])
        self.assertTrue(first["coverage"]["classes"])
        self.assertTrue(first["coverage"]["mechanisms"])

    def test_evaluation_does_not_change_training_or_rng_state(self) -> None:
        trainer = self.make_trainer()
        model_before = {
            key: value.detach().clone()
            for key, value in trainer.model.state_dict().items()
        }
        optimizer_before = pickle.dumps(trainer.optimizer.state_dict())
        generator_before = trainer.torch_generator.get_state().clone()
        environment_before = trainer.env._core.deterministic_fingerprint()
        progress_before = (
            trainer.agent_steps,
            trainer.completed_episodes,
            trainer.next_episode_id,
            trainer.current_episode_id,
        )
        evaluate(
            trainer,
            self.snapshot,
            EvaluationConfig(
                seed_count=1,
                max_agent_steps=8,
                opponent_kind="fixed",
                class_ids=(1,),
            ),
        )
        for key, value in model_before.items():
            torch.testing.assert_close(value, trainer.model.state_dict()[key])
        self.assertEqual(optimizer_before, pickle.dumps(trainer.optimizer.state_dict()))
        torch.testing.assert_close(generator_before, trainer.torch_generator.get_state())
        self.assertEqual(environment_before, trainer.env._core.deterministic_fingerprint())
        self.assertEqual(
            progress_before,
            (
                trainer.agent_steps,
                trainer.completed_episodes,
                trainer.next_episode_id,
                trainer.current_episode_id,
            ),
        )


if __name__ == "__main__":
    unittest.main()
