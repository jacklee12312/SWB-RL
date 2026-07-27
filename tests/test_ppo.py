from __future__ import annotations

import math
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from swb.db.repository import CardRepository
from swb.rl.ppo import (
    ObservationFlattener,
    PPOConfig,
    PPOTrainer,
    RecurrentMaskedActorCritic,
)
from swb.rl.opponents import OpponentPool
from swb.rl.fixed_decks import (
    OFFICIAL_QR_EVOLVE_HAVEN,
    get_fixed_training_deck,
)
from swb.rl.runtime import WorkerAssetsSnapshot


DATABASE = Path("data/cards.sqlite3")


class MaskedPolicyTests(unittest.TestCase):
    def test_training_class_ids_are_validated_and_frozen(self) -> None:
        config = PPOConfig(training_class_ids=[1, 2, 3])
        self.assertEqual(config.training_class_ids, (1, 2, 3))
        self.assertEqual(config.match_setup, "official")
        with self.assertRaises(ValueError):
            PPOConfig(training_class_ids=(1, 1))
        with self.assertRaisesRegex(ValueError, "match_setup"):
            PPOConfig(match_setup="unknown")
        with self.assertRaisesRegex(ValueError, "requires training_class_ids"):
            PPOConfig(training_deck=OFFICIAL_QR_EVOLVE_HAVEN)

    def test_multiprocess_rollout_rejects_unsupported_opponent_mixing(self) -> None:
        with self.assertRaisesRegex(ValueError, "self-play only"):
            PPOConfig(
                rollout_workers=2,
                opponent_current_weight=0.5,
                opponent_random_weight=0.5,
            )

    def test_card_slots_use_stable_indices_and_trainable_embeddings(self) -> None:
        observation = {
            "continuous": np.asarray([1.0, 2.0], dtype=np.float32),
            "own_hand_cards": np.asarray([5, 9], dtype=np.int32),
            "public_board_cards": np.asarray([0, 3], dtype=np.int32),
            "action_mask": np.ones(4, dtype=np.int8),
        }
        flattener = ObservationFlattener.from_observation(observation)
        np.testing.assert_array_equal(
            flattener.encode_cards(observation),
            np.asarray([5, 9, 0, 3], dtype=np.int64),
        )
        changed_companions = dict(observation)
        changed_companions["own_hand_cards"] = np.asarray(
            [5, 100], dtype=np.int32
        )
        self.assertEqual(flattener.encode_cards(changed_companions)[0], 5)

        model = RecurrentMaskedActorCritic(
            flattener.size,
            action_size=4,
            hidden_size=8,
            card_vocabulary_size=100,
            card_slot_count=flattener.card_slots,
            card_embedding_dim=4,
        )
        numeric = torch.from_numpy(flattener.encode(observation)).unsqueeze(0)
        cards = torch.from_numpy(flattener.encode_cards(observation)).unsqueeze(0)
        logits, value, _ = model.forward_step(
            numeric,
            model.initial_state(1, device=torch.device("cpu")),
            cards,
        )
        (logits.sum() + value.sum()).backward()
        self.assertIsNotNone(model.card_embedding)
        gradient = model.card_embedding.weight.grad
        self.assertGreater(float(gradient[5].abs().sum()), 0.0)
        self.assertEqual(float(gradient[0].abs().sum()), 0.0)
        self.assertEqual(
            float(model.card_embedding.weight[0].detach().abs().sum()), 0.0
        )

    def test_illegal_logits_receive_zero_probability(self) -> None:
        logits = torch.tensor([[100.0, 1.0, 50.0, 0.0]])
        mask = torch.tensor([[False, True, False, True]])
        masked = RecurrentMaskedActorCritic.masked_logits(logits, mask)
        probabilities = torch.softmax(masked, dim=-1)
        self.assertEqual(probabilities[0, 0].item(), 0.0)
        self.assertEqual(probabilities[0, 2].item(), 0.0)
        self.assertAlmostEqual(probabilities.sum().item(), 1.0)

    def test_empty_or_mismatched_mask_is_rejected(self) -> None:
        logits = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, "legal action"):
            RecurrentMaskedActorCritic.masked_logits(
                logits, torch.zeros(1, 3, dtype=torch.bool)
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            RecurrentMaskedActorCritic.masked_logits(
                logits, torch.ones(1, 2, dtype=torch.bool)
            )


@unittest.skipUnless(DATABASE.exists(), "real card database is unavailable")
class PPOTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = WorkerAssetsSnapshot.build(CardRepository(DATABASE))

    def make_trainer(self) -> PPOTrainer:
        return PPOTrainer(
            self.snapshot,
            master_seed=777,
            config=PPOConfig(
                rollout_steps=16,
                sequence_length=4,
                minibatch_sequences=2,
                update_epochs=1,
                hidden_size=32,
                max_agent_steps_per_episode=8,
            ),
        )

    def test_collection_uses_shared_policy_with_separate_player_state(self) -> None:
        trainer = self.make_trainer()
        records, bootstrap, boundaries = trainer.collect_rollout()
        self.assertEqual(len(records), 16)
        self.assertEqual({record.player_id for record in records}, {0, 1})
        self.assertEqual(set(trainer.hidden_by_player), {0, 1})
        self.assertIsNot(
            trainer.hidden_by_player[0], trainer.hidden_by_player[1]
        )
        for record in records:
            self.assertTrue(record.action_mask[record.action])
            self.assertTrue(math.isfinite(record.old_log_prob))
            self.assertTrue(math.isfinite(record.value))
        for (episode_id, player_id), value in bootstrap.items():
            self.assertIn(player_id, (0, 1))
            self.assertIn(episode_id, boundaries)
            self.assertTrue(math.isfinite(value))

    def test_episode_schedule_changes_training_classes_deterministically(self) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=779,
            config=PPOConfig(
                rollout_steps=4,
                hidden_size=16,
                max_agent_steps_per_episode=4,
                training_class_ids=(1, 2),
            ),
        )
        self.assertEqual(trainer.env._core.player_classes, (1, 1))
        trainer._start_episode()
        self.assertEqual(trainer.env._core.player_classes, (1, 2))
        assignment = trainer.opponent_assignments[-1]
        self.assertEqual((assignment["class_a"], assignment["class_b"]), (1, 2))

    def test_fixed_training_deck_is_used_for_both_players(self) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=780,
            config=PPOConfig(
                rollout_steps=8,
                hidden_size=16,
                max_agent_steps_per_episode=8,
                training_class_ids=(6,),
                training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
            ),
        )
        recipe = get_fixed_training_deck(OFFICIAL_QR_EVOLVE_HAVEN)
        expected = Counter(recipe.card_ids)
        self.assertEqual(trainer.env._core.player_classes, (6, 6))
        for deck in trainer.env._core.deck_lists:
            self.assertEqual(Counter(card.card_id for card in deck), expected)
        assignment = trainer.opponent_assignments[-1]
        self.assertEqual(
            assignment["training_deck"],
            OFFICIAL_QR_EVOLVE_HAVEN,
        )
        records, _, _ = trainer.collect_rollout()
        self.assertEqual(len(records), 8)
        self.assertTrue(all(record.action_mask[record.action] for record in records))

    def test_update_changes_parameters_and_reports_finite_metrics(self) -> None:
        trainer = self.make_trainer()
        records, bootstrap, _ = trainer.collect_rollout()
        before = {
            key: value.detach().clone()
            for key, value in trainer.model.state_dict().items()
        }
        metrics = trainer.update(records, bootstrap)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertTrue(any(
            not torch.equal(before[key], value)
            for key, value in trainer.model.state_dict().items()
        ))

    def test_external_fixed_opponent_actions_are_executed_but_not_trained(self) -> None:
        trainer = self.make_trainer()
        trainer.opponent_pool = OpponentPool(
            trainer.master_seed,
            current_weight=0.0,
            random_weight=0.0,
            fixed_weight=1.0,
        )
        trainer._start_episode()
        records, bootstrap, _ = trainer.collect_rollout()
        opponent_records = [record for record in records if not record.trainable]
        self.assertTrue(opponent_records)
        self.assertTrue(all(not record.trainable for record in opponent_records))
        self.assertTrue(all(record.opponent_id == "fixed_first_legal" for record in records))
        learner_records = [record for record in records if record.trainable]
        self.assertTrue(learner_records)
        self.assertTrue(all(record.player_id == record.episode_id % 2 for record in learner_records))
        metrics = trainer.update(records, bootstrap)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_seeded_short_training_is_reproducible(self) -> None:
        summaries = []
        for _ in range(2):
            trainer = self.make_trainer()
            records, bootstrap, _ = trainer.collect_rollout()
            metrics = trainer.update(records, bootstrap)
            summaries.append((
                [(record.player_id, record.action, record.reward) for record in records],
                metrics,
                {
                    key: value.detach().cpu().numpy().copy()
                    for key, value in trainer.model.state_dict().items()
                },
            ))
        self.assertEqual(summaries[0][0], summaries[1][0])
        self.assertEqual(summaries[0][1], summaries[1][1])
        for key in summaries[0][2]:
            np.testing.assert_array_equal(
                summaries[0][2][key], summaries[1][2][key]
            )

    def test_multiprocess_policy_rollout_is_persistent_and_trainable(self) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=778,
            config=PPOConfig(
                rollout_steps=8,
                rollout_workers=2,
                sequence_length=4,
                minibatch_sequences=2,
                update_epochs=1,
                hidden_size=16,
                max_agent_steps_per_episode=8,
            ),
        )
        try:
            records, bootstrap, boundaries = trainer.collect_rollout()
            self.assertGreaterEqual(len(records), 8)
            self.assertEqual(len(boundaries), 2)
            rollout = trainer._policy_vector_rollout
            self.assertIsNotNone(rollout)
            process_ids = tuple(process.pid for process in rollout.processes)
            self.assertTrue(all(process.is_alive() for process in rollout.processes))
            metrics = trainer.update(records, bootstrap)
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
            trainer.collect_rollout()
            self.assertEqual(
                process_ids,
                tuple(process.pid for process in rollout.processes),
            )
        finally:
            trainer.close()
        self.assertFalse(any(process.is_alive() for process in rollout.processes))


if __name__ == "__main__":
    unittest.main()
