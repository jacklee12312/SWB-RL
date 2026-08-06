from __future__ import annotations

import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from swb.db.repository import CardRepository
from swb.rl.checkpoint import save_checkpoint_atomic
from swb.rl.ppo import (
    ObservationFlattener,
    PPOConfig,
    PPOTrainer,
    RecurrentMaskedActorCritic,
    build_policy,
)
from swb.rl.policy import (
    ENTITY_ACTION_POLICY_ARCHITECTURE,
    LEGACY_POLICY_ARCHITECTURE,
    EntityActionRecurrentActorCritic,
)
from swb.rl.opponents import OpponentEpisodeScheduler, OpponentPool
from swb.rl.fixed_decks import (
    OFFICIAL_QR_EVOLVE_HAVEN,
    get_fixed_training_deck,
)
from swb.rl.runtime import WorkerAssetsSnapshot


DATABASE = Path("data/cards.sqlite3")
SPECIALIST_OPPONENT_DECKS = (
    "international_qr_forest_20260728",
    "international_qr_sword_20260728",
)


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
        with self.assertRaisesRegex(ValueError, "requires one fixed"):
            PPOConfig(opponent_decks=SPECIALIST_OPPONENT_DECKS)
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            PPOConfig(
                training_class_ids=(6,),
                training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
                opponent_decks=(
                    SPECIALIST_OPPONENT_DECKS[0],
                    SPECIALIST_OPPONENT_DECKS[0],
                ),
            )
        self.assertEqual(
            PPOConfig().policy_architecture,
            LEGACY_POLICY_ARCHITECTURE,
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            PPOConfig(
                policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
                model_dim=30,
                attention_heads=8,
            )

    def test_multiprocess_rollout_rejects_unsupported_opponent_mixing(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero random/fixed"):
            PPOConfig(
                rollout_workers=2,
                opponent_current_weight=0.5,
                opponent_random_weight=0.5,
            )
        config = PPOConfig(
            rollout_workers=2,
            opponent_current_weight=1.0,
            opponent_historical_weight=0.25,
        )
        self.assertEqual(config.opponent_historical_weight, 0.25)
        external = PPOConfig(
            rollout_workers=2,
            opponent_current_weight=0.0,
            opponent_external_manifest="generation.json",
            opponent_external_weight=1.0,
            opponent_model_cache_size=2,
        )
        self.assertEqual(external.opponent_external_weight, 1.0)
        clustered = PPOConfig(
            rollout_workers=3,
            opponent_current_weight=0.0,
            opponent_external_manifest="generation.json",
            opponent_external_weight=1.0,
            opponent_model_cache_size=1,
            opponent_batching_mode="episode_seed_clustered",
        )
        self.assertEqual(clustered.opponent_model_cache_size, 1)
        with self.assertRaisesRegex(ValueError, "requires opponent_external_manifest"):
            PPOConfig(
                opponent_current_weight=0.0,
                opponent_external_weight=1.0,
            )
        with self.assertRaisesRegex(ValueError, "at least rollout_workers"):
            PPOConfig(
                rollout_workers=3,
                opponent_current_weight=0.0,
                opponent_external_manifest="generation.json",
                opponent_external_weight=1.0,
                opponent_model_cache_size=2,
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

    @staticmethod
    def _entity_action_observation() -> dict[str, np.ndarray]:
        choice = np.zeros(36, dtype=np.int32)
        choice[:4] = (1, 2, 1, 0)
        choice[4:6] = (1, 6)
        board_cards = np.zeros(10, dtype=np.int32)
        board_cards[0] = 5
        board_cards[5] = 9
        return {
            "continuous": np.asarray([0.25, 0.75], dtype=np.float32),
            "own_hand_cards": np.zeros(9, dtype=np.int32),
            "public_board_cards": board_cards,
            "own_hand_origins": np.zeros(9, dtype=np.int32),
            "public_board_origins": np.zeros(10, dtype=np.int32),
            "own_hand_runtime": np.zeros(126, dtype=np.float32),
            "public_board_runtime": np.zeros(230, dtype=np.float32),
            "public_board_keywords": np.zeros(90, dtype=np.int8),
            "choice_categorical": choice,
            "action_mask": np.ones(112, dtype=np.int8),
        }

    def test_entity_action_choice_scores_follow_candidate_not_option_position(
        self,
    ) -> None:
        observation = self._entity_action_observation()
        flattener = ObservationFlattener.from_observation(observation)
        config = PPOConfig(
            policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
            hidden_size=32,
            card_embedding_dim=8,
            model_dim=32,
            transformer_layers=1,
            attention_heads=4,
            feedforward_dim=64,
        )
        model = build_policy(
            config,
            flattener,
            action_size=112,
            card_vocabulary_size=100,
        )
        self.assertIsInstance(model, EntityActionRecurrentActorCritic)
        model.eval()

        def forward(candidate_observation):
            numeric = torch.from_numpy(
                flattener.encode(candidate_observation)
            ).unsqueeze(0)
            cards = torch.from_numpy(
                flattener.encode_cards(candidate_observation)
            ).unsqueeze(0)
            with torch.no_grad():
                return model.forward_step(
                    numeric,
                    model.initial_state(1, device=torch.device("cpu")),
                    cards,
                )

        first_logits, first_value, _ = forward(observation)
        swapped = {
            name: value.copy() for name, value in observation.items()
        }
        swapped["choice_categorical"][4:6] = (6, 1)
        second_logits, second_value, _ = forward(swapped)
        choice = 45
        torch.testing.assert_close(
            first_logits[0, choice],
            second_logits[0, choice + 1],
        )
        torch.testing.assert_close(
            first_logits[0, choice + 1],
            second_logits[0, choice],
        )
        torch.testing.assert_close(first_value, second_value)

    def test_standard_entity_action_configuration_is_multi_million_parameter(
        self,
    ) -> None:
        observation = self._entity_action_observation()
        flattener = ObservationFlattener.from_observation(observation)
        model = build_policy(
            PPOConfig(
                policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
                hidden_size=512,
                card_embedding_dim=128,
            ),
            flattener,
            action_size=112,
            card_vocabulary_size=826,
        )
        parameter_count = sum(
            parameter.numel() for parameter in model.parameters()
        )
        self.assertGreater(parameter_count, 5_000_000)
        self.assertLess(parameter_count, 10_000_000)


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

    def test_specialist_rollout_trains_only_haven_against_deck_cycle(
        self,
    ) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=783,
            config=PPOConfig(
                rollout_steps=16,
                rollout_workers=2,
                sequence_length=4,
                minibatch_sequences=2,
                update_epochs=1,
                hidden_size=16,
                max_agent_steps_per_episode=8,
                training_class_ids=(6,),
                training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
                opponent_decks=SPECIALIST_OPPONENT_DECKS,
            ),
        )
        try:
            records, bootstrap, _ = trainer.collect_rollout()
            assignments = {
                int(assignment["episode_id"]): assignment
                for assignment in trainer.opponent_assignments
            }
            self.assertEqual(
                {
                    assignment["opponent_deck"]
                    for assignment in assignments.values()
                },
                {SPECIALIST_OPPONENT_DECKS[0]},
            )
            self.assertTrue(any(record.trainable for record in records))
            self.assertTrue(any(not record.trainable for record in records))
            for record in records:
                with self.subTest(
                    episode_id=record.episode_id,
                    player_id=record.player_id,
                ):
                    self.assertEqual(
                        record.trainable,
                        record.player_id
                        == int(
                            assignments[record.episode_id]["learner_player"]
                        ),
                    )
            trainer.update(records, bootstrap)
            trainer.collect_rollout()
            self.assertEqual(
                {
                    stats["opponent_deck"]
                    for stats in trainer.matchup_statistics.values()
                },
                set(SPECIALIST_OPPONENT_DECKS),
            )
        finally:
            trainer.close()

    def test_multiprocess_history_league_trains_only_current_side(
        self,
    ) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=784,
            config=PPOConfig(
                rollout_steps=16,
                rollout_workers=2,
                sequence_length=4,
                minibatch_sequences=2,
                update_epochs=1,
                hidden_size=16,
                max_agent_steps_per_episode=8,
                training_class_ids=(1, 2),
                opponent_current_weight=1.0,
                opponent_historical_weight=1.0,
                opponent_snapshot_interval_steps=8,
            ),
        )
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint = Path(temp_dir) / "history.pt"
                save_checkpoint_atomic(checkpoint, trainer)
                trainer.opponent_pool = OpponentPool(
                    trainer.master_seed,
                    current_weight=0.0,
                    historical_weight=1.0,
                    snapshot_interval_steps=8,
                )
                history = trainer.opponent_pool.register_snapshot(
                    checkpoint,
                    agent_steps=8,
                )
                trainer.opponent_scheduler = OpponentEpisodeScheduler(
                    trainer.opponent_pool,
                    worker_count=trainer.config.rollout_workers,
                    mode=trainer.config.opponent_batching_mode,
                )
                records, bootstrap, _ = trainer.collect_rollout()
            assignments = {
                int(assignment["episode_id"]): assignment
                for assignment in trainer.opponent_assignments
            }
            self.assertEqual(
                {assignment["opponent_kind"] for assignment in assignments.values()},
                {"historical"},
            )
            self.assertEqual(
                {assignment["opponent_id"] for assignment in assignments.values()},
                {history.opponent_id},
            )
            self.assertTrue(any(record.trainable for record in records))
            self.assertTrue(any(not record.trainable for record in records))
            for record in records:
                self.assertEqual(
                    record.trainable,
                    record.player_id
                    == int(assignments[record.episode_id]["learner_player"]),
                )
            metrics = trainer.update(records, bootstrap)
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        finally:
            trainer.close()

    def test_entity_action_policy_collects_and_updates_fixed_deck(self) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=782,
            config=PPOConfig(
                rollout_steps=8,
                rollout_workers=2,
                sequence_length=4,
                minibatch_sequences=2,
                update_epochs=1,
                hidden_size=64,
                card_embedding_dim=16,
                policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
                observation_version="v4.1",
                model_dim=32,
                transformer_layers=1,
                attention_heads=4,
                feedforward_dim=64,
                max_agent_steps_per_episode=8,
                profile_ipc_timing=True,
                profile_central_timing=True,
                profile_learner_timing=True,
                training_class_ids=(6,),
                training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
            ),
        )
        try:
            records, bootstrap, _ = trainer.collect_rollout()
            first = records[0]
            trainer.model.eval()
            with torch.no_grad():
                logits, value, _ = trainer.model.forward_step(
                    torch.from_numpy(first.observation).unsqueeze(0),
                    torch.from_numpy(first.hidden_before).unsqueeze(0),
                    torch.from_numpy(first.card_indices).unsqueeze(0),
                )
                masked = trainer.model.masked_logits(
                    logits,
                    torch.from_numpy(first.action_mask).unsqueeze(0),
                )
                recomputed_log_prob = torch.log_softmax(
                    masked,
                    dim=-1,
                )[0, first.action]
            self.assertAlmostEqual(
                float(recomputed_log_prob.item()),
                first.old_log_prob,
                places=6,
            )
            self.assertAlmostEqual(
                float(value.item()),
                first.value,
                places=6,
            )
            trainer.model.train()
            metrics = trainer.update(records, bootstrap)
            self.assertEqual(
                trainer.model.architecture,
                ENTITY_ACTION_POLICY_ARCHITECTURE,
            )
            self.assertTrue(
                all(math.isfinite(value) for value in metrics.values())
            )
            self.assertEqual(
                trainer.last_collect_timing["records"],
                float(len(records)),
            )
            self.assertEqual(
                trainer.last_collect_timing["worker_agent_steps"],
                float(len(records)),
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_inference_round_trip_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_ipc_request_serialization_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_ipc_request_send_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_ipc_response_wait_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_ipc_request_payload_bytes"
                ],
                0.0,
            )
            self.assertEqual(
                trainer.last_collect_timing[
                    "worker_ipc_profiled_requests"
                ],
                float(len(records)),
            )
            self.assertLessEqual(
                (
                    trainer.last_collect_timing[
                        "worker_ipc_request_serialization_seconds"
                    ]
                    + trainer.last_collect_timing[
                        "worker_ipc_request_send_seconds"
                    ]
                    + trainer.last_collect_timing[
                        "worker_ipc_response_wait_seconds"
                    ]
                ),
                trainer.last_collect_timing[
                    "worker_inference_round_trip_seconds"
                ],
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_ipc_accounted_fraction_of_round_trip"
                ],
                0.99,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_ipc_request_payload_bytes_per_request"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing["worker_engine_step_seconds"],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_deck_construction_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_environment_construction_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing["worker_reset_seconds"],
                0.0,
            )
            self.assertEqual(
                trainer.last_collect_timing["worker_mulligan_steps"],
                4.0,
            )
            self.assertGreater(
                trainer.last_collect_timing["worker_mulligan_seconds"],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_response_queue_wait_seconds"
                ],
                0.0,
            )
            self.assertGreaterEqual(
                trainer.last_collect_timing[
                    "worker_assignment_wait_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing["worker_idle_fraction"],
                0.0,
            )
            self.assertLessEqual(
                trainer.last_collect_timing["worker_idle_fraction"],
                1.0,
            )
            self.assertEqual(
                trainer.last_collect_timing["worker_episode_count"],
                2.0,
            )
            self.assertEqual(
                trainer.last_collect_timing[
                    "worker_terminated_episode_count"
                ]
                + trainer.last_collect_timing[
                    "worker_truncated_episode_count"
                ],
                trainer.last_collect_timing["worker_episode_count"],
            )
            self.assertLessEqual(
                trainer.last_collect_timing["worker_episode_steps_p50"],
                trainer.last_collect_timing["worker_episode_steps_p95"],
            )
            self.assertLessEqual(
                trainer.last_collect_timing["worker_episode_steps_p95"],
                trainer.last_collect_timing["worker_episode_steps_max"],
            )
            self.assertGreater(
                trainer.last_collect_timing["worker_action_mask_seconds"],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "worker_observation_construction_seconds"
                ],
                0.0,
            )
            self.assertAlmostEqual(
                trainer.last_collect_timing[
                    "worker_observation_construction_seconds"
                ],
                (
                    trainer.last_collect_timing[
                        "worker_decision_observation_construction_seconds"
                    ]
                    + trainer.last_collect_timing[
                        "worker_step_observation_construction_seconds"
                    ]
                    + trainer.last_collect_timing[
                        "worker_bootstrap_observation_construction_seconds"
                    ]
                ),
            )
            self.assertGreater(
                trainer.last_collect_timing["worker_command_decode_seconds"],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing["worker_resolution_seconds"],
                0.0,
            )
            self.assertLessEqual(
                (
                    trainer.last_collect_timing["worker_action_mask_seconds"]
                    + trainer.last_collect_timing[
                        "worker_command_decode_seconds"
                    ]
                    + trainer.last_collect_timing[
                        "worker_resolution_seconds"
                    ]
                ),
                trainer.last_collect_timing["worker_engine_step_seconds"],
            )
            self.assertEqual(
                trainer.last_collect_timing["central_inference_requests"],
                float(len(records)),
            )
            self.assertGreater(
                trainer.last_collect_timing["central_forward_seconds"],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "central_cpu_input_assembly_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "central_cpu_input_bytes_per_request"
                ],
                0.0,
            )
            self.assertGreaterEqual(
                trainer.last_collect_timing[
                    "central_queue_to_batch_wait_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "central_model_forward_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "central_transformer_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing["central_gru_seconds"],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "central_action_value_stage_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "central_policy_head_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "central_value_head_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "central_masked_distribution_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "central_device_to_host_seconds"
                ],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing["central_sampling_seconds"],
                0.0,
            )
            self.assertGreater(
                trainer.last_collect_timing[
                    "central_result_distribution_seconds"
                ],
                0.0,
            )
            self.assertEqual(
                trainer.last_collect_timing["central_profiled_batches"],
                trainer.last_collect_timing[
                    "central_inference_batches"
                ],
            )
            self.assertAlmostEqual(
                trainer.last_collect_timing[
                    "central_gpu_busy_fraction_of_busy_plus_worker_wait"
                ]
                + trainer.last_collect_timing[
                    "central_gpu_worker_wait_fraction_of_busy_plus_worker_wait"
                ],
                1.0,
            )
            self.assertLessEqual(
                trainer.last_collect_timing["central_batch_size_p50"],
                trainer.last_collect_timing["central_batch_size_p95"],
            )
            self.assertLessEqual(
                trainer.last_collect_timing["central_batch_size_p95"],
                trainer.last_collect_timing["central_batch_size_max"],
            )
            self.assertGreaterEqual(
                trainer.last_collect_timing[
                    "central_batch_empty_slot_fraction"
                ],
                0.0,
            )
            self.assertLessEqual(
                trainer.last_collect_timing[
                    "central_batch_empty_slot_fraction"
                ],
                1.0,
            )
            self.assertEqual(
                trainer.last_collect_timing["policy_transmitted_bytes"],
                0.0,
            )
            self.assertGreaterEqual(
                trainer.last_collect_timing[
                    "central_average_batch_size"
                ],
                1.0,
            )
            self.assertEqual(
                trainer.last_collect_timing["worker_torch_threads"],
                2.0,
            )
            self.assertTrue(trainer.model.training)
            self.assertGreater(
                trainer.last_update_timing["forward_loss_seconds"],
                0.0,
            )
            self.assertGreater(
                trainer.last_update_timing["backward_clip_seconds"],
                0.0,
            )
            self.assertEqual(
                trainer.last_update_timing[
                    "learner_profiled_minibatches"
                ],
                trainer.last_update_timing["minibatches"],
            )
            for field in (
                "learner_padding_and_numpy_seconds",
                "learner_cpu_tensor_construction_seconds",
                "learner_host_to_device_seconds",
                "learner_forward_seconds",
                "learner_loss_seconds",
                "learner_backward_seconds",
                "learner_gradient_clip_seconds",
                "learner_optimizer_seconds",
            ):
                with self.subTest(field=field):
                    self.assertGreater(
                        trainer.last_update_timing[field],
                        0.0,
                    )
            self.assertEqual(
                trainer.last_update_timing["learner_effective_tokens"]
                + trainer.last_update_timing["learner_padding_tokens"],
                trainer.last_update_timing["learner_token_slots"],
            )
            self.assertAlmostEqual(
                trainer.last_update_timing[
                    "learner_effective_token_fraction"
                ]
                + trainer.last_update_timing[
                    "learner_padding_fraction"
                ],
                1.0,
            )
            self.assertEqual(
                trainer.last_update_timing["records"],
                float(len(records)),
            )
        finally:
            trainer.close()

    def test_batched_v41_learner_update_drift_is_bounded(
        self,
    ) -> None:
        config = PPOConfig(
            rollout_steps=8,
            rollout_workers=2,
            sequence_length=4,
            minibatch_sequences=2,
            update_epochs=1,
            hidden_size=64,
            card_embedding_dim=16,
            policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
            observation_version="v4.1",
            model_dim=32,
            transformer_layers=1,
            attention_heads=4,
            feedforward_dim=64,
            max_agent_steps_per_episode=8,
            training_class_ids=(6,),
            training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
        )
        trainers = [
            PPOTrainer(
                self.snapshot,
                master_seed=791,
                config=config,
            )
            for _ in range(2)
        ]
        try:
            trainers[0]._batched_v41_learner = False
            records = []
            bootstraps = []
            for trainer in trainers:
                rollout, bootstrap, _ = trainer.collect_rollout()
                records.append(rollout)
                bootstraps.append(bootstrap)
            self.assertEqual(
                [
                    (record.action, record.old_log_prob, record.value)
                    for record in records[0]
                ],
                [
                    (record.action, record.old_log_prob, record.value)
                    for record in records[1]
                ],
            )
            metrics = [
                trainer.update(rollout, bootstrap)
                for trainer, rollout, bootstrap in zip(
                    trainers, records, bootstraps
                )
            ]
            for key in metrics[0]:
                self.assertAlmostEqual(
                    metrics[0][key], metrics[1][key], places=5
                )
            maximum_parameter_error = max(
                float(
                    (
                        reference
                        - trainers[1].model.state_dict()[name]
                    ).abs().max()
                )
                for name, reference in (
                    trainers[0].model.state_dict().items()
                )
            )
            # The optimized and reference paths may converge more closely as
            # PyTorch kernels change; only the upper drift bound is a semantic
            # requirement.  Requiring a minimum error makes improved numeric
            # agreement fail the regression.
            self.assertLess(maximum_parameter_error, 1e-3)
        finally:
            for trainer in trainers:
                trainer.close()

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_collection_and_update_keep_randomness_on_device(self) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=781,
            config=PPOConfig(
                rollout_steps=16,
                sequence_length=4,
                minibatch_sequences=2,
                update_epochs=1,
                hidden_size=32,
                max_agent_steps_per_episode=8,
            ),
            device="cuda",
        )
        try:
            records, bootstrap, _ = trainer.collect_rollout()
            metrics = trainer.update(records, bootstrap)
            self.assertEqual(
                next(trainer.model.parameters()).device.type,
                "cuda",
            )
            self.assertTrue(
                all(math.isfinite(value) for value in metrics.values())
            )
        finally:
            trainer.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_multiprocess_rollout_uses_central_cuda_policy(self) -> None:
        trainer = PPOTrainer(
            self.snapshot,
            master_seed=783,
            config=PPOConfig(
                rollout_steps=8,
                rollout_workers=2,
                sequence_length=4,
                minibatch_sequences=2,
                update_epochs=1,
                hidden_size=32,
                max_agent_steps_per_episode=8,
            ),
            device="cuda",
        )
        try:
            before = {
                key: value.detach().clone()
                for key, value in trainer.model.state_dict().items()
            }
            records, _, _ = trainer.collect_rollout()
            self.assertTrue(all(
                record.action_mask[record.action]
                for record in records
            ))
            self.assertEqual(
                trainer.last_collect_timing["central_inference_requests"],
                float(len(records)),
            )
            self.assertGreater(
                trainer.last_collect_timing["central_forward_seconds"],
                0.0,
            )
            self.assertEqual(
                trainer.last_collect_timing["policy_transmitted_bytes"],
                0.0,
            )
            self.assertTrue(trainer.model.training)
            for key, value in trainer.model.state_dict().items():
                torch.testing.assert_close(value, before[key])
        finally:
            trainer.close()

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

    def test_learner_profile_preserves_update_results(self) -> None:
        summaries = []
        timings = []
        for profile_learner_timing in (False, True):
            trainer = PPOTrainer(
                self.snapshot,
                master_seed=786,
                config=PPOConfig(
                    rollout_steps=8,
                    rollout_workers=2,
                    sequence_length=4,
                    minibatch_sequences=2,
                    update_epochs=1,
                    hidden_size=16,
                    max_agent_steps_per_episode=8,
                    profile_learner_timing=profile_learner_timing,
                ),
            )
            try:
                records, bootstrap, boundaries = trainer.collect_rollout()
                metrics = trainer.update(records, bootstrap)
                summaries.append((
                    [
                        (
                            record.episode_id,
                            record.player_id,
                            record.action,
                            record.old_log_prob,
                            record.value,
                            record.reward,
                            record.hidden_before.copy(),
                        )
                        for record in records
                    ],
                    dict(bootstrap),
                    dict(boundaries),
                    metrics,
                    {
                        key: value.detach().cpu().numpy().copy()
                        for key, value in trainer.model.state_dict().items()
                    },
                ))
                timings.append(dict(trainer.last_update_timing))
            finally:
                trainer.close()
        self.assertEqual(
            [item[:-1] for item in summaries[0][0]],
            [item[:-1] for item in summaries[1][0]],
        )
        for first, second in zip(summaries[0][0], summaries[1][0]):
            np.testing.assert_array_equal(first[-1], second[-1])
        self.assertEqual(summaries[0][1:4], summaries[1][1:4])
        for key in summaries[0][4]:
            np.testing.assert_array_equal(
                summaries[0][4][key],
                summaries[1][4][key],
            )
        self.assertNotIn("learner_profiled_minibatches", timings[0])
        self.assertEqual(
            timings[1]["learner_profiled_minibatches"],
            timings[1]["minibatches"],
        )

    def test_seeded_central_policy_rollout_is_reproducible(self) -> None:
        summaries = []
        timings = []
        for profile_ipc_timing, profile_central_timing in (
            (False, False),
            (True, False),
            (False, True),
        ):
            trainer = PPOTrainer(
                self.snapshot,
                master_seed=784,
                config=PPOConfig(
                    rollout_steps=8,
                    rollout_workers=2,
                    sequence_length=4,
                    minibatch_sequences=2,
                    update_epochs=1,
                    hidden_size=64,
                    card_embedding_dim=16,
                    policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
                    model_dim=32,
                    transformer_layers=1,
                    attention_heads=4,
                    feedforward_dim=64,
                    max_agent_steps_per_episode=8,
                    observation_version="v4.1",
                    training_class_ids=(6,),
                    training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
                    profile_ipc_timing=profile_ipc_timing,
                    profile_central_timing=profile_central_timing,
                ),
            )
            try:
                records, bootstrap, boundaries = trainer.collect_rollout()
                summaries.append((
                    [
                        (
                            record.episode_id,
                            record.player_id,
                            record.action,
                            record.old_log_prob,
                            record.value,
                            record.reward,
                            record.observation.copy(),
                            record.card_indices.copy(),
                            record.action_mask.copy(),
                            record.hidden_before.copy(),
                        )
                        for record in records
                    ],
                    dict(bootstrap),
                    dict(boundaries),
                    trainer._policy_vector_rollout._generation,
                ))
                timings.append(dict(trainer.last_collect_timing))
            finally:
                trainer.close()
        for candidate in summaries[1:]:
            self.assertEqual(len(summaries[0][0]), len(candidate[0]))
            for first, second in zip(summaries[0][0], candidate[0]):
                self.assertEqual(first[:6], second[:6])
                for first_array, second_array in zip(
                    first[6:],
                    second[6:],
                ):
                    np.testing.assert_array_equal(
                        first_array,
                        second_array,
                    )
            self.assertEqual(summaries[0][1:], candidate[1:])
        self.assertEqual(
            timings[0]["worker_ipc_profiled_requests"],
            0.0,
        )
        self.assertEqual(
            timings[1]["worker_ipc_profiled_requests"],
            timings[1]["records"],
        )
        self.assertEqual(
            timings[2]["central_profiled_batches"],
            timings[2]["central_inference_batches"],
        )
        for timing in timings:
            self.assertEqual(
                timing["worker_decision_observation_construction_seconds"],
                0.0,
            )
            self.assertAlmostEqual(
                timing["worker_observation_construction_seconds"],
                timing["worker_step_observation_construction_seconds"]
                + timing[
                    "worker_bootstrap_observation_construction_seconds"
                ],
            )

    def test_central_profile_preserves_entity_action_rollout(self) -> None:
        summaries = []
        timings = []
        for profile_central_timing in (False, True):
            trainer = PPOTrainer(
                self.snapshot,
                master_seed=785,
                config=PPOConfig(
                    rollout_steps=8,
                    rollout_workers=2,
                    sequence_length=4,
                    minibatch_sequences=2,
                    update_epochs=1,
                    hidden_size=64,
                    card_embedding_dim=16,
                    policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
                    observation_version="v4.1",
                    model_dim=32,
                    transformer_layers=1,
                    attention_heads=4,
                    feedforward_dim=64,
                    max_agent_steps_per_episode=8,
                    profile_central_timing=profile_central_timing,
                    training_class_ids=(6,),
                    training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
                ),
            )
            try:
                records, bootstrap, boundaries = trainer.collect_rollout()
                summaries.append((
                    [
                        (
                            record.episode_id,
                            record.player_id,
                            record.action,
                            record.old_log_prob,
                            record.value,
                            record.reward,
                            record.hidden_before.copy(),
                        )
                        for record in records
                    ],
                    dict(bootstrap),
                    dict(boundaries),
                ))
                timings.append(dict(trainer.last_collect_timing))
            finally:
                trainer.close()
        self.assertEqual(
            [item[:-1] for item in summaries[0][0]],
            [item[:-1] for item in summaries[1][0]],
        )
        for first, second in zip(summaries[0][0], summaries[1][0]):
            np.testing.assert_array_equal(first[-1], second[-1])
        self.assertEqual(summaries[0][1:], summaries[1][1:])
        self.assertNotIn("central_profiled_batches", timings[0])
        self.assertEqual(
            timings[1]["central_profiled_batches"],
            timings[1]["central_inference_batches"],
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
            self.assertEqual(rollout.config.max_agent_steps, 8)
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
