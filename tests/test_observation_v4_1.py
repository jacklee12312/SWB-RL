from __future__ import annotations

import unittest

import numpy as np
import torch

from swb.db.repository import CardDefinition
from swb.engine.card_rules import RuleBook
from swb.engine.commands import ChoiceKind, ChoiceOption, ChoiceRequest
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType, GameEvent
from swb.engine import observation_v4_1
from swb.engine import observation_v4
from swb.engine.state import Amulet, CostModifier, StatModifier, Unit
from swb.rl.policy import (
    ENTITY_ACTION_POLICY_ARCHITECTURE,
    EntityActionRecurrentActorCritic,
)
from swb.rl.ppo import ObservationFlattener, PPOConfig, build_policy
from swb.rl.versioning import (
    OBSERVATION_SCHEMA_VERSIONS,
    observation_schema_manifest,
)


def card(card_id: int, *, cost: int = 3) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=cost,
        card_type="随从",
        attack=2,
        life=2,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


class ObservationV41Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.deck_a = [card(100 + index) for index in range(40)]
        self.deck_b = [card(200 + index) for index in range(40)]
        self.vocabulary = tuple(range(100, 240))

    def make_env(self, **kwargs) -> ShadowverseEnv:
        env = ShadowverseEnv(
            self.deck_a,
            self.deck_b,
            class_a=1,
            class_b=1,
            seed=42,
            observation_version="v4.1",
            card_vocabulary=self.vocabulary,
            rulebook=kwargs.pop("rulebook", RuleBook()),
            **kwargs,
        )
        env.reset(seed=42)
        return env

    @staticmethod
    def observe_after_direct_mutation(env: ShadowverseEnv):
        env._observation_cache.clear()
        return env.observation()

    def assert_observations_equal(self, first, second) -> None:
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            np.testing.assert_array_equal(
                first[name], second[name], err_msg=name
            )

    def make_policy(self, observation):
        flattener = ObservationFlattener.from_observation(observation)
        config = PPOConfig(
            policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
            observation_version="v4.1",
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
            action_size=ShadowverseEnv.ACTION_SIZE,
            card_vocabulary_size=len(self.vocabulary),
        )
        return flattener, model

    @staticmethod
    def forward(model, flattener, observation):
        numeric = torch.from_numpy(
            flattener.encode(observation)
        ).unsqueeze(0)
        cards = torch.from_numpy(
            flattener.encode_cards(observation)
        ).unsqueeze(0)
        with torch.no_grad():
            return model.forward_step(
                numeric,
                model.initial_state(1, device=torch.device("cpu")),
                cards,
            )

    def test_schema_is_fixed_compact_and_space_checked(self) -> None:
        env = self.make_env()
        observation = env.observation()
        self.assertTrue(env.observation_v4_1_space().contains(observation))
        self.assertEqual(len(observation), 61)
        self.assertEqual(observation["action_mask"].shape, (112,))
        flattener = ObservationFlattener.from_observation(observation)
        self.assertEqual(flattener.size, 15_757)
        self.assertEqual(flattener.card_slots, 1_290)
        self.assertIn("zone_cards", flattener.card_field_layout)
        self.assertIn("record_cards", flattener.card_field_layout)

    def test_audit_only_extra_pp_events_do_not_change_frozen_vocab(self) -> None:
        self.assertEqual(observation_v4_1.EVENT_TYPE_COUNT, 102)
        self.assertNotIn(
            EventType.EXTRA_PP_ACTIVATED,
            observation_v4.EVENT_INDEX,
        )
        self.assertNotIn(
            EventType.EXTRA_PP_REFUNDED,
            observation_v4.EVENT_INDEX,
        )
        env = self.make_env()
        before = self.observe_after_direct_mutation(env)
        before_count = int(np.count_nonzero(
            before["history_event_types"]
        ))
        env._core.event_history.extend((
            GameEvent(EventType.EXTRA_PP_ACTIVATED, 0),
            GameEvent(EventType.EXTRA_PP_REFUNDED, 0),
            GameEvent(EventType.TURN_ENDED, 0),
        ))

        observation = self.observe_after_direct_mutation(env)

        non_padding = observation["history_event_types"]
        non_padding = non_padding[non_padding != 0]
        self.assertEqual(non_padding.size, before_count + 1)
        self.assertEqual(
            int(non_padding[-1]),
            observation_v4.EVENT_INDEX[EventType.TURN_ENDED] + 1,
        )
        flattener, model = self.make_policy(observation)
        self.assertEqual(
            model.v41_history_event_embedding.num_embeddings,
            103,
        )
        self.assertEqual(flattener.size, 15_757)

    def test_hidden_hand_deck_order_and_hidden_draw_metadata_do_not_leak(
        self,
    ) -> None:
        first = self.make_env()
        second = self.make_env()
        second.players[1].hand[0].definition = card(239, cost=9)
        second.players[1].hand[0].add_keyword("疾驰")
        second.players[1].deck.reverse()
        first.core.event_history.append(GameEvent(
            EventType.CARD_DRAWN,
            player_index=1,
            metadata={"card_id": 201, "origin": "deck"},
        ))
        second.core.event_history.append(GameEvent(
            EventType.CARD_DRAWN,
            player_index=1,
            metadata={"card_id": 239, "origin": "generated"},
        ))
        self.assert_observations_equal(
            self.observe_after_direct_mutation(first),
            self.observe_after_direct_mutation(second),
        )

    def test_own_deck_is_identity_preserving_but_order_independent(self) -> None:
        first = self.make_env()
        second = self.make_env()
        second.players[0].deck.reverse()
        self.assert_observations_equal(
            self.observe_after_direct_mutation(first),
            self.observe_after_direct_mutation(second),
        )

        third = self.make_env()
        fourth = self.make_env()
        fourth.players[0].deck[0] = card(239, cost=8)
        third_observation = self.observe_after_direct_mutation(third)
        fourth_observation = self.observe_after_direct_mutation(fourth)
        self.assertFalse(np.array_equal(
            third_observation["own_deck_cards"],
            fourth_observation["own_deck_cards"],
        ))
        self.assertFalse(np.array_equal(
            third_observation["zone_cards"],
            fourth_observation["zone_cards"],
        ))

    def test_modifier_duration_has_a_typed_row_and_changes_policy_input(
        self,
    ) -> None:
        permanent = self.make_env()
        temporary = self.make_env()
        permanent.players[0].hand[0].cost_modifiers.append(CostModifier(
            modifier_id=1,
            mode="subtract",
            amount=1,
            duration="permanent",
        ))
        temporary.players[0].hand[0].cost_modifiers.append(CostModifier(
            modifier_id=1,
            mode="subtract",
            amount=1,
            duration="until_end_of_turn",
            expires_for_player=0,
        ))
        self.assertEqual(
            permanent.players[0].hand[0].current_cost,
            temporary.players[0].hand[0].current_cost,
        )
        first = self.observe_after_direct_mutation(permanent)
        second = self.observe_after_direct_mutation(temporary)
        self.assertEqual(first["hand_modifier_kind"][0], 1)
        self.assertEqual(second["hand_modifier_kind"][0], 1)
        self.assertNotEqual(
            first["hand_modifier_duration"][0],
            second["hand_modifier_duration"][0],
        )
        flattener, model = self.make_policy(first)
        model.eval()
        first_logits, _, _ = self.forward(model, flattener, first)
        second_logits, _, _ = self.forward(model, flattener, second)
        self.assertFalse(torch.equal(first_logits, second_logits))

    def test_modifier_overflow_does_not_crowd_out_other_kinds(self) -> None:
        env = self.make_env()
        hand_card = env.players[0].hand[0]
        for index in range(
            observation_v4_1.v4.MAX_COST_MODIFIERS + 1
        ):
            hand_card.cost_modifiers.append(CostModifier(
                modifier_id=index + 1,
                mode="subtract",
                amount=1,
                duration="permanent",
            ))
        hand_card.stat_modifiers.append(StatModifier(
            100, 1, 1, "permanent"
        ))
        observation = self.observe_after_direct_mutation(env)
        kinds = observation["hand_modifier_kind"].reshape(
            env.MAX_HAND, observation_v4_1.MAX_HAND_MODIFIERS
        )[0]
        self.assertIn(
            observation_v4_1.MODIFIER_KINDS["stat"], kinds
        )
        summary = observation["hand_modifier_summary"].reshape(
            env.MAX_HAND, observation_v4_1.MODIFIER_SUMMARY_SIZE
        )[0]
        self.assertEqual(summary[1], 1.0)
        self.assertGreater(summary[2], 0.0)

    def test_raw_entity_ids_are_excluded_but_public_history_identity_remains(
        self,
    ) -> None:
        first = self.make_env()
        second = self.make_env()
        first.players[0].board.append(Unit.summon(
            card(138), entity_id=9001
        ))
        second.players[0].board.append(Unit.summon(
            card(138), entity_id=123456
        ))
        self.assert_observations_equal(
            self.observe_after_direct_mutation(first),
            self.observe_after_direct_mutation(second),
        )

        first.core.event_history.append(GameEvent(
            EventType.CARD_PLAYED,
            player_index=0,
            metadata={"card_id": 135, "play_mode": "normal"},
        ))
        second.core.event_history.append(GameEvent(
            EventType.CARD_PLAYED,
            player_index=0,
            metadata={"card_id": 139, "play_mode": "normal"},
        ))
        first_observation = self.observe_after_direct_mutation(first)
        second_observation = self.observe_after_direct_mutation(second)
        self.assertFalse(np.array_equal(
            first_observation["history_source_cards"],
            second_observation["history_source_cards"],
        ))

    def test_public_amulet_uses_its_own_keyword_shape(self) -> None:
        env = self.make_env()
        definition = CardDefinition(
            card_id=138,
            card_set_id=10000,
            class_id=1,
            class_name="精灵",
            name="amulet-138",
            cost=2,
            card_type="护符",
            attack=None,
            life=None,
            keywords=frozenset({"土之印"}),
            support_level="basic",
            is_collectible=True,
        )
        env.players[0].board.append(Amulet(
            definition=definition,
            entity_id=env.core.state.allocate_entity_id(),
            countdown=3,
        ))
        observation = self.observe_after_direct_mutation(env)
        self.assertTrue(env.observation_v4_1_space().contains(observation))
        board = observation["public_board_base"].reshape(
            2 * env.MAX_BOARD, -1
        )
        self.assertEqual(board[0, 3], 1.0)
        self.assertEqual(board[0, 5], 1.0)

    def test_policy_uses_93_tokens_and_choice_scores_follow_candidate(
        self,
    ) -> None:
        env = self.make_env()
        first_target = Unit.summon(
            card(135),
            entity_id=env.core.state.allocate_entity_id(),
        )
        second_target = Unit.summon(
            card(139),
            entity_id=env.core.state.allocate_entity_id(),
        )
        env.players[1].board.extend((first_target, second_target))
        env.core.state.pending_choice = ChoiceRequest(
            player_index=0,
            choice_kind=ChoiceKind.BOARD,
            prompt="target",
            options=(
                ChoiceOption(
                    "first", "first", entity_id=first_target.entity_id
                ),
                ChoiceOption(
                    "second", "second", entity_id=second_target.entity_id
                ),
            ),
            continuation_id="policy-v4.1-test",
        )
        observation = self.observe_after_direct_mutation(env)
        flattener, model = self.make_policy(observation)
        self.assertIsInstance(model, EntityActionRecurrentActorCritic)
        self.assertEqual(model.structured_token_count, 93)
        self.assertLess(model.global_input_size, 32)
        model.eval()
        first_logits, first_value, first_hidden = self.forward(
            model, flattener, observation
        )
        self.assertEqual(first_logits.shape, (1, 112))
        self.assertEqual(first_value.shape, (1,))
        self.assertEqual(first_hidden.shape, (1, 32))

        swapped = {
            name: value.copy() for name, value in observation.items()
        }
        for name in (
            "choice_option_cards",
            "choice_option_references",
            "choice_option_relations",
            "choice_option_selected",
        ):
            swapped[name][[0, 1]] = swapped[name][[1, 0]]
        semantic_rows = swapped["choice_option_semantics"].reshape(16, -1)
        semantic_rows[[0, 1]] = semantic_rows[[1, 0]]
        second_logits, second_value, _ = self.forward(
            model, flattener, swapped
        )
        choice = env.CHOICE_OFFSET
        torch.testing.assert_close(
            first_logits[0, choice],
            second_logits[0, choice + 1],
        )
        torch.testing.assert_close(
            first_logits[0, choice + 1],
            second_logits[0, choice],
        )
        torch.testing.assert_close(first_value, second_value)

        model.train()
        numeric = torch.from_numpy(
            flattener.encode(observation)
        ).unsqueeze(0)
        cards = torch.from_numpy(
            flattener.encode_cards(observation)
        ).unsqueeze(0)
        logits, value, _ = model.forward_step(
            numeric,
            model.initial_state(1, device=torch.device("cpu")),
            cards,
        )
        (logits.sum() + value.sum()).backward()
        present_card = int(cards[cards != 0][0])
        gradient = model.card_embedding.weight.grad
        self.assertGreater(
            float(gradient[present_card].abs().sum()), 0.0
        )
        self.assertEqual(float(gradient[0].abs().sum()), 0.0)

    def test_history_source_and_target_roles_do_not_collapse(self) -> None:
        env = self.make_env()
        observation = env.observation()
        observation["history_event_types"][-1] = 1
        observation["history_actors"][-1] = 1
        observation["history_source_references"][-1] = 1
        observation["history_target_references"][-1] = 6
        observation["history_source_cards"][-1] = (
            env._v2_card_index[135]
        )
        observation["history_target_cards"][-1] = (
            env._v2_card_index[139]
        )
        swapped = {
            name: value.copy() for name, value in observation.items()
        }
        swapped["history_source_references"][-1] = 6
        swapped["history_target_references"][-1] = 1
        swapped["history_source_cards"][-1] = env._v2_card_index[139]
        swapped["history_target_cards"][-1] = env._v2_card_index[135]
        flattener, model = self.make_policy(observation)
        model.eval()
        first_logits, first_value, _ = self.forward(
            model, flattener, observation
        )
        second_logits, second_value, _ = self.forward(
            model, flattener, swapped
        )
        self.assertFalse(torch.equal(first_logits, second_logits))
        self.assertFalse(torch.equal(first_value, second_value))

    def test_semantic_byte_positions_and_zone_count_pairs_are_preserved(
        self,
    ) -> None:
        env = self.make_env()
        observation = env.observation()
        observation["history_event_types"][-1] = 1
        semantic_rows = observation["history_semantics"].reshape(32, 5)
        semantic_rows[-1] = (1, 1, 2, 3, 4)
        swapped_semantic = {
            name: value.copy() for name, value in observation.items()
        }
        swapped_semantic["history_semantics"].reshape(32, 5)[-1] = (
            1, 4, 3, 2, 1
        )
        flattener, model = self.make_policy(observation)
        model.eval()
        first_logits, _, _ = self.forward(
            model, flattener, observation
        )
        second_logits, _, _ = self.forward(
            model, flattener, swapped_semantic
        )
        self.assertFalse(torch.equal(first_logits, second_logits))

        paired = {
            name: value.copy() for name, value in observation.items()
        }
        paired["zone_counts"][:2] = (1 / 40, 2 / 40)
        swapped_pairs = {
            name: value.copy() for name, value in paired.items()
        }
        swapped_pairs["zone_counts"][:2] = (2 / 40, 1 / 40)
        paired_logits, _, _ = self.forward(model, flattener, paired)
        swapped_pair_logits, _, _ = self.forward(
            model, flattener, swapped_pairs
        )
        self.assertFalse(torch.equal(
            paired_logits, swapped_pair_logits
        ))

    def test_version_manifest_and_policy_guard_are_explicit(self) -> None:
        manifest = observation_schema_manifest(self.make_env())
        self.assertEqual(
            manifest["version"],
            OBSERVATION_SCHEMA_VERSIONS["v4.1"],
        )
        self.assertEqual(
            manifest["encoding"]["categorical_values"],
            "typed-indices",
        )
        self.assertEqual(manifest["fixed_limits"]["transformer_tokens"], 93)
        with self.assertRaisesRegex(
            ValueError, "requires.*entity_action_v1"
        ):
            PPOConfig(observation_version="v4.1")


if __name__ == "__main__":
    unittest.main()
