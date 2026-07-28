from __future__ import annotations

import unittest

import numpy as np
import torch

from swb.db.repository import CardDefinition
from swb.engine.card_rules import RuleBook
from swb.engine.commands import ChoiceKind, ChoiceOption, ChoiceRequest
from swb.engine.emblem import EmblemDefinition
from swb.engine.environment import ShadowverseEnv
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType, GameEvent
from swb.engine.origin import CardOrigin
from swb.engine.state import (
    AttackRestriction,
    AttackRestrictionModifier,
    CostModifier,
    EmblemInstance,
    GraveyardCard,
    Unit,
)
from swb.engine.union_burst import UnionBurstDefinition, UnionBurstKind
from swb.rl.policy import (
    ENTITY_ACTION_POLICY_ARCHITECTURE,
    EntityActionRecurrentActorCritic,
)
from swb.rl.ppo import ObservationFlattener, PPOConfig, build_policy


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


class ObservationV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.deck_a = [card(100 + index) for index in range(40)]
        self.deck_b = [card(200 + index) for index in range(40)]
        self.vocabulary = tuple(range(100, 240))

    def make_env(self, **kwargs) -> ShadowverseEnv:
        observation_version = kwargs.pop("observation_version", "v4")
        env = ShadowverseEnv(
            self.deck_a,
            self.deck_b,
            class_a=1,
            class_b=1,
            seed=42,
            observation_version=observation_version,
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
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

    def test_fixed_numpy_schema_matches_space(self) -> None:
        env = self.make_env()
        observation = env.observation()
        self.assertTrue(env.observation_v4_space().contains(observation))
        self.assertTrue(
            all(isinstance(value, np.ndarray) for value in observation.values())
        )
        self.assertEqual(observation["action_mask"].shape, (112,))
        self.assertEqual(observation["own_hand_cards"].shape, (9,))
        self.assertEqual(observation["public_board_cards"].shape, (10,))

    def test_hidden_opponent_hand_and_deck_order_do_not_leak(self) -> None:
        first = self.make_env()
        second = self.make_env()
        opponent = second.players[1]
        opponent.hand[0].definition = card(139, cost=9)
        opponent.hand[0].add_keyword("疾驰")
        opponent.deck.reverse()
        self.assert_observations_equal(
            self.observe_after_direct_mutation(first),
            self.observe_after_direct_mutation(second),
        )

    def test_opponent_hidden_event_metadata_does_not_leak(self) -> None:
        first = self.make_env()
        second = self.make_env()
        first.core.event_history.append(GameEvent(
            EventType.CARD_DRAWN,
            player_index=1,
            metadata={"card_id": 201, "origin": "generated"},
        ))
        second.core.event_history.append(GameEvent(
            EventType.CARD_DRAWN,
            player_index=1,
            metadata={"card_id": 239, "origin": "deck"},
        ))
        self.assert_observations_equal(
            self.observe_after_direct_mutation(first),
            self.observe_after_direct_mutation(second),
        )

    def test_hand_keyword_collision_is_removed(self) -> None:
        legacy_first = self.make_env(observation_version="v3")
        legacy_second = self.make_env(observation_version="v3")
        legacy_second.players[0].hand[0].add_keyword("疾驰")
        self.assert_observations_equal(
            self.observe_after_direct_mutation(legacy_first),
            self.observe_after_direct_mutation(legacy_second),
        )

        first = self.make_env()
        second = self.make_env()
        second.players[0].hand[0].add_keyword("疾驰")
        first_observation = self.observe_after_direct_mutation(first)
        second_observation = self.observe_after_direct_mutation(second)
        self.assertFalse(np.array_equal(
            first_observation["own_hand_keyword_bits"],
            second_observation["own_hand_keyword_bits"],
        ))

    def test_union_burst_kind_threshold_progress_and_ready_state_are_explicit(
        self,
    ) -> None:
        definition = UnionBurstDefinition(
            card_id=100,
            kind=UnionBurstKind.UNION_BURST,
            operations=(
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    1,
                ),
            ),
        )
        env = self.make_env(rulebook=RuleBook(
            union_burst_defs={100: (definition,)},
        ))
        env.players[0].hand[0].definition = card(100)
        env.players[0].turns_started = 9
        env.players[0].hand[0].evolutions_while_in_hand = 1
        rows = self.observe_after_direct_mutation(env)[
            "own_hand_state"
        ].reshape(9, -1)
        first = rows[0]
        self.assertEqual(first[12], 10 / 15)
        self.assertEqual(first[16], 1.0)
        self.assertEqual(first[17], 0.0)
        self.assertEqual(first[18], 10 / 15)
        self.assertEqual(first[20], 1.0)
        self.assertEqual(first[21], 0.0)

    def test_cost_modifier_duration_collision_is_removed(self) -> None:
        legacy_first = self.make_env(observation_version="v3")
        legacy_second = self.make_env(observation_version="v3")
        legacy_first.players[0].hand[0].cost_modifiers.append(CostModifier(
            modifier_id=1,
            mode="subtract",
            amount=1,
            duration="permanent",
        ))
        legacy_second.players[0].hand[0].cost_modifiers.append(CostModifier(
            modifier_id=1,
            mode="subtract",
            amount=1,
            duration="until_end_of_turn",
            expires_for_player=0,
        ))
        self.assert_observations_equal(
            self.observe_after_direct_mutation(legacy_first),
            self.observe_after_direct_mutation(legacy_second),
        )

        first = self.make_env()
        second = self.make_env()
        first.players[0].hand[0].cost_modifiers.append(CostModifier(
            modifier_id=1,
            mode="subtract",
            amount=1,
            duration="permanent",
        ))
        second.players[0].hand[0].cost_modifiers.append(CostModifier(
            modifier_id=1,
            mode="subtract",
            amount=1,
            duration="until_end_of_turn",
            expires_for_player=0,
        ))
        self.assertEqual(
            first.players[0].hand[0].current_cost,
            second.players[0].hand[0].current_cost,
        )
        first_observation = self.observe_after_direct_mutation(first)
        second_observation = self.observe_after_direct_mutation(second)
        self.assertFalse(np.array_equal(
            first_observation["own_hand_modifier_state"],
            second_observation["own_hand_modifier_state"],
        ))

    def test_board_restriction_kind_and_duration_are_distinguishable(self) -> None:
        first = self.make_env()
        second = self.make_env()
        for env in (first, second):
            env.players[0].board.append(Unit.summon(
                card(138),
                entity_id=env.core.state.allocate_entity_id(),
            ))
        first.players[0].board[0].attack_restrictions.append(
            AttackRestrictionModifier(
                AttackRestriction.CANNOT_ATTACK_LEADER,
                "permanent",
            )
        )
        second.players[0].board[0].attack_restrictions.append(
            AttackRestrictionModifier(
                AttackRestriction.CANNOT_ATTACK_UNITS,
                "until_end_of_turn",
                0,
            )
        )
        first_observation = self.observe_after_direct_mutation(first)
        second_observation = self.observe_after_direct_mutation(second)
        self.assertFalse(np.array_equal(
            first_observation["public_board_modifier_state"],
            second_observation["public_board_modifier_state"],
        ))

    def test_source_origin_and_raw_entity_id_have_correct_semantics(self) -> None:
        first = self.make_env()
        second = self.make_env()
        first_unit = Unit.summon(card(138), entity_id=9001)
        second_unit = Unit.summon(card(138), entity_id=123456)
        first.players[0].board.append(first_unit)
        second.players[0].board.append(second_unit)
        self.assert_observations_equal(
            self.observe_after_direct_mutation(first),
            self.observe_after_direct_mutation(second),
        )
        second_unit.source_origin = CardOrigin.TOKEN
        second_observation = self.observe_after_direct_mutation(second)
        first_observation = self.observe_after_direct_mutation(first)
        self.assertFalse(np.array_equal(
            first_observation["public_board_origin_bits"],
            second_observation["public_board_origin_bits"],
        ))

    def test_exact_opponent_max_mana_and_current_own_deck_are_exposed(self) -> None:
        first = self.make_env()
        second = self.make_env()
        second.players[1].max_mana = 8
        second.players[1].mana = 3
        self.assertFalse(np.array_equal(
            self.observe_after_direct_mutation(first)["player_state"],
            self.observe_after_direct_mutation(second)["player_state"],
        ))

        third = self.make_env()
        fourth = self.make_env()
        fourth.players[0].deck[0] = card(239)
        self.assertFalse(np.array_equal(
            self.observe_after_direct_mutation(third)["own_current_deck"],
            self.observe_after_direct_mutation(fourth)["own_current_deck"],
        ))

    def test_graveyard_choice_slots_carry_card_identity_and_origin(self) -> None:
        env = self.make_env()
        first = GraveyardCard(
            definition=card(135),
            entity_id=env.core.state.allocate_entity_id(),
            owner=0,
            entered_sequence=1,
            entry_cause="destroy",
            origin=CardOrigin.DECK,
        )
        second = GraveyardCard(
            definition=card(139),
            entity_id=env.core.state.allocate_entity_id(),
            owner=0,
            entered_sequence=2,
            entry_cause="discard",
            origin=CardOrigin.GENERATED,
        )
        env.players[0].graveyard.extend((first, second))
        env.core.state.pending_choice = ChoiceRequest(
            player_index=0,
            choice_kind=ChoiceKind.GRAVEYARD,
            prompt="choose",
            options=(
                ChoiceOption("first", "first", entity_id=first.entity_id),
                ChoiceOption("second", "second", entity_id=second.entity_id),
            ),
            continuation_id="graveyard-test",
        )
        observation = self.observe_after_direct_mutation(env)
        self.assertEqual(
            tuple(observation["graveyard_page_cards"][:2]),
            (env._v2_card_index[135], env._v2_card_index[139]),
        )
        first_row = observation["graveyard_option_state"].reshape(16, -1)[0]
        second_row = observation["graveyard_option_state"].reshape(16, -1)[1]
        self.assertFalse(np.array_equal(first_row, second_row))

    def test_generic_choice_effect_identity_is_not_only_an_ordinal(self) -> None:
        first = self.make_env()
        second = self.make_env()
        for env, option_id, label in (
            (first, "draw-two", "draw two"),
            (second, "damage-three", "deal three"),
        ):
            env.core.state.pending_choice = ChoiceRequest(
                player_index=0,
                choice_kind=ChoiceKind.GENERIC,
                prompt="choose",
                options=(ChoiceOption(option_id, label),),
                continuation_id="generic-test",
            )
        first_state = self.observe_after_direct_mutation(first)[
            "choice_option_state"
        ]
        second_state = self.observe_after_direct_mutation(second)[
            "choice_option_state"
        ]
        self.assertFalse(np.array_equal(first_state, second_state))

    def test_public_history_includes_card_identity_but_not_raw_entity_id(self) -> None:
        first = self.make_env()
        second = self.make_env()
        first.core.event_history.append(GameEvent(
            EventType.CARD_PLAYED,
            player_index=1,
            source_id=9001,
            metadata={"card_id": 135},
        ))
        second.core.event_history.append(GameEvent(
            EventType.CARD_PLAYED,
            player_index=1,
            source_id=123456,
            metadata={"card_id": 139},
        ))
        first_observation = self.observe_after_direct_mutation(first)
        second_observation = self.observe_after_direct_mutation(second)
        self.assertNotEqual(
            first_observation["history_source_cards"][-1],
            second_observation["history_source_cards"][-1],
        )
        np.testing.assert_array_equal(
            first_observation["history_event_bits"],
            second_observation["history_event_bits"],
        )

    def test_emblem_and_listener_activation_runtime_is_exposed(self) -> None:
        definition = EmblemDefinition("test-emblem", 135)
        first = self.make_env(rulebook=RuleBook(
            emblem_defs={"test-emblem": definition}
        ))
        second = self.make_env(rulebook=RuleBook(
            emblem_defs={"test-emblem": definition}
        ))
        for env in (first, second):
            env.players[0].emblems.append(EmblemInstance(
                emblem_id="test-emblem",
                definition=definition,
                entity_id=env.core.state.allocate_entity_id(),
                controller=0,
                created_sequence=1,
            ))
        second.players[0].emblems[0].activation_counts[0] = 1
        second.players[0].emblems[0]._once_per_turn_used.add(0)
        self.assertFalse(np.array_equal(
            self.observe_after_direct_mutation(first)["leader_area_state"],
            self.observe_after_direct_mutation(second)["leader_area_state"],
        ))

        for env in (first, second):
            env.players[0].board.append(Unit.summon(
                card(138),
                entity_id=env.core.state.allocate_entity_id(),
            ))
        source = second.players[0].board[0]
        key = (source.entity_id, source.definition.card_id, 0)
        second.core.state.listener_activation_counts[key] = 1
        second.core.state.listener_once_per_turn_used.add(key)
        self.assertFalse(np.array_equal(
            self.observe_after_direct_mutation(first)["listener_state"],
            self.observe_after_direct_mutation(second)["listener_state"],
        ))

    def test_non_decision_perspective_has_no_private_choice_or_mask(self) -> None:
        env = self.make_env()
        other = 1 - env.decision_player
        observation = env.observation(perspective=other)
        self.assertFalse(observation["action_mask"].any())
        self.assertFalse(observation["choice_state"].any())
        self.assertFalse(observation["choice_option_state"].any())
        self.assertFalse(observation["graveyard_option_state"].any())

    def test_flattener_preserves_one_hot_semantics_and_embeds_auxiliary_cards(
        self,
    ) -> None:
        env = self.make_env()
        env.players[0].hand[0].origin = CardOrigin.GENERATED
        observation = self.observe_after_direct_mutation(env)
        flattener = ObservationFlattener.from_observation(observation)
        numeric = flattener.encode(observation)
        offset, size = flattener.field_layout["own_hand_origin_bits"]
        self.assertEqual(float(numeric[offset : offset + size].max()), 1.0)
        self.assertIn("graveyard_page_cards", flattener.card_field_layout)
        self.assertIn("history_source_cards", flattener.card_field_layout)
        self.assertGreater(flattener.card_slots, 19)

    def test_entity_action_policy_accepts_v4_and_choice_scores_follow_candidate(
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
                ChoiceOption("first", "first", entity_id=first_target.entity_id),
                ChoiceOption("second", "second", entity_id=second_target.entity_id),
            ),
            continuation_id="policy-v4-test",
        )
        observation = self.observe_after_direct_mutation(env)
        flattener = ObservationFlattener.from_observation(observation)
        model = build_policy(
            PPOConfig(
                policy_architecture=ENTITY_ACTION_POLICY_ARCHITECTURE,
                observation_version="v4",
                hidden_size=32,
                card_embedding_dim=8,
                model_dim=32,
                transformer_layers=1,
                attention_heads=4,
                feedforward_dim=64,
            ),
            flattener,
            action_size=112,
            card_vocabulary_size=len(self.vocabulary),
        )
        self.assertIsInstance(model, EntityActionRecurrentActorCritic)
        self.assertLess(model.global_input_size, flattener.size)
        model.eval()

        def forward(candidate):
            numeric = torch.from_numpy(
                flattener.encode(candidate)
            ).unsqueeze(0)
            cards = torch.from_numpy(
                flattener.encode_cards(candidate)
            ).unsqueeze(0)
            with torch.no_grad():
                return model.forward_step(
                    numeric,
                    model.initial_state(1, device=torch.device("cpu")),
                    cards,
                )

        first_logits, first_value, _ = forward(observation)
        swapped = {name: value.copy() for name, value in observation.items()}
        option_rows = swapped["choice_option_state"].reshape(16, -1)
        option_rows[[0, 1]] = option_rows[[1, 0]]
        swapped["choice_option_cards"][[0, 1]] = (
            swapped["choice_option_cards"][[1, 0]]
        )
        second_logits, second_value, _ = forward(swapped)
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


if __name__ == "__main__":
    unittest.main()
