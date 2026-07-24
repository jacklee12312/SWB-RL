from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import RuleBook
from swb.engine.commands import ChoiceKind, ChoiceOption, ChoiceRequest
from swb.engine.emblem import EmblemDefinition
from swb.engine.environment import ShadowverseEnv
from swb.engine.effects import TurnEndDestroyTiming
from swb.engine.events import EventType, GameEvent
from swb.engine.faith import FaithDefinition, FaithInstance
from swb.engine.observation_v2 import _board_runtime
from swb.engine.origin import CardOrigin
from swb.engine.state import (
    EmblemInstance,
    FusionMaterial,
    LeaderDamageModifier,
    Unit,
)


def card(card_id: int, *, cost: int = 1, card_type: str = "随从") -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=1 if card_type == "随从" else None,
        life=2 if card_type == "随从" else None,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


class ObservationV2Tests(unittest.TestCase):
    def setUp(self):
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
            observation_version="v2",
            card_vocabulary=self.vocabulary,
            **kwargs,
        )
        env.reset(seed=42)
        return env

    def test_v1_remains_default_and_fixed(self):
        env = ShadowverseEnv(
            self.deck_a,
            self.deck_b,
            class_a=1,
            class_b=1,
            seed=42,
        )
        observation, _ = env.reset(seed=42)
        self.assertIsInstance(observation, list)
        self.assertEqual(len(observation), ShadowverseEnv.OBSERVATION_V1_SIZE)
        self.assertEqual(ShadowverseEnv.ACTION_SIZE, 111)
        with self.assertRaisesRegex(ValueError, "observation_version='v2'"):
            env.observation_v2_spec()
        with self.assertRaisesRegex(ValueError, "observation_version='v2'"):
            env.recurrent_observation()

    def test_v2_shapes_and_action_mask_are_fixed_for_vocabulary(self):
        env = self.make_env()
        observation = env.observation()
        spec = env.observation_v2_spec()

        self.assertEqual(observation["version"], 2)
        self.assertEqual(
            len(observation["continuous_v1"]),
            ShadowverseEnv.OBSERVATION_V1_SIZE,
        )
        self.assertEqual(len(observation["card_indices"]["own_hand"]), 9)
        self.assertEqual(len(observation["card_indices"]["public_board"]), 10)
        self.assertEqual(len(observation["card_indices"]["initial_decks"][0]), 140)
        self.assertEqual(len(observation["own_hand_runtime"]), 108)
        self.assertEqual(len(observation["public_board_runtime"]), 180)
        self.assertEqual(len(observation["choice"]["option_references"]), 16)
        self.assertEqual(len(observation["public_history"]["event_types"]), 16)
        self.assertEqual(
            len(observation["leader_area"]["leader_damage_modifier_runtime"]),
            80,
        )
        self.assertEqual(observation["action_mask"], tuple(env.action_mask()))
        self.assertEqual(spec["action_size"], 111)
        self.assertEqual(spec["card_vocabulary_size"], 140)
        self.assertEqual(spec["categorical_vocabulary"]["cards"], self.vocabulary)

    def test_turn_end_removal_runtime_slots_encode_destroy_and_banish(self):
        unit = Unit.summon(card(138), entity_id=1)
        unit.turn_end_destroy_timings.add(TurnEndDestroyTiming.OWNER_TURN)
        unit.turn_end_banish_timings.add(TurnEndDestroyTiming.OPPONENT_TURN)
        runtime = _board_runtime(unit)
        self.assertEqual(len(runtime), 18)
        self.assertEqual(runtime[14:16], (1.0, 2.0))

        unit.turn_end_banish_timings.add(TurnEndDestroyTiming.OWNER_TURN)
        self.assertEqual(_board_runtime(unit)[14], 3.0)

    def test_own_hand_and_public_board_identity_are_categorical(self):
        env = self.make_env()
        before = env.observation()
        own_hand = env.players[0].hand[0]
        own_hand.definition = card(139)
        after_hand = env.observation()
        self.assertNotEqual(
            before["card_indices"]["own_hand"],
            after_hand["card_indices"]["own_hand"],
        )

        unit = Unit.summon(card(138), entity_id=env.core.state.allocate_entity_id())
        env.players[0].board.append(unit)
        after_board = env.observation()
        self.assertEqual(
            after_board["card_indices"]["public_board"][0],
            env._v2_card_index[138],
        )
        self.assertNotIn(138, after_board["card_indices"]["public_board"])

    def test_opponent_hidden_hand_identity_and_deck_order_do_not_leak(self):
        env = self.make_env()
        before = env.observation()
        opponent = env.players[1]
        hidden = opponent.hand[0]
        hidden.definition = card(239, cost=9, card_type="法术")
        material_id = env.core.state.allocate_entity_id()
        opponent.fusion_materials.append(FusionMaterial(
            definition=card(238),
            entity_id=material_id,
            owner=1,
            consumed_sequence=1,
            fused_into_entity_id=hidden.entity_id,
            origin=CardOrigin.DECK,
        ))
        hidden.fused_material_ids.append(material_id)
        hidden.fusion_used_turn = env.turn
        opponent.deck.reverse()

        self.assertEqual(env.observation(), before)

    def test_public_state_ignores_entity_ids_but_not_origin_or_modifiers(self):
        first = self.make_env()
        second = self.make_env()
        first_unit = Unit.summon(card(137), entity_id=9001)
        second_unit = Unit.summon(card(137), entity_id=123456)
        first.players[0].board.append(first_unit)
        second.players[0].board.append(second_unit)
        self.assertEqual(first.observation(), second.observation())

        second_unit.origin = CardOrigin.TRANSFORMED
        second_unit.printed_abilities_removed = True
        self.assertNotEqual(first.observation(), second.observation())

        before_last_words = first.observation()
        first_unit.last_words_removed = True
        self.assertNotEqual(before_last_words, first.observation())

        before_scope = first.observation()
        first_unit.turn_end_destroy_timings.add(
            TurnEndDestroyTiming.OWNER_TURN
        )
        self.assertNotEqual(before_scope, first.observation())

    def test_deck_composition_is_order_independent_and_distinguishable(self):
        first = self.make_env()
        reversed_env = ShadowverseEnv(
            list(reversed(self.deck_a)),
            list(reversed(self.deck_b)),
            class_a=1,
            class_b=1,
            seed=42,
            observation_version="v2",
            card_vocabulary=self.vocabulary,
        )
        reversed_env.reset(seed=42)
        self.assertEqual(
            first.observation()["card_indices"]["initial_decks"],
            reversed_env.observation()["card_indices"]["initial_decks"],
        )

        changed_deck = [card(139), *self.deck_a[1:]]
        changed = ShadowverseEnv(
            changed_deck,
            self.deck_b,
            class_a=1,
            class_b=1,
            seed=42,
            observation_version="v2",
            card_vocabulary=self.vocabulary,
        )
        changed.reset(seed=42)
        self.assertNotEqual(
            first.observation()["card_indices"]["initial_decks"],
            changed.observation()["card_indices"]["initial_decks"],
        )

    def test_leader_area_identity_and_values_are_public(self):
        faith_def = FaithDefinition("faith-test", 100)
        emblem_def = EmblemDefinition("emblem-test", 101, countdown=3)
        rulebook = RuleBook(
            faith_defs={100: faith_def},
            emblem_defs={"emblem-test": emblem_def},
        )
        env = self.make_env(rulebook=rulebook)
        env.players[0].faiths = [FaithInstance(
            definition=faith_def,
            entity_id=env.core.state.allocate_entity_id(),
            controller=0,
            created_sequence=1,
            value=7,
        )]
        env.players[1].emblems = [EmblemInstance(
            emblem_id="emblem-test",
            definition=emblem_def,
            entity_id=env.core.state.allocate_entity_id(),
            controller=1,
            created_sequence=1,
            countdown=3,
        )]

        leader = env.observation()["leader_area"]
        self.assertEqual(leader["faith_ids"][0], 1)
        self.assertEqual(leader["faith_values"][0], 7 / 50)
        self.assertEqual(leader["emblem_ids"][5], 1)
        self.assertEqual(leader["emblem_countdowns"][5], 3 / 20)

    def test_leader_modifier_duration_and_source_lifetime_are_distinguishable(self):
        env = self.make_env()
        source = Unit.summon(card(135), entity_id=98765)
        env.players[0].board.append(source)
        env.players[1].leader_damage_modifiers.append(LeaderDamageModifier(
            modifier_id=1,
            amount=1,
            duration="while_source_in_play",
            source_controller=0,
            source_entity_id=source.entity_id,
            source_card_id=source.definition.card_id,
        ))
        while_present = env.observation()["leader_area"][
            "leader_damage_modifier_runtime"
        ]
        env.players[0].board.clear()
        source_absent = env.observation()["leader_area"][
            "leader_damage_modifier_runtime"
        ]
        self.assertNotEqual(while_present, source_absent)

    def test_parameterized_choice_uses_public_references_not_entity_ids(self):
        env = self.make_env()
        target = Unit.summon(card(136), entity_id=987654)
        env.players[1].board.append(target)
        env.core.state.pending_choice = ChoiceRequest(
            player_index=0,
            choice_kind=ChoiceKind.BOARD,
            prompt="target",
            options=(ChoiceOption("entity:987654", "target", entity_id=987654),),
            continuation_id="test-choice",
            target_count=1,
        )

        choice = env.observation()["choice"]
        self.assertEqual(choice["kind"], 1)
        self.assertEqual(choice["option_references"][0], 6)
        self.assertNotIn(987654, choice["option_references"])

    def test_public_history_ignores_hidden_event_metadata(self):
        first = self.make_env()
        second = self.make_env()
        first.core.event_history.append(GameEvent(
            EventType.CARD_DRAWN,
            player_index=1,
            metadata={"card_id": 201, "card_name": "hidden-a"},
        ))
        second.core.event_history.append(GameEvent(
            EventType.CARD_DRAWN,
            player_index=1,
            metadata={"card_id": 239, "card_name": "hidden-b"},
        ))
        self.assertEqual(first.observation(), second.observation())
        recurrent = first.recurrent_observation()
        self.assertIn("public_history", recurrent)
        self.assertNotIn(201, recurrent["public_history"]["event_types"])

        first.core.event_history.append(GameEvent(
            EventType.DEATH_BATCH_START,
            player_index=-1,
        ))
        self.assertEqual(
            first.observation()["public_history"]["actor_relations"][-1],
            0,
        )

    def test_invalid_vocabulary_and_version_are_rejected(self):
        kwargs = dict(
            deck_a=self.deck_a,
            deck_b=self.deck_b,
            class_a=1,
            class_b=1,
            seed=42,
        )
        with self.assertRaisesRegex(ValueError, "observation_version"):
            ShadowverseEnv(**kwargs, observation_version="v4")
        with self.assertRaisesRegex(ValueError, "card_vocabulary"):
            ShadowverseEnv(
                **kwargs,
                observation_version="v2",
                card_vocabulary=(100, 100),
            )


if __name__ == "__main__":
    unittest.main()
