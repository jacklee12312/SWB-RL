from __future__ import annotations

import copy
import unittest

from swb.db.repository import CardDefinition
from swb.engine.abilities import (
    ABILITY_DEFINITIONS,
    AbilityEvent,
    AbilityKeyword,
    AbilityStatus,
    normalize_abilities,
)
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import ChoiceKind, ChoiceOption, ChoiceRequest
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.emblem import EmblemDefinition, EmblemTriggerRule
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.model import Amulet, HandCard, Unit
from swb.engine.state import Phase
from swb.rules import EffectDefinition


def card(
    card_id: int,
    *,
    cost: int = 1,
    attack: int = 1,
    life: int = 1,
    card_type: str = "随从",
    keywords: frozenset[str] = frozenset(),
    fanfare_effects: tuple[EffectDefinition, ...] = (),
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack,
        life=life,
        keywords=keywords,
        support_level="basic",
        is_collectible=True,
        fanfare_effects=fanfare_effects,
    )


PUBLIC_INFO_KEYS = {
    "current_player",
    "decision_player",
    "turn",
    "winner",
    "player_classes",
    "action_mask",
    "super_evolution_points",
    "placeholder_ability_count",
    "graveyard_page",
    "graveyard_total_pages",
}

DEBUG_INFO_KEYS = {
    "log",
    "events",
    "placeholder_ability_events",
}


def env_snapshot(env: ShadowverseEnv):
    return (
        copy.deepcopy(env.core.state),
        tuple(env.logs),
        tuple(env.core.event_history),
        tuple(env.placeholder_ability_events),
        env.core.random.getstate(),
        dict(env.core._death_causes),
        copy.deepcopy(env.core._suspended_batch),
        copy.deepcopy(env.core._suspended_record),
        copy.deepcopy(env.core._suspended_lw_records),
        env.core._suspended_action,
        copy.deepcopy(env.core._suspended_action_state),
        copy.deepcopy(env.core._suspended_event_state),
        env.core._spellboost_pending,
        env.core._pending_spellboost_player,
        env.core._pending_spellboost_source_card_id,
        env.core._pending_spellboost_source_entity_id,
        copy.deepcopy(env.core._emblem_batches),
        env.core._next_emblem_batch_id,
        copy.deepcopy(env.core._emblem_expiration_batches),
        env.core._next_emblem_expiration_batch_id,
        env.core._stabilizing,
        env.core._next_modifier_id,
        env.core._next_choice_request_id,
        env._graveyard_page,
        copy.deepcopy(env._last_choice_request_key),
    )


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        deck = [card(index) for index in range(40)]
        self.env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=1,
        )
        self.env.reset(seed=1)

    def test_observation_and_action_space_are_fixed(self) -> None:
        self.assertEqual(len(self.env.action_mask()), ShadowverseEnv.ACTION_SIZE)
        self.assertEqual(len(self.env.observation()), 255)
        self.assertTrue(self.env.action_mask()[ShadowverseEnv.END_TURN])
        self.assertEqual(sum(self.env.observation()[30:37]), 1.0)
        self.assertEqual(sum(self.env.observation()[37:44]), 1.0)
        self.assertEqual(self.env.observation()[44:48], [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(self.env.observation()[48:50], [0.0, 0.0])
        self.assertEqual(self.env.observation()[-2:], [0.0, 0.0])

    def test_observation_exposes_public_combo_counts(self) -> None:
        self.env.players[0].cards_played_this_turn = 3
        self.env.players[1].cards_played_this_turn = 1
        self.assertEqual(self.env.observation()[20:22], [0.3, 0.1])

    def test_observation_exposes_public_overflow_flags(self) -> None:
        self.env.players[0].max_mana = 7
        self.env.players[1].max_mana = 6
        self.assertEqual(self.env.observation()[48:50], [1.0, 0.0])

        self.env.players[1].max_mana = 7
        self.assertEqual(self.env.observation()[48:50], [1.0, 1.0])

    def test_opening_hands_are_four_plus_first_player_draw(self) -> None:
        self.assertEqual(len(self.env.players[0].hand), 5)
        self.assertEqual(len(self.env.players[1].hand), 4)
        self.assertEqual(len(self.env.players[0].deck), 35)
        self.assertEqual(len(self.env.players[1].deck), 36)
        self.assertEqual(self.env.players[0].class_name, "精灵")

    def test_play_and_attack_leader(self) -> None:
        self.env.players[0].mana = 10
        play_action = ShadowverseEnv.PLAY_OFFSET
        self.env.step(play_action)
        self.assertEqual(len(self.env.players[0].board), 1)
        self.env.players[0].board[0].can_attack = True
        attack_action = ShadowverseEnv.ATTACK_OFFSET
        result = self.env.step(attack_action)
        self.assertEqual(self.env.players[1].health, 19)
        self.assertFalse(result.terminated)

    def test_guard_blocks_leader_and_other_targets(self) -> None:
        attacker = Unit.summon(card(100, attack=2, life=2))
        attacker.can_attack = True
        guard = Unit.summon(card(101, keywords=frozenset({"守护"})))
        other = Unit.summon(card(102))
        self.env.players[0].board = [attacker]
        self.env.players[1].board = [guard, other]

        mask = self.env.action_mask()
        base = ShadowverseEnv.ATTACK_OFFSET
        self.assertFalse(mask[base])
        self.assertTrue(mask[base + 1])
        self.assertFalse(mask[base + 2])

    def test_terminal_reward_belongs_to_actor(self) -> None:
        attacker = Unit.summon(card(100, attack=20, life=1))
        attacker.can_attack = True
        self.env.players[0].board = [attacker]
        result = self.env.step(ShadowverseEnv.ATTACK_OFFSET)
        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, 1.0)
        self.assertEqual(self.env.winner, 0)

    def test_evolution_adds_stats_and_grants_rush(self) -> None:
        unit = Unit.summon(card(100, attack=2, life=3))
        self.env.players[0].board = [unit]
        self.env.players[0].turns_started = ShadowverseEnv.EVOLUTION_UNLOCK_TURN
        action = ShadowverseEnv.EVOLVE_OFFSET
        self.assertTrue(self.env.action_mask()[action])
        self.env.step(action)
        self.assertTrue(unit.evolved)
        self.assertEqual((unit.attack, unit.health), (4, 5))
        self.assertTrue(unit.can_attack)
        self.assertTrue(unit.rush_only)
        self.assertEqual(self.env.players[0].evolution_points, 1)
        self.assertEqual(
            self.env.players[0].super_evolution_points,
            ShadowverseEnv.STARTING_SUPER_EVOLUTION_POINTS,
        )
        self.assertFalse(self.env.action_mask()[action])

    def test_super_evolution_uses_appended_action_slots(self) -> None:
        unit = Unit.summon(card(100, attack=2, life=3))
        self.env.players[0].board = [unit]
        self.env.players[0].turns_started = (
            ShadowverseEnv.FIRST_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN
        )
        action = ShadowverseEnv.SUPER_EVOLVE_OFFSET
        self.assertTrue(self.env.action_mask()[action])
        self.env.step(action)
        self.assertTrue(unit.evolved)
        self.assertTrue(unit.super_evolved)
        self.assertEqual((unit.attack, unit.health), (4, 5))
        self.assertEqual(self.env.players[0].evolution_points, 2)
        self.assertEqual(self.env.players[0].super_evolution_points, 1)
        self.assertTrue(self.env.players[0].super_evolved_this_turn)
        self.assertGreaterEqual(action, 106)

    def test_super_evolution_unlocks_later_for_first_player(self) -> None:
        unit = Unit.summon(card(100, attack=2, life=3))
        self.env.players[0].board = [unit]
        self.env.players[0].turns_started = (
            ShadowverseEnv.FIRST_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN - 1
        )
        action = ShadowverseEnv.SUPER_EVOLVE_OFFSET
        self.assertFalse(self.env.action_mask()[action])
        self.env.players[0].turns_started = (
            ShadowverseEnv.FIRST_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN
        )
        self.assertTrue(self.env.action_mask()[action])

    def test_super_evolution_unlocks_on_second_players_sixth_turn(self) -> None:
        unit = Unit.summon(card(101, attack=2, life=3))
        self.env.core.state.active_player = 1
        self.env.players[1].board = [unit]
        self.env.players[1].turns_started = (
            ShadowverseEnv.SECOND_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN - 1
        )
        action = ShadowverseEnv.SUPER_EVOLVE_OFFSET
        self.assertFalse(self.env.action_mask()[action])
        self.env.players[1].turns_started = (
            ShadowverseEnv.SECOND_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN
        )
        self.assertTrue(self.env.action_mask()[action])

    def test_super_evolution_has_two_manual_charges_and_once_per_turn(self) -> None:
        first = Unit.summon(card(100, attack=2, life=3))
        second = Unit.summon(card(101, attack=2, life=3))
        self.env.players[0].board = [first, second]
        self.env.players[0].turns_started = (
            ShadowverseEnv.FIRST_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN
        )

        self.env.step(ShadowverseEnv.SUPER_EVOLVE_OFFSET)
        self.assertEqual(self.env.players[0].super_evolution_points, 1)
        self.assertFalse(self.env.action_mask()[ShadowverseEnv.SUPER_EVOLVE_OFFSET + 1])

        self.env.players[0].evolved_this_turn = False
        self.env.players[0].super_evolved_this_turn = False
        self.assertTrue(self.env.action_mask()[ShadowverseEnv.SUPER_EVOLVE_OFFSET + 1])
        self.env.step(ShadowverseEnv.SUPER_EVOLVE_OFFSET + 1)
        self.assertEqual(self.env.players[0].super_evolution_points, 0)

        third = Unit.summon(card(102, attack=2, life=3))
        self.env.players[0].board = [third]
        self.env.players[0].evolved_this_turn = False
        self.env.players[0].super_evolved_this_turn = False
        self.assertFalse(self.env.action_mask()[ShadowverseEnv.SUPER_EVOLVE_OFFSET])

    def test_super_evolution_trigger_choice_uses_choice_action_mask(self) -> None:
        rulebook = RuleBook((CardRule(
            card_id=900,
            trigger=Trigger.SUPER_EVOLVE,
            operations=(
                EffectOperation(EffectKind.RETURN_TO_HAND, TargetKind.ENEMY_UNIT),
            ),
        ),))
        env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        env.reset(seed=1)
        source = Unit.summon(
            card(900, attack=2, life=3, keywords=frozenset({"超进化时"})),
            entity_id=env.core.state.allocate_entity_id(),
        )
        target = Unit.summon(
            card(901, attack=2, life=2),
            entity_id=env.core.state.allocate_entity_id(),
        )
        env.players[0].board = [source]
        env.players[1].board = [target]
        env.players[0].turns_started = (
            ShadowverseEnv.FIRST_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN
        )

        self.assertTrue(env.action_mask()[ShadowverseEnv.SUPER_EVOLVE_OFFSET])
        first_result = env.step(ShadowverseEnv.SUPER_EVOLVE_OFFSET)

        self.assertIsNotNone(env.core.state.pending_choice)
        self.assertTrue(source.super_evolved)
        self.assertTrue(first_result.info["action_mask"][ShadowverseEnv.CHOICE_OFFSET])
        self.assertFalse(
            first_result.info["action_mask"][ShadowverseEnv.SUPER_EVOLVE_OFFSET]
        )

        env.step(ShadowverseEnv.CHOICE_OFFSET)

        self.assertIsNone(env.core.state.pending_choice)
        self.assertNotIn(target, env.players[1].board)
        self.assertTrue(any(h.card_id == 901 for h in env.players[1].hand))

    def test_simple_fanfare_effects_resolve(self) -> None:
        fanfare = card(
            200,
            fanfare_effects=(
                EffectDefinition("damage_enemy_leader", 3),
                EffectDefinition("buff_self", 2, 1),
            ),
        )
        self.env.players[0].hand[0] = fanfare
        self.env.players[0].mana = 10
        self.env.step(ShadowverseEnv.PLAY_OFFSET)
        unit = self.env.players[0].board[0]
        self.assertEqual(self.env.players[1].health, 17)
        self.assertEqual((unit.attack, unit.health), (3, 2))
        self.assertTrue(any("入场曲" in line for line in self.env.logs))

    def test_non_collectible_card_is_rejected_from_deck(self) -> None:
        token = card(90044110)
        token = CardDefinition(
            **{
                **token.__dict__,
                "card_set_id": 90000,
                "is_collectible": False,
            }
        )
        with self.assertRaisesRegex(ValueError, "non-collectible"):
            ShadowverseEnv(
                [token] * 40,
                [card(1)] * 40,
                class_a=1,
                class_b=1,
            )

    def test_decks_must_have_40_cards_and_match_player_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 40"):
            ShadowverseEnv(
                [card(1)] * 39,
                [card(2)] * 40,
                class_a=1,
                class_b=1,
            )

        royal = CardDefinition(
            **{
                **card(3).__dict__,
                "class_id": 2,
                "class_name": "皇家护卫",
            }
        )
        with self.assertRaisesRegex(ValueError, "off-class"):
            ShadowverseEnv(
                [royal] * 40,
                [card(2)] * 40,
                class_a=1,
                class_b=1,
            )

    def test_all_documented_abilities_have_registered_handlers(self) -> None:
        self.assertEqual(len(ABILITY_DEFINITIONS), 34)
        self.assertEqual(len({item.keyword for item in ABILITY_DEFINITIONS}), 34)
        self.assertIn(
            AbilityKeyword.BANE,
            normalize_abilities({"毁灭"}),
        )
        self.assertIn(
            AbilityKeyword.DRAIN,
            normalize_abilities({"虹吸"}),
        )
        self.assertEqual(
            next(
                item.status
                for item in ABILITY_DEFINITIONS
                if item.keyword is AbilityKeyword.FANFARE
            ),
            AbilityStatus.PARTIAL,
        )

    def test_placeholder_ability_event_is_recorded_without_state_change(self) -> None:
        attacker = Unit.summon(
            card(300, attack=2, life=2, keywords=frozenset({"威慑"}))
        )
        attacker.can_attack = True
        self.env.players[0].board = [attacker]
        before_health = self.env.players[1].health
        self.env.step(ShadowverseEnv.ATTACK_OFFSET)
        events = self.env.info(debug=True)["placeholder_ability_events"]
        self.assertTrue(
            any(
                event.ability is AbilityKeyword.INTIMIDATE
                and event.event is AbilityEvent.BEFORE_ATTACK
                for event in events
            )
        )
        self.assertEqual(self.env.players[1].health, before_health - 2)

    def test_public_info_redacts_debug_transcript(self) -> None:
        info = self.env.info()
        self.assertEqual(set(info), PUBLIC_INFO_KEYS)
        self.assertIn("action_mask", info)
        self.assertIn("placeholder_ability_count", info)
        self.assertNotIn("log", info)
        self.assertNotIn("events", info)
        self.assertNotIn("placeholder_ability_events", info)
        self.assertNotIn("players", info)
        self.assertNotIn("deck_lists", info)
        self.assertNotIn("pending_choice", info)

    def test_debug_info_exposes_transcript_when_requested(self) -> None:
        info = self.env.info(debug=True)
        self.assertEqual(set(info), PUBLIC_INFO_KEYS | DEBUG_INFO_KEYS)
        self.assertIn("log", info)
        self.assertIn("events", info)
        self.assertIn("placeholder_ability_events", info)

    def test_constructor_debug_info_affects_reset_and_step_info(self) -> None:
        env = ShadowverseEnv(
            [card(i) for i in range(40)],
            [card(i) for i in range(40)],
            class_a=1,
            class_b=1,
            seed=1,
            debug_info=True,
        )
        _, info = env.reset(seed=1)
        self.assertIn("log", info)
        result = env.step(ShadowverseEnv.END_TURN)
        self.assertIn("events", result.info)

    def test_debug_false_redacts_even_when_constructor_debug_is_enabled(self) -> None:
        env = ShadowverseEnv(
            [card(i) for i in range(40)],
            [card(i) for i in range(40)],
            class_a=1,
            class_b=1,
            seed=1,
            debug_info=True,
        )
        env.reset(seed=1)
        info = env.info(debug=False)
        self.assertEqual(set(info), PUBLIC_INFO_KEYS)
        self.assertIn("action_mask", info)
        self.assertNotIn("log", info)
        self.assertNotIn("events", info)
        self.assertNotIn("placeholder_ability_events", info)

    def test_reset_and_step_public_info_use_redacted_key_set(self) -> None:
        env = ShadowverseEnv(
            [card(i) for i in range(40)],
            [card(i) for i in range(40)],
            class_a=1,
            class_b=1,
            seed=1,
        )
        _, reset_info = env.reset(seed=1)
        self.assertEqual(set(reset_info), PUBLIC_INFO_KEYS)

        step_info = env.step(ShadowverseEnv.END_TURN).info
        self.assertEqual(set(step_info), PUBLIC_INFO_KEYS)

    def test_pending_choice_public_info_redacts_options_and_labels(self) -> None:
        target = Unit.summon(card(400, attack=1, life=2), entity_id=902)
        self.env.players[1].board = [target]
        self.env.players[0].hand[0] = card(
            10041310,
            attack=None,
            life=None,
            card_type="法术",
        )
        self.env.players[0].mana = 10

        play_result = self.env.step(ShadowverseEnv.PLAY_OFFSET)

        self.assertIsNotNone(self.env.core.state.pending_choice)
        info = play_result.info
        self.assertEqual(set(info), PUBLIC_INFO_KEYS)
        for key in DEBUG_INFO_KEYS | {
            "pending_choice",
            "pending_choice_options",
            "choice_options",
            "players",
            "deck_lists",
        }:
            self.assertNotIn(key, info)
        public_repr = repr(info)
        self.assertNotIn(str(target.entity_id), public_repr)
        self.assertNotIn(target.definition.name, public_repr)

    def test_real_card_pending_choice_observation_and_info_hide_opponent_zones(self) -> None:
        target = Unit.summon(card(400, attack=1, life=2), entity_id=902)
        self.env.players[1].board = [target]
        self.env.players[0].hand[0] = card(
            10041310,
            attack=None,
            life=None,
            card_type="法术",
        )
        self.env.players[0].mana = 10

        self.env.step(ShadowverseEnv.PLAY_OFFSET)
        self.assertIsNotNone(self.env.core.state.pending_choice)
        before_obs = self.env.observation()
        before_info = self.env.info()
        opponent_hand = list(self.env.players[1].hand)
        self.env.players[1].hand = [
            HandCard(
                definition=card(
                    90000 + index,
                    cost=10,
                    attack=None,
                    life=None,
                    card_type="法术",
                ),
                entity_id=hidden_card.entity_id,
                origin=hidden_card.origin,
                source_origin=hidden_card.source_origin,
            )
            for index, hidden_card in enumerate(opponent_hand)
        ]
        opponent_deck_len = len(self.env.players[1].deck)
        self.env.players[1].deck = [
            card(
                91000 + index,
                cost=(index % 10) + 1,
                attack=None,
                life=None,
                card_type="法术",
            )
            for index, _ in enumerate(reversed(self.env.players[1].deck))
        ]
        self.assertEqual(len(self.env.players[1].deck), opponent_deck_len)

        self.assertEqual(self.env.observation(), before_obs)
        self.assertEqual(self.env.info(), before_info)

    def test_graveyard_page_turn_public_info_redacts_options_and_labels(self) -> None:
        self.env.core.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="墓地选择",
            options=tuple(
                ChoiceOption(
                    f"entity:{990000 + index}",
                    f"secret-graveyard-card-{index}",
                    entity_id=990000 + index,
                )
                for index in range(17)
            ),
            continuation_id="graveyard-redaction",
            choice_kind=ChoiceKind.GRAVEYARD,
            request_id=1234,
        )
        self.env.core.state.phase = Phase.AWAITING_CHOICE
        before_events = tuple(self.env.core.event_history)

        result = self.env.step(ShadowverseEnv.GRAVEYARD_NEXT_PAGE)

        self.assertEqual(set(result.info), PUBLIC_INFO_KEYS)
        self.assertEqual(result.info["graveyard_page"], 1)
        self.assertEqual(tuple(self.env.core.event_history), before_events)
        public_repr = repr(result.info)
        for index in range(17):
            self.assertNotIn(f"secret-graveyard-card-{index}", public_repr)
            self.assertNotIn(str(990000 + index), public_repr)
        self.assertNotIn("log", result.info)
        self.assertNotIn("events", result.info)
        self.assertNotIn("placeholder_ability_events", result.info)

    def test_public_observation_and_info_do_not_depend_on_opponent_hand_identity(self) -> None:
        before_obs = self.env.observation()
        before_info = self.env.info()
        old = self.env.players[1].hand[0]
        replacement = card(
            9000,
            cost=10,
            attack=None,
            life=None,
            card_type="法术",
        )
        self.env.players[1].hand[0] = HandCard(
            definition=replacement,
            entity_id=old.entity_id,
            origin=old.origin,
            source_origin=old.source_origin,
        )
        self.assertEqual(self.env.observation(), before_obs)
        self.assertEqual(self.env.info(), before_info)

    def test_public_observation_and_info_do_not_depend_on_opponent_deck_identity_or_order(self) -> None:
        before_obs = self.env.observation()
        before_info = self.env.info()
        opponent_deck_len = len(self.env.players[1].deck)
        self.env.players[1].deck = [
            card(
                9000 + index,
                cost=(index % 10) + 1,
                attack=None,
                life=None,
                card_type="法术",
            )
            for index, _ in enumerate(reversed(self.env.players[1].deck))
        ]
        self.assertEqual(len(self.env.players[1].deck), opponent_deck_len)
        self.assertEqual(self.env.observation(), before_obs)
        self.assertEqual(self.env.info(), before_info)

    def test_illegal_rl_action_does_not_mutate_core_state(self) -> None:
        before = env_snapshot(self.env)
        with self.assertRaises(ValueError):
            self.env.step(-1)
        self.assertEqual(env_snapshot(self.env), before)

    def test_illegal_rl_action_does_not_mutate_choice_page_state(self) -> None:
        self.env.core.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="pick",
            options=(
                ChoiceOption(
                    "entity:902",
                    "redacted target label",
                    entity_id=902,
                ),
            ),
            continuation_id="manual-test",
            choice_kind=ChoiceKind.BOARD,
            request_id=902,
        )
        self.env.core.state.phase = Phase.AWAITING_CHOICE
        self.env._graveyard_page = 7
        self.env._last_choice_request_key = ("stale-request", 99)
        before = env_snapshot(self.env)

        with self.assertRaises(ValueError):
            self.env.step(ShadowverseEnv.END_TURN)

        self.assertEqual(env_snapshot(self.env), before)

    def test_rl_choice_actions_resume_targeted_spell(self) -> None:
        spell = card(
            10041310,
            attack=None,
            life=None,
            card_type="法术",
        )
        target = Unit.summon(card(400, attack=1, life=2))
        self.env.players[0].hand[0] = spell
        self.env.players[0].mana = 10
        self.env.players[1].board = [target]

        self.env.step(ShadowverseEnv.PLAY_OFFSET)
        mask = self.env.action_mask()
        choices = [
            action
            for action in range(
                ShadowverseEnv.CHOICE_OFFSET,
                ShadowverseEnv.ACTION_SIZE,
            )
            if mask[action]
        ]
        self.assertEqual(len(choices), 1)
        self.env.step(choices[0])
        self.assertEqual(self.env.players[1].board, [])
        self.assertIsNone(self.env.core.state.pending_choice)

    def test_rl_multi_target_mask_decode_and_observation_progress_agree(self) -> None:
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        EffectKind.DAMAGE_UNIT,
                        TargetKind.ENEMY_UNIT,
                        amount=1,
                        target_count=2,
                    ),
                ),
            ),
        ))
        env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=3,
            rulebook=rulebook,
        )
        env.reset(seed=3)
        targets = [
            Unit.summon(card(900 + index, life=5), entity_id=900 + index)
            for index in range(2)
        ]
        env.players[1].board = targets
        env.players[0].hand[0] = card(
            1,
            attack=None,
            life=None,
            card_type="法术",
        )
        env.players[0].mana = 10

        env.step(ShadowverseEnv.PLAY_OFFSET)

        first_mask = env.action_mask()
        first_actions = [
            action
            for action in range(
                ShadowverseEnv.CHOICE_OFFSET,
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
            )
            if first_mask[action]
        ]
        self.assertEqual(len(first_actions), 2)
        self.assertEqual(env.observation()[-4:-2], [2 / 16, 0.0])
        first_option_id = env.core.state.pending_choice.options[0].option_id
        env.step(first_actions[0])

        request = env.core.state.pending_choice
        self.assertEqual(len(request.selected_options), 1)
        self.assertNotIn(
            first_option_id,
            {option.option_id for option in request.options},
        )
        second_mask = env.action_mask()
        second_actions = [
            action
            for action in range(
                ShadowverseEnv.CHOICE_OFFSET,
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
            )
            if second_mask[action]
        ]
        self.assertEqual(len(second_actions), 1)
        self.assertEqual(env.observation()[-4:-2], [2 / 16, 0.5])
        env.step(second_actions[0])

        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual([target.health for target in targets], [4, 4])

    def test_rl_choice_action_revalidates_target_after_controller_change(self) -> None:
        spell = card(
            10041310,
            attack=None,
            life=None,
            card_type="法术",
        )
        target = Unit.summon(card(400, attack=1, life=5), entity_id=902)
        self.env.players[0].hand[0] = spell
        self.env.players[0].mana = 10
        self.env.players[1].board = [target]

        self.env.step(ShadowverseEnv.PLAY_OFFSET)
        mask = self.env.action_mask()
        choice_action = next(
            action
            for action in range(
                ShadowverseEnv.CHOICE_OFFSET,
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
            )
            if mask[action]
        )
        self.env.players[1].board.remove(target)
        self.env.players[0].board.append(target)

        self.env.step(choice_action)

        self.assertIsNone(self.env.core.state.pending_choice)
        self.assertEqual(target.health, 5)
        self.assertEqual(self.env.players[0].board, [target])

    def test_rl_previous_target_revalidates_after_pause(self) -> None:
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.DAMAGE_UNIT,
                    target=TargetKind.ENEMY_UNIT,
                    amount=0,
                    target_key="sel",
                ),
                EffectOperation(
                    kind=EffectKind.OPTIONAL,
                    target=TargetKind.OWN_LEADER,
                    optional_operations=(
                        EffectOperation(
                            EffectKind.DRAW,
                            TargetKind.OWN_LEADER,
                            amount=1,
                        ),
                    ),
                ),
                EffectOperation(
                    kind=EffectKind.SET_STATS,
                    target=TargetKind.PREVIOUS_TARGET,
                    target_key="sel",
                    secondary_amount=1,
                    set_health=True,
                ),
            ),),
        ))
        env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        env.reset(seed=1)
        target = Unit.summon(card(902, attack=1, life=5), entity_id=902)
        env.players[1].board = [target]
        env.players[0].hand[0] = card(
            1,
            card_type="法术",
            attack=None,
            life=None,
        )
        env.players[0].mana = 10

        env.step(ShadowverseEnv.PLAY_OFFSET)
        target_action = next(
            action
            for action in range(
                ShadowverseEnv.CHOICE_OFFSET,
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
            )
            if env.action_mask()[action]
        )
        env.step(target_action)
        self.assertIsNotNone(env.core.state.pending_choice)
        env.players[1].board.remove(target)
        env.players[0].board.append(target)
        no_action = ShadowverseEnv.CHOICE_OFFSET + 1
        self.assertTrue(env.action_mask()[no_action])

        env.step(no_action)

        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual(target.health, 5)
        self.assertEqual(env.players[0].board, [target])

    def test_rl_target_exists_uses_existing_play_and_choice_actions(self) -> None:
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.TARGET_EXISTS,
                    target=TargetKind.ENEMY_UNIT,
                    then_operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.ENEMY_UNIT,
                            amount=3,
                        ),
                    ),
                    else_operations=(
                        EffectOperation(
                            kind=EffectKind.DRAW,
                            target=TargetKind.OWN_LEADER,
                            amount=1,
                        ),
                    ),
                ),
            ),),
        ))
        env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        env.reset(seed=1)
        target = Unit.summon(card(902, life=5), entity_id=902)
        env.players[1].board = [target]
        env.players[0].hand[0] = card(
            1,
            card_type="法术",
            attack=None,
            life=None,
        )
        env.players[0].mana = 10

        play_result = env.step(ShadowverseEnv.PLAY_OFFSET)

        self.assertNotIn("events", play_result.info)
        mask = env.action_mask()
        choices = [
            action
            for action in range(
                ShadowverseEnv.CHOICE_OFFSET,
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
            )
            if mask[action]
        ]
        self.assertEqual(len(choices), 1)
        env.step(choices[0])
        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual(target.health, 2)

        no_target_env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        no_target_env.reset(seed=1)
        no_target_env.players[0].hand[0] = card(
            1,
            card_type="法术",
            attack=None,
            life=None,
        )
        no_target_env.players[0].mana = 10
        deck_before = len(no_target_env.players[0].deck)
        self.assertTrue(no_target_env.action_mask()[ShadowverseEnv.PLAY_OFFSET])

        no_target_env.step(ShadowverseEnv.PLAY_OFFSET)

        self.assertIsNone(no_target_env.core.state.pending_choice)
        self.assertEqual(len(no_target_env.players[0].deck), deck_before - 1)

    def test_rl_target_exists_unit_or_leader_uses_choice_actions(self) -> None:
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.TARGET_EXISTS,
                    target=TargetKind.ENEMY_UNIT_OR_LEADER,
                    then_operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.ENEMY_UNIT_OR_LEADER,
                            amount=3,
                        ),
                    ),
                    else_operations=(
                        EffectOperation(
                            kind=EffectKind.DRAW,
                            target=TargetKind.OWN_LEADER,
                            amount=1,
                        ),
                    ),
                ),
            ),),
        ))
        env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        env.reset(seed=1)
        env.players[1].board = []
        env.players[0].hand[0] = card(
            1,
            card_type="法术",
            attack=None,
            life=None,
        )
        env.players[0].mana = 10
        self.assertTrue(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])

        play_result = env.step(ShadowverseEnv.PLAY_OFFSET)

        self.assertNotIn("events", play_result.info)
        self.assertIsNotNone(env.core.state.pending_choice)
        mask = env.action_mask()
        choices = [
            action
            for action in range(
                ShadowverseEnv.CHOICE_OFFSET,
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
            )
            if mask[action]
        ]
        self.assertEqual(choices, [ShadowverseEnv.CHOICE_OFFSET])
        env.step(choices[0])

        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual(env.players[1].health, 17)

    def test_rl_choice_actions_resume_follower_destroyed_emblem(self) -> None:
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(
                    kind=EffectKind.DRAW,
                    target=TargetKind.OWN_LEADER,
                    amount=1,
                ),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.DAMAGE_UNIT,
                    target=TargetKind.ALL_ENEMY_UNITS,
                    amount=5,
                ),
            ),),
        ))
        env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        env.reset(seed=1)
        target = Unit.summon(
            card(902, life=5),
            entity_id=env.core.state.allocate_entity_id(),
        )
        dead = Unit.summon(
            card(900, life=2),
            entity_id=env.core.state.allocate_entity_id(),
        )
        env.players[0].board = [target]
        env.players[1].board = [dead]
        first_emblem = EmblemDefinition(
            "env_death_choice",
            999963,
            triggers=(
                EmblemTriggerRule(
                    "follower_destroyed",
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.ENEMY_UNIT,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )
        second_emblem = EmblemDefinition(
            "env_death_leader",
            999964,
            triggers=(
                EmblemTriggerRule(
                    "follower_destroyed",
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_LEADER,
                            target=TargetKind.ENEMY_LEADER,
                            amount=2,
                        ),
                    ),
                ),
            ),
        )
        env.core._add_emblem_to_player(1, first_emblem, first_emblem.source_card_id)
        env.core._add_emblem_to_player(1, second_emblem, second_emblem.source_card_id)
        env.players[0].hand[0] = card(
            1,
            card_type="法术",
            attack=None,
            life=None,
        )
        env.players[0].mana = 10

        play_result = env.step(ShadowverseEnv.PLAY_OFFSET)

        self.assertEqual(env.decision_player, 1)
        self.assertNotIn("events", play_result.info)
        first_trigger = next(
            i for i, event in enumerate(env.core.event_history)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "env_death_choice"
        )
        first_destroyed = next(
            i for i, event in enumerate(env.core.event_history)
            if event.type == EventType.FOLLOWER_DESTROYED
            and event.metadata["card_id"] == 900
        )
        self.assertLess(first_destroyed, first_trigger)
        mask = env.action_mask()
        choices = [
            action
            for action in range(
                ShadowverseEnv.CHOICE_OFFSET,
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
            )
            if mask[action]
        ]
        self.assertEqual(len(choices), 1)

        env.step(choices[0])

        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual(target.health, 4)
        self.assertEqual(env.players[0].health, 18)
        events = env.core.event_history
        choice_damage = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DAMAGE_APPLIED
            and event.target_id == target.entity_id
        )
        self.assertLess(first_trigger, choice_damage)
        leader_damage = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DAMAGE_APPLIED
            and event.metadata.get("target_player") == 0
            and event.amount == 2
        )
        lw_start = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 900
        )
        self.assertLess(leader_damage, lw_start)

    def test_rl_choice_actions_resume_amulet_destroyed_emblem(self) -> None:
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.DESTROY,
                    target=TargetKind.ALL_OWN_AMULETS,
                ),
            ),),
        ))
        env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        env.reset(seed=1)
        target = Unit.summon(
            card(902, life=5),
            entity_id=env.core.state.allocate_entity_id(),
        )
        amulet = Amulet(
            definition=card(903, card_type="护符", attack=None, life=None),
            entity_id=env.core.state.allocate_entity_id(),
        )
        env.players[0].board = [amulet]
        env.players[1].board = [target]
        emblem = EmblemDefinition(
            "env_amulet_death_choice",
            999967,
            triggers=(
                EmblemTriggerRule(
                    "amulet_destroyed",
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.ENEMY_UNIT,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )
        env.core._add_emblem_to_player(0, emblem, emblem.source_card_id)
        env.players[0].hand[0] = card(
            1,
            card_type="法术",
            attack=None,
            life=None,
        )
        env.players[0].mana = 10

        play_result = env.step(ShadowverseEnv.PLAY_OFFSET)

        self.assertEqual(env.decision_player, 0)
        self.assertNotIn("events", play_result.info)
        mask = env.action_mask()
        choices = [
            action
            for action in range(
                ShadowverseEnv.CHOICE_OFFSET,
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
            )
            if mask[action]
        ]
        self.assertEqual(len(choices), 1)

        env.step(choices[0])

        self.assertEqual(env.players[0].board, [])
        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual(target.health, 4)
        events = env.core.event_history
        destroyed = next(
            i for i, event in enumerate(events)
            if event.type == EventType.AMULET_DESTROYED
            and event.metadata["card_id"] == 903
        )
        triggered = next(
            i for i, event in enumerate(events)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "env_amulet_death_choice"
        )
        damage = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DAMAGE_APPLIED
            and event.target_id == target.entity_id
        )
        self.assertLess(destroyed, triggered)
        self.assertLess(triggered, damage)

    def test_rl_defender_clash_choice_accepted(self) -> None:
        """Defender CLASH choice: RL decodes with request.player_index."""
        from swb.engine.card_rules import CardRule, RuleBook, Trigger
        from swb.engine.effects import EffectKind, EffectOperation, TargetKind
        from swb.engine.commands import Attack, Choose

        rulebook = RuleBook((
            CardRule(card_id=400, trigger=Trigger.CLASH, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_BOARD, amount=1),
            ),),
        ))
        env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        env.reset(seed=1)

        a = Unit.summon(card(300, attack=2, life=5), entity_id=env.core.state.allocate_entity_id())
        a.can_attack = True
        d = Unit.summon(card(400, attack=1, life=5, keywords=frozenset({"交战时"})), entity_id=env.core.state.allocate_entity_id())
        env.players[0].board = [a]
        env.players[1].board = [d]
        env.core._ensure_entity_ids()

        result = env.core.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertIsNotNone(env.core.state.pending_choice)
        request = env.core.state.pending_choice
        self.assertEqual(request.player_index, 1)

        encoded = env._encode_command(Choose(request.player_index, request.options[0].option_id))
        self.assertIsNotNone(encoded)
        decoded = env._decode_action(encoded)
        self.assertIsInstance(decoded, Choose)
        self.assertEqual(decoded.player_index, request.player_index)


if __name__ == "__main__":
    unittest.main()
