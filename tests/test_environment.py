from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.abilities import (
    ABILITY_DEFINITIONS,
    AbilityEvent,
    AbilityKeyword,
    AbilityStatus,
    normalize_abilities,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.model import HandCard, Unit
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
        self.assertEqual(len(self.env.observation()), 215)
        self.assertTrue(self.env.action_mask()[ShadowverseEnv.END_TURN])
        self.assertEqual(sum(self.env.observation()[28:35]), 1.0)
        self.assertEqual(sum(self.env.observation()[35:42]), 1.0)

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
        self.assertFalse(self.env.action_mask()[action])

    def test_super_evolution_uses_appended_action_slots(self) -> None:
        unit = Unit.summon(card(100, attack=2, life=3))
        self.env.players[0].board = [unit]
        self.env.players[0].turns_started = ShadowverseEnv.EVOLUTION_UNLOCK_TURN
        action = ShadowverseEnv.SUPER_EVOLVE_OFFSET
        self.assertTrue(self.env.action_mask()[action])
        self.env.step(action)
        self.assertTrue(unit.evolved)
        self.assertTrue(unit.super_evolved)
        self.assertEqual((unit.attack, unit.health), (4, 5))
        self.assertEqual(self.env.players[0].evolution_points, 1)
        self.assertGreaterEqual(action, 106)

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
        self.assertIn("action_mask", info)
        self.assertIn("placeholder_ability_count", info)
        self.assertNotIn("log", info)
        self.assertNotIn("events", info)
        self.assertNotIn("placeholder_ability_events", info)

    def test_debug_info_exposes_transcript_when_requested(self) -> None:
        info = self.env.info(debug=True)
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

    def test_illegal_rl_action_does_not_mutate_core_state(self) -> None:
        import copy

        before = (
            copy.deepcopy(self.env.core.state),
            tuple(self.env.logs),
            tuple(self.env.core.event_history),
            self.env.core.random.getstate(),
        )
        with self.assertRaises(ValueError):
            self.env.step(-1)
        after = (
            copy.deepcopy(self.env.core.state),
            tuple(self.env.logs),
            tuple(self.env.core.event_history),
            self.env.core.random.getstate(),
        )
        self.assertEqual(after, before)

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
