from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from swb.db.repository import CardDefinition
from swb.engine.card_rules import RuleBook
from swb.engine.conditions import EvalContext, evaluate_condition
from swb.engine.effects import Condition, ConditionType, SourceStateSnapshot
from swb.engine.environment import ShadowverseEnv
from swb.engine.state import HandCard, PlayerState, StatModifier, Unit


def _card(card_id: int, *, card_type: str = "随从") -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=1,
        card_type=card_type,
        attack=2 if card_type == "随从" else None,
        life=3 if card_type == "随从" else None,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


class HandStatModifierStateTests(unittest.TestCase):
    def test_modifier_changes_visible_stats_and_expires_by_owner_turn(self):
        card = HandCard(_card(1), entity_id=10)
        modifier = StatModifier(7, 2, 3, "until_end_of_turn", 0)
        card.add_stat_modifier(modifier)

        self.assertEqual((card.attack, card.life), (4, 6))
        card.expire_stat_modifiers("until_end_of_turn", 1)
        self.assertEqual((card.attack, card.life), (4, 6))
        card.expire_stat_modifiers("until_end_of_turn", 0)
        self.assertEqual((card.attack, card.life), (2, 3))

    def test_non_follower_cannot_receive_hand_stat_modifier(self):
        card = HandCard(_card(2, card_type="法术"), entity_id=11)
        with self.assertRaisesRegex(ValueError, "Only follower"):
            card.add_stat_modifier(StatModifier(8, 1, 1, "permanent"))

    def test_hand_transform_resets_stat_modifiers(self):
        deck = [_card(100 + index) for index in range(40)]
        env = ShadowverseEnv(deck, deck, class_a=1, class_b=1, seed=19)
        env.reset(seed=19)
        card = HandCard(_card(4), entity_id=env.core.state.allocate_entity_id())
        card.add_stat_modifier(StatModifier(9, 2, 2, "permanent"))

        env.core._transform_hand_card(
            card,
            _card(5),
            0,
            preserve_fused_materials=False,
        )

        self.assertEqual(card.stat_modifiers, [])
        self.assertEqual((card.attack, card.life), (2, 3))

    def test_source_health_condition_uses_live_entity_then_snapshot(self):
        source = Unit.summon(_card(3), entity_id=12)
        source.health = 2
        players = [
            PlayerState(deck=[], class_id=1, class_name="精灵"),
            PlayerState(deck=[], class_id=1, class_name="精灵"),
        ]
        players[0].board.append(source)
        condition = Condition(ConditionType.SOURCE_HEALTH_AT_MOST, value=2)
        self.assertTrue(evaluate_condition(
            condition,
            EvalContext(controller=0, players=players, source_entity_id=12),
        ))

        players[0].board.clear()
        snapshot = SourceStateSnapshot(
            entity_id=12,
            controller=0,
            card_id=3,
            card_type="随从",
            attack=2,
            health=2,
            evolved=False,
            super_evolved=False,
            effective_keywords=frozenset(),
        )
        self.assertTrue(evaluate_condition(
            condition,
            EvalContext(controller=0, players=players, source_snapshot=snapshot),
        ))
        self.assertFalse(evaluate_condition(
            Condition(ConditionType.SOURCE_HEALTH_AT_LEAST, value=3),
            EvalContext(controller=0, players=players, source_snapshot=snapshot),
        ))


class HandStatModifierSchemaTests(unittest.TestCase):
    def _load(self, operation: dict) -> None:
        payload = {
            "rules": [{
                "card_id": 999001,
                "trigger": "play",
                "operations": [operation],
            }]
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rules.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            RuleBook.from_directory(tmp)

    def test_hand_buff_accepts_enemy_followers_and_requires_follower_filter(self):
        self._load({
            "kind": "buff_hand_card",
            "target": "all_enemy_hand",
            "amount": 1,
            "hand_filter": {"card_type": "随从"},
        })
        with self.assertRaisesRegex(ValueError, "requires card_type='随从'"):
            self._load({
                "kind": "buff_hand_card",
                "target": "all_own_hand",
                "amount": 1,
            })


class HandStatModifierObservationTests(unittest.TestCase):
    def test_v1_observation_exposes_own_buffed_stats_without_width_change(self):
        deck = [_card(1000 + index) for index in range(40)]
        env = ShadowverseEnv(deck, deck, class_a=1, class_b=1, seed=23)
        env.reset(seed=23)
        hand_card = HandCard(_card(2001), entity_id=env.core.state.allocate_entity_id())
        env.players[0].hand = [hand_card]
        env.players[0].hand_entity_ids = [hand_card.entity_id]
        before = env.observation()

        hand_card.add_stat_modifier(StatModifier(1, 1, 2, "permanent"))
        after = env.observation()
        changed = [index for index, values in enumerate(zip(before, after)) if values[0] != values[1]]

        self.assertEqual(len(before), ShadowverseEnv.OBSERVATION_V1_SIZE)
        self.assertEqual(len(after), ShadowverseEnv.OBSERVATION_V1_SIZE)
        self.assertEqual(len(changed), 2)
        self.assertAlmostEqual(after[changed[0]] - before[changed[0]], 1 / 20)
        self.assertAlmostEqual(after[changed[1]] - before[changed[1]], 2 / 20)


if __name__ == "__main__":
    unittest.main()
