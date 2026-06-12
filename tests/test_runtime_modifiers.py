from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import (
    Condition,
    ConditionType,
    CostChangeMode,
    EffectKind,
    EffectOperation,
    ModifierDuration,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import CostModifier, HandCard, StatModifier, Unit


def card(
    card_id: int,
    *,
    cost: int = 1,
    attack: int | None = 1,
    life: int | None = 1,
    card_type: str = "随从",
    keywords: frozenset[str] = frozenset(),
    name: str | None = None,
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=name or f"card-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack,
        life=life,
        keywords=keywords,
        support_level="basic",
        is_collectible=True,
    )


def engine_with_rules(
    rules: tuple[CardRule, ...],
    *,
    seed: int = 1,
    resolver=None,
) -> GameEngine:
    deck_a = [card(1000 + index) for index in range(40)]
    deck_b = [card(2000 + index) for index in range(40)]
    engine = GameEngine(
        deck_a,
        deck_b,
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=RuleBook(rules),
        card_resolver=resolver,
    )
    engine.reset(seed=seed)
    engine.players[0].mana = 10
    return engine


def play_spell(
    engine: GameEngine,
    spell: CardDefinition,
    *,
    target_id: int | None = None,
) -> None:
    engine.players[0].hand[0] = spell
    engine.apply(PlayCard(0, 0))
    if target_id is None:
        return
    choices = [
        command
        for command in engine.legal_commands()
        if isinstance(command, Choose)
    ]
    choice = next(
        command
        for command in choices
        if next(
            option
            for option in engine.state.pending_choice.options
            if option.option_id == command.option_id
        ).entity_id
        == target_id
    )
    engine.apply(choice)


class RuntimeKeywordTests(unittest.TestCase):
    def test_add_and_remove_all_implemented_combat_keywords(self):
        unit = Unit.summon(card(1))
        for keyword in ("守护", "疾驰", "突进", "必杀", "吸血", "屏障", "潜行"):
            with self.subTest(keyword=keyword):
                unit.add_keyword(keyword)
                self.assertTrue(unit.has_keyword(keyword))
                unit.remove_keyword(keyword)
                self.assertFalse(unit.has_keyword(keyword))

    def test_keyword_aliases_and_runtime_state_are_synchronized(self):
        unit = Unit.summon(card(1))
        unit.add_keyword("毁灭")
        unit.add_keyword("虹吸")
        unit.add_keyword("屏障")
        unit.add_keyword("潜行")
        self.assertTrue(unit.has_keyword("必杀"))
        self.assertTrue(unit.has_keyword("吸血"))
        self.assertEqual(unit.barrier_charges, 1)
        self.assertTrue(unit.ambush_active)

        unit.remove_keyword("屏障")
        unit.remove_keyword("潜行")
        self.assertEqual(unit.barrier_charges, 0)
        self.assertFalse(unit.ambush_active)

    def test_dynamic_storm_and_rush_update_attack_permissions(self):
        storm = Unit.summon(card(1))
        storm.add_keyword("疾驰")
        self.assertTrue(storm.can_attack)
        self.assertTrue(storm.can_attack_leader)
        storm.remove_keyword("疾驰")
        self.assertFalse(storm.can_attack)

        rush = Unit.summon(card(2))
        rush.add_keyword("突进")
        self.assertTrue(rush.can_attack)
        self.assertFalse(rush.can_attack_leader)
        rush.remove_keyword("突进")
        self.assertFalse(rush.can_attack)

    def test_temporary_keyword_expires_at_end_of_turn(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rule = CardRule(
            10,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.ADD_KEYWORD,
                    TargetKind.ALL_OWN_UNITS,
                    keyword="守护",
                    duration=ModifierDuration.UNTIL_END_OF_TURN,
                ),
            ),
        )
        engine = engine_with_rules((rule,))
        target = Unit.summon(card(20))
        engine.players[0].board = [target]
        play_spell(engine, spell)
        self.assertTrue(target.has_keyword("守护"))
        engine.apply(EndTurn(0))
        self.assertFalse(target.has_keyword("守护"))

    def test_temporary_keyword_removal_restores_keyword_state(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rule = CardRule(
            10,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.REMOVE_KEYWORD,
                    TargetKind.OWN_UNIT,
                    keyword="潜行",
                    duration=ModifierDuration.UNTIL_END_OF_TURN,
                ),
                EffectOperation(
                    EffectKind.REMOVE_KEYWORD,
                    TargetKind.OWN_UNIT,
                    keyword="屏障",
                    duration=ModifierDuration.UNTIL_END_OF_TURN,
                ),
            ),
        )
        engine = engine_with_rules((rule,))
        target = Unit.summon(
            card(20, keywords=frozenset({"潜行", "屏障"}))
        )
        engine.players[0].board = [target]
        engine._ensure_entity_ids()
        engine.players[0].hand[0] = spell
        engine.apply(PlayCard(0, 0))
        first_choice = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        engine.apply(first_choice)
        second_choice = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        engine.apply(second_choice)
        self.assertFalse(target.has_keyword("潜行"))
        self.assertFalse(target.has_keyword("屏障"))
        self.assertFalse(target.ambush_active)
        self.assertEqual(target.barrier_charges, 0)

        engine.apply(EndTurn(0))
        self.assertTrue(target.has_keyword("潜行"))
        self.assertTrue(target.has_keyword("屏障"))
        self.assertTrue(target.ambush_active)
        self.assertEqual(target.barrier_charges, 1)

    def test_rush_only_restriction_ends_on_next_turn(self):
        unit = Unit.summon(card(1, keywords=frozenset({"突进"})))
        self.assertFalse(unit.can_attack_leader)
        unit.summoned_this_turn = False
        unit.rush_only = False
        unit.can_attack = True
        self.assertTrue(unit.can_attack_leader)

    def test_all_and_random_keyword_targets_are_deterministic(self):
        all_spell = card(10, card_type="法术", attack=None, life=None)
        random_spell = card(11, card_type="法术", attack=None, life=None)
        rules = (
            CardRule(
                10,
                Trigger.PLAY,
                (
                    EffectOperation(
                        EffectKind.ADD_KEYWORD,
                        TargetKind.ALL_ENEMY_UNITS,
                        keyword="守护",
                    ),
                ),
            ),
            CardRule(
                11,
                Trigger.PLAY,
                (
                    EffectOperation(
                        EffectKind.ADD_KEYWORD,
                        TargetKind.RANDOM_ENEMY_UNIT,
                        keyword="毁灭",
                    ),
                ),
            ),
        )
        results = []
        for _ in range(2):
            engine = engine_with_rules(rules, seed=42)
            engine.players[1].board = [
                Unit.summon(card(20 + index)) for index in range(3)
            ]
            play_spell(engine, all_spell)
            self.assertTrue(
                all(unit.has_keyword("守护") for unit in engine.players[1].board)
            )
            engine.players[0].hand[0] = random_spell
            engine.players[0].mana = 10
            engine.apply(PlayCard(0, 0))
            results.append(
                [
                    unit.definition.card_id
                    for unit in engine.players[1].board
                    if unit.has_keyword("必杀")
                ]
            )
        self.assertEqual(results[0], results[1])

    def test_false_condition_skips_keyword_change(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rule = CardRule(
            10,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.ADD_KEYWORD,
                    TargetKind.ALL_ENEMY_UNITS,
                    keyword="守护",
                    conditions=(
                        Condition(
                            ConditionType.CONTROLLER_HEALTH_AT_MOST,
                            value=5,
                        ),
                    ),
                ),
            ),
        )
        engine = engine_with_rules((rule,))
        target = Unit.summon(card(20))
        engine.players[1].board = [target]
        play_spell(engine, spell)
        self.assertFalse(target.has_keyword("守护"))


class HandCostTests(unittest.TestCase):
    def test_cost_set_add_subtract_and_floor(self):
        hand_card = HandCard(card(1, cost=5), entity_id=1)
        hand_card.cost_modifiers.append(CostModifier(1, "add", -2, "permanent"))
        self.assertEqual(hand_card.current_cost, 3)
        hand_card.cost_modifiers.append(CostModifier(2, "set", 7, "permanent"))
        self.assertEqual(hand_card.current_cost, 7)
        hand_card.cost_modifiers.append(CostModifier(3, "subtract", 20, "permanent"))
        self.assertEqual(hand_card.current_cost, 0)

    def test_action_mask_and_play_use_current_cost(self):
        deck = [card(index, cost=5) for index in range(40)]
        env = ShadowverseEnv(deck, deck, class_a=1, class_b=1, seed=1)
        env.reset(seed=1)
        hand_card = env.players[0].hand[0]
        self.assertIsInstance(hand_card, HandCard)
        hand_card.cost_modifiers.append(CostModifier(1, "set", 0, "permanent"))
        env.players[0].mana = 0
        self.assertTrue(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])
        env.step(ShadowverseEnv.PLAY_OFFSET)
        self.assertEqual(env.players[0].mana, 0)
        self.assertTrue(any("0费" in line for line in env.logs[-3:]))

    def test_temporary_cost_expires_at_end_of_turn(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rule = CardRule(
            10,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.CHANGE_COST,
                    TargetKind.OWN_HAND,
                    amount=-2,
                    mode=CostChangeMode.ADD,
                    duration=ModifierDuration.UNTIL_END_OF_TURN,
                ),
            ),
        )
        engine = engine_with_rules((rule,))
        engine._ensure_entity_ids()
        target = engine.players[0].hand[1]
        target.definition = card(99, cost=5)
        play_spell(engine, spell, target_id=target.entity_id)
        self.assertEqual(target.current_cost, 3)
        engine.apply(EndTurn(0))
        self.assertEqual(target.current_cost, 5)

    def test_all_hand_cost_change_and_return_to_hand_reset(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rule = CardRule(
            10,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.CHANGE_COST,
                    TargetKind.ALL_OWN_HAND,
                    amount=2,
                    mode=CostChangeMode.SUBTRACT,
                ),
            ),
        )
        engine = engine_with_rules((rule,))
        play_spell(engine, spell)
        self.assertTrue(
            all(hand_card.current_cost == 0 for hand_card in engine.players[0].hand)
        )

        returned = Unit.summon(card(99, cost=6))
        engine.players[0].board = [returned]
        engine._ensure_entity_ids()
        returned_id = returned.entity_id
        return_spell = card(11, card_type="法术", attack=None, life=None)
        engine.rulebook = RuleBook(
            (
                CardRule(
                    11,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.RETURN_TO_HAND,
                            TargetKind.OWN_UNIT,
                        ),
                    ),
                ),
            )
        )
        engine.players[0].hand[0] = return_spell
        engine.players[0].mana = 10
        play_spell(engine, return_spell, target_id=returned_id)
        returned_hand = next(
            card for card in engine.players[0].hand if card.card_id == 99
        )
        self.assertEqual(returned_hand.current_cost, 6)
        self.assertEqual(returned_hand.cost_modifiers, [])


class TransformAndStatModifierTests(unittest.TestCase):
    def test_self_target_is_resolved_for_generic_effects(self):
        follower = card(10, attack=2, life=2)
        rule = CardRule(
            10,
            Trigger.FANFARE,
            (
                EffectOperation(
                    EffectKind.DESTROY,
                    TargetKind.SELF,
                ),
            ),
        )
        engine = engine_with_rules((rule,))
        engine.players[0].hand[0] = follower
        engine.apply(PlayCard(0, 0))
        self.assertFalse(engine.players[0].board)
        self.assertEqual(engine.players[0].graveyard[-1].card_id, 10)

    def test_negative_buff_values_apply_debuffs(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rule = CardRule(
            10,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.BUFF_UNIT,
                    TargetKind.OWN_UNIT,
                    amount=-2,
                    secondary_amount=-3,
                ),
            ),
        )
        engine = engine_with_rules((rule,))
        target = Unit.summon(card(20, attack=5, life=6))
        engine.players[0].board = [target]
        engine._ensure_entity_ids()
        play_spell(engine, spell, target_id=target.entity_id)
        self.assertEqual((target.attack, target.health), (3, 3))
        self.assertTrue(
            any("属性变化 -2/-3" in line for line in engine.logs)
        )

    def test_transform_keeps_entity_id_and_clears_old_state(self):
        transform_spell = card(10, card_type="法术", attack=None, life=None)
        old = card(20, attack=5, life=6, keywords=frozenset({"守护"}))
        replacement = card(
            30,
            attack=2,
            life=3,
            keywords=frozenset({"屏障"}),
            name="replacement",
        )
        rules = (
            CardRule(
                10,
                Trigger.PLAY,
                (
                    EffectOperation(
                        EffectKind.TRANSFORM,
                        TargetKind.OWN_UNIT,
                        card_id=30,
                    ),
                ),
            ),
            CardRule(
                20,
                Trigger.LAST_WORDS,
                (
                    EffectOperation(
                        EffectKind.DAMAGE_LEADER,
                        TargetKind.ENEMY_LEADER,
                        amount=5,
                    ),
                ),
            ),
            CardRule(
                30,
                Trigger.FANFARE,
                (
                    EffectOperation(
                        EffectKind.DAMAGE_LEADER,
                        TargetKind.ENEMY_LEADER,
                        amount=5,
                    ),
                ),
            ),
        )
        resolver = lambda card_id: replacement if card_id == 30 else None
        engine = engine_with_rules(rules, resolver=resolver)
        target = Unit.summon(old)
        target.entity_id = 777
        target.evolved = True
        target.add_keyword("潜行")
        target.add_stat_modifier(
            StatModifier(1, 3, 4, "until_end_of_turn", 1)
        )
        engine.players[0].board = [target]
        before_health = [player.health for player in engine.players]

        play_spell(engine, transform_spell, target_id=777)

        self.assertEqual(target.entity_id, 777)
        self.assertEqual(target.definition.card_id, 30)
        self.assertEqual((target.attack, target.health), (2, 3))
        self.assertFalse(target.evolved)
        self.assertEqual(target.stat_modifiers, [])
        self.assertEqual(target.permanent_keywords, set())
        self.assertFalse(target.ambush_active)
        self.assertEqual(target.barrier_charges, 1)
        self.assertEqual(before_health, [player.health for player in engine.players])
        self.assertFalse(
            any(
                event.type
                in (EventType.FOLLOWER_DESTROYED, EventType.LAST_WORDS_START)
                for event in engine.event_history
            )
        )

    def test_transform_missing_or_non_follower_definition_is_explicit(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rule = CardRule(
            10,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.TRANSFORM,
                    TargetKind.ENEMY_UNIT,
                    card_id=999,
                ),
            ),
        )
        engine = engine_with_rules((rule,), resolver=lambda _: None)
        target = Unit.summon(card(20))
        engine.players[1].board = [target]
        engine.players[0].hand[0] = spell
        engine.apply(PlayCard(0, 0))
        choice = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        with self.assertRaisesRegex(IllegalCommand, "not found for TRANSFORM"):
            engine.apply(choice)

    def test_temporary_stats_expire_independently_and_preserve_damage(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rules = (
            CardRule(
                10,
                Trigger.PLAY,
                (
                    EffectOperation(
                        EffectKind.BUFF_UNIT,
                        TargetKind.ALL_OWN_UNITS,
                        amount=1,
                        secondary_amount=2,
                        duration=ModifierDuration.UNTIL_END_OF_TURN,
                    ),
                    EffectOperation(
                        EffectKind.BUFF_UNIT,
                        TargetKind.ALL_OWN_UNITS,
                        amount=3,
                        secondary_amount=4,
                        duration=ModifierDuration.UNTIL_END_OF_TURN,
                    ),
                ),
            ),
        )
        engine = engine_with_rules(rules)
        target = Unit.summon(card(20, attack=2, life=5))
        engine.players[0].board = [target]
        play_spell(engine, spell)
        self.assertEqual((target.attack, target.health), (6, 11))
        target.health -= 3
        engine.apply(EndTurn(0))
        self.assertEqual((target.attack, target.health), (2, 2))
        self.assertEqual(target.stat_modifiers, [])

    def test_expiring_health_modifier_stabilizes_before_player_switch(self):
        engine = engine_with_rules(())
        target = Unit.summon(card(20, attack=1, life=1))
        target.add_stat_modifier(
            StatModifier(1, 0, 2, "until_end_of_turn", 0)
        )
        target.health = 2
        engine.players[0].board = [target]
        engine.apply(EndTurn(0))
        destroyed = [
            event
            for event in engine.event_history
            if event.type is EventType.FOLLOWER_DESTROYED
            and event.source_id == target.entity_id
        ]
        self.assertEqual(len(destroyed), 1)
        self.assertEqual(destroyed[0].player_index, 0)


class ModifierSchemaTests(unittest.TestCase):
    def test_new_schema_loads_and_old_schema_remains_compatible(self):
        payload = {
            "rules": [
                {
                    "card_id": 10,
                    "trigger": "play",
                    "operations": [
                        {
                            "kind": "add_keyword",
                            "target": "own_unit",
                            "keyword": "毁灭",
                        },
                        {
                            "kind": "change_cost",
                            "target": "own_hand",
                            "amount": -1,
                            "mode": "add",
                            "duration": "until_end_of_turn",
                        },
                        {
                            "kind": "draw",
                            "target": "own_leader",
                            "amount": 1,
                        },
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rulebook = RuleBook.from_directory(tmp)
        operations = rulebook.operations_for(10, Trigger.PLAY)
        self.assertEqual(operations[0].keyword, "必杀")
        self.assertEqual(operations[1].mode, CostChangeMode.ADD)
        self.assertEqual(
            operations[1].duration,
            ModifierDuration.UNTIL_END_OF_TURN,
        )
        self.assertEqual(operations[2].amount, 1)

    def test_schema_errors_include_path_and_card_id(self):
        cases = (
            ({"kind": "add_keyword", "target": "own_unit", "keyword": "不存在"}, "keyword"),
            ({"kind": "add_keyword", "target": "own_unit", "keyword": "入场曲"}, "keyword"),
            ({"kind": "change_cost", "target": "own_hand", "mode": "bad"}, "mode"),
            ({"kind": "buff_unit", "target": "own_unit", "duration": "bad"}, "duration"),
            ({"kind": "transform", "target": "enemy_unit"}, "card_id"),
        )
        for operation, field in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                payload = {
                    "rules": [
                        {
                            "card_id": 123,
                            "trigger": "play",
                            "operations": [operation],
                        }
                    ]
                }
                Path(tmp, "bad.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    ValueError, rf"{field}.*card 123"
                ):
                    RuleBook.from_directory(tmp)


class RealCardRuleTests(unittest.TestCase):
    def test_real_keyword_and_cost_cards_execute_from_database(self):
        repository = CardRepository("data/cards.sqlite3")
        shield_assault = repository.get(10321310)
        dragon_charge = repository.get(10243310)
        self.assertEqual(shield_assault.name, "护盾强袭")
        self.assertEqual(dragon_charge.name, "龙骑突击")
        rulebook = RuleBook.from_directory("data/rules")

        engine = engine_with_rules(())
        engine.rulebook = rulebook
        own = Unit.summon(card(20, life=5))
        enemy = Unit.summon(card(21, life=6))
        engine.players[0].board = [own]
        engine.players[1].board = [enemy]
        engine._ensure_entity_ids()
        play_spell(engine, shield_assault, target_id=own.entity_id)
        self.assertTrue(own.has_keyword("守护"))
        self.assertEqual(enemy.health, 2)

        engine = engine_with_rules(())
        engine.rulebook = rulebook
        engine._ensure_entity_ids()
        hand_target = engine.players[0].hand[1]
        hand_target.definition = card(22, cost=5)
        enemies = [
            Unit.summon(card(30 + index, life=5)) for index in range(2)
        ]
        engine.players[1].board = enemies
        play_spell(engine, dragon_charge, target_id=hand_target.entity_id)
        self.assertEqual(hand_target.current_cost, 3)
        self.assertEqual([unit.health for unit in enemies], [2, 2])
