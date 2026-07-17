# -*- coding: utf-8 -*-
"""Tests for decision system: conditional, choose_one, optional."""

from __future__ import annotations

import json
import tempfile
import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import ChoiceKind, Choose, EndTurn, PlayCard
from swb.engine.effects import (
    ChooseOneOption,
    Condition,
    ConditionType,
    EffectKind,
    EffectOperation,
    TargetKind,
)
from swb.engine.emblem import EmblemDefinition, EmblemTriggerRule, EventScope
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType, GameEvent
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import EmblemInstance, HandCard, Unit


def _card(cid, **kw):
    return CardDefinition(
        card_id=cid, card_set_id=10000, class_id=1, class_name="\u7cbe\u7075",
        name=kw.get("name", f"c{cid}"), cost=kw.get("cost", 1),
        card_type=kw.get("card_type", "\u968f\u4ece"),
        attack=kw.get("attack", 1), life=kw.get("life", 1),
        keywords=frozenset(), support_level="basic", is_collectible=True,
    )


def _engine(*rules, seed=42):
    return GameEngine(
        deck_a=[_card(i) for i in range(1000, 1040)],
        deck_b=[_card(i) for i in range(2000, 2040)],
        class_a=1, class_b=1, seed=seed,
        rulebook=RuleBook(tuple(rules)),
    )


def _sr(cid, *ops, trigger=Trigger.PLAY):
    return CardRule(card_id=cid, trigger=trigger, operations=ops)


def _insert_card(engine, card_def, origin=CardOrigin.DECK):
    hc = HandCard(definition=card_def, entity_id=engine.state.allocate_entity_id(), origin=origin)
    engine.players[0].hand.insert(0, hc)
    engine.players[0].hand_entity_ids.insert(0, hc.entity_id)
    return hc


def _add_emblem(engine, player_index, definition):
    player = engine.players[player_index]
    instance = EmblemInstance(
        emblem_id=definition.emblem_id,
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        controller=player_index,
        created_sequence=player._next_emblem_sequence,
        countdown=definition.countdown,
        countdown_before=definition.countdown,
    )
    player._next_emblem_sequence += 1
    player.emblems.append(instance)
    return instance


class ConditionalTests(unittest.TestCase):
    def test_conditional_then_branch(self):
        cond = EffectOperation(
            EffectKind.CONDITIONAL, TargetKind.OWN_LEADER,
            conditions=(Condition(type=ConditionType.ALWAYS),),
            then_operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 2),),
        )
        engine = _engine(_sr(999950, cond))
        engine.reset(seed=42)
        _insert_card(engine, _card(999950, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        deck_before = len(engine.players[0].deck)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].deck), deck_before - 2)

    def test_conditional_else_branch(self):
        cond = EffectOperation(
            EffectKind.CONDITIONAL, TargetKind.OWN_LEADER,
            conditions=(Condition(type=ConditionType.CONTROLLER_HEALTH_AT_MOST, value=0),),
            then_operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 3),),
            else_operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),),
        )
        engine = _engine(_sr(999951, cond))
        engine.reset(seed=42)
        _insert_card(engine, _card(999951, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        deck_before = len(engine.players[0].deck)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)


class ChooseOneTests(unittest.TestCase):
    def test_choose_one_creates_pending_choice(self):
        op = EffectOperation(
            EffectKind.CHOOSE_ONE, TargetKind.OWN_LEADER,
            choose_one_options=(
                ChooseOneOption("draw", "\u62bd\u724c", operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 2),)),
                ChooseOneOption("damage", "\u4f24\u5bb3", operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 3),)),
            ),
        )
        engine = _engine(_sr(999952, op))
        engine.reset(seed=42)
        _insert_card(engine, _card(999952, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        self.assertEqual(engine.state.pending_choice.choice_kind, ChoiceKind.MODE)
        self.assertGreaterEqual(len(engine.state.pending_choice.options), 2)

    def test_choose_one_choice_executes_selected_branch_immediately(self):
        op = EffectOperation(
            EffectKind.CHOOSE_ONE, TargetKind.OWN_LEADER,
            choose_one_options=(
                ChooseOneOption("draw", "抽牌", operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 2),)),
                ChooseOneOption("damage", "伤害", operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 3),)),
            ),
        )
        engine = _engine(_sr(999962, op))
        engine.reset(seed=42)
        _insert_card(engine, _card(999962, card_type="法术", cost=1))
        engine.players[0].mana = 10
        deck_before = len(engine.players[0].deck)
        engine.apply(PlayCard(0, 0))
        request = engine.state.pending_choice
        self.assertIsNotNone(request)

        engine.apply(Choose(request.player_index, "choose_one:draw"))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(len(engine.state.effect_stack), 0)
        self.assertEqual(len(engine.players[0].deck), deck_before - 2)

    def test_choose_multiple_collects_distinct_modes_before_resolving(self):
        op = EffectOperation(
            EffectKind.CHOOSE_ONE,
            TargetKind.OWN_LEADER,
            choose_one_options=(
                ChooseOneOption(
                    "first",
                    "第一项",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            1,
                        ),
                    ),
                ),
                ChooseOneOption(
                    "second",
                    "第二项",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            2,
                        ),
                    ),
                ),
                ChooseOneOption(
                    "third",
                    "第三项",
                    operations=(
                        EffectOperation(
                            EffectKind.DRAW,
                            TargetKind.OWN_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
            choose_count=2,
        )
        engine = _engine(_sr(999973, op))
        engine.reset(seed=42)
        _insert_card(engine, _card(999973, card_type="法术", cost=1))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        first_request = engine.state.pending_choice
        self.assertEqual(first_request.target_count, 2)
        self.assertEqual(engine.players[1].health, 20)

        engine.apply(Choose(0, "choose_one:second"))

        second_request = engine.state.pending_choice
        self.assertIsNotNone(second_request)
        self.assertNotEqual(second_request.request_id, first_request.request_id)
        self.assertEqual(
            tuple(option.option_id for option in second_request.selected_options),
            ("choose_one:second",),
        )
        self.assertNotIn(
            "choose_one:second",
            {option.option_id for option in second_request.options},
        )
        self.assertEqual(engine.players[1].health, 20)

        engine.apply(Choose(0, "choose_one:first"))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[1].health, 17)
        damage_amounts = [
            event.amount
            for event in engine.event_history
            if event.type is EventType.DAMAGE_APPLIED
            and event.metadata.get("target_player") == 1
        ]
        self.assertEqual(damage_amounts[-2:], [1, 2])

    def test_choose_multiple_duplicate_rejection_is_atomic(self):
        op = EffectOperation(
            EffectKind.CHOOSE_ONE,
            TargetKind.OWN_LEADER,
            choose_one_options=(
                ChooseOneOption("a", "A"),
                ChooseOneOption("b", "B"),
            ),
            choose_count=2,
        )
        engine = _engine(_sr(999974, op))
        engine.reset(seed=42)
        _insert_card(engine, _card(999974, card_type="法术", cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, "choose_one:a"))
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "choose_one:a"))

        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_choose_one_branch_can_request_target_choice(self):
        op = EffectOperation(
            EffectKind.CHOOSE_ONE, TargetKind.OWN_LEADER,
            choose_one_options=(
                ChooseOneOption(
                    "damage",
                    "伤害",
                    operations=(EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 2),),
                ),
            ),
        )
        engine = _engine(_sr(999963, op))
        engine.reset(seed=42)
        enemy = Unit.summon(_card(999964, life=3), entity_id=engine.state.allocate_entity_id())
        engine.players[1].board.append(enemy)
        _insert_card(engine, _card(999963, card_type="法术", cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        first = engine.state.pending_choice
        self.assertEqual(first.choice_kind, ChoiceKind.MODE)

        engine.apply(Choose(first.player_index, first.options[0].option_id))
        second = engine.state.pending_choice
        self.assertIsNotNone(second)
        self.assertEqual(second.choice_kind, ChoiceKind.BOARD)

        engine.apply(Choose(second.player_index, second.options[0].option_id))
        self.assertEqual(enemy.health, 1)
        self.assertEqual(len(engine.state.effect_stack), 0)

    def test_choose_one_mode_remains_selectable_without_current_target(self):
        op = EffectOperation(
            EffectKind.CHOOSE_ONE,
            TargetKind.OWN_LEADER,
            choose_one_options=(
                ChooseOneOption(
                    "damage",
                    "伤害",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_UNIT,
                            TargetKind.ENEMY_UNIT,
                            2,
                            requires_target=True,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine(_sr(999975, op))
        engine.reset(seed=42)
        _insert_card(engine, _card(999975, card_type="法术", cost=1))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        self.assertEqual(
            tuple(option.option_id for option in request.options),
            ("choose_one:damage",),
        )
        engine.apply(Choose(0, "choose_one:damage"))
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(len(engine.state.effect_stack), 0)

    def test_choose_one_branch_stale_target_skips_and_continues(self):
        op = EffectOperation(
            EffectKind.CHOOSE_ONE, TargetKind.OWN_LEADER,
            choose_one_options=(
                ChooseOneOption(
                    "damage_draw",
                    "伤害并抽牌",
                    operations=(
                        EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 2),
                        EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),
                    ),
                ),
            ),
        )
        engine = _engine(_sr(999971, op))
        engine.reset(seed=42)
        enemy = Unit.summon(_card(999972, life=5), entity_id=engine.state.allocate_entity_id())
        engine.players[1].board.append(enemy)
        _insert_card(engine, _card(999971, card_type="法术", cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        mode_choice = engine.state.pending_choice
        self.assertIsNotNone(mode_choice)

        engine.apply(Choose(mode_choice.player_index, mode_choice.options[0].option_id))
        target_choice = engine.state.pending_choice
        self.assertIsNotNone(target_choice)
        engine.players[1].board.remove(enemy)
        engine._send_to_graveyard(
            1,
            enemy.definition,
            "test_target_left_play",
            source_entity_id=enemy.entity_id,
        )
        deck_before = len(engine.players[0].deck)

        engine.apply(Choose(target_choice.player_index, target_choice.options[0].option_id))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(enemy.health, 5)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)
        self.assertEqual(len(engine.state.effect_stack), 0)

    def test_choose_one_uses_frame_controller_for_target_legality(self):
        op = EffectOperation(
            EffectKind.CHOOSE_ONE, TargetKind.OWN_LEADER,
            choose_one_options=(
                ChooseOneOption(
                    "damage",
                    "伤害",
                    operations=(EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 2),),
                ),
            ),
        )
        emblem_def = EmblemDefinition(
            "defender_choice",
            999969,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    event_scope=EventScope.OPPONENT_EVENT,
                    operations=(op,),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(999970, life=3), entity_id=engine.state.allocate_entity_id())
        engine.players[0].board.append(target)
        _add_emblem(engine, 1, emblem_def)

        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=1))
        engine._resolve_event_queue()

        self.assertIsNotNone(engine.state.pending_choice)
        self.assertEqual(engine.state.pending_choice.player_index, 1)
        self.assertEqual(engine.state.pending_choice.choice_kind, ChoiceKind.MODE)


class OptionalTests(unittest.TestCase):
    def test_optional_creates_pending_choice(self):
        op = EffectOperation(
            EffectKind.OPTIONAL, TargetKind.OWN_LEADER,
            optional_prompt="\u53d1\u52a8\uff1f",
            optional_operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),),
        )
        engine = _engine(_sr(999953, op))
        engine.reset(seed=42)
        _insert_card(engine, _card(999953, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        self.assertEqual(engine.state.pending_choice.choice_kind, ChoiceKind.CONFIRM)

    def test_optional_yes_executes_operations(self):
        op = EffectOperation(
            EffectKind.OPTIONAL, TargetKind.OWN_LEADER,
            optional_prompt="发动？",
            optional_operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),),
        )
        engine = _engine(_sr(999965, op))
        engine.reset(seed=42)
        _insert_card(engine, _card(999965, card_type="法术", cost=1))
        engine.players[0].mana = 10
        deck_before = len(engine.players[0].deck)
        engine.apply(PlayCard(0, 0))
        request = engine.state.pending_choice

        engine.apply(Choose(request.player_index, "optional:yes"))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(len(engine.state.effect_stack), 0)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_optional_no_skips_operations(self):
        op = EffectOperation(
            EffectKind.OPTIONAL, TargetKind.OWN_LEADER,
            optional_prompt="发动？",
            optional_operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),),
        )
        engine = _engine(_sr(999966, op))
        engine.reset(seed=42)
        _insert_card(engine, _card(999966, card_type="法术", cost=1))
        engine.players[0].mana = 10
        deck_before = len(engine.players[0].deck)
        engine.apply(PlayCard(0, 0))
        request = engine.state.pending_choice

        engine.apply(Choose(request.player_index, "optional:no"))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(len(engine.state.effect_stack), 0)
        self.assertEqual(len(engine.players[0].deck), deck_before)

    def test_emblem_optional_no_does_not_consume_once_per_turn(self):
        optional = EffectOperation(
            EffectKind.OPTIONAL, TargetKind.OWN_LEADER,
            optional_prompt="发动？",
            optional_operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),),
        )
        emblem_def = EmblemDefinition(
            "optional_emblem",
            999967,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    once_per_turn=True,
                    operations=(optional,),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        instance = _add_emblem(engine, 0, emblem_def)
        deck_before = len(engine.players[0].deck)

        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=1))
        engine._resolve_event_queue()
        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        engine.apply(Choose(request.player_index, "optional:no"))
        self.assertEqual(instance.activation_counts, {})

        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=2))
        engine._resolve_event_queue()
        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        engine.apply(Choose(request.player_index, "optional:yes"))

        self.assertEqual(instance.activation_counts[0], 1)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)


class SchemaValidationTests(unittest.TestCase):
    def test_conditional_requires_then(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaises(ValueError):
            _parse_operation(
                {"kind": "conditional", "conditions": [{"type": "always"}]},
                "test.json", 1,
            )

    def test_choose_one_requires_options(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaises(ValueError):
            _parse_operation(
                {"kind": "choose_one", "options": []},
                "test.json", 1,
            )

    def test_choose_one_rejects_duplicate_ids(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaises(ValueError):
            _parse_operation(
                {"kind": "choose_one", "options": [
                    {"id": "dup", "label": "a"},
                    {"id": "dup", "label": "b"},
                ]},
                "test.json", 1,
            )

    def test_choose_one_parses_choose_count(self):
        from swb.engine.card_rules import _parse_operation
        operation = _parse_operation(
            {
                "kind": "choose_one",
                "choose_count": 2,
                "options": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            },
            "test.json",
            1,
        )
        self.assertEqual(operation.choose_count, 2)

    def test_choose_one_rejects_invalid_choose_count(self):
        from swb.engine.card_rules import _parse_operation
        for value in (0, True, 3):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "choose_count",
            ):
                _parse_operation(
                    {
                        "kind": "choose_one",
                        "choose_count": value,
                        "options": [{"id": "a"}, {"id": "b"}],
                    },
                    "test.json",
                    1,
                )

    def test_conditional_rejects_unknown_field(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            _parse_operation(
                {
                    "kind": "conditional",
                    "conditions": [{"type": "always"}],
                    "then": [],
                    "bogus": True,
                },
                "test.json", 1,
            )

    def test_optional_rejects_empty_prompt(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaisesRegex(ValueError, "prompt"):
            _parse_operation(
                {"kind": "optional", "prompt": "", "operations": []},
                "test.json", 1,
            )

    def test_choose_one_rejects_unknown_field(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            _parse_operation(
                {
                    "kind": "choose_one",
                    "options": [{"id": "a", "label": "A"}],
                    "extra": 1,
                },
                "test.json", 1,
            )

    def test_conditional_rejects_target_dependent_condition(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaisesRegex(ValueError, "selected target"):
            _parse_operation(
                {
                    "kind": "conditional",
                    "conditions": [
                        {"type": "target_health_at_most", "value": 3}
                    ],
                    "then": [],
                },
                "test.json", 1,
            )

    def test_choose_one_option_rejects_target_dependent_condition(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaisesRegex(ValueError, "selected target"):
            _parse_operation(
                {
                    "kind": "choose_one",
                    "options": [
                        {
                            "id": "a",
                            "conditions": [
                                {"type": "target_health_at_most", "value": 3}
                            ],
                            "operations": [],
                        }
                    ],
                },
                "test.json", 1,
            )

    def test_previous_target_cannot_cross_choose_one_branch_boundary(self):
        payload = {
            "rules": [
                {
                    "card_id": 999968,
                    "trigger": "play",
                    "operations": [
                        {
                            "kind": "choose_one",
                            "options": [
                                {
                                    "id": "pick",
                                    "operations": [
                                        {
                                            "kind": "damage_unit",
                                            "target": "enemy_unit",
                                            "amount": 1,
                                            "target_key": "picked",
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "kind": "damage_unit",
                            "target": "previous_target",
                            "target_key": "picked",
                            "amount": 1,
                        },
                    ],
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "target_key"):
            with tempfile.TemporaryDirectory() as directory:
                path = f"{directory}/rules.json"
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                RuleBook.from_directory(directory)

    def test_nested_depth_limit(self):
        from swb.engine.card_rules import _parse_operation
        nested = {"kind": "draw", "target": "own_leader", "amount": 1}
        current = {"kind": "conditional", "conditions": [{"type": "always"}], "then": [nested]}
        for _ in range(17):
            current = {"kind": "conditional", "conditions": [{"type": "always"}], "then": [current]}
        with self.assertRaises(ValueError):
            _parse_operation(current, "test.json", 1)

    def test_duplicate_card_trigger_rule_rejected(self):
        payload = {
            "rules": [
                {
                    "card_id": 999969,
                    "trigger": "play",
                    "operations": [{"kind": "draw", "target": "own_leader", "amount": 1}],
                },
                {
                    "card_id": 999969,
                    "trigger": "play",
                    "operations": [{"kind": "draw", "target": "own_leader", "amount": 1}],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/rules.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "duplicate rule"):
                RuleBook.from_directory(directory)

    def test_nested_emblem_reference_is_validated(self):
        payload = {
            "rules": [
                {
                    "card_id": 999970,
                    "trigger": "play",
                    "operations": [
                        {
                            "kind": "conditional",
                            "conditions": [{"type": "always"}],
                            "then": [
                                {
                                    "kind": "gain_emblem",
                                    "target": "own_leader",
                                    "emblem_id": "missing",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/rules.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "unknown emblem_id"):
                RuleBook.from_directory(directory)

    def test_old_json_still_loads(self):
        rb = RuleBook.from_directory("data/rules")
        self.assertIsInstance(rb, RuleBook)

    def test_decisions_demo_loads(self):
        rb = RuleBook.from_directory("data/rules")
        ops = rb.operations_for(999950, Trigger.PLAY)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].kind, EffectKind.CONDITIONAL)


class ObservationTests(unittest.TestCase):
    def test_observation_294(self):
        env = ShadowverseEnv(
            deck_a=[_card(100 + i) for i in range(40)],
            deck_b=[_card(200 + i) for i in range(40)],
            class_a=1, class_b=1, seed=42,
        )
        obs, _ = env.reset(seed=42)
        self.assertEqual(len(obs), 294)

    def test_action_size(self):
        self.assertGreater(ShadowverseEnv.ACTION_SIZE, 100)


class BackwardCompatTests(unittest.TestCase):
    def test_rules_load(self):
        rb = RuleBook.from_directory("data/rules")
        self.assertIsInstance(rb, RuleBook)

    def test_old_play_works(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))


if __name__ == "__main__":
    unittest.main()
