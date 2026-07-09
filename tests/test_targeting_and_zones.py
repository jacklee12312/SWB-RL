from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard, EndTurn
from swb.engine.effects import (
    BoardFilter,
    Condition,
    ConditionType,
    EffectKind,
    EffectOperation,
    ExprType,
    TargetKind,
    ValueExpression,
)
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import Amulet, Phase, Unit


def card(
    card_id: int,
    *,
    attack: int | None = 1,
    life: int | None = 1,
    cost: int = 1,
    card_type: str = "随从",
    name: str | None = None,
    is_collectible: bool = True,
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
        keywords=frozenset(),
        support_level="basic",
        is_collectible=is_collectible,
    )


def make_resolver(defs: dict[int, CardDefinition]):
    def resolve(cid: int) -> CardDefinition | None:
        return defs.get(cid)
    return resolve


def spell_rule(card_id: int, kind: EffectKind, target: TargetKind, **kwargs) -> CardRule:
    return CardRule(
        card_id=card_id,
        trigger=Trigger.PLAY,
        operations=(EffectOperation(kind=kind, target=target, **kwargs),),
    )


class TargetingTests(unittest.TestCase):
    """Tests for the unified targeting system."""

    def _real_pending_target_moved_to_graveyard(self):
        rulebook = RuleBook.from_directory("data/rules")
        spell = card(
            10153310,
            attack=None,
            life=None,
            cost=2,
            card_type="法术",
        )
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=11,
            rulebook=rulebook,
        )
        engine.reset(seed=11)
        target = Unit.summon(
            card(900, attack=1, life=5),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = spell

        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        choice = next(
            command
            for command in engine.legal_commands()
            if (
                isinstance(command, Choose)
                and command.option_id == f"entity:{target.entity_id}"
            )
        )
        engine.players[1].board.remove(target)
        engine._send_to_graveyard(
            1,
            target.definition,
            "test_pending_target_left_play",
            source_entity_id=target.entity_id,
        )
        return engine, target, choice

    def test_enemy_unit_or_leader_can_damage_selected_unit(self):
        rulebook = RuleBook((
            spell_rule(
                1,
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT_OR_LEADER,
                amount=3,
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=5), entity_id=900)
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        options = engine.state.pending_choice.options
        self.assertEqual(
            [option.option_id for option in options],
            [f"entity:{target.entity_id}", "leader:1"],
        )
        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertEqual(target.health, 2)
        self.assertEqual(engine.players[1].health, 20)

    def test_enemy_unit_or_leader_can_damage_selected_leader(self):
        rulebook = RuleBook((
            spell_rule(
                1,
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT_OR_LEADER,
                amount=3,
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=5), entity_id=900)
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, "leader:1"))

        self.assertEqual(target.health, 5)
        self.assertEqual(engine.players[1].health, 17)

    def test_real_card_pending_target_moved_to_graveyard_skips_and_continues(self):
        engine, target, choice = self._real_pending_target_moved_to_graveyard()

        engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(target.health, 5)
        self.assertEqual(engine.players[1].health, 20)
        self.assertEqual(engine.players[0].health, 18)
        self.assertTrue(
            any(
                graveyard_card.entity_id == target.entity_id
                for graveyard_card in engine.players[1].graveyard
            )
        )
        self.assertEqual(
            engine.players[0].graveyard[-1].definition.card_id,
            10153310,
        )
        self.assertTrue(any("已离场，跳过" in log for log in engine.logs))

        replay, _, replay_choice = self._real_pending_target_moved_to_graveyard()
        replay.apply(replay_choice)
        self.assertEqual(
            engine.deterministic_fingerprint(),
            replay.deterministic_fingerprint(),
        )

    def test_invalid_choice_after_target_zone_change_does_not_mutate(self):
        engine, target, choice = self._real_pending_target_moved_to_graveyard()
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "entity:999999"))

        self.assertEqual(engine.deterministic_fingerprint(), before)
        engine.apply(choice)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[0].health, 18)

    def test_pending_choice_target_changed_controller_skips_and_draws(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DAMAGE_UNIT,
                        target=TargetKind.ENEMY_UNIT,
                        amount=3,
                    ),
                    EffectOperation(
                        kind=EffectKind.DRAW,
                        target=TargetKind.OWN_LEADER,
                        amount=1,
                    ),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=13,
            rulebook=rulebook,
        )
        engine.reset(seed=13)
        target = Unit.summon(
            card(901, attack=1, life=5),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(
            1,
            attack=None,
            life=None,
            card_type="法术",
        )
        engine.apply(PlayCard(0, 0))
        choice = next(
            command
            for command in engine.legal_commands()
            if (
                isinstance(command, Choose)
                and command.option_id == f"entity:{target.entity_id}"
            )
        )
        deck_before = len(engine.players[0].deck)
        engine.players[1].board.remove(target)
        engine.players[0].board.append(target)

        engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertIn(target, engine.players[0].board)
        self.assertEqual(target.health, 5)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)
        self.assertTrue(any("已不再是合法目标，跳过" in log for log in engine.logs))

    def test_enemy_unit_or_leader_remains_playable_without_enemy_units(self):
        rulebook = RuleBook((
            spell_rule(
                1,
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT_OR_LEADER,
                amount=3,
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        play_cmds = [
            command for command in engine.legal_commands()
            if isinstance(command, PlayCard) and command.hand_index == 0
        ]
        self.assertEqual(len(play_cmds), 1)
        engine.apply(play_cmds[0])
        self.assertEqual(
            [option.option_id for option in engine.state.pending_choice.options],
            ["leader:1"],
        )
        engine.apply(Choose(0, "leader:1"))

        self.assertEqual(engine.players[1].health, 17)

    def test_unit_or_leader_target_rejects_board_filter_schema(self):
        with self.assertRaisesRegex(ValueError, "target_.*filter fields"):
            from swb.engine.card_rules import _parse_operation
            _parse_operation(
                {
                    "kind": "damage_unit",
                    "target": "enemy_unit_or_leader",
                    "amount": 3,
                    "target_card_type_filter": "随从",
                },
                "test.json/operations[0]",
                1,
            )

    def test_requires_target_schema_requires_bool(self):
        with self.assertRaisesRegex(ValueError, "requires_target.*boolean"):
            from swb.engine.card_rules import _parse_operation
            _parse_operation(
                {
                    "kind": "destroy",
                    "target": "enemy_unit",
                    "requires_target": "true",
                },
                "test.json/operations[0]",
                1,
            )

    def test_requires_target_schema_rejects_ambiguous_targets(self):
        from swb.engine.card_rules import _parse_operation

        cases = (
            {"kind": "draw", "target": "own_leader", "amount": 1},
            {"kind": "damage_leader", "target": "enemy_leader", "amount": 1},
            {"kind": "buff_unit", "target": "self", "amount": 1},
            {
                "kind": "damage_unit",
                "target": "previous_target",
                "target_key": "picked",
                "amount": 1,
            },
            {
                "kind": "damage_unit",
                "target": "enemy_unit_or_leader",
                "amount": 1,
            },
        )
        for raw in cases:
            with self.subTest(target=raw["target"]):
                raw = {**raw, "requires_target": True}
                with self.assertRaisesRegex(
                    ValueError,
                    "requires_target.*explicit candidate sets",
                ):
                    _parse_operation(raw, "test.json/operations[0]", 1)

    def test_requires_target_schema_allows_candidate_targets(self):
        from swb.engine.card_rules import _parse_operation

        cases = (
            {"kind": "destroy", "target": "enemy_unit"},
            {"kind": "destroy", "target": "random_enemy_unit"},
            {"kind": "damage_unit", "target": "all_enemy_units", "amount": 1},
            {"kind": "destroy", "target": "all_board"},
            {"kind": "discard", "target": "own_hand"},
            {
                "kind": "return_from_graveyard_to_hand",
                "target": "own_graveyard_card",
            },
        )
        for raw in cases:
            with self.subTest(target=raw["target"]):
                _parse_operation(
                    {**raw, "requires_target": True},
                    "test.json/operations[0]",
                    1,
                )

    def test_all_board_requires_target_no_candidates_is_illegal_no_mutation(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.DESTROY,
                    target=TargetKind.ALL_BOARD,
                    requires_target=True,
                ),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        before = (
            engine.players[0].mana,
            tuple(c.card_id for c in engine.players[0].hand),
            tuple(engine.players[0].board),
            tuple(engine.players[1].board),
            tuple(engine.event_history),
            tuple(engine.logs),
            tuple(engine.state.death_queue),
            engine.state.phase,
        )

        with self.assertRaisesRegex(IllegalCommand, "not currently playable"):
            engine.apply(PlayCard(0, 0))

        after = (
            engine.players[0].mana,
            tuple(c.card_id for c in engine.players[0].hand),
            tuple(engine.players[0].board),
            tuple(engine.players[1].board),
            tuple(engine.event_history),
            tuple(engine.logs),
            tuple(engine.state.death_queue),
            engine.state.phase,
        )
        self.assertEqual(after, before)

    def test_multi_target_schema_fields_are_explicitly_unsupported(self):
        from swb.engine.card_rules import _parse_operation

        cases = (
            {"target_count": 2},
            {"target_count_expr": {"type": "constant", "value": 2}},
            {"allow_duplicate_targets": False},
            {"allow_duplicates": True},
            {"targets": ["enemy_unit", "enemy_unit"]},
        )
        for extra in cases:
            with self.subTest(extra=extra):
                raw = {
                    "kind": "damage_unit",
                    "target": "enemy_unit",
                    "amount": 1,
                    **extra,
                }
                with self.assertRaisesRegex(ValueError, "multi-target choices are unsupported"):
                    _parse_operation(raw, "test.json/operations[0]", 1)

    def test_nested_multi_target_schema_fields_are_explicitly_unsupported(self):
        from swb.engine.card_rules import _parse_operation

        raw = {
            "kind": "choose_one",
            "target": "own_leader",
            "options": [
                {
                    "id": "bad",
                    "operations": [
                        {
                            "kind": "damage_unit",
                            "target": "enemy_unit",
                            "amount": 1,
                            "target_count": 2,
                        },
                    ],
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "multi-target choices are unsupported"):
            _parse_operation(raw, "test.json/operations[0]", 1)

    def test_target_exists_schema_loads_then_else_branches(self):
        from swb.engine.card_rules import _parse_operation

        operation = _parse_operation(
            {
                "kind": "target_exists",
                "target": "enemy_unit",
                "then": [
                    {
                        "kind": "damage_unit",
                        "target": "enemy_unit",
                        "amount": 2,
                    },
                ],
                "else": [
                    {"kind": "draw", "target": "own_leader", "amount": 1},
                ],
            },
            "test.json/operations[0]",
            1,
        )

        self.assertEqual(operation.kind, EffectKind.TARGET_EXISTS)
        self.assertEqual(operation.target, TargetKind.ENEMY_UNIT)
        self.assertEqual(len(operation.then_operations), 1)
        self.assertEqual(len(operation.else_operations), 1)

    def test_target_exists_schema_allows_unit_or_leader_targets(self):
        from swb.engine.card_rules import _parse_operation

        operation = _parse_operation(
            {
                "kind": "target_exists",
                "target": "enemy_unit_or_leader",
                "then": [
                    {
                        "kind": "damage_unit",
                        "target": "enemy_unit_or_leader",
                        "amount": 2,
                    },
                ],
            },
            "test.json/operations[0]",
            1,
        )

        self.assertEqual(operation.kind, EffectKind.TARGET_EXISTS)
        self.assertEqual(operation.target, TargetKind.ENEMY_UNIT_OR_LEADER)
        self.assertEqual(len(operation.then_operations), 1)

    def test_target_exists_schema_rejects_non_candidate_targets(self):
        from swb.engine.card_rules import _parse_operation

        cases = (
            {
                "kind": "target_exists",
                "target": "own_leader",
                "then": [{"kind": "draw", "target": "own_leader", "amount": 1}],
            },
            {
                "kind": "target_exists",
                "target": "previous_target",
                "target_key": "old",
                "then": [{"kind": "draw", "target": "own_leader", "amount": 1}],
            },
        )
        for raw in cases:
            with self.subTest(target=raw["target"]):
                with self.assertRaisesRegex(ValueError, "target_exists requires"):
                    _parse_operation(raw, "test.json/operations[0]", 1)

    def test_target_exists_schema_rejects_requires_target(self):
        from swb.engine.card_rules import _parse_operation

        with self.assertRaisesRegex(ValueError, "defines its own no-target"):
            _parse_operation(
                {
                    "kind": "target_exists",
                    "target": "enemy_unit",
                    "requires_target": True,
                    "then": [
                        {"kind": "destroy", "target": "enemy_unit"},
                    ],
                },
                "test.json/operations[0]",
                1,
            )

    def test_target_exists_then_branch_uses_existing_choice_flow(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
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
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=5), entity_id=900)
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        engine.apply(choice)
        self.assertEqual(target.health, 2)
        self.assertEqual(len(engine.players[0].deck), deck_before)

    def test_target_exists_else_branch_runs_without_target(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
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
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        deck_before = len(engine.players[0].deck)

        play_cmds = [
            command for command in engine.legal_commands()
            if isinstance(command, PlayCard) and command.hand_index == 0
        ]
        self.assertEqual(len(play_cmds), 1)
        engine.apply(play_cmds[0])

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_target_exists_unit_or_leader_then_branch_can_choose_leader(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
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
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        self.assertEqual(
            [option.option_id for option in engine.state.pending_choice.options],
            ["leader:1"],
        )
        engine.apply(Choose(0, "leader:1"))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[1].health, 17)
        self.assertEqual(len(engine.players[0].deck), deck_before)

    def test_target_exists_unit_or_leader_target_conditions_do_not_match_leader(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.TARGET_EXISTS,
                        target=TargetKind.ENEMY_UNIT_OR_LEADER,
                        conditions=(
                            Condition(
                                ConditionType.TARGET_HEALTH_AT_MOST,
                                value=3,
                            ),
                        ),
                        then_operations=(
                            EffectOperation(
                                kind=EffectKind.DAMAGE_LEADER,
                                target=TargetKind.ENEMY_LEADER,
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
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[1].health, 20)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_target_exists_conditions_filter_candidates_for_else_branch(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.TARGET_EXISTS,
                        target=TargetKind.ENEMY_UNIT,
                        conditions=(
                            Condition(
                                ConditionType.TARGET_HEALTH_AT_MOST,
                                value=3,
                            ),
                        ),
                        then_operations=(
                            EffectOperation(
                                kind=EffectKind.DAMAGE_UNIT,
                                target=TargetKind.ENEMY_UNIT,
                                conditions=(
                                    Condition(
                                        ConditionType.TARGET_HEALTH_AT_MOST,
                                        value=3,
                                    ),
                                ),
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
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=5), entity_id=900)
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(target.health, 5)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_banish_enemy_unit_moves_to_banished_zone(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=2, life=3))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)

        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(len(engine.players[1].graveyard), 0)
        self.assertEqual(len(engine.players[1].banished), 1)
        self.assertEqual(engine.players[1].banished[0].card_id, 900)

    def test_banish_does_not_trigger_last_words(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=2))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        transition = engine.apply(choice)

        destroyed_events = [e for e in transition.events if e.type == EventType.FOLLOWER_DESTROYED]
        banished_events = [e for e in transition.events if e.type == EventType.CARD_BANISHED]
        self.assertEqual(len(destroyed_events), 0)
        self.assertEqual(len(banished_events), 1)

    def test_random_target_reproducible_with_fixed_seed(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.DAMAGE_UNIT, TargetKind.RANDOM_ENEMY_UNIT, amount=5),
        ))
        results = []
        for trial in range(3):
            engine = GameEngine(
                [card(i) for i in range(100, 140)],
                [card(i) for i in range(200, 240)],
                class_a=1, class_b=1, seed=42, rulebook=rulebook,
            )
            engine.reset(seed=42)
            for i in range(3):
                unit = Unit.summon(card(900 + i, attack=1, life=10))
                engine.players[1].board.append(unit)
            engine.players[0].mana = 10
            engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
            engine.apply(PlayCard(0, 0))
            results.append([u.health for u in engine.players[1].board])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])

    def test_all_enemy_units_hits_all_targets(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.DAMAGE_UNIT, TargetKind.ALL_ENEMY_UNITS, amount=3),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        for i in range(3):
            unit = Unit.summon(card(900 + i, attack=1, life=5))
            engine.players[1].board.append(unit)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))

        for unit in engine.players[1].board:
            self.assertEqual(unit.health, 2)

    def test_no_legal_target_skips_operation(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(kind=EffectKind.BANISH, target=TargetKind.ENEMY_UNIT),
                    EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        hand_before = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(len(engine.players[0].hand), hand_before)

    def test_random_no_target_safe_skip(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.RANDOM_ENEMY_UNIT, amount=2),
                    EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertIsNone(engine.state.pending_choice)

    def test_all_no_target_safe_skip(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=2),
                    EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertIsNone(engine.state.pending_choice)

    def test_card_unplayable_when_all_choice_ops_have_no_targets(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        commands = engine.legal_commands()
        play_cmds = [c for c in commands if isinstance(c, PlayCard) and c.hand_index == 0]
        self.assertEqual(len(play_cmds), 0)

    def test_target_leaves_play_during_choice_does_not_crash(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.ANY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=2))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)

        engine.players[1].board = []
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertIsNone(engine.state.pending_choice)

    def test_own_target_entered_graveyard_during_choice_skips_and_continues(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(EffectKind.DESTROY, TargetKind.OWN_UNIT),
                    EffectOperation(
                        EffectKind.DRAW,
                        TargetKind.OWN_LEADER,
                        amount=1,
                    ),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=2))
        engine.players[0].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        choice = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        engine.players[0].board.remove(target)
        engine._send_to_graveyard(
            0,
            target.definition,
            "test_target_left_play",
            source_entity_id=target.entity_id,
        )
        deck_before = len(engine.players[0].deck)

        engine.apply(choice)

        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[0].board, [])
        self.assertEqual(len(engine.players[0].graveyard), 2)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_board_target_changed_controller_during_choice_skips_and_continues(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        EffectKind.DAMAGE_UNIT,
                        TargetKind.ENEMY_UNIT,
                        amount=3,
                    ),
                    EffectOperation(
                        EffectKind.DRAW,
                        TargetKind.OWN_LEADER,
                        amount=1,
                    ),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=5), entity_id=900)
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        choice = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        engine.players[1].board.remove(target)
        engine.players[0].board.append(target)
        deck_before = len(engine.players[0].deck)

        engine.apply(choice)

        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(target.health, 5)
        self.assertEqual(engine.players[0].board, [target])
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)
        self.assertTrue(any("已不再是合法目标，跳过" in log for log in engine.logs))

    def test_manual_board_filter_limits_choice_candidates(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DESTROY,
                        target=TargetKind.OWN_UNIT,
                        board_filter=BoardFilter(card_id=900),
                    ),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        valid = Unit.summon(card(900, name="valid"))
        invalid = Unit.summon(card(901, name="invalid"))
        engine.players[0].board = [valid, invalid]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))

        choices = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertEqual([c.option_id for c in choices], [f"entity:{valid.entity_id}"])

    def test_manual_board_filter_can_require_evolved_unit(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DAMAGE_UNIT,
                        target=TargetKind.ENEMY_UNIT,
                        amount=2,
                        board_filter=BoardFilter(evolved=True),
                    ),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        evolved = Unit.summon(card(900, life=5), entity_id=900)
        evolved.evolved = True
        unevolved = Unit.summon(card(901, life=5), entity_id=901)
        engine.players[1].board = [unevolved, evolved]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))

        choices = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertEqual([c.option_id for c in choices], [f"entity:{evolved.entity_id}"])

    def test_random_and_all_board_filters_share_candidate_logic(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DAMAGE_UNIT,
                        target=TargetKind.RANDOM_ENEMY_UNIT,
                        amount=2,
                        board_filter=BoardFilter(cost_min=3, cost_max=3),
                    ),
                    EffectOperation(
                        kind=EffectKind.DAMAGE_UNIT,
                        target=TargetKind.ALL_ENEMY_UNITS,
                        amount=1,
                        board_filter=BoardFilter(card_name="target"),
                    ),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        low = Unit.summon(card(900, cost=1, life=10, name="skip"))
        target = Unit.summon(card(901, cost=3, life=10, name="target"))
        high = Unit.summon(card(902, cost=5, life=10, name="skip"))
        engine.players[1].board = [low, target, high]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(low.health, 10)
        self.assertEqual(target.health, 7)
        self.assertEqual(high.health, 10)


class ZoneChangeTests(unittest.TestCase):
    """Tests for zone change effects."""

    def test_summon_follower_to_board(self):
        summoned_card = card(700, attack=3, life=4, is_collectible=False)
        resolver = make_resolver({700: summoned_card})
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.SUMMON, target=TargetKind.SELF, card_id=700,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
            card_resolver=resolver,
        )
        engine.reset(seed=1)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].board), 1)
        unit = engine.players[0].board[0]
        self.assertEqual(unit.attack, 3)
        self.assertEqual(unit.health, 4)

    def test_summon_fails_when_board_full(self):
        summoned_card = card(700, attack=1, life=1, is_collectible=False)
        resolver = make_resolver({700: summoned_card})
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.SUMMON, target=TargetKind.SELF, card_id=700,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
            card_resolver=resolver,
        )
        engine.reset(seed=1)
        for i in range(5):
            engine.players[0].board.append(Unit.summon(card(800 + i)))
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].board), 5)

    def test_add_card_to_hand(self):
        added_card = card(700, attack=None, life=None, card_type="法术", is_collectible=False)
        resolver = make_resolver({700: added_card})
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.ADD_CARD, target=TargetKind.OWN_LEADER, card_id=700,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
            card_resolver=resolver,
        )
        engine.reset(seed=1)
        hand_size_before = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].hand), hand_size_before)
        self.assertEqual(engine.players[0].hand[-1].card_id, 700)

    def test_add_card_full_hand_discards_to_graveyard(self):
        added_card = card(700, attack=None, life=None, card_type="法术", is_collectible=False)
        resolver = make_resolver({700: added_card})
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.ADD_CARD, target=TargetKind.OWN_LEADER, card_id=700,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
            card_resolver=resolver,
        )
        engine.reset(seed=1)
        engine.players[0].hand = [card(800 + i) for i in range(engine.config.max_hand + 1)]
        engine.players[0].hand_entity_ids = [engine.state.allocate_entity_id() for _ in range(engine.config.max_hand + 1)]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].hand), engine.config.max_hand)
        self.assertTrue(any(g.definition.card_id == 700 for g in engine.players[0].graveyard))

    def test_return_unit_to_hand(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.RETURN_TO_HAND, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=2, life=3))
        engine.players[1].board = [target]
        hand_before = len(engine.players[1].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(len(engine.players[1].hand), hand_before + 1)
        self.assertEqual(engine.players[1].hand[-1].card_id, 900)

    def test_return_to_hand_full_hand_banishes(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.RETURN_TO_HAND, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        for i in range(engine.config.max_hand):
            engine.players[1].hand.append(card(800 + i))
            engine.players[1].hand_entity_ids.append(engine.state.allocate_entity_id())
        target = Unit.summon(card(900, attack=2, life=3))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(len(engine.players[1].banished), 1)
        self.assertEqual(engine.players[1].banished[0].card_id, 900)

    def test_return_to_deck(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.RETURN_TO_DECK, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target_card = card(900, attack=2, life=3)
        target = Unit.summon(target_card)
        engine.players[1].board = [target]
        deck_before = len(engine.players[1].deck)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(len(engine.players[1].deck), deck_before + 1)
        self.assertTrue(any(c.card_id == 900 for c in engine.players[1].deck))

    def test_discard_moves_hand_to_graveyard(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.DISCARD, target=TargetKind.OWN_HAND,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        known = card(999, card_type="法术", attack=None, life=None, name="discard-target")
        engine.players[0].hand.append(known)
        engine.players[0].hand_entity_ids.append(engine.state.allocate_entity_id())
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        hand_before = len(engine.players[0].hand)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        options = engine.state.pending_choice.options
        known_option = next(o for o in options if "discard-target" in o.label)
        choice = Choose(engine.current_player, known_option.option_id)
        engine.apply(choice)
        self.assertEqual(len(engine.players[0].hand), hand_before - 2)
        self.assertTrue(any(g.definition.card_id == 999 for g in engine.players[0].graveyard))

    def test_old_rulebook_remains_compatible(self):
        rulebook = RuleBook.from_directory("data/rules")
        operations = rulebook.operations_for(10041130, Trigger.FANFARE)
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].amount, 6)

        shark = card(10041130, attack=4, life=3)
        engine = GameEngine(
            [shark] * 40, [card(2)] * 40,
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 14)

    def test_spell_with_both_choice_and_implicit_targets(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=3),
                    EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=5))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(target.health, 2)

    def test_any_unit_target_includes_both_sides(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.ANY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        own_unit = Unit.summon(card(700, attack=1, life=1))
        enemy_unit = Unit.summon(card(800, attack=1, life=1))
        engine.players[0].board = [own_unit]
        engine.players[1].board = [enemy_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        options = engine.state.pending_choice.options
        option_labels = [o.label for o in options]
        self.assertIn("card-700", option_labels)
        self.assertIn("card-800", option_labels)

    def test_own_amulet_target_correctly_filters(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.OWN_AMULET),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        unit = Unit.summon(card(700, attack=1, life=1))
        amulet = Amulet(definition=card(701, card_type="护符", attack=None, life=None), entity_id=1000)
        engine.players[0].board = [unit, amulet]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        options = engine.state.pending_choice.options
        option_ids = [o.entity_id for o in options]
        self.assertNotIn(unit.entity_id, option_ids)
        self.assertIn(amulet.entity_id, option_ids)

    def test_all_units_hits_both_sides(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.DAMAGE_UNIT, TargetKind.ALL_UNITS, amount=1),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        own_unit = Unit.summon(card(700, attack=1, life=3))
        enemy_unit = Unit.summon(card(800, attack=1, life=3))
        engine.players[0].board = [own_unit]
        engine.players[1].board = [enemy_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(own_unit.health, 2)
        self.assertEqual(enemy_unit.health, 2)

    def test_random_own_board_picks_from_own_side(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.RANDOM_OWN_BOARD),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        own_unit = Unit.summon(card(700, attack=1, life=1))
        engine.players[0].board = [own_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].board, [])
        self.assertEqual(len(engine.players[0].banished), 1)

    def test_hand_entity_ids_are_consistent(self):
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1,
        )
        engine.reset(seed=1)
        player = engine.players[0]
        self.assertEqual(len(player.hand), len(player.hand_entity_ids))
        self.assertTrue(all(eid > 0 for eid in player.hand_entity_ids))
        entity_set = set(player.hand_entity_ids)
        self.assertEqual(len(entity_set), len(player.hand_entity_ids))

    def test_damage_unit_with_any_board_filters_out_amulets(self):
        """Regression: DAMAGE_UNIT + ANY_BOARD must not allow selecting amulets."""
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ANY_BOARD, amount=3,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        unit = Unit.summon(card(700, attack=1, life=3))
        amulet = Amulet(definition=card(701, card_type="护符", attack=None, life=None), entity_id=1000)
        engine.players[0].board = [unit, amulet]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        options = engine.state.pending_choice.options
        option_eids = [o.entity_id for o in options]
        self.assertIn(unit.entity_id, option_eids)
        self.assertNotIn(amulet.entity_id, option_eids)

    def test_amulet_entity_ids_are_validated_and_unique(self):
        """Regression: _ensure_entity_ids must handle Amulet, not just Unit."""
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1,
        )
        engine.reset(seed=1)
        a1 = Amulet(definition=card(701, card_type="护符", attack=None, life=None))
        a2 = Amulet(definition=card(702, card_type="护符", attack=None, life=None))
        engine.players[0].board = [a1, a2]
        rules = list(engine.legal_commands())
        self.assertGreater(a1.entity_id, 0)
        self.assertGreater(a2.entity_id, 0)
        self.assertNotEqual(a1.entity_id, a2.entity_id)

    def test_discard_spell_unplayable_when_only_self_in_hand(self):
        """Regression: a discard spell must not count itself as a valid target."""
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.DISCARD, target=TargetKind.OWN_HAND,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].hand = [card(1, card_type="法术", attack=None, life=None)]
        engine.players[0].hand_entity_ids = [engine.state.allocate_entity_id()]
        engine.players[0].mana = 10
        commands = engine.legal_commands()
        play_cmds = [c for c in commands if isinstance(c, PlayCard)]
        self.assertEqual(len(play_cmds), 0)

    def test_discard_spell_playable_when_other_card_in_hand(self):
        """Discard spell is playable when at least one other card exists in hand."""
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.DISCARD, target=TargetKind.OWN_HAND,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].hand = [
            card(1, card_type="法术", attack=None, life=None),
            card(800, card_type="法术", attack=None, life=None),
        ]
        engine.players[0].hand_entity_ids = [
            engine.state.allocate_entity_id(),
            engine.state.allocate_entity_id(),
        ]
        engine.players[0].mana = 10
        commands = engine.legal_commands()
        play_cmds = [c for c in commands if isinstance(c, PlayCard)]
        self.assertEqual(len(play_cmds), 1)


class SourceLeavesPlayResolutionTests(unittest.TestCase):
    def _engine(self, rulebook: RuleBook) -> GameEngine:
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].mana = 10
        return engine

    def test_self_operation_after_source_leaves_play_is_skipped(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.FANFARE,
                operations=(
                    EffectOperation(EffectKind.DESTROY, TargetKind.SELF),
                    EffectOperation(
                        EffectKind.BUFF_UNIT,
                        TargetKind.SELF,
                        amount=2,
                        secondary_amount=2,
                    ),
                    EffectOperation(
                        EffectKind.DRAW,
                        TargetKind.OWN_LEADER,
                        amount=1,
                    ),
                ),
            ),
        ))
        engine = self._engine(rulebook)
        engine.players[0].hand[0] = card(1, attack=2, life=2)
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].board, [])
        self.assertEqual(engine.players[0].graveyard[-1].definition.card_id, 1)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_source_expression_after_source_leaves_play_is_skipped(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.FANFARE,
                operations=(
                    EffectOperation(EffectKind.DESTROY, TargetKind.SELF),
                    EffectOperation(
                        EffectKind.DAMAGE_LEADER,
                        TargetKind.ENEMY_LEADER,
                        amount_expr=ValueExpression(ExprType.SOURCE_ATTACK),
                    ),
                    EffectOperation(
                        EffectKind.DRAW,
                        TargetKind.OWN_LEADER,
                        amount=1,
                    ),
                ),
            ),
        ))
        engine = self._engine(rulebook)
        engine.players[0].hand[0] = card(1, attack=4, life=2)
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[1].health, 20)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_pending_choice_resumes_when_source_left_play(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.FANFARE,
                operations=(
                    EffectOperation(
                        EffectKind.DAMAGE_UNIT,
                        TargetKind.ENEMY_UNIT,
                        amount_expr=ValueExpression(ExprType.SOURCE_ATTACK),
                    ),
                    EffectOperation(
                        EffectKind.DRAW,
                        TargetKind.OWN_LEADER,
                        amount=1,
                    ),
                ),
            ),
        ))
        engine = self._engine(rulebook)
        target = Unit.summon(card(20, attack=1, life=5))
        engine.players[1].board = [target]
        engine.players[0].hand[0] = card(1, attack=3, life=2)

        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        source = engine.players[0].board[0]
        engine.players[0].board.remove(source)
        engine._send_to_graveyard(
            0,
            source.definition,
            "test_source_left_play",
            source_entity_id=source.entity_id,
        )
        deck_before = len(engine.players[0].deck)
        choice = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
        )

        engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(target.health, 5)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_source_left_play_resume_is_seed_deterministic(self):
        def run_once() -> tuple[int, int, tuple[str, ...]]:
            rulebook = RuleBook((
                CardRule(
                    card_id=1,
                    trigger=Trigger.FANFARE,
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_UNIT,
                            TargetKind.ENEMY_UNIT,
                            amount_expr=ValueExpression(ExprType.SOURCE_ATTACK),
                        ),
                        EffectOperation(
                            EffectKind.DRAW,
                            TargetKind.OWN_LEADER,
                            amount=1,
                        ),
                    ),
                ),
            ))
            engine = self._engine(rulebook)
            target = Unit.summon(card(20, attack=1, life=5))
            engine.players[1].board = [target]
            engine.players[0].hand[0] = card(1, attack=3, life=2)
            engine.apply(PlayCard(0, 0))
            source = engine.players[0].board[0]
            engine.players[0].board.remove(source)
            engine._send_to_graveyard(
                0,
                source.definition,
                "test_source_left_play",
                source_entity_id=source.entity_id,
            )
            choice = next(
                command
                for command in engine.legal_commands()
                if isinstance(command, Choose)
            )
            engine.apply(choice)
            return target.health, len(engine.players[0].deck), tuple(engine.logs)

        self.assertEqual(run_once(), run_once())


if __name__ == "__main__":
    unittest.main()
