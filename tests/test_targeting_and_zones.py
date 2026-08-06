from __future__ import annotations

import os
import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard, EndTurn
from swb.engine.effects import (
    BoardFilter,
    CandidateExtreme,
    Condition,
    ConditionType,
    EffectKind,
    EffectOperation,
    ExprType,
    TargetKind,
    ValueExpression,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import Amulet, HandCard, Phase, Unit


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

    def test_candidate_extreme_schema_and_leader_boundaries(self):
        from swb.engine.card_rules import _parse_operation

        operation = _parse_operation(
            {
                "kind": "destroy",
                "target": "random_enemy_unit",
                "candidate_extreme": "highest_attack",
            },
            "extreme.json",
            77,
        )
        self.assertIs(operation.candidate_extreme, CandidateExtreme.HIGHEST_ATTACK)

        leftmost = _parse_operation(
            {
                "kind": "grant_attacks_per_turn",
                "target": "all_own_units",
                "candidate_extreme": "leftmost",
                "amount": 2,
            },
            "extreme.json",
            77,
        )
        self.assertIs(leftmost.candidate_extreme, CandidateExtreme.LEFTMOST)

        invalid = (
            {
                "kind": "destroy",
                "target": "random_enemy_unit",
                "candidate_extreme": "largest_attack",
            },
            {
                "kind": "damage_leader",
                "target": "own_leader",
                "candidate_extreme": "lowest_health",
            },
            {
                "kind": "damage_leader",
                "target": "all_leaders",
                "candidate_extreme": "highest_attack",
            },
            {
                "kind": "damage_unit",
                "target": "all_leaders",
                "candidate_extreme": "highest_health",
            },
            {
                "kind": "damage_leader",
                "target": "all_leaders",
                "candidate_extreme": "leftmost",
            },
        )
        for raw in invalid:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    _parse_operation(raw, "extreme.json", 77)

    def test_random_extreme_selects_seeded_tie_only(self):
        fingerprints = []
        for _ in range(2):
            operation = EffectOperation(
                EffectKind.DESTROY,
                TargetKind.RANDOM_ENEMY_UNIT,
                candidate_extreme=CandidateExtreme.HIGHEST_ATTACK,
            )
            engine = GameEngine(
                [card(i) for i in range(100, 140)],
                [card(i) for i in range(200, 240)],
                class_a=1,
                class_b=1,
                seed=29,
                rulebook=RuleBook((CardRule(1, Trigger.PLAY, (operation,)),)),
            )
            engine.reset(seed=29)
            low = Unit.summon(card(900, attack=3, life=5), entity_id=900)
            high_a = Unit.summon(card(901, attack=6, life=5), entity_id=901)
            high_b = Unit.summon(card(902, attack=6, life=5), entity_id=902)
            engine.players[1].board = [low, high_a, high_b]
            engine.players[0].mana = 10
            engine.players[0].hand[0] = card(
                1, card_type="法术", attack=None, life=None
            )

            engine.apply(PlayCard(0, 0))

            self.assertIn(low, engine.players[1].board)
            self.assertEqual(
                sum(unit in engine.players[1].board for unit in (high_a, high_b)),
                1,
            )
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_leftmost_extreme_uses_filtered_board_order(self):
        operation = EffectOperation(
            EffectKind.GRANT_ATTACKS_PER_TURN,
            TargetKind.ALL_OWN_UNITS,
            amount=2,
            board_filter=BoardFilter(card_name="皇家随从"),
            candidate_extreme=CandidateExtreme.LEFTMOST,
        )
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=30,
            rulebook=RuleBook((CardRule(1, Trigger.PLAY, (operation,)),)),
        )
        engine.reset(seed=30)
        neutral = Unit.summon(card(900, name="中立随从"), entity_id=900)
        left_royal = Unit.summon(card(901, name="皇家随从"), entity_id=901)
        right_royal = Unit.summon(card(902, name="皇家随从"), entity_id=902)
        engine.players[0].board = [neutral, left_royal, right_royal]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(neutral.attacks_per_turn, 1)
        self.assertEqual(left_royal.attacks_per_turn, 2)
        self.assertEqual(right_royal.attacks_per_turn, 1)

    def test_all_extreme_snapshots_every_tied_follower(self):
        operation = EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.ALL_UNITS,
            amount=2,
            candidate_extreme=CandidateExtreme.HIGHEST_HEALTH,
        )
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=31,
            rulebook=RuleBook((CardRule(1, Trigger.PLAY, (operation,)),)),
        )
        engine.reset(seed=31)
        own = Unit.summon(card(910, life=5), entity_id=910)
        tied = Unit.summon(card(911, life=5), entity_id=911)
        lower = Unit.summon(card(912, life=4), entity_id=912)
        engine.players[0].board = [own]
        engine.players[1].board = [tied, lower]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))

        self.assertEqual((own.health, tied.health, lower.health), (3, 3, 4))

    def test_all_leaders_extreme_keeps_ties_and_targets_current_health(self):
        def run(own_health: int, enemy_health: int) -> tuple[int, int]:
            operation = EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ALL_LEADERS,
                amount=3,
                candidate_extreme=CandidateExtreme.LOWEST_HEALTH,
            )
            engine = GameEngine(
                [card(i) for i in range(100, 140)],
                [card(i) for i in range(200, 240)],
                class_a=1,
                class_b=1,
                seed=37,
                rulebook=RuleBook((CardRule(1, Trigger.PLAY, (operation,)),)),
            )
            engine.reset(seed=37)
            engine.players[0].health = own_health
            engine.players[1].health = enemy_health
            engine.players[0].mana = 10
            engine.players[0].hand[0] = card(
                1, card_type="法术", attack=None, life=None
            )
            engine.apply(PlayCard(0, 0))
            return engine.players[0].health, engine.players[1].health

        self.assertEqual(run(10, 15), (7, 15))
        self.assertEqual(run(10, 10), (7, 7))

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

    def test_multi_target_schema_fields_are_supported(self):
        from swb.engine.card_rules import _parse_operation

        cases = (
            ({"target_count": 2}, 2, None, False),
            (
                {"target_count_expr": {"type": "constant", "value": 2}},
                1,
                ExprType.CONSTANT,
                False,
            ),
            ({"allow_duplicate_targets": False}, 1, None, False),
            ({"allow_duplicates": True}, 1, None, True),
        )
        for extra, count, expr_type, allow_duplicates in cases:
            with self.subTest(extra=extra):
                raw = {
                    "kind": "damage_unit",
                    "target": "enemy_unit",
                    "amount": 1,
                    **extra,
                }
                operation = _parse_operation(raw, "test.json/operations[0]", 1)
                self.assertEqual(operation.target_count, count)
                self.assertEqual(
                    None if operation.target_count_expr is None else operation.target_count_expr.type,
                    expr_type,
                )
                self.assertEqual(
                    operation.allow_duplicate_targets,
                    allow_duplicates,
                )

    def test_random_board_multi_target_schema_is_explicitly_supported(self):
        from swb.engine.card_rules import _parse_operation

        for target in (
            "random_own_unit",
            "random_enemy_unit",
            "random_own_board",
            "random_enemy_board",
        ):
            with self.subTest(target=target):
                operation = _parse_operation(
                    {
                        "kind": "damage_unit",
                        "target": target,
                        "amount": 1,
                        "target_count": 2,
                    },
                    "test.json/operations[0]",
                    1,
                )
                self.assertEqual(operation.target_count, 2)

        for target in (
            "random_own_hand",
            "random_own_graveyard_card",
            "random_enemy_unit_or_leader",
        ):
            with self.subTest(target=target):
                with self.assertRaisesRegex(
                    ValueError,
                    "selected or random board target",
                ):
                    _parse_operation(
                        {
                            "kind": "damage_unit",
                            "target": target,
                            "amount": 1,
                            "target_count": 2,
                        },
                        "test.json/operations[0]",
                        1,
                    )

    def test_source_exclusion_schema_is_explicit_and_board_only(self):
        from swb.engine.card_rules import _parse_operation

        operation = _parse_operation(
            {
                "kind": "destroy",
                "target": "any_board",
                "target_count": 3,
                "exclude_source": True,
            },
            "test.json/operations[0]",
            1,
        )
        self.assertTrue(operation.exclude_source)
        all_operation = _parse_operation(
            {
                "kind": "buff_unit",
                "target": "all_own_units",
                "amount": 1,
                "secondary_amount": 1,
                "exclude_source": True,
            },
            "test.json/operations[0]",
            1,
        )
        self.assertTrue(all_operation.exclude_source)
        random_operation = _parse_operation(
            {
                "kind": "damage_unit",
                "target": "random_own_unit",
                "amount": 1,
                "exclude_source": True,
            },
            "test.json/operations[0]",
            1,
        )
        self.assertTrue(random_operation.exclude_source)
        for invalid in (
            {
                "kind": "draw",
                "target": "own_leader",
                "amount": 1,
                "exclude_source": True,
            },
            {
                "kind": "destroy",
                "target": "any_board",
                "exclude_source": "yes",
            },
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _parse_operation(invalid, "test.json/operations[0]", 1)

    def test_preselected_multi_target_payload_fields_remain_unsupported(self):
        from swb.engine.card_rules import _parse_operation

        with self.assertRaisesRegex(ValueError, "preselected multi-target payloads"):
            _parse_operation(
                {
                    "kind": "damage_unit",
                    "target": "enemy_unit",
                    "amount": 1,
                    "targets": ["enemy_unit", "enemy_unit"],
                },
                "test.json/operations[0]",
                1,
            )

    def test_nested_multi_target_schema_fields_are_supported(self):
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
        operation = _parse_operation(raw, "test.json/operations[0]", 1)
        nested = operation.choose_one_options[0].operations[0]
        self.assertEqual(nested.target_count, 2)

    def test_multi_target_schema_rejects_ambiguous_combinations(self):
        from swb.engine.card_rules import _parse_operation

        cases = (
            (
                {
                    "target_count": 2,
                    "target_count_expr": {"type": "constant", "value": 2},
                },
                "mutually exclusive",
            ),
            (
                {
                    "allow_duplicate_targets": True,
                    "allow_duplicates": False,
                },
                "conflict",
            ),
            ({"target_count": 0}, "must be positive"),
        )
        for extra, message in cases:
            with self.subTest(extra=extra):
                with self.assertRaisesRegex(ValueError, message):
                    _parse_operation(
                        {
                            "kind": "damage_unit",
                            "target": "enemy_unit",
                            "amount": 1,
                            **extra,
                        },
                        "test.json/operations[0]",
                        1,
                    )

    def test_multi_target_schema_accepts_target_key_binding(self):
        from swb.engine.card_rules import _parse_operation, _validate_target_keys

        operation = _parse_operation(
            {
                "kind": "damage_unit",
                "target": "enemy_unit",
                "amount": 1,
                "target_count": 2,
                "target_key": "picked",
            },
            "test.json/operations[0]",
            1,
        )
        _validate_target_keys((operation,), "test.json")
        self.assertEqual(operation.target_count, 2)
        self.assertEqual(operation.target_key, "picked")

    def _multi_target_engine(
        self,
        operation: EffectOperation,
        *,
        target_count: int,
    ):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(operation,)),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=17,
            rulebook=rulebook,
        )
        engine.reset(seed=17)
        targets = [
            Unit.summon(
                card(900 + index, attack=1, life=5),
                entity_id=engine.state.allocate_entity_id(),
            )
            for index in range(target_count)
        ]
        engine.players[1].board = list(targets)
        engine.players[0].hand[0] = card(
            1,
            card_type="法术",
            attack=None,
            life=None,
        )
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        return engine, targets

    def test_multi_target_choice_resolves_distinct_targets(self):
        engine, targets = self._multi_target_engine(
            EffectOperation(
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT,
                amount=2,
                target_count=2,
            ),
            target_count=2,
        )
        request = engine.state.pending_choice
        self.assertEqual(request.target_count, 2)

        engine.apply(Choose(0, f"entity:{targets[0].entity_id}"))

        request = engine.state.pending_choice
        self.assertEqual(len(request.selected_options), 1)
        self.assertNotIn(
            f"entity:{targets[0].entity_id}",
            {command.option_id for command in engine.legal_commands()},
        )
        engine.assert_invariants()
        engine.apply(Choose(0, f"entity:{targets[1].entity_id}"))
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual([target.health for target in targets], [3, 3])

    def test_mixed_board_multi_target_excludes_source_from_legality_and_choice(self):
        operation = EffectOperation(
            EffectKind.DESTROY,
            TargetKind.ANY_BOARD,
            target_count=3,
            exclude_source=True,
        )
        rulebook = RuleBook((
            CardRule(1, Trigger.FANFARE, (operation,)),
        ))
        engine = GameEngine(
            [card(index) for index in range(100, 140)],
            [card(index) for index in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=17,
            rulebook=rulebook,
        )
        engine.reset(seed=17)
        own_unit = Unit.summon(
            card(901), entity_id=engine.state.allocate_entity_id()
        )
        own_amulet = Amulet(
            definition=card(902, card_type="护符", attack=None, life=None),
            entity_id=engine.state.allocate_entity_id(),
        )
        enemy_unit = Unit.summon(
            card(903), entity_id=engine.state.allocate_entity_id()
        )
        enemy_amulet = Amulet(
            definition=card(904, card_type="护符", attack=None, life=None),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board = [own_unit, own_amulet]
        engine.players[1].board = [enemy_unit, enemy_amulet]
        engine.players[0].hand[0] = card(1)
        engine.players[0].max_mana = engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        source = next(
            entity for entity in engine.players[0].board
            if entity.definition.card_id == 1
        )
        request = engine.state.pending_choice
        self.assertEqual(request.target_count, 3)
        option_ids = {option.option_id for option in request.options}
        self.assertNotIn(f"entity:{source.entity_id}", option_ids)
        self.assertEqual(len(option_ids), 4)
        self.assertNotIn(
            Choose(0, f"entity:{source.entity_id}"),
            engine.legal_commands(),
        )
        for option in request.options[:3]:
            engine.apply(Choose(0, option.option_id))
        self.assertIn(source, engine.players[0].board)
        self.assertIsNone(engine.state.pending_choice)

    def test_all_own_units_excludes_source_without_creating_a_choice(self):
        operation = EffectOperation(
            EffectKind.BUFF_UNIT,
            TargetKind.ALL_OWN_UNITS,
            amount=1,
            secondary_amount=1,
            exclude_source=True,
        )
        rulebook = RuleBook((CardRule(1, Trigger.FANFARE, (operation,)),))
        engine = GameEngine(
            [card(index) for index in range(100, 140)],
            [card(index) for index in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=23,
            rulebook=rulebook,
        )
        engine.reset(seed=23)
        allies = [
            Unit.summon(
                card(911 + index, attack=1, life=2),
                entity_id=engine.state.allocate_entity_id(),
            )
            for index in range(2)
        ]
        engine.players[0].board = list(allies)
        engine.players[0].hand[0] = card(1, attack=2, life=3)
        engine.players[0].max_mana = engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        source = engine.players[0].board[-1]
        self.assertEqual([(ally.attack, ally.health) for ally in allies], [(2, 3), (2, 3)])
        self.assertEqual((source.attack, source.health), (2, 3))
        self.assertIsNone(engine.state.pending_choice)

    def test_random_own_unit_excludes_source_before_seeded_selection(self):
        operation = EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.RANDOM_OWN_UNIT,
            amount=1,
            exclude_source=True,
        )
        rulebook = RuleBook((CardRule(1, Trigger.FANFARE, (operation,)),))
        engine = GameEngine(
            [card(index) for index in range(100, 140)],
            [card(index) for index in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=31,
            rulebook=rulebook,
        )
        engine.reset(seed=31)
        ally = Unit.summon(
            card(915, attack=1, life=3),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board = [ally]
        engine.players[0].hand[0] = card(1, attack=2, life=3)
        engine.players[0].max_mana = engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        source = engine.players[0].board[-1]
        self.assertEqual(ally.health, 2)
        self.assertEqual(source.health, 3)

    def test_source_excluding_choice_and_rl_mask_share_option_set(self):
        operation = EffectOperation(
            EffectKind.DESTROY,
            TargetKind.ANY_BOARD,
            target_count=3,
            exclude_source=True,
        )
        rulebook = RuleBook((CardRule(1, Trigger.FANFARE, (operation,)),))
        env = ShadowverseEnv(
            [card(index) for index in range(100, 140)],
            [card(index) for index in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=29,
            rulebook=rulebook,
        )
        env.reset(seed=29)
        env.players[0].board = [
            Unit.summon(card(921), entity_id=env.core.state.allocate_entity_id()),
            Amulet(
                definition=card(922, card_type="护符", attack=None, life=None),
                entity_id=env.core.state.allocate_entity_id(),
            ),
        ]
        env.players[1].board = [
            Unit.summon(card(923), entity_id=env.core.state.allocate_entity_id()),
            Amulet(
                definition=card(924, card_type="护符", attack=None, life=None),
                entity_id=env.core.state.allocate_entity_id(),
            ),
        ]
        env.players[0].hand[0] = card(1)
        env.players[0].max_mana = env.players[0].mana = 10

        result = env.step(ShadowverseEnv.PLAY_OFFSET)

        request = env.core.state.pending_choice
        source = next(
            entity for entity in env.players[0].board
            if entity.definition.card_id == 1
        )
        self.assertNotIn(
            f"entity:{source.entity_id}",
            {option.option_id for option in request.options},
        )
        choice_mask = result.info["action_mask"][
            ShadowverseEnv.CHOICE_OFFSET:
            ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET
        ]
        self.assertEqual(choice_mask[:5], [True, True, True, True, False])

    @unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
    def test_real_lyanthoth_selects_three_other_mixed_board_cards(self):
        repo = CardRepository("data/cards.sqlite3")
        rulebook = RuleBook.from_directory("data/rules")
        engine = GameEngine(
            [card(index) for index in range(100, 140)],
            [card(index) for index in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=23,
            rulebook=rulebook,
            card_resolver=repo.get,
        )
        engine.reset(seed=23)
        own_unit = Unit.summon(
            card(911), entity_id=engine.state.allocate_entity_id()
        )
        own_amulet = Amulet(
            definition=card(912, card_type="护符", attack=None, life=None),
            entity_id=engine.state.allocate_entity_id(),
        )
        enemy_unit = Unit.summon(
            card(913), entity_id=engine.state.allocate_entity_id()
        )
        enemy_amulet = Amulet(
            definition=card(914, card_type="护符", attack=None, life=None),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board = [own_unit, own_amulet]
        engine.players[1].board = [enemy_unit, enemy_amulet]
        source_definition = repo.get(10664120)
        source = HandCard(
            source_definition,
            engine.state.allocate_entity_id(),
        )
        engine.players[0].hand = [source]
        engine.players[0].hand_entity_ids = [source.entity_id]
        engine.players[0].max_mana = engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        source_unit = next(
            entity for entity in engine.players[0].board
            if entity.definition.card_id == 10664120
        )
        request = engine.state.pending_choice
        self.assertEqual(request.target_count, 3)
        self.assertNotIn(
            f"entity:{source_unit.entity_id}",
            {option.option_id for option in request.options},
        )
        selected = request.options[:3]
        for option in selected:
            engine.apply(Choose(0, option.option_id))

        self.assertIn(source_unit, engine.players[0].board)
        self.assertIsNone(engine.state.pending_choice)
        remaining_other_ids = {
            entity.entity_id
            for player in engine.players
            for entity in player.board
            if entity.entity_id != source_unit.entity_id
        }
        self.assertEqual(len(remaining_other_ids), 1)

    def test_multi_target_choice_uses_available_count_when_targets_are_short(self):
        engine, targets = self._multi_target_engine(
            EffectOperation(
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT,
                amount=2,
                target_count=2,
            ),
            target_count=1,
        )
        self.assertEqual(engine.state.pending_choice.target_count, 1)
        engine.apply(Choose(0, f"entity:{targets[0].entity_id}"))
        self.assertEqual(targets[0].health, 3)

    def test_multi_target_forbidden_duplicate_is_illegal_without_mutation(self):
        engine, targets = self._multi_target_engine(
            EffectOperation(
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT,
                amount=1,
                target_count=2,
            ),
            target_count=2,
        )
        duplicate = Choose(0, f"entity:{targets[0].entity_id}")
        engine.apply(duplicate)
        before = engine.deterministic_fingerprint()

        with self.assertRaisesRegex(IllegalCommand, "Choice option is invalid"):
            engine.apply(duplicate)

        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_multi_target_allowed_duplicate_applies_repeatedly(self):
        engine, targets = self._multi_target_engine(
            EffectOperation(
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT,
                amount=1,
                target_count=2,
                allow_duplicate_targets=True,
            ),
            target_count=1,
        )
        choice = Choose(0, f"entity:{targets[0].entity_id}")
        engine.apply(choice)
        self.assertIn(choice, engine.legal_commands())
        engine.apply(choice)
        self.assertEqual(targets[0].health, 3)

    def test_multi_target_revalidates_selected_target_that_left_play(self):
        engine, targets = self._multi_target_engine(
            EffectOperation(
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT,
                amount=1,
                target_count=2,
            ),
            target_count=2,
        )
        engine.apply(Choose(0, f"entity:{targets[0].entity_id}"))
        engine.players[1].board.remove(targets[0])
        engine._send_to_graveyard(
            1,
            targets[0].definition,
            "test_multi_target_left_play",
            source_entity_id=targets[0].entity_id,
        )

        engine.apply(Choose(0, f"entity:{targets[1].entity_id}"))

        self.assertEqual(targets[0].health, 5)
        self.assertEqual(targets[1].health, 4)
        self.assertIsNone(engine.state.pending_choice)

    def test_multi_target_revalidates_controller_and_filter_changes(self):
        operation = EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.ENEMY_UNIT,
            amount=1,
            target_count=3,
            board_filter=BoardFilter(evolved=False),
        )
        engine, targets = self._multi_target_engine(operation, target_count=3)
        engine.apply(Choose(0, f"entity:{targets[0].entity_id}"))
        engine.apply(Choose(0, f"entity:{targets[1].entity_id}"))
        engine.players[1].board.remove(targets[0])
        engine.players[0].board.append(targets[0])
        targets[1].evolved = True

        engine.apply(Choose(0, f"entity:{targets[2].entity_id}"))

        self.assertEqual([target.health for target in targets], [5, 5, 4])

    def test_target_count_expr_resolves_when_choice_starts(self):
        engine, targets = self._multi_target_engine(
            EffectOperation(
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT,
                amount=1,
                target_count_expr=ValueExpression.constant(2),
            ),
            target_count=2,
        )
        self.assertEqual(engine.state.pending_choice.target_count, 2)
        for target in targets:
            engine.apply(Choose(0, f"entity:{target.entity_id}"))
        self.assertEqual([target.health for target in targets], [4, 4])

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

    def test_random_multi_target_is_distinct_seeded_and_caps_to_candidates(self):
        rulebook = RuleBook((
            spell_rule(
                1,
                EffectKind.DAMAGE_UNIT,
                TargetKind.RANDOM_ENEMY_UNIT,
                amount=2,
                target_count=2,
            ),
        ))
        results = []
        for _ in range(3):
            engine = GameEngine(
                [card(i) for i in range(100, 140)],
                [card(i) for i in range(200, 240)],
                class_a=1,
                class_b=1,
                seed=43,
                rulebook=rulebook,
            )
            engine.reset(seed=43)
            targets = [
                Unit.summon(
                    card(910 + index, life=5),
                    entity_id=engine.state.allocate_entity_id(),
                )
                for index in range(3)
            ]
            engine.players[1].board = list(targets)
            engine.players[0].mana = 10
            engine.players[0].hand[0] = card(
                1,
                card_type="法术",
                attack=None,
                life=None,
            )

            engine.apply(PlayCard(0, 0))

            results.append([target.health for target in targets])
            self.assertEqual(
                sorted(target.health for target in targets),
                [3, 3, 5],
            )
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])

        short_engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=43,
            rulebook=RuleBook((
                spell_rule(
                    1,
                    EffectKind.DAMAGE_UNIT,
                    TargetKind.RANDOM_ENEMY_UNIT,
                    amount=2,
                    target_count=3,
                ),
            )),
        )
        short_engine.reset(seed=43)
        only_target = Unit.summon(
            card(920, life=5),
            entity_id=short_engine.state.allocate_entity_id(),
        )
        short_engine.players[1].board = [only_target]
        short_engine.players[0].mana = 10
        short_engine.players[0].hand[0] = card(
            1,
            card_type="法术",
            attack=None,
            life=None,
        )

        short_engine.apply(PlayCard(0, 0))

        self.assertEqual(only_target.health, 3)

    def test_random_multi_target_deaths_share_one_state_based_batch(self):
        rulebook = RuleBook((
            spell_rule(
                1,
                EffectKind.DAMAGE_UNIT,
                TargetKind.RANDOM_ENEMY_UNIT,
                amount=1,
                target_count=2,
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=47,
            rulebook=rulebook,
        )
        engine.reset(seed=47)
        targets = [
            Unit.summon(
                card(930 + index, life=1),
                entity_id=engine.state.allocate_entity_id(),
            )
            for index in range(2)
        ]
        engine.players[1].board = list(targets)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(
            1,
            card_type="法术",
            attack=None,
            life=None,
        )

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(
            [[record.card_id for record in batch.records]
             for batch in engine.state.death_queue],
            [[930, 931]],
        )

    def test_random_multi_target_count_expression_can_repeat_targets(self):
        rulebook = RuleBook((
            spell_rule(
                1,
                EffectKind.DAMAGE_UNIT,
                TargetKind.RANDOM_ENEMY_UNIT,
                amount=1,
                target_count_expr=ValueExpression.constant(2),
                allow_duplicate_targets=True,
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1,
            class_b=1,
            seed=53,
            rulebook=rulebook,
        )
        engine.reset(seed=53)
        target = Unit.summon(
            card(940, life=2),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(
            1,
            card_type="法术",
            attack=None,
            life=None,
        )

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(
            [[record.card_id for record in batch.records]
             for batch in engine.state.death_queue],
            [[940]],
        )

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

    def test_random_and_all_only_operations_are_playable_without_candidates(self):
        for target in (
            TargetKind.RANDOM_ENEMY_UNIT,
            TargetKind.ALL_ENEMY_UNITS,
        ):
            with self.subTest(target=target):
                rulebook = RuleBook((
                    CardRule(
                        card_id=1,
                        trigger=Trigger.PLAY,
                        operations=(EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=target,
                            amount=2,
                        ),),
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
                engine.players[0].hand[0] = card(
                    1, card_type="法术", attack=None, life=None
                )

                command = PlayCard(0, 0)
                self.assertIn(command, engine.legal_commands())
                engine.apply(command)
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
