from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard
from swb.engine.effects import (
    BoardFilter, Condition, ConditionType, EffectKind, EffectOperation,
    ExprType, TargetKind, ValueExpression,
)
from swb.engine.conditions import EvalContext, evaluate_condition, evaluate_expression
from swb.engine.resolution import GameEngine
from swb.engine.state import Amulet, Unit


def card(cid, **kw):
    defaults = dict(card_id=cid, card_set_id=10000, class_id=1, class_name="elf",
                    name="c%d" % cid, cost=1, card_type="随从", attack=1, life=1,
                    keywords=frozenset(), support_level="basic", is_collectible=True)
    defaults.update(kw)
    return CardDefinition(**defaults)


class ConditionEvalTests(unittest.TestCase):
    """Unit tests for condition and expression evaluators."""

    def test_always(self):
        self.assertTrue(evaluate_condition(Condition(type=ConditionType.ALWAYS), None))

    def test_controller_health_at_most(self):
        ctx = EvalContext(0, [_player(health=10), _player(health=20)])
        self.assertTrue(evaluate_condition(Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 10), ctx))
        self.assertFalse(evaluate_condition(Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 9), ctx))

    def test_all_conditions(self):
        ctx = EvalContext(0, [_player(health=10), _player(health=20)])
        cond = Condition(ConditionType.ALL, conditions=[
            Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 10),
            Condition(ConditionType.OPPONENT_HEALTH_AT_LEAST, 20),
        ])
        self.assertTrue(evaluate_condition(cond, ctx))

    def test_any_conditions(self):
        ctx = EvalContext(0, [_player(health=10), _player(health=20)])
        cond = Condition(ConditionType.ANY, conditions=[
            Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 5),
            Condition(ConditionType.OPPONENT_HEALTH_AT_LEAST, 20),
        ])
        self.assertTrue(evaluate_condition(cond, ctx))

    def test_not_condition(self):
        ctx = EvalContext(0, [_player(health=10), _player(health=20)])
        cond = Condition(ConditionType.NOT, conditions=[
            Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 5),
        ])
        self.assertTrue(evaluate_condition(cond, ctx))

    def test_constant_expression(self):
        self.assertEqual(evaluate_expression(ValueExpression.constant(5), None), 5)

    def test_add_expression(self):
        expr = ValueExpression(ExprType.ADD, values=[
            ValueExpression.constant(3), ValueExpression.constant(7),
        ])
        self.assertEqual(evaluate_expression(expr, None), 10)

    def test_subtract_clamped(self):
        expr = ValueExpression(ExprType.SUBTRACT, values=[
            ValueExpression.constant(3), ValueExpression.constant(7),
        ])
        self.assertEqual(evaluate_expression(expr, None), 0)

    def test_controller_board_count(self):
        ctx = EvalContext(0, [_player(board_count=3), _player(board_count=1)])
        self.assertEqual(evaluate_expression(ValueExpression(ExprType.CONTROLLER_BOARD_COUNT), ctx), 3)

    def test_opponent_board_count(self):
        ctx = EvalContext(0, [_player(board_count=3), _player(board_count=1)])
        self.assertEqual(evaluate_expression(ValueExpression(ExprType.OPPONENT_BOARD_COUNT), ctx), 1)

    def test_board_count_expression_applies_entity_filter(self):
        own = _player()
        follower = Unit.summon(card(10), entity_id=10)
        evolved = Unit.summon(card(11), entity_id=11)
        evolved.evolved = True
        amulet = Amulet(
            definition=card(
                12,
                card_type="护符",
                attack=None,
                life=None,
            ),
            entity_id=12,
        )
        own.board = [follower, evolved, amulet]
        ctx = EvalContext(0, [own, _player()])

        followers = ValueExpression(
            ExprType.CONTROLLER_BOARD_COUNT,
            board_filter=BoardFilter(card_type="随从"),
        )
        unevolved = ValueExpression(
            ExprType.CONTROLLER_BOARD_COUNT,
            board_filter=BoardFilter(card_type="随从", evolved=False),
        )

        self.assertEqual(evaluate_expression(followers, ctx), 2)
        self.assertEqual(evaluate_expression(unevolved, ctx), 1)

    def test_board_has_with_filter(self):
        own = _player()
        other = Unit.summon(card(10, cost=2, name="other"), entity_id=10)
        evolved = Unit.summon(card(11, cost=5, name="target"), entity_id=11)
        evolved.evolved = True
        own.board = [other, evolved]
        ctx = EvalContext(0, [own, _player()])

        self.assertTrue(evaluate_condition(
            Condition(
                ConditionType.CONTROLLER_BOARD_HAS,
                board_filter=BoardFilter(card_type="随从", cost_min=5, evolved=True),
            ),
            ctx,
        ))
        self.assertFalse(evaluate_condition(
            Condition(
                ConditionType.OPPONENT_BOARD_HAS,
                board_filter=BoardFilter(card_name="target"),
            ),
            ctx,
        ))

    def test_board_filter_distinguishes_evolved_super_evolved_and_damaged(self):
        normal = Unit.summon(card(20, life=5), entity_id=20)
        evolved = Unit.summon(card(21, life=5), entity_id=21)
        evolved.evolved = True
        super_evolved = Unit.summon(card(22, life=5), entity_id=22)
        super_evolved.evolved = True
        super_evolved.super_evolved = True
        damaged = Unit.summon(card(23, life=5), entity_id=23)
        damaged.health = 3

        self.assertFalse(BoardFilter(super_evolved=True).matches_entity(evolved))
        self.assertTrue(
            BoardFilter(evolved=True, super_evolved=True).matches_entity(
                super_evolved
            )
        )
        self.assertTrue(BoardFilter(damaged=True).matches_entity(damaged))
        self.assertFalse(BoardFilter(damaged=True).matches_entity(normal))
        self.assertTrue(BoardFilter(damaged=False).matches_entity(normal))
        self.assertFalse(
            BoardFilter(damaged=True).matches_entity(
                Amulet(definition=card(24, card_type="护符", attack=None, life=None), entity_id=24)
            )
        )

    def test_controller_deck_has_no_duplicates(self):
        own = _player()
        own.deck = [card(10), card(11), card(12)]
        ctx = EvalContext(0, [own, _player()])

        self.assertTrue(evaluate_condition(
            Condition(ConditionType.CONTROLLER_DECK_HAS_NO_DUPLICATES),
            ctx,
        ))
        own.deck.append(card(11))
        self.assertFalse(evaluate_condition(
            Condition(ConditionType.CONTROLLER_DECK_HAS_NO_DUPLICATES),
            ctx,
        ))


def _player(health=20, board_count=0):
    from swb.engine.state import PlayerState
    return PlayerState(
        deck=[], class_id=1, class_name="elf", health=health,
        board=[_dummy_unit() for _ in range(board_count)],
    )


def _dummy_unit():
    return Unit.summon(card(1), entity_id=1)


class BackwardCompatTests(unittest.TestCase):
    """Old JSON rules must still work."""

    def test_old_rulebook_loads(self):
        rulebook = RuleBook.from_directory("data/rules")
        ops = rulebook.operations_for(10041130, Trigger.FANFARE)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].amount, 6)

    def test_old_spell_still_works(self):
        rulebook = RuleBook.from_directory("data/rules")
        spell = card(10041310, attack=None, life=None, card_type="法术")
        engine = GameEngine(
            [spell] * 40, [card(2)] * 40,
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(99, attack=1, life=2), entity_id=engine.state.allocate_entity_id())
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = spell
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(target.health, 0)


class ConditionOperationTests(unittest.TestCase):
    """Conditional effects via the engine."""

    def test_condition_met_executes(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1,
                    conditions=(Condition(ConditionType.ALWAYS),),
                ),
            ),),
        ))
        engine = GameEngine([card(i) for i in range(100,140)], [card(i) for i in range(200,240)], class_a=1, class_b=1, seed=1, rulebook=rulebook)
        engine.reset(seed=1)
        hb = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].hand), hb)

    def test_condition_not_met_skips(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1,
                    conditions=(Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 5),),
                ),
            ),),
        ))
        engine = GameEngine([card(i) for i in range(100,140)], [card(i) for i in range(200,240)], class_a=1, class_b=1, seed=1, rulebook=rulebook)
        engine.reset(seed=1)
        hb = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].hand), hb - 1)

    def test_dynamic_amount_expression(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER,
                    amount=0,
                    amount_expr=ValueExpression(ExprType.ADD, values=[
                        ValueExpression.constant(1),
                        ValueExpression(ExprType.CONTROLLER_BOARD_COUNT),
                    ]),
                ),
            ),),
        ))
        engine = GameEngine([card(i) for i in range(100,140)], [card(i) for i in range(200,240)], class_a=1, class_b=1, seed=1, rulebook=rulebook)
        engine.reset(seed=1)
        engine.players[0].board = [Unit.summon(card(900), entity_id=engine.state.allocate_entity_id())]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 18)

    def test_state_change_visible_to_next_condition(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.OWN_LEADER, amount=5),
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1,
                    conditions=(Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 15),),
                ),
            ),),
        ))
        engine = GameEngine([card(i) for i in range(100,140)], [card(i) for i in range(200,240)], class_a=1, class_b=1, seed=1, rulebook=rulebook)
        engine.reset(seed=1)
        hb = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].hand), hb)

    def test_board_has_condition_executes_extra_operation(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.DRAW,
                    target=TargetKind.OWN_LEADER,
                    amount=1,
                    conditions=(
                        Condition(
                            ConditionType.CONTROLLER_BOARD_HAS,
                            board_filter=BoardFilter(cost_min=5),
                        ),
                    ),
                ),
            ),),
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
        engine.players[0].board = [
            Unit.summon(card(900, cost=5), entity_id=engine.state.allocate_entity_id())
        ]
        hb = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(len(engine.players[0].hand), hb)

    def test_illegal_expression_raises(self):
        from swb.engine.card_rules import _parse_expression
        with self.assertRaises((ValueError, KeyError)):
            _parse_expression({"type": "no_such_type"}, "test.json", 1)

    def test_mixed_all_condition_does_not_block_play_or_consume_rng(self):
        condition = Condition(ConditionType.ALL, conditions=[
            Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 5),
            Condition(ConditionType.TARGET_HEALTH_AT_MOST, 3),
        ])
        operation = EffectOperation(
            kind=EffectKind.DAMAGE_UNIT,
            target=TargetKind.RANDOM_ENEMY_UNIT,
            amount=2,
            conditions=(condition,),
        )
        engine = _engine_with_spell(operation)
        engine.players[1].board = [
            Unit.summon(
                card(900, life=2),
                entity_id=engine.state.allocate_entity_id(),
            )
        ]
        before_rng = engine.random.getstate()

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[1].board[0].health, 2)
        self.assertEqual(engine.random.getstate(), before_rng)

    def test_mixed_any_filters_manual_and_random_targets(self):
        condition = Condition(ConditionType.ANY, conditions=[
            Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 5),
            Condition(ConditionType.TARGET_HEALTH_AT_MOST, 3),
        ])
        manual = EffectOperation(
            kind=EffectKind.DAMAGE_UNIT,
            target=TargetKind.ENEMY_UNIT,
            amount=2,
            conditions=(condition,),
        )
        engine = _engine_with_spell(manual)
        valid = Unit.summon(
            card(900, life=2),
            entity_id=engine.state.allocate_entity_id(),
        )
        invalid = Unit.summon(
            card(901, life=5),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board = [valid, invalid]

        engine.apply(PlayCard(0, 0))

        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [valid.entity_id],
        )

        random_operation = EffectOperation(
            kind=EffectKind.DAMAGE_UNIT,
            target=TargetKind.RANDOM_ENEMY_UNIT,
            amount=2,
            conditions=(condition,),
        )
        for seed in range(1, 6):
            engine = _engine_with_spell(random_operation, seed=seed)
            invalid = Unit.summon(
                card(901, life=5),
                entity_id=engine.state.allocate_entity_id(),
            )
            valid = Unit.summon(
                card(900, life=2),
                entity_id=engine.state.allocate_entity_id(),
            )
            engine.players[1].board = [invalid, valid]
            engine.apply(PlayCard(0, 0))
            self.assertEqual(invalid.health, 5)
            self.assertEqual(valid.health, 0)

    def test_mixed_not_condition_keeps_valid_target_playable(self):
        condition = Condition(ConditionType.NOT, conditions=[
            Condition(ConditionType.ANY, conditions=[
                Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 5),
                Condition(ConditionType.TARGET_HEALTH_AT_MOST, 3),
            ])
        ])
        operation = EffectOperation(
            kind=EffectKind.DAMAGE_UNIT,
            target=TargetKind.ENEMY_UNIT,
            amount=2,
            conditions=(condition,),
        )
        engine = _engine_with_spell(operation)
        target = Unit.summon(
            card(900, life=5),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board = [target]

        playable = [
            command
            for command in engine.legal_commands()
            if isinstance(command, PlayCard) and command.hand_index == 0
        ]

        self.assertEqual(len(playable), 1)

    def test_mixed_conditions_filter_all_targets(self):
        non_target = Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 5)
        target = Condition(ConditionType.TARGET_HEALTH_AT_MOST, 3)

        any_operation = EffectOperation(
            kind=EffectKind.DAMAGE_UNIT,
            target=TargetKind.ALL_ENEMY_UNITS,
            amount=2,
            conditions=(
                Condition(
                    ConditionType.ANY,
                    conditions=[non_target, target],
                ),
            ),
        )
        engine = _engine_with_spell(any_operation)
        low = Unit.summon(
            card(900, life=2),
            entity_id=engine.state.allocate_entity_id(),
        )
        high = Unit.summon(
            card(901, life=5),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board = [low, high]

        engine.apply(PlayCard(0, 0))

        self.assertEqual(low.health, 0)
        self.assertEqual(high.health, 5)

    def test_schema_errors_include_card_id(self):
        from swb.engine.card_rules import _parse_operation

        invalid_rules = [
            {
                "kind": "draw",
                "target": "own_leader",
                "conditions": [{"type": "all", "conditions": []}],
                "amount": 1,
            },
            {
                "kind": "draw",
                "target": "own_leader",
                "amount": {"type": "multiply", "values": []},
            },
            {
                "kind": "draw",
                "target": "own_leader",
                "amount": {
                    "type": "source_attack",
                    "values": [{"type": "constant", "value": 1}],
                },
            },
        ]
        for raw in invalid_rules:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "card 77"):
                    _parse_operation(raw, "bad.json", 77)

    def test_parse_board_has_condition(self):
        from swb.engine.card_rules import _parse_condition

        cond = _parse_condition(
            {
                "type": "controller_board_has",
                "value": 2,
                "card_type_filter": "随从",
                "cost_min": 5,
                "evolved_filter": True,
                "super_evolved_filter": True,
                "damaged_filter": False,
            },
            "test.json",
            77,
        )

        self.assertEqual(cond.type, ConditionType.CONTROLLER_BOARD_HAS)
        self.assertEqual(cond.value, 2)
        self.assertEqual(cond.board_filter.card_type, "随从")
        self.assertEqual(cond.board_filter.cost_min, 5)
        self.assertTrue(cond.board_filter.evolved)
        self.assertTrue(cond.board_filter.super_evolved)
        self.assertFalse(cond.board_filter.damaged)

    def test_board_state_filter_schema_rejects_bad_booleans_and_conflicts(self):
        from swb.engine.card_rules import _parse_condition, _parse_operation

        invalid_conditions = (
            {"type": "controller_board_has", "super_evolved_filter": 1},
            {"type": "controller_board_has", "damaged_filter": "yes"},
            {
                "type": "controller_board_has",
                "evolved_filter": False,
                "super_evolved_filter": True,
            },
        )
        for raw in invalid_conditions:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    _parse_condition(raw, "bad.json", 77)

        with self.assertRaisesRegex(ValueError, "board target"):
            _parse_operation(
                {
                    "kind": "draw",
                    "target": "own_leader",
                    "amount": 1,
                    "target_super_evolved_filter": True,
                },
                "bad.json",
                77,
            )


def _engine_with_spell(
    operation: EffectOperation,
    *,
    seed: int = 1,
) -> GameEngine:
    rulebook = RuleBook((
        CardRule(card_id=1, trigger=Trigger.PLAY, operations=(operation,)),
    ))
    engine = GameEngine(
        [card(i) for i in range(100, 140)],
        [card(i) for i in range(200, 240)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
    )
    engine.reset(seed=seed)
    engine.players[0].mana = 10
    engine.players[0].hand[0] = card(
        1,
        card_type="法术",
        attack=None,
        life=None,
    )
    return engine


if __name__ == "__main__":
    unittest.main()
