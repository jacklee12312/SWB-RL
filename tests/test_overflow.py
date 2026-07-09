from __future__ import annotations

import copy
import os
import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard
from swb.engine.conditions import EvalContext, evaluate_condition, evaluate_expression
from swb.engine.effects import (
    Condition,
    ConditionType,
    EffectKind,
    EffectOperation,
    ExprType,
    TargetKind,
    ValueExpression,
)
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import HandCard, PlayerState, Unit


def card(cid: int, **kw) -> CardDefinition:
    defaults = dict(
        card_id=cid,
        card_set_id=10000,
        class_id=4,
        class_name="龙族",
        name=f"c{cid}",
        cost=1,
        card_type="随从",
        attack=1,
        life=1,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )
    defaults.update(kw)
    return CardDefinition(**defaults)


def player(max_mana: int) -> PlayerState:
    return PlayerState(
        deck=[],
        class_id=4,
        class_name="龙族",
        max_mana=max_mana,
        mana=max_mana,
    )


def overflow_spell_rule(card_id: int = 1) -> CardRule:
    return CardRule(
        card_id=card_id,
        trigger=Trigger.PLAY,
        operations=(
            EffectOperation(
                kind=EffectKind.DAMAGE_UNIT,
                target=TargetKind.ENEMY_UNIT,
                amount=4,
                requires_target=True,
                conditions=(Condition(ConditionType.CONTROLLER_OVERFLOW),),
            ),
            EffectOperation(
                kind=EffectKind.DAMAGE_UNIT,
                target=TargetKind.ENEMY_UNIT,
                amount=2,
                requires_target=True,
                conditions=(
                    Condition(
                        ConditionType.NOT,
                        conditions=[Condition(ConditionType.CONTROLLER_OVERFLOW)],
                    ),
                ),
            ),
        ),
    )


def make_engine(rulebook: RuleBook) -> GameEngine:
    engine = GameEngine(
        [card(1000 + i) for i in range(40)],
        [card(2000 + i) for i in range(40)],
        class_a=4,
        class_b=4,
        seed=42,
        rulebook=rulebook,
    )
    engine.reset(seed=42)
    return engine


def insert_spell(engine: GameEngine, definition: CardDefinition) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


def choose_first(engine: GameEngine) -> None:
    choice = next(command for command in engine.legal_commands() if isinstance(command, Choose))
    engine.apply(choice)


def snapshot(engine: GameEngine):
    return (
        copy.deepcopy(engine.state),
        tuple(engine.logs),
        tuple(engine.event_history),
        tuple(engine.placeholder_ability_events),
        engine.random.getstate(),
    )


class OverflowConditionTests(unittest.TestCase):
    def test_controller_and_opponent_overflow_conditions(self):
        ctx = EvalContext(controller=0, players=[player(6), player(7)])

        self.assertFalse(
            evaluate_condition(Condition(ConditionType.CONTROLLER_OVERFLOW), ctx)
        )
        self.assertTrue(
            evaluate_condition(Condition(ConditionType.OPPONENT_OVERFLOW), ctx)
        )

        ctx.players[0].max_mana = 7
        self.assertTrue(
            evaluate_condition(Condition(ConditionType.CONTROLLER_OVERFLOW), ctx)
        )

    def test_overflow_expressions_return_boolean_numbers(self):
        ctx = EvalContext(controller=0, players=[player(7), player(6)])

        self.assertEqual(
            evaluate_expression(ValueExpression(ExprType.CONTROLLER_OVERFLOW), ctx),
            1,
        )
        self.assertEqual(
            evaluate_expression(ValueExpression(ExprType.OPPONENT_OVERFLOW), ctx),
            0,
        )

    def test_overflow_condition_and_expression_parse_from_json(self):
        from swb.engine.card_rules import _parse_operation

        op = _parse_operation(
            {
                "kind": "damage_leader",
                "target": "enemy_leader",
                "amount": {"type": "controller_overflow"},
                "conditions": [{"type": "controller_overflow"}],
            },
            "overflow.json",
            1,
        )

        self.assertEqual(op.conditions[0].type, ConditionType.CONTROLLER_OVERFLOW)
        self.assertEqual(op.amount_expr.type, ExprType.CONTROLLER_OVERFLOW)


class OverflowRuleTests(unittest.TestCase):
    def test_non_overflow_boundary_deals_base_damage(self):
        engine = make_engine(RuleBook((overflow_spell_rule(),)))
        target = Unit.summon(card(900, life=5), entity_id=engine.state.allocate_entity_id())
        engine.players[1].board = [target]
        engine.players[0].max_mana = 6
        engine.players[0].mana = 10
        insert_spell(engine, card(1, card_type="法术", attack=None, life=None))

        engine.apply(PlayCard(0, 0))
        choose_first(engine)

        self.assertEqual(target.health, 3)

    def test_overflow_boundary_deals_upgraded_damage(self):
        engine = make_engine(RuleBook((overflow_spell_rule(),)))
        target = Unit.summon(card(900, life=5), entity_id=engine.state.allocate_entity_id())
        engine.players[1].board = [target]
        engine.players[0].max_mana = 7
        engine.players[0].mana = 10
        insert_spell(engine, card(1, card_type="法术", attack=None, life=None))

        engine.apply(PlayCard(0, 0))
        choose_first(engine)

        self.assertEqual(target.health, 1)

    def test_overflow_target_required_illegal_play_does_not_mutate(self):
        engine = make_engine(RuleBook((overflow_spell_rule(),)))
        engine.players[0].max_mana = 7
        engine.players[0].mana = 10
        insert_spell(engine, card(1, card_type="法术", attack=None, life=None))
        engine._ensure_entity_ids()
        before = snapshot(engine)

        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

        self.assertEqual(snapshot(engine), before)

    def test_same_seed_overflow_resolution_is_reproducible(self):
        def run():
            engine = make_engine(RuleBook((overflow_spell_rule(),)))
            target = Unit.summon(
                card(900, life=5),
                entity_id=engine.state.allocate_entity_id(),
            )
            engine.players[1].board = [target]
            engine.players[0].max_mana = 7
            engine.players[0].mana = 10
            insert_spell(engine, card(1, card_type="法术", attack=None, life=None))
            engine.apply(PlayCard(0, 0))
            choose_first(engine)
            return target.health, tuple(engine.logs)

        self.assertEqual(run(), run())


class RealCardOverflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")
        cls.repo = CardRepository(db_path)
        cls.rulebook = RuleBook.from_directory("data/rules")

    def make_real_engine(self) -> GameEngine:
        return make_engine(self.rulebook)

    def test_dragonewt_strike_uses_base_damage_before_overflow(self):
        engine = self.make_real_engine()
        target = Unit.summon(card(900, life=5), entity_id=engine.state.allocate_entity_id())
        engine.players[1].board = [target]
        engine.players[0].max_mana = 6
        engine.players[0].mana = 10
        insert_spell(engine, self.repo.get(10041310))

        engine.apply(PlayCard(0, 0))
        choose_first(engine)

        self.assertEqual(target.health, 3)

    def test_dragonewt_strike_uses_overflow_damage_and_no_placeholder(self):
        real_card = self.repo.get(10041310)
        self.assertIn(AbilityKeyword.OVERFLOW, real_card.abilities)
        engine = self.make_real_engine()
        target = Unit.summon(card(900, life=5), entity_id=engine.state.allocate_entity_id())
        engine.players[1].board = [target]
        engine.players[0].max_mana = 7
        engine.players[0].mana = 10
        insert_spell(engine, real_card)

        engine.apply(PlayCard(0, 0))
        choose_first(engine)

        self.assertEqual(target.health, 1)
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.OVERFLOW
                for event in engine.placeholder_ability_events
            )
        )


if __name__ == "__main__":
    unittest.main()
