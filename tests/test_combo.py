from __future__ import annotations

import copy
import os
import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, EndTurn, PlayCard
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
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import HandCard, PlayerState, Unit


def card(card_id: int, **kw) -> CardDefinition:
    defaults = dict(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="elf",
        name=f"card-{card_id}",
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


def player(combo: int) -> PlayerState:
    return PlayerState(
        deck=[],
        class_id=1,
        class_name="elf",
        cards_played_this_turn=combo,
    )


def make_engine(rulebook: RuleBook, *, seed: int = 42) -> GameEngine:
    engine = GameEngine(
        [card(1000 + i) for i in range(40)],
        [card(2000 + i) for i in range(40)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
    )
    engine.reset(seed=seed)
    return engine


def set_hand(engine: GameEngine, definitions: list[CardDefinition]) -> None:
    engine.players[0].hand = []
    engine.players[0].hand_entity_ids = []
    for definition in definitions:
        hand_card = HandCard(
            definition=definition,
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].hand.append(hand_card)
        engine.players[0].hand_entity_ids.append(hand_card.entity_id)


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


class ComboPrimitiveTests(unittest.TestCase):
    def test_controller_and_opponent_combo_conditions(self):
        ctx = EvalContext(controller=0, players=[player(2), player(3)])

        self.assertFalse(
            evaluate_condition(
                Condition(ConditionType.CONTROLLER_COMBO_AT_LEAST, value=3),
                ctx,
            )
        )
        self.assertTrue(
            evaluate_condition(
                Condition(ConditionType.OPPONENT_COMBO_AT_LEAST, value=3),
                ctx,
            )
        )

    def test_combo_expressions_return_current_counts(self):
        ctx = EvalContext(controller=0, players=[player(4), player(1)])

        self.assertEqual(
            evaluate_expression(ValueExpression(ExprType.CONTROLLER_COMBO), ctx),
            4,
        )
        self.assertEqual(
            evaluate_expression(ValueExpression(ExprType.OPPONENT_COMBO), ctx),
            1,
        )

    def test_combo_condition_expression_and_add_combo_parse_from_json(self):
        op = _parse_operation(
            {
                "kind": "add_combo",
                "target": "own_leader",
                "amount": 1,
                "conditions": [
                    {"type": "controller_combo_at_least", "value": 2}
                ],
            },
            "combo.json",
            1,
        )

        self.assertEqual(op.kind, EffectKind.ADD_COMBO)
        self.assertEqual(op.conditions[0].type, ConditionType.CONTROLLER_COMBO_AT_LEAST)

        op = _parse_operation(
            {
                "kind": "damage_leader",
                "target": "enemy_leader",
                "amount": {"type": "controller_combo"},
            },
            "combo.json",
            1,
        )
        self.assertEqual(op.amount_expr.type, ExprType.CONTROLLER_COMBO)

    def test_combo_schema_rejects_missing_threshold_and_bad_add_combo(self):
        with self.assertRaisesRegex(ValueError, "required"):
            _parse_operation(
                {
                    "kind": "draw",
                    "target": "own_leader",
                    "amount": 1,
                    "conditions": [{"type": "controller_combo_at_least"}],
                },
                "combo.json",
                1,
            )

        with self.assertRaisesRegex(ValueError, "add_combo requires"):
            _parse_operation(
                {"kind": "add_combo", "target": "own_unit", "amount": 1},
                "combo.json",
                1,
            )

        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            _parse_operation(
                {"kind": "add_combo", "target": "own_leader", "amount": -1},
                "combo.json",
                1,
            )


class ComboResolutionTests(unittest.TestCase):
    def test_cards_played_this_turn_increments_and_resets_at_end_turn(self):
        engine = make_engine(RuleBook(()))
        set_hand(engine, [card(1), card(2)])
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].cards_played_this_turn, 1)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].cards_played_this_turn, 2)

        engine.apply(EndTurn(0))
        self.assertEqual(engine.players[0].cards_played_this_turn, 0)

    def test_turn_end_combo_trigger_sees_count_before_reset(self):
        rulebook = RuleBook((
            CardRule(
                card_id=77,
                trigger=Trigger.TURN_END,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DRAW,
                        target=TargetKind.OWN_LEADER,
                        amount=1,
                        conditions=(
                            Condition(ConditionType.CONTROLLER_COMBO_AT_LEAST, value=2),
                        ),
                    ),
                ),
            ),
        ))
        engine = make_engine(rulebook)
        engine.players[0].board = [
            Unit.summon(card(77), entity_id=engine.state.allocate_entity_id())
        ]
        engine.players[0].cards_played_this_turn = 2
        before_hand = len(engine.players[0].hand)

        engine.apply(EndTurn(0))

        self.assertEqual(len(engine.players[0].hand), before_hand + 1)
        self.assertEqual(engine.players[0].cards_played_this_turn, 0)

    def test_combo_threshold_includes_currently_played_card(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rulebook = RuleBook((
            CardRule(
                card_id=10,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DAMAGE_LEADER,
                        target=TargetKind.ENEMY_LEADER,
                        amount=1,
                        conditions=(
                            Condition(ConditionType.CONTROLLER_COMBO_AT_LEAST, value=2),
                        ),
                    ),
                ),
            ),
        ))
        engine = make_engine(rulebook)
        set_hand(engine, [card(1), spell])
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 20)
        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[1].health, 19)

    def test_combo_expression_uses_count_after_play_increment(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rulebook = RuleBook((
            CardRule(
                card_id=10,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DAMAGE_LEADER,
                        target=TargetKind.ENEMY_LEADER,
                        amount=0,
                        amount_expr=ValueExpression(ExprType.CONTROLLER_COMBO),
                    ),
                ),
            ),
        ))
        engine = make_engine(rulebook)
        set_hand(engine, [spell])
        engine.players[0].cards_played_this_turn = 2
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[1].health, 17)

    def test_add_combo_effect_adds_to_natural_play_count_and_emits_event(self):
        combo_card = card(
            10,
            keywords=frozenset({AbilityKeyword.COMBO}),
        )
        rulebook = RuleBook((
            CardRule(
                card_id=10,
                trigger=Trigger.FANFARE,
                operations=(
                    EffectOperation(
                        kind=EffectKind.ADD_COMBO,
                        target=TargetKind.OWN_LEADER,
                        amount=1,
                    ),
                ),
            ),
        ))
        engine = make_engine(rulebook)
        set_hand(engine, [combo_card])
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].cards_played_this_turn, 2)
        combo_events = [
            event for event in engine.event_history
            if event.type is EventType.COMBO_CHANGED
        ]
        self.assertEqual([event.amount for event in combo_events], [1, 1])
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.COMBO
                for event in engine.placeholder_ability_events
            )
        )

    def test_illegal_target_required_combo_spell_does_not_mutate(self):
        spell = card(10, card_type="法术", attack=None, life=None)
        rulebook = RuleBook((
            CardRule(
                card_id=10,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DAMAGE_UNIT,
                        target=TargetKind.ENEMY_UNIT,
                        amount=1,
                        requires_target=True,
                    ),
                ),
            ),
        ))
        engine = make_engine(rulebook)
        set_hand(engine, [spell])
        engine.players[0].mana = 10
        engine._ensure_entity_ids()
        before = snapshot(engine)

        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

        self.assertEqual(snapshot(engine), before)

    def test_same_seed_combo_resolution_is_reproducible(self):
        def run():
            spell = card(10, card_type="法术", attack=None, life=None)
            rulebook = RuleBook((
                CardRule(
                    card_id=10,
                    trigger=Trigger.PLAY,
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_LEADER,
                            target=TargetKind.ENEMY_LEADER,
                            amount=0,
                            amount_expr=ValueExpression(ExprType.CONTROLLER_COMBO),
                        ),
                    ),
                ),
            ))
            engine = make_engine(rulebook)
            set_hand(engine, [card(1), spell])
            engine.players[0].mana = 10
            engine.apply(PlayCard(0, 0))
            engine.apply(PlayCard(0, 0))
            return engine.players[1].health, tuple(engine.logs)

        self.assertEqual(run(), run())


class RealCardComboTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")
        cls.repo = CardRepository(db_path)
        cls.rulebook = RuleBook.from_directory("data/rules")

    def make_real_engine(self) -> GameEngine:
        return make_engine(self.rulebook)

    def test_wandering_orc_adds_combo_and_no_placeholder(self):
        real_card = self.repo.get(10011120)
        self.assertIn(AbilityKeyword.COMBO, real_card.abilities)
        engine = self.make_real_engine()
        set_hand(engine, [real_card])
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].cards_played_this_turn, 2)
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.COMBO
                for event in engine.placeholder_ability_events
            )
        )

    def test_wild_pounce_damages_without_combo_draw(self):
        real_card = self.repo.get(10311310)
        target = Unit.summon(card(900, life=5), entity_id=1)
        engine = self.make_real_engine()
        target.entity_id = engine.state.allocate_entity_id()
        engine.players[1].board = [target]
        set_hand(engine, [real_card])
        engine.players[0].mana = 10
        before_deck = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))
        choose_first(engine)

        self.assertEqual(target.health, 1)
        self.assertEqual(len(engine.players[0].deck), before_deck)

    def test_wild_pounce_combo_three_draws_and_no_placeholder(self):
        real_card = self.repo.get(10311310)
        self.assertIn(AbilityKeyword.COMBO, real_card.abilities)
        target = Unit.summon(card(900, life=5), entity_id=1)
        engine = self.make_real_engine()
        target.entity_id = engine.state.allocate_entity_id()
        engine.players[1].board = [target]
        set_hand(engine, [card(1), card(2), real_card])
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply(PlayCard(0, 0))
        before_hand = len(engine.players[0].hand)
        engine.apply(PlayCard(0, 0))
        choose_first(engine)

        self.assertEqual(target.health, 1)
        self.assertEqual(len(engine.players[0].hand), before_hand)
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.COMBO
                for event in engine.placeholder_ability_events
            )
        )


if __name__ == "__main__":
    unittest.main()
