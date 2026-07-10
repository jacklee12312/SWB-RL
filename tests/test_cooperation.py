# -*- coding: utf-8 -*-
"""Tests for cooperation system."""
from __future__ import annotations
import sqlite3
import unittest
from pathlib import Path
from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, PlayCard, EndTurn
from swb.engine.effects import Condition, ConditionType, EffectKind, EffectOperation, ExprType, TargetKind, ValueExpression
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import AttackRestriction, DeathCause, DestroyedFollowerRecord, GraveyardCard, HandCard, Phase, Unit
from swb.engine.environment import ShadowverseEnv
from swb.engine.conditions import evaluate_condition, evaluate_expression, EvalContext

def _card(cid, **kw):
    return CardDefinition(card_id=cid, card_set_id=10000, class_id=1, class_name="精灵",
        name=kw.get("name", f"c{cid}"), cost=kw.get("cost", 1), card_type=kw.get("card_type", "随从"),
        attack=kw.get("attack", 1), life=kw.get("life", 1), keywords=frozenset(), support_level="basic", is_collectible=True)
def _resolver(defs): return lambda cid: defs.get(cid)
def _sr(cid, *ops): return CardRule(card_id=cid, trigger=Trigger.PLAY, operations=ops)
def _engine(*rules, defs=None, seed=42):
    d = dict(defs) if defs else {}
    return GameEngine(deck_a=[_card(i) for i in range(1000, 1040)], deck_b=[_card(i) for i in range(2000, 2040)],
        class_a=1, class_b=1, seed=seed, rulebook=RuleBook(tuple(rules)), card_resolver=_resolver(d))

class UnifiedCooperationTests(unittest.TestCase):
    def test_play_follower_increments(self):
        engine = _engine(); engine.reset(seed=42); engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].cooperation, 1)

    def test_reanimate_increments(self):
        f=_card(200, cost=2)
        engine = _engine(_sr(100, EffectOperation(kind=EffectKind.REANIMATE, amount=3, target=TargetKind.OWN_LEADER)), defs={100: _card(100, card_type="法术", cost=1), 200: f})
        engine.reset(seed=42); engine.state.destroyed_followers = [DestroyedFollowerRecord(definition=f, owner=0, death_sequence=1, cause=DeathCause.COMBAT)]
        engine.players[0].mana = 10
        sp = _card(100, card_type="法术", cost=1)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].cooperation, 1)

    def test_summon_from_graveyard_increments(self):
        f1 = _card(200)
        engine = _engine(_sr(100, EffectOperation(kind=EffectKind.SUMMON_FROM_GRAVEYARD, target=TargetKind.OWN_GRAVEYARD_CARD)), defs={100: _card(100, card_type="法术", cost=4), 200: f1})
        engine.reset(seed=42); engine.players[0].graveyard = [GraveyardCard(definition=f1, entity_id=20000, owner=0, entered_sequence=1, entry_cause="test")]
        engine.players[0].mana = 10
        sp = _card(100, card_type="法术", cost=4)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))
        self.assertEqual(engine.players[0].cooperation, 1)

    def test_spell_does_not_increment(self):
        engine = _engine(_sr(100, EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=0)), defs={100: _card(100, card_type="法术", cost=1)})
        engine.reset(seed=42); engine.players[0].mana = 10
        sp = _card(100, card_type="法术", cost=1)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].cooperation, 0)

    def test_effect_summon_increments(self):
        token = _card(200)
        spell = _card(100, card_type="法术", cost=1)
        engine = _engine(
            _sr(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=200,
            )),
            defs={100: spell, 200: token},
        )
        engine.reset(seed=42)
        engine.players[0].mana = 10
        engine.players[0].hand.insert(
            0, HandCard(definition=spell, entity_id=engine.state.allocate_entity_id())
        )
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].cooperation, 1)
        self.assertEqual(engine.players[0].board[0].definition.card_id, 200)

    def test_multiple_effect_summons_increment_independently(self):
        token = _card(200)
        spell = _card(100, card_type="法术", cost=1)
        summon = EffectOperation(
            kind=EffectKind.SUMMON,
            target=TargetKind.OWN_LEADER,
            card_id=200,
        )
        engine = _engine(_sr(100, summon, summon), defs={100: spell, 200: token})
        engine.reset(seed=42)
        engine.players[0].mana = 10
        engine.players[0].hand.insert(
            0, HandCard(definition=spell, entity_id=engine.state.allocate_entity_id())
        )
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].cooperation, 2)
        self.assertEqual(len(engine.players[0].board), 2)

    def test_failed_summon_on_full_board_does_not_increment(self):
        token = _card(200)
        spell = _card(100, card_type="法术", cost=1)
        engine = _engine(
            _sr(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=200,
            )),
            defs={100: spell, 200: token},
        )
        engine.reset(seed=42)
        engine.players[0].board = [
            Unit.summon(_card(300 + i), entity_id=engine.state.allocate_entity_id())
            for i in range(engine.config.max_board)
        ]
        engine.players[0].mana = 10
        engine.players[0].hand.insert(
            0, HandCard(definition=spell, entity_id=engine.state.allocate_entity_id())
        )
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].cooperation, 0)
        self.assertEqual(len(engine.players[0].board), engine.config.max_board)

    def test_transform_does_not_increment(self):
        original = _card(200)
        replacement = _card(201)
        spell = _card(100, card_type="法术", cost=1)
        engine = _engine(
            _sr(100, EffectOperation(
                kind=EffectKind.TRANSFORM,
                target=TargetKind.OWN_UNIT,
                card_id=201,
            )),
            defs={100: spell, 201: replacement},
        )
        engine.reset(seed=42)
        unit = Unit.summon(original, entity_id=engine.state.allocate_entity_id())
        engine.players[0].board.append(unit)
        engine.players[0].mana = 10
        engine.players[0].hand.insert(
            0, HandCard(definition=spell, entity_id=engine.state.allocate_entity_id())
        )
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)

        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))

        self.assertEqual(engine.players[0].cooperation, 0)
        self.assertEqual(engine.players[0].board[0].definition.card_id, 201)

    def test_choice_resume_summons_once(self):
        target = _card(200, life=5)
        token = _card(201)
        spell = _card(100, card_type="法术", cost=1)
        engine = _engine(
            _sr(
                100,
                EffectOperation(
                    kind=EffectKind.DAMAGE_UNIT,
                    target=TargetKind.ENEMY_UNIT,
                    amount=1,
                ),
                EffectOperation(
                    kind=EffectKind.SUMMON,
                    target=TargetKind.OWN_LEADER,
                    card_id=201,
                ),
            ),
            defs={100: spell, 201: token},
        )
        engine.reset(seed=42)
        engine.players[1].board.append(
            Unit.summon(target, entity_id=engine.state.allocate_entity_id())
        )
        engine.players[0].mana = 10
        engine.players[0].hand.insert(
            0, HandCard(definition=spell, entity_id=engine.state.allocate_entity_id())
        )
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].cooperation, 0)
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))

        self.assertEqual(engine.players[0].cooperation, 1)
        self.assertEqual(
            len([
                event for event in engine.event_history
                if event.type is EventType.COOPERATION_CHANGED
            ]),
            1,
        )

class CooperationEventTests(unittest.TestCase):
    def test_event_metadata(self):
        engine = _engine(); engine.reset(seed=42); engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        e = [e for e in engine.event_history if e.type is EventType.COOPERATION_CHANGED][0]
        self.assertEqual(e.metadata["cooperation_before"], 0)
        self.assertEqual(e.metadata["cooperation_after"], 1)
        self.assertEqual(e.metadata["summon_cause"], "play")

    def test_no_event_without_increment(self):
        engine = _engine(_sr(100, EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=0)), defs={100: _card(100, card_type="法术", cost=1)})
        engine.reset(seed=42); engine.players[0].mana = 10
        sp = _card(100, card_type="法术", cost=1)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len([e for e in engine.event_history if e.type is EventType.COOPERATION_CHANGED]), 0)

    def test_zero_increment_emits_no_event(self):
        engine = _engine()
        engine.reset(seed=42)

        engine._record_cooperation(0, 0)
        engine._resolve_event_queue()

        self.assertEqual(engine.players[0].cooperation, 0)
        self.assertFalse(any(
            event.type is EventType.COOPERATION_CHANGED
            for event in engine.event_history
        ))

    def test_cooperation_event_precedes_follower_summoned(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        event_types = [event.type for event in engine.event_history]
        self.assertLess(
            event_types.index(EventType.COOPERATION_CHANGED),
            event_types.index(EventType.FOLLOWER_SUMMONED),
        )

class CooperationConditionTests(unittest.TestCase):
    def test_condition_met(self):
        ctx = EvalContext(controller=0, players=[_make_player(15), _make_player(0)])
        self.assertTrue(evaluate_condition(Condition(type=ConditionType.CONTROLLER_COOPERATION_AT_LEAST, value=10), ctx))
    def test_condition_not_met(self):
        ctx = EvalContext(controller=0, players=[_make_player(5), _make_player(0)])
        self.assertFalse(evaluate_condition(Condition(type=ConditionType.CONTROLLER_COOPERATION_AT_LEAST, value=10), ctx))
    def test_opponent_condition(self):
        ctx = EvalContext(controller=0, players=[_make_player(0), _make_player(7)])
        self.assertTrue(evaluate_condition(Condition(type=ConditionType.OPPONENT_COOPERATION_AT_LEAST, value=7), ctx))
    def test_expression(self):
        ctx = EvalContext(controller=0, players=[_make_player(8), _make_player(3)])
        self.assertEqual(evaluate_expression(ValueExpression(type=ExprType.CONTROLLER_COOPERATION), ctx), 8)
        self.assertEqual(evaluate_expression(ValueExpression(type=ExprType.OPPONENT_COOPERATION), ctx), 3)
    def test_rule_condition_met_executes(self):
        engine = _engine(_sr(100, EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=3,
            conditions=(Condition(type=ConditionType.CONTROLLER_COOPERATION_AT_LEAST, value=2),))), defs={100: _card(100, card_type="法术", cost=1)})
        engine.reset(seed=42); engine.players[0].cooperation=5; engine.players[0].mana=10
        sp = _card(100, card_type="法术", cost=1)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 17)
    def test_rule_condition_not_met_skips(self):
        engine = _engine(_sr(100, EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=3,
            conditions=(Condition(type=ConditionType.CONTROLLER_COOPERATION_AT_LEAST, value=10),))), defs={100: _card(100, card_type="法术", cost=1)})
        engine.reset(seed=42); engine.players[0].cooperation=2; engine.players[0].mana=10
        sp = _card(100, card_type="法术", cost=1)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 20)

class CooperationSchemaTests(unittest.TestCase):
    def test_condition_parses(self):
        op = _parse_operation({"kind":"damage_leader","target":"enemy_leader","amount":3,"conditions":[{"type":"controller_cooperation_at_least","value":10}]},"t.json",1)
        self.assertEqual(op.conditions[0].type, ConditionType.CONTROLLER_COOPERATION_AT_LEAST)
    def test_expression_parses(self):
        op = _parse_operation({"kind":"damage_leader","target":"enemy_leader","amount":{"type":"controller_cooperation"}},"t.json",1)
        self.assertEqual(op.amount_expr.type, ExprType.CONTROLLER_COOPERATION)
    def test_condition_rejects_bool(self):
        with self.assertRaises(ValueError):
            _parse_operation({"kind":"damage_leader","target":"enemy_leader","amount":3,"conditions":[{"type":"controller_cooperation_at_least","value":True}]},"t.json",1)
    def test_condition_rejects_float(self):
        with self.assertRaises(ValueError):
            _parse_operation({"kind":"damage_leader","target":"enemy_leader","amount":3,"conditions":[{"type":"controller_cooperation_at_least","value":3.5}]},"t.json",1)

    def test_condition_rejects_negative_string_and_missing_value(self):
        invalid_values = (-1, "10")
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _parse_operation({
                    "kind": "damage_leader",
                    "target": "enemy_leader",
                    "amount": 3,
                    "conditions": [{
                        "type": "controller_cooperation_at_least",
                        "value": value,
                    }],
                }, "t.json", 1)
        with self.assertRaises(ValueError):
            _parse_operation({
                "kind": "damage_leader",
                "target": "enemy_leader",
                "amount": 3,
                "conditions": [{
                    "type": "controller_cooperation_at_least",
                }],
            }, "t.json", 1)

    def test_cooperation_leaf_expression_rejects_values(self):
        with self.assertRaises(ValueError):
            _parse_operation({
                "kind": "damage_leader",
                "target": "enemy_leader",
                "amount": {
                    "type": "controller_cooperation",
                    "values": [{"type": "constant", "value": 1}],
                },
            }, "t.json", 1)

class CooperationObservationTests(unittest.TestCase):
    def test_observation_includes_cooperation(self):
        env = ShadowverseEnv([_card(i) for i in range(1000, 1040)], [_card(i) for i in range(1100, 1140)], class_a=1, class_b=1, seed=42, rulebook=RuleBook())
        env.reset(seed=42); env.core.players[0].cooperation=7; env.core.players[1].cooperation=3
        obs = env.observation()
        self.assertEqual(len(obs), 290)
        self.assertEqual(obs[18], 0.7); self.assertEqual(obs[19], 0.3)

class CooperationDeterminismTests(unittest.TestCase):
    def test_same_seed_same_cooperation(self):
        for seed in (42, 42):
            engine = _engine(seed=seed); engine.reset(seed=seed); engine.players[0].mana = 10
            engine.apply(PlayCard(0, 0))
            self.assertEqual(engine.players[0].cooperation, 1)

class RealCardCooperationTests(unittest.TestCase):
    DATABASE = Path("data/cards.sqlite3")

    def test_database_text_matches_supported_rule(self):
        with sqlite3.connect(self.DATABASE) as connection:
            name = connection.execute(
                """
                SELECT name FROM card_names
                WHERE card_id = 10721310 AND language = 'zh-CN'
                """
            ).fetchone()[0]
            text = connection.execute(
                """
                SELECT text_chs FROM skill_texts
                WHERE card_id = 10721310 ORDER BY position
                """
            ).fetchone()[0]
        rulebook = RuleBook.from_directory("data/rules")
        operations = rulebook.operations_for(10721310, Trigger.PLAY)

        self.assertEqual(name, "敌我的调律")
        self.assertIn("造成3点伤害", text)
        self.assertIn("协作</color>_10", text)
        self.assertEqual(len(operations), 2)
        self.assertEqual(operations[0].kind, EffectKind.DAMAGE_UNIT)
        self.assertEqual(operations[0].target, TargetKind.ENEMY_UNIT)
        self.assertEqual(operations[0].target_key, "selected_follower")
        self.assertEqual(operations[1].kind, EffectKind.ADD_ATTACK_RESTRICTION)
        self.assertEqual(operations[1].target, TargetKind.PREVIOUS_TARGET)

    def test_supported_real_card_executes_threshold_effect(self):
        rulebook = RuleBook.from_directory("data/rules")
        spell = _card(10721310, card_type="法术", cost=1)
        target = _card(500, life=5)
        engine = GameEngine(
            deck_a=[_card(i) for i in range(1000, 1040)],
            deck_b=[_card(i) for i in range(2000, 2040)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=rulebook,
        )
        engine.reset(seed=42)
        enemy = Unit.summon(target, entity_id=engine.state.allocate_entity_id())
        engine.players[1].board.append(enemy)
        engine.players[0].cooperation = 10
        engine.players[0].mana = 10
        engine.players[0].hand.insert(
            0, HandCard(definition=spell, entity_id=engine.state.allocate_entity_id())
        )
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)

        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))

        self.assertEqual(enemy.health, 2)
        self.assertTrue(any(
            modifier.restriction is AttackRestriction.CANNOT_ATTACK
            for modifier in enemy.attack_restrictions
        ))

    def test_supported_real_card_keeps_base_effect_below_threshold(self):
        rulebook = RuleBook.from_directory("data/rules")
        spell = _card(10721310, card_type="法术", cost=1)
        target = _card(500, life=5)
        engine = GameEngine(
            deck_a=[_card(i) for i in range(1000, 1040)],
            deck_b=[_card(i) for i in range(2000, 2040)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=rulebook,
        )
        engine.reset(seed=42)
        enemy = Unit.summon(target, entity_id=engine.state.allocate_entity_id())
        engine.players[1].board.append(enemy)
        engine.players[0].cooperation = 9
        engine.players[0].mana = 10
        engine.players[0].hand.insert(
            0, HandCard(definition=spell, entity_id=engine.state.allocate_entity_id())
        )
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)

        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))

        self.assertEqual(enemy.health, 2)
        self.assertEqual(enemy.attack_restrictions, [])

    def test_unsupported_real_card_has_no_fabricated_rule(self):
        with sqlite3.connect(self.DATABASE) as connection:
            text = connection.execute(
                """
                SELECT text_chs FROM skill_texts
                WHERE card_id = 10724110 ORDER BY position
                """
            ).fetchone()[0]
        rulebook = RuleBook.from_directory("data/rules")

        self.assertIn("纹章", text)
        self.assertIn("本随从进化", text)
        self.assertEqual(
            rulebook.operations_for(10724110, Trigger.FANFARE),
            (),
        )

def _make_player(coop=0):
    from swb.engine.state import PlayerState
    p = PlayerState(deck=[], class_id=1, class_name="精灵"); p.cooperation = coop; return p

if __name__ == "__main__":
    unittest.main()
