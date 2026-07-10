# -*- coding: utf-8 -*-
"""Advanced emblem tests: scope, activation limits, on_expire, continuation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import (
    Condition,
    ConditionType,
    EffectKind,
    EffectOperation,
    TargetKind,
)
from swb.engine.emblem import (
    EmblemDefinition,
    EmblemStacking,
    EmblemTriggerRule,
    EventScope,
    TurnScope,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType, GameEvent
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import (
    EmblemInstance,
    HandCard,
    Unit,
)


def _card(cid, **kw):
    return CardDefinition(
        card_id=cid, card_set_id=10000, class_id=1, class_name="\u7cbe\u7075",
        name=kw.get("name", f"c{cid}"), cost=kw.get("cost", 1),
        card_type=kw.get("card_type", "\u968f\u4ece"),
        attack=kw.get("attack", 1), life=kw.get("life", 1),
        keywords=frozenset(), support_level="basic", is_collectible=True,
    )


def _fr(cid, *ops):
    return CardRule(card_id=cid, trigger=Trigger.FANFARE, operations=ops)


def _engine(*rules, seed=42):
    return GameEngine(
        deck_a=[_card(i) for i in range(1000, 1040)],
        deck_b=[_card(i) for i in range(2000, 2040)],
        class_a=1, class_b=1, seed=seed,
        rulebook=RuleBook(tuple(rules)),
    )


def _insert_card(engine, card_def, origin=CardOrigin.DECK):
    hc = HandCard(definition=card_def, entity_id=engine.state.allocate_entity_id(), origin=origin)
    engine.players[0].hand.insert(0, hc)
    engine.players[0].hand_entity_ids.insert(0, hc.entity_id)
    return hc


def _advance_two(engine):
    engine.apply(EndTurn(engine.current_player))
    engine.apply(EndTurn(engine.current_player))


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


# ---------------------------------------------------------------------------
class TurnScopeTests(unittest.TestCase):
    def test_owner_turn_scope_fires_on_controller(self):
        ed = EmblemDefinition("ot", 999910, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OWNER_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(_fr(999910, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="ot")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"ot": ed}
        _insert_card(engine, _card(999910, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        _advance_two(engine)
        self.assertLess(engine.players[1].health, 20)

    def test_owner_turn_scope_not_on_opponent(self):
        ed = EmblemDefinition("ot2", 999910, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OWNER_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(_fr(999910, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="ot2")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"ot2": ed}
        _insert_card(engine, _card(999910, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        opp = engine.players[1].health
        engine.apply(EndTurn(engine.current_player))
        self.assertEqual(engine.players[1].health, opp)

    def test_opponent_turn_scope_fires(self):
        ed = EmblemDefinition("opp", 999911, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OPPONENT_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(_fr(999911, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="opp")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"opp": ed}
        _insert_card(engine, _card(999911, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        engine.apply(EndTurn(engine.current_player))
        self.assertLess(engine.players[1].health, 20)

    def test_any_turn_default_works(self):
        ed = EmblemDefinition("any", 999910, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.ANY_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(_fr(999910, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="any")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"any": ed}
        _insert_card(engine, _card(999910, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        _advance_two(engine)
        self.assertLess(engine.players[1].health, 20)


# ---------------------------------------------------------------------------
class OncePerTurnTests(unittest.TestCase):
    def test_once_per_turn_fires_once(self):
        ed = EmblemDefinition("opt1", 999912, triggers=(EmblemTriggerRule("follower_summoned", once_per_turn=True, operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),)),))
        engine = _engine(_fr(999912, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="opt1")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"opt1": ed}
        _insert_card(engine, _card(999912, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        deck_before = len(engine.players[0].deck)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 1))
        drawn_once = deck_before - len(engine.players[0].deck)
        deck_after_two = len(engine.players[0].deck)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 2))
        self.assertLess(len(engine.players[0].deck), deck_before)
        self.assertEqual(len(engine.players[0].deck), deck_after_two)

    def test_once_per_turn_resets(self):
        ed = EmblemDefinition("opt2", 999912, triggers=(EmblemTriggerRule("follower_summoned", once_per_turn=True, operations=(EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),)),))
        engine = _engine(_fr(999912, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="opt2")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"opt2": ed}
        _insert_card(engine, _card(999912, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 1))
        _advance_two(engine)
        deck_before = len(engine.players[0].deck)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertLess(len(engine.players[0].deck), deck_before)


# ---------------------------------------------------------------------------
class MaxActivationsTests(unittest.TestCase):
    def test_max_activations_caps(self):
        ed = EmblemDefinition("max", 999913, triggers=(EmblemTriggerRule("card_played", max_activations=2, event_scope=EventScope.OWNER_EVENT, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(_fr(999913, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="max")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"max": ed}
        _insert_card(engine, _card(999913, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 20)
        _advance_two(engine)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 19)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 18)
        _advance_two(engine)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 18)

    def test_condition_false_does_not_consume(self):
        ed = EmblemDefinition(
            "maxc",
            999913,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    max_activations=1,
                    conditions=(
                        Condition(ConditionType.CONTROLLER_HEALTH_AT_MOST, 5),
                    ),
                    operations=(
                        EffectOperation(
                            EffectKind.DRAW,
                            TargetKind.OWN_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        instance = _add_emblem(engine, 0, ed)
        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=100))
        engine._resolve_event_queue()
        self.assertEqual(instance.activation_counts, {})
        engine.players[0].health = 5
        deck_before = len(engine.players[0].deck)
        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=101))
        engine._resolve_event_queue()
        self.assertLess(len(engine.players[0].deck), deck_before)
        self.assertEqual(instance.activation_counts[0], 1)

    def test_no_legal_target_does_not_consume(self):
        ed = EmblemDefinition(
            "max_target",
            999913,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    max_activations=1,
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_UNIT,
                            TargetKind.RANDOM_ENEMY_UNIT,
                            2,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        instance = _add_emblem(engine, 0, ed)
        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=100))
        engine._resolve_event_queue()
        self.assertEqual(instance.activation_counts, {})

        target = Unit.summon(
            _card(999950, life=3),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board.append(target)
        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=101))
        engine._resolve_event_queue()
        self.assertEqual(target.health, 1)
        self.assertEqual(instance.activation_counts[0], 1)


class EventScopeTests(unittest.TestCase):
    def test_event_source_is_available_as_self_target(self):
        ed = EmblemDefinition(
            "buff_summoned",
            999921,
            triggers=(
                EmblemTriggerRule(
                    "follower_summoned",
                    operations=(
                        EffectOperation(
                            EffectKind.BUFF_UNIT,
                            TargetKind.SELF,
                            2,
                            0,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        _add_emblem(engine, 0, ed)
        unit = Unit.summon(
            _card(999953, attack=1, life=3),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board.append(unit)
        engine._emit(
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                0,
                source_id=unit.entity_id,
                metadata={"source": unit},
            )
        )
        engine._resolve_event_queue()
        self.assertEqual(unit.attack, 3)

    def test_legacy_event_scope_defaults_to_owner_event(self):
        ed = EmblemDefinition(
            "legacy_event",
            999914,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        _add_emblem(engine, 0, ed)
        engine._emit(GameEvent(EventType.CARD_PLAYED, 1, source_id=100))
        engine._resolve_event_queue()
        self.assertEqual(engine.players[1].health, 20)
        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=101))
        engine._resolve_event_queue()
        self.assertEqual(engine.players[1].health, 19)

    def test_owner_event_does_not_fire_for_opponent(self):
        ed = EmblemDefinition(
            "owner_event",
            999915,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    event_scope=EventScope.OWNER_EVENT,
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        _add_emblem(engine, 0, ed)
        engine._emit(GameEvent(EventType.CARD_PLAYED, 1, source_id=100))
        engine._resolve_event_queue()
        self.assertEqual(engine.players[1].health, 20)

    def test_opponent_event_fires_for_opponent_only(self):
        ed = EmblemDefinition(
            "opponent_event",
            999916,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    event_scope=EventScope.OPPONENT_EVENT,
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        _add_emblem(engine, 0, ed)
        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=100))
        engine._resolve_event_queue()
        self.assertEqual(engine.players[1].health, 20)
        engine._emit(GameEvent(EventType.CARD_PLAYED, 1, source_id=101))
        engine._resolve_event_queue()
        self.assertEqual(engine.players[1].health, 19)

    def test_once_per_turn_resets_at_each_turn_boundary(self):
        ed = EmblemDefinition(
            "every_turn",
            999917,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    once_per_turn=True,
                    event_scope=EventScope.ANY_EVENT,
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        _add_emblem(engine, 0, ed)
        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=100))
        engine._resolve_event_queue()
        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=101))
        engine._resolve_event_queue()
        self.assertEqual(engine.players[1].health, 19)
        engine.apply(EndTurn(0))
        engine._emit(GameEvent(EventType.CARD_PLAYED, 1, source_id=102))
        engine._resolve_event_queue()
        self.assertEqual(engine.players[1].health, 18)


# ---------------------------------------------------------------------------
class OnExpireTests(unittest.TestCase):
    def test_on_expire_executes(self):
        ed = EmblemDefinition("oe1", 999914, countdown=2, triggers=(), on_expire=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 3),))
        engine = _engine(_fr(999914, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="oe1")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"oe1": ed}
        _insert_card(engine, _card(999914, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].emblems), 1)
        _advance_two(engine)
        _advance_two(engine)
        self.assertEqual(len(engine.players[0].emblems), 0)
        self.assertEqual(engine.players[1].health, 17)

    def test_on_expire_removes_only_once(self):
        ed = EmblemDefinition("oe3", 999914, countdown=2, on_expire=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),))
        engine = _engine(_fr(999914, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="oe3")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"oe3": ed}
        _insert_card(engine, _card(999914, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        _advance_two(engine)
        _advance_two(engine)
        opp_after = engine.players[1].health
        _advance_two(engine)
        self.assertEqual(engine.players[1].health, opp_after)

    def test_multiple_choice_expirations_resume_in_order(self):
        choose_expire = EmblemDefinition(
            "choose_expire",
            999918,
            countdown=1,
            on_expire=(
                EffectOperation(
                    EffectKind.DAMAGE_UNIT,
                    TargetKind.ENEMY_UNIT,
                    1,
                ),
            ),
        )
        leader_expire = EmblemDefinition(
            "leader_expire",
            999919,
            countdown=1,
            on_expire=(
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    2,
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        first = _add_emblem(engine, 0, choose_expire)
        second = _add_emblem(engine, 0, leader_expire)
        target = Unit.summon(
            _card(999951, life=3),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board.append(target)

        engine._tick_emblem_countdowns(0)
        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        self.assertIn(first, engine.players[0].emblems)
        self.assertIn(second, engine.players[0].emblems)
        self.assertEqual(engine.players[1].health, 20)

        engine.apply(Choose(request.player_index, request.options[0].option_id))
        self.assertEqual(target.health, 2)
        self.assertEqual(engine.players[1].health, 18)
        self.assertEqual(engine.players[0].emblems, [])

        event_types = [
            event.type
            for event in engine.event_history
            if event.type in {EventType.EMBLEM_EXPIRED, EventType.EMBLEM_REMOVED}
        ]
        self.assertEqual(
            event_types[-4:],
            [
                EventType.EMBLEM_EXPIRED,
                EventType.EMBLEM_REMOVED,
                EventType.EMBLEM_EXPIRED,
                EventType.EMBLEM_REMOVED,
            ],
        )

    def test_choice_expiration_pauses_and_resumes_turn_start(self):
        definition = EmblemDefinition(
            "turn_start_expire",
            999920,
            countdown=1,
            on_expire=(
                EffectOperation(
                    EffectKind.DAMAGE_UNIT,
                    TargetKind.ENEMY_UNIT,
                    1,
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        _add_emblem(engine, 1, definition)
        target = Unit.summon(
            _card(999952, life=3),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board.append(target)
        deck_before = len(engine.players[1].deck)

        engine.apply(EndTurn(0))
        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        self.assertEqual(engine.current_player, 1)
        self.assertEqual(len(engine.players[1].deck), deck_before)

        engine.apply(Choose(request.player_index, request.options[0].option_id))
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(target.health, 2)
        self.assertEqual(engine.players[1].emblems, [])
        self.assertEqual(len(engine.players[1].deck), deck_before - 1)


# ---------------------------------------------------------------------------
class SchemaValidationTests(unittest.TestCase):
    def test_unknown_emblem_reference_in_on_expire_rejected(self):
        payload = {
            "emblems": [
                {
                    "id": "bad_expire_ref",
                    "source_card_id": 999970,
                    "countdown": 1,
                    "on_expire": [
                        {
                            "kind": "gain_emblem",
                            "target": "own_leader",
                            "emblem_id": "missing",
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, "unknown emblem_id"):
                RuleBook.from_directory(directory)

    def test_unbound_previous_target_in_on_expire_rejected(self):
        from swb.engine.card_rules import _parse_emblem_definition, _parse_operation

        with self.assertRaisesRegex(ValueError, "target_key"):
            _parse_emblem_definition(
                {
                    "id": "bad_previous",
                    "source_card_id": 999971,
                    "countdown": 1,
                    "on_expire": [
                        {
                            "kind": "damage_unit",
                            "target": "previous_target",
                            "target_key": "missing",
                            "amount": 1,
                        }
                    ],
                },
                "test.json",
                _parse_operation,
            )

    def test_unknown_turn_scope_rejected(self):
        with self.assertRaises(ValueError):
            TurnScope("bad")

    def test_unknown_event_scope_rejected(self):
        with self.assertRaises(ValueError):
            EventScope("bad")

    def test_bool_on_expire_rejected(self):
        from swb.engine.card_rules import _parse_emblem_definition, _parse_operation
        with self.assertRaises(ValueError):
            _parse_emblem_definition(
                {"id": "bad", "source_card_id": 1, "on_expire": True},
                "test.json", _parse_operation,
            )

    def test_string_max_activations_rejected(self):
        from swb.engine.card_rules import _parse_emblem_definition, _parse_operation
        with self.assertRaises(ValueError):
            _parse_emblem_definition(
                {"id": "bad", "source_card_id": 1, "triggers": [
                    {"trigger": "turn_start", "max_activations": "abc"}
                ]},
                "test.json", _parse_operation,
            )

    def test_bool_once_per_turn_rejected(self):
        from swb.engine.card_rules import _parse_emblem_definition, _parse_operation
        with self.assertRaises(ValueError):
            _parse_emblem_definition(
                {"id": "bad", "source_card_id": 1, "triggers": [
                    {"trigger": "turn_start", "once_per_turn": "yes"}
                ]},
                "test.json", _parse_operation,
            )

    def test_old_rules_still_load(self):
        rb = RuleBook.from_directory("data/rules")
        self.assertIsInstance(rb, RuleBook)

    def test_advanced_demo_loads(self):
        rb = RuleBook.from_directory("data/rules")
        ed = rb.emblem_def("scope_turn_emblem")
        self.assertIsNotNone(ed)
        self.assertEqual(ed.triggers[0].turn_scope, TurnScope.OWNER_TURN)


# ---------------------------------------------------------------------------
class ContinuationTests(unittest.TestCase):
    def test_nested_event_cannot_exceed_activation_limit(self):
        summoner = EmblemDefinition(
            "nested_summoner",
            999922,
            triggers=(
                EmblemTriggerRule(
                    "follower_summoned",
                    once_per_turn=True,
                    operations=(
                        EffectOperation(
                            EffectKind.SUMMON,
                            TargetKind.OWN_LEADER,
                            card_id=999960,
                        ),
                    ),
                ),
            ),
        )
        limited = EmblemDefinition(
            "nested_limited",
            999923,
            triggers=(
                EmblemTriggerRule(
                    "follower_summoned",
                    max_activations=1,
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        engine.card_resolver = lambda card_id: (
            _card(card_id) if card_id == 999960 else None
        )
        _add_emblem(engine, 0, summoner)
        limited_instance = _add_emblem(engine, 0, limited)
        source = Unit.summon(
            _card(999959),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board.append(source)

        engine._emit(
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                0,
                source_id=source.entity_id,
                metadata={"source": source},
            )
        )
        engine._resolve_event_queue()
        self.assertEqual(engine.players[1].health, 19)
        self.assertEqual(limited_instance.activation_counts[0], 1)

    def test_game_end_stops_remaining_emblem_batch(self):
        lethal = EmblemDefinition(
            "lethal",
            999924,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
        )
        after = EmblemDefinition(
            "after_lethal",
            999925,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.OWN_LEADER,
                            5,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        _add_emblem(engine, 0, lethal)
        _add_emblem(engine, 0, after)
        engine.players[1].health = 1
        engine._emit(GameEvent(EventType.CARD_PLAYED, 0, source_id=100))
        engine._resolve_event_queue()
        engine._stabilize()
        self.assertTrue(engine.terminated)
        self.assertEqual(engine.players[0].health, 20)

    def test_new_emblem_not_in_current_batch(self):
        ed1 = EmblemDefinition("batch1", 999901, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OWNER_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(_fr(999901, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="batch1")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"batch1": ed1}
        _insert_card(engine, _card(999901, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        _advance_two(engine)
        self.assertEqual(engine.players[1].health, 19)

    def test_emblems_sorted_by_creation_order(self):
        ed1 = EmblemDefinition("co1", 999901, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OWNER_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        ed2 = EmblemDefinition("co2", 999902, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OWNER_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(
            _fr(999901, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="co1")),
            _fr(999902, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="co2")),
        )
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"co1": ed1, "co2": ed2}
        _insert_card(engine, _card(999901, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        _insert_card(engine, _card(999902, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        _advance_two(engine)
        self.assertEqual(engine.players[1].health, 18)


# ---------------------------------------------------------------------------
class DeterminismTests(unittest.TestCase):
    def test_seeded_determinism(self):
        for _ in range(2):
            ed = EmblemDefinition("det", 999910, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OWNER_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
            engine = _engine(_fr(999910, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="det")))
            engine.reset(seed=42)
            engine.rulebook._emblem_defs = {"det": ed}
            _insert_card(engine, _card(999910, cost=1))
            engine.players[0].mana = 10
            engine.apply(PlayCard(0, 0))
            _advance_two(engine)
            self.assertEqual(engine.players[1].health, 19)


# ---------------------------------------------------------------------------
class ObservationTests(unittest.TestCase):
    def test_obs_dimension_is_227(self):
        env = ShadowverseEnv(
            deck_a=[_card(100 + i) for i in range(40)],
            deck_b=[_card(200 + i) for i in range(40)],
            class_a=1, class_b=1, seed=42,
        )
        obs, _ = env.reset(seed=42)
        self.assertEqual(len(obs), 227)

    def test_action_size_is_111(self):
        self.assertEqual(ShadowverseEnv.ACTION_SIZE, 111)


# ---------------------------------------------------------------------------
class BackwardCompatTests(unittest.TestCase):
    def test_old_rules_load(self):
        rb = RuleBook.from_directory("data/rules")
        self.assertIsInstance(rb, RuleBook)

    def test_old_play_works(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))


if __name__ == "__main__":
    unittest.main()
