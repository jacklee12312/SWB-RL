# -*- coding: utf-8 -*-
"""Tests for emblem system."""

from __future__ import annotations

import os
import sqlite3
import json
import tempfile
import unittest
from pathlib import Path

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.emblem import (
    EmblemDefinition,
    EmblemStacking,
    EmblemTriggerRule,
    EventScope,
    TurnScope,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import (
    Amulet,
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


def _resolver(defs):
    return lambda cid: defs.get(cid)


def _engine(*rules, defs=None, seed=42):
    d = dict(defs) if defs else {}
    return GameEngine(
        deck_a=[_card(i) for i in range(1000, 1040)],
        deck_b=[_card(i) for i in range(2000, 2040)],
        class_a=1, class_b=1, seed=seed,
        rulebook=RuleBook(tuple(rules)),
        card_resolver=_resolver(d),
    )


def _fr(cid, *ops):
    return CardRule(card_id=cid, trigger=Trigger.FANFARE, operations=ops)


def _insert_card(engine, card_def, origin=CardOrigin.DECK):
    hc = HandCard(definition=card_def, entity_id=engine.state.allocate_entity_id(), origin=origin)
    engine.players[0].hand.insert(0, hc)
    engine.players[0].hand_entity_ids.insert(0, hc.entity_id)
    return hc


def _advance_two_turns(engine):
    engine.apply(EndTurn(engine.current_player))
    engine.apply(EndTurn(engine.current_player))


# ---------------------------------------------------------------------------
class EmblemGainTests(unittest.TestCase):
    def test_gain_permanent_emblem(self):
        ed = EmblemDefinition("test_emblem", 999900, stacking=EmblemStacking.ALLOW, triggers=())
        engine = _engine(_fr(999900, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="test_emblem")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"test_emblem": ed}
        _insert_card(engine, _card(999900, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].emblems), 1)

    def test_multiple_emblems_stacking_allow(self):
        ed = EmblemDefinition("multi", 999900, stacking=EmblemStacking.ALLOW, triggers=())
        engine = _engine(_fr(999900, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="multi"), EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="multi")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"multi": ed}
        _insert_card(engine, _card(999900, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].emblems), 2)

    def test_emblem_stacking_replace(self):
        ed = EmblemDefinition("replace_me", 999900, stacking=EmblemStacking.REPLACE, triggers=())
        engine = _engine(_fr(999900, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="replace_me"), EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="replace_me")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"replace_me": ed}
        _insert_card(engine, _card(999900, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].emblems), 1)
        removed = [
            event for event in engine.event_history
            if event.type is EventType.EMBLEM_REMOVED
        ]
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].metadata["removal_cause"], "replace")

    def test_emblem_stacking_ignore(self):
        ed = EmblemDefinition("ignore_me", 999900, stacking=EmblemStacking.IGNORE, triggers=())
        engine = _engine(_fr(999900, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="ignore_me"), EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="ignore_me")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"ignore_me": ed}
        _insert_card(engine, _card(999900, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].emblems), 1)

    def test_reset_clears_emblems(self):
        ed = EmblemDefinition("test_emblem", 999900, stacking=EmblemStacking.ALLOW, triggers=())
        engine = _engine(_fr(999900, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="test_emblem")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"test_emblem": ed}
        _insert_card(engine, _card(999900, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].emblems), 1)
        engine.reset(seed=42)
        self.assertEqual(len(engine.players[0].emblems), 0)


# ---------------------------------------------------------------------------
class TurnStartEmblemTests(unittest.TestCase):
    def test_turn_start_emblem_fires_on_controller_turn(self):
        ed = EmblemDefinition("fire", 999901, stacking=EmblemStacking.ALLOW, triggers=(EmblemTriggerRule("turn_start", operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(_fr(999901, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="fire")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"fire": ed}
        _insert_card(engine, _card(999901, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        _advance_two_turns(engine)
        self.assertLess(engine.players[1].health, 20)

    def test_turn_start_does_not_fire_on_opponent_turn(self):
        ed = EmblemDefinition("fire2", 999901, stacking=EmblemStacking.ALLOW, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OWNER_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(_fr(999901, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="fire2")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"fire2": ed}
        _insert_card(engine, _card(999901, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        opp_health = engine.players[1].health
        engine.apply(EndTurn(engine.current_player))
        self.assertEqual(engine.players[1].health, opp_health)


# ---------------------------------------------------------------------------
class CountdownEmblemTests(unittest.TestCase):
    def test_countdown_emblem_expires_no_trigger(self):
        ed = EmblemDefinition("cd_emblem", 999902, stacking=EmblemStacking.ALLOW, countdown=1, triggers=(EmblemTriggerRule("turn_start", operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(_fr(999902, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="cd_emblem")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"cd_emblem": ed}
        _insert_card(engine, _card(999902, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].emblems), 1)
        self.assertEqual(engine.players[0].emblems[0].countdown, 1)
        _advance_two_turns(engine)
        self.assertEqual(len(engine.players[0].emblems), 0)


# ---------------------------------------------------------------------------
class TurnEndEmblemTests(unittest.TestCase):
    def test_turn_end_emblem_fires(self):
        ed = EmblemDefinition("heal", 999902, stacking=EmblemStacking.ALLOW, triggers=(EmblemTriggerRule("turn_end", operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(_fr(999902, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="heal")))
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"heal": ed}
        _insert_card(engine, _card(999902, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        engine.apply(EndTurn(engine.current_player))
        self.assertLess(engine.players[1].health, 20)

    def test_choice_emblem_pauses_turn_and_resumes_remaining_emblems(self):
        first = EmblemDefinition(
            "choice",
            999901,
            triggers=(
                EmblemTriggerRule(
                    "turn_end",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_UNIT,
                            TargetKind.ENEMY_UNIT,
                            1,
                        ),
                    ),
                ),
            ),
        )
        second = EmblemDefinition(
            "after",
            999902,
            triggers=(
                EmblemTriggerRule(
                    "turn_end",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            2,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(900, life=5), entity_id=900)
        engine.players[1].board.append(target)
        engine._add_emblem_to_player(0, first, first.source_card_id)
        engine._add_emblem_to_player(0, second, second.source_card_id)

        engine.apply(EndTurn(0))

        self.assertIsNotNone(engine.state.pending_choice)
        self.assertEqual(engine.current_player, 0)
        self.assertEqual(engine.turn, 1)
        choose = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        engine.apply(choose)

        self.assertEqual(target.health, 4)
        self.assertEqual(engine.players[1].health, 18)
        self.assertEqual(engine.current_player, 1)
        self.assertEqual(engine.turn, 2)

    def test_choice_emblem_stale_target_skips_and_resumes_remaining_emblems(self):
        first = EmblemDefinition(
            "choice_stale",
            999901,
            triggers=(
                EmblemTriggerRule(
                    "turn_end",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_UNIT,
                            TargetKind.ENEMY_UNIT,
                            2,
                        ),
                    ),
                ),
            ),
        )
        second = EmblemDefinition(
            "after_stale",
            999902,
            triggers=(
                EmblemTriggerRule(
                    "turn_end",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            3,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(901, life=5), entity_id=901)
        engine.players[1].board.append(target)
        engine._add_emblem_to_player(0, first, first.source_card_id)
        engine._add_emblem_to_player(0, second, second.source_card_id)

        engine.apply(EndTurn(0))
        self.assertIsNotNone(engine.state.pending_choice)
        choose = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        engine.players[1].board.remove(target)
        engine._send_to_graveyard(
            1,
            target.definition,
            "test_target_left_play",
            source_entity_id=target.entity_id,
        )

        engine.apply(choose)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(target.health, 5)
        self.assertEqual(engine.players[1].health, 17)
        self.assertEqual(engine.current_player, 1)
        self.assertEqual(engine.turn, 2)


class EventEmblemTests(unittest.TestCase):
    def _damage_emblem(self, trigger):
        return EmblemDefinition(
            f"watch_{trigger}",
            999910,
            triggers=(
                EmblemTriggerRule(
                    trigger,
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

    def test_follower_summoned_emblem_fires(self):
        engine = _engine()
        engine.reset(seed=42)
        definition = self._damage_emblem("follower_summoned")
        engine._add_emblem_to_player(0, definition, definition.source_card_id)
        _insert_card(engine, _card(999920, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 19)

    def test_follower_destroyed_emblem_fires(self):
        destroy_card = 999924
        engine = _engine(
            CardRule(card_id=destroy_card, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    EffectKind.DAMAGE_UNIT,
                    TargetKind.ALL_ENEMY_UNITS,
                    5,
                ),
            ),)
        )
        engine.reset(seed=42)
        definition = self._damage_emblem("follower_destroyed")
        engine._add_emblem_to_player(1, definition, definition.source_card_id)
        engine.players[1].board.append(Unit.summon(_card(999925, life=1)))
        _insert_card(
            engine,
            _card(destroy_card, cost=1, card_type="法术", attack=None, life=None),
        )
        engine.players[0].mana = 10

        transition = engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].health, 19)
        self.assertEqual(engine.players[1].board, [])
        self.assertTrue(
            any(
                event.type == EventType.EMBLEM_TRIGGERED
                and event.metadata["trigger"] == "follower_destroyed"
                for event in transition.events
            )
        )

    def test_amulet_destroyed_emblem_fires(self):
        destroy_card = 999927
        engine = _engine(
            CardRule(card_id=destroy_card, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    EffectKind.DESTROY,
                    TargetKind.ALL_OWN_AMULETS,
                ),
            ),)
        )
        engine.reset(seed=42)
        definition = self._damage_emblem("amulet_destroyed")
        engine._add_emblem_to_player(0, definition, definition.source_card_id)
        engine.players[0].board.append(
            Amulet(
                definition=_card(999928, card_type="护符", attack=None, life=None),
                entity_id=999928,
            )
        )
        _insert_card(
            engine,
            _card(destroy_card, cost=1, card_type="法术", attack=None, life=None),
        )
        engine.players[0].mana = 10

        transition = engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[1].health, 19)
        self.assertEqual(engine.players[0].board, [])
        self.assertTrue(
            any(
                event.type == EventType.EMBLEM_TRIGGERED
                and event.metadata["trigger"] == "amulet_destroyed"
                for event in transition.events
            )
        )

    def test_follower_destroyed_emblem_trigger_loads_from_json(self):
        payload = {
            "emblems": [
                {
                    "id": "death_watch",
                    "source_card_id": 999926,
                    "triggers": [
                        {
                            "trigger": "follower_destroyed",
                            "event_scope": "any_event",
                            "operations": [
                                {
                                    "kind": "damage_leader",
                                    "target": "enemy_leader",
                                    "amount": 1,
                                },
                            ],
                        },
                    ],
                },
            ],
            "rules": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rules.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            rulebook = RuleBook.from_directory(tmp)

        operations = rulebook.emblem_trigger_ops_for(
            "death_watch",
            "follower_destroyed",
        )
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].kind, EffectKind.DAMAGE_LEADER)
        self.assertEqual(operations[0].target, TargetKind.ENEMY_LEADER)
        self.assertEqual(
            rulebook.emblem_def("death_watch").triggers[0].event_scope,
            EventScope.ANY_EVENT,
        )

    def test_amulet_destroyed_emblem_trigger_loads_from_json(self):
        payload = {
            "emblems": [
                {
                    "id": "amulet_death_watch",
                    "source_card_id": 999929,
                    "triggers": [
                        {
                            "trigger": "amulet_destroyed",
                            "event_scope": "owner_event",
                            "operations": [
                                {
                                    "kind": "damage_leader",
                                    "target": "enemy_leader",
                                    "amount": 1,
                                },
                            ],
                        },
                    ],
                },
            ],
            "rules": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rules.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            rulebook = RuleBook.from_directory(tmp)

        operations = rulebook.emblem_trigger_ops_for(
            "amulet_death_watch",
            "amulet_destroyed",
        )
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].kind, EffectKind.DAMAGE_LEADER)
        self.assertEqual(operations[0].target, TargetKind.ENEMY_LEADER)
        self.assertEqual(
            rulebook.emblem_def("amulet_death_watch").triggers[0].event_scope,
            EventScope.OWNER_EVENT,
        )

    def test_death_batch_end_emblem_trigger_loads_from_json(self):
        payload = {
            "emblems": [
                {
                    "id": "batch_end_watch",
                    "source_card_id": 999930,
                    "triggers": [
                        {
                            "trigger": "death_batch_end",
                            "event_scope": "any_event",
                            "operations": [
                                {
                                    "kind": "damage_leader",
                                    "target": "enemy_leader",
                                    "amount": 1,
                                },
                            ],
                        },
                    ],
                },
            ],
            "rules": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rules.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            rulebook = RuleBook.from_directory(tmp)

        operations = rulebook.emblem_trigger_ops_for(
            "batch_end_watch",
            "death_batch_end",
        )
        trigger = rulebook.emblem_def("batch_end_watch").triggers[0]
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].kind, EffectKind.DAMAGE_LEADER)
        self.assertEqual(trigger.trigger, "death_batch_end")
        self.assertEqual(trigger.event_scope, EventScope.ANY_EVENT)

    def test_death_batch_start_emblem_trigger_remains_unsupported(self):
        payload = {
            "emblems": [
                {
                    "id": "batch_start_watch",
                    "source_card_id": 999931,
                    "triggers": [
                        {
                            "trigger": "death_batch_start",
                            "operations": [
                                {
                                    "kind": "damage_leader",
                                    "target": "enemy_leader",
                                    "amount": 1,
                                },
                            ],
                        },
                    ],
                },
            ],
            "rules": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rules.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "death_batch_start"):
                RuleBook.from_directory(tmp)

    def test_emblem_trigger_multi_target_fields_are_supported(self):
        payload = {
            "emblems": [
                {
                    "id": "bad_multi_trigger",
                    "source_card_id": 999932,
                    "triggers": [
                        {
                            "trigger": "card_played",
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
                },
            ],
            "rules": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rules.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            rulebook = RuleBook.from_directory(tmp)
            operation = rulebook.emblem_trigger_ops_for(
                "bad_multi_trigger",
                "card_played",
            )[0]
            self.assertEqual(operation.target_count, 2)

    def test_emblem_on_expire_multi_target_fields_are_supported(self):
        payload = {
            "emblems": [
                {
                    "id": "bad_multi_expire",
                    "source_card_id": 999933,
                    "countdown": 1,
                    "on_expire": [
                        {
                            "kind": "damage_unit",
                            "target": "enemy_unit",
                            "amount": 1,
                            "allow_duplicate_targets": False,
                        },
                    ],
                },
            ],
            "rules": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rules.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            definition = RuleBook.from_directory(tmp).emblem_def(
                "bad_multi_expire"
            )
            self.assertEqual(len(definition.on_expire), 1)
            self.assertFalse(
                definition.on_expire[0].allow_duplicate_targets
            )

    def test_card_played_emblem_fires(self):
        engine = _engine()
        engine.reset(seed=42)
        definition = self._damage_emblem("card_played")
        engine._add_emblem_to_player(0, definition, definition.source_card_id)
        _insert_card(engine, _card(999921, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 19)

    def test_follower_evolved_emblem_fires(self):
        from swb.engine.commands import Evolve

        engine = _engine()
        engine.reset(seed=42)
        definition = self._damage_emblem("follower_evolved")
        engine._add_emblem_to_player(0, definition, definition.source_card_id)
        unit = Unit.summon(_card(999922), entity_id=999922)
        unit.summoned_this_turn = False
        engine.players[0].board.append(unit)
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        engine.apply(Evolve(0, unit.entity_id))
        self.assertEqual(engine.players[1].health, 19)

    def test_leader_healed_emblem_fires(self):
        heal_card = 999923
        engine = _engine(
            _fr(
                heal_card,
                EffectOperation(
                    EffectKind.HEAL_LEADER,
                    TargetKind.OWN_LEADER,
                    2,
                ),
            )
        )
        engine.reset(seed=42)
        definition = self._damage_emblem("leader_healed")
        engine._add_emblem_to_player(0, definition, definition.source_card_id)
        engine.players[0].health = 10
        _insert_card(engine, _card(heal_card, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].health, 12)
        self.assertEqual(engine.players[1].health, 19)

    def test_choice_event_emblem_resumes_remaining_emblems(self):
        first = EmblemDefinition(
            "summon_choice",
            999930,
            triggers=(
                EmblemTriggerRule(
                    "follower_summoned",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_UNIT,
                            TargetKind.ENEMY_UNIT,
                            1,
                        ),
                    ),
                ),
            ),
        )
        second = self._damage_emblem("follower_summoned")
        engine = _engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(999931, life=5), entity_id=999931)
        engine.players[1].board.append(target)
        engine._add_emblem_to_player(0, first, first.source_card_id)
        engine._add_emblem_to_player(0, second, second.source_card_id)
        _insert_card(engine, _card(999932, cost=1))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        choose = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        engine.apply(choose)

        self.assertEqual(target.health, 4)
        self.assertEqual(engine.players[1].health, 19)
        self.assertIsNone(engine.state.pending_choice)

    def test_card_played_choice_finishes_before_fanfare(self):
        played_card_id = 999940
        watcher = EmblemDefinition(
            "play_choice",
            999941,
            triggers=(
                EmblemTriggerRule(
                    "card_played",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_UNIT,
                            TargetKind.ENEMY_UNIT,
                            1,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine(
            _fr(
                played_card_id,
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    2,
                ),
            )
        )
        engine.reset(seed=42)
        target = Unit.summon(_card(999942, life=5), entity_id=999942)
        engine.players[1].board.append(target)
        engine._add_emblem_to_player(0, watcher, watcher.source_card_id)
        _insert_card(engine, _card(played_card_id, cost=1))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        self.assertEqual(engine.players[1].health, 20)
        choose = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        engine.apply(choose)

        self.assertEqual(target.health, 4)
        self.assertEqual(engine.players[1].health, 18)
        self.assertIsNone(engine.state.pending_choice)


# ---------------------------------------------------------------------------
class MultiEmblemOrderTests(unittest.TestCase):
    def test_emblems_fire_in_creation_order(self):
        ed1 = EmblemDefinition("first", 999901, stacking=EmblemStacking.ALLOW, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OWNER_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        ed2 = EmblemDefinition("second", 999902, stacking=EmblemStacking.ALLOW, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OWNER_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
        engine = _engine(
            _fr(999901, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="first")),
            _fr(999902, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="second")),
            defs={999901: _card(999901, cost=1), 999902: _card(999902, cost=1)},
        )
        engine.reset(seed=42)
        engine.rulebook._emblem_defs = {"first": ed1, "second": ed2}
        _insert_card(engine, _card(999901, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        _insert_card(engine, _card(999902, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].emblems), 2)
        opp_before = engine.players[1].health
        _advance_two_turns(engine)
        self.assertEqual(engine.players[1].health, opp_before - 2)


# ---------------------------------------------------------------------------
class DeterminismTests(unittest.TestCase):
    def test_emblem_determinism(self):
        for _ in range(2):
            ed = EmblemDefinition("det", 999900, stacking=EmblemStacking.ALLOW, triggers=(EmblemTriggerRule("turn_start", turn_scope=TurnScope.OWNER_TURN, operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),)),))
            engine = _engine(_fr(999900, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="det")))
            engine.reset(seed=42)
            engine.rulebook._emblem_defs = {"det": ed}
            _insert_card(engine, _card(999900, cost=1))
            engine.players[0].mana = 10
            engine.apply(PlayCard(0, 0))
            _advance_two_turns(engine)
            self.assertEqual(engine.players[1].health, 19)


# ---------------------------------------------------------------------------
class SchemaValidationTests(unittest.TestCase):
    def test_unknown_emblem_id_graceful(self):
        engine = _engine(_fr(999900, EffectOperation(EffectKind.GAIN_EMBLEM, TargetKind.OWN_LEADER, emblem_id="nonexistent")))
        engine.reset(seed=42)
        _insert_card(engine, _card(999900, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].emblems), 0)

    def test_rulebook_rejects_unknown_emblem_reference(self):
        payload = {
            "rules": [
                {
                    "card_id": 999900,
                    "trigger": "fanfare",
                    "operations": [
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
            Path(directory, "bad.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                RuleBook.from_directory(directory)

    def test_remove_all_emblems(self):
        definition = EmblemDefinition("remove_me", 999900)
        remove_rule = _fr(
            999901,
            EffectOperation(
                EffectKind.REMOVE_EMBLEM,
                TargetKind.OWN_LEADER,
                emblem_id="remove_me",
                emblem_remove_mode="all",
            ),
        )
        engine = _engine(remove_rule)
        engine.reset(seed=42)
        engine._add_emblem_to_player(0, definition, definition.source_card_id)
        engine._add_emblem_to_player(0, definition, definition.source_card_id)
        _insert_card(engine, _card(999901, cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].emblems, [])
        removed = [
            event for event in engine.event_history
            if event.type is EventType.EMBLEM_REMOVED
        ]
        self.assertEqual(len(removed), 2)


# ---------------------------------------------------------------------------
class BackwardCompatTests(unittest.TestCase):
    def test_old_rules_loading_still_works(self):
        rb = RuleBook.from_directory("data/rules")
        self.assertIsInstance(rb, RuleBook)

    def test_old_play_card_still_works(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))


# ---------------------------------------------------------------------------
class ObservationTests(unittest.TestCase):
    def test_observation_dimension(self):
        env = ShadowverseEnv(
            deck_a=[_card(100 + i) for i in range(40)],
            deck_b=[_card(200 + i) for i in range(40)],
            class_a=1, class_b=1, seed=42,
        )
        obs, _ = env.reset(seed=42)
        self.assertEqual(len(obs), 255)

    def test_observation_exposes_public_emblem_state(self):
        env = ShadowverseEnv(
            deck_a=[_card(100 + i) for i in range(40)],
            deck_b=[_card(200 + i) for i in range(40)],
            class_a=1, class_b=1, seed=42,
        )
        before, _ = env.reset(seed=42)
        definition = EmblemDefinition("visible", 999900, countdown=3)
        env.core._add_emblem_to_player(0, definition, definition.source_card_id)
        after = env.observation()
        self.assertNotEqual(before, after)
        self.assertEqual(after[24:30], [0.1, 0.0, 1.0, 0.0, 0.3, 0.0])

    def test_action_size(self):
        self.assertEqual(ShadowverseEnv.ACTION_SIZE, 111)


# ---------------------------------------------------------------------------
class RealCardEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(str(cls.db_path)):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_10214110_exists_and_has_fairy(self):
        repo = CardRepository(str(self.db_path))
        try:
            card = repo.get(10214110)
        except KeyError:
            self.skipTest("Card 10214110 not found")
        self.assertEqual(card.cost, 4)
        try:
            fairy = repo.get(90011110)
            self.assertEqual(fairy.card_set_id, 90000)
        except KeyError:
            self.skipTest("Fairy 90011110 not found")

    def test_emblem_def_loaded_from_rules(self):
        rulebook = RuleBook.from_directory("data/rules")
        ed = rulebook.emblem_def("wings_queen_titania")
        self.assertIsNotNone(ed)
        self.assertTrue(any(tr.trigger == "turn_start" for tr in ed.triggers))

    def test_10214110_real_card_gains_emblem_and_generates_fairy(self):
        repo = CardRepository(str(self.db_path))
        card = repo.get(10214110)
        rulebook = RuleBook.from_directory("data/rules")
        engine = GameEngine(
            deck_a=[_card(i) for i in range(1000, 1040)],
            deck_b=[_card(i) for i in range(2000, 2040)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=rulebook,
            card_resolver=repo.get,
        )
        engine.reset(seed=42)
        _insert_card(engine, card)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertTrue(
            any(
                emblem.emblem_id == "wings_queen_titania"
                for emblem in engine.players[0].emblems
            )
        )
        fairies_before = sum(
            hand.definition.card_id == 90011110
            for hand in engine.players[0].hand
        )
        _advance_two_turns(engine)
        fairies_after = sum(
            hand.definition.card_id == 90011110
            for hand in engine.players[0].hand
        )
        self.assertEqual(fairies_after, fairies_before + 1)

    def test_10153140_real_card_turn_end_emblem(self):
        repo = CardRepository(str(self.db_path))
        card = repo.get(10153140)
        rulebook = RuleBook.from_directory("data/rules")
        engine = GameEngine(
            deck_a=[_card(i) for i in range(1000, 1040)],
            deck_b=[_card(i) for i in range(2000, 2040)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=rulebook,
            card_resolver=repo.get,
        )
        engine.reset(seed=42)
        _insert_card(engine, card)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        engine.apply(EndTurn(0))
        self.assertEqual(engine.players[0].health, 19)
        self.assertEqual(engine.players[1].health, 19)


if __name__ == "__main__":
    unittest.main()
