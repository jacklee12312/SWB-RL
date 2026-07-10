# -*- coding: utf-8 -*-
"""Structured board, hand, and leader-area card listener contracts."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.effects import Condition, ConditionType
from swb.engine.emblem import EventScope, TurnScope
from swb.engine.events import EventType, GameEvent
from swb.engine.faith import FaithDefinition, FaithInstance
from swb.engine.listeners import (
    CardListenerDefinition,
    EventCardFilter,
    ListenerZone,
    SourceRelation,
)
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


def _card(card_id: int, **overrides) -> CardDefinition:
    values = {
        "card_id": card_id,
        "card_set_id": 10000,
        "class_id": 1,
        "class_name": "精灵",
        "name": f"card-{card_id}",
        "cost": 1,
        "card_type": "随从",
        "attack": 2,
        "life": 3,
        "keywords": frozenset(),
        "support_level": "basic",
        "is_collectible": True,
    }
    values.update(overrides)
    return CardDefinition(**values)


def _listener(
    card_id: int,
    zone: ListenerZone,
    event: EventType,
    *operations: EffectOperation,
    **overrides,
) -> CardListenerDefinition:
    return CardListenerDefinition(
        card_id=card_id,
        zone=zone,
        event=event,
        operations=tuple(operations),
        **overrides,
    )


def _engine(
    *listeners: CardListenerDefinition,
    rules: tuple[CardRule, ...] = (),
    catalog: dict[int, CardDefinition] | None = None,
) -> GameEngine:
    grouped: dict[int, list[CardListenerDefinition]] = {}
    for definition in listeners:
        grouped.setdefault(definition.card_id, []).append(definition)
    cards = catalog or {}
    engine = GameEngine(
        [_card(1000 + index) for index in range(40)],
        [_card(2000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=42,
        rulebook=RuleBook(
            rules=rules,
            listener_defs={
                card_id: tuple(definitions)
                for card_id, definitions in grouped.items()
            },
        ),
        card_resolver=lambda card_id: cards.get(card_id),
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=42)
    return engine


def _place(
    engine: GameEngine,
    player_index: int,
    definition: CardDefinition,
) -> Unit:
    unit = Unit.summon(
        definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[player_index].board.append(unit)
    return unit


def _insert_hand(
    engine: GameEngine,
    player_index: int,
    definition: CardDefinition,
) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    player = engine.players[player_index]
    player.hand.append(card)
    player.hand_entity_ids.append(card.entity_id)
    return card


def _resolve_event(engine: GameEngine, event: GameEvent) -> None:
    engine._emit(event)
    engine._resolve_event_queue()
    engine._stabilize()


class ListenerSchemaTests(unittest.TestCase):
    def test_schema_parses_filters_scopes_limits_and_turn_alias(self):
        payload = {
            "listeners": [
                {
                    "card_id": 999101,
                    "zone": "hand",
                    "event": "turn_start",
                    "event_scope": "opponent_event",
                    "turn_scope": "opponent_turn",
                    "source_relation": "any",
                    "once_per_turn": True,
                    "max_activations": 2,
                    "operations": [
                        {
                            "kind": "damage_leader",
                            "target": "enemy_leader",
                            "amount": 1,
                        }
                    ],
                },
                {
                    "card_id": 999102,
                    "zone": "board",
                    "event": "follower_summoned",
                    "event_filter": {
                        "card_type": "随从",
                        "class_id": 1,
                        "class_name": "精灵",
                        "cost_min": 2,
                        "cost_max": 4,
                        "card_id": 999200,
                        "card_name": "filtered",
                        "keyword": "守护",
                    },
                    "operations": [
                        {
                            "kind": "add_keyword",
                            "target": "event_source",
                            "keyword": "突进",
                        }
                    ],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "listeners.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            rulebook = RuleBook.from_directory(directory)

        first = rulebook.listeners_for(999101)[0]
        self.assertIs(first.event, EventType.TURN_STARTED)
        self.assertIs(first.event_scope, EventScope.OPPONENT_EVENT)
        self.assertIs(first.turn_scope, TurnScope.OPPONENT_TURN)
        self.assertTrue(first.once_per_turn)
        self.assertEqual(first.max_activations, 2)
        second = rulebook.listeners_for(999102)[0]
        self.assertEqual(second.event_filter.card_name, "filtered")
        self.assertIs(
            second.operations[0].target,
            TargetKind.EVENT_SOURCE,
        )

    def test_schema_rejects_bad_zone_event_filter_and_limit(self):
        base = {
            "card_id": 999101,
            "zone": "board",
            "event": "card_played",
            "operations": [
                {"kind": "draw", "target": "own_leader", "amount": 1}
            ],
        }
        invalid_entries = [
            {**base, "zone": "deck"},
            {**base, "event": "damage_dealt"},
            {**base, "once_per_turn": 1},
            {**base, "max_activations": 0},
            {**base, "event_filter": {"unknown": 1}},
            {
                **base,
                "event": "turn_started",
                "event_filter": {"cost_min": 1},
            },
            {
                **base,
                "operations": [
                    {
                        "kind": "draw",
                        "target": "event_source",
                        "amount": 1,
                    }
                ],
            },
        ]
        for index, entry in enumerate(invalid_entries):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                with open(
                    os.path.join(directory, "bad.json"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump({"listeners": [entry]}, handle, ensure_ascii=False)
                with self.assertRaises(ValueError):
                    RuleBook.from_directory(directory)


class ListenerEventTests(unittest.TestCase):
    def test_all_supported_events_dispatch_from_hand(self):
        supported = (
            EventType.AMULET_ACTIVATED,
            EventType.CARD_FUSED,
            EventType.FOLLOWER_SUMMONED,
            EventType.FOLLOWER_EVOLVED,
            EventType.FOLLOWER_SUPER_EVOLVED,
            EventType.FOLLOWER_DESTROYED,
            EventType.AMULET_DESTROYED,
            EventType.ENTITY_LEFT_PLAY,
            EventType.CARD_PLAYED,
            EventType.TURN_STARTED,
            EventType.TURN_ENDED,
        )
        for offset, event_type in enumerate(supported):
            with self.subTest(event=event_type.value):
                source_card = _card(300 + offset)
                definition = _listener(
                    source_card.card_id,
                    ListenerZone.HAND,
                    event_type,
                    EffectOperation(
                        EffectKind.DAMAGE_LEADER,
                        TargetKind.ENEMY_LEADER,
                        1,
                    ),
                )
                engine = _engine(definition)
                _insert_hand(engine, 0, source_card)
                subject = _card(900 + offset)
                _resolve_event(
                    engine,
                    GameEvent(
                        event_type,
                        0,
                        source_id=(None if event_type in {
                            EventType.TURN_STARTED,
                            EventType.TURN_ENDED,
                        } else engine.state.allocate_entity_id()),
                        metadata=(
                            {}
                            if event_type in {
                                EventType.TURN_STARTED,
                                EventType.TURN_ENDED,
                            }
                            else {"definition": subject}
                        ),
                    ),
                )
                self.assertEqual(engine.players[1].health, 19)

    def test_full_event_filter_uses_runtime_keyword(self):
        listener_card = _card(300)
        subject_card = _card(
            400,
            name="filtered",
            cost=2,
            keywords=frozenset(),
        )
        definition = _listener(
            listener_card.card_id,
            ListenerZone.HAND,
            EventType.FOLLOWER_SUMMONED,
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                1,
            ),
            event_filter=EventCardFilter(
                card_type="随从",
                class_id=1,
                class_name="精灵",
                cost_min=2,
                cost_max=2,
                card_id=400,
                card_name="filtered",
                keyword="守护",
            ),
        )
        engine = _engine(definition)
        _insert_hand(engine, 0, listener_card)
        subject = _place(engine, 0, subject_card)
        subject.add_keyword("守护")

        _resolve_event(
            engine,
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                0,
                source_id=subject.entity_id,
                metadata={"source": subject},
            ),
        )

        self.assertEqual(engine.players[1].health, 19)

    def test_event_scope_turn_scope_and_source_relation(self):
        listener_card = _card(300)
        definition = _listener(
            listener_card.card_id,
            ListenerZone.BOARD,
            EventType.CARD_PLAYED,
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                1,
            ),
            event_scope=EventScope.OWNER_EVENT,
            turn_scope=TurnScope.OWNER_TURN,
            source_relation=SourceRelation.OTHER,
        )
        engine = _engine(definition)
        source = _place(engine, 0, listener_card)
        other = _card(400)

        _resolve_event(
            engine,
            GameEvent(
                EventType.CARD_PLAYED,
                1,
                source_id=engine.state.allocate_entity_id(),
                metadata={"definition": other},
            ),
        )
        _resolve_event(
            engine,
            GameEvent(
                EventType.CARD_PLAYED,
                0,
                source_id=source.entity_id,
                metadata={"source": source},
            ),
        )
        self.assertEqual(engine.players[1].health, 20)

        _resolve_event(
            engine,
            GameEvent(
                EventType.CARD_PLAYED,
                0,
                source_id=engine.state.allocate_entity_id(),
                metadata={"definition": other},
            ),
        )
        self.assertEqual(engine.players[1].health, 19)


class ListenerZoneAndLifecycleTests(unittest.TestCase):
    def test_board_hand_and_leader_area_sources_all_activate(self):
        cards = {card_id: _card(card_id) for card_id in (300, 301, 302)}
        listeners = tuple(
            _listener(
                card_id,
                zone,
                EventType.CARD_PLAYED,
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    1,
                ),
            )
            for card_id, zone in (
                (300, ListenerZone.BOARD),
                (301, ListenerZone.HAND),
                (302, ListenerZone.LEADER_AREA),
            )
        )
        engine = _engine(*listeners, catalog=cards)
        _place(engine, 0, cards[300])
        _insert_hand(engine, 0, cards[301])
        faith_definition = FaithDefinition("listener-faith", 302)
        engine.players[0].faiths.append(FaithInstance(
            definition=faith_definition,
            entity_id=engine.state.allocate_entity_id(),
            controller=0,
            created_sequence=1,
        ))

        _resolve_event(
            engine,
            GameEvent(
                EventType.CARD_PLAYED,
                0,
                source_id=engine.state.allocate_entity_id(),
                metadata={"definition": _card(400)},
            ),
        )

        self.assertEqual(engine.players[1].health, 17)
        triggered = [
            event.metadata["listener_zone"]
            for event in engine.event_history
            if event.type is EventType.CARD_LISTENER_TRIGGERED
        ]
        self.assertEqual(triggered[-3:], ["board", "hand", "leader_area"])

    def test_removed_later_source_is_skipped_in_same_snapshot(self):
        first_card = _card(300)
        second_card = _card(301, life=1)
        first = _listener(
            300,
            ListenerZone.BOARD,
            EventType.FOLLOWER_SUMMONED,
            EffectOperation(
                EffectKind.DESTROY,
                TargetKind.ALL_ENEMY_UNITS,
            ),
        )
        second = _listener(
            301,
            ListenerZone.BOARD,
            EventType.FOLLOWER_SUMMONED,
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                5,
            ),
        )
        engine = _engine(first, second)
        first_source = _place(engine, 0, first_card)
        second_source = _place(engine, 1, second_card)
        subject = _place(engine, 0, _card(400))

        _resolve_event(
            engine,
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                0,
                source_id=subject.entity_id,
                metadata={"source": subject},
            ),
        )

        self.assertNotIn(second_source, engine.players[1].board)
        self.assertEqual(engine.players[0].health, 20)
        triggered_ids = [
            event.metadata["listener_card_id"]
            for event in engine.event_history
            if event.type is EventType.CARD_LISTENER_TRIGGERED
        ]
        self.assertIn(first_source.definition.card_id, triggered_ids)
        self.assertNotIn(second_source.definition.card_id, triggered_ids)

    def test_event_source_is_revalidated_after_it_leaves(self):
        listener_card = _card(300)
        definition = _listener(
            300,
            ListenerZone.HAND,
            EventType.FOLLOWER_SUMMONED,
            EffectOperation(EffectKind.DESTROY, TargetKind.EVENT_SOURCE),
            EffectOperation(
                EffectKind.ADD_KEYWORD,
                TargetKind.EVENT_SOURCE,
                keyword="守护",
            ),
        )
        engine = _engine(definition)
        _insert_hand(engine, 0, listener_card)
        subject = _place(engine, 0, _card(400, life=1))

        _resolve_event(
            engine,
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                0,
                source_id=subject.entity_id,
                metadata={"source": subject},
            ),
        )

        self.assertNotIn(subject, engine.players[0].board)
        engine.assert_invariants()

    def test_nested_operations_keep_event_source_context(self):
        listener_card = _card(300)
        definition = _listener(
            300,
            ListenerZone.HAND,
            EventType.FOLLOWER_SUMMONED,
            EffectOperation(
                EffectKind.CONDITIONAL,
                TargetKind.OWN_LEADER,
                conditions=(Condition(ConditionType.ALWAYS),),
                then_operations=(
                    EffectOperation(
                        EffectKind.ADD_KEYWORD,
                        TargetKind.EVENT_SOURCE,
                        keyword="守护",
                    ),
                ),
            ),
        )
        engine = _engine(definition)
        _insert_hand(engine, 0, listener_card)
        subject = _place(engine, 0, _card(400))

        _resolve_event(
            engine,
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                0,
                source_id=subject.entity_id,
                metadata={"source": subject},
            ),
        )

        self.assertTrue(subject.has_keyword("守护"))
        engine.assert_invariants()

    def test_source_independent_continuation_survives_listener_departure(self):
        listener_card = _card(300, life=1)
        definition = _listener(
            300,
            ListenerZone.BOARD,
            EventType.FOLLOWER_SUMMONED,
            EffectOperation(EffectKind.DESTROY, TargetKind.SELF),
            EffectOperation(
                EffectKind.ADD_KEYWORD,
                TargetKind.SELF,
                keyword="守护",
            ),
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                2,
            ),
        )
        engine = _engine(definition)
        source = _place(engine, 0, listener_card)
        subject = _place(engine, 0, _card(400))

        _resolve_event(
            engine,
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                0,
                source_id=subject.entity_id,
                metadata={"source": subject},
            ),
        )

        self.assertNotIn(source, engine.players[0].board)
        self.assertFalse(source.has_keyword("守护"))
        self.assertEqual(engine.players[1].health, 18)

    def test_once_per_turn_resets_and_max_activations_persists(self):
        once_card = _card(300)
        capped_card = _card(301)
        once = _listener(
            300,
            ListenerZone.HAND,
            EventType.CARD_PLAYED,
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                1,
            ),
            once_per_turn=True,
        )
        capped = _listener(
            301,
            ListenerZone.HAND,
            EventType.CARD_PLAYED,
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                2,
            ),
            max_activations=2,
        )
        engine = _engine(once, capped)
        _insert_hand(engine, 0, once_card)
        _insert_hand(engine, 0, capped_card)
        for _ in range(3):
            _resolve_event(
                engine,
                GameEvent(
                    EventType.CARD_PLAYED,
                    0,
                    source_id=engine.state.allocate_entity_id(),
                    metadata={"definition": _card(400)},
                ),
            )
        self.assertEqual(engine.players[1].health, 15)

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        _resolve_event(
            engine,
            GameEvent(
                EventType.CARD_PLAYED,
                0,
                source_id=engine.state.allocate_entity_id(),
                metadata={"definition": _card(401)},
            ),
        )
        self.assertEqual(engine.players[1].health, 14)


class ListenerChoiceTests(unittest.TestCase):
    def test_choice_pauses_and_resumes_remaining_listeners_in_order(self):
        first = _listener(
            300,
            ListenerZone.BOARD,
            EventType.FOLLOWER_SUMMONED,
            EffectOperation(
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT,
                1,
                requires_target=True,
            ),
        )
        second = _listener(
            301,
            ListenerZone.BOARD,
            EventType.FOLLOWER_SUMMONED,
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                2,
            ),
        )
        engine = _engine(first, second)
        _place(engine, 0, _card(300))
        _place(engine, 0, _card(301))
        target = _place(engine, 1, _card(500))
        subject = _place(engine, 0, _card(400))

        engine._emit(GameEvent(
            EventType.FOLLOWER_SUMMONED,
            0,
            source_id=subject.entity_id,
            metadata={"source": subject},
        ))
        engine._resolve_event_queue()

        self.assertIsNotNone(engine.state.pending_choice)
        self.assertEqual(engine.players[1].health, 20)
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "entity:999999"))
        self.assertEqual(engine.deterministic_fingerprint(), before)
        diagnostics = engine._loop_diagnostics()
        self.assertTrue(diagnostics["listener_batches"])
        self.assertTrue(diagnostics["recent_card_listener_triggers"])

        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertEqual(target.health, 2)
        self.assertEqual(engine.players[1].health, 18)
        triggered_ids = [
            event.metadata["listener_card_id"]
            for event in engine.event_history
            if event.type is EventType.CARD_LISTENER_TRIGGERED
        ]
        self.assertEqual(triggered_ids[-2:], [300, 301])


class EntityLeftPlayListenerTests(unittest.TestCase):
    def test_return_to_hand_emits_filterable_entity_left_play(self):
        listener_card = _card(300)
        spell = _card(600, card_type="法术", attack=None, life=None)
        listener = _listener(
            300,
            ListenerZone.HAND,
            EventType.ENTITY_LEFT_PLAY,
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                1,
            ),
            event_filter=EventCardFilter(card_type="随从", cost_max=2),
        )
        rule = CardRule(
            600,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.RETURN_TO_HAND,
                    TargetKind.ENEMY_UNIT,
                    requires_target=True,
                ),
            ),
        )
        engine = _engine(listener, rules=(rule,))
        _insert_hand(engine, 0, listener_card)
        _insert_hand(engine, 0, spell)
        target = _place(engine, 1, _card(500, cost=2))
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[1].health, 19)
        event = next(
            event
            for event in engine.event_history
            if event.type is EventType.ENTITY_LEFT_PLAY
            and event.source_id == target.entity_id
        )
        self.assertEqual(event.metadata["cause"], "return_to_hand")
        self.assertEqual(event.metadata["definition"].card_id, 500)


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealMegListenerTests(unittest.TestCase):
    def test_meg_gains_ward_only_for_allied_original_cost_two_follower(self):
        repo = CardRepository("data/cards.sqlite3")
        rulebook = RuleBook.from_directory("data/rules")
        engine = GameEngine(
            [_card(1000 + index) for index in range(40)],
            [_card(2000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=rulebook,
            card_resolver=repo.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=42)
        meg = engine._summon_follower_to_board(
            0,
            repo.get(10443110),
            summon_cause="test_setup",
        )
        self.assertIsNotNone(meg)
        enemy_two = _place(engine, 1, _card(700, cost=2))

        _resolve_event(
            engine,
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                1,
                source_id=enemy_two.entity_id,
                metadata={"source": enemy_two},
            ),
        )
        self.assertFalse(meg.has_keyword("守护"))

        allied_three = _place(engine, 0, _card(701, cost=3))
        _resolve_event(
            engine,
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                0,
                source_id=allied_three.entity_id,
                metadata={"source": allied_three},
            ),
        )
        self.assertFalse(meg.has_keyword("守护"))

        _insert_hand(engine, 0, _card(702, cost=2))
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        self.assertTrue(meg.has_keyword("守护"))


if __name__ == "__main__":
    unittest.main()
