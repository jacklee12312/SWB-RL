from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.report_rule_coverage import _build_coverage_report, _classify_card
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_union_burst_definition
from swb.engine.commands import Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit
from swb.engine.union_burst import UnionBurstDefinition, UnionBurstKind


def _card(card_id: int, **overrides) -> CardDefinition:
    values = {
        "card_id": card_id,
        "card_set_id": 10000,
        "class_id": 1,
        "class_name": "精灵",
        "name": f"card-{card_id}",
        "cost": 1,
        "card_type": "随从",
        "attack": 1,
        "life": 3,
        "keywords": frozenset(),
        "support_level": "basic",
        "is_collectible": True,
    }
    values.update(overrides)
    return CardDefinition(**values)


def _burst_card(card_id: int = 100) -> CardDefinition:
    return _card(
        card_id,
        name=f"Burst-{card_id}",
        keywords=frozenset({"入场曲", "奥义"}),
    )


def _definition(
    kind: UnionBurstKind = UnionBurstKind.UNION_BURST,
    *,
    card_id: int = 100,
    operations: tuple[EffectOperation, ...] | None = None,
) -> UnionBurstDefinition:
    return UnionBurstDefinition(
        card_id=card_id,
        kind=kind,
        operations=operations
        or (
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                2,
            ),
        ),
    )


def _engine(
    *definitions: UnionBurstDefinition,
    seed: int = 42,
) -> GameEngine:
    engine = GameEngine(
        [_card(1000 + index) for index in range(40)],
        [_card(2000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=RuleBook(
            union_burst_defs={
                definition.card_id: tuple(
                    item
                    for item in definitions
                    if item.card_id == definition.card_id
                )
                for definition in definitions
            }
        ),
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    return engine


def _insert_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int = 0,
    evolutions: int = 0,
) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
        evolutions_while_in_hand=evolutions,
    )
    player = engine.players[player_index]
    player.hand.append(hand_card)
    player.hand_entity_ids.append(hand_card.entity_id)
    return hand_card


def _place_unit(
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


def _unlock_evolution(engine: GameEngine, *, super_evolve: bool = False) -> None:
    player = engine.players[0]
    player.turns_started = max(
        player.turns_started,
        (
            engine.config.first_player_super_evolution_unlock_turn
            if super_evolve
            else engine.config.evolution_unlock_turn
        ),
    )


def _play_burst(
    engine: GameEngine,
    *,
    turns_started: int,
    evolutions: int = 0,
    definition: CardDefinition | None = None,
) -> None:
    card = definition or _burst_card()
    _insert_hand(engine, card, evolutions=evolutions)
    player = engine.players[0]
    player.turns_started = turns_started
    player.max_mana = 10
    player.mana = 10
    engine.apply(PlayCard(0, len(player.hand) - 1))


class UnionBurstSchemaTests(unittest.TestCase):
    def test_schema_parses_fixed_threshold_kinds_and_random_target(self):
        union = _parse_union_burst_definition(
            {
                "card_id": 123,
                "kind": "union_burst",
                "operations": [
                    {
                        "kind": "damage_unit",
                        "target": "random_enemy_unit_or_leader",
                        "amount": 2,
                    }
                ],
            },
            "test/union_bursts[0]",
        )
        super_art = _parse_union_burst_definition(
            {
                "card_id": 123,
                "kind": "super_skybound_art",
                "replace_base_operations": True,
                "operations": [
                    {
                        "kind": "damage_leader",
                        "target": "enemy_leader",
                        "amount": 1,
                    }
                ],
            },
            "test/union_bursts[1]",
        )

        self.assertEqual(union.threshold, 10)
        self.assertEqual(super_art.threshold, 15)
        self.assertFalse(union.replace_base_operations)
        self.assertTrue(super_art.replace_base_operations)
        self.assertIs(
            union.operations[0].target,
            TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER,
        )

    def test_schema_rejects_invalid_or_unsupported_definitions(self):
        invalid = (
            {"card_id": 123, "kind": "unknown", "operations": [{}]},
            {"card_id": 123, "kind": "union_burst", "operations": []},
            {
                "card_id": 123,
                "kind": "union_burst",
                "operations": [
                    {
                        "kind": "heal_unit",
                        "target": "random_enemy_unit_or_leader",
                        "amount": 1,
                    }
                ],
            },
            {
                "card_id": True,
                "kind": "union_burst",
                "operations": [{}],
            },
            {
                "card_id": 123,
                "kind": "union_burst",
                "operations": [{}],
                "unknown": 1,
            },
            {
                "card_id": 123,
                "kind": "super_skybound_art",
                "replace_base_operations": 1,
                "operations": [
                    {
                        "kind": "damage_leader",
                        "target": "enemy_leader",
                        "amount": 1,
                    }
                ],
            },
            {
                "card_id": 123,
                "kind": "union_burst",
                "operations": [
                    {
                        "kind": "damage_unit",
                        "target": "random_enemy_unit_or_leader",
                        "amount": 2,
                        "conditions": [
                            {"type": "target_health_at_least", "value": 1}
                        ],
                    }
                ],
            },
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_union_burst_definition(raw, "test/union_bursts[0]")

    def test_rulebook_sorts_kinds_and_rejects_duplicates_or_id_mismatch(self):
        union = _definition()
        super_art = _definition(UnionBurstKind.SUPER_SKYBOUND_ART)
        rulebook = RuleBook(union_burst_defs={100: (super_art, union)})
        self.assertEqual(
            [item.kind for item in rulebook.union_bursts_for(100)],
            [UnionBurstKind.UNION_BURST, UnionBurstKind.SUPER_SKYBOUND_ART],
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            RuleBook(union_burst_defs={100: (union, union)})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            RuleBook(union_burst_defs={101: (union,)})
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            RuleBook(union_burst_defs={100: ()})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            RuleBook(union_burst_defs={True: (union,)})

    def test_directory_rejects_duplicate_kind_across_files(self):
        payload = {
            "union_bursts": [
                {
                    "card_id": 100,
                    "kind": "union_burst",
                    "operations": [
                        {
                            "kind": "damage_leader",
                            "target": "enemy_leader",
                            "amount": 1,
                        }
                    ],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            for name in ("a.json", "b.json"):
                Path(directory, name).write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                RuleBook.from_directory(directory)

    def test_coverage_requires_per_card_union_burst_definition(self):
        card = _burst_card()
        common = {
            "card": card,
            "ruled_cards": {card.card_id},
            "ruled_ops": {
                card.card_id: {
                    "triggers": ["fanfare"],
                    "effect_kinds": ["damage_leader"],
                }
            },
            "rule_metadata": {},
            "ability_map": {card.card_id: ["奥义"]},
            "skill_text_map": {card.card_id: ["【奥义】造成2点伤害。"]},
            "support_map": {card.card_id: "unsupported"},
        }

        missing = _classify_card(**common, union_burst_cards=set())
        covered = _classify_card(
            **common,
            union_burst_cards={card.card_id},
        )

        self.assertEqual(missing["coverage"], "covered_partial")
        self.assertIn("奥义", missing["missing_rule_mechanics"])
        self.assertEqual(covered["coverage"], "covered_exact")


class UnionBurstGaugeTests(unittest.TestCase):
    def test_hand_card_gauge_is_turns_started_plus_own_hand_evolutions(self):
        hand_card = HandCard(_burst_card(), entity_id=1, evolutions_while_in_hand=3)
        self.assertEqual(hand_card.union_burst_gauge(7), 10)

    def test_successful_normal_evolution_increments_current_own_hand_only(self):
        engine = _engine(_definition())
        first = _insert_hand(engine, _burst_card(), evolutions=2)
        second = _insert_hand(engine, _card(101))
        opponent = _insert_hand(engine, _burst_card(102), player_index=1)
        unit = _place_unit(engine, 0, _card(200))
        _unlock_evolution(engine)

        engine.apply(Evolve(0, unit.entity_id))

        self.assertEqual(first.evolutions_while_in_hand, 3)
        self.assertEqual(second.evolutions_while_in_hand, 1)
        self.assertEqual(opponent.evolutions_while_in_hand, 0)
        engine.assert_invariants()

    def test_successful_super_evolution_also_increments_hand_gauge(self):
        engine = _engine(_definition())
        hand_card = _insert_hand(engine, _burst_card())
        unit = _place_unit(engine, 0, _card(200))
        _unlock_evolution(engine, super_evolve=True)

        engine.apply(SuperEvolve(0, unit.entity_id))

        self.assertEqual(hand_card.evolutions_while_in_hand, 1)

    def test_card_entering_hand_after_evolution_starts_with_zero_bonus(self):
        engine = _engine(_definition())
        unit = _place_unit(engine, 0, _card(200))
        _unlock_evolution(engine)
        engine.apply(Evolve(0, unit.entity_id))

        hand_card = _insert_hand(engine, _burst_card())

        self.assertEqual(hand_card.evolutions_while_in_hand, 0)
        self.assertEqual(
            hand_card.union_burst_gauge(engine.players[0].turns_started),
            engine.players[0].turns_started,
        )

    def test_illegal_evolution_does_not_change_gauge_or_fingerprint(self):
        engine = _engine(_definition())
        hand_card = _insert_hand(engine, _burst_card())
        unit = _place_unit(engine, 0, _card(200))
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(Evolve(0, unit.entity_id))

        self.assertEqual(hand_card.evolutions_while_in_hand, 0)
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_negative_or_boolean_hand_evolution_bonus_fails_invariants(self):
        for invalid in (-1, True):
            with self.subTest(invalid=invalid):
                engine = _engine(_definition())
                hand_card = _insert_hand(engine, _burst_card())
                hand_card.evolutions_while_in_hand = invalid
                with self.assertRaisesRegex(
                    IllegalCommand,
                    "evolutions_while_in_hand",
                ):
                    engine.assert_invariants()


class UnionBurstResolutionTests(unittest.TestCase):
    def test_replacement_suppresses_base_for_followers_spells_and_amulets(self):
        for card_type, trigger in (
            ("随从", Trigger.FANFARE),
            ("法术", Trigger.PLAY),
            ("护符", Trigger.PLAY),
        ):
            with self.subTest(card_type=card_type):
                definition = _card(
                    100,
                    card_type=card_type,
                    attack=1 if card_type == "随从" else None,
                    life=3 if card_type == "随从" else None,
                )
                base = CardRule(
                    100,
                    trigger,
                    (
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            1,
                        ),
                    ),
                )
                replacement = _definition(
                    UnionBurstKind.SUPER_SKYBOUND_ART,
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            2,
                        ),
                    ),
                )
                replacement = UnionBurstDefinition(
                    card_id=replacement.card_id,
                    kind=replacement.kind,
                    operations=replacement.operations,
                    replace_base_operations=True,
                )
                engine = GameEngine(
                    [_card(1000 + index) for index in range(40)],
                    [_card(2000 + index) for index in range(40)],
                    class_a=1,
                    class_b=1,
                    seed=73,
                    rulebook=RuleBook(
                        (base,),
                        union_burst_defs={100: (replacement,)},
                    ),
                    config=GameConfig(validate_invariants=True),
                )
                engine.reset(seed=73)

                _play_burst(
                    engine,
                    turns_started=15,
                    definition=definition,
                )

                self.assertEqual(engine.players[1].health, 18)

    def test_spell_and_amulet_activate_at_exact_threshold(self):
        for card_type in ("法术", "护符"):
            with self.subTest(card_type=card_type):
                definition = _card(
                    100,
                    name=f"Burst-{card_type}",
                    card_type=card_type,
                    attack=None,
                    life=None,
                )
                engine = _engine(_definition())

                _play_burst(
                    engine,
                    turns_started=10,
                    definition=definition,
                )

                self.assertEqual(engine.players[1].health, 18)
                events = [
                    event for event in engine.event_history
                    if event.type is EventType.UNION_BURST_ACTIVATED
                ]
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0].metadata["card_id"], 100)
                if card_type == "法术":
                    self.assertEqual(
                        engine.players[0].graveyard[-1].definition.card_id,
                        100,
                    )
                else:
                    self.assertEqual(
                        engine.players[0].board[-1].definition.card_id,
                        100,
                    )

    def test_below_threshold_does_not_activate(self):
        engine = _engine(_definition())

        _play_burst(engine, turns_started=9)

        self.assertEqual(engine.players[1].health, 20)
        self.assertFalse(
            any(
                event.type is EventType.UNION_BURST_ACTIVATED
                for event in engine.event_history
            )
        )

    def test_exact_threshold_uses_turn_and_hand_evolution_bonus(self):
        engine = _engine(_definition())

        _play_burst(engine, turns_started=7, evolutions=3)

        self.assertEqual(engine.players[1].health, 18)
        events = [
            event
            for event in engine.event_history
            if event.type is EventType.UNION_BURST_ACTIVATED
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].amount, 10)
        self.assertEqual(events[0].metadata["kind"], "union_burst")

    def test_gauge_fifteen_activates_union_then_super_skybound_art(self):
        union = _definition(
            operations=(
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    1,
                ),
            )
        )
        super_art = _definition(
            UnionBurstKind.SUPER_SKYBOUND_ART,
            operations=(
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    2,
                ),
            ),
        )
        engine = _engine(super_art, union)

        _play_burst(engine, turns_started=15)

        self.assertEqual(engine.players[1].health, 17)
        events = [
            event
            for event in engine.event_history
            if event.type is EventType.UNION_BURST_ACTIVATED
        ]
        self.assertEqual(
            [event.metadata["kind"] for event in events],
            ["union_burst", "super_skybound_art"],
        )

    def test_random_enemy_unit_or_leader_uses_leader_when_board_empty(self):
        operation = EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER,
            2,
        )
        engine = _engine(
            _definition(operations=(operation, operation, operation))
        )

        _play_burst(engine, turns_started=10)

        self.assertEqual(engine.players[1].health, 14)

    def test_repeated_random_hits_revalidate_after_unit_dies(self):
        operation = EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER,
            2,
        )
        engine = _engine(
            _definition(operations=(operation,) * 5),
        )
        target = _place_unit(engine, 1, _card(300, life=1))
        engine.random.seed(1)

        _play_burst(engine, turns_started=10)

        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[1].health, 12)
        engine.assert_invariants()

    def test_seeded_random_unit_or_leader_resolution_is_reproducible(self):
        operation = EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER,
            2,
        )
        engines = [
            _engine(_definition(operations=(operation,) * 5))
            for _ in range(2)
        ]
        for engine in engines:
            _place_unit(engine, 1, _card(300, life=7))
            engine.random.seed(1234)
            _play_burst(engine, turns_started=10)

        self.assertEqual(
            engines[0].deterministic_fingerprint(),
            engines[1].deterministic_fingerprint(),
        )


class UnionBurstEnvironmentTests(unittest.TestCase):
    def test_observation_exposes_only_defined_hand_card_gauge(self):
        definition = _definition()
        deck = [_card(3000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=RuleBook(union_burst_defs={100: (definition,)}),
        )
        env.reset(seed=42)
        burst = _insert_hand(env.core, _burst_card(), evolutions=3)
        ordinary = _insert_hand(env.core, _card(101), evolutions=9)
        env.players[0].turns_started = 7

        self.assertEqual(env._card_features(burst)[-3], 10 / 15)
        self.assertEqual(env._card_features(ordinary)[-3], 0.0)
        self.assertEqual(len(env.observation()), 290)


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealSandalphonUnionBurstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def _real_engine(self) -> GameEngine:
        engine = GameEngine(
            [_card(5000 + index) for index in range(40)],
            [_card(6000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=self.rulebook,
            card_resolver=self.repo.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=42)
        return engine

    def test_sandalphon_super_skybound_art_hits_enemy_leader_five_times(self):
        engine = self._real_engine()
        sandalphon = self.repo.get(10404110)
        self.assertIn(AbilityKeyword.UNION_BURST, sandalphon.abilities)

        _play_burst(
            engine,
            turns_started=15,
            definition=sandalphon,
        )

        self.assertEqual(engine.players[1].health, 10)
        events = [
            event
            for event in engine.event_history
            if event.type is EventType.UNION_BURST_ACTIVATED
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].metadata["kind"], "super_skybound_art")
        self.assertFalse(
            any(
                event.card_id == sandalphon.card_id
                and event.ability
                in {
                    AbilityKeyword.FANFARE,
                    AbilityKeyword.EMBLEM,
                    AbilityKeyword.UNION_BURST,
                }
                for event in engine.placeholder_ability_events
            )
        )

    def test_sandalphon_is_exact_in_rule_coverage(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        classification = report["classifications"]["10404110"]
        self.assertEqual(classification["coverage"], "covered_exact")
        self.assertIn("super_skybound_art", classification["reason"])


if __name__ == "__main__":
    unittest.main()
