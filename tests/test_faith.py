from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import (
    CardRule,
    RuleBook,
    Trigger,
    _parse_faith_definition,
    _parse_operation,
)
from swb.engine.commands import Choose, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.faith import (
    FaithAbilityStacking,
    FaithDefinition,
    FaithGrantedAbility,
    FaithTrigger,
    FaithTriggerRule,
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
        "attack": 1,
        "life": 3,
        "keywords": frozenset(),
        "support_level": "basic",
        "is_collectible": True,
    }
    values.update(overrides)
    return CardDefinition(**values)


def _faith_card(card_id: int = 100) -> CardDefinition:
    return _card(
        card_id,
        name=f"Faith-{card_id}",
        keywords=frozenset({"信仰"}),
    )


def _faith_definition(
    card_id: int = 100,
    *,
    faith_id: str = "faith-test",
    initial_value: int = 0,
    amount: int = 1,
) -> FaithDefinition:
    return FaithDefinition(
        faith_id=faith_id,
        source_card_id=card_id,
        initial_value=initial_value,
        triggers=(
            FaithTriggerRule(FaithTrigger.FOLLOWER_EVOLVED, amount),
        ),
    )


def _deck(*faith_cards: CardDefinition) -> list[CardDefinition]:
    return [
        *faith_cards,
        *[
            _card(1000 + index)
            for index in range(40 - len(faith_cards))
        ],
    ]


def _engine(
    deck_a: list[CardDefinition],
    *,
    definitions: tuple[FaithDefinition, ...],
    deck_b: list[CardDefinition] | None = None,
    rules: tuple[CardRule, ...] = (),
) -> GameEngine:
    engine = GameEngine(
        deck_a,
        deck_b or _deck(),
        class_a=1,
        class_b=1,
        seed=42,
        rulebook=RuleBook(
            rules=rules,
            faith_defs={
                definition.source_card_id: definition
                for definition in definitions
            }
        ),
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=42)
    return engine


def _place_unit(
    engine: GameEngine,
    player_index: int,
    *,
    card_id: int,
    definition: CardDefinition | None = None,
) -> Unit:
    unit = Unit.summon(
        definition or _card(card_id),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[player_index].board.append(unit)
    return unit


def _unlock_evolution(engine: GameEngine, player_index: int) -> None:
    player = engine.players[player_index]
    player.turns_started = max(
        player.turns_started,
        engine.config.first_player_super_evolution_unlock_turn
        if player_index == 0
        else engine.config.second_player_super_evolution_unlock_turn,
    )


def _insert_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    origin: CardOrigin = CardOrigin.DECK,
) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=origin,
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


class FaithConsumptionTests(unittest.TestCase):
    @staticmethod
    def _rule() -> CardRule:
        return CardRule(
            100,
            Trigger.FANFARE,
            (
                EffectOperation(
                    EffectKind.CONSUME_FAITH,
                    TargetKind.OWN_LEADER,
                    amount=10,
                    faith_id="faith-test",
                    faith_operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            amount=2,
                        ),
                    ),
                ),
            ),
        )

    def test_full_cost_is_consumed_before_nested_operations(self):
        source = _faith_card()
        game = _engine(
            _deck(source),
            definitions=(_faith_definition(initial_value=10),),
            rules=(self._rule(),),
        )
        _insert_hand(game, source)
        game.players[0].max_mana = game.players[0].mana = 10

        game.apply(PlayCard(0, 0))

        self.assertEqual(game.players[0].faiths[0].value, 0)
        self.assertEqual(game.players[1].health, 18)
        relevant = [
            event.type
            for event in game.event_history
            if event.type in {
                EventType.FAITH_CONSUMED,
                EventType.FAITH_VALUE_CHANGED,
                EventType.DAMAGE_APPLIED,
            }
        ]
        self.assertLess(
            relevant.index(EventType.FAITH_CONSUMED),
            relevant.index(EventType.DAMAGE_APPLIED),
        )
        changed = next(
            event for event in game.event_history
            if event.type is EventType.FAITH_VALUE_CHANGED
            and event.metadata.get("change") == "spend"
        )
        self.assertEqual(changed.amount, -10)

    def test_insufficient_value_skips_entire_nested_payoff_without_clamping(self):
        source = _faith_card()
        game = _engine(
            _deck(source),
            definitions=(_faith_definition(initial_value=9),),
            rules=(self._rule(),),
        )
        _insert_hand(game, source)
        game.players[0].max_mana = game.players[0].mana = 10

        game.apply(PlayCard(0, 0))

        self.assertEqual(game.players[0].faiths[0].value, 9)
        self.assertEqual(game.players[1].health, 20)
        failed = next(
            event for event in game.event_history
            if event.type is EventType.FAITH_CONSUME_FAILED
        )
        self.assertEqual(failed.metadata["reason"], "insufficient")
        self.assertEqual(failed.metadata["faith_value"], 9)

    def test_missing_named_faith_skips_payoff_deterministically(self):
        source = _card(100)
        game = _engine(
            _deck(),
            definitions=(),
            rules=(self._rule(),),
        )
        _insert_hand(game, source)
        game.players[0].max_mana = game.players[0].mana = 10

        game.apply(PlayCard(0, 0))

        self.assertEqual(game.players[1].health, 20)
        failed = next(
            event for event in game.event_history
            if event.type is EventType.FAITH_CONSUME_FAILED
        )
        self.assertEqual(failed.metadata["reason"], "missing")

    def test_schema_requires_positive_cost_named_faith_and_nested_operations(self):
        operation = _parse_operation(
            {
                "kind": "consume_faith",
                "target": "own_leader",
                "faith_id": "faith-test",
                "amount": 10,
                "operations": [
                    {"kind": "draw", "target": "own_leader", "amount": 1}
                ],
            },
            "test",
            100,
        )
        self.assertEqual(operation.faith_id, "faith-test")
        self.assertEqual(operation.faith_operations[0].kind, EffectKind.DRAW)
        for invalid in (
            {"kind": "consume_faith", "target": "own_leader", "amount": 10, "operations": []},
            {"kind": "consume_faith", "target": "own_leader", "faith_id": "x", "amount": 0, "operations": [{"kind": "draw", "target": "own_leader", "amount": 1}]},
            {"kind": "consume_faith", "target": "enemy_leader", "faith_id": "x", "amount": 1, "operations": [{"kind": "draw", "target": "own_leader", "amount": 1}]},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _parse_operation(invalid, "test", 100)


class FaithDynamicAbilityTests(unittest.TestCase):
    @staticmethod
    def _grant_rule(
        *,
        stacking: str = "unique",
        operations: tuple[EffectOperation, ...] | None = None,
    ) -> CardRule:
        return CardRule(
            100,
            Trigger.FANFARE,
            (
                EffectOperation(
                    EffectKind.GRANT_FAITH_ABILITY,
                    TargetKind.OWN_LEADER,
                    faith_id="faith-test",
                    faith_ability_id="evolve-payoff",
                    faith_trigger=FaithTrigger.FOLLOWER_EVOLVED.value,
                    faith_stacking=stacking,
                    faith_operations=operations or (
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )

    def _grant_then_evolve(
        self,
        *,
        stacking: str = "unique",
        copies: int = 1,
        operations: tuple[EffectOperation, ...] | None = None,
    ) -> GameEngine:
        source = _faith_card()
        game = _engine(
            _deck(source),
            definitions=(_faith_definition(),),
            rules=(self._grant_rule(stacking=stacking, operations=operations),),
        )
        game.players[0].hand.clear()
        game.players[0].hand_entity_ids.clear()
        for _ in range(copies):
            _insert_hand(game, source)
        game.players[0].max_mana = game.players[0].mana = 10
        for _ in range(copies):
            game.apply(PlayCard(0, 0))
        evolving = _place_unit(game, 0, card_id=401)
        _unlock_evolution(game, 0)
        game.apply(Evolve(0, evolving.entity_id))
        return game

    def test_granted_evolution_ability_fires_after_faith_progression(self):
        game = self._grant_then_evolve()
        faith = game.players[0].faiths[0]

        self.assertEqual(faith.value, 1)
        self.assertEqual(game.players[1].health, 19)
        self.assertEqual(len(faith.granted_abilities), 1)
        value_event_index = next(
            index for index, event in enumerate(game.event_history)
            if event.type is EventType.FAITH_VALUE_CHANGED
        )
        trigger_event_index = next(
            index for index, event in enumerate(game.event_history)
            if event.type is EventType.FAITH_ABILITY_TRIGGERED
        )
        self.assertLess(value_event_index, trigger_event_index)

    def test_unique_grant_deduplicates_but_allow_stacks_in_grant_order(self):
        unique = self._grant_then_evolve(copies=2)
        allowed = self._grant_then_evolve(
            copies=2,
            stacking=FaithAbilityStacking.ALLOW.value,
        )

        self.assertEqual(len(unique.players[0].faiths[0].granted_abilities), 1)
        self.assertEqual(unique.players[1].health, 19)
        self.assertEqual(len(allowed.players[0].faiths[0].granted_abilities), 2)
        self.assertEqual(allowed.players[1].health, 18)
        self.assertEqual(
            [
                ability.granted_sequence
                for ability in allowed.players[0].faiths[0].granted_abilities
            ],
            [1, 2],
        )

    def test_pending_choice_pauses_after_progress_and_resumes_event(self):
        operations = (
            EffectOperation(
                EffectKind.BUFF_UNIT,
                TargetKind.OWN_UNIT,
                amount=1,
                secondary_amount=1,
                requires_target=True,
            ),
        )
        game = self._grant_then_evolve(operations=operations)

        self.assertIsNotNone(game.state.pending_choice)
        self.assertEqual(game.players[0].faiths[0].value, 1)
        self.assertEqual(game._suspended_event_state["phase"], "faith_done")
        target = game.players[0].board[0]
        before = (target.attack, target.health)
        game.apply(Choose(0, f"entity:{target.entity_id}"))
        self.assertEqual((target.attack, target.health), (before[0] + 1, before[1] + 1))
        self.assertIsNone(game._suspended_event_state)

    def test_granted_ability_state_is_fingerprinted_and_invariant_checked(self):
        source = _faith_card()
        game = _engine(
            _deck(source),
            definitions=(_faith_definition(),),
            rules=(self._grant_rule(),),
        )
        before = game.deterministic_fingerprint()
        game.players[0].faiths[0].granted_abilities.append(
            FaithGrantedAbility(
                "manual",
                FaithTrigger.FOLLOWER_EVOLVED,
                (EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),),
                1,
            )
        )
        game.players[0].faiths[0]._next_granted_ability_sequence = 2
        self.assertNotEqual(before, game.deterministic_fingerprint())
        game.assert_invariants()

    def test_schema_parses_dynamic_faith_ability_and_stacking(self):
        operation = _parse_operation(
            {
                "kind": "grant_faith_ability",
                "target": "own_leader",
                "faith_id": "faith-test",
                "ability_id": "evolve-payoff",
                "faith_trigger": "follower_evolved",
                "stacking": "allow",
                "operations": [
                    {
                        "kind": "damage_leader",
                        "target": "enemy_leader",
                        "amount": 1,
                    }
                ],
            },
            "test",
            100,
        )
        self.assertEqual(operation.faith_ability_id, "evolve-payoff")
        self.assertEqual(operation.faith_stacking, "allow")
        self.assertEqual(operation.faith_operations[0].kind, EffectKind.DAMAGE_LEADER)


class FaithInitializationTests(unittest.TestCase):
    def test_initial_deck_copies_place_one_faith_without_removing_cards(self):
        source = _faith_card()
        engine = _engine(
            _deck(source, source, source),
            definitions=(_faith_definition(),),
        )

        self.assertEqual(len(engine.players[0].faiths), 1)
        faith = engine.players[0].faiths[0]
        self.assertEqual(faith.faith_id, "faith-test")
        self.assertEqual(faith.value, 0)
        self.assertEqual(
            len(engine.players[0].deck) + len(engine.players[0].hand),
            40,
        )
        self.assertEqual(
            sum(
                card.card_id == source.card_id
                for card in [
                    *engine.players[0].deck,
                    *(hand.definition for hand in engine.players[0].hand),
                ]
            ),
            3,
        )
        placed = [
            event for event in engine.event_history
            if event.type is EventType.FAITH_PLACED
            and event.player_index == 0
        ]
        self.assertEqual(len(placed), 1)
        self.assertEqual(placed[0].metadata["source_card_id"], source.card_id)

    def test_same_named_faith_definitions_are_deduplicated(self):
        first = _faith_card(100)
        second = _faith_card(101)
        engine = _engine(
            _deck(first, second),
            definitions=(
                _faith_definition(100, faith_id="same-faith"),
                _faith_definition(101, faith_id="same-faith"),
            ),
        )

        self.assertEqual(len(engine.players[0].faiths), 1)
        self.assertEqual(engine.players[0].faiths[0].source_card_id, 100)

    def test_faith_not_present_in_initial_deck_is_not_created(self):
        engine = _engine(
            _deck(),
            definitions=(_faith_definition(),),
        )
        engine.players[0].hand.append(
            HandCard(
                definition=_faith_card(),
                entity_id=engine.state.allocate_entity_id(),
                origin=CardOrigin.GENERATED,
            )
        )
        engine.players[0].hand_entity_ids.append(
            engine.players[0].hand[-1].entity_id
        )

        self.assertEqual(engine.players[0].faiths, [])

    def test_reset_recreates_initial_value_and_identity_deterministically(self):
        source = _faith_card()
        definition = _faith_definition(initial_value=2)
        engine = _engine(_deck(source), definitions=(definition,))
        first = engine.deterministic_fingerprint()
        engine.players[0].faiths[0].value = 9

        engine.reset(seed=42)

        self.assertEqual(engine.players[0].faiths[0].value, 2)
        self.assertEqual(engine.deterministic_fingerprint(), first)


class FaithEvolutionTriggerTests(unittest.TestCase):
    def test_illegal_evolution_does_not_increment_faith_or_mutate_state(self):
        source = _faith_card()
        engine = _engine(
            _deck(source),
            definitions=(_faith_definition(),),
        )
        unit = _place_unit(engine, 0, card_id=299)
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(Evolve(0, unit.entity_id))

        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_normal_evolution_increments_own_faith_and_emits_event(self):
        source = _faith_card()
        engine = _engine(
            _deck(source),
            definitions=(_faith_definition(amount=2),),
        )
        unit = _place_unit(engine, 0, card_id=300)
        _unlock_evolution(engine, 0)

        transition = engine.apply(Evolve(0, unit.entity_id))

        self.assertEqual(engine.players[0].faiths[0].value, 2)
        event = next(
            event for event in transition.events
            if event.type is EventType.FAITH_VALUE_CHANGED
        )
        self.assertEqual(event.amount, 2)
        self.assertEqual(event.metadata["faith_value_before"], 0)
        self.assertEqual(event.metadata["faith_value_after"], 2)
        self.assertFalse(event.metadata["super_evolution"])

    def test_super_evolution_counts_as_follower_evolution(self):
        source = _faith_card()
        engine = _engine(
            _deck(source),
            definitions=(_faith_definition(),),
        )
        unit = _place_unit(engine, 0, card_id=301)
        _unlock_evolution(engine, 0)

        transition = engine.apply(SuperEvolve(0, unit.entity_id))

        self.assertEqual(engine.players[0].faiths[0].value, 1)
        event = next(
            event for event in transition.events
            if event.type is EventType.FAITH_VALUE_CHANGED
        )
        self.assertTrue(event.metadata["super_evolution"])

    def test_faith_increments_before_evolve_trigger_pending_choice(self):
        source = _faith_card()
        evolving = _card(304, keywords=frozenset({"进化时"}))
        engine = _engine(
            _deck(source),
            definitions=(_faith_definition(),),
            rules=(
                CardRule(
                    304,
                    Trigger.EVOLVE,
                    (
                        EffectOperation(
                            EffectKind.BUFF_UNIT,
                            TargetKind.OWN_UNIT,
                            1,
                            1,
                            requires_target=True,
                        ),
                    ),
                ),
            ),
        )
        unit = _place_unit(
            engine,
            0,
            card_id=304,
            definition=evolving,
        )
        target = _place_unit(engine, 0, card_id=305)
        _unlock_evolution(engine, 0)

        engine.apply(Evolve(0, unit.entity_id))

        self.assertEqual(engine.players[0].faiths[0].value, 1)
        self.assertIsNotNone(engine.state.pending_choice)
        engine.apply(Choose(0, f"entity:{target.entity_id}"))
        self.assertEqual((target.attack, target.health), (2, 4))

    def test_opponents_evolution_does_not_increment_other_players_faith(self):
        source = _faith_card()
        engine = _engine(
            _deck(source),
            definitions=(_faith_definition(),),
        )
        engine.state.active_player = 1
        unit = _place_unit(engine, 1, card_id=302)
        _unlock_evolution(engine, 1)

        engine.apply(Evolve(1, unit.entity_id))

        self.assertEqual(engine.players[0].faiths[0].value, 0)

    def test_invalid_negative_faith_value_is_caught_by_invariants(self):
        source = _faith_card()
        engine = _engine(
            _deck(source),
            definitions=(_faith_definition(),),
        )
        engine.players[0].faiths[0].value = -1

        with self.assertRaisesRegex(IllegalCommand, r"faiths\[0\].*value"):
            engine.assert_invariants()


class FaithSchemaTests(unittest.TestCase):
    def test_schema_parses_evolution_trigger(self):
        definition = _parse_faith_definition(
            {
                "id": "faith-test",
                "source_card_id": 123,
                "initial_value": 2,
                "triggers": [
                    {"trigger": "follower_evolved", "amount": 3},
                ],
                "coverage": "partial",
            },
            "test",
        )

        self.assertEqual(definition.initial_value, 2)
        self.assertEqual(
            definition.triggers,
            (FaithTriggerRule(FaithTrigger.FOLLOWER_EVOLVED, 3),),
        )

    def test_schema_rejects_unknown_or_non_positive_trigger_amount(self):
        invalid = (
            {
                "id": "faith-test",
                "source_card_id": 123,
                "triggers": [{"trigger": "mode_selected"}],
            },
            {
                "id": "faith-test",
                "source_card_id": 123,
                "triggers": [{"trigger": "follower_evolved", "amount": 0}],
            },
            {
                "id": "faith-test",
                "source_card_id": 123,
                "triggers": [],
            },
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_faith_definition(raw, "test")

    def test_rulebook_rejects_duplicate_source_card_definitions(self):
        first = {
            "faiths": [
                {
                    "id": "first",
                    "source_card_id": 100,
                    "triggers": [{"trigger": "follower_evolved"}],
                }
            ]
        }
        second = {
            "faiths": [
                {
                    "id": "second",
                    "source_card_id": 100,
                    "triggers": [{"trigger": "follower_evolved"}],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "a.json").write_text(
                json.dumps(first),
                encoding="utf-8",
            )
            Path(directory, "b.json").write_text(
                json.dumps(second),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                RuleBook.from_directory(directory)


class FaithEnvironmentTests(unittest.TestCase):
    def test_observation_exposes_public_faith_count_and_value(self):
        source = _faith_card()
        definition = _faith_definition()
        deck = _deck(source)
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=RuleBook(faith_defs={100: definition}),
        )
        env.reset(seed=42)

        self.assertEqual(len(env.observation()), 290)
        self.assertEqual(env.observation()[-4:], [0.2, 0.2, 0.0, 0.0])

        unit = _place_unit(env.core, 0, card_id=303)
        _unlock_evolution(env.core, 0)
        env.core.apply(Evolve(0, unit.entity_id))

        self.assertEqual(env.observation()[-4:], [0.2, 0.2, 0.02, 0.0])


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealFaithCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def _engine(self) -> GameEngine:
        source = self.repo.get(10614120)
        deck = [
            source,
            source,
            source,
            *[_card(7000 + index) for index in range(37)],
        ]
        engine = GameEngine(
            deck,
            _deck(),
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=self.rulebook,
            card_resolver=self.repo.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=42)
        return engine

    def test_ancient_heavenspear_faith_starts_at_zero_and_tracks_evolution(self):
        engine = self._engine()
        self.assertEqual(len(engine.players[0].faiths), 1)
        faith = engine.players[0].faiths[0]
        self.assertEqual(faith.faith_id, "ancient_heavenspear")
        self.assertEqual(faith.value, 0)
        unit = _place_unit(engine, 0, card_id=8000)
        _unlock_evolution(engine, 0)

        engine.apply(Evolve(0, unit.entity_id))

        self.assertEqual(faith.value, 1)

    def test_defined_faith_does_not_emit_faith_placeholder_on_play(self):
        engine = self._engine()
        source = self.repo.get(10614120)
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        hand = HandCard(
            definition=source,
            entity_id=engine.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        engine.players[0].hand.append(hand)
        engine.players[0].hand_entity_ids.append(hand.entity_id)
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertFalse(
            any(
                event.ability is AbilityKeyword.FAITH
                for event in engine.placeholder_ability_events
            )
        )

    def test_sasanid_pays_ten_and_generates_abyssal_lance_with_token_origin(self):
        engine = self._engine()
        source = self.repo.get(10614120)
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        _insert_hand(engine, source)
        engine.players[0].max_mana = engine.players[0].mana = 10
        faith = engine.players[0].faiths[0]
        faith.value = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(faith.value, 0)
        lance = next(
            card for card in engine.players[0].hand
            if card.card_id == 90014330
        )
        self.assertIs(lance.origin, CardOrigin.TOKEN)
        added = next(
            event for event in engine.event_history
            if event.type is EventType.CARD_ADDED_TO_HAND
            and event.metadata["card_id"] == 90014330
        )
        self.assertEqual(added.metadata["origin"], CardOrigin.TOKEN.value)
        self.assertTrue(added.metadata["derived"])
        self.assertTrue(added.metadata["token"])

    def test_sasanid_insufficient_faith_adds_no_lance_and_keeps_value(self):
        engine = self._engine()
        source = self.repo.get(10614120)
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        _insert_hand(engine, source)
        engine.players[0].max_mana = engine.players[0].mana = 10
        faith = engine.players[0].faiths[0]
        faith.value = 9

        engine.apply(PlayCard(0, 0))

        self.assertEqual(faith.value, 9)
        self.assertFalse(
            any(card.card_id == 90014330 for card in engine.players[0].hand)
        )
        self.assertTrue(
            any(
                event.type is EventType.FAITH_CONSUME_FAILED
                and event.metadata["reason"] == "insufficient"
                for event in engine.event_history
            )
        )

    def test_abyssal_lance_evolves_selected_unevolved_follower(self):
        engine = self._engine()
        lance = self.repo.get(90014330)
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        _insert_hand(engine, lance, origin=CardOrigin.TOKEN)
        engine.players[0].max_mana = engine.players[0].mana = 10
        target = _place_unit(engine, 0, card_id=8099)
        ep_before = engine.players[0].evolution_points

        engine.apply(PlayCard(0, 0))
        request = engine.state.pending_choice
        engine.apply(Choose(0, request.options[0].option_id))

        self.assertTrue(target.evolved)
        self.assertFalse(target.super_evolved)
        self.assertEqual(engine.players[0].evolution_points, ep_before)
        event = next(
            event for event in engine.event_history
            if event.type is EventType.FOLLOWER_EVOLVED
            and event.source_id == target.entity_id
        )
        self.assertEqual(event.metadata["cause"], "effect")

    def test_sasanid_grants_stacking_evolution_damage_ability(self):
        engine = self._engine()
        source = self.repo.get(10614120)
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        _insert_hand(engine, source)
        engine.players[0].max_mana = engine.players[0].mana = 10
        faith = engine.players[0].faiths[0]
        faith.value = 10
        engine.apply(PlayCard(0, 0))
        evolving = _place_unit(engine, 0, card_id=8100)
        _unlock_evolution(engine, 0)

        engine.apply(Evolve(0, evolving.entity_id))

        self.assertEqual(faith.value, 1)
        self.assertEqual(engine.players[1].health, 19)
        self.assertEqual(
            [ability.ability_id for ability in faith.granted_abilities],
            ["sasanid_evolve_leader_damage"],
        )

    def test_sasanid_repeated_grants_trigger_in_grant_order(self):
        engine = self._engine()
        source = self.repo.get(10614120)
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        _insert_hand(engine, source)
        _insert_hand(engine, source)
        engine.players[0].max_mana = engine.players[0].mana = 10
        faith = engine.players[0].faiths[0]
        faith.value = 20
        engine.apply(PlayCard(0, 0))
        engine.apply(PlayCard(0, 0))
        evolving = _place_unit(engine, 0, card_id=8101)
        _unlock_evolution(engine, 0)

        engine.apply(Evolve(0, evolving.entity_id))

        self.assertEqual(len(faith.granted_abilities), 2)
        self.assertEqual(engine.players[1].health, 18)
        triggered = [
            event.metadata["granted_sequence"]
            for event in engine.event_history
            if event.type is EventType.FAITH_ABILITY_TRIGGERED
        ]
        self.assertEqual(triggered, [1, 2])

    def test_coverage_marks_complete_heavenspear_rule_exact(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        classification = report["classifications"]["10614120"]

        self.assertEqual(classification["coverage"], "covered_exact")
        self.assertNotIn("信仰", classification["missing_primitives"])
        self.assertNotIn("unsupported_text", classification.get("rule_metadata", {}))


if __name__ == "__main__":
    unittest.main()
