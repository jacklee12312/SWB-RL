from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_faith_definition
from swb.engine.commands import Choose, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.faith import FaithDefinition, FaithTrigger, FaithTriggerRule
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

        self.assertEqual(len(env.observation()), 261)
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

    def test_coverage_keeps_unimplemented_heavenspear_payoff_partial(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        classification = report["classifications"]["10614120"]

        self.assertEqual(classification["coverage"], "covered_partial")
        self.assertNotIn("信仰", classification["missing_primitives"])
        self.assertIn("unsupported_text", classification["rule_metadata"])


if __name__ == "__main__":
    unittest.main()
