from __future__ import annotations

import os
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import (
    CardRule,
    InvocationDefinition,
    RuleBook,
    Trigger,
    _parse_invocation_definition,
)
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import (
    Condition,
    ConditionType,
    EffectKind,
    EffectOperation,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Phase, Unit


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


def _invocation_card(card_id: int = 100, *, fanfare: bool = False) -> CardDefinition:
    keywords = {"瞬念召唤"}
    if fanfare:
        keywords.add("入场曲")
    return _card(
        card_id,
        name=f"Invoke-{card_id}",
        cost=6,
        attack=4,
        life=5,
        keywords=frozenset(keywords),
    )


def _invocation_definition(
    card_id: int = 100,
    *,
    threshold: int = 1,
) -> InvocationDefinition:
    return InvocationDefinition(
        card_id=card_id,
        trigger=Trigger.TURN_START,
        conditions=(
            Condition(
                ConditionType.CONTROLLER_EVOLUTIONS_THIS_MATCH_AT_LEAST,
                threshold,
            ),
        ),
    )


def _engine(
    *,
    invocation_definitions: tuple[InvocationDefinition, ...] = (
        _invocation_definition(),
    ),
    rules: tuple[CardRule, ...] = (),
    validate_invariants: bool = True,
) -> GameEngine:
    engine = GameEngine(
        [_card(1000 + index) for index in range(40)],
        [_card(2000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=42,
        rulebook=RuleBook(
            rules=rules,
            invocation_defs={
                definition.card_id: definition
                for definition in invocation_definitions
            },
        ),
        config=GameConfig(validate_invariants=validate_invariants),
    )
    engine.reset(seed=42)
    return engine


def _put_in_deck_for_next_turn(
    engine: GameEngine,
    *cards: CardDefinition,
) -> None:
    engine.players[0].deck.extend(cards)
    engine.players[0].deck.append(_card(9000, name="normal draw"))


def _advance_to_next_own_turn(engine: GameEngine) -> None:
    engine.apply(EndTurn(0))
    engine.apply(EndTurn(1))


def _summon(
    engine: GameEngine,
    player_index: int,
    definition: CardDefinition,
) -> Unit:
    unit = engine._summon_follower_to_board(
        player_index,
        definition,
        summon_cause="test",
    )
    assert unit is not None
    return unit


def _insert_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[0].hand.append(hand_card)
    engine.players[0].hand_entity_ids.append(hand_card.entity_id)
    return hand_card


class InvocationResolutionTests(unittest.TestCase):
    def test_invokes_before_draw_without_cost_play_or_fanfare(self):
        invoked = _invocation_card(fanfare=True)
        fanfare = CardRule(
            invoked.card_id,
            Trigger.FANFARE,
            (
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    5,
                ),
            ),
        )
        engine = _engine(rules=(fanfare,))
        engine.players[0].followers_evolved_this_match = 1
        _put_in_deck_for_next_turn(engine, invoked)

        _advance_to_next_own_turn(engine)

        unit = next(
            entity
            for entity in engine.players[0].board
            if entity.definition.card_id == invoked.card_id
        )
        self.assertIs(unit.origin, CardOrigin.DECK)
        self.assertEqual(engine.players[0].mana, engine.players[0].max_mana)
        self.assertEqual(engine.players[0].cards_played_this_turn, 0)
        self.assertEqual(engine.players[0].cooperation, 1)
        self.assertEqual(engine.players[1].health, 20)
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.FANFARE
                for event in engine.placeholder_ability_events
            )
        )
        event_types = [event.type for event in engine.event_history]
        self.assertLess(
            event_types.index(EventType.CARD_INVOKED),
            event_types.index(EventType.CARD_DRAWN, event_types.index(EventType.CARD_INVOKED)),
        )
        self.assertFalse(
            any(
                event.type is EventType.CARD_PLAYED
                and event.metadata.get("card_id") == invoked.card_id
                for event in engine.event_history
            )
        )

    def test_condition_not_met_leaves_card_in_deck(self):
        invoked = _invocation_card()
        engine = _engine()
        _put_in_deck_for_next_turn(engine, invoked)

        _advance_to_next_own_turn(engine)

        self.assertTrue(
            any(card.card_id == invoked.card_id for card in engine.players[0].deck)
        )
        self.assertFalse(
            any(event.type is EventType.CARD_INVOKED for event in engine.event_history)
        )

    def test_board_full_skips_invocation_and_keeps_card_in_deck(self):
        invoked = _invocation_card()
        engine = _engine()
        engine.players[0].followers_evolved_this_match = 1
        for index in range(engine.config.max_board):
            _summon(engine, 0, _card(300 + index))
        _put_in_deck_for_next_turn(engine, invoked)

        _advance_to_next_own_turn(engine)

        self.assertTrue(
            any(card.card_id == invoked.card_id for card in engine.players[0].deck)
        )
        self.assertFalse(
            any(event.type is EventType.CARD_INVOKED for event in engine.event_history)
        )

    def test_duplicate_copies_invoke_only_one_copy_per_timing(self):
        invoked = _invocation_card()
        engine = _engine()
        engine.players[0].followers_evolved_this_match = 1
        _put_in_deck_for_next_turn(engine, invoked, invoked)

        _advance_to_next_own_turn(engine)

        self.assertEqual(
            sum(entity.definition.card_id == invoked.card_id for entity in engine.players[0].board),
            1,
        )
        self.assertEqual(
            sum(card.card_id == invoked.card_id for card in engine.players[0].deck),
            1,
        )
        self.assertEqual(
            sum(event.type is EventType.CARD_INVOKED for event in engine.event_history),
            1,
        )

    def test_seeded_weighted_candidate_order_uses_duplicate_instances(self):
        lower = _invocation_card(100)
        upper = _invocation_card(101)
        engine = _engine(
            invocation_definitions=(
                _invocation_definition(100),
                _invocation_definition(101),
            ),
        )
        engine.players[0].followers_evolved_this_match = 1
        for index in range(engine.config.max_board - 1):
            _summon(engine, 0, _card(400 + index))
        _put_in_deck_for_next_turn(engine, upper, upper, upper, lower)
        engine.random.seed(1)

        _advance_to_next_own_turn(engine)

        self.assertTrue(
            any(entity.definition.card_id == upper.card_id for entity in engine.players[0].board)
        )
        self.assertTrue(
            any(card.card_id == lower.card_id for card in engine.players[0].deck)
        )

    def test_same_seed_and_deck_order_reproduce_invocation(self):
        def run_once():
            invoked = _invocation_card()
            engine = _engine()
            engine.players[0].followers_evolved_this_match = 1
            _put_in_deck_for_next_turn(engine, invoked, invoked)
            _advance_to_next_own_turn(engine)
            return (
                engine.deterministic_fingerprint(),
                tuple(event.type for event in engine.event_history),
            )

        self.assertEqual(run_once(), run_once())

    def test_on_invoke_choice_resumes_before_normal_draw(self):
        invoked = _invocation_card()
        invoke_rule = CardRule(
            invoked.card_id,
            Trigger.INVOKE,
            (
                EffectOperation(
                    EffectKind.DAMAGE_UNIT,
                    TargetKind.ENEMY_UNIT,
                    2,
                ),
            ),
        )
        engine = _engine(rules=(invoke_rule,))
        engine.players[0].followers_evolved_this_match = 1
        enemy = _summon(engine, 1, _card(500, life=4))
        _put_in_deck_for_next_turn(engine, invoked)
        engine.apply(EndTurn(0))
        deck_before = len(engine.players[0].deck)

        engine.apply(EndTurn(1))

        self.assertIs(engine.state.phase, Phase.AWAITING_CHOICE)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)
        self.assertFalse(
            any(event.type is EventType.TURN_STARTED and event.player_index == 0
                for event in engine.event_history[-3:])
        )
        option = next(
            choice
            for choice in engine.state.pending_choice.options
            if choice.entity_id == enemy.entity_id
        )
        engine.apply(Choose(0, option.option_id))

        self.assertEqual(enemy.health, 2)
        self.assertEqual(len(engine.players[0].deck), deck_before - 2)
        self.assertIs(engine.state.phase, Phase.MAIN)

    def test_non_follower_definition_stays_visible_as_unsupported(self):
        invalid = _card(
            100,
            name="invalid invocation spell",
            card_type="法术",
            attack=None,
            life=None,
            keywords=frozenset({"瞬念召唤"}),
        )
        engine = _engine()
        engine.players[0].followers_evolved_this_match = 1
        _put_in_deck_for_next_turn(engine, invalid)
        _advance_to_next_own_turn(engine)

        self.assertTrue(
            any(card.card_id == invalid.card_id for card in engine.players[0].deck)
        )
        self.assertTrue(
            any(
                event.ability is AbilityKeyword.INVOCATION
                and event.card_id == invalid.card_id
                for event in engine.placeholder_ability_events
            )
        )
        self.assertFalse(
            any(event.type is EventType.CARD_INVOKED for event in engine.event_history)
        )


class EvolutionCounterAndHealingTests(unittest.TestCase):
    def test_normal_and_super_evolution_increment_match_counter(self):
        engine = _engine(invocation_definitions=())
        player = engine.players[0]
        player.turns_started = 10
        normal = _summon(engine, 0, _card(600))

        engine.apply(Evolve(0, normal.entity_id))
        self.assertEqual(player.followers_evolved_this_match, 1)
        self.assertEqual(
            next(event for event in reversed(engine.event_history)
                 if event.type is EventType.FOLLOWER_EVOLVED).metadata["evolutions_this_match"],
            1,
        )

        player.evolved_this_turn = False
        player.super_evolved_this_turn = False
        super_unit = _summon(engine, 0, _card(601))
        engine.apply(SuperEvolve(0, super_unit.entity_id))
        self.assertEqual(player.followers_evolved_this_match, 2)

    def test_heal_unit_all_targets_caps_at_max_and_emits_actual_heal(self):
        spell = _card(
            700,
            name="Group Heal",
            card_type="法术",
            attack=None,
            life=None,
        )
        rule = CardRule(
            spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.HEAL_UNIT,
                    TargetKind.ALL_OWN_UNITS,
                    2,
                ),
            ),
        )
        engine = _engine(invocation_definitions=(), rules=(rule,))
        damaged = _summon(engine, 0, _card(701, life=5))
        full = _summon(engine, 0, _card(702, life=5))
        damaged.health = 2
        _insert_hand(engine, spell)
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))

        self.assertEqual(damaged.health, 4)
        self.assertEqual(full.health, 5)
        healed = [
            event
            for event in engine.event_history
            if event.type is EventType.FOLLOWER_HEALED
        ]
        self.assertEqual([(event.target_id, event.amount) for event in healed], [(damaged.entity_id, 2)])

    def test_public_observation_exposes_both_evolution_totals(self):
        env = ShadowverseEnv(
            [_card(3000 + index) for index in range(40)],
            [_card(4000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=42,
        )
        env.reset(seed=42)
        env.players[0].followers_evolved_this_match = 3
        env.players[1].followers_evolved_this_match = 5

        observation = env.observation()

        self.assertEqual(len(observation), 270)
        self.assertEqual(observation[-10:-8], [0.3, 0.5])

    def test_negative_evolution_counter_fails_invariant(self):
        engine = _engine(invocation_definitions=())
        engine.players[0].followers_evolved_this_match = -1

        with self.assertRaisesRegex(IllegalCommand, "followers_evolved_this_match"):
            engine.assert_invariants()


class InvocationSchemaTests(unittest.TestCase):
    def test_schema_parses_turn_start_evolution_condition(self):
        definition = _parse_invocation_definition(
            {
                "card_id": 100,
                "trigger": "turn_start",
                "conditions": [
                    {
                        "type": "controller_evolutions_this_match_at_least",
                        "value": 6,
                    }
                ],
            },
            "test/invocations[0]",
        )

        self.assertEqual(definition.card_id, 100)
        self.assertIs(definition.trigger, Trigger.TURN_START)
        self.assertEqual(definition.conditions[0].value, 6)

    def test_schema_rejects_bad_trigger_and_source_or_target_conditions(self):
        invalid = (
            {"card_id": 100, "trigger": "turn_end"},
            {"card_id": 100, "conditions": [{"type": "source_evolved"}]},
            {
                "card_id": 100,
                "conditions": [{"type": "target_health_at_least", "value": 1}],
            },
            {"card_id": True},
            {"card_id": 100, "unknown": 1},
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_invocation_definition(raw, "test/invocations[0]")


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealSandalphonInvocationTests(unittest.TestCase):
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

    def test_sandalphon_invokes_gains_crest_returns_and_crest_heals(self):
        engine = self._real_engine()
        sandalphon = self.repo.get(10404110)
        self.assertIn(AbilityKeyword.INVOCATION, sandalphon.abilities)
        player = engine.players[0]
        player.followers_evolved_this_match = 6
        player.health = 15
        ally = _summon(engine, 0, _card(800, life=5))
        ally.health = 2
        _put_in_deck_for_next_turn(engine, sandalphon)

        _advance_to_next_own_turn(engine)

        self.assertFalse(
            any(entity.definition.card_id == sandalphon.card_id for entity in player.board)
        )
        self.assertTrue(
            any(card.card_id == sandalphon.card_id for card in player.hand)
        )
        self.assertEqual(len(player.emblems), 1)
        self.assertEqual(player.emblems[0].countdown, 2)
        self.assertTrue(
            any(event.type is EventType.CARD_INVOKED for event in engine.event_history)
        )
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.INVOCATION
                for event in engine.placeholder_ability_events
            )
        )

        engine.apply(EndTurn(0))

        self.assertEqual(player.health, 16)
        self.assertEqual(ally.health, 3)

    def test_sandalphon_rule_is_exact_after_union_burst_slice(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10404110"]

        self.assertEqual(info["coverage"], "covered_exact")
        self.assertIn("super_skybound_art", info["reason"])
        self.assertNotIn("INVOCATION", " ".join(info["missing_primitives"]))


if __name__ == "__main__":
    unittest.main()
