from __future__ import annotations

import os
import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
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
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import Amulet, HandCard, Unit
from swb.engine.targeting import target_candidates


def _card(card_id: int, **overrides) -> CardDefinition:
    values = {
        "card_id": card_id,
        "card_set_id": 10000,
        "class_id": 3,
        "class_name": "巫师",
        "name": f"card-{card_id}",
        "cost": 1,
        "card_type": "随从",
        "attack": 1,
        "life": 1,
        "keywords": frozenset(),
        "support_level": "basic",
        "is_collectible": True,
    }
    values.update(overrides)
    return CardDefinition(**values)


def _earth_sigil(card_id: int = 90031210, **overrides) -> CardDefinition:
    values = {
        "name": "大地之魔片",
        "card_set_id": 90000,
        "card_type": "护符",
        "attack": None,
        "life": None,
        "keywords": frozenset({"土之印"}),
        "is_collectible": False,
    }
    values.update(overrides)
    return _card(card_id, **values)


def _engine(
    *rules: CardRule,
    definitions: dict[int, CardDefinition] | None = None,
    validate_invariants: bool = True,
) -> GameEngine:
    defs = {90031210: _earth_sigil()}
    defs.update(definitions or {})
    engine = GameEngine(
        [_card(1000 + index) for index in range(40)],
        [_card(2000 + index) for index in range(40)],
        class_a=3,
        class_b=3,
        seed=42,
        rulebook=RuleBook(rules),
        card_resolver=lambda card_id: defs.get(card_id),
        config=GameConfig(validate_invariants=validate_invariants),
    )
    engine.reset(seed=42)
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _insert(engine: GameEngine, definition: CardDefinition) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


def _sigil_entity(
    engine: GameEngine,
    *,
    player_index: int = 0,
    count: int = 1,
    definition: CardDefinition | None = None,
) -> Amulet:
    amulet = Amulet(
        definition=definition or _earth_sigil(),
        entity_id=engine.state.allocate_entity_id(),
        earth_sigil_count=count,
        entered_turn=engine.turn,
        origin=CardOrigin.TOKEN,
    )
    engine.players[player_index].board.append(amulet)
    return amulet


class EarthSigilStateTests(unittest.TestCase):
    def test_entry_starts_at_one_and_merges_existing_sigils_by_banishing(self):
        furnace = _earth_sigil(
            10031210,
            name="魔女的炼金炉",
            card_set_id=10000,
            is_collectible=True,
        )
        engine = _engine(
            CardRule(
                10031210,
                Trigger.PLAY,
                (EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),),
            ),
            definitions={10031210: furnace},
        )
        old = _sigil_entity(engine, count=3)
        _insert(engine, furnace)

        engine.apply(PlayCard(0, 0))

        sigils = engine._earth_sigil_amulets(0)
        self.assertEqual(len(sigils), 1)
        self.assertNotEqual(sigils[0].entity_id, old.entity_id)
        self.assertEqual(sigils[0].earth_sigil_count, 4)
        self.assertIn(old.definition, engine.players[0].banished)
        self.assertFalse(
            any(card.entity_id == old.entity_id for card in engine.players[0].graveyard)
        )
        merged = [
            event
            for event in engine.event_history
            if event.type is EventType.EARTH_SIGILS_MERGED
        ]
        self.assertEqual(merged[-1].metadata["merged_entity_ids"], (old.entity_id,))

    def test_add_sigils_creates_magic_sediment_with_requested_count(self):
        spell = _card(100, name="Add Sigils", card_type="法术", attack=None, life=None)
        rule = CardRule(
            100,
            Trigger.PLAY,
            (EffectOperation(EffectKind.ADD_EARTH_SIGILS, TargetKind.OWN_LEADER, 2),),
        )
        engine = _engine(rule, definitions={100: spell})
        _insert(engine, spell)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].earth_sigils, 2)
        sigil = engine._earth_sigil_amulets(0)[0]
        self.assertEqual(sigil.definition.card_id, 90031210)
        self.assertIs(sigil.origin, CardOrigin.TOKEN)

    def test_add_sigils_increments_existing_stack(self):
        spell = _card(101, name="Add Sigils", card_type="法术", attack=None, life=None)
        rule = CardRule(
            101,
            Trigger.PLAY,
            (EffectOperation(EffectKind.ADD_EARTH_SIGILS, TargetKind.OWN_LEADER, 2),),
        )
        engine = _engine(rule, definitions={101: spell})
        sigil = _sigil_entity(engine, count=3)
        _insert(engine, spell)

        engine.apply(PlayCard(0, 0))

        self.assertIs(engine._earth_sigil_amulets(0)[0], sigil)
        self.assertEqual(sigil.earth_sigil_count, 5)

    def test_add_sigils_without_stack_fails_cleanly_on_full_board(self):
        spell = _card(102, name="Add Sigils", card_type="法术", attack=None, life=None)
        rule = CardRule(
            102,
            Trigger.PLAY,
            (EffectOperation(EffectKind.ADD_EARTH_SIGILS, TargetKind.OWN_LEADER, 2),),
        )
        engine = _engine(rule, definitions={102: spell})
        engine.players[0].board = [
            Unit.summon(_card(300 + index), entity_id=engine.state.allocate_entity_id())
            for index in range(5)
        ]
        _insert(engine, spell)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].earth_sigils, 0)
        self.assertEqual(len(engine.players[0].board), 5)

    def test_opponent_cannot_select_earth_sigil_and_illegal_play_is_atomic(self):
        spell = _card(103, name="Destroy", card_type="法术", attack=None, life=None)
        operation = EffectOperation(
            EffectKind.DESTROY,
            TargetKind.ENEMY_AMULET,
            requires_target=True,
        )
        rule = CardRule(103, Trigger.PLAY, (operation,))
        engine = _engine(rule, definitions={103: spell})
        sigil = _sigil_entity(engine, player_index=1, count=2)
        _insert(engine, spell)

        self.assertNotIn(sigil, target_candidates(operation, 0, engine.players))
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_owner_can_select_own_earth_sigil(self):
        engine = _engine()
        sigil = _sigil_entity(engine, count=2)
        operation = EffectOperation(EffectKind.BANISH, TargetKind.OWN_AMULET)
        self.assertIn(sigil, target_candidates(operation, 0, engine.players))

    def test_effect_destroy_is_prevented_but_nonselecting_banish_works(self):
        destroy_spell = _card(104, name="Destroy All", card_type="法术", attack=None, life=None)
        destroy_rule = CardRule(
            104,
            Trigger.PLAY,
            (EffectOperation(EffectKind.DESTROY, TargetKind.ALL_ENEMY_AMULETS),),
        )
        engine = _engine(destroy_rule, definitions={104: destroy_spell})
        sigil = _sigil_entity(engine, player_index=1, count=2)
        _insert(engine, destroy_spell)
        engine.apply(PlayCard(0, 0))
        self.assertIn(sigil, engine.players[1].board)
        self.assertTrue(
            any(
                event.type is EventType.EARTH_SIGIL_DESTROY_PREVENTED
                for event in engine.event_history
            )
        )

        banish_spell = _card(105, name="Banish All", card_type="法术", attack=None, life=None)
        banish_rule = CardRule(
            105,
            Trigger.PLAY,
            (EffectOperation(EffectKind.BANISH, TargetKind.ALL_ENEMY_AMULETS),),
        )
        engine = _engine(banish_rule, definitions={105: banish_spell})
        sigil = _sigil_entity(engine, player_index=1, count=2)
        _insert(engine, banish_spell)
        engine.apply(PlayCard(0, 0))
        self.assertNotIn(sigil, engine.players[1].board)
        self.assertIn(sigil.definition, engine.players[1].banished)


class EarthRiteResolutionTests(unittest.TestCase):
    def _rite_spell(self, card_id: int, cost: int) -> tuple[CardDefinition, CardRule]:
        spell = _card(card_id, name="Earth Rite", card_type="法术", attack=None, life=None)
        rule = CardRule(
            card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.EARTH_RITE,
                    TargetKind.OWN_LEADER,
                    cost,
                    earth_rite_operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            3,
                        ),
                    ),
                ),
            ),
        )
        return spell, rule

    def test_insufficient_sigils_skip_without_consuming(self):
        spell, rule = self._rite_spell(110, 2)
        engine = _engine(rule, definitions={110: spell})
        sigil = _sigil_entity(engine, count=1)
        _insert(engine, spell)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(sigil.earth_sigil_count, 1)
        self.assertEqual(engine.players[1].health, 20)
        self.assertFalse(
            any(event.type is EventType.EARTH_RITE_ACTIVATED for event in engine.event_history)
        )

    def test_sufficient_sigils_pay_before_effect_and_zero_destroys_stack(self):
        spell, rule = self._rite_spell(111, 1)
        engine = _engine(rule, definitions={111: spell})
        sigil = _sigil_entity(engine, count=1)
        _insert(engine, spell)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].earth_sigils, 0)
        self.assertEqual(engine.players[1].health, 17)
        grave_entry = next(
            card for card in engine.players[0].graveyard if card.entity_id == sigil.entity_id
        )
        self.assertEqual(grave_entry.entry_cause, "earth_sigil_depleted")
        activated_index = next(
            index
            for index, event in enumerate(engine.event_history)
            if event.type is EventType.EARTH_RITE_ACTIVATED
        )
        destroyed_index = next(
            index
            for index, event in enumerate(engine.event_history)
            if event.type is EventType.AMULET_DESTROYED
            and event.source_id == sigil.entity_id
        )
        damage_index = next(
            index
            for index, event in enumerate(engine.event_history)
            if event.type is EventType.DAMAGE_APPLIED
        )
        self.assertLess(activated_index, destroyed_index)
        self.assertLess(destroyed_index, damage_index)

    def test_pending_choice_does_not_repeat_earth_rite_payment(self):
        target = _card(400, name="Target", life=4)
        spell = _card(112, name="Choice Rite", card_type="法术", attack=None, life=None)
        rule = CardRule(
            112,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.EARTH_RITE,
                    TargetKind.OWN_LEADER,
                    1,
                    earth_rite_operations=(
                        EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 2),
                        EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),
                    ),
                ),
            ),
        )
        engine = _engine(rule, definitions={112: spell, 400: target})
        sigil = _sigil_entity(engine, count=2)
        unit = Unit.summon(target, entity_id=engine.state.allocate_entity_id())
        engine.players[1].board.append(unit)
        _insert(engine, spell)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(sigil.earth_sigil_count, 1)
        self.assertIsNotNone(engine.state.pending_choice)
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "entity:999999"))
        self.assertEqual(engine.deterministic_fingerprint(), before)

        choice = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        engine.apply(choice)
        self.assertEqual(sigil.earth_sigil_count, 1)
        self.assertEqual(unit.health, 2)
        self.assertEqual(engine.players[1].health, 19)
        self.assertEqual(
            sum(
                event.type is EventType.EARTH_RITE_ACTIVATED
                for event in engine.event_history
            ),
            1,
        )


class EarthRuleSchemaTests(unittest.TestCase):
    def test_earth_rite_schema_parses_nested_operations(self):
        operation = _parse_operation(
            {
                "kind": "earth_rite",
                "amount": 2,
                "operations": [
                    {"kind": "draw", "target": "own_leader", "amount": 1}
                ],
            },
            "test.json",
            1,
        )
        self.assertEqual(operation.target, TargetKind.OWN_LEADER)
        self.assertEqual(operation.amount, 2)
        self.assertEqual(operation.earth_rite_operations[0].kind, EffectKind.DRAW)

    def test_earth_rite_requires_positive_integer_and_operations(self):
        invalid = (
            {"kind": "earth_rite", "amount": 0, "operations": [{"kind": "draw", "target": "own_leader", "amount": 1}]},
            {"kind": "earth_rite", "amount": True, "operations": [{"kind": "draw", "target": "own_leader", "amount": 1}]},
            {"kind": "earth_rite", "amount": 1},
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test.json", 1)

    def test_add_earth_sigils_requires_own_leader_and_positive_integer(self):
        for raw in (
            {"kind": "add_earth_sigils", "target": "enemy_leader", "amount": 1},
            {"kind": "add_earth_sigils", "amount": -1},
            {"kind": "add_earth_sigils", "amount": "2"},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test.json", 1)

    def test_conditions_and_expressions_read_public_sigil_count(self):
        engine = _engine()
        _sigil_entity(engine, count=4)
        _sigil_entity(engine, player_index=1, count=2)
        context = EvalContext(0, engine.players)
        self.assertTrue(
            evaluate_condition(
                Condition(ConditionType.CONTROLLER_EARTH_SIGILS_AT_LEAST, 4),
                context,
            )
        )
        self.assertEqual(
            evaluate_expression(
                ValueExpression(ExprType.OPPONENT_EARTH_SIGILS),
                context,
            ),
            2,
        )


class EarthSigilEnvironmentTests(unittest.TestCase):
    def test_observation_exposes_both_public_sigil_counts(self):
        deck_a = [_card(5000 + index) for index in range(40)]
        deck_b = [_card(6000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=3,
            class_b=3,
            seed=42,
            rulebook=RuleBook(()),
            card_resolver=lambda card_id: _earth_sigil() if card_id == 90031210 else None,
            validate_invariants=True,
        )
        env.reset(seed=42)
        _sigil_entity(env.core, count=4)
        _sigil_entity(env.core, player_index=1, count=2)

        observation = env.observation()

        self.assertEqual(len(observation), 261)
        self.assertEqual(observation[-6:-4], [0.2, 0.1])


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealEarthCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def _real_engine(self) -> GameEngine:
        engine = GameEngine(
            [_card(7000 + index) for index in range(40)],
            [_card(8000 + index) for index in range(40)],
            class_a=3,
            class_b=3,
            seed=42,
            rulebook=self.rulebook,
            card_resolver=self.repo.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=42)
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10
        return engine

    def test_sweet_predator_creates_two_earth_sigils(self):
        engine = self._real_engine()
        predator = self.repo.get(10732120)
        _insert(engine, predator)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].earth_sigils, 2)
        self.assertEqual(engine._earth_sigil_amulets(0)[0].definition.card_id, 90031210)

    def test_magical_blast_damages_all_followers_then_spends_to_draw(self):
        engine = self._real_engine()
        own_unit = Unit.summon(_card(9001, life=3), entity_id=engine.state.allocate_entity_id())
        enemy_unit = Unit.summon(_card(9002, life=3), entity_id=engine.state.allocate_entity_id())
        engine.players[0].board.append(own_unit)
        engine.players[1].board.append(enemy_unit)
        sigil = _sigil_entity(engine, definition=self.repo.get(90031210), count=1)
        magical_blast = self.repo.get(10032310)
        self.assertIn(AbilityKeyword.EARTH_RITE, magical_blast.abilities)
        _insert(engine, magical_blast)
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(own_unit.health, 1)
        self.assertEqual(enemy_unit.health, 1)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)
        self.assertEqual(engine.players[0].earth_sigils, 0)
        self.assertFalse(any(card.entity_id == sigil.entity_id for card in engine.players[0].board))
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.EARTH_RITE
                for event in engine.placeholder_ability_events
            )
        )

    def test_cauldron_earth_sigil_and_activate_are_both_supported(self):
        engine = self._real_engine()
        furnace = self.repo.get(10031210)
        _insert(engine, furnace)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].earth_sigils, 1)
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.EARTH_SIGIL
                for event in engine.placeholder_ability_events
            )
        )
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.ACTIVATE
                for event in engine.placeholder_ability_events
            )
        )

    def test_coverage_marks_completed_cauldron_rule_exact(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        classifications = report["classifications"]
        self.assertEqual(classifications["10032310"]["coverage"], "covered_exact")
        self.assertEqual(classifications["10732120"]["coverage"], "covered_exact")
        self.assertEqual(classifications["10031210"]["coverage"], "covered_exact")
        self.assertNotIn("策动", classifications["10031210"]["missing_primitives"])


if __name__ == "__main__":
    unittest.main()
