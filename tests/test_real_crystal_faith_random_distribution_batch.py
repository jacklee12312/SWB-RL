from __future__ import annotations

import os
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import (
    RuleBook,
    _parse_faith_definition,
    _parse_operation,
)
from swb.engine.commands import Evolve, PlayCard
from swb.engine.effects import EffectKind, ExprType
from swb.engine.events import EventType
from swb.engine.faith import FaithTrigger
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


def _card(card_id: int, *, class_id: int = 3, class_name: str = "巫师") -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=class_id,
        class_name=class_name,
        name=f"fixture-{card_id}",
        cost=1,
        card_type="随从",
        attack=1,
        life=2,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def _insert_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    origin: CardOrigin = CardOrigin.DECK,
) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=origin,
    )
    engine.players[0].hand.append(card)
    engine.players[0].hand_entity_ids.append(card.entity_id)
    return card


def _put_unit(engine: GameEngine, player_index: int, card_id: int) -> Unit:
    unit = Unit.summon(
        _card(card_id),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[player_index].board.append(unit)
    return unit


class CrystalFaithSchemaTests(unittest.TestCase):
    def test_follower_entry_faith_filter_parses_and_other_triggers_reject_it(self):
        definition = _parse_faith_definition(
            {
                "id": "crystals",
                "source_card_id": 123,
                "triggers": [
                    {
                        "trigger": "follower_summoned",
                        "event_filter": {"card_id": 456},
                    }
                ],
            },
            "test",
        )

        trigger = definition.triggers[0]
        self.assertIs(trigger.trigger, FaithTrigger.FOLLOWER_SUMMONED)
        self.assertEqual(trigger.event_filter.card_id, 456)
        with self.assertRaisesRegex(ValueError, "only valid"):
            _parse_faith_definition(
                {
                    "id": "bad",
                    "source_card_id": 123,
                    "triggers": [
                        {
                            "trigger": "follower_evolved",
                            "event_filter": {"card_id": 456},
                        }
                    ],
                },
                "test",
            )

    def test_random_distribution_schema_is_named_bounded_and_contextual(self):
        operation = _parse_operation(
            {
                "kind": "random_distribute",
                "faith_id": "crystals",
                "buckets": [
                    [
                        {
                            "kind": "heal_leader",
                            "target": "own_leader",
                            "amount": {"type": "distributed_value"},
                        }
                    ],
                    [
                        {
                            "kind": "damage_leader",
                            "target": "enemy_leader",
                            "amount": {"type": "distributed_value"},
                        }
                    ],
                ],
            },
            "test",
            123,
        )

        self.assertIs(operation.kind, EffectKind.RANDOM_DISTRIBUTE)
        self.assertEqual(operation.faith_id, "crystals")
        self.assertEqual(len(operation.random_distribution_operations), 2)
        self.assertIs(
            operation.random_distribution_operations[0][0].amount_expr.type,
            ExprType.DISTRIBUTED_VALUE,
        )
        invalid = (
            {
                "kind": "random_distribute",
                "buckets": [[{"kind": "draw", "target": "own_leader", "amount": 1}]],
            },
            {
                "kind": "random_distribute",
                "faith_id": "crystals",
                "buckets": [[], [{"kind": "draw", "target": "own_leader", "amount": 1}]],
            },
            {
                "kind": "heal_leader",
                "target": "own_leader",
                "amount": {"type": "distributed_value"},
            },
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test", 123)


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealCrystalFaithRandomDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def fresh(self, *, seed: int = 4201) -> GameEngine:
        source = self.repository.get(10634120)
        deck = [
            source,
            source,
            source,
            *[_card(960000 + index) for index in range(37)],
        ]
        other_deck = [
            _card(970000 + index, class_id=1, class_name="精灵")
            for index in range(40)
        ]
        engine = GameEngine(
            deck,
            other_deck,
            class_a=3,
            class_b=1,
            seed=seed,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=seed)
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        engine.players[0].board.clear()
        return engine

    @staticmethod
    def _unlock_evolution(engine: GameEngine) -> None:
        engine.players[0].turns_started = max(
            engine.players[0].turns_started,
            engine.config.first_player_super_evolution_unlock_turn,
        )

    @staticmethod
    def _distribution_event(engine: GameEngine):
        return next(
            event
            for event in engine.event_history
            if event.type is EventType.RANDOM_DISTRIBUTION_RESOLVED
            and event.metadata["faith_id"] == "eld_crystals"
        )

    def test_fanfare_summons_two_storm_crystalspawns_and_updates_hand_and_faith(self):
        engine = self.fresh()
        source = self.repository.get(10634120)
        _insert_hand(engine, source)
        remaining = _insert_hand(engine, source)
        engine.players[0].max_mana = engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.players[0].faiths[0].value, 2)
        self.assertEqual(remaining.current_cost, 8)
        source_unit = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10634120
        )
        crystalspawns = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10631110
        ]
        self.assertEqual(len(crystalspawns), 2)
        self.assertFalse(source_unit.has_keyword("疾驰"))
        self.assertTrue(all(unit.has_keyword("突进") for unit in crystalspawns))
        self.assertTrue(all(unit.has_keyword("疾驰") for unit in crystalspawns))

    def test_board_capacity_counts_only_successful_crystalspawn_entry(self):
        engine = self.fresh()
        for index in range(3):
            _put_unit(engine, 0, 961000 + index)
        source = self.repository.get(10634120)
        _insert_hand(engine, source)
        remaining = _insert_hand(engine, source)
        engine.players[0].max_mana = engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(len(engine.players[0].board), 5)
        self.assertEqual(engine.players[0].faiths[0].value, 1)
        self.assertEqual(remaining.current_cost, 9)
        crystalspawn = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10631110
        )
        self.assertTrue(crystalspawn.has_keyword("疾驰"))

    def test_evolve_adds_depths_as_a_token_card(self):
        engine = self.fresh()
        source = self.repository.get(10634120)
        _insert_hand(engine, source)
        engine.players[0].max_mana = engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        source_unit = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10634120
        )
        self._unlock_evolution(engine)

        engine.apply(Evolve(0, source_unit.entity_id))

        depths = next(
            card for card in engine.players[0].hand
            if card.card_id == 90034330
        )
        self.assertIs(depths.origin, CardOrigin.TOKEN)

    def test_depths_uses_live_faith_and_applies_x_y_z_without_consuming_it(self):
        engine = self.fresh(seed=4202)
        token = self.repository.get(90034330)
        _insert_hand(engine, token, origin=CardOrigin.TOKEN)
        faith = engine.players[0].faiths[0]
        faith.value = 6
        engine.players[0].health = 10
        engine.players[0].max_mana = engine.players[0].mana = token.cost

        engine.apply(PlayCard(0, 0))

        event = self._distribution_event(engine)
        x, y, z = event.metadata["bucket_values"]
        self.assertEqual(x + y + z, 7)
        self.assertEqual(event.amount, 7)
        self.assertEqual(faith.value, 7)
        crystalspawn = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10631110
        )
        self.assertEqual((crystalspawn.attack, crystalspawn.health), (1 + x, 1 + x))
        self.assertEqual(engine.players[0].health, min(20, 10 + y))
        self.assertEqual(engine.players[1].health, 20 - z)

    def test_full_board_skips_summon_but_still_resolves_y_and_z(self):
        engine = self.fresh(seed=4203)
        for index in range(5):
            _put_unit(engine, 0, 962000 + index)
        token = self.repository.get(90034330)
        _insert_hand(engine, token, origin=CardOrigin.TOKEN)
        faith = engine.players[0].faiths[0]
        faith.value = 5
        engine.players[0].health = 10
        engine.players[0].max_mana = engine.players[0].mana = token.cost

        engine.apply(PlayCard(0, 0))

        x, y, z = self._distribution_event(engine).metadata["bucket_values"]
        self.assertEqual(x + y + z, 5)
        self.assertEqual(faith.value, 5)
        self.assertFalse(
            any(unit.definition.card_id == 10631110 for unit in engine.players[0].board)
        )
        self.assertEqual(engine.players[0].health, min(20, 10 + y))
        self.assertEqual(engine.players[1].health, 20 - z)

    def test_zero_total_consumes_no_rng_and_seeded_runs_reproduce(self):
        zero = self.fresh(seed=4204)
        for index in range(5):
            _put_unit(zero, 0, 963000 + index)
        token = self.repository.get(90034330)
        _insert_hand(zero, token, origin=CardOrigin.TOKEN)
        zero.players[0].max_mana = zero.players[0].mana = token.cost
        rng_before = zero.random.getstate()

        zero.apply(PlayCard(0, 0))

        self.assertEqual(zero.random.getstate(), rng_before)
        self.assertEqual(
            self._distribution_event(zero).metadata["bucket_values"],
            (0, 0, 0),
        )

        def run_once():
            engine = self.fresh(seed=4205)
            _insert_hand(engine, token, origin=CardOrigin.TOKEN)
            engine.players[0].faiths[0].value = 12
            engine.players[0].max_mana = engine.players[0].mana = token.cost
            engine.apply(PlayCard(0, 0))
            return (
                self._distribution_event(engine).metadata["bucket_values"],
                engine.deterministic_fingerprint(),
            )

        self.assertEqual(run_once(), run_once())

    def test_source_and_generated_token_are_exact_and_producer_complete(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(
            report["classifications"]["10634120"]["coverage"],
            "covered_exact",
        )
        token_audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        token_info = next(
            card for card in token_audit["cards"]
            if card["card_id"] == 90034330
        )
        self.assertEqual(token_info["category"], "entry_behavior_complete")
        self.assertEqual(token_info["explicit_coverage"], "exact")
        self.assertEqual(
            [producer["source_card_id"] for producer in token_info["authored_producers"]],
            [10634120],
        )


if __name__ == "__main__":
    unittest.main()
