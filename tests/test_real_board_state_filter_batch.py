# -*- coding: utf-8 -*-
"""Exact real-card tests for damaged and super-evolved board filters."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


CARD_IDS = (
    10201110,
    10204120,
    10222110,
    10231110,
    10242110,
    10261110,
    10341120,
    10441120,
    10462110,
    10863210,
)
SOURCE_HASHES = {
    10201110: "ea28e54fca103edc77b19c9a664650cdb0379527f525128ecf6468927834eab6",
    10204120: "a21365774862301a9e8cad5172a1c7361bb264b5203fad9c1ed561e913fa9d92",
    10222110: "9966eb0d82d52f021b9d73be4687022ec3cbdcefc25cfa0f42140019a3c6ab4d",
    10231110: "deafa81073e4f94bd7e7e4aab943e6f88940d519a20014d56506db8fc9ac1148",
    10242110: "86b34056031710deb70329ff6b23a9df36df777b5d7e5ebb26b03c00a2c1582d",
    10261110: "117855c7cfe1f9f510828b27e8b2f55045bb3046c9dd45098dbb8c22cddcf49b",
    10341120: "fd332206fac23ac8d9d7106e9b7c2538954e947c361944dee247e8de5f2888a5",
    10441120: "2dfca726f3c0900c1bf26f154715aefa6ce869850a988b0d40e58319d03ef84e",
    10462110: "15d16503f1e7fdc76ca675f391a419e7bbe589aadc14cd6ed9c3331c41811ce5",
    10863210: "780ac2a13ddc05197c787e4c80bcef825968a0ed6b4035a1ee178c6ca3673874",
}
STRUCTURED_EVIDENCE = {
    10201110: {"triggers": ["fanfare"], "effect_kinds": ["conditional", "damage_unit"]},
    10204120: {"triggers": ["fanfare", "intrinsic_keywords", "emblem_source"], "effect_kinds": ["gain_emblem", "keyword:守护", "damage_unit"]},
    10222110: {"triggers": ["turn_end", "intrinsic_keywords"], "effect_kinds": ["conditional", "buff_unit", "keyword:守护"]},
    10231110: {"triggers": ["fanfare"], "effect_kinds": ["add_card", "conditional", "add_earth_sigils"]},
    10242110: {"triggers": ["fanfare"], "effect_kinds": ["conditional", "draw"]},
    10261110: {"triggers": ["evolve", "intrinsic_keywords"], "effect_kinds": ["destroy", "keyword:守护"]},
    10341120: {"triggers": ["fanfare"], "effect_kinds": ["destroy"]},
    10441120: {"triggers": ["turn_end", "listener:hand:follower_super_evolved"], "effect_kinds": ["buff_unit", "change_cost"]},
    10462110: {"triggers": ["fanfare", "evolve", "play_modes", "intrinsic_keywords"], "effect_kinds": ["destroy", "play_mode", "buff_unit", "keyword:守护"]},
    10863210: {"triggers": ["play", "turn_end"], "effect_kinds": ["draw", "conditional", "heal_leader", "conditional", "heal_leader"]},
}


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 1),
        class_name=overrides.get("class_name", "精灵"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 5),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _engine(rulebook: RuleBook, repository: CardRepository, seed: int = 2117) -> GameEngine:
    engine = GameEngine(
        [_card(i) for i in range(1000, 1040)],
        [_card(i) for i in range(2000, 2040)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
        card_resolver=repository.get,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    for player in engine.players:
        player.hand.clear()
        player.hand_entity_ids.clear()
        player.max_mana = player.mana = 10
    return engine


def _put(engine: GameEngine, definition: CardDefinition, owner: int = 0) -> HandCard:
    card = HandCard(definition=definition, entity_id=engine.state.allocate_entity_id())
    player = engine.players[owner]
    player.hand.insert(0, card)
    player.hand_entity_ids.insert(0, card.entity_id)
    return card


def _play(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
    *,
    mode: str = "normal",
):
    definition = repository.get(card_id)
    _put(engine, definition)
    engine.apply(PlayCard(0, 0, mode))
    if definition.card_type == "法术":
        return None
    return next(
        entity
        for entity in reversed(engine.players[0].board)
        if entity.definition.card_id == card_id
    )


def _unit(
    engine: GameEngine,
    owner: int,
    card_id: int,
    *,
    cost: int = 1,
    attack: int = 1,
    life: int = 5,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, cost=cost, attack=attack, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[owner].board.append(unit)
    return unit


def _mark_super_evolved(engine: GameEngine, unit: Unit) -> None:
    unit.evolved = True
    unit.super_evolved = True
    unit.super_evolved_turn = engine.turn


def _enable_evolution(engine: GameEngine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _enable_super_evolution(engine: GameEngine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.first_player_super_evolution_unlock_turn
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.super_evolved_this_turn = False


def _choose(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


class DatabaseAndCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_cards_and_single_reference_are_present(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            placeholders = ",".join("?" for _ in CARD_IDS)
            self.assertEqual(
                connection.execute(
                    f"SELECT COUNT(*) FROM cards WHERE card_id IN ({placeholders})",
                    CARD_IDS,
                ).fetchone()[0],
                len(CARD_IDS),
            )
            self.assertEqual(
                connection.execute(
                    f"SELECT card_id, referenced_card_id FROM card_references WHERE card_id IN ({placeholders})",
                    CARD_IDS,
                ).fetchall(),
                [(10231110, 10031210)],
            )
            self.assertEqual(
                connection.execute(
                    f"SELECT card_id, mode_type FROM alt_modes WHERE card_id IN ({placeholders})",
                    CARD_IDS,
                ).fetchall(),
                [(10204120, "纹章")],
            )

    def test_all_ten_cards_are_exact_with_hash_and_direct_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                audit = report["classifications"][str(card_id)]["clause_audit"]
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(audit["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(audit["structured_evidence"], STRUCTURED_EVIDENCE[card_id])
                self.assertEqual(audit["test_evidence"], ["tests/test_real_board_state_filter_batch.py"])


class RealBoardStateFilterBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, seed: int = 2117) -> GameEngine:
        return _engine(self.rulebook, self.repository, seed)

    def test_double_blade_goblin_requires_a_super_evolved_ally_before_targeting(self):
        inactive = self.fresh()
        target = _unit(inactive, 1, 3000)
        _play(inactive, self.repository, 10201110)
        self.assertIsNone(inactive.state.pending_choice)
        self.assertEqual(target.health, 5)

        active = self.fresh()
        ally = _unit(active, 0, 3001)
        _mark_super_evolved(active, ally)
        target = _unit(active, 1, 3002)
        _play(active, self.repository, 10201110)
        _choose(active, target.entity_id)
        self.assertEqual(target.health, 1)

    def test_grimnir_emblem_checks_super_evolved_board_at_owner_turn_end(self):
        inactive = self.fresh()
        target = _unit(inactive, 1, 3100)
        source = _play(inactive, self.repository, 10204120)
        inactive.apply(EndTurn(0))
        self.assertTrue(source.has_keyword("守护"))
        self.assertEqual(target.health, 5)
        self.assertEqual(len(inactive.players[0].emblems), 1)

        active = self.fresh()
        ally = _unit(active, 0, 3101)
        _mark_super_evolved(active, ally)
        first = _unit(active, 1, 3102, life=2)
        second = _unit(active, 1, 3103, life=3)
        _play(active, self.repository, 10204120)
        active.apply(EndTurn(0))
        self.assertNotIn(first, active.players[1].board)
        self.assertEqual(second.health, 1)

    def test_gelt_buffs_all_allies_only_for_a_super_evolved_board(self):
        inactive = self.fresh()
        evolved = _unit(inactive, 0, 3200)
        evolved.evolved = True
        source = _play(inactive, self.repository, 10222110)
        before = (source.attack, source.health, evolved.attack, evolved.health)
        inactive.apply(EndTurn(0))
        self.assertEqual((source.attack, source.health, evolved.attack, evolved.health), before)

        active = self.fresh()
        ally = _unit(active, 0, 3201)
        _mark_super_evolved(active, ally)
        source = _play(active, self.repository, 10222110)
        before = (source.attack, source.health, ally.attack, ally.health)
        active.apply(EndTurn(0))
        self.assertEqual(
            (source.attack, source.health, ally.attack, ally.health),
            tuple(value + 1 for value in before),
        )
        self.assertTrue(source.has_keyword("守护"))

    def test_mavey_always_adds_furnace_and_conditionally_adds_two_sigils(self):
        inactive = self.fresh()
        evolved = _unit(inactive, 0, 3300)
        evolved.evolved = True
        _play(inactive, self.repository, 10231110)
        self.assertIn(10031210, [card.card_id for card in inactive.players[0].hand])
        self.assertEqual(inactive.players[0].earth_sigils, 0)

        active = self.fresh()
        ally = _unit(active, 0, 3301)
        _mark_super_evolved(active, ally)
        _play(active, self.repository, 10231110)
        self.assertIn(10031210, [card.card_id for card in active.players[0].hand])
        self.assertEqual(active.players[0].earth_sigils, 2)

    def test_dragon_princess_draws_exactly_two_only_with_super_evolved_ally(self):
        inactive = self.fresh()
        inactive.players[0].deck = [_card(3400), _card(3401), _card(3402)]
        _play(inactive, self.repository, 10242110)
        self.assertEqual(len(inactive.players[0].hand), 0)

        active = self.fresh()
        active.players[0].deck = [_card(3410), _card(3411), _card(3412)]
        ally = _unit(active, 0, 3413)
        _mark_super_evolved(active, ally)
        _play(active, self.repository, 10242110)
        self.assertEqual(len(active.players[0].hand), 2)

    def test_crushing_cleric_evolve_targets_only_super_evolved_enemy(self):
        engine = self.fresh()
        source = _play(engine, self.repository, 10261110)
        normal = _unit(engine, 1, 3500)
        normal.evolved = True
        target = _unit(engine, 1, 3501)
        _mark_super_evolved(engine, target)
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [target.entity_id],
        )
        fingerprint = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, f"entity:{normal.entity_id}"))
        self.assertEqual(engine.deterministic_fingerprint(), fingerprint)
        _choose(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)
        self.assertIn(normal, engine.players[1].board)
        self.assertTrue(source.has_keyword("守护"))

    def test_snow_dragon_destroys_all_and_only_damaged_enemy_followers(self):
        engine = self.fresh()
        first = _unit(engine, 1, 3600)
        second = _unit(engine, 1, 3601)
        full = _unit(engine, 1, 3602)
        first.health = 1
        second.health = 4
        _play(engine, self.repository, 10341120)
        self.assertNotIn(first, engine.players[1].board)
        self.assertNotIn(second, engine.players[1].board)
        self.assertIn(full, engine.players[1].board)

    def test_marychin_hand_listener_and_seeded_random_turn_end_buff(self):
        wrong = self.fresh()
        tracked = _put(wrong, self.repository.get(10441120))
        cost_four = _unit(wrong, 0, 3700, cost=4)
        _enable_super_evolution(wrong)
        wrong.apply(SuperEvolve(0, cost_four.entity_id))
        self.assertEqual(tracked.current_cost, 2)

        active = self.fresh()
        tracked = _put(active, self.repository.get(10441120))
        cost_three = _unit(active, 0, 3701, cost=3)
        _enable_super_evolution(active)
        active.apply(SuperEvolve(0, cost_three.entity_id))
        self.assertEqual(tracked.current_cost, 0)
        active.apply(EndTurn(0))
        self.assertEqual(tracked.current_cost, 2)

        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=73)
            source = _play(engine, self.repository, 10441120)
            first = _unit(engine, 0, 3710, attack=2)
            second = _unit(engine, 0, 3711, attack=2)
            normal = _unit(engine, 0, 3712, attack=2)
            _mark_super_evolved(engine, first)
            _mark_super_evolved(engine, second)
            normal.evolved = True
            engine.apply(EndTurn(0))
            self.assertEqual(sum(unit.attack == 3 for unit in (first, second)), 1)
            self.assertEqual(normal.attack, 2)
            self.assertEqual(source.attack, 2)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_sarah_enhance_and_evolve_use_damaged_filter(self):
        normal = self.fresh()
        normal_source = _play(normal, self.repository, 10462110)
        self.assertEqual((normal_source.attack, normal_source.health), (4, 5))

        enhanced = self.fresh()
        enhanced_source = _play(enhanced, self.repository, 10462110, mode="enhance_6")
        self.assertEqual((enhanced_source.attack, enhanced_source.health), (4, 15))
        self.assertTrue(enhanced_source.has_keyword("守护"))

        engine = self.fresh()
        source = _play(engine, self.repository, 10462110)
        damaged = _unit(engine, 1, 3800)
        damaged.health = 2
        full = _unit(engine, 1, 3801)
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [damaged.entity_id],
        )
        _choose(engine, damaged.entity_id)
        self.assertNotIn(damaged, engine.players[1].board)
        self.assertIn(full, engine.players[1].board)

    def test_classmates_draws_then_heals_zero_one_or_replacement_two(self):
        for state, expected_health in (("none", 10), ("evolved", 11), ("super", 12)):
            with self.subTest(state=state):
                engine = self.fresh()
                engine.players[0].health = 10
                engine.players[0].deck = [_card(3900), _card(3901), _card(3902)]
                if state != "none":
                    ally = _unit(engine, 0, 3903)
                    ally.evolved = True
                    if state == "super":
                        _mark_super_evolved(engine, ally)
                amulet = _play(engine, self.repository, 10863210)
                engine.apply(EndTurn(0))
                self.assertEqual(engine.players[0].health, expected_health)
                self.assertEqual(len(engine.players[0].hand), 1)
                self.assertEqual(amulet.countdown, 2)

    def test_rl_evolve_choice_mask_exposes_only_super_evolved_target(self):
        env = ShadowverseEnv(
            [_card(i) for i in range(4000, 4040)],
            [_card(i) for i in range(4100, 4140)],
            class_a=1,
            class_b=1,
            seed=81,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=81)
        env.players[0].board.clear()
        env.players[1].board.clear()
        source = _unit(env.core, 0, 10261110)
        normal = _unit(env.core, 1, 4200)
        normal.evolved = True
        target = _unit(env.core, 1, 4201)
        _mark_super_evolved(env.core, target)
        _enable_evolution(env.core)

        env.step(env.EVOLVE_OFFSET)
        enabled = [
            action
            for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        ]
        self.assertTrue(source.evolved)
        self.assertEqual(enabled, [env.CHOICE_OFFSET])
        self.assertEqual(env.core.state.pending_choice.options[0].entity_id, target.entity_id)


if __name__ == "__main__":
    unittest.main()
