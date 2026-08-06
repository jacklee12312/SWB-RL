# -*- coding: utf-8 -*-
"""Exact real cards with repeated evolve abilities and draw listeners."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import ActivateAmulet, EndTurn, Evolve, SuperEvolve
from swb.engine.state import AttackRestriction
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _choose,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10104110,
    10113110,
    10232120,
    10253110,
    10372110,
    10444110,
    10461120,
    10552120,
    10561120,
    10562120,
    10643110,
    10661210,
    10731120,
    10841110,
    10861110,
)


def _enable_evolve(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _enable_super_evolve(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.first_player_super_evolution_unlock_turn
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.super_evolved_this_turn = False


class RealRepeatedEvolveListenerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 751):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_destroyer_grants_other_followers_lethal_then_storm(self):
        engine = self.fresh(seed=3)
        ally = _put_unit(engine, 0, _card(10))
        source = _play(engine, self.repository, 10253110)
        _choose(engine, ally.entity_id)
        self.assertTrue(ally.has_keyword("必杀"))
        self.assertFalse(source.has_keyword("必杀"))

        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose(engine, ally.entity_id)
        self.assertTrue(ally.has_keyword("疾驰"))
        self.assertFalse(source.has_keyword("疾驰"))

    def test_olivia_fanfare_and_super_evolve_another_follower(self):
        engine = self.fresh(seed=5)
        engine.players[0].health = 17
        engine.players[0].deck = [_card(20), _card(21), _card(22)]
        ally = _put_unit(engine, 0, _card(23))
        source = _play(engine, self.repository, 10104110)
        self.assertEqual(len(engine.players[0].hand), 2)
        self.assertEqual(engine.players[0].health, 19)
        self.assertEqual(engine.players[0].mana, 5)

        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose(engine, ally.entity_id)
        self.assertTrue(ally.evolved)
        self.assertTrue(ally.super_evolved)

    def test_draw_listener_fires_on_fanfare_evolve_and_only_owner_turn(self):
        engine = self.fresh(seed=7)
        enemy = _put_unit(engine, 1, _card(30, life=8))
        engine.players[0].deck = [_card(31), _card(32), _card(33)]
        source = _play(engine, self.repository, 10562120)
        self.assertEqual(enemy.health, 7)

        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(enemy.health, 6)

        engine.apply(EndTurn(0))
        before = enemy.health
        engine._draw(0, reason="opponent-turn test")
        self.assertEqual(enemy.health, before)

    def test_humanoid_evolve_silences_and_damages_same_target_then_draws_on_death(self):
        engine = self.fresh(seed=11)
        engine.players[0].deck = [_card(40)]
        enemy = _put_unit(
            engine,
            1,
            _card(41, life=7),
        )
        enemy.add_keyword("守护")
        source = _play(engine, self.repository, 10861110)
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, enemy.entity_id)
        self.assertTrue(enemy.printed_abilities_removed)
        self.assertFalse(enemy.has_keyword("守护"))
        self.assertEqual(enemy.health, 5)

        source.health = 0
        engine._stabilize()
        self.assertEqual(len(engine.players[0].hand), 1)

    def test_damage_and_earth_sigil_fanfares_repeat_on_evolve(self):
        for card_id, damage in ((10232120, 3), (10731120, 1)):
            with self.subTest(card_id=card_id):
                engine = self.fresh(seed=card_id)
                first = _put_unit(engine, 1, _card(50, life=10))
                second = _put_unit(engine, 1, _card(51, life=10))
                source = _play(engine, self.repository, card_id)
                _choose(engine, first.entity_id)
                self.assertEqual(first.health, 10 - damage)
                self.assertEqual(engine.players[0].earth_sigils, 1)
                _enable_evolve(engine)
                engine.apply(Evolve(0, source.entity_id))
                _choose(engine, second.entity_id)
                self.assertEqual(second.health, 10 - damage)
                self.assertEqual(engine.players[0].earth_sigils, 2)

    def test_aura_follower_deals_eight_on_fanfare_and_evolve(self):
        engine = self.fresh(seed=13)
        first = _put_unit(engine, 1, _card(60, life=12))
        second = _put_unit(engine, 1, _card(61, life=12))
        source = _play(engine, self.repository, 10444110)
        _choose(engine, first.entity_id)
        self.assertEqual(first.health, 4)
        self.assertTrue(source.has_keyword("威慑"))
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, second.entity_id)
        self.assertEqual(second.health, 4)

    def test_reaper_optional_sacrifice_does_not_block_random_damage(self):
        empty = self.fresh(seed=17)
        enemy = _put_unit(empty, 1, _card(70, life=5))
        _play(empty, self.repository, 10372110)
        self.assertEqual(enemy.health, 3)

        active = self.fresh(seed=19)
        ally = _put_unit(active, 0, _card(71))
        enemy = _put_unit(active, 1, _card(72, life=5))
        _play(active, self.repository, 10372110)
        _choose(active, ally.entity_id)
        self.assertNotIn(ally, active.players[0].board)
        self.assertEqual(enemy.health, 3)

    def test_attack_lock_draws_and_expires_after_enemy_turn(self):
        engine = self.fresh(seed=23)
        engine.players[0].deck = [_card(80), _card(81)]
        enemy = _put_unit(engine, 1, _card(82))
        source = _play(engine, self.repository, 10552120)
        _choose(engine, enemy.entity_id)
        self.assertEqual(len(engine.players[0].hand), 1)
        self.assertTrue(any(
            modifier.restriction is AttackRestriction.CANNOT_ATTACK
            for modifier in enemy.attack_restrictions
        ))
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        self.assertEqual(enemy.attack_restrictions, [])
        self.assertIn(source, engine.players[0].board)

    def test_combo_health_set_and_evolve_damage_paths(self):
        below = self.fresh(seed=29)
        enemy = _put_unit(below, 1, _card(90, life=8))
        below.players[0].cards_played_this_turn = 1
        _play(below, self.repository, 10113110)
        self.assertEqual(enemy.health, 8)

        active = self.fresh(seed=31)
        active.players[0].deck = [_card(91)]
        enemy = _put_unit(active, 1, _card(92, life=8))
        active.players[0].cards_played_this_turn = 2
        source = _play(active, self.repository, 10113110)
        _choose(active, enemy.entity_id)
        self.assertEqual(enemy.health, 1)
        _enable_evolve(active)
        active.apply(Evolve(0, source.entity_id))
        _choose(active, enemy.entity_id)
        self.assertNotIn(enemy, active.players[1].board)
        self.assertEqual(len(active.players[0].hand), 1)

    def test_evolved_turn_end_source_damages_all_units(self):
        engine = self.fresh(seed=37)
        ally = _put_unit(engine, 0, _card(100, life=6))
        enemy = _put_unit(engine, 1, _card(101, life=6))
        source = _play(engine, self.repository, 10461120)
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertTrue(any(
            modifier.restriction is AttackRestriction.CANNOT_ATTACK
            for modifier in source.attack_restrictions
        ))
        engine.apply(EndTurn(0))
        self.assertEqual(ally.health, 4)
        self.assertEqual(enemy.health, 4)

    def test_enhance_draw_listener_grants_rush_and_lethal(self):
        normal = self.fresh(seed=41)
        normal_source = _play(normal, self.repository, 10561120)
        self.assertFalse(normal_source.has_keyword("突进"))
        self.assertFalse(normal_source.has_keyword("必杀"))

        enhanced = self.fresh(seed=43)
        enhanced.players[0].deck = [_card(110)]
        source = _play(
            enhanced,
            self.repository,
            10561120,
            mode_id="enhance_4",
        )
        self.assertTrue(source.has_keyword("突进"))
        self.assertTrue(source.has_keyword("必杀"))
        self.assertEqual(len(enhanced.players[0].hand), 1)

    def test_countdown_activation_reaches_last_words(self):
        engine = self.fresh(seed=47)
        engine.players[0].deck = [_card(120)]
        enemy = _put_unit(engine, 1, _card(121))
        amulet = _play(engine, self.repository, 10661210)
        self.assertEqual(amulet.countdown, 3)
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertEqual(amulet.countdown, 2)
        amulet.countdown = 1
        amulet.activated_turn = None
        engine.players[0].mana = 10
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertNotIn(amulet, engine.players[0].board)
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertEqual(len(engine.players[0].hand), 1)

    def test_pugilist_discards_when_possible_and_damage_still_runs_without_hand(self):
        empty = self.fresh(seed=53)
        enemy = _put_unit(empty, 1, _card(130, life=7))
        source = _play(empty, self.repository, 10643110)
        self.assertEqual(enemy.health, 1)
        self.assertTrue(source.has_keyword("守护"))
        self.assertTrue(source.has_keyword("屏障"))

        active = self.fresh(seed=59)
        discarded = _put_hand(active, _card(131))
        enemy = _put_unit(active, 1, _card(132, life=13))
        source = _play(active, self.repository, 10643110)
        _choose(active, discarded.entity_id)
        self.assertEqual(enemy.health, 7)
        _enable_evolve(active)
        another = _put_hand(active, _card(133))
        active.apply(Evolve(0, source.entity_id))
        _choose(active, another.entity_id)
        self.assertEqual(enemy.health, 1)

    def test_low_health_fanfare_auto_evolves_only_at_threshold(self):
        healthy = self.fresh(seed=61)
        healthy.players[0].health = 11
        source = _play(healthy, self.repository, 10841110)
        self.assertFalse(source.evolved)

        active = self.fresh(seed=67)
        active.players[0].health = 10
        source = _play(active, self.repository, 10841110)
        self.assertTrue(source.evolved)
        self.assertTrue(source.has_keyword("必杀"))
        self.assertTrue(source.has_keyword("守护"))


class DatabaseAndAuditTests(unittest.TestCase):
    def test_cards_are_mapped_exact_with_direct_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                classification = report["classifications"][str(card_id)]
                audit = classification["clause_audit"]
                self.assertEqual(classification["coverage"], "covered_exact")
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(
                    audit["test_evidence"],
                    ["tests/test_real_repeated_evolve_listener_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
