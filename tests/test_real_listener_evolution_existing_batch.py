# -*- coding: utf-8 -*-
"""Exact listener/evolution cards composed from established primitives."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import ActivateAmulet, Evolve, SuperEvolve
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _choose,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10022210,
    10062120,
    10112130,
    10131130,
    10132110,
    10133120,
    10161110,
    10461110,
    10463110,
    10612310,
    10622110,
    10652120,
    10712110,
    10862120,
    10871120,
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


class RealListenerEvolutionExistingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 701):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_three_board_listeners_react_once_to_own_amulet_activation(self):
        engine = self.fresh(seed=3)
        griffin = _play(engine, self.repository, 10062120)
        engine.players[0].mana = 10
        tor = _play(engine, self.repository, 10461110)
        engine.players[0].mana = 10
        surgeon = _play(engine, self.repository, 10463110)
        engine.players[0].health = 18
        engine.players[0].mana = 10
        amulet = _play(engine, self.repository, 10762210)

        engine.apply(ActivateAmulet(0, amulet.entity_id))

        self.assertTrue(griffin.has_keyword("守护"))
        self.assertTrue(griffin.has_keyword("疾驰"))
        self.assertTrue(tor.has_keyword("疾驰"))
        self.assertTrue(tor.has_keyword("虹吸"))
        self.assertEqual(engine.players[0].health, 19)
        self.assertIn(surgeon, engine.players[0].board)

    def test_crown_buffs_each_own_follower_entry_and_has_countdown(self):
        engine = self.fresh(seed=5)
        crown = _play(engine, self.repository, 10022210)
        self.assertEqual(crown.countdown, 4)
        engine.players[0].mana = 10
        follower = _play(engine, self.repository, 10712110)
        definition = self.repository.get(10712110)
        self.assertEqual(
            (follower.attack, follower.max_health),
            (definition.attack + 1, definition.life + 1),
        )

    def test_hand_evolution_listeners_reduce_only_their_own_cards(self):
        engine = self.fresh(seed=7)
        super_listener = _put_hand(engine, self.repository.get(10862120))
        evolve_listener = _put_hand(engine, self.repository.get(10612310))
        untouched = _put_hand(engine, _card(10, cost=7))
        source = _put_unit(engine, 0, _card(11))
        _enable_super_evolve(engine)

        engine.apply(SuperEvolve(0, source.entity_id))

        self.assertEqual(
            super_listener.current_cost,
            self.repository.get(10862120).cost - 3,
        )
        self.assertEqual(
            evolve_listener.current_cost,
            self.repository.get(10612310).cost - 1,
        )
        self.assertEqual(untouched.current_cost, 7)

    def test_combo_three_grants_storm_only_at_threshold(self):
        below = self.fresh(seed=11)
        below.players[0].cards_played_this_turn = 1
        source = _play(below, self.repository, 10712110)
        self.assertFalse(source.has_keyword("疾驰"))

        active = self.fresh(seed=13)
        active.players[0].cards_played_this_turn = 2
        source = _play(active, self.repository, 10712110)
        self.assertTrue(source.has_keyword("疾驰"))

    def test_two_enhance_followers_keep_evolve_target_choices(self):
        demon = self.fresh(seed=17)
        target = _put_unit(demon, 1, _card(20, life=7))
        source = _play(demon, self.repository, 10652120, mode_id="enhance_7")
        self.assertTrue(source.has_keyword("疾驰"))
        _enable_evolve(demon)
        demon.apply(Evolve(0, source.entity_id))
        _choose(demon, target.entity_id)
        self.assertEqual(target.health, 3)

        guard = self.fresh(seed=19)
        target = _put_unit(guard, 1, _card(21, life=7))
        source = _play(guard, self.repository, 10622110, mode_id="enhance_4")
        definition = self.repository.get(10622110)
        self.assertEqual(
            (source.attack, source.max_health),
            (definition.attack + 2, definition.life + 2),
        )
        self.assertTrue(source.has_keyword("守护"))
        _enable_evolve(guard)
        guard.apply(Evolve(0, source.entity_id))
        _choose(guard, target.entity_id)
        self.assertEqual(target.health, 4)

    def test_leona_super_evolve_excludes_source_and_grants_stealth(self):
        engine = self.fresh(seed=23)
        ally = _put_unit(engine, 0, _card(30))
        source = _play(engine, self.repository, 10871120)
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose(engine, ally.entity_id)
        self.assertTrue(ally.has_keyword("潜行"))
        self.assertFalse(source.has_keyword("潜行"))
        self.assertTrue(source.has_keyword("守护"))

    def test_priest_draws_amulet_then_evolve_reduces_selected_hand_cost(self):
        engine = self.fresh(seed=29)
        amulet = _card(40, card_type="护符", attack=None, life=None, cost=4)
        engine.players[0].deck = [amulet, _card(41)]
        enemy = _put_unit(engine, 1, _card(42))
        source = _play(engine, self.repository, 10161110)
        drawn = next(card for card in engine.players[0].hand if card.card_id == 40)
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, drawn.entity_id)
        self.assertEqual(drawn.current_cost, 3)
        self.assertIn(enemy, engine.players[1].board)

    def test_spellboost_fanfare_and_evolve_each_boost_current_hand(self):
        engine = self.fresh(seed=31)
        tracked = _put_hand(
            engine,
            _card(50, card_type="法术", attack=None, life=None),
        )
        enemy = _put_unit(engine, 1, _card(51, life=6))
        source = _play(engine, self.repository, 10132110)
        self.assertEqual(tracked.spellboost_count, 1)
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, enemy.entity_id)
        self.assertEqual(enemy.health, 3)
        self.assertEqual(tracked.spellboost_count, 2)

    def test_earth_sigil_cards_and_super_evolve_followups(self):
        owl = self.fresh(seed=37)
        enemy = _put_unit(owl, 1, _card(60, life=7))
        source = _play(owl, self.repository, 10131130)
        self.assertEqual(owl.players[0].earth_sigils, 1)
        _enable_evolve(owl)
        owl.apply(Evolve(0, source.entity_id))
        _choose(owl, enemy.entity_id)
        self.assertEqual(enemy.health, 2)

        potion = self.fresh(seed=41)
        potion.players[0].health = 16
        potion.players[0].deck = [_card(70), _card(71), _card(72)]
        source = _play(potion, self.repository, 10133120)
        self.assertEqual(potion.players[0].earth_sigils, 2)
        _enable_super_evolve(potion)
        potion.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(potion.players[0].earth_sigils, 4)
        self.assertEqual(potion.players[0].health, 18)
        self.assertEqual(len(potion.players[0].hand), 2)

    def test_gem_beast_returns_other_board_card_and_restores_pp(self):
        engine = self.fresh(seed=43)
        ally = _put_unit(engine, 0, _card(80))
        engine.players[0].mana = 10
        source = _play(engine, self.repository, 10112130)
        _choose(engine, ally.entity_id)
        self.assertNotIn(ally, engine.players[0].board)
        self.assertTrue(any(card.card_id == 80 for card in engine.players[0].hand))
        engine.players[0].mana = 2
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(engine.players[0].mana, 5)


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
                    ["tests/test_real_listener_evolution_existing_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
