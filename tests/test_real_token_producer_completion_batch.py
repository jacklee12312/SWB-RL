# -*- coding: utf-8 -*-
"""Exact generated-card behavior paired with executable real producers."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import (
    ActivateAmulet,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
    SuperEvolve,
)
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _choose,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (
    10133320,
    10141140,
    10154110,
    10161310,
    10163210,
    10331120,
    10371310,
    10373310,
    10831120,
)

TOKEN_IDS = (
    90031130,
    90031140,
    90031210,
    90031310,
    90041110,
    90054110,
    90054120,
    90061110,
    90061120,
    90071110,
    90074210,
    90074220,
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


class RealTokenProducerCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 851):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_feather_rain_damages_before_summoning_storm_falcon(self):
        engine = self.fresh(seed=3)
        enemies = [_put_unit(engine, 1, _card(10 + i, life=3)) for i in range(2)]
        _play(engine, self.repository, 10161310)
        self.assertTrue(all(enemy not in engine.players[1].board for enemy in enemies))
        falcon = engine.players[0].board[0]
        self.assertEqual(falcon.definition.card_id, 90061110)
        self.assertTrue(falcon.has_keyword("疾驰"))

    def test_string_assault_adds_two_rush_puppets_that_die_at_opponent_turn_end(self):
        engine = self.fresh(seed=5)
        enemy = _put_unit(engine, 1, _card(20))
        _play(engine, self.repository, 10371310)
        _choose(engine, enemy.entity_id)
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertEqual([card.card_id for card in engine.players[0].hand], [90071110, 90071110])

        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        puppet = engine.players[0].board[0]
        self.assertTrue(puppet.has_keyword("突进"))
        engine.apply(EndTurn(0))
        self.assertIn(puppet, engine.players[0].board)
        engine.apply(EndTurn(1))
        self.assertNotIn(puppet, engine.players[0].board)

    def test_annihilation_song_starts_deterministic_white_black_cycle(self):
        engine = self.fresh(seed=7)
        sacrifice = _put_unit(engine, 0, _card(30))
        engine.players[0].health = 18
        _play(engine, self.repository, 10373310)
        _choose(engine, sacrifice.entity_id)
        white = engine.players[0].board[0]
        self.assertEqual((white.definition.card_id, white.countdown), (90074210, 1))

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        black = engine.players[0].board[0]
        self.assertEqual(engine.players[0].health, 19)
        self.assertEqual((black.definition.card_id, black.countdown), (90074220, 1))

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        self.assertEqual(engine.players[1].health, 19)
        self.assertEqual(engine.players[0].board[0].definition.card_id, 90074210)

    def test_cerberus_summons_both_arms_necromancy_buffs_others_and_last_words(self):
        engine = self.fresh(seed=11)
        engine.players[0].shadows = 6
        engine.players[0].health = 17
        ally = _put_unit(engine, 0, _card(40))
        source = _play(engine, self.repository, 10154110)
        mimi, coco = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id in {90054110, 90054120}
        ]
        self.assertEqual(engine.players[0].shadows, 0)
        self.assertEqual(ally.attack, 3)
        self.assertEqual(source.attack, self.repository.get(10154110).attack)
        self.assertTrue(mimi.has_keyword("突进"))
        self.assertTrue(coco.has_keyword("突进"))

        mimi.health = 0
        coco.health = 0
        engine._stabilize()
        self.assertEqual(engine.players[1].health, 18)
        self.assertEqual(engine.players[0].health, 19)

    def test_cerberus_super_evolve_resolves_reanimate_one_twice(self):
        engine = self.fresh(seed=13)
        for card_id in (50, 51):
            corpse = _put_unit(engine, 0, _card(card_id, cost=1))
            corpse.health = 0
            engine._stabilize()
        source = _play(engine, self.repository, 10154110)
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        revived = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id in {50, 51}
        ]
        self.assertEqual(len(revived), 2)
        self.assertTrue(all(unit.definition.card_id in {50, 51} for unit in revived))

    def test_shikigami_producers_and_last_words_spellboost_current_hand(self):
        oni_engine = self.fresh(seed=17)
        oni_spell = _put_hand(oni_engine, self.repository.get(10133320))
        enemy = _put_unit(oni_engine, 1, _card(60, life=5))
        trigger_spell = _put_hand(oni_engine, self.repository.get(90031310))
        oni_engine.apply(PlayCard(0, oni_engine.players[0].hand.index(trigger_spell)))
        self.assertEqual(enemy.health, 2)
        self.assertEqual(oni_spell.current_cost, 6)
        oni_engine.players[0].mana = 10
        oni_engine.apply(PlayCard(0, oni_engine.players[0].hand.index(oni_spell)))
        oni = next(unit for unit in oni_engine.players[0].board if unit.definition.card_id == 90031140)
        self.assertTrue(oni.has_keyword("突进"))
        tracked = _put_hand(oni_engine, _card(61, card_type="法术", attack=None, life=None))
        oni.health = 0
        oni_engine._stabilize()
        self.assertEqual(tracked.spellboost_count, 1)

        paper_engine = self.fresh(seed=19)
        source = _play(paper_engine, self.repository, 10331120)
        first = next(unit for unit in paper_engine.players[0].board if unit.definition.card_id == 90031130)
        _enable_evolve(paper_engine)
        paper_engine.apply(Evolve(0, source.entity_id))
        papers = [unit for unit in paper_engine.players[0].board if unit.definition.card_id == 90031130]
        self.assertEqual(len(papers), 2)
        self.assertTrue(all(paper.has_keyword("突进") for paper in papers))
        self.assertIn(first, papers)

    def test_manaria_scribe_adds_executable_random_damage_spell(self):
        engine = self.fresh(seed=23)
        enemy = _put_unit(engine, 1, _card(70, life=5))
        _play(engine, self.repository, 10831120)
        missile = next(card for card in engine.players[0].hand if card.card_id == 90031310)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, engine.players[0].hand.index(missile)))
        self.assertEqual(enemy.health, 2)

    def test_dragon_breeder_repeats_aura_token_summon_on_evolve(self):
        engine = self.fresh(seed=29)
        source = _play(engine, self.repository, 10141140)
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        dragons = [unit for unit in engine.players[0].board if unit.definition.card_id == 90041110]
        self.assertEqual(len(dragons), 2)
        self.assertTrue(all(dragon.has_keyword("威慑") for dragon in dragons))

    def test_beast_princess_countdown_activation_summons_rush_tiger(self):
        engine = self.fresh(seed=31)
        amulet = _play(engine, self.repository, 10163210)
        self.assertEqual(amulet.countdown, 2)
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertEqual(amulet.countdown, 1)
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        tiger = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90061120)
        self.assertTrue(tiger.has_keyword("突进"))

    def test_canonical_earth_sigil_token_activation_adds_one_to_stack(self):
        engine = self.fresh(seed=37)
        _play(engine, self.repository, 10732310)
        request = engine.state.pending_choice
        option = next(option for option in request.options if option.option_id == "choose_one:earth_sigils")
        engine.apply(Choose(0, option.option_id))
        sigil = engine._earth_sigil_amulets(0)[0]
        self.assertEqual(sigil.earth_sigil_count, 4)
        engine.players[0].mana = 10
        engine.apply(ActivateAmulet(0, sigil.entity_id))
        self.assertEqual(sigil.earth_sigil_count, 5)


class DatabaseAndAuditTests(unittest.TestCase):
    def test_collectibles_and_tokens_are_exact_with_real_producers(self):
        coverage = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(coverage["clause_audit_issues"], [])
        for card_id in COLLECTIBLE_IDS:
            with self.subTest(card_id=card_id):
                audit = coverage["classifications"][str(card_id)]["clause_audit"]
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(
                    audit["test_evidence"],
                    ["tests/test_real_token_producer_completion_batch.py"],
                )
        for card_id in TOKEN_IDS:
            with self.subTest(token_clause_id=card_id):
                audit = coverage["classifications"][str(card_id)]["clause_audit"]
                self.assertEqual(audit["status"], "token_separate_audit")
                self.assertIsNone(audit["audit_validation_error"])
                self.assertEqual(
                    audit["test_evidence"],
                    ["tests/test_real_token_producer_completion_batch.py"],
                )

        token_report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        tokens = {card["card_id"]: card for card in token_report["cards"]}
        for card_id in TOKEN_IDS:
            with self.subTest(token_id=card_id):
                self.assertEqual(tokens[card_id]["category"], "entry_behavior_complete")
                self.assertEqual(tokens[card_id]["explicit_coverage"], "exact")
                self.assertTrue(tokens[card_id]["authored_producers"])


if __name__ == "__main__":
    unittest.main()
