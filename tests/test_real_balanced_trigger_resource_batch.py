# -*- coding: utf-8 -*-
"""Exact balanced batch for attacks, tokens, resources, modes, and listeners."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Attack, Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import Amulet, CostModifier, HandCard, Unit
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


BATCH_CARD_IDS = (
    10411110,
    10212110,
    10811110,
    10611110,
    10614110,
    10213110,
    10123110,
    10522120,
    10522110,
    10134310,
    10632120,
    10132130,
    10533110,
    10542110,
    10661110,
)

SOURCE_HASHES = {
    10411110: "c5fd18fce0d40afa663d9d8cd6e1ffa96fcfdc5453e7b485ec903b003525c57e",
    10212110: "5ee5e160f0a0f2f2f51bc1805119c2487ca0bc375c18867abff0725c22e11709",
    10811110: "50d87ed9254c655320831d2e2a4949bbfdacb3b3a03ef7ea5b199f85194cf904",
    10611110: "945d9c9d8cd59c9c8d20521a1f7641fbfb7f063c8cfaa6b819bc8d4981f3fcbb",
    10614110: "1fbb138a261daea912aaebe0b15e72df49777f8c714dec2371199a6f056cada2",
    10213110: "ace3c2f31898178382efdd8db388ce9abe0721ddd5dfa2fbe17b0d1d9fa68acc",
    10123110: "747170dc5cb1f40f21dd977d2840b1c2129e15745700d969bf5a92b2e1810cf0",
    10522120: "9ed8f99ad12b94c314cd83a5fffe1d07ea914d55abc5095c81b5ae94a24970d9",
    10522110: "dc296783bff5011ff86e0b7349803caa6f08ebe88b577022c5cc4068a7be84d1",
    10134310: "64f345669caaab3eec0410dd2d6fed8e31e784c2629e96a137b57bd6115e7d57",
    10632120: "da97edcff3c71c0ee8df7df5c919e9e0a6766476f3a112c74680a2237e5d2aba",
    10132130: "3cc17750b785ae0d8e9e0f2a436248f9f2ac4990262e870939e5420968962298",
    10533110: "e6937d03ed1561127bde505624300d02acc33ebf81c117935d1b0f5bf6b352e4",
    10542110: "5c821a8cf0997c46d5b6615b8041089af1c1035d5d5b29259855a72504a98d79",
    10661110: "0638286c73db3ad2ecd9222236d994422dc91821e27688970aed3dd3576cf155",
}

TEST_EVIDENCE = "tests/test_real_balanced_trigger_resource_batch.py"


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolution(engine, *, super_evolve: bool = False) -> None:
    player = engine.players[0]
    if super_evolve:
        player.turns_started = engine.config.first_player_super_evolution_unlock_turn
        player.super_evolution_points = max(1, player.super_evolution_points)
        player.super_evolved_this_turn = False
    else:
        player.turns_started = engine.config.evolution_unlock_turn
        player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _put_earth_sigils(engine, count: int) -> Amulet:
    sigil = Amulet(
        definition=_card(
            90031210,
            card_set_id=90000,
            name="大地之魔片",
            card_type="护符",
            attack=None,
            life=None,
            keywords=frozenset({"土之印"}),
            is_collectible=False,
        ),
        entity_id=engine.state.allocate_entity_id(),
        earth_sigil_count=count,
        entered_turn=engine.turn,
        origin=CardOrigin.TOKEN,
    )
    engine.players[0].board.append(sigil)
    return sigil


def _draw_via_effect(engine, amount: int = 1) -> None:
    engine._start_effects(
        _card(989999, name="测试抽牌来源", card_type="法术", attack=None, life=None),
        None,
        (EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, amount),),
        controller=0,
        label="测试抽牌",
    )


class BalancedTriggerResourceBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 8101):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_all_cards_modes_listeners_and_keywords(self):
        self.assertEqual(self.rulebook.attacks_per_turn(10411110), 2)
        self.assertEqual(self.rulebook.attacks_per_turn(10614110), 2)
        self.assertEqual(self.rulebook.attacks_per_turn(10123110), 2)
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10661110),
            ("必杀", "守护"),
        )
        self.assertEqual(
            [mode.mode_id for mode in self.rulebook.modes_for(10632120)],
            ["enhance_6"],
        )
        crystal = self.rulebook.modes_for(10661110)[0]
        self.assertEqual(
            (crystal.mode_id, crystal.mode_type, crystal.cost, crystal.countdown),
            ("crystallize_2", "crystallize", 2, 3),
        )
        self.assertEqual(len(self.rulebook.listeners_for(10522110)), 1)
        self.assertEqual(len(self.rulebook.listeners_for(10533110)), 1)
        self.assertEqual(len(self.rulebook.listeners_for(10661110)), 1)
        ruled = {
            card_id
            for card_id in BATCH_CARD_IDS
            if any(
                self.rulebook.operations_for(card_id, trigger)
                for trigger in Trigger
            )
            or self.rulebook.listeners_for(card_id)
        }
        self.assertEqual(ruled, set(BATCH_CARD_IDS))

    def test_strike_heals_all_allies_and_allows_exactly_two_attacks(self):
        engine = self.fresh(seed=3)
        engine.players[0].health = 12
        source = _play(engine, self.repository, 10411110)
        ally = _put_unit(engine, 0, _card(981001, life=6))
        source.health = 2
        ally.health = 1
        targets = [
            _put_unit(engine, 1, _card(981010 + index, attack=1, life=20))
            for index in range(2)
        ]
        source.can_attack = True

        for target in targets:
            engine.apply(Attack(0, source.entity_id, target.entity_id))

        self.assertEqual(engine.players[0].health, 18)
        self.assertEqual((source.health, ally.health), (5, 6))
        self.assertEqual(source.attacks_remaining, 0)
        self.assertNotIn(Attack(0, source.entity_id, None), engine.legal_commands())

    def test_forest_token_fanfares_and_pixie_evolve_respect_capacity_and_traits(self):
        carbuncle = self.fresh(seed=5)
        _play(carbuncle, self.repository, 10212110)
        self.assertEqual(
            [entity.definition.card_id for entity in carbuncle.players[0].board],
            [10212110, 10112130, 10112130],
        )
        self.assertTrue(carbuncle.players[0].board[0].has_keyword("守护"))

        shortage = self.fresh(seed=7)
        for index in range(shortage.config.max_board - 2):
            _put_unit(shortage, 0, _card(981100 + index))
        _play(shortage, self.repository, 10212110)
        self.assertEqual(
            sum(entity.definition.card_id == 10112130 for entity in shortage.players[0].board),
            1,
        )

        cynthia = self.fresh(seed=11)
        non_pixie = _put_unit(cynthia, 0, _card(981110, attack=3, life=4))
        source = _play(cynthia, self.repository, 10213110)
        fairies = [
            entity for entity in cynthia.players[0].board
            if entity.definition.card_id == 90011110
        ]
        self.assertEqual(len(fairies), 2)
        _enable_evolution(cynthia)
        cynthia.apply(Evolve(0, source.entity_id))
        self.assertEqual([fairy.attack for fairy in fairies], [2, 2])
        self.assertEqual(non_pixie.attack, 3)

    def test_arboreal_core_random_destroy_uses_count_difference_and_seed(self):
        def resolved(seed: int):
            engine = self.fresh(seed=seed)
            enemies = [
                _put_unit(engine, 1, _card(981200 + index, life=5))
                for index in range(4)
            ]
            _play(engine, self.repository, 10811110)
            return (
                tuple(entity.entity_id for entity in engine.players[1].board),
                engine.deterministic_fingerprint(),
                enemies,
            )

        first = resolved(13)
        second = resolved(13)
        self.assertEqual(first[:2], second[:2])
        self.assertEqual(len(first[0]), 1)

        non_positive = self.fresh(seed=17)
        _put_unit(non_positive, 0, _card(981220))
        _put_unit(non_positive, 1, _card(981221))
        rng_before = non_positive.random.getstate()
        _play(non_positive, self.repository, 10811110)
        self.assertEqual(len(non_positive.players[1].board), 1)
        self.assertEqual(non_positive.random.getstate(), rng_before)

    def test_springbloom_producers_cover_hand_evolve_capacity_and_super_targeting(self):
        rose = self.fresh(seed=19)
        rose.players[0].health = 15
        source = _play(rose, self.repository, 10611110)
        self.assertEqual([card.card_id for card in rose.players[0].hand], [90011120])
        _enable_evolution(rose)
        rose.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [entity.definition.card_id for entity in rose.players[0].board],
            [10611110, 90011120],
        )

        queen = self.fresh(seed=23)
        source = _play(queen, self.repository, 10614110)
        self.assertEqual(
            sum(entity.definition.card_id == 90011120 for entity in queen.players[0].board),
            3,
        )
        self.assertEqual(source.attacks_remaining, 2)
        target = _put_unit(queen, 1, _card(981230, life=20))
        _enable_evolution(queen, super_evolve=True)
        queen.apply(SuperEvolve(0, source.entity_id))
        _choose_entity(queen, target.entity_id)
        self.assertNotIn(target, queen.players[1].board)

        no_target = self.fresh(seed=29)
        source = _play(no_target, self.repository, 10614110)
        _enable_evolution(no_target, super_evolve=True)
        command = SuperEvolve(0, source.entity_id)
        self.assertIn(command, no_target.legal_commands())
        no_target.apply(command)
        self.assertTrue(source.super_evolved)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertFalse(no_target.players[1].board)

    def test_chivalrous_bandit_strike_adds_barrier_and_knights_for_two_attacks(self):
        engine = self.fresh(seed=31)
        source = _play(engine, self.repository, 10123110)
        targets = [
            _put_unit(engine, 1, _card(981300 + index, attack=20, life=20))
            for index in range(2)
        ]
        for target in targets:
            engine.apply(Attack(0, source.entity_id, target.entity_id))
            self.assertIn(source, engine.players[0].board)
        self.assertEqual(
            sum(entity.definition.card_id == 90021110 for entity in engine.players[0].board),
            2,
        )
        self.assertEqual(source.attacks_remaining, 0)

    def test_royal_turn_end_spell_threshold_evolve_tokens_and_hand_capacity(self):
        below = self.fresh(seed=37)
        _put_hand(below, _card(981400, card_type="法术", attack=None, life=None))
        enemy = _put_unit(below, 1, _card(981401, life=8))
        _play(below, self.repository, 10522120)
        below.apply(EndTurn(0))
        self.assertEqual(enemy.health, 8)

        active = self.fresh(seed=41)
        for index in range(2):
            _put_hand(active, _card(981410 + index, card_type="法术", attack=None, life=None))
        enemy = _put_unit(active, 1, _card(981412, life=8))
        _play(active, self.repository, 10522120)
        active.apply(EndTurn(0))
        self.assertEqual(enemy.health, 3)

        evolved = self.fresh(seed=42)
        source = _play(evolved, self.repository, 10522120)
        _enable_evolution(evolved)
        evolved.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            sum(card.card_id == 90021350 for card in evolved.players[0].hand),
            2,
        )

        full = self.fresh(seed=43)
        source = _play(full, self.repository, 10522120)
        for index in range(full.config.max_hand - 1):
            _put_hand(full, _card(981420 + index))
        _enable_evolution(full)
        full.apply(Evolve(0, source.entity_id))
        self.assertEqual(len(full.players[0].hand), full.config.max_hand)
        self.assertTrue(any(
            card.definition.card_id == 90021350 and card.entry_cause == "hand_full"
            for card in full.players[0].graveyard
        ))

    def test_drawn_royal_cost_listener_expires_and_fanfare_draws_heals_and_rushes(self):
        engine = self.fresh(seed=47)
        engine.players[0].deck = [self.repository.get(10522110)]
        _draw_via_effect(engine)
        drawn = engine.players[0].hand[0]
        self.assertEqual(drawn.current_cost, 3)
        engine.players[0].health = 14
        engine.players[0].deck = [_card(981500)]
        engine.apply(PlayCard(0, 0))
        source = next(entity for entity in engine.players[0].board if entity.definition.card_id == 10522110)
        self.assertEqual(engine.players[0].health, 17)
        self.assertTrue(source.has_keyword("突进"))
        self.assertEqual(len(engine.players[0].hand), 1)

        expiry = self.fresh(seed=53)
        expiry.players[0].deck = [self.repository.get(10522110)]
        _draw_via_effect(expiry)
        drawn = expiry.players[0].hand[0]
        expiry.apply(EndTurn(0))
        self.assertEqual(drawn.current_cost, 6)

    def test_dimension_shift_recycles_hand_draws_boosts_and_restores_mana_deterministically(self):
        def resolved(seed: int):
            engine = self.fresh(seed=seed)
            boostable = self.repository.get(10132130)
            for _ in range(2):
                _put_hand(engine, boostable)
            source = _put_hand(engine, self.repository.get(10134310))
            source.apply_spellboost(8)
            engine.players[0].deck = [boostable for _ in range(5)]
            engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
            return (
                tuple(card.entity_id for card in engine.players[0].hand),
                tuple(card.spellboost_count for card in engine.players[0].hand),
                len(engine.players[0].deck),
                engine.players[0].mana,
                engine.deterministic_fingerprint(),
            )

        first = resolved(59)
        second = resolved(59)
        self.assertEqual(first, second)
        self.assertEqual((len(first[0]), first[1], first[2], first[3]), (5, (6, 6, 6, 6, 6), 2, 10))

    def test_grimoire_enhance_capacity_rush_and_simultaneous_last_words_spellboost(self):
        engine = self.fresh(seed=61)
        target = _put_hand(engine, self.repository.get(10132130))
        _play(engine, self.repository, 10632120, mode_id="enhance_6")
        copies = [entity for entity in engine.players[0].board if entity.definition.card_id == 10632120]
        self.assertEqual(len(copies), 3)
        self.assertTrue(all(entity.has_keyword("突进") for entity in copies))
        for copy in copies:
            copy.health = 0
        engine._stabilize()
        self.assertEqual(target.spellboost_count, 3)

        shortage = self.fresh(seed=67)
        for index in range(3):
            _put_unit(shortage, 0, _card(981600 + index))
        _play(shortage, self.repository, 10632120, mode_id="enhance_6")
        self.assertEqual(
            sum(entity.definition.card_id == 10632120 for entity in shortage.players[0].board),
            2,
        )

    def test_spellboost_student_uses_source_count_then_evolve_boosts_hand_twice(self):
        engine = self.fresh(seed=71)
        target = _put_hand(engine, self.repository.get(10132130))
        source_card = _put_hand(engine, self.repository.get(10132130))
        source_card.apply_spellboost(4)
        enemies = [
            _put_unit(engine, 1, _card(981700 + index, life=7))
            for index in range(2)
        ]
        engine.apply(PlayCard(0, engine.players[0].hand.index(source_card)))
        source = next(entity for entity in engine.players[0].board if entity.definition.card_id == 10132130)
        self.assertEqual([enemy.health for enemy in enemies], [3, 3])
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(target.spellboost_count, 2)

    def test_golem_listener_spends_each_sigil_and_evolves_only_golem_followers(self):
        active = self.fresh(seed=73)
        _put_earth_sigils(active, 2)
        _play(active, self.repository, 10533110)
        golems = [entity for entity in active.players[0].board if entity.definition.card_id == 90031120]
        self.assertEqual(len(golems), 2)
        self.assertTrue(all(golem.evolved for golem in golems))
        self.assertEqual(active.players[0].earth_sigils, 0)

        insufficient = self.fresh(seed=79)
        _put_earth_sigils(insufficient, 1)
        _play(insufficient, self.repository, 10533110)
        golems = [entity for entity in insufficient.players[0].board if entity.definition.card_id == 90031120]
        self.assertEqual(sum(golem.evolved for golem in golems), 1)
        self.assertEqual(insufficient.players[0].earth_sigils, 0)

    def test_dragon_target_damage_continues_to_summon_on_empty_or_stale_target(self):
        selected = self.fresh(seed=83)
        enemy = _put_unit(selected, 1, _card(981800, life=9))
        _play(selected, self.repository, 10542110)
        _choose_entity(selected, enemy.entity_id)
        self.assertEqual(enemy.health, 2)
        self.assertTrue(any(entity.definition.card_id == 90041120 for entity in selected.players[0].board))

        empty = self.fresh(seed=89)
        source = _put_hand(empty, self.repository.get(10542110))
        command = PlayCard(0, empty.players[0].hand.index(source))
        self.assertIn(command, empty.legal_commands())
        empty.apply(command)
        self.assertIsNone(empty.state.pending_choice)
        self.assertTrue(any(entity.definition.card_id == 90041120 for entity in empty.players[0].board))

        stale = self.fresh(seed=97)
        enemy = _put_unit(stale, 1, _card(981810, life=9))
        _play(stale, self.repository, 10542110)
        stale.players[1].board.remove(enemy)
        _choose_entity(stale, enemy.entity_id)
        self.assertTrue(any(entity.definition.card_id == 90041120 for entity in stale.players[0].board))

    def test_bishop_entry_keywords_crystallize_last_words_and_source_form_guard(self):
        normal = self.fresh(seed=101)
        normal.players[0].health = 14
        source = _play(normal, self.repository, 10661110)
        self.assertEqual(normal.players[0].health, 16)
        self.assertTrue(source.has_keyword("必杀"))
        self.assertTrue(source.has_keyword("守护"))
        source.health = 0
        normal._stabilize()
        self.assertFalse(any(entity.definition.card_id == 10661110 for entity in normal.players[0].board))

        crystal = self.fresh(seed=103)
        crystal.players[0].health = 14
        amulet = _play(crystal, self.repository, 10661110, mode_id="crystallize_2")
        self.assertIsInstance(amulet, Amulet)
        self.assertEqual((amulet.countdown, crystal.players[0].health), (3, 14))
        amulet.countdown = 1
        while amulet in crystal.players[0].board:
            crystal.apply(EndTurn(crystal.current_player))
        summoned = next(entity for entity in crystal.players[0].board if entity.definition.card_id == 10661110)
        self.assertIsInstance(summoned, Unit)
        self.assertEqual(crystal.players[0].health, 16)
        self.assertTrue(summoned.has_keyword("守护"))

    def test_rl_mask_exposes_modes_attacks_and_independent_no_target_play(self):
        deck = [_card(982000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=107,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=107)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10

        grimoire = _put_hand(env.core, self.repository.get(10632120))
        normal = PlayCard(0, env.players[0].hand.index(grimoire))
        enhance = PlayCard(0, env.players[0].hand.index(grimoire), "enhance_6")
        env.invalidate_cache(reason="enhance setup")
        self.assertTrue(env.action_mask()[env._encode_command(normal)])
        self.assertTrue(env.action_mask()[env._encode_command(enhance)])

        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        bishop = _put_hand(env.core, self.repository.get(10661110))
        crystal = PlayCard(0, env.players[0].hand.index(bishop), "crystallize_2")
        env.invalidate_cache(reason="crystallize setup")
        self.assertTrue(env.action_mask()[env._encode_command(crystal)])

        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        dragon = _put_hand(env.core, self.repository.get(10542110))
        dragon_play = PlayCard(0, env.players[0].hand.index(dragon))
        env.invalidate_cache(reason="no target summon setup")
        self.assertTrue(env.action_mask()[env._encode_command(dragon_play)])


class BalancedTriggerResourceDatabaseAuditTests(unittest.TestCase):
    def test_database_text_modes_and_references_match_reviewed_cards(self):
        expected_phrases = {
            10411110: ("attack 2 times", "Strike", "all allies"),
            10212110: ("Fanfare", "2 copies", "Baby Carbuncle", "Ward"),
            10811110: ("Destroy X random", "minus", "allied followers"),
            10611110: ("Springbloom Fairy", "Evolve", "Summon"),
            10614110: ("3 copies", "attack 2 times", "Super-Evolve", "destroy it"),
            10213110: ("2 copies", "Fairy", "Evolve", "Pixie followers"),
            10123110: ("Rush", "attack 2 times", "Barrier", "Knight"),
            10522120: ("at least 2 spells", "5 damage", "Glittering Gold"),
            10522110: ("draw this card", "cost to 3", "Draw a card", "Rush"),
            10134310: ("On Spellboost", "Return your hand", "Draw 5", "Fully recover"),
            10632120: ("Enhance", "2 copies", "Rush", "Last Words", "Spellboost"),
            10132130: ("X starts at 0", "On Spellboost", "all enemy followers", "2 times"),
            10533110: ("Guardian Golem", "allied Golem", "Earth Rite", "Evolve it"),
            10542110: ("Select an enemy follower", "7 damage", "Vastwing Dragon"),
            10661110: ("enters the field", "restore 2", "Bane", "Ward", "Countdown", "Last Words"),
        }
        expected_references = {
            10411110: [],
            10212110: [10112130],
            10811110: [],
            10611110: [90011120],
            10614110: [90011120],
            10213110: [90011110],
            10123110: [90021110],
            10522120: [90021350],
            10522110: [],
            10134310: [],
            10632120: [10632120],
            10132130: [],
            10533110: [90031120],
            10542110: [90041120],
            10661110: [10661110],
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id in BATCH_CARD_IDS:
                with self.subTest(card_id=card_id):
                    texts = [
                        row[0]
                        for row in connection.execute(
                            "SELECT text_eng FROM skill_texts WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    ]
                    texts.extend(
                        row[0]
                        for row in connection.execute(
                            "SELECT text_eng FROM alt_modes WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    )
                    normalized = " ".join(re.sub(r"<[^>]+>", "", text) for text in texts)
                    for phrase in expected_phrases[card_id]:
                        self.assertIn(phrase, normalized)
                    self.assertEqual(
                        [
                            row[0]
                            for row in connection.execute(
                                "SELECT referenced_card_id FROM card_references "
                                "WHERE card_id=? ORDER BY position",
                                (card_id,),
                            )
                        ],
                        expected_references[card_id],
                    )
                    expected_modes = 1 if card_id == 10661110 else 0
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        expected_modes,
                    )

    def test_batch_cards_have_exact_clause_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in BATCH_CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    [TEST_EVIDENCE],
                )


if __name__ == "__main__":
    unittest.main()
