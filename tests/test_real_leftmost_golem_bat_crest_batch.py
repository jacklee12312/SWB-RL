# -*- coding: utf-8 -*-
"""Exact Rune, Nightmare, Royal, and Neutral leftmost/crest batch."""

from __future__ import annotations

import re
import sqlite3
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import CandidateExtreme
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.state import Amulet
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _destroy_units,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


BATCH_CARD_IDS = (
    10032110,
    10132120,
    10433110,
    10732110,
    10452130,
    10752110,
    10852110,
    10852120,
    10423310,
    10524120,
    10723110,
    10603210,
)

SOURCE_HASHES = {
    10032110: "8f0e44afabb64c1bc0bf0d9494acc821dea5ab0eeb84bca0e6256c2795f1e634",
    10132120: "8561bb6ffb15987528e01990e179efb934861d14ce8b1c1a1f79fbce6d9ee34e",
    10433110: "89af88d2aa0d69f2198bece11158c8c7b0ef5f4abe2c88ed3a78bd552aab3800",
    10732110: "4efe0af42ee01ebec361f7ba2abf84d69809ffefc3c7dc115829004d2985858d",
    10452130: "5deb879060a30d4eb34c063b4440c1ad5f2fcbfba00c52cb39cd4b5a03328f27",
    10752110: "217aba869a6a1625a7b325e46264f28c95737758e75439086f9185003021093f",
    10852110: "d1542c0984fe5cc86e75d6cebe11049f9dbe092e5a6c068e75c212df0c8ecdb6",
    10852120: "6c72c145827d3c9e5325da86958b1b16d9824c94dbc791b896ff4255f1d4f6de",
    10423310: "7e11781b5396a1b845329f5d6cf3ca5c7ce00a694970710982a92d3d92a4cc6b",
    10524120: "f781b4044bb25dbcde67047bcb746877b7105f6c17972c51e468fbfef9d0fd7f",
    10723110: "7a964836514ae864b4a85fc4fd984070f3966ed409e1f610cf1ae7457f2bd6d0",
    10603210: "1fd1c7f8161815474382b8565de896f830278127798862cecb72c855c34ef4d4",
}

TEST_EVIDENCE = "tests/test_real_leftmost_golem_bat_crest_batch.py"


def _choose(engine, option_id: str) -> None:
    request = engine.state.pending_choice
    engine.apply(Choose(request.player_index, option_id))


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


class LeftmostGolemBatCrestBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 8701):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_every_card_and_generic_filters(self):
        self.assertEqual(self.rulebook.intrinsic_keywords_for(10732110), ("突进",))
        self.assertEqual(self.rulebook.countdown_for(10603210), 2)
        self.assertEqual(
            self.rulebook.emblem_def("unkei_goldbloom").countdown,
            4,
        )
        listener = self.rulebook.listeners_for(10852110)[0]
        self.assertEqual(listener.event_filter.card_name, "蝙蝠")
        leftmost = self.rulebook.operations_for(10423310, Trigger.PLAY)[0]
        leftmost = leftmost.choose_one_options[0].operations[0]
        self.assertIs(leftmost.candidate_extreme, CandidateExtreme.LEFTMOST)
        erosion = self.rulebook.operations_for(10603210, Trigger.TURN_END)[0]
        self.assertEqual(erosion.board_filter.exclude_tribe_name, "侵蚀者")
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

    def test_remi_rami_earth_rite_and_super_evolve_target_contract(self):
        engine = self.fresh(seed=3)
        _put_earth_sigils(engine, 1)
        source = _play(engine, self.repository, 10032110)
        guardian = next(
            entity
            for entity in engine.players[0].board
            if entity.definition.card_id == 90031120
        )
        self.assertFalse(any(isinstance(card, Amulet) for card in engine.players[0].board))

        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose_entity(engine, guardian.entity_id)
        self.assertTrue(guardian.evolved)
        self.assertEqual((guardian.attack, guardian.health), (8, 8))

        insufficient = self.fresh(seed=5)
        _play(insufficient, self.repository, 10032110)
        self.assertFalse(any(
            entity.definition.card_id == 90031120
            for entity in insufficient.players[0].board
        ))

        no_target = self.fresh(seed=7)
        source = _play(no_target, self.repository, 10032110)
        _enable_evolution(no_target, super_evolve=True)
        no_target.apply(SuperEvolve(0, source.entity_id))
        self.assertTrue(source.super_evolved)
        self.assertIsNone(no_target.state.pending_choice)

    def test_emmylou_spellboost_and_post_summon_golem_count(self):
        engine = self.fresh(seed=11)
        source_hand = _put_hand(engine, self.repository.get(10132120))
        spell = _put_hand(engine, self.repository.get(10423310))
        engine.apply(PlayCard(0, engine.players[0].hand.index(spell)))
        _choose(engine, "choose_one:restore_leader")
        self.assertEqual((source_hand.spellboost_count, source_hand.current_cost), (1, 4))

        source = _play(engine, self.repository, 10132120)
        _put_unit(
            engine,
            0,
            _card(987002, tribe_name="巨像", attack=2, life=4),
        )
        enemy_a = _put_unit(engine, 1, _card(987003, life=5))
        enemy_b = _put_unit(engine, 1, _card(987004, life=2))
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))

        self.assertTrue(any(
            entity.definition.card_id == 90031110
            for entity in engine.players[0].board
        ))
        self.assertEqual(enemy_a.health, 3)
        self.assertNotIn(enemy_b, engine.players[1].board)

    def test_elmott_requires_target_silences_then_damages_and_stale_target_skips(self):
        engine = self.fresh(seed=13)
        enemy = _put_unit(
            engine,
            1,
            _card(987010, keywords=frozenset({"守护"}), life=6),
        )
        _play(engine, self.repository, 10433110)
        _choose_entity(engine, enemy.entity_id)
        self.assertFalse(enemy.has_keyword("守护"))
        self.assertEqual(enemy.health, 3)

        no_target = self.fresh(seed=17)
        source = _play(no_target, self.repository, 10433110)
        self.assertIn(source, no_target.players[0].board)
        self.assertIsNone(no_target.state.pending_choice)

        stale = self.fresh(seed=19)
        enemy = _put_unit(stale, 1, _card(987011, life=6))
        _play(stale, self.repository, 10433110)
        request = stale.state.pending_choice
        stale.players[1].board.remove(enemy)
        stale._send_to_graveyard(
            1,
            enemy.definition,
            "batch_test_target_left_play",
            source_entity_id=enemy.entity_id,
        )
        stale.apply(Choose(0, request.options[0].option_id))
        self.assertEqual(enemy.health, 6)
        self.assertIsNone(stale.state.pending_choice)

    def test_elmott_crest_triggers_only_at_owner_turn_start(self):
        engine = self.fresh(seed=23)
        enemy = _put_unit(engine, 1, _card(987020, life=5))
        source = _play(engine, self.repository, 10433110)
        _choose_entity(engine, enemy.entity_id)
        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(engine.players[0].emblems[0].definition.emblem_id, "elmott_remembrance_aflame")

        enemy_health = engine.players[1].health
        engine.apply(EndTurn(0))
        self.assertEqual(engine.players[1].health, enemy_health)
        engine.apply(EndTurn(1))
        self.assertEqual(engine.players[1].health, enemy_health - 1)

    def test_charming_monster_last_words_and_super_evolve_earth_rite_capacity(self):
        engine = self.fresh(seed=29)
        _put_earth_sigils(engine, 2)
        source = _play(engine, self.repository, 10732110)
        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, source.entity_id))
        copies = [
            entity for entity in engine.players[0].board
            if entity.definition.card_id == 10732110
        ]
        self.assertEqual(len(copies), 3)
        self.assertTrue(all(copy.has_keyword("突进") for copy in copies))

        _destroy_units(engine, source)
        sigil = next(entity for entity in engine.players[0].board if isinstance(entity, Amulet))
        self.assertEqual(sigil.earth_sigil_count, 2)

        insufficient = self.fresh(seed=31)
        _put_earth_sigils(insufficient, 1)
        source = _play(insufficient, self.repository, 10732110)
        _enable_evolution(insufficient, super_evolve=True)
        insufficient.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(
            sum(entity.definition.card_id == 10732110 for entity in insufficient.players[0].board),
            1,
        )

    def test_baal_modes_handle_random_other_and_empty_candidate_sets(self):
        growth = self.fresh(seed=37)
        ally = _put_unit(growth, 0, _card(987030, attack=2, life=3))
        source = _play(growth, self.repository, 10452130)
        _choose(growth, "choose_one:resonant_growth")
        self.assertEqual((ally.attack, ally.health), (3, 4))
        self.assertEqual((source.attack, source.health), (3, 3))

        alone = self.fresh(seed=41)
        source = _play(alone, self.repository, 10452130)
        _choose(alone, "choose_one:resonant_growth")
        self.assertEqual((source.attack, source.health), (3, 3))

        damage = self.fresh(seed=43)
        enemy = _put_unit(damage, 1, _card(987031, life=5))
        _play(damage, self.repository, 10452130)
        _choose(damage, "choose_one:resonant_damage")
        self.assertEqual(enemy.health, 2)

        no_enemy = self.fresh(seed=47)
        _play(no_enemy, self.repository, 10452130)
        _choose(no_enemy, "choose_one:resonant_damage")
        self.assertIsNone(no_enemy.state.pending_choice)

    def test_baal_random_mode_is_seed_deterministic(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=53)
            for index in range(3):
                _put_unit(engine, 0, _card(987040 + index, attack=2, life=3))
            _play(engine, self.repository, 10452130)
            _choose(engine, "choose_one:resonant_growth")
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_tightrope_cat_target_no_target_stale_target_and_capacity(self):
        engine = self.fresh(seed=59)
        enemy = _put_unit(engine, 1, _card(987050, life=5))
        _play(engine, self.repository, 10752110)
        _choose_entity(engine, enemy.entity_id)
        self.assertEqual(enemy.health, 2)
        self.assertTrue(any(
            entity.definition.card_id == 90051110
            for entity in engine.players[0].board
        ))

        no_target = self.fresh(seed=61)
        _play(no_target, self.repository, 10752110)
        self.assertTrue(any(
            entity.definition.card_id == 90051110
            for entity in no_target.players[0].board
        ))

        stale = self.fresh(seed=67)
        enemy = _put_unit(stale, 1, _card(987051, life=5))
        _play(stale, self.repository, 10752110)
        request = stale.state.pending_choice
        stale.players[1].board.remove(enemy)
        stale._send_to_graveyard(
            1,
            enemy.definition,
            "batch_test_target_left_play",
            source_entity_id=enemy.entity_id,
        )
        stale.apply(Choose(0, request.options[0].option_id))
        self.assertTrue(any(
            entity.definition.card_id == 90051110
            for entity in stale.players[0].board
        ))

        full = self.fresh(seed=71)
        for index in range(4):
            _put_unit(full, 0, _card(987060 + index))
        _play(full, self.repository, 10752110)
        self.assertEqual(len(full.players[0].board), 5)
        self.assertFalse(any(
            entity.definition.card_id == 90051110
            for entity in full.players[0].board
        ))

    def test_fiole_summons_only_available_bats_and_listener_grants_rush(self):
        engine = self.fresh(seed=73)
        for index in range(2):
            _put_unit(engine, 0, _card(987070 + index))
        source = _play(engine, self.repository, 10852110)
        bats = [
            entity for entity in engine.players[0].board
            if entity.definition.card_id == 90051120
        ]
        self.assertEqual(len(bats), 2)
        self.assertTrue(all(bat.has_keyword("突进") for bat in bats))
        self.assertFalse(source.has_keyword("突进"))

    def test_marsha_repeats_fanfare_on_evolve_for_both_leaders_and_enemy_board(self):
        engine = self.fresh(seed=79)
        enemy = _put_unit(engine, 1, _card(987080, life=4))
        source = _play(engine, self.repository, 10852120)
        self.assertEqual((engine.players[0].health, engine.players[1].health), (19, 19))
        self.assertEqual(enemy.health, 3)

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual((engine.players[0].health, engine.players[1].health), (18, 18))
        self.assertEqual(enemy.health, 2)

    def test_knightly_ardor_leftmost_and_barrier_modes_filter_royal_followers(self):
        leftmost = self.fresh(seed=83)
        neutral = _put_unit(leftmost, 0, _card(987090, class_name="中立"))
        royal_left = _put_unit(leftmost, 0, _card(987091, class_name="皇家护卫"))
        royal_right = _put_unit(leftmost, 0, _card(987092, class_name="皇家护卫"))
        _play(leftmost, self.repository, 10423310)
        _choose(leftmost, "choose_one:leftmost_double_attack")
        self.assertEqual(
            (neutral.attacks_per_turn, royal_left.attacks_per_turn, royal_right.attacks_per_turn),
            (1, 2, 1),
        )

        barrier = self.fresh(seed=89)
        neutral = _put_unit(barrier, 0, _card(987093, class_name="中立", attack=2, life=3))
        royal = _put_unit(barrier, 0, _card(987094, class_name="皇家护卫", attack=2, life=3))
        _play(barrier, self.repository, 10423310)
        _choose(barrier, "choose_one:royal_barrier")
        self.assertEqual((neutral.attack, neutral.health), (2, 3))
        self.assertEqual((royal.attack, royal.health), (3, 4))
        self.assertTrue(royal.has_keyword("屏障"))

    def test_knightly_ardor_resource_heal_and_empty_leftmost_modes(self):
        resource = self.fresh(seed=97)
        resource.players[0].evolution_points = 0
        _play(resource, self.repository, 10423310)
        _choose(resource, "choose_one:restore_pp_ep")
        self.assertEqual((resource.players[0].mana, resource.players[0].evolution_points), (7, 1))

        healing = self.fresh(seed=101)
        healing.players[0].health = 10
        _play(healing, self.repository, 10423310)
        _choose(healing, "choose_one:restore_leader")
        self.assertEqual(healing.players[0].health, 16)

        empty = self.fresh(seed=103)
        _play(empty, self.repository, 10423310)
        _choose(empty, "choose_one:leftmost_double_attack")
        self.assertIsNone(empty.state.pending_choice)

    def test_unkei_banish_coin_no_target_full_hand_and_countdown_crest(self):
        engine = self.fresh(seed=107)
        enemy = _put_unit(engine, 1, _card(987100, life=6))
        source = _play(engine, self.repository, 10524120)
        _choose_entity(engine, enemy.entity_id)
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertEqual(engine.players[0].hand[-1].definition.card_id, 90021350)

        no_target = self.fresh(seed=109)
        _play(no_target, self.repository, 10524120)
        self.assertEqual(no_target.players[0].hand[-1].definition.card_id, 90021350)

        full_hand = self.fresh(seed=113)
        enemy = _put_unit(full_hand, 1, _card(987101))
        _play(full_hand, self.repository, 10524120)
        while len(full_hand.players[0].hand) < full_hand.config.max_hand:
            _put_hand(full_hand, _card(987110 + len(full_hand.players[0].hand)))
        _choose_entity(full_hand, enemy.entity_id)
        self.assertFalse(any(
            card.definition.card_id == 90021350
            for card in full_hand.players[0].hand
        ))

        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, source.entity_id))
        emblem = engine.players[0].emblems[0]
        self.assertEqual((emblem.definition.emblem_id, emblem.countdown), ("unkei_goldbloom", 4))
        coins_before = sum(
            card.definition.card_id == 90021350
            for card in engine.players[0].hand
        )
        engine.apply(EndTurn(0))
        self.assertEqual(
            sum(card.definition.card_id == 90021350 for card in engine.players[0].hand),
            coins_before + 1,
        )

    def test_resonant_squad_capacity_and_both_modes_exclude_source(self):
        rush = self.fresh(seed=127)
        existing = [
            _put_unit(rush, 0, _card(987120 + index, attack=2, life=3))
            for index in range(3)
        ]
        source = _play(rush, self.repository, 10723110)
        knights = [
            entity for entity in rush.players[0].board
            if entity.definition.card_id == 90021110
        ]
        self.assertEqual(len(knights), 1)
        _choose(rush, "choose_one:squad_rush")
        self.assertEqual((source.attack, source.health), (3, 2))
        self.assertTrue(all(unit.has_keyword("突进") for unit in existing + knights))
        self.assertTrue(all(unit.attack == 3 for unit in existing))

        ward = self.fresh(seed=131)
        source = _play(ward, self.repository, 10723110)
        _choose(ward, "choose_one:squad_ward")
        knights = [
            entity for entity in ward.players[0].board
            if entity.definition.card_id == 90021110
        ]
        self.assertEqual(len(knights), 3)
        self.assertTrue(all(knight.has_keyword("守护") for knight in knights))
        self.assertTrue(all(knight.health == 2 for knight in knights))
        self.assertFalse(source.has_keyword("守护"))

    def test_dark_dimension_excludes_encroachers_and_obeys_turn_and_countdown(self):
        engine = self.fresh(seed=137)
        normal_own = _put_unit(engine, 0, _card(987130, life=4))
        encroacher_own = _put_unit(engine, 0, _card(987131, tribe_name="侵蚀者", life=4))
        normal_enemy = _put_unit(engine, 1, _card(987132, life=4))
        encroacher_enemy = _put_unit(engine, 1, _card(987133, tribe_name="侵蚀者", life=4))
        amulet = _play(engine, self.repository, 10603210)
        self.assertEqual(amulet.countdown, 2)

        engine.apply(EndTurn(0))
        self.assertEqual(
            (normal_own.health, encroacher_own.health, normal_enemy.health, encroacher_enemy.health),
            (2, 4, 2, 4),
        )
        engine.apply(EndTurn(1))
        self.assertEqual(amulet.countdown, 1)
        engine.apply(EndTurn(0))
        self.assertNotIn(normal_own, engine.players[0].board)
        self.assertNotIn(normal_enemy, engine.players[1].board)
        self.assertIn(encroacher_own, engine.players[0].board)
        self.assertIn(encroacher_enemy, engine.players[1].board)
        engine.apply(EndTurn(1))
        self.assertNotIn(amulet, engine.players[0].board)

    def test_filter_schema_rejects_empty_exclusion_and_preserves_legacy_payloads(self):
        operation = _parse_operation(
            {
                "kind": "damage_unit",
                "target": "all_units",
                "target_exclude_tribe_name_filter": "侵蚀者",
                "amount": 2,
            },
            "batch.json",
            10603210,
        )
        self.assertEqual(operation.board_filter.exclude_tribe_name, "侵蚀者")
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            _parse_operation(
                {
                    "kind": "damage_unit",
                    "target": "all_units",
                    "target_exclude_tribe_name_filter": "",
                    "amount": 2,
                },
                "batch.json",
                10603210,
            )
        legacy = _parse_operation(
            {
                "kind": "damage_unit",
                "target": "all_units",
                "target_tribe_name_filter": "巨像",
                "amount": 1,
            },
            "legacy.json",
            1,
        )
        self.assertEqual(legacy.board_filter.tribe_name, "巨像")
        self.assertIsNone(legacy.board_filter.exclude_tribe_name)

    def test_rl_mask_exposes_play_and_all_four_followup_modes(self):
        deck = [_card(987200 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=139,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=139)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        card = _put_hand(env.core, self.repository.get(10423310))
        play = PlayCard(0, env.players[0].hand.index(card))
        env.invalidate_cache(reason="leftmost mode setup")
        self.assertTrue(env.action_mask()[env._encode_command(play)])

        env.core.apply(play)
        env.invalidate_cache(reason="leftmost pending modes")
        mask = env.action_mask()
        request = env.core.state.pending_choice
        self.assertEqual(len(request.options), 4)
        for option in request.options:
            command = Choose(request.player_index, option.option_id)
            self.assertTrue(mask[env._encode_command(command)])


class LeftmostGolemBatCrestDatabaseAuditTests(unittest.TestCase):
    def test_database_text_modes_and_references_match_reviewed_cards(self):
        expected_phrases = {
            10032110: ("Earth Rite", "Super-Evolve", "Guardian Golem", "+3/+3"),
            10132120: ("On Spellboost", "Clay Golem", "all enemy followers"),
            10433110: ("remove all abilities", "Crest: Elmott"),
            10732110: ("Rush", "Last Words", "Earth Rite", "2 copies"),
            10452130: ("Select a Mode", "another random allied", "random enemy"),
            10752110: ("Fanfare", "Skeleton", "Evolve"),
            10852110: ("3 copies", "Bat", "Rush"),
            10852120: ("all enemy followers and both leaders", "Evolve"),
            10423310: ("leftmost", "Barrier", "evolution point", "Restore 6"),
            10524120: ("banish", "Glittering Gold", "Crest: Unkei"),
            10723110: ("3 copies", "Knight", "Rush", "Ward"),
            10603210: ("Countdown", "non-Encroacher", "2 damage"),
        }
        expected_references = {
            10032110: [90031120],
            10132120: [90031110],
            10433110: [],
            10732110: [10732110],
            10452130: [],
            10752110: [90051110],
            10852110: [90051120],
            10852120: [],
            10423310: [],
            10524120: [90021350],
            10723110: [90021110],
            10603210: [],
        }
        mode_cards = {10433110, 10524120}
        with sqlite3.connect("data/cards.sqlite3") as connection:
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
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        1 if card_id in mode_cards else 0,
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
