# -*- coding: utf-8 -*-
"""Exact cross-class crest, token-chain, mode, and activation batch."""

from __future__ import annotations

import re
import sqlite3
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import ActivateAmulet, Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import IllegalCommand
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _destroy_units,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


BATCH_CARD_IDS = (
    10403110,
    10702110,
    10114110,
    10513110,
    10423110,
    10321120,
    10644110,
    10841130,
    10163130,
    10462210,
    10174110,
)

SOURCE_HASHES = {
    10403110: "965cd4b3848f817571db5e52ad9535c73f0b8969d7d03acd457f9f6494ddef3d",
    10702110: "54b83c706140fb3ff20007aaa8d7d1f5a811c3181b0011588e84f1e9abd8ebb8",
    10114110: "55ab51a70fc29dbd546f72aa615f271566f860e586389e40f20efb15e55055f2",
    10513110: "a4b5c30743f64631e98d25aa601824d513a7fb0a40daa309da2c71316103466a",
    10423110: "aefaa6d85b5d9a12bdac67499f970e2a18887d15bcba693b9e642f0ecf17d16d",
    10321120: "3565dbac6bf1ad4a0dfe7956dfc1f7c66293ac347dc8774348f802ca73984b3e",
    10644110: "7e9d6d6ba8a3088abee97c81bd062c50b5d101bd6b1dd90754019f51cb331896",
    10841130: "d03f9cd8d5d3e1774dc2986680685a56486ee97bf6d263d50e8e47434045834d",
    10163130: "ad95d79108fffdbabe22c9f3ca22b325c702eeb70fe1353824cba519e8dcb3f9",
    10462210: "c1d58e4bee5c0edcde4353fd025d411f7440281c3f8a020ed7fd513d2180dc4e",
    10174110: "5f0433361132ae034117778c3d3db94e6195425f4dccd8b42d9f45c2806c8ed0",
}

TEST_EVIDENCE = "tests/test_real_crest_token_activation_exact_batch.py"


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


def _start_operation(engine, operation: EffectOperation, *, controller: int = 0) -> None:
    engine._start_effects(
        _card(988900, name="批次效果来源", card_type="法术", attack=None, life=None),
        None,
        (operation,),
        controller=controller,
        label="批次测试效果",
    )


class CrestTokenActivationBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 8801):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_all_cards_modes_crest_activation_and_intrinsics(self):
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10644110),
            ("疾驰", "必杀", "灵气"),
        )
        self.assertEqual(self.rulebook.activation_for(10462210).cost, 0)
        self.assertEqual(
            [mode.mode_id for mode in self.rulebook.modes_for(10423110)],
            ["enhance_9"],
        )
        self.assertEqual(self.rulebook.emblem_def("lapis_shining_seraph").countdown, 2)
        self.assertEqual(self.rulebook.emblem_def("eudie_maiden_reborn").countdown, 3)
        self.assertIsNone(self.rulebook.emblem_def("aria_lady_of_the_woods").countdown)
        self.assertIsNone(self.rulebook.emblem_def("spirit_of_wadatsumi").countdown)
        self.assertEqual(self.rulebook.listeners_for(10462210)[0].zone.value, "hand")
        ruled = {
            card_id
            for card_id in BATCH_CARD_IDS
            if any(self.rulebook.operations_for(card_id, trigger) for trigger in Trigger)
            or self.rulebook.listeners_for(card_id)
        }
        self.assertEqual(ruled, set(BATCH_CARD_IDS))

    def test_skyfarers_modes_cover_random_damage_empty_board_and_filtered_draw(self):
        damage = self.fresh(seed=3)
        enemies = [
            _put_unit(damage, 1, _card(988001 + index, life=6))
            for index in range(2)
        ]
        source = _play(damage, self.repository, 10403110)
        _choose(damage, "choose_one:skyfarer_damage")
        self.assertEqual(sorted(enemy.health for enemy in enemies), [1, 6])
        self.assertFalse(source.evolved)

        empty = self.fresh(seed=5)
        _play(empty, self.repository, 10403110)
        _choose(empty, "choose_one:skyfarer_damage")
        self.assertIsNone(empty.state.pending_choice)

        draw = self.fresh(seed=7)
        draw.players[0].deck = [
            _card(988010, card_type="法术", attack=None, life=None),
            _card(988011),
            _card(988012),
        ]
        _play(draw, self.repository, 10403110)
        _choose(draw, "choose_one:skyfarer_draw")
        self.assertEqual(
            sorted(card.definition.card_id for card in draw.players[0].hand),
            [988011, 988012],
        )

    def test_skyfarers_random_mode_and_skybound_art_are_deterministic(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=11)
            engine.players[0].turns_started = 10
            for index in range(3):
                _put_unit(engine, 1, _card(988020 + index, life=6))
            source = _play(engine, self.repository, 10403110)
            _choose(engine, "choose_one:skyfarer_damage")
            self.assertTrue(source.evolved)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_intrepid_newshound_super_evolve_capacity_and_last_words_draw(self):
        engine = self.fresh(seed=13)
        engine.players[0].deck = [_card(988030)]
        source = _play(engine, self.repository, 10702110)
        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(
            sum(unit.definition.card_id == 10702110 for unit in engine.players[0].board),
            3,
        )
        _destroy_units(engine, source)
        self.assertEqual([card.definition.card_id for card in engine.players[0].hand], [988030])

        shortage = self.fresh(seed=17)
        source = _play(shortage, self.repository, 10702110)
        for index in range(3):
            _put_unit(shortage, 0, _card(988040 + index))
        _enable_evolution(shortage, super_evolve=True)
        shortage.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(
            sum(unit.definition.card_id == 10702110 for unit in shortage.players[0].board),
            2,
        )

    def test_aria_crest_grants_storm_only_to_entering_allied_pixies(self):
        engine = self.fresh(seed=19)
        source = _play(engine, self.repository, 10114110)
        self.assertFalse(source.has_keyword("疾驰"))
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        fairies = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90011110
        ]
        self.assertEqual(len(fairies), 3)
        self.assertTrue(all(fairy.has_keyword("疾驰") for fairy in fairies))

        _start_operation(
            engine,
            EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90021110),
        )
        knight = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90021110
        )
        self.assertFalse(knight.has_keyword("疾驰"))

    def test_wayfaring_ferryman_repeats_fanfare_and_super_evolve_buffs_pixies(self):
        evolved = self.fresh(seed=23)
        source = _play(evolved, self.repository, 10513110)
        self.assertEqual(
            sum(unit.definition.card_id == 90011110 for unit in evolved.players[0].board),
            3,
        )
        _enable_evolution(evolved)
        evolved.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            sum(unit.definition.card_id == 90011110 for unit in evolved.players[0].board),
            4,
        )

        super_evolved = self.fresh(seed=29)
        source = _play(super_evolved, self.repository, 10513110)
        _enable_evolution(super_evolved, super_evolve=True)
        super_evolved.apply(SuperEvolve(0, source.entity_id))
        fairies = [
            unit for unit in super_evolved.players[0].board
            if unit.definition.card_id == 90011110
        ]
        self.assertEqual(len(fairies), 4)
        self.assertTrue(all(fairy.has_keyword("必杀") for fairy in fairies))
        self.assertFalse(source.has_keyword("必杀"))

    def test_golden_warrior_three_modes_and_enhance_all(self):
        evolved = self.fresh(seed=31)
        source = _play(evolved, self.repository, 10423110)
        _choose(evolved, "choose_one:golden_super_evolve")
        self.assertTrue(source.super_evolved)

        damage = self.fresh(seed=37)
        enemies = [_put_unit(damage, 1, _card(988100 + index, life=5)) for index in range(2)]
        _play(damage, self.repository, 10423110)
        _choose(damage, "choose_one:golden_damage")
        self.assertEqual([enemy.health for enemy in enemies], [1, 1])

        healing = self.fresh(seed=41)
        healing.players[0].health = 10
        _play(healing, self.repository, 10423110)
        _choose(healing, "choose_one:golden_heal")
        self.assertEqual(healing.players[0].health, 14)

        enhanced = self.fresh(seed=43)
        enhanced.players[0].health = 10
        enemy = _put_unit(enhanced, 1, _card(988110, life=5))
        source = _play(enhanced, self.repository, 10423110, mode_id="enhance_9")
        self.assertTrue(source.super_evolved)
        self.assertEqual(enemy.health, 1)
        self.assertEqual(enhanced.players[0].health, 14)
        self.assertIsNone(enhanced.state.pending_choice)

    def test_swordmasters_companion_returns_once_without_last_words(self):
        engine = self.fresh(seed=47)
        source = _play(engine, self.repository, 10321120)
        for index in range(4):
            _put_unit(engine, 0, _card(988120 + index))
        _destroy_units(engine, source)
        returned = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10321120
        )
        self.assertTrue(returned.printed_abilities_removed)
        self.assertEqual(len(engine.players[0].board), 5)
        _destroy_units(engine, returned)
        self.assertFalse(any(
            unit.definition.card_id == 10321120 for unit in engine.players[0].board
        ))

    def test_izanami_discard_contract_intrinsics_capacity_and_no_target_fallback(self):
        engine = self.fresh(seed=53)
        discard = _put_hand(engine, _card(988130, card_type="法术", attack=None, life=None))
        source = _play(engine, self.repository, 10644110)
        _choose_entity(engine, discard.entity_id)
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertTrue(source.has_keyword("必杀"))
        self.assertTrue(source.has_keyword("灵气"))
        self.assertEqual(
            [card.definition.card_id for card in engine.players[0].hand],
            [10642310, 10642310],
        )

        no_target = self.fresh(seed=59)
        hand_card = _put_hand(no_target, self.repository.get(10644110))
        command = PlayCard(0, no_target.players[0].hand.index(hand_card))
        self.assertIn(command, no_target.legal_commands())
        no_target.apply(command)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertEqual(
            [card.definition.card_id for card in no_target.players[0].hand],
            [10642310, 10642310],
        )

        capacity = self.fresh(seed=60)
        source_hand = _put_hand(capacity, self.repository.get(10644110))
        while len(capacity.players[0].hand) < capacity.config.max_hand:
            _put_hand(capacity, _card(988140 + len(capacity.players[0].hand)))
        capacity.apply(PlayCard(0, capacity.players[0].hand.index(source_hand)))
        _choose_entity(capacity, capacity.players[0].hand[0].entity_id)
        self.assertEqual(len(capacity.players[0].hand), capacity.config.max_hand)
        self.assertEqual(
            sum(card.definition.card_id == 10642310 for card in capacity.players[0].hand),
            2,
        )

    def test_wadatsumi_adds_orca_and_crest_buffs_only_entering_marines(self):
        engine = self.fresh(seed=61)
        source = _play(engine, self.repository, 10841130)
        self.assertEqual([card.definition.card_id for card in engine.players[0].hand], [90041130])
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))

        _start_operation(
            engine,
            EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90041130),
        )
        orca = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90041130
        )
        self.assertEqual((orca.attack, orca.health), (3, 3))

        _start_operation(
            engine,
            EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90021110),
        )
        knight = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90021110
        )
        self.assertEqual((knight.attack, knight.health), (1, 1))

    def test_lapis_last_words_crest_expires_into_storm_copy_and_respects_capacity(self):
        engine = self.fresh(seed=67)
        source = _play(engine, self.repository, 10163130)
        self.assertFalse(source.has_keyword("疾驰"))
        _destroy_units(engine, source)
        emblem = engine.players[0].emblems[0]
        self.assertEqual((emblem.definition.emblem_id, emblem.countdown), ("lapis_shining_seraph", 2))
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        returned = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10163130
        )
        self.assertTrue(returned.has_keyword("疾驰"))
        self.assertEqual(engine.players[0].emblems, [])

        full = self.fresh(seed=71)
        source = _play(full, self.repository, 10163130)
        _destroy_units(full, source)
        for index in range(5):
            _put_unit(full, 0, _card(988150 + index))
        full.apply(EndTurn(0))
        full.apply(EndTurn(1))
        full.apply(EndTurn(0))
        full.apply(EndTurn(1))
        self.assertFalse(any(
            unit.definition.card_id == 10163130 for unit in full.players[0].board
        ))

    def test_skybound_ark_hand_listener_activation_target_and_stale_target(self):
        engine = self.fresh(seed=73)
        hand_copy = _put_hand(engine, self.repository.get(10462210))
        amulet = _play(engine, self.repository, 10462210)
        target = _put_unit(engine, 0, _card(988160))
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        _choose_entity(engine, target.entity_id)
        self.assertNotIn(amulet, engine.players[0].board)
        self.assertTrue(target.evolved)
        self.assertEqual(hand_copy.current_cost, 3)

        stale = self.fresh(seed=79)
        amulet = _play(stale, self.repository, 10462210)
        target = _put_unit(stale, 0, _card(988161))
        stale.apply(ActivateAmulet(0, amulet.entity_id))
        request = stale.state.pending_choice
        stale.players[0].board.remove(target)
        stale._send_to_graveyard(
            0,
            target.definition,
            "batch_test_target_left_play",
            source_entity_id=target.entity_id,
        )
        stale.apply(Choose(0, request.options[0].option_id))
        self.assertNotIn(amulet, stale.players[0].board)
        self.assertIsNone(stale.state.pending_choice)

    def test_skybound_ark_no_unevolved_target_is_illegal_and_does_not_mutate(self):
        engine = self.fresh(seed=83)
        amulet = _play(engine, self.repository, 10462210)
        evolved = _put_unit(engine, 0, _card(988170))
        evolved.evolved = True
        command = ActivateAmulet(0, amulet.entity_id)
        before = engine.deterministic_fingerprint()
        self.assertNotIn(command, engine.legal_commands())
        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_eudie_draws_or_heals_at_owner_turn_end_and_countdown_decays(self):
        draw = self.fresh(seed=89)
        draw.players[0].deck = [_card(988180), _card(988181)]
        source = _play(draw, self.repository, 10174110)
        self.assertEqual(len(draw.players[0].hand), 1)
        _enable_evolution(draw)
        draw.apply(Evolve(0, source.entity_id))
        draw.apply(EndTurn(0))
        self.assertEqual(len(draw.players[0].hand), 2)
        self.assertEqual(draw.players[0].emblems[0].countdown, 3)
        draw.apply(EndTurn(1))
        self.assertEqual(draw.players[0].emblems[0].countdown, 2)

        heal = self.fresh(seed=97)
        heal.players[0].deck = [_card(988190)]
        source = _play(heal, self.repository, 10174110)
        while len(heal.players[0].hand) < 6:
            _put_hand(heal, _card(988191 + len(heal.players[0].hand)))
        heal.players[0].health = 10
        _enable_evolution(heal)
        heal.apply(Evolve(0, source.entity_id))
        hand_count = len(heal.players[0].hand)
        heal.apply(EndTurn(0))
        self.assertEqual(heal.players[0].health, 11)
        self.assertEqual(len(heal.players[0].hand), hand_count)

    def test_rl_mask_exposes_enhance_modes_activation_and_target_choice(self):
        deck = [_card(988300 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=101,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=101)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        golden = _put_hand(env.core, self.repository.get(10423110))
        normal = PlayCard(0, env.players[0].hand.index(golden))
        enhanced = PlayCard(0, env.players[0].hand.index(golden), "enhance_9")
        env.invalidate_cache(reason="golden modes")
        mask = env.action_mask()
        self.assertTrue(mask[env._encode_command(normal)])
        self.assertTrue(mask[env._encode_command(enhanced)])
        env.core.apply(normal)
        env.invalidate_cache(reason="golden choose one")
        mask = env.action_mask()
        for option in env.core.state.pending_choice.options:
            self.assertTrue(mask[env._encode_command(Choose(0, option.option_id))])

        env.reset(seed=103)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        amulet = _play(env.core, self.repository, 10462210)
        target = _put_unit(env.core, 0, _card(988400))
        activate = ActivateAmulet(0, amulet.entity_id)
        env.invalidate_cache(reason="ark activation")
        self.assertTrue(env.action_mask()[env._encode_command(activate)])
        env.core.apply(activate)
        env.invalidate_cache(reason="ark target choice")
        choose = next(
            option for option in env.core.state.pending_choice.options
            if option.entity_id == target.entity_id
        )
        self.assertTrue(
            env.action_mask()[env._encode_command(Choose(0, choose.option_id))]
        )


class CrestTokenActivationDatabaseAuditTests(unittest.TestCase):
    def test_generated_fairy_and_orca_have_the_new_executable_producer_paths(self):
        report = _build_token_audit("data/cards.sqlite3", "data/rules")
        tokens = {card["card_id"]: card for card in report["cards"]}
        expected = {
            90011110: {
                (10114110, "summon"),
                (10513110, "summon"),
            },
            90041130: {
                (10841130, "add_card"),
            },
        }
        for token_id, producer_pairs in expected.items():
            with self.subTest(token_id=token_id):
                token = tokens[token_id]
                self.assertEqual(token["category"], "entry_behavior_complete")
                actual = {
                    (producer["source_card_id"], producer["entry_kind"])
                    for producer in token["authored_producers"]
                    if producer["rule_file"]
                    == "real_crest_token_activation_exact_batch.json"
                }
                self.assertEqual(actual, producer_pairs)

    def test_database_text_modes_and_references_match_reviewed_cards(self):
        expected_phrases = {
            10403110: ("Select a Mode", "Skybound Art", "random", "Draw 2 followers"),
            10702110: ("Last Words", "Draw a card", "Super-Evolve", "2 copies"),
            10114110: ("Crest: Aria", "3 copies", "Fairy", "Storm"),
            10513110: ("3 copies", "Fairy", "Evolve", "Pixie", "Bane"),
            10423110: ("Select a Mode", "Super-evolve", "all enemy followers", "Enhance"),
            10321120: ("Last Words", "Comrade of the Swordmaster", "remove Last Words"),
            10644110: ("discard", "2 copies", "Spilling Red", "Storm", "Bane", "Aura"),
            10841130: ("Majestic Megalorca", "Crest: Spirit", "Marine", "+1/+1"),
            10163130: ("Last Words", "Crest: Lapis", "Countdown", "Storm"),
            10462210: ("Activates in hand", "Engage", "reduce the cost", "unevolved"),
            10174110: ("Draw a card", "Crest: Eudie", "5 cards or less", "at least 6"),
        }
        expected_references = {
            10403110: [],
            10702110: [10702110],
            10114110: [90011110],
            10513110: [90011110],
            10423110: [],
            10321120: [10321120],
            10644110: [10642310],
            10841130: [90041130],
            10163130: [10163130],
            10462210: [],
            10174110: [],
        }
        expected_alt_counts = {
            10114110: 2,
            10841130: 1,
            10163130: 2,
            10174110: 2,
        }
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
                        expected_alt_counts.get(card_id, 0),
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
