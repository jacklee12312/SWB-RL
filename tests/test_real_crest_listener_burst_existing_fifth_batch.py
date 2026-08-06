# -*- coding: utf-8 -*-
"""Direct contracts for the fifth exact existing-primitive card slice."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import Amulet
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _destroy_units,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10124130,
    10133110,
    10243110,
    10414120,
    10714120,
    10724110,
    10833110,
    10864120,
)
SOURCE_HASHES = {
    10124130: "dfedfd62cf78c9a12a17ce576abd40745e130bac023c40f3e450bbfd270caf32",
    10133110: "ca7112e1bb353ddb7e55b7a8a7208c729b8949671fde9993842fb374245771f6",
    10243110: "ad2345bdd0e8b3b4fb370106b5776f263d716970c818353a2f460c597aaff878",
    10414120: "08218ae56ceed8594e741f389d1de9cfac796ebe31f5e832b98c77570046db16",
    10714120: "58c238be4ca264678829aa9524871f428888d82a087b8f3cc92fe691d31a0b1e",
    10724110: "36094cf64bfd07909fa6202243ccc866939381fd83e638b5b62894d662d32a4d",
    10833110: "0c2bb49c4ccafb1b510c7ebea12461333c8602a0b2018fa4865214179c171bdd",
    10864120: "3e57c69e40414444332057f758222c9017f85341711fdf3ce2d25398a912d9fb",
}
TEST_EVIDENCE = "tests/test_real_crest_listener_burst_existing_fifth_batch.py"


def _choose(engine, option_id: str) -> None:
    engine.apply(Choose(engine.state.pending_choice.player_index, option_id))


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


def _advance_round(engine) -> None:
    engine.apply(EndTurn(engine.current_player))
    engine.apply(EndTurn(engine.current_player))


def _put_sigil(engine, repository: CardRepository, count: int) -> Amulet:
    sigil = Amulet(
        definition=repository.get(90031210),
        entity_id=engine.state.allocate_entity_id(),
        earth_sigil_count=count,
        entered_turn=engine.turn,
        origin=CardOrigin.TOKEN,
    )
    engine.players[0].board.append(sigil)
    return sigil


def _effect_summon(engine, card_id: int, *, controller: int = 0) -> None:
    engine._start_effects(
        _card(997900, name="批次效果来源", card_type="法术", attack=None, life=None),
        None,
        (EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=card_id),),
        controller=controller,
        label="批次测试召唤",
    )


class CrestListenerBurstExistingFifthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 7401):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_all_cards_crests_bursts_intrinsics_and_listeners(self):
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10124130),
            frozenset({"疾驰"}),
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10724110),
            frozenset({"突进"}),
        )
        self.assertEqual(self.rulebook.intrinsic_keywords_for(10243110), ("守护",))
        self.assertEqual(
            self.rulebook.emblem_def("kagemitsu_enduring_warrior").countdown,
            2,
        )
        self.assertEqual(
            self.rulebook.emblem_def("yuel_societte_dancing_duo").countdown,
            4,
        )
        self.assertIsNone(
            self.rulebook.emblem_def("tico_mysterian_spellcrafter").countdown
        )
        yuel = self.rulebook.listeners_for(10414120)[0]
        self.assertEqual(yuel.zone.value, "leader_area")
        self.assertIs(yuel.event, EventType.CARD_PLAYED)
        self.assertTrue(yuel.once_per_turn)
        gildaria = self.rulebook.listeners_for(10724110)[0]
        self.assertEqual(gildaria.source_relation.value, "other")
        self.assertEqual(gildaria.turn_scope.value, "owner_turn")
        ruled = {
            card_id
            for card_id in CARD_IDS
            if any(self.rulebook.operations_for(card_id, trigger) for trigger in Trigger)
            or self.rulebook.listeners_for(card_id)
            or self.rulebook.union_bursts_for(card_id)
        }
        self.assertEqual(ruled, set(CARD_IDS))

    def test_kagemitsu_last_words_crest_expiry_capacity_and_super_evolve_storm(self):
        engine = self.fresh(seed=3)
        source = _play(engine, self.repository, 10124130)
        _destroy_units(engine, source)
        self.assertEqual(
            [emblem.emblem_id for emblem in engine.players[0].emblems],
            ["kagemitsu_enduring_warrior"],
        )
        _advance_round(engine)
        self.assertFalse(any(
            unit.definition.card_id == 10124130 for unit in engine.players[0].board
        ))
        _advance_round(engine)
        returned = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10124130
        ]
        self.assertEqual(len(returned), 1)

        capped = self.fresh(seed=5)
        source = _play(capped, self.repository, 10124130)
        _destroy_units(capped, source)
        for index in range(capped.config.max_board):
            _put_unit(capped, 0, _card(997000 + index))
        _advance_round(capped)
        _advance_round(capped)
        self.assertFalse(any(
            unit.definition.card_id == 10124130 for unit in capped.players[0].board
        ))

        storm = self.fresh(seed=7)
        source = _play(storm, self.repository, 10124130)
        self.assertFalse(source.has_keyword("疾驰"))
        _enable_evolution(storm, super_evolve=True)
        storm.apply(SuperEvolve(0, source.entity_id))
        self.assertTrue(source.has_keyword("疾驰"))
        storm.assert_invariants()

    def test_juno_earth_sigil_damage_crest_rite_and_no_target_skip(self):
        engine = self.fresh(seed=11)
        _put_sigil(engine, self.repository, 3)
        enemy = _put_unit(engine, 1, _card(997020, life=10))
        source = _play(engine, self.repository, 10133110)
        _choose_entity(engine, enemy.entity_id)
        self.assertEqual(enemy.health, 7)
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(engine.players[0].emblems[0].countdown, 3)
        engine.apply(EndTurn(0))
        self.assertEqual(engine.players[0].earth_sigils, 2)
        self.assertTrue(any(
            unit.definition.card_id == 90031120 for unit in engine.players[0].board
        ))

        no_rite = self.fresh(seed=13)
        enemy = _put_unit(no_rite, 1, _card(997021, life=10))
        source = _play(no_rite, self.repository, 10133110)
        _choose_entity(no_rite, enemy.entity_id)
        _enable_evolution(no_rite)
        no_rite.apply(Evolve(0, source.entity_id))
        no_rite.apply(EndTurn(0))
        self.assertFalse(any(
            unit.definition.card_id == 90031120 for unit in no_rite.players[0].board
        ))

        no_target = self.fresh(seed=17)
        _put_hand(no_target, self.repository.get(10133110))
        no_target.apply(PlayCard(0, 0))
        self.assertIsNone(no_target.state.pending_choice)
        self.assertTrue(any(
            unit.definition.card_id == 10133110
            for unit in no_target.players[0].board
        ))
        no_target.assert_invariants()

    def test_neptune_crest_heals_for_successful_marine_summons_and_board_shortage(self):
        engine = self.fresh(seed=19)
        engine.players[0].health = 10
        source = _play(engine, self.repository, 10243110)
        orcas = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90041130
        ]
        self.assertTrue(source.has_keyword("守护"))
        self.assertEqual(len(orcas), 2)
        self.assertEqual(engine.players[0].health, 12)
        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(
            sum(unit.definition.card_id == 90041130 for unit in engine.players[0].board),
            4,
        )
        self.assertEqual(engine.players[0].health, 14)

        shortage = self.fresh(seed=23)
        shortage.players[0].health = 10
        for index in range(3):
            _put_unit(shortage, 0, _card(997030 + index))
        _play(shortage, self.repository, 10243110)
        self.assertEqual(
            sum(unit.definition.card_id == 90041130 for unit in shortage.players[0].board),
            1,
        )
        self.assertEqual(shortage.players[0].health, 11)

    def test_yuel_random_damage_and_crest_evolves_only_first_played_follower(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=29)
            enemies = [
                _put_unit(engine, 1, _card(997040 + index, life=10))
                for index in range(2)
            ]
            _play(engine, self.repository, 10414120)
            self.assertEqual(sum(unit.health for unit in enemies), 12)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

        empty = self.fresh(seed=31)
        _play(empty, self.repository, 10414120)
        self.assertIsNone(empty.state.pending_choice)

        crest = self.fresh(seed=37)
        source = _play(crest, self.repository, 10414120)
        _enable_evolution(crest, super_evolve=True)
        crest.apply(SuperEvolve(0, source.entity_id))
        first = _put_hand(crest, _card(997050, cost=1))
        crest.apply(PlayCard(0, crest.players[0].hand.index(first)))
        second = _put_hand(crest, _card(997051, cost=1))
        crest.apply(PlayCard(0, crest.players[0].hand.index(second)))
        first_unit = next(unit for unit in crest.players[0].board if unit.definition.card_id == 997050)
        second_unit = next(unit for unit in crest.players[0].board if unit.definition.card_id == 997051)
        self.assertTrue(first_unit.evolved)
        self.assertFalse(second_unit.evolved)

    def test_glacial_hart_hand_capacity_distributed_damage_combo_crest_and_seed(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=53)
            enemies = [
                _put_unit(engine, 1, _card(997080 + index, life=10))
                for index in range(3)
            ]
            _play(engine, self.repository, 10714120)
            engine.apply(EndTurn(0))
            self.assertEqual(sum(unit.health for unit in enemies), 25)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

        capped = self.fresh(seed=59)
        for index in range(capped.config.max_hand - 1):
            _put_hand(capped, _card(997100 + index))
        _play(capped, self.repository, 10714120)
        self.assertEqual(
            sum(card.card_id == 90011310 for card in capped.players[0].hand),
            1,
        )

        combo = self.fresh(seed=61)
        source = _play(combo, self.repository, 10714120)
        _enable_evolution(combo, super_evolve=True)
        combo.apply(SuperEvolve(0, source.entity_id))
        combo.players[0].cards_played_this_turn = 3
        before = sum(card.card_id == 90011310 for card in combo.players[0].hand)
        combo.apply(EndTurn(0))
        after = sum(card.card_id == 90011310 for card in combo.players[0].hand)
        self.assertEqual(after, before + 1)

    def test_gildaria_rally_tokens_listener_crest_shortage_and_source_departure(self):
        engine = self.fresh(seed=67)
        engine.players[0].cooperation = 20
        source = _play(engine, self.repository, 10724110)
        knights = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90021120
        ]
        self.assertTrue(source.evolved)
        self.assertEqual(len(knights), 2)
        self.assertTrue(all(knight.has_keyword("突进") for knight in knights))
        self.assertEqual(engine.players[1].health, 18)

        _destroy_units(engine, source)
        before = engine.players[1].health
        _effect_summon(engine, 90021110)
        summoned = next(
            unit for unit in reversed(engine.players[0].board)
            if unit.definition.card_id == 90021110
        )
        self.assertFalse(summoned.has_keyword("突进"))
        self.assertEqual(engine.players[1].health, before - 1)

        no_rally = self.fresh(seed=71)
        source = _play(no_rally, self.repository, 10724110)
        self.assertFalse(source.evolved)
        self.assertFalse(no_rally.players[0].emblems)
        self.assertFalse(any(
            unit.definition.card_id == 90021120 for unit in no_rally.players[0].board
        ))

        shortage = self.fresh(seed=73)
        shortage.players[0].cooperation = 20
        for index in range(3):
            _put_unit(shortage, 0, _card(997120 + index))
        _play(shortage, self.repository, 10724110)
        self.assertEqual(
            sum(unit.definition.card_id == 90021120 for unit in shortage.players[0].board),
            1,
        )
        self.assertEqual(shortage.players[1].health, 19)

    def test_tico_tokens_evolve_filter_super_crest_and_spell_event_filter(self):
        engine = self.fresh(seed=79)
        source = _play(engine, self.repository, 10833110)
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [90031310, 90031310],
        )
        mysterian_spell = _put_hand(
            engine,
            _card(997130, cost=5, card_type="法术", attack=None, life=None, tribe_name="玛纳利亚"),
        )
        mysterian_follower = _put_hand(
            engine,
            _card(997131, cost=5, tribe_name="玛纳利亚"),
        )
        other_spell = _put_hand(
            engine,
            _card(997132, cost=5, card_type="法术", attack=None, life=None),
        )
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(mysterian_spell.current_cost, 4)
        self.assertEqual(mysterian_follower.current_cost, 5)
        self.assertEqual(other_spell.current_cost, 5)
        self.assertTrue(all(
            card.current_cost == 0
            for card in engine.players[0].hand
            if card.card_id == 90031310
        ))

        crest = self.fresh(seed=83)
        source = _play(crest, self.repository, 10833110)
        _enable_evolution(crest, super_evolve=True)
        crest.apply(SuperEvolve(0, source.entity_id))
        mysterian = next(
            card for card in crest.players[0].hand if card.card_id == 90031310
        )
        before = crest.players[1].health
        crest.apply(PlayCard(0, crest.players[0].hand.index(mysterian)))
        self.assertEqual(crest.players[1].health, before - 1)
        other = _put_hand(crest, self.repository.get(90011310))
        crest.apply(PlayCard(0, crest.players[0].hand.index(other)))
        self.assertEqual(crest.players[1].health, before - 1)

    def test_zoe_modes_invalid_choice_self_damage_crest_expiry_and_capacity(self):
        followers = self.fresh(seed=89)
        enemies = [_put_unit(followers, 1, _card(997150 + index, life=5)) for index in range(2)]
        source = _play(followers, self.repository, 10864120)
        before = followers.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            followers.apply(Choose(0, "choose_one:missing"))
        self.assertEqual(before, followers.deterministic_fingerprint())
        _choose(followers, "choose_one:damage_followers")
        self.assertEqual([enemy.health for enemy in enemies], [2, 2])
        self.assertEqual(source.health, 1)

        leader = self.fresh(seed=97)
        source = _play(leader, self.repository, 10864120)
        _choose(leader, "choose_one:damage_leader")
        self.assertEqual(leader.players[1].health, 17)
        self.assertEqual(source.health, 1)

        heal = self.fresh(seed=101)
        heal.players[0].health = 10
        source = _play(heal, self.repository, 10864120)
        _choose(heal, "choose_one:heal_leader")
        self.assertEqual(heal.players[0].health, 13)
        _enable_evolution(heal)
        heal.apply(Evolve(0, source.entity_id))
        _advance_round(heal)
        copies = [
            unit for unit in heal.players[0].board
            if unit.definition.card_id == 10864120 and unit.entity_id != source.entity_id
        ]
        self.assertEqual(len(copies), 1)
        self.assertTrue(copies[0].evolved)

        capped = self.fresh(seed=103)
        source = _play(capped, self.repository, 10864120)
        _choose(capped, "choose_one:damage_leader")
        for index in range(4):
            _put_unit(capped, 0, _card(997170 + index))
        _enable_evolution(capped)
        capped.apply(Evolve(0, source.entity_id))
        _advance_round(capped)
        self.assertEqual(
            sum(unit.definition.card_id == 10864120 for unit in capped.players[0].board),
            1,
        )
        capped.assert_invariants()

    def test_action_masks_match_zoe_choices_and_juno_target_legality(self):
        deck = [_card(997200 + index) for index in range(40)]
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
        _put_hand(env.core, self.repository.get(10864120))
        play = PlayCard(0, 0)
        self.assertTrue(env.action_mask()[env._encode_command(play)])
        env.core.apply(play)
        env.invalidate_cache(reason="zoe pending mode")
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        self.assertEqual(
            {command.option_id for command in decoded},
            {
                "choose_one:damage_followers",
                "choose_one:damage_leader",
                "choose_one:heal_leader",
            },
        )

        blocked = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=109,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        blocked.reset(seed=109)
        for player in blocked.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        blocked.players[0].mana = blocked.players[0].max_mana = 10
        _put_hand(blocked.core, self.repository.get(10133110))
        command = PlayCard(0, 0)
        self.assertIn(command, blocked.core.legal_commands())
        self.assertTrue(blocked.action_mask()[blocked._encode_command(command)])


class CrestListenerBurstExistingFifthAuditTests(unittest.TestCase):
    def test_database_multilingual_text_modes_and_references_are_reviewed(self):
        expected_phrases = {
            10124130: ("Last Words", "Super-Evolve", "Storm"),
            10133110: ("earth sigils", "Evolve", "Crest"),
            10243110: ("Majestic Megalorca", "Ward", "Super-Evolve"),
            10414120: ("2 times", "random enemy follower", "Super-Evolve"),
            10714120: ("Deepwood Bounty", "damage split", "Super-Evolve"),
            10724110: ("Rally", "Steelclad Knight", "Rush"),
            10833110: ("Mysterian Missile", "Mysteria spells", "Super-Evolve"),
            10864120: ("Select a", "Deal 3 damage", "Evolve"),
        }
        expected_references = {
            10124130: (10124130,),
            10133110: (90031120,),
            10243110: (90041130,),
            10414120: (),
            10714120: (90011310,),
            10724110: (90021120,),
            10833110: (90031310,),
            10864120: (10864120,),
        }
        expected_mode_phrases = {
            10124130: "Kagemitsu",
            10133110: "Earth Rite",
            10243110: "Marine follower",
            10414120: "Once on each",
            10714120: "Combo",
            10724110: "During your turn",
            10833110: "Mysteria spell",
            10864120: "Summon a",
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id, phrases in expected_phrases.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        "SELECT text_chs, text_cht, text_eng, text_jpn, text_kor "
                        "FROM skill_texts WHERE card_id=? ORDER BY position",
                        (card_id,),
                    ).fetchone()
                    self.assertIsNotNone(row)
                    self.assertTrue(all(row))
                    english = re.sub(r"<[^>]+>", "", row[2])
                    for phrase in phrases:
                        self.assertIn(phrase, english)
                    references = tuple(
                        entry[0]
                        for entry in connection.execute(
                            "SELECT referenced_card_id FROM card_references "
                            "WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    )
                    self.assertEqual(references, expected_references[card_id])
                    modes = connection.execute(
                        "SELECT text_chs, text_cht, text_eng, text_jpn, text_kor "
                        "FROM alt_modes WHERE card_id=? ORDER BY position",
                        (card_id,),
                    ).fetchall()
                    if card_id in expected_mode_phrases:
                        self.assertTrue(modes)
                        self.assertTrue(all(all(mode) for mode in modes))
                        self.assertIn(
                            expected_mode_phrases[card_id],
                            " ".join(re.sub(r"<[^>]+>", "", mode[2]) for mode in modes),
                        )
                    else:
                        self.assertFalse(modes)

    def test_all_eight_cards_have_exact_clause_evidence_and_tokens_stay_complete(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        coverage_counts = report["summary"]["coverage_counts"]
        clause_counts = report["summary"]["clause_audit_counts"]
        self.assertEqual(coverage_counts["covered_exact"], clause_counts["mapped_exact"])
        self.assertEqual(
            coverage_counts["supported_missing_rule"],
            clause_counts["missing_rule"],
        )
        self.assertFalse(report["rule_consistency_issues"])
        self.assertFalse(report["clause_audit_issues"])
        for card_id in CARD_IDS:
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

        token_report = _build_token_audit("data/cards.sqlite3", "data/rules")
        token_total = token_report["summary"]["total"]
        token_categories = token_report["summary"]["categories"]
        self.assertEqual(token_categories["entry_behavior_complete"], token_total)
        self.assertEqual(sum(token_categories.values()), token_total)
        self.assertTrue(all(
            card["category"] == "entry_behavior_complete"
            for card in token_report["cards"]
        ))


if __name__ == "__main__":
    unittest.main()
