# -*- coding: utf-8 -*-
"""Exact Portalcraft Artifact/Puppet listener and producer follow-up."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import BeginFusion, Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.origin import CardOrigin
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _choose,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (
    10171140,
    10173210,
    10271110,
    10371120,
    10372210,
    10374110,
    10471130,
    10571110,
    10671120,
    10672310,
)
TOKEN_IDS = (90071110, 90071120, 90071210, 90071220, 90072110, 90073110, 90073130)
SOURCE_HASHES = {
    10171140: "5b61a6b0948f76c1769536d04e186da8dd99b2627df840af346a3ade5035a557",
    10173210: "c548c014d4817cc1230a6d75ed6871e2f1936928f9a2cb37f7b04d1eb6c101f0",
    10271110: "426d0cc7ef29736e6ab98489b725bff8a335d96cbb6162caf1a618a4080a043c",
    10371120: "129a3e926da31a77aa22873cf6b6eb38329ba3340c2b8a512bea40ad0d3ffe9f",
    10372210: "328309fa3814fbc5e10e991bbb8536eaa8a45a921520a5652602f20a483479b4",
    10374110: "222c350f27203309020880ce837dcbdaa00aa2e75bc9c1b0c21be19ad2ed667a",
    10471130: "e77af91ee743b9fcc291ec9dc1cced9387e2f203009f0a8c7f7ec8810980eece",
    10571110: "104bb4d43461fb06a7b698a2c9f6612a431badec9f780d4a2ced0518ac265552",
    10671120: "4c50d46978594d5b9a1d373da7d7df9a19b29d8341f7f7f39118426fb3b06733",
    10672310: "a010ce1b32873bf58fc5ff798abd964395ae48ccdc3e76146f6f737200110455",
}


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _enable_super_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.first_player_super_evolution_unlock_turn
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.super_evolved_this_turn = False


def _fuse(engine, destination, *materials) -> None:
    engine.apply(BeginFusion(0, destination.entity_id))
    for material in materials:
        _choose(engine, material.entity_id)
    request = engine.state.pending_choice
    engine.apply(Choose(request.player_index, "fusion:confirm"))


class RealPortalArtifactListenerFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1901):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_automata_assassin_grants_bane_only_to_first_simultaneous_puppet_each_turn(self):
        engine = self.fresh(seed=3)
        assassin = _play(engine, self.repository, 10171140)
        self.assertFalse(assassin.has_keyword("必杀"))
        self.assertEqual([card.card_id for card in engine.players[0].hand], [90071120])

        _play(engine, self.repository, 10171310)
        puppets = [unit for unit in engine.players[0].board if unit.definition.card_id == 90071120]
        self.assertEqual(len(puppets), 2)
        self.assertTrue(puppets[0].has_keyword("必杀"))
        self.assertFalse(puppets[1].has_keyword("必杀"))

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        next_puppet = _play(engine, self.repository, 90071120)
        self.assertTrue(next_puppet.has_keyword("必杀"))

    def test_heritage_barrage_triggers_once_for_one_multi_material_fusion(self):
        engine = self.fresh(seed=5)
        heritage = _play(engine, self.repository, 10173210)
        self.assertIsNone(heritage.countdown)
        self.assertEqual([card.card_id for card in engine.players[0].hand], [90071210])
        enemy = _put_unit(engine, 1, _card(9101, life=5))
        destination = _put_hand(engine, self.repository.get(10213310))
        material_a = _put_hand(engine, _card(9102, class_id=1))
        material_b = _put_hand(engine, _card(9103, class_id=1))
        _fuse(engine, destination, material_a, material_b)
        self.assertEqual(enemy.health, 3)

    def test_engine_swordsman_repeats_summon_and_core_generation_on_evolve(self):
        engine = self.fresh(seed=7)
        _enable_evolution(engine)
        source = _play(engine, self.repository, 10271110)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [unit.definition.card_id for unit in engine.players[0].board if unit is not source],
            [90072110, 90072110],
        )
        self.assertEqual([card.card_id for card in engine.players[0].hand], [90071220, 90071220])

    def test_sonic_trooper_evolve_targets_only_an_artifact(self):
        engine = self.fresh(seed=11)
        _enable_evolution(engine)
        ordinary = _put_unit(engine, 0, _card(9201))
        source = _play(engine, self.repository, 10371120)
        gamma = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90073130)
        before = (gamma.attack, gamma.health, gamma.max_health)
        engine.apply(Evolve(0, source.entity_id))
        options = engine.state.pending_choice.options
        self.assertEqual([option.entity_id for option in options], [gamma.entity_id])
        _choose(engine, gamma.entity_id)
        self.assertEqual(
            (gamma.attack, gamma.health, gamma.max_health),
            (before[0] + 3, before[1] + 3, before[2] + 3),
        )
        self.assertEqual((ordinary.attack, ordinary.health), (1, 5))

    def test_destruction_wasteland_draws_without_target_and_resolves_selected_last_words(self):
        no_target = self.fresh(seed=13)
        deck_before = len(no_target.players[0].deck)
        first = _play(no_target, self.repository, 10372210)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertEqual(len(no_target.players[0].deck), deck_before - 2)

        deck_before_second = len(no_target.players[0].deck)
        _play(no_target, self.repository, 10372210)
        _choose(no_target, first.entity_id)
        self.assertNotIn(first, no_target.players[0].board)
        self.assertEqual(len(no_target.players[0].deck), deck_before_second - 3)

    def test_axia_super_evolve_counts_then_destroys_other_cards_and_has_immunity(self):
        engine = self.fresh(seed=17)
        _enable_super_evolution(engine)
        other_unit = _put_unit(engine, 0, _card(9301))
        other_amulet = _play(engine, self.repository, 10173210)
        source = _play(engine, self.repository, 10374110)
        self.assertTrue(source.has_keyword("守护"))
        self.assertTrue(self.rulebook.cannot_be_destroyed_by_effects(10374110))
        enemy_health = engine.players[1].health
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(engine.players[1].health, enemy_health - 2)
        self.assertIn(source, engine.players[0].board)
        self.assertNotIn(other_unit, engine.players[0].board)
        self.assertNotIn(other_amulet, engine.players[0].board)

        immunity = self.fresh(seed=18)
        sacrifice = _put_unit(immunity, 0, _card(9302))
        protected = _put_unit(immunity, 1, self.repository.get(10374110))
        _play(immunity, self.repository, 10151310)
        _choose(immunity, sacrifice.entity_id)
        _choose(immunity, protected.entity_id)
        self.assertIn(protected, immunity.players[1].board)

    def test_isaac_last_words_adds_attack_artifact(self):
        engine = self.fresh(seed=19)
        isaac = _play(engine, self.repository, 10471130)
        enemy = _put_unit(engine, 1, _card(9401))
        _play(engine, self.repository, 10151310)
        _choose(engine, isaac.entity_id)
        _choose(engine, enemy.entity_id)
        generated = next(card for card in engine.players[0].hand if card.card_id == 90072110)
        self.assertIs(generated.origin, CardOrigin.TOKEN)

    def test_stage_maker_enhance_grants_storm_only_to_successful_outputs(self):
        normal = self.fresh(seed=23)
        source = _play(normal, self.repository, 10571110)
        normal_outputs = [unit for unit in normal.players[0].board if unit is not source]
        self.assertEqual([unit.definition.card_id for unit in normal_outputs], [90071120, 90071110])
        self.assertTrue(all(not unit.has_keyword("疾驰") for unit in normal_outputs))
        self.assertFalse(source.has_keyword("疾驰"))

        enhanced = self.fresh(seed=29)
        source = _play(enhanced, self.repository, 10571110, mode_id="enhance_7")
        enhanced_outputs = [unit for unit in enhanced.players[0].board if unit is not source]
        self.assertEqual([unit.definition.card_id for unit in enhanced_outputs], [90071120, 90071110])
        self.assertTrue(all(unit.has_keyword("疾驰") for unit in enhanced_outputs))

        shortage = self.fresh(seed=31)
        for index in range(3):
            _put_unit(shortage, 0, _card(9500 + index))
        source = _play(shortage, self.repository, 10571110, mode_id="enhance_7")
        outputs = [unit for unit in shortage.players[0].board if unit.definition.card_id in TOKEN_IDS]
        self.assertEqual([unit.definition.card_id for unit in outputs], [90071120])
        self.assertTrue(outputs[0].has_keyword("疾驰"))
        self.assertFalse(source.has_keyword("疾驰"))

    def test_smart_creator_grants_bane_and_ward_only_to_summoned_alpha(self):
        engine = self.fresh(seed=37)
        source = _play(engine, self.repository, 10671120)
        alpha = next(unit for unit in engine.players[0].board if unit is not source)
        self.assertEqual(alpha.definition.card_id, 90073110)
        self.assertTrue(alpha.has_keyword("必杀"))
        self.assertTrue(alpha.has_keyword("守护"))
        self.assertFalse(source.has_keyword("必杀"))
        self.assertFalse(source.has_keyword("守护"))

    def test_mediocre_cartography_buffs_each_successful_summon_and_handles_shortage(self):
        engine = self.fresh(seed=41)
        _play(engine, self.repository, 10672310)
        expected_ids = [10673110, 10671110, 10672110]
        self.assertEqual([unit.definition.card_id for unit in engine.players[0].board], expected_ids)
        for unit in engine.players[0].board:
            self.assertEqual(unit.max_health, unit.definition.life + 1)
            self.assertEqual(unit.health, unit.max_health)

        shortage = self.fresh(seed=43)
        for index in range(4):
            _put_unit(shortage, 0, _card(9600 + index))
        _play(shortage, self.repository, 10672310)
        output = next(unit for unit in shortage.players[0].board if unit.definition.card_id == 10673110)
        self.assertEqual(output.max_health, output.definition.life + 1)
        self.assertFalse(any(unit.definition.card_id in {10671110, 10672110} for unit in shortage.players[0].board))

    def test_all_cards_are_exact_and_tokens_list_new_real_producers(self):
        coverage = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in COLLECTIBLE_IDS:
            with self.subTest(card_id=card_id):
                info = coverage["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(info["clause_audit"]["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_portal_artifact_listener_followup_batch.py"],
                )

        token_report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        tokens = {card["card_id"]: card for card in token_report["cards"]}
        expected_sources = {
            90071110: {10571110},
            90071120: {10171140, 10571110},
            90071210: {10173210},
            90071220: {10271110},
            90072110: {10271110, 10471130},
            90073110: {10671120},
            90073130: {10371120},
        }
        for token_id, expected in expected_sources.items():
            with self.subTest(token_id=token_id):
                actual = {
                    producer["source_card_id"]
                    for producer in tokens[token_id]["authored_producers"]
                    if producer["rule_file"] == "real_portal_artifact_listener_followup_batch.json"
                }
                self.assertEqual(actual, expected)

    def test_listener_and_output_sequences_are_seed_reproducible(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=47)
            _play(engine, self.repository, 10171140)
            _play(engine, self.repository, 10171310)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])


if __name__ == "__main__":
    unittest.main()
