# -*- coding: utf-8 -*-
"""Exact generated follower/spell chains plus printed destroy immunity."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import CardPassive, CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _choose,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (
    10223120,
    10134120,
    10872120,
    10873110,
    10354120,
    10374120,
)
TOKEN_IDS = (
    90023110,
    90034130,
    90071160,
    90054310,
    90054320,
    90074310,
)
SOURCE_HASHES = {
    10223120: "c596e86c7ca280e2cbc92967a9d1a81e3433d96aa54c2f46fcc382f3f52eb200",
    90023110: "f9dc5dfe3c1de8195e51d661caea8d851ff5056a88c2ae47893a8c3ce9bd0084",
    10134120: "173118b5e5fd3db5d4b267d7ae8fed88930cd64c6f1c7539197c7d79e272a7eb",
    90034130: "2740067c10bc5ae651dd41cbfd47a48eef6fc634a3ac8faaeb1b246f1fee01c6",
    10872120: "6d6d4915688b46fa0681108ddb0c65d0ccb606a4a5f91a4fa5afb7746aeadb9b",
    10873110: "b4f9a8678468b1831e615317c254c59b8014ef0e736f06f7bda23a0d0f1fbe53",
    90071160: "487bbc3ead279c8faccb6c74842ddc77225535cb22afa3b651dcdc587a172eb8",
    10354120: "ac02362707407e30cbc964a3cc24367059b5d0b03964d99e95b43dcfd1eff3f8",
    90054310: "4e18c4aded07daf7b04ac6ca56010ff3dad843bd14798041e53227135277b1b7",
    90054320: "d8fe1863df621415fee39a74ed6bd52aa773758f638b33fe1aceac66c26cfb55",
    10374120: "98f1d0abcc7ac91dcf03ba05ebe5f204ba989fed5a435b1a899baeb938345c42",
    90074310: "2aceca0355e1214fbe6c308bc2adc8dd97641e9b85b74032ced4bc8a8aa3f842",
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
    player.evolved_this_turn = False


def _choose_mode(engine, option_id: str) -> None:
    engine.apply(Choose(engine.current_player, f"choose_one:{option_id}"))


class RealGeneratedSpellAndFollowerChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1101):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_prim_produces_conditional_norga_and_super_evolve_buffs_others(self):
        engine = self.fresh(seed=3)
        prim = _play(engine, self.repository, 10223120)
        self.assertTrue(prim.has_keyword("潜行"))
        norga_card = next(card for card in engine.players[0].hand if card.card_id == 90023110)
        engine.apply(PlayCard(0, engine.players[0].hand.index(norga_card)))
        norga = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90023110)
        self.assertEqual((norga.attack, norga.max_health), (3, 3))
        self.assertTrue(norga.has_keyword("守护"))
        self.assertTrue(norga.has_keyword("必杀"))

        _enable_super_evolution(engine)
        engine.apply(SuperEvolve(0, prim.entity_id))
        self.assertEqual((prim.attack, prim.max_health), (4, 4))
        self.assertEqual((norga.attack, norga.max_health), (4, 4))

        no_prim = self.fresh(seed=5)
        _put_hand(no_prim, self.repository.get(90023110))
        no_prim.apply(PlayCard(0, 0))
        plain = no_prim.players[0].board[0]
        self.assertEqual((plain.attack, plain.max_health), (2, 2))
        self.assertTrue(plain.has_keyword("守护"))
        self.assertFalse(plain.has_keyword("必杀"))

    def test_manaria_pair_summons_spirit_spellboosts_then_evolve_targets(self):
        engine = self.fresh(seed=7)
        tracked = _put_hand(engine, _card(997001, card_type="法术", attack=None, life=None))
        enemy = _put_unit(engine, 1, _card(997002, life=6))
        source = _play(engine, self.repository, 10134120)
        spirit = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90034130)
        self.assertEqual(tracked.spellboost_count, 3)
        self.assertTrue(spirit.has_keyword("突进"))
        self.assertTrue(spirit.has_keyword("守护"))

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, enemy.entity_id)
        self.assertEqual(enemy.health, 3)

        engine.apply(EndTurn(0))
        self.assertIn(spirit, engine.players[0].board)
        engine.apply(EndTurn(1))
        self.assertNotIn(spirit, engine.players[0].board)

    def test_manaria_pair_full_board_skips_summon_but_still_spellboosts(self):
        engine = self.fresh(seed=11)
        for index in range(4):
            _put_unit(engine, 0, _card(997010 + index))
        tracked = _put_hand(engine, _card(997020, card_type="法术", attack=None, life=None))
        _play(engine, self.repository, 10134120)
        self.assertFalse(any(unit.definition.card_id == 90034130 for unit in engine.players[0].board))
        self.assertEqual(tracked.spellboost_count, 3)

    def test_brilliant_artifact_has_hand_and_destroy_then_summon_producers(self):
        hand_engine = self.fresh(seed=13)
        _play(hand_engine, self.repository, 10872120)
        artifact_card = next(card for card in hand_engine.players[0].hand if card.card_id == 90071160)
        hand_engine.apply(PlayCard(0, hand_engine.players[0].hand.index(artifact_card)))
        artifact = next(unit for unit in hand_engine.players[0].board if unit.definition.card_id == 90071160)
        self.assertTrue(artifact.has_keyword("疾驰"))

        destroy_engine = self.fresh(seed=17)
        enemy = _put_unit(destroy_engine, 1, _card(997030))
        _play(destroy_engine, self.repository, 10873110)
        _choose(destroy_engine, enemy.entity_id)
        self.assertNotIn(enemy, destroy_engine.players[1].board)
        self.assertTrue(any(
            unit.definition.card_id == 90071160 and unit.has_keyword("疾驰")
            for unit in destroy_engine.players[0].board
        ))

        no_enemy = self.fresh(seed=19)
        _play(no_enemy, self.repository, 10873110)
        self.assertIsNone(no_enemy.state.pending_choice)
        self.assertTrue(any(unit.definition.card_id == 90071160 for unit in no_enemy.players[0].board))

    def test_manifestation_mode_choices_have_rl_mask_parity(self):
        env = ShadowverseEnv(
            [_card(997100 + index) for index in range(40)],
            [_card(997200 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=23,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=23)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        _put_hand(env.core, self.repository.get(10354120))
        env.players[0].max_mana = env.players[0].mana = 10
        env.step(ShadowverseEnv.PLAY_OFFSET)
        command = Choose(0, "choose_one:love_flight")
        action = env._encode_command(command)
        self.assertIsNotNone(action)
        self.assertTrue(env.action_mask()[action])
        env.step(action)
        self.assertIsNone(env.core.state.pending_choice)
        self.assertTrue(any(card.card_id == 90054320 for card in env.players[0].hand))

        alternate = self.fresh(seed=29)
        _play(alternate, self.repository, 10354120)
        _choose_mode(alternate, "scream_spread")
        self.assertTrue(any(card.card_id == 90054310 for card in alternate.players[0].hand))

    def test_scream_spread_only_grants_rush_to_successful_summon_outputs(self):
        engine = self.fresh(seed=31)
        original = _play(engine, self.repository, 10354120)
        _choose_mode(engine, "scream_spread")
        spell = next(card for card in engine.players[0].hand if card.card_id == 90054310)
        engine.apply(PlayCard(0, engine.players[0].hand.index(spell)))
        copies = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10354120 and unit.entity_id != original.entity_id
        ]
        self.assertEqual(len(copies), 2)
        self.assertTrue(all(unit.has_keyword("突进") for unit in copies))
        self.assertFalse(original.has_keyword("突进"))

        shortage = self.fresh(seed=37)
        source = _play(shortage, self.repository, 10354120)
        _choose_mode(shortage, "scream_spread")
        for index in range(3):
            _put_unit(shortage, 0, _card(997300 + index))
        spell = next(card for card in shortage.players[0].hand if card.card_id == 90054310)
        shortage.apply(PlayCard(0, shortage.players[0].hand.index(spell)))
        copies = [
            unit for unit in shortage.players[0].board
            if unit.definition.card_id == 10354120 and unit.entity_id != source.entity_id
        ]
        self.assertEqual(len(copies), 1)
        self.assertTrue(copies[0].has_keyword("突进"))

    def test_love_flight_requires_exact_follower_and_illegal_play_is_atomic(self):
        illegal = self.fresh(seed=41)
        _put_hand(illegal, self.repository.get(90054320))
        before = illegal.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            illegal.apply(PlayCard(0, 0))
        self.assertEqual(illegal.deterministic_fingerprint(), before)

        engine = self.fresh(seed=43)
        source = _play(engine, self.repository, 10354120)
        _choose_mode(engine, "love_flight")
        spell = next(card for card in engine.players[0].hand if card.card_id == 90054320)
        engine.apply(PlayCard(0, engine.players[0].hand.index(spell)))
        _choose(engine, source.entity_id)
        self.assertTrue(source.has_keyword("疾驰"))

    def test_lishenna_immunity_prevents_destroy_but_solo_damage_continues(self):
        engine = self.fresh(seed=47)
        enemy = _put_unit(engine, 1, _card(997400, life=6))
        source = _play(engine, self.repository, 10374120)
        solo = next(card for card in engine.players[0].hand if card.card_id == 90074310)
        engine.apply(PlayCard(0, engine.players[0].hand.index(solo)))
        _choose(engine, source.entity_id)

        self.assertIn(source, engine.players[0].board)
        self.assertEqual(source.health, source.max_health)
        self.assertEqual(enemy.health, 2)
        prevented = [
            event for event in engine.event_history
            if event.type is EventType.EFFECT_DESTROY_PREVENTED
        ]
        self.assertEqual(len(prevented), 1)
        self.assertEqual(prevented[0].target_id, source.entity_id)
        self.assertEqual(prevented[0].metadata["protected_card_id"], 10374120)

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertTrue(any(unit.definition.card_id == 90074210 for unit in engine.players[0].board))

    def test_destroy_immunity_does_not_block_banish_or_zero_health_and_removal_disables_it(self):
        def effect_engine(kind: EffectKind, *, seed: int):
            spell_id = 997500 + seed
            rulebook = RuleBook(
                rules=(CardRule(
                    spell_id,
                    Trigger.PLAY,
                    (EffectOperation(kind, TargetKind.ALL_ENEMY_UNITS),),
                ),),
                passives=(CardPassive(10374120, "cannot_be_destroyed_by_effects", 0),),
            )
            engine = _fresh(rulebook, self.repository, seed=seed)
            target = _put_unit(engine, 1, self.repository.get(10374120))
            spell = _card(spell_id, cost=0, card_type="法术", attack=None, life=None)
            _put_hand(engine, spell)
            return engine, target

        destroy, protected = effect_engine(EffectKind.DESTROY, seed=53)
        destroy.apply(PlayCard(0, 0))
        self.assertIn(protected, destroy.players[1].board)

        protected.remove_all_abilities()
        _put_hand(destroy, _card(997553, cost=0, card_type="法术", attack=None, life=None))
        destroy.apply(PlayCard(0, 0))
        self.assertNotIn(protected, destroy.players[1].board)

        banish, target = effect_engine(EffectKind.BANISH, seed=59)
        banish.apply(PlayCard(0, 0))
        self.assertNotIn(target, banish.players[1].board)
        self.assertIn(10374120, [card.card_id for card in banish.players[1].banished])

        lethal, target = effect_engine(EffectKind.DESTROY, seed=61)
        target.health = 0
        lethal._stabilize()
        self.assertNotIn(target, lethal.players[1].board)

    def test_solo_requires_own_board_target_without_rng_or_state_mutation(self):
        engine = self.fresh(seed=67)
        _put_hand(engine, self.repository.get(90074310))
        rng_before = engine.random.getstate()
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.deterministic_fingerprint(), before)
        self.assertEqual(engine.random.getstate(), rng_before)

    def test_all_twelve_cards_are_exact_and_tokens_have_authored_producers(self):
        coverage = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in COLLECTIBLE_IDS:
            with self.subTest(card_id=card_id):
                info = coverage["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )

        audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        tokens = {card["card_id"]: card for card in audit["cards"]}
        for card_id in TOKEN_IDS:
            with self.subTest(card_id=card_id):
                info = tokens[card_id]
                self.assertEqual(info["category"], "entry_behavior_complete")
                self.assertEqual(info["explicit_coverage"], "exact")
                self.assertTrue(info["authored_producers"])
                self.assertEqual(len(SOURCE_HASHES[card_id]), 64)


if __name__ == "__main__":
    unittest.main()
