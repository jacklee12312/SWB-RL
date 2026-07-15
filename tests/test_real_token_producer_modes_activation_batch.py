# -*- coding: utf-8 -*-
"""Exact producer chains for six previously database-only generated cards."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import ActivateAmulet, PlayCard, SuperEvolve
from swb.engine.environment import ShadowverseEnv
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _choose,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (
    10042120,
    10141110,
    10143210,
    10144130,
    10061210,
    10821110,
)
TOKEN_IDS = (
    90021130,
    90041120,
    90043110,
    90044110,
    90044120,
    90061130,
)
SOURCE_HASHES = {
    10042120: "431f28ca7de868273517859a97944e1494cfb33ebad3e361e2168605bc14d586",
    10141110: "57c220c1ed2a6aa82526e674c920cad50f93a167256d90e663e1c91d0414f4b1",
    10143210: "9137386af617951740320409a35a17b73c5f6836ebd1ce0c57e0229038b7a731",
    10144130: "ded162b03cc2cd5af2311731d26965ff665de05c8aea82d549d3cd665ef6c450",
    10061210: "b33a6aea9d5182695d5b51700765effc8d5eb5abb1e5ad7d90733253f58cc07e",
    10821110: "fe7b5584c629eec384b59979e20202057678becef3d9fccf755b18b7145c14b2",
    90021130: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    90041120: "cd92ce1e5198aaf7cc502f0da656dc53a87900054d0086142000fe2dfe081b09",
    90043110: "aa52b7161a6c63106daab4129b5846597d52b722f34aad71e5973667f1f3affa",
    90044110: "fd22b46c04522f13395a9cb2a5444e83007a05cf6121d94ba170126602ed58c5",
    90044120: "7abde1fa8f2c436a6805d99abe026718965b9776b4b5a2dffe05f57cd5b2776f",
    90061130: "487bbc3ead279c8faccb6c74842ddc77225535cb22afa3b651dcdc587a172eb8",
}


def _enable_super_evolve(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.first_player_super_evolution_unlock_turn
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.super_evolved_this_turn = False
    player.evolved_this_turn = False


class RealTokenProducerModesActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_roaring_dragoneer_enhance_and_rl_mask_summon_intimidate_dragon(self):
        normal = self.fresh(seed=3)
        _play(normal, self.repository, 10042120)
        self.assertFalse(any(
            unit.definition.card_id == 90041120
            for unit in normal.players[0].board
        ))

        enhanced = self.fresh(seed=5)
        _play(enhanced, self.repository, 10042120, mode_id="enhance_7")
        token = next(
            unit for unit in enhanced.players[0].board
            if unit.definition.card_id == 90041120
        )
        self.assertTrue(token.has_keyword("威慑"))

        env = ShadowverseEnv(
            [_card(995000 + index) for index in range(40)],
            [_card(996000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=7,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=7)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        _put_hand(env.core, self.repository.get(10042120))
        mode_action = ShadowverseEnv.MODE_PLAY_OFFSET
        env.players[0].max_mana = 7
        env.players[0].mana = 6
        self.assertFalse(env.action_mask()[mode_action])
        env.players[0].mana = 7
        self.assertTrue(env.action_mask()[mode_action])
        env.step(mode_action)
        self.assertTrue(any(
            unit.definition.card_id == 90041120
            for unit in env.players[0].board
        ))

    def test_cloudsea_dragon_rider_last_words_frees_slot_then_summons_token(self):
        engine = self.fresh(seed=11)
        source = _play(engine, self.repository, 10141110)
        self.assertTrue(source.has_keyword("守护"))
        for index in range(4):
            _put_unit(engine, 0, _card(995100 + index))
        self.assertEqual(len(engine.players[0].board), 5)

        source.health = 0
        engine._stabilize()

        self.assertNotIn(source, engine.players[0].board)
        token = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90041120
        )
        self.assertTrue(token.has_keyword("威慑"))
        self.assertEqual(len(engine.players[0].board), 5)

    def test_otohime_fan_activation_summons_before_selected_discard(self):
        engine = self.fresh(seed=13)
        amulet = _play(engine, self.repository, 10143210)
        discard = _put_hand(engine, _card(995200))

        engine.apply(ActivateAmulet(0, amulet.entity_id))

        guard = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90043110
        )
        self.assertTrue(guard.has_keyword("疾驰"))
        self.assertTrue(guard.has_keyword("守护"))
        self.assertIsNotNone(engine.state.pending_choice)
        _choose(engine, discard.entity_id)
        self.assertNotIn(discard, engine.players[0].hand)
        self.assertEqual(engine.players[0].mana, 6)

    def test_otohime_fan_full_board_skips_summon_but_still_discards(self):
        engine = self.fresh(seed=17)
        amulet = _play(engine, self.repository, 10143210)
        for index in range(4):
            _put_unit(engine, 0, _card(995300 + index))
        discard = _put_hand(engine, _card(995310))

        engine.apply(ActivateAmulet(0, amulet.entity_id))
        _choose(engine, discard.entity_id)

        self.assertFalse(any(
            unit.definition.card_id == 90043110
            for unit in engine.players[0].board
        ))
        self.assertNotIn(discard, engine.players[0].hand)

    def test_wolong_summons_both_dragons_then_super_evolve_filters_each_name(self):
        engine = self.fresh(seed=19)
        source = _play(engine, self.repository, 10144130)
        gold = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90044110)
        silver = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90044120)
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertTrue(source.has_keyword("屏障"))
        self.assertTrue(gold.has_keyword("守护"))
        self.assertTrue(silver.has_keyword("突进"))
        self.assertFalse(gold.has_keyword("疾驰"))
        self.assertFalse(silver.has_keyword("屏障"))

        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))

        self.assertTrue(gold.has_keyword("疾驰"))
        self.assertTrue(silver.has_keyword("屏障"))

    def test_wolong_board_shortage_only_buffs_successfully_summoned_gold(self):
        engine = self.fresh(seed=23)
        for index in range(3):
            _put_unit(engine, 0, _card(995400 + index))
        source = _play(engine, self.repository, 10144130)
        self.assertEqual(
            [unit.definition.card_id for unit in engine.players[0].board].count(90044110),
            1,
        )
        self.assertFalse(any(
            unit.definition.card_id == 90044120
            for unit in engine.players[0].board
        ))
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        gold = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90044110)
        self.assertTrue(gold.has_keyword("疾驰"))

    def test_projected_bird_statue_activation_expires_and_summons_storm_falcon(self):
        engine = self.fresh(seed=29)
        amulet = _play(engine, self.repository, 10061210)
        self.assertEqual(amulet.countdown, 2)

        engine.apply(ActivateAmulet(0, amulet.entity_id))

        self.assertNotIn(amulet, engine.players[0].board)
        falcon = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90061130
        )
        self.assertTrue(falcon.has_keyword("疾驰"))
        self.assertEqual(engine.players[0].mana, 4)

    def test_nahato_fanfare_and_super_evolve_repeat_summon_then_damage(self):
        engine = self.fresh(seed=31)
        for index in range(3):
            _put_unit(engine, 0, _card(995500 + index))
        enemies = [
            _put_unit(engine, 1, _card(995510 + index, life=7))
            for index in range(2)
        ]
        source = _play(engine, self.repository, 10821110)
        self.assertEqual([enemy.health for enemy in enemies], [4, 4])
        self.assertEqual(
            [unit.definition.card_id for unit in engine.players[0].board].count(90021130),
            1,
        )

        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))

        self.assertEqual([enemy.health for enemy in enemies], [1, 1])
        self.assertEqual(
            [unit.definition.card_id for unit in engine.players[0].board].count(90021130),
            1,
        )
        token = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90021130)
        self.assertEqual((token.attack, token.max_health), (4, 4))

    def test_all_twelve_cards_are_exact_and_tokens_have_real_producers(self):
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
