# -*- coding: utf-8 -*-
"""Exact generated burst-spell producers and their executable token chains."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.origin import CardOrigin
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import IllegalCommand
from swb.engine.state import HandCard
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _choose,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (
    10304120,
    10824110,
    10654120,
    10314120,
    10834110,
    10114120,
)
TOKEN_IDS = (
    90004330,
    90024330,
    90054330,
    90014320,
    90034340,
    90034350,
    90014310,
)
SOURCE_HASHES = {
    10304120: "b7cfef33c62035057b0c42c94c7718b0221b8daa42af38ffb44c94bb24f62ed2",
    90004330: "6d7dc2ea7e15a0d53b62ca97fd1fe43ee4f4fd577b52e3820e42a7e6ddd73140",
    10824110: "9498a685b03e8762ce89d5eb39820ce87499ab5be05ebf51ef6c26d26d472213",
    90024330: "8ee84d0e4eacb74d7a6b59cb5c69a1f3b4c2edc788042a50fac17d1bf0ec5e20",
    10654120: "e9c0b41c8031ebdd745f745f0f9690bf8840f3eafda3126d96bcd7cc850f5fbc",
    90054330: "b19dd5e5718f692fb1413eff1bf9d6eafc40bce626a4cb13da5cd7c079c003db",
    10314120: "bda0d5939bd7bc554f1e9dad6720c08d4e26c9e72c547ef1c96d94d7d45d5242",
    90014320: "408c462fbcebefcdfa6d284cf0730e433c6309c4afac5ee114e85e63899e04c6",
    10834110: "3cfa73a9e12a75d8711b089b28a466e5bb8e9656ba211600545886c17b8d6434",
    90034340: "da2047c1114ad05dde91d34925b5edb8499d4355fb705648b00e587977844bdb",
    90034350: "10566adb565e5c730051dd5e3583193b4b64ad067a767a7b739b0c01be328e3d",
    10114120: "0a49dbe5d679a43854e226204d998208b4da6f2aff95416375d984bee75e26f3",
    90014310: "4dd04ef5953b94879278d179ba16ce47cd85f7987dd39ce8b430ad67ddc8cc88",
}


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _choose_leader(engine, player_index: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option
        for option in request.options
        if option.leader_player_index == player_index
    )
    engine.apply(Choose(request.player_index, option.option_id))


class RealGeneratedBurstSpellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1201):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_gilnelise_no_other_target_skips_debuff_but_continues_fanfare(self):
        engine = self.fresh(seed=3)
        engine.players[1].max_mana = engine.players[1].mana = 10
        source = _play(engine, self.repository, 10304120)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(len(engine.players[0].board), 1)
        self.assertEqual(source.definition.card_id, 10304120)
        self.assertTrue(any(
            card.card_id == 90004330 for card in engine.players[0].hand
        ))

    def test_gilnelise_both_max_mana_boundary_and_nectar_resolution(self):
        engine = self.fresh(seed=5)
        enemy = _put_unit(engine, 1, _card(998001, attack=2, life=7))
        engine.players[1].max_mana = 10
        source = _play(engine, self.repository, 10304120)
        self.assertTrue(source.has_keyword("虹吸"))
        _choose(engine, enemy.entity_id)
        self.assertEqual((enemy.attack, enemy.max_health, enemy.health), (4, 5, 5))
        nectar = next(card for card in engine.players[0].hand if card.card_id == 90004330)
        engine.apply(PlayCard(0, engine.players[0].hand.index(nectar)))
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertEqual(engine.players[1].health, 15)

        below = self.fresh(seed=7)
        enemy = _put_unit(below, 1, _card(998002, life=7))
        below.players[1].max_mana = below.players[1].mana = 9
        _play(below, self.repository, 10304120)
        _choose(below, enemy.entity_id)
        self.assertFalse(any(card.card_id == 90004330 for card in below.players[0].hand))

        no_follower = self.fresh(seed=9)
        _put_hand(no_follower, self.repository.get(90004330))
        no_follower.apply(PlayCard(0, 0))
        self.assertEqual(no_follower.players[1].health, 15)

    def test_gilnelise_evolution_can_select_itself(self):
        engine = self.fresh(seed=11)
        enemy = _put_unit(engine, 1, _card(998003, life=7))
        source = _play(engine, self.repository, 10304120)
        _choose(engine, enemy.entity_id)
        _enable_evolution(engine)
        before = (source.attack, source.max_health)
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, source.entity_id)
        self.assertEqual(
            (source.attack, source.max_health),
            (before[0] + 4, before[1]),
        )

    def test_super_evolution_after_set_health_uses_official_visible_plus_three(self):
        engine = self.fresh(seed=12)
        medusa = _put_unit(engine, 0, self.repository.get(10154120))
        _play(engine, self.repository, 10304120)
        _choose(engine, medusa.entity_id)
        self.assertEqual((medusa.health, medusa.max_health), (5, 5))

        snowman_rampage = HandCard(
            definition=self.repository.get(10132320),
            entity_id=engine.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        opponent = engine.players[1]
        opponent.hand = [snowman_rampage]
        opponent.hand_entity_ids = [snowman_rampage.entity_id]
        engine.state.active_player = 1
        engine.apply(PlayCard(1, 0))
        _choose(engine, medusa.entity_id)
        self.assertEqual((medusa.health, medusa.max_health), (1, 1))

        engine.state.active_player = 0
        player = engine.players[0]
        player.turns_started = (
            engine.config.first_player_super_evolution_unlock_turn
        )
        engine.apply(SuperEvolve(0, medusa.entity_id))

        self.assertEqual(
            (medusa.attack, medusa.health, medusa.max_health),
            (8, 4, 4),
        )

    def test_ordinary_evolution_after_set_health_uses_official_visible_plus_two(self):
        engine = self.fresh(seed=120)
        medusa = _put_unit(engine, 0, self.repository.get(10154120))
        _play(engine, self.repository, 10304120)
        _choose(engine, medusa.entity_id)

        snowman_rampage = HandCard(
            definition=self.repository.get(10132320),
            entity_id=engine.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        opponent = engine.players[1]
        opponent.hand = [snowman_rampage]
        opponent.hand_entity_ids = [snowman_rampage.entity_id]
        engine.state.active_player = 1
        engine.apply(PlayCard(1, 0))
        _choose(engine, medusa.entity_id)
        self.assertEqual(
            (medusa.attack, medusa.health, medusa.max_health),
            (5, 1, 1),
        )

        engine.state.active_player = 0
        _enable_evolution(engine)
        engine.apply(Evolve(0, medusa.entity_id))

        self.assertEqual(
            (medusa.attack, medusa.health, medusa.max_health),
            (7, 3, 3),
        )

    def test_bunny_barons_cooperation_threshold_capacity_and_two_shots(self):
        engine = self.fresh(seed=13)
        engine.players[0].cooperation = 18
        source = _play(engine, self.repository, 10824110)
        copies = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10824110
        ]
        self.assertEqual(len(copies), 2)
        self.assertTrue(all(unit.has_keyword("突进") for unit in copies))
        self.assertEqual((engine.players[0].cooperation, engine.players[1].health), (20, 16))

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        shot = next(card for card in engine.players[0].hand if card.card_id == 90024330)
        first = _put_unit(engine, 1, _card(998010, life=4))
        second = _put_unit(engine, 1, _card(998011, life=4))
        engine.apply(PlayCard(0, engine.players[0].hand.index(shot)))
        self.assertNotIn(first, engine.players[1].board)
        self.assertNotIn(second, engine.players[1].board)

        full = self.fresh(seed=15)
        full.players[0].cooperation = 18
        for index in range(4):
            _put_unit(full, 0, _card(998020 + index))
        _play(full, self.repository, 10824110)
        self.assertEqual(full.players[0].cooperation, 19)
        self.assertEqual(full.players[1].health, 20)

    def test_bibati_necromancy_evolves_then_produces_draw_spell(self):
        engine = self.fresh(seed=17)
        engine.players[0].shadows = 4
        source = _play(engine, self.repository, 10654120)
        self.assertTrue(source.evolved)
        self.assertEqual(engine.players[0].shadows, 0)
        abyss = next(card for card in engine.players[0].hand if card.card_id == 90054330)
        hand_before = len(engine.players[0].hand)
        engine.apply(PlayCard(0, engine.players[0].hand.index(abyss)))
        self.assertEqual(len(engine.players[0].hand), hand_before + 1)

        short = self.fresh(seed=19)
        short.players[0].shadows = 3
        source = _play(short, self.repository, 10654120)
        self.assertFalse(source.evolved)
        self.assertEqual(short.players[0].shadows, 3)
        self.assertFalse(any(card.card_id == 90054330 for card in short.players[0].hand))

    def test_izudia_debuff_producer_hand_listener_and_damage_branches(self):
        engine = self.fresh(seed=21)
        enemy = _put_unit(engine, 1, _card(998030, life=6))
        source = _play(engine, self.repository, 10314120)
        _choose(engine, enemy.entity_id)
        self.assertNotIn(enemy, engine.players[1].board)
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        strike = next(card for card in engine.players[0].hand if card.card_id == 90014320)
        self.assertEqual(strike.current_cost, 6)
        engine.apply(EndTurn(0))
        self.assertEqual(strike.current_cost, 5)
        engine.apply(EndTurn(1))
        self.assertEqual(strike.current_cost, 5)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, engine.players[0].hand.index(strike)))
        self.assertEqual(engine.players[1].health, 0)

        no_source = self.fresh(seed=23)
        _put_hand(no_source, self.repository.get(90014320))
        no_source.apply(PlayCard(0, 0))
        self.assertEqual(no_source.players[1].health, 14)

    def test_tantara_uses_captured_play_time_spellboost_at_both_thresholds(self):
        expected = {
            9: set(),
            10: {90034340},
            19: {90034340},
            20: {90034340, 90034350},
        }
        for count, generated in expected.items():
            with self.subTest(spellboost_count=count):
                engine = self.fresh(seed=30 + count)
                hand_card = _put_hand(engine, self.repository.get(10834110))
                hand_card.spellboost_count = count
                engine.apply(PlayCard(0, 0))
                source = next(
                    unit for unit in engine.players[0].board
                    if unit.definition.card_id == 10834110
                )
                self.assertTrue(source.has_keyword("疾驰"))
                self.assertEqual(
                    {
                        card.card_id for card in engine.players[0].hand
                        if card.card_id in {90034340, 90034350}
                    },
                    generated,
                )

    def test_azure_four_and_grand_return_execute_with_rl_mask_parity(self):
        engine = self.fresh(seed=53)
        first = _put_unit(engine, 1, _card(998040, life=5))
        second = _put_unit(engine, 1, _card(998041, life=6))
        _put_hand(engine, self.repository.get(90034340))
        engine.apply(PlayCard(0, 0))
        self.assertNotIn(first, engine.players[1].board)
        self.assertEqual(second.health, 1)

        env = ShadowverseEnv(
            [_card(998100 + index) for index in range(40)],
            [_card(998200 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=59,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=59)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].board.clear()
        target = _put_unit(env.core, 0, _card(998300))
        _put_hand(env.core, self.repository.get(90034350))
        env.players[0].max_mana = env.players[0].mana = 10
        env.step(ShadowverseEnv.PLAY_OFFSET)
        option = next(
            option for option in env.core.state.pending_choice.options
            if option.entity_id == target.entity_id
        )
        command = Choose(0, option.option_id)
        action = env._encode_command(command)
        self.assertIsNotNone(action)
        self.assertTrue(env.action_mask()[action])
        env.step(action)
        self.assertEqual(target.attacks_per_turn, 2)

        illegal = self.fresh(seed=61)
        _put_hand(illegal, self.repository.get(90034350))
        before = illegal.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            illegal.apply(PlayCard(0, 0))
        self.assertEqual(illegal.deterministic_fingerprint(), before)

    def test_rose_queen_transforms_only_matching_hand_cards_and_has_static_capacity(self):
        engine = self.fresh(seed=67)
        eligible_follower = _put_hand(
            engine,
            _card(998401, class_id=1, class_name="精灵", cost=2),
        )
        eligible_spell = _put_hand(
            engine,
            _card(
                998402,
                class_id=1,
                class_name="精灵",
                cost=1,
                card_type="法术",
                attack=None,
                life=None,
            ),
        )
        eligible_spell.spellboost_count = 4
        neutral = _put_hand(engine, _card(998403, class_id=0, class_name="中立", cost=1))
        expensive = _put_hand(engine, _card(998404, class_id=1, class_name="精灵", cost=3))
        other_class = _put_hand(engine, _card(998405, class_id=2, class_name="皇家", cost=1))
        original_ids = (eligible_follower.entity_id, eligible_spell.entity_id)

        rose = _play(engine, self.repository, 10114120)
        self.assertTrue(rose.has_keyword("守护"))
        self.assertEqual(rose.attacks_per_turn, 2)
        self.assertEqual(
            (eligible_follower.card_id, eligible_spell.card_id),
            (90014310, 90014310),
        )
        self.assertEqual(
            (eligible_follower.entity_id, eligible_spell.entity_id),
            original_ids,
        )
        self.assertEqual(eligible_spell.spellboost_count, 0)
        self.assertEqual(
            (neutral.card_id, expensive.card_id, other_class.card_id),
            (998403, 998404, 998405),
        )

        enemy = _put_unit(engine, 1, _card(998406, life=5))
        engine.players[0].mana = 10
        engine.apply(
            PlayCard(0, engine.players[0].hand.index(eligible_follower))
        )
        _choose(engine, enemy.entity_id)
        self.assertEqual(enemy.health, 2)

        rose.remove_all_abilities()
        self.assertFalse(rose.has_keyword("守护"))
        self.assertEqual(rose.attacks_per_turn, 1)

    def test_rose_strike_can_select_enemy_leader(self):
        engine = self.fresh(seed=71)
        _put_hand(engine, self.repository.get(90014310))
        engine.apply(PlayCard(0, 0))
        _choose_leader(engine, 1)
        self.assertEqual(engine.players[1].health, 17)

    def test_static_attack_capacity_passive_schema_rejects_invalid_amounts(self):
        for amount in (0, -1, True, 1.5):
            with self.subTest(amount=amount), tempfile.TemporaryDirectory() as tmp:
                payload = {
                    "passives": [
                        {
                            "card_id": 998500,
                            "kind": "attacks_per_turn",
                            "amount": amount,
                        }
                    ],
                    "rules": [],
                }
                Path(tmp, "rules.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "amount|attacks_per_turn"):
                    RuleBook.from_directory(tmp)

    def test_all_thirteen_cards_are_exact_and_tokens_have_producers(self):
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
