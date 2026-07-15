# -*- coding: utf-8 -*-
"""Exact spell modes, summons, Fusion, and Earth Rite hand listeners."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import BeginFusion, Choose, EndTurn, PlayCard
from swb.engine.origin import CardOrigin
from swb.engine.state import Amulet
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10122310,
    10132310,
    10211310,
    10241310,
    10323310,
    10332310,
    10432310,
    10542310,
    10611310,
    10621310,
    10731310,
    10732310,
    10733310,
    10803310,
    10823310,
)


def _choose_id(engine, option_id: str) -> None:
    engine.apply(Choose(engine.state.pending_choice.player_index, option_id))


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


class RealSpellModesAndEarthListenerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 801):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_royal_mode_summons_exact_pair_or_buffs_existing_followers(self):
        summon = self.fresh(seed=3)
        _play(summon, self.repository, 10122310)
        _choose_id(summon, "choose_one:summon_knights")
        self.assertEqual(
            [entity.definition.card_id for entity in summon.players[0].board],
            [90021120, 90021110],
        )

        buff = self.fresh(seed=5)
        followers = [_put_unit(buff, 0, _card(10 + i)) for i in range(2)]
        before = [(unit.attack, unit.max_health) for unit in followers]
        _play(buff, self.repository, 10122310)
        _choose_id(buff, "choose_one:buff_followers")
        self.assertEqual(
            [(unit.attack, unit.max_health) for unit in followers],
            [(attack + 1, health + 1) for attack, health in before],
        )

    def test_proof_modes_cover_sigils_heal_and_paid_mass_damage(self):
        sigils = self.fresh(seed=7)
        _play(sigils, self.repository, 10132310)
        _choose_id(sigils, "choose_one:earth_sigils")
        self.assertEqual(sigils.players[0].earth_sigils, 4)

        heal = self.fresh(seed=11)
        heal.players[0].health = 15
        _play(heal, self.repository, 10132310)
        _choose_id(heal, "choose_one:heal")
        self.assertEqual(heal.players[0].health, 19)

        damage = self.fresh(seed=13)
        _put_sigil(damage, self.repository, 3)
        enemies = [_put_unit(damage, 1, _card(20 + i, life=6)) for i in range(2)]
        _play(damage, self.repository, 10132310)
        _choose_id(damage, "choose_one:earth_rite_damage")
        self.assertEqual(damage.players[0].earth_sigils, 0)
        self.assertTrue(all(enemy.health == 2 for enemy in enemies))

    def test_exact_multi_summons_preserve_printed_order(self):
        forest = self.fresh(seed=17)
        _play(forest, self.repository, 10211310)
        self.assertEqual(
            [entity.definition.card_id for entity in forest.players[0].board],
            [10011130, 10011130, 10011130],
        )

        wizard = self.fresh(seed=19)
        _play(wizard, self.repository, 10332310)
        self.assertEqual(
            [entity.definition.card_id for entity in wizard.players[0].board],
            [90031120, 90031110],
        )

    def test_orca_call_draws_exact_copy_only_at_overflow(self):
        below = self.fresh(seed=23)
        below.players[0].max_mana = below.players[0].mana = 6
        below.players[0].deck = [self.repository.get(10241310), _card(30)]
        _play(below, self.repository, 10241310)
        self.assertEqual(len(below.players[0].hand), 0)
        self.assertEqual(below.players[0].board[0].definition.card_id, 90041130)

        active = self.fresh(seed=29)
        active.players[0].max_mana = active.players[0].mana = 7
        active.players[0].deck = [_card(31), self.repository.get(10241310)]
        _play(active, self.repository, 10241310)
        self.assertEqual([card.card_id for card in active.players[0].hand], [10241310])

    def test_treasure_fusion_adds_dagger_and_only_fused_copy_draws(self):
        plain = self.fresh(seed=31)
        plain.players[0].deck = [_card(40)]
        enemy = _put_unit(plain, 1, _card(41, life=5))
        _play(plain, self.repository, 10323310)
        self.assertEqual(enemy.health, 3)
        self.assertEqual([card.card_id for card in plain.players[0].hand], [90021310])

        fused = self.fresh(seed=37)
        fused.players[0].deck = [_card(42)]
        target = _put_hand(fused, self.repository.get(10323310))
        material = _put_hand(fused, self.repository.get(90021310))
        fused.apply(BeginFusion(0, target.entity_id))
        _choose_id(fused, f"hand:{material.entity_id}")
        _choose_id(fused, "fusion:confirm")
        fused.apply(PlayCard(0, fused.players[0].hand.index(target)))
        self.assertEqual(
            sorted(card.card_id for card in fused.players[0].hand),
            [42, 90021310],
        )

    def test_energy_overflow_modes_draw_before_seeded_distinct_damage(self):
        single = self.fresh(seed=41)
        single.players[0].deck = [_card(50)]
        enemies = [_put_unit(single, 1, _card(51 + i, life=6)) for i in range(2)]
        _play(single, self.repository, 10432310)
        _choose_id(single, "choose_one:single")
        self.assertEqual(len(single.players[0].hand), 1)
        self.assertEqual(sorted(enemy.health for enemy in enemies), [2, 6])

        double = self.fresh(seed=43)
        double.players[0].deck = [_card(60), _card(61)]
        enemies = [_put_unit(double, 1, _card(62 + i, life=6)) for i in range(3)]
        _play(double, self.repository, 10432310)
        _choose_id(double, "choose_one:double")
        self.assertEqual(len(double.players[0].hand), 2)
        self.assertEqual(sorted(enemy.health for enemy in enemies), [2, 2, 6])
        self.assertEqual(double.players[0].health, 18)

    def test_prominence_roar_snapshots_total_follower_count_for_all_damage(self):
        engine = self.fresh(seed=47)
        own = [_put_unit(engine, 0, _card(70 + i, life=5)) for i in range(2)]
        enemies = [_put_unit(engine, 1, _card(72 + i, life=5)) for i in range(2)]
        _play(engine, self.repository, 10542310)
        self.assertTrue(all(unit.health == 1 for unit in own + enemies))

    def test_spear_grant_damages_before_summoning_two_exact_tokens(self):
        engine = self.fresh(seed=53)
        enemies = [_put_unit(engine, 1, _card(80 + i, life=4)) for i in range(2)]
        _play(engine, self.repository, 10611310)
        self.assertTrue(all(enemy not in engine.players[1].board for enemy in enemies))
        self.assertEqual(
            [entity.definition.card_id for entity in engine.players[0].board],
            [90011120, 90011120],
        )

    def test_sword_grant_enhance_buffs_only_successful_summon_outputs(self):
        normal = self.fresh(seed=59)
        _play(normal, self.repository, 10621310)
        definition = self.repository.get(10621110)
        self.assertEqual(len(normal.players[0].board), 3)
        self.assertTrue(all(
            (unit.attack, unit.max_health) == (definition.attack, definition.life)
            for unit in normal.players[0].board
        ))

        enhanced = self.fresh(seed=61)
        for i in range(3):
            _put_unit(enhanced, 0, _card(90 + i))
        _play(enhanced, self.repository, 10621310, mode_id="enhance_7")
        summoned = [
            unit for unit in enhanced.players[0].board
            if unit.definition.card_id == 10621110
        ]
        self.assertEqual(len(summoned), 2)
        self.assertTrue(all(
            (unit.attack, unit.max_health)
            == (definition.attack + 2, definition.life + 2)
            for unit in summoned
        ))

    def test_earth_rite_hand_listeners_stack_until_turn_end_and_play_normally(self):
        engine = self.fresh(seed=67)
        _put_sigil(engine, self.repository, 4)
        draw_spell = _put_hand(engine, self.repository.get(10731310))
        destroy_spell = _put_hand(engine, self.repository.get(10733310))
        _play(engine, self.repository, 10732310)
        _choose_id(engine, "choose_one:earth_rite_damage")
        self.assertEqual(draw_spell.current_cost, 3)
        self.assertEqual(destroy_spell.current_cost, 3)

        engine.players[0].mana = 10
        _play(engine, self.repository, 10732310)
        _choose_id(engine, "choose_one:earth_rite_damage")
        self.assertEqual(draw_spell.current_cost, 2)
        self.assertEqual(destroy_spell.current_cost, 2)
        engine.apply(EndTurn(0))
        self.assertEqual(draw_spell.current_cost, 4)
        self.assertEqual(destroy_spell.current_cost, 4)

        draw_engine = self.fresh(seed=71)
        draw_engine.players[0].deck = [_card(100), _card(101)]
        _play(draw_engine, self.repository, 10731310)
        self.assertEqual(len(draw_engine.players[0].hand), 2)
        self.assertEqual(draw_engine.players[0].earth_sigils, 1)

        destroy_engine = self.fresh(seed=73)
        enemy = _put_unit(destroy_engine, 1, _card(102))
        _play(destroy_engine, self.repository, 10733310)
        option = next(
            option for option in destroy_engine.state.pending_choice.options
            if option.entity_id == enemy.entity_id
        )
        _choose_id(destroy_engine, option.option_id)
        self.assertNotIn(enemy, destroy_engine.players[1].board)
        self.assertEqual(destroy_engine.players[0].earth_sigils, 2)

    def test_snack_modes_do_not_spend_insufficient_sigils(self):
        gain = self.fresh(seed=79)
        _play(gain, self.repository, 10732310)
        _choose_id(gain, "choose_one:earth_sigils")
        self.assertEqual(gain.players[0].earth_sigils, 4)

        insufficient = self.fresh(seed=83)
        _put_sigil(insufficient, self.repository, 1)
        enemy = _put_unit(insufficient, 1, _card(110, life=5))
        _play(insufficient, self.repository, 10732310)
        _choose_id(insufficient, "choose_one:earth_rite_damage")
        self.assertEqual(insufficient.players[0].earth_sigils, 1)
        self.assertEqual(enemy.health, 5)

    def test_inherited_will_all_three_modes_resolve(self):
        damage = self.fresh(seed=89)
        enemy = _put_unit(damage, 1, _card(120, life=5))
        _play(damage, self.repository, 10803310)
        _choose_id(damage, "choose_one:damage")
        self.assertEqual(enemy.health, 2)

        reanimate = self.fresh(seed=97)
        corpse = _put_unit(reanimate, 0, _card(121, cost=2))
        corpse.health = 0
        reanimate._stabilize()
        _play(reanimate, self.repository, 10803310)
        _choose_id(reanimate, "choose_one:reanimate")
        self.assertTrue(any(unit.definition.card_id == 121 for unit in reanimate.players[0].board))

        buff = self.fresh(seed=101)
        allies = [_put_unit(buff, 0, _card(122 + i)) for i in range(2)]
        _play(buff, self.repository, 10803310)
        _choose_id(buff, "choose_one:buff")
        self.assertTrue(all(unit.attack == 2 for unit in allies))

    def test_daily_life_chooses_below_two_cards_and_auto_runs_both_at_two(self):
        choice = self.fresh(seed=103)
        choice.players[0].health = 16
        _put_unit(choice, 0, _card(130))
        _play(choice, self.repository, 10823310)
        self.assertIsNotNone(choice.state.pending_choice)
        _choose_id(choice, "choose_one:heal")
        self.assertEqual(choice.players[0].health, 18)

        automatic = self.fresh(seed=107)
        automatic.players[0].health = 16
        for i in range(2):
            _put_unit(automatic, 0, _card(131 + i))
        enemy = _put_unit(automatic, 1, _card(133, life=6))
        _play(automatic, self.repository, 10823310)
        self.assertIsNone(automatic.state.pending_choice)
        self.assertEqual(automatic.players[0].health, 18)
        self.assertEqual(enemy.health, 2)


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
                    ["tests/test_real_spell_modes_and_earth_listener_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
