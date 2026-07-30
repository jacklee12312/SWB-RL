# -*- coding: utf-8 -*-
"""Exact Ward, Marine, crest, discard, mode, and evolution listener batch."""

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
from swb.engine.events import EventType
from swb.engine.state import Amulet, Unit
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _destroy_units,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


BATCH_CARD_IDS = (
    10162110,
    10361110,
    10362110,
    10563110,
    10663110,
    10142110,
    10241110,
    10342120,
    10542120,
    10424110,
    10512110,
    10613110,
)

SOURCE_HASHES = {
    10162110: "7a6733fb672789a2a951c91c32d6c84bd7859279c4a5a4f244443765a725ab39",
    10361110: "fc17baeca1fdde1daec8216de05c4cee04a7d7d039b22d9c42c137805eb4af3e",
    10362110: "e810049efc9ce6b4561d6610c850f53edc57cdb377a1d0d68122794485ab79ea",
    10563110: "ad24a7a76907912e0c0569dde751ddbe9d6769ce770fc930251e6cc96a5f21b0",
    10663110: "668516d68ed83813701f2d1b618688823e5bc7ed320021083742dc0dcaf7f1c6",
    10142110: "060ea15ef33e955cd724e8be6ec13acdd824627ccbb960568929040e59c4765f",
    10241110: "4fe43356c001de4558ee50469cd6a54807e9b06ce76f284b62268679bfe9dfa8",
    10342120: "563d66ffcfa95577cbcddededf12646712cdd8a51f2ad478032d1bf00ebc18f7",
    10542120: "e02cf844274bab699c15eb368d587efec1aeeb724ce3571278bb31daa888317f",
    10424110: "e20e9f68142cacea722c3da69adaa7e20e4bd2741dd8188368882c9b3d21a824",
    10512110: "c1e3d85d4d89c9c14251c7613d7b9afb5b7860733f70ed3801f6902ffce006ff",
    10613110: "96377e4f366b77e7a6bd820c7e34e62761dd7b25f44f25aa9c2b3665782e4f3b",
}

TEST_EVIDENCE = "tests/test_real_ward_marine_crest_listener_batch.py"


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


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _start_operation(engine, operation: EffectOperation) -> None:
    engine._start_effects(
        _card(989700, name="测试效果来源", card_type="法术", attack=None, life=None),
        None,
        (operation,),
        controller=0,
        label="批次测试效果",
    )


class WardMarineCrestListenerBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 8601):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_modes_emblems_intrinsics_and_leader_heal_listener(self):
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10663110),
            ("必杀", "灵气"),
        )
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10424110),
            ("突进",),
        )
        self.assertEqual(
            [mode.mode_id for mode in self.rulebook.modes_for(10663110)],
            ["crystallize_1"],
        )
        self.assertEqual(
            [mode.mode_id for mode in self.rulebook.modes_for(10424110)],
            ["enhance_6"],
        )
        self.assertEqual(self.rulebook.emblem_def("restful_affirmer").countdown, 4)
        self.assertEqual(self.rulebook.emblem_def("restful_prayer").countdown, 4)
        heal_listener = self.rulebook.listeners_for(10563110)[0]
        self.assertIs(heal_listener.event, EventType.LEADER_HEALED)
        self.assertEqual(heal_listener.turn_scope.value, "owner_turn")
        ward_listener = self.rulebook.listeners_for(10162110)[0]
        self.assertEqual(ward_listener.event_filter.keyword, "守护")
        self.assertEqual(ward_listener.source_relation.value, "other")
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

    def test_salissa_uses_destroyed_ward_snapshot_and_evolve_grants_barrier(self):
        engine = self.fresh(seed=3)
        source = _play(engine, self.repository, 10162110)
        ward = _put_unit(
            engine,
            0,
            _card(986001, keywords=frozenset({"守护"}), attack=2, life=2),
        )
        nonward = _put_unit(engine, 0, _card(986002, attack=2, life=2))
        before = (source.attack, source.max_health)

        _destroy_units(engine, ward, nonward)
        self.assertEqual((source.attack, source.max_health), (before[0] + 1, before[1] + 1))

        enemy_ward = _put_unit(
            engine,
            1,
            _card(986003, keywords=frozenset({"守护"}), attack=2, life=2),
        )
        after_own_death = (source.attack, source.max_health)
        _destroy_units(engine, enemy_ward)
        self.assertEqual((source.attack, source.max_health), after_own_death)

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertTrue(source.has_keyword("屏障"))

    def test_repose_crests_share_attack_history_and_bind_one_random_target(self):
        affirmer = self.fresh(seed=5)
        source = _play(affirmer, self.repository, 10361110)
        ally = _put_unit(affirmer, 0, _card(986010, attack=5, life=5))
        before = {unit.entity_id: unit.attack for unit in (source, ally)}
        affirmer.apply(EndTurn(0))
        warded = [unit for unit in (source, ally) if unit.has_keyword("守护")]
        self.assertEqual(len(warded), 1)
        self.assertEqual(warded[0].attack, max(0, before[warded[0].entity_id] - 2))

        prayer = self.fresh(seed=7)
        prayer.players[0].health = 12
        supplicant = _play(prayer, self.repository, 10362110)
        self.assertTrue(supplicant.has_keyword("守护"))
        prayer.apply(EndTurn(0))
        self.assertEqual(prayer.players[0].health, 13)

    def test_repose_crests_skip_after_an_attack_and_are_seeded(self):
        attacked = self.fresh(seed=11)
        source = _play(attacked, self.repository, 10361110)
        defender = _put_unit(attacked, 1, _card(986020, attack=0, life=20))
        source.summoned_this_turn = False
        source.can_attack = True
        before = source.attack
        attacked.apply(Attack(0, source.entity_id, defender.entity_id))
        attacked.apply(EndTurn(0))
        self.assertFalse(source.has_keyword("守护"))
        self.assertEqual(source.attack, before)

        chosen_ids = []
        for _ in range(2):
            engine = self.fresh(seed=13)
            crest_source = _play(engine, self.repository, 10361110)
            allies = [
                _put_unit(engine, 0, _card(986030 + index, attack=5, life=5))
                for index in range(2)
            ]
            engine.apply(EndTurn(0))
            chosen_ids.append(
                next(
                    unit.definition.card_id
                    for unit in (crest_source, *allies)
                    if unit.has_keyword("守护")
                )
            )
        self.assertEqual(chosen_ids[0], chosen_ids[1])

        empty = self.fresh(seed=17)
        crest_source = _play(empty, self.repository, 10361110)
        _destroy_units(empty, crest_source)
        empty.apply(EndTurn(0))
        self.assertIsNone(empty.state.pending_choice)

    def test_holy_dignity_heals_then_summons_only_on_actual_owner_turn_healing(self):
        engine = self.fresh(seed=19)
        engine.players[0].health = 10
        source = _play(engine, self.repository, 10563110)
        self.assertEqual(engine.players[0].health, 11)
        self.assertTrue(source.has_keyword("守护"))
        self.assertEqual(
            sum(entity.definition.card_id == 10061120 for entity in engine.players[0].board),
            1,
        )

        full_health = self.fresh(seed=23)
        full_health.players[0].health = full_health.players[0].max_health
        _play(full_health, self.repository, 10563110)
        self.assertFalse(any(
            entity.definition.card_id == 10061120
            for entity in full_health.players[0].board
        ))

        opponent_turn = self.fresh(seed=29)
        opponent_turn.players[0].health = opponent_turn.players[0].max_health
        _play(opponent_turn, self.repository, 10563110)
        opponent_turn.apply(EndTurn(0))
        opponent_turn.players[0].health = 10
        _start_operation(
            opponent_turn,
            EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 1),
        )
        self.assertEqual(opponent_turn.players[0].health, 11)
        self.assertFalse(any(
            entity.definition.card_id == 10061120
            for entity in opponent_turn.players[0].board
        ))

    def test_holy_dignity_evolve_super_evolve_and_full_board_capacity(self):
        evolved = self.fresh(seed=31)
        evolved.players[0].health = 10
        source = _play(evolved, self.repository, 10563110)
        _enable_evolution(evolved)
        evolved.apply(Evolve(0, source.entity_id))
        self.assertEqual(evolved.players[0].health, 12)
        self.assertEqual(
            sum(entity.definition.card_id == 10061120 for entity in evolved.players[0].board),
            2,
        )

        super_evolved = self.fresh(seed=37)
        super_evolved.players[0].health = 10
        source = _play(super_evolved, self.repository, 10563110)
        _enable_evolution(super_evolved, super_evolve=True)
        super_evolved.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(super_evolved.players[0].health, 13)
        self.assertEqual(
            sum(
                entity.definition.card_id == 10061120
                for entity in super_evolved.players[0].board
            ),
            3,
        )

        full_board = self.fresh(seed=41)
        full_board.players[0].health = 10
        for index in range(4):
            _put_unit(full_board, 0, _card(986100 + index))
        _play(full_board, self.repository, 10563110)
        self.assertEqual(full_board.players[0].health, 11)
        self.assertEqual(len(full_board.players[0].board), 5)
        self.assertFalse(any(
            entity.definition.card_id == 10061120
            for entity in full_board.players[0].board
        ))

    def test_worshipful_crusader_normal_evolve_stale_target_and_crystallize(self):
        normal = self.fresh(seed=43)
        source = _play(normal, self.repository, 10663110)
        crusaders = [
            entity
            for entity in normal.players[0].board
            if entity.definition.card_id == 10663110
        ]
        self.assertEqual(len(crusaders), 2)
        self.assertTrue(all(unit.has_keyword("必杀") and unit.has_keyword("灵气") for unit in crusaders))
        _destroy_units(normal, source)
        self.assertEqual(
            sum(entity.definition.card_id == 10663110 for entity in normal.players[0].board),
            1,
        )

        stale = self.fresh(seed=47)
        source = _play(stale, self.repository, 10663110)
        enemy = _put_unit(stale, 1, _card(986120, life=9))
        _enable_evolution(stale)
        stale.apply(Evolve(0, source.entity_id))
        stale.players[1].board.remove(enemy)
        _choose_entity(stale, enemy.entity_id)
        self.assertIsNone(stale.state.pending_choice)

        crystal = self.fresh(seed=53)
        crystal.players[0].mana = 1
        amulet = _play(crystal, self.repository, 10663110, mode_id="crystallize_1")
        self.assertIsInstance(amulet, Amulet)
        self.assertEqual(amulet.countdown, 3)
        amulet.countdown = 1
        while amulet in crystal.players[0].board:
            crystal.apply(EndTurn(crystal.current_player))
        summoned = [
            entity
            for entity in crystal.players[0].board
            if entity.definition.card_id == 10663110
        ]
        self.assertEqual(len(summoned), 1)
        self.assertTrue(summoned[0].has_keyword("必杀"))
        self.assertTrue(summoned[0].has_keyword("灵气"))

    def test_worshipful_crusader_capacity_and_no_evolve_target_are_safe(self):
        shortage = self.fresh(seed=59)
        for index in range(4):
            _put_unit(shortage, 0, _card(986130 + index))
        source = _play(shortage, self.repository, 10663110)
        self.assertEqual(len(shortage.players[0].board), 5)
        self.assertEqual(
            sum(entity.definition.card_id == 10663110 for entity in shortage.players[0].board),
            1,
        )
        _enable_evolution(shortage)
        shortage.apply(Evolve(0, source.entity_id))
        self.assertIsNone(shortage.state.pending_choice)
        self.assertTrue(source.evolved)

    def test_discarded_kyd_buffs_one_seeded_follower_and_empty_board_is_safe(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=61)
            allies = [
                _put_unit(engine, 0, _card(986150 + index, attack=2, life=4))
                for index in range(2)
            ]
            _put_hand(engine, self.repository.get(10142110))
            _start_operation(
                engine,
                EffectOperation(EffectKind.DISCARD, TargetKind.RANDOM_OWN_HAND),
            )
            outcomes.append(tuple(unit.attack for unit in allies))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(sorted(outcomes[0]), [2, 3])

        empty = self.fresh(seed=67)
        _put_hand(empty, self.repository.get(10142110))
        _start_operation(
            empty,
            EffectOperation(EffectKind.DISCARD, TargetKind.RANDOM_OWN_HAND),
        )
        self.assertEqual(empty.players[0].hand, [])
        self.assertIsNone(empty.state.pending_choice)

        played = self.fresh(seed=71)
        source = _play(played, self.repository, 10142110)
        self.assertTrue(source.has_keyword("突进"))

    def test_sheltering_dragon_hand_listener_reduces_cost_and_fanfare_summons(self):
        engine = self.fresh(seed=73)
        hand_card = _put_hand(engine, self.repository.get(10241110))
        ally = _put_unit(engine, 0, _card(986170, attack=2, life=3))
        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, ally.entity_id))
        self.assertEqual(hand_card.current_cost, 7)

        engine.players[0].mana = 7
        engine.apply(PlayCard(0, engine.players[0].hand.index(hand_card)))
        self.assertTrue(any(
            entity.definition.card_id == 90041120
            for entity in engine.players[0].board
        ))

        shortage = self.fresh(seed=79)
        for index in range(4):
            _put_unit(shortage, 0, _card(986180 + index))
        _play(shortage, self.repository, 10241110)
        self.assertEqual(len(shortage.players[0].board), 5)
        self.assertFalse(any(
            entity.definition.card_id == 90041120
            for entity in shortage.players[0].board
        ))

    def test_ocean_rider_overflow_replacement_marine_ward_and_capacity(self):
        normal = self.fresh(seed=83)
        normal.players[0].max_mana = normal.players[0].mana = 6
        _play(normal, self.repository, 10342120)
        normal_orcas = [
            entity
            for entity in normal.players[0].board
            if entity.definition.card_id == 90041130
        ]
        self.assertEqual(len(normal_orcas), 1)
        self.assertTrue(normal_orcas[0].has_keyword("守护"))

        overflow = self.fresh(seed=89)
        _play(overflow, self.repository, 10342120)
        overflow_orcas = [
            entity
            for entity in overflow.players[0].board
            if entity.definition.card_id == 90041130
        ]
        self.assertEqual(len(overflow_orcas), 2)
        self.assertTrue(all(orca.has_keyword("守护") for orca in overflow_orcas))

        shortage = self.fresh(seed=97)
        for index in range(3):
            _put_unit(shortage, 0, _card(986200 + index))
        _play(shortage, self.repository, 10342120)
        shortage_orcas = [
            entity
            for entity in shortage.players[0].board
            if entity.definition.card_id == 90041130
        ]
        self.assertEqual(len(shortage_orcas), 1)
        self.assertTrue(shortage_orcas[0].has_keyword("守护"))

    def test_jellyfish_dancer_adds_orca_and_reacts_only_to_entering_marines(self):
        engine = self.fresh(seed=101)
        source = _play(engine, self.repository, 10542120)
        self.assertEqual([card.card_id for card in engine.players[0].hand], [90041130])
        self.assertFalse(source.has_keyword("突进"))
        self.assertFalse(source.has_keyword("必杀"))

        _start_operation(
            engine,
            EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90021110),
        )
        self.assertFalse(source.has_keyword("突进"))
        _start_operation(
            engine,
            EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90041130),
        )
        self.assertTrue(source.has_keyword("突进"))
        self.assertTrue(source.has_keyword("必杀"))

        full = self.fresh(seed=103)
        source = _play(full, self.repository, 10542120)
        for index in range(4):
            _put_unit(full, 0, _card(986220 + index))
        _start_operation(
            full,
            EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90041130),
        )
        self.assertFalse(source.has_keyword("突进"))
        self.assertFalse(source.has_keyword("必杀"))

    def test_zeta_and_bea_normal_enhance_output_binding_and_capacity(self):
        normal = self.fresh(seed=107)
        source = _play(normal, self.repository, 10424110)
        partner = next(
            unit
            for unit in normal.players[0].board
            if unit.definition.card_id == 10424110 and unit is not source
        )
        self.assertTrue(source.has_keyword("突进"))
        self.assertTrue(partner.has_keyword("突进"))
        self.assertFalse(source.has_keyword("疾驰"))
        self.assertFalse(partner.has_keyword("必杀"))

        enhanced = self.fresh(seed=109)
        source = _play(enhanced, self.repository, 10424110, mode_id="enhance_6")
        partner = next(
            unit
            for unit in enhanced.players[0].board
            if unit.definition.card_id == 10424110 and unit is not source
        )
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertTrue(partner.has_keyword("必杀"))

        shortage = self.fresh(seed=113)
        for index in range(4):
            _put_unit(shortage, 0, _card(986240 + index))
        source = _play(shortage, self.repository, 10424110, mode_id="enhance_6")
        self.assertEqual(
            sum(entity.definition.card_id == 10424110 for entity in shortage.players[0].board),
            1,
        )
        self.assertTrue(source.has_keyword("疾驰"))

    def test_flowering_friendship_combo_evolves_both_and_handles_full_board(self):
        below = self.fresh(seed=127)
        below.players[0].cards_played_this_turn = 3
        source = _play(below, self.repository, 10512110)
        self.assertFalse(source.evolved)
        self.assertEqual(
            sum(entity.definition.card_id == 10512110 for entity in below.players[0].board),
            1,
        )

        threshold = self.fresh(seed=131)
        threshold.players[0].cards_played_this_turn = 4
        _play(threshold, self.repository, 10512110)
        partners = [
            entity
            for entity in threshold.players[0].board
            if entity.definition.card_id == 10512110
        ]
        self.assertEqual(len(partners), 2)
        self.assertTrue(all(partner.evolved for partner in partners))

        full = self.fresh(seed=137)
        full.players[0].cards_played_this_turn = 4
        for index in range(4):
            _put_unit(full, 0, _card(986260 + index))
        source = _play(full, self.repository, 10512110)
        self.assertEqual(len(full.players[0].board), 5)
        self.assertTrue(source.evolved)

    def test_gracious_attendant_summons_token_and_heals_for_evolve_and_super_evolve(self):
        evolved = self.fresh(seed=139)
        evolved.players[0].health = 10
        source = _play(evolved, self.repository, 10613110)
        self.assertTrue(any(
            entity.definition.card_id == 90011120
            for entity in evolved.players[0].board
        ))
        _enable_evolution(evolved)
        evolved.apply(Evolve(0, source.entity_id))
        self.assertEqual(evolved.players[0].health, 11)

        super_evolved = self.fresh(seed=149)
        super_evolved.players[0].health = 10
        source = _play(super_evolved, self.repository, 10613110)
        _enable_evolution(super_evolved, super_evolve=True)
        super_evolved.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(super_evolved.players[0].health, 11)

        opponent = self.fresh(seed=151)
        opponent.players[0].health = 10
        _play(opponent, self.repository, 10613110)
        opponent.apply(EndTurn(0))
        before_enemy_evolve = opponent.players[0].health
        enemy = _put_unit(opponent, 1, _card(986280))
        opponent.players[1].turns_started = opponent.config.evolution_unlock_turn
        opponent.players[1].evolution_points = 1
        opponent.players[1].evolved_this_turn = False
        opponent.apply(Evolve(1, enemy.entity_id))
        self.assertEqual(opponent.players[0].health, before_enemy_evolve)

    def test_rl_mask_exposes_enhance_crystallize_and_normal_combo_play(self):
        deck = [_card(986300 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=157,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=157)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10

        zeta = _put_hand(env.core, self.repository.get(10424110))
        normal = PlayCard(0, env.players[0].hand.index(zeta))
        enhance = PlayCard(0, env.players[0].hand.index(zeta), "enhance_6")
        env.invalidate_cache(reason="zeta modes")
        self.assertFalse(env.action_mask()[env._encode_command(normal)])
        self.assertTrue(env.action_mask()[env._encode_command(enhance)])

        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].mana = 1
        crusader = _put_hand(env.core, self.repository.get(10663110))
        crystal = PlayCard(0, env.players[0].hand.index(crusader), "crystallize_1")
        env.invalidate_cache(reason="crusader crystallize")
        self.assertTrue(env.action_mask()[env._encode_command(crystal)])

        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].mana = 10
        partner = _put_hand(env.core, self.repository.get(10512110))
        combo_play = PlayCard(0, env.players[0].hand.index(partner))
        env.invalidate_cache(reason="combo play")
        self.assertTrue(env.action_mask()[env._encode_command(combo_play)])


class WardMarineCrestListenerDatabaseAuditTests(unittest.TestCase):
    def test_database_text_modes_and_references_match_reviewed_cards(self):
        expected_phrases = {
            10162110: ("allied follower", "Ward", "destroyed", "Barrier"),
            10361110: ("Crest", "Countdown", "didn't attack", "-2/-0", "Ward"),
            10362110: ("Crest", "Countdown", "didn't attack", "restore 1", "Ward"),
            10563110: ("Restore 1", "whenever your leader", "Fox of Purity", "Super-Evolve"),
            10663110: ("Worshipful Crusader", "Bane", "Aura", "Countdown", "Last Words"),
            10142110: ("discarded", "random allied follower", "+1/+0", "Rush"),
            10241110: ("Activates in hand", "super-evolves", "cost", "Vastwing Dragon"),
            10342120: ("Majestic Megalorca", "Overflow", "Marine", "Ward"),
            10542120: ("Majestic Megalorca", "Marine", "Rush", "Bane"),
            10424110: ("Zeta & Bea", "Enhance", "Bane", "Storm", "Rush"),
            10512110: ("Combo", "Flowering Friendship", "evolve it", "this follower"),
            10613110: ("Springbloom Fairy", "allied follower evolves", "restore 1"),
        }
        expected_references = {
            10162110: [],
            10361110: [],
            10362110: [],
            10563110: [10061120],
            10663110: [10663110],
            10142110: [],
            10241110: [90041120],
            10342120: [90041130],
            10542120: [90041130],
            10424110: [10424110],
            10512110: [10512110],
            10613110: [90011120],
        }
        mode_cards = {10361110, 10362110, 10663110}
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
