# -*- coding: utf-8 -*-
"""Exact Portalcraft dynamic board-count, evolve, and play-mode cards."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_expression
from swb.engine.commands import Attack, PlayCard
from swb.engine.conditions import evaluate_expression
from swb.engine.effects import ExprType, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import IllegalCommand
from swb.engine.state import Amulet
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _choose,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10371110, 10472110, 10573110, 10672110, 10874110)
SOURCE_HASHES = {
    10371110: "055d03c29d5346d13245512afd5408e5f5c0985108ab53c82292fed7aeb96149",
    10472110: "34e9c974b7523801f290b05a3a1f3ab18672cbb88d98f15c1c83c629ba4a397a",
    10573110: "6ab32ddb59d04e66ffdd313921e6606760b924ab5668d5a9868d5768c5623ce8",
    10672110: "6489f6974a77e03047796dee332476faa86f89adb627afa770803d13974279f3",
    10874110: "37457fc0e3985c81b87787d588a56450e9d5023ef92f7ef6e841d26b25245ee4",
}


def _put_amulet(engine, card_id: int = 99001) -> Amulet:
    amulet = Amulet(
        definition=_card(
            card_id,
            card_type="护符",
            attack=None,
            life=None,
        ),
        entity_id=engine.state.allocate_entity_id(),
        entered_turn=engine.turn,
    )
    engine.players[0].board.append(amulet)
    return amulet


def _grant_ward(unit) -> None:
    unit.add_keyword("守护")


class PortalDynamicEvolutionSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")

    def test_board_count_expression_parses_follower_filter(self):
        expression = _parse_expression(
            {
                "type": "controller_board_count",
                "filter": {"card_type": "随从", "evolved": False},
            },
            "test",
            1,
        )

        self.assertIs(expression.type, ExprType.CONTROLLER_BOARD_COUNT)
        self.assertEqual(expression.board_filter.card_type, "随从")
        self.assertFalse(expression.board_filter.evolved)

    def test_board_count_expression_rejects_invalid_filter_shape(self):
        with self.assertRaisesRegex(ValueError, "filter must be an object"):
            _parse_expression(
                {"type": "controller_board_count", "filter": []},
                "test",
                1,
            )

    def test_union_burst_excludes_source_and_clash_binds_attack_target(self):
        burst = self.rulebook.union_bursts_for(10472110)[0]
        guard = burst.operations[0]
        clash = self.rulebook.operations_for(10472110, Trigger.CLASH)[0]

        self.assertTrue(guard.exclude_source)
        self.assertFalse(guard.board_filter.evolved)
        self.assertIs(clash.target, TargetKind.ATTACK_TARGET)


class RealPortalDynamicEvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2201):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_assertion_counts_other_cards_buffs_then_destroys_them_together(self):
        engine = self.fresh(seed=3)
        follower = _put_unit(engine, 0, _card(9101, attack=2, life=3))
        amulet = _put_amulet(engine, 9102)
        source = _play(engine, self.repository, 10371110)
        definition = self.repository.get(10371110)

        self.assertEqual(
            (source.attack, source.health, source.max_health),
            (definition.attack + 2, definition.life + 2, definition.life + 2),
        )
        self.assertTrue(source.has_keyword("守护"))
        self.assertNotIn(follower, engine.players[0].board)
        self.assertNotIn(amulet, engine.players[0].board)
        self.assertEqual(engine.players[0].board, [source])

        empty = self.fresh(seed=5)
        only = _play(empty, self.repository, 10371110)
        self.assertEqual((only.attack, only.max_health), (definition.attack, definition.life))

    def test_neural_blocker_counts_only_other_followers_and_last_words_draws_two(self):
        engine = self.fresh(seed=7)
        _put_unit(engine, 0, _card(9201))
        _put_unit(engine, 0, _card(9202))
        _put_amulet(engine, 9203)
        source = _play(engine, self.repository, 10573110)

        self.assertEqual(engine.players[0].mana, 7)
        self.assertTrue(source.has_keyword("突进"))

        draw_engine = self.fresh(seed=11)
        draw_engine.players[0].deck = [_card(9210), _card(9211), _card(9212)]
        blocker = _play(draw_engine, self.repository, 10573110)
        enemy = _put_unit(draw_engine, 1, _card(9213, attack=5, life=10))
        draw_engine.apply(Attack(0, blocker.entity_id, enemy.entity_id))
        self.assertNotIn(blocker, draw_engine.players[0].board)
        self.assertEqual(len(draw_engine.players[0].hand), 2)
        self.assertEqual(len(draw_engine.players[0].deck), 1)

    def test_eustace_union_burst_requires_another_unevolved_follower(self):
        below = self.fresh(seed=13)
        _put_unit(below, 0, _card(9301))
        below.players[0].turns_started = 9
        source = _play(below, self.repository, 10472110)
        self.assertIsNone(below.state.pending_choice)
        self.assertFalse(source.evolved)

        no_target = self.fresh(seed=17)
        no_target.players[0].turns_started = 10
        source = _play(no_target, self.repository, 10472110)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertFalse(source.evolved)

        active = self.fresh(seed=19)
        ally = _put_unit(active, 0, _card(9302))
        evolved = _put_unit(active, 0, _card(9303))
        evolved.evolved = True
        active.players[0].turns_started = 10
        ep_before = active.players[0].evolution_points
        source = _play(active, self.repository, 10472110)
        self.assertEqual(
            [option.entity_id for option in active.state.pending_choice.options],
            [ally.entity_id],
        )
        _choose(active, ally.entity_id)
        self.assertTrue(ally.evolved)
        self.assertTrue(source.evolved)
        self.assertEqual(active.players[0].evolution_points, ep_before)

    def test_eustace_clash_damages_the_opposing_follower_for_both_roles(self):
        attacker_game = self.fresh(seed=23)
        source = _put_unit(attacker_game, 0, self.repository.get(10472110))
        defender = _put_unit(attacker_game, 1, _card(9401, attack=0, life=12))
        source.summoned_this_turn = False
        source.can_attack = True
        attacker_game.apply(Attack(0, source.entity_id, defender.entity_id))
        self.assertEqual(defender.health, 4)

        defender_game = self.fresh(seed=29)
        attacker = _put_unit(defender_game, 0, _card(9402, attack=1, life=3))
        eustace = _put_unit(defender_game, 1, self.repository.get(10472110))
        attacker.summoned_this_turn = False
        attacker.can_attack = True
        defender_game.apply(Attack(0, attacker.entity_id, eustace.entity_id))
        self.assertNotIn(attacker, defender_game.players[0].board)
        self.assertEqual(eustace.health, eustace.max_health)

    def test_substandard_puppet_normal_and_accelerate_modes(self):
        normal = self.fresh(seed=31)
        ep_before = normal.players[0].evolution_points
        source = _play(normal, self.repository, 10672110)
        copies = [
            unit
            for unit in normal.players[0].board
            if unit.definition.card_id == 10672110
        ]
        self.assertEqual(len(copies), 2)
        self.assertIn(source, copies)
        self.assertTrue(all(unit.evolved for unit in copies))
        self.assertEqual(normal.players[0].evolution_points, ep_before)

        accelerated = self.fresh(seed=37)
        accelerated.players[0].mana = 3
        _put_hand(accelerated, self.repository.get(10672110))
        commands = accelerated.legal_commands()
        self.assertIn(PlayCard(0, 0, "accelerate_3"), commands)
        self.assertNotIn(PlayCard(0, 0), commands)
        accelerated.apply(PlayCard(0, 0, "accelerate_3"))
        outputs = [
            unit
            for unit in accelerated.players[0].board
            if unit.definition.card_id == 10672110
        ]
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(not unit.evolved for unit in outputs))
        self.assertEqual(accelerated.players[0].graveyard[-1].definition.card_id, 10672110)

        shortage = self.fresh(seed=41)
        for index in range(4):
            _put_unit(shortage, 0, _card(9410 + index))
        shortage.players[0].mana = 3
        _put_hand(shortage, self.repository.get(10672110))
        shortage.apply(PlayCard(0, 0, "accelerate_3"))
        self.assertEqual(len(shortage.players[0].board), 5)
        self.assertEqual(
            sum(unit.definition.card_id == 10672110 for unit in shortage.players[0].board),
            1,
        )

    def test_illegal_special_modes_preserve_full_deterministic_state(self):
        for card_id, mode_id, mana in (
            (10672110, "accelerate_3", 2),
            (10874110, "enhance_9", 8),
        ):
            with self.subTest(card_id=card_id):
                engine = self.fresh(seed=43)
                engine.players[0].mana = mana
                _put_hand(engine, self.repository.get(card_id))
                before = engine.deterministic_fingerprint()
                with self.assertRaises(IllegalCommand):
                    engine.apply(PlayCard(0, 0, mode_id))
                self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_ashray_normal_fanfare_is_optional_when_no_enemy_exists(self):
        no_target = self.fresh(seed=47)
        source = _play(no_target, self.repository, 10874110)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertFalse(source.has_keyword("守护"))
        self.assertFalse(source.has_keyword("疾驰"))

        selected = self.fresh(seed=53)
        enemy = _put_unit(selected, 1, _card(9501))
        source = _play(selected, self.repository, 10874110)
        self.assertFalse(source.has_keyword("守护"))
        self.assertFalse(source.has_keyword("疾驰"))
        _choose(selected, enemy.entity_id)
        self.assertTrue(enemy.has_keyword("守护"))
        self.assertFalse(source.has_keyword("守护"))

    def test_ashray_enhance_evolves_grants_storm_and_destroys_two_random_wards(self):
        engine = self.fresh(seed=59)
        chosen = _put_unit(engine, 1, _card(9601))
        wards = [_put_unit(engine, 1, _card(9602 + index)) for index in range(2)]
        for ward in wards:
            _grant_ward(ward)
        ordinary = _put_unit(engine, 1, _card(9604))
        ep_before = engine.players[0].evolution_points
        source = _play(engine, self.repository, 10874110, mode_id="enhance_9")
        _choose(engine, chosen.entity_id)

        self.assertTrue(source.evolved)
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertFalse(source.has_keyword("守护"))
        self.assertEqual(engine.players[0].evolution_points, ep_before)
        self.assertIn(ordinary, engine.players[1].board)
        remaining_wards = [
            unit for unit in engine.players[1].board if unit.has_keyword("守护")
        ]
        self.assertEqual(len(remaining_wards), 1)
        self.assertEqual(len(engine.players[1].board), 2)

    def test_ashray_random_ward_shortage_and_seeded_replay(self):
        shortage = self.fresh(seed=61)
        only = _put_unit(shortage, 1, _card(9701))
        ordinary = _put_unit(shortage, 1, _card(9702))
        _grant_ward(only)
        source = _put_unit(shortage, 0, self.repository.get(10874110))
        shortage._start_effects(
            source.definition,
            source.entity_id,
            self.rulebook.operations_for(10874110, Trigger.SELF_EVOLVED),
            label="test-evolve",
        )
        self.assertNotIn(only, shortage.players[1].board)
        self.assertIn(ordinary, shortage.players[1].board)

        fingerprints = []
        for _ in range(2):
            replay = self.fresh(seed=67)
            candidates = [_put_unit(replay, 1, _card(9710 + index)) for index in range(3)]
            for candidate in candidates:
                _grant_ward(candidate)
            source = _put_unit(replay, 0, self.repository.get(10874110))
            replay._start_effects(
                source.definition,
                source.entity_id,
                self.rulebook.operations_for(10874110, Trigger.SELF_EVOLVED),
                label="test-evolve",
            )
            fingerprints.append(replay.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_rl_mode_and_target_masks_match_executable_commands(self):
        deck = [_card(9800 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=71,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=71)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10

        env.players[0].mana = 3
        _put_hand(env.core, self.repository.get(10672110))
        mode_action = ShadowverseEnv.MODE_PLAY_OFFSET
        mask = env.action_mask()
        self.assertFalse(mask[ShadowverseEnv.PLAY_OFFSET])
        self.assertTrue(mask[mode_action])
        env.step(mode_action)
        self.assertEqual(len(env.players[0].board), 2)

        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].board.clear()
        env.players[1].board.clear()
        env.players[0].mana = 5
        enemy = _put_unit(env.core, 1, _card(9810))
        _put_hand(env.core, self.repository.get(10874110))
        self.assertTrue(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])
        result = env.step(ShadowverseEnv.PLAY_OFFSET)
        self.assertTrue(result.info["action_mask"][ShadowverseEnv.CHOICE_OFFSET])
        result = env.step(ShadowverseEnv.CHOICE_OFFSET)
        self.assertTrue(enemy.has_keyword("守护"))
        self.assertEqual(
            len(result.observation),
            ShadowverseEnv.OBSERVATION_V1_SIZE,
        )


class PortalDynamicEvolutionAuditTests(unittest.TestCase):
    def test_all_five_cards_are_exact_with_hash_and_direct_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                classification = report["classifications"][str(card_id)]
                audit = classification["clause_audit"]
                self.assertEqual(classification["coverage"], "covered_exact")
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(audit["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(
                    audit["test_evidence"],
                    ["tests/test_real_portal_dynamic_evolution_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
