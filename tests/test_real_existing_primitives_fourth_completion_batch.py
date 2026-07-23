# -*- coding: utf-8 -*-
"""Direct contracts for the fourth existing-primitive exact-card slice."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Attack, BeginFusion, Choose, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import IllegalCommand
from tests.test_real_basic_existing_primitives_batch import _card, _fresh, _put_hand, _put_unit


CARD_IDS = (
    10153130,
    10323110,
    10413110,
    10442120,
    10624110,
    10674110,
    10754120,
    10823110,
)
SOURCE_HASHES = {
    10153130: "b8c48b5ed9bcab4ef98805de8026f323351a7a866405574ed01d03a1db06838d",
    10323110: "0ecefb083487e1acf8109fce9edd17844a0d3b5e21053ae6887faa562a84f0a8",
    10413110: "fcd3347f6100e2d201d0f6f0bddf451ffd86d19e28b908adb72bc08a009a8811",
    10442120: "d03dc68bb0fcd123ee16bbf4b6ee3118a6ca5f7f82a7d250ec0a740278dc5918",
    10624110: "547447a997b2451f233629a81ae28a8ab5dee2deb8404ee2ff420cf04bd4dd0f",
    10674110: "787f4134e5a636574779eb402e45edc2c99e27e3b12e45ebf457038d8b24de58",
    10754120: "bfc25cddfe66b08368c74e5646491c6dc550c53ec98c34cccdee1c48ec65c31e",
    10823110: "ef86c20de8b2577eed1f6c27edbd60817b0eb2ac26fc46ab6d4dd9640bfb0f44",
}


def _play(
    engine,
    repository: CardRepository,
    card_id: int,
    *,
    mode_id: str = "normal",
):
    hand_card = _put_hand(engine, repository.get(card_id))
    engine.apply(
        PlayCard(
            0,
            engine.players[0].hand.index(hand_card),
            mode_id=mode_id,
        )
    )
    return next(
        (
            entity
            for entity in reversed(engine.players[0].board)
            if entity.definition.card_id == card_id
        ),
        None,
    )


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolve(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = 2
    player.evolved_this_turn = False


def _enable_super_evolve(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.first_player_super_evolution_unlock_turn
    player.super_evolution_points = 2
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False


def _effect_summon(engine, source, card_id: int) -> None:
    engine._start_effects(
        source.definition,
        source.entity_id,
        (EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=card_id),),
        controller=0,
    )


class RealExistingPrimitivesFourthCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 7301):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_granted_keywords_bursts_modes_filters_and_strike(self):
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10153130),
            frozenset({"必杀"}),
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10442120),
            frozenset({"疾驰"}),
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10624110),
            frozenset({"必杀", "吸血", "疾驰"}),
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10754120),
            frozenset({"突进", "守护"}),
        )
        self.assertEqual(self.rulebook.intrinsic_keywords_for(10823110), ("突进",))

        bursts = self.rulebook.union_bursts_for(10413110)
        self.assertEqual([definition.threshold for definition in bursts], [10, 15])
        self.assertEqual(
            [definition.operations[0].kind for definition in bursts],
            [EffectKind.EVOLVE_UNIT, EffectKind.DAMAGE_LEADER],
        )
        infinity = self.rulebook.union_bursts_for(10442120)[0]
        self.assertEqual(infinity.threshold, 15)
        self.assertEqual(infinity.operations[0].keyword, "疾驰")

        modes = self.rulebook.modes_for(10624110)
        self.assertEqual([mode.mode_id for mode in modes], ["enhance_7", "enhance_8"])
        self.assertEqual([len(mode.operations) for mode in modes], [2, 4])
        kamishira = self.rulebook.listeners_for(10674110)[0]
        self.assertEqual(kamishira.event_filter.cost_min, 5)
        self.assertEqual(kamishira.source_relation.value, "other")
        maximilian = self.rulebook.listeners_for(10754120)[0]
        self.assertEqual(maximilian.turn_scope.value, "owner_turn")
        self.assertEqual(maximilian.event_filter.tribe_name, "亡者")
        strike = self.rulebook.operations_for(10823110, Trigger.ATTACK)[0]
        self.assertIs(strike.kind, EffectKind.CONDITIONAL)
        self.assertIs(strike.then_operations[0].kind, EffectKind.REPEAT)

    def test_mugan_necromancy_evolves_summons_ghost_and_grants_bane(self):
        engine = self.fresh(seed=11)
        engine.players[0].shadows = 8
        source = _play(engine, self.repository, 10153130)
        ghosts = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90051130
        ]
        self.assertTrue(source.evolved)
        self.assertEqual(engine.players[0].shadows, 0)
        self.assertEqual(len(ghosts), 1)
        self.assertTrue(ghosts[0].has_keyword("疾驰"))
        self.assertTrue(ghosts[0].has_keyword("必杀"))
        self.assertFalse(source.has_keyword("必杀"))

        no_shadows = self.fresh(seed=13)
        source = _play(no_shadows, self.repository, 10153130)
        self.assertFalse(source.evolved)
        self.assertFalse(any(
            unit.definition.card_id == 90051130
            for unit in no_shadows.players[0].board
        ))

        capped = self.fresh(seed=17)
        for index in range(4):
            _put_unit(capped, 0, _card(995000 + index))
        capped.players[0].shadows = 8
        source = _play(capped, self.repository, 10153130)
        self.assertTrue(source.evolved)
        self.assertEqual(len(capped.players[0].board), capped.config.max_board)
        self.assertFalse(any(
            unit.definition.card_id == 90051130
            for unit in capped.players[0].board
        ))
        capped.assert_invariants()

    def test_congregant_adds_loot_and_play_or_fusion_listener_deals_seeded_damage(self):
        played = self.fresh(seed=19)
        enemies = [_put_unit(played, 1, _card(995020 + index, life=10)) for index in range(2)]
        source = _play(played, self.repository, 10323110)
        self.assertEqual(
            [card.card_id for card in played.players[0].hand],
            [90021320, 90021330],
        )
        goblet = next(card for card in played.players[0].hand if card.card_id == 90021320)
        played.apply(PlayCard(0, played.players[0].hand.index(goblet)))
        self.assertEqual(sorted(unit.health for unit in enemies), [7, 10])

        fused = self.fresh(seed=23)
        source = _play(fused, self.repository, 10323110)
        fused.players[0].hand.clear()
        fused.players[0].hand_entity_ids.clear()
        destination = _put_hand(fused, self.repository.get(10323310))
        material = _put_hand(fused, self.repository.get(90021330))
        enemy = _put_unit(fused, 1, _card(995030, life=10))
        fused.apply(BeginFusion(0, destination.entity_id))
        fused.apply(Choose(0, f"hand:{material.entity_id}"))
        fused.apply(Choose(0, "fusion:confirm"))
        self.assertEqual(enemy.health, 7)
        self.assertIn(source, fused.players[0].board)

        mana = self.fresh(seed=29)
        source = _play(mana, self.repository, 10323110)
        mana.players[0].mana = 5
        _enable_evolve(mana)
        mana.apply(Evolve(0, source.entity_id))
        self.assertEqual(mana.players[0].mana, 6)

    def test_congregant_hand_capacity_and_listener_source_lifecycle_are_safe(self):
        capped = self.fresh(seed=31)
        for index in range(capped.config.max_hand - 1):
            _put_hand(capped, _card(995040 + index))
        _play(capped, self.repository, 10323110)
        loot = [card.card_id for card in capped.players[0].hand if card.card_id >= 90000000]
        self.assertEqual(loot, [90021320])
        self.assertEqual(len(capped.players[0].hand), capped.config.max_hand)

        gone = self.fresh(seed=37)
        source = _play(gone, self.repository, 10323110)
        gone.players[0].board.remove(source)
        gone._send_to_graveyard(
            0,
            source.definition,
            "test_source_left_play",
            source_entity_id=source.entity_id,
        )
        enemy = _put_unit(gone, 1, _card(995060, life=10))
        goblet = next(card for card in gone.players[0].hand if card.card_id == 90021320)
        gone.apply(PlayCard(0, gone.players[0].hand.index(goblet)))
        self.assertEqual(enemy.health, 10)
        gone.assert_invariants()

    def test_cupidain_thresholds_repeat_no_target_and_seeded_randomness(self):
        below = self.fresh(seed=41)
        below.players[0].turns_started = 9
        source = _play(below, self.repository, 10413110)
        self.assertFalse(source.evolved)

        exact = self.fresh(seed=43)
        exact.players[0].turns_started = 10
        enemy = _put_unit(exact, 1, _card(995070, life=20))
        source = _play(exact, self.repository, 10413110)
        self.assertTrue(source.evolved)
        self.assertEqual(enemy.health, 13)
        self.assertEqual(exact.players[1].health, 20)

        super_art = self.fresh(seed=47)
        super_art.players[0].turns_started = 15
        source = _play(super_art, self.repository, 10413110)
        self.assertTrue(source.evolved)
        self.assertEqual(super_art.players[1].health, 17)
        self.assertIsNone(super_art.state.pending_choice)

        outcomes = []
        for _ in range(2):
            seeded = self.fresh(seed=53)
            seeded.players[0].turns_started = 10
            enemies = [_put_unit(seeded, 1, _card(995080 + index, life=10)) for index in range(3)]
            _play(seeded, self.repository, 10413110)
            outcomes.append((
                [unit.health for unit in enemies],
                seeded.deterministic_fingerprint(),
            ))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_infinity_burst_and_super_evolve_multi_target_revalidate_atomically(self):
        below = self.fresh(seed=59)
        below.players[0].turns_started = 14
        source = _play(below, self.repository, 10442120)
        self.assertFalse(source.has_keyword("疾驰"))

        active = self.fresh(seed=61)
        active.players[0].turns_started = 15
        targets = [_put_unit(active, 1, _card(995100 + index)) for index in range(2)]
        source = _play(active, self.repository, 10442120)
        self.assertTrue(source.has_keyword("疾驰"))
        _enable_super_evolve(active)
        active.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(active.state.pending_choice.target_count, 2)
        first = Choose(0, f"entity:{targets[0].entity_id}")
        active.apply(first)
        before = active.deterministic_fingerprint()
        with self.assertRaisesRegex(IllegalCommand, "Choice option is invalid"):
            active.apply(first)
        self.assertEqual(active.deterministic_fingerprint(), before)
        active.players[1].board.remove(targets[0])
        active._send_to_graveyard(
            1,
            targets[0].definition,
            "test_target_left_play",
            source_entity_id=targets[0].entity_id,
        )
        _choose_entity(active, targets[1].entity_id)
        self.assertNotIn(targets[1], active.players[1].board)
        active.assert_invariants()

        shortage = self.fresh(seed=67)
        only = _put_unit(shortage, 1, _card(995110))
        source = _play(shortage, self.repository, 10442120)
        _enable_super_evolve(shortage)
        shortage.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(shortage.state.pending_choice.target_count, 1)
        _choose_entity(shortage, only.entity_id)
        self.assertNotIn(only, shortage.players[1].board)

    def test_ladranoel_enhance_levels_bind_each_soldier_and_super_buffs_others(self):
        expected = {
            "normal": ({"必杀": 1, "吸血": 0, "疾驰": 0}, 1),
            "enhance_7": ({"必杀": 1, "吸血": 1, "疾驰": 0}, 2),
            "enhance_8": ({"必杀": 1, "吸血": 1, "疾驰": 1}, 3),
        }
        for mode_id, (keyword_counts, soldier_count) in expected.items():
            with self.subTest(mode_id=mode_id):
                engine = self.fresh(seed=71)
                source = _play(engine, self.repository, 10624110, mode_id=mode_id)
                soldiers = [
                    unit for unit in engine.players[0].board
                    if unit.definition.card_id == 10621110
                ]
                self.assertEqual(len(soldiers), soldier_count)
                for keyword, count in keyword_counts.items():
                    self.assertEqual(sum(unit.has_keyword(keyword) for unit in soldiers), count)
                    self.assertFalse(source.has_keyword(keyword))

        capped = self.fresh(seed=73)
        for index in range(2):
            _put_unit(capped, 0, _card(995120 + index))
        _play(capped, self.repository, 10624110, mode_id="enhance_8")
        soldiers = [
            unit for unit in capped.players[0].board
            if unit.definition.card_id == 10621110
        ]
        self.assertEqual(len(soldiers), 2)
        self.assertEqual(sum(unit.has_keyword("疾驰") for unit in soldiers), 0)

        buffed = self.fresh(seed=79)
        ally = _put_unit(buffed, 0, _card(995130, attack=2, life=3))
        source = _play(buffed, self.repository, 10624110)
        soldier = next(unit for unit in buffed.players[0].board if unit.definition.card_id == 10621110)
        before = ((ally.attack, ally.max_health), (soldier.attack, soldier.max_health), source.attack)
        _enable_super_evolve(buffed)
        buffed.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual((ally.attack, ally.max_health), (before[0][0] + 1, before[0][1] + 1))
        self.assertEqual((soldier.attack, soldier.max_health), (before[1][0] + 1, before[1][1] + 1))
        self.assertGreater(source.attack, before[2])
        self.assertEqual(source.attack, before[2] + 3)

    def test_kamishira_evolves_other_base_cost_five_followers_and_counts_board(self):
        engine = self.fresh(seed=83)
        source = _play(engine, self.repository, 10674110)
        tokens = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id in {10671110, 10672110}
        ]
        self.assertEqual([unit.definition.card_id for unit in tokens], [10671110, 10672110])
        self.assertTrue(all(unit.evolved for unit in tokens))
        self.assertFalse(source.evolved)
        _effect_summon(engine, source, 90021110)
        knight = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90021110)
        self.assertFalse(knight.evolved)
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(engine.players[1].health, 17)

        capped = self.fresh(seed=89)
        for index in range(3):
            _put_unit(capped, 0, _card(995140 + index))
        source = _play(capped, self.repository, 10674110)
        tokens = [
            unit for unit in capped.players[0].board
            if unit.definition.card_id in {10671110, 10672110}
        ]
        self.assertEqual([unit.definition.card_id for unit in tokens], [10671110])
        self.assertTrue(tokens[0].evolved)

    def test_maximilian_necromancy_listener_turn_scope_capacity_and_source_lifecycle(self):
        engine = self.fresh(seed=97)
        engine.players[0].shadows = 10
        source = _play(engine, self.repository, 10754120)
        zombies = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90051140
        ]
        self.assertEqual(len(zombies), 3)
        self.assertEqual(engine.players[0].shadows, 0)
        self.assertEqual(engine.players[1].health, 17)
        self.assertTrue(all(unit.attack == unit.definition.attack + 1 for unit in zombies))
        self.assertTrue(all(unit.has_keyword("突进") and unit.has_keyword("守护") for unit in zombies))
        self.assertFalse(source.has_keyword("突进"))
        self.assertFalse(source.has_keyword("守护"))

        opponent_turn = self.fresh(seed=101)
        source = _play(opponent_turn, self.repository, 10754120)
        opponent_turn.state.active_player = 1
        _effect_summon(opponent_turn, source, 90051140)
        zombie = next(unit for unit in opponent_turn.players[0].board if unit.definition.card_id == 90051140)
        self.assertEqual(zombie.attack, zombie.definition.attack)
        self.assertFalse(zombie.has_keyword("突进"))
        self.assertEqual(opponent_turn.players[1].health, 20)

        gone = self.fresh(seed=103)
        source = _play(gone, self.repository, 10754120)
        gone.players[0].board.remove(source)
        gone._send_to_graveyard(
            0,
            source.definition,
            "test_source_left_play",
            source_entity_id=source.entity_id,
        )
        _effect_summon(gone, source, 90051140)
        zombie = next(unit for unit in gone.players[0].board if unit.definition.card_id == 90051140)
        self.assertEqual(zombie.attack, zombie.definition.attack)
        self.assertEqual(gone.players[1].health, 20)
        gone.assert_invariants()

    def test_okita_unlock_and_follower_strike_repeat_but_leader_attack_skips_it(self):
        normal = self.fresh(seed=107)
        normal.players[0].turns_started = normal.config.first_player_super_evolution_unlock_turn - 1
        source = _play(normal, self.repository, 10823110)
        self.assertFalse(source.evolved)
        self.assertTrue(source.has_keyword("突进"))
        target = _put_unit(normal, 1, _card(995160, attack=0, life=20))
        source.can_attack = True
        source.attacks_remaining = 1
        normal.apply(Attack(0, source.entity_id, target.entity_id))
        self.assertEqual(target.health, 20 - 3 - source.attack)

        evolved = self.fresh(seed=109)
        evolved.players[0].turns_started = evolved.config.first_player_super_evolution_unlock_turn
        source = _play(evolved, self.repository, 10823110)
        self.assertTrue(source.evolved)
        target = _put_unit(evolved, 1, _card(995161, attack=0, life=20))
        source.can_attack = True
        source.attacks_remaining = 1
        evolved.apply(Attack(0, source.entity_id, target.entity_id))
        self.assertEqual(target.health, 20 - 9 - source.attack)

        leader = self.fresh(seed=113)
        leader.players[0].turns_started = leader.config.first_player_super_evolution_unlock_turn
        source = _play(leader, self.repository, 10823110)
        source.can_attack = True
        source.attacks_remaining = 1
        source.rush_only = False
        before = leader.players[1].health
        leader.apply(Attack(0, source.entity_id, None))
        self.assertEqual(leader.players[1].health, before - source.attack)
        leader.assert_invariants()

    def test_action_masks_match_enhance_commands_and_multi_target_choices(self):
        deck = [_card(996000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=127,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=127)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        _put_hand(env.core, self.repository.get(10624110))
        legal = set(env.core.legal_commands())
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, legal)
        for mode_id in ("normal", "enhance_7", "enhance_8"):
            command = PlayCard(0, 0, mode_id=mode_id)
            self.assertIn(command, legal)
            self.assertTrue(env.action_mask()[env._encode_command(command)])

        target_env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=131,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        target_env.reset(seed=131)
        for player in target_env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        target_env.players[0].mana = target_env.players[0].max_mana = 10
        source = _put_unit(target_env.core, 0, self.repository.get(10442120))
        targets = [_put_unit(target_env.core, 1, _card(996100 + index)) for index in range(2)]
        _enable_super_evolve(target_env.core)
        command = SuperEvolve(0, source.entity_id)
        self.assertTrue(target_env.action_mask()[target_env._encode_command(command)])
        target_env.core.apply(command)
        choices = [index for index, allowed in enumerate(target_env.action_mask()) if allowed]
        self.assertEqual(
            {target_env._decode_action(index).option_id for index in choices},
            {f"entity:{target.entity_id}" for target in targets},
        )


class ExistingPrimitivesFourthCompletionAuditTests(unittest.TestCase):
    def test_database_multilingual_text_references_and_embedded_modes_are_reviewed(self):
        expected_phrases = {
            10153130: ("Necromancy", "Departed follower", "Ghost"),
            10323110: ("Gilded Goblet", "Fuse", "Recover 1 play point"),
            10413110: ("Skybound Art", "7 times", "random enemy follower"),
            10442120: ("Super Skybound Art", "Storm", "Select 2 enemy followers"),
            10624110: ("Fearless Soldier", "Enhance", "all other allied followers"),
            10674110: ("Shoddy Plaything", "base cost of 5", "number of allied followers"),
            10754120: ("Necromancy", "Departed follower", "enemy leader"),
            10823110: ("unlocked super-evolution", "Follower", "3 times instead"),
        }
        expected_references = {
            10153130: (90051130,),
            10323110: (90021320, 90021330),
            10413110: (),
            10442120: (),
            10624110: (10621110,),
            10674110: (10671110, 10672110),
            10754120: (90051140,),
            10823110: (),
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id, phrases in expected_phrases.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        "SELECT text_chs, text_cht, text_eng, text_jpn, text_kor FROM skill_texts WHERE card_id=?",
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
                            "SELECT referenced_card_id FROM card_references WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    )
                    self.assertEqual(references, expected_references[card_id])
                    self.assertFalse(connection.execute(
                        "SELECT 1 FROM alt_modes WHERE card_id=?",
                        (card_id,),
                    ).fetchall())

    def test_all_eight_cards_have_exact_clause_evidence_and_token_audit_stays_complete(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        coverage_counts = report["summary"]["coverage_counts"]
        clause_counts = report["summary"]["clause_audit_counts"]
        self.assertEqual(coverage_counts["covered_exact"], clause_counts["mapped_exact"])
        self.assertEqual(coverage_counts["supported_missing_rule"], clause_counts["missing_rule"])
        self.assertEqual(sum(coverage_counts.values()), report["summary"]["total_cards"])
        self.assertFalse(report["rule_consistency_issues"])
        self.assertFalse(report["clause_audit_issues"])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(info["clause_audit"]["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_existing_primitives_fourth_completion_batch.py"],
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
