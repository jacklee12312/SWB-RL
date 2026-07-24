# -*- coding: utf-8 -*-
"""Direct contracts for the ninth exact existing-primitive rule slice."""

from __future__ import annotations

import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import (
    _build_coverage_report,
    _load_source_text_map,
    _source_text_sha256,
)
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import (
    Attack,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
    SuperEvolve,
)
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from swb.engine.state import HandCard
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10131110,
    10144110,
    10254120,
    10313110,
    10434110,
    10532110,
    10634110,
    10862110,
    10871110,
)
SOURCE_HASHES = {
    10131110: "612855ae9602ad0f8ea1f3f833426fa459fba180c65bbd3b44ced596efb7eb76",
    10144110: "4f7fb512146a0029ee084ffe179faf2c1ac9a303299a2b6c92e2774ec129f77e",
    10254120: "1f92fd41a9347c1b437c8e8d23d373e8e5d5166b0ba49bd90f35259a253e767d",
    10313110: "1fa2b6be56b29648e0e01950847f3296c996ae3a9406f6ef9ac95af50c627902",
    10434110: "d9b2eff4d8c499acde0ddb7ce8eedeac834150e91cc345e1939a747383dac8c4",
    10532110: "f2d8219ff2fe110b45deb3b42f4d356e008cdc56543c80097ffe7b4b14c5a101",
    10634110: "b6301b0e837e0d1bb8dbfd1602040b5d4cf6ce4fd6b311a8e3fae6e03bf47b78",
    10862110: "3b33d0f47acbb79f803df7c5bcaa109e24e3c303ee32204a791112ec724bb7e6",
    10871110: "97bfb3aac677317ed1d67edb876dd63e96bedfca261060ab4abd0ce9f9439e1f",
}
TEST_EVIDENCE = "tests/test_real_existing_primitives_ninth_batch.py"
RULE_FILE = "real_existing_primitives_ninth_batch.json"


def _enable_evolve(engine, *, owner: int = 0) -> None:
    player = engine.players[owner]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False
    engine.state.active_player = owner


def _enable_super_evolve(engine, *, owner: int = 0) -> None:
    player = engine.players[owner]
    player.turns_started = (
        engine.config.first_player_super_evolution_unlock_turn
        if owner == 0
        else engine.config.second_player_super_evolution_unlock_turn
    )
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.super_evolved_this_turn = False
    player.evolved_this_turn = False
    engine.state.active_player = owner


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _choose_mode(engine, mode_id: str) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options
        if option.option_id
        in {mode_id, f"mode:{mode_id}", f"choose_one:{mode_id}"}
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _put_rule_hand(engine, definition, *, owner: int = 0) -> HandCard:
    return _put_hand(engine, definition, owner=owner)


def _kill(engine, *units) -> None:
    for unit in units:
        unit.health = 0
    engine._stabilize()


class ExistingPrimitivesNinthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 9101):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_contextual_generic_schema_contracts(self):
        congregant = self.rulebook.listeners_for(10313110)[0]
        self.assertEqual(len(congregant.operations), 1)
        self.assertIs(
            congregant.operations[0].kind,
            EffectKind.SUMMON_EXACT_COPY,
        )
        self.assertEqual(
            (
                congregant.operations[0].amount,
                congregant.operations[0].secondary_amount,
            ),
            (0, -1),
        )
        for card_id in (10862110, 10871110):
            self.assertIs(
                self.rulebook.operations_for(
                    card_id,
                    Trigger.LAST_WORDS,
                )[1].kind,
                EffectKind.REMOVE_LAST_WORDS,
            )
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10862110),
            ("守护", "灵气"),
        )
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10634110),
            ("吸血",),
        )

        with self.assertRaisesRegex(ValueError, "only valid inside a hand listener"):
            _parse_operation(
                {
                    "kind": "buff_hand_card",
                    "target": "self",
                    "amount": 1,
                    "secondary_amount": 1,
                },
                "test.json",
                1,
            )
        hand_self = _parse_operation(
            {
                "kind": "buff_hand_card",
                "target": "self",
                "amount": 1,
                "secondary_amount": 1,
            },
            "test.json",
            1,
            _allow_hand_self=True,
        )
        self.assertIs(hand_self.target, TargetKind.SELF)
        with self.assertRaisesRegex(ValueError, "requires a follower target"):
            _parse_operation(
                {"kind": "remove_last_words", "target": "own_leader"},
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "requires a follower"):
            _parse_operation(
                {"kind": "summon_exact_copy", "target": "enemy_leader"},
                "test.json",
                1,
            )

    def test_spellboost_hand_listeners_buff_rune_blade_and_vam_then_damage(self):
        engine = self.fresh(seed=3)
        rune = _put_rule_hand(engine, self.repository.get(10131110))
        vam = _put_rule_hand(engine, self.repository.get(10434110))

        _play(engine, self.repository, 10434110)

        self.assertEqual((rune.attack, rune.life), (2, 2))
        self.assertEqual((vam.attack, vam.life), (1, 2))
        enemy = _put_unit(engine, 1, _card(991001, attack=0, life=5))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, engine.players[0].hand.index(rune)))
        self.assertIsNotNone(engine.state.pending_choice)
        _choose_entity(engine, enemy.entity_id)
        self.assertEqual(enemy.health, 3)

        no_target = self.fresh(seed=5)
        source = _play(no_target, self.repository, 10131110)
        self.assertIsNotNone(source)
        self.assertIsNone(no_target.state.pending_choice)
        no_target.assert_invariants()

    def test_bandnight_discard_binds_physical_cost_and_empty_hand_is_atomic(self):
        engine = self.fresh(seed=7)
        discarded = _put_hand(
            engine,
            _card(
                991010,
                name="六费测试牌",
                cost=6,
                card_type="法术",
                attack=None,
                life=None,
            ),
        )
        enemies = [
            _put_unit(engine, 1, _card(991011 + index, attack=0, life=9))
            for index in range(2)
        ]

        _play(engine, self.repository, 10144110)
        _choose_entity(engine, discarded.entity_id)

        self.assertNotIn(discarded, engine.players[0].hand)
        self.assertEqual([unit.health for unit in enemies], [3, 3])

        empty = self.fresh(seed=11)
        bandnight = _put_hand(empty, self.repository.get(10144110))
        command = PlayCard(0, empty.players[0].hand.index(bandnight))
        self.assertIn(command, empty.legal_commands())
        empty.apply(command)
        self.assertIsNone(empty.state.pending_choice)
        self.assertEqual(len(empty.players[0].board), 1)
        empty.assert_invariants()

    def test_bandnight_stale_discard_skips_bound_damage_safely(self):
        engine = self.fresh(seed=13)
        discarded = _put_hand(engine, _card(991020, cost=8))
        enemy = _put_unit(engine, 1, _card(991021, attack=0, life=12))
        _play(engine, self.repository, 10144110)
        request = engine.state.pending_choice
        option = next(
            option for option in request.options
            if option.entity_id == discarded.entity_id
        )
        engine.players[0].hand.remove(discarded)
        engine.players[0].hand_entity_ids.remove(discarded.entity_id)

        engine.apply(Choose(0, option.option_id))

        self.assertEqual(enemy.health, 12)
        self.assertIsNone(engine.state.pending_choice)
        engine.assert_invariants()

    def test_bandnight_enemy_crest_turn_start_and_once_per_owner_turn_heal(self):
        engine = self.fresh(seed=17)
        discarded = _put_hand(engine, _card(991030, cost=1))
        source = _play(engine, self.repository, 10144110)
        _choose_entity(engine, discarded.entity_id)
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(
            [emblem.emblem_id for emblem in engine.players[1].emblems],
            ["bandnight_anathema_of_flame"],
        )

        engine.players[1].health = 15
        engine.apply(EndTurn(0))
        self.assertEqual(engine.players[1].health, 14)
        heal = (
            EffectOperation(
                EffectKind.HEAL_LEADER,
                TargetKind.OWN_LEADER,
                amount=2,
            ),
        )
        engine._start_effects(
            _card(
                991031,
                name="回复测试",
                card_type="法术",
                attack=None,
                life=None,
            ),
            None,
            heal,
            controller=1,
        )
        self.assertEqual(engine.players[1].health, 15)
        engine._start_effects(
            _card(
                991032,
                name="再次回复测试",
                card_type="法术",
                attack=None,
                life=None,
            ),
            None,
            heal,
            controller=1,
        )
        self.assertEqual(engine.players[1].health, 17)

    def test_charon_reanimates_two_departed_followers_and_grants_ward(self):
        engine = self.fresh(seed=19)
        departed_two = _put_unit(
            engine,
            0,
            _card(
                991040,
                name="二费亡者",
                cost=2,
                tribe_id=20,
                tribe_name="亡者",
            ),
        )
        departed_one = _put_unit(
            engine,
            0,
            _card(
                991041,
                name="一费亡者",
                cost=1,
                tribe_id=20,
                tribe_name="亡者",
            ),
        )
        _kill(engine, departed_two, departed_one)

        _play(engine, self.repository, 10254120)

        reanimated = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id in {991040, 991041}
        ]
        self.assertEqual(
            {unit.definition.card_id for unit in reanimated},
            {991040, 991041},
        )
        self.assertTrue(all(unit.has_keyword("守护") for unit in reanimated))
        engine.assert_invariants()

    def test_charon_crest_countdown_reanimates_three_at_owner_turn_start(self):
        engine = self.fresh(seed=23)
        departed = _put_unit(
            engine,
            0,
            _card(
                991050,
                name="三费亡者",
                cost=3,
                tribe_id=20,
                tribe_name="亡者",
            ),
        )
        _kill(engine, departed)
        source = _play(engine, self.repository, 10254120)
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))

        revived = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 991050
        )
        self.assertTrue(revived.has_keyword("守护"))
        crest = next(
            emblem for emblem in engine.players[0].emblems
            if emblem.emblem_id == "charon_stygian_oarswoman"
        )
        self.assertEqual(crest.countdown, 1)

    def test_congregant_exact_copy_chain_decrements_each_copy_and_stops_at_cap(self):
        engine = self.fresh(seed=29)

        _play(engine, self.repository, 10313110)

        copies = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10313110
        ]
        self.assertEqual(len(copies), engine.config.max_board)
        self.assertEqual(
            [(unit.attack, unit.health, unit.max_health) for unit in copies],
            [(4, 6, 6), (4, 5, 5), (4, 4, 4), (4, 3, 3), (4, 2, 2)],
        )
        self.assertTrue(all(unit.has_keyword("突进") for unit in copies))
        self.assertTrue(all(unit.has_keyword("守护") for unit in copies))

        capped = self.fresh(seed=31)
        for index in range(3):
            _put_unit(capped, 0, _card(991060 + index))
        _play(capped, self.repository, 10313110)
        self.assertEqual(len(capped.players[0].board), capped.config.max_board)
        congregants = [
            unit for unit in capped.players[0].board
            if unit.definition.card_id == 10313110
        ]
        self.assertEqual(
            [(unit.health, unit.max_health) for unit in congregants],
            [(6, 6), (5, 5)],
        )

    def test_exact_copy_preserves_runtime_abilities_but_source_leaving_skips(self):
        engine = self.fresh(seed=37)
        source = _put_unit(
            engine,
            0,
            _card(
                991070,
                attack=2,
                life=5,
                keywords=frozenset({"守护"}),
            ),
        )
        source.add_keyword("灵气")
        source.health = 3
        operation = EffectOperation(
            EffectKind.SUMMON_EXACT_COPY,
            TargetKind.SELF,
            amount=1,
            secondary_amount=-1,
            target_key="copy",
        )
        engine._start_effects(
            source.definition,
            source.entity_id,
            (
                operation,
                EffectOperation(
                    EffectKind.REMOVE_LAST_WORDS,
                    TargetKind.PREVIOUS_TARGET,
                    target_key="copy",
                ),
            ),
            controller=0,
        )
        copied = engine.players[0].board[-1]
        self.assertEqual((copied.attack, copied.health, copied.max_health), (3, 2, 4))
        self.assertTrue(copied.has_keyword("守护"))
        self.assertTrue(copied.has_keyword("灵气"))
        self.assertTrue(copied.last_words_removed)
        self.assertFalse(source.last_words_removed)

        engine.players[0].board.remove(source)
        engine._start_effects(
            source.definition,
            source.entity_id,
            (
                operation,
                EffectOperation(
                    EffectKind.REMOVE_LAST_WORDS,
                    TargetKind.PREVIOUS_TARGET,
                    target_key="copy",
                ),
            ),
            controller=0,
        )
        self.assertEqual(len(engine.players[0].board), 1)
        engine.assert_invariants()

    def test_vam_super_evolve_modes_barrier_and_seeded_distributed_damage(self):
        barrier_game = self.fresh(seed=41)
        ally = _put_unit(barrier_game, 0, _card(991080))
        source = _play(barrier_game, self.repository, 10434110)
        _enable_super_evolve(barrier_game)
        barrier_game.apply(SuperEvolve(0, source.entity_id))
        source_barriers = source.barrier_charges
        self.assertIsNotNone(barrier_game.state.pending_choice)
        _choose_mode(barrier_game, "barrier")
        self.assertTrue(ally.has_keyword("屏障"))
        self.assertEqual(source.barrier_charges, source_barriers)

        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=43)
            source = _play(engine, self.repository, 10434110)
            enemies = [
                _put_unit(engine, 1, _card(991090 + index, attack=0, life=10))
                for index in range(3)
            ]
            _enable_super_evolve(engine)
            engine.apply(SuperEvolve(0, source.entity_id))
            _choose_mode(engine, "distribute_damage")
            damages = tuple(10 - unit.health for unit in enemies)
            self.assertEqual(sum(damages), source.attack)
            outcomes.append((damages, engine.deterministic_fingerprint()))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_vam_mode_choice_action_mask_matches_commands_and_illegal_is_atomic(self):
        deck = [_card(991100 + index, class_id=3) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=3,
            class_b=3,
            seed=47,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=47)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        source = _play(env.core, self.repository, 10434110)
        _enable_super_evolve(env.core)
        env.core.apply(SuperEvolve(0, source.entity_id))
        env.invalidate_cache(reason="Vam mode choice")
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        self.assertEqual(len(decoded), 2)
        before = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(Choose(0, "mode:missing"))
        self.assertEqual(env.core.deterministic_fingerprint(), before)

    def test_insomniac_evolve_destroys_crest_and_last_words_resolve_simultaneously(self):
        engine = self.fresh(seed=53)
        ally = _put_unit(engine, 0, _card(991120, attack=1, life=3))
        enemy = _put_unit(engine, 1, _card(991121, attack=1, life=3))
        source = _play(engine, self.repository, 10532110)
        self.assertTrue(any(
            emblem.emblem_id == "insomniac_witch"
            for emblem in engine.players[0].emblems
        ))
        _enable_evolve(engine)

        engine.apply(Evolve(0, source.entity_id))

        self.assertFalse(engine.players[0].emblems)
        self.assertNotIn(ally, engine.players[0].board)
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertIn(source, engine.players[0].board)
        self.assertEqual(source.health, 2)
        deaths = [
            event for event in engine.event_history
            if event.type is EventType.FOLLOWER_DESTROYED
            and event.source_id in {ally.entity_id, enemy.entity_id}
        ]
        self.assertEqual(len(deaths), 2)

    def test_shymm_tokens_capacity_drain_and_crest_attack_buff_before_combat(self):
        engine = self.fresh(seed=59)
        engine.players[0].health = 10
        source = _play(engine, self.repository, 10634110)
        tokens = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10631110
        ]
        self.assertEqual(len(tokens), 2)
        self.assertTrue(source.has_keyword("吸血"))

        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        target = _put_unit(engine, 1, _card(991130, attack=0, life=5))
        token = tokens[0]
        self.assertIn(Attack(0, token.entity_id, target.entity_id), engine.legal_commands())
        engine.apply(Attack(0, token.entity_id, target.entity_id))
        self.assertEqual(token.attack, 2)
        self.assertEqual(target.health, 3)

        capped = self.fresh(seed=61)
        for index in range(3):
            _put_unit(capped, 0, _card(991140 + index))
        _play(capped, self.repository, 10634110)
        self.assertEqual(len(capped.players[0].board), capped.config.max_board)
        self.assertEqual(
            sum(
                unit.definition.card_id == 10631110
                for unit in capped.players[0].board
            ),
            1,
        )

    def test_edeth_and_kratos_replacements_only_lose_last_words(self):
        engine = self.fresh(seed=67)
        edeth = _put_unit(engine, 0, self.repository.get(10862110))
        kratos = _put_unit(engine, 0, self.repository.get(10871110))

        _kill(engine, edeth, kratos)

        replacements = {
            unit.definition.card_id: unit
            for unit in engine.players[0].board
            if unit.definition.card_id in {10862110, 10871110}
        }
        self.assertEqual(set(replacements), {10862110, 10871110})
        self.assertTrue(replacements[10862110].last_words_removed)
        self.assertTrue(replacements[10871110].last_words_removed)
        self.assertFalse(replacements[10862110].printed_abilities_removed)
        self.assertTrue(replacements[10862110].has_keyword("守护"))
        self.assertTrue(replacements[10862110].has_keyword("灵气"))
        self.assertTrue(replacements[10871110].has_keyword("守护"))
        removal_events = [
            event for event in engine.event_history
            if event.type is EventType.FOLLOWER_LAST_WORDS_REMOVED
        ]
        self.assertEqual(len(removal_events), 2)

        _kill(engine, *replacements.values())
        self.assertFalse(any(
            unit.definition.card_id in {10862110, 10871110}
            for unit in engine.players[0].board
        ))
        engine.assert_invariants()

    def test_edeth_super_evolve_target_no_target_and_stale_target_paths(self):
        targeted = self.fresh(seed=71)
        source = _play(targeted, self.repository, 10862110)
        enemy = _put_unit(targeted, 1, _card(991150))
        _enable_super_evolve(targeted)
        targeted.apply(SuperEvolve(0, source.entity_id))
        _choose_entity(targeted, enemy.entity_id)
        self.assertNotIn(enemy, targeted.players[1].board)

        no_target = self.fresh(seed=73)
        source = _play(no_target, self.repository, 10862110)
        _enable_super_evolve(no_target)
        no_target.apply(SuperEvolve(0, source.entity_id))
        self.assertTrue(source.super_evolved)
        self.assertIsNone(no_target.state.pending_choice)

        stale = self.fresh(seed=79)
        source = _play(stale, self.repository, 10862110)
        enemy = _put_unit(stale, 1, _card(991151))
        _enable_super_evolve(stale)
        stale.apply(SuperEvolve(0, source.entity_id))
        request = stale.state.pending_choice
        option = next(
            option for option in request.options
            if option.entity_id == enemy.entity_id
        )
        stale.players[1].board.remove(enemy)
        stale.apply(Choose(0, option.option_id))
        self.assertIsNone(stale.state.pending_choice)
        stale.assert_invariants()

    def test_seeded_multi_card_sequence_fingerprints_match(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=83)
            _play(engine, self.repository, 10313110)
            engine.players[0].board.clear()
            engine.players[0].mana = 10
            _play(engine, self.repository, 10634110)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])


class ExistingPrimitivesNinthDatabaseAuditTests(unittest.TestCase):
    def test_database_multilingual_text_stats_references_modes_and_hashes(self):
        expected = {
            10131110: (10001, 3, 5, 1, 1),
            10144110: (10001, 4, 7, 7, 7),
            10254120: (10002, 5, 5, 4, 4),
            10313110: (10003, 1, 9, 4, 6),
            10434110: (10004, 3, 3, 0, 1),
            10532110: (10005, 3, 4, 3, 3),
            10634110: (10006, 3, 3, 1, 2),
            10862110: (10008, 6, 8, 6, 7),
            10871110: (10008, 7, 7, 7, 7),
        }
        expected_refs = {
            10131110: (),
            10144110: (),
            10254120: (),
            10313110: (),
            10434110: (),
            10532110: (),
            10634110: (10631110,),
            10862110: (10862110,),
            10871110: (10871110,),
        }
        expected_phrases = {
            10131110: ("On Spellboost", "this follower's attack"),
            10144110: ("discard it", "cost of the selected card"),
            10254120: ("Reanimate", "Departed follower", "Ward"),
            10313110: ("exact copy", "-0/-1"),
            10434110: ("Spellboost your hand", "Select a", "Barrier"),
            10532110: ("Crest: Insomniac Witch", "Destroy"),
            10634110: ("Crystalspawn", "Drain"),
            10862110: ("remove", "Last Words", "Aura"),
            10871110: ("remove", "Last Words", "Ward"),
        }
        expected_mode_counts = {
            10131110: 0,
            10144110: 2,
            10254120: 1,
            10313110: 0,
            10434110: 0,
            10532110: 1,
            10634110: 1,
            10862110: 0,
            10871110: 0,
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            source_map = _load_source_text_map(connection)
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT card_set_id, class_id, cost, attack, life
                        FROM cards WHERE card_id=?
                        """,
                        (card_id,),
                    ).fetchone()
                    self.assertEqual(row, values)
                    self.assertEqual(
                        _source_text_sha256(source_map[card_id]),
                        SOURCE_HASHES[card_id],
                    )
                    text = "\n".join(
                        row[0] for row in connection.execute(
                            "SELECT text_eng FROM skill_texts WHERE card_id=? "
                            "ORDER BY position",
                            (card_id,),
                        )
                    )
                    for phrase in expected_phrases[card_id]:
                        self.assertIn(phrase, text)
                    references = tuple(
                        row[0] for row in connection.execute(
                            "SELECT referenced_card_id FROM card_references "
                            "WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    )
                    self.assertEqual(references, expected_refs[card_id])
                    mode_count = connection.execute(
                        "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                        (card_id,),
                    ).fetchone()[0]
                    self.assertEqual(mode_count, expected_mode_counts[card_id])

    def test_all_nine_cards_are_exact_with_direct_clause_and_token_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertEqual(
            report["summary"]["coverage_counts"],
            {
                "covered_exact": 668,
                "text_unclear": 16,
                "supported_missing_rule": 51,
                "token_or_non_collectible": 91,
            },
        )
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                classification = report["classifications"][str(card_id)]
                audit = classification["clause_audit"]
                self.assertEqual(classification["coverage"], "covered_exact")
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(
                    audit["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(audit["test_evidence"], [TEST_EVIDENCE])

        token_audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        self.assertEqual(
            token_audit["summary"]["categories"]["entry_behavior_complete"],
            91,
        )


if __name__ == "__main__":
    unittest.main()
