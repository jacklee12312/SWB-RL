# -*- coding: utf-8 -*-
"""Exact listener, Spellboost, Token, and two-mode follow-up cards."""

from __future__ import annotations

import itertools
import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, EndTurn, PlayCard, SuperEvolve
from swb.engine.effects import (
    EffectKind,
    EffectOperation,
    HandFilter,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import DeckCard, HandCard
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10121130,
    10131310,
    10212120,
    10232310,
    10252110,
    10353310,
    10361120,
    10612110,
    10673310,
    10721110,
    10812110,
)
SOURCE_HASHES = {
    10121130: "817103e747a52c0650eaa95191bece9a11ff040544af8c0787833fdb4b187754",
    10131310: "791e998fb2d4b24b7451bad01446b888b33e6e04b10f26c883f10556671869c8",
    10212120: "7ac811fa16cd65431f480acb8f0935a906190cd757c87fa28eb77e1f0c59b8a5",
    10232310: "4633b88ed54f0e8eea1f47ccb80c327ced86bf2fa1c06e27ce68e326a7f345be",
    10252110: "cd9c7850e998a5cd9f319306d4ab515d3d70747bea96d55e6866e13afe5c1953",
    10353310: "c4a6d126cf572b50327d292fc8f55b704d5b70efb3e081e1a08a24759361edb1",
    10361120: "b78ee6b7326fcc6d5642bc9aad9476de1b47e1b466a37c3d2f12dfc1d28dd1dc",
    10612110: "9dbe1162cd0efacd40722f7e018f53e7abb7a685fff231b6520661007dd57af5",
    10673310: "f84f805481d616f35f1226db48f4971b6ffb6ce6d8041057a15305453b28735d",
    10721110: "f3aa9170b582c7774341c53ea429aee779de1220a1d6b98f8ccc7f3d8c64e416",
    10812110: "40dd874630409ab4e4cf3f8512892ac07e81926d3ba091fc95d9b577f550f844",
}
TEST_EVIDENCE = "tests/test_real_listener_spellboost_followup_batch.py"


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _choose_mode(engine, option_id: str) -> None:
    request = engine.state.pending_choice
    engine.apply(Choose(request.player_index, f"choose_one:{option_id}"))


def _enable_super_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.first_player_super_evolution_unlock_turn
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.super_evolved_this_turn = False


def _buff_self(engine, source, attack: int, health: int) -> None:
    engine._start_effects(
        source.definition,
        source.entity_id,
        (
            EffectOperation(
                kind=EffectKind.BUFF_UNIT,
                target=TargetKind.SELF,
                amount=attack,
                secondary_amount=health,
            ),
        ),
        controller=0,
        label="测试属性增加",
    )


class ListenerSpellboostSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository("data/cards.sqlite3")

    def test_hand_filter_accepts_and_normalizes_ability_keyword(self):
        operation = _parse_operation(
            {
                "kind": "spellboost_hand",
                "target": "own_hand",
                "amount": 1,
                "hand_filter": {"keyword": "魔力增幅时"},
            },
            "test",
            10131310,
        )

        self.assertEqual(operation.hand_filter, HandFilter(keyword="魔力增幅"))
        self.assertTrue(operation.hand_filter.matches(self.repository.get(10232310)))
        self.assertFalse(operation.hand_filter.matches(self.repository.get(10612110)))

    def test_hand_filter_rejects_unknown_or_non_string_keyword(self):
        for keyword in ("不存在的能力", 1, True):
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                _parse_operation(
                    {
                        "kind": "spellboost_hand",
                        "target": "own_hand",
                        "amount": 1,
                        "hand_filter": {"keyword": keyword},
                    },
                    "test",
                    10131310,
                )


class RealListenerSpellboostFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1701):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_all_cards_listeners_modes_and_keywords(self):
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10252110),
            ("突进",),
        )
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10361120),
            ("守护",),
        )
        self.assertEqual(
            self.rulebook.operations_for(10131310, Trigger.PLAY)[0].hand_filter,
            HandFilter(keyword="魔力增幅"),
        )
        self.assertEqual(
            self.rulebook.operations_for(10232310, Trigger.PLAY)[0].kind,
            EffectKind.DISTRIBUTE_DAMAGE,
        )
        mode = self.rulebook.operations_for(10353310, Trigger.PLAY)[0]
        self.assertEqual(mode.choose_count, 2)
        self.assertEqual(
            [option.option_id for option in mode.choose_one_options],
            [
                "restore_mana",
                "draw_follower",
                "draw_three_cost_spell",
                "damage_follower",
            ],
        )
        listener_cards = {
            10121130,
            10212120,
            10252110,
            10361120,
            10673310,
            10721110,
            10812110,
        }
        self.assertTrue(all(self.rulebook.listeners_for(card_id) for card_id in listener_cards))
        ruled_cards = {
            card_id
            for card_id in CARD_IDS
            if any(
                self.rulebook.operations_for(card_id, trigger)
                for trigger in Trigger
            )
            or self.rulebook.listeners_for(card_id)
        }
        self.assertEqual(ruled_cards, set(CARD_IDS))

    def test_rainbow_miracle_requires_spellboost_target_then_boosts_and_draws(self):
        engine = self.fresh(seed=3)
        target = _put_hand(engine, self.repository.get(10232310))
        ineligible = _put_hand(engine, self.repository.get(10612110))
        source = _put_hand(engine, self.repository.get(10131310))
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
        request = engine.state.pending_choice
        self.assertEqual(
            [option.entity_id for option in request.options],
            [target.entity_id],
        )
        _choose_entity(engine, target.entity_id)

        self.assertEqual(target.spellboost_count, 2)
        # Playing the spell performs the normal global Spellboost after its
        # text resolves; only the selected eligible card receives the extra one.
        self.assertEqual(ineligible.spellboost_count, 1)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)
        target_events = [
            event
            for event in engine.event_history
            if event.type is EventType.SPELLBOOSTED
            and event.source_id == target.entity_id
        ]
        self.assertEqual(len(target_events), 2)

    def test_rainbow_miracle_no_candidate_is_illegal_and_atomic(self):
        engine = self.fresh(seed=5)
        _put_hand(engine, self.repository.get(10612110))
        source = _put_hand(engine, self.repository.get(10131310))
        command = PlayCard(0, engine.players[0].hand.index(source))
        before = engine.deterministic_fingerprint()
        before_rng = engine.random.getstate()

        with self.assertRaises(IllegalCommand):
            engine.apply(command)

        self.assertEqual(engine.deterministic_fingerprint(), before)
        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertNotIn(command, engine.legal_commands())

    def test_chaos_flame_uses_frozen_spellboost_x_and_oldest_first_distribution(self):
        engine = self.fresh(seed=7)
        first = _put_unit(engine, 1, _card(991001, life=3))
        second = _put_unit(engine, 1, _card(991002, life=9))
        source = _put_hand(engine, self.repository.get(10232310))
        source.apply_spellboost(7)

        engine.apply(PlayCard(0, engine.players[0].hand.index(source)))

        self.assertNotIn(first, engine.players[1].board)
        self.assertEqual(second.health, 5)
        self.assertEqual(
            sum(
                event.amount
                for event in engine.event_history
                if event.type is EventType.DAMAGE_APPLIED
                and event.target_id in {first.entity_id, second.entity_id}
            ),
            7,
        )

        zero = self.fresh(seed=11)
        untouched = _put_unit(zero, 1, _card(991003, life=5))
        zero_source = _put_hand(zero, self.repository.get(10232310))
        before_rng = zero.random.getstate()
        zero.apply(PlayCard(0, zero.players[0].hand.index(zero_source)))
        self.assertEqual(untouched.health, 5)
        self.assertEqual(zero.random.getstate(), before_rng)

    def test_super_evolution_sets_both_hand_cards_to_one_and_fanfares_resolve(self):
        engine = self.fresh(seed=13)
        engineer = _put_hand(engine, self.repository.get(10721110))
        fencer = _put_hand(engine, self.repository.get(10212120))
        ally = _put_unit(engine, 0, _card(991004, attack=2, life=4))
        _enable_super_evolution(engine)

        engine.apply(SuperEvolve(0, ally.entity_id))

        self.assertEqual((fencer.current_cost, engineer.current_cost), (1, 1))
        engine.apply(PlayCard(0, engine.players[0].hand.index(fencer)))
        self.assertTrue(any(card.card_id == 90011110 for card in engine.players[0].hand))

        enemy = _put_unit(engine, 1, _card(991005, life=5))
        engine.apply(PlayCard(0, engine.players[0].hand.index(engineer)))
        _choose_entity(engine, enemy.entity_id)
        self.assertEqual(enemy.health, 2)

        no_target = self.fresh(seed=17)
        source = _put_hand(no_target, self.repository.get(10721110))
        no_target.apply(PlayCard(0, no_target.players[0].hand.index(source)))
        self.assertIsNone(no_target.state.pending_choice)

    def test_piyura_buffs_other_super_evolved_follower_and_not_itself(self):
        engine = self.fresh(seed=19)
        source = _play(engine, self.repository, 10252110)
        other = _put_unit(engine, 0, _card(991006, attack=2, life=4))
        _enable_super_evolution(engine)

        engine.apply(SuperEvolve(0, other.entity_id))

        self.assertEqual((source.attack, source.max_health), (4, 4))
        self.assertEqual((other.attack, other.max_health), (7, 7))
        self.assertTrue(source.has_keyword("突进"))

        self_case = self.fresh(seed=23)
        self_source = _play(self_case, self.repository, 10252110)
        _enable_super_evolution(self_case)
        self_case.apply(SuperEvolve(0, self_source.entity_id))
        self.assertEqual((self_source.attack, self_source.max_health), (5, 7))

    def test_holy_knight_heals_per_positive_buff_but_not_set_stats(self):
        engine = self.fresh(seed=29)
        engine.players[0].health = 15
        source = _play(engine, self.repository, 10361120)
        self.assertTrue(source.has_keyword("守护"))

        _buff_self(engine, source, 0, 1)
        _buff_self(engine, source, 2, 0)
        _buff_self(engine, source, -1, 0)
        self.assertEqual(engine.players[0].health, 17)

        engine._start_effects(
            source.definition,
            source.entity_id,
            (
                EffectOperation(
                    kind=EffectKind.SET_STATS,
                    target=TargetKind.SELF,
                    amount=8,
                    secondary_amount=8,
                    set_attack=True,
                    set_health=True,
                ),
            ),
            controller=0,
            label="测试属性设定",
        )
        # A specific-value assignment is not an additive buff and replaces
        # older changes in the dimensions it assigns.
        self.assertEqual((source.attack, source.max_health), (8, 8))
        self.assertEqual(engine.players[0].health, 17)

    def test_lilara_heals_for_each_successful_soldier_entry(self):
        engine = self.fresh(seed=31)
        engine.players[0].health = 10
        source = _play(engine, self.repository, 10121130)
        self.assertEqual(engine.players[0].health, 11)
        self.assertEqual(
            sum(unit.definition.card_id == 90021120 for unit in engine.players[0].board),
            1,
        )

        engine._start_effects(
            source.definition,
            source.entity_id,
            (
                EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90021120),
                EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90021120),
                EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90021120),
                EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90021120),
            ),
            controller=0,
            label="测试士兵入场",
        )

        self.assertEqual(len(engine.players[0].board), engine.config.max_board)
        self.assertEqual(engine.players[0].health, 14)

    def test_springbloom_fanfare_and_lufle_once_per_own_turn_last_words(self):
        spring = self.fresh(seed=37)
        _play(spring, self.repository, 10612110)
        generated = next(card for card in spring.players[0].hand if card.card_id == 90011120)
        self.assertIs(generated.origin, CardOrigin.TOKEN)

        engine = self.fresh(seed=41)
        source = _play(engine, self.repository, 10812110)
        _buff_self(engine, source, 1, 0)
        _buff_self(engine, source, 0, 1)
        self.assertEqual(
            sum(unit.definition.card_id == 90011110 for unit in engine.players[0].board),
            1,
        )

        engine.apply(EndTurn(0))
        _buff_self(engine, source, 1, 1)
        self.assertEqual(
            sum(unit.definition.card_id == 90011110 for unit in engine.players[0].board),
            1,
        )
        engine.apply(EndTurn(1))
        _buff_self(engine, source, 1, 0)
        self.assertEqual(
            sum(unit.definition.card_id == 90011110 for unit in engine.players[0].board),
            2,
        )

        engine._start_effects(
            source.definition,
            source.entity_id,
            (EffectOperation(EffectKind.DESTROY, TargetKind.SELF),),
            controller=0,
            label="测试谢幕曲",
        )
        self.assertTrue(any(card.card_id == 90011110 for card in engine.players[0].hand))

    def test_bad_axe_hand_reductions_stack_and_expire_at_turn_end(self):
        engine = self.fresh(seed=43)
        first = _put_hand(engine, self.repository.get(10673310))
        second = _put_hand(engine, self.repository.get(10673310))
        test_source = _card(991007, card_type="法术", attack=None, life=None)

        engine._start_effects(
            test_source,
            None,
            (EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=90011110),),
            controller=0,
            label="低费随从入场",
        )
        self.assertEqual((first.current_cost, second.current_cost), (3, 3))

        engine._start_effects(
            test_source,
            None,
            (
                EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=10524110),
                EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=10524110),
            ),
            controller=0,
            label="高费随从入场",
        )
        self.assertEqual((first.current_cost, second.current_cost), (1, 1))

        engine.apply(EndTurn(0))
        self.assertEqual((first.current_cost, second.current_cost), (3, 3))

    def test_bad_axe_play_evolves_only_eligible_ally_and_seeded_damage_replays(self):
        outcomes = []
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=47)
            eligible = _put_unit(engine, 0, self.repository.get(10524110))
            ineligible = _put_unit(engine, 0, _card(991008, cost=4, attack=3, life=4))
            enemies = [
                _put_unit(engine, 1, _card(991009 + index, life=8))
                for index in range(2)
            ]
            source = _put_hand(engine, self.repository.get(10673310))

            engine.apply(PlayCard(0, engine.players[0].hand.index(source)))

            self.assertTrue(eligible.evolved)
            self.assertFalse(ineligible.evolved)
            outcomes.append(tuple(enemy.health for enemy in enemies))
            fingerprints.append(engine.deterministic_fingerprint())

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(sum(8 - health for health in outcomes[0]), 6)
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_clamor_all_six_mode_pairs_execute_every_mode(self):
        option_ids = (
            "restore_mana",
            "draw_follower",
            "draw_three_cost_spell",
            "damage_follower",
        )
        for index, pair in enumerate(itertools.combinations(option_ids, 2)):
            with self.subTest(pair=pair):
                engine = self.fresh(seed=53 + index)
                follower = _card(991100 + index, cost=7)
                spell = _card(
                    991200 + index,
                    cost=3,
                    card_type="法术",
                    attack=None,
                    life=None,
                )
                filler = _card(
                    991300 + index,
                    cost=2,
                    card_type="法术",
                    attack=None,
                    life=None,
                )
                engine.players[0].deck = [
                    DeckCard(follower),
                    DeckCard(spell),
                    DeckCard(filler),
                ]
                enemy = _put_unit(engine, 1, _card(991400 + index, life=10))
                source = _put_hand(engine, self.repository.get(10353310))
                engine.players[0].mana = 4

                engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
                _choose_mode(engine, pair[1])
                _choose_mode(engine, pair[0])

                self.assertEqual(
                    engine.players[0].mana,
                    2 if "restore_mana" in pair else 1,
                )
                self.assertEqual(
                    any(card.card_id == follower.card_id for card in engine.players[0].hand),
                    "draw_follower" in pair,
                )
                self.assertEqual(
                    any(card.card_id == spell.card_id for card in engine.players[0].hand),
                    "draw_three_cost_spell" in pair,
                )
                self.assertEqual(
                    enemy.health,
                    6 if "damage_follower" in pair else 10,
                )

    def test_clamor_empty_random_mode_is_safe_rng_neutral(self):
        engine = self.fresh(seed=67)
        source = _put_hand(engine, self.repository.get(10353310))
        engine.players[0].mana = 4
        engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
        before_rng = engine.random.getstate()

        _choose_mode(engine, "damage_follower")
        _choose_mode(engine, "restore_mana")

        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertEqual(engine.players[0].mana, 2)

    def test_rl_mask_reuses_fixed_hand_choice_and_mode_actions(self):
        deck = [_card(992000 + index) for index in range(40)]
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
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        target = _put_hand(env.core, self.repository.get(10232310))
        source = _put_hand(env.core, self.repository.get(10131310))
        play = PlayCard(0, env.players[0].hand.index(source))
        play_action = env._encode_command(play)

        self.assertEqual(
            (env.ACTION_SIZE, len(env.observation())),
            (112, ShadowverseEnv.OBSERVATION_V1_SIZE),
        )
        self.assertTrue(env.action_mask()[play_action])
        env.step(play_action)
        request = env.core.state.pending_choice
        self.assertEqual([option.entity_id for option in request.options], [target.entity_id])
        choice = Choose(0, request.options[0].option_id)
        choice_action = env._encode_command(choice)
        self.assertTrue(env.action_mask()[choice_action])
        env.step(choice_action)
        self.assertIsNone(env.core.state.pending_choice)


class ListenerSpellboostDatabaseAuditTests(unittest.TestCase):
    def test_database_text_references_and_alt_modes_match_reviewed_source(self):
        expected_references = {
            10121130: (90021120,),
            10212120: (90011110,),
            10612110: (90011120,),
            10812110: (90011110,),
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id in CARD_IDS:
                with self.subTest(card_id=card_id):
                    texts = [
                        re.sub(r"<[^>]+>", "", row[0])
                        for row in connection.execute(
                            "SELECT text_eng FROM skill_texts WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    ]
                    self.assertTrue(texts)
                    self.assertTrue(all(text.strip() for text in texts))
                    references = tuple(
                        row[0]
                        for row in connection.execute(
                            "SELECT referenced_card_id FROM card_references "
                            "WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    )
                    self.assertEqual(references, expected_references.get(card_id, ()))
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_all_cards_have_exact_mapped_clause_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
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


if __name__ == "__main__":
    unittest.main()
