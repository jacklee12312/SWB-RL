# -*- coding: utf-8 -*-
"""Exact selected-hand stat buffs and filtered hand-count conditions."""

from __future__ import annotations

import sqlite3
import re
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_condition
from swb.engine.commands import ActivateAmulet, Choose, Evolve, PlayCard
from swb.engine.conditions import EvalContext, evaluate_condition
from swb.engine.effects import ConditionType, EffectKind, HandFilter, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10112210, 10521120, 10741120, 10853110)
SOURCE_HASHES = {
    10112210: "06a57556640cef59731b9793efa43e31924faf8ea2016888e4ed3e87c6523544",
    10521120: "824afd414b4918b263b680a577993396b971783da7af984b666c1f813f103a11",
    10741120: "b440f19fcd5ae36673fc3fc18195bd2758ddc270b5be9805494670f6056460fa",
    10853110: "86b144215e4269cbe80f1943f3af0c68761e8ecee93dc1d4c019cd9f3d1047b2",
}


def _choose(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


class SelectedHandStatSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")

    def test_filtered_hand_count_condition_parses_and_evaluates_definition_type(self):
        condition = _parse_condition(
            {
                "type": "controller_hand_count_at_least",
                "value": 2,
                "filter": {"card_type": "法术"},
            },
            "test",
            1,
        )
        self.assertEqual(condition.card_filter, HandFilter(card_type="法术"))

        engine = _fresh(self.rulebook, CardRepository("data/cards.sqlite3"), seed=3)
        _put_hand(engine, _card(9001, card_type="法术", attack=None, life=None))
        _put_hand(engine, _card(9002, card_type="法术", attack=None, life=None))
        _put_hand(engine, _card(9003))
        self.assertTrue(evaluate_condition(
            condition,
            EvalContext(controller=0, players=engine.players),
        ))

    def test_filtered_hand_count_requires_value_and_rejects_unrelated_condition(self):
        with self.assertRaisesRegex(ValueError, "required"):
            _parse_condition(
                {
                    "type": "controller_hand_count_at_least",
                    "filter": {"card_type": "法术"},
                },
                "test",
                1,
            )
        with self.assertRaisesRegex(ValueError, "only valid"):
            _parse_condition(
                {
                    "type": "source_evolved",
                    "filter": {"card_type": "法术"},
                },
                "test",
                1,
            )

    def test_real_rule_shapes_and_non_intrinsic_ward_are_auditable(self):
        activation = self.rulebook.operations_for(10112210, Trigger.ACTIVATE)
        self.assertEqual(
            [(operation.kind, operation.target) for operation in activation],
            [
                (EffectKind.DESTROY, TargetKind.SELF),
                (EffectKind.BUFF_UNIT, TargetKind.OWN_UNIT),
            ],
        )
        self.assertTrue(activation[1].requires_target)
        self.assertIn("守护", self.rulebook.non_intrinsic_keywords(10521120))
        for card_id in (10741120, 10853110):
            operation = self.rulebook.operations_for(card_id, Trigger.EVOLVE)[0]
            self.assertEqual(operation.kind, EffectKind.BUFF_HAND_CARD)
            self.assertEqual(operation.hand_filter, HandFilter(card_type="随从"))


class RealSelectedHandStatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2401):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_phosphorescent_rock_combo_threshold_and_exact_token_order(self):
        below = self.fresh(seed=5)
        below.players[0].cards_played_this_turn = 1
        _play(below, self.repository, 10112210)
        self.assertEqual([card.card_id for card in below.players[0].hand], [90011110])
        self.assertEqual(below.players[0].hand[0].origin, CardOrigin.TOKEN)

        threshold = self.fresh(seed=7)
        threshold.players[0].cards_played_this_turn = 2
        _play(threshold, self.repository, 10112210)
        self.assertEqual(
            [card.card_id for card in threshold.players[0].hand],
            [90011110, 90011310],
        )
        self.assertTrue(all(card.origin is CardOrigin.TOKEN for card in threshold.players[0].hand))

    def test_phosphorescent_rock_activation_requires_target_then_survives_source_death(self):
        engine = self.fresh(seed=11)
        ally = _put_unit(engine, 0, _card(9101, attack=2, life=3))
        source = _play(engine, self.repository, 10112210)
        command = ActivateAmulet(0, source.entity_id)
        self.assertIn(command, engine.legal_commands())
        engine.apply(command)
        self.assertNotIn(source, engine.players[0].board)
        self.assertEqual([option.entity_id for option in engine.state.pending_choice.options], [ally.entity_id])
        _choose(engine, ally.entity_id)
        self.assertEqual((ally.attack, ally.health, ally.max_health), (3, 4, 4))

        no_target = self.fresh(seed=13)
        source = _play(no_target, self.repository, 10112210)
        command = ActivateAmulet(0, source.entity_id)
        before = no_target.deterministic_fingerprint()
        self.assertNotIn(command, no_target.legal_commands())
        with self.assertRaises(IllegalCommand):
            no_target.apply(command)
        self.assertEqual(no_target.deterministic_fingerprint(), before)

    def test_pipe_jewel_counts_only_spells_and_evolve_adds_glittering_gold(self):
        below = self.fresh(seed=17)
        _put_hand(below, _card(9201, card_type="法术", attack=None, life=None))
        _put_hand(below, _card(9202))
        source = _play(below, self.repository, 10521120)
        definition = self.repository.get(10521120)
        self.assertEqual((source.attack, source.max_health), (definition.attack, definition.life))
        self.assertFalse(source.has_keyword("守护"))

        active = self.fresh(seed=19)
        _put_hand(active, _card(9203, card_type="法术", attack=None, life=None))
        _put_hand(active, _card(9204, card_type="法术", attack=None, life=None))
        source = _play(active, self.repository, 10521120)
        self.assertEqual(
            (source.attack, source.health, source.max_health),
            (definition.attack + 1, definition.life + 1, definition.life + 1),
        )
        self.assertTrue(source.has_keyword("守护"))
        active.players[0].turns_started = active.config.evolution_unlock_turn
        active.apply(Evolve(0, source.entity_id))
        gold = next(card for card in active.players[0].hand if card.card_id == 90021350)
        self.assertEqual(gold.origin, CardOrigin.TOKEN)

    def test_transport_wyvern_excludes_self_then_buffs_selected_hand_follower(self):
        engine = self.fresh(seed=23)
        ally = _put_unit(engine, 0, _card(9301, attack=1, life=4))
        hand_follower = _put_hand(engine, _card(9302, attack=2, life=3))
        hand_spell = _put_hand(
            engine,
            _card(9303, card_type="法术", attack=None, life=None),
        )
        source = _play(engine, self.repository, 10741120)
        self.assertEqual([option.entity_id for option in engine.state.pending_choice.options], [ally.entity_id])
        _choose(engine, ally.entity_id)
        self.assertEqual((ally.attack, ally.max_health), (3, 6))

        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [hand_follower.entity_id],
        )
        _choose(engine, hand_follower.entity_id)
        self.assertEqual((hand_follower.attack, hand_follower.life), (4, 5))
        self.assertEqual((hand_spell.attack, hand_spell.life), (None, None))

        follower_index = engine.players[0].hand.index(hand_follower)
        engine.apply(PlayCard(0, follower_index))
        played = next(unit for unit in engine.players[0].board if unit.definition.card_id == 9302)
        self.assertEqual((played.attack, played.health, played.max_health), (4, 5, 5))

    def test_evolve_hand_target_leaving_is_revalidated_without_cross_targeting(self):
        engine = self.fresh(seed=29)
        source = _put_unit(engine, 0, self.repository.get(10853110))
        target = _put_hand(engine, _card(9401, attack=1, life=2))
        survivor = _put_hand(engine, _card(9402, attack=2, life=3))
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        engine.apply(Evolve(0, source.entity_id))
        request = engine.state.pending_choice
        option = next(option for option in request.options if option.entity_id == target.entity_id)
        target_index = engine.players[0].hand.index(target)
        engine.players[0].hand.pop(target_index)
        engine.players[0].hand_entity_ids.pop(target_index)

        engine.apply(Choose(0, option.option_id))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual((survivor.attack, survivor.life), (2, 3))

    def test_honest_cursecaster_health_reduction_deaths_and_hand_attack_buff(self):
        no_enemy = self.fresh(seed=31)
        source = _play(no_enemy, self.repository, 10853110)
        self.assertIsNone(no_enemy.state.pending_choice)

        engine = self.fresh(seed=37)
        enemy = _put_unit(engine, 1, _card(9501, attack=4, life=3))
        hand_follower = _put_hand(engine, _card(9502, attack=2, life=4))
        source = _play(engine, self.repository, 10853110)
        _choose(engine, enemy.entity_id)
        self.assertNotIn(enemy, engine.players[1].board)
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, hand_follower.entity_id)
        self.assertEqual((hand_follower.attack, hand_follower.life), (5, 4))

    def test_rl_evolve_and_hand_choice_masks_match_executable_commands(self):
        deck = [_card(9600 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=41,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=41)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].board.clear()
        env.players[0].turns_started = env.core.config.evolution_unlock_turn
        source = _put_unit(env.core, 0, self.repository.get(10741120))
        target = _put_hand(env.core, _card(9701))

        evolve = Evolve(0, source.entity_id)
        evolve_action = env._encode_command(evolve)
        self.assertTrue(env.action_mask()[evolve_action])
        env.step(evolve_action)
        request = env.core.state.pending_choice
        choice = Choose(
            0,
            next(option.option_id for option in request.options if option.entity_id == target.entity_id),
        )
        choice_action = env._encode_command(choice)
        self.assertTrue(env.action_mask()[choice_action])
        env.step(choice_action)
        self.assertEqual((target.attack, target.life), (3, 7))

    def test_seeded_replay_is_identical(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=43)
            ally = _put_unit(engine, 0, _card(9801))
            _play(engine, self.repository, 10741120)
            _choose(engine, ally.entity_id)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])


class SelectedHandStatDatabaseAuditTests(unittest.TestCase):
    def test_database_text_references_and_absent_modes_match_reviewed_sources(self):
        expected_phrases = {
            10112210: ("Add a Fairy", "Combo (3)", "Engage"),
            10521120: ("at least 2 spells", "Glittering Gold"),
            10741120: ("another allied follower", "follower in your hand"),
            10853110: ("give it -0/-3", "follower in your hand"),
        }
        expected_references = {
            10112210: [(90011110,), (90011310,)],
            10521120: [(90021350,)],
            10741120: [],
            10853110: [],
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id, phrases in expected_phrases.items():
                with self.subTest(card_id=card_id):
                    rows = connection.execute(
                        "SELECT text_eng FROM skill_texts WHERE card_id=? ORDER BY position",
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(len(rows), 1)
                    normalized_text = re.sub(r"<[^>]+>", "", rows[0][0])
                    for phrase in phrases:
                        self.assertIn(phrase, normalized_text)
                    self.assertEqual(
                        connection.execute(
                            "SELECT referenced_card_id FROM card_references WHERE card_id=? ORDER BY position",
                            (card_id,),
                        ).fetchall(),
                        expected_references[card_id],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_cards_have_exact_mapped_clause_evidence(self):
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
                    ["tests/test_real_selected_hand_stat_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
