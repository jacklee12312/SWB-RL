# -*- coding: utf-8 -*-
"""Exact Portalcraft cards using distinct Artifact follower entry history."""

from __future__ import annotations

from dataclasses import replace
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import (
    RuleBook,
    _parse_condition,
    _parse_expression,
)
from swb.engine.commands import Choose, PlayCard, SuperEvolve
from swb.engine.conditions import evaluate_condition, evaluate_expression
from swb.engine.effects import ConditionType, ExprType
from swb.engine.environment import ShadowverseEnv
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
    10771120,
    10771310,
    10772120,
    10773310,
    10774110,
    10774120,
    10873310,
)
ARTIFACT_IDS = (90071130, 90071140, 90071150)
SOURCE_HASHES = {
    10771120: "a1a934e0cf87edfa40e821f5850075beb508daca5490c4e1f626db80274c24bd",
    10771310: "95d5fe03fe842fd35a41ffe4582cbef0c9efbf7ef0a0cc2438e879f873b7a4a2",
    10772120: "435f8fa1f2f2dc38dae69260883cab90bae896368289be8e09c49f0f62f9d8b4",
    10773310: "a37594eedb3286d59f47efe09a2d89759f8e4285043270ff79047d75ecc4f546",
    10774110: "c6fa4071b1171b4f7bba5c5d4c1de768eacbf733e7520dfcadc526d9897c2318",
    10774120: "737995cc8cffc09b3d67c68ab41fbc7c61084d4608ce1fa7f993b694bb2e6a7c",
    10873310: "dc7862d073a4e6e433324648fd15c2267c5a86263674d5bf898e40e55910b3d7",
}


def _prime_artifact_history(engine, repository, ids=ARTIFACT_IDS) -> None:
    for card_id in ids:
        engine.players[0].mana = 10
        _play(engine, repository, card_id)
    engine.players[0].board.clear()
    engine.players[0].hand.clear()
    engine.players[0].hand_entity_ids.clear()
    engine.players[0].mana = 10
    engine.assert_invariants()


def _artifact_count(engine) -> int:
    expression = _parse_expression(
        {
            "type": "controller_entered_follower_distinct_count",
            "filter": {"card_type": "随从", "tribe_name": "创造物"},
        },
        "test",
        1,
    )
    return evaluate_expression(expression, engine._eval_context(0))


def _choose_option(engine, option_id: str) -> None:
    request = engine.state.pending_choice
    engine.apply(Choose(request.player_index, option_id))


def _enable_super_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.first_player_super_evolution_unlock_turn
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.super_evolved_this_turn = False


class RealPortalArtifactEntryHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_entry_history_schema_counts_different_names_and_is_in_fingerprint(self):
        condition = _parse_condition(
            {
                "type": "controller_entered_follower_distinct_count_at_least",
                "value": 2,
                "filter": {"card_type": "随从", "tribe_name": "创造物"},
            },
            "test",
            1,
        )
        expression = _parse_expression(
            {
                "type": "controller_entered_follower_distinct_count",
                "filter": {"tribe_id": 14},
            },
            "test",
            1,
        )
        self.assertIs(
            condition.type,
            ConditionType.CONTROLLER_ENTERED_FOLLOWER_DISTINCT_COUNT_AT_LEAST,
        )
        self.assertIs(
            expression.type,
            ExprType.CONTROLLER_ENTERED_FOLLOWER_DISTINCT_COUNT,
        )

        engine = self.fresh(seed=3)
        before = engine.deterministic_fingerprint()
        analyzing = self.repository.get(90071130)
        _play(engine, self.repository, 90071130)
        same_name_new_id = replace(analyzing, card_id=990071130)
        _put_hand(engine, same_name_new_id)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        _play(engine, self.repository, 90071140)
        _put_hand(engine, _card(9901, name="普通随从"))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        engine._summon_follower_to_board(
            1,
            self.repository.get(90071150),
            summon_cause="test_opponent_entry",
        )

        context = engine._eval_context(0)
        self.assertTrue(evaluate_condition(condition, context))
        self.assertEqual(evaluate_expression(expression, context), 2)
        self.assertEqual(_artifact_count(engine), 2)
        self.assertEqual(
            [record.entry_sequence for record in engine.state.follower_entries],
            list(range(1, 6)),
        )
        self.assertNotEqual(engine.deterministic_fingerprint(), before)
        engine.assert_invariants()

        engine.reset(seed=3)
        self.assertEqual(engine.state.follower_entries, [])
        self.assertEqual(engine.state._next_follower_entry_sequence, 1)

    def test_entry_history_filter_schema_rejects_wrong_uses(self):
        with self.assertRaisesRegex(ValueError, "entered-follower distinct-count"):
            _parse_condition(
                {"type": "always", "filter": {"tribe_name": "创造物"}},
                "test",
                9,
            )
        with self.assertRaisesRegex(ValueError, "follower-history aggregate"):
            _parse_expression(
                {"type": "constant", "value": 1, "filter": {}},
                "test",
                9,
            )
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            _parse_expression(
                {
                    "type": "controller_entered_follower_distinct_count",
                    "filter": {"unknown": 1},
                },
                "test",
                9,
            )

    def test_entry_history_invariant_rejects_invalid_sequence(self):
        engine = self.fresh(seed=4)
        _play(engine, self.repository, 90071140)
        engine.state.follower_entries[0] = replace(
            engine.state.follower_entries[0],
            entry_sequence=0,
        )
        with self.assertRaisesRegex(
            IllegalCommand,
            "follower entry sequence must be unique and positive",
        ):
            engine.assert_invariants()

    def test_dope_dancer_threshold_and_board_shortage(self):
        below = self.fresh(seed=5)
        source = _play(below, self.repository, 10771120)
        dancers = [
            unit
            for unit in below.players[0].board
            if unit.definition.card_id == 10771120
        ]
        self.assertEqual(len(dancers), 2)
        self.assertTrue(all(unit.has_keyword("突进") for unit in dancers))
        self.assertIn(source, dancers)

        active = self.fresh(seed=7)
        _prime_artifact_history(active, self.repository)
        _play(active, self.repository, 10771120)
        self.assertEqual(
            sum(
                unit.definition.card_id == 10771120
                for unit in active.players[0].board
            ),
            3,
        )

        shortage = self.fresh(seed=11)
        _prime_artifact_history(shortage, self.repository)
        for index in range(3):
            _put_unit(shortage, 0, _card(10010 + index))
        _play(shortage, self.repository, 10771120)
        self.assertEqual(len(shortage.players[0].board), 5)
        self.assertEqual(
            sum(
                unit.definition.card_id == 10771120
                for unit in shortage.players[0].board
            ),
            2,
        )

    def test_street_run_choice_all_modes_and_rl_mask(self):
        below = self.fresh(seed=13)
        _play(below, self.repository, 10771310)
        self.assertEqual(
            [option.option_id for option in below.state.pending_choice.options],
            [
                "choose_one:analyzing_artifact",
                "choose_one:ancient_artifact",
            ],
        )
        fingerprint = below.deterministic_fingerprint()
        with self.assertRaisesRegex(IllegalCommand, "Choice option is invalid"):
            _choose_option(below, "choose_one:not_real")
        self.assertEqual(below.deterministic_fingerprint(), fingerprint)
        _choose_option(below, "choose_one:ancient_artifact")
        self.assertEqual(
            [card.card_id for card in below.players[0].hand],
            [90071140],
        )

        active = self.fresh(seed=17)
        _prime_artifact_history(active, self.repository)
        _play(active, self.repository, 10771310)
        self.assertIsNone(active.state.pending_choice)
        self.assertEqual(
            [card.card_id for card in active.players[0].hand],
            [90071130, 90071140],
        )

        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(11000, 11040)],
            [_card(card_id) for card_id in range(12000, 12040)],
            class_a=1,
            class_b=1,
            seed=19,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=19)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].board.clear()
        env.players[0].max_mana = env.players[0].mana = 10
        _put_hand(env.core, self.repository.get(10771310))
        env.step(env.PLAY_OFFSET)
        mask = env.action_mask()
        self.assertEqual(sum(mask), 2)
        self.assertTrue(mask[env.CHOICE_OFFSET])
        self.assertTrue(mask[env.CHOICE_OFFSET + 1])
        env.step(env.CHOICE_OFFSET)
        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual(env.observation()[-2:], [0.0, 0.0])
        env.players[0].mana = 10
        env.step(env.PLAY_OFFSET)
        self.assertEqual(env.observation()[-2:], [1 / 40, 0.0])

    def test_bold_painter_target_no_target_and_board_space_order(self):
        below = self.fresh(seed=23)
        target = _put_unit(below, 1, _card(13001))
        _play(below, self.repository, 10772120)
        _choose(below, target.entity_id)
        self.assertNotIn(target, below.players[1].board)
        self.assertFalse(
            any(unit.definition.card_id in ARTIFACT_IDS for unit in below.players[0].board)
        )

        no_target = self.fresh(seed=29)
        _prime_artifact_history(no_target, self.repository)
        _play(no_target, self.repository, 10772120)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertEqual(
            [
                unit.definition.card_id
                for unit in no_target.players[0].board
                if unit.definition.card_id in ARTIFACT_IDS
            ],
            [90071140, 90071150],
        )

        shortage = self.fresh(seed=31)
        _prime_artifact_history(shortage, self.repository)
        for index in range(3):
            _put_unit(shortage, 0, _card(13100 + index))
        _play(shortage, self.repository, 10772120)
        self.assertEqual(
            [
                unit.definition.card_id
                for unit in shortage.players[0].board
                if unit.definition.card_id in ARTIFACT_IDS
            ],
            [90071140],
        )

    def test_teleport_slash_and_scarlet_use_persistent_dynamic_count(self):
        zero = self.fresh(seed=37)
        zero_target = _put_unit(zero, 1, _card(14001, life=5))
        leader_before = zero.players[1].health
        _play(zero, self.repository, 10773310)
        self.assertEqual(zero_target.health, 5)
        self.assertEqual(zero.players[1].health, leader_before - 1)

        slash = self.fresh(seed=41)
        _prime_artifact_history(slash, self.repository)
        targets = [_put_unit(slash, 1, _card(14100 + i, life=5)) for i in range(2)]
        leader_before = slash.players[1].health
        _play(slash, self.repository, 10773310)
        self.assertEqual([target.health for target in targets], [2, 2])
        self.assertEqual(slash.players[1].health, leader_before - 1)

        scarlet = self.fresh(seed=43)
        _prime_artifact_history(scarlet, self.repository)
        targets = [_put_unit(scarlet, 1, _card(14200 + i, life=6)) for i in range(2)]
        source = _play(scarlet, self.repository, 10774110)
        self.assertEqual([target.health for target in targets], [3, 3])
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertTrue(source.has_keyword("守护"))

    def test_myuu_listener_and_post_evolve_super_threshold(self):
        listener = self.fresh(seed=47)
        source = _play(listener, self.repository, 10774120)
        self.assertFalse(source.has_keyword("疾驰"))
        target = _put_unit(listener, 1, _card(15001, life=10))
        _play(listener, self.repository, 90071130)
        self.assertEqual(target.health, 7)
        duplicate = replace(self.repository.get(90071130), card_id=990071131)
        listener.players[0].mana = 10
        _put_hand(listener, duplicate)
        listener.apply(PlayCard(0, 0))
        self.assertEqual(target.health, 4)
        self.assertEqual(_artifact_count(listener), 1)

        crossing = self.fresh(seed=53)
        _prime_artifact_history(crossing, self.repository, (90071130, 90071150))
        _enable_super_evolution(crossing)
        source = _play(crossing, self.repository, 10774120)
        target = _put_unit(crossing, 1, _card(15002, life=8))
        crossing.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(_artifact_count(crossing), 3)
        self.assertEqual(target.health, 5)
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertTrue(
            any(
                unit.definition.card_id == 90071140
                for unit in crossing.players[0].board
            )
        )

        below = self.fresh(seed=59)
        _prime_artifact_history(below, self.repository, (90071130,))
        _enable_super_evolution(below)
        source = _play(below, self.repository, 10774120)
        below.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(_artifact_count(below), 2)
        self.assertFalse(source.has_keyword("疾驰"))

    def test_journey_ahead_requires_target_and_recovers_ep_at_threshold(self):
        illegal = self.fresh(seed=61)
        card = _put_hand(illegal, self.repository.get(10873310))
        command = PlayCard(0, 0)
        self.assertNotIn(command, illegal.legal_commands())
        before = illegal.deterministic_fingerprint()
        with self.assertRaisesRegex(IllegalCommand, "not currently playable"):
            illegal.apply(command)
        self.assertEqual(illegal.deterministic_fingerprint(), before)
        self.assertIn(card, illegal.players[0].hand)

        below = self.fresh(seed=67)
        below.players[0].evolution_points = 0
        target = _put_unit(below, 1, _card(16001, life=7))
        _play(below, self.repository, 10873310)
        _choose(below, target.entity_id)
        self.assertEqual(target.health, 1)
        self.assertEqual(below.players[0].evolution_points, 0)

        active = self.fresh(seed=71)
        _prime_artifact_history(active, self.repository)
        active.players[0].evolution_points = 0
        target = _put_unit(active, 1, _card(16002, life=5))
        _play(active, self.repository, 10873310)
        _choose(active, target.entity_id)
        self.assertNotIn(target, active.players[1].board)
        self.assertEqual(active.players[0].evolution_points, 1)

        capped = self.fresh(seed=73)
        _prime_artifact_history(capped, self.repository)
        capped.players[0].evolution_points = capped.config.starting_evolution_points
        target = _put_unit(capped, 1, _card(16003, life=7))
        _play(capped, self.repository, 10873310)
        _choose(capped, target.entity_id)
        self.assertEqual(
            capped.players[0].evolution_points,
            capped.config.starting_evolution_points,
        )

    def test_seeded_entry_history_sequence_is_deterministic(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=79)
            _prime_artifact_history(engine, self.repository)
            targets = [_put_unit(engine, 1, _card(17000 + i, life=5)) for i in range(2)]
            _play(engine, self.repository, 10773310)
            self.assertEqual([target.health for target in targets], [2, 2])
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_all_cards_are_exact_and_tokens_list_real_producers(self):
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
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_portal_artifact_entry_history_batch.py"],
                )

        token_report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        tokens = {card["card_id"]: card for card in token_report["cards"]}
        expected_sources = {
            90071130: {10771310},
            90071140: {10771310, 10772120, 10774120},
            90071150: {10772120},
        }
        for token_id, required_sources in expected_sources.items():
            with self.subTest(token_id=token_id):
                self.assertEqual(
                    tokens[token_id]["category"],
                    "entry_behavior_complete",
                )
                authored_sources = {
                    producer["source_card_id"]
                    for producer in tokens[token_id]["authored_producers"]
                }
                self.assertTrue(required_sources <= authored_sources)


if __name__ == "__main__":
    unittest.main()
