# -*- coding: utf-8 -*-
"""Direct contracts for the nineteenth listener-context/leader-runtime slice."""

from __future__ import annotations

import json
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
from swb.engine.card_rules import (
    RuleBook,
    Trigger,
    _parse_condition,
    _parse_operation,
)
from swb.engine.commands import (
    ActivateAmulet,
    Choose,
    EndTurn,
    PlayCard,
    SuperEvolve,
)
from swb.engine.effects import (
    ConditionType,
    EffectKind,
    LeaderDamageMode,
    TargetKind,
)
from swb.engine.emblem import EmblemDefinition
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import DamageType, IllegalCommand
from swb.engine.state import CostModifier, LeaderDamageModifier
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10362210, 10444120, 10503210, 10604110, 10703210)
SOURCE_HASHES = {
    10362210: "919d48af5a6d173302bed974fc634b2df0d009a466b6e0b5a458b4b4cf602035",
    10444120: "c4924ffe583de57e289533931d1738ea9a501786f0f28aa07e623b7a5078389a",
    10503210: "775dbee15d2e36c3d7cfb48e8c2dbda73c6635bd3703544ce6ca27a23fb546cd",
    10604110: "dadb4df04d040e637d701b0762558fe80c0dc4ab5285da511a9e6711c881a9cb",
    10703210: "a9d5cacca4c0fa1243aa55d09a8de6db48f5d1b1f2265b88dc5388379388132b",
}
TEST_EVIDENCE = (
    "tests/test_real_listener_context_leader_runtime_nineteenth_batch.py"
)


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_super_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = (
        engine.config.first_player_super_evolution_unlock_turn
    )
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False


class ListenerContextLeaderRuntimeNineteenthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 19001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_generic_schema_are_strict(self):
        temple_last_words = self.rulebook.operations_for(
            10362210,
            Trigger.LAST_WORDS,
        )
        self.assertEqual(
            [operation.kind for operation in temple_last_words],
            [EffectKind.HEAL_LEADER, EffectKind.ADD_LEADER_BARRIER],
        )
        self.assertEqual(self.rulebook.countdown_for(10362210), 4)

        zooey_mode = self.rulebook.modes_for(10444120)[0]
        self.assertTrue(zooey_mode.is_enhance)
        replacement = zooey_mode.operations[-1]
        self.assertIs(
            replacement.leader_damage_mode,
            LeaderDamageMode.SET_ZERO_IF_POSITIVE,
        )

        world_listener = self.rulebook.listeners_for(10503210)[0]
        world_condition = world_listener.operations[0].conditions[0]
        self.assertIs(
            world_condition.type,
            ConditionType.BOARD_HAS_OTHER_CARD_WITH_EVENT_SOURCE_BASE_COST,
        )

        omegotep = self.rulebook.operations_for(
            10604110,
            Trigger.FANFARE,
        )[0]
        self.assertIs(omegotep.kind, EffectKind.RANDOM_CHOICE)
        self.assertEqual((omegotep.amount, len(omegotep.random_choice_options)), (2, 4))
        self.assertIs(
            omegotep.random_choice_options[-1].operations[-1].kind,
            EffectKind.REPLAY_SOURCE_FANFARE,
        )

        babelon_listener = self.rulebook.listeners_for(10703210)[0]
        self.assertEqual(babelon_listener.max_activations, 3)
        self.assertEqual(
            [
                operation.conditions[0].value
                for operation in babelon_listener.operations
            ],
            [1, 2, 3],
        )

        for raw, message in (
            (
                {
                    "kind": "add_leader_barrier",
                    "target": "own_leader",
                    "amount": 0,
                },
                "positive integer",
            ),
            (
                {
                    "kind": "add_leader_barrier",
                    "target": "own_leader",
                    "amount": 1,
                    "duration": "permanent",
                },
                "unknown fields",
            ),
            (
                {
                    "kind": "add_leader_damage_modifier",
                    "target": "own_leader",
                    "amount": 1,
                    "damage_mode": "set_zero_if_positive",
                },
                "requires amount 0",
            ),
            (
                {
                    "kind": "add_leader_damage_modifier",
                    "target": "own_leader",
                    "amount": 0,
                    "damage_mode": "unknown",
                },
                "invalid leader damage mode",
            ),
            (
                {
                    "kind": "replay_source_fanfare",
                    "target": "own_leader",
                },
                "requires target 'self'",
            ),
            (
                {
                    "kind": "replay_source_fanfare",
                    "target": "self",
                    "amount": 1,
                },
                "unknown fields",
            ),
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, message):
                    _parse_operation(raw, "test.json", 1)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            _parse_condition(
                {"type": "listener_activation_count_equals", "value": 0},
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "does not accept a value"):
            _parse_condition(
                {
                    "type": "board_has_other_card_with_event_source_base_cost",
                    "value": 1,
                },
                "test.json",
                1,
            )

    def test_temple_engage_uses_crest_count_and_leader_barrier_is_consumed_once(self):
        engine = self.fresh(seed=11)
        engine.players[0].health = 15
        temple = _play(engine, self.repository, 10362210)
        for index in range(4):
            engine._add_emblem_to_player(
                0,
                EmblemDefinition(f"test_crest_{index}", 990000 + index),
                source_card_id=990000 + index,
            )

        before = engine.deterministic_fingerprint()
        engine.apply(ActivateAmulet(0, temple.entity_id))
        self.assertNotEqual(before, engine.deterministic_fingerprint())
        self.assertNotIn(temple, engine.players[0].board)
        self.assertEqual(engine.players[0].health, 17)
        self.assertEqual(engine.players[0].leader_barrier_charges, 1)
        self.assertTrue(any(
            event.type is EventType.LEADER_BARRIER_GRANTED
            and event.metadata["target_player"] == 0
            for event in engine.event_history
        ))

        first = engine.apply_damage(
            None,
            None,
            3,
            DamageType.EFFECT,
            1,
            target_player_index=0,
        )
        second = engine.apply_damage(
            None,
            None,
            3,
            DamageType.EFFECT,
            1,
            target_player_index=0,
        )
        self.assertEqual(
            (
                first.actual_amount,
                first.prevented_amount,
                first.barrier_consumed,
                second.actual_amount,
                engine.players[0].health,
            ),
            (0, 3, True, 3, 14),
        )

    def test_zooey_enhance_replacement_precedes_barrier_and_expires(self):
        engine = self.fresh(seed=13)
        engine.players[0].leader_barrier_charges = 1
        zooey = _play(
            engine,
            self.repository,
            10444120,
            mode_id="enhance_10",
        )
        self.assertTrue(zooey.has_keyword("疾驰"))
        self.assertEqual(
            (engine.players[0].max_health, engine.players[0].health),
            (1, 1),
        )
        self.assertEqual(
            engine.players[0].leader_damage_modifiers[0].mode,
            LeaderDamageMode.SET_ZERO_IF_POSITIVE.value,
        )

        prevented = engine.apply_damage(
            None,
            None,
            7,
            DamageType.EFFECT,
            1,
            target_player_index=0,
        )
        self.assertEqual((prevented.actual_amount, prevented.prevented_amount), (0, 7))
        self.assertEqual(engine.players[0].leader_barrier_charges, 1)

        engine.apply(EndTurn(0))
        self.assertEqual(len(engine.players[0].leader_damage_modifiers), 1)
        engine.apply(EndTurn(1))
        self.assertEqual(engine.players[0].leader_damage_modifiers, [])
        barrier = engine.apply_damage(
            None,
            None,
            1,
            DamageType.EFFECT,
            1,
            target_player_index=0,
        )
        self.assertTrue(barrier.barrier_consumed)
        self.assertEqual(engine.players[0].health, 1)

    def test_world_of_games_uses_frozen_base_cost_and_respects_hand_capacity(self):
        engine = self.fresh(seed=17)
        world = _play(engine, self.repository, 10503210)
        world.countdown = 1
        _put_unit(engine, 1, _card(990100, cost=3, life=5))
        for index in range(8):
            _put_hand(engine, _card(990110 + index, cost=4))
        played = _put_hand(engine, _card(990120, cost=3))
        played.cost_modifiers.append(CostModifier(
            modifier_id=990120,
            mode="set",
            amount=1,
            duration="permanent",
        ))
        draws_before = sum(
            event.type is EventType.CARD_DRAWN
            and event.player_index == 0
            for event in engine.event_history
        )
        engine.apply(PlayCard(0, engine.players[0].hand.index(played)))

        self.assertNotIn(world, engine.players[0].board)
        self.assertEqual(len(engine.players[0].hand), engine.config.max_hand)
        self.assertEqual(
            sum(
                event.type is EventType.CARD_DRAWN
                and event.player_index == 0
                for event in engine.event_history
            ) - draws_before,
            1,
        )

        no_match = self.fresh(seed=19)
        world = _play(no_match, self.repository, 10503210)
        played = _put_hand(no_match, _card(990130, cost=6))
        no_match.apply(PlayCard(0, no_match.players[0].hand.index(played)))
        self.assertEqual(world.countdown, 5)

    def test_omegotep_selects_distinct_options_replays_and_is_seeded(self):
        fingerprints = []
        event_options = []
        for _ in range(2):
            engine = self.fresh(seed=23)
            omegotep = _play(engine, self.repository, 10604110)
            _enable_super_evolution(engine)
            engine.apply(SuperEvolve(0, omegotep.entity_id))
            choices = [
                event.metadata["option_ids"]
                for event in engine.event_history
                if event.type is EventType.RANDOM_CHOICES_SELECTED
                and event.metadata.get("source_card_id") == 10604110
            ]
            self.assertGreaterEqual(len(choices), 2)
            self.assertTrue(all(
                len(option_ids) == len(set(option_ids)) == 2
                for option_ids in choices
            ))
            fingerprints.append(engine.deterministic_fingerprint())
            event_options.append(choices)
        self.assertEqual(fingerprints[0], fingerprints[1])
        self.assertEqual(event_options[0], event_options[1])

        source_left = self.fresh(seed=29)
        source = _put_unit(
            source_left,
            0,
            self.repository.get(10604110),
        )
        target = _put_unit(source_left, 1, _card(990140, life=5))
        source_left._start_effects(
            source.definition,
            source.entity_id,
            (
                _parse_operation(
                    {
                        "kind": "select_targets",
                        "target": "enemy_unit",
                        "target_key": "pause_for_source_leave",
                        "requires_target": True,
                    },
                    "test.json",
                    10604110,
                ),
                _parse_operation(
                    {"kind": "replay_source_fanfare", "target": "self"},
                    "test.json",
                    10604110,
                ),
            ),
        )
        source_left.players[0].board.remove(source)
        _choose_entity(source_left, target.entity_id)
        self.assertFalse(any(
            event.type is EventType.RANDOM_CHOICES_SELECTED
            for event in source_left.event_history
        ))

    def test_babelon_engage_choice_mask_atomicity_and_three_step_sequence(self):
        engine = self.fresh(seed=31)
        engine.players[0].health = 15
        victim = _put_unit(engine, 1, _card(990150, life=8))
        babelon = _play(engine, self.repository, 10703210)

        for expected_count in (1, 2, 3):
            discard = _put_hand(engine, _card(990160 + expected_count))
            command = ActivateAmulet(0, babelon.entity_id)
            self.assertIn(command, engine.legal_commands())
            engine.apply(command)
            before = engine.deterministic_fingerprint()
            with self.assertRaises(IllegalCommand):
                engine.apply(Choose(0, "hand:999999"))
            self.assertEqual(engine.deterministic_fingerprint(), before)
            _choose_entity(engine, discard.entity_id)
            self.assertEqual(babelon.countdown, 2)
            engine.apply(EndTurn(0))
            if expected_count == 1:
                self.assertEqual(victim.health, 6)
            elif expected_count == 2:
                self.assertEqual(engine.players[0].health, 17)
            else:
                self.assertEqual(engine.players[1].health, 18)
                self.assertNotIn(babelon, engine.players[0].board)
                break
            engine.apply(EndTurn(1))
            self.assertEqual(babelon.countdown, 1)

        trigger_events = [
            event
            for event in engine.event_history
            if event.type is EventType.CARD_LISTENER_TRIGGERED
            and event.metadata.get("listener_card_id") == 10703210
        ]
        self.assertEqual(
            [event.metadata["activation_count"] for event in trigger_events],
            [1, 2, 3],
        )

        deck_a = [_card(990400 + index) for index in range(40)]
        deck_b = [_card(990500 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=33,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=33)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        babelon = _play(env.core, self.repository, 10703210)
        _put_hand(env.core, _card(990600))
        env.invalidate_cache(reason="nineteenth activation mask")
        command = ActivateAmulet(0, babelon.entity_id)
        self.assertTrue(env.action_mask()[env._encode_command(command)])
        self.assertEqual(
            {
                env._decode_action(index)
                for index, allowed in enumerate(env.action_mask())
                if allowed
            },
            set(env.core.legal_commands()),
        )

    def test_v2_v3_observations_expose_public_leader_runtime_without_action_migration(self):
        deck_a = [_card(990200 + index) for index in range(40)]
        deck_b = [_card(990300 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=37,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            observation_version="v2",
            validate_invariants=True,
        )
        env.reset(seed=37)
        env.players[0].leader_barrier_charges = 2
        env.players[0].leader_damage_modifiers.append(LeaderDamageModifier(
            modifier_id=1,
            amount=0,
            duration="permanent",
            mode=LeaderDamageMode.SET_ZERO_IF_POSITIVE.value,
        ))
        observation = env.observation()
        self.assertEqual(
            observation["leader_area"]["leader_barrier_charges"],
            (2, 0),
        )
        self.assertEqual(
            env.observation_v2_spec()["leader_damage_modifier_runtime"],
            96,
        )
        self.assertEqual(env.observation_v2_spec()["leader_barrier_charges"], 2)
        self.assertEqual(
            observation["leader_area"]["leader_damage_modifier_runtime"][5],
            1.0,
        )

        env.observation_version = "v3"
        env.invalidate_cache(reason="nineteenth observation schema")
        structured = env.observation()
        self.assertTrue(env.observation_v3_space().contains(structured))
        self.assertEqual(env.ACTION_SIZE, 111)


class ListenerContextLeaderRuntimeNineteenthAuditTests(unittest.TestCase):
    def test_database_multilingual_modes_references_and_raw_json(self):
        expected_stats = {
            10362210: (10003, 6, 2, 3, None, None),
            10444120: (10004, 4, 1, 5, 5, 5),
            10503210: (10005, 0, 2, 1, None, None),
            10604110: (10006, 0, 1, 9, 4, 4),
            10703210: (10007, 0, 2, 1, None, None),
        }
        expected_english = {
            10362210: "number of crests you have",
            10444120: "Can't take more than 0 damage",
            10503210: "same base cost",
            10604110: "Activate 2 random abilities",
            10703210: "activate an ability in sequence",
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            source_map = _load_source_text_map(connection)
            for card_id in CARD_IDS:
                with self.subTest(card_id=card_id):
                    self.assertEqual(
                        connection.execute(
                            """
                            SELECT card_set_id, class_id, type_id, cost,
                                   attack, life
                            FROM cards WHERE card_id=?
                            """,
                            (card_id,),
                        ).fetchone(),
                        expected_stats[card_id],
                    )
                    self.assertEqual(
                        _source_text_sha256(source_map[card_id]),
                        SOURCE_HASHES[card_id],
                    )
                    texts = connection.execute(
                        """
                        SELECT text_chs, text_eng, text_jpn, text_kor, text_cht
                        FROM skill_texts WHERE card_id=? ORDER BY position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertTrue(texts)
                    self.assertTrue(all(all(row) for row in texts))
                    self.assertIn(expected_english[card_id], texts[0][1])
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
                    self.assertEqual(
                        connection.execute(
                            """
                            SELECT COUNT(*) FROM card_references
                            WHERE card_id=? AND referenced_card_id IS NOT NULL
                            """,
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
                    raw = json.loads(connection.execute(
                        "SELECT raw_json FROM cards WHERE card_id=?",
                        (card_id,),
                    ).fetchone()[0])
                    self.assertEqual(raw["card_id"], card_id)
                    self.assertEqual(raw["alt_modes"], [])

    def test_all_five_cards_are_exact_with_clause_and_token_audits(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertEqual(
            report["summary"]["coverage_counts"],
            {
                "covered_exact": 735,
                "supported_missing_rule": 0,
                "token_or_non_collectible": 91,
            },
        )
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(
                    info["clause_audit"]["status"],
                    "mapped_exact",
                )
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    [TEST_EVIDENCE],
                )

        token_audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        self.assertEqual(
            token_audit["summary"]["categories"]["entry_behavior_complete"],
            91,
        )


if __name__ == "__main__":
    unittest.main()
