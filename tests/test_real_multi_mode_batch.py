# -*- coding: utf-8 -*-
"""Multi-mode decisions and exact coverage for Departure for the Journey."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, ChoiceKind, PlayCard
from swb.engine.effects import EffectKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _put_hand,
    _put_unit,
)


CARD_ID = 10852310
SOURCE_HASH = "d60f335eebff188c3c72e355d6132a6c5253a1b05273d420947bc3cc6f7f3f7f"


def _choose(engine, option_id: str) -> None:
    request = engine.state.pending_choice
    engine.apply(Choose(request.player_index, f"choose_one:{option_id}"))


class RealMultiModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2701):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def play(self, engine) -> None:
        _put_hand(engine, self.repository.get(CARD_ID))
        engine.apply(PlayCard(0, 0))

    def test_rule_shape_has_four_modes_and_requires_two(self):
        operation = self.rulebook.operations_for(CARD_ID, Trigger.PLAY)[0]

        self.assertIs(operation.kind, EffectKind.CHOOSE_ONE)
        self.assertEqual(operation.choose_count, 2)
        self.assertEqual(
            [option.option_id for option in operation.choose_one_options],
            [
                "damage_leader",
                "heal_leader",
                "damage_follower",
                "gain_shadows",
            ],
        )
        self.assertEqual(
            [option.operations[0].kind for option in operation.choose_one_options],
            [
                EffectKind.DAMAGE_LEADER,
                EffectKind.HEAL_LEADER,
                EffectKind.DAMAGE_UNIT,
                EffectKind.ADD_SHADOWS,
            ],
        )

    def test_first_mode_waits_and_duplicate_choice_is_atomic(self):
        engine = self.fresh(seed=3)
        engine.players[0].health = 15
        self.play(engine)
        request = engine.state.pending_choice

        self.assertIs(request.choice_kind, ChoiceKind.MODE)
        self.assertEqual(request.target_count, 2)
        self.assertEqual(len(request.options), 4)
        _choose(engine, "heal_leader")

        self.assertEqual(engine.players[0].health, 15)
        self.assertEqual(len(engine.state.pending_choice.selected_options), 1)
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            _choose(engine, "heal_leader")
        self.assertEqual(engine.deterministic_fingerprint(), before)

        _choose(engine, "damage_leader")
        self.assertEqual((engine.players[0].health, engine.players[1].health), (17, 19))

    def test_all_six_mode_pairs_apply_exact_effects_and_spell_shadow(self):
        pairs = {
            ("damage_leader", "heal_leader"): (19, 17, 4, 1),
            ("damage_leader", "damage_follower"): (19, 15, 1, 1),
            ("damage_leader", "gain_shadows"): (19, 15, 4, 5),
            ("heal_leader", "damage_follower"): (20, 17, 1, 1),
            ("heal_leader", "gain_shadows"): (20, 17, 4, 5),
            ("damage_follower", "gain_shadows"): (20, 15, 1, 5),
        }
        for index, (pair, expected) in enumerate(pairs.items()):
            with self.subTest(pair=pair):
                engine = self.fresh(seed=10 + index)
                engine.players[0].health = 15
                enemy = _put_unit(engine, 1, _card(99710 + index, life=4))
                self.play(engine)
                _choose(engine, pair[1])
                _choose(engine, pair[0])

                self.assertIsNone(engine.state.pending_choice)
                self.assertEqual(
                    (
                        engine.players[1].health,
                        engine.players[0].health,
                        enemy.health if enemy in engine.players[1].board else 0,
                        engine.players[0].shadows,
                    ),
                    expected,
                )
                self.assertEqual(
                    sum(
                        card.definition.card_id == CARD_ID
                        for card in engine.players[0].graveyard
                    ),
                    1,
                )

    def test_follower_mode_without_enemy_is_selectable_safe_and_rng_neutral(self):
        engine = self.fresh(seed=23)
        self.play(engine)
        request = engine.state.pending_choice
        self.assertIn(
            "choose_one:damage_follower",
            {option.option_id for option in request.options},
        )
        before_rng = engine.random.getstate()

        _choose(engine, "damage_follower")
        _choose(engine, "gain_shadows")

        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertEqual(engine.players[0].shadows, 5)
        self.assertFalse(any(
            event.type is EventType.DAMAGE_APPLIED
            for event in engine.event_history
        ))

    def test_seeded_random_mode_and_fingerprint_replay_match(self):
        outcomes = []
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=29)
            enemies = [
                _put_unit(engine, 1, _card(99810 + index, life=8))
                for index in range(2)
            ]
            self.play(engine)
            _choose(engine, "gain_shadows")
            _choose(engine, "damage_follower")
            outcomes.append(tuple(enemy.health for enemy in enemies))
            fingerprints.append(engine.deterministic_fingerprint())

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(sum(8 - health for health in outcomes[0]), 3)
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_rl_mask_and_observation_track_both_mode_selections(self):
        deck = [_card(100300 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=31,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=31)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        _put_hand(env.core, self.repository.get(CARD_ID))
        play_action = env._encode_command(PlayCard(0, 0))

        self.assertTrue(env.action_mask()[play_action])
        env.step(play_action)
        first_actions = [
            action
            for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        ]
        self.assertEqual(len(first_actions), 4)
        self.assertEqual(env.observation()[-10:-8], [2 / 16, 0.0])

        env.step(first_actions[0])
        second_actions = [
            action
            for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        ]
        self.assertEqual(len(second_actions), 3)
        self.assertEqual(env.observation()[-10:-8], [2 / 16, 0.5])
        env.step(second_actions[0])

        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual(env.players[1].health, 19)


class MultiModeDatabaseAuditTests(unittest.TestCase):
    def test_database_text_and_references_match_reviewed_source(self):
        phrases = (
            "Select 2 Modes",
            "Deal 1 damage to the enemy leader",
            "Restore 2 defense to your leader",
            "random enemy follower",
            "Gain 4 shadows",
        )
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            rows = connection.execute(
                "SELECT text_eng FROM skill_texts WHERE card_id=? ORDER BY position",
                (CARD_ID,),
            ).fetchall()
            self.assertEqual(len(rows), 1)
            normalized = re.sub(r"<[^>]+>", "", rows[0][0])
            for phrase in phrases:
                self.assertIn(phrase, normalized)
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM card_references WHERE card_id=?",
                    (CARD_ID,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                    (CARD_ID,),
                ).fetchone()[0],
                0,
            )

    def test_card_has_exact_mapped_clause_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"][str(CARD_ID)]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
        self.assertEqual(info["clause_audit"]["source_text_sha256"], SOURCE_HASH)
        self.assertEqual(
            info["clause_audit"]["test_evidence"],
            ["tests/test_real_multi_mode_batch.py"],
        )


if __name__ == "__main__":
    unittest.main()
