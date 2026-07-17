# -*- coding: utf-8 -*-
"""Exact hand-follower stat buffs for Noah, Disdain, and Fleeting Flash."""

from __future__ import annotations

import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import EndTurn, PlayCard
from swb.engine.effects import ConditionType, EffectKind, HandFilter, TargetKind
from swb.engine.events import EventType
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10172130, 10343110, 10772310)
SOURCE_HASHES = {
    10172130: "2c2e09319d08af62e7cf23eb5de57050abcc8ac5f2e9f21b5442a112ecd74432",
    10343110: "256c97edb4ae0ac5c027f883cde93813987bb2d32dfa68bc9f205bd910e9d5e5",
    10772310: "0bdeb564f0b10662107dd180824f84d34d824b225d16c728d13f0b6b5b928675",
}


class HandFollowerBuffSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")

    def test_real_operations_use_auditable_hand_filters_and_source_health(self):
        noah = self.rulebook.operations_for(10172130, Trigger.FANFARE)
        self.assertEqual([operation.kind for operation in noah[:3]], [
            EffectKind.ADD_CARD,
            EffectKind.ADD_CARD,
            EffectKind.ADD_CARD,
        ])
        self.assertEqual(noah[-1].kind, EffectKind.BUFF_HAND_CARD)
        self.assertEqual(noah[-1].target, TargetKind.ALL_OWN_HAND)
        self.assertEqual(
            noah[-1].hand_filter,
            HandFilter(card_type="随从", tribe_id=15),
        )

        disdain = self.rulebook.operations_for(10343110, Trigger.TURN_END)[0]
        self.assertEqual(disdain.kind, EffectKind.BUFF_HAND_CARD)
        self.assertEqual(disdain.hand_filter, HandFilter(card_type="随从", class_id=4))
        self.assertEqual(disdain.conditions[0].type, ConditionType.SOURCE_HEALTH_AT_MOST)
        self.assertEqual(disdain.conditions[0].value, 3)

        flash = self.rulebook.operations_for(10772310, Trigger.PLAY)
        self.assertEqual(
            [(operation.kind, operation.target) for operation in flash],
            [
                (EffectKind.BUFF_UNIT, TargetKind.ALL_OWN_UNITS),
                (EffectKind.BUFF_HAND_CARD, TargetKind.ALL_OWN_HAND),
            ],
        )


class RealHandFollowerBuffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2301):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_noah_adds_three_puppets_then_buffs_all_puppetry_followers(self):
        engine = self.fresh(seed=3)
        preexisting = replace(
            _card(9101, attack=3, life=4),
            tribe_id=15,
            tribe_name="人偶",
        )
        outsider = replace(
            _card(9102, attack=2, life=5),
            tribe_id=16,
            tribe_name="创造物",
        )
        _put_hand(engine, outsider)
        existing_card = _put_hand(engine, preexisting)
        source = _play(engine, self.repository, 10172130)

        puppetry = [card for card in engine.players[0].hand if card.definition.tribe_id == 15]
        generated = [card for card in puppetry if card.card_id == 90071110]
        self.assertEqual(len(generated), 3)
        self.assertEqual((existing_card.attack, existing_card.life), (4, 4))
        self.assertTrue(all((card.attack, card.life) == (2, 1) for card in generated))
        other = next(card for card in engine.players[0].hand if card.card_id == 9102)
        self.assertEqual((other.attack, other.life), (2, 5))
        definition = self.repository.get(10172130)
        self.assertEqual((source.attack, source.max_health), (definition.attack, definition.life))

        puppet_index = engine.players[0].hand.index(generated[0])
        engine.apply(PlayCard(0, puppet_index))
        played = next(
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == 90071110
        )
        self.assertEqual((played.attack, played.health, played.max_health), (2, 1, 1))

    def test_disdain_checks_live_source_health_and_only_buffs_dragoncraft(self):
        inactive = self.fresh(seed=5)
        dragon = _put_hand(inactive, _card(9201, class_id=4, class_name="龙族"))
        source = _play(inactive, self.repository, 10343110)
        self.assertEqual(source.health, 4)
        inactive.apply(EndTurn(0))
        self.assertEqual((dragon.attack, dragon.life), (1, 5))

        active = self.fresh(seed=7)
        neutral = _put_hand(active, _card(9202, class_id=0, class_name="中立"))
        dragon = _put_hand(active, _card(9203, class_id=4, class_name="龙族"))
        source = _play(active, self.repository, 10343110)
        source.health = 3
        active.apply(EndTurn(0))
        self.assertEqual((dragon.attack, dragon.life), (2, 6))
        self.assertEqual((neutral.attack, neutral.life), (1, 5))

    def test_disdain_does_not_trigger_after_source_leaves_play(self):
        engine = self.fresh(seed=11)
        dragon = _put_hand(engine, _card(9301, class_id=4, class_name="龙族"))
        source = _play(engine, self.repository, 10343110)
        source.health = 0
        engine._stabilize()
        self.assertNotIn(source, engine.players[0].board)
        engine.apply(EndTurn(0))
        self.assertEqual((dragon.attack, dragon.life), (1, 5))

    def test_fleeting_flash_always_buffs_board_and_gates_hand_on_unlock(self):
        locked = self.fresh(seed=13)
        ally = _put_unit(locked, 0, _card(9401, attack=2, life=4))
        hand = _put_hand(locked, _card(9402, attack=3, life=5))
        _play(locked, self.repository, 10772310)
        self.assertEqual((ally.attack, ally.max_health), (3, 4))
        self.assertEqual((hand.attack, hand.life), (3, 5))

        unlocked = self.fresh(seed=17)
        unlocked.players[0].turns_started = (
            unlocked.config.first_player_super_evolution_unlock_turn
        )
        ally = _put_unit(unlocked, 0, _card(9403, attack=2, life=4))
        hand = _put_hand(unlocked, _card(9404, attack=3, life=5))
        spell = _put_hand(
            unlocked,
            _card(9405, card_type="法术", attack=None, life=None),
        )
        _play(unlocked, self.repository, 10772310)
        self.assertEqual((ally.attack, ally.max_health), (3, 4))
        self.assertEqual((hand.attack, hand.life), (4, 5))
        self.assertEqual((spell.attack, spell.life), (None, None))
        events = [
            event
            for event in unlocked.event_history
            if event.type is EventType.HAND_FOLLOWER_STATS_INCREASED
        ]
        self.assertEqual([event.target_id for event in events], [hand.entity_id])

    def test_seeded_replay_includes_hand_modifier_state(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=19)
            engine.players[0].turns_started = (
                engine.config.first_player_super_evolution_unlock_turn
            )
            _put_hand(engine, _card(9501))
            _play(engine, self.repository, 10772310)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])
        hand_state = fingerprints[0]["state"]["players"][0]["hand"][0]
        self.assertEqual(len(hand_state["stat_modifiers"]), 1)


class HandFollowerBuffDatabaseAuditTests(unittest.TestCase):
    def test_database_text_and_references_match_reviewed_sources(self):
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            expected_clauses = {
                10172130: ("Add 3 copies", "Puppetry followers"),
                10343110: ("Deal 3 damage to all followers", "defense is 3 or less"),
                10772310: ("all allied followers", "unlocked super-evolution"),
            }
            for card_id, clauses in expected_clauses.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        "SELECT text_eng FROM skill_texts WHERE card_id=? ORDER BY position",
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(len(row), 1)
                    for clause in clauses:
                        self.assertIn(clause, row[0][0])
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
            self.assertEqual(
                connection.execute(
                    "SELECT referenced_card_id FROM card_references WHERE card_id=10172130"
                ).fetchall(),
                [(90071110,)],
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
                    ["tests/test_real_hand_follower_buff_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
