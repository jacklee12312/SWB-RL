# -*- coding: utf-8 -*-
"""Exact deck-cost and temporary transformed-token production chain."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import EndTurn, PlayCard
from swb.engine.effects import CostChangeMode, EffectKind, ModifierDuration
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from swb.engine.state import CostModifier, DeckCard
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


SOURCE_ID = 10334120
TOKEN_ID = 90034310
SOURCE_HASHES = {
    SOURCE_ID: "c570136705a80bce9ea0ce38f7d91c22ca5686aee84dfb1ffce2e724a9f37c07",
    TOKEN_ID: "65e6b9ba2e5b79ed6611fbe68b455719fa74a91fca47714d9e0d553a5c12903d",
}


class RealDeckCostTransformTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 3101):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_schema_validation(self):
        source_ops = self.rulebook.operations_for(SOURCE_ID, Trigger.FANFARE)
        self.assertEqual([op.kind for op in source_ops], [
            EffectKind.CHANGE_DECK_COST,
            EffectKind.TRANSFORM,
        ])
        self.assertIs(source_ops[0].mode, CostChangeMode.SUBTRACT)
        self.assertEqual(source_ops[0].deck_filter.card_type, "随从")
        self.assertIs(source_ops[1].mode, CostChangeMode.SET)
        self.assertIs(
            source_ops[1].duration,
            ModifierDuration.UNTIL_END_OF_TURN,
        )
        self.assertEqual(source_ops[1].hand_filter.card_type, "法术")

        invalid = (
            {
                "kind": "change_deck_cost",
                "target": "enemy_leader",
                "amount": 3,
                "mode": "subtract",
            },
            {
                "kind": "change_deck_cost",
                "target": "own_leader",
                "amount": -1,
                "mode": "subtract",
            },
            {
                "kind": "change_deck_cost",
                "target": "own_leader",
                "amount": 1,
                "mode": "subtract",
                "duration": "until_end_of_turn",
            },
            {
                "kind": "transform",
                "target": "enemy_unit",
                "card_id": TOKEN_ID,
                "amount": 0,
                "mode": "set",
            },
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test.json/operations[0]", SOURCE_ID)

    def test_source_modifies_only_current_deck_followers_and_draw_preserves_cost(self):
        engine = self.fresh(seed=3)
        follower = _card(99501, cost=5)
        spell = _card(
            99502,
            cost=2,
            card_type="法术",
            attack=None,
            life=None,
        )
        engine.players[0].deck = [spell, follower]
        _put_hand(engine, _card(
            99503,
            cost=7,
            card_type="法术",
            attack=None,
            life=None,
        ))

        _play(engine, self.repository, SOURCE_ID)

        self.assertIsInstance(engine.players[0].deck[1], DeckCard)
        self.assertEqual(engine.players[0].deck[1].current_cost, 2)
        self.assertNotIsInstance(engine.players[0].deck[0], DeckCard)
        changed = [
            event
            for event in engine.event_history
            if event.type is EventType.DECK_CARD_COST_CHANGED
        ]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0].metadata["target_card_id"], follower.card_id)

        engine._draw(0, reason="测试抽牌")
        drawn = next(card for card in engine.players[0].hand if card.card_id == follower.card_id)
        self.assertEqual(drawn.current_cost, 2)

    def test_deck_changes_stack_clamp_and_do_not_retroactively_affect_new_entries(self):
        engine = self.fresh(seed=5)
        original = _card(99510, cost=4)
        first_spell = _card(
            99511,
            card_type="法术",
            attack=None,
            life=None,
        )
        second_spell = _card(
            99512,
            card_type="法术",
            attack=None,
            life=None,
        )
        engine.players[0].deck = [original]
        _put_hand(engine, first_spell)
        _play(engine, self.repository, SOURCE_ID)
        engine.players[0].mana = 10
        _put_hand(engine, second_spell)
        _play(engine, self.repository, SOURCE_ID)
        self.assertEqual(engine.players[0].deck[0].current_cost, 0)

        later = _card(99513, cost=4)
        engine.players[0].deck.append(later)
        engine._draw(0, reason="后加入抽牌")
        self.assertEqual(engine.players[0].hand[-1].card_id, later.card_id)
        self.assertEqual(engine.players[0].hand[-1].current_cost, 4)
        engine._draw(0, reason="原牌抽牌")
        self.assertEqual(engine.players[0].hand[-1].card_id, original.card_id)
        self.assertEqual(engine.players[0].hand[-1].current_cost, 0)

    def test_random_spell_transform_is_seeded_zero_cost_and_expires_at_turn_end(self):
        outcomes = []
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=7)
            for card_id in (99520, 99521):
                _put_hand(engine, _card(
                    card_id,
                    cost=7,
                    card_type="法术",
                    attack=None,
                    life=None,
                ))
            _put_hand(engine, _card(99522, cost=7))
            _play(engine, self.repository, SOURCE_ID)
            transformed = next(
                card for card in engine.players[0].hand if card.card_id == TOKEN_ID
            )
            untouched_spell = next(
                card
                for card in engine.players[0].hand
                if card.card_type == "法术" and card.card_id != TOKEN_ID
            )
            self.assertEqual(transformed.current_cost, 0)
            self.assertEqual(untouched_spell.current_cost, 7)
            self.assertEqual(
                next(
                    event
                    for event in engine.event_history
                    if event.type is EventType.HAND_CARD_TRANSFORMED
                ).metadata["cost_duration"],
                "until_end_of_turn",
            )
            outcomes.append(untouched_spell.card_id)
            fingerprints.append(engine.deterministic_fingerprint())
            engine.apply(EndTurn(0))
            self.assertEqual(transformed.current_cost, 4)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_no_spell_skips_transform_without_consuming_rng(self):
        engine = self.fresh(seed=11)
        engine.players[0].deck = [_card(99530, cost=6)]
        _put_hand(engine, _card(99531, cost=2))
        before_rng = engine.random.getstate()

        _play(engine, self.repository, SOURCE_ID)

        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertFalse(any(card.card_id == TOKEN_ID for card in engine.players[0].hand))
        self.assertEqual(engine.players[0].deck[0].current_cost, 3)

    def test_generated_spell_buffs_all_hand_followers_then_destroys_enemy_batch(self):
        engine = self.fresh(seed=13)
        follower_a = _put_hand(engine, _card(99540, cost=2))
        follower_b = _put_hand(engine, _card(99541, cost=0))
        other_spell = _put_hand(engine, _card(
            99542,
            cost=3,
            card_type="法术",
            attack=None,
            life=None,
        ))
        enemies = [
            _put_unit(engine, 1, _card(99550 + index, life=5))
            for index in range(2)
        ]

        _play(engine, self.repository, TOKEN_ID)

        self.assertEqual((follower_a.current_cost, follower_b.current_cost), (3, 1))
        self.assertEqual(other_spell.current_cost, 3)
        self.assertTrue(all(enemy not in engine.players[1].board for enemy in enemies))
        destroyed = [
            event.source_id
            for event in engine.event_history
            if event.type is EventType.FOLLOWER_DESTROYED
            and event.source_id in {enemy.entity_id for enemy in enemies}
        ]
        self.assertCountEqual(destroyed, [enemy.entity_id for enemy in enemies])

    def test_illegal_source_play_preserves_complete_state_and_rng(self):
        engine = self.fresh(seed=17)
        _put_hand(engine, self.repository.get(SOURCE_ID))
        engine.players[0].mana = 8
        before = engine.deterministic_fingerprint()
        before_rng = engine.random.getstate()

        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.deterministic_fingerprint(), before)
        self.assertEqual(engine.random.getstate(), before_rng)

    def test_deck_cost_modifier_invariants_reject_corrupt_runtime_state(self):
        engine = self.fresh(seed=18)
        engine.players[0].deck = [DeckCard(
            definition=_card(99560),
            cost_modifiers=[CostModifier(
                modifier_id=0,
                mode="subtract",
                amount=1,
                duration="permanent",
            )],
        )]
        with self.assertRaisesRegex(IllegalCommand, "cost modifier ids"):
            engine.assert_invariants()

    def test_rl_mask_exposes_transformed_zero_cost_spell(self):
        deck = [_card(99600 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
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
        env.players[0].mana = env.players[0].max_mana = 9
        _put_hand(env.core, _card(
            99650,
            cost=8,
            card_type="法术",
            attack=None,
            life=None,
        ))
        _put_hand(env.core, self.repository.get(SOURCE_ID))

        source_action = env._encode_command(PlayCard(0, 0))
        self.assertTrue(env.action_mask()[source_action])
        env.step(source_action)
        self.assertEqual(env.players[0].hand[0].card_id, TOKEN_ID)
        self.assertEqual(env.players[0].hand[0].current_cost, 0)
        token_action = env._encode_command(PlayCard(0, 0))
        self.assertTrue(env.action_mask()[token_action])


class DeckCostTransformDatabaseAuditTests(unittest.TestCase):
    def test_database_text_and_reference_match_reviewed_chain(self):
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            source_text = connection.execute(
                "SELECT text_eng FROM skill_texts WHERE card_id=?",
                (SOURCE_ID,),
            ).fetchone()[0]
            token_text = connection.execute(
                "SELECT text_eng FROM skill_texts WHERE card_id=?",
                (TOKEN_ID,),
            ).fetchone()[0]
            self.assertIn("followers in your deck", re.sub(r"<[^>]+>", "", source_text))
            self.assertIn("random spell in your hand", re.sub(r"<[^>]+>", "", source_text))
            self.assertIn("followers in your hand", token_text)
            self.assertIn("Destroy all enemy followers", token_text)
            self.assertEqual(
                connection.execute(
                    "SELECT referenced_card_id FROM card_references WHERE card_id=?",
                    (SOURCE_ID,),
                ).fetchall(),
                [(TOKEN_ID,)],
            )

    def test_both_cards_are_exact_and_token_has_real_executable_producer(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in (SOURCE_ID, TOKEN_ID):
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                if card_id == SOURCE_ID:
                    self.assertEqual(info["coverage"], "covered_exact")
                    self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )

        audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        token = next(card for card in audit["cards"] if card["card_id"] == TOKEN_ID)
        self.assertEqual(token["category"], "entry_behavior_complete")
        self.assertEqual(token["explicit_coverage"], "exact")
        self.assertEqual(
            token["authored_producers"],
            [{
                "source_card_id": SOURCE_ID,
                "entry_kind": "transform",
                "rule_file": "real_deck_cost_transform_token_batch.json",
                "rule_group": "rules",
            }],
        )


if __name__ == "__main__":
    unittest.main()
