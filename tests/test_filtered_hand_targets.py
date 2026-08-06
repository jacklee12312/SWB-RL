# -*- coding: utf-8 -*-
"""Audits for filtered hand targeting and Illusory Conjuration."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard
from swb.engine.effects import (
    CostChangeMode,
    EffectKind,
    EffectOperation,
    HandFilter,
    ModifierDuration,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


REAL_CARD_ID = 10333310
REAL_TEXT = "选择自己的手牌中的1张随从，使其费用+1。破坏对手的战场上的随机1个随从。"


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 3),
        class_name=overrides.get("class_name", "巫师"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 1),
        keywords=frozenset(),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
        tribe_id=overrides.get("tribe_id", 0),
        tribe_name=overrides.get("tribe_name", ""),
    )


def _spell(card_id: int, *, cost: int = 1, name: str | None = None) -> CardDefinition:
    return _card(
        card_id,
        cost=cost,
        card_type="法术",
        attack=None,
        life=None,
        name=name or f"spell-{card_id}",
    )


def _engine(rulebook: RuleBook, *, seed: int = 17) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=3,
        class_b=3,
        seed=seed,
        rulebook=rulebook,
    )
    engine.reset(seed=seed)
    engine.players[0].mana = 10
    return engine


def _set_hand(engine: GameEngine, *definitions: CardDefinition) -> list[HandCard]:
    cards = [
        HandCard(
            definition=definition,
            entity_id=engine.state.allocate_entity_id(),
        )
        for definition in definitions
    ]
    engine.players[0].hand = cards
    engine.players[0].hand_entity_ids = [card.entity_id for card in cards]
    return cards


def _add_enemy_unit(engine: GameEngine, card_id: int) -> Unit:
    unit = Unit.summon(
        _card(card_id, life=4),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[1].board.append(unit)
    return unit


class FilteredHandSchemaTests(unittest.TestCase):
    def _load_operation(self, operation: dict) -> EffectOperation:
        payload = {
            "rules": [
                {
                    "card_id": 999001,
                    "trigger": "play",
                    "operations": [operation],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "rules.json").write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            return RuleBook.from_directory(tmp).operations_for(
                999001, Trigger.PLAY
            )[0]

    def test_schema_loads_all_printed_definition_filter_fields(self):
        operation = self._load_operation(
            {
                "kind": "change_cost",
                "target": "own_hand",
                "amount": 1,
                "hand_filter": {
                    "card_type": "随从",
                    "class_id": 3,
                    "class_name": "巫师",
                    "cost_min": 2,
                    "cost_max": 6,
                    "card_id": 123,
                    "card_name": "目标",
                    "tribe_id": 4,
                    "tribe_name": "测试种族",
                },
            }
        )
        self.assertEqual(
            operation.hand_filter,
            HandFilter(
                card_type="随从",
                class_id=3,
                class_name="巫师",
                cost_min=2,
                cost_max=6,
                card_id=123,
                card_name="目标",
                tribe_id=4,
                tribe_name="测试种族",
            ),
        )

    def test_schema_rejects_invalid_or_misplaced_hand_filters(self):
        cases = (
            ([], "must be an object"),
            ({"card_type": "主战者"}, "unknown card type"),
            ({"cost_min": 5, "cost_max": 2}, "must not exceed"),
            ({"unknown": 1}, "unknown fields"),
        )
        for hand_filter, message in cases:
            with self.subTest(hand_filter=hand_filter):
                with self.assertRaisesRegex(ValueError, message):
                    self._load_operation(
                        {
                            "kind": "change_cost",
                            "target": "own_hand",
                            "amount": 1,
                            "hand_filter": hand_filter,
                        }
                    )
        with self.assertRaisesRegex(ValueError, "requires a hand target"):
            self._load_operation(
                {
                    "kind": "destroy",
                    "target": "enemy_unit",
                    "hand_filter": {"card_type": "随从"},
                }
            )


class DatabaseClauseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_text_and_absent_modes_or_references_match_audit(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            row = connection.execute(
                """
                SELECT cl.class_name, cl.type_name, c.cost, st.text_chs,
                       st.text_eng, st.text_jpn
                FROM cards c
                JOIN card_localizations cl
                  ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                JOIN skill_texts st ON st.card_id = c.card_id
                WHERE c.card_id = ?
                ORDER BY st.position
                """,
                (REAL_CARD_ID,),
            ).fetchall()
            self.assertEqual(len(row), 1)
            self.assertEqual(row[0][:4], ("巫师", "法术", 2, REAL_TEXT))
            self.assertIn("follower in your hand", row[0][4])
            self.assertIn("手札のフォロワー1枚", row[0][5])
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                    (REAL_CARD_ID,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM card_references WHERE card_id = ?",
                    (REAL_CARD_ID,),
                ).fetchone()[0],
                0,
            )

    def test_card_has_exact_mapped_clause_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"][str(REAL_CARD_ID)]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
        self.assertEqual(
            info["clause_audit"]["test_evidence"],
            ["tests/test_filtered_hand_targets.py"],
        )


class FilteredHandBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")

    def fresh_engine(self, *, seed: int = 17) -> GameEngine:
        return _engine(self.rulebook, seed=seed)

    def test_real_rule_structure_is_exact(self):
        operations = self.rulebook.operations_for(REAL_CARD_ID, Trigger.PLAY)
        self.assertEqual(len(operations), 2)
        self.assertEqual(operations[0].kind, EffectKind.CHANGE_COST)
        self.assertEqual(operations[0].target, TargetKind.OWN_HAND)
        self.assertEqual(operations[0].amount, 1)
        self.assertEqual(operations[0].mode, CostChangeMode.ADD)
        self.assertEqual(operations[0].duration, ModifierDuration.PERMANENT)
        self.assertTrue(operations[0].requires_target)
        self.assertEqual(
            operations[0].hand_filter,
            HandFilter(card_type="随从"),
        )
        self.assertEqual(operations[1].kind, EffectKind.DESTROY)
        self.assertEqual(operations[1].target, TargetKind.RANDOM_ENEMY_UNIT)

    def test_real_card_only_offers_followers_then_resolves_both_clauses(self):
        engine = self.fresh_engine()
        source, follower, other_spell = _set_hand(
            engine,
            _spell(REAL_CARD_ID, cost=2, name="虚假的术式"),
            _card(3001, cost=4, name="合法随从"),
            _spell(3002, cost=4, name="非法法术"),
        )
        enemies = [_add_enemy_unit(engine, 4001), _add_enemy_unit(engine, 4002)]

        engine.apply(PlayCard(0, 0))

        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        self.assertEqual(
            [(option.entity_id, option.label) for option in request.options],
            [(follower.entity_id, "合法随从")],
        )
        self.assertNotEqual(source.entity_id, follower.entity_id)
        engine.apply(Choose(0, request.options[0].option_id))

        self.assertEqual(follower.current_cost, 5)
        self.assertEqual(other_spell.current_cost, 4)
        self.assertEqual(len(engine.players[1].board), 1)
        self.assertEqual(
            len({enemy.entity_id for enemy in enemies} - {
                engine.players[1].board[0].entity_id
            }),
            1,
        )

    def test_real_card_random_destruction_is_seed_deterministic(self):
        destroyed: list[int] = []
        for _ in range(2):
            engine = self.fresh_engine(seed=2718)
            _, follower = _set_hand(
                engine,
                _spell(REAL_CARD_ID, cost=2),
                _card(3010, cost=3),
            )
            enemy_ids = {
                _add_enemy_unit(engine, card_id).entity_id
                for card_id in (4010, 4011, 4012)
            }
            engine.apply(PlayCard(0, 0))
            engine.apply(Choose(0, f"hand:{follower.entity_id}"))
            remaining = {unit.entity_id for unit in engine.players[1].board}
            destroyed.append((enemy_ids - remaining).pop())
        self.assertEqual(destroyed[0], destroyed[1])

    def test_no_enemy_follower_skips_destroy_after_cost_change(self):
        engine = self.fresh_engine()
        _, follower = _set_hand(
            engine,
            _spell(REAL_CARD_ID, cost=2),
            _card(3020, cost=6),
        )
        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, f"hand:{follower.entity_id}"))
        self.assertEqual(follower.current_cost, 7)
        self.assertEqual(engine.players[1].board, [])

    def test_no_follower_makes_play_illegal_without_mutation(self):
        engine = self.fresh_engine()
        _set_hand(
            engine,
            _spell(REAL_CARD_ID, cost=2),
            _spell(3030, cost=1),
        )
        _add_enemy_unit(engine, 4030)
        self.assertFalse(
            any(isinstance(command, PlayCard) for command in engine.legal_commands())
        )
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_non_follower_choice_is_illegal_without_mutation(self):
        engine = self.fresh_engine()
        _, follower, other_spell = _set_hand(
            engine,
            _spell(REAL_CARD_ID, cost=2),
            _card(3040),
            _spell(3041),
        )
        engine.apply(PlayCard(0, 0))
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [follower.entity_id],
        )
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, f"hand:{other_spell.entity_id}"))
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_pending_follower_that_left_hand_is_skipped_and_resolution_continues(self):
        engine = self.fresh_engine()
        _, follower = _set_hand(
            engine,
            _spell(REAL_CARD_ID, cost=2),
            _card(3050),
        )
        enemy = _add_enemy_unit(engine, 4050)
        engine.apply(PlayCard(0, 0))
        choice = Choose(0, f"hand:{follower.entity_id}")
        index = engine.players[0].hand.index(follower)
        engine.players[0].hand.pop(index)
        engine.players[0].hand_entity_ids.pop(index)
        engine._send_to_graveyard(
            0,
            follower.definition,
            "test_pending_hand_target_left",
            source_entity_id=follower.entity_id,
        )

        engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(follower.current_cost, 1)
        self.assertNotIn(enemy, engine.players[1].board)

    def test_random_and_all_hand_targets_share_the_filter(self):
        random_rule = CardRule(
            910001,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.DISCARD,
                    TargetKind.RANDOM_OWN_HAND,
                    hand_filter=HandFilter(card_type="随从"),
                ),
            ),
        )
        engine = _engine(RuleBook((random_rule,)))
        _, follower, other_spell = _set_hand(
            engine,
            _spell(910001),
            _card(3060),
            _spell(3061),
        )
        engine.apply(PlayCard(0, 0))
        self.assertNotIn(follower, engine.players[0].hand)
        self.assertIn(other_spell, engine.players[0].hand)

        all_rule = CardRule(
            910002,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.CHANGE_COST,
                    TargetKind.ALL_OWN_HAND,
                    amount=1,
                    mode=CostChangeMode.ADD,
                    hand_filter=HandFilter(card_type="随从"),
                ),
            ),
        )
        engine = _engine(RuleBook((all_rule,)))
        _, first, second, other_spell = _set_hand(
            engine,
            _spell(910002),
            _card(3070, cost=2),
            _card(3071, cost=5),
            _spell(3072, cost=3),
        )
        engine.apply(PlayCard(0, 0))
        self.assertEqual((first.current_cost, second.current_cost), (3, 6))
        self.assertEqual(other_spell.current_cost, 3)

    def test_rl_masks_match_filtered_play_and_choice_legality(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(5000, 5040)],
            [_card(card_id) for card_id in range(6000, 6040)],
            class_a=3,
            class_b=3,
            seed=41,
            rulebook=self.rulebook,
        )
        env.reset(seed=41)
        env.players[0].mana = 10
        _set_hand(
            env.core,
            _spell(REAL_CARD_ID, cost=2),
            _spell(3080),
        )
        self.assertFalse(env.action_mask()[env.PLAY_OFFSET])

        _, follower, _ = _set_hand(
            env.core,
            _spell(REAL_CARD_ID, cost=2),
            _card(3081),
            _spell(3082),
        )
        self.assertTrue(env.action_mask()[env.PLAY_OFFSET])
        env.step(env.PLAY_OFFSET)
        mask = env.action_mask()
        enabled_choices = [
            action
            for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if mask[action]
        ]
        self.assertEqual(enabled_choices, [env.CHOICE_OFFSET])
        self.assertEqual(
            env.core.state.pending_choice.options[0].entity_id,
            follower.entity_id,
        )


if __name__ == "__main__":
    unittest.main()
