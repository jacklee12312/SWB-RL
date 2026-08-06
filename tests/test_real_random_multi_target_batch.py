# -*- coding: utf-8 -*-
"""Direct audits for random multi-target resolution and its first real batch."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import PlayCard
from swb.engine.effects import ConditionType, EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


BATCH_CARD_IDS = (
    10111150,
    10212310,
    10221110,
    10351310,
    10531120,
)


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 1),
        class_name=overrides.get("class_name", "精灵"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 3),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _make_engine(rulebook: RuleBook, *, seed: int = 163) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _clear_hand(engine: GameEngine) -> None:
    engine.players[0].hand.clear()
    engine.players[0].hand_entity_ids.clear()


def _put_in_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        spellboost_cost_reduction=engine.rulebook.spellboost_cost_reduction(
            definition.card_id
        ),
        cannot_be_played=engine.rulebook.cannot_be_played(definition.card_id),
        origin=CardOrigin.DECK,
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


def _add_enemy(
    engine: GameEngine,
    card_id: int,
    *,
    life: int,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[1].board.append(unit)
    return unit


class DatabaseClauseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_texts_keywords_and_absent_modes_or_references_match_audit(self):
        expected = {
            10111150: (
                "言传的杂草人长老", "精灵", "随从", 4,
                "【<color=Keyword>入场曲</color>】对对手的战场上的随机1个随从造成3点伤害。"
                "【<color=Keyword>连击</color>_3】改为3个。",
                {("入场曲", "入场曲"), ("连击", "连击")},
            ),
            10212310: (
                "来自树上的偷袭", "精灵", "法术", 1,
                "发动1次「对对手的战场上的随机1个随从造成2点伤害」。"
                "【<color=Keyword>连击</color>_3】改为发动2次。",
                {("连击", "连击")},
            ),
            10221110: (
                "扳机女仆·赛莉亚", "皇家护卫", "随从", 2,
                "【<color=Keyword>入场曲</color>】对对手的战场上的随机2个随从造成1点伤害。",
                {("入场曲", "入场曲")},
            ),
            10351310: (
                "前进的暴虐", "梦魇", "法术", 4,
                "对对手的战场上的随机2个随从和对手的主战者造成2点伤害。",
                set(),
            ),
            10531120: (
                "流动控符师", "巫师", "随从", 6,
                "【<color=Keyword>入场曲</color>】对对手的战场上的随机3个随从造成3点伤害。"
                "使自己的所有手牌发动3次魔力增幅。",
                {("入场曲", "入场曲")},
            ),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    rows = connection.execute(
                        """
                        SELECT json_extract(c.raw_json, '$.name_chs'),
                               cl.class_name, cl.type_name, c.cost,
                               st.text_chs, st.text_eng, st.text_jpn, st.text_cht
                        FROM cards c
                        JOIN card_localizations cl
                          ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                        JOIN skill_texts st ON st.card_id = c.card_id
                        WHERE c.card_id = ? ORDER BY st.position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0][:5], values[:5])
                    self.assertTrue(all(rows[0][index] for index in (5, 6, 7)))
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT ability_keyword, raw_keyword "
                            "FROM card_abilities WHERE card_id = ?",
                            (card_id,),
                        )),
                        values[5],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM card_references WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_all_batch_cards_have_exact_clause_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in BATCH_CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_random_multi_target_batch.py"],
                )


class RealRandomMultiTargetBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 163) -> GameEngine:
        return _make_engine(self.rulebook, seed=seed)

    def play_real(self, engine: GameEngine, card_id: int):
        _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0))
        return next(
            (
                entity
                for entity in engine.players[0].board
                if entity.definition.card_id == card_id
            ),
            None,
        )

    def test_rule_shapes_distinguish_distinct_targets_from_repeated_hits(self):
        elder = self.rulebook.operations_for(10111150, Trigger.FANFARE)[0]
        self.assertEqual(elder.kind, EffectKind.CONDITIONAL)
        self.assertEqual(
            elder.conditions[0].type,
            ConditionType.CONTROLLER_COMBO_AT_LEAST,
        )
        self.assertEqual(elder.then_operations[0].target_count, 3)

        repeated = self.rulebook.operations_for(10212310, Trigger.PLAY)[0]
        self.assertEqual(len(repeated.then_operations), 2)
        self.assertTrue(
            all(operation.target_count == 1 for operation in repeated.then_operations)
        )

        expected_counts = {10221110: 2, 10351310: 2, 10531120: 3}
        for card_id, target_count in expected_counts.items():
            trigger = Trigger.PLAY if card_id == 10351310 else Trigger.FANFARE
            with self.subTest(card_id=card_id):
                operation = self.rulebook.operations_for(card_id, trigger)[0]
                self.assertEqual(
                    (operation.target, operation.target_count),
                    (TargetKind.RANDOM_ENEMY_UNIT, target_count),
                )

    def test_celia_hits_two_distinct_followers_in_one_death_batch(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        targets = [_add_enemy(engine, 3000 + index, life=1) for index in range(3)]

        self.play_real(engine, 10221110)

        self.assertEqual(len(engine.players[1].board), 1)
        self.assertEqual(sum(target in engine.players[1].board for target in targets), 1)
        self.assertEqual(
            [len(batch.records) for batch in engine.state.death_queue],
            [2],
        )

    def test_forward_tyranny_uses_available_followers_then_hits_leader(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        target = _add_enemy(engine, 3100, life=3)

        self.play_real(engine, 10351310)

        self.assertEqual(target.health, 1)
        self.assertEqual(engine.players[1].health, 18)
        self.assertIsNone(engine.state.pending_choice)

    def test_flowing_runecaster_hits_three_and_spellboosts_the_whole_hand(self):
        def resolve_once():
            engine = self.fresh_engine(seed=167)
            _clear_hand(engine)
            tracked = _put_in_hand(engine, _card(3200, cost=9))
            targets = [
                _add_enemy(engine, 3210 + index, life=3)
                for index in range(4)
            ]
            self.play_real(engine, 10531120)
            survivors = tuple(
                target.definition.card_id
                for target in targets
                if target in engine.players[1].board
            )
            return survivors, tracked.spellboost_count

        first = resolve_once()
        second = resolve_once()

        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 1)
        self.assertEqual(first[1], 3)

    def test_combo_changes_elder_to_three_distinct_targets(self):
        normal = self.fresh_engine(seed=173)
        _clear_hand(normal)
        normal_targets = [
            _add_enemy(normal, 3300 + index, life=3)
            for index in range(3)
        ]
        self.play_real(normal, 10111150)
        self.assertEqual(sum(target in normal.players[1].board for target in normal_targets), 2)

        combo = self.fresh_engine(seed=173)
        _clear_hand(combo)
        combo.players[0].cards_played_this_turn = 2
        combo_targets = [
            _add_enemy(combo, 3310 + index, life=3)
            for index in range(3)
        ]
        self.play_real(combo, 10111150)
        self.assertEqual(sum(target in combo.players[1].board for target in combo_targets), 0)

    def test_combo_repeated_hits_remain_independent_random_operations(self):
        normal = self.fresh_engine(seed=179)
        _clear_hand(normal)
        normal_targets = [_add_enemy(normal, 3400 + index, life=5) for index in range(2)]
        self.play_real(normal, 10212310)
        self.assertEqual(sum(target.health for target in normal_targets), 8)

        combo = self.fresh_engine(seed=179)
        _clear_hand(combo)
        combo.players[0].cards_played_this_turn = 2
        combo_targets = [_add_enemy(combo, 3410 + index, life=5) for index in range(2)]
        self.play_real(combo, 10212310)
        self.assertEqual(sum(target.health for target in combo_targets), 6)

    def test_random_multi_target_card_is_one_automatic_rl_action(self):
        deck_a = [_card(card_id) for card_id in range(4000, 4040)]
        deck_b = [_card(card_id) for card_id in range(5000, 5040)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=181,
            rulebook=self.rulebook,
        )
        env.reset(seed=181)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        hand_card = HandCard(
            definition=self.repository.get(10221110),
            entity_id=env.core.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        env.players[0].hand.append(hand_card)
        env.players[0].hand_entity_ids.append(hand_card.entity_id)
        env.players[0].max_mana = env.players[0].mana = 10
        for index in range(3):
            env.players[1].board.append(
                Unit.summon(
                    _card(5100 + index, life=2),
                    entity_id=env.core.state.allocate_entity_id(),
                )
            )

        mask = env.action_mask()
        self.assertTrue(mask[ShadowverseEnv.PLAY_OFFSET])
        result = env.step(ShadowverseEnv.PLAY_OFFSET)

        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual(len(env.players[1].board), 3)
        self.assertEqual(
            sorted(unit.health for unit in env.players[1].board),
            [1, 1, 2],
        )
        self.assertFalse(any(
            result.info["action_mask"][
                ShadowverseEnv.CHOICE_OFFSET:
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET
            ]
        ))


if __name__ == "__main__":
    unittest.main()
