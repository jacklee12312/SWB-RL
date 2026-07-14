# -*- coding: utf-8 -*-
"""Direct audits for the real Dragoncraft Overflow coverage batch."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard
from swb.engine.effects import ConditionType, EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


BATCH_CARD_IDS = (
    10141120,
    10141150,
    10142120,
    10142140,
    10142310,
    10241120,
    10343310,
    10543310,
    10742310,
)


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 4),
        class_name=overrides.get("class_name", "龙族"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 1),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _spell(card_id: int, **overrides) -> CardDefinition:
    return _card(
        card_id,
        card_type="法术",
        attack=None,
        life=None,
        **overrides,
    )


def _make_engine(
    rulebook: RuleBook,
    *,
    seed: int = 47,
    max_mana: int = 7,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=4,
        class_b=4,
        seed=seed,
        rulebook=rulebook,
    )
    engine.reset(seed=seed)
    engine.players[0].max_mana = max_mana
    engine.players[0].mana = max_mana
    return engine


def _put_in_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


def _add_unit(
    engine: GameEngine,
    owner: int,
    card_id: int,
    *,
    attack: int = 1,
    life: int = 5,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, attack=attack, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[owner].board.append(unit)
    return unit


def _choose(engine: GameEngine, entity_id: int) -> None:
    command = next(
        command
        for command in engine.legal_commands()
        if isinstance(command, Choose)
        and command.option_id == f"entity:{entity_id}"
    )
    engine.apply(command)


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
            10141120: (
                "海沟大剑龙", "龙族", "随从", 5,
                "【<color=Keyword>入场曲</color>】若为【<color=Keyword>觉醒</color>】，"
                "则本随从获得【<color=Keyword>疾驰</color>】。",
                {("入场曲", "入场曲"), ("疾驰", "疾驰"), ("觉醒", "觉醒")},
            ),
            10141150: (
                "白鳞的使者", "龙族", "随从", 4,
                "【<color=Keyword>入场曲</color>】若为【<color=Keyword>觉醒</color>】，"
                "则回复自己的主战者4点生命值。",
                {("入场曲", "入场曲"), ("觉醒", "觉醒")},
            ),
            10142120: (
                "御风者·叶花", "龙族", "随从", 3,
                "【<color=Keyword>入场曲</color>】若为【<color=Keyword>觉醒</color>】，"
                "则本随从获得【<color=Keyword>威慑</color>】。\n<hr>"
                "【<color=Keyword>疾驰</color>】",
                {
                    ("入场曲", "入场曲"), ("威慑", "威慑"),
                    ("疾驰", "疾驰"), ("觉醒", "觉醒"),
                },
            ),
            10142140: (
                "艳丽龙人·玛利翁", "龙族", "随从", 4,
                "【<color=Keyword>入场曲</color>】选择自己的战场上的1个其他随从，"
                "使其+2/+2。若为【<color=Keyword>觉醒</color>】，则改为+3/+3。",
                {("入场曲", "入场曲"), ("觉醒", "觉醒")},
            ),
            10142310: (
                "荣弦的奏乐", "龙族", "法术", 3,
                "抽取2张卡牌。若为【<color=Keyword>觉醒</color>】，"
                "则回复自己的主战者2点生命值。",
                {("觉醒", "觉醒")},
            ),
            10241120: (
                "飞跃的银白幼龙", "龙族", "随从", 1,
                "【<color=Keyword>入场曲</color>】若为【<color=Keyword>觉醒</color>】，"
                "则抽取1张卡牌。\n<hr>【<color=Keyword>突进</color>】",
                {("入场曲", "入场曲"), ("突进", "突进"), ("觉醒", "觉醒")},
            ),
            10343310: (
                "威猛炽焰", "龙族", "法术", 1,
                "选择自己的战场上的1个随从，对其造成1点伤害。"
                "对对手的战场上的随机1个随从造成3点伤害。"
                "若为【<color=Keyword>觉醒</color>】，则抽取1张龙族·随从。",
                {("觉醒", "觉醒")},
            ),
            10543310: (
                "懒惰的波摇花", "龙族", "法术", 2,
                "发动2次「对对手的战场上的随机1个随从造成2点伤害」。"
                "若为【<color=Keyword>觉醒</color>】，则对对手的主战者造成2点伤害。",
                {("觉醒", "觉醒")},
            ),
            10742310: (
                "焦龙的午睡", "龙族", "法术", 3,
                "回复自己的主战者3点生命值。若为【<color=Keyword>觉醒</color>】，"
                "则抽取1张卡牌。",
                {("觉醒", "觉醒")},
            ),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, expected_row in expected.items():
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
                        WHERE c.card_id = ?
                        ORDER BY st.position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0][:5], expected_row[:5])
                    self.assertTrue(all(rows[0][index] for index in (5, 6, 7)))
                    self.assertEqual(
                        set(connection.execute(
                            """
                            SELECT ability_keyword, raw_keyword
                            FROM card_abilities
                            WHERE card_id = ?
                            """,
                            (card_id,),
                        )),
                        expected_row[5],
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
                    ["tests/test_real_overflow_batch.py"],
                )


class RealOverflowBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(
        self,
        *,
        seed: int = 47,
        max_mana: int = 7,
    ) -> GameEngine:
        return _make_engine(self.rulebook, seed=seed, max_mana=max_mana)

    def play_real(self, engine: GameEngine, card_id: int) -> HandCard:
        source = _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0))
        return source

    def test_rule_shapes_conditions_and_filters_are_exact(self):
        expected = {
            10141120: (Trigger.FANFARE, [(EffectKind.ADD_KEYWORD, TargetKind.SELF, 0)]),
            10141150: (Trigger.FANFARE, [(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 4)]),
            10142120: (Trigger.FANFARE, [(EffectKind.ADD_KEYWORD, TargetKind.SELF, 0)]),
            10142140: (
                Trigger.FANFARE,
                [
                    (EffectKind.BUFF_UNIT, TargetKind.OWN_UNIT, 3),
                    (EffectKind.BUFF_UNIT, TargetKind.OWN_UNIT, 2),
                ],
            ),
            10142310: (
                Trigger.PLAY,
                [
                    (EffectKind.DRAW, TargetKind.OWN_LEADER, 2),
                    (EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 2),
                ],
            ),
            10241120: (Trigger.FANFARE, [(EffectKind.DRAW, TargetKind.OWN_LEADER, 1)]),
            10343310: (
                Trigger.PLAY,
                [
                    (EffectKind.DAMAGE_UNIT, TargetKind.OWN_UNIT, 1),
                    (EffectKind.DAMAGE_UNIT, TargetKind.RANDOM_ENEMY_UNIT, 3),
                    (EffectKind.DRAW_FILTERED, TargetKind.OWN_LEADER, 1),
                ],
            ),
            10543310: (
                Trigger.PLAY,
                [
                    (EffectKind.DAMAGE_UNIT, TargetKind.RANDOM_ENEMY_UNIT, 2),
                    (EffectKind.DAMAGE_UNIT, TargetKind.RANDOM_ENEMY_UNIT, 2),
                    (EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 2),
                ],
            ),
            10742310: (
                Trigger.PLAY,
                [
                    (EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 3),
                    (EffectKind.DRAW, TargetKind.OWN_LEADER, 1),
                ],
            ),
        }
        for card_id, (trigger, shapes) in expected.items():
            with self.subTest(card_id=card_id):
                operations = self.rulebook.operations_for(card_id, trigger)
                self.assertEqual(
                    [(op.kind, op.target, op.amount) for op in operations],
                    shapes,
                )

        marion = self.rulebook.operations_for(10142140, Trigger.FANFARE)
        self.assertTrue(all(op.exclude_source for op in marion))
        self.assertEqual([op.secondary_amount for op in marion], [3, 2])
        self.assertEqual(marion[0].conditions[0].type, ConditionType.CONTROLLER_OVERFLOW)
        self.assertEqual(marion[1].conditions[0].type, ConditionType.NOT)
        filtered = self.rulebook.operations_for(10343310, Trigger.PLAY)[2].deck_filter
        self.assertEqual((filtered.card_type, filtered.class_id), ("随从", 4))
        self.assertTrue(
            self.rulebook.operations_for(10343310, Trigger.PLAY)[0].requires_target
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10141120),
            frozenset({"疾驰"}),
        )
        self.assertIn("威慑", self.rulebook.non_intrinsic_keywords(10142120))

    def test_fanfare_overflow_boundary_and_keyword_provenance(self):
        for max_mana, has_storm in ((6, False), (7, True)):
            with self.subTest(card_id=10141120, max_mana=max_mana):
                engine = self.fresh_engine(max_mana=max_mana)
                self.play_real(engine, 10141120)
                source = engine.players[0].board[-1]
                self.assertEqual(source.has_keyword("疾驰"), has_storm)
                self.assertEqual(source.can_attack_leader, has_storm)

        for max_mana, has_intimidate in ((6, False), (7, True)):
            with self.subTest(card_id=10142120, max_mana=max_mana):
                engine = self.fresh_engine(max_mana=max_mana)
                self.play_real(engine, 10142120)
                source = engine.players[0].board[-1]
                self.assertTrue(source.has_keyword("疾驰"))
                self.assertEqual(source.has_intimidate, has_intimidate)

        for max_mana, expected_health in ((6, 10), (7, 14)):
            with self.subTest(card_id=10141150, max_mana=max_mana):
                engine = self.fresh_engine(max_mana=max_mana)
                engine.players[0].health = 10
                self.play_real(engine, 10141150)
                self.assertEqual(engine.players[0].health, expected_health)

        for max_mana, expected_draw in ((6, 0), (7, 1)):
            with self.subTest(card_id=10241120, max_mana=max_mana):
                engine = self.fresh_engine(max_mana=max_mana)
                deck_before = len(engine.players[0].deck)
                self.play_real(engine, 10241120)
                source = engine.players[0].board[-1]
                self.assertTrue(source.has_keyword("突进"))
                self.assertEqual(deck_before - len(engine.players[0].deck), expected_draw)

    def test_marion_excludes_source_and_uses_overflow_replacement_amount(self):
        for max_mana, amount in ((6, 2), (7, 3)):
            with self.subTest(max_mana=max_mana):
                engine = self.fresh_engine(max_mana=max_mana)
                ally = _add_unit(engine, 0, 7100 + max_mana, attack=2, life=3)
                self.play_real(engine, 10142140)
                source = engine.players[0].board[-1]
                self.assertEqual(
                    [option.entity_id for option in engine.state.pending_choice.options],
                    [ally.entity_id],
                )
                _choose(engine, ally.entity_id)
                self.assertEqual((ally.attack, ally.health), (2 + amount, 3 + amount))
                self.assertEqual(
                    (source.attack, source.health),
                    (source.definition.attack, source.definition.life),
                )

        engine = self.fresh_engine(max_mana=7)
        self.play_real(engine, 10142140)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(len(engine.players[0].board), 1)

    def test_simple_overflow_spells_keep_unconditional_effects(self):
        for max_mana, expected_health in ((6, 10), (7, 12)):
            with self.subTest(card_id=10142310, max_mana=max_mana):
                engine = self.fresh_engine(max_mana=max_mana)
                engine.players[0].health = 10
                deck_before = len(engine.players[0].deck)
                self.play_real(engine, 10142310)
                self.assertEqual(deck_before - len(engine.players[0].deck), 2)
                self.assertEqual(engine.players[0].health, expected_health)

        for max_mana, expected_draw in ((6, 0), (7, 1)):
            with self.subTest(card_id=10742310, max_mana=max_mana):
                engine = self.fresh_engine(max_mana=max_mana)
                engine.players[0].health = 10
                deck_before = len(engine.players[0].deck)
                self.play_real(engine, 10742310)
                self.assertEqual(engine.players[0].health, 13)
                self.assertEqual(deck_before - len(engine.players[0].deck), expected_draw)

    def test_mighty_blaze_orders_self_damage_random_damage_and_filtered_draw(self):
        engine = self.fresh_engine(max_mana=7)
        own = _add_unit(engine, 0, 7300, life=1)
        enemy = _add_unit(engine, 1, 7301, life=5)
        matching = _card(7302, class_id=4, class_name="龙族", card_type="随从")
        engine.players[0].deck = [
            _spell(7303),
            _card(7304, class_id=1, class_name="精灵"),
            matching,
        ]

        self.play_real(engine, 10343310)
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [own.entity_id],
        )
        _choose(engine, own.entity_id)

        self.assertNotIn(own, engine.players[0].board)
        self.assertEqual(enemy.health, 2)
        self.assertIn(7302, [card.card_id for card in engine.players[0].hand])
        self.assertNotIn(7304, [card.card_id for card in engine.players[0].hand])

    def test_mighty_blaze_without_own_target_is_illegal_and_non_mutating(self):
        engine = self.fresh_engine(max_mana=7)
        _put_in_hand(engine, self.repository.get(10343310))
        engine._ensure_entity_ids()
        before = engine.deterministic_fingerprint()

        self.assertFalse(
            any(
                isinstance(command, PlayCard) and command.hand_index == 0
                for command in engine.legal_commands()
            )
        )
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_lazy_waverose_repeats_random_damage_and_is_seeded(self):
        def run(seed: int, max_mana: int) -> tuple[tuple[int, ...], int]:
            engine = self.fresh_engine(seed=seed, max_mana=max_mana)
            _add_unit(engine, 1, 7500, life=5)
            _add_unit(engine, 1, 7501, life=5)
            self.play_real(engine, 10543310)
            return (
                tuple(unit.health for unit in engine.players[1].board),
                engine.players[1].health,
            )

        self.assertEqual(run(901, 7), run(901, 7))
        healths, leader_health = run(901, 7)
        self.assertEqual(sum(healths), 6)
        self.assertEqual(leader_health, 18)
        self.assertEqual(run(901, 6)[1], 20)

        engine = self.fresh_engine(max_mana=7)
        self.play_real(engine, 10543310)
        self.assertEqual(engine.players[1].health, 18)

    def test_rl_mask_matches_real_source_excluding_overflow_choice(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(8000, 8040)],
            [_card(card_id) for card_id in range(9000, 9040)],
            class_a=4,
            class_b=4,
            seed=67,
            rulebook=self.rulebook,
        )
        env.reset(seed=67)
        env.players[0].max_mana = 7
        env.players[0].mana = 7
        ally = _add_unit(env.core, 0, 8050)
        _put_in_hand(env.core, self.repository.get(10142140))

        self.assertTrue(env.action_mask()[env.PLAY_OFFSET])
        env.step(env.PLAY_OFFSET)
        self.assertEqual(
            [option.entity_id for option in env.core.state.pending_choice.options],
            [ally.entity_id],
        )
        mask = env.action_mask()
        self.assertTrue(mask[env.CHOICE_OFFSET])
        self.assertFalse(mask[env.CHOICE_OFFSET + 1])

        spell_env = ShadowverseEnv(
            [_card(card_id) for card_id in range(8100, 8140)],
            [_card(card_id) for card_id in range(9100, 9140)],
            class_a=4,
            class_b=4,
            seed=71,
            rulebook=self.rulebook,
        )
        spell_env.reset(seed=71)
        spell_env.players[0].max_mana = 7
        spell_env.players[0].mana = 7
        _put_in_hand(spell_env.core, self.repository.get(10343310))
        self.assertFalse(spell_env.action_mask()[spell_env.PLAY_OFFSET])
        _add_unit(spell_env.core, 0, 8150)
        self.assertTrue(spell_env.action_mask()[spell_env.PLAY_OFFSET])


if __name__ == "__main__":
    unittest.main()
