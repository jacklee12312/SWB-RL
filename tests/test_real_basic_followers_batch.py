# -*- coding: utf-8 -*-
"""Direct audits for the real basic-follower coverage batch."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import DamageType, GameEngine
from swb.engine.state import Amulet, HandCard, Unit


BATCH_CARD_IDS = (
    10021120,
    10112110,
    10121110,
    10141130,
    10204110,
    10262120,
    10412120,
    10422110,
    10431110,
    10601120,
    10742110,
    10822120,
    10841120,
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
        life=overrides.get("life", 1),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _spell(card_id: int, *, name: str | None = None) -> CardDefinition:
    return _card(
        card_id,
        name=name or f"spell-{card_id}",
        card_type="法术",
        attack=None,
        life=None,
    )


def _amulet(card_id: int, *, name: str | None = None) -> CardDefinition:
    return _card(
        card_id,
        name=name or f"amulet-{card_id}",
        card_type="护符",
        attack=None,
        life=None,
    )


def _make_engine(rulebook: RuleBook, *, seed: int = 29) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
    )
    engine.reset(seed=seed)
    engine.players[0].mana = 10
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
    life: int = 5,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, life=life),
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
            10021120: (
                "战斗商贩",
                "皇家护卫",
                4,
                "【<color=Keyword>突进</color>】\n【<color=Keyword>谢幕曲</color>】抽取1张卡牌。",
                {("突进", "突进"), ("谢幕曲", "谢幕曲")},
            ),
            10112110: (
                "霜寒冰晶·艾琳",
                "精灵",
                7,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，破坏该随从。"
                "回复自己的主战者2点生命值。\n<hr>【<color=Keyword>守护</color>】",
                {("入场曲", "入场曲"), ("守护", "守护")},
            ),
            10121110: (
                "爱之骑士·尹安",
                "皇家护卫",
                3,
                "【<color=Keyword>入场曲</color>】选择自己的战场上的1个其他随从，使其+1/+1。"
                "\n<hr>【<color=Keyword>守护</color>】",
                {("入场曲", "入场曲"), ("守护", "守护")},
            ),
            10141130: (
                "初出茅庐的屠龙者",
                "龙族",
                4,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，破坏该随从。",
                {("入场曲", "入场曲")},
            ),
            10204110: (
                "命运黄昏·奥丁",
                "中立",
                7,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1张卡牌，使其消失。"
                "\n<hr>【<color=Keyword>疾驰</color>】",
                {("入场曲", "入场曲"), ("疾驰", "疾驰")},
            ),
            10262120: (
                "有洁癖的审判者",
                "主教",
                5,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，使其消失。",
                {("入场曲", "入场曲")},
            ),
            10412120: (
                "绯焰舞姬·安苏莉娅",
                "精灵",
                5,
                "【<color=Keyword>入场曲</color>】使自己的战场上的所有随从获得"
                "【<color=Keyword>屏障</color>】。",
                {("入场曲", "入场曲"), ("屏障", "屏障")},
            ),
            10422110: (
                "冰心霸王·艾格罗瓦尔",
                "皇家护卫",
                6,
                "【<color=Keyword>入场曲</color>】对对手的战场上的所有随从造成3点伤害。"
                "\n<hr>【<color=Keyword>威慑</color>】",
                {("入场曲", "入场曲"), ("威慑", "威慑")},
            ),
            10431110: (
                "不可思议的哲学家·菲拉索佩娅",
                "巫师",
                3,
                "【<color=Keyword>入场曲</color>】抽取1张法术。",
                {("入场曲", "入场曲")},
            ),
            10601120: (
                "匍匐的异类",
                "中立",
                5,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，破坏该随从。"
                "\n<hr>【<color=Keyword>毁灭</color>】",
                {("入场曲", "入场曲"), ("必杀", "毁灭")},
            ),
            10742110: (
                "豪龙守门人",
                "龙族",
                7,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，破坏该随从。"
                "\n<hr>【<color=Keyword>守护</color>】",
                {("入场曲", "入场曲"), ("守护", "守护")},
            ),
            10822120: (
                "织田信长",
                "皇家护卫",
                10,
                "【<color=Keyword>入场曲</color>】对对手的战场上的所有随从造成6点伤害。"
                "\n<hr>【<color=Keyword>威慑</color>】",
                {("入场曲", "入场曲"), ("威慑", "威慑")},
            ),
            10841120: (
                "沙尘守宝龙",
                "龙族",
                7,
                "【<color=Keyword>守护</color>】\n【<color=Keyword>谢幕曲</color>】抽取3张卡牌。",
                {("守护", "守护"), ("谢幕曲", "谢幕曲")},
            ),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, (name, class_name, cost, text, abilities) in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
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
                    self.assertEqual(len(row), 1)
                    self.assertEqual(row[0][:5], (name, class_name, "随从", cost, text))
                    self.assertTrue(all(row[0][index] for index in (5, 6, 7)))
                    self.assertEqual(
                        set(connection.execute(
                            """
                            SELECT ability_keyword, raw_keyword
                            FROM card_abilities
                            WHERE card_id = ?
                            """,
                            (card_id,),
                        )),
                        abilities,
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
                    ["tests/test_real_basic_followers_batch.py"],
                )


class RealBasicFollowerBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 29) -> GameEngine:
        return _make_engine(self.rulebook, seed=seed)

    def play_real(self, engine: GameEngine, card_id: int) -> HandCard:
        source = _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0))
        return source

    def test_rule_triggers_and_operation_shapes_are_exact(self):
        expected = {
            10021120: (Trigger.LAST_WORDS, [(EffectKind.DRAW, TargetKind.OWN_LEADER, 1)]),
            10112110: (
                Trigger.FANFARE,
                [
                    (EffectKind.DESTROY, TargetKind.ENEMY_UNIT, 0),
                    (EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 2),
                ],
            ),
            10121110: (Trigger.FANFARE, [(EffectKind.BUFF_UNIT, TargetKind.OWN_UNIT, 1)]),
            10141130: (Trigger.FANFARE, [(EffectKind.DESTROY, TargetKind.ENEMY_UNIT, 0)]),
            10204110: (Trigger.FANFARE, [(EffectKind.BANISH, TargetKind.ENEMY_BOARD, 0)]),
            10262120: (Trigger.FANFARE, [(EffectKind.BANISH, TargetKind.ENEMY_UNIT, 0)]),
            10412120: (Trigger.FANFARE, [(EffectKind.ADD_KEYWORD, TargetKind.ALL_OWN_UNITS, 0)]),
            10422110: (Trigger.FANFARE, [(EffectKind.DAMAGE_UNIT, TargetKind.ALL_ENEMY_UNITS, 3)]),
            10431110: (Trigger.FANFARE, [(EffectKind.DRAW_FILTERED, TargetKind.OWN_LEADER, 1)]),
            10601120: (Trigger.FANFARE, [(EffectKind.DESTROY, TargetKind.ENEMY_UNIT, 0)]),
            10742110: (Trigger.FANFARE, [(EffectKind.DESTROY, TargetKind.ENEMY_UNIT, 0)]),
            10822120: (Trigger.FANFARE, [(EffectKind.DAMAGE_UNIT, TargetKind.ALL_ENEMY_UNITS, 6)]),
            10841120: (Trigger.LAST_WORDS, [(EffectKind.DRAW, TargetKind.OWN_LEADER, 3)]),
        }
        for card_id, (trigger, shapes) in expected.items():
            with self.subTest(card_id=card_id):
                operations = self.rulebook.operations_for(card_id, trigger)
                self.assertEqual(
                    [(op.kind, op.target, op.amount) for op in operations],
                    shapes,
                )
        buff = self.rulebook.operations_for(10121110, Trigger.FANFARE)[0]
        self.assertEqual(buff.secondary_amount, 1)
        self.assertTrue(buff.exclude_source)
        draw_spell = self.rulebook.operations_for(10431110, Trigger.FANFARE)[0]
        self.assertEqual(draw_spell.deck_filter.card_type, "法术")
        grant_barrier = self.rulebook.operations_for(10412120, Trigger.FANFARE)[0]
        self.assertEqual(grant_barrier.keyword, "屏障")
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10412120),
            frozenset({"屏障"}),
        )

    def test_destroy_fanfares_and_real_static_keywords(self):
        for card_id, keyword in (
            (10141130, None),
            (10601120, "必杀"),
            (10742110, "守护"),
        ):
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine()
                target = _add_unit(engine, 1, 7000 + card_id)
                self.play_real(engine, card_id)
                source = engine.players[0].board[-1]
                self.assertEqual(
                    [option.entity_id for option in engine.state.pending_choice.options],
                    [target.entity_id],
                )
                _choose(engine, target.entity_id)
                self.assertNotIn(target, engine.players[1].board)
                if keyword is not None:
                    self.assertTrue(source.has_keyword(keyword))

    def test_frost_crystal_eileen_destroys_then_heals_and_has_ward(self):
        engine = self.fresh_engine()
        engine.players[0].health = 15
        target = _add_unit(engine, 1, 7100)
        self.play_real(engine, 10112110)
        source = engine.players[0].board[-1]
        _choose(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[0].health, 17)
        self.assertTrue(source.has_guard)

    def test_targeted_fanfare_without_candidates_skips_but_follower_still_plays(self):
        engine = self.fresh_engine()
        engine.players[0].health = 15
        self.play_real(engine, 10112110)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[0].board[-1].definition.card_id, 10112110)
        self.assertEqual(engine.players[0].health, 17)

    def test_love_knight_excludes_source_and_buffs_only_selected_ally(self):
        engine = self.fresh_engine()
        ally = _add_unit(engine, 0, 7200, life=3)
        self.play_real(engine, 10121110)
        source = engine.players[0].board[-1]
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [ally.entity_id],
        )
        _choose(engine, ally.entity_id)
        self.assertEqual((ally.attack, ally.health), (2, 4))
        self.assertEqual((source.attack, source.health), (2, 2))
        self.assertTrue(source.has_guard)

        engine = self.fresh_engine()
        self.play_real(engine, 10121110)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(len(engine.players[0].board), 1)

    def test_odins_board_banish_includes_amulets_and_storm_is_live(self):
        engine = self.fresh_engine()
        unit = _add_unit(engine, 1, 7300)
        amulet = Amulet(
            definition=_amulet(7301),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board.append(amulet)
        self.play_real(engine, 10204110)
        source = engine.players[0].board[-1]
        self.assertEqual(
            {option.entity_id for option in engine.state.pending_choice.options},
            {unit.entity_id, amulet.entity_id},
        )
        _choose(engine, amulet.entity_id)
        self.assertNotIn(amulet, engine.players[1].board)
        self.assertTrue(any(card.card_id == 7301 for card in engine.players[1].banished))
        self.assertTrue(source.can_attack_leader)

    def test_judge_banishes_followers_but_does_not_offer_amulets(self):
        engine = self.fresh_engine()
        unit = _add_unit(engine, 1, 7400)
        amulet = Amulet(
            definition=_amulet(7401),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board.append(amulet)
        self.play_real(engine, 10262120)
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [unit.entity_id],
        )
        _choose(engine, unit.entity_id)
        self.assertNotIn(unit, engine.players[1].board)
        self.assertIn(amulet, engine.players[1].board)

    def test_ansulia_grants_exactly_one_barrier_to_all_own_followers(self):
        engine = self.fresh_engine()
        ally = _add_unit(engine, 0, 7500)
        self.play_real(engine, 10412120)
        source = engine.players[0].board[-1]
        self.assertEqual(ally.barrier_charges, 1)
        self.assertEqual(source.barrier_charges, 1)
        self.assertTrue(ally.has_keyword("屏障"))
        self.assertTrue(source.has_keyword("屏障"))

    def test_enemy_wide_damage_is_simultaneous_and_intimidate_is_present(self):
        for card_id, amount in ((10422110, 3), (10822120, 6)):
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine()
                doomed = _add_unit(engine, 1, 7600 + card_id, life=amount)
                survivor = _add_unit(engine, 1, 7700 + card_id, life=amount + 2)
                self.play_real(engine, card_id)
                source = engine.players[0].board[-1]
                self.assertNotIn(doomed, engine.players[1].board)
                self.assertEqual(survivor.health, 2)
                self.assertTrue(source.has_intimidate)

    def test_philosopher_draws_only_a_spell_and_is_seed_deterministic(self):
        drawn: list[int] = []
        for _ in range(2):
            engine = self.fresh_engine(seed=314)
            engine.players[0].deck = [
                _card(7800),
                _spell(7801),
                _spell(7802),
            ]
            self.play_real(engine, 10431110)
            hand_ids = {card.card_id for card in engine.players[0].hand}
            selected = ({7801, 7802} & hand_ids).pop()
            drawn.append(selected)
            self.assertNotIn(7800, hand_ids)
        self.assertEqual(drawn[0], drawn[1])

    def test_real_last_words_draw_exact_amount_and_keep_static_keyword(self):
        for card_id, amount, keyword in (
            (10021120, 1, "突进"),
            (10841120, 3, "守护"),
        ):
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine()
                unit = Unit.summon(
                    self.repository.get(card_id),
                    entity_id=engine.state.allocate_entity_id(),
                )
                engine.players[0].board = [unit]
                hand_before = len(engine.players[0].hand)
                deck_before = len(engine.players[0].deck)
                self.assertTrue(unit.has_keyword(keyword))
                engine.apply_damage(
                    None,
                    unit,
                    99,
                    DamageType.EFFECT,
                    controller=1,
                )
                engine._stabilize()
                self.assertNotIn(unit, engine.players[0].board)
                self.assertEqual(len(engine.players[0].hand), hand_before + amount)
                self.assertEqual(len(engine.players[0].deck), deck_before - amount)

    def test_rl_choice_mask_matches_source_excluding_real_fanfare(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(8000, 8040)],
            [_card(card_id) for card_id in range(9000, 9040)],
            class_a=1,
            class_b=1,
            seed=61,
            rulebook=self.rulebook,
        )
        env.reset(seed=61)
        env.players[0].mana = 10
        ally = _add_unit(env.core, 0, 8050)
        _put_in_hand(env.core, self.repository.get(10121110))
        self.assertTrue(env.action_mask()[env.PLAY_OFFSET])
        env.step(env.PLAY_OFFSET)
        self.assertEqual(
            [option.entity_id for option in env.core.state.pending_choice.options],
            [ally.entity_id],
        )
        mask = env.action_mask()
        self.assertTrue(mask[env.CHOICE_OFFSET])
        self.assertFalse(mask[env.CHOICE_OFFSET + 1])


if __name__ == "__main__":
    unittest.main()
