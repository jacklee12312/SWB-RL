# -*- coding: utf-8 -*-
"""Direct audits for the second real evolution-trigger coverage batch."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import GameEngine
from swb.engine.state import AttackRestriction, HandCard, Unit


BATCH_CARD_IDS = (
    10222120,
    10311110,
    10312110,
    10511110,
    10561110,
    10701110,
    10712120,
    10801110,
    10802110,
    10821120,
    10831110,
    10861130,
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


def _spell(card_id: int, **overrides) -> CardDefinition:
    return _card(card_id, card_type="法术", attack=None, life=None, **overrides)


def _make_engine(rulebook: RuleBook, *, seed: int = 97) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
    )
    engine.reset(seed=seed)
    engine.players[0].max_mana = 10
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


def _enable_evolution(engine: GameEngine) -> None:
    engine.players[0].turns_started = engine.config.evolution_unlock_turn
    engine.players[0].evolution_points = 2


def _enable_super_evolution(engine: GameEngine) -> None:
    engine.players[0].turns_started = (
        engine.config.first_player_super_evolution_unlock_turn
    )
    engine.players[0].super_evolution_points = 2


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
            10222120: (
                "平凡骑士·拉奇尔", "皇家护卫", 4,
                "【<color=Keyword>入场曲</color>】抽取1张法术。\n<hr>"
                "<ev>【<color=Keyword>进化时</color>】回复自己2点能量点。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时")},
            ),
            10311110: (
                "不弑的肯定者", "精灵", 2,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，使其-0/-1。"
                "\n<hr><ev>【<color=Keyword>进化时</color>】发动与"
                "【<color=Keyword>入场曲</color>】相同的能力。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时")},
            ),
            10312110: (
                "不弑的祈祷者", "精灵", 6,
                "【<color=Keyword>入场曲</color>】使对手的战场上的所有随从-0/-3。"
                "\n<hr><sev>【<color=Keyword>超进化时</color>】发动与"
                "【<color=Keyword>入场曲</color>】相同的能力。</sev>",
                {("入场曲", "入场曲"), ("超进化时", "超进化时")},
            ),
            10511110: (
                "熟虑的狸猫", "精灵", 2,
                "【<color=Keyword>潜行</color>】\n<hr>"
                "<ev>【<color=Keyword>进化时</color>】抽取1张卡牌。</ev>",
                {("潜行", "潜行"), ("进化时", "进化时")},
            ),
            10561110: (
                "先见的神官", "主教", 3,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，"
                "对其造成2点伤害。\n<hr>【<color=Keyword>守护</color>】\n<hr>"
                "<ev>【<color=Keyword>进化时</color>】发动与"
                "【<color=Keyword>入场曲</color>】相同的能力。</ev>",
                {("入场曲", "入场曲"), ("守护", "守护"), ("进化时", "进化时")},
            ),
            10701110: (
                "纯真孩童", "中立", 2,
                "<ev>【<color=Keyword>进化时</color>】抽取1张中立·卡牌。</ev>",
                {("进化时", "进化时")},
            ),
            10712120: (
                "弓兵指挥者", "精灵", 5,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，"
                "对其造成4点伤害。使自己的【<color=Keyword>连击</color>】+1。\n<hr>"
                "<ev>【<color=Keyword>进化时</color>】发动与"
                "【<color=Keyword>入场曲</color>】相同的能力。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时"), ("连击", "连击")},
            ),
            10801110: (
                "自律的圣鸟·汉萨", "中立", 2,
                "<ev>【<color=Keyword>进化时</color>】本随从+5/+5。</ev>",
                {("进化时", "进化时")},
            ),
            10802110: (
                "激动的欢喜·阿尔菲德", "中立", 4,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，"
                "对其造成4点伤害。\n<hr><ev>【<color=Keyword>进化时</color>】"
                "本随从获得【<color=Keyword>疾驰</color>】。</ev>",
                {("入场曲", "入场曲"), ("疾驰", "疾驰"), ("进化时", "进化时")},
            ),
            10821120: (
                "寡言的刺客·夏伊莉", "皇家护卫", 2,
                "【<color=Keyword>潜行</color>】\n<hr><ev>【<color=Keyword>进化时</color>】"
                "选择对手的战场上的1个随从，对手的回合结束前，使其获得"
                "「无法攻击随从或主战者」。</ev>",
                {("潜行", "潜行"), ("进化时", "进化时")},
            ),
            10831110: (
                "森绿的恩惠·喵鲁&圆滚滚2号&吉娜", "巫师", 2,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，"
                "对其造成1点伤害。使自己的所有手牌发动1次魔力增幅。\n<hr>"
                "<ev>【<color=Keyword>进化时</color>】发动与"
                "【<color=Keyword>入场曲</color>】相同的能力。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时")},
            ),
            10861130: (
                "亡灵猎人·格兰特", "主教", 5,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，"
                "破坏该随从。\n<hr><ev>【<color=Keyword>进化时</color>】发动与"
                "【<color=Keyword>入场曲</color>】相同的能力。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时")},
            ),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, (name, class_name, cost, text, abilities) in expected.items():
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
                    self.assertEqual(rows[0][:5], (name, class_name, "随从", cost, text))
                    self.assertTrue(all(rows[0][index] for index in (5, 6, 7)))
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT ability_keyword, raw_keyword FROM card_abilities WHERE card_id = ?",
                            (card_id,),
                        )),
                        abilities,
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?", (card_id,)
                        ).fetchone()[0], 0,
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM card_references WHERE card_id = ?", (card_id,)
                        ).fetchone()[0], 0,
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
                    ["tests/test_real_evolution_followup_batch.py"],
                )


class RealEvolutionFollowupBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 97) -> GameEngine:
        return _make_engine(self.rulebook, seed=seed)

    def play_real(self, engine: GameEngine, card_id: int) -> Unit:
        _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0))
        return engine.players[0].board[-1]

    def test_rule_shapes_filters_and_keyword_provenance_are_exact(self):
        expected = {
            (10222120, Trigger.FANFARE): [(EffectKind.DRAW_FILTERED, TargetKind.OWN_LEADER, 1)],
            (10222120, Trigger.EVOLVE): [(EffectKind.RESTORE_MANA, TargetKind.OWN_LEADER, 2)],
            (10311110, Trigger.FANFARE): [(EffectKind.BUFF_UNIT, TargetKind.ENEMY_UNIT, 0)],
            (10311110, Trigger.EVOLVE): [(EffectKind.BUFF_UNIT, TargetKind.ENEMY_UNIT, 0)],
            (10312110, Trigger.FANFARE): [(EffectKind.BUFF_UNIT, TargetKind.ALL_ENEMY_UNITS, 0)],
            (10312110, Trigger.SUPER_EVOLVE): [(EffectKind.BUFF_UNIT, TargetKind.ALL_ENEMY_UNITS, 0)],
            (10511110, Trigger.EVOLVE): [(EffectKind.DRAW, TargetKind.OWN_LEADER, 1)],
            (10561110, Trigger.FANFARE): [(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 2)],
            (10561110, Trigger.EVOLVE): [(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 2)],
            (10701110, Trigger.EVOLVE): [(EffectKind.DRAW_FILTERED, TargetKind.OWN_LEADER, 1)],
            (10712120, Trigger.FANFARE): [
                (EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 4),
                (EffectKind.ADD_COMBO, TargetKind.OWN_LEADER, 1),
            ],
            (10712120, Trigger.EVOLVE): [
                (EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 4),
                (EffectKind.ADD_COMBO, TargetKind.OWN_LEADER, 1),
            ],
            (10801110, Trigger.EVOLVE): [(EffectKind.BUFF_UNIT, TargetKind.SELF, 5)],
            (10802110, Trigger.FANFARE): [(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 4)],
            (10802110, Trigger.EVOLVE): [(EffectKind.ADD_KEYWORD, TargetKind.SELF, 0)],
            (10821120, Trigger.EVOLVE): [(EffectKind.ADD_ATTACK_RESTRICTION, TargetKind.ENEMY_UNIT, 0)],
            (10831110, Trigger.FANFARE): [
                (EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 1),
                (EffectKind.SPELLBOOST_HAND, TargetKind.ALL_OWN_HAND, 1),
            ],
            (10831110, Trigger.EVOLVE): [
                (EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 1),
                (EffectKind.SPELLBOOST_HAND, TargetKind.ALL_OWN_HAND, 1),
            ],
            (10861130, Trigger.FANFARE): [(EffectKind.DESTROY, TargetKind.ENEMY_UNIT, 0)],
            (10861130, Trigger.EVOLVE): [(EffectKind.DESTROY, TargetKind.ENEMY_UNIT, 0)],
        }
        for key, shapes in expected.items():
            with self.subTest(card_id=key[0], trigger=key[1]):
                operations = self.rulebook.operations_for(*key)
                self.assertEqual([(op.kind, op.target, op.amount) for op in operations], shapes)
        self.assertEqual(
            self.rulebook.operations_for(10222120, Trigger.FANFARE)[0].deck_filter.card_type,
            "法术",
        )
        self.assertEqual(
            self.rulebook.operations_for(10701110, Trigger.EVOLVE)[0].deck_filter.class_id,
            0,
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10802110), frozenset({"疾驰"})
        )

    def test_lachil_draws_only_spell_then_evolve_restores_two_mana(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        engine.players[0].deck = [_card(7100), _spell(7101)]
        source = self.play_real(engine, 10222120)
        hand_ids = [card.card_id for card in engine.players[0].hand]
        self.assertIn(7101, hand_ids)
        self.assertNotIn(7100, hand_ids)
        self.assertEqual(engine.players[0].mana, 6)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(engine.players[0].mana, 8)

    def test_unpierced_debuff_repeats_on_evolve_and_can_kill(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        target = _add_unit(engine, 1, 7200, life=2)
        source = self.play_real(engine, 10311110)
        _choose(engine, target.entity_id)
        self.assertEqual((target.health, target.max_health), (1, 1))
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)

    def test_unpierced_prayer_repeats_wide_debuff_on_super_evolve(self):
        engine = self.fresh_engine()
        _enable_super_evolution(engine)
        doomed = _add_unit(engine, 1, 7300, life=3)
        survivor = _add_unit(engine, 1, 7301, life=6)
        source = self.play_real(engine, 10312110)
        self.assertNotIn(doomed, engine.players[1].board)
        self.assertEqual((survivor.health, survivor.max_health), (3, 3))
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertNotIn(survivor, engine.players[1].board)

    def test_tanuki_draws_on_evolve_and_keeps_ambush(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        source = self.play_real(engine, 10511110)
        deck_before = len(engine.players[0].deck)
        self.assertTrue(source.has_keyword("潜行"))
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(deck_before - len(engine.players[0].deck), 1)

    def test_priest_repeats_selected_damage_and_keeps_guard(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        target = _add_unit(engine, 1, 7400, life=5)
        source = self.play_real(engine, 10561110)
        _choose(engine, target.entity_id)
        self.assertEqual(target.health, 3)
        self.assertTrue(source.has_guard)
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, target.entity_id)
        self.assertEqual(target.health, 1)

    def test_innocent_child_draws_only_neutral_card_on_evolve(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        neutral = _spell(7500, class_id=0, class_name="中立")
        nonneutral = _card(7501, class_id=1, class_name="精灵")
        engine.players[0].deck = [nonneutral, neutral]
        source = self.play_real(engine, 10701110)
        engine.apply(Evolve(0, source.entity_id))
        hand_ids = [card.card_id for card in engine.players[0].hand]
        self.assertIn(7500, hand_ids)
        self.assertNotIn(7501, hand_ids)

    def test_archer_repeats_damage_and_combo_gain_even_without_target(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        target = _add_unit(engine, 1, 7600, life=10)
        combo_before = engine.players[0].cards_played_this_turn
        source = self.play_real(engine, 10712120)
        _choose(engine, target.entity_id)
        combo_after_play = engine.players[0].cards_played_this_turn
        self.assertEqual(combo_after_play, combo_before + 2)
        self.assertEqual(target.health, 6)
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, target.entity_id)
        self.assertEqual(engine.players[0].cards_played_this_turn, combo_after_play + 1)
        self.assertEqual(target.health, 2)

        engine = self.fresh_engine()
        _enable_evolution(engine)
        source = self.play_real(engine, 10712120)
        combo_after_play = engine.players[0].cards_played_this_turn
        self.assertIsNone(engine.state.pending_choice)
        engine.apply(Evolve(0, source.entity_id))
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[0].cards_played_this_turn, combo_after_play + 1)

    def test_hansa_evolve_applies_five_five_after_normal_evolution_stats(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        source = self.play_real(engine, 10801110)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            (source.attack, source.health),
            (source.definition.attack + 7, source.definition.life + 7),
        )

    def test_alfred_starts_without_storm_then_gains_it_on_evolve(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        target = _add_unit(engine, 1, 7700, life=5)
        source = self.play_real(engine, 10802110)
        self.assertFalse(source.has_keyword("疾驰"))
        _choose(engine, target.entity_id)
        self.assertEqual(target.health, 1)
        engine.apply(Evolve(0, source.entity_id))
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertTrue(source.can_attack_leader)

    def test_shairi_evolve_restricts_enemy_and_matches_rl_mask(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(8000, 8040)],
            [_card(card_id) for card_id in range(9000, 9040)],
            class_a=1,
            class_b=1,
            seed=101,
            rulebook=self.rulebook,
        )
        env.reset(seed=101)
        env.players[0].max_mana = env.players[0].mana = 10
        env.players[0].turns_started = env.EVOLUTION_UNLOCK_TURN
        target = _add_unit(env.core, 1, 8050)
        _put_in_hand(env.core, self.repository.get(10821120))
        env.step(env.PLAY_OFFSET)
        source = env.players[0].board[-1]
        self.assertTrue(source.has_keyword("潜行"))
        self.assertTrue(env.action_mask()[env.EVOLVE_OFFSET])
        env.step(env.EVOLVE_OFFSET)
        self.assertEqual(
            [option.entity_id for option in env.core.state.pending_choice.options],
            [target.entity_id],
        )
        self.assertTrue(env.action_mask()[env.CHOICE_OFFSET])
        env.step(env.CHOICE_OFFSET)
        self.assertEqual(len(target.attack_restrictions), 1)
        self.assertEqual(
            target.attack_restrictions[0].restriction,
            AttackRestriction.CANNOT_ATTACK,
        )

    def test_witch_trio_repeats_damage_and_whole_hand_spellboost(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        tracked = _put_in_hand(engine, _spell(8100))
        target = _add_unit(engine, 1, 8101, life=4)
        source = self.play_real(engine, 10831110)
        _choose(engine, target.entity_id)
        self.assertEqual((target.health, tracked.spellboost_count), (3, 1))
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, target.entity_id)
        self.assertEqual((target.health, tracked.spellboost_count), (2, 2))

    def test_ghost_hunter_repeats_selected_destroy_on_evolve(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        first = _add_unit(engine, 1, 8200)
        second = _add_unit(engine, 1, 8201)
        source = self.play_real(engine, 10861130)
        _choose(engine, first.entity_id)
        self.assertNotIn(first, engine.players[1].board)
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, second.entity_id)
        self.assertNotIn(second, engine.players[1].board)


if __name__ == "__main__":
    unittest.main()
