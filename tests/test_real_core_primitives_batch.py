# -*- coding: utf-8 -*-
"""Direct audits for real cards expressible through established core primitives."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard
from swb.engine.effects import ConditionType, EffectKind, ExprType, TargetKind
from swb.engine.resolution import GameEngine
from swb.engine.state import DeathCause, HandCard, Unit


BATCH_CARD_IDS = (
    10032120,
    10101110,
    10203110,
    10301110,
    10372120,
    10421110,
    10513310,
    10611120,
    10621110,
    10651310,
    10704120,
    10752120,
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


def _make_engine(rulebook: RuleBook, *, seed: int = 131) -> GameEngine:
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
    cost: int = 1,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, attack=attack, life=life, cost=cost),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[owner].board.append(unit)
    return unit


def _choose(engine: GameEngine, entity_id: int) -> None:
    command = next(
        command
        for command in engine.legal_commands()
        if isinstance(command, Choose)
        and command.option_id in {f"entity:{entity_id}", f"hand:{entity_id}"}
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
            10032120: (
                "魔焰毁灭者", "巫师", "随从", 10,
                "【<color=Keyword>魔力增幅时</color>】使本卡牌的费用-1。",
                {("魔力增幅", "魔力增幅时")},
            ),
            10101110: (
                "贪婪的智天使·露比", "中立", "随从", 2,
                "【<color=Keyword>入场曲</color>】选择自己的1张手牌，使其返回牌组。抽取1张卡牌。",
                {("入场曲", "入场曲")},
            ),
            10203110: (
                "联结的天使·蕾娜", "中立", "随从", 7,
                "<ev>【<color=Keyword>进化时</color>】使自己的战场上的所有进化前的随从进化。</ev>",
                {("进化时", "进化时")},
            ),
            10301110: (
                "涸绝的使徒", "中立", "随从", 4,
                "【<color=Keyword>入场曲</color>】选择战场上的1个其他随从，使其+4/-4。",
                {("入场曲", "入场曲")},
            ),
            10372120: (
                "现场工程师", "超越者", "随从", 5,
                "【<color=Keyword>入场曲</color>】选择自己的1张手牌，舍弃该手牌。抽取3张卡牌。",
                {("入场曲", "入场曲")},
            ),
            10421110: (
                "信念腿法·兰德尔", "皇家护卫", "随从", 2,
                "【<color=Keyword>爆能强化</color>_5】本随从获得【<color=Keyword>疾驰</color>】。",
                {("爆能强化", "爆能强化"), ("疾驰", "疾驰")},
            ),
            10513310: (
                "优雅的虫风花", "精灵", "法术", 3,
                "抽取X张卡牌。X为自己的【<color=Keyword>连击</color>】。",
                {("连击", "连击")},
            ),
            10611120: (
                "伊甸之猴", "精灵", "随从", 3,
                "【<color=Keyword>入场曲</color>】【<color=Keyword>连击</color>_3】本随从进化。",
                {("入场曲", "入场曲"), ("连击", "连击")},
            ),
            10621110: (
                "勇烈的士兵", "皇家护卫", "随从", 2,
                "【<color=Keyword>爆能强化</color>_3】本随从+1/+1。\n<hr>【<color=Keyword>突进</color>】",
                {("爆能强化", "爆能强化"), ("突进", "突进")},
            ),
            10651310: (
                "天眼授予", "梦魇", "法术", 3,
                "抽取2张卡牌。【<color=Keyword>唤灵</color>_4】回复自己的主战者2点生命值。",
                {("死灵术", "唤灵")},
            ),
            10704120: (
                "巴别隆市长·埃尔塔罗", "中立", "随从", 3,
                "【<color=Keyword>潜行</color>】\n自己的回合结束时，抽取1张卡牌。",
                {("潜行", "潜行")},
            ),
            10752120: (
                "乌鸦杂耍师", "梦魇", "随从", 6,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，破坏该随从。发动【<color=Keyword>亡者召还</color>_2】。",
                {("亡者召还", "亡者召还"), ("入场曲", "入场曲")},
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
                            "SELECT ability_keyword, raw_keyword FROM card_abilities WHERE card_id = ?",
                            (card_id,),
                        )),
                        values[5],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?", (card_id,)
                        ).fetchone()[0],
                        0,
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM card_references WHERE card_id = ?", (card_id,)
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
                    ["tests/test_real_core_primitives_batch.py"],
                )


class RealCorePrimitiveBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 131) -> GameEngine:
        return _make_engine(self.rulebook, seed=seed)

    def play_real(self, engine: GameEngine, card_id: int, mode_id: str = "normal"):
        _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0, mode_id))
        return engine.players[0].board[-1] if engine.players[0].board else None

    def test_rule_shapes_modes_filters_and_passives_are_exact(self):
        self.assertEqual(self.rulebook.spellboost_cost_reduction(10032120), 1)
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10421110), frozenset({"疾驰"})
        )
        combo_draw = self.rulebook.operations_for(10513310, Trigger.PLAY)[0]
        self.assertEqual((combo_draw.kind, combo_draw.target), (EffectKind.DRAW, TargetKind.OWN_LEADER))
        self.assertEqual(combo_draw.amount_expr.type, ExprType.CONTROLLER_COMBO)

        monkey = self.rulebook.operations_for(10611120, Trigger.FANFARE)[0]
        self.assertEqual(monkey.kind, EffectKind.CONDITIONAL)
        self.assertEqual(monkey.conditions[0].type, ConditionType.CONTROLLER_COMBO_AT_LEAST)
        self.assertEqual(monkey.then_operations[0].kind, EffectKind.EVOLVE_UNIT)

        for card_id, mode_id, cost in ((10421110, "enhance_5", 5), (10621110, "enhance_3", 3)):
            with self.subTest(card_id=card_id):
                mode = self.rulebook.modes_for(card_id)[0]
                self.assertEqual((mode.mode_id, mode.cost), (mode_id, cost))

        evolve_all = self.rulebook.operations_for(10203110, Trigger.EVOLVE)[0]
        self.assertEqual((evolve_all.kind, evolve_all.target), (EffectKind.EVOLVE_UNIT, TargetKind.ALL_OWN_UNITS))
        self.assertFalse(evolve_all.board_filter.evolved)
        other = self.rulebook.operations_for(10301110, Trigger.FANFARE)[0]
        self.assertTrue(other.exclude_source)
        self.assertEqual((other.target, other.amount, other.secondary_amount), (TargetKind.ANY_UNIT, 4, -4))

    def test_magic_flame_destroyer_reduces_cost_per_spellboost(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        tracked = _put_in_hand(engine, self.repository.get(10032120))
        _put_in_hand(engine, self.repository.get(10513310))
        engine.apply(PlayCard(0, 0))
        self.assertEqual((tracked.spellboost_count, tracked.current_cost), (1, 9))

    def test_graceful_insectflower_draws_current_combo_count(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        engine.players[0].deck = [_card(card_id) for card_id in range(7100, 7105)]
        engine.players[0].cards_played_this_turn = 2
        _put_in_hand(engine, self.repository.get(10513310))
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].deck), 2)
        self.assertEqual(engine.players[0].cards_played_this_turn, 3)

    def test_eden_monkey_auto_evolves_at_combo_three_without_ep(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        engine.players[0].cards_played_this_turn = 2
        ep_before = engine.players[0].evolution_points
        source = self.play_real(engine, 10611120)
        self.assertTrue(source.evolved)
        self.assertEqual(engine.players[0].evolution_points, ep_before)
        self.assertEqual(engine.players[0].followers_evolved_this_match, 1)

    def test_randall_starts_without_storm_and_enhance_grants_it(self):
        normal = self.fresh_engine()
        _clear_hand(normal)
        normal.players[0].mana = 2
        base = self.play_real(normal, 10421110)
        self.assertFalse(base.has_keyword("疾驰"))

        enhanced = self.fresh_engine()
        _clear_hand(enhanced)
        enhanced.players[0].mana = 5
        source = self.play_real(enhanced, 10421110, "enhance_5")
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertTrue(source.can_attack_leader)
        self.assertEqual(enhanced.players[0].mana, 0)

    def test_ruby_returns_selected_hand_card_then_draws_one(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        engine.players[0].deck = [_card(card_id) for card_id in range(7200, 7205)]
        returned = _put_in_hand(engine, _card(7210))
        self.play_real(engine, 10101110)
        _choose(engine, returned.entity_id)
        self.assertEqual(len(engine.players[0].deck), 5)
        self.assertFalse(any(card.entity_id == returned.entity_id for card in engine.players[0].hand))
        self.assertTrue(any(card.card_id == 7210 for card in engine.players[0].deck))

    def test_eltaro_keeps_ambush_and_draws_at_own_turn_end(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        engine.players[0].deck = [_card(card_id) for card_id in range(7300, 7305)]
        source = self.play_real(engine, 10704120)
        self.assertTrue(source.has_keyword("潜行"))
        deck_before = len(engine.players[0].deck)
        engine.apply(EndTurn(0))
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_reina_evolves_every_other_unevolved_ally_without_extra_ep(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        first = _add_unit(engine, 0, 7400)
        second = _add_unit(engine, 0, 7401)
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        engine.players[0].evolution_points = 2
        source = self.play_real(engine, 10203110)
        engine.apply(Evolve(0, source.entity_id))
        self.assertTrue(all(unit.evolved for unit in (source, first, second)))
        self.assertEqual(engine.players[0].evolution_points, 1)
        self.assertEqual(engine.players[0].followers_evolved_this_match, 3)

    def test_apostle_can_target_either_side_but_never_itself(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        ally = _add_unit(engine, 0, 7500, life=5)
        enemy = _add_unit(engine, 1, 7501, life=4)
        source = self.play_real(engine, 10301110)
        options = {option.entity_id for option in engine.state.pending_choice.options}
        self.assertEqual(options, {ally.entity_id, enemy.entity_id})
        self.assertNotIn(source.entity_id, options)
        _choose(engine, enemy.entity_id)
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertIn(source, engine.players[0].board)

    def test_brave_soldier_enhance_buffs_and_preserves_rush(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        engine.players[0].mana = 3
        definition = self.repository.get(10621110)
        source = self.play_real(engine, 10621110, "enhance_3")
        self.assertEqual((source.attack, source.health), (definition.attack + 1, definition.life + 1))
        self.assertTrue(source.has_keyword("突进"))
        self.assertTrue(source.can_attack_units)

    def test_heavensight_draws_two_and_necromancy_heals_only_with_four_shadows(self):
        enough = self.fresh_engine()
        _clear_hand(enough)
        enough.players[0].deck = [_card(card_id) for card_id in range(7600, 7605)]
        enough.players[0].health = 15
        enough.players[0].shadows = 4
        _put_in_hand(enough, self.repository.get(10651310))
        enough.apply(PlayCard(0, 0))
        self.assertEqual((len(enough.players[0].deck), enough.players[0].health), (3, 17))
        self.assertEqual(enough.players[0].shadows, 1)

        short = self.fresh_engine()
        _clear_hand(short)
        short.players[0].deck = [_card(card_id) for card_id in range(7700, 7705)]
        short.players[0].health = 15
        short.players[0].shadows = 3
        _put_in_hand(short, self.repository.get(10651310))
        short.apply(PlayCard(0, 0))
        self.assertEqual((len(short.players[0].deck), short.players[0].health), (3, 15))
        self.assertEqual(short.players[0].shadows, 4)

    def test_field_engineer_discards_then_draws_three_and_no_target_still_draws(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        engine.players[0].deck = [_card(card_id) for card_id in range(7800, 7805)]
        discarded = _put_in_hand(engine, _card(7810))
        self.play_real(engine, 10372120)
        _choose(engine, discarded.entity_id)
        self.assertEqual(len(engine.players[0].deck), 2)
        self.assertTrue(any(card.entity_id == discarded.entity_id for card in engine.players[0].graveyard))

        no_target = self.fresh_engine()
        _clear_hand(no_target)
        no_target.players[0].deck = [_card(card_id) for card_id in range(7900, 7905)]
        self.play_real(no_target, 10372120)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertEqual(len(no_target.players[0].deck), 2)

    def test_crow_juggler_destroys_then_reanimates_even_without_enemy(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        reanimated = _card(8000, cost=2, attack=2, life=3)
        engine._record_destroyed_follower(0, reanimated, DeathCause.EFFECT_DESTROY)
        target = _add_unit(engine, 1, 8001)
        self.play_real(engine, 10752120)
        _choose(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)
        self.assertIn(8000, [unit.definition.card_id for unit in engine.players[0].board])

        no_enemy = self.fresh_engine()
        _clear_hand(no_enemy)
        no_enemy._record_destroyed_follower(0, reanimated, DeathCause.EFFECT_DESTROY)
        self.play_real(no_enemy, 10752120)
        self.assertIsNone(no_enemy.state.pending_choice)
        self.assertIn(8000, [unit.definition.card_id for unit in no_enemy.players[0].board])


if __name__ == "__main__":
    unittest.main()
