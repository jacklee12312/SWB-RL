# -*- coding: utf-8 -*-
"""Direct audits for the real evolution-trigger coverage batch."""

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
from swb.engine.resolution import DamageType, GameEngine
from swb.engine.state import HandCard, Unit


BATCH_CARD_IDS = (
    10001120,
    10061130,
    10102110,
    10103110,
    10121120,
    10131140,
    10142130,
    10143140,
    10151150,
    10152120,
    10163120,
    10164130,
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


def _spell(card_id: int) -> CardDefinition:
    return _card(card_id, card_type="法术", attack=None, life=None)


def _make_engine(rulebook: RuleBook, *, seed: int = 83) -> GameEngine:
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
            10001120: (
                "叮当天使·莉亚", "中立", 2,
                "【<color=Keyword>守护</color>】\n"
                "【<color=Keyword>谢幕曲</color>】抽取1张卡牌。\n<hr>"
                "<ev>【<color=Keyword>进化时</color>】抽取1张卡牌。</ev>",
                {("守护", "守护"), ("谢幕曲", "谢幕曲"), ("进化时", "进化时")},
            ),
            10061130: (
                "圣翼战士", "主教", 4,
                "【<color=Keyword>入场曲</color>】选择自己的战场上的1个其他随从，"
                "使其+1/+1。\n<hr><ev>【<color=Keyword>进化时</color>】发动与"
                "【<color=Keyword>入场曲</color>】相同的能力。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时")},
            ),
            10102110: (
                "迸发的光明·阿波罗", "中立", 3,
                "【<color=Keyword>入场曲</color>】对对手的战场上的所有随从造成1点伤害。"
                "\n<hr><ev>【<color=Keyword>进化时</color>】发动与"
                "【<color=Keyword>入场曲</color>】相同的能力。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时")},
            ),
            10103110: (
                "爽朗的天宫·菲尔德亚", "中立", 2,
                "<ev>【<color=Keyword>进化时</color>】选择对手的战场上的1个随从，"
                "破坏该随从。</ev>",
                {("进化时", "进化时")},
            ),
            10121120: (
                "和平商人·艾尔涅丝塔", "皇家护卫", 6,
                "【<color=Keyword>入场曲</color>】使自己的战场上的其他所有随从+1/+1。"
                "\n<hr><ev>【<color=Keyword>进化时</color>】发动与"
                "【<color=Keyword>入场曲</color>】相同的能力。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时")},
            ),
            10131140: (
                "追梦的企鹅魔法师", "巫师", 4,
                "【<color=Keyword>入场曲</color>】抽取2张卡牌。\n<hr>"
                "<ev>【<color=Keyword>进化时</color>】使自己的所有手牌发动2次魔力增幅。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时")},
            ),
            10142130: (
                "读风者·杰鲁", "龙族", 3,
                "<sev>【<color=Keyword>超进化时</color>】选择自己的战场上的1个其他随从，"
                "使其获得【<color=Keyword>疾驰</color>】。</sev>",
                {("疾驰", "疾驰"), ("超进化时", "超进化时")},
            ),
            10143140: (
                "夜幕龙", "龙族", 9,
                "【<color=Keyword>入场曲</color>】使对手的战场上的所有随从-0/-9。"
                "\n<hr><sev>【<color=Keyword>超进化时</color>】抽取3张卡牌。</sev>",
                {("入场曲", "入场曲"), ("超进化时", "超进化时")},
            ),
            10151150: (
                "禁约恶魔", "梦魇", 5,
                "【<color=Keyword>入场曲</color>】抽取2张卡牌。对自己的主战者造成2点伤害。"
                "\n<hr><ev>【<color=Keyword>进化时</color>】选择对手的战场上的1个随从，"
                "对其造成6点伤害。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时")},
            ),
            10152120: (
                "梦魇·维莉", "梦魇", 2,
                "【<color=Keyword>入场曲</color>】对自己的主战者造成3点伤害。"
                "\n<hr><ev>【<color=Keyword>进化时</color>】回复自己的主战者5点生命值。</ev>",
                {("入场曲", "入场曲"), ("进化时", "进化时")},
            ),
            10163120: (
                "禁密的天宫·罗纳维罗", "主教", 4,
                "【<color=Keyword>潜行</color>】\n<hr>"
                "<ev>【<color=Keyword>进化时</color>】选择对手的战场上的1个随从，"
                "破坏该随从。</ev>",
                {("潜行", "潜行"), ("进化时", "进化时")},
            ),
            10164130: (
                "水之守护神·萨蕾法", "主教", 5,
                "【<color=Keyword>入场曲</color>】回复自己的主战者3点生命值。\n<hr>"
                "【<color=Keyword>守护</color>】\n<hr>"
                "<ev>【<color=Keyword>进化时</color>】对对手的战场上的所有随从造成3点伤害。</ev>",
                {("入场曲", "入场曲"), ("守护", "守护"), ("进化时", "进化时")},
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
                        WHERE c.card_id = ?
                        ORDER BY st.position
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
                    ["tests/test_real_evolution_batch.py"],
                )


class RealEvolutionBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 83) -> GameEngine:
        return _make_engine(self.rulebook, seed=seed)

    def play_real(self, engine: GameEngine, card_id: int) -> Unit:
        _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0))
        return engine.players[0].board[-1]

    def test_rule_trigger_and_operation_shapes_are_exact(self):
        expected = {
            (10001120, Trigger.LAST_WORDS): [(EffectKind.DRAW, TargetKind.OWN_LEADER, 1)],
            (10001120, Trigger.EVOLVE): [(EffectKind.DRAW, TargetKind.OWN_LEADER, 1)],
            (10061130, Trigger.FANFARE): [(EffectKind.BUFF_UNIT, TargetKind.OWN_UNIT, 1)],
            (10061130, Trigger.EVOLVE): [(EffectKind.BUFF_UNIT, TargetKind.OWN_UNIT, 1)],
            (10102110, Trigger.FANFARE): [(EffectKind.DAMAGE_UNIT, TargetKind.ALL_ENEMY_UNITS, 1)],
            (10102110, Trigger.EVOLVE): [(EffectKind.DAMAGE_UNIT, TargetKind.ALL_ENEMY_UNITS, 1)],
            (10103110, Trigger.EVOLVE): [(EffectKind.DESTROY, TargetKind.ENEMY_UNIT, 0)],
            (10121120, Trigger.FANFARE): [(EffectKind.BUFF_UNIT, TargetKind.ALL_OWN_UNITS, 1)],
            (10121120, Trigger.EVOLVE): [(EffectKind.BUFF_UNIT, TargetKind.ALL_OWN_UNITS, 1)],
            (10131140, Trigger.FANFARE): [(EffectKind.DRAW, TargetKind.OWN_LEADER, 2)],
            (10131140, Trigger.EVOLVE): [(EffectKind.SPELLBOOST_HAND, TargetKind.ALL_OWN_HAND, 2)],
            (10142130, Trigger.SUPER_EVOLVE): [(EffectKind.ADD_KEYWORD, TargetKind.OWN_UNIT, 0)],
            (10143140, Trigger.FANFARE): [(EffectKind.BUFF_UNIT, TargetKind.ALL_ENEMY_UNITS, 0)],
            (10143140, Trigger.SUPER_EVOLVE): [(EffectKind.DRAW, TargetKind.OWN_LEADER, 3)],
            (10151150, Trigger.FANFARE): [
                (EffectKind.DRAW, TargetKind.OWN_LEADER, 2),
                (EffectKind.DAMAGE_LEADER, TargetKind.OWN_LEADER, 2),
            ],
            (10151150, Trigger.EVOLVE): [(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 6)],
            (10152120, Trigger.FANFARE): [(EffectKind.DAMAGE_LEADER, TargetKind.OWN_LEADER, 3)],
            (10152120, Trigger.EVOLVE): [(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 5)],
            (10163120, Trigger.EVOLVE): [(EffectKind.DESTROY, TargetKind.ENEMY_UNIT, 0)],
            (10164130, Trigger.FANFARE): [(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 3)],
            (10164130, Trigger.EVOLVE): [(EffectKind.DAMAGE_UNIT, TargetKind.ALL_ENEMY_UNITS, 3)],
        }
        for key, shapes in expected.items():
            with self.subTest(card_id=key[0], trigger=key[1]):
                operations = self.rulebook.operations_for(*key)
                self.assertEqual(
                    [(op.kind, op.target, op.amount) for op in operations],
                    shapes,
                )
        for card_id, trigger in (
            (10061130, Trigger.FANFARE),
            (10061130, Trigger.EVOLVE),
            (10121120, Trigger.FANFARE),
            (10121120, Trigger.EVOLVE),
            (10142130, Trigger.SUPER_EVOLVE),
        ):
            self.assertTrue(self.rulebook.operations_for(card_id, trigger)[0].exclude_source)
        nightfall = self.rulebook.operations_for(10143140, Trigger.FANFARE)[0]
        self.assertEqual(nightfall.secondary_amount, -9)
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10142130),
            frozenset({"疾驰"}),
        )

    def test_ria_evolve_and_last_words_each_draw_one_and_keep_guard(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        source = self.play_real(engine, 10001120)
        self.assertTrue(source.has_guard)
        deck_before = len(engine.players[0].deck)

        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(deck_before - len(engine.players[0].deck), 1)
        engine.apply_damage(None, source, 99, DamageType.EFFECT, controller=1)
        engine._stabilize()
        self.assertEqual(deck_before - len(engine.players[0].deck), 2)

    def test_holywing_repeats_selected_buff_and_excludes_source(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        ally = _add_unit(engine, 0, 7100, attack=2, life=3)
        source = self.play_real(engine, 10061130)
        _choose(engine, ally.entity_id)
        self.assertEqual((ally.attack, ally.health), (3, 4))

        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [ally.entity_id],
        )
        _choose(engine, ally.entity_id)
        self.assertEqual((ally.attack, ally.health), (4, 5))
        self.assertEqual(
            (source.attack, source.health),
            (source.definition.attack + 2, source.definition.life + 2),
        )

    def test_apollo_repeats_simultaneous_enemy_wide_damage_on_evolve(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        doomed = _add_unit(engine, 1, 7200, life=2)
        survivor = _add_unit(engine, 1, 7201, life=3)
        source = self.play_real(engine, 10102110)
        self.assertEqual((doomed.health, survivor.health), (1, 2))

        engine.apply(Evolve(0, source.entity_id))
        self.assertNotIn(doomed, engine.players[1].board)
        self.assertEqual(survivor.health, 1)

    def test_targeted_evolve_destroy_cards_and_no_target_path(self):
        for card_id, static_keyword in ((10103110, None), (10163120, "潜行")):
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine()
                _enable_evolution(engine)
                target = _add_unit(engine, 1, 7300 + card_id)
                source = self.play_real(engine, card_id)
                if static_keyword is not None:
                    self.assertTrue(source.has_keyword(static_keyword))
                engine.apply(Evolve(0, source.entity_id))
                _choose(engine, target.entity_id)
                self.assertNotIn(target, engine.players[1].board)

                engine = self.fresh_engine()
                _enable_evolution(engine)
                source = self.play_real(engine, card_id)
                engine.apply(Evolve(0, source.entity_id))
                self.assertTrue(source.evolved)
                self.assertIsNone(engine.state.pending_choice)

    def test_ernesta_repeats_all_other_follower_buff(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        allies = [
            _add_unit(engine, 0, 7400, attack=1, life=2),
            _add_unit(engine, 0, 7401, attack=2, life=3),
        ]
        source = self.play_real(engine, 10121120)
        self.assertEqual([(u.attack, u.health) for u in allies], [(2, 3), (3, 4)])
        self.assertEqual(
            (source.attack, source.health),
            (source.definition.attack, source.definition.life),
        )

        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual([(u.attack, u.health) for u in allies], [(3, 4), (4, 5)])
        self.assertEqual(
            (source.attack, source.health),
            (source.definition.attack + 2, source.definition.life + 2),
        )

    def test_penguin_draws_two_then_spellboosts_all_hand_twice(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        tracked = _put_in_hand(engine, _spell(7500))
        deck_before = len(engine.players[0].deck)
        source = self.play_real(engine, 10131140)
        self.assertEqual(deck_before - len(engine.players[0].deck), 2)
        self.assertEqual(tracked.spellboost_count, 0)

        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(tracked.spellboost_count, 2)
        self.assertTrue(all(card.spellboost_count == 2 for card in engine.players[0].hand))

    def test_jeru_super_evolve_grants_storm_only_to_other_and_matches_rl_mask(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(8000, 8040)],
            [_card(card_id) for card_id in range(9000, 9040)],
            class_a=1,
            class_b=1,
            seed=89,
            rulebook=self.rulebook,
        )
        env.reset(seed=89)
        env.players[0].max_mana = 10
        env.players[0].mana = 10
        env.players[0].turns_started = env.FIRST_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN
        ally = _add_unit(env.core, 0, 8050)
        _put_in_hand(env.core, self.repository.get(10142130))
        env.step(env.PLAY_OFFSET)
        source = env.players[0].board[-1]
        self.assertFalse(source.has_keyword("疾驰"))

        action = env.SUPER_EVOLVE_OFFSET + 1
        self.assertTrue(env.action_mask()[action])
        env.step(action)
        self.assertEqual(
            [option.entity_id for option in env.core.state.pending_choice.options],
            [ally.entity_id],
        )
        mask = env.action_mask()
        self.assertTrue(mask[env.CHOICE_OFFSET])
        self.assertFalse(mask[env.CHOICE_OFFSET + 1])
        env.step(env.CHOICE_OFFSET)
        self.assertTrue(ally.has_keyword("疾驰"))
        self.assertFalse(source.has_keyword("疾驰"))

    def test_nightfall_debuff_collects_deaths_then_super_evolve_draws_three(self):
        engine = self.fresh_engine()
        _enable_super_evolution(engine)
        doomed = _add_unit(engine, 1, 7600, life=8)
        survivor = _add_unit(engine, 1, 7601, life=10)
        source = self.play_real(engine, 10143140)
        self.assertNotIn(doomed, engine.players[1].board)
        self.assertEqual((survivor.health, survivor.max_health), (1, 1))
        deck_before = len(engine.players[0].deck)

        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(deck_before - len(engine.players[0].deck), 3)

    def test_forbidden_demon_draws_self_damages_then_evolve_damages_target(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        engine.players[0].health = 20
        target = _add_unit(engine, 1, 7700, life=6)
        deck_before = len(engine.players[0].deck)
        source = self.play_real(engine, 10151150)
        self.assertEqual(deck_before - len(engine.players[0].deck), 2)
        self.assertEqual(engine.players[0].health, 18)

        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)

    def test_nightmare_very_self_damages_then_evolve_heals(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        engine.players[0].health = 10
        source = self.play_real(engine, 10152120)
        self.assertEqual(engine.players[0].health, 7)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(engine.players[0].health, 12)

    def test_salepha_heals_has_guard_and_evolve_damages_all_enemies(self):
        engine = self.fresh_engine()
        _enable_evolution(engine)
        engine.players[0].health = 10
        doomed = _add_unit(engine, 1, 7800, life=3)
        survivor = _add_unit(engine, 1, 7801, life=5)
        source = self.play_real(engine, 10164130)
        self.assertEqual(engine.players[0].health, 13)
        self.assertTrue(source.has_guard)

        engine.apply(Evolve(0, source.entity_id))
        self.assertNotIn(doomed, engine.players[1].board)
        self.assertEqual(survivor.health, 2)


if __name__ == "__main__":
    unittest.main()
