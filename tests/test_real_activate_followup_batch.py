# -*- coding: utf-8 -*-
"""Direct audits for the activation-only amulet legality slice and real cards."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import ActivateAmulet, Choose, PlayCard
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, HandCard, Unit


BATCH_CARD_IDS = (
    10001210,
    10162210,
    10163220,
    10261210,
    10312210,
    10342210,
    10761210,
    10763210,
)


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 6),
        class_name=overrides.get("class_name", "主教"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 3),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _make_engine(rulebook: RuleBook, *, seed: int = 149) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=6,
        class_b=6,
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
        origin=CardOrigin.DECK,
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
    life: int = 3,
    keywords: frozenset[str] = frozenset(),
) -> Unit:
    unit = Unit.summon(
        _card(card_id, attack=attack, life=life, keywords=keywords),
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
            10001210: (
                "侦探的放大镜", "中立", 2,
                "【<color=Keyword>启动</color>】破坏本卡牌。选择对手的战场上的1个随从，"
                "使其失去【<color=Keyword>守护</color>】。",
                {("守护", "守护"), ("策动", "启动")},
            ),
            10162210: (
                "禁密的圣地", "主教", 2,
                "费用1【<color=Keyword>启动</color>】选择自己的战场上的1个随从，"
                "使其+1/+1。回复自己的主战者1点生命值。",
                {("策动", "启动")},
            ),
            10163220: (
                "邪教法器", "主教", 6,
                "【<color=Keyword>启动</color>】破坏本卡牌。破坏战场上的所有随从。",
                {("策动", "启动")},
            ),
            10261210: (
                "流光香炉", "主教", 2,
                "费用1【<color=Keyword>启动</color>】破坏本卡牌。回复自己的主战者4点生命值。",
                {("策动", "启动")},
            ),
            10312210: (
                "不弑之乡", "精灵", 3,
                "【<color=Keyword>入场曲</color>】选择自己的1张手牌，舍弃该手牌。"
                "抽取2张卡牌。\n<hr>【<color=Keyword>启动</color>】破坏本卡牌。"
                "选择对手的战场上的1个随从，使其-0/-2。",
                {("入场曲", "入场曲"), ("策动", "启动")},
            ),
            10342210: (
                "侮蔑之国", "龙族", 2,
                "【<color=Keyword>吟唱</color>_4】\n<hr>费用1【<color=Keyword>启动</color>】"
                "对战场上的所有随从造成1点伤害。",
                {("倒数", "吟唱"), ("策动", "启动")},
            ),
            10761210: (
                "阳光耳饰", "主教", 1,
                "【<color=Keyword>入场曲</color>】选择自己的1张手牌，使其返回牌组。"
                "抽取1张卡牌。\n<hr>【<color=Keyword>启动</color>】破坏本卡牌。"
                "发动与【<color=Keyword>入场曲</color>】相同的能力。",
                {("入场曲", "入场曲"), ("策动", "启动")},
            ),
            10763210: (
                "海蚀三叉戟", "主教", 3,
                "【<color=Keyword>入场曲</color>】选择对手的战场上的1个随从，"
                "对其造成4点伤害。\n<hr>【<color=Keyword>启动</color>】破坏本卡牌。"
                "选择对手的战场上的1个随从，对其造成2点伤害。",
                {("入场曲", "入场曲"), ("策动", "启动")},
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
                    self.assertEqual(rows[0][:5], (name, class_name, "护符", cost, text))
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
                    ["tests/test_real_activate_followup_batch.py"],
                )


class RealActivateFollowupBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 149) -> GameEngine:
        return _make_engine(self.rulebook, seed=seed)

    def play_real(self, engine: GameEngine, card_id: int) -> Amulet:
        _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0))
        return next(
            entity
            for entity in engine.players[0].board
            if isinstance(entity, Amulet) and entity.definition.card_id == card_id
        )

    def test_activation_costs_rule_shapes_and_keyword_provenance_are_exact(self):
        expected_costs = {
            10001210: 0,
            10162210: 1,
            10163220: 0,
            10261210: 1,
            10312210: 0,
            10342210: 1,
            10761210: 0,
            10763210: 0,
        }
        self.assertEqual(
            {card_id: self.rulebook.activation_for(card_id).cost for card_id in expected_costs},
            expected_costs,
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10001210), frozenset({"守护"})
        )
        self.assertEqual(self.rulebook.countdown_for(10342210), 4)
        self.assertEqual(
            [op.kind for op in self.rulebook.operations_for(10163220, Trigger.ACTIVATE)],
            [EffectKind.DESTROY, EffectKind.DESTROY],
        )
        self.assertEqual(
            self.rulebook.operations_for(10342210, Trigger.ACTIVATE)[0].target,
            TargetKind.ALL_UNITS,
        )

    def test_detectives_magnifier_is_playable_then_removes_enemy_guard(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        target = _add_unit(engine, 1, 7000, keywords=frozenset({"守护"}))
        source = self.play_real(engine, 10001210)
        self.assertIn(ActivateAmulet(0, source.entity_id), engine.legal_commands())
        engine.apply(ActivateAmulet(0, source.entity_id))
        _choose(engine, target.entity_id)
        self.assertNotIn(source, engine.players[0].board)
        self.assertFalse(target.has_guard)

    def test_secret_sanctuary_buffs_heals_and_can_activate_only_once_per_turn(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        engine.players[0].health = 18
        target = _add_unit(engine, 0, 7100, attack=2, life=2)
        source = self.play_real(engine, 10162210)
        command = ActivateAmulet(0, source.entity_id)
        mana_before = engine.players[0].mana
        engine.apply(command)
        _choose(engine, target.entity_id)
        self.assertEqual((target.attack, target.health, engine.players[0].health), (3, 3, 19))
        self.assertEqual(engine.players[0].mana, mana_before - 1)
        self.assertNotIn(command, engine.legal_commands())
        self.assertIn(source, engine.players[0].board)

    def test_cult_relic_destroys_itself_and_all_followers(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        own = _add_unit(engine, 0, 7200)
        enemy = _add_unit(engine, 1, 7201)
        source = self.play_real(engine, 10163220)
        engine.apply(ActivateAmulet(0, source.entity_id))
        self.assertNotIn(source, engine.players[0].board)
        self.assertNotIn(own, engine.players[0].board)
        self.assertNotIn(enemy, engine.players[1].board)

    def test_flowing_light_censer_destroys_itself_and_heals_four(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        engine.players[0].health = 14
        source = self.play_real(engine, 10261210)
        mana_before = engine.players[0].mana
        engine.apply(ActivateAmulet(0, source.entity_id))
        self.assertEqual(engine.players[0].health, 18)
        self.assertEqual(engine.players[0].mana, mana_before - 1)
        self.assertNotIn(source, engine.players[0].board)

    def test_unpierced_land_discards_draws_then_activation_debuffs_lethally(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        engine.players[0].deck = [_card(card_id) for card_id in range(7300, 7305)]
        discarded = _put_in_hand(engine, _card(7310))
        target = _add_unit(engine, 1, 7311, life=2)
        source = self.play_real(engine, 10312210)
        _choose(engine, discarded.entity_id)
        self.assertEqual(len(engine.players[0].deck), 3)
        engine.apply(ActivateAmulet(0, source.entity_id))
        _choose(engine, target.entity_id)
        self.assertNotIn(source, engine.players[0].board)
        self.assertNotIn(target, engine.players[1].board)

    def test_land_of_disdain_countdown_activation_damages_all_followers(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        own = _add_unit(engine, 0, 7400, life=1)
        enemy = _add_unit(engine, 1, 7401, life=1)
        source = self.play_real(engine, 10342210)
        self.assertEqual(source.countdown, 4)
        engine.apply(ActivateAmulet(0, source.entity_id))
        self.assertNotIn(own, engine.players[0].board)
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertIn(source, engine.players[0].board)

    def test_sunshine_earrings_cycles_on_play_and_repeats_on_activation(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        engine.players[0].deck = [_card(card_id) for card_id in range(7500, 7505)]
        first = _put_in_hand(engine, _card(7510))
        source = self.play_real(engine, 10761210)
        _choose(engine, first.entity_id)
        second = engine.players[0].hand[0]
        engine.apply(ActivateAmulet(0, source.entity_id))
        _choose(engine, second.entity_id)
        self.assertEqual(len(engine.players[0].deck), 5)
        self.assertEqual(len(engine.players[0].hand), 1)
        self.assertNotIn(source, engine.players[0].board)

    def test_sea_erosion_trident_deals_four_then_activation_deals_two(self):
        engine = self.fresh_engine()
        _clear_hand(engine)
        target = _add_unit(engine, 1, 7600, life=6)
        source = self.play_real(engine, 10763210)
        _choose(engine, target.entity_id)
        self.assertEqual(target.health, 2)
        engine.apply(ActivateAmulet(0, source.entity_id))
        _choose(engine, target.entity_id)
        self.assertNotIn(source, engine.players[0].board)
        self.assertNotIn(target, engine.players[1].board)

    def test_real_activation_only_amulet_matches_rl_play_and_activate_masks(self):
        deck = [_card(card_id) for card_id in range(8000, 8040)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=6,
            class_b=6,
            seed=157,
            rulebook=self.rulebook,
        )
        env.reset(seed=157)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].mana = 10
        _put_in_hand(env.core, self.repository.get(10163220))
        self.assertTrue(env.action_mask()[env.PLAY_OFFSET])
        env.step(env.PLAY_OFFSET)
        source = next(entity for entity in env.players[0].board if isinstance(entity, Amulet))
        self.assertTrue(env.action_mask()[env.EVOLVE_OFFSET])
        self.assertEqual(
            env._decode_action(env.EVOLVE_OFFSET),
            ActivateAmulet(0, source.entity_id),
        )


if __name__ == "__main__":
    unittest.main()
