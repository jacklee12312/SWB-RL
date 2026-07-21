# -*- coding: utf-8 -*-
"""Direct audits for spell Enhance routing, replacement, and real cards."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard
from swb.engine.effects import ConditionType, EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


BATCH_CARD_IDS = (
    10222310,
    10361310,
    10522310,
    10523310,
    10623310,
    10862310,
)
TOKEN_CARD_ID = 90021350


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 2),
        class_name=overrides.get("class_name", "皇家护卫"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 3),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _make_engine(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 191,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=2,
        class_b=2,
        seed=seed,
        rulebook=rulebook,
        card_resolver=repository.get,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _clear_hand(engine: GameEngine) -> None:
    engine.players[0].hand.clear()
    engine.players[0].hand_entity_ids.clear()


def _put_in_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    origin: CardOrigin = CardOrigin.DECK,
) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=origin,
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


def _add_enemy(engine: GameEngine, card_id: int, *, life: int) -> Unit:
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

    def test_database_texts_keywords_modes_and_references_match_audit(self):
        expected = {
            10222310: (
                "三将姬的乱击", "皇家护卫", "法术", 2,
                "对对手的战场上的随机1个随从造成4点伤害。\n"
                "【<color=Keyword>爆能强化</color>_4】改为对对手的战场上的随机3个随从造成4点伤害。",
                {("爆能强化", "爆能强化")}, set(),
            ),
            10361310: (
                "圣辉闪烁", "主教", "法术", 4,
                "对对手的战场上的所有随从造成3点伤害。\n"
                "【<color=Keyword>爆能强化</color>_8】抽取3张卡牌。回复自己的主战者3点生命值。",
                {("爆能强化", "爆能强化")}, set(),
            ),
            10522310: (
                "温柔援军", "皇家护卫", "法术", 2,
                "召唤2个『<color=Keyword>骑士</color>』。\n"
                "【<color=Keyword>爆能强化</color>_4】改为召唤4个。",
                {("爆能强化", "爆能强化")}, {(0, 90021110, "骑士")},
            ),
            10523310: (
                "荣耀的丽金花", "皇家护卫", "法术", 3,
                "将2张『<color=Keyword>闪耀的金币</color>』加入手牌。\n"
                "【<color=Keyword>爆能强化</color>_5】改为4张。",
                {("爆能强化", "爆能强化")}, {(0, 90021350, "闪耀的金币")},
            ),
            10623310: (
                "惨烈的天剑", "皇家护卫", "法术", 1,
                "【<color=Keyword>模式</color>】选择1个能力发动。\n"
                "<ridx=0>（1）抽取1张卡牌。</ridx>\n"
                "<ridx=1>（2）对对手的战场上的随机1个随从造成3点伤害。</ridx>\n"
                "【<color=Keyword>爆能强化</color>_3】改为发动所有能力。",
                {("模式", "模式"), ("爆能强化", "爆能强化")}, set(),
            ),
            10862310: (
                "威胁的残渣", "主教", "法术", 2,
                "选择对手的战场上的1个生命值为3或以下的随从，使其消失。\n"
                "【<color=Keyword>爆能强化</color>_5】改为使对手的战场上的所有生命值为3或以下的随从消失。",
                {("爆能强化", "爆能强化")}, set(),
            ),
            90021350: (
                "闪耀的金币", "皇家护卫", "法术", 0,
                "【<color=Keyword>模式</color>】选择1个能力发动。\n"
                "<ridx=0>（1）抽取1张卡牌。</ridx>\n"
                "<ridx=1>（2）对对手的战场上的随机1个随从造成2点伤害。</ridx>",
                {("模式", "模式")}, set(),
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
                    references = set(connection.execute(
                        "SELECT position, referenced_card_id, referenced_name "
                        "FROM card_references WHERE card_id = ?",
                        (card_id,),
                    ))
                    self.assertEqual(references, values[6])
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_collectible_cards_and_generated_coin_have_exact_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in BATCH_CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_spell_enhance_batch.py"],
                )

        token_report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        token = next(
            card
            for card in token_report["cards"]
            if card["card_id"] == TOKEN_CARD_ID
        )
        self.assertEqual(token["category"], "entry_behavior_complete")
        self.assertEqual(token["explicit_coverage"], "exact")
        self.assertEqual(
            token["authored_producers"],
            [
                {
                    "source_card_id": 10521120,
                    "entry_kind": "add_card",
                    "rule_file": "real_selected_hand_stat_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10523310,
                    "entry_kind": "add_card",
                    "rule_file": "real_spell_enhance_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10822110,
                    "entry_kind": "add_card",
                    "rule_file": "real_royal_rune_bishop_mixed_batch.json",
                    "rule_group": "rules",
                },
            ],
        )


class RealSpellEnhanceBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 191) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def play_real(
        self,
        engine: GameEngine,
        card_id: int,
        mode_id: str = "normal",
    ) -> None:
        _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0, mode_id))

    def test_rule_shapes_make_append_and_replacement_explicit(self):
        expected = {
            10222310: ("enhance_4", True),
            10361310: ("enhance_8", False),
            10522310: ("enhance_4", True),
            10523310: ("enhance_5", True),
            10623310: ("enhance_3", True),
            10862310: ("enhance_5", True),
        }
        for card_id, values in expected.items():
            with self.subTest(card_id=card_id):
                mode = self.rulebook.modes_for(card_id)[0]
                self.assertEqual(
                    (mode.mode_id, mode.replace_base_operations),
                    values,
                )
        low_health = self.rulebook.operations_for(10862310, Trigger.PLAY)[0]
        self.assertTrue(low_health.requires_target)
        self.assertEqual(
            low_health.conditions[0].type,
            ConditionType.TARGET_HEALTH_AT_MOST,
        )
        coin = self.rulebook.operations_for(TOKEN_CARD_ID, Trigger.PLAY)[0]
        self.assertEqual(coin.kind, EffectKind.CHOOSE_ONE)
        self.assertEqual(len(coin.choose_one_options), 2)

    def test_three_commanders_onslaught_replaces_one_target_with_three(self):
        normal = self.fresh_engine(seed=193)
        _clear_hand(normal)
        normal_targets = [_add_enemy(normal, 3000 + i, life=4) for i in range(3)]
        self.play_real(normal, 10222310)
        self.assertEqual(sum(target in normal.players[1].board for target in normal_targets), 2)

        enhanced = self.fresh_engine(seed=193)
        _clear_hand(enhanced)
        enhanced_targets = [_add_enemy(enhanced, 3010 + i, life=4) for i in range(3)]
        self.play_real(enhanced, 10222310, "enhance_4")
        self.assertEqual(sum(target in enhanced.players[1].board for target in enhanced_targets), 0)
        self.assertEqual(enhanced.players[0].board, [])
        self.assertEqual(enhanced.players[0].graveyard[-1].definition.card_id, 10222310)

    def test_holy_radiance_appends_draw_and_heal_after_base_damage(self):
        engine = self.fresh_engine(seed=197)
        _clear_hand(engine)
        engine.players[0].health = 14
        engine.players[0].deck = [_card(3100 + i) for i in range(5)]
        targets = [_add_enemy(engine, 3110 + i, life=3) for i in range(2)]

        self.play_real(engine, 10361310, "enhance_8")

        self.assertFalse(any(target in engine.players[1].board for target in targets))
        self.assertEqual(len(engine.players[0].deck), 2)
        self.assertEqual(engine.players[0].health, 17)

    def test_gentle_reinforcements_replacement_respects_board_capacity(self):
        normal = self.fresh_engine(seed=199)
        _clear_hand(normal)
        self.play_real(normal, 10522310)
        self.assertEqual(
            [unit.definition.card_id for unit in normal.players[0].board],
            [90021110, 90021110],
        )

        enhanced = self.fresh_engine(seed=199)
        _clear_hand(enhanced)
        for index in range(2):
            unit = Unit.summon(
                _card(3200 + index),
                entity_id=enhanced.state.allocate_entity_id(),
            )
            enhanced.players[0].board.append(unit)
        self.play_real(enhanced, 10522310, "enhance_4")
        self.assertEqual(len(enhanced.players[0].board), 5)
        self.assertEqual(
            sum(unit.definition.card_id == 90021110 for unit in enhanced.players[0].board),
            3,
        )

    def test_glorious_marigold_creates_four_exact_playable_coins(self):
        engine = self.fresh_engine(seed=211)
        _clear_hand(engine)
        target = _add_enemy(engine, 3300, life=4)
        self.play_real(engine, 10523310, "enhance_5")

        coins = [card for card in engine.players[0].hand if card.card_id == TOKEN_CARD_ID]
        self.assertEqual(len(coins), 4)
        self.assertTrue(all(card.origin is CardOrigin.TOKEN for card in coins))

        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        engine.apply(Choose(0, "choose_one:damage"))
        self.assertEqual(target.health, 2)
        self.assertEqual(engine.players[0].graveyard[-1].definition.card_id, TOKEN_CARD_ID)

    def test_glittering_gold_draw_mode_is_directly_executable(self):
        engine = self.fresh_engine(seed=223)
        _clear_hand(engine)
        engine.players[0].deck = [_card(3400), _card(3401)]
        _put_in_hand(
            engine,
            self.repository.get(TOKEN_CARD_ID),
            origin=CardOrigin.GENERATED,
        )
        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, "choose_one:draw"))
        self.assertEqual(len(engine.players[0].deck), 1)
        self.assertIsNone(engine.state.pending_choice)

    def test_brutal_heavenblade_enhance_activates_both_without_choice(self):
        normal = self.fresh_engine(seed=227)
        _clear_hand(normal)
        normal.players[0].deck = [_card(3500), _card(3501)]
        normal_target = _add_enemy(normal, 3510, life=5)
        self.play_real(normal, 10623310)
        self.assertIsNotNone(normal.state.pending_choice)
        normal.apply(Choose(0, "choose_one:draw"))
        self.assertEqual(normal_target.health, 5)
        self.assertEqual(len(normal.players[0].deck), 1)

        enhanced = self.fresh_engine(seed=227)
        _clear_hand(enhanced)
        enhanced.players[0].deck = [_card(3520), _card(3521)]
        enhanced_target = _add_enemy(enhanced, 3530, life=5)
        self.play_real(enhanced, 10623310, "enhance_3")
        self.assertIsNone(enhanced.state.pending_choice)
        self.assertEqual(enhanced_target.health, 2)
        self.assertEqual(len(enhanced.players[0].deck), 1)

    def test_threatening_remnants_replaces_selection_with_filtered_all(self):
        normal = self.fresh_engine(seed=229)
        _clear_hand(normal)
        eligible = _add_enemy(normal, 3600, life=3)
        ineligible = _add_enemy(normal, 3601, life=4)
        self.play_real(normal, 10862310)
        option_ids = {option.entity_id for option in normal.state.pending_choice.options}
        self.assertEqual(option_ids, {eligible.entity_id})
        normal.apply(Choose(0, f"entity:{eligible.entity_id}"))
        self.assertNotIn(eligible, normal.players[1].board)
        self.assertIn(ineligible, normal.players[1].board)

        enhanced = self.fresh_engine(seed=229)
        _clear_hand(enhanced)
        low = [_add_enemy(enhanced, 3610 + i, life=3) for i in range(2)]
        high = _add_enemy(enhanced, 3620, life=4)
        self.play_real(enhanced, 10862310, "enhance_5")
        self.assertIsNone(enhanced.state.pending_choice)
        self.assertTrue(all(target not in enhanced.players[1].board for target in low))
        self.assertIn(high, enhanced.players[1].board)
        self.assertEqual(len(enhanced.players[1].banished), 2)

    def test_replacement_spell_mode_has_rl_mask_parity_and_no_board_entity(self):
        deck_a = [_card(card_id) for card_id in range(4000, 4040)]
        deck_b = [_card(card_id) for card_id in range(5000, 5040)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=2,
            class_b=2,
            seed=233,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=233)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        hand_card = HandCard(
            self.repository.get(10862310),
            entity_id=env.core.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        env.players[0].hand.append(hand_card)
        env.players[0].hand_entity_ids.append(hand_card.entity_id)
        env.players[0].max_mana = env.players[0].mana = 10
        _add_enemy(env.core, 5100, life=3)

        normal = PlayCard(0, 0, "normal")
        enhanced = PlayCard(0, 0, "enhance_5")
        legal = env.core.legal_commands()
        self.assertIn(normal, legal)
        self.assertIn(enhanced, legal)
        action = env._encode_command(enhanced)
        self.assertIsNotNone(action)
        self.assertTrue(env.action_mask()[action])

        result = env.step(action)

        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual(env.players[0].board, [])
        self.assertFalse(result.terminated)


if __name__ == "__main__":
    unittest.main()
