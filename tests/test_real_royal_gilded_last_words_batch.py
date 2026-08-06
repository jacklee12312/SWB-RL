# -*- coding: utf-8 -*-
"""Direct audits for the Royal gilded-token Fanfare/Last Words slice."""

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
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


COLLECTIBLE_IDS = (10321110, 10322110)
TOKEN_IDS = (90021310, 90021320, 90021330, 90021340)
SOURCE_HASHES = {
    10321110: "938a2826937cd3f9430d16857a6337611cd699891dc3395edfd668e6442efc76",
    10322110: "9fefb9d5d824d5ae82ce3e9a50c8487302930afe6810b8cc106727f2ce325a24",
    90021310: "a6dda36f4ef0cf417b153a6392a0ee3a9610cc31b866730262b989cfa0adda5d",
    90021320: "4174c9f39a8f729a0f798642a58df774b37299ad5d0a9b5a1c156a8d3ba562a4",
    90021330: "6c029362b5d9005bef026b5c53f9e0399e07b5bdec7457ccb858b2c36d765ee9",
    90021340: "e0781287b6536cd321dba5f9572ed78d7e7067dd653209df36a209142130aa09",
}


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 0),
        class_name=overrides.get("class_name", "中立"),
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
    seed: int = 331,
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


def _add_unit(
    engine: GameEngine,
    player_index: int,
    card_id: int,
    *,
    attack: int = 1,
    life: int = 3,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, attack=attack, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[player_index].board.append(unit)
    return unit


class DatabaseClauseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_texts_keywords_references_and_collectibility_match_audit(self):
        expected = {
            10321110: (
                "篡夺的肯定者", "皇家护卫", "随从", 2, 2, 1, True,
                "【<color=Keyword>入场曲</color>】将1张『<color=Keyword>黄金之靴</color>』加入手牌。\n"
                "<hr>【<color=Keyword>谢幕曲</color>】将1张『<color=Keyword>黄金之杯</color>』加入手牌。",
                {("入场曲", "入场曲"), ("谢幕曲", "谢幕曲")},
                {(0, 90021330, "黄金之靴"), (1, 90021320, "黄金之杯")},
            ),
            10322110: (
                "篡夺的祈祷者", "皇家护卫", "随从", 2, 1, 2, True,
                "【<color=Keyword>入场曲</color>】将1张『<color=Keyword>黄金项链</color>』加入手牌。\n"
                "<hr>【<color=Keyword>谢幕曲</color>】将1张『<color=Keyword>黄金短剑</color>』加入手牌。",
                {("入场曲", "入场曲"), ("谢幕曲", "谢幕曲")},
                {(0, 90021340, "黄金项链"), (1, 90021310, "黄金短剑")},
            ),
            90021310: (
                "黄金短剑", "皇家护卫", "法术", 1, None, None, False,
                "选择对手的战场上的1个随从或对手的主战者，对其造成1点伤害。",
                set(), set(),
            ),
            90021320: (
                "黄金之杯", "皇家护卫", "法术", 1, None, None, False,
                "回复自己的主战者2点生命值。", set(), set(),
            ),
            90021330: (
                "黄金之靴", "皇家护卫", "法术", 1, None, None, False,
                "选择自己的战场上的1个随从，使其+1/+0且获得【<color=Keyword>突进</color>】。",
                {("突进", "突进")}, set(),
            ),
            90021340: (
                "黄金项链", "皇家护卫", "法术", 1, None, None, False,
                "选择自己的战场上的1个随从，使其+0/+1且获得【<color=Keyword>守护</color>】。",
                {("守护", "守护")}, set(),
            ),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    rows = connection.execute(
                        """
                        SELECT json_extract(c.raw_json, '$.name_chs'),
                               cl.class_name, cl.type_name, c.cost, c.attack, c.life,
                               cs.is_collectible,
                               st.text_chs, st.text_eng, st.text_jpn, st.text_cht
                        FROM cards c
                        JOIN card_sets cs ON cs.id = c.card_set_id
                        JOIN card_localizations cl
                          ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                        JOIN skill_texts st ON st.card_id = c.card_id
                        WHERE c.card_id = ? ORDER BY st.position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0][:8], values[:8])
                    self.assertTrue(all(rows[0][index] for index in (8, 9, 10)))
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT ability_keyword, raw_keyword "
                            "FROM card_abilities WHERE card_id = ?",
                            (card_id,),
                        )),
                        values[8],
                    )
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT position, referenced_card_id, referenced_name "
                            "FROM card_references WHERE card_id = ?",
                            (card_id,),
                        )),
                        values[9],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_collectibles_and_tokens_have_exact_source_hash_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in (*COLLECTIBLE_IDS, *TOKEN_IDS):
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                expected_status = (
                    "mapped_exact"
                    if card_id in COLLECTIBLE_IDS
                    else "token_separate_audit"
                )
                self.assertEqual(info["clause_audit"]["status"], expected_status)
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_royal_gilded_last_words_batch.py"],
                )
        self.assertTrue(all(
            report["classifications"][str(card_id)]["coverage"] == "covered_exact"
            for card_id in COLLECTIBLE_IDS
        ))

        token_report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        tokens = {
            card["card_id"]: card
            for card in token_report["cards"]
            if card["card_id"] in TOKEN_IDS
        }
        expected_producers = {
            90021310: [
                {
                    "source_card_id": 10322110,
                    "entry_kind": "add_card",
                    "rule_file": "real_royal_gilded_last_words_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10322210,
                    "entry_kind": "add_card",
                    "rule_file": "real_royal_rune_bishop_mixed_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10323310,
                    "entry_kind": "add_card",
                    "rule_file": "real_spell_modes_and_earth_listener_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10324120,
                    "entry_kind": "add_card",
                    "rule_file": "real_generated_distributed_damage_crest_batch.json",
                    "rule_group": "rules",
                },
            ],
            90021320: [
                {
                    "source_card_id": 10321110,
                    "entry_kind": "add_card",
                    "rule_file": "real_royal_gilded_last_words_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10322210,
                    "entry_kind": "add_card",
                    "rule_file": "real_royal_rune_bishop_mixed_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10323110,
                    "entry_kind": "add_card",
                    "rule_file": "real_existing_primitives_fourth_completion_batch.json",
                    "rule_group": "rules",
                },
            ],
            90021330: [
                {
                    "source_card_id": 10321110,
                    "entry_kind": "add_card",
                    "rule_file": "real_royal_gilded_last_words_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10322210,
                    "entry_kind": "add_card",
                    "rule_file": "real_royal_rune_bishop_mixed_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10323110,
                    "entry_kind": "add_card",
                    "rule_file": "real_existing_primitives_fourth_completion_batch.json",
                    "rule_group": "rules",
                },
            ],
            90021340: [
                {
                    "source_card_id": 10322110,
                    "entry_kind": "add_card",
                    "rule_file": "real_royal_gilded_last_words_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10322210,
                    "entry_kind": "add_card",
                    "rule_file": "real_royal_rune_bishop_mixed_batch.json",
                    "rule_group": "rules",
                },
                {
                    "source_card_id": 10324120,
                    "entry_kind": "add_card",
                    "rule_file": "real_generated_distributed_damage_crest_batch.json",
                    "rule_group": "rules",
                },
            ],
        }
        for card_id, producers in expected_producers.items():
            with self.subTest(token_card_id=card_id):
                self.assertEqual(tokens[card_id]["category"], "entry_behavior_complete")
                self.assertEqual(tokens[card_id]["explicit_coverage"], "exact")
                self.assertEqual(
                    tokens[card_id]["authored_producers"],
                    producers,
                )


class RealRoyalGildedLastWordsBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 331) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def play_real(self, engine: GameEngine, card_id: int) -> None:
        _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0))

    def test_rule_shapes_bind_each_gilded_buff_to_one_required_target(self):
        expected_generated = {
            (10321110, Trigger.FANFARE): 90021330,
            (10321110, Trigger.LAST_WORDS): 90021320,
            (10322110, Trigger.FANFARE): 90021340,
            (10322110, Trigger.LAST_WORDS): 90021310,
        }
        for key, generated_id in expected_generated.items():
            with self.subTest(card_id=key[0], trigger=key[1]):
                operation = self.rulebook.operations_for(*key)[0]
                self.assertEqual(
                    (operation.kind, operation.target, operation.card_id),
                    (EffectKind.ADD_CARD, TargetKind.OWN_LEADER, generated_id),
                )

        for card_id, amounts, keyword in (
            (90021330, (1, 0), "突进"),
            (90021340, (0, 1), "守护"),
        ):
            with self.subTest(card_id=card_id):
                buff, grant = self.rulebook.operations_for(card_id, Trigger.PLAY)
                self.assertEqual(
                    (buff.kind, buff.target, buff.amount, buff.secondary_amount),
                    (EffectKind.BUFF_UNIT, TargetKind.OWN_UNIT, *amounts),
                )
                self.assertTrue(buff.requires_target)
                self.assertEqual(buff.target_key, "selected_follower")
                self.assertEqual(
                    (grant.kind, grant.target, grant.keyword, grant.target_key),
                    (
                        EffectKind.ADD_KEYWORD,
                        TargetKind.PREVIOUS_TARGET,
                        keyword,
                        "selected_follower",
                    ),
                )

    def test_both_followers_generate_fanfare_and_last_words_tokens(self):
        expected = {
            10321110: (90021330, 90021320),
            10322110: (90021340, 90021310),
        }
        for card_id, (fanfare_id, last_words_id) in expected.items():
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine(seed=337 + card_id)
                _clear_hand(engine)
                self.play_real(engine, card_id)
                source = next(
                    unit
                    for unit in engine.players[0].board
                    if unit.definition.card_id == card_id
                )
                generated = [
                    card
                    for card in engine.players[0].hand
                    if card.card_id == fanfare_id
                ]
                self.assertEqual(len(generated), 1)
                self.assertIs(generated[0].origin, CardOrigin.TOKEN)

                source.health = 0
                engine._stabilize()

                last_words = [
                    card
                    for card in engine.players[0].hand
                    if card.card_id == last_words_id
                ]
                self.assertEqual(len(last_words), 1)
                self.assertIs(last_words[0].origin, CardOrigin.TOKEN)

    def test_last_words_generated_token_overflows_a_full_hand_to_graveyard(self):
        engine = self.fresh_engine(seed=347)
        _clear_hand(engine)
        self.play_real(engine, 10321110)
        source = next(
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == 10321110
        )
        while len(engine.players[0].hand) < engine.config.max_hand:
            _put_in_hand(engine, _card(7000 + len(engine.players[0].hand)))

        source.health = 0
        engine._stabilize()

        overflow = [
            card
            for card in engine.players[0].graveyard
            if card.definition.card_id == 90021320
        ]
        self.assertEqual(len(overflow), 1)
        self.assertIs(overflow[0].origin, CardOrigin.TOKEN)
        self.assertTrue(overflow[0].derived)
        self.assertFalse(any(
            card.card_id == 90021320 for card in engine.players[0].hand
        ))

    def test_gilded_boots_and_necklace_apply_both_effects_to_selected_follower(self):
        for card_id, expected_stats, keyword in (
            (90021330, (3, 3), "突进"),
            (90021340, (2, 4), "守护"),
        ):
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine(seed=349 + card_id)
                _clear_hand(engine)
                target = _add_unit(engine, 0, 7100, attack=2, life=3)
                self.play_real(engine, card_id)
                self.assertEqual(
                    [option.option_id for option in engine.state.pending_choice.options],
                    [f"entity:{target.entity_id}"],
                )
                engine.apply(Choose(0, f"entity:{target.entity_id}"))

                self.assertEqual((target.attack, target.health), expected_stats)
                self.assertTrue(target.has_keyword(keyword))
                if keyword == "突进":
                    self.assertTrue(target.can_attack)
                    self.assertFalse(target.can_attack_leader)
                else:
                    self.assertTrue(target.has_guard)
                self.assertEqual(
                    engine.players[0].graveyard[-1].definition.card_id,
                    card_id,
                )

    def test_gilded_boots_no_target_is_illegal_without_mutation_and_matches_rl_mask(self):
        engine = self.fresh_engine(seed=353)
        _clear_hand(engine)
        _put_in_hand(engine, self.repository.get(90021330), origin=CardOrigin.TOKEN)
        command = PlayCard(0, 0)
        before = engine.deterministic_fingerprint()
        self.assertNotIn(command, engine.legal_commands())
        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.deterministic_fingerprint(), before)

        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(8000, 8040)],
            [_card(card_id) for card_id in range(8100, 8140)],
            class_a=2,
            class_b=2,
            seed=353,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=353)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        _put_in_hand(env.core, self.repository.get(90021330), origin=CardOrigin.TOKEN)
        env.players[0].max_mana = env.players[0].mana = 10
        action = env._encode_command(command)
        self.assertIsNotNone(action)
        self.assertFalse(env.action_mask()[action])

        target = _add_unit(env.core, 0, 8200)
        self.assertIn(command, env.core.legal_commands())
        self.assertTrue(env.action_mask()[action])
        result = env.step(action)
        self.assertFalse(result.terminated)
        self.assertEqual(
            [option.entity_id for option in env.core.state.pending_choice.options],
            [target.entity_id],
        )

    def test_gilded_boots_safely_skips_when_pending_target_leaves_play(self):
        engine = self.fresh_engine(seed=359)
        _clear_hand(engine)
        target = _add_unit(engine, 0, 8300, attack=2, life=3)
        self.play_real(engine, 90021330)
        choice = Choose(0, f"entity:{target.entity_id}")

        engine.players[0].board.remove(target)
        engine._send_to_graveyard(
            0,
            target.definition,
            "test_pending_target_left_play",
            source_entity_id=target.entity_id,
        )
        engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual((target.attack, target.health), (2, 3))
        self.assertFalse(target.has_keyword("突进"))
        self.assertEqual(
            engine.players[0].graveyard[-1].definition.card_id,
            90021330,
        )

    def test_gilded_blade_can_choose_enemy_follower_or_leader(self):
        unit_case = self.fresh_engine(seed=367)
        _clear_hand(unit_case)
        target = _add_unit(unit_case, 1, 8400, life=3)
        self.play_real(unit_case, 90021310)
        self.assertEqual(
            [option.option_id for option in unit_case.state.pending_choice.options],
            [f"entity:{target.entity_id}", "leader:1"],
        )
        unit_case.apply(Choose(0, f"entity:{target.entity_id}"))
        self.assertEqual((target.health, unit_case.players[1].health), (2, 20))

        leader_case = self.fresh_engine(seed=367)
        _clear_hand(leader_case)
        self.play_real(leader_case, 90021310)
        self.assertEqual(
            [option.option_id for option in leader_case.state.pending_choice.options],
            ["leader:1"],
        )
        leader_case.apply(Choose(0, "leader:1"))
        self.assertEqual(leader_case.players[1].health, 19)

    def test_gilded_goblet_heals_leader_with_health_cap(self):
        engine = self.fresh_engine(seed=373)
        _clear_hand(engine)
        engine.players[0].health = 19
        self.play_real(engine, 90021320)

        self.assertEqual(engine.players[0].health, 20)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(
            engine.players[0].graveyard[-1].definition.card_id,
            90021320,
        )


if __name__ == "__main__":
    unittest.main()
