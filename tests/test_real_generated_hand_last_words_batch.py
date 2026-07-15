# -*- coding: utf-8 -*-
"""Direct audits for real Last Words cards that generate cards in hand."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Attack, PlayCard
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin, is_derived
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard


COLLECTIBLE_IDS = (10101120, 10152110, 10312120, 10751110, 10872110)
TOKEN_IDS = (90011110, 90011310, 90051110, 90051120, 90071140)
SOURCE_HASHES = {
    10101120: "eea2266e4f302e824805ee2f23466b44eca679a4f06420f4149d1298a995f8f3",
    10152110: "8fd4ca393828b21ddb0971c77269c5ec41d53068a617361690fba4abb9cd6a4e",
    10312120: "355acd8ac9d98f1c741b608c537631e142b55055994ab46f8f4a589ddda28393",
    10751110: "0ef33b20fd1dc9bcabbd7d7faf740825f5a6d149b6bbd3385dd260f5b6434efc",
    10872110: "1b112049772693bca279902d7dd1c3c0c806b0ca17016b21fa3caf141542b45e",
    90011110: "7abde1fa8f2c436a6805d99abe026718965b9776b4b5a2dffe05f57cd5b2776f",
    90011310: "8d0e71bcf0a8398d1d1d7653dfd0fb506775ae65c9bed03c4fcdeab5e78c0c51",
    90051110: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    90051120: "0769bb2b006b52b82b34de388772a3a7c6210530311b5793c2c532bc217a95f6",
    90071140: "7abde1fa8f2c436a6805d99abe026718965b9776b4b5a2dffe05f57cd5b2776f",
}
GENERATED_BY_SOURCE = {
    10101120: 10001210,
    10152110: 90051110,
    10751110: 90051120,
    10872110: 90071140,
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
    seed: int = 401,
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
    engine.players[0].hand.append(hand_card)
    engine.players[0].hand_entity_ids.append(hand_card.entity_id)
    return hand_card


def _play_real(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
    mode_id: str = "normal",
):
    definition = repository.get(card_id)
    _put_in_hand(engine, definition)
    engine.apply(PlayCard(0, len(engine.players[0].hand) - 1, mode_id))
    return next(
        entity
        for entity in engine.players[0].board
        if entity.definition.card_id == card_id
    ) if definition.card_type != "法术" else None


def _destroy(engine: GameEngine, source) -> None:
    source.health = 0
    engine._stabilize()


class DatabaseClauseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_cards_match_names_stats_keywords_and_references(self):
        expected = {
            10101120: ("观察的侦探", "中立", "随从", 3, 3, 3, True),
            10001210: ("侦探的放大镜", "中立", "护符", 2, None, None, True),
            10152110: ("干练的死神·蜜诺", "梦魇", "随从", 2, 2, 1, True),
            90051110: ("骸骨士兵", "梦魇", "随从", 0, 1, 1, False),
            10312120: ("树海的战士", "精灵", "随从", 4, 4, 4, True),
            90011310: ("森林的奥秘", "精灵", "法术", 0, None, None, False),
            90011110: ("妖精", "精灵", "随从", 1, 1, 1, False),
            10751110: ("暗夜键盘手·露露米", "梦魇", "随从", 2, 2, 1, True),
            90051120: ("蝙蝠", "梦魇", "随从", 1, 1, 1, False),
            10872110: ("人造的馈赠·蕾拉", "超越者", "随从", 4, 6, 4, True),
            90071140: ("古老的创造物", "超越者", "随从", 1, 3, 1, False),
        }
        expected_abilities = {
            10101120: {("谢幕曲", "谢幕曲")},
            10001210: {("守护", "守护"), ("策动", "启动")},
            10152110: {
                ("必杀", "毁灭"), ("爆能强化", "爆能强化"),
                ("突进", "突进"), ("谢幕曲", "谢幕曲"),
            },
            10312120: {("谢幕曲", "谢幕曲")},
            10751110: {("突进", "突进"), ("谢幕曲", "谢幕曲")},
            10872110: {("突进", "突进"), ("谢幕曲", "谢幕曲")},
            90011110: {("突进", "突进")},
            90051120: {("吸血", "虹吸")},
            90071140: {("突进", "突进")},
        }
        expected_references = {
            10101120: [(0, 10001210, "侦探的放大镜")],
            10152110: [(0, 90051110, "骸骨士兵")],
            10312120: [(0, 90011310, "森林的奥秘"), (1, 90011110, "妖精")],
            10751110: [(0, 90051120, "蝙蝠")],
            10872110: [(0, 90071140, "古老的创造物")],
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT json_extract(c.raw_json, '$.name_chs'),
                               cl.class_name, cl.type_name, c.cost, c.attack, c.life,
                               cs.is_collectible
                        FROM cards c
                        JOIN card_sets cs ON cs.id = c.card_set_id
                        JOIN card_localizations cl
                          ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                        WHERE c.card_id = ?
                        """,
                        (card_id,),
                    ).fetchone()
                    self.assertEqual(row, values)
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT ability_keyword, raw_keyword FROM card_abilities "
                            "WHERE card_id = ?",
                            (card_id,),
                        )),
                        expected_abilities.get(card_id, set()),
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT position, referenced_card_id, referenced_name "
                            "FROM card_references WHERE card_id = ? ORDER BY position",
                            (card_id,),
                        ).fetchall(),
                        expected_references.get(card_id, []),
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
                self.assertEqual(
                    info["clause_audit"]["status"],
                    "mapped_exact" if card_id in COLLECTIBLE_IDS else "token_separate_audit",
                )
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_generated_hand_last_words_batch.py"],
                )
        self.assertTrue(all(
            report["classifications"][str(card_id)]["coverage"] == "covered_exact"
            for card_id in COLLECTIBLE_IDS
        ))

        token_report = _build_token_audit(
            "data/cards.sqlite3", "data/rules", "data/audits/token_overrides.json"
        )
        tokens = {
            card["card_id"]: card
            for card in token_report["cards"]
            if card["card_id"] in TOKEN_IDS
        }
        producers = {
            90011110: 10312120,
            90011310: 10312120,
            90051110: 10152110,
            90051120: 10751110,
            90071140: 10872110,
        }
        for card_id, source_card_id in producers.items():
            with self.subTest(token_card_id=card_id):
                self.assertEqual(tokens[card_id]["category"], "entry_behavior_complete")
                self.assertEqual(tokens[card_id]["explicit_coverage"], "exact")
                self.assertIn(
                    {
                        "source_card_id": source_card_id,
                        "entry_kind": "add_card",
                        "rule_file": "real_generated_hand_last_words_batch.json",
                        "rule_group": "rules",
                    },
                    tokens[card_id]["authored_producers"],
                )


class RealGeneratedHandLastWordsBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 401) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_non_intrinsic_enhance_keywords(self):
        expected = {
            10101120: (10001210,),
            10152110: (90051110,),
            10312120: (90011310, 90011110),
            10751110: (90051120,),
            10872110: (90071140,),
        }
        for card_id, generated_ids in expected.items():
            with self.subTest(card_id=card_id):
                operations = self.rulebook.operations_for(card_id, Trigger.LAST_WORDS)
                self.assertEqual(
                    tuple((op.kind, op.target, op.card_id) for op in operations),
                    tuple(
                        (EffectKind.ADD_CARD, TargetKind.OWN_LEADER, generated_id)
                        for generated_id in generated_ids
                    ),
                )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10152110),
            frozenset({"突进", "必杀"}),
        )
        mode = self.rulebook.modes_for(10152110)[0]
        self.assertEqual((mode.mode_id, mode.mode_type, mode.cost), ("enhance_4", "enhance", 4))
        self.assertEqual(
            [(op.kind, op.keyword) for op in mode.operations],
            [(EffectKind.ADD_KEYWORD, "突进"), (EffectKind.ADD_KEYWORD, "必杀")],
        )

    def test_each_last_words_adds_the_printed_card_with_correct_origin(self):
        for source_id, generated_id in GENERATED_BY_SOURCE.items():
            with self.subTest(source_id=source_id):
                engine = self.fresh_engine(seed=409 + source_id)
                _clear_hand(engine)
                source = _play_real(engine, self.repository, source_id)
                _destroy(engine, source)

                generated = [
                    card for card in engine.players[0].hand
                    if card.card_id == generated_id
                ]
                self.assertEqual(len(generated), 1)
                expected_origin = (
                    CardOrigin.GENERATED if generated_id == 10001210 else CardOrigin.TOKEN
                )
                self.assertIs(generated[0].origin, expected_origin)
                self.assertTrue(is_derived(generated[0].origin))

    def test_tree_warrior_preserves_printed_order_and_overflows_second_card(self):
        engine = self.fresh_engine(seed=419)
        _clear_hand(engine)
        source = _play_real(engine, self.repository, 10312120)
        while len(engine.players[0].hand) < engine.config.max_hand - 1:
            _put_in_hand(engine, _card(7000 + len(engine.players[0].hand)))

        _destroy(engine, source)

        self.assertEqual(len(engine.players[0].hand), engine.config.max_hand)
        self.assertEqual(engine.players[0].hand[-1].card_id, 90011310)
        self.assertIs(engine.players[0].hand[-1].origin, CardOrigin.TOKEN)
        overflow = [
            card for card in engine.players[0].graveyard
            if card.definition.card_id == 90011110
        ]
        self.assertEqual(len(overflow), 1)
        self.assertIs(overflow[0].origin, CardOrigin.TOKEN)
        self.assertTrue(overflow[0].derived)

    def test_mino_gains_rush_and_bane_only_in_enhance_mode(self):
        normal = self.fresh_engine(seed=421)
        _clear_hand(normal)
        normal_source = _play_real(normal, self.repository, 10152110)
        self.assertEqual(normal.players[0].mana, 8)
        self.assertFalse(normal_source.has_keyword("突进"))
        self.assertFalse(normal_source.has_keyword("必杀"))
        self.assertFalse(normal_source.can_attack)

        enhanced = self.fresh_engine(seed=421)
        _clear_hand(enhanced)
        enhanced_source = _play_real(
            enhanced, self.repository, 10152110, "enhance_4"
        )
        self.assertEqual(enhanced.players[0].mana, 6)
        self.assertTrue(enhanced_source.has_keyword("突进"))
        self.assertTrue(enhanced_source.has_keyword("必杀"))
        self.assertTrue(enhanced_source.can_attack)
        self.assertFalse(enhanced_source.can_attack_leader)

        _destroy(enhanced, enhanced_source)
        skeleton = next(card for card in enhanced.players[0].hand if card.card_id == 90051110)
        self.assertIs(skeleton.origin, CardOrigin.TOKEN)
        skeleton_index = enhanced.players[0].hand.index(skeleton)
        enhanced.apply(PlayCard(0, skeleton_index))
        played = next(
            unit for unit in enhanced.players[0].board
            if unit.definition.card_id == 90051110
        )
        self.assertEqual((played.attack, played.health), (1, 1))
        self.assertEqual(played.effective_keywords, frozenset())
        self.assertEqual(enhanced.players[0].mana, 6)

    def test_generated_spell_and_keyword_followers_execute_exact_behavior(self):
        spell_case = self.fresh_engine(seed=431)
        _clear_hand(spell_case)
        source = _play_real(spell_case, self.repository, 10312120)
        _destroy(spell_case, source)
        spell_case.players[0].health = 19
        mystery = next(card for card in spell_case.players[0].hand if card.card_id == 90011310)
        spell_case.apply(PlayCard(0, spell_case.players[0].hand.index(mystery)))
        self.assertEqual(spell_case.players[0].health, 20)
        self.assertEqual(spell_case.players[0].graveyard[-1].definition.card_id, 90011310)

        fairy = next(card for card in spell_case.players[0].hand if card.card_id == 90011110)
        spell_case.apply(PlayCard(0, spell_case.players[0].hand.index(fairy)))
        fairy_unit = next(
            unit for unit in spell_case.players[0].board
            if unit.definition.card_id == 90011110
        )
        self.assertTrue(fairy_unit.has_keyword("突进"))
        self.assertTrue(fairy_unit.can_attack)
        self.assertFalse(fairy_unit.can_attack_leader)

        bat_case = self.fresh_engine(seed=433)
        _clear_hand(bat_case)
        source = _play_real(bat_case, self.repository, 10751110)
        _destroy(bat_case, source)
        bat = next(card for card in bat_case.players[0].hand if card.card_id == 90051120)
        bat_case.apply(PlayCard(0, bat_case.players[0].hand.index(bat)))
        bat_unit = next(
            unit for unit in bat_case.players[0].board
            if unit.definition.card_id == 90051120
        )
        self.assertTrue(bat_unit.has_keyword("吸血"))
        bat_unit.can_attack = True
        bat_case.players[0].health = 18
        bat_case.apply(Attack(0, bat_unit.entity_id, None))
        self.assertEqual(bat_case.players[0].health, 19)
        self.assertEqual(bat_case.players[1].health, 19)

        artifact_case = self.fresh_engine(seed=439)
        _clear_hand(artifact_case)
        source = _play_real(artifact_case, self.repository, 10872110)
        _destroy(artifact_case, source)
        artifact = next(
            card for card in artifact_case.players[0].hand if card.card_id == 90071140
        )
        artifact_case.apply(PlayCard(0, artifact_case.players[0].hand.index(artifact)))
        artifact_unit = next(
            unit for unit in artifact_case.players[0].board
            if unit.definition.card_id == 90071140
        )
        self.assertEqual((artifact_unit.attack, artifact_unit.health), (3, 1))
        self.assertTrue(artifact_unit.has_keyword("突进"))
        self.assertTrue(artifact_unit.can_attack)
        self.assertFalse(artifact_unit.can_attack_leader)

    def test_enhance_mode_matches_legal_commands_and_rl_mask(self):
        deck_a = [_card(card_id) for card_id in range(8000, 8040)]
        deck_b = [_card(card_id) for card_id in range(8100, 8140)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=2,
            class_b=2,
            seed=443,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=443)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        _put_in_hand(env.core, self.repository.get(10152110))
        env.players[0].max_mana = env.players[0].mana = 10

        normal = PlayCard(0, 0)
        enhanced = PlayCard(0, 0, "enhance_4")
        self.assertIn(normal, env.core.legal_commands())
        self.assertIn(enhanced, env.core.legal_commands())
        normal_action = env._encode_command(normal)
        enhanced_action = env._encode_command(enhanced)
        self.assertIsNotNone(normal_action)
        self.assertIsNotNone(enhanced_action)
        mask = env.action_mask()
        self.assertTrue(mask[normal_action])
        self.assertTrue(mask[enhanced_action])

    def test_same_seed_and_sequence_have_identical_fingerprint(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh_engine(seed=449)
            _clear_hand(engine)
            source = _play_real(engine, self.repository, 10312120)
            _destroy(engine, source)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])


if __name__ == "__main__":
    unittest.main()
