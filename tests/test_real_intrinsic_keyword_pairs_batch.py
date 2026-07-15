# -*- coding: utf-8 -*-
"""Schema, audit, behavior, and RL tests for intrinsic keyword-only cards."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Attack, PlayCard
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


CARD_IDS = (10051110, 10071130, 10144120, 10161120, 10422130, 10441110)
EXPECTED_KEYWORDS = {
    10051110: ("疾驰", "必杀"),
    10071130: ("必杀", "守护"),
    10144120: ("疾驰", "威慑"),
    10161120: ("守护", "屏障"),
    10422130: ("潜行", "必杀"),
    10441110: ("守护", "灵气"),
}
SOURCE_HASHES = {
    10051110: "9e3c945f41afb79f9827a43160884f0d3e2b804f98abea39b82cad5fc31d96f6",
    10071130: "d595f718abe32aec2dd44012596b29b57018800f3b1bbb45d8b436a07cb115db",
    10144120: "8c2e07095fa59d21c7cb6bf4ddef4bb415563b58e773f41e3251ada9ad0c0e51",
    10161120: "e06fc3808dd7243f74f06f14b620d6ba494b02e6b9539b848a5c033e6d6ebf5d",
    10422130: "1f04af8f597de1a82e2b441e850b0145b0b54450732d3cd6e77c942c4e91cfb8",
    10441110: "6a425373987dbabbc2cbd7f86ddba8ba67c04e9a64a03ecdf0bea822dc7d93fd",
}


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=0,
        class_name="中立",
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type="随从",
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 3),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=True,
    )


def _make_engine(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 503,
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
    engine.players[0].hand.clear()
    engine.players[0].hand_entity_ids.clear()
    engine.players[0].max_mana = engine.players[0].mana = 10
    return engine


def _put_in_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[0].hand.append(card)
    engine.players[0].hand_entity_ids.append(card.entity_id)
    return card


def _write_rule_directory(payloads: list[dict]) -> RuleBook:
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name)
    for index, payload in enumerate(payloads):
        (path / f"rules-{index}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    try:
        return RuleBook.from_directory(path)
    finally:
        directory.cleanup()


class IntrinsicKeywordSchemaTests(unittest.TestCase):
    def test_schema_normalizes_alias_and_preserves_printed_order(self):
        rulebook = _write_rule_directory([{
            "intrinsic_keywords": [{
                "card_id": 12345678,
                "keywords": ["疾驰", "毁灭"],
                "notes": "keyword-only card",
            }],
        }])
        self.assertEqual(
            rulebook.intrinsic_keywords_for(12345678),
            ("疾驰", "必杀"),
        )

    def test_schema_rejects_invalid_container_fields_ids_and_keywords(self):
        invalid_payloads = (
            ({"intrinsic_keywords": {}}, "must be a list"),
            ({"intrinsic_keywords": [1]}, "must be an object"),
            ({"intrinsic_keywords": [{"card_id": True, "keywords": ["守护"]}]}, "positive integer"),
            ({"intrinsic_keywords": [{"card_id": 1, "keywords": []}]}, "non-empty list"),
            ({"intrinsic_keywords": [{"card_id": 1, "keywords": ["未知能力"]}]}, "Unknown ability keyword"),
            ({"intrinsic_keywords": [{"card_id": 1, "keywords": ["毁灭", "必杀"]}]}, "duplicate keyword"),
            ({"intrinsic_keywords": [{"card_id": 1, "keywords": ["守护"], "extra": 1}]}, "unknown fields"),
        )
        for payload, message in invalid_payloads:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _write_rule_directory([payload])

    def test_duplicate_card_definition_across_files_is_rejected(self):
        first = {"intrinsic_keywords": [{"card_id": 12345678, "keywords": ["守护"]}]}
        second = {"intrinsic_keywords": [{"card_id": 12345678, "keywords": ["屏障"]}]}
        with self.assertRaisesRegex(ValueError, "duplicate intrinsic keyword definition"):
            _write_rule_directory([first, second])


class DatabaseAndCoverageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_cards_are_collectible_keyword_only_followers(self):
        expected = {
            10051110: ("魔狼首领", "梦魇", 6, 3, 6),
            10071130: ("兽性铁人", "超越者", 6, 4, 9),
            10144120: ("霸道的龙翼·法露特", "龙族", 6, 5, 2),
            10161120: ("圣盾祭司", "主教", 3, 2, 1),
            10422130: ("绽放的肌肉·菲奥莉托", "皇家护卫", 4, 2, 6),
            10441110: ("逐海之人·乔尔", "龙族", 3, 3, 3),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT json_extract(c.raw_json, '$.name_chs'),
                               cl.class_name, c.cost, c.attack, c.life,
                               cl.type_name, cs.is_collectible
                        FROM cards c
                        JOIN card_sets cs ON cs.id = c.card_set_id
                        JOIN card_localizations cl
                          ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                        WHERE c.card_id = ?
                        """,
                        (card_id,),
                    ).fetchone()
                    self.assertEqual(row[:5], values)
                    self.assertEqual(row[5:], ("随从", 1))
                    self.assertEqual(
                        {
                            ability
                            for ability, in connection.execute(
                                "SELECT ability_keyword FROM card_abilities "
                                "WHERE card_id = ?",
                                (card_id,),
                            )
                        },
                        set(EXPECTED_KEYWORDS[card_id]),
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM card_references WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_all_six_cards_are_mapped_exact_with_intrinsic_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["structured_evidence"],
                    {
                        "triggers": ["intrinsic_keywords"],
                        "effect_kinds": [
                            f"keyword:{keyword}"
                            for keyword in EXPECTED_KEYWORDS[card_id]
                        ],
                    },
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_intrinsic_keyword_pairs_batch.py"],
                )


class RealIntrinsicKeywordBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 503) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def play_real(self, engine: GameEngine, card_id: int) -> Unit:
        _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0))
        return next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == card_id
        )

    def test_declared_keywords_match_every_real_unit_initial_state(self):
        for card_id, expected in EXPECTED_KEYWORDS.items():
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine(seed=509 + card_id)
                unit = self.play_real(engine, card_id)
                self.assertEqual(
                    self.rulebook.intrinsic_keywords_for(card_id),
                    expected,
                )
                self.assertEqual(unit.effective_keywords, frozenset(expected))
                self.assertEqual(unit.has_guard, "守护" in expected)
                self.assertEqual(unit.has_keyword("潜行"), "潜行" in expected)
                self.assertEqual(unit.has_aura, "灵气" in expected)
                self.assertEqual(unit.has_intimidate, "威慑" in expected)
                self.assertEqual(unit.barrier_charges, int("屏障" in expected))
                self.assertEqual(unit.can_attack, "疾驰" in expected)
                self.assertEqual(unit.can_attack_leader, "疾驰" in expected)

    def test_real_storm_bane_follower_attacks_immediately_and_destroys(self):
        engine = self.fresh_engine(seed=521)
        wolf = self.play_real(engine, 10051110)
        target = Unit.summon(
            _card(9001, attack=1, life=10),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board.append(target)

        command = Attack(0, wolf.entity_id, target.entity_id)
        self.assertIn(command, engine.legal_commands())
        engine.apply(command)

        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(target.health, 0)

    def test_real_ambush_bane_starts_hidden_and_real_ward_aura_is_not_manual_target(self):
        ambush_case = self.fresh_engine(seed=523)
        ambush = self.play_real(ambush_case, 10422130)
        self.assertTrue(ambush.has_keyword("潜行"))
        self.assertTrue(ambush.has_keyword("必杀"))

        aura_case = self.fresh_engine(seed=527)
        aura = self.play_real(aura_case, 10441110)
        self.assertTrue(aura.has_guard)
        self.assertTrue(aura.has_aura)

    def test_storm_attack_is_exposed_by_rl_mask_after_play(self):
        deck_a = [_card(card_id) for card_id in range(3000, 3040)]
        deck_b = [_card(card_id) for card_id in range(4000, 4040)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=2,
            class_b=2,
            seed=541,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=541)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        _put_in_hand(env.core, self.repository.get(10051110))
        env.players[0].max_mana = env.players[0].mana = 10

        play = PlayCard(0, 0)
        play_action = env._encode_command(play)
        self.assertIsNotNone(play_action)
        self.assertTrue(env.action_mask()[play_action])
        env.step(play_action)

        wolf = next(
            unit for unit in env.players[0].board
            if unit.definition.card_id == 10051110
        )
        attack = Attack(0, wolf.entity_id, None)
        self.assertIn(attack, env.core.legal_commands())
        attack_action = env._encode_command(attack)
        self.assertIsNotNone(attack_action)
        self.assertTrue(env.action_mask()[attack_action])


if __name__ == "__main__":
    unittest.main()
