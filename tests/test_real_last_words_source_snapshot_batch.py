# -*- coding: utf-8 -*-
"""Direct audit for the first real Last Words source-snapshot rule."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Evolve, PlayCard
from swb.engine.effects import ConditionType, EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


CARD_ID = 10203120


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
        life=overrides.get("life", 5),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _make_engine(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 307,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=1,
        class_b=1,
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


def _put_in_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


def _add_enemy(engine: GameEngine, card_id: int, *, life: int = 5) -> Unit:
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

    def test_database_text_keywords_and_absent_modes_or_references_match_audit(self):
        expected_text = (
            "【<color=Keyword>爆能强化</color>_5】本随从进化。\n"
            "<hr>【<color=Keyword>谢幕曲</color>】若本随从为进化后，则对对手的战场上的随机1个随从造成4点伤害。"
        )
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute(
                """
                SELECT json_extract(c.raw_json, '$.name_chs'),
                       cl.class_name, cl.type_name, c.cost, c.attack, c.life,
                       st.text_chs, st.text_eng, st.text_jpn, st.text_cht
                FROM cards c
                JOIN card_localizations cl
                  ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                JOIN skill_texts st ON st.card_id = c.card_id
                WHERE c.card_id = ? ORDER BY st.position
                """,
                (CARD_ID,),
            ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0][:7],
                (
                    "雷火双神·福尼加尔&亚文哈尔",
                    "中立",
                    "随从",
                    2,
                    2,
                    2,
                    expected_text,
                ),
            )
            self.assertTrue(all(rows[0][index] for index in (7, 8, 9)))
            self.assertEqual(
                set(connection.execute(
                    "SELECT ability_keyword, raw_keyword "
                    "FROM card_abilities WHERE card_id = ?",
                    (CARD_ID,),
                )),
                {("爆能强化", "爆能强化"), ("谢幕曲", "谢幕曲")},
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                    (CARD_ID,),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM card_references WHERE card_id = ?",
                    (CARD_ID,),
                ).fetchone()[0],
                0,
            )

    def test_card_has_exact_clause_evidence_and_source_hash(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"][str(CARD_ID)]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
        self.assertEqual(
            info["clause_audit"]["source_text_sha256"],
            "699983f2f7a5c36863c705113c6315c564fbf8285a5cc00e767ae024435691d3",
        )
        self.assertEqual(
            info["clause_audit"]["test_evidence"],
            ["tests/test_real_last_words_source_snapshot_batch.py"],
        )


class RealLastWordsSourceSnapshotBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 307) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def play_real(
        self,
        engine: GameEngine,
        *,
        mode_id: str = "normal",
    ) -> Unit:
        _put_in_hand(engine, self.repository.get(CARD_ID))
        engine.apply(PlayCard(0, 0, mode_id))
        return next(
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == CARD_ID
        )

    def test_rule_shape_links_enhance_evolution_to_snapshot_condition(self):
        mode = self.rulebook.modes_for(CARD_ID)[0]
        self.assertEqual((mode.mode_id, mode.mode_type, mode.cost), ("enhance_5", "enhance", 5))
        self.assertFalse(mode.replace_base_operations)
        self.assertEqual(
            (mode.operations[0].kind, mode.operations[0].target),
            (EffectKind.EVOLVE_UNIT, TargetKind.SELF),
        )
        last_words = self.rulebook.operations_for(CARD_ID, Trigger.LAST_WORDS)[0]
        self.assertEqual(
            (last_words.kind, last_words.target, last_words.amount),
            (EffectKind.DAMAGE_UNIT, TargetKind.RANDOM_ENEMY_UNIT, 4),
        )
        self.assertEqual(last_words.conditions[0].type, ConditionType.SOURCE_EVOLVED)

    def test_normal_death_skips_but_enhance_death_uses_evolved_snapshot(self):
        normal = self.fresh_engine(seed=311)
        _clear_hand(normal)
        normal_target = _add_enemy(normal, 3000)
        normal_source = self.play_real(normal)
        self.assertFalse(normal_source.evolved)
        self.assertEqual(normal.players[0].mana, 8)
        normal_source.health = 0
        normal._stabilize()
        self.assertEqual(normal_target.health, 5)

        enhanced = self.fresh_engine(seed=311)
        _clear_hand(enhanced)
        enhanced_target = _add_enemy(enhanced, 3010)
        ep_before = enhanced.players[0].evolution_points
        enhanced_source = self.play_real(enhanced, mode_id="enhance_5")
        self.assertTrue(enhanced_source.evolved)
        self.assertEqual((enhanced_source.attack, enhanced_source.health), (4, 4))
        self.assertEqual(enhanced.players[0].mana, 5)
        self.assertEqual(enhanced.players[0].evolution_points, ep_before)
        self.assertFalse(enhanced.players[0].evolved_this_turn)

        enhanced_source.health = 0
        enhanced._stabilize()

        self.assertEqual(enhanced_target.health, 1)
        record = enhanced.state.death_queue[-1].records[0]
        self.assertTrue(record.evolved)

    def test_manual_evolution_also_enables_conditional_last_words(self):
        engine = self.fresh_engine(seed=313)
        _clear_hand(engine)
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        target = _add_enemy(engine, 3100)
        source = self.play_real(engine)

        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(engine.players[0].evolution_points, 1)
        source.health = 0
        engine._stabilize()

        self.assertEqual(target.health, 1)

    def test_enhanced_last_words_no_target_skips_and_seeded_target_is_reproducible(self):
        empty = self.fresh_engine(seed=317)
        _clear_hand(empty)
        empty_source = self.play_real(empty, mode_id="enhance_5")
        empty_source.health = 0
        empty._stabilize()
        self.assertIsNone(empty.state.pending_choice)
        self.assertEqual(empty.players[1].health, 20)

        def resolve_once():
            engine = self.fresh_engine(seed=319)
            _clear_hand(engine)
            targets = [_add_enemy(engine, 3200 + index) for index in range(2)]
            source = self.play_real(engine, mode_id="enhance_5")
            source.health = 0
            engine._stabilize()
            return tuple(target.health for target in targets)

        first = resolve_once()
        second = resolve_once()
        self.assertEqual(first, second)
        self.assertEqual(sum(first), 6)

    def test_enhance_mode_has_rl_mask_parity_and_effect_evolves(self):
        deck_a = [_card(card_id) for card_id in range(4000, 4040)]
        deck_b = [_card(card_id) for card_id in range(5000, 5040)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=331,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=331)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        _put_in_hand(env.core, self.repository.get(CARD_ID))
        env.players[0].max_mana = env.players[0].mana = 5
        command = PlayCard(0, 0, "enhance_5")
        action = env._encode_command(command)
        self.assertIsNotNone(action)
        self.assertIn(command, env.core.legal_commands())
        self.assertTrue(env.action_mask()[action])

        result = env.step(action)

        source = next(
            unit
            for unit in env.players[0].board
            if unit.definition.card_id == CARD_ID
        )
        self.assertTrue(source.evolved)
        self.assertIsNone(env.core.state.pending_choice)
        self.assertFalse(result.terminated)


if __name__ == "__main__":
    unittest.main()
