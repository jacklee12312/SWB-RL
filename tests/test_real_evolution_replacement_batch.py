# -*- coding: utf-8 -*-
"""Exact audits for evolution effects replaced by super-evolution text."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, Evolve, PlayCard, SuperEvolve
from swb.engine.conditions import EvalContext, evaluate_condition
from swb.engine.effects import (
    Condition,
    ConditionType,
    EffectKind,
    SourceStateSnapshot,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


CARD_IDS = (10002110, 10042110, 10062110, 10072120, 10541110)
SOURCE_HASHES = {
    10002110: "874e2555bdd29dbe60ebe72f8043545d8c2dc3d866663866913644c1b8e6447d",
    10042110: "b074dc257f9b2ec3a09ac012a290fc8dde33e0b0ac7c9de7a79dc7b63e7ba52b",
    10062110: "47b854adc98e4b2ee3545dfb38400a2939cf7afb5df2ab90e30f67d8d70f8ef6",
    10072120: "0195b4829cc52776d32eedf288f0abc9625f5c3605e66a2bf24830586bc5ed38",
    10541110: "977a0b696b91f0536820fd02d984b7281d56ee0a3ba4a80c77405f9379d513d8",
}
STRUCTURED_EVIDENCE = {
    10002110: {
        "triggers": ["evolve", "super_evolve"],
        "effect_kinds": ["heal_leader", "heal_leader"],
    },
    10042110: {
        "triggers": ["evolve", "super_evolve"],
        "effect_kinds": ["damage_unit", "damage_unit"],
    },
    10062110: {
        "triggers": ["evolve", "super_evolve"],
        "effect_kinds": ["banish", "banish"],
    },
    10072120: {
        "triggers": ["evolve", "super_evolve", "intrinsic_keywords"],
        "effect_kinds": ["summon", "summon", "summon", "keyword:守护"],
    },
    10541110: {
        "triggers": ["evolve", "super_evolve"],
        "effect_kinds": ["damage_unit", "damage_unit"],
    },
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
        life=overrides.get("life", 6),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _make_engine(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 1201,
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
    for player in engine.players:
        player.hand.clear()
        player.hand_entity_ids.clear()
        player.max_mana = player.mana = 10
    return engine


def _put_in_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[0].hand.insert(0, card)
    engine.players[0].hand_entity_ids.insert(0, card.entity_id)
    return card


def _play_real(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
) -> Unit:
    card = _put_in_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, engine.players[0].hand.index(card)))
    return next(
        unit for unit in engine.players[0].board
        if unit.definition.card_id == card_id
    )


def _add_enemy(
    engine: GameEngine,
    card_id: int,
    *,
    attack: int = 0,
    life: int = 8,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, attack=attack, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[1].board.append(unit)
    return unit


def _choose(engine: GameEngine, entity_id: int) -> None:
    engine.apply(Choose(0, f"entity:{entity_id}"))


def _enable_evolution(engine: GameEngine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = 2


def _enable_super_evolution(engine: GameEngine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.first_player_super_evolution_unlock_turn
    player.super_evolution_points = 2


class SourceSuperEvolvedConditionTests(unittest.TestCase):
    def test_live_source_and_missing_source_snapshot_paths(self):
        repository = CardRepository("data/cards.sqlite3")
        engine = _make_engine(RuleBook(), repository)
        source = Unit.summon(
            _card(99001), entity_id=engine.state.allocate_entity_id()
        )
        engine.players[0].board.append(source)
        condition = Condition(ConditionType.SOURCE_SUPER_EVOLVED)
        context = EvalContext(
            controller=0,
            players=engine.players,
            source_entity_id=source.entity_id,
        )

        self.assertFalse(evaluate_condition(condition, context))
        source.evolved = True
        source.super_evolved = True
        source.super_evolved_turn = engine.turn
        self.assertTrue(evaluate_condition(condition, context))
        engine.players[0].board.remove(source)
        self.assertFalse(evaluate_condition(condition, context))
        context.source_snapshot = SourceStateSnapshot(
            entity_id=source.entity_id,
            controller=0,
            card_id=source.definition.card_id,
            card_type=source.definition.card_type,
            attack=source.attack,
            health=source.health,
            evolved=True,
            super_evolved=True,
            effective_keywords=frozenset(),
        )
        self.assertTrue(evaluate_condition(condition, context))

    def test_not_composition_distinguishes_normal_and_super_evolution(self):
        repository = CardRepository("data/cards.sqlite3")
        engine = _make_engine(RuleBook(), repository)
        source = Unit.summon(
            _card(99002), entity_id=engine.state.allocate_entity_id()
        )
        engine.players[0].board.append(source)
        condition = Condition(
            ConditionType.NOT,
            conditions=[Condition(ConditionType.SOURCE_SUPER_EVOLVED)],
        )
        context = EvalContext(
            controller=0,
            players=engine.players,
            source_entity_id=source.entity_id,
        )
        self.assertTrue(evaluate_condition(condition, context))
        source.evolved = source.super_evolved = True
        source.super_evolved_turn = engine.turn
        self.assertFalse(evaluate_condition(condition, context))


class DatabaseAndClauseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_stats_text_abilities_references_and_modes(self):
        expected = {
            10002110: (
                "煌响使者·亨莉雅妲", "中立", 3, 3, 3,
                "<ev>【<color=Keyword>进化时</color>】回复自己的主战者2点生命值。</ev>\n"
                "<sev>【<color=Keyword>超进化时</color>】改为回复4点。</sev>",
                {("进化时", "进化时"), ("超进化时", "超进化时")},
                set(),
            ),
            10042110: (
                "猛攻的龙战士", "龙族", 4, 4, 5,
                "<ev>【<color=Keyword>进化时</color>】选择对手的战场上的1个随从，对其造成4点伤害。</ev>\n"
                "<sev>【<color=Keyword>超进化时</color>】改为对对手的战场上的所有随从造成4点伤害。</sev>",
                {("进化时", "进化时"), ("超进化时", "超进化时")},
                set(),
            ),
            10062110: (
                "铁拳神父", "主教", 4, 5, 4,
                "<ev>【<color=Keyword>进化时</color>】选择对手的战场上的1个生命值为3或以下的随从，使其消失。</ev>\n"
                "<sev>【<color=Keyword>超进化时</color>】改为使对手的战场上的所有生命值为3或以下的随从消失。</sev>",
                {("进化时", "进化时"), ("超进化时", "超进化时")},
                set(),
            ),
            10072120: (
                "魔钢骑兵", "超越者", 5, 4, 4,
                "【<color=Keyword>守护</color>】\n<hr><ev>【<color=Keyword>进化时</color>】召唤1个『<color=Keyword>魔钢骑兵</color>』。</ev>\n"
                "<sev>【<color=Keyword>超进化时</color>】改为召唤2个。</sev>",
                {
                    ("守护", "守护"),
                    ("进化时", "进化时"),
                    ("超进化时", "超进化时"),
                },
                {(10072120, "魔钢骑兵")},
            ),
            10541110: (
                "涌泉打水人", "龙族", 2, 2, 2,
                "<ev>【<color=Keyword>进化时</color>】选择对手的战场上的1个随从，对其造成5点伤害。</ev>\n"
                "<sev>【<color=Keyword>超进化时</color>】改为选择2个随从。</sev>",
                {("进化时", "进化时"), ("超进化时", "超进化时")},
                set(),
            ),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT json_extract(c.raw_json, '$.name_chs'),
                               cl.class_name, cl.type_name, c.cost,
                               c.attack, c.life, st.text_chs
                        FROM cards c
                        JOIN card_localizations cl
                          ON cl.card_id=c.card_id AND cl.language='zh-CN'
                        JOIN skill_texts st ON st.card_id=c.card_id
                        WHERE c.card_id=? ORDER BY st.position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(
                        row,
                        [(
                            values[0], values[1], "随从", values[2],
                            values[3], values[4], values[5],
                        )],
                    )
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT ability_keyword, raw_keyword FROM card_abilities WHERE card_id=?",
                            (card_id,),
                        )),
                        values[6],
                    )
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT referenced_card_id, referenced_name FROM card_references WHERE card_id=?",
                            (card_id,),
                        )),
                        values[7],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_all_five_cards_are_exact_with_hash_and_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                audit = info["clause_audit"]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(audit["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(audit["structured_evidence"], STRUCTURED_EVIDENCE[card_id])
                self.assertEqual(
                    audit["test_evidence"],
                    ["tests/test_real_evolution_replacement_batch.py"],
                )
                self.assertTrue(audit["implemented_text"])


class RealEvolutionReplacementBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 1201) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_normal_branch_guard(self):
        expected = {
            10002110: (EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 2, 4),
            10042110: (EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 4, 4),
            10062110: (EffectKind.BANISH, TargetKind.ENEMY_UNIT, 0, 0),
            10072120: (EffectKind.SUMMON, TargetKind.OWN_LEADER, 0, 0),
            10541110: (EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 5, 5),
        }
        for card_id, (kind, target, amount, super_amount) in expected.items():
            with self.subTest(card_id=card_id):
                normal = self.rulebook.operations_for(card_id, Trigger.EVOLVE)
                super_ops = self.rulebook.operations_for(card_id, Trigger.SUPER_EVOLVE)
                self.assertEqual(len(normal), 1)
                self.assertEqual((normal[0].kind, normal[0].target, normal[0].amount), (kind, target, amount))
                self.assertEqual(normal[0].conditions[0].type, ConditionType.NOT)
                self.assertEqual(
                    normal[0].conditions[0].conditions[0].type,
                    ConditionType.SOURCE_SUPER_EVOLVED,
                )
                self.assertEqual(super_ops[0].amount, super_amount)
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10072120), ("守护",)
        )
        self.assertEqual(
            self.rulebook.operations_for(10541110, Trigger.SUPER_EVOLVE)[0].target_count,
            2,
        )

    def test_heal_is_two_normally_and_replaced_by_exactly_four_on_super(self):
        normal = self.fresh_engine()
        _enable_evolution(normal)
        normal.players[0].health = 10
        source = _play_real(normal, self.repository, 10002110)
        normal.apply(Evolve(0, source.entity_id))
        self.assertEqual(normal.players[0].health, 12)

        super_engine = self.fresh_engine()
        _enable_super_evolution(super_engine)
        super_engine.players[0].health = 10
        source = _play_real(super_engine, self.repository, 10002110)
        super_engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(super_engine.players[0].health, 14)
        self.assertNotEqual(super_engine.players[0].health, 16)

        capped = self.fresh_engine()
        _enable_super_evolution(capped)
        capped.players[0].health = 18
        source = _play_real(capped, self.repository, 10002110)
        capped.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(capped.players[0].health, capped.config.starting_health)

    def test_damage_replaces_selected_normal_target_with_all_on_super(self):
        normal = self.fresh_engine()
        _enable_evolution(normal)
        targets = [_add_enemy(normal, 3100 + index) for index in range(2)]
        source = _play_real(normal, self.repository, 10042110)
        normal.apply(Evolve(0, source.entity_id))
        self.assertEqual(normal.state.pending_choice.target_count, 1)
        _choose(normal, targets[0].entity_id)
        self.assertEqual([target.health for target in targets], [4, 8])

        super_engine = self.fresh_engine()
        _enable_super_evolution(super_engine)
        targets = [_add_enemy(super_engine, 3200 + index) for index in range(2)]
        source = _play_real(super_engine, self.repository, 10042110)
        super_engine.apply(SuperEvolve(0, source.entity_id))
        self.assertIsNone(super_engine.state.pending_choice)
        self.assertEqual([target.health for target in targets], [4, 4])

    def test_banish_filters_health_and_super_banishes_all_eligible(self):
        normal = self.fresh_engine()
        _enable_evolution(normal)
        eligible = _add_enemy(normal, 3300, life=3)
        ineligible = _add_enemy(normal, 3301, life=4)
        source = _play_real(normal, self.repository, 10062110)
        normal.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [option.entity_id for option in normal.state.pending_choice.options],
            [eligible.entity_id],
        )
        _choose(normal, eligible.entity_id)
        self.assertNotIn(eligible, normal.players[1].board)
        self.assertIn(ineligible, normal.players[1].board)
        self.assertNotIn(eligible.definition, normal.players[1].graveyard)
        self.assertEqual(normal.players[1].banished[-1].card_id, eligible.definition.card_id)

        super_engine = self.fresh_engine()
        _enable_super_evolution(super_engine)
        eligible = [
            _add_enemy(super_engine, 3400, life=1),
            _add_enemy(super_engine, 3401, life=3),
        ]
        ineligible = _add_enemy(super_engine, 3402, life=4)
        source = _play_real(super_engine, self.repository, 10062110)
        super_engine.apply(SuperEvolve(0, source.entity_id))
        self.assertTrue(all(unit not in super_engine.players[1].board for unit in eligible))
        self.assertIn(ineligible, super_engine.players[1].board)
        self.assertEqual(
            {record.card_id for record in super_engine.players[1].banished},
            {3400, 3401},
        )

    def test_no_eligible_target_does_not_block_evolution(self):
        for command_type, enable in (
            (Evolve, _enable_evolution),
            (SuperEvolve, _enable_super_evolution),
        ):
            with self.subTest(command=command_type.__name__):
                engine = self.fresh_engine()
                enable(engine)
                _add_enemy(engine, 3500, life=4)
                source = _play_real(engine, self.repository, 10062110)
                engine.apply(command_type(0, source.entity_id))
                self.assertTrue(source.evolved)
                self.assertIsNone(engine.state.pending_choice)

    def test_self_summon_is_one_normally_two_on_super_and_keeps_ward(self):
        normal = self.fresh_engine()
        _enable_evolution(normal)
        source = _play_real(normal, self.repository, 10072120)
        normal.apply(Evolve(0, source.entity_id))
        copies = [unit for unit in normal.players[0].board if unit.definition.card_id == 10072120]
        self.assertEqual(len(copies), 2)
        self.assertTrue(all(unit.has_guard for unit in copies))
        self.assertEqual(sum(unit.evolved for unit in copies), 1)

        super_engine = self.fresh_engine()
        _enable_super_evolution(super_engine)
        source = _play_real(super_engine, self.repository, 10072120)
        super_engine.apply(SuperEvolve(0, source.entity_id))
        copies = [unit for unit in super_engine.players[0].board if unit.definition.card_id == 10072120]
        self.assertEqual(len(copies), 3)
        self.assertTrue(all(unit.has_guard for unit in copies))
        self.assertEqual(sum(unit.super_evolved for unit in copies), 1)

    def test_super_self_summon_respects_each_remaining_board_slot(self):
        engine = self.fresh_engine()
        _enable_super_evolution(engine)
        for index in range(3):
            filler = Unit.summon(
                _card(3600 + index), entity_id=engine.state.allocate_entity_id()
            )
            engine.players[0].board.append(filler)
        source = _play_real(engine, self.repository, 10072120)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(len(engine.players[0].board), engine.config.max_board)
        self.assertEqual(
            sum(unit.definition.card_id == 10072120 for unit in engine.players[0].board),
            2,
        )

    def test_two_target_super_damage_clamps_shortage_and_rejects_duplicate(self):
        shortage = self.fresh_engine()
        _enable_super_evolution(shortage)
        target = _add_enemy(shortage, 3700)
        source = _play_real(shortage, self.repository, 10541110)
        shortage.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(shortage.state.pending_choice.target_count, 1)
        _choose(shortage, target.entity_id)
        self.assertEqual(target.health, 3)

        engine = self.fresh_engine()
        _enable_super_evolution(engine)
        targets = [_add_enemy(engine, 3710 + index) for index in range(2)]
        source = _play_real(engine, self.repository, 10541110)
        engine.apply(SuperEvolve(0, source.entity_id))
        choice = Choose(0, f"entity:{targets[0].entity_id}")
        engine.apply(choice)
        before = engine.deterministic_fingerprint()
        with self.assertRaisesRegex(IllegalCommand, "Choice option is invalid"):
            engine.apply(choice)
        self.assertEqual(engine.deterministic_fingerprint(), before)
        _choose(engine, targets[1].entity_id)
        self.assertEqual([target.health for target in targets], [3, 3])

    def test_two_target_super_damage_revalidates_target_leaving_play(self):
        engine = self.fresh_engine()
        _enable_super_evolution(engine)
        targets = [_add_enemy(engine, 3800 + index) for index in range(2)]
        source = _play_real(engine, self.repository, 10541110)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose(engine, targets[0].entity_id)
        engine.players[1].board.remove(targets[0])
        engine._send_to_graveyard(
            1,
            targets[0].definition,
            "test_target_left_play",
            source_entity_id=targets[0].entity_id,
        )
        _choose(engine, targets[1].entity_id)
        self.assertEqual(targets[0].health, 8)
        self.assertEqual(targets[1].health, 3)
        self.assertIsNone(engine.state.pending_choice)

    def test_replacement_sequence_is_deterministic(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh_engine(seed=1237)
            _enable_super_evolution(engine)
            targets = [_add_enemy(engine, 3900 + index) for index in range(2)]
            source = _play_real(engine, self.repository, 10541110)
            engine.apply(SuperEvolve(0, source.entity_id))
            for target in reversed(targets):
                _choose(engine, target.entity_id)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_rl_mask_exposes_super_evolve_then_two_distinct_choices(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(4000, 4040)],
            [_card(card_id) for card_id in range(5000, 5040)],
            class_a=1,
            class_b=1,
            seed=1249,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=1249)
        env.players[0].max_mana = env.players[0].mana = 10
        env.players[0].turns_started = env.FIRST_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN
        targets = [_add_enemy(env.core, 4100 + index) for index in range(2)]
        _put_in_hand(env.core, self.repository.get(10541110))
        env.step(env.PLAY_OFFSET)
        source_index = len(env.players[0].board) - 1
        action = env.SUPER_EVOLVE_OFFSET + source_index
        self.assertTrue(env.action_mask()[action])
        env.step(action)
        self.assertEqual(env.core.state.pending_choice.target_count, 2)
        mask = env.action_mask()
        self.assertEqual(sum(mask[env.CHOICE_OFFSET:]), 2)
        env.step(env.CHOICE_OFFSET)
        mask = env.action_mask()
        self.assertEqual(sum(mask[env.CHOICE_OFFSET:]), 1)
        env.step(env.CHOICE_OFFSET)
        self.assertEqual([target.health for target in targets], [3, 3])


if __name__ == "__main__":
    unittest.main()
