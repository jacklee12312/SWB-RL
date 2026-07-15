# -*- coding: utf-8 -*-
"""Evolution-resource primitives and five exact real-card audits."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Attack, Choose, Evolve, PlayCard
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


CARD_IDS = (10414110, 10653310, 10801120, 10804120, 10854110)
SOURCE_HASHES = {
    10414110: "1102aae9720ffeb91ffaf01fac247f65d1cdf5a949a9db177a5af6c25b6fc7a5",
    10653310: "a48afcd515e0b6314781a0b40bca943512f351e4bbf680fdfc35201d453b127c",
    10801120: "b4ce77b27f2c085d731e52439e02ad21de00ed0a289a4e8ceec6aee63e387667",
    10804120: "138b7bc0a51d856d9c47f5068f62488f2757106af1497a9dd4d37237f6997af9",
    10854110: "7b74a1c2e73c9fa4777b22b1ad21fb238e86fa959c30cdfe459ff5248f7461e1",
}
STRUCTURED_EVIDENCE = {
    10414110: {
        "triggers": ["union_burst", "intrinsic_keywords"],
        "effect_kinds": ["restore_evolution_points", "keyword:突进"],
    },
    10653310: {
        "triggers": ["play"],
        "effect_kinds": [
            "choose_one", "restore_evolution_points", "damage_unit",
        ],
    },
    10801120: {
        "triggers": ["fanfare"],
        "effect_kinds": ["restore_evolution_points"],
    },
    10804120: {
        "triggers": ["fanfare", "intrinsic_keywords"],
        "effect_kinds": [
            "restore_super_evolution_points", "keyword:守护",
        ],
    },
    10854110: {
        "triggers": ["fanfare", "evolve"],
        "effect_kinds": [
            "choose_one",
            "damage_leader",
            "heal_leader",
            "damage_unit",
            "restore_evolution_points",
            "choose_one",
            "draw",
            "restore_mana",
        ],
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
        life=overrides.get("life", 5),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _make_engine(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 1301,
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
) -> Unit | None:
    definition = repository.get(card_id)
    _put_in_hand(engine, definition)
    engine.apply(PlayCard(0, 0))
    if definition.card_type == "法术":
        return None
    return next(
        unit for unit in engine.players[0].board
        if unit.definition.card_id == card_id
    )


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


def _choose(engine: GameEngine, option_id: str) -> None:
    engine.apply(Choose(0, option_id))


class EvolutionResourcePrimitiveTests(unittest.TestCase):
    def test_schema_requires_positive_amount_and_own_leader(self):
        for kind in (
            "restore_evolution_points",
            "restore_super_evolution_points",
        ):
            with self.subTest(kind=kind, case="valid"):
                operation = _parse_operation(
                    {"kind": kind, "target": "own_leader", "amount": 1},
                    "test.json/operations[0]",
                    1,
                )
                self.assertEqual(operation.target, TargetKind.OWN_LEADER)
            for bad_amount in (None, 0, -1, True):
                raw = {"kind": kind, "target": "own_leader"}
                if bad_amount is not None:
                    raw["amount"] = bad_amount
                with self.subTest(kind=kind, bad_amount=bad_amount):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        _parse_operation(raw, "test.json/operations[0]", 1)
            with self.subTest(kind=kind, case="wrong_target"):
                with self.assertRaisesRegex(ValueError, "requires own_leader"):
                    _parse_operation(
                        {"kind": kind, "target": "enemy_leader", "amount": 1},
                        "test.json/operations[0]",
                        1,
                    )


class DatabaseAndClauseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_stats_text_abilities_and_no_refs_or_modes(self):
        expected = {
            10414110: (
                "风之法则·艾云尼亚", "精灵", "随从", 2, 2, 1,
                "【<color=Keyword>入场曲</color>】【<color=Keyword>奥义</color>】回复自己1点进化点。\n"
                "<hr>【<color=Keyword>突进</color>】",
                {("入场曲", "入场曲"), ("奥义", "奥义"), ("突进", "突进")},
            ),
            10653310: (
                "枯渴的天眼", "梦魇", "法术", 3, None, None,
                "【<color=Keyword>模式</color>】选择1个能力发动。\n"
                "<ridx=0>（1）回复自己1点进化点。</ridx>\n"
                "<ridx=1>（2）对对手的战场上的所有随从造成2点伤害。</ridx>",
                {("模式", "模式")},
            ),
            10801120: (
                "无尽旅途·蕾娜", "中立", "随从", 5, 4, 4,
                "【<color=Keyword>入场曲</color>】回复自己1点进化点。",
                {("入场曲", "入场曲")},
            ),
            10804120: (
                "高洁的黑翼·奥莉薇", "中立", "随从", 9, 7, 7,
                "【<color=Keyword>入场曲</color>】回复自己2点超进化点。\n"
                "<hr>【<color=Keyword>守护</color>】",
                {("入场曲", "入场曲"), ("守护", "守护")},
            ),
            10854110: (
                "出发的憧憬·苇剑&武津御", "梦魇", "随从", 8, 6, 5,
                "【<color=Keyword>入场曲</color>】【<color=Keyword>模式</color>】选择1个能力发动。\n"
                "<ridx=0>（1）对对手的主战者造成4点伤害。回复自己的主战者4点生命值。</ridx>\n"
                "<ridx=1>（2）对对手的战场上的所有随从造成5点伤害。回复自己1点进化点。</ridx>\n"
                "<hr><ev>【<color=Keyword>进化时</color>】【<color=Keyword>模式</color>】选择1个能力发动。\n"
                "<ridx=0>（1）抽取2张卡牌。</ridx>\n"
                "<ridx=1>（2）回复自己2点能量点。</ridx></ev>",
                {("入场曲", "入场曲"), ("模式", "模式"), ("进化时", "进化时")},
            ),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    rows = connection.execute(
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
                    self.assertEqual(rows, [values[:7]])
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT ability_keyword, raw_keyword FROM card_abilities WHERE card_id=?",
                            (card_id,),
                        )),
                        values[7],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM card_references WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
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
                    ["tests/test_real_evolution_resource_recovery_batch.py"],
                )


class RealEvolutionResourceBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 1301) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_include_nested_modes_and_intrinsic_keywords(self):
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10414110), ("突进",)
        )
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10804120), ("守护",)
        )
        burst = self.rulebook.union_bursts_for(10414110)
        self.assertEqual(len(burst), 1)
        self.assertEqual(burst[0].threshold, 10)
        self.assertEqual(
            burst[0].operations[0].kind,
            EffectKind.RESTORE_EVOLUTION_POINTS,
        )
        for card_id, trigger in (
            (10653310, Trigger.PLAY),
            (10854110, Trigger.FANFARE),
            (10854110, Trigger.EVOLVE),
        ):
            operation = self.rulebook.operations_for(card_id, trigger)[0]
            self.assertEqual(operation.kind, EffectKind.CHOOSE_ONE)
            self.assertEqual(len(operation.choose_one_options), 2)

    def test_union_burst_threshold_restores_ep_and_rush_is_intrinsic(self):
        below = self.fresh_engine()
        below.players[0].turns_started = 9
        below.players[0].evolution_points = 0
        source = _play_real(below, self.repository, 10414110)
        self.assertEqual(below.players[0].evolution_points, 0)
        self.assertTrue(source.has_keyword("突进"))
        self.assertFalse(any(
            event.type is EventType.UNION_BURST_ACTIVATED
            and event.source_id == source.entity_id
            for event in below.event_history
        ))

        active = self.fresh_engine()
        active.players[0].turns_started = 10
        active.players[0].evolution_points = 0
        source = _play_real(active, self.repository, 10414110)
        self.assertEqual(active.players[0].evolution_points, 1)
        self.assertTrue(source.has_keyword("突进"))
        burst = next(
            event for event in active.event_history
            if event.type is EventType.UNION_BURST_ACTIVATED
            and event.source_id == source.entity_id
        )
        self.assertEqual((burst.amount, burst.metadata["threshold"]), (10, 10))

    def test_follower_ep_restore_clamps_and_event_is_auditable(self):
        for before, expected, actual in ((0, 1, 1), (2, 2, 0)):
            with self.subTest(before=before):
                engine = self.fresh_engine()
                engine.players[0].evolution_points = before
                source = _play_real(engine, self.repository, 10801120)
                self.assertEqual(engine.players[0].evolution_points, expected)
                event = next(
                    event for event in engine.event_history
                    if event.type is EventType.EVOLUTION_POINTS_RESTORED
                    and event.source_id == source.entity_id
                )
                self.assertEqual(event.amount, actual)
                self.assertEqual(event.metadata["requested_amount"], 1)
                self.assertEqual(event.metadata["before"], before)
                self.assertEqual(event.metadata["after"], expected)

    def test_olivia_restores_sep_to_cap_and_ward_changes_attack_legality(self):
        engine = self.fresh_engine()
        engine.players[0].super_evolution_points = 1
        source = _play_real(engine, self.repository, 10804120)
        self.assertEqual(engine.players[0].super_evolution_points, 2)
        self.assertTrue(source.has_guard)
        event = next(
            event for event in engine.event_history
            if event.type is EventType.SUPER_EVOLUTION_POINTS_RESTORED
        )
        self.assertEqual(event.amount, 1)
        self.assertEqual(event.metadata["requested_amount"], 2)
        attacker = _add_unit(engine, 1, 1400, attack=3)
        attacker.can_attack = True
        attacker.attacks_remaining = 1
        attacker.rush_only = False
        engine.state.active_player = 1
        commands = engine.legal_commands()
        self.assertIn(Attack(1, attacker.entity_id, source.entity_id), commands)
        self.assertNotIn(Attack(1, attacker.entity_id, None), commands)

    def test_spell_modes_restore_ep_or_damage_all_and_illegal_choice_is_atomic(self):
        restore = self.fresh_engine()
        restore.players[0].evolution_points = 0
        _play_real(restore, self.repository, 10653310)
        self.assertEqual(
            [option.option_id for option in restore.state.pending_choice.options],
            ["choose_one:restore_ep", "choose_one:damage_all"],
        )
        before = restore.deterministic_fingerprint()
        with self.assertRaisesRegex(IllegalCommand, "Choice option is invalid"):
            _choose(restore, "choose_one:not_real")
        self.assertEqual(restore.deterministic_fingerprint(), before)
        _choose(restore, "choose_one:restore_ep")
        self.assertEqual(restore.players[0].evolution_points, 1)

        damage = self.fresh_engine()
        targets = [_add_unit(damage, 1, 1500 + index, life=2) for index in range(2)]
        _play_real(damage, self.repository, 10653310)
        _choose(damage, "choose_one:damage_all")
        self.assertTrue(all(target not in damage.players[1].board for target in targets))

    def test_departure_fanfare_modes_resolve_complete_distinct_branches(self):
        leader = self.fresh_engine()
        leader.players[0].health = 10
        source = _play_real(leader, self.repository, 10854110)
        self.assertIsNotNone(source)
        _choose(leader, "choose_one:leader_swing")
        self.assertEqual((leader.players[0].health, leader.players[1].health), (14, 16))

        board = self.fresh_engine()
        board.players[0].evolution_points = 0
        targets = [_add_unit(board, 1, 1600 + index, life=5) for index in range(2)]
        _play_real(board, self.repository, 10854110)
        _choose(board, "choose_one:board_and_ep")
        self.assertTrue(all(target not in board.players[1].board for target in targets))
        self.assertEqual(board.players[0].evolution_points, 1)

    def test_departure_evolve_modes_draw_two_or_restore_two_mana(self):
        draw = self.fresh_engine()
        draw.players[0].turns_started = draw.config.evolution_unlock_turn
        draw.players[0].evolution_points = 2
        source = _play_real(draw, self.repository, 10854110)
        _choose(draw, "choose_one:leader_swing")
        deck_before = len(draw.players[0].deck)
        draw.apply(Evolve(0, source.entity_id))
        _choose(draw, "choose_one:draw_two")
        self.assertEqual(deck_before - len(draw.players[0].deck), 2)
        self.assertEqual(draw.players[0].evolution_points, 1)

        mana = self.fresh_engine()
        mana.players[0].turns_started = mana.config.evolution_unlock_turn
        source = _play_real(mana, self.repository, 10854110)
        _choose(mana, "choose_one:leader_swing")
        mana.players[0].mana = 0
        mana.apply(Evolve(0, source.entity_id))
        _choose(mana, "choose_one:restore_two_mana")
        self.assertEqual(mana.players[0].mana, 2)

    def test_mode_sequence_is_seeded_and_deterministic(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh_engine(seed=1361)
            engine.players[0].turns_started = engine.config.evolution_unlock_turn
            source = _play_real(engine, self.repository, 10854110)
            _choose(engine, "choose_one:board_and_ep")
            engine.apply(Evolve(0, source.entity_id))
            _choose(engine, "choose_one:draw_two")
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_rl_masks_spell_and_evolve_mode_choices(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(4000, 4040)],
            [_card(card_id) for card_id in range(5000, 5040)],
            class_a=1,
            class_b=1,
            seed=1373,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=1373)
        env.players[0].max_mana = env.players[0].mana = 10
        env.players[0].evolution_points = 0
        _put_in_hand(env.core, self.repository.get(10653310))
        env.step(env.PLAY_OFFSET)
        mask = env.action_mask()
        self.assertTrue(mask[env.CHOICE_OFFSET])
        self.assertTrue(mask[env.CHOICE_OFFSET + 1])
        self.assertEqual(sum(mask), 2)
        env.step(env.CHOICE_OFFSET)
        self.assertEqual(env.players[0].evolution_points, 1)

        env.players[0].mana = 10
        env.players[0].turns_started = env.EVOLUTION_UNLOCK_TURN
        _put_in_hand(env.core, self.repository.get(10854110))
        env.step(env.PLAY_OFFSET)
        env.step(env.CHOICE_OFFSET)
        source_index = len(env.players[0].board) - 1
        evolve_action = env.EVOLVE_OFFSET + source_index
        self.assertTrue(env.action_mask()[evolve_action])
        env.step(evolve_action)
        mask = env.action_mask()
        self.assertTrue(mask[env.CHOICE_OFFSET])
        self.assertTrue(mask[env.CHOICE_OFFSET + 1])
        self.assertEqual(sum(mask), 2)


if __name__ == "__main__":
    unittest.main()
