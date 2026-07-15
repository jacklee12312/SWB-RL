# -*- coding: utf-8 -*-
"""Exact audits for simple real cards that use already-complete tokens."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Attack, Evolve, PlayCard
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


CARD_IDS = (10011110, 10112310, 10151120, 10151130, 10512120)
SOURCE_HASHES = {
    10011110: "97606f7d745abd960c3d7a1e0194ddb5a6e01cffe2045baa39250bc4b0fb4b0d",
    10112310: "893b2bce5b99bc63aef8e54de2f3292deed43180506b5c040f5a9cdb872ad8a8",
    10151120: "d76d43298fe17b30e50895d78214a38116dec5b649a3c2abd8e0f84bf1410e31",
    10151130: "036969af7d13badd544a05136c6b0976d184a0f5a59bde8620604749fa8c7dfb",
    10512120: "19e549ea0cba0e885ce86eef0a0e9f2ee66ad2785f71da8fa9a9086e0af97253",
}
STRUCTURED_EVIDENCE = {
    10011110: {
        "triggers": ["fanfare"],
        "effect_kinds": ["add_card", "add_card"],
    },
    10112310: {"triggers": ["play"], "effect_kinds": ["add_card", "draw"]},
    10151120: {
        "triggers": ["evolve"],
        "effect_kinds": ["summon", "summon"],
    },
    10151130: {
        "triggers": ["last_words"],
        "effect_kinds": ["summon", "summon"],
    },
    10512120: {
        "triggers": ["intrinsic_keywords"],
        "effect_kinds": [
            "keyword:\u7a81\u8fdb",
            "keyword:\u5fc5\u6740",
            "keyword:\u5438\u8840",
        ],
    },
}


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=overrides.get("class_id", 0),
        class_name=overrides.get("class_name", "\u4e2d\u7acb"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "\u968f\u4ece"),
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
    seed: int = 601,
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


def _put_in_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    origin: CardOrigin = CardOrigin.DECK,
) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=origin,
    )
    engine.players[0].hand.append(card)
    engine.players[0].hand_entity_ids.append(card.entity_id)
    return card


def _play_real(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
) -> Unit | None:
    card = _put_in_hand(engine, repository.get(card_id))
    index = engine.players[0].hand.index(card)
    engine.apply(PlayCard(0, index))
    if card.definition.card_type == "\u6cd5\u672f":
        return None
    return next(
        unit
        for unit in engine.players[0].board
        if unit.definition.card_id == card_id
    )


def _add_filler_unit(engine: GameEngine, card_id: int) -> Unit:
    unit = Unit.summon(
        _card(card_id),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[0].board.append(unit)
    return unit


def _destroy(engine: GameEngine, source: Unit) -> None:
    source.health = 0
    engine._stabilize()


class DatabaseAndCoverageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_cards_match_printed_stats_keywords_and_references(self):
        expected = {
            10011110: ("\u5996\u7cbe\u9a6f\u670d\u8005", "\u7cbe\u7075", "\u968f\u4ece", 2, 1, 1, 1),
            10112310: ("\u85b0\u4ea4\u7684\u601d\u6155", "\u7cbe\u7075", "\u6cd5\u672f", 3, None, None, 1),
            10151120: ("\u65e0\u540d\u6076\u9b54", "\u68a6\u9b47", "\u968f\u4ece", 2, 2, 2, 1),
            10151130: ("\u767d\u9aa8\u5c11\u5973", "\u68a6\u9b47", "\u968f\u4ece", 3, 1, 2, 1),
            10512120: ("\u548c\u6c14\u853c\u853c\u7684\u5996\u7cbe", "\u7cbe\u7075", "\u968f\u4ece", 5, 2, 6, 1),
        }
        expected_abilities = {
            10011110: {("\u5165\u573a\u66f2", "\u5165\u573a\u66f2")},
            10151120: {("\u8fdb\u5316\u65f6", "\u8fdb\u5316\u65f6")},
            10151130: {("\u8c22\u5e55\u66f2", "\u8c22\u5e55\u66f2")},
            10512120: {
                ("\u7a81\u8fdb", "\u7a81\u8fdb"),
                ("\u5fc5\u6740", "\u6bc1\u706d"),
                ("\u5438\u8840", "\u8679\u5438"),
            },
        }
        expected_references = {
            10011110: [(0, 90011110, "\u5996\u7cbe")],
            10112310: [(0, 90011310, "\u68ee\u6797\u7684\u5965\u79d8")],
            10151120: [(0, 90051120, "\u8759\u8760")],
            10151130: [(0, 90051110, "\u9ab8\u9aa8\u58eb\u5175")],
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT json_extract(c.raw_json, '$.name_chs'),
                               cl.class_name, cl.type_name, c.cost, c.attack,
                               c.life, cs.is_collectible
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

    def test_all_five_cards_are_exact_with_structured_and_hash_evidence(self):
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
                    STRUCTURED_EVIDENCE[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_exact_token_followup_batch.py"],
                )

    def test_each_reused_token_has_a_new_auditable_producer(self):
        report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        tokens = {card["card_id"]: card for card in report["cards"]}
        expected = {
            90011110: (10011110, "add_card"),
            90011310: (10112310, "add_card"),
            90051120: (10151120, "summon"),
            90051110: (10151130, "summon"),
        }
        for token_id, (source_id, entry_kind) in expected.items():
            with self.subTest(token_id=token_id):
                self.assertEqual(
                    tokens[token_id]["category"], "entry_behavior_complete"
                )
                self.assertIn(
                    {
                        "source_card_id": source_id,
                        "entry_kind": entry_kind,
                        "rule_file": "real_exact_token_followup_batch.json",
                        "rule_group": "rules",
                    },
                    tokens[token_id]["authored_producers"],
                )


class RealExactTokenBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 601) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_preserve_counts_order_and_intrinsic_keywords(self):
        tamer = self.rulebook.operations_for(10011110, Trigger.FANFARE)
        self.assertEqual(
            [(op.kind, op.target, op.card_id, op.amount) for op in tamer],
            [
                (EffectKind.ADD_CARD, TargetKind.OWN_LEADER, 90011110, 0),
                (EffectKind.ADD_CARD, TargetKind.OWN_LEADER, 90011110, 0),
            ],
        )
        spell = self.rulebook.operations_for(10112310, Trigger.PLAY)
        self.assertEqual(
            [(op.kind, op.card_id, op.amount) for op in spell],
            [
                (EffectKind.ADD_CARD, 90011310, 0),
                (EffectKind.DRAW, None, 1),
            ],
        )
        for card_id, trigger, token_id in (
            (10151120, Trigger.EVOLVE, 90051120),
            (10151130, Trigger.LAST_WORDS, 90051110),
        ):
            operations = self.rulebook.operations_for(card_id, trigger)
            self.assertEqual(
                [
                    (operation.kind, operation.target, operation.card_id, operation.amount)
                    for operation in operations
                ],
                [(EffectKind.SUMMON, TargetKind.OWN_LEADER, token_id, 0)] * 2,
            )
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10512120),
            ("\u7a81\u8fdb", "\u5fc5\u6740", "\u5438\u8840"),
        )

    def test_fairy_tamer_adds_two_tokens_and_overflows_only_the_second(self):
        normal = self.fresh_engine(seed=607)
        _play_real(normal, self.repository, 10011110)
        fairies = [card for card in normal.players[0].hand if card.card_id == 90011110]
        self.assertEqual(len(fairies), 2)
        self.assertTrue(all(card.origin is CardOrigin.TOKEN for card in fairies))

        overflow = self.fresh_engine(seed=607)
        source = _put_in_hand(overflow, self.repository.get(10011110))
        while len(overflow.players[0].hand) < overflow.config.max_hand:
            _put_in_hand(overflow, _card(7000 + len(overflow.players[0].hand)))
        overflow.apply(PlayCard(0, overflow.players[0].hand.index(source)))
        self.assertEqual(overflow.players[0].hand[-1].card_id, 90011110)
        discarded = [
            card for card in overflow.players[0].graveyard
            if card.definition.card_id == 90011110
        ]
        self.assertEqual(len(discarded), 1)
        self.assertIs(discarded[0].origin, CardOrigin.TOKEN)

    def test_fragrantwood_whispers_adds_before_draw_and_preserves_overflow_order(self):
        normal = self.fresh_engine(seed=613)
        top_card_id = normal.players[0].deck[-1].card_id
        _play_real(normal, self.repository, 10112310)
        self.assertEqual(
            [card.card_id for card in normal.players[0].hand],
            [90011310, top_card_id],
        )
        self.assertIs(normal.players[0].hand[0].origin, CardOrigin.TOKEN)

        overflow = self.fresh_engine(seed=613)
        source = _put_in_hand(overflow, self.repository.get(10112310))
        while len(overflow.players[0].hand) < overflow.config.max_hand:
            _put_in_hand(overflow, _card(7100 + len(overflow.players[0].hand)))
        drawn_id = overflow.players[0].deck[-1].card_id
        overflow.apply(PlayCard(0, overflow.players[0].hand.index(source)))
        self.assertEqual(overflow.players[0].hand[-1].card_id, 90011310)
        self.assertIn(
            drawn_id,
            [card.definition.card_id for card in overflow.players[0].graveyard],
        )

    def test_nameless_demon_evolve_summons_two_drain_bats_with_capacity_limit(self):
        normal = self.fresh_engine(seed=617)
        normal.players[0].turns_started = 4
        source = _play_real(normal, self.repository, 10151120)
        normal.apply(Evolve(0, source.entity_id))
        bats = [
            unit for unit in normal.players[0].board
            if unit.definition.card_id == 90051120
        ]
        self.assertEqual(len(bats), 2)
        self.assertTrue(all(unit.origin is CardOrigin.TOKEN for unit in bats))
        self.assertTrue(all(unit.has_keyword("\u5438\u8840") for unit in bats))

        capacity = self.fresh_engine(seed=619)
        capacity.players[0].turns_started = 4
        for card_id in range(7200, 7203):
            _add_filler_unit(capacity, card_id)
        source = _play_real(capacity, self.repository, 10151120)
        capacity.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            sum(
                unit.definition.card_id == 90051120
                for unit in capacity.players[0].board
            ),
            1,
        )

    def test_bonemancer_last_words_summons_two_skeletons_with_capacity_limit(self):
        normal = self.fresh_engine(seed=623)
        source = _play_real(normal, self.repository, 10151130)
        _destroy(normal, source)
        skeletons = [
            unit for unit in normal.players[0].board
            if unit.definition.card_id == 90051110
        ]
        self.assertEqual(len(skeletons), 2)
        self.assertTrue(all(unit.origin is CardOrigin.TOKEN for unit in skeletons))
        self.assertEqual([(unit.attack, unit.health) for unit in skeletons], [(1, 1)] * 2)

        capacity = self.fresh_engine(seed=627)
        for card_id in range(7300, 7304):
            _add_filler_unit(capacity, card_id)
        source = _play_real(capacity, self.repository, 10151130)
        _destroy(capacity, source)
        self.assertEqual(
            sum(
                unit.definition.card_id == 90051110
                for unit in capacity.players[0].board
            ),
            1,
        )

    def test_fairy_beastwhisperer_rush_bane_and_drain_all_resolve(self):
        engine = self.fresh_engine(seed=631)
        engine.players[0].health = 10
        source = _play_real(engine, self.repository, 10512120)
        target = Unit.summon(
            _card(7400, attack=0, life=8),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board.append(target)

        self.assertTrue(source.can_attack)
        self.assertFalse(source.can_attack_leader)
        self.assertTrue(source.has_keyword("\u5fc5\u6740"))
        self.assertTrue(source.has_keyword("\u5438\u8840"))
        command = Attack(0, source.entity_id, target.entity_id)
        self.assertIn(command, engine.legal_commands())
        engine.apply(command)

        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[0].health, 12)

    def test_seeded_spell_sequence_has_identical_fingerprint(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh_engine(seed=641)
            _play_real(engine, self.repository, 10112310)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_rush_attack_is_exposed_by_rl_action_mask(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(3000, 3040)],
            [_card(card_id) for card_id in range(4000, 4040)],
            class_a=2,
            class_b=2,
            seed=647,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=647)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        _put_in_hand(env.core, self.repository.get(10512120))
        env.players[0].max_mana = env.players[0].mana = 10
        target = Unit.summon(
            _card(7500, attack=0, life=8),
            entity_id=env.core.state.allocate_entity_id(),
        )
        env.players[1].board.append(target)

        play = PlayCard(0, 0)
        play_action = env._encode_command(play)
        self.assertIsNotNone(play_action)
        self.assertTrue(env.action_mask()[play_action])
        env.step(play_action)

        source = next(
            unit for unit in env.players[0].board
            if unit.definition.card_id == 10512120
        )
        attack = Attack(0, source.entity_id, target.entity_id)
        attack_action = env._encode_command(attack)
        self.assertIsNotNone(attack_action)
        self.assertTrue(env.action_mask()[attack_action])


if __name__ == "__main__":
    unittest.main()
