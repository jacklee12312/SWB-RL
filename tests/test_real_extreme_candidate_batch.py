# -*- coding: utf-8 -*-
"""Exact real-card and Token tests for current-state extreme candidates."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, HandCard, Unit


COLLECTIBLE_IDS = (
    10103310,
    10341310,
    10503310,
    10552310,
    10613310,
    10743110,
    10822310,
    10832320,
)
TOKEN_IDS = (90011120, 90022110)
SOURCE_HASHES = {
    10103310: "6b3f22a4e04da0416785aecc0c1e85068e938d246535e20da206d0b49c19f94e",
    10341310: "e4a67a924a0b1e4e234910d996dd8baf82bb3e228a8bed2ae36b008ea08401b0",
    10503310: "da3932c2f1378835828c9de062b73b769e37d765f990fad21a1ddb587395b8f2",
    10552310: "7db3816a93130a2a940587a4fa3bd2f8848c45b22e499fd1b95b19fd48a5f95b",
    10613310: "503c22e033c324d39914dbfce5832b587b52d74b115e6d806b075eae24bcaf92",
    10743110: "543a97169710e0e510e0ace4eddbad624ad4f602fc206ecbf05f30eb1231aaed",
    10822310: "1b5be70d3578dfda535dff298f89b32dca641beb5cfc4efb46720a0c5af5c93f",
    10832320: "0e413195f62b2ea951c92b1f82d45cb40f5ceef059b528425ca9c235f9e6a33d",
    90011120: "582e285f4d10d37e28b88292900f200aeb5a86b9bcff6b77d101ba5c4ddfa142",
    90022110: "7abde1fa8f2c436a6805d99abe026718965b9776b4b5a2dffe05f57cd5b2776f",
}
STRUCTURED_EVIDENCE = {
    10103310: {"triggers": ["play"], "effect_kinds": ["destroy", "damage_unit"]},
    10341310: {"triggers": ["play"], "effect_kinds": ["damage_unit", "conditional", "damage_leader"]},
    10503310: {"triggers": ["play", "play_modes"], "effect_kinds": ["draw", "destroy", "play_mode", "damage_unit", "damage_leader"]},
    10552310: {"triggers": ["play"], "effect_kinds": ["damage_leader"]},
    10613310: {"triggers": ["play"], "effect_kinds": ["choose_one", "destroy", "summon"]},
    10743110: {"triggers": ["fanfare"], "effect_kinds": ["choose_one", "destroy", "evolve_unit", "destroy", "add_keyword"]},
    10822310: {"triggers": ["play", "play_modes"], "effect_kinds": ["buff_unit", "play_mode", "summon", "summon", "summon"]},
    10832320: {"triggers": ["play"], "effect_kinds": ["damage_unit", "damage_leader", "earth_rite", "add_card"]},
    90011120: {"triggers": ["turn_end", "intrinsic_keywords"], "effect_kinds": ["evolve_unit", "keyword:守护"]},
    90022110: {"triggers": ["intrinsic_keywords"], "effect_kinds": ["keyword:突进"]},
}


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
        life=overrides.get("life", 5),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _engine(rulebook: RuleBook, repository: CardRepository, seed: int = 2441) -> GameEngine:
    engine = GameEngine(
        [_card(i) for i in range(1000, 1040)],
        [_card(i) for i in range(2000, 2040)],
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


def _put(engine: GameEngine, definition: CardDefinition) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


def _play(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
    *,
    mode: str = "normal",
):
    definition = repository.get(card_id)
    _put(engine, definition)
    engine.apply(PlayCard(0, 0, mode))
    if definition.card_type == "法术":
        return None
    return next(
        entity
        for entity in reversed(engine.players[0].board)
        if entity.definition.card_id == card_id
    )


def _unit(
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


def _sigil(engine: GameEngine, count: int = 2) -> Amulet:
    sigil = Amulet(
        definition=_card(
            90031210,
            card_set_id=90000,
            card_type="护符",
            attack=None,
            life=None,
            keywords=frozenset({"土之印"}),
            is_collectible=False,
        ),
        entity_id=engine.state.allocate_entity_id(),
        earth_sigil_count=count,
        entered_turn=engine.turn,
        origin=CardOrigin.TOKEN,
    )
    engine.players[0].board.append(sigil)
    return sigil


class DatabaseAndAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_cards_references_and_modes_match_snapshot(self):
        all_ids = COLLECTIBLE_IDS + TOKEN_IDS
        with closing(sqlite3.connect(self.db_path)) as connection:
            placeholders = ",".join("?" for _ in all_ids)
            self.assertEqual(
                connection.execute(
                    f"SELECT COUNT(*) FROM cards WHERE card_id IN ({placeholders})",
                    all_ids,
                ).fetchone()[0],
                len(all_ids),
            )
            collectible_placeholders = ",".join("?" for _ in COLLECTIBLE_IDS)
            self.assertEqual(
                connection.execute(
                    f"SELECT card_id, referenced_card_id FROM card_references WHERE card_id IN ({collectible_placeholders}) ORDER BY card_id",
                    COLLECTIBLE_IDS,
                ).fetchall(),
                [(10613310, 90011120), (10822310, 90022110), (10832320, 10832320)],
            )
            self.assertEqual(
                connection.execute(
                    f"SELECT COUNT(*) FROM alt_modes WHERE card_id IN ({collectible_placeholders})",
                    COLLECTIBLE_IDS,
                ).fetchone()[0],
                0,
            )

    def test_collectibles_and_tokens_have_hash_and_direct_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in COLLECTIBLE_IDS:
            with self.subTest(card_id=card_id):
                audit = report["classifications"][str(card_id)]["clause_audit"]
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(audit["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(audit["structured_evidence"], STRUCTURED_EVIDENCE[card_id])
                self.assertEqual(audit["test_evidence"], ["tests/test_real_extreme_candidate_batch.py"])

        token_report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        tokens = {card["card_id"]: card for card in token_report["cards"]}
        for card_id in TOKEN_IDS:
            with self.subTest(token_id=card_id):
                token = tokens[card_id]
                self.assertEqual(token["category"], "entry_behavior_complete")
                self.assertEqual(token["explicit_coverage"], "exact")
                self.assertEqual(
                    {producer["source_card_id"] for producer in token["authored_producers"]},
                    {10611110, 10611310, 10612110, 10613310, 10614110}
                    if card_id == 90011120
                    else {10822310},
                )


class RealExtremeCandidateBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, seed: int = 2441) -> GameEngine:
        return _engine(self.rulebook, self.repository, seed)

    def test_divine_thunder_randomly_destroys_only_a_highest_attack_tie_then_aoes(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=41)
            low = _unit(engine, 1, 3000, attack=3, life=5)
            high_a = _unit(engine, 1, 3001, attack=6, life=5)
            high_b = _unit(engine, 1, 3002, attack=6, life=5)
            _play(engine, self.repository, 10103310)
            self.assertIn(low, engine.players[1].board)
            self.assertEqual(low.health, 4)
            remaining_high = [unit for unit in (high_a, high_b) if unit in engine.players[1].board]
            self.assertEqual(len(remaining_high), 1)
            self.assertEqual(remaining_high[0].health, 4)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_thunder_fury_hits_all_highest_health_units_and_overflow_highest_leaders(self):
        below = self.fresh()
        below.players[0].max_mana = below.players[0].mana = 6
        own = _unit(below, 0, 3100, life=7)
        enemy_tied = _unit(below, 1, 3101, life=7)
        enemy_lower = _unit(below, 1, 3102, life=6)
        below.players[0].health = 15
        below.players[1].health = 20
        _play(below, self.repository, 10341310)
        self.assertEqual((own.health, enemy_tied.health, enemy_lower.health), (2, 2, 6))
        self.assertEqual((below.players[0].health, below.players[1].health), (15, 20))

        overflow = self.fresh()
        overflow.players[0].health = 12
        overflow.players[1].health = 18
        _play(overflow, self.repository, 10341310)
        self.assertEqual((overflow.players[0].health, overflow.players[1].health), (12, 15))

        tied = self.fresh()
        tied.players[0].health = tied.players[1].health = 16
        _play(tied, self.repository, 10341310)
        self.assertEqual((tied.players[0].health, tied.players[1].health), (13, 13))

    def test_world_presentation_draws_destroys_and_enhance_appends_four_damage(self):
        normal = self.fresh(seed=47)
        normal.players[0].deck = [_card(3200), _card(3201), _card(3202)]
        high = _unit(normal, 1, 3203, attack=7, life=8)
        low = _unit(normal, 1, 3204, attack=2, life=8)
        _play(normal, self.repository, 10503310)
        self.assertEqual(len(normal.players[0].hand), 2)
        self.assertNotIn(high, normal.players[1].board)
        self.assertEqual(low.health, 8)

        enhanced = self.fresh(seed=47)
        enhanced.players[0].deck = [_card(3210), _card(3211), _card(3212)]
        high = _unit(enhanced, 1, 3213, attack=7, life=8)
        low = _unit(enhanced, 1, 3214, attack=2, life=8)
        _play(enhanced, self.repository, 10503310, mode="enhance_10")
        self.assertEqual(len(enhanced.players[0].hand), 2)
        self.assertNotIn(high, enhanced.players[1].board)
        self.assertEqual(low.health, 4)
        self.assertEqual(enhanced.players[1].health, 16)

    def test_cruel_blast_damages_every_leader_tied_for_lowest_current_health(self):
        lower = self.fresh()
        lower.players[0].health = 10
        lower.players[1].health = 15
        _play(lower, self.repository, 10552310)
        self.assertEqual((lower.players[0].health, lower.players[1].health), (7, 15))

        tied = self.fresh()
        tied.players[0].health = tied.players[1].health = 11
        _play(tied, self.repository, 10552310)
        self.assertEqual((tied.players[0].health, tied.players[1].health), (8, 8))

    def test_charitable_spear_modes_destroy_or_summon_complete_evolving_token(self):
        destroy = self.fresh(seed=53)
        low = _unit(destroy, 1, 3300, attack=2)
        high = _unit(destroy, 1, 3301, attack=5)
        _play(destroy, self.repository, 10613310)
        destroy.apply(Choose(0, "choose_one:destroy"))
        self.assertIn(low, destroy.players[1].board)
        self.assertNotIn(high, destroy.players[1].board)

        summon = self.fresh(seed=53)
        _play(summon, self.repository, 10613310)
        summon.apply(Choose(0, "choose_one:summon"))
        token = summon.players[0].board[0]
        self.assertEqual(token.definition.card_id, 90011120)
        self.assertEqual(token.origin, CardOrigin.TOKEN)
        self.assertTrue(token.has_keyword("守护"))
        ep_before = summon.players[0].evolution_points
        summon.apply(EndTurn(0))
        self.assertTrue(token.evolved)
        self.assertEqual(summon.players[0].evolution_points, ep_before)

    def test_dragon_pioneer_mode_choice_precedes_extreme_destroy_and_grants_branch(self):
        evolve = self.fresh(seed=59)
        target = _unit(evolve, 1, 3400, attack=6)
        source = _play(evolve, self.repository, 10743110)
        self.assertIn(target, evolve.players[1].board)
        self.assertFalse(source.evolved)
        evolve.apply(Choose(0, "choose_one:evolve"))
        self.assertNotIn(target, evolve.players[1].board)
        self.assertTrue(source.evolved)

        ambush = self.fresh(seed=59)
        target = _unit(ambush, 1, 3401, attack=6)
        source = _play(ambush, self.repository, 10743110)
        self.assertFalse(source.has_keyword("潜行"))
        ambush.apply(Choose(0, "choose_one:ambush"))
        self.assertNotIn(target, ambush.players[1].board)
        self.assertTrue(source.has_keyword("潜行"))
        self.assertFalse(source.evolved)

    def test_rebirth_debuffs_highest_attack_and_enhance_summons_three_rush_tokens(self):
        normal = self.fresh(seed=61)
        high = _unit(normal, 1, 3500, attack=8, life=12)
        low = _unit(normal, 1, 3501, attack=2, life=12)
        _play(normal, self.repository, 10822310)
        self.assertEqual((high.attack, high.health), (0, 2))
        self.assertEqual((low.attack, low.health), (2, 12))

        enhanced = self.fresh(seed=61)
        _play(enhanced, self.repository, 10822310, mode="enhance_6")
        tokens = [unit for unit in enhanced.players[0].board if unit.definition.card_id == 90022110]
        self.assertEqual(len(tokens), 3)
        self.assertTrue(all(unit.has_keyword("突进") for unit in tokens))
        self.assertTrue(all(unit.origin is CardOrigin.TOKEN for unit in tokens))

    def test_landmine_hits_highest_then_leader_and_earth_rite_adds_exact_copy(self):
        insufficient = self.fresh(seed=67)
        high = _unit(insufficient, 1, 3600, attack=8, life=10)
        low = _unit(insufficient, 1, 3601, attack=3, life=10)
        _play(insufficient, self.repository, 10832320)
        self.assertEqual((high.health, low.health), (2, 10))
        self.assertEqual(insufficient.players[1].health, 18)
        self.assertNotIn(10832320, [card.card_id for card in insufficient.players[0].hand])

        active = self.fresh(seed=67)
        _sigil(active, 2)
        _unit(active, 1, 3610, attack=8, life=10)
        _play(active, self.repository, 10832320)
        self.assertEqual(active.players[0].earth_sigils, 0)
        copies = [card for card in active.players[0].hand if card.card_id == 10832320]
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0].origin, CardOrigin.GENERATED)

    def test_rl_generic_choice_mask_exposes_both_spear_modes(self):
        env = ShadowverseEnv(
            [_card(i) for i in range(4000, 4040)],
            [_card(i) for i in range(4100, 4140)],
            class_a=1,
            class_b=1,
            seed=71,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=71)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].max_mana = env.players[0].mana = 10
        _put(env.core, self.repository.get(10613310))
        env.step(env.PLAY_OFFSET)
        enabled = [
            action
            for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        ]
        self.assertEqual(enabled, [env.CHOICE_OFFSET, env.CHOICE_OFFSET + 1])


if __name__ == "__main__":
    unittest.main()
