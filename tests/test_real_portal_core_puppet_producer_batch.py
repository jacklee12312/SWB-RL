# -*- coding: utf-8 -*-
"""Exact Portalcraft Core and Puppet producer chains."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import EndTurn, Evolve, PlayCard
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _choose,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (
    10071110,
    10071120,
    10071310,
    10072110,
    10072210,
    10171120,
    10171130,
    10172110,
    10172120,
    10173110,
)
TOKEN_IDS = (90071110, 90071120, 90071210, 90071220, 90072110, 90072120)
SOURCE_HASHES = {
    10071110: "17fb0db76812382bfbf772f1fbd99a03358c3e37ecc72a8496d3413d7f173074",
    10071120: "6edafdb030760b86db36792de9c1fa0295aa18e9facc35e4bf8301fd29002b8f",
    10071310: "ae0443a9bd80fc0877a4ef456d84aa020155373e0374af49bc536f6cbaad1f2a",
    10072110: "b7b12a02555f9c6b51aa4172ecb092e81c3fb7056550c1a92dab4ad2bb6133be",
    10072210: "0106f1578b29eb6ab694498db3e5f8f70a5a342520f8c001d0e64f6b645d9547",
    10171120: "d1bb7e40cb1de446d45e68b6599f193fc95b573ccdf26657dc119216f3619e0e",
    10171130: "7e90bc6ccfa831800feb12e85ba0dec81e2915bb09881afe2bc105df4b04e952",
    10172110: "e5724cc845a9ebfc9053b7bacad4eb870846aa1c80491ef7466d7bd465dd88a3",
    10172120: "b18fd6e8bfb464abf6e2864ffdbf9bacec342c44ff6c7836d251ca4bccc138a5",
    10173110: "18c7a0f971807e28a9b153d02ba6839ef32a1edad5f87790046878505b493e35",
}


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


class RealPortalCorePuppetProducerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1801):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_simple_fanfares_add_the_exact_generated_cards(self):
        cases = (
            (10071110, 90071210),
            (10071120, 90071120),
            (10072110, 90071220),
        )
        for source_id, generated_id in cases:
            with self.subTest(source_id=source_id):
                engine = self.fresh(seed=source_id)
                source = _play(engine, self.repository, source_id)
                self.assertEqual([card.card_id for card in engine.players[0].hand], [generated_id])
                self.assertIs(engine.players[0].hand[0].origin, CardOrigin.TOKEN)
                if source_id == 10071110:
                    self.assertTrue(source.has_keyword("突进"))

    def test_dimensional_shot_requires_target_then_destroys_and_adds_both_cores(self):
        no_target = self.fresh(seed=3)
        _put_hand(no_target, self.repository.get(10071310))
        before = no_target.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            no_target.apply(PlayCard(0, 0))
        self.assertEqual(no_target.deterministic_fingerprint(), before)

        engine = self.fresh(seed=5)
        target = _put_unit(engine, 1, _card(8101))
        _play(engine, self.repository, 10071310)
        _choose(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [90071210, 90071220],
        )
        self.assertTrue(all(card.origin is CardOrigin.TOKEN for card in engine.players[0].hand))

    def test_puppet_theater_adds_on_play_and_only_at_own_turn_end(self):
        engine = self.fresh(seed=7)
        theater = _play(engine, self.repository, 10072210)
        self.assertEqual(theater.countdown, 2)
        self.assertEqual([card.card_id for card in engine.players[0].hand], [90071110])

        engine.apply(EndTurn(0))
        self.assertEqual([card.card_id for card in engine.players[0].hand], [90071110, 90071110])
        engine.apply(EndTurn(1))
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand if card.card_id == 90071110],
            [90071110, 90071110],
        )
        self.assertEqual(theater.countdown, 1)

    def test_dierk_summons_castle_artifact_and_board_shortage_skips_safely(self):
        engine = self.fresh(seed=11)
        source = _play(engine, self.repository, 10171120)
        castle = next(unit for unit in engine.players[0].board if unit is not source)
        self.assertEqual(castle.definition.card_id, 90072120)
        self.assertIs(castle.origin, CardOrigin.TOKEN)

        full = self.fresh(seed=13)
        for index in range(4):
            _put_unit(full, 0, _card(8200 + index))
        source = _play(full, self.repository, 10171120)
        self.assertIn(source, full.players[0].board)
        self.assertFalse(any(unit.definition.card_id == 90072120 for unit in full.players[0].board))

    def test_gunner_fanfare_and_targeted_evolve_damage_with_no_target_skip(self):
        engine = self.fresh(seed=17)
        _enable_evolution(engine)
        target = _put_unit(engine, 1, _card(8301, life=5))
        source = _play(engine, self.repository, 10171130)
        self.assertEqual([card.card_id for card in engine.players[0].hand], [90071210])
        engine.apply(Evolve(0, source.entity_id))
        _choose(engine, target.entity_id)
        self.assertEqual(target.health, 2)

        no_target = self.fresh(seed=19)
        _enable_evolution(no_target)
        source = _play(no_target, self.repository, 10171130)
        no_target.apply(Evolve(0, source.entity_id))
        self.assertTrue(source.evolved)
        self.assertIsNone(no_target.state.pending_choice)

    def test_lucina_adds_both_cores_then_evolve_summons_attack_artifact(self):
        engine = self.fresh(seed=23)
        _enable_evolution(engine)
        source = _play(engine, self.repository, 10172110)
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [90071210, 90071220],
        )
        engine.apply(Evolve(0, source.entity_id))
        attack = next(unit for unit in engine.players[0].board if unit is not source)
        self.assertEqual(attack.definition.card_id, 90072110)
        self.assertIs(attack.origin, CardOrigin.TOKEN)

    def test_puppeteer_and_miriam_repeat_their_fanfares_on_evolve(self):
        cases = (
            (10172120, [90071110, 90071110]),
            (10173110, [90071210, 90071220, 90071210, 90071220]),
        )
        for source_id, expected in cases:
            with self.subTest(source_id=source_id):
                engine = self.fresh(seed=source_id)
                _enable_evolution(engine)
                source = _play(engine, self.repository, source_id)
                engine.apply(Evolve(0, source.entity_id))
                self.assertEqual([card.card_id for card in engine.players[0].hand], expected)

    def test_repeated_core_production_is_seed_reproducible(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=29)
            _enable_evolution(engine)
            source = _play(engine, self.repository, 10173110)
            engine.apply(Evolve(0, source.entity_id))
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_all_collectibles_are_exact_and_token_audit_lists_real_producers(self):
        coverage = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in COLLECTIBLE_IDS:
            with self.subTest(card_id=card_id):
                info = coverage["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_portal_core_puppet_producer_batch.py"],
                )

        token_report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        tokens = {card["card_id"]: card for card in token_report["cards"]}
        expected_sources = {
            90071110: {10072210, 10172120},
            90071120: {10071120},
            90071210: {10071110, 10071310, 10171130, 10172110, 10173110},
            90071220: {10071310, 10072110, 10172110, 10173110},
            90072110: {10172110},
            90072120: {10171120},
        }
        for token_id in TOKEN_IDS:
            with self.subTest(token_id=token_id):
                self.assertEqual(tokens[token_id]["category"], "entry_behavior_complete")
                actual_sources = {
                    producer["source_card_id"]
                    for producer in tokens[token_id]["authored_producers"]
                    if producer["rule_file"] == "real_portal_core_puppet_producer_batch.json"
                }
                self.assertEqual(actual_sources, expected_sources[token_id])


if __name__ == "__main__":
    unittest.main()
