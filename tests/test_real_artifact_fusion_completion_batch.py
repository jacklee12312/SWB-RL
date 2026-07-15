# -*- coding: utf-8 -*-
"""Exact real-token coverage for the complete Artifact fusion chain."""

from __future__ import annotations

import unittest
from dataclasses import replace

from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.commands import BeginFusion, Choose, EndTurn, PlayCard
from swb.engine.resolution import IllegalCommand
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _put_hand,
    _put_unit,
    _play,
)


TOKEN_IDS = (
    90071210,
    90071220,
    90072110,
    90072120,
    90073110,
    90073120,
    90073130,
    90074110,
)

SOURCE_HASHES = {
    90071210: "8adc7f0d7e16e01ba111df4017bfa5a980608518606051a13f7a45b904222ae8",
    90071220: "9ec881e97b2e61698b3a3836cadedc776b5e35d37b0628ed5c7d55ad4718be13",
    90072110: "f157e288e7070ca7f6368fbe09d6cb8a7dd878cfd2d77446c9691b8f2e7f3bbe",
    90072120: "d128805bf1f4860a13a07395e4b1e1fe1a231f39a5e65446009b3618aa395c7d",
    90073110: "2c37c76399cea215c602c658ae2aa4b06cb408e9f50c2de4a76228101191ed5d",
    90073120: "7c2044572fb68db347c0052aaa6e6dc12f899d8d96551e250e2bc8def4e4cf38",
    90073130: "e94292dc284d2e79f28cb2ad20359aa7cdb491d893fbc1d804ea77af0eeaaa98",
    90074110: "e5f5195045c2e674b7e7241aaf2ab223ad48deb55fd97f860f49e5f3c08b3f80",
}


def _fusion_options(engine) -> set[str]:
    return {option.option_id for option in engine.state.pending_choice.options}


def _fuse_one(engine, target, material) -> None:
    engine.apply(BeginFusion(0, target.entity_id))
    engine.apply(Choose(0, f"hand:{material.entity_id}"))
    engine.apply(Choose(0, "fusion:confirm"))


def _next_own_turn(engine) -> None:
    engine.apply(EndTurn(0))
    engine.apply(EndTurn(1))


class RealArtifactFusionCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from swb.engine.card_rules import RuleBook

        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 901):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_cores_are_unplayable_and_only_accept_artifact_amulets(self):
        engine = self.fresh(seed=3)
        future = _put_hand(engine, self.repository.get(90071210))
        past = _put_hand(engine, self.repository.get(90071220))
        invalid_amulet = _put_hand(
            engine,
            _card(991001, card_type="护符", attack=None, life=None),
        )
        engine._ensure_entity_ids()
        self.assertTrue(future.cannot_be_played)
        self.assertNotIn(PlayCard(0, engine.players[0].hand.index(future)), engine.legal_commands())
        before = (tuple(engine.players[0].hand), engine.players[0].mana)
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, engine.players[0].hand.index(future)))
        self.assertEqual((tuple(engine.players[0].hand), engine.players[0].mana), before)

        engine.apply(BeginFusion(0, future.entity_id))
        options = _fusion_options(engine)
        self.assertIn(f"hand:{past.entity_id}", options)
        self.assertNotIn(f"hand:{invalid_amulet.entity_id}", options)
        engine.apply(Choose(0, f"hand:{past.entity_id}"))
        engine.apply(Choose(0, "fusion:confirm"))
        self.assertEqual(future.card_id, 90072110)
        self.assertEqual(future.fused_material_ids, [past.entity_id])
        self.assertFalse(future.cannot_be_played)
        self.assertIn("突进", {ability.value for ability in future.definition.abilities})

    def test_attack_and_castle_artifacts_use_all_three_total_cost_branches(self):
        for cost, result_id in ((1, 90073110), (2, 90073120), (3, 90073130)):
            with self.subTest(cost=cost, result_id=result_id):
                engine = self.fresh(seed=10 + cost)
                target_id = 90072110 if cost != 2 else 90072120
                target = _put_hand(engine, self.repository.get(target_id))
                material = _put_hand(
                    engine,
                    replace(
                        self.repository.get(90071210),
                        card_id=991100 + cost,
                        cost=cost,
                    ),
                )
                _fuse_one(engine, target, material)
                self.assertEqual(target.card_id, result_id)
                self.assertEqual(target.fused_material_ids, [material.entity_id])

    def test_alpha_accepts_only_beta_gamma_and_requires_two_distinct_kinds(self):
        engine = self.fresh(seed=19)
        alpha = _put_hand(engine, self.repository.get(90073110))
        beta_one = _put_hand(engine, self.repository.get(90073120))
        beta_two = _put_hand(engine, self.repository.get(90073120))
        gamma = _put_hand(engine, self.repository.get(90073130))
        invalid_artifact = _put_hand(engine, self.repository.get(90072110))

        engine.apply(BeginFusion(0, alpha.entity_id))
        options = _fusion_options(engine)
        self.assertIn(f"hand:{beta_one.entity_id}", options)
        self.assertIn(f"hand:{gamma.entity_id}", options)
        self.assertNotIn(f"hand:{invalid_artifact.entity_id}", options)
        engine.apply(Choose(0, f"hand:{beta_one.entity_id}"))
        engine.apply(Choose(0, "fusion:confirm"))
        self.assertEqual(alpha.card_id, 90073110)

        _next_own_turn(engine)
        _fuse_one(engine, alpha, beta_two)
        self.assertEqual(alpha.card_id, 90073110)

        _next_own_turn(engine)
        _fuse_one(engine, alpha, gamma)
        self.assertEqual(alpha.card_id, 90074110)
        self.assertEqual(
            alpha.fused_material_ids,
            [beta_one.entity_id, beta_two.entity_id, gamma.entity_id],
        )

    def test_destruction_artifacts_resolve_their_own_turn_end_effects(self):
        engine = self.fresh(seed=23)
        engine.players[0].health = 15
        _put_unit(engine, 0, self.repository.get(90073110))
        _put_unit(engine, 0, self.repository.get(90073120))
        _put_unit(engine, 0, self.repository.get(90073130))
        enemies = [
            _put_unit(engine, 1, _card(992000 + index, life=5))
            for index in range(2)
        ]

        engine.apply(EndTurn(0))

        self.assertEqual(engine.players[0].health, 18)
        self.assertEqual(engine.players[1].health, 17)
        self.assertEqual([enemy.health for enemy in enemies], [2, 2])

    def test_omega_fanfare_and_all_intrinsic_keywords_are_exact(self):
        engine = self.fresh(seed=29)
        engine.players[0].health = 12
        doomed = _put_unit(engine, 1, _card(993001, life=5))
        survivor = _put_unit(engine, 1, _card(993002, life=6))

        omega = _play(engine, self.repository, 90074110)

        self.assertNotIn(doomed, engine.players[1].board)
        self.assertEqual(survivor.health, 1)
        self.assertEqual(engine.players[0].health, 17)
        self.assertTrue(omega.has_keyword("疾驰"))
        self.assertTrue(omega.has_keyword("守护"))
        self.assertTrue(omega.has_keyword("灵气"))

    def test_all_eight_tokens_are_exact_and_keep_authored_entry_paths(self):
        report = _build_token_audit("data/cards.sqlite3", "data/rules")
        cards = {card["card_id"]: card for card in report["cards"]}
        for card_id in TOKEN_IDS:
            with self.subTest(card_id=card_id):
                info = cards[card_id]
                self.assertEqual(info["category"], "entry_behavior_complete")
                self.assertEqual(info["explicit_coverage"], "exact")
                self.assertTrue(info["authored_producers"])
                self.assertEqual(len(SOURCE_HASHES[card_id]), 64)


if __name__ == "__main__":
    unittest.main()
