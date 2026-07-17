# -*- coding: utf-8 -*-
"""Exact real-card follow-up composed from established engine primitives."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Attack, Choose, Evolve, PlayCard
from swb.engine.effects import EffectKind, ExprType
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.state import DeathCause, DestroyedFollowerRecord, HandCard
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10151140,
    10652110,
    10352120,
    10113140,
    10501110,
    10253120,
    10761120,
    10251110,
    10233110,
    10532120,
)
SOURCE_HASHES = {
    10151140: "c4e0d5569c42a80b886bfc984446bbc3b2e4d60c84d49623336f3db4444abaea",
    10652110: "6a313926f9c1b12561c1c82d617497a4fc84eb85676db4cce1eade8a38f3b856",
    10352120: "46b6fa11080a0ef5e18891f1193d614594b32aca9823358cfdf28b7745759f9a",
    10113140: "b24184f852ab015ceaa19b6613e19f8695b5acfa62a6d095bc31efefbfb212b4",
    10501110: "122936da3a789b489fab16a025764d5da9dd8af8e2ed74f9a3d47fc1e4f968e0",
    10253120: "95e4da1306400c9ba9157cf35f1ab87e5ec91ea354356ded8eecd556d7198199",
    10761120: "a57384700b6e514122f96df283270e0d9969d6788549f25048273dc6802eb6b3",
    10251110: "4580634f7028001ecb32cdf0f193222335c09e20eddabf5f6dc8cc08d4d5f9a2",
    10233110: "e8a39384b85b98c1108e1047d414d8be7e397523fe40f935fae55bb504258aaa",
    10532120: "3f0ab2832f42b2ee7f44d8ca89eb5040f88a95783ebf1e6fd525a4b571b032d6",
}


def _record(definition, sequence: int) -> DestroyedFollowerRecord:
    return DestroyedFollowerRecord(
        definition=definition,
        owner=0,
        death_sequence=sequence,
        cause=DeathCause.EFFECT_DESTROY,
    )


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


class RealExistingPrimitivesFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2801):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_every_complete_clause(self):
        self.assertEqual(
            self.rulebook.operations_for(10151140, Trigger.FANFARE)[0].amount,
            4,
        )
        self.assertEqual(
            self.rulebook.modes_for(10652110)[0].operations[0].amount,
            9,
        )
        self.assertEqual(
            [operation.amount for operation in self.rulebook.modes_for(10352120)[0].operations],
            [5, 3],
        )
        combo = self.rulebook.operations_for(10113140, Trigger.FANFARE)[0]
        self.assertIs(combo.amount_expr.type, ExprType.CONTROLLER_COMBO)
        preacher = self.rulebook.operations_for(10761120, Trigger.EVOLVE)[0]
        self.assertIs(preacher.amount_expr.type, ExprType.CONTROLLER_HAND_COUNT)
        self.assertEqual(preacher.amount_expr.card_filter.card_type, "护符")
        self.assertEqual(
            self.rulebook.operations_for(10251110, Trigger.EVOLVE)[0].target_count,
            2,
        )
        self.assertEqual(self.rulebook.spellboost_cost_reduction(10532120), 1)

    def test_juggling_ghost_reanimates_highest_eligible_follower(self):
        engine = self.fresh(seed=3)
        eligible = _card(99101, cost=4, attack=4, life=6)
        lower = _card(99102, cost=2)
        too_large = _card(99103, cost=5)
        engine.state.destroyed_followers = [
            _record(lower, 1),
            _record(too_large, 2),
            _record(eligible, 3),
        ]

        _play(engine, self.repository, 10151140)

        reanimated = [
            unit
            for unit in engine.players[0].board
            if unit.origin is CardOrigin.REANIMATED
        ]
        self.assertEqual([unit.definition.card_id for unit in reanimated], [99101])
        event = next(
            event
            for event in engine.event_history
            if event.type is EventType.REANIMATE_RESOLVED
        )
        self.assertEqual(event.amount, 4)

    def test_enhance_reanimate_is_absent_in_normal_mode_and_present_in_enhance(self):
        candidate = _card(99110, cost=9)

        normal = self.fresh(seed=5)
        normal.state.destroyed_followers = [_record(candidate, 1)]
        _play(normal, self.repository, 10652110)
        self.assertFalse(any(
            unit.origin is CardOrigin.REANIMATED
            for unit in normal.players[0].board
        ))

        enhanced = self.fresh(seed=5)
        enhanced.state.destroyed_followers = [_record(candidate, 1)]
        source = _play(
            enhanced,
            self.repository,
            10652110,
            mode_id="enhance_8",
        )
        self.assertTrue(source.has_keyword("守护"))
        self.assertEqual(
            [
                unit.definition.card_id
                for unit in enhanced.players[0].board
                if unit.origin is CardOrigin.REANIMATED
            ],
            [99110],
        )

    def test_new_gravedigger_reanimates_five_then_three_with_capacity_boundary(self):
        five = _card(99120, cost=5)
        three = _card(99121, cost=3)
        engine = self.fresh(seed=7)
        engine.state.destroyed_followers = [_record(three, 1), _record(five, 2)]

        _play(engine, self.repository, 10352120, mode_id="enhance_7")

        self.assertEqual(
            [
                unit.definition.card_id
                for unit in engine.players[0].board
                if unit.origin is CardOrigin.REANIMATED
            ],
            [99120, 99121],
        )

        full = self.fresh(seed=7)
        for index in range(3):
            _put_unit(full, 0, _card(99200 + index))
        full.state.destroyed_followers = [_record(three, 1), _record(five, 2)]
        _play(full, self.repository, 10352120, mode_id="enhance_7")
        self.assertEqual(len(full.players[0].board), full.config.max_board)
        self.assertEqual(
            [
                unit.definition.card_id
                for unit in full.players[0].board
                if unit.origin is CardOrigin.REANIMATED
            ],
            [99120],
        )

    def test_combo_and_other_follower_counts_are_snapshotted_on_fanfare(self):
        combo = self.fresh(seed=11)
        combo.players[0].cards_played_this_turn = 2
        definition = self.repository.get(10113140)
        insect = _play(combo, self.repository, 10113140)
        self.assertEqual(insect.attack, definition.attack + 3)
        self.assertTrue(insect.has_keyword("疾驰"))

        board = self.fresh(seed=13)
        _put_unit(board, 0, _card(99210))
        _put_unit(board, 0, _card(99211))
        _put_unit(board, 1, _card(99212))
        definition = self.repository.get(10253120)
        source = _play(board, self.repository, 10253120)
        self.assertEqual(source.attack, definition.attack + 2)
        self.assertEqual(board.players[0].health, 18)
        self.assertTrue(source.has_keyword("疾驰"))

    def test_brushstroke_monster_excludes_itself_but_counts_either_battlefield(self):
        alone = self.fresh(seed=17)
        definition = self.repository.get(10501110)
        source = _play(alone, self.repository, 10501110)
        self.assertEqual((source.attack, source.max_health), (definition.attack, definition.life))

        with_other = self.fresh(seed=19)
        _put_unit(with_other, 0, _card(99220, cost=2))
        _put_unit(with_other, 1, _card(99221, cost=1))
        source = _play(with_other, self.repository, 10501110)
        self.assertEqual(
            (source.attack, source.max_health),
            (definition.attack + 1, definition.life + 1),
        )

    def test_broad_area_preacher_draws_amulets_then_counts_current_hand(self):
        engine = self.fresh(seed=23)
        amulets = [
            _card(99230 + index, card_type="护符", attack=None, life=None)
            for index in range(2)
        ]
        engine.players[0].deck = [
            _card(99232, card_type="法术", attack=None, life=None),
            amulets[0],
            _card(99233),
            amulets[1],
        ]
        enemies = [
            _put_unit(engine, 1, _card(99240 + index, life=5))
            for index in range(2)
        ]
        source = _play(engine, self.repository, 10761120)
        self.assertEqual(
            [card.definition.card_type for card in engine.players[0].hand],
            ["护符", "护符"],
        )
        engine.players[0].turns_started = engine.config.evolution_unlock_turn

        engine.apply(Evolve(0, source.entity_id))

        self.assertEqual([enemy.health for enemy in enemies], [3, 3])

    def test_silver_bullet_collects_two_targets_before_destroy_and_always_self_damages(self):
        engine = self.fresh(seed=29)
        source = _play(engine, self.repository, 10251110)
        enemies = [
            _put_unit(engine, 1, _card(99250 + index))
            for index in range(3)
        ]
        engine.players[0].turns_started = engine.config.evolution_unlock_turn

        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(engine.state.pending_choice.target_count, 2)
        _choose_entity(engine, enemies[1].entity_id)
        self.assertIn(enemies[1], engine.players[1].board)
        _choose_entity(engine, enemies[2].entity_id)

        self.assertEqual(engine.players[1].board, [enemies[0]])
        self.assertEqual(engine.players[0].health, 18)

        empty = self.fresh(seed=31)
        source = _play(empty, self.repository, 10251110)
        empty.players[0].turns_started = empty.config.evolution_unlock_turn
        empty.apply(Evolve(0, source.entity_id))
        self.assertIsNone(empty.state.pending_choice)
        self.assertEqual(empty.players[0].health, 18)

    def test_owen_clash_spellboosts_hand_and_evolve_draws_two_followers(self):
        clash = self.fresh(seed=37)
        source = _play(clash, self.repository, 10233110)
        source.can_attack = True
        target = _put_unit(clash, 1, _card(99260, attack=0, life=10))
        hand_cards = [
            _put_hand(clash, _card(99261)),
            _put_hand(clash, _card(99262, card_type="法术", attack=None, life=None)),
        ]
        clash.apply(Attack(0, source.entity_id, target.entity_id))
        self.assertEqual([card.spellboost_count for card in hand_cards], [1, 1])
        self.assertTrue(source.has_keyword("守护"))

        evolve = self.fresh(seed=41)
        evolve.players[0].deck = [
            _card(99270, card_type="法术", attack=None, life=None),
            _card(99271),
            _card(99272, card_type="护符", attack=None, life=None),
            _card(99273),
        ]
        source = _play(evolve, self.repository, 10233110)
        evolve.players[0].turns_started = evolve.config.evolution_unlock_turn
        evolve.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [card.definition.card_type for card in evolve.players[0].hand],
            ["随从", "随从"],
        )

    def test_aftertaste_poet_reduces_cost_draws_and_spellboosts_on_last_words(self):
        engine = self.fresh(seed=43)
        engine.players[0].deck = [_card(99280)]
        kept = _put_hand(engine, _card(99281))
        poet = _put_hand(engine, self.repository.get(10532120))
        engine._ensure_entity_ids()
        poet.apply_spellboost(2)
        self.assertEqual(poet.current_cost, poet.definition.cost - 2)

        engine.apply(PlayCard(0, 0))

        source = next(
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == 10532120
        )
        self.assertEqual(len(engine.players[0].hand), 2)
        source.health = 0
        engine._stabilize()
        self.assertEqual(
            [card.spellboost_count for card in engine.players[0].hand],
            [1, 1],
        )
        self.assertIn(kept, engine.players[0].hand)

    def test_seeded_reanimate_tie_and_fingerprint_replay_match(self):
        outcomes = []
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=47)
            engine.state.destroyed_followers = [
                _record(_card(99290, cost=4), 1),
                _record(_card(99291, cost=4), 2),
            ]
            _play(engine, self.repository, 10151140)
            outcomes.append(tuple(
                unit.definition.card_id
                for unit in engine.players[0].board
                if unit.origin is CardOrigin.REANIMATED
            ))
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_rl_masks_expose_enhance_and_two_step_evolve_choice(self):
        deck = [_card(99300 + index) for index in range(40)]
        enhance = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=53,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        enhance.reset(seed=53)
        enhance.players[0].hand.clear()
        enhance.players[0].hand_entity_ids.clear()
        enhance.players[0].board.clear()
        enhance.players[0].mana = enhance.players[0].max_mana = 10
        _put_hand(enhance.core, self.repository.get(10652110))
        command = PlayCard(0, 0, mode_id="enhance_8")
        action = enhance._encode_command(command)
        self.assertTrue(enhance.action_mask()[action])
        enhance.step(action)

        raven = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=59,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        raven.reset(seed=59)
        for player in raven.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        source = _put_unit(raven.core, 0, self.repository.get(10251110))
        targets = [
            _put_unit(raven.core, 1, _card(99400 + index))
            for index in range(2)
        ]
        raven.players[0].turns_started = raven.core.config.evolution_unlock_turn
        evolve = Evolve(0, source.entity_id)
        evolve_action = raven._encode_command(evolve)
        self.assertTrue(raven.action_mask()[evolve_action])
        raven.step(evolve_action)
        first_actions = [
            action
            for action in range(raven.CHOICE_OFFSET, raven.GRAVEYARD_CHOICE_OFFSET)
            if raven.action_mask()[action]
        ]
        self.assertEqual(len(first_actions), 2)
        raven.step(first_actions[0])
        second_actions = [
            action
            for action in range(raven.CHOICE_OFFSET, raven.GRAVEYARD_CHOICE_OFFSET)
            if raven.action_mask()[action]
        ]
        self.assertEqual(len(second_actions), 1)
        raven.step(second_actions[0])
        self.assertTrue(all(target not in raven.players[1].board for target in targets))


class ExistingPrimitivesFollowupDatabaseAuditTests(unittest.TestCase):
    def test_database_text_references_and_modes_match_reviewed_sources(self):
        expected_phrases = {
            10151140: ("Reanimate (4)",),
            10652110: ("Enhance (8)", "Reanimate (9)", "Ward"),
            10352120: ("Reanimate (5)", "Reanimate (3)"),
            10113140: ("your Combo", "Storm"),
            10501110: ("another 1-base-cost card",),
            10253120: ("other allied followers", "2 damage to your leader", "Storm"),
            10761120: ("Draw 2 amulets", "number of amulets in your hand"),
            10251110: ("Select 2 enemy followers", "2 damage to your leader"),
            10233110: ("Spellboost your hand", "Draw 2 followers"),
            10532120: ("On Spellboost", "Draw a card", "Spellboost your hand"),
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id, phrases in expected_phrases.items():
                with self.subTest(card_id=card_id):
                    rows = connection.execute(
                        "SELECT text_eng FROM skill_texts WHERE card_id=? ORDER BY position",
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(len(rows), 1)
                    normalized = re.sub(r"<[^>]+>", "", rows[0][0])
                    for phrase in phrases:
                        self.assertIn(phrase, normalized)
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

    def test_cards_have_exact_mapped_clause_evidence(self):
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
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_existing_primitives_followup_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
