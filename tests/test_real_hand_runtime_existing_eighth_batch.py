# -*- coding: utf-8 -*-
"""Direct contracts for the eighth exact hand-runtime rule slice."""

from __future__ import annotations

import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import (
    _build_coverage_report,
    _load_source_text_map,
    _source_text_sha256,
)
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Attack, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10302110,
    10303110,
    10022120,
    10722310,
    10271120,
    10471120,
    10223110,
)
SOURCE_HASHES = {
    10302110: "55be12acd325d6369de598f6641eda87beb7d1e278b0471f31708c92ad8ac1f1",
    10303110: "ba6a335ac042585c4e540d4262036c1e326ab5d1dfc783219fb4bb515fb7975f",
    10022120: "fb74c6affd1bd42b080d906cbd7b7e5b43c2cc6761b20a1938664c89805c267a",
    10722310: "1e7936c429e505c31cd03c4fe8d5fc2441024b1bbba2af1ee35b66b73e9be9ae",
    10271120: "83a61a08f645e918061614b061d4d5a325c2f56a1cc8a4e6a56d24fd673fc867",
    10471120: "1b5c9f7185bd82b6961c3a95b8163b43fb889861771d36ed77107969bd3d638d",
    10223110: "68c9115eb365cd9112b6f4124134a568ba93df3040c62e12f74ce4cb9c8125f4",
}
TEST_EVIDENCE = "tests/test_real_hand_runtime_existing_eighth_batch.py"
RULE_FILE = "real_hand_runtime_existing_eighth_batch.json"


def _enable_super_evolve(engine, owner: int) -> None:
    player = engine.players[owner]
    player.turns_started = (
        engine.config.first_player_super_evolution_unlock_turn
        if owner == 0
        else engine.config.second_player_super_evolution_unlock_turn
    )
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.super_evolved_this_turn = False
    player.evolved_this_turn = False
    engine.state.active_player = owner


def _put_rule_hand(engine, definition, *, owner: int = 0):
    hand_card = engine._make_hand_card(
        definition,
        engine.state.allocate_entity_id(),
    )
    engine.players[owner].hand.append(hand_card)
    engine.players[owner].hand_entity_ids.append(hand_card.entity_id)
    return hand_card


class HandRuntimeEighthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 8101):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_expose_all_seven_cards_and_new_generic_operations(self):
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10302110),
            ("守护",),
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10302110),
            frozenset({"必杀"}),
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10303110),
            frozenset({"疾驰"}),
        )
        rusty = self.rulebook.operations_for(10022120, Trigger.SUPER_EVOLVE)
        self.assertEqual(
            [operation.kind for operation in rusty],
            [EffectKind.DRAW_FILTERED, EffectKind.ADD_KEYWORD],
        )
        self.assertEqual(rusty[0].deck_filter.card_id, 10022120)
        self.assertEqual(rusty[0].target_key, "drawn_rustys")
        tsubasa = self.rulebook.operations_for(10471120, Trigger.FANFARE)[0]
        self.assertIs(tsubasa.kind, EffectKind.ADD_UNION_BURST_GAUGE)
        self.assertIs(tsubasa.target, TargetKind.ALL_OWN_HAND)
        rose_attack = self.rulebook.operations_for(10223110, Trigger.ATTACK)[0]
        self.assertTrue(rose_attack.board_filter.damaged)
        self.assertEqual(
            [mode.mode_id for mode in self.rulebook.modes_for(10223110)],
            ["enhance_5"],
        )
        ruled = {
            card_id
            for card_id in CARD_IDS
            if self.rulebook.listeners_for(card_id)
            or self.rulebook.modes_for(card_id)
            or any(
                self.rulebook.operations_for(card_id, trigger)
                for trigger in Trigger
            )
        }
        self.assertEqual(ruled, set(CARD_IDS))

        with self.assertRaisesRegex(ValueError, "positive integer"):
            _parse_operation(
                {
                    "kind": "add_union_burst_gauge",
                    "target": "all_own_hand",
                    "amount": True,
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "own-hand target"):
            _parse_operation(
                {
                    "kind": "add_union_burst_gauge",
                    "target": "enemy_unit",
                    "amount": 1,
                },
                "test.json",
                1,
            )
        parsed = _parse_operation(
            {
                "kind": "buff_hand_card",
                "target": "previous_target",
                "target_key": "created",
                "amount": 3,
            },
            "test.json",
            1,
        )
        self.assertIs(parsed.target, TargetKind.PREVIOUS_TARGET)

    def test_enemy_super_evolution_grants_only_hand_listeners_and_transfers(self):
        engine = self.fresh(seed=3)
        inspiration = _put_rule_hand(engine, self.repository.get(10302110))
        dogged = _put_rule_hand(engine, self.repository.get(10303110))
        enemy = _put_unit(engine, 1, _card(981001, attack=1, life=4))
        _enable_super_evolve(engine, 1)

        engine.apply(SuperEvolve(1, enemy.entity_id))

        self.assertTrue(inspiration.has_keyword("必杀"))
        self.assertTrue(inspiration.has_keyword("守护"))
        self.assertTrue(dogged.has_keyword("疾驰"))
        self.assertTrue(dogged.has_keyword("突进"))

        engine.state.active_player = 0
        engine.players[0].mana = 10
        dogged_index = engine.players[0].hand.index(dogged)
        engine.apply(PlayCard(0, dogged_index))
        played = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10303110
        )
        self.assertTrue(played.has_keyword("疾驰"))
        self.assertTrue(played.can_attack)
        self.assertFalse(played.rush_only)
        self.assertIn(Attack(0, played.entity_id, None), engine.legal_commands())
        engine.assert_invariants()

    def test_own_super_evolution_does_not_trigger_opponent_event_listener(self):
        engine = self.fresh(seed=5)
        inspiration = _put_rule_hand(engine, self.repository.get(10302110))
        dogged = _put_rule_hand(engine, self.repository.get(10303110))
        source = _put_unit(engine, 0, _card(981010, attack=1, life=4))
        _enable_super_evolve(engine, 0)

        engine.apply(SuperEvolve(0, source.entity_id))

        self.assertFalse(inspiration.has_keyword("必杀"))
        self.assertFalse(dogged.has_keyword("疾驰"))

        dogged.add_keyword(
            "疾驰",
            duration="until_end_of_turn",
            expires_for_player=0,
        )
        self.assertTrue(dogged.has_keyword("疾驰"))
        dogged.expire_keywords("until_end_of_turn", 0)
        self.assertFalse(dogged.has_keyword("疾驰"))

    def test_rusty_draws_only_same_name_cards_and_grants_storm_to_outputs(self):
        engine = self.fresh(seed=7)
        rusty_definition = self.repository.get(10022120)
        source = _play(engine, self.repository, 10022120)
        engine.players[0].deck = [
            _card(981020),
            rusty_definition,
            _card(981021),
            rusty_definition,
        ]
        _enable_super_evolve(engine, 0)

        engine.apply(SuperEvolve(0, source.entity_id))

        drawn = [
            card for card in engine.players[0].hand
            if card.card_id == 10022120
        ]
        self.assertEqual(len(drawn), 2)
        self.assertTrue(all(card.has_keyword("疾驰") for card in drawn))
        self.assertEqual(
            [card.card_id for card in engine.players[0].deck],
            [981020, 981021],
        )
        engine.assert_invariants()

    def test_rusty_overdraw_binds_only_successful_draws(self):
        engine = self.fresh(seed=11)
        source = _play(engine, self.repository, 10022120)
        while len(engine.players[0].hand) < engine.config.max_hand - 1:
            _put_hand(engine, _card(981100 + len(engine.players[0].hand)))
        rusty_definition = self.repository.get(10022120)
        engine.players[0].deck = [rusty_definition, rusty_definition]
        _enable_super_evolve(engine, 0)

        engine.apply(SuperEvolve(0, source.entity_id))

        drawn = [
            card for card in engine.players[0].hand
            if card.card_id == 10022120
        ]
        self.assertEqual(len(drawn), 1)
        self.assertTrue(drawn[0].has_keyword("疾驰"))
        self.assertTrue(any(
            card.definition.card_id == 10022120
            and card.entry_cause == "overdraw"
            for card in engine.players[0].graveyard
        ))

    def test_orchestrated_silence_adds_rush_tokens_and_cooperation_respects_capacity(self):
        normal = self.fresh(seed=13)
        _play(normal, self.repository, 10722310)
        tokens = [
            card for card in normal.players[0].hand
            if card.card_id == 90021120
        ]
        self.assertEqual(len(tokens), 1)
        self.assertTrue(tokens[0].has_keyword("突进"))
        normal.players[0].mana = 10
        normal.apply(PlayCard(0, normal.players[0].hand.index(tokens[0])))
        summoned = next(
            unit for unit in normal.players[0].board
            if unit.definition.card_id == 90021120
        )
        self.assertTrue(summoned.has_keyword("突进"))
        self.assertTrue(summoned.rush_only)

        full = self.fresh(seed=17)
        full.players[0].cooperation = 10
        spell = _put_hand(full, self.repository.get(10722310))
        while len(full.players[0].hand) < full.config.max_hand:
            _put_hand(full, _card(981200 + len(full.players[0].hand)))
        full.apply(PlayCard(0, full.players[0].hand.index(spell)))
        self.assertEqual(len(full.players[0].hand), full.config.max_hand)
        self.assertEqual(
            sum(card.card_id == 90021120 for card in full.players[0].hand),
            1,
        )
        self.assertTrue(any(
            card.definition.card_id == 90021120
            and card.entry_cause == "hand_full"
            for card in full.players[0].graveyard
        ))
        full.assert_invariants()

    def test_add_card_binding_safely_skips_after_target_leaves_hand(self):
        engine = self.fresh(seed=19)
        source = _card(
            981300,
            name="binding source",
            card_type="法术",
            attack=None,
            life=None,
        )
        operations = (
            EffectOperation(
                EffectKind.ADD_CARD,
                TargetKind.OWN_LEADER,
                card_id=90021120,
                target_key="created",
            ),
            EffectOperation(
                EffectKind.RETURN_TO_DECK,
                TargetKind.PREVIOUS_TARGET,
                target_key="created",
            ),
            EffectOperation(
                EffectKind.ADD_KEYWORD,
                TargetKind.PREVIOUS_TARGET,
                target_key="created",
                keyword="突进",
            ),
        )

        engine._start_effects(source, None, operations, controller=0)

        self.assertFalse(any(
            card.card_id == 90021120 for card in engine.players[0].hand
        ))
        self.assertTrue(any(
            card.card_id == 90021120 for card in engine.players[0].deck
        ))
        engine.assert_invariants()

    def test_puppet_cat_condition_adds_and_buffs_exact_generated_puppet(self):
        active = self.fresh(seed=23)
        evolved = _put_unit(active, 0, _card(981400))
        evolved.evolved = True
        evolved.super_evolved = True
        evolved.super_evolved_turn = active.turn
        _play(active, self.repository, 10271120)
        puppets = [
            card for card in active.players[0].hand
            if card.card_id == 90071110
        ]
        self.assertEqual(len(puppets), 1)
        definition = self.repository.get(90071110)
        self.assertEqual(puppets[0].attack, definition.attack + 3)
        self.assertEqual(puppets[0].life, definition.life)

        inactive = self.fresh(seed=29)
        _play(inactive, self.repository, 10271120)
        self.assertFalse(any(
            card.card_id == 90071110 for card in inactive.players[0].hand
        ))

    def test_tsubasa_increases_only_defined_skybound_gauges_and_emits_event(self):
        engine = self.fresh(seed=31)
        burst = _put_hand(engine, self.repository.get(10403110))
        ordinary = _put_hand(engine, _card(981500))
        player = engine.players[0]
        player.turns_started = 4
        before = burst.union_burst_gauge(player.turns_started)

        _play(engine, self.repository, 10471120)

        self.assertEqual(burst.union_burst_gauge(player.turns_started), before + 1)
        self.assertEqual(burst.union_burst_gauge_bonus, 1)
        self.assertEqual(ordinary.union_burst_gauge_bonus, 0)
        events = [
            event for event in engine.event_history
            if event.type is EventType.UNION_BURST_GAUGE_CHANGED
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].target_id, burst.entity_id)
        self.assertEqual(events[0].metadata["gauge_after"], before + 1)
        engine.assert_invariants()

    def test_rose_enhance_draws_eligible_royal_follower_sets_zero_and_is_atomic(self):
        engine = self.fresh(seed=37)
        eligible = _card(981601, class_id=2, class_name="皇家护卫", cost=2)
        too_expensive = _card(981602, class_id=2, class_name="皇家护卫", cost=3)
        neutral = _card(981603, class_id=0, class_name="中立", cost=1)
        engine.players[0].deck = [too_expensive, neutral, eligible]
        rose = _put_hand(engine, self.repository.get(10223110))
        command = PlayCard(0, engine.players[0].hand.index(rose), "enhance_5")
        self.assertIn(command, engine.legal_commands())

        engine.apply(command)

        drawn = next(card for card in engine.players[0].hand if card.card_id == 981601)
        self.assertEqual(drawn.current_cost, 0)
        self.assertEqual(
            {card.card_id for card in engine.players[0].deck},
            {981602, 981603},
        )

        illegal = self.fresh(seed=41)
        illegal.players[0].mana = 4
        rose = _put_hand(illegal, self.repository.get(10223110))
        command = PlayCard(0, illegal.players[0].hand.index(rose), "enhance_5")
        self.assertNotIn(command, illegal.legal_commands())
        before = illegal.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            illegal.apply(command)
        self.assertEqual(illegal.deterministic_fingerprint(), before)

        no_match = self.fresh(seed=43)
        no_match.players[0].deck = [neutral, too_expensive]
        _play(no_match, self.repository, 10223110, mode_id="enhance_5")
        self.assertFalse(no_match.players[0].hand)
        no_match.assert_invariants()

    def test_rose_attack_destroys_only_a_previously_damaged_follower(self):
        active = self.fresh(seed=47)
        rose = _play(active, self.repository, 10223110)
        damaged = _put_unit(active, 1, _card(981700, attack=0, life=7))
        damaged.health = 6
        rose.can_attack = True
        rose.rush_only = False

        active.apply(Attack(0, rose.entity_id, damaged.entity_id))

        self.assertNotIn(damaged, active.players[1].board)
        self.assertIn(rose, active.players[0].board)

        undamaged_game = self.fresh(seed=53)
        rose = _play(undamaged_game, self.repository, 10223110)
        undamaged = _put_unit(
            undamaged_game,
            1,
            _card(981701, attack=0, life=7),
        )
        rose.can_attack = True
        rose.rush_only = False

        undamaged_game.apply(Attack(0, rose.entity_id, undamaged.entity_id))

        self.assertIn(undamaged, undamaged_game.players[1].board)
        self.assertEqual(undamaged.health, 4)

    def test_seeded_sequence_fingerprints_match(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=59)
            engine.players[0].cooperation = 10
            _play(engine, self.repository, 10722310)
            burst = _put_hand(engine, self.repository.get(10403110))
            engine.players[0].mana = 10
            _play(engine, self.repository, 10471120)
            self.assertEqual(burst.union_burst_gauge_bonus, 1)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_rl_mask_encodes_enhance_and_post_storm_attack_without_id_dependency(self):
        deck = [
            _card(
                982000 + index,
                class_id=2,
                class_name="皇家护卫",
            )
            for index in range(40)
        ]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=2,
            class_b=2,
            seed=61,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=61)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        rose = _put_hand(env.core, self.repository.get(10223110))
        enhance = PlayCard(0, 0, "enhance_5")
        encoded = env._encode_command(enhance)
        self.assertIsNotNone(encoded)
        self.assertTrue(env.action_mask()[encoded])

        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        dogged = _put_rule_hand(env.core, self.repository.get(10303110))
        dogged.add_keyword("疾驰")
        env.core.apply(PlayCard(0, 0))
        unit = next(
            entity for entity in env.players[0].board
            if entity.definition.card_id == 10303110
        )
        attack = Attack(0, unit.entity_id, None)
        self.assertIn(attack, env.core.legal_commands())
        attack_action = env._encode_command(attack)
        self.assertIsNotNone(attack_action)
        self.assertTrue(env.action_mask()[attack_action])
        for legal in env.core.legal_commands():
            action = env._encode_command(legal)
            self.assertIsNotNone(action)
            self.assertTrue(env.action_mask()[action])


class HandRuntimeEighthDatabaseAuditTests(unittest.TestCase):
    def test_database_snapshot_source_hashes_stats_references_and_modes(self):
        expected = {
            10302110: (10003, 0, 2, 1, 3),
            10303110: (10003, 0, 3, 3, 1),
            10022120: (10000, 2, 3, 3, 3),
            10722310: (10007, 2, 1, None, None),
            10271120: (10002, 7, 1, 1, 1),
            10471120: (10004, 7, 2, 3, 1),
            10223110: (10002, 2, 3, 3, 2),
        }
        expected_refs = {
            10302110: [],
            10303110: [],
            10022120: [10022120],
            10722310: [90021120],
            10271120: [90071110],
            10471120: [],
            10223110: [],
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            source_map = _load_source_text_map(connection)
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT card_set_id, class_id, cost, attack, life
                        FROM cards WHERE card_id=?
                        """,
                        (card_id,),
                    ).fetchone()
                    self.assertEqual(row, values)
                    self.assertEqual(
                        _source_text_sha256(source_map[card_id]),
                        SOURCE_HASHES[card_id],
                    )
                    references = [
                        row[0] for row in connection.execute(
                            "SELECT referenced_card_id FROM card_references "
                            "WHERE card_id=? ORDER BY referenced_card_id",
                            (card_id,),
                        )
                    ]
                    self.assertEqual(references, expected_refs[card_id])
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_all_cards_are_exact_with_direct_clause_and_token_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                classification = report["classifications"][str(card_id)]
                audit = classification["clause_audit"]
                self.assertEqual(classification["coverage"], "covered_exact")
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(audit["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(audit["test_evidence"], [TEST_EVIDENCE])

        token_audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        self.assertEqual(
            token_audit["summary"]["categories"]["entry_behavior_complete"],
            91,
        )
        for token_id in (90021120, 90071110):
            token = next(
                item for item in token_audit["cards"]
                if item["card_id"] == token_id
            )
            self.assertEqual(token["category"], "entry_behavior_complete")
            self.assertTrue(any(
                producer["rule_file"] == RULE_FILE
                for producer in token["authored_producers"]
            ))


if __name__ == "__main__":
    unittest.main()
