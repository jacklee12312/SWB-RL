# -*- coding: utf-8 -*-
"""Direct contracts for the eighteenth destroyed-history/bulk-zone slice."""

from __future__ import annotations

import json
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
from swb.engine.commands import Choose, EndTurn, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, ExprType, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from swb.engine.state import CostModifier, DeathCause, DeckCard
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _fresh,
    _play,
    _put_amulet,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10162130, 10663210, 10664110, 10543110, 10554120)
SOURCE_HASHES = {
    10162130: "7b50e98856d60e7d2df461f28f5cbbe269cdbd61b0d4be7eccde22053c2a34bf",
    10663210: "15f996ad251957381e779175865d5e1dd495340666ddf04343b1f860ced1f379",
    10664110: "992abad431cb71f3ffdbf9d8bb8b6276e03f09669d5692d338f033df9eeb18fc",
    10543110: "0182ef0baedb6a542b5c111f0601157a47d97e0b76d3f1645f073cd1e5be12b6",
    10554120: "63e4e328836792f56bb18aede9096d1678d51e9e2b820feda05ab5b6f3b55ec1",
}
TEST_EVIDENCE = (
    "tests/test_real_destroyed_amulet_bulk_zone_eighteenth_batch.py"
)


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option
        for option in request.options
        if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_super_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = (
        engine.config.first_player_super_evolution_unlock_turn
    )
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False


def _amulet(
    card_id: int,
    *,
    name: str,
    cost: int,
    last_words: bool = True,
):
    return _card(
        card_id,
        name=name,
        cost=cost,
        card_type="护符",
        attack=None,
        life=None,
        keywords=frozenset({"谢幕曲"} if last_words else set()),
    )


class DestroyedAmuletBulkZoneEighteenthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 18001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def record_amulet(self, engine, definition) -> None:
        engine._record_destroyed_amulet(
            0,
            definition,
            DeathCause.EFFECT_DESTROY,
        )

    def test_rule_shapes_and_new_schema_are_strict(self):
        maeve = self.rulebook.operations_for(
            10162130,
            Trigger.LAST_WORDS,
        )[0]
        self.assertIs(maeve.kind, EffectKind.SUMMON_DESTROYED_AMULETS)
        self.assertTrue(maeve.highest_base_cost_only)
        self.assertEqual(maeve.history_filter.card_type, "护符")
        self.assertIn("守护", self.rulebook.intrinsic_keywords_for(10162130))

        tome = self.rulebook.operations_for(10663210, Trigger.PLAY)
        self.assertEqual(
            [operation.kind for operation in tome],
            [
                EffectKind.SELECT_TARGETS,
                EffectKind.DESTROY,
                EffectKind.CONDITIONAL,
            ],
        )
        self.assertEqual(self.rulebook.countdown_for(10663210), 2)
        self.assertEqual(
            self.rulebook.operations_for(
                10663210,
                Trigger.LAST_WORDS,
            )[0].history_filter.keyword,
            "谢幕曲",
        )

        kandima = self.rulebook.operations_for(10664110, Trigger.FANFARE)[0]
        self.assertTrue(kandima.distinct_card_names)
        self.assertEqual(kandima.amount, 2)

        ruin = self.rulebook.operations_for(
            10543110,
            Trigger.SUPER_EVOLVE,
        )[0]
        self.assertIs(ruin.kind, EffectKind.BANISH_DECK_FILTERED)
        self.assertEqual(ruin.deck_filter.costs, (1, 3, 5, 7, 9))
        self.assertIs(
            ruin.then_operations[0].amount_expr.type,
            ExprType.DISTRIBUTED_VALUE,
        )

        shakdoh = self.rulebook.operations_for(
            10554120,
            Trigger.FANFARE,
        )[0]
        self.assertEqual(shakdoh.amount, 2)
        self.assertIs(
            shakdoh.repeat_operations[0].kind,
            EffectKind.REDRAW_HAND,
        )
        damage = shakdoh.repeat_operations[1].then_operations[0]
        self.assertIs(
            damage.target,
            TargetKind.ALL_ENEMY_UNITS_AND_LEADER,
        )

        parsed = _parse_operation(
            {
                "kind": "banish_deck_filtered",
                "target": "own_leader",
                "costs": [1, 3],
                "operations": [
                    {
                        "kind": "distribute_damage",
                        "target": "all_enemy_units",
                        "amount": {"type": "distributed_value"},
                    }
                ],
            },
            "test.json",
            1,
        )
        self.assertEqual(parsed.deck_filter.costs, (1, 3))

        invalid_operations = (
            {
                "kind": "banish_deck_filtered",
                "target": "own_leader",
                "costs": [],
                "operations": [
                    {"kind": "draw", "target": "own_leader", "amount": 1}
                ],
            },
            {
                "kind": "banish_deck_filtered",
                "target": "own_leader",
                "costs": [1, 1],
                "operations": [
                    {"kind": "draw", "target": "own_leader", "amount": 1}
                ],
            },
            {
                "kind": "summon_destroyed_amulets",
                "target": "own_leader",
                "amount": 1,
                "history_filter": {"card_type": "随从"},
            },
            {
                "kind": "draw",
                "target": "own_leader",
                "amount": 1,
                "highest_base_cost_only": True,
            },
            {
                "kind": "redraw_hand",
                "target": "own_leader",
                "amount": 1,
            },
            {
                "kind": "heal_unit",
                "target": "all_enemy_units_and_leader",
                "amount": 1,
            },
        )
        for raw in invalid_operations:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    _parse_operation(raw, "test.json", 1)

    def test_destroyed_amulet_history_is_persistent_and_auditable(self):
        engine = self.fresh(seed=3)
        amulet = _put_amulet(engine, 0, 998001)
        amulet.pending_destroy = True
        engine._stabilize()

        self.assertEqual(len(engine.state.destroyed_amulets), 1)
        record = engine.state.destroyed_amulets[0]
        self.assertEqual(record.definition.card_id, 998001)
        self.assertEqual(record.destroyed_turn, engine.turn)
        engine.players[0].graveyard.clear()
        self.assertEqual(
            [entry.definition.card_id for entry in engine.state.destroyed_amulets],
            [998001],
        )
        engine.assert_invariants()

    def test_crystallized_follower_history_preserves_amulet_form(self):
        engine = self.fresh(seed=5)
        crystal = _play(
            engine,
            self.repository,
            10661110,
            mode_id="crystallize_2",
        )
        self.assertEqual(crystal.play_mode_id, "crystallize_2")
        self.assertEqual(crystal.countdown, 3)
        crystal.countdown = 1
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))

        record = engine.state.destroyed_amulets[-1]
        self.assertEqual(record.definition.card_type, "随从")
        self.assertEqual(record.play_mode_id, "crystallize_2")
        self.assertEqual(record.summon_countdown, 3)
        engine.players[0].graveyard.clear()

        maeve = _play(engine, self.repository, 10162130)
        maeve.health = 0
        engine._stabilize()
        copied_crystal = next(
            entity
            for entity in engine.players[0].board
            if entity.definition.card_id == 10661110
            and getattr(entity, "play_mode_id", None) == "crystallize_2"
        )
        self.assertEqual(copied_crystal.countdown, 3)
        engine.assert_invariants()

    def test_maeve_uses_highest_base_cost_and_seeded_ties(self):
        choices = []
        for seed in (7, 7):
            engine = self.fresh(seed=seed)
            low = _amulet(998010, name="low", cost=1)
            high_a = _amulet(998011, name="high-a", cost=5)
            high_b = _amulet(998012, name="high-b", cost=5)
            for definition in (low, high_a, high_b):
                self.record_amulet(engine, definition)
            maeve = _play(engine, self.repository, 10162130)
            self.assertTrue(maeve.has_keyword("守护"))
            maeve.health = 0
            engine._stabilize()
            summoned = [
                entity.definition.card_id
                for entity in engine.players[0].board
                if entity.definition.card_type == "护符"
            ]
            self.assertEqual(len(summoned), 1)
            self.assertIn(summoned[0], {998011, 998012})
            choices.append(summoned[0])
            engine.assert_invariants()
        self.assertEqual(choices[0], choices[1])

        empty = self.fresh(seed=11)
        maeve = _play(empty, self.repository, 10162130)
        random_before = empty.random.getstate()
        maeve.health = 0
        empty._stabilize()
        self.assertEqual(empty.random.getstate(), random_before)
        self.assertFalse(any(
            entity.definition.card_type == "护符"
            for entity in empty.players[0].board
        ))

    def test_tome_selection_snapshot_mana_and_last_words_filters(self):
        own = self.fresh(seed=13)
        target = _put_amulet(own, 0, 998020)
        tome = _play(own, self.repository, 10663210)
        self.assertEqual(tome.countdown, 2)
        self.assertIsNotNone(own.state.pending_choice)
        _choose_entity(own, target.entity_id)
        self.assertEqual(own.players[0].mana, 8)
        self.assertNotIn(target, own.players[0].board)

        enemy = self.fresh(seed=17)
        target = _put_amulet(enemy, 1, 998021)
        _play(enemy, self.repository, 10663210)
        _choose_entity(enemy, target.entity_id)
        self.assertEqual(enemy.players[0].mana, 6)

        no_target = self.fresh(seed=19)
        source = _put_hand(no_target, self.repository.get(10663210))
        command = PlayCard(0, no_target.players[0].hand.index(source))
        self.assertIn(command, no_target.legal_commands())
        no_target.apply(command)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertEqual(
            [entity.definition.card_id for entity in no_target.players[0].board],
            [10663210],
        )

        last_words = self.fresh(seed=23)
        valid = _amulet(998022, name="valid", cost=2)
        too_large = _amulet(998023, name="large", cost=3)
        no_last_words = _amulet(
            998024,
            name="plain",
            cost=1,
            last_words=False,
        )
        for definition in (valid, too_large, no_last_words):
            self.record_amulet(last_words, definition)
        tome = _play(last_words, self.repository, 10663210)
        tome.pending_destroy = True
        last_words._stabilize()
        self.assertEqual(
            [
                entity.definition.card_id
                for entity in last_words.players[0].board
            ],
            [998022],
        )

    def test_kandima_distinct_names_capacity_and_super_evolve_branch(self):
        engine = self.fresh(seed=29)
        first = _amulet(998030, name="first", cost=1)
        first_copy = _amulet(998031, name="first", cost=1)
        second = _amulet(998032, name="second", cost=2)
        invalid = _amulet(998033, name="invalid", cost=3)
        for definition in (first, first_copy, second, invalid):
            self.record_amulet(engine, definition)
        _play(engine, self.repository, 10664110)
        summoned_names = {
            entity.definition.name
            for entity in engine.players[0].board
            if entity.definition.card_type == "护符"
        }
        self.assertEqual(summoned_names, {"first", "second"})

        full = self.fresh(seed=31)
        for index in range(4):
            _put_unit(full, 0, _card(998040 + index))
        self.record_amulet(full, first)
        source = _put_hand(full, self.repository.get(10664110))
        random_before = full.random.getstate()
        full.apply(PlayCard(0, full.players[0].hand.index(source)))
        self.assertEqual(full.random.getstate(), random_before)
        self.assertEqual(len(full.players[0].board), 5)

        evolved = self.fresh(seed=37)
        target = _put_amulet(evolved, 0, 998050)
        enemies = [
            _put_unit(evolved, 1, _card(998051 + index, life=6))
            for index in range(2)
        ]
        kandima = _play(evolved, self.repository, 10664110)
        _enable_super_evolution(evolved)
        evolved.apply(SuperEvolve(0, kandima.entity_id))
        _choose_entity(evolved, target.entity_id)
        self.assertEqual([unit.health for unit in enemies], [3, 3])

        stale = self.fresh(seed=41)
        target = _put_amulet(stale, 0, 998060)
        enemy_unit = _put_unit(stale, 1, _card(998061, life=6))
        kandima = _play(stale, self.repository, 10664110)
        _enable_super_evolution(stale)
        stale.apply(SuperEvolve(0, kandima.entity_id))
        selected_option = next(
            option.option_id
            for option in stale.state.pending_choice.options
            if option.entity_id == target.entity_id
        )
        stale.players[0].board.remove(target)
        stale.apply(Choose(0, selected_option))
        self.assertEqual(enemy_unit.health, 6)
        self.assertIsNone(stale.state.pending_choice)

    def test_ruinbringer_banishes_current_costs_and_uses_actual_count(self):
        engine = self.fresh(seed=43)
        modified_one = DeckCard(
            _card(998070, cost=2),
            [CostModifier(1, "set", 1, "permanent")],
        )
        modified_seven = DeckCard(
            _card(998071, cost=6),
            [CostModifier(2, "set", 7, "permanent")],
        )
        retained = _card(998072, cost=4)
        engine.players[0].deck = [
            modified_one,
            _card(998073, cost=3),
            retained,
            modified_seven,
            _card(998074, cost=9),
        ]
        first = _put_unit(engine, 1, _card(998075, life=2))
        second = _put_unit(engine, 1, _card(998076, life=10))
        ruin = _play(engine, self.repository, 10543110)
        _enable_super_evolution(engine)
        banish_before = sum(
            event.type is EventType.CARD_BANISHED
            for event in engine.event_history
        )
        engine.apply(SuperEvolve(0, ruin.entity_id))

        self.assertEqual(
            [card.card_id for card in engine.players[0].deck],
            [998072],
        )
        self.assertEqual(len(engine.players[0].banished), 4)
        self.assertEqual(first.health, 0)
        self.assertEqual(second.health, 8)
        self.assertEqual(
            sum(
                event.type is EventType.CARD_BANISHED
                for event in engine.event_history
            ) - banish_before,
            4,
        )

        no_match = self.fresh(seed=47)
        no_match.players[0].deck = [_card(998080, cost=2)]
        ruin = _play(no_match, self.repository, 10543110)
        _enable_super_evolution(no_match)
        random_before = no_match.random.getstate()
        no_match.apply(SuperEvolve(0, ruin.entity_id))
        self.assertEqual(no_match.random.getstate(), random_before)
        self.assertEqual(len(no_match.players[0].deck), 1)

    def test_shakdoh_redraws_twice_and_damages_all_enemies(self):
        engine = self.fresh(seed=53)
        engine.players[0].deck = [
            _card(998100 + index, cost=2)
            for index in range(20)
        ]
        for index in range(4):
            _put_hand(engine, _card(998120 + index, cost=2))
        enemy = _put_unit(engine, 1, _card(998130, life=30))
        returned_before = sum(
            event.type is EventType.CARD_RETURNED_TO_DECK
            for event in engine.event_history
        )
        drawn_before = sum(
            event.type is EventType.CARD_DRAWN
            for event in engine.event_history
        )
        shakdoh = _play(engine, self.repository, 10554120)
        self.assertEqual(engine.players[1].health, 12)
        self.assertEqual(enemy.health, 22)
        self.assertEqual(
            sum(
                event.type is EventType.CARD_RETURNED_TO_DECK
                for event in engine.event_history
            ) - returned_before,
            8,
        )
        self.assertEqual(
            sum(
                event.type is EventType.CARD_DRAWN
                for event in engine.event_history
            ) - drawn_before,
            8,
        )

        while len(engine.players[0].hand) < engine.config.max_hand:
            _put_hand(
                engine,
                _card(998131 + len(engine.players[0].hand), cost=2),
            )
        self.assertEqual(len(engine.players[0].hand), engine.config.max_hand)
        _enable_super_evolution(engine)
        engine.apply(SuperEvolve(0, shakdoh.entity_id))
        self.assertEqual(len(engine.players[0].hand), engine.config.max_hand)
        self.assertEqual(engine.players[1].health, 4)
        self.assertEqual(enemy.health, 14)

    def test_shakdoh_empty_hand_no_rng_and_seeded_replay(self):
        empty = self.fresh(seed=59)
        source = _put_hand(empty, self.repository.get(10554120))
        random_before = empty.random.getstate()
        empty.apply(PlayCard(0, empty.players[0].hand.index(source)))
        self.assertEqual(empty.random.getstate(), random_before)
        self.assertEqual(empty.players[1].health, 20)

        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=61)
            engine.players[0].deck = [
                _card(998140 + index, cost=2)
                for index in range(20)
            ]
            for index in range(4):
                _put_hand(engine, _card(998160 + index, cost=2))
            _play(engine, self.repository, 10554120)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_all_enemy_damage_is_stabilized_after_leader_damage(self):
        engine = self.fresh(seed=67)
        engine.players[0].deck = [
            _card(998180 + index, cost=2)
            for index in range(20)
        ]
        for index in range(4):
            _put_hand(engine, _card(998200 + index, cost=2))
        victim = _put_unit(engine, 1, _card(998210, life=1))
        history_start = len(engine.event_history)
        _play(engine, self.repository, 10554120)
        relevant = engine.event_history[history_start:]
        first_leader_damage = next(
            index
            for index, event in enumerate(relevant)
            if event.type is EventType.DAMAGE_APPLIED
            and event.metadata.get("target_player") == 1
        )
        victim_left = next(
            index
            for index, event in enumerate(relevant)
            if event.type is EventType.ENTITY_LEFT_PLAY
            and event.source_id == victim.entity_id
        )
        self.assertLess(first_leader_damage, victim_left)

    def test_pending_choice_action_mask_and_illegal_choice_are_atomic(self):
        deck_a = [_card(998300 + index) for index in range(40)]
        deck_b = [_card(998400 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=71,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=71)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        target = _put_amulet(env.core, 0, 998500)
        source = _put_hand(env.core, self.repository.get(10663210))
        env.invalidate_cache(reason="eighteenth batch setup")
        play = PlayCard(0, env.players[0].hand.index(source))
        self.assertTrue(env.action_mask()[env._encode_command(play)])
        env.core.apply(play)
        env.invalidate_cache(reason="eighteenth batch pending target")
        self.assertEqual(
            {
                env._decode_action(index)
                for index, allowed in enumerate(env.action_mask())
                if allowed
            },
            set(env.core.legal_commands()),
        )
        self.assertTrue(any(
            option.entity_id == target.entity_id
            for option in env.core.state.pending_choice.options
        ))

        before = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(Choose(0, "entity:999999"))
        self.assertEqual(env.core.deterministic_fingerprint(), before)


class DestroyedAmuletBulkZoneEighteenthAuditTests(unittest.TestCase):
    def test_database_multilingual_modes_references_and_raw_json(self):
        expected_stats = {
            10162130: (10001, 6, 1, 7, 5, 7),
            10663210: (10006, 6, 2, 4, None, None),
            10664110: (10006, 6, 1, 4, 3, 3),
            10543110: (10005, 4, 1, 7, 5, 5),
            10554120: (10005, 5, 1, 10, 4, 4),
        }
        expected_english = {
            10162130: "highest base cost",
            10663210: "recover 2 play points",
            10664110: "differently named allied amulets",
            10543110: "number of cards you banished",
            10554120: "Return your hand to deck",
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            source_map = _load_source_text_map(connection)
            for card_id in CARD_IDS:
                with self.subTest(card_id=card_id):
                    self.assertEqual(
                        connection.execute(
                            """
                            SELECT card_set_id, class_id, type_id, cost,
                                   attack, life
                            FROM cards WHERE card_id=?
                            """,
                            (card_id,),
                        ).fetchone(),
                        expected_stats[card_id],
                    )
                    self.assertEqual(
                        _source_text_sha256(source_map[card_id]),
                        SOURCE_HASHES[card_id],
                    )
                    texts = connection.execute(
                        """
                        SELECT text_chs, text_eng, text_jpn, text_kor, text_cht
                        FROM skill_texts WHERE card_id=? ORDER BY position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertTrue(texts)
                    self.assertTrue(all(all(row) for row in texts))
                    self.assertIn(expected_english[card_id], texts[0][1])
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
                    self.assertEqual(
                        connection.execute(
                            """
                            SELECT COUNT(*) FROM card_references
                            WHERE card_id=? AND referenced_card_id IS NOT NULL
                            """,
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
                    raw = json.loads(connection.execute(
                        "SELECT raw_json FROM cards WHERE card_id=?",
                        (card_id,),
                    ).fetchone()[0])
                    self.assertEqual(raw["card_id"], card_id)
                    self.assertEqual(raw["alt_modes"], [])

    def test_all_five_cards_are_exact_with_clause_and_token_audits(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertEqual(
            report["summary"]["coverage_counts"],
            {
                "covered_exact": 719,
                "text_unclear": 16,
                "supported_missing_rule": 0,
                "token_or_non_collectible": 91,
            },
        )
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(
                    info["clause_audit"]["status"],
                    "mapped_exact",
                )
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    [TEST_EVIDENCE],
                )

        token_audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        self.assertEqual(
            token_audit["summary"]["categories"]["entry_behavior_complete"],
            91,
        )


if __name__ == "__main__":
    unittest.main()
