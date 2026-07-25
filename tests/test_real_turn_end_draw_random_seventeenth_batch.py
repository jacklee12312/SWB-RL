# -*- coding: utf-8 -*-
"""Direct contracts for the seventeenth turn-end/draw/random-target slice."""

from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from dataclasses import replace

from scripts.report_rule_coverage import (
    _build_coverage_report,
    _load_source_text_map,
    _source_text_sha256,
)
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import (
    RuleBook,
    Trigger,
    _parse_condition,
    _parse_event_card_filter,
    _parse_operation,
)
from swb.engine.commands import Attack, EndTurn, Evolve
from swb.engine.effects import ConditionType, EffectKind, TargetKind
from swb.engine.events import EventType
from swb.engine.state import (
    AttackRestriction,
    CostModifier,
    DeckCard,
)
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10464110, 10474110, 10524110, 10553310, 10564120)
SOURCE_HASHES = {
    10464110: "a3845eeb4a47065430a5cf5545aafe008012a6ed60888955f17ad19fb5ec5198",
    10474110: "46305ea828d76444ef198553d01fadfeccfb9267e8f0c798bc828ec9e51a4b02",
    10524110: "4fa14a9c9f57e3265bdce20880d08376f8faf2f25d6ad71bf5f4666d05f9beaf",
    10553310: "eb3b70b9761e6aca2bc05fa815c599ccae639114a3b698b5c1855049f9dedeae",
    10564120: "c0f28363fa8f0ee1f0ca7199f6a0d68057d5ce82e644db0ec297c9bfa2490d69",
}
TEST_EVIDENCE = (
    "tests/test_real_turn_end_draw_random_seventeenth_batch.py"
)


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _enable_super_evolution_turn(engine) -> None:
    player = engine.players[0]
    player.turns_started = (
        engine.config.first_player_super_evolution_unlock_turn
    )


def _set_cost(card, value: int, modifier_id: int) -> None:
    card.cost_modifiers.append(CostModifier(
        modifier_id=modifier_id,
        mode="set",
        amount=value,
        duration="permanent",
    ))


class TurnEndDrawRandomSeventeenthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 17001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_schema_extensions_are_strict(self):
        galleon = self.rulebook.operations_for(
            10464110,
            Trigger.TURN_END,
        )[0]
        self.assertIs(galleon.kind, EffectKind.EVOLVE_UNIT)
        self.assertIs(galleon.target, TargetKind.RANDOM_OWN_UNIT)
        self.assertFalse(galleon.board_filter.evolved)
        self.assertFalse(galleon.board_filter.attacked_this_turn)

        lu_woh = self.rulebook.operations_for(
            10474110,
            Trigger.FANFARE,
        )
        self.assertEqual(
            [operation.kind for operation in lu_woh],
            [EffectKind.REPEAT, EffectKind.BUFF_HAND_CARD],
        )
        self.assertIs(lu_woh[1].target, TargetKind.ALL_ENEMY_HAND)
        self.assertEqual(
            self.rulebook.union_bursts_for(10474110)[0].operations[0].emblem_id,
            "lu_woh_light_personified",
        )

        oluon = self.rulebook.operations_for(
            10524110,
            Trigger.TURN_END,
        )[0]
        mixed_damage = oluon.else_operations[0].repeat_operations[0]
        self.assertIs(
            mixed_damage.target,
            TargetKind.RANDOM_ANY_UNIT_OR_LEADER,
        )
        self.assertTrue(mixed_damage.exclude_source)

        rigor = self.rulebook.emblem_def("rigor_of_the_nightblossom")
        condition = rigor.triggers[0].operations[1].conditions[0]
        self.assertIs(
            condition.type,
            ConditionType.CONTROLLER_HAND_SAME_CURRENT_COST_COUNT_AT_LEAST,
        )
        self.assertEqual(condition.value, 4)

        kukishiro = self.rulebook.emblem_def("kukishiro_mistbloom")
        self.assertEqual(
            [trigger.event_filter.current_costs for trigger in kukishiro.triggers],
            [(1, 3, 5), (2, 4, 6)],
        )

        parsed_filter = _parse_event_card_filter(
            {"current_costs": [1, 3, 5]},
            "test.json",
            1,
        )
        self.assertEqual(parsed_filter.current_costs, (1, 3, 5))
        parsed_condition = _parse_condition(
            {
                "type": "controller_hand_same_current_cost_count_at_least",
                "value": 4,
            },
            "test.json",
            1,
        )
        self.assertEqual(parsed_condition.value, 4)

    def test_current_cost_event_filter_participates_in_state_fingerprint(self):
        left = self.fresh(seed=17003)
        right = self.fresh(seed=17003)
        _play(left, self.repository, 10564120)
        _play(right, self.repository, 10564120)

        emblem = right.players[0].emblems[0]
        definition = emblem.definition
        trigger = definition.triggers[0]
        changed_filter = replace(
            trigger.event_filter,
            current_costs=(1, 3, 7),
        )
        emblem.definition = replace(
            definition,
            triggers=(
                replace(trigger, event_filter=changed_filter),
                *definition.triggers[1:],
            ),
        )

        self.assertNotEqual(
            left.deterministic_fingerprint(),
            right.deterministic_fingerprint(),
        )
        parsed_target = _parse_operation(
            {
                "kind": "damage_unit",
                "target": "random_any_unit_or_leader",
                "amount": 7,
                "exclude_source": True,
            },
            "test.json",
            1,
        )
        self.assertIs(
            parsed_target.target,
            TargetKind.RANDOM_ANY_UNIT_OR_LEADER,
        )

        invalid_filters = (
            {"current_costs": []},
            {"current_costs": [1, 1]},
            {"current_costs": "1"},
            {"current_costs": [True]},
        )
        for raw in invalid_filters:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_event_card_filter(raw, "bad.json", 1)
        with self.assertRaises(ValueError):
            _parse_operation(
                {
                    "kind": "buff_unit",
                    "target": "random_any_unit_or_leader",
                    "amount": 1,
                },
                "bad.json",
                1,
            )
        with self.assertRaises(ValueError):
            _parse_operation(
                {
                    "kind": "evolve_unit",
                    "target": "random_own_unit",
                    "target_attacked_this_turn_filter": 0,
                },
                "bad.json",
                1,
            )

    def test_galleon_cannot_attack_and_evolves_only_an_unattacked_target(self):
        engine = self.fresh(seed=17)
        eligible = _put_unit(engine, 0, _card(997001))
        attacked = _put_unit(engine, 0, _card(997002, attack=2))
        source = _play(engine, self.repository, 10464110)
        self.assertIn(
            AttackRestriction.CANNOT_ATTACK,
            {modifier.restriction for modifier in source.attack_restrictions},
        )
        self.assertFalse(any(
            isinstance(command, Attack)
            and command.attacker_id == source.entity_id
            for command in engine.legal_commands()
        ))

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        attacked.summoned_this_turn = False
        attacked.can_attack = True
        attacked.attacks_remaining = 1
        engine.apply(Attack(0, attacked.entity_id, None))
        self.assertLess(attacked.attacks_remaining, attacked.attacks_per_turn)

        _enable_super_evolution_turn(engine)
        engine.apply(EndTurn(0))
        self.assertTrue(eligible.evolved)
        self.assertFalse(attacked.evolved)
        engine.assert_invariants()

    def test_galleon_skips_without_unlock_or_eligible_target(self):
        locked = self.fresh(seed=19)
        target = _put_unit(locked, 0, _card(997010))
        source = _play(locked, self.repository, 10464110)
        source.evolved = True
        locked.apply(EndTurn(0))
        self.assertFalse(target.evolved)

        no_target = self.fresh(seed=23)
        target = _put_unit(no_target, 0, _card(997011))
        source = _play(no_target, self.repository, 10464110)
        source.evolved = True
        target.attacks_remaining = 0
        _enable_super_evolution_turn(no_target)
        no_target.apply(EndTurn(0))
        self.assertFalse(target.evolved)
        no_target.assert_invariants()

    def test_lu_woh_fanfare_repeats_damage_and_buffs_only_enemy_followers(self):
        engine = self.fresh(seed=29)
        target = _put_unit(engine, 1, _card(997020, life=10))
        follower = _put_hand(
            engine,
            _card(997021, attack=2, life=3),
            owner=1,
        )
        spell = _put_hand(
            engine,
            _card(
                997022,
                card_type="法术",
                attack=None,
                life=None,
            ),
            owner=1,
        )
        source = _play(engine, self.repository, 10474110)

        self.assertEqual(target.health, 4)
        self.assertEqual((follower.attack, follower.life), (3, 3))
        self.assertEqual(spell.current_cost, spell.definition.cost)
        self.assertFalse(engine.players[0].emblems)
        self.assertIn(source, engine.players[0].board)

    def test_lu_woh_crest_reduces_only_storm_attacks_on_a_leader(self):
        engine = self.fresh(seed=31)
        engine.players[0].turns_started = 10
        source = _play(engine, self.repository, 10474110)
        self.assertEqual(
            [emblem.emblem_id for emblem in engine.players[0].emblems],
            ["lu_woh_light_personified"],
        )
        engine.players[0].board.remove(source)
        attacker = _put_unit(
            engine,
            1,
            _card(
                997030,
                attack=5,
                life=8,
                keywords=frozenset({"疾驰"}),
            ),
        )
        engine.apply(EndTurn(0))
        engine.apply(Attack(1, attacker.entity_id, None))
        self.assertEqual(attacker.attack, 2)
        self.assertEqual(engine.players[0].health, 18)
        engine.apply(EndTurn(1))
        self.assertEqual(attacker.attack, 5)

        follower_attack = self.fresh(seed=37)
        follower_attack.players[0].turns_started = 10
        _play(follower_attack, self.repository, 10474110)
        defender = _put_unit(
            follower_attack,
            0,
            _card(997031, attack=1, life=10),
        )
        storm = _put_unit(
            follower_attack,
            1,
            _card(
                997032,
                attack=5,
                life=8,
                keywords=frozenset({"疾驰"}),
            ),
        )
        follower_attack.apply(EndTurn(0))
        follower_attack.apply(Attack(1, storm.entity_id, defender.entity_id))
        self.assertEqual(storm.attack, 5)
        follower_attack.assert_invariants()

    def test_oluon_unevolved_and_evolved_branches_are_exact_and_seeded(self):
        unevolved = self.fresh(seed=41)
        enemies = [
            _put_unit(unevolved, 1, _card(997040 + index, life=10))
            for index in range(2)
        ]
        source = _play(unevolved, self.repository, 10524110)
        unevolved.apply(EndTurn(0))
        self.assertEqual([unit.health for unit in enemies], [3, 3])
        self.assertEqual(source.health, source.max_health)

        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=43)
            ally = _put_unit(engine, 0, _card(997050, life=30))
            enemy = _put_unit(engine, 1, _card(997051, life=30))
            source = _play(engine, self.repository, 10524110)
            _enable_evolution(engine)
            engine.apply(Evolve(0, source.entity_id))
            source_health = source.health
            engine.apply(EndTurn(0))
            self.assertEqual(source.health, source_health)
            total_damage = (
                20 - engine.players[0].health
                + 20 - engine.players[1].health
                + 30 - ally.health
                + 30 - enemy.health
            )
            self.assertEqual(total_damage, 21)
            outcomes.append(engine.deterministic_fingerprint())
        self.assertEqual(outcomes[0], outcomes[1])

    def test_rigor_checks_current_cost_after_draw_and_grants_ward(self):
        engine = self.fresh(seed=47)
        _play(engine, self.repository, 10553310)
        for index, base_cost in enumerate((1, 3, 5)):
            hand_card = _put_hand(
                engine,
                _card(997060 + index, cost=base_cost),
            )
            _set_cost(hand_card, 2, index + 1)
        drawn = DeckCard(_card(997063, cost=8))
        _set_cost(drawn, 2, 10)
        engine.players[0].deck = [drawn]
        engine.apply(EndTurn(0))

        skeleton = next(
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == 90051110
        )
        self.assertTrue(skeleton.has_keyword("守护"))
        self.assertEqual(
            [card.current_cost for card in engine.players[0].hand],
            [2, 2, 2, 2],
        )
        engine.assert_invariants()

    def test_rigor_no_match_and_full_board_paths_do_not_misgrant(self):
        no_match = self.fresh(seed=53)
        _play(no_match, self.repository, 10553310)
        for index, cost in enumerate((1, 1, 2)):
            _put_hand(no_match, _card(997070 + index, cost=cost))
        no_match.players[0].deck = [_card(997073, cost=3)]
        no_match.apply(EndTurn(0))
        self.assertFalse(any(
            unit.definition.card_id == 90051110
            for unit in no_match.players[0].board
        ))

        full = self.fresh(seed=59)
        _play(full, self.repository, 10553310)
        for index in range(5):
            _put_unit(full, 0, _card(997080 + index))
        for index in range(3):
            _put_hand(full, _card(997090 + index, cost=2))
        full.players[0].deck = [_card(997093, cost=2)]
        full.apply(EndTurn(0))
        self.assertEqual(len(full.players[0].board), 5)
        self.assertFalse(any(
            unit.definition.card_id == 90051110
            for unit in full.players[0].board
        ))
        full.assert_invariants()

    def test_kukishiro_uses_drawn_current_cost_and_both_board_owners(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=61)
            odd = DeckCard(_card(997100, cost=8))
            even = DeckCard(_card(997101, cost=9))
            _set_cost(odd, 1, 1)
            _set_cost(even, 2, 2)
            engine.players[0].deck = [even, odd]
            source = _play(engine, self.repository, 10564120)

            own_tokens = [
                unit.definition.card_id
                for unit in engine.players[0].board
                if unit is not source
            ]
            enemy_tokens = [
                unit.definition.card_id
                for unit in engine.players[1].board
            ]
            self.assertEqual(len(own_tokens), 1)
            self.assertEqual(len(enemy_tokens), 1)
            self.assertIn(own_tokens[0], {10061120, 90061110})
            self.assertIn(enemy_tokens[0], {10061120, 90061110})
            self.assertEqual(
                [card.current_cost for card in engine.players[0].hand],
                [1, 2],
            )
            outcomes.append((
                tuple(own_tokens),
                tuple(enemy_tokens),
                engine.deterministic_fingerprint(),
            ))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_kukishiro_turn_gate_capacity_and_source_departure(self):
        engine = self.fresh(seed=67)
        engine.players[0].deck = [
            _card(997110, cost=8),
            _card(997111, cost=8),
        ]
        source = _play(engine, self.repository, 10564120)
        engine.players[0].board.remove(source)
        before = (
            len(engine.players[0].board),
            len(engine.players[1].board),
        )
        engine.apply(EndTurn(0))
        engine.players[0].deck = [_card(997112, cost=1)]
        engine._draw(0, reason="opponent-turn test draw")
        engine._resolve_event_queue()
        self.assertEqual(
            (
                len(engine.players[0].board),
                len(engine.players[1].board),
            ),
            before,
        )

        engine.players[0].deck = [_card(997114, cost=8)]
        engine.apply(EndTurn(1))
        engine.players[0].deck = [_card(997113, cost=1)]
        engine._draw(0, reason="owner-turn test draw")
        engine._resolve_event_queue()
        self.assertEqual(len(engine.players[0].board), before[0] + 1)

        full = self.fresh(seed=71)
        for index in range(4):
            _put_unit(full, 0, _card(997120 + index))
        full.players[0].deck = [
            _card(997130, cost=8),
            _card(997131, cost=1),
        ]
        _play(full, self.repository, 10564120)
        self.assertEqual(len(full.players[0].board), 5)
        self.assertFalse(any(
            unit.definition.card_id in {10061120, 90061110}
            for unit in full.players[0].board
        ))
        full.assert_invariants()

    def test_random_and_draw_events_are_explicit(self):
        engine = self.fresh(seed=73)
        engine.players[0].deck = [
            _card(997140, cost=2),
            _card(997141, cost=1),
        ]
        drawn_before = sum(
            event.type is EventType.CARD_DRAWN
            for event in engine.event_history
        )
        random_before = sum(
            event.type is EventType.RANDOM_CHOICES_SELECTED
            for event in engine.event_history
        )
        _play(engine, self.repository, 10564120)
        self.assertEqual(
            sum(
                event.type is EventType.CARD_DRAWN
                for event in engine.event_history
            ) - drawn_before,
            2,
        )
        self.assertEqual(
            sum(
                event.type is EventType.RANDOM_CHOICES_SELECTED
                for event in engine.event_history
            ) - random_before,
            2,
        )


class TurnEndDrawRandomSeventeenthAuditTests(unittest.TestCase):
    def test_database_multilingual_modes_references_and_raw_json(self):
        expected_stats = {
            10464110: (10004, 6, 3, 5, 5),
            10474110: (10004, 7, 5, 6, 6),
            10524110: (10005, 2, 9, 7, 7),
            10553310: (10005, 5, 2, None, None),
            10564120: (10005, 6, 7, 5, 5),
        }
        expected_english = {
            10464110: "didn't attack this turn",
            10474110: "Do this 6 times",
            10524110: "At the end of your turn",
            10553310: "Crest: Rigor of the Nightblossom",
            10564120: "Return 2 random cards",
        }
        expected_modes = {
            10464110: 0,
            10474110: 1,
            10524110: 0,
            10553310: 1,
            10564120: 1,
        }
        expected_reference_ids = {
            10464110: [],
            10474110: [],
            10524110: [],
            10553310: [90051110],
            10564120: [10061120, 90061110],
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            source_map = _load_source_text_map(connection)
            for card_id in CARD_IDS:
                with self.subTest(card_id=card_id):
                    self.assertEqual(
                        connection.execute(
                            """
                            SELECT card_set_id, class_id, cost, attack, life
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
                        expected_modes[card_id],
                    )
                    self.assertEqual(
                        [
                            row[0]
                            for row in connection.execute(
                                """
                                SELECT referenced_card_id
                                FROM card_references
                                WHERE card_id=?
                                  AND referenced_card_id IS NOT NULL
                                ORDER BY position
                                """,
                                (card_id,),
                            ).fetchall()
                        ],
                        expected_reference_ids[card_id],
                    )
                    raw = json.loads(connection.execute(
                        "SELECT raw_json FROM cards WHERE card_id=?",
                        (card_id,),
                    ).fetchone()[0])
                    self.assertEqual(raw["card_id"], card_id)
                    self.assertEqual(
                        len(raw["alt_modes"]),
                        expected_modes[card_id],
                    )

    def test_all_five_cards_are_exact_with_clause_and_token_audits(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertEqual(
            report["summary"]["coverage_counts"],
            {
                "covered_exact": 703,
                "text_unclear": 16,
                "supported_missing_rule": 16,
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
