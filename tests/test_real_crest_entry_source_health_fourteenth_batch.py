# -*- coding: utf-8 -*-
"""Direct contracts for the fourteenth crest/entry/source-health slice."""

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
    _parse_expression,
    _parse_operation,
)
from swb.engine.commands import (
    ActivateAmulet,
    Attack,
    Choose,
    EndTurn,
    PlayCard,
    SuperEvolve,
)
from swb.engine.conditions import evaluate_condition, evaluate_expression
from swb.engine.effects import ConditionType, EffectKind, ExprType, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from tests.test_real_attack_history_emblem_countdown_token_batch import (
    _ready_attacker,
)
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _destroy_units,
    _fresh,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10242210, 10344110, 10534110, 10844110, 10864110)
SOURCE_HASHES = {
    10242210: "a29f6ba92a259137886dc95b8869ccc771c4d4fc193d872e8a130fffd5d5165f",
    10344110: "dd82e223c162ae0cbbbb4fcddd3032fedeef5ce7dcb9906e7cee226ac52b5187",
    10534110: "989edeb1ce2ffcae47975a22410f07d96c8261e87d69f33a1abdcb998a3c7512",
    10844110: "99bb0d437e5d85d3af0501739cc48cf5d59ecfa0e12dec566013bbf6d019e271",
    10864110: "45d98b4b499088b3c1ae17d5ebb17e514dde1aa6469800b0cb2daca01d1ed6bc",
}
TEST_EVIDENCE = "tests/test_real_crest_entry_source_health_fourteenth_batch.py"


def _play(engine, repository: CardRepository, card_id: int):
    hand_card = _put_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, engine.players[0].hand.index(hand_card)))
    return next(
        (
            entity
            for entity in engine.players[0].board
            if entity.definition.card_id == card_id
        ),
        None,
    )


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options if option.entity_id == entity_id
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
    engine.state.active_player = 0


class CrestEntrySourceHealthFourteenthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 14001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_new_generic_schema_are_strict(self):
        blade = self.rulebook.operations_for(10242210, Trigger.ACTIVATE)
        self.assertEqual(
            [operation.kind for operation in blade],
            [
                EffectKind.DESTROY,
                EffectKind.BUFF_UNIT,
                EffectKind.GRANT_LAST_WORDS,
            ],
        )
        self.assertEqual(self.rulebook.activation_for(10242210).cost, 1)
        self.assertTrue(blade[1].requires_target)
        self.assertEqual(
            blade[2].granted_operations[0].card_id,
            10242210,
        )

        angela = self.rulebook.operations_for(10344110, Trigger.SUPER_EVOLVE)
        self.assertIs(
            angela[0].amount_expr.type,
            ExprType.SOURCE_MISSING_HEALTH,
        )
        self.assertEqual(angela[1].amount, 3)

        lhynkal_emblem = self.rulebook.emblem_def("lhynkal_wandering_fool")
        self.assertIs(
            lhynkal_emblem.triggers[0].operations[0].kind,
            EffectKind.CHANGE_LEADER_MAX_HEALTH,
        )
        self.assertEqual(lhynkal_emblem.triggers[0].operations[0].amount, -2)

        drache = self.rulebook.operations_for(10844110, Trigger.FANFARE)
        self.assertIs(
            drache[0].amount_expr.type,
            ExprType.SUBTRACT,
        )
        self.assertIs(
            drache[0].amount_expr.values[0].type,
            ExprType.CONTROLLER_ENTERED_FOLLOWER_COUNT,
        )
        self.assertIs(
            drache[1].conditions[0].type,
            ConditionType.CONTROLLER_ENTERED_FOLLOWER_COUNT_AT_LEAST,
        )

        sisters = self.rulebook.operations_for(10864110, Trigger.FANFARE)
        self.assertEqual(
            [operation.kind for operation in sisters],
            [EffectKind.SUMMON_FROM_DECK, EffectKind.SUPER_EVOLVE_UNIT],
        )
        self.assertEqual(sisters[0].deck_filter.cost_max, 2)

        parsed = _parse_operation(
            {
                "kind": "change_leader_max_health",
                "target": "enemy_leader",
                "amount": -2,
            },
            "test.json",
            1,
        )
        self.assertEqual(parsed.amount, -2)
        for raw in (
            {
                "kind": "change_leader_max_health",
                "target": "enemy_unit",
                "amount": -2,
            },
            {
                "kind": "change_leader_max_health",
                "target": "enemy_leader",
                "amount": 0,
            },
            {
                "kind": "change_leader_max_health",
                "target": "enemy_leader",
                "amount": True,
            },
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test.json", 1)

    def test_entry_count_and_missing_health_expressions(self):
        count_expr = _parse_expression(
            {
                "type": "controller_entered_follower_count",
                "filter": {"card_id": 10844110},
            },
            "test",
            1,
        )
        threshold = _parse_condition(
            {
                "type": "controller_entered_follower_count_at_least",
                "value": 2,
                "filter": {"card_id": 10844110},
            },
            "test",
            1,
        )
        missing = _parse_expression(
            {"type": "source_missing_health"},
            "test",
            1,
        )
        engine = self.fresh(seed=3)
        first = engine._summon_follower_to_board(
            0,
            self.repository.get(10844110),
            summon_cause="test",
        )
        second = engine._summon_follower_to_board(
            0,
            self.repository.get(10844110),
            summon_cause="test",
        )
        first.health -= 3
        context = engine._eval_context(0, source_entity_id=first.entity_id)
        self.assertEqual(evaluate_expression(count_expr, context), 2)
        self.assertTrue(evaluate_condition(threshold, context))
        self.assertEqual(evaluate_expression(missing, context), 3)
        engine.players[0].board.remove(first)
        engine.players[0].board.remove(second)
        self.assertEqual(evaluate_expression(count_expr, engine._eval_context(0)), 2)

    def test_pyrewyrm_blade_activation_target_last_words_and_capacity(self):
        engine = self.fresh(seed=5)
        ally = _put_unit(engine, 0, _card(991401, attack=2, life=3))
        source = _play(engine, self.repository, 10242210)
        mana_before = engine.players[0].mana
        command = ActivateAmulet(0, source.entity_id)
        self.assertIn(command, engine.legal_commands())
        engine.apply(command)
        self.assertNotIn(source, engine.players[0].board)
        self.assertEqual(engine.players[0].mana, mana_before - 1)
        _choose_entity(engine, ally.entity_id)
        self.assertEqual((ally.attack, ally.health, ally.max_health), (3, 4, 4))
        _destroy_units(engine, ally)
        replacement = next(
            entity
            for entity in engine.players[0].board
            if entity.definition.card_id == 10242210
        )
        self.assertIsNotNone(replacement)

        limited = self.fresh(seed=7)
        ally = _put_unit(limited, 0, _card(991402))
        source = _play(limited, self.repository, 10242210)
        for index in range(3):
            _put_unit(limited, 0, _card(991410 + index))
        limited.apply(ActivateAmulet(0, source.entity_id))
        _choose_entity(limited, ally.entity_id)
        _destroy_units(limited, ally)
        self.assertEqual(len(limited.players[0].board), 4)
        self.assertTrue(any(
            entity.definition.card_id == 10242210
            for entity in limited.players[0].board
        ))
        limited.assert_invariants()

    def test_pyrewyrm_no_target_and_illegal_choice_are_atomic_with_rl_mask(self):
        no_target = self.fresh(seed=11)
        source = _play(no_target, self.repository, 10242210)
        command = ActivateAmulet(0, source.entity_id)
        before = no_target.deterministic_fingerprint()
        self.assertNotIn(command, no_target.legal_commands())
        with self.assertRaises(IllegalCommand):
            no_target.apply(command)
        self.assertEqual(no_target.deterministic_fingerprint(), before)

        deck = [
            _card(
                991500 + index,
                class_id=4,
                class_name="龙族",
            )
            for index in range(40)
        ]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=4,
            class_b=4,
            seed=13,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=13)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        ally = _put_unit(env.core, 0, _card(991550))
        _put_hand(env.core, self.repository.get(10242210))
        env.players[0].max_mana = 10
        env.players[0].mana = 10
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        env.core.apply(PlayCard(0, 0))
        source = next(
            entity
            for entity in env.players[0].board
            if entity.definition.card_id == 10242210
        )
        env.invalidate_cache(reason="blade played")
        self.assertTrue(env.action_mask()[env._encode_command(
            ActivateAmulet(0, source.entity_id)
        )])
        env.core.apply(ActivateAmulet(0, source.entity_id))
        env.invalidate_cache(reason="blade choice")
        self.assertEqual(
            {
                env._decode_action(index)
                for index, allowed in enumerate(env.action_mask())
                if allowed
            },
            set(env.core.legal_commands()),
        )
        before = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(Choose(0, "entity:999999"))
        self.assertEqual(env.core.deterministic_fingerprint(), before)
        _choose_entity(env.core, ally.entity_id)

    def test_angela_repeats_sequential_damage_and_self_listener(self):
        engine = self.fresh(seed=17)
        own = _put_unit(engine, 0, _card(991601, life=5))
        enemy = _put_unit(engine, 1, _card(991602, life=5))
        source = _play(engine, self.repository, 10344110)
        self.assertEqual(source.health, 1)
        self.assertNotIn(own, engine.players[0].board)
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertEqual(engine.players[1].health, 17)
        survived = [
            event
            for event in engine.event_history
            if event.type is EventType.FOLLOWER_DAMAGED_SURVIVED
            and event.source_id == source.entity_id
        ]
        self.assertEqual(len(survived), 3)
        engine.assert_invariants()

        leaves_mid_repeat = self.fresh(seed=18)
        weakened = replace(self.repository.get(10344110), life=5)
        hand_card = _put_hand(leaves_mid_repeat, weakened)
        leaves_mid_repeat.apply(PlayCard(0, 0))
        self.assertFalse(any(
            unit.definition.card_id == 10344110
            for unit in leaves_mid_repeat.players[0].board
        ))
        self.assertEqual(leaves_mid_repeat.players[1].health, 18)
        self.assertNotIn(hand_card, leaves_mid_repeat.players[0].hand)
        leaves_mid_repeat.assert_invariants()

    def test_angela_super_evolve_heals_to_full_then_repeats(self):
        engine = self.fresh(seed=19)
        source = _put_unit(engine, 0, self.repository.get(10344110))
        source.health = 2
        _enable_super_evolution(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertTrue(source.super_evolved)
        self.assertEqual(source.health, source.max_health)
        self.assertEqual(engine.players[1].health, 20)
        self.assertEqual(
            sum(
                event.type is EventType.DAMAGE_PREVENTED
                and event.target_id == source.entity_id
                for event in engine.event_history
            ),
            3,
        )

        opposing_turn = self.fresh(seed=23)
        source = _put_unit(opposing_turn, 0, self.repository.get(10344110))
        opposing_turn.state.active_player = 1
        attacker = _ready_attacker(opposing_turn, 1, 991650)
        opposing_turn.apply(Attack(1, attacker.entity_id, source.entity_id))
        self.assertEqual(opposing_turn.players[1].health, 20)

    def test_lhynkal_emblems_stack_reduce_and_clamp_max_health(self):
        engine = self.fresh(seed=29)
        first = _play(engine, self.repository, 10534110)
        self.assertTrue(first.has_keyword("突进"))
        self.assertEqual(engine.players[1].max_health, 20)
        _destroy_units(engine, first)
        _play(engine, self.repository, 10534110)
        self.assertEqual(engine.players[1].max_health, 18)
        _destroy_units(engine, *list(engine.players[0].board))
        _play(engine, self.repository, 10534110)
        self.assertEqual(engine.players[1].max_health, 14)

        clamped = self.fresh(seed=31)
        definition = self.rulebook.emblem_def("lhynkal_wandering_fool")
        for _ in range(10):
            clamped._add_emblem_to_player(0, definition, 10534110)
        clamped.players[1].health = 20
        _play(clamped, self.repository, 10534110)
        self.assertEqual(
            (clamped.players[1].max_health, clamped.players[1].health),
            (1, 1),
        )
        changed = [
            event
            for event in clamped.event_history
            if event.type is EventType.LEADER_MAX_HEALTH_CHANGED
        ]
        self.assertEqual(sum(event.amount for event in changed), -19)
        self.assertEqual(changed[-1].metadata["applied_amount"], -1)
        clamped.assert_invariants()

    def test_lhynkal_super_evolve_adds_ten_seeded_copies(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=37)
            source = _put_unit(engine, 0, self.repository.get(10534110))
            before = len(engine.players[0].deck)
            _enable_super_evolution(engine)
            engine.apply(SuperEvolve(0, source.entity_id))
            self.assertEqual(len(engine.players[0].deck), before + 10)
            self.assertEqual(
                sum(
                    card.card_id == 10534110
                    for card in engine.players[0].deck
                ),
                10,
            )
            self.assertEqual(
                sum(
                    event.type is EventType.CARD_ADDED_TO_DECK
                    for event in engine.event_history
                ),
                10,
            )
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_drache_counts_other_entries_evolves_and_keeps_history(self):
        engine = self.fresh(seed=41)
        first = engine._summon_follower_to_board(
            0,
            self.repository.get(10844110),
            summon_cause="test",
        )
        second = engine._summon_follower_to_board(
            0,
            self.repository.get(10844110),
            summon_cause="test",
        )
        engine.players[0].board.remove(first)
        engine.players[0].board.remove(second)
        source = _play(engine, self.repository, 10844110)
        self.assertTrue(source.evolved)
        self.assertFalse(source.super_evolved)
        self.assertGreaterEqual(source.attack, 6)
        self.assertEqual(
            sum(
                record.definition.card_id == 10844110
                for record in engine.state.follower_entries
            ),
            3,
        )

        below = self.fresh(seed=43)
        previous = below._summon_follower_to_board(
            0,
            self.repository.get(10844110),
            summon_cause="test",
        )
        below.players[0].board.remove(previous)
        source = _play(below, self.repository, 10844110)
        self.assertFalse(source.evolved)
        self.assertEqual(source.attack, 5)

    def test_drache_last_words_countdown_adds_cost_two_and_full_hand_burns(self):
        engine = self.fresh(seed=47)
        source = _play(engine, self.repository, 10844110)
        _destroy_units(engine, source)
        emblem = engine.players[0].emblems[0]
        self.assertEqual(emblem.countdown, 2)
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        added = next(
            card
            for card in engine.players[0].hand
            if card.card_id == 10844110
        )
        self.assertEqual(added.current_cost, 2)
        self.assertFalse(engine.players[0].emblems)

        full = self.fresh(seed=53)
        definition = self.rulebook.emblem_def("drache_aluzard_burning_blood")
        full._add_emblem_to_player(0, definition, 10844110)
        for index in range(full.config.max_hand):
            _put_hand(full, _card(991700 + index))
        full.apply(EndTurn(0))
        full.apply(EndTurn(1))
        full.apply(EndTurn(0))
        full.apply(EndTurn(1))
        self.assertEqual(len(full.players[0].hand), full.config.max_hand)
        self.assertFalse(any(
            card.card_id == 10844110 for card in full.players[0].hand
        ))

    def test_sisters_seeded_deck_summon_super_evolves_and_handles_no_slot(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=59)
            eligible = _card(991801, cost=2, attack=2, life=3)
            engine.players[0].deck = [
                _card(991802, cost=3),
                _card(
                    991803,
                    cost=1,
                    card_type="法术",
                    attack=None,
                    life=None,
                ),
                eligible,
            ]
            sep_before = engine.players[0].super_evolution_points
            _play(engine, self.repository, 10864110)
            summoned = next(
                unit
                for unit in engine.players[0].board
                if unit.definition.card_id == eligible.card_id
            )
            self.assertTrue(summoned.super_evolved)
            self.assertEqual(
                engine.players[0].super_evolution_points,
                sep_before,
            )
            self.assertNotIn(eligible, engine.players[0].deck)
            outcomes.append(engine.deterministic_fingerprint())
        self.assertEqual(outcomes[0], outcomes[1])

        full = self.fresh(seed=61)
        for index in range(4):
            _put_unit(full, 0, _card(991820 + index))
        full.players[0].deck = [_card(991830, cost=1)]
        rng_before = full.random.getstate()
        _play(full, self.repository, 10864110)
        self.assertEqual(len(full.players[0].board), full.config.max_board)
        self.assertEqual(full.random.getstate(), rng_before)
        self.assertEqual(len(full.players[0].deck), 1)

        no_candidate = self.fresh(seed=63)
        no_candidate.players[0].deck = [
            _card(991840, cost=3),
            _card(
                991841,
                cost=1,
                card_type="法术",
                attack=None,
                life=None,
            ),
        ]
        rng_before = no_candidate.random.getstate()
        _play(no_candidate, self.repository, 10864110)
        self.assertEqual(no_candidate.random.getstate(), rng_before)
        self.assertEqual(len(no_candidate.players[0].deck), 2)

    def test_sisters_emblem_only_grants_second_attack_for_super_follower_combat(self):
        engine = self.fresh(seed=67)
        definition = self.rulebook.emblem_def("verdilia_castelle_sisters")
        engine._add_emblem_to_player(0, definition, 10864110)
        attacker = _ready_attacker(engine, 0, 991901)
        attacker.evolved = True
        attacker.super_evolved = True
        attacker.super_evolved_turn = engine.turn
        target = _put_unit(engine, 1, _card(991902, attack=0, life=10))
        engine.apply(Attack(0, attacker.entity_id, target.entity_id))
        self.assertEqual(attacker.attacks_per_turn, 2)
        self.assertEqual(attacker.attacks_remaining, 1)
        second = Attack(0, attacker.entity_id, target.entity_id)
        self.assertIn(second, engine.legal_commands())
        engine.apply(second)
        self.assertEqual(attacker.attacks_remaining, 0)
        engine.apply(EndTurn(0))
        self.assertEqual(attacker.attacks_per_turn, 1)

        leader = self.fresh(seed=71)
        leader._add_emblem_to_player(0, definition, 10864110)
        attacker = _ready_attacker(leader, 0, 991903)
        attacker.evolved = True
        attacker.super_evolved = True
        attacker.super_evolved_turn = leader.turn
        leader.apply(Attack(0, attacker.entity_id, None))
        self.assertEqual(attacker.attacks_per_turn, 1)

        ordinary = self.fresh(seed=73)
        ordinary._add_emblem_to_player(0, definition, 10864110)
        attacker = _ready_attacker(ordinary, 0, 991904)
        target = _put_unit(ordinary, 1, _card(991905, attack=0, life=10))
        ordinary.apply(Attack(0, attacker.entity_id, target.entity_id))
        self.assertEqual(attacker.attacks_per_turn, 1)

        deck = [_card(991950 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=79,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=79)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.core._add_emblem_to_player(0, definition, 10864110)
        attacker = _ready_attacker(env.core, 0, 991999)
        attacker.evolved = True
        attacker.super_evolved = True
        attacker.super_evolved_turn = env.core.turn
        target = _put_unit(env.core, 1, _card(992000, attack=0, life=10))
        first = Attack(0, attacker.entity_id, target.entity_id)
        env.core.apply(first)
        env.invalidate_cache(reason="sisters first attack")
        second = Attack(0, attacker.entity_id, target.entity_id)
        self.assertTrue(env.action_mask()[env._encode_command(second)])
        self.assertEqual(
            {
                env._decode_action(index)
                for index, allowed in enumerate(env.action_mask())
                if allowed
            },
            set(env.core.legal_commands()),
        )


class CrestEntrySourceHealthFourteenthAuditTests(unittest.TestCase):
    def test_database_multilingual_text_alt_modes_references_and_raw_json(self):
        expected_stats = {
            10242210: (10002, 4, 1, None, None),
            10344110: (10003, 4, 9, 5, 7),
            10534110: (10005, 3, 1, 1, 1),
            10844110: (10008, 4, 4, 4, 4),
            10864110: (10008, 6, 7, 4, 4),
        }
        expected_english = {
            10242210: "Destroy this card",
            10344110: "Do this 3 times",
            10534110: "Add 10 copies",
            10844110: "number of other allied copies",
            10864110: "super-evolve it",
        }
        expected_modes = {
            10242210: 0,
            10344110: 0,
            10534110: 1,
            10844110: 1,
            10864110: 1,
        }
        expected_references = {
            10242210: [(10242210, "炎龙之剑")],
            10344110: [],
            10534110: [(10534110, "漫步的《愚者》·琳库露")],
            10844110: [(10844110, "反照的赤红·德莱克&亚瑞札特")],
            10864110: [],
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
                        connection.execute(
                            """
                            SELECT referenced_card_id, referenced_name
                            FROM card_references
                            WHERE card_id=? AND referenced_card_id IS NOT NULL
                            ORDER BY position
                            """,
                            (card_id,),
                        ).fetchall(),
                        expected_references[card_id],
                    )
                    raw = json.loads(connection.execute(
                        "SELECT raw_json FROM cards WHERE card_id=?",
                        (card_id,),
                    ).fetchone()[0])
                    self.assertEqual(raw["card_id"], card_id)
                    self.assertEqual(len(raw["alt_modes"]), expected_modes[card_id])

    def test_all_five_cards_are_exact_with_clause_and_token_audits(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertEqual(
            report["summary"]["coverage_counts"],
            {
                "covered_exact": 698,
                "text_unclear": 16,
                "supported_missing_rule": 21,
                "token_or_non_collectible": 91,
            },
        )
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
                    [TEST_EVIDENCE],
                )

        token_audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        self.assertEqual(
            token_audit["summary"]["categories"]["entry_behavior_complete"],
            91,
        )


if __name__ == "__main__":
    unittest.main()
