# -*- coding: utf-8 -*-
"""Direct contracts for the twelfth selected-hand/granted-ability slice."""

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
from swb.engine.commands import Choose, Evolve, PlayCard
from swb.engine.effects import (
    EffectKind,
    EffectOperation,
    ExprType,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.observation_v2 import _board_runtime, _hand_runtime
from swb.engine.resolution import IllegalCommand
from swb.engine.state import StatModifier
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _destroy_units,
    _fresh,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10111140,
    10272310,
    10273110,
    10412110,
    10473110,
)
SOURCE_HASHES = {
    10111140: "c03ca725305eb5c8d2a2ce7086033b86d3f15b9e35deb87faa7399065a0f08f2",
    10272310: "ff0121e2f8c1fa4cc427e43462f268dd52c551dfbc68330a78d44055d7d1c7ad",
    10273110: "53275e87305f12b81af4756f2e1b673d68f5c85d2d3ea8c0341d10d83e5c2edf",
    10412110: "024a354d4e084ba23ecbdfbaf5eaebcb1d5604cead9bf90a80e9ed66d28d30c7",
    10473110: "64da489805e1e2d54d1cde067b0349b69beb9f92b100cfda8bead30077688503",
}
TEST_EVIDENCE = "tests/test_real_selected_hand_grants_twelfth_batch.py"


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options
        if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _remove_hand_entity(engine, entity_id: int, *, owner: int = 0) -> None:
    player = engine.players[owner]
    index = player.hand_entity_ids.index(entity_id)
    card = player.hand.pop(index)
    player.hand_entity_ids.pop(index)
    player.banished.append(card.definition)


def _enable_evolve(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False
    engine.state.active_player = 0


class SelectedHandGrantsTwelfthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 12001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_strict_generic_schema(self):
        grasshopper = self.rulebook.operations_for(
            10111140,
            Trigger.FANFARE,
        )[0]
        self.assertIs(grasshopper.kind, EffectKind.DRAW_FILTERED)
        self.assertEqual(grasshopper.deck_filter.card_type, "随从")
        self.assertIs(
            grasshopper.deck_filter_cost_expr.type,
            ExprType.CONTROLLER_COMBO,
        )

        flight = self.rulebook.operations_for(10272310, Trigger.PLAY)
        self.assertEqual(
            [operation.kind for operation in flight],
            [
                EffectKind.SELECT_TARGETS,
                EffectKind.ADD_KEYWORD,
                EffectKind.GRANT_LAST_WORDS,
            ],
        )
        self.assertTrue(flight[0].requires_target)
        self.assertEqual(
            (
                flight[0].hand_filter.card_type,
                flight[0].hand_filter.tribe_name,
            ),
            ("随从", "创造物"),
        )
        self.assertEqual(
            [operation.kind for operation in flight[2].granted_operations],
            [EffectKind.DRAW],
        )

        carnelia = self.rulebook.operations_for(10273110, Trigger.EVOLVE)
        self.assertEqual(
            [operation.kind for operation in carnelia],
            [
                EffectKind.SELECT_TARGETS,
                EffectKind.ADD_KEYWORD,
                EffectKind.GRANT_EFFECT_DESTROY_IMMUNITY,
            ],
        )
        chloe = self.rulebook.modes_for(10412110)[0]
        self.assertEqual((chloe.mode_id, chloe.cost), ("enhance_8", 8))
        self.assertEqual(
            [operation.kind for operation in chloe.operations],
            [EffectKind.SUMMON_FROM_HAND, EffectKind.RETURN_TO_HAND],
        )
        cassius = self.rulebook.operations_for(10473110, Trigger.FANFARE)
        self.assertIs(cassius[1].amount_expr.type, ExprType.BOUND_CARD_ATTACK)
        self.assertEqual(
            cassius[1].amount_expr.binding_key,
            "cassius_artifact",
        )

        with self.assertRaisesRegex(ValueError, "only valid"):
            _parse_operation(
                {
                    "kind": "draw",
                    "target": "own_leader",
                    "cost_equals": {"type": "controller_combo"},
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _parse_operation(
                {
                    "kind": "draw_filtered",
                    "target": "own_leader",
                    "cost_min": 1,
                    "cost_equals": {"type": "controller_combo"},
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "requires a non-empty"):
            _parse_operation(
                {
                    "kind": "grant_last_words",
                    "target": "own_hand",
                    "operations": [],
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "follower target"):
            _parse_operation(
                {
                    "kind": "grant_effect_destroy_immunity",
                    "target": "own_leader",
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "requires own_hand"):
            _parse_operation(
                {
                    "kind": "summon_from_hand",
                    "target": "own_unit",
                },
                "test.json",
                1,
            )

    def test_grasshopper_uses_combo_after_current_play_and_only_draws_matching_follower(self):
        engine = self.fresh(seed=7)
        player = engine.players[0]
        player.cards_played_this_turn = 2
        matching = _card(991201, cost=3)
        wrong_type = _card(
            991202,
            cost=3,
            card_type="法术",
            attack=None,
            life=None,
        )
        wrong_cost = _card(991203, cost=2)
        player.deck = [wrong_type, matching, wrong_cost]
        source = _put_hand(engine, self.repository.get(10111140))

        engine.apply(PlayCard(0, player.hand.index(source)))

        self.assertEqual(player.cards_played_this_turn, 3)
        self.assertEqual([card.card_id for card in player.hand], [991201])
        self.assertEqual(
            [card.card_id for card in player.deck],
            [991202, 991203],
        )
        draw_event = next(
            event for event in engine.event_history
            if (
                event.type is EventType.CARD_DRAWN
                and event.metadata.get("filtered")
            )
        )
        self.assertEqual(
            (
                draw_event.metadata["card_type_filter"],
                draw_event.metadata["cost_min_filter"],
                draw_event.metadata["cost_max_filter"],
            ),
            ("随从", 3, 3),
        )

        no_match = self.fresh(seed=9)
        no_match.players[0].cards_played_this_turn = 4
        no_match.players[0].deck = [_card(991204, cost=4, card_type="护符")]
        source = _put_hand(no_match, self.repository.get(10111140))
        no_match.apply(PlayCard(0, no_match.players[0].hand.index(source)))
        self.assertEqual(len(no_match.players[0].deck), 1)
        self.assertEqual(no_match.players[0].fatigue, 0)

    def test_dynamic_filtered_draw_handles_full_hand_and_is_seed_reproducible(self):
        operations = self.rulebook.operations_for(10111140, Trigger.FANFARE)
        full = self.fresh(seed=11)
        full.players[0].cards_played_this_turn = 2
        full.players[0].deck = [_card(991205, cost=2)]
        for index in range(full.config.max_hand):
            _put_hand(full, _card(991210 + index))
        full._start_effects(
            self.repository.get(10111140),
            None,
            operations,
            controller=0,
        )
        self.assertEqual(len(full.players[0].hand), full.config.max_hand)
        self.assertEqual(
            [card.definition.card_id for card in full.players[0].graveyard],
            [991205],
        )

        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=13)
            engine.players[0].cards_played_this_turn = 1
            engine.players[0].deck = [
                _card(991230, cost=2),
                _card(991231, cost=2),
            ]
            source = _put_hand(engine, self.repository.get(10111140))
            engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
            outcomes.append(
                (
                    engine.players[0].hand[0].card_id,
                    engine.deterministic_fingerprint(),
                )
            )
        self.assertEqual(outcomes[0], outcomes[1])

    def test_flight_grants_rush_and_last_words_then_runtime_survives_play_and_clone(self):
        engine = self.fresh(seed=17)
        artifact = _put_hand(
            engine,
            _card(
                991240,
                class_id=7,
                class_name="超越者",
                tribe_id=14,
                tribe_name="创造物",
            ),
        )
        non_artifact = _put_hand(engine, _card(991241))
        flight = _put_hand(engine, self.repository.get(10272310))
        engine.players[0].deck = [_card(991242)]

        engine.apply(PlayCard(0, engine.players[0].hand.index(flight)))
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [artifact.entity_id],
        )
        _choose_entity(engine, artifact.entity_id)

        self.assertTrue(artifact.has_keyword("突进"))
        self.assertEqual(len(artifact.granted_last_words), 1)
        self.assertEqual(
            [event.metadata["ability"] for event in engine.event_history
             if event.type is EventType.CARD_ABILITY_GRANTED],
            ["last_words"],
        )
        self.assertEqual(
            _hand_runtime(artifact, engine.state.turn)[10:12],
            (0.25, 0.0),
        )
        self.assertFalse(non_artifact.has_keyword("突进"))

        clone = engine.clone()
        self.assertEqual(
            clone.deterministic_fingerprint(),
            engine.deterministic_fingerprint(),
        )
        played_entity_ids = []
        for current in (engine, clone):
            selected = next(
                card for card in current.players[0].hand
                if card.card_id == 991240
            )
            current.apply(
                PlayCard(0, current.players[0].hand.index(selected))
            )
            unit = next(
                unit for unit in current.players[0].board
                if unit.definition.card_id == 991240
            )
            played_entity_ids.append(unit.entity_id)
            self.assertTrue(unit.has_keyword("突进"))
            self.assertEqual(len(unit.granted_last_words), 1)
            self.assertEqual(_board_runtime(unit)[16:18], (0.25, 0.0))
            _destroy_units(current, unit)
            self.assertTrue(
                any(card.card_id == 991242 for card in current.players[0].hand)
            )
        self.assertEqual(played_entity_ids[0], played_entity_ids[1])
        self.assertEqual(
            clone.deterministic_fingerprint(),
            engine.deterministic_fingerprint(),
        )

    def test_flight_requires_a_live_artifact_follower_and_illegal_paths_are_atomic(self):
        no_target = self.fresh(seed=19)
        _put_hand(
            no_target,
            _card(
                991250,
                card_type="护符",
                attack=None,
                life=None,
                tribe_name="创造物",
            ),
        )
        flight = _put_hand(no_target, self.repository.get(10272310))
        before = no_target.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            no_target.apply(
                PlayCard(0, no_target.players[0].hand.index(flight))
            )
        self.assertEqual(no_target.deterministic_fingerprint(), before)

        stale = self.fresh(seed=23)
        artifact = _put_hand(
            stale,
            _card(991251, tribe_id=14, tribe_name="创造物"),
        )
        flight = _put_hand(stale, self.repository.get(10272310))
        stale.apply(PlayCard(0, stale.players[0].hand.index(flight)))
        stale_option = next(
            option for option in stale.state.pending_choice.options
            if option.entity_id == artifact.entity_id
        )
        _remove_hand_entity(stale, artifact.entity_id)
        stale.apply(Choose(0, stale_option.option_id))
        self.assertIsNone(stale.state.pending_choice)
        self.assertFalse(artifact.has_keyword("突进"))
        stale.assert_invariants()

    def test_carnelia_adds_core_and_grants_ward_and_effect_destroy_immunity(self):
        engine = self.fresh(seed=29)
        artifact = _put_hand(
            engine,
            _card(991260, tribe_id=14, tribe_name="创造物", life=3),
        )
        source_card = _put_hand(engine, self.repository.get(10273110))
        engine.apply(PlayCard(0, engine.players[0].hand.index(source_card)))
        source = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10273110
        )
        self.assertTrue(
            any(card.card_id == 90071220 for card in engine.players[0].hand)
        )

        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [artifact.entity_id],
        )
        _choose_entity(engine, artifact.entity_id)
        self.assertTrue(artifact.has_keyword("守护"))
        self.assertTrue(artifact.effect_destroy_immunity)
        self.assertEqual(_hand_runtime(artifact, engine.state.turn)[11], 1.0)

        engine.apply(PlayCard(0, engine.players[0].hand.index(artifact)))
        protected = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 991260
        )
        self.assertTrue(protected.has_keyword("守护"))
        self.assertTrue(protected.effect_destroy_immunity)
        engine._start_effects(
            _card(991261, card_type="法术", attack=None, life=None),
            None,
            (
                EffectOperation(
                    EffectKind.DESTROY,
                    TargetKind.ALL_ENEMY_UNITS,
                ),
            ),
            controller=1,
        )
        self.assertIn(protected, engine.players[0].board)
        prevention = next(
            event for event in engine.event_history
            if (
                event.type is EventType.EFFECT_DESTROY_PREVENTED
                and event.target_id == protected.entity_id
            )
        )
        self.assertFalse(prevention.metadata["printed_ability"])
        self.assertTrue(prevention.metadata["granted_ability"])

        engine._start_effects(
            _card(991262, card_type="法术", attack=None, life=None),
            None,
            (
                EffectOperation(
                    EffectKind.REMOVE_ALL_ABILITIES,
                    TargetKind.OWN_UNIT,
                ),
            ),
            controller=0,
        )
        _choose_entity(engine, protected.entity_id)
        self.assertFalse(protected.has_keyword("守护"))
        self.assertFalse(protected.effect_destroy_immunity)
        engine._start_effects(
            _card(991263, card_type="法术", attack=None, life=None),
            None,
            (
                EffectOperation(
                    EffectKind.DESTROY,
                    TargetKind.ALL_ENEMY_UNITS,
                ),
            ),
            controller=1,
        )
        self.assertNotIn(protected, engine.players[0].board)

    def test_carnelia_no_target_and_full_hand_token_paths_are_safe(self):
        no_target = self.fresh(seed=31)
        source_card = _put_hand(no_target, self.repository.get(10273110))
        no_target.apply(
            PlayCard(0, no_target.players[0].hand.index(source_card))
        )
        source = next(
            unit for unit in no_target.players[0].board
            if unit.definition.card_id == 10273110
        )
        _enable_evolve(no_target)
        no_target.apply(Evolve(0, source.entity_id))
        self.assertIsNone(no_target.state.pending_choice)
        self.assertTrue(source.evolved)

        full = self.fresh(seed=37)
        for index in range(full.config.max_hand):
            _put_hand(full, _card(991270 + index))
        full._start_effects(
            self.repository.get(10273110),
            None,
            self.rulebook.operations_for(10273110, Trigger.FANFARE),
            controller=0,
        )
        self.assertTrue(
            any(
                card.definition.card_id == 90071220
                for card in full.players[0].graveyard
            )
        )

    def test_chloe_enhance_summons_same_hand_entity_without_fanfare_then_returns(self):
        normal = self.fresh(seed=41)
        target = _put_hand(normal, self.repository.get(10111140))
        chloe = _put_hand(normal, self.repository.get(10412110))
        normal.apply(PlayCard(0, normal.players[0].hand.index(chloe)))
        self.assertIn(target, normal.players[0].hand)
        self.assertTrue(
            any(unit.definition.card_id == 10412110 for unit in normal.players[0].board)
        )

        engine = self.fresh(seed=43)
        engine.players[0].deck = [_card(991280, cost=1)]
        target = _put_hand(engine, self.repository.get(10111140))
        target.add_keyword("守护")
        target.stat_modifiers.append(
            StatModifier(
                modifier_id=991,
                attack_delta=2,
                health_delta=3,
                duration="permanent",
            )
        )
        target.effect_destroy_immunity = True
        target_entity_id = target.entity_id
        chloe = _put_hand(engine, self.repository.get(10412110))
        engine.apply(
            PlayCard(
                0,
                engine.players[0].hand.index(chloe),
                mode_id="enhance_8",
            )
        )
        _choose_entity(engine, target.entity_id)

        summoned = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10111140
        )
        self.assertEqual(summoned.entity_id, target_entity_id)
        self.assertEqual(
            (summoned.attack, summoned.max_health),
            (
                self.repository.get(10111140).attack + 2,
                self.repository.get(10111140).life + 3,
            ),
        )
        self.assertTrue(summoned.has_keyword("守护"))
        self.assertTrue(summoned.effect_destroy_immunity)
        self.assertEqual([card.card_id for card in engine.players[0].deck], [991280])
        self.assertTrue(
            any(card.card_id == 10412110 for card in engine.players[0].hand)
        )
        self.assertFalse(
            any(unit.definition.card_id == 10412110 for unit in engine.players[0].board)
        )
        self.assertTrue(
            any(
                event.type is EventType.HAND_CARD_SUMMONED
                and event.source_id == target_entity_id
                for event in engine.event_history
            )
        )

    def test_chloe_handles_full_board_stale_target_and_missing_target(self):
        full = self.fresh(seed=47)
        for index in range(4):
            _put_unit(full, 0, _card(991290 + index))
        target = _put_hand(full, _card(991294))
        chloe = _put_hand(full, self.repository.get(10412110))
        full.apply(
            PlayCard(
                0,
                full.players[0].hand.index(chloe),
                mode_id="enhance_8",
            )
        )
        _choose_entity(full, target.entity_id)
        self.assertIn(target, full.players[0].hand)
        self.assertEqual(len(full.players[0].board), 4)
        self.assertTrue(
            any(card.card_id == 10412110 for card in full.players[0].hand)
        )

        stale = self.fresh(seed=53)
        target = _put_hand(stale, _card(991295))
        chloe = _put_hand(stale, self.repository.get(10412110))
        stale.apply(
            PlayCard(
                0,
                stale.players[0].hand.index(chloe),
                mode_id="enhance_8",
            )
        )
        stale_option = next(
            option for option in stale.state.pending_choice.options
            if option.entity_id == target.entity_id
        )
        _remove_hand_entity(stale, target.entity_id)
        stale.apply(Choose(0, stale_option.option_id))
        self.assertTrue(
            any(card.card_id == 10412110 for card in stale.players[0].hand)
        )
        self.assertIsNone(stale.state.pending_choice)
        stale.assert_invariants()

        missing = self.fresh(seed=59)
        chloe = _put_hand(missing, self.repository.get(10412110))
        before = missing.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            missing.apply(
                PlayCard(
                    0,
                    missing.players[0].hand.index(chloe),
                    mode_id="enhance_8",
                )
            )
        self.assertEqual(missing.deterministic_fingerprint(), before)
        self.assertIn(
            PlayCard(0, 0, mode_id="normal"),
            missing.legal_commands(),
        )

    def test_cassius_uses_selected_current_attack_for_simultaneous_enemy_damage(self):
        engine = self.fresh(seed=61)
        artifact = _put_hand(
            engine,
            _card(
                991300,
                attack=2,
                tribe_id=14,
                tribe_name="创造物",
            ),
        )
        artifact.stat_modifiers.append(
            StatModifier(
                modifier_id=992,
                attack_delta=3,
                health_delta=0,
                duration="permanent",
            )
        )
        enemies = [
            _put_unit(engine, 1, _card(991301, life=5)),
            _put_unit(engine, 1, _card(991302, life=4)),
        ]
        cassius_card = _put_hand(engine, self.repository.get(10473110))
        engine.apply(
            PlayCard(0, engine.players[0].hand.index(cassius_card))
        )
        _choose_entity(engine, artifact.entity_id)

        self.assertTrue(all(enemy not in engine.players[1].board for enemy in enemies))
        destroyed = [
            event for event in engine.event_history
            if event.type is EventType.FOLLOWER_DESTROYED
            and event.source_id in {enemy.entity_id for enemy in enemies}
        ]
        self.assertEqual(len(destroyed), 2)
        self.assertIn(artifact, engine.players[0].hand)
        engine.assert_invariants()

    def test_cassius_no_target_stale_target_and_last_words_capacity(self):
        no_target = self.fresh(seed=67)
        enemy = _put_unit(no_target, 1, _card(991310, life=7))
        source_card = _put_hand(no_target, self.repository.get(10473110))
        no_target.apply(
            PlayCard(0, no_target.players[0].hand.index(source_card))
        )
        source = next(
            unit for unit in no_target.players[0].board
            if unit.definition.card_id == 10473110
        )
        self.assertIsNone(no_target.state.pending_choice)
        self.assertEqual(enemy.health, 7)
        _destroy_units(no_target, source)
        self.assertTrue(
            any(card.card_id == 90072120 for card in no_target.players[0].hand)
        )

        stale = self.fresh(seed=71)
        enemy = _put_unit(stale, 1, _card(991311, life=7))
        artifact = _put_hand(
            stale,
            _card(991312, attack=5, tribe_id=14, tribe_name="创造物"),
        )
        source_card = _put_hand(stale, self.repository.get(10473110))
        stale.apply(
            PlayCard(0, stale.players[0].hand.index(source_card))
        )
        option = next(
            option for option in stale.state.pending_choice.options
            if option.entity_id == artifact.entity_id
        )
        _remove_hand_entity(stale, artifact.entity_id)
        stale.apply(Choose(0, option.option_id))
        self.assertEqual(enemy.health, 7)
        self.assertIsNone(stale.state.pending_choice)

        full = self.fresh(seed=73)
        source_card = _put_hand(full, self.repository.get(10473110))
        full.apply(PlayCard(0, full.players[0].hand.index(source_card)))
        source = next(
            unit for unit in full.players[0].board
            if unit.definition.card_id == 10473110
        )
        for index in range(full.config.max_hand):
            _put_hand(full, _card(991320 + index))
        _destroy_units(full, source)
        self.assertTrue(
            any(
                card.definition.card_id == 90072120
                for card in full.players[0].graveyard
            )
        )

    def test_action_mask_matches_commands_for_enhance_and_required_hand_target(self):
        deck = [
            _card(
                992000 + index,
                class_id=1,
                class_name="精灵",
            )
            for index in range(40)
        ]
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
            player.max_mana = player.mana = 10
        _put_hand(env.core, _card(992050))
        _put_hand(env.core, self.repository.get(10412110))
        env.invalidate_cache(reason="twelfth batch hand modes")
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        self.assertIn(
            PlayCard(0, 1, mode_id="normal"),
            decoded,
        )
        self.assertIn(
            PlayCard(0, 1, mode_id="enhance_8"),
            decoded,
        )
        before = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(PlayCard(0, 1, mode_id="enhance_9"))
        self.assertEqual(env.core.deterministic_fingerprint(), before)


class SelectedHandGrantsTwelfthAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository("data/cards.sqlite3")

    def test_database_snapshot_multilingual_text_references_and_modes(self):
        expected_stats = {
            10111140: (10001, 1, 3, 2, 1),
            10272310: (10002, 7, 1, None, None),
            10273110: (10002, 7, 2, 2, 2),
            10412110: (10004, 1, 2, 2, 2),
            10473110: (10004, 7, 5, 3, 2),
        }
        expected_text = {
            10111140: ("X is your", "コンボ"),
            10272310: ("Last Words", "ラストワード"),
            10273110: ("destroyed by abilities", "能力によって破壊できない"),
            10412110: ("Return this card to hand", "手札に戻す"),
            10473110: ("selected follower's attack", "選んだフォロワーの攻撃力"),
        }
        expected_references = {
            10111140: [],
            10272310: [],
            10273110: [(90071220, "过往核心")],
            10412110: [],
            10473110: [(90072120, "城堡创造物")],
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            source_map = _load_source_text_map(connection)
            for card_id in CARD_IDS:
                with self.subTest(card_id=card_id):
                    stats = connection.execute(
                        """
                        SELECT card_set_id, class_id, cost, attack, life
                        FROM cards WHERE card_id=?
                        """,
                        (card_id,),
                    ).fetchone()
                    self.assertEqual(stats, expected_stats[card_id])
                    self.assertEqual(
                        _source_text_sha256(source_map[card_id]),
                        SOURCE_HASHES[card_id],
                    )
                    text_eng, text_jpn = connection.execute(
                        """
                        SELECT text_eng, text_jpn FROM skill_texts
                        WHERE card_id=? ORDER BY position
                        """,
                        (card_id,),
                    ).fetchone()
                    self.assertIn(expected_text[card_id][0], text_eng)
                    self.assertIn(expected_text[card_id][1], text_jpn)
                    references = connection.execute(
                        """
                        SELECT referenced_card_id, referenced_name
                        FROM card_references
                        WHERE card_id=? AND referenced_card_id IS NOT NULL
                        ORDER BY position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(references, expected_references[card_id])
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_all_five_cards_are_exact_with_clause_and_token_evidence(self):
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
                classification = report["classifications"][str(card_id)]
                audit = classification["clause_audit"]
                self.assertEqual(classification["coverage"], "covered_exact")
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(
                    audit["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(audit["test_evidence"], [TEST_EVIDENCE])

        token_audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        self.assertEqual(
            token_audit["summary"]["categories"]["entry_behavior_complete"],
            91,
        )
        for token_id in (90071220, 90072120):
            token = next(
                item for item in token_audit["cards"]
                if item["card_id"] == token_id
            )
            self.assertEqual(token["category"], "entry_behavior_complete")


if __name__ == "__main__":
    unittest.main()
