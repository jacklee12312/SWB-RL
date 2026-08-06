# -*- coding: utf-8 -*-
"""Exact hand-copy summons and granted turn-end destruction."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import (
    ActivateAmulet,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
    SuperEvolve,
)
from swb.engine.effects import (
    EffectKind,
    HandFilter,
    TargetKind,
    TurnEndDestroyTiming,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import CostModifier, HandCard, StatModifier
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10172320,
    10173140,
    10174130,
    10261120,
    10271210,
    10274120,
    10572110,
)
SOURCE_HASHES = {
    10172320: "7bde96addfbbc1c8f15b11ba1961e313ce5212d05d7b4ca6a12027a08b214599",
    10173140: "bfd9204a87010bbac380cdfaa9dbaf77ebdc4ecb2bfca569e0f3c89fbe6f3803",
    10174130: "436ec26f501c8d33fffb67bc833bb7fdf3d6e1330e64b28d54b13a6f40638c35",
    10261120: "ad33b423354df823f3224d280220237c2f0e3e453f8b7b1c7389d83c42a158c9",
    10271210: "4736e02fb7874738f8a51d6844fd0bc71bc36007341b2872bd07e3f154490bd8",
    10274120: "768c24c22a23a3688650ce81a6602cba981a18c19c1f938ef4545cc31c5e3f3e",
    10572110: "56cab5b53b019d0ca252a88b09dc4d776b753d4d3f207611de7fbec5d85ae12d",
}
TEST_EVIDENCE = "tests/test_real_hand_exact_copy_turn_end_destroy_batch.py"


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        candidate
        for candidate in request.options
        if candidate.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False


def _enable_super_evolution(engine, player_index: int = 0) -> None:
    player = engine.players[player_index]
    unlock = (
        engine.config.first_player_super_evolution_unlock_turn
        if player_index == 0
        else engine.config.second_player_super_evolution_unlock_turn
    )
    player.turns_started = unlock
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False


class HandExactCopySchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository("data/cards.sqlite3")

    def test_new_operations_and_full_target_policy_parse_strictly(self):
        summon = _parse_operation(
            {
                "kind": "summon_hand_copy",
                "target": "own_hand",
                "target_count": 2,
                "requires_target": True,
                "requires_full_target_count": True,
                "target_key": "copies",
                "hand_filter": {
                    "card_type": "随从",
                    "cost_max": 5,
                    "tribe_name": "创造物",
                },
            },
            "test",
            10172320,
        )
        self.assertIs(summon.kind, EffectKind.SUMMON_HAND_COPY)
        self.assertTrue(summon.requires_full_target_count)
        self.assertEqual(summon.target_count, 2)

        grant = _parse_operation(
            {
                "kind": "grant_turn_end_destroy",
                "target": "enemy_unit",
                "turn_end_destroy_timing": "owner_turn",
            },
            "test",
            10261120,
        )
        self.assertIs(
            grant.turn_end_destroy_timing,
            TurnEndDestroyTiming.OWNER_TURN,
        )

        invalid_operations = (
            {
                "kind": "summon_hand_copy",
                "target": "enemy_unit",
            },
            {
                "kind": "summon_hand_copy",
                "target": "own_hand",
                "requires_full_target_count": True,
            },
            {
                "kind": "grant_turn_end_destroy",
                "target": "enemy_unit",
            },
            {
                "kind": "damage_unit",
                "target": "enemy_unit",
                "turn_end_destroy_timing": "owner_turn",
            },
        )
        for raw in invalid_operations:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test", 990001)

    def test_hand_cost_filter_uses_the_physical_current_cost(self):
        card = HandCard(
            definition=self.repository.get(90074110),
            entity_id=1,
            origin=CardOrigin.GENERATED,
        )
        filter_ = HandFilter(
            card_type="随从",
            cost_max=5,
            tribe_name="创造物",
        )
        self.assertFalse(filter_.matches(card))
        card.cost_modifiers.append(
            CostModifier(
                modifier_id=1,
                mode="set",
                amount=5,
                duration="permanent",
            )
        )
        self.assertTrue(filter_.matches(card))


class RealHandExactCopyTurnEndDestroyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1901):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_all_cards_and_activation(self):
        restart = self.rulebook.operations_for(10172320, Trigger.PLAY)
        self.assertEqual(
            [operation.kind for operation in restart],
            [EffectKind.SUMMON_HAND_COPY, EffectKind.GRANT_TURN_END_DESTROY],
        )
        self.assertEqual(restart[0].target_count, 2)
        self.assertTrue(restart[0].requires_full_target_count)
        self.assertIs(restart[1].target, TargetKind.PREVIOUS_TARGET)
        self.assertIs(
            restart[1].turn_end_destroy_timing,
            TurnEndDestroyTiming.OPPONENT_TURN,
        )
        self.assertEqual(self.rulebook.activation_for(10271210).cost, 3)
        for card_id in CARD_IDS:
            self.assertTrue(
                any(cid == card_id for cid, _trigger in self.rulebook._rules),
                card_id,
            )

    def test_restart_requires_two_candidates_atomically_and_matches_rl_mask(self):
        engine = self.fresh(seed=11)
        _put_hand(engine, self.repository.get(90071130))
        source = _put_hand(engine, self.repository.get(10172320))
        source_index = engine.players[0].hand.index(source)
        before = (
            engine.deterministic_fingerprint(),
            engine.random.getstate(),
            tuple(engine.event_history),
            tuple(engine.logs),
        )
        self.assertFalse(
            any(
                isinstance(command, PlayCard)
                and command.hand_index == source_index
                for command in engine.legal_commands()
            )
        )

        env = ShadowverseEnv(
            [_card(index) for index in range(1000, 1040)],
            [_card(index) for index in range(2000, 2040)],
            class_a=1,
            class_b=1,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.core = engine
        self.assertEqual(
            env.action_mask()[env.PLAY_OFFSET + source_index],
            0,
        )
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, source_index))
        self.assertEqual(
            before,
            (
                engine.deterministic_fingerprint(),
                engine.random.getstate(),
                tuple(engine.event_history),
                tuple(engine.logs),
            ),
        )

    def test_restart_preserves_hand_and_stat_copy_then_destroys_on_opponent_end(self):
        engine = self.fresh(seed=13)
        first = _put_hand(engine, self.repository.get(90072110))
        second = _put_hand(engine, self.repository.get(90072120))
        first.stat_modifiers.append(
            StatModifier(
                modifier_id=engine._allocate_modifier_id(),
                attack_delta=2,
                health_delta=3,
                duration="permanent",
            )
        )
        source = _put_hand(engine, self.repository.get(10172320))
        engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
        self.assertEqual(engine.state.pending_choice.target_count, 2)
        _choose_entity(engine, first.entity_id)
        self.assertNotIn(
            first.entity_id,
            [option.entity_id for option in engine.state.pending_choice.options],
        )
        _choose_entity(engine, second.entity_id)

        self.assertEqual(
            {card.entity_id for card in engine.players[0].hand},
            {first.entity_id, second.entity_id},
        )
        copies = list(engine.players[0].board)
        self.assertEqual(len(copies), 2)
        copied_first = next(
            unit for unit in copies if unit.definition.card_id == first.card_id
        )
        self.assertEqual((copied_first.attack, copied_first.health), (7, 4))
        self.assertEqual(
            {tuple(unit.turn_end_destroy_timings) for unit in copies},
            {(TurnEndDestroyTiming.OPPONENT_TURN,)},
        )
        self.assertEqual(
            len({modifier.modifier_id for modifier in copied_first.stat_modifiers}),
            1,
        )
        summon_events = [
            event
            for event in engine.event_history
            if event.type is EventType.FOLLOWER_SUMMONED
            and event.metadata.get("via") == "hand_copy_summon"
        ]
        self.assertEqual(len(summon_events), 2)

        engine.apply(EndTurn(0))
        self.assertEqual(len(engine.players[0].board), 2)
        engine.apply(EndTurn(1))
        self.assertEqual(engine.players[0].board, [])

    def test_restart_board_shortage_binds_only_successful_copy(self):
        engine = self.fresh(seed=17)
        for index in range(4):
            _put_unit(engine, 0, _card(300 + index))
        first = _put_hand(engine, self.repository.get(90071130))
        second = _put_hand(engine, self.repository.get(90072120))
        source = _put_hand(engine, self.repository.get(10172320))
        engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
        _choose_entity(engine, first.entity_id)
        _choose_entity(engine, second.entity_id)

        copies = [
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id in {first.card_id, second.card_id}
        ]
        self.assertEqual(len(copies), 1)
        self.assertEqual(
            copies[0].turn_end_destroy_timings,
            {TurnEndDestroyTiming.OPPONENT_TURN},
        )

    def test_hand_copy_and_delayed_destruction_are_seed_deterministic(self):
        def run(seed: int):
            engine = self.fresh(seed=seed)
            first = _put_hand(engine, self.repository.get(90072110))
            second = _put_hand(engine, self.repository.get(90072120))
            source = _put_hand(engine, self.repository.get(10172320))
            engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
            _choose_entity(engine, first.entity_id)
            _choose_entity(engine, second.entity_id)
            engine.apply(EndTurn(0))
            engine.apply(EndTurn(1))
            return (
                engine.deterministic_fingerprint(),
                tuple(engine.event_history),
                tuple(engine.logs),
            )

        self.assertEqual(run(53), run(53))

    def test_stale_second_hand_choice_cancels_safely_without_partial_summon(self):
        engine = self.fresh(seed=19)
        first = _put_hand(engine, self.repository.get(90071130))
        second = _put_hand(engine, self.repository.get(90072120))
        source = _put_hand(engine, self.repository.get(10172320))
        engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
        _choose_entity(engine, first.entity_id)
        request = engine.state.pending_choice
        stale_option = next(
            option for option in request.options if option.entity_id == second.entity_id
        )
        index = engine.players[0].hand.index(second)
        engine.players[0].hand.pop(index)
        engine.players[0].hand_entity_ids.pop(index)

        engine.apply(Choose(0, stale_option.option_id))
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[0].board, [])
        self.assertTrue(
            any("取消本次多目标效果" in log for log in engine.logs)
        )

    def test_loramia_selects_as_many_as_exist_and_buffs_only_artifacts(self):
        engine = self.fresh(seed=23)
        first = _put_hand(engine, self.repository.get(90071130))
        second = _put_hand(engine, self.repository.get(90072120))
        _enable_super_evolution(engine)
        source_card = _put_hand(engine, self.repository.get(10174130))
        engine.apply(PlayCard(0, engine.players[0].hand.index(source_card)))
        source = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10174130
        )
        self.assertEqual(engine.state.pending_choice.target_count, 2)
        _choose_entity(engine, first.entity_id)
        _choose_entity(engine, second.entity_id)
        non_artifact = _put_unit(engine, 0, _card(901, attack=3, life=3))
        artifact_before = [
            (unit, unit.attack, unit.max_health)
            for unit in engine.players[0].board
            if unit.definition.tribe_name == "创造物"
        ]

        engine.apply(SuperEvolve(0, source.entity_id))
        for unit, attack, max_health in artifact_before:
            self.assertEqual((unit.attack, unit.max_health), (attack + 1, max_health + 1))
        self.assertEqual((non_artifact.attack, non_artifact.max_health), (3, 3))

    def test_aloette_adds_both_cores_and_evolves_into_a_hand_copy(self):
        engine = self.fresh(seed=29)
        target = _put_hand(engine, self.repository.get(90071130))
        _enable_evolution(engine)
        source_card = _put_hand(engine, self.repository.get(10173140))
        engine.apply(PlayCard(0, engine.players[0].hand.index(source_card)))
        source = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10173140
        )
        self.assertEqual(
            {card.card_id for card in engine.players[0].hand},
            {target.card_id, 90071210, 90071220},
        )
        engine.apply(Evolve(0, source.entity_id))
        _choose_entity(engine, target.entity_id)
        self.assertIn(target, engine.players[0].hand)
        self.assertEqual(
            sum(unit.definition.card_id == target.card_id for unit in engine.players[0].board),
            1,
        )

    def test_dams_owner_turn_destroy_and_super_evolution_protection(self):
        normal = self.fresh(seed=31)
        target = _put_unit(normal, 1, _card(910))
        source = _put_hand(normal, self.repository.get(10261120))
        normal.apply(PlayCard(0, normal.players[0].hand.index(source)))
        _choose_entity(normal, target.entity_id)
        normal.apply(EndTurn(0))
        normal.apply(EndTurn(1))
        self.assertNotIn(target, normal.players[1].board)

        protected = self.fresh(seed=37)
        target = _put_unit(protected, 1, _card(911))
        source = _put_hand(protected, self.repository.get(10261120))
        protected.apply(PlayCard(0, protected.players[0].hand.index(source)))
        _choose_entity(protected, target.entity_id)
        protected.apply(EndTurn(0))
        _enable_super_evolution(protected, 1)
        protected.apply(SuperEvolve(1, target.entity_id))
        protected.apply(EndTurn(1))
        self.assertIn(target, protected.players[1].board)

        target.remove_all_abilities()
        self.assertEqual(target.turn_end_destroy_timings, set())

    def test_artifact_launcher_activation_destroys_source_and_marks_copy(self):
        unavailable = self.fresh(seed=39)
        source_card = _put_hand(unavailable, self.repository.get(10271210))
        unavailable.apply(
            PlayCard(0, unavailable.players[0].hand.index(source_card))
        )
        unavailable_amulet = next(
            entity for entity in unavailable.players[0].board
            if entity.definition.card_id == 10271210
        )
        before = unavailable.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            unavailable.apply(ActivateAmulet(0, unavailable_amulet.entity_id))
        self.assertEqual(unavailable.deterministic_fingerprint(), before)

        engine = self.fresh(seed=41)
        target = _put_hand(engine, self.repository.get(90071130))
        source_card = _put_hand(engine, self.repository.get(10271210))
        engine.apply(PlayCard(0, engine.players[0].hand.index(source_card)))
        amulet = next(
            entity for entity in engine.players[0].board
            if entity.definition.card_id == 10271210
        )
        self.assertEqual(
            {card.card_id for card in engine.players[0].hand},
            {target.card_id, 90071210, 90071220},
        )
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        _choose_entity(engine, target.entity_id)
        self.assertNotIn(amulet, engine.players[0].board)
        copy = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == target.card_id
        )
        self.assertEqual(
            copy.turn_end_destroy_timings,
            {TurnEndDestroyTiming.OPPONENT_TURN},
        )

    def test_garula_and_geographer_complete_copy_and_super_evolve_clauses(self):
        garula = self.fresh(seed=43)
        target = _put_hand(garula, self.repository.get(90071130))
        _enable_super_evolution(garula)
        source_card = _put_hand(garula, self.repository.get(10274120))
        garula.apply(PlayCard(0, garula.players[0].hand.index(source_card)))
        source = next(
            unit for unit in garula.players[0].board
            if unit.definition.card_id == 10274120
        )
        _choose_entity(garula, target.entity_id)
        garula.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(source.attacks_per_turn, 2)
        self.assertIn(target, garula.players[0].hand)

        geographer = self.fresh(seed=47)
        _enable_super_evolution(geographer)
        source_card = _put_hand(geographer, self.repository.get(10572110))
        geographer.apply(
            PlayCard(0, geographer.players[0].hand.index(source_card))
        )
        source = next(
            unit for unit in geographer.players[0].board
            if unit.definition.card_id == 10572110
        )
        beta = next(
            card for card in geographer.players[0].hand
            if card.card_id == 90073120
        )
        geographer.apply(SuperEvolve(0, source.entity_id))
        _choose_entity(geographer, beta.entity_id)
        self.assertIn(beta, geographer.players[0].hand)
        self.assertEqual(
            sum(
                unit.definition.card_id == 90073120
                for unit in geographer.players[0].board
            ),
            1,
        )

    def test_coverage_maps_all_seven_cards_exactly(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    [TEST_EVIDENCE],
                )
                self.assertEqual(
                    info["clause_audit"]["status"],
                    "mapped_exact",
                )


if __name__ == "__main__":
    unittest.main()
