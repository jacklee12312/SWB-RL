# -*- coding: utf-8 -*-
"""Exact Lloyd target forcing, Victoria strike, and Puppetry producer chains."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import (
    CardPassive,
    CardRule,
    RuleBook,
    Trigger,
    _parse_operation,
    _parse_passive,
)
from swb.engine.commands import Attack, Choose, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import (
    BoardFilter,
    EffectKind,
    EffectOperation,
    ExprType,
    TargetKind,
)
from swb.engine.resolution import IllegalCommand
from swb.engine.state import TargetingRestriction, Unit
from swb.engine.targeting import target_candidates
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (10174120, 10274110, 10273310, 10173120)
TOKEN_IDS = (90074120, 90074130)
SOURCE_HASHES = {
    10174120: "546af1a901720d730671111f37e53d9799e263842c42a1ee5ffec056c67ee0c4",
    10274110: "b2343c2ec75fc900c4b3965022f5dd55c8b977abb7352a8dfa8a1d48b0f5cd15",
    10273310: "15c9f048455f622f215ea5c5dca41fe43e6505a8fc7d2cd02ad254f0cc03733c",
    10173120: "48c6935b42047f3da6479b06c881139c4bd30b13e5c49ddb7596bd7b3662f4bc",
    90074120: "25c5da3e44af5d48f6759bed8ebaf8a3099524c64aa98498d987ae27b66083be",
    90074130: "a5ab4669f437aca69d5515fcecb14b2595367bcb5f1003b4ef9fe72feadde29f",
}


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _choose_option(engine, option_id: str) -> None:
    request = engine.state.pending_choice
    engine.apply(Choose(request.player_index, option_id))


def _enable_evolution(engine, *, super_evolve: bool = False) -> None:
    player = engine.players[0]
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False
    if super_evolve:
        player.turns_started = engine.config.first_player_super_evolution_unlock_turn
        player.super_evolution_points = max(1, player.super_evolution_points)
    else:
        player.turns_started = engine.config.evolution_unlock_turn
        player.evolution_points = max(1, player.evolution_points)


class ForcedTargetStrikeSchemaTests(unittest.TestCase):
    def test_attack_target_parses_source_attack_expression(self):
        operation = _parse_operation(
            {
                "kind": "damage_unit",
                "target": "attack_target",
                "amount": {"type": "source_attack"},
            },
            "test",
            123,
        )
        self.assertIs(operation.target, TargetKind.ATTACK_TARGET)
        self.assertIs(operation.amount_expr.type, ExprType.SOURCE_ATTACK)

    def test_attack_target_rejects_non_board_effect_and_non_attack_rule(self):
        with self.assertRaises(ValueError):
            _parse_operation(
                {"kind": "draw", "target": "attack_target", "amount": 1},
                "test",
                123,
            )
        with self.assertRaisesRegex(ValueError, "only valid.*attack"):
            RuleBook(
                rules=(
                    CardRule(
                        123,
                        Trigger.EVOLVE,
                        (
                            EffectOperation(
                                EffectKind.DAMAGE_UNIT,
                                TargetKind.ATTACK_TARGET,
                                1,
                            ),
                        ),
                    ),
                )
            )

    def test_forced_target_passive_schema_is_zero_amount_static(self):
        passive = _parse_passive(
            {"card_id": 123, "kind": "forces_enemy_ability_target"},
            "test",
        )
        self.assertEqual(
            passive,
            CardPassive(123, "forces_enemy_ability_target", 0),
        )
        with self.assertRaises(ValueError):
            _parse_passive(
                {
                    "card_id": 123,
                    "kind": "forces_enemy_ability_target",
                    "amount": 1,
                },
                "test",
            )


class ForcedTargetCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1701):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def _forced_board(self):
        engine = self.fresh()
        own = _put_unit(engine, 0, _card(700001, name="己方随从"))
        lloyd = engine._summon_follower_to_board(
            1,
            self.repository.get(90074120),
            summon_cause="test",
        )
        other = _put_unit(engine, 1, _card(700002, name="其他敌方随从"))
        return engine, own, lloyd, other

    def test_manual_enemy_and_mixed_targets_are_forced_to_lloyd(self):
        engine, own, lloyd, other = self._forced_board()
        enemy = EffectOperation(EffectKind.DESTROY, TargetKind.ENEMY_UNIT)
        mixed = EffectOperation(EffectKind.DESTROY, TargetKind.ANY_UNIT)

        self.assertEqual(target_candidates(enemy, 0, engine.players), [lloyd])
        self.assertEqual(
            target_candidates(mixed, 0, engine.players),
            [lloyd],
        )
        options = engine._target_options(
            EffectOperation(
                EffectKind.DAMAGE_UNIT,
                TargetKind.ENEMY_UNIT_OR_LEADER,
                1,
            ),
            0,
        )
        self.assertEqual([option.entity_id for option in options], [lloyd.entity_id])
        self.assertNotIn(other.entity_id, [option.entity_id for option in options])
        self.assertFalse(any(option.leader_player_index == 1 for option in options))

    def test_own_targets_random_and_all_targets_are_not_redirected(self):
        engine, own, lloyd, other = self._forced_board()
        own_candidates = target_candidates(
            EffectOperation(EffectKind.BUFF_UNIT, TargetKind.OWN_UNIT, 1, 1),
            0,
            engine.players,
        )
        random_candidates = target_candidates(
            EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.RANDOM_ENEMY_UNIT, 1),
            0,
            engine.players,
        )
        all_candidates = target_candidates(
            EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ALL_ENEMY_UNITS, 1),
            0,
            engine.players,
        )
        self.assertEqual(own_candidates, [own])
        self.assertEqual(random_candidates, [lloyd, other])
        self.assertEqual(all_candidates, [lloyd, other])

    def test_incompatible_filter_has_no_enemy_target(self):
        engine, _, _, other = self._forced_board()
        operation = EffectOperation(
            EffectKind.DESTROY,
            TargetKind.ENEMY_UNIT,
            board_filter=BoardFilter(cost_max=2),
        )
        self.assertEqual(other.definition.cost, 1)
        self.assertEqual(target_candidates(operation, 0, engine.players), [])

    def test_blocked_unit_or_leader_fallback_makes_play_illegal_atomically(self):
        spell = _card(
            730001,
            name="受限目标法术",
            card_type="法术",
            attack=None,
            life=None,
        )
        operation = EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.ENEMY_UNIT_OR_LEADER,
            1,
            board_filter=BoardFilter(cost_max=2),
            requires_target=True,
        )
        engine = _fresh(
            RuleBook((CardRule(730001, Trigger.PLAY, (operation,)),)),
            self.repository,
            seed=31,
        )
        lloyd = _put_unit(engine, 1, self.repository.get(90074120))
        lloyd.add_targeting_restriction(
            TargetingRestriction.FORCES_ENEMY_ABILITY_TARGET,
            duration="permanent",
        )
        _put_unit(engine, 1, _card(730002, cost=1))
        _put_hand(engine, spell)
        self.assertEqual(engine._target_options(operation, 0), [])
        self.assertFalse(engine._has_candidates_for(operation, 0))
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_multiple_lloyds_and_runtime_revalidation(self):
        engine, _, first, other = self._forced_board()
        second = engine._summon_follower_to_board(
            1,
            self.repository.get(90074120),
            summon_cause="test",
        )
        operation = EffectOperation(EffectKind.DESTROY, TargetKind.ENEMY_UNIT)
        self.assertEqual(
            target_candidates(operation, 0, engine.players),
            [first, second],
        )

        first.remove_all_abilities()
        self.assertEqual(target_candidates(operation, 0, engine.players), [second])
        engine.players[1].board.remove(second)
        self.assertEqual(target_candidates(operation, 0, engine.players), [first, other])
        self.assertFalse(first.forces_enemy_ability_target)


class RealGeneratedForcedTargetStrikeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1801):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_victoria_follower_strike_uses_current_attack_before_combat(self):
        engine = self.fresh(seed=3)
        victoria = engine._summon_follower_to_board(
            0,
            self.repository.get(90074130),
            summon_cause="test",
        )
        victoria.base_attack += 2
        victoria._recompute_attack()
        victoria.can_attack = True
        target = _put_unit(engine, 1, _card(710001, attack=2, life=20))

        engine.apply(Attack(0, victoria.entity_id, target.entity_id))

        self.assertEqual(victoria.attack, 8)
        self.assertEqual(target.health, 4)
        self.assertEqual(victoria.health, 0)

    def test_victoria_strike_can_remove_target_before_combat_and_skips_leader(self):
        follower = self.fresh(seed=5)
        victoria = follower._summon_follower_to_board(
            0,
            self.repository.get(90074130),
            summon_cause="test",
        )
        victoria.can_attack = True
        target = _put_unit(follower, 1, _card(710002, attack=9, life=6))
        follower.apply(Attack(0, victoria.entity_id, target.entity_id))
        self.assertNotIn(target, follower.players[1].board)
        self.assertIn(victoria, follower.players[0].board)
        self.assertEqual(victoria.attacks_remaining, 0)

        leader = self.fresh(seed=7)
        victoria = leader._summon_follower_to_board(
            0,
            self.repository.get(90074130),
            summon_cause="test",
        )
        victoria.summoned_this_turn = False
        victoria.can_attack = True
        victoria.rush_only = False
        leader.apply(Attack(0, victoria.entity_id, None))
        self.assertEqual(leader.players[1].health, 14)

    def test_cooperation_summons_lloyd_then_victoria_with_board_shortage(self):
        engine = self.fresh(seed=11)
        _play(engine, self.repository, 10273310)
        lloyd, victoria = engine.players[0].board
        self.assertEqual(
            (lloyd.definition.card_id, victoria.definition.card_id),
            (90074120, 90074130),
        )
        self.assertTrue(lloyd.has_keyword("守护"))
        self.assertTrue(lloyd.forces_enemy_ability_target)
        self.assertTrue(victoria.has_keyword("突进"))

        shortage = self.fresh(seed=13)
        for index in range(shortage.config.max_board - 1):
            _put_unit(shortage, 0, _card(720000 + index))
        _play(shortage, self.repository, 10273310)
        generated = [
            entity.definition.card_id
            for entity in shortage.players[0].board
            if entity.definition.card_id in TOKEN_IDS
        ]
        self.assertEqual(generated, [90074120])

    def test_orchis_grants_storm_bane_and_super_evolve_produces_two_puppets(self):
        engine = self.fresh(seed=17)
        orchis = _play(engine, self.repository, 10174120)
        lloyd = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90074120
        )
        self.assertFalse(orchis.has_keyword("疾驰"))
        self.assertFalse(orchis.has_keyword("必杀"))
        self.assertTrue(lloyd.has_keyword("疾驰"))
        self.assertTrue(lloyd.has_keyword("必杀"))
        self.assertTrue(lloyd.has_keyword("守护"))
        self.assertTrue(lloyd.can_attack_leader)

        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, orchis.entity_id))
        puppets = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90071120
        ]
        self.assertEqual(len(puppets), 2)
        self.assertTrue(all(unit.has_keyword("疾驰") for unit in puppets))
        self.assertTrue(all(unit.has_keyword("必杀") for unit in puppets))

    def test_zwei_grants_ward_to_victoria_and_evolution_puppet(self):
        engine = self.fresh(seed=19)
        zwei = _play(engine, self.repository, 10274110)
        victoria = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90074130
        )
        self.assertFalse(zwei.has_keyword("守护"))
        self.assertTrue(victoria.has_keyword("守护"))
        self.assertTrue(victoria.has_keyword("突进"))

        _enable_evolution(engine)
        engine.apply(Evolve(0, zwei.entity_id))
        puppet = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90071120
        )
        self.assertTrue(puppet.has_keyword("守护"))

    def test_sylvia_modes_and_official_lloyd_super_evolve_sequence(self):
        mode = self.fresh(seed=23)
        mode.players[0].health = 14
        _put_hand(mode, self.repository.get(10173120))
        mode.apply(PlayCard(0, 0))
        self.assertEqual(
            [option.option_id for option in mode.state.pending_choice.options],
            ["choose_one:draw_2", "choose_one:heal_4"],
        )
        _choose_option(mode, "choose_one:heal_4")
        self.assertEqual(mode.players[0].health, 18)

        engine = self.fresh(seed=29)
        sylvia = engine._summon_follower_to_board(
            0,
            self.repository.get(10173120),
            summon_cause="test",
        )
        lloyd = engine._summon_follower_to_board(
            1,
            self.repository.get(90074120),
            summon_cause="test",
        )
        orchis = engine._summon_follower_to_board(
            1,
            self.repository.get(10174120),
            summon_cause="test",
        )
        _enable_evolution(engine, super_evolve=True)

        engine.apply(SuperEvolve(0, sylvia.entity_id))
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [lloyd.entity_id],
        )
        _choose_entity(engine, lloyd.entity_id)
        self.assertNotIn(lloyd, engine.players[1].board)
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [orchis.entity_id],
        )
        _choose_entity(engine, orchis.entity_id)
        self.assertNotIn(orchis, engine.players[1].board)

    def test_all_six_cards_are_exact_and_tokens_have_real_producers(self):
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

        audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        tokens = {card["card_id"]: card for card in audit["cards"]}
        for card_id in TOKEN_IDS:
            with self.subTest(card_id=card_id):
                info = tokens[card_id]
                self.assertEqual(info["category"], "entry_behavior_complete")
                self.assertEqual(info["explicit_coverage"], "exact")
                self.assertTrue(info["authored_producers"])
                self.assertEqual(len(SOURCE_HASHES[card_id]), 64)


if __name__ == "__main__":
    unittest.main()
