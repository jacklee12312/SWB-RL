# -*- coding: utf-8 -*-
"""Exact Enhance-faith and destroyed-Shikigami aggregation behavior."""

from __future__ import annotations

from dataclasses import replace
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, _parse_expression
from swb.engine.commands import Attack, Choose, Evolve, PlayCard, SuperEvolve
from swb.engine.conditions import evaluate_expression
from swb.engine.effects import ExprType
from swb.engine.events import EventType, GameEvent
from swb.engine.faith import FaithTrigger
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import DeathCause, DestroyedFollowerRecord
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (10624120, 10134110)
TOKEN_IDS = (90024320, 90034110, 90034120)
SOURCE_HASHES = {
    10624120: "99cea234b82589870a279861221aa332afc1fbeb6db8e02eb7de80468c8b7111",
    90024320: "77af6ca7705d3eb3bd234c37e2b4405ca97b29c4fdf3c115be6a8ac170492e28",
    10134110: "40b458a8cdb80930a186ad279df5d737794590dc33e03eb73ea251e5f9d511a3",
    90034110: "a48c82626490d58eac57033a72b3edae6f513f8221deb9e44f2fb0a9794d47b2",
    90034120: "247d2107ae14446cef7fbc4e09e04322cad29cfaba5faea37dd5a0062c236dc3",
}


def _choose_entity(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolution(engine: GameEngine, *, super_evolve: bool = False) -> None:
    player = engine.players[0]
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False
    if super_evolve:
        player.turns_started = engine.config.first_player_super_evolution_unlock_turn
        player.super_evolution_points = max(1, player.super_evolution_points)
    else:
        player.turns_started = engine.config.evolution_unlock_turn
        player.evolution_points = max(1, player.evolution_points)


class EnhanceFaithShikigamiSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def fresh(self, *, seed: int = 401) -> GameEngine:
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_destroyed_follower_aggregate_expression_parses_filter(self):
        attack = _parse_expression(
            {
                "type": "controller_destroyed_follower_base_attack_sum_this_turn",
                "filter": {"card_type": "随从", "tribe_name": "式神"},
            },
            "test",
            123,
        )
        health = _parse_expression(
            {
                "type": "controller_destroyed_follower_base_health_sum_this_turn",
                "filter": {"tribe_id": 13},
            },
            "test",
            123,
        )

        self.assertIs(
            attack.type,
            ExprType.CONTROLLER_DESTROYED_FOLLOWER_BASE_ATTACK_SUM_THIS_TURN,
        )
        self.assertEqual(attack.card_filter.tribe_name, "式神")
        self.assertEqual(health.card_filter.tribe_id, 13)

    def test_aggregate_uses_printed_stats_owner_turn_and_filter(self):
        engine = self.fresh(seed=3)
        shikigami = self.repository.get(90031130)
        non_shikigami = self.repository.get(10134110)
        engine.state.destroyed_followers = [
            DestroyedFollowerRecord(
                shikigami, 0, 1, DeathCause.COMBAT,
                destroyed_turn=engine.turn,
            ),
            DestroyedFollowerRecord(
                shikigami, 1, 2, DeathCause.COMBAT,
                destroyed_turn=engine.turn,
            ),
            DestroyedFollowerRecord(
                shikigami, 0, 3, DeathCause.COMBAT,
                destroyed_turn=0,
            ),
            DestroyedFollowerRecord(
                non_shikigami, 0, 4, DeathCause.COMBAT,
                destroyed_turn=engine.turn,
            ),
        ]
        context = engine._eval_context(0)
        attack = _parse_expression(
            {
                "type": "controller_destroyed_follower_base_attack_sum_this_turn",
                "filter": {"tribe_name": "式神"},
            },
            "test",
            123,
        )
        health = _parse_expression(
            {
                "type": "controller_destroyed_follower_base_health_sum_this_turn",
                "filter": {"tribe_name": "式神"},
            },
            "test",
            123,
        )

        self.assertEqual(evaluate_expression(attack, context), 2)
        self.assertEqual(evaluate_expression(health, context), 1)

    def test_expression_filter_is_rejected_on_unrelated_expression(self):
        with self.assertRaisesRegex(ValueError, "filter.*only valid"):
            _parse_expression(
                {"type": "source_attack", "filter": {"tribe_name": "式神"}},
                "test",
                123,
            )

    def test_faith_trigger_distinguishes_enhance_from_other_modes(self):
        enhanced = GameEvent(
            EventType.CARD_PLAYED,
            0,
            metadata={"card_id": 90024320, "mode_id": "enhance_1"},
        )
        normal = GameEvent(
            EventType.CARD_PLAYED,
            0,
            metadata={"card_id": 90024320, "mode_id": "normal"},
        )
        accelerate = GameEvent(
            EventType.CARD_PLAYED,
            0,
            metadata={"card_id": 10671110, "mode_id": "accelerate_2"},
        )
        crystallize = GameEvent(
            EventType.CARD_PLAYED,
            0,
            metadata={"card_id": 999803, "mode_id": "crystallize_2"},
        )

        self.assertTrue(self.fresh()._event_is_enhanced_card_play(enhanced))
        self.assertFalse(self.fresh()._event_is_enhanced_card_play(normal))
        self.assertFalse(self.fresh()._event_is_enhanced_card_play(accelerate))
        self.assertFalse(self.fresh()._event_is_enhanced_card_play(crystallize))
        self.assertIs(
            self.rulebook.faith_for(10624120).triggers[0].trigger,
            FaithTrigger.CARD_ENHANCED,
        )

    def test_destroyed_turn_is_fingerprinted_and_invariant_checked(self):
        engine = self.fresh(seed=5)
        definition = self.repository.get(90031130)
        engine.state.destroyed_followers.append(DestroyedFollowerRecord(
            definition,
            0,
            1,
            DeathCause.COMBAT,
            destroyed_turn=engine.turn,
        ))
        before = engine.deterministic_fingerprint()
        engine.state.destroyed_followers[0] = replace(
            engine.state.destroyed_followers[0],
            destroyed_turn=0,
        )
        self.assertNotEqual(before, engine.deterministic_fingerprint())
        engine.state.destroyed_followers[0] = replace(
            engine.state.destroyed_followers[0],
            destroyed_turn=engine.turn + 1,
        )
        with self.assertRaisesRegex(IllegalCommand, "destroyed_turn"):
            engine.assert_invariants()


class RealEnhanceFaithShikigamiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def fresh(self, *, seed: int = 401) -> GameEngine:
        return _fresh(self.rulebook, self.repository, seed=seed)

    def faith_fresh(self, *, seed: int = 401) -> GameEngine:
        deck_a = [self.repository.get(10624120)] + [
            _card(
                5000 + index,
                class_id=2,
                class_name="皇家护卫",
            )
            for index in range(39)
        ]
        deck_b = [
            _card(
                6000 + index,
                class_id=2,
                class_name="皇家护卫",
            )
            for index in range(40)
        ]
        engine = GameEngine(
            deck_a,
            deck_b,
            class_a=2,
            class_b=2,
            seed=seed,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=seed)
        for player in engine.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        return engine

    def _play_and_evolve_yidmetra(
        self,
        engine: GameEngine,
        *,
        faith_value: int,
    ):
        faith = engine.players[0].faiths[0]
        faith.value = faith_value
        _put_hand(engine, self.repository.get(10624120))
        engine.apply(PlayCard(0, 0))
        yidmetra = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10624120
        )
        _enable_evolution(engine)
        engine.apply(Evolve(0, yidmetra.entity_id))
        return faith, yidmetra

    def test_yidmetra_fanfare_and_faith_payment_gate(self):
        insufficient = self.faith_fresh(seed=7)
        faith, _ = self._play_and_evolve_yidmetra(
            insufficient,
            faith_value=4,
        )
        self.assertEqual(faith.value, 4)
        self.assertEqual(faith.granted_abilities, [])
        self.assertEqual(
            [card.definition.card_id for card in insufficient.players[0].hand],
            [90024320],
        )

        paid = self.faith_fresh(seed=11)
        faith, _ = self._play_and_evolve_yidmetra(paid, faith_value=5)
        self.assertEqual(faith.value, 0)
        self.assertEqual(
            [(ability.ability_id, ability.trigger) for ability in faith.granted_abilities],
            [("yidmetra_enhance_allied_buff", FaithTrigger.CARD_ENHANCED)],
        )

    def test_normal_depths_does_not_trigger_faith_but_enhance_does(self):
        engine = self.faith_fresh(seed=13)
        faith, yidmetra = self._play_and_evolve_yidmetra(engine, faith_value=5)
        normal_target = _put_unit(engine, 1, _card(7001, life=8))

        engine.players[0].mana = 0
        engine.apply(PlayCard(0, 0, mode_id="normal"))
        _choose_entity(engine, normal_target.entity_id)
        self.assertEqual(normal_target.health, 7)
        self.assertEqual(faith.value, 0)
        self.assertEqual((yidmetra.attack, yidmetra.health), (3, 4))

        enhanced_target = _put_unit(engine, 1, _card(7002, life=8))
        _put_hand(engine, self.repository.get(90024320))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0, mode_id="enhance_1"))
        _choose_entity(engine, enhanced_target.entity_id)

        self.assertEqual(enhanced_target.health, 5)
        self.assertEqual(faith.value, 1)
        self.assertEqual((yidmetra.attack, yidmetra.health), (4, 5))

    def test_depths_requires_enemy_target_and_illegal_play_is_atomic(self):
        for mode_id in ("normal", "enhance_1"):
            with self.subTest(mode_id=mode_id):
                engine = self.fresh(seed=17)
                _put_hand(engine, self.repository.get(90024320))
                self.assertFalse(any(
                    isinstance(command, PlayCard)
                    and command.mode_id == mode_id
                    for command in engine.legal_commands()
                ))
                before = engine.deterministic_fingerprint()
                with self.assertRaises(IllegalCommand):
                    engine.apply(PlayCard(0, 0, mode_id=mode_id))
                self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_kuon_base_summons_ordered_tokens_and_super_evolve_grants_storm(self):
        engine = self.fresh(seed=19)
        _put_hand(engine, self.repository.get(10134110))
        engine.players[0].mana = 9
        engine.apply(PlayCard(0, 0))

        self.assertEqual(
            [unit.definition.card_id for unit in engine.players[0].board],
            [10134110, 90034110, 90031140, 90031130],
        )
        kuon, celestial, demonic, paper = engine.players[0].board
        self.assertFalse(kuon.has_keyword("疾驰"))
        self.assertTrue(celestial.has_keyword("守护"))
        self.assertTrue(demonic.has_keyword("突进"))
        self.assertTrue(paper.has_keyword("突进"))

        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, kuon.entity_id))
        self.assertEqual(
            [option.entity_id for option in engine.state.pending_choice.options],
            [celestial.entity_id, demonic.entity_id, paper.entity_id],
        )
        _choose_entity(engine, celestial.entity_id)
        self.assertTrue(celestial.has_keyword("疾驰"))
        self.assertTrue(any(
            isinstance(command, Attack)
            and command.attacker_id == celestial.entity_id
            for command in engine.legal_commands()
        ))

    def _enhanced_kuon(self, *, seed: int) -> GameEngine:
        engine = self.fresh(seed=seed)
        shikigami = self.repository.get(90031130)
        old_turn = max(0, engine.turn - 1)
        engine.state.destroyed_followers = [
            DestroyedFollowerRecord(
                shikigami, 0, 1, DeathCause.COMBAT,
                destroyed_turn=engine.turn,
            ),
            DestroyedFollowerRecord(
                shikigami, 0, 2, DeathCause.COMBAT,
                destroyed_turn=old_turn,
            ),
            DestroyedFollowerRecord(
                shikigami, 1, 3, DeathCause.COMBAT,
                destroyed_turn=engine.turn,
            ),
            DestroyedFollowerRecord(
                self.repository.get(10134110), 0, 4, DeathCause.COMBAT,
                destroyed_turn=engine.turn,
            ),
        ]
        engine.state._next_death_sequence = 5
        _put_hand(engine, _card(8001, card_type="法术"))
        _put_hand(engine, self.repository.get(10134110))
        engine.apply(PlayCard(0, 0, mode_id="enhance_10"))
        return engine

    def test_kuon_enhance_destroys_shikigami_then_noble_uses_printed_totals(self):
        engine = self._enhanced_kuon(seed=23)

        self.assertEqual(
            [unit.definition.card_id for unit in engine.players[0].board],
            [10134110, 90034120],
        )
        kuon, noble = engine.players[0].board
        self.assertEqual((kuon.attack, kuon.health), (3, 3))
        self.assertEqual((noble.attack, noble.health), (12, 11))
        self.assertTrue(noble.has_keyword("守护"))
        self.assertTrue(noble.has_keyword("灵气"))
        self.assertEqual(engine.players[0].hand[0].spellboost_count, 5)
        self.assertEqual(
            [
                record.definition.card_id
                for record in engine.state.destroyed_followers[-3:]
            ],
            [90034110, 90031140, 90031130],
        )
        self.assertTrue(all(
            record.destroyed_turn == engine.turn
            for record in engine.state.destroyed_followers[-3:]
        ))

    def test_kuon_board_shortage_and_no_target_super_evolve_are_explicit(self):
        shortage = self.fresh(seed=29)
        for index in range(3):
            _put_unit(shortage, 0, _card(8100 + index))
        _put_hand(shortage, self.repository.get(10134110))
        shortage.apply(PlayCard(0, 0, mode_id="enhance_10"))
        self.assertEqual(
            [unit.definition.card_id for unit in shortage.players[0].board],
            [8100, 8101, 8102, 10134110, 90034120],
        )
        noble = shortage.players[0].board[-1]
        self.assertEqual((noble.attack, noble.health), (5, 6))

        no_target = self.fresh(seed=31)
        kuon = _put_unit(no_target, 0, self.repository.get(10134110))
        _enable_evolution(no_target, super_evolve=True)
        no_target.apply(SuperEvolve(0, kuon.entity_id))
        self.assertTrue(kuon.super_evolved)
        self.assertIsNone(no_target.state.pending_choice)

    def test_enhanced_sequence_is_seed_reproducible(self):
        first = self._enhanced_kuon(seed=37)
        second = self._enhanced_kuon(seed=37)
        self.assertEqual(
            first.deterministic_fingerprint(),
            second.deterministic_fingerprint(),
        )

    def test_all_five_cards_are_exact_and_tokens_have_real_producers(self):
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
