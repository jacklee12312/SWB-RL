# -*- coding: utf-8 -*-
"""Config-aware super-evolution unlock condition and exact real-card audits."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Attack, Choose, PlayCard
from swb.engine.conditions import EvalContext, evaluate_condition
from swb.engine.effects import (
    Condition,
    ConditionType,
    EffectKind,
    EffectOperation,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import DamageType, GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


CARD_IDS = (10121150, 10202110, 10401120, 10471110, 10872310)
SOURCE_HASHES = {
    10121150: "2cfb99516f0af7a837597a1e6b2a6e79ef3c1e16942a7806fd45ca45ee5bef30",
    10202110: "0a2aa127357c5a3d81e8c46892ccc4af6d341e8a5c606a73e9e57be42ef62620",
    10401120: "cc90b3c88de83e53e88f05facb0d16fc3e3f1c72eb66cb72d7cd22fa51c62bdf",
    10471110: "c1ce72d05501d1061ff0f607e20862cb3ff4fb0ab277fd046f511e7710baeaa6",
    10872310: "9a093516c1d1ffd51dfe0ec29e4e3672a201735088c27378bb16dc555ed866c9",
}
STRUCTURED_EVIDENCE = {
    10121150: {
        "triggers": ["fanfare", "intrinsic_keywords"],
        "effect_kinds": ["conditional", "add_keyword", "keyword:\u7a81\u8fdb"],
    },
    10202110: {
        "triggers": ["fanfare", "intrinsic_keywords"],
        "effect_kinds": ["conditional", "buff_unit", "keyword:\u5b88\u62a4"],
    },
    10401120: {
        "triggers": ["fanfare"],
        "effect_kinds": ["conditional", "evolve_unit"],
    },
    10471110: {
        "triggers": ["fanfare", "intrinsic_keywords"],
        "effect_kinds": ["conditional", "add_keyword", "keyword:\u75be\u9a70"],
    },
    10872310: {
        "triggers": ["play"],
        "effect_kinds": ["draw", "conditional", "heal_leader"],
    },
}


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=overrides.get("class_id", 0),
        class_name=overrides.get("class_name", "\u4e2d\u7acb"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "\u968f\u4ece"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 3),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _make_engine(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 907,
    config: GameConfig | None = None,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=2,
        class_b=2,
        seed=seed,
        rulebook=rulebook,
        card_resolver=repository.get,
        config=config or GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    for player in engine.players:
        player.hand.clear()
        player.hand_entity_ids.clear()
        player.max_mana = player.mana = 10
    return engine


def _put_in_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int = 0,
) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    player = engine.players[player_index]
    player.hand.append(card)
    player.hand_entity_ids.append(card.entity_id)
    return card


def _play_real(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
    *,
    player_index: int = 0,
) -> Unit | None:
    card = _put_in_hand(
        engine,
        repository.get(card_id),
        player_index=player_index,
    )
    player = engine.players[player_index]
    engine.apply(PlayCard(player_index, player.hand.index(card)))
    if card.definition.card_type == "\u6cd5\u672f":
        return None
    return next(
        unit
        for unit in player.board
        if unit.definition.card_id == card_id
    )


def _add_enemy(engine: GameEngine, card_id: int, *, life: int = 8) -> Unit:
    unit = Unit.summon(
        _card(card_id, attack=0, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[1].board.append(unit)
    return unit


class SuperEvolutionUnlockConditionCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository("data/cards.sqlite3")

    def test_direct_condition_evaluates_controller_and_opponent_flags(self):
        engine = _make_engine(RuleBook(), self.repository, seed=911)
        context = EvalContext(
            controller=0,
            players=engine.players,
            controller_super_evolution_unlocked=True,
            opponent_super_evolution_unlocked=False,
        )
        self.assertTrue(evaluate_condition(
            Condition(ConditionType.CONTROLLER_SUPER_EVOLUTION_UNLOCKED),
            context,
        ))
        self.assertFalse(evaluate_condition(
            Condition(ConditionType.OPPONENT_SUPER_EVOLUTION_UNLOCKED),
            context,
        ))

    def test_engine_context_uses_configured_asymmetric_unlock_turns(self):
        engine = _make_engine(
            RuleBook(),
            self.repository,
            seed=919,
            config=GameConfig(
                validate_invariants=True,
                first_player_super_evolution_unlock_turn=3,
                second_player_super_evolution_unlock_turn=2,
            ),
        )
        engine.players[0].turns_started = 2
        engine.players[1].turns_started = 1
        before = engine._eval_context(0)
        self.assertFalse(before.controller_super_evolution_unlocked)
        self.assertFalse(before.opponent_super_evolution_unlocked)

        engine.players[0].turns_started = 3
        engine.players[1].turns_started = 2
        first = engine._eval_context(0)
        second = engine._eval_context(1)
        self.assertTrue(first.controller_super_evolution_unlocked)
        self.assertTrue(first.opponent_super_evolution_unlocked)
        self.assertTrue(second.controller_super_evolution_unlocked)
        self.assertTrue(second.opponent_super_evolution_unlocked)

    def test_target_dependent_condition_uses_unlock_context_for_candidates(self):
        spell = _card(999901, card_type="\u6cd5\u672f", attack=None, life=None)
        operation = EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.ENEMY_UNIT,
            amount=1,
            conditions=(
                Condition(ConditionType.CONTROLLER_SUPER_EVOLUTION_UNLOCKED),
                Condition(ConditionType.TARGET_HEALTH_AT_LEAST, value=5),
            ),
        )
        rulebook = RuleBook((CardRule(999901, Trigger.PLAY, (operation,)),))
        engine = _make_engine(rulebook, self.repository, seed=929)
        low = _add_enemy(engine, 9001, life=4)
        valid = _add_enemy(engine, 9002, life=5)
        engine.players[0].turns_started = (
            engine.config.first_player_super_evolution_unlock_turn
        )
        _put_in_hand(engine, spell)
        engine.apply(PlayCard(0, 0))

        request = engine.state.pending_choice
        self.assertEqual(
            [option.entity_id for option in request.options],
            [valid.entity_id],
        )
        engine.apply(Choose(0, request.options[0].option_id))
        self.assertEqual((low.health, valid.health), (4, 4))


class DatabaseAndCoverageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_cards_match_stats_abilities_and_have_no_refs_or_modes(self):
        expected = {
            10121150: ("\u5f02\u7aef\u6b66\u58eb", "\u7687\u5bb6\u62a4\u536b", "\u968f\u4ece", 2, 2, 1),
            10202110: ("\u5973\u4ec6\u5929\u4f7f\u00b7\u5207\u857e\u5854", "\u4e2d\u7acb", "\u968f\u4ece", 2, 2, 2),
            10401120: ("\u4eb2\u7231\u7684\u642d\u6863\u00b7\u78a7", "\u4e2d\u7acb", "\u968f\u4ece", 2, 2, 2),
            10471110: ("\u591c\u738b\u518d\u8d77\u00b7\u7fd4", "\u8d85\u8d8a\u8005", "\u968f\u4ece", 3, 2, 1),
            10872310: ("\u7eaf\u51c0\u65e0\u57a2\u7684\u65e5\u5e38", "\u8d85\u8d8a\u8005", "\u6cd5\u672f", 3, None, None),
        }
        abilities = {
            10121150: {("\u5165\u573a\u66f2", "\u5165\u573a\u66f2"), ("\u5fc5\u6740", "\u6bc1\u706d"), ("\u7a81\u8fdb", "\u7a81\u8fdb")},
            10202110: {("\u5165\u573a\u66f2", "\u5165\u573a\u66f2"), ("\u5b88\u62a4", "\u5b88\u62a4")},
            10401120: {("\u5165\u573a\u66f2", "\u5165\u573a\u66f2")},
            10471110: {("\u5165\u573a\u66f2", "\u5165\u573a\u66f2"), ("\u5c4f\u969c", "\u5c4f\u969c"), ("\u75be\u9a70", "\u75be\u9a70")},
            10872310: set(),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT json_extract(c.raw_json, '$.name_chs'),
                               cl.class_name, cl.type_name, c.cost, c.attack, c.life,
                               cs.is_collectible
                        FROM cards c
                        JOIN card_sets cs ON cs.id = c.card_set_id
                        JOIN card_localizations cl
                          ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                        WHERE c.card_id = ?
                        """,
                        (card_id,),
                    ).fetchone()
                    self.assertEqual(row[:6], values)
                    self.assertEqual(row[6], 1)
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT ability_keyword, raw_keyword FROM card_abilities "
                            "WHERE card_id = ?",
                            (card_id,),
                        )),
                        abilities[card_id],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM card_references WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_all_five_cards_are_exact_with_hash_and_structured_evidence(self):
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
                    info["clause_audit"]["structured_evidence"],
                    STRUCTURED_EVIDENCE[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_super_evolution_unlock_condition.py"],
                )


class RealSuperEvolutionUnlockBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 937) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def test_rule_schema_uses_new_condition_and_keyword_provenance(self):
        for card_id, trigger in (
            (10121150, Trigger.FANFARE),
            (10202110, Trigger.FANFARE),
            (10401120, Trigger.FANFARE),
            (10471110, Trigger.FANFARE),
        ):
            operation = self.rulebook.operations_for(card_id, trigger)[0]
            self.assertIs(operation.kind, EffectKind.CONDITIONAL)
            self.assertEqual(
                [condition.type for condition in operation.conditions],
                [ConditionType.CONTROLLER_SUPER_EVOLUTION_UNLOCKED],
            )
        spell = self.rulebook.operations_for(10872310, Trigger.PLAY)
        self.assertEqual([operation.kind for operation in spell], [
            EffectKind.DRAW,
            EffectKind.CONDITIONAL,
        ])
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10121150),
            frozenset({"\u5fc5\u6740"}),
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10471110),
            frozenset({"\u5c4f\u969c"}),
        )

    def test_heretic_samurai_gains_bane_only_at_unlock_and_bane_resolves(self):
        before = self.fresh_engine(seed=941)
        before.players[0].turns_started = (
            before.config.first_player_super_evolution_unlock_turn - 1
        )
        source = _play_real(before, self.repository, 10121150)
        self.assertTrue(source.has_keyword("\u7a81\u8fdb"))
        self.assertFalse(source.has_keyword("\u5fc5\u6740"))

        unlocked = self.fresh_engine(seed=947)
        unlocked.players[0].turns_started = (
            unlocked.config.first_player_super_evolution_unlock_turn
        )
        target = _add_enemy(unlocked, 9100, life=10)
        source = _play_real(unlocked, self.repository, 10121150)
        self.assertTrue(source.has_keyword("\u7a81\u8fdb"))
        self.assertTrue(source.has_keyword("\u5fc5\u6740"))
        unlocked.apply(Attack(0, source.entity_id, target.entity_id))
        self.assertNotIn(target, unlocked.players[1].board)

    def test_maid_angel_health_buff_and_ward_obey_first_player_boundary(self):
        before = self.fresh_engine(seed=953)
        before.players[0].turns_started = (
            before.config.first_player_super_evolution_unlock_turn - 1
        )
        source = _play_real(before, self.repository, 10202110)
        self.assertEqual((source.attack, source.health), (2, 2))
        self.assertTrue(source.has_guard)

        unlocked = self.fresh_engine(seed=953)
        unlocked.players[0].turns_started = (
            unlocked.config.first_player_super_evolution_unlock_turn
        )
        source = _play_real(unlocked, self.repository, 10202110)
        self.assertEqual((source.attack, source.health, source.max_health), (2, 5, 5))
        self.assertTrue(source.has_guard)

    def test_second_player_uses_its_own_sixth_turn_boundary(self):
        engine = self.fresh_engine(seed=967)
        engine.state.active_player = 1
        engine.players[1].turns_started = (
            engine.config.second_player_super_evolution_unlock_turn
        )
        source = _play_real(
            engine,
            self.repository,
            10202110,
            player_index=1,
        )
        self.assertEqual((source.attack, source.health), (2, 5))

    def test_dear_partner_effect_evolves_without_spending_ep_only_at_unlock(self):
        before = self.fresh_engine(seed=971)
        before.players[0].turns_started = (
            before.config.first_player_super_evolution_unlock_turn - 1
        )
        source = _play_real(before, self.repository, 10401120)
        self.assertFalse(source.evolved)

        unlocked = self.fresh_engine(seed=977)
        unlocked.players[0].turns_started = (
            unlocked.config.first_player_super_evolution_unlock_turn
        )
        ep_before = unlocked.players[0].evolution_points
        source = _play_real(unlocked, self.repository, 10401120)
        self.assertTrue(source.evolved)
        self.assertEqual((source.attack, source.health), (4, 4))
        self.assertEqual(unlocked.players[0].evolution_points, ep_before)
        self.assertEqual(unlocked.players[0].followers_evolved_this_match, 1)

    def test_night_king_storm_is_intrinsic_and_barrier_is_conditional(self):
        before = self.fresh_engine(seed=983)
        before.players[0].turns_started = (
            before.config.first_player_super_evolution_unlock_turn - 1
        )
        source = _play_real(before, self.repository, 10471110)
        self.assertTrue(source.can_attack_leader)
        self.assertEqual(source.barrier_charges, 0)

        unlocked = self.fresh_engine(seed=991)
        unlocked.players[0].turns_started = (
            unlocked.config.first_player_super_evolution_unlock_turn
        )
        source = _play_real(unlocked, self.repository, 10471110)
        health_before = source.health
        self.assertTrue(source.can_attack_leader)
        self.assertEqual(source.barrier_charges, 1)
        unlocked.apply_damage(None, source, 5, DamageType.EFFECT, controller=1)
        self.assertEqual(source.health, health_before)
        self.assertEqual(source.barrier_charges, 0)

    def test_pure_daily_life_always_draws_and_heals_only_at_unlock_with_cap(self):
        before = self.fresh_engine(seed=997)
        before.players[0].turns_started = (
            before.config.first_player_super_evolution_unlock_turn - 1
        )
        before.players[0].health = 10
        deck_before = len(before.players[0].deck)
        _play_real(before, self.repository, 10872310)
        self.assertEqual(len(before.players[0].deck), deck_before - 2)
        self.assertEqual(before.players[0].health, 10)

        unlocked = self.fresh_engine(seed=1009)
        unlocked.players[0].turns_started = (
            unlocked.config.first_player_super_evolution_unlock_turn
        )
        unlocked.players[0].health = 19
        deck_before = len(unlocked.players[0].deck)
        _play_real(unlocked, self.repository, 10872310)
        self.assertEqual(len(unlocked.players[0].deck), deck_before - 2)
        self.assertEqual(unlocked.players[0].health, 20)

    def test_unlocked_real_sequence_is_deterministic(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh_engine(seed=1013)
            engine.players[0].turns_started = (
                engine.config.first_player_super_evolution_unlock_turn
            )
            _play_real(engine, self.repository, 10401120)
            _play_real(engine, self.repository, 10471110)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_auto_evolution_result_is_visible_through_rl_play_mask(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(3000, 3040)],
            [_card(card_id) for card_id in range(4000, 4040)],
            class_a=2,
            class_b=2,
            seed=1019,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=1019)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].max_mana = env.players[0].mana = 10
        env.players[0].turns_started = (
            env.core.config.first_player_super_evolution_unlock_turn
        )
        _put_in_hand(env.core, self.repository.get(10401120))

        play = PlayCard(0, 0)
        action = env._encode_command(play)
        self.assertIsNotNone(action)
        self.assertTrue(env.action_mask()[action])
        env.step(action)
        source = next(
            unit for unit in env.players[0].board
            if unit.definition.card_id == 10401120
        )
        self.assertTrue(source.evolved)
        self.assertEqual((source.attack, source.health), (4, 4))


if __name__ == "__main__":
    unittest.main()
