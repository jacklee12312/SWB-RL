# -*- coding: utf-8 -*-
"""Direct contracts for the twentieth supported-card completion slice."""

from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path

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
    _parse_emblem_definition,
    _parse_operation,
)
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, TargetKind, TurnEndDestroyTiming
from swb.engine.emblem import EmblemPassive
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.faith import FaithInstance
from swb.engine.resolution import IllegalCommand
from swb.engine.state import DeathCause, DeckCard, Unit
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10214120,
    10314110,
    10354110,
    10554110,
    10574110,
    10714110,
)
SOURCE_HASHES = {
    10214120: "3b24c8aad8f9623314c3714b80fa40c1d2dd9c74950fbc401340ae1dcfe970bc",
    10314110: "0f52b4f468bd8c14cda8fc5411782b331f2a50af11bb4ba55fd75cf4519a4ede",
    10354110: "733248fef11c9e5a074865bbe7867091fd6da5cdfb22cb1ed78dc7724d734546",
    10554110: "6947b92bca4cacc47508763587a0c5483fb375472a36fa073b49e04e60b38ec9",
    10574110: "b7bd91fcc07e682ce76d96255ca9cd2a891ea31074295570def68e975604d36d",
    10714110: "ac0718c219b823ec12240ffd248aad26ef2f1a2669c88d598d682d7bcbe96d32",
}
TEST_EVIDENCE = (
    "tests/test_real_final_supported_completion_twentieth_batch.py"
)


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _enable_super_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = (
        engine.config.first_player_super_evolution_unlock_turn
    )
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False


def _add_sham_faith(engine, rulebook, *, value: int) -> FaithInstance:
    definition = rulebook.faith_for(10354110)
    instance = FaithInstance(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        controller=0,
        created_sequence=engine.players[0]._next_faith_sequence,
        value=value,
    )
    engine.players[0]._next_faith_sequence += 1
    engine.players[0].faiths.append(instance)
    return instance


class FinalSupportedCompletionTwentiethBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 20001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_new_generic_schema_are_strict(self):
        lymaga = self.rulebook.operations_for(
            10214120,
            Trigger.SUPER_EVOLVE,
        )[1]
        self.assertIs(lymaga.kind, EffectKind.GRANT_TURN_END_ABILITY)
        self.assertIs(
            lymaga.turn_end_ability_timing,
            TurnEndDestroyTiming.OWNER_TURN,
        )
        self.assertEqual(
            [operation.kind for operation in lymaga.granted_operations],
            [EffectKind.DAMAGE_LEADER, EffectKind.DAMAGE_UNIT],
        )

        milteo = self.rulebook.operations_for(10554110, Trigger.EVOLVE)[0]
        self.assertIs(milteo.target, TargetKind.RANDOM_ANY_UNIT)
        self.assertEqual(milteo.target_count, 6)
        self.assertTrue(milteo.exclude_source)

        positive_wheel = self.rulebook.operations_for(
            10574110,
            Trigger.TURN_START,
        )[0]
        self.assertEqual(
            positive_wheel.random_choice_history_key,
            "slaus_positive_wheel",
        )

        thestae = self.rulebook.emblem_def(
            "thestae_anathema_of_distortion"
        ).triggers[0].operations[0]
        self.assertIs(thestae.kind, EffectKind.BUFF_DECK_CARDS)
        self.assertEqual(thestae.deck_filter.card_type, "随从")

        milteo_emblem = self.rulebook.emblem_def("milteo_luzen_truth")
        self.assertEqual(
            milteo_emblem.passives,
            frozenset({
                EmblemPassive.SUPPRESS_FOLLOWER_FANFARE,
                EmblemPassive.SUPPRESS_FOLLOWER_ENHANCE,
                EmblemPassive.AUTO_EVOLVE_PLAYED_FOLLOWERS,
            }),
        )

        invalid_operations = (
            (
                {
                    "kind": "grant_turn_end_ability",
                    "target": "self",
                    "operations": [
                        {
                            "kind": "damage_unit",
                            "target": "self",
                            "amount": 1,
                        }
                    ],
                },
                "requires owner_turn or opponent_turn",
            ),
            (
                {
                    "kind": "random_choice",
                    "target": "own_leader",
                    "amount": 1,
                    "history_key": 3,
                    "options": [
                        {
                            "id": "a",
                            "operations": [
                                {
                                    "kind": "draw",
                                    "target": "own_leader",
                                    "amount": 1,
                                }
                            ],
                        },
                        {
                            "id": "b",
                            "operations": [
                                {
                                    "kind": "heal_leader",
                                    "target": "own_leader",
                                    "amount": 1,
                                }
                            ],
                        },
                    ],
                },
                "non-empty string",
            ),
            (
                {
                    "kind": "buff_deck_cards",
                    "target": "own_leader",
                    "amount": 1,
                    "secondary_amount": 1,
                    "card_type_filter": "法术",
                },
                "requires card_type_filter='随从'",
            ),
            (
                {
                    "kind": "grant_faith_mode_selection_bonus",
                    "target": "own_leader",
                    "faith_id": "x",
                    "amount": 0,
                },
                "positive integer",
            ),
            (
                {
                    "kind": "buff_unit",
                    "target": "self",
                    "secondary_amount": {
                        "type": "negate",
                        "values": [
                            {"type": "constant", "value": 1},
                            {"type": "constant", "value": 2},
                        ],
                    },
                },
                "requires exactly one value",
            ),
        )
        for raw, message in invalid_operations:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, message):
                    _parse_operation(raw, "test.json", 1)

        with self.assertRaisesRegex(ValueError, "invalid emblem passive"):
            _parse_emblem_definition(
                {
                    "id": "bad",
                    "source_card_id": 1,
                    "passives": ["card_id_specific_magic"],
                },
                "test.json",
                _parse_operation,
            )

    def test_lymaga_multi_target_choice_is_atomic_and_granted_ability_survives_source(self):
        engine = self.fresh(seed=21)
        first = _put_unit(engine, 1, _card(991001, attack=2, life=5))
        second = _put_unit(engine, 1, _card(991002, attack=2, life=5))
        first.can_attack = second.can_attack = True
        source = _play(engine, self.repository, 10214120)

        request = engine.state.pending_choice
        self.assertEqual(request.target_count, 2)
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(request.player_index, "entity:missing"))
        self.assertEqual(before, engine.deterministic_fingerprint())

        _choose_entity(engine, first.entity_id)
        self.assertIsNotNone(engine.state.pending_choice)
        _choose_entity(engine, second.entity_id)
        self.assertFalse(first.can_attack_units)
        self.assertFalse(second.can_attack_leader)

        _enable_super_evolution(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose_entity(engine, first.entity_id)
        _choose_entity(engine, second.entity_id)
        self.assertEqual(
            (
                len(first.granted_turn_end_abilities),
                len(second.granted_turn_end_abilities),
            ),
            (1, 1),
        )

        source.health = 0
        second.health = 0
        engine._stabilize()
        engine.apply(EndTurn(0))
        before_health = engine.players[1].health
        first_before = first.health
        engine.apply(EndTurn(1))
        self.assertEqual(engine.players[1].health, before_health - 1)
        self.assertEqual(first.health, first_before - 2)
        self.assertNotIn(second, engine.players[1].board)

    def test_krulle_decrease_listener_once_per_turn_and_opponent_crest(self):
        engine = self.fresh(seed=22)
        engine.players[0].health = 15
        enemies = [
            _put_unit(engine, 1, _card(991010 + index, life=5))
            for index in range(2)
        ]
        source = _play(engine, self.repository, 10314110)
        self.assertEqual([unit.health for unit in enemies], [3, 3])
        self.assertEqual(engine.players[0].health, 16)
        self.assertTrue(source.has_keyword("潜行"))
        self.assertEqual(
            sum(
                event.type is EventType.FOLLOWER_STATS_DECREASED
                for event in engine.event_history
            ),
            2,
        )

        _enable_super_evolution(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(
            engine.players[1].emblems[0].emblem_id,
            "krulle_heir_to_unkilling",
        )
        engine.apply(EndTurn(0))
        follower = _put_hand(
            engine,
            _card(991020, attack=2, life=3),
            owner=1,
        )
        engine.apply(
            PlayCard(1, engine.players[1].hand.index(follower))
        )
        summoned = next(
            unit
            for unit in engine.players[1].board
            if unit.definition.card_id == 991020
        )
        self.assertEqual((summoned.attack, summoned.health), (1, 2))

    def test_sham_faith_bonus_drives_multi_mode_choice_and_rl_mask(self):
        engine = self.fresh(seed=23)
        faith = _add_sham_faith(engine, self.rulebook, value=10)
        _play(engine, self.repository, 10354110)
        self.assertEqual(
            (faith.value, faith.mode_selection_bonus),
            (0, 1),
        )

        mode_card = _put_hand(engine, self.repository.get(10051310))
        engine.apply(
            PlayCard(0, engine.players[0].hand.index(mode_card))
        )
        request = engine.state.pending_choice
        self.assertEqual(request.target_count, 2)
        first_option = request.options[0]
        engine.apply(Choose(0, first_option.option_id))
        self.assertEqual(
            len(engine.state.pending_choice.selected_options),
            1,
        )
        engine.apply(
            Choose(0, engine.state.pending_choice.options[0].option_id)
        )
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(faith.value, 2)
        self.assertTrue(any(
            event.type is EventType.MODE_SELECTED and event.amount == 2
            for event in engine.event_history
        ))

        deck = [_card(992000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            list(deck),
            class_a=1,
            class_b=1,
            seed=2301,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            observation_version="v3",
        )
        env.reset(seed=2301)
        core = env.core
        for player in core.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        env_faith = _add_sham_faith(core, self.rulebook, value=0)
        env_faith.mode_selection_bonus = 1
        env_mode = _put_hand(core, self.repository.get(10051310))
        core.apply(PlayCard(0, core.players[0].hand.index(env_mode)))
        env.invalidate_cache(reason="test pending mode")
        mask = env.action_mask()
        self.assertEqual(
            sum(mask[env.CHOICE_OFFSET:env.CHOICE_OFFSET + env.MAX_CHOICE_OPTIONS]),
            len(core.state.pending_choice.options),
        )
        env.step(env.CHOICE_OFFSET)
        second_mask = env.action_mask()
        self.assertEqual(
            sum(
                second_mask[
                    env.CHOICE_OFFSET:
                    env.CHOICE_OFFSET + env.MAX_CHOICE_OPTIONS
                ]
            ),
            len(core.state.pending_choice.options),
        )

    def test_sham_super_evolve_copy_and_hand_capacity(self):
        engine = self.fresh(seed=24)
        enemy = _put_unit(
            engine,
            1,
            _card(991030, name="stolen", attack=4, life=5),
        )
        source = _play(engine, self.repository, 10354110)
        _enable_super_evolution(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose_entity(engine, enemy.entity_id)
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertTrue(any(card.card_id == 991030 for card in engine.players[0].hand))

        full = self.fresh(seed=25)
        full_enemy = _put_unit(
            full,
            1,
            _card(991031, name="overflow-stolen", attack=4, life=5),
        )
        full_source = _play(full, self.repository, 10354110)
        for index in range(full.config.max_hand):
            _put_hand(full, _card(991100 + index))
        _enable_super_evolution(full)
        full.apply(SuperEvolve(0, full_source.entity_id))
        _choose_entity(full, full_enemy.entity_id)
        self.assertEqual(len(full.players[0].hand), full.config.max_hand)
        self.assertTrue(any(
            card.definition.card_id == 991031
            and card.entry_cause == "hand_full"
            for card in full.players[0].graveyard
        ))

    def test_milteo_reanimate_random_six_and_crest_suppresses_played_follower_text(self):
        engine = self.fresh(seed=26)
        for definition in (
            _card(991201, cost=4, attack=4, life=4),
            _card(991202, cost=2, attack=2, life=2),
        ):
            engine._record_destroyed_follower(
                0,
                definition,
                DeathCause.EFFECT_DESTROY,
            )
        source = _play(engine, self.repository, 10554110)
        self.assertEqual(
            {
                unit.definition.card_id
                for unit in engine.players[0].board
            },
            {10554110, 991201, 991202},
        )
        for index in range(3):
            _put_unit(engine, 1, _card(991210 + index, life=4))
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(engine.players[0].board, [source])
        self.assertEqual(engine.players[1].board, [])

        crest = self.fresh(seed=27)
        crest_source = _play(crest, self.repository, 10554110)
        _enable_super_evolution(crest)
        crest.apply(SuperEvolve(0, crest_source.entity_id))
        emblem = crest.players[0].emblems[0]
        self.assertEqual(emblem.emblem_id, "milteo_luzen_truth")

        points = crest.players[0].evolution_points
        max_health = crest.players[0].max_health
        crest.players[0].mana = 10
        zooey = _put_hand(crest, self.repository.get(10444120))
        crest.apply(
            PlayCard(
                0,
                crest.players[0].hand.index(zooey),
                mode_id="enhance_10",
            )
        )
        played = next(
            unit
            for unit in crest.players[0].board
            if unit.definition.card_id == 10444120
        )
        self.assertTrue(played.evolved)
        self.assertEqual(crest.players[0].evolution_points, points)
        self.assertEqual(crest.players[0].max_health, max_health)
        self.assertEqual(crest.players[0].max_mana, 10)

    def test_slaus_random_options_are_seeded_non_repeating_and_crest_banishes_source(self):
        def sequence(seed: int):
            engine = self.fresh(seed=seed)
            engine.players[0].health = 10
            tracked = _put_hand(engine, _card(991300, cost=5))
            source = _play(engine, self.repository, 10574110)
            selected = []
            for _ in range(3):
                engine.apply(EndTurn(0))
                engine.apply(EndTurn(1))
                event = next(
                    event
                    for event in reversed(engine.event_history)
                    if event.type is EventType.RANDOM_CHOICES_SELECTED
                    and event.source_id == source.entity_id
                )
                selected.append(event.metadata["option_indices"][0])
            return selected, source.random_choice_history, tracked.current_cost

        first = sequence(28)
        second = sequence(28)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first[0])), 3)
        self.assertEqual(
            first[1]["slaus_positive_wheel"],
            tuple(first[0]),
        )

        evolved = self.fresh(seed=29)
        source = _play(evolved, self.repository, 10574110)
        _enable_evolution(evolved)
        evolved.apply(Evolve(0, source.entity_id))
        evolved.apply(EndTurn(0))
        self.assertNotIn(source, evolved.players[0].board)
        self.assertTrue(any(
            card.card_id == 10574110 for card in evolved.players[0].banished
        ))
        self.assertEqual(
            evolved.players[1].emblems[0].emblem_id,
            "slaus_negative_wheel",
        )
        first_negative = evolved.players[1].emblems[0]
        evolved.apply(EndTurn(1))
        evolved.apply(EndTurn(0))
        activated = first_negative.random_choice_history[
            "slaus_negative_wheel"
        ]
        self.assertEqual(len(activated), len(set(activated)))

    def test_thestae_combo_debuff_and_deck_buff_persists_to_hand_without_leak(self):
        engine = self.fresh(seed=30)
        enemy = _put_unit(engine, 1, _card(991400, attack=1, life=8))
        source = _play(engine, self.repository, 10714110)
        _choose_entity(engine, enemy.entity_id)
        self.assertEqual(enemy.health, 5)
        self.assertEqual(engine.players[0].cards_played_this_turn, 2)

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        follower = _card(991401, attack=2, life=3)
        spell = _card(
            991402,
            card_type="法术",
            attack=None,
            life=None,
        )
        engine.players[0].deck = [spell, follower]
        engine.players[0].cards_played_this_turn = 3
        engine.apply(EndTurn(0))
        self.assertIsInstance(engine.players[0].deck[-1], DeckCard)
        buffed = engine.players[0].deck[-1]
        self.assertEqual((buffed.attack, buffed.life), (3, 4))
        self.assertNotIsInstance(engine.players[0].deck[0], DeckCard)

        engine.apply(EndTurn(1))
        drawn = next(
            card
            for card in engine.players[0].hand
            if card.card_id == 991401
        )
        self.assertEqual((drawn.attack, drawn.life), (3, 4))

        left = self.fresh(seed=31)
        right = self.fresh(seed=31)
        left.players[0].deck = [DeckCard(
            definition=follower,
            stat_modifiers=list(buffed.stat_modifiers),
        )]
        right.players[0].deck = [follower]
        left.event_history.clear()
        right.event_history.clear()
        vocabulary = [follower.card_id]
        left_env = ShadowverseEnv(
            [_card(993000 + i) for i in range(40)],
            [_card(994000 + i) for i in range(40)],
            class_a=1,
            class_b=1,
            seed=31,
            rulebook=self.rulebook,
            card_vocabulary=vocabulary,
            observation_version="v3",
        )
        right_env = ShadowverseEnv(
            [_card(993000 + i) for i in range(40)],
            [_card(994000 + i) for i in range(40)],
            class_a=1,
            class_b=1,
            seed=31,
            rulebook=self.rulebook,
            card_vocabulary=vocabulary,
            observation_version="v3",
        )
        left_env.core = left
        right_env.core = right
        left_obs = left_env.observation(perspective=1)
        right_obs = right_env.observation(perspective=1)
        for key in left_obs:
            self.assertTrue((left_obs[key] == right_obs[key]).all(), key)

    def test_observation_exposes_new_public_runtime_without_raw_entity_ids(self):
        deck = [_card(995000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            list(deck),
            class_a=1,
            class_b=1,
            seed=32,
            rulebook=self.rulebook,
            observation_version="v2",
            card_vocabulary=[card.card_id for card in deck],
        )
        env.reset(seed=32)
        core = env.core
        for player in core.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        unit = _put_unit(core, 0, _card(995100))
        unit.random_choice_history["wheel"] = (1, 2)
        faith = _add_sham_faith(core, self.rulebook, value=4)
        faith.mode_selection_bonus = 1
        core._add_emblem_to_player(
            0,
            self.rulebook.emblem_def("slaus_negative_wheel"),
            10574110,
        )
        core.players[0].emblems[0].random_choice_history["wheel"] = (0, 2)
        env.invalidate_cache(reason="test public runtime")
        observation = env.observation()
        self.assertEqual(len(observation["own_hand_runtime"]), 126)
        self.assertEqual(len(observation["public_board_runtime"]), 230)
        self.assertEqual(
            observation["leader_area"]["faith_mode_selection_bonuses"][0],
            1,
        )
        emblem_runtime = observation["leader_area"][
            "emblem_random_choice_runtime"
        ][:5]
        self.assertEqual(emblem_runtime, (0.5, 1.0, 0.0, 1.0, 0.0))
        self.assertNotIn(unit.entity_id, observation["public_board_runtime"])


class FinalSupportedCompletionTwentiethAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")

    def test_database_contract_and_source_hashes(self):
        repository = CardRepository("data/cards.sqlite3")
        expected = {
            10214120: (1, 7, 7, 7),
            10314110: (1, 4, 1, 3),
            10354110: (5, 2, 2, 2),
            10554110: (5, 7, 3, 3),
            10574110: (7, 3, 0, 2),
            10714110: (1, 4, 3, 3),
        }
        for card_id, contract in expected.items():
            card = repository.get(card_id)
            self.assertEqual(
                (
                    card.class_id,
                    card.cost,
                    card.attack,
                    card.life,
                ),
                contract,
            )
            self.assertTrue(card.is_collectible)

        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            connection.row_factory = sqlite3.Row
            source_map = _load_source_text_map(connection)
        self.assertEqual(
            {
                card_id: _source_text_sha256(source_map[card_id])
                for card_id in CARD_IDS
            },
            SOURCE_HASHES,
        )

    def test_clause_token_and_coverage_audits_are_consistent(self):
        report = _build_coverage_report(
            "data/cards.sqlite3",
            "data/rules",
        )
        token_audit = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
        )

        self.assertEqual(
            report["summary"]["coverage_counts"],
            {
                "covered_exact": 735,
                "token_or_non_collectible": 91,
                "supported_missing_rule": 0,
            },
        )
        self.assertEqual(
            report["summary"]["clause_audit_counts"],
            {
                "mapped_exact": 735,
                "unverified_exact": 0,
                "partial": 0,
                "missing_rule": 0,
                "missing_primitive": 0,
                "text_unclear": 0,
                "token_separate_audit": 91,
            },
        )
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertEqual(token_audit["summary"]["total"], 91)
        self.assertEqual(
            token_audit["summary"]["categories"],
            {
                "entry_behavior_complete": 91,
                "entry_behavior_partial": 0,
                "database_only_no_entry": 0,
                "text_unclear": 0,
                "external_blocker": 0,
            },
        )

        audit = json.loads(
            Path("data/audits/rule_clauses.json").read_text(encoding="utf-8")
        )
        entries = {
            entry["card_id"]: entry
            for entry in audit["cards"]
            if entry["card_id"] in CARD_IDS
        }
        self.assertEqual(set(entries), set(CARD_IDS))
        for card_id, entry in entries.items():
            self.assertEqual(entry["source_text_sha256"], SOURCE_HASHES[card_id])
            self.assertIn(TEST_EVIDENCE, entry["test_evidence"])


if __name__ == "__main__":
    unittest.main()
