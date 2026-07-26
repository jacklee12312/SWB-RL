# -*- coding: utf-8 -*-
"""Direct contracts for the sixteenth fusion/combo/cost/repose slice."""

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
from swb.engine.card_rules import (
    RuleBook,
    Trigger,
    _parse_event_card_filter,
    _parse_expression,
    _parse_operation,
)
from swb.engine.commands import (
    ActivateAmulet,
    Attack,
    BeginFusion,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
    SuperEvolve,
)
from swb.engine.effects import (
    ExprType,
    EffectKind,
    TargetKind,
    TurnEndDestroyTiming,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import CostModifier, DeckCard
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _destroy_units,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10324110, 10514110, 10332210, 10342110, 10364110)
SOURCE_HASHES = {
    10324110: "11b097590e3defce45cc9430528a1e285734aa3b55142ae3cd4acec48ddb668d",
    10514110: "f9579692adc07ebe3ffec346e128c81ee50879167911cd0e247cc1ae3c06745f",
    10332210: "9a648ea728e9c42c51386d4efc24d64582aefd5e92f2d99d19b4ca96aad808ae",
    10342110: "493a0df515d67ca5fb53c4c19153f58a93bf35b3b40d1818def186fca36b1bfc",
    10364110: "557fc6c04320034f4a5888c0754bd00b940b6ab4366365e89c202f8fcc7cffa3",
}
TEST_EVIDENCE = (
    "tests/test_real_fusion_combo_cost_repose_sixteenth_batch.py"
)


def _choose_option(engine, option_id: str) -> None:
    request = engine.state.pending_choice
    engine.apply(Choose(request.player_index, option_id))


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option
        for option in request.options
        if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _enable_super_evolution(engine) -> None:
    player = engine.players[0]
    unlock_turn = (
        engine.config.first_player_super_evolution_unlock_turn
    )
    player.turns_started = unlock_turn
    player.super_evolution_points = max(
        1,
        player.super_evolution_points,
    )
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False


def _fuse_all(engine, target, materials) -> None:
    engine.apply(BeginFusion(0, target.entity_id))
    for material in materials:
        _choose_option(engine, f"hand:{material.entity_id}")
    _choose_option(engine, "fusion:confirm")


class FusionComboCostReposeSixteenthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 16001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_new_schema_are_strict(self):
        sinciro_fusion = self.rulebook.fusion_for(10324110)
        self.assertEqual(sinciro_fusion.material_filter.tribe_id, 18)
        sinciro = self.rulebook.operations_for(
            10324110,
            Trigger.FANFARE,
        )
        self.assertEqual(
            [operation.kind for operation in sinciro],
            [EffectKind.DAMAGE_UNIT, EffectKind.DAMAGE_LEADER],
        )
        self.assertTrue(all(
            operation.amount_expr.type
            is ExprType.SOURCE_FUSION_DISTINCT_NAME_COUNT
            for operation in sinciro
        ))

        wolfraud = self.rulebook.operations_for(
            10514110,
            Trigger.EVOLVE,
        )
        self.assertEqual(
            [operation.kind for operation in wolfraud],
            [EffectKind.DISCARD, EffectKind.COPY_RANDOM_ENEMY_DECK_TO_HAND],
        )
        self.assertEqual(wolfraud[1].amount, 5)

        facility = self.rulebook.operations_for(10332210, Trigger.PLAY)
        self.assertEqual(facility, ())
        self.assertEqual(self.rulebook.countdown_for(10332210), 5)
        self.assertEqual(self.rulebook.activation_for(10332210).cost, 0)
        listener = self.rulebook.listeners_for(10332210)[0]
        self.assertTrue(listener.event_filter.cost_changed)
        self.assertEqual(
            [operation.kind for operation in listener.operations],
            [EffectKind.DRAW, EffectKind.REDUCE_COUNTDOWN],
        )

        worshipper = self.rulebook.operations_for(
            10342110,
            Trigger.TURN_END,
        )[0]
        self.assertIs(worshipper.kind, EffectKind.HEAL_UNIT_AND_LEADER)
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10342110),
            ("突进", "守护"),
        )

        himeka = self.rulebook.emblem_def("himeka_heir_to_repose")
        self.assertEqual((himeka.source_card_id, himeka.countdown), (10364110, 4))
        selection = himeka.triggers[0].operations[0]
        self.assertIs(selection.kind, EffectKind.SELECT_TARGETS)
        self.assertIs(
            selection.target_count_expr.type,
            ExprType.CONTROLLER_EMBLEM_COUNT,
        )
        self.assertIs(
            himeka.triggers[0].operations[2].turn_end_banish_timing,
            TurnEndDestroyTiming.OWNER_TURN,
        )

        expression = _parse_expression(
            {"type": "source_fusion_distinct_name_count"},
            "test.json",
            1,
        )
        self.assertIs(
            expression.type,
            ExprType.SOURCE_FUSION_DISTINCT_NAME_COUNT,
        )
        event_filter = _parse_event_card_filter(
            {"card_type": "随从", "cost_changed": True},
            "test.json",
            1,
        )
        self.assertTrue(event_filter.cost_changed)
        with self.assertRaises(ValueError):
            _parse_event_card_filter(
                {"cost_changed": 1},
                "test.json",
                1,
            )
        for raw in (
            {
                "kind": "copy_random_enemy_deck_to_hand",
                "target": "enemy_leader",
                "amount": 5,
            },
            {
                "kind": "copy_random_enemy_deck_to_hand",
                "target": "own_leader",
                "amount": 0,
            },
            {
                "kind": "heal_unit_and_leader",
                "target": "enemy_unit",
                "amount": 3,
            },
            {
                "kind": "heal_unit_and_leader",
                "target": "self",
            },
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test.json", 1)

    def test_sinciro_counts_differently_named_loot_for_fanfare_and_super_evolve(self):
        engine = self.fresh(seed=3)
        source = _put_hand(engine, self.repository.get(10324110))
        materials = [
            _put_hand(engine, self.repository.get(90021310)),
            _put_hand(engine, self.repository.get(90021310)),
            _put_hand(engine, self.repository.get(90021320)),
        ]
        non_loot = _put_hand(engine, _card(996001, name="not-loot"))
        engine.apply(BeginFusion(0, source.entity_id))
        options = {
            option.entity_id
            for option in engine.state.pending_choice.options
            if option.entity_id is not None
        }
        self.assertEqual(options, {material.entity_id for material in materials})
        _choose_option(engine, "fusion:cancel")

        _fuse_all(engine, source, materials)
        enemies = [
            _put_unit(engine, 1, _card(996010 + index, life=6))
            for index in range(2)
        ]
        engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
        source_unit = next(
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == 10324110
        )
        self.assertEqual([unit.health for unit in enemies], [4, 4])
        self.assertEqual(engine.players[1].health, 18)
        self.assertIn(non_loot, engine.players[0].hand)

        new_enemy = _put_unit(engine, 1, _card(996020, life=5))
        _enable_super_evolution(engine)
        engine.apply(SuperEvolve(0, source_unit.entity_id))
        self.assertEqual(new_enemy.health, 3)
        self.assertEqual(engine.players[1].health, 16)
        self.assertEqual(
            len({
                record.definition.name
                for record in engine.players[0].fusion_materials
            }),
            2,
        )
        engine.assert_invariants()

        plain = self.fresh(seed=5)
        enemy = _put_unit(plain, 1, _card(996030, life=4))
        _play(plain, self.repository, 10324110)
        self.assertEqual(enemy.health, 4)
        self.assertEqual(plain.players[1].health, 20)

    def test_wolfraud_combo_includes_current_play_and_evolve_copies_exact_deck_cards(self):
        combo = self.fresh(seed=7)
        combo.players[0].cards_played_this_turn = 2
        source = _play(combo, self.repository, 10514110)
        self.assertEqual((source.attack, source.max_health), (4, 4))
        self.assertEqual(combo.players[0].cards_played_this_turn, 3)

        engine = self.fresh(seed=11)
        source = _put_unit(engine, 0, self.repository.get(10514110))
        _put_hand(engine, _card(996100))
        _put_hand(engine, _card(996101))
        modifier_id = engine._allocate_modifier_id()
        modified = DeckCard(
            definition=_card(996110, cost=8),
            cost_modifiers=[
                CostModifier(
                    modifier_id=modifier_id,
                    mode="set",
                    amount=2,
                    duration="permanent",
                )
            ],
        )
        enemy_deck = [
            modified,
            _card(996111, cost=3),
            _card(996112, cost=4),
            _card(996113, cost=5),
            _card(996114, cost=6),
        ]
        engine.players[1].deck = list(enemy_deck)
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))

        self.assertEqual(
            {card.card_id for card in engine.players[0].hand},
            {996110, 996111, 996112, 996113, 996114},
        )
        self.assertEqual(
            [card.card_id for card in engine.players[1].deck],
            [card.card_id for card in enemy_deck],
        )
        copied_modified = next(
            card for card in engine.players[0].hand if card.card_id == 996110
        )
        self.assertEqual(copied_modified.current_cost, 2)
        self.assertNotEqual(
            copied_modified.cost_modifiers[0].modifier_id,
            modifier_id,
        )
        copy_events = [
            event
            for event in engine.event_history
            if event.type is EventType.CARD_ADDED_TO_HAND
            and event.metadata.get("copied_from_zone") == "enemy_deck"
        ]
        self.assertEqual(len(copy_events), 5)
        self.assertTrue(all(
            event.metadata["exact_copy"]
            and not event.metadata["revealed"]
            for event in copy_events
        ))
        engine.assert_invariants()

    def test_wolfraud_empty_enemy_deck_skips_rng_and_copy_respects_hand_capacity(self):
        empty = self.fresh(seed=13)
        source = _put_unit(empty, 0, self.repository.get(10514110))
        _put_hand(empty, _card(996120))
        empty.players[1].deck = []
        _enable_evolution(empty)
        rng_before = empty.random.getstate()
        empty.apply(Evolve(0, source.entity_id))
        self.assertEqual(empty.random.getstate(), rng_before)
        self.assertEqual(empty.players[0].hand, [])

        full = self.fresh(seed=17)
        source = _put_unit(full, 0, self.repository.get(10514110))
        for index in range(full.config.max_hand - 1):
            _put_hand(full, _card(996130 + index))
        full.players[1].deck = [
            _card(996150 + index)
            for index in range(5)
        ]
        copy_operation = self.rulebook.operations_for(
            10514110,
            Trigger.EVOLVE,
        )[1]
        graveyard_before = len(full.players[0].graveyard)
        full._start_effects(
            source.definition,
            source.entity_id,
            (copy_operation,),
            label="容量测试",
        )
        self.assertEqual(len(full.players[0].hand), full.config.max_hand)
        self.assertEqual(
            len(full.players[0].graveyard) - graveyard_before,
            4,
        )
        full.assert_invariants()

    def test_truth_facility_only_listens_to_changed_cost_and_engage_binds_one_follower(self):
        engine = self.fresh(seed=19)
        facility = _play(engine, self.repository, 10332210)
        self.assertEqual(facility.countdown, 5)
        engine.players[0].deck = [_card(996200)]
        changed = _put_hand(engine, _card(996201, cost=2))
        changed.cost_modifiers.append(CostModifier(
            modifier_id=engine._allocate_modifier_id(),
            mode="subtract",
            amount=1,
            duration="permanent",
        ))
        engine.apply(PlayCard(0, engine.players[0].hand.index(changed)))
        self.assertEqual(facility.countdown, 4)
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [996200],
        )
        changed_play = next(
            event
            for event in engine.event_history
            if event.type is EventType.CARD_PLAYED
            and event.metadata.get("card_id") == 996201
        )
        self.assertTrue(changed_play.metadata["cost_changed"])
        self.assertEqual(
            (changed_play.metadata["base_cost"], changed_play.metadata["source_cost"]),
            (2, 1),
        )

        engine.players[0].deck = [_card(996202)]
        normal = _put_hand(engine, _card(996203, cost=1))
        engine.apply(PlayCard(0, engine.players[0].hand.index(normal)))
        self.assertEqual(facility.countdown, 4)
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [996200],
        )

        engage = self.fresh(seed=23)
        facility = _play(engage, self.repository, 10332210)
        follower = _put_hand(engage, _card(996210, cost=4, attack=2, life=3))
        spell = _put_hand(
            engage,
            _card(
                996211,
                card_type="法术",
                attack=None,
                life=None,
            ),
        )
        engage.apply(ActivateAmulet(0, facility.entity_id))
        self.assertEqual(
            {
                option.entity_id
                for option in engage.state.pending_choice.options
            },
            {follower.entity_id},
        )
        before = engage.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engage.apply(Choose(0, f"hand:{spell.entity_id}"))
        self.assertEqual(engage.deterministic_fingerprint(), before)
        _choose_entity(engage, follower.entity_id)
        self.assertEqual(follower.current_cost, 5)
        self.assertEqual(
            (
                follower.stat_modifiers[0].attack_delta,
                follower.stat_modifiers[0].health_delta,
            ),
            (1, 1),
        )
        engage.assert_invariants()

        stale = self.fresh(seed=27)
        facility = _play(stale, self.repository, 10332210)
        departed = _put_hand(stale, _card(996212, cost=3))
        stale.apply(ActivateAmulet(0, facility.entity_id))
        selected_option = stale.state.pending_choice.options[0].option_id
        departed_index = stale.players[0].hand.index(departed)
        stale.players[0].hand.pop(departed_index)
        stale.players[0].hand_entity_ids.pop(departed_index)
        stale.apply(Choose(0, selected_option))
        self.assertIsNone(stale.state.pending_choice)
        self.assertEqual(departed.cost_modifiers, [])
        self.assertEqual(departed.stat_modifiers, [])
        stale.assert_invariants()

    def test_truth_facility_no_target_is_illegal_and_rl_masks_match_commands(self):
        no_target = self.fresh(seed=29)
        facility = _play(no_target, self.repository, 10332210)
        _put_hand(
            no_target,
            _card(
                996220,
                card_type="法术",
                attack=None,
                life=None,
            ),
        )
        command = ActivateAmulet(0, facility.entity_id)
        self.assertNotIn(command, no_target.legal_commands())
        before = no_target.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            no_target.apply(command)
        self.assertEqual(no_target.deterministic_fingerprint(), before)

        deck_a = [_card(996300 + index) for index in range(40)]
        deck_b = [_card(996400 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=31,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=31)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        facility = _play(env.core, self.repository, 10332210)
        follower = _put_hand(env.core, _card(996500))
        fusion = _put_hand(env.core, self.repository.get(10324110))
        loot = _put_hand(env.core, self.repository.get(90021310))
        env.invalidate_cache(reason="sixteenth batch setup")
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        activate = ActivateAmulet(0, facility.entity_id)
        begin_fusion = BeginFusion(0, fusion.entity_id)
        self.assertTrue(env.action_mask()[env._encode_command(activate)])
        self.assertTrue(env.action_mask()[env._encode_command(begin_fusion)])

        env.core.apply(activate)
        env.invalidate_cache(reason="sixteenth batch pending choice")
        self.assertEqual(
            {
                env._decode_action(index)
                for index, allowed in enumerate(env.action_mask())
                if allowed
            },
            set(env.core.legal_commands()),
        )
        self.assertIn(
            follower.entity_id,
            {
                option.entity_id
                for option in env.core.state.pending_choice.options
            },
        )
        before = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(Choose(0, f"hand:{loot.entity_id}"))
        self.assertEqual(env.core.deterministic_fingerprint(), before)

    def test_worshipper_restores_self_then_leader_by_actual_amount(self):
        engine = self.fresh(seed=37)
        source = _play(engine, self.repository, 10342110)
        self.assertTrue(source.has_keyword("突进"))
        self.assertTrue(source.has_keyword("守护"))
        source.health = 1
        engine.players[0].health = 10
        engine.apply(EndTurn(0))
        self.assertEqual(source.health, source.max_health)
        self.assertEqual(engine.players[0].health, 14)
        relevant = [
            event.type
            for event in engine.event_history
            if event.type in {
                EventType.FOLLOWER_HEALED,
                EventType.LEADER_HEALED,
            }
        ]
        self.assertEqual(
            relevant[-2:],
            [EventType.FOLLOWER_HEALED, EventType.LEADER_HEALED],
        )

        full_source = self.fresh(seed=41)
        source = _play(full_source, self.repository, 10342110)
        full_source.players[0].health = 10
        full_source.apply(EndTurn(0))
        self.assertEqual(full_source.players[0].health, 10)

        capped_leader = self.fresh(seed=43)
        source = _play(capped_leader, self.repository, 10342110)
        source.health = 2
        capped_leader.players[0].health = capped_leader.players[0].max_health
        capped_leader.apply(EndTurn(0))
        self.assertEqual(source.health, source.max_health)
        self.assertEqual(
            capped_leader.players[0].health,
            capped_leader.players[0].max_health,
        )

    def test_himeka_crest_uses_crest_count_random_filter_and_target_owner_banish(self):
        engine = self.fresh(seed=47)
        first = _play(engine, self.repository, 10364110)
        engine.players[0].mana = 10
        _play(engine, self.repository, 10364110)
        self.assertEqual(len(engine.players[0].emblems), 2)
        eligible = [
            _put_unit(engine, 1, _card(996600 + index, attack=attack))
            for index, attack in enumerate((2, 4))
        ]
        too_large = _put_unit(engine, 1, _card(996610, attack=5))
        engine.apply(EndTurn(0))
        self.assertTrue(all(unit.attack_restrictions for unit in eligible))
        self.assertTrue(all(
            TurnEndDestroyTiming.OWNER_TURN in unit.turn_end_banish_timings
            for unit in eligible
        ))
        self.assertFalse(too_large.attack_restrictions)
        attack_commands = {
            command.attacker_id
            for command in engine.legal_commands()
            if isinstance(command, Attack)
        }
        self.assertTrue(all(
            unit.entity_id not in attack_commands for unit in eligible
        ))

        engine.apply(EndTurn(1))
        self.assertTrue(all(
            unit not in engine.players[1].board for unit in eligible
        ))
        self.assertIn(too_large, engine.players[1].board)
        self.assertIn(first, engine.players[0].board)
        engine.assert_invariants()

        no_source = self.fresh(seed=53)
        source = _play(no_source, self.repository, 10364110)
        _destroy_units(no_source, source)
        enemy = _put_unit(no_source, 1, _card(996620, attack=2))
        no_source.apply(EndTurn(0))
        self.assertFalse(enemy.attack_restrictions)
        self.assertFalse(enemy.turn_end_banish_timings)

    def test_himeka_super_evolve_sets_attack_only_and_seeded_target_is_reproducible(self):
        engine = self.fresh(seed=59)
        source = _put_unit(engine, 0, self.repository.get(10364110))
        enemies = [
            _put_unit(
                engine,
                1,
                _card(996700 + index, attack=attack, life=life),
            )
            for index, (attack, life) in enumerate(((1, 3), (7, 8)))
        ]
        _enable_super_evolution(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(
            [(unit.attack, unit.max_health) for unit in enemies],
            [(4, 3), (4, 8)],
        )

        def selected(seed: int):
            replay = self.fresh(seed=seed)
            _play(replay, self.repository, 10364110)
            targets = [
                _put_unit(replay, 1, _card(996720 + index, attack=3))
                for index in range(3)
            ]
            replay.apply(EndTurn(0))
            return tuple(
                unit.entity_id
                for unit in targets
                if unit.attack_restrictions
            )

        self.assertEqual(selected(61), selected(61))


class FusionComboCostReposeSixteenthAuditTests(unittest.TestCase):
    def test_database_multilingual_text_alt_modes_references_and_raw_json(self):
        expected_stats = {
            10324110: (10003, 2, 6, 5, 6),
            10514110: (10005, 1, 2, 1, 1),
            10332210: (10003, 3, 3, None, None),
            10342110: (10003, 4, 5, 3, 5),
            10364110: (10003, 6, 6, 0, 4),
        }
        expected_english = {
            10324110: "differently named cards",
            10514110: "exact copy each of 5 random cards",
            10332210: "cost has been changed",
            10342110: "fully restore the defense",
            10364110: "Set the attack of all enemy followers",
        }
        expected_modes = {
            10324110: 0,
            10514110: 0,
            10332210: 0,
            10342110: 0,
            10364110: 1,
        }
        expected_references = {
            10324110: [],
            10514110: [],
            10332210: [],
            10342110: [],
            10364110: [(10364110, "安息的继承者·妃花")],
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
                    self.assertEqual(
                        len(raw["alt_modes"]),
                        expected_modes[card_id],
                    )
            himeka_mode = connection.execute(
                """
                SELECT text_chs, text_eng, text_jpn, text_kor, text_cht
                FROM alt_modes WHERE card_id=10364110
                """
            ).fetchone()
            self.assertTrue(all(himeka_mode))
            self.assertIn("number of crests you have", himeka_mode[1])

    def test_all_five_cards_are_exact_with_clause_and_token_audits(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertEqual(
            report["summary"]["coverage_counts"],
            {
                "covered_exact": 735,
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
