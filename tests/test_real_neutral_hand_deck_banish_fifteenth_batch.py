# -*- coding: utf-8 -*-
"""Direct contracts for the fifteenth neutral hand/deck/banish slice."""

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
    _parse_condition,
    _parse_operation,
)
from swb.engine.commands import (
    ActivateAmulet,
    Attack,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
)
from swb.engine.effects import (
    ConditionType,
    EffectKind,
    TargetKind,
    TurnEndDestroyTiming,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.observation_v2 import _board_runtime
from swb.engine.resolution import IllegalCommand
from swb.engine.state import CostModifier, DeckCard, StatModifier
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _destroy_units,
    _fresh,
    _put_hand,
    _put_unit,
)


CARD_IDS = (10303210, 10502110, 10502120, 10602210, 10704110)
SOURCE_HASHES = {
    10303210: "b069f189e8db96393bb78f6bee0d22ede01f657feffe243e58da6721d2058c9f",
    10502110: "61594c632bed115c9de38d75610536570e0aaa84eae759548e027455934b689f",
    10502120: "25b62cc601c551947b2b4791870baede87597aa16575397df44051331ff944bc",
    10602210: "597d7f4411c2f25d787b179e249aa1a2de8fc5dc608b9d57e8510128156b0e91",
    10704110: "57ca8c5e1fa5adbacda8f453c94f9b344dc20052ebe1e25269d0c9bed0c1bdb2",
}
TEST_EVIDENCE = (
    "tests/test_real_neutral_hand_deck_banish_fifteenth_batch.py"
)


def _play(engine, repository: CardRepository, card_id: int):
    hand_card = _put_hand(engine, repository.get(card_id))
    engine.apply(
        PlayCard(0, engine.players[0].hand.index(hand_card))
    )
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


def _ready_actual_attacker(engine, repository: CardRepository):
    unit = _put_unit(engine, 0, repository.get(10704110))
    engine._apply_initial_keyword_overrides(unit)
    unit._synchronize_keyword_state()
    unit.summoned_this_turn = False
    unit.can_attack = True
    unit.attacks_remaining = 1
    unit.rush_only = False
    return unit


class NeutralHandDeckBanishFifteenthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 15001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_new_schema_are_strict(self):
        tablet = self.rulebook.operations_for(10303210, Trigger.PLAY)
        self.assertEqual(
            [operation.kind for operation in tablet],
            [EffectKind.BANISH_DECK_DUPLICATES],
        )
        self.assertEqual(self.rulebook.activation_for(10303210).cost, 1)

        goddess = self.rulebook.operations_for(10502110, Trigger.EVOLVE)
        self.assertEqual(
            [operation.kind for operation in goddess],
            [EffectKind.DISCARD, EffectKind.COPY_LEFTMOST_HAND_TO_HAND],
        )
        self.assertEqual(goddess[0].target_count, 3)
        self.assertTrue(goddess[0].requires_full_target_count)

        general = self.rulebook.operations_for(10502120, Trigger.EVOLVE)[0]
        self.assertIs(
            general.conditions[0].type,
            ConditionType.CONTROLLER_HAND_TOP_BASE_COST_SUM_GREATER_THAN_OPPONENT,
        )
        self.assertEqual(general.conditions[0].value, 3)

        invaded = self.rulebook.operations_for(10602210, Trigger.ACTIVATE)[0]
        self.assertIs(
            invaded.kind,
            EffectKind.TRANSFORM_HAND_FROM_RANDOM_ENEMY_DECK,
        )
        self.assertEqual(self.rulebook.activation_for(10602210).cost, 0)
        self.assertTrue(invaded.requires_target)

        illamrita = self.rulebook.operations_for(10704110, Trigger.ATTACK)
        self.assertEqual(
            [operation.kind for operation in illamrita],
            [
                EffectKind.ADD_KEYWORD,
                EffectKind.ADD_ATTACK_RESTRICTION,
                EffectKind.GRANT_TURN_END_BANISH,
            ],
        )
        self.assertIs(
            illamrita[2].turn_end_banish_timing,
            TurnEndDestroyTiming.OWNER_TURN,
        )
        emblem = self.rulebook.emblem_def("illamrita_designated_target")
        self.assertEqual(emblem.countdown, 2)
        self.assertEqual(
            [operation.kind for operation in emblem.on_expire],
            [EffectKind.SUMMON, EffectKind.EVOLVE_UNIT],
        )

        valid = _parse_operation(
            {
                "kind": "grant_turn_end_banish",
                "target": "enemy_unit",
                "turn_end_banish_timing": "owner_turn",
            },
            "test.json",
            1,
        )
        self.assertIs(
            valid.turn_end_banish_timing,
            TurnEndDestroyTiming.OWNER_TURN,
        )
        for raw in (
            {
                "kind": "banish_deck_duplicates",
                "target": "enemy_leader",
            },
            {
                "kind": "banish_deck_duplicates",
                "target": "own_leader",
                "amount": 1,
            },
            {
                "kind": "copy_leftmost_hand_to_hand",
                "target": "own_leader",
                "amount": 0,
            },
            {
                "kind": "transform_hand_from_random_enemy_deck",
                "target": "enemy_unit",
            },
            {
                "kind": "grant_turn_end_banish",
                "target": "enemy_unit",
            },
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test.json", 1)
        with self.assertRaises(ValueError):
            _parse_condition(
                {
                    "type": (
                        "controller_hand_top_base_cost_sum_greater_than_opponent"
                    ),
                    "value": 0,
                },
                "test.json",
                1,
            )

    def test_tablet_banishes_only_later_physical_duplicates_in_deck_order(self):
        engine = self.fresh(seed=3)
        first = _card(995001, cost=2)
        duplicate = DeckCard(
            definition=first,
            cost_modifiers=[
                CostModifier(
                    modifier_id=engine._allocate_modifier_id(),
                    mode="set",
                    amount=0,
                    duration="permanent",
                )
            ],
        )
        second = _card(995002, cost=3)
        third = _card(995003, cost=4)
        engine.players[0].deck = [
            first,
            duplicate,
            second,
            second,
            third,
        ]
        source = _play(engine, self.repository, 10303210)
        self.assertEqual(
            [card.card_id for card in engine.players[0].deck],
            [995001, 995002, 995003],
        )
        self.assertIs(engine.players[0].deck[0], first)
        self.assertEqual(
            [card.card_id for card in engine.players[0].banished],
            [995001, 995002],
        )
        summary = [
            event
            for event in engine.event_history
            if event.type is EventType.DECK_DUPLICATES_BANISHED
        ]
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0].amount, 2)

        mana_before = engine.players[0].mana
        deck_before = len(engine.players[0].deck)
        engine.apply(ActivateAmulet(0, source.entity_id))
        self.assertEqual(engine.players[0].mana, mana_before - 1)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)
        engine.assert_invariants()

    def test_tablet_no_duplicates_preserves_rng_and_emits_no_summary(self):
        engine = self.fresh(seed=5)
        engine.players[0].deck = [
            _card(995010),
            _card(995011),
            _card(995012),
        ]
        rng_before = engine.random.getstate()
        _play(engine, self.repository, 10303210)
        self.assertEqual(engine.random.getstate(), rng_before)
        self.assertFalse(any(
            event.type is EventType.DECK_DUPLICATES_BANISHED
            for event in engine.event_history
        ))

    def test_goddess_discards_three_then_exactly_copies_new_leftmost_three(self):
        engine = self.fresh(seed=7)
        source = _put_unit(engine, 0, self.repository.get(10502110))
        discarded = [
            _put_hand(engine, _card(995020 + index, cost=index + 1))
            for index in range(3)
        ]
        retained = [
            _put_hand(engine, _card(995030 + index, cost=5 + index))
            for index in range(3)
        ]
        retained[0].cost_modifiers.append(CostModifier(
            modifier_id=engine._allocate_modifier_id(),
            mode="subtract",
            amount=2,
            duration="permanent",
        ))
        retained[0].stat_modifiers.append(StatModifier(
            modifier_id=engine._allocate_modifier_id(),
            attack_delta=2,
            health_delta=3,
            duration="permanent",
        ))
        retained[0].spellboost_count = 4
        retained[0].spellboost_cost_reduction = 1
        retained[0].permanent_keywords.add("守护")
        retained[0].effect_destroy_immunity = True

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        for card in discarded:
            _choose_entity(engine, card.entity_id)

        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [995030, 995031, 995032, 995030, 995031, 995032],
        )
        original = engine.players[0].hand[0]
        copied = engine.players[0].hand[3]
        self.assertNotEqual(original.entity_id, copied.entity_id)
        self.assertEqual(original.current_cost, copied.current_cost)
        self.assertEqual(original.attack, copied.attack)
        self.assertEqual(original.life, copied.life)
        self.assertEqual(original.spellboost_count, copied.spellboost_count)
        self.assertEqual(original.effective_keywords, copied.effective_keywords)
        self.assertTrue(copied.effect_destroy_immunity)
        self.assertNotEqual(
            original.cost_modifiers[0].modifier_id,
            copied.cost_modifiers[0].modifier_id,
        )
        self.assertNotEqual(
            original.stat_modifiers[0].modifier_id,
            copied.stat_modifiers[0].modifier_id,
        )
        added = [
            event
            for event in engine.event_history
            if event.type is EventType.CARD_ADDED_TO_HAND
            and event.metadata.get("exact_copy")
        ]
        self.assertEqual(len(added), 3)
        self.assertTrue(all(
            event.metadata["revealed"] is False for event in added
        ))
        engine.assert_invariants()

    def test_goddess_shortage_skips_discard_then_copies_available_leftmost(self):
        engine = self.fresh(seed=11)
        source = _put_unit(engine, 0, self.repository.get(10502110))
        _put_hand(engine, _card(995040))
        _put_hand(engine, _card(995041))
        _enable_evolution(engine)
        command = Evolve(0, source.entity_id)
        self.assertIn(command, engine.legal_commands())
        engine.apply(command)
        self.assertTrue(source.evolved)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [995040, 995041, 995040, 995041],
        )
        self.assertFalse(any(
            event.type is EventType.CARD_DISCARDED
            for event in engine.event_history
        ))

    def test_behemoth_uses_three_highest_base_costs_and_strict_comparison(self):
        engine = self.fresh(seed=13)
        source = _put_unit(engine, 0, self.repository.get(10502120))
        own = [
            _put_hand(engine, _card(995050, cost=8)),
            _put_hand(engine, _card(995051, cost=4)),
            _put_hand(engine, _card(995052, cost=1)),
            _put_hand(engine, _card(995053, cost=0)),
        ]
        own[0].cost_modifiers.append(CostModifier(
            modifier_id=engine._allocate_modifier_id(),
            mode="set",
            amount=0,
            duration="permanent",
        ))
        for index, cost in enumerate((5, 3, 3, 0)):
            _put_hand(
                engine,
                _card(995060 + index, cost=cost),
                owner=1,
            )
        enemies = [
            _put_unit(engine, 1, _card(995070 + index, life=4))
            for index in range(3)
        ]
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertTrue(all(
            enemy not in engine.players[1].board for enemy in enemies
        ))

        tied = self.fresh(seed=17)
        source = _put_unit(tied, 0, self.repository.get(10502120))
        for owner in (0, 1):
            for index, cost in enumerate((5, 3, 3)):
                _put_hand(
                    tied,
                    _card(995080 + owner * 10 + index, cost=cost),
                    owner=owner,
                )
        survivor = _put_unit(tied, 1, _card(995099, life=4))
        _enable_evolution(tied)
        tied.apply(Evolve(0, source.entity_id))
        self.assertIn(survivor, tied.players[1].board)

    def test_invaded_world_seeded_transform_copies_physical_deck_cost(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=19)
            source = _play(engine, self.repository, 10602210)
            target = _put_hand(engine, _card(995100, cost=7))
            replacement = _card(
                995101,
                cost=6,
                card_type="法术",
                attack=None,
                life=None,
            )
            engine.players[1].deck = [DeckCard(
                definition=replacement,
                cost_modifiers=[CostModifier(
                    modifier_id=engine._allocate_modifier_id(),
                    mode="set",
                    amount=1,
                    duration="permanent",
                )],
            )]
            original_entity_id = target.entity_id
            engine.apply(ActivateAmulet(0, source.entity_id))
            _choose_entity(engine, original_entity_id)
            self.assertEqual(target.entity_id, original_entity_id)
            self.assertEqual(target.card_id, 995101)
            self.assertEqual(target.current_cost, 1)
            self.assertEqual(len(engine.players[1].deck), 1)
            transformed = [
                event
                for event in engine.event_history
                if event.type is EventType.HAND_CARD_TRANSFORMED
            ]
            self.assertEqual(transformed[-1].metadata["copied_from_zone"], "enemy_deck")
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_invaded_world_empty_deck_skips_without_rng_and_no_hand_is_illegal(self):
        empty = self.fresh(seed=23)
        source = _play(empty, self.repository, 10602210)
        target = _put_hand(empty, _card(995110, cost=4))
        empty.players[1].deck = []
        rng_before = empty.random.getstate()
        empty.apply(ActivateAmulet(0, source.entity_id))
        _choose_entity(empty, target.entity_id)
        self.assertEqual(target.card_id, 995110)
        self.assertEqual(empty.random.getstate(), rng_before)

        no_target = self.fresh(seed=29)
        source = _play(no_target, self.repository, 10602210)
        command = ActivateAmulet(0, source.entity_id)
        before = no_target.deterministic_fingerprint()
        self.assertNotIn(command, no_target.legal_commands())
        with self.assertRaises(IllegalCommand):
            no_target.apply(command)
        self.assertEqual(no_target.deterministic_fingerprint(), before)

    def test_invaded_world_pending_choice_and_rl_mask_remain_consistent(self):
        deck = [_card(995200 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
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
        source = _play(env.core, self.repository, 10602210)
        target = _put_hand(env.core, _card(995250))
        env.players[1].deck = [_card(995251)]
        env.invalidate_cache(reason="invaded setup")
        activation = ActivateAmulet(0, source.entity_id)
        self.assertTrue(env.action_mask()[env._encode_command(activation)])
        env.core.apply(activation)
        env.invalidate_cache(reason="invaded choice")
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
            env.core.apply(Choose(0, "hand:999999"))
        self.assertEqual(env.core.deterministic_fingerprint(), before)
        _choose_entity(env.core, target.entity_id)

    def test_illamrita_follower_strike_grants_barrier_lock_and_owner_end_banish(self):
        engine = self.fresh(seed=37)
        source = _ready_actual_attacker(engine, self.repository)
        target = _put_unit(
            engine,
            1,
            _card(995300, attack=0, life=10),
        )
        engine.apply(Attack(0, source.entity_id, target.entity_id))
        self.assertTrue(source.has_keyword("屏障"))
        self.assertTrue(target.attack_restrictions)
        self.assertIn(
            TurnEndDestroyTiming.OWNER_TURN,
            target.turn_end_banish_timings,
        )
        self.assertEqual(_board_runtime(target)[14], 2.0)
        engine.apply(EndTurn(0))
        self.assertNotIn(
            Attack(1, target.entity_id, None),
            engine.legal_commands(),
        )
        engine.apply(EndTurn(1))
        self.assertNotIn(target, engine.players[1].board)
        self.assertIn(target.definition, engine.players[1].banished)
        self.assertFalse(any(
            card.definition.card_id == target.definition.card_id
            for card in engine.players[1].graveyard
        ))
        engine.assert_invariants()

    def test_illamrita_leader_attack_does_not_grant_and_silence_clears_banish(self):
        leader = self.fresh(seed=41)
        source = _ready_actual_attacker(leader, self.repository)
        leader.apply(Attack(0, source.entity_id, None))
        self.assertFalse(source.has_keyword("屏障"))

        silenced = self.fresh(seed=43)
        source = _ready_actual_attacker(silenced, self.repository)
        target = _put_unit(
            silenced,
            1,
            _card(995310, attack=0, life=10),
        )
        silenced.apply(Attack(0, source.entity_id, target.entity_id))
        target.remove_all_abilities()
        self.assertFalse(target.attack_restrictions)
        self.assertFalse(target.turn_end_banish_timings)
        silenced.apply(EndTurn(0))
        silenced.apply(EndTurn(1))
        self.assertIn(target, silenced.players[1].board)

    def test_illamrita_last_words_crest_expires_to_evolved_copy_and_handles_full_board(self):
        engine = self.fresh(seed=47)
        source = _put_unit(engine, 0, self.repository.get(10704110))
        _destroy_units(engine, source)
        self.assertEqual(len(engine.players[0].emblems), 1)
        self.assertEqual(engine.players[0].emblems[0].countdown, 2)
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        returned = next(
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == 10704110
        )
        self.assertTrue(returned.evolved)
        self.assertFalse(engine.players[0].emblems)

        full = self.fresh(seed=53)
        source = _put_unit(full, 0, self.repository.get(10704110))
        _destroy_units(full, source)
        for index in range(full.config.max_board):
            _put_unit(full, 0, _card(995320 + index))
        full.apply(EndTurn(0))
        full.apply(EndTurn(1))
        full.apply(EndTurn(0))
        full.apply(EndTurn(1))
        self.assertEqual(len(full.players[0].board), full.config.max_board)
        self.assertFalse(any(
            unit.definition.card_id == 10704110
            for unit in full.players[0].board
        ))
        full.assert_invariants()


class NeutralHandDeckBanishFifteenthAuditTests(unittest.TestCase):
    def test_database_multilingual_text_alt_modes_references_and_raw_json(self):
        expected_stats = {
            10303210: (10003, 0, 3, None, None),
            10502110: (10005, 0, 5, 5, 5),
            10502120: (10005, 0, 4, 4, 5),
            10602210: (10006, 0, 3, None, None),
            10704110: (10007, 0, 6, 1, 4),
        }
        expected_english = {
            10303210: "Banish all duplicates",
            10502110: "exact copy each",
            10502120: "3 highest base costs",
            10602210: "random card in your opponent's deck",
            10704110: "At the end of your turn, banish this card",
        }
        expected_modes = {
            10303210: 0,
            10502110: 0,
            10502120: 0,
            10602210: 0,
            10704110: 1,
        }
        expected_references = {
            10303210: [],
            10502110: [],
            10502120: [],
            10602210: [],
            10704110: [(10704110, "特殊目标·海雷姆哈妮")],
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

    def test_all_five_cards_are_exact_with_clause_and_token_audits(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertEqual(
            report["summary"]["coverage_counts"],
            {
                "covered_exact": 713,
                "text_unclear": 16,
                "supported_missing_rule": 6,
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
