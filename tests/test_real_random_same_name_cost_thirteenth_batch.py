# -*- coding: utf-8 -*-
"""Direct contracts for the thirteenth random/same-name/deck-cost slice."""

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
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Attack, Choose, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import CostChangeMode, EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import Amulet, CostModifier, DeckCard
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _destroy_units,
    _fresh,
    _put_amulet,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10173130,
    10244120,
    10263110,
    10334110,
    10532310,
)
SOURCE_HASHES = {
    10173130: "ae1124c9ba504f5c528f7dfb31a0198c4e85259933775bf3df14c94e75486a55",
    10244120: "1e70e77fffabbbbb906121bc5393c5d55576d8bc62f574ff78920758b1636087",
    10263110: "846f38b2983bff1a73b7601b457cb5cc928f7dfb4ae0ab87d590db0609152937",
    10334110: "2bc120eb5b4af565f0618832249f9f9b1e24999c3ab90cba60cfc30482ad8778",
    10532310: "753fb81e0282fb936e42e0cd5c8792a346085741e0842025ef35a257664baff6",
}
TEST_EVIDENCE = "tests/test_real_random_same_name_cost_thirteenth_batch.py"


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
        option for option in request.options
        if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolve(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False
    engine.state.active_player = 0


def _enable_super_evolve(engine) -> None:
    player = engine.players[0]
    player.turns_started = (
        engine.config.first_player_super_evolution_unlock_turn
    )
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False
    engine.state.active_player = 0


def _put_sigil(engine, repository: CardRepository, count: int) -> Amulet:
    sigil = Amulet(
        definition=repository.get(90031210),
        entity_id=engine.state.allocate_entity_id(),
        earth_sigil_count=count,
        entered_turn=engine.turn,
        origin=CardOrigin.TOKEN,
    )
    engine.players[0].board.append(sigil)
    return sigil


class RandomSameNameCostThirteenthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 13001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_strict_generic_schema(self):
        liam = self.rulebook.operations_for(10173130, Trigger.EVOLVE)
        self.assertEqual(
            [operation.kind for operation in liam],
            [EffectKind.ADD_KEYWORD, EffectKind.GRANT_LAST_WORDS],
        )
        self.assertEqual(liam[0].board_filter.tribe_name, "人偶")
        self.assertEqual(liam[1].granted_operations[0].amount, 2)

        fennie = self.rulebook.operations_for(
            10244120,
            Trigger.FANFARE,
        )[0]
        self.assertIs(fennie.kind, EffectKind.CHANGE_DECK_COST)
        self.assertIs(fennie.mode, CostChangeMode.HALVE_ROUND_UP)
        self.assertIsNotNone(fennie.deck_filter)

        agnes = self.rulebook.operations_for(10263110, Trigger.ATTACK)[0]
        self.assertIs(agnes.target, TargetKind.RANDOM_ENEMY_UNIT)
        self.assertTrue(agnes.exclude_attack_target)
        self.assertEqual(agnes.conditions[0].value, 2)
        self.assertEqual(agnes.conditions[0].board_filter.card_type, "护符")

        velharia = self.rulebook.operations_for(10334110, Trigger.EVOLVE)
        self.assertEqual(
            [operation.kind for operation in velharia],
            [
                EffectKind.SELECT_TARGETS,
                EffectKind.BANISH,
                EffectKind.BANISH_SAME_NAME,
            ],
        )
        kitty = self.rulebook.operations_for(10532310, Trigger.PLAY)[0]
        random_choice = kitty.earth_rite_operations[0]
        self.assertIs(random_choice.kind, EffectKind.RANDOM_CHOICE)
        self.assertEqual(random_choice.amount, 2)
        self.assertEqual(
            [option.option_id for option in random_choice.random_choice_options],
            ["summon_clay_golem", "heal_leader", "gain_earth_sigils"],
        )

        with self.assertRaisesRegex(ValueError, "duplicate option id"):
            _parse_operation(
                {
                    "kind": "random_choice",
                    "target": "own_leader",
                    "amount": 1,
                    "options": [
                        {
                            "id": "same",
                            "operations": [
                                {"kind": "draw", "target": "own_leader"}
                            ],
                        },
                        {
                            "id": "same",
                            "operations": [
                                {"kind": "draw", "target": "own_leader"}
                            ],
                        },
                    ],
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "no greater"):
            _parse_operation(
                {
                    "kind": "random_choice",
                    "target": "own_leader",
                    "amount": 3,
                    "options": [
                        {
                            "id": "a",
                            "operations": [
                                {"kind": "draw", "target": "own_leader"}
                            ],
                        },
                        {
                            "id": "b",
                            "operations": [
                                {"kind": "draw", "target": "own_leader"}
                            ],
                        },
                    ],
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "requires target 'random_enemy_unit'"):
            _parse_operation(
                {
                    "kind": "destroy",
                    "target": "enemy_unit",
                    "exclude_attack_target": True,
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "derives its value"):
            _parse_operation(
                {
                    "kind": "change_deck_cost",
                    "target": "own_leader",
                    "mode": "halve_round_up",
                    "amount": 2,
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "requires target 'previous_target'"):
            _parse_operation(
                {
                    "kind": "banish_same_name",
                    "target": "enemy_unit",
                },
                "test.json",
                1,
            )

    def test_liam_summons_three_enhanced_puppets_and_respects_capacity(self):
        engine = self.fresh(seed=3)
        source = _play(engine, self.repository, 10173130)
        puppets = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90071120
        ]
        self.assertIsNotNone(source)
        self.assertEqual(len(puppets), 3)
        self.assertTrue(all(unit.origin is CardOrigin.TOKEN for unit in puppets))

        limited = self.fresh(seed=5)
        for index in range(3):
            _put_unit(limited, 0, _card(993100 + index))
        _play(limited, self.repository, 10173130)
        self.assertEqual(len(limited.players[0].board), limited.config.max_board)
        self.assertEqual(
            sum(
                unit.definition.card_id == 90071120
                for unit in limited.players[0].board
            ),
            1,
        )
        limited.assert_invariants()

    def test_liam_evolve_grants_only_puppetry_ward_and_last_words(self):
        engine = self.fresh(seed=7)
        source = _put_unit(engine, 0, self.repository.get(10173130))
        token = _put_unit(engine, 0, self.repository.get(90071120))
        puppet = _put_unit(
            engine,
            0,
            _card(993110, tribe_id=15, tribe_name="人偶"),
        )
        outsider = _put_unit(engine, 0, _card(993111))
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))

        self.assertTrue(token.has_keyword("守护"))
        self.assertTrue(puppet.has_keyword("守护"))
        self.assertFalse(outsider.has_keyword("守护"))
        self.assertEqual(len(token.granted_last_words), 1)
        self.assertEqual(len(puppet.granted_last_words), 1)
        self.assertEqual(len(outsider.granted_last_words), 0)
        before = engine.players[1].health
        _destroy_units(engine, token, puppet)
        self.assertEqual(engine.players[1].health, before - 4)
        damage = [
            event for event in engine.event_history
            if (
                event.type is EventType.DAMAGE_APPLIED
                and event.metadata.get("target_player") == 1
                and event.amount == 2
            )
        ]
        self.assertEqual(len(damage), 2)
        engine.assert_invariants()

    def test_fennie_halves_current_deck_cost_rounding_up_and_stacks(self):
        engine = self.fresh(seed=11)
        cards = [
            _card(993120, cost=9),
            _card(
                993121,
                cost=8,
                card_type="法术",
                attack=None,
                life=None,
            ),
            _card(
                993122,
                cost=1,
                card_type="护符",
                attack=None,
                life=None,
            ),
        ]
        already_halved = DeckCard(definition=_card(993123, cost=9))
        already_halved.cost_modifiers.append(
            CostModifier(1, "halve_round_up", 0, "permanent")
        )
        engine.players[0].deck = [*cards, already_halved]
        _play(engine, self.repository, 10244120)
        self.assertEqual(
            [card.current_cost for card in engine.players[0].deck],
            [5, 4, 1, 3],
        )

        engine.players[0].mana = 10
        _play(engine, self.repository, 10244120)
        self.assertEqual(
            [card.current_cost for card in engine.players[0].deck],
            [3, 2, 1, 2],
        )
        changed = [
            event for event in engine.event_history
            if event.type is EventType.DECK_CARD_COST_CHANGED
        ]
        self.assertEqual(len(changed), 8)
        self.assertTrue(all(event.metadata["mode"] == "halve_round_up" for event in changed))

        engine._draw(0, reason="费用继承测试")
        drawn = next(
            card for card in engine.players[0].hand
            if card.card_id == already_halved.card_id
        )
        self.assertEqual(drawn.current_cost, 2)
        engine.assert_invariants()

    def test_agnes_excludes_attack_target_and_is_seed_reproducible(self):
        outcomes = []
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=13)
            attacker = _put_unit(engine, 0, self.repository.get(10263110))
            _put_amulet(engine, 0, 993130)
            _put_amulet(engine, 0, 993131)
            attack_target = _put_unit(
                engine,
                1,
                _card(993132, attack=0, life=10),
            )
            candidates = [
                _put_unit(engine, 1, _card(993133 + index))
                for index in range(2)
            ]
            engine.apply(Attack(0, attacker.entity_id, attack_target.entity_id))
            self.assertIn(attack_target, engine.players[1].board)
            destroyed = [
                candidate.entity_id
                for candidate in candidates
                if candidate not in engine.players[1].board
            ]
            self.assertEqual(len(destroyed), 1)
            outcomes.append(destroyed[0])
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_agnes_threshold_leader_attack_and_no_other_candidate_paths(self):
        below = self.fresh(seed=17)
        attacker = _put_unit(below, 0, self.repository.get(10263110))
        _put_amulet(below, 0, 993140)
        enemy = _put_unit(below, 1, _card(993141))
        rng_before = below.random.getstate()
        below.apply(Attack(0, attacker.entity_id, None))
        self.assertIn(enemy, below.players[1].board)
        self.assertEqual(below.random.getstate(), rng_before)

        leader = self.fresh(seed=19)
        attacker = _put_unit(leader, 0, self.repository.get(10263110))
        _put_amulet(leader, 0, 993142)
        _put_amulet(leader, 0, 993143)
        enemy = _put_unit(leader, 1, _card(993144))
        leader.apply(Attack(0, attacker.entity_id, None))
        self.assertNotIn(enemy, leader.players[1].board)

        only_target = self.fresh(seed=23)
        attacker = _put_unit(
            only_target,
            0,
            self.repository.get(10263110),
        )
        _put_amulet(only_target, 0, 993145)
        _put_amulet(only_target, 0, 993146)
        target = _put_unit(
            only_target,
            1,
            _card(993147, attack=0, life=10),
        )
        rng_before = only_target.random.getstate()
        only_target.apply(Attack(0, attacker.entity_id, target.entity_id))
        self.assertIn(target, only_target.players[1].board)
        self.assertEqual(only_target.random.getstate(), rng_before)

    def test_velharia_fanfare_and_normal_evolve_banish_only_selected(self):
        engine = self.fresh(seed=29)
        drawn = _card(993150)
        engine.players[0].deck = [drawn]
        source = _play(engine, self.repository, 10334110)
        self.assertTrue(any(card.card_id == drawn.card_id for card in engine.players[0].hand))

        first = _put_unit(engine, 1, _card(993151, name="same-name"))
        second = _put_unit(engine, 1, _card(993152, name="same-name"))
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        _choose_entity(engine, first.entity_id)
        self.assertNotIn(first, engine.players[1].board)
        self.assertIn(second, engine.players[1].board)
        self.assertEqual(
            [card.card_id for card in engine.players[1].banished],
            [first.definition.card_id],
        )

        no_target = self.fresh(seed=31)
        source = _put_unit(no_target, 0, self.repository.get(10334110))
        _enable_evolve(no_target)
        no_target.apply(Evolve(0, source.entity_id))
        self.assertIsNone(no_target.state.pending_choice)
        self.assertTrue(source.evolved)

    def test_velharia_super_evolve_banishes_all_same_name_and_handles_stale(self):
        engine = self.fresh(seed=37)
        source = _put_unit(engine, 0, self.repository.get(10334110))
        same = [
            _put_unit(engine, 1, _card(993160 + index, name="shared"))
            for index in range(3)
        ]
        different = _put_unit(engine, 1, _card(993163, name="different"))
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose_entity(engine, same[1].entity_id)
        self.assertTrue(all(unit not in engine.players[1].board for unit in same))
        self.assertIn(different, engine.players[1].board)
        self.assertCountEqual(
            [card.card_id for card in engine.players[1].banished],
            [unit.definition.card_id for unit in same],
        )

        stale = self.fresh(seed=41)
        source = _put_unit(stale, 0, self.repository.get(10334110))
        selected = _put_unit(stale, 1, _card(993170, name="shared"))
        survivor = _put_unit(stale, 1, _card(993171, name="shared"))
        _enable_super_evolve(stale)
        stale.apply(SuperEvolve(0, source.entity_id))
        option = next(
            option for option in stale.state.pending_choice.options
            if option.entity_id == selected.entity_id
        )
        stale.players[1].board.remove(selected)
        stale.apply(Choose(0, option.option_id))
        self.assertIn(survivor, stale.players[1].board)
        self.assertIsNone(stale.state.pending_choice)
        stale.assert_invariants()

    def test_velharia_choice_action_mask_and_illegal_choice_are_atomic(self):
        deck = [
            _card(
                993200 + index,
                class_id=3,
                class_name="巫师",
            )
            for index in range(40)
        ]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=3,
            class_b=3,
            seed=43,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=43)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        source = _put_unit(env.core, 0, self.repository.get(10334110))
        targets = [
            _put_unit(env.core, 1, _card(993250 + index))
            for index in range(2)
        ]
        _enable_evolve(env.core)
        env.core.apply(Evolve(0, source.entity_id))
        env.invalidate_cache(reason="Velharia target choice")

        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        self.assertEqual(len(decoded), 2)
        before = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(Choose(0, "entity:999999"))
        self.assertEqual(env.core.deterministic_fingerprint(), before)
        _choose_entity(env.core, targets[0].entity_id)

    def test_kitty_chooses_two_distinct_seeded_branches_and_audits_event(self):
        pairs = []
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=47)
            engine.players[0].health = 15
            _put_sigil(engine, self.repository, 2)
            _play(engine, self.repository, 10532310)
            event = next(
                event for event in engine.event_history
                if event.type is EventType.RANDOM_CHOICES_SELECTED
            )
            option_ids = tuple(event.metadata["option_ids"])
            self.assertEqual(len(option_ids), 2)
            self.assertEqual(len(set(option_ids)), 2)
            self.assertEqual(event.metadata["option_count"], 3)
            self.assertEqual(
                engine.players[0].health,
                17 if "heal_leader" in option_ids else 15,
            )
            self.assertEqual(
                sum(
                    unit.definition.card_id == 90031110
                    for unit in engine.players[0].board
                ),
                1 if "summon_clay_golem" in option_ids else 0,
            )
            self.assertEqual(
                sum(
                    entity.earth_sigil_count
                    for entity in engine.players[0].board
                    if isinstance(entity, Amulet)
                ),
                3 if "gain_earth_sigils" in option_ids else 0,
            )
            pairs.append(option_ids)
            fingerprints.append(engine.deterministic_fingerprint())
            engine.assert_invariants()
        self.assertEqual(pairs[0], pairs[1])
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_kitty_insufficient_sigils_skips_rng_and_board_capacity_is_bounded(self):
        insufficient = self.fresh(seed=53)
        sigil = _put_sigil(insufficient, self.repository, 1)
        rng_before = insufficient.random.getstate()
        _play(insufficient, self.repository, 10532310)
        self.assertEqual(sigil.earth_sigil_count, 1)
        self.assertEqual(insufficient.random.getstate(), rng_before)
        self.assertFalse(any(
            event.type is EventType.RANDOM_CHOICES_SELECTED
            for event in insufficient.event_history
        ))

        limited = self.fresh(seed=59)
        _put_sigil(limited, self.repository, 2)
        for index in range(4):
            _put_unit(limited, 0, _card(993300 + index))
        _play(limited, self.repository, 10532310)
        self.assertLessEqual(len(limited.players[0].board), limited.config.max_board)
        event = next(
            event for event in limited.event_history
            if event.type is EventType.RANDOM_CHOICES_SELECTED
        )
        self.assertEqual(len(set(event.metadata["option_ids"])), 2)
        limited.assert_invariants()


class RandomSameNameCostThirteenthAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository("data/cards.sqlite3")

    def test_database_snapshot_multilingual_text_references_modes_and_raw_json(self):
        expected_stats = {
            10173130: (10001, 7, 9, 7, 7),
            10244120: (10002, 4, 8, 4, 4),
            10263110: (10002, 6, 5, 3, 4),
            10334110: (10003, 3, 2, 1, 1),
            10532310: (10005, 3, 1, None, None),
        }
        expected_text = {
            10173130: ("all allied Puppetry", "人形・フォロワー"),
            10244120: ("Halve the cost", "コストを半分"),
            10263110: ("other than the attack target", "交戦相手でない"),
            10334110: ("all enemy copies", "同名のフォロワー"),
            10532310: ("2 random abilities", "ランダム2つ"),
        }
        expected_references = {
            10173130: [(90071120, "改良型·悬丝傀儡")],
            10244120: [],
            10263110: [],
            10334110: [],
            10532310: [(90031110, "泥尘巨像")],
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
                    raw = json.loads(connection.execute(
                        "SELECT raw_json FROM cards WHERE card_id=?",
                        (card_id,),
                    ).fetchone()[0])
                    self.assertEqual(raw["card_id"], card_id)
                    self.assertTrue(raw["skill_texts"])
                    self.assertEqual(raw["alt_modes"], [])

    def test_all_five_cards_are_exact_with_clause_token_and_authority_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertEqual(
            report["summary"]["coverage_counts"],
            {
                "covered_exact": 730,
                "text_unclear": 5,
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

        metadata = report["classifications"]["10244120"]["clause_audit"]
        self.assertTrue(metadata["errata"])
        with open(
            "data/audits/rule_clauses.json",
            encoding="utf-8",
        ) as audit_file:
            clause_payload = json.load(audit_file)
        fennie = next(
            entry for entry in clause_payload["cards"]
            if entry["card_id"] == 10244120
        )
        self.assertEqual(
            fennie["authority_sources"][0]["kind"],
            "official_card_faq",
        )
        self.assertIn("shadowverse-wb.com", fennie["authority_sources"][0]["url"])

        token_audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        self.assertEqual(
            token_audit["summary"]["categories"]["entry_behavior_complete"],
            91,
        )
        for token_id in (90071120, 90031110):
            token = next(
                item for item in token_audit["cards"]
                if item["card_id"] == token_id
            )
            self.assertEqual(token["category"], "entry_behavior_complete")


if __name__ == "__main__":
    unittest.main()
