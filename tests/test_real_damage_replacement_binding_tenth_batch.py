"""Direct contracts for the tenth exact damage/binding rule slice."""

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
from swb.engine.card_rules import (
    RuleBook,
    Trigger,
    _parse_operation,
    _parse_passive,
)
from swb.engine.commands import Attack, Choose, PlayCard
from swb.engine.effects import (
    EffectKind,
    ExprType,
    TargetKind,
)
from swb.engine.emblem import EmblemDefinition
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import DamageType, IllegalCommand
from swb.engine.state import Amulet
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _fresh,
    _play,
    _put_amulet,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10163110,
    10401110,
    10464120,
    10711110,
    10804110,
)
SOURCE_HASHES = {
    10163110: "0e5f43bac960dd359ea23c63479fe9e9d0ed211896e6ac8c65ad918871705d58",
    10401110: "b6c40f7a9a2068431bd76ecb580e5736e8c0d97d0209e4a74e61edfbcbcae8db",
    10464120: "3045f81c75950cafce1a53c1354038c7a229ecd4b795400893b1c6ed170f075a",
    10711110: "9fdb2f32125d533e142e001d929dc0d9a83fd80313ec5b4075e31861cf363b29",
    10804110: "7af0d47c8162fdf781c5f383e022533509a039b71ec11662d9c2208b5715db8b",
}
TEST_EVIDENCE = "tests/test_real_damage_replacement_binding_tenth_batch.py"


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options
        if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _choose_mode(engine, mode_id: str) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options
        if option.option_id == f"choose_one:{mode_id}"
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _put_earth_sigil(engine, repository, *, count: int = 3) -> Amulet:
    amulet = Amulet(
        definition=repository.get(90031210),
        entity_id=engine.state.allocate_entity_id(),
        earth_sigil_count=count,
        entered_turn=engine.turn,
        origin=CardOrigin.TOKEN,
    )
    engine.players[0].board.append(amulet)
    return amulet


def _add_emblem(engine, owner: int, emblem_id: str, source_card_id: int) -> None:
    engine._add_emblem_to_player(
        owner,
        EmblemDefinition(emblem_id, source_card_id),
        source_card_id=source_card_id,
    )


class DamageReplacementBindingTenthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 10001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_generic_schema_contracts(self):
        for card_id in (10401110, 10464120, 10711110):
            self.assertEqual(
                self.rulebook.incoming_damage_replacement(card_id),
                (4, 3),
            )
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10401110),
            ("守护",),
        )
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10464120),
            ("守护",),
        )
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10711110),
            ("突进", "守护"),
        )

        katalina = self.rulebook.union_bursts_for(10401110)[0]
        self.assertEqual(
            (
                katalina.threshold,
                katalina.operations[0].kind,
                katalina.operations[0].target,
                katalina.operations[0].target_count,
            ),
            (
                10,
                EffectKind.DAMAGE_UNIT,
                TargetKind.RANDOM_ENEMY_UNIT,
                2,
            ),
        )
        vira = self.rulebook.union_bursts_for(10464120)[0]
        self.assertEqual(
            (vira.threshold, vira.operations[0].kind),
            (15, EffectKind.SUPER_EVOLVE_UNIT),
        )
        bone = self.rulebook.operations_for(10163110, Trigger.FANFARE)
        self.assertEqual(
            (
                bone[0].kind,
                bone[0].target,
                bone[0].bind_successful_targets,
                bone[1].amount_expr.type,
                bone[1].amount_expr.binding_key,
            ),
            (
                EffectKind.DESTROY,
                TargetKind.ALL_OWN_AMULETS,
                True,
                ExprType.BOUND_TARGET_COUNT,
                "destroyed_amulets",
            ),
        )
        albion = self.rulebook.operations_for(10804110, Trigger.FANFARE)[0]
        self.assertEqual(
            [option.option_id for option in albion.choose_one_options],
            ["banish_followers", "banish_amulets", "banish_emblems"],
        )
        self.assertIs(
            albion.choose_one_options[2].operations[0].kind,
            EffectKind.REMOVE_ALL_EMBLEMS,
        )

        passive = _parse_passive(
            {
                "card_id": 1,
                "kind": "incoming_damage_replacement",
                "threshold": 4,
                "amount": 3,
            },
            "test.json",
        )
        self.assertEqual((passive.threshold, passive.amount), (4, 3))
        for bad in (
            {
                "card_id": 1,
                "kind": "incoming_damage_replacement",
                "threshold": 0,
                "amount": 0,
            },
            {
                "card_id": 1,
                "kind": "incoming_damage_replacement",
                "threshold": 4,
                "amount": 4,
            },
            {
                "card_id": 1,
                "kind": "incoming_damage_replacement",
                "threshold": True,
                "amount": 0,
            },
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _parse_passive(bad, "test.json")
        with self.assertRaisesRegex(ValueError, "requires all_leaders"):
            _parse_operation(
                {
                    "kind": "remove_all_emblems",
                    "target": "own_leader",
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "non-empty target_key"):
            _parse_operation(
                {
                    "kind": "destroy",
                    "target": "all_own_amulets",
                    "bind_successful_targets": True,
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "only valid for destroy"):
            _parse_operation(
                {
                    "kind": "banish",
                    "target": "all_enemy_units",
                    "target_key": "bad",
                    "bind_successful_targets": True,
                },
                "test.json",
                1,
            )

    def test_bone_master_counts_only_successfully_destroyed_amulets(self):
        engine = self.fresh(seed=11)
        destroyed = [
            _put_amulet(engine, 0, 990001),
            _put_amulet(engine, 0, 990002),
        ]
        protected = _put_earth_sigil(engine, self.repository)
        enemies = [
            _put_unit(engine, 1, _card(990010 + index, life=2))
            for index in range(2)
        ]

        _play(engine, self.repository, 10163110)

        self.assertTrue(all(amulet not in engine.players[0].board for amulet in destroyed))
        self.assertIn(protected, engine.players[0].board)
        self.assertTrue(all(enemy not in engine.players[1].board for enemy in enemies))
        self.assertEqual(engine.players[1].health, 18)
        prevented = [
            event for event in engine.event_history
            if event.type is EventType.EARTH_SIGIL_DESTROY_PREVENTED
            and event.target_id == protected.entity_id
        ]
        self.assertEqual(len(prevented), 1)
        engine.assert_invariants()

        empty = self.fresh(seed=13)
        enemy = _put_unit(empty, 1, _card(990020, life=3))
        _play(empty, self.repository, 10163110)
        self.assertEqual((enemy.health, empty.players[1].health), (3, 20))
        empty.assert_invariants()

    def test_damage_replacement_threshold_barrier_and_ability_removal(self):
        for card_id in (10401110, 10464120, 10711110):
            with self.subTest(card_id=card_id):
                engine = self.fresh(seed=17 + card_id)
                target = _put_unit(
                    engine,
                    0,
                    self.repository.get(card_id),
                )
                target.health = target.max_health = 20

                below = engine.apply_damage(
                    None,
                    target,
                    3,
                    DamageType.EFFECT,
                    1,
                )
                threshold = engine.apply_damage(
                    None,
                    target,
                    4,
                    DamageType.COMBAT,
                    1,
                )
                self.assertEqual(
                    (
                        below.requested_amount,
                        below.actual_amount,
                        threshold.requested_amount,
                        threshold.prevented_amount,
                        threshold.actual_amount,
                    ),
                    (3, 3, 4, 1, 3),
                )

                target.add_keyword("屏障")
                barrier = engine.apply_damage(
                    None,
                    target,
                    10,
                    DamageType.EFFECT,
                    1,
                )
                self.assertEqual(
                    (
                        barrier.requested_amount,
                        barrier.prevented_amount,
                        barrier.actual_amount,
                        barrier.barrier_consumed,
                    ),
                    (10, 10, 0, True),
                )

                target.remove_all_abilities()
                unprotected = engine.apply_damage(
                    None,
                    target,
                    4,
                    DamageType.EFFECT,
                    1,
                )
                self.assertEqual(
                    (unprotected.prevented_amount, unprotected.actual_amount),
                    (0, 4),
                )
                engine.assert_invariants()

    def test_katalina_skybound_art_is_seeded_distinct_and_thresholded(self):
        below = self.fresh(seed=23)
        below.players[0].turns_started = 9
        below_targets = [
            _put_unit(below, 1, _card(990030 + index, life=10))
            for index in range(3)
        ]
        _play(below, self.repository, 10401110)
        self.assertEqual([target.health for target in below_targets], [10, 10, 10])

        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=29)
            engine.players[0].turns_started = 10
            targets = [
                _put_unit(engine, 1, _card(990040 + index, life=10))
                for index in range(3)
            ]
            _play(engine, self.repository, 10401110)
            healths = tuple(target.health for target in targets)
            self.assertEqual(sorted(healths), [5, 5, 10])
            outcomes.append((healths, engine.deterministic_fingerprint()))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_vira_multi_target_and_super_skybound_art(self):
        engine = self.fresh(seed=31)
        engine.players[0].turns_started = 15
        targets = [
            _put_unit(engine, 1, _card(990050 + index, life=7))
            for index in range(2)
        ]

        source = _play(engine, self.repository, 10464120)
        self.assertIsNotNone(engine.state.pending_choice)
        _choose_entity(engine, targets[0].entity_id)
        self.assertIsNotNone(engine.state.pending_choice)
        _choose_entity(engine, targets[1].entity_id)

        self.assertTrue(all(target not in engine.players[1].board for target in targets))
        self.assertTrue(source.super_evolved)
        self.assertIsNone(engine.state.pending_choice)
        engine.assert_invariants()

        no_targets = self.fresh(seed=37)
        no_targets.players[0].turns_started = 15
        source = _play(no_targets, self.repository, 10464120)
        self.assertTrue(source.super_evolved)
        self.assertIsNone(no_targets.state.pending_choice)

    def test_vira_target_choice_action_mask_and_illegal_choice_are_atomic(self):
        deck = [
            _card(
                990100 + index,
                class_id=6,
                class_name="主教",
            )
            for index in range(40)
        ]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=6,
            class_b=6,
            seed=41,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=41)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        targets = [
            _put_unit(env.core, 1, _card(990150 + index))
            for index in range(2)
        ]
        _play(env.core, self.repository, 10464120)
        env.invalidate_cache(reason="Vira target choice")

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
        env.invalidate_cache(reason="Vira second target choice")
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        self.assertEqual(len(decoded), 1)

    def test_giant_bear_exact_copy_has_rush_ward_and_respects_capacity(self):
        engine = self.fresh(seed=43)
        source = _play(engine, self.repository, 10711110)
        bears = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10711110
        ]
        self.assertEqual(len(bears), 2)
        copy = next(unit for unit in bears if unit.entity_id != source.entity_id)
        self.assertEqual(
            (copy.attack, copy.health, copy.max_health),
            (source.attack, source.health, source.max_health),
        )
        self.assertTrue(copy.has_keyword("突进"))
        self.assertTrue(copy.has_keyword("守护"))
        self.assertTrue(copy.can_attack)
        self.assertTrue(copy.rush_only)
        enemy = _put_unit(engine, 1, _card(990200, attack=0, life=10))
        engine.apply(Attack(0, copy.entity_id, enemy.entity_id))
        self.assertEqual(enemy.health, 4)

        full = self.fresh(seed=47)
        for index in range(4):
            _put_unit(full, 0, _card(990210 + index))
        _play(full, self.repository, 10711110)
        self.assertEqual(
            sum(
                unit.definition.card_id == 10711110
                for unit in full.players[0].board
            ),
            1,
        )
        self.assertEqual(len(full.players[0].board), 5)
        full.assert_invariants()

    def test_albion_modes_remove_exact_global_zone_and_keep_source(self):
        for mode_id in (
            "banish_followers",
            "banish_amulets",
            "banish_emblems",
        ):
            with self.subTest(mode_id=mode_id):
                engine = self.fresh(seed=53)
                ally = _put_unit(engine, 0, _card(990300))
                enemy = _put_unit(engine, 1, _card(990301))
                own_amulet = _put_amulet(engine, 0, 990302)
                enemy_amulet = _put_amulet(engine, 1, 990303)
                _add_emblem(engine, 0, "own_test_crest", 990304)
                _add_emblem(engine, 1, "enemy_test_crest", 990305)

                source = _play(engine, self.repository, 10804110)
                self.assertEqual(len(engine.state.pending_choice.options), 3)
                _choose_mode(engine, mode_id)

                self.assertIn(source, engine.players[0].board)
                if mode_id == "banish_followers":
                    self.assertNotIn(ally, engine.players[0].board)
                    self.assertNotIn(enemy, engine.players[1].board)
                    self.assertIn(own_amulet, engine.players[0].board)
                    self.assertIn(enemy_amulet, engine.players[1].board)
                    self.assertTrue(engine.players[0].emblems)
                    self.assertTrue(engine.players[1].emblems)
                elif mode_id == "banish_amulets":
                    self.assertIn(ally, engine.players[0].board)
                    self.assertIn(enemy, engine.players[1].board)
                    self.assertNotIn(own_amulet, engine.players[0].board)
                    self.assertNotIn(enemy_amulet, engine.players[1].board)
                    self.assertTrue(engine.players[0].emblems)
                    self.assertTrue(engine.players[1].emblems)
                else:
                    self.assertIn(ally, engine.players[0].board)
                    self.assertIn(enemy, engine.players[1].board)
                    self.assertIn(own_amulet, engine.players[0].board)
                    self.assertIn(enemy_amulet, engine.players[1].board)
                    self.assertEqual(engine.players[0].emblems, [])
                    self.assertEqual(engine.players[1].emblems, [])
                engine.assert_invariants()

    def test_albion_mode_action_mask_and_illegal_choice_are_atomic(self):
        deck = [
            _card(
                990400 + index,
                class_id=0,
                class_name="中立",
            )
            for index in range(40)
        ]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=59,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=59)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        _play(env.core, self.repository, 10804110)
        env.invalidate_cache(reason="Albion mode choice")
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        self.assertEqual(len(decoded), 3)
        before = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(Choose(0, "choose_one:missing"))
        self.assertEqual(env.core.deterministic_fingerprint(), before)


class DamageReplacementBindingTenthAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository("data/cards.sqlite3")

    def test_database_stats_multilingual_text_references_and_modes(self):
        expected_stats = {
            10163110: (10001, 6, 6, 4, 4),
            10401110: (10004, 0, 5, 5, 5),
            10464120: (10004, 6, 8, 6, 8),
            10711110: (10007, 1, 8, 6, 6),
            10804110: (10008, 0, 9, 13, 13),
        }
        expected_text = {
            10163110: ("number of amulets destroyed", "破壊した枚数"),
            10401110: ("Skybound Art", "受ける4以上"),
            10464120: ("Super Skybound Art", "超進化"),
            10711110: ("exact copy of this card", "受ける4以上"),
            10804110: ("Banish all crests", "クレストすべて"),
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            source_map = _load_source_text_map(connection)
            for card_id in CARD_IDS:
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT card_set_id, class_id, cost, attack, life
                        FROM cards WHERE card_id=?
                        """,
                        (card_id,),
                    ).fetchone()
                    self.assertEqual(row, expected_stats[card_id])
                    self.assertEqual(
                        _source_text_sha256(source_map[card_id]),
                        SOURCE_HASHES[card_id],
                    )
                    languages = connection.execute(
                        """
                        SELECT text_eng, text_jpn
                        FROM skill_texts
                        WHERE card_id=?
                        ORDER BY position
                        """,
                        (card_id,),
                    ).fetchall()
                    combined_eng = "\n".join(row[0] for row in languages)
                    combined_jpn = "\n".join(row[1] for row in languages)
                    self.assertIn(expected_text[card_id][0], combined_eng)
                    self.assertIn(expected_text[card_id][1], combined_jpn)
                    references = connection.execute(
                        """
                        SELECT referenced_card_id
                        FROM card_references
                        WHERE card_id=? AND referenced_card_id IS NOT NULL
                        ORDER BY position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(references, [])
                    mode_count = connection.execute(
                        "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                        (card_id,),
                    ).fetchone()[0]
                    self.assertEqual(mode_count, 0)

    def test_all_five_cards_are_exact_with_direct_clause_and_token_evidence(self):
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

        token_audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        self.assertEqual(
            token_audit["summary"]["categories"]["entry_behavior_complete"],
            91,
        )


if __name__ == "__main__":
    unittest.main()
