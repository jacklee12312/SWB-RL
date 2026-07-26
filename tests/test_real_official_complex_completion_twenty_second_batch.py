# -*- coding: utf-8 -*-
"""Official-source closure tests for the final five collectible cards."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from scripts.ability_status import build_ability_audit
from scripts.report_rule_coverage import (
    _build_coverage_report,
    _load_source_text_map,
    _source_text_sha256,
)
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardPassive, CardRule, RuleBook, Trigger
from swb.engine.commands import Attack, Choose, PlayCard
from swb.engine.effects import (
    DeckFilter,
    EffectKind,
    EffectOperation,
    HandFilter,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import CostModifier, DeckCard, HandCard, StatModifier, Unit


CARD_IDS = (10201310, 10533310, 10572120, 10741310, 10851110)
TEST_EVIDENCE = (
    "tests/test_real_official_complex_completion_twenty_second_batch.py"
)
SOURCE_HASHES = {
    10201310: "bcc71c129901654eb3aface401adaf8d865624c42f52335085fbd21ecfec11df",
    10533310: "d9daf0fc677ca60574f087df5667bfae9035c7565f31f2ce1081f80a7bc321ae",
    10572120: "6799ee0fec31b68882a105309ddffe96be552a1d24e65dbfadea481ed8e1fa77",
    10741310: "ee78f8d2100e25c510e625393b7ce0363ac0ec7900a6eed577eb029e9661d86f",
    10851110: "4dcbefa70bb6864e83c5ec2acfcf35f95267f008aa96bd1c71dee60196d28fd0",
}
DATABASE_CONTRACTS = {
    10201310: ("Dark Side", 0, 2, "法术", None, None),
    10533310: ("Grandeur of the Dawnblossom", 3, 7, "法术", None, None),
    10572120: ("Lunar Bunny", 7, 2, "随从", 1, 1),
    10741310: ("Apathetic Gaze", 4, 4, "法术", None, None),
    10851110: ("Anisage, Clear Resolve", 5, 4, "随从", 2, 3),
}


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 0),
        class_name=overrides.get("class_name", "中立"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 3),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=True,
    )


def _engine(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 2201,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(220000, 220040)],
        [_card(card_id) for card_id in range(221000, 221040)],
        class_a=2,
        class_b=2,
        seed=seed,
        rulebook=rulebook,
        card_resolver=repository.get,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    for player in engine.players:
        player.hand.clear()
        player.hand_entity_ids.clear()
        player.board.clear()
    engine.players[0].max_mana = engine.players[0].mana = 10
    return engine


def _put_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int = 0,
) -> HandCard:
    hand_card = engine._make_hand_card(
        definition,
        engine.state.allocate_entity_id(),
    )
    engine.players[player_index].hand.append(hand_card)
    engine.players[player_index].hand_entity_ids.append(hand_card.entity_id)
    return hand_card


def _put_unit(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int = 0,
) -> Unit:
    unit = engine._summon_follower_to_board(
        player_index,
        definition,
        summon_cause="test",
    )
    if unit is None:
        raise AssertionError("test board unexpectedly full")
    return unit


def _play(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
) -> None:
    _put_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(engine.current_player, len(engine.players[0].hand) - 1))


def _write_rulebook(payload: dict) -> RuleBook:
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "rules.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return RuleBook.from_directory(directory)


class FinalMechanicSchemaTests(unittest.TestCase):
    def test_new_generic_effects_and_ward_passive_load(self):
        rulebook = _write_rulebook({
            "passives": [{"card_id": 3, "kind": "ignores_ward"}],
            "rules": [{
                "card_id": 1,
                "trigger": "play",
                "operations": [{
                    "kind": "transform_board_from_random_own_deck",
                    "target": "all_own_units",
                    "card_type_filter": "随从",
                }],
            }, {
                "card_id": 2,
                "trigger": "play",
                "operations": [{
                    "kind": "transform_deck_cards",
                    "target": "own_leader",
                    "card_id": 4,
                    "card_id_filter": 2,
                }],
            }],
        })
        self.assertTrue(rulebook.ignores_ward(3))
        self.assertFalse(rulebook.ignores_ward(4))
        self.assertEqual(
            rulebook.operations_for(1, Trigger.PLAY)[0].deck_filter,
            DeckFilter(card_type="随从"),
        )
        self.assertEqual(
            rulebook.operations_for(2, Trigger.PLAY)[0].deck_filter,
            DeckFilter(card_id=2),
        )

    def test_new_effect_schema_rejects_unsafe_or_ambiguous_shapes(self):
        invalid = (
            ({
                "kind": "transform_board_from_random_own_deck",
                "target": "all_enemy_units",
                "card_type_filter": "随从",
            }, "requires all_own_units"),
            ({
                "kind": "transform_board_from_random_own_deck",
                "target": "all_own_units",
                "card_type_filter": "法术",
            }, "requires card_type_filter='随从'"),
            ({
                "kind": "transform_deck_cards",
                "target": "enemy_leader",
                "card_id": 4,
                "card_id_filter": 2,
            }, "requires own_leader"),
            ({
                "kind": "transform_deck_cards",
                "target": "own_leader",
                "card_id": 4,
            }, "requires at least one deck filter"),
            ({
                "kind": "transform_deck_cards",
                "target": "own_leader",
                "card_id_filter": 2,
            }, "requires card_id"),
        )
        for operation, message in invalid:
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, message):
                    _write_rulebook({
                        "rules": [{
                            "card_id": 1,
                            "trigger": "play",
                            "operations": [operation],
                        }],
                    })

    def test_ignores_ward_passive_rejects_amount_and_keyword(self):
        for extra in ({"amount": 1}, {"keyword": "守护"}):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    _write_rulebook({
                        "passives": [{
                            "card_id": 1,
                            "kind": "ignores_ward",
                            **extra,
                        }],
                    })


class OfficialSourceAndAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "cards.sqlite3",
        )
        cls.repository = CardRepository(cls.db_path)
        cls.coverage_report = json.loads(
            Path("data/reports/rule_coverage.json").read_text(encoding="utf-8")
        )

    def test_database_multilingual_text_stats_modes_and_references(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, expected in DATABASE_CONTRACTS.items():
                with self.subTest(card_id=card_id):
                    card = self.repository.get(card_id)
                    english = connection.execute(
                        "SELECT name FROM card_names "
                        "WHERE card_id = ? AND language = 'en'",
                        (card_id,),
                    ).fetchone()[0]
                    self.assertEqual(
                        (
                            english,
                            card.class_id,
                            card.cost,
                            card.card_type,
                            card.attack,
                            card.life,
                        ),
                        expected,
                    )
                    self.assertTrue(card.is_collectible)
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
                    text = connection.execute(
                        "SELECT text_chs, text_cht, text_eng, text_jpn, text_kor "
                        "FROM skill_texts WHERE card_id = ?",
                        (card_id,),
                    ).fetchone()
                    self.assertTrue(all(text))
                    raw = json.loads(connection.execute(
                        "SELECT raw_json FROM cards WHERE card_id = ?",
                        (card_id,),
                    ).fetchone()[0])
                    self.assertEqual(raw["alt_modes"], [])
                    self.assertTrue(raw["skill_texts"])
            self.assertEqual(
                connection.execute(
                    "SELECT referenced_card_id FROM card_references "
                    "WHERE card_id = 10741310 ORDER BY position"
                ).fetchall(),
                [(10741310,), (10742310,)],
            )
            for card_id in set(CARD_IDS) - {10741310}:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM card_references WHERE card_id = ?",
                        (card_id,),
                    ).fetchone()[0],
                    0,
                )

    def test_source_clause_hashes_match_database_snapshot(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            source_map = _load_source_text_map(connection)
        self.assertEqual(
            {
                card_id: _source_text_sha256(source_map.get(card_id, []))
                for card_id in CARD_IDS
            },
            SOURCE_HASHES,
        )

    def test_rulebook_maps_every_official_clause_and_related_card(self):
        rulebook = RuleBook.from_directory("data/rules")
        dark_side = rulebook.operations_for(10201310, Trigger.PLAY)
        self.assertEqual(
            (dark_side[0].kind, dark_side[0].target),
            (EffectKind.BUFF_UNIT, TargetKind.ANY_UNIT),
        )
        self.assertTrue(dark_side[0].requires_target)
        self.assertEqual(
            (dark_side[0].amount, dark_side[0].secondary_amount),
            (2, -2),
        )
        grandeur = rulebook.operations_for(10533310, Trigger.PLAY)
        self.assertEqual(
            grandeur[0].kind,
            EffectKind.TRANSFORM_BOARD_FROM_RANDOM_OWN_DECK,
        )
        self.assertEqual(grandeur[0].deck_filter.card_type, "随从")
        self.assertEqual(rulebook.intrinsic_keywords_for(10572120), ("守护",))
        bunny = rulebook.listeners_for(10572120)
        self.assertEqual(bunny[0].event, EventType.CARD_PLAYED)
        self.assertEqual(bunny[0].event_filter.card_type, "法术")
        self.assertEqual(
            bunny[0].operations[0].kind,
            EffectKind.EVOLVE_UNIT,
        )
        gaze = rulebook.operations_for(10741310, Trigger.PLAY)
        self.assertEqual(
            [operation.kind for operation in gaze],
            [
                EffectKind.CHANGE_MAX_MANA,
                EffectKind.TRANSFORM,
                EffectKind.TRANSFORM_DECK_CARDS,
            ],
        )
        self.assertEqual(gaze[1].hand_filter, HandFilter(card_id=10741310))
        self.assertEqual(gaze[2].deck_filter, DeckFilter(card_id=10741310))
        self.assertEqual(gaze[1].card_id, 10742310)
        self.assertEqual(gaze[2].card_id, 10742310)
        self.assertEqual(rulebook.intrinsic_keywords_for(10851110), ("疾驰",))
        self.assertTrue(rulebook.ignores_ward(10851110))

    def test_official_faq_and_provenance_metadata_are_auditable(self):
        registry = json.loads(
            Path("data/audits/rule_clauses.json").read_text(encoding="utf-8")
        )
        by_id = {entry["card_id"]: entry for entry in registry["cards"]}
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                entry = by_id[card_id]
                self.assertEqual(entry["coverage"], "exact")
                self.assertEqual(
                    entry["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertIn(TEST_EVIDENCE, entry["test_evidence"])
                self.assertIn(
                    f"card_id={card_id}",
                    entry["official_source_url"],
                )
                self.assertEqual(entry["official_source_retrieved_at"], "2026-07-25")
                report_metadata = self.coverage_report["classifications"][
                    str(card_id)
                ]["rule_metadata"]
                self.assertEqual(
                    report_metadata["official_source_url"],
                    entry["official_source_url"],
                )
                self.assertEqual(
                    report_metadata["official_source_retrieved_at"],
                    "2026-07-25",
                )
                self.assertEqual(
                    report_metadata["official_ruling"],
                    entry["official_ruling"],
                )
                self.assertTrue(entry["implemented_text"])
        self.assertIn("Lloyd", by_id[10201310]["official_ruling"])
        self.assertIn("cannot select an allied follower", by_id[10201310]["official_ruling"])
        self.assertIn("independently", by_id[10533310]["official_ruling"])
        self.assertIn("with replacement", by_id[10533310]["official_ruling"])
        self.assertIn("10742310", by_id[10741310]["notes"])

    def test_final_coverage_clause_token_and_ability_audits_are_clean(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        token_audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        ability_audit = build_ability_audit("data/audits/ability_registry.json")
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
        self.assertEqual(
            report["summary"]["blocker_counts"],
            {
                "missing_rule": 0,
                "missing_schema": 0,
                "missing_primitive": 0,
                "missing_targeting": 0,
                "timing_unclear": 0,
                "text_unclear": 0,
                "external_blocker": 0,
                "audit_unverified": 0,
            },
        )
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertTrue(all(
            info["coverage"] == "covered_exact"
            for info in report["classifications"].values()
            if info["is_collectible"]
        ))
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
        self.assertEqual(
            ability_audit["summary"]["primitive_statuses"],
            {"covered": 34},
        )


class DarkSideBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2201) -> GameEngine:
        return _engine(self.rulebook, self.repository, seed=seed)

    def test_normal_choice_buffs_and_kills_after_source_leaves_hand(self):
        engine = self.fresh()
        own = _put_unit(engine, _card(222001, attack=2, life=4))
        enemy = _put_unit(
            engine,
            _card(222002, attack=3, life=2),
            player_index=1,
        )
        _put_hand(engine, self.repository.get(10201310))
        engine.apply(PlayCard(0, 0))
        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        self.assertEqual(
            {option.option_id for option in request.options},
            {f"entity:{own.entity_id}", f"entity:{enemy.entity_id}"},
        )
        self.assertFalse(any(
            isinstance(card, HandCard) and card.card_id == 10201310
            for card in engine.players[0].hand
        ))
        engine.apply(Choose(0, f"entity:{enemy.entity_id}"))
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertEqual(engine.players[1].shadows, 1)
        self.assertEqual(own.attack, 2)
        engine.assert_invariants()

    def test_lloyd_forces_the_only_legal_target_per_official_faq(self):
        engine = self.fresh()
        own = _put_unit(engine, _card(222010, attack=1, life=5))
        lloyd = _put_unit(
            engine,
            self.repository.get(90074120),
            player_index=1,
        )
        other = _put_unit(
            engine,
            _card(222011, attack=1, life=5),
            player_index=1,
        )
        _put_hand(engine, self.repository.get(10201310))
        engine.apply(PlayCard(0, 0))
        self.assertEqual(
            [option.option_id for option in engine.state.pending_choice.options],
            [f"entity:{lloyd.entity_id}"],
        )
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, f"entity:{own.entity_id}"))
        self.assertEqual(engine.deterministic_fingerprint(), before)
        engine.apply(Choose(0, f"entity:{lloyd.entity_id}"))
        self.assertEqual((lloyd.attack, lloyd.health), (3, 4))
        self.assertEqual((own.attack, own.health), (1, 5))
        self.assertEqual((other.attack, other.health), (1, 5))

    def test_no_legal_target_is_unplayable_and_atomic_in_engine_and_rl(self):
        engine = self.fresh()
        _put_hand(engine, self.repository.get(10201310))
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.deterministic_fingerprint(), before)

        deck_a = [_card(card_id) for card_id in range(223000, 223040)]
        deck_b = [_card(card_id) for card_id in range(224000, 224040)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=2,
            class_b=2,
            seed=9,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=9)
        core = env.core
        core.players[0].hand.clear()
        core.players[0].hand_entity_ids.clear()
        core.players[0].board.clear()
        core.players[1].board.clear()
        core.players[0].mana = core.players[0].max_mana = 10
        _put_hand(core, self.repository.get(10201310))
        env.invalidate_cache(reason="test setup")
        self.assertFalse(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])
        _put_unit(core, _card(224100), player_index=1)
        env.invalidate_cache(reason="target added")
        self.assertTrue(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])

    def test_choice_target_leaving_play_safely_skips_effect(self):
        engine = self.fresh()
        target = _put_unit(
            engine,
            _card(222020, attack=2, life=4),
            player_index=1,
        )
        _put_hand(engine, self.repository.get(10201310))
        engine.apply(PlayCard(0, 0))
        engine.players[1].board.remove(target)
        engine.apply(Choose(0, f"entity:{target.entity_id}"))
        self.assertIsNone(engine.state.pending_choice)
        engine.assert_invariants()


class DeckAndBoardTransformBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2203) -> GameEngine:
        return _engine(self.rulebook, self.repository, seed=seed)

    def test_grandeur_transforms_every_follower_with_independent_replacement(self):
        engine = self.fresh(seed=17)
        first = _put_unit(engine, _card(225001, attack=8, life=8))
        second = _put_unit(engine, _card(225002, attack=9, life=9))
        first.add_keyword("守护")
        second.evolved = True
        options = (
            _card(225010, name="alpha", attack=1, life=2),
            _card(225011, name="beta", attack=4, life=5),
        )
        engine.players[0].deck = [
            options[0],
            options[1],
            _card(225012, card_type="法术", attack=None, life=None),
        ]
        ids = (first.entity_id, second.entity_id)
        _play(engine, self.repository, 10533310)
        transformed = [entity for entity in engine.players[0].board if isinstance(entity, Unit)]
        self.assertEqual(tuple(entity.entity_id for entity in transformed), ids)
        self.assertTrue(all(
            entity.definition.card_id in {225010, 225011}
            for entity in transformed
        ))
        self.assertTrue(all(not entity.evolved for entity in transformed))
        self.assertTrue(all(not entity.has_keyword("守护") for entity in transformed))
        events = [
            event
            for event in engine.event_history
            if event.type is EventType.BOARD_CARD_TRANSFORMED
        ]
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event.metadata["exact_copy"] for event in events))
        self.assertTrue(all(event.metadata["with_replacement"] for event in events))
        engine.assert_invariants()

    def test_grandeur_copies_physical_deck_buffs_and_resets_old_state(self):
        engine = self.fresh(seed=21)
        target = _put_unit(engine, _card(225020, attack=9, life=9))
        target.add_keyword("必杀")
        target.health = 2
        source = _card(225021, attack=2, life=3)
        engine.players[0].deck = [DeckCard(
            definition=source,
            stat_modifiers=[StatModifier(
                modifier_id=991,
                attack_delta=3,
                health_delta=4,
                duration="permanent",
            )],
        )]
        _play(engine, self.repository, 10533310)
        self.assertEqual(target.entity_id, engine.players[0].board[0].entity_id)
        self.assertEqual(
            (
                target.definition.card_id,
                target.attack,
                target.health,
                target.max_health,
            ),
            (225021, 5, 7, 7),
        )
        self.assertFalse(target.has_keyword("必杀"))
        self.assertNotEqual(target.stat_modifiers[0].modifier_id, 991)

    def test_grandeur_empty_candidate_empty_board_and_full_board_are_safe(self):
        no_candidate = self.fresh()
        original = _put_unit(no_candidate, _card(225030, attack=2, life=6))
        no_candidate.players[0].deck = [
            _card(225031, card_type="法术", attack=None, life=None)
        ]
        _play(no_candidate, self.repository, 10533310)
        self.assertEqual(original.definition.card_id, 225030)

        empty = self.fresh()
        empty.players[0].deck = [_card(225032)]
        _play(empty, self.repository, 10533310)
        self.assertEqual(empty.players[0].board, [])

        full = self.fresh()
        full.players[0].deck = [_card(225033, attack=3, life=4)]
        old_ids = tuple(
            _put_unit(full, _card(225040 + index)).entity_id
            for index in range(full.config.max_board)
        )
        _play(full, self.repository, 10533310)
        self.assertEqual(len(full.players[0].board), full.config.max_board)
        self.assertEqual(
            tuple(entity.entity_id for entity in full.players[0].board),
            old_ids,
        )
        self.assertTrue(all(
            entity.definition.card_id == 225033
            for entity in full.players[0].board
        ))

    def test_grandeur_fixed_seed_replays_identically(self):
        def resolved(seed: int):
            engine = self.fresh(seed=seed)
            for index in range(5):
                _put_unit(engine, _card(225100 + index))
            engine.players[0].deck = [
                _card(225200, attack=1, life=2),
                _card(225201, attack=3, life=4),
            ]
            _play(engine, self.repository, 10533310)
            return (
                tuple(entity.definition.card_id for entity in engine.players[0].board),
                engine.deterministic_fingerprint(),
            )

        first = resolved(31)
        second = resolved(31)
        third = resolved(32)
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], third[0])
        self.assertLess(len(set(first[0])), len(first[0]))

    def test_apathetic_gaze_transforms_all_matching_hand_and_deck_copies(self):
        engine = self.fresh()
        engine.players[0].max_mana = engine.players[0].mana = 6
        source = self.repository.get(10741310)
        replacement = self.repository.get(10742310)
        played = _put_hand(engine, source)
        hand_copy = _put_hand(engine, source)
        hand_copy.cost_modifiers.append(CostModifier(
            modifier_id=990,
            mode="set",
            amount=1,
            duration="permanent",
        ))
        unrelated = _put_hand(engine, self.repository.get(10742310))
        modified_deck_copy = DeckCard(
            definition=source,
            stat_modifiers=[],
        )
        engine.players[0].deck = [
            source,
            modified_deck_copy,
            replacement,
            _card(225300),
        ]
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].max_mana, 7)
        self.assertNotIn(played, engine.players[0].hand)
        self.assertEqual(hand_copy.card_id, 10742310)
        self.assertEqual(hand_copy.entity_id, engine.players[0].hand[0].entity_id)
        self.assertEqual(hand_copy.cost_modifiers, [])
        self.assertIs(unrelated, engine.players[0].hand[1])
        self.assertEqual(
            [
                card.definition.card_id if isinstance(card, DeckCard) else card.card_id
                for card in engine.players[0].deck
            ],
            [10742310, 10742310, 10742310, 225300],
        )
        self.assertTrue(all(
            not isinstance(card, DeckCard)
            for card in engine.players[0].deck[:2]
        ))
        deck_events = [
            event
            for event in engine.event_history
            if event.type is EventType.DECK_CARD_TRANSFORMED
        ]
        self.assertEqual(len(deck_events), 2)
        self.assertEqual(
            {event.metadata["deck_index"] for event in deck_events},
            {0, 1},
        )
        engine.assert_invariants()

    def test_apathetic_gaze_caps_max_mana_and_works_without_other_copies(self):
        engine = self.fresh()
        engine.players[0].max_mana = engine.players[0].mana = 10
        engine.players[0].deck = [_card(225310)]
        _play(engine, self.repository, 10741310)
        self.assertEqual((engine.players[0].max_mana, engine.players[0].mana), (10, 6))
        self.assertFalse(any(
            event.type is EventType.DECK_CARD_TRANSFORMED
            for event in engine.event_history
        ))


class ListenerAndWardIgnoreBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 2205) -> GameEngine:
        return _engine(self.rulebook, self.repository, seed=seed)

    def test_lunar_bunny_has_ward_and_evolves_on_each_own_spell_play(self):
        engine = self.fresh()
        _play(engine, self.repository, 10572120)
        bunny = next(
            entity
            for entity in engine.players[0].board
            if isinstance(entity, Unit) and entity.definition.card_id == 10572120
        )
        self.assertTrue(bunny.has_keyword("守护"))
        self.assertFalse(bunny.evolved)
        points = engine.players[0].evolution_points
        spell = self.repository.get(10742310)
        _put_hand(engine, spell)
        engine.apply(PlayCard(0, 0))
        self.assertTrue(bunny.evolved)
        self.assertEqual((bunny.attack, bunny.health), (3, 3))
        self.assertEqual(engine.players[0].evolution_points, points)
        _put_hand(engine, spell)
        engine.apply(PlayCard(0, 0))
        self.assertEqual((bunny.attack, bunny.health), (3, 3))

    def test_lunar_bunny_ignores_nonspell_and_stops_after_leaving_play(self):
        engine = self.fresh()
        _play(engine, self.repository, 10572120)
        bunny = engine.players[0].board[0]
        _put_hand(engine, _card(226010))
        engine.apply(PlayCard(0, 0))
        self.assertFalse(bunny.evolved)
        engine.players[0].board.remove(bunny)
        _put_hand(engine, self.repository.get(10742310))
        engine.apply(PlayCard(0, 0))
        self.assertFalse(bunny.evolved)
        engine.assert_invariants()

    def test_anisage_storm_can_attack_any_target_while_ward_exists(self):
        engine = self.fresh()
        _play(engine, self.repository, 10851110)
        anisage = engine.players[0].board[0]
        guard = _put_unit(
            engine,
            _card(226020, keywords=frozenset({"守护"})),
            player_index=1,
        )
        other = _put_unit(engine, _card(226021), player_index=1)
        legal_attacks = {
            command.target_id
            for command in engine.legal_commands()
            if isinstance(command, Attack)
            and command.attacker_id == anisage.entity_id
        }
        self.assertEqual(legal_attacks, {None, guard.entity_id, other.entity_id})
        engine.apply(Attack(0, anisage.entity_id, None))
        self.assertEqual(engine.players[1].health, 18)

    def test_remove_all_abilities_removes_ward_ignore_and_storm(self):
        engine = self.fresh()
        anisage = _put_unit(engine, self.repository.get(10851110))
        guard = _put_unit(
            engine,
            _card(226030, keywords=frozenset({"守护"})),
            player_index=1,
        )
        other = _put_unit(engine, _card(226031), player_index=1)
        anisage.remove_all_abilities()
        anisage.summoned_this_turn = False
        anisage.can_attack = True
        self.assertFalse(anisage.has_keyword("疾驰"))
        legal_attacks = {
            command.target_id
            for command in engine.legal_commands()
            if isinstance(command, Attack)
            and command.attacker_id == anisage.entity_id
        }
        self.assertEqual(legal_attacks, {guard.entity_id})
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Attack(0, anisage.entity_id, other.entity_id))
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_anisage_action_mask_matches_command_layer_with_ward(self):
        deck_a = [_card(card_id) for card_id in range(226100, 226140)]
        deck_b = [_card(card_id) for card_id in range(226200, 226240)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=2,
            class_b=2,
            seed=15,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=15)
        core = env.core
        for player in core.players:
            player.board.clear()
        anisage = _put_unit(core, self.repository.get(10851110))
        guard = _put_unit(
            core,
            _card(226300, keywords=frozenset({"守护"})),
            player_index=1,
        )
        other = _put_unit(core, _card(226301), player_index=1)
        anisage.can_attack = True
        env.invalidate_cache(reason="test board")
        mask = env.action_mask()
        base = ShadowverseEnv.ATTACK_OFFSET
        self.assertTrue(mask[base])
        self.assertTrue(mask[base + 1])
        self.assertTrue(mask[base + 2])
        encoded = {
            env._encode_command(command)
            for command in core.legal_commands()
            if isinstance(command, Attack)
        }
        self.assertEqual(encoded, {base, base + 1, base + 2})
        self.assertEqual(
            {
                env._decode_action(action).target_id
                for action in encoded
            },
            {None, guard.entity_id, other.entity_id},
        )


if __name__ == "__main__":
    unittest.main()
