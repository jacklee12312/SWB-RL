# -*- coding: utf-8 -*-
"""Official-source closure tests for keyword-only and vanilla followers."""

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
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Attack, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import (
    GameConfig,
    GameEngine,
    IllegalCommand,
)
from swb.engine.state import HandCard, Unit


CARD_IDS = (
    10001130,
    10002120,
    10021110,
    10021130,
    10041120,
    10061120,
    10143110,
    10211110,
    10221120,
    10422120,
    10612120,
)
KEYWORD_CARDS = {
    10001130: ("守护",),
    10021110: ("疾驰",),
    10021130: ("疾驰",),
    10041120: ("守护",),
    10061120: ("守护",),
    10143110: ("疾驰",),
    10211110: ("突进",),
    10221120: ("潜行",),
    10612120: ("疾驰",),
}
VANILLA_CARDS = {
    10002120: ("商队猛犸象", "Caravan Mammoth", 0, 7, 10, 10),
    10422120: ("如火斗志·菲泽", "Feather, Bombastic Brawler", 2, 3, 5, 4),
}
DATABASE_CONTRACTS = {
    10001130: ("激震的歌利亚", "Quake Goliath", 0, 4, 4, 5),
    10002120: VANILLA_CARDS[10002120],
    10021110: ("须臾剑士", "Flashstep Quickblader", 2, 1, 1, 1),
    10021130: ("人马骑士", "Centaur Centurion", 2, 8, 7, 5),
    10041120: ("战斧屠龙者", "Axe-Wielding Dragonslayer", 4, 6, 5, 10),
    10061120: ("纯洁白狐", "Fox of Purity", 6, 2, 1, 3),
    10143110: ("再临之创世龙", "Genesis Dragon Reborn", 4, 10, 9, 10),
    10211110: ("狂野女孩", "Wildheart", 1, 3, 3, 3),
    10221120: ("暗斗的忍者大师", "Nightshadow Ninja Master", 2, 5, 4, 5),
    10422120: VANILLA_CARDS[10422120],
    10612120: ("咆哮狼人", "Howling Wolfman", 1, 6, 5, 3),
}
SOURCE_HASHES = {
    10001130: "fd22b46c04522f13395a9cb2a5444e83007a05cf6121d94ba170126602ed58c5",
    10002120: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    10021110: "487bbc3ead279c8faccb6c74842ddc77225535cb22afa3b651dcdc587a172eb8",
    10021130: "487bbc3ead279c8faccb6c74842ddc77225535cb22afa3b651dcdc587a172eb8",
    10041120: "fd22b46c04522f13395a9cb2a5444e83007a05cf6121d94ba170126602ed58c5",
    10061120: "fd22b46c04522f13395a9cb2a5444e83007a05cf6121d94ba170126602ed58c5",
    10143110: "487bbc3ead279c8faccb6c74842ddc77225535cb22afa3b651dcdc587a172eb8",
    10211110: "7abde1fa8f2c436a6805d99abe026718965b9776b4b5a2dffe05f57cd5b2776f",
    10221120: "bad7d96c9e09aec7bc75d1363de4a29b5bea77a57abd6969190e8ac616b30881",
    10422120: "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    10612120: "487bbc3ead279c8faccb6c74842ddc77225535cb22afa3b651dcdc587a172eb8",
}
TEST_EVIDENCE = (
    "tests/test_real_official_basic_completion_twenty_first_batch.py"
)


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=0,
        class_name="中立",
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 3),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=True,
    )


def _make_engine(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 2101,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=2,
        class_b=2,
        seed=seed,
        rulebook=rulebook,
        card_resolver=repository.get,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    engine.players[0].hand.clear()
    engine.players[0].hand_entity_ids.clear()
    engine.players[0].max_mana = engine.players[0].mana = 10
    return engine


def _put_in_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[0].hand.append(hand_card)
    engine.players[0].hand_entity_ids.append(hand_card.entity_id)
    return hand_card


def _play_real(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
) -> Unit:
    _put_in_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, 0))
    return next(
        entity
        for entity in engine.players[0].board
        if isinstance(entity, Unit)
        and entity.definition.card_id == card_id
    )


def _write_rule_directory(payloads: list[dict]) -> RuleBook:
    directory = tempfile.TemporaryDirectory()
    path = Path(directory.name)
    for index, payload in enumerate(payloads):
        (path / f"rules-{index}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    try:
        return RuleBook.from_directory(path)
    finally:
        directory.cleanup()


class VanillaDeclarationSchemaTests(unittest.TestCase):
    def test_schema_loads_audited_vanilla_and_keyword_declarations(self):
        rulebook = _write_rule_directory([{
            "vanilla_cards": [{
                "card_id": 12345678,
                "notes": "Official text is empty.",
            }],
            "intrinsic_keywords": [{
                "card_id": 12345679,
                "keywords": ["守护"],
            }],
        }])
        self.assertTrue(rulebook.is_explicit_vanilla(12345678))
        self.assertFalse(rulebook.is_explicit_vanilla(12345679))
        self.assertEqual(
            rulebook.intrinsic_keywords_for(12345679),
            ("守护",),
        )

    def test_schema_rejects_invalid_container_item_id_fields_and_notes(self):
        invalid_payloads = (
            ({"vanilla_cards": {}}, "must be a list"),
            ({"vanilla_cards": [1]}, "must be an object"),
            (
                {"vanilla_cards": [{"card_id": True, "notes": "official"}]},
                "positive integer",
            ),
            (
                {"vanilla_cards": [{"card_id": 1}]},
                "notes: must be a non-empty string",
            ),
            (
                {"vanilla_cards": [{"card_id": 1, "notes": "  "}]},
                "notes: must be a non-empty string",
            ),
            (
                {
                    "vanilla_cards": [{
                        "card_id": 1,
                        "notes": "official",
                        "extra": 1,
                    }],
                },
                "unknown fields",
            ),
        )
        for payload, message in invalid_payloads:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _write_rule_directory([payload])

    def test_duplicate_and_behavior_overlap_are_rejected(self):
        definition = {
            "vanilla_cards": [{"card_id": 12345678, "notes": "official"}],
        }
        with self.assertRaisesRegex(ValueError, "duplicate vanilla definition"):
            _write_rule_directory([definition, definition])

        with self.assertRaisesRegex(
            ValueError,
            "vanilla cards cannot also have behavior definitions",
        ):
            _write_rule_directory([{
                **definition,
                "intrinsic_keywords": [{
                    "card_id": 12345678,
                    "keywords": ["守护"],
                }],
            }])


class OfficialDatabaseAndAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "cards.sqlite3",
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")
        cls.repository = CardRepository(cls.db_path)

    def test_database_contracts_multilingual_text_and_raw_json_are_complete(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, expected in DATABASE_CONTRACTS.items():
                with self.subTest(card_id=card_id):
                    card = self.repository.get(card_id)
                    self.assertEqual(
                        (
                            card.name,
                            connection.execute(
                                "SELECT name FROM card_names "
                                "WHERE card_id = ? AND language = 'en'",
                                (card_id,),
                            ).fetchone()[0],
                            card.class_id,
                            card.cost,
                            card.attack,
                            card.life,
                        ),
                        expected,
                    )
                    self.assertEqual(card.card_type, "随从")
                    self.assertTrue(card.is_collectible)
                    self.assertEqual(
                        tuple(sorted(card.keywords)),
                        KEYWORD_CARDS.get(card_id, ()),
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM card_references "
                            "WHERE card_id = ?",
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
                    names = dict(connection.execute(
                        "SELECT language, name FROM card_names "
                        "WHERE card_id = ?",
                        (card_id,),
                    ))
                    self.assertEqual(
                        set(names),
                        {"en", "ja", "ko", "zh-CN", "zh-TW"},
                    )
                    raw = json.loads(connection.execute(
                        "SELECT raw_json FROM cards WHERE card_id = ?",
                        (card_id,),
                    ).fetchone()[0])
                    self.assertEqual(raw["name_chs"], card.name)
                    self.assertEqual(raw["name_eng"], expected[1])
                    self.assertEqual(raw["alt_modes"], [])
                    if card_id in VANILLA_CARDS:
                        self.assertEqual(raw["skills"], [])
                        self.assertEqual(raw["skill_texts"], [])
                    else:
                        self.assertEqual(len(raw["skills"]), 1)
                        self.assertEqual(len(raw["skill_texts"]), 1)
                        text = raw["skill_texts"][0]
                        for language in ("chs", "cht", "eng", "jpn", "kor"):
                            self.assertTrue(text[f"text_{language}"])

    def test_source_clause_hashes_match_database_snapshot(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.row_factory = sqlite3.Row
            source_map = _load_source_text_map(connection)
        self.assertEqual(
            {
                card_id: _source_text_sha256(source_map.get(card_id, []))
                for card_id in CARD_IDS
            },
            SOURCE_HASHES,
        )

    def test_rulebook_shapes_distinguish_keywords_from_no_ability(self):
        rulebook = RuleBook.from_directory("data/rules")
        for card_id, keywords in KEYWORD_CARDS.items():
            with self.subTest(card_id=card_id):
                self.assertEqual(
                    rulebook.intrinsic_keywords_for(card_id),
                    keywords,
                )
                self.assertFalse(rulebook.is_explicit_vanilla(card_id))
        for card_id in VANILLA_CARDS:
            with self.subTest(card_id=card_id):
                self.assertTrue(rulebook.is_explicit_vanilla(card_id))
                self.assertEqual(rulebook.intrinsic_keywords_for(card_id), ())

    def test_clause_coverage_token_and_ability_audits_are_consistent(self):
        report = _build_coverage_report(
            "data/cards.sqlite3",
            "data/rules",
        )
        token_audit = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
        )
        ability_audit = build_ability_audit(
            "data/audits/ability_registry.json"
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
        self.assertEqual(ability_audit["summary"]["total"], 34)
        self.assertEqual(
            ability_audit["summary"]["primitive_statuses"],
            {"covered": 34},
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
                expected_evidence = (
                    {
                        "triggers": ["vanilla_declaration"],
                        "effect_kinds": ["no_printed_ability"],
                    }
                    if card_id in VANILLA_CARDS
                    else {
                        "triggers": ["intrinsic_keywords"],
                        "effect_kinds": [
                            f"keyword:{keyword}"
                            for keyword in KEYWORD_CARDS[card_id]
                        ],
                    }
                )
                self.assertEqual(
                    info["clause_audit"]["structured_evidence"],
                    expected_evidence,
                )

    def test_audit_records_official_source_date_and_direct_evidence(self):
        audit = json.loads(
            Path("data/audits/rule_clauses.json").read_text(
                encoding="utf-8"
            )
        )
        entries = {
            entry["card_id"]: entry
            for entry in audit["cards"]
            if entry["card_id"] in CARD_IDS
        }
        self.assertEqual(set(entries), set(CARD_IDS))
        for card_id, entry in entries.items():
            with self.subTest(card_id=card_id):
                self.assertEqual(entry["coverage"], "exact")
                self.assertEqual(
                    entry["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(entry["test_evidence"], [TEST_EVIDENCE])
                self.assertIn("2026-07-25", entry["notes"])
                self.assertIn(
                    "https://shadowverse-wb.com/en/deck/cardslist/card/"
                    f"?card_id={card_id}",
                    entry["notes"],
                )


class OfficialBasicFollowerBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 2101) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def test_every_real_card_enters_with_exact_stats_and_keyword_state(self):
        for card_id, expected in DATABASE_CONTRACTS.items():
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine(seed=card_id)
                unit = _play_real(engine, self.repository, card_id)
                self.assertEqual(
                    (unit.attack, unit.health, unit.max_health),
                    (expected[4], expected[5], expected[5]),
                )
                self.assertEqual(
                    unit.effective_keywords,
                    frozenset(KEYWORD_CARDS.get(card_id, ())),
                )
                self.assertEqual(
                    unit.has_guard,
                    "守护" in KEYWORD_CARDS.get(card_id, ()),
                )
                self.assertEqual(
                    unit.ambush_active,
                    "潜行" in KEYWORD_CARDS.get(card_id, ()),
                )
                self.assertEqual(
                    unit.can_attack,
                    bool(
                        {"疾驰", "突进"}
                        & set(KEYWORD_CARDS.get(card_id, ()))
                    ),
                )
                self.assertEqual(engine.placeholder_ability_events, [])

    def test_vanilla_followers_have_no_hidden_runtime_behavior(self):
        for card_id in VANILLA_CARDS:
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine(seed=card_id + 1)
                mana_before = engine.players[0].mana
                unit = _play_real(engine, self.repository, card_id)
                definition = self.repository.get(card_id)
                self.assertEqual(
                    engine.players[0].mana,
                    mana_before - definition.cost,
                )
                self.assertFalse(unit.can_attack)
                self.assertFalse(unit.can_attack_leader)
                self.assertFalse(unit.rush_only)
                self.assertEqual(unit.effective_keywords, frozenset())
                self.assertEqual(engine.state.pending_choice, None)
                self.assertEqual(engine.placeholder_ability_events, [])

    def test_all_real_ward_followers_force_combat_targeting(self):
        for card_id in (10001130, 10041120, 10061120):
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine(seed=card_id + 3)
                ward = _play_real(engine, self.repository, card_id)
                other = Unit.summon(
                    _card(900001, attack=1, life=8),
                    entity_id=engine.state.allocate_entity_id(),
                )
                attacker = Unit.summon(
                    _card(900002, attack=2, life=8),
                    entity_id=engine.state.allocate_entity_id(),
                )
                attacker.can_attack = True
                attacker.attacks_remaining = 1
                engine.players[0].board.append(other)
                engine.players[1].board.append(attacker)
                engine.state.active_player = 1

                legal = engine.legal_commands()
                self.assertIn(
                    Attack(1, attacker.entity_id, ward.entity_id),
                    legal,
                )
                self.assertNotIn(Attack(1, attacker.entity_id, None), legal)
                illegal = Attack(1, attacker.entity_id, other.entity_id)
                self.assertNotIn(illegal, legal)
                before = engine.snapshot().payload
                with self.assertRaisesRegex(
                    IllegalCommand,
                    "guard follower must be attacked",
                ):
                    engine.apply(illegal)
                self.assertEqual(engine.snapshot().payload, before)

    def test_all_real_storm_followers_can_attack_leader_immediately(self):
        for card_id in (10021110, 10021130, 10143110, 10612120):
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine(seed=card_id + 5)
                storm = _play_real(engine, self.repository, card_id)
                command = Attack(0, storm.entity_id, None)
                self.assertIn(command, engine.legal_commands())
                health_before = engine.players[1].health
                engine.apply(command)
                self.assertEqual(
                    engine.players[1].health,
                    health_before - storm.attack,
                )

    def test_real_rush_has_unit_only_attack_and_illegal_leader_is_atomic(self):
        engine = self.fresh_engine(seed=2111)
        target = Unit.summon(
            _card(900003, attack=1, life=8),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board.append(target)
        rush = _play_real(engine, self.repository, 10211110)
        unit_attack = Attack(0, rush.entity_id, target.entity_id)
        leader_attack = Attack(0, rush.entity_id, None)
        self.assertIn(unit_attack, engine.legal_commands())
        self.assertNotIn(leader_attack, engine.legal_commands())
        before = engine.snapshot().payload
        with self.assertRaisesRegex(
            IllegalCommand,
            "Leader is not a legal target",
        ):
            engine.apply(leader_attack)
        self.assertEqual(engine.snapshot().payload, before)
        engine.apply(unit_attack)
        self.assertEqual(target.health, 5)

    def test_real_rush_with_no_unit_target_is_playable_but_cannot_attack(self):
        engine = self.fresh_engine(seed=2113)
        rush = _play_real(engine, self.repository, 10211110)
        self.assertTrue(rush.can_attack)
        self.assertEqual(
            [
                command
                for command in engine.legal_commands()
                if isinstance(command, Attack)
                and command.attacker_id == rush.entity_id
            ],
            [],
        )

    def test_real_ambush_is_neither_attackable_nor_manual_enemy_target(self):
        engine = self.fresh_engine(seed=2117)
        ambush = _play_real(engine, self.repository, 10221120)
        attacker = Unit.summon(
            _card(900004, attack=2, life=8),
            entity_id=engine.state.allocate_entity_id(),
        )
        attacker.can_attack = True
        attacker.attacks_remaining = 1
        engine.players[1].board.append(attacker)
        engine.state.active_player = 1

        illegal_attack = Attack(1, attacker.entity_id, ambush.entity_id)
        self.assertNotIn(illegal_attack, engine.legal_commands())
        before = engine.snapshot().payload
        with self.assertRaisesRegex(
            IllegalCommand,
            "Cannot attack an ambush follower",
        ):
            engine.apply(illegal_attack)
        self.assertEqual(engine.snapshot().payload, before)

        targeting_rulebook = RuleBook((CardRule(
            card_id=900005,
            trigger=Trigger.PLAY,
            operations=(EffectOperation(
                kind=EffectKind.DAMAGE_UNIT,
                target=TargetKind.ENEMY_UNIT,
                amount=1,
            ),),
        ),))
        targeting_engine = _make_engine(
            targeting_rulebook,
            self.repository,
            seed=2118,
        )
        real_ambush = _play_real(
            targeting_engine,
            self.repository,
            10221120,
        )
        spell = HandCard(
            definition=_card(
                900005,
                card_type="法术",
                attack=None,
                life=None,
            ),
            entity_id=targeting_engine.state.allocate_entity_id(),
        )
        targeting_engine.players[1].hand = [spell]
        targeting_engine.players[1].hand_entity_ids = [spell.entity_id]
        targeting_engine.players[1].max_mana = 10
        targeting_engine.players[1].mana = 10
        targeting_engine.state.active_player = 1
        self.assertTrue(real_ambush.ambush_active)
        self.assertNotIn(
            PlayCard(1, 0),
            targeting_engine.legal_commands(),
        )

    def test_full_board_rejects_real_follower_play_without_mutation(self):
        engine = self.fresh_engine(seed=2119)
        for index in range(engine.config.max_board):
            engine.players[0].board.append(Unit.summon(
                _card(900100 + index),
                entity_id=engine.state.allocate_entity_id(),
            ))
        _put_in_hand(engine, self.repository.get(10021110))
        command = PlayCard(0, 0)
        self.assertNotIn(command, engine.legal_commands())
        before = engine.snapshot().payload
        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.snapshot().payload, before)

    def test_same_seed_and_command_sequence_reproduces_identical_state(self):
        first = self.fresh_engine(seed=2123)
        second = self.fresh_engine(seed=2123)
        for engine in (first, second):
            storm = _play_real(engine, self.repository, 10021110)
            engine.apply(Attack(0, storm.entity_id, None))
        self.assertEqual(first.snapshot().payload, second.snapshot().payload)

    def test_rl_masks_match_storm_and_rush_legal_commands(self):
        cases = (
            (10021110, False),
            (10211110, True),
        )
        for card_id, needs_target in cases:
            with self.subTest(card_id=card_id):
                env = ShadowverseEnv(
                    [_card(card_id) for card_id in range(3000, 3040)],
                    [_card(card_id) for card_id in range(4000, 4040)],
                    class_a=2,
                    class_b=2,
                    seed=card_id,
                    rulebook=self.rulebook,
                    card_resolver=self.repository.get,
                )
                env.reset(seed=card_id)
                env.players[0].hand.clear()
                env.players[0].hand_entity_ids.clear()
                env.players[0].max_mana = env.players[0].mana = 10
                if needs_target:
                    env.players[1].board.append(Unit.summon(
                        _card(900200, attack=1, life=8),
                        entity_id=env.core.state.allocate_entity_id(),
                    ))
                _put_in_hand(env.core, self.repository.get(card_id))

                play = PlayCard(0, 0)
                play_action = env._encode_command(play)
                self.assertIsNotNone(play_action)
                self.assertTrue(env.action_mask()[play_action])
                env.step(play_action)

                legal = env.core.legal_commands()
                mask = env.action_mask()
                for command in legal:
                    action = env._encode_command(command)
                    self.assertIsNotNone(action, command)
                    self.assertTrue(mask[action], command)
                enabled_commands = {
                    env._decode_action(action)
                    for action, enabled in enumerate(mask)
                    if enabled
                }
                self.assertEqual(enabled_commands, set(legal))

                played = next(
                    entity
                    for entity in env.players[0].board
                    if isinstance(entity, Unit)
                    and entity.definition.card_id == card_id
                )
                leader_attack = Attack(0, played.entity_id, None)
                leader_action = env._encode_command(leader_attack)
                self.assertIsNotNone(leader_action)
                self.assertEqual(
                    bool(mask[leader_action]),
                    not needs_target,
                )


if __name__ == "__main__":
    unittest.main()
