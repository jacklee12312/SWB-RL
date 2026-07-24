"""Direct contracts for the eleventh exact listener/Enhance rule slice."""

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
    _parse_listener_definition,
    _parse_operation,
)
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType, GameEvent
from swb.engine.resolution import IllegalCommand
from swb.engine.state import AttackRestriction
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10224120,
    10424120,
    10574120,
    10603110,
    10622310,
)
SOURCE_HASHES = {
    10224120: "3797884bf0c276db7b8ce3e367b3c5efe10d4d7bee3c748b2093d905fb3049e0",
    10424120: "122456b1bb6db26f8078b68804692b2c43310015b3f4c6ef990a711ca36ea475",
    10574120: "c8ed7291167d1e8981df89dfd8b2c7aa8cc5ebfd884af52b1b07ffa3ac3208f0",
    10603110: "e1ba3b31431eccfdc411ae79b8b0667d5c8f6c829bc7efd9ee9156489a7daf0b",
    10622310: "e32a640d04b074a2a976312e5b9eb52fd87c82ded68dc3a7ae2d6bca996cf3d4",
}
TEST_EVIDENCE = (
    "tests/test_real_listener_enhance_random_keyword_eleventh_batch.py"
)


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options
        if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolution(engine, *, super_evolution: bool = False) -> None:
    player = engine.players[0]
    player.turns_started = (
        engine.config.first_player_super_evolution_unlock_turn
        if super_evolution
        else engine.config.evolution_unlock_turn
    )
    player.evolution_points = max(1, player.evolution_points)
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False
    engine.state.active_player = 0


class ListenerEnhanceRandomKeywordEleventhBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 11001):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_strict_generic_schema(self):
        julius = self.rulebook.operations_for(10224120, Trigger.FANFARE)
        self.assertEqual(
            [(operation.kind, operation.target, operation.card_id) for operation in julius],
            [
                (EffectKind.SUMMON, TargetKind.ENEMY_LEADER, 90021110),
                (EffectKind.SUMMON, TargetKind.ENEMY_LEADER, 90021110),
            ],
        )
        julius_listener = self.rulebook.listeners_for(10224120)[0]
        self.assertIs(julius_listener.event, EventType.FOLLOWER_SUMMONED)
        self.assertEqual(
            [operation.kind for operation in julius_listener.operations],
            [
                EffectKind.ADD_ATTACK_RESTRICTION,
                EffectKind.DAMAGE_LEADER,
                EffectKind.HEAL_LEADER,
            ],
        )

        bursts = self.rulebook.union_bursts_for(10424120)
        self.assertEqual([burst.threshold for burst in bursts], [10, 15])
        self.assertFalse(bursts[0].replace_lower_bursts)
        self.assertTrue(bursts[1].replace_lower_bursts)
        self.assertIs(bursts[0].operations[0].kind, EffectKind.EVOLVE_UNIT)
        self.assertIs(
            bursts[1].operations[0].kind,
            EffectKind.SUPER_EVOLVE_UNIT,
        )

        imari_super = self.rulebook.operations_for(
            10574120,
            Trigger.SUPER_EVOLVE,
        )[0]
        self.assertIs(imari_super.kind, EffectKind.DRAW_FILTERED)
        self.assertTrue(imari_super.distinct_card_names)
        self.assertEqual(
            (
                imari_super.deck_filter.card_type,
                imari_super.deck_filter.cost_min,
                imari_super.deck_filter.cost_max,
            ),
            ("法术", 1, 1),
        )

        beast = self.rulebook.operations_for(10603110, Trigger.FANFARE)[0]
        self.assertIs(beast.kind, EffectKind.ADD_RANDOM_KEYWORDS)
        self.assertEqual(beast.amount, 3)
        self.assertEqual(
            beast.keywords,
            ("疾驰", "必杀", "威慑", "吸血", "灵气", "屏障"),
        )

        crest = self.rulebook.emblem_def("majestic_conquest")
        self.assertEqual((crest.source_card_id, crest.countdown), (10622310, 2))
        self.assertTrue(crest.triggers[0].event_filter.enhanced)
        self.assertEqual(
            [mode.mode_id for mode in self.rulebook.modes_for(10622310)],
            ["enhance_3"],
        )

        invalid_random_operations = (
            {
                "kind": "add_random_keywords",
                "target": "own_leader",
                "amount": 1,
                "keywords": ["守护"],
            },
            {
                "kind": "add_random_keywords",
                "target": "self",
                "amount": 2,
                "keywords": ["毁灭", "必杀"],
            },
            {
                "kind": "add_random_keywords",
                "target": "self",
                "amount": 2,
                "keywords": ["守护"],
            },
        )
        for operation in invalid_random_operations:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                _parse_operation(operation, "test.json", 1)
        with self.assertRaisesRegex(ValueError, "only valid for card_played"):
            _parse_listener_definition(
                {
                    "card_id": 1,
                    "zone": "board",
                    "event": "follower_summoned",
                    "event_filter": {"enhanced": True},
                    "operations": [
                        {
                            "kind": "draw",
                            "target": "own_leader",
                            "amount": 1,
                        }
                    ],
                },
                "test/listeners[0]",
            )

    def test_julius_enemy_summons_listener_capacity_expiry_and_departure(self):
        engine = self.fresh(seed=11)
        engine.players[0].health = 17

        source = _play(engine, self.repository, 10224120)

        knights = [
            unit for unit in engine.players[1].board
            if unit.definition.card_id == 90021110
        ]
        self.assertEqual(len(knights), 2)
        self.assertEqual(
            (engine.players[0].health, engine.players[1].health),
            (19, 18),
        )
        for knight in knights:
            self.assertTrue(
                any(
                    modifier.restriction is AttackRestriction.CANNOT_ATTACK
                    for modifier in knight.attack_restrictions
                )
            )

        engine.apply(EndTurn(0))
        self.assertTrue(all(not knight.can_attack_units for knight in knights))
        engine.apply(EndTurn(1))
        self.assertTrue(all(knight.can_attack_units for knight in knights))
        self.assertTrue(all(not knight.attack_restrictions for knight in knights))

        source.health = 0
        engine._stabilize()
        before_health = (
            engine.players[0].health,
            engine.players[1].health,
        )
        late_enemy = _put_unit(engine, 1, _card(991101))
        engine._emit(
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                1,
                source_id=late_enemy.entity_id,
                metadata={"source": late_enemy},
            )
        )
        engine._resolve_event_queue()
        engine._stabilize()
        self.assertEqual(
            (engine.players[0].health, engine.players[1].health),
            before_health,
        )

        limited = self.fresh(seed=13)
        limited.players[0].health = 18
        for index in range(4):
            _put_unit(limited, 1, _card(991110 + index))
        _play(limited, self.repository, 10224120)
        self.assertEqual(len(limited.players[1].board), 5)
        self.assertEqual(
            sum(
                unit.definition.card_id == 90021110
                for unit in limited.players[1].board
            ),
            1,
        )
        self.assertEqual(
            (limited.players[0].health, limited.players[1].health),
            (19, 19),
        )
        limited.assert_invariants()

    def test_siete_thresholds_and_super_art_replaces_lower_burst(self):
        outcomes = {}
        for gauge in (9, 10, 15):
            engine = self.fresh(seed=17)
            engine.players[0].turns_started = gauge
            ally = _put_unit(engine, 0, _card(991201, attack=1, life=3))
            already_evolved = _put_unit(
                engine,
                0,
                _card(991202, attack=1, life=3),
            )
            already_evolved.evolved = True

            source = _play(engine, self.repository, 10424120)

            outcomes[gauge] = (
                ally.evolved,
                ally.super_evolved,
                source.evolved,
                source.super_evolved,
            )
            self.assertTrue(already_evolved.evolved)
            self.assertFalse(already_evolved.super_evolved)
            burst_events = [
                event.metadata["kind"]
                for event in engine.event_history
                if event.type is EventType.UNION_BURST_ACTIVATED
            ]
            if gauge == 9:
                self.assertEqual(burst_events, [])
            elif gauge == 10:
                self.assertEqual(burst_events, ["union_burst"])
            else:
                self.assertEqual(burst_events, ["super_skybound_art"])
                self.assertFalse(
                    any(
                        event.type is EventType.FOLLOWER_EVOLVED
                        and event.source_id in {ally.entity_id, source.entity_id}
                        for event in engine.event_history
                    )
                )
            engine.assert_invariants()

        self.assertEqual(outcomes[9], (False, False, False, False))
        self.assertEqual(outcomes[10], (True, False, True, False))
        self.assertEqual(outcomes[15], (True, True, True, True))

    def test_imari_choice_action_mask_illegal_and_stale_targets(self):
        deck = [
            _card(
                991300 + index,
                class_id=0,
                class_name="中立",
            )
            for index in range(40)
        ]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=7,
            class_b=7,
            seed=23,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=23)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        env.players[0].deck = [
            _card(
                991390,
                name="filtered-spell",
                card_type="法术",
                attack=None,
                life=None,
            )
        ]
        discarded = _put_hand(env.core, _card(991391))
        imari = _put_hand(env.core, self.repository.get(10574120))

        env.core.apply(
            PlayCard(0, env.players[0].hand.index(imari))
        )
        env.invalidate_cache(reason="Imari discard choice")
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        self.assertEqual(len(decoded), 1)
        before = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(Choose(0, "hand:999999"))
        self.assertEqual(env.core.deterministic_fingerprint(), before)

        _choose_entity(env.core, discarded.entity_id)
        self.assertIsNone(env.core.state.pending_choice)
        self.assertTrue(
            any(card.card_id == 991390 for card in env.players[0].hand)
        )
        self.assertTrue(
            any(
                card.definition.card_id == 991391
                for card in env.players[0].graveyard
            )
        )

        stale = self.fresh(seed=29)
        stale.players[0].deck = [
            _card(
                991392,
                name="replacement-spell",
                card_type="法术",
                attack=None,
                life=None,
            )
        ]
        stale_target = _put_hand(stale, _card(991393))
        source = _put_hand(stale, self.repository.get(10574120))
        stale.apply(PlayCard(0, stale.players[0].hand.index(source)))
        stale.players[0].hand.remove(stale_target)
        stale.players[0].hand_entity_ids.remove(stale_target.entity_id)
        _choose_entity(stale, stale_target.entity_id)
        self.assertIsNone(stale.state.pending_choice)
        self.assertTrue(
            any(card.card_id == 991392 for card in stale.players[0].hand)
        )
        stale.assert_invariants()

        empty = self.fresh(seed=31)
        empty.players[0].deck = [
            _card(
                991394,
                card_type="法术",
                attack=None,
                life=None,
            )
        ]
        _play(empty, self.repository, 10574120)
        self.assertIsNone(empty.state.pending_choice)
        self.assertTrue(
            any(card.card_id == 991394 for card in empty.players[0].hand)
        )

    def test_imari_evolved_spell_listener_and_distinct_super_draw_capacity(self):
        engine = self.fresh(seed=37)
        discarded = _put_hand(engine, _card(991401))
        source = _play(engine, self.repository, 10574120)
        _choose_entity(engine, discarded.entity_id)
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        spell = _put_hand(
            engine,
            self.repository.get(10622310),
        )
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, engine.players[0].hand.index(spell)))
        self.assertEqual(
            sum(
                unit.definition.card_id == 90074140
                for unit in engine.players[0].board
            ),
            1,
        )
        source.health = 0
        engine._stabilize()
        second_spell = _put_hand(
            engine,
            self.repository.get(10622310),
        )
        engine.players[0].mana = 10
        engine.apply(
            PlayCard(0, engine.players[0].hand.index(second_spell))
        )
        self.assertEqual(
            sum(
                unit.definition.card_id == 90074140
                for unit in engine.players[0].board
            ),
            1,
        )

        super_engine = self.fresh(seed=41)
        super_source = _put_unit(
            super_engine,
            0,
            self.repository.get(10574120),
        )
        for index in range(8):
            _put_hand(super_engine, _card(991410 + index))
        candidates = [
            _card(
                991420,
                name="same",
                cost=1,
                card_type="法术",
                attack=None,
                life=None,
            ),
            _card(
                991421,
                name="same",
                cost=1,
                card_type="法术",
                attack=None,
                life=None,
            ),
            _card(
                991422,
                name="other",
                cost=1,
                card_type="法术",
                attack=None,
                life=None,
            ),
            _card(
                991423,
                name="wrong-cost",
                cost=2,
                card_type="法术",
                attack=None,
                life=None,
            ),
        ]
        super_engine.players[0].deck = list(candidates)
        _enable_evolution(super_engine, super_evolution=True)

        super_engine.apply(SuperEvolve(0, super_source.entity_id))

        remaining_ids = {
            card.card_id for card in super_engine.players[0].deck
        }
        drawn = [
            card for card in candidates
            if card.card_id not in remaining_ids
        ]
        self.assertEqual(len(drawn), 2)
        self.assertEqual({card.name for card in drawn}, {"same", "other"})
        self.assertEqual(len(super_engine.players[0].hand), 9)
        self.assertTrue(
            any(
                card.definition.card_id in {item.card_id for item in drawn}
                for card in super_engine.players[0].graveyard
            )
        )
        super_engine.assert_invariants()

    def test_beast_random_keywords_are_distinct_seeded_and_auditable(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=43)
            source = _play(engine, self.repository, 10603110)
            self.assertEqual(len(source.permanent_keywords), 3)
            self.assertTrue(
                source.permanent_keywords
                <= {"疾驰", "必杀", "威慑", "吸血", "灵气", "屏障"}
            )
            outcomes.append(
                (
                    frozenset(source.permanent_keywords),
                    engine.deterministic_fingerprint(),
                )
            )
            engine.assert_invariants()
        self.assertEqual(outcomes[0], outcomes[1])

        different = self.fresh(seed=47)
        different_source = _play(different, self.repository, 10603110)
        self.assertEqual(len(different_source.permanent_keywords), 3)
        self.assertNotEqual(
            outcomes[0][0],
            frozenset(different_source.permanent_keywords),
        )

    def test_majestic_conquest_enhance_filter_countdown_capacity_and_mask(self):
        first_enhance = self.fresh(seed=53)
        _play(
            first_enhance,
            self.repository,
            10622310,
            mode_id="enhance_3",
        )
        self.assertEqual(
            [emblem.countdown for emblem in first_enhance.players[0].emblems],
            [4],
        )
        self.assertFalse(first_enhance.players[0].board)

        engine = self.fresh(seed=59)
        _play(engine, self.repository, 10622310)
        _play(
            engine,
            self.repository,
            10622310,
            mode_id="enhance_3",
        )
        self.assertEqual(
            [emblem.countdown for emblem in engine.players[0].emblems],
            [4, 4],
        )
        self.assertEqual(
            sum(
                unit.definition.card_id == 10621110
                for unit in engine.players[0].board
            ),
            1,
        )

        normal_only = self.fresh(seed=61)
        _play(normal_only, self.repository, 10622310)
        _play(normal_only, self.repository, 10622310)
        self.assertEqual(
            [emblem.countdown for emblem in normal_only.players[0].emblems],
            [2, 2],
        )
        self.assertFalse(normal_only.players[0].board)

        full = self.fresh(seed=67)
        _play(full, self.repository, 10622310)
        for index in range(5):
            _put_unit(full, 0, _card(991500 + index))
        _play(
            full,
            self.repository,
            10622310,
            mode_id="enhance_3",
        )
        self.assertEqual(len(full.players[0].board), 5)
        self.assertEqual(
            [emblem.countdown for emblem in full.players[0].emblems],
            [4, 4],
        )
        full.assert_invariants()

        deck = [
            _card(
                991600 + index,
                class_id=2,
                class_name="皇家护卫",
            )
            for index in range(40)
        ]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=2,
            class_b=2,
            seed=71,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=71)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        card = _put_hand(env.core, self.repository.get(10622310))
        env.invalidate_cache(reason="Majestic Conquest modes")
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        self.assertIn(
            PlayCard(0, 0, mode_id="normal"),
            decoded,
        )
        self.assertIn(
            PlayCard(0, 0, mode_id="enhance_3"),
            decoded,
        )
        before = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(
                PlayCard(
                    0,
                    env.players[0].hand.index(card),
                    mode_id="enhance_9",
                )
            )
        self.assertEqual(env.core.deterministic_fingerprint(), before)


class ListenerEnhanceRandomKeywordEleventhAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository("data/cards.sqlite3")

    def test_database_stats_multilingual_text_references_and_modes(self):
        expected_stats = {
            10224120: (10002, 2, 8, 5, 7),
            10424120: (10004, 2, 4, 4, 3),
            10574120: (10005, 7, 2, 2, 2),
            10603110: (10006, 0, 6, 3, 3),
            10622310: (10006, 2, 1, None, None),
        }
        expected_text = {
            10224120: ("Summon 2 enemy copies", "相手のフォロワー"),
            10424120: ("Super-evolve them instead", "超進化"),
            10574120: ("differently named 1-cost spells", "コスト1のスペル"),
            10603110: ("3 random abilities", "ドレイン"),
            10622310: ("Enhance", "エンハンス"),
        }
        expected_references = {
            10224120: [(90021110, "骑士")],
            10424120: [],
            10574120: [(90074140, "伊鞠的小鬼")],
            10603110: [],
            10622310: [(10621110, "勇烈的士兵")],
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
                    skill_rows = connection.execute(
                        """
                        SELECT text_eng, text_jpn FROM skill_texts
                        WHERE card_id=? ORDER BY position
                        """,
                        (card_id,),
                    ).fetchall()
                    mode_rows = connection.execute(
                        """
                        SELECT text_eng, text_jpn FROM alt_modes
                        WHERE card_id=? ORDER BY position
                        """,
                        (card_id,),
                    ).fetchall()
                    combined_eng = "\n".join(
                        row[0] for row in (*skill_rows, *mode_rows)
                    )
                    combined_jpn = "\n".join(
                        row[1] for row in (*skill_rows, *mode_rows)
                    )
                    self.assertIn(expected_text[card_id][0], combined_eng)
                    self.assertIn(expected_text[card_id][1], combined_jpn)
                    references = connection.execute(
                        """
                        SELECT referenced_card_id, referenced_name
                        FROM card_references
                        WHERE card_id=? AND referenced_card_id IS NOT NULL
                        ORDER BY position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(
                        [(row[0], row[1]) for row in references],
                        expected_references[card_id],
                    )
            majestic_mode = connection.execute(
                """
                SELECT mode_type, cost, text_eng, text_jpn
                FROM alt_modes WHERE card_id=10622310
                """
            ).fetchone()
            self.assertIsNotNone(majestic_mode)
            self.assertEqual(majestic_mode[1], 0)
            self.assertIn("Countdown", majestic_mode[2])
            self.assertIn("カウントダウン", majestic_mode[3])

    def test_all_five_cards_are_exact_with_direct_clause_and_token_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["rule_consistency_issues"], [])
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertEqual(
            report["summary"]["coverage_counts"],
            {
                "covered_exact": 673,
                "text_unclear": 16,
                "supported_missing_rule": 46,
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
