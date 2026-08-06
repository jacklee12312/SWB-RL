# -*- coding: utf-8 -*-
"""Exact Royal/Bishop cards built from established generic primitives."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import ActivateAmulet, Choose, EndTurn, Evolve, PlayCard
from swb.engine.effects import EffectKind, ExprType, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import Amulet, HandCard, Unit


BATCH_CARD_IDS = (
    10723310,
    10263310,
    10062210,
    10362220,
    10122140,
    10421120,
    10421130,
    10821130,
    10761110,
    10662120,
    10562110,
    10762120,
)
SOURCE_HASHES = {
    10723310: "deed034a06178f525bb8925c47a18ee1f583d67cc73706fa3f63c38f9b7e9c0d",
    10263310: "4e1f852e83df10630b0bdbb985598d5a0145194565fa54b8880bdd6a0e8ce245",
    10062210: "0ca07db3d147ca0cf2f63c310a6ce8c47b54a13c2bf36785cb7608c0a943010f",
    10362220: "5eb782bad60b4bb7261cec5b8c159426716fb16b6f3ea16b87a2e87c46c27675",
    10122140: "85cb770aa28cdde9e30cb30d34f3b64b61a9e6bfb2e35ac794359dc5174c486d",
    10421120: "4c53753da93e9a26f28e0370c2ed6263468449a09c3f0bdf3efd58a73cc7d361",
    10421130: "a63c74cf3b277405ce4b358510a7036e1fdc2c14cf7b3a4e2c81946b38300377",
    10821130: "247563fdceea0f05842e463fef6b72b96a3594ec153c8a67211af701e53c3227",
    10761110: "f75f4b4acf1145f31b8e5fa19f63bbcda99ad275bc0a678e535a1a9be5f356ea",
    10662120: "578e278cccfc18d975226c0a76e508e561acc05a398f82eb995bdb98351305d9",
    10562110: "a27210fd8a0778c766cff44abba19d1d03fd463ba52af7947fc03cef53c610c8",
    10762120: "4bb3863417e73e0773f757ef31ff8baa8dce22918e21909a3dfad948a2bb868b",
}


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 1),
        class_name=overrides.get("class_name", "精灵"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 5),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _fresh(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 3101,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=1,
        class_b=1,
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
        player.max_mana = player.mana = 10
    return engine


def _put_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


def _put_unit(
    engine: GameEngine,
    owner: int,
    definition: CardDefinition,
) -> Unit:
    unit = Unit.summon(
        definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[owner].board.append(unit)
    return unit


def _put_amulet(engine: GameEngine, owner: int, card_id: int) -> Amulet:
    amulet = Amulet(
        definition=_card(
            card_id,
            card_type="护符",
            attack=None,
            life=None,
        ),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[owner].board.append(amulet)
    return amulet


def _play(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
) -> Unit | Amulet | None:
    source = _put_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
    return next(
        (
            entity
            for entity in engine.players[0].board
            if entity.definition.card_id == card_id
        ),
        None,
    )


def _choose(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolution(engine: GameEngine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


class RoyalBishopExistingPrimitiveBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 3101) -> GameEngine:
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_use_only_existing_generic_primitives(self):
        base, rally = self.rulebook.operations_for(10723310, Trigger.PLAY)
        self.assertIs(base.kind, EffectKind.DAMAGE_UNIT)
        self.assertIs(base.target, TargetKind.ENEMY_UNIT)
        self.assertTrue(base.requires_target)
        self.assertIs(rally.target, TargetKind.ALL_ENEMY_UNITS)
        self.assertIs(
            rally.amount_expr.type,
            ExprType.CONTROLLER_BOARD_COUNT,
        )
        self.assertEqual(
            rally.amount_expr.board_filter.card_type,
            "随从",
        )

        self.assertEqual(self.rulebook.activation_for(10062210).cost, 1)
        self.assertEqual(self.rulebook.activation_for(10362220).cost, 1)
        self.assertEqual(self.rulebook.countdown_for(10062210), 4)
        self.assertEqual(self.rulebook.countdown_for(10362220), 3)
        self.assertEqual(
            self.rulebook.emblem_def("maddening_benison").countdown,
            2,
        )

    def test_silent_blade_requires_target_below_rally_and_is_atomic(self):
        engine = self.fresh(seed=3)
        spell = _put_hand(engine, self.repository.get(10723310))
        command = PlayCard(0, engine.players[0].hand.index(spell))
        before = engine.deterministic_fingerprint()

        self.assertNotIn(command, engine.legal_commands())
        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.deterministic_fingerprint(), before)

        target = _put_unit(engine, 1, _card(9101, life=8))
        engine.apply(command)
        request = engine.state.pending_choice
        choice = next(
            Choose(request.player_index, option.option_id)
            for option in request.options
            if option.entity_id == target.entity_id
        )
        engine.players[1].board.remove(target)
        engine.apply(choice)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[1].health, 20)

    def test_silent_blade_rally_replaces_selection_with_filtered_area_damage(self):
        engine = self.fresh(seed=5)
        engine.players[0].cooperation = 10
        _put_unit(engine, 0, _card(9110))
        _put_unit(engine, 0, _card(9111))
        _put_amulet(engine, 0, 9112)
        enemies = [
            _put_unit(engine, 1, _card(9120 + index, life=6))
            for index in range(2)
        ]

        _play(engine, self.repository, 10723310)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual([enemy.health for enemy in enemies], [4, 4])

        empty = self.fresh(seed=7)
        empty.players[0].cooperation = 10
        spell = _put_hand(empty, self.repository.get(10723310))
        command = PlayCard(0, empty.players[0].hand.index(spell))
        self.assertIn(command, empty.legal_commands())
        empty.apply(command)

    def test_maddening_benison_heals_then_expires_for_self_damage_once(self):
        engine = self.fresh(seed=11)
        engine.players[0].health = 4
        _play(engine, self.repository, 10263310)

        self.assertEqual(engine.players[0].health, 14)
        self.assertEqual(
            [emblem.countdown for emblem in engine.players[0].emblems],
            [2],
        )
        for _ in range(2):
            engine.apply(EndTurn(engine.current_player))
            engine.apply(EndTurn(engine.current_player))
        self.assertFalse(engine.players[0].emblems)
        self.assertEqual(engine.players[0].health, 4)
        for _ in range(2):
            engine.apply(EndTurn(engine.current_player))
        self.assertEqual(engine.players[0].health, 4)

    def test_countdown_amulets_activate_and_resolve_tokens_in_printed_order(self):
        for card_id, countdown, token_ids in (
            (10062210, 4, [90061110]),
            (10362220, 3, [90061110, 90061120]),
        ):
            with self.subTest(card_id=card_id):
                engine = self.fresh(seed=13)
                amulet = _play(engine, self.repository, card_id)
                self.assertIsInstance(amulet, Amulet)
                self.assertEqual(amulet.countdown, countdown)
                command = ActivateAmulet(0, amulet.entity_id)
                self.assertIn(command, engine.legal_commands())
                engine.apply(command)
                self.assertEqual(amulet.countdown, countdown - 1)
                while amulet in engine.players[0].board:
                    engine.apply(EndTurn(engine.current_player))
                    engine.apply(EndTurn(engine.current_player))
                self.assertEqual(
                    [unit.definition.card_id for unit in engine.players[0].board],
                    token_ids,
                )

        shortage = self.fresh(seed=17)
        for index in range(4):
            _put_unit(shortage, 0, _card(9130 + index))
        amulet = _play(shortage, self.repository, 10362220)
        amulet.countdown = 1
        shortage.apply(EndTurn(0))
        shortage.apply(EndTurn(1))
        self.assertEqual(len(shortage.players[0].board), shortage.config.max_board)
        self.assertEqual(
            [
                unit.definition.card_id
                for unit in shortage.players[0].board
                if isinstance(unit, Unit) and unit.definition.card_id >= 90000000
            ],
            [90061110],
        )

    def test_evolve_producer_chain_preserves_intrinsic_keywords_and_capacity(self):
        expected = {
            10122140: (10122140, "潜行"),
            10421120: (10421130, "疾驰"),
            10421130: (10421120, "守护"),
        }
        for card_id, (produced_id, keyword) in expected.items():
            with self.subTest(card_id=card_id):
                engine = self.fresh(seed=19 + card_id)
                source = _play(engine, self.repository, card_id)
                _enable_evolution(engine)
                engine.apply(Evolve(0, source.entity_id))
                produced = next(unit for unit in engine.players[0].board if unit is not source)
                self.assertEqual(produced.definition.card_id, produced_id)
                self.assertTrue(produced.has_keyword(keyword))

                full = self.fresh(seed=23 + card_id)
                source = _play(full, self.repository, card_id)
                for index in range(4):
                    _put_unit(full, 0, _card(9140 + index))
                _enable_evolution(full)
                full.apply(Evolve(0, source.entity_id))
                self.assertEqual(len(full.players[0].board), full.config.max_board)

    def test_sasha_grants_only_successfully_summoned_knights(self):
        engine = self.fresh(seed=29)
        source = _play(engine, self.repository, 10821130)
        fanfare_knight = next(unit for unit in engine.players[0].board if unit is not source)
        self.assertEqual(fanfare_knight.definition.card_id, 90021120)
        self.assertTrue(fanfare_knight.has_keyword("突进"))
        self.assertFalse(fanfare_knight.has_keyword("守护"))

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        evolve_knight = engine.players[0].board[-1]
        self.assertTrue(evolve_knight.has_keyword("守护"))
        self.assertFalse(evolve_knight.has_keyword("突进"))

        full = self.fresh(seed=31)
        for index in range(4):
            _put_unit(full, 0, _card(9150 + index))
        source = _play(full, self.repository, 10821130)
        self.assertEqual(len(full.players[0].board), full.config.max_board)
        self.assertEqual(
            [unit.definition.card_id for unit in full.players[0].board].count(90021120),
            0,
        )
        _enable_evolution(full)
        full.apply(Evolve(0, source.entity_id))
        self.assertEqual(len(full.players[0].board), full.config.max_board)

    def test_finance_reverend_checks_three_amulets_and_each_copy_has_last_words(self):
        below = self.fresh(seed=37)
        for index in range(2):
            _put_amulet(below, 0, 9160 + index)
        source = _play(below, self.repository, 10761110)
        self.assertEqual(
            [unit.definition.card_id for unit in below.players[0].board].count(10761110),
            1,
        )
        self.assertTrue(source.has_keyword("突进"))

        enough = self.fresh(seed=41)
        for index in range(3):
            _put_amulet(enough, 0, 9170 + index)
        source = _play(enough, self.repository, 10761110)
        copies = [
            unit for unit in enough.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == 10761110
        ]
        self.assertEqual(len(copies), 2)
        deck_before = len(enough.players[0].deck)
        for unit in copies:
            unit.health = 0
        enough._stabilize()
        self.assertEqual(len(enough.players[0].deck), deck_before - 2)

    def test_falcon_rider_and_paladin_respect_board_capacity(self):
        rider = self.fresh(seed=43)
        source = _play(rider, self.repository, 10662120)
        _enable_evolution(rider)
        rider.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [unit.definition.card_id for unit in rider.players[0].board],
            [10662120, 90061110, 90061110],
        )

        paladin = self.fresh(seed=47)
        source = _play(paladin, self.repository, 10562110)
        copies = [
            unit for unit in paladin.players[0].board
            if unit.definition.card_id == 10562110
        ]
        self.assertEqual(len(copies), 4)
        self.assertTrue(all(unit.has_keyword("守护") for unit in copies))

        shortage = self.fresh(seed=53)
        for index in range(3):
            _put_unit(shortage, 0, _card(9180 + index))
        _play(shortage, self.repository, 10562110)
        self.assertEqual(len(shortage.players[0].board), shortage.config.max_board)
        self.assertEqual(
            [unit.definition.card_id for unit in shortage.players[0].board].count(10562110),
            2,
        )

    def test_rumored_holy_bird_banishes_then_heals_and_continues_after_stale_target(self):
        engine = self.fresh(seed=59)
        engine.players[0].health = 12
        for index in range(3):
            _put_amulet(engine, 0, 9190 + index)
        target = _put_unit(engine, 1, _card(9195))
        source = _play(engine, self.repository, 10762120)
        self.assertTrue(source.has_keyword("疾驰"))
        _choose(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[0].health, 15)

        stale = self.fresh(seed=61)
        stale.players[0].health = 12
        for index in range(3):
            _put_amulet(stale, 0, 9200 + index)
        target = _put_unit(stale, 1, _card(9205))
        _play(stale, self.repository, 10762120)
        request = stale.state.pending_choice
        choice = next(
            Choose(request.player_index, option.option_id)
            for option in request.options
            if option.entity_id == target.entity_id
        )
        stale.players[1].board.remove(target)
        stale.apply(choice)
        self.assertIsNone(stale.state.pending_choice)
        self.assertEqual(stale.players[0].health, 15)

        empty = self.fresh(seed=67)
        empty.players[0].health = 12
        for index in range(3):
            _put_amulet(empty, 0, 9210 + index)
        _play(empty, self.repository, 10762120)
        self.assertIsNone(empty.state.pending_choice)
        self.assertEqual(empty.players[0].health, 15)

    def test_seeded_batch_sequence_has_identical_fingerprint(self):
        def run(seed: int):
            engine = self.fresh(seed=seed)
            _play(engine, self.repository, 10562110)
            engine.apply(EndTurn(engine.current_player))
            return engine.deterministic_fingerprint(), tuple(engine.event_history)

        self.assertEqual(run(71), run(71))

    def test_rl_mask_matches_rally_target_requirement(self):
        deck = [_card(9300 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=73,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=73)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        spell = _put_hand(env.core, self.repository.get(10723310))
        command = PlayCard(0, env.players[0].hand.index(spell))
        action = env._encode_command(command)
        self.assertFalse(env.action_mask()[action])

        env.players[0].cooperation = 10
        self.assertTrue(env.action_mask()[action])
        env.step(action)
        self.assertIsNone(env.core.state.pending_choice)


class RoyalBishopExistingPrimitiveDatabaseAuditTests(unittest.TestCase):
    def test_imported_text_modes_and_token_references_match_reviewed_cards(self):
        expected_phrases = {
            10723310: ("Rally", "all enemy followers"),
            10263310: ("Restore 10 defense", "Deal 10 damage to your leader"),
            10062210: ("Countdown", "Holy Falcon", "Engage"),
            10362220: ("Holy Falcon", "Holyflame Tiger", "Engage"),
            10122140: ("Ambush", "Shinobi Squirrel"),
            10421120: ("Ward", "Mordred, Illusory Lion"),
            10421130: ("Storm", "Arthur, Staunch Dragon"),
            10821130: ("Steelclad Knight", "Rush", "Ward"),
            10761110: ("at least 3 allied amulets", "Last Words"),
            10662120: ("Holy Falcon", "Evolve"),
            10562110: ("Summon 3", "Ward"),
            10762120: ("banish it", "at least 3 allied amulets", "Storm"),
        }
        expected_references = {
            10723310: [],
            10263310: [],
            10062210: [90061110],
            10362220: [90061110, 90061120],
            10122140: [10122140],
            10421120: [10421130],
            10421130: [10421120],
            10821130: [90021120],
            10761110: [10761110],
            10662120: [90061110],
            10562110: [10562110],
            10762120: [],
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id in BATCH_CARD_IDS:
                with self.subTest(card_id=card_id):
                    texts = [
                        row[0]
                        for row in connection.execute(
                            "SELECT text_eng FROM skill_texts WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    ]
                    texts.extend(
                        row[0]
                        for row in connection.execute(
                            "SELECT text_eng FROM alt_modes WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    )
                    normalized = " ".join(re.sub(r"<[^>]+>", "", text) for text in texts)
                    for phrase in expected_phrases[card_id]:
                        self.assertIn(phrase, normalized)
                    self.assertEqual(
                        [
                            row[0]
                            for row in connection.execute(
                                "SELECT referenced_card_id FROM card_references "
                                "WHERE card_id=? ORDER BY position",
                                (card_id,),
                            )
                        ],
                        expected_references[card_id],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        1 if card_id == 10263310 else 0,
                    )

    def test_batch_cards_have_exact_clause_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in BATCH_CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_royal_bishop_existing_primitives_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
