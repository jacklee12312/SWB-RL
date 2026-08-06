# -*- coding: utf-8 -*-
"""Exact audits for real cards using complete board and hand Tokens."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, Evolve, PlayCard
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


CARD_IDS = (
    10111120,
    10123130,
    10242120,
    10721120,
    10722110,
    10722120,
    10751120,
    10772110,
    10811120,
)
SOURCE_HASHES = {
    10111120: "d3845537b016a483104299bc63b7d266c8f9b3ce0deb109b1bb05baeabce2c4e",
    10123130: "d4efb21ad5224a38d68bd4ecfb5b84fa08d01110c54f82b3c2fc9237e9e90e50",
    10242120: "ba2cc4c24cb27ec5e6a1dfa242397e2d49f1283a88b48e4fefa7115146f6cdd9",
    10721120: "bd03e2099f1f0356033f62850d0e6677ec41505bd8d5f1a9855cd32cfa360d3f",
    10722110: "283641f2f167e9f50699e73962fb90cd47c7ee216f87494752ce291d268bb149",
    10722120: "8708368051598810b593d26a95f5be83acbd12ca8179e2ee9024980b47d0acf2",
    10751120: "b239db72c8b9eb378991e3895db04b12101c940891096841c56ed41584c4ec34",
    10772110: "8fc5e144605323e38fc724782ecd97adaad2e55a838cf192d796dbef5e71ccdd",
    10811120: "fbcae014b625ddd831494f70ea6bd193afa1f716fd6badf1fbaaecd31b15f351",
}
STRUCTURED_EVIDENCE = {
    10111120: {
        "triggers": ["fanfare", "evolve"],
        "effect_kinds": ["summon", "add_card", "summon", "add_card"],
    },
    10123130: {
        "triggers": ["evolve"],
        "effect_kinds": ["summon", "summon", "buff_unit"],
    },
    10242120: {
        "triggers": ["fanfare", "evolve"],
        "effect_kinds": ["summon", "summon", "summon"],
    },
    10721120: {
        "triggers": ["fanfare"],
        "effect_kinds": ["summon", "add_card"],
    },
    10722110: {
        "triggers": ["last_words", "evolve"],
        "effect_kinds": ["summon", "damage_unit"],
    },
    10722120: {
        "triggers": ["fanfare", "evolve"],
        "effect_kinds": ["draw", "summon", "summon", "damage_unit"],
    },
    10751120: {
        "triggers": ["last_words", "evolve"],
        "effect_kinds": ["summon", "damage_unit"],
    },
    10772110: {
        "triggers": ["fanfare", "evolve"],
        "effect_kinds": ["add_card", "add_card"],
    },
    10811120: {
        "triggers": ["fanfare", "evolve"],
        "effect_kinds": ["summon", "summon", "summon", "summon"],
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
    seed: int = 701,
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


def _put_in_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    origin: CardOrigin = CardOrigin.DECK,
) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=origin,
    )
    engine.players[0].hand.append(card)
    engine.players[0].hand_entity_ids.append(card.entity_id)
    return card


def _play_real(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
) -> Unit:
    card = _put_in_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, engine.players[0].hand.index(card)))
    return next(
        unit
        for unit in reversed(engine.players[0].board)
        if unit.definition.card_id == card_id
    )


def _add_unit(
    engine: GameEngine,
    player_index: int,
    card_id: int,
    *,
    attack: int = 1,
    life: int = 5,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, attack=attack, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[player_index].board.append(unit)
    return unit


def _enable_evolution(engine: GameEngine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _choose_entity(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _destroy(engine: GameEngine, source: Unit) -> None:
    source.health = 0
    engine._stabilize()


class DatabaseAndCoverageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_cards_match_stats_abilities_references_and_modes(self):
        expected = {
            10111120: ("\u604b\u89e6\u5996\u7cbe", "\u7cbe\u7075", 3, 1, 1),
            10123130: ("\u738b\u65ad\u7684\u5929\u5bab\u00b7\u65af\u5854\u5947\u4e4c\u59c6", "\u7687\u5bb6\u62a4\u536b", 4, 4, 4),
            10242120: ("\u8eab\u7ecf\u767e\u6218\u7684\u9c7c\u4eba", "\u9f99\u65cf", 5, 3, 3),
            10721120: ("\u4f20\u8c03\u8054\u7edc\u5175", "\u7687\u5bb6\u62a4\u536b", 3, 2, 3),
            10722110: ("\u542c\u7565\u8c0d\u62a5\u5175", "\u7687\u5bb6\u62a4\u536b", 2, 2, 1),
            10722120: ("\u65a9\u594f\u533b\u62a4\u5175", "\u7687\u5bb6\u62a4\u536b", 5, 2, 3),
            10751120: ("\u6076\u9b54\u9f13\u624b\u00b7\u62c9\u5179", "\u68a6\u9b47", 2, 1, 2),
            10772110: ("\u60a0\u7136\u7684\u6ed1\u624b", "\u8d85\u8d8a\u8005", 2, 2, 1),
            10811120: ("\u5f02\u7aef\u9690\u58eb\u00b7\u897f\u7279\u62c9\u65af", "\u7cbe\u7075", 3, 1, 1),
        }
        abilities = {
            10111120: {"\u5165\u573a\u66f2", "\u8fdb\u5316\u65f6"},
            10123130: {"\u8fdb\u5316\u65f6"},
            10242120: {"\u5165\u573a\u66f2", "\u8fdb\u5316\u65f6"},
            10721120: {"\u5165\u573a\u66f2"},
            10722110: {"\u8c22\u5e55\u66f2", "\u8fdb\u5316\u65f6"},
            10722120: {"\u5165\u573a\u66f2", "\u8fdb\u5316\u65f6"},
            10751120: {"\u8c22\u5e55\u66f2", "\u8fdb\u5316\u65f6"},
            10772110: {"\u5165\u573a\u66f2", "\u8fdb\u5316\u65f6"},
            10811120: {"\u5165\u573a\u66f2", "\u8fdb\u5316\u65f6"},
        }
        references = {
            10111120: [(0, 90011110, "\u5996\u7cbe")],
            10123130: [(0, 90021110, "\u9a91\u58eb")],
            10242120: [(0, 90041130, "\u5927\u6d77\u864e\u9cb8")],
            10721120: [(0, 90021110, "\u9a91\u58eb"), (1, 90021120, "\u94c1\u7532\u9a91\u58eb")],
            10722110: [(0, 90021110, "\u9a91\u58eb")],
            10722120: [(0, 90021110, "\u9a91\u58eb")],
            10751120: [(0, 90051110, "\u9ab8\u9aa8\u58eb\u5175")],
            10772110: [(0, 90071140, "\u53e4\u8001\u7684\u521b\u9020\u7269")],
            10811120: [(0, 90011110, "\u5996\u7cbe")],
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT json_extract(c.raw_json, '$.name_chs'),
                               cl.class_name, c.cost, c.attack, c.life,
                               cl.type_name, cs.is_collectible
                        FROM cards c
                        JOIN card_sets cs ON cs.id = c.card_set_id
                        JOIN card_localizations cl
                          ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                        WHERE c.card_id = ?
                        """,
                        (card_id,),
                    ).fetchone()
                    self.assertEqual(row[:5], values)
                    self.assertEqual(row[5:], ("\u968f\u4ece", 1))
                    self.assertEqual(
                        {
                            keyword
                            for keyword, in connection.execute(
                                "SELECT ability_keyword FROM card_abilities "
                                "WHERE card_id = ?",
                                (card_id,),
                            )
                        },
                        abilities[card_id],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT position, referenced_card_id, referenced_name "
                            "FROM card_references WHERE card_id = ? ORDER BY position",
                            (card_id,),
                        ).fetchall(),
                        references[card_id],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_all_nine_cards_are_exact_with_hash_and_structured_evidence(self):
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
                    ["tests/test_real_complete_token_board_batch.py"],
                )

    def test_all_complete_tokens_keep_auditable_new_producer_paths(self):
        report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        tokens = {card["card_id"]: card for card in report["cards"]}
        expected = {
            90011110: {(10111120, "summon"), (10111120, "add_card"), (10811120, "summon")},
            90021110: {(10123130, "summon"), (10721120, "summon"), (10722110, "summon"), (10722120, "summon")},
            90021120: {(10721120, "add_card")},
            90041130: {(10242120, "summon")},
            90051110: {(10751120, "summon")},
            90071140: {(10772110, "add_card")},
        }
        for token_id, producer_pairs in expected.items():
            with self.subTest(token_id=token_id):
                self.assertEqual(tokens[token_id]["category"], "entry_behavior_complete")
                actual = {
                    (producer["source_card_id"], producer["entry_kind"])
                    for producer in tokens[token_id]["authored_producers"]
                    if producer["rule_file"] == "real_complete_token_board_batch.json"
                }
                self.assertEqual(actual, producer_pairs)


class RealCompleteTokenBoardBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 701) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def test_loving_fairy_repeats_summon_then_add_on_evolve_and_overflow(self):
        engine = self.fresh_engine(seed=709)
        _enable_evolution(engine)
        source = _play_real(engine, self.repository, 10111120)
        engine.apply(Evolve(0, source.entity_id))

        board_fairies = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90011110
        ]
        hand_fairies = [card for card in engine.players[0].hand if card.card_id == 90011110]
        self.assertEqual((len(board_fairies), len(hand_fairies)), (2, 2))
        self.assertTrue(all(unit.origin is CardOrigin.TOKEN for unit in board_fairies))
        self.assertTrue(all(unit.has_keyword("\u7a81\u8fdb") for unit in board_fairies))

        overflow = self.fresh_engine(seed=711)
        _enable_evolution(overflow)
        source_card = _put_in_hand(overflow, self.repository.get(10111120))
        while len(overflow.players[0].hand) < overflow.config.max_hand:
            _put_in_hand(overflow, _card(7000 + len(overflow.players[0].hand)))
        overflow.apply(PlayCard(0, overflow.players[0].hand.index(source_card)))
        source = next(unit for unit in overflow.players[0].board if unit.definition.card_id == 10111120)
        overflow.apply(Evolve(0, source.entity_id))
        discarded = [
            card for card in overflow.players[0].graveyard
            if card.definition.card_id == 90011110
        ]
        self.assertEqual(len(discarded), 1)
        self.assertIs(discarded[0].origin, CardOrigin.TOKEN)

    def test_stachium_summons_then_buffs_every_other_follower(self):
        engine = self.fresh_engine(seed=719)
        _enable_evolution(engine)
        ally = _add_unit(engine, 0, 7100, attack=2, life=3)
        source = _play_real(engine, self.repository, 10123130)
        engine.apply(Evolve(0, source.entity_id))

        knights = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90021110
        ]
        self.assertEqual([(unit.attack, unit.health) for unit in knights], [(2, 2)] * 2)
        self.assertEqual((ally.attack, ally.health), (3, 4))
        self.assertEqual(
            (source.attack, source.health),
            (source.definition.attack + 2, source.definition.life + 2),
        )

    def test_seasoned_fishman_summons_two_orcas_then_one_on_evolve(self):
        engine = self.fresh_engine(seed=727)
        _enable_evolution(engine)
        source = _play_real(engine, self.repository, 10242120)
        engine.apply(Evolve(0, source.entity_id))
        orcas = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90041130
        ]
        self.assertEqual(len(orcas), 3)
        self.assertTrue(all(unit.origin is CardOrigin.TOKEN for unit in orcas))
        self.assertTrue(all(unit.has_keyword("\u7a81\u8fdb") for unit in orcas))

    def test_dispatch_liaison_summons_knight_then_adds_iron_knight(self):
        engine = self.fresh_engine(seed=733)
        source = _play_real(engine, self.repository, 10721120)
        self.assertIn(source, engine.players[0].board)
        knight = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90021110
        )
        iron = next(card for card in engine.players[0].hand if card.card_id == 90021120)
        self.assertIs(knight.origin, CardOrigin.TOKEN)
        self.assertIs(iron.origin, CardOrigin.TOKEN)

        full_board = self.fresh_engine(seed=739)
        for card_id in range(7200, 7204):
            _add_unit(full_board, 0, card_id)
        _play_real(full_board, self.repository, 10721120)
        self.assertFalse(any(
            unit.definition.card_id == 90021110 for unit in full_board.players[0].board
        ))
        self.assertTrue(any(card.card_id == 90021120 for card in full_board.players[0].hand))

    def test_targeted_evolve_damage_and_last_words_tokens_for_two_cards(self):
        for card_id, token_id in ((10722110, 90021110), (10751120, 90051110)):
            with self.subTest(card_id=card_id):
                engine = self.fresh_engine(seed=743 + card_id)
                _enable_evolution(engine)
                target = _add_unit(engine, 1, 7300 + card_id, life=5)
                source = _play_real(engine, self.repository, card_id)
                engine.apply(Evolve(0, source.entity_id))
                _choose_entity(engine, target.entity_id)
                self.assertEqual(target.health, 2)
                _destroy(engine, source)
                self.assertTrue(any(
                    unit.definition.card_id == token_id
                    for unit in engine.players[0].board
                ))

    def test_targeted_evolve_no_target_and_stale_target_paths_are_safe(self):
        no_target = self.fresh_engine(seed=751)
        _enable_evolution(no_target)
        source = _play_real(no_target, self.repository, 10722110)
        no_target.apply(Evolve(0, source.entity_id))
        self.assertTrue(source.evolved)
        self.assertIsNone(no_target.state.pending_choice)

        stale = self.fresh_engine(seed=757)
        _enable_evolution(stale)
        target = _add_unit(stale, 1, 7400, life=5)
        source = _play_real(stale, self.repository, 10722110)
        stale.apply(Evolve(0, source.entity_id))
        choice = next(
            command for command in stale.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        stale.players[1].board.remove(target)
        stale._send_to_graveyard(
            1,
            target.definition,
            "test_target_left_play",
            source_entity_id=target.entity_id,
        )
        stale.apply(choice)
        self.assertIsNone(stale.state.pending_choice)
        self.assertEqual(target.health, 5)

    def test_medical_soldier_draws_then_summons_and_evolve_damages(self):
        engine = self.fresh_engine(seed=761)
        _enable_evolution(engine)
        deck_before = len(engine.players[0].deck)
        target = _add_unit(engine, 1, 7500, life=3)
        source = _play_real(engine, self.repository, 10722120)
        self.assertEqual(len(engine.players[0].deck), deck_before - 2)
        self.assertEqual(
            sum(unit.definition.card_id == 90021110 for unit in engine.players[0].board),
            2,
        )
        engine.apply(Evolve(0, source.entity_id))
        _choose_entity(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)

    def test_skater_adds_ancient_artifact_on_fanfare_and_evolve(self):
        engine = self.fresh_engine(seed=769)
        _enable_evolution(engine)
        source = _play_real(engine, self.repository, 10772110)
        engine.apply(Evolve(0, source.entity_id))
        artifacts = [card for card in engine.players[0].hand if card.card_id == 90071140]
        self.assertEqual(len(artifacts), 2)
        self.assertTrue(all(card.origin is CardOrigin.TOKEN for card in artifacts))

    def test_citras_fanfare_and_evolve_fill_board_with_four_fairies(self):
        engine = self.fresh_engine(seed=773)
        _enable_evolution(engine)
        source = _play_real(engine, self.repository, 10811120)
        engine.apply(Evolve(0, source.entity_id))
        fairies = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90011110
        ]
        self.assertEqual(len(fairies), 4)
        self.assertEqual(len(engine.players[0].board), engine.config.max_board)
        self.assertTrue(all(unit.has_keyword("\u7a81\u8fdb") for unit in fairies))

    def test_locked_evolution_is_illegal_without_any_state_mutation(self):
        engine = self.fresh_engine(seed=787)
        source = _play_real(engine, self.repository, 10722110)
        command = Evolve(0, source.entity_id)
        before = engine.deterministic_fingerprint()
        self.assertNotIn(command, engine.legal_commands())
        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_seeded_repeated_fanfare_and_evolve_is_deterministic(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh_engine(seed=797)
            _enable_evolution(engine)
            source = _play_real(engine, self.repository, 10111120)
            engine.apply(Evolve(0, source.entity_id))
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_targeted_evolve_choice_is_exposed_by_rl_mask(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(3000, 3040)],
            [_card(card_id) for card_id in range(4000, 4040)],
            class_a=2,
            class_b=2,
            seed=809,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=809)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].max_mana = env.players[0].mana = 10
        env.players[0].turns_started = env.core.config.evolution_unlock_turn
        _put_in_hand(env.core, self.repository.get(10722110))
        target = _add_unit(env.core, 1, 7600, life=5)

        play = PlayCard(0, 0)
        play_action = env._encode_command(play)
        self.assertTrue(env.action_mask()[play_action])
        env.step(play_action)
        source = next(
            unit for unit in env.players[0].board
            if unit.definition.card_id == 10722110
        )
        evolve = Evolve(0, source.entity_id)
        evolve_action = env._encode_command(evolve)
        self.assertIsNotNone(evolve_action)
        self.assertTrue(env.action_mask()[evolve_action])
        env.step(evolve_action)

        request = env.core.state.pending_choice
        choice = Choose(
            request.player_index,
            next(option.option_id for option in request.options if option.entity_id == target.entity_id),
        )
        choice_action = env._encode_command(choice)
        self.assertIsNotNone(choice_action)
        self.assertTrue(env.action_mask()[choice_action])


if __name__ == "__main__":
    unittest.main()
