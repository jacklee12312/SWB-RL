# -*- coding: utf-8 -*-
"""Exact spell, amulet, crest, mode, and referenced-follower batch."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import ActivateAmulet, Choose, EndTurn, PlayCard
from swb.engine.effects import EffectKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, CostModifier, HandCard, StatModifier, Unit
from tests.play_mode_test_support import prepare_mana_for_play_mode


BATCH_CARD_IDS = (
    10412310,
    10441310,
    10451310,
    10712310,
    10713310,
    10233310,
    10352210,
    10633310,
    10413310,
    10332110,
    10631110,
)

# Filled from the canonical primary + alternate-mode source text map.
SOURCE_HASHES = {
    10412310: "15dc339f1a185e71435ff6eb08c7622011f15e7886e95f69fd980ce82a742cb2",
    10441310: "580b13dfc91a95c45056260215c78a87d7bc346a8123b04c6d1426d7a7bda27f",
    10451310: "d9f716a00206cdd3e95d94b09c85e7c00d2421dd5f67a356bc4cff4c67eccc83",
    10712310: "8b645fdbc1fc7a2ca6ec377d35b9d1bb9996992f495c75cd0a3e286964e2bdd4",
    10713310: "fb507edccf4b2d6ed5ed487c22b247efe67fba3e9fe2a5b416d6914d459dadec",
    10233310: "af2da2e610e1cda1dbbcccd840a41b96868296d7cd67ca5a3635a37bd5ef98c8",
    10352210: "4e10101732a5a65bf6bd363018c3c26d25ab15c294413d4fb22fe62772836f96",
    10633310: "63d2a9136076b0a12cf1c02ee4b889fabf6b37f6a72218f5193dc589e54d491c",
    10413310: "db6839a77c3b0c26e722440d421bfe9f3378532324c041592f9621b3abba1158",
    10332110: "807fa4f4113430557d247bee29ef4faa6dbc0a0cdbe481edce9b5adeb4261232",
    10631110: "7abde1fa8f2c436a6805d99abe026718965b9776b4b5a2dffe05f57cd5b2776f",
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
        life=overrides.get("life", 6),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _fresh(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 6201,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(21000, 21040)],
        [_card(card_id) for card_id in range(22000, 22040)],
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


def _put_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    owner: int = 0,
) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[owner].hand.append(card)
    engine.players[owner].hand_entity_ids.append(card.entity_id)
    return card


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


def _put_earth_sigils(engine: GameEngine, count: int) -> Amulet:
    sigil = Amulet(
        definition=_card(
            90031210,
            card_set_id=90000,
            name="大地之魔片",
            card_type="护符",
            attack=None,
            life=None,
            keywords=frozenset({"土之印"}),
            is_collectible=False,
        ),
        entity_id=engine.state.allocate_entity_id(),
        earth_sigil_count=count,
        entered_turn=engine.turn,
        origin=CardOrigin.TOKEN,
    )
    engine.players[0].board.append(sigil)
    return sigil


def _play(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
    *,
    mode_id: str = "normal",
) -> Unit | Amulet | None:
    source = _put_hand(engine, repository.get(card_id))
    prepare_mana_for_play_mode(engine, source, mode_id)
    engine.apply(
        PlayCard(
            0,
            engine.players[0].hand.index(source),
            mode_id=mode_id,
        )
    )
    return next(
        (
            entity
            for entity in engine.players[0].board
            if entity.definition.card_id == card_id
        ),
        None,
    )


def _choose_mode(engine: GameEngine, option_id: str) -> None:
    request = engine.state.pending_choice
    engine.apply(Choose(request.player_index, f"choose_one:{option_id}"))


def _emblem(engine: GameEngine, emblem_id: str):
    return next(
        emblem
        for emblem in engine.players[0].emblems
        if emblem.emblem_id == emblem_id
    )


class RealSpellAmuletCrestBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 6201) -> GameEngine:
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_crest_activate_enhance_burst_and_rush(self):
        self.assertEqual(self.rulebook.activation_for(10352210).cost, 1)
        self.assertEqual(
            [mode.mode_id for mode in self.rulebook.modes_for(10633310)],
            ["enhance_5"],
        )
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10631110),
            ("突进",),
        )
        self.assertEqual(
            self.rulebook.emblem_def("pascales_dance").countdown,
            1,
        )
        burst = self.rulebook.union_bursts_for(10413310)
        self.assertEqual(len(burst), 1)
        self.assertTrue(burst[0].replace_base_operations)
        supplicant = self.rulebook.operations_for(10332110, Trigger.FANFARE)
        self.assertEqual(
            [operation.kind for operation in supplicant],
            [EffectKind.DAMAGE_UNIT, EffectKind.DRAW],
        )

    def test_starry_sky_combo_and_expiration_cover_empty_target_and_hand_cap(self):
        below = self.fresh(seed=3)
        below.players[0].cards_played_this_turn = 3
        _play(below, self.repository, 10412310)
        self.assertFalse(below.players[0].emblems)

        active = self.fresh(seed=5)
        active.players[0].cards_played_this_turn = 4
        active.players[1].health = 20
        _play(active, self.repository, 10412310)
        self.assertEqual(_emblem(active, "starry_sky").countdown, 1)
        active.apply(EndTurn(0))
        active.apply(EndTurn(1))
        self.assertEqual(active.players[1].health, 19)
        self.assertEqual(
            sum(card.card_id == 10412310 for card in active.players[0].hand),
            1,
        )
        self.assertFalse(active.players[0].emblems)

        full = self.fresh(seed=7)
        full.players[0].cards_played_this_turn = 4
        _play(full, self.repository, 10412310)
        for index in range(full.config.max_hand):
            _put_hand(full, _card(23000 + index))
        full.apply(EndTurn(0))
        full.apply(EndTurn(1))
        self.assertEqual(len(full.players[0].hand), full.config.max_hand)
        self.assertTrue(any(
            card.definition.card_id == 10412310
            and card.entry_cause == "hand_full"
            for card in full.players[0].graveyard
        ))

    def test_crescent_and_valiant_crests_use_owner_turn_and_safe_random_targets(self):
        crescent = self.fresh(seed=11)
        ally = _put_unit(crescent, 0, _card(23100, attack=2, life=3))
        _play(crescent, self.repository, 10441310)
        crescent.apply(EndTurn(0))
        self.assertEqual((ally.attack, ally.health, ally.max_health), (3, 4, 4))
        crescent.apply(EndTurn(1))
        self.assertEqual((ally.attack, ally.health), (3, 4))

        valiant = self.fresh(seed=13)
        valiant.players[0].health = 10
        enemy = _put_unit(valiant, 1, _card(23101, life=5))
        _play(valiant, self.repository, 10451310)
        self.assertEqual(valiant.players[0].health, 8)
        valiant.apply(EndTurn(0))
        self.assertEqual((enemy.health, valiant.players[0].health), (3, 9))

        empty = self.fresh(seed=17)
        empty.players[0].health = 10
        _play(empty, self.repository, 10451310)
        empty.apply(EndTurn(0))
        self.assertEqual(empty.players[0].health, 9)

    def test_anxiety_malice_pair_heal_damage_combo_and_generate_each_other(self):
        anxiety = self.fresh(seed=19)
        anxiety.players[0].health = 18
        anxiety.players[0].cards_played_this_turn = 2
        _play(anxiety, self.repository, 10712310)
        self.assertEqual(anxiety.players[0].health, 19)
        anxiety.apply(EndTurn(0))
        anxiety.apply(EndTurn(1))
        self.assertTrue(any(card.card_id == 10713310 for card in anxiety.players[0].hand))

        malice = self.fresh(seed=23)
        target = _put_unit(malice, 1, _card(23200, life=4))
        malice.players[0].cards_played_this_turn = 2
        _play(malice, self.repository, 10713310)
        self.assertEqual(target.health, 2)
        malice.apply(EndTurn(0))
        malice.apply(EndTurn(1))
        self.assertTrue(any(card.card_id == 10712310 for card in malice.players[0].hand))

        below = self.fresh(seed=29)
        below.players[0].cards_played_this_turn = 1
        _play(below, self.repository, 10713310)
        self.assertFalse(below.players[0].emblems)

    def test_pascales_dance_draws_then_pays_ten_and_doubles_current_stats(self):
        engine = self.fresh(seed=31)
        unit = _put_unit(engine, 0, _card(23300, attack=3, life=5))
        unit.add_stat_modifier(
            StatModifier(
                modifier_id=23301,
                attack_delta=1,
                health_delta=2,
                duration="permanent",
            )
        )
        unit.health = 6
        _put_earth_sigils(engine, 9)
        deck_before = len(engine.players[0].deck)
        _play(engine, self.repository, 10233310)
        self.assertEqual(engine.players[0].earth_sigils, 10)
        engine.apply(EndTurn(0))
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)
        self.assertEqual(engine.players[0].earth_sigils, 0)
        self.assertEqual((unit.attack, unit.health, unit.max_health), (8, 12, 12))
        self.assertEqual(unit.stat_modifiers, [])

        below = self.fresh(seed=37)
        unit = _put_unit(below, 0, _card(23301, attack=2, life=3))
        _put_earth_sigils(below, 8)
        _play(below, self.repository, 10233310)
        below.apply(EndTurn(0))
        self.assertEqual((unit.attack, unit.health), (2, 3))
        self.assertEqual(below.players[0].earth_sigils, 9)

    def test_pascales_dance_full_board_skips_new_sigil_but_keeps_crest(self):
        engine = self.fresh(seed=41)
        for index in range(engine.config.max_board):
            _put_amulet(engine, 0, 23400 + index)
        _play(engine, self.repository, 10233310)
        self.assertEqual(engine.players[0].earth_sigils, 0)
        self.assertEqual(_emblem(engine, "pascales_dance").countdown, 1)

    def test_castle_fanfare_and_engage_replicate_each_mode_after_destroy(self):
        draw = self.fresh(seed=43)
        amulet = _play(draw, self.repository, 10352210)
        deck_before = len(draw.players[0].deck)
        _choose_mode(draw, "draw_one")
        self.assertEqual(len(draw.players[0].deck), deck_before - 1)
        mana_before = draw.players[0].mana
        draw.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertNotIn(amulet, draw.players[0].board)
        self.assertEqual(draw.players[0].mana, mana_before - 1)
        _choose_mode(draw, "draw_one")
        self.assertEqual(len(draw.players[0].deck), deck_before - 2)

        buff = self.fresh(seed=47)
        ally = _put_unit(buff, 0, _card(23500, attack=2, life=3))
        _play(buff, self.repository, 10352210)
        _choose_mode(buff, "buff_random")
        self.assertEqual((ally.attack, ally.health), (3, 4))

        empty = self.fresh(seed=53)
        _play(empty, self.repository, 10352210)
        _choose_mode(empty, "buff_random")
        self.assertIsNone(empty.state.pending_choice)

    def test_bewitching_crystals_modes_enhance_capacity_and_output_binding(self):
        storm = self.fresh(seed=59)
        _play(storm, self.repository, 10633310)
        _choose_mode(storm, "storm_spawn")
        spawned = storm.players[0].board[0]
        self.assertEqual(
            (spawned.definition.card_id, spawned.attack, spawned.health),
            (10631110, 2, 1),
        )
        self.assertTrue(spawned.has_keyword("突进"))
        self.assertTrue(spawned.has_keyword("疾驰"))

        pair = self.fresh(seed=61)
        _play(pair, self.repository, 10633310)
        _choose_mode(pair, "two_spawns")
        self.assertEqual(
            [
                (unit.definition.card_id, unit.attack, unit.health)
                for unit in pair.players[0].board
            ],
            [(10631110, 2, 1), (10631110, 2, 1)],
        )

        enhance = self.fresh(seed=67)
        _put_unit(enhance, 0, _card(23600))
        _put_unit(enhance, 0, _card(23601))
        _put_unit(enhance, 0, _card(23602))
        _play(enhance, self.repository, 10633310, mode_id="enhance_5")
        spawned = [
            unit
            for unit in enhance.players[0].board
            if unit.definition.card_id == 10631110
        ]
        self.assertEqual(len(spawned), 2)
        self.assertTrue(spawned[0].has_keyword("疾驰"))
        self.assertFalse(spawned[1].has_keyword("疾驰"))
        self.assertTrue(all(unit.attack == 2 for unit in spawned))

    def test_alfheimr_modes_and_super_skybound_art_activate_exact_branches(self):
        attack = self.fresh(seed=71)
        ally = _put_unit(attack, 0, _card(23700, attack=2, life=3))
        _play(attack, self.repository, 10413310)
        _choose_mode(attack, "attack_rush")
        self.assertEqual((ally.attack, ally.health), (3, 3))
        self.assertTrue(ally.has_keyword("突进"))
        self.assertFalse(ally.has_keyword("守护"))

        health = self.fresh(seed=73)
        ally = _put_unit(health, 0, _card(23701, attack=2, life=3))
        _play(health, self.repository, 10413310)
        _choose_mode(health, "health_ward")
        self.assertEqual((ally.attack, ally.health, ally.max_health), (2, 4, 4))
        self.assertTrue(ally.has_keyword("守护"))

        super_art = self.fresh(seed=79)
        super_art.players[0].health = 18
        super_art.players[0].turns_started = 15
        ally = _put_unit(super_art, 0, _card(23702, attack=2, life=3))
        deck_before = len(super_art.players[0].deck)
        _play(super_art, self.repository, 10413310)
        self.assertIsNone(super_art.state.pending_choice)
        self.assertEqual(len(super_art.players[0].deck), deck_before - 1)
        self.assertEqual(super_art.players[0].health, 19)
        self.assertEqual((ally.attack, ally.health, ally.max_health), (3, 4, 4))
        self.assertTrue(ally.has_keyword("突进"))
        self.assertTrue(ally.has_keyword("守护"))

    def test_supplicant_uses_play_cost_and_damages_other_followers_simultaneously(self):
        printed = self.fresh(seed=83)
        ally = _put_unit(printed, 0, _card(23800, life=3))
        enemy = _put_unit(printed, 1, _card(23801, life=3))
        deck_before = len(printed.players[0].deck)
        source = _play(printed, self.repository, 10332110)
        self.assertIn(source, printed.players[0].board)
        self.assertNotIn(ally, printed.players[0].board)
        self.assertNotIn(enemy, printed.players[1].board)
        self.assertEqual(len(printed.players[0].deck), deck_before)

        discounted = self.fresh(seed=89)
        source = _put_hand(discounted, self.repository.get(10332110))
        source.cost_modifiers.append(CostModifier(1, "set", 4, "permanent"))
        deck_before = len(discounted.players[0].deck)
        discounted.apply(PlayCard(0, 0))
        self.assertEqual(len(discounted.players[0].deck), deck_before - 2)

    def test_seeded_random_and_rl_action_mask_are_reproducible_and_executable(self):
        def resolved(seed: int):
            engine = self.fresh(seed=seed)
            _put_unit(engine, 1, _card(23900, life=5))
            _put_unit(engine, 1, _card(23901, life=5))
            engine.players[0].cards_played_this_turn = 2
            _play(engine, self.repository, 10713310)
            return engine.deterministic_fingerprint(), tuple(engine.event_history)

        self.assertEqual(resolved(97), resolved(97))

        deck = [_card(24000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=101,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=101)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        _put_hand(env.core, self.repository.get(10633310))
        normal = PlayCard(0, 0)
        enhance = PlayCard(0, 0, mode_id="enhance_5")
        self.assertFalse(env.action_mask()[env._encode_command(normal)])
        self.assertTrue(env.action_mask()[env._encode_command(enhance)])

        env.players[0].mana = 4
        env.invalidate_cache(reason="bewitching crystals normal threshold")
        self.assertTrue(env.action_mask()[env._encode_command(normal)])
        self.assertIsNone(env._encode_command(enhance))
        env.step(env._encode_command(normal))
        choices = [command for command in env.core.legal_commands() if isinstance(command, Choose)]
        self.assertEqual(len(choices), 2)
        self.assertTrue(all(env.action_mask()[env._encode_command(command)] for command in choices))


class SpellAmuletCrestDatabaseAuditTests(unittest.TestCase):
    def test_database_text_modes_and_references_match_reviewed_cards(self):
        expected_phrases = {
            10412310: ("random enemy follower", "Combo", "Starry Sky", "Last Words"),
            10441310: ("Crescent Tube Ride", "Countdown", "random allied follower"),
            10451310: ("your leader", "Valiant Edge", "random enemy follower"),
            10712310: ("Restore 1 defense", "Minimized Anxiety", "Magnified Malice"),
            10713310: ("random enemy follower", "Magnified Malice", "Minimized Anxiety"),
            10233310: ("earth sigil", "Pascale's Dance", "Earth Rite", "Double"),
            10352210: ("Fanfare", "Mode", "Engage", "Destroy this card"),
            10633310: ("Mode", "Crystalspawn", "Storm", "Enhance"),
            10413310: ("Super Skybound Art", "Rush", "Ward"),
            10332110: ("all other followers", "cost isn't 5", "draw 2 cards"),
            10631110: ("Rush",),
        }
        expected_references = {
            10412310: [10412310],
            10441310: [],
            10451310: [],
            10712310: [10713310],
            10713310: [10712310],
            10233310: [],
            10352210: [],
            10633310: [10631110],
            10413310: [],
            10332110: [],
            10631110: [],
        }
        crest_cards = {10412310, 10441310, 10451310, 10712310, 10713310, 10233310}
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
                        1 if card_id in crest_cards else 0,
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
                    ["tests/test_real_spell_amulet_crest_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
