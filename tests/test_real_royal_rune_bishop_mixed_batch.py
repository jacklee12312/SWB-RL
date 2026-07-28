# -*- coding: utf-8 -*-
"""Exact mixed Royal/Runecraft/Bishop batch and keyword-filter primitive."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_condition, _parse_expression
from swb.engine.commands import (
    ActivateAmulet,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
    SuperEvolve,
)
from swb.engine.effects import BoardFilter, ConditionType, EffectKind, ExprType
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, HandCard, Unit


BATCH_CARD_IDS = (
    10121140,
    10322210,
    10623110,
    10822110,
    10031110,
    10331310,
    10531110,
    10631120,
    10633110,
    10731110,
    10833310,
    10463210,
    10562210,
    10662110,
    10763110,
)

# Filled from the generated report's canonical primary + alternate-mode text map.
SOURCE_HASHES = {
    10121140: "7af5d223e70dd603421a8514c345f30ca9fccdebf9e9cbaaa7416680447b1b2b",
    10322210: "f64fa951556b082dba2a661955ab5f6233af265b40f1fb7141f07e9e3eb5adff",
    10623110: "496419dee2cdcd5efe9e6c0abd6903c65e8514a4e1b7e92bbbfbf982bc8684b0",
    10822110: "27dfe594d8822b060399863df22f416b506af63ed638f361e6e2c5c933435af1",
    10031110: "dc67c3aa01c468008fe06b83b7c4260a93aed543b6e3ad421b009d68239af0e9",
    10331310: "ca4dc52756671bfe7fa3207e9d5d3649c2dbc5f2a307369b0748e0a686648101",
    10531110: "a176ff3bd6ba6a1a628b348689e6153e8518f52eafa3cbdcaa8fc167f83a5a8c",
    10631120: "839f301f23eb146392812734de223cc9e7b99d2a875849d43bba298e599d5c84",
    10633110: "d197edfc04551faa02467f95e664c53650e3d1c9b8e559db6d9ac3bd03823079",
    10731110: "b6b80810684c06ceb12761ebb0e7ad98de5643ff128cebf3cb490f28ab56ef75",
    10833310: "69b7918281d5e07ccf59e4ebba3d48d161ce902e2cfabc719c59137e6609f8d9",
    10463210: "26e65d10be7e1a75efbff5247c640fee04673ff5cac32d2e3f0f65de1aa5be74",
    10562210: "7ffd30e53f24652099ade5cce20acd92618ce2efd6e46608d526b52fba67252a",
    10662110: "fc33832a634400bd21315e7dcd1432f9cea08e1a6a98253f1bcd19d943752b7e",
    10763110: "348a546ea069a93573b26e23210f1c4801575ceaff3817a42894e118194cf396",
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
        life=overrides.get("life", 8),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _fresh(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 4101,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(11000, 11040)],
        [_card(card_id) for card_id in range(12000, 12040)],
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
    engine.players[owner].hand.insert(0, card)
    engine.players[owner].hand_entity_ids.insert(0, card.entity_id)
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
    amulet = Amulet(
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
    engine.players[0].board.append(amulet)
    return amulet


def _play(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
    *,
    mode_id: str = "normal",
) -> Unit | Amulet | None:
    source = _put_hand(engine, repository.get(card_id))
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


def _choose_entity(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolution(engine: GameEngine, *, super_evolve: bool = False) -> None:
    player = engine.players[0]
    player.turns_started = (
        engine.config.first_player_super_evolution_unlock_turn
        if super_evolve
        else engine.config.evolution_unlock_turn
    )
    player.evolution_points = max(1, player.evolution_points)
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.evolved_this_turn = False


class BoardKeywordFilterPrimitiveTests(unittest.TestCase):
    def test_board_count_keyword_filter_uses_live_runtime_keywords(self):
        expression = _parse_expression(
            {
                "type": "controller_board_count",
                "filter": {"card_type": "随从", "keyword": "守护"},
            },
            "test",
            1,
        )
        self.assertIs(expression.type, ExprType.CONTROLLER_BOARD_COUNT)
        self.assertEqual(expression.board_filter.keyword, "守护")

        plain = Unit.summon(_card(1), entity_id=1)
        self.assertFalse(expression.board_filter.matches_entity(plain))
        plain.add_keyword("守护")
        self.assertTrue(expression.board_filter.matches_entity(plain))

    def test_board_keyword_filter_rejects_unknown_keyword_and_defaults_compatibly(self):
        with self.assertRaisesRegex(ValueError, "Unknown ability keyword"):
            _parse_expression(
                {
                    "type": "controller_board_count",
                    "filter": {"keyword": "not-a-keyword"},
                },
                "test",
                1,
            )
        self.assertIsNone(BoardFilter(card_type="随从").keyword)

    def test_source_card_type_condition_is_validated_generically(self):
        condition = _parse_condition(
            {"type": "source_card_type_is", "card_type": "护符"},
            "test",
            1,
        )
        self.assertIs(condition.type, ConditionType.SOURCE_CARD_TYPE_IS)
        self.assertEqual(condition.card_type, "护符")
        with self.assertRaisesRegex(ValueError, "unknown card type"):
            _parse_condition(
                {"type": "source_card_type_is", "card_type": "leader"},
                "test",
                1,
            )


class RealRoyalRuneBishopMixedBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 4101) -> GameEngine:
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_modes_activations_crest_and_keyword_filter(self):
        for card_id in (10322210, 10463210, 10562210):
            self.assertEqual(self.rulebook.activation_for(card_id).cost, 0)
        self.assertEqual(self.rulebook.emblem_def("crystal_gazing").countdown, 2)
        self.assertEqual(
            [mode.mode_id for mode in self.rulebook.modes_for(10662110)],
            ["crystallize_1"],
        )
        dyer_last_words = self.rulebook.operations_for(10662110, Trigger.LAST_WORDS)[0]
        self.assertEqual(
            dyer_last_words.conditions[0].type,
            ConditionType.SOURCE_CARD_TYPE_IS,
        )
        self.assertEqual(dyer_last_words.conditions[0].card_type, "护符")
        draw = self.rulebook.operations_for(10562210, Trigger.ACTIVATE)[1]
        self.assertIs(draw.kind, EffectKind.DRAW)
        self.assertEqual(draw.amount_expr.board_filter.keyword, "守护")

    def test_hound_enhance_summons_two_rush_copies_and_respects_capacity(self):
        normal = self.fresh(seed=3)
        source = _play(normal, self.repository, 10121140)
        self.assertEqual(len(normal.players[0].board), 1)
        self.assertTrue(source.has_keyword("突进"))

        enhanced = self.fresh(seed=5)
        _play(enhanced, self.repository, 10121140, mode_id="enhance_6")
        hounds = [
            unit for unit in enhanced.players[0].board
            if unit.definition.card_id == 10121140
        ]
        self.assertEqual(len(hounds), 3)
        self.assertTrue(all(unit.has_keyword("突进") for unit in hounds))

        shortage = self.fresh(seed=7)
        for index in range(3):
            _put_unit(shortage, 0, _card(13000 + index))
        _play(shortage, self.repository, 10121140, mode_id="enhance_6")
        self.assertEqual(len(shortage.players[0].board), shortage.config.max_board)
        self.assertEqual(
            sum(unit.definition.card_id == 10121140 for unit in shortage.players[0].board),
            2,
        )

    def test_lair_activation_destroys_source_then_resolves_each_mode_and_overflow(self):
        expected = {
            "blade_necklace": [90021310, 90021340],
            "goblet_boots": [90021320, 90021330],
        }
        for option_id, token_ids in expected.items():
            with self.subTest(option_id=option_id):
                engine = self.fresh(seed=11)
                amulet = _play(engine, self.repository, 10322210)
                engine.apply(ActivateAmulet(0, amulet.entity_id))
                self.assertNotIn(amulet, engine.players[0].board)
                _choose_mode(engine, option_id)
                self.assertEqual(
                    [card.definition.card_id for card in engine.players[0].hand[:2]],
                    token_ids,
                )

        full = self.fresh(seed=13)
        for index in range(full.config.max_hand - 1):
            _put_hand(full, _card(13100 + index))
        amulet = _play(full, self.repository, 10322210)
        full.apply(ActivateAmulet(0, amulet.entity_id))
        _choose_mode(full, "blade_necklace")
        self.assertEqual(len(full.players[0].hand), full.config.max_hand)
        self.assertEqual(
            full.players[0].graveyard[-1].definition.card_id,
            90021340,
        )

    def test_strategist_optional_target_and_enhance_continue_after_stale_target(self):
        empty = self.fresh(seed=17)
        _play(empty, self.repository, 10623110)
        self.assertIsNone(empty.state.pending_choice)

        enhanced = self.fresh(seed=19)
        target = _put_unit(enhanced, 1, _card(13200))
        enhanced.players[0].mana = 10
        _play(enhanced, self.repository, 10623110, mode_id="enhance_6")
        request = enhanced.state.pending_choice
        choice = next(
            Choose(request.player_index, option.option_id)
            for option in request.options
            if option.entity_id == target.entity_id
        )
        enhanced.players[1].board.remove(target)
        enhanced.apply(choice)
        self.assertEqual(enhanced.players[0].mana, 7)
        self.assertEqual(enhanced.players[0].hand[0].definition.card_id, 10621110)

    def test_katze_spell_listener_is_once_per_owner_turn_and_evolve_adds_gold(self):
        engine = self.fresh(seed=23)
        source = _play(engine, self.repository, 10822110)
        enemy = _put_unit(engine, 1, _card(13300, life=10))
        for card_id in (13310, 13311):
            _put_hand(engine, self.repository.get(10571310))
            engine.apply(PlayCard(0, 0))
        self.assertEqual(enemy.health, 8)

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertTrue(any(
            card.definition.card_id == 90021350 for card in engine.players[0].hand
        ))

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        _put_hand(engine, self.repository.get(10571310))
        engine.apply(PlayCard(0, 0))
        self.assertEqual(enemy.health, 6)

    def test_runeknight_modes_spellboost_or_pay_earth_rite(self):
        boost = self.fresh(seed=29)
        hand_spell = _put_hand(
            boost,
            _card(13400, card_type="法术", attack=None, life=None),
        )
        source = _play(boost, self.repository, 10031110)
        _choose_mode(boost, "spellboost_twice")
        self.assertEqual(hand_spell.spellboost_count, 2)
        self.assertFalse(source.has_keyword("守护"))

        rite = self.fresh(seed=31)
        _put_earth_sigils(rite, 1)
        source = _play(rite, self.repository, 10031110)
        _choose_mode(rite, "earth_rite_buff")
        self.assertEqual((source.attack, source.health), (4, 4))
        self.assertTrue(source.has_keyword("守护"))
        self.assertEqual(rite.players[0].earth_sigils, 0)

        insufficient = self.fresh(seed=37)
        source = _play(insufficient, self.repository, 10031110)
        _choose_mode(insufficient, "earth_rite_buff")
        self.assertEqual((source.attack, source.health), (2, 2))
        self.assertFalse(source.has_keyword("守护"))

    def test_crystal_gazing_duplicate_is_ignored_and_expires_once(self):
        engine = self.fresh(seed=41)
        enemies = [_put_unit(engine, 1, _card(13500 + index)) for index in range(2)]
        _play(engine, self.repository, 10331310)
        _play(engine, self.repository, 10331310)
        self.assertEqual(len(engine.players[0].emblems), 1)
        deck_before = len(engine.players[0].deck)
        for _ in range(2):
            engine.apply(EndTurn(engine.current_player))
            engine.apply(EndTurn(engine.current_player))
        self.assertFalse(engine.players[0].emblems)
        self.assertEqual(len(engine.players[0].deck), deck_before - 4)
        self.assertEqual([enemy.health for enemy in enemies], [4, 4])

    def test_terraforming_and_librarian_super_evolve_outputs_respect_capacity(self):
        terraform = self.fresh(seed=43)
        source = _play(terraform, self.repository, 10531110)
        self.assertEqual(terraform.players[0].earth_sigils, 2)
        _enable_evolution(terraform, super_evolve=True)
        terraform.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(
            [
                unit.definition.card_id
                for unit in terraform.players[0].board
                if isinstance(unit, Unit)
            ],
            [10531110, 90031120, 90031120],
        )

        librarian = self.fresh(seed=47)
        source = _play(librarian, self.repository, 10631120)
        self.assertEqual(librarian.players[0].board[-1].definition.card_id, 10002120)
        ally = _put_unit(librarian, 0, _card(13600))
        _enable_evolution(librarian, super_evolve=True)
        librarian.apply(SuperEvolve(0, source.entity_id))
        self.assertFalse(source.has_keyword("突进"))
        self.assertTrue(ally.has_keyword("突进"))

        shortage = self.fresh(seed=53)
        for index in range(4):
            _put_unit(shortage, 0, _card(13610 + index))
        _play(shortage, self.repository, 10631120)
        self.assertEqual(len(shortage.players[0].board), shortage.config.max_board)
        self.assertFalse(any(
            unit.definition.card_id == 10002120 for unit in shortage.players[0].board
        ))

    def test_professor_summons_two_and_evolve_buffs_only_crystalspawn(self):
        engine = self.fresh(seed=59)
        other = _put_unit(engine, 0, _card(13700, attack=3, life=4))
        source = _play(engine, self.repository, 10633110)
        tokens = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10631110
        ]
        self.assertEqual(len(tokens), 2)
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual([token.attack for token in tokens], [2, 2])
        self.assertEqual(other.attack, 3)

    def test_dainty_horror_pays_earth_rite_to_evolve_and_keeps_ward(self):
        insufficient = self.fresh(seed=61)
        source = _play(insufficient, self.repository, 10731110)
        self.assertFalse(source.evolved)
        self.assertTrue(source.has_keyword("守护"))

        active = self.fresh(seed=67)
        _put_earth_sigils(active, 1)
        source = _play(active, self.repository, 10731110)
        self.assertTrue(source.evolved)
        self.assertEqual(active.players[0].earth_sigils, 0)
        self.assertTrue(source.has_keyword("守护"))

    def test_amethyst_draws_always_and_threshold_heals_and_restores_mana(self):
        below = self.fresh(seed=71)
        below.players[0].health = 10
        below.players[0].mana = 3
        source = _put_hand(below, self.repository.get(10833310))
        source.spellboost_count = 4
        deck_before = len(below.players[0].deck)
        below.apply(PlayCard(0, 0))
        self.assertEqual(len(below.players[0].deck), deck_before - 2)
        self.assertEqual((below.players[0].health, below.players[0].mana), (10, 0))

        active = self.fresh(seed=73)
        active.players[0].health = 10
        active.players[0].mana = 3
        source = _put_hand(active, self.repository.get(10833310))
        source.spellboost_count = 5
        active.apply(PlayCard(0, 0))
        self.assertEqual((active.players[0].health, active.players[0].mana), (12, 2))

    def test_gleaming_gems_modes_are_selectable_with_or_without_enemy(self):
        empty = self.fresh(seed=79)
        amulet = _play(empty, self.repository, 10463210)
        empty.apply(ActivateAmulet(0, amulet.entity_id))
        _choose_mode(empty, "destroy_random")
        self.assertIsNone(empty.state.pending_choice)

        draw = self.fresh(seed=83)
        amulet = _play(draw, self.repository, 10463210)
        deck_before = len(draw.players[0].deck)
        draw.apply(ActivateAmulet(0, amulet.entity_id))
        _choose_mode(draw, "draw_two")
        self.assertEqual(len(draw.players[0].deck), deck_before - 2)

        destroy = self.fresh(seed=89)
        targets = [_put_unit(destroy, 1, _card(13800 + index)) for index in range(2)]
        amulet = _play(destroy, self.repository, 10463210)
        destroy.apply(ActivateAmulet(0, amulet.entity_id))
        _choose_mode(destroy, "destroy_random")
        self.assertEqual(sum(target in destroy.players[1].board for target in targets), 1)

    def test_protective_shell_counts_printed_and_runtime_ward_followers_only(self):
        engine = self.fresh(seed=97)
        printed = _put_unit(engine, 0, self.repository.get(10763110))
        granted = _put_unit(engine, 0, _card(13900))
        granted.add_keyword("守护")
        _put_unit(engine, 0, _card(13901))
        _put_amulet(engine, 0, 13902)
        self.assertTrue(printed.has_keyword("守护"))
        amulet = _play(engine, self.repository, 10562210)
        deck_before = len(engine.players[0].deck)
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertEqual(len(engine.players[0].deck), deck_before - 2)

    def test_dyer_normal_keywords_and_crystallize_last_words(self):
        normal = self.fresh(seed=101)
        source = _play(normal, self.repository, 10662110)
        self.assertIsInstance(source, Unit)
        self.assertTrue(source.has_keyword("突进"))
        self.assertTrue(source.has_keyword("必杀"))

        source.health = 0
        normal._stabilize()
        self.assertFalse(any(
            entity.definition.card_id == 10662110
            for entity in normal.players[0].board
        ))

        crystal = self.fresh(seed=103)
        amulet = _play(
            crystal,
            self.repository,
            10662110,
            mode_id="crystallize_1",
        )
        self.assertIsInstance(amulet, Amulet)
        self.assertEqual(amulet.countdown, 3)
        amulet.countdown = 1
        while amulet in crystal.players[0].board:
            crystal.apply(EndTurn(crystal.current_player))
        summoned = next(
            unit for unit in crystal.players[0].board
            if unit.definition.card_id == 10662110
        )
        self.assertTrue(summoned.has_keyword("突进"))
        self.assertTrue(summoned.has_keyword("必杀"))

    def test_deacon_threshold_evolves_barriers_and_last_words_heals(self):
        below = self.fresh(seed=107)
        for index in range(2):
            _put_amulet(below, 0, 14000 + index)
        source = _play(below, self.repository, 10763110)
        self.assertFalse(source.evolved)
        self.assertFalse(source.has_keyword("屏障"))

        active = self.fresh(seed=109)
        active.players[0].health = 10
        for index in range(3):
            _put_amulet(active, 0, 14010 + index)
        source = _play(active, self.repository, 10763110)
        self.assertTrue(source.evolved)
        self.assertTrue(source.has_keyword("屏障"))
        self.assertTrue(source.has_keyword("守护"))
        source.health = 0
        active._stabilize()
        self.assertEqual(active.players[0].health, 13)

    def test_seeded_random_listener_and_mode_replay_are_identical(self):
        def run(seed: int):
            engine = self.fresh(seed=seed)
            _put_unit(engine, 1, _card(14100))
            _put_unit(engine, 1, _card(14101))
            _play(engine, self.repository, 10822110)
            _put_hand(engine, self.repository.get(10571310))
            engine.apply(PlayCard(0, 0))
            amulet = _play(engine, self.repository, 10463210)
            engine.apply(ActivateAmulet(0, amulet.entity_id))
            _choose_mode(engine, "destroy_random")
            return engine.deterministic_fingerprint(), tuple(engine.event_history)

        self.assertEqual(run(113), run(113))

    def test_rl_mask_exposes_enhance_crystallize_and_activation_modes(self):
        deck = [_card(14200 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=127,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=127)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10

        _put_hand(env.core, self.repository.get(10662110))
        normal = PlayCard(0, 0)
        crystal = PlayCard(0, 0, mode_id="crystallize_1")
        self.assertTrue(env.action_mask()[env._encode_command(normal)])
        self.assertTrue(env.action_mask()[env._encode_command(crystal)])
        env.step(env._encode_command(crystal))
        amulet = next(entity for entity in env.players[0].board if isinstance(entity, Amulet))
        activate = ActivateAmulet(0, amulet.entity_id)
        self.assertNotIn(activate, env.core.legal_commands())

        gems = _play(env.core, self.repository, 10463210)
        activate = ActivateAmulet(0, gems.entity_id)
        self.assertTrue(env.action_mask()[env._encode_command(activate)])
        env.step(env._encode_command(activate))
        self.assertEqual(
            {option.option_id for option in env.core.state.pending_choice.options},
            {"choose_one:destroy_random", "choose_one:draw_two"},
        )


class RoyalRuneBishopMixedDatabaseAuditTests(unittest.TestCase):
    def test_database_text_modes_and_references_match_reviewed_cards(self):
        expected_phrases = {
            10121140: ("Enhance", "Summon 2", "Rush"),
            10322210: ("Engage", "Gilded Blade", "Gilded Boots"),
            10623110: ("destroy it", "Recover 3 play points", "Fearless Soldier"),
            10822110: ("Once on each of your turns", "Glittering Gold"),
            10031110: ("Spellboost your hand 2 times", "Earth Rite", "Ward"),
            10331310: ("Crystal Gazing", "Draw 2 cards", "all enemy followers"),
            10531110: ("Gain 2 earth sigils", "Guardian Golem"),
            10631120: ("Caravan Mammoth", "all other allied followers", "Rush"),
            10633110: ("Summon 2", "Crystalspawn", "+1/+0"),
            10731110: ("Earth Rite", "Evolve this follower", "Ward"),
            10833310: ("X starts at 0", "X is at least 5", "recover 2 play points"),
            10463210: ("Engage", "Destroy a random enemy follower", "Draw 2 cards"),
            10562210: ("number of allied followers", "Ward"),
            10662110: ("Rush", "Bane", "Countdown", "Venerating Dyer"),
            10763110: ("at least 3 allied amulets", "Barrier", "Last Words"),
        }
        expected_references = {
            10121140: [10121140],
            10322210: [90021310, 90021340, 90021320, 90021330],
            10623110: [10621110],
            10822110: [90021350],
            10031110: [],
            10331310: [],
            10531110: [90031120],
            10631120: [10002120],
            10633110: [10631110],
            10731110: [],
            10833310: [],
            10463210: [],
            10562210: [],
            10662110: [10662110],
            10763110: [],
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
                    normalized = " ".join(
                        re.sub(r"<[^>]+>", "", text) for text in texts
                    )
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
                        1 if card_id in {10331310, 10662110} else 0,
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
                    ["tests/test_real_royal_rune_bishop_mixed_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
