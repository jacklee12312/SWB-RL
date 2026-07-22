# -*- coding: utf-8 -*-
"""Exact Dragon/Nightmare token chains plus Forest amulets and a Ward spell."""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import (
    ActivateAmulet,
    Attack,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
)
from swb.engine.effects import EffectKind, ExprType, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import Amulet, HandCard, Unit


BATCH_CARD_IDS = (
    10143130,
    10741110,
    10641120,
    10651120,
    10252120,
    10152130,
    10552110,
    10851120,
    10154120,
    10262310,
    10113210,
    10011210,
)

SOURCE_HASHES = {
    10143130: "a2ae2bea130ad1210f336f796f467927e038a5c626acc9e292220ad97f97e8da",
    10741110: "161941dfc9e6d5d0d5d782de8c57b35f4e00cbd72368d9b0cec04a2e10104723",
    10641120: "bdcaef57c4272ec3f466c6def0111dbc740978dd20ec0a2f0efa9879b64515bf",
    10651120: "57aa0f4f4b34c275bbc7891af55bced3bd15a6678277957838eead17720a0df0",
    10252120: "aa2575937758352cbd03dce271254104ae63633d6aeeb8db349cc4b6b13b0b53",
    10152130: "4bf016c818feef26ff83536d407a96c7863b66e78cffe9564d314cf3f73229d7",
    10552110: "390796b896389e5d981c075adfa454faa62cbae9ddf6f3fafcf18485ea7233cd",
    10851120: "4e7fa39c53b4cba6b8c19d58087828b110dd7066687566238810f979e014f850",
    10154120: "4e915929394ae016ddc2802d7071226afcba6226c332d2e18103f703fe1b07b4",
    10262310: "89d284bc4e510e2b26981e685d0fd3e0dbfcdc477fb4ae495a0f04404312c972",
    10113210: "3b0d9af6187a913f10a607384c374d3919765427a34c58fa4f4a64c14b876df3",
    10011210: "594bfc7f2f282cd2f603ac303f02f9c396e1ecc86492c380b3386a8ad0fe6d9f",
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
        life=overrides.get("life", 3),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
        tribe_id=overrides.get("tribe_id", 0),
        tribe_name=overrides.get("tribe_name", ""),
    )


def _fresh(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 7101,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(31000, 31040)],
        [_card(card_id) for card_id in range(32000, 32040)],
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


def _choose_entity(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    engine.apply(Choose(request.player_index, f"entity:{entity_id}"))


def _destroy_units(engine: GameEngine, *units: Unit) -> None:
    for unit in units:
        unit.health = 0
    engine._stabilize()


def _enable_evolve(engine: GameEngine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _rulebook_for(operations: list[dict]) -> RuleBook:
    payload = {
        "rules": [{
            "card_id": 990001,
            "trigger": "play",
            "operations": operations,
        }]
    }
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "rule.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        return RuleBook.from_directory(directory)


class LowCoverageTokenAmuletBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 7101) -> GameEngine:
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_modes_traits_intrinsics_and_live_health_binding(self):
        self.assertEqual(
            [mode.mode_id for mode in self.rulebook.modes_for(10741110)],
            ["enhance_4"],
        )
        self.assertEqual(
            [mode.mode_id for mode in self.rulebook.modes_for(10641120)],
            ["enhance_6"],
        )
        self.assertEqual(self.rulebook.attacks_per_turn(10154120), 3)
        self.assertEqual(
            self.rulebook.intrinsic_keywords_for(10154120),
            ("突进",),
        )
        self.assertEqual(self.rulebook.activation_for(10113210).cost, 0)
        self.assertIsNone(self.rulebook.countdown_for(10113210))
        self.assertEqual(self.rulebook.countdown_for(10011210), 2)
        listener = self.rulebook.listeners_for(10011210)[0]
        self.assertEqual(listener.event_filter.tribe_name, "妖精")
        self.assertIsNone(listener.event_filter.card_id)

        spell = self.rulebook.operations_for(10262310, Trigger.PLAY)
        self.assertEqual(
            [operation.kind for operation in spell],
            [EffectKind.BUFF_UNIT, EffectKind.DAMAGE_UNIT],
        )
        self.assertEqual(spell[0].target_key, "ward_target")
        self.assertEqual(spell[0].board_filter.keyword, "守护")
        self.assertIs(spell[1].amount_expr.type, ExprType.BOUND_TARGET_HEALTH)
        self.assertEqual(spell[1].amount_expr.binding_key, "ward_target")

    def test_bound_target_health_schema_requires_a_single_prior_binding(self):
        with self.assertRaisesRegex(ValueError, "requires a non-empty binding_key"):
            _rulebook_for([{
                "kind": "damage_unit",
                "target": "random_enemy_unit",
                "amount": {"type": "bound_target_health"},
            }])
        with self.assertRaisesRegex(ValueError, "was not defined"):
            _rulebook_for([{
                "kind": "damage_unit",
                "target": "random_enemy_unit",
                "amount": {
                    "type": "bound_target_health",
                    "binding_key": "missing",
                },
            }])
        with self.assertRaisesRegex(ValueError, "exactly one card"):
            _rulebook_for([
                {
                    "kind": "buff_unit",
                    "target": "own_unit",
                    "amount": 1,
                    "target_count": 2,
                    "target_key": "many",
                },
                {
                    "kind": "damage_unit",
                    "target": "random_enemy_unit",
                    "amount": {
                        "type": "bound_target_health",
                        "binding_key": "many",
                    },
                },
            ])

    def test_dragon_token_fanfares_and_enhance_obey_board_capacity(self):
        zaha = self.fresh(seed=3)
        source = _play(zaha, self.repository, 10143130)
        self.assertTrue(source.has_keyword("守护"))
        self.assertEqual(
            [entity.definition.card_id for entity in zaha.players[0].board],
            [10143130, 90041120],
        )

        zaha_full = self.fresh(seed=5)
        for index in range(zaha_full.config.max_board - 1):
            _put_unit(zaha_full, 0, _card(33000 + index))
        _play(zaha_full, self.repository, 10143130)
        self.assertFalse(any(
            entity.definition.card_id == 90041120
            for entity in zaha_full.players[0].board
        ))

        normal = self.fresh(seed=7)
        _play(normal, self.repository, 10741110)
        self.assertEqual(len(normal.players[0].board), 1)

        enhanced = self.fresh(seed=11)
        _play(enhanced, self.repository, 10741110, mode_id="enhance_4")
        promoters = [
            entity
            for entity in enhanced.players[0].board
            if entity.definition.card_id == 10741110
        ]
        self.assertEqual(len(promoters), 3)
        self.assertTrue(all(entity.has_keyword("突进") for entity in promoters))

        shortage = self.fresh(seed=13)
        for index in range(3):
            _put_unit(shortage, 0, _card(33100 + index))
        _play(shortage, self.repository, 10741110, mode_id="enhance_4")
        self.assertEqual(
            sum(
                entity.definition.card_id == 10741110
                for entity in shortage.players[0].board
            ),
            2,
        )

    def test_fruitfish_enhance_copies_keep_last_words_and_heal_simultaneously(self):
        engine = self.fresh(seed=17)
        engine.players[0].health = 16
        _play(engine, self.repository, 10641120, mode_id="enhance_6")
        fish = [
            entity
            for entity in engine.players[0].board
            if entity.definition.card_id == 10641120
        ]
        self.assertEqual(len(fish), 3)
        _destroy_units(engine, *fish)
        self.assertEqual(engine.players[0].health, 19)
        self.assertFalse(any(
            entity.definition.card_id == 10641120
            for entity in engine.players[0].board
        ))

    def test_nightmare_last_words_add_tokens_in_order_and_respect_hand_cap(self):
        ghost = self.fresh(seed=19)
        source = _play(ghost, self.repository, 10651120)
        self.assertTrue(source.has_keyword("突进"))
        _destroy_units(ghost, source)
        self.assertEqual(
            [card.card_id for card in ghost.players[0].hand],
            [90051130],
        )

        yuna = self.fresh(seed=23)
        source = _play(yuna, self.repository, 10152130)
        self.assertTrue(source.has_keyword("守护"))
        _destroy_units(yuna, source)
        self.assertEqual(
            [card.card_id for card in yuna.players[0].hand],
            [90051130, 90051120],
        )

        full = self.fresh(seed=29)
        source = _play(full, self.repository, 10152130)
        for index in range(full.config.max_hand):
            _put_hand(full, _card(33200 + index))
        _destroy_units(full, source)
        self.assertEqual(len(full.players[0].hand), full.config.max_hand)
        self.assertTrue(any(
            card.definition.card_id in {90051130, 90051120}
            and card.entry_cause == "hand_full"
            for card in full.players[0].graveyard
        ))

    def test_nightmare_fanfares_and_evolve_produce_complete_token_chain(self):
        zombies = self.fresh(seed=31)
        _play(zombies, self.repository, 10252120)
        self.assertEqual(
            [entity.definition.card_id for entity in zombies.players[0].board],
            [10252120, 90051140, 90051140],
        )

        shortage = self.fresh(seed=37)
        for index in range(3):
            _put_unit(shortage, 0, _card(33300 + index))
        _play(shortage, self.repository, 10252120)
        self.assertEqual(
            sum(
                entity.definition.card_id == 90051140
                for entity in shortage.players[0].board
            ),
            1,
        )

        summoner = self.fresh(seed=41)
        source = _play(summoner, self.repository, 10552110)
        self.assertEqual(
            [entity.definition.card_id for entity in summoner.players[0].board],
            [10552110, 90051130, 90051110],
        )
        _enable_evolve(summoner)
        summoner.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [entity.definition.card_id for entity in summoner.players[0].board],
            [10552110, 90051130, 90051110, 90051140],
        )

    def test_lilim_strike_hits_both_leaders_then_last_words_adds_bat(self):
        engine = self.fresh(seed=43)
        lilim = _play(engine, self.repository, 10851120)
        target = _put_unit(engine, 1, _card(33400, attack=1, life=4))
        lilim.can_attack = True
        engine.apply(Attack(0, lilim.entity_id, target.entity_id))

        self.assertEqual(
            (engine.players[0].health, engine.players[1].health),
            (19, 19),
        )
        self.assertNotIn(lilim, engine.players[0].board)
        self.assertEqual(target.health, 3)
        self.assertTrue(any(
            card.card_id == 90051120 for card in engine.players[0].hand
        ))

    def test_medusa_can_follower_strike_three_times_without_combat(self):
        engine = self.fresh(seed=47)
        medusa = _play(engine, self.repository, 10154120)
        targets = [
            _put_unit(engine, 1, _card(33500 + index, attack=20, life=20))
            for index in range(3)
        ]
        self.assertEqual(medusa.attacks_remaining, 3)
        self.assertTrue(medusa.has_keyword("突进"))
        for target in targets:
            command = Attack(0, medusa.entity_id, target.entity_id)
            self.assertIn(command, engine.legal_commands())
            engine.apply(command)
            self.assertNotIn(target, engine.players[1].board)
            self.assertIn(medusa, engine.players[0].board)
        self.assertEqual(medusa.attacks_remaining, 0)
        self.assertNotIn(Attack(0, medusa.entity_id, None), engine.legal_commands())

    def test_holy_guard_uses_post_buff_live_health_and_seeded_random_target(self):
        def resolved(seed: int):
            engine = self.fresh(seed=seed)
            ward = _put_unit(
                engine,
                0,
                _card(33600, life=4, keywords=frozenset({"守护"})),
            )
            enemies = [
                _put_unit(engine, 1, _card(33610 + index, life=10))
                for index in range(2)
            ]
            _play(engine, self.repository, 10262310)
            _choose_entity(engine, ward.entity_id)
            return (
                ward.max_health,
                tuple(enemy.health for enemy in enemies),
                engine.deterministic_fingerprint(),
            )

        first = resolved(53)
        second = resolved(53)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 5)
        self.assertEqual(sorted(first[1]), [5, 10])

        empty_enemy = self.fresh(seed=59)
        ward = _put_unit(
            empty_enemy,
            0,
            _card(33620, life=2, keywords=frozenset({"守护"})),
        )
        _play(empty_enemy, self.repository, 10262310)
        _choose_entity(empty_enemy, ward.entity_id)
        self.assertEqual((ward.health, ward.max_health), (3, 3))

    def test_holy_guard_no_target_is_atomic_and_stale_target_deals_zero(self):
        illegal = self.fresh(seed=61)
        source = _put_hand(illegal, self.repository.get(10262310))
        command = PlayCard(0, illegal.players[0].hand.index(source))
        before = illegal.deterministic_fingerprint()
        self.assertNotIn(command, illegal.legal_commands())
        with self.assertRaises(IllegalCommand):
            illegal.apply(command)
        self.assertEqual(illegal.deterministic_fingerprint(), before)

        stale = self.fresh(seed=67)
        ward = _put_unit(
            stale,
            0,
            _card(33700, life=6, keywords=frozenset({"守护"})),
        )
        enemy = _put_unit(stale, 1, _card(33701, life=9))
        _play(stale, self.repository, 10262310)
        stale.players[0].board.remove(ward)
        _choose_entity(stale, ward.entity_id)
        self.assertEqual(enemy.health, 9)
        self.assertIsNone(stale.state.pending_choice)

    def test_sacred_tree_staff_combo_draw_and_engage_legality(self):
        below = self.fresh(seed=71)
        below.players[0].cards_played_this_turn = 1
        amulet = _play(below, self.repository, 10113210)
        deck_before = len(below.players[0].deck)
        below.apply(EndTurn(0))
        self.assertEqual(len(below.players[0].deck), deck_before)
        self.assertIn(amulet, below.players[0].board)
        self.assertIsNone(amulet.countdown)

        active = self.fresh(seed=73)
        active.players[0].cards_played_this_turn = 2
        _play(active, self.repository, 10113210)
        deck_before = len(active.players[0].deck)
        active.apply(EndTurn(0))
        self.assertEqual(len(active.players[0].deck), deck_before - 1)

        no_target = self.fresh(seed=79)
        amulet = _play(no_target, self.repository, 10113210)
        command = ActivateAmulet(0, amulet.entity_id)
        before = no_target.deterministic_fingerprint()
        self.assertNotIn(command, no_target.legal_commands())
        with self.assertRaises(IllegalCommand):
            no_target.apply(command)
        self.assertEqual(no_target.deterministic_fingerprint(), before)

    def test_sacred_tree_staff_engage_destroys_source_then_returns_or_skips_stale_target(self):
        engine = self.fresh(seed=83)
        amulet = _play(engine, self.repository, 10113210)
        target = _put_unit(engine, 0, _card(33800))
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertNotIn(amulet, engine.players[0].board)
        _choose_entity(engine, target.entity_id)
        self.assertNotIn(target, engine.players[0].board)
        self.assertTrue(any(card.card_id == 33800 for card in engine.players[0].hand))

        stale = self.fresh(seed=89)
        amulet = _play(stale, self.repository, 10113210)
        target = _put_unit(stale, 0, _card(33801))
        stale.apply(ActivateAmulet(0, amulet.entity_id))
        stale.players[0].board.remove(target)
        _choose_entity(stale, target.entity_id)
        self.assertIsNone(stale.state.pending_choice)
        self.assertFalse(any(card.card_id == 33801 for card in stale.players[0].hand))

    def test_flowerbed_adds_fairy_and_trait_listener_is_exact_and_counted_down(self):
        engine = self.fresh(seed=97)
        amulet = _play(engine, self.repository, 10011210)
        self.assertEqual(amulet.countdown, 2)
        self.assertTrue(any(card.card_id == 90011110 for card in engine.players[0].hand))
        enemy = _put_unit(engine, 1, _card(33900, life=5))

        engine.players[0].mana = 10
        pixie = _put_hand(
            engine,
            _card(33901, tribe_id=5, tribe_name="妖精"),
        )
        engine.apply(PlayCard(0, engine.players[0].hand.index(pixie)))
        self.assertEqual(enemy.health, 4)

        engine.players[0].mana = 10
        other = _put_hand(engine, _card(33902, tribe_name="非妖精"))
        engine.apply(PlayCard(0, engine.players[0].hand.index(other)))
        self.assertEqual(enemy.health, 4)

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        self.assertEqual(amulet.countdown, 1)
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        self.assertNotIn(amulet, engine.players[0].board)

    def test_rl_mask_matches_targeted_spell_enhance_activation_and_rush_attacks(self):
        deck = [_card(34000 + index) for index in range(40)]
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

        spell = _put_hand(env.core, self.repository.get(10262310))
        play_spell = PlayCard(0, env.players[0].hand.index(spell))
        env.invalidate_cache(reason="test setup")
        self.assertFalse(env.action_mask()[env._encode_command(play_spell)])
        _put_unit(
            env.core,
            0,
            _card(34050, keywords=frozenset({"守护"})),
        )
        env.invalidate_cache(reason="ward added")
        self.assertTrue(env.action_mask()[env._encode_command(play_spell)])

        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        promoter = _put_hand(env.core, self.repository.get(10741110))
        normal = PlayCard(0, env.players[0].hand.index(promoter))
        enhance = PlayCard(0, env.players[0].hand.index(promoter), "enhance_4")
        env.invalidate_cache(reason="enhance setup")
        self.assertTrue(env.action_mask()[env._encode_command(normal)])
        self.assertTrue(env.action_mask()[env._encode_command(enhance)])

        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].board.clear()
        staff = _play(env.core, self.repository, 10113210)
        activate = ActivateAmulet(0, staff.entity_id)
        env.invalidate_cache(reason="activation setup")
        self.assertFalse(env.action_mask()[env._encode_command(activate)])
        _put_amulet(env.core, 0, 34060)
        env.invalidate_cache(reason="activation target added")
        self.assertTrue(env.action_mask()[env._encode_command(activate)])

        env.players[0].board.clear()
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        medusa = _play(env.core, self.repository, 10154120)
        target = _put_unit(env.core, 1, _card(34070, life=10))
        attack_follower = Attack(0, medusa.entity_id, target.entity_id)
        attack_leader = Attack(0, medusa.entity_id, None)
        env.invalidate_cache(reason="attack setup")
        self.assertTrue(env.action_mask()[env._encode_command(attack_follower)])
        self.assertFalse(env.action_mask()[env._encode_command(attack_leader)])


class LowCoverageTokenAmuletDatabaseAuditTests(unittest.TestCase):
    def test_database_text_modes_and_references_match_reviewed_cards(self):
        expected_phrases = {
            10143130: ("Fanfare", "Vastwing Dragon", "Ward"),
            10741110: ("Enhance", "2 copies", "Rush"),
            10641120: ("Enhance", "2 copies", "Last Words", "Restore 1 defense"),
            10651120: ("Rush", "Last Words", "Ghost"),
            10252120: ("Fanfare", "2 copies", "Rotting Zombie"),
            10152130: ("Ward", "Last Words", "Ghost", "Bat"),
            10552110: ("Fanfare", "Skeleton", "Evolve", "Rotting Zombie"),
            10851120: ("Strike", "both leaders", "Last Words", "Bat"),
            10154120: ("attack 3 times", "Follower", "Destroy the opposing follower"),
            10262310: ("Ward", "+0/+1", "random enemy follower", "defense"),
            10113210: ("Combo", "Engage", "another allied card", "return it to hand"),
            10011210: ("Fanfare", "Fairy", "Countdown", "allied Pixie follower"),
        }
        expected_references = {
            10143130: [90041120],
            10741110: [10741110],
            10641120: [10641120],
            10651120: [90051130],
            10252120: [90051140],
            10152130: [90051130, 90051120],
            10552110: [90051130, 90051110, 90051140],
            10851120: [90051120],
            10154120: [],
            10262310: [],
            10113210: [],
            10011210: [90011110],
        }
        with sqlite3.connect("data/cards.sqlite3") as connection:
            for card_id in BATCH_CARD_IDS:
                with self.subTest(card_id=card_id):
                    texts = [
                        row[0]
                        for row in connection.execute(
                            "SELECT text_eng FROM skill_texts "
                            "WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    ]
                    texts.extend(
                        row[0]
                        for row in connection.execute(
                            "SELECT text_eng FROM alt_modes "
                            "WHERE card_id=? ORDER BY position",
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
                        0,
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
                    ["tests/test_real_low_coverage_token_amulet_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
