# -*- coding: utf-8 -*-
"""Exact real cards composed entirely from established engine primitives."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import ActivateAmulet, Choose, EndTurn, PlayCard
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, HandCard, Unit
from tests.play_mode_test_support import prepare_mana_for_play_mode


CARD_IDS = (
    10002210,
    10123120,
    10124110,
    10162220,
    10164120,
    10403120,
    10551110,
    10662210,
    10762210,
    10814110,
)


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=overrides.get("class_id", 1),
        class_name=overrides.get("class_name", "精灵"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 5),
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def _fresh(rulebook: RuleBook, repository: CardRepository, *, seed: int = 401) -> GameEngine:
    engine = GameEngine(
        [_card(i) for i in range(3000, 3040)],
        [_card(i) for i in range(4000, 4040)],
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
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[0].hand.insert(0, card)
    engine.players[0].hand_entity_ids.insert(0, card.entity_id)
    return card


def _put_unit(engine: GameEngine, owner: int, definition: CardDefinition) -> Unit:
    unit = Unit.summon(definition, entity_id=engine.state.allocate_entity_id())
    engine.players[owner].board.append(unit)
    return unit


def _play(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
    *,
    mode_id: str = "normal",
):
    hand_card = _put_hand(engine, repository.get(card_id))
    prepare_mana_for_play_mode(engine, hand_card, mode_id)
    engine.apply(PlayCard(0, 0, mode_id=mode_id))
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


class RealBasicExistingPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 401) -> GameEngine:
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_four_enhance_followers_use_additive_modes_and_intrinsic_keywords(self):
        sniper = self.fresh(seed=3)
        target = _put_unit(sniper, 1, _card(10, life=9))
        source = _play(sniper, self.repository, 10123120, mode_id="enhance_6")
        _choose(sniper, target.entity_id)
        self.assertEqual(target.health, 4)
        definition = self.repository.get(10123120)
        self.assertEqual(
            (source.attack, source.max_health),
            (definition.attack + 2, definition.life + 2),
        )
        self.assertTrue(source.has_keyword("潜行"))

        albert = self.fresh(seed=5)
        enemies = [_put_unit(albert, 1, _card(20 + i, life=6)) for i in range(2)]
        source = _play(albert, self.repository, 10124110, mode_id="enhance_9")
        self.assertEqual([unit.health for unit in enemies], [3, 3])
        self.assertEqual(source.attacks_per_turn, 2)
        self.assertTrue(source.has_keyword("疾驰"))

        luria = self.fresh(seed=7)
        high = _card(30, cost=7)
        luria.players[0].deck = [high, _card(31, cost=6)]
        source = _play(luria, self.repository, 10403120, mode_id="enhance_8")
        self.assertEqual([card.card_id for card in luria.players[0].hand], [30])
        self.assertEqual(luria.players[0].mana, 9)
        self.assertTrue(source.has_keyword("屏障"))

        wolf = self.fresh(seed=11)
        source = _play(wolf, self.repository, 10551110, mode_id="enhance_6")
        definition = self.repository.get(10551110)
        self.assertEqual(source.max_health, definition.life + 6)
        self.assertTrue(source.has_keyword("突进"))
        self.assertTrue(source.has_keyword("必杀"))
        self.assertTrue(source.has_keyword("屏障"))

    def test_normal_mode_does_not_apply_enhance_operations(self):
        engine = self.fresh(seed=13)
        enemy = _put_unit(engine, 1, _card(40, life=6))
        source = _play(engine, self.repository, 10124110)
        self.assertEqual(enemy.health, 6)
        self.assertEqual(source.attacks_per_turn, 1)

    def test_jeanne_and_sedus_apply_ordered_fanfare_effects(self):
        jeanne = self.fresh(seed=17)
        ally = _put_unit(jeanne, 0, _card(50, attack=2, life=3))
        enemies = [_put_unit(jeanne, 1, _card(51 + i, life=7)) for i in range(2)]
        source = _play(jeanne, self.repository, 10164120)
        self.assertEqual([unit.health for unit in enemies], [1, 1])
        self.assertEqual((ally.attack, ally.max_health), (4, 7))
        self.assertTrue(source.has_keyword("守护"))

        sedus = self.fresh(seed=19)
        allies = [_put_unit(sedus, 0, _card(60 + i)) for i in range(2)]
        target = _put_unit(sedus, 1, _card(62))
        source = _play(sedus, self.repository, 10814110)
        _choose(sedus, target.entity_id)
        self.assertNotIn(target, sedus.players[1].board)
        self.assertTrue(all((unit.attack, unit.max_health) == (2, 6) for unit in allies))
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertTrue(source.has_keyword("守护"))

    def test_adventurers_guild_draws_then_activates_after_destroying_itself(self):
        engine = self.fresh(seed=23)
        follower = _card(70, card_type="随从")
        engine.players[0].deck = [follower, _card(71, card_type="法术", attack=None, life=None)]
        ally = _put_unit(engine, 0, _card(72))
        amulet = _play(engine, self.repository, 10002210)
        self.assertEqual([card.card_id for card in engine.players[0].hand], [70])
        mana_before = engine.players[0].mana
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertNotIn(amulet, engine.players[0].board)
        self.assertEqual(engine.players[0].mana, mana_before - 1)
        _choose(engine, ally.entity_id)
        self.assertTrue(ally.has_keyword("突进"))

    def test_holy_injection_and_perfect_clock_destroy_before_followups(self):
        injection = self.fresh(seed=29)
        injection.players[0].health = 18
        enemy = _put_unit(injection, 1, _card(80, life=6))
        amulet = _play(injection, self.repository, 10162220)
        injection.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertNotIn(amulet, injection.players[0].board)
        _choose(injection, enemy.entity_id)
        self.assertEqual(enemy.health, 2)
        self.assertEqual(injection.players[0].health, 19)

        clock = self.fresh(seed=31)
        enemies = [_put_unit(clock, 1, _card(90 + i, life=5)) for i in range(2)]
        amulet = _play(clock, self.repository, 10762210, mode_id="enhance_4")
        self.assertEqual([unit.health for unit in enemies], [4, 4])
        clock.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertNotIn(amulet, clock.players[0].board)
        self.assertEqual([unit.health for unit in enemies], [3, 3])

    def test_scriptures_countdown_last_words_draw_damage_and_heal(self):
        engine = self.fresh(seed=37)
        engine.players[0].health = 16
        engine.players[0].deck = [_card(100), _card(101), _card(102)]
        enemies = [_put_unit(engine, 1, _card(103 + i, life=5)) for i in range(2)]
        amulet = _play(engine, self.repository, 10662210)
        self.assertIsInstance(amulet, Amulet)
        self.assertEqual(amulet.countdown, 4)
        amulet.countdown = 1
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        self.assertNotIn(amulet, engine.players[0].board)
        self.assertEqual(len(engine.players[0].hand), 3)
        self.assertEqual([unit.health for unit in enemies], [3, 3])
        self.assertEqual(engine.players[0].health, 18)


class DatabaseAndAuditTests(unittest.TestCase):
    def test_cards_are_mapped_exact_with_direct_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                classification = report["classifications"][str(card_id)]
                audit = classification["clause_audit"]
                self.assertEqual(classification["coverage"], "covered_exact")
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(
                    audit["test_evidence"],
                    ["tests/test_real_basic_existing_primitives_batch.py"],
                )


if __name__ == "__main__":
    unittest.main()
