# -*- coding: utf-8 -*-
"""Summon-output bindings, board class/trait filters, and exact real cards."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import (
    CardRule,
    RuleBook,
    Trigger,
    _parse_operation,
    _validate_target_keys,
)
from swb.engine.commands import Choose, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import DeckFilter, EffectKind, EffectOperation, TargetKind
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import HandCard, Unit


REAL_CARD_IDS = (
    10124120,
    10224110,
    10653110,
    10724120,
    10753110,
    10754110,
    10834120,
)
SOURCE_HASHES = {
    10124120: "e2596c65ae579e6a3f0d8abfacdfc2999d8def0e62c0b6a601f848dd2af9ef29",
    10224110: "70ceb395e1d5fad42c60482379471fdbd69ce9836dc6e01af36f78f50350e953",
    10653110: "0c43b4947fa33c11dbe455aebe9d9b3fe0fc5e274aae9088d9df8fb1d408cd01",
    10724120: "c7864871693019674225a5a71848cb8b0498c2340a3a5d8274def00ed2132a78",
    10753110: "e26ab1bc205bd1a8c48754cb8557519735ba01f5638a918704ef5e95cc7ba7e8",
    10754110: "35a1333e6072449eec05934e0aa39a05eb9e8ced1f8e0bad45ba2cca359a7937",
    10834120: "e03ce91cb30db657bd406e39387a344b81d92b9ccb15528f1f7af0d474eaed6f",
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
        life=overrides.get("life", 4),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
        tribe_id=overrides.get("tribe_id", 0),
        tribe_name=overrides.get("tribe_name", ""),
    )


def _engine(
    rulebook: RuleBook,
    resolver=None,
    *,
    seed: int = 211,
) -> GameEngine:
    engine = GameEngine(
        [_card(i) for i in range(1000, 1040)],
        [_card(i) for i in range(2000, 2040)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
        card_resolver=resolver,
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


def _play_real(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
) -> Unit:
    _put_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, 0))
    return next(
        entity
        for entity in engine.players[0].board
        if isinstance(entity, Unit) and entity.definition.card_id == card_id
    )


def _choose_entity(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolve(engine: GameEngine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _enable_super_evolve(engine: GameEngine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.first_player_super_evolution_unlock_turn
    player.super_evolution_points = max(1, player.super_evolution_points)
    player.super_evolved_this_turn = False


class OutputBindingPrimitiveTests(unittest.TestCase):
    def test_schema_accepts_summon_outputs_and_class_trait_board_filters(self):
        summon = _parse_operation(
            {
                "kind": "summon",
                "target": "own_leader",
                "card_id": 50,
                "target_key": "created",
            },
            "output.json",
            1,
        )
        followup = _parse_operation(
            {
                "kind": "buff_unit",
                "target": "previous_target",
                "target_key": "created",
                "amount": 1,
            },
            "output.json",
            1,
        )
        _validate_target_keys((summon, followup), "output.json")
        filtered = _parse_operation(
            {
                "kind": "buff_unit",
                "target": "all_own_units",
                "amount": 1,
                "target_class_id_filter": 5,
                "target_tribe_name_filter": "亡者",
            },
            "output.json",
            1,
        )
        self.assertEqual(
            (filtered.board_filter.class_id, filtered.board_filter.tribe_name),
            (5, "亡者"),
        )
        matching = _card(60, class_id=5, tribe_name="亡者")
        self.assertTrue(filtered.board_filter.matches(matching))
        self.assertFalse(
            filtered.board_filter.matches(_card(61, class_id=2, tribe_name="亡者"))
        )

        multi = _parse_operation(
            {
                "kind": "summon_from_deck",
                "target": "own_leader",
                "amount": 2,
                "card_type_filter": "随从",
                "target_key": "created_many",
            },
            "output.json",
            1,
        )
        conditional = EffectOperation(
            EffectKind.CONDITIONAL,
            TargetKind.OWN_LEADER,
            condition_target_key="created_many",
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _validate_target_keys((multi, conditional), "output.json")

    def test_summon_binding_only_modifies_the_new_entity(self):
        source = _card(1, card_type="法术", attack=None, life=None)
        token = _card(50, is_collectible=False)
        rules = RuleBook(
            (
                CardRule(
                    1,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.SUMMON,
                            TargetKind.OWN_LEADER,
                            card_id=50,
                            target_key="created",
                        ),
                        EffectOperation(
                            EffectKind.BUFF_UNIT,
                            TargetKind.PREVIOUS_TARGET,
                            amount=2,
                            secondary_amount=3,
                            target_key="created",
                        ),
                    ),
                ),
            )
        )
        engine = _engine(rules, {50: token}.get)
        old = _put_unit(engine, 0, _card(51))
        _put_hand(engine, source)
        engine.apply(PlayCard(0, 0))
        created = next(unit for unit in engine.players[0].board if unit is not old)
        self.assertEqual((old.attack, old.max_health), (1, 4))
        self.assertEqual((created.attack, created.max_health), (3, 7))

    def test_failed_summon_binds_empty_and_skips_followup(self):
        source = _card(1, card_type="法术", attack=None, life=None)
        token = _card(50, is_collectible=False)
        rule = CardRule(
            1,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.SUMMON,
                    TargetKind.OWN_LEADER,
                    card_id=50,
                    target_key="created",
                ),
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.PREVIOUS_TARGET,
                    amount=20,
                    target_key="created",
                ),
            ),
        )
        engine = _engine(RuleBook((rule,)), {50: token}.get)
        for cid in range(60, 65):
            _put_unit(engine, 0, _card(cid))
        _put_hand(engine, source)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].board), 5)
        self.assertEqual(engine.players[0].health, 20)

    def test_deck_summon_multi_output_binding_modifies_each_new_entity(self):
        source = _card(1, card_type="法术", attack=None, life=None)
        operation = EffectOperation(
            EffectKind.SUMMON_FROM_DECK,
            TargetKind.OWN_LEADER,
            amount=2,
            target_key="created",
            deck_filter=DeckFilter(card_type="随从"),
        )
        followup = EffectOperation(
            EffectKind.ADD_KEYWORD,
            TargetKind.PREVIOUS_TARGET,
            keyword="突进",
            target_key="created",
        )
        engine = _engine(
            RuleBook((CardRule(1, Trigger.PLAY, (operation, followup)),)),
        )
        old = _put_unit(engine, 0, _card(70))
        engine.players[0].deck = [_card(71), _card(72), _card(73, card_type="护符")]
        _put_hand(engine, source)
        engine.apply(PlayCard(0, 0))
        created = [unit for unit in engine.players[0].board if unit is not old]
        self.assertEqual(len(created), 2)
        self.assertTrue(all(unit.has_keyword("突进") for unit in created))
        self.assertFalse(old.has_keyword("突进"))


class DatabaseAndAuditTests(unittest.TestCase):
    def test_collectibles_are_mapped_exact_with_direct_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in REAL_CARD_IDS:
            with self.subTest(card_id=card_id):
                classification = report["classifications"][str(card_id)]
                audit = classification["clause_audit"]
                self.assertEqual(classification["coverage"], "covered_exact")
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(audit["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(
                    audit["test_evidence"],
                    ["tests/test_real_summon_output_trait_batch.py"],
                )


class RealSummonOutputTraitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 211) -> GameEngine:
        return _engine(self.rulebook, self.repository.get, seed=seed)

    def test_amelia_draws_royal_followers_restores_mana_and_filters_barrier(self):
        engine = self.fresh(seed=13)
        royal_a = _card(80, class_id=2, class_name="皇家护卫")
        royal_b = _card(81, class_id=2, class_name="皇家护卫")
        wrong = _card(82, class_id=5, class_name="梦魇")
        engine.players[0].deck = [royal_a, royal_b, wrong]
        engine.players[0].mana = 6
        source = _play_real(engine, self.repository, 10124120)
        self.assertEqual({card.card_id for card in engine.players[0].hand}, {80, 81})
        self.assertEqual(engine.players[0].mana, 3)
        royal = _put_unit(engine, 0, _card(83, class_id=2))
        nightmare = _put_unit(engine, 0, _card(84, class_id=5))
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertTrue(royal.has_keyword("屏障"))
        self.assertFalse(nightmare.has_keyword("屏障"))
        self.assertFalse(source.has_keyword("屏障"))

    def test_gildaria_cooperation_super_evolves_and_binds_both_knights(self):
        fanfare = self.rulebook.operations_for(10224110, Trigger.FANFARE)
        self.assertEqual(len(fanfare), 1)
        self.assertIs(fanfare[0].kind, EffectKind.SUPER_EVOLVE_UNIT)
        self.assertEqual(
            [operation.kind for operation in self.rulebook.operations_for(
                10224110,
                Trigger.SELF_EVOLVED,
            )],
            [
                EffectKind.SUMMON,
                EffectKind.SUMMON,
                EffectKind.ADD_KEYWORD,
                EffectKind.ADD_KEYWORD,
            ],
        )
        engine = self.fresh(seed=17)
        enemy_a = _put_unit(engine, 1, _card(90, life=5))
        enemy_b = _put_unit(engine, 1, _card(91, life=5))
        engine.players[0].cooperation = 19
        source = _play_real(engine, self.repository, 10224110)
        knights = [
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == 90021120
        ]
        self.assertTrue(source.super_evolved)
        self.assertEqual(len(knights), 2)
        self.assertTrue(all(unit.has_keyword("突进") for unit in knights))
        self.assertEqual((enemy_a.health, enemy_b.health), (3, 3))

    def test_destroyer_destroys_other_then_evolves_its_exact_summoned_bat(self):
        engine = self.fresh(seed=19)
        sacrifice = _put_unit(engine, 0, _card(100))
        source = _play_real(engine, self.repository, 10653110)
        self.assertIsNotNone(engine.state.pending_choice)
        _choose_entity(engine, sacrifice.entity_id)
        bats = [
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == 90051120
        ]
        self.assertTrue(source.evolved)
        self.assertEqual(len(bats), 1)
        self.assertTrue(bats[0].evolved)
        self.assertNotIn(sacrifice, engine.players[0].board)

    def test_cesar_buffs_only_other_royal_followers_and_destroys_on_super_evolve(self):
        engine = self.fresh(seed=23)
        royal = _put_unit(engine, 0, _card(110, class_id=2))
        nightmare = _put_unit(engine, 0, _card(111, class_id=5))
        source = _play_real(engine, self.repository, 10724120)
        knights = [unit for unit in engine.players[0].board if unit.definition.card_id == 90021120]
        self.assertEqual((royal.attack, royal.max_health), (2, 7))
        self.assertTrue(royal.has_keyword("守护"))
        self.assertEqual((nightmare.attack, nightmare.max_health), (1, 4))
        self.assertTrue(all((u.attack, u.max_health) == (3, 5) for u in knights))
        self.assertTrue(source.has_keyword("守护"))
        enemy = _put_unit(engine, 1, _card(112))
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose_entity(engine, enemy.entity_id)
        self.assertNotIn(enemy, engine.players[1].board)

    def test_beastmaster_grants_storm_and_zombie_replaces_itself_without_last_words(self):
        engine = self.fresh(seed=29)
        _play_real(engine, self.repository, 10753110)
        zombie = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90051140)
        skeleton = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90051110)
        self.assertTrue(zombie.has_keyword("疾驰"))
        self.assertTrue(skeleton.has_keyword("疾驰"))
        zombie.health = 0
        engine._stabilize()
        replacement = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90051140)
        self.assertNotEqual(replacement.entity_id, zombie.entity_id)
        self.assertTrue(replacement.printed_abilities_removed)
        replacement.health = 0
        engine._stabilize()
        self.assertFalse(any(unit.definition.card_id == 90051140 for unit in engine.players[0].board))

    def test_adahime_filters_deck_listener_and_super_evolve_buff_by_class(self):
        engine = self.fresh(seed=31)
        nightmare_a = _card(120, class_id=5, class_name="梦魇", cost=1)
        nightmare_b = _card(121, class_id=5, class_name="梦魇", cost=2)
        wrong_class = _card(122, class_id=2, cost=1)
        too_large = _card(123, class_id=5, cost=3)
        engine.players[0].deck = [nightmare_a, nightmare_b, wrong_class, too_large]
        source = _play_real(engine, self.repository, 10754110)
        summoned = [unit for unit in engine.players[0].board if unit is not source]
        self.assertEqual({unit.definition.card_id for unit in summoned}, {120, 121})
        self.assertTrue(all(unit.has_keyword("突进") for unit in summoned))
        royal = _put_unit(engine, 0, _card(124, class_id=2))
        before = {unit.entity_id: (unit.attack, unit.max_health) for unit in summoned}
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        for unit in summoned:
            attack, health = before[unit.entity_id]
            self.assertEqual((unit.attack, unit.max_health), (attack + 2, health + 2))
        self.assertEqual((royal.attack, royal.max_health), (1, 4))

    def test_ginger_summons_guard_golems_grants_rush_and_spellboosts_per_entry(self):
        engine = self.fresh(seed=37)
        tracked = _put_hand(engine, _card(130, card_type="法术", attack=None, life=None))
        source = _play_real(engine, self.repository, 10834120)
        golems = [unit for unit in engine.players[0].board if unit.definition.card_id == 90031120]
        self.assertEqual(len(golems), 2)
        self.assertTrue(all(unit.has_keyword("守护") and unit.has_keyword("突进") for unit in golems))
        self.assertEqual(tracked.spellboost_count, 2)
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        golems = [unit for unit in engine.players[0].board if unit.definition.card_id == 90031120]
        self.assertEqual(len(golems), 3)
        self.assertTrue(golems[-1].has_keyword("突进"))
        self.assertEqual(tracked.spellboost_count, 3)


if __name__ == "__main__":
    unittest.main()
