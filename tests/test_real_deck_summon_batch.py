# -*- coding: utf-8 -*-
"""Direct deck-summon primitive and exact real-card behavior tests."""

from __future__ import annotations

import os
import sqlite3
import unittest
from collections import Counter
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import DeckFilter, EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, HandCard, Unit


CARD_IDS = (10164110, 10264110, 10322120, 10462120, 10813310)
SOURCE_HASHES = {
    10164110: "9d4f3ebfbbaa8f30a927083a52762ea0482d6dcdedefc7cae2ab841a7382dd95",
    10264110: "3879f8da367d7c6c6e4f0a76f07099026fb0d953038ff8060c6ef90ed14d0462",
    10322120: "ea8a237521dbb2577c47ecec5ef8784dd522e750550f905293471e3145e82197",
    10462120: "9189695175f2d0e517324f6e36380291d0e2822c4990164ce3e8ebba32f92ae9",
    10813310: "fe037adf8b0b9f3f29a0269f8c36cc8748f7c83ce5f3df6c9f62c2a7091d364a",
}
STRUCTURED_EVIDENCE = {
    10164110: {
        "triggers": ["fanfare", "super_evolve"],
        "effect_kinds": ["discard", "summon_from_deck", "destroy", "damage_unit"],
    },
    10264110: {
        "triggers": ["fanfare", "super_evolve", "intrinsic_keywords"],
        "effect_kinds": ["summon_from_deck", "buff_unit", "add_keyword", "keyword:守护"],
    },
    10322120: {
        "triggers": ["fanfare", "evolve"],
        "effect_kinds": ["summon_from_deck", "buff_unit"],
    },
    10462120: {
        "triggers": ["fanfare", "super_evolve"],
        "effect_kinds": ["summon_from_deck", "add_keyword"],
    },
    10813310: {
        "triggers": ["play"],
        "effect_kinds": ["summon_from_deck", "buff_unit"],
    },
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
        is_collectible=True,
    )


def _engine(
    rulebook: RuleBook,
    repository: CardRepository | None = None,
    *,
    seed: int = 3181,
) -> GameEngine:
    engine = GameEngine(
        [_card(i) for i in range(1000, 1040)],
        [_card(i) for i in range(2000, 2040)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
        card_resolver=None if repository is None else repository.get,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    for player in engine.players:
        player.hand.clear()
        player.hand_entity_ids.clear()
        player.max_mana = player.mana = 10
    return engine


def _put(engine: GameEngine, definition: CardDefinition) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[0].hand.insert(0, card)
    engine.players[0].hand_entity_ids.insert(0, card.entity_id)
    return card


def _play_real(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
):
    definition = repository.get(card_id)
    _put(engine, definition)
    engine.apply(PlayCard(0, 0))
    if definition.card_type == "法术":
        return None
    return next(
        entity
        for entity in reversed(engine.players[0].board)
        if entity.definition.card_id == card_id
    )


def _unit(engine: GameEngine, owner: int, card_id: int, *, attack=1, life=4) -> Unit:
    unit = Unit.summon(
        _card(card_id, attack=attack, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[owner].board.append(unit)
    return unit


class DeckSummonPrimitiveTests(unittest.TestCase):
    def test_schema_accepts_filters_and_rejects_unsafe_shapes(self):
        operation = _parse_operation(
            {
                "kind": "summon_from_deck",
                "target": "own_leader",
                "amount": 3,
                "card_type_filter": "随从",
                "class_id_filter": 2,
                "cost_max": 3,
                "tribe_name_filter": "学园",
            },
            "deck_summon.json",
            1,
        )
        self.assertEqual(operation.kind, EffectKind.SUMMON_FROM_DECK)
        self.assertEqual(
            (
                operation.deck_filter.card_type,
                operation.deck_filter.class_id,
                operation.deck_filter.cost_max,
                operation.deck_filter.tribe_name,
            ),
            ("随从", 2, 3, "学园"),
        )

        invalid = (
            {"kind": "summon_from_deck", "target": "enemy_leader", "amount": 1, "card_type_filter": "随从"},
            {"kind": "summon_from_deck", "target": "own_leader", "amount": 0, "card_type_filter": "随从"},
            {"kind": "summon_from_deck", "target": "own_leader", "amount": True, "card_type_filter": "随从"},
            {"kind": "summon_from_deck", "target": "own_leader", "amount": 1},
            {"kind": "summon_from_deck", "target": "own_leader", "amount": 1, "card_type_filter": "法术"},
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "deck_summon.json", 1)

    def _generic_rulebook(self, *, amount: int, card_type: str) -> RuleBook:
        return RuleBook((
            CardRule(
                1,
                Trigger.PLAY,
                (
                    EffectOperation(
                        EffectKind.SUMMON_FROM_DECK,
                        TargetKind.OWN_LEADER,
                        amount=amount,
                        deck_filter=DeckFilter(card_type=card_type, cost_max=3),
                    ),
                ),
            ),
            CardRule(
                10,
                Trigger.PLAY,
                (EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, amount=9),),
                countdown=2,
            ),
            CardRule(
                40,
                Trigger.FANFARE,
                (EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, amount=9),),
            ),
        ))

    def test_distinct_candidate_shortage_preserves_duplicates_and_skips_play_effects(self):
        fingerprints = []
        for _ in range(2):
            engine = _engine(self._generic_rulebook(amount=3, card_type="护符"), seed=19)
            amulet_a = _card(10, card_type="护符", attack=None, life=None, cost=1)
            amulet_b = _card(11, card_type="护符", attack=None, life=None, cost=2)
            engine.players[0].deck = [amulet_a, amulet_a, amulet_a, amulet_b]
            _put(engine, _card(1, card_type="法术", attack=None, life=None))

            engine.apply(PlayCard(0, 0))

            summoned = [entity for entity in engine.players[0].board if isinstance(entity, Amulet)]
            self.assertEqual({entity.definition.card_id for entity in summoned}, {10, 11})
            self.assertEqual(Counter(card.card_id for card in engine.players[0].deck), {10: 2})
            self.assertEqual(engine.players[1].health, 20)
            self.assertEqual(next(entity for entity in summoned if entity.definition.card_id == 10).countdown, 2)
            self.assertTrue(all(entity.origin is CardOrigin.DECK for entity in summoned))
            events = [event for event in engine.event_history if event.type is EventType.AMULET_ENTERED]
            self.assertEqual(len(events), 2)
            self.assertTrue(all(event.metadata["via"] == "deck_summon" for event in events))
            self.assertEqual([event.metadata["deck_summon_index"] for event in events], [0, 1])
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_board_shortage_removes_only_cards_that_enter(self):
        engine = _engine(self._generic_rulebook(amount=3, card_type="随从"), seed=23)
        for card_id in range(20, 24):
            _unit(engine, 0, card_id)
        engine.players[0].deck = [_card(30), _card(31), _card(32)]
        _put(engine, _card(1, card_type="法术", attack=None, life=None))
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].board), 5)
        self.assertEqual(len(engine.players[0].deck), 2)
        self.assertEqual(
            len([event for event in engine.event_history if event.type is EventType.FOLLOWER_SUMMONED]),
            1,
        )

        full = _engine(self._generic_rulebook(amount=3, card_type="随从"), seed=23)
        for card_id in range(50, 55):
            _unit(full, 0, card_id)
        full.players[0].deck = [_card(60), _card(61)]
        _put(full, _card(1, card_type="法术", attack=None, life=None))
        deck_before = list(full.players[0].deck)
        rng_before = full.random.getstate()
        full.apply(PlayCard(0, 0))
        self.assertEqual(full.players[0].deck, deck_before)
        self.assertEqual(full.random.getstate(), rng_before)

    def test_different_card_ids_with_the_same_name_count_as_one_kind(self):
        engine = _engine(self._generic_rulebook(amount=3, card_type="随从"), seed=29)
        engine.players[0].deck = [
            _card(35, name="同名随从"),
            _card(36, name="同名随从"),
            _card(37, name="另一随从"),
        ]
        _put(engine, _card(1, card_type="法术", attack=None, life=None))
        engine.apply(PlayCard(0, 0))
        self.assertEqual(
            {unit.definition.name for unit in engine.players[0].board},
            {"同名随从", "另一随从"},
        )
        self.assertEqual(len(engine.players[0].board), 2)

    def test_physical_copies_increase_selection_probability(self):
        selected = Counter()
        rulebook = self._generic_rulebook(amount=1, card_type="随从")
        for seed in range(80):
            engine = _engine(rulebook, seed=seed)
            engine.players[0].deck = [_card(40), _card(40), _card(40), _card(41)]
            _put(engine, _card(1, card_type="法术", attack=None, life=None))
            engine.apply(PlayCard(0, 0))
            selected[engine.players[0].board[0].definition.card_id] += 1
            self.assertEqual(engine.players[1].health, 20)
        self.assertGreater(selected[40], selected[41] * 2)


class DatabaseAndAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_snapshot_has_cards_without_modes_or_references(self):
        placeholders = ",".join("?" for _ in CARD_IDS)
        with closing(sqlite3.connect(self.db_path)) as connection:
            self.assertEqual(
                connection.execute(
                    f"SELECT COUNT(*) FROM cards WHERE card_id IN ({placeholders})",
                    CARD_IDS,
                ).fetchone()[0],
                len(CARD_IDS),
            )
            self.assertEqual(
                connection.execute(
                    f"SELECT COUNT(*) FROM alt_modes WHERE card_id IN ({placeholders})",
                    CARD_IDS,
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    f"SELECT COUNT(*) FROM card_references WHERE card_id IN ({placeholders})",
                    CARD_IDS,
                ).fetchone()[0],
                0,
            )

    def test_cards_have_exact_hashes_and_direct_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                audit = report["classifications"][str(card_id)]["clause_audit"]
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(audit["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(audit["structured_evidence"], STRUCTURED_EVIDENCE[card_id])
                self.assertEqual(audit["test_evidence"], ["tests/test_real_deck_summon_batch.py"])


class RealDeckSummonBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 3181) -> GameEngine:
        return _engine(self.rulebook, self.repository, seed=seed)

    def test_rodeo_discards_then_summons_three_amulets_and_super_evolves(self):
        engine = self.fresh(seed=31)
        amulet_a = _card(100, card_type="护符", attack=None, life=None, cost=1)
        amulet_b = _card(101, card_type="护符", attack=None, life=None, cost=2)
        amulet_c = _card(102, card_type="护符", attack=None, life=None, cost=3)
        engine.players[0].deck = [amulet_a, amulet_a, amulet_a, amulet_b, amulet_c, _card(103, cost=1)]
        discarded = _put(engine, _card(104))
        source = _play_real(engine, self.repository, 10164110)
        self.assertIsNotNone(engine.state.pending_choice)
        engine.apply(Choose(0, f"hand:{discarded.entity_id}"))
        self.assertEqual({entity.definition.card_id for entity in engine.players[0].board}, {10164110, 100, 101, 102})
        self.assertEqual(Counter(card.card_id for card in engine.players[0].deck)[100], 2)

        low = _unit(engine, 1, 110, attack=2, life=5)
        high_a = _unit(engine, 1, 111, attack=7, life=5)
        high_b = _unit(engine, 1, 112, attack=7, life=5)
        engine.players[0].turns_started = 7
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(low.health, 4)
        remaining_high = [unit for unit in (high_a, high_b) if unit in engine.players[1].board]
        self.assertEqual(len(remaining_high), 1)
        self.assertEqual(remaining_high[0].health, 4)

    def test_rodeo_only_card_in_hand_skips_discard_and_still_summons(self):
        engine = self.fresh(seed=37)
        engine.players[0].deck = [
            _card(120, card_type="护符", attack=None, life=None, cost=1),
            _card(121, card_type="护符", attack=None, life=None, cost=2),
            _card(122, card_type="护符", attack=None, life=None, cost=3),
        ]
        _play_real(engine, self.repository, 10164110)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(len(engine.players[0].board), 4)

    def test_aether_summons_distinct_followers_then_buffs_other_allies(self):
        engine = self.fresh(seed=41)
        engine.players[0].deck = [_card(130), _card(130), _card(131, cost=2), _card(132, cost=3), _card(133, cost=4)]
        source = _play_real(engine, self.repository, 10264110)
        summoned = [unit for unit in engine.players[0].board if unit is not source]
        self.assertEqual({unit.definition.card_id for unit in summoned}, {130, 131, 132})
        self.assertTrue(source.has_keyword("守护"))
        before = {unit.entity_id: unit.max_health for unit in summoned}
        engine.players[0].turns_started = 7
        engine.apply(SuperEvolve(0, source.entity_id))
        for unit in summoned:
            self.assertEqual(unit.max_health, before[unit.entity_id] + 2)
            self.assertTrue(unit.has_keyword("灵气"))
        self.assertFalse(source.has_keyword("灵气"))

    def test_scout_filters_royal_and_evolve_selects_other_follower(self):
        engine = self.fresh(seed=43)
        royal = _card(140, class_id=2, class_name="皇家护卫", cost=3)
        engine.players[0].deck = [royal, _card(141, class_id=1, cost=2), _card(142, class_id=2, cost=4)]
        source = _play_real(engine, self.repository, 10322120)
        summoned = next(unit for unit in engine.players[0].board if unit is not source)
        self.assertEqual(summoned.definition.card_id, 140)
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        engine.apply(Evolve(0, source.entity_id))
        self.assertIsNotNone(engine.state.pending_choice)
        engine.apply(Choose(0, f"entity:{summoned.entity_id}"))
        self.assertEqual((summoned.attack, summoned.health), (3, 6))

    def test_sophia_filters_bishop_and_super_evolve_grants_other_barrier(self):
        engine = self.fresh(seed=47)
        bishop = _card(150, class_id=6, class_name="主教", cost=2)
        engine.players[0].deck = [bishop, _card(151, class_id=6, cost=3), _card(152, class_id=1, cost=1)]
        source = _play_real(engine, self.repository, 10462120)
        summoned = next(unit for unit in engine.players[0].board if unit is not source)
        self.assertEqual(summoned.definition.card_id, 150)
        engine.players[0].turns_started = 7
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertTrue(summoned.has_keyword("屏障"))
        self.assertFalse(source.has_keyword("屏障"))

    def test_water_mirror_summons_two_distinct_then_buffs_both(self):
        engine = self.fresh(seed=53)
        engine.players[0].deck = [_card(160), _card(160), _card(161, cost=2), _card(162, cost=3)]
        _play_real(engine, self.repository, 10813310)
        summoned = [unit for unit in engine.players[0].board if isinstance(unit, Unit)]
        self.assertEqual({unit.definition.card_id for unit in summoned}, {160, 161})
        self.assertTrue(all((unit.attack, unit.health) == (2, 5) for unit in summoned))
        self.assertTrue(all(unit.origin is CardOrigin.DECK for unit in summoned))

    def test_rl_play_mask_auto_resolves_deck_summon_without_new_action_ids(self):
        env = ShadowverseEnv(
            [_card(i) for i in range(3000, 3040)],
            [_card(i) for i in range(3100, 3140)],
            class_a=1,
            class_b=1,
            seed=59,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=59)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].max_mana = env.players[0].mana = 10
        env.players[0].deck = [_card(170), _card(171, cost=2)]
        _put(env.core, self.repository.get(10813310))
        self.assertTrue(env.action_mask()[env.PLAY_OFFSET])
        env.step(env.PLAY_OFFSET)
        self.assertEqual(len(env.players[0].board), 2)
        self.assertIsNone(env.core.state.pending_choice)


if __name__ == "__main__":
    unittest.main()
