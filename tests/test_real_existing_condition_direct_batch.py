# -*- coding: utf-8 -*-
"""Exact behavior audits for ten cards built from existing conditions/effects."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Choose, PlayCard
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, HandCard, Unit


CARD_IDS = (
    10012110, 10111110, 10111130, 10152140, 10433310,
    10451110, 10452120, 10512310, 10672120, 10762110,
)
SOURCE_HASHES = {
    10012110: "26aefc1cab89e9d86487f2cbed3cffcdc7ef3820e9d21ebed9b6fb6b85d6ae65",
    10111110: "b3c29d21a0caedb92079b60bc012185f03cc0e5303c7067218aa40024b12a5cb",
    10111130: "3feaf8101d173fd0d052330e02507160cde525c2e3e0577e4679bc12c148fc3c",
    10152140: "03d9194b590c0695aa4bb1792873f14341378584b19c504cad4404ab832f786a",
    10433310: "fba12a4d72e88498e08e31c9bb08a794eb9ac2c8cead941fd929d2bf97e0278a",
    10451110: "def1095057eda3f4e4b94016e9d736a08b476508d0b550f7746488341dfcbe34",
    10452120: "3dfc69f276f5f328a49984aac825794228ab508c44b17d627d1ffa42fd050406",
    10512310: "c8f2caf34a628c3ad37fc3150d3985f66f06de0989c82c9d49ca8704b13a6c2c",
    10672120: "1b806b135b05f99ecbd12aec3878bc95654898725550c2de49e5e33e051567b4",
    10762110: "200edddc2e78753ebb18605871687cc342f146a62fabbb975cd6ff679ed9b0eb",
}
STRUCTURED_EVIDENCE = {
    10012110: {"triggers": ["fanfare"], "effect_kinds": ["damage_unit"]},
    10111110: {"triggers": ["fanfare"], "effect_kinds": ["buff_unit"]},
    10111130: {"triggers": ["fanfare"], "effect_kinds": ["draw", "heal_leader"]},
    10152140: {"triggers": ["fanfare"], "effect_kinds": ["damage_unit", "heal_leader"]},
    10433310: {
        "triggers": ["play", "union_burst"],
        "effect_kinds": ["damage_unit", "add_earth_sigils", "damage_leader"],
    },
    10451110: {
        "triggers": ["fanfare", "play_modes", "intrinsic_keywords"],
        "effect_kinds": ["play_mode", "evolve_unit", "buff_unit", "keyword:突进"],
    },
    10452120: {
        "triggers": ["fanfare", "intrinsic_keywords"],
        "effect_kinds": ["conditional", "evolve_unit", "keyword:灵气"],
    },
    10512310: {
        "triggers": ["play"],
        "effect_kinds": [
            "conditional", "damage_unit", "damage_leader",
            "damage_unit", "damage_leader",
        ],
    },
    10672120: {
        "triggers": ["fanfare", "intrinsic_keywords"],
        "effect_kinds": ["banish", "keyword:潜行"],
    },
    10762110: {"triggers": ["fanfare"], "effect_kinds": ["damage_unit"]},
}


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
        life=overrides.get("life", 3),
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def _make_engine(rulebook: RuleBook, repository: CardRepository, seed: int = 1553) -> GameEngine:
    engine = GameEngine(
        [_card(i) for i in range(1000, 1040)],
        [_card(i) for i in range(2000, 2040)],
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
        player.max_mana = player.mana = 10
    return engine


def _put(engine: GameEngine, definition: CardDefinition) -> HandCard:
    card = HandCard(definition=definition, entity_id=engine.state.allocate_entity_id())
    engine.players[0].hand.insert(0, card)
    engine.players[0].hand_entity_ids.insert(0, card.entity_id)
    return card


def _play(engine: GameEngine, repository: CardRepository, card_id: int, mode: str = "normal") -> Unit:
    _put(engine, repository.get(card_id))
    engine.apply(PlayCard(0, 0, mode))
    return next(unit for unit in engine.players[0].board if unit.definition.card_id == card_id)


def _unit(engine: GameEngine, owner: int, card_id: int, *, life: int = 5) -> Unit:
    unit = Unit.summon(_card(card_id, life=life), entity_id=engine.state.allocate_entity_id())
    engine.players[owner].board.append(unit)
    return unit


def _amulet(engine: GameEngine, card_id: int) -> Amulet:
    amulet = Amulet(
        definition=_card(card_id, card_type="护符", attack=None, life=None),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[0].board.append(amulet)
    return amulet


def _choose(engine: GameEngine, entity_id: int) -> None:
    engine.apply(Choose(0, f"entity:{entity_id}"))


class CoverageAuditTests(unittest.TestCase):
    def test_all_ten_cards_are_exact_with_hash_structure_and_direct_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                audit = info["clause_audit"]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(audit["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(audit["structured_evidence"], STRUCTURED_EVIDENCE[card_id])
                self.assertEqual(audit["test_evidence"], ["tests/test_real_existing_condition_direct_batch.py"])


class ExistingConditionDirectBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, seed: int = 1553) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed)

    def test_combo_three_fanfares_use_post_play_count_and_exclude_source(self):
        low = self.fresh()
        enemy = _unit(low, 1, 3000)
        low.players[0].cards_played_this_turn = 1
        _play(low, self.repository, 10012110)
        self.assertIsNone(low.state.pending_choice)
        self.assertEqual(enemy.health, 5)

        active = self.fresh()
        enemy = _unit(active, 1, 3001)
        active.players[0].cards_played_this_turn = 2
        _play(active, self.repository, 10012110)
        _choose(active, enemy.entity_id)
        self.assertEqual(enemy.health, 2)

        buffs = self.fresh()
        ally = _unit(buffs, 0, 3002)
        buffs.players[0].cards_played_this_turn = 2
        source = _play(buffs, self.repository, 10111110)
        self.assertEqual((ally.attack, ally.health), (2, 6))
        definition = self.repository.get(10111110)
        self.assertEqual((source.attack, source.health), (definition.attack, definition.life))

    def test_draw_then_hand_count_heal_uses_resulting_hand_and_cap(self):
        engine = self.fresh()
        engine.players[0].deck = [_card(3100)]
        engine.players[0].health = 15
        _put(engine, _card(3101))
        _play(engine, self.repository, 10111130)
        self.assertEqual((len(engine.players[0].hand), engine.players[0].health), (2, 17))

        empty = self.fresh()
        empty.players[0].deck = []
        empty.players[0].health = 19
        _play(empty, self.repository, 10111130)
        self.assertEqual(empty.players[0].health, 18)

    def test_vlad_damages_selected_target_then_heals_even_without_target(self):
        engine = self.fresh()
        engine.players[0].health = 10
        target = _unit(engine, 1, 3200, life=5)
        _play(engine, self.repository, 10152140)
        _choose(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[0].health, 15)

        empty = self.fresh()
        empty.players[0].health = 18
        _play(empty, self.repository, 10152140)
        self.assertIsNone(empty.state.pending_choice)
        self.assertEqual(empty.players[0].health, 20)

    def test_alchemy_blast_damage_sigil_and_union_burst_are_ordered(self):
        engine = self.fresh()
        engine.players[0].turns_started = 10
        target = _unit(engine, 1, 3300, life=4)
        _put(engine, self.repository.get(10433310))
        engine.apply(PlayCard(0, 0))
        _choose(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[0].earth_sigils, 1)
        self.assertEqual(engine.players[1].health, 18)
        burst = next(
            event for event in engine.event_history
            if event.type is EventType.UNION_BURST_ACTIVATED
        )
        self.assertEqual((burst.amount, burst.metadata["threshold"]), (10, 10))

        empty = self.fresh()
        _put(empty, self.repository.get(10433310))
        empty.apply(PlayCard(0, 0))
        self.assertIsNone(empty.state.pending_choice)
        self.assertEqual((empty.players[0].earth_sigils, empty.players[1].health), (1, 20))

    def test_enhance_four_evolves_buffs_and_keeps_rush_without_ep(self):
        normal = self.fresh()
        source = _play(normal, self.repository, 10451110)
        self.assertFalse(source.evolved)
        self.assertTrue(source.has_keyword("突进"))

        enhanced = self.fresh()
        enhanced.players[0].mana = 4
        ep_before = enhanced.players[0].evolution_points
        source = _play(enhanced, self.repository, 10451110, "enhance_4")
        definition = self.repository.get(10451110)
        self.assertTrue(source.evolved)
        self.assertEqual((source.attack, source.health), (definition.attack + 3, definition.life + 3))
        self.assertEqual(enhanced.players[0].evolution_points, ep_before)

    def test_evolved_ally_condition_effect_evolves_aura_source(self):
        absent = self.fresh()
        source = _play(absent, self.repository, 10452120)
        self.assertFalse(source.evolved)
        self.assertTrue(source.has_aura)

        active = self.fresh()
        ally = _unit(active, 0, 3400)
        ally.evolved = True
        ep_before = active.players[0].evolution_points
        source = _play(active, self.repository, 10452120)
        self.assertTrue(source.evolved)
        self.assertTrue(source.has_aura)
        self.assertEqual(active.players[0].evolution_points, ep_before)

    def test_combo_replacement_spell_damages_board_and_leader_once(self):
        for before, amount in ((1, 1), (2, 2)):
            with self.subTest(before=before):
                engine = self.fresh()
                first = _unit(engine, 1, 3500, life=amount)
                survivor = _unit(engine, 1, 3501, life=amount + 2)
                engine.players[0].cards_played_this_turn = before
                _put(engine, self.repository.get(10512310))
                engine.apply(PlayCard(0, 0))
                self.assertNotIn(first, engine.players[1].board)
                self.assertEqual(survivor.health, 2)
                self.assertEqual(engine.players[1].health, 20 - amount)

    def test_health_filtered_banish_and_ambush_skip_invalid_or_missing_targets(self):
        engine = self.fresh()
        legal = _unit(engine, 1, 3600, life=3)
        too_large = _unit(engine, 1, 3601, life=4)
        source = _play(engine, self.repository, 10672120)
        self.assertTrue(source.ambush_active)
        self.assertEqual([option.entity_id for option in engine.state.pending_choice.options], [legal.entity_id])
        _choose(engine, legal.entity_id)
        self.assertNotIn(legal, engine.players[1].board)
        self.assertIn(too_large, engine.players[1].board)
        self.assertFalse(any(card.definition.card_id == 3600 for card in engine.players[1].graveyard))

        empty = self.fresh()
        _play(empty, self.repository, 10672120)
        self.assertIsNone(empty.state.pending_choice)

    def test_three_amulet_threshold_controls_target_choice_and_damage(self):
        below = self.fresh()
        for card_id in (3700, 3701):
            _amulet(below, card_id)
        target = _unit(below, 1, 3702)
        _play(below, self.repository, 10762110)
        self.assertIsNone(below.state.pending_choice)
        self.assertEqual(target.health, 5)

        active = self.fresh()
        for card_id in (3710, 3711, 3712):
            _amulet(active, card_id)
        target = _unit(active, 1, 3713)
        _play(active, self.repository, 10762110)
        _choose(active, target.entity_id)
        self.assertNotIn(target, active.players[1].board)

    def test_seeded_replay_and_rl_filtered_choice_mask(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=91)
            target = _unit(engine, 1, 3800, life=3)
            _play(engine, self.repository, 10672120)
            _choose(engine, target.entity_id)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

        env = ShadowverseEnv(
            [_card(i) for i in range(4000, 4040)],
            [_card(i) for i in range(4100, 4140)],
            class_a=1,
            class_b=1,
            seed=92,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=92)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].max_mana = env.players[0].mana = 10
        legal = _unit(env.core, 1, 3900, life=3)
        _unit(env.core, 1, 3901, life=4)
        _put(env.core, self.repository.get(10672120))
        env.step(env.PLAY_OFFSET)
        enabled = [
            action for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        ]
        self.assertEqual(enabled, [env.CHOICE_OFFSET])
        self.assertEqual(env.core.state.pending_choice.options[0].entity_id, legal.entity_id)


if __name__ == "__main__":
    unittest.main()
