# -*- coding: utf-8 -*-
"""Exact tests for evolve, Earth Rite, listener, and Burst direct effects."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Attack, Choose, EndTurn, PlayCard
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, HandCard, Unit


CARD_IDS = (
    10231310, 10411120, 10432110, 10442110, 10453110, 10473310,
    10571120, 10622120, 10642120, 10742120, 10832110,
)
SOURCE_HASHES = {
    10231310: "8da1132496276b2794ce1408301c78d73f553d98a544931a37c4e2ab50267617",
    10411120: "7decd7f0c95b94965df52af0211ce27fc942c54e5f4c94eb9f83553f92230cd8",
    10432110: "c367663a3976ee5168025ac401c9aae571629dc0250e55f4cd22583168c080e6",
    10442110: "cef589d2f3f27bb64c8f8fb20c2664b7bf99f694bab273dd90fc18a309c113a5",
    10453110: "b03670707991dd551a36bb6db7f83a48a5398b769294c37036513a2a12b39c48",
    10473310: "8394d8e7db7d1b4d173b7e2df3299cac4581bef39acdff9130623e17bf623e01",
    10571120: "f7ffcfa90b6d070c9624d924d43bcc80c340f9ce259eb979b1370c8c3b68ea29",
    10622120: "abb6c7aeb08490b5d36517e183b64925bd19ef1108fe70a91c6e774f28240a94",
    10642120: "3fec74d8e5cb405f37515990917b48b191d5cf322fe317e05e08d41def740e35",
    10742120: "e3b72e6e62153abd5f5351417cbae2374d435e9a3c250dae0d9be1f946faa709",
    10832110: "d9a03d3746f6901bef81ef2dcae7798407125c146771e9954d19b024f4f7f5ae",
}
STRUCTURED_EVIDENCE = {
    10231310: {"triggers": ["play"], "effect_kinds": ["destroy", "earth_rite", "damage_leader"]},
    10411120: {"triggers": ["turn_end", "evolve"], "effect_kinds": ["evolve_unit", "damage_unit"]},
    10432110: {"triggers": ["fanfare"], "effect_kinds": ["damage_unit", "add_earth_sigils"]},
    10442110: {"triggers": ["fanfare", "evolve"], "effect_kinds": ["conditional", "evolve_unit", "damage_unit"]},
    10453110: {"triggers": ["fanfare", "intrinsic_keywords"], "effect_kinds": ["evolve_unit", "damage_leader", "keyword:吸血"]},
    10473310: {"triggers": ["play", "super_skybound_art"], "effect_kinds": ["damage_unit", "damage_leader", "damage_unit", "damage_leader"]},
    10571120: {"triggers": ["fanfare", "listener:board:card_played"], "effect_kinds": ["draw_filtered", "damage_unit"]},
    10622120: {"triggers": ["fanfare", "intrinsic_keywords"], "effect_kinds": ["destroy", "keyword:疾驰"]},
    10642120: {"triggers": ["turn_end", "evolve"], "effect_kinds": ["evolve_unit", "damage_unit", "damage_leader"]},
    10742120: {"triggers": ["fanfare", "turn_end"], "effect_kinds": ["conditional", "evolve_unit", "conditional", "heal_leader", "heal_leader"]},
    10832110: {"triggers": ["fanfare", "passive"], "effect_kinds": ["draw", "draw", "spellboost_cost_reduction"]},
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
    )


def _spell(card_id: int, **overrides) -> CardDefinition:
    return _card(card_id, card_type="法术", attack=None, life=None, **overrides)


def _engine(rulebook: RuleBook, repository: CardRepository, seed: int = 1667) -> GameEngine:
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


def _put(engine: GameEngine, definition: CardDefinition, player_index: int = 0) -> HandCard:
    card = HandCard(definition=definition, entity_id=engine.state.allocate_entity_id())
    player = engine.players[player_index]
    player.hand.insert(0, card)
    player.hand_entity_ids.insert(0, card.entity_id)
    return card


def _play(engine: GameEngine, repository: CardRepository, card_id: int) -> Unit | None:
    definition = repository.get(card_id)
    _put(engine, definition)
    engine.apply(PlayCard(0, 0))
    if definition.card_type == "法术":
        return None
    return next(unit for unit in engine.players[0].board if unit.definition.card_id == card_id)


def _unit(engine: GameEngine, owner: int, card_id: int, *, life: int = 5) -> Unit:
    unit = Unit.summon(_card(card_id, life=life), entity_id=engine.state.allocate_entity_id())
    engine.players[owner].board.append(unit)
    return unit


def _sigil(engine: GameEngine, count: int = 1) -> Amulet:
    amulet = Amulet(
        definition=_card(
            90031210,
            card_set_id=90000,
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


def _choose(engine: GameEngine, entity_id: int) -> None:
    engine.apply(Choose(0, f"entity:{entity_id}"))


class DatabaseAndCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_cards_have_no_references_or_alternate_modes(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            placeholders = ",".join("?" for _ in CARD_IDS)
            self.assertEqual(
                connection.execute(
                    f"SELECT COUNT(*) FROM cards WHERE card_id IN ({placeholders})",
                    CARD_IDS,
                ).fetchone()[0],
                len(CARD_IDS),
            )
            self.assertEqual(
                connection.execute(
                    f"SELECT COUNT(*) FROM card_references WHERE card_id IN ({placeholders})",
                    CARD_IDS,
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute(
                    f"SELECT COUNT(*) FROM alt_modes WHERE card_id IN ({placeholders})",
                    CARD_IDS,
                ).fetchone()[0],
                0,
            )

    def test_all_eleven_cards_are_exact_with_hash_and_structure(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                audit = report["classifications"][str(card_id)]["clause_audit"]
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(audit["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(audit["structured_evidence"], STRUCTURED_EVIDENCE[card_id])
                self.assertEqual(audit["test_evidence"], ["tests/test_real_evolve_and_burst_direct_batch.py"])


class RealEvolveAndBurstBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, seed: int = 1667) -> GameEngine:
        return _engine(self.rulebook, self.repository, seed)

    def test_ice_lance_destroys_then_pays_earth_rite_for_leader_damage(self):
        engine = self.fresh()
        _sigil(engine)
        target = _unit(engine, 1, 3000)
        _play(engine, self.repository, 10231310)
        _choose(engine, target.entity_id)
        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[0].earth_sigils, 0)
        self.assertEqual(engine.players[1].health, 18)

        insufficient = self.fresh()
        target = _unit(insufficient, 1, 3001)
        _play(insufficient, self.repository, 10231310)
        _choose(insufficient, target.entity_id)
        self.assertEqual(insufficient.players[1].health, 20)

    def test_manamar_turn_end_effect_evolves_and_damages_simultaneously(self):
        engine = self.fresh()
        doomed = _unit(engine, 1, 3100, life=1)
        survivor = _unit(engine, 1, 3101, life=3)
        source = _play(engine, self.repository, 10411120)
        ep_before = engine.players[0].evolution_points
        engine.apply(EndTurn(0))
        self.assertTrue(source.evolved)
        self.assertEqual(engine.players[0].evolution_points, ep_before)
        self.assertNotIn(doomed, engine.players[1].board)
        self.assertEqual(survivor.health, 2)

    def test_fortune_teller_selects_two_then_adds_sigils_with_shortage(self):
        engine = self.fresh()
        first = _unit(engine, 1, 3200, life=4)
        second = _unit(engine, 1, 3201, life=5)
        _play(engine, self.repository, 10432110)
        _choose(engine, first.entity_id)
        _choose(engine, second.entity_id)
        self.assertNotIn(first, engine.players[1].board)
        self.assertEqual(second.health, 1)
        self.assertEqual(engine.players[0].earth_sigils, 2)

        shortage = self.fresh()
        only = _unit(shortage, 1, 3202, life=5)
        _play(shortage, self.repository, 10432110)
        _choose(shortage, only.entity_id)
        self.assertEqual((only.health, shortage.players[0].earth_sigils), (1, 2))

    def test_max_mana_effect_evolution_triggers_enemy_wide_damage(self):
        below = self.fresh()
        below.players[0].max_mana = below.players[0].mana = 9
        target = _unit(below, 1, 3300)
        source = _play(below, self.repository, 10442110)
        self.assertFalse(source.evolved)
        self.assertEqual(target.health, 5)

        active = self.fresh()
        target = _unit(active, 1, 3301, life=3)
        ep_before = active.players[0].evolution_points
        source = _play(active, self.repository, 10442110)
        self.assertTrue(source.evolved)
        self.assertNotIn(target, active.players[1].board)
        self.assertEqual(active.players[0].evolution_points, ep_before)

    def test_nirvana_evolves_all_unevolved_then_self_damages_and_has_drain(self):
        engine = self.fresh()
        first = _unit(engine, 0, 3400)
        already = _unit(engine, 0, 3401)
        already.evolved = True
        engine.players[0].health = 20
        source = _play(engine, self.repository, 10453110)
        self.assertTrue(first.evolved)
        self.assertTrue(already.evolved)
        self.assertTrue(source.evolved)
        self.assertTrue(source.has_keyword("吸血"))
        self.assertEqual(engine.players[0].health, 18)

    def test_chaos_force_super_art_replaces_three_with_exactly_six(self):
        for gauge, amount in ((14, 3), (15, 6)):
            with self.subTest(gauge=gauge):
                engine = self.fresh()
                engine.players[0].turns_started = gauge
                target = _unit(engine, 1, 3500, life=10)
                _put(engine, self.repository.get(10473310))
                engine.apply(PlayCard(0, 0))
                self.assertEqual(target.health, 10 - amount)
                self.assertEqual(engine.players[1].health, 20 - amount)
                bursts = [event for event in engine.event_history if event.type is EventType.UNION_BURST_ACTIVATED]
                self.assertEqual(len(bursts), int(gauge == 15))

    def test_flower_technician_draws_spell_and_listens_only_to_owner_spells(self):
        engine = self.fresh()
        engine.players[0].deck = [_card(3600), _spell(3601), _spell(3602)]
        source = _play(engine, self.repository, 10571120)
        self.assertIn(engine.players[0].hand[0].card_id, {3601, 3602})
        target = _unit(engine, 1, 3603, life=4)
        _put(engine, self.repository.get(10512310))
        engine.apply(PlayCard(0, 0))
        self.assertIn(source, engine.players[0].board)
        self.assertNotIn(target, engine.players[1].board)

        engine.apply(EndTurn(0))
        opponent_target = _unit(engine, 1, 3604, life=5)
        _put(engine, self.repository.get(10512310), player_index=1)
        engine.players[1].max_mana = engine.players[1].mana = 10
        engine.apply(PlayCard(1, 0))
        self.assertEqual(opponent_target.health, 5)

    def test_cat_sailor_destroys_exact_health_one_batch_and_has_storm(self):
        engine = self.fresh()
        one = _unit(engine, 1, 3700, life=1)
        two = _unit(engine, 1, 3701, life=2)
        source = _play(engine, self.repository, 10622120)
        self.assertNotIn(one, engine.players[1].board)
        self.assertIn(two, engine.players[1].board)
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertIn(Attack(0, source.entity_id, None), engine.legal_commands())
        destroyed = [event for event in engine.event_history if event.type is EventType.DEATH_BATCH_START]
        self.assertEqual(destroyed[-1].metadata["batch_record_count"], 1)

    def test_spike_dragon_turn_end_evolves_then_damages_board_and_leader(self):
        engine = self.fresh()
        target = _unit(engine, 1, 3800, life=3)
        source = _play(engine, self.repository, 10642120)
        ep_before = engine.players[0].evolution_points
        engine.apply(EndTurn(0))
        self.assertTrue(source.evolved)
        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[1].health, 17)
        self.assertEqual(engine.players[0].evolution_points, ep_before)

    def test_dragon_attendant_heals_one_or_replaces_with_two_when_evolved(self):
        below = self.fresh()
        below.players[0].max_mana = below.players[0].mana = 9
        below.players[0].health = 15
        source = _play(below, self.repository, 10742120)
        below.apply(EndTurn(0))
        self.assertFalse(source.evolved)
        self.assertEqual(below.players[0].health, 16)

        active = self.fresh()
        active.players[0].health = 15
        source = _play(active, self.repository, 10742120)
        active.apply(EndTurn(0))
        self.assertTrue(source.evolved)
        self.assertEqual(active.players[0].health, 17)

    def test_sammy_spellboost_cost_and_asymmetric_draw_order(self):
        engine = self.fresh()
        tracked = _put(engine, self.repository.get(10832110))
        _put(engine, self.repository.get(10512310))
        engine.apply(PlayCard(0, 0))
        self.assertEqual(tracked.current_cost, tracked.definition.cost - 1)

        draw = self.fresh()
        draw.players[0].deck = [_card(3900), _card(3901), _card(3902)]
        draw.players[1].deck = [_card(3910), _card(3911)]
        own_before = len(draw.players[0].hand)
        enemy_before = len(draw.players[1].hand)
        _play(draw, self.repository, 10832110)
        self.assertEqual(len(draw.players[0].hand), own_before + 2)
        self.assertEqual(len(draw.players[1].hand), enemy_before + 1)

    def test_seeded_replay_and_rl_two_target_choice_mask(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=99)
            first = _unit(engine, 1, 4000, life=5)
            second = _unit(engine, 1, 4001, life=5)
            _play(engine, self.repository, 10432110)
            _choose(engine, first.entity_id)
            _choose(engine, second.entity_id)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

        env = ShadowverseEnv(
            [_card(i) for i in range(4100, 4140)],
            [_card(i) for i in range(4200, 4240)],
            class_a=1,
            class_b=1,
            seed=100,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=100)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].max_mana = env.players[0].mana = 10
        _unit(env.core, 1, 4300)
        _unit(env.core, 1, 4301)
        _put(env.core, self.repository.get(10432110))
        env.step(env.PLAY_OFFSET)
        enabled = [
            action for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        ]
        self.assertEqual(enabled, [env.CHOICE_OFFSET, env.CHOICE_OFFSET + 1])


if __name__ == "__main__":
    unittest.main()
