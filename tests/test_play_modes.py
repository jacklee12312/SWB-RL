# -*- coding: utf-8 -*-
"""Tests for play modes: enhance, accelerate, crystallize."""

from __future__ import annotations

import os
import sqlite3
import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.play_modes import PlayModeDefinition
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import (
    Amulet,
    DeathCause,
    DestroyedFollowerRecord,
    GraveyardCard,
    HandCard,
    Unit,
)


def _card(cid, **kw):
    return CardDefinition(
        card_id=cid, card_set_id=10000, class_id=1, class_name="\u7cbe\u7075",
        name=kw.get("name", f"c{cid}"), cost=kw.get("cost", 1),
        card_type=kw.get("card_type", "\u968f\u4ece"),
        attack=kw.get("attack", 1), life=kw.get("life", 1),
        keywords=frozenset(), support_level="basic", is_collectible=True,
    )


def _resolver(defs):
    return lambda cid: defs.get(cid)

def _fanfare_rule(cid, *ops):
    return CardRule(card_id=cid, trigger=Trigger.FANFARE, operations=ops)

def _engine(*rules, defs=None, seed=42):
    d = dict(defs) if defs else {}
    return GameEngine(
        deck_a=[_card(i) for i in range(1000, 1040)],
        deck_b=[_card(i) for i in range(2000, 2040)],
        class_a=1, class_b=1, seed=seed,
        rulebook=RuleBook(tuple(rules)),
        card_resolver=_resolver(d),
    )


def _sr(cid, *ops, modes=None):
    return CardRule(card_id=cid, trigger=Trigger.PLAY, operations=ops)


# ---------------------------------------------------------------------------
# Backward compat: old PlayCard works as normal
# ---------------------------------------------------------------------------

class BackwardCompatTests(unittest.TestCase):
    def test_old_playcard_still_works(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].board), 1)
        self.assertEqual(engine.players[0].board[0].origin, CardOrigin.DECK)

    def test_playcard_with_mode_id_normal_works(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0, "normal"))
        self.assertEqual(len(engine.players[0].board), 1)

    def test_unknown_mode_id_raises_illegal(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0, "nonexistent"))


# ---------------------------------------------------------------------------
# Enhance
# ---------------------------------------------------------------------------

class EnhanceTests(unittest.TestCase):
    def setUp(self):
        enhance_mode = PlayModeDefinition(
            mode_id="enhance_5", mode_type="enhance", cost=5,
            operations=(
                EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 5),
                EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 3),
            ),
        )
        normal_ops = (EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 2),)
        self.engine = _engine(
            _fanfare_rule(999801, *normal_ops),
            defs={999801: _card(999801, cost=3)},
        )
        self.engine.reset(seed=42)
        self.engine.rulebook._play_modes = {999801: (enhance_mode,)}
        # Place the enhance card into hand
        hc = HandCard(definition=_card(999801, cost=3), entity_id=999001)
        self.engine.players[0].hand.insert(0, hc)
        self.engine.players[0].hand_entity_ids.insert(0, 999001)
        self.engine.players[0].mana = 10

    def test_normal_enhance_card_deals_2_damage(self):
        opp_health_before = self.engine.players[1].health
        self.engine.apply(PlayCard(0, 0, "normal"))
        self.assertEqual(self.engine.players[1].health, opp_health_before - 2)

    def test_enhance_mode_deals_5_heals_3(self):
        self.engine.players[0].health = 15
        opp_health_before = self.engine.players[1].health
        own_health_before = self.engine.players[0].health
        self.engine.apply(PlayCard(0, 0, "enhance_5"))
        self.assertEqual(self.engine.players[1].health, opp_health_before - 7)
        self.assertEqual(self.engine.players[0].health, own_health_before + 3)

    def test_enhance_mode_costs_5_mana(self):
        mana_before = self.engine.players[0].mana
        self.engine.apply(PlayCard(0, 0, "enhance_5"))
        self.assertEqual(self.engine.players[0].mana, mana_before - 5)

    def test_enhance_unaffordable_raises_illegal(self):
        self.engine.players[0].mana = 3
        with self.assertRaises(IllegalCommand):
            self.engine.apply(PlayCard(0, 0, "enhance_5"))

    def test_both_modes_in_legal_commands(self):
        cmds = self.engine.legal_commands()
        play_cmds = [c for c in cmds if isinstance(c, PlayCard) and c.hand_index == 0]
        mode_ids = {c.mode_id for c in play_cmds}
        self.assertIn("normal", mode_ids)
        self.assertIn("enhance_5", mode_ids)

    def test_only_normal_legal_when_mana_insufficient(self):
        self.engine.players[0].mana = 3
        cmds = self.engine.legal_commands()
        play_cmds = [c for c in cmds if isinstance(c, PlayCard) and c.hand_index == 0]
        mode_ids = {c.mode_id for c in play_cmds}
        self.assertIn("normal", mode_ids)
        self.assertNotIn("enhance_5", mode_ids)

    def test_enhance_keeps_deck_origin(self):
        self.engine.apply(PlayCard(0, 0, "enhance_5"))
        board = self.engine.players[0].board
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0].origin, CardOrigin.DECK)

    def test_enhance_emits_one_card_played(self):
        self.engine.apply(PlayCard(0, 0, "enhance_5"))
        played = [e for e in self.engine.event_history if e.type == EventType.CARD_PLAYED]
        self.assertEqual(len(played), 1)


class SpellEnhanceTests(unittest.TestCase):
    def _spell_engine(
        self,
        base_operations,
        mode_operations,
        *,
        replace_base_operations=False,
    ):
        card_id = 999806
        engine = _engine(
            _sr(card_id, *base_operations),
            defs={
                card_id: _card(
                    card_id,
                    cost=2,
                    card_type="法术",
                    attack=None,
                    life=None,
                )
            },
        )
        engine.reset(seed=42)
        engine.rulebook._play_modes = {
            card_id: (
                PlayModeDefinition(
                    mode_id="enhance_4",
                    mode_type="enhance",
                    cost=4,
                    operations=mode_operations,
                    replace_base_operations=replace_base_operations,
                ),
            )
        }
        definition = _card(
            card_id,
            cost=2,
            card_type="法术",
            attack=None,
            life=None,
        )
        hand_card = HandCard(definition=definition, entity_id=999806)
        engine.players[0].hand.insert(0, hand_card)
        engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
        engine.players[0].mana = 10
        return engine

    def test_spell_enhance_appends_mode_operations_and_enters_graveyard(self):
        engine = self._spell_engine(
            (
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    2,
                ),
            ),
            (
                EffectOperation(
                    EffectKind.HEAL_LEADER,
                    TargetKind.OWN_LEADER,
                    3,
                ),
            ),
        )
        engine.players[0].health = 15

        engine.apply(PlayCard(0, 0, "enhance_4"))

        self.assertEqual(engine.players[1].health, 18)
        self.assertEqual(engine.players[0].health, 18)
        self.assertEqual(engine.players[0].board, [])
        self.assertEqual(
            engine.players[0].graveyard[-1].definition.card_id,
            999806,
        )
        self.assertEqual(engine.players[0].mana, 6)
        played = [
            event
            for event in engine.event_history
            if event.type is EventType.CARD_PLAYED
            and event.metadata.get("card_id") == 999806
        ]
        self.assertEqual(played[-1].metadata["mode_id"], "enhance_4")

    def test_spell_enhance_can_explicitly_replace_base_operations(self):
        engine = self._spell_engine(
            (
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    2,
                ),
            ),
            (
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    5,
                ),
            ),
            replace_base_operations=True,
        )

        engine.apply(PlayCard(0, 0, "enhance_4"))

        self.assertEqual(engine.players[1].health, 15)

    def test_replacement_legality_ignores_replaced_required_target(self):
        engine = self._spell_engine(
            (
                EffectOperation(
                    EffectKind.BANISH,
                    TargetKind.ENEMY_UNIT,
                    requires_target=True,
                ),
            ),
            (
                EffectOperation(
                    EffectKind.DRAW,
                    TargetKind.OWN_LEADER,
                    1,
                ),
            ),
            replace_base_operations=True,
        )
        engine.players[1].board.clear()
        engine.players[0].deck = [_card(999807)]

        legal = engine.legal_commands()

        self.assertNotIn(PlayCard(0, 0, "normal"), legal)
        self.assertIn(PlayCard(0, 0, "enhance_4"), legal)
        engine.apply(PlayCard(0, 0, "enhance_4"))
        self.assertIsNone(engine.state.pending_choice)
        self.assertTrue(any(card.card_id == 999807 for card in engine.players[0].hand))

    def test_follower_enhance_replacement_skips_fanfare(self):
        card_id = 999807
        engine = _engine(
            _fanfare_rule(
                card_id,
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    2,
                ),
            ),
            defs={card_id: _card(card_id, cost=2)},
        )
        engine.reset(seed=42)
        engine.rulebook._play_modes = {
            card_id: (
                PlayModeDefinition(
                    mode_id="enhance_4",
                    mode_type="enhance",
                    cost=4,
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            5,
                        ),
                    ),
                    replace_base_operations=True,
                ),
            )
        }
        hand_card = HandCard(_card(card_id, cost=2), entity_id=999807)
        engine.players[0].hand.insert(0, hand_card)
        engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0, "enhance_4"))

        self.assertEqual(engine.players[1].health, 15)
        self.assertEqual(
            engine.players[0].board[-1].definition.card_id,
            card_id,
        )

    def test_amulet_enhance_replacement_keeps_amulet_route(self):
        card_id = 999808
        definition = _card(
            card_id,
            cost=2,
            card_type="护符",
            attack=None,
            life=None,
        )
        engine = _engine(
            _sr(
                card_id,
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    2,
                ),
            ),
            defs={card_id: definition},
        )
        engine.reset(seed=42)
        engine.rulebook._play_modes = {
            card_id: (
                PlayModeDefinition(
                    mode_id="enhance_4",
                    mode_type="enhance",
                    cost=4,
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            5,
                        ),
                    ),
                    replace_base_operations=True,
                ),
            )
        }
        hand_card = HandCard(definition, entity_id=999808)
        engine.players[0].hand.insert(0, hand_card)
        engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0, "enhance_4"))

        self.assertEqual(engine.players[1].health, 15)
        self.assertIsInstance(engine.players[0].board[-1], Amulet)
        played = [
            event
            for event in engine.event_history
            if event.type is EventType.CARD_PLAYED
            and event.metadata.get("card_id") == card_id
        ]
        self.assertEqual(played[-1].metadata["mode_id"], "enhance_4")


# ---------------------------------------------------------------------------
# Accelerate
# ---------------------------------------------------------------------------

class AccelerateTests(unittest.TestCase):
    def setUp(self):
        accel_mode = PlayModeDefinition(
            mode_id="accelerate_2", mode_type="accelerate", cost=2,
            resulting_card_type="\u6cd5\u672f",
            operations=(
                EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 3),
            ),
        )
        self.engine = _engine(
            _sr(999802),
            defs={999802: _card(999802, cost=5, card_type="\u968f\u4ece")},
        )
        self.engine.reset(seed=42)
        self.engine.rulebook._play_modes = {999802: (accel_mode,)}
        hc = HandCard(definition=_card(999802, cost=5, card_type="\u968f\u4ece"), entity_id=999002)
        self.engine.players[0].hand.insert(0, hc)
        self.engine.players[0].hand_entity_ids.insert(0, 999002)
        self.engine.players[0].mana = 10

    def test_accelerate_does_not_summon_follower(self):
        self.engine.apply(PlayCard(0, 0, "accelerate_2"))
        self.assertEqual(len(self.engine.players[0].board), 0)

    def test_accelerate_deals_damage(self):
        opp_before = self.engine.players[1].health
        self.engine.apply(PlayCard(0, 0, "accelerate_2"))
        self.assertEqual(self.engine.players[1].health, opp_before - 3)

    def test_accelerate_goes_to_graveyard(self):
        self.engine.apply(PlayCard(0, 0, "accelerate_2"))
        gy = [g for g in self.engine.players[0].graveyard
              if g.entity_id == 999002]
        self.assertEqual(len(gy), 1)

    def test_accelerate_preserves_entity_id(self):
        self.engine.apply(PlayCard(0, 0, "accelerate_2"))
        gy = self.engine.players[0].graveyard
        self.assertTrue(any(g.entity_id == 999002 for g in gy))

    def test_accelerate_preserves_origin(self):
        self.engine.apply(PlayCard(0, 0, "accelerate_2"))
        gy = [g for g in self.engine.players[0].graveyard
              if g.entity_id == 999002]
        self.assertEqual(len(gy), 1)
        self.assertEqual(gy[0].origin, CardOrigin.DECK)

    def test_accelerate_triggers_spellboost(self):
        spell = _card(990, card_type="\u6cd5\u672f", cost=3)
        shc = HandCard(definition=spell, entity_id=990099)
        self.engine.players[0].hand.append(shc)
        self.engine.players[0].hand_entity_ids.append(990099)
        self.engine.apply(PlayCard(0, 0, "accelerate_2"))
        boosted = [e for e in self.engine.event_history if e.type == EventType.SPELLBOOSTED]
        self.assertTrue(len(boosted) > 0)

    def test_accelerate_no_follower_summoned_event(self):
        self.engine.apply(PlayCard(0, 0, "accelerate_2"))
        fs_events = [e for e in self.engine.event_history
                     if e.type == EventType.FOLLOWER_SUMMONED]
        self.assertEqual(len(fs_events), 0)


# ---------------------------------------------------------------------------
# Crystallize
# ---------------------------------------------------------------------------

class CrystallizeTests(unittest.TestCase):
    def setUp(self):
        crys_mode = PlayModeDefinition(
            mode_id="crystallize_2", mode_type="crystallize", cost=2,
            resulting_card_type="\u62a4\u7b26", countdown=3,
            operations=(
                EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),
            ),
        )
        self.engine = _engine(
            _sr(999803),
            defs={999803: _card(999803, cost=5, card_type="\u968f\u4ece")},
        )
        self.engine.reset(seed=42)
        self.engine.rulebook._play_modes = {999803: (crys_mode,)}
        hc = HandCard(definition=_card(999803, cost=5, card_type="\u968f\u4ece"), entity_id=999003)
        self.engine.players[0].hand.insert(0, hc)
        self.engine.players[0].hand_entity_ids.insert(0, 999003)
        self.engine.players[0].mana = 10

    def test_crystallize_creates_amulet_not_follower(self):
        self.engine.apply(PlayCard(0, 0, "crystallize_2"))
        board = self.engine.players[0].board
        self.assertEqual(len(board), 1)
        self.assertIsInstance(board[0], Amulet)

    def test_crystallize_preserves_entity_id(self):
        self.engine.apply(PlayCard(0, 0, "crystallize_2"))
        self.assertEqual(self.engine.players[0].board[0].entity_id, 999003)

    def test_crystallize_preserves_origin(self):
        self.engine.apply(PlayCard(0, 0, "crystallize_2"))
        self.assertEqual(self.engine.players[0].board[0].origin, CardOrigin.DECK)

    def test_crystallize_has_countdown(self):
        self.engine.apply(PlayCard(0, 0, "crystallize_2"))
        amulet = self.engine.players[0].board[0]
        self.assertEqual(amulet.countdown, 3)

    def test_crystallize_emits_amulet_entered(self):
        self.engine.apply(PlayCard(0, 0, "crystallize_2"))
        ae = [e for e in self.engine.event_history if e.type == EventType.AMULET_ENTERED]
        self.assertTrue(len(ae) > 0)

    def test_crystallize_no_follower_summoned(self):
        self.engine.apply(PlayCard(0, 0, "crystallize_2"))
        fs = [e for e in self.engine.event_history if e.type == EventType.FOLLOWER_SUMMONED]
        self.assertEqual(len(fs), 0)

    def test_crystallize_draws_card(self):
        deck_before = len(self.engine.players[0].deck)
        self.engine.apply(PlayCard(0, 0, "crystallize_2"))
        self.assertEqual(len(self.engine.players[0].deck), deck_before - 1)

    def test_crystallize_death_preserves_origin(self):
        self.engine.apply(PlayCard(0, 0, "crystallize_2"))
        amulet = self.engine.players[0].board[0]
        amulet.pending_destroy = True
        self.engine.apply(EndTurn(0))
        gy = self.engine.players[0].graveyard
        self.assertTrue(any(g.entity_id == 999003 for g in gy))
        gc = [g for g in gy if g.entity_id == 999003][0]
        self.assertEqual(gc.origin, CardOrigin.DECK)

    def test_crystallize_board_full_rejected(self):
        for i in range(5):
            self.engine.players[0].board.append(Unit.summon(_card(400 + i), entity_id=400 + i))
        with self.assertRaises(IllegalCommand):
            self.engine.apply(PlayCard(0, 0, "crystallize_2"))


# ---------------------------------------------------------------------------
# JSON schema validation
# ---------------------------------------------------------------------------

class SchemaValidationTests(unittest.TestCase):
    def test_old_rules_load_without_error(self):
        rulebook = RuleBook.from_directory("data/rules")
        self.assertIsInstance(rulebook, RuleBook)

    def test_play_modes_demo_loads(self):
        rulebook = RuleBook.from_directory("data/rules")
        modes = rulebook.modes_for(999801)
        self.assertEqual(len(modes), 1)
        self.assertEqual(modes[0].mode_id, "enhance_5")
        self.assertEqual(modes[0].mode_type, "enhance")
        self.assertEqual(modes[0].cost, 5)

    def test_enhance_card_has_normal_and_enhance_modes(self):
        rulebook = RuleBook.from_directory("data/rules")
        modes = rulebook.modes_for(999804)
        self.assertEqual(len(modes), 1)
        self.assertEqual(modes[0].mode_id, "enhance_4")

    def test_accelerate_card_has_resulting_card_type(self):
        rulebook = RuleBook.from_directory("data/rules")
        modes = rulebook.modes_for(999802)
        self.assertEqual(modes[0].resulting_card_type, "\u6cd5\u672f")

    def test_crystallize_card_has_countdown(self):
        rulebook = RuleBook.from_directory("data/rules")
        modes = rulebook.modes_for(999803)
        self.assertEqual(modes[0].countdown, 3)

    def test_duplicate_mode_id_rejected(self):
        import json, tempfile, os
        from swb.engine.card_rules import RuleBook
        payload = {
            "rules": [{
                "card_id": 999999,
                "trigger": "play",
                "operations": [],
                "play_modes": [
                    {"id": "dup", "type": "enhance", "cost": 1},
                    {"id": "dup", "type": "enhance", "cost": 2},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "dup_test.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            with self.assertRaises(ValueError):
                RuleBook.from_directory(tmpdir)

    def test_normal_type_in_play_modes_rejected(self):
        from swb.engine.card_rules import _parse_operation
        from swb.engine.play_modes import validate_play_mode_definition
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "test", "type": "normal", "cost": 3}, "test.json", 1, _parse_operation,
            )

    def test_missing_cost_rejected(self):
        from swb.engine.card_rules import _parse_operation
        from swb.engine.play_modes import validate_play_mode_definition
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "test", "type": "enhance"}, "test.json", 1, _parse_operation,
            )

    def test_negative_cost_rejected(self):
        from swb.engine.card_rules import _parse_operation
        from swb.engine.play_modes import validate_play_mode_definition
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "test", "type": "enhance", "cost": -1}, "test.json", 1, _parse_operation,
            )

    def test_bool_cost_rejected(self):
        from swb.engine.card_rules import _parse_operation
        from swb.engine.play_modes import validate_play_mode_definition
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "test", "type": "enhance", "cost": True}, "test.json", 1, _parse_operation,
            )

    def test_string_cost_rejected(self):
        from swb.engine.card_rules import _parse_operation
        from swb.engine.play_modes import validate_play_mode_definition
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "test", "type": "enhance", "cost": "abc"}, "test.json", 1, _parse_operation,
            )

    def test_unknown_mode_type_rejected(self):
        from swb.engine.card_rules import _parse_operation
        from swb.engine.play_modes import validate_play_mode_definition
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "test", "type": "fakemode", "cost": 3}, "test.json", 1, _parse_operation,
            )

    def test_bad_resulting_card_type_rejected(self):
        from swb.engine.card_rules import _parse_operation
        from swb.engine.play_modes import validate_play_mode_definition
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "test", "type": "accelerate", "cost": 1, "resulting_card_type": "bad"},
                "test.json", 1, _parse_operation,
            )

    def test_resulting_card_type_on_enhance_rejected(self):
        from swb.engine.card_rules import _parse_operation
        from swb.engine.play_modes import validate_play_mode_definition
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "test", "type": "enhance", "cost": 1, "resulting_card_type": "\u6cd5\u672f"},
                "test.json", 1, _parse_operation,
            )


# ---------------------------------------------------------------------------
# CardOrigin and entity_id propagation
# ---------------------------------------------------------------------------

class OriginPropagationTests(unittest.TestCase):
    def test_enhance_follower_death_keeps_origin(self):
        enhance_mode = PlayModeDefinition(
            mode_id="enhance_3", mode_type="enhance", cost=3,
            operations=(),
        )
        engine = _engine(
            _sr(999900),
            defs={999900: _card(999900, cost=2, attack=2, life=2)},
        )
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999900: (enhance_mode,)}
        hc = HandCard(definition=_card(999900, cost=2, attack=2, life=2), entity_id=999900, origin=CardOrigin.TOKEN)
        engine.players[0].hand.insert(0, hc)
        engine.players[0].hand_entity_ids.insert(0, 999900)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0, "enhance_3"))
        unit = engine.players[0].board[0]
        self.assertEqual(unit.origin, CardOrigin.TOKEN)
        board_eid = unit.entity_id
        unit.health = 0
        engine.apply(EndTurn(0))
        gy = [g for g in engine.players[0].graveyard
              if g.definition.card_id == 999900]
        self.assertEqual(len(gy), 1)
        self.assertEqual(gy[0].origin, CardOrigin.TOKEN)

    def test_crystallize_death_keeps_origin_in_graveyard(self):
        crys_mode = PlayModeDefinition(
            mode_id="crystallize_1", mode_type="crystallize", cost=1,
            resulting_card_type="\u62a4\u7b26", countdown=1,
            operations=(),
        )
        engine = _engine(
            _sr(999901),
            defs={999901: _card(999901, cost=5)},
        )
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999901: (crys_mode,)}
        hc = HandCard(definition=_card(999901, cost=5), entity_id=999901, origin=CardOrigin.GENERATED)
        engine.players[0].hand.insert(0, hc)
        engine.players[0].hand_entity_ids.insert(0, 999901)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0, "crystallize_1"))
        self.assertEqual(engine.players[0].board[0].origin, CardOrigin.GENERATED)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class DeterminismTests(unittest.TestCase):
    def test_enhance_same_seed_same_result(self):
        for _ in range(2):
            enhance_mode = PlayModeDefinition(
                mode_id="enhance_5", mode_type="enhance", cost=5,
                operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 5),),
            )
            engine = _engine(seed=42)
            engine.reset(seed=42)
            engine.rulebook._play_modes = {999801: (enhance_mode,)}
            hc = HandCard(definition=_card(999801, cost=3), entity_id=999001)
            engine.players[0].hand.insert(0, hc)
            engine.players[0].hand_entity_ids.insert(0, 999001)
            engine.players[0].mana = 10
            engine.apply(PlayCard(0, 0, "enhance_5"))
            self.assertEqual(engine.players[1].health, 15)


# ---------------------------------------------------------------------------
# RL observation / action mask invariants
# ---------------------------------------------------------------------------

class RLObservationTests(unittest.TestCase):
    def test_observation_dimension_is_fixed(self):
        env = ShadowverseEnv(
            deck_a=[_card(100 + i) for i in range(40)],
            deck_b=[_card(200 + i) for i in range(40)],
            class_a=1, class_b=1, seed=42,
        )
        obs, _ = env.reset(seed=42)
        self.assertEqual(len(obs), ShadowverseEnv.OBSERVATION_V1_SIZE)

    def test_action_size_is_112(self):
        self.assertEqual(ShadowverseEnv.ACTION_SIZE, 112)

    def test_action_mask_matches_legal_commands(self):
        env = ShadowverseEnv(
            deck_a=[_card(100 + i) for i in range(40)],
            deck_b=[_card(200 + i) for i in range(40)],
            class_a=1, class_b=1, seed=42,
        )
        env.reset(seed=42)
        mask = env.action_mask()
        legal = env.core.legal_commands()
        for cmd in legal:
            action = env._encode_command(cmd)
            if action is not None:
                self.assertTrue(mask[action], f"Action {action} for {cmd} not in mask")

    def test_playcard_defaults_to_normal(self):
        cmd = PlayCard(0, 0)
        self.assertEqual(cmd.mode_id, "normal")
        self.assertEqual(cmd.type.value, "play_card")


# ---------------------------------------------------------------------------
# Real card verification
# ---------------------------------------------------------------------------

class RealCardVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(str(db_path)):
            raise unittest.SkipTest("cards.sqlite3 not found")
        cls.db = sqlite3.connect(str(db_path))

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_enhance_card_10001110_exists(self):
        row = self.db.execute(
            "SELECT n.name, c.cost FROM cards c JOIN card_names n ON c.card_id=n.card_id WHERE c.card_id=10001110"
        ).fetchone()
        if row is None:
            self.skipTest("Card 10001110 not found")
        self.assertIn("Indomitable", row[0])
        self.assertEqual(row[1], 2)

    def test_accelerate_card_10671110_exists(self):
        row = self.db.execute(
            "SELECT n.name, c.cost, am.cost FROM alt_modes am JOIN cards c ON am.card_id=c.card_id JOIN card_names n ON am.card_id=n.card_id WHERE am.card_id=10671110 AND am.mode_type='激奏'"
        ).fetchone()
        if row is None:
            self.skipTest("Card 10671110 accelerate not found")
        self.assertEqual(row[1], 6)
        self.assertEqual(row[2], 2)

    def test_crystallize_card_10661110_exists(self):
        row = self.db.execute(
            "SELECT n.name, c.cost, am.cost FROM alt_modes am JOIN cards c ON am.card_id=c.card_id JOIN card_names n ON am.card_id=n.card_id WHERE am.card_id=10661110 AND am.mode_type='结晶'"
        ).fetchone()
        if row is None:
            self.skipTest("Card 10661110 crystallize not found")
        self.assertEqual(row[1], 5)
        self.assertEqual(row[2], 2)


if __name__ == "__main__":
    unittest.main()
