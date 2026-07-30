# -*- coding: utf-8 -*-
"""Audit tests for play mode architecture fixes."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import Condition, ConditionType, EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.play_modes import (
    MAX_SPECIAL_MODES_PER_CARD,
    PlayModeDefinition,
    validate_play_mode_definition,
)
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import (
    Amulet,
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


def _engine(*rules, defs=None, seed=42):
    d = dict(defs) if defs else {}
    return GameEngine(
        deck_a=[_card(i) for i in range(1000, 1040)],
        deck_b=[_card(i) for i in range(2000, 2040)],
        class_a=1, class_b=1, seed=seed,
        rulebook=RuleBook(tuple(rules)),
        card_resolver=_resolver(d),
    )


def _insert_card(engine, card_def, origin=CardOrigin.DECK, source_origin=None):
    hc = HandCard(
        definition=card_def,
        entity_id=engine.state.allocate_entity_id(),
        origin=origin,
        source_origin=source_origin,
    )
    engine.players[0].hand.insert(0, hc)
    engine.players[0].hand_entity_ids.insert(0, hc.entity_id)
    return hc


# ---------------------------------------------------------------------------
# 1. Unified mode legality check
# ---------------------------------------------------------------------------

class ModeLegalityTests(unittest.TestCase):
    def test_special_mode_no_legal_target_not_in_legal_commands(self):
        accel_mode = PlayModeDefinition(
            mode_id="accel_no_target", mode_type="accelerate", cost=1,
            operations=(EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 3),),
        )
        engine = _engine()
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999801: (accel_mode,)}
        _insert_card(engine, _card(999801, cost=5))
        engine.players[0].mana = 1
        cmds = engine.legal_commands()
        play_cmds = [c for c in cmds if isinstance(c, PlayCard) and c.hand_index == 0]
        mode_ids = {c.mode_id for c in play_cmds}
        self.assertNotIn("accel_no_target", mode_ids)

    def test_special_mode_no_target_apply_raises_illegal(self):
        accel_mode = PlayModeDefinition(
            mode_id="accel_no_target", mode_type="accelerate", cost=1,
            operations=(EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 3),),
        )
        engine = _engine()
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999801: (accel_mode,)}
        _insert_card(engine, _card(999801, cost=5))
        engine.players[0].mana = 1
        snap_mana = engine.players[0].mana
        snap_hand = len(engine.players[0].hand)
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0, "accel_no_target"))
        self.assertEqual(engine.players[0].mana, snap_mana)
        self.assertEqual(len(engine.players[0].hand), snap_hand)

    def test_mode_conditions_false_not_playable(self):
        enhance_mode = PlayModeDefinition(
            mode_id="enhance_cond", mode_type="enhance", cost=1,
            operations=(),
            conditions=(Condition(type=ConditionType.CONTROLLER_COOPERATION_AT_LEAST, value=10),),
        )
        engine = _engine()
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999801: (enhance_mode,)}
        _insert_card(engine, _card(999801, cost=3))
        engine.players[0].mana = 10
        cmds = engine.legal_commands()
        play_cmds = [c for c in cmds if isinstance(c, PlayCard) and c.hand_index == 0]
        mode_ids = {c.mode_id for c in play_cmds}
        self.assertNotIn("enhance_cond", mode_ids)

    def test_mode_conditions_false_apply_raises(self):
        enhance_mode = PlayModeDefinition(
            mode_id="enhance_cond", mode_type="enhance", cost=1,
            operations=(),
            conditions=(Condition(type=ConditionType.CONTROLLER_COOPERATION_AT_LEAST, value=10),),
        )
        engine = _engine()
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999801: (enhance_mode,)}
        _insert_card(engine, _card(999801, cost=3))
        engine.players[0].mana = 10
        snap_mana = engine.players[0].mana
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0, "enhance_cond"))
        self.assertEqual(engine.players[0].mana, snap_mana)

    def test_mode_conditions_true_playable(self):
        enhance_mode = PlayModeDefinition(
            mode_id="enhance_cond", mode_type="enhance", cost=1,
            operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),),
            conditions=(Condition(type=ConditionType.CONTROLLER_COOPERATION_AT_LEAST, value=5),),
        )
        engine = _engine()
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999801: (enhance_mode,)}
        _insert_card(engine, _card(999801, cost=3))
        engine.players[0].mana = 10
        engine.players[0].cooperation = 5
        cmds = engine.legal_commands()
        play_cmds = [c for c in cmds if isinstance(c, PlayCard) and c.hand_index == 0]
        mode_ids = {c.mode_id for c in play_cmds}
        self.assertIn("enhance_cond", mode_ids)

    def test_target_dependent_mode_condition_is_defensively_unplayable(self):
        enhance_mode = PlayModeDefinition(
            mode_id="bad_target_condition",
            mode_type="enhance",
            cost=1,
            conditions=(
                Condition(type=ConditionType.TARGET_HEALTH_AT_MOST, value=2),
            ),
        )
        engine = _engine()
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999801: (enhance_mode,)}
        _insert_card(engine, _card(999801, cost=3))
        engine.players[0].mana = 10
        play_modes = {
            command.mode_id
            for command in engine.legal_commands()
            if isinstance(command, PlayCard) and command.hand_index == 0
        }
        self.assertNotIn("bad_target_condition", play_modes)


# ---------------------------------------------------------------------------
# 2. Condition evaluation does not consume RNG
# ---------------------------------------------------------------------------

class RNGSafetyTests(unittest.TestCase):
    def test_condition_check_does_not_consume_rng(self):
        engine = _engine(seed=42)
        engine.reset(seed=42)
        rng_before = engine.random.getstate()
        engine.legal_commands()
        rng_after = engine.random.getstate()
        self.assertEqual(rng_before, rng_after)


# ---------------------------------------------------------------------------
# 3. Schema validation
# ---------------------------------------------------------------------------

class SchemaValidationTests(unittest.TestCase):
    def test_schema_rejects_choose_mode_type(self):
        with self.assertRaises(ValueError) as cm:
            validate_play_mode_definition(
                {"id": "choose_1", "type": "choose", "cost": 2},
                "test.json", 1, _parse_operation,
            )
        self.assertIn("choose", str(cm.exception).lower())

    def test_schema_rejects_enhance_with_countdown(self):
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "bad", "type": "enhance", "cost": 1, "countdown": 3},
                "test.json", 1, _parse_operation,
            )

    def test_schema_rejects_enhance_resulting_card_type(self):
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "bad", "type": "enhance", "cost": 1, "resulting_card_type": "\u6cd5\u672f"},
                "test.json", 1, _parse_operation,
            )

    def test_schema_rejects_accelerate_bad_card_type(self):
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "bad", "type": "accelerate", "cost": 1, "resulting_card_type": "\u968f\u4ece"},
                "test.json", 1, _parse_operation,
            )

    def test_schema_rejects_accelerate_with_countdown(self):
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "bad", "type": "accelerate", "cost": 1, "countdown": 1},
                "test.json", 1, _parse_operation,
            )

    def test_schema_rejects_crystallize_bad_card_type(self):
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "bad", "type": "crystallize", "cost": 1, "resulting_card_type": "\u6cd5\u672f"},
                "test.json", 1, _parse_operation,
            )

    def test_schema_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "bad", "type": "enhance", "cost": 1, "fakery": 5},
                "test.json", 1, _parse_operation,
            )

    def test_schema_rejects_countdown_string(self):
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "bad", "type": "crystallize", "cost": 1, "countdown": "abc"},
                "test.json", 1, _parse_operation,
            )

    def test_schema_rejects_empty_mode_id(self):
        with self.assertRaises(ValueError):
            validate_play_mode_definition(
                {"id": "", "type": "enhance", "cost": 1},
                "test.json", 1, _parse_operation,
            )

    def test_max_modes_exceeded_rejected(self):
        payload = {"rules": [{"card_id": 999999, "trigger": "play", "operations": [],
            "play_modes": [
                {"id": f"m{i}", "type": "enhance", "cost": i + 1}
                for i in range(MAX_SPECIAL_MODES_PER_CARD + 1)
            ]}]}
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "test.json")
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            with self.assertRaises(ValueError):
                RuleBook.from_directory(d)

    def test_schema_rejects_target_dependent_mode_condition(self):
        with self.assertRaises(ValueError) as cm:
            validate_play_mode_definition(
                {
                    "id": "bad_condition",
                    "type": "enhance",
                    "cost": 3,
                    "conditions": [
                        {"type": "target_health_at_most", "value": 2}
                    ],
                },
                "test.json",
                1,
                _parse_operation,
            )
        self.assertIn("conditions", str(cm.exception))
        self.assertIn("target_health_at_most", str(cm.exception))

    def test_schema_accepts_explicit_enhance_operation_replacement(self):
        from swb.engine.card_rules import _parse_operation
        from swb.engine.play_modes import validate_play_mode_definition

        mode = validate_play_mode_definition(
            {
                "id": "enhance_4",
                "type": "enhance",
                "cost": 4,
                "replace_base_operations": True,
                "operations": [
                    {
                        "kind": "draw",
                        "target": "own_leader",
                        "amount": 1,
                    }
                ],
            },
            "test.json",
            1,
            _parse_operation,
        )

        self.assertTrue(mode.replace_base_operations)

    def test_schema_rejects_invalid_operation_replacement_policy(self):
        from swb.engine.card_rules import _parse_operation
        from swb.engine.play_modes import validate_play_mode_definition

        cases = (
            ("enhance", "yes", "must be boolean"),
            ("accelerate", True, "only valid for enhance"),
        )
        for mode_type, value, message in cases:
            with self.subTest(mode_type=mode_type, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    validate_play_mode_definition(
                        {
                            "id": f"{mode_type}_4",
                            "type": mode_type,
                            "cost": 4,
                            "replace_base_operations": value,
                            "operations": [],
                        },
                        "test.json",
                        1,
                        _parse_operation,
                    )


# ---------------------------------------------------------------------------
# 4. RuleBook manual construction also validates
# ---------------------------------------------------------------------------

class RuleBookValidationTests(unittest.TestCase):
    def test_manual_choose_mode_rejected(self):
        m = PlayModeDefinition(mode_id="m1", mode_type="choose", cost=1)
        with self.assertRaises(ValueError):
            RuleBook(rules=(), play_modes={1: (m,)})

    def test_manual_max_modes_rejected(self):
        modes = tuple(PlayModeDefinition(mode_id=f"m{i}", mode_type="enhance", cost=i + 1)
                      for i in range(MAX_SPECIAL_MODES_PER_CARD + 1))
        with self.assertRaises(ValueError):
            RuleBook(rules=(), play_modes={1: modes})

    def test_manual_duplicate_mode_id_rejected(self):
        modes = (
            PlayModeDefinition(mode_id="dup", mode_type="enhance", cost=1),
            PlayModeDefinition(mode_id="dup", mode_type="enhance", cost=2),
        )
        with self.assertRaises(ValueError):
            RuleBook(rules=(), play_modes={1: modes})

    def test_manual_unknown_mode_type_rejected(self):
        mode = PlayModeDefinition(mode_id="bad", mode_type="bogus", cost=1)
        with self.assertRaises(ValueError):
            RuleBook(rules=(), play_modes={1: (mode,)})

    def test_manual_replacement_policy_is_enhance_only(self):
        mode = PlayModeDefinition(
            mode_id="bad",
            mode_type="accelerate",
            cost=1,
            replace_base_operations=True,
        )
        with self.assertRaisesRegex(ValueError, "only valid for enhance"):
            RuleBook(rules=(), play_modes={1: (mode,)})

    def test_manual_accelerate_wrong_result_type_rejected(self):
        mode = PlayModeDefinition(
            mode_id="bad",
            mode_type="accelerate",
            cost=1,
            resulting_card_type="护符",
        )
        with self.assertRaises(ValueError):
            RuleBook(rules=(), play_modes={1: (mode,)})

    def test_manual_crystallize_wrong_result_type_rejected(self):
        mode = PlayModeDefinition(
            mode_id="bad",
            mode_type="crystallize",
            cost=1,
            resulting_card_type="法术",
        )
        with self.assertRaises(ValueError):
            RuleBook(rules=(), play_modes={1: (mode,)})

    def test_manual_target_dependent_mode_condition_rejected(self):
        mode = PlayModeDefinition(
            mode_id="bad",
            mode_type="enhance",
            cost=1,
            conditions=(
                Condition(type=ConditionType.TARGET_HEALTH_AT_MOST, value=2),
            ),
        )
        with self.assertRaises(ValueError):
            RuleBook(rules=(), play_modes={1: (mode,)})


# ---------------------------------------------------------------------------
# 5. Choose mode runtime rejection
# ---------------------------------------------------------------------------

class ChooseRejectionTests(unittest.TestCase):
    def test_choose_mode_not_in_legal_commands(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10
        cmds = engine.legal_commands()
        for c in cmds:
            if isinstance(c, PlayCard):
                self.assertNotEqual(c.mode_id, "choose_anything")

    def test_choose_mode_apply_does_not_mutate(self):
        engine = _engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(999801, cost=3))
        engine.players[0].mana = 10
        snap_mana = engine.players[0].mana
        snap_hand = len(engine.players[0].hand)
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0, "choose_anything"))
        self.assertEqual(engine.players[0].mana, snap_mana)
        self.assertEqual(len(engine.players[0].hand), snap_hand)


# ---------------------------------------------------------------------------
# 6. RL encoding round-trip
# ---------------------------------------------------------------------------

class RLEncodingTests(unittest.TestCase):
    def setUp(self):
        self.env = ShadowverseEnv(
            deck_a=[_card(100 + i) for i in range(40)],
            deck_b=[_card(200 + i) for i in range(40)],
            class_a=1, class_b=1, seed=42,
        )
        self.env.reset(seed=42)

    def test_every_legal_command_has_action(self):
        for cmd in self.env.core.legal_commands():
            action = self.env._encode_command(cmd)
            self.assertIsNotNone(action, f"No action for {cmd}")
            mask = self.env.action_mask()
            self.assertTrue(mask[action], f"Action {action} not in mask for {cmd}")

    def test_normal_play_round_trip(self):
        self.env.core.players[0].mana = 10
        cmd = PlayCard(0, 0, "normal")
        encoded = self.env._encode_command(cmd)
        self.assertIsNotNone(encoded)
        decoded = self.env._decode_action(encoded)
        self.assertEqual(decoded, cmd)

    def test_end_turn_round_trip(self):
        cmd = EndTurn(0)
        encoded = self.env._encode_command(cmd)
        self.assertEqual(encoded, 0)
        decoded = self.env._decode_action(encoded)
        self.assertEqual(decoded, cmd)


# ---------------------------------------------------------------------------
# 7. Enhance/Crystallize source_origin and semantics
# ---------------------------------------------------------------------------

class OriginSemanticsTests(unittest.TestCase):
    def test_enhance_preserves_source_origin(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10
        enhance_mode = PlayModeDefinition(
            mode_id="enhance_3", mode_type="enhance", cost=3, operations=(),
        )
        engine.rulebook._play_modes = {999800: (enhance_mode,)}
        _insert_card(engine, _card(999800, cost=2, attack=2, life=2),
                     source_origin=CardOrigin.REANIMATED)
        engine.apply(PlayCard(0, 0, "enhance_3"))
        self.assertEqual(engine.players[0].board[0].source_origin, CardOrigin.REANIMATED)

    def test_enhance_runs_fanfare_then_mode_ops(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10
        enhance_mode = PlayModeDefinition(
            mode_id="enhance_1", mode_type="enhance", cost=1,
            operations=(EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 1),),
        )
        engine.rulebook._play_modes = {999800: (enhance_mode,)}
        _insert_card(engine, _card(999800, cost=2, attack=2, life=2))
        opp_before = engine.players[1].health
        engine.apply(PlayCard(0, 0, "enhance_1"))
        self.assertEqual(engine.players[1].health, opp_before - 1)

    def test_enhance_choice_fanfare_and_mode_share_one_continuation(self):
        card_id = 999804
        fanfare = CardRule(
            card_id,
            Trigger.FANFARE,
            (
                EffectOperation(
                    EffectKind.DAMAGE_UNIT,
                    TargetKind.ENEMY_UNIT,
                    2,
                ),
            ),
        )
        engine = _engine(fanfare)
        engine.reset(seed=42)
        engine.players[0].mana = 10
        enhance_mode = PlayModeDefinition(
            mode_id="enhance_3",
            mode_type="enhance",
            cost=3,
            operations=(
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    1,
                ),
            ),
        )
        engine.rulebook._play_modes = {card_id: (enhance_mode,)}
        target = Unit.summon(
            _card(900, attack=1, life=5),
            entity_id=900,
        )
        engine.players[1].board.append(target)
        _insert_card(engine, _card(card_id, cost=2, attack=2, life=2))

        engine.apply(PlayCard(0, 0, "enhance_3"))

        self.assertIsNotNone(engine.state.pending_choice)
        self.assertEqual(len(engine.state.effect_stack), 1)
        leader_before = engine.players[1].health
        choose = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        engine.apply(choose)

        self.assertEqual(target.health, 3)
        self.assertEqual(engine.players[1].health, leader_before - 1)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.effect_stack, [])

    def test_crystallize_preserves_source_origin(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 1
        crys_mode = PlayModeDefinition(
            mode_id="crystal", mode_type="crystallize", cost=1,
            countdown=3, operations=(),
        )
        engine.rulebook._play_modes = {999800: (crys_mode,)}
        _insert_card(engine, _card(999800, cost=5, attack=2, life=2),
                     source_origin=CardOrigin.GENERATED)
        engine.apply(PlayCard(0, 0, "crystal"))
        self.assertEqual(engine.players[0].board[0].source_origin, CardOrigin.GENERATED)

    def test_crystallize_amulet_not_damage_target(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 1
        crys_mode = PlayModeDefinition(
            mode_id="crystal", mode_type="crystallize", cost=1,
            countdown=3, operations=(),
        )
        engine.rulebook._play_modes = {999800: (crys_mode,)}
        _insert_card(engine, _card(999800, cost=5, attack=2, life=2))
        engine.apply(PlayCard(0, 0, "crystal"))
        from swb.engine.targeting import target_candidates
        candidates = target_candidates(
            EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ANY_BOARD, 1),
            engine.current_player, engine.players,
        )
        amulet_eid = engine.players[0].board[0].entity_id
        self.assertNotIn(amulet_eid, [e.entity_id for e in candidates])

    def test_crystallize_amulet_is_amulet_target(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 1
        crys_mode = PlayModeDefinition(
            mode_id="crystal", mode_type="crystallize", cost=1,
            countdown=3, operations=(),
        )
        engine.rulebook._play_modes = {999800: (crys_mode,)}
        _insert_card(engine, _card(999800, cost=5, attack=2, life=2))
        engine.apply(PlayCard(0, 0, "crystal"))
        from swb.engine.targeting import target_candidates
        candidates = target_candidates(
            EffectOperation(EffectKind.DESTROY, TargetKind.OWN_AMULET, 1),
            engine.current_player, engine.players,
        )
        amulet_eid = engine.players[0].board[0].entity_id
        self.assertIn(amulet_eid, [e.entity_id for e in candidates])

    def test_crystallize_death_no_destroyed_follower(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 1
        crys_mode = PlayModeDefinition(
            mode_id="crystal", mode_type="crystallize", cost=1,
            countdown=1, operations=(),
        )
        engine.rulebook._play_modes = {999800: (crys_mode,)}
        _insert_card(engine, _card(999800, cost=5, attack=2, life=2))
        engine.apply(PlayCard(0, 0, "crystal"))
        df_before = len(engine.state.destroyed_followers)
        self.assertIsInstance(engine.players[0].board[0], Amulet)
        engine.apply(EndTurn(engine.current_player))
        engine.apply(EndTurn(engine.current_player))
        self.assertEqual(len(engine.state.destroyed_followers), df_before)


# ---------------------------------------------------------------------------
# 8. Accelerate continuation
# ---------------------------------------------------------------------------

class AccelerateContinuationTests(unittest.TestCase):
    def test_accelerate_choice_pauses_and_resumes(self):
        accel = PlayModeDefinition(
            mode_id="accel_1", mode_type="accelerate", cost=1,
            operations=(EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 1),),
        )
        engine = _engine()
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999802: (accel,)}
        engine.players[1].board.append(Unit.summon(_card(900), entity_id=900))
        _insert_card(engine, _card(999802, cost=5, attack=2, life=2))
        engine.players[0].mana = 1
        engine.apply(PlayCard(0, 0, "accel_1"))
        self.assertIsNotNone(engine.state.pending_choice)
        cmds = engine.legal_commands()
        choose_cmds = [c for c in cmds if isinstance(c, Choose)]
        self.assertTrue(len(choose_cmds) > 0)
        engine.apply(choose_cmds[0])
        gy = [g for g in engine.players[0].graveyard
              if g.definition.card_id == 999802]
        self.assertEqual(len(gy), 1)

    def test_accelerate_target_leaves_play_safe(self):
        accel = PlayModeDefinition(
            mode_id="accel_1", mode_type="accelerate", cost=1,
            operations=(EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 1),),
        )
        engine = _engine()
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999802: (accel,)}
        engine.players[1].board.append(Unit.summon(_card(900), entity_id=900))
        _insert_card(engine, _card(999802, cost=5, attack=2, life=2))
        engine.players[0].mana = 1
        engine.apply(PlayCard(0, 0, "accel_1"))
        engine.players[1].board.clear()
        cmds = engine.legal_commands()
        choose_cmds = [c for c in cmds if isinstance(c, Choose)]
        engine.apply(choose_cmds[0])
        gy = [g for g in engine.players[0].graveyard
              if g.definition.card_id == 999802]
        self.assertEqual(len(gy), 1)

    def test_accelerate_origin_preserved_after_choice(self):
        accel = PlayModeDefinition(
            mode_id="accel_1", mode_type="accelerate", cost=1,
            operations=(EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 1),),
        )
        engine = _engine()
        engine.reset(seed=42)
        engine.rulebook._play_modes = {999802: (accel,)}
        engine.players[1].board.append(Unit.summon(_card(900), entity_id=900))
        _insert_card(engine, _card(999802, cost=5, attack=2, life=2),
                     origin=CardOrigin.TOKEN)
        engine.players[0].mana = 1
        engine.apply(PlayCard(0, 0, "accel_1"))
        cmds = engine.legal_commands()
        choose_cmds = [c for c in cmds if isinstance(c, Choose)]
        engine.apply(choose_cmds[0])
        gy = [g for g in engine.players[0].graveyard
              if g.definition.card_id == 999802]
        self.assertEqual(gy[0].origin, CardOrigin.TOKEN)


# ---------------------------------------------------------------------------
# 9. Old backward compat
# ---------------------------------------------------------------------------

class BackwardCompatTests(unittest.TestCase):
    def test_old_json_rules_load(self):
        rb = RuleBook.from_directory("data/rules")
        self.assertIsInstance(rb, RuleBook)

    def test_old_play_card_works(self):
        engine = _engine()
        engine.reset(seed=42)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))


# ---------------------------------------------------------------------------
# 10. Real card end-to-end
# ---------------------------------------------------------------------------

class RealCardEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(str(cls.db_path)):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_10001110_name_cost_type(self):
        from swb.db.repository import CardRepository
        repo = CardRepository(str(self.db_path))
        try:
            card = repo.get(10001110)
        except KeyError:
            self.skipTest("Card 10001110 not found")
        self.assertEqual(card.cost, 2)
        self.assertEqual(card.card_type, "\u968f\u4ece")
        self.assertIn("\u5251", card.name)

    def test_10001110_enhance_from_rules(self):
        from swb.db.repository import CardRepository
        repo = CardRepository(str(self.db_path))
        try:
            repo.get(10001110)
        except KeyError:
            self.skipTest("Card 10001110 not found")
        rulebook = RuleBook.from_directory("data/rules")
        modes = rulebook.modes_for(10001110)
        self.assertEqual(len(modes), 1)
        engine = GameEngine(
            deck_a=[_card(i) for i in range(1000, 1040)],
            deck_b=[_card(i) for i in range(2000, 2040)],
            class_a=1, class_b=1, seed=42, rulebook=rulebook,
        )
        engine.reset(seed=42)
        engine.players[0].mana = 10
        _insert_card(engine, _card(10001110, cost=2, attack=2, life=2))
        engine.apply(PlayCard(0, 0, "enhance_4"))
        unit = engine.players[0].board[0]
        self.assertEqual(unit.attack, 5)
        self.assertEqual(unit.health, 5)

    def test_10671110_name_cost(self):
        from swb.db.repository import CardRepository
        repo = CardRepository(str(self.db_path))
        try:
            card = repo.get(10671110)
        except KeyError:
            self.skipTest("Card 10671110 not found")
        self.assertEqual(card.cost, 6)
        self.assertEqual(card.card_type, "\u968f\u4ece")

    def test_10671110_accelerate_from_rules(self):
        from swb.db.repository import CardRepository
        repo = CardRepository(str(self.db_path))
        try:
            real_card = repo.get(10671110)
        except KeyError:
            self.skipTest("Card 10671110 not found")
        rulebook = RuleBook.from_directory("data/rules")
        modes = rulebook.modes_for(10671110)
        self.assertEqual([m.mode_id for m in modes], ["accelerate_2"])
        accelerate = modes[0]
        self.assertEqual(len(accelerate.operations), 1)
        self.assertEqual(accelerate.operations[0].kind, EffectKind.SUMMON)
        self.assertEqual(accelerate.operations[0].card_id, 10671110)
        engine = GameEngine(
            deck_a=[_card(i) for i in range(1000, 1040)],
            deck_b=[_card(i) for i in range(2000, 2040)],
            class_a=1, class_b=1, seed=42, rulebook=rulebook,
            card_resolver=repo.get,
        )
        engine.reset(seed=42)
        engine.players[0].mana = 2
        _insert_card(engine, real_card)
        deck_before = len(engine.players[0].deck)
        engine.apply(PlayCard(0, 0, "accelerate_2"))
        self.assertEqual(len(engine.players[0].board), 1)
        self.assertEqual(
            engine.players[0].board[0].definition.card_id,
            10671110,
        )
        self.assertEqual(len(engine.players[0].deck), deck_before)
        gy = [g for g in engine.players[0].graveyard
              if g.definition.card_id == 10671110]
        self.assertEqual(len(gy), 1)

    def test_10661110_crystallize_is_hidden_when_body_is_affordable(self):
        from swb.db.repository import CardRepository

        repo = CardRepository(str(self.db_path))
        try:
            real_card = repo.get(10661110)
        except KeyError:
            self.skipTest("Card 10661110 not found")
        rulebook = RuleBook.from_directory("data/rules")
        self.assertEqual(
            [mode.mode_id for mode in rulebook.modes_for(10661110)],
            ["crystallize_2"],
        )
        env = ShadowverseEnv(
            deck_a=[_card(i) for i in range(1000, 1040)],
            deck_b=[_card(i) for i in range(2000, 2040)],
            class_a=1,
            class_b=1,
            seed=10661110,
            rulebook=rulebook,
            card_resolver=repo.get,
        )
        env.reset(seed=10661110)
        _insert_card(env.core, real_card)
        env.core.players[0].mana = real_card.cost

        normal = PlayCard(0, 0, "normal")
        crystallize = PlayCard(0, 0, "crystallize_2")
        legal = env.core.legal_commands()
        mask = env.action_mask()
        self.assertIn(normal, legal)
        self.assertNotIn(crystallize, legal)
        self.assertTrue(mask[env.PLAY_OFFSET])
        self.assertFalse(
            any(
                mask[env.MODE_PLAY_OFFSET + slot]
                for slot in range(MAX_SPECIAL_MODES_PER_CARD)
            )
        )

        fingerprint = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(crystallize)
        self.assertEqual(env.core.deterministic_fingerprint(), fingerprint)

    def test_10424110_enhance_replaces_normal_command_and_action(self):
        from swb.db.repository import CardRepository

        repo = CardRepository(str(self.db_path))
        try:
            real_card = repo.get(10424110)
        except KeyError:
            self.skipTest("Card 10424110 not found")
        rulebook = RuleBook.from_directory("data/rules")
        self.assertEqual(
            [mode.mode_id for mode in rulebook.modes_for(10424110)],
            ["enhance_6"],
        )
        env = ShadowverseEnv(
            deck_a=[_card(i) for i in range(1000, 1040)],
            deck_b=[_card(i) for i in range(2000, 2040)],
            class_a=1,
            class_b=1,
            seed=10424110,
            rulebook=rulebook,
            card_resolver=repo.get,
        )
        env.reset(seed=10424110)
        _insert_card(env.core, real_card)
        env.core.players[0].mana = 10

        normal = PlayCard(0, 0, "normal")
        enhance = PlayCard(0, 0, "enhance_6")
        legal = env.core.legal_commands()
        mask = env.action_mask()
        self.assertNotIn(normal, legal)
        self.assertIn(enhance, legal)
        self.assertFalse(mask[env._encode_command(normal)])
        self.assertTrue(mask[env._encode_command(enhance)])

        fingerprint = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(normal)
        self.assertEqual(env.core.deterministic_fingerprint(), fingerprint)


if __name__ == "__main__":
    unittest.main()
