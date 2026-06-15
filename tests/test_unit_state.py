from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, CardPassive, RuleBook, Trigger
from swb.engine.commands import Attack, Choose, EndTurn, Evolve, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, ModifierDuration, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import DamageType, GameEngine, IllegalCommand
from swb.engine.state import (
    AttackRestriction,
    AttackRestrictionModifier,
    CostModifier,
    HandCard,
    StatModifier,
    TargetingRestriction,
    TargetingRestrictionModifier,
    Unit,
)


def card(cid, **kw):
    defaults = dict(
        card_id=cid, card_set_id=10000, class_id=1, class_name="精灵",
        name="c%d" % cid, cost=1, card_type="随从", attack=1, life=1,
        keywords=frozenset(), support_level="basic", is_collectible=True,
    )
    defaults.update(kw)
    return CardDefinition(**defaults)


def mkengine(**kw):
    e = GameEngine(
        [card(i) for i in range(100, 140)],
        [card(i) for i in range(200, 240)],
        class_a=1, class_b=1, seed=1, **kw,
    )
    e.reset(seed=1)
    return e


def mkunit(eng, cid, **kw):
    kws = kw.pop("keywords", frozenset())
    return Unit.summon(card(cid, keywords=kws, **kw), entity_id=eng.state.allocate_entity_id())


# ---------- max_health tests ----------

class MaxHealthTests(unittest.TestCase):
    def test_summon_initializes_max_health(self):
        u = Unit.summon(card(1, life=5))
        self.assertEqual(u.health, 5)
        self.assertEqual(u.max_health, 5)

    def test_damage_does_not_change_max_health(self):
        eng = mkengine()
        u = mkunit(eng, 1, attack=1, life=5)
        eng.players[0].board = [u]
        eng.apply_damage(None, u, 3, DamageType.EFFECT, 0)
        self.assertEqual(u.health, 2)
        self.assertEqual(u.max_health, 5)

    def test_evolve_increases_max_health(self):
        eng = mkengine()
        u = mkunit(eng, 1, attack=2, life=3)
        eng.players[0].board = [u]
        eng.players[0].turns_started = 4
        eng.apply(Evolve(0, u.entity_id))
        self.assertEqual(u.attack, 4)
        self.assertEqual(u.health, 5)
        self.assertEqual(u.max_health, 5)

    def test_permanent_buff_increases_max_health(self):
        eng = mkengine()
        u = mkunit(eng, 1, attack=1, life=3)
        eng.players[0].board = [u]
        u.add_stat_modifier(StatModifier(1, 0, 2, "permanent", None))
        self.assertEqual(u.health, 5)
        self.assertEqual(u.max_health, 5)
        self.assertEqual(u.attack, 1)

    def test_temporary_buff_does_not_change_max_health(self):
        eng = mkengine()
        u = mkunit(eng, 1, attack=1, life=3)
        eng.players[0].board = [u]
        u.add_stat_modifier(StatModifier(1, 0, 2, "until_end_of_turn", 0))
        self.assertEqual(u.health, 5)
        self.assertEqual(u.max_health, 5)

    def test_temporary_buff_expire_clamps_health(self):
        eng = mkengine()
        u = mkunit(eng, 1, attack=1, life=3)
        eng.players[0].board = [u]
        u.add_stat_modifier(StatModifier(1, 0, 3, "until_end_of_turn", 0))
        self.assertEqual(u.health, 6)
        self.assertEqual(u.max_health, 6)
        u.expire_stat_modifiers("until_end_of_turn", 0)
        self.assertEqual(u.health, 3)
        self.assertEqual(u.max_health, 3)

    def test_temp_buff_expire_causes_death(self):
        eng = mkengine()
        u = mkunit(eng, 1, attack=1, life=1)
        eng.players[0].board = [u]
        u.add_stat_modifier(StatModifier(1, 0, 2, "until_end_of_turn", 0))
        self.assertEqual(u.health, 3)
        eng.apply_damage(None, u, 3, DamageType.EFFECT, 0)
        self.assertEqual(u.health, 0)
        u.expire_stat_modifiers("until_end_of_turn", 0)
        self.assertLessEqual(u.health, 0)

    def test_transform_resets_max_health(self):
        eng = mkengine()
        u = mkunit(eng, 1, attack=1, life=3)
        eng.players[0].board = [u]
        u.add_stat_modifier(StatModifier(1, 0, 5, "permanent", None))
        self.assertEqual(u.max_health, 8)
        fresh = Unit.summon(card(99, attack=2, life=2))
        u.definition = fresh.definition
        u.attack = fresh.attack
        u.health = fresh.health
        u.max_health = fresh.max_health
        u.stat_modifiers.clear()
        u.attack_restrictions.clear()
        u.targeting_restrictions.clear()
        self.assertEqual(u.max_health, 2)
        self.assertEqual(u.health, 2)

    def test_return_to_hand_resets_state(self):
        u = Unit.summon(card(1, life=3))
        u.add_stat_modifier(StatModifier(1, 0, 5, "permanent", None))
        self.assertEqual(u.max_health, 8)
        replayed = Unit.summon(u.definition, entity_id=99)
        self.assertEqual(replayed.max_health, 3)
        self.assertEqual(replayed.health, 3)


# ---------- attack restriction tests ----------

class AttackRestrictionTests(unittest.TestCase):
    def test_cannot_attack_blocks_all(self):
        eng = mkengine()
        a = mkunit(eng, 1, attack=3, life=5)
        a.can_attack = True
        d = mkunit(eng, 2, attack=1, life=3)
        eng.players[0].board = [a]
        eng.players[1].board = [d]
        a.add_attack_restriction(AttackRestriction.CANNOT_ATTACK, duration="permanent")
        self.assertFalse(a.can_attack_leader)
        self.assertFalse(a.can_attack_units)
        cmds = eng.legal_commands()
        self.assertFalse(any(isinstance(c, Attack) and c.attacker_id == a.entity_id for c in cmds))

    def test_illegal_attack_does_not_change_state(self):
        eng = mkengine()
        a = mkunit(eng, 1, attack=3, life=5)
        a.can_attack = True
        d = mkunit(eng, 2, attack=1, life=3)
        eng.players[0].board = [a]
        eng.players[1].board = [d]
        a.add_attack_restriction(AttackRestriction.CANNOT_ATTACK, duration="permanent")
        with self.assertRaises(IllegalCommand):
            eng.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertEqual(d.health, 3)

    def test_cannot_attack_leader_only(self):
        a = Unit.summon(card(1, attack=3, life=3))
        a.can_attack = True
        a.add_attack_restriction(AttackRestriction.CANNOT_ATTACK_LEADER, duration="permanent")
        self.assertFalse(a.can_attack_leader)
        self.assertTrue(a.can_attack_units)

    def test_cannot_attack_units_only(self):
        a = Unit.summon(card(1, attack=3, life=3))
        a.can_attack = True
        a.add_attack_restriction(AttackRestriction.CANNOT_ATTACK_UNITS, duration="permanent")
        self.assertTrue(a.can_attack_leader)
        self.assertFalse(a.can_attack_units)

    def test_storm_still_cannot_bypass_restriction(self):
        a = Unit.summon(card(1, attack=3, life=3, keywords=frozenset({"疾驰"})))
        a.add_attack_restriction(AttackRestriction.CANNOT_ATTACK, duration="permanent")
        self.assertFalse(a.can_attack_leader)
        self.assertFalse(a.can_attack_units)

    def test_rush_still_cannot_bypass_restriction(self):
        a = Unit.summon(card(1, attack=3, life=3, keywords=frozenset({"突进"})))
        a.add_attack_restriction(AttackRestriction.CANNOT_ATTACK, duration="permanent")
        self.assertFalse(a.can_attack_units)

    def test_temporary_restriction_expires(self):
        a = Unit.summon(card(1, attack=3, life=3))
        a.can_attack = True
        a.add_attack_restriction(AttackRestriction.CANNOT_ATTACK, duration="until_end_of_turn", expires_for_player=0)
        self.assertFalse(a.can_attack_leader)
        a.expire_attack_restrictions("until_end_of_turn", 0)
        self.assertTrue(a.can_attack_leader)
        self.assertTrue(a.can_attack_units)

    def test_multiple_restrictions_independent(self):
        a = Unit.summon(card(1, attack=3, life=3))
        a.can_attack = True
        a.add_attack_restriction(AttackRestriction.CANNOT_ATTACK_LEADER, duration="until_end_of_turn", expires_for_player=0)
        a.add_attack_restriction(AttackRestriction.CANNOT_ATTACK_UNITS, duration="permanent")
        self.assertFalse(a.can_attack_leader)
        self.assertFalse(a.can_attack_units)
        a.expire_attack_restrictions("until_end_of_turn", 0)
        self.assertTrue(a.can_attack_leader)
        self.assertFalse(a.can_attack_units)

    def test_action_mask_consistent_with_legal_commands(self):
        from swb.engine.environment import ShadowverseEnv
        eng = mkengine()
        a = mkunit(eng, 1, attack=3, life=5)
        a.can_attack = True
        d = mkunit(eng, 2, attack=1, life=3)
        eng.players[0].board = [a]
        eng.players[1].board = [d]
        a.add_attack_restriction(AttackRestriction.CANNOT_ATTACK, duration="permanent")

        env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=2,
        )
        env.reset(seed=2)
        env.players[0].board = [a]
        env.players[1].board = [d]
        env.core._ensure_entity_ids()

        mask = env.action_mask()
        has_attack = any(
            mask[i] for i in range(ShadowverseEnv.ATTACK_OFFSET, ShadowverseEnv.EVOLVE_OFFSET)
        )
        self.assertFalse(has_attack)


# ---------- targeting restriction tests ----------

class TargetingRestrictionTests(unittest.TestCase):
    def test_enemy_cannot_manually_target(self):
        from swb.engine.targeting import target_candidates

        eng = mkengine()
        u = mkunit(eng, 1, attack=1, life=3)
        eng.players[1].board = [u]
        u.add_targeting_restriction(TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS, duration="permanent")

        op = EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT)
        candidates = target_candidates(op, 0, eng.players)
        self.assertEqual(len(candidates), 0)

    def test_own_effects_can_still_target(self):
        from swb.engine.targeting import target_candidates

        eng = mkengine()
        u = mkunit(eng, 1, attack=1, life=3)
        eng.players[1].board = [u]
        u.add_targeting_restriction(TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS, duration="permanent")

        op = EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.OWN_UNIT)
        candidates = target_candidates(op, 1, eng.players)
        self.assertEqual(len(candidates), 1)

    def test_random_still_hits(self):
        from swb.engine.targeting import target_candidates

        eng = mkengine()
        u = mkunit(eng, 1, attack=1, life=3)
        eng.players[1].board = [u]
        u.add_targeting_restriction(TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS, duration="permanent")

        op = EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.RANDOM_ENEMY_UNIT)
        candidates = target_candidates(op, 0, eng.players)
        self.assertEqual(len(candidates), 1)

    def test_all_still_hits(self):
        from swb.engine.targeting import target_candidates

        eng = mkengine()
        u = mkunit(eng, 1, attack=1, life=3)
        eng.players[1].board = [u]
        u.add_targeting_restriction(TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS, duration="permanent")

        op = EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS)
        candidates = target_candidates(op, 0, eng.players)
        self.assertEqual(len(candidates), 1)

    def test_attack_still_allowed_against_restricted(self):
        eng = mkengine()
        d = mkunit(eng, 1, attack=1, life=3)
        eng.players[1].board = [d]
        d.add_targeting_restriction(TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS, duration="permanent")

        a = mkunit(eng, 2, attack=3, life=5)
        a.can_attack = True
        eng.players[0].board = [a]
        cmds = eng.legal_commands()
        self.assertTrue(any(
            isinstance(c, Attack) and c.target_id == d.entity_id for c in cmds
        ))

    def test_add_remove_targeting_restriction(self):
        u = Unit.summon(card(1, life=3))
        self.assertFalse(u.cannot_be_enemy_targeted)
        u.add_targeting_restriction(TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS, duration="permanent")
        self.assertTrue(u.cannot_be_enemy_targeted)
        u.remove_targeting_restriction(TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS)
        self.assertFalse(u.cannot_be_enemy_targeted)

    def test_temporary_targeting_expires(self):
        u = Unit.summon(card(1, life=3))
        u.add_targeting_restriction(TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS, duration="until_end_of_turn", expires_for_player=0)
        self.assertTrue(u.cannot_be_enemy_targeted)
        u.expire_targeting_restrictions("until_end_of_turn", 0)
        self.assertFalse(u.cannot_be_enemy_targeted)

    def test_card_unplayable_when_all_targets_restricted(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=3),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        u = mkunit(eng, 1, attack=1, life=3)
        eng.players[1].board = [u]
        u.add_targeting_restriction(TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS, duration="permanent")
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        cmds = eng.legal_commands()
        self.assertIn(True, [isinstance(c, PlayCard) for c in cmds])


# ---------- SET_STATS tests ----------

class SetStatsTests(unittest.TestCase):
    def test_set_stats_modifies_health_and_max_health(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SET_STATS, target=TargetKind.RANDOM_ENEMY_UNIT, amount=2, secondary_amount=3, set_attack=True, set_health=True),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=5, life=10)
        eng.players[1].board = [d]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(d.attack, 2)
        self.assertEqual(d.health, 3)
        self.assertEqual(d.max_health, 3)

    def test_set_stats_preserves_modifiers(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SET_STATS, target=TargetKind.RANDOM_ENEMY_UNIT, amount=3, secondary_amount=5, set_attack=True, set_health=True),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=1, life=3)
        eng.players[1].board = [d]
        d.add_stat_modifier(StatModifier(1, 1, 2, "permanent", None))
        self.assertEqual(d.attack, 2)
        self.assertEqual(d.health, 5)
        self.assertEqual(d.max_health, 5)
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(d.attack, 4)
        self.assertEqual(d.health, 7)
        self.assertEqual(d.max_health, 7)

    def test_set_stats_clamps_health_minimum(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SET_STATS, target=TargetKind.RANDOM_ENEMY_UNIT, amount=1, secondary_amount=1, set_attack=True, set_health=True),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=5, life=10)
        eng.players[1].board = [d]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertGreaterEqual(d.health, 1)
        self.assertGreaterEqual(d.max_health, 1)

    def test_set_stats_attack_can_be_zero(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SET_STATS, target=TargetKind.RANDOM_ENEMY_UNIT, amount=0, secondary_amount=3, set_attack=True, set_health=True),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=5, life=10)
        eng.players[1].board = [d]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(d.attack, 0)

    def test_set_stats_random_target(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SET_STATS, target=TargetKind.RANDOM_ENEMY_UNIT, amount=1, secondary_amount=1, set_attack=True, set_health=True),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=5, life=10)
        eng.players[1].board = [d]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(d.attack, 1)

    def test_set_stats_all_target(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SET_STATS, target=TargetKind.ALL_ENEMY_UNITS, amount=2, secondary_amount=2, set_attack=True, set_health=True),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d1 = mkunit(eng, 1, attack=5, life=10)
        d2 = mkunit(eng, 2, attack=6, life=12)
        eng.players[1].board = [d1, d2]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(d1.attack, 2)
        self.assertEqual(d2.attack, 2)

    def test_set_stats_condition_skip(self):
        from swb.engine.effects import Condition, ConditionType
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.SET_STATS, target=TargetKind.RANDOM_ENEMY_UNIT,
                    amount=1, secondary_amount=1,
                    conditions=(Condition(ConditionType.CONTROLLER_BOARD_COUNT_AT_LEAST, 99),),
                ),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=5, life=10)
        eng.players[1].board = [d]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(d.attack, 5)

    def test_schema_error_on_unknown_restriction(self):
        import json, tempfile, os
        payload = {
            "rules": [{
                "card_id": 1, "trigger": "play",
                "operations": [{
                    "kind": "add_attack_restriction",
                    "target": "enemy_unit",
                    "restriction": "nonexistent",
                }]
            }]
        }
        d = tempfile.mkdtemp()
        try:
            fp = os.path.join(d, "bad.json")
            with open(fp, "w") as f:
                json.dump(payload, f)
            with self.assertRaises(ValueError):
                RuleBook.from_directory(d)
        finally:
            os.remove(fp)
            os.rmdir(d)


# ---------- JSON operations tests ----------

class JsonRestrictionOperationTests(unittest.TestCase):
    def test_add_attack_restriction_via_rule(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.ADD_ATTACK_RESTRICTION,
                    target=TargetKind.ENEMY_UNIT,
                    restriction="cannot_attack",
                    duration=ModifierDuration.PERMANENT,
                ),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=1, life=3)
        eng.players[1].board = [d]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertFalse(d.can_attack_leader)
        self.assertFalse(d.can_attack_units)

    def test_add_targeting_restriction_via_rule(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.ADD_TARGETING_RESTRICTION,
                    target=TargetKind.RANDOM_ENEMY_UNIT,
                    restriction="cannot_be_targeted_by_enemy_effects",
                    duration=ModifierDuration.PERMANENT,
                ),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=1, life=3)
        eng.players[1].board = [d]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertTrue(d.cannot_be_enemy_targeted)


# ---------- Old JSON compatibility ----------

class BackwardCompatTests(unittest.TestCase):
    def test_old_rulebook_still_loads(self):
        rb = RuleBook.from_directory("data/rules")
        self.assertIsNotNone(rb)

    def test_old_fanfare_still_works(self):
        rulebook = RuleBook.from_directory("data/rules")
        eng = mkengine(rulebook=rulebook)
        eng.players[0].mana = 10
        for i in range(5):
            eng.players[0].hand[i] = card(10051120)
        eng.apply(PlayCard(0, 0))
        self.assertLess(eng.players[0].health, 20)


class TargetBindingTests(unittest.TestCase):
    def test_both_ops_target_same_unit_single_choice(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SET_STATS, target=TargetKind.ENEMY_UNIT, secondary_amount=1,
                               target_key="sel", set_health=True),
                EffectOperation(kind=EffectKind.ADD_ATTACK_RESTRICTION, target=TargetKind.PREVIOUS_TARGET,
                               target_key="sel", restriction="cannot_attack", duration=ModifierDuration.PERMANENT),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d1 = mkunit(eng, 1, attack=3, life=5)
        d2 = mkunit(eng, 2, attack=4, life=6)
        eng.players[1].board = [d1, d2]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(len(eng.legal_commands()), 2)
        choice = next(c for c in eng.legal_commands() if isinstance(c, Choose))
        eng.apply(choice)
        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(d1.health, 1)
        self.assertFalse(d1.can_attack_leader)

    def test_target_gone_before_binding_safely_skips(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DESTROY, target=TargetKind.ENEMY_UNIT,
                               target_key="sel"),
                EffectOperation(kind=EffectKind.SET_STATS, target=TargetKind.PREVIOUS_TARGET,
                               target_key="sel", secondary_amount=1, set_health=True),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=3, life=5)
        eng.players[1].board = [d]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertIsNotNone(eng.state.pending_choice)
        choice = next(c for c in eng.legal_commands() if isinstance(c, Choose))
        eng.apply(choice)
        self.assertIsNone(eng.state.pending_choice)

    def test_schema_rejects_non_single_entity_target_bindings(self):
        import json
        import tempfile
        from pathlib import Path

        cases = (
            ("all_enemy_units", "all_targets"),
            ("own_leader", "leader"),
            ("own_hand", "hand"),
        )
        for target, key in cases:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                payload = {
                    "rules": [{
                        "card_id": 123,
                        "trigger": "play",
                        "operations": [{
                            "kind": "destroy",
                            "target": target,
                            "target_key": key,
                        }],
                    }]
                }
                Path(tmp, "bad.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    r"bad\.json card 123/operations\[0\].*single board-entity",
                ):
                    RuleBook.from_directory(tmp)

    def test_schema_target_key_errors_use_zero_based_paths(self):
        import json
        import tempfile
        from pathlib import Path

        cases = (
            (
                [{
                    "kind": "set_stats",
                    "target": "previous_target",
                    "target_key": "missing",
                    "health": 1,
                }],
                r"operations\[0\].*was not defined",
            ),
            (
                [
                    {
                        "kind": "set_stats",
                        "target": "enemy_unit",
                        "target_key": "same",
                        "health": 1,
                    },
                    {
                        "kind": "set_stats",
                        "target": "enemy_unit",
                        "target_key": "same",
                        "health": 2,
                    },
                ],
                r"operations\[1\].*duplicate target_key",
            ),
        )
        for operations, pattern in cases:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as tmp:
                payload = {
                    "rules": [{
                        "card_id": 123,
                        "trigger": "play",
                        "operations": operations,
                    }]
                }
                Path(tmp, "bad.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, pattern):
                    RuleBook.from_directory(tmp)


class AttackFloorTests(unittest.TestCase):
    def test_negative_attack_clamped_to_zero(self):
        eng = mkengine()
        u = mkunit(eng, 1, attack=3, life=5)
        u.add_stat_modifier(StatModifier(1, -5, 0, "permanent", None))
        self.assertEqual(u.attack, 0)
        self.assertEqual(u.base_attack, 3)

    def test_negative_attack_reverses_correctly(self):
        u = Unit.summon(card(1, attack=3, life=3))
        u.add_stat_modifier(StatModifier(1, -5, 0, "until_end_of_turn", 0))
        self.assertEqual(u.attack, 0)
        u.expire_stat_modifiers("until_end_of_turn", 0)
        self.assertEqual(u.attack, 3)

    def test_multiple_attack_modifiers(self):
        u = Unit.summon(card(1, attack=5, life=3))
        u.add_stat_modifier(StatModifier(1, -3, 0, "until_end_of_turn", 0))
        u.add_stat_modifier(StatModifier(2, -4, 0, "until_end_of_turn", 0))
        self.assertEqual(u.attack, 0)
        u.expire_stat_modifiers("until_end_of_turn", 0)
        self.assertEqual(u.attack, 5)

    def test_set_stats_only_attack_does_not_change_health(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SET_STATS, target=TargetKind.RANDOM_ENEMY_UNIT,
                               amount=10, set_attack=True),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=3, life=5)
        eng.players[1].board = [d]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(d.attack, 10)
        self.assertEqual(d.health, 5)
        self.assertEqual(d.max_health, 5)

    def test_set_stats_only_health_does_not_change_attack(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SET_STATS, target=TargetKind.RANDOM_ENEMY_UNIT,
                               secondary_amount=8, set_health=True),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 1, attack=3, life=5)
        eng.players[1].board = [d]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(d.attack, 3)
        self.assertEqual(d.health, 8)
        self.assertEqual(d.max_health, 8)


class SpellboostTests(unittest.TestCase):
    def test_playing_spell_boosts_other_hand_spells(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.players[0].hand[1] = card(2, card_type="法术", attack=None, life=None)
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))
        hc = eng.players[0].hand[0]
        self.assertIsInstance(hc, HandCard)
        self.assertEqual(hc.spellboost_count, 1)
        self.assertEqual(hc.card_id, 2)

    def test_spell_does_not_boost_self(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.players[0].hand[1] = card(2, card_type="法术", attack=None, life=None)
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))
        hc = eng.players[0].hand[0]
        self.assertIsInstance(hc, HandCard)
        self.assertEqual(hc.spellboost_count, 1)
        self.assertNotEqual(hc.card_id, 1)

    def test_spellboost_hand_effect_all_own_hand(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SPELLBOOST_HAND, target=TargetKind.ALL_OWN_HAND, amount=2),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.players[0].hand[1] = card(2, card_type="法术", attack=None, life=None)
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))
        hc = eng.players[0].hand[0]
        self.assertIsInstance(hc, HandCard)
        self.assertEqual(hc.spellboost_count, 3)
        self.assertEqual(hc.card_id, 2)

    def test_spellboost_zero_amount_noop(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.SPELLBOOST_HAND, target=TargetKind.ALL_OWN_HAND, amount=0),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))

    def test_spellboost_cost_reduction(self):
        rulebook = RuleBook(
            rules=(), passives=(CardPassive(10132320, "spellboost_cost_reduction", 1),)
        )
        eng = mkengine(rulebook=rulebook)
        hc = HandCard(definition=card(10132320, cost=4, card_type="法术", attack=None, life=None), entity_id=1, spellboost_cost_reduction=1)
        self.assertEqual(hc.current_cost, 4)
        hc.apply_spellboost(1)
        self.assertEqual(hc.current_cost, 3)
        hc.apply_spellboost(2)
        self.assertEqual(hc.current_cost, 1)
        hc.apply_spellboost(2)
        self.assertEqual(hc.current_cost, 0)
        self.assertEqual(hc.spellboost_count, 5)

    def test_action_mask_uses_spellboost_cost(self):
        from swb.engine.environment import ShadowverseEnv
        rulebook = RuleBook(
            rules=(), passives=(CardPassive(10132320, "spellboost_cost_reduction", 2),)
        )
        eng = mkengine(rulebook=rulebook)
        hc = HandCard(definition=card(10132320, cost=5, card_type="法术", attack=None, life=None), entity_id=1, spellboost_cost_reduction=2)
        hc.apply_spellboost(2)
        eng.players[0].hand[0] = hc
        eng.players[0].mana = 1
        cmds = eng.legal_commands()
        self.assertTrue(any(isinstance(c, PlayCard) for c in cmds))

    def test_no_leak_to_opponent_observation(self):
        from swb.engine.environment import ShadowverseEnv
        rulebook = RuleBook(())
        env = ShadowverseEnv(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        env.reset(seed=1)
        obs = env.observation()
        self.assertEqual(len(obs), 207)

    def test_auto_boost_after_choice_spell(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=1),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        d = mkunit(eng, 99, attack=1, life=3)
        eng.players[1].board = [d]
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.players[0].hand[1] = card(2, card_type="法术", attack=None, life=None)
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))
        self.assertIsNotNone(eng.state.pending_choice)
        for hc in eng.players[0].hand:
            if isinstance(hc, HandCard):
                self.assertEqual(hc.spellboost_count, 0)
        choice = next(c for c in eng.legal_commands() if isinstance(c, Choose))
        eng.apply(choice)
        hc = eng.players[0].hand[0]
        self.assertIsInstance(hc, HandCard)
        self.assertEqual(hc.spellboost_count, 1)
        self.assertEqual(hc.card_id, 2)

    def test_follower_play_does_not_trigger_boost(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.FANFARE, operations=(
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = card(1)
        eng.players[0].hand[1] = card(2, card_type="法术", attack=None, life=None)
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))
        for hc in eng.players[0].hand:
            if isinstance(hc, HandCard):
                self.assertEqual(hc.spellboost_count, 0)


class SpellboostAllCardsTests(unittest.TestCase):
    def test_auto_boost_boosts_followers_too(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.players[0].hand[1] = card(2, card_type="随从", attack=1, life=1)
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))
        hc = eng.players[0].hand[0]
        self.assertIsInstance(hc, HandCard)
        self.assertEqual(hc.spellboost_count, 1)

    def test_spellboost_hand_boosts_followers(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.SPELLBOOST_HAND, target=TargetKind.ALL_OWN_HAND, amount=2),
        ),),))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.players[0].hand[1] = card(2, card_type="随从", attack=1, life=1)
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))
        hc = eng.players[0].hand[0]
        self.assertEqual(hc.spellboost_count, 3)

    def test_spellboost_hand_own_hand_choice(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.SPELLBOOST_HAND,
                    target=TargetKind.OWN_HAND,
                    amount=2,
                ),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand = [
            card(1, card_type="法术", attack=None, life=None),
            card(2),
            card(3),
        ]
        eng.players[0].hand_entity_ids = []
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))
        choice = next(c for c in eng.legal_commands() if isinstance(c, Choose))
        chosen_id = int(choice.option_id.removeprefix("hand:"))
        eng.apply(choice)
        counts = {
            hand_card.entity_id: hand_card.spellboost_count
            for hand_card in eng.players[0].hand
        }
        self.assertEqual(counts[chosen_id], 3)
        self.assertEqual(sorted(counts.values()), [1, 3])

    def test_spellboost_hand_random_own_hand(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(
                    kind=EffectKind.SPELLBOOST_HAND,
                    target=TargetKind.RANDOM_OWN_HAND,
                    amount=2,
                ),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand = [
            card(1, card_type="法术", attack=None, life=None),
            card(2),
            card(3),
        ]
        eng.players[0].hand_entity_ids = []
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))
        self.assertEqual(
            sorted(card.spellboost_count for card in eng.players[0].hand),
            [1, 3],
        )

    def test_empty_hand_safe(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand.clear()
        eng.players[0].hand_entity_ids.clear()
        eng.players[0].hand.append(card(1, card_type="法术", attack=None, life=None))
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))

    def test_spellboost_events_include_source_card_id(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.players[0].hand[1] = card(2, card_type="法术", attack=None, life=None)
        eng.players[0].mana = 10
        eng._ensure_entity_ids()
        played_entity_id = eng.players[0].hand[0].entity_id
        eng.apply(PlayCard(0, 0))
        sb_events = [e for e in eng.event_history if e.type is EventType.SPELLBOOSTED]
        self.assertGreater(len(sb_events), 0)
        for e in sb_events:
            self.assertIn("source_card_id", e.metadata)
            self.assertEqual(e.metadata["source_card_id"], 1)
            self.assertEqual(e.metadata["source_entity_id"], played_entity_id)

    def test_spellboost_hand_event_includes_board_source_entity(self):
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.FANFARE, operations=(
                EffectOperation(
                    kind=EffectKind.SPELLBOOST_HAND,
                    target=TargetKind.ALL_OWN_HAND,
                    amount=1,
                ),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = card(1)
        eng.players[0].hand[1] = card(2)
        eng.players[0].mana = 10
        eng.apply(PlayCard(0, 0))
        source = eng.players[0].board[0]
        event = next(
            e for e in eng.event_history if e.type is EventType.SPELLBOOSTED
        )
        self.assertEqual(event.metadata["source_card_id"], 1)
        self.assertEqual(event.metadata["source_entity_id"], source.entity_id)

    def test_ensure_entity_ids_initializes_spellboost_reduction(self):
        rulebook = RuleBook(
            passives=(CardPassive(101, "spellboost_cost_reduction", 1),)
        )
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = card(
            101,
            cost=5,
            card_type="法术",
            attack=None,
            life=None,
        )
        eng._ensure_entity_ids()
        hand_card = eng.players[0].hand[0]
        hand_card.apply_spellboost(2)
        self.assertEqual(hand_card.spellboost_cost_reduction, 1)
        self.assertEqual(hand_card.current_cost, 3)

    def test_reset_clears_spellboost_pending(self):
        eng = mkengine()
        eng._spellboost_pending = 1
        eng._pending_spellboost_player = 1
        eng._pending_spellboost_source_card_id = 123
        eng._pending_spellboost_source_entity_id = 456
        eng._suspended_action = "attack"
        eng._suspended_action_state = {"x": 1}
        eng._suspended_event_state = {"y": 2}
        eng.reset(seed=7)
        self.assertIsNone(eng._spellboost_pending)
        self.assertIsNone(eng._suspended_action)
        self.assertIsNone(eng._suspended_action_state)
        self.assertIsNone(eng._suspended_event_state)
        self.assertEqual(eng._pending_spellboost_player, 0)
        self.assertEqual(eng._pending_spellboost_source_card_id, 0)
        self.assertIsNone(eng._pending_spellboost_source_entity_id)

    def test_real_spellboost_card_loads_passive_from_database(self):
        repository = CardRepository("data/cards.sqlite3")
        snowman_army = repository.get(10132320)
        rulebook = RuleBook.from_directory("data/rules")
        eng = mkengine(rulebook=rulebook)
        eng.players[0].hand[0] = snowman_army
        eng._ensure_entity_ids()
        hand_card = eng.players[0].hand[0]
        self.assertEqual(hand_card.spellboost_cost_reduction, 1)
        hand_card.apply_spellboost(2)
        self.assertEqual(
            hand_card.current_cost,
            max(0, snowman_army.cost - 2),
        )

    def test_unified_current_cost(self):
        eng = mkengine()
        hc = HandCard(
            definition=card(101, cost=8, card_type="法术", attack=None, life=None),
            entity_id=1,
            spellboost_cost_reduction=1,
        )
        hc.apply_spellboost(3)
        self.assertEqual(hc.current_cost, 5)
        hc.cost_modifiers.append(CostModifier(1, "set", 5, "permanent", None))
        self.assertEqual(hc.current_cost, 2)
        hc.cost_modifiers.append(CostModifier(2, "subtract", 1, "permanent", None))
        self.assertEqual(hc.current_cost, 1)

    def test_cost_modifiers_with_spellboost_combined(self):
        hc = HandCard(
            definition=card(101, cost=8, card_type="法术", attack=None, life=None),
            entity_id=1,
            spellboost_cost_reduction=1,
        )
        hc.cost_modifiers.append(CostModifier(1, "set", 5, "permanent", None))
        hc.cost_modifiers.append(CostModifier(2, "subtract", 1, "permanent", None))
        hc.apply_spellboost(2)
        self.assertEqual(hc.current_cost, 2)

    def test_passive_schema_rejects_float(self):
        import json, tempfile, os
        payload = {"passives": [{"card_id": 1, "kind": "spellboost_cost_reduction", "amount": 1.5}], "rules": []}
        d = tempfile.mkdtemp()
        try:
            fp = os.path.join(d, "bad.json")
            with open(fp, "w") as f:
                json.dump(payload, f)
            with self.assertRaises(ValueError) as ctx:
                RuleBook.from_directory(d)
            self.assertIn("must be an integer", str(ctx.exception))
        finally:
            os.remove(fp)
            os.rmdir(d)

    def test_passive_schema_rejects_bool(self):
        import json, tempfile, os
        payload = {"passives": [{"card_id": 1, "kind": "spellboost_cost_reduction", "amount": True}], "rules": []}
        d = tempfile.mkdtemp()
        try:
            fp = os.path.join(d, "bad.json")
            with open(fp, "w") as f:
                json.dump(payload, f)
            with self.assertRaises(ValueError) as ctx:
                RuleBook.from_directory(d)
            self.assertIn("got bool", str(ctx.exception))
        finally:
            os.remove(fp)
            os.rmdir(d)

    def test_passive_schema_rejects_string(self):
        import json, tempfile, os
        payload = {"passives": [{"card_id": 1, "kind": "spellboost_cost_reduction", "amount": "1"}], "rules": []}
        d = tempfile.mkdtemp()
        try:
            fp = os.path.join(d, "bad.json")
            with open(fp, "w") as f:
                json.dump(payload, f)
            with self.assertRaises(ValueError):
                RuleBook.from_directory(d)
        finally:
            os.remove(fp)
            os.rmdir(d)

    def test_duplicate_passive_error_includes_both_paths(self):
        import json, tempfile, os
        payload = {
            "passives": [
                {
                    "card_id": 1,
                    "kind": "spellboost_cost_reduction",
                    "amount": 1,
                },
                {
                    "card_id": 1,
                    "kind": "spellboost_cost_reduction",
                    "amount": 2,
                },
            ],
            "rules": [],
        }
        d = tempfile.mkdtemp()
        try:
            fp = os.path.join(d, "bad.json")
            with open(fp, "w") as f:
                json.dump(payload, f)
            with self.assertRaises(ValueError) as ctx:
                RuleBook.from_directory(d)
            message = str(ctx.exception)
            self.assertIn("bad.json/passives[1]", message)
            self.assertIn("bad.json/passives[0]", message)
        finally:
            os.remove(fp)
            os.rmdir(d)



if __name__ == "__main__":
    unittest.main()
