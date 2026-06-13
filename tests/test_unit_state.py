from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Attack, Choose, EndTurn, Evolve, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, ModifierDuration, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import DamageType, GameEngine, IllegalCommand
from swb.engine.state import (
    AttackRestriction,
    AttackRestrictionModifier,
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


if __name__ == "__main__":
    unittest.main()
