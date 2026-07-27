from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, EndTurn, Evolve
from swb.engine.effects import (
    EffectKind,
    EffectOperation,
    ModifierDuration,
    TargetKind,
)
from swb.engine.events import EventType
from swb.engine.resolution import DamageType, GameEngine
from swb.engine.state import (
    AttackRestriction,
    LeaderDamageModifier,
    StatModifier,
    TargetingRestriction,
    Unit,
)


def card(card_id: int, **overrides) -> CardDefinition:
    values = dict(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=1,
        card_type="随从",
        attack=2,
        life=3,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )
    values.update(overrides)
    return CardDefinition(**values)


def engine(rulebook: RuleBook = RuleBook(())) -> GameEngine:
    result = GameEngine(
        [card(1000 + index) for index in range(40)],
        [card(2000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=7,
        rulebook=rulebook,
        card_resolver=lambda card_id: card(card_id),
    )
    result.reset(seed=7)
    return result


class AbilityRemovalStateTests(unittest.TestCase):
    def test_removes_printed_and_runtime_abilities_without_changing_identity_or_stats(self):
        game = engine()
        unit = Unit.summon(
            card(10, attack=4, life=5, keywords=frozenset({"守护", "屏障", "潜行"})),
            entity_id=game.state.allocate_entity_id(),
        )
        unit.add_keyword("疾驰")
        unit.add_stat_modifier(StatModifier(1, 2, 3, "permanent"))
        unit.add_attack_restriction(AttackRestriction.CANNOT_ATTACK, duration="permanent")
        unit.add_targeting_restriction(
            TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS,
            duration="permanent",
        )
        unit.evolved = True
        identity = (unit.entity_id, unit.definition.card_id, unit.attack, unit.health, unit.max_health)

        unit.remove_all_abilities()

        self.assertEqual(unit.effective_keywords, frozenset())
        self.assertEqual(unit.attack_restrictions, [])
        self.assertEqual(unit.targeting_restrictions, [])
        self.assertEqual(unit.barrier_charges, 0)
        self.assertFalse(unit.ambush_active)
        self.assertTrue(unit.evolved)
        self.assertEqual(
            (unit.entity_id, unit.definition.card_id, unit.attack, unit.health, unit.max_health),
            identity,
        )
        unit.add_keyword("守护")
        self.assertTrue(unit.has_keyword("守护"))

    def test_removed_turn_trigger_and_last_words_do_not_activate(self):
        rules = RuleBook((
            CardRule(10, Trigger.TURN_END, (
                EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 2),
            )),
            CardRule(10, Trigger.LAST_WORDS, (
                EffectOperation(EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, 3),
            )),
        ))
        game = engine(rules)
        unit = Unit.summon(card(10), entity_id=game.state.allocate_entity_id())
        unit.remove_all_abilities()
        game.players[0].board.append(unit)

        game.apply(EndTurn(0))
        self.assertEqual(game.players[1].health, 20)
        unit.health = 0
        game._stabilize()
        self.assertEqual(game.players[1].health, 20)

    def test_transform_and_reentry_restore_printed_abilities(self):
        game = engine()
        unit = Unit.summon(
            card(10, keywords=frozenset({"守护"})),
            entity_id=game.state.allocate_entity_id(),
        )
        unit.remove_all_abilities()
        reentered = Unit.summon(unit.definition, entity_id=game.state.allocate_entity_id())
        transformed = Unit.summon(
            card(11, keywords=frozenset({"疾驰"})),
            entity_id=unit.entity_id,
        )
        self.assertTrue(reentered.has_keyword("守护"))
        self.assertTrue(transformed.has_keyword("疾驰"))

    def test_evolution_keeps_abilities_removed_but_queued_effect_continues(self):
        game = engine()
        source = Unit.summon(
            card(10, keywords=frozenset({"守护"})),
            entity_id=game.state.allocate_entity_id(),
        )
        target = Unit.summon(card(11, life=5), entity_id=game.state.allocate_entity_id())
        game.players[0].board.append(source)
        game.players[1].board.append(target)
        game.players[0].turns_started = game.config.evolution_unlock_turn
        game._start_effects(
            source.definition,
            source.entity_id,
            (EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT, 2),),
        )
        source.remove_all_abilities()
        game.apply(Choose(0, f"entity:{target.entity_id}"))
        game.apply(Evolve(0, source.entity_id))

        self.assertEqual(target.health, 3)
        self.assertTrue(source.evolved)
        self.assertTrue(source.printed_abilities_removed)
        self.assertFalse(source.has_keyword("守护"))


class LeaderDamageModifierTests(unittest.TestCase):
    def test_schema_requires_leader_target_and_integer_amount(self):
        operation = _parse_operation(
            {"kind": "add_leader_damage_modifier", "target": "enemy_leader", "amount": 1},
            "test",
            1,
        )
        self.assertEqual(operation.kind, EffectKind.ADD_LEADER_DAMAGE_MODIFIER)
        with self.assertRaisesRegex(ValueError, "requires a leader target"):
            _parse_operation(
                {"kind": "add_leader_damage_modifier", "target": "enemy_unit", "amount": 1},
                "test",
                1,
            )

    def test_stacking_applies_to_effect_combat_and_self_damage_with_floor_zero(self):
        game = engine()
        target = game.players[1]
        target.leader_damage_modifiers.extend((
            LeaderDamageModifier(1, 2, "permanent"),
            LeaderDamageModifier(2, -1, "permanent"),
        ))
        effect = game.apply_damage(None, None, 3, DamageType.EFFECT, 0, target_player_index=1)
        combat = game.apply_damage(None, None, 2, DamageType.COMBAT, 0, target_player_index=1)
        self_damage = game.apply_damage(None, None, 1, DamageType.EFFECT, 1, target_player_index=1)
        self.assertEqual((effect.actual_amount, combat.actual_amount, self_damage.actual_amount), (4, 3, 2))
        target.leader_damage_modifiers.append(LeaderDamageModifier(3, -20, "permanent"))
        prevented = game.apply_damage(None, None, 5, DamageType.EFFECT, 0, target_player_index=1)
        self.assertEqual(prevented.actual_amount, 0)

    def test_modifier_applies_to_later_damage_in_same_effect_batch(self):
        game = engine()
        source = Unit.summon(card(10), entity_id=game.state.allocate_entity_id())
        game.players[0].board.append(source)
        game._start_effects(
            source.definition,
            source.entity_id,
            (
                EffectOperation(
                    EffectKind.ADD_LEADER_DAMAGE_MODIFIER,
                    TargetKind.ENEMY_LEADER,
                    1,
                ),
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    2,
                ),
            ),
        )
        self.assertEqual(game.players[1].health, 17)

    def test_turn_scoped_modifier_expires_at_its_boundary(self):
        game = engine()
        game.players[1].leader_damage_modifiers.append(
            LeaderDamageModifier(
                1,
                1,
                ModifierDuration.UNTIL_END_OF_TURN.value,
                expires_for_player=0,
            )
        )
        game.apply(EndTurn(0))
        self.assertEqual(game.players[1].leader_damage_modifiers, [])

    def test_source_bound_modifier_stops_on_leave_transform_or_control_change(self):
        for mutation in ("leave", "transform", "control"):
            with self.subTest(mutation=mutation):
                game = engine()
                source = Unit.summon(card(10), entity_id=game.state.allocate_entity_id())
                game.players[0].board.append(source)
                modifier = LeaderDamageModifier(
                    1, 1, ModifierDuration.WHILE_SOURCE_IN_PLAY.value,
                    source_controller=0,
                    source_entity_id=source.entity_id,
                    source_card_id=source.definition.card_id,
                )
                game.players[1].leader_damage_modifiers.append(modifier)
                self.assertEqual(
                    game.apply_damage(None, None, 1, DamageType.EFFECT, 0, target_player_index=1).actual_amount,
                    2,
                )
                if mutation == "leave":
                    game.players[0].board.remove(source)
                elif mutation == "transform":
                    source.definition = card(11)
                else:
                    game.players[0].board.remove(source)
                    game.players[1].board.append(source)
                self.assertEqual(
                    game.apply_damage(None, None, 1, DamageType.EFFECT, 0, target_player_index=1).actual_amount,
                    1,
                )

    def test_modifier_state_is_in_fingerprint_and_event_metadata_is_auditable(self):
        game = engine()
        before = game.deterministic_fingerprint()
        game.players[1].leader_damage_modifiers.append(
            LeaderDamageModifier(1, 1, "permanent")
        )
        self.assertNotEqual(before, game.deterministic_fingerprint())
        game.apply_damage(None, None, 2, DamageType.EFFECT, 0, target_player_index=1)
        event = next(e for e in reversed(game.state.event_queue) if e.type is EventType.DAMAGE_APPLIED)
        self.assertEqual(event.metadata["base_amount"], 2)
        self.assertEqual(event.metadata["modifier_amount"], 1)


if __name__ == "__main__":
    unittest.main()
