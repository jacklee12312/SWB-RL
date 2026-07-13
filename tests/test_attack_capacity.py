from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Attack, EndTurn, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, ModifierDuration, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import Unit


def card(
    card_id: int,
    *,
    attack: int = 1,
    life: int = 4,
    keywords: frozenset[str] = frozenset(),
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=1,
        card_type="随从",
        attack=attack,
        life=life,
        keywords=keywords,
        support_level="basic",
        is_collectible=True,
    )


def engine(
    rulebook: RuleBook | None = None,
    *,
    card_resolver=None,
) -> GameEngine:
    game = GameEngine(
        [card(1000 + index) for index in range(40)],
        [card(2000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=17,
        rulebook=rulebook or RuleBook(()),
        card_resolver=card_resolver,
    )
    game.reset(seed=17)
    return game


class AttackCapacityStateTests(unittest.TestCase):
    def test_grant_preserves_attacks_used_and_does_not_stack(self):
        unit = Unit.summon(card(1), entity_id=1)
        unit.summoned_this_turn = False
        unit.can_attack = True
        unit.consume_attack()
        self.assertEqual(unit.attacks_remaining, 0)

        unit.grant_attacks_per_turn(2)
        self.assertEqual(unit.attacks_per_turn, 2)
        self.assertEqual(unit.attacks_remaining, 1)
        self.assertTrue(unit.can_attack)
        unit.grant_attacks_per_turn(2)
        self.assertEqual(len(unit.attack_capacity_modifiers), 1)
        self.assertEqual(unit.attacks_remaining, 1)

    def test_grant_does_not_bypass_summoning_sickness(self):
        unit = Unit.summon(card(1), entity_id=1)
        self.assertFalse(unit.can_attack)
        unit.grant_attacks_per_turn(2)
        self.assertEqual(unit.attacks_remaining, 2)
        self.assertFalse(unit.can_attack)

    def test_temporary_capacity_and_ability_removal_preserve_attacks_used(self):
        unit = Unit.summon(card(1), entity_id=1)
        unit.summoned_this_turn = False
        unit.can_attack = True
        unit.grant_attacks_per_turn(
            2,
            duration=ModifierDuration.UNTIL_END_OF_TURN.value,
            expires_for_player=0,
        )
        unit.consume_attack()
        unit.expire_attack_capacity(
            ModifierDuration.UNTIL_END_OF_TURN.value,
            0,
        )
        self.assertEqual(unit.attacks_per_turn, 1)
        self.assertEqual(unit.attacks_remaining, 0)
        self.assertFalse(unit.can_attack)

        fresh = Unit.summon(card(2), entity_id=2)
        fresh.summoned_this_turn = False
        fresh.can_attack = True
        fresh.grant_attacks_per_turn(2)
        fresh.consume_attack()
        fresh.remove_all_abilities()
        self.assertEqual(fresh.attacks_per_turn, 1)
        self.assertEqual(fresh.attacks_remaining, 0)


class AttackCapacityEngineTests(unittest.TestCase):
    def test_effect_grants_two_attacks_and_emits_auditable_event(self):
        rulebook = RuleBook((CardRule(
            card_id=900,
            trigger=Trigger.SUPER_EVOLVE,
            operations=(EffectOperation(
                kind=EffectKind.GRANT_ATTACKS_PER_TURN,
                target=TargetKind.SELF,
                amount=2,
            ),),
        ),))
        game = engine(rulebook)
        source = Unit.summon(card(900, attack=2), entity_id=game.state.allocate_entity_id())
        source.summoned_this_turn = False
        source.can_attack = True
        game.players[0].board = [source]
        game.players[0].turns_started = (
            game.config.first_player_super_evolution_unlock_turn
        )

        game.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(source.attacks_per_turn, 2)
        self.assertEqual(source.attacks_remaining, 2)
        game.apply(Attack(0, source.entity_id, None))
        self.assertTrue(source.can_attack)
        game.apply(Attack(0, source.entity_id, None))
        self.assertFalse(source.can_attack)
        before = game.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            game.apply(Attack(0, source.entity_id, None))
        self.assertEqual(game.deterministic_fingerprint(), before)
        self.assertTrue(any(
            event.type is EventType.FOLLOWER_ATTACK_CAPACITY_GRANTED
            and event.target_id == source.entity_id
            and event.amount == 2
            for event in game.event_history
        ))
        game.assert_invariants()

    def test_real_10162120_clash_and_super_evolve_rule_is_complete(self):
        game = engine(RuleBook.from_directory("data/rules"))
        source = Unit.summon(
            card(10162120, attack=2, life=6),
            entity_id=game.state.allocate_entity_id(),
        )
        defender = Unit.summon(
            card(901, attack=1, life=12),
            entity_id=game.state.allocate_entity_id(),
        )
        source.summoned_this_turn = False
        source.can_attack = True
        game.players[0].board = [source]
        game.players[1].board = [defender]
        game.players[0].turns_started = (
            game.config.first_player_super_evolution_unlock_turn
        )

        game.apply(SuperEvolve(0, source.entity_id))
        enemy_health = game.players[1].health
        game.apply(Attack(0, source.entity_id, defender.entity_id))
        self.assertEqual(game.players[1].health, enemy_health - 1)
        self.assertEqual(source.attacks_remaining, 1)
        self.assertTrue(source.can_attack)
        game.apply(Attack(0, source.entity_id, defender.entity_id))
        self.assertEqual(game.players[1].health, enemy_health - 2)
        self.assertEqual(source.attacks_remaining, 0)

    def test_turn_start_refreshes_granted_capacity(self):
        game = engine()
        source = Unit.summon(card(900), entity_id=game.state.allocate_entity_id())
        source.summoned_this_turn = False
        source.grant_attacks_per_turn(2)
        source.attacks_remaining = 0
        source.can_attack = False
        game.players[0].board = [source]

        game.apply(EndTurn(0))
        game.apply(EndTurn(1))
        self.assertEqual(source.attacks_remaining, 2)
        self.assertTrue(source.can_attack)

    def test_rush_only_is_preserved_between_attacks(self):
        game = engine()
        source = Unit.summon(
            card(900, attack=2, keywords=frozenset({"突进"})),
            entity_id=game.state.allocate_entity_id(),
        )
        source.grant_attacks_per_turn(2)
        defenders = [
            Unit.summon(card(901 + index, life=6), entity_id=game.state.allocate_entity_id())
            for index in range(2)
        ]
        game.players[0].board = [source]
        game.players[1].board = defenders

        game.apply(Attack(0, source.entity_id, defenders[0].entity_id))
        self.assertTrue(source.can_attack)
        self.assertTrue(source.rush_only)
        self.assertFalse(source.can_attack_leader)
        game.apply(Attack(0, source.entity_id, defenders[1].entity_id))
        self.assertEqual(source.attacks_remaining, 0)

    def test_transform_clears_capacity_without_resetting_attack_usage(self):
        replacement = card(901, attack=3)
        rulebook = RuleBook((CardRule(
            card_id=900,
            trigger=Trigger.ATTACK,
            operations=(EffectOperation(
                kind=EffectKind.TRANSFORM,
                target=TargetKind.SELF,
                card_id=replacement.card_id,
            ),),
        ),))
        game = engine(
            rulebook,
            card_resolver=lambda card_id: (
                replacement if card_id == replacement.card_id else None
            ),
        )
        source = Unit.summon(card(900, attack=2), entity_id=game.state.allocate_entity_id())
        source.summoned_this_turn = False
        source.can_attack = True
        source.grant_attacks_per_turn(2)
        game.players[0].board = [source]

        game.apply(Attack(0, source.entity_id, None))
        self.assertEqual(source.definition.card_id, replacement.card_id)
        self.assertEqual(source.attacks_per_turn, 1)
        self.assertEqual(source.attacks_remaining, 0)
        self.assertFalse(source.can_attack)

    def test_rl_mask_keeps_attacker_legal_after_first_attack(self):
        rulebook = RuleBook((CardRule(
            card_id=900,
            trigger=Trigger.SUPER_EVOLVE,
            operations=(EffectOperation(
                kind=EffectKind.GRANT_ATTACKS_PER_TURN,
                target=TargetKind.SELF,
                amount=2,
            ),),
        ),))
        env = ShadowverseEnv(
            [card(1000 + index) for index in range(40)],
            [card(2000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=17,
            rulebook=rulebook,
        )
        env.reset(seed=17)
        source = Unit.summon(card(900, attack=2), entity_id=env.core.state.allocate_entity_id())
        source.summoned_this_turn = False
        source.can_attack = True
        source.grant_attacks_per_turn(2)
        env.players[0].board = [source]

        leader_action = env.ATTACK_OFFSET
        self.assertEqual(env.action_mask()[leader_action], 1)
        env.step(leader_action)
        self.assertEqual(env.action_mask()[leader_action], 1)
        env.step(leader_action)
        self.assertEqual(env.action_mask()[leader_action], 0)


class AttackCapacitySchemaTests(unittest.TestCase):
    def test_schema_rejects_invalid_amount_and_target(self):
        for operation in (
            {"kind": "grant_attacks_per_turn", "target": "self", "amount": 0},
            {
                "kind": "grant_attacks_per_turn",
                "target": "own_leader",
                "amount": 2,
            },
        ):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                Path(tmp, "rules.json").write_text(json.dumps({
                    "rules": [{
                        "card_id": 900,
                        "trigger": "fanfare",
                        "operations": [operation],
                    }],
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "grant_attacks_per_turn"):
                    RuleBook.from_directory(tmp)


if __name__ == "__main__":
    unittest.main()
