from __future__ import annotations

import os
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import Attack, Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.emblem import EmblemDefinition, EmblemTriggerRule
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import (
    DamageType,
    GameConfig,
    GameEngine,
    IllegalCommand,
    SuperEvolutionAttackContext,
)
from swb.engine.state import HandCard, Unit
from swb.engine.union_burst import UnionBurstDefinition, UnionBurstKind


def _card(card_id: int, **overrides) -> CardDefinition:
    values = {
        "card_id": card_id,
        "card_set_id": 10000,
        "class_id": 1,
        "class_name": "精灵",
        "name": f"card-{card_id}",
        "cost": 1,
        "card_type": "随从",
        "attack": 2,
        "life": 3,
        "keywords": frozenset(),
        "support_level": "basic",
        "is_collectible": True,
    }
    values.update(overrides)
    return CardDefinition(**values)


def _engine(rulebook: RuleBook | None = None, *, seed: int = 42) -> GameEngine:
    engine = GameEngine(
        [_card(1000 + index) for index in range(40)],
        [_card(2000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook or RuleBook(),
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    return engine


def _place(
    engine: GameEngine,
    player_index: int,
    definition: CardDefinition,
) -> Unit:
    unit = Unit.summon(
        definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[player_index].board.append(unit)
    return unit


def _insert_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int = 0,
    evolutions: int = 0,
) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
        evolutions_while_in_hand=evolutions,
    )
    player = engine.players[player_index]
    player.hand.append(hand_card)
    player.hand_entity_ids.append(hand_card.entity_id)
    return hand_card


def _play_last_hand_card(engine: GameEngine) -> None:
    player = engine.players[0]
    player.max_mana = 10
    player.mana = 10
    engine.apply(PlayCard(0, len(player.hand) - 1))


def _mark_super_evolved(engine: GameEngine, unit: Unit) -> None:
    unit.evolved = True
    unit.super_evolved = True
    unit.super_evolved_turn = engine.turn


class SuperEvolutionSchemaTests(unittest.TestCase):
    def test_schema_accepts_follower_targets_and_rejects_leaders(self):
        operation = _parse_operation(
            {"kind": "super_evolve_unit", "target": "self"},
            "test/operations[0]",
            100,
        )
        self.assertIs(operation.kind, EffectKind.SUPER_EVOLVE_UNIT)
        self.assertIs(operation.target, TargetKind.SELF)

        with self.assertRaisesRegex(ValueError, "follower target"):
            _parse_operation(
                {"kind": "super_evolve_unit", "target": "own_leader"},
                "test/operations[0]",
                100,
            )

    def test_evolved_followers_are_not_effect_candidates(self):
        spell = _card(100, card_type="法术", attack=None, life=None)
        rulebook = RuleBook(
            rules=(
                CardRule(
                    100,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.SUPER_EVOLVE_UNIT,
                            TargetKind.OWN_UNIT,
                            requires_target=True,
                        ),
                    ),
                ),
            )
        )
        engine = _engine(rulebook)
        unevolved = _place(engine, 0, _card(200))
        evolved = _place(engine, 0, _card(201))
        evolved.evolved = True
        _insert_hand(engine, spell)

        _play_last_hand_card(engine)

        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        self.assertEqual(
            [option.entity_id for option in request.options],
            [unevolved.entity_id],
        )


class ManualSuperEvolutionTests(unittest.TestCase):
    def test_manual_super_evolution_adds_three_three_and_spends_sep(self):
        engine = _engine()
        unit = _place(engine, 0, _card(100, attack=2, life=3))
        engine.players[0].turns_started = (
            engine.config.first_player_super_evolution_unlock_turn
        )

        engine.apply(SuperEvolve(0, unit.entity_id))

        self.assertEqual((unit.attack, unit.health, unit.max_health), (5, 6, 6))
        self.assertEqual(engine.players[0].super_evolution_points, 1)
        self.assertTrue(engine.players[0].evolved_this_turn)
        self.assertTrue(engine.players[0].super_evolved_this_turn)

    def test_manual_super_evolution_fires_evolve_then_super_evolve(self):
        source = _card(
            100,
            keywords=frozenset({"进化时", "超进化时"}),
        )
        rulebook = RuleBook(
            rules=(
                CardRule(
                    100,
                    Trigger.EVOLVE,
                    (
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            1,
                        ),
                    ),
                ),
                CardRule(
                    100,
                    Trigger.SUPER_EVOLVE,
                    (
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            2,
                        ),
                    ),
                ),
            )
        )
        engine = _engine(rulebook)
        unit = _place(engine, 0, source)
        engine.players[0].turns_started = (
            engine.config.first_player_super_evolution_unlock_turn
        )

        engine.apply(SuperEvolve(0, unit.entity_id))

        self.assertEqual(engine.players[1].health, 17)
        evolution_events = [
            event.type
            for event in engine.event_history
            if event.source_id == unit.entity_id
            and event.type
            in {EventType.FOLLOWER_EVOLVED, EventType.FOLLOWER_SUPER_EVOLVED}
        ]
        self.assertEqual(
            evolution_events,
            [EventType.FOLLOWER_EVOLVED, EventType.FOLLOWER_SUPER_EVOLVED],
        )

    def test_evolve_choice_finishes_before_super_evolve_trigger(self):
        source = _card(
            100,
            keywords=frozenset({"进化时", "超进化时"}),
        )
        rulebook = RuleBook(
            rules=(
                CardRule(
                    100,
                    Trigger.EVOLVE,
                    (
                        EffectOperation(
                            EffectKind.DAMAGE_UNIT,
                            TargetKind.ENEMY_UNIT,
                            1,
                        ),
                    ),
                ),
                CardRule(
                    100,
                    Trigger.SUPER_EVOLVE,
                    (
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            2,
                        ),
                    ),
                ),
            )
        )
        engine = _engine(rulebook)
        unit = _place(engine, 0, source)
        target = _place(engine, 1, _card(200, life=3))
        emblem = EmblemDefinition(
            "evolution-watch",
            999900,
            triggers=(
                EmblemTriggerRule(
                    "follower_evolved",
                    operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
        )
        engine._add_emblem_to_player(0, emblem, emblem.source_card_id)
        engine.players[0].turns_started = (
            engine.config.first_player_super_evolution_unlock_turn
        )

        engine.apply(SuperEvolve(0, unit.entity_id))

        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        self.assertEqual(engine.players[1].health, 20)
        self.assertFalse(
            any(
                event.type is EventType.FOLLOWER_SUPER_EVOLVED
                for event in engine.event_history
            )
        )

        engine.apply(Choose(0, request.options[0].option_id))

        self.assertEqual(target.health, 2)
        self.assertEqual(engine.players[1].health, 17)
        self.assertIsNone(engine.state.pending_choice)
        emblem_index = next(
            index
            for index, event in enumerate(engine.event_history)
            if event.type is EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "evolution-watch"
        )
        super_index = next(
            index
            for index, event in enumerate(engine.event_history)
            if event.type is EventType.FOLLOWER_SUPER_EVOLVED
        )
        self.assertLess(emblem_index, super_index)


class EvolutionTriggerSemanticsTests(unittest.TestCase):
    KEYWORD_EVOLVE_DAMAGE = 1
    KEYWORD_SUPER_EVOLVE_DAMAGE = 2
    SELF_EVOLVED_DAMAGE = 4
    SELF_SUPER_EVOLVED_DAMAGE = 8

    @classmethod
    def _rulebook(cls) -> RuleBook:
        damage = lambda amount: (
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                amount,
            ),
        )
        return RuleBook(
            rules=(
                CardRule(
                    100,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.EVOLVE_UNIT,
                            TargetKind.OWN_UNIT,
                            requires_target=True,
                        ),
                    ),
                ),
                CardRule(
                    101,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.SUPER_EVOLVE_UNIT,
                            TargetKind.OWN_UNIT,
                            requires_target=True,
                        ),
                    ),
                ),
                CardRule(
                    200,
                    Trigger.EVOLVE,
                    damage(cls.KEYWORD_EVOLVE_DAMAGE),
                ),
                CardRule(
                    200,
                    Trigger.SUPER_EVOLVE,
                    damage(cls.KEYWORD_SUPER_EVOLVE_DAMAGE),
                ),
                CardRule(
                    200,
                    Trigger.SELF_EVOLVED,
                    damage(cls.SELF_EVOLVED_DAMAGE),
                ),
                CardRule(
                    200,
                    Trigger.SELF_SUPER_EVOLVED,
                    damage(cls.SELF_SUPER_EVOLVED_DAMAGE),
                ),
            )
        )

    def _engine_and_source(self) -> tuple[GameEngine, Unit]:
        engine = _engine(self._rulebook())
        source = _place(
            engine,
            0,
            _card(
                200,
                keywords=frozenset({"进化时", "超进化时"}),
            ),
        )
        return engine, source

    @staticmethod
    def _play_evolution_spell(
        engine: GameEngine,
        source: Unit,
        spell_id: int,
    ) -> None:
        _insert_hand(
            engine,
            _card(spell_id, card_type="法术", attack=None, life=None),
        )
        _play_last_hand_card(engine)
        request = engine.state.pending_choice
        engine.apply(Choose(0, request.options[0].option_id))
        assert source.evolved

    def test_ep_evolution_fires_keyword_and_self_evolved(self):
        engine, source = self._engine_and_source()
        engine.players[0].turns_started = engine.config.evolution_unlock_turn

        engine.apply(Evolve(0, source.entity_id))

        self.assertEqual(
            engine.players[1].health,
            20 - self.KEYWORD_EVOLVE_DAMAGE - self.SELF_EVOLVED_DAMAGE,
        )

    def test_sep_evolution_fires_all_matching_keyword_and_self_triggers(self):
        engine, source = self._engine_and_source()
        engine.players[0].turns_started = (
            engine.config.first_player_super_evolution_unlock_turn
        )

        engine.apply(SuperEvolve(0, source.entity_id))

        self.assertEqual(
            engine.players[1].health,
            20
            - self.KEYWORD_EVOLVE_DAMAGE
            - self.KEYWORD_SUPER_EVOLVE_DAMAGE
            - self.SELF_EVOLVED_DAMAGE
            - self.SELF_SUPER_EVOLVED_DAMAGE,
        )

    def test_effect_evolution_fires_only_self_evolved(self):
        engine, source = self._engine_and_source()

        self._play_evolution_spell(engine, source, 100)

        self.assertFalse(source.super_evolved)
        self.assertEqual(
            engine.players[1].health,
            20 - self.SELF_EVOLVED_DAMAGE,
        )

    def test_effect_super_evolution_fires_only_self_state_triggers(self):
        engine, source = self._engine_and_source()

        self._play_evolution_spell(engine, source, 101)

        self.assertTrue(source.super_evolved)
        self.assertEqual(
            engine.players[1].health,
            20 - self.SELF_EVOLVED_DAMAGE - self.SELF_SUPER_EVOLVED_DAMAGE,
        )


class RealSelfEvolutionRuleTests(unittest.TestCase):
    SELF_EVOLVED_CARD_IDS = (
        10133130,
        10143120,
        10153130,
        10224110,
        10411120,
        10413110,
        10442110,
        10554110,
        10642120,
        10653110,
        10654120,
        10724110,
        10863110,
        10874110,
    )

    def test_plain_self_evolution_text_uses_distinct_triggers(self):
        rulebook = RuleBook.from_directory("data/rules")

        for card_id in self.SELF_EVOLVED_CARD_IDS:
            with self.subTest(card_id=card_id):
                self.assertTrue(
                    rulebook.operations_for(card_id, Trigger.SELF_EVOLVED)
                )
                self.assertFalse(
                    rulebook.operations_for(card_id, Trigger.EVOLVE)
                )
        self.assertTrue(
            rulebook.operations_for(10554110, Trigger.SELF_SUPER_EVOLVED)
        )
        self.assertFalse(
            rulebook.operations_for(10554110, Trigger.SUPER_EVOLVE)
        )


class EffectNormalEvolutionTests(unittest.TestCase):
    @staticmethod
    def _rulebook() -> RuleBook:
        return RuleBook(rules=(
            CardRule(
                100,
                Trigger.PLAY,
                (EffectOperation(
                    EffectKind.EVOLVE_UNIT,
                    TargetKind.OWN_UNIT,
                    requires_target=True,
                ),),
            ),
            CardRule(
                200,
                Trigger.EVOLVE,
                (EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    3,
                ),),
            ),
        ))

    def test_schema_parses_effect_normal_evolution(self):
        operation = _parse_operation(
            {"kind": "evolve_unit", "target": "own_unit", "requires_target": True},
            "test",
            100,
        )
        self.assertIs(operation.kind, EffectKind.EVOLVE_UNIT)

    def test_effect_evolution_is_free_and_skips_keyword_evolve_ability(self):
        engine = _engine(self._rulebook())
        target = _place(engine, 0, _card(200, keywords=frozenset({"进化时"})))
        hand_card = _insert_hand(engine, _card(300), evolutions=2)
        _insert_hand(engine, _card(100, card_type="法术", attack=None, life=None))
        ep_before = engine.players[0].evolution_points

        _play_last_hand_card(engine)
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))

        self.assertTrue(target.evolved)
        self.assertFalse(target.super_evolved)
        self.assertEqual((target.attack, target.health), (4, 5))
        self.assertEqual(engine.players[0].evolution_points, ep_before)
        self.assertFalse(engine.players[0].evolved_this_turn)
        self.assertEqual(engine.players[0].followers_evolved_this_match, 1)
        self.assertEqual(hand_card.evolutions_while_in_hand, 3)
        self.assertEqual(engine.players[1].health, 20)
        event = next(
            event for event in engine.event_history
            if event.type is EventType.FOLLOWER_EVOLVED
            and event.source_id == target.entity_id
        )
        self.assertEqual(event.metadata["cause"], "effect")
        self.assertFalse(event.metadata["trigger_abilities"])

    def test_stale_choice_skips_already_evolved_target(self):
        engine = _engine(self._rulebook())
        target = _place(engine, 0, _card(200))
        _insert_hand(engine, _card(100, card_type="法术", attack=None, life=None))
        _play_last_hand_card(engine)
        request = engine.state.pending_choice
        target.evolved = True
        before = engine.deterministic_fingerprint()

        engine.apply(Choose(0, request.options[0].option_id))

        self.assertEqual(engine.players[0].followers_evolved_this_match, 0)
        self.assertEqual(engine.players[1].health, 20)
        self.assertNotEqual(before, engine.deterministic_fingerprint())


class EffectSuperEvolutionTests(unittest.TestCase):
    @staticmethod
    def _rulebook_for_target(target_card_id: int = 200) -> RuleBook:
        spell = CardRule(
            100,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.SUPER_EVOLVE_UNIT,
                    TargetKind.OWN_UNIT,
                    requires_target=True,
                ),
            ),
        )
        evolve = CardRule(
            target_card_id,
            Trigger.EVOLVE,
            (
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    3,
                ),
            ),
        )
        super_evolve = CardRule(
            target_card_id,
            Trigger.SUPER_EVOLVE,
            (
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    5,
                ),
            ),
        )
        return RuleBook(rules=(spell, evolve, super_evolve))

    def test_effect_super_evolution_is_free_and_counts_as_one_evolution(self):
        engine = _engine(self._rulebook_for_target())
        target = _place(
            engine,
            0,
            _card(
                200,
                keywords=frozenset({"进化时", "超进化时"}),
            ),
        )
        hand_card = _insert_hand(engine, _card(300), evolutions=2)
        _insert_hand(
            engine,
            _card(100, card_type="法术", attack=None, life=None),
        )
        sep_before = engine.players[0].super_evolution_points

        _play_last_hand_card(engine)
        request = engine.state.pending_choice
        engine.apply(Choose(0, request.options[0].option_id))

        self.assertTrue(target.evolved)
        self.assertTrue(target.super_evolved)
        self.assertEqual((target.attack, target.health), (5, 6))
        self.assertEqual(engine.players[0].super_evolution_points, sep_before)
        self.assertFalse(engine.players[0].evolved_this_turn)
        self.assertFalse(engine.players[0].super_evolved_this_turn)
        self.assertEqual(engine.players[0].followers_evolved_this_match, 1)
        self.assertEqual(hand_card.evolutions_while_in_hand, 3)

    def test_effect_super_evolution_does_not_fire_evolution_keywords(self):
        engine = _engine(self._rulebook_for_target())
        target = _place(
            engine,
            0,
            _card(
                200,
                keywords=frozenset({"进化时", "超进化时"}),
            ),
        )
        _insert_hand(
            engine,
            _card(100, card_type="法术", attack=None, life=None),
        )

        _play_last_hand_card(engine)
        request = engine.state.pending_choice
        engine.apply(Choose(0, request.options[0].option_id))

        self.assertEqual(engine.players[1].health, 20)
        self.assertFalse(
            any(
                event.type is EventType.FOLLOWER_EVOLVED
                and event.source_id == target.entity_id
                for event in engine.event_history
            )
        )
        event = next(
            event
            for event in engine.event_history
            if event.type is EventType.FOLLOWER_SUPER_EVOLVED
            and event.source_id == target.entity_id
        )
        self.assertEqual(event.metadata["cause"], "effect")
        self.assertFalse(event.metadata["trigger_abilities"])

    def test_effect_super_evolution_has_full_same_turn_protection(self):
        engine = _engine(self._rulebook_for_target())
        target = _place(engine, 0, _card(200, life=5))
        _insert_hand(
            engine,
            _card(100, card_type="法术", attack=None, life=None),
        )
        _play_last_hand_card(engine)
        request = engine.state.pending_choice
        engine.apply(Choose(0, request.options[0].option_id))
        before = target.health

        combat = engine.apply_damage(
            None,
            target,
            4,
            DamageType.COMBAT,
            1,
        )
        effect = engine.apply_damage(
            None,
            target,
            4,
            DamageType.EFFECT,
            1,
        )

        self.assertEqual(target.health, before)
        self.assertEqual(combat.prevented_amount, 4)
        self.assertEqual(effect.prevented_amount, 4)

    def test_stale_choice_skips_if_target_evolves_before_resolution(self):
        engine = _engine(self._rulebook_for_target())
        target = _place(engine, 0, _card(200))
        _insert_hand(
            engine,
            _card(100, card_type="法术", attack=None, life=None),
        )
        _play_last_hand_card(engine)
        request = engine.state.pending_choice
        target.evolved = True
        before_stats = (target.attack, target.health)

        engine.apply(Choose(0, request.options[0].option_id))

        self.assertFalse(target.super_evolved)
        self.assertEqual((target.attack, target.health), before_stats)
        self.assertEqual(engine.players[0].followers_evolved_this_match, 0)


class SuperEvolutionAttackBonusTests(unittest.TestCase):
    def test_attack_context_invariant_rejects_invalid_identity(self):
        engine = _engine()
        engine._active_super_evolution_attack = SuperEvolutionAttackContext(
            controller=0,
            attacker_id=-1,
            target_id=2,
            attacker_card_id=100,
            attacker_name="invalid",
        )

        with self.assertRaisesRegex(IllegalCommand, "attacker_id"):
            engine.assert_invariants()

    def test_combat_destroy_deals_one_to_enemy_leader(self):
        engine = _engine()
        attacker = _place(engine, 0, _card(100, attack=3, life=5))
        defender = _place(engine, 1, _card(200, attack=9, life=3))
        _mark_super_evolved(engine, attacker)
        attacker.can_attack = True

        engine.apply(Attack(0, attacker.entity_id, defender.entity_id))

        self.assertEqual(attacker.health, 5)
        self.assertNotIn(defender, engine.players[1].board)
        self.assertEqual(engine.players[1].health, 19)
        bonuses = [
            event
            for event in engine.event_history
            if event.type is EventType.SUPER_EVOLUTION_ATTACK_BONUS
        ]
        self.assertEqual(len(bonuses), 1)
        self.assertEqual(bonuses[0].source_id, attacker.entity_id)

    def test_nonlethal_attack_does_not_deal_bonus(self):
        engine = _engine()
        attacker = _place(engine, 0, _card(100, attack=2, life=5))
        defender = _place(engine, 1, _card(200, attack=1, life=5))
        _mark_super_evolved(engine, attacker)
        attacker.can_attack = True

        engine.apply(Attack(0, attacker.entity_id, defender.entity_id))

        self.assertEqual(engine.players[1].health, 20)
        self.assertFalse(
            any(
                event.type is EventType.SUPER_EVOLUTION_ATTACK_BONUS
                for event in engine.event_history
            )
        )

    def test_attack_trigger_destroy_also_deals_bonus(self):
        attacker_definition = _card(
            100,
            attack=1,
            life=5,
            keywords=frozenset({"攻击时"}),
        )
        rulebook = RuleBook(
            rules=(
                CardRule(
                    100,
                    Trigger.ATTACK,
                    (
                        EffectOperation(
                            EffectKind.DESTROY,
                            TargetKind.ALL_ENEMY_UNITS,
                        ),
                    ),
                ),
            )
        )
        engine = _engine(rulebook)
        attacker = _place(engine, 0, attacker_definition)
        defender = _place(engine, 1, _card(200, life=10))
        _mark_super_evolved(engine, attacker)
        attacker.can_attack = True

        engine.apply(Attack(0, attacker.entity_id, defender.entity_id))

        self.assertNotIn(defender, engine.players[1].board)
        self.assertEqual(engine.players[1].health, 19)

    def test_attack_trigger_return_is_not_a_destroy_bonus(self):
        attacker_definition = _card(
            100,
            attack=1,
            life=5,
            keywords=frozenset({"攻击时"}),
        )
        rulebook = RuleBook(
            rules=(
                CardRule(
                    100,
                    Trigger.ATTACK,
                    (
                        EffectOperation(
                            EffectKind.RETURN_TO_HAND,
                            TargetKind.ALL_ENEMY_UNITS,
                        ),
                    ),
                ),
            )
        )
        engine = _engine(rulebook)
        attacker = _place(engine, 0, attacker_definition)
        defender = _place(engine, 1, _card(200, life=1))
        _mark_super_evolved(engine, attacker)
        attacker.can_attack = True

        engine.apply(Attack(0, attacker.entity_id, defender.entity_id))

        self.assertEqual(engine.players[1].health, 20)
        self.assertTrue(
            any(card.card_id == defender.definition.card_id for card in engine.players[1].hand)
        )

    def test_pending_attack_choice_preserves_context_and_invalid_choice_state(self):
        attacker_definition = _card(
            100,
            attack=1,
            life=5,
            keywords=frozenset({"攻击时"}),
        )
        rulebook = RuleBook(
            rules=(
                CardRule(
                    100,
                    Trigger.ATTACK,
                    (
                        EffectOperation(
                            EffectKind.DAMAGE_UNIT,
                            TargetKind.ENEMY_UNIT,
                            10,
                        ),
                    ),
                ),
            )
        )
        engine = _engine(rulebook)
        attacker = _place(engine, 0, attacker_definition)
        defender = _place(engine, 1, _card(200, life=5))
        _mark_super_evolved(engine, attacker)
        attacker.can_attack = True

        engine.apply(Attack(0, attacker.entity_id, defender.entity_id))
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "not-an-option"))
        self.assertEqual(engine.deterministic_fingerprint(), before)

        request = engine.state.pending_choice
        engine.apply(Choose(0, request.options[0].option_id))

        self.assertEqual(engine.players[1].health, 19)
        self.assertIsNone(engine.state.pending_choice)
        engine.assert_invariants()


class SuperEvolutionEnvironmentTests(unittest.TestCase):
    def test_effect_super_evolution_uses_existing_public_board_state(self):
        burst_card = _card(
            100,
            keywords=frozenset({"入场曲", "奥义"}),
        )
        definition = UnionBurstDefinition(
            card_id=100,
            kind=UnionBurstKind.UNION_BURST,
            operations=(
                EffectOperation(
                    EffectKind.SUPER_EVOLVE_UNIT,
                    TargetKind.SELF,
                ),
            ),
        )
        deck = [_card(3000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=RuleBook(union_burst_defs={100: (definition,)}),
        )
        env.reset(seed=42)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        other = _place(env.core, 0, _card(200))
        _insert_hand(env.core, burst_card)
        env.players[0].turns_started = 10
        env.players[0].max_mana = 10
        env.players[0].mana = 10

        result = env.step(ShadowverseEnv.PLAY_OFFSET)

        source = env.players[0].board[1]
        self.assertTrue(source.super_evolved)
        self.assertEqual(env._board_features(source)[8], 1.0)
        self.assertEqual(env.players[0].super_evolution_points, 2)
        self.assertTrue(result.info["action_mask"][ShadowverseEnv.SUPER_EVOLVE_OFFSET])
        self.assertFalse(
            result.info["action_mask"][ShadowverseEnv.SUPER_EVOLVE_OFFSET + 1]
        )
        self.assertFalse(other.evolved)
        self.assertEqual(len(result.observation), 294)


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealMegSuperEvolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def _real_engine(self) -> GameEngine:
        engine = GameEngine(
            [_card(5000 + index) for index in range(40)],
            [_card(6000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=self.rulebook,
            card_resolver=self.repo.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=42)
        return engine

    def test_meg_union_burst_super_evolves_without_sep_or_keyword_triggers(self):
        engine = self._real_engine()
        meg = self.repo.get(10443110)
        _insert_hand(engine, meg)
        engine.players[0].turns_started = 10
        sep_before = engine.players[0].super_evolution_points

        _play_last_hand_card(engine)

        source = next(
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == meg.card_id
        )
        self.assertEqual((source.attack, source.health), (5, 4))
        self.assertTrue(source.super_evolved)
        self.assertEqual(engine.players[0].super_evolution_points, sep_before)
        self.assertFalse(engine.players[0].super_evolved_this_turn)
        event = next(
            event
            for event in engine.event_history
            if event.type is EventType.FOLLOWER_SUPER_EVOLVED
            and event.source_id == source.entity_id
        )
        self.assertEqual(event.metadata["cause"], "effect")
        self.assertFalse(event.metadata["trigger_abilities"])
        self.assertFalse(
            any(
                placeholder.card_id == meg.card_id
                and placeholder.ability
                in {AbilityKeyword.FANFARE, AbilityKeyword.UNION_BURST}
                for placeholder in engine.placeholder_ability_events
            )
        )

    def test_meg_rule_is_exact_with_structured_ward_listener(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10443110"]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertIn("守护", info["rule_metadata"]["implemented_text"])
        self.assertNotIn("unsupported_text", info["rule_metadata"])
        self.assertIn("listener:board:follower_summoned", info["reason"])
        self.assertIn("super_evolve_unit", info["reason"])


if __name__ == "__main__":
    unittest.main()
