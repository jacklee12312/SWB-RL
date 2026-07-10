from __future__ import annotations

import os
import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import (
    CardRule,
    FusionDefinition,
    RuleBook,
    Trigger,
    _parse_fusion_definition,
)
from swb.engine.commands import BeginFusion, ChoiceKind, Choose, EndTurn, PlayCard
from swb.engine.effects import Condition, ConditionType, DeckFilter, EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import FusionMaterial, HandCard


def _card(card_id: int, **overrides) -> CardDefinition:
    values = {
        "card_id": card_id,
        "card_set_id": 10000,
        "class_id": 1,
        "class_name": "精灵",
        "name": f"card-{card_id}",
        "cost": 1,
        "card_type": "随从",
        "attack": 1,
        "life": 1,
        "keywords": frozenset(),
        "support_level": "basic",
        "is_collectible": True,
    }
    values.update(overrides)
    return CardDefinition(**values)


def _fusion_spell(card_id: int = 100) -> CardDefinition:
    return _card(
        card_id,
        name="Fusion Spell",
        cost=2,
        card_type="法术",
        attack=None,
        life=None,
        keywords=frozenset({"融合"}),
    )


def _fusion_definition(
    card_id: int = 100,
    *,
    min_materials: int = 1,
    max_materials: int | None = None,
) -> FusionDefinition:
    return FusionDefinition(
        card_id=card_id,
        material_filter=DeckFilter(class_id=1),
        min_materials=min_materials,
        max_materials=max_materials,
    )


def _engine(
    *,
    rules: tuple[CardRule, ...] = (),
    fusion_definitions: tuple[FusionDefinition, ...] = (_fusion_definition(),),
    definitions: dict[int, CardDefinition] | None = None,
    validate_invariants: bool = True,
) -> GameEngine:
    defs = dict(definitions or {})
    engine = GameEngine(
        [_card(1000 + index) for index in range(40)],
        [_card(2000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=42,
        rulebook=RuleBook(
            rules=rules,
            fusion_defs={definition.card_id: definition for definition in fusion_definitions},
        ),
        card_resolver=lambda card_id: defs.get(card_id),
        config=GameConfig(validate_invariants=validate_invariants),
    )
    engine.reset(seed=42)
    engine.players[0].hand.clear()
    engine.players[0].hand_entity_ids.clear()
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _insert(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    index: int = 0,
    origin: CardOrigin = CardOrigin.DECK,
) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=origin,
    )
    engine.players[0].hand.insert(index, hand_card)
    engine.players[0].hand_entity_ids.insert(index, hand_card.entity_id)
    return hand_card


def _choose(engine: GameEngine, option_id: str) -> None:
    engine.apply(Choose(engine.state.pending_choice.player_index, option_id))


def _fuse_one(engine: GameEngine, target: HandCard, material: HandCard) -> None:
    engine.apply(BeginFusion(0, target.entity_id))
    _choose(engine, f"hand:{material.entity_id}")
    _choose(engine, "fusion:confirm")


class FusionCommandTests(unittest.TestCase):
    def test_begin_fusion_is_legal_without_mana_and_uses_hand_choice(self):
        target = _fusion_spell()
        material = _card(200)
        engine = _engine(definitions={100: target, 200: material})
        target_hand = _insert(engine, target)
        material_hand = _insert(engine, material, index=1)
        engine.players[0].mana = 0

        begin = BeginFusion(0, target_hand.entity_id)
        self.assertIn(begin, engine.legal_commands())
        self.assertNotIn(PlayCard(0, 0), engine.legal_commands())

        engine.apply(begin)

        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        self.assertIs(request.choice_kind, ChoiceKind.FUSION)
        self.assertEqual(
            {option.option_id for option in request.options},
            {f"hand:{material_hand.entity_id}", "fusion:cancel"},
        )

    def test_fusion_consumes_material_without_graveyard_shadows_or_banish(self):
        target = _fusion_spell()
        material = _card(201, name="Material")
        engine = _engine(definitions={100: target, 201: material})
        target_hand = _insert(engine, target)
        material_hand = _insert(engine, material, index=1, origin=CardOrigin.GENERATED)
        grave_before = len(engine.players[0].graveyard)
        shadows_before = engine.players[0].shadows

        _fuse_one(engine, target_hand, material_hand)

        self.assertNotIn(material_hand, engine.players[0].hand)
        self.assertEqual(len(engine.players[0].graveyard), grave_before)
        self.assertEqual(engine.players[0].shadows, shadows_before)
        self.assertNotIn(material, engine.players[0].banished)
        self.assertEqual(target_hand.fused_material_ids, [material_hand.entity_id])
        record = engine.players[0].fusion_materials[0]
        self.assertEqual(record.entity_id, material_hand.entity_id)
        self.assertIs(record.origin, CardOrigin.GENERATED)
        self.assertEqual(record.fused_into_entity_id, target_hand.entity_id)
        event = next(
            event for event in engine.event_history if event.type is EventType.CARD_FUSED
        )
        self.assertEqual(event.amount, 1)
        self.assertEqual(event.metadata["material_card_ids"], (material.card_id,))

    def test_unspecified_count_can_select_multiple_then_confirm(self):
        target = _fusion_spell()
        first = _card(202)
        second = _card(203)
        engine = _engine(definitions={100: target, 202: first, 203: second})
        target_hand = _insert(engine, target)
        first_hand = _insert(engine, first, index=1)
        second_hand = _insert(engine, second, index=2)

        engine.apply(BeginFusion(0, target_hand.entity_id))
        _choose(engine, f"hand:{first_hand.entity_id}")
        self.assertIn(
            f"hand:{second_hand.entity_id}",
            {option.option_id for option in engine.state.pending_choice.options},
        )
        _choose(engine, f"hand:{second_hand.entity_id}")
        self.assertEqual(
            {option.option_id for option in engine.state.pending_choice.options},
            {"fusion:confirm", "fusion:cancel"},
        )
        _choose(engine, "fusion:confirm")

        self.assertEqual(
            target_hand.fused_material_ids,
            [first_hand.entity_id, second_hand.entity_id],
        )
        self.assertEqual(len(engine.players[0].fusion_materials), 2)

    def test_max_materials_stops_additional_selection(self):
        target = _fusion_spell()
        first = _card(204)
        second = _card(205)
        engine = _engine(
            fusion_definitions=(_fusion_definition(max_materials=1),),
            definitions={100: target, 204: first, 205: second},
        )
        target_hand = _insert(engine, target)
        first_hand = _insert(engine, first, index=1)
        _insert(engine, second, index=2)

        engine.apply(BeginFusion(0, target_hand.entity_id))
        _choose(engine, f"hand:{first_hand.entity_id}")

        self.assertEqual(
            {option.option_id for option in engine.state.pending_choice.options},
            {"fusion:confirm", "fusion:cancel"},
        )

    def test_cancel_after_selection_consumes_nothing(self):
        target = _fusion_spell()
        material = _card(206)
        engine = _engine(definitions={100: target, 206: material})
        target_hand = _insert(engine, target)
        material_hand = _insert(engine, material, index=1)

        engine.apply(BeginFusion(0, target_hand.entity_id))
        _choose(engine, f"hand:{material_hand.entity_id}")
        _choose(engine, "fusion:cancel")

        self.assertIn(material_hand, engine.players[0].hand)
        self.assertEqual(target_hand.fused_material_ids, [])
        self.assertIsNone(target_hand.fusion_used_turn)
        self.assertEqual(engine.players[0].fusion_materials, [])

    def test_wrong_class_material_is_not_a_candidate(self):
        target = _fusion_spell()
        wrong = _card(207, class_id=2, class_name="皇家护卫")
        engine = _engine(definitions={100: target, 207: wrong})
        target_hand = _insert(engine, target)
        _insert(engine, wrong, index=1)

        self.assertNotIn(BeginFusion(0, target_hand.entity_id), engine.legal_commands())
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(BeginFusion(0, target_hand.entity_id))
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_each_fusion_card_can_fuse_once_per_turn_and_again_next_turn(self):
        target = _fusion_spell()
        first = _card(208)
        second = _card(209)
        engine = _engine(definitions={100: target, 208: first, 209: second})
        target_hand = _insert(engine, target)
        first_hand = _insert(engine, first, index=1)
        _insert(engine, second, index=2)

        _fuse_one(engine, target_hand, first_hand)
        self.assertNotIn(BeginFusion(0, target_hand.entity_id), engine.legal_commands())

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))

        self.assertIn(BeginFusion(0, target_hand.entity_id), engine.legal_commands())

    def test_two_fusion_cards_can_each_fuse_in_same_turn(self):
        first_target = _fusion_spell(100)
        second_target = _fusion_spell(101)
        first_material = _card(210)
        second_material = _card(211)
        engine = _engine(
            fusion_definitions=(_fusion_definition(100), _fusion_definition(101)),
            definitions={
                100: first_target,
                101: second_target,
                210: first_material,
                211: second_material,
            },
        )
        first_target_hand = _insert(engine, first_target)
        second_target_hand = _insert(engine, second_target, index=1)
        first_material_hand = _insert(engine, first_material, index=2)
        second_material_hand = _insert(engine, second_material, index=3)

        _fuse_one(engine, first_target_hand, first_material_hand)
        _fuse_one(engine, second_target_hand, second_material_hand)

        self.assertEqual(len(engine.players[0].fusion_materials), 2)

    def test_fusion_target_leaving_hand_is_revalidated_atomically(self):
        target = _fusion_spell()
        material = _card(212)
        engine = _engine(definitions={100: target, 212: material})
        target_hand = _insert(engine, target)
        material_hand = _insert(engine, material, index=1)
        engine.apply(BeginFusion(0, target_hand.entity_id))
        target_index = engine.players[0].hand.index(target_hand)
        engine.players[0].hand.pop(target_index)
        engine.players[0].hand_entity_ids.pop(target_index)
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            _choose(engine, f"hand:{material_hand.entity_id}")
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_selected_material_leaving_hand_is_revalidated_atomically(self):
        target = _fusion_spell()
        material = _card(213)
        engine = _engine(definitions={100: target, 213: material})
        target_hand = _insert(engine, target)
        material_hand = _insert(engine, material, index=1)
        engine.apply(BeginFusion(0, target_hand.entity_id))
        _choose(engine, f"hand:{material_hand.entity_id}")
        material_index = engine.players[0].hand.index(material_hand)
        engine.players[0].hand.pop(material_index)
        engine.players[0].hand_entity_ids.pop(material_index)
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            _choose(engine, "fusion:confirm")
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_consuming_previously_fused_card_preserves_inherited_material_ids(self):
        first_target = _fusion_spell(100)
        second_target = _fusion_spell(101)
        first_material = _card(214)
        engine = _engine(
            fusion_definitions=(_fusion_definition(100), _fusion_definition(101)),
            definitions={100: first_target, 101: second_target, 214: first_material},
        )
        first_target_hand = _insert(engine, first_target)
        second_target_hand = _insert(engine, second_target, index=1)
        first_material_hand = _insert(engine, first_material, index=2)
        _fuse_one(engine, first_target_hand, first_material_hand)

        _fuse_one(engine, second_target_hand, first_target_hand)

        consumed_target = next(
            record
            for record in engine.players[0].fusion_materials
            if record.entity_id == first_target_hand.entity_id
        )
        self.assertEqual(
            consumed_target.inherited_material_ids,
            (first_material_hand.entity_id,),
        )


class FusionSchemaTests(unittest.TestCase):
    def test_schema_parses_filters_and_limits(self):
        definition = _parse_fusion_definition(
            {
                "card_id": 100,
                "material_filter": {
                    "class_id": 1,
                    "card_type": "随从",
                    "cost_max": 3,
                },
                "min_materials": 2,
                "max_materials": 3,
            },
            "test/fusions[0]",
        )
        self.assertEqual(definition.min_materials, 2)
        self.assertEqual(definition.max_materials, 3)
        self.assertEqual(definition.material_filter.class_id, 1)
        self.assertEqual(definition.material_filter.card_type, "随从")

    def test_schema_rejects_bad_filters_and_limits(self):
        invalid = (
            {"card_id": 100, "material_filter": []},
            {"card_id": 100, "material_filter": {"unknown": 1}},
            {"card_id": 100, "material_filter": {}, "min_materials": 0},
            {"card_id": 100, "material_filter": {}, "min_materials": 2, "max_materials": 1},
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_fusion_definition(raw, "test/fusions[0]")

    def test_fusion_and_play_modes_share_three_special_action_slots(self):
        from swb.engine.play_modes import PlayModeDefinition

        modes = tuple(
            PlayModeDefinition(f"mode-{index}", "enhance", index)
            for index in range(3)
        )
        with self.assertRaises(ValueError):
            RuleBook(
                play_modes={100: modes},
                fusion_defs={100: _fusion_definition()},
            )

    def test_source_fusion_condition_evaluates_from_spell_frame(self):
        target = _fusion_spell()
        material = _card(215)
        rule = CardRule(
            100,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    2,
                    conditions=(
                        Condition(ConditionType.SOURCE_FUSION_COUNT_AT_LEAST, 1),
                    ),
                ),
            ),
        )
        engine = _engine(rules=(rule,), definitions={100: target, 215: material})
        target_hand = _insert(engine, target)
        material_hand = _insert(engine, material, index=1)
        _fuse_one(engine, target_hand, material_hand)
        hand_index = engine.players[0].hand.index(target_hand)

        engine.apply(PlayCard(0, hand_index))

        self.assertEqual(engine.players[1].health, 18)


class FusionEnvironmentTests(unittest.TestCase):
    def test_rl_reuses_special_and_choice_actions_for_fusion(self):
        target = _fusion_spell()
        material = _card(216)
        rulebook = RuleBook(
            rules=(
                CardRule(
                    100,
                    Trigger.PLAY,
                    (EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, 1),),
                ),
            ),
            fusion_defs={100: _fusion_definition()},
        )
        env = ShadowverseEnv(
            [_card(3000 + index) for index in range(40)],
            [_card(4000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=rulebook,
            validate_invariants=True,
        )
        env.reset(seed=42)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].max_mana = 10
        env.players[0].mana = 10
        target_hand = _insert(env.core, target)
        material_hand = _insert(env.core, material, index=1)
        fusion_action = ShadowverseEnv.MODE_PLAY_OFFSET

        mask = env.action_mask()
        self.assertTrue(mask[ShadowverseEnv.PLAY_OFFSET])
        self.assertTrue(mask[fusion_action])
        env.step(fusion_action)
        self.assertIs(env.core.state.pending_choice.choice_kind, ChoiceKind.FUSION)

        material_option_index = next(
            index
            for index, option in enumerate(env.core.state.pending_choice.options)
            if option.option_id == f"hand:{material_hand.entity_id}"
        )
        env.step(ShadowverseEnv.CHOICE_OFFSET + material_option_index)
        confirm_index = next(
            index
            for index, option in enumerate(env.core.state.pending_choice.options)
            if option.option_id == "fusion:confirm"
        )
        env.step(ShadowverseEnv.CHOICE_OFFSET + confirm_index)

        self.assertEqual(env._card_features(target_hand)[-2:], [1 / 9, 1.0])
        self.assertFalse(env.action_mask()[fusion_action])
        self.assertEqual(len(env.observation()), 255)
        self.assertEqual(ShadowverseEnv.ACTION_SIZE, 111)

    def test_opponent_hidden_fusion_state_does_not_change_observation(self):
        env = ShadowverseEnv(
            [_card(3000 + index) for index in range(40)],
            [_card(4000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=42,
            validate_invariants=True,
        )
        env.reset(seed=42)
        before = env.observation()
        opponent = env.players[1]
        hidden_card = opponent.hand[0]
        material_id = env.core.state.allocate_entity_id()
        opponent.fusion_materials.append(
            FusionMaterial(
                definition=_card(4999),
                entity_id=material_id,
                owner=1,
                consumed_sequence=1,
                fused_into_entity_id=hidden_card.entity_id,
                origin=CardOrigin.DECK,
            )
        )
        opponent._next_fusion_sequence = 2
        hidden_card.fused_material_ids.append(material_id)
        hidden_card.fusion_used_turn = env.core.turn
        env.core.assert_invariants()

        self.assertEqual(env.observation(), before)


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealFusionCardTests(unittest.TestCase):
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
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10
        return engine

    def test_garden_guidance_draws_one_without_fusion(self):
        engine = self._real_engine()
        guidance = self.repo.get(10213310)
        _insert(engine, guidance)
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_garden_guidance_fuses_only_elf_card_and_draws_two(self):
        engine = self._real_engine()
        guidance = self.repo.get(10213310)
        self.assertIn(AbilityKeyword.FUSION, guidance.abilities)
        guidance_hand = _insert(engine, guidance)
        elf_material = _insert(engine, _card(7001), index=1)
        royal_material = _insert(
            engine,
            _card(7002, class_id=2, class_name="皇家护卫"),
            index=2,
        )
        engine.apply(BeginFusion(0, guidance_hand.entity_id))
        option_ids = {option.option_id for option in engine.state.pending_choice.options}
        self.assertIn(f"hand:{elf_material.entity_id}", option_ids)
        self.assertNotIn(f"hand:{royal_material.entity_id}", option_ids)
        _choose(engine, f"hand:{elf_material.entity_id}")
        _choose(engine, "fusion:confirm")
        deck_before = len(engine.players[0].deck)
        hand_index = engine.players[0].hand.index(guidance_hand)

        engine.apply(PlayCard(0, hand_index))

        self.assertEqual(len(engine.players[0].deck), deck_before - 2)
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.FUSION
                for event in engine.placeholder_ability_events
            )
        )

    def test_garden_guidance_coverage_is_exact(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10213310"]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertEqual(info["missing_primitives"], [])


if __name__ == "__main__":
    unittest.main()
