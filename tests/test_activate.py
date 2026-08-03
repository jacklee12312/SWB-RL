from __future__ import annotations

import os
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import (
    ActivationDefinition,
    CardRule,
    RuleBook,
    Trigger,
    _parse_activation_definition,
    _parse_operation,
)
from swb.engine.commands import (
    ActivateAmulet,
    Choose,
    EndTurn,
    PlayCard,
    UseExtraPP,
)
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.emblem import EmblemDefinition, EmblemTriggerRule
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import Amulet, HandCard, Unit


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
        "life": 3,
        "keywords": frozenset(),
        "support_level": "basic",
        "is_collectible": True,
    }
    values.update(overrides)
    return CardDefinition(**values)


def _activate_amulet(card_id: int = 100) -> CardDefinition:
    return _card(
        card_id,
        name=f"Act-{card_id}",
        card_type="护符",
        attack=None,
        life=None,
        keywords=frozenset({"启动"}),
    )


def _engine(
    operations: tuple[EffectOperation, ...],
    *,
    cost: int = 1,
    amulet: CardDefinition | None = None,
    emblem_defs: dict[str, EmblemDefinition] | None = None,
) -> GameEngine:
    definition = amulet or _activate_amulet()
    engine = GameEngine(
        [_card(1000 + index) for index in range(40)],
        [_card(2000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=42,
        rulebook=RuleBook(
            rules=(CardRule(definition.card_id, Trigger.ACTIVATE, operations),),
            activation_defs={
                definition.card_id: ActivationDefinition(definition.card_id, cost)
            },
            emblem_defs=emblem_defs,
        ),
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=42)
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _place_amulet(
    engine: GameEngine,
    definition: CardDefinition | None = None,
    *,
    player_index: int = 0,
) -> Amulet:
    amulet = Amulet(
        definition=definition or _activate_amulet(),
        entity_id=engine.state.allocate_entity_id(),
        entered_turn=engine.turn,
        origin=CardOrigin.DECK,
    )
    engine.players[player_index].board.append(amulet)
    return amulet


def _place_unit(
    engine: GameEngine,
    *,
    player_index: int = 0,
    card_id: int = 300,
) -> Unit:
    unit = Unit.summon(
        _card(card_id),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[player_index].board.append(unit)
    return unit


class ActivateCommandTests(unittest.TestCase):
    def test_activation_only_amulet_can_be_played_without_a_synthetic_play_rule(self):
        engine = _engine((
            EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 1),
        ))
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        definition = _activate_amulet()
        hand_card = HandCard(
            definition=definition,
            entity_id=engine.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        engine.players[0].hand.append(hand_card)
        engine.players[0].hand_entity_ids.append(hand_card.entity_id)
        command = PlayCard(0, 0)

        self.assertIn(command, engine.legal_commands())
        engine.apply(command)

        amulet = next(entity for entity in engine.players[0].board if isinstance(entity, Amulet))
        self.assertEqual(amulet.definition.card_id, definition.card_id)
        self.assertIn(ActivateAmulet(0, amulet.entity_id), engine.legal_commands())

    def test_reduce_countdown_clamps_at_zero_and_expires_source(self):
        engine = _engine((
            EffectOperation(
                EffectKind.REDUCE_COUNTDOWN,
                TargetKind.SELF,
                amount=2,
            ),
        ), cost=0)
        amulet = _place_amulet(engine)
        amulet.countdown = 1

        engine.apply(ActivateAmulet(0, amulet.entity_id))

        self.assertNotIn(amulet, engine.players[0].board)
        self.assertTrue(any(
            card.definition.card_id == amulet.definition.card_id
            for card in engine.players[0].graveyard
        ))

    def test_activation_pays_cost_once_and_emits_auditable_event(self):
        engine = _engine(
            (EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 2),),
            cost=2,
        )
        amulet = _place_amulet(engine)
        engine.players[0].health = 15
        command = ActivateAmulet(0, amulet.entity_id)

        self.assertIn(command, engine.legal_commands())
        transition = engine.apply(command)

        self.assertEqual(engine.players[0].mana, 8)
        self.assertEqual(engine.players[0].health, 17)
        self.assertEqual(amulet.activated_turn, engine.turn)
        self.assertNotIn(command, engine.legal_commands())
        event = next(
            event for event in transition.events
            if event.type is EventType.AMULET_ACTIVATED
        )
        self.assertEqual(event.source_id, amulet.entity_id)
        self.assertEqual(event.metadata["cost"], 2)
        self.assertEqual(event.metadata["card_id"], amulet.definition.card_id)

    def test_activation_commits_pending_extra_pp_when_needed(self):
        engine = _engine(
            (EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 1),),
            cost=2,
        )
        amulet = _place_amulet(engine)
        player = engine.players[0]
        engine.state.first_player = 1
        engine.players[1].extra_pp_available = False
        player.max_mana = 1
        player.mana = 1
        player.extra_pp_available = True

        self.assertNotIn(
            ActivateAmulet(0, amulet.entity_id),
            engine.legal_commands(),
        )
        engine.apply(UseExtraPP(0))
        transition = engine.apply(ActivateAmulet(0, amulet.entity_id))

        self.assertEqual(player.mana, 0)
        self.assertEqual(player.extra_pp_uses, 1)
        self.assertFalse(player.extra_pp_pending)
        self.assertIn(
            EventType.EXTRA_PP_USED,
            [event.type for event in transition.events],
        )

    def test_activation_refreshes_on_the_controllers_next_turn(self):
        engine = _engine(
            (EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 1),)
        )
        amulet = _place_amulet(engine)
        command = ActivateAmulet(0, amulet.entity_id)
        engine.apply(command)
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))

        self.assertIn(command, engine.legal_commands())

    def test_second_activation_same_turn_is_illegal_without_mutation(self):
        engine = _engine(
            (EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 1),)
        )
        amulet = _place_amulet(engine)
        command = ActivateAmulet(0, amulet.entity_id)
        engine.apply(command)
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_insufficient_mana_is_illegal_without_mutation(self):
        engine = _engine(
            (EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 1),),
            cost=3,
        )
        amulet = _place_amulet(engine)
        engine.players[0].mana = 2
        command = ActivateAmulet(0, amulet.entity_id)
        before = engine.deterministic_fingerprint()

        self.assertNotIn(command, engine.legal_commands())
        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_changed_controller_rejects_stale_command_without_mutation(self):
        engine = _engine(
            (EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 1),)
        )
        amulet = _place_amulet(engine)
        command = ActivateAmulet(0, amulet.entity_id)
        engine.players[0].board.remove(amulet)
        engine.players[1].board.append(amulet)
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_required_target_absence_prohibits_activation_atomically(self):
        engine = _engine(
            (
                EffectOperation(
                    EffectKind.BUFF_UNIT,
                    TargetKind.OWN_UNIT,
                    1,
                    1,
                    requires_target=True,
                ),
            )
        )
        amulet = _place_amulet(engine)
        command = ActivateAmulet(0, amulet.entity_id)
        before = engine.deterministic_fingerprint()

        self.assertNotIn(command, engine.legal_commands())
        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_illegal_pending_choice_does_not_mutate_paid_activation(self):
        engine = _engine(
            (
                EffectOperation(
                    EffectKind.BUFF_UNIT,
                    TargetKind.OWN_UNIT,
                    1,
                    1,
                    requires_target=True,
                ),
            )
        )
        amulet = _place_amulet(engine)
        _place_unit(engine)
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "entity:999999"))
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_target_leaving_play_is_revalidated_without_repaying(self):
        engine = _engine(
            (
                EffectOperation(
                    EffectKind.BUFF_UNIT,
                    TargetKind.OWN_UNIT,
                    2,
                    2,
                    requires_target=True,
                ),
            ),
            cost=2,
        )
        amulet = _place_amulet(engine)
        target = _place_unit(engine)
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        mana_after_activation = engine.players[0].mana
        engine.players[0].board.remove(target)

        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[0].mana, mana_after_activation)
        self.assertEqual((target.attack, target.health), (1, 3))

    def test_target_changing_controller_is_revalidated(self):
        engine = _engine(
            (
                EffectOperation(
                    EffectKind.BUFF_UNIT,
                    TargetKind.OWN_UNIT,
                    2,
                    2,
                    requires_target=True,
                ),
            )
        )
        amulet = _place_amulet(engine)
        target = _place_unit(engine)
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        engine.players[0].board.remove(target)
        engine.players[1].board.append(target)

        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual((target.attack, target.health), (1, 3))

    def test_source_can_destroy_itself_before_targeted_effect_resolves(self):
        engine = _engine(
            (
                EffectOperation(EffectKind.DESTROY, TargetKind.SELF),
                EffectOperation(
                    EffectKind.BUFF_UNIT,
                    TargetKind.OWN_UNIT,
                    1,
                    1,
                    requires_target=True,
                ),
            )
        )
        amulet = _place_amulet(engine)
        target = _place_unit(engine)

        engine.apply(ActivateAmulet(0, amulet.entity_id))

        self.assertFalse(
            any(entity.entity_id == amulet.entity_id for entity in engine.players[0].board)
        )
        self.assertIsNotNone(engine.state.pending_choice)
        engine.apply(Choose(0, f"entity:{target.entity_id}"))
        self.assertEqual((target.attack, target.health), (2, 4))

    def test_activation_event_can_trigger_structured_emblem(self):
        emblem = EmblemDefinition(
            emblem_id="act-listener",
            source_card_id=900,
            triggers=(
                EmblemTriggerRule(
                    trigger="amulet_activated",
                    operations=(
                        EffectOperation(
                            EffectKind.HEAL_LEADER,
                            TargetKind.OWN_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
        )
        engine = _engine(
            (EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 1),),
            emblem_defs={emblem.emblem_id: emblem},
        )
        engine._add_emblem_to_player(0, emblem, source_card_id=900)
        amulet = _place_amulet(engine)
        engine.players[0].health = 10

        engine.apply(ActivateAmulet(0, amulet.entity_id))

        self.assertEqual(engine.players[0].health, 12)


class ActivateRuleSchemaTests(unittest.TestCase):
    def test_reduce_countdown_requires_positive_amount_and_amulet_target(self):
        operation = _parse_operation(
            {"kind": "reduce_countdown", "target": "self", "amount": 1},
            "test",
            123,
        )
        self.assertEqual(operation.kind, EffectKind.REDUCE_COUNTDOWN)
        self.assertEqual(operation.amount, 1)

        for raw in (
            {"kind": "reduce_countdown", "target": "self", "amount": 0},
            {"kind": "reduce_countdown", "target": "self", "amount": True},
            {"kind": "reduce_countdown", "target": "own_leader", "amount": 1},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test", 123)

    def test_increase_countdown_requires_positive_amount_and_amulet_target(self):
        operation = _parse_operation(
            {"kind": "increase_countdown", "target": "own_amulet", "amount": 1},
            "test",
            123,
        )
        self.assertEqual(operation.kind, EffectKind.INCREASE_COUNTDOWN)
        self.assertEqual(operation.amount, 1)

        for raw in (
            {"kind": "increase_countdown", "target": "own_amulet", "amount": 0},
            {"kind": "increase_countdown", "target": "own_amulet", "amount": True},
            {"kind": "increase_countdown", "target": "own_leader", "amount": 1},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test", 123)

    def test_random_countdown_change_requires_explicit_amulet_filter(self):
        for kind in ("reduce_countdown", "increase_countdown"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                _parse_operation(
                    {"kind": kind, "target": "random_own_board", "amount": 1},
                    "test",
                    123,
                )

            operation = _parse_operation(
                {
                    "kind": kind,
                    "target": "random_own_board",
                    "amount": 1,
                    "target_card_type_filter": "护符",
                },
                "test",
                123,
            )
            self.assertEqual(operation.board_filter.card_type, "护符")

    def test_activation_definition_defaults_to_zero_cost(self):
        definition = _parse_activation_definition({"card_id": 123}, "test")
        self.assertEqual(definition, ActivationDefinition(123, 0))

    def test_activation_definition_rejects_bad_cost_and_unknown_fields(self):
        for raw in (
            {"card_id": 123, "cost": -1},
            {"card_id": 123, "cost": True},
            {"card_id": 123, "unknown": 1},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_activation_definition(raw, "test")

    def test_activation_definition_and_trigger_rule_require_each_other(self):
        operation = EffectOperation(
            EffectKind.HEAL_LEADER,
            TargetKind.OWN_LEADER,
            1,
        )
        with self.assertRaises(ValueError):
            RuleBook(activation_defs={123: ActivationDefinition(123, 1)})
        with self.assertRaises(ValueError):
            RuleBook(rules=(CardRule(123, Trigger.ACTIVATE, (operation,)),))


class ActivateEnvironmentTests(unittest.TestCase):
    def test_rl_mask_exposes_play_then_activation_for_activation_only_amulet(self):
        amulet_definition = _activate_amulet()
        rules = RuleBook(
            rules=(
                CardRule(
                    100,
                    Trigger.ACTIVATE,
                    (EffectOperation(EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, 1),),
                ),
            ),
            activation_defs={100: ActivationDefinition(100, 1)},
        )
        deck = [_card(1000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=rules,
        )
        env.reset(seed=42)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].mana = 2
        hand_card = HandCard(
            definition=amulet_definition,
            entity_id=env.core.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        env.players[0].hand.append(hand_card)
        env.players[0].hand_entity_ids.append(hand_card.entity_id)

        self.assertTrue(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])
        env.step(ShadowverseEnv.PLAY_OFFSET)

        amulet = next(entity for entity in env.players[0].board if isinstance(entity, Amulet))
        self.assertTrue(env.action_mask()[ShadowverseEnv.EVOLVE_OFFSET])
        self.assertEqual(
            env._decode_action(ShadowverseEnv.EVOLVE_OFFSET),
            ActivateAmulet(0, amulet.entity_id),
        )

    def test_rl_reuses_amulet_evolve_slot_and_exposes_turn_usage(self):
        amulet_definition = _activate_amulet()
        rules = RuleBook(
            rules=(
                CardRule(
                    100,
                    Trigger.ACTIVATE,
                    (
                        EffectOperation(
                            EffectKind.HEAL_LEADER,
                            TargetKind.OWN_LEADER,
                            1,
                        ),
                    ),
                ),
            ),
            activation_defs={100: ActivationDefinition(100, 1)},
        )
        deck = [_card(1000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=rules,
        )
        env.reset(seed=42)
        env.players[0].board.clear()
        env.players[0].health = 10
        env.players[0].mana = 2
        amulet = _place_amulet(env.core, amulet_definition)
        action = ShadowverseEnv.EVOLVE_OFFSET

        self.assertEqual(env._board_features(amulet)[10], 0.0)
        self.assertTrue(env.action_mask()[action])
        self.assertEqual(
            env._decode_action(action),
            ActivateAmulet(0, amulet.entity_id),
        )

        result = env.step(action)

        self.assertEqual(env.players[0].health, 11)
        self.assertFalse(result.info["action_mask"][action])
        self.assertEqual(env._board_features(amulet)[10], 1.0)
        self.assertEqual(
            len(result.observation),
            ShadowverseEnv.OBSERVATION_V1_SIZE,
        )


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealActivateCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def _bishop_engine(self) -> GameEngine:
        filler = [
            _card(
                8000 + index,
                class_id=6,
                class_name="主教",
            )
            for index in range(40)
        ]
        engine = GameEngine(
            filler,
            filler,
            class_a=6,
            class_b=6,
            seed=42,
            rulebook=self.rulebook,
            card_resolver=self.repo.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=42)
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10
        return engine

    @staticmethod
    def _insert_hand(engine: GameEngine, definition: CardDefinition) -> None:
        hand_card = HandCard(
            definition=definition,
            entity_id=engine.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        engine.players[0].hand.append(hand_card)
        engine.players[0].hand_entity_ids.append(hand_card.entity_id)

    def test_serene_sanctuary_engage_reduces_countdown_and_runs_last_words(self):
        engine = self._bishop_engine()
        self._insert_hand(engine, self.repo.get(10161210))
        engine.apply(PlayCard(0, 0))
        sanctuary = next(
            entity
            for entity in engine.players[0].board
            if isinstance(entity, Amulet)
            and entity.definition.card_id == 10161210
        )
        sanctuary.countdown = 1
        mana_before = engine.players[0].mana

        transition = engine.apply(ActivateAmulet(0, sanctuary.entity_id))

        self.assertEqual(engine.players[0].mana, mana_before - 1)
        self.assertNotIn(sanctuary, engine.players[0].board)
        self.assertEqual(len(engine.players[0].hand), 2)
        self.assertEqual(
            sum(event.type is EventType.CARD_DRAWN for event in transition.events),
            2,
        )

    def _resolve_mistbloom_engage(self):
        engine = self._bishop_engine()
        enemy = Unit.summon(
            _card(9900, life=6),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board.append(enemy)
        self._insert_hand(engine, self.repo.get(10563210))
        engine.apply(PlayCard(0, 0))
        engine.apply(next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
        ))
        mistbloom = next(
            entity
            for entity in engine.players[0].board
            if isinstance(entity, Amulet)
            and entity.definition.card_id == 10563210
        )
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        engine.players[0].deck = [
            _card(9200 + index, class_id=6, class_name="主教")
            for index in range(10)
        ]
        for index in range(3):
            self._insert_hand(
                engine,
                _card(9100 + index, class_id=6, class_name="主教"),
            )

        transition = engine.apply(ActivateAmulet(0, mistbloom.entity_id))
        returned_ids = tuple(
            event.metadata["source"].card_id
            for event in transition.events
            if event.type is EventType.CARD_RETURNED_TO_DECK
        )
        return engine, mistbloom, transition, returned_ids

    def test_mistbloom_engage_destroys_self_cycles_two_and_is_deterministic(self):
        first = self._resolve_mistbloom_engage()
        second = self._resolve_mistbloom_engage()
        engine, mistbloom, transition, returned_ids = first

        self.assertNotIn(mistbloom, engine.players[0].board)
        self.assertEqual(len(returned_ids), 2)
        self.assertEqual(len(set(returned_ids)), 2)
        self.assertEqual(len(engine.players[0].hand), 3)
        self.assertEqual(len(engine.players[0].deck), 10)
        self.assertEqual(
            sum(event.type is EventType.CARD_DRAWN for event in transition.events),
            2,
        )
        self.assertEqual(returned_ids, second[3])
        self.assertEqual(
            engine.deterministic_fingerprint(),
            second[0].deterministic_fingerprint(),
        )

    def test_real_bishop_engage_cards_are_covered_exact(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in (10161210, 10563210):
            with self.subTest(card_id=card_id):
                classification = report["classifications"][str(card_id)]
                self.assertEqual(classification["coverage"], "covered_exact")
                self.assertNotIn("策动", classification["missing_primitives"])

    def test_witchs_new_brew_activates_for_one_pp_and_is_exact(self):
        filler = [
            _card(
                7000 + index,
                class_id=3,
                class_name="巫师",
            )
            for index in range(40)
        ]
        engine = GameEngine(
            filler,
            filler,
            class_a=3,
            class_b=3,
            seed=42,
            rulebook=self.rulebook,
            card_resolver=self.repo.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=42)
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10
        definition = self.repo.get(10031210)
        hand_card = HandCard(
            definition=definition,
            entity_id=engine.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        engine.players[0].hand.append(hand_card)
        engine.players[0].hand_entity_ids.append(hand_card.entity_id)

        engine.apply(PlayCard(0, 0))
        amulet = next(
            entity for entity in engine.players[0].board
            if isinstance(entity, Amulet) and entity.definition.card_id == 10031210
        )
        mana_before = engine.players[0].mana
        engine.apply(ActivateAmulet(0, amulet.entity_id))

        self.assertEqual(engine.players[0].mana, mana_before - 1)
        self.assertEqual(engine.players[0].earth_sigils, 2)
        self.assertFalse(
            any(
                event.ability is AbilityKeyword.ACTIVATE
                for event in engine.placeholder_ability_events
            )
        )
        self.assertNotIn(ActivateAmulet(0, amulet.entity_id), engine.legal_commands())

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        classification = report["classifications"]["10031210"]
        self.assertEqual(classification["coverage"], "covered_exact")
        self.assertNotIn("策动", classification["missing_primitives"])


if __name__ == "__main__":
    unittest.main()
