from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import RuleBook
from swb.engine.commands import (
    ChoiceKind,
    ChoiceOption,
    ChoiceRequest,
    Choose,
    EndTurn,
    PlayCard,
    SuperEvolve,
)
from swb.engine.effects import EffectFrame, EffectKind, EffectOperation, TargetKind
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import Phase, Unit


def card(cid: int, **kw) -> CardDefinition:
    defaults = dict(
        card_id=cid,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"c{cid}",
        cost=1,
        card_type="随从",
        attack=1,
        life=1,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )
    defaults.update(kw)
    return CardDefinition(**defaults)


def mkengine(*, validate_invariants: bool = False) -> GameEngine:
    engine = GameEngine(
        [card(100 + i) for i in range(40)],
        [card(200 + i) for i in range(40)],
        class_a=1,
        class_b=1,
        seed=1,
        config=GameConfig(validate_invariants=validate_invariants),
    )
    engine.reset(seed=1)
    return engine


def effect_frame(**kw) -> EffectFrame:
    source = kw.pop("source_card", card(900))
    operations = kw.pop(
        "operations",
        (EffectOperation(EffectKind.DRAW, TargetKind.OWN_LEADER, amount=1),),
    )
    defaults = dict(
        controller=0,
        source_card_id=source.card_id,
        source_name=source.name,
        source_entity_id=None,
        source_card=source,
        operations=operations,
    )
    defaults.update(kw)
    return EffectFrame(**defaults)


class RuntimeInvariantTests(unittest.TestCase):
    def test_invariants_pass_after_reset_and_legal_turn(self):
        engine = mkengine(validate_invariants=True)
        engine.assert_invariants()
        engine.apply(EndTurn(0))
        engine.assert_invariants()

    def test_duplicate_entity_across_zones_is_rejected(self):
        engine = mkengine()
        duplicate_id = engine.players[0].hand_entity_ids[0]
        unit = Unit.summon(card(900), entity_id=duplicate_id)
        engine.players[0].board.append(unit)
        with self.assertRaisesRegex(IllegalCommand, "appears in both"):
            engine.assert_invariants()

    def test_hand_entity_id_mismatch_is_rejected(self):
        engine = mkengine()
        engine.players[0].hand_entity_ids[0] += 999
        with self.assertRaisesRegex(IllegalCommand, "hand\\[0\\] entity_id"):
            engine.assert_invariants()

    def test_board_overflow_is_rejected(self):
        engine = mkengine()
        engine.players[0].board = [
            Unit.summon(card(900 + i), entity_id=engine.state.allocate_entity_id())
            for i in range(engine.config.max_board + 1)
        ]
        with self.assertRaisesRegex(IllegalCommand, "board exceeds max_board"):
            engine.assert_invariants()

    def test_negative_super_evolution_points_are_rejected(self):
        engine = mkengine()
        engine.players[0].super_evolution_points = -1
        with self.assertRaisesRegex(IllegalCommand, "super_evolution_points"):
            engine.assert_invariants()

    def test_super_evolved_unit_without_turn_stamp_is_rejected(self):
        engine = mkengine()
        unit = Unit.summon(card(900), entity_id=engine.state.allocate_entity_id())
        unit.evolved = True
        unit.super_evolved = True
        engine.players[0].board.append(unit)
        with self.assertRaisesRegex(IllegalCommand, "super_evolved without turn stamp"):
            engine.assert_invariants()

    def test_super_evolved_turn_stamp_without_state_is_rejected(self):
        engine = mkengine()
        unit = Unit.summon(card(900), entity_id=engine.state.allocate_entity_id())
        unit.super_evolved_turn = engine.turn
        engine.players[0].board.append(unit)
        with self.assertRaisesRegex(IllegalCommand, "super_evolved_turn without super_evolved"):
            engine.assert_invariants()

    def test_pending_choice_duplicate_option_ids_are_rejected(self):
        engine = mkengine()
        engine.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="pick",
            options=(
                ChoiceOption("same", "A"),
                ChoiceOption("same", "B"),
            ),
            continuation_id="test",
            request_id=1,
        )
        engine.state.phase = Phase.AWAITING_CHOICE
        with self.assertRaisesRegex(IllegalCommand, "duplicate choice option_id"):
            engine.assert_invariants()

    def test_pending_choice_bad_payloads_are_rejected(self):
        cases = (
            (ChoiceOption("", "empty"), "empty option_id"),
            (ChoiceOption("entity:0", "bad entity", entity_id=0), "non-positive entity_id"),
            (ChoiceOption("leader:2", "bad leader", leader_player_index=2), "leader_player_index"),
            (
                ChoiceOption("mixed", "mixed", entity_id=1, leader_player_index=0),
                "cannot target both entity and leader",
            ),
        )
        for option, message in cases:
            with self.subTest(message=message):
                engine = mkengine()
                engine.state.pending_choice = ChoiceRequest(
                    player_index=0,
                    prompt="pick",
                    options=(option,),
                    continuation_id="test",
                    request_id=1,
                )
                engine.state.phase = Phase.AWAITING_CHOICE
                with self.assertRaisesRegex(IllegalCommand, message):
                    engine.assert_invariants()

    def test_pending_choice_leader_option_id_mismatch_is_rejected(self):
        engine = mkengine()
        engine.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="pick",
            options=(ChoiceOption("leader:0", "wrong leader", leader_player_index=1),),
            continuation_id="test",
            request_id=1,
        )
        engine.state.phase = Phase.AWAITING_CHOICE
        with self.assertRaisesRegex(IllegalCommand, "leader option_id mismatch"):
            engine.assert_invariants()

    def test_pending_choice_entity_option_id_matches_payload(self):
        for choice_kind, option in (
            (
                ChoiceKind.BOARD,
                ChoiceOption("entity:123", "board target", entity_id=123),
            ),
            (
                ChoiceKind.HAND,
                ChoiceOption("hand:456", "hand target", entity_id=456),
            ),
        ):
            with self.subTest(choice_kind=choice_kind):
                engine = mkengine()
                engine.state.pending_choice = ChoiceRequest(
                    player_index=0,
                    prompt="pick",
                    options=(option,),
                    continuation_id="test",
                    choice_kind=choice_kind,
                    request_id=1,
                )
                engine.state.phase = Phase.AWAITING_CHOICE
                engine.assert_invariants()

    def test_pending_choice_entity_option_id_mismatch_is_rejected(self):
        cases = (
            (
                ChoiceOption("entity:124", "wrong board target", entity_id=123),
                "entity: option_id mismatch",
            ),
            (
                ChoiceOption("hand:457", "wrong hand target", entity_id=456),
                "hand: option_id mismatch",
            ),
            (
                ChoiceOption("entity:not-int", "bad board target", entity_id=123),
                "malformed entity id",
            ),
        )
        for option, message in cases:
            with self.subTest(message=message):
                engine = mkengine()
                engine.state.pending_choice = ChoiceRequest(
                    player_index=0,
                    prompt="pick",
                    options=(option,),
                    continuation_id="test",
                    request_id=1,
                )
                engine.state.phase = Phase.AWAITING_CHOICE
                with self.assertRaisesRegex(IllegalCommand, message):
                    engine.assert_invariants()

    def test_pending_choice_invalid_kind_is_rejected(self):
        engine = mkengine()
        engine.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="pick",
            options=(ChoiceOption("ok", "OK"),),
            continuation_id="test",
            choice_kind="not-a-choice-kind",
            request_id=1,
        )
        engine.state.phase = Phase.AWAITING_CHOICE
        with self.assertRaisesRegex(IllegalCommand, "invalid choice_kind"):
            engine.assert_invariants()

    def test_pending_choice_stale_entity_option_is_allowed(self):
        engine = mkengine()
        engine.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="pick",
            options=(ChoiceOption("entity:999999", "Gone", entity_id=999999),),
            continuation_id="test",
            choice_kind=ChoiceKind.BOARD,
            request_id=1,
        )
        engine.state.phase = Phase.AWAITING_CHOICE
        engine.assert_invariants()

    def test_effect_stack_next_index_out_of_range_is_rejected(self):
        engine = mkengine()
        engine.state.effect_stack.append(effect_frame(next_index=2))
        with self.assertRaisesRegex(IllegalCommand, "next_index out of range"):
            engine.assert_invariants()

    def test_effect_stack_source_card_id_mismatch_is_rejected(self):
        engine = mkengine()
        engine.state.effect_stack.append(effect_frame(source_card_id=901))
        with self.assertRaisesRegex(IllegalCommand, "source_card_id mismatch"):
            engine.assert_invariants()

    def test_effect_stack_bad_pending_target_id_is_rejected(self):
        cases = (
            (0, "must be positive or leader"),
            (-3, "leader index out of range"),
        )
        for target_id, message in cases:
            with self.subTest(target_id=target_id):
                engine = mkengine()
                engine.state.effect_stack.append(
                    effect_frame(pending_target_id=target_id)
                )
                with self.assertRaisesRegex(IllegalCommand, message):
                    engine.assert_invariants()

    def test_effect_stack_operation_payload_is_rejected(self):
        engine = mkengine()
        engine.state.effect_stack.append(effect_frame(operations=("bad",)))
        with self.assertRaisesRegex(IllegalCommand, "is not EffectOperation"):
            engine.assert_invariants()

    def test_effect_stack_emblem_fields_must_be_consistent(self):
        engine = mkengine()
        engine.state.effect_stack.append(
            effect_frame(emblem_activation_owner=0, emblem_activation_entity_id=1)
        )
        with self.assertRaisesRegex(IllegalCommand, "activation fields require batch"):
            engine.assert_invariants()

        engine = mkengine()
        engine.state.effect_stack.append(
            effect_frame(
                emblem_batch_id=1,
                emblem_activation_owner=0,
                emblem_activation_entity_id=1,
                emblem_activation_trigger_index=-1,
            )
        )
        with self.assertRaisesRegex(IllegalCommand, "activation_trigger_index"):
            engine.assert_invariants()


class IllegalCommandNoMutationTests(unittest.TestCase):
    def assert_illegal_does_not_mutate(self, engine: GameEngine, command) -> None:
        engine._ensure_entity_ids()
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def real_choice_engine(self) -> GameEngine:
        rulebook = RuleBook.from_directory("data/rules")
        choice_spell = card(
            10153310,
            attack=None,
            life=None,
            cost=2,
            card_type="法术",
        )
        engine = GameEngine(
            [choice_spell] * 40,
            [card(200 + i) for i in range(40)],
            class_a=1,
            class_b=1,
            seed=7,
            rulebook=rulebook,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=7)
        target = Unit.summon(
            card(901, attack=1, life=5),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board = [target]
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        return engine

    def test_non_active_end_turn_does_not_mutate(self):
        engine = mkengine(validate_invariants=True)
        self.assert_illegal_does_not_mutate(engine, EndTurn(1))

    def test_unaffordable_play_does_not_mutate(self):
        engine = mkengine(validate_invariants=True)
        engine.players[0].mana = 0
        self.assert_illegal_does_not_mutate(engine, PlayCard(0, 0))

    def test_wrong_player_choice_does_not_mutate(self):
        engine = mkengine(validate_invariants=True)
        engine.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="pick",
            options=(ChoiceOption("ok", "OK"),),
            continuation_id="test",
            request_id=1,
        )
        engine.state.phase = Phase.AWAITING_CHOICE
        self.assert_illegal_does_not_mutate(engine, Choose(1, "ok"))

    def test_invalid_choice_option_does_not_mutate(self):
        engine = mkengine(validate_invariants=True)
        engine.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="pick",
            options=(ChoiceOption("ok", "OK"),),
            continuation_id="test",
            request_id=1,
        )
        engine.state.phase = Phase.AWAITING_CHOICE
        self.assert_illegal_does_not_mutate(engine, Choose(0, "missing"))

    def test_invalid_leader_choice_payload_does_not_mutate(self):
        engine = mkengine(validate_invariants=True)
        engine.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="pick",
            options=(ChoiceOption("leader:0", "wrong leader", leader_player_index=1),),
            continuation_id="test",
            request_id=1,
        )
        engine.state.phase = Phase.AWAITING_CHOICE
        self.assert_illegal_does_not_mutate(engine, Choose(0, "leader:0"))

    def test_invalid_entity_choice_payload_does_not_mutate(self):
        engine = mkengine(validate_invariants=True)
        engine.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="pick",
            options=(ChoiceOption("entity:123", "wrong target", entity_id=456),),
            continuation_id="test",
            request_id=1,
        )
        engine.state.phase = Phase.AWAITING_CHOICE
        self.assert_illegal_does_not_mutate(engine, Choose(0, "entity:123"))

    def test_real_card_pending_choice_blocks_other_command_without_mutation(self):
        engine = self.real_choice_engine()
        self.assert_illegal_does_not_mutate(engine, EndTurn(0))

    def test_real_card_invalid_choice_option_keeps_pending_state_usable(self):
        engine = self.real_choice_engine()
        self.assert_illegal_does_not_mutate(engine, Choose(0, "entity:999999"))

        engine.apply(Choose(0, "leader:1"))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[1].health, 17)
        self.assertEqual(engine.players[0].health, 18)

    def test_locked_super_evolve_does_not_mutate(self):
        engine = mkengine(validate_invariants=True)
        unit = Unit.summon(card(900), entity_id=engine.state.allocate_entity_id())
        engine.players[0].board = [unit]
        engine.players[0].turns_started = (
            engine.config.first_player_super_evolution_unlock_turn - 1
        )
        self.assert_illegal_does_not_mutate(engine, SuperEvolve(0, unit.entity_id))

    def test_super_evolve_without_resource_does_not_mutate(self):
        engine = mkengine(validate_invariants=True)
        unit = Unit.summon(card(900), entity_id=engine.state.allocate_entity_id())
        engine.players[0].board = [unit]
        engine.players[0].turns_started = engine.config.first_player_super_evolution_unlock_turn
        engine.players[0].super_evolution_points = 0
        self.assert_illegal_does_not_mutate(engine, SuperEvolve(0, unit.entity_id))


if __name__ == "__main__":
    unittest.main()
