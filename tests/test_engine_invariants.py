from __future__ import annotations

import copy
import unittest

from swb.db.repository import CardDefinition
from swb.engine.commands import ChoiceOption, ChoiceRequest, Choose, EndTurn, PlayCard
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


def snapshot(engine: GameEngine):
    return (
        copy.deepcopy(engine.state),
        tuple(engine.logs),
        tuple(engine.event_history),
        tuple(engine.placeholder_ability_events),
        engine.random.getstate(),
        dict(engine._death_causes),
        engine._suspended_action,
        copy.deepcopy(engine._suspended_action_state),
        copy.deepcopy(engine._suspended_event_state),
        engine._next_modifier_id,
        engine._next_choice_request_id,
    )


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


class IllegalCommandNoMutationTests(unittest.TestCase):
    def assert_illegal_does_not_mutate(self, engine: GameEngine, command) -> None:
        engine._ensure_entity_ids()
        before = snapshot(engine)
        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(snapshot(engine), before)

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


if __name__ == "__main__":
    unittest.main()
