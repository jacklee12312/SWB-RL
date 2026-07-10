from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import (
    CardRule,
    RuleBook,
    Trigger,
    _parse_operation,
)
from swb.engine.commands import ChoiceKind, ChoiceOption, ChoiceRequest, Choose
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import IllegalCommand
from swb.engine.state import GraveyardCard, Unit


def _card(card_id: int, *, card_type: str = "随从") -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=1,
        card_type=card_type,
        attack=1 if card_type == "随从" else None,
        life=1 if card_type == "随从" else None,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def _env(*, rulebook: RuleBook | None = None) -> ShadowverseEnv:
    return ShadowverseEnv(
        [_card(index) for index in range(1000, 1040)],
        [_card(index) for index in range(2000, 2040)],
        class_a=1,
        class_b=1,
        seed=1,
        rulebook=rulebook or RuleBook(),
    )


class EnvironmentContractTests(unittest.TestCase):
    def test_reset_returns_observation_and_info(self):
        env = _env()
        observation, info = env.reset(seed=1)
        self.assertEqual(len(observation), 227)
        self.assertEqual(info["decision_player"], 0)

    def test_defender_choice_terminal_reward_uses_decision_player(self):
        rulebook = RuleBook((
            CardRule(
                card_id=400,
                trigger=Trigger.CLASH,
                operations=(
                    EffectOperation(
                        EffectKind.DAMAGE_UNIT,
                        TargetKind.ENEMY_BOARD,
                        amount=0,
                    ),
                    EffectOperation(
                        EffectKind.DAMAGE_LEADER,
                        TargetKind.ENEMY_LEADER,
                        amount=20,
                    ),
                ),
            ),
        ))
        env = _env(rulebook=rulebook)
        env.reset(seed=1)
        attacker = Unit.summon(
            _card(300), entity_id=env.core.state.allocate_entity_id()
        )
        attacker.can_attack = True
        defender_definition = CardDefinition(
            **{
                **_card(400).__dict__,
                "keywords": frozenset({"交战时"}),
            }
        )
        defender = Unit.summon(
            defender_definition,
            entity_id=env.core.state.allocate_entity_id(),
        )
        env.players[0].board = [attacker]
        env.players[1].board = [defender]

        env.step(env.ATTACK_OFFSET + 1)
        self.assertEqual(env.decision_player, 1)
        choice_action = next(
            index for index, allowed in enumerate(env.action_mask()) if allowed
        )
        result = env.step(choice_action)
        self.assertEqual(env.winner, 1)
        self.assertEqual(result.reward, 1.0)


class GraveyardPaginationTests(unittest.TestCase):
    def _set_request(
        self,
        env: ShadowverseEnv,
        *,
        count: int,
        request_id: int,
        continuation_id: str = "same",
    ) -> None:
        env.core.state.pending_choice = ChoiceRequest(
            player_index=0,
            prompt="墓地选择",
            options=tuple(
                ChoiceOption(f"entity:{50000 + index}", f"G{index}", 50000 + index)
                for index in range(count)
            ),
            continuation_id=continuation_id,
            choice_kind=ChoiceKind.GRAVEYARD,
            request_id=request_id,
        )

    def test_page_turn_changes_observation_without_core_mutation(self):
        env = _env()
        env.reset(seed=1)
        self._set_request(env, count=25, request_id=1)
        before_observation = env.observation()
        before_events = tuple(env.core.event_history)
        before_request = env.core.state.pending_choice

        result = env.step(env.GRAVEYARD_NEXT_PAGE)

        self.assertNotEqual(before_observation, result.observation)
        self.assertEqual(result.info["graveyard_page"], 1)
        self.assertIs(env.core.state.pending_choice, before_request)
        self.assertEqual(tuple(env.core.event_history), before_events)

    def test_every_option_is_reachable_through_pages(self):
        env = _env()
        env.reset(seed=1)
        self._set_request(env, count=41, request_id=1)
        reachable: set[str] = set()

        while True:
            for command in env.core.legal_commands():
                if env._encode_command(command) is not None:
                    reachable.add(command.option_id)
            mask = env.action_mask()
            if not mask[env.GRAVEYARD_NEXT_PAGE]:
                break
            env.step(env.GRAVEYARD_NEXT_PAGE)

        expected = {
            option.option_id for option in env.core.state.pending_choice.options
        }
        self.assertEqual(reachable, expected)

    def test_new_request_with_same_continuation_resets_page(self):
        env = _env()
        env.reset(seed=1)
        self._set_request(env, count=25, request_id=1)
        env.step(env.GRAVEYARD_NEXT_PAGE)
        self.assertEqual(env.info()["graveyard_page"], 1)

        self._set_request(env, count=25, request_id=2)
        self.assertEqual(env.info()["graveyard_page"], 0)


class SchemaAndInvariantTests(unittest.TestCase):
    def test_graveyard_source_conditions_are_allowed(self):
        for condition in (
            {"type": "source_evolved"},
            {"type": "source_has_keyword", "keyword": "守护"},
        ):
            operation = _parse_operation(
                {
                    "kind": "banish_from_graveyard",
                    "target": "own_graveyard_card",
                    "conditions": [condition],
                },
                "audit.json",
                1,
            )
            self.assertEqual(len(operation.conditions), 1)

    def test_graveyard_target_conditions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "target_health_at_most"):
            _parse_operation(
                {
                    "kind": "banish_from_graveyard",
                    "target": "own_graveyard_card",
                    "conditions": [
                        {"type": "target_health_at_most", "value": 3}
                    ],
                },
                "audit.json",
                1,
            )

    def test_duplicate_entity_across_hand_and_graveyard_is_rejected(self):
        env = _env()
        env.reset(seed=1)
        duplicate_id = env.players[0].hand[0].entity_id
        env.players[0].graveyard.append(
            GraveyardCard(
                definition=_card(500),
                entity_id=duplicate_id,
                owner=0,
                entered_sequence=1,
                entry_cause="test",
            )
        )
        with self.assertRaisesRegex(IllegalCommand, "multiple zones"):
            env.core.legal_commands()


class ScenarioTests(unittest.TestCase):
    def test_graveyard_zone_scenario_is_deterministic(self):
        from scripts.graveyard_zone_scenario import run

        self.assertEqual(run(seed=17), run(seed=17))


if __name__ == "__main__":
    unittest.main()
