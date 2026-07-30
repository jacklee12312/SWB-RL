from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.report_rl_interface_privacy_audit import (
    DEFAULT_MARKDOWN,
    DEFAULT_OUTPUT,
    ROOT,
    build_report,
    render_json,
    render_markdown,
)
from swb.db.repository import CardDefinition
from swb.engine import observation_v4
from swb.engine.card_rules import (
    ActivationDefinition,
    CardRule,
    FusionDefinition,
    RuleBook,
    Trigger,
)
from swb.engine.commands import (
    ChoiceKind,
    CommandType,
    PlayCard,
)
from swb.engine.effects import (
    DeckFilter,
    EffectKind,
    EffectOperation,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv, _PageTurn
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.play_modes import PlayModeDefinition
from swb.engine.state import Amulet, GraveyardCard, HandCard, Unit
from swb.rl.versioning import (
    ACTION_LAYOUT_VERSION,
    OBSERVATION_SCHEMA_VERSIONS,
    action_layout_manifest,
    observation_schema_manifest,
)


def card(
    card_id: int,
    *,
    cost: int = 1,
    card_type: str = "随从",
    attack: int | None = 1,
    life: int | None = 2,
    keywords: frozenset[str] = frozenset(),
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"audit-card-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack,
        life=life,
        keywords=keywords,
        support_level="basic",
        is_collectible=True,
    )


def make_env(
    *,
    rulebook: RuleBook | None = None,
    observation_version: str = "v3",
    enable_mulligan: bool = False,
) -> ShadowverseEnv:
    deck_a = [card(100 + index) for index in range(40)]
    deck_b = [card(200 + index) for index in range(40)]
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=1,
        class_b=1,
        seed=110,
        rulebook=rulebook or RuleBook(),
        observation_version=observation_version,
        card_vocabulary=tuple(range(100, 1000)),
        starting_player=0,
        enable_mulligan=enable_mulligan,
    )
    env.reset(seed=110)
    return env


def replace_hand(env: ShadowverseEnv, definitions: list[CardDefinition]) -> None:
    player = env.players[env.current_player]
    player.hand.clear()
    player.hand_entity_ids.clear()
    for definition in definitions:
        hand_card = HandCard(
            definition=definition,
            entity_id=env.core.state.allocate_entity_id(),
            origin=CardOrigin.DECK,
        )
        player.hand.append(hand_card)
        player.hand_entity_ids.append(hand_card.entity_id)
    env.invalidate_cache(reason="audit hand setup")


def decision_state(env: ShadowverseEnv) -> tuple[object, ...]:
    return (
        env._core.deterministic_fingerprint(),
        env._core.random.getstate(),
        env._graveyard_page,
        env._last_choice_request_key,
        env.state_version,
        env.transition_version,
        env.agent_steps,
    )


class RLInterfacePrivacyAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = build_report(ROOT)

    @staticmethod
    def basic_env() -> ShadowverseEnv:
        env = make_env()
        replace_hand(env, [card(900, cost=1)])
        player = env.players[0]
        opponent = env.players[1]
        player.max_mana = 10
        player.mana = 10
        player.evolution_points = 2
        player.super_evolution_points = 2
        player.turns_started = 10
        player.evolved_this_turn = False
        player.super_evolved_this_turn = False
        player.extra_pp_available = True
        player.extra_pp_uses = 0
        env.core.state.first_player = 1
        attacker = Unit.summon(
            card(901, attack=2, life=5),
            entity_id=env.core.state.allocate_entity_id(),
        )
        attacker.can_attack = True
        defender = Unit.summon(
            card(902, attack=1, life=5),
            entity_id=env.core.state.allocate_entity_id(),
        )
        player.board = [attacker]
        opponent.board = [defender]
        env.invalidate_cache(reason="audit base command setup")
        return env

    @staticmethod
    def activation_env() -> ShadowverseEnv:
        definition = card(
            910,
            card_type="护符",
            attack=None,
            life=None,
            keywords=frozenset({"启动"}),
        )
        rules = RuleBook(
            rules=(
                CardRule(
                    910,
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
            activation_defs={910: ActivationDefinition(910, 1)},
        )
        env = make_env(rulebook=rules)
        player = env.players[0]
        player.max_mana = 10
        player.mana = 10
        player.board = [
            Amulet(
                definition=definition,
                entity_id=env.core.state.allocate_entity_id(),
                entered_turn=env.turn,
                origin=CardOrigin.DECK,
            )
        ]
        env.invalidate_cache(reason="audit activation setup")
        return env

    @staticmethod
    def fusion_env() -> ShadowverseEnv:
        fusion_card = card(
            920,
            cost=2,
            card_type="法术",
            attack=None,
            life=None,
            keywords=frozenset({"融合"}),
        )
        material = card(921)
        rules = RuleBook(
            fusion_defs={
                920: FusionDefinition(
                    card_id=920,
                    material_filter=DeckFilter(class_id=1),
                )
            }
        )
        env = make_env(rulebook=rules)
        replace_hand(env, [fusion_card, material])
        env.players[0].mana = 10
        env.invalidate_cache(reason="audit fusion setup")
        return env

    @staticmethod
    def mode_env() -> ShadowverseEnv:
        definition = card(930, cost=2)
        rules = RuleBook(
            play_modes={
                930: (
                    PlayModeDefinition(
                        "enhance_4",
                        "enhance",
                        4,
                    ),
                )
            }
        )
        env = make_env(rulebook=rules)
        replace_hand(env, [definition])
        env.players[0].mana = 4
        env.invalidate_cache(reason="audit mode setup")
        return env

    def assert_current_round_trips(
        self,
        env: ShadowverseEnv,
        *,
        execute: bool,
    ) -> None:
        commands = tuple(env._core.legal_commands())
        mask = env.action_mask()
        encoded: dict[int, object] = {}
        for command in commands:
            action = env._encode_command(command)
            if action is None:
                self.assertIs(
                    env.core.state.pending_choice.choice_kind,
                    ChoiceKind.GRAVEYARD,
                )
                continue
            self.assertNotIn(action, encoded, (command, encoded.get(action)))
            self.assertTrue(mask[action], (command, action))
            self.assertEqual(env._decode_action(action), command)
            encoded[action] = command

        for action, legal in enumerate(mask):
            if not legal:
                continue
            if action in {
                env.GRAVEYARD_PREV_PAGE,
                env.GRAVEYARD_NEXT_PAGE,
            }:
                clone = env.clone()
                before = clone.core.deterministic_fingerprint()
                clone.step(action)
                self.assertEqual(
                    clone.core.deterministic_fingerprint(),
                    before,
                )
                continue
            decoded = env._decode_action(action)
            self.assertIn(decoded, commands)
            self.assertEqual(encoded[action], decoded)
            if execute:
                clone = env.clone()
                clone.step(action)

    def test_base_command_round_trips_are_unique(self) -> None:
        env = self.basic_env()
        command_types = {
            command.type for command in env.core.legal_commands()
        }
        self.assertTrue({
            CommandType.END_TURN,
            CommandType.PLAY_CARD,
            CommandType.ATTACK,
            CommandType.EVOLVE,
            CommandType.SUPER_EVOLVE,
            CommandType.USE_EXTRA_PP,
        }.issubset(command_types))
        self.assert_current_round_trips(env, execute=False)

    def test_every_true_action_executes_expected_command(self) -> None:
        environments = [
            self.basic_env(),
            self.activation_env(),
            self.fusion_env(),
            self.mode_env(),
            make_env(enable_mulligan=True),
        ]
        for env in environments:
            with self.subTest(
                phase=env.core.state.phase.value,
                commands=[command.type.value for command in env.core.legal_commands()],
            ):
                self.assert_current_round_trips(env, execute=True)

        fusion = self.fusion_env()
        fusion_command = next(
            command
            for command in fusion.core.legal_commands()
            if command.type is CommandType.BEGIN_FUSION
        )
        fusion.step(fusion._encode_command(fusion_command))
        self.assertIs(
            fusion.core.state.pending_choice.choice_kind,
            ChoiceKind.FUSION,
        )
        self.assert_current_round_trips(fusion, execute=True)

    def test_illegal_mask_samples_are_atomic_across_layout_ranges(self) -> None:
        for env in (
            self.basic_env(),
            self.activation_env(),
            self.fusion_env(),
            self.mode_env(),
            make_env(enable_mulligan=True),
        ):
            mask = env.action_mask()
            layout = action_layout_manifest(env)
            samples = []
            for row in layout["ranges"]:
                sample = next(
                    (
                        action
                        for action in range(row["start"], row["stop"])
                        if not mask[action]
                    ),
                    None,
                )
                if sample is not None:
                    samples.append(sample)
            self.assertTrue(samples)
            for action in samples:
                clone = env.clone()
                before = decision_state(clone)
                with self.assertRaisesRegex(ValueError, "Illegal action"):
                    clone.step(action)
                self.assertEqual(decision_state(clone), before)

    def graveyard_env(self) -> ShadowverseEnv:
        spell = card(
            940,
            cost=1,
            card_type="法术",
            attack=None,
            life=None,
        )
        rules = RuleBook(
            rules=(
                CardRule(
                    940,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.BANISH_FROM_GRAVEYARD,
                            TargetKind.OWN_GRAVEYARD_CARD,
                        ),
                    ),
                ),
            ),
        )
        env = make_env(rulebook=rules)
        replace_hand(env, [spell])
        player = env.players[0]
        player.mana = 10
        player.graveyard = [
            GraveyardCard(
                definition=card(500 + index),
                entity_id=env.core.state.allocate_entity_id(),
                owner=0,
                entered_sequence=index + 1,
                entry_cause="destroy",
                origin=CardOrigin.DECK,
            )
            for index in range(41)
        ]
        env.invalidate_cache(reason="audit graveyard setup")
        env.step(env.PLAY_OFFSET)
        self.assertIs(
            env.core.state.pending_choice.choice_kind,
            ChoiceKind.GRAVEYARD,
        )
        return env

    def test_graveyard_pagination_is_complete_unique_and_bounded(self) -> None:
        env = self.graveyard_env()
        expected = {
            option.option_id
            for option in env.core.state.pending_choice.options
        }
        reached: list[str] = []
        pages = 0
        while True:
            pages += 1
            self.assertLessEqual(pages, 3)
            mask = env.action_mask()
            for action in range(
                env.GRAVEYARD_SLOT_OFFSET,
                env.MODE_PLAY_OFFSET,
            ):
                if not mask[action]:
                    continue
                command = env._decode_action(action)
                reached.append(command.option_id)
                clone = env.clone()
                clone.step(action)
            if not mask[env.GRAVEYARD_NEXT_PAGE]:
                break
            before = env.core.deterministic_fingerprint()
            env.step(env.GRAVEYARD_NEXT_PAGE)
            self.assertEqual(env.core.deterministic_fingerprint(), before)

        self.assertEqual(pages, 3)
        self.assertEqual(len(reached), 41)
        self.assertEqual(len(set(reached)), 41)
        self.assertEqual(set(reached), expected)

        reverse_steps = 0
        while env.action_mask()[env.GRAVEYARD_PREV_PAGE]:
            reverse_steps += 1
            self.assertLessEqual(reverse_steps, 2)
            env.step(env.GRAVEYARD_PREV_PAGE)
        self.assertEqual(reverse_steps, 2)
        self.assertEqual(env.info()["graveyard_page"], 0)

    def test_v3_6_and_v4_1_hide_hand_and_deck_identity(self) -> None:
        for version in ("v3", "v4.1"):
            first = make_env(observation_version=version)
            second = make_env(observation_version=version)
            hidden = second.players[1]
            hidden.hand[0].definition = card(999, cost=9)
            hidden.hand[0].add_keyword("疾驰")
            hidden.deck.reverse()
            hidden.deck[0] = card(998, cost=8)
            first._observation_cache.clear()
            second._observation_cache.clear()
            first_observation = first.observation()
            second_observation = second.observation()
            self.assertEqual(
                first_observation.keys(),
                second_observation.keys(),
            )
            for name in first_observation:
                np.testing.assert_array_equal(
                    first_observation[name],
                    second_observation[name],
                    err_msg=f"{version}:{name}",
                )

    def test_public_history_tracks_play_attack_target_and_zone_change(self) -> None:
        spell = card(
            950,
            cost=1,
            card_type="法术",
            attack=None,
            life=None,
        )
        rules = RuleBook(
            rules=(
                CardRule(
                    950,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.DESTROY,
                            TargetKind.ENEMY_BOARD,
                        ),
                    ),
                ),
            ),
        )
        env = make_env(rulebook=rules, observation_version="v4.1")
        replace_hand(env, [spell])
        env.players[0].mana = 10
        target = Unit.summon(
            card(951, attack=1, life=2),
            entity_id=env.core.state.allocate_entity_id(),
        )
        env.players[1].board = [target]
        env.invalidate_cache(reason="audit public history spell setup")
        env.step(env.PLAY_OFFSET)
        choice_action = next(
            action
            for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        )
        env.step(choice_action)
        observation = env.observation()
        event_types = set(int(value) for value in observation["history_event_types"])
        self.assertIn(
            observation_v4.EVENT_INDEX[EventType.CARD_PLAYED] + 1,
            event_types,
        )
        self.assertIn(
            observation_v4.EVENT_INDEX[EventType.FOLLOWER_DESTROYED] + 1,
            event_types,
        )
        self.assertGreater(int(observation["history_source_cards"].max()), 0)

        combat = make_env(observation_version="v4.1")
        attacker = Unit.summon(
            card(960, attack=2, life=5),
            entity_id=combat.core.state.allocate_entity_id(),
        )
        attacker.can_attack = True
        defender = Unit.summon(
            card(961, attack=1, life=5),
            entity_id=combat.core.state.allocate_entity_id(),
        )
        combat.players[0].board = [attacker]
        combat.players[1].board = [defender]
        combat.invalidate_cache(reason="audit public history combat setup")
        attack_action = combat._encode_command(
            next(
                command
                for command in combat.core.legal_commands()
                if (
                    command.type is CommandType.ATTACK
                    and command.target_id == defender.entity_id
                )
            )
        )
        combat.step(attack_action)
        combat_observation = combat.observation()
        attack_type = observation_v4.EVENT_INDEX[EventType.ATTACK_DECLARED] + 1
        attack_positions = np.flatnonzero(
            combat_observation["history_event_types"] == attack_type
        )
        self.assertTrue(attack_positions.size)
        self.assertGreater(
            int(
                combat_observation["history_target_references"][
                    attack_positions[-1]
                ]
            ),
            0,
        )

    def test_formal_observation_manifests_match_live_spaces(self) -> None:
        expected = {
            "v3": OBSERVATION_SCHEMA_VERSIONS["v3"],
            "v4.1": OBSERVATION_SCHEMA_VERSIONS["v4.1"],
        }
        for version, formal_version in expected.items():
            env = make_env(observation_version=version)
            observation = env.observation()
            space = (
                env.observation_v3_space()
                if version == "v3"
                else env.observation_v4_1_space()
            )
            manifest = observation_schema_manifest(env)
            self.assertEqual(manifest["version"], formal_version)
            self.assertTrue(space.contains(observation))
            self.assertEqual(
                {
                    row["name"]: (tuple(row["shape"]), row["dtype"])
                    for row in manifest["fields"]
                },
                {
                    name: (field.shape, str(field.dtype))
                    for name, field in space.spaces.items()
                },
            )
        self.assertEqual(
            self.report["summary"]["action_layout_version"],
            ACTION_LAYOUT_VERSION,
        )

    def test_migration_decisions_are_explicit_and_non_schema_changes(self) -> None:
        decisions = self.report["migration_decisions"]
        self.assertEqual(
            {decision["change"] for decision in decisions},
            {"60d1c2f", "82bd251"},
        )
        for decision in decisions:
            self.assertFalse(decision["observation_fields_changed"])
            self.assertFalse(decision["action_layout_changed"])
            self.assertFalse(decision["migration_required"])
            self.assertTrue(decision["reason"])

    def test_report_has_no_failures(self) -> None:
        self.assertTrue(self.report["summary"]["passed"])
        self.assertEqual(self.report["failures"], [])
        self.assertEqual(
            self.report["summary"]["command_type_count"],
            len(CommandType),
        )
        self.assertEqual(
            self.report["summary"]["checklist_contract_count"],
            10,
        )

    def test_saved_reports_match_deterministic_generation(self) -> None:
        self.assertEqual(
            (ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"),
            render_json(self.report),
        )
        self.assertEqual(
            (ROOT / DEFAULT_MARKDOWN).read_text(encoding="utf-8"),
            render_markdown(self.report),
        )


if __name__ == "__main__":
    unittest.main()
