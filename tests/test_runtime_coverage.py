from __future__ import annotations

import copy
import pickle
import unittest
from unittest.mock import patch

from swb.db.repository import CardDefinition
from swb.engine.abilities import (
    AbilityContext,
    AbilityEvent,
    AbilityKeyword,
)
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import EndTurn, PlayCard
from swb.engine.effects import (
    Condition,
    ConditionType,
    EffectKind,
    EffectOperation,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType, GameEvent
from swb.engine.play_modes import PlayModeDefinition
from swb.engine.resolution import (
    MAX_RESOLUTION_STEPS,
    GameConfig,
    GameEngine,
    IllegalCommand,
)
from swb.engine.runtime_coverage import (
    aggregate_runtime_coverage,
    render_runtime_coverage_markdown,
)
from swb.engine.state import ResolutionLoopError, Unit


def card(
    card_id: int,
    *,
    card_type: str = "随从",
    cost: int = 1,
    keywords: frozenset[str] = frozenset(),
    ability_keywords: frozenset[AbilityKeyword] = frozenset(),
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="elf",
        name=f"card-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=None if card_type != "随从" else 1,
        life=None if card_type != "随从" else 2,
        keywords=keywords,
        support_level="basic",
        is_collectible=True,
        ability_keywords=ability_keywords,
    )


def audit_rulebook() -> RuleBook:
    return RuleBook(rules=(
        CardRule(
            1001,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.DRAW,
                    TargetKind.OWN_LEADER,
                    amount=1,
                    conditions=(
                        Condition(
                            ConditionType.CONTROLLER_HEALTH_AT_LEAST,
                            20,
                        ),
                    ),
                ),
                EffectOperation(
                    EffectKind.DRAW,
                    TargetKind.OWN_LEADER,
                    amount=1,
                    conditions=(
                        Condition(
                            ConditionType.CONTROLLER_HEALTH_AT_MOST,
                            1,
                        ),
                    ),
                ),
                EffectOperation(
                    EffectKind.DAMAGE_UNIT,
                    TargetKind.RANDOM_ENEMY_UNIT,
                    amount=1,
                ),
                EffectOperation(
                    EffectKind.SUMMON,
                    TargetKind.OWN_LEADER,
                    amount=1,
                    card_id=2001,
                ),
            ),
        ),
    ))


def make_engine(*, audit: bool) -> GameEngine:
    spell = card(1001, card_type="法术")
    follower = card(2001)
    definitions = {spell.card_id: spell, follower.card_id: follower}
    engine = GameEngine(
        [spell] * 40,
        [follower] * 40,
        class_a=1,
        class_b=1,
        seed=11,
        rulebook=audit_rulebook(),
        card_resolver=definitions.get,
        config=GameConfig(audit_runtime_coverage=audit),
    )
    engine.reset(seed=11)
    return engine


class RuntimeCoverageTests(unittest.TestCase):
    def test_real_resolution_records_clause_branches_targets_and_capacity(self) -> None:
        engine = make_engine(audit=True)
        engine.players[0].mana = 10
        engine.players[0].board = [
            Unit.summon(
                card(3000 + index),
                entity_id=engine.state.allocate_entity_id(),
            )
            for index in range(engine.config.max_board)
        ]

        engine.apply(PlayCard(0, 0))
        session = engine.runtime_coverage.to_session(card_ids={1001})
        clauses = {row["clause_id"]: row for row in session["clauses"]}
        counts = [row["counts"] for row in clauses.values()]

        self.assertEqual(len(clauses), 4)
        self.assertTrue(any(row.get("condition_true", 0) for row in counts))
        self.assertTrue(any(row.get("condition_false", 0) for row in counts))
        self.assertTrue(any(row.get("no_target", 0) for row in counts))
        self.assertTrue(any(row.get("capacity_shortage", 0) for row in counts))
        self.assertEqual(
            {row["status"] for row in clauses.values()},
            {"triggered_passed", "triggered_not_executed"},
        )
        self.assertTrue(
            all(
                str(clause_id).encode("ascii")
                for clause_id in clauses
            )
        )
        lifecycle = {
            (row["card_id"], row["event"]): row["count"]
            for row in session["lifecycle"]
        }
        self.assertEqual(lifecycle[(1001, "played")], 1)
        random_target = next(
            row
            for row in session["targets"]
            if row["target_kind"] == "random_enemy_unit"
        )
        self.assertEqual(random_target["counts"]["candidate_min"], 0)
        self.assertEqual(random_target["counts"]["no_target"], 1)

    def test_lifecycle_and_alternate_modes_use_structured_events(self) -> None:
        engine = make_engine(audit=True)
        recorder = engine.runtime_coverage
        event_types = (
            EventType.CARD_DRAWN,
            EventType.CARD_PLAYED,
            EventType.FOLLOWER_EVOLVED,
            EventType.FOLLOWER_SUPER_EVOLVED,
            EventType.ATTACK_DECLARED,
            EventType.ENTITY_LEFT_PLAY,
        )
        for event_type in event_types:
            recorder.record_event(
                GameEvent(
                    event_type,
                    0,
                    metadata={"card_id": 1001},
                )
            )
        for event_type, metadata in (
            (EventType.CARD_PLAYED, {"card_id": 1001, "mode_id": "enhance_7"}),
            (EventType.CARD_FUSED, {"fusion_card_id": 1001}),
            (EventType.CARD_INVOKED, {"card_id": 1001}),
            (EventType.AMULET_ACTIVATED, {"card_id": 1001}),
            (
                EventType.UNION_BURST_ACTIVATED,
                {"card_id": 1001, "kind": "union_burst"},
            ),
            (EventType.MODE_SELECTED, {"source_card_id": 1001}),
        ):
            recorder.record_event(GameEvent(event_type, 0, metadata=metadata))
        session = recorder.to_session(card_ids={1001})

        self.assertEqual(
            {row["event"] for row in session["lifecycle"]},
            {
                "drawn",
                "played",
                "evolved",
                "super_evolved",
                "attacked",
                "left_play",
            },
        )
        self.assertEqual(len(session["alternate_modes"]), 6)

    def test_stable_clause_ids_survive_equivalent_rulebook_reload(self) -> None:
        first = make_engine(audit=True).runtime_coverage.to_session(
            card_ids={1001}
        )
        second = make_engine(audit=True).runtime_coverage.to_session(
            card_ids={1001}
        )
        self.assertEqual(
            [row["clause_id"] for row in first["clauses"]],
            [row["clause_id"] for row in second["clauses"]],
        )

    def test_stable_clause_id_survives_snapshot_operation_round_trip(self) -> None:
        engine = make_engine(audit=True)
        operations = engine.rulebook.operations_for(1001, Trigger.PLAY)
        restored = pickle.loads(pickle.dumps(operations))
        expected = engine.runtime_coverage.resolve_clause_id(
            1001,
            operations[0],
            operations,
        )

        self.assertEqual(
            engine.runtime_coverage.resolve_clause_id(
                1001,
                restored[0],
                restored,
            ),
            expected,
        )

    def test_diagnostics_capture_all_required_failure_classes(self) -> None:
        engine = make_engine(audit=True)
        with self.assertRaises(IllegalCommand):
            engine.apply(EndTurn(1))
        placeholder = card(
            4001,
            ability_keywords=frozenset({AbilityKeyword.COOPERATION}),
        )
        engine.ability_handlers.dispatch(
            AbilityContext(
                AbilityEvent.CHECK_PLAY,
                0,
                source=placeholder,
            )
        )
        engine.state.resolution_steps = MAX_RESOLUTION_STEPS
        with self.assertRaises(ResolutionLoopError):
            engine._step()

        deck_a = [card(5000 + index) for index in range(40)]
        deck_b = [card(6000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=3,
            audit_runtime_coverage=True,
        )
        env.reset(seed=3)
        with self.assertRaises(ValueError):
            env.step(env.ACTION_SIZE)
        env.invalidate_cache(reason="mask audit test")
        with patch.object(env, "_encode_command", return_value=None):
            env.action_mask()

        engine_diagnostics = engine.runtime_coverage.to_session()[
            "diagnostics"
        ]
        env_diagnostics = env.runtime_coverage.to_session()["diagnostics"]
        self.assertEqual(engine_diagnostics["illegal_command"]["total"], 1)
        self.assertEqual(engine_diagnostics["placeholder"]["total"], 1)
        self.assertEqual(engine_diagnostics["unsupported"]["total"], 1)
        self.assertEqual(
            engine_diagnostics["resolution_step_limit"]["total"],
            1,
        )
        self.assertEqual(env_diagnostics["illegal_action"]["total"], 1)
        self.assertEqual(
            env_diagnostics["action_mask_mismatch"]["total"],
            1,
        )

    def test_structured_modes_do_not_emit_false_placeholder_diagnostics(self) -> None:
        enhanced = card(
            7001,
            ability_keywords=frozenset({AbilityKeyword.ENHANCE}),
        )
        opponent = card(7002)
        rulebook = RuleBook(
            play_modes={
                7001: (
                    PlayModeDefinition(
                        mode_id="enhance_4",
                        mode_type="enhance",
                        cost=4,
                    ),
                ),
            },
        )
        engine = GameEngine(
            [enhanced] * 40,
            [opponent] * 40,
            class_a=1,
            class_b=1,
            seed=5,
            rulebook=rulebook,
            config=GameConfig(audit_runtime_coverage=True),
        )
        engine.reset(seed=5)
        engine.ability_handlers.dispatch(
            AbilityContext(
                AbilityEvent.CHECK_PLAY,
                0,
                source=enhanced,
            )
        )
        diagnostics = engine.runtime_coverage.to_session()["diagnostics"]
        self.assertEqual(diagnostics["placeholder"], {})

        unstructured = card(
            7003,
            ability_keywords=frozenset({AbilityKeyword.ENHANCE}),
        )
        engine.ability_handlers.dispatch(
            AbilityContext(
                AbilityEvent.CHECK_PLAY,
                0,
                source=unstructured,
            )
        )
        diagnostics = engine.runtime_coverage.to_session()["diagnostics"]
        self.assertEqual(diagnostics["placeholder"]["total"], 1)

    def test_audit_mode_does_not_change_deterministic_engine_state(self) -> None:
        normal = make_engine(audit=False)
        audited = make_engine(audit=True)
        self.assertEqual(
            normal.deterministic_fingerprint(),
            audited.deterministic_fingerprint(),
        )
        normal.apply(EndTurn(0))
        audited.apply(EndTurn(0))
        self.assertEqual(
            normal.deterministic_fingerprint(),
            audited.deterministic_fingerprint(),
        )
        self.assertIsNone(normal.runtime_coverage)
        self.assertIsNotNone(audited.runtime_coverage)

    def test_aggregations_and_markdown_keep_not_triggered_distinct(self) -> None:
        engine = make_engine(audit=True)
        engine.runtime_coverage.set_context(
            matchup_id="deck-a__vs__deck-b",
        )
        session = engine.runtime_coverage.to_session(card_ids={1001})
        report = aggregate_runtime_coverage(
            [session],
            deck_memberships={1001: ["deck-a"]},
        )

        self.assertEqual(
            report["summary"]["clause_status_counts"]["not_triggered"],
            4,
        )
        for dimension in (
            "by_card",
            "by_mechanic",
            "by_deck",
            "by_matchup",
        ):
            self.assertTrue(report["aggregations"][dimension])
        markdown = render_runtime_coverage_markdown(report)
        self.assertIn("Not triggered: 4", markdown)
        self.assertIn("Triggered and passed: 0", markdown)
        self.assertIn("不能作为", markdown)

    def test_aggregation_merges_each_clause_across_sessions(self) -> None:
        engine = make_engine(audit=True)
        engine.runtime_coverage.set_context(
            matchup_id="deck-a__vs__deck-b",
        )
        not_triggered = engine.runtime_coverage.to_session(card_ids={1001})
        triggered = copy.deepcopy(not_triggered)
        triggered["clauses"][0]["status"] = "triggered_passed"
        triggered["clauses"][0]["counts"] = {
            "entered": 1,
            "operation_executed": 1,
        }

        report = aggregate_runtime_coverage(
            [not_triggered, triggered],
            deck_memberships={1001: ["deck-a"]},
        )

        self.assertEqual(
            report["summary"]["clause_status_counts"],
            {"not_triggered": 3, "triggered_passed": 1},
        )
        matchup = report["aggregations"]["by_matchup"][0]["counts"]
        self.assertEqual(matchup["not_triggered"], 3)
        self.assertEqual(matchup["triggered_passed"], 1)
        self.assertEqual(
            matchup["not_triggered"] + matchup["triggered_passed"],
            4,
        )


if __name__ == "__main__":
    unittest.main()
