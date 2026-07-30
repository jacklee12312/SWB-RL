from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.card_audit_sampling import (
    SAMPLING_POLICY,
    SAMPLING_RANDOM,
    build_fixed_matrix_specs,
    build_full_pool_specs,
)
from swb.db.repository import CardDefinition
from swb.engine.abilities import (
    AbilityContext,
    AbilityEvent,
    AbilityKeyword,
)
from swb.engine.card_rules import CardPassive, CardRule, RuleBook, Trigger
from swb.engine.commands import Attack
from swb.engine.effects import (
    Condition,
    ConditionType,
    EffectKind,
    EffectOperation,
    ExprType,
    HandFilter,
    TargetKind,
    ValueExpression,
)
from swb.engine.emblem import (
    EmblemDefinition,
    EmblemPassive,
    EmblemTriggerRule,
)
from swb.engine.events import EventType
from swb.engine.faith import (
    FaithDefinition,
    FaithTrigger,
    FaithTriggerRule,
)
from swb.engine.forced_scenarios import run_minimal_forced_scenarios
from swb.engine.listeners import CardListenerDefinition, ListenerZone
from swb.engine.play_modes import PlayModeDefinition
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Unit


REPORT = Path(
    "data/reports/card_bug_audit/forced_scenario_audit.json"
)
DISTRIBUTION_REPORT = Path(
    "data/reports/card_bug_audit/"
    "long_truncation_myuu_distribution.json"
)


def card(
    card_id: int,
    *,
    abilities: frozenset[AbilityKeyword] = frozenset(),
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"audit-{card_id}",
        cost=1,
        card_type="随从",
        attack=2,
        life=2,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
        ability_keywords=abilities,
    )


class ForcedScenarioAndSamplingTests(unittest.TestCase):
    def test_minimum_public_interface_fixtures_pass_with_invariants(self) -> None:
        results = run_minimal_forced_scenarios()

        self.assertEqual(len(results), 9)
        self.assertEqual(
            {result.category for result in results},
            {
                "cost",
                "target",
                "capacity",
                "resource",
                "ordinary_evolution",
                "super_evolution",
                "turn_start",
                "turn_end",
                "simultaneous_death",
            },
        )
        self.assertTrue(all(result.status == "passed" for result in results))
        self.assertTrue(
            all(
                result.invariant_checks > result.direct_state_mutations
                for result in results
            )
        )

    def test_fixed_matrix_covers_every_ordered_deck_pair_with_both_policies(
        self,
    ) -> None:
        specs = build_fixed_matrix_specs(master_seed=120012)

        self.assertEqual(len(specs), 1024)
        self.assertEqual(
            Counter(spec.sampling_kind for spec in specs),
            Counter({SAMPLING_RANDOM: 960, SAMPLING_POLICY: 64}),
        )
        self.assertEqual(
            len({
                (
                    spec.deck_a_name,
                    spec.deck_b_name,
                    spec.sampling_kind,
                )
                for spec in specs
            }),
            128,
        )
        self.assertTrue(all(spec.verify_replay for spec in specs))

    def test_full_pool_schedule_is_exactly_ten_thousand_and_stratified(
        self,
    ) -> None:
        specs = build_full_pool_specs(master_seed=120012)

        self.assertEqual(len(specs), 10_000)
        self.assertEqual(
            Counter(spec.sampling_kind for spec in specs),
            Counter({SAMPLING_RANDOM: 9804, SAMPLING_POLICY: 196}),
        )
        self.assertEqual(
            len({
                (spec.class_a, spec.class_b, spec.sampling_kind)
                for spec in specs
            }),
            98,
        )
        self.assertEqual(sum(spec.verify_replay for spec in specs), 98)

    def test_structured_last_words_does_not_emit_placeholder(self) -> None:
        source = card(
            99100001,
            abilities=frozenset({AbilityKeyword.LAST_WORDS}),
        )
        rulebook = RuleBook(rules=(
            CardRule(
                source.card_id,
                Trigger.LAST_WORDS,
                (
                    EffectOperation(
                        EffectKind.DAMAGE_LEADER,
                        TargetKind.ENEMY_LEADER,
                        amount=1,
                    ),
                ),
            ),
        ))
        deck_a = [source, *[card(99100100 + index) for index in range(39)]]
        deck_b = [card(99100200 + index) for index in range(40)]
        engine = GameEngine(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=5,
            rulebook=rulebook,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=5)
        attacker = Unit.summon(
            source,
            entity_id=engine.state.allocate_entity_id(),
        )
        attacker.can_attack = True
        attacker.summoned_this_turn = False
        defender = Unit.summon(
            card(99100002),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board = [attacker]
        engine.players[1].board = [defender]
        engine.assert_invariants()

        engine.apply(Attack(0, attacker.entity_id, defender.entity_id))

        self.assertEqual(engine.players[1].health, 19)
        self.assertEqual(
            [
                event
                for event in engine.placeholder_ability_events
                if event.card_id == source.card_id
            ],
            [],
        )

    def test_unstructured_last_words_remains_explicit_placeholder(self) -> None:
        source = card(
            99100003,
            abilities=frozenset({AbilityKeyword.LAST_WORDS}),
        )
        deck_a = [source, *[card(99100300 + index) for index in range(39)]]
        deck_b = [card(99100400 + index) for index in range(40)]
        engine = GameEngine(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=6,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=6)
        attacker = Unit.summon(
            source,
            entity_id=engine.state.allocate_entity_id(),
        )
        attacker.can_attack = True
        attacker.summoned_this_turn = False
        defender = Unit.summon(
            card(99100004),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board = [attacker]
        engine.players[1].board = [defender]
        engine.assert_invariants()

        engine.apply(Attack(0, attacker.entity_id, defender.entity_id))

        self.assertEqual(
            [
                (event.card_id, event.ability)
                for event in engine.placeholder_ability_events
            ],
            [(source.card_id, AbilityKeyword.LAST_WORDS)],
        )

    def test_emblem_last_words_source_does_not_mark_follower_unsupported(
        self,
    ) -> None:
        source = card(
            99100005,
            abilities=frozenset({AbilityKeyword.LAST_WORDS}),
        )
        emblem = EmblemDefinition(
            emblem_id="audit-emblem",
            source_card_id=source.card_id,
            countdown=1,
            on_expire=(
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    amount=1,
                ),
            ),
        )
        deck_a = [source, *[card(99100500 + index) for index in range(39)]]
        deck_b = [card(99100600 + index) for index in range(40)]
        engine = GameEngine(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=7,
            rulebook=RuleBook(emblem_defs={emblem.emblem_id: emblem}),
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=7)
        attacker = Unit.summon(
            source,
            entity_id=engine.state.allocate_entity_id(),
        )
        attacker.can_attack = True
        attacker.summoned_this_turn = False
        defender = Unit.summon(
            card(99100006),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board = [attacker]
        engine.players[1].board = [defender]
        engine.assert_invariants()

        engine.apply(Attack(0, attacker.entity_id, defender.entity_id))

        self.assertEqual(engine.placeholder_ability_events, [])

    @unittest.skipUnless(REPORT.is_file(), "forced scenario report unavailable")
    def test_saved_forced_scenario_report_covers_full_catalog(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        summary = report["summary"]

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["fixed_deck_count"], 8)
        self.assertEqual(summary["training_closure_card_count"], 147)
        self.assertEqual(summary["collectible_card_count"], 735)
        self.assertEqual(summary["generated_card_count"], 91)
        self.assertEqual(summary["unexplained_runtime_clause_count"], 0)
        self.assertEqual(summary["missing_test_file_count"], 0)
        self.assertTrue(
            all(
                row["all_applicable_scenarios_passed"]
                for row in report["fixed_decks"]
            )
        )

    @unittest.skipUnless(
        DISTRIBUTION_REPORT.is_file(),
        "long/truncation/Myuu report unavailable",
    )
    def test_saved_long_truncation_myuu_report_is_reproducible(self) -> None:
        report = json.loads(
            DISTRIBUTION_REPORT.read_text(encoding="utf-8")
        )

        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(report["summary"]["source_games"], 11_024)
        self.assertGreater(report["summary"]["long_games"], 0)
        self.assertEqual(report["summary"]["truncations"], 0)
        self.assertEqual(report["summary"]["myuu_games"], 240)
        self.assertEqual(report["summary"]["myuu_truncations"], 0)
        self.assertTrue(report["long_games"]["reproductions"])
        self.assertTrue(report["myuu"]["reproductions"])
        self.assertTrue(all(
            row["action_trace_sha256"]
            and row["final_fingerprint_sha256"]
            for row in (
                report["long_games"]["reproductions"]
                + report["myuu"]["reproductions"]
            )
        ))


class StructuredAbilityCoverageTests(unittest.TestCase):
    def assert_dispatch_is_covered(
        self,
        *,
        source: CardDefinition,
        rulebook: RuleBook,
        event: AbilityEvent,
    ) -> None:
        deck_a = [
            source,
            *[card(99200100 + index) for index in range(39)],
        ]
        deck_b = [card(99200200 + index) for index in range(40)]
        engine = GameEngine(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=8,
            rulebook=rulebook,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=8)

        engine.ability_handlers.dispatch(AbilityContext(
            event=event,
            player_index=0,
            source=source,
        ))

        self.assertEqual(engine.placeholder_ability_events, [])

    def test_spellboost_expression_and_condition_are_structured_coverage(
        self,
    ) -> None:
        source = card(
            99200001,
            abilities=frozenset({AbilityKeyword.SPELLBOOST}),
        )
        rulebook = RuleBook(rules=(
            CardRule(
                source.card_id,
                Trigger.PLAY,
                (
                    EffectOperation(
                        EffectKind.DAMAGE_LEADER,
                        TargetKind.ENEMY_LEADER,
                        conditions=(
                            Condition(
                                ConditionType.SOURCE_SPELLBOOST_COUNT_AT_LEAST,
                                value=1,
                            ),
                        ),
                        amount_expr=ValueExpression(
                            ExprType.SOURCE_SPELLBOOST_COUNT
                        ),
                    ),
                ),
            ),
        ))

        self.assert_dispatch_is_covered(
            source=source,
            rulebook=rulebook,
            event=AbilityEvent.CARD_PLAYED,
        )

    def test_cooperation_condition_is_structured_coverage(self) -> None:
        source = card(
            99200002,
            abilities=frozenset({AbilityKeyword.COOPERATION}),
        )
        rulebook = RuleBook(rules=(
            CardRule(
                source.card_id,
                Trigger.FANFARE,
                (
                    EffectOperation(
                        EffectKind.CONDITIONAL,
                        TargetKind.OWN_LEADER,
                        conditions=(
                            Condition(
                                ConditionType.CONTROLLER_COOPERATION_AT_LEAST,
                                value=10,
                            ),
                        ),
                    ),
                ),
            ),
        ))

        self.assert_dispatch_is_covered(
            source=source,
            rulebook=rulebook,
            event=AbilityEvent.CHECK_PLAY,
        )

    def test_faith_mode_and_enhance_provenance_are_covered(self) -> None:
        cases = (
            (
                AbilityKeyword.CHOOSE,
                FaithTrigger.MODE_SELECTED,
                99200003,
            ),
            (
                AbilityKeyword.ENHANCE,
                FaithTrigger.CARD_ENHANCED,
                99200004,
            ),
        )
        for ability, trigger, card_id in cases:
            with self.subTest(ability=ability):
                source = card(
                    card_id,
                    abilities=frozenset({ability}),
                )
                faith = FaithDefinition(
                    faith_id=f"audit-faith-{card_id}",
                    source_card_id=card_id,
                    triggers=(FaithTriggerRule(trigger=trigger),),
                )
                self.assert_dispatch_is_covered(
                    source=source,
                    rulebook=RuleBook(faith_defs={card_id: faith}),
                    event=AbilityEvent.CHECK_PLAY,
                )

    def test_operation_faith_id_is_structured_coverage(self) -> None:
        source = card(
            99200005,
            abilities=frozenset({AbilityKeyword.FAITH}),
        )
        rulebook = RuleBook(rules=(
            CardRule(
                source.card_id,
                Trigger.PLAY,
                (
                    EffectOperation(
                        EffectKind.RANDOM_DISTRIBUTE,
                        TargetKind.OWN_LEADER,
                        faith_id="audit-faith",
                        random_distribution_operations=((
                            EffectOperation(
                                EffectKind.DAMAGE_LEADER,
                                TargetKind.ENEMY_LEADER,
                                amount=1,
                            ),
                        ),),
                    ),
                ),
            ),
        ))

        self.assert_dispatch_is_covered(
            source=source,
            rulebook=rulebook,
            event=AbilityEvent.CARD_PLAYED,
        )

    def test_play_mode_listener_and_emblem_sources_are_covered(self) -> None:
        reanimate = card(
            99200006,
            abilities=frozenset({AbilityKeyword.REANIMATE}),
        )
        reanimate_mode = PlayModeDefinition(
            mode_id="enhance_8",
            mode_type="enhance",
            cost=8,
            operations=(
                EffectOperation(
                    EffectKind.REANIMATE,
                    TargetKind.OWN_LEADER,
                    amount=9,
                ),
            ),
        )
        earth_rite = card(
            99200007,
            abilities=frozenset({AbilityKeyword.EARTH_RITE}),
        )
        earth_listener = CardListenerDefinition(
            card_id=earth_rite.card_id,
            zone=ListenerZone.HAND,
            event=EventType.EARTH_RITE_ACTIVATED,
            operations=(
                EffectOperation(
                    EffectKind.EARTH_RITE,
                    TargetKind.OWN_LEADER,
                    amount=1,
                ),
            ),
        )
        combo = card(
            99200008,
            abilities=frozenset({AbilityKeyword.COMBO}),
        )
        combo_emblem = EmblemDefinition(
            emblem_id="audit-combo-emblem",
            source_card_id=combo.card_id,
            triggers=(
                EmblemTriggerRule(
                    trigger=EventType.TURN_ENDED.value,
                    conditions=(
                        Condition(
                            ConditionType.CONTROLLER_COMBO_AT_LEAST,
                            value=3,
                        ),
                    ),
                ),
            ),
        )
        cases = (
            (
                reanimate,
                RuleBook(
                    play_modes={
                        reanimate.card_id: (reanimate_mode,),
                    },
                ),
                AbilityEvent.CARD_PLAYED,
            ),
            (
                earth_rite,
                RuleBook(
                    listener_defs={
                        earth_rite.card_id: (earth_listener,),
                    },
                ),
                AbilityEvent.CARD_PLAYED,
            ),
            (
                combo,
                RuleBook(
                    emblem_defs={
                        combo_emblem.emblem_id: combo_emblem,
                    },
                ),
                AbilityEvent.CHECK_PLAY,
            ),
        )
        for source, rulebook, event in cases:
            with self.subTest(ability=next(iter(source.abilities))):
                self.assert_dispatch_is_covered(
                    source=source,
                    rulebook=rulebook,
                    event=event,
                )

    def test_spellboost_passive_and_listener_event_are_covered(self) -> None:
        passive_source = card(
            99200009,
            abilities=frozenset({AbilityKeyword.SPELLBOOST}),
        )
        listener_source = card(
            99200010,
            abilities=frozenset({AbilityKeyword.SPELLBOOST}),
        )
        listener = CardListenerDefinition(
            card_id=listener_source.card_id,
            zone=ListenerZone.HAND,
            event=EventType.SPELLBOOSTED,
            operations=(
                EffectOperation(
                    EffectKind.BUFF_HAND_CARD,
                    TargetKind.SELF,
                    amount=1,
                ),
            ),
        )
        cases = (
            (
                passive_source,
                RuleBook(passives=(
                    CardPassive(
                        passive_source.card_id,
                        "spellboost_cost_reduction",
                        1,
                    ),
                )),
            ),
            (
                listener_source,
                RuleBook(listener_defs={
                    listener_source.card_id: (listener,),
                }),
            ),
        )
        for source, rulebook in cases:
            with self.subTest(card_id=source.card_id):
                self.assert_dispatch_is_covered(
                    source=source,
                    rulebook=rulebook,
                    event=AbilityEvent.CARD_PLAYED,
                )

    def test_fusion_listener_and_emblem_event_sources_are_covered(self) -> None:
        listener_source = card(
            99200011,
            abilities=frozenset({AbilityKeyword.FUSION}),
        )
        listener = CardListenerDefinition(
            card_id=listener_source.card_id,
            zone=ListenerZone.HAND,
            event=EventType.CARD_FUSED,
            operations=(
                EffectOperation(
                    EffectKind.BUFF_HAND_CARD,
                    TargetKind.SELF,
                    amount=1,
                ),
            ),
        )
        emblem_source = card(
            99200012,
            abilities=frozenset({AbilityKeyword.FUSION}),
        )
        emblem = EmblemDefinition(
            emblem_id="audit-fusion-emblem",
            source_card_id=emblem_source.card_id,
            triggers=(
                EmblemTriggerRule(
                    trigger=EventType.CARD_FUSED.value,
                    operations=(
                        EffectOperation(
                            EffectKind.DRAW,
                            TargetKind.OWN_LEADER,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )
        cases = (
            (
                listener_source,
                RuleBook(listener_defs={
                    listener_source.card_id: (listener,),
                }),
            ),
            (
                emblem_source,
                RuleBook(emblem_defs={emblem.emblem_id: emblem}),
            ),
        )
        for source, rulebook in cases:
            with self.subTest(card_id=source.card_id):
                self.assert_dispatch_is_covered(
                    source=source,
                    rulebook=rulebook,
                    event=AbilityEvent.CHECK_PLAY,
                )

    def test_expression_and_keyword_filter_references_are_covered(self) -> None:
        combo = card(
            99200013,
            abilities=frozenset({AbilityKeyword.COMBO}),
        )
        combo_rule = CardRule(
            combo.card_id,
            Trigger.FANFARE,
            (
                EffectOperation(
                    EffectKind.DRAW_FILTERED,
                    TargetKind.OWN_LEADER,
                    amount=1,
                    deck_filter_cost_expr=ValueExpression(
                        ExprType.CONTROLLER_COMBO,
                    ),
                ),
            ),
        )
        last_words = card(
            99200014,
            abilities=frozenset({AbilityKeyword.LAST_WORDS}),
        )
        last_words_rule = CardRule(
            last_words.card_id,
            Trigger.FANFARE,
            (
                EffectOperation(
                    EffectKind.SUMMON_DESTROYED_AMULETS,
                    TargetKind.OWN_LEADER,
                    amount=2,
                    history_filter=HandFilter(
                        card_type="护符",
                        keyword=AbilityKeyword.LAST_WORDS.value,
                    ),
                ),
            ),
        )
        cases = (
            (
                combo,
                RuleBook(rules=(combo_rule,)),
                AbilityEvent.CHECK_PLAY,
            ),
            (
                last_words,
                RuleBook(rules=(last_words_rule,)),
                AbilityEvent.FOLLOWER_DESTROYED,
            ),
        )
        for source, rulebook, event in cases:
            with self.subTest(ability=next(iter(source.abilities))):
                self.assert_dispatch_is_covered(
                    source=source,
                    rulebook=rulebook,
                    event=event,
                )

    def test_granted_last_words_and_emblem_enhance_passive_are_covered(
        self,
    ) -> None:
        last_words = card(
            99200015,
            abilities=frozenset({AbilityKeyword.LAST_WORDS}),
        )
        grant_rule = CardRule(
            last_words.card_id,
            Trigger.EVOLVE,
            (
                EffectOperation(
                    EffectKind.GRANT_LAST_WORDS,
                    TargetKind.ALL_OWN_UNITS,
                    granted_operations=(
                        EffectOperation(
                            EffectKind.DAMAGE_LEADER,
                            TargetKind.ENEMY_LEADER,
                            amount=2,
                        ),
                    ),
                ),
            ),
        )
        enhance = card(
            99200016,
            abilities=frozenset({AbilityKeyword.ENHANCE}),
        )
        emblem = EmblemDefinition(
            emblem_id="audit-enhance-passive",
            source_card_id=enhance.card_id,
            passives=frozenset({
                EmblemPassive.SUPPRESS_FOLLOWER_ENHANCE,
            }),
        )
        cases = (
            (
                last_words,
                RuleBook(rules=(grant_rule,)),
                AbilityEvent.FOLLOWER_DESTROYED,
            ),
            (
                enhance,
                RuleBook(emblem_defs={emblem.emblem_id: emblem}),
                AbilityEvent.CHECK_PLAY,
            ),
        )
        for source, rulebook, event in cases:
            with self.subTest(ability=next(iter(source.abilities))):
                self.assert_dispatch_is_covered(
                    source=source,
                    rulebook=rulebook,
                    event=event,
                )


if __name__ == "__main__":
    unittest.main()
