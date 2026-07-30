from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Sequence

from swb.db.repository import CardDefinition
from swb.engine.abilities import (
    AbilityContext,
    AbilityEvent,
    AbilityHandlers,
    AbilityKeyword,
    PlaceholderAbilityEvent,
    RUNTIME_UNIT_KEYWORDS,
)
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import (
    ActivateAmulet,
    Attack,
    BeginFusion,
    ChoiceKind,
    ChoiceOption,
    ChoiceRequest,
    Choose,
    EndTurn,
    Evolve,
    GameCommand,
    PlayCard,
    SuperEvolve,
    UseExtraPP,
)
from swb.engine.deck import CLASS_NAMES, validate_deck
from swb.engine.effects import (
    BoundTargetSnapshot,
    Condition,
    ConditionType,
    CostChangeMode,
    DeckFilter,
    EmptyDeckOutcome,
    EffectFrame,
    EffectKind,
    EffectOperation,
    ExprType,
    HandFilter,
    LeaderDamageMode,
    MAX_RANDOM_DISTRIBUTION_TOTAL,
    MAX_REPEAT_COUNT,
    ModifierDuration,
    SourceStateSnapshot,
    TargetKind,
    TurnEndDestroyTiming,
    ValueExpression,
)
from swb.engine.events import EventType, GameEvent
from swb.engine.faith import (
    FaithAbilityStacking,
    FaithGrantedAbility,
    FaithInstance,
    FaithTrigger,
)
from swb.engine.listeners import (
    CardListenerDefinition,
    ListenerZone,
    SourceRelation,
)
from swb.engine.emblem import EmblemPassive, EventScope, TurnScope
from swb.engine.origin import (
    CardOrigin,
    is_derived,
    is_graveyard_return_eligible,
    is_reanimate_eligible,
    is_token,
    is_token_definition,
    origin_for_added_card,
    origin_for_summoned_card,
)
from swb.engine.play_modes import PlayModeDefinition, validate_runtime_play_mode
from swb.engine.runtime_coverage import RuntimeCoverageRecorder
from swb.engine.state import (
    Amulet,
    BoardCard,
    DeathBatch,
    DeathCause,
    DeathRecord,
    DeckCard,
    DestroyedAmuletRecord,
    DestroyedFollowerRecord,
    FollowerEntryRecord,
    FusionMaterial,
    GameState,
    GrantedTurnEndAbility,
    GraveyardCard,
    HandCard,
    LeaderDamageModifier,
    CostModifier,
    Phase,
    PlayerState,
    ResolutionLoopError,
    StatModifier,
    Unit,
)
from swb.engine.targeting import (
    build_choice_options,
    build_graveyard_choice_options,
    graveyard_candidates,
    hand_candidates,
    hand_choice_options,
    is_all_target,
    is_choice_target,
    is_graveyard_target,
    is_random_target,
    leader_target_ids,
    leader_choice_options,
    pick_random_graveyard,
    target_candidates,
)
from swb.engine.triggers import TriggerBatch, TriggerRecord, collect_triggers
from swb.engine.conditions import (
    EvalContext,
    PartialConditionResult,
    evaluate_condition,
    evaluate_conditions_without_target,
    evaluate_expression,
)


class DamageType(str, Enum):
    COMBAT = "combat"
    EFFECT = "effect"
    LEADER = "leader"
    ABILITY = "ability"


@dataclass
class DamageResult:
    requested_amount: int
    prevented_amount: int = 0
    actual_amount: int = 0
    target_health_before: int = 0
    target_health_after: int = 0
    barrier_consumed: bool = False
    lethal: bool = False


@dataclass
class SuperEvolutionAttackContext:
    controller: int
    attacker_id: int
    target_id: int
    attacker_card_id: int
    attacker_name: str
    bonus_resolved: bool = False


class IllegalCommand(ValueError):
    pass


MAX_RESOLUTION_STEPS = 20_000
EARTH_SIGIL_TOKEN_CARD_ID = 90031210


def _leader_target_id(player_index: int) -> int:
    return -1 - player_index


def _is_leader_target_id(target_id: int | None) -> bool:
    return target_id is not None and target_id < 0


def _leader_index_from_target_id(target_id: int) -> int:
    return -1 - target_id

_DURATION_EXPANSION: dict[ModifierDuration, tuple[str, ...]] = {
    ModifierDuration.UNTIL_END_OF_TURN: (
        "until_end_of_turn",
        "until_end_of_controller_turn",
        "until_end_of_opponent_turn",
    ),
    ModifierDuration.UNTIL_START_OF_NEXT_TURN: (
        "until_start_of_next_turn",
        "until_start_of_controller_next_turn",
    ),
}

_SOURCE_REQUIRED_SELF_TARGET_EFFECTS = frozenset({
    EffectKind.DAMAGE_UNIT,
    EffectKind.HEAL_UNIT,
    EffectKind.HEAL_UNIT_AND_LEADER,
    EffectKind.BUFF_UNIT,
    EffectKind.DESTROY,
    EffectKind.BANISH,
    EffectKind.RETURN_TO_HAND,
    EffectKind.RETURN_TO_DECK,
    EffectKind.REDUCE_COUNTDOWN,
    EffectKind.INCREASE_COUNTDOWN,
    EffectKind.ADD_KEYWORD,
    EffectKind.ADD_RANDOM_KEYWORDS,
    EffectKind.REMOVE_KEYWORD,
    EffectKind.REMOVE_ALL_ABILITIES,
    EffectKind.REMOVE_LAST_WORDS,
    EffectKind.GRANT_ATTACKS_PER_TURN,
    EffectKind.GRANT_TURN_END_ABILITY,
    EffectKind.GRANT_TURN_END_BANISH,
    EffectKind.SUMMON_EXACT_COPY,
    EffectKind.TRANSFORM,
    EffectKind.SET_STATS,
    EffectKind.ADD_ATTACK_RESTRICTION,
    EffectKind.REMOVE_ATTACK_RESTRICTION,
    EffectKind.ADD_TARGETING_RESTRICTION,
    EffectKind.REMOVE_TARGETING_RESTRICTION,
    EffectKind.REPLAY_SOURCE_FANFARE,
})

_EVENT_SOURCE_BOARD_EFFECTS = _SOURCE_REQUIRED_SELF_TARGET_EFFECTS | frozenset({
    EffectKind.EVOLVE_UNIT,
    EffectKind.SUPER_EVOLVE_UNIT,
})

_OUTPUT_BINDING_EFFECTS = frozenset({
    EffectKind.SUMMON,
    EffectKind.SUMMON_EXACT_COPY,
    EffectKind.SUMMON_HAND_COPY,
    EffectKind.SUMMON_FROM_DECK,
    EffectKind.SUMMON_DESTROYED_AMULETS,
    EffectKind.DRAW,
    EffectKind.DRAW_FILTERED,
    EffectKind.REANIMATE,
    EffectKind.ADD_CARD,
})


def _operation_produces_output_binding(
    operation: EffectOperation,
) -> bool:
    return (
        operation.kind in _OUTPUT_BINDING_EFFECTS
        or (
            operation.kind is EffectKind.DESTROY
            and operation.bind_successful_targets
        )
    )


_SOURCE_CONDITION_TYPES = frozenset({
    ConditionType.SOURCE_EVOLVED,
    ConditionType.SOURCE_HAS_KEYWORD,
    ConditionType.SOURCE_CARD_TYPE_IS,
})

_SOURCE_EXPRESSION_TYPES = frozenset({
    ExprType.SOURCE_ATTACK,
    ExprType.SOURCE_HEALTH,
    ExprType.SOURCE_MISSING_HEALTH,
})


def _condition_depends_on_source(condition: Condition) -> bool:
    return (
        condition.type in _SOURCE_CONDITION_TYPES
        or any(
            _condition_depends_on_source(child)
            for child in condition.conditions
        )
    )


def _expression_depends_on_source(
    expression: ValueExpression | None,
) -> bool:
    if expression is None:
        return False
    return (
        expression.type in _SOURCE_EXPRESSION_TYPES
        or any(
            _expression_depends_on_source(child)
            for child in expression.values
        )
    )


def _expression_binding_keys(
    expression: ValueExpression | None,
) -> set[str]:
    if expression is None:
        return set()
    keys = (
        {expression.binding_key}
        if expression.type in {
            ExprType.BOUND_CARD_COST,
            ExprType.BOUND_TARGET_HEALTH,
            ExprType.BOUND_TARGET_COUNT,
        }
        and expression.binding_key is not None
        else set()
    )
    for child in expression.values:
        keys.update(_expression_binding_keys(child))
    return keys


def _expression_bindings_available(
    expression: ValueExpression | None,
    bindings: dict[str, tuple[BoundTargetSnapshot, ...]],
) -> bool:
    if expression is None:
        return True
    if expression.type is ExprType.BOUND_TARGET_COUNT:
        if expression.binding_key not in bindings:
            return False
    elif expression.type in {
        ExprType.BOUND_CARD_COST,
        ExprType.BOUND_TARGET_HEALTH,
    }:
        if len(bindings.get(expression.binding_key or "", ())) != 1:
            return False
    return all(
        _expression_bindings_available(child, bindings)
        for child in expression.values
    )


def _expire_duration_values(duration: ModifierDuration) -> tuple[str, ...]:
    return _DURATION_EXPANSION.get(duration, (duration.value,))


def _expires_for_player(
    duration: ModifierDuration,
    controller: int,
    active_player: int | None = None,
) -> int | None:
    if duration == ModifierDuration.PERMANENT:
        return None
    if duration == ModifierDuration.UNTIL_END_OF_TURN:
        return controller if active_player is None else active_player
    if duration == ModifierDuration.UNTIL_START_OF_NEXT_TURN:
        return controller
    if duration in (ModifierDuration.UNTIL_END_OF_CONTROLLER_TURN, ModifierDuration.UNTIL_START_OF_CONTROLLER_NEXT_TURN):
        return controller
    if duration == ModifierDuration.UNTIL_END_OF_OPPONENT_TURN:
        return 1 - controller
    return controller


@dataclass(frozen=True)
class GameConfig:
    max_hand: int = 9
    max_board: int = 5
    max_mana: int = 10
    max_turns: int | None = None
    starting_hand: int = 4
    starting_evolution_points: int = 2
    evolution_unlock_turn: int = 5
    second_player_evolution_unlock_turn: int = 4
    starting_super_evolution_points: int = 2
    first_player_super_evolution_unlock_turn: int = 7
    second_player_super_evolution_unlock_turn: int = 6
    starting_health: int = 20
    starting_player: int | None = 0
    enable_mulligan: bool = False
    extra_pp_refresh_turn: int = 6
    leader_area_limit: int = 5
    validate_invariants: bool = False
    retain_text_logs: bool = True
    event_history_limit: int | None = None
    audit_runtime_coverage: bool = False


@dataclass(frozen=True)
class CoreTransition:
    command: GameCommand
    events: tuple[GameEvent, ...]
    acting_player: int
    winner: int | None
    terminated: bool


@dataclass(frozen=True)
class GameEngineSnapshot:
    """Immutable serialized state needed to reproduce every future transition."""

    compatibility: tuple[object, ...]
    payload: bytes


class GameEngine:
    """Deterministic rules core. It has no RL action or observation concepts."""

    def __init__(
        self,
        deck_a: Sequence[CardDefinition],
        deck_b: Sequence[CardDefinition],
        *,
        class_a: int,
        class_b: int,
        seed: int | None = None,
        config: GameConfig | None = None,
        rulebook: RuleBook | None = None,
        card_resolver: Callable[[int], CardDefinition | None] | None = None,
    ):
        for player_index, (deck, class_id) in enumerate(
            zip((deck_a, deck_b), (class_a, class_b)),
            start=1,
        ):
            validate_deck(deck, class_id, player_index=player_index)
        self.deck_lists = (list(deck_a), list(deck_b))
        self.player_classes = (class_a, class_b)
        self.config = config or GameConfig()
        if self.config.event_history_limit is not None and (
            not isinstance(self.config.event_history_limit, int)
            or isinstance(self.config.event_history_limit, bool)
            or self.config.event_history_limit <= 0
        ):
            raise ValueError("event_history_limit must be a positive integer or None")
        if self.config.starting_player not in (None, 0, 1):
            raise ValueError("starting_player must be 0, 1, or None")
        if self.config.leader_area_limit <= 0:
            raise ValueError("leader_area_limit must be positive")
        if self.config.extra_pp_refresh_turn <= 0:
            raise ValueError("extra_pp_refresh_turn must be positive")
        self.rulebook = rulebook or RuleBook()
        self.card_resolver = card_resolver
        self.random = random.Random(seed)
        self.state = GameState(players=[])
        # Monotonic across resets and successful command transitions.  This is
        # deliberately engine-owned so every adapter can key derived caches
        # without depending on RL action identifiers.
        self._state_version: int = 0
        self.logs: list[str] = []
        self.event_history: list[GameEvent] = []
        self._active_transition_events: list[GameEvent] | None = None
        self.placeholder_ability_events: list[PlaceholderAbilityEvent] = []
        self.runtime_coverage = (
            RuntimeCoverageRecorder(self.rulebook)
            if self.config.audit_runtime_coverage
            else None
        )
        self.ability_handlers = AbilityHandlers(self)
        self._stabilizing: bool = False
        self._death_causes: dict[int, DeathCause] = {}
        self._suspended_batch: DeathBatch | None = None
        self._suspended_record: DeathRecord | None = None
        self._suspended_lw_records: list[DeathRecord] = []
        self._defer_last_words: bool = False
        self._deferred_death_batches: list[
            tuple[DeathBatch, list[DeathRecord]]
        ] = []

        self.ability_handlers.environment._execute_trigger_rules = self._execute_trigger_rules
        self.ability_handlers.environment._is_ability_covered = self._is_ability_covered
        self._next_modifier_id: int = 1
        self._next_choice_request_id: int = 1
        self._suspended_action: str | None = None
        self._suspended_action_state: dict | None = None
        self._suspended_event_state: dict | None = None
        self._active_super_evolution_attack: (
            SuperEvolutionAttackContext | None
        ) = None
        self._spellboost_pending: int | None = None
        self._pending_spellboost_player: int = 0
        self._pending_spellboost_source_card_id: int = 0
        self._pending_spellboost_source_entity_id: int | None = None
        self._emblem_batches: dict[int, dict[str, object]] = {}
        self._next_emblem_batch_id: int = 1
        self._listener_batches: dict[int, dict[str, object]] = {}
        self._next_listener_batch_id: int = 1
        self._emblem_expiration_batches: dict[int, dict[str, object]] = {}
        self._next_emblem_expiration_batch_id: int = 1

    def _execute_trigger_rules(self, trigger, context) -> None:
        if isinstance(trigger, str):
            trigger = Trigger(trigger)
        self._execute_trigger_rule_batch((trigger,), context)

    def _execute_trigger_rule_batch(
        self,
        triggers: tuple[Trigger, ...],
        context: AbilityContext,
    ) -> None:
        source = context.source
        if not isinstance(source, Unit):
            return
        if source.printed_abilities_removed:
            return
        pending = tuple(
            (trigger, operations)
            for trigger in triggers
            if (
                operations := self.rulebook.operations_for(
                    source.definition.card_id,
                    trigger,
                )
            )
        )
        if not pending:
            return
        for trigger, operations in reversed(pending):
            self._queue_effects(
                source.definition,
                source.entity_id,
                operations,
                controller=context.player_index,
                label=trigger.value,
                attack_target_entity_id=(
                    context.target.entity_id
                    if trigger in {Trigger.ATTACK, Trigger.CLASH}
                    and isinstance(context.target, Unit)
                    else None
                ),
            )
        self._continue_effects()

    def _is_ability_covered(self, context, ability) -> bool:
        card = (
            context.source.definition
            if hasattr(context.source, "definition")
            else context.source
        )
        if card is None:
            return False
        emblem_definitions = tuple(
            definition
            for definition in self.rulebook._emblem_defs.values()
            if definition.source_card_id == card.card_id
        )
        listeners = self.rulebook.listeners_for(card.card_id)
        modes = self.rulebook.modes_for(card.card_id)
        operation_groups = (
            tuple(
                self.rulebook.operations_for(card.card_id, trigger)
                for trigger in Trigger
            )
            + tuple(mode.operations for mode in modes)
            + tuple(
                definition.operations
                for definition in self.rulebook.union_bursts_for(card.card_id)
            )
            + tuple(definition.operations for definition in listeners)
            + tuple(
                operations
                for definition in emblem_definitions
                for operations in (
                    definition.on_gain,
                    definition.on_expire,
                    definition.last_words,
                    *(rule.operations for rule in definition.triggers),
                )
            )
        )
        container_condition_groups = (
            tuple(mode.conditions for mode in modes)
            + tuple(definition.conditions for definition in listeners)
            + tuple(
                rule.conditions
                for definition in emblem_definitions
                for rule in definition.triggers
            )
        )

        def operation_children(
            operation: EffectOperation,
        ) -> tuple[EffectOperation, ...]:
            return (
                operation.earth_rite_operations
                + operation.necromancy_operations
                + operation.faith_operations
                + operation.then_operations
                + operation.else_operations
                + operation.optional_operations
                + operation.repeat_operations
                + operation.granted_operations
                + tuple(
                    child
                    for option in operation.random_choice_options
                    for child in option.operations
                )
                + tuple(
                    child
                    for bucket in operation.random_distribution_operations
                    for child in bucket
                )
                + tuple(
                    child
                    for option in operation.choose_one_options
                    for child in option.operations
                )
            )

        def operation_kind_present(
            operations: tuple[EffectOperation, ...],
            kinds: frozenset[EffectKind],
        ) -> bool:
            return any(
                operation.kind in kinds
                or operation_kind_present(
                    operation_children(operation),
                    kinds,
                )
                for operation in operations
            )

        def filter_references_keyword(
            operations: tuple[EffectOperation, ...],
            keyword: AbilityKeyword,
        ) -> bool:
            for operation in operations:
                filters = (
                    operation.hand_filter,
                    operation.history_filter,
                    operation.board_filter,
                )
                if any(
                    definition is not None
                    and definition.keyword == keyword.value
                    for definition in filters
                ):
                    return True
                if filter_references_keyword(
                    operation_children(operation),
                    keyword,
                ):
                    return True
            return False

        if ability is AbilityKeyword.FUSION:
            return (
                self.rulebook.fusion_for(card.card_id) is not None
                or any(
                    definition.event is EventType.CARD_FUSED
                    for definition in listeners
                )
                or any(
                    rule.trigger == EventType.CARD_FUSED.value
                    for definition in emblem_definitions
                    for rule in definition.triggers
                )
            )
        if ability is AbilityKeyword.INVOCATION:
            return (
                card.card_type == "随从"
                and self.rulebook.invocation_for(card.card_id) is not None
            )
        if ability is AbilityKeyword.ACTIVATE:
            return (
                card.card_type == "护符"
                and self.rulebook.activation_for(card.card_id) is not None
            ) or any(
                definition.event is EventType.AMULET_ACTIVATED
                for definition in self.rulebook.listeners_for(card.card_id)
            )
        if (
            ability is AbilityKeyword.FAITH
            and self.rulebook.faith_for(card.card_id) is not None
        ):
            return True
        if ability is AbilityKeyword.UNION_BURST:
            return bool(self.rulebook.union_bursts_for(card.card_id))
        if ability in {
            AbilityKeyword.ENHANCE,
            AbilityKeyword.ACCELERATE,
            AbilityKeyword.CRYSTALLIZE,
        }:
            expected_mode_type = {
                AbilityKeyword.ENHANCE: "enhance",
                AbilityKeyword.ACCELERATE: "accelerate",
                AbilityKeyword.CRYSTALLIZE: "crystallize",
            }[ability]
            if any(
                mode.mode_type == expected_mode_type
                for mode in modes
            ):
                return True
            if (
                ability is AbilityKeyword.ENHANCE
                and any(
                    EmblemPassive.SUPPRESS_FOLLOWER_ENHANCE
                    in definition.passives
                    for definition in emblem_definitions
                )
            ):
                return True
            faith = self.rulebook.faith_for(card.card_id)
            return bool(
                ability is AbilityKeyword.ENHANCE
                and faith is not None
                and any(
                    rule.trigger is FaithTrigger.CARD_ENHANCED
                    for rule in faith.triggers
                )
            )
        if ability is AbilityKeyword.COUNTDOWN:
            return (
                self.rulebook.countdown_for(card.card_id) is not None
                or any(
                    mode.is_crystallize and mode.countdown is not None
                    for mode in modes
                )
                or any(
                    definition.source_card_id == card.card_id
                    and definition.countdown is not None
                    for definition in self.rulebook._emblem_defs.values()
                )
            )
        if ability is AbilityKeyword.CHOOSE:
            def contains_choose(
                operations: tuple[EffectOperation, ...],
            ) -> bool:
                for operation in operations:
                    if operation.kind is EffectKind.CHOOSE_ONE:
                        return True
                    nested = (
                        operation.earth_rite_operations
                        + operation.necromancy_operations
                        + operation.faith_operations
                        + operation.then_operations
                        + operation.else_operations
                        + operation.optional_operations
                        + operation.repeat_operations
                        + tuple(
                            child
                            for option in operation.random_choice_options
                            for child in option.operations
                        )
                        + tuple(
                            child
                            for bucket in (
                                operation.random_distribution_operations
                            )
                            for child in bucket
                        )
                    )
                    if contains_choose(nested):
                        return True
                    if any(
                        contains_choose(option.operations)
                        for option in operation.choose_one_options
                    ):
                        return True
                return False

            return any(
                contains_choose(operations)
                for operations in operation_groups
            ) or bool(
                (faith := self.rulebook.faith_for(card.card_id))
                is not None
                and any(
                    rule.trigger is FaithTrigger.MODE_SELECTED
                    for rule in faith.triggers
                )
            )
        if ability is AbilityKeyword.FANFARE:
            return bool(
                self.rulebook.operations_for(card.card_id, Trigger.FANFARE)
                or self.rulebook.union_bursts_for(card.card_id)
            )
        if ability is AbilityKeyword.LAST_WORDS:
            return bool(
                self.rulebook.operations_for(
                    card.card_id,
                    Trigger.LAST_WORDS,
                )
                or any(
                    (
                        definition.last_words
                        or definition.on_expire
                    )
                    for definition in emblem_definitions
                )
                or any(
                    filter_references_keyword(
                        operations,
                        AbilityKeyword.LAST_WORDS,
                    )
                    for operations in operation_groups
                )
                or any(
                    operation_kind_present(
                        operations,
                        frozenset({EffectKind.GRANT_LAST_WORDS}),
                    )
                    for operations in operation_groups
                )
            )
        if ability is AbilityKeyword.EMBLEM:
            return bool(emblem_definitions)
        if ability is AbilityKeyword.FAITH:
            if self.rulebook.faith_for(card.card_id) is not None:
                return True

            def contains_faith(
                operations: tuple[EffectOperation, ...],
            ) -> bool:
                for operation in operations:
                    if (
                        operation.faith_id is not None
                        or operation.kind
                        in {
                            EffectKind.CONSUME_FAITH,
                            EffectKind.GRANT_FAITH_ABILITY,
                            EffectKind.GRANT_FAITH_MODE_SELECTION_BONUS,
                        }
                    ):
                        return True
                    nested = (
                        operation.earth_rite_operations
                        + operation.necromancy_operations
                        + operation.faith_operations
                        + operation.then_operations
                        + operation.else_operations
                        + operation.optional_operations
                        + operation.repeat_operations
                        + tuple(
                            child
                            for option in operation.random_choice_options
                            for child in option.operations
                        )
                        + tuple(
                            child
                            for bucket in (
                                operation.random_distribution_operations
                            )
                            for child in bucket
                        )
                    )
                    if contains_faith(nested) or any(
                        contains_faith(option.operations)
                        for option in operation.choose_one_options
                    ):
                        return True
                return False

            return any(
                contains_faith(operations)
                for operations in operation_groups
            )
        if ability in {
            AbilityKeyword.OVERFLOW,
            AbilityKeyword.COMBO,
            AbilityKeyword.COOPERATION,
            AbilityKeyword.SPELLBOOST,
        }:
            if ability is AbilityKeyword.OVERFLOW:
                condition_types = (
                    ConditionType.CONTROLLER_OVERFLOW,
                    ConditionType.OPPONENT_OVERFLOW,
                )
                expression_types = (
                    ExprType.CONTROLLER_OVERFLOW,
                    ExprType.OPPONENT_OVERFLOW,
                )
                effect_kinds: tuple[EffectKind, ...] = ()
            elif ability is AbilityKeyword.COMBO:
                condition_types = (
                    ConditionType.CONTROLLER_COMBO_AT_LEAST,
                    ConditionType.OPPONENT_COMBO_AT_LEAST,
                )
                expression_types = (
                    ExprType.CONTROLLER_COMBO,
                    ExprType.OPPONENT_COMBO,
                )
                effect_kinds = (EffectKind.ADD_COMBO,)
            elif ability is AbilityKeyword.COOPERATION:
                condition_types = (
                    ConditionType.CONTROLLER_COOPERATION_AT_LEAST,
                    ConditionType.OPPONENT_COOPERATION_AT_LEAST,
                )
                expression_types = (
                    ExprType.CONTROLLER_COOPERATION,
                    ExprType.OPPONENT_COOPERATION,
                )
                effect_kinds = ()
            else:
                condition_types = (
                    ConditionType.SOURCE_SPELLBOOST_COUNT_AT_LEAST,
                )
                expression_types = (
                    ExprType.SOURCE_SPELLBOOST_COUNT,
                )
                effect_kinds = (EffectKind.SPELLBOOST_HAND,)

            if (
                ability is AbilityKeyword.SPELLBOOST
                and (
                    self.rulebook.spellboost_cost_reduction(card.card_id) > 0
                    or any(
                        definition.event is EventType.SPELLBOOSTED
                        for definition in listeners
                    )
                )
            ):
                return True

            def condition_contains(conditions: tuple[Condition, ...]) -> bool:
                for condition in conditions:
                    if condition.type in condition_types:
                        return True
                    if condition_contains(tuple(condition.conditions)):
                        return True
                return False

            def expression_contains(expression: ValueExpression | None) -> bool:
                if expression is None:
                    return False
                if expression.type in expression_types:
                    return True
                return any(
                    expression_contains(value)
                    for value in expression.values
                )

            def operation_contains(
                operations: tuple[EffectOperation, ...],
            ) -> bool:
                for operation in operations:
                    if operation.kind in effect_kinds:
                        return True
                    if condition_contains(operation.conditions):
                        return True
                    if expression_contains(operation.amount_expr):
                        return True
                    if expression_contains(operation.secondary_expr):
                        return True
                    if expression_contains(operation.deck_filter_cost_expr):
                        return True
                    if expression_contains(operation.target_count_expr):
                        return True
                    nested = (
                        operation.earth_rite_operations
                        + operation.necromancy_operations
                        + operation.faith_operations
                        + operation.then_operations
                        + operation.else_operations
                        + operation.optional_operations
                        + operation.repeat_operations
                        + tuple(
                            child
                            for option in operation.random_choice_options
                            for child in option.operations
                        )
                        + tuple(
                            child
                            for bucket in operation.random_distribution_operations
                            for child in bucket
                        )
                    )
                    if operation_contains(nested):
                        return True
                    for option in operation.choose_one_options:
                        if condition_contains(option.conditions):
                            return True
                        if operation_contains(option.operations):
                            return True
                return False

            return any(
                condition_contains(conditions)
                for conditions in container_condition_groups
            ) or any(
                operation_contains(operations)
                for operations in operation_groups
            )

        expected_kind = {
            AbilityKeyword.EARTH_RITE: EffectKind.EARTH_RITE,
            AbilityKeyword.NECROMANCY: EffectKind.NECROMANCY,
            AbilityKeyword.REANIMATE: EffectKind.REANIMATE,
        }.get(ability)
        if expected_kind is None:
            return False

        def contains_kind(operations: tuple[EffectOperation, ...]) -> bool:
            for operation in operations:
                if operation.kind is expected_kind:
                    return True
                nested = (
                    operation.earth_rite_operations
                    + operation.necromancy_operations
                    + operation.faith_operations
                    + operation.then_operations
                    + operation.else_operations
                    + operation.optional_operations
                    + operation.repeat_operations
                    + tuple(
                        child
                        for option in operation.random_choice_options
                        for child in option.operations
                    )
                    + tuple(
                        child
                        for bucket in operation.random_distribution_operations
                        for child in bucket
                    )
                )
                if contains_kind(nested):
                    return True
                if any(
                    contains_kind(option.operations)
                    for option in operation.choose_one_options
                ):
                    return True
            return False

        return (
            (
                ability is AbilityKeyword.EARTH_RITE
                and any(
                    definition.event is EventType.EARTH_RITE_ACTIVATED
                    for definition in listeners
                )
            )
            or any(
                contains_kind(operations)
                for operations in operation_groups
            )
        )

    @property
    def players(self) -> list[PlayerState]:
        return self.state.players

    @property
    def current_player(self) -> int:
        return self.state.active_player

    @property
    def turn(self) -> int:
        return self.state.turn

    @property
    def terminated(self) -> bool:
        return self.state.terminated

    @property
    def winner(self) -> int | None:
        return self.state.winner

    @property
    def state_version(self) -> int:
        """Monotonic version of the last completed official state transition."""
        return self._state_version

    def snapshot(self) -> GameEngineSnapshot:
        """Capture all mutable future-determining state between transitions."""
        if self._active_transition_events is not None:
            raise RuntimeError("cannot snapshot while a command transition is active")
        excluded = {
            "deck_lists",
            "player_classes",
            "config",
            "rulebook",
            "card_resolver",
            "ability_handlers",
            "runtime_coverage",
        }
        mutable = {
            name: value
            for name, value in self.__dict__.items()
            if name not in excluded
        }
        compatibility = (
            self.player_classes,
            tuple(
                tuple(card.card_id for card in deck)
                for deck in self.deck_lists
            ),
            self.config,
        )
        return GameEngineSnapshot(
            compatibility=compatibility,
            payload=pickle.dumps(mutable, protocol=pickle.HIGHEST_PROTOCOL),
        )

    def restore(self, snapshot: GameEngineSnapshot) -> None:
        """Restore without replacing immutable rules or card resolver assets."""
        expected = (
            self.player_classes,
            tuple(
                tuple(card.card_id for card in deck)
                for deck in self.deck_lists
            ),
            self.config,
        )
        if snapshot.compatibility != expected:
            raise ValueError("snapshot is incompatible with this engine configuration")
        previous_version = self._state_version
        restored = pickle.loads(snapshot.payload)
        for name, value in restored.items():
            setattr(self, name, value)
        self._state_version = max(previous_version, self._state_version) + 1
        self._active_transition_events = None

    def clone(self) -> GameEngine:
        """Return a deterministic branch sharing no mutable match state."""
        clone = GameEngine(
            self.deck_lists[0],
            self.deck_lists[1],
            class_a=self.player_classes[0],
            class_b=self.player_classes[1],
            seed=0,
            config=self.config,
            rulebook=self.rulebook,
            card_resolver=self.card_resolver,
        )
        clone.restore(self.snapshot())
        return clone

    def reset(self, *, seed: int | None = None) -> GameState:
        if seed is not None:
            self.random.seed(seed)
        first_player = (
            self.random.randrange(2)
            if self.config.starting_player is None
            else self.config.starting_player
        )
        decks = [list(deck) for deck in self.deck_lists]
        for deck in decks:
            self.random.shuffle(deck)
        self.state = GameState(
            active_player=first_player,
            first_player=first_player,
            players=[
                PlayerState(
                    deck=deck,
                    class_id=self.player_classes[index],
                    class_name=CLASS_NAMES[self.player_classes[index]],
                    health=self.config.starting_health,
                    max_health=self.config.starting_health,
                    evolution_points=self.config.starting_evolution_points,
                    super_evolution_points=self.config.starting_super_evolution_points,
                    extra_pp_available=(index != first_player),
                )
                for index, deck in enumerate(decks)
            ]
        )
        self.logs = (
            [
                "=== 对局开始 ===",
                *[
                    f"[玩家 {index + 1}] 职业：{player.class_name}，牌组 {len(self.deck_lists[index])} 张"
                    for index, player in enumerate(self.state.players)
                ],
            ]
            if self.config.retain_text_logs
            else []
        )
        self.event_history = []
        self._active_transition_events = None
        self.placeholder_ability_events = []
        if self.runtime_coverage is not None:
            self.runtime_coverage.reset()
        self._next_modifier_id = 1
        self._next_choice_request_id = 1
        self._suspended_batch = None
        self._suspended_record = None
        self._suspended_lw_records = []
        self._defer_last_words = False
        self._deferred_death_batches = []
        self._suspended_action = None
        self._suspended_action_state = None
        self._suspended_event_state = None
        self._active_super_evolution_attack = None
        self._spellboost_pending = None
        self._pending_spellboost_player = 0
        self._pending_spellboost_source_card_id = 0
        self._pending_spellboost_source_entity_id = None
        self._emblem_batches: dict[int, dict[str, object]] = {}
        self._next_emblem_batch_id = 1
        self._listener_batches: dict[int, dict[str, object]] = {}
        self._next_listener_batch_id = 1
        self._emblem_expiration_batches: dict[int, dict[str, object]] = {}
        self._next_emblem_expiration_batch_id = 1
        self._stabilizing = False
        self.state.destroyed_followers.clear()
        self.state.destroyed_amulets.clear()
        self.state.follower_entries.clear()
        self.state._next_death_sequence = 1
        self.state._next_follower_entry_sequence = 1
        self._initialize_faiths()
        for player_index in range(2):
            for _ in range(self.config.starting_hand):
                card = self.players[player_index].deck.pop()
                self._append_hand_card(
                    self.players[player_index],
                    card,
                    origin=CardOrigin.DECK,
                )
                self._log(player_index, f"起手：{card.name}")
        if self.config.enable_mulligan:
            self.state.phase = Phase.MULLIGAN
            self._request_mulligan(first_player)
        else:
            self.state.mulligan_completed[:] = [True, True]
            self._start_turn(first_player)
            self._resolve_event_queue()
        if self.config.validate_invariants:
            self.assert_invariants()
        self._state_version += 1
        return self.state

    def _request_mulligan(self, player_index: int) -> None:
        player = self.players[player_index]
        options: list[ChoiceOption] = []
        for mask in range(1 << len(player.hand)):
            names = [
                card.name
                for index, card in enumerate(player.hand)
                if mask & (1 << index)
            ]
            label = "保留全部" if not names else f"更换：{'、'.join(names)}"
            options.append(
                ChoiceOption(
                    option_id=f"mulligan:{mask}",
                    label=label,
                )
            )
        self.state.active_player = player_index
        self.state.phase = Phase.MULLIGAN
        self.state.pending_choice = ChoiceRequest(
            player_index=player_index,
            prompt="选择要重新抽取的起手牌",
            options=tuple(options),
            continuation_id="match_mulligan",
            choice_kind=ChoiceKind.GENERIC,
            request_id=self._allocate_choice_request_id(),
        )

    def _resolve_mulligan_choice(
        self,
        command: Choose,
        request: ChoiceRequest,
    ) -> None:
        try:
            mask = int(command.option_id.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise IllegalCommand("Mulligan option is malformed") from exc
        player = self.players[command.player_index]
        if mask < 0 or mask >= (1 << len(player.hand)):
            raise IllegalCommand("Mulligan mask is out of range")

        original_hand = list(player.hand)
        replacements: dict[int, HandCard] = {}
        returned: list[CardDefinition] = []
        for index, card in enumerate(original_hand):
            if not mask & (1 << index):
                continue
            if not player.deck:
                raise IllegalCommand("Mulligan replacement requires a deck card")
            replacement_source = player.deck.pop()
            replacement = self._append_hand_card(
                player,
                replacement_source,
                origin=CardOrigin.DECK,
            )
            player.hand.pop()
            player.hand_entity_ids.pop()
            replacements[index] = replacement
            returned.append(card.definition)

        player.hand = [
            replacements.get(index, card)
            for index, card in enumerate(original_hand)
        ]
        player.hand_entity_ids = [card.entity_id for card in player.hand]
        player.deck.extend(returned)
        self.random.shuffle(player.deck)
        self.state.mulligan_completed[command.player_index] = True
        self._log(
            command.player_index,
            f"起手换牌：更换 {len(returned)} 张",
        )
        self.state.pending_choice = None

        other_player = 1 - command.player_index
        if not self.state.mulligan_completed[other_player]:
            self._request_mulligan(other_player)
            return

        self.state.active_player = self.state.first_player
        self.state.phase = Phase.MAIN
        self._start_turn(self.state.first_player)

    def _initialize_faiths(self) -> None:
        for player_index, initial_deck in enumerate(self.deck_lists):
            player = self.players[player_index]
            candidates = {
                definition.source_card_id: definition
                for card in initial_deck
                if (definition := self.rulebook.faith_for(card.card_id))
                is not None
            }
            seen_faith_ids: set[str] = set()
            for definition in sorted(
                candidates.values(),
                key=lambda item: (item.faith_id, item.source_card_id),
            ):
                if definition.faith_id in seen_faith_ids:
                    continue
                seen_faith_ids.add(definition.faith_id)
                if not self._leader_area_has_capacity(player_index):
                    self._log(
                        player_index,
                        f"主战者区域已满，无法放置信仰 {definition.faith_id}",
                    )
                    continue
                sequence = player._next_faith_sequence
                player._next_faith_sequence += 1
                instance = FaithInstance(
                    definition=definition,
                    entity_id=self.state.allocate_entity_id(),
                    controller=player_index,
                    created_sequence=sequence,
                    value=definition.initial_value,
                )
                player.faiths.append(instance)
                self._emit(
                    GameEvent(
                        EventType.FAITH_PLACED,
                        player_index,
                        source_id=instance.entity_id,
                        amount=instance.value,
                        metadata={
                            "faith_id": instance.faith_id,
                            "source_card_id": instance.source_card_id,
                            "initial_value": instance.value,
                        },
                    )
                )
                self._log(
                    player_index,
                    f"信仰 {instance.faith_id} 置入主战者区域"
                    f"（信仰值 {instance.value}）",
                )

    def _leader_area_has_capacity(self, player_index: int) -> bool:
        player = self.players[player_index]
        return (
            len(player.faiths) + len(player.emblems)
            < self.config.leader_area_limit
        )

    def _advance_faiths_for_event(
        self,
        player_index: int,
        trigger: FaithTrigger,
        event: GameEvent,
    ) -> None:
        player = self.players[player_index]
        for instance in sorted(
            player.faiths,
            key=lambda item: item.created_sequence,
        ):
            for rule in instance.definition.triggers:
                if rule.trigger is not trigger:
                    continue
                if not self._event_card_filter_matches(
                    rule.event_filter,
                    event.type.value,
                    event.metadata,
                ):
                    continue
                delta = rule.amount * (
                    event.amount
                    if trigger is FaithTrigger.MODE_SELECTED
                    else 1
                )
                before = instance.value
                instance.value += delta
                self._emit(
                    GameEvent(
                        EventType.FAITH_VALUE_CHANGED,
                        player_index,
                        source_id=instance.entity_id,
                        amount=delta,
                        metadata={
                            "faith_id": instance.faith_id,
                            "source_card_id": instance.source_card_id,
                            "faith_value_before": before,
                            "faith_value_after": instance.value,
                            "trigger": trigger.value,
                            "trigger_source_id": event.source_id,
                            "super_evolution": bool(
                                event.metadata.get(
                                    "super_evolution",
                                    event.type
                                    is EventType.FOLLOWER_SUPER_EVOLVED,
                                )
                            ),
                        },
                    )
                )
                self._log(
                    player_index,
                    f"信仰 {instance.faith_id} 信仰值 "
                    f"{before} → {instance.value}",
                )

        accepted: list[tuple[FaithInstance, FaithGrantedAbility, CardDefinition]] = []
        for instance in sorted(
            player.faiths,
            key=lambda item: item.created_sequence,
        ):
            source_card = self._listener_source_definition(
                player_index,
                ListenerZone.LEADER_AREA.value,
                instance.entity_id,
                instance.source_card_id,
            )
            if source_card is None:
                continue
            for ability in sorted(
                instance.granted_abilities,
                key=lambda item: item.granted_sequence,
            ):
                if ability.trigger is trigger:
                    accepted.append((instance, ability, source_card))

        for instance, ability, _source_card in accepted:
            self._emit(GameEvent(
                EventType.FAITH_ABILITY_TRIGGERED,
                player_index,
                source_id=instance.entity_id,
                target_id=event.source_id,
                metadata={
                    "faith_id": instance.faith_id,
                    "ability_id": ability.ability_id,
                    "faith_trigger": trigger.value,
                    "granted_sequence": ability.granted_sequence,
                    "trigger_source_id": event.source_id,
                },
            ))
        for instance, ability, source_card in reversed(accepted):
            self._queue_effects(
                source_card,
                instance.entity_id,
                ability.operations,
                controller=player_index,
                label=f"信仰能力:{ability.ability_id}",
            )
        if accepted:
            self._continue_effects()

    def apply(self, command: GameCommand) -> CoreTransition:
        try:
            return self._apply_command(command)
        except IllegalCommand as exc:
            self._record_runtime_diagnostic(
                "illegal_command",
                detail=type(command).__name__ + ":" + str(exc),
            )
            raise

    def _apply_command(self, command: GameCommand) -> CoreTransition:
        self._ensure_entity_ids()
        if self.config.validate_invariants:
            self.assert_invariants()
        if self.terminated:
            raise IllegalCommand("The match has ended")
        if self.state.pending_choice is not None and not isinstance(command, Choose):
            raise IllegalCommand("A pending choice must be resolved first")
        if isinstance(command, Choose):
            request = self.state.pending_choice
            if request is not None and command.player_index != request.player_index:
                raise IllegalCommand("Choice command belongs to the wrong player")
        elif command.player_index != self.current_player:
            raise IllegalCommand("Command belongs to the non-active player")

        acting_player = self.current_player
        self._active_transition_events = []
        try:
            if isinstance(command, EndTurn):
                self._end_turn()
            elif isinstance(command, PlayCard):
                self._play_card(command)
            elif isinstance(command, Attack):
                self._attack(command)
            elif isinstance(command, Evolve):
                self._evolve(command)
            elif isinstance(command, SuperEvolve):
                self._super_evolve(command)
            elif isinstance(command, BeginFusion):
                self._begin_fusion(command)
            elif isinstance(command, ActivateAmulet):
                self._activate_amulet(command)
            elif isinstance(command, UseExtraPP):
                self._use_extra_pp(command)
            elif isinstance(command, Choose):
                self._choose(command)
            else:
                raise TypeError(f"Unknown command: {command!r}")

            self._resolve_event_queue()
            self._stabilize()
            self._resume_suspended_action()
            if self._suspended_action != "attack":
                self._active_super_evolution_attack = None
            if self.config.validate_invariants:
                self.assert_invariants()
            transition = CoreTransition(
                command=command,
                events=tuple(self._active_transition_events),
                acting_player=acting_player,
                winner=self.winner,
                terminated=self.terminated,
            )
        except Exception:
            self._active_transition_events = None
            raise
        self._active_transition_events = None
        self._state_version += 1
        return transition

    def _resume_suspended_action(self) -> None:
        while self._suspended_action is not None and self.state.pending_choice is None:
            action = self._suspended_action
            state = self._suspended_action_state
            if action == "attack":
                self._resume_attack(state)
            elif action == "evolve":
                self._suspended_action = None
                self._suspended_action_state = None
            elif action == "super_evolve":
                self._suspended_action = None
                self._suspended_action_state = None
            elif action == "turn_end":
                self._resume_end_turn(state)
            elif action == "turn_start":
                self._resume_start_turn(state)
            elif action == "play_follower":
                self._suspended_action = None
                self._suspended_action_state = None
                self._finish_follower_play(
                    state["unit_id"],
                    state["mode_operations"],
                    state["replace_base_operations"],
                    state["burst_operations"],
                    state["burst_metadata"],
                    state["burst_gauge"],
                    state["burst_replaces_base_operations"],
                    state["source_spellboost_count"],
                    state["source_cost"],
                    state["suppress_fanfare"],
                    state["suppress_enhance"],
                    state["auto_evolve"],
                )
            else:
                self._suspended_action = None
                self._suspended_action_state = None
            self._resolve_event_queue()
            self._stabilize()

    def legal_commands(self) -> list[GameCommand]:
        self._ensure_entity_ids()
        if self.terminated:
            return []
        if self.state.pending_choice is not None:
            request = self.state.pending_choice
            return [
                Choose(request.player_index, option.option_id)
                for option in request.options
            ]

        player = self.players[self.current_player]
        opponent = self.players[1 - self.current_player]
        commands: list[GameCommand] = [EndTurn(self.current_player)]
        if self._can_use_extra_pp(self.current_player):
            commands.append(UseExtraPP(self.current_player))
        board_full = len(player.board) >= self.config.max_board
        for index, card in enumerate(player.hand[: self.config.max_hand]):
            modes = self.rulebook.modes_for(card.card_id)
            if self._can_begin_fusion(card, player):
                commands.append(BeginFusion(self.current_player, card.entity_id))
            normal_playable = self._is_mode_playable(card, player, None)
            if normal_playable:
                commands.append(PlayCard(self.current_player, index, "normal"))
            for mode_def in modes:
                if self._is_mode_playable(card, player, mode_def):
                    commands.append(PlayCard(self.current_player, index, mode_def.mode_id))
        commands.extend(
            ActivateAmulet(self.current_player, entity.entity_id)
            for entity in player.board
            if isinstance(entity, Amulet)
            and self._can_activate_amulet(entity, self.current_player)
        )
        can_evolve = (
            player.evolution_points > 0
            and player.turns_started
            >= self._evolution_unlock_turn(self.current_player)
            and not player.evolved_this_turn
        )
        can_super_evolve = (
            player.super_evolution_points > 0
            and player.turns_started >= self._super_evolution_unlock_turn(self.current_player)
            and not player.evolved_this_turn
            and not player.super_evolved_this_turn
        )
        if can_evolve or can_super_evolve:
            for unit in player.board:
                if isinstance(unit, Unit) and not unit.evolved:
                    if can_evolve:
                        commands.append(Evolve(self.current_player, unit.entity_id))
                    if can_super_evolve:
                        commands.append(SuperEvolve(self.current_player, unit.entity_id))
        guards = [
            unit
            for unit in opponent.board
            if (
                isinstance(unit, Unit)
                and unit.has_guard
                and self._is_follower_attack_target(unit)
            )
        ]
        for unit in player.board:
            if not isinstance(unit, Unit):
                continue
            if not unit.can_attack or unit.attacks_remaining <= 0 or unit.attack <= 0:
                continue
            effective_guards = (
                []
                if self._unit_ignores_ward(unit)
                else guards
            )
            if not effective_guards and unit.can_attack_leader:
                commands.append(Attack(self.current_player, unit.entity_id, None))
            if unit.can_attack_units:
                targets = effective_guards or [
                    target
                    for target in opponent.board
                    if isinstance(target, Unit)
                    and self._is_follower_attack_target(target)
                ]
                commands.extend(
                    Attack(self.current_player, unit.entity_id, target.entity_id)
                    for target in targets
                )
        return commands

    def _can_use_extra_pp(self, player_index: int) -> bool:
        player = self.players[player_index]
        return (
            self.state.phase is Phase.MAIN
            and player_index != self.state.first_player
            and player.extra_pp_available
            and player.extra_pp_uses < 2
            and player.extra_pp_active_turn is None
        )

    def _use_extra_pp(self, command: UseExtraPP) -> None:
        if not self._can_use_extra_pp(command.player_index):
            raise IllegalCommand("Extra PP is not available")
        player = self.players[command.player_index]
        before = player.mana
        player.extra_pp_available = False
        player.extra_pp_uses += 1
        player.extra_pp_active_turn = self.turn
        player.mana += 1
        self._emit(
            GameEvent(
                EventType.EXTRA_PP_USED,
                command.player_index,
                amount=1,
                metadata={
                    "before": before,
                    "after": player.mana,
                    "max_mana": player.max_mana,
                    "uses": player.extra_pp_uses,
                },
            )
        )
        self._log(
            command.player_index,
            f"使用额外PP：{before} → {player.mana}",
        )

    def _effective_mana_cap(self, player: PlayerState) -> int:
        return player.max_mana + int(player.extra_pp_active_turn == self.turn)

    @staticmethod
    def _is_follower_attack_target(target: Unit) -> bool:
        return not target.ambush_active and not target.has_intimidate

    def _unit_ignores_ward(self, unit: Unit) -> bool:
        return (
            not unit.printed_abilities_removed
            and self.rulebook.ignores_ward(unit.definition.card_id)
        )

    def _can_activate_amulet(
        self,
        amulet: Amulet,
        player_index: int,
    ) -> bool:
        if player_index != self.current_player:
            return False
        if amulet not in self.players[player_index].board:
            return False
        definition = self.rulebook.activation_for(amulet.definition.card_id)
        if definition is None:
            return False
        if amulet.activated_turn == self.turn:
            return False
        if self.players[player_index].mana < definition.cost:
            return False
        operations = self.rulebook.operations_for(
            amulet.definition.card_id,
            Trigger.ACTIVATE,
        )
        if not operations:
            return False
        if any(
            operation.requires_target
            and not self._has_candidates_for(
                operation,
                player_index,
                source_entity_id=amulet.entity_id,
                source_fusion_count=len(amulet.fused_material_ids),
            )
            for operation in operations
        ):
            return False
        all_consume_targets = all(
            self._operation_consumes_target(operation)
            for operation in operations
        )
        if all_consume_targets and all(
            not self._has_candidates_for(
                operation,
                player_index,
                source_entity_id=amulet.entity_id,
                source_fusion_count=len(amulet.fused_material_ids),
            )
            for operation in operations
        ):
            return False
        return True

    def _activate_amulet(self, command: ActivateAmulet) -> None:
        player = self.players[self.current_player]
        amulet = next(
            (
                entity
                for entity in player.board
                if isinstance(entity, Amulet)
                and entity.entity_id == command.amulet_id
            ),
            None,
        )
        if amulet is None:
            raise IllegalCommand("Amulet is not controlled by the active player")
        if not self._can_activate_amulet(amulet, self.current_player):
            raise IllegalCommand("Amulet activation is not currently available")

        definition = self.rulebook.activation_for(amulet.definition.card_id)
        if definition is None:
            raise IllegalCommand("Amulet activation definition is unavailable")
        operations = self.rulebook.operations_for(
            amulet.definition.card_id,
            Trigger.ACTIVATE,
        )
        player.mana -= definition.cost
        amulet.activated_turn = self.turn
        self._log(
            self.current_player,
            f"策动 {amulet.definition.name}（{definition.cost}费）",
        )
        self._emit(
            GameEvent(
                EventType.AMULET_ACTIVATED,
                self.current_player,
                source_id=amulet.entity_id,
                metadata={
                    "source": amulet,
                    "card_id": amulet.definition.card_id,
                    "cost": definition.cost,
                },
            )
        )
        self._start_effects(
            amulet.definition,
            amulet.entity_id,
            operations,
            controller=self.current_player,
            label="策动",
        )

    def _evolution_unlock_turn(self, player_index: int) -> int:
        if player_index == self.state.first_player:
            return self.config.evolution_unlock_turn
        return self.config.second_player_evolution_unlock_turn

    def _super_evolution_unlock_turn(self, player_index: int) -> int:
        if player_index == self.state.first_player:
            return self.config.first_player_super_evolution_unlock_turn
        return self.config.second_player_super_evolution_unlock_turn

    def _super_evolution_is_unlocked(self, player_index: int) -> bool:
        return (
            self.players[player_index].turns_started
            >= self._super_evolution_unlock_turn(player_index)
        )

    def _eval_context(
        self,
        controller: int,
        *,
        source_entity_id: int | None = None,
        target_entity_id: int | None = None,
        attack_target_entity_id: int | None = None,
        source_card_id: int | None = None,
        source_fusion_count: int = 0,
        source_fusion_distinct_name_count: int = 0,
        source_spellboost_count: int = 0,
        source_cost: int = 0,
        distributed_value: int = 0,
        listener_activation_count: int = 0,
        event_source_entity_id: int | None = None,
        event_source_base_cost: int | None = None,
        source_snapshot: SourceStateSnapshot | None = None,
        target_snapshot: BoundTargetSnapshot | None = None,
        bound_target_snapshots: (
            dict[str, tuple[BoundTargetSnapshot, ...]] | None
        ) = None,
    ) -> EvalContext:
        return EvalContext(
            controller=controller,
            players=self.players,
            controller_super_evolution_unlocked=(
                self._super_evolution_is_unlocked(controller)
            ),
            opponent_super_evolution_unlocked=(
                self._super_evolution_is_unlocked(1 - controller)
            ),
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            attack_target_entity_id=attack_target_entity_id,
            source_card_id=source_card_id,
            source_fusion_count=source_fusion_count,
            source_fusion_distinct_name_count=(
                source_fusion_distinct_name_count
            ),
            source_spellboost_count=source_spellboost_count,
            source_cost=source_cost,
            distributed_value=distributed_value,
            listener_activation_count=listener_activation_count,
            event_source_entity_id=event_source_entity_id,
            event_source_base_cost=event_source_base_cost,
            source_snapshot=source_snapshot,
            target_snapshot=target_snapshot,
            bound_target_snapshots=bound_target_snapshots,
            turn=self.turn,
            destroyed_followers=tuple(self.state.destroyed_followers),
            follower_entries=tuple(self.state.follower_entries),
        )

    def _target_conditions_met(
        self,
        conditions: tuple[Condition, ...],
        entity: BoardCard,
        controller: int,
        *,
        source_entity_id: int | None = None,
        source_fusion_count: int = 0,
    ) -> bool:
        context = self._eval_context(
            controller,
            source_entity_id=source_entity_id,
            target_entity_id=entity.entity_id,
            source_fusion_count=source_fusion_count,
        )
        return all(evaluate_condition(condition, context) for condition in conditions)

    def effective_play_cost(self, hand_card, mode_def) -> int:
        """Compute the effective play cost for a hand card with optional mode."""
        if mode_def is not None:
            return mode_def.cost
        if isinstance(hand_card, HandCard):
            return hand_card.current_cost
        return hand_card.cost

    def _player_has_emblem_passive(
        self,
        player_index: int,
        passive: EmblemPassive,
    ) -> bool:
        return any(
            passive in emblem.definition.passives
            for emblem in self.players[player_index].emblems
        )

    def _selected_enhance_mode(self, card, player):
        """Return the mandatory highest affordable applicable Enhance mode."""
        applicable = []
        for mode_def in self.rulebook.modes_for(card.card_id):
            try:
                validate_runtime_play_mode(
                    mode_def,
                    f"card {card.card_id}/play_modes/{mode_def.mode_id}",
                )
            except ValueError:
                continue
            if not mode_def.is_enhance or mode_def.cost > player.mana:
                continue
            if mode_def.conditions:
                ctx = self._eval_context(self.current_player)
                from swb.engine.conditions import (
                    PartialConditionResult,
                    evaluate_conditions_without_target,
                )
                result = evaluate_conditions_without_target(
                    mode_def.conditions,
                    ctx,
                )
                if result is not PartialConditionResult.TRUE:
                    continue
            applicable.append(mode_def)
        if not applicable:
            return None
        return max(applicable, key=lambda item: item.cost)

    def _is_mode_playable(self, card, player, mode_def) -> bool:
        if isinstance(card, HandCard) and card.cannot_be_played:
            return False
        selected_enhance = self._selected_enhance_mode(card, player)
        if mode_def is None:
            if selected_enhance is not None:
                return False
        elif mode_def.is_enhance and mode_def is not selected_enhance:
            return False
        cost = self.effective_play_cost(card, mode_def)
        if cost > player.mana:
            return False
        effective_type = card.card_type
        if mode_def is not None:
            try:
                validate_runtime_play_mode(
                    mode_def,
                    f"card {card.card_id}/play_modes/{mode_def.mode_id}",
                )
            except ValueError:
                return False
            if mode_def.is_accelerate or mode_def.is_crystallize:
                # Accelerate and Crystallize are alternate play routes only
                # while the card's current normal cost is greater than the
                # player's remaining PP. This comparison intentionally uses
                # the hand card's runtime cost so cost changes are respected.
                if player.mana >= self.effective_play_cost(card, None):
                    return False
                effective_type = (
                    "法术" if mode_def.is_accelerate else "护符"
                )
            elif mode_def.resulting_card_type:
                effective_type = mode_def.resulting_card_type
        if effective_type in {"随从", "护符"} and len(player.board) >= self.config.max_board:
            return False
        if mode_def is None:
            return self._is_card_playable(card, player)
        if mode_def.conditions:
            ctx = self._eval_context(self.current_player)
            from swb.engine.conditions import evaluate_conditions_without_target, PartialConditionResult
            result = evaluate_conditions_without_target(mode_def.conditions, ctx)
            if result is not PartialConditionResult.TRUE:
                return False
        suppress_enhance = (
            card.card_type == "随从"
            and mode_def.is_enhance
            and self._player_has_emblem_passive(
                self.current_player,
                EmblemPassive.SUPPRESS_FOLLOWER_ENHANCE,
            )
        )
        suppress_fanfare = (
            card.card_type == "随从"
            and self._player_has_emblem_passive(
                self.current_player,
                EmblemPassive.SUPPRESS_FOLLOWER_FANFARE,
            )
        )
        ops = (
            ()
            if suppress_enhance
            else (mode_def.operations if mode_def.operations else ())
        )
        if (
            mode_def.is_enhance
            and not mode_def.replace_base_operations
            and not suppress_fanfare
        ):
            trigger = (
                Trigger.FANFARE
                if card.card_type == "随从"
                else Trigger.PLAY
            )
            ops = self.rulebook.operations_for(card.card_id, trigger) + ops
        if ops:
            source_entity_id = (
                card.entity_id if isinstance(card, HandCard) else None
            )
            if any(
                op.requires_target
                and not self._has_candidates(
                    op,
                    source_entity_id=source_entity_id,
                )
                for op in ops
            ):
                return False
            all_require_target = all(
                self._operation_consumes_target(op)
                for op in ops
            )
            if all_require_target and all(
                not self._has_candidates(
                    op,
                    source_entity_id=source_entity_id,
                )
                for op in ops
            ):
                return False
        return True

    def _record_cooperation(
        self,
        player_index: int,
        amount: int,
        *,
        source_card_id: int | None = None,
        source_entity_id: int | None = None,
        summon_cause: str = "play",
    ) -> None:
        if amount == 0:
            return
        player = self.players[player_index]
        before = player.cooperation
        player.add_cooperation(amount)
        self._emit(
            GameEvent(
                EventType.COOPERATION_CHANGED,
                player_index,
                amount=amount,
                source_id=source_entity_id,
                metadata={
                    "cooperation_before": before,
                    "cooperation_after": player.cooperation,
                    "source_card_id": source_card_id,
                    "source_entity_id": source_entity_id,
                    "summon_cause": summon_cause,
                },
            )
        )

    def _record_combo(
        self,
        player_index: int,
        amount: int,
        *,
        source_card_id: int | None = None,
        source_entity_id: int | None = None,
        cause: str = "effect",
    ) -> None:
        if amount == 0:
            return
        player = self.players[player_index]
        before = player.cards_played_this_turn
        player.add_combo(amount)
        self._emit(
            GameEvent(
                EventType.COMBO_CHANGED,
                player_index,
                amount=amount,
                source_id=source_entity_id,
                metadata={
                    "combo_before": before,
                    "combo_after": player.cards_played_this_turn,
                    "source_card_id": source_card_id,
                    "source_entity_id": source_entity_id,
                    "cause": cause,
                },
            )
        )

    def _summon_follower_to_board(
        self,
        player_index: int,
        definition: CardDefinition,
        *,
        summon_cause: str,
        entity_id: int | None = None,
        origin: CardOrigin = CardOrigin.DECK,
        source_origin: CardOrigin | None = None,
        fused_material_ids: tuple[int, ...] = (),
    ) -> Unit | None:
        player = self.players[player_index]
        if len(player.board) >= self.config.max_board:
            return None
        unit = Unit.summon(
            definition,
            entity_id=(
                entity_id
                if entity_id is not None
                else self.state.allocate_entity_id()
            ),
            origin=origin,
            source_origin=source_origin,
        )
        self._apply_initial_keyword_overrides(unit)
        self._apply_initial_passives(unit)
        player.board.append(unit)
        unit.fused_material_ids.extend(fused_material_ids)
        self._record_follower_entry(
            player_index,
            definition,
            entry_cause=summon_cause,
        )
        self._record_cooperation(
            player_index,
            1,
            source_card_id=definition.card_id,
            source_entity_id=unit.entity_id,
            summon_cause=summon_cause,
        )
        return unit

    def _apply_initial_keyword_overrides(self, unit: Unit) -> None:
        excluded = self.rulebook.non_intrinsic_keywords(
            unit.definition.card_id
        )
        if not excluded:
            return
        unit.removed_keywords.update(excluded)
        unit._synchronize_keyword_state()

    def _apply_initial_passives(self, unit: Unit) -> None:
        attacks_per_turn = self.rulebook.attacks_per_turn(
            unit.definition.card_id
        )
        if attacks_per_turn > 1:
            unit.grant_attacks_per_turn(attacks_per_turn)
        if self.rulebook.forces_enemy_ability_target(unit.definition.card_id):
            from swb.engine.state import TargetingRestriction

            unit.add_targeting_restriction(
                TargetingRestriction.FORCES_ENEMY_ABILITY_TARGET,
                duration="permanent",
            )

    def _fusion_material_candidates(
        self,
        fusion_card: HandCard,
        player: PlayerState,
    ) -> list[HandCard]:
        definition = self.rulebook.fusion_for(fusion_card.card_id)
        if definition is None:
            return []
        return [
            candidate
            for candidate in player.hand
            if isinstance(candidate, HandCard)
            and candidate.entity_id != fusion_card.entity_id
            and definition.material_filter.matches(candidate.definition)
        ]

    @staticmethod
    def _fusion_material_records(
        player: PlayerState,
        material_ids: tuple[int, ...] | list[int],
    ) -> tuple[FusionMaterial, ...]:
        by_id = {record.entity_id: record for record in player.fusion_materials}
        try:
            return tuple(by_id[material_id] for material_id in material_ids)
        except KeyError as exc:
            raise IllegalCommand(
                f"Fusion material record {exc.args[0]} is missing"
            ) from exc

    @staticmethod
    def _fusion_transform_result(
        definition,
        material_definitions: tuple[CardDefinition, ...],
    ):
        total_count = len(material_definitions)
        total_cost = sum(card.cost for card in material_definitions)
        distinct_cards = len({card.card_id for card in material_definitions})
        for result in definition.transform_results:
            if (
                result.min_total_materials is not None
                and total_count < result.min_total_materials
            ):
                continue
            if (
                result.max_total_materials is not None
                and total_count > result.max_total_materials
            ):
                continue
            if (
                result.min_total_material_cost is not None
                and total_cost < result.min_total_material_cost
            ):
                continue
            if (
                result.max_total_material_cost is not None
                and total_cost > result.max_total_material_cost
            ):
                continue
            if (
                result.min_distinct_material_cards is not None
                and distinct_cards < result.min_distinct_material_cards
            ):
                continue
            if result.material_filter is not None:
                matches = [
                    result.material_filter.matches(card)
                    for card in material_definitions
                ]
                if result.material_match == "all" and not all(matches):
                    continue
                if result.material_match == "any" and not any(matches):
                    continue
            return result
        return None

    def _transform_hand_card(
        self,
        hand_card: HandCard,
        replacement: CardDefinition,
        player_index: int,
        *,
        preserve_fused_materials: bool,
    ) -> GameEvent:
        old_definition = hand_card.definition
        previous_origin = hand_card.source_origin or hand_card.origin
        fused_material_ids = (
            list(hand_card.fused_material_ids)
            if preserve_fused_materials
            else []
        )
        hand_card.definition = replacement
        hand_card.cost_modifiers.clear()
        hand_card.stat_modifiers.clear()
        hand_card.printed_keyword_overrides.clear()
        hand_card.printed_keyword_overrides.update(
            self.rulebook.non_intrinsic_keywords(replacement.card_id)
        )
        hand_card.permanent_keywords.clear()
        hand_card.temporary_keywords.clear()
        hand_card.removed_keywords.clear()
        hand_card.temporary_keyword_removals.clear()
        hand_card.granted_last_words.clear()
        hand_card.effect_destroy_immunity = False
        hand_card.spellboost_count = 0
        hand_card.spellboost_cost_reduction = (
            self.rulebook.spellboost_cost_reduction(replacement.card_id)
        )
        hand_card.cannot_be_played = self.rulebook.cannot_be_played(
            replacement.card_id
        )
        hand_card.origin = CardOrigin.TRANSFORMED
        hand_card.source_origin = previous_origin
        hand_card.fused_material_ids = fused_material_ids
        hand_card.fusion_used_turn = None
        hand_card.evolutions_while_in_hand = 0
        hand_card.union_burst_gauge_bonus = 0
        return GameEvent(
            EventType.HAND_CARD_TRANSFORMED,
            player_index,
            source_id=hand_card.entity_id,
            metadata={
                "from_card_id": old_definition.card_id,
                "to_card_id": replacement.card_id,
                "fused_material_ids": tuple(fused_material_ids),
            },
        )

    def _can_begin_fusion(
        self,
        fusion_card: HandCard,
        player: PlayerState,
    ) -> bool:
        definition = self.rulebook.fusion_for(fusion_card.card_id)
        if definition is None or fusion_card.fusion_used_turn == self.turn:
            return False
        return len(self._fusion_material_candidates(fusion_card, player)) >= definition.min_materials

    @staticmethod
    def _fusion_material_option(card: HandCard) -> ChoiceOption:
        return ChoiceOption(
            option_id=f"hand:{card.entity_id}",
            label=card.name,
            entity_id=card.entity_id,
        )

    def _begin_fusion(self, command: BeginFusion) -> None:
        player = self.players[self.current_player]
        try:
            fusion_card = next(
                card
                for card in player.hand
                if isinstance(card, HandCard)
                and card.entity_id == command.fusion_entity_id
            )
        except StopIteration as exc:
            raise IllegalCommand("Fusion card is not in hand") from exc
        definition = self.rulebook.fusion_for(fusion_card.card_id)
        if definition is None:
            raise IllegalCommand("Card has no structured fusion definition")
        if fusion_card.fusion_used_turn == self.turn:
            raise IllegalCommand("This card has already fused this turn")
        candidates = self._fusion_material_candidates(fusion_card, player)
        if len(candidates) < definition.min_materials:
            raise IllegalCommand("Not enough legal fusion materials")
        maximum = min(
            len(candidates),
            definition.max_materials
            if definition.max_materials is not None
            else len(candidates),
        )
        options = tuple(
            [self._fusion_material_option(card) for card in candidates]
            + [ChoiceOption("fusion:cancel", "取消融合")]
        )
        self.state.pending_choice = ChoiceRequest(
            player_index=self.current_player,
            prompt=f"为 {fusion_card.name} 选择融合材料",
            options=options,
            continuation_id=f"fusion:{fusion_card.entity_id}",
            choice_kind=ChoiceKind.FUSION,
            request_id=self._allocate_choice_request_id(),
            target_count=maximum,
        )
        self.state.phase = Phase.AWAITING_CHOICE
        self._log(self.current_player, f"{fusion_card.name} 开始选择融合材料")

    def _fusion_choice_target(self, request: ChoiceRequest) -> HandCard:
        try:
            target_id = int(request.continuation_id.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise IllegalCommand("Fusion continuation is malformed") from exc
        try:
            return next(
                card
                for card in self.players[request.player_index].hand
                if isinstance(card, HandCard) and card.entity_id == target_id
            )
        except StopIteration as exc:
            raise IllegalCommand("Fusion card left hand") from exc

    def _resolve_fusion_choice(
        self,
        command: Choose,
        request: ChoiceRequest,
    ) -> None:
        player = self.players[request.player_index]
        fusion_card = self._fusion_choice_target(request)
        definition = self.rulebook.fusion_for(fusion_card.card_id)
        if definition is None:
            raise IllegalCommand("Fusion definition is no longer available")
        if fusion_card.fusion_used_turn == self.turn:
            raise IllegalCommand("This card has already fused this turn")

        if command.option_id == "fusion:cancel":
            self.state.pending_choice = None
            self.state.phase = Phase.MAIN
            self._log(command.player_index, f"取消 {fusion_card.name} 的融合")
            return

        if command.option_id == "fusion:confirm":
            selected_ids = tuple(
                option.entity_id
                for option in request.selected_options
                if option.entity_id is not None
            )
            if len(selected_ids) < definition.min_materials:
                raise IllegalCommand("Fusion requires more selected materials")
            if (
                definition.max_materials is not None
                and len(selected_ids) > definition.max_materials
            ):
                raise IllegalCommand("Fusion selected too many materials")
            by_id = {
                card.entity_id: card
                for card in player.hand
                if isinstance(card, HandCard)
            }
            materials: list[HandCard] = []
            for material_id in selected_ids:
                material = by_id.get(material_id)
                if (
                    material is None
                    or material.entity_id == fusion_card.entity_id
                    or not definition.material_filter.matches(material.definition)
                ):
                    raise IllegalCommand("Selected fusion material is no longer legal")
                materials.append(material)

            existing_records = self._fusion_material_records(
                player, fusion_card.fused_material_ids
            )
            prospective_definitions = tuple(
                record.definition for record in existing_records
            ) + tuple(material.definition for material in materials)
            transform_result = self._fusion_transform_result(
                definition, prospective_definitions
            )
            replacement = None
            if transform_result is not None:
                if self.card_resolver is None:
                    raise IllegalCommand(
                        "Fusion hand transform requires a card resolver"
                    )
                try:
                    replacement = self.card_resolver(transform_result.card_id)
                except KeyError as exc:
                    raise IllegalCommand(
                        f"Fusion transform card {transform_result.card_id} not found"
                    ) from exc
                if replacement is None:
                    raise IllegalCommand(
                        f"Fusion transform card {transform_result.card_id} not found"
                    )

            for material in materials:
                index = next(
                    idx
                    for idx, card in enumerate(player.hand)
                    if isinstance(card, HandCard)
                    and card.entity_id == material.entity_id
                )
                player.hand.pop(index)
                player.hand_entity_ids.pop(index)
                record = FusionMaterial(
                    definition=material.definition,
                    entity_id=material.entity_id,
                    owner=request.player_index,
                    consumed_sequence=player._next_fusion_sequence,
                    fused_into_entity_id=fusion_card.entity_id,
                    origin=material.origin,
                    source_origin=material.source_origin,
                    inherited_material_ids=tuple(material.fused_material_ids),
                )
                player._next_fusion_sequence += 1
                player.fusion_materials.append(record)
                fusion_card.fused_material_ids.append(record.entity_id)

            fusion_card.fusion_used_turn = self.turn
            fused_from_definition = fusion_card.definition
            fused_from_card_id = fusion_card.card_id
            fusion_count = len(fusion_card.fused_material_ids)
            transform_event = None
            if replacement is not None:
                transform_event = self._transform_hand_card(
                    fusion_card,
                    replacement,
                    request.player_index,
                    preserve_fused_materials=(
                        transform_result.preserve_fused_materials
                    ),
                )
            self.state.pending_choice = None
            self.state.phase = Phase.MAIN
            self._emit(
                GameEvent(
                    EventType.CARD_FUSED,
                    request.player_index,
                    source_id=fusion_card.entity_id,
                    amount=len(materials),
                    metadata={
                        "source": fused_from_definition,
                        "fusion_card_id": fused_from_card_id,
                        "result_card_id": fusion_card.card_id,
                        "material_entity_ids": selected_ids,
                        "material_card_ids": tuple(
                            material.card_id for material in materials
                        ),
                        "material_definitions": tuple(
                            material.definition for material in materials
                        ),
                        "fusion_count": fusion_count,
                    },
                )
            )
            if transform_event is not None:
                self._emit(transform_event)
            self._log(
                request.player_index,
                f"{fusion_card.name} 融合 {len(materials)} 张卡牌",
            )
            return

        if command.option_id.startswith("hand:"):
            material_id = int(command.option_id.split(":", 1)[1])
            material = next(
                (
                    card
                    for card in player.hand
                    if isinstance(card, HandCard)
                    and card.entity_id == material_id
                ),
                None,
            )
            if (
                material is None
                or material.entity_id == fusion_card.entity_id
                or not definition.material_filter.matches(material.definition)
            ):
                raise IllegalCommand("Fusion material is no longer legal")
            selected_ids = {
                option.entity_id for option in request.selected_options
            }
            if material.entity_id in selected_ids:
                raise IllegalCommand("Fusion material was already selected")
            selected = (*request.selected_options, self._fusion_material_option(material))
            maximum = request.target_count
            remaining = [
                candidate
                for candidate in self._fusion_material_candidates(fusion_card, player)
                if candidate.entity_id not in {
                    option.entity_id for option in selected
                }
            ]
            if len(selected) >= maximum:
                remaining = []
            options: list[ChoiceOption] = [
                self._fusion_material_option(candidate)
                for candidate in remaining
            ]
            if len(selected) >= definition.min_materials:
                options.append(ChoiceOption("fusion:confirm", "确认融合"))
            options.append(ChoiceOption("fusion:cancel", "取消融合"))
            self.state.pending_choice = replace(
                request,
                options=tuple(options),
                request_id=self._allocate_choice_request_id(),
                selected_options=selected,
            )
            self._log(
                request.player_index,
                f"已为 {fusion_card.name} 选择 {len(selected)} 张融合材料",
            )
            return

        raise IllegalCommand("Unknown fusion choice")

    def _play_card(self, command: PlayCard) -> None:
        player = self.players[self.current_player]
        if not 0 <= command.hand_index < len(player.hand):
            raise IllegalCommand("Hand index is out of range")
        hand_card = player.hand[command.hand_index]
        if not isinstance(hand_card, HandCard):
            self._ensure_entity_ids()
            hand_card = player.hand[command.hand_index]
        card = hand_card.definition
        mode_id = command.mode_id if hasattr(command, 'mode_id') and command.mode_id else "normal"

        modes = self.rulebook.modes_for(card.card_id)
        mode_def = None
        if mode_id != "normal":
            if not modes:
                raise IllegalCommand(
                    f"Card {card.card_id} has no play modes; cannot use mode {mode_id!r}"
                )
            for m in modes:
                if m.mode_id == mode_id:
                    mode_def = m
                    break
            if mode_def is None:
                raise IllegalCommand(
                    f"Unknown play mode {mode_id!r} for card {card.card_id}; "
                    f"available: {[m.mode_id for m in modes]}"
                )
            if mode_def.mode_type == "choose":
                raise IllegalCommand("'choose' play mode is not yet implemented")

        if not self._is_mode_playable(hand_card, player, mode_def):
            raise IllegalCommand(
                f"Play mode {mode_id!r} is not currently playable"
            )

        play_cost = self.effective_play_cost(hand_card, mode_def)
        hand_entity_id = hand_card.entity_id
        hand_origin = hand_card.origin
        hand_source_origin = hand_card.source_origin
        hand_spellboost_count = hand_card.spellboost_count
        hand_current_cost = hand_card.current_cost
        hand_stat_modifiers = tuple(hand_card.stat_modifiers)
        suppress_fanfare = self._player_has_emblem_passive(
            self.current_player,
            EmblemPassive.SUPPRESS_FOLLOWER_FANFARE,
        )
        suppress_enhance = (
            mode_def is not None
            and mode_def.is_enhance
            and self._player_has_emblem_passive(
                self.current_player,
                EmblemPassive.SUPPRESS_FOLLOWER_ENHANCE,
            )
        )
        auto_evolve = self._player_has_emblem_passive(
            self.current_player,
            EmblemPassive.AUTO_EVOLVE_PLAYED_FOLLOWERS,
        )
        fused_material_ids = tuple(hand_card.fused_material_ids)
        fusion_materials = self._fusion_material_records(
            player,
            fused_material_ids,
        )
        burst_gauge = hand_card.union_burst_gauge(player.turns_started)
        active_bursts = tuple(
            definition
            for definition in self.rulebook.union_bursts_for(card.card_id)
            if burst_gauge >= definition.threshold
        )
        replacing_thresholds = tuple(
            definition.threshold
            for definition in active_bursts
            if definition.replace_lower_bursts
        )
        if replacing_thresholds:
            replacement_threshold = max(replacing_thresholds)
            active_bursts = tuple(
                definition
                for definition in active_bursts
                if definition.threshold >= replacement_threshold
            )
        burst_operations = tuple(
            operation
            for definition in active_bursts
            for operation in definition.operations
        )
        burst_metadata = tuple(
            (definition.kind.value, definition.threshold)
            for definition in active_bursts
        )
        burst_replaces_base_operations = any(
            definition.replace_base_operations
            for definition in active_bursts
        )

        self._dispatch_card_ability(AbilityEvent.CHECK_PLAY, card)
        player.hand.pop(command.hand_index)
        player.hand_entity_ids.pop(command.hand_index)
        player.mana -= play_cost
        self._record_combo(
            self.current_player,
            1,
            source_card_id=card.card_id,
            source_entity_id=hand_entity_id,
            cause="play",
        )

        if mode_def is not None and mode_def.is_accelerate:
            self._play_accelerate(
                card, play_cost, hand_entity_id, hand_origin,
                hand_source_origin, mode_def, fusion_materials,
                source_cost=hand_current_cost,
            )
            return
        if mode_def is not None and mode_def.is_crystallize:
            self._play_crystallize(
                card, play_cost, hand_entity_id, hand_origin,
                hand_source_origin, mode_def, fused_material_ids,
                source_cost=hand_current_cost,
            )
            return
        if card.card_type == "法术" and (
            mode_def is None or mode_def.is_enhance
        ):
            self._play_spell(
                card, play_cost, hand_entity_id, origin=hand_origin,
                source_origin=hand_source_origin,
                source_spellboost_count=hand_spellboost_count,
                source_cost=hand_current_cost,
                fusion_materials=fusion_materials,
                mode_def=mode_def,
                burst_operations=burst_operations,
                burst_metadata=burst_metadata,
                burst_gauge=burst_gauge,
                burst_replaces_base_operations=(
                    burst_replaces_base_operations
                ),
            )
            return
        if card.card_type == "护符" and (
            mode_def is None or mode_def.is_enhance
        ):
            self._play_amulet(
                card, play_cost, origin=hand_origin,
                fused_material_ids=fused_material_ids,
                source_cost=hand_current_cost,
                mode_def=mode_def,
                burst_operations=burst_operations,
                burst_metadata=burst_metadata,
                burst_gauge=burst_gauge,
                burst_replaces_base_operations=(
                    burst_replaces_base_operations
                ),
            )
            return

        unit = self._summon_follower_to_board(
            self.current_player,
            card,
            summon_cause="play",
            origin=hand_origin,
            source_origin=hand_source_origin,
            fused_material_ids=fused_material_ids,
        )
        if unit is None:
            raise IllegalCommand("Board is full")
        for modifier in hand_stat_modifiers:
            unit.add_stat_modifier(modifier)
        self._apply_hand_card_runtime_to_unit(hand_card, unit)
        self._log(
            self.current_player,
            f"打出 {card.name} ({play_cost}费 {unit.attack}/{unit.health})",
        )
        self._emit(
            GameEvent(
                EventType.CARD_PLAYED,
                self.current_player,
                source_id=unit.entity_id,
                metadata={
                    "source": unit,
                    "mode_id": mode_id,
                    "card_id": card.card_id,
                    "entity_id": unit.entity_id,
                    "base_cost": card.cost,
                    "source_cost": hand_current_cost,
                    "cost_changed": hand_current_cost != card.cost,
                },
            )
        )
        self._emit(
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                self.current_player,
                source_id=unit.entity_id,
                metadata={
                    "source": unit,
                    "card_id": unit.definition.card_id,
                    "origin": unit.origin.value,
                    "derived": is_derived(unit.origin),
                    "token": is_token_definition(unit.definition) or unit.origin is CardOrigin.TOKEN,
                    "via": "play",
                    "mode_id": mode_id,
                },
            )
        )
        self._resolve_event_queue()
        mode_operations = mode_def.operations if mode_def is not None else ()
        replace_base_operations = (
            mode_def.replace_base_operations
            if mode_def is not None
            else False
        )
        if self.state.pending_choice is not None:
            self._suspended_action = "play_follower"
            self._suspended_action_state = {
                "unit_id": unit.entity_id,
                "mode_operations": mode_operations,
                "replace_base_operations": replace_base_operations,
                "burst_operations": burst_operations,
                "burst_metadata": burst_metadata,
                "burst_gauge": burst_gauge,
                "burst_replaces_base_operations": (
                    burst_replaces_base_operations
                ),
                "source_spellboost_count": hand_spellboost_count,
                "source_cost": hand_current_cost,
                "suppress_fanfare": suppress_fanfare,
                "suppress_enhance": suppress_enhance,
                "auto_evolve": auto_evolve,
            }
            return
        self._finish_follower_play(
            unit.entity_id,
            mode_operations,
            replace_base_operations,
            burst_operations,
            burst_metadata,
            burst_gauge,
            burst_replaces_base_operations,
            hand_spellboost_count,
            hand_current_cost,
            suppress_fanfare,
            suppress_enhance,
            auto_evolve,
        )

    def _finish_follower_play(
        self,
        unit_id: int,
        mode_operations: tuple[EffectOperation, ...],
        replace_base_operations: bool = False,
        burst_operations: tuple[EffectOperation, ...] = (),
        burst_metadata: tuple[tuple[str, int], ...] = (),
        burst_gauge: int = 0,
        burst_replaces_base_operations: bool = False,
        source_spellboost_count: int = 0,
        source_cost: int = 0,
        suppress_fanfare: bool = False,
        suppress_enhance: bool = False,
        auto_evolve: bool = False,
    ) -> None:
        try:
            unit = self._find_board_entity(unit_id)
        except IllegalCommand:
            return
        if not isinstance(unit, Unit):
            return
        fanfare_operations = (
            () if suppress_fanfare else self._fanfare_operations(unit)
        )
        for kind, threshold in burst_metadata:
            self._emit(
                GameEvent(
                    EventType.UNION_BURST_ACTIVATED,
                    self.current_player,
                    source_id=unit.entity_id,
                    amount=burst_gauge,
                    metadata={
                        "source": unit,
                        "card_id": unit.definition.card_id,
                        "kind": kind,
                        "threshold": threshold,
                        "gauge": burst_gauge,
                    },
                )
            )
        base_operations = (
            ()
            if replace_base_operations or burst_replaces_base_operations
            else fanfare_operations
        )
        effective_mode_operations = (
            () if suppress_enhance else mode_operations
        )
        operations = (
            base_operations
            + burst_operations
            + effective_mode_operations
        )
        if auto_evolve:
            operations += (
                EffectOperation(
                    kind=EffectKind.EVOLVE_UNIT,
                    target=TargetKind.SELF,
                ),
            )
        if operations:
            label = "入场曲"
            if burst_operations:
                label = "入场曲/奥义"
            if effective_mode_operations and (
                base_operations or burst_operations
            ):
                label = "入场曲/强化"
            elif effective_mode_operations:
                label = "强化"
            self._start_effects(
                unit.definition,
                unit.entity_id,
                operations,
                label=label,
                source_spellboost_count=source_spellboost_count,
                source_cost=source_cost,
            )

    def _play_spell(
        self,
        card: CardDefinition,
        play_cost: int,
        source_entity_id: int,
        *,
        origin: CardOrigin,
        source_origin: CardOrigin | None,
        source_spellboost_count: int = 0,
        source_cost: int = 0,
        fusion_materials: tuple[FusionMaterial, ...] = (),
        mode_def: PlayModeDefinition | None = None,
        burst_operations: tuple[EffectOperation, ...] = (),
        burst_metadata: tuple[tuple[str, int], ...] = (),
        burst_gauge: int = 0,
        burst_replaces_base_operations: bool = False,
    ) -> None:
        self._log(self.current_player, f"使用法术 {card.name}（{play_cost}费）")
        self._dispatch_card_ability(AbilityEvent.CARD_PLAYED, card)
        played_metadata = {
            "card_id": card.card_id,
            "card": card,
            "base_cost": card.cost,
            "source_cost": source_cost,
            "cost_changed": source_cost != card.cost,
        }
        if mode_def is not None:
            played_metadata["mode_id"] = mode_def.mode_id
        self._emit(
            GameEvent(
                EventType.CARD_PLAYED,
                self.current_player,
                source_id=source_entity_id,
                metadata=played_metadata,
            )
        )
        base_operations = self.rulebook.operations_for(card.card_id, Trigger.PLAY)
        mode_operations = mode_def.operations if mode_def is not None else ()
        resolved_base_operations = (
            ()
            if (
                burst_replaces_base_operations
                or (
                    mode_def is not None
                    and mode_def.replace_base_operations
                )
            )
            else base_operations
        )
        operations = resolved_base_operations + burst_operations + mode_operations
        self._emit_union_burst_activations(
            card,
            source_entity_id,
            card,
            burst_metadata,
            burst_gauge,
        )
        frame = self._queue_effects(
            card,
            None,
            operations,
            move_source_to_graveyard=True,
            label=(
                "法术/奥义/强化"
                if burst_operations and mode_def is not None
                else "法术/奥义"
                if burst_operations
                else "法术/强化"
                if mode_def is not None
                else "法术"
            ),
            fusion_materials=fusion_materials,
            source_spellboost_count=source_spellboost_count,
            source_cost=source_cost,
        )
        frame._hand_source_entity_id = source_entity_id
        frame._hand_source_origin = origin
        frame._hand_source_origin_parent = source_origin
        self._continue_effects()
        self._spellboost_pending = 1
        self._pending_spellboost_player = self.current_player
        self._pending_spellboost_source_card_id = card.card_id
        self._pending_spellboost_source_entity_id = source_entity_id
        self._try_spellboost_hand()

    def _play_amulet(
        self,
        card: CardDefinition,
        play_cost: int,
        *,
        origin: CardOrigin = CardOrigin.DECK,
        fused_material_ids: tuple[int, ...] = (),
        source_cost: int = 0,
        mode_def: PlayModeDefinition | None = None,
        burst_operations: tuple[EffectOperation, ...] = (),
        burst_metadata: tuple[tuple[str, int], ...] = (),
        burst_gauge: int = 0,
        burst_replaces_base_operations: bool = False,
    ) -> None:
        amulet = Amulet(
            definition=card,
            entity_id=self.state.allocate_entity_id(),
            countdown=self.rulebook.countdown_for(card.card_id),
            entered_turn=self.turn,
            origin=origin,
            fused_material_ids=list(fused_material_ids),
        )
        self.players[self.current_player].board.append(amulet)
        countdown = (
            f"，倒数 {amulet.countdown}" if amulet.countdown is not None else ""
        )
        self._log(
            self.current_player,
            f"打出护符 {card.name}（{play_cost}费{countdown}）",
        )
        self._dispatch_card_ability(AbilityEvent.CARD_PLAYED, card)
        played_metadata = {
            "card_id": card.card_id,
            "source": amulet,
            "base_cost": card.cost,
            "source_cost": source_cost,
            "cost_changed": source_cost != card.cost,
        }
        entered_metadata = {"source": amulet}
        if mode_def is not None:
            played_metadata["mode_id"] = mode_def.mode_id
            entered_metadata["mode_id"] = mode_def.mode_id
        self._emit(
            GameEvent(
                EventType.CARD_PLAYED,
                self.current_player,
                source_id=amulet.entity_id,
                metadata=played_metadata,
            )
        )
        self._emit(
            GameEvent(
                EventType.AMULET_ENTERED,
                self.current_player,
                source_id=amulet.entity_id,
                metadata=entered_metadata,
            )
        )
        self._initialize_earth_sigil(amulet, self.current_player)
        base_operations = self.rulebook.operations_for(card.card_id, Trigger.PLAY)
        mode_operations = mode_def.operations if mode_def is not None else ()
        resolved_base_operations = (
            ()
            if (
                burst_replaces_base_operations
                or (
                    mode_def is not None
                    and mode_def.replace_base_operations
                )
            )
            else base_operations
        )
        operations = resolved_base_operations + burst_operations + mode_operations
        self._emit_union_burst_activations(
            card,
            amulet.entity_id,
            amulet,
            burst_metadata,
            burst_gauge,
        )
        self._start_effects(
            card,
            amulet.entity_id,
            operations,
            label=(
                "入场曲/奥义/强化"
                if burst_operations and mode_def is not None
                else "入场曲/奥义"
                if burst_operations
                else "入场曲/强化"
                if mode_def is not None
                else "入场曲"
            ),
            source_cost=source_cost,
        )

    def _emit_union_burst_activations(
        self,
        card: CardDefinition,
        source_entity_id: int,
        source: object,
        burst_metadata: tuple[tuple[str, int], ...],
        burst_gauge: int,
    ) -> None:
        for kind, threshold in burst_metadata:
            self._emit(
                GameEvent(
                    EventType.UNION_BURST_ACTIVATED,
                    self.current_player,
                    source_id=source_entity_id,
                    amount=burst_gauge,
                    metadata={
                        "source": source,
                        "card_id": card.card_id,
                        "kind": kind,
                        "threshold": threshold,
                        "gauge": burst_gauge,
                    },
                )
            )

    def _play_accelerate(
        self,
        card: CardDefinition,
        play_cost: int,
        source_entity_id: int,
        origin: CardOrigin,
        source_origin: CardOrigin | None,
        mode_def,
        fusion_materials: tuple[FusionMaterial, ...] = (),
        *,
        source_cost: int,
    ) -> None:
        self._log(self.current_player, f"激奏 {card.name}（{play_cost}费）")
        self._dispatch_card_ability(AbilityEvent.CARD_PLAYED, card)
        self._emit(
            GameEvent(
                EventType.CARD_PLAYED,
                self.current_player,
                source_id=source_entity_id,
                metadata={
                    "card_id": card.card_id,
                    "card": card,
                    "mode_id": mode_def.mode_id,
                    "base_cost": card.cost,
                    "source_cost": source_cost,
                    "cost_changed": source_cost != card.cost,
                },
            )
        )
        ops = mode_def.operations if mode_def else ()
        frame = self._queue_effects(
            card,
            None,
            ops,
            move_source_to_graveyard=True,
            label="激奏",
            fusion_materials=fusion_materials,
        )
        frame._hand_source_entity_id = source_entity_id
        frame._hand_source_origin = origin
        frame._hand_source_origin_parent = source_origin
        self._continue_effects()
        self._spellboost_pending = 1
        self._pending_spellboost_player = self.current_player
        self._pending_spellboost_source_card_id = card.card_id
        self._pending_spellboost_source_entity_id = source_entity_id
        self._try_spellboost_hand()

    def _play_crystallize(
        self,
        card: CardDefinition,
        play_cost: int,
        source_entity_id: int,
        origin: CardOrigin,
        source_origin: CardOrigin | None,
        mode_def,
        fused_material_ids: tuple[int, ...] = (),
        *,
        source_cost: int,
    ) -> None:
        countdown = mode_def.countdown if mode_def else None
        amulet = Amulet(
            definition=card,
            entity_id=source_entity_id,
            countdown=countdown,
            play_mode_id=mode_def.mode_id if mode_def is not None else None,
            entered_turn=self.turn,
            origin=origin,
            source_origin=source_origin,
            fused_material_ids=list(fused_material_ids),
        )
        self.players[self.current_player].board.append(amulet)
        cd_str = f"，倒数 {amulet.countdown}" if amulet.countdown is not None else ""
        self._log(
            self.current_player,
            f"结晶 {card.name}（{play_cost}费{cd_str}）",
        )
        self._dispatch_card_ability(AbilityEvent.CARD_PLAYED, card)
        self._emit(
            GameEvent(
                EventType.CARD_PLAYED,
                self.current_player,
                source_id=amulet.entity_id,
                metadata={
                    "card_id": card.card_id,
                    "source": amulet,
                    "mode_id": mode_def.mode_id,
                    "base_cost": card.cost,
                    "source_cost": source_cost,
                    "cost_changed": source_cost != card.cost,
                },
            )
        )
        self._emit(
            GameEvent(
                EventType.AMULET_ENTERED,
                self.current_player,
                source_id=amulet.entity_id,
                metadata={"source": amulet},
            )
        )
        self._initialize_earth_sigil(amulet, self.current_player)
        ops = mode_def.operations if mode_def else ()
        self._start_effects(card, amulet.entity_id, ops, label="结晶")

    def _start_effects(
        self,
        card: CardDefinition,
        source_entity_id: int | None,
        operations: tuple[EffectOperation, ...],
        *,
        controller: int | None = None,
        move_source_to_graveyard: bool = False,
        label: str = "效果",
        source_spellboost_count: int = 0,
        source_cost: int | None = None,
        attack_target_entity_id: int | None = None,
        source_snapshot: SourceStateSnapshot | None = None,
    ) -> None:
        self._queue_effects(
            card,
            source_entity_id,
            operations,
            controller=controller,
            move_source_to_graveyard=move_source_to_graveyard,
            label=label,
            source_spellboost_count=source_spellboost_count,
            source_cost=source_cost,
            attack_target_entity_id=attack_target_entity_id,
            source_snapshot=source_snapshot,
        )
        self._continue_effects()

    def _queue_effects(
        self,
        card: CardDefinition,
        source_entity_id: int | None,
        operations: tuple[EffectOperation, ...],
        *,
        controller: int | None = None,
        move_source_to_graveyard: bool = False,
        label: str = "效果",
        fusion_materials: tuple[FusionMaterial, ...] | None = None,
        source_snapshot: SourceStateSnapshot | None = None,
        source_spellboost_count: int = 0,
        source_cost: int | None = None,
        attack_target_entity_id: int | None = None,
    ) -> EffectFrame:
        resolved_controller = self.current_player if controller is None else controller
        if fusion_materials is None:
            fusion_materials = ()
            if source_entity_id is not None:
                try:
                    source = self._find_board_entity(source_entity_id)
                except IllegalCommand:
                    source = None
                if source is not None and source.fused_material_ids:
                    owner = self._entity_owner(source.entity_id)
                    fusion_materials = self._fusion_material_records(
                        self.players[owner],
                        source.fused_material_ids,
                    )
        frame = EffectFrame(
            controller=resolved_controller,
            source_card_id=card.card_id,
            source_name=card.name,
            source_entity_id=source_entity_id,
            source_card=card,
            operations=operations,
            source_snapshot=source_snapshot,
            source_spellboost_count=source_spellboost_count,
            source_cost=(
                getattr(card, "cost", 0)
                if source_cost is None
                else source_cost
            ),
            fusion_materials=fusion_materials,
            label=label,
            move_source_to_graveyard=move_source_to_graveyard,
            attack_target_entity_id=attack_target_entity_id,
        )
        self.state.effect_stack.append(frame)
        return frame

    def _queue_effects_from_frame(
        self,
        parent: EffectFrame,
        operations: tuple[EffectOperation, ...],
        *,
        label: str,
        distributed_value: int | None = None,
    ) -> EffectFrame:
        child = self._queue_effects(
            parent.source_card,
            parent.source_entity_id,
            operations,
            controller=parent.controller,
            label=label,
            fusion_materials=parent.fusion_materials,
            source_snapshot=parent.source_snapshot,
            source_spellboost_count=parent.source_spellboost_count,
            source_cost=parent.source_cost,
        )
        child.listener_activation_owner = parent.listener_activation_owner
        child.listener_activation_zone = parent.listener_activation_zone
        child.listener_activation_entity_id = (
            parent.listener_activation_entity_id
        )
        child.listener_activation_card_id = parent.listener_activation_card_id
        child.listener_activation_definition_index = (
            parent.listener_activation_definition_index
        )
        child.listener_activation_count = parent.listener_activation_count
        child.event_source_entity_id = parent.event_source_entity_id
        child.event_source_base_cost = parent.event_source_base_cost
        child.attack_target_entity_id = parent.attack_target_entity_id
        child.distributed_value = (
            parent.distributed_value
            if distributed_value is None
            else distributed_value
        )
        inherited_keys = {
            key
            for key, target_ids in parent._target_bindings.items()
            if target_ids
        }
        child._target_bindings = {
            key: parent._target_bindings[key]
            for key in inherited_keys
        }
        child._target_binding_operations = {
            key: parent._target_binding_operations[key]
            for key in inherited_keys
        }
        child._target_binding_snapshots = {
            key: parent._target_binding_snapshots[key]
            for key in inherited_keys
        }
        return child

    def _debug_value(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, CardDefinition):
            return {
                "card_id": value.card_id,
                "name": value.name,
                "card_type": value.card_type,
            }
        if isinstance(value, SourceStateSnapshot):
            return {
                "entity_id": value.entity_id,
                "controller": value.controller,
                "card_id": value.card_id,
                "card_type": value.card_type,
                "attack": value.attack,
                "health": value.health,
                "evolved": value.evolved,
                "super_evolved": value.super_evolved,
                "effective_keywords": tuple(sorted(value.effective_keywords)),
            }
        if isinstance(value, (Unit, Amulet, HandCard, GraveyardCard, FusionMaterial)):
            definition = getattr(value, "definition", None)
            return {
                "entity_id": getattr(value, "entity_id", None),
                "card_id": None if definition is None else definition.card_id,
                "name": None if definition is None else definition.name,
                "zone_type": type(value).__name__,
            }
        if isinstance(value, (list, tuple)):
            return [self._debug_value(item) for item in value[:5]]
        if isinstance(value, dict):
            return {
                str(key): self._debug_value(value[key])
                for key in sorted(value, key=str)
            }
        return repr(value)

    def _event_debug_summary(self, event: GameEvent) -> dict:
        return {
            "type": event.type.value,
            "player_index": event.player_index,
            "source_id": event.source_id,
            "target_id": event.target_id,
            "amount": event.amount,
            "metadata": {
                key: self._debug_value(event.metadata[key])
                for key in sorted(event.metadata, key=str)
            },
        }

    def _operation_debug_summary(self, operation: EffectOperation) -> dict:
        summary = {
            "kind": operation.kind.value,
            "target": operation.target.value,
            "amount": operation.amount,
            "secondary_amount": operation.secondary_amount,
            "requires_target": operation.requires_target,
            "requires_full_target_count": operation.requires_full_target_count,
            "target_count": operation.target_count,
            "allow_duplicate_targets": operation.allow_duplicate_targets,
            "exclude_source": operation.exclude_source,
            "exclude_attack_target": operation.exclude_attack_target,
            "include_leader": operation.include_leader,
        }
        if operation.card_id is not None:
            summary["card_id"] = operation.card_id
        if operation.emblem_id is not None:
            summary["emblem_id"] = operation.emblem_id
        if operation.faith_id is not None:
            summary["faith_id"] = operation.faith_id
        if operation.faith_ability_id is not None:
            summary["faith_ability_id"] = operation.faith_ability_id
            summary["faith_trigger"] = operation.faith_trigger
            summary["faith_stacking"] = operation.faith_stacking
        if operation.target_key is not None:
            summary["target_key"] = operation.target_key
        if operation.bind_successful_targets:
            summary["bind_successful_targets"] = True
        if operation.condition_target_key is not None:
            summary["condition_target_key"] = operation.condition_target_key
        if operation.keyword is not None:
            summary["keyword"] = operation.keyword
        if operation.keywords:
            summary["keywords"] = operation.keywords
        if operation.amount_expr is not None:
            summary["amount_expr"] = operation.amount_expr.type.value
        if operation.secondary_expr is not None:
            summary["secondary_expr"] = operation.secondary_expr.type.value
        if operation.target_count_expr is not None:
            summary["target_count_expr"] = operation.target_count_expr.type.value
        if operation.turn_end_destroy_timing is not None:
            summary["turn_end_destroy_timing"] = (
                operation.turn_end_destroy_timing.value
            )
        if operation.turn_end_banish_timing is not None:
            summary["turn_end_banish_timing"] = (
                operation.turn_end_banish_timing.value
            )
        nested_counts = {
            "earth_rite": len(operation.earth_rite_operations),
            "necromancy": len(operation.necromancy_operations),
            "faith": len(operation.faith_operations),
            "then": len(operation.then_operations),
            "else": len(operation.else_operations),
            "choose_one": len(operation.choose_one_options),
            "optional": len(operation.optional_operations),
            "repeat": len(operation.repeat_operations),
            "random_choice": sum(
                len(option.operations)
                for option in operation.random_choice_options
            ),
            "random_distribution": sum(
                len(bucket)
                for bucket in operation.random_distribution_operations
            ),
        }
        if any(nested_counts.values()):
            summary["nested_counts"] = nested_counts
        return summary

    def _frame_debug_summary(self, frame: EffectFrame) -> dict:
        upcoming = frame.operations[frame.next_index:frame.next_index + 3]
        return {
            "controller": frame.controller,
            "source_card_id": frame.source_card_id,
            "source_name": frame.source_name,
            "source_entity_id": frame.source_entity_id,
            "source_snapshot": self._debug_value(frame.source_snapshot),
            "source_spellboost_count": frame.source_spellboost_count,
            "source_cost": frame.source_cost,
            "label": frame.label,
            "next_index": frame.next_index,
            "operation_count": len(frame.operations),
            "pending_target_id": frame.pending_target_id,
            "pending_target_ids": tuple(frame.pending_target_ids),
            "fusion_material_count": len(frame.fusion_materials),
            "listener_batch_id": frame.listener_batch_id,
            "listener_zone": frame.listener_activation_zone,
            "event_source_entity_id": frame.event_source_entity_id,
            "attack_target_entity_id": frame.attack_target_entity_id,
            "upcoming_operations": [
                self._operation_debug_summary(operation)
                for operation in upcoming
            ],
        }

    def _death_record_debug_summary(self, record: DeathRecord) -> dict:
        return {
            "owner": record.owner,
            "entity_id": record.entity_id,
            "card_id": record.card_id,
            "card_name": record.card_name,
            "cause": record.cause.value,
            "source_player": record.source_player,
            "source_entity_id": record.source_entity_id,
            "allows_last_words": record.allows_last_words,
            "attack": record.attack,
            "health": record.health,
            "evolved": record.evolved,
            "super_evolved": record.super_evolved,
            "effective_keywords": tuple(sorted(record.effective_keywords)),
        }

    def _emblem_batch_debug_summary(
        self,
        batch_id: int,
        batch: dict[str, object],
    ) -> dict:
        records = batch.get("records", [])
        record_summaries = []
        if isinstance(records, list):
            for record in records[:5]:
                if (
                    isinstance(record, tuple)
                    and len(record) >= 4
                ):
                    owner, entity_id, trigger_index, event_type = record[:4]
                    record_summaries.append({
                        "owner": owner,
                        "emblem_entity_id": entity_id,
                        "trigger_index": trigger_index,
                        "trigger": event_type,
                        "conditions_frozen": (
                            bool(record[4]) if len(record) >= 5 else False
                        ),
                    })
                else:
                    record_summaries.append(self._debug_value(record))
        return {
            "batch_id": batch_id,
            "record_count": len(records) if isinstance(records, list) else None,
            "records": record_summaries,
            "source_id": batch.get("source_id"),
            "attack_target_entity_id": batch.get("attack_target_entity_id"),
            "event_player": batch.get("event_player"),
            "trigger_batch_id": batch.get("trigger_batch_id"),
            "trigger_batch_order_index": batch.get("trigger_batch_order_index"),
            "trigger_batch_record_count": batch.get("trigger_batch_record_count"),
        }

    def _suspended_event_debug_summary(self) -> dict | None:
        state = self._suspended_event_state
        if state is None:
            return None
        remaining = state.get("remaining_events", [])
        event = state.get("event")
        return {
            "phase": state.get("phase"),
            "event": (
                self._event_debug_summary(event)
                if isinstance(event, GameEvent)
                else self._debug_value(event)
            ),
            "remaining_event_count": (
                len(remaining) if isinstance(remaining, list) else None
            ),
            "remaining_events": [
                self._event_debug_summary(candidate)
                for candidate in remaining[:5]
                if isinstance(candidate, GameEvent)
            ] if isinstance(remaining, list) else [],
        }

    def _suspended_death_batch_debug_summary(self) -> dict | None:
        batch = self._suspended_batch
        if batch is None:
            return None
        return {
            "batch_id": batch.batch_id,
            "record_count": len(batch.records),
            "records": [
                self._death_record_debug_summary(record)
                for record in batch.records[:5]
            ],
            "current_record": (
                None
                if self._suspended_record is None
                else self._death_record_debug_summary(self._suspended_record)
            ),
            "remaining_lw_count": len(self._suspended_lw_records),
            "remaining_lw_records": [
                self._death_record_debug_summary(record)
                for record in self._suspended_lw_records[:5]
            ],
        }

    def _loop_diagnostics(self) -> dict:
        pending_choice = None
        if self.state.pending_choice is not None:
            pending_choice = {
                "player_index": self.state.pending_choice.player_index,
                "choice_kind": self.state.pending_choice.choice_kind.value,
                "request_id": self.state.pending_choice.request_id,
                "option_count": len(self.state.pending_choice.options),
                "target_count": self.state.pending_choice.target_count,
                "selected_count": len(self.state.pending_choice.selected_options),
                "allow_duplicate_targets": (
                    self.state.pending_choice.allow_duplicate_targets
                ),
                "options": [
                    {
                        "option_id": option.option_id,
                        "entity_id": option.entity_id,
                        "leader_player_index": option.leader_player_index,
                    }
                    for option in self.state.pending_choice.options[:5]
                ],
            }
        return {
            "turn": self.turn,
            "active_player": self.current_player,
            "resolution_steps": self.state.resolution_steps,
            "limit": MAX_RESOLUTION_STEPS,
            "pending_choice": pending_choice,
            "recent_events": [
                self._event_debug_summary(event)
                for event in self.event_history[-20:]
            ],
            "event_queue": [
                self._event_debug_summary(event)
                for event in list(self.state.event_queue)[:20]
            ],
            "effect_stack": [
                self._frame_debug_summary(frame)
                for frame in self.state.effect_stack[-5:]
            ],
            "death_queue": [
                {
                    "batch_id": batch.batch_id,
                    "record_count": len(batch.records),
                    "records": [
                        self._death_record_debug_summary(record)
                        for record in batch.records[:5]
                    ],
                }
                for batch in self.state.death_queue[-3:]
            ],
            "emblem_batches": [
                self._emblem_batch_debug_summary(batch_id, batch)
                for batch_id, batch in sorted(self._emblem_batches.items())[-5:]
            ],
            "recent_emblem_triggers": [
                self._event_debug_summary(event)
                for event in self.event_history[-40:]
                if event.type is EventType.EMBLEM_TRIGGERED
            ][-10:],
            "listener_batches": [
                {
                    "batch_id": batch_id,
                    "event_type": batch.get("event_type"),
                    "event_player": batch.get("event_player"),
                    "event_source_id": batch.get("event_source_id"),
                    "record_count": len(batch.get("records", [])),
                    "records": self._debug_value(batch.get("records", [])[:5]),
                }
                for batch_id, batch in sorted(self._listener_batches.items())[-5:]
            ],
            "recent_card_listener_triggers": [
                self._event_debug_summary(event)
                for event in self.event_history[-40:]
                if event.type is EventType.CARD_LISTENER_TRIGGERED
            ][-10:],
            "suspended_event": self._suspended_event_debug_summary(),
            "suspended_death_batch": self._suspended_death_batch_debug_summary(),
            "suspended": {
                "action": self._suspended_action,
                "has_action_state": self._suspended_action_state is not None,
                "has_event_state": self._suspended_event_state is not None,
                "has_death_batch": self._suspended_batch is not None,
                "has_death_record": self._suspended_record is not None,
            },
            "active_super_evolution_attack": (
                None
                if self._active_super_evolution_attack is None
                else {
                    "controller": self._active_super_evolution_attack.controller,
                    "attacker_id": self._active_super_evolution_attack.attacker_id,
                    "target_id": self._active_super_evolution_attack.target_id,
                    "attacker_card_id": (
                        self._active_super_evolution_attack.attacker_card_id
                    ),
                    "bonus_resolved": (
                        self._active_super_evolution_attack.bonus_resolved
                    ),
                }
            ),
            "logs_tail": self.logs[-10:],
        }

    def deterministic_fingerprint(self) -> dict[str, object]:
        """Return a comparable full-state summary for replay diagnostics.

        The fingerprint intentionally includes hidden state such as deck order
        and RNG state. It is for engine tests and debugging, not RL
        observations or public match info.
        """
        return {
            "state": {
                "active_player": self.state.active_player,
                "first_player": self.state.first_player,
                "mulligan_completed": tuple(self.state.mulligan_completed),
                "turn": self.state.turn,
                "phase": self.state.phase.value,
                "winner": self.state.winner,
                "resolution_steps": self.state.resolution_steps,
                "next_entity_id": self.state.next_entity_id,
                "next_death_sequence": self.state._next_death_sequence,
                "next_follower_entry_sequence": (
                    self.state._next_follower_entry_sequence
                ),
                "players": tuple(
                    self._player_fingerprint(player)
                    for player in self.state.players
                ),
                "event_queue": tuple(
                    self._event_fingerprint(event)
                    for event in self.state.event_queue
                ),
                "pending_choice": self._choice_fingerprint(
                    self.state.pending_choice
                ),
                "effect_stack": tuple(
                    self._effect_frame_fingerprint(frame)
                    for frame in self.state.effect_stack
                ),
                "death_queue": tuple(
                    self._death_batch_fingerprint(batch)
                    for batch in self.state.death_queue
                ),
                "destroyed_followers": tuple(
                    self._destroyed_follower_fingerprint(record)
                    for record in self.state.destroyed_followers
                ),
                "destroyed_amulets": tuple(
                    self._destroyed_amulet_fingerprint(record)
                    for record in self.state.destroyed_amulets
                ),
                "follower_entries": tuple(
                    self._follower_entry_fingerprint(record)
                    for record in self.state.follower_entries
                ),
                "listener_activation_counts": tuple(
                    sorted(self.state.listener_activation_counts.items())
                ),
                "listener_once_per_turn_used": tuple(
                    sorted(self.state.listener_once_per_turn_used)
                ),
            },
            "logs": tuple(self.logs),
            "event_history": tuple(
                self._event_fingerprint(event)
                for event in self.event_history
            ),
            "placeholder_ability_events": tuple(
                self._placeholder_event_fingerprint(event)
                for event in self.placeholder_ability_events
            ),
            "rng_state": self.random.getstate(),
            "internals": {
                "death_causes": tuple(
                    (entity_id, cause.value)
                    for entity_id, cause in sorted(self._death_causes.items())
                ),
                "suspended_batch": self._death_batch_fingerprint(
                    self._suspended_batch
                ),
                "suspended_record": self._death_record_fingerprint(
                    self._suspended_record
                ),
                "suspended_lw_records": tuple(
                    self._death_record_fingerprint(record)
                    for record in self._suspended_lw_records
                ),
                "defer_last_words": self._defer_last_words,
                "deferred_death_batches": tuple(
                    (
                        self._death_batch_fingerprint(batch),
                        tuple(
                            self._death_record_fingerprint(record)
                            for record in records
                        ),
                    )
                    for batch, records in self._deferred_death_batches
                ),
                "suspended_action": self._suspended_action,
                "suspended_action_state": self._fingerprint_value(
                    self._suspended_action_state
                ),
                "suspended_event_state": self._fingerprint_value(
                    self._suspended_event_state
                ),
                "spellboost_pending": self._spellboost_pending,
                "pending_spellboost_player": self._pending_spellboost_player,
                "pending_spellboost_source_card_id": (
                    self._pending_spellboost_source_card_id
                ),
                "pending_spellboost_source_entity_id": (
                    self._pending_spellboost_source_entity_id
                ),
                "emblem_batches": tuple(
                    (
                        batch_id,
                        self._fingerprint_value(batch),
                    )
                    for batch_id, batch in sorted(self._emblem_batches.items())
                ),
                "next_emblem_batch_id": self._next_emblem_batch_id,
                "listener_batches": tuple(
                    (
                        batch_id,
                        self._fingerprint_value(batch),
                    )
                    for batch_id, batch in sorted(self._listener_batches.items())
                ),
                "next_listener_batch_id": self._next_listener_batch_id,
                "emblem_expiration_batches": tuple(
                    (
                        batch_id,
                        self._fingerprint_value(batch),
                    )
                    for batch_id, batch in sorted(
                        self._emblem_expiration_batches.items()
                    )
                ),
                "next_emblem_expiration_batch_id": (
                    self._next_emblem_expiration_batch_id
                ),
                "stabilizing": self._stabilizing,
                "next_modifier_id": self._next_modifier_id,
                "next_choice_request_id": self._next_choice_request_id,
                "active_super_evolution_attack": (
                    None
                    if self._active_super_evolution_attack is None
                    else (
                        self._active_super_evolution_attack.controller,
                        self._active_super_evolution_attack.attacker_id,
                        self._active_super_evolution_attack.target_id,
                        self._active_super_evolution_attack.attacker_card_id,
                        self._active_super_evolution_attack.attacker_name,
                        self._active_super_evolution_attack.bonus_resolved,
                    )
                ),
            },
        }

    def _player_fingerprint(self, player: PlayerState) -> dict[str, object]:
        return {
            "class_id": player.class_id,
            "class_name": player.class_name,
            "health": player.health,
            "max_health": player.max_health,
            "max_mana": player.max_mana,
            "mana": player.mana,
            "extra_pp_available": player.extra_pp_available,
            "extra_pp_uses": player.extra_pp_uses,
            "extra_pp_refresh_done": player.extra_pp_refresh_done,
            "extra_pp_active_turn": player.extra_pp_active_turn,
            "fatigue": player.fatigue,
            "empty_deck_outcome": player.empty_deck_outcome.value,
            "evolution_points": player.evolution_points,
            "super_evolution_points": player.super_evolution_points,
            "turns_started": player.turns_started,
            "evolved_this_turn": player.evolved_this_turn,
            "super_evolved_this_turn": player.super_evolved_this_turn,
            "followers_evolved_this_match": player.followers_evolved_this_match,
            "cards_played_this_turn": player.cards_played_this_turn,
            "follower_attacks_this_turn": player.follower_attacks_this_turn,
            "followers_destroyed_this_turn": (
                player.followers_destroyed_this_turn
            ),
            "cooperation": player.cooperation,
            "shadows": player.shadows,
            "leader_barrier_charges": player.leader_barrier_charges,
            "leader_damage_modifiers": tuple(
                (
                    modifier.modifier_id,
                    modifier.amount,
                    modifier.duration,
                    modifier.expires_for_player,
                    modifier.source_controller,
                    modifier.source_entity_id,
                    modifier.source_card_id,
                    modifier.mode,
                )
                for modifier in player.leader_damage_modifiers
            ),
            "next_graveyard_sequence": player._next_graveyard_sequence,
            "next_emblem_sequence": player._next_emblem_sequence,
            "next_faith_sequence": player._next_faith_sequence,
            "next_fusion_sequence": player._next_fusion_sequence,
            "deck": tuple(
                self._deck_card_fingerprint(card)
                for card in player.deck
            ),
            "hand": tuple(
                self._hand_card_fingerprint(card)
                for card in player.hand
            ),
            "hand_entity_ids": tuple(player.hand_entity_ids),
            "fusion_materials": tuple(
                self._fusion_material_fingerprint(material)
                for material in player.fusion_materials
            ),
            "board": tuple(
                self._board_entity_fingerprint(entity)
                for entity in player.board
            ),
            "graveyard": tuple(
                self._graveyard_card_fingerprint(card)
                for card in player.graveyard
            ),
            "banished": tuple(
                self._card_fingerprint(card)
                for card in player.banished
            ),
            "emblems": tuple(
                self._emblem_instance_fingerprint(emblem)
                for emblem in player.emblems
            ),
            "faiths": tuple(
                self._faith_instance_fingerprint(faith)
                for faith in player.faiths
            ),
        }

    def _card_fingerprint(
        self,
        card: CardDefinition | None,
    ) -> tuple[object, ...] | None:
        if card is None:
            return None
        return (
            card.card_id,
            card.card_set_id,
            card.class_id,
            card.class_name,
            card.tribe_id,
            card.tribe_name,
            card.name,
            card.cost,
            card.card_type,
            card.attack,
            card.life,
            tuple(sorted(card.keywords)),
            card.support_level,
            card.is_collectible,
            tuple(
                (
                    effect.kind,
                    effect.amount,
                    effect.secondary_amount,
                )
                for effect in card.fanfare_effects
            ),
            tuple(sorted(ability.value for ability in card.ability_keywords)),
        )

    def _deck_card_fingerprint(
        self,
        card: CardDefinition | DeckCard,
    ) -> object:
        if not isinstance(card, DeckCard):
            return self._card_fingerprint(card)
        return {
            "zone_type": "DeckCard",
            "definition": self._card_fingerprint(card.definition),
            "current_cost": card.current_cost,
            "attack": card.attack,
            "life": card.life,
            "cost_modifiers": tuple(
                self._cost_modifier_fingerprint(modifier)
                for modifier in card.cost_modifiers
            ),
            "stat_modifiers": tuple(
                self._stat_modifier_fingerprint(modifier)
                for modifier in card.stat_modifiers
            ),
        }

    def _hand_card_fingerprint(self, card) -> dict[str, object]:
        if not isinstance(card, HandCard):
            return {
                "zone_type": type(card).__name__,
                "definition": self._card_fingerprint(card),
                "entity_id": None,
            }
        return {
            "zone_type": "HandCard",
            "definition": self._card_fingerprint(card.definition),
            "entity_id": card.entity_id,
            "current_cost": card.current_cost,
            "spellboost_count": card.spellboost_count,
            "spellboost_cost_reduction": card.spellboost_cost_reduction,
            "cannot_be_played": card.cannot_be_played,
            "origin": card.origin.value,
            "source_origin": (
                None if card.source_origin is None else card.source_origin.value
            ),
            "cost_modifiers": tuple(
                self._cost_modifier_fingerprint(modifier)
                for modifier in card.cost_modifiers
            ),
            "stat_modifiers": tuple(
                self._stat_modifier_fingerprint(modifier)
                for modifier in card.stat_modifiers
            ),
            "fused_material_ids": tuple(card.fused_material_ids),
            "fusion_used_turn": card.fusion_used_turn,
            "evolutions_while_in_hand": card.evolutions_while_in_hand,
            "union_burst_gauge_bonus": card.union_burst_gauge_bonus,
            "effective_keywords": tuple(sorted(card.effective_keywords)),
            "printed_keyword_overrides": tuple(
                sorted(card.printed_keyword_overrides)
            ),
            "permanent_keywords": tuple(sorted(card.permanent_keywords)),
            "temporary_keywords": tuple(
                (
                    modifier.keyword,
                    modifier.duration,
                    modifier.expires_for_player,
                )
                for modifier in card.temporary_keywords
            ),
            "removed_keywords": tuple(sorted(card.removed_keywords)),
            "temporary_keyword_removals": tuple(
                (
                    modifier.keyword,
                    modifier.duration,
                    modifier.expires_for_player,
                )
                for modifier in card.temporary_keyword_removals
            ),
            "granted_last_words": tuple(
                tuple(
                    self._operation_fingerprint(operation)
                    for operation in granted_ability
                )
                for granted_ability in card.granted_last_words
            ),
            "effect_destroy_immunity": card.effect_destroy_immunity,
        }

    def _board_entity_fingerprint(
        self,
        entity: BoardCard,
    ) -> dict[str, object]:
        base = {
            "zone_type": type(entity).__name__,
            "definition": self._card_fingerprint(entity.definition),
            "entity_id": entity.entity_id,
            "origin": entity.origin.value,
            "source_origin": (
                None if entity.source_origin is None else entity.source_origin.value
            ),
            "fused_material_ids": tuple(entity.fused_material_ids),
        }
        if isinstance(entity, Unit):
            base.update({
                "attack": entity.attack,
                "health": entity.health,
                "max_health": entity.max_health,
                "base_attack": entity.base_attack,
                "base_health": entity.base_health,
                "can_attack": entity.can_attack,
                "attacks_remaining": entity.attacks_remaining,
                "attacks_per_turn": entity.attacks_per_turn,
                "attack_capacity_modifiers": tuple(
                    (
                        modifier.attacks_per_turn,
                        modifier.duration,
                        modifier.expires_for_player,
                    )
                    for modifier in entity.attack_capacity_modifiers
                ),
                "evolved": entity.evolved,
                "super_evolved": entity.super_evolved,
                "super_evolved_turn": entity.super_evolved_turn,
                "rush_only": entity.rush_only,
                "barrier_charges": entity.barrier_charges,
                "ambush_active": entity.ambush_active,
                "summoned_this_turn": entity.summoned_this_turn,
                "permanent_keywords": tuple(sorted(entity.permanent_keywords)),
                "temporary_keywords": tuple(
                    self._keyword_modifier_fingerprint(modifier)
                    for modifier in entity.temporary_keywords
                ),
                "removed_keywords": tuple(sorted(entity.removed_keywords)),
                "temporary_keyword_removals": tuple(
                    self._keyword_removal_fingerprint(modifier)
                    for modifier in entity.temporary_keyword_removals
                ),
                "stat_modifiers": tuple(
                    self._stat_modifier_fingerprint(modifier)
                    for modifier in entity.stat_modifiers
                ),
                "attack_restrictions": tuple(
                    self._attack_restriction_fingerprint(modifier)
                    for modifier in entity.attack_restrictions
                ),
                "targeting_restrictions": tuple(
                    self._targeting_restriction_fingerprint(modifier)
                    for modifier in entity.targeting_restrictions
                ),
                "printed_abilities_removed": entity.printed_abilities_removed,
                "last_words_removed": entity.last_words_removed,
                "granted_last_words": tuple(
                    tuple(
                        self._operation_fingerprint(operation)
                        for operation in granted_ability
                    )
                    for granted_ability in entity.granted_last_words
                ),
                "effect_destroy_immunity": entity.effect_destroy_immunity,
                "turn_end_destroy_timings": tuple(
                    sorted(
                        timing.value
                        for timing in entity.turn_end_destroy_timings
                    )
                ),
                "turn_end_banish_timings": tuple(
                    sorted(
                        timing.value
                        for timing in entity.turn_end_banish_timings
                    )
                ),
                "granted_turn_end_abilities": tuple(
                    (
                        ability.timing.value,
                        tuple(
                            self._operation_fingerprint(operation)
                            for operation in ability.operations
                        ),
                    )
                    for ability in entity.granted_turn_end_abilities
                ),
                "random_choice_history": tuple(
                    sorted(entity.random_choice_history.items())
                ),
            })
        elif isinstance(entity, Amulet):
            base.update({
                "countdown": entity.countdown,
                "play_mode_id": entity.play_mode_id,
                "earth_sigil_count": entity.earth_sigil_count,
                "entered_turn": entity.entered_turn,
                "activated_turn": entity.activated_turn,
                "pending_destroy": entity.pending_destroy,
            })
        return base

    def _fusion_material_fingerprint(
        self,
        material: FusionMaterial,
    ) -> tuple[object, ...]:
        return (
            self._card_fingerprint(material.definition),
            material.entity_id,
            material.owner,
            material.consumed_sequence,
            material.fused_into_entity_id,
            material.origin.value,
            None if material.source_origin is None else material.source_origin.value,
            material.inherited_material_ids,
        )

    def _graveyard_card_fingerprint(
        self,
        card: GraveyardCard,
    ) -> tuple[object, ...]:
        return (
            self._card_fingerprint(card.definition),
            card.entity_id,
            card.owner,
            card.entered_sequence,
            card.entry_cause,
            card.derived,
            card.origin.value,
            card.token,
            None if card.source_origin is None else card.source_origin.value,
        )

    def _destroyed_follower_fingerprint(
        self,
        record: DestroyedFollowerRecord,
    ) -> tuple[object, ...]:
        return (
            self._card_fingerprint(record.definition),
            record.owner,
            record.death_sequence,
            record.cause.value,
            record.derived,
            record.token,
            record.origin.value,
            None if record.source_origin is None else record.source_origin.value,
            record.destroyed_turn,
        )

    def _destroyed_amulet_fingerprint(
        self,
        record: DestroyedAmuletRecord,
    ) -> tuple[object, ...]:
        return (
            self._card_fingerprint(record.definition),
            record.owner,
            record.death_sequence,
            record.cause.value,
            record.derived,
            record.token,
            record.origin.value,
            None if record.source_origin is None else record.source_origin.value,
            record.destroyed_turn,
            record.play_mode_id,
            record.summon_countdown,
        )

    def _follower_entry_fingerprint(
        self,
        record: FollowerEntryRecord,
    ) -> tuple[object, ...]:
        return (
            self._card_fingerprint(record.definition),
            record.owner,
            record.entry_sequence,
            record.entered_turn,
            record.entry_cause,
        )

    def _event_fingerprint(self, event: GameEvent) -> tuple[object, ...]:
        return (
            event.type.value,
            event.player_index,
            event.source_id,
            event.target_id,
            event.amount,
            self._fingerprint_value(event.metadata),
            event.listener_sources,
            event.emblem_sources,
        )

    def _choice_fingerprint(
        self,
        request: ChoiceRequest | None,
    ) -> dict[str, object] | None:
        if request is None:
            return None
        return {
            "player_index": request.player_index,
            "prompt": request.prompt,
            "continuation_id": request.continuation_id,
            "choice_kind": request.choice_kind.value,
            "request_id": request.request_id,
            "target_count": request.target_count,
            "allow_duplicate_targets": request.allow_duplicate_targets,
            "selected_options": tuple(
                (
                    option.option_id,
                    option.label,
                    option.entity_id,
                    option.leader_player_index,
                )
                for option in request.selected_options
            ),
            "options": tuple(
                (
                    option.option_id,
                    option.label,
                    option.entity_id,
                    option.leader_player_index,
                )
                for option in request.options
            ),
        }

    def _effect_frame_fingerprint(
        self,
        frame: EffectFrame,
    ) -> dict[str, object]:
        return {
            "controller": frame.controller,
            "source_card_id": frame.source_card_id,
            "source_name": frame.source_name,
            "source_entity_id": frame.source_entity_id,
            "source_card": self._card_fingerprint(frame.source_card),
            "source_snapshot": self._fingerprint_value(frame.source_snapshot),
            "source_spellboost_count": frame.source_spellboost_count,
            "source_cost": frame.source_cost,
            "distributed_value": frame.distributed_value,
            "operations": tuple(
                self._operation_fingerprint(operation)
                for operation in frame.operations
            ),
            "fusion_materials": tuple(
                self._fusion_material_fingerprint(material)
                for material in frame.fusion_materials
            ),
            "label": frame.label,
            "next_index": frame.next_index,
            "pending_target_id": frame.pending_target_id,
            "pending_target_ids": tuple(frame.pending_target_ids),
            "move_source_to_graveyard": frame.move_source_to_graveyard,
            "all_target_ids": tuple(frame._all_target_ids),
            "all_target_index": frame._all_target_index,
            "defer_stabilize": frame.defer_stabilize,
            "auto_resolve_choices": frame.auto_resolve_choices,
            "hand_source_entity_id": frame._hand_source_entity_id,
            "hand_source_origin": self._fingerprint_value(
                frame._hand_source_origin
            ),
            "hand_source_origin_parent": self._fingerprint_value(
                frame._hand_source_origin_parent
            ),
            "target_bindings": self._fingerprint_value(frame._target_bindings),
            "target_binding_operations": self._fingerprint_value(
                frame._target_binding_operations
            ),
            "target_binding_snapshots": self._fingerprint_value(
                frame._target_binding_snapshots
            ),
            "decision_meta": self._fingerprint_value(frame._decision_meta),
            "emblem_batch_id": frame.emblem_batch_id,
            "emblem_activation_owner": frame.emblem_activation_owner,
            "emblem_activation_entity_id": frame.emblem_activation_entity_id,
            "emblem_activation_trigger_index": (
                frame.emblem_activation_trigger_index
            ),
            "listener_batch_id": frame.listener_batch_id,
            "listener_activation_owner": frame.listener_activation_owner,
            "listener_activation_zone": frame.listener_activation_zone,
            "listener_activation_entity_id": (
                frame.listener_activation_entity_id
            ),
            "listener_activation_card_id": frame.listener_activation_card_id,
            "listener_activation_definition_index": (
                frame.listener_activation_definition_index
            ),
            "listener_activation_count": frame.listener_activation_count,
            "event_source_entity_id": frame.event_source_entity_id,
            "event_source_base_cost": frame.event_source_base_cost,
            "attack_target_entity_id": frame.attack_target_entity_id,
            "emblem_expiration_batch_id": frame.emblem_expiration_batch_id,
            "expiring_emblem_owner": frame.expiring_emblem_owner,
            "expiring_emblem_entity_id": frame.expiring_emblem_entity_id,
        }

    def _operation_fingerprint(
        self,
        operation: EffectOperation,
    ) -> tuple[object, ...]:
        return (
            operation.kind.value,
            operation.target.value,
            operation.amount,
            operation.secondary_amount,
            operation.card_id,
            operation.card_ids,
            operation.shuffle,
            (
                None
                if operation.empty_deck_outcome is None
                else operation.empty_deck_outcome.value
            ),
            operation.emblem_id,
            operation.keyword,
            operation.keywords,
            operation.restriction,
            tuple(
                self._condition_fingerprint(condition)
                for condition in operation.conditions
            ),
            self._expression_fingerprint(operation.amount_expr),
            self._expression_fingerprint(operation.secondary_expr),
            None if operation.mode is None else operation.mode.value,
            operation.leader_damage_mode.value,
            operation.duration.value,
            operation.set_attack,
            operation.set_health,
            operation.target_key,
            operation.bind_successful_targets,
            operation.condition_target_key,
            tuple(
                self._operation_fingerprint(nested)
                for nested in operation.earth_rite_operations
            ),
            tuple(
                self._operation_fingerprint(nested)
                for nested in operation.necromancy_operations
            ),
            operation.faith_id,
            operation.faith_ability_id,
            operation.faith_trigger,
            operation.faith_stacking,
            tuple(
                self._operation_fingerprint(nested)
                for nested in operation.faith_operations
            ),
            operation.graveyard_cost_max,
            operation.graveyard_cost_min,
            operation.graveyard_follower_only,
            operation.graveyard_card_type,
            self._deck_filter_fingerprint(operation.deck_filter),
            self._expression_fingerprint(operation.deck_filter_cost_expr),
            self._board_filter_fingerprint(operation.board_filter),
            tuple(
                self._operation_fingerprint(nested)
                for nested in operation.then_operations
            ),
            tuple(
                self._operation_fingerprint(nested)
                for nested in operation.else_operations
            ),
            tuple(
                (
                    option.option_id,
                    option.label,
                    tuple(
                        self._condition_fingerprint(condition)
                        for condition in option.conditions
                    ),
                    tuple(
                        self._operation_fingerprint(nested)
                        for nested in option.operations
                    ),
                )
                for option in operation.choose_one_options
            ),
            operation.choose_count,
            operation.optional_prompt,
            tuple(
                self._operation_fingerprint(nested)
                for nested in operation.optional_operations
            ),
            tuple(
                self._operation_fingerprint(nested)
                for nested in operation.repeat_operations
            ),
            tuple(
                self._operation_fingerprint(nested)
                for nested in operation.granted_operations
            ),
            tuple(
                (
                    option.option_id,
                    option.label,
                    tuple(
                        self._operation_fingerprint(nested)
                        for nested in option.operations
                    ),
                )
                for option in operation.random_choice_options
            ),
            tuple(
                tuple(
                    self._operation_fingerprint(nested)
                    for nested in bucket
                )
                for bucket in operation.random_distribution_operations
            ),
            operation.random_choice_history_key,
            operation.emblem_remove_mode,
            operation.requires_target,
            operation.requires_full_target_count,
            operation.target_count,
            self._expression_fingerprint(operation.target_count_expr),
            operation.allow_duplicate_targets,
            operation.exclude_source,
            operation.exclude_attack_target,
            self._hand_filter_fingerprint(operation.hand_filter),
            self._hand_filter_fingerprint(operation.history_filter),
            operation.distinct_card_names,
            operation.highest_base_cost_only,
            operation.include_leader,
            (
                None
                if operation.candidate_extreme is None
                else operation.candidate_extreme.value
            ),
            (
                None
                if operation.turn_end_destroy_timing is None
                else operation.turn_end_destroy_timing.value
            ),
            (
                None
                if operation.turn_end_banish_timing is None
                else operation.turn_end_banish_timing.value
            ),
            (
                None
                if operation.turn_end_ability_timing is None
                else operation.turn_end_ability_timing.value
            ),
        )

    def _condition_fingerprint(
        self,
        condition: Condition,
    ) -> tuple[object, ...]:
        return (
            condition.type.value,
            condition.value,
            condition.keyword,
            condition.card_type,
            self._board_filter_fingerprint(condition.board_filter),
            self._hand_filter_fingerprint(condition.card_filter),
            tuple(
                self._condition_fingerprint(nested)
                for nested in condition.conditions
            ),
        )

    def _expression_fingerprint(
        self,
        expression: ValueExpression | None,
    ) -> tuple[object, ...] | None:
        if expression is None:
            return None
        return (
            expression.type.value,
            expression.value,
            expression.binding_key,
            self._hand_filter_fingerprint(expression.card_filter),
            self._board_filter_fingerprint(expression.board_filter),
            tuple(
                self._expression_fingerprint(value)
                for value in expression.values
            ),
        )

    def _deck_filter_fingerprint(
        self,
        deck_filter: DeckFilter | None,
    ) -> tuple[object, ...] | None:
        if deck_filter is None:
            return None
        return (
            deck_filter.card_type,
            deck_filter.class_id,
            deck_filter.class_name,
            deck_filter.cost_min,
            deck_filter.cost_max,
            deck_filter.costs,
            deck_filter.card_id,
            deck_filter.card_ids,
            deck_filter.card_name,
            deck_filter.tribe_id,
            deck_filter.tribe_name,
            deck_filter.life_min,
            deck_filter.life_max,
        )

    def _board_filter_fingerprint(
        self,
        board_filter,
    ) -> tuple[object, ...] | None:
        if board_filter is None:
            return None
        return (
            board_filter.card_type,
            board_filter.class_id,
            board_filter.class_name,
            board_filter.cost_min,
            board_filter.cost_max,
            board_filter.card_id,
            board_filter.card_name,
            board_filter.tribe_id,
            board_filter.tribe_name,
            board_filter.exclude_tribe_name,
            board_filter.evolved,
            board_filter.super_evolved,
            board_filter.damaged,
            board_filter.attacked_this_turn,
            board_filter.keyword,
        )

    def _hand_filter_fingerprint(
        self,
        hand_filter: HandFilter | None,
    ) -> tuple[object, ...] | None:
        if hand_filter is None:
            return None
        return (
            hand_filter.card_type,
            hand_filter.class_id,
            hand_filter.class_name,
            hand_filter.cost_min,
            hand_filter.cost_max,
            hand_filter.card_id,
            hand_filter.exclude_card_ids,
            hand_filter.card_name,
            hand_filter.tribe_id,
            hand_filter.tribe_name,
            hand_filter.keyword,
        )

    def _death_batch_fingerprint(
        self,
        batch: DeathBatch | None,
    ) -> tuple[object, ...] | None:
        if batch is None:
            return None
        return (
            batch.batch_id,
            tuple(
                self._death_record_fingerprint(record)
                for record in batch.records
            ),
        )

    def _death_record_fingerprint(
        self,
        record: DeathRecord | None,
    ) -> tuple[object, ...] | None:
        if record is None:
            return None
        return (
            record.owner,
            record.entity_id,
            record.card_id,
            record.card_name,
            record.card_type,
            self._card_fingerprint(record.definition),
            record.cause.value,
            record.source_player,
            record.source_entity_id,
            record.board_position,
            record.allows_last_words,
            tuple(sorted(record.effective_keywords)),
            record.attack,
            record.health,
            record.evolved,
            record.super_evolved,
            tuple(
                self._operation_fingerprint(operation)
                for operation in record.granted_last_words
            ),
        )

    def _emblem_instance_fingerprint(
        self,
        emblem,
    ) -> dict[str, object]:
        return {
            "emblem_id": emblem.emblem_id,
            "definition": self._emblem_definition_fingerprint(
                emblem.definition
            ),
            "entity_id": emblem.entity_id,
            "controller": emblem.controller,
            "created_sequence": emblem.created_sequence,
            "countdown": emblem.countdown,
            "countdown_before": emblem.countdown_before,
            "activation_counts": tuple(
                sorted(emblem.activation_counts.items())
            ),
            "random_choice_history": tuple(
                sorted(emblem.random_choice_history.items())
            ),
            "once_per_turn_used": tuple(sorted(emblem._once_per_turn_used)),
        }

    def _faith_instance_fingerprint(
        self,
        faith: FaithInstance,
    ) -> tuple[object, ...]:
        return (
            faith.faith_id,
            faith.source_card_id,
            faith.entity_id,
            faith.controller,
            faith.created_sequence,
            faith.value,
            faith.definition.initial_value,
            tuple(
                (
                    trigger.trigger.value,
                    trigger.amount,
                    (
                        None
                        if trigger.event_filter is None
                        else (
                            trigger.event_filter.card_type,
                            trigger.event_filter.class_id,
                            trigger.event_filter.class_name,
                            trigger.event_filter.tribe_id,
                            trigger.event_filter.tribe_name,
                            trigger.event_filter.cost_min,
                            trigger.event_filter.cost_max,
                            trigger.event_filter.current_costs,
                            trigger.event_filter.card_id,
                            trigger.event_filter.card_name,
                            trigger.event_filter.keyword,
                            trigger.event_filter.enhanced,
                            trigger.event_filter.cost_changed,
                        )
                    ),
                )
                for trigger in faith.definition.triggers
            ),
            tuple(
                (
                    ability.ability_id,
                    ability.trigger.value,
                    ability.granted_sequence,
                    tuple(
                        self._operation_fingerprint(operation)
                        for operation in ability.operations
                    ),
                )
                for ability in faith.granted_abilities
            ),
            faith._next_granted_ability_sequence,
            faith.mode_selection_bonus,
        )

    def _emblem_definition_fingerprint(
        self,
        definition,
    ) -> tuple[object, ...]:
        return (
            definition.emblem_id,
            definition.source_card_id,
            definition.stacking.value,
            definition.countdown,
            tuple(
                (
                    trigger.trigger,
                    tuple(
                        self._operation_fingerprint(operation)
                        for operation in trigger.operations
                    ),
                    tuple(
                        self._condition_fingerprint(condition)
                        for condition in trigger.conditions
                    ),
                    None if trigger.turn_scope is None else trigger.turn_scope.value,
                    None if trigger.event_scope is None else trigger.event_scope.value,
                    trigger.once_per_turn,
                    trigger.max_activations,
                    (
                        None
                        if trigger.event_filter is None
                        else (
                            trigger.event_filter.card_type,
                            trigger.event_filter.class_id,
                            trigger.event_filter.class_name,
                            trigger.event_filter.tribe_id,
                            trigger.event_filter.tribe_name,
                            trigger.event_filter.cost_min,
                            trigger.event_filter.cost_max,
                            trigger.event_filter.current_costs,
                            trigger.event_filter.card_id,
                            trigger.event_filter.card_name,
                            trigger.event_filter.keyword,
                            trigger.event_filter.enhanced,
                            trigger.event_filter.cost_changed,
                        )
                    ),
                )
                for trigger in definition.triggers
            ),
            tuple(
                self._operation_fingerprint(operation)
                for operation in definition.on_gain
            ),
            tuple(
                self._operation_fingerprint(operation)
                for operation in definition.on_expire
            ),
            tuple(
                self._operation_fingerprint(operation)
                for operation in definition.last_words
            ),
            tuple(sorted(passive.value for passive in definition.passives)),
        )

    def _cost_modifier_fingerprint(
        self,
        modifier: CostModifier,
    ) -> tuple[object, ...]:
        return (
            modifier.modifier_id,
            modifier.mode,
            modifier.amount,
            modifier.duration,
            modifier.expires_for_player,
        )

    def _keyword_modifier_fingerprint(
        self,
        modifier,
    ) -> tuple[object, ...]:
        return (
            modifier.keyword,
            modifier.duration,
            modifier.expires_for_player,
        )

    def _keyword_removal_fingerprint(
        self,
        modifier,
    ) -> tuple[object, ...]:
        return (
            modifier.keyword,
            modifier.duration,
            modifier.expires_for_player,
            modifier.restore_barrier_charge,
            modifier.restore_ambush,
        )

    def _stat_modifier_fingerprint(
        self,
        modifier: StatModifier,
    ) -> tuple[object, ...]:
        return (
            modifier.modifier_id,
            modifier.attack_delta,
            modifier.health_delta,
            modifier.duration,
            modifier.expires_for_player,
        )

    def _attack_restriction_fingerprint(
        self,
        modifier,
    ) -> tuple[object, ...]:
        return (
            modifier.restriction.value,
            modifier.duration,
            modifier.expires_for_player,
        )

    def _targeting_restriction_fingerprint(
        self,
        modifier,
    ) -> tuple[object, ...]:
        return (
            modifier.restriction.value,
            modifier.duration,
            modifier.expires_for_player,
        )

    def _placeholder_event_fingerprint(
        self,
        event: PlaceholderAbilityEvent,
    ) -> tuple[object, ...]:
        return (
            event.turn,
            event.player_index,
            event.card_id,
            event.card_name,
            event.ability.value,
            event.event.value,
        )

    def _fingerprint_value(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, CardDefinition):
            return self._card_fingerprint(value)
        if isinstance(value, GameEvent):
            return self._event_fingerprint(value)
        if isinstance(value, EffectOperation):
            return self._operation_fingerprint(value)
        if isinstance(value, BoundTargetSnapshot):
            return (
                value.entity_id,
                value.controller,
                value.zone,
                value.card_id,
                value.card_type,
                value.card_name,
                value.cost,
                value.attack,
                self._card_fingerprint(value.definition),
            )
        if isinstance(value, SourceStateSnapshot):
            return (
                value.entity_id,
                value.controller,
                value.card_id,
                value.card_type,
                value.attack,
                value.health,
                value.evolved,
                value.super_evolved,
                tuple(sorted(value.effective_keywords)),
            )
        if isinstance(value, Condition):
            return self._condition_fingerprint(value)
        if isinstance(value, ValueExpression):
            return self._expression_fingerprint(value)
        if isinstance(value, EffectFrame):
            return self._effect_frame_fingerprint(value)
        if isinstance(value, ChoiceRequest):
            return self._choice_fingerprint(value)
        if isinstance(value, ChoiceOption):
            return (
                value.option_id,
                value.label,
                value.entity_id,
                value.leader_player_index,
            )
        if isinstance(value, DeathBatch):
            return self._death_batch_fingerprint(value)
        if isinstance(value, DeathRecord):
            return self._death_record_fingerprint(value)
        if isinstance(value, DestroyedFollowerRecord):
            return self._destroyed_follower_fingerprint(value)
        if isinstance(value, DestroyedAmuletRecord):
            return self._destroyed_amulet_fingerprint(value)
        if isinstance(value, FollowerEntryRecord):
            return self._follower_entry_fingerprint(value)
        if isinstance(value, GraveyardCard):
            return self._graveyard_card_fingerprint(value)
        if isinstance(value, HandCard):
            return self._hand_card_fingerprint(value)
        if isinstance(value, (Unit, Amulet)):
            return self._board_entity_fingerprint(value)
        if isinstance(value, (list, tuple)):
            return tuple(self._fingerprint_value(item) for item in value)
        if isinstance(value, (set, frozenset)):
            return tuple(
                sorted(
                    (
                        self._fingerprint_value(item)
                        for item in value
                    ),
                    key=repr,
                )
            )
        if isinstance(value, dict):
            return tuple(
                (
                    self._fingerprint_value(key),
                    self._fingerprint_value(item),
                )
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            )
        return repr(value)

    def _step(self) -> None:
        self.state.resolution_steps += 1
        if self.state.resolution_steps > MAX_RESOLUTION_STEPS:
            diagnostics = self._loop_diagnostics()
            self._record_runtime_diagnostic(
                "resolution_step_limit",
                detail="resolution_steps",
            )
            raise ResolutionLoopError(
                f"Resolution step limit exceeded at turn {self.turn}, "
                f"player {self.current_player + 1}. "
                f"Recent events: "
                f"{[event['type'] for event in diagnostics['recent_events']]}. "
                f"Queued events: "
                f"{[event['type'] for event in diagnostics['event_queue']]}. "
                f"Effect stack: {diagnostics['effect_stack']}. "
                f"Death queue: {diagnostics['death_queue']}. "
                f"Emblem batches: {diagnostics['emblem_batches']}."
                ,
                diagnostics=diagnostics,
            )

    def apply_damage(
        self,
        source: Unit | CardDefinition | None,
        target: Unit | None,
        amount: int,
        damage_type: DamageType,
        controller: int,
        *,
        attacker: Unit | None = None,
        target_player_index: int | None = None,
    ) -> DamageResult:
        zero_combat_contact = (
            amount == 0
            and isinstance(target, Unit)
            and damage_type is DamageType.COMBAT
            and isinstance(attacker, Unit)
        )
        if amount <= 0 and not zero_combat_contact:
            return DamageResult(requested_amount=amount)

        if target is not None and isinstance(target, Unit):
            return self._apply_damage_to_unit(
                source, target, amount, damage_type, controller, attacker=attacker,
            )
        else:
            player_idx = target_player_index if target_player_index is not None else (1 - controller)
            player = self.players[player_idx]
            return self._apply_damage_to_leader(
                source, player, amount, damage_type, controller,
            )
            return DamageResult(requested_amount=amount)

    def _unit_is_on_owners_turn(self, target: Unit) -> bool:
        return self._entity_owner(target.entity_id) == self.state.active_player

    def _super_evolution_prevents_damage(
        self,
        target: Unit,
        damage_type: DamageType,
    ) -> bool:
        return self._super_evolution_protection_active(target)

    def _super_evolution_prevents_effect_destroy(self, target: Unit) -> bool:
        return self._super_evolution_protection_active(target)

    def _effect_destroy_immunity_active(self, target: Unit) -> bool:
        return (
            not target.printed_abilities_removed
            and (
                target.effect_destroy_immunity
                or self.rulebook.cannot_be_destroyed_by_effects(
                    target.definition.card_id
                )
            )
        )

    def _attempt_effect_destroy_unit(
        self,
        target: Unit,
        *,
        controller: int,
        source_entity_id: int | None,
        source_card_id: int | None,
        ability: str | None = None,
    ) -> bool:
        if self._effect_destroy_immunity_active(target):
            printed_immunity = (
                not target.printed_abilities_removed
                and self.rulebook.cannot_be_destroyed_by_effects(
                    target.definition.card_id
                )
            )
            metadata: dict[str, object] = {
                "source_card_id": source_card_id,
                "protected_card_id": target.definition.card_id,
                "printed_ability": printed_immunity,
                "granted_ability": (
                    not target.printed_abilities_removed
                    and target.effect_destroy_immunity
                ),
            }
            if ability is not None:
                metadata["ability"] = ability
            self._emit(GameEvent(
                EventType.EFFECT_DESTROY_PREVENTED,
                controller,
                source_id=source_entity_id,
                target_id=target.entity_id,
                metadata=metadata,
            ))
            self._log(
                controller,
                f"{target.definition.name} 的能力阻止了效果破坏",
            )
            return False
        if self._super_evolution_prevents_effect_destroy(target):
            self._log(
                controller,
                f"{target.definition.name} 的超进化保护阻止了效果破坏",
            )
            return False
        self._death_causes[target.entity_id] = DeathCause.EFFECT_DESTROY
        target.health = 0
        return True

    def _printed_incoming_damage_replacement(
        self,
        target: Unit,
    ) -> tuple[int, int] | None:
        if target.printed_abilities_removed:
            return None
        return self.rulebook.incoming_damage_replacement(
            target.definition.card_id
        )

    def _super_evolution_protection_active(self, target: Unit) -> bool:
        return (
            target.super_evolved
            and self._unit_is_on_owners_turn(target)
        )

    def _apply_damage_to_unit(
        self,
        source: Unit | CardDefinition | None,
        target: Unit,
        amount: int,
        damage_type: DamageType,
        controller: int,
        *,
        attacker: Unit | None = None,
    ) -> DamageResult:
        requested_amount = amount
        health_before = target.health
        prevented = 0
        barrier_consumed = False

        if (
            amount > 0
            and self._super_evolution_prevents_damage(target, damage_type)
        ):
            self._emit(GameEvent(
                EventType.DAMAGE_PREVENTED, controller,
                source_id=source.entity_id if hasattr(source, 'entity_id') else None,
                target_id=target.entity_id,
                amount=amount,
                metadata={
                    "super_evolved": True,
                    "damage_type": damage_type.value,
                },
            ))
            self._log(
                controller,
                f"{target.definition.name} 的超进化保护阻止了 {amount} 点伤害",
            )
            if (
                damage_type is DamageType.COMBAT
                and isinstance(attacker, Unit)
                and attacker.has_keyword("必杀")
            ):
                self._emit(GameEvent(
                    EventType.BANE_TRIGGERED,
                    controller,
                    source_id=attacker.entity_id,
                    target_id=target.entity_id,
                    metadata={"card_id": attacker.definition.card_id},
                ))
                self._attempt_effect_destroy_unit(
                    target,
                    controller=controller,
                    source_entity_id=attacker.entity_id,
                    source_card_id=attacker.definition.card_id,
                    ability="必杀",
                )
            return DamageResult(
                requested_amount=amount,
                prevented_amount=amount,
                actual_amount=0,
                target_health_before=health_before,
                target_health_after=health_before,
                barrier_consumed=False,
                lethal=False,
            )

        replacement = self._printed_incoming_damage_replacement(target)
        if replacement is not None:
            threshold, replacement_amount = replacement
            if amount >= threshold:
                prevented_by_replacement = amount - replacement_amount
                amount = replacement_amount
                prevented += prevented_by_replacement
                self._emit(GameEvent(
                    EventType.DAMAGE_PREVENTED,
                    controller,
                    source_id=(
                        source.entity_id
                        if hasattr(source, "entity_id")
                        else None
                    ),
                    target_id=target.entity_id,
                    amount=prevented_by_replacement,
                    metadata={
                        "incoming_damage_replacement": True,
                        "damage_type": damage_type.value,
                        "threshold": threshold,
                        "requested_amount": requested_amount,
                        "replacement_amount": replacement_amount,
                        "card_id": target.definition.card_id,
                    },
                ))
                self._log(
                    controller,
                    f"{target.definition.name} 将 {requested_amount} 点伤害"
                    f"变为 {replacement_amount} 点",
                )

        if amount > 0 and target.barrier_charges > 0:
            target.barrier_charges -= 1
            prevented += amount
            barrier_consumed = True
            self._emit(GameEvent(
                EventType.DAMAGE_PREVENTED, controller,
                source_id=source.entity_id if hasattr(source, 'entity_id') else None,
                target_id=target.entity_id,
                amount=amount,
                metadata={"barrier": True},
            ))
            self._emit(GameEvent(
                EventType.BARRIER_CONSUMED, controller,
                target_id=target.entity_id,
                metadata={"card_id": target.definition.card_id},
            ))

        actual = min(amount, health_before) if not barrier_consumed else 0
        target.health -= actual
        health_after = target.health
        lethal = health_after <= 0

        if actual > 0:
            self._emit(GameEvent(
                EventType.DAMAGE_APPLIED, controller,
                source_id=source.entity_id if hasattr(source, 'entity_id') else None,
                target_id=target.entity_id,
                amount=actual,
                metadata={"damage_type": damage_type.value},
            ))

        if damage_type == DamageType.COMBAT:
            if lethal:
                self._death_causes[target.entity_id] = DeathCause.COMBAT
        elif damage_type == DamageType.EFFECT:
            if lethal:
                self._death_causes.setdefault(target.entity_id, DeathCause.ZERO_HEALTH)

        if (
            damage_type is DamageType.COMBAT
            and isinstance(attacker, Unit)
            and attacker.has_keyword("必杀")
        ):
            self._emit(GameEvent(
                EventType.BANE_TRIGGERED, controller,
                source_id=attacker.entity_id,
                target_id=target.entity_id,
                metadata={"card_id": attacker.definition.card_id},
            ))
            if self._attempt_effect_destroy_unit(
                target,
                controller=controller,
                source_entity_id=attacker.entity_id,
                source_card_id=attacker.definition.card_id,
                ability="必杀",
            ):
                health_after = 0
                lethal = True
                self._log(
                    controller,
                    f"{attacker.definition.name} 的必杀破坏了 "
                    f"{target.definition.name}",
                )

        if actual > 0 and attacker is not None and isinstance(attacker, Unit):
            if attacker.has_keyword("吸血"):
                owner_idx = self._entity_owner(attacker.entity_id)
                heal_amount = min(actual, health_before)
                owner = self.players[owner_idx]
                before_heal = owner.health
                owner.health = min(
                    owner.health + heal_amount,
                    owner.max_health,
                )
                actual_heal = owner.health - before_heal
                self._log(controller, f"{attacker.definition.name} 的吸血回复了 {actual_heal} 点生命")
                self._emit(GameEvent(
                    EventType.DRAIN_HEALED, owner_idx,
                    source_id=attacker.entity_id,
                    amount=actual_heal,
                    metadata={"card_id": attacker.definition.card_id},
                ))
                if actual_heal > 0:
                    self._emit(GameEvent(
                        EventType.LEADER_HEALED,
                        owner_idx,
                        source_id=attacker.entity_id,
                        amount=actual_heal,
                        metadata={"card_id": attacker.definition.card_id},
                    ))

        if actual > 0 and target.health > 0:
            target_owner = self._entity_owner(target.entity_id)
            damage_source_card_id = (
                source.definition.card_id
                if isinstance(source, Unit)
                else source.card_id
                if isinstance(source, CardDefinition)
                else None
            )
            self._emit(
                GameEvent(
                    EventType.FOLLOWER_DAMAGED_SURVIVED,
                    target_owner,
                    source_id=target.entity_id,
                    amount=actual,
                    metadata={
                        "source": target,
                        "card_id": target.definition.card_id,
                        "damage_type": damage_type.value,
                        "damage_source_card_id": damage_source_card_id,
                        "damage_source_entity_id": (
                            source.entity_id
                            if isinstance(source, Unit)
                            else None
                        ),
                        "health_before": health_before,
                        "health_after": target.health,
                    },
                )
            )

        source_name = source.definition.name if hasattr(source, 'definition') else (source.name if source else "效果")
        self._log(
            controller,
            f"{source_name} 对 {target.definition.name} 造成 {actual} 点伤害"
            f"{'（被屏障阻止）' if barrier_consumed else ''}"
            f"（剩余生命 {target.health}）",
        )

        return DamageResult(
            requested_amount=requested_amount,
            prevented_amount=prevented,
            actual_amount=actual,
            target_health_before=health_before,
            target_health_after=health_after,
            barrier_consumed=barrier_consumed,
            lethal=lethal,
        )

    def _apply_damage_to_leader(
        self,
        source: Unit | CardDefinition | None,
        target_player: PlayerState,
        amount: int,
        damage_type: DamageType,
        controller: int,
    ) -> DamageResult:
        requested_amount = amount
        health_before = target_player.health
        active_modifiers = tuple(
            modifier
            for modifier in target_player.leader_damage_modifiers
            if self._leader_damage_modifier_active(modifier)
        )
        modifier_amount = sum(
            modifier.amount
            for modifier in active_modifiers
            if modifier.mode == LeaderDamageMode.ADDITIVE.value
        )
        modified_amount = max(0, amount + modifier_amount)
        replacement_modifier_ids = tuple(
            modifier.modifier_id
            for modifier in active_modifiers
            if modifier.mode == LeaderDamageMode.SET_ZERO_IF_POSITIVE.value
        )
        prevented = 0
        barrier_consumed = False
        if modified_amount > 0 and replacement_modifier_ids:
            prevented = modified_amount
            modified_amount = 0
            self._emit(GameEvent(
                EventType.DAMAGE_PREVENTED,
                controller,
                source_id=(
                    source.entity_id if hasattr(source, "entity_id") else None
                ),
                amount=prevented,
                metadata={
                    "target_player": self.players.index(target_player),
                    "leader_damage_replacement": True,
                    "damage_type": damage_type.value,
                    "requested_amount": requested_amount,
                    "modifier_ids": replacement_modifier_ids,
                },
            ))
        elif modified_amount > 0 and target_player.leader_barrier_charges > 0:
            target_player.leader_barrier_charges -= 1
            prevented = modified_amount
            modified_amount = 0
            barrier_consumed = True
            target_index = self.players.index(target_player)
            self._emit(GameEvent(
                EventType.DAMAGE_PREVENTED,
                controller,
                source_id=(
                    source.entity_id if hasattr(source, "entity_id") else None
                ),
                amount=prevented,
                metadata={
                    "target_player": target_index,
                    "barrier": True,
                    "damage_type": damage_type.value,
                },
            ))
            self._emit(GameEvent(
                EventType.BARRIER_CONSUMED,
                controller,
                metadata={
                    "target_player": target_index,
                    "leader_barrier": True,
                },
            ))
        actual = min(modified_amount, health_before)
        target_player.health -= actual

        self._emit(GameEvent(
            EventType.DAMAGE_APPLIED, controller,
            source_id=source.entity_id if hasattr(source, 'entity_id') else None,
            amount=actual,
            metadata={
                "target_player": self.players.index(target_player),
                "damage_type": damage_type.value,
                "base_amount": amount,
                "modifier_amount": modifier_amount,
                "replacement_modifier_ids": replacement_modifier_ids,
                "barrier_consumed": barrier_consumed,
            },
        ))

        if isinstance(source, Unit) and source.has_keyword("吸血"):
            owner_idx = self._entity_owner(source.entity_id)
            owner = self.players[owner_idx]
            before_heal = owner.health
            owner.health = min(
                owner.health + actual,
                owner.max_health,
            )
            actual_heal = owner.health - before_heal
            self._log(controller, f"{source.definition.name} 的吸血回复了 {actual_heal} 点生命")
            self._emit(GameEvent(
                EventType.DRAIN_HEALED, owner_idx,
                source_id=source.entity_id,
                amount=actual_heal,
                metadata={"card_id": source.definition.card_id},
            ))
            if actual_heal > 0:
                self._emit(GameEvent(
                    EventType.LEADER_HEALED,
                    owner_idx,
                    source_id=source.entity_id,
                    amount=actual_heal,
                    metadata={"card_id": source.definition.card_id},
                ))

        return DamageResult(
            requested_amount=requested_amount,
            prevented_amount=prevented,
            actual_amount=actual,
            target_health_before=health_before,
            target_health_after=target_player.health,
            barrier_consumed=barrier_consumed,
            lethal=target_player.health <= 0,
        )

    def _leader_damage_modifier_active(
        self, modifier: LeaderDamageModifier
    ) -> bool:
        if modifier.duration != ModifierDuration.WHILE_SOURCE_IN_PLAY.value:
            return True
        if (
            modifier.source_controller not in (0, 1)
            or modifier.source_entity_id is None
            or modifier.source_card_id is None
        ):
            return False
        return any(
            entity.entity_id == modifier.source_entity_id
            and entity.definition.card_id == modifier.source_card_id
            for entity in self.players[modifier.source_controller].board
        )

    def _send_to_graveyard(
        self,
        player_index: int,
        card: CardDefinition,
        cause: str,
        source_entity_id: int | None = None,
        *,
        derived: bool = False,
        origin: CardOrigin = CardOrigin.UNKNOWN,
        token: bool = False,
        source_origin: CardOrigin | None = None,
    ) -> GraveyardCard:
        player = self.players[player_index]
        entity_id = source_entity_id if source_entity_id is not None else self.state.allocate_entity_id()
        seq = player._next_graveyard_sequence
        player._next_graveyard_sequence += 1
        gc = GraveyardCard(
            definition=card,
            entity_id=entity_id,
            owner=player_index,
            entered_sequence=seq,
            entry_cause=cause,
            derived=derived,
            origin=origin,
            token=token,
            source_origin=source_origin,
        )
        player.graveyard.append(gc)
        before = player.shadows
        player.add_shadows(1)
        self._emit(
            GameEvent(
                EventType.GRAVEYARD_ENTERED,
                player_index,
                source_id=source_entity_id,
                amount=1,
                metadata={
                    "card_id": card.card_id,
                    "entity_id": entity_id,
                    "cause": cause,
                    "derived": derived,
                    "origin": origin.value,
                    "token": token,
                    "shadows_before": before,
                    "shadows_after": player.shadows,
                },
            )
        )
        self._emit(
            GameEvent(
                EventType.SHADOWS_CHANGED,
                player_index,
                amount=1,
                metadata={
                    "change": "gain",
                    "shadows_before": before,
                    "shadows_after": player.shadows,
                },
            )
        )
        return gc

    def _record_destroyed_follower(
        self, player_index: int, definition: CardDefinition, cause: DeathCause,
        *,
        derived: bool = False,
        token: bool = False,
        origin: CardOrigin = CardOrigin.DECK,
        source_origin: CardOrigin | None = None,
    ) -> None:
        self.state.destroyed_followers.append(
            DestroyedFollowerRecord(
                definition=definition,
                owner=player_index,
                death_sequence=self.state._next_death_sequence,
                cause=cause,
                derived=derived,
                token=token,
                origin=origin,
                source_origin=source_origin,
                destroyed_turn=self.turn,
            )
        )
        self.state._next_death_sequence += 1

    def _record_destroyed_amulet(
        self,
        player_index: int,
        definition: CardDefinition,
        cause: DeathCause,
        *,
        derived: bool = False,
        token: bool = False,
        origin: CardOrigin = CardOrigin.DECK,
        source_origin: CardOrigin | None = None,
        play_mode_id: str | None = None,
        summon_countdown: int | None = None,
    ) -> None:
        self.state.destroyed_amulets.append(
            DestroyedAmuletRecord(
                definition=definition,
                owner=player_index,
                death_sequence=self.state._next_death_sequence,
                cause=cause,
                derived=derived,
                token=token,
                origin=origin,
                source_origin=source_origin,
                destroyed_turn=self.turn,
                play_mode_id=play_mode_id,
                summon_countdown=summon_countdown,
            )
        )
        self.state._next_death_sequence += 1

    def _record_follower_entry(
        self,
        player_index: int,
        definition: CardDefinition,
        *,
        entry_cause: str,
    ) -> None:
        self.state.follower_entries.append(
            FollowerEntryRecord(
                definition=definition,
                owner=player_index,
                entry_sequence=self.state._next_follower_entry_sequence,
                entered_turn=self.turn,
                entry_cause=entry_cause,
            )
        )
        self.state._next_follower_entry_sequence += 1

    def _runtime_clause_id(
        self,
        operation: EffectOperation,
        frame: EffectFrame,
    ) -> str | None:
        if self.runtime_coverage is None:
            return None
        return self.runtime_coverage.resolve_clause_id(
            frame.source_card_id,
            operation,
            frame.operations,
        )

    def _runtime_target_candidate_count(
        self,
        operation: EffectOperation,
        frame: EffectFrame,
        condition_state: PartialConditionResult,
    ) -> int | None:
        if operation.target in {
            TargetKind.SELF,
            TargetKind.EMBLEM_SELF,
            TargetKind.EVENT_SOURCE,
            TargetKind.ATTACK_TARGET,
            TargetKind.OWN_LEADER,
            TargetKind.ENEMY_LEADER,
        }:
            return None
        if operation.target is TargetKind.PREVIOUS_TARGET:
            return len(frame._target_bindings.get(operation.target_key or "", ()))
        if operation.target in {
            TargetKind.OWN_HAND,
            TargetKind.RANDOM_OWN_HAND,
            TargetKind.RANDOM_ENEMY_HAND,
            TargetKind.ALL_OWN_HAND,
            TargetKind.ALL_ENEMY_HAND,
        }:
            return len(
                hand_candidates(
                    operation,
                    frame.controller,
                    self.players,
                    source_entity_id=frame.source_entity_id,
                )
            )
        if is_graveyard_target(operation.target):
            return len(
                graveyard_candidates(
                    operation,
                    frame.controller,
                    self.players,
                )
            )
        if operation.target is TargetKind.ALL_LEADERS:
            return len(
                leader_target_ids(
                    operation,
                    frame.controller,
                    self.players,
                )
            )
        if operation.target is TargetKind.ALL_ENEMY_UNITS_AND_LEADER:
            return 1 + sum(
                isinstance(entity, Unit)
                for entity in self.players[1 - frame.controller].board
            )
        if operation.target is TargetKind.ALL_OWN_EMBLEMS:
            return sum(
                emblem.countdown is not None
                and (
                    operation.emblem_id is None
                    or emblem.emblem_id == operation.emblem_id
                )
                for emblem in self.players[frame.controller].emblems
            )
        candidates = target_candidates(
            operation,
            frame.controller,
            self.players,
            source_entity_id=frame.source_entity_id,
        )
        if condition_state is PartialConditionResult.DEPENDS_ON_TARGET:
            candidates = [
                entity
                for entity in candidates
                if self._target_conditions_met(
                    operation.conditions,
                    entity,
                    frame.controller,
                    source_entity_id=frame.source_entity_id,
                    source_fusion_count=len(frame.fusion_materials),
                )
            ]
        if (
            operation.exclude_attack_target
            and frame.attack_target_entity_id is not None
        ):
            candidates = [
                entity
                for entity in candidates
                if entity.entity_id != frame.attack_target_entity_id
            ]
        leader_count = len(
            leader_choice_options(
                operation.target,
                frame.controller,
                self.players,
            )
        )
        return len(candidates) + leader_count

    def _record_runtime_target(
        self,
        operation: EffectOperation,
        frame: EffectFrame,
        clause_id: str | None,
        condition_state: PartialConditionResult,
    ) -> None:
        if self.runtime_coverage is None or clause_id is None:
            return
        candidate_count = self._runtime_target_candidate_count(
            operation,
            frame,
            condition_state,
        )
        self.runtime_coverage.record_target(
            clause_id,
            operation.target,
            candidate_count=candidate_count,
            random=is_random_target(operation.target),
            no_target=candidate_count == 0,
        )

    def _record_runtime_capacity(
        self,
        operation: EffectOperation,
        frame: EffectFrame,
        clause_id: str | None,
    ) -> None:
        if self.runtime_coverage is None or clause_id is None:
            return
        player = self.players[frame.controller]
        board_kinds = {
            EffectKind.SUMMON,
            EffectKind.SUMMON_COPY,
            EffectKind.SUMMON_EXACT_COPY,
            EffectKind.SUMMON_HAND_COPY,
            EffectKind.SUMMON_FROM_HAND,
            EffectKind.SUMMON_FROM_DECK,
            EffectKind.SUMMON_DESTROYED_AMULETS,
            EffectKind.REANIMATE,
            EffectKind.SUMMON_FROM_GRAVEYARD,
        }
        hand_kinds = {
            EffectKind.DRAW,
            EffectKind.DRAW_FILTERED,
            EffectKind.ADD_CARD,
            EffectKind.COPY_TO_HAND,
            EffectKind.COPY_DESTROYED_FOLLOWERS_TO_HAND,
            EffectKind.COPY_RANDOM_ENEMY_DECK_TO_HAND,
            EffectKind.COPY_LEFTMOST_HAND_TO_HAND,
            EffectKind.RETURN_TO_HAND,
            EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND,
        }
        if (
            operation.kind in board_kinds
            and len(player.board) >= self.config.max_board
        ):
            self.runtime_coverage.record_capacity_shortage(
                clause_id,
                "board",
            )
        if (
            operation.kind in hand_kinds
            and len(player.hand) >= self.config.max_hand
        ):
            self.runtime_coverage.record_capacity_shortage(
                clause_id,
                "hand",
            )
        if (
            operation.kind in {EffectKind.GAIN_EMBLEM, EffectKind.ADD_EMBLEM}
            and len(player.emblems) >= self.config.leader_area_limit
        ):
            self.runtime_coverage.record_capacity_shortage(
                clause_id,
                "leader_area",
            )

    def _continue_effects(
        self,
        *,
        stop_at_depth: int | None = None,
    ) -> None:
        while self.state.effect_stack and self.state.pending_choice is None:
            if (
                stop_at_depth is not None
                and len(self.state.effect_stack) <= stop_at_depth
            ):
                break
            if self.terminated:
                for active_frame in self.state.effect_stack:
                    active_frame.next_index = len(active_frame.operations)
            self._step()
            frame = self.state.effect_stack[-1]

            if frame.defer_stabilize:
                break
            if frame.next_index >= len(frame.operations):
                self.state.effect_stack.pop()
                if frame.defer_stabilize:
                    self._stabilize()
                if frame.move_source_to_graveyard:
                    spell_eid = frame._hand_source_entity_id
                    self._send_to_graveyard(
                        frame.controller, frame.source_card, "spell_resolved",
                        source_entity_id=spell_eid,
                        origin=(
                            frame._hand_source_origin
                            if frame._hand_source_origin is not None
                            else CardOrigin.UNKNOWN
                        ),
                        source_origin=frame._hand_source_origin_parent,
                    )
                    self._emit(
                        GameEvent(
                            EventType.SPELL_RESOLVED,
                            frame.controller,
                            metadata={"card_id": frame.source_card_id},
                        )
                    )
                if frame.emblem_batch_id is not None:
                    self._record_emblem_frame_activation(frame)
                    self._queue_next_emblem_trigger(frame.emblem_batch_id)
                if frame.listener_batch_id is not None:
                    self._queue_next_card_listener(frame.listener_batch_id)
                if frame.emblem_expiration_batch_id is not None:
                    self._complete_emblem_expiration(
                        frame.emblem_expiration_batch_id,
                        frame.expiring_emblem_owner,
                        frame.expiring_emblem_entity_id,
                    )
                continue

            operation = frame.operations[frame.next_index]
            clause_id = self._runtime_clause_id(operation, frame)
            if self.runtime_coverage is not None and clause_id is not None:
                self.runtime_coverage.record_clause(clause_id, "entered")
            if (
                operation.target_key
                and operation.target is not TargetKind.PREVIOUS_TARGET
                and _operation_produces_output_binding(operation)
                and operation.target_key not in frame._target_bindings
            ):
                self._bind_targets(
                    frame,
                    operation.target_key,
                    (),
                    operation,
                )
            if not self._bound_expression_inputs_available(operation, frame):
                frame.next_index += 1
                continue
            is_meta_effect = operation.kind in (
                EffectKind.CONDITIONAL,
                EffectKind.CHOOSE_ONE,
                EffectKind.OPTIONAL,
                EffectKind.TARGET_EXISTS,
            )
            if not is_meta_effect:
                condition_state = evaluate_conditions_without_target(
                    operation.conditions,
                    self._build_eval_context(frame, None),
                )
                if (
                    self.runtime_coverage is not None
                    and clause_id is not None
                    and operation.conditions
                ):
                    self.runtime_coverage.record_clause(
                        clause_id,
                        "condition_evaluated",
                    )
                    self.runtime_coverage.record_clause(
                        clause_id,
                        (
                            "condition_false"
                            if condition_state is PartialConditionResult.FALSE
                            else (
                                "condition_true"
                                if condition_state is PartialConditionResult.TRUE
                                else "condition_deferred"
                            )
                        ),
                    )
                if condition_state is PartialConditionResult.FALSE:
                    frame.next_index += 1
                    continue
            else:
                condition_state = PartialConditionResult.TRUE
            self._record_runtime_target(
                operation,
                frame,
                clause_id,
                condition_state,
            )

            if operation.kind is EffectKind.TARGET_EXISTS:
                self._checked_execute(operation, frame, None)
                frame.next_index += 1
                self._resolve_event_queue()
                self._stabilize()
                continue

            if operation.kind in {
                EffectKind.DISTRIBUTE_DAMAGE,
                EffectKind.RANDOM_CHOICE,
                EffectKind.RANDOM_DISTRIBUTE,
                EffectKind.SUMMON_DESTROYED_AMULETS,
                EffectKind.BANISH_DECK_FILTERED,
                EffectKind.REDRAW_HAND,
            }:
                self._checked_execute(operation, frame, None)
                frame.next_index += 1
                self._resolve_event_queue()
                self._stabilize()
                continue

            if operation.target is TargetKind.PREVIOUS_TARGET:
                if not operation.target_key:
                    raise IllegalCommand(
                        f"PREVIOUS_TARGET requires a bound target_key"
                    )
                target_ids = frame._target_bindings.get(operation.target_key, ())
                if not target_ids:
                    frame.next_index += 1
                    continue
                frame.defer_stabilize = True
                for bound_target_id in target_ids:
                    if not self._previous_target_still_legal(
                        frame,
                        operation.target_key,
                        bound_target_id,
                        operation,
                    ):
                        continue
                    self._checked_execute(operation, frame, bound_target_id)
                frame.defer_stabilize = False
                frame.next_index += 1
                self._resolve_event_queue()
                self._stabilize()
                continue

            if (
                is_graveyard_target(operation.target)
                and is_choice_target(operation.target)
                and frame.pending_target_id is None
                and not frame.pending_target_ids
            ):
                candidates = graveyard_candidates(operation, frame.controller, self.players)
                options = build_graveyard_choice_options(candidates)
                if not options:
                    frame.next_index += 1
                    continue
                choice_started = self._request_target_choice(
                    operation,
                    frame,
                    options,
                    choice_kind=ChoiceKind.GRAVEYARD,
                    prompt=f"为 {frame.source_name} 从墓地选择目标",
                )
                if not choice_started:
                    frame.next_index += 1
                    continue
                if self.state.pending_choice is not None:
                    return

            if is_graveyard_target(operation.target) and is_all_target(operation.target) and not frame.defer_stabilize:
                candidates = graveyard_candidates(operation, frame.controller, self.players)
                if not candidates:
                    frame.next_index += 1
                    continue
                frame.defer_stabilize = True
                for gc in candidates:
                    if gc not in self.players[frame.controller].graveyard:
                        continue
                    self._checked_execute(operation, frame, gc.entity_id)
                self._resolve_event_queue()
                self._stabilize()
                frame.defer_stabilize = False
                frame.next_index += 1
                continue

            if is_graveyard_target(operation.target) and is_random_target(operation.target) and frame.pending_target_id is None:
                candidates = graveyard_candidates(operation, frame.controller, self.players)
                chosen_gc = pick_random_graveyard(candidates, self.random) if candidates else None
                if chosen_gc is None:
                    frame.next_index += 1
                    continue
                target_id = chosen_gc.entity_id

            if (
                is_choice_target(operation.target)
                and not is_graveyard_target(operation.target)
                and frame.pending_target_id is None
                and not frame.pending_target_ids
            ):
                options = self._target_choice_options(operation, frame)

                if not options:
                    frame.next_index += 1
                    continue
                choice_kind = ChoiceKind.GENERIC
                if operation.target in (TargetKind.OWN_HAND,):
                    choice_kind = ChoiceKind.HAND
                elif operation.target not in (TargetKind.OWN_GRAVEYARD_CARD,):
                    choice_kind = ChoiceKind.BOARD
                choice_started = self._request_target_choice(
                    operation,
                    frame,
                    options,
                    choice_kind=choice_kind,
                    prompt=f"为 {frame.source_name} 选择目标",
                )
                if not choice_started:
                    frame.next_index += 1
                    continue
                if self.state.pending_choice is not None:
                    return

            if frame.pending_target_ids:
                target_ids = tuple(frame.pending_target_ids)
                frame.pending_target_ids.clear()
                pending_snapshots = frame._decision_meta.pop(
                    "pending_target_snapshots",
                    None,
                )
                output_binding = _operation_produces_output_binding(operation)
                if operation.target_key and not output_binding:
                    self._bind_targets(
                        frame,
                        operation.target_key,
                        target_ids,
                        operation,
                        snapshots=pending_snapshots,
                    )
                frame.defer_stabilize = True
                legal_target_ids: list[int] = []
                legal_snapshots: list[BoundTargetSnapshot] = []
                for selected_index, selected_target_id in enumerate(target_ids):
                    if not self._target_id_still_legal(
                        operation,
                        frame,
                        selected_target_id,
                    ):
                        self._log(
                            frame.controller,
                            f"已选目标 {selected_target_id} 已不再合法，跳过",
                        )
                        continue
                    legal_target_ids.append(selected_target_id)
                    if pending_snapshots is not None and not output_binding:
                        legal_snapshots.append(pending_snapshots[selected_index])
                    self._checked_execute(
                        operation,
                        frame,
                        selected_target_id,
                    )
                frame.defer_stabilize = False
                if (
                    operation.target_key
                    and not output_binding
                    and len(legal_target_ids) != len(target_ids)
                ):
                    self._bind_targets(
                        frame,
                        operation.target_key,
                        tuple(legal_target_ids),
                        operation,
                        snapshots=(
                            tuple(legal_snapshots)
                            if pending_snapshots is not None
                            else None
                        ),
                    )
                frame.next_index += 1
                self._resolve_event_queue()
                self._stabilize()
                continue

            if is_all_target(operation.target) and not frame.defer_stabilize:
                if (
                    operation.target
                    is TargetKind.ALL_ENEMY_UNITS_AND_LEADER
                ):
                    target_ids = tuple(
                        entity.entity_id
                        for entity in self.players[
                            1 - frame.controller
                        ].board
                        if isinstance(entity, Unit)
                    ) + (_leader_target_id(1 - frame.controller),)
                    frame.defer_stabilize = True
                    for simultaneous_target_id in target_ids:
                        if (
                            simultaneous_target_id >= 0
                            and not any(
                                entity.entity_id
                                == simultaneous_target_id
                                for entity in self.players[
                                    1 - frame.controller
                                ].board
                            )
                        ):
                            continue
                        self._checked_execute(
                            operation,
                            frame,
                            simultaneous_target_id,
                        )
                    frame.defer_stabilize = False
                    frame.next_index += 1
                    self._resolve_event_queue()
                    self._stabilize()
                    continue
                if operation.target is TargetKind.ALL_OWN_EMBLEMS:
                    target_ids = tuple(
                        emblem.entity_id
                        for emblem in self.players[frame.controller].emblems
                        if emblem.countdown is not None
                        and (
                            operation.emblem_id is None
                            or emblem.emblem_id == operation.emblem_id
                        )
                    )
                    if not target_ids:
                        frame.next_index += 1
                        continue
                    frame.defer_stabilize = True
                    for emblem_entity_id in target_ids:
                        if not any(
                            emblem.entity_id == emblem_entity_id
                            for emblem in self.players[frame.controller].emblems
                        ):
                            continue
                        self._checked_execute(
                            operation,
                            frame,
                            emblem_entity_id,
                        )
                    frame.defer_stabilize = False
                    frame.next_index += 1
                    self._resolve_event_queue()
                    self._stabilize()
                    continue
                if operation.target is TargetKind.ALL_LEADERS:
                    target_ids = leader_target_ids(
                        operation,
                        frame.controller,
                        self.players,
                    )
                    frame.defer_stabilize = True
                    for leader_target_id in target_ids:
                        self._checked_execute(
                            operation, frame, leader_target_id
                        )
                    frame.defer_stabilize = False
                    frame.next_index += 1
                    self._resolve_event_queue()
                    self._stabilize()
                    continue
                if operation.target in {
                    TargetKind.ALL_OWN_HAND,
                    TargetKind.ALL_ENEMY_HAND,
                }:
                    target_ids = [
                        card.entity_id
                        for card in hand_candidates(
                            operation,
                            frame.controller,
                            self.players,
                            source_entity_id=frame.source_entity_id,
                        )
                    ]
                    if not target_ids:
                        frame.next_index += 1
                        continue
                    frame.defer_stabilize = True
                    for hand_entity_id in target_ids:
                        self._checked_execute(
                            operation, frame, hand_entity_id
                        )
                    frame.defer_stabilize = False
                    frame.next_index += 1
                    continue
                candidates = target_candidates(
                    operation,
                    frame.controller,
                    self.players,
                    source_entity_id=frame.source_entity_id,
                )
                if (
                    condition_state is PartialConditionResult.DEPENDS_ON_TARGET
                ):
                    candidates = [
                        entity
                        for entity in candidates
                        if self._target_conditions_met(
                            operation.conditions,
                            entity,
                            frame.controller,
                            source_entity_id=frame.source_entity_id,
                            source_fusion_count=len(frame.fusion_materials),
                        )
                    ]
                if not candidates:
                    frame.next_index += 1
                    continue
                frame.defer_stabilize = True
                for entity in candidates:
                    if entity.entity_id not in [e.entity_id for board in [self.players[p].board for p in (0, 1)] for e in board]:
                        continue
                    self._checked_execute(operation, frame, entity.entity_id)
                self._resolve_event_queue()
                self._stabilize()
                frame.defer_stabilize = False
                frame.next_index += 1
                continue

            pending_snapshots = None
            if is_random_target(operation.target) and frame.pending_target_id is None:
                if operation.target in {
                    TargetKind.RANDOM_OWN_HAND,
                    TargetKind.RANDOM_ENEMY_HAND,
                }:
                    hand_cards = hand_candidates(
                        operation,
                        frame.controller,
                        self.players,
                        source_entity_id=frame.source_entity_id,
                    )
                    chosen_hand = (
                        self.random.choice(hand_cards) if hand_cards else None
                    )
                    if chosen_hand is None:
                        frame.next_index += 1
                        continue
                    target_id = chosen_hand.entity_id
                    self._checked_execute(operation, frame, target_id)
                    frame.next_index += 1
                    self._resolve_event_queue()
                    self._stabilize()
                    continue
                candidates = target_candidates(
                    operation,
                    frame.controller,
                    self.players,
                    source_entity_id=frame.source_entity_id,
                )
                if (
                    condition_state is PartialConditionResult.DEPENDS_ON_TARGET
                ):
                    candidates = [
                        entity
                        for entity in candidates
                        if self._target_conditions_met(
                            operation.conditions,
                            entity,
                            frame.controller,
                            source_entity_id=frame.source_entity_id,
                            source_fusion_count=len(frame.fusion_materials),
                        )
                    ]
                if (
                    operation.exclude_attack_target
                    and frame.attack_target_entity_id is not None
                ):
                    candidates = [
                        entity
                        for entity in candidates
                        if entity.entity_id != frame.attack_target_entity_id
                    ]
                if operation.target in {
                    TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER,
                    TargetKind.RANDOM_ANY_UNIT_OR_LEADER,
                }:
                    target_ids = [entity.entity_id for entity in candidates]
                    if (
                        operation.target
                        is TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER
                    ):
                        target_ids.append(
                            _leader_target_id(1 - frame.controller)
                        )
                    else:
                        target_ids.extend((
                            _leader_target_id(frame.controller),
                            _leader_target_id(1 - frame.controller),
                        ))
                else:
                    target_ids = [entity.entity_id for entity in candidates]
                target_count = self._resolved_target_count(operation, frame)
                if target_count <= 0 or not target_ids:
                    frame.next_index += 1
                    continue
                if not operation.allow_duplicate_targets:
                    target_count = min(target_count, len(target_ids))
                if target_count == 1:
                    target_id = self.random.choice(target_ids)
                elif operation.allow_duplicate_targets:
                    selected_target_ids = tuple(
                        self.random.choice(target_ids)
                        for _ in range(target_count)
                    )
                else:
                    selected_target_ids = tuple(
                        self.random.sample(target_ids, target_count)
                    )
                if target_count > 1:
                    if operation.target_key:
                        self._bind_targets(
                            frame,
                            operation.target_key,
                            selected_target_ids,
                            operation,
                        )
                    frame.defer_stabilize = True
                    for selected_target_id in selected_target_ids:
                        if (
                            not _is_leader_target_id(selected_target_id)
                            and not any(
                                entity.entity_id == selected_target_id
                                for player in self.players
                                for entity in player.board
                            )
                        ):
                            self._log(
                                frame.controller,
                                f"随机目标 {selected_target_id} 已离场，跳过",
                            )
                            continue
                        self._checked_execute(
                            operation,
                            frame,
                            selected_target_id,
                        )
                    frame.defer_stabilize = False
                    frame.next_index += 1
                    self._resolve_event_queue()
                    self._stabilize()
                    continue
            else:
                target_id = frame.pending_target_id
                frame.pending_target_id = None
                pending_snapshots = frame._decision_meta.pop(
                    "pending_target_snapshots",
                    None,
                )

            if (
                operation.target_key
                and not _operation_produces_output_binding(operation)
            ):
                if target_id is None:
                    raise IllegalCommand(
                        "target_key requires a resolved board or hand entity"
                    )
                if pending_snapshots is None:
                    try:
                        self._find_board_entity(target_id)
                    except IllegalCommand as exc:
                        raise IllegalCommand(
                            "target_key requires a resolved board or hand entity"
                        ) from exc
                self._bind_targets(
                    frame,
                    operation.target_key,
                    (target_id,),
                    operation,
                    snapshots=pending_snapshots,
                )
            self._checked_execute(operation, frame, target_id)
            frame.next_index += 1
            # A selected effect may make a board entity state-based dead.
            # Remove it before a suspended event (for example, the following
            # Super-Evolve event) is allowed to generate another target choice.
            self._stabilize()
            self._resolve_event_queue()
            self._stabilize()

    def _try_spellboost_hand(self) -> None:
        if self.state.pending_choice is not None or self.terminated:
            return
        if self._spellboost_pending is None:
            return
        amount = self._spellboost_pending
        player_index = self._pending_spellboost_player
        source_card_id = self._pending_spellboost_source_card_id
        source_entity_id = self._pending_spellboost_source_entity_id
        self._spellboost_pending = None

        player = self.players[player_index]
        for hand_card in player.hand:
            if isinstance(hand_card, HandCard):
                hand_card.apply_spellboost(amount)
                self._emit(
                    GameEvent(
                        EventType.SPELLBOOSTED,
                        player_index,
                        source_id=hand_card.entity_id,
                        amount=amount,
                        metadata={
                            "card_id": hand_card.card_id,
                            "spellboost_count": hand_card.spellboost_count,
                            "source_card_id": source_card_id,
                            "source_entity_id": source_entity_id,
                        },
                    )
                )

    def _is_card_playable(
        self, card: CardDefinition | HandCard, player: PlayerState
    ) -> bool:
        cost = card.current_cost if isinstance(card, HandCard) else card.cost
        if cost > player.mana:
            return False
        if card.card_type == "随从":
            return len(player.board) < self.config.max_board
        if card.card_type not in {"法术", "护符"}:
            return False
        if card.card_type == "护符" and len(player.board) >= self.config.max_board:
            return False
        operations = self.rulebook.operations_for(card.card_id, Trigger.PLAY)
        has_union_burst = bool(
            self.rulebook.union_bursts_for(card.card_id)
        )
        if card.card_type == "护符" and (
            operations
            or has_union_burst
            or self.rulebook.countdown_for(card.card_id) is not None
            or self.rulebook.activation_for(card.card_id) is not None
        ):
            pass
        elif not operations and not has_union_burst:
            return False

        if not operations:
            return True

        source_entity_id = card.entity_id if isinstance(card, HandCard) else None
        if any(
            op.requires_target
            and not self._has_candidates(
                op,
                source_entity_id=source_entity_id,
            )
            for op in operations
        ):
            return False

        all_require_target = all(
            self._operation_consumes_target(op)
            for op in operations
        )
        if all_require_target and all(
            not self._has_candidates(
                op,
                source_entity_id=source_entity_id,
            )
            for op in operations
        ):
            return False
        return True

    def _has_candidates(
        self,
        operation: EffectOperation,
        *,
        source_entity_id: int | None = None,
    ) -> bool:
        return self._has_candidates_for(
            operation,
            self.current_player,
            source_entity_id=source_entity_id,
        )

    def _has_candidates_for(
        self,
        operation: EffectOperation,
        controller: int,
        *,
        source_entity_id: int | None = None,
        source_fusion_count: int = 0,
    ) -> bool:
        if (
            is_choice_target(operation.target)
            and self._requested_target_count_for(
                operation,
                controller,
                source_entity_id=source_entity_id,
            ) <= 0
        ):
            return False
        condition_state = evaluate_conditions_without_target(
            operation.conditions,
            self._eval_context(
                controller,
                source_entity_id=source_entity_id,
                source_fusion_count=source_fusion_count,
            ),
        )
        if condition_state is PartialConditionResult.FALSE:
            return True
        if operation.target in {
            TargetKind.OWN_HAND,
            TargetKind.RANDOM_OWN_HAND,
            TargetKind.RANDOM_ENEMY_HAND,
        }:
            candidates = hand_candidates(
                operation,
                controller,
                self.players,
                source_entity_id=source_entity_id,
            )
            return self._candidate_count_is_sufficient(
                operation,
                controller,
                len(candidates),
                source_entity_id=source_entity_id,
            )
        if operation.target in {
            TargetKind.ALL_OWN_HAND,
            TargetKind.ALL_ENEMY_HAND,
        }:
            return not operation.requires_target or bool(
                hand_candidates(
                    operation,
                    controller,
                    self.players,
                    source_entity_id=source_entity_id,
                )
            )
        if operation.target is TargetKind.ALL_LEADERS:
            return bool(leader_target_ids(operation, controller, self.players))
        if operation.target is TargetKind.ALL_ENEMY_UNITS_AND_LEADER:
            return True
        if operation.target in {
            TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER,
            TargetKind.RANDOM_ANY_UNIT_OR_LEADER,
        }:
            return True
        if is_graveyard_target(operation.target):
            candidates = graveyard_candidates(operation, controller, self.players)
            return self._candidate_count_is_sufficient(
                operation,
                controller,
                len(candidates),
                source_entity_id=source_entity_id,
            )
        candidates = target_candidates(
            operation,
            controller,
            self.players,
            source_entity_id=source_entity_id,
        )
        if is_choice_target(operation.target):
            candidates = [
                e for e in candidates
                if not (
                    isinstance(e, Unit)
                    and e.ambush_active
                    and self._entity_owner(e.entity_id) != controller
                )
            ]
        if condition_state is PartialConditionResult.DEPENDS_ON_TARGET:
            candidates = [
                entity
                for entity in candidates
                if self._target_conditions_met(
                    operation.conditions,
                    entity,
                    controller,
                    source_entity_id=source_entity_id,
                    source_fusion_count=source_fusion_count,
                )
            ]
        leader_options = (
            leader_choice_options(
                operation.target,
                controller,
                self.players,
            )
            if condition_state is not PartialConditionResult.DEPENDS_ON_TARGET
            else []
        )
        return self._candidate_count_is_sufficient(
            operation,
            controller,
            len(candidates) + len(leader_options),
            source_entity_id=source_entity_id,
        )

    def _candidate_count_is_sufficient(
        self,
        operation: EffectOperation,
        controller: int,
        candidate_count: int,
        *,
        source_entity_id: int | None = None,
    ) -> bool:
        if candidate_count <= 0:
            return False
        if not operation.requires_full_target_count:
            return True
        return candidate_count >= self._requested_target_count_for(
            operation,
            controller,
            source_entity_id=source_entity_id,
        )

    def _requested_target_count_for(
        self,
        operation: EffectOperation,
        controller: int,
        *,
        source_entity_id: int | None = None,
    ) -> int:
        if operation.target_count_expr is None:
            return operation.target_count
        return max(
            0,
            evaluate_expression(
                operation.target_count_expr,
                self._eval_context(
                    controller,
                    source_entity_id=source_entity_id,
                ),
            ),
        )

    @staticmethod
    def _requires_choice(operation: EffectOperation) -> bool:
        return is_choice_target(operation.target)

    @staticmethod
    def _operation_consumes_target(operation: EffectOperation) -> bool:
        if operation.kind in {
            EffectKind.CONDITIONAL,
            EffectKind.CHOOSE_ONE,
            EffectKind.OPTIONAL,
            EffectKind.TARGET_EXISTS,
        }:
            return False
        # Selected targets require at least one legal candidate unless the
        # operation is paired with another independently-resolving effect.
        # Random and all-target effects instead resolve as safe no-ops when
        # their candidate set is empty; `requires_target` remains the explicit
        # schema switch for cards whose text prohibits that path.
        return is_choice_target(operation.target)

    def _target_exists_for(
        self,
        operation: EffectOperation,
        controller: int,
        *,
        source_entity_id: int | None = None,
        source_fusion_count: int = 0,
    ) -> bool:
        condition_state = evaluate_conditions_without_target(
            operation.conditions,
            self._eval_context(
                controller,
                source_entity_id=source_entity_id,
                source_fusion_count=source_fusion_count,
            ),
        )
        if condition_state is PartialConditionResult.FALSE:
            return False
        if operation.target in {
            TargetKind.OWN_HAND,
            TargetKind.RANDOM_OWN_HAND,
            TargetKind.RANDOM_ENEMY_HAND,
            TargetKind.ALL_OWN_HAND,
            TargetKind.ALL_ENEMY_HAND,
        }:
            return bool(
                hand_candidates(
                    operation,
                    controller,
                    self.players,
                    source_entity_id=source_entity_id,
                )
            )
        if operation.target is TargetKind.ALL_LEADERS:
            return bool(leader_target_ids(operation, controller, self.players))
        if operation.target is TargetKind.ALL_ENEMY_UNITS_AND_LEADER:
            return True
        if is_graveyard_target(operation.target):
            return bool(graveyard_candidates(operation, controller, self.players))
        candidates = target_candidates(
            operation,
            controller,
            self.players,
            source_entity_id=source_entity_id,
        )
        if is_choice_target(operation.target):
            candidates = [
                e for e in candidates
                if not (
                    isinstance(e, Unit)
                    and e.ambush_active
                    and self._entity_owner(e.entity_id) != controller
                )
            ]
        if condition_state is PartialConditionResult.DEPENDS_ON_TARGET:
            candidates = [
                entity
                for entity in candidates
                if self._target_conditions_met(
                    operation.conditions,
                    entity,
                    controller,
                    source_entity_id=source_entity_id,
                    source_fusion_count=source_fusion_count,
                )
            ]
        return bool(candidates) or (
            condition_state is not PartialConditionResult.DEPENDS_ON_TARGET
            and bool(
                leader_choice_options(
                    operation.target,
                    controller,
                    self.players,
                )
            )
        )

    def _target_options(
        self,
        operation: EffectOperation,
        controller: int,
        *,
        source_entity_id: int | None = None,
    ) -> list[ChoiceOption]:
        if operation.target == TargetKind.OWN_HAND:
            return hand_choice_options(
                hand_candidates(
                    operation,
                    controller,
                    self.players,
                    source_entity_id=source_entity_id,
                )
            )
        if is_graveyard_target(operation.target):
            gc = graveyard_candidates(operation, controller, self.players)
            return build_graveyard_choice_options(gc)
        candidates = target_candidates(
            operation,
            controller,
            self.players,
            source_entity_id=source_entity_id,
        )
        candidates = [e for e in candidates if not (isinstance(e, Unit) and e.ambush_active and self._entity_owner(e.entity_id) != controller)]
        options = build_choice_options(candidates)
        options.extend(
            leader_choice_options(operation.target, controller, self.players)
        )
        return options

    def _target_choice_options(
        self, operation: EffectOperation, frame: EffectFrame
    ) -> list[ChoiceOption]:
        if operation.target == TargetKind.OWN_HAND:
            return hand_choice_options(
                hand_candidates(
                    operation,
                    frame.controller,
                    self.players,
                    source_entity_id=frame.source_entity_id,
                )
            )
        if is_graveyard_target(operation.target):
            gc = graveyard_candidates(operation, frame.controller, self.players)
            return build_graveyard_choice_options(gc)
        options = self._target_options(
            operation,
            frame.controller,
            source_entity_id=frame.source_entity_id,
        )
        if operation.conditions:
            candidates = target_candidates(
                operation,
                frame.controller,
                self.players,
                source_entity_id=frame.source_entity_id,
            )
            candidates = [
                e for e in candidates
                if not (
                    isinstance(e, Unit)
                    and e.ambush_active
                    and self._entity_owner(e.entity_id) != frame.controller
                )
            ]
            candidates = [
                e for e in candidates
                if self._target_conditions_met(
                    operation.conditions,
                    e,
                    frame.controller,
                    source_entity_id=frame.source_entity_id,
                    source_fusion_count=len(frame.fusion_materials),
                )
            ]
            options = build_choice_options(candidates)
            condition_state_for_choice = evaluate_conditions_without_target(
                operation.conditions,
                self._build_eval_context(frame, None),
            )
            if condition_state_for_choice is not PartialConditionResult.DEPENDS_ON_TARGET:
                options.extend(
                    leader_choice_options(
                        operation.target,
                        frame.controller,
                        self.players,
                    )
                )
        return options

    def _resolved_target_count(
        self,
        operation: EffectOperation,
        frame: EffectFrame,
    ) -> int:
        return self._requested_target_count_for(
            operation,
            frame.controller,
            source_entity_id=frame.source_entity_id,
        )

    def _effective_target_count(
        self,
        operation: EffectOperation,
        frame: EffectFrame,
        options: list[ChoiceOption],
    ) -> int:
        requested = self._resolved_target_count(operation, frame)
        if requested <= 0 or not options:
            return 0
        if operation.allow_duplicate_targets:
            return requested
        if operation.requires_full_target_count and len(options) < requested:
            return 0
        return min(requested, len(options))

    @staticmethod
    def _choice_option_target_id(option: ChoiceOption) -> int | None:
        if option.leader_player_index is not None:
            return _leader_target_id(option.leader_player_index)
        return option.entity_id

    def _target_id_still_legal(
        self,
        operation: EffectOperation,
        frame: EffectFrame,
        target_id: int,
    ) -> bool:
        return target_id in {
            self._choice_option_target_id(option)
            for option in self._target_choice_options(operation, frame)
        }

    def _set_auto_selected_targets(
        self,
        operation: EffectOperation,
        frame: EffectFrame,
        options: list[ChoiceOption],
        target_count: int,
    ) -> None:
        if target_count == 1:
            selected = (self.random.choice(options),)
        elif operation.allow_duplicate_targets:
            selected = tuple(
                self.random.choice(options)
                for _ in range(target_count)
            )
        else:
            selected = tuple(self.random.sample(options, target_count))
        self._log(
            frame.controller,
            "自动选择目标：" + "、".join(option.label for option in selected),
        )
        target_ids = [
            self._choice_option_target_id(option)
            for option in selected
        ]
        if any(target_id is None for target_id in target_ids):
            raise IllegalCommand("Selected target option has no target identity")
        if (
            operation.target_key
            and not _operation_produces_output_binding(operation)
        ):
            if operation.target is TargetKind.OWN_HAND:
                cards_by_id = {
                    card.entity_id: card
                    for card in self._hand_cards(frame.controller)
                }
                snapshots = tuple(
                    self._bound_hand_snapshot(
                        frame.controller,
                        cards_by_id[target_id],
                    )
                    for target_id in target_ids
                )
            else:
                snapshots = tuple(
                    self._bound_target_snapshot(target_id)
                    for target_id in target_ids
                )
            frame._decision_meta["pending_target_snapshots"] = snapshots
        if target_count == 1:
            frame.pending_target_id = target_ids[0]
        else:
            frame.pending_target_ids = list(target_ids)

    def _request_target_choice(
        self,
        operation: EffectOperation,
        frame: EffectFrame,
        options: list[ChoiceOption],
        *,
        choice_kind: ChoiceKind,
        prompt: str,
    ) -> bool:
        target_count = self._effective_target_count(operation, frame, options)
        if target_count <= 0:
            return False
        if frame.auto_resolve_choices:
            self._set_auto_selected_targets(
                operation,
                frame,
                options,
                target_count,
            )
            return True
        self.state.pending_choice = ChoiceRequest(
            player_index=frame.controller,
            prompt=prompt,
            options=tuple(options),
            continuation_id=f"{frame.source_card_id}:{frame.next_index}",
            choice_kind=choice_kind,
            request_id=self._allocate_choice_request_id(),
            target_count=target_count,
            allow_duplicate_targets=operation.allow_duplicate_targets,
        )
        self.state.phase = Phase.AWAITING_CHOICE
        self._log(
            frame.controller,
            f"{frame.source_name} 等待选择 {target_count} 个目标："
            + "、".join(option.label for option in options),
        )
        return True

    def _choice_option_still_legal(
        self,
        frame: EffectFrame,
        request: ChoiceRequest,
        option: ChoiceOption,
    ) -> bool:
        if frame.next_index >= len(frame.operations):
            return False
        if option.leader_player_index is not None:
            return option.option_id in {
                current.option_id
                for current in self._target_choice_options(
                    frame.operations[frame.next_index],
                    frame,
                )
            }
        if option.entity_id is None:
            return True
        operation = frame.operations[frame.next_index]
        if request.choice_kind is ChoiceKind.GRAVEYARD:
            legal_options = self._target_choice_options(operation, frame)
        elif option.option_id.startswith("hand:"):
            legal_options = self._target_choice_options(operation, frame)
        elif option.option_id.startswith("entity:"):
            legal_options = self._target_choice_options(operation, frame)
        else:
            return True
        return option.option_id in {
            current.option_id for current in legal_options
        }

    def _previous_target_still_legal(
        self,
        frame: EffectFrame,
        target_key: str,
        target_id: int,
        consuming_operation: EffectOperation,
    ) -> bool:
        binding_operation = frame._target_binding_operations.get(target_key)
        snapshots = frame._target_binding_snapshots.get(target_key, ())
        snapshot = next(
            (
                candidate
                for candidate in snapshots
                if candidate.entity_id == target_id
            ),
            None,
        )
        if binding_operation is not None and binding_operation.kind in {
            EffectKind.DRAW,
            EffectKind.DRAW_FILTERED,
            EffectKind.ADD_CARD,
        }:
            if snapshot is None or snapshot.zone != "hand":
                return False
            return any(
                card.entity_id == target_id
                for card in self._hand_cards(snapshot.controller)
            )
        if (
            snapshot is not None
            and snapshot.zone == "hand"
            and consuming_operation.kind
            in {
                EffectKind.ADD_KEYWORD,
                EffectKind.BUFF_HAND_CARD,
                EffectKind.CHANGE_COST,
                EffectKind.GRANT_LAST_WORDS,
                EffectKind.GRANT_EFFECT_DESTROY_IMMUNITY,
                EffectKind.SUMMON_FROM_HAND,
            }
        ):
            return any(
                card.entity_id == target_id
                for card in self._hand_cards(snapshot.controller)
            )
        if consuming_operation.kind in {
            EffectKind.BANISH_SAME_NAME,
            EffectKind.COPY_TO_HAND,
            EffectKind.SUMMON_COPY,
        }:
            return snapshot is not None
        try:
            self._find_board_entity(target_id)
        except IllegalCommand:
            return False
        if binding_operation is None:
            return True
        if binding_operation.kind in {
            EffectKind.SUMMON,
            EffectKind.SUMMON_EXACT_COPY,
            EffectKind.SUMMON_HAND_COPY,
            EffectKind.SUMMON_FROM_DECK,
            EffectKind.SUMMON_DESTROYED_AMULETS,
            EffectKind.REANIMATE,
            EffectKind.EVOLVE_UNIT,
            EffectKind.SUPER_EVOLVE_UNIT,
        }:
            return True
        return f"entity:{target_id}" in {
            option.option_id
            for option in self._target_choice_options(binding_operation, frame)
        }

    def _bind_targets(
        self,
        frame: EffectFrame,
        target_key: str,
        target_ids: tuple[int, ...],
        operation: EffectOperation,
        *,
        snapshots: tuple[BoundTargetSnapshot, ...] | None = None,
    ) -> None:
        if snapshots is None:
            snapshots = tuple(
                self._bound_target_snapshot(target_id)
                for target_id in target_ids
            )
        if tuple(snapshot.entity_id for snapshot in snapshots) != target_ids:
            raise IllegalCommand("Bound target snapshots do not match target IDs")
        frame._target_bindings[target_key] = target_ids
        frame._target_binding_operations[target_key] = operation
        frame._target_binding_snapshots[target_key] = snapshots

    def _bound_target_snapshot(self, target_id: int) -> BoundTargetSnapshot:
        entity = self._find_board_entity(target_id)
        return BoundTargetSnapshot(
            entity_id=target_id,
            controller=self._entity_owner(target_id),
            zone="board",
            card_id=entity.definition.card_id,
            card_type=entity.definition.card_type,
            card_name=entity.definition.name,
            cost=entity.definition.cost,
            definition=entity.definition,
            attack=entity.attack if isinstance(entity, Unit) else None,
        )

    def _bound_choice_snapshot(
        self,
        frame: EffectFrame,
        request: ChoiceRequest,
        target_id: int,
    ) -> BoundTargetSnapshot:
        if request.choice_kind is ChoiceKind.HAND:
            hand_card = next(
                (
                    card
                    for card in self._hand_cards(frame.controller)
                    if card.entity_id == target_id
                ),
                None,
            )
            if hand_card is None:
                raise IllegalCommand("Selected hand target is no longer present")
            return self._bound_hand_snapshot(frame.controller, hand_card)
        if request.choice_kind is ChoiceKind.GRAVEYARD:
            graveyard_card = next(
                (
                    card
                    for card in self.players[frame.controller].graveyard
                    if card.entity_id == target_id
                ),
                None,
            )
            if graveyard_card is None:
                raise IllegalCommand("Selected graveyard target is no longer present")
            return self._bound_graveyard_snapshot(graveyard_card)
        return self._bound_target_snapshot(target_id)

    @staticmethod
    def _bound_hand_snapshot(
        controller: int,
        card: HandCard,
    ) -> BoundTargetSnapshot:
        return BoundTargetSnapshot(
            entity_id=card.entity_id,
            controller=controller,
            zone="hand",
            card_id=card.card_id,
            card_type=card.card_type,
            card_name=card.name,
            cost=card.current_cost,
            definition=card.definition,
            attack=card.attack,
        )

    @staticmethod
    def _bound_graveyard_snapshot(
        card: GraveyardCard,
        *,
        cost: int | None = None,
    ) -> BoundTargetSnapshot:
        return BoundTargetSnapshot(
            entity_id=card.entity_id,
            controller=card.owner,
            zone="graveyard",
            card_id=card.definition.card_id,
            card_type=card.definition.card_type,
            card_name=card.definition.name,
            cost=card.definition.cost if cost is None else cost,
            definition=card.definition,
            attack=card.definition.attack,
        )

    def _stale_choice_reason(
        self,
        frame: EffectFrame,
        request: ChoiceRequest,
        option: ChoiceOption,
    ) -> str:
        if option.entity_id is None:
            return "已不再是合法目标"
        if request.choice_kind is ChoiceKind.GRAVEYARD:
            in_graveyard = any(
                gc.entity_id == option.entity_id
                for gc in self.players[frame.controller].graveyard
            )
            return "已不在墓地" if not in_graveyard else "已不再是合法目标"
        if option.option_id.startswith("hand:"):
            in_hand = any(
                hand_card.entity_id == option.entity_id
                for hand_card in self._hand_cards(frame.controller)
            )
            return "已离手" if not in_hand else "已不再是合法目标"
        if option.option_id.startswith("entity:"):
            in_play = any(
                entity.entity_id == option.entity_id
                for player in self.players
                for entity in player.board
            )
            return "已离场" if not in_play else "已不再是合法目标"
        return "已不再是合法目标"

    def _tick_countdowns(self, player_index: int) -> None:
        amulets = [
            entity
            for entity in tuple(self.players[player_index].board)
            if isinstance(entity, Amulet) and entity.countdown is not None
        ]
        for amulet in amulets:
            amulet.countdown -= 1
            self._log(
                player_index,
                f"{amulet.definition.name} 倒数减为 {amulet.countdown}",
            )
            if amulet.countdown <= 0:
                self._destroy_amulet(amulet, player_index=player_index)

    def _destroy_amulet(
        self, amulet: Amulet, *, player_index: int | None = None
    ) -> None:
        owner = (
            self._entity_owner(amulet.entity_id)
            if player_index is None
            else player_index
        )
        player = self.players[owner]
        if amulet not in player.board:
            return
        self._log(owner, f"护符 {amulet.definition.name} 倒数归零")
        amulet.pending_destroy = True
        self._stabilize()

    def _evolve(self, command: Evolve) -> None:
        if self._suspended_action_state is not None and self._suspended_action == "evolve":
            self._suspended_action = None
            self._suspended_action_state = None
            return
        self._evolve_unit(command.unit_id, super_evolve=False)

    def _super_evolve(self, command: SuperEvolve) -> None:
        if self._suspended_action_state is not None and self._suspended_action == "super_evolve":
            self._suspended_action = None
            self._suspended_action_state = None
            return
        self._evolve_unit(command.unit_id, super_evolve=True)

    def _evolve_unit(self, unit_id: int, *, super_evolve: bool) -> None:
        player = self.players[self.current_player]
        unit = self._find_unit(player.board, unit_id)
        if player.evolved_this_turn or unit.evolved:
            raise IllegalCommand("Evolution is not available")
        if super_evolve:
            if player.super_evolution_points <= 0:
                raise IllegalCommand("No super evolution points")
            unlock_turn = self._super_evolution_unlock_turn(self.current_player)
            if player.turns_started < unlock_turn:
                raise IllegalCommand("Super evolution is not unlocked")
            if player.super_evolved_this_turn:
                raise IllegalCommand("Super evolution is already used this turn")
        else:
            if player.evolution_points <= 0:
                raise IllegalCommand("No evolution points")
            if player.turns_started < self._evolution_unlock_turn(
                self.current_player
            ):
                raise IllegalCommand("Evolution is not unlocked")
        if super_evolve:
            player.super_evolution_points -= 1
            player.super_evolved_this_turn = True
        else:
            player.evolution_points -= 1
        player.evolved_this_turn = True
        self._apply_evolution_state(
            unit,
            self.current_player,
            super_evolve=super_evolve,
            cause="sep" if super_evolve else "ep",
            trigger_abilities=True,
        )
        action_name = "超进化" if super_evolve else "进化"
        resource_text = (
            f"剩余超进化次数 {player.super_evolution_points}"
            if super_evolve
            else f"剩余进化点 {player.evolution_points}"
        )
        self._log(
            self.current_player,
            f"{action_name} {unit.definition.name}，变为 {unit.attack}/{unit.health}，"
            f"{resource_text}",
        )
        self._resolve_event_queue()
        if self.state.pending_choice is not None:
            self._suspended_action = "super_evolve" if super_evolve else "evolve"
            self._suspended_action_state = {"unit_id": unit.entity_id}
            return

    def _apply_evolution_state(
        self,
        unit: Unit,
        owner: int,
        *,
        super_evolve: bool,
        cause: str,
        trigger_abilities: bool,
    ) -> bool:
        if unit.evolved:
            return False
        could_attack_leader = unit.can_attack_leader
        stat_bonus = 3 if super_evolve else 2
        unit.evolved = True
        unit.super_evolved = super_evolve
        unit.super_evolved_turn = self.turn if super_evolve else None
        previous_max_health = unit.max_health
        unit.base_attack += stat_bonus
        unit.base_health += stat_bonus
        unit._recompute_attack()
        unit._recompute_max()
        unit.health += unit.max_health - previous_max_health
        if unit.attacks_remaining > 0:
            unit.can_attack = True
            unit.rush_only = not could_attack_leader

        player = self.players[owner]
        player.followers_evolved_this_match += 1
        for hand_card in self._hand_cards(owner):
            hand_card.evolutions_while_in_hand += 1

        metadata = {
            "source": unit,
            "cause": cause,
            "trigger_abilities": trigger_abilities,
            "evolutions_this_match": player.followers_evolved_this_match,
            "super_evolution": super_evolve,
        }
        if super_evolve and trigger_abilities:
            self._emit(
                GameEvent(
                    EventType.FOLLOWER_EVOLVED,
                    owner,
                    source_id=unit.entity_id,
                    metadata={**metadata, "counts_as_evolution": True},
                )
            )
            self._emit(
                GameEvent(
                    EventType.FOLLOWER_SUPER_EVOLVED,
                    owner,
                    source_id=unit.entity_id,
                    metadata={**metadata, "counts_as_evolution": False},
                )
            )
        else:
            self._emit(
                GameEvent(
                    (
                        EventType.FOLLOWER_SUPER_EVOLVED
                        if super_evolve
                        else EventType.FOLLOWER_EVOLVED
                    ),
                    owner,
                    source_id=unit.entity_id,
                    metadata={**metadata, "counts_as_evolution": True},
                )
            )
        return True

    def _attack(self, command: Attack) -> None:
        if self._suspended_action_state is not None and self._suspended_action == "attack":
            return self._resume_attack(self._suspended_action_state)

        player = self.players[self.current_player]
        opponent = self.players[1 - self.current_player]
        attacker = self._find_unit(player.board, command.attacker_id)
        if not attacker.can_attack or attacker.attacks_remaining <= 0:
            raise IllegalCommand("Attacker cannot attack")
        guards = [
            unit
            for unit in opponent.board
            if (
                isinstance(unit, Unit)
                and unit.has_guard
                and self._is_follower_attack_target(unit)
            )
        ]
        if self._unit_ignores_ward(attacker):
            guards = []
        if command.target_id is None:
            if guards or not attacker.can_attack_leader:
                raise IllegalCommand("Leader is not a legal target")
            target = None
        else:
            if not attacker.can_attack_units:
                raise IllegalCommand("Attacker cannot attack units")
            target = self._find_unit(opponent.board, command.target_id)
            if target.ambush_active:
                raise IllegalCommand("Cannot attack an ambush follower")
            if target.has_intimidate:
                raise IllegalCommand("Cannot attack an intimidate follower")
            if guards and target not in guards:
                raise IllegalCommand("A guard follower must be attacked")

        self._active_super_evolution_attack = (
            SuperEvolutionAttackContext(
                controller=self.current_player,
                attacker_id=attacker.entity_id,
                target_id=target.entity_id,
                attacker_card_id=attacker.definition.card_id,
                attacker_name=attacker.definition.name,
            )
            if target is not None and attacker.super_evolved
            else None
        )

        player.follower_attacks_this_turn += 1

        self._emit(
            GameEvent(
                EventType.ATTACK_DECLARED,
                self.current_player,
                source_id=attacker.entity_id,
                target_id=target.entity_id if target else None,
                metadata={
                    "source": attacker,
                    "target": target,
                    "follower_attacks_this_turn": (
                        player.follower_attacks_this_turn
                    ),
                },
            )
        )
        self._resolve_event_queue()
        if self.state.pending_choice is not None:
            self._suspended_action = "attack"
            self._suspended_action_state = {
                "attacker_id": attacker.entity_id,
                "target_id": target.entity_id if target else None,
                "phase": "declared",
            }
            return

        if attacker not in player.board:
            return

        if target is not None and target not in opponent.board:
            attacker.consume_attack()
            return

        attacker.consume_attack()
        if target is None:
            result = self.apply_damage(attacker, None, attacker.attack, DamageType.COMBAT, self.current_player, attacker=attacker)
            was_ambush = attacker.ambush_active
            attacker.ambush_active = False
            if was_ambush:
                self._emit(GameEvent(EventType.AMBUSH_LOST, self.current_player, source_id=attacker.entity_id))
            self._emit(
                GameEvent(
                    EventType.DAMAGE_DEALT,
                    self.current_player,
                    source_id=attacker.entity_id,
                    amount=result.actual_amount,
                    metadata={"source": attacker, "target_player": 1 - self.current_player},
                )
            )
            self._log(
                self.current_player,
                f"{attacker.definition.name} 攻击对方主战者，造成 {result.actual_amount} 点伤害"
                f"（对方生命 {opponent.health}）",
            )
            return

        self._emit(
            GameEvent(
                EventType.COMBAT_STARTED,
                self.current_player,
                source_id=attacker.entity_id,
                target_id=target.entity_id,
                metadata={"source": attacker, "target": target},
            )
        )
        self._resolve_event_queue()
        if self.state.pending_choice is not None:
            self._suspended_action = "attack"
            self._suspended_action_state = {
                "attacker_id": attacker.entity_id,
                "target_id": target.entity_id,
                "phase": "combat",
            }
            return

        if attacker not in player.board or target not in opponent.board:
            return

        self._finish_attack_combat(attacker, target)

    def _finish_attack_combat(self, attacker: Unit, target: Unit) -> None:
        player = self.players[self.current_player]
        opponent = self.players[1 - self.current_player]
        if attacker not in player.board or target not in opponent.board:
            return

        attack_damage = attacker.attack
        counter_damage = target.attack

        result_c = self.apply_damage(target, attacker, counter_damage, DamageType.COMBAT, 1 - self.current_player, attacker=target)
        result_t = self.apply_damage(attacker, target, attack_damage, DamageType.COMBAT, self.current_player, attacker=attacker)

        was_ambush = attacker.ambush_active
        attacker.ambush_active = False
        if was_ambush:
            self._emit(GameEvent(EventType.AMBUSH_LOST, self.current_player, source_id=attacker.entity_id))
        self._emit(
            GameEvent(
                EventType.DAMAGE_DEALT,
                self.current_player,
                source_id=attacker.entity_id,
                target_id=target.entity_id,
                amount=result_t.actual_amount,
                metadata={"source": attacker, "target": target},
            )
        )
        self._emit(
            GameEvent(
                EventType.DAMAGE_DEALT,
                1 - self.current_player,
                source_id=target.entity_id,
                target_id=attacker.entity_id,
                amount=result_c.actual_amount,
                metadata={"source": target, "target": attacker},
            )
        )
        self._log(
            self.current_player,
            f"{attacker.definition.name} 攻击 {target.definition.name}，"
            f"造成 {result_t.actual_amount} 点并受到 {result_c.actual_amount} 点伤害",
        )

    def _resume_attack(self, state: dict) -> None:
        self._suspended_action = None
        self._suspended_action_state = None
        phase = state["phase"]
        attacker_id = state["attacker_id"]
        target_id = state["target_id"]

        player = self.players[self.current_player]
        opponent = self.players[1 - self.current_player]
        try:
            attacker = self._find_unit(player.board, attacker_id)
        except IllegalCommand:
            return
        if target_id is not None:
            try:
                found = self._find_board_entity(target_id)
            except IllegalCommand:
                found = None
            if found is None or not isinstance(found, Unit) or found not in opponent.board:
                return
            target = found
        else:
            target = None

        if phase == "declared":
            attacker.consume_attack()
            if target is None:
                result = self.apply_damage(attacker, None, attacker.attack, DamageType.COMBAT, self.current_player, attacker=attacker)
                was_ambush = attacker.ambush_active
                attacker.ambush_active = False
                if was_ambush:
                    self._emit(GameEvent(EventType.AMBUSH_LOST, self.current_player, source_id=attacker.entity_id))
                self._emit(
                    GameEvent(
                        EventType.DAMAGE_DEALT,
                        self.current_player,
                        source_id=attacker.entity_id,
                        amount=result.actual_amount,
                        metadata={"source": attacker, "target_player": 1 - self.current_player},
                    )
                )
                self._log(
                    self.current_player,
                    f"{attacker.definition.name} 攻击对方主战者，造成 {result.actual_amount} 点伤害"
                    f"（对方生命 {opponent.health}）",
                )
                return
            self._emit(
                GameEvent(
                    EventType.COMBAT_STARTED,
                    self.current_player,
                    source_id=attacker.entity_id,
                    target_id=target.entity_id,
                    metadata={"source": attacker, "target": target},
                )
            )
            self._resolve_event_queue()
            if self.state.pending_choice is not None:
                self._suspended_action = "attack"
                self._suspended_action_state = {
                    "attacker_id": attacker.entity_id,
                    "target_id": target.entity_id,
                    "phase": "combat",
                }
                return
            if attacker not in player.board or target not in opponent.board:
                return
            self._finish_attack_combat(attacker, target)
        elif phase == "combat":
            if target is None:
                return
            self._finish_attack_combat(attacker, target)

    def _end_turn(self) -> None:
        if (
            self._suspended_action_state is not None
            and self._suspended_action == "turn_end"
        ):
            state = self._suspended_action_state
            self._suspended_action = None
            self._suspended_action_state = None
            self._resume_end_turn(state)
            return

        player_index = self.current_player
        self._stabilize()
        if self.terminated:
            return
        records = self._collect_turn_end_records(player_index)
        self._begin_last_words_deferral()
        self._dispatch_emblem_triggers(
            player_index,
            "turn_end",
            freeze_conditions=True,
        )
        if self.terminated:
            self._abort_last_words_deferral()
            return
        if self.state.pending_choice is not None:
            self._suspend_turn_end(player_index, "records", records)
            return
        self._continue_turn_end_records(player_index, records)

    def _resume_end_turn(self, state: dict) -> None:
        self._suspended_action = None
        self._suspended_action_state = None
        player_index = state["player_index"]
        phase = state.get("phase", "records")
        if phase == "records":
            self._continue_turn_end_records(
                player_index,
                state.get("remaining_records", []),
            )
            return
        if phase == "last_words":
            self._finish_turn_end_timing(player_index)
            return
        if phase == "transition":
            self._complete_end_turn_transition(player_index)
            return
        raise IllegalCommand(f"Unknown turn-end resume phase: {phase}")

    def _suspend_turn_end(
        self,
        player_index: int,
        phase: str,
        remaining_records: list[dict[str, object]] | None = None,
    ) -> None:
        self._suspended_action = "turn_end"
        self._suspended_action_state = {
            "player_index": player_index,
            "phase": phase,
        }
        if remaining_records is not None:
            self._suspended_action_state["remaining_records"] = list(
                remaining_records
            )

    def _collect_turn_end_records(
        self,
        ending_player: int,
    ) -> list[dict[str, object]]:
        entities = (
            tuple(
                (ending_player, entity)
                for entity in self.players[ending_player].board
            )
            + tuple(
                (1 - ending_player, entity)
                for entity in self.players[1 - ending_player].board
                if isinstance(entity, Unit)
            )
        )
        records: list[dict[str, object]] = []
        for owner, entity in entities:
            snapshot = self._source_snapshot_from_board_card(entity, owner)
            operations = self._freeze_timing_operations(
                self._turn_end_operations(entity, owner, ending_player),
                owner,
                entity.entity_id,
                snapshot,
            )
            if not operations:
                continue
            records.append({
                "definition": entity.definition,
                "entity_id": entity.entity_id,
                "owner": owner,
                "operations": operations,
                "source_snapshot": snapshot,
            })
        return records

    def _continue_turn_end_records(
        self,
        player_index: int,
        remaining_records: list[dict[str, object]],
    ) -> None:
        records = list(remaining_records)
        while records:
            record = records.pop(0)
            self._start_effects(
                record["definition"],
                record["entity_id"],
                record["operations"],
                controller=record["owner"],
                label="回合结束",
                source_snapshot=record["source_snapshot"],
            )
            if self.terminated:
                self._abort_last_words_deferral()
                return
            if self.state.pending_choice is not None:
                self._suspend_turn_end(player_index, "records", records)
                return
        self._finish_turn_end_timing(player_index)

    def _finish_turn_end_timing(self, player_index: int) -> None:
        self._defer_last_words = False
        if not self._flush_deferred_death_batches():
            self._suspend_turn_end(player_index, "last_words")
            return
        if self.terminated:
            self._abort_last_words_deferral()
            return
        self._complete_end_turn_transition(player_index)

    def _complete_end_turn_transition(self, player_index: int) -> None:
        self._expire_modifiers(
            ModifierDuration.UNTIL_END_OF_TURN,
            player_index,
        )
        self._stabilize()
        if self.terminated:
            return
        if self.state.pending_choice is not None:
            self._suspend_turn_end(player_index, "transition")
            return
        self._emit(GameEvent(EventType.TURN_ENDED, player_index))
        self._log(player_index, "结束回合")
        self.players[player_index].cards_played_this_turn = 0
        self.players[player_index].follower_attacks_this_turn = 0
        self.players[player_index].mana = min(
            self.players[player_index].mana,
            self.players[player_index].max_mana,
        )
        self.players[player_index].extra_pp_active_turn = None
        self.state.active_player = 1 - player_index
        self.state.turn += 1
        self._start_turn(self.current_player)

    @staticmethod
    def _source_snapshot_from_board_card(
        entity: BoardCard,
        owner: int,
    ) -> SourceStateSnapshot:
        return SourceStateSnapshot(
            entity_id=entity.entity_id,
            controller=owner,
            card_id=entity.definition.card_id,
            card_type=entity.definition.card_type,
            attack=entity.attack if isinstance(entity, Unit) else None,
            health=entity.health if isinstance(entity, Unit) else None,
            evolved=entity.evolved if isinstance(entity, Unit) else False,
            super_evolved=(
                entity.super_evolved if isinstance(entity, Unit) else False
            ),
            effective_keywords=(
                entity.effective_keywords
                if isinstance(entity, Unit)
                else frozenset()
            ),
        )

    def _freeze_timing_operations(
        self,
        operations: tuple[EffectOperation, ...],
        controller: int,
        source_entity_id: int,
        source_snapshot: SourceStateSnapshot,
    ) -> tuple[EffectOperation, ...]:
        frozen: list[EffectOperation] = []
        meta_effects = {
            EffectKind.CONDITIONAL,
            EffectKind.CHOOSE_ONE,
            EffectKind.OPTIONAL,
            EffectKind.TARGET_EXISTS,
        }
        context = self._eval_context(
            controller,
            source_entity_id=source_entity_id,
            source_card_id=source_snapshot.card_id,
            source_snapshot=source_snapshot,
        )
        for operation in operations:
            if not operation.conditions or operation.kind in meta_effects:
                frozen.append(operation)
                continue
            state = evaluate_conditions_without_target(
                operation.conditions,
                context,
            )
            if state is PartialConditionResult.FALSE:
                continue
            frozen.append(
                replace(operation, conditions=())
                if state is PartialConditionResult.TRUE
                else operation
            )
        return tuple(frozen)

    def _turn_end_operations(
        self,
        unit: BoardCard,
        owner: int,
        ending_player: int,
    ) -> tuple[EffectOperation, ...]:
        operations: list[EffectOperation] = []
        if owner == ending_player:
            self._dispatch_ability(
                AbilityEvent.TURN_ENDED,
                unit,
                player_index=ending_player,
            )
            if not (
                isinstance(unit, Unit)
                and unit.printed_abilities_removed
            ):
                operations.extend(
                    self.rulebook.operations_for(
                        unit.definition.card_id,
                        Trigger.TURN_END,
                    )
                )
        matching_timing = (
            TurnEndDestroyTiming.OWNER_TURN
            if owner == ending_player
            else TurnEndDestroyTiming.OPPONENT_TURN
        )
        if isinstance(unit, Unit):
            for granted_ability in unit.granted_turn_end_abilities:
                if granted_ability.timing is matching_timing:
                    operations.extend(granted_ability.operations)
        if (
            isinstance(unit, Unit)
            and matching_timing in unit.turn_end_destroy_timings
        ):
            operations.append(
                EffectOperation(
                    kind=EffectKind.DESTROY,
                    target=TargetKind.SELF,
                )
            )
        if (
            isinstance(unit, Unit)
            and matching_timing in unit.turn_end_banish_timings
        ):
            operations.append(
                EffectOperation(
                    kind=EffectKind.BANISH,
                    target=TargetKind.SELF,
                )
            )
        return tuple(operations)

    def _choose(self, command: Choose) -> None:
        request = self.state.pending_choice
        if request is None:
            raise IllegalCommand("There is no pending choice")
        if command.option_id not in {option.option_id for option in request.options}:
            raise IllegalCommand("Choice option is invalid")
        option = next(
            option for option in request.options if option.option_id == command.option_id
        )
        if request.continuation_id == "match_mulligan":
            self._resolve_mulligan_choice(command, request)
            return
        if request.choice_kind is ChoiceKind.FUSION:
            self._resolve_fusion_choice(command, request)
            return
        self._log(
            command.player_index,
            f"选择目标：{option.label}",
        )
        if self.state.effect_stack:
            frame = self.state.effect_stack[-1]
            if request.choice_kind is ChoiceKind.MODE:
                selected_options = (*request.selected_options, option)
                if len(selected_options) < request.target_count:
                    remaining_options = tuple(
                        candidate
                        for candidate in request.options
                        if candidate.option_id != option.option_id
                    )
                    if not remaining_options:
                        raise IllegalCommand(
                            "Multi-mode choice has no remaining legal options"
                        )
                    self.state.pending_choice = replace(
                        request,
                        options=remaining_options,
                        request_id=self._allocate_choice_request_id(),
                        selected_options=selected_options,
                    )
                    self._log(
                        command.player_index,
                        f"已选择 {len(selected_options)}/{request.target_count} 个模式",
                    )
                    return
                self.state.pending_choice = None
                self.state.phase = Phase.MAIN
                self._resolve_choose_one_choice(
                    frame,
                    tuple(selected.option_id for selected in selected_options),
                )
                self._continue_effects()
                self._try_spellboost_hand()
                return
            if request.choice_kind is ChoiceKind.CONFIRM:
                self.state.pending_choice = None
                self.state.phase = Phase.MAIN
                if option.option_id == "optional:yes":
                    frame._decision_meta["optional_accepted"] = True
                    optional_ops = frame._decision_meta.get("optional_operations", ())
                    self._queue_effects_from_frame(
                        frame,
                        optional_ops,
                        label=f"{frame.label}/optional",
                    )
                else:
                    frame._decision_meta["optional_declined"] = all(
                        operation.kind is EffectKind.OPTIONAL
                        for operation in frame.operations
                    )
                self._continue_effects()
                self._try_spellboost_hand()
                return
            if request.target_count > 1 or request.selected_options:
                selected_target_id = self._choice_option_target_id(option)
                if selected_target_id is None:
                    raise IllegalCommand(
                        "Multi-target choice option has no target identity"
                    )
                if not self._choice_option_still_legal(frame, request, option):
                    self._log(
                        command.player_index,
                        f"目标 {option.label} "
                        f"{self._stale_choice_reason(frame, request, option)}，"
                        "取消本次多目标效果",
                    )
                    self.state.pending_choice = None
                    self.state.phase = Phase.MAIN
                    frame._decision_meta.pop("pending_target_snapshots", None)
                    frame.pending_target_ids.clear()
                    frame.next_index += 1
                    self._continue_effects()
                    self._try_spellboost_hand()
                    return
                selected_snapshots = (
                    *frame._decision_meta.get("pending_target_snapshots", ()),
                    self._bound_choice_snapshot(
                        frame,
                        request,
                        selected_target_id,
                    ),
                )
                selected_options = (*request.selected_options, option)
                if len(selected_options) < request.target_count:
                    remaining_options = request.options
                    if not request.allow_duplicate_targets:
                        remaining_options = tuple(
                            candidate
                            for candidate in request.options
                            if candidate.option_id != option.option_id
                        )
                    if not remaining_options:
                        raise IllegalCommand(
                            "Multi-target choice has no remaining legal options"
                        )
                    frame._decision_meta[
                        "pending_target_snapshots"
                    ] = selected_snapshots
                    self.state.pending_choice = replace(
                        request,
                        options=remaining_options,
                        request_id=self._allocate_choice_request_id(),
                        selected_options=selected_options,
                    )
                    self._log(
                        command.player_index,
                        f"已选择 {len(selected_options)}/{request.target_count} 个目标",
                    )
                    return
                target_ids = [
                    self._choice_option_target_id(selected)
                    for selected in selected_options
                ]
                if any(target_id is None for target_id in target_ids):
                    raise IllegalCommand(
                        "Multi-target choice option has no target identity"
                    )
                frame._decision_meta[
                    "pending_target_snapshots"
                ] = selected_snapshots
                frame.pending_target_ids = list(target_ids)
                self.state.pending_choice = None
                self.state.phase = Phase.MAIN
                self._continue_effects()
                self._try_spellboost_hand()
                return
            if not self._choice_option_still_legal(frame, request, option):
                self._log(
                    command.player_index,
                    f"目标 {option.label} {self._stale_choice_reason(frame, request, option)}，跳过",
                )
                self.state.pending_choice = None
                self.state.phase = Phase.MAIN
                frame.pending_target_id = None
                frame.next_index += 1
                self._continue_effects()
                self._try_spellboost_hand()
                return
            selected_target_id = (
                _leader_target_id(option.leader_player_index)
                if option.leader_player_index is not None
                else option.entity_id
            )
            frame.pending_target_id = selected_target_id
            operation = frame.operations[frame.next_index]
            if (
                selected_target_id is not None
                and operation.target_key
                and not _operation_produces_output_binding(operation)
            ):
                frame._decision_meta["pending_target_snapshots"] = (
                    self._bound_choice_snapshot(
                        frame,
                        request,
                        selected_target_id,
                    ),
                )
        self.state.pending_choice = None
        self.state.phase = Phase.MAIN
        if self.state.effect_stack:
            self._continue_effects()
        self._try_spellboost_hand()

    def _start_turn(self, player_index: int) -> None:
        if (
            self._suspended_action_state is not None
            and self._suspended_action == "turn_start"
        ):
            state = self._suspended_action_state
            self._suspended_action = None
            self._suspended_action_state = None
            self._resume_start_turn(state)
            return

        player = self.players[player_index]
        self._expire_modifiers(
            ModifierDuration.UNTIL_START_OF_NEXT_TURN,
            player_index,
        )
        self._stabilize()
        if self.terminated:
            return
        for emblem_owner in self.players:
            for ei in emblem_owner.emblems:
                ei.reset_turn_limits()
        self.state.listener_once_per_turn_used.clear()
        player.turns_started += 1
        player.extra_pp_active_turn = None
        if (
            player_index != self.state.first_player
            and player.turns_started == self.config.extra_pp_refresh_turn
        ):
            player.extra_pp_refresh_done = True
            if player.extra_pp_uses == 1:
                player.extra_pp_available = True
                self._emit(
                    GameEvent(
                        EventType.EXTRA_PP_REFRESHED,
                        player_index,
                        amount=1,
                        metadata={"turns_started": player.turns_started},
                    )
                )

        player.evolved_this_turn = False
        player.super_evolved_this_turn = False
        player.cards_played_this_turn = 0
        player.follower_attacks_this_turn = 0
        player.followers_destroyed_this_turn = 0
        player.max_mana = min(self.config.max_mana, player.max_mana + 1)
        player.mana = player.max_mana
        self._begin_last_words_deferral()
        self._tick_countdowns(player_index)
        self._tick_emblem_countdowns(player_index)
        if self.terminated:
            self._abort_last_words_deferral()
            return
        if self.state.pending_choice is not None:
            self._suspend_start_turn(player_index, "collect_timing")
            return
        self._continue_start_turn_after_countdowns(player_index)

    def _continue_start_turn_after_countdowns(self, player_index: int) -> None:
        board_records = self._collect_turn_start_records(player_index)
        invocation_card_ids = [
            card_id
            for card_id in self._turn_start_invocation_candidates(player_index)
            if self._invocation_conditions_met(player_index, card_id)
        ]
        self._dispatch_emblem_triggers(
            player_index,
            "turn_start",
            freeze_conditions=True,
        )
        if self.terminated:
            self._abort_last_words_deferral()
            return
        if self.state.pending_choice is not None:
            self._suspend_start_turn(
                player_index,
                "prepare_invocations",
                board_records=board_records,
                remaining_card_ids=invocation_card_ids,
            )
            return
        self._prepare_turn_start_invocations(
            player_index,
            board_records,
            invocation_card_ids,
        )

    def _collect_turn_start_records(
        self,
        player_index: int,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for unit in tuple(self.players[player_index].board):
            if not isinstance(unit, Unit):
                continue
            unit.can_attack = True
            unit.attacks_remaining = unit.attacks_per_turn
            unit.rush_only = False
            unit.summoned_this_turn = False
            self._dispatch_ability(
                AbilityEvent.TURN_STARTED, unit, player_index=player_index
            )
            operations = (
                ()
                if unit.printed_abilities_removed
                else self.rulebook.operations_for(
                    unit.definition.card_id, Trigger.TURN_START
                )
            )
            snapshot = self._source_snapshot_from_board_card(
                unit,
                player_index,
            )
            operations = self._freeze_timing_operations(
                operations,
                player_index,
                unit.entity_id,
                snapshot,
            )
            if operations:
                records.append({
                    "definition": unit.definition,
                    "entity_id": unit.entity_id,
                    "owner": player_index,
                    "operations": operations,
                    "source_snapshot": snapshot,
                })
        return records

    def _prepare_turn_start_invocations(
        self,
        player_index: int,
        board_records: list[dict[str, object]],
        remaining_card_ids: list[int],
    ) -> None:
        player = self.players[player_index]
        for card in player.hand:
            card_definition = (
                card.definition if isinstance(card, HandCard) else card
            )
            self._dispatch_card_ability(
                AbilityEvent.TURN_STARTED,
                card_definition,
                player_index=player_index,
            )
        for card in player.deck:
            self._dispatch_card_ability(
                AbilityEvent.TURN_STARTED,
                card.definition if isinstance(card, DeckCard) else card,
                player_index=player_index,
            )
        self._continue_turn_start_invocations(
            player_index,
            remaining_card_ids,
            board_records=board_records,
            invoked_records=[],
        )

    def _turn_start_invocation_candidates(self, player_index: int) -> list[int]:
        card_ids: list[int] = []
        for card in self.players[player_index].deck:
            definition = self.rulebook.invocation_for(card.card_id)
            if (
                definition is None
                or definition.trigger is not Trigger.TURN_START
                or card.card_type != "随从"
            ):
                continue
            card_ids.append(card.card_id)
        return card_ids

    def _invocation_conditions_met(self, player_index: int, card_id: int) -> bool:
        definition = self.rulebook.invocation_for(card_id)
        if definition is None:
            return False
        result = evaluate_conditions_without_target(
            definition.conditions,
            self._eval_context(player_index),
        )
        return result is PartialConditionResult.TRUE

    def _suspend_start_turn(
        self,
        player_index: int,
        phase: str,
        *,
        board_records: list[dict[str, object]] | None = None,
        remaining_card_ids: list[int] | None = None,
        invoked_records: list[dict[str, object]] | None = None,
    ) -> None:
        self._suspended_action = "turn_start"
        self._suspended_action_state = {
            "player_index": player_index,
            "phase": phase,
        }
        if board_records is not None:
            self._suspended_action_state["board_records"] = list(
                board_records
            )
        if remaining_card_ids is not None:
            self._suspended_action_state["remaining_card_ids"] = list(
                remaining_card_ids
            )
        if invoked_records is not None:
            self._suspended_action_state["invoked_records"] = list(
                invoked_records
            )

    def _continue_turn_start_invocations(
        self,
        player_index: int,
        remaining_card_ids: list[int],
        *,
        board_records: list[dict[str, object]],
        invoked_records: list[dict[str, object]],
    ) -> None:
        player = self.players[player_index]
        remaining_card_ids = list(remaining_card_ids)
        invoked_records = list(invoked_records)
        while remaining_card_ids:
            if len(player.board) >= self.config.max_board:
                break
            eligible_card_ids = [
                card_id
                for card_id in remaining_card_ids
                if any(card.card_id == card_id for card in player.deck)
            ]
            if not eligible_card_ids:
                break
            distinct_card_ids = set(eligible_card_ids)
            card_id = (
                eligible_card_ids[0]
                if len(distinct_card_ids) == 1
                else self.random.choice(eligible_card_ids)
            )
            remaining_card_ids = [
                remaining_id
                for remaining_id in remaining_card_ids
                if remaining_id != card_id
            ]
            definition = self.rulebook.invocation_for(card_id)
            if (
                definition is None
                or definition.trigger is not Trigger.TURN_START
            ):
                continue
            deck_index = next(
                (
                    index
                    for index in range(len(player.deck) - 1, -1, -1)
                    if player.deck[index].card_id == card_id
                ),
                None,
            )
            if deck_index is None:
                continue
            card = player.deck[deck_index]
            if card.card_type != "随从":
                continue
            player.deck.pop(deck_index)
            card_definition = (
                card.definition if isinstance(card, DeckCard) else card
            )
            unit = self._summon_follower_to_board(
                player_index,
                card_definition,
                summon_cause="invocation",
                origin=CardOrigin.DECK,
            )
            if unit is None:
                raise IllegalCommand("Invocation failed after board-space validation")
            self._log(
                player_index,
                f"瞬念召唤 {card.name} ({unit.attack}/{unit.health})",
            )
            self._emit(
                GameEvent(
                    EventType.FOLLOWER_SUMMONED,
                    player_index,
                    source_id=unit.entity_id,
                    metadata={
                        "source": unit,
                        "card_id": card.card_id,
                        "origin": unit.origin.value,
                        "derived": False,
                        "token": False,
                        "via": "invocation",
                    },
                )
            )
            self._emit(
                GameEvent(
                    EventType.CARD_INVOKED,
                    player_index,
                    source_id=unit.entity_id,
                    metadata={
                        "source": unit,
                        "card_id": card.card_id,
                        "origin": unit.origin.value,
                    },
                )
            )
            invoke_operations = self.rulebook.operations_for(
                unit.definition.card_id,
                Trigger.INVOKE,
            )
            if invoke_operations:
                invoked_records.append({
                    "definition": unit.definition,
                    "entity_id": unit.entity_id,
                    "owner": player_index,
                    "operations": invoke_operations,
                    "source_snapshot": (
                        self._source_snapshot_from_board_card(
                            unit,
                            player_index,
                        )
                    ),
                })
            self._resolve_event_queue()
            self._stabilize()
            if self.state.pending_choice is not None:
                self._suspend_start_turn(
                    player_index,
                    "invocation_scan",
                    board_records=board_records,
                    remaining_card_ids=remaining_card_ids,
                    invoked_records=invoked_records,
                )
                return
        self._finish_start_turn_timing(
            player_index,
            board_records,
            invoked_records,
        )

    def _finish_start_turn_timing(
        self,
        player_index: int,
        board_records: list[dict[str, object]],
        invoked_records: list[dict[str, object]],
    ) -> None:
        self._defer_last_words = False
        if not self._flush_deferred_death_batches():
            self._suspend_start_turn(
                player_index,
                "deferred_last_words",
                board_records=board_records,
                invoked_records=invoked_records,
            )
            return
        if self.terminated:
            self._abort_last_words_deferral()
            return
        self._continue_start_turn_board_records(
            player_index,
            board_records,
            invoked_records,
        )

    def _continue_start_turn_board_records(
        self,
        player_index: int,
        board_records: list[dict[str, object]],
        invoked_records: list[dict[str, object]],
    ) -> None:
        records = list(board_records)
        while records:
            record = records.pop(0)
            self._start_effects(
                record["definition"],
                record["entity_id"],
                record["operations"],
                controller=record["owner"],
                label="回合开始",
                source_snapshot=record["source_snapshot"],
            )
            if self.terminated:
                return
            if self.state.pending_choice is not None:
                self._suspend_start_turn(
                    player_index,
                    "board",
                    board_records=records,
                    invoked_records=invoked_records,
                )
                return
        self._continue_start_turn_invoked_effects(
            player_index,
            invoked_records,
        )

    def _continue_start_turn_invoked_effects(
        self,
        player_index: int,
        invoked_records: list[dict[str, object]],
    ) -> None:
        records = list(invoked_records)
        while records:
            record = records.pop(0)
            self._start_effects(
                record["definition"],
                record["entity_id"],
                record["operations"],
                controller=record["owner"],
                label="瞬念召唤",
                source_snapshot=record["source_snapshot"],
            )
            if self.terminated:
                return
            if self.state.pending_choice is not None:
                self._suspend_start_turn(
                    player_index,
                    "invoked_effects",
                    invoked_records=records,
                )
                return
        self._complete_start_turn(player_index)

    def _complete_start_turn(self, player_index: int) -> None:
        player = self.players[player_index]
        self._emit(GameEvent(EventType.TURN_STARTED, player_index))
        self._log(
            player_index,
            f"第 {player.turns_started} 个行动回合开始，"
            f"能量 {player.mana}/{player.max_mana}",
        )
        self._draw(player_index, reason="回合抽牌")
        self._stabilize()

    def _resume_start_turn(self, state: dict) -> None:
        self._suspended_action = None
        self._suspended_action_state = None
        player_index = state["player_index"]
        phase = state["phase"]
        board_records = state.get("board_records", [])
        invoked_records = state.get("invoked_records", [])
        if phase == "collect_timing":
            self._continue_start_turn_after_countdowns(player_index)
            return
        if phase == "prepare_invocations":
            self._prepare_turn_start_invocations(
                player_index,
                board_records,
                state.get("remaining_card_ids", []),
            )
            return
        if phase == "invocation_scan":
            self._continue_turn_start_invocations(
                player_index,
                state.get("remaining_card_ids", []),
                board_records=board_records,
                invoked_records=invoked_records,
            )
            return
        if phase == "deferred_last_words":
            self._finish_start_turn_timing(
                player_index,
                board_records,
                invoked_records,
            )
            return
        if phase == "board":
            self._continue_start_turn_board_records(
                player_index,
                board_records,
                invoked_records,
            )
            return
        if phase == "invoked_effects":
            self._continue_start_turn_invoked_effects(
                player_index,
                invoked_records,
            )
            return
        raise IllegalCommand(f"Unknown turn-start resume phase: {phase}")

    def _draw(
        self,
        player_index: int,
        *,
        reason: str,
    ) -> BoundTargetSnapshot | None:
        player = self.players[player_index]
        if player.deck:
            card = player.deck.pop()
            drawn_cost = (
                card.current_cost if isinstance(card, DeckCard) else card.cost
            )
            if len(player.hand) < self.config.max_hand:
                hand_card = self._append_hand_card(
                    player, card, origin=CardOrigin.DECK
                )
                self._emit(
                    GameEvent(
                        EventType.CARD_DRAWN,
                        player_index,
                        source_id=hand_card.entity_id,
                        metadata={
                            "card_id": card.card_id,
                            "source": hand_card,
                        },
                    )
                )
                self._log(player_index, f"{reason}：{card.name}")
                return self._bound_hand_snapshot(player_index, hand_card)
            else:
                card_definition = (
                    card.definition if isinstance(card, DeckCard) else card
                )
                graveyard_card = self._send_to_graveyard(
                    player_index, card_definition, "overdraw",
                    origin=CardOrigin.DECK,
                )
                self._log(player_index, f"{reason}：{card.name}，手牌已满而被弃置")
                return self._bound_graveyard_snapshot(
                    graveyard_card,
                    cost=drawn_cost,
                )
        outcome = player.empty_deck_outcome
        winner = (
            player_index
            if outcome is EmptyDeckOutcome.VICTORY
            else 1 - player_index
        )
        self.state.winner = winner
        self.state.phase = Phase.FINISHED
        self._emit(GameEvent(
            EventType.EMPTY_DECK_DRAW_RESOLVED,
            player_index,
            metadata={
                "outcome": outcome.value,
                "winner": winner,
                "reason": reason,
            },
        ))
        result = "获得胜利" if outcome is EmptyDeckOutcome.VICTORY else "战败"
        self._log(player_index, f"牌组为0张时抽牌，{result}")
        return None

    def _draw_filtered(
        self,
        player_index: int,
        *,
        deck_filter: DeckFilter | None = None,
        excluded_card_names: frozenset[str] = frozenset(),
        reason: str,
    ) -> BoundTargetSnapshot | None:
        player = self.players[player_index]
        candidates = [
            index
            for index, card in enumerate(player.deck)
            if (
                card.name not in excluded_card_names
                and (deck_filter is None or deck_filter.matches(card))
            )
        ]
        if not candidates:
            self._log(player_index, f"{reason}：没有符合条件的卡牌")
            return None
        index = self.random.choice(candidates)
        card = player.deck.pop(index)
        drawn_cost = (
            card.current_cost if isinstance(card, DeckCard) else card.cost
        )
        if len(player.hand) < self.config.max_hand:
            hand_card = self._append_hand_card(
                player, card, origin=CardOrigin.DECK
            )
            self._emit(
                GameEvent(
                    EventType.CARD_DRAWN,
                    player_index,
                    source_id=hand_card.entity_id,
                    metadata={
                        "card_id": card.card_id,
                        "source": hand_card,
                        "filtered": True,
                        "card_type_filter": None if deck_filter is None else deck_filter.card_type,
                        "class_id_filter": None if deck_filter is None else deck_filter.class_id,
                        "class_name_filter": None if deck_filter is None else deck_filter.class_name,
                        "cost_min_filter": None if deck_filter is None else deck_filter.cost_min,
                        "cost_max_filter": None if deck_filter is None else deck_filter.cost_max,
                        "card_id_filter": None if deck_filter is None else deck_filter.card_id,
                        "card_name_filter": None if deck_filter is None else deck_filter.card_name,
                        "life_min_filter": None if deck_filter is None else deck_filter.life_min,
                        "life_max_filter": None if deck_filter is None else deck_filter.life_max,
                    },
                )
            )
            self._log(player_index, f"{reason}：{card.name}")
            return self._bound_hand_snapshot(player_index, hand_card)
        else:
            card_definition = (
                card.definition if isinstance(card, DeckCard) else card
            )
            graveyard_card = self._send_to_graveyard(
                player_index,
                card_definition,
                "overdraw",
                origin=CardOrigin.DECK,
            )
            self._log(player_index, f"{reason}：{card.name}，手牌已满而被弃置")
            return self._bound_graveyard_snapshot(
                graveyard_card,
                cost=drawn_cost,
            )

    def _fanfare_operations(self, unit: Unit) -> tuple[EffectOperation, ...]:
        explicit = self.rulebook.operations_for(
            unit.definition.card_id, Trigger.FANFARE
        )
        if explicit:
            return explicit
        operations: list[EffectOperation] = []
        for effect in unit.definition.fanfare_effects:
            operation = {
                "draw": EffectOperation(
                    EffectKind.DRAW, TargetKind.OWN_LEADER, effect.amount
                ),
                "heal_leader": EffectOperation(
                    EffectKind.HEAL_LEADER, TargetKind.OWN_LEADER, effect.amount
                ),
                "damage_enemy_leader": EffectOperation(
                    EffectKind.DAMAGE_LEADER, TargetKind.ENEMY_LEADER, effect.amount
                ),
                "damage_own_leader": EffectOperation(
                    EffectKind.DAMAGE_LEADER, TargetKind.OWN_LEADER, effect.amount
                ),
                "restore_mana": EffectOperation(
                    EffectKind.RESTORE_MANA, TargetKind.OWN_LEADER, effect.amount
                ),
                "buff_self": EffectOperation(
                    EffectKind.BUFF_UNIT,
                    TargetKind.SELF,
                    effect.amount,
                    effect.secondary_amount,
                ),
            }.get(effect.kind)
            if operation is not None:
                operations.append(operation)
        return tuple(operations)

    def _execute_fanfare(self, unit: Unit) -> None:
        operations = self._fanfare_operations(unit)
        if operations:
            self._start_effects(
                unit.definition, unit.entity_id, operations, label="入场曲"
            )

    def _build_eval_context(self, frame: EffectFrame, target_id: int | None) -> EvalContext:
        return self._eval_context(
            frame.controller,
            source_entity_id=frame.source_entity_id,
            target_entity_id=target_id,
            source_card_id=frame.source_card_id,
            source_fusion_count=len(frame.fusion_materials),
            source_fusion_distinct_name_count=len({
                material.definition.name
                for material in frame.fusion_materials
            }),
            source_spellboost_count=frame.source_spellboost_count,
            source_cost=frame.source_cost,
            distributed_value=frame.distributed_value,
            listener_activation_count=frame.listener_activation_count,
            event_source_entity_id=frame.event_source_entity_id,
            event_source_base_cost=frame.event_source_base_cost,
            source_snapshot=frame.source_snapshot,
            attack_target_entity_id=frame.attack_target_entity_id,
            bound_target_snapshots=frame._target_binding_snapshots,
        )

    def _resolve_amount(self, operation: EffectOperation, ctx: EvalContext) -> int:
        if operation.amount_expr is not None:
            return evaluate_expression(operation.amount_expr, ctx)
        return operation.amount

    @staticmethod
    def _bound_expression_inputs_available(
        operation: EffectOperation,
        frame: EffectFrame,
    ) -> bool:
        return all(
            _expression_bindings_available(
                expression,
                frame._target_binding_snapshots,
            )
            for expression in (
                operation.amount_expr,
                operation.secondary_expr,
            )
        )

    def _resolve_secondary(self, operation: EffectOperation, ctx: EvalContext) -> int:
        if operation.secondary_expr is not None:
            return evaluate_expression(operation.secondary_expr, ctx)
        return operation.secondary_amount

    def _source_entity_in_play(self, frame: EffectFrame) -> bool:
        if frame.listener_activation_zone is not None:
            return self._listener_source_definition(
                frame.listener_activation_owner,
                frame.listener_activation_zone,
                frame.listener_activation_entity_id,
                frame.listener_activation_card_id,
            ) is not None
        if frame.source_entity_id is None:
            return False
        if any(
            entity.entity_id == frame.source_entity_id
            for player in self.players
            for entity in player.board
        ):
            return True
        return any(
            instance.entity_id == frame.source_entity_id
            and instance.source_card_id == frame.source_card_id
            for player in self.players
            for instance in (*player.emblems, *player.faiths)
        )

    @staticmethod
    def _operation_requires_live_source_target(
        operation: EffectOperation,
    ) -> bool:
        return (
            operation.target is TargetKind.SELF
            and operation.kind in _SOURCE_REQUIRED_SELF_TARGET_EFFECTS
        )

    @staticmethod
    def _operation_requires_source_state(
        operation: EffectOperation,
    ) -> bool:
        return (
            _expression_depends_on_source(operation.amount_expr)
            or _expression_depends_on_source(operation.secondary_expr)
            or _expression_depends_on_source(operation.deck_filter_cost_expr)
            or _expression_depends_on_source(operation.target_count_expr)
            or any(
                _condition_depends_on_source(condition)
                for condition in operation.conditions
            )
        )

    def _checked_execute(
        self, operation: EffectOperation, frame: EffectFrame, target_id: int | None,
    ) -> None:
        clause_id = self._runtime_clause_id(operation, frame)
        source_in_play = self._source_entity_in_play(frame)
        if self._operation_requires_live_source_target(operation) and not source_in_play:
            return
        if (
            self._operation_requires_source_state(operation)
            and not source_in_play
            and frame.source_snapshot is None
        ):
            return
        if operation.target is TargetKind.EMBLEM_SELF:
            target_id = frame.emblem_activation_entity_id
            if target_id is None:
                return
        elif operation.target is TargetKind.SELF:
            target_id = frame.source_entity_id
        elif operation.target is TargetKind.EVENT_SOURCE:
            target_id = frame.event_source_entity_id
            if target_id is None:
                return
            if operation.kind in _EVENT_SOURCE_BOARD_EFFECTS:
                if operation.kind is EffectKind.TRANSFORM:
                    try:
                        self._find_hand_card(frame.controller, target_id)
                    except IllegalCommand:
                        try:
                            self._find_board_entity(target_id)
                        except IllegalCommand:
                            return
                else:
                    try:
                        self._find_board_entity(target_id)
                    except IllegalCommand:
                        return
        elif operation.target is TargetKind.ATTACK_TARGET:
            target_id = frame.attack_target_entity_id
            if target_id is None:
                return
            try:
                self._find_board_entity(target_id)
            except IllegalCommand:
                return
        if operation.board_filter is not None and target_id is not None:
            try:
                filtered_target = self._find_board_entity(target_id)
            except IllegalCommand:
                return
            if not operation.board_filter.matches_entity(filtered_target):
                return
        ctx = self._build_eval_context(frame, target_id)
        is_meta = operation.kind in (
            EffectKind.CONDITIONAL,
            EffectKind.CHOOSE_ONE,
            EffectKind.OPTIONAL,
            EffectKind.TARGET_EXISTS,
        )
        if not is_meta:
            for cond in operation.conditions:
                if not evaluate_condition(cond, ctx):
                    if (
                        self.runtime_coverage is not None
                        and clause_id is not None
                    ):
                        self.runtime_coverage.record_clause(
                            clause_id,
                            "condition_evaluated",
                        )
                        self.runtime_coverage.record_clause(
                            clause_id,
                            "condition_false",
                        )
                    return
            if (
                self.runtime_coverage is not None
                and clause_id is not None
                and operation.conditions
            ):
                self.runtime_coverage.record_clause(
                    clause_id,
                    "condition_evaluated",
                )
                self.runtime_coverage.record_clause(
                    clause_id,
                    "condition_true",
                )
        amount = self._resolve_amount(operation, ctx)
        secondary = self._resolve_secondary(operation, ctx)
        resolved_deck_filter = operation.deck_filter
        if operation.deck_filter_cost_expr is not None:
            resolved_cost = max(
                0,
                evaluate_expression(operation.deck_filter_cost_expr, ctx),
            )
            resolved_deck_filter = replace(
                operation.deck_filter or DeckFilter(),
                cost_min=resolved_cost,
                cost_max=resolved_cost,
            )
        if operation.kind in (
            EffectKind.HEAL_LEADER,
            EffectKind.HEAL_UNIT,
            EffectKind.HEAL_UNIT_AND_LEADER,
        ):
            amount = max(0, amount)
        self._record_runtime_capacity(operation, frame, clause_id)
        if self.runtime_coverage is not None and clause_id is not None:
            self.runtime_coverage.record_clause(
                clause_id,
                "operation_executed",
            )
        if (
            operation.amount_expr is not None
            or operation.secondary_expr is not None
            or operation.deck_filter_cost_expr is not None
            or amount != operation.amount
            or secondary != operation.secondary_amount
        ):
            resolved = replace(
                operation,
                amount=amount,
                secondary_amount=secondary,
                amount_expr=None,
                secondary_expr=None,
                deck_filter=resolved_deck_filter,
                deck_filter_cost_expr=None,
            )
            self._execute_effect(resolved, frame, target_id)
        else:
            self._execute_effect(operation, frame, target_id)

    def _execute_effect(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        player = self.players[frame.controller]
        opponent = self.players[1 - frame.controller]
        name = frame.source_name
        if effect.kind is EffectKind.SELECT_TARGETS:
            return
        if effect.kind is EffectKind.DRAW:
            draw_player = (
                1 - frame.controller
                if effect.target is TargetKind.ENEMY_LEADER
                else frame.controller
            )
            drawn_snapshots: list[BoundTargetSnapshot] = []
            for _ in range(effect.amount):
                snapshot = self._draw(
                    draw_player,
                    reason=f"{name} {frame.label}抽牌",
                )
                if snapshot is not None:
                    drawn_snapshots.append(snapshot)
                if self.terminated:
                    break
            if effect.target_key:
                self._bind_targets(
                    frame,
                    effect.target_key,
                    tuple(snapshot.entity_id for snapshot in drawn_snapshots),
                    effect,
                    snapshots=tuple(drawn_snapshots),
                )
        elif effect.kind is EffectKind.DRAW_FILTERED:
            draw_player = (
                1 - frame.controller
                if effect.target is TargetKind.ENEMY_LEADER
                else frame.controller
            )
            drawn_snapshots = []
            drawn_card_names: set[str] = set()
            for _ in range(effect.amount):
                snapshot = self._draw_filtered(
                    draw_player,
                    deck_filter=effect.deck_filter,
                    excluded_card_names=(
                        frozenset(drawn_card_names)
                        if effect.distinct_card_names
                        else frozenset()
                    ),
                    reason=f"{name} {frame.label}抽牌",
                )
                if snapshot is not None:
                    drawn_snapshots.append(snapshot)
                    drawn_card_names.add(snapshot.card_name)
            if effect.target_key:
                self._bind_targets(
                    frame,
                    effect.target_key,
                    tuple(snapshot.entity_id for snapshot in drawn_snapshots),
                    effect,
                    snapshots=tuple(drawn_snapshots),
                )
        elif effect.kind is EffectKind.HEAL_LEADER:
            target_idx = (
                _leader_index_from_target_id(target_id)
                if _is_leader_target_id(target_id)
                else frame.controller
            )
            target_player = self.players[target_idx]
            before = target_player.health
            target_player.health = min(
                target_player.max_health,
                target_player.health + effect.amount,
            )
            actual_heal = target_player.health - before
            self._log(
                frame.controller,
                f"{name} {frame.label}回复 {actual_heal} 点生命"
                f"（生命 {target_player.health}）",
            )
            if actual_heal > 0:
                self._emit(GameEvent(
                    EventType.LEADER_HEALED,
                    target_idx,
                    source_id=frame.source_entity_id,
                    amount=actual_heal,
                    metadata={"card_id": frame.source_card_id},
                ))
        elif effect.kind is EffectKind.HEAL_UNIT:
            target = self._find_board_entity(target_id)
            if not isinstance(target, Unit):
                raise IllegalCommand("Heal target must be a follower")
            before = target.health
            target.health = min(target.max_health, target.health + effect.amount)
            actual_heal = target.health - before
            self._log(
                frame.controller,
                f"{name} {frame.label}回复 {target.definition.name} "
                f"{actual_heal} 点生命（生命 {target.health}）",
            )
            if actual_heal > 0:
                self._emit(
                    GameEvent(
                        EventType.FOLLOWER_HEALED,
                        self._entity_owner(target.entity_id),
                        source_id=frame.source_entity_id,
                        target_id=target.entity_id,
                        amount=actual_heal,
                        metadata={
                            "card_id": frame.source_card_id,
                            "target": target,
                        },
                    )
                )
        elif effect.kind is EffectKind.HEAL_UNIT_AND_LEADER:
            target = self._find_board_entity(target_id)
            if not isinstance(target, Unit):
                raise IllegalCommand("Heal target must be a follower")
            unit_before = target.health
            target.health = min(
                target.max_health,
                target.health + effect.amount,
            )
            actual_heal = target.health - unit_before
            self._log(
                frame.controller,
                f"{name} {frame.label}回复 {target.definition.name} "
                f"{actual_heal} 点生命（生命 {target.health}）",
            )
            if actual_heal > 0:
                self._emit(
                    GameEvent(
                        EventType.FOLLOWER_HEALED,
                        self._entity_owner(target.entity_id),
                        source_id=frame.source_entity_id,
                        target_id=target.entity_id,
                        amount=actual_heal,
                        metadata={
                            "card_id": frame.source_card_id,
                            "target": target,
                        },
                    )
                )
                leader = self.players[frame.controller]
                leader_before = leader.health
                leader.health = min(
                    leader.max_health,
                    leader.health + actual_heal,
                )
                leader_actual_heal = leader.health - leader_before
                self._log(
                    frame.controller,
                    f"{name} {frame.label}回复主战者 "
                    f"{leader_actual_heal} 点生命（生命 {leader.health}）",
                )
                if leader_actual_heal > 0:
                    self._emit(
                        GameEvent(
                            EventType.LEADER_HEALED,
                            frame.controller,
                            source_id=frame.source_entity_id,
                            amount=leader_actual_heal,
                            metadata={"card_id": frame.source_card_id},
                        )
                    )
        elif effect.kind is EffectKind.DAMAGE_LEADER:
            target_idx = (
                _leader_index_from_target_id(target_id)
                if _is_leader_target_id(target_id)
                else (
                    1 - frame.controller
                    if effect.target is TargetKind.ENEMY_LEADER
                    else frame.controller
                )
            )
            is_enemy = target_idx != frame.controller
            self.apply_damage(None, None, effect.amount,
                             DamageType.EFFECT if is_enemy else DamageType.ABILITY,
                             frame.controller, target_player_index=target_idx)
            target_player = self.players[target_idx]
            target_name = "对方" if is_enemy else "己方"
            self._log(
                frame.controller,
                f"{name} {frame.label}对{target_name}主战者造成 {effect.amount} 点伤害"
                f"（生命 {target_player.health}）",
            )
        elif effect.kind is EffectKind.DAMAGE_UNIT:
            if _is_leader_target_id(target_id):
                target_idx = _leader_index_from_target_id(target_id)
                self.apply_damage(
                    None,
                    None,
                    effect.amount,
                    DamageType.EFFECT,
                    frame.controller,
                    target_player_index=target_idx,
                )
                target_player = self.players[target_idx]
                target_name = "己方" if target_idx == frame.controller else "对方"
                self._log(
                    frame.controller,
                    f"{name} {frame.label}对{target_name}主战者造成 {effect.amount} 点伤害"
                    f"（生命 {target_player.health}）",
                )
                return
            target = self._find_board_entity(target_id)
            if not isinstance(target, Unit):
                raise IllegalCommand("Damage target must be a follower")
            self.apply_damage(None, target, effect.amount, DamageType.EFFECT, frame.controller)
        elif effect.kind is EffectKind.DISTRIBUTE_DAMAGE:
            self._execute_distribute_damage(effect, frame)
        elif effect.kind is EffectKind.RESTORE_MANA:
            restored = min(
                effect.amount,
                self._effective_mana_cap(player) - player.mana,
            )
            player.mana += restored
            self._log(
                frame.controller,
                f"{name} {frame.label}回复 {restored} 点能量",
            )
        elif effect.kind in {
            EffectKind.RESTORE_EVOLUTION_POINTS,
            EffectKind.RESTORE_SUPER_EVOLUTION_POINTS,
        }:
            is_super = effect.kind is EffectKind.RESTORE_SUPER_EVOLUTION_POINTS
            attribute = (
                "super_evolution_points" if is_super else "evolution_points"
            )
            maximum = (
                self.config.starting_super_evolution_points
                if is_super
                else self.config.starting_evolution_points
            )
            before = getattr(player, attribute)
            after = min(maximum, before + effect.amount)
            restored = after - before
            setattr(player, attribute, after)
            self._emit(GameEvent(
                (
                    EventType.SUPER_EVOLUTION_POINTS_RESTORED
                    if is_super
                    else EventType.EVOLUTION_POINTS_RESTORED
                ),
                frame.controller,
                source_id=frame.source_entity_id,
                amount=restored,
                metadata={
                    "source_card_id": frame.source_card_id,
                    "requested_amount": effect.amount,
                    "before": before,
                    "after": after,
                    "maximum": maximum,
                },
            ))
            resource_name = "超进化点" if is_super else "进化点"
            self._log(
                frame.controller,
                f"{name} {frame.label}回复 {restored} 点{resource_name}",
            )
        elif effect.kind is EffectKind.CHANGE_MAX_MANA:
            target_player_index = (
                1 - frame.controller
                if effect.target is TargetKind.ENEMY_LEADER
                else frame.controller
            )
            target_player = self.players[target_player_index]
            before_max = target_player.max_mana
            before_mana = target_player.mana
            target_player.max_mana = max(
                0,
                min(self.config.max_mana, before_max + effect.amount),
            )
            target_player.mana = min(
                target_player.mana,
                self._effective_mana_cap(target_player),
            )
            self._emit(GameEvent(
                EventType.MAX_MANA_CHANGED,
                target_player_index,
                source_id=frame.source_entity_id,
                metadata={
                    "source_card_id": frame.source_card_id,
                    "requested_amount": effect.amount,
                    "applied_amount": target_player.max_mana - before_max,
                    "before_max_mana": before_max,
                    "after_max_mana": target_player.max_mana,
                    "before_mana": before_mana,
                    "after_mana": target_player.mana,
                },
            ))
            self._log(
                frame.controller,
                f"{name} {frame.label}使能量上限由 {before_max} "
                f"变为 {target_player.max_mana}",
            )
        elif effect.kind in {
            EffectKind.SET_LEADER_MAX_HEALTH,
            EffectKind.CHANGE_LEADER_MAX_HEALTH,
        }:
            target_player_index = (
                1 - frame.controller
                if effect.target is TargetKind.ENEMY_LEADER
                else frame.controller
            )
            target_player = self.players[target_player_index]
            previous_max = target_player.max_health
            previous_health = target_player.health
            target_player.max_health = (
                effect.amount
                if effect.kind is EffectKind.SET_LEADER_MAX_HEALTH
                else max(1, previous_max + effect.amount)
            )
            target_player.health = min(
                target_player.health,
                target_player.max_health,
            )
            self._emit(GameEvent(
                EventType.LEADER_MAX_HEALTH_CHANGED,
                target_player_index,
                source_id=frame.source_entity_id,
                amount=target_player.max_health - previous_max,
                metadata={
                    "source_card_id": frame.source_card_id,
                    "requested_amount": effect.amount,
                    "applied_amount": target_player.max_health - previous_max,
                    "previous_max_health": previous_max,
                    "current_max_health": target_player.max_health,
                    "previous_health": previous_health,
                    "current_health": target_player.health,
                },
            ))
            self._log(
                frame.controller,
                f"{name} {frame.label}使玩家 {target_player_index + 1} "
                f"的生命值上限由 {previous_max} "
                f"变为 {target_player.max_health}",
            )
        elif effect.kind is EffectKind.BUFF_UNIT:
            target = (
                self._find_board_entity(target_id)
                if target_id is not None
                else (
                    self._find_board_entity(frame.source_entity_id)
                    if frame.source_entity_id is not None
                    else None
                )
            )
            if not isinstance(target, Unit):
                raise IllegalCommand("Buff target must be a follower")
            modifier = StatModifier(
                modifier_id=self._allocate_modifier_id(),
                attack_delta=effect.amount,
                health_delta=effect.secondary_amount,
                duration=effect.duration.value,
                expires_for_player=_expires_for_player(
                    effect.duration,
                    frame.controller,
                    self.state.active_player,
                ),
            )
            target.add_stat_modifier(modifier)
            if effect.amount > 0 or effect.secondary_amount > 0:
                owner = self._entity_owner(target.entity_id)
                self._emit(
                    GameEvent(
                        EventType.FOLLOWER_STATS_INCREASED,
                        owner,
                        source_id=target.entity_id,
                        amount=max(0, effect.amount),
                        metadata={
                            "source": target,
                            "card_id": target.definition.card_id,
                            "attack_delta": max(0, effect.amount),
                            "health_delta": max(0, effect.secondary_amount),
                            "effect_source_card_id": frame.source_card_id,
                            "effect_source_entity_id": frame.source_entity_id,
                        },
                    )
                )
            if effect.amount < 0 or effect.secondary_amount < 0:
                owner = self._entity_owner(target.entity_id)
                self._emit(
                    GameEvent(
                        EventType.FOLLOWER_STATS_DECREASED,
                        owner,
                        source_id=target.entity_id,
                        amount=max(
                            0,
                            -effect.amount,
                            -effect.secondary_amount,
                        ),
                        metadata={
                            "source": target,
                            "card_id": target.definition.card_id,
                            "attack_delta": min(0, effect.amount),
                            "health_delta": min(
                                0,
                                effect.secondary_amount,
                            ),
                            "effect_source_card_id": frame.source_card_id,
                            "effect_source_entity_id": (
                                frame.source_entity_id
                            ),
                        },
                    )
                )
            self._log(
                frame.controller,
                f"属性变化 {effect.amount}/{effect.secondary_amount}",
            )
        elif effect.kind is EffectKind.BUFF_HAND_CARD:
            owner, hand_card = self._find_hand_card_with_owner(target_id)
            expected_owner = (
                1 - frame.controller
                if effect.target in {
                    TargetKind.RANDOM_ENEMY_HAND,
                    TargetKind.ALL_ENEMY_HAND,
                }
                else frame.controller
            )
            if owner != expected_owner:
                raise IllegalCommand(
                    "Hand stat buff target belongs to the wrong player"
                )
            if hand_card.card_type != "随从":
                raise IllegalCommand("Hand stat buff target must be a follower")
            modifier = StatModifier(
                modifier_id=self._allocate_modifier_id(),
                attack_delta=effect.amount,
                health_delta=effect.secondary_amount,
                duration=effect.duration.value,
                expires_for_player=_expires_for_player(
                    effect.duration,
                    frame.controller,
                    self.state.active_player,
                ),
            )
            hand_card.add_stat_modifier(modifier)
            if effect.amount > 0 or effect.secondary_amount > 0:
                self._emit(
                    GameEvent(
                        EventType.HAND_FOLLOWER_STATS_INCREASED,
                        owner,
                        source_id=frame.source_entity_id,
                        target_id=hand_card.entity_id,
                        amount=max(0, effect.amount),
                        metadata={
                            "card_id": hand_card.card_id,
                            "attack_delta": max(0, effect.amount),
                            "health_delta": max(0, effect.secondary_amount),
                            "effect_source_card_id": frame.source_card_id,
                            "effect_source_entity_id": frame.source_entity_id,
                        },
                    )
                )
            self._log(
                frame.controller,
                f"手牌 {hand_card.name} 属性变化 "
                f"{effect.amount}/{effect.secondary_amount}",
            )
        elif effect.kind is EffectKind.ADD_UNION_BURST_GAUGE:
            owner, hand_card = self._find_hand_card_with_owner(target_id)
            if owner != frame.controller:
                raise IllegalCommand(
                    "Union Burst gauge target must be in the controller's hand"
                )
            if not self.rulebook.union_bursts_for(hand_card.card_id):
                return
            before = hand_card.union_burst_gauge(
                self.players[owner].turns_started
            )
            hand_card.union_burst_gauge_bonus += effect.amount
            after = hand_card.union_burst_gauge(
                self.players[owner].turns_started
            )
            self._emit(
                GameEvent(
                    EventType.UNION_BURST_GAUGE_CHANGED,
                    owner,
                    source_id=frame.source_entity_id,
                    target_id=hand_card.entity_id,
                    amount=effect.amount,
                    metadata={
                        "card_id": hand_card.card_id,
                        "gauge_before": before,
                        "gauge_after": after,
                        "source_card_id": frame.source_card_id,
                    },
                )
            )
            self._log(
                frame.controller,
                f"手牌 {hand_card.name} 的奥义计量 +{effect.amount}",
            )
        elif effect.kind is EffectKind.DESTROY:
            target = self._find_board_entity(target_id)
            target_snapshot = (
                self._bound_target_snapshot(target.entity_id)
                if effect.bind_successful_targets
                else None
            )
            if isinstance(target, Unit):
                if not self._attempt_effect_destroy_unit(
                    target,
                    controller=frame.controller,
                    source_entity_id=frame.source_entity_id,
                    source_card_id=frame.source_card_id,
                ):
                    return
            elif isinstance(target, Amulet):
                if self._is_earth_sigil_amulet(target):
                    self._emit(
                        GameEvent(
                            EventType.EARTH_SIGIL_DESTROY_PREVENTED,
                            frame.controller,
                            source_id=frame.source_entity_id,
                            target_id=target.entity_id,
                            metadata={"source_card_id": frame.source_card_id},
                        )
                    )
                    self._log(
                        frame.controller,
                        f"{target.definition.name} 的土之印规则阻止了效果破坏",
                    )
                    return
                target.pending_destroy = True
            if effect.bind_successful_targets and target_snapshot is not None:
                previous_ids = frame._target_bindings.get(
                    effect.target_key,
                    (),
                )
                previous_snapshots = frame._target_binding_snapshots.get(
                    effect.target_key,
                    (),
                )
                self._bind_targets(
                    frame,
                    effect.target_key,
                    (*previous_ids, target.entity_id),
                    effect,
                    snapshots=(*previous_snapshots, target_snapshot),
                )
        elif effect.kind is EffectKind.SUMMON:
            self._execute_summon(effect, frame)
        elif effect.kind is EffectKind.SUMMON_COPY:
            self._execute_summon_copy(effect, frame, target_id)
        elif effect.kind is EffectKind.SUMMON_EXACT_COPY:
            self._execute_summon_exact_copy(effect, frame, target_id)
        elif effect.kind is EffectKind.SUMMON_HAND_COPY:
            self._execute_summon_hand_copy(effect, frame, target_id)
        elif effect.kind is EffectKind.SUMMON_FROM_HAND:
            self._execute_summon_from_hand(effect, frame, target_id)
        elif effect.kind is EffectKind.SUMMON_FROM_DECK:
            self._execute_summon_from_deck(effect, frame)
        elif effect.kind is EffectKind.SUMMON_DESTROYED_AMULETS:
            self._execute_summon_destroyed_amulets(effect, frame)
        elif effect.kind is EffectKind.BANISH:
            self._execute_banish(target_id, frame)
        elif effect.kind is EffectKind.BANISH_SAME_NAME:
            self._execute_banish_same_name(effect, frame, target_id)
        elif effect.kind is EffectKind.ADD_CARD:
            self._execute_add_card(effect, frame)
        elif effect.kind is EffectKind.ADD_CARD_TO_DECK:
            self._execute_add_card_to_deck(effect, frame)
        elif effect.kind is EffectKind.COPY_TO_HAND:
            self._execute_copy_to_hand(effect, frame, target_id)
        elif effect.kind is EffectKind.COPY_LEFTMOST_HAND_TO_HAND:
            self._execute_copy_leftmost_hand_to_hand(effect, frame)
        elif effect.kind is EffectKind.COPY_RANDOM_ENEMY_DECK_TO_HAND:
            self._execute_copy_random_enemy_deck_to_hand(effect, frame)
        elif effect.kind is EffectKind.COPY_DESTROYED_FOLLOWERS_TO_HAND:
            self._execute_copy_destroyed_followers_to_hand(effect, frame)
        elif effect.kind is EffectKind.REDRAW_HAND:
            self._execute_redraw_hand(frame)
        elif effect.kind is EffectKind.RETURN_TO_HAND:
            self._execute_return_to_hand(target_id, frame)
        elif effect.kind is EffectKind.RETURN_TO_DECK:
            self._execute_return_to_deck(target_id, frame)
        elif effect.kind is EffectKind.REDUCE_COUNTDOWN:
            self._execute_reduce_countdown(effect, frame, target_id)
        elif effect.kind is EffectKind.INCREASE_COUNTDOWN:
            self._execute_increase_countdown(effect, frame, target_id)
        elif effect.kind is EffectKind.DISCARD:
            self._execute_discard(target_id, frame)
        elif effect.kind is EffectKind.ADD_KEYWORD:
            self._execute_keyword_change(effect, frame, target_id, add=True)
        elif effect.kind is EffectKind.ADD_RANDOM_KEYWORDS:
            self._execute_add_random_keywords(effect, frame, target_id)
        elif effect.kind is EffectKind.GRANT_LAST_WORDS:
            self._execute_grant_last_words(effect, frame, target_id)
        elif effect.kind is EffectKind.GRANT_EFFECT_DESTROY_IMMUNITY:
            self._execute_grant_effect_destroy_immunity(frame, target_id)
        elif effect.kind is EffectKind.REMOVE_KEYWORD:
            self._execute_keyword_change(effect, frame, target_id, add=False)
        elif effect.kind is EffectKind.REMOVE_ALL_ABILITIES:
            self._execute_remove_all_abilities(frame, target_id)
        elif effect.kind is EffectKind.REMOVE_LAST_WORDS:
            self._execute_remove_last_words(frame, target_id)
        elif effect.kind is EffectKind.GRANT_ATTACKS_PER_TURN:
            self._execute_grant_attacks_per_turn(effect, frame, target_id)
        elif effect.kind is EffectKind.GRANT_TURN_END_DESTROY:
            self._execute_grant_turn_end_destroy(effect, frame, target_id)
        elif effect.kind is EffectKind.GRANT_TURN_END_BANISH:
            self._execute_grant_turn_end_banish(effect, frame, target_id)
        elif effect.kind is EffectKind.GRANT_TURN_END_ABILITY:
            self._execute_grant_turn_end_ability(effect, frame, target_id)
        elif effect.kind is EffectKind.ADD_LEADER_BARRIER:
            self._execute_add_leader_barrier(effect, frame)
        elif effect.kind is EffectKind.ADD_LEADER_DAMAGE_MODIFIER:
            self._execute_add_leader_damage_modifier(effect, frame)
        elif effect.kind is EffectKind.CHANGE_COST:
            self._execute_change_cost(effect, frame, target_id)
        elif effect.kind is EffectKind.CHANGE_DECK_COST:
            self._execute_change_deck_cost(effect, frame)
        elif effect.kind is EffectKind.BUFF_DECK_CARDS:
            self._execute_buff_deck_cards(effect, frame)
        elif effect.kind is EffectKind.REPLACE_DECK:
            self._execute_replace_deck(effect, frame)
        elif effect.kind is EffectKind.BANISH_DECK_DUPLICATES:
            self._execute_banish_deck_duplicates(frame)
        elif effect.kind is EffectKind.BANISH_DECK_FILTERED:
            self._execute_banish_deck_filtered(effect, frame)
        elif effect.kind is EffectKind.SET_EMPTY_DECK_OUTCOME:
            self._execute_set_empty_deck_outcome(effect, frame)
        elif effect.kind is EffectKind.TRANSFORM:
            self._execute_transform(effect, frame, target_id)
        elif effect.kind is EffectKind.TRANSFORM_BOARD_FROM_RANDOM_OWN_DECK:
            self._execute_transform_board_from_random_own_deck(
                effect,
                frame,
                target_id,
            )
        elif effect.kind is EffectKind.TRANSFORM_DECK_CARDS:
            self._execute_transform_deck_cards(effect, frame)
        elif effect.kind is EffectKind.TRANSFORM_HAND_FROM_RANDOM_ENEMY_DECK:
            self._execute_transform_hand_from_random_enemy_deck(
                effect,
                frame,
                target_id,
            )
        elif effect.kind is EffectKind.SET_STATS:
            self._execute_set_stats(effect, frame, target_id)
        elif effect.kind is EffectKind.EVOLVE_UNIT:
            self._execute_evolve_unit(frame, target_id)
        elif effect.kind is EffectKind.SUPER_EVOLVE_UNIT:
            self._execute_super_evolve_unit(frame, target_id)
        elif effect.kind is EffectKind.ADD_ATTACK_RESTRICTION:
            self._execute_attack_restriction(effect, frame, target_id, add=True)
        elif effect.kind is EffectKind.REMOVE_ATTACK_RESTRICTION:
            self._execute_attack_restriction(effect, frame, target_id, add=False)
        elif effect.kind is EffectKind.ADD_TARGETING_RESTRICTION:
            self._execute_targeting_restriction(effect, frame, target_id, add=True)
        elif effect.kind is EffectKind.REMOVE_TARGETING_RESTRICTION:
            self._execute_targeting_restriction(effect, frame, target_id, add=False)
        elif effect.kind is EffectKind.SPELLBOOST_HAND:
            self._execute_spellboost_hand(effect, frame, target_id)
        elif effect.kind is EffectKind.ADD_COMBO:
            self._execute_add_combo(effect, frame)
        elif effect.kind is EffectKind.ADD_SHADOWS:
            self._execute_add_shadows(effect, frame)
        elif effect.kind is EffectKind.ADD_EARTH_SIGILS:
            self._execute_add_earth_sigils(effect, frame)
        elif effect.kind is EffectKind.EARTH_RITE:
            self._execute_earth_rite(effect, frame)
        elif effect.kind is EffectKind.CONSUME_FAITH:
            self._execute_consume_faith(effect, frame)
        elif effect.kind is EffectKind.GRANT_FAITH_ABILITY:
            self._execute_grant_faith_ability(effect, frame)
        elif effect.kind is EffectKind.GRANT_FAITH_MODE_SELECTION_BONUS:
            self._execute_grant_faith_mode_selection_bonus(effect, frame)
        elif effect.kind is EffectKind.RANDOM_CHOICE:
            self._execute_random_choice(effect, frame)
        elif effect.kind is EffectKind.RANDOM_DISTRIBUTE:
            self._execute_random_distribute(effect, frame)
        elif effect.kind is EffectKind.NECROMANCY:
            self._execute_necromancy(effect, frame, target_id)
        elif effect.kind is EffectKind.REANIMATE:
            self._execute_reanimate(effect, frame, target_id)
        elif effect.kind is EffectKind.SUMMON_FROM_GRAVEYARD:
            self._execute_summon_from_graveyard(effect, frame, target_id)
        elif effect.kind is EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND:
            self._execute_return_from_graveyard_to_hand(effect, frame, target_id)
        elif effect.kind is EffectKind.BANISH_FROM_GRAVEYARD:
            self._execute_banish_from_graveyard(effect, frame, target_id)
        elif effect.kind in (EffectKind.GAIN_EMBLEM, EffectKind.ADD_EMBLEM):
            self._execute_gain_emblem(effect, frame)
        elif effect.kind is EffectKind.REMOVE_EMBLEM:
            self._execute_remove_emblem(effect, frame)
        elif effect.kind is EffectKind.REMOVE_ALL_EMBLEMS:
            self._execute_remove_all_emblems(frame, target_id)
        elif effect.kind is EffectKind.CONDITIONAL:
            self._execute_conditional(effect, frame)
        elif effect.kind is EffectKind.CHOOSE_ONE:
            self._execute_choose_one(effect, frame)
        elif effect.kind is EffectKind.OPTIONAL:
            self._execute_optional(effect, frame)
        elif effect.kind is EffectKind.TARGET_EXISTS:
            self._execute_target_exists(effect, frame)
        elif effect.kind is EffectKind.REPEAT:
            self._execute_repeat(effect, frame)
        elif effect.kind is EffectKind.REPLAY_SOURCE_FANFARE:
            self._execute_replay_source_fanfare(frame, target_id)
        else:
            self._record_runtime_diagnostic(
                "unsupported",
                card_id=frame.source_card_id,
                clause_id=self._runtime_clause_id(effect, frame),
                detail=effect.kind.value,
            )
            self._log(
                frame.controller,
                f"[未实现效果] {name} {frame.label}: {effect.kind.value}",
            )

    def _execute_evolve_unit(
        self,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        try:
            target = self._find_board_entity(target_id)
        except IllegalCommand:
            return
        if not isinstance(target, Unit) or target.evolved:
            return
        owner = self._entity_owner(target.entity_id)
        if not self._apply_evolution_state(
            target,
            owner,
            super_evolve=False,
            cause="effect",
            trigger_abilities=False,
        ):
            return
        self._log(
            frame.controller,
            f"{frame.source_name} {frame.label}使 "
            f"{target.definition.name} 进化，变为 "
            f"{target.attack}/{target.health}",
        )

    def _execute_super_evolve_unit(
        self,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        try:
            target = self._find_board_entity(target_id)
        except IllegalCommand:
            return
        if not isinstance(target, Unit) or target.evolved:
            return
        owner = self._entity_owner(target.entity_id)
        if not self._apply_evolution_state(
            target,
            owner,
            super_evolve=True,
            cause="effect",
            trigger_abilities=False,
        ):
            return
        self._log(
            frame.controller,
            f"{frame.source_name} {frame.label}使 "
            f"{target.definition.name} 超进化，变为 "
            f"{target.attack}/{target.health}",
        )

    def _execute_add_combo(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.amount < 0:
            raise IllegalCommand("add_combo amount must be non-negative")
        player_index = (
            1 - frame.controller
            if effect.target is TargetKind.ENEMY_LEADER
            else frame.controller
        )
        self._record_combo(
            player_index,
            effect.amount,
            source_card_id=frame.source_card_id,
            source_entity_id=frame.source_entity_id,
            cause="effect",
        )
        if effect.amount:
            owner_name = "对方" if player_index != frame.controller else "自己"
            self._log(
                frame.controller,
                f"{frame.source_name} {frame.label}使{owner_name}的连击 +{effect.amount}",
            )

    def _execute_add_shadows(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.amount < 0:
            raise IllegalCommand("add_shadows amount must be non-negative")
        if effect.amount == 0:
            return
        player_index = (
            1 - frame.controller
            if effect.target is TargetKind.ENEMY_LEADER
            else frame.controller
        )
        player = self.players[player_index]
        before = player.shadows
        player.add_shadows(effect.amount)
        self._emit(
            GameEvent(
                EventType.SHADOWS_CHANGED,
                player_index,
                source_id=frame.source_entity_id,
                amount=effect.amount,
                metadata={
                    "change": "gain",
                    "shadows_before": before,
                    "shadows_after": player.shadows,
                    "source_card_id": frame.source_card_id,
                    "target_player": player_index,
                },
            )
        )
        owner_name = "对方" if player_index != frame.controller else "自己"
        self._log(
            frame.controller,
            f"{frame.source_name} {frame.label}使{owner_name}的墓场 +{effect.amount}",
        )

    def _execute_keyword_change(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
        *,
        add: bool,
    ) -> None:
        if effect.keyword is None:
            raise IllegalCommand(f"{effect.kind.value} requires a keyword")
        resolved_id = (
            (
                frame.listener_activation_entity_id
                if frame.listener_activation_zone == ListenerZone.HAND.value
                else frame.source_entity_id
            )
            if effect.target is TargetKind.SELF
            else target_id
        )
        hand_target: HandCard | None = None
        hand_target_required = (
            effect.target in {
                TargetKind.OWN_HAND,
                TargetKind.RANDOM_OWN_HAND,
                TargetKind.RANDOM_ENEMY_HAND,
                TargetKind.ALL_OWN_HAND,
                TargetKind.ALL_ENEMY_HAND,
            }
            or (
                effect.target is TargetKind.SELF
                and frame.listener_activation_zone == ListenerZone.HAND.value
            )
        )
        if hand_target_required or effect.target is TargetKind.PREVIOUS_TARGET:
            try:
                _, hand_target = self._find_hand_card_with_owner(resolved_id)
            except IllegalCommand:
                if hand_target_required:
                    raise
        expires_for_player = _expires_for_player(
            effect.duration,
            frame.controller,
            self.state.active_player,
        )
        if hand_target is not None:
            if hand_target.card_type != "随从":
                raise IllegalCommand("Keyword target must be a follower")
            if add:
                hand_target.add_keyword(
                    effect.keyword,
                    duration=effect.duration.value,
                    expires_for_player=expires_for_player,
                )
                verb = "获得"
            else:
                hand_target.remove_keyword(
                    effect.keyword,
                    duration=effect.duration.value,
                    expires_for_player=expires_for_player,
                )
                verb = "失去"
            self._log(
                frame.controller,
                f"手牌 {hand_target.name} {verb}关键词 {effect.keyword}",
            )
            return
        target = self._find_board_entity(resolved_id)
        if not isinstance(target, Unit):
            raise IllegalCommand("Keyword target must be a follower")
        if add:
            target.add_keyword(
                effect.keyword,
                duration=effect.duration.value,
                expires_for_player=expires_for_player,
            )
            verb = "获得"
        else:
            target.remove_keyword(
                effect.keyword,
                duration=effect.duration.value,
                expires_for_player=expires_for_player,
            )
            verb = "失去"
        self._log(
            frame.controller,
            f"{target.definition.name} {verb}关键词 {effect.keyword}",
        )

    def _execute_add_random_keywords(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        clause_id = self._runtime_clause_id(effect, frame)
        if self.runtime_coverage is not None and clause_id is not None:
            self.runtime_coverage.record_target(
                clause_id,
                "random_keyword",
                candidate_count=len(effect.keywords),
                random=True,
                no_target=not effect.keywords,
            )
        if not effect.keywords or effect.amount > len(effect.keywords):
            raise IllegalCommand(
                "add_random_keywords requires enough keyword candidates"
            )
        selected = self.random.sample(effect.keywords, effect.amount)
        for keyword in selected:
            self._execute_keyword_change(
                replace(
                    effect,
                    kind=EffectKind.ADD_KEYWORD,
                    keyword=keyword,
                    keywords=(),
                ),
                frame,
                target_id,
                add=True,
            )

    def _grantable_follower_target(
        self,
        target_id: int | None,
    ) -> HandCard | Unit | None:
        if target_id is None:
            return None
        try:
            _, hand_card = self._find_hand_card_with_owner(target_id)
            return hand_card if hand_card.card_type == "随从" else None
        except IllegalCommand:
            pass
        try:
            target = self._find_board_entity(target_id)
        except IllegalCommand:
            return None
        return target if isinstance(target, Unit) else None

    def _execute_grant_last_words(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        target = self._grantable_follower_target(target_id)
        if target is None:
            return
        if not effect.granted_operations:
            raise IllegalCommand("grant_last_words requires operations")
        target.granted_last_words.append(effect.granted_operations)
        self._emit(
            GameEvent(
                EventType.CARD_ABILITY_GRANTED,
                frame.controller,
                source_id=frame.source_entity_id,
                target_id=target.entity_id,
                metadata={
                    "card_id": target.definition.card_id,
                    "ability": "last_words",
                    "zone": "hand" if isinstance(target, HandCard) else "board",
                    "source_card_id": frame.source_card_id,
                },
            )
        )
        self._log(
            frame.controller,
            f"{target.definition.name} 获得谢幕曲",
        )

    def _execute_grant_effect_destroy_immunity(
        self,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        target = self._grantable_follower_target(target_id)
        if target is None:
            return
        target.effect_destroy_immunity = True
        self._emit(
            GameEvent(
                EventType.CARD_ABILITY_GRANTED,
                frame.controller,
                source_id=frame.source_entity_id,
                target_id=target.entity_id,
                metadata={
                    "card_id": target.definition.card_id,
                    "ability": "cannot_be_destroyed_by_effects",
                    "zone": "hand" if isinstance(target, HandCard) else "board",
                    "source_card_id": frame.source_card_id,
                },
            )
        )
        self._log(frame.controller, f"{target.name} 获得能力破坏免疫")

    def _execute_remove_all_abilities(
        self, frame: EffectFrame, target_id: int | None
    ) -> None:
        target = self._find_board_entity(target_id)
        if not isinstance(target, Unit):
            raise IllegalCommand("remove_all_abilities target must be a follower")
        target.remove_all_abilities()
        self._emit(GameEvent(
            EventType.FOLLOWER_ABILITIES_REMOVED,
            frame.controller,
            source_id=frame.source_entity_id,
            target_id=target.entity_id,
            metadata={"card_id": target.definition.card_id},
        ))
        self._log(frame.controller, f"{target.definition.name} 失去所有能力")

    def _execute_remove_last_words(
        self, frame: EffectFrame, target_id: int | None
    ) -> None:
        target = self._find_board_entity(target_id)
        if not isinstance(target, Unit):
            raise IllegalCommand("remove_last_words target must be a follower")
        target.remove_last_words()
        self._emit(GameEvent(
            EventType.FOLLOWER_LAST_WORDS_REMOVED,
            frame.controller,
            source_id=frame.source_entity_id,
            target_id=target.entity_id,
            metadata={"card_id": target.definition.card_id},
        ))
        self._log(frame.controller, f"{target.definition.name} 失去谢幕曲")

    def _execute_add_leader_damage_modifier(
        self, effect: EffectOperation, frame: EffectFrame
    ) -> None:
        player_index = (
            frame.controller
            if effect.target is TargetKind.OWN_LEADER
            else 1 - frame.controller
        )
        source_bound = effect.duration is ModifierDuration.WHILE_SOURCE_IN_PLAY
        modifier = LeaderDamageModifier(
            modifier_id=self._allocate_modifier_id(),
            amount=effect.amount,
            duration=effect.duration.value,
            expires_for_player=_expires_for_player(
                effect.duration, frame.controller, self.state.active_player
            ),
            source_controller=frame.controller if source_bound else None,
            source_entity_id=frame.source_entity_id if source_bound else None,
            source_card_id=frame.source_card_id if source_bound else None,
            mode=effect.leader_damage_mode.value,
        )
        self.players[player_index].leader_damage_modifiers.append(modifier)
        self._emit(GameEvent(
            EventType.LEADER_DAMAGE_MODIFIER_ADDED,
            frame.controller,
            source_id=frame.source_entity_id,
            amount=effect.amount,
            metadata={
                "target_player": player_index,
                "duration": effect.duration.value,
                "modifier_id": modifier.modifier_id,
                "damage_mode": effect.leader_damage_mode.value,
            },
        ))

    def _execute_add_leader_barrier(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        player_index = (
            frame.controller
            if effect.target is TargetKind.OWN_LEADER
            else 1 - frame.controller
        )
        player = self.players[player_index]
        player.leader_barrier_charges += effect.amount
        self._emit(GameEvent(
            EventType.LEADER_BARRIER_GRANTED,
            frame.controller,
            source_id=frame.source_entity_id,
            amount=effect.amount,
            metadata={
                "target_player": player_index,
                "charges": player.leader_barrier_charges,
            },
        ))

    def _execute_replay_source_fanfare(
        self,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        source = self._find_board_entity(target_id)
        if not isinstance(source, Unit) or source.entity_id != frame.source_entity_id:
            return
        operations = self.rulebook.operations_for(
            source.definition.card_id,
            Trigger.FANFARE,
        )
        if not operations:
            return
        self._queue_effects_from_frame(
            frame,
            operations,
            label=f"{frame.label}/replay_fanfare",
        )

    def _execute_grant_attacks_per_turn(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        target = self._find_board_entity(target_id)
        if not isinstance(target, Unit):
            raise IllegalCommand(
                "grant_attacks_per_turn target must be a follower"
            )
        before = target.attacks_per_turn
        target.grant_attacks_per_turn(
            effect.amount,
            duration=effect.duration.value,
            expires_for_player=_expires_for_player(
                effect.duration,
                frame.controller,
                self.state.active_player,
            ),
        )
        self._emit(GameEvent(
            EventType.FOLLOWER_ATTACK_CAPACITY_GRANTED,
            frame.controller,
            source_id=frame.source_entity_id,
            target_id=target.entity_id,
            amount=target.attacks_per_turn,
            metadata={
                "before": before,
                "duration": effect.duration.value,
            },
        ))
        self._log(
            frame.controller,
            f"{target.definition.name} 每回合可攻击 {target.attacks_per_turn} 次",
        )

    def _execute_grant_turn_end_destroy(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        if effect.turn_end_destroy_timing is None:
            raise IllegalCommand(
                "GRANT_TURN_END_DESTROY requires turn_end_destroy_timing"
            )
        target = self._find_board_entity(target_id)
        if not isinstance(target, Unit):
            raise IllegalCommand(
                "grant_turn_end_destroy target must be a follower"
            )
        if effect.turn_end_destroy_timing in target.turn_end_destroy_timings:
            return
        target.turn_end_destroy_timings.add(effect.turn_end_destroy_timing)
        self._emit(
            GameEvent(
                EventType.FOLLOWER_TURN_END_DESTROY_GRANTED,
                self._entity_owner(target.entity_id),
                source_id=frame.source_entity_id,
                target_id=target.entity_id,
                metadata={
                    "source_card_id": frame.source_card_id,
                    "target_card_id": target.definition.card_id,
                    "timing": effect.turn_end_destroy_timing.value,
                },
            )
        )
        timing_label = (
            "其控制者回合结束"
            if effect.turn_end_destroy_timing is TurnEndDestroyTiming.OWNER_TURN
            else "其控制者的对手回合结束"
        )
        self._log(
            frame.controller,
            f"{target.definition.name} 获得“{timing_label}时破坏自身”",
        )

    def _execute_grant_turn_end_banish(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        if effect.turn_end_banish_timing is None:
            raise IllegalCommand(
                "GRANT_TURN_END_BANISH requires turn_end_banish_timing"
            )
        target = self._find_board_entity(target_id)
        if not isinstance(target, Unit):
            raise IllegalCommand(
                "grant_turn_end_banish target must be a follower"
            )
        if effect.turn_end_banish_timing in target.turn_end_banish_timings:
            return
        target.turn_end_banish_timings.add(effect.turn_end_banish_timing)
        self._emit(
            GameEvent(
                EventType.FOLLOWER_TURN_END_BANISH_GRANTED,
                self._entity_owner(target.entity_id),
                source_id=frame.source_entity_id,
                target_id=target.entity_id,
                metadata={
                    "source_card_id": frame.source_card_id,
                    "target_card_id": target.definition.card_id,
                    "timing": effect.turn_end_banish_timing.value,
                },
            )
        )
        timing_label = (
            "其控制者回合结束"
            if effect.turn_end_banish_timing is TurnEndDestroyTiming.OWNER_TURN
            else "其控制者的对手回合结束"
        )
        self._log(
            frame.controller,
            f"{target.definition.name} 获得“{timing_label}时消失自身”",
        )

    def _execute_grant_turn_end_ability(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        if (
            effect.turn_end_ability_timing is None
            or not effect.granted_operations
        ):
            raise IllegalCommand(
                "GRANT_TURN_END_ABILITY requires timing and operations"
            )
        target = self._find_board_entity(target_id)
        if not isinstance(target, Unit):
            raise IllegalCommand(
                "grant_turn_end_ability target must be a follower"
            )
        granted = GrantedTurnEndAbility(
            timing=effect.turn_end_ability_timing,
            operations=effect.granted_operations,
        )
        target.granted_turn_end_abilities.append(granted)
        owner = self._entity_owner(target.entity_id)
        self._emit(
            GameEvent(
                EventType.FOLLOWER_TURN_END_ABILITY_GRANTED,
                owner,
                source_id=frame.source_entity_id,
                target_id=target.entity_id,
                metadata={
                    "source_card_id": frame.source_card_id,
                    "target_card_id": target.definition.card_id,
                    "timing": granted.timing.value,
                    "operation_count": len(granted.operations),
                },
            )
        )
        self._log(
            frame.controller,
            f"{target.definition.name} 获得回合结束能力",
        )

    def _execute_change_cost(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        if effect.mode is None:
            raise IllegalCommand("CHANGE_COST requires a mode")
        hand_owner = (
            1 - frame.controller
            if effect.target is TargetKind.ALL_ENEMY_HAND
            else frame.controller
        )
        hand_card = self._find_hand_card(hand_owner, target_id)
        before = hand_card.current_cost
        hand_card.cost_modifiers.append(
            CostModifier(
                modifier_id=self._allocate_modifier_id(),
                mode=effect.mode.value,
                amount=effect.amount,
                duration=effect.duration.value,
                expires_for_player=_expires_for_player(
                    effect.duration,
                    frame.controller,
                    self.state.active_player,
                ),
            )
        )
        self._log(
            frame.controller,
            f"{hand_card.name} 费用由 {before} 变为 {hand_card.current_cost}",
        )

    def _execute_change_deck_cost(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.mode is None:
            raise IllegalCommand("CHANGE_DECK_COST requires a mode")
        if effect.deck_filter is None:
            raise IllegalCommand("CHANGE_DECK_COST requires a deck filter")
        player = self.players[frame.controller]
        changed = 0
        for index, raw_card in enumerate(list(player.deck)):
            definition = (
                raw_card.definition
                if isinstance(raw_card, DeckCard)
                else raw_card
            )
            if not effect.deck_filter.matches(definition):
                continue
            deck_card = (
                raw_card
                if isinstance(raw_card, DeckCard)
                else DeckCard(definition=definition)
            )
            before = deck_card.current_cost
            deck_card.cost_modifiers.append(CostModifier(
                modifier_id=self._allocate_modifier_id(),
                mode=effect.mode.value,
                amount=effect.amount,
                duration=ModifierDuration.PERMANENT.value,
                expires_for_player=None,
            ))
            player.deck[index] = deck_card
            changed += 1
            self._emit(GameEvent(
                EventType.DECK_CARD_COST_CHANGED,
                frame.controller,
                source_id=frame.source_entity_id,
                amount=deck_card.current_cost - before,
                metadata={
                    "source_card_id": frame.source_card_id,
                    "target_card_id": definition.card_id,
                    "before": before,
                    "after": deck_card.current_cost,
                    "mode": effect.mode.value,
                    "requested_amount": effect.amount,
                },
            ))
        self._log(
            frame.controller,
            f"{frame.source_name} 使牌组中 {changed} 张卡牌的费用发生变化",
        )

    def _execute_buff_deck_cards(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.deck_filter is None:
            raise IllegalCommand("BUFF_DECK_CARDS requires a deck filter")
        player = self.players[frame.controller]
        changed = 0
        for index, raw_card in enumerate(list(player.deck)):
            definition = (
                raw_card.definition
                if isinstance(raw_card, DeckCard)
                else raw_card
            )
            if not effect.deck_filter.matches(definition):
                continue
            deck_card = (
                raw_card
                if isinstance(raw_card, DeckCard)
                else DeckCard(definition=definition)
            )
            deck_card.stat_modifiers.append(
                StatModifier(
                    modifier_id=self._allocate_modifier_id(),
                    attack_delta=effect.amount,
                    health_delta=effect.secondary_amount,
                    duration=ModifierDuration.PERMANENT.value,
                    expires_for_player=None,
                )
            )
            player.deck[index] = deck_card
            changed += 1
        self._emit(
            GameEvent(
                EventType.DECK_FOLLOWERS_BUFFED,
                frame.controller,
                source_id=frame.source_entity_id,
                amount=changed,
                metadata={
                    "source_card_id": frame.source_card_id,
                    "attack_delta": effect.amount,
                    "health_delta": effect.secondary_amount,
                    "changed_count": changed,
                },
            )
        )
        self._log(
            frame.controller,
            f"{frame.source_name} 使牌组中 {changed} 张随从 "
            f"+{effect.amount}/+{effect.secondary_amount}",
        )

    def _execute_replace_deck(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if not effect.card_ids:
            raise IllegalCommand("REPLACE_DECK requires card_ids")
        if self.card_resolver is None:
            raise IllegalCommand("No card_resolver registered for REPLACE_DECK")
        definitions: list[CardDefinition] = []
        for card_id in effect.card_ids:
            definition = self.card_resolver(card_id)
            if definition is None:
                raise IllegalCommand(
                    f"REPLACE_DECK could not resolve card {card_id}"
                )
            definitions.append(definition)
        if effect.shuffle:
            self.random.shuffle(definitions)
        target_player_index = (
            1 - frame.controller
            if effect.target is TargetKind.ENEMY_LEADER
            else frame.controller
        )
        player = self.players[target_player_index]
        previous_count = len(player.deck)
        player.deck = definitions
        self._emit(GameEvent(
            EventType.DECK_REPLACED,
            target_player_index,
            source_id=frame.source_entity_id,
            amount=len(definitions),
            metadata={
                "source_card_id": frame.source_card_id,
                "previous_count": previous_count,
                "new_count": len(definitions),
                "card_ids": tuple(card.card_id for card in definitions),
                "shuffled": effect.shuffle,
            },
        ))
        self._log(
            frame.controller,
            f"{frame.source_name} 将牌组替换为 {len(definitions)} 张卡牌",
        )

    def _execute_banish_deck_duplicates(
        self,
        frame: EffectFrame,
    ) -> None:
        player = self.players[frame.controller]
        retained_card_ids: set[int] = set()
        retained_cards: list[CardDefinition | DeckCard] = []
        banished_definitions: list[CardDefinition] = []
        for raw_card in player.deck:
            definition = (
                raw_card.definition
                if isinstance(raw_card, DeckCard)
                else raw_card
            )
            if definition.card_id not in retained_card_ids:
                retained_card_ids.add(definition.card_id)
                retained_cards.append(raw_card)
                continue
            banished_definitions.append(definition)

        if not banished_definitions:
            return
        player.deck = retained_cards
        for definition in banished_definitions:
            player.banished.append(definition)
            self._emit(
                GameEvent(
                    EventType.CARD_BANISHED,
                    frame.controller,
                    source_id=frame.source_entity_id,
                    metadata={
                        "source_card_id": frame.source_card_id,
                        "card_id": definition.card_id,
                        "definition": definition,
                        "from_zone": "deck",
                    },
                )
            )
        self._emit(
            GameEvent(
                EventType.DECK_DUPLICATES_BANISHED,
                frame.controller,
                source_id=frame.source_entity_id,
                amount=len(banished_definitions),
                metadata={
                    "source_card_id": frame.source_card_id,
                    "banished_card_ids": tuple(
                        definition.card_id
                        for definition in banished_definitions
                    ),
                    "remaining_count": len(player.deck),
                },
            )
        )
        self._log(
            frame.controller,
            f"{frame.source_name} 使牌组中 {len(banished_definitions)} "
            "张重复卡牌消失",
        )

    def _execute_banish_deck_filtered(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.deck_filter is None:
            raise IllegalCommand("BANISH_DECK_FILTERED requires a deck filter")

        player = self.players[frame.controller]
        retained_cards: list[CardDefinition | DeckCard] = []
        banished_cards: list[tuple[CardDefinition, int]] = []
        for raw_card in player.deck:
            if not effect.deck_filter.matches(raw_card):
                retained_cards.append(raw_card)
                continue
            definition = (
                raw_card.definition
                if isinstance(raw_card, DeckCard)
                else raw_card
            )
            current_cost = (
                raw_card.current_cost
                if isinstance(raw_card, DeckCard)
                else raw_card.cost
            )
            banished_cards.append((definition, current_cost))

        player.deck = retained_cards
        for definition, current_cost in banished_cards:
            player.banished.append(definition)
            self._emit(
                GameEvent(
                    EventType.CARD_BANISHED,
                    frame.controller,
                    source_id=frame.source_entity_id,
                    metadata={
                        "source_card_id": frame.source_card_id,
                        "card_id": definition.card_id,
                        "definition": definition,
                        "from_zone": "deck",
                        "current_cost": current_cost,
                    },
                )
            )
        count = len(banished_cards)
        self._log(
            frame.controller,
            f"{frame.source_name} 使牌组中 {count} 张符合条件的卡牌消失",
        )
        if effect.then_operations:
            self._queue_effects_from_frame(
                frame,
                effect.then_operations,
                label=f"{frame.label}/banished×{count}",
                distributed_value=count,
            )

    def _execute_set_empty_deck_outcome(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.empty_deck_outcome is None:
            raise IllegalCommand(
                "SET_EMPTY_DECK_OUTCOME requires empty_deck_outcome"
            )
        target_player_index = (
            1 - frame.controller
            if effect.target is TargetKind.ENEMY_LEADER
            else frame.controller
        )
        player = self.players[target_player_index]
        previous = player.empty_deck_outcome
        player.empty_deck_outcome = effect.empty_deck_outcome
        self._emit(GameEvent(
            EventType.EMPTY_DECK_OUTCOME_CHANGED,
            target_player_index,
            source_id=frame.source_entity_id,
            metadata={
                "source_card_id": frame.source_card_id,
                "previous": previous.value,
                "current": player.empty_deck_outcome.value,
            },
        ))
        self._log(
            frame.controller,
            f"{frame.source_name} 将空牌组抽牌结果设为"
            f"{player.empty_deck_outcome.value}",
        )

    def _execute_copy_to_hand(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        target = None
        if target_id is not None:
            try:
                target = self._find_board_entity(target_id)
            except IllegalCommand:
                target = next(
                    (
                        card
                        for player_index in (0, 1)
                        for card in self._hand_cards(player_index)
                        if card.entity_id == target_id
                    ),
                    None,
                )
        snapshot = self._bound_snapshot_for_effect(
            effect,
            frame,
            target_id,
        )
        if target is not None:
            card_def = target.definition
        elif snapshot is not None:
            card_def = snapshot.definition
        else:
            return
        player = self.players[frame.controller]
        origin = origin_for_added_card(card_def)
        if len(player.hand) >= self.config.max_hand:
            self._log(
                frame.controller,
                f"{frame.source_name} 复制卡牌失败：手牌已满",
            )
            self._send_to_graveyard(
                frame.controller,
                card_def,
                "hand_full",
                derived=True,
                origin=origin,
                token=is_token_definition(card_def),
            )
            return
        copied = self._append_hand_card(player, card_def, origin=origin)
        if effect.mode is not None:
            copied.cost_modifiers.append(
                CostModifier(
                    modifier_id=self._allocate_modifier_id(),
                    mode=effect.mode.value,
                    amount=effect.amount,
                    duration=effect.duration.value,
                    expires_for_player=_expires_for_player(
                        effect.duration,
                        frame.controller,
                        self.state.active_player,
                    ),
                )
            )
        self._log(
            frame.controller,
            f"{frame.source_name} 将1张复制卡以非公开形式加入手牌",
        )
        self._emit(
            GameEvent(
                EventType.CARD_ADDED_TO_HAND,
                frame.controller,
                source_id=copied.entity_id,
                metadata={
                    "card_id": card_def.card_id,
                    "card": card_def,
                    "source": copied,
                    "origin": origin.value,
                    "derived": True,
                    "token": is_token_definition(card_def),
                    "copied_from_entity_id": target_id,
                    "revealed": False,
                    "cost_after": copied.current_cost,
                },
            )
        )

    def _execute_copy_leftmost_hand_to_hand(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        player = self.players[frame.controller]
        sources = tuple(self._hand_cards(frame.controller)[:effect.amount])
        for source in sources:
            origin = origin_for_added_card(source.definition)
            if len(player.hand) >= self.config.max_hand:
                self._send_to_graveyard(
                    frame.controller,
                    source.definition,
                    "hand_full",
                    derived=True,
                    origin=origin,
                    token=(
                        is_token_definition(source.definition)
                        or origin is CardOrigin.TOKEN
                    ),
                )
                self._log(
                    frame.controller,
                    f"{frame.source_name} 完全相同复制卡牌失败：手牌已满",
                )
                continue

            copied = self._make_hand_card(
                source.definition,
                self.state.allocate_entity_id(),
                origin=origin,
                source_origin=source.source_origin or source.origin,
                fused_material_ids=tuple(source.fused_material_ids),
            )
            copied.cost_modifiers = [
                replace(
                    modifier,
                    modifier_id=self._allocate_modifier_id(),
                )
                for modifier in source.cost_modifiers
            ]
            copied.stat_modifiers = [
                replace(
                    modifier,
                    modifier_id=self._allocate_modifier_id(),
                )
                for modifier in source.stat_modifiers
            ]
            copied.spellboost_count = source.spellboost_count
            copied.spellboost_cost_reduction = source.spellboost_cost_reduction
            copied.cannot_be_played = source.cannot_be_played
            copied.fusion_used_turn = source.fusion_used_turn
            copied.evolutions_while_in_hand = source.evolutions_while_in_hand
            copied.union_burst_gauge_bonus = source.union_burst_gauge_bonus
            copied.printed_keyword_overrides = set(
                source.printed_keyword_overrides
            )
            copied.permanent_keywords = set(source.permanent_keywords)
            copied.temporary_keywords = list(source.temporary_keywords)
            copied.removed_keywords = set(source.removed_keywords)
            copied.temporary_keyword_removals = list(
                source.temporary_keyword_removals
            )
            copied.granted_last_words = list(source.granted_last_words)
            copied.effect_destroy_immunity = source.effect_destroy_immunity
            player.hand.append(copied)
            player.hand_entity_ids.append(copied.entity_id)
            self._emit(
                GameEvent(
                    EventType.CARD_ADDED_TO_HAND,
                    frame.controller,
                    source_id=copied.entity_id,
                    metadata={
                        "card_id": copied.card_id,
                        "card": copied.definition,
                        "source": copied,
                        "origin": copied.origin.value,
                        "derived": True,
                        "token": is_token_definition(copied.definition),
                        "copied_from_entity_id": source.entity_id,
                        "revealed": False,
                        "exact_copy": True,
                        "cost_after": copied.current_cost,
                    },
                )
            )
        if sources:
            self._log(
                frame.controller,
                f"{frame.source_name} 将最左侧 {len(sources)} 张手牌的"
                "完全相同复制品以非公开形式加入手牌",
            )

    def _execute_copy_random_enemy_deck_to_hand(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        enemy_deck = self.players[1 - frame.controller].deck
        if not enemy_deck:
            self._log(
                frame.controller,
                f"{frame.source_name} 复制失败：敌方牌组为空",
            )
            return

        selected = self.random.sample(
            enemy_deck,
            min(effect.amount, len(enemy_deck)),
        )
        player = self.players[frame.controller]
        copied_count = 0
        for raw_card in selected:
            definition = (
                raw_card.definition
                if isinstance(raw_card, DeckCard)
                else raw_card
            )
            origin = origin_for_added_card(definition)
            if len(player.hand) >= self.config.max_hand:
                self._send_to_graveyard(
                    frame.controller,
                    definition,
                    "hand_full",
                    derived=True,
                    origin=origin,
                    token=(
                        is_token_definition(definition)
                        or origin is CardOrigin.TOKEN
                    ),
                )
                self._log(
                    frame.controller,
                    f"{frame.source_name} 复制敌方牌组卡牌失败：手牌已满",
                )
                continue

            copied = self._append_hand_card(
                player,
                definition,
                origin=origin,
                source_origin=CardOrigin.DECK,
            )
            if isinstance(raw_card, DeckCard):
                copied.cost_modifiers = [
                    replace(
                        modifier,
                        modifier_id=self._allocate_modifier_id(),
                    )
                    for modifier in raw_card.cost_modifiers
                ]
            copied_count += 1
            self._emit(
                GameEvent(
                    EventType.CARD_ADDED_TO_HAND,
                    frame.controller,
                    source_id=copied.entity_id,
                    metadata={
                        "card_id": definition.card_id,
                        "card": definition,
                        "source": copied,
                        "origin": copied.origin.value,
                        "derived": True,
                        "token": (
                            is_token_definition(definition)
                            or copied.origin is CardOrigin.TOKEN
                        ),
                        "copied_from_zone": "enemy_deck",
                        "revealed": False,
                        "exact_copy": True,
                        "cost_after": copied.current_cost,
                    },
                )
            )
        self._log(
            frame.controller,
            f"{frame.source_name} 将敌方牌组中随机 {len(selected)} 张卡牌"
            f"的完全相同复制品以非公开形式加入手牌（成功 {copied_count} 张）",
        )

    def _execute_copy_destroyed_followers_to_hand(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        records = [
            record
            for record in sorted(
                self.state.destroyed_followers,
                key=lambda candidate: candidate.death_sequence,
            )
            if record.owner == frame.controller
            and (
                effect.history_filter is None
                or effect.history_filter.matches(record.definition)
            )
        ]
        if effect.distinct_card_names:
            distinct_records: dict[str, DestroyedFollowerRecord] = {}
            for record in records:
                distinct_records.setdefault(record.definition.name, record)
            records = list(distinct_records.values())
        if not records:
            return

        selected = self.random.sample(records, min(effect.amount, len(records)))
        player = self.players[frame.controller]
        added_count = 0
        for record in selected:
            card_def = record.definition
            origin = origin_for_added_card(card_def)
            if len(player.hand) >= self.config.max_hand:
                self._send_to_graveyard(
                    frame.controller,
                    card_def,
                    "hand_full",
                    derived=True,
                    origin=origin,
                    token=(
                        is_token_definition(card_def)
                        or origin is CardOrigin.TOKEN
                    ),
                )
                self._log(
                    frame.controller,
                    f"{frame.source_name} 复制已破坏随从失败：手牌已满",
                )
                continue
            copied = self._append_hand_card(player, card_def, origin=origin)
            added_count += 1
            self._emit(
                GameEvent(
                    EventType.CARD_ADDED_TO_HAND,
                    frame.controller,
                    source_id=copied.entity_id,
                    metadata={
                        "card_id": card_def.card_id,
                        "card": card_def,
                        "source": copied,
                        "origin": origin.value,
                        "derived": True,
                        "token": (
                            is_token_definition(card_def)
                            or origin is CardOrigin.TOKEN
                        ),
                        "copied_from_death_sequence": record.death_sequence,
                        "revealed": False,
                        "cost_after": copied.current_cost,
                    },
                )
            )
        if added_count:
            self._log(
                frame.controller,
                f"{frame.source_name} 将{added_count}张已破坏随从的同名卡"
                "以非公开形式加入手牌",
            )

    def _execute_summon_destroyed_amulets(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        player = self.players[frame.controller]
        available_slots = self.config.max_board - len(player.board)
        if available_slots <= 0:
            if effect.target_key:
                self._bind_targets(frame, effect.target_key, (), effect)
            self._log(
                frame.controller,
                f"{frame.source_name} 召唤已破坏护符失败：场地已满",
            )
            return

        records = [
            record
            for record in sorted(
                self.state.destroyed_amulets,
                key=lambda candidate: candidate.death_sequence,
            )
            if record.owner == frame.controller
            and (
                effect.history_filter is None
                or replace(
                    effect.history_filter,
                    card_type=None,
                ).matches(record.definition)
            )
        ]
        if effect.highest_base_cost_only and records:
            highest_cost = max(record.definition.cost for record in records)
            records = [
                record
                for record in records
                if record.definition.cost == highest_cost
            ]
        if effect.distinct_card_names:
            distinct_records: dict[str, DestroyedAmuletRecord] = {}
            for record in records:
                distinct_records.setdefault(record.definition.name, record)
            records = list(distinct_records.values())
        if not records:
            if effect.target_key:
                self._bind_targets(frame, effect.target_key, (), effect)
            return

        summon_count = min(effect.amount, available_slots, len(records))
        selected = self.random.sample(records, summon_count)
        summoned_entity_ids: list[int] = []
        for record in selected:
            definition = record.definition
            origin = origin_for_summoned_card(definition)
            amulet = Amulet(
                definition=definition,
                entity_id=self.state.allocate_entity_id(),
                countdown=record.summon_countdown,
                play_mode_id=record.play_mode_id,
                entered_turn=self.turn,
                origin=origin,
                source_origin=record.source_origin or record.origin,
            )
            player.board.append(amulet)
            summoned_entity_ids.append(amulet.entity_id)
            self._emit(
                GameEvent(
                    EventType.AMULET_ENTERED,
                    frame.controller,
                    source_id=amulet.entity_id,
                    metadata={
                        "source": amulet,
                        "card_id": definition.card_id,
                        "origin": amulet.origin.value,
                        "derived": True,
                        "token": (
                            is_token_definition(definition)
                            or amulet.origin is CardOrigin.TOKEN
                        ),
                        "via": "destroyed_amulet_history",
                        "copied_from_death_sequence": record.death_sequence,
                    },
                )
            )
            self._initialize_earth_sigil(amulet, frame.controller)
        if effect.target_key:
            self._bind_targets(
                frame,
                effect.target_key,
                tuple(summoned_entity_ids),
                effect,
            )
        self._log(
            frame.controller,
            f"{frame.source_name} 召唤 {len(summoned_entity_ids)} "
            "张已破坏护符的同名卡",
        )

    def _execute_redraw_hand(self, frame: EffectFrame) -> None:
        player = self.players[frame.controller]
        self._ensure_entity_ids()
        returned_cards = list(player.hand)
        returned_entity_ids = list(player.hand_entity_ids)
        if not returned_cards:
            return

        player.hand.clear()
        player.hand_entity_ids.clear()
        for raw_card, entity_id in zip(
            returned_cards,
            returned_entity_ids,
            strict=True,
        ):
            definition = (
                raw_card.definition
                if isinstance(raw_card, HandCard)
                else raw_card
            )
            insert_pos = self.random.randint(0, len(player.deck))
            player.deck.insert(insert_pos, definition)
            self._emit(
                GameEvent(
                    EventType.CARD_RETURNED_TO_DECK,
                    frame.controller,
                    source_id=entity_id,
                    metadata={
                        "source": raw_card,
                        "definition": definition,
                        "card_id": definition.card_id,
                        "from_zone": "hand",
                    },
                )
            )
        returned_count = len(returned_cards)
        self._log(
            frame.controller,
            f"{frame.source_name} 将 {returned_count} 张手牌返回牌组",
        )
        for _ in range(returned_count):
            self._draw(
                frame.controller,
                reason=f"{frame.source_name} 重抽",
            )
            if self.terminated:
                break

    @staticmethod
    def _bound_snapshot_for_effect(
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> BoundTargetSnapshot | None:
        if effect.target_key is None or target_id is None:
            return None
        return next(
            (
                snapshot
                for snapshot in frame._target_binding_snapshots.get(
                    effect.target_key,
                    (),
                )
                if snapshot.entity_id == target_id
            ),
            None,
        )

    def _execute_summon_copy(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        target = None
        if target_id is not None:
            try:
                target = self._find_board_entity(target_id)
            except IllegalCommand:
                target = None
        snapshot = self._bound_snapshot_for_effect(
            effect,
            frame,
            target_id,
        )
        definition = (
            target.definition
            if target is not None
            else (snapshot.definition if snapshot is not None else None)
        )
        if definition is None:
            return
        if definition.card_type != "随从":
            raise IllegalCommand("SUMMON_COPY requires a follower definition")
        player = self.players[frame.controller]
        if len(player.board) >= self.config.max_board:
            self._log(
                frame.controller,
                f"{frame.source_name} 复制召唤失败：场地已满",
            )
            return
        origin = origin_for_summoned_card(definition)
        unit = self._summon_follower_to_board(
            frame.controller,
            definition,
            summon_cause="copy_summon",
            origin=origin,
        )
        if unit is None:
            self._log(
                frame.controller,
                f"{frame.source_name} 复制召唤失败：场地已满",
            )
            return
        self._log(
            frame.controller,
            f"{frame.source_name} 召唤 {definition.name} 的复制随从 "
            f"({unit.attack}/{unit.health})",
        )
        self._emit(
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                frame.controller,
                source_id=unit.entity_id,
                metadata={
                    "source": unit,
                    "card_id": definition.card_id,
                    "origin": unit.origin.value,
                    "derived": True,
                    "token": is_token_definition(definition),
                    "via": "copy_summon",
                    "copied_from_entity_id": target_id,
                    "source_card_id": frame.source_card_id,
                },
            )
        )

    def _execute_summon_exact_copy(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        try:
            source = self._find_board_entity(target_id)
        except IllegalCommand:
            return
        if not isinstance(source, Unit):
            raise IllegalCommand(
                "SUMMON_EXACT_COPY requires a live follower target"
            )
        player = self.players[frame.controller]
        if len(player.board) >= self.config.max_board:
            self._log(
                frame.controller,
                f"{frame.source_name} 完全相同复制召唤失败：场地已满",
            )
            return

        unit = self._summon_follower_to_board(
            frame.controller,
            source.definition,
            summon_cause="exact_copy_summon",
            origin=origin_for_summoned_card(source.definition),
            source_origin=source.source_origin or source.origin,
        )
        if unit is None:
            return

        unit.base_attack = source.base_attack
        unit.base_health = source.base_health
        unit.attack = source.attack
        unit.health = source.health
        unit.max_health = source.max_health
        unit.evolved = source.evolved
        unit.super_evolved = source.super_evolved
        unit.super_evolved_turn = source.super_evolved_turn
        unit.permanent_keywords = set(source.permanent_keywords)
        unit.temporary_keywords = list(source.temporary_keywords)
        unit.removed_keywords = set(source.removed_keywords)
        unit.temporary_keyword_removals = list(
            source.temporary_keyword_removals
        )
        unit.stat_modifiers = [
            replace(
                modifier,
                modifier_id=self._allocate_modifier_id(),
            )
            for modifier in source.stat_modifiers
        ]
        unit.attack_capacity_modifiers = list(
            source.attack_capacity_modifiers
        )
        unit.attack_restrictions = list(source.attack_restrictions)
        unit.targeting_restrictions = list(source.targeting_restrictions)
        unit.printed_abilities_removed = source.printed_abilities_removed
        unit.last_words_removed = source.last_words_removed
        unit.turn_end_destroy_timings = set(source.turn_end_destroy_timings)
        unit.turn_end_banish_timings = set(source.turn_end_banish_timings)
        unit.granted_turn_end_abilities = list(
            source.granted_turn_end_abilities
        )
        unit.granted_last_words = list(source.granted_last_words)
        unit.random_choice_history = dict(source.random_choice_history)
        unit.effect_destroy_immunity = source.effect_destroy_immunity
        unit.barrier_charges = source.barrier_charges
        unit.ambush_active = source.ambush_active
        unit.attacks_remaining = unit.attacks_per_turn
        unit.summoned_this_turn = True
        unit._synchronize_keyword_state()
        unit.can_attack = (
            unit.attacks_remaining > 0
            and (
                unit.has_keyword("疾驰")
                or unit.has_keyword("突进")
            )
        )
        unit.rush_only = (
            unit.can_attack
            and unit.has_keyword("突进")
            and not unit.has_keyword("疾驰")
        )

        if effect.amount or effect.secondary_amount:
            unit.add_stat_modifier(StatModifier(
                modifier_id=self._allocate_modifier_id(),
                attack_delta=effect.amount,
                health_delta=effect.secondary_amount,
                duration=effect.duration.value,
                expires_for_player=_expires_for_player(
                    effect.duration,
                    frame.controller,
                    self.state.active_player,
                ),
            ))

        if effect.target_key:
            previous_outputs = frame._target_bindings.get(
                effect.target_key,
                (),
            )
            self._bind_targets(
                frame,
                effect.target_key,
                (*previous_outputs, unit.entity_id),
                effect,
            )
        self._log(
            frame.controller,
            f"{frame.source_name} 召唤 {source.definition.name} 的完全相同复制品 "
            f"({unit.attack}/{unit.health})",
        )
        self._emit(GameEvent(
            EventType.FOLLOWER_SUMMONED,
            frame.controller,
            source_id=unit.entity_id,
            metadata={
                "source": unit,
                "card_id": unit.definition.card_id,
                "origin": unit.origin.value,
                "derived": True,
                "token": is_token_definition(unit.definition),
                "via": "exact_copy_summon",
                "copied_from_entity_id": source.entity_id,
                "source_card_id": frame.source_card_id,
            },
        ))

    def _execute_summon_hand_copy(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        hand_card = next(
            (
                card
                for card in self._hand_cards(frame.controller)
                if card.entity_id == target_id
            ),
            None,
        )
        if hand_card is None:
            return
        if hand_card.card_type != "随从":
            raise IllegalCommand("SUMMON_HAND_COPY requires a follower in hand")
        player = self.players[frame.controller]
        if len(player.board) >= self.config.max_board:
            self._log(
                frame.controller,
                f"{frame.source_name} 手牌复制召唤失败：场地已满",
            )
            return
        origin = origin_for_summoned_card(hand_card.definition)
        unit = self._summon_follower_to_board(
            frame.controller,
            hand_card.definition,
            summon_cause="hand_copy_summon",
            origin=origin,
            source_origin=hand_card.source_origin or hand_card.origin,
        )
        if unit is None:
            self._log(
                frame.controller,
                f"{frame.source_name} 手牌复制召唤失败：场地已满",
            )
            return
        for modifier in hand_card.stat_modifiers:
            unit.add_stat_modifier(
                replace(
                    modifier,
                    modifier_id=self._allocate_modifier_id(),
                )
            )
        self._apply_hand_card_runtime_to_unit(hand_card, unit)
        if effect.target_key:
            previous_outputs = frame._target_bindings.get(
                effect.target_key,
                (),
            )
            self._bind_targets(
                frame,
                effect.target_key,
                (*previous_outputs, unit.entity_id),
                effect,
            )
        self._log(
            frame.controller,
            f"{frame.source_name} 召唤手牌中 {hand_card.name} 的完全相同复制品 "
            f"({unit.attack}/{unit.health})",
        )
        self._emit(
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                frame.controller,
                source_id=unit.entity_id,
                metadata={
                    "source": unit,
                    "card_id": hand_card.card_id,
                    "origin": unit.origin.value,
                    "derived": True,
                    "token": is_token_definition(hand_card.definition),
                    "via": "hand_copy_summon",
                    "copied_from_entity_id": hand_card.entity_id,
                    "source_card_id": frame.source_card_id,
                },
            )
        )

    def _execute_summon_from_hand(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        player = self.players[frame.controller]
        hand_card = next(
            (
                card
                for card in self._hand_cards(frame.controller)
                if card.entity_id == target_id
            ),
            None,
        )
        if hand_card is None:
            return
        if hand_card.card_type != "随从":
            raise IllegalCommand("SUMMON_FROM_HAND requires a follower in hand")
        if len(player.board) >= self.config.max_board:
            self._log(
                frame.controller,
                f"{frame.source_name} 从手牌召唤失败：场地已满",
            )
            return

        hand_index = player.hand.index(hand_card)
        player.hand.pop(hand_index)
        player.hand_entity_ids.pop(hand_index)
        unit = self._summon_follower_to_board(
            frame.controller,
            hand_card.definition,
            summon_cause="summon_from_hand",
            entity_id=hand_card.entity_id,
            origin=hand_card.origin,
            source_origin=hand_card.source_origin,
            fused_material_ids=tuple(hand_card.fused_material_ids),
        )
        if unit is None:
            player.hand.insert(hand_index, hand_card)
            player.hand_entity_ids.insert(hand_index, hand_card.entity_id)
            return
        for modifier in hand_card.stat_modifiers:
            unit.add_stat_modifier(modifier)
        self._apply_hand_card_runtime_to_unit(hand_card, unit)
        self._emit(
            GameEvent(
                EventType.HAND_CARD_SUMMONED,
                frame.controller,
                source_id=unit.entity_id,
                metadata={
                    "card_id": unit.definition.card_id,
                    "entity_id": unit.entity_id,
                    "source_card_id": frame.source_card_id,
                    "from_zone": "hand",
                    "to_zone": "board",
                },
            )
        )
        self._emit(
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                frame.controller,
                source_id=unit.entity_id,
                metadata={
                    "source": unit,
                    "card_id": unit.definition.card_id,
                    "via": "summon_from_hand",
                    "origin": unit.origin.value,
                    "derived": is_derived(unit.origin),
                    "token": is_token_definition(unit.definition)
                    or unit.origin is CardOrigin.TOKEN,
                    "source_card_id": frame.source_card_id,
                },
            )
        )
        self._log(
            frame.controller,
            f"{frame.source_name} 从手牌召唤 {hand_card.name} "
            f"({unit.attack}/{unit.health})",
        )

    def _execute_transform_hand_from_random_enemy_deck(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        try:
            hand_target = self._find_hand_card(frame.controller, target_id)
        except IllegalCommand:
            return
        enemy_deck = self.players[1 - frame.controller].deck
        if not enemy_deck:
            self._log(
                frame.controller,
                f"{frame.source_name} 变身失败：敌方牌组为空",
            )
            return

        selected = self.random.choice(enemy_deck)
        replacement = (
            selected.definition
            if isinstance(selected, DeckCard)
            else selected
        )
        inherited_cost_modifiers = (
            [
                replace(
                    modifier,
                    modifier_id=self._allocate_modifier_id(),
                )
                for modifier in selected.cost_modifiers
            ]
            if isinstance(selected, DeckCard)
            else []
        )
        old_name = hand_target.name
        transform_event = self._transform_hand_card(
            hand_target,
            replacement,
            frame.controller,
            preserve_fused_materials=False,
        )
        hand_target.cost_modifiers.extend(inherited_cost_modifiers)
        transform_event.metadata.update({
            "source_card_id": frame.source_card_id,
            "copied_from_zone": "enemy_deck",
            "copied_card_id": replacement.card_id,
            "cost_after": hand_target.current_cost,
        })
        self._emit(transform_event)
        self._log(
            frame.controller,
            f"手牌 {old_name} 变身为敌方牌组中的随机卡牌",
        )

    def _execute_transform_board_from_random_own_deck(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        if effect.deck_filter is None:
            raise IllegalCommand(
                "TRANSFORM_BOARD_FROM_RANDOM_OWN_DECK requires a deck filter"
            )
        candidates = tuple(
            card
            for card in self.players[frame.controller].deck
            if effect.deck_filter.matches(card)
        )
        if not candidates:
            self._log(
                frame.controller,
                f"{frame.source_name} 变形失败：牌组中没有符合条件的卡牌",
            )
            return
        try:
            target = self._find_board_entity(target_id)
        except IllegalCommand:
            return
        if not isinstance(target, Unit):
            raise IllegalCommand(
                "TRANSFORM_BOARD_FROM_RANDOM_OWN_DECK target must be a follower"
            )

        selected = self.random.choice(candidates)
        replacement = (
            selected.definition
            if isinstance(selected, DeckCard)
            else selected
        )
        self._execute_transform(
            replace(
                effect,
                kind=EffectKind.TRANSFORM,
                card_id=replacement.card_id,
            ),
            frame,
            target.entity_id,
            replacement=replacement,
        )
        transformed = self._find_board_entity(target.entity_id)
        if not isinstance(transformed, Unit):
            raise IllegalCommand(
                "TRANSFORM_BOARD_FROM_RANDOM_OWN_DECK replacement must be "
                "a follower"
            )
        if isinstance(selected, DeckCard):
            for modifier in selected.stat_modifiers:
                transformed.add_stat_modifier(
                    replace(
                        modifier,
                        modifier_id=self._allocate_modifier_id(),
                    )
                )
        for event in reversed(self.state.event_queue):
            if (
                event.type is EventType.BOARD_CARD_TRANSFORMED
                and event.source_id == transformed.entity_id
            ):
                event.metadata.update({
                    "copied_from_zone": "own_deck",
                    "copied_card_id": replacement.card_id,
                    "exact_copy": True,
                    "with_replacement": True,
                })
                break

    def _execute_transform_deck_cards(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.card_id is None:
            raise IllegalCommand("TRANSFORM_DECK_CARDS requires a card_id")
        if effect.deck_filter is None:
            raise IllegalCommand(
                "TRANSFORM_DECK_CARDS requires a deck filter"
            )
        if self.card_resolver is None:
            raise IllegalCommand(
                "No card_resolver registered for TRANSFORM_DECK_CARDS"
            )
        try:
            replacement = self.card_resolver(effect.card_id)
        except KeyError as exc:
            raise IllegalCommand(
                f"Card {effect.card_id} not found for TRANSFORM_DECK_CARDS"
            ) from exc
        if replacement is None:
            raise IllegalCommand(
                f"Card {effect.card_id} not found for TRANSFORM_DECK_CARDS"
            )

        player = self.players[frame.controller]
        transformed_count = 0
        for index, raw_card in enumerate(player.deck):
            if not effect.deck_filter.matches(raw_card):
                continue
            old_definition = (
                raw_card.definition
                if isinstance(raw_card, DeckCard)
                else raw_card
            )
            player.deck[index] = replacement
            transformed_count += 1
            self._emit(GameEvent(
                EventType.DECK_CARD_TRANSFORMED,
                frame.controller,
                source_id=frame.source_entity_id,
                metadata={
                    "source_card_id": frame.source_card_id,
                    "old_card_id": old_definition.card_id,
                    "new_card_id": replacement.card_id,
                    "old_definition": old_definition,
                    "new_definition": replacement,
                    "deck_index": index,
                },
            ))
        if transformed_count:
            self._log(
                frame.controller,
                f"{frame.source_name} 使牌组中的 {transformed_count} 张卡牌变形",
            )

    def _execute_transform(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
        *,
        replacement: CardDefinition | None = None,
    ) -> None:
        if replacement is None:
            if effect.card_id is None:
                raise IllegalCommand("TRANSFORM requires a card_id")
            if self.card_resolver is None:
                raise IllegalCommand("No card_resolver registered for TRANSFORM")
            try:
                replacement = self.card_resolver(effect.card_id)
            except KeyError as exc:
                raise IllegalCommand(
                    f"Card {effect.card_id} not found for TRANSFORM"
                ) from exc
            if replacement is None:
                raise IllegalCommand(
                    f"Card {effect.card_id} not found for TRANSFORM"
                )

        hand_target = next(
            (
                card
                for card in self._hand_cards(frame.controller)
                if card.entity_id == target_id
            ),
            None,
        )
        if hand_target is not None:
            old_name = hand_target.name
            transform_event = self._transform_hand_card(
                hand_target,
                replacement,
                frame.controller,
                preserve_fused_materials=False,
            )
            if effect.mode is not None:
                hand_target.cost_modifiers.append(CostModifier(
                    modifier_id=self._allocate_modifier_id(),
                    mode=effect.mode.value,
                    amount=effect.amount,
                    duration=effect.duration.value,
                    expires_for_player=_expires_for_player(
                        effect.duration,
                        frame.controller,
                        self.state.active_player,
                    ),
                ))
                transform_event.metadata.update({
                    "cost_after": hand_target.current_cost,
                    "cost_mode": effect.mode.value,
                    "cost_amount": effect.amount,
                    "cost_duration": effect.duration.value,
                })
            self._emit(transform_event)
            self._log(
                frame.controller,
                f"手牌 {old_name} 变形为 {replacement.name}",
            )
            return

        target = self._find_board_entity(target_id)
        if replacement.card_type not in {"随从", "护符"}:
            raise IllegalCommand(
                "TRANSFORM board replacement must be a follower or amulet"
            )
        old_name = target.definition.name
        old_definition = target.definition
        owner = self._entity_owner(target.entity_id)
        if replacement.card_type == "护符":
            previous_origin = target.source_origin or target.origin
            fresh = Amulet(
                definition=replacement,
                entity_id=target.entity_id,
                countdown=self.rulebook.countdown_for(replacement.card_id),
                entered_turn=self.turn,
                origin=CardOrigin.TRANSFORMED,
                source_origin=previous_origin,
                fused_material_ids=list(target.fused_material_ids),
            )
            board = self.players[owner].board
            board[board.index(target)] = fresh
            self._death_causes.pop(fresh.entity_id, None)
            self._emit(
                GameEvent(
                    EventType.BOARD_CARD_TRANSFORMED,
                    owner,
                    source_id=fresh.entity_id,
                    metadata={
                        "source": fresh,
                        "old_definition": old_definition,
                        "new_definition": replacement,
                        "old_card_id": old_definition.card_id,
                        "new_card_id": replacement.card_id,
                        "old_card_type": old_definition.card_type,
                        "new_card_type": replacement.card_type,
                    },
                )
            )
            self._initialize_earth_sigil(fresh, owner)
            self._log(
                frame.controller,
                f"{old_name} 变形为护符 {replacement.name}",
            )
            return
        if isinstance(target, Amulet):
            previous_origin = target.source_origin or target.origin
            fresh = Unit.summon(
                replacement,
                entity_id=target.entity_id,
                origin=CardOrigin.TRANSFORMED,
                source_origin=previous_origin,
            )
            fresh.fused_material_ids = list(target.fused_material_ids)
            self._apply_initial_keyword_overrides(fresh)
            self._apply_initial_passives(fresh)
            fresh._synchronize_keyword_state()
            board = self.players[owner].board
            board[board.index(target)] = fresh
            self._death_causes.pop(fresh.entity_id, None)
            self._emit(
                GameEvent(
                    EventType.BOARD_CARD_TRANSFORMED,
                    owner,
                    source_id=fresh.entity_id,
                    metadata={
                        "source": fresh,
                        "old_definition": old_definition,
                        "new_definition": replacement,
                        "old_card_id": old_definition.card_id,
                        "new_card_id": replacement.card_id,
                        "old_card_type": old_definition.card_type,
                        "new_card_type": replacement.card_type,
                    },
                )
            )
            self._log(
                frame.controller,
                f"{old_name} 变形为 {replacement.name}"
                f"（{fresh.attack}/{fresh.health}）",
            )
            return
        if not isinstance(target, Unit):
            raise IllegalCommand("TRANSFORM target must be a board card or own hand card")
        can_attack = target.can_attack
        attacks_used = max(
            0, target.attacks_per_turn - target.attacks_remaining
        )
        summoned_this_turn = target.summoned_this_turn
        previous_origin = target.source_origin or target.origin
        fresh = Unit.summon(
            replacement,
            entity_id=target.entity_id,
            origin=CardOrigin.TRANSFORMED,
            source_origin=previous_origin,
        )
        target.definition = fresh.definition
        target.base_attack = fresh.base_attack
        target.base_health = fresh.base_health
        target.attack = fresh.attack
        target.health = fresh.health
        target.max_health = fresh.max_health
        target.origin = fresh.origin
        target.source_origin = fresh.source_origin
        target.can_attack = can_attack
        target.attacks_remaining = max(0, 1 - attacks_used)
        target.can_attack = can_attack and target.attacks_remaining > 0
        target.evolved = False
        target.super_evolved = False
        target.super_evolved_turn = None
        target.rush_only = False
        target.barrier_charges = fresh.barrier_charges
        target.ambush_active = fresh.ambush_active
        target.summoned_this_turn = summoned_this_turn
        target.permanent_keywords.clear()
        target.temporary_keywords.clear()
        target.removed_keywords.clear()
        target.temporary_keyword_removals.clear()
        target.stat_modifiers.clear()
        target.attack_capacity_modifiers.clear()
        target.attack_restrictions.clear()
        target.targeting_restrictions.clear()
        target.printed_abilities_removed = False
        target.last_words_removed = False
        target.granted_last_words.clear()
        target.effect_destroy_immunity = False
        target.turn_end_destroy_timings.clear()
        target.turn_end_banish_timings.clear()
        self._apply_initial_keyword_overrides(target)
        self._apply_initial_passives(target)
        target._synchronize_keyword_state()
        self._death_causes.pop(target.entity_id, None)
        self._emit(
            GameEvent(
                EventType.BOARD_CARD_TRANSFORMED,
                owner,
                source_id=target.entity_id,
                metadata={
                    "source": target,
                    "old_definition": old_definition,
                    "new_definition": replacement,
                    "old_card_id": old_definition.card_id,
                    "new_card_id": replacement.card_id,
                    "old_card_type": old_definition.card_type,
                    "new_card_type": replacement.card_type,
                },
            )
        )
        self._log(
            frame.controller,
            f"{old_name} 变形为 {replacement.name}"
            f"（{target.attack}/{target.health}）",
        )

    def _execute_set_stats(
        self, effect: EffectOperation, frame: EffectFrame, target_id: int | None
    ) -> None:
        target = self._find_board_entity(target_id)
        if not isinstance(target, Unit):
            raise IllegalCommand("SET_STATS target must be a follower")

        if effect.set_attack:
            attack_val = (
                effect.amount
                if not effect.amount_expr
                else evaluate_expression(
                    effect.amount_expr,
                    self._build_eval_context(frame, target_id),
                )
            )
            if attack_val < 0:
                attack_val = 0
            target.base_attack = attack_val
            target._recompute_attack()

        if effect.set_health:
            health_val = (
                effect.secondary_amount
                if not effect.secondary_expr
                else evaluate_expression(
                    effect.secondary_expr,
                    self._build_eval_context(frame, target_id),
                )
            )
            if health_val < 1:
                health_val = 1
            target.base_health = health_val
            target._recompute_max()
            target.health = target.max_health

    def _execute_attack_restriction(
        self, effect: EffectOperation, frame: EffectFrame, target_id: int | None, *, add: bool
    ) -> None:
        if effect.restriction is None:
            raise IllegalCommand(f"{effect.kind.value} requires a restriction")
        target = self._find_board_entity(target_id)
        if not isinstance(target, Unit):
            raise IllegalCommand("Attack restriction target must be a follower")
        from swb.engine.state import AttackRestriction
        restriction = AttackRestriction(effect.restriction)
        if add:
            target.add_attack_restriction(
                restriction,
                duration=effect.duration.value,
                expires_for_player=_expires_for_player(
                    effect.duration,
                    frame.controller,
                    self.state.active_player,
                ),
            )
        else:
            target.remove_attack_restriction(restriction)

    def _execute_targeting_restriction(
        self, effect: EffectOperation, frame: EffectFrame, target_id: int | None, *, add: bool
    ) -> None:
        if effect.restriction is None:
            raise IllegalCommand(f"{effect.kind.value} requires a restriction")
        target = self._find_board_entity(target_id)
        if not isinstance(target, Unit):
            raise IllegalCommand("Targeting restriction target must be a follower")
        from swb.engine.state import TargetingRestriction
        restriction = TargetingRestriction(effect.restriction)
        if add:
            target.add_targeting_restriction(
                restriction,
                duration=effect.duration.value,
                expires_for_player=_expires_for_player(
                    effect.duration,
                    frame.controller,
                    self.state.active_player,
                ),
            )
        else:
            target.remove_targeting_restriction(restriction)

    def _execute_spellboost_hand(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        amount = effect.amount
        if effect.amount_expr is not None:
            amount = evaluate_expression(
                effect.amount_expr,
                self._build_eval_context(frame, target_id),
            )
        if amount <= 0:
            return

        if target_id is None:
            return
        hand_card = self._find_hand_card(frame.controller, target_id)
        if not isinstance(hand_card, HandCard):
            return

        hand_card.apply_spellboost(amount)
        self._emit(
            GameEvent(
                EventType.SPELLBOOSTED,
                frame.controller,
                source_id=hand_card.entity_id,
                amount=amount,
                metadata={
                    "card_id": hand_card.card_id,
                    "spellboost_count": hand_card.spellboost_count,
                    "source_card_id": frame.source_card_id,
                    "source_entity_id": frame.source_entity_id,
                },
            )
        )

    def _execute_necromancy(
        self, effect: EffectOperation, frame: EffectFrame, target_id: int | None,
    ) -> None:
        cost = effect.amount
        if effect.amount_expr is not None:
            cost = evaluate_expression(effect.amount_expr, self._build_eval_context(frame, target_id))
        player = self.players[frame.controller]
        before = player.shadows
        if not player.consume_shadows(cost):
            return
        self._emit(GameEvent(EventType.NECROMANCY_ACTIVATED, frame.controller, amount=cost,
            metadata={"shadows_before": before, "shadows_after": player.shadows, "source_card_id": frame.source_card_id}))
        self._log(frame.controller, f"死灵术 {cost}：墓场 {before} → {player.shadows}")
        self._emit(
            GameEvent(
                EventType.SHADOWS_CHANGED,
                frame.controller,
                amount=cost,
                metadata={
                    "change": "spend",
                    "shadows_before": before,
                    "shadows_after": player.shadows,
                },
            )
        )
        self._queue_effects_from_frame(
            frame,
            effect.necromancy_operations,
            label="死灵术",
        )

    def _execute_reanimate(
        self, effect: EffectOperation, frame: EffectFrame, target_id: int | None,
    ) -> None:
        player = self.players[frame.controller]
        max_cost = effect.amount
        candidates = [
            record
            for record in self.state.destroyed_followers
            if record.owner == frame.controller
            and record.definition.cost <= max_cost
            and is_reanimate_eligible(record)
        ]
        if not candidates:
            if effect.target_key:
                self._bind_targets(frame, effect.target_key, (), effect)
            return
        max_c = max(r.definition.cost for r in candidates)
        best = [r for r in candidates if r.definition.cost == max_c]
        chosen = self.random.choice(best)
        unit = self._summon_follower_to_board(
            frame.controller,
            chosen.definition,
            summon_cause="reanimate",
            origin=CardOrigin.REANIMATED,
            source_origin=chosen.source_origin or chosen.origin,
        )
        if unit is None:
            if effect.target_key:
                self._bind_targets(frame, effect.target_key, (), effect)
            return
        if effect.target_key:
            self._bind_targets(
                frame,
                effect.target_key,
                (unit.entity_id,),
                effect,
            )
        self._emit(GameEvent(EventType.REANIMATE_RESOLVED, frame.controller,
            amount=max_cost,
            metadata={"reanimated_card_id": chosen.definition.card_id, "new_entity_id": unit.entity_id,
                       "source_card_id": frame.source_card_id,
                       "origin": CardOrigin.REANIMATED.value,
                       "derived": True,
                       "token": is_token_definition(chosen.definition) or chosen.token,
                       "was_derived": chosen.derived,
                       "was_token": chosen.token,
                       "source_origin": (chosen.source_origin or chosen.origin).value}))
        self._log(frame.controller, f"亡者召还：{chosen.definition.name} ({unit.attack}/{unit.health})")
        self._emit(GameEvent(EventType.FOLLOWER_SUMMONED, frame.controller, source_id=unit.entity_id,
            metadata={"source": unit, "card_id": unit.definition.card_id, "via": "reanimate",
                       "origin": unit.origin.value, "derived": is_derived(unit.origin),
                       "token": is_token_definition(unit.definition) or unit.origin is CardOrigin.TOKEN}))

    def _execute_summon_from_graveyard(
        self, effect: EffectOperation, frame: EffectFrame, target_id: int | None,
    ) -> None:
        player = self.players[frame.controller]
        gc = next((g for g in player.graveyard if g.entity_id == target_id), None)
        if gc is None:
            return
        unit = self._summon_follower_to_board(
            frame.controller,
            gc.definition,
            summon_cause="summon_from_graveyard",
            entity_id=gc.entity_id,
            origin=gc.origin,
            source_origin=gc.source_origin,
        )
        if unit is None:
            return
        player.graveyard.remove(gc)
        self._emit(GameEvent(EventType.GRAVEYARD_CARD_SUMMONED, frame.controller,
            source_id=unit.entity_id,
            metadata={
                "card_id": gc.definition.card_id,
                "entity_id": unit.entity_id,
                "source_card_id": frame.source_card_id,
                "from_zone": "graveyard",
                "to_zone": "board",
                "cause": "summon_from_graveyard",
                "origin": gc.origin.value,
                "derived": gc.derived,
                "token": gc.token,
            }))
        self._log(frame.controller, f"从墓地召唤：{gc.definition.name} ({unit.attack}/{unit.health})")
        self._emit(GameEvent(EventType.FOLLOWER_SUMMONED, frame.controller, source_id=unit.entity_id,
            metadata={"source": unit, "card_id": unit.definition.card_id, "via": "summon_from_graveyard",
                       "origin": unit.origin.value, "derived": is_derived(unit.origin),
                       "token": is_token_definition(unit.definition) or unit.origin is CardOrigin.TOKEN}))

    def _execute_return_from_graveyard_to_hand(
        self, effect: EffectOperation, frame: EffectFrame, target_id: int | None,
    ) -> None:
        player = self.players[frame.controller]
        gc = next((g for g in player.graveyard if g.entity_id == target_id), None)
        if gc is None or not is_graveyard_return_eligible(gc):
            return
        if len(player.hand) >= self.config.max_hand:
            return
        player.graveyard.remove(gc)
        hand_card = self._make_hand_card(
            gc.definition,
            gc.entity_id,
            origin=gc.origin,
            source_origin=gc.source_origin,
        )
        player.hand.append(hand_card)
        player.hand_entity_ids.append(hand_card.entity_id)
        self._emit(GameEvent(EventType.GRAVEYARD_CARD_RETURNED, frame.controller,
            source_id=frame.source_entity_id,
            target_id=gc.entity_id,
            metadata={
                "card_id": gc.definition.card_id,
                "entity_id": gc.entity_id,
                "source_card_id": frame.source_card_id,
                "from_zone": "graveyard",
                "to_zone": "hand",
                "cause": "return_from_graveyard",
                "origin": gc.origin.value,
                "derived": gc.derived,
                "token": gc.token,
            }))
        self._log(frame.controller, f"从墓地回手：{gc.definition.name}")

    def _execute_banish_from_graveyard(
        self, effect: EffectOperation, frame: EffectFrame, target_id: int | None,
    ) -> None:
        player = self.players[frame.controller]
        gc = next((g for g in player.graveyard if g.entity_id == target_id), None)
        if gc is None:
            return
        player.graveyard.remove(gc)
        player.banished.append(gc.definition)
        self._emit(GameEvent(EventType.GRAVEYARD_CARD_BANISHED, frame.controller,
            source_id=frame.source_entity_id,
            target_id=gc.entity_id,
            metadata={
                "card_id": gc.definition.card_id,
                "entity_id": gc.entity_id,
                "source_card_id": frame.source_card_id,
                "from_zone": "graveyard",
                "to_zone": "banished",
                "cause": "banish_from_graveyard",
            }))
        self._log(frame.controller, f"从墓地消失：{gc.definition.name}")

    def _execute_gain_emblem(self, effect: EffectOperation, frame: EffectFrame) -> None:
        emblem_id = effect.emblem_id
        if not emblem_id:
            raise IllegalCommand("GAIN_EMBLEM requires emblem_id")
        emblem_def = self.rulebook.emblem_def(emblem_id)
        if emblem_def is None:
            self._log(frame.controller, f"[未实现] 纹章 '{emblem_id}' 未定义")
            return
        player_index = (
            1 - frame.controller
            if effect.target is TargetKind.ENEMY_LEADER
            else frame.controller
        )
        self._add_emblem_to_player(player_index, emblem_def, frame.source_card)

    def _add_emblem_to_player(
        self,
        player_index: int,
        emblem_def,
        source_card=None,
        *,
        source_card_id: int | None = None,
    ):
        from swb.engine.emblem import EmblemStacking
        from swb.engine.state import EmblemInstance
        player = self.players[player_index]
        if source_card is not None and source_card_id is not None:
            raise TypeError("provide source_card or source_card_id, not both")
        if source_card is None:
            source_card = source_card_id
        if source_card is None:
            raise TypeError("source_card or source_card_id is required")
        if isinstance(source_card, int):
            source_card_id = source_card
            resolved_source = None
            if self.card_resolver is not None:
                try:
                    resolved_source = self.card_resolver(source_card_id)
                except KeyError:
                    # Synthetic tests and external callers may use an audit-only
                    # source ID that intentionally has no database definition.
                    resolved_source = None
            source_card = resolved_source or type("_EmblemSourceCard", (), {
                "card_id": source_card_id,
                "name": f"纹章_{emblem_def.emblem_id}",
            })()
        else:
            source_card_id = source_card.card_id

        existing_emblems = [
            emblem
            for emblem in player.emblems
            if emblem.emblem_id == emblem_def.emblem_id
        ]
        if emblem_def.stacking is EmblemStacking.REPLACE:
            for existing in tuple(player.emblems):
                if existing.emblem_id == emblem_def.emblem_id:
                    self._remove_emblem_instance(
                        player_index,
                        existing,
                        removal_cause="replace",
                    )
        elif existing_emblems:
            # SWB's leader area cannot contain multiple copies of the same
            # emblem.  ``allow`` remains parseable for old rule files, but it
            # only permits different emblems to coexist; it never stacks an
            # identical emblem's triggers.
            return

        if not self._leader_area_has_capacity(player_index):
            self._log(
                player_index,
                f"主战者区域已满，无法获得纹章 {emblem_def.emblem_id}",
            )
            return

        seq = player._next_emblem_sequence
        player._next_emblem_sequence += 1
        countdown = emblem_def.countdown
        instance = EmblemInstance(
            emblem_id=emblem_def.emblem_id,
            definition=emblem_def,
            entity_id=self.state.allocate_entity_id(),
            controller=player_index,
            created_sequence=seq,
            countdown=countdown,
            countdown_before=countdown,
        )
        player.emblems.append(instance)
        self._emit(GameEvent(
            EventType.EMBLEM_GAINED, player_index,
            source_id=instance.entity_id,
            metadata={
                "emblem_id": emblem_def.emblem_id,
                "entity_id": instance.entity_id,
                "source_card_id": source_card_id,
                "controller": player_index,
                "countdown": instance.countdown,
                "stacking": emblem_def.stacking.value,
            },
        ))
        if emblem_def.on_gain:
            self._queue_effects(
                source_card,
                instance.entity_id,
                emblem_def.on_gain,
                controller=player_index,
                label="纹章 获得时",
            )

    def _execute_remove_emblem(self, effect: EffectOperation, frame: EffectFrame) -> None:
        emblem_id = effect.emblem_id
        if not emblem_id:
            raise IllegalCommand("REMOVE_EMBLEM requires emblem_id")
        player_index = (
            1 - frame.controller
            if effect.target is TargetKind.ENEMY_LEADER
            else frame.controller
        )
        player = self.players[player_index]
        removed = [e for e in player.emblems if e.emblem_id == emblem_id]
        if not removed:
            return
        targets = removed if effect.emblem_remove_mode == "all" else removed[:1]
        for target in targets:
            self._remove_emblem_instance(
                player_index,
                target,
                removal_cause="effect",
            )
        for target in reversed(targets):
            if target.definition.last_words:
                self._queue_effects(
                    self._emblem_effect_source_card(target.definition),
                    None,
                    target.definition.last_words,
                    controller=player_index,
                    label="纹章 谢幕曲",
                )

    def _execute_remove_all_emblems(
        self,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        if not _is_leader_target_id(target_id):
            raise IllegalCommand(
                "REMOVE_ALL_EMBLEMS requires a leader target"
            )
        player_index = _leader_index_from_target_id(target_id)
        targets = tuple(self.players[player_index].emblems)
        for target in targets:
            self._remove_emblem_instance(
                player_index,
                target,
                removal_cause="effect",
            )
        for target in reversed(targets):
            if target.definition.last_words:
                self._queue_effects(
                    self._emblem_effect_source_card(target.definition),
                    None,
                    target.definition.last_words,
                    controller=player_index,
                    label="纹章 谢幕曲",
                )

    def _emblem_effect_source_card(self, definition):
        source_card = (
            self.card_resolver(definition.source_card_id)
            if self.card_resolver
            else None
        )
        if source_card is not None:
            return source_card
        return type("_EmblemCard", (), {
            "card_id": definition.source_card_id,
            "name": f"纹章_{definition.emblem_id}",
        })()

    def _remove_emblem_instance(
        self,
        player_index: int,
        target,
        *,
        removal_cause: str,
    ) -> None:
        player = self.players[player_index]
        if target not in player.emblems:
            return
        player.emblems.remove(target)
        self._emit(GameEvent(
            EventType.EMBLEM_REMOVED, player_index,
            source_id=target.entity_id,
            metadata={
                "emblem_id": target.emblem_id,
                "emblem_entity_id": target.entity_id,
                "owner": player_index,
                "cause": removal_cause,
                "countdown_before": target.countdown_before,
                "countdown_after": target.countdown,
                "entity_id": target.entity_id,
                "source_card_id": target.definition.source_card_id,
                "controller": player_index,
                "removal_cause": removal_cause,
            },
        ))

    def _check_emblem_trigger_scope(
        self,
        player_index: int,
        tr,
        event_type: str,
        event_player: int | None,
    ) -> bool:
        from swb.engine.emblem import TurnScope, EventScope
        active_player = self.state.active_player
        turn_scope = tr.turn_scope
        if turn_scope is None:
            turn_scope = (
                TurnScope.OWNER_TURN
                if event_type in {"turn_start", "turn_end"}
                else TurnScope.ANY_TURN
            )
        event_scope = tr.event_scope
        if event_scope is None:
            event_scope = (
                EventScope.ANY_EVENT
                if event_type in {"turn_start", "turn_end"}
                else EventScope.OWNER_EVENT
            )
        if turn_scope is TurnScope.OWNER_TURN and active_player != player_index:
            return False
        if turn_scope is TurnScope.OPPONENT_TURN and active_player == player_index:
            return False
        if event_scope is EventScope.OWNER_EVENT:
            return event_player == player_index
        if event_scope is EventScope.OPPONENT_EVENT:
            return event_player is not None and event_player != player_index
        return True

    def _emblem_operation_can_start(
        self,
        operation: EffectOperation,
        controller: int,
        source_entity_id: int | None,
        emblem_entity_id: int | None = None,
    ) -> bool:
        if operation.kind is EffectKind.TARGET_EXISTS:
            branch_ops = (
                operation.then_operations
                if self._target_exists_for(
                    operation,
                    controller,
                    source_entity_id=source_entity_id,
                )
                else operation.else_operations
            )
            return bool(branch_ops)
        if operation.kind is EffectKind.CONDITIONAL:
            condition_state = evaluate_conditions_without_target(
                operation.conditions,
                self._eval_context(
                    controller,
                    source_entity_id=source_entity_id,
                ),
            )
            if condition_state is PartialConditionResult.TRUE:
                branch_ops = operation.then_operations
            elif condition_state is PartialConditionResult.FALSE:
                branch_ops = operation.else_operations
            else:
                branch_ops = (
                    operation.then_operations + operation.else_operations
                )
            return any(
                self._emblem_operation_can_start(
                    branch_operation,
                    controller,
                    source_entity_id,
                    emblem_entity_id,
                )
                for branch_operation in branch_ops
            )
        condition_state = evaluate_conditions_without_target(
            operation.conditions,
            self._eval_context(
                controller,
                source_entity_id=source_entity_id,
            ),
        )
        if condition_state is PartialConditionResult.FALSE:
            return False
        if operation.target in {
            TargetKind.OWN_LEADER,
            TargetKind.ENEMY_LEADER,
        }:
            return True
        if operation.target is TargetKind.EMBLEM_SELF:
            return (
                emblem_entity_id is not None
                and any(
                    emblem.entity_id == emblem_entity_id
                    for player in self.players
                    for emblem in player.emblems
                )
            )
        if operation.target is TargetKind.SELF:
            if source_entity_id is None:
                return False
            try:
                self._find_board_entity(source_entity_id)
            except IllegalCommand:
                return (
                    operation.kind
                    in {
                        EffectKind.REDUCE_COUNTDOWN,
                        EffectKind.INCREASE_COUNTDOWN,
                    }
                    and any(
                        emblem.entity_id == source_entity_id
                        for player in self.players
                        for emblem in player.emblems
                    )
                )
            return True
        if operation.target is TargetKind.PREVIOUS_TARGET:
            return False
        if operation.kind is EffectKind.DISTRIBUTE_DAMAGE:
            return operation.include_leader or any(
                isinstance(entity, Unit)
                for entity in self.players[1 - controller].board
            )
        if operation.target in {
            TargetKind.OWN_HAND,
            TargetKind.RANDOM_OWN_HAND,
            TargetKind.RANDOM_ENEMY_HAND,
            TargetKind.ALL_OWN_HAND,
            TargetKind.ALL_ENEMY_HAND,
        }:
            return bool(
                hand_candidates(
                    operation,
                    controller,
                    self.players,
                    source_entity_id=source_entity_id,
                )
            )
        if operation.target is TargetKind.ALL_LEADERS:
            return bool(leader_target_ids(operation, controller, self.players))
        if operation.target is TargetKind.ALL_ENEMY_UNITS_AND_LEADER:
            return True
        if operation.target is TargetKind.ALL_OWN_EMBLEMS:
            return any(
                emblem.countdown is not None
                and (
                    operation.emblem_id is None
                    or emblem.emblem_id == operation.emblem_id
                )
                for emblem in self.players[controller].emblems
            )
        if is_graveyard_target(operation.target):
            return bool(graveyard_candidates(operation, controller, self.players))
        if (
            is_choice_target(operation.target)
            or is_random_target(operation.target)
            or is_all_target(operation.target)
        ):
            candidates = target_candidates(
                operation,
                controller,
                self.players,
                source_entity_id=source_entity_id,
            )
            if is_choice_target(operation.target):
                candidates = [
                    entity for entity in candidates
                    if not (
                        isinstance(entity, Unit)
                        and entity.ambush_active
                        and self._entity_owner(entity.entity_id) != controller
                    )
                ]
            if condition_state is PartialConditionResult.DEPENDS_ON_TARGET:
                candidates = [
                    entity for entity in candidates
                    if self._target_conditions_met(
                        operation.conditions,
                        entity,
                        controller,
                        source_entity_id=source_entity_id,
                    )
                ]
            return bool(candidates)
        return True

    def _event_source_card(
        self,
        event: GameEvent,
    ) -> tuple[object | None, CardDefinition | None]:
        source = event.metadata.get("source")
        if source is None:
            source = event.metadata.get("definition")
        if source is None:
            source = event.metadata.get("card")
        if isinstance(source, CardDefinition):
            return source, source
        definition = getattr(source, "definition", None)
        if isinstance(definition, CardDefinition):
            return source, definition
        return source, None

    @staticmethod
    def _listener_scope_matches(
        owner: int,
        definition: CardListenerDefinition,
        event: GameEvent,
        active_player: int,
        source_entity_id: int,
    ) -> bool:
        if (
            definition.event_scope is EventScope.OWNER_EVENT
            and event.player_index != owner
        ):
            return False
        if (
            definition.event_scope is EventScope.OPPONENT_EVENT
            and event.player_index == owner
        ):
            return False
        if (
            definition.turn_scope is TurnScope.OWNER_TURN
            and active_player != owner
        ):
            return False
        if (
            definition.turn_scope is TurnScope.OPPONENT_TURN
            and active_player == owner
        ):
            return False
        if definition.source_relation is SourceRelation.SELF:
            return event.source_id == source_entity_id
        if definition.source_relation is SourceRelation.OTHER:
            return (
                event.source_id is not None
                and event.source_id != source_entity_id
            )
        return True

    def _listener_activation_key(
        self,
        entity_id: int,
        card_id: int,
        definition_index: int,
    ) -> tuple[int, int, int]:
        return (entity_id, card_id, definition_index)

    def _listener_can_activate(
        self,
        entity_id: int,
        card_id: int,
        definition_index: int,
        definition: CardListenerDefinition,
    ) -> bool:
        key = self._listener_activation_key(
            entity_id,
            card_id,
            definition_index,
        )
        if (
            definition.once_per_turn
            and key in self.state.listener_once_per_turn_used
        ):
            return False
        return (
            definition.max_activations is None
            or self.state.listener_activation_counts.get(key, 0)
            < definition.max_activations
        )

    def _record_listener_activation(
        self,
        entity_id: int,
        card_id: int,
        definition_index: int,
        definition: CardListenerDefinition,
    ) -> int:
        key = self._listener_activation_key(
            entity_id,
            card_id,
            definition_index,
        )
        count = self.state.listener_activation_counts.get(key, 0) + 1
        self.state.listener_activation_counts[key] = count
        if definition.once_per_turn:
            self.state.listener_once_per_turn_used.add(key)
        return count

    def _leader_area_listener_sources(
        self,
        player_index: int,
    ) -> list[tuple[int, int, int]]:
        player = self.players[player_index]
        sources = [
            (instance.created_sequence, 0, instance.entity_id)
            for instance in player.emblems
        ]
        sources.extend(
            (instance.created_sequence, 1, instance.entity_id)
            for instance in player.faiths
        )
        return sorted(sources)

    def _listener_source_definition(
        self,
        owner: int | None,
        zone: str | None,
        entity_id: int | None,
        card_id: int | None,
    ) -> CardDefinition | None:
        if (
            owner not in (0, 1)
            or zone is None
            or entity_id is None
            or card_id is None
        ):
            return None
        player = self.players[owner]
        if zone == ListenerZone.BOARD.value:
            for entity in player.board:
                if (
                    entity.entity_id == entity_id
                    and entity.definition.card_id == card_id
                ):
                    if (
                        isinstance(entity, Unit)
                        and entity.printed_abilities_removed
                    ):
                        return None
                    return entity.definition
            return None
        if zone == ListenerZone.HAND.value:
            for card in self._hand_cards(owner):
                if card.entity_id == entity_id and card.card_id == card_id:
                    return card.definition
            return None
        if zone != ListenerZone.LEADER_AREA.value:
            return None
        present = any(
            instance.entity_id == entity_id
            and instance.source_card_id == card_id
            for instance in (*player.emblems, *player.faiths)
        )
        if not present:
            return None
        if self.card_resolver is not None:
            try:
                resolved = self.card_resolver(card_id)
            except KeyError:
                resolved = None
            if resolved is not None:
                return resolved
        for card in self.deck_lists[owner]:
            if card.card_id == card_id:
                return card
        return None

    def _listener_source_records(
        self,
        event: GameEvent,
    ) -> list[dict[str, object]]:
        event_active_player = (
            event.player_index
            if event.type in {EventType.TURN_STARTED, EventType.TURN_ENDED}
            else self.state.active_player
        )
        records: list[dict[str, object]] = []
        for owner in (0, 1):
            player = self.players[owner]
            zone_sources: list[tuple[int, int, int, int, CardDefinition]] = []
            zone_sources.extend(
                (0, index, 0, entity.entity_id, entity.definition)
                for index, entity in enumerate(player.board)
            )
            zone_sources.extend(
                (1, index, 0, card.entity_id, card.definition)
                for index, card in enumerate(self._hand_cards(owner))
            )
            for sequence, subtype, entity_id in self._leader_area_listener_sources(
                owner
            ):
                card_id = next(
                    (
                        instance.source_card_id
                        for instance in (*player.emblems, *player.faiths)
                        if instance.entity_id == entity_id
                    ),
                    None,
                )
                definition = self._listener_source_definition(
                    owner,
                    ListenerZone.LEADER_AREA.value,
                    entity_id,
                    card_id,
                )
                if definition is not None:
                    zone_sources.append(
                        (2, sequence, subtype, entity_id, definition)
                    )
            for zone_order, source_order, subtype, entity_id, source_card in zone_sources:
                zone = (
                    ListenerZone.BOARD
                    if zone_order == 0
                    else ListenerZone.HAND
                    if zone_order == 1
                    else ListenerZone.LEADER_AREA
                )
                if (
                    event.listener_sources is not None
                    and (
                        owner,
                        zone.value,
                        entity_id,
                        source_card.card_id,
                    )
                    not in event.listener_sources
                ):
                    continue
                for definition_index, definition in enumerate(
                    self.rulebook.listeners_for(source_card.card_id)
                ):
                    if definition.zone is not zone or definition.event is not event.type:
                        continue
                    if not self._listener_scope_matches(
                        owner,
                        definition,
                        event,
                        event_active_player,
                        entity_id,
                    ):
                        continue
                    if (
                        definition.event_filter is not None
                        and not self._event_card_filter_matches(
                            definition.event_filter,
                            event.type.value,
                            event.metadata,
                        )
                    ):
                        continue
                    if not self._listener_can_activate(
                        entity_id,
                        source_card.card_id,
                        definition_index,
                        definition,
                    ):
                        continue
                    records.append({
                        "owner": owner,
                        "zone": zone.value,
                        "source_entity_id": entity_id,
                        "source_card_id": source_card.card_id,
                        "definition_index": definition_index,
                        "order": (
                            0 if owner == event_active_player else 1,
                            zone_order,
                            source_order,
                            subtype,
                            definition_index,
                        ),
                    })
        records.sort(key=lambda record: record["order"])
        return records

    def _dispatch_card_listeners(self, event: GameEvent) -> None:
        records = self._listener_source_records(event)
        if not records:
            return
        batch_id = self._next_listener_batch_id
        self._next_listener_batch_id += 1
        self._listener_batches[batch_id] = {
            "records": records,
            "event_type": event.type.value,
            "event_player": event.player_index,
            "event_source_id": event.source_id,
            "event_target_id": event.target_id,
            "event_amount": event.amount,
            "event_metadata": dict(event.metadata),
        }
        self._queue_next_card_listener(batch_id)
        self._continue_effects()

    def _queue_next_card_listener(self, batch_id: int) -> None:
        batch = self._listener_batches.get(batch_id)
        if batch is None:
            return
        if self.terminated:
            self._listener_batches.pop(batch_id, None)
            return
        records = batch.get("records")
        if not isinstance(records, list):
            self._listener_batches.pop(batch_id, None)
            return
        while records:
            self._step()
            record = records.pop(0)
            owner = record["owner"]
            zone = record["zone"]
            entity_id = record["source_entity_id"]
            card_id = record["source_card_id"]
            definition_index = record["definition_index"]
            source_card = self._listener_source_definition(
                owner,
                zone,
                entity_id,
                card_id,
            )
            if source_card is None:
                continue
            definitions = self.rulebook.listeners_for(card_id)
            if definition_index >= len(definitions):
                continue
            definition = definitions[definition_index]
            if not self._listener_can_activate(
                entity_id,
                card_id,
                definition_index,
                definition,
            ):
                continue
            source_spellboost_count = 0
            if zone == ListenerZone.HAND.value:
                try:
                    source_spellboost_count = self._find_hand_card(
                        owner,
                        entity_id,
                    ).spellboost_count
                except IllegalCommand:
                    continue
            if definition.conditions:
                result = evaluate_conditions_without_target(
                    definition.conditions,
                    self._eval_context(
                        owner,
                        source_entity_id=entity_id,
                        source_spellboost_count=source_spellboost_count,
                    ),
                )
                if result is not PartialConditionResult.TRUE:
                    continue
            activation_count = self._record_listener_activation(
                entity_id,
                card_id,
                definition_index,
                definition,
            )
            self._record_event(GameEvent(
                EventType.CARD_LISTENER_TRIGGERED,
                owner,
                source_id=entity_id,
                target_id=batch.get("event_source_id"),
                metadata={
                    "listener_card_id": card_id,
                    "listener_entity_id": entity_id,
                    "listener_zone": zone,
                    "definition_index": definition_index,
                    "activation_count": activation_count,
                    "trigger": batch.get("event_type"),
                    "event_player": batch.get("event_player"),
                    "event_source_id": batch.get("event_source_id"),
                    "active_player": self.state.active_player,
                    "listener_batch_id": batch_id,
                },
            ))
            frame = self._queue_effects(
                source_card,
                entity_id,
                definition.operations,
                controller=owner,
                label=f"监听 {batch.get('event_type')}",
                source_spellboost_count=source_spellboost_count,
            )
            frame.listener_batch_id = batch_id
            frame.listener_activation_owner = owner
            frame.listener_activation_zone = zone
            frame.listener_activation_entity_id = entity_id
            frame.listener_activation_card_id = card_id
            frame.listener_activation_definition_index = definition_index
            frame.listener_activation_count = activation_count
            frame.event_source_entity_id = batch.get("event_source_id")
            event_base_cost = batch.get("event_metadata", {}).get("base_cost")
            frame.event_source_base_cost = (
                event_base_cost
                if isinstance(event_base_cost, int)
                and not isinstance(event_base_cost, bool)
                and event_base_cost >= 0
                else None
            )
            return
        self._listener_batches.pop(batch_id, None)

    def _card_listener_batch_active(self) -> bool:
        return any(
            frame.listener_batch_id is not None
            for frame in self.state.effect_stack
        )

    def _dispatch_emblem_triggers(
        self, player_index: int, event_type: str,
        event_player: int | None = None,
        source_id: int | None = None,
        event_metadata: dict[str, object] | None = None,
        eligible_sources: tuple[tuple[int, int], ...] | None = None,
        freeze_conditions: bool = False,
    ) -> None:
        records: list[tuple[int, int, int, str, bool]] = []
        for pi in (0, 1):
            player = self.players[pi]
            for ei in player.emblems:
                if (
                    eligible_sources is not None
                    and (pi, ei.entity_id) not in eligible_sources
                ):
                    continue
                for ti, tr in enumerate(ei.definition.triggers):
                    if tr.trigger == event_type:
                        if ei.can_activate(ti) and self._check_emblem_trigger_scope(
                            pi, tr, event_type, event_player,
                        ) and self._event_card_filter_matches(
                            tr.event_filter,
                            event_type,
                            event_metadata,
                        ):
                            if freeze_conditions:
                                context = self._eval_context(
                                    pi,
                                    source_entity_id=source_id,
                                )
                                condition_state = (
                                    evaluate_conditions_without_target(
                                        tr.conditions,
                                        context,
                                    )
                                )
                                if (
                                    condition_state
                                    is not PartialConditionResult.TRUE
                                ):
                                    continue
                                if not tr.operations or not any(
                                    self._emblem_operation_can_start(
                                        operation,
                                        pi,
                                        source_id,
                                        ei.entity_id,
                                    )
                                    for operation in tr.operations
                                ):
                                    continue
                            records.append((
                                pi,
                                ei.entity_id,
                                ti,
                                event_type,
                                freeze_conditions,
                            ))
        records.sort(
            key=lambda record: self._emblem_order_key(
                record[0],
                record[1],
                record[2],
            )
        )
        if not records:
            return
        event_target = (
            None
            if event_metadata is None
            else event_metadata.get("target")
        )
        attack_target_entity_id = (
            getattr(event_target, "entity_id", None)
            if event_type == EventType.ATTACK_DECLARED.value
            else None
        )
        batch_id = self._next_emblem_batch_id
        self._next_emblem_batch_id += 1
        self._emblem_batches[batch_id] = {
            "records": records,
            "source_id": source_id,
            "attack_target_entity_id": attack_target_entity_id,
            "event_player": event_player,
            "trigger_batch_id": (
                None if event_metadata is None else event_metadata.get("batch_id")
            ),
            "trigger_batch_order_index": (
                None
                if event_metadata is None
                else event_metadata.get("batch_order_index")
            ),
            "trigger_batch_record_count": (
                None
                if event_metadata is None
                else event_metadata.get("batch_record_count")
            ),
        }
        self._queue_next_emblem_trigger(batch_id)
        self._continue_effects()

    def _event_card_filter_matches(
        self,
        event_filter,
        event_type: str,
        event_metadata: dict[str, object] | None,
    ) -> bool:
        if event_filter is None:
            return True
        metadata = event_metadata or {}
        if event_type == "card_fused":
            definitions = metadata.get("material_definitions", ())
            return isinstance(definitions, tuple) and any(
                isinstance(definition, CardDefinition)
                and event_filter.matches(definition, definition)
                for definition in definitions
            )
        source = metadata.get("source")
        if source is None:
            source = metadata.get("definition")
        if source is None:
            source = metadata.get("card")
        definition = (
            source
            if isinstance(source, CardDefinition)
            else getattr(source, "definition", None)
        )
        enhanced = False
        if event_type == EventType.CARD_PLAYED.value:
            card_id = metadata.get("card_id")
            mode_id = metadata.get("mode_id")
            if (
                isinstance(card_id, int)
                and not isinstance(card_id, bool)
                and isinstance(mode_id, str)
            ):
                enhanced = any(
                    mode.mode_id == mode_id and mode.is_enhance
                    for mode in self.rulebook.modes_for(card_id)
                )
        return event_filter.matches(
            definition if isinstance(definition, CardDefinition) else None,
            source,
            metadata.get("keywords"),
            enhanced=enhanced,
            cost_changed=bool(metadata.get("cost_changed", False)),
        )

    def _emblem_order_key(
        self,
        player_index: int,
        entity_id: int,
        trigger_index: int,
    ) -> tuple[int, int, int]:
        instance = next(
            (
                emblem for emblem in self.players[player_index].emblems
                if emblem.entity_id == entity_id
            ),
            None,
        )
        sequence = instance.created_sequence if instance is not None else 10**9
        controller_priority = (
            0 if player_index == self.state.active_player else 1
        )
        return (controller_priority, sequence, trigger_index)

    def _queue_next_emblem_trigger(self, batch_id: int) -> None:
        batch = self._emblem_batches.get(batch_id)
        if batch is None:
            return
        if self.terminated:
            self._emblem_batches.pop(batch_id, None)
            return
        records = batch["records"]
        while records:
            record = records.pop(0)
            (
                player_index,
                entity_id,
                trigger_index,
                event_type,
            ) = record[:4]
            conditions_frozen = len(record) >= 5 and bool(record[4])
            player = self.players[player_index]
            ei = next(
                (
                    emblem for emblem in player.emblems
                    if emblem.entity_id == entity_id
                ),
                None,
            )
            if ei is None:
                continue
            if trigger_index >= len(ei.definition.triggers):
                continue
            tr = ei.definition.triggers[trigger_index]
            if tr.trigger != event_type:
                continue
            if not ei.can_activate(trigger_index):
                continue
            if tr.conditions and not conditions_frozen:
                ctx = self._eval_context(
                    player_index,
                    source_entity_id=batch.get("source_id"),
                    attack_target_entity_id=(
                        batch.get("attack_target_entity_id")
                    ),
                )
                result = evaluate_conditions_without_target(
                    tr.conditions,
                    ctx,
                )
                if result is not PartialConditionResult.TRUE:
                    continue
            if not tr.operations:
                continue
            if not conditions_frozen and not any(
                self._emblem_operation_can_start(
                    operation,
                    player_index,
                    batch.get("source_id"),
                    ei.entity_id,
                )
                for operation in tr.operations
            ):
                continue
            self._record_emblem_trigger_event(
                player_index,
                ei,
                trigger_index,
                event_type,
                batch,
            )
            if not tr.operations:
                ei.record_activation(trigger_index)
                continue
            source_card_id = ei.definition.source_card_id
            source_card_name = f"纹章_{ei.emblem_id}"
            source_card = (
                self.card_resolver(source_card_id)
                if self.card_resolver
                else None
            )
            if source_card is None:
                source_card = type("_EmblemCard", (), {
                    "card_id": source_card_id,
                    "name": source_card_name,
                })()
            frame = self._queue_effects(
                source_card,
                batch.get("source_id"),
                tr.operations,
                controller=player_index,
                label=f"纹章 {event_type}",
                attack_target_entity_id=(
                    batch.get("attack_target_entity_id")
                ),
            )
            frame.event_source_entity_id = batch.get("source_id")
            frame.emblem_batch_id = batch_id
            frame.emblem_activation_owner = player_index
            frame.emblem_activation_entity_id = ei.entity_id
            frame.emblem_activation_trigger_index = trigger_index
            return
        self._emblem_batches.pop(batch_id, None)

    def _record_emblem_trigger_event(
        self,
        player_index: int,
        emblem,
        trigger_index: int,
        event_type: str,
        batch: dict[str, object],
    ) -> None:
        self._record_event(GameEvent(
            EventType.EMBLEM_TRIGGERED,
            player_index,
            source_id=emblem.entity_id,
            metadata={
                "emblem_id": emblem.emblem_id,
                "emblem_entity_id": emblem.entity_id,
                "owner": player_index,
                "trigger": event_type,
                "trigger_index": trigger_index,
                "activation_count": emblem.activation_counts.get(trigger_index, 0) + 1,
                "source_card_id": emblem.definition.source_card_id,
                "source_entity_id": batch.get("source_id"),
                "event_player": batch.get("event_player"),
                "active_player": self.state.active_player,
                "trigger_batch_id": batch.get("trigger_batch_id"),
                "trigger_batch_order_index": batch.get("trigger_batch_order_index"),
                "trigger_batch_record_count": batch.get("trigger_batch_record_count"),
            },
        ))

    def _record_emblem_frame_activation(self, frame: EffectFrame) -> None:
        player_index = frame.emblem_activation_owner
        entity_id = frame.emblem_activation_entity_id
        trigger_index = frame.emblem_activation_trigger_index
        if player_index is None or entity_id is None or trigger_index is None:
            return
        if (
            frame._decision_meta.get("optional_declined")
            and not frame._decision_meta.get("optional_accepted")
        ):
            return
        player = self.players[player_index]
        emblem = next(
            (candidate for candidate in player.emblems if candidate.entity_id == entity_id),
            None,
        )
        if emblem is not None and emblem.can_activate(trigger_index):
            emblem.record_activation(trigger_index)

    def _resolve_choose_one_choice(
        self,
        frame: EffectFrame,
        option_ids: tuple[str, ...],
    ) -> None:
        selected_ids = set(option_ids)
        selected = tuple(
            option
            for option in frame._decision_meta.get("choose_one_options", ())
            if f"choose_one:{option.option_id}" in selected_ids
        )
        if len(selected) != len(option_ids):
            raise IllegalCommand("Selected mode option is invalid")
        self._emit(
            GameEvent(
                EventType.MODE_SELECTED,
                frame.controller,
                source_id=frame.source_entity_id,
                amount=len(selected),
                metadata={
                    "source_card_id": frame.source_card_id,
                    "option_ids": tuple(
                        option.option_id for option in selected
                    ),
                },
            )
        )
        operations = tuple(
            operation
            for option in selected
            for operation in option.operations
        )
        if operations:
            labels = "+".join(option.label for option in selected)
            self._queue_effects_from_frame(
                frame,
                operations,
                label=f"{frame.label}/choose/{labels}",
            )

    def _execute_conditional(self, effect, frame) -> None:
        clause_id = self._runtime_clause_id(effect, frame)
        target_snapshot = None
        if effect.condition_target_key is not None:
            snapshots = frame._target_binding_snapshots.get(
                effect.condition_target_key,
                (),
            )
            if not snapshots:
                if self.runtime_coverage is not None and clause_id is not None:
                    self.runtime_coverage.record_clause(
                        clause_id,
                        "condition_evaluated",
                    )
                    self.runtime_coverage.record_clause(
                        clause_id,
                        "condition_false",
                    )
                self._log(
                    frame.controller,
                    f"{frame.source_name} 的条件目标已离开，跳过条件分支",
                )
                return
            if len(snapshots) != 1:
                raise IllegalCommand(
                    "condition_target_key requires exactly one bound target snapshot"
                )
            target_snapshot = snapshots[0]
        ctx = self._eval_context(
            frame.controller,
            source_entity_id=frame.source_entity_id,
            source_card_id=frame.source_card_id,
            source_fusion_count=len(frame.fusion_materials),
            source_spellboost_count=frame.source_spellboost_count,
            source_cost=frame.source_cost,
            distributed_value=frame.distributed_value,
            listener_activation_count=frame.listener_activation_count,
            event_source_entity_id=frame.event_source_entity_id,
            event_source_base_cost=frame.event_source_base_cost,
            source_snapshot=frame.source_snapshot,
            attack_target_entity_id=frame.attack_target_entity_id,
            target_snapshot=target_snapshot,
            bound_target_snapshots=frame._target_binding_snapshots,
        )
        if target_snapshot is None:
            condition_matches = (
                evaluate_conditions_without_target(effect.conditions, ctx)
                is PartialConditionResult.TRUE
            )
        else:
            condition_matches = all(
                evaluate_condition(condition, ctx)
                for condition in effect.conditions
            )
        if self.runtime_coverage is not None and clause_id is not None:
            self.runtime_coverage.record_clause(
                clause_id,
                "condition_evaluated",
            )
            self.runtime_coverage.record_clause(
                clause_id,
                "condition_true" if condition_matches else "condition_false",
            )
        branch_ops = effect.then_operations if condition_matches else effect.else_operations
        if branch_ops:
            self._queue_effects_from_frame(
                frame,
                branch_ops,
                label=f"{frame.label}/conditional",
            )

    def _execute_repeat(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        repeat_count = max(0, effect.amount)
        if repeat_count == 0 or not effect.repeat_operations:
            return
        if repeat_count > MAX_REPEAT_COUNT:
            diagnostics = self._loop_diagnostics()
            diagnostics.update({
                "repeat_count": repeat_count,
                "repeat_limit": MAX_REPEAT_COUNT,
                "repeat_source_card_id": frame.source_card_id,
            })
            self._record_runtime_diagnostic(
                "resolution_step_limit",
                card_id=frame.source_card_id,
                detail="repeat_count",
            )
            raise ResolutionLoopError(
                f"Repeat count {repeat_count} exceeds maximum of "
                f"{MAX_REPEAT_COUNT} for card {frame.source_card_id}",
                diagnostics=diagnostics,
            )
        self._queue_effects_from_frame(
            frame,
            effect.repeat_operations * repeat_count,
            label=f"{frame.label}/repeat×{repeat_count}",
        )

    def _execute_target_exists(self, effect, frame) -> None:
        target_exists = self._target_exists_for(
            effect,
            frame.controller,
            source_entity_id=frame.source_entity_id,
            source_fusion_count=len(frame.fusion_materials),
        )
        clause_id = self._runtime_clause_id(effect, frame)
        if self.runtime_coverage is not None and clause_id is not None:
            self.runtime_coverage.record_clause(
                clause_id,
                "condition_evaluated",
            )
            self.runtime_coverage.record_clause(
                clause_id,
                "condition_true" if target_exists else "condition_false",
            )
        branch_ops = (
            effect.then_operations if target_exists else effect.else_operations
        )
        if branch_ops:
            self._queue_effects_from_frame(
                frame,
                branch_ops,
                label=f"{frame.label}/target_exists",
            )

    def _execute_choose_one(self, effect, frame) -> None:
        if (
            effect.choose_count < 1
            or effect.choose_count > len(effect.choose_one_options)
        ):
            raise IllegalCommand(
                "choose_count must be positive and cannot exceed the number of modes"
            )
        legal_options = []
        for opt in effect.choose_one_options:
            if opt.conditions:
                ctx = self._eval_context(
                    frame.controller,
                    source_entity_id=frame.source_entity_id,
                    source_fusion_count=len(frame.fusion_materials),
                )
                result = evaluate_conditions_without_target(opt.conditions, ctx)
                if result is not PartialConditionResult.TRUE:
                    continue
            legal_options.append(opt)
        selection_bonus = sum(
            faith.mode_selection_bonus
            for faith in self.players[frame.controller].faiths
        )
        effective_choose_count = min(
            len(legal_options),
            effect.choose_count + selection_bonus,
        )
        if effective_choose_count < 1:
            return

        if frame.auto_resolve_choices:
            sampled = set(
                self.random.sample(
                    range(len(legal_options)),
                    effective_choose_count,
                )
            )
            chosen = tuple(
                option
                for index, option in enumerate(legal_options)
                if index in sampled
            )
            self._log(
                frame.controller,
                "自动选择：" + "、".join(option.label for option in chosen),
            )
            self._emit(
                GameEvent(
                    EventType.MODE_SELECTED,
                    frame.controller,
                    source_id=frame.source_entity_id,
                    amount=len(chosen),
                    metadata={
                        "source_card_id": frame.source_card_id,
                        "option_ids": tuple(
                            option.option_id for option in chosen
                        ),
                    },
                )
            )
            self._queue_effects_from_frame(
                frame,
                tuple(
                    operation
                    for option in chosen
                    for operation in option.operations
                ),
                label=(
                    f"{frame.label}/choose/"
                    + "+".join(option.label for option in chosen)
                ),
            )
            return

        request_id = self._allocate_choice_request_id()
        frame._decision_meta["choose_one_options"] = legal_options
        self.state.pending_choice = ChoiceRequest(
            player_index=frame.controller,
            prompt=(
                f"{frame.source_name} 选择 {effective_choose_count} 项"
            ),
            options=tuple(
                ChoiceOption(option_id=f"choose_one:{opt.option_id}", label=opt.label)
                for opt in legal_options
            ),
            continuation_id=f"{frame.source_card_id}:{frame.next_index}",
            choice_kind=ChoiceKind.MODE,
            request_id=request_id,
            target_count=effective_choose_count,
        )
        self.state.phase = Phase.AWAITING_CHOICE

    def _execute_optional(self, effect, frame) -> None:
        ops = effect.optional_operations
        if not ops:
            return
        if any(
            op.requires_target
            and not self._has_candidates_for(
                op,
                frame.controller,
                source_entity_id=frame.source_entity_id,
                source_fusion_count=len(frame.fusion_materials),
            )
            for op in ops
        ):
            return
        all_need_target = all(
            self._operation_consumes_target(op)
            for op in ops
        )
        if all_need_target and all(
            not self._has_candidates_for(
                op,
                frame.controller,
                source_entity_id=frame.source_entity_id,
                source_fusion_count=len(frame.fusion_materials),
            )
            for op in ops
        ):
            return

        if frame.auto_resolve_choices:
            self._queue_effects_from_frame(
                frame,
                ops,
                label=f"{frame.label}/optional",
            )
            return

        request_id = self._allocate_choice_request_id()
        prompt = effect.optional_prompt or "\u662f\u5426\u53d1\u52a8\uff1f"
        frame._decision_meta["optional_operations"] = ops
        self.state.pending_choice = ChoiceRequest(
            player_index=frame.controller,
            prompt=prompt,
            options=(
                ChoiceOption(option_id="optional:yes", label="\u53d1\u52a8"),
                ChoiceOption(option_id="optional:no", label="\u4e0d\u53d1\u52a8"),
            ),
            continuation_id=f"{frame.source_card_id}:{frame.next_index}",
            choice_kind=ChoiceKind.CONFIRM,
            request_id=request_id,
        )
        self.state.phase = Phase.AWAITING_CHOICE

    def _tick_emblem_countdowns(self, player_index: int) -> None:
        """Decrement countdowns on emblems at start of controller's turn."""
        player = self.players[player_index]
        expired = []
        for ei in player.emblems:
            if ei.countdown is not None:
                ei.countdown_before = ei.countdown
                ei.countdown -= 1
                self._emit(GameEvent(
                    EventType.EMBLEM_COUNTDOWN_CHANGED, player_index,
                    source_id=ei.entity_id,
                    metadata={
                        "emblem_id": ei.emblem_id,
                        "entity_id": ei.entity_id,
                        "source_card_id": ei.definition.source_card_id,
                        "controller": player_index,
                        "countdown_before": ei.countdown_before,
                        "countdown_after": ei.countdown,
                    },
                ))
                if ei.countdown <= 0:
                    expired.append(ei)
        if not expired:
            return
        batch_id = self._next_emblem_expiration_batch_id
        self._next_emblem_expiration_batch_id += 1
        self._emblem_expiration_batches[batch_id] = {
            "records": [
                (player_index, ei.entity_id)
                for ei in expired
            ],
        }
        self._queue_next_emblem_expiration(batch_id)
        self._continue_effects()

    def _queue_next_emblem_expiration(self, batch_id: int) -> None:
        batch = self._emblem_expiration_batches.get(batch_id)
        if batch is None:
            return
        if self.terminated:
            self._emblem_expiration_batches.pop(batch_id, None)
            return
        records = batch["records"]
        while records:
            player_index, entity_id = records.pop(0)
            player = self.players[player_index]
            ei = next(
                (emblem for emblem in player.emblems if emblem.entity_id == entity_id),
                None,
            )
            if ei is None or ei.countdown is None or ei.countdown > 0:
                continue
            if self._start_emblem_expiration(batch_id, player_index, ei):
                return
        self._emblem_expiration_batches.pop(batch_id, None)

    def _start_emblem_expiration(self, batch_id: int, player_index: int, ei) -> bool:
        player = self.players[player_index]
        if ei not in player.emblems:
            return False
        definition = ei.definition
        self._emit(GameEvent(
            EventType.EMBLEM_EXPIRED, player_index,
            source_id=ei.entity_id,
            metadata={
                "emblem_id": ei.emblem_id,
                "emblem_entity_id": ei.entity_id,
                "owner": player_index,
                "cause": "countdown",
                "countdown_before": ei.countdown_before,
                "countdown_after": ei.countdown,
                "source_card_id": definition.source_card_id,
            },
        ))
        expiration_operations = definition.on_expire + definition.last_words
        if expiration_operations:
            frame = self._queue_effects(
                self._emblem_effect_source_card(definition),
                None,
                expiration_operations,
                controller=player_index,
                label="纹章 到期/谢幕曲",
            )
            frame.emblem_expiration_batch_id = batch_id
            frame.expiring_emblem_owner = player_index
            frame.expiring_emblem_entity_id = ei.entity_id
            return True
        self._remove_emblem_instance(
            player_index,
            ei,
            removal_cause="countdown",
        )
        return False

    def _complete_emblem_expiration(
        self,
        batch_id: int,
        player_index: int | None,
        entity_id: int | None,
    ) -> None:
        if player_index is not None and entity_id is not None:
            player = self.players[player_index]
            ei = next(
                (emblem for emblem in player.emblems if emblem.entity_id == entity_id),
                None,
            )
            if ei is not None:
                self._remove_emblem_instance(
                    player_index,
                    ei,
                    removal_cause="countdown",
                )
        self._queue_next_emblem_expiration(batch_id)

    def _execute_summon(self, effect: EffectOperation, frame: EffectFrame) -> None:
        if effect.card_id is None:
            raise IllegalCommand("SUMMON requires a card_id")
        if self.card_resolver is None:
            raise IllegalCommand("No card_resolver registered for SUMMON")
        card_def = self.card_resolver(effect.card_id)
        if card_def is None:
            raise IllegalCommand(
                f"Card {effect.card_id} not found for SUMMON"
            )
        summon_owner = (
            1 - frame.controller
            if effect.target is TargetKind.ENEMY_LEADER
            else frame.controller
        )
        player = self.players[summon_owner]
        if len(player.board) >= self.config.max_board:
            if effect.target_key:
                self._bind_targets(frame, effect.target_key, (), effect)
            self._log(frame.controller, f"{frame.source_name} 召唤失败：场地已满")
            return
        if card_def.card_type == "随从":
            origin = origin_for_summoned_card(card_def)
            unit = self._summon_follower_to_board(
                summon_owner,
                card_def,
                summon_cause="effect_summon",
                origin=origin,
            )
            if unit is None:
                if effect.target_key:
                    self._bind_targets(frame, effect.target_key, (), effect)
                self._log(
                    frame.controller,
                    f"{frame.source_name} 召唤失败：场地已满",
                )
                return
            if effect.target_key:
                self._bind_targets(
                    frame, effect.target_key, (unit.entity_id,), effect
                )
            self._log(
                frame.controller,
                f"{frame.source_name} 召唤 {card_def.name} ({unit.attack}/{unit.health})",
            )
            self._emit(
                GameEvent(
                    EventType.FOLLOWER_SUMMONED,
                    summon_owner,
                    source_id=unit.entity_id,
                    metadata={
                        "source": unit,
                        "card_id": card_def.card_id,
                        "origin": unit.origin.value,
                        "derived": is_derived(unit.origin),
                        "token": is_token_definition(card_def) or unit.origin is CardOrigin.TOKEN,
                        "via": "effect_summon",
                    },
                )
            )
        elif card_def.card_type == "护符":
            origin = origin_for_summoned_card(card_def)
            amulet = Amulet(
                definition=card_def,
                entity_id=self.state.allocate_entity_id(),
                countdown=self.rulebook.countdown_for(card_def.card_id),
                entered_turn=self.turn,
                origin=origin,
            )
            player.board.append(amulet)
            if effect.target_key:
                self._bind_targets(
                    frame, effect.target_key, (amulet.entity_id,), effect
                )
            self._log(
                frame.controller,
                f"{frame.source_name} 召唤护符 {card_def.name}",
            )
            self._emit(
                GameEvent(
                    EventType.AMULET_ENTERED,
                    summon_owner,
                    source_id=amulet.entity_id,
                    metadata={"source": amulet},
                )
            )
            self._initialize_earth_sigil(amulet, summon_owner)
        else:
            if effect.target_key:
                self._bind_targets(frame, effect.target_key, (), effect)
            self._log(
                frame.controller,
                f"{frame.source_name} 召唤失败：{card_def.card_type} 类型不可召唤",
            )

    def _execute_summon_from_deck(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        """Summon copy-weighted, distinct card names from the controller's deck."""

        if effect.amount <= 0:
            raise IllegalCommand("SUMMON_FROM_DECK requires a positive amount")
        if effect.deck_filter is None or effect.deck_filter.card_type not in {
            "随从",
            "护符",
        }:
            raise IllegalCommand(
                "SUMMON_FROM_DECK requires a follower or amulet deck filter"
            )

        player = self.players[frame.controller]
        available_slots = self.config.max_board - len(player.board)
        if available_slots <= 0:
            if effect.target_key:
                self._bind_targets(frame, effect.target_key, (), effect)
            self._log(frame.controller, f"{frame.source_name} 牌组召唤失败：场地已满")
            return

        candidates = [
            (index, card)
            for index, card in enumerate(player.deck)
            if effect.deck_filter.matches(card)
        ]
        selected: list[tuple[int, CardDefinition | DeckCard]] = []
        while candidates and len(selected) < min(effect.amount, available_slots):
            chosen = self.random.choice(candidates)
            selected.append(chosen)
            chosen_card_name = chosen[1].name
            candidates = [
                candidate
                for candidate in candidates
                if candidate[1].name != chosen_card_name
            ]

        if not selected:
            if effect.target_key:
                self._bind_targets(frame, effect.target_key, (), effect)
            self._log(frame.controller, f"{frame.source_name} 牌组中没有符合条件的卡牌")
            return

        # Remove every chosen physical copy before emitting entry events.  This
        # snapshots one direct-summon timing while leaving same-name copies in deck.
        for deck_index, card in sorted(selected, key=lambda item: item[0], reverse=True):
            removed = player.deck.pop(deck_index)
            if removed is not card:
                raise IllegalCommand("Deck changed during direct-summon selection")

        summon_count = len(selected)
        summoned_entity_ids: list[int] = []
        for summon_index, (_, deck_card) in enumerate(selected):
            card_def = (
                deck_card.definition
                if isinstance(deck_card, DeckCard)
                else deck_card
            )
            metadata = {
                "card_id": card_def.card_id,
                "origin": CardOrigin.DECK.value,
                "derived": False,
                "token": is_token_definition(card_def),
                "via": "deck_summon",
                "source_card_id": frame.source_card_id,
                "deck_summon_index": summon_index,
                "deck_summon_count": summon_count,
            }
            if card_def.card_type == "随从":
                unit = self._summon_follower_to_board(
                    frame.controller,
                    card_def,
                    summon_cause="deck_summon",
                    origin=CardOrigin.DECK,
                )
                if unit is None:
                    raise IllegalCommand(
                        "Board capacity changed during direct-summon resolution"
                    )
                metadata["source"] = unit
                summoned_entity_ids.append(unit.entity_id)
                self._log(
                    frame.controller,
                    f"{frame.source_name} 从牌组召唤 {card_def.name} "
                    f"({unit.attack}/{unit.health})",
                )
                self._emit(
                    GameEvent(
                        EventType.FOLLOWER_SUMMONED,
                        frame.controller,
                        source_id=unit.entity_id,
                        metadata=metadata,
                    )
                )
                continue

            amulet = Amulet(
                definition=card_def,
                entity_id=self.state.allocate_entity_id(),
                countdown=self.rulebook.countdown_for(card_def.card_id),
                entered_turn=self.turn,
                origin=CardOrigin.DECK,
            )
            player.board.append(amulet)
            summoned_entity_ids.append(amulet.entity_id)
            metadata["source"] = amulet
            self._log(
                frame.controller,
                f"{frame.source_name} 从牌组召唤护符 {card_def.name}",
            )
            self._emit(
                GameEvent(
                    EventType.AMULET_ENTERED,
                    frame.controller,
                    source_id=amulet.entity_id,
                    metadata=metadata,
                )
            )
            self._initialize_earth_sigil(amulet, frame.controller)

        if effect.target_key:
            self._bind_targets(
                frame,
                effect.target_key,
                tuple(summoned_entity_ids),
                effect,
            )

    @staticmethod
    def _is_earth_sigil_amulet(entity: BoardCard) -> bool:
        return (
            isinstance(entity, Amulet)
            and AbilityKeyword.EARTH_SIGIL in entity.definition.abilities
        )

    def _earth_sigil_amulets(self, player_index: int) -> list[Amulet]:
        return [
            entity
            for entity in self.players[player_index].board
            if self._is_earth_sigil_amulet(entity)
        ]

    def _initialize_earth_sigil(
        self,
        amulet: Amulet,
        player_index: int,
        *,
        initial_count: int = 1,
    ) -> None:
        if not self._is_earth_sigil_amulet(amulet):
            return
        if initial_count <= 0:
            raise IllegalCommand("Earth Sigil initial count must be positive")

        player = self.players[player_index]
        existing = [
            entity
            for entity in self._earth_sigil_amulets(player_index)
            if entity.entity_id != amulet.entity_id
        ]
        before = sum(entity.earth_sigil_count for entity in existing)
        merged_ids: list[int] = []
        for entity in existing:
            if entity not in player.board:
                continue
            player.board.remove(entity)
            player.banished.append(entity.definition)
            merged_ids.append(entity.entity_id)
            self._emit(
                GameEvent(
                    EventType.CARD_BANISHED,
                    player_index,
                    source_id=entity.entity_id,
                    metadata={"source": entity, "cause": "earth_sigil_merge"},
                )
            )
            self._emit(
                GameEvent(
                    EventType.ENTITY_LEFT_PLAY,
                    player_index,
                    source_id=entity.entity_id,
                    metadata={
                        "source": entity,
                        "definition": entity.definition,
                        "card_id": entity.definition.card_id,
                        "card_type": entity.definition.card_type,
                        "owner": player_index,
                        "cause": "earth_sigil_merge",
                    },
                )
            )

        amulet.earth_sigil_count = before + initial_count
        if merged_ids:
            self._emit(
                GameEvent(
                    EventType.EARTH_SIGILS_MERGED,
                    player_index,
                    source_id=amulet.entity_id,
                    amount=amulet.earth_sigil_count,
                    metadata={
                        "merged_entity_ids": tuple(merged_ids),
                        "earth_sigils_before": before,
                        "earth_sigils_after": amulet.earth_sigil_count,
                    },
                )
            )
        self._emit_earth_sigils_changed(
            player_index,
            source_id=amulet.entity_id,
            before=before,
            after=amulet.earth_sigil_count,
            change="enter",
        )
        self._log(
            player_index,
            f"土之印 {before} → {amulet.earth_sigil_count}",
        )

    def _emit_earth_sigils_changed(
        self,
        player_index: int,
        *,
        source_id: int | None,
        before: int,
        after: int,
        change: str,
        source_card_id: int | None = None,
    ) -> None:
        self._emit(
            GameEvent(
                EventType.EARTH_SIGILS_CHANGED,
                player_index,
                source_id=source_id,
                amount=abs(after - before),
                metadata={
                    "change": change,
                    "earth_sigils_before": before,
                    "earth_sigils_after": after,
                    "source_card_id": source_card_id,
                },
            )
        )

    def _execute_add_earth_sigils(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        amount = effect.amount
        if amount <= 0:
            return
        sigils = self._earth_sigil_amulets(frame.controller)
        if sigils:
            sigil = sigils[0]
            before = sigil.earth_sigil_count
            sigil.earth_sigil_count += amount
            self._emit_earth_sigils_changed(
                frame.controller,
                source_id=sigil.entity_id,
                before=before,
                after=sigil.earth_sigil_count,
                change="gain",
                source_card_id=frame.source_card_id,
            )
            self._log(
                frame.controller,
                f"{frame.source_name} 使土之印 {before} → {sigil.earth_sigil_count}",
            )
            return

        player = self.players[frame.controller]
        if len(player.board) >= self.config.max_board:
            self._log(frame.controller, f"{frame.source_name} 增加土之印失败：战场已满")
            return
        if self.card_resolver is None:
            raise IllegalCommand("No card_resolver registered for Earth Sigil token")
        try:
            token = self.card_resolver(EARTH_SIGIL_TOKEN_CARD_ID)
        except KeyError as exc:
            raise IllegalCommand(
                f"Card {EARTH_SIGIL_TOKEN_CARD_ID} not found for Earth Sigil token"
            ) from exc
        if token is None:
            raise IllegalCommand(
                f"Card {EARTH_SIGIL_TOKEN_CARD_ID} not found for Earth Sigil token"
            )
        if (
            token.card_type != "护符"
            or AbilityKeyword.EARTH_SIGIL not in token.abilities
        ):
            raise IllegalCommand("Earth Sigil token definition is invalid")

        amulet = Amulet(
            definition=token,
            entity_id=self.state.allocate_entity_id(),
            countdown=self.rulebook.countdown_for(token.card_id),
            entered_turn=self.turn,
            origin=CardOrigin.TOKEN,
        )
        player.board.append(amulet)
        self._emit(
            GameEvent(
                EventType.AMULET_ENTERED,
                frame.controller,
                source_id=amulet.entity_id,
                metadata={
                    "source": amulet,
                    "origin": CardOrigin.TOKEN.value,
                    "token": True,
                    "via": "earth_sigil_gain",
                },
            )
        )
        self._initialize_earth_sigil(
            amulet,
            frame.controller,
            initial_count=amount,
        )

    def _execute_earth_rite(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        sigils = self._earth_sigil_amulets(frame.controller)
        if not sigils:
            return
        sigil = sigils[0]
        cost = effect.amount
        before = sigil.earth_sigil_count
        if before < cost:
            return

        sigil.earth_sigil_count -= cost
        after = sigil.earth_sigil_count
        self._emit(
            GameEvent(
                EventType.EARTH_RITE_ACTIVATED,
                frame.controller,
                source_id=frame.source_entity_id,
                amount=cost,
                metadata={
                    "earth_sigils_before": before,
                    "earth_sigils_after": after,
                    "earth_sigil_entity_id": sigil.entity_id,
                    "source_card_id": frame.source_card_id,
                },
            )
        )
        self._emit_earth_sigils_changed(
            frame.controller,
            source_id=sigil.entity_id,
            before=before,
            after=after,
            change="spend",
            source_card_id=frame.source_card_id,
        )
        self._log(frame.controller, f"土之秘术 {cost}：土之印 {before} → {after}")
        if after == 0:
            sigil.pending_destroy = True
        self._queue_effects_from_frame(
            frame,
            effect.earth_rite_operations,
            label="土之秘术",
        )

    def _execute_consume_faith(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.faith_id is None:
            raise IllegalCommand("consume_faith requires faith_id")
        player = self.players[frame.controller]
        instance = next(
            (
                faith
                for faith in player.faiths
                if faith.faith_id == effect.faith_id
            ),
            None,
        )
        cost = effect.amount
        if instance is None or instance.value < cost:
            self._emit(GameEvent(
                EventType.FAITH_CONSUME_FAILED,
                frame.controller,
                source_id=frame.source_entity_id,
                amount=cost,
                metadata={
                    "faith_id": effect.faith_id,
                    "faith_entity_id": (
                        None if instance is None else instance.entity_id
                    ),
                    "faith_value": None if instance is None else instance.value,
                    "source_card_id": frame.source_card_id,
                    "reason": "missing" if instance is None else "insufficient",
                },
            ))
            return

        before = instance.value
        instance.value -= cost
        self._emit(GameEvent(
            EventType.FAITH_CONSUMED,
            frame.controller,
            source_id=frame.source_entity_id,
            target_id=instance.entity_id,
            amount=cost,
            metadata={
                "faith_id": instance.faith_id,
                "faith_value_before": before,
                "faith_value_after": instance.value,
                "source_card_id": frame.source_card_id,
            },
        ))
        self._emit(GameEvent(
            EventType.FAITH_VALUE_CHANGED,
            frame.controller,
            source_id=instance.entity_id,
            amount=-cost,
            metadata={
                "faith_id": instance.faith_id,
                "faith_value_before": before,
                "faith_value_after": instance.value,
                "change": "spend",
                "source_card_id": frame.source_card_id,
            },
        ))
        self._log(
            frame.controller,
            f"信仰 {instance.faith_id} 消费 {cost}：{before} → {instance.value}",
        )
        self._queue_effects_from_frame(
            frame,
            effect.faith_operations,
            label="信仰消费",
        )

    def _execute_random_distribute(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.faith_id is None:
            raise IllegalCommand("random_distribute requires faith_id")
        buckets = effect.random_distribution_operations
        clause_id = self._runtime_clause_id(effect, frame)
        if self.runtime_coverage is not None and clause_id is not None:
            self.runtime_coverage.record_target(
                clause_id,
                "random_distribution_bucket",
                candidate_count=len(buckets),
                random=True,
                no_target=not buckets,
            )
        if len(buckets) < 2 or any(not bucket for bucket in buckets):
            raise IllegalCommand(
                "random_distribute requires at least two non-empty buckets"
            )
        instance = next(
            (
                faith
                for faith in self.players[frame.controller].faiths
                if faith.faith_id == effect.faith_id
            ),
            None,
        )
        total = 0 if instance is None else instance.value
        if total < 0 or total > MAX_RANDOM_DISTRIBUTION_TOTAL:
            raise IllegalCommand(
                "random_distribute total must be between 0 and "
                f"{MAX_RANDOM_DISTRIBUTION_TOTAL}, got {total}"
            )

        counts = [0] * len(buckets)
        for _ in range(total):
            counts[self.random.randrange(len(buckets))] += 1
        self._emit(GameEvent(
            EventType.RANDOM_DISTRIBUTION_RESOLVED,
            frame.controller,
            source_id=frame.source_entity_id,
            target_id=None if instance is None else instance.entity_id,
            amount=total,
            metadata={
                "faith_id": effect.faith_id,
                "faith_entity_id": (
                    None if instance is None else instance.entity_id
                ),
                "source_card_id": frame.source_card_id,
                "bucket_values": tuple(counts),
                "bucket_count": len(buckets),
                "missing_faith": instance is None,
            },
        ))
        self._log(
            frame.controller,
            f"{frame.source_name} 随机分配信仰值 {total}："
            + "/".join(str(value) for value in counts),
        )
        for bucket_index in reversed(range(len(buckets))):
            distributed_value = counts[bucket_index]
            if distributed_value == 0:
                continue
            self._queue_effects_from_frame(
                frame,
                buckets[bucket_index],
                label=f"随机分配:{bucket_index}",
                distributed_value=distributed_value,
            )

    def _execute_random_choice(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        options = effect.random_choice_options
        choice_count = effect.amount
        clause_id = self._runtime_clause_id(effect, frame)
        if (
            len(options) < 2
            or choice_count < 1
            or choice_count > len(options)
            or any(not option.operations for option in options)
        ):
            raise IllegalCommand("random_choice payload is invalid")
        history: dict[str, tuple[int, ...]] | None = None
        previous_indices: tuple[int, ...] = ()
        if effect.random_choice_history_key is not None:
            if (
                frame.emblem_activation_owner is not None
                and frame.emblem_activation_entity_id is not None
            ):
                history_owner = self.players[
                    frame.emblem_activation_owner
                ]
                history_source = next(
                    (
                        emblem
                        for emblem in history_owner.emblems
                        if emblem.entity_id
                        == frame.emblem_activation_entity_id
                    ),
                    None,
                )
                if history_source is not None:
                    history = history_source.random_choice_history
            elif frame.source_entity_id is not None:
                try:
                    history_source = self._find_board_entity(
                        frame.source_entity_id
                    )
                except IllegalCommand:
                    history_source = None
                if isinstance(history_source, Unit):
                    history = history_source.random_choice_history
            if history is not None:
                previous_indices = history.get(
                    effect.random_choice_history_key,
                    (),
                )
        available_indices = tuple(
            index
            for index in range(len(options))
            if index not in previous_indices
        )
        if self.runtime_coverage is not None and clause_id is not None:
            self.runtime_coverage.record_target(
                clause_id,
                "random_option",
                candidate_count=len(available_indices),
                random=True,
                no_target=len(available_indices) < choice_count,
            )
        if len(available_indices) < choice_count:
            self._log(
                frame.controller,
                f"{frame.source_name} 没有尚未发动的随机选项",
            )
            return
        chosen_indices = tuple(
            self.random.sample(available_indices, choice_count)
        )
        if history is not None and effect.random_choice_history_key is not None:
            history[effect.random_choice_history_key] = (
                *previous_indices,
                *chosen_indices,
            )
        chosen = tuple(options[index] for index in chosen_indices)
        self._emit(GameEvent(
            EventType.RANDOM_CHOICES_SELECTED,
            frame.controller,
            source_id=frame.source_entity_id,
            amount=choice_count,
            metadata={
                "source_card_id": frame.source_card_id,
                "option_ids": tuple(option.option_id for option in chosen),
                "option_indices": chosen_indices,
                "option_count": len(options),
                "history_key": effect.random_choice_history_key,
                "previous_option_indices": previous_indices,
                "activated_option_indices": (
                    ()
                    if effect.random_choice_history_key is None
                    else (*previous_indices, *chosen_indices)
                ),
            },
        ))
        self._log(
            frame.controller,
            f"{frame.source_name} 随机发动："
            + "、".join(option.label for option in chosen),
        )
        self._queue_effects_from_frame(
            frame,
            tuple(
                operation
                for option in chosen
                for operation in option.operations
            ),
            label=(
                f"{frame.label}/random/"
                + "+".join(option.option_id for option in chosen)
            ),
        )

    def _execute_grant_faith_ability(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if (
            effect.faith_id is None
            or effect.faith_ability_id is None
            or effect.faith_trigger is None
        ):
            raise IllegalCommand("grant_faith_ability payload is incomplete")
        player = self.players[frame.controller]
        instance = next(
            (
                faith
                for faith in player.faiths
                if faith.faith_id == effect.faith_id
            ),
            None,
        )
        if instance is None:
            return
        stacking = FaithAbilityStacking(effect.faith_stacking)
        if (
            stacking is FaithAbilityStacking.UNIQUE
            and any(
                ability.ability_id == effect.faith_ability_id
                for ability in instance.granted_abilities
            )
        ):
            return
        ability = FaithGrantedAbility(
            ability_id=effect.faith_ability_id,
            trigger=FaithTrigger(effect.faith_trigger),
            operations=effect.faith_operations,
            granted_sequence=instance._next_granted_ability_sequence,
        )
        instance._next_granted_ability_sequence += 1
        instance.granted_abilities.append(ability)
        self._emit(GameEvent(
            EventType.FAITH_ABILITY_GRANTED,
            frame.controller,
            source_id=frame.source_entity_id,
            target_id=instance.entity_id,
            metadata={
                "faith_id": instance.faith_id,
                "ability_id": ability.ability_id,
                "faith_trigger": ability.trigger.value,
                "granted_sequence": ability.granted_sequence,
                "stacking": stacking.value,
                "source_card_id": frame.source_card_id,
            },
        ))

    def _execute_grant_faith_mode_selection_bonus(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.faith_id is None or effect.amount <= 0:
            raise IllegalCommand(
                "grant_faith_mode_selection_bonus payload is incomplete"
            )
        instance = next(
            (
                faith
                for faith in self.players[frame.controller].faiths
                if faith.faith_id == effect.faith_id
            ),
            None,
        )
        if instance is None:
            return
        before = instance.mode_selection_bonus
        instance.mode_selection_bonus += effect.amount
        self._emit(
            GameEvent(
                EventType.FAITH_MODE_SELECTION_BONUS_GRANTED,
                frame.controller,
                source_id=frame.source_entity_id,
                target_id=instance.entity_id,
                amount=effect.amount,
                metadata={
                    "faith_id": instance.faith_id,
                    "source_card_id": frame.source_card_id,
                    "before": before,
                    "after": instance.mode_selection_bonus,
                },
            )
        )

    def _execute_banish(self, target_id: int | None, frame: EffectFrame) -> None:
        entity = self._find_board_entity(target_id)
        owner = self._entity_owner(entity.entity_id)
        self._banish_board_entity(entity, owner)

    def _execute_banish_same_name(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        snapshot = self._bound_snapshot_for_effect(effect, frame, target_id)
        if snapshot is None:
            return
        opponent = 1 - frame.controller
        matching_ids = tuple(
            entity.entity_id
            for entity in self.players[opponent].board
            if isinstance(entity, Unit)
            and entity.definition.name == snapshot.card_name
        )
        for entity_id in matching_ids:
            try:
                self._execute_banish(entity_id, frame)
            except IllegalCommand:
                continue

    def _banishes_on_leave(self, entity: BoardCard) -> bool:
        return (
            self.rulebook.banish_on_leave(entity.definition.card_id)
            and not (
                isinstance(entity, Unit)
                and entity.printed_abilities_removed
            )
        )

    def _banish_board_entity(
        self,
        entity: BoardCard,
        owner: int,
        *,
        replaced_leave_cause: DeathCause | None = None,
    ) -> None:
        player = self.players[owner]
        if entity not in player.board:
            return
        player.board.remove(entity)
        player.banished.append(entity.definition)
        self._log(
            owner,
            f"{entity.definition.name} 被消失",
        )
        banish_metadata = {"source": entity}
        if replaced_leave_cause is not None:
            banish_metadata["replaced_leave_cause"] = replaced_leave_cause.value
        self._emit(
            GameEvent(
                EventType.CARD_BANISHED,
                owner,
                source_id=entity.entity_id,
                metadata=banish_metadata,
            )
        )
        leave_metadata = {
            "source": entity,
            "definition": entity.definition,
            "card_id": entity.definition.card_id,
            "card_type": entity.definition.card_type,
            "owner": owner,
            "cause": DeathCause.BANISH.value,
        }
        if replaced_leave_cause is not None:
            leave_metadata["replaced_leave_cause"] = replaced_leave_cause.value
        self._emit(
            GameEvent(
                EventType.ENTITY_LEFT_PLAY,
                owner,
                source_id=entity.entity_id,
                metadata=leave_metadata,
            )
        )

    def _execute_add_card(self, effect: EffectOperation, frame: EffectFrame) -> None:
        if effect.card_id is None:
            raise IllegalCommand("ADD_CARD requires a card_id")
        if self.card_resolver is None:
            raise IllegalCommand("No card_resolver registered for ADD_CARD")
        card_def = self.card_resolver(effect.card_id)
        if card_def is None:
            raise IllegalCommand(
                f"Card {effect.card_id} not found for ADD_CARD"
            )
        player = self.players[frame.controller]
        if len(player.hand) >= self.config.max_hand:
            if effect.target_key:
                self._bind_targets(frame, effect.target_key, (), effect)
            self._log(
                frame.controller,
                f"{frame.source_name} 加牌失败：手牌已满，{card_def.name} 被弃置",
            )
            origin = origin_for_added_card(card_def)
            self._send_to_graveyard(
                frame.controller, card_def, "hand_full",
                derived=is_derived(origin), origin=origin,
                token=is_token_definition(card_def) or origin is CardOrigin.TOKEN,
            )
            return
        origin = origin_for_added_card(card_def)
        added = self._append_hand_card(player, card_def, origin=origin)
        if effect.target_key:
            self._bind_targets(
                frame,
                effect.target_key,
                (added.entity_id,),
                effect,
                snapshots=(
                    self._bound_hand_snapshot(frame.controller, added),
                ),
            )
        if effect.mode is not None:
            added.cost_modifiers.append(
                CostModifier(
                    modifier_id=self._allocate_modifier_id(),
                    mode=effect.mode.value,
                    amount=effect.amount,
                    duration=effect.duration.value,
                    expires_for_player=_expires_for_player(
                        effect.duration,
                        frame.controller,
                        self.state.active_player,
                    ),
                )
            )
        self._log(
            frame.controller,
            f"{frame.source_name} 将 {card_def.name} 加入手牌",
        )
        self._emit(
            GameEvent(
                EventType.CARD_ADDED_TO_HAND,
                frame.controller,
                metadata={
                    "card_id": card_def.card_id,
                    "entity_id": added.entity_id,
                    "card": card_def,
                    "origin": origin.value,
                    "derived": is_derived(origin),
                    "token": is_token_definition(card_def) or origin is CardOrigin.TOKEN,
                    "cost_after": added.current_cost,
                },
            )
        )

    def _execute_add_card_to_deck(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        if effect.card_id is None:
            raise IllegalCommand("ADD_CARD_TO_DECK requires a card_id")
        if self.card_resolver is None:
            raise IllegalCommand(
                "No card_resolver registered for ADD_CARD_TO_DECK"
            )
        card_def = self.card_resolver(effect.card_id)
        if card_def is None:
            raise IllegalCommand(
                f"Card {effect.card_id} not found for ADD_CARD_TO_DECK"
            )
        player = self.players[frame.controller]
        insert_pos = self.random.randint(0, len(player.deck))
        player.deck.insert(insert_pos, card_def)
        self._log(
            frame.controller,
            f"{frame.source_name} 将 {card_def.name} 加入牌组",
        )
        self._emit(
            GameEvent(
                EventType.CARD_ADDED_TO_DECK,
                frame.controller,
                source_id=frame.source_entity_id,
                metadata={
                    "card_id": card_def.card_id,
                    "card": card_def,
                    "source_card_id": frame.source_card_id,
                    "derived": True,
                },
            )
        )

    def _execute_return_to_hand(
        self, target_id: int | None, frame: EffectFrame
    ) -> None:
        entity = self._find_board_entity(target_id)
        owner_index = self._entity_owner(entity.entity_id)
        owner = self.players[owner_index]
        if entity not in owner.board:
            return
        if self._banishes_on_leave(entity):
            self._banish_board_entity(
                entity,
                owner_index,
                replaced_leave_cause=DeathCause.RETURN_TO_HAND,
            )
            return
        owner.board.remove(entity)
        card_def = entity.definition
        board_origin = entity.origin
        board_source_origin = entity.source_origin
        if len(owner.hand) < self.config.max_hand:
            self._append_hand_card(
                owner,
                card_def,
                origin=board_origin,
                source_origin=board_source_origin,
                fused_material_ids=tuple(entity.fused_material_ids),
            )
            self._log(
                owner_index,
                f"{card_def.name} 返回手牌",
            )
            self._emit(
                GameEvent(
                    EventType.CARD_RETURNED_TO_HAND,
                    owner_index,
                    source_id=entity.entity_id,
                    metadata={"source": entity},
                )
            )
        else:
            owner.banished.append(card_def)
            self._log(
                owner_index,
                f"{card_def.name} 返回手牌失败（手牌已满），被消失",
            )
            self._emit(
                GameEvent(
                    EventType.CARD_BANISHED,
                    owner_index,
                    source_id=entity.entity_id,
                    metadata={"source": entity},
                )
            )
        self._emit(
            GameEvent(
                EventType.ENTITY_LEFT_PLAY,
                owner_index,
                source_id=entity.entity_id,
                metadata={
                    "source": entity,
                    "definition": card_def,
                    "card_id": card_def.card_id,
                    "card_type": card_def.card_type,
                    "owner": owner_index,
                    "cause": DeathCause.RETURN_TO_HAND.value,
                },
            )
        )

    def _execute_return_to_deck(
        self, target_id: int | None, frame: EffectFrame
    ) -> None:
        if target_id is None:
            return
        player = self.players[frame.controller]
        for idx, eid in enumerate(list(player.hand_entity_ids)):
            if eid == target_id:
                hand_card = player.hand.pop(idx)
                player.hand_entity_ids.pop(idx)
                card_def = (
                    hand_card.definition
                    if isinstance(hand_card, HandCard)
                    else hand_card
                )
                insert_pos = self.random.randint(0, len(player.deck))
                player.deck.insert(insert_pos, card_def)
                self._log(
                    frame.controller,
                    f"{card_def.name} 返回牌组",
                )
                self._emit(
                    GameEvent(
                        EventType.CARD_RETURNED_TO_DECK,
                        frame.controller,
                        source_id=target_id,
                        metadata={"source": card_def},
                    )
                )
                return
        entity = self._find_board_entity(target_id)
        owner_index = self._entity_owner(entity.entity_id)
        owner = self.players[owner_index]
        if entity not in owner.board:
            return
        if self._banishes_on_leave(entity):
            self._banish_board_entity(
                entity,
                owner_index,
                replaced_leave_cause=DeathCause.RETURN_TO_DECK,
            )
            return
        owner.board.remove(entity)
        card_def = entity.definition
        insert_pos = self.random.randint(0, len(owner.deck))
        owner.deck.insert(insert_pos, card_def)
        self._log(
            owner_index,
            f"{card_def.name} 返回牌组",
        )
        self._emit(
            GameEvent(
                EventType.CARD_RETURNED_TO_DECK,
                owner_index,
                source_id=entity.entity_id,
                metadata={"source": entity},
            )
        )
        self._emit(
            GameEvent(
                EventType.ENTITY_LEFT_PLAY,
                owner_index,
                source_id=entity.entity_id,
                metadata={
                    "source": entity,
                    "definition": card_def,
                    "card_id": card_def.card_id,
                    "card_type": card_def.card_type,
                    "owner": owner_index,
                    "cause": DeathCause.RETURN_TO_DECK.value,
                },
            )
        )

    def _execute_distribute_damage(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
    ) -> None:
        """Apply the official oldest-first damage-distribution procedure."""

        remaining = max(0, effect.amount)
        if remaining == 0:
            return
        target_player_index = 1 - frame.controller
        followers = [
            entity
            for entity in self.players[target_player_index].board
            if isinstance(entity, Unit)
        ]
        allocations: list[tuple[Unit, int]] = []
        for index, follower in enumerate(followers):
            if remaining <= 0:
                break
            is_last_follower = index == len(followers) - 1
            amount = (
                min(remaining, max(0, follower.health))
                if effect.include_leader or not is_last_follower
                else remaining
            )
            if amount > 0:
                allocations.append((follower, amount))
                remaining -= amount

        for follower, amount in allocations:
            self.apply_damage(
                None,
                follower,
                amount,
                DamageType.EFFECT,
                frame.controller,
            )
            self._log(
                frame.controller,
                f"{frame.source_name} {frame.label}向"
                f"{follower.definition.name}分配 {amount} 点伤害",
            )

        if effect.include_leader and remaining > 0:
            self.apply_damage(
                None,
                None,
                remaining,
                DamageType.EFFECT,
                frame.controller,
                target_player_index=target_player_index,
            )
            self._log(
                frame.controller,
                f"{frame.source_name} {frame.label}向对方主战者分配 "
                f"{remaining} 点伤害（生命 "
                f"{self.players[target_player_index].health}）",
            )

    def _execute_reduce_countdown(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        self._execute_countdown_change(
            effect,
            frame,
            target_id,
            delta=-effect.amount,
        )

    def _execute_increase_countdown(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        self._execute_countdown_change(
            effect,
            frame,
            target_id,
            delta=effect.amount,
        )

    def _execute_countdown_change(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
        *,
        delta: int,
    ) -> None:
        if target_id is None or effect.amount <= 0:
            return
        try:
            target = self._find_board_entity(target_id)
        except IllegalCommand:
            target = None
        if target is None:
            emblem_owner = next(
                (
                    owner
                    for owner in (0, 1)
                    if any(
                        emblem.entity_id == target_id
                        for emblem in self.players[owner].emblems
                    )
                ),
                None,
            )
            if emblem_owner is None:
                return
            emblem = next(
                candidate
                for candidate in self.players[emblem_owner].emblems
                if candidate.entity_id == target_id
            )
            if emblem.countdown is None:
                return
            previous = emblem.countdown
            emblem.countdown_before = previous
            emblem.countdown = max(0, previous + delta)
            self._emit(GameEvent(
                EventType.EMBLEM_COUNTDOWN_CHANGED,
                emblem_owner,
                source_id=emblem.entity_id,
                amount=emblem.countdown - previous,
                metadata={
                    "emblem_id": emblem.emblem_id,
                    "entity_id": emblem.entity_id,
                    "source_card_id": emblem.definition.source_card_id,
                    "controller": emblem_owner,
                    "countdown_before": previous,
                    "countdown_after": emblem.countdown,
                    "effect_source_card_id": frame.source_card_id,
                    "effect_source_entity_id": frame.source_entity_id,
                },
            ))
            self._log(
                frame.controller,
                f"纹章 {emblem.emblem_id} 倒数由 {previous} "
                f"变为 {emblem.countdown}",
            )
            if emblem.countdown == 0:
                batch_id = self._next_emblem_expiration_batch_id
                self._next_emblem_expiration_batch_id += 1
                self._emblem_expiration_batches[batch_id] = {
                    "records": [(emblem_owner, emblem.entity_id)],
                }
                self._queue_next_emblem_expiration(batch_id)
            return
        if (
            not isinstance(target, Amulet)
            or target.countdown is None
            or target.pending_destroy
        ):
            return
        previous = target.countdown
        target.countdown = max(0, previous + delta)
        owner = self._entity_owner(target.entity_id)
        self._emit(
            GameEvent(
                EventType.AMULET_COUNTDOWN_CHANGED,
                owner,
                source_id=target.entity_id,
                amount=target.countdown - previous,
                metadata={
                    "source": target,
                    "card_id": target.definition.card_id,
                    "countdown_before": previous,
                    "countdown_after": target.countdown,
                    "effect_source_card_id": frame.source_card_id,
                    "effect_source_entity_id": frame.source_entity_id,
                },
            )
        )
        self._log(
            frame.controller,
            f"{target.definition.name} 倒数由 {previous} 变为 {target.countdown}",
        )
        if target.countdown == 0:
            target.pending_destroy = True

    def _execute_discard(self, target_id: int | None, frame: EffectFrame) -> None:
        player = self.players[frame.controller]
        if target_id is None:
            return
        for idx, eid in enumerate(player.hand_entity_ids):
            if eid == target_id:
                hand_card = player.hand[idx]
                card_def = (
                    hand_card.definition
                    if isinstance(hand_card, HandCard)
                    else hand_card
                )
                discarded_cost = (
                    hand_card.current_cost
                    if isinstance(hand_card, HandCard)
                    else hand_card.cost
                )
                discard_origin = hand_card.origin if isinstance(hand_card, HandCard) else CardOrigin.DECK
                player.hand.pop(idx)
                player.hand_entity_ids.pop(idx)
                discarded_eid = target_id
                self._send_to_graveyard(
                    frame.controller, card_def, "discard",
                    source_entity_id=discarded_eid,
                    derived=is_derived(discard_origin),
                    origin=discard_origin,
                    token=is_token_definition(card_def) or discard_origin is CardOrigin.TOKEN,
                )
                self._log(
                    frame.controller,
                    f"{frame.source_name} 弃置手牌 {card_def.name}",
                )
                self._emit(
                    GameEvent(
                        EventType.CARD_DISCARDED,
                        frame.controller,
                        source_id=discarded_eid,
                        metadata={
                            "card_id": card_def.card_id,
                            "card": card_def,
                            "source": hand_card,
                            "origin": discard_origin.value,
                            "cost": discarded_cost,
                        },
                    )
                )
                discarded_operations = self.rulebook.operations_for(
                    card_def.card_id,
                    Trigger.DISCARDED,
                )
                if discarded_operations:
                    self._queue_effects(
                        card_def,
                        discarded_eid,
                        discarded_operations,
                        controller=frame.controller,
                        label="被弃置",
                        source_spellboost_count=(
                            hand_card.spellboost_count
                            if isinstance(hand_card, HandCard)
                            else 0
                        ),
                        source_cost=discarded_cost,
                    )
                return

    @staticmethod
    def _event_counts_as_evolution(event: GameEvent) -> bool:
        return (
            event.type is EventType.FOLLOWER_EVOLVED
            or (
                event.type is EventType.FOLLOWER_SUPER_EVOLVED
                and bool(event.metadata.get("counts_as_evolution", True))
            )
        )

    def _emblem_trigger_for_event(self, event: GameEvent) -> str | None:
        trigger = {
            EventType.CARD_PLAYED: "card_played",
            EventType.CARD_DRAWN: "card_drawn",
            EventType.CARD_FUSED: "card_fused",
            EventType.FOLLOWER_SUMMONED: "follower_summoned",
            EventType.FOLLOWER_EVOLVED: "follower_evolved",
            EventType.FOLLOWER_DESTROYED: "follower_destroyed",
            EventType.AMULET_DESTROYED: "amulet_destroyed",
            EventType.DEATH_BATCH_END: "death_batch_end",
            EventType.LEADER_HEALED: "leader_healed",
            EventType.AMULET_ACTIVATED: "amulet_activated",
            EventType.ATTACK_DECLARED: "attack_declared",
        }.get(event.type)
        if (
            event.type is EventType.FOLLOWER_SUPER_EVOLVED
            and self._event_counts_as_evolution(event)
        ):
            return "follower_evolved"
        return trigger

    def _resolve_event_queue(self) -> None:
        if self._card_listener_batch_active():
            return
        if self._suspended_event_state is not None and self.state.pending_choice is None:
            if self._event_trigger_batch_active():
                return
            self._resume_event_queue()
            return

        event_to_ability = {
            EventType.CARD_PLAYED: AbilityEvent.CARD_PLAYED,
            EventType.FOLLOWER_SUMMONED: AbilityEvent.FOLLOWER_SUMMONED,
            EventType.FOLLOWER_EVOLVED: AbilityEvent.FOLLOWER_EVOLVED,
            EventType.FOLLOWER_SUPER_EVOLVED: AbilityEvent.FOLLOWER_SUPER_EVOLVED,
            EventType.ATTACK_DECLARED: AbilityEvent.BEFORE_ATTACK,
            EventType.COMBAT_STARTED: AbilityEvent.BEFORE_COMBAT,
            EventType.DAMAGE_DEALT: AbilityEvent.AFTER_DAMAGE,
            EventType.FOLLOWER_DESTROYED: AbilityEvent.FOLLOWER_DESTROYED,
        }
        while self.state.event_queue:
            self._step()
            event = self.state.event_queue.popleft()
            self._record_event(event)
            if self.terminated:
                continue
            if event.type is EventType.FOLLOWER_DESTROYED:
                self._resolve_super_evolution_attack_bonus(event)
                if self.terminated:
                    continue
            if self._event_is_enhanced_card_play(event):
                self._advance_faiths_for_event(
                    event.player_index,
                    FaithTrigger.CARD_ENHANCED,
                    event,
                )
            if event.type is EventType.FOLLOWER_SUMMONED:
                self._advance_faiths_for_event(
                    event.player_index,
                    FaithTrigger.FOLLOWER_SUMMONED,
                    event,
                )
            counts_as_evolution = self._event_counts_as_evolution(event)
            if counts_as_evolution:
                self._advance_faiths_for_event(
                    event.player_index,
                    FaithTrigger.FOLLOWER_EVOLVED,
                    event,
                )
            if event.type is EventType.AMULET_DESTROYED:
                self._advance_faiths_for_event(
                    event.player_index,
                    FaithTrigger.AMULET_DESTROYED,
                    event,
                )
            if event.type is EventType.MODE_SELECTED:
                self._advance_faiths_for_event(
                    event.player_index,
                    FaithTrigger.MODE_SELECTED,
                    event,
                )
            ability_event = event_to_ability.get(event.type)
            trigger_keywords = event.metadata.get("trigger_abilities") is not False
            if (
                not trigger_keywords
                and event.type not in {
                    EventType.FOLLOWER_EVOLVED,
                    EventType.FOLLOWER_SUPER_EVOLVED,
                }
            ):
                ability_event = None
            if (
                event.type is EventType.FOLLOWER_SUMMONED
                and event.metadata.get("via") != "play"
            ):
                ability_event = None
            source = event.metadata.get("source")
            if source is None:
                source = event.metadata.get("definition")
            target = event.metadata.get("target")
            if self.state.pending_choice is not None:
                self._save_event_continuation(
                    event,
                    source,
                    target,
                    ability_event,
                    phase="faith_done",
                )
                return
            if ability_event is not None and source is not None:
                if isinstance(source, Unit):
                    self._dispatch_ability(
                        ability_event,
                        source,
                        target if isinstance(target, Unit) else None,
                        player_index=event.player_index,
                        trigger_keywords=trigger_keywords,
                        counts_as_evolution=counts_as_evolution,
                    )
                elif hasattr(source, "card_id"):
                    self._dispatch_card_ability(
                        ability_event,
                        source,
                        player_index=event.player_index,
                    )
                if self.state.pending_choice is not None:
                    self._save_event_continuation(event, source, target, ability_event, phase="source_done")
                    return

                if event.type is EventType.COMBAT_STARTED and isinstance(target, Unit):
                    self._dispatch_ability(
                        ability_event,
                        target,
                        source,
                        player_index=1 - event.player_index,
                    )
                    if self.state.pending_choice is not None:
                        self._save_event_continuation(event, source, target, ability_event, phase="target_done")
                        return
            self._dispatch_card_listeners(event)
            if self.state.pending_choice is not None:
                self._save_event_continuation(
                    event,
                    source,
                    target,
                    ability_event,
                    phase="listeners_done",
                )
                return

            emblem_trigger = self._emblem_trigger_for_event(event)
            if emblem_trigger is not None:
                self._dispatch_emblem_triggers(
                    event.player_index,
                    emblem_trigger,
                    event_player=event.player_index,
                    source_id=event.source_id,
                    event_metadata=event.metadata,
                    eligible_sources=event.emblem_sources,
                )
                if self.state.pending_choice is not None:
                    self._save_event_continuation(
                        event,
                        source,
                        target,
                        ability_event,
                        phase="emblems_done",
                    )
                    return

    def _event_is_enhanced_card_play(self, event: GameEvent) -> bool:
        if event.type is not EventType.CARD_PLAYED:
            return False
        card_id = event.metadata.get("card_id")
        mode_id = event.metadata.get("mode_id")
        if (
            not isinstance(card_id, int)
            or isinstance(card_id, bool)
            or not isinstance(mode_id, str)
        ):
            return False
        return any(
            mode.mode_id == mode_id and mode.is_enhance
            for mode in self.rulebook.modes_for(card_id)
        )

    def _resolve_super_evolution_attack_bonus(
        self,
        event: GameEvent,
    ) -> None:
        context = self._active_super_evolution_attack
        if (
            context is None
            or context.bonus_resolved
            or event.source_id != context.target_id
        ):
            return
        context.bonus_resolved = True
        target_player = 1 - context.controller
        self._emit(
            GameEvent(
                EventType.SUPER_EVOLUTION_ATTACK_BONUS,
                context.controller,
                source_id=context.attacker_id,
                target_id=_leader_target_id(target_player),
                amount=1,
                metadata={
                    "attacker_card_id": context.attacker_card_id,
                    "destroyed_follower_id": context.target_id,
                },
            )
        )
        self.apply_damage(
            None,
            None,
            1,
            DamageType.EFFECT,
            context.controller,
            target_player_index=target_player,
        )
        self._log(
            context.controller,
            f"{context.attacker_name} 的超进化攻击规则对对方主战者造成 1 点伤害"
            f"（生命 {self.players[target_player].health}）",
        )
        self._check_game_over()

    def _save_event_continuation(self, event, source, target, ability_event, phase):
        remaining = []
        while self.state.event_queue:
            remaining.append(self.state.event_queue.popleft())
        self._suspended_event_state = {
            "remaining_events": remaining,
            "event": event,
            "source": source,
            "target": target,
            "ability_event": ability_event,
            "phase": phase,
        }

    def _resume_event_queue(self):
        state = self._suspended_event_state
        self._suspended_event_state = None
        phase = state["phase"]
        event = state["event"]
        source = state["source"]
        target = state["target"]
        ability_event = state["ability_event"]

        if phase == "faith_done":
            if ability_event is not None and source is not None:
                if isinstance(source, Unit):
                    self._dispatch_ability(
                        ability_event,
                        source,
                        target if isinstance(target, Unit) else None,
                        player_index=event.player_index,
                        trigger_keywords=(
                            event.metadata.get("trigger_abilities") is not False
                        ),
                        counts_as_evolution=self._event_counts_as_evolution(event),
                    )
                elif hasattr(source, "card_id"):
                    self._dispatch_card_ability(
                        ability_event,
                        source,
                        player_index=event.player_index,
                    )
                if self.state.pending_choice is not None:
                    state["phase"] = "source_done"
                    self._suspended_event_state = state
                    return
            phase = "source_done"

        if phase == "source_done" and event.type is EventType.COMBAT_STARTED and isinstance(target, Unit):
            self._dispatch_ability(
                ability_event,
                target,
                source,
                player_index=1 - event.player_index,
            )
            if self.state.pending_choice is not None:
                state["phase"] = "target_done"
                self._suspended_event_state = state
                return

        if phase in {"source_done", "target_done"}:
            self._dispatch_card_listeners(event)
            if self.state.pending_choice is not None:
                state["phase"] = "listeners_done"
                self._suspended_event_state = state
                return

        if phase in {"source_done", "target_done", "listeners_done"}:
            emblem_trigger = self._emblem_trigger_for_event(event)
            if emblem_trigger is not None:
                self._dispatch_emblem_triggers(
                    event.player_index,
                    emblem_trigger,
                    event_player=event.player_index,
                    source_id=event.source_id,
                    event_metadata=event.metadata,
                    eligible_sources=event.emblem_sources,
                )
                if self.state.pending_choice is not None:
                    state["phase"] = "emblems_done"
                    self._suspended_event_state = state
                    return

        remaining = state["remaining_events"]
        for e in remaining:
            self.state.event_queue.append(e)
        self._resolve_event_queue()

    def _event_trigger_batch_active(self) -> bool:
        if self._suspended_event_state is None:
            return False
        return any(
            frame.emblem_batch_id is not None
            or frame.listener_batch_id is not None
            for frame in self.state.effect_stack
        )

    def _begin_last_words_deferral(self) -> None:
        if self._defer_last_words or self._deferred_death_batches:
            raise RuntimeError("last-words deferral is already active")
        self._defer_last_words = True

    def _abort_last_words_deferral(self) -> None:
        self._defer_last_words = False
        self._deferred_death_batches.clear()

    def _flush_deferred_death_batches(self) -> bool:
        while self._suspended_batch is not None or self._deferred_death_batches:
            if self._suspended_batch is None:
                batch, lw_records = self._deferred_death_batches.pop(0)
                self._suspended_batch = batch
                self._suspended_lw_records = list(lw_records)
            self._stabilize()
            if self.state.pending_choice is not None:
                return False
            if self._suspended_batch is not None:
                raise RuntimeError(
                    "deferred last-words batch did not make progress"
                )
        return True

    def _stabilize(self) -> None:
        if self._stabilizing:
            return
        if self.state.pending_choice is not None:
            return
        if self._event_trigger_batch_active():
            return
        self._stabilizing = True
        try:
            if self._suspended_record is not None:
                self._resume_death_batch()
            while self._suspended_batch is not None:
                self._continue_batch_lws()
                if self.state.pending_choice is not None:
                    return
            self._do_stabilize()
        finally:
            self._stabilizing = False

    def _resume_death_batch(self) -> None:
        record = self._suspended_record
        self._suspended_record = None
        batch = self._suspended_batch
        metadata = (
            self._last_words_event_metadata(batch, record)
            if batch is not None
            else {"card_id": record.card_id}
        )
        self._emit(GameEvent(
            EventType.LAST_WORDS_COMPLETE,
            record.owner,
            source_id=record.entity_id,
            metadata=metadata,
        ))

    def _continue_batch_lws(self) -> None:
        batch = self._suspended_batch
        lw_records = self._suspended_lw_records
        while lw_records:
            record = lw_records[0]
            self._suspended_lw_records = lw_records[1:]
            parent_effect_depth = len(self.state.effect_stack)
            self._execute_last_words(record, batch)
            self._continue_effects(stop_at_depth=parent_effect_depth)
            if self.state.pending_choice is not None:
                self._suspended_record = record
                self._suspended_batch = batch
                self._stabilizing = False
                return
            self._emit(GameEvent(
                EventType.LAST_WORDS_COMPLETE,
                record.owner,
                source_id=record.entity_id,
                metadata=self._last_words_event_metadata(batch, record),
            ))
            lw_records = self._suspended_lw_records
        self._emit(GameEvent(
            EventType.DEATH_BATCH_END,
            self.current_player,
            metadata=self._death_batch_event_metadata(batch),
        ))
        self._resolve_event_queue()
        self._suspended_batch = None

    def _do_stabilize(self) -> None:
        while True:
            self._step()
            batch, banish_replacements = self._collect_death_batch()
            if not batch.records:
                if banish_replacements:
                    self._resolve_event_queue()
                    if self.state.pending_choice is not None:
                        return
                    continue
                break
            self.state.death_queue.append(batch)
            self._emit(
                GameEvent(
                    EventType.DEATH_BATCH_START,
                    self.current_player,
                    metadata=self._death_batch_event_metadata(batch),
                )
            )
            ordered_records = self._death_batch_ordered_records(batch)

            for record in ordered_records:
                player = self.players[record.owner]
                event_metadata = self._death_event_metadata(batch, record)
                if record.card_type == "护符":
                    self._log(record.owner, f"护符 {record.card_name} 被破坏")
                    self._emit(GameEvent(
                        EventType.AMULET_DESTROYED, record.owner,
                        source_id=record.entity_id,
                        metadata={
                            **event_metadata,
                            "definition": record.definition,
                        },
                    ))
                else:
                    player.followers_destroyed_this_turn += 1
                    self._log(record.owner, f"随从 {record.card_name} 被破坏")
                    self._emit(GameEvent(
                        EventType.FOLLOWER_DESTROYED, record.owner,
                        source_id=record.entity_id,
                        metadata={
                            **event_metadata,
                            "definition": record.definition,
                        },
                    ))
                self._emit(GameEvent(
                    EventType.ENTITY_LEFT_PLAY, record.owner,
                    source_id=record.entity_id,
                    metadata={
                        **event_metadata,
                        "definition": record.definition,
                    },
                ))

            lw_records = [r for r in ordered_records if r.allows_last_words]
            self._resolve_event_queue()
            if self.terminated:
                return
            if self.state.pending_choice is not None:
                if self._defer_last_words:
                    self._deferred_death_batches.append(
                        (batch, list(lw_records))
                    )
                else:
                    self._suspended_batch = batch
                    self._suspended_lw_records = list(lw_records)
                return

            if self._defer_last_words:
                self._deferred_death_batches.append(
                    (batch, list(lw_records))
                )
                continue

            self._suspended_batch = batch
            self._suspended_lw_records = list(lw_records)
            self._continue_batch_lws()
            if self.state.pending_choice is not None:
                return

        self._check_game_over()

    def _collect_death_batch(self) -> tuple[DeathBatch, int]:
        records: list[DeathRecord] = []
        banish_replacements = 0
        batch_id = len(self.state.death_queue) + 1

        for player_index, player in enumerate(self.players):
            for pos, entity in enumerate(tuple(player.board)):
                if isinstance(entity, Unit) and entity.health <= 0:
                    cause = self._death_causes.pop(entity.entity_id, DeathCause.ZERO_HEALTH)
                    if self._banishes_on_leave(entity):
                        self._banish_board_entity(
                            entity,
                            player_index,
                            replaced_leave_cause=cause,
                        )
                        banish_replacements += 1
                        continue
                    record = DeathRecord(
                        owner=player_index,
                        entity_id=entity.entity_id,
                        card_id=entity.definition.card_id,
                        card_name=entity.definition.name,
                        card_type="随从",
                        definition=entity.definition,
                        cause=cause,
                        board_position=pos,
                        allows_last_words=not (
                            entity.printed_abilities_removed
                            or entity.last_words_removed
                        ),
                        effective_keywords=entity.effective_keywords,
                        attack=entity.attack,
                        health=entity.health,
                        evolved=entity.evolved,
                        super_evolved=entity.super_evolved,
                        granted_last_words=tuple(
                            operation
                            for granted_ability in entity.granted_last_words
                            for operation in granted_ability
                        ),
                    )
                    unit_origin = entity.origin
                    unit_source_origin = entity.source_origin
                    unit_derived = is_derived(unit_origin)
                    unit_token = (
                        is_token(entity.definition, unit_origin)
                        or unit_source_origin is CardOrigin.TOKEN
                    )
                    player.board.remove(entity)
                    self._send_to_graveyard(
                        player_index, entity.definition, cause.value, entity.entity_id,
                        derived=unit_derived, origin=unit_origin, token=unit_token,
                        source_origin=unit_source_origin,
                    )
                    self._record_destroyed_follower(
                        player_index, entity.definition, cause,
                        derived=unit_derived, token=unit_token,
                        origin=unit_origin,
                        source_origin=unit_source_origin,
                    )
                    records.append(record)
                elif isinstance(entity, Amulet) and entity.pending_destroy:
                    if self._is_earth_sigil_amulet(entity) and entity.earth_sigil_count == 0:
                        cause = DeathCause.EARTH_SIGIL_DEPLETED
                    else:
                        cause = DeathCause.COUNTDOWN_EXPIRED if entity.countdown is not None and entity.countdown <= 0 else DeathCause.EFFECT_DESTROY
                    record = DeathRecord(
                        owner=player_index,
                        entity_id=entity.entity_id,
                        card_id=entity.definition.card_id,
                        card_name=entity.definition.name,
                        card_type="护符",
                        definition=entity.definition,
                        cause=cause,
                        board_position=pos,
                        allows_last_words=True,
                        effective_keywords=frozenset(),
                        attack=None,
                        health=None,
                        evolved=False,
                        super_evolved=False,
                    )
                    amulet_origin = entity.origin
                    amulet_source_origin = entity.source_origin
                    amulet_derived = is_derived(amulet_origin)
                    amulet_token = (
                        is_token(entity.definition, amulet_origin)
                        or amulet_source_origin is CardOrigin.TOKEN
                    )
                    player.board.remove(entity)
                    self._send_to_graveyard(
                        player_index, entity.definition, cause.value, entity.entity_id,
                        derived=amulet_derived, origin=amulet_origin, token=amulet_token,
                        source_origin=amulet_source_origin,
                    )
                    destroyed_mode = next(
                        (
                            mode
                            for mode in self.rulebook.modes_for(
                                entity.definition.card_id
                            )
                            if mode.mode_id == entity.play_mode_id
                        ),
                        None,
                    )
                    self._record_destroyed_amulet(
                        player_index,
                        entity.definition,
                        cause,
                        derived=amulet_derived,
                        token=amulet_token,
                        origin=amulet_origin,
                        source_origin=amulet_source_origin,
                        play_mode_id=entity.play_mode_id,
                        summon_countdown=(
                            destroyed_mode.countdown
                            if destroyed_mode is not None
                            else self.rulebook.countdown_for(
                                entity.definition.card_id
                            )
                        ),
                    )
                    records.append(record)

        return DeathBatch(records=records, batch_id=batch_id), banish_replacements

    def _last_words_order_key(self, record: DeathRecord) -> tuple[int, int, int]:
        active = self.state.active_player
        return (0 if record.owner == active else 1, record.owner, record.board_position)

    def _death_batch_ordered_records(self, batch: DeathBatch) -> list[DeathRecord]:
        return sorted(batch.records, key=self._last_words_order_key)

    def _death_record_order_index(self, batch: DeathBatch, record: DeathRecord) -> int:
        for index, candidate in enumerate(self._death_batch_ordered_records(batch)):
            if candidate.entity_id == record.entity_id:
                return index
        return -1

    def _death_batch_composition(self, batch: DeathBatch) -> dict[str, object]:
        owner_counts = []
        total_followers = 0
        total_amulets = 0
        for owner in (0, 1):
            owner_records = [record for record in batch.records if record.owner == owner]
            follower_count = sum(
                1 for record in owner_records
                if record.card_type != "护符"
            )
            amulet_count = sum(
                1 for record in owner_records
                if record.card_type == "护符"
            )
            total_followers += follower_count
            total_amulets += amulet_count
            owner_counts.append({
                "owner": owner,
                "record_count": len(owner_records),
                "follower_count": follower_count,
                "amulet_count": amulet_count,
            })
        return {
            "follower_count": total_followers,
            "amulet_count": total_amulets,
            "owner_counts": owner_counts,
        }

    def _death_batch_event_metadata(self, batch: DeathBatch) -> dict[str, object]:
        ordered_records = self._death_batch_ordered_records(batch)
        composition = self._death_batch_composition(batch)
        return {
            "batch_id": batch.batch_id,
            "count": len(batch.records),
            "batch_record_count": len(batch.records),
            "active_player": self.state.active_player,
            **composition,
            "ordered_records": [
                self._death_record_order_summary(batch, record)
                for record in ordered_records
            ],
        }

    def _death_record_order_summary(
        self,
        batch: DeathBatch,
        record: DeathRecord,
    ) -> dict[str, object]:
        return {
            "batch_order_index": self._death_record_order_index(batch, record),
            "owner": record.owner,
            "entity_id": record.entity_id,
            "card_id": record.card_id,
            "card_type": record.card_type,
            "board_position": record.board_position,
            "cause": record.cause.value,
            "allows_last_words": record.allows_last_words,
            "keywords": tuple(sorted(record.effective_keywords)),
        }

    def _death_event_metadata(
        self,
        batch: DeathBatch,
        record: DeathRecord,
    ) -> dict[str, object]:
        composition = self._death_batch_composition(batch)
        return {
            "card_id": record.card_id,
            "cause": record.cause.value,
            "batch_id": batch.batch_id,
            "batch_order_index": self._death_record_order_index(batch, record),
            "batch_record_count": len(batch.records),
            "batch_follower_count": composition["follower_count"],
            "batch_amulet_count": composition["amulet_count"],
            "batch_owner_counts": composition["owner_counts"],
            "active_player": self.state.active_player,
            "owner": record.owner,
            "card_type": record.card_type,
            "board_position": record.board_position,
            "keywords": tuple(sorted(record.effective_keywords)),
            "attack": record.attack,
            "health": record.health,
            "evolved": record.evolved,
            "super_evolved": record.super_evolved,
        }

    def _last_words_event_metadata(
        self,
        batch: DeathBatch,
        record: DeathRecord,
    ) -> dict[str, object]:
        return self._death_event_metadata(batch, record)

    @staticmethod
    def _source_snapshot_from_death_record(
        record: DeathRecord,
    ) -> SourceStateSnapshot:
        return SourceStateSnapshot(
            entity_id=record.entity_id,
            controller=record.owner,
            card_id=record.card_id,
            card_type=record.card_type,
            attack=record.attack,
            health=record.health,
            evolved=record.evolved,
            super_evolved=record.super_evolved,
            effective_keywords=record.effective_keywords,
        )

    def _execute_last_words(self, record: DeathRecord, batch: DeathBatch) -> None:
        self._step()
        self._emit(GameEvent(
            EventType.LAST_WORDS_START,
            record.owner,
            source_id=record.entity_id,
            metadata=self._last_words_event_metadata(batch, record),
        ))
        self._log(record.owner, f"{record.card_name} 谢幕曲开始")

        operations = (
            self.rulebook.operations_for(record.card_id, Trigger.LAST_WORDS)
            + record.granted_last_words
        )
        if not operations and record.card_type == "护符":
            operations = self.rulebook.operations_for(record.card_id, Trigger.COUNTDOWN_EXPIRED)

        if operations:
            self._queue_effects(
                record.definition,
                record.entity_id,
                operations,
                controller=record.owner,
                label="谢幕曲",
                source_snapshot=self._source_snapshot_from_death_record(record),
            )

    def _check_game_over(self) -> None:
        dead = [index for index, player in enumerate(self.players) if player.health <= 0]
        if dead:
            self.state.winner = None if len(dead) == 2 else 1 - dead[0]
            self.state.phase = Phase.FINISHED
        elif self.config.max_turns is not None and self.turn > self.config.max_turns:
            health = [player.health for player in self.players]
            self.state.winner = (
                None if health[0] == health[1] else int(health[1] > health[0])
            )
            self.state.phase = Phase.FINISHED
        if self.terminated and (
            not self.logs or not self.logs[-1].startswith("=== 对局结束")
        ):
            result = (
                "平局"
                if self.winner is None
                else f"玩家 {self.winner + 1} 获胜"
            )
            if self.config.retain_text_logs:
                self.logs.append(f"=== 对局结束：{result} ===")
            self._emit(
                GameEvent(
                    EventType.GAME_ENDED,
                    self.current_player,
                    metadata={"winner": self.winner},
                )
            )
            self._resolve_event_queue()

    def assert_invariants(self) -> None:
        """Raise IllegalCommand if the mutable game state is internally invalid."""
        state = self.state
        if len(state.players) != 2:
            raise IllegalCommand("Invariant failed: game must have exactly two players")
        if state.active_player not in (0, 1):
            raise IllegalCommand(
                f"Invariant failed: active_player out of range: {state.active_player}"
            )
        if state.first_player not in (0, 1):
            raise IllegalCommand(
                f"Invariant failed: first_player out of range: {state.first_player}"
            )
        if (
            len(state.mulligan_completed) != 2
            or any(not isinstance(value, bool) for value in state.mulligan_completed)
        ):
            raise IllegalCommand(
                "Invariant failed: mulligan completion state is invalid"
            )
        if state.turn < 1:
            raise IllegalCommand(f"Invariant failed: turn must be positive: {state.turn}")
        if state.next_entity_id <= 0:
            raise IllegalCommand(
                f"Invariant failed: next_entity_id must be positive: {state.next_entity_id}"
            )
        if state.winner not in (None, 0, 1):
            raise IllegalCommand(f"Invariant failed: invalid winner {state.winner!r}")
        if state.terminated:
            if state.phase is not Phase.FINISHED:
                raise IllegalCommand("Invariant failed: terminated state must be FINISHED")
        elif state.phase is Phase.FINISHED:
            raise IllegalCommand("Invariant failed: FINISHED phase without termination")

        attack_context = self._active_super_evolution_attack
        if attack_context is not None:
            if attack_context.controller not in (0, 1):
                raise IllegalCommand(
                    "Invariant failed: super-evolution attack controller out of range"
                )
            for field_name, value in (
                ("attacker_id", attack_context.attacker_id),
                ("target_id", attack_context.target_id),
                ("attacker_card_id", attack_context.attacker_card_id),
            ):
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value <= 0
                ):
                    raise IllegalCommand(
                        "Invariant failed: super-evolution attack "
                        f"{field_name} must be positive"
                    )
            if not isinstance(attack_context.bonus_resolved, bool):
                raise IllegalCommand(
                    "Invariant failed: super-evolution attack bonus flag invalid"
                )

        if state.pending_choice is not None:
            request = state.pending_choice
            expected_choice_phase = (
                Phase.MULLIGAN
                if request.continuation_id == "match_mulligan"
                else Phase.AWAITING_CHOICE
            )
            if state.phase is not expected_choice_phase:
                raise IllegalCommand(
                    "Invariant failed: pending_choice phase mismatch"
                )
            if request.player_index not in (0, 1):
                raise IllegalCommand(
                    f"Invariant failed: choice player out of range: {request.player_index}"
                )
            if not request.options:
                raise IllegalCommand("Invariant failed: pending choice has no options")
            if request.request_id < 0:
                raise IllegalCommand(
                    f"Invariant failed: negative choice request_id {request.request_id}"
                )
            if not isinstance(request.choice_kind, ChoiceKind):
                raise IllegalCommand(
                    f"Invariant failed: invalid choice_kind {request.choice_kind!r}"
                )
            if (
                not isinstance(request.target_count, int)
                or isinstance(request.target_count, bool)
                or request.target_count <= 0
            ):
                raise IllegalCommand(
                    "Invariant failed: pending choice target_count must be positive"
                )
            if not isinstance(request.allow_duplicate_targets, bool):
                raise IllegalCommand(
                    "Invariant failed: pending choice duplicate-target policy must be boolean"
                )
            if not isinstance(request.selected_options, tuple):
                raise IllegalCommand(
                    "Invariant failed: pending choice selected_options must be a tuple"
                )
            if (
                request.choice_kind is not ChoiceKind.FUSION
                and len(request.selected_options) >= request.target_count
            ):
                raise IllegalCommand(
                    "Invariant failed: completed multi-target choice is still pending"
                )
            if any(
                not isinstance(selected, ChoiceOption)
                for selected in request.selected_options
            ):
                raise IllegalCommand(
                    "Invariant failed: pending choice selected_options contains invalid value"
                )
            selected_option_ids = [
                selected.option_id for selected in request.selected_options
            ]
            if (
                not request.allow_duplicate_targets
                and len(selected_option_ids) != len(set(selected_option_ids))
            ):
                raise IllegalCommand(
                    "Invariant failed: pending choice contains duplicate selected targets"
                )
            if (
                not request.allow_duplicate_targets
                and set(selected_option_ids)
                & {candidate.option_id for candidate in request.options}
            ):
                raise IllegalCommand(
                    "Invariant failed: selected target remains available when duplicates are forbidden"
                )
            if (
                not request.allow_duplicate_targets
                and len(request.selected_options) + len(request.options)
                < request.target_count
            ):
                raise IllegalCommand(
                    "Invariant failed: pending choice cannot reach target_count"
                )
            for selected_index, selected in enumerate(request.selected_options):
                zone = f"pending_choice selected_options[{selected_index}]"
                if not isinstance(selected, ChoiceOption):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} is not ChoiceOption"
                    )
                if not isinstance(selected.option_id, str) or not selected.option_id:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} has empty option_id"
                    )
                if selected.entity_id is not None and selected.entity_id <= 0:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} has non-positive entity_id"
                    )
                if selected.entity_id is not None:
                    matching_prefix = next(
                        (
                            prefix
                            for prefix in ("entity:", "hand:")
                            if selected.option_id.startswith(prefix)
                        ),
                        None,
                    )
                    if matching_prefix is not None:
                        raw_target_id = selected.option_id[len(matching_prefix):]
                        try:
                            selected_target_id = int(raw_target_id)
                        except ValueError as exc:
                            raise IllegalCommand(
                                f"Invariant failed: {zone} has malformed entity id"
                            ) from exc
                        if selected_target_id != selected.entity_id:
                            raise IllegalCommand(
                                f"Invariant failed: {zone} option_id mismatch"
                            )
                if (
                    selected.leader_player_index is not None
                    and selected.leader_player_index not in (0, 1)
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} leader_player_index out of range"
                    )
                if (
                    selected.entity_id is not None
                    and selected.leader_player_index is not None
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} cannot target both entity and leader"
                    )
                if selected.leader_player_index is not None:
                    expected_option_id = f"leader:{selected.leader_player_index}"
                    if selected.option_id != expected_option_id:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} leader option_id mismatch"
                        )
            seen_option_ids: set[str] = set()
            for option_index, option in enumerate(request.options):
                zone = f"pending_choice option[{option_index}]"
                if not isinstance(option.option_id, str) or not option.option_id:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} has empty option_id"
                    )
                if option.option_id in seen_option_ids:
                    raise IllegalCommand(
                        f"Invariant failed: duplicate choice option_id {option.option_id!r}"
                    )
                seen_option_ids.add(option.option_id)
                if option.entity_id is not None and option.entity_id <= 0:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} has non-positive entity_id "
                        f"{option.entity_id}"
                    )
                if option.entity_id is not None:
                    for prefix in ("entity:", "hand:"):
                        if not option.option_id.startswith(prefix):
                            continue
                        raw_target_id = option.option_id[len(prefix):]
                        try:
                            option_target_id = int(raw_target_id)
                        except ValueError as exc:
                            raise IllegalCommand(
                                f"Invariant failed: {zone} {prefix} option_id "
                                f"has malformed entity id {raw_target_id!r}"
                            ) from exc
                        if option_target_id != option.entity_id:
                            raise IllegalCommand(
                                f"Invariant failed: {zone} {prefix} option_id mismatch: "
                                f"{option_target_id} != {option.entity_id}"
                            )
                        break
                if (
                    option.leader_player_index is not None
                    and option.leader_player_index not in (0, 1)
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} leader_player_index out of range: "
                        f"{option.leader_player_index}"
                    )
                if option.entity_id is not None and option.leader_player_index is not None:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} cannot target both entity and leader"
                    )
                if option.leader_player_index is not None:
                    expected_option_id = f"leader:{option.leader_player_index}"
                    if option.option_id != expected_option_id:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} leader option_id mismatch: "
                            f"{option.option_id!r} != {expected_option_id!r}"
                        )
        elif state.phase in (Phase.AWAITING_CHOICE, Phase.MULLIGAN):
            raise IllegalCommand(
                "Invariant failed: decision phase without pending_choice"
            )

        for key, count in state.listener_activation_counts.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 3
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < (0 if index == 2 else 1)
                    for index, value in enumerate(key)
                )
            ):
                raise IllegalCommand(
                    "Invariant failed: invalid card-listener activation key"
                )
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
            ):
                raise IllegalCommand(
                    "Invariant failed: card-listener activation count must be positive"
                )
        if not state.listener_once_per_turn_used.issubset(
            state.listener_activation_counts
        ):
            raise IllegalCommand(
                "Invariant failed: card-listener turn limit lacks activation count"
            )
        for batch_id, batch in self._listener_batches.items():
            if (
                not isinstance(batch_id, int)
                or isinstance(batch_id, bool)
                or batch_id <= 0
                or not isinstance(batch, dict)
                or not isinstance(batch.get("records"), list)
            ):
                raise IllegalCommand(
                    "Invariant failed: invalid card-listener batch"
                )
        if (
            not isinstance(self._next_listener_batch_id, int)
            or isinstance(self._next_listener_batch_id, bool)
            or self._next_listener_batch_id <= max(
                self._listener_batches,
                default=0,
            )
        ):
            raise IllegalCommand(
                "Invariant failed: next card-listener batch id is not ahead"
            )

        seen_entities: dict[int, str] = {}

        def remember(entity_id: int, zone: str) -> None:
            if entity_id <= 0:
                raise IllegalCommand(
                    f"Invariant failed: {zone} has non-positive entity_id {entity_id}"
                )
            previous = seen_entities.get(entity_id)
            if previous is not None:
                raise IllegalCommand(
                    f"Invariant failed: entity_id {entity_id} appears in both "
                    f"{previous} and {zone}"
                )
            seen_entities[entity_id] = zone

        for player_index, player in enumerate(state.players):
            prefix = f"player {player_index + 1}"
            fusion_material_ids = [
                material.entity_id for material in player.fusion_materials
            ]
            if len(fusion_material_ids) != len(set(fusion_material_ids)):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} has duplicate fusion material ids"
                )
            fusion_sequences = [
                material.consumed_sequence for material in player.fusion_materials
            ]
            if len(fusion_sequences) != len(set(fusion_sequences)):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} has duplicate fusion sequences"
                )
            if player._next_fusion_sequence <= max(fusion_sequences, default=0):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} next fusion sequence is not ahead"
                )
            fusion_material_id_set = set(fusion_material_ids)
            for material_index, material in enumerate(player.fusion_materials):
                zone = f"{prefix} fusion_materials[{material_index}]"
                if material.owner != player_index:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} owner mismatch"
                    )
                if material.consumed_sequence <= 0:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} consumed_sequence must be positive"
                    )
                if material.fused_into_entity_id <= 0:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} fused target id must be positive"
                    )
                if any(
                    inherited not in fusion_material_id_set
                    for inherited in material.inherited_material_ids
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} inherited material is missing"
                    )
                remember(material.entity_id, zone)
            if len(player.hand) != len(player.hand_entity_ids):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} hand/entity_id length mismatch"
                )
            if len(player.hand) > self.config.max_hand:
                raise IllegalCommand(
                    f"Invariant failed: {prefix} hand exceeds max_hand"
                )
            if len(player.board) > self.config.max_board:
                raise IllegalCommand(
                    f"Invariant failed: {prefix} board exceeds max_board"
                )
            if (
                len(player.faiths) + len(player.emblems)
                > self.config.leader_area_limit
            ):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} leader area exceeds limit"
                )
            if not (0 <= player.max_mana <= self.config.max_mana):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} max_mana out of range: {player.max_mana}"
                )
            if not (0 <= player.mana <= self._effective_mana_cap(player)):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} mana out of range: {player.mana}/{player.max_mana}"
                )
            if (
                not isinstance(player.extra_pp_available, bool)
                or not isinstance(player.extra_pp_refresh_done, bool)
                or not isinstance(player.extra_pp_uses, int)
                or isinstance(player.extra_pp_uses, bool)
                or not 0 <= player.extra_pp_uses <= 2
                or (
                    player.extra_pp_active_turn is not None
                    and (
                        not isinstance(player.extra_pp_active_turn, int)
                        or isinstance(player.extra_pp_active_turn, bool)
                        or player.extra_pp_active_turn != self.turn
                    )
                )
            ):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} extra PP state is invalid"
                )
            if (
                player_index == state.first_player
                and (
                    player.extra_pp_available
                    or player.extra_pp_uses
                    or player.extra_pp_refresh_done
                    or player.extra_pp_active_turn is not None
                )
            ):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} first player has extra PP"
                )
            if player.max_health < 1:
                raise IllegalCommand(
                    f"Invariant failed: {prefix} max_health is below 1: "
                    f"{player.max_health}"
                )
            if player.health < 0 or player.health > player.max_health:
                raise IllegalCommand(
                    f"Invariant failed: {prefix} health out of range: "
                    f"{player.health}/{player.max_health}"
                )
            if (
                isinstance(player.leader_barrier_charges, bool)
                or not isinstance(player.leader_barrier_charges, int)
                or player.leader_barrier_charges < 0
            ):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} leader barrier charges are invalid"
                )
            modifier_ids = [
                modifier.modifier_id
                for modifier in player.leader_damage_modifiers
            ]
            if len(modifier_ids) != len(set(modifier_ids)):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} has duplicate leader damage modifier ids"
                )
            for modifier in player.leader_damage_modifiers:
                if modifier.modifier_id <= 0:
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} leader damage modifier id is not positive"
                    )
                if modifier.duration == ModifierDuration.WHILE_SOURCE_IN_PLAY.value:
                    if (
                        modifier.source_controller not in (0, 1)
                        or modifier.source_entity_id is None
                        or modifier.source_entity_id <= 0
                        or modifier.source_card_id is None
                        or modifier.source_card_id <= 0
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {prefix} source-bound leader modifier is invalid"
                        )
                if modifier.mode not in {mode.value for mode in LeaderDamageMode}:
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} leader damage modifier "
                        "mode is invalid"
                    )
            for name, value in (
                ("fatigue", player.fatigue),
                ("evolution_points", player.evolution_points),
                ("super_evolution_points", player.super_evolution_points),
                ("turns_started", player.turns_started),
                ("followers_evolved_this_match", player.followers_evolved_this_match),
                ("cards_played_this_turn", player.cards_played_this_turn),
                ("follower_attacks_this_turn", player.follower_attacks_this_turn),
                ("followers_destroyed_this_turn", player.followers_destroyed_this_turn),
                ("cooperation", player.cooperation),
                ("shadows", player.shadows),
            ):
                if value < 0:
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} {name} is negative: {value}"
                    )
            if not isinstance(player.empty_deck_outcome, EmptyDeckOutcome):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} empty_deck_outcome is invalid: "
                    f"{player.empty_deck_outcome!r}"
                )

            for deck_index, deck_entry in enumerate(player.deck):
                zone = f"{prefix} deck[{deck_index}]"
                if isinstance(deck_entry, CardDefinition):
                    continue
                if not isinstance(deck_entry, DeckCard):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} is not a card definition or DeckCard"
                    )
                modifier_ids = [
                    modifier.modifier_id
                    for modifier in deck_entry.cost_modifiers
                ]
                if (
                    len(modifier_ids) != len(set(modifier_ids))
                    or any(modifier_id <= 0 for modifier_id in modifier_ids)
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} has invalid cost modifier ids"
                    )
                for modifier in deck_entry.cost_modifiers:
                    if modifier.mode not in {
                        mode.value for mode in CostChangeMode
                    }:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} has invalid cost modifier mode"
                        )
                    if (
                        modifier.duration != ModifierDuration.PERMANENT.value
                        or modifier.expires_for_player is not None
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} has non-permanent cost modifier"
                        )
                deck_stat_modifier_ids = [
                    modifier.modifier_id
                    for modifier in deck_entry.stat_modifiers
                ]
                if (
                    len(deck_stat_modifier_ids)
                    != len(set(deck_stat_modifier_ids))
                    or any(
                        modifier_id <= 0
                        for modifier_id in deck_stat_modifier_ids
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} has invalid stat modifier ids"
                    )
                if (
                    deck_entry.stat_modifiers
                    and deck_entry.card_type != "随从"
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} has stat modifiers but is "
                        "not a follower"
                    )
                for modifier in deck_entry.stat_modifiers:
                    if (
                        modifier.duration
                        != ModifierDuration.PERMANENT.value
                        or modifier.expires_for_player is not None
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} has non-permanent stat modifier"
                        )

            for hand_index, hand_card in enumerate(player.hand):
                if not isinstance(hand_card, HandCard):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] is not HandCard"
                    )
                expected = player.hand_entity_ids[hand_index]
                if hand_card.entity_id != expected:
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] "
                        f"entity_id {hand_card.entity_id} != hand_entity_ids {expected}"
                    )
                remember(hand_card.entity_id, f"{prefix} hand[{hand_index}]")
                hand_cost_modifier_ids = [
                    modifier.modifier_id
                    for modifier in hand_card.cost_modifiers
                ]
                if (
                    len(hand_cost_modifier_ids)
                    != len(set(hand_cost_modifier_ids))
                    or any(
                        modifier_id <= 0
                        for modifier_id in hand_cost_modifier_ids
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has "
                        "invalid cost modifier ids"
                    )
                for modifier in hand_card.cost_modifiers:
                    if modifier.mode not in {
                        mode.value for mode in CostChangeMode
                    } or modifier.duration not in {
                        duration.value for duration in ModifierDuration
                    }:
                        raise IllegalCommand(
                            f"Invariant failed: {prefix} hand[{hand_index}] has "
                            "invalid cost modifier"
                        )
                hand_stat_modifier_ids = [
                    modifier.modifier_id
                    for modifier in hand_card.stat_modifiers
                ]
                if len(hand_stat_modifier_ids) != len(set(hand_stat_modifier_ids)):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has "
                        "duplicate stat modifier ids"
                    )
                if any(modifier_id <= 0 for modifier_id in hand_stat_modifier_ids):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has a "
                        "non-positive stat modifier id"
                    )
                if hand_card.stat_modifiers and hand_card.card_type != "随从":
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has "
                        "stat modifiers but is not a follower"
                    )
                runtime_keyword_values = (
                    set(hand_card.permanent_keywords)
                    | set(hand_card.removed_keywords)
                    | {
                        modifier.keyword
                        for modifier in hand_card.temporary_keywords
                    }
                    | {
                        modifier.keyword
                        for modifier in hand_card.temporary_keyword_removals
                    }
                )
                if not hand_card.printed_keyword_overrides.issubset(
                    RUNTIME_UNIT_KEYWORDS
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has "
                        "an unsupported printed keyword override"
                    )
                if runtime_keyword_values and hand_card.card_type != "随从":
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has "
                        "runtime keywords but is not a follower"
                    )
                if not runtime_keyword_values.issubset(RUNTIME_UNIT_KEYWORDS):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has "
                        "an unsupported runtime keyword"
                    )
                if hand_card.permanent_keywords & hand_card.removed_keywords:
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has "
                        "contradictory permanent keyword state"
                    )
                if not isinstance(hand_card.effect_destroy_immunity, bool):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has "
                        "invalid effect-destroy-immunity state"
                    )
                if (
                    hand_card.granted_last_words
                    or hand_card.effect_destroy_immunity
                ) and hand_card.card_type != "随从":
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has "
                        "follower-only granted abilities"
                    )
                if (
                    not isinstance(hand_card.granted_last_words, list)
                    or any(
                        not isinstance(granted_ability, tuple)
                        or not granted_ability
                        or any(
                            not isinstance(operation, EffectOperation)
                            for operation in granted_ability
                        )
                        for granted_ability in hand_card.granted_last_words
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has "
                        "invalid granted last words"
                    )
                for modifier in (
                    *hand_card.temporary_keywords,
                    *hand_card.temporary_keyword_removals,
                ):
                    if modifier.duration not in {
                        duration.value for duration in ModifierDuration
                    } or modifier.duration == ModifierDuration.PERMANENT.value:
                        raise IllegalCommand(
                            f"Invariant failed: {prefix} hand[{hand_index}] has "
                            "an invalid temporary keyword duration"
                        )
                if len(hand_card.fused_material_ids) != len(set(hand_card.fused_material_ids)):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has duplicate fused materials"
                    )
                if (
                    not isinstance(hand_card.evolutions_while_in_hand, int)
                    or isinstance(hand_card.evolutions_while_in_hand, bool)
                    or hand_card.evolutions_while_in_hand < 0
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has invalid "
                        "evolutions_while_in_hand"
                    )
                if (
                    not isinstance(hand_card.union_burst_gauge_bonus, int)
                    or isinstance(hand_card.union_burst_gauge_bonus, bool)
                    or hand_card.union_burst_gauge_bonus < 0
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has invalid "
                        "union_burst_gauge_bonus"
                    )
                if any(
                    material_id not in fusion_material_id_set
                    for material_id in hand_card.fused_material_ids
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] references missing fusion material"
                    )
                if (
                    hand_card.fusion_used_turn is not None
                    and (
                        hand_card.fusion_used_turn <= 0
                        or hand_card.fusion_used_turn > state.turn
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} hand[{hand_index}] has invalid fusion turn"
                    )

            for board_index, entity in enumerate(player.board):
                zone = f"{prefix} board[{board_index}]"
                if not isinstance(entity, (Unit, Amulet)):
                    raise IllegalCommand(f"Invariant failed: {zone} is not a board entity")
                remember(entity.entity_id, zone)
                if len(entity.fused_material_ids) != len(set(entity.fused_material_ids)):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} has duplicate fused materials"
                    )
                if any(
                    material_id not in fusion_material_id_set
                    for material_id in entity.fused_material_ids
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} references missing fusion material"
                    )
                if isinstance(entity, Unit):
                    if not isinstance(entity.printed_abilities_removed, bool):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} ability-removal flag is invalid"
                        )
                    if not isinstance(entity.last_words_removed, bool):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} last-words-removal flag is invalid"
                        )
                    if not isinstance(entity.effect_destroy_immunity, bool):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} effect-destroy-immunity "
                            "flag is invalid"
                        )
                    if (
                        not isinstance(entity.granted_last_words, list)
                        or any(
                            not isinstance(granted_ability, tuple)
                            or not granted_ability
                            or any(
                                not isinstance(operation, EffectOperation)
                                for operation in granted_ability
                            )
                            for granted_ability in entity.granted_last_words
                        )
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} has invalid granted last words"
                        )
                    if (
                        not isinstance(entity.turn_end_destroy_timings, set)
                        or any(
                            not isinstance(timing, TurnEndDestroyTiming)
                            for timing in entity.turn_end_destroy_timings
                        )
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} has invalid turn-end "
                            "destroy timings"
                        )
                    if (
                        not isinstance(entity.turn_end_banish_timings, set)
                        or any(
                            not isinstance(timing, TurnEndDestroyTiming)
                            for timing in entity.turn_end_banish_timings
                        )
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} has invalid turn-end "
                            "banish timings"
                        )
                    if (
                        not isinstance(
                            entity.granted_turn_end_abilities,
                            list,
                        )
                        or any(
                            not isinstance(
                                ability,
                                GrantedTurnEndAbility,
                            )
                            or not isinstance(
                                ability.timing,
                                TurnEndDestroyTiming,
                            )
                            or not isinstance(ability.operations, tuple)
                            or not ability.operations
                            or any(
                                not isinstance(
                                    operation,
                                    EffectOperation,
                                )
                                for operation in ability.operations
                            )
                            for ability in entity.granted_turn_end_abilities
                        )
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} has invalid granted "
                            "turn-end abilities"
                        )
                    if (
                        not isinstance(entity.random_choice_history, dict)
                        or any(
                            not isinstance(key, str)
                            or not key
                            or not isinstance(indices, tuple)
                            or len(indices) != len(set(indices))
                            or any(
                                isinstance(index, bool)
                                or not isinstance(index, int)
                                or index < 0
                                for index in indices
                            )
                            for key, indices
                            in entity.random_choice_history.items()
                        )
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} has invalid random "
                            "choice history"
                        )
                    if entity.attack < 0:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} attack is negative"
                        )
                    if entity.max_health < 1:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} max_health is below 1"
                        )
                    if entity.health <= 0:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} has non-positive health"
                        )
                    if entity.health > entity.max_health:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} health exceeds max_health"
                        )
                    if entity.attacks_remaining < 0:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} attacks_remaining is negative"
                        )
                    if entity.attacks_remaining > entity.attacks_per_turn:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} attacks_remaining exceeds capacity"
                        )
                    for modifier in entity.attack_capacity_modifiers:
                        if modifier.attacks_per_turn < 1:
                            raise IllegalCommand(
                                f"Invariant failed: {zone} attack capacity is not positive"
                            )
                        if modifier.duration not in {
                            duration.value for duration in ModifierDuration
                        }:
                            raise IllegalCommand(
                                f"Invariant failed: {zone} attack capacity duration is invalid"
                            )
                    if entity.super_evolved:
                        if entity.super_evolved_turn is None:
                            raise IllegalCommand(
                                f"Invariant failed: {zone} super_evolved without turn stamp"
                            )
                        if entity.super_evolved_turn <= 0:
                            raise IllegalCommand(
                                f"Invariant failed: {zone} super_evolved_turn is not positive"
                            )
                        if entity.super_evolved_turn > state.turn:
                            raise IllegalCommand(
                                f"Invariant failed: {zone} super_evolved_turn is in the future"
                            )
                    elif entity.super_evolved_turn is not None:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} super_evolved_turn without super_evolved"
                        )
                    if entity.barrier_charges < 0:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} barrier_charges is negative"
                        )
                else:
                    if entity.countdown is not None and entity.countdown < 0:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} countdown is negative"
                        )
                    if entity.play_mode_id is not None and (
                        not isinstance(entity.play_mode_id, str)
                        or not entity.play_mode_id
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} play_mode_id is invalid"
                        )
                    if entity.activated_turn is not None and (
                        not isinstance(entity.activated_turn, int)
                        or isinstance(entity.activated_turn, bool)
                        or entity.activated_turn <= 0
                        or entity.activated_turn > state.turn
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} activated_turn is invalid"
                        )
                    is_earth_sigil = self._is_earth_sigil_amulet(entity)
                    if entity.earth_sigil_count < 0:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} earth_sigil_count is negative"
                        )
                    if not is_earth_sigil and entity.earth_sigil_count != 0:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} non-Earth-Sigil amulet "
                            "has earth_sigil_count"
                        )
                    if (
                        is_earth_sigil
                        and entity.earth_sigil_count == 0
                        and not entity.pending_destroy
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} Earth Sigil has zero count "
                            "without pending destruction"
                        )

            earth_sigil_amulets = [
                entity
                for entity in player.board
                if self._is_earth_sigil_amulet(entity)
            ]
            if len(earth_sigil_amulets) > 1:
                raise IllegalCommand(
                    f"Invariant failed: {prefix} has multiple Earth Sigil amulets"
                )

            for grave_index, graveyard_card in enumerate(player.graveyard):
                zone = f"{prefix} graveyard[{grave_index}]"
                if graveyard_card.owner != player_index:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} owner mismatch"
                    )
                if graveyard_card.entered_sequence <= 0:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} entered_sequence must be positive"
                    )
                remember(graveyard_card.entity_id, zone)

            emblem_ids: set[str] = set()
            for emblem_index, emblem in enumerate(player.emblems):
                zone = f"{prefix} emblems[{emblem_index}]"
                if emblem.controller != player_index:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} controller mismatch"
                    )
                if emblem.created_sequence <= 0:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} created_sequence must be positive"
                    )
                if emblem.countdown is not None and emblem.countdown < 0:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} countdown is negative"
                    )
                if emblem.emblem_id in emblem_ids:
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} has duplicate emblem "
                        f"{emblem.emblem_id!r}"
                    )
                emblem_ids.add(emblem.emblem_id)
                if (
                    not isinstance(emblem.random_choice_history, dict)
                    or any(
                        not isinstance(key, str)
                        or not key
                        or not isinstance(indices, tuple)
                        or len(indices) != len(set(indices))
                        or any(
                            isinstance(index, bool)
                            or not isinstance(index, int)
                            or index < 0
                            for index in indices
                        )
                        for key, indices
                        in emblem.random_choice_history.items()
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} has invalid random choice history"
                    )
                remember(emblem.entity_id, zone)

            faith_ids: set[str] = set()
            for faith_index, faith in enumerate(player.faiths):
                zone = f"{prefix} faiths[{faith_index}]"
                if faith.controller != player_index:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} controller mismatch"
                    )
                if faith.created_sequence <= 0:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} created_sequence must be positive"
                    )
                if (
                    not isinstance(faith.value, int)
                    or isinstance(faith.value, bool)
                    or faith.value < 0
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} value is invalid"
                    )
                if (
                    not isinstance(faith.mode_selection_bonus, int)
                    or isinstance(faith.mode_selection_bonus, bool)
                    or faith.mode_selection_bonus < 0
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} mode selection bonus is invalid"
                    )
                if faith.faith_id in faith_ids:
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} has duplicate faith "
                        f"{faith.faith_id!r}"
                    )
                faith_ids.add(faith.faith_id)
                granted_sequences = [
                    ability.granted_sequence
                    for ability in faith.granted_abilities
                ]
                if (
                    len(granted_sequences) != len(set(granted_sequences))
                    or any(sequence <= 0 for sequence in granted_sequences)
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} granted ability sequences are invalid"
                    )
                if faith._next_granted_ability_sequence <= max(
                    granted_sequences, default=0
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} next granted ability sequence is not ahead"
                    )
                for ability_index, ability in enumerate(faith.granted_abilities):
                    ability_zone = f"{zone}.granted_abilities[{ability_index}]"
                    if (
                        not isinstance(ability.ability_id, str)
                        or not ability.ability_id
                        or not isinstance(ability.trigger, FaithTrigger)
                        or not isinstance(ability.operations, tuple)
                        or not ability.operations
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {ability_zone} is invalid"
                        )
                remember(faith.entity_id, zone)

        for frame_index, frame in enumerate(state.effect_stack):
            zone = f"effect_stack[{frame_index}]"
            if not isinstance(frame, EffectFrame):
                raise IllegalCommand(
                    f"Invariant failed: {zone} is not EffectFrame"
                )
            if frame.controller not in (0, 1):
                raise IllegalCommand(
                    f"Invariant failed: {zone} controller out of range"
                )
            if not isinstance(frame.operations, tuple):
                raise IllegalCommand(
                    f"Invariant failed: {zone} operations must be a tuple"
                )
            if not isinstance(frame.fusion_materials, tuple) or any(
                not isinstance(material, FusionMaterial)
                for material in frame.fusion_materials
            ):
                raise IllegalCommand(
                    f"Invariant failed: {zone} fusion_materials must be a tuple of records"
                )
            if (
                not isinstance(frame.next_index, int)
                or isinstance(frame.next_index, bool)
                or frame.next_index < 0
                or frame.next_index > len(frame.operations)
            ):
                raise IllegalCommand(
                    f"Invariant failed: {zone} next_index out of range: "
                    f"{frame.next_index}/{len(frame.operations)}"
                )
            source_card_id = getattr(frame.source_card, "card_id", None)
            if frame.source_card_id != source_card_id:
                raise IllegalCommand(
                    f"Invariant failed: {zone} source_card_id mismatch"
                )
            if not isinstance(frame.source_name, str) or not frame.source_name:
                raise IllegalCommand(
                    f"Invariant failed: {zone} source_name is empty"
                )
            if frame.source_entity_id is not None and frame.source_entity_id <= 0:
                raise IllegalCommand(
                    f"Invariant failed: {zone} source_entity_id must be positive"
                )
            snapshot = frame.source_snapshot
            if snapshot is not None:
                if not isinstance(snapshot, SourceStateSnapshot):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} source_snapshot is invalid"
                    )
                if (
                    snapshot.entity_id != frame.source_entity_id
                    or snapshot.controller != frame.controller
                    or snapshot.card_id != frame.source_card_id
                    or snapshot.card_type != frame.source_card.card_type
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} source_snapshot identity mismatch"
                    )
                if any(
                    value is not None
                    and (
                        not isinstance(value, int)
                        or isinstance(value, bool)
                    )
                    for value in (snapshot.attack, snapshot.health)
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} source_snapshot stats are invalid"
                    )
                if (
                    not isinstance(snapshot.evolved, bool)
                    or not isinstance(snapshot.super_evolved, bool)
                    or snapshot.super_evolved and not snapshot.evolved
                    or not isinstance(snapshot.effective_keywords, frozenset)
                    or any(
                        not isinstance(keyword, str) or not keyword
                        for keyword in snapshot.effective_keywords
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} source_snapshot state is invalid"
                    )
            if (
                not isinstance(frame.source_spellboost_count, int)
                or isinstance(frame.source_spellboost_count, bool)
                or frame.source_spellboost_count < 0
            ):
                raise IllegalCommand(
                    f"Invariant failed: {zone} source_spellboost_count is invalid"
                )
            if (
                not isinstance(frame.source_cost, int)
                or isinstance(frame.source_cost, bool)
                or frame.source_cost < 0
            ):
                raise IllegalCommand(
                    f"Invariant failed: {zone} source_cost is invalid"
                )
            if (
                not isinstance(frame.distributed_value, int)
                or isinstance(frame.distributed_value, bool)
                or frame.distributed_value < 0
            ):
                raise IllegalCommand(
                    f"Invariant failed: {zone} distributed_value is invalid"
                )
            if not isinstance(frame.label, str) or not frame.label:
                raise IllegalCommand(
                    f"Invariant failed: {zone} label is empty"
                )

            def check_effect_target_id(value: int | None, field: str) -> None:
                if value is None:
                    return
                if not isinstance(value, int) or isinstance(value, bool):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} {field} is not an integer target id"
                    )
                if _is_leader_target_id(value):
                    leader_index = _leader_index_from_target_id(value)
                    if leader_index not in (0, 1):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} {field} leader index out of range"
                        )
                elif value <= 0:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} {field} must be positive or leader"
                    )

            check_effect_target_id(frame.pending_target_id, "pending_target_id")
            if not isinstance(frame.pending_target_ids, list):
                raise IllegalCommand(
                    f"Invariant failed: {zone} pending_target_ids must be a list"
                )
            if frame.pending_target_id is not None and frame.pending_target_ids:
                raise IllegalCommand(
                    f"Invariant failed: {zone} has both single and multi pending targets"
                )
            for target_index, target_id in enumerate(frame.pending_target_ids):
                check_effect_target_id(
                    target_id,
                    f"pending_target_ids[{target_index}]",
                )
            if not isinstance(frame._all_target_ids, list):
                raise IllegalCommand(
                    f"Invariant failed: {zone} _all_target_ids must be a list"
                )
            for target_index, target_id in enumerate(frame._all_target_ids):
                check_effect_target_id(target_id, f"_all_target_ids[{target_index}]")
            if (
                not isinstance(frame._all_target_index, int)
                or isinstance(frame._all_target_index, bool)
                or frame._all_target_index < 0
                or frame._all_target_index > len(frame._all_target_ids)
            ):
                raise IllegalCommand(
                    f"Invariant failed: {zone} _all_target_index out of range"
                )
            if not isinstance(frame._target_bindings, dict):
                raise IllegalCommand(
                    f"Invariant failed: {zone} _target_bindings must be a dict"
                )
            for key, target_ids in frame._target_bindings.items():
                if not isinstance(key, str) or not key:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} has invalid target binding key"
                    )
                if not isinstance(target_ids, tuple):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} target binding must be a tuple"
                    )
                for target_index, target_id in enumerate(target_ids):
                    check_effect_target_id(
                        target_id,
                        f"_target_bindings[{key!r}][{target_index}]",
                    )
            if not isinstance(frame._target_binding_operations, dict):
                raise IllegalCommand(
                    f"Invariant failed: {zone} _target_binding_operations must be a dict"
                )
            if set(frame._target_binding_operations) != set(frame._target_bindings):
                raise IllegalCommand(
                    f"Invariant failed: {zone} target bindings and operations differ"
                )
            for key, operation in frame._target_binding_operations.items():
                if key not in frame._target_bindings:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} target binding operation lacks target"
                    )
                if not isinstance(operation, EffectOperation):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} target binding operation is invalid"
                    )
                if (
                    not frame._target_bindings[key]
                    and not _operation_produces_output_binding(operation)
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} selected target binding is empty"
                    )
            if not isinstance(frame._target_binding_snapshots, dict):
                raise IllegalCommand(
                    f"Invariant failed: {zone} _target_binding_snapshots must be a dict"
                )
            if set(frame._target_binding_snapshots) != set(frame._target_bindings):
                raise IllegalCommand(
                    f"Invariant failed: {zone} target bindings and snapshots differ"
                )
            for key, snapshots in frame._target_binding_snapshots.items():
                if not isinstance(snapshots, tuple) or len(snapshots) != len(
                    frame._target_bindings[key]
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} target snapshots do not match binding"
                    )
                for snapshot_index, snapshot in enumerate(snapshots):
                    if not isinstance(snapshot, BoundTargetSnapshot):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} target snapshot is invalid"
                        )
                    if snapshot.entity_id != frame._target_bindings[key][snapshot_index]:
                        raise IllegalCommand(
                            f"Invariant failed: {zone} target snapshot entity differs"
                        )
                    if (
                        snapshot.controller not in (0, 1)
                        or snapshot.zone not in {"board", "hand", "graveyard"}
                        or not isinstance(snapshot.card_id, int)
                        or isinstance(snapshot.card_id, bool)
                        or snapshot.card_id <= 0
                        or not isinstance(snapshot.card_type, str)
                        or not snapshot.card_type
                        or not isinstance(snapshot.card_name, str)
                        or not isinstance(snapshot.cost, int)
                        or isinstance(snapshot.cost, bool)
                        or snapshot.cost < 0
                        or not isinstance(snapshot.definition, CardDefinition)
                        or snapshot.definition.card_id != snapshot.card_id
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {zone} target snapshot payload is invalid"
                        )
            if not isinstance(frame._decision_meta, dict):
                raise IllegalCommand(
                    f"Invariant failed: {zone} _decision_meta must be a dict"
                )
            for operation_index, operation in enumerate(frame.operations):
                operation_zone = f"{zone} operation[{operation_index}]"
                if not isinstance(operation, EffectOperation):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} is not EffectOperation"
                    )
                if not isinstance(operation.kind, EffectKind):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} kind is invalid"
                    )
                if not isinstance(operation.target, TargetKind):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} target is invalid"
                    )
                if (
                    not isinstance(operation.target_count, int)
                    or isinstance(operation.target_count, bool)
                    or operation.target_count <= 0
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} target_count must be positive"
                    )
                if not isinstance(operation.allow_duplicate_targets, bool):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} duplicate-target policy is invalid"
                    )
                if not isinstance(operation.requires_full_target_count, bool):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} full-target policy is invalid"
                    )
                if (
                    operation.turn_end_destroy_timing is not None
                    and not isinstance(
                        operation.turn_end_destroy_timing,
                        TurnEndDestroyTiming,
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} turn-end destroy timing is invalid"
                    )
                if (
                    operation.turn_end_banish_timing is not None
                    and not isinstance(
                        operation.turn_end_banish_timing,
                        TurnEndDestroyTiming,
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} turn-end banish timing is invalid"
                    )
                if (
                    operation.turn_end_ability_timing is not None
                    and not isinstance(
                        operation.turn_end_ability_timing,
                        TurnEndDestroyTiming,
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} turn-end ability timing is invalid"
                    )
                if not isinstance(operation.exclude_source, bool):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} source-exclusion policy is invalid"
                    )
                if not isinstance(operation.exclude_attack_target, bool):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} attack-target exclusion policy is invalid"
                    )
                if not isinstance(operation.distinct_card_names, bool):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} distinct-name policy is invalid"
                    )
                if not isinstance(operation.highest_base_cost_only, bool):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} "
                        "highest-cost policy is invalid"
                    )
                if not isinstance(operation.bind_successful_targets, bool):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} successful-binding policy is invalid"
                    )
                if (
                    not isinstance(operation.keywords, tuple)
                    or any(
                        not isinstance(keyword, str) or not keyword
                        for keyword in operation.keywords
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} keyword candidates are invalid"
                    )
                if (
                    operation.history_filter is not None
                    and not isinstance(operation.history_filter, HandFilter)
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} history filter is invalid"
                    )
                if operation.condition_target_key is not None and (
                    operation.kind is not EffectKind.CONDITIONAL
                    or not isinstance(operation.condition_target_key, str)
                    or not operation.condition_target_key
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} condition target key is invalid"
                    )
                if operation.kind is EffectKind.CONSUME_FAITH:
                    if (
                        not isinstance(operation.faith_id, str)
                        or not operation.faith_id
                        or operation.amount <= 0
                        or not operation.faith_operations
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {operation_zone} consume_faith payload is invalid"
                        )
                if operation.kind is EffectKind.GRANT_FAITH_ABILITY:
                    if (
                        not isinstance(operation.faith_id, str)
                        or not operation.faith_id
                        or not isinstance(operation.faith_ability_id, str)
                        or not operation.faith_ability_id
                        or operation.faith_trigger not in {
                            trigger.value for trigger in FaithTrigger
                        }
                        or operation.faith_stacking not in {
                            policy.value for policy in FaithAbilityStacking
                        }
                        or not operation.faith_operations
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {operation_zone} grant_faith_ability payload is invalid"
                        )
                if (
                    operation.kind
                    is EffectKind.GRANT_FAITH_MODE_SELECTION_BONUS
                ):
                    if (
                        not isinstance(operation.faith_id, str)
                        or not operation.faith_id
                        or operation.amount <= 0
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {operation_zone} "
                            "grant_faith_mode_selection_bonus payload is invalid"
                        )
                if operation.kind is EffectKind.GRANT_TURN_END_ABILITY:
                    if (
                        operation.turn_end_ability_timing is None
                        or not operation.granted_operations
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {operation_zone} "
                            "grant_turn_end_ability payload is invalid"
                        )
                if operation.kind is EffectKind.BUFF_DECK_CARDS:
                    if (
                        operation.deck_filter is None
                        or operation.deck_filter.card_type != "随从"
                        or (
                            operation.amount == 0
                            and operation.secondary_amount == 0
                        )
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {operation_zone} "
                            "buff_deck_cards payload is invalid"
                        )
                if operation.kind is EffectKind.RANDOM_DISTRIBUTE:
                    if (
                        not isinstance(operation.faith_id, str)
                        or not operation.faith_id
                        or len(operation.random_distribution_operations) < 2
                        or any(
                            not bucket
                            for bucket in operation.random_distribution_operations
                        )
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {operation_zone} random_distribute payload is invalid"
                        )
                if operation.kind is EffectKind.RANDOM_CHOICE:
                    if (
                        len(operation.random_choice_options) < 2
                        or operation.amount < 1
                        or operation.amount
                        > len(operation.random_choice_options)
                        or any(
                            not option.operations
                            for option in operation.random_choice_options
                        )
                        or (
                            operation.random_choice_history_key is not None
                            and (
                                not isinstance(
                                    operation.random_choice_history_key,
                                    str,
                                )
                                or not operation.random_choice_history_key
                            )
                        )
                    ):
                        raise IllegalCommand(
                            f"Invariant failed: {operation_zone} random_choice payload is invalid"
                        )

            def check_positive_int(value: int | None, field: str) -> None:
                if value is None:
                    return
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} {field} must be positive"
                    )

            check_positive_int(frame.emblem_batch_id, "emblem_batch_id")
            if frame.emblem_batch_id is None:
                if any(
                    value is not None
                    for value in (
                        frame.emblem_activation_owner,
                        frame.emblem_activation_entity_id,
                        frame.emblem_activation_trigger_index,
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} emblem activation fields require batch"
                    )
            else:
                if frame.emblem_activation_owner not in (0, 1):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} emblem_activation_owner out of range"
                    )
                check_positive_int(
                    frame.emblem_activation_entity_id,
                    "emblem_activation_entity_id",
                )
                if (
                    not isinstance(frame.emblem_activation_trigger_index, int)
                    or isinstance(frame.emblem_activation_trigger_index, bool)
                    or frame.emblem_activation_trigger_index < 0
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} emblem_activation_trigger_index invalid"
                    )

            check_positive_int(frame.listener_batch_id, "listener_batch_id")
            listener_context_fields = (
                frame.listener_activation_owner,
                frame.listener_activation_zone,
                frame.listener_activation_entity_id,
                frame.listener_activation_card_id,
                frame.listener_activation_definition_index,
            )
            has_listener_context = any(
                value is not None for value in listener_context_fields
            )
            if frame.listener_batch_id is not None and not has_listener_context:
                raise IllegalCommand(
                    f"Invariant failed: {zone} listener batch requires context"
                )
            if has_listener_context:
                if frame.listener_activation_owner not in (0, 1):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} listener owner out of range"
                    )
                if frame.listener_activation_zone not in {
                    item.value for item in ListenerZone
                }:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} listener zone is invalid"
                    )
                check_positive_int(
                    frame.listener_activation_entity_id,
                    "listener_activation_entity_id",
                )
                check_positive_int(
                    frame.listener_activation_card_id,
                    "listener_activation_card_id",
                )
                if (
                    not isinstance(
                        frame.listener_activation_definition_index,
                        int,
                    )
                    or isinstance(
                        frame.listener_activation_definition_index,
                        bool,
                    )
                    or frame.listener_activation_definition_index < 0
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} listener definition index invalid"
                    )
                if (
                    isinstance(frame.listener_activation_count, bool)
                    or not isinstance(frame.listener_activation_count, int)
                    or frame.listener_activation_count < 1
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} listener activation count invalid"
                    )
            elif frame.listener_activation_count != 0:
                raise IllegalCommand(
                    f"Invariant failed: {zone} listener activation count "
                    "requires listener context"
                )
            check_positive_int(
                frame.event_source_entity_id,
                "event_source_entity_id",
            )
            if (
                frame.event_source_base_cost is not None
                and (
                    isinstance(frame.event_source_base_cost, bool)
                    or not isinstance(frame.event_source_base_cost, int)
                    or frame.event_source_base_cost < 0
                )
            ):
                raise IllegalCommand(
                    f"Invariant failed: {zone} event source base cost invalid"
                )
            check_positive_int(
                frame.attack_target_entity_id,
                "attack_target_entity_id",
            )

            check_positive_int(
                frame.emblem_expiration_batch_id,
                "emblem_expiration_batch_id",
            )
            if frame.emblem_expiration_batch_id is None:
                if any(
                    value is not None
                    for value in (
                        frame.expiring_emblem_owner,
                        frame.expiring_emblem_entity_id,
                    )
                ):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} emblem expiration fields require batch"
                    )
            else:
                if frame.expiring_emblem_owner not in (0, 1):
                    raise IllegalCommand(
                        f"Invariant failed: {zone} expiring_emblem_owner out of range"
                    )
                check_positive_int(
                    frame.expiring_emblem_entity_id,
                    "expiring_emblem_entity_id",
                )
        for event_index, event in enumerate(state.event_queue):
            if event.player_index not in (0, 1):
                raise IllegalCommand(
                    f"Invariant failed: event_queue[{event_index}] player out of range"
                )
            if (
                event.listener_sources is not None
                and (
                    not isinstance(event.listener_sources, tuple)
                    or any(
                        not isinstance(record, tuple)
                        or len(record) != 4
                        or record[0] not in (0, 1)
                        or record[1] not in {
                            zone.value for zone in ListenerZone
                        }
                        or not isinstance(record[2], int)
                        or isinstance(record[2], bool)
                        or record[2] <= 0
                        or not isinstance(record[3], int)
                        or isinstance(record[3], bool)
                        or record[3] <= 0
                        for record in event.listener_sources
                    )
                )
            ):
                raise IllegalCommand(
                    f"Invariant failed: event_queue[{event_index}] "
                    "listener source snapshot is invalid"
                )
            if (
                event.emblem_sources is not None
                and (
                    not isinstance(event.emblem_sources, tuple)
                    or any(
                        not isinstance(record, tuple)
                        or len(record) != 2
                        or record[0] not in (0, 1)
                        or not isinstance(record[1], int)
                        or isinstance(record[1], bool)
                        or record[1] <= 0
                        for record in event.emblem_sources
                    )
                )
            ):
                raise IllegalCommand(
                    f"Invariant failed: event_queue[{event_index}] "
                    "emblem source snapshot is invalid"
                )
        death_sequences: set[int] = set()
        for record in state.destroyed_followers:
            if record.owner not in (0, 1):
                raise IllegalCommand("Invariant failed: destroyed follower owner out of range")
            if (
                record.death_sequence <= 0
                or record.death_sequence in death_sequences
            ):
                raise IllegalCommand(
                    "Invariant failed: destroyed follower death_sequence "
                    "must be unique and positive"
                )
            death_sequences.add(record.death_sequence)
            if record.definition.card_type != "随从":
                raise IllegalCommand(
                    "Invariant failed: destroyed follower definition is not a follower"
                )
            if (
                not isinstance(record.destroyed_turn, int)
                or isinstance(record.destroyed_turn, bool)
                or record.destroyed_turn < 0
                or record.destroyed_turn > state.turn
            ):
                raise IllegalCommand(
                    "Invariant failed: destroyed follower destroyed_turn is invalid"
                )
        for record in state.destroyed_amulets:
            if record.owner not in (0, 1):
                raise IllegalCommand(
                    "Invariant failed: destroyed amulet owner out of range"
                )
            if (
                record.death_sequence <= 0
                or record.death_sequence in death_sequences
            ):
                raise IllegalCommand(
                    "Invariant failed: destroyed amulet death_sequence "
                    "must be unique and positive"
                )
            death_sequences.add(record.death_sequence)
            if record.play_mode_id is not None and (
                not isinstance(record.play_mode_id, str)
                or not record.play_mode_id
            ):
                raise IllegalCommand(
                    "Invariant failed: destroyed amulet play_mode_id is invalid"
                )
            if record.summon_countdown is not None and (
                not isinstance(record.summon_countdown, int)
                or isinstance(record.summon_countdown, bool)
                or record.summon_countdown < 0
            ):
                raise IllegalCommand(
                    "Invariant failed: destroyed amulet summon_countdown is invalid"
                )
            if (
                not isinstance(record.destroyed_turn, int)
                or isinstance(record.destroyed_turn, bool)
                or record.destroyed_turn < 0
                or record.destroyed_turn > state.turn
            ):
                raise IllegalCommand(
                    "Invariant failed: destroyed amulet destroyed_turn is invalid"
                )
        follower_entry_sequences: set[int] = set()
        for record in state.follower_entries:
            if record.owner not in (0, 1):
                raise IllegalCommand(
                    "Invariant failed: follower entry owner out of range"
                )
            if (
                record.entry_sequence <= 0
                or record.entry_sequence in follower_entry_sequences
            ):
                raise IllegalCommand(
                    "Invariant failed: follower entry sequence must be unique and positive"
                )
            follower_entry_sequences.add(record.entry_sequence)
            if (
                not isinstance(record.entered_turn, int)
                or isinstance(record.entered_turn, bool)
                or record.entered_turn < 0
                or record.entered_turn > state.turn
            ):
                raise IllegalCommand(
                    "Invariant failed: follower entry entered_turn is invalid"
                )
            if record.definition.card_type != "随从":
                raise IllegalCommand(
                    "Invariant failed: follower entry definition is not a follower"
                )
            if not isinstance(record.entry_cause, str) or not record.entry_cause:
                raise IllegalCommand(
                    "Invariant failed: follower entry cause is invalid"
                )
        if (
            state._next_follower_entry_sequence <= 0
            or state._next_follower_entry_sequence
            <= max(follower_entry_sequences, default=0)
        ):
            raise IllegalCommand(
                "Invariant failed: next follower entry sequence is invalid"
            )
        for batch in state.death_queue:
            if batch.batch_id <= 0:
                raise IllegalCommand("Invariant failed: death batch id must be positive")
            for record in batch.records:
                if record.owner not in (0, 1):
                    raise IllegalCommand("Invariant failed: death record owner out of range")
                if (
                    not isinstance(record.effective_keywords, frozenset)
                    or any(
                        not isinstance(keyword, str) or not keyword
                        for keyword in record.effective_keywords
                    )
                ):
                    raise IllegalCommand(
                        "Invariant failed: death record keywords are invalid"
                    )
                if any(
                    value is not None
                    and (
                        not isinstance(value, int)
                        or isinstance(value, bool)
                    )
                    for value in (record.attack, record.health)
                ):
                    raise IllegalCommand(
                        "Invariant failed: death record source stats are invalid"
                    )
                if (
                    not isinstance(record.evolved, bool)
                    or not isinstance(record.super_evolved, bool)
                    or record.super_evolved and not record.evolved
                ):
                    raise IllegalCommand(
                        "Invariant failed: death record evolution state is invalid"
                    )

    def _record_event(self, event: GameEvent) -> None:
        """Record diagnostics without truncating the current transition."""
        self.event_history.append(event)
        if self.runtime_coverage is not None:
            self.runtime_coverage.record_event(event)
        if self._active_transition_events is not None:
            self._active_transition_events.append(event)
        limit = self.config.event_history_limit
        if limit is not None and len(self.event_history) > limit:
            del self.event_history[:-limit]

    def _record_runtime_diagnostic(
        self,
        kind: str,
        *,
        card_id: int | None = None,
        clause_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        if self.runtime_coverage is not None:
            self.runtime_coverage.record_diagnostic(
                kind,
                card_id=card_id,
                clause_id=clause_id,
                detail=detail,
            )

    def _emit(self, event: GameEvent) -> None:
        if event.listener_sources is None:
            listener_sources: list[tuple[int, str, int, int]] = []
            for owner, player in enumerate(self.players):
                listener_sources.extend(
                    (
                        owner,
                        ListenerZone.BOARD.value,
                        entity.entity_id,
                        entity.definition.card_id,
                    )
                    for entity in player.board
                    if self.rulebook.listeners_for(entity.definition.card_id)
                )
                listener_sources.extend(
                    (
                        owner,
                        ListenerZone.HAND.value,
                        card.entity_id,
                        card.definition.card_id,
                    )
                    for card in player.hand
                    if (
                        isinstance(card, HandCard)
                        and self.rulebook.listeners_for(card.definition.card_id)
                    )
                )
                for _, _, entity_id in self._leader_area_listener_sources(owner):
                    card_id = next(
                        (
                            instance.source_card_id
                            for instance in (*player.emblems, *player.faiths)
                            if instance.entity_id == entity_id
                        ),
                        None,
                    )
                    if (
                        isinstance(card_id, int)
                        and self.rulebook.listeners_for(card_id)
                    ):
                        listener_sources.append(
                            (
                                owner,
                                ListenerZone.LEADER_AREA.value,
                                entity_id,
                                card_id,
                            )
                        )
            event = replace(
                event,
                listener_sources=tuple(listener_sources),
            )
        if event.emblem_sources is None:
            event = replace(
                event,
                emblem_sources=tuple(
                    (owner, emblem.entity_id)
                    for owner, player in enumerate(self.players)
                    for emblem in player.emblems
                ),
            )
        self.state.event_queue.append(event)

    def _dispatch_ability(
        self,
        event: AbilityEvent,
        source: Unit,
        target: Unit | None = None,
        *,
        player_index: int | None = None,
        trigger_keywords: bool = True,
        counts_as_evolution: bool = False,
    ) -> None:
        if isinstance(source, Unit) and source.printed_abilities_removed:
            return
        resolved_player_index = (
            self.current_player if player_index is None else player_index
        )
        if event in {
            AbilityEvent.FOLLOWER_EVOLVED,
            AbilityEvent.FOLLOWER_SUPER_EVOLVED,
        }:
            triggers: list[Trigger] = []
            if counts_as_evolution:
                triggers.append(Trigger.SELF_EVOLVED)
            if event is AbilityEvent.FOLLOWER_SUPER_EVOLVED:
                triggers.append(Trigger.SELF_SUPER_EVOLVED)
            if trigger_keywords:
                triggers.append(
                    Trigger.SUPER_EVOLVE
                    if event is AbilityEvent.FOLLOWER_SUPER_EVOLVED
                    else Trigger.EVOLVE
                )
            self._execute_trigger_rule_batch(
                tuple(triggers),
                AbilityContext(
                    event=event,
                    player_index=resolved_player_index,
                    source=source,
                    target=target,
                ),
            )
            return
        structured_trigger = {
            AbilityEvent.BEFORE_ATTACK: (
                Trigger.ATTACK,
                AbilityKeyword.ON_ATTACK,
            ),
            AbilityEvent.BEFORE_COMBAT: (
                Trigger.CLASH,
                AbilityKeyword.ON_CLASH,
            ),
        }.get(event)
        if structured_trigger is not None:
            trigger, keyword = structured_trigger
            if (
                keyword not in source.definition.abilities
                and self.rulebook.operations_for(source.definition.card_id, trigger)
            ):
                self._execute_trigger_rules(
                    trigger,
                    AbilityContext(
                        event=event,
                        player_index=resolved_player_index,
                        source=source,
                        target=target,
                    ),
                )
        self.ability_handlers.dispatch(
            AbilityContext(
                event=event,
                player_index=resolved_player_index,
                source=source,
                target=target,
            )
        )

    def _dispatch_card_ability(
        self,
        event: AbilityEvent,
        card: CardDefinition,
        *,
        player_index: int | None = None,
    ) -> None:
        self.ability_handlers.dispatch(
            AbilityContext(
                event=event,
                player_index=(
                    self.current_player if player_index is None else player_index
                ),
                source=card,
            )
        )

    def _log(self, player_index: int, message: str) -> None:
        if not self.config.retain_text_logs:
            return
        self.logs.append(
            f"[半回合 {self.turn:03d}][玩家 {player_index + 1}] {message}"
        )

    def _allocate_choice_request_id(self) -> int:
        request_id = self._next_choice_request_id
        self._next_choice_request_id += 1
        return request_id

    def _ensure_entity_ids(self) -> None:
        seen: set[int] = set()
        for player in self.players:
            for material in player.fusion_materials:
                if material.entity_id <= 0:
                    raise IllegalCommand("Fusion material entity_id must be positive")
                if material.entity_id in seen:
                    raise IllegalCommand(
                        f"Fusion material {material.entity_id} exists in multiple zones"
                    )
                seen.add(material.entity_id)
            normalized_hand: list[HandCard] = []
            normalized_ids: list[int] = []
            for index, card in enumerate(player.hand):
                old_id = (
                    player.hand_entity_ids[index]
                    if index < len(player.hand_entity_ids)
                    else 0
                )
                if isinstance(card, HandCard):
                    hand_card = card
                    if hand_card.entity_id <= 0:
                        hand_card.entity_id = old_id
                    hand_card.spellboost_cost_reduction = (
                        self.rulebook.spellboost_cost_reduction(
                            hand_card.definition.card_id
                        )
                    )
                    hand_card.cannot_be_played = self.rulebook.cannot_be_played(
                        hand_card.definition.card_id
                    )
                else:
                    hand_card = self._make_hand_card(card, old_id)
                if hand_card.entity_id <= 0 or hand_card.entity_id in seen:
                    hand_card.entity_id = self.state.allocate_entity_id()
                seen.add(hand_card.entity_id)
                normalized_hand.append(hand_card)
                normalized_ids.append(hand_card.entity_id)
            player.hand = normalized_hand
            player.hand_entity_ids = normalized_ids
            for entity in player.board:
                if entity.entity_id <= 0 or entity.entity_id in seen:
                    entity.entity_id = self.state.allocate_entity_id()
                seen.add(entity.entity_id)
            for emblem in player.emblems:
                if emblem.entity_id <= 0 or emblem.entity_id in seen:
                    emblem.entity_id = self.state.allocate_entity_id()
                seen.add(emblem.entity_id)
            for faith in player.faiths:
                if faith.entity_id <= 0 or faith.entity_id in seen:
                    faith.entity_id = self.state.allocate_entity_id()
                seen.add(faith.entity_id)
            for graveyard_card in player.graveyard:
                if graveyard_card.entity_id <= 0:
                    raise IllegalCommand("Graveyard entity_id must be positive")
                if graveyard_card.entity_id in seen:
                    raise IllegalCommand(
                        f"Entity {graveyard_card.entity_id} exists in multiple zones"
                    )
                seen.add(graveyard_card.entity_id)

    def _append_hand_card(
        self,
        player: PlayerState,
        definition: CardDefinition | DeckCard,
        *,
        origin: CardOrigin = CardOrigin.DECK,
        source_origin: CardOrigin | None = None,
        fused_material_ids: tuple[int, ...] = (),
    ) -> HandCard:
        inherited_cost_modifiers: list[CostModifier] = []
        inherited_stat_modifiers: list[StatModifier] = []
        if isinstance(definition, DeckCard):
            inherited_cost_modifiers = list(definition.cost_modifiers)
            inherited_stat_modifiers = list(definition.stat_modifiers)
            definition = definition.definition
        hand_card = self._make_hand_card(
            definition,
            self.state.allocate_entity_id(),
            origin=origin,
            source_origin=source_origin,
            fused_material_ids=fused_material_ids,
        )
        hand_card.cost_modifiers.extend(inherited_cost_modifiers)
        hand_card.stat_modifiers.extend(inherited_stat_modifiers)
        player.hand.append(hand_card)
        player.hand_entity_ids.append(hand_card.entity_id)
        return hand_card

    def _make_hand_card(
        self,
        definition: CardDefinition,
        entity_id: int,
        *,
        origin: CardOrigin = CardOrigin.DECK,
        source_origin: CardOrigin | None = None,
        fused_material_ids: tuple[int, ...] = (),
    ) -> HandCard:
        return HandCard(
            definition=definition,
            entity_id=entity_id,
            spellboost_cost_reduction=self.rulebook.spellboost_cost_reduction(
                definition.card_id
            ),
            cannot_be_played=self.rulebook.cannot_be_played(definition.card_id),
            origin=origin,
            source_origin=source_origin,
            fused_material_ids=list(fused_material_ids),
            printed_keyword_overrides=set(
                self.rulebook.non_intrinsic_keywords(definition.card_id)
            ),
        )

    def _hand_cards(self, player_index: int) -> list[HandCard]:
        self._ensure_entity_ids()
        return [
            card
            for card in self.players[player_index].hand
            if isinstance(card, HandCard)
        ]

    def _find_hand_card(
        self, player_index: int, entity_id: int | None
    ) -> HandCard:
        if entity_id is None:
            raise IllegalCommand("A hand target is required")
        for card in self._hand_cards(player_index):
            if card.entity_id == entity_id:
                return card
        raise IllegalCommand(f"Hand entity {entity_id} does not exist")

    def _find_hand_card_with_owner(
        self, entity_id: int | None
    ) -> tuple[int, HandCard]:
        if entity_id is None:
            raise IllegalCommand("A hand target is required")
        for owner in (0, 1):
            for card in self._hand_cards(owner):
                if card.entity_id == entity_id:
                    return owner, card
        raise IllegalCommand(f"Hand entity {entity_id} does not exist")

    @staticmethod
    def _apply_hand_card_runtime_to_unit(
        hand_card: HandCard,
        unit: Unit,
    ) -> None:
        for keyword in sorted(hand_card.permanent_keywords):
            unit.add_keyword(keyword)
        for modifier in hand_card.temporary_keywords:
            unit.add_keyword(
                modifier.keyword,
                duration=modifier.duration,
                expires_for_player=modifier.expires_for_player,
            )
        for keyword in sorted(hand_card.removed_keywords):
            unit.remove_keyword(keyword)
        for modifier in hand_card.temporary_keyword_removals:
            unit.remove_keyword(
                modifier.keyword,
                duration=modifier.duration,
                expires_for_player=modifier.expires_for_player,
            )
        unit.granted_last_words = list(hand_card.granted_last_words)
        unit.effect_destroy_immunity = hand_card.effect_destroy_immunity

    def _allocate_modifier_id(self) -> int:
        modifier_id = self._next_modifier_id
        self._next_modifier_id += 1
        return modifier_id

    def _expire_modifiers(
        self, duration: ModifierDuration, player_index: int
    ) -> None:
        expire_durations = _expire_duration_values(duration)
        for owner_index, player in enumerate(self.players):
            for dur_str in expire_durations:
                player.expire_leader_damage_modifiers(dur_str, player_index)
            for entity in player.board:
                if not isinstance(entity, Unit):
                    continue
                for dur_str in expire_durations:
                    entity.expire_keywords(dur_str, player_index)
                    entity.expire_stat_modifiers(dur_str, player_index)
                    entity.expire_attack_capacity(dur_str, player_index)
                    entity.expire_attack_restrictions(dur_str, player_index)
                    entity.expire_targeting_restrictions(dur_str, player_index)
            for hand_card in self._hand_cards(owner_index):
                for dur_str in expire_durations:
                    hand_card.expire_cost_modifiers(dur_str, player_index)
                    hand_card.expire_stat_modifiers(dur_str, player_index)
                    hand_card.expire_keywords(dur_str, player_index)

    def _find_board_entity(self, entity_id: int | None) -> BoardCard:
        if entity_id is None:
            raise IllegalCommand("A board entity target is required")
        for player in self.players:
            for entity in player.board:
                if entity.entity_id == entity_id:
                    return entity
        raise IllegalCommand(f"Board entity {entity_id} does not exist")

    def _entity_owner(self, entity_id: int) -> int:
        for player_index, player in enumerate(self.players):
            if any(entity.entity_id == entity_id for entity in player.board):
                return player_index
        raise IllegalCommand(f"Board entity {entity_id} does not have an owner")

    @staticmethod
    def _find_unit(board: list[BoardCard], entity_id: int) -> Unit:
        for unit in board:
            if isinstance(unit, Unit) and unit.entity_id == entity_id:
                return unit
        raise IllegalCommand(f"Unit {entity_id} is not on the expected board")
