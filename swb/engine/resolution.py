from __future__ import annotations

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
)
from swb.engine.deck import CLASS_NAMES, validate_deck
from swb.engine.effects import (
    Condition,
    ConditionType,
    CostChangeMode,
    DeckFilter,
    EffectFrame,
    EffectKind,
    EffectOperation,
    ExprType,
    ModifierDuration,
    TargetKind,
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
from swb.engine.emblem import EventScope, TurnScope
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
from swb.engine.play_modes import validate_runtime_play_mode
from swb.engine.state import (
    Amulet,
    BoardCard,
    DeathBatch,
    DeathCause,
    DeathRecord,
    DestroyedFollowerRecord,
    FusionMaterial,
    GameState,
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
    hand_choice_options,
    has_leader_choice,
    is_all_target,
    is_choice_target,
    is_graveyard_target,
    is_random_target,
    leader_choice_options,
    pick_random,
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
    evaluate_target_conditions,
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
    EffectKind.BUFF_UNIT,
    EffectKind.DESTROY,
    EffectKind.BANISH,
    EffectKind.RETURN_TO_HAND,
    EffectKind.RETURN_TO_DECK,
    EffectKind.REDUCE_COUNTDOWN,
    EffectKind.ADD_KEYWORD,
    EffectKind.REMOVE_KEYWORD,
    EffectKind.TRANSFORM,
    EffectKind.SET_STATS,
    EffectKind.ADD_ATTACK_RESTRICTION,
    EffectKind.REMOVE_ATTACK_RESTRICTION,
    EffectKind.ADD_TARGETING_RESTRICTION,
    EffectKind.REMOVE_TARGETING_RESTRICTION,
})

_EVENT_SOURCE_BOARD_EFFECTS = _SOURCE_REQUIRED_SELF_TARGET_EFFECTS | frozenset({
    EffectKind.EVOLVE_UNIT,
    EffectKind.SUPER_EVOLVE_UNIT,
})

_SOURCE_CONDITION_TYPES = frozenset({
    ConditionType.SOURCE_EVOLVED,
    ConditionType.SOURCE_HAS_KEYWORD,
})

_SOURCE_EXPRESSION_TYPES = frozenset({
    ExprType.SOURCE_ATTACK,
    ExprType.SOURCE_HEALTH,
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
    max_turns: int = 200
    starting_hand: int = 4
    starting_evolution_points: int = 2
    evolution_unlock_turn: int = 4
    starting_super_evolution_points: int = 2
    first_player_super_evolution_unlock_turn: int = 7
    second_player_super_evolution_unlock_turn: int = 6
    starting_health: int = 20
    validate_invariants: bool = False


@dataclass(frozen=True)
class CoreTransition:
    command: GameCommand
    events: tuple[GameEvent, ...]
    acting_player: int
    winner: int | None
    terminated: bool


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
        self.rulebook = rulebook or RuleBook()
        self.card_resolver = card_resolver
        self.random = random.Random(seed)
        self.state = GameState(players=[])
        self.logs: list[str] = []
        self.event_history: list[GameEvent] = []
        self.placeholder_ability_events: list[PlaceholderAbilityEvent] = []
        self.ability_handlers = AbilityHandlers(self)
        self._stabilizing: bool = False
        self._death_causes: dict[int, DeathCause] = {}
        self._suspended_batch: DeathBatch | None = None
        self._suspended_record: DeathRecord | None = None
        self._suspended_lw_records: list[DeathRecord] = []

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
        source = context.source
        if not isinstance(source, Unit):
            return
        if isinstance(source, Unit) and source.printed_abilities_removed:
            return
        ops = self.rulebook.operations_for(source.definition.card_id, trigger)
        if not ops:
            return
        self._start_effects(
            source.definition,
            source.entity_id,
            ops,
            controller=context.player_index,
            label=trigger.value,
        )

    def _is_ability_covered(self, context, ability) -> bool:
        card = (
            context.source.definition
            if hasattr(context.source, "definition")
            else context.source
        )
        if card is None:
            return False
        if ability is AbilityKeyword.FUSION:
            return self.rulebook.fusion_for(card.card_id) is not None
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
        if ability is AbilityKeyword.FAITH:
            return self.rulebook.faith_for(card.card_id) is not None
        if ability is AbilityKeyword.UNION_BURST:
            return bool(self.rulebook.union_bursts_for(card.card_id))
        if ability is AbilityKeyword.FANFARE:
            return bool(
                self.rulebook.operations_for(card.card_id, Trigger.FANFARE)
                or self.rulebook.union_bursts_for(card.card_id)
            )
        if ability is AbilityKeyword.EMBLEM:
            return any(
                definition.source_card_id == card.card_id
                for definition in self.rulebook._emblem_defs.values()
            )
        if ability in (AbilityKeyword.OVERFLOW, AbilityKeyword.COMBO):
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
            else:
                condition_types = (
                    ConditionType.CONTROLLER_COMBO_AT_LEAST,
                    ConditionType.OPPONENT_COMBO_AT_LEAST,
                )
                expression_types = (
                    ExprType.CONTROLLER_COMBO,
                    ExprType.OPPONENT_COMBO,
                )
                effect_kinds = (EffectKind.ADD_COMBO,)

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
                    nested = (
                        operation.earth_rite_operations
                        + operation.necromancy_operations
                        + operation.faith_operations
                        + operation.then_operations
                        + operation.else_operations
                        + operation.optional_operations
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
                operation_contains(
                    self.rulebook.operations_for(card.card_id, trigger)
                )
                for trigger in Trigger
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
                )
                if contains_kind(nested):
                    return True
                if any(
                    contains_kind(option.operations)
                    for option in operation.choose_one_options
                ):
                    return True
            return False

        return any(
            contains_kind(self.rulebook.operations_for(card.card_id, trigger))
            for trigger in Trigger
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

    def reset(self, *, seed: int | None = None) -> GameState:
        if seed is not None:
            self.random.seed(seed)
        decks = [list(deck) for deck in self.deck_lists]
        for deck in decks:
            self.random.shuffle(deck)
        self.state = GameState(
            players=[
                PlayerState(
                    deck=deck,
                    class_id=self.player_classes[index],
                    class_name=CLASS_NAMES[self.player_classes[index]],
                    health=self.config.starting_health,
                    evolution_points=self.config.starting_evolution_points,
                    super_evolution_points=self.config.starting_super_evolution_points,
                )
                for index, deck in enumerate(decks)
            ]
        )
        self.logs = [
            "=== 对局开始 ===",
            *[
                f"[玩家 {index + 1}] 职业：{player.class_name}，牌组 {len(self.deck_lists[index])} 张"
                for index, player in enumerate(self.state.players)
            ],
        ]
        self.event_history = []
        self.placeholder_ability_events = []
        self._next_modifier_id = 1
        self._next_choice_request_id = 1
        self._suspended_batch = None
        self._suspended_record = None
        self._suspended_lw_records = []
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
        self.state._next_death_sequence = 1
        self._initialize_faiths()
        for player_index in range(2):
            for _ in range(self.config.starting_hand):
                self._draw(player_index, reason="起手")
        self._start_turn(0)
        self._resolve_event_queue()
        if self.config.validate_invariants:
            self.assert_invariants()
        return self.state

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
                before = instance.value
                instance.value += rule.amount
                self._emit(
                    GameEvent(
                        EventType.FAITH_VALUE_CHANGED,
                        player_index,
                        source_id=instance.entity_id,
                        amount=rule.amount,
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
        event_start = len(self.event_history)
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
        return CoreTransition(
            command=command,
            events=tuple(self.event_history[event_start:]),
            acting_player=acting_player,
            winner=self.winner,
            terminated=self.terminated,
        )

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
                    state["burst_operations"],
                    state["burst_metadata"],
                    state["burst_gauge"],
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
            and player.turns_started >= self.config.evolution_unlock_turn
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
            if not guards and unit.can_attack_leader:
                commands.append(Attack(self.current_player, unit.entity_id, None))
            if unit.can_attack_units:
                targets = guards or [
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

    @staticmethod
    def _is_follower_attack_target(target: Unit) -> bool:
        return not target.ambush_active and not target.has_intimidate

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

    def _super_evolution_unlock_turn(self, player_index: int) -> int:
        if player_index == 0:
            return self.config.first_player_super_evolution_unlock_turn
        return self.config.second_player_super_evolution_unlock_turn

    def effective_play_cost(self, hand_card, mode_def) -> int:
        """Compute the effective play cost for a hand card with optional mode."""
        if mode_def is not None:
            return mode_def.cost
        if isinstance(hand_card, HandCard):
            return hand_card.current_cost
        return hand_card.cost

    def _is_mode_playable(self, card, player, mode_def) -> bool:
        if isinstance(card, HandCard) and card.cannot_be_played:
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
            if mode_def.is_accelerate:
                effective_type = "法术"
            elif mode_def.is_crystallize:
                effective_type = "护符"
            elif mode_def.resulting_card_type:
                effective_type = mode_def.resulting_card_type
        if effective_type in {"随从", "护符"} and len(player.board) >= self.config.max_board:
            return False
        if mode_def is None:
            return self._is_card_playable(card, player)
        if mode_def.conditions:
            ctx = EvalContext(
                controller=self.current_player,
                players=self.players,
            )
            from swb.engine.conditions import evaluate_conditions_without_target, PartialConditionResult
            result = evaluate_conditions_without_target(mode_def.conditions, ctx)
            if result is not PartialConditionResult.TRUE:
                return False
        ops = mode_def.operations if mode_def.operations else ()
        if ops:
            if any(
                op.requires_target and not self._has_candidates(op)
                for op in ops
            ):
                return False
            all_require_target = all(
                self._operation_consumes_target(op)
                for op in ops
            )
            if all_require_target and all(not self._has_candidates(op) for op in ops):
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
        player.board.append(unit)
        unit.fused_material_ids.extend(fused_material_ids)
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

    def _transform_hand_card_after_fusion(
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
                transform_event = self._transform_hand_card_after_fusion(
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
        burst_operations = tuple(
            operation
            for definition in active_bursts
            for operation in definition.operations
        )
        burst_metadata = tuple(
            (definition.kind.value, definition.threshold)
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
            )
            return
        if mode_def is not None and mode_def.is_crystallize:
            self._play_crystallize(
                card, play_cost, hand_entity_id, hand_origin,
                hand_source_origin, mode_def, fused_material_ids,
            )
            return
        if card.card_type == "法术" and mode_id == "normal":
            self._play_spell(
                card, play_cost, hand_entity_id, origin=hand_origin,
                source_origin=hand_source_origin,
                fusion_materials=fusion_materials,
            )
            return
        if card.card_type == "护符" and mode_id == "normal":
            self._play_amulet(
                card, play_cost, origin=hand_origin,
                fused_material_ids=fused_material_ids,
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
        if self.state.pending_choice is not None:
            self._suspended_action = "play_follower"
            self._suspended_action_state = {
                "unit_id": unit.entity_id,
                "mode_operations": mode_operations,
                "burst_operations": burst_operations,
                "burst_metadata": burst_metadata,
                "burst_gauge": burst_gauge,
            }
            return
        self._finish_follower_play(
            unit.entity_id,
            mode_operations,
            burst_operations,
            burst_metadata,
            burst_gauge,
        )

    def _finish_follower_play(
        self,
        unit_id: int,
        mode_operations: tuple[EffectOperation, ...],
        burst_operations: tuple[EffectOperation, ...] = (),
        burst_metadata: tuple[tuple[str, int], ...] = (),
        burst_gauge: int = 0,
    ) -> None:
        try:
            unit = self._find_board_entity(unit_id)
        except IllegalCommand:
            return
        if not isinstance(unit, Unit):
            return
        fanfare_operations = self._fanfare_operations(unit)
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
        operations = fanfare_operations + burst_operations + mode_operations
        if operations:
            label = "入场曲"
            if burst_operations:
                label = "入场曲/奥义"
            if mode_operations and (fanfare_operations or burst_operations):
                label = "入场曲/强化"
            elif mode_operations:
                label = "强化"
            self._start_effects(
                unit.definition,
                unit.entity_id,
                operations,
                label=label,
            )

    def _play_spell(
        self,
        card: CardDefinition,
        play_cost: int,
        source_entity_id: int,
        *,
        origin: CardOrigin,
        source_origin: CardOrigin | None,
        fusion_materials: tuple[FusionMaterial, ...] = (),
    ) -> None:
        self._log(self.current_player, f"使用法术 {card.name}（{play_cost}费）")
        self._dispatch_card_ability(AbilityEvent.CARD_PLAYED, card)
        self._emit(
            GameEvent(
                EventType.CARD_PLAYED,
                self.current_player,
                source_id=source_entity_id,
                metadata={"card_id": card.card_id, "card": card},
            )
        )
        operations = self.rulebook.operations_for(card.card_id, Trigger.PLAY)
        frame = self._queue_effects(
            card,
            None,
            operations,
            move_source_to_graveyard=True,
            label="法术",
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

    def _play_amulet(
        self,
        card: CardDefinition,
        play_cost: int,
        *,
        origin: CardOrigin = CardOrigin.DECK,
        fused_material_ids: tuple[int, ...] = (),
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
        self._emit(
            GameEvent(
                EventType.CARD_PLAYED,
                self.current_player,
                source_id=amulet.entity_id,
                metadata={"card_id": card.card_id, "source": amulet},
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
        operations = self.rulebook.operations_for(card.card_id, Trigger.PLAY)
        self._start_effects(card, amulet.entity_id, operations, label="入场曲")

    def _play_accelerate(
        self,
        card: CardDefinition,
        play_cost: int,
        source_entity_id: int,
        origin: CardOrigin,
        source_origin: CardOrigin | None,
        mode_def,
        fusion_materials: tuple[FusionMaterial, ...] = (),
    ) -> None:
        self._log(self.current_player, f"激奏 {card.name}（{play_cost}费）")
        self._dispatch_card_ability(AbilityEvent.CARD_PLAYED, card)
        self._emit(
            GameEvent(
                EventType.CARD_PLAYED,
                self.current_player,
                source_id=source_entity_id,
                metadata={"card_id": card.card_id, "card": card, "mode_id": mode_def.mode_id},
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
    ) -> None:
        countdown = mode_def.countdown if mode_def else None
        amulet = Amulet(
            definition=card,
            entity_id=source_entity_id,
            countdown=countdown,
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
                metadata={"card_id": card.card_id, "source": amulet, "mode_id": mode_def.mode_id},
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
    ) -> None:
        self._queue_effects(
            card,
            source_entity_id,
            operations,
            controller=controller,
            move_source_to_graveyard=move_source_to_graveyard,
            label=label,
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
            fusion_materials=fusion_materials,
            label=label,
            move_source_to_graveyard=move_source_to_graveyard,
        )
        self.state.effect_stack.append(frame)
        return frame

    def _queue_effects_from_frame(
        self,
        parent: EffectFrame,
        operations: tuple[EffectOperation, ...],
        *,
        label: str,
    ) -> EffectFrame:
        child = self._queue_effects(
            parent.source_card,
            parent.source_entity_id,
            operations,
            controller=parent.controller,
            label=label,
            fusion_materials=parent.fusion_materials,
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
        child.event_source_entity_id = parent.event_source_entity_id
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
            "target_count": operation.target_count,
            "allow_duplicate_targets": operation.allow_duplicate_targets,
            "exclude_source": operation.exclude_source,
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
        if operation.keyword is not None:
            summary["keyword"] = operation.keyword
        if operation.amount_expr is not None:
            summary["amount_expr"] = operation.amount_expr.type.value
        if operation.secondary_expr is not None:
            summary["secondary_expr"] = operation.secondary_expr.type.value
        if operation.target_count_expr is not None:
            summary["target_count_expr"] = operation.target_count_expr.type.value
        nested_counts = {
            "earth_rite": len(operation.earth_rite_operations),
            "necromancy": len(operation.necromancy_operations),
            "faith": len(operation.faith_operations),
            "then": len(operation.then_operations),
            "else": len(operation.else_operations),
            "choose_one": len(operation.choose_one_options),
            "optional": len(operation.optional_operations),
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
            "label": frame.label,
            "next_index": frame.next_index,
            "operation_count": len(frame.operations),
            "pending_target_id": frame.pending_target_id,
            "pending_target_ids": tuple(frame.pending_target_ids),
            "fusion_material_count": len(frame.fusion_materials),
            "listener_batch_id": frame.listener_batch_id,
            "listener_zone": frame.listener_activation_zone,
            "event_source_entity_id": frame.event_source_entity_id,
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
                    and len(record) == 4
                ):
                    owner, entity_id, trigger_index, event_type = record
                    record_summaries.append({
                        "owner": owner,
                        "emblem_entity_id": entity_id,
                        "trigger_index": trigger_index,
                        "trigger": event_type,
                    })
                else:
                    record_summaries.append(self._debug_value(record))
        return {
            "batch_id": batch_id,
            "record_count": len(records) if isinstance(records, list) else None,
            "records": record_summaries,
            "source_id": batch.get("source_id"),
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
                "turn": self.state.turn,
                "phase": self.state.phase.value,
                "winner": self.state.winner,
                "resolution_steps": self.state.resolution_steps,
                "next_entity_id": self.state.next_entity_id,
                "next_death_sequence": self.state._next_death_sequence,
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
            "max_mana": player.max_mana,
            "mana": player.mana,
            "fatigue": player.fatigue,
            "evolution_points": player.evolution_points,
            "super_evolution_points": player.super_evolution_points,
            "turns_started": player.turns_started,
            "evolved_this_turn": player.evolved_this_turn,
            "super_evolved_this_turn": player.super_evolved_this_turn,
            "followers_evolved_this_match": player.followers_evolved_this_match,
            "cards_played_this_turn": player.cards_played_this_turn,
            "followers_destroyed_this_turn": (
                player.followers_destroyed_this_turn
            ),
            "cooperation": player.cooperation,
            "shadows": player.shadows,
            "leader_damage_modifiers": tuple(
                (
                    modifier.modifier_id,
                    modifier.amount,
                    modifier.duration,
                    modifier.expires_for_player,
                    modifier.source_controller,
                    modifier.source_entity_id,
                    modifier.source_card_id,
                )
                for modifier in player.leader_damage_modifiers
            ),
            "next_graveyard_sequence": player._next_graveyard_sequence,
            "next_emblem_sequence": player._next_emblem_sequence,
            "next_faith_sequence": player._next_faith_sequence,
            "next_fusion_sequence": player._next_fusion_sequence,
            "deck": tuple(
                self._card_fingerprint(card)
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
            "fused_material_ids": tuple(card.fused_material_ids),
            "fusion_used_turn": card.fusion_used_turn,
            "evolutions_while_in_hand": card.evolutions_while_in_hand,
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
            })
        elif isinstance(entity, Amulet):
            base.update({
                "countdown": entity.countdown,
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
        )

    def _event_fingerprint(self, event: GameEvent) -> tuple[object, ...]:
        return (
            event.type.value,
            event.player_index,
            event.source_id,
            event.target_id,
            event.amount,
            self._fingerprint_value(event.metadata),
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
            "event_source_entity_id": frame.event_source_entity_id,
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
            operation.emblem_id,
            operation.keyword,
            operation.restriction,
            tuple(
                self._condition_fingerprint(condition)
                for condition in operation.conditions
            ),
            self._expression_fingerprint(operation.amount_expr),
            self._expression_fingerprint(operation.secondary_expr),
            None if operation.mode is None else operation.mode.value,
            operation.duration.value,
            operation.set_attack,
            operation.set_health,
            operation.target_key,
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
            operation.optional_prompt,
            tuple(
                self._operation_fingerprint(nested)
                for nested in operation.optional_operations
            ),
            operation.emblem_remove_mode,
            operation.requires_target,
            operation.target_count,
            self._expression_fingerprint(operation.target_count_expr),
            operation.allow_duplicate_targets,
            operation.exclude_source,
        )

    def _condition_fingerprint(
        self,
        condition: Condition,
    ) -> tuple[object, ...]:
        return (
            condition.type.value,
            condition.value,
            condition.keyword,
            self._board_filter_fingerprint(condition.board_filter),
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
            deck_filter.card_id,
            deck_filter.card_name,
            deck_filter.tribe_id,
            deck_filter.tribe_name,
        )

    def _board_filter_fingerprint(
        self,
        board_filter,
    ) -> tuple[object, ...] | None:
        if board_filter is None:
            return None
        return (
            board_filter.card_type,
            board_filter.cost_min,
            board_filter.cost_max,
            board_filter.card_id,
            board_filter.card_name,
            board_filter.evolved,
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
                (trigger.trigger.value, trigger.amount)
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
                )
                for trigger in definition.triggers
            ),
            tuple(
                self._operation_fingerprint(operation)
                for operation in definition.on_expire
            ),
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
        if amount <= 0:
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
        health_before = target.health
        prevented = 0
        barrier_consumed = False

        if self._super_evolution_prevents_damage(target, damage_type):
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
            return DamageResult(
                requested_amount=amount,
                prevented_amount=amount,
                actual_amount=0,
                target_health_before=health_before,
                target_health_after=health_before,
                barrier_consumed=False,
                lethal=False,
            )

        if amount > 0 and target.barrier_charges > 0:
            target.barrier_charges -= 1
            prevented = amount
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

        actual = min(amount - prevented, health_before) if not barrier_consumed else 0
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

        if actual > 0 and attacker is not None and isinstance(attacker, Unit):
            if attacker.has_keyword("必杀"):
                target.health = 0
                health_after = 0
                lethal = True
                self._death_causes[target.entity_id] = DeathCause.EFFECT_DESTROY
                self._log(controller, f"{attacker.definition.name} 的必杀破坏了 {target.definition.name}")
                self._emit(GameEvent(
                    EventType.BANE_TRIGGERED, controller,
                    source_id=attacker.entity_id,
                    target_id=target.entity_id,
                    metadata={"card_id": attacker.definition.card_id},
                ))
            if attacker.has_keyword("吸血"):
                owner_idx = self._entity_owner(attacker.entity_id)
                heal_amount = min(actual, health_before)
                owner = self.players[owner_idx]
                before_heal = owner.health
                owner.health = min(
                    owner.health + heal_amount,
                    self.config.starting_health,
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

        source_name = source.definition.name if hasattr(source, 'definition') else (source.name if source else "效果")
        self._log(
            controller,
            f"{source_name} 对 {target.definition.name} 造成 {actual} 点伤害"
            f"{'（被屏障阻止）' if barrier_consumed else ''}"
            f"（剩余生命 {target.health}）",
        )

        return DamageResult(
            requested_amount=amount,
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
        health_before = target_player.health
        modifier_amount = sum(
            modifier.amount
            for modifier in target_player.leader_damage_modifiers
            if self._leader_damage_modifier_active(modifier)
        )
        modified_amount = max(0, amount + modifier_amount)
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
            },
        ))

        if isinstance(source, Unit) and source.has_keyword("吸血"):
            owner_idx = self._entity_owner(source.entity_id)
            owner = self.players[owner_idx]
            before_heal = owner.health
            owner.health = min(
                owner.health + actual,
                self.config.starting_health,
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
            requested_amount=amount,
            actual_amount=actual,
            target_health_before=health_before,
            target_health_after=target_player.health,
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
            )
        )
        self.state._next_death_sequence += 1

    def _continue_effects(self) -> None:
        while self.state.effect_stack and self.state.pending_choice is None:
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
                if condition_state is PartialConditionResult.FALSE:
                    frame.next_index += 1
                    continue

            if operation.kind is EffectKind.TARGET_EXISTS:
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
                if operation.target_key:
                    frame._target_bindings[operation.target_key] = target_ids
                    frame._target_binding_operations[operation.target_key] = operation
                frame.defer_stabilize = True
                for selected_target_id in target_ids:
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

            if is_all_target(operation.target) and not frame.defer_stabilize:
                if operation.target is TargetKind.ALL_OWN_HAND:
                    target_ids = [
                        card.entity_id
                        for card in self._hand_cards(frame.controller)
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
                        if evaluate_target_conditions(
                            operation.conditions,
                            entity,
                            frame.controller,
                            self.players,
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

            if is_random_target(operation.target) and frame.pending_target_id is None:
                if operation.target is TargetKind.RANDOM_OWN_HAND:
                    hand_cards = self._hand_cards(frame.controller)
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
                        if evaluate_target_conditions(
                            operation.conditions,
                            entity,
                            frame.controller,
                            self.players,
                            source_entity_id=frame.source_entity_id,
                            source_fusion_count=len(frame.fusion_materials),
                        )
                    ]
                if operation.target is TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER:
                    target_ids = [entity.entity_id for entity in candidates]
                    target_ids.append(_leader_target_id(1 - frame.controller))
                    target_id = self.random.choice(target_ids)
                else:
                    chosen = pick_random(candidates, self.random) if candidates else None
                    if chosen is None:
                        frame.next_index += 1
                        continue
                    target_id = chosen.entity_id
            else:
                target_id = frame.pending_target_id
                frame.pending_target_id = None

            if operation.target_key:
                if target_id is None:
                    raise IllegalCommand(
                        "target_key requires a resolved board entity"
                    )
                try:
                    self._find_board_entity(target_id)
                except IllegalCommand as exc:
                    raise IllegalCommand(
                        "target_key requires a resolved board entity"
                    ) from exc
                frame._target_bindings[operation.target_key] = (target_id,)
                frame._target_binding_operations[operation.target_key] = operation
            self._checked_execute(operation, frame, target_id)
            frame.next_index += 1
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
        if card.card_type == "护符" and (
            operations or self.rulebook.countdown_for(card.card_id) is not None
        ):
            pass
        elif not operations:
            return False

        if not operations:
            return True

        if any(
            op.requires_target and not self._has_candidates(op)
            for op in operations
        ):
            return False

        all_require_target = all(
            self._operation_consumes_target(op)
            for op in operations
        )
        if all_require_target and all(
            not self._has_candidates(op)
            for op in operations
        ):
            return False
        return True

    def _has_candidates(self, operation: EffectOperation) -> bool:
        return self._has_candidates_for(
            operation,
            self.current_player,
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
        if operation.target == TargetKind.OWN_HAND:
            return len(self.players[controller].hand) > 1
        if operation.target == TargetKind.RANDOM_OWN_HAND:
            return len(self.players[controller].hand) > 1
        if operation.target == TargetKind.ALL_OWN_HAND:
            return True
        if operation.target == TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER:
            return True
        if is_graveyard_target(operation.target):
            candidates = graveyard_candidates(operation, controller, self.players)
            return bool(candidates)
        condition_state = evaluate_conditions_without_target(
            operation.conditions,
            EvalContext(
                controller=controller,
                players=self.players,
                source_entity_id=source_entity_id,
                source_fusion_count=source_fusion_count,
            ),
        )
        if condition_state is PartialConditionResult.FALSE:
            return True
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
                if evaluate_target_conditions(
                    operation.conditions,
                    entity,
                    controller,
                    self.players,
                    source_entity_id=source_entity_id,
                    source_fusion_count=source_fusion_count,
                )
            ]
        return bool(candidates) or (
            has_leader_choice(operation.target)
            and condition_state is not PartialConditionResult.DEPENDS_ON_TARGET
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
                EvalContext(
                    controller=controller,
                    players=self.players,
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
        return (
            is_choice_target(operation.target)
            or is_random_target(operation.target)
            or is_all_target(operation.target)
        )

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
            EvalContext(
                controller=controller,
                players=self.players,
                source_entity_id=source_entity_id,
                source_fusion_count=source_fusion_count,
            ),
        )
        if condition_state is PartialConditionResult.FALSE:
            return False
        if operation.target in {
            TargetKind.OWN_HAND,
            TargetKind.RANDOM_OWN_HAND,
            TargetKind.ALL_OWN_HAND,
        }:
            return bool(self._hand_cards(controller))
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
                if evaluate_target_conditions(
                    operation.conditions,
                    entity,
                    controller,
                    self.players,
                    source_entity_id=source_entity_id,
                    source_fusion_count=source_fusion_count,
                )
            ]
        return bool(candidates) or (
            has_leader_choice(operation.target)
            and condition_state is not PartialConditionResult.DEPENDS_ON_TARGET
        )

    def _target_options(
        self,
        operation: EffectOperation,
        controller: int,
        *,
        source_entity_id: int | None = None,
    ) -> list[ChoiceOption]:
        if operation.target == TargetKind.OWN_HAND:
            return hand_choice_options(self.players[controller])
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
        options.extend(leader_choice_options(operation.target, controller))
        return options

    def _target_choice_options(
        self, operation: EffectOperation, frame: EffectFrame
    ) -> list[ChoiceOption]:
        if operation.target == TargetKind.OWN_HAND:
            return hand_choice_options(self.players[frame.controller])
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
                if evaluate_target_conditions(
                    operation.conditions,
                    e,
                    frame.controller,
                    self.players,
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
                    leader_choice_options(operation.target, frame.controller)
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
    ) -> bool:
        try:
            self._find_board_entity(target_id)
        except IllegalCommand:
            return False
        binding_operation = frame._target_binding_operations.get(target_key)
        if binding_operation is None:
            return True
        return f"entity:{target_id}" in {
            option.option_id
            for option in self._target_choice_options(binding_operation, frame)
        }

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
            if player.turns_started < self.config.evolution_unlock_turn:
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
        unit.base_attack += stat_bonus
        unit.base_health += stat_bonus
        unit._recompute_attack()
        unit.health += stat_bonus
        unit._recompute_max()
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

        self._emit(
            GameEvent(
                EventType.ATTACK_DECLARED,
                self.current_player,
                source_id=attacker.entity_id,
                target_id=target.entity_id if target else None,
                metadata={"source": attacker, "target": target},
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
            attacker.attacks_remaining -= 1
            attacker.can_attack = False
            attacker.rush_only = False
            return

        attacker.attacks_remaining -= 1
        attacker.can_attack = False
        attacker.rush_only = False
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
            attacker.attacks_remaining -= 1
            attacker.can_attack = False
            attacker.rush_only = False
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
        if self._suspended_action_state is not None and self._suspended_action == "turn_end":
            state = self._suspended_action_state
            self._suspended_action = None
            self._suspended_action_state = None
            self._resume_end_turn(state)
            return

        player_index = self.current_player
        self._expire_modifiers(
            ModifierDuration.UNTIL_END_OF_TURN,
            player_index,
        )
        self._stabilize()
        if self.terminated:
            return
        board = tuple(self.players[player_index].board)
        self._dispatch_emblem_triggers(player_index, "turn_end")
        if self.state.pending_choice is not None:
            self._suspended_action = "turn_end"
            self._suspended_action_state = {
                "player_index": player_index,
                "remaining_ids": [
                    entity.entity_id for entity in board
                    if isinstance(entity, Unit)
                ],
            }
            return
        for idx, unit in enumerate(board):
            self._dispatch_ability(
                AbilityEvent.TURN_ENDED, unit, player_index=player_index
            )
            ops = (
                ()
                if isinstance(unit, Unit) and unit.printed_abilities_removed
                else self.rulebook.operations_for(
                    unit.definition.card_id, Trigger.TURN_END
                )
            )
            if ops:
                self._start_effects(unit.definition, unit.entity_id, ops, label="回合结束")
                if self.state.pending_choice is not None:
                    self._suspended_action = "turn_end"
                    self._suspended_action_state = {
                        "player_index": player_index,
                        "remaining_ids": [
                            e.entity_id
                            for e in board[idx + 1:]
                        ],
                    }
                    return
        self._emit(GameEvent(EventType.TURN_ENDED, player_index))
        self._log(player_index, "结束回合")
        self.players[player_index].cards_played_this_turn = 0
        self.state.active_player = 1 - player_index
        self.state.turn += 1
        self._start_turn(self.current_player)

    def _resume_end_turn(self, state: dict) -> None:
        self._suspended_action = None
        self._suspended_action_state = None
        player_index = state["player_index"]
        remaining_ids = state.get("remaining_ids", [])

        while remaining_ids:
            entity_id = remaining_ids[0]
            remaining_ids = remaining_ids[1:]
            try:
                unit = self._find_board_entity(entity_id)
            except IllegalCommand:
                continue
            if not isinstance(unit, Unit):
                continue
            owner = self._entity_owner(entity_id)
            if owner != player_index:
                continue
            self._dispatch_ability(
                AbilityEvent.TURN_ENDED, unit, player_index=player_index
            )
            ops = (
                ()
                if isinstance(unit, Unit) and unit.printed_abilities_removed
                else self.rulebook.operations_for(
                    unit.definition.card_id, Trigger.TURN_END
                )
            )
            if ops:
                self._start_effects(unit.definition, unit.entity_id, ops, label="回合结束")
                if self.state.pending_choice is not None:
                    self._suspended_action = "turn_end"
                    self._suspended_action_state = {
                        "player_index": player_index,
                        "remaining_ids": remaining_ids,
                    }
                    return
        self._emit(GameEvent(EventType.TURN_ENDED, player_index))
        self._log(player_index, "结束回合")
        self.players[player_index].cards_played_this_turn = 0
        self.state.active_player = 1 - player_index
        self.state.turn += 1
        self._start_turn(self.current_player)

    def _choose(self, command: Choose) -> None:
        request = self.state.pending_choice
        if request is None:
            raise IllegalCommand("There is no pending choice")
        if command.option_id not in {option.option_id for option in request.options}:
            raise IllegalCommand("Choice option is invalid")
        option = next(
            option for option in request.options if option.option_id == command.option_id
        )
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
                self.state.pending_choice = None
                self.state.phase = Phase.MAIN
                self._resolve_choose_one_choice(frame, option.option_id)
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
            frame.pending_target_id = (
                _leader_target_id(option.leader_player_index)
                if option.leader_player_index is not None
                else option.entity_id
            )
        self.state.pending_choice = None
        self.state.phase = Phase.MAIN
        if self.state.effect_stack:
            self._continue_effects()
        self._try_spellboost_hand()

    def _start_turn(self, player_index: int) -> None:
        if self._suspended_action_state is not None and self._suspended_action == "turn_start":
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
        player.evolved_this_turn = False
        player.super_evolved_this_turn = False
        player.cards_played_this_turn = 0
        player.followers_destroyed_this_turn = 0
        player.max_mana = min(self.config.max_mana, player.max_mana + 1)
        player.mana = player.max_mana
        self._tick_countdowns(player_index)
        self._tick_emblem_countdowns(player_index)
        board = tuple(player.board)
        if self.state.pending_choice is not None:
            self._suspended_action = "turn_start"
            self._suspended_action_state = {
                "player_index": player_index,
                "phase": "emblem_triggers",
                "remaining_ids": [
                    entity.entity_id for entity in board
                    if isinstance(entity, Unit)
                ],
            }
            return
        self._dispatch_emblem_triggers(player_index, "turn_start")
        if self.state.pending_choice is not None:
            self._suspended_action = "turn_start"
            self._suspended_action_state = {
                "player_index": player_index,
                "phase": "board",
                "remaining_ids": [
                    entity.entity_id for entity in board
                    if isinstance(entity, Unit)
                ],
            }
            return
        for idx, unit in enumerate(board):
            if not isinstance(unit, Unit):
                continue
            unit.can_attack = True
            unit.attacks_remaining = 1
            unit.rush_only = False
            unit.summoned_this_turn = False
            self._dispatch_ability(
                AbilityEvent.TURN_STARTED, unit, player_index=player_index
            )
            ops = (
                ()
                if isinstance(unit, Unit) and unit.printed_abilities_removed
                else self.rulebook.operations_for(
                    unit.definition.card_id, Trigger.TURN_START
                )
            )
            if ops:
                self._start_effects(unit.definition, unit.entity_id, ops, label="回合开始")
                if self.state.pending_choice is not None:
                    self._suspended_action = "turn_start"
                    self._suspended_action_state = {
                        "player_index": player_index,
                        "phase": "board",
                        "remaining_ids": [
                            e.entity_id
                            for e in board[idx + 1:]
                            if isinstance(e, Unit)
                        ],
                    }
                    return
        self._finish_start_turn(player_index)

    def _finish_start_turn(self, player_index: int) -> None:
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
                AbilityEvent.TURN_STARTED, card, player_index=player_index
            )
        self._continue_turn_start_invocations(
            player_index,
            self._turn_start_invocation_candidates(player_index),
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
            EvalContext(controller=player_index, players=self.players),
        )
        return result is PartialConditionResult.TRUE

    def _suspend_start_turn_invocation(
        self,
        player_index: int,
        phase: str,
        remaining_card_ids: list[int],
        *,
        invoked_entity_id: int | None = None,
    ) -> None:
        self._suspended_action = "turn_start"
        self._suspended_action_state = {
            "player_index": player_index,
            "phase": phase,
            "remaining_card_ids": list(remaining_card_ids),
        }
        if invoked_entity_id is not None:
            self._suspended_action_state["invoked_entity_id"] = invoked_entity_id

    def _finish_invoked_card(self, player_index: int, entity_id: int) -> None:
        try:
            unit = self._find_board_entity(entity_id)
        except IllegalCommand:
            return
        if not isinstance(unit, Unit) or self._entity_owner(entity_id) != player_index:
            return
        operations = self.rulebook.operations_for(
            unit.definition.card_id,
            Trigger.INVOKE,
        )
        if operations:
            self._start_effects(
                unit.definition,
                unit.entity_id,
                operations,
                controller=player_index,
                label="瞬念召唤",
            )

    def _continue_turn_start_invocations(
        self,
        player_index: int,
        remaining_card_ids: list[int],
    ) -> None:
        player = self.players[player_index]
        while remaining_card_ids:
            if len(player.board) >= self.config.max_board:
                break
            eligible_card_ids = [
                card_id
                for card_id in remaining_card_ids
                if self._invocation_conditions_met(player_index, card_id)
                and any(card.card_id == card_id for card in player.deck)
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
            unit = self._summon_follower_to_board(
                player_index,
                card,
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
            self._resolve_event_queue()
            if self.state.pending_choice is not None:
                self._suspend_start_turn_invocation(
                    player_index,
                    "invocation_source",
                    remaining_card_ids,
                    invoked_entity_id=unit.entity_id,
                )
                return
            self._finish_invoked_card(player_index, unit.entity_id)
            if self.state.pending_choice is not None:
                self._suspend_start_turn_invocation(
                    player_index,
                    "invocation_scan",
                    remaining_card_ids,
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
        if phase == "invocation_source":
            self._finish_invoked_card(
                player_index,
                state["invoked_entity_id"],
            )
            if self.state.pending_choice is not None:
                self._suspend_start_turn_invocation(
                    player_index,
                    "invocation_scan",
                    state.get("remaining_card_ids", []),
                )
                return
            self._continue_turn_start_invocations(
                player_index,
                state.get("remaining_card_ids", []),
            )
            return
        if phase == "invocation_scan":
            self._continue_turn_start_invocations(
                player_index,
                state.get("remaining_card_ids", []),
            )
            return
        remaining_ids = state.get("remaining_ids", [])
        if phase == "emblem_triggers":
            self._dispatch_emblem_triggers(player_index, "turn_start")
            if self.state.pending_choice is not None:
                self._suspended_action = "turn_start"
                self._suspended_action_state = {
                    "player_index": player_index,
                    "phase": "board",
                    "remaining_ids": remaining_ids,
                }
                return
            phase = "board"
        if phase == "board":
            while remaining_ids:
                entity_id = remaining_ids[0]
                remaining_ids = remaining_ids[1:]
                try:
                    unit = self._find_board_entity(entity_id)
                except IllegalCommand:
                    continue
                if not isinstance(unit, Unit):
                    continue
                owner = self._entity_owner(entity_id)
                if owner != player_index:
                    continue
                unit.can_attack = True
                unit.attacks_remaining = 1
                unit.rush_only = False
                unit.summoned_this_turn = False
                self._dispatch_ability(
                    AbilityEvent.TURN_STARTED, unit, player_index=player_index
                )
                ops = (
                    ()
                    if isinstance(unit, Unit) and unit.printed_abilities_removed
                    else self.rulebook.operations_for(
                        unit.definition.card_id, Trigger.TURN_START
                    )
                )
                if ops:
                    self._start_effects(unit.definition, unit.entity_id, ops, label="回合开始")
                    if self.state.pending_choice is not None:
                        self._suspended_action = "turn_start"
                        self._suspended_action_state = {
                            "player_index": player_index,
                            "phase": "board",
                            "remaining_ids": remaining_ids,
                        }
                        return
        self._finish_start_turn(player_index)

    def _draw(self, player_index: int, *, reason: str) -> None:
        player = self.players[player_index]
        if player.deck:
            card = player.deck.pop()
            if len(player.hand) < self.config.max_hand:
                self._append_hand_card(player, card, origin=CardOrigin.DECK)
                self._emit(
                    GameEvent(
                        EventType.CARD_DRAWN,
                        player_index,
                        metadata={"card_id": card.card_id},
                    )
                )
                self._log(player_index, f"{reason}：{card.name}")
            else:
                self._send_to_graveyard(
                    player_index, card, "overdraw",
                    origin=CardOrigin.DECK,
                )
                self._log(player_index, f"{reason}：{card.name}，手牌已满而被弃置")
            return
        player.fatigue += 1
        player.health -= player.fatigue
        self._log(
            player_index,
            f"牌库耗尽，受到 {player.fatigue} 点疲劳伤害（生命 {player.health}）",
        )

    def _draw_filtered(
        self,
        player_index: int,
        *,
        deck_filter: DeckFilter | None = None,
        reason: str,
    ) -> None:
        player = self.players[player_index]
        candidates = [
            index
            for index, card in enumerate(player.deck)
            if deck_filter is None or deck_filter.matches(card)
        ]
        if not candidates:
            self._log(player_index, f"{reason}：没有符合条件的卡牌")
            return
        index = self.random.choice(candidates)
        card = player.deck.pop(index)
        if len(player.hand) < self.config.max_hand:
            self._append_hand_card(player, card, origin=CardOrigin.DECK)
            self._emit(
                GameEvent(
                    EventType.CARD_DRAWN,
                    player_index,
                    metadata={
                        "card_id": card.card_id,
                        "filtered": True,
                        "card_type_filter": None if deck_filter is None else deck_filter.card_type,
                        "class_id_filter": None if deck_filter is None else deck_filter.class_id,
                        "class_name_filter": None if deck_filter is None else deck_filter.class_name,
                        "cost_min_filter": None if deck_filter is None else deck_filter.cost_min,
                        "cost_max_filter": None if deck_filter is None else deck_filter.cost_max,
                        "card_id_filter": None if deck_filter is None else deck_filter.card_id,
                        "card_name_filter": None if deck_filter is None else deck_filter.card_name,
                    },
                )
            )
            self._log(player_index, f"{reason}：{card.name}")
        else:
            self._send_to_graveyard(
                player_index,
                card,
                "overdraw",
                origin=CardOrigin.DECK,
            )
            self._log(player_index, f"{reason}：{card.name}，手牌已满而被弃置")

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
        return EvalContext(
            controller=frame.controller,
            players=self.players,
            source_entity_id=frame.source_entity_id,
            target_entity_id=target_id,
            source_card_id=frame.source_card_id,
            source_fusion_count=len(frame.fusion_materials),
        )

    def _resolve_amount(self, operation: EffectOperation, ctx: EvalContext) -> int:
        if operation.amount_expr is not None:
            return evaluate_expression(operation.amount_expr, ctx)
        return operation.amount

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
    def _operation_requires_source_in_play(
        operation: EffectOperation,
    ) -> bool:
        return (
            (
                operation.target is TargetKind.SELF
                and operation.kind in _SOURCE_REQUIRED_SELF_TARGET_EFFECTS
            )
            or _expression_depends_on_source(operation.amount_expr)
            or _expression_depends_on_source(operation.secondary_expr)
            or _expression_depends_on_source(operation.target_count_expr)
            or any(
                _condition_depends_on_source(condition)
                for condition in operation.conditions
            )
        )

    def _checked_execute(
        self, operation: EffectOperation, frame: EffectFrame, target_id: int | None,
    ) -> None:
        if (
            self._operation_requires_source_in_play(operation)
            and not self._source_entity_in_play(frame)
        ):
            return
        if operation.target is TargetKind.SELF:
            target_id = frame.source_entity_id
        elif operation.target is TargetKind.EVENT_SOURCE:
            target_id = frame.event_source_entity_id
            if target_id is None:
                return
            if operation.kind in _EVENT_SOURCE_BOARD_EFFECTS:
                try:
                    self._find_board_entity(target_id)
                except IllegalCommand:
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
                    return
        amount = self._resolve_amount(operation, ctx)
        secondary = self._resolve_secondary(operation, ctx)
        if operation.kind in (EffectKind.HEAL_LEADER, EffectKind.HEAL_UNIT):
            amount = max(0, amount)
        if operation.amount_expr is not None or operation.secondary_expr is not None or amount != operation.amount or secondary != operation.secondary_amount:
            resolved = replace(
                operation,
                amount=amount,
                secondary_amount=secondary,
                amount_expr=None,
                secondary_expr=None,
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
            for _ in range(effect.amount):
                self._draw(
                    draw_player,
                    reason=f"{name} {frame.label}抽牌",
                )
        elif effect.kind is EffectKind.DRAW_FILTERED:
            draw_player = (
                1 - frame.controller
                if effect.target is TargetKind.ENEMY_LEADER
                else frame.controller
            )
            for _ in range(effect.amount):
                self._draw_filtered(
                    draw_player,
                    deck_filter=effect.deck_filter,
                    reason=f"{name} {frame.label}抽牌",
                )
        elif effect.kind is EffectKind.HEAL_LEADER:
            before = player.health
            player.health = min(self.config.starting_health, player.health + effect.amount)
            actual_heal = player.health - before
            self._log(
                frame.controller,
                f"{name} {frame.label}回复 {actual_heal} 点生命"
                f"（生命 {player.health}）",
            )
            if actual_heal > 0:
                self._emit(GameEvent(
                    EventType.LEADER_HEALED,
                    frame.controller,
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
        elif effect.kind is EffectKind.DAMAGE_LEADER:
            is_enemy = effect.target is TargetKind.ENEMY_LEADER
            target_idx = 1 - frame.controller if is_enemy else frame.controller
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
        elif effect.kind is EffectKind.RESTORE_MANA:
            restored = min(effect.amount, player.max_mana - player.mana)
            player.mana += restored
            self._log(
                frame.controller,
                f"{name} {frame.label}回复 {restored} 点能量",
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
            target_player.mana = min(target_player.mana, target_player.max_mana)
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
            self._log(
                frame.controller,
                f"属性变化 {effect.amount}/{effect.secondary_amount}",
            )
        elif effect.kind is EffectKind.DESTROY:
            target = self._find_board_entity(target_id)
            if isinstance(target, Unit):
                if self._super_evolution_prevents_effect_destroy(target):
                    self._log(
                        frame.controller,
                        f"{target.definition.name} 的超进化保护阻止了效果破坏",
                    )
                    return
                self._death_causes[target.entity_id] = DeathCause.EFFECT_DESTROY
                target.health = 0
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
        elif effect.kind is EffectKind.SUMMON:
            self._execute_summon(effect, frame)
        elif effect.kind is EffectKind.BANISH:
            self._execute_banish(target_id, frame)
        elif effect.kind is EffectKind.ADD_CARD:
            self._execute_add_card(effect, frame)
        elif effect.kind is EffectKind.RETURN_TO_HAND:
            self._execute_return_to_hand(target_id, frame)
        elif effect.kind is EffectKind.RETURN_TO_DECK:
            self._execute_return_to_deck(target_id, frame)
        elif effect.kind is EffectKind.REDUCE_COUNTDOWN:
            self._execute_reduce_countdown(effect, frame, target_id)
        elif effect.kind is EffectKind.DISCARD:
            self._execute_discard(target_id, frame)
        elif effect.kind is EffectKind.ADD_KEYWORD:
            self._execute_keyword_change(effect, frame, target_id, add=True)
        elif effect.kind is EffectKind.REMOVE_KEYWORD:
            self._execute_keyword_change(effect, frame, target_id, add=False)
        elif effect.kind is EffectKind.REMOVE_ALL_ABILITIES:
            self._execute_remove_all_abilities(frame, target_id)
        elif effect.kind is EffectKind.ADD_LEADER_DAMAGE_MODIFIER:
            self._execute_add_leader_damage_modifier(effect, frame)
        elif effect.kind is EffectKind.CHANGE_COST:
            self._execute_change_cost(effect, frame, target_id)
        elif effect.kind is EffectKind.TRANSFORM:
            self._execute_transform(effect, frame, target_id)
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
        elif effect.kind is EffectKind.ADD_EARTH_SIGILS:
            self._execute_add_earth_sigils(effect, frame)
        elif effect.kind is EffectKind.EARTH_RITE:
            self._execute_earth_rite(effect, frame)
        elif effect.kind is EffectKind.CONSUME_FAITH:
            self._execute_consume_faith(effect, frame)
        elif effect.kind is EffectKind.GRANT_FAITH_ABILITY:
            self._execute_grant_faith_ability(effect, frame)
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
        elif effect.kind is EffectKind.CONDITIONAL:
            self._execute_conditional(effect, frame)
        elif effect.kind is EffectKind.CHOOSE_ONE:
            self._execute_choose_one(effect, frame)
        elif effect.kind is EffectKind.OPTIONAL:
            self._execute_optional(effect, frame)
        elif effect.kind is EffectKind.TARGET_EXISTS:
            self._execute_target_exists(effect, frame)
        else:
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
            trigger_abilities=True,
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
            frame.source_entity_id
            if effect.target is TargetKind.SELF
            else target_id
        )
        target = self._find_board_entity(resolved_id)
        if not isinstance(target, Unit):
            raise IllegalCommand("Keyword target must be a follower")
        if add:
            target.add_keyword(
                effect.keyword,
                duration=effect.duration.value,
                expires_for_player=_expires_for_player(
                    effect.duration,
                    frame.controller,
                    self.state.active_player,
                ),
            )
            verb = "获得"
        else:
            target.remove_keyword(
                effect.keyword,
                duration=effect.duration.value,
                expires_for_player=_expires_for_player(
                    effect.duration,
                    frame.controller,
                    self.state.active_player,
                ),
            )
            verb = "失去"
        self._log(
            frame.controller,
            f"{target.definition.name} {verb}关键词 {effect.keyword}",
        )

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
            },
        ))

    def _execute_change_cost(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        if effect.mode is None:
            raise IllegalCommand("CHANGE_COST requires a mode")
        hand_card = self._find_hand_card(frame.controller, target_id)
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

    def _execute_transform(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        if effect.card_id is None:
            raise IllegalCommand("TRANSFORM requires a card_id")
        if self.card_resolver is None:
            raise IllegalCommand("No card_resolver registered for TRANSFORM")
        target = self._find_board_entity(target_id)
        if not isinstance(target, Unit):
            raise IllegalCommand(
                "TRANSFORM currently supports follower targets only"
            )
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
        if replacement.card_type != "随从":
            raise IllegalCommand(
                "TRANSFORM currently supports follower-to-follower only"
            )
        old_name = target.definition.name
        can_attack = target.can_attack
        attacks_remaining = target.attacks_remaining
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
        target.attacks_remaining = attacks_remaining
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
        target.attack_restrictions.clear()
        target.targeting_restrictions.clear()
        target.printed_abilities_removed = False
        self._apply_initial_keyword_overrides(target)
        target._synchronize_keyword_state()
        self._death_causes.pop(target.entity_id, None)
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
            return
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
        self._add_emblem_to_player(frame.controller, emblem_def, frame.source_card_id)

    def _add_emblem_to_player(self, player_index: int, emblem_def, source_card_id: int):
        from swb.engine.emblem import EmblemStacking
        from swb.engine.state import EmblemInstance
        player = self.players[player_index]

        if emblem_def.stacking is EmblemStacking.IGNORE:
            existing = [e for e in player.emblems if e.emblem_id == emblem_def.emblem_id]
            if existing:
                return
        elif emblem_def.stacking is EmblemStacking.REPLACE:
            for existing in tuple(player.emblems):
                if existing.emblem_id == emblem_def.emblem_id:
                    self._remove_emblem_instance(
                        player_index,
                        existing,
                        removal_cause="replace",
                    )

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

    def _execute_remove_emblem(self, effect: EffectOperation, frame: EffectFrame) -> None:
        emblem_id = effect.emblem_id
        if not emblem_id:
            raise IllegalCommand("REMOVE_EMBLEM requires emblem_id")
        player = self.players[frame.controller]
        removed = [e for e in player.emblems if e.emblem_id == emblem_id]
        if not removed:
            return
        targets = removed if effect.emblem_remove_mode == "all" else removed[:1]
        for target in targets:
            self._remove_emblem_instance(
                frame.controller,
                target,
                removal_cause="effect",
            )

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
        condition_state = evaluate_conditions_without_target(
            operation.conditions,
            EvalContext(
                controller=controller,
                players=self.players,
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
        if operation.target is TargetKind.SELF:
            if source_entity_id is None:
                return False
            try:
                self._find_board_entity(source_entity_id)
            except IllegalCommand:
                return False
            return True
        if operation.target is TargetKind.PREVIOUS_TARGET:
            return False
        if operation.target in {
            TargetKind.OWN_HAND,
            TargetKind.RANDOM_OWN_HAND,
            TargetKind.ALL_OWN_HAND,
        }:
            return bool(self._hand_cards(controller))
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
                    if evaluate_target_conditions(
                        operation.conditions,
                        entity,
                        controller,
                        self.players,
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
        event_source, event_definition = self._event_source_card(event)
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
                        and not definition.event_filter.matches(
                            event_definition,
                            event_source,
                            event.metadata.get("keywords"),
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
            if definition.conditions:
                result = evaluate_conditions_without_target(
                    definition.conditions,
                    EvalContext(
                        controller=owner,
                        players=self.players,
                        source_entity_id=entity_id,
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
            self.event_history.append(GameEvent(
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
            )
            frame.listener_batch_id = batch_id
            frame.listener_activation_owner = owner
            frame.listener_activation_zone = zone
            frame.listener_activation_entity_id = entity_id
            frame.listener_activation_card_id = card_id
            frame.listener_activation_definition_index = definition_index
            frame.event_source_entity_id = batch.get("event_source_id")
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
    ) -> None:
        records: list[tuple[int, int, int, str]] = []
        for pi in (0, 1):
            player = self.players[pi]
            for ei in player.emblems:
                for ti, tr in enumerate(ei.definition.triggers):
                    if tr.trigger == event_type:
                        if ei.can_activate(ti) and self._check_emblem_trigger_scope(
                            pi, tr, event_type, event_player,
                        ):
                            records.append((pi, ei.entity_id, ti, event_type))
        records.sort(
            key=lambda record: self._emblem_order_key(
                record[0],
                record[1],
                record[2],
            )
        )
        if not records:
            return
        batch_id = self._next_emblem_batch_id
        self._next_emblem_batch_id += 1
        self._emblem_batches[batch_id] = {
            "records": records,
            "source_id": source_id,
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
            player_index, entity_id, trigger_index, event_type = records.pop(0)
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
            if tr.conditions:
                ctx = EvalContext(
                    controller=player_index,
                    players=self.players,
                    source_entity_id=batch.get("source_id"),
                )
                result = evaluate_conditions_without_target(
                    tr.conditions,
                    ctx,
                )
                if result is not PartialConditionResult.TRUE:
                    continue
            if not tr.operations or not any(
                self._emblem_operation_can_start(
                    operation,
                    player_index,
                    batch.get("source_id"),
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
            )
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
        self.event_history.append(GameEvent(
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

    def _resolve_choose_one_choice(self, frame, option_id: str) -> None:
        for opt in frame._decision_meta.get("choose_one_options", ()):
            if f"choose_one:{opt.option_id}" == option_id:
                self._queue_effects_from_frame(
                    frame,
                    opt.operations,
                    label=f"{frame.label}/choose_one/{opt.label}",
                )
                return

    def _execute_conditional(self, effect, frame) -> None:
        ctx = EvalContext(
            controller=frame.controller,
            players=self.players,
            source_entity_id=frame.source_entity_id,
            source_fusion_count=len(frame.fusion_materials),
        )
        result = evaluate_conditions_without_target(effect.conditions, ctx)
        branch_ops = effect.then_operations if result is PartialConditionResult.TRUE else effect.else_operations
        if branch_ops:
            self._queue_effects_from_frame(
                frame,
                branch_ops,
                label=f"{frame.label}/conditional",
            )

    def _execute_target_exists(self, effect, frame) -> None:
        branch_ops = (
            effect.then_operations
            if self._target_exists_for(
                effect,
                frame.controller,
                source_entity_id=frame.source_entity_id,
                source_fusion_count=len(frame.fusion_materials),
            )
            else effect.else_operations
        )
        if branch_ops:
            self._queue_effects_from_frame(
                frame,
                branch_ops,
                label=f"{frame.label}/target_exists",
            )

    def _execute_choose_one(self, effect, frame) -> None:
        legal_options = []
        for opt in effect.choose_one_options:
            if opt.conditions:
                ctx = EvalContext(
                    controller=frame.controller,
                    players=self.players,
                    source_entity_id=frame.source_entity_id,
                    source_fusion_count=len(frame.fusion_materials),
                )
                result = evaluate_conditions_without_target(opt.conditions, ctx)
                if result is not PartialConditionResult.TRUE:
                    continue
            if opt.operations:
                if any(
                    op.requires_target
                    and not self._has_candidates_for(
                        op,
                        frame.controller,
                        source_entity_id=frame.source_entity_id,
                        source_fusion_count=len(frame.fusion_materials),
                    )
                    for op in opt.operations
                ):
                    continue
                all_need_target = all(
                    self._operation_consumes_target(op)
                    for op in opt.operations
                )
                if all_need_target and all(
                    not self._has_candidates_for(
                        op,
                        frame.controller,
                        source_entity_id=frame.source_entity_id,
                        source_fusion_count=len(frame.fusion_materials),
                    )
                    for op in opt.operations
                ):
                    continue
            legal_options.append(opt)
        if not legal_options:
            return

        if frame.auto_resolve_choices:
            chosen = self.random.choice(legal_options)
            self._log(frame.controller, f"自动选择：{chosen.label}")
            self._queue_effects_from_frame(
                frame,
                chosen.operations,
                label=f"{frame.label}/choose_one/{chosen.label}",
            )
            return

        request_id = self._allocate_choice_request_id()
        frame._decision_meta["choose_one_options"] = legal_options
        self.state.pending_choice = ChoiceRequest(
            player_index=frame.controller,
            prompt=f"{frame.source_name} \u9009\u62e9\u4e00\u9879",
            options=tuple(
                ChoiceOption(option_id=f"choose_one:{opt.option_id}", label=opt.label)
                for opt in legal_options
            ),
            continuation_id=f"{frame.source_card_id}:{frame.next_index}",
            choice_kind=ChoiceKind.MODE,
            request_id=request_id,
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
        if definition.on_expire:
            source_card_id = definition.source_card_id
            source_card = (
                self.card_resolver(source_card_id)
                if self.card_resolver
                else None
            )
            if source_card is None:
                source_card = type("_EmblemCard", (), {
                    "card_id": source_card_id,
                    "name": f"纹章_{ei.emblem_id}",
                })()
            frame = self._queue_effects(
                source_card, None,
                definition.on_expire,
                controller=player_index,
                label=f"纹章 on_expire",
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
        player = self.players[frame.controller]
        if len(player.board) >= self.config.max_board:
            self._log(frame.controller, f"{frame.source_name} 召唤失败：场地已满")
            return
        if card_def.card_type == "随从":
            origin = origin_for_summoned_card(card_def)
            unit = self._summon_follower_to_board(
                frame.controller,
                card_def,
                summon_cause="effect_summon",
                origin=origin,
            )
            if unit is None:
                self._log(
                    frame.controller,
                    f"{frame.source_name} 召唤失败：场地已满",
                )
                return
            self._log(
                frame.controller,
                f"{frame.source_name} 召唤 {card_def.name} ({unit.attack}/{unit.health})",
            )
            self._emit(
                GameEvent(
                    EventType.FOLLOWER_SUMMONED,
                    frame.controller,
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
            self._log(
                frame.controller,
                f"{frame.source_name} 召唤护符 {card_def.name}",
            )
            self._emit(
                GameEvent(
                    EventType.AMULET_ENTERED,
                    frame.controller,
                    source_id=amulet.entity_id,
                    metadata={"source": amulet},
                )
            )
            self._initialize_earth_sigil(amulet, frame.controller)
        else:
            self._log(
                frame.controller,
                f"{frame.source_name} 召唤失败：{card_def.card_type} 类型不可召唤",
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

    def _execute_banish(self, target_id: int | None, frame: EffectFrame) -> None:
        entity = self._find_board_entity(target_id)
        owner = self._entity_owner(entity.entity_id)
        player = self.players[owner]
        if entity not in player.board:
            return
        player.board.remove(entity)
        player.banished.append(entity.definition)
        self._log(
            owner,
            f"{entity.definition.name} 被消失",
        )
        self._emit(
            GameEvent(
                EventType.CARD_BANISHED,
                owner,
                source_id=entity.entity_id,
                metadata={"source": entity},
            )
        )
        self._emit(
            GameEvent(
                EventType.ENTITY_LEFT_PLAY,
                owner,
                source_id=entity.entity_id,
                metadata={
                    "source": entity,
                    "definition": entity.definition,
                    "card_id": entity.definition.card_id,
                    "card_type": entity.definition.card_type,
                    "owner": owner,
                    "cause": DeathCause.BANISH.value,
                },
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
        self._append_hand_card(player, card_def, origin=origin)
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
                    "card": card_def,
                    "origin": origin.value,
                    "derived": is_derived(origin),
                    "token": is_token_definition(card_def) or origin is CardOrigin.TOKEN,
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

    def _execute_reduce_countdown(
        self,
        effect: EffectOperation,
        frame: EffectFrame,
        target_id: int | None,
    ) -> None:
        if target_id is None:
            return
        target = self._find_board_entity(target_id)
        if not isinstance(target, Amulet) or target.countdown is None:
            return
        previous = target.countdown
        target.countdown = max(0, previous - effect.amount)
        self._log(
            frame.controller,
            f"{target.definition.name} 倒数由 {previous} 减为 {target.countdown}",
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
                        metadata={"card_id": card_def.card_id, "card": card_def},
                    )
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
            EventType.FOLLOWER_SUMMONED: "follower_summoned",
            EventType.FOLLOWER_EVOLVED: "follower_evolved",
            EventType.FOLLOWER_DESTROYED: "follower_destroyed",
            EventType.AMULET_DESTROYED: "amulet_destroyed",
            EventType.DEATH_BATCH_END: "death_batch_end",
            EventType.LEADER_HEALED: "leader_healed",
            EventType.AMULET_ACTIVATED: "amulet_activated",
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
            self.event_history.append(event)
            if event.type is EventType.FOLLOWER_DESTROYED:
                self._resolve_super_evolution_attack_bonus(event)
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
            ability_event = event_to_ability.get(event.type)
            if event.metadata.get("trigger_abilities") is False:
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
            self._execute_last_words(record, batch)
            self._continue_effects()
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
            batch = self._collect_death_batch()
            if not batch.records:
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
            self._suspended_batch = batch
            self._suspended_lw_records = list(lw_records)
            self._resolve_event_queue()
            if self.state.pending_choice is not None:
                return

            self._continue_batch_lws()
            if self.state.pending_choice is not None:
                return

        self._check_game_over()

    def _collect_death_batch(self) -> DeathBatch:
        records: list[DeathRecord] = []
        batch_id = len(self.state.death_queue) + 1

        for player_index, player in enumerate(self.players):
            for pos, entity in enumerate(tuple(player.board)):
                if isinstance(entity, Unit) and entity.health <= 0:
                    cause = self._death_causes.pop(entity.entity_id, DeathCause.ZERO_HEALTH)
                    record = DeathRecord(
                        owner=player_index,
                        entity_id=entity.entity_id,
                        card_id=entity.definition.card_id,
                        card_name=entity.definition.name,
                        card_type="随从",
                        definition=entity.definition,
                        cause=cause,
                        board_position=pos,
                        allows_last_words=not entity.printed_abilities_removed,
                        effective_keywords=entity.effective_keywords,
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
                    records.append(record)

        return DeathBatch(records=records, batch_id=batch_id)

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
        }

    def _last_words_event_metadata(
        self,
        batch: DeathBatch,
        record: DeathRecord,
    ) -> dict[str, object]:
        return self._death_event_metadata(batch, record)

    def _execute_last_words(self, record: DeathRecord, batch: DeathBatch) -> None:
        self._step()
        self._emit(GameEvent(
            EventType.LAST_WORDS_START,
            record.owner,
            source_id=record.entity_id,
            metadata=self._last_words_event_metadata(batch, record),
        ))
        self._log(record.owner, f"{record.card_name} 谢幕曲开始")

        operations = self.rulebook.operations_for(record.card_id, Trigger.LAST_WORDS)
        if not operations and record.card_type == "护符":
            operations = self.rulebook.operations_for(record.card_id, Trigger.COUNTDOWN_EXPIRED)

        if operations:
            self._queue_effects(
                record.definition,
                record.entity_id,
                operations,
                controller=record.owner,
                label="谢幕曲",
            )

    def _check_game_over(self) -> None:
        dead = [index for index, player in enumerate(self.players) if player.health <= 0]
        if dead:
            self.state.winner = None if len(dead) == 2 else 1 - dead[0]
            self.state.phase = Phase.FINISHED
        elif self.turn > self.config.max_turns:
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
            if state.phase is not Phase.AWAITING_CHOICE:
                raise IllegalCommand(
                    "Invariant failed: pending_choice requires AWAITING_CHOICE phase"
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
        elif state.phase is Phase.AWAITING_CHOICE:
            raise IllegalCommand("Invariant failed: AWAITING_CHOICE without pending_choice")

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
            if not (0 <= player.max_mana <= self.config.max_mana):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} max_mana out of range: {player.max_mana}"
                )
            if not (0 <= player.mana <= player.max_mana):
                raise IllegalCommand(
                    f"Invariant failed: {prefix} mana out of range: {player.mana}/{player.max_mana}"
                )
            if player.health < 0 or player.health > self.config.starting_health:
                raise IllegalCommand(
                    f"Invariant failed: {prefix} health out of range: {player.health}"
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
            for name, value in (
                ("fatigue", player.fatigue),
                ("evolution_points", player.evolution_points),
                ("super_evolution_points", player.super_evolution_points),
                ("turns_started", player.turns_started),
                ("followers_evolved_this_match", player.followers_evolved_this_match),
                ("cards_played_this_turn", player.cards_played_this_turn),
                ("followers_destroyed_this_turn", player.followers_destroyed_this_turn),
                ("cooperation", player.cooperation),
                ("shadows", player.shadows),
            ):
                if value < 0:
                    raise IllegalCommand(
                        f"Invariant failed: {prefix} {name} is negative: {value}"
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
                if not isinstance(target_ids, tuple) or not target_ids:
                    raise IllegalCommand(
                        f"Invariant failed: {zone} target binding must be a non-empty tuple"
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
                if not isinstance(operation.exclude_source, bool):
                    raise IllegalCommand(
                        f"Invariant failed: {operation_zone} source-exclusion policy is invalid"
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
            check_positive_int(
                frame.event_source_entity_id,
                "event_source_entity_id",
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
        for record in state.destroyed_followers:
            if record.owner not in (0, 1):
                raise IllegalCommand("Invariant failed: destroyed follower owner out of range")
            if record.death_sequence <= 0:
                raise IllegalCommand(
                    "Invariant failed: destroyed follower death_sequence must be positive"
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
                        not isinstance(keyword, str)
                        for keyword in record.effective_keywords
                    )
                ):
                    raise IllegalCommand(
                        "Invariant failed: death record keywords are invalid"
                    )

    def _emit(self, event: GameEvent) -> None:
        self.state.event_queue.append(event)

    def _dispatch_ability(
        self,
        event: AbilityEvent,
        source: Unit,
        target: Unit | None = None,
        *,
        player_index: int | None = None,
    ) -> None:
        if isinstance(source, Unit) and source.printed_abilities_removed:
            return
        structured_trigger = {
            AbilityEvent.FOLLOWER_EVOLVED: (
                Trigger.EVOLVE,
                AbilityKeyword.ON_EVOLVE,
            ),
            AbilityEvent.FOLLOWER_SUPER_EVOLVED: (
                Trigger.SUPER_EVOLVE,
                AbilityKeyword.ON_SUPER_EVOLVE,
            ),
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
                        player_index=(
                            self.current_player
                            if player_index is None
                            else player_index
                        ),
                        source=source,
                        target=target,
                    ),
                )
        self.ability_handlers.dispatch(
            AbilityContext(
                event=event,
                player_index=(
                    self.current_player if player_index is None else player_index
                ),
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
        self, player: PlayerState, definition: CardDefinition, *,
        origin: CardOrigin = CardOrigin.DECK,
        source_origin: CardOrigin | None = None,
        fused_material_ids: tuple[int, ...] = (),
    ) -> HandCard:
        hand_card = self._make_hand_card(
            definition,
            self.state.allocate_entity_id(),
            origin=origin,
            source_origin=source_origin,
            fused_material_ids=fused_material_ids,
        )
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
                    entity.expire_attack_restrictions(dur_str, player_index)
                    entity.expire_targeting_restrictions(dur_str, player_index)
            for hand_card in self._hand_cards(owner_index):
                for dur_str in expire_durations:
                    hand_card.expire_cost_modifiers(dur_str, player_index)

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
