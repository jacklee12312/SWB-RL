from __future__ import annotations

import random
from dataclasses import dataclass
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
    Attack,
    ChoiceKind,
    ChoiceOption,
    ChoiceRequest,
    Choose,
    EndTurn,
    Evolve,
    GameCommand,
    PlayCard,
)
from swb.engine.deck import CLASS_NAMES, validate_deck
from swb.engine.effects import (
    Condition,
    ConditionType,
    CostChangeMode,
    EffectFrame,
    EffectKind,
    EffectOperation,
    ExprType,
    ModifierDuration,
    TargetKind,
    ValueExpression,
)
from swb.engine.events import EventType, GameEvent
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
    GameState,
    GraveyardCard,
    HandCard,
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
    is_all_target,
    is_choice_target,
    is_graveyard_target,
    is_random_target,
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


class IllegalCommand(ValueError):
    pass


MAX_RESOLUTION_STEPS = 20_000

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


def _expire_duration_values(duration: ModifierDuration) -> tuple[str, ...]:
    return _DURATION_EXPANSION.get(duration, (duration.value,))


def _expires_for_player(duration: ModifierDuration, controller: int) -> int | None:
    if duration == ModifierDuration.PERMANENT:
        return None
    if duration in (ModifierDuration.UNTIL_END_OF_TURN, ModifierDuration.UNTIL_START_OF_NEXT_TURN):
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
    starting_health: int = 20


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
        self._spellboost_pending: int | None = None
        self._pending_spellboost_player: int = 0
        self._pending_spellboost_source_card_id: int = 0
        self._pending_spellboost_source_entity_id: int | None = None

    def _execute_trigger_rules(self, trigger, context) -> None:
        if isinstance(trigger, str):
            trigger = Trigger(trigger)
        source = context.source
        if not isinstance(source, Unit):
            return
        ops = self.rulebook.operations_for(source.definition.card_id, trigger)
        if not ops:
            return
        saved = self.state.active_player
        self.state.active_player = context.player_index
        try:
            self._start_effects(source.definition, source.entity_id, ops, label=trigger.value)
        finally:
            self.state.active_player = saved

    def _is_ability_covered(self, context, ability) -> bool:
        card = (
            context.source.definition
            if hasattr(context.source, "definition")
            else context.source
        )
        expected_kind = {
            AbilityKeyword.NECROMANCY: EffectKind.NECROMANCY,
            AbilityKeyword.REANIMATE: EffectKind.REANIMATE,
        }.get(ability)
        if card is None or expected_kind is None:
            return False

        def contains_kind(operations: tuple[EffectOperation, ...]) -> bool:
            return any(
                operation.kind is expected_kind
                or contains_kind(operation.necromancy_operations)
                for operation in operations
            )

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
        self._spellboost_pending = None
        self._pending_spellboost_player = 0
        self._pending_spellboost_source_card_id = 0
        self._pending_spellboost_source_entity_id = None
        self._emblem_batches: dict[int, dict[str, object]] = {}
        self._next_emblem_batch_id = 1
        self._emblem_expiration_batches: dict[int, dict[str, object]] = {}
        self._next_emblem_expiration_batch_id = 1
        self._stabilizing = False
        self.state.destroyed_followers.clear()
        self.state._next_death_sequence = 1
        for player_index in range(2):
            for _ in range(self.config.starting_hand):
                self._draw(player_index, reason="起手")
        self._start_turn(0)
        self._resolve_event_queue()
        return self.state

    def apply(self, command: GameCommand) -> CoreTransition:
        self._ensure_entity_ids()
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
        elif isinstance(command, Choose):
            self._choose(command)
        else:
            raise TypeError(f"Unknown command: {command!r}")

        self._resolve_event_queue()
        self._stabilize()
        self._resume_suspended_action()
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
            normal_playable = self._is_mode_playable(card, player, None)
            if normal_playable:
                commands.append(PlayCard(self.current_player, index, "normal"))
            for mode_def in modes:
                if self._is_mode_playable(card, player, mode_def):
                    commands.append(PlayCard(self.current_player, index, mode_def.mode_id))
        if (
            player.evolution_points > 0
            and player.turns_started >= self.config.evolution_unlock_turn
            and not player.evolved_this_turn
        ):
            commands.extend(
                Evolve(self.current_player, unit.entity_id)
                for unit in player.board
                if isinstance(unit, Unit)
                and not unit.evolved
            )
        guards = [
            unit
            for unit in opponent.board
            if isinstance(unit, Unit) and unit.has_guard and not unit.ambush_active
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
                    target for target in opponent.board if isinstance(target, Unit) and not target.ambush_active
                ]
                commands.extend(
                    Attack(self.current_player, unit.entity_id, target.entity_id)
                    for target in targets
                )
        return commands

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
            all_require_target = all(
                is_choice_target(op.target) or is_random_target(op.target) or is_all_target(op.target)
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

    def _summon_follower_to_board(
        self,
        player_index: int,
        definition: CardDefinition,
        *,
        summon_cause: str,
        entity_id: int | None = None,
        origin: CardOrigin = CardOrigin.DECK,
        source_origin: CardOrigin | None = None,
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
        player.board.append(unit)
        self._record_cooperation(
            player_index,
            1,
            source_card_id=definition.card_id,
            source_entity_id=unit.entity_id,
            summon_cause=summon_cause,
        )
        return unit

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

        self._dispatch_card_ability(AbilityEvent.CHECK_PLAY, card)
        player.hand.pop(command.hand_index)
        player.hand_entity_ids.pop(command.hand_index)
        player.mana -= play_cost
        player.cards_played_this_turn += 1

        if mode_def is not None and mode_def.is_accelerate:
            self._play_accelerate(card, play_cost, hand_entity_id, hand_origin, hand_source_origin, mode_def)
            return
        if mode_def is not None and mode_def.is_crystallize:
            self._play_crystallize(card, play_cost, hand_entity_id, hand_origin, hand_source_origin, mode_def)
            return
        if card.card_type == "法术" and mode_id == "normal":
            self._play_spell(card, play_cost, hand_entity_id, origin=hand_origin, source_origin=hand_source_origin)
            return
        if card.card_type == "护符" and mode_id == "normal":
            self._play_amulet(card, play_cost, origin=hand_origin)
            return

        unit = self._summon_follower_to_board(
            self.current_player,
            card,
            summon_cause="play",
            origin=hand_origin,
            source_origin=hand_source_origin,
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
            }
            return
        self._finish_follower_play(unit.entity_id, mode_operations)

    def _finish_follower_play(
        self,
        unit_id: int,
        mode_operations: tuple[EffectOperation, ...],
    ) -> None:
        try:
            unit = self._find_board_entity(unit_id)
        except IllegalCommand:
            return
        if not isinstance(unit, Unit):
            return
        fanfare_operations = self._fanfare_operations(unit)
        operations = fanfare_operations + mode_operations
        if operations:
            label = "入场曲"
            if fanfare_operations and mode_operations:
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

    def _play_amulet(self, card: CardDefinition, play_cost: int, *, origin: CardOrigin = CardOrigin.DECK) -> None:
        amulet = Amulet(
            definition=card,
            entity_id=self.state.allocate_entity_id(),
            countdown=self.rulebook.countdown_for(card.card_id),
            entered_turn=self.turn,
            origin=origin,
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
    ) -> None:
        countdown = mode_def.countdown if mode_def else None
        amulet = Amulet(
            definition=card,
            entity_id=source_entity_id,
            countdown=countdown,
            entered_turn=self.turn,
            origin=origin,
            source_origin=source_origin,
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
    ) -> EffectFrame:
        frame = EffectFrame(
            controller=self.current_player if controller is None else controller,
            source_card_id=card.card_id,
            source_name=card.name,
            source_entity_id=source_entity_id,
            source_card=card,
            operations=operations,
            label=label,
            move_source_to_graveyard=move_source_to_graveyard,
        )
        self.state.effect_stack.append(frame)
        return frame

    def _step(self) -> None:
        self.state.resolution_steps += 1
        if self.state.resolution_steps > MAX_RESOLUTION_STEPS:
            recent = [e.type.value for e in self.event_history[-20:]]
            frames = [(f.source_name, f.next_index, f.label) for f in self.state.effect_stack[-5:]]
            batches = [len(b.records) for b in self.state.death_queue[-3:]]
            raise ResolutionLoopError(
                f"Resolution step limit exceeded at turn {self.turn}, "
                f"player {self.current_player + 1}. "
                f"Recent events: {recent}. "
                f"Effect stack: {frames}. "
                f"Death queue sizes: {batches}."
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
        actual = min(amount, health_before)
        target_player.health -= actual

        self._emit(GameEvent(
            EventType.DAMAGE_APPLIED, controller,
            source_id=source.entity_id if hasattr(source, 'entity_id') else None,
            amount=actual,
            metadata={"target_player": self.players.index(target_player), "damage_type": damage_type.value},
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
                if frame.emblem_expiration_batch_id is not None:
                    self._complete_emblem_expiration(
                        frame.emblem_expiration_batch_id,
                        frame.expiring_emblem_owner,
                        frame.expiring_emblem_entity_id,
                    )
                continue

            operation = frame.operations[frame.next_index]
            is_meta_effect = operation.kind in (
                EffectKind.CONDITIONAL, EffectKind.CHOOSE_ONE, EffectKind.OPTIONAL,
            )
            if not is_meta_effect:
                condition_state = evaluate_conditions_without_target(
                    operation.conditions,
                    self._build_eval_context(frame, None),
                )
                if condition_state is PartialConditionResult.FALSE:
                    frame.next_index += 1
                    continue

            if operation.target is TargetKind.PREVIOUS_TARGET:
                if not operation.target_key or operation.target_key not in frame._target_bindings:
                    raise IllegalCommand(
                        f"PREVIOUS_TARGET requires a bound target_key"
                    )
                target_id = frame._target_bindings[operation.target_key]
                try:
                    self._find_board_entity(target_id)
                except IllegalCommand:
                    frame.next_index += 1
                    continue
                self._checked_execute(operation, frame, target_id)
                frame.next_index += 1
                self._resolve_event_queue()
                self._stabilize()
                continue

            if is_graveyard_target(operation.target) and is_choice_target(operation.target) and frame.pending_target_id is None:
                candidates = graveyard_candidates(operation, frame.controller, self.players)
                options = build_graveyard_choice_options(candidates)
                if not options:
                    frame.next_index += 1
                    continue
                if frame.auto_resolve_choices:
                    chosen = self.random.choice(options)
                    self._log(frame.controller, f"自动选择目标：{chosen.label}")
                    frame.pending_target_id = chosen.entity_id
                else:
                    self.state.pending_choice = ChoiceRequest(
                        player_index=frame.controller,
                        prompt=f"为 {frame.source_name} 从墓地选择目标",
                        options=tuple(options),
                        continuation_id=f"{frame.source_card_id}:{frame.next_index}",
                        choice_kind=ChoiceKind.GRAVEYARD,
                        request_id=self._allocate_choice_request_id(),
                    )
                    self.state.phase = Phase.AWAITING_CHOICE
                    self._log(
                        frame.controller,
                        f"{frame.source_name} 等待从墓地选择目标："
                        + "、".join(option.label for option in options),
                    )
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

            if is_choice_target(operation.target) and not is_graveyard_target(operation.target) and frame.pending_target_id is None:
                options = self._target_options(operation, frame.controller)
                if operation.conditions and operation.target is not TargetKind.OWN_HAND:
                    candidates = target_candidates(operation, frame.controller, self.players)
                    candidates = [e for e in candidates if not (isinstance(e, Unit) and e.ambush_active and self._entity_owner(e.entity_id) != frame.controller)]
                    candidates = [e for e in candidates if evaluate_target_conditions(operation.conditions, e, frame.controller, self.players, source_entity_id=frame.source_entity_id)]
                    options = build_choice_options(candidates)

                if not options:
                    frame.next_index += 1
                    continue
                if frame.auto_resolve_choices:
                    chosen = self.random.choice(options)
                    self._log(frame.controller, f"自动选择目标：{chosen.label}")
                    frame.pending_target_id = chosen.entity_id
                else:
                    choice_kind = ChoiceKind.GENERIC
                    if operation.target in (TargetKind.OWN_HAND,):
                        choice_kind = ChoiceKind.HAND
                    elif operation.target not in (TargetKind.OWN_GRAVEYARD_CARD,):
                        choice_kind = ChoiceKind.BOARD
                    self.state.pending_choice = ChoiceRequest(
                        player_index=frame.controller,
                        prompt=f"为 {frame.source_name} 选择目标",
                        options=tuple(options),
                        continuation_id=f"{frame.source_card_id}:{frame.next_index}",
                        choice_kind=choice_kind,
                        request_id=self._allocate_choice_request_id(),
                    )
                    self.state.phase = Phase.AWAITING_CHOICE
                    self._log(
                        frame.controller,
                        f"{frame.source_name} 等待选择目标："
                        + "、".join(option.label for option in options),
                    )
                    return

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
                candidates = target_candidates(operation, frame.controller, self.players)
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
                candidates = target_candidates(operation, frame.controller, self.players)
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
                        )
                    ]
                chosen = pick_random(candidates, self.random) if candidates else None
                if chosen is None:
                    frame.next_index += 1
                    continue
                target_id = chosen.entity_id
            else:
                target_id = frame.pending_target_id
                frame.pending_target_id = None

            self._checked_execute(operation, frame, target_id)
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
                frame._target_bindings[operation.target_key] = target_id
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

        all_require_target = all(
            is_choice_target(op.target) or is_random_target(op.target) or is_all_target(op.target)
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
    ) -> bool:
        if operation.target == TargetKind.OWN_HAND:
            return len(self.players[controller].hand) > 1
        if operation.target == TargetKind.RANDOM_OWN_HAND:
            return len(self.players[controller].hand) > 1
        if operation.target == TargetKind.ALL_OWN_HAND:
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
            ),
        )
        if condition_state is PartialConditionResult.FALSE:
            return True
        candidates = target_candidates(operation, controller, self.players)
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
                )
            ]
        return bool(candidates)

    @staticmethod
    def _requires_choice(operation: EffectOperation) -> bool:
        return is_choice_target(operation.target)

    def _target_options(
        self, operation: EffectOperation, controller: int
    ) -> list[ChoiceOption]:
        if operation.target == TargetKind.OWN_HAND:
            return hand_choice_options(self.players[controller])
        if is_graveyard_target(operation.target):
            gc = graveyard_candidates(operation, controller, self.players)
            return build_graveyard_choice_options(gc)
        candidates = target_candidates(operation, controller, self.players)
        candidates = [e for e in candidates if not (isinstance(e, Unit) and e.ambush_active and self._entity_owner(e.entity_id) != controller)]
        return build_choice_options(candidates)

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

        player = self.players[self.current_player]
        unit = self._find_unit(player.board, command.unit_id)
        if player.evolution_points <= 0:
            raise IllegalCommand("No evolution points")
        if player.turns_started < self.config.evolution_unlock_turn:
            raise IllegalCommand("Evolution is not unlocked")
        if player.evolved_this_turn or unit.evolved:
            raise IllegalCommand("Evolution is not available")

        could_attack_leader = unit.can_attack_leader
        unit.evolved = True
        unit.base_attack += 2
        unit.base_health += 2
        unit._recompute_attack()
        unit.health += 2
        unit._recompute_max()
        player.evolution_points -= 1
        player.evolved_this_turn = True
        if unit.attacks_remaining > 0:
            unit.can_attack = True
            unit.rush_only = not could_attack_leader
        self._log(
            self.current_player,
            f"进化 {unit.definition.name}，变为 {unit.attack}/{unit.health}，"
            f"剩余进化点 {player.evolution_points}",
        )
        self._emit(
            GameEvent(
                EventType.FOLLOWER_EVOLVED,
                self.current_player,
                source_id=unit.entity_id,
                metadata={"source": unit},
            )
        )
        self._resolve_event_queue()
        if self.state.pending_choice is not None:
            self._suspended_action = "evolve"
            self._suspended_action_state = {"unit_id": unit.entity_id}
            return

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
            if isinstance(unit, Unit) and unit.has_guard and not unit.ambush_active
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
            if guards and target not in guards:
                raise IllegalCommand("A guard follower must be attacked")

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
            ops = self.rulebook.operations_for(unit.definition.card_id, Trigger.TURN_END)
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
            ops = self.rulebook.operations_for(unit.definition.card_id, Trigger.TURN_END)
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
                    self._queue_effects(
                        frame.source_card, frame.source_entity_id,
                        optional_ops, controller=frame.controller,
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
            if option.entity_id is not None and option.option_id.startswith("entity:"):
                on_board = False
                for p in self.players:
                    if any(e.entity_id == option.entity_id for e in p.board):
                        on_board = True
                        break
                in_graveyard = any(
                    gc.entity_id == option.entity_id for gc in self.players[command.player_index].graveyard
                )
                if not on_board and not in_graveyard:
                    self._log(
                        command.player_index,
                        f"目标 {option.label} 已离场，跳过",
                    )
                    self.state.pending_choice = None
                    self.state.phase = Phase.MAIN
                    frame.pending_target_id = None
                    frame.next_index += 1
                    self._continue_effects()
                    self._try_spellboost_hand()
                    return
            if option.entity_id is not None and option.option_id.startswith("hand:"):
                found = False
                for p in self.players:
                    if option.entity_id in p.hand_entity_ids:
                        found = True
                        break
                if not found:
                    self._log(
                        command.player_index,
                        f"目标 {option.label} 已离手，跳过",
                    )
                    self.state.pending_choice = None
                    self.state.phase = Phase.MAIN
                    frame.pending_target_id = None
                    frame.next_index += 1
                    self._continue_effects()
                    self._try_spellboost_hand()
                    return
            frame.pending_target_id = option.entity_id
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
        player.turns_started += 1
        player.evolved_this_turn = False
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
            ops = self.rulebook.operations_for(unit.definition.card_id, Trigger.TURN_START)
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
                ops = self.rulebook.operations_for(unit.definition.card_id, Trigger.TURN_START)
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
        )

    def _resolve_amount(self, operation: EffectOperation, ctx: EvalContext) -> int:
        if operation.amount_expr is not None:
            return evaluate_expression(operation.amount_expr, ctx)
        return operation.amount

    def _resolve_secondary(self, operation: EffectOperation, ctx: EvalContext) -> int:
        if operation.secondary_expr is not None:
            return evaluate_expression(operation.secondary_expr, ctx)
        return operation.secondary_amount

    def _checked_execute(
        self, operation: EffectOperation, frame: EffectFrame, target_id: int | None,
    ) -> None:
        if operation.target is TargetKind.SELF:
            target_id = frame.source_entity_id
        ctx = self._build_eval_context(frame, target_id)
        is_meta = operation.kind in (EffectKind.CONDITIONAL, EffectKind.CHOOSE_ONE, EffectKind.OPTIONAL)
        if not is_meta:
            for cond in operation.conditions:
                if not evaluate_condition(cond, ctx):
                    return
        amount = self._resolve_amount(operation, ctx)
        secondary = self._resolve_secondary(operation, ctx)
        if operation.kind is EffectKind.HEAL_LEADER:
            amount = max(0, amount)
        if operation.amount_expr is not None or operation.secondary_expr is not None or amount != operation.amount or secondary != operation.secondary_amount:
            resolved = EffectOperation(
                kind=operation.kind,
                target=operation.target,
                amount=amount,
                secondary_amount=secondary,
                card_id=operation.card_id,
                emblem_id=operation.emblem_id,
                keyword=operation.keyword,
                restriction=operation.restriction,
                conditions=operation.conditions,
                mode=operation.mode,
                duration=operation.duration,
                set_attack=operation.set_attack,
                set_health=operation.set_health,
                target_key=operation.target_key,
                necromancy_operations=operation.necromancy_operations,
                graveyard_cost_max=operation.graveyard_cost_max,
                graveyard_cost_min=operation.graveyard_cost_min,
                graveyard_follower_only=operation.graveyard_follower_only,
                graveyard_card_type=operation.graveyard_card_type,
                emblem_remove_mode=operation.emblem_remove_mode,
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
        elif effect.kind is EffectKind.BUFF_UNIT:
            source = (
                self._find_board_entity(frame.source_entity_id)
                if frame.source_entity_id is not None
                else None
            )
            if target_id is not None:
                source = self._find_board_entity(target_id)
            if not isinstance(source, Unit):
                raise IllegalCommand("Buff target must be a follower")
            modifier = StatModifier(
                modifier_id=self._allocate_modifier_id(),
                attack_delta=effect.amount,
                health_delta=effect.secondary_amount,
                duration=effect.duration.value,
                expires_for_player=_expires_for_player(effect.duration, frame.controller),
            )
            source.add_stat_modifier(modifier)
            self._log(
                frame.controller,
                f"属性变化 {effect.amount}/{effect.secondary_amount}",
            )
        elif effect.kind is EffectKind.DESTROY:
            target = self._find_board_entity(target_id)
            if isinstance(target, Unit):
                self._death_causes[target.entity_id] = DeathCause.EFFECT_DESTROY
                target.health = 0
            elif isinstance(target, Amulet):
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
        elif effect.kind is EffectKind.DISCARD:
            self._execute_discard(target_id, frame)
        elif effect.kind is EffectKind.ADD_KEYWORD:
            self._execute_keyword_change(effect, frame, target_id, add=True)
        elif effect.kind is EffectKind.REMOVE_KEYWORD:
            self._execute_keyword_change(effect, frame, target_id, add=False)
        elif effect.kind is EffectKind.CHANGE_COST:
            self._execute_change_cost(effect, frame, target_id)
        elif effect.kind is EffectKind.TRANSFORM:
            self._execute_transform(effect, frame, target_id)
        elif effect.kind is EffectKind.SET_STATS:
            self._execute_set_stats(effect, frame, target_id)
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
        else:
            self._log(
                frame.controller,
                f"[未实现效果] {name} {frame.label}: {effect.kind.value}",
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
                expires_for_player=_expires_for_player(effect.duration, frame.controller),
            )
            verb = "获得"
        else:
            target.remove_keyword(
                effect.keyword,
                duration=effect.duration.value,
                expires_for_player=_expires_for_player(effect.duration, frame.controller),
            )
            verb = "失去"
        self._log(
            frame.controller,
            f"{target.definition.name} {verb}关键词 {effect.keyword}",
        )

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
                expires_for_player=_expires_for_player(effect.duration, frame.controller),
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
                expires_for_player=_expires_for_player(effect.duration, frame.controller),
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
                expires_for_player=_expires_for_player(effect.duration, frame.controller),
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
        self._queue_effects(
            frame.source_card,
            frame.source_entity_id,
            effect.necromancy_operations,
            controller=frame.controller,
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
            candidates = target_candidates(operation, controller, self.players)
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

    def _dispatch_emblem_triggers(
        self, player_index: int, event_type: str,
        event_player: int | None = None,
        source_id: int | None = None,
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
            self._emit(GameEvent(
                EventType.EMBLEM_TRIGGERED, player_index,
                source_id=ei.entity_id,
                metadata={
                    "emblem_id": ei.emblem_id,
                    "emblem_entity_id": ei.entity_id,
                    "owner": player_index,
                    "trigger": event_type,
                    "trigger_index": trigger_index,
                    "activation_count": ei.activation_counts.get(trigger_index, 0) + 1,
                    "source_card_id": ei.definition.source_card_id,
                    "source_entity_id": batch.get("source_id"),
                    "event_player": batch.get("event_player"),
                    "active_player": self.state.active_player,
                },
            ))
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
                self._queue_effects(
                    frame.source_card, frame.source_entity_id,
                    opt.operations, controller=frame.controller,
                    label=f"{frame.label}/choose_one/{opt.label}",
                )
                return

    def _execute_conditional(self, effect, frame) -> None:
        ctx = EvalContext(
            controller=frame.controller,
            players=self.players,
            source_entity_id=frame.source_entity_id,
        )
        result = evaluate_conditions_without_target(effect.conditions, ctx)
        branch_ops = effect.then_operations if result is PartialConditionResult.TRUE else effect.else_operations
        if branch_ops:
            self._queue_effects(
                frame.source_card, frame.source_entity_id,
                branch_ops, controller=frame.controller,
                label=f"{frame.label}/conditional",
            )

    def _execute_choose_one(self, effect, frame) -> None:
        legal_options = []
        for opt in effect.choose_one_options:
            if opt.conditions:
                ctx = EvalContext(
                    controller=frame.controller,
                    players=self.players,
                    source_entity_id=frame.source_entity_id,
                )
                result = evaluate_conditions_without_target(opt.conditions, ctx)
                if result is not PartialConditionResult.TRUE:
                    continue
            if opt.operations:
                all_need_target = all(
                    is_choice_target(op.target) or is_random_target(op.target) or is_all_target(op.target)
                    for op in opt.operations
                )
                if all_need_target and all(
                    not self._has_candidates_for(
                        op,
                        frame.controller,
                        source_entity_id=frame.source_entity_id,
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
            self._queue_effects(
                frame.source_card, frame.source_entity_id,
                chosen.operations, controller=frame.controller,
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
        all_need_target = all(
            is_choice_target(op.target) or is_random_target(op.target) or is_all_target(op.target)
            for op in ops
        )
        if all_need_target and all(
            not self._has_candidates_for(
                op,
                frame.controller,
                source_entity_id=frame.source_entity_id,
            )
            for op in ops
        ):
            return

        if frame.auto_resolve_choices:
            self._queue_effects(
                frame.source_card, frame.source_entity_id,
                ops, controller=frame.controller,
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
        else:
            self._log(
                frame.controller,
                f"{frame.source_name} 召唤失败：{card_def.card_type} 类型不可召唤",
            )

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

    def _execute_return_to_deck(
        self, target_id: int | None, frame: EffectFrame
    ) -> None:
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

    def _resolve_event_queue(self) -> None:
        if self._suspended_event_state is not None and self.state.pending_choice is None:
            self._resume_event_queue()
            return

        event_to_ability = {
            EventType.CARD_PLAYED: AbilityEvent.CARD_PLAYED,
            EventType.FOLLOWER_SUMMONED: AbilityEvent.FOLLOWER_SUMMONED,
            EventType.FOLLOWER_EVOLVED: AbilityEvent.FOLLOWER_EVOLVED,
            EventType.ATTACK_DECLARED: AbilityEvent.BEFORE_ATTACK,
            EventType.COMBAT_STARTED: AbilityEvent.BEFORE_COMBAT,
            EventType.DAMAGE_DEALT: AbilityEvent.AFTER_DAMAGE,
            EventType.FOLLOWER_DESTROYED: AbilityEvent.FOLLOWER_DESTROYED,
        }
        event_to_emblem_trigger = {
            EventType.CARD_PLAYED: "card_played",
            EventType.FOLLOWER_SUMMONED: "follower_summoned",
            EventType.FOLLOWER_EVOLVED: "follower_evolved",
            EventType.LEADER_HEALED: "leader_healed",
        }
        while self.state.event_queue:
            self._step()
            event = self.state.event_queue.popleft()
            self.event_history.append(event)
            ability_event = event_to_ability.get(event.type)
            source = event.metadata.get("source")
            if source is None:
                source = event.metadata.get("definition")
            target = event.metadata.get("target")
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
            emblem_trigger = event_to_emblem_trigger.get(event.type)
            if emblem_trigger is not None:
                self._dispatch_emblem_triggers(
                    event.player_index,
                    emblem_trigger,
                    event_player=event.player_index,
                    source_id=event.source_id,
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

        remaining = state["remaining_events"]
        for e in remaining:
            self.state.event_queue.append(e)
        self._resolve_event_queue()

    def _stabilize(self) -> None:
        if self._stabilizing:
            return
        if self.state.pending_choice is not None:
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
        self._emit(GameEvent(
            EventType.LAST_WORDS_COMPLETE,
            record.owner,
            source_id=record.entity_id,
            metadata={"card_id": record.card_id},
        ))

    def _continue_batch_lws(self) -> None:
        batch = self._suspended_batch
        lw_records = self._suspended_lw_records
        while lw_records:
            record = lw_records[0]
            self._suspended_lw_records = lw_records[1:]
            self._execute_last_words(record)
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
                metadata={"card_id": record.card_id},
            ))
            lw_records = self._suspended_lw_records
        self._emit(GameEvent(
            EventType.DEATH_BATCH_END,
            self.current_player,
            metadata={"batch_id": batch.batch_id},
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
                    metadata={"batch_id": batch.batch_id, "count": len(batch.records)},
                )
            )

            for record in sorted(batch.records, key=self._last_words_order_key):
                player = self.players[record.owner]
                if record.card_type == "护符":
                    self._log(record.owner, f"护符 {record.card_name} 被破坏")
                    self._emit(GameEvent(
                        EventType.AMULET_DESTROYED, record.owner,
                        source_id=record.entity_id,
                        metadata={"card_id": record.card_id, "cause": record.cause.value, "definition": record.definition},
                    ))
                else:
                    player.followers_destroyed_this_turn += 1
                    self._log(record.owner, f"随从 {record.card_name} 被破坏")
                    self._emit(GameEvent(
                        EventType.FOLLOWER_DESTROYED, record.owner,
                        source_id=record.entity_id,
                        metadata={"card_id": record.card_id, "cause": record.cause.value, "definition": record.definition},
                    ))
                self._emit(GameEvent(
                    EventType.ENTITY_LEFT_PLAY, record.owner,
                    source_id=record.entity_id,
                    metadata={"card_id": record.card_id, "cause": record.cause.value},
                ))

            self._resolve_event_queue()

            lw_records = [r for r in sorted(batch.records, key=self._last_words_order_key) if r.allows_last_words]
            if not lw_records:
                self._emit(GameEvent(
                    EventType.DEATH_BATCH_END, self.current_player,
                    metadata={"batch_id": batch.batch_id},
                ))
                self._resolve_event_queue()
                continue

            self._suspended_batch = batch
            self._suspended_lw_records = list(lw_records)
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
                        allows_last_words=True,
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

    def _execute_last_words(self, record: DeathRecord) -> None:
        self._step()
        self._emit(GameEvent(
            EventType.LAST_WORDS_START,
            record.owner,
            source_id=record.entity_id,
            metadata={"card_id": record.card_id},
        ))
        self._log(record.owner, f"{record.card_name} 谢幕曲开始")

        operations = self.rulebook.operations_for(record.card_id, Trigger.LAST_WORDS)
        if not operations and record.card_type == "护符":
            operations = self.rulebook.operations_for(record.card_id, Trigger.COUNTDOWN_EXPIRED)

        if operations:
            saved_active = self.state.active_player
            self.state.active_player = record.owner
            try:
                self._queue_effects(
                    record.definition,
                    record.entity_id,
                    operations,
                    label="谢幕曲",
                )
            finally:
                self.state.active_player = saved_active

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
    ) -> HandCard:
        hand_card = self._make_hand_card(
            definition,
            self.state.allocate_entity_id(),
            origin=origin,
            source_origin=source_origin,
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
