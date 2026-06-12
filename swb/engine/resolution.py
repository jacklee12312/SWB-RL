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
    PlaceholderAbilityEvent,
)
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import (
    Attack,
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
    EffectFrame,
    EffectKind,
    EffectOperation,
    ExprType,
    TargetKind,
    ValueExpression,
)
from swb.engine.events import EventType, GameEvent
from swb.engine.state import (
    Amulet,
    BoardCard,
    DeathBatch,
    DeathCause,
    DeathRecord,
    GameState,
    Phase,
    PlayerState,
    ResolutionLoopError,
    Unit,
)
from swb.engine.targeting import (
    build_choice_options,
    hand_choice_options,
    is_all_target,
    is_choice_target,
    is_random_target,
    pick_random,
    target_candidates,
)
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
        if command.player_index != self.current_player:
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
        return CoreTransition(
            command=command,
            events=tuple(self.event_history[event_start:]),
            acting_player=acting_player,
            winner=self.winner,
            terminated=self.terminated,
        )

    def legal_commands(self) -> list[GameCommand]:
        self._ensure_entity_ids()
        if self.terminated:
            return []
        if self.state.pending_choice is not None:
            request = self.state.pending_choice
            return [
                Choose(self.current_player, option.option_id)
                for option in request.options
            ]

        player = self.players[self.current_player]
        opponent = self.players[1 - self.current_player]
        commands: list[GameCommand] = [EndTurn(self.current_player)]
        if len(player.board) < self.config.max_board:
            for index, card in enumerate(player.hand[: self.config.max_hand]):
                if self._is_card_playable(card, player):
                    commands.append(PlayCard(self.current_player, index))
        else:
            for index, card in enumerate(player.hand[: self.config.max_hand]):
                if card.card_type == "法术" and self._is_card_playable(card, player):
                    commands.append(PlayCard(self.current_player, index))
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
            targets = guards or [
                target for target in opponent.board if isinstance(target, Unit) and not target.ambush_active
            ]
            commands.extend(
                Attack(self.current_player, unit.entity_id, target.entity_id)
                for target in targets
            )
        return commands

    def _play_card(self, command: PlayCard) -> None:
        player = self.players[self.current_player]
        if not 0 <= command.hand_index < len(player.hand):
            raise IllegalCommand("Hand index is out of range")
        card = player.hand[command.hand_index]
        if card.cost > player.mana:
            raise IllegalCommand("Not enough mana")
        if card.card_type in {"随从", "护符"} and len(player.board) >= self.config.max_board:
            raise IllegalCommand("Board is full")
        if not self._is_card_playable(card, player):
            raise IllegalCommand("Card has no executable rule or legal target")

        self._dispatch_card_ability(AbilityEvent.CHECK_PLAY, card)
        player.hand.pop(command.hand_index)
        player.hand_entity_ids.pop(command.hand_index)
        player.mana -= card.cost
        player.cards_played_this_turn += 1
        if card.card_type == "法术":
            self._play_spell(card)
            return
        if card.card_type == "护符":
            self._play_amulet(card)
            return

        unit = Unit.summon(card, entity_id=self.state.allocate_entity_id())
        player.board.append(unit)
        player.cooperation += 1
        self._log(
            self.current_player,
            f"打出 {card.name} ({card.cost}费 {unit.attack}/{unit.health})",
        )
        self._emit(
            GameEvent(
                EventType.CARD_PLAYED,
                self.current_player,
                source_id=unit.entity_id,
                metadata={"source": unit},
            )
        )
        self._emit(
            GameEvent(
                EventType.FOLLOWER_SUMMONED,
                self.current_player,
                source_id=unit.entity_id,
                metadata={"source": unit},
            )
        )
        self._resolve_event_queue()
        self._execute_fanfare(unit)

    def _play_spell(self, card: CardDefinition) -> None:
        self._log(self.current_player, f"使用法术 {card.name}（{card.cost}费）")
        self._dispatch_card_ability(AbilityEvent.CARD_PLAYED, card)
        self._emit(
            GameEvent(
                EventType.CARD_PLAYED,
                self.current_player,
                metadata={"card_id": card.card_id, "card": card},
            )
        )
        operations = self.rulebook.operations_for(card.card_id, Trigger.PLAY)
        self._start_effects(
            card,
            None,
            operations,
            move_source_to_graveyard=True,
            label="法术",
        )

    def _play_amulet(self, card: CardDefinition) -> None:
        amulet = Amulet(
            definition=card,
            entity_id=self.state.allocate_entity_id(),
            countdown=self.rulebook.countdown_for(card.card_id),
            entered_turn=self.turn,
        )
        self.players[self.current_player].board.append(amulet)
        countdown = (
            f"，倒数 {amulet.countdown}" if amulet.countdown is not None else ""
        )
        self._log(
            self.current_player,
            f"打出护符 {card.name}（{card.cost}费{countdown}）",
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

    def _start_effects(
        self,
        card: CardDefinition,
        source_entity_id: int | None,
        operations: tuple[EffectOperation, ...],
        *,
        move_source_to_graveyard: bool = False,
        label: str = "效果",
    ) -> None:
        self._queue_effects(card, source_entity_id, operations, move_source_to_graveyard=move_source_to_graveyard, label=label)
        self._continue_effects()

    def _queue_effects(
        self,
        card: CardDefinition,
        source_entity_id: int | None,
        operations: tuple[EffectOperation, ...],
        *,
        move_source_to_graveyard: bool = False,
        label: str = "效果",
    ) -> EffectFrame:
        frame = EffectFrame(
            controller=self.current_player,
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
            if "必杀" in attacker.definition.keywords or "毁灭" in attacker.definition.keywords:
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
            if "吸血" in attacker.definition.keywords or "虹吸" in attacker.definition.keywords:
                owner_idx = self._entity_owner(attacker.entity_id)
                heal_amount = min(actual, health_before)
                owner = self.players[owner_idx]
                owner.health = min(owner.health + heal_amount, self.config.starting_health)
                self._log(controller, f"{attacker.definition.name} 的吸血回复了 {heal_amount} 点生命")
                self._emit(GameEvent(
                    EventType.DRAIN_HEALED, owner_idx,
                    source_id=attacker.entity_id,
                    amount=heal_amount,
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

        if isinstance(source, Unit) and ("吸血" in source.definition.keywords or "虹吸" in source.definition.keywords):
            owner_idx = self._entity_owner(source.entity_id)
            owner = self.players[owner_idx]
            owner.health = min(owner.health + actual, self.config.starting_health)
            self._log(controller, f"{source.definition.name} 的吸血回复了 {actual} 点生命")
            self._emit(GameEvent(
                EventType.DRAIN_HEALED, owner_idx,
                source_id=source.entity_id,
                amount=actual,
                metadata={"card_id": source.definition.card_id},
            ))

        return DamageResult(
            requested_amount=amount,
            actual_amount=actual,
            target_health_before=health_before,
            target_health_after=target_player.health,
            lethal=target_player.health <= 0,
        )

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
                    self.players[frame.controller].graveyard.append(frame.source_card)
                    self._emit(
                        GameEvent(
                            EventType.SPELL_RESOLVED,
                            frame.controller,
                            metadata={"card_id": frame.source_card_id},
                        )
                    )
                continue

            operation = frame.operations[frame.next_index]
            condition_state = evaluate_conditions_without_target(
                operation.conditions,
                self._build_eval_context(frame, None),
            )
            if condition_state is PartialConditionResult.FALSE:
                frame.next_index += 1
                continue

            if is_choice_target(operation.target) and frame.pending_target_id is None:
                options = self._target_options(operation, frame.controller)
                if operation.conditions:
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
                    self.state.pending_choice = ChoiceRequest(
                        player_index=frame.controller,
                        prompt=f"为 {frame.source_name} 选择目标",
                        options=tuple(options),
                        continuation_id=f"{frame.source_card_id}:{frame.next_index}",
                    )
                    self.state.phase = Phase.AWAITING_CHOICE
                    self._log(
                        frame.controller,
                        f"{frame.source_name} 等待选择目标："
                        + "、".join(option.label for option in options),
                    )
                    return

            if is_all_target(operation.target) and not frame.defer_stabilize:
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
            frame.next_index += 1
            self._resolve_event_queue()
            self._stabilize()

    def _is_card_playable(
        self, card: CardDefinition, player: PlayerState
    ) -> bool:
        if card.cost > player.mana:
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
        if operation.target == TargetKind.OWN_HAND:
            return len(self.players[self.current_player].hand) > 1
        condition_state = evaluate_conditions_without_target(
            operation.conditions,
            EvalContext(
                controller=self.current_player,
                players=self.players,
            ),
        )
        if condition_state is PartialConditionResult.FALSE:
            # The operation will be skipped and therefore requires no target.
            return True
        candidates = target_candidates(operation, self.current_player, self.players)
        if is_choice_target(operation.target):
            candidates = [e for e in candidates if not (isinstance(e, Unit) and e.ambush_active and self._entity_owner(e.entity_id) != self.current_player)]
        if condition_state is PartialConditionResult.DEPENDS_ON_TARGET:
            candidates = [
                entity
                for entity in candidates
                if evaluate_target_conditions(
                    operation.conditions,
                    entity,
                    self.current_player,
                    self.players,
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
        unit.attack += 2
        unit.health += 2
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

    def _attack(self, command: Attack) -> None:
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

    def _end_turn(self) -> None:
        player_index = self.current_player
        for unit in tuple(self.players[player_index].board):
            self._dispatch_ability(
                AbilityEvent.TURN_ENDED, unit, player_index=player_index
            )
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
            if option.entity_id is not None and option.option_id.startswith("entity:"):
                try:
                    self._find_board_entity(option.entity_id)
                except IllegalCommand:
                    self._log(
                        command.player_index,
                        f"目标 {option.label} 已离场，跳过",
                    )
                    self.state.pending_choice = None
                    self.state.phase = Phase.MAIN
                    frame.pending_target_id = None
                    frame.next_index += 1
                    self._continue_effects()
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
                    return
            frame.pending_target_id = option.entity_id
        self.state.pending_choice = None
        self.state.phase = Phase.MAIN
        if self.state.effect_stack:
            self._continue_effects()

    def _start_turn(self, player_index: int) -> None:
        player = self.players[player_index]
        player.turns_started += 1
        player.evolved_this_turn = False
        player.cards_played_this_turn = 0
        player.followers_destroyed_this_turn = 0
        player.max_mana = min(self.config.max_mana, player.max_mana + 1)
        player.mana = player.max_mana
        self._tick_countdowns(player_index)
        for unit in player.board:
            if not isinstance(unit, Unit):
                continue
            unit.can_attack = True
            unit.attacks_remaining = 1
            unit.rush_only = False
            self._dispatch_ability(
                AbilityEvent.TURN_STARTED, unit, player_index=player_index
            )
        for card in (*player.hand, *player.deck):
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

    def _draw(self, player_index: int, *, reason: str) -> None:
        player = self.players[player_index]
        if player.deck:
            card = player.deck.pop()
            if len(player.hand) < self.config.max_hand:
                player.hand.append(card)
                player.hand_entity_ids.append(self.state.allocate_entity_id())
                self._emit(
                    GameEvent(
                        EventType.CARD_DRAWN,
                        player_index,
                        metadata={"card_id": card.card_id},
                    )
                )
                self._log(player_index, f"{reason}：{card.name}")
            else:
                player.graveyard.append(card)
                self._log(player_index, f"{reason}：{card.name}，手牌已满而被弃置")
            return
        player.fatigue += 1
        player.health -= player.fatigue
        self._log(
            player_index,
            f"牌库耗尽，受到 {player.fatigue} 点疲劳伤害（生命 {player.health}）",
        )

    def _execute_fanfare(self, unit: Unit) -> None:
        explicit = self.rulebook.operations_for(
            unit.definition.card_id, Trigger.FANFARE
        )
        if explicit:
            self._start_effects(
                unit.definition, unit.entity_id, explicit, label="入场曲"
            )
            return
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
                self._start_effects(
                    unit.definition,
                    unit.entity_id,
                    (operation,),
                    label="入场曲",
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
        ctx = self._build_eval_context(frame, target_id)
        for cond in operation.conditions:
            if not evaluate_condition(cond, ctx):
                return
        amount = self._resolve_amount(operation, ctx)
        secondary = self._resolve_secondary(operation, ctx)
        if operation.kind in (EffectKind.HEAL_LEADER, EffectKind.BUFF_UNIT):
            amount = max(0, amount)
            secondary = max(0, secondary)
        if operation.amount_expr is not None or operation.secondary_expr is not None or amount != operation.amount or secondary != operation.secondary_amount:
            resolved = EffectOperation(
                kind=operation.kind,
                target=operation.target,
                amount=amount,
                secondary_amount=secondary,
                card_id=operation.card_id,
                keyword=operation.keyword,
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
            for _ in range(effect.amount):
                self._draw(
                    frame.controller,
                    reason=f"{name} {frame.label}抽牌",
                )
        elif effect.kind is EffectKind.HEAL_LEADER:
            before = player.health
            player.health = min(self.config.starting_health, player.health + effect.amount)
            self._log(
                frame.controller,
                f"{name} {frame.label}回复 {player.health - before} 点生命"
                f"（生命 {player.health}）",
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
            source.attack += effect.amount
            source.health += effect.secondary_amount
            self._log(
                frame.controller,
                f"{source.definition.name} 获得 +{effect.amount}/+{effect.secondary_amount}"
                f"（{source.attack}/{source.health}）",
            )
        elif effect.kind is EffectKind.DESTROY:
            target = self._find_board_entity(target_id)
            if isinstance(target, Unit):
                target.health = 0
                self._death_causes[target.entity_id] = DeathCause.EFFECT_DESTROY
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
        else:
            self._log(
                frame.controller,
                f"[未实现效果] {name} {frame.label}: {effect.kind.value}",
            )

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
            unit = Unit.summon(card_def, entity_id=self.state.allocate_entity_id())
            player.board.append(unit)
            player.cooperation += 1
            self._log(
                frame.controller,
                f"{frame.source_name} 召唤 {card_def.name} ({unit.attack}/{unit.health})",
            )
            self._emit(
                GameEvent(
                    EventType.FOLLOWER_SUMMONED,
                    frame.controller,
                    source_id=unit.entity_id,
                    metadata={"source": unit},
                )
            )
        elif card_def.card_type == "护符":
            amulet = Amulet(
                definition=card_def,
                entity_id=self.state.allocate_entity_id(),
                countdown=self.rulebook.countdown_for(card_def.card_id),
                entered_turn=self.turn,
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
            player.graveyard.append(card_def)
            return
        player.hand.append(card_def)
        player.hand_entity_ids.append(self.state.allocate_entity_id())
        self._log(
            frame.controller,
            f"{frame.source_name} 将 {card_def.name} 加入手牌",
        )
        self._emit(
            GameEvent(
                EventType.CARD_ADDED_TO_HAND,
                frame.controller,
                metadata={"card_id": card_def.card_id, "card": card_def},
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
        if len(owner.hand) < self.config.max_hand:
            owner.hand.append(card_def)
            owner.hand_entity_ids.append(self.state.allocate_entity_id())
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
                card_def = player.hand[idx]
                player.hand.pop(idx)
                player.hand_entity_ids.pop(idx)
                player.graveyard.append(card_def)
                player.shadows += 1
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
        event_to_ability = {
            EventType.CARD_PLAYED: AbilityEvent.CARD_PLAYED,
            EventType.FOLLOWER_SUMMONED: AbilityEvent.FOLLOWER_SUMMONED,
            EventType.FOLLOWER_EVOLVED: AbilityEvent.FOLLOWER_EVOLVED,
            EventType.ATTACK_DECLARED: AbilityEvent.BEFORE_ATTACK,
            EventType.COMBAT_STARTED: AbilityEvent.BEFORE_COMBAT,
            EventType.DAMAGE_DEALT: AbilityEvent.AFTER_DAMAGE,
            EventType.FOLLOWER_DESTROYED: AbilityEvent.FOLLOWER_DESTROYED,
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
                if event.type is EventType.COMBAT_STARTED and isinstance(target, Unit):
                    self._dispatch_ability(
                        ability_event,
                        target,
                        source,
                        player_index=1 - event.player_index,
                    )

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
                    player.board.remove(entity)
                    player.graveyard.append(entity.definition)
                    player.shadows += 1
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
                    player.board.remove(entity)
                    player.graveyard.append(entity.definition)
                    player.shadows += 1
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

    def _ensure_entity_ids(self) -> None:
        seen: set[int] = set()
        for player in self.players:
            for entity in player.board:
                if entity.entity_id <= 0 or entity.entity_id in seen:
                    entity.entity_id = self.state.allocate_entity_id()
                seen.add(entity.entity_id)

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
