from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

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
    EffectFrame,
    EffectKind,
    EffectOperation,
    TargetKind,
)
from swb.engine.events import EventType, GameEvent
from swb.engine.state import Amulet, BoardCard, GameState, Phase, PlayerState, Unit


class IllegalCommand(ValueError):
    pass


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
        self.random = random.Random(seed)
        self.state = GameState(players=[])
        self.logs: list[str] = []
        self.event_history: list[GameEvent] = []
        self.placeholder_ability_events: list[PlaceholderAbilityEvent] = []
        self.ability_handlers = AbilityHandlers(self)

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
            if isinstance(unit, Unit) and unit.has_guard
        ]
        for unit in player.board:
            if not isinstance(unit, Unit):
                continue
            if not unit.can_attack or unit.attacks_remaining <= 0 or unit.attack <= 0:
                continue
            if not guards and unit.can_attack_leader:
                commands.append(Attack(self.current_player, unit.entity_id, None))
            targets = guards or [
                target for target in opponent.board if isinstance(target, Unit)
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
        self._continue_effects()

    def _continue_effects(self) -> None:
        while self.state.effect_stack and self.state.pending_choice is None:
            frame = self.state.effect_stack[-1]
            if frame.next_index >= len(frame.operations):
                self.state.effect_stack.pop()
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
            if self._requires_choice(operation) and frame.pending_target_id is None:
                options = self._target_options(operation, frame.controller)
                if not options:
                    raise IllegalCommand(
                        f"{frame.source_name} has no legal target for "
                        f"{operation.target.value}"
                    )
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

            target_id = frame.pending_target_id
            frame.pending_target_id = None
            self._execute_effect(operation, frame, target_id)
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
        return all(
            not self._requires_choice(operation)
            or bool(self._target_options(operation, self.current_player))
            for operation in operations
        )

    @staticmethod
    def _requires_choice(operation: EffectOperation) -> bool:
        return operation.target in {
            TargetKind.OWN_UNIT,
            TargetKind.ENEMY_UNIT,
            TargetKind.OWN_BOARD,
            TargetKind.ENEMY_BOARD,
        }

    def _target_options(
        self, operation: EffectOperation, controller: int
    ) -> list[ChoiceOption]:
        owner = (
            controller
            if operation.target in {TargetKind.OWN_UNIT, TargetKind.OWN_BOARD}
            else 1 - controller
        )
        entities = self.players[owner].board
        if operation.target in {TargetKind.OWN_UNIT, TargetKind.ENEMY_UNIT}:
            entities = [entity for entity in entities if isinstance(entity, Unit)]
        return [
            ChoiceOption(
                option_id=f"entity:{entity.entity_id}",
                label=entity.definition.name,
                entity_id=entity.entity_id,
            )
            for entity in entities
        ]

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
        player.board.remove(amulet)
        player.graveyard.append(amulet.definition)
        player.shadows += 1
        self._log(owner, f"护符 {amulet.definition.name} 被破坏")
        self._emit(
            GameEvent(
                EventType.AMULET_DESTROYED,
                owner,
                source_id=amulet.entity_id,
                metadata={"source": amulet},
            )
        )
        operations = self.rulebook.operations_for(
            amulet.definition.card_id, Trigger.COUNTDOWN_EXPIRED
        )
        if not operations:
            operations = self.rulebook.operations_for(
                amulet.definition.card_id, Trigger.LAST_WORDS
            )
        if operations:
            active = self.state.active_player
            self.state.active_player = owner
            try:
                self._start_effects(
                    amulet.definition,
                    amulet.entity_id,
                    operations,
                    label="谢幕曲",
                )
            finally:
                self.state.active_player = active

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
            if isinstance(unit, Unit) and unit.has_guard
        ]
        if command.target_id is None:
            if guards or not attacker.can_attack_leader:
                raise IllegalCommand("Leader is not a legal target")
            target = None
        else:
            target = self._find_unit(opponent.board, command.target_id)
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
            opponent.health -= attacker.attack
            self._emit(
                GameEvent(
                    EventType.DAMAGE_DEALT,
                    self.current_player,
                    source_id=attacker.entity_id,
                    amount=attacker.attack,
                    metadata={"source": attacker, "target_player": 1 - self.current_player},
                )
            )
            self._log(
                self.current_player,
                f"{attacker.definition.name} 攻击对方主战者，造成 {attacker.attack} 点伤害"
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
        attacker.health -= counter_damage
        target.health -= attack_damage
        self._emit(
            GameEvent(
                EventType.DAMAGE_DEALT,
                self.current_player,
                source_id=attacker.entity_id,
                target_id=target.entity_id,
                amount=attack_damage,
                metadata={"source": attacker, "target": target},
            )
        )
        self._emit(
            GameEvent(
                EventType.DAMAGE_DEALT,
                1 - self.current_player,
                source_id=target.entity_id,
                target_id=attacker.entity_id,
                amount=counter_damage,
                metadata={"source": target, "target": attacker},
            )
        )
        self._log(
            self.current_player,
            f"{attacker.definition.name} 攻击 {target.definition.name}，"
            f"造成 {attack_damage} 点并受到 {counter_damage} 点伤害",
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
            self.state.effect_stack[-1].pending_target_id = option.entity_id
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
            target = opponent if effect.target is TargetKind.ENEMY_LEADER else player
            target.health -= effect.amount
            target_name = "对方" if target is opponent else "己方"
            self._log(
                frame.controller,
                f"{name} {frame.label}对{target_name}主战者造成 {effect.amount} 点伤害"
                f"（生命 {target.health}）",
            )
        elif effect.kind is EffectKind.DAMAGE_UNIT:
            target = self._find_board_entity(target_id)
            if not isinstance(target, Unit):
                raise IllegalCommand("Damage target must be a follower")
            target.health -= effect.amount
            self._log(
                frame.controller,
                f"{name} 对 {target.definition.name} 造成 {effect.amount} 点伤害"
                f"（剩余生命 {target.health}）",
            )
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
            elif isinstance(target, Amulet):
                self._destroy_amulet(target)

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
            event = self.state.event_queue.popleft()
            self.event_history.append(event)
            ability_event = event_to_ability.get(event.type)
            source = event.metadata.get("source")
            target = event.metadata.get("target")
            if ability_event is not None and isinstance(source, Unit):
                self._dispatch_ability(
                    ability_event,
                    source,
                    target if isinstance(target, Unit) else None,
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
        while True:
            dead_units: list[tuple[int, Unit]] = []
            for player_index, player in enumerate(self.players):
                dead_units.extend(
                    (player_index, unit)
                    for unit in player.board
                    if isinstance(unit, Unit) and unit.health <= 0
                )
            if not dead_units:
                break
            for player_index, unit in dead_units:
                player = self.players[player_index]
                if unit not in player.board:
                    continue
                player.board.remove(unit)
                player.graveyard.append(unit.definition)
                player.followers_destroyed_this_turn += 1
                player.shadows += 1
                self._emit(
                    GameEvent(
                        EventType.FOLLOWER_DESTROYED,
                        player_index,
                        source_id=unit.entity_id,
                        metadata={"source": unit},
                    )
                )
            self._resolve_event_queue()
        self._check_game_over()

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
            for unit in player.board:
                if not isinstance(unit, Unit):
                    continue
                if unit.entity_id <= 0 or unit.entity_id in seen:
                    unit.entity_id = self.state.allocate_entity_id()
                seen.add(unit.entity_id)

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
