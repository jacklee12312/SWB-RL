from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from swb.db.repository import CardDefinition
from swb.engine.commands import Attack, ChoiceKind, Choose, EndTurn, Evolve, GameCommand, PlayCard
from swb.engine.card_rules import RuleBook
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, BoardCard, HandCard, PlayerState, Unit


@dataclass(frozen=True)
class StepResult:
    observation: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


class ShadowverseEnv:
    """RL adapter around the deterministic command-based game engine."""

    MAX_HAND = 9
    MAX_BOARD = 5
    MAX_MANA = 10
    MAX_TURNS = 200
    STARTING_HAND = 4
    STARTING_EVOLUTION_POINTS = 2
    EVOLUTION_UNLOCK_TURN = 4
    CLASS_COUNT = 7

    END_TURN = 0
    PLAY_OFFSET = 1
    ATTACK_OFFSET = PLAY_OFFSET + MAX_HAND
    TARGETS_PER_ATTACKER = 1 + MAX_BOARD
    EVOLVE_OFFSET = ATTACK_OFFSET + MAX_BOARD * TARGETS_PER_ATTACKER
    CHOICE_OFFSET = EVOLVE_OFFSET + MAX_BOARD
    MAX_CHOICE_OPTIONS = 16

    GRAVEYARD_PAGE_SIZE = 16
    GRAVEYARD_CHOICE_OFFSET = CHOICE_OFFSET + MAX_CHOICE_OPTIONS
    GRAVEYARD_PREV_PAGE = GRAVEYARD_CHOICE_OFFSET
    GRAVEYARD_NEXT_PAGE = GRAVEYARD_CHOICE_OFFSET + 1
    GRAVEYARD_SLOT_OFFSET = GRAVEYARD_CHOICE_OFFSET + 2

    ACTION_SIZE = GRAVEYARD_SLOT_OFFSET + GRAVEYARD_PAGE_SIZE

    DEFAULT_RULE_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "rules"

    def __init__(
        self,
        deck_a: Sequence[CardDefinition],
        deck_b: Sequence[CardDefinition],
        *,
        class_a: int,
        class_b: int,
        seed: int | None = None,
        rulebook: RuleBook | None = None,
        card_resolver: Callable[[int], CardDefinition | None] | None = None,
    ):
        resolved_rulebook = rulebook or RuleBook.from_directory(
            self.DEFAULT_RULE_DIRECTORY
        )
        self.core = GameEngine(
            deck_a,
            deck_b,
            class_a=class_a,
            class_b=class_b,
            seed=seed,
            rulebook=resolved_rulebook,
            card_resolver=card_resolver,
            config=GameConfig(
                max_hand=self.MAX_HAND,
                max_board=self.MAX_BOARD,
                max_mana=self.MAX_MANA,
                max_turns=self.MAX_TURNS,
                starting_hand=self.STARTING_HAND,
                starting_evolution_points=self.STARTING_EVOLUTION_POINTS,
                evolution_unlock_turn=self.EVOLUTION_UNLOCK_TURN,
            ),
        )
        self._graveyard_page: int = 0
        self._last_choice_request_key: tuple[str, int] | None = None

    @property
    def deck_lists(self) -> tuple[list[CardDefinition], list[CardDefinition]]:
        return self.core.deck_lists

    @property
    def players(self) -> list[PlayerState]:
        return self.core.players

    @property
    def current_player(self) -> int:
        return self.core.current_player

    @property
    def decision_player(self) -> int:
        pending = self.core.state.pending_choice
        if pending is not None:
            return pending.player_index
        return self.core.current_player

    @property
    def turn(self) -> int:
        return self.core.turn

    @property
    def terminated(self) -> bool:
        return self.core.terminated

    @property
    def winner(self) -> int | None:
        return self.core.winner

    @property
    def logs(self) -> list[str]:
        return self.core.logs

    @property
    def placeholder_ability_events(self) -> list[object]:
        return self.core.placeholder_ability_events

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[list[float], dict[str, object]]:
        self.core.reset(seed=seed)
        self._graveyard_page = 0
        self._last_choice_request_key = None
        return self.observation(), self.info()

    def step(self, action: int) -> StepResult:
        self._sync_choice_page()
        mask = self.action_mask()
        if action < 0 or action >= self.ACTION_SIZE or not mask[action]:
            raise ValueError(f"Illegal action: {action}")
        acting_player = self.decision_player
        try:
            command = self._decode_action(action)
        except _PageTurn:
            return StepResult(
                observation=self.observation(),
                reward=0.0,
                terminated=False,
                truncated=False,
                info=self.info(),
            )
        result = self.core.apply(command)
        reward = 0.0 if result.winner is None else (
            1.0 if result.winner == acting_player else -1.0
        )
        self._sync_choice_page()
        return StepResult(
            observation=self.observation(),
            reward=reward,
            terminated=self.core.terminated,
            truncated=self.turn > self.MAX_TURNS,
            info=self.info(),
        )

    def observation(self) -> list[float]:
        self._sync_choice_page()
        perspective = self.decision_player
        me = self.players[perspective]
        opponent = self.players[1 - perspective]
        values = [
            me.health / 20, opponent.health / 20,
            me.mana / self.MAX_MANA, me.max_mana / self.MAX_MANA,
            len(me.deck) / max(1, len(self.deck_lists[perspective])),
            len(opponent.deck) / max(1, len(self.deck_lists[1 - perspective])),
            len(me.hand) / self.MAX_HAND, len(opponent.hand) / self.MAX_HAND,
            self.turn / self.MAX_TURNS,
            me.evolution_points / self.STARTING_EVOLUTION_POINTS,
            opponent.evolution_points / self.STARTING_EVOLUTION_POINTS,
            me.turns_started / self.MAX_TURNS,
            opponent.turns_started / self.MAX_TURNS,
            float(me.evolved_this_turn),
            float(opponent.evolved_this_turn),
            float(self.core.state.pending_choice is not None),
            me.shadows / 20, opponent.shadows / 20,
        ]
        total_pages = self._graveyard_total_pages()
        values.extend([
            (
                self._graveyard_page / max(1, total_pages - 1)
                if total_pages > 0
                else 0.0
            ),
            float(total_pages),
        ])
        values.extend(float(me.class_id == cid) for cid in range(1, self.CLASS_COUNT + 1))
        values.extend(float(opponent.class_id == cid) for cid in range(1, self.CLASS_COUNT + 1))
        for index in range(self.MAX_HAND):
            card = me.hand[index] if index < len(me.hand) else None
            values.extend(self._card_features(card))
        for board in (me.board, opponent.board):
            for index in range(self.MAX_BOARD):
                unit = board[index] if index < len(board) else None
                values.extend(self._board_features(unit))
        return values

    def info(self) -> dict[str, object]:
        self._sync_choice_page()
        request = self.core.state.pending_choice
        total_pages = self._graveyard_total_pages()
        return {
            "current_player": self.current_player,
            "decision_player": self.decision_player,
            "turn": self.turn,
            "winner": self.winner,
            "player_classes": self.core.player_classes,
            "action_mask": self.action_mask(),
            "log": tuple(self.logs),
            "events": tuple(self.core.event_history),
            "placeholder_ability_events": tuple(self.placeholder_ability_events),
            "graveyard_page": self._graveyard_page,
            "graveyard_total_pages": total_pages,
        }

    def _decode_action(self, action: int) -> GameCommand:
        if action == self.END_TURN:
            return EndTurn(self.current_player)
        if action == self.GRAVEYARD_PREV_PAGE:
            self._graveyard_page = max(0, self._graveyard_page - 1)
            raise _PageTurn()
        if action == self.GRAVEYARD_NEXT_PAGE:
            self._graveyard_page += 1
            raise _PageTurn()
        if action >= self.GRAVEYARD_SLOT_OFFSET:
            request = self.core.state.pending_choice
            if request is None or request.choice_kind is not ChoiceKind.GRAVEYARD:
                raise ValueError("No graveyard choice is pending")
            slot_idx = action - self.GRAVEYARD_SLOT_OFFSET
            global_idx = self._graveyard_page * self.GRAVEYARD_PAGE_SIZE + slot_idx
            if global_idx >= len(request.options):
                raise ValueError(f"Graveyard choice index {global_idx} out of range")
            return Choose(request.player_index, request.options[global_idx].option_id)
        if action >= self.CHOICE_OFFSET and action < self.GRAVEYARD_CHOICE_OFFSET:
            request = self.core.state.pending_choice
            if request is None:
                raise ValueError("No choice is pending")
            option_index = action - self.CHOICE_OFFSET
            if option_index >= len(request.options):
                raise ValueError(f"Choice index {option_index} out of range")
            return Choose(request.player_index, request.options[option_index].option_id)
        if action < self.ATTACK_OFFSET:
            return PlayCard(self.current_player, action - self.PLAY_OFFSET)
        if action >= self.EVOLVE_OFFSET:
            board_index = action - self.EVOLVE_OFFSET
            return Evolve(self.current_player, self.players[self.current_player].board[board_index].entity_id)
        relative = action - self.ATTACK_OFFSET
        attacker_index = relative // self.TARGETS_PER_ATTACKER
        target_slot = relative % self.TARGETS_PER_ATTACKER
        attacker = self.players[self.current_player].board[attacker_index]
        target_id = None
        if target_slot:
            target_id = self.players[1 - self.current_player].board[target_slot - 1].entity_id
        return Attack(self.current_player, attacker.entity_id, target_id)

    def _encode_command(self, command: GameCommand) -> int | None:
        if isinstance(command, EndTurn):
            return self.END_TURN
        if isinstance(command, Choose):
            request = self.core.state.pending_choice
            if request is None:
                return None
            if request.choice_kind is ChoiceKind.GRAVEYARD:
                for index, option in enumerate(request.options):
                    if option.option_id == command.option_id:
                        page_start = self._graveyard_page * self.GRAVEYARD_PAGE_SIZE
                        page_end = page_start + self.GRAVEYARD_PAGE_SIZE
                        if page_start <= index < page_end:
                            slot = index - page_start
                            return self.GRAVEYARD_SLOT_OFFSET + slot
                        return None
                return None
            for index, option in enumerate(request.options[: self.MAX_CHOICE_OPTIONS]):
                if option.option_id == command.option_id:
                    return self.CHOICE_OFFSET + index
            return None
        if isinstance(command, PlayCard):
            return self.PLAY_OFFSET + command.hand_index
        if isinstance(command, Evolve):
            index = self._unit_index(self.players[self.current_player].board, command.unit_id)
            return self.EVOLVE_OFFSET + index
        if isinstance(command, Attack):
            attacker_index = self._unit_index(self.players[self.current_player].board, command.attacker_id)
            base = self.ATTACK_OFFSET + attacker_index * self.TARGETS_PER_ATTACKER
            if command.target_id is None:
                return base
            target_index = self._unit_index(self.players[1 - self.current_player].board, command.target_id)
            return base + 1 + target_index
        return None

    def _build_grave_page_mask(self) -> list[bool]:
        mask = [False] * self.ACTION_SIZE
        request = self.core.state.pending_choice
        if request is None or request.choice_kind is not ChoiceKind.GRAVEYARD:
            return mask
        total = len(request.options)
        page_start = self._graveyard_page * self.GRAVEYARD_PAGE_SIZE
        page_end = min(page_start + self.GRAVEYARD_PAGE_SIZE, total)
        for i in range(page_start, page_end):
            mask[self.GRAVEYARD_SLOT_OFFSET + (i - page_start)] = True
        if self._graveyard_page > 0:
            mask[self.GRAVEYARD_PREV_PAGE] = True
        if page_end < total:
            mask[self.GRAVEYARD_NEXT_PAGE] = True
        return mask

    def _legal_non_choice_mask(self) -> list[bool]:
        mask = [False] * self.ACTION_SIZE
        for command in self.core.legal_commands():
            if not isinstance(command, Choose):
                action = self._encode_command(command)
                if action is not None:
                    mask[action] = True
        return mask

    def _legal_choice_mask(self) -> list[bool]:
        mask = [False] * self.ACTION_SIZE
        request = self.core.state.pending_choice
        if request is None:
            return mask
        if request.choice_kind is ChoiceKind.GRAVEYARD:
            return self._build_grave_page_mask()
        for command in self.core.legal_commands():
            if isinstance(command, Choose) and request.choice_kind is not ChoiceKind.GRAVEYARD:
                action = self._encode_command(command)
                if action is not None:
                    mask[action] = True
        return mask

    def action_mask(self) -> list[bool]:
        self._sync_choice_page()
        non_choice = self._legal_non_choice_mask()
        choice = self._legal_choice_mask()
        return [a or b for a, b in zip(non_choice, choice)]

    def _graveyard_total_pages(self) -> int:
        request = self.core.state.pending_choice
        if request is None or request.choice_kind is not ChoiceKind.GRAVEYARD:
            return 0
        return max(
            1,
            (len(request.options) + self.GRAVEYARD_PAGE_SIZE - 1)
            // self.GRAVEYARD_PAGE_SIZE,
        )

    def _sync_choice_page(self) -> None:
        request = self.core.state.pending_choice
        if request is None:
            request_key = None
        else:
            identity = request.request_id if request.request_id > 0 else id(request)
            request_key = (request.continuation_id, identity)
        if request_key != self._last_choice_request_key:
            self._graveyard_page = 0
            self._last_choice_request_key = request_key

    @staticmethod
    def _unit_index(board: list[BoardCard], entity_id: int) -> int:
        for index, unit in enumerate(board):
            if unit.entity_id == entity_id:
                return index
        raise ValueError(f"Unit {entity_id} is not on the board")

    @staticmethod
    def _card_features(card: CardDefinition | HandCard | None) -> list[float]:
        if card is None:
            return [0.0] * 7
        return [
            1.0, card.cost / ShadowverseEnv.MAX_MANA,
            (card.attack or 0) / 20, (card.life or 0) / 20,
            float(card.card_type == "随从"),
            float(card.card_type == "护符"),
            float(card.card_type == "法术"),
        ]

    @staticmethod
    def _board_features(entity: BoardCard | None) -> list[float]:
        if entity is None:
            return [0.0] * 11
        if isinstance(entity, Amulet):
            return [1.0, 0.0, (entity.countdown or 0) / 10, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        unit = entity
        return [
            1.0, unit.attack / 20, unit.health / 20,
            float(unit.can_attack), float(unit.has_guard),
            float(unit.has_keyword("疾驰")),
            float(unit.evolved), float(unit.rush_only), 0.0,
            float(unit.barrier_charges > 0), float(unit.ambush_active),
        ]


class _PageTurn(Exception):
    """Signal that a page-turn action was executed; step() catches this."""
    pass
