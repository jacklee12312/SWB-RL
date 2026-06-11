from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommandType(str, Enum):
    END_TURN = "end_turn"
    PLAY_CARD = "play_card"
    ATTACK = "attack"
    EVOLVE = "evolve"
    CHOOSE = "choose"


@dataclass(frozen=True)
class EndTurn:
    player_index: int
    type: CommandType = CommandType.END_TURN


@dataclass(frozen=True)
class PlayCard:
    player_index: int
    hand_index: int
    type: CommandType = CommandType.PLAY_CARD


@dataclass(frozen=True)
class Attack:
    player_index: int
    attacker_id: int
    target_id: int | None
    type: CommandType = CommandType.ATTACK


@dataclass(frozen=True)
class Evolve:
    player_index: int
    unit_id: int
    type: CommandType = CommandType.EVOLVE


@dataclass(frozen=True)
class Choose:
    player_index: int
    option_id: str
    type: CommandType = CommandType.CHOOSE


GameCommand = EndTurn | PlayCard | Attack | Evolve | Choose


@dataclass(frozen=True)
class ChoiceOption:
    option_id: str
    label: str
    entity_id: int | None = None


@dataclass(frozen=True)
class ChoiceRequest:
    player_index: int
    prompt: str
    options: tuple[ChoiceOption, ...]
    continuation_id: str

