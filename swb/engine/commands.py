from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommandType(str, Enum):
    END_TURN = "end_turn"
    PLAY_CARD = "play_card"
    ATTACK = "attack"
    EVOLVE = "evolve"
    SUPER_EVOLVE = "super_evolve"
    CHOOSE = "choose"
    BEGIN_FUSION = "begin_fusion"
    ACTIVATE_AMULET = "activate_amulet"
    USE_EXTRA_PP = "use_extra_pp"


class ChoiceKind(str, Enum):
    BOARD = "board"
    HAND = "hand"
    GRAVEYARD = "graveyard"
    MODE = "mode"
    CONFIRM = "confirm"
    GENERIC = "generic"
    FUSION = "fusion"


@dataclass(frozen=True)
class EndTurn:
    player_index: int
    type: CommandType = CommandType.END_TURN


@dataclass(frozen=True)
class PlayCard:
    player_index: int
    hand_index: int
    mode_id: str = "normal"
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
class SuperEvolve:
    player_index: int
    unit_id: int
    type: CommandType = CommandType.SUPER_EVOLVE


@dataclass(frozen=True)
class Choose:
    player_index: int
    option_id: str
    type: CommandType = CommandType.CHOOSE


@dataclass(frozen=True)
class BeginFusion:
    player_index: int
    fusion_entity_id: int
    type: CommandType = CommandType.BEGIN_FUSION


@dataclass(frozen=True)
class ActivateAmulet:
    player_index: int
    amulet_id: int
    type: CommandType = CommandType.ACTIVATE_AMULET


@dataclass(frozen=True)
class UseExtraPP:
    player_index: int
    type: CommandType = CommandType.USE_EXTRA_PP


GameCommand = (
    EndTurn
    | PlayCard
    | Attack
    | Evolve
    | SuperEvolve
    | Choose
    | BeginFusion
    | ActivateAmulet
    | UseExtraPP
)


@dataclass(frozen=True)
class ChoiceOption:
    option_id: str
    label: str
    entity_id: int | None = None
    leader_player_index: int | None = None


@dataclass(frozen=True)
class ChoiceRequest:
    player_index: int
    prompt: str
    options: tuple[ChoiceOption, ...]
    continuation_id: str
    choice_kind: ChoiceKind = ChoiceKind.GENERIC
    request_id: int = 0
    target_count: int = 1
    allow_duplicate_targets: bool = False
    selected_options: tuple[ChoiceOption, ...] = ()
