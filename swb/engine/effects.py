from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swb.db.repository import CardDefinition


class EffectKind(str, Enum):
    DRAW = "draw"
    HEAL_LEADER = "heal_leader"
    DAMAGE_LEADER = "damage_leader"
    DAMAGE_UNIT = "damage_unit"
    RESTORE_MANA = "restore_mana"
    BUFF_UNIT = "buff_unit"
    SUMMON = "summon"
    DESTROY = "destroy"
    BANISH = "banish"
    ADD_CARD = "add_card"
    ADD_KEYWORD = "add_keyword"
    CHANGE_COST = "change_cost"
    GAIN_EMBLEM = "gain_emblem"
    RETURN_TO_HAND = "return_to_hand"
    RETURN_TO_DECK = "return_to_deck"
    DISCARD = "discard"


class TargetKind(str, Enum):
    SELF = "self"
    OWN_LEADER = "own_leader"
    ENEMY_LEADER = "enemy_leader"
    OWN_UNIT = "own_unit"
    ENEMY_UNIT = "enemy_unit"
    OWN_BOARD = "own_board"
    ENEMY_BOARD = "enemy_board"
    ANY_UNIT = "any_unit"
    OWN_AMULET = "own_amulet"
    ENEMY_AMULET = "enemy_amulet"
    ANY_AMULET = "any_amulet"
    ANY_BOARD = "any_board"
    RANDOM_OWN_UNIT = "random_own_unit"
    RANDOM_ENEMY_UNIT = "random_enemy_unit"
    RANDOM_OWN_BOARD = "random_own_board"
    RANDOM_ENEMY_BOARD = "random_enemy_board"
    ALL_OWN_UNITS = "all_own_units"
    ALL_ENEMY_UNITS = "all_enemy_units"
    ALL_UNITS = "all_units"
    ALL_OWN_BOARD = "all_own_board"
    ALL_ENEMY_BOARD = "all_enemy_board"
    ALL_OWN_AMULETS = "all_own_amulets"
    ALL_ENEMY_AMULETS = "all_enemy_amulets"
    OWN_HAND = "own_hand"


@dataclass(frozen=True)
class EffectOperation:
    kind: EffectKind
    target: TargetKind
    amount: int = 0
    secondary_amount: int = 0
    card_id: int | None = None
    keyword: str | None = None


@dataclass
class EffectFrame:
    controller: int
    source_card_id: int
    source_name: str
    source_entity_id: int | None
    source_card: CardDefinition
    operations: tuple[EffectOperation, ...]
    label: str = "效果"
    next_index: int = 0
    pending_target_id: int | None = None
    move_source_to_graveyard: bool = False
    _all_target_ids: list[int] = field(default_factory=list)
    _all_target_index: int = 0
