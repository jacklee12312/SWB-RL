from __future__ import annotations

from dataclasses import dataclass
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


class TargetKind(str, Enum):
    SELF = "self"
    OWN_LEADER = "own_leader"
    ENEMY_LEADER = "enemy_leader"
    OWN_UNIT = "own_unit"
    ENEMY_UNIT = "enemy_unit"
    OWN_BOARD = "own_board"
    ENEMY_BOARD = "enemy_board"


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
