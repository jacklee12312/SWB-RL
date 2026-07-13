from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swb.engine.effects import EffectOperation


class FaithTrigger(str, Enum):
    FOLLOWER_EVOLVED = "follower_evolved"
    AMULET_DESTROYED = "amulet_destroyed"


class FaithAbilityStacking(str, Enum):
    UNIQUE = "unique"
    ALLOW = "allow"


@dataclass(frozen=True)
class FaithTriggerRule:
    trigger: FaithTrigger
    amount: int = 1


@dataclass(frozen=True)
class FaithDefinition:
    faith_id: str
    source_card_id: int
    initial_value: int = 0
    triggers: tuple[FaithTriggerRule, ...] = ()


@dataclass(frozen=True)
class FaithGrantedAbility:
    ability_id: str
    trigger: FaithTrigger
    operations: tuple["EffectOperation", ...]
    granted_sequence: int


@dataclass
class FaithInstance:
    definition: FaithDefinition
    entity_id: int
    controller: int
    created_sequence: int
    value: int = 0
    granted_abilities: list[FaithGrantedAbility] = field(default_factory=list)
    _next_granted_ability_sequence: int = 1

    @property
    def faith_id(self) -> str:
        return self.definition.faith_id

    @property
    def source_card_id(self) -> int:
        return self.definition.source_card_id
