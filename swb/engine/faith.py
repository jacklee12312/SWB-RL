from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FaithTrigger(str, Enum):
    FOLLOWER_EVOLVED = "follower_evolved"


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


@dataclass
class FaithInstance:
    definition: FaithDefinition
    entity_id: int
    controller: int
    created_sequence: int
    value: int = 0

    @property
    def faith_id(self) -> str:
        return self.definition.faith_id

    @property
    def source_card_id(self) -> int:
        return self.definition.source_card_id
