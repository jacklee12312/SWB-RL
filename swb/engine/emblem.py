from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swb.engine.effects import Condition, EffectOperation


class EmblemStacking(str, Enum):
    ALLOW = "allow"
    REPLACE = "replace"
    IGNORE = "ignore"


@dataclass(frozen=True)
class EmblemDefinition:
    emblem_id: str
    source_card_id: int
    stacking: EmblemStacking = EmblemStacking.ALLOW
    countdown: int | None = None
    triggers: tuple[EmblemTriggerRule, ...] = ()

    @property
    def is_permanent(self) -> bool:
        return self.countdown is None


@dataclass(frozen=True)
class EmblemTriggerRule:
    trigger: str
    operations: tuple["EffectOperation", ...] = ()
    conditions: tuple["Condition", ...] = ()
