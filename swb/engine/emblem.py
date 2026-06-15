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


class TurnScope(str, Enum):
    OWNER_TURN = "owner_turn"
    OPPONENT_TURN = "opponent_turn"
    ANY_TURN = "any_turn"


class EventScope(str, Enum):
    OWNER_EVENT = "owner_event"
    OPPONENT_EVENT = "opponent_event"
    ANY_EVENT = "any_event"


@dataclass(frozen=True)
class EmblemTriggerRule:
    trigger: str
    operations: tuple["EffectOperation", ...] = ()
    conditions: tuple["Condition", ...] = ()
    turn_scope: TurnScope | None = None
    event_scope: EventScope | None = None
    once_per_turn: bool = False
    max_activations: int | None = None


@dataclass(frozen=True)
class EmblemDefinition:
    emblem_id: str
    source_card_id: int
    stacking: EmblemStacking = EmblemStacking.ALLOW
    countdown: int | None = None
    triggers: tuple[EmblemTriggerRule, ...] = ()
    on_expire: tuple["EffectOperation", ...] = ()

    @property
    def is_permanent(self) -> bool:
        return self.countdown is None
