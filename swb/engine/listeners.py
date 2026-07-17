from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from swb.engine.abilities import normalize_keyword_name
from swb.engine.emblem import EventScope, TurnScope
from swb.engine.events import EventType

if TYPE_CHECKING:
    from swb.db.repository import CardDefinition
    from swb.engine.effects import Condition, EffectOperation


class ListenerZone(str, Enum):
    BOARD = "board"
    HAND = "hand"
    LEADER_AREA = "leader_area"


class SourceRelation(str, Enum):
    SELF = "self"
    OTHER = "other"
    ANY = "any"


LISTENER_EVENT_TYPES = frozenset({
    EventType.AMULET_ACTIVATED,
    EventType.CARD_DRAWN,
    EventType.CARD_DISCARDED,
    EventType.CARD_FUSED,
    EventType.FOLLOWER_SUMMONED,
    EventType.FOLLOWER_EVOLVED,
    EventType.FOLLOWER_SUPER_EVOLVED,
    EventType.FOLLOWER_DESTROYED,
    EventType.FOLLOWER_STATS_INCREASED,
    EventType.FOLLOWER_DAMAGED_SURVIVED,
    EventType.AMULET_DESTROYED,
    EventType.ENTITY_LEFT_PLAY,
    EventType.EARTH_RITE_ACTIVATED,
    EventType.SPELLBOOSTED,
    EventType.CARD_PLAYED,
    EventType.TURN_STARTED,
    EventType.TURN_ENDED,
})


@dataclass(frozen=True)
class EventCardFilter:
    card_type: str | None = None
    class_id: int | None = None
    class_name: str | None = None
    tribe_id: int | None = None
    tribe_name: str | None = None
    cost_min: int | None = None
    cost_max: int | None = None
    card_id: int | None = None
    card_name: str | None = None
    keyword: str | None = None

    def matches(
        self,
        definition: CardDefinition | None,
        event_source: Any = None,
        event_keywords: tuple[str, ...] | frozenset[str] | None = None,
    ) -> bool:
        if definition is None:
            return False
        if not (
            (self.card_type is None or definition.card_type == self.card_type)
            and (self.class_id is None or definition.class_id == self.class_id)
            and (self.class_name is None or definition.class_name == self.class_name)
            and (self.tribe_id is None or definition.tribe_id == self.tribe_id)
            and (
                self.tribe_name is None
                or definition.tribe_name == self.tribe_name
            )
            and (self.cost_min is None or definition.cost >= self.cost_min)
            and (self.cost_max is None or definition.cost <= self.cost_max)
            and (self.card_id is None or definition.card_id == self.card_id)
            and (self.card_name is None or definition.name == self.card_name)
        ):
            return False
        if self.keyword is None:
            return True
        canonical = normalize_keyword_name(self.keyword)
        if event_keywords is not None:
            return canonical in {
                normalize_keyword_name(keyword)
                for keyword in event_keywords
            }
        has_keyword = getattr(event_source, "has_keyword", None)
        if callable(has_keyword):
            return bool(has_keyword(canonical))
        return canonical in {
            normalize_keyword_name(keyword)
            for keyword in definition.keywords
        }


@dataclass(frozen=True)
class CardListenerDefinition:
    card_id: int
    zone: ListenerZone
    event: EventType
    operations: tuple[EffectOperation, ...]
    conditions: tuple[Condition, ...] = ()
    event_filter: EventCardFilter | None = None
    event_scope: EventScope = EventScope.ANY_EVENT
    turn_scope: TurnScope = TurnScope.ANY_TURN
    source_relation: SourceRelation = SourceRelation.ANY
    once_per_turn: bool = False
    max_activations: int | None = None

    def __post_init__(self) -> None:
        if self.card_id <= 0:
            raise ValueError("listener card_id must be positive")
        if self.event not in LISTENER_EVENT_TYPES:
            raise ValueError(
                f"unsupported listener event {self.event.value!r}"
            )
        if not self.operations:
            raise ValueError("listener operations must not be empty")
        if self.max_activations is not None and self.max_activations <= 0:
            raise ValueError("listener max_activations must be positive")
