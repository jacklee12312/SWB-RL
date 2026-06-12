from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from swb.db.repository import CardDefinition

if TYPE_CHECKING:
    from swb.engine.commands import ChoiceRequest
    from swb.engine.effects import EffectFrame
    from swb.engine.events import GameEvent


class Phase(str, Enum):
    MAIN = "main"
    AWAITING_CHOICE = "awaiting_choice"
    FINISHED = "finished"


class DeathCause(str, Enum):
    COMBAT = "combat"
    ZERO_HEALTH = "zero_health"
    EFFECT_DESTROY = "effect_destroy"
    COUNTDOWN_EXPIRED = "countdown_expired"
    BANISH = "banish"
    RETURN_TO_HAND = "return_to_hand"
    RETURN_TO_DECK = "return_to_deck"
    TRANSFORM = "transform"


@dataclass(frozen=True)
class DeathRecord:
    owner: int
    entity_id: int
    card_id: int
    card_name: str
    card_type: str
    definition: CardDefinition
    cause: DeathCause
    source_player: int | None = None
    source_entity_id: int | None = None
    board_position: int = 0
    allows_last_words: bool = False


@dataclass
class DeathBatch:
    records: list[DeathRecord] = field(default_factory=list)
    batch_id: int = 0


class ResolutionLoopError(Exception):
    pass


@dataclass(kw_only=True)
class BoardEntity:
    definition: CardDefinition
    entity_id: int = 0


@dataclass
class Unit(BoardEntity):
    attack: int
    health: int
    can_attack: bool = False
    attacks_remaining: int = 1
    evolved: bool = False
    rush_only: bool = False
    barrier_charges: int = 0
    ambush_active: bool = False

    @classmethod
    def summon(cls, card: CardDefinition, *, entity_id: int = 0) -> "Unit":
        if card.attack is None or card.life is None:
            raise ValueError(f"{card.name} is not a playable follower")
        barrier = 1 if "屏障" in card.keywords else 0
        ambush = "潜行" in card.keywords
        return cls(
            definition=card,
            attack=card.attack,
            health=card.life,
            entity_id=entity_id,
            can_attack="疾驰" in card.keywords or "突进" in card.keywords,
            barrier_charges=barrier,
            ambush_active=ambush,
        )

    @property
    def has_guard(self) -> bool:
        return "守护" in self.definition.keywords

    @property
    def can_attack_leader(self) -> bool:
        return (
            self.can_attack
            and "突进" not in self.definition.keywords
            and not self.rush_only
        )


@dataclass
class Amulet(BoardEntity):
    countdown: int | None = None
    entered_turn: int = 0
    pending_destroy: bool = False


BoardCard = Unit | Amulet


@dataclass
class PlayerState:
    deck: list[CardDefinition]
    class_id: int
    class_name: str
    hand: list[CardDefinition] = field(default_factory=list)
    hand_entity_ids: list[int] = field(default_factory=list)
    board: list[BoardCard] = field(default_factory=list)
    graveyard: list[CardDefinition] = field(default_factory=list)
    banished: list[CardDefinition] = field(default_factory=list)
    emblems: list[str] = field(default_factory=list)
    health: int = 20
    max_mana: int = 0
    mana: int = 0
    fatigue: int = 0
    evolution_points: int = 2
    turns_started: int = 0
    evolved_this_turn: bool = False
    cards_played_this_turn: int = 0
    followers_destroyed_this_turn: int = 0
    cooperation: int = 0
    shadows: int = 0
    faith: int = 0


@dataclass
class GameState:
    players: list[PlayerState]
    active_player: int = 0
    turn: int = 1
    phase: Phase = Phase.MAIN
    winner: int | None = None
    event_queue: deque[GameEvent] = field(default_factory=deque)
    pending_choice: ChoiceRequest | None = None
    effect_stack: list[EffectFrame] = field(default_factory=list)
    death_queue: list[DeathBatch] = field(default_factory=list)
    resolution_steps: int = 0
    next_entity_id: int = 1

    @property
    def terminated(self) -> bool:
        return self.phase is Phase.FINISHED

    def allocate_entity_id(self) -> int:
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        return entity_id
