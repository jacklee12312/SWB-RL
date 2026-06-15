from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    CARD_DRAWN = "card_drawn"
    CARD_PLAYED = "card_played"
    FOLLOWER_SUMMONED = "follower_summoned"
    AMULET_ENTERED = "amulet_entered"
    SPELL_RESOLVED = "spell_resolved"
    FOLLOWER_EVOLVED = "follower_evolved"
    ATTACK_DECLARED = "attack_declared"
    COMBAT_STARTED = "combat_started"
    DAMAGE_DEALT = "damage_dealt"
    DAMAGE_APPLIED = "damage_applied"
    FOLLOWER_DESTROYED = "follower_destroyed"
    AMULET_DESTROYED = "amulet_destroyed"
    GAME_ENDED = "game_ended"
    CARD_BANISHED = "card_banished"
    CARD_ADDED_TO_HAND = "card_added_to_hand"
    CARD_RETURNED_TO_HAND = "card_returned_to_hand"
    CARD_RETURNED_TO_DECK = "card_returned_to_deck"
    CARD_DISCARDED = "card_discarded"
    DEATH_DETECTED = "death_detected"
    ENTITY_LEFT_PLAY = "entity_left_play"
    DEATH_BATCH_START = "death_batch_start"
    DEATH_BATCH_END = "death_batch_end"
    LAST_WORDS_START = "last_words_start"
    LAST_WORDS_COMPLETE = "last_words_complete"
    DAMAGE_PREVENTED = "damage_prevented"
    BARRIER_CONSUMED = "barrier_consumed"
    BANE_TRIGGERED = "bane_triggered"
    DRAIN_HEALED = "drain_healed"
    AMBUSH_LOST = "ambush_lost"
    SPELLBOOSTED = "spellboosted"
    GRAVEYARD_ENTERED = "graveyard_entered"
    SHADOWS_CHANGED = "shadows_changed"
    NECROMANCY_ACTIVATED = "necromancy_activated"
    REANIMATE_RESOLVED = "reanimate_resolved"
    GRAVEYARD_CARD_RETURNED = "graveyard_card_returned"
    GRAVEYARD_CARD_SUMMONED = "graveyard_card_summoned"
    GRAVEYARD_CARD_BANISHED = "graveyard_card_banished"


@dataclass(frozen=True)
class GameEvent:
    type: EventType
    player_index: int
    source_id: int | None = None
    target_id: int | None = None
    amount: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
