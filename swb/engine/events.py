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
    FOLLOWER_SUPER_EVOLVED = "follower_super_evolved"
    SUPER_EVOLUTION_ATTACK_BONUS = "super_evolution_attack_bonus"
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
    LEADER_HEALED = "leader_healed"
    AMBUSH_LOST = "ambush_lost"
    SPELLBOOSTED = "spellboosted"
    GRAVEYARD_ENTERED = "graveyard_entered"
    SHADOWS_CHANGED = "shadows_changed"
    NECROMANCY_ACTIVATED = "necromancy_activated"
    REANIMATE_RESOLVED = "reanimate_resolved"
    GRAVEYARD_CARD_RETURNED = "graveyard_card_returned"
    GRAVEYARD_CARD_SUMMONED = "graveyard_card_summoned"
    GRAVEYARD_CARD_BANISHED = "graveyard_card_banished"
    COOPERATION_CHANGED = "cooperation_changed"
    COMBO_CHANGED = "combo_changed"
    EARTH_SIGILS_CHANGED = "earth_sigils_changed"
    EARTH_SIGILS_MERGED = "earth_sigils_merged"
    EARTH_RITE_ACTIVATED = "earth_rite_activated"
    EARTH_SIGIL_DESTROY_PREVENTED = "earth_sigil_destroy_prevented"
    CARD_FUSED = "card_fused"
    HAND_CARD_TRANSFORMED = "hand_card_transformed"
    CARD_INVOKED = "card_invoked"
    AMULET_ACTIVATED = "amulet_activated"
    FAITH_PLACED = "faith_placed"
    FAITH_VALUE_CHANGED = "faith_value_changed"
    FAITH_CONSUMED = "faith_consumed"
    FAITH_CONSUME_FAILED = "faith_consume_failed"
    FAITH_ABILITY_GRANTED = "faith_ability_granted"
    FAITH_ABILITY_TRIGGERED = "faith_ability_triggered"
    MAX_MANA_CHANGED = "max_mana_changed"
    UNION_BURST_ACTIVATED = "union_burst_activated"
    FOLLOWER_HEALED = "follower_healed"
    FOLLOWER_ABILITIES_REMOVED = "follower_abilities_removed"
    LEADER_DAMAGE_MODIFIER_ADDED = "leader_damage_modifier_added"
    EMBLEM_GAINED = "emblem_gained"
    EMBLEM_REMOVED = "emblem_removed"
    EMBLEM_TRIGGERED = "emblem_triggered"
    CARD_LISTENER_TRIGGERED = "card_listener_triggered"
    EMBLEM_COUNTDOWN_CHANGED = "emblem_countdown_changed"
    EMBLEM_EXPIRED = "emblem_expired"


@dataclass(frozen=True)
class GameEvent:
    type: EventType
    player_index: int
    source_id: int | None = None
    target_id: int | None = None
    amount: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
