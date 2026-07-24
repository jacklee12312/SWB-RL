from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from swb.db.repository import CardDefinition
from swb.engine.abilities import RUNTIME_UNIT_KEYWORDS, normalize_keyword_name
from swb.engine.emblem import EmblemDefinition
from swb.engine.faith import FaithInstance
from swb.engine.effects import EffectOperation, EmptyDeckOutcome, TurnEndDestroyTiming
from swb.engine.origin import CardOrigin, is_derived, is_token

if TYPE_CHECKING:
    from swb.engine.commands import ChoiceRequest
    from swb.engine.effects import EffectFrame
    from swb.engine.events import GameEvent


class Phase(str, Enum):
    MAIN = "main"
    AWAITING_CHOICE = "awaiting_choice"
    FINISHED = "finished"


@dataclass(frozen=True)
class GraveyardCard:
    definition: CardDefinition
    entity_id: int
    owner: int
    entered_sequence: int
    entry_cause: str
    derived: bool = False
    origin: CardOrigin = CardOrigin.UNKNOWN
    token: bool = False
    source_origin: CardOrigin | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "derived",
            self.derived
            or is_derived(self.origin)
            or (
                self.source_origin is not None
                and is_derived(self.source_origin)
            ),
        )
        object.__setattr__(
            self,
            "token",
            self.token
            or is_token(self.definition, self.origin)
            or self.source_origin is CardOrigin.TOKEN,
        )


@dataclass(frozen=True)
class FusionMaterial:
    definition: CardDefinition
    entity_id: int
    owner: int
    consumed_sequence: int
    fused_into_entity_id: int
    origin: CardOrigin = CardOrigin.UNKNOWN
    source_origin: CardOrigin | None = None
    inherited_material_ids: tuple[int, ...] = ()


class DeathCause(str, Enum):
    COMBAT = "combat"
    ZERO_HEALTH = "zero_health"
    EFFECT_DESTROY = "effect_destroy"
    COUNTDOWN_EXPIRED = "countdown_expired"
    EARTH_SIGIL_DEPLETED = "earth_sigil_depleted"
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
    effective_keywords: frozenset[str] = frozenset()
    attack: int | None = None
    health: int | None = None
    evolved: bool = False
    super_evolved: bool = False
    granted_last_words: tuple[EffectOperation, ...] = ()


@dataclass(frozen=True)
class DestroyedFollowerRecord:
    definition: CardDefinition
    owner: int
    death_sequence: int
    cause: DeathCause
    derived: bool = False
    token: bool = False
    origin: CardOrigin = CardOrigin.DECK
    source_origin: CardOrigin | None = None
    destroyed_turn: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "derived",
            self.derived
            or is_derived(self.origin)
            or (
                self.source_origin is not None
                and is_derived(self.source_origin)
            ),
        )
        object.__setattr__(
            self,
            "token",
            self.token
            or is_token(self.definition, self.origin)
            or self.source_origin is CardOrigin.TOKEN,
        )


@dataclass(frozen=True)
class FollowerEntryRecord:
    """Public match history for a follower that successfully entered play."""

    definition: CardDefinition
    owner: int
    entry_sequence: int
    entered_turn: int
    entry_cause: str


@dataclass
class DeathBatch:
    records: list[DeathRecord] = field(default_factory=list)
    batch_id: int = 0


class ResolutionLoopError(Exception):
    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(kw_only=True)
class BoardEntity:
    definition: CardDefinition
    entity_id: int = 0
    origin: CardOrigin = CardOrigin.DECK
    source_origin: CardOrigin | None = None
    fused_material_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class KeywordModifier:
    keyword: str
    duration: str
    expires_for_player: int | None = None


@dataclass(frozen=True)
class KeywordRemovalModifier:
    keyword: str
    duration: str
    expires_for_player: int
    restore_barrier_charge: bool = False
    restore_ambush: bool = False


@dataclass(frozen=True)
class StatModifier:
    modifier_id: int
    attack_delta: int
    health_delta: int
    duration: str
    expires_for_player: int | None = None


@dataclass(frozen=True)
class AttackCapacityModifier:
    attacks_per_turn: int
    duration: str
    expires_for_player: int | None = None


@dataclass(frozen=True)
class LeaderDamageModifier:
    modifier_id: int
    amount: int
    duration: str
    expires_for_player: int | None = None
    source_controller: int | None = None
    source_entity_id: int | None = None
    source_card_id: int | None = None


class AttackRestriction(str, Enum):
    CANNOT_ATTACK = "cannot_attack"
    CANNOT_ATTACK_LEADER = "cannot_attack_leader"
    CANNOT_ATTACK_UNITS = "cannot_attack_units"


@dataclass(frozen=True)
class AttackRestrictionModifier:
    restriction: AttackRestriction
    duration: str
    expires_for_player: int | None = None


class TargetingRestriction(str, Enum):
    CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS = "cannot_be_targeted_by_enemy_effects"
    FORCES_ENEMY_ABILITY_TARGET = "forces_enemy_ability_target"


@dataclass(frozen=True)
class TargetingRestrictionModifier:
    restriction: TargetingRestriction
    duration: str
    expires_for_player: int | None = None


@dataclass(frozen=True)
class CostModifier:
    modifier_id: int
    mode: str
    amount: int
    duration: str
    expires_for_player: int | None = None


@dataclass
class DeckCard:
    """A physical card in a deck that can retain runtime cost changes.

    Ordinary deck entries remain ``CardDefinition`` objects until an effect
    modifies that physical copy.  This wrapper keeps the existing deck API
    compatible while distinguishing copies that later carry different costs.
    """

    definition: CardDefinition
    cost_modifiers: list[CostModifier] = field(default_factory=list)

    @property
    def current_cost(self) -> int:
        cost = self.definition.cost
        for modifier in self.cost_modifiers:
            if modifier.mode == "set":
                cost = modifier.amount
            elif modifier.mode == "add":
                cost += modifier.amount
            elif modifier.mode == "subtract":
                cost -= modifier.amount
            elif modifier.mode == "halve_round_up":
                cost = (cost + 1) // 2
        return max(0, cost)

    @property
    def cost(self) -> int:
        return self.current_cost

    def __getattr__(self, name: str):
        definition = self.__dict__.get("definition")
        if definition is None:
            raise AttributeError(name)
        return getattr(definition, name)


@dataclass
class HandCard:
    definition: CardDefinition
    entity_id: int
    cost_modifiers: list[CostModifier] = field(default_factory=list)
    stat_modifiers: list[StatModifier] = field(default_factory=list)
    spellboost_count: int = 0
    spellboost_cost_reduction: int = 0
    cannot_be_played: bool = False
    origin: CardOrigin = CardOrigin.DECK
    source_origin: CardOrigin | None = None
    fused_material_ids: list[int] = field(default_factory=list)
    fusion_used_turn: int | None = None
    evolutions_while_in_hand: int = 0
    union_burst_gauge_bonus: int = 0
    printed_keyword_overrides: set[str] = field(default_factory=set)
    permanent_keywords: set[str] = field(default_factory=set)
    temporary_keywords: list[KeywordModifier] = field(default_factory=list)
    removed_keywords: set[str] = field(default_factory=set)
    temporary_keyword_removals: list[KeywordRemovalModifier] = field(
        default_factory=list
    )
    granted_last_words: list[tuple[EffectOperation, ...]] = field(
        default_factory=list
    )
    effect_destroy_immunity: bool = False

    @property
    def current_cost(self) -> int:
        cost = self.definition.cost
        for modifier in self.cost_modifiers:
            if modifier.mode == "set":
                cost = modifier.amount
            elif modifier.mode == "add":
                cost += modifier.amount
            elif modifier.mode == "subtract":
                cost -= modifier.amount
            elif modifier.mode == "halve_round_up":
                cost = (cost + 1) // 2
        cost -= self.spellboost_count * self.spellboost_cost_reduction
        return max(0, cost)

    def apply_spellboost(self, amount: int) -> None:
        if amount < 0:
            raise ValueError(f"spellboost amount must be non-negative, got {amount}")
        self.spellboost_count += amount

    def union_burst_gauge(self, turns_started: int) -> int:
        return (
            turns_started
            + self.evolutions_while_in_hand
            + self.union_burst_gauge_bonus
        )

    @property
    def cost(self) -> int:
        return self.current_cost

    @property
    def card_id(self) -> int:
        return self.definition.card_id

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def card_type(self) -> str:
        return self.definition.card_type

    @property
    def attack(self) -> int | None:
        if self.definition.attack is None:
            return None
        return max(
            0,
            self.definition.attack
            + sum(modifier.attack_delta for modifier in self.stat_modifiers),
        )

    @property
    def life(self) -> int | None:
        if self.definition.life is None:
            return None
        return max(
            1,
            self.definition.life
            + sum(modifier.health_delta for modifier in self.stat_modifiers),
        )

    @property
    def keywords(self) -> frozenset[str]:
        return self.effective_keywords

    @property
    def original_keywords(self) -> frozenset[str]:
        return frozenset(
            normalize_keyword_name(keyword)
            for keyword in self.definition.keywords
        )

    @property
    def effective_keywords(self) -> frozenset[str]:
        temporary = {modifier.keyword for modifier in self.temporary_keywords}
        temporarily_removed = {
            modifier.keyword for modifier in self.temporary_keyword_removals
        }
        return frozenset(
            (
                (set(self.original_keywords) - self.printed_keyword_overrides)
                | self.permanent_keywords
                | temporary
            )
            - self.removed_keywords
            - temporarily_removed
        )

    def has_keyword(self, keyword: str) -> bool:
        return normalize_keyword_name(keyword) in self.effective_keywords

    def add_keyword(
        self,
        keyword: str,
        *,
        duration: str = "permanent",
        expires_for_player: int | None = None,
    ) -> None:
        if self.definition.card_type != "随从":
            raise ValueError("Only follower cards can receive hand keywords")
        canonical = normalize_keyword_name(keyword, strict=True)
        if canonical not in RUNTIME_UNIT_KEYWORDS:
            raise ValueError(
                f"Keyword {canonical!r} is not a supported runtime unit keyword"
            )
        self.removed_keywords.discard(canonical)
        self.temporary_keyword_removals = [
            modifier
            for modifier in self.temporary_keyword_removals
            if modifier.keyword != canonical
        ]
        if duration == "permanent":
            self.permanent_keywords.add(canonical)
        else:
            self.temporary_keywords.append(
                KeywordModifier(canonical, duration, expires_for_player)
            )

    def remove_keyword(
        self,
        keyword: str,
        *,
        duration: str = "permanent",
        expires_for_player: int | None = None,
    ) -> None:
        if self.definition.card_type != "随从":
            raise ValueError("Only follower cards can lose hand keywords")
        canonical = normalize_keyword_name(keyword, strict=True)
        if canonical not in RUNTIME_UNIT_KEYWORDS:
            raise ValueError(
                f"Keyword {canonical!r} is not a supported runtime unit keyword"
            )
        if duration == "permanent":
            self.permanent_keywords.discard(canonical)
            self.temporary_keywords = [
                modifier
                for modifier in self.temporary_keywords
                if modifier.keyword != canonical
            ]
            self.removed_keywords.add(canonical)
        else:
            if expires_for_player is None:
                raise ValueError(
                    "Temporary keyword removal requires expires_for_player"
                )
            self.temporary_keyword_removals.append(
                KeywordRemovalModifier(
                    keyword=canonical,
                    duration=duration,
                    expires_for_player=expires_for_player,
                )
            )

    def expire_keywords(self, duration: str, player_index: int) -> None:
        self.temporary_keywords = [
            modifier
            for modifier in self.temporary_keywords
            if not (
                modifier.duration == duration
                and modifier.expires_for_player == player_index
            )
        ]
        self.temporary_keyword_removals = [
            modifier
            for modifier in self.temporary_keyword_removals
            if not (
                modifier.duration == duration
                and modifier.expires_for_player == player_index
            )
        ]

    @property
    def abilities(self):
        return self.definition.abilities

    def expire_cost_modifiers(self, duration: str, player_index: int) -> None:
        self.cost_modifiers = [
            modifier
            for modifier in self.cost_modifiers
            if not (
                modifier.duration == duration
                and modifier.expires_for_player == player_index
            )
        ]

    def add_stat_modifier(self, modifier: StatModifier) -> None:
        if self.definition.card_type != "随从":
            raise ValueError("Only follower cards can receive hand stat modifiers")
        self.stat_modifiers.append(modifier)

    def expire_stat_modifiers(self, duration: str, player_index: int) -> None:
        self.stat_modifiers = [
            modifier
            for modifier in self.stat_modifiers
            if not (
                modifier.duration == duration
                and modifier.expires_for_player == player_index
            )
        ]


@dataclass
class Unit(BoardEntity):
    attack: int
    health: int
    max_health: int
    base_attack: int
    base_health: int
    can_attack: bool = False
    attacks_remaining: int = 1
    evolved: bool = False
    super_evolved: bool = False
    super_evolved_turn: int | None = None
    rush_only: bool = False
    barrier_charges: int = 0
    ambush_active: bool = False
    summoned_this_turn: bool = True
    permanent_keywords: set[str] = field(default_factory=set)
    temporary_keywords: list[KeywordModifier] = field(default_factory=list)
    removed_keywords: set[str] = field(default_factory=set)
    temporary_keyword_removals: list[KeywordRemovalModifier] = field(
        default_factory=list
    )
    stat_modifiers: list[StatModifier] = field(default_factory=list)
    attack_capacity_modifiers: list[AttackCapacityModifier] = field(
        default_factory=list
    )
    attack_restrictions: list[AttackRestrictionModifier] = field(default_factory=list)
    targeting_restrictions: list[TargetingRestrictionModifier] = field(default_factory=list)
    printed_abilities_removed: bool = False
    last_words_removed: bool = False
    turn_end_destroy_timings: set[TurnEndDestroyTiming] = field(
        default_factory=set
    )
    granted_last_words: list[tuple[EffectOperation, ...]] = field(
        default_factory=list
    )
    effect_destroy_immunity: bool = False

    @classmethod
    def summon(
        cls,
        card: CardDefinition,
        *,
        entity_id: int = 0,
        origin: CardOrigin = CardOrigin.DECK,
        source_origin: CardOrigin | None = None,
    ) -> "Unit":
        if card.attack is None or card.life is None:
            raise ValueError(f"{card.name} is not a playable follower")
        keywords = {
            normalize_keyword_name(keyword)
            for keyword in card.keywords
        }
        barrier = 1 if "屏障" in keywords else 0
        ambush = "潜行" in keywords
        return cls(
            definition=card,
            attack=card.attack,
            health=card.life,
            max_health=card.life,
            base_attack=card.attack,
            base_health=card.life,
            entity_id=entity_id,
            can_attack="疾驰" in keywords or "突进" in keywords,
            rush_only="突进" in keywords and "疾驰" not in keywords,
            barrier_charges=barrier,
            ambush_active=ambush,
            origin=origin,
            source_origin=source_origin,
        )

    @property
    def original_keywords(self) -> frozenset[str]:
        return frozenset(
            normalize_keyword_name(keyword)
            for keyword in self.definition.keywords
        )

    @property
    def effective_keywords(self) -> frozenset[str]:
        temporary = {modifier.keyword for modifier in self.temporary_keywords}
        temporarily_removed = {
            modifier.keyword for modifier in self.temporary_keyword_removals
        }
        return frozenset(
            (
                (set() if self.printed_abilities_removed else set(self.original_keywords))
                | self.permanent_keywords
                | temporary
            )
            - self.removed_keywords
            - temporarily_removed
        )

    def remove_all_abilities(self) -> None:
        old_capacity = self.attacks_per_turn
        self.printed_abilities_removed = True
        self.permanent_keywords.clear()
        self.temporary_keywords.clear()
        self.removed_keywords.clear()
        self.temporary_keyword_removals.clear()
        self.attack_restrictions.clear()
        self.targeting_restrictions.clear()
        self.attack_capacity_modifiers.clear()
        self.turn_end_destroy_timings.clear()
        self.granted_last_words.clear()
        self.effect_destroy_immunity = False
        self._adjust_attack_capacity(old_capacity)
        self._synchronize_keyword_state()

    def remove_last_words(self) -> None:
        self.last_words_removed = True

    @property
    def attacks_per_turn(self) -> int:
        return max(
            (modifier.attacks_per_turn for modifier in self.attack_capacity_modifiers),
            default=1,
        )

    def grant_attacks_per_turn(
        self,
        attacks_per_turn: int,
        *,
        duration: str = "permanent",
        expires_for_player: int | None = None,
    ) -> None:
        if attacks_per_turn < 1:
            raise ValueError("attacks_per_turn must be positive")
        old_capacity = self.attacks_per_turn
        if not any(
            modifier.attacks_per_turn >= attacks_per_turn
            and modifier.duration == duration
            and modifier.expires_for_player == expires_for_player
            for modifier in self.attack_capacity_modifiers
        ):
            self.attack_capacity_modifiers.append(AttackCapacityModifier(
                attacks_per_turn=attacks_per_turn,
                duration=duration,
                expires_for_player=expires_for_player,
            ))
        self._adjust_attack_capacity(old_capacity)

    def expire_attack_capacity(self, duration: str, player_index: int) -> None:
        old_capacity = self.attacks_per_turn
        self.attack_capacity_modifiers = [
            modifier
            for modifier in self.attack_capacity_modifiers
            if not (
                modifier.duration == duration
                and modifier.expires_for_player == player_index
            )
        ]
        self._adjust_attack_capacity(old_capacity)

    def consume_attack(self) -> None:
        if self.attacks_remaining <= 0:
            raise ValueError("cannot consume attack with no attacks remaining")
        self.attacks_remaining -= 1
        self.can_attack = self.attacks_remaining > 0

    def _adjust_attack_capacity(self, old_capacity: int) -> None:
        attacks_used = max(0, old_capacity - self.attacks_remaining)
        was_capacity_exhausted = self.attacks_remaining == 0
        self.attacks_remaining = max(0, self.attacks_per_turn - attacks_used)
        if self.attacks_remaining == 0:
            self.can_attack = False
        elif was_capacity_exhausted and (
            not self.summoned_this_turn
            or self.evolved
            or self.has_keyword("疾驰")
            or self.has_keyword("突进")
        ):
            self.can_attack = True

    def has_keyword(self, keyword: str) -> bool:
        return normalize_keyword_name(keyword) in self.effective_keywords

    def add_keyword(
        self,
        keyword: str,
        *,
        duration: str = "permanent",
        expires_for_player: int | None = None,
    ) -> None:
        canonical = normalize_keyword_name(keyword, strict=True)
        if canonical not in RUNTIME_UNIT_KEYWORDS:
            raise ValueError(
                f"Keyword {canonical!r} is not a supported runtime unit keyword"
            )
        self.removed_keywords.discard(canonical)
        self.temporary_keyword_removals = [
            modifier
            for modifier in self.temporary_keyword_removals
            if modifier.keyword != canonical
        ]
        if duration == "permanent":
            self.permanent_keywords.add(canonical)
        else:
            self.temporary_keywords.append(
                KeywordModifier(canonical, duration, expires_for_player)
            )
        if canonical == "屏障":
            self.barrier_charges += 1
        elif canonical == "潜行":
            self.ambush_active = True
        elif canonical == "疾驰" and self.attacks_remaining > 0:
            self.can_attack = True
            self.rush_only = False
        elif canonical == "突进" and self.attacks_remaining > 0:
            self.can_attack = True
            self.rush_only = not self.has_keyword("疾驰")

    def remove_keyword(
        self,
        keyword: str,
        *,
        duration: str = "permanent",
        expires_for_player: int | None = None,
    ) -> None:
        canonical = normalize_keyword_name(keyword, strict=True)
        if canonical not in RUNTIME_UNIT_KEYWORDS:
            raise ValueError(
                f"Keyword {canonical!r} is not a supported runtime unit keyword"
            )
        if duration == "permanent":
            self.permanent_keywords.discard(canonical)
            self.temporary_keywords = [
                modifier
                for modifier in self.temporary_keywords
                if modifier.keyword != canonical
            ]
            self.removed_keywords.add(canonical)
        else:
            if expires_for_player is None:
                raise ValueError(
                    "Temporary keyword removal requires expires_for_player"
                )
            self.temporary_keyword_removals.append(
                KeywordRemovalModifier(
                    keyword=canonical,
                    duration=duration,
                    expires_for_player=expires_for_player,
                    restore_barrier_charge=(
                        canonical == "屏障" and self.barrier_charges > 0
                    ),
                    restore_ambush=(
                        canonical == "潜行" and self.ambush_active
                    ),
                )
            )
        self._synchronize_keyword_state()

    def expire_keywords(self, duration: str, player_index: int) -> None:
        expired = [
            modifier
            for modifier in self.temporary_keywords
            if (
                modifier.duration == duration
                and modifier.expires_for_player == player_index
            )
        ]
        self.temporary_keywords = [
            modifier
            for modifier in self.temporary_keywords
            if not (
                modifier.duration == duration
                and modifier.expires_for_player == player_index
            )
        ]
        expired_barriers = sum(
            modifier.keyword == "屏障" for modifier in expired
        )
        self.barrier_charges = max(0, self.barrier_charges - expired_barriers)
        expired_removals = [
            modifier
            for modifier in self.temporary_keyword_removals
            if (
                modifier.duration == duration
                and modifier.expires_for_player == player_index
            )
        ]
        self.temporary_keyword_removals = [
            modifier
            for modifier in self.temporary_keyword_removals
            if modifier not in expired_removals
        ]
        self._synchronize_keyword_state()
        if self.has_keyword("屏障") and any(
            modifier.restore_barrier_charge
            for modifier in expired_removals
        ):
            self.barrier_charges += 1
        if self.has_keyword("潜行") and any(
            modifier.restore_ambush for modifier in expired_removals
        ):
            self.ambush_active = True

    def _synchronize_keyword_state(self) -> None:
        if not self.has_keyword("屏障"):
            self.barrier_charges = 0
        if not self.has_keyword("潜行"):
            self.ambush_active = False
        if self.summoned_this_turn:
            if self.has_keyword("疾驰"):
                self.can_attack = self.attacks_remaining > 0
                self.rush_only = False
            elif self.has_keyword("突进"):
                self.can_attack = self.attacks_remaining > 0
                self.rush_only = True
            else:
                self.can_attack = False
                self.rush_only = False

    def _recompute_max(self) -> None:
        self.max_health = self.base_health
        for m in self.stat_modifiers:
            self.max_health += m.health_delta
        if self.max_health < 1:
            self.max_health = 1

    def _recompute_attack(self) -> None:
        self.attack = self.base_attack
        for m in self.stat_modifiers:
            self.attack += m.attack_delta
        if self.attack < 0:
            self.attack = 0

    def add_stat_modifier(self, modifier: StatModifier) -> None:
        self.stat_modifiers.append(modifier)
        self.health += modifier.health_delta
        self._recompute_attack()
        self._recompute_max()

    def expire_stat_modifiers(self, duration: str, player_index: int) -> None:
        remaining: list[StatModifier] = []
        for modifier in self.stat_modifiers:
            if (
                modifier.duration == duration
                and modifier.expires_for_player == player_index
            ):
                self.health -= modifier.health_delta
            else:
                remaining.append(modifier)
        self.stat_modifiers = remaining
        self._recompute_attack()
        self._recompute_max()
        if self.health > self.max_health:
            self.health = self.max_health

    @property
    def has_guard(self) -> bool:
        return self.has_keyword("守护")

    @property
    def has_intimidate(self) -> bool:
        return self.has_keyword("威慑")

    @property
    def has_aura(self) -> bool:
        return self.has_keyword("灵气")

    @property
    def can_attack_leader(self) -> bool:
        if any(r.restriction == AttackRestriction.CANNOT_ATTACK for r in self.attack_restrictions):
            return False
        if any(r.restriction == AttackRestriction.CANNOT_ATTACK_LEADER for r in self.attack_restrictions):
            return False
        return self.can_attack and not self.rush_only

    @property
    def can_attack_units(self) -> bool:
        if not self.can_attack:
            return False
        if any(r.restriction == AttackRestriction.CANNOT_ATTACK for r in self.attack_restrictions):
            return False
        if any(r.restriction == AttackRestriction.CANNOT_ATTACK_UNITS for r in self.attack_restrictions):
            return False
        return True

    @property
    def cannot_be_enemy_targeted(self) -> bool:
        return self.has_aura or any(
            r.restriction == TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS
            for r in self.targeting_restrictions
        )

    @property
    def forces_enemy_ability_target(self) -> bool:
        return any(
            r.restriction == TargetingRestriction.FORCES_ENEMY_ABILITY_TARGET
            for r in self.targeting_restrictions
        )

    def add_attack_restriction(
        self, restriction: AttackRestriction, *, duration: str, expires_for_player: int | None = None
    ) -> None:
        self.attack_restrictions.append(
            AttackRestrictionModifier(restriction, duration, expires_for_player)
        )

    def remove_attack_restriction(self, restriction: AttackRestriction) -> None:
        self.attack_restrictions = [
            r for r in self.attack_restrictions if r.restriction != restriction
        ]

    def expire_attack_restrictions(self, duration: str, player_index: int) -> None:
        self.attack_restrictions = [
            r for r in self.attack_restrictions
            if not (r.duration == duration and r.expires_for_player == player_index)
        ]

    def add_targeting_restriction(
        self, restriction: TargetingRestriction, *, duration: str, expires_for_player: int | None = None
    ) -> None:
        self.targeting_restrictions.append(
            TargetingRestrictionModifier(restriction, duration, expires_for_player)
        )

    def remove_targeting_restriction(self, restriction: TargetingRestriction) -> None:
        self.targeting_restrictions = [
            r for r in self.targeting_restrictions if r.restriction != restriction
        ]

    def expire_targeting_restrictions(self, duration: str, player_index: int) -> None:
        self.targeting_restrictions = [
            r for r in self.targeting_restrictions
            if not (r.duration == duration and r.expires_for_player == player_index)
        ]


@dataclass
class Amulet(BoardEntity):
    countdown: int | None = None
    earth_sigil_count: int = 0
    entered_turn: int = 0
    activated_turn: int | None = None
    pending_destroy: bool = False


BoardCard = Unit | Amulet


@dataclass
class EmblemInstance:
    emblem_id: str
    definition: EmblemDefinition
    entity_id: int
    controller: int
    created_sequence: int
    countdown: int | None = None
    countdown_before: int | None = None
    activation_counts: dict[int, int] = field(default_factory=dict)
    _once_per_turn_used: set[int] = field(default_factory=set)

    @property
    def is_permanent(self) -> bool:
        return self.countdown is None and self.definition.countdown is None

    @property
    def source_card_id(self) -> int:
        return self.definition.source_card_id

    def can_activate(self, trigger_index: int) -> bool:
        tr = self.definition.triggers[trigger_index]
        if tr.once_per_turn and trigger_index in self._once_per_turn_used:
            return False
        if tr.max_activations is not None:
            current = self.activation_counts.get(trigger_index, 0)
            if current >= tr.max_activations:
                return False
        return True

    def record_activation(self, trigger_index: int) -> None:
        self.activation_counts[trigger_index] = self.activation_counts.get(trigger_index, 0) + 1
        tr = self.definition.triggers[trigger_index]
        if tr.once_per_turn:
            self._once_per_turn_used.add(trigger_index)

    def reset_turn_limits(self) -> None:
        self._once_per_turn_used.clear()


@dataclass
class PlayerState:
    deck: list[CardDefinition | DeckCard]
    class_id: int
    class_name: str
    hand: list[HandCard] = field(default_factory=list)
    hand_entity_ids: list[int] = field(default_factory=list)
    board: list[BoardCard] = field(default_factory=list)
    graveyard: list[GraveyardCard] = field(default_factory=list)
    banished: list[CardDefinition] = field(default_factory=list)
    emblems: list[EmblemInstance] = field(default_factory=list)
    faiths: list[FaithInstance] = field(default_factory=list)
    fusion_materials: list[FusionMaterial] = field(default_factory=list)
    health: int = 20
    max_health: int = 20
    max_mana: int = 0
    mana: int = 0
    fatigue: int = 0
    empty_deck_outcome: EmptyDeckOutcome = EmptyDeckOutcome.DEFEAT
    evolution_points: int = 2
    super_evolution_points: int = 2
    turns_started: int = 0
    evolved_this_turn: bool = False
    super_evolved_this_turn: bool = False
    followers_evolved_this_match: int = 0
    cards_played_this_turn: int = 0
    follower_attacks_this_turn: int = 0
    followers_destroyed_this_turn: int = 0
    cooperation: int = 0
    shadows: int = 0
    leader_damage_modifiers: list[LeaderDamageModifier] = field(default_factory=list)
    _next_graveyard_sequence: int = 1
    _next_emblem_sequence: int = 1
    _next_faith_sequence: int = 1
    _next_fusion_sequence: int = 1

    def add_shadows(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("shadows increase must be non-negative")
        self.shadows += amount

    def consume_shadows(self, amount: int) -> bool:
        if amount < 0:
            raise ValueError("shadows consume amount must be non-negative")
        if self.shadows < amount:
            return False
        self.shadows -= amount
        return True

    def add_cooperation(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("cooperation increase must be non-negative")
        self.cooperation += amount

    def add_combo(self, amount: int) -> None:
        if amount < 0:
            raise ValueError("combo increase must be non-negative")
        self.cards_played_this_turn += amount

    def expire_leader_damage_modifiers(
        self, duration: str, player_index: int
    ) -> None:
        self.leader_damage_modifiers = [
            modifier
            for modifier in self.leader_damage_modifiers
            if not (
                modifier.duration == duration
                and modifier.expires_for_player == player_index
            )
        ]

    @property
    def earth_sigils(self) -> int:
        return sum(
            entity.earth_sigil_count
            for entity in self.board
            if isinstance(entity, Amulet)
        )


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
    destroyed_followers: list[DestroyedFollowerRecord] = field(default_factory=list)
    follower_entries: list[FollowerEntryRecord] = field(default_factory=list)
    listener_activation_counts: dict[tuple[int, int, int], int] = field(
        default_factory=dict
    )
    listener_once_per_turn_used: set[tuple[int, int, int]] = field(
        default_factory=set
    )
    _next_death_sequence: int = 1
    _next_follower_entry_sequence: int = 1

    @property
    def terminated(self) -> bool:
        return self.phase is Phase.FINISHED

    def allocate_entity_id(self) -> int:
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        return entity_id
