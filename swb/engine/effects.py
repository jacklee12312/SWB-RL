from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from swb.db.repository import CardDefinition
    from swb.engine.state import FusionMaterial


MAX_REPEAT_COUNT = 100
MAX_RANDOM_DISTRIBUTION_TOTAL = 10_000


class ConditionType(str, Enum):
    ALWAYS = "always"
    ALL = "all"
    ANY = "any"
    NOT = "not"
    CONTROLLER_HEALTH_AT_MOST = "controller_health_at_most"
    CONTROLLER_HEALTH_AT_LEAST = "controller_health_at_least"
    OPPONENT_HEALTH_AT_MOST = "opponent_health_at_most"
    OPPONENT_HEALTH_AT_LEAST = "opponent_health_at_least"
    CONTROLLER_BOARD_COUNT_AT_LEAST = "controller_board_count_at_least"
    OPPONENT_BOARD_COUNT_AT_LEAST = "opponent_board_count_at_least"
    CONTROLLER_BOARD_HAS = "controller_board_has"
    OPPONENT_BOARD_HAS = "opponent_board_has"
    CONTROLLER_HAND_COUNT_AT_LEAST = "controller_hand_count_at_least"
    CONTROLLER_MAX_MANA_AT_LEAST = "controller_max_mana_at_least"
    OPPONENT_MAX_MANA_AT_LEAST = "opponent_max_mana_at_least"
    CONTROLLER_DECK_HAS_NO_DUPLICATES = "controller_deck_has_no_duplicates"
    TARGET_ATTACK_AT_MOST = "target_attack_at_most"
    TARGET_ATTACK_AT_LEAST = "target_attack_at_least"
    TARGET_HEALTH_AT_MOST = "target_health_at_most"
    TARGET_HEALTH_AT_LEAST = "target_health_at_least"
    SOURCE_EVOLVED = "source_evolved"
    SOURCE_SUPER_EVOLVED = "source_super_evolved"
    SOURCE_HEALTH_AT_MOST = "source_health_at_most"
    SOURCE_HEALTH_AT_LEAST = "source_health_at_least"
    SOURCE_HAS_KEYWORD = "source_has_keyword"
    SOURCE_CARD_TYPE_IS = "source_card_type_is"
    TARGET_HAS_KEYWORD = "target_has_keyword"
    TARGET_IS_OWN = "target_is_own"
    TARGET_CARD_TYPE_IS = "target_card_type_is"
    CONTROLLER_SHADOWS_AT_LEAST = "controller_shadows_at_least"
    OPPONENT_SHADOWS_AT_LEAST = "opponent_shadows_at_least"
    CONTROLLER_COOPERATION_AT_LEAST = "controller_cooperation_at_least"
    OPPONENT_COOPERATION_AT_LEAST = "opponent_cooperation_at_least"
    CONTROLLER_OVERFLOW = "controller_overflow"
    OPPONENT_OVERFLOW = "opponent_overflow"
    CONTROLLER_COMBO_AT_LEAST = "controller_combo_at_least"
    OPPONENT_COMBO_AT_LEAST = "opponent_combo_at_least"
    CONTROLLER_FOLLOWER_ATTACKS_THIS_TURN_AT_MOST = (
        "controller_follower_attacks_this_turn_at_most"
    )
    CONTROLLER_EARTH_SIGILS_AT_LEAST = "controller_earth_sigils_at_least"
    OPPONENT_EARTH_SIGILS_AT_LEAST = "opponent_earth_sigils_at_least"
    CONTROLLER_EVOLUTIONS_THIS_MATCH_AT_LEAST = "controller_evolutions_this_match_at_least"
    OPPONENT_EVOLUTIONS_THIS_MATCH_AT_LEAST = "opponent_evolutions_this_match_at_least"
    CONTROLLER_SUPER_EVOLUTION_UNLOCKED = "controller_super_evolution_unlocked"
    OPPONENT_SUPER_EVOLUTION_UNLOCKED = "opponent_super_evolution_unlocked"
    SOURCE_FUSION_COUNT_AT_LEAST = "source_fusion_count_at_least"
    SOURCE_SPELLBOOST_COUNT_AT_LEAST = "source_spellboost_count_at_least"
    SOURCE_COST_EQUALS = "source_cost_equals"
    CONTROLLER_ENTERED_FOLLOWER_DISTINCT_COUNT_AT_LEAST = (
        "controller_entered_follower_distinct_count_at_least"
    )


class ExprType(str, Enum):
    CONSTANT = "constant"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    MIN = "min"
    MAX = "max"
    CONTROLLER_BOARD_COUNT = "controller_board_count"
    OPPONENT_BOARD_COUNT = "opponent_board_count"
    CONTROLLER_HAND_COUNT = "controller_hand_count"
    CONTROLLER_EMBLEM_COUNT = "controller_emblem_count"
    SOURCE_SPELLBOOST_COUNT = "source_spellboost_count"
    SOURCE_COST = "source_cost"
    BOUND_CARD_COST = "bound_card_cost"
    BOUND_TARGET_HEALTH = "bound_target_health"
    SOURCE_ATTACK = "source_attack"
    SOURCE_HEALTH = "source_health"
    TARGET_ATTACK = "target_attack"
    TARGET_HEALTH = "target_health"
    CONTROLLER_SHADOWS = "controller_shadows"
    OPPONENT_SHADOWS = "opponent_shadows"
    CONTROLLER_COOPERATION = "controller_cooperation"
    OPPONENT_COOPERATION = "opponent_cooperation"
    CONTROLLER_OVERFLOW = "controller_overflow"
    OPPONENT_OVERFLOW = "opponent_overflow"
    CONTROLLER_COMBO = "controller_combo"
    OPPONENT_COMBO = "opponent_combo"
    CONTROLLER_EARTH_SIGILS = "controller_earth_sigils"
    OPPONENT_EARTH_SIGILS = "opponent_earth_sigils"
    CONTROLLER_DESTROYED_FOLLOWER_BASE_ATTACK_SUM_THIS_TURN = (
        "controller_destroyed_follower_base_attack_sum_this_turn"
    )
    CONTROLLER_DESTROYED_FOLLOWER_BASE_HEALTH_SUM_THIS_TURN = (
        "controller_destroyed_follower_base_health_sum_this_turn"
    )
    CONTROLLER_ENTERED_FOLLOWER_DISTINCT_COUNT = (
        "controller_entered_follower_distinct_count"
    )
    DISTRIBUTED_VALUE = "distributed_value"


@dataclass
class Condition:
    type: ConditionType
    value: int = 0
    keyword: str | None = None
    card_type: str | None = None
    board_filter: "BoardFilter | None" = None
    card_filter: "HandFilter | None" = None
    conditions: list[Condition] = field(default_factory=list)

    @classmethod
    def always(cls) -> "Condition":
        return cls(type=ConditionType.ALWAYS)


@dataclass
class ValueExpression:
    type: ExprType
    value: int = 0
    values: list[ValueExpression] = field(default_factory=list)
    card_filter: "HandFilter | None" = None
    board_filter: "BoardFilter | None" = None
    binding_key: str | None = None

    @classmethod
    def constant(cls, v: int) -> "ValueExpression":
        return cls(type=ExprType.CONSTANT, value=v)


class EffectKind(str, Enum):
    SELECT_TARGETS = "select_targets"
    DRAW = "draw"
    DRAW_FILTERED = "draw_filtered"
    HEAL_LEADER = "heal_leader"
    HEAL_UNIT = "heal_unit"
    DAMAGE_LEADER = "damage_leader"
    DAMAGE_UNIT = "damage_unit"
    DISTRIBUTE_DAMAGE = "distribute_damage"
    RESTORE_MANA = "restore_mana"
    RESTORE_EVOLUTION_POINTS = "restore_evolution_points"
    RESTORE_SUPER_EVOLUTION_POINTS = "restore_super_evolution_points"
    CHANGE_MAX_MANA = "change_max_mana"
    BUFF_UNIT = "buff_unit"
    BUFF_HAND_CARD = "buff_hand_card"
    SUMMON = "summon"
    SUMMON_COPY = "summon_copy"
    SUMMON_HAND_COPY = "summon_hand_copy"
    SUMMON_FROM_DECK = "summon_from_deck"
    DESTROY = "destroy"
    BANISH = "banish"
    ADD_CARD = "add_card"
    ADD_CARD_TO_DECK = "add_card_to_deck"
    COPY_TO_HAND = "copy_to_hand"
    COPY_DESTROYED_FOLLOWERS_TO_HAND = "copy_destroyed_followers_to_hand"
    ADD_KEYWORD = "add_keyword"
    REMOVE_KEYWORD = "remove_keyword"
    REMOVE_ALL_ABILITIES = "remove_all_abilities"
    GRANT_ATTACKS_PER_TURN = "grant_attacks_per_turn"
    GRANT_TURN_END_DESTROY = "grant_turn_end_destroy"
    ADD_LEADER_DAMAGE_MODIFIER = "add_leader_damage_modifier"
    CHANGE_COST = "change_cost"
    CHANGE_DECK_COST = "change_deck_cost"
    REPLACE_DECK = "replace_deck"
    SET_EMPTY_DECK_OUTCOME = "set_empty_deck_outcome"
    SET_LEADER_MAX_HEALTH = "set_leader_max_health"
    TRANSFORM = "transform"
    GAIN_EMBLEM = "gain_emblem"
    ADD_EMBLEM = "add_emblem"
    REMOVE_EMBLEM = "remove_emblem"
    RETURN_TO_HAND = "return_to_hand"
    RETURN_TO_DECK = "return_to_deck"
    REDUCE_COUNTDOWN = "reduce_countdown"
    INCREASE_COUNTDOWN = "increase_countdown"
    DISCARD = "discard"
    SET_STATS = "set_stats"
    EVOLVE_UNIT = "evolve_unit"
    SUPER_EVOLVE_UNIT = "super_evolve_unit"
    ADD_ATTACK_RESTRICTION = "add_attack_restriction"
    REMOVE_ATTACK_RESTRICTION = "remove_attack_restriction"
    ADD_TARGETING_RESTRICTION = "add_targeting_restriction"
    REMOVE_TARGETING_RESTRICTION = "remove_targeting_restriction"
    SPELLBOOST_HAND = "spellboost_hand"
    ADD_COMBO = "add_combo"
    ADD_SHADOWS = "add_shadows"
    ADD_EARTH_SIGILS = "add_earth_sigils"
    EARTH_RITE = "earth_rite"
    CONSUME_FAITH = "consume_faith"
    GRANT_FAITH_ABILITY = "grant_faith_ability"
    NECROMANCY = "necromancy"
    REANIMATE = "reanimate"
    RETURN_FROM_GRAVEYARD_TO_HAND = "return_from_graveyard_to_hand"
    SUMMON_FROM_GRAVEYARD = "summon_from_graveyard"
    BANISH_FROM_GRAVEYARD = "banish_from_graveyard"
    CONDITIONAL = "conditional"
    CHOOSE_ONE = "choose_one"
    OPTIONAL = "optional"
    TARGET_EXISTS = "target_exists"
    REPEAT = "repeat"
    RANDOM_DISTRIBUTE = "random_distribute"


class CandidateExtreme(str, Enum):
    HIGHEST_ATTACK = "highest_attack"
    LOWEST_ATTACK = "lowest_attack"
    HIGHEST_HEALTH = "highest_health"
    LOWEST_HEALTH = "lowest_health"


class EmptyDeckOutcome(str, Enum):
    """Result of attempting to draw while a player's deck is empty."""

    DEFEAT = "defeat"
    VICTORY = "victory"


class TargetKind(str, Enum):
    SELF = "self"
    EMBLEM_SELF = "emblem_self"
    EVENT_SOURCE = "event_source"
    ATTACK_TARGET = "attack_target"
    OWN_LEADER = "own_leader"
    ENEMY_LEADER = "enemy_leader"
    OWN_UNIT = "own_unit"
    ENEMY_UNIT = "enemy_unit"
    OWN_UNIT_OR_LEADER = "own_unit_or_leader"
    ENEMY_UNIT_OR_LEADER = "enemy_unit_or_leader"
    ANY_UNIT_OR_LEADER = "any_unit_or_leader"
    OWN_BOARD = "own_board"
    ENEMY_BOARD = "enemy_board"
    ANY_UNIT = "any_unit"
    OWN_AMULET = "own_amulet"
    ENEMY_AMULET = "enemy_amulet"
    ANY_AMULET = "any_amulet"
    ANY_BOARD = "any_board"
    RANDOM_OWN_UNIT = "random_own_unit"
    RANDOM_ENEMY_UNIT = "random_enemy_unit"
    RANDOM_ENEMY_UNIT_OR_LEADER = "random_enemy_unit_or_leader"
    RANDOM_OWN_BOARD = "random_own_board"
    RANDOM_ENEMY_BOARD = "random_enemy_board"
    ALL_OWN_UNITS = "all_own_units"
    ALL_ENEMY_UNITS = "all_enemy_units"
    ALL_UNITS = "all_units"
    ALL_OWN_BOARD = "all_own_board"
    ALL_ENEMY_BOARD = "all_enemy_board"
    ALL_BOARD = "all_board"
    ALL_OWN_AMULETS = "all_own_amulets"
    ALL_ENEMY_AMULETS = "all_enemy_amulets"
    ALL_OWN_EMBLEMS = "all_own_emblems"
    ALL_LEADERS = "all_leaders"
    OWN_HAND = "own_hand"
    RANDOM_OWN_HAND = "random_own_hand"
    RANDOM_ENEMY_HAND = "random_enemy_hand"
    ALL_OWN_HAND = "all_own_hand"
    ALL_ENEMY_HAND = "all_enemy_hand"
    PREVIOUS_TARGET = "previous_target"
    OWN_GRAVEYARD_CARD = "own_graveyard_card"
    RANDOM_OWN_GRAVEYARD_CARD = "random_own_graveyard_card"
    ALL_OWN_GRAVEYARD_CARDS = "all_own_graveyard_cards"


class ModifierDuration(str, Enum):
    PERMANENT = "permanent"
    UNTIL_END_OF_TURN = "until_end_of_turn"
    UNTIL_START_OF_NEXT_TURN = "until_start_of_next_turn"
    UNTIL_END_OF_CONTROLLER_TURN = "until_end_of_controller_turn"
    UNTIL_END_OF_OPPONENT_TURN = "until_end_of_opponent_turn"
    UNTIL_START_OF_CONTROLLER_NEXT_TURN = "until_start_of_controller_next_turn"
    WHILE_SOURCE_IN_PLAY = "while_source_in_play"


class TurnEndDestroyTiming(str, Enum):
    OWNER_TURN = "owner_turn"
    OPPONENT_TURN = "opponent_turn"


class CostChangeMode(str, Enum):
    SET = "set"
    ADD = "add"
    SUBTRACT = "subtract"


@dataclass(frozen=True)
class ChooseOneOption:
    option_id: str
    label: str
    conditions: tuple["Condition", ...] = ()
    operations: tuple["EffectOperation", ...] = ()


@dataclass(frozen=True)
class DeckFilter:
    card_type: str | None = None
    class_id: int | None = None
    class_name: str | None = None
    cost_min: int | None = None
    cost_max: int | None = None
    card_id: int | None = None
    card_ids: tuple[int, ...] = ()
    card_name: str | None = None
    tribe_id: int | None = None
    tribe_name: str | None = None

    def matches(self, card: CardDefinition) -> bool:
        return (
            (self.card_type is None or card.card_type == self.card_type)
            and (self.class_id is None or card.class_id == self.class_id)
            and (self.class_name is None or card.class_name == self.class_name)
            and (self.cost_min is None or card.cost >= self.cost_min)
            and (self.cost_max is None or card.cost <= self.cost_max)
            and (self.card_id is None or card.card_id == self.card_id)
            and (not self.card_ids or card.card_id in self.card_ids)
            and (self.card_name is None or card.name == self.card_name)
            and (self.tribe_id is None or card.tribe_id == self.tribe_id)
            and (self.tribe_name is None or card.tribe_name == self.tribe_name)
        )


@dataclass(frozen=True)
class HandFilter:
    """Matches physical hand cards, using their current cost when available."""

    card_type: str | None = None
    class_id: int | None = None
    class_name: str | None = None
    cost_min: int | None = None
    cost_max: int | None = None
    card_id: int | None = None
    exclude_card_ids: tuple[int, ...] = ()
    card_name: str | None = None
    tribe_id: int | None = None
    tribe_name: str | None = None
    keyword: str | None = None

    def matches(self, card: CardDefinition) -> bool:
        definition = getattr(card, "definition", card)
        cost = getattr(card, "current_cost", definition.cost)
        return (
            (self.card_type is None or definition.card_type == self.card_type)
            and (self.class_id is None or definition.class_id == self.class_id)
            and (self.class_name is None or definition.class_name == self.class_name)
            and (self.cost_min is None or cost >= self.cost_min)
            and (self.cost_max is None or cost <= self.cost_max)
            and (self.card_id is None or definition.card_id == self.card_id)
            and definition.card_id not in self.exclude_card_ids
            and (self.card_name is None or definition.name == self.card_name)
            and (self.tribe_id is None or definition.tribe_id == self.tribe_id)
            and (self.tribe_name is None or definition.tribe_name == self.tribe_name)
            and (
                self.keyword is None
                or self.keyword
                in {
                    getattr(ability, "value", str(ability))
                    for ability in definition.abilities
                }
            )
        )


@dataclass(frozen=True)
class BoardFilter:
    card_type: str | None = None
    class_id: int | None = None
    class_name: str | None = None
    cost_min: int | None = None
    cost_max: int | None = None
    card_id: int | None = None
    card_name: str | None = None
    tribe_id: int | None = None
    tribe_name: str | None = None
    evolved: bool | None = None
    super_evolved: bool | None = None
    damaged: bool | None = None
    keyword: str | None = None

    def _matches_definition(self, card: CardDefinition) -> bool:
        return (
            (self.card_type is None or card.card_type == self.card_type)
            and (self.class_id is None or card.class_id == self.class_id)
            and (self.class_name is None or card.class_name == self.class_name)
            and (self.cost_min is None or card.cost >= self.cost_min)
            and (self.cost_max is None or card.cost <= self.cost_max)
            and (self.card_id is None or card.card_id == self.card_id)
            and (self.card_name is None or card.name == self.card_name)
            and (self.tribe_id is None or card.tribe_id == self.tribe_id)
            and (self.tribe_name is None or card.tribe_name == self.tribe_name)
        )

    def matches(self, card: CardDefinition) -> bool:
        return self._matches_definition(card) and (
            self.keyword is None
            or self.keyword
            in {
                getattr(ability, "value", str(ability))
                for ability in card.abilities
            }
        )

    def matches_entity(self, entity: Any) -> bool:
        definition = getattr(entity, "definition", None)
        if definition is None or not self._matches_definition(definition):
            return False
        if self.keyword is not None:
            has_keyword = getattr(entity, "has_keyword", None)
            if not callable(has_keyword) or not has_keyword(self.keyword):
                return False
        if self.evolved is not None and getattr(entity, "evolved", False) is not self.evolved:
            return False
        if (
            self.super_evolved is not None
            and getattr(entity, "super_evolved", False) is not self.super_evolved
        ):
            return False
        if self.damaged is not None:
            health = getattr(entity, "health", None)
            max_health = getattr(entity, "max_health", None)
            if health is None or max_health is None:
                return False
            if (health < max_health) is not self.damaged:
                return False
        return True


@dataclass(frozen=True)
class EffectOperation:
    kind: EffectKind
    target: TargetKind
    amount: int = 0
    secondary_amount: int = 0
    card_id: int | None = None
    card_ids: tuple[int, ...] = ()
    shuffle: bool = True
    empty_deck_outcome: EmptyDeckOutcome | None = None
    emblem_id: str | None = None
    keyword: str | None = None
    restriction: str | None = None
    conditions: tuple[Condition, ...] = ()
    amount_expr: ValueExpression | None = None
    secondary_expr: ValueExpression | None = None
    mode: CostChangeMode | None = None
    duration: ModifierDuration = ModifierDuration.PERMANENT
    set_attack: bool = False
    set_health: bool = False
    target_key: str | None = None
    condition_target_key: str | None = None
    earth_rite_operations: tuple["EffectOperation", ...] = ()
    necromancy_operations: tuple["EffectOperation", ...] = ()
    faith_id: str | None = None
    faith_ability_id: str | None = None
    faith_trigger: str | None = None
    faith_stacking: str = "unique"
    faith_operations: tuple["EffectOperation", ...] = ()
    graveyard_cost_max: int | None = None
    graveyard_cost_min: int | None = None
    graveyard_follower_only: bool = False
    graveyard_card_type: str | None = None
    deck_filter: DeckFilter | None = None
    board_filter: BoardFilter | None = None
    candidate_extreme: CandidateExtreme | None = None
    then_operations: tuple["EffectOperation", ...] = ()
    else_operations: tuple["EffectOperation", ...] = ()
    choose_one_options: tuple["ChooseOneOption", ...] = ()
    choose_count: int = 1
    optional_prompt: str | None = None
    optional_operations: tuple["EffectOperation", ...] = ()
    repeat_operations: tuple["EffectOperation", ...] = ()
    random_distribution_operations: tuple[
        tuple["EffectOperation", ...], ...
    ] = ()
    emblem_remove_mode: str = "first"
    requires_target: bool = False
    requires_full_target_count: bool = False
    target_count: int = 1
    target_count_expr: ValueExpression | None = None
    allow_duplicate_targets: bool = False
    exclude_source: bool = False
    hand_filter: HandFilter | None = None
    history_filter: HandFilter | None = None
    distinct_card_names: bool = False
    include_leader: bool = False
    turn_end_destroy_timing: TurnEndDestroyTiming | None = None


@dataclass(frozen=True)
class BoundTargetSnapshot:
    entity_id: int
    controller: int
    zone: str
    card_id: int
    card_type: str
    card_name: str
    cost: int
    definition: "CardDefinition"


@dataclass(frozen=True)
class SourceStateSnapshot:
    """Immutable source state retained for effects that fire after it leaves play."""

    entity_id: int
    controller: int
    card_id: int
    card_type: str
    attack: int | None
    health: int | None
    evolved: bool
    super_evolved: bool
    effective_keywords: frozenset[str]


@dataclass
class EffectFrame:
    controller: int
    source_card_id: int
    source_name: str
    source_entity_id: int | None
    source_card: CardDefinition
    operations: tuple[EffectOperation, ...]
    source_snapshot: SourceStateSnapshot | None = None
    source_spellboost_count: int = 0
    source_cost: int = 0
    distributed_value: int = 0
    fusion_materials: tuple["FusionMaterial", ...] = ()
    label: str = "效果"
    next_index: int = 0
    pending_target_id: int | None = None
    pending_target_ids: list[int] = field(default_factory=list)
    move_source_to_graveyard: bool = False
    _all_target_ids: list[int] = field(default_factory=list)
    _all_target_index: int = 0
    defer_stabilize: bool = False
    auto_resolve_choices: bool = False
    _hand_source_entity_id: int | None = None
    _hand_source_origin: Any = None
    _hand_source_origin_parent: Any = None
    _target_bindings: dict[str, tuple[int, ...]] = field(default_factory=dict)
    _target_binding_operations: dict[str, EffectOperation] = field(default_factory=dict)
    _target_binding_snapshots: dict[
        str, tuple[BoundTargetSnapshot, ...]
    ] = field(default_factory=dict)
    _decision_meta: dict[str, Any] = field(default_factory=dict)
    emblem_batch_id: int | None = None
    emblem_activation_owner: int | None = None
    emblem_activation_entity_id: int | None = None
    emblem_activation_trigger_index: int | None = None
    listener_batch_id: int | None = None
    listener_activation_owner: int | None = None
    listener_activation_zone: str | None = None
    listener_activation_entity_id: int | None = None
    listener_activation_card_id: int | None = None
    listener_activation_definition_index: int | None = None
    event_source_entity_id: int | None = None
    attack_target_entity_id: int | None = None
    emblem_expiration_batch_id: int | None = None
    expiring_emblem_owner: int | None = None
    expiring_emblem_entity_id: int | None = None
