from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from swb.db.repository import CardDefinition


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
    CONTROLLER_DECK_HAS_NO_DUPLICATES = "controller_deck_has_no_duplicates"
    TARGET_ATTACK_AT_MOST = "target_attack_at_most"
    TARGET_ATTACK_AT_LEAST = "target_attack_at_least"
    TARGET_HEALTH_AT_MOST = "target_health_at_most"
    TARGET_HEALTH_AT_LEAST = "target_health_at_least"
    SOURCE_EVOLVED = "source_evolved"
    SOURCE_HAS_KEYWORD = "source_has_keyword"
    TARGET_HAS_KEYWORD = "target_has_keyword"
    CONTROLLER_SHADOWS_AT_LEAST = "controller_shadows_at_least"
    OPPONENT_SHADOWS_AT_LEAST = "opponent_shadows_at_least"
    CONTROLLER_COOPERATION_AT_LEAST = "controller_cooperation_at_least"
    OPPONENT_COOPERATION_AT_LEAST = "opponent_cooperation_at_least"
    CONTROLLER_OVERFLOW = "controller_overflow"
    OPPONENT_OVERFLOW = "opponent_overflow"
    CONTROLLER_COMBO_AT_LEAST = "controller_combo_at_least"
    OPPONENT_COMBO_AT_LEAST = "opponent_combo_at_least"


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


@dataclass
class Condition:
    type: ConditionType
    value: int = 0
    keyword: str | None = None
    board_filter: "BoardFilter | None" = None
    conditions: list[Condition] = field(default_factory=list)

    @classmethod
    def always(cls) -> "Condition":
        return cls(type=ConditionType.ALWAYS)


@dataclass
class ValueExpression:
    type: ExprType
    value: int = 0
    values: list[ValueExpression] = field(default_factory=list)

    @classmethod
    def constant(cls, v: int) -> "ValueExpression":
        return cls(type=ExprType.CONSTANT, value=v)


class EffectKind(str, Enum):
    DRAW = "draw"
    DRAW_FILTERED = "draw_filtered"
    HEAL_LEADER = "heal_leader"
    DAMAGE_LEADER = "damage_leader"
    DAMAGE_UNIT = "damage_unit"
    RESTORE_MANA = "restore_mana"
    BUFF_UNIT = "buff_unit"
    SUMMON = "summon"
    DESTROY = "destroy"
    BANISH = "banish"
    ADD_CARD = "add_card"
    ADD_KEYWORD = "add_keyword"
    REMOVE_KEYWORD = "remove_keyword"
    CHANGE_COST = "change_cost"
    TRANSFORM = "transform"
    GAIN_EMBLEM = "gain_emblem"
    ADD_EMBLEM = "add_emblem"
    REMOVE_EMBLEM = "remove_emblem"
    RETURN_TO_HAND = "return_to_hand"
    RETURN_TO_DECK = "return_to_deck"
    DISCARD = "discard"
    SET_STATS = "set_stats"
    ADD_ATTACK_RESTRICTION = "add_attack_restriction"
    REMOVE_ATTACK_RESTRICTION = "remove_attack_restriction"
    ADD_TARGETING_RESTRICTION = "add_targeting_restriction"
    REMOVE_TARGETING_RESTRICTION = "remove_targeting_restriction"
    SPELLBOOST_HAND = "spellboost_hand"
    ADD_COMBO = "add_combo"
    NECROMANCY = "necromancy"
    REANIMATE = "reanimate"
    RETURN_FROM_GRAVEYARD_TO_HAND = "return_from_graveyard_to_hand"
    SUMMON_FROM_GRAVEYARD = "summon_from_graveyard"
    BANISH_FROM_GRAVEYARD = "banish_from_graveyard"
    CONDITIONAL = "conditional"
    CHOOSE_ONE = "choose_one"
    OPTIONAL = "optional"
    TARGET_EXISTS = "target_exists"


class TargetKind(str, Enum):
    SELF = "self"
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
    OWN_HAND = "own_hand"
    RANDOM_OWN_HAND = "random_own_hand"
    ALL_OWN_HAND = "all_own_hand"
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
    card_name: str | None = None

    def matches(self, card: CardDefinition) -> bool:
        return (
            (self.card_type is None or card.card_type == self.card_type)
            and (self.class_id is None or card.class_id == self.class_id)
            and (self.class_name is None or card.class_name == self.class_name)
            and (self.cost_min is None or card.cost >= self.cost_min)
            and (self.cost_max is None or card.cost <= self.cost_max)
            and (self.card_id is None or card.card_id == self.card_id)
            and (self.card_name is None or card.name == self.card_name)
        )


@dataclass(frozen=True)
class BoardFilter:
    card_type: str | None = None
    cost_min: int | None = None
    cost_max: int | None = None
    card_id: int | None = None
    card_name: str | None = None
    evolved: bool | None = None

    def matches(self, card: CardDefinition) -> bool:
        return (
            (self.card_type is None or card.card_type == self.card_type)
            and (self.cost_min is None or card.cost >= self.cost_min)
            and (self.cost_max is None or card.cost <= self.cost_max)
            and (self.card_id is None or card.card_id == self.card_id)
            and (self.card_name is None or card.name == self.card_name)
        )

    def matches_entity(self, entity: Any) -> bool:
        definition = getattr(entity, "definition", None)
        if definition is None or not self.matches(definition):
            return False
        if self.evolved is not None and getattr(entity, "evolved", False) is not self.evolved:
            return False
        return True


@dataclass(frozen=True)
class EffectOperation:
    kind: EffectKind
    target: TargetKind
    amount: int = 0
    secondary_amount: int = 0
    card_id: int | None = None
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
    necromancy_operations: tuple["EffectOperation", ...] = ()
    graveyard_cost_max: int | None = None
    graveyard_cost_min: int | None = None
    graveyard_follower_only: bool = False
    graveyard_card_type: str | None = None
    deck_filter: DeckFilter | None = None
    board_filter: BoardFilter | None = None
    then_operations: tuple["EffectOperation", ...] = ()
    else_operations: tuple["EffectOperation", ...] = ()
    choose_one_options: tuple["ChooseOneOption", ...] = ()
    optional_prompt: str | None = None
    optional_operations: tuple["EffectOperation", ...] = ()
    emblem_remove_mode: str = "first"
    requires_target: bool = False
    target_count: int = 1
    target_count_expr: ValueExpression | None = None
    allow_duplicate_targets: bool = False


@dataclass
class EffectFrame:
    controller: int
    source_card_id: int
    source_name: str
    source_entity_id: int | None
    source_card: CardDefinition
    operations: tuple[EffectOperation, ...]
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
    _target_bindings: dict[str, int] = field(default_factory=dict)
    _target_binding_operations: dict[str, EffectOperation] = field(default_factory=dict)
    _decision_meta: dict[str, Any] = field(default_factory=dict)
    emblem_batch_id: int | None = None
    emblem_activation_owner: int | None = None
    emblem_activation_entity_id: int | None = None
    emblem_activation_trigger_index: int | None = None
    emblem_expiration_batch_id: int | None = None
    expiring_emblem_owner: int | None = None
    expiring_emblem_entity_id: int | None = None
