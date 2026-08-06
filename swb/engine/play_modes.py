from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from swb.engine.effects import ConditionType

if TYPE_CHECKING:
    from swb.engine.effects import Condition, EffectOperation

NORMAL_MODE_ID = "normal"
_IMPLEMENTED_MODE_TYPES = frozenset({"enhance", "accelerate", "crystallize"})
_VALID_RESULTING_CARD_TYPES = frozenset({"法术", "护符"})
MAX_SPECIAL_MODES_PER_CARD = 3

_KNOWN_KEYS_ENHANCE = frozenset({
    "id",
    "type",
    "cost",
    "operations",
    "conditions",
    "replace_base_operations",
})
_KNOWN_KEYS_ACCELERATE = frozenset({"id", "type", "cost", "resulting_card_type", "operations", "conditions"})
_KNOWN_KEYS_CRYSTALLIZE = frozenset({"id", "type", "cost", "resulting_card_type", "countdown", "operations", "conditions"})
_TARGET_DEPENDENT_CONDITIONS = frozenset({
    ConditionType.TARGET_ATTACK_AT_MOST,
    ConditionType.TARGET_ATTACK_AT_LEAST,
    ConditionType.TARGET_HEALTH_AT_MOST,
    ConditionType.TARGET_HEALTH_AT_LEAST,
    ConditionType.TARGET_HAS_KEYWORD,
})


def _check_unknown_keys(raw: dict, allowed: frozenset[str], source: str) -> None:
    for key in raw:
        if key not in allowed:
            raise ValueError(
                f"{source}: unknown key {key!r}; allowed: {sorted(allowed)}"
            )


def _validate_cost(raw: dict, mode_id: str, error_prefix: str) -> int:
    cost = raw.get("cost")
    if cost is None:
        raise ValueError(f"{error_prefix}/play_modes/{mode_id}: 'cost' is required")
    if isinstance(cost, bool):
        raise ValueError(f"{error_prefix}/play_modes/{mode_id}/cost: must be an integer, got bool")
    if not isinstance(cost, int):
        raise ValueError(f"{error_prefix}/play_modes/{mode_id}/cost: must be an integer, got {type(cost).__name__}")
    if cost < 0:
        raise ValueError(f"{error_prefix}/play_modes/{mode_id}/cost: must be non-negative, got {cost}")
    return cost


def _validate_countdown(raw: dict, mode_id: str, error_prefix: str) -> int | None:
    if "countdown" not in raw:
        return None
    c = raw["countdown"]
    if c is None:
        return None
    if isinstance(c, bool):
        raise ValueError(f"{error_prefix}/play_modes/{mode_id}/countdown: must be an integer, got bool")
    if isinstance(c, str):
        raise ValueError(f"{error_prefix}/play_modes/{mode_id}/countdown: must be an integer, got string {c!r}")
    if isinstance(c, float):
        raise ValueError(f"{error_prefix}/play_modes/{mode_id}/countdown: must be an integer, got float")
    if not isinstance(c, int) or c < 0:
        raise ValueError(f"{error_prefix}/play_modes/{mode_id}/countdown: must be a non-negative integer, got {c!r}")
    return c


def _target_dependent_condition_names(
    conditions: tuple["Condition", ...],
) -> set[str]:
    invalid: set[str] = set()
    for condition in conditions:
        if condition.type in _TARGET_DEPENDENT_CONDITIONS:
            invalid.add(condition.type.value)
        invalid.update(_target_dependent_condition_names(condition.conditions))
    return invalid


def validate_play_mode_definition(
    raw: dict,
    source_file: str,
    card_id: int,
    operations_parser,
) -> PlayModeDefinition:
    error_prefix = f"{source_file} card {card_id}"

    mode_id = raw.get("id")
    if not isinstance(mode_id, str) or not mode_id or mode_id.strip() == "":
        raise ValueError(
            f"{error_prefix}/play_modes: 'id' must be a non-empty string, got {mode_id!r}"
        )
    if mode_id == "normal":
        raise ValueError(
            f"{error_prefix}/play_modes/{mode_id}: 'normal' mode must not appear in play_modes; "
            f"use the top-level 'operations' and card_type instead"
        )

    mode_type = raw.get("type")
    if mode_type == "choose":
        raise ValueError(
            f"{error_prefix}/play_modes/{mode_id}: 'choose' play mode is not yet implemented"
        )
    if mode_type not in _IMPLEMENTED_MODE_TYPES:
        raise ValueError(
            f"{error_prefix}/play_modes/{mode_id}: 'type' must be one of "
            f"{sorted(_IMPLEMENTED_MODE_TYPES)}, got {mode_type!r}"
        )

    cost = _validate_cost(raw, mode_id, error_prefix)

    replace_base_operations = raw.get("replace_base_operations", False)
    if not isinstance(replace_base_operations, bool):
        raise ValueError(
            f"{error_prefix}/play_modes/{mode_id}/replace_base_operations: "
            "must be boolean"
        )
    if replace_base_operations and mode_type != "enhance":
        raise ValueError(
            f"{error_prefix}/play_modes/{mode_id}/replace_base_operations: "
            "is only valid for enhance"
        )

    resulting_card_type = raw.get("resulting_card_type")
    if mode_type == "enhance":
        _check_unknown_keys(raw, _KNOWN_KEYS_ENHANCE, f"{error_prefix}/play_modes/{mode_id}")
        if resulting_card_type is not None:
            raise ValueError(
                f"{error_prefix}/play_modes/{mode_id}/resulting_card_type: "
                f"enhance does not support resulting_card_type"
            )
        if "countdown" in raw:
            raise ValueError(
                f"{error_prefix}/play_modes/{mode_id}/countdown: enhance does not support countdown"
            )
    elif mode_type == "accelerate":
        _check_unknown_keys(raw, _KNOWN_KEYS_ACCELERATE, f"{error_prefix}/play_modes/{mode_id}")
        if resulting_card_type is not None and resulting_card_type != "法术":
            raise ValueError(
                f"{error_prefix}/play_modes/{mode_id}/resulting_card_type: "
                f"accelerate requires '法术', got {resulting_card_type!r}"
            )
        if resulting_card_type is None:
            resulting_card_type = "法术"
        if "countdown" in raw:
            raise ValueError(
                f"{error_prefix}/play_modes/{mode_id}/countdown: accelerate does not support countdown"
            )
    elif mode_type == "crystallize":
        _check_unknown_keys(raw, _KNOWN_KEYS_CRYSTALLIZE, f"{error_prefix}/play_modes/{mode_id}")
        if resulting_card_type is not None and resulting_card_type != "护符":
            raise ValueError(
                f"{error_prefix}/play_modes/{mode_id}/resulting_card_type: "
                f"crystallize requires '护符', got {resulting_card_type!r}"
            )
        if resulting_card_type is None:
            resulting_card_type = "护符"
        countdown = _validate_countdown(raw, mode_id, error_prefix)
    else:
        raise ValueError(f"{error_prefix}/play_modes/{mode_id}: unknown mode type {mode_type!r}")

    raw_ops = raw.get("operations", [])
    if not isinstance(raw_ops, list):
        raise ValueError(
            f"{error_prefix}/play_modes/{mode_id}: 'operations' must be a list"
        )
    operations = tuple(
        operations_parser(
            op,
            f"{source_file}/play_modes/{mode_id}/operations[{index}]",
            card_id,
        )
        for index, op in enumerate(raw_ops)
    )

    countdown_val = _validate_countdown(raw, mode_id, error_prefix) if mode_type == "crystallize" else None

    conditions: tuple = ()
    raw_conds = raw.get("conditions")
    if raw_conds is not None:
        if not isinstance(raw_conds, list):
            raise ValueError(
                f"{error_prefix}/play_modes/{mode_id}: 'conditions' must be a list"
            )
        from swb.engine.card_rules import _parse_condition
        conditions = tuple(
            _parse_condition(c, f"{source_file}/play_modes/{mode_id}/conditions[{i}]", card_id)
            for i, c in enumerate(raw_conds)
        )

    mode = PlayModeDefinition(
        mode_id=mode_id,
        mode_type=mode_type,
        cost=cost,
        resulting_card_type=resulting_card_type,
        operations=operations,
        countdown=countdown_val,
        conditions=conditions,
        replace_base_operations=replace_base_operations,
    )
    validate_runtime_play_mode(mode, f"{error_prefix}/play_modes/{mode_id}")
    return mode


@dataclass(frozen=True)
class PlayModeDefinition:
    mode_id: str
    mode_type: str
    cost: int
    resulting_card_type: str | None = None
    operations: tuple["EffectOperation", ...] = ()
    countdown: int | None = None
    conditions: tuple["Condition", ...] = ()
    replace_base_operations: bool = False

    @property
    def is_normal(self) -> bool:
        return self.mode_type == "normal"

    @property
    def is_enhance(self) -> bool:
        return self.mode_type == "enhance"

    @property
    def is_accelerate(self) -> bool:
        return self.mode_type == "accelerate"

    @property
    def is_crystallize(self) -> bool:
        return self.mode_type == "crystallize"


def validate_runtime_play_mode(
    mode: PlayModeDefinition,
    source: str,
) -> None:
    if not isinstance(mode.mode_id, str) or not mode.mode_id.strip():
        raise ValueError(f"{source}/id: must be a non-empty string")
    if mode.mode_id == NORMAL_MODE_ID:
        raise ValueError(f"{source}/id: {NORMAL_MODE_ID!r} is reserved")
    if mode.mode_type == "choose":
        raise ValueError(f"{source}/type: 'choose' play mode is not yet implemented")
    if mode.mode_type not in _IMPLEMENTED_MODE_TYPES:
        raise ValueError(
            f"{source}/type: must be one of {sorted(_IMPLEMENTED_MODE_TYPES)}, "
            f"got {mode.mode_type!r}"
        )
    if isinstance(mode.cost, bool) or not isinstance(mode.cost, int):
        raise ValueError(f"{source}/cost: must be an integer")
    if mode.cost < 0:
        raise ValueError(f"{source}/cost: must be non-negative")
    if not isinstance(mode.replace_base_operations, bool):
        raise ValueError(f"{source}/replace_base_operations: must be boolean")
    if mode.replace_base_operations and mode.mode_type != "enhance":
        raise ValueError(
            f"{source}/replace_base_operations: is only valid for enhance"
        )

    if mode.mode_type == "enhance":
        if mode.resulting_card_type is not None:
            raise ValueError(
                f"{source}/resulting_card_type: enhance does not support it"
            )
        if mode.countdown is not None:
            raise ValueError(f"{source}/countdown: enhance does not support it")
    elif mode.mode_type == "accelerate":
        if mode.resulting_card_type not in {None, "法术"}:
            raise ValueError(
                f"{source}/resulting_card_type: accelerate requires '法术'"
            )
        if mode.countdown is not None:
            raise ValueError(f"{source}/countdown: accelerate does not support it")
    else:
        if mode.resulting_card_type not in {None, "护符"}:
            raise ValueError(
                f"{source}/resulting_card_type: crystallize requires '护符'"
            )
        if (
            mode.countdown is not None
            and (
                isinstance(mode.countdown, bool)
                or not isinstance(mode.countdown, int)
                or mode.countdown < 0
            )
        ):
            raise ValueError(
                f"{source}/countdown: must be a non-negative integer"
            )

    invalid_conditions = _target_dependent_condition_names(mode.conditions)
    if invalid_conditions:
        raise ValueError(
            f"{source}/conditions: play-mode conditions cannot depend on a "
            f"target; move {sorted(invalid_conditions)} to an operation"
        )
