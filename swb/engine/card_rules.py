from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from swb.engine.abilities import RUNTIME_UNIT_KEYWORDS, normalize_keyword_name
from swb.engine.effects import (
    Condition,
    ConditionType,
    CostChangeMode,
    EffectKind,
    EffectOperation,
    ExprType,
    ModifierDuration,
    TargetKind,
    ValueExpression,
)


class Trigger(str, Enum):
    PLAY = "play"
    FANFARE = "fanfare"
    LAST_WORDS = "last_words"
    EVOLVE = "evolve"
    SUPER_EVOLVE = "super_evolve"
    ATTACK = "attack"
    CLASH = "clash"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    COUNTDOWN_EXPIRED = "countdown_expired"


@dataclass(frozen=True)
class CardRule:
    card_id: int
    trigger: Trigger
    operations: tuple[EffectOperation, ...]
    countdown: int | None = None


class RuleBook:
    def __init__(self, rules: tuple[CardRule, ...] = ()):
        self._rules: dict[tuple[int, Trigger], tuple[EffectOperation, ...]] = {
            (rule.card_id, rule.trigger): rule.operations for rule in rules
        }
        self._countdowns = {
            rule.card_id: rule.countdown
            for rule in rules
            if rule.countdown is not None
        }

    def operations_for(
        self, card_id: int, trigger: Trigger
    ) -> tuple[EffectOperation, ...]:
        return self._rules.get((card_id, trigger), ())

    def countdown_for(self, card_id: int) -> int | None:
        return self._countdowns.get(card_id)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "RuleBook":
        path = Path(directory)
        if not path.exists():
            return cls()
        rules: list[CardRule] = []
        for file_path in sorted(path.glob("*.json")):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            entries = payload if isinstance(payload, list) else payload["rules"]
            for entry in entries:
                operations = tuple(
                    _parse_operation(
                        operation,
                        f"{file_path.name}/operations[{index}]",
                        entry["card_id"],
                    )
                    for index, operation in enumerate(
                        entry.get("operations", [])
                    )
                )
                _validate_target_keys(operations, f"{file_path.name} card {entry['card_id']}")
                rules.append(
                    CardRule(
                        card_id=int(entry["card_id"]),
                        trigger=Trigger(entry["trigger"]),
                        operations=operations,
                        countdown=entry.get("countdown"),
                    )
                )
        return cls(tuple(rules))


_BINDABLE_TARGETS = frozenset({
    TargetKind.OWN_UNIT,
    TargetKind.ENEMY_UNIT,
    TargetKind.ANY_UNIT,
    TargetKind.OWN_AMULET,
    TargetKind.ENEMY_AMULET,
    TargetKind.ANY_AMULET,
    TargetKind.OWN_BOARD,
    TargetKind.ENEMY_BOARD,
    TargetKind.ANY_BOARD,
    TargetKind.RANDOM_OWN_UNIT,
    TargetKind.RANDOM_ENEMY_UNIT,
    TargetKind.RANDOM_OWN_BOARD,
    TargetKind.RANDOM_ENEMY_BOARD,
})


def _validate_target_keys(operations: tuple[EffectOperation, ...], source: str) -> None:
    defined: set[str] = set()
    for i, op in enumerate(operations):
        if op.target is TargetKind.PREVIOUS_TARGET:
            if not op.target_key:
                raise ValueError(
                    f"{source}/operations[{i}]: PREVIOUS_TARGET requires target_key"
                )
            if op.target_key not in defined:
                raise ValueError(
                    f"{source}/operations[{i}]: target_key "
                    f"{op.target_key!r} was not defined by a previous operation"
                )
        elif op.target_key:
            if op.target not in _BINDABLE_TARGETS:
                raise ValueError(
                    f"{source}/operations[{i}]: target {op.target.value!r} "
                    f"cannot define target_key {op.target_key!r}; "
                    f"target_key requires a single board-entity target"
                )
            if op.target_key in defined:
                raise ValueError(
                    f"{source}/operations[{i}]: duplicate target_key "
                    f"{op.target_key!r}"
                )
            defined.add(op.target_key)


def _parse_operation(raw: dict, source_file: str, card_id: int) -> EffectOperation:
    error_prefix = f"{source_file} card {card_id}"
    if not isinstance(raw, dict):
        raise ValueError(
            f"{error_prefix}: operation must be an object, "
            f"got {type(raw).__name__}"
        )
    try:
        kind = EffectKind(raw["kind"])
        target = TargetKind(raw["target"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"{error_prefix}: invalid kind/target: {e}") from e

    conditions = ()
    raw_conds = raw.get("conditions")
    if raw_conds is not None:
        if not isinstance(raw_conds, list):
            raise ValueError(
                f"{error_prefix}: 'conditions' must be a list, "
                f"got {type(raw_conds).__name__}"
            )
        conditions = tuple(
            _parse_condition(c, f"{source_file}/conditions[{i}]", card_id)
            for i, c in enumerate(raw_conds)
        )

    amount_expr = None
    raw_amount = raw.get("amount")
    if kind is EffectKind.SET_STATS and raw_amount is None:
        raw_amount = raw.get("attack")
    if isinstance(raw_amount, dict):
        amount_expr = _parse_expression(raw_amount, f"{source_file}/amount", card_id)

    secondary_expr = None
    raw_secondary = raw.get("secondary_amount")
    if kind is EffectKind.SET_STATS and raw_secondary is None:
        raw_secondary = raw.get("health")
    if isinstance(raw_secondary, dict):
        secondary_expr = _parse_expression(raw_secondary, f"{source_file}/secondary_amount", card_id)

    if kind is EffectKind.SET_STATS:
        if raw_amount is None and raw_secondary is None:
            raise ValueError(
                f"{source_file} card {card_id}: "
                f"SET_STATS requires at least one of 'attack'/'amount' or 'health'/'secondary_amount'"
            )

    keyword = raw.get("keyword")
    if kind in (EffectKind.ADD_KEYWORD, EffectKind.REMOVE_KEYWORD):
        if not isinstance(keyword, str) or not keyword:
            raise ValueError(
                f"{source_file}/keyword card {card_id}: "
                f"'{kind.value}' requires a keyword"
            )
        try:
            keyword = normalize_keyword_name(keyword, strict=True)
        except ValueError as exc:
            raise ValueError(
                f"{source_file}/keyword card {card_id}: {exc}"
            ) from exc
        if keyword not in RUNTIME_UNIT_KEYWORDS:
            raise ValueError(
                f"{source_file}/keyword card {card_id}: keyword "
                f"{keyword!r} is not a supported runtime unit keyword"
            )

    restriction = raw.get("restriction")
    if kind in (
        EffectKind.ADD_ATTACK_RESTRICTION,
        EffectKind.REMOVE_ATTACK_RESTRICTION,
        EffectKind.ADD_TARGETING_RESTRICTION,
        EffectKind.REMOVE_TARGETING_RESTRICTION,
    ):
        if not isinstance(restriction, str) or not restriction:
            raise ValueError(
                f"{source_file}/restriction card {card_id}: "
                f"'{kind.value}' requires a non-empty restriction"
            )
        from swb.engine.state import AttackRestriction, TargetingRestriction
        valid_attack = set(r.value for r in AttackRestriction)
        valid_targeting = set(r.value for r in TargetingRestriction)
        if kind in (EffectKind.ADD_ATTACK_RESTRICTION, EffectKind.REMOVE_ATTACK_RESTRICTION):
            if restriction not in valid_attack:
                raise ValueError(
                    f"{source_file}/restriction card {card_id}: "
                    f"unknown attack restriction {restriction!r}"
                )
        else:
            if restriction not in valid_targeting:
                raise ValueError(
                    f"{source_file}/restriction card {card_id}: "
                    f"unknown targeting restriction {restriction!r}"
                )

    mode = None
    if kind is EffectKind.CHANGE_COST:
        raw_mode = raw.get("mode", CostChangeMode.ADD.value)
        try:
            mode = CostChangeMode(raw_mode)
        except ValueError as exc:
            raise ValueError(
                f"{source_file}/mode card {card_id}: invalid cost mode "
                f"{raw_mode!r}"
            ) from exc

    raw_duration = raw.get("duration", ModifierDuration.PERMANENT.value)
    try:
        duration = ModifierDuration(raw_duration)
    except ValueError as exc:
        raise ValueError(
            f"{source_file}/duration card {card_id}: invalid duration "
            f"{raw_duration!r}"
        ) from exc

    operation_card_id = raw.get("card_id")
    if kind in (EffectKind.SUMMON, EffectKind.ADD_CARD, EffectKind.TRANSFORM):
        if operation_card_id is None:
            raise ValueError(
                f"{source_file}/card_id card {card_id}: "
                f"'{kind.value}' requires card_id"
            )
        try:
            operation_card_id = int(operation_card_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_file}/card_id card {card_id}: expected an integer"
            ) from exc

    target_key = raw.get("target_key")
    if target_key is not None and not isinstance(target_key, str):
        raise ValueError(
            f"{source_file}/target_key card {card_id}: must be a string"
        )
    if target is TargetKind.PREVIOUS_TARGET:
        if not target_key:
            raise ValueError(
                f"{source_file} card {card_id}: "
                f"PREVIOUS_TARGET requires a non-empty target_key"
            )

    return EffectOperation(
        kind=kind,
        target=target,
        amount=_parse_optional_int(raw_amount, f"{source_file}/amount", card_id),
        secondary_amount=_parse_optional_int(
            raw_secondary,
            f"{source_file}/secondary_amount",
            card_id,
        ),
        card_id=operation_card_id,
        keyword=keyword,
        restriction=restriction,
        conditions=conditions,
        amount_expr=amount_expr,
        secondary_expr=secondary_expr,
        mode=mode,
        duration=duration,
        set_attack=(
            kind is EffectKind.SET_STATS
            and (raw.get("amount") is not None or raw.get("attack") is not None or amount_expr is not None)
        ),
        set_health=(
            kind is EffectKind.SET_STATS
            and (raw.get("secondary_amount") is not None or raw.get("health") is not None or secondary_expr is not None)
        ),
        target_key=target_key,
    )


def _parse_condition(raw: dict, source_path: str, card_id: int) -> Condition:
    error_prefix = f"{source_path} card {card_id}"
    if not isinstance(raw, dict):
        raise ValueError(
            f"{error_prefix}: condition must be an object, "
            f"got {type(raw).__name__}"
        )
    try:
        t = ConditionType(raw["type"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"{error_prefix}: invalid condition type: {e}") from e

    sub_raws = raw.get("conditions", [])
    if not isinstance(sub_raws, list):
        raise ValueError(f"{error_prefix}: 'conditions' must be a list")
    sub = [
        _parse_condition(c, f"{source_path}/conditions[{i}]", card_id)
        for i, c in enumerate(sub_raws)
    ]

    if t in (ConditionType.ALL, ConditionType.ANY) and not sub:
        raise ValueError(
            f"{error_prefix}: '{t.value}' requires at least one sub-condition"
        )
    if t == ConditionType.NOT and len(sub) != 1:
        raise ValueError(
            f"{error_prefix}: 'not' requires exactly one sub-condition"
        )

    keyword = raw.get("keyword")
    if t in (ConditionType.SOURCE_HAS_KEYWORD, ConditionType.TARGET_HAS_KEYWORD):
        if not isinstance(keyword, str) or not keyword:
            raise ValueError(f"{source_path}/keyword card {card_id}: keyword required")
        try:
            keyword = normalize_keyword_name(keyword, strict=True)
        except ValueError as exc:
            raise ValueError(
                f"{source_path}/keyword card {card_id}: {exc}"
            ) from exc

    return Condition(
        type=t,
        value=_parse_optional_int(raw.get("value"), f"{source_path}/value", card_id),
        keyword=keyword,
        conditions=sub,
    )


def _parse_expression(raw: dict, source_path: str, card_id: int) -> ValueExpression:
    error_prefix = f"{source_path} card {card_id}"
    if not isinstance(raw, dict):
        raise ValueError(
            f"{error_prefix}: expression must be an object, "
            f"got {type(raw).__name__}"
        )
    try:
        t = ExprType(raw["type"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"{error_prefix}: invalid expression type: {e}") from e

    sub_raws = raw.get("values", [])
    if not isinstance(sub_raws, list):
        raise ValueError(f"{error_prefix}: 'values' must be a list")
    sub = [
        _parse_expression(v, f"{source_path}/values[{i}]", card_id)
        for i, v in enumerate(sub_raws)
    ]

    if t in (ExprType.ADD, ExprType.SUBTRACT, ExprType.MULTIPLY, ExprType.MIN, ExprType.MAX):
        if len(sub) < 1:
            raise ValueError(
                f"{error_prefix}: '{t.value}' requires at least one value"
            )

    if t == ExprType.CONSTANT:
        if sub:
            raise ValueError(
                f"{error_prefix}: 'constant' must not have 'values'"
            )
    elif t in (ExprType.CONTROLLER_BOARD_COUNT, ExprType.OPPONENT_BOARD_COUNT,
               ExprType.CONTROLLER_HAND_COUNT, ExprType.SOURCE_ATTACK, ExprType.SOURCE_HEALTH,
               ExprType.TARGET_ATTACK, ExprType.TARGET_HEALTH):
        if sub:
            raise ValueError(
                f"{error_prefix}: '{t.value}' must not have 'values'"
            )

    return ValueExpression(
        type=t,
        value=_parse_optional_int(raw.get("value"), f"{source_path}/value", card_id),
        values=sub,
    )


def _parse_optional_int(raw, source_path: str, card_id: int) -> int:
    if raw is None or isinstance(raw, dict):
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_path} card {card_id}: expected an integer, got {raw!r}"
        ) from exc
