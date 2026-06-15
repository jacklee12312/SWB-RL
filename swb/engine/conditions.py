from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from swb.engine.effects import Condition, ConditionType, ExprType, ValueExpression
from swb.engine.state import Unit

if TYPE_CHECKING:
    from swb.engine.state import PlayerState


@dataclass
class EvalContext:
    controller: int
    players: list[PlayerState]
    source_entity_id: int | None = None
    target_entity_id: int | None = None
    source_card_id: int | None = None

    @property
    def controller_player(self) -> PlayerState:
        return self.players[self.controller]

    @property
    def opponent_player(self) -> PlayerState:
        return self.players[1 - self.controller]

    def find_board_entity(self, entity_id: int | None):
        if entity_id is None:
            return None
        for p in self.players:
            for e in p.board:
                if e.entity_id == entity_id:
                    return e
        return None

    @property
    def source_entity(self):
        return self.find_board_entity(self.source_entity_id)

    @property
    def target_entity(self):
        return self.find_board_entity(self.target_entity_id)


_TARGET_DEPENDENT_CONDITIONS = frozenset({
    ConditionType.TARGET_ATTACK_AT_MOST,
    ConditionType.TARGET_ATTACK_AT_LEAST,
    ConditionType.TARGET_HEALTH_AT_MOST,
    ConditionType.TARGET_HEALTH_AT_LEAST,
    ConditionType.TARGET_HAS_KEYWORD,
})


class PartialConditionResult(str, Enum):
    TRUE = "true"
    FALSE = "false"
    DEPENDS_ON_TARGET = "depends_on_target"


def evaluate_conditions_without_target(
    conds: tuple[Condition, ...],
    ctx: EvalContext,
) -> PartialConditionResult:
    results = [_evaluate_without_target(cond, ctx) for cond in conds]
    if any(result is PartialConditionResult.FALSE for result in results):
        return PartialConditionResult.FALSE
    if any(result is PartialConditionResult.DEPENDS_ON_TARGET for result in results):
        return PartialConditionResult.DEPENDS_ON_TARGET
    return PartialConditionResult.TRUE


def _evaluate_without_target(
    cond: Condition,
    ctx: EvalContext,
) -> PartialConditionResult:
    if cond.type in _TARGET_DEPENDENT_CONDITIONS:
        return PartialConditionResult.DEPENDS_ON_TARGET

    if cond.type == ConditionType.ALL:
        results = [_evaluate_without_target(child, ctx) for child in cond.conditions]
        if any(result is PartialConditionResult.FALSE for result in results):
            return PartialConditionResult.FALSE
        if any(
            result is PartialConditionResult.DEPENDS_ON_TARGET
            for result in results
        ):
            return PartialConditionResult.DEPENDS_ON_TARGET
        return PartialConditionResult.TRUE

    if cond.type == ConditionType.ANY:
        results = [_evaluate_without_target(child, ctx) for child in cond.conditions]
        if any(result is PartialConditionResult.TRUE for result in results):
            return PartialConditionResult.TRUE
        if any(
            result is PartialConditionResult.DEPENDS_ON_TARGET
            for result in results
        ):
            return PartialConditionResult.DEPENDS_ON_TARGET
        return PartialConditionResult.FALSE

    if cond.type == ConditionType.NOT:
        result = _evaluate_without_target(cond.conditions[0], ctx)
        if result is PartialConditionResult.TRUE:
            return PartialConditionResult.FALSE
        if result is PartialConditionResult.FALSE:
            return PartialConditionResult.TRUE
        return PartialConditionResult.DEPENDS_ON_TARGET

    return (
        PartialConditionResult.TRUE
        if evaluate_condition(cond, ctx)
        else PartialConditionResult.FALSE
    )


def evaluate_target_conditions(
    conds: tuple,
    entity,
    controller: int,
    players,
    source_entity_id: int | None = None,
) -> bool:
    ctx = EvalContext(
        controller=controller,
        players=players,
        target_entity_id=entity.entity_id,
        source_entity_id=source_entity_id,
    )
    for cond in conds:
        if not evaluate_condition(cond, ctx):
            return False
    return True


def evaluate_condition(cond: Condition | None, ctx: EvalContext | None) -> bool:
    if cond is None:
        return True
    t = cond.type
    if t == ConditionType.ALWAYS:
        return True
    elif t == ConditionType.ALL:
        return all(evaluate_condition(c, ctx) for c in cond.conditions)
    elif t == ConditionType.ANY:
        return any(evaluate_condition(c, ctx) for c in cond.conditions)
    elif t == ConditionType.NOT:
        return not evaluate_condition(cond.conditions[0], ctx) if cond.conditions else True

    if ctx is None:
        return False
    player = ctx.controller_player
    opponent = ctx.opponent_player
    target = ctx.target_entity
    source = ctx.source_entity

    if t == ConditionType.CONTROLLER_HEALTH_AT_MOST:
        return player.health <= cond.value
    elif t == ConditionType.CONTROLLER_HEALTH_AT_LEAST:
        return player.health >= cond.value
    elif t == ConditionType.OPPONENT_HEALTH_AT_MOST:
        return opponent.health <= cond.value
    elif t == ConditionType.OPPONENT_HEALTH_AT_LEAST:
        return opponent.health >= cond.value
    elif t == ConditionType.CONTROLLER_BOARD_COUNT_AT_LEAST:
        return len(player.board) >= cond.value
    elif t == ConditionType.OPPONENT_BOARD_COUNT_AT_LEAST:
        return len(opponent.board) >= cond.value
    elif t == ConditionType.CONTROLLER_HAND_COUNT_AT_LEAST:
        return len(player.hand) >= cond.value
    elif t == ConditionType.CONTROLLER_SHADOWS_AT_LEAST:
        return player.shadows >= cond.value
    elif t == ConditionType.OPPONENT_SHADOWS_AT_LEAST:
        return opponent.shadows >= cond.value
    elif t == ConditionType.CONTROLLER_COOPERATION_AT_LEAST:
        return player.cooperation >= cond.value
    elif t == ConditionType.OPPONENT_COOPERATION_AT_LEAST:
        return opponent.cooperation >= cond.value
    elif t == ConditionType.TARGET_ATTACK_AT_MOST:
        return isinstance(target, Unit) and target.attack <= cond.value
    elif t == ConditionType.TARGET_ATTACK_AT_LEAST:
        return isinstance(target, Unit) and target.attack >= cond.value
    elif t == ConditionType.TARGET_HEALTH_AT_MOST:
        return isinstance(target, Unit) and target.health <= cond.value
    elif t == ConditionType.TARGET_HEALTH_AT_LEAST:
        return isinstance(target, Unit) and target.health >= cond.value
    elif t == ConditionType.SOURCE_EVOLVED:
        return isinstance(source, Unit) and source.evolved
    elif t == ConditionType.SOURCE_HAS_KEYWORD:
        return isinstance(source, Unit) and source.has_keyword(cond.keyword or "")
    elif t == ConditionType.TARGET_HAS_KEYWORD:
        return isinstance(target, Unit) and target.has_keyword(cond.keyword or "")

    raise ValueError(f"Unknown condition type: {t}")


def evaluate_expression(expr: ValueExpression | None, ctx: EvalContext | None) -> int:
    if expr is None:
        return 0
    t = expr.type
    if t == ExprType.CONSTANT:
        return expr.value
    elif t == ExprType.ADD:
        return sum(evaluate_expression(v, ctx) for v in expr.values)
    elif t == ExprType.SUBTRACT:
        vals = [evaluate_expression(v, ctx) for v in expr.values]
        if not vals:
            return 0
        result = vals[0]
        for v in vals[1:]:
            result -= v
        return max(0, result)
    elif t == ExprType.MULTIPLY:
        result = 1
        for v in expr.values:
            result *= evaluate_expression(v, ctx)
        return result
    elif t == ExprType.MIN:
        vals = [evaluate_expression(v, ctx) for v in expr.values]
        return min(vals) if vals else 0
    elif t == ExprType.MAX:
        vals = [evaluate_expression(v, ctx) for v in expr.values]
        return max(vals) if vals else 0

    player = ctx.controller_player if ctx else None
    opponent = ctx.opponent_player if ctx else None
    target = ctx.target_entity if ctx else None
    source = ctx.source_entity if ctx else None

    if t == ExprType.CONTROLLER_BOARD_COUNT:
        return len(player.board) if player else 0
    elif t == ExprType.OPPONENT_BOARD_COUNT:
        return len(opponent.board) if opponent else 0
    elif t == ExprType.CONTROLLER_HAND_COUNT:
        return len(player.hand) if player else 0
    elif t == ExprType.SOURCE_ATTACK:
        return source.attack if isinstance(source, Unit) else 0
    elif t == ExprType.SOURCE_HEALTH:
        return source.health if isinstance(source, Unit) else 0
    elif t == ExprType.TARGET_ATTACK:
        return target.attack if isinstance(target, Unit) else 0
    elif t == ExprType.TARGET_HEALTH:
        return target.health if isinstance(target, Unit) else 0
    elif t == ExprType.CONTROLLER_SHADOWS:
        return player.shadows if player else 0
    elif t == ExprType.OPPONENT_SHADOWS:
        return opponent.shadows if opponent else 0
    elif t == ExprType.CONTROLLER_COOPERATION:
        return player.cooperation if player else 0
    elif t == ExprType.OPPONENT_COOPERATION:
        return opponent.cooperation if opponent else 0

    raise ValueError(f"Unknown expression type: {t}")
