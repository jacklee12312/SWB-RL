from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from swb.engine.effects import (
    BoundTargetSnapshot,
    Condition,
    ConditionType,
    ExprType,
    SourceStateSnapshot,
    ValueExpression,
)
from swb.engine.state import Unit

if TYPE_CHECKING:
    from swb.engine.state import (
        DestroyedFollowerRecord,
        FollowerEntryRecord,
        PlayerState,
    )


OVERFLOW_MAX_MANA_THRESHOLD = 7


@dataclass
class EvalContext:
    controller: int
    players: list[PlayerState]
    source_entity_id: int | None = None
    target_entity_id: int | None = None
    attack_target_entity_id: int | None = None
    source_card_id: int | None = None
    source_fusion_count: int = 0
    source_fusion_distinct_name_count: int = 0
    source_spellboost_count: int = 0
    source_cost: int = 0
    distributed_value: int = 0
    listener_activation_count: int = 0
    event_source_entity_id: int | None = None
    event_source_base_cost: int | None = None
    source_snapshot: SourceStateSnapshot | None = None
    target_snapshot: BoundTargetSnapshot | None = None
    bound_target_snapshots: dict[str, tuple[BoundTargetSnapshot, ...]] | None = None
    controller_super_evolution_unlocked: bool = False
    opponent_super_evolution_unlocked: bool = False
    turn: int = 0
    destroyed_followers: tuple[DestroyedFollowerRecord, ...] = ()
    follower_entries: tuple[FollowerEntryRecord, ...] = ()

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
    ConditionType.TARGET_IS_OWN,
    ConditionType.TARGET_CARD_TYPE_IS,
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
    source_fusion_count: int = 0,
    controller_super_evolution_unlocked: bool = False,
    opponent_super_evolution_unlocked: bool = False,
) -> bool:
    ctx = EvalContext(
        controller=controller,
        players=players,
        controller_super_evolution_unlocked=(
            controller_super_evolution_unlocked
        ),
        opponent_super_evolution_unlocked=(
            opponent_super_evolution_unlocked
        ),
        target_entity_id=entity.entity_id,
        source_entity_id=source_entity_id,
        source_fusion_count=source_fusion_count,
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

    if t == ConditionType.TARGET_IS_OWN:
        if ctx.target_snapshot is not None:
            return ctx.target_snapshot.controller == ctx.controller
        return any(
            target in player.board and player_index == ctx.controller
            for player_index, player in enumerate(ctx.players)
        )
    elif t == ConditionType.TARGET_CARD_TYPE_IS:
        if ctx.target_snapshot is not None:
            return ctx.target_snapshot.card_type == cond.card_type
        return (
            target is not None
            and target.definition.card_type == cond.card_type
        )

    if t == ConditionType.CONTROLLER_HEALTH_AT_MOST:
        return player.health <= cond.value
    elif t == ConditionType.CONTROLLER_HEALTH_AT_LEAST:
        return player.health >= cond.value
    elif t == ConditionType.CONTROLLER_HEALTH_GREATER_THAN_OPPONENT:
        return player.health > opponent.health
    elif t == ConditionType.OPPONENT_HEALTH_AT_MOST:
        return opponent.health <= cond.value
    elif t == ConditionType.OPPONENT_HEALTH_AT_LEAST:
        return opponent.health >= cond.value
    elif t == ConditionType.CONTROLLER_BOARD_COUNT_AT_LEAST:
        return len(player.board) >= cond.value
    elif t == ConditionType.OPPONENT_BOARD_COUNT_AT_LEAST:
        return len(opponent.board) >= cond.value
    elif t == ConditionType.CONTROLLER_BOARD_HAS:
        threshold = cond.value if cond.value > 0 else 1
        return _matching_board_count(player.board, cond) >= threshold
    elif t == ConditionType.OPPONENT_BOARD_HAS:
        threshold = cond.value if cond.value > 0 else 1
        return _matching_board_count(opponent.board, cond) >= threshold
    elif t == ConditionType.CONTROLLER_HAND_COUNT_AT_LEAST:
        return sum(
            1
            for card in player.hand
            if cond.card_filter is None
            or cond.card_filter.matches(getattr(card, "definition", card))
        ) >= cond.value
    elif (
        t
        == ConditionType.CONTROLLER_HAND_TOP_BASE_COST_SUM_GREATER_THAN_OPPONENT
    ):
        controller_costs = sorted(
            (
                getattr(card, "definition", card).cost
                for card in player.hand
            ),
            reverse=True,
        )
        opponent_costs = sorted(
            (
                getattr(card, "definition", card).cost
                for card in opponent.hand
            ),
            reverse=True,
        )
        return (
            sum(controller_costs[: cond.value])
            > sum(opponent_costs[: cond.value])
        )
    elif (
        t
        == ConditionType.CONTROLLER_HAND_SAME_CURRENT_COST_COUNT_AT_LEAST
    ):
        cost_counts: dict[int, int] = {}
        for card in player.hand:
            definition = getattr(card, "definition", card)
            current_cost = getattr(card, "current_cost", definition.cost)
            cost_counts[current_cost] = cost_counts.get(current_cost, 0) + 1
        return max(cost_counts.values(), default=0) >= cond.value
    elif t == ConditionType.CONTROLLER_MAX_MANA_AT_LEAST:
        return player.max_mana >= cond.value
    elif t == ConditionType.OPPONENT_MAX_MANA_AT_LEAST:
        return opponent.max_mana >= cond.value
    elif t == ConditionType.CONTROLLER_DECK_HAS_NO_DUPLICATES:
        card_ids = [card.card_id for card in player.deck]
        return len(card_ids) == len(set(card_ids))
    elif t == ConditionType.CONTROLLER_SHADOWS_AT_LEAST:
        return player.shadows >= cond.value
    elif t == ConditionType.OPPONENT_SHADOWS_AT_LEAST:
        return opponent.shadows >= cond.value
    elif t == ConditionType.CONTROLLER_COOPERATION_AT_LEAST:
        return player.cooperation >= cond.value
    elif t == ConditionType.OPPONENT_COOPERATION_AT_LEAST:
        return opponent.cooperation >= cond.value
    elif t == ConditionType.CONTROLLER_OVERFLOW:
        return player.max_mana >= OVERFLOW_MAX_MANA_THRESHOLD
    elif t == ConditionType.OPPONENT_OVERFLOW:
        return opponent.max_mana >= OVERFLOW_MAX_MANA_THRESHOLD
    elif t == ConditionType.CONTROLLER_COMBO_AT_LEAST:
        return player.cards_played_this_turn >= cond.value
    elif t == ConditionType.OPPONENT_COMBO_AT_LEAST:
        return opponent.cards_played_this_turn >= cond.value
    elif t == ConditionType.CONTROLLER_FOLLOWER_ATTACKS_THIS_TURN_AT_MOST:
        return player.follower_attacks_this_turn <= cond.value
    elif t == ConditionType.CONTROLLER_EARTH_SIGILS_AT_LEAST:
        return player.earth_sigils >= cond.value
    elif t == ConditionType.OPPONENT_EARTH_SIGILS_AT_LEAST:
        return opponent.earth_sigils >= cond.value
    elif t == ConditionType.CONTROLLER_EVOLUTIONS_THIS_MATCH_AT_LEAST:
        return player.followers_evolved_this_match >= cond.value
    elif t == ConditionType.OPPONENT_EVOLUTIONS_THIS_MATCH_AT_LEAST:
        return opponent.followers_evolved_this_match >= cond.value
    elif t == ConditionType.CONTROLLER_SUPER_EVOLUTION_UNLOCKED:
        return ctx.controller_super_evolution_unlocked
    elif t == ConditionType.OPPONENT_SUPER_EVOLUTION_UNLOCKED:
        return ctx.opponent_super_evolution_unlocked
    elif t == ConditionType.SOURCE_FUSION_COUNT_AT_LEAST:
        return ctx.source_fusion_count >= cond.value
    elif t == ConditionType.SOURCE_SPELLBOOST_COUNT_AT_LEAST:
        return ctx.source_spellboost_count >= cond.value
    elif t == ConditionType.SOURCE_COST_EQUALS:
        return ctx.source_cost == cond.value
    elif t == ConditionType.ATTACK_TARGET_EXISTS:
        return isinstance(
            ctx.find_board_entity(ctx.attack_target_entity_id),
            Unit,
        )
    elif t == ConditionType.CONTROLLER_ENTERED_FOLLOWER_DISTINCT_COUNT_AT_LEAST:
        return (
            _distinct_entered_follower_count(
                ctx,
                card_filter=cond.card_filter,
            )
            >= cond.value
        )
    elif t == ConditionType.CONTROLLER_ENTERED_FOLLOWER_COUNT_AT_LEAST:
        return (
            _entered_follower_count(
                ctx,
                card_filter=cond.card_filter,
            )
            >= cond.value
        )
    elif t == ConditionType.BOARD_HAS_OTHER_CARD_WITH_EVENT_SOURCE_BASE_COST:
        return (
            ctx.event_source_base_cost is not None
            and any(
                entity.entity_id != ctx.event_source_entity_id
                and entity.definition.cost == ctx.event_source_base_cost
                for board_owner in ctx.players
                for entity in board_owner.board
            )
        )
    elif t == ConditionType.LISTENER_ACTIVATION_COUNT_EQUALS:
        return ctx.listener_activation_count == cond.value
    elif t == ConditionType.TARGET_ATTACK_AT_MOST:
        return isinstance(target, Unit) and target.attack <= cond.value
    elif t == ConditionType.TARGET_ATTACK_AT_LEAST:
        return isinstance(target, Unit) and target.attack >= cond.value
    elif t == ConditionType.TARGET_HEALTH_AT_MOST:
        return isinstance(target, Unit) and target.health <= cond.value
    elif t == ConditionType.TARGET_HEALTH_AT_LEAST:
        return isinstance(target, Unit) and target.health >= cond.value
    elif t == ConditionType.SOURCE_EVOLVED:
        if isinstance(source, Unit):
            return source.evolved
        return bool(ctx.source_snapshot and ctx.source_snapshot.evolved)
    elif t == ConditionType.SOURCE_SUPER_EVOLVED:
        if isinstance(source, Unit):
            return source.super_evolved
        return bool(ctx.source_snapshot and ctx.source_snapshot.super_evolved)
    elif t == ConditionType.SOURCE_CARD_TYPE_IS:
        if source is not None:
            return source.definition.card_type == cond.card_type
        return bool(
            ctx.source_snapshot
            and ctx.source_snapshot.card_type == cond.card_type
        )
    elif t in {
        ConditionType.SOURCE_HEALTH_AT_MOST,
        ConditionType.SOURCE_HEALTH_AT_LEAST,
    }:
        source_health = (
            source.health
            if isinstance(source, Unit)
            else (
                ctx.source_snapshot.health
                if ctx.source_snapshot is not None
                else None
            )
        )
        if source_health is None:
            return False
        if t is ConditionType.SOURCE_HEALTH_AT_MOST:
            return source_health <= cond.value
        return source_health >= cond.value
    elif t == ConditionType.SOURCE_HAS_KEYWORD:
        keyword = cond.keyword or ""
        if isinstance(source, Unit):
            return source.has_keyword(keyword)
        return bool(
            ctx.source_snapshot
            and keyword in ctx.source_snapshot.effective_keywords
        )
    elif t == ConditionType.TARGET_HAS_KEYWORD:
        return isinstance(target, Unit) and target.has_keyword(cond.keyword or "")

    raise ValueError(f"Unknown condition type: {t}")


def _matching_board_count(board, cond: Condition) -> int:
    if cond.board_filter is None:
        return len(board)
    return sum(1 for entity in board if cond.board_filter.matches_entity(entity))


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
    elif t == ExprType.NEGATE:
        return -evaluate_expression(expr.values[0], ctx)
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
        return (
            sum(
                1
                for entity in player.board
                if expr.board_filter is None
                or expr.board_filter.matches_entity(entity)
            )
            if player
            else 0
        )
    elif t == ExprType.OPPONENT_BOARD_COUNT:
        return (
            sum(
                1
                for entity in opponent.board
                if expr.board_filter is None
                or expr.board_filter.matches_entity(entity)
            )
            if opponent
            else 0
        )
    elif t == ExprType.CONTROLLER_HAND_COUNT:
        return (
            sum(
                1
                for card in player.hand
                if expr.card_filter is None
                or expr.card_filter.matches(card.definition)
            )
            if player
            else 0
        )
    elif t == ExprType.CONTROLLER_EMBLEM_COUNT:
        return len(player.emblems) if player else 0
    elif t == ExprType.SOURCE_FUSION_DISTINCT_NAME_COUNT:
        return ctx.source_fusion_distinct_name_count if ctx else 0
    elif t == ExprType.SOURCE_SPELLBOOST_COUNT:
        return ctx.source_spellboost_count if ctx else 0
    elif t == ExprType.SOURCE_COST:
        return ctx.source_cost if ctx else 0
    elif t == ExprType.BOUND_CARD_COST:
        snapshots = (
            ()
            if ctx is None or ctx.bound_target_snapshots is None
            else ctx.bound_target_snapshots.get(expr.binding_key or "", ())
        )
        return snapshots[0].cost if len(snapshots) == 1 else 0
    elif t == ExprType.BOUND_CARD_ATTACK:
        snapshots = (
            ()
            if ctx is None or ctx.bound_target_snapshots is None
            else ctx.bound_target_snapshots.get(expr.binding_key or "", ())
        )
        return (
            snapshots[0].attack
            if len(snapshots) == 1 and snapshots[0].attack is not None
            else 0
        )
    elif t == ExprType.BOUND_TARGET_HEALTH:
        snapshots = (
            ()
            if ctx is None or ctx.bound_target_snapshots is None
            else ctx.bound_target_snapshots.get(expr.binding_key or "", ())
        )
        if len(snapshots) != 1 or ctx is None:
            return 0
        target = ctx.find_board_entity(snapshots[0].entity_id)
        return target.health if isinstance(target, Unit) else 0
    elif t == ExprType.BOUND_TARGET_COUNT:
        snapshots = (
            ()
            if ctx is None or ctx.bound_target_snapshots is None
            else ctx.bound_target_snapshots.get(expr.binding_key or "", ())
        )
        return len(snapshots)
    elif t == ExprType.SOURCE_ATTACK:
        if isinstance(source, Unit):
            return source.attack
        return (
            ctx.source_snapshot.attack
            if ctx and ctx.source_snapshot and ctx.source_snapshot.attack is not None
            else 0
        )
    elif t == ExprType.SOURCE_HEALTH:
        if isinstance(source, Unit):
            return source.health
        return (
            ctx.source_snapshot.health
            if ctx and ctx.source_snapshot and ctx.source_snapshot.health is not None
            else 0
        )
    elif t == ExprType.SOURCE_MISSING_HEALTH:
        if isinstance(source, Unit):
            return max(0, source.max_health - source.health)
        return 0
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
    elif t == ExprType.CONTROLLER_OVERFLOW:
        return (
            1
            if player and player.max_mana >= OVERFLOW_MAX_MANA_THRESHOLD
            else 0
        )
    elif t == ExprType.OPPONENT_OVERFLOW:
        return (
            1
            if opponent and opponent.max_mana >= OVERFLOW_MAX_MANA_THRESHOLD
            else 0
        )
    elif t == ExprType.CONTROLLER_COMBO:
        return player.cards_played_this_turn if player else 0
    elif t == ExprType.OPPONENT_COMBO:
        return opponent.cards_played_this_turn if opponent else 0
    elif t == ExprType.CONTROLLER_EARTH_SIGILS:
        return player.earth_sigils if player else 0
    elif t == ExprType.OPPONENT_EARTH_SIGILS:
        return opponent.earth_sigils if opponent else 0
    elif t in {
        ExprType.CONTROLLER_DESTROYED_FOLLOWER_BASE_ATTACK_SUM_THIS_TURN,
        ExprType.CONTROLLER_DESTROYED_FOLLOWER_BASE_HEALTH_SUM_THIS_TURN,
    }:
        if ctx is None:
            return 0
        records = (
            record
            for record in ctx.destroyed_followers
            if record.owner == ctx.controller
            and record.destroyed_turn == ctx.turn
            and (
                expr.card_filter is None
                or expr.card_filter.matches(record.definition)
            )
        )
        if (
            t
            is ExprType.CONTROLLER_DESTROYED_FOLLOWER_BASE_ATTACK_SUM_THIS_TURN
        ):
            return sum(record.definition.attack or 0 for record in records)
        return sum(record.definition.life or 0 for record in records)
    elif t == ExprType.CONTROLLER_ENTERED_FOLLOWER_DISTINCT_COUNT:
        if ctx is None:
            return 0
        return _distinct_entered_follower_count(
            ctx,
            card_filter=expr.card_filter,
        )
    elif t == ExprType.CONTROLLER_ENTERED_FOLLOWER_COUNT:
        if ctx is None:
            return 0
        return _entered_follower_count(
            ctx,
            card_filter=expr.card_filter,
        )
    elif t == ExprType.DISTRIBUTED_VALUE:
        return ctx.distributed_value if ctx is not None else 0

    raise ValueError(f"Unknown expression type: {t}")


def _distinct_entered_follower_count(ctx: EvalContext, *, card_filter) -> int:
    """Count differently named matching followers that entered for controller."""

    return len({
        record.definition.name
        for record in ctx.follower_entries
        if record.owner == ctx.controller
        and (
            card_filter is None
            or card_filter.matches(record.definition)
        )
    })


def _entered_follower_count(ctx: EvalContext, *, card_filter) -> int:
    """Count matching follower entries for the controller, including repeats."""

    return sum(
        1
        for record in ctx.follower_entries
        if record.owner == ctx.controller
        and (
            card_filter is None
            or card_filter.matches(record.definition)
        )
    )
