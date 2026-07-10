from __future__ import annotations

import random
from typing import TYPE_CHECKING

from swb.engine.abilities import AbilityKeyword
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.origin import is_graveyard_return_eligible
from swb.engine.state import Amulet, BoardCard, BoardEntity, GraveyardCard, Unit

if TYPE_CHECKING:
    from swb.engine.commands import ChoiceOption


_UNIT_TARGETS = frozenset({
    TargetKind.OWN_UNIT,
    TargetKind.ENEMY_UNIT,
    TargetKind.ANY_UNIT,
    TargetKind.OWN_UNIT_OR_LEADER,
    TargetKind.ENEMY_UNIT_OR_LEADER,
    TargetKind.ANY_UNIT_OR_LEADER,
    TargetKind.RANDOM_OWN_UNIT,
    TargetKind.RANDOM_ENEMY_UNIT,
    TargetKind.ALL_OWN_UNITS,
    TargetKind.ALL_ENEMY_UNITS,
    TargetKind.ALL_UNITS,
})

_AMULET_TARGETS = frozenset({
    TargetKind.OWN_AMULET,
    TargetKind.ENEMY_AMULET,
    TargetKind.ANY_AMULET,
    TargetKind.ALL_OWN_AMULETS,
    TargetKind.ALL_ENEMY_AMULETS,
})

_BOARD_TARGETS = frozenset({
    TargetKind.OWN_BOARD,
    TargetKind.ENEMY_BOARD,
    TargetKind.ANY_BOARD,
    TargetKind.RANDOM_OWN_BOARD,
    TargetKind.RANDOM_ENEMY_BOARD,
    TargetKind.ALL_OWN_BOARD,
    TargetKind.ALL_ENEMY_BOARD,
})

_MANUAL_TARGETS = frozenset({
    TargetKind.OWN_UNIT,
    TargetKind.ENEMY_UNIT,
    TargetKind.ANY_UNIT,
    TargetKind.OWN_UNIT_OR_LEADER,
    TargetKind.ENEMY_UNIT_OR_LEADER,
    TargetKind.ANY_UNIT_OR_LEADER,
    TargetKind.OWN_AMULET,
    TargetKind.ENEMY_AMULET,
    TargetKind.ANY_AMULET,
    TargetKind.OWN_BOARD,
    TargetKind.ENEMY_BOARD,
    TargetKind.ANY_BOARD,
    TargetKind.OWN_HAND,
    TargetKind.OWN_GRAVEYARD_CARD,
})

_RANDOM_TARGETS = frozenset({
    TargetKind.RANDOM_OWN_UNIT,
    TargetKind.RANDOM_ENEMY_UNIT,
    TargetKind.RANDOM_OWN_BOARD,
    TargetKind.RANDOM_ENEMY_BOARD,
    TargetKind.RANDOM_OWN_HAND,
    TargetKind.RANDOM_OWN_GRAVEYARD_CARD,
})

_ALL_TARGETS = frozenset({
    TargetKind.ALL_OWN_UNITS,
    TargetKind.ALL_ENEMY_UNITS,
    TargetKind.ALL_UNITS,
    TargetKind.ALL_OWN_BOARD,
    TargetKind.ALL_ENEMY_BOARD,
    TargetKind.ALL_BOARD,
    TargetKind.ALL_OWN_AMULETS,
    TargetKind.ALL_ENEMY_AMULETS,
    TargetKind.ALL_OWN_HAND,
    TargetKind.ALL_OWN_GRAVEYARD_CARDS,
})

_IMPLICIT_TARGETS = frozenset({
    TargetKind.SELF,
    TargetKind.OWN_LEADER,
    TargetKind.ENEMY_LEADER,
})

_GRAVEYARD_TARGETS = frozenset({
    TargetKind.OWN_GRAVEYARD_CARD,
    TargetKind.RANDOM_OWN_GRAVEYARD_CARD,
    TargetKind.ALL_OWN_GRAVEYARD_CARDS,
})


def is_manual_target(kind: TargetKind) -> bool:
    return kind in _MANUAL_TARGETS


def is_random_target(kind: TargetKind) -> bool:
    return kind in _RANDOM_TARGETS


def is_all_target(kind: TargetKind) -> bool:
    return kind in _ALL_TARGETS


def is_choice_target(kind: TargetKind) -> bool:
    return is_manual_target(kind)


def is_graveyard_target(kind: TargetKind) -> bool:
    return kind in _GRAVEYARD_TARGETS


_EFFECT_UNIT_ONLY = frozenset({
    EffectKind.DAMAGE_UNIT,
    EffectKind.BUFF_UNIT,
    EffectKind.ADD_KEYWORD,
    EffectKind.REMOVE_KEYWORD,
    EffectKind.TRANSFORM,
    EffectKind.SET_STATS,
    EffectKind.ADD_ATTACK_RESTRICTION,
    EffectKind.REMOVE_ATTACK_RESTRICTION,
    EffectKind.ADD_TARGETING_RESTRICTION,
    EffectKind.REMOVE_TARGETING_RESTRICTION,
})

_EFFECT_AMULET_ONLY = frozenset()


def _effect_compatible(entity: BoardCard, kind: EffectKind) -> bool:
    if kind in _EFFECT_UNIT_ONLY:
        return isinstance(entity, Unit)
    if kind in _EFFECT_AMULET_ONLY:
        return isinstance(entity, Amulet)
    return True


def _board_entities(board: list[BoardCard], *, units_only: bool, amulets_only: bool, effect_kind: EffectKind | None = None) -> list[BoardCard]:
    result: list[BoardCard] = []
    for entity in board:
        if units_only and not isinstance(entity, Unit):
            continue
        if amulets_only and not isinstance(entity, Amulet):
            continue
        if effect_kind is not None and not _effect_compatible(entity, effect_kind):
            continue
        result.append(entity)
    return result


def _filter_graveyard_candidates(
    candidates: list[GraveyardCard],
    operation: EffectOperation,
) -> list[GraveyardCard]:
    if operation.kind is EffectKind.SUMMON_FROM_GRAVEYARD:
        candidates = [c for c in candidates if c.definition.card_type == "随从"]
    elif operation.kind is EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND:
        candidates = [c for c in candidates if is_graveyard_return_eligible(c)]
    if operation.graveyard_card_type is not None:
        candidates = [
            c for c in candidates
            if c.definition.card_type == operation.graveyard_card_type
        ]
    if operation.graveyard_follower_only:
        candidates = [
            c for c in candidates
            if c.definition.card_type == "随从"
        ]
    if operation.graveyard_cost_max is not None:
        candidates = [
            c for c in candidates
            if c.definition.cost <= operation.graveyard_cost_max
        ]
    if operation.graveyard_cost_min is not None:
        candidates = [
            c for c in candidates
            if c.definition.cost >= operation.graveyard_cost_min
        ]
    if operation.card_id is not None:
        candidates = [
            c for c in candidates
            if c.definition.card_id == operation.card_id
        ]
    return candidates


def graveyard_candidates(
    operation: EffectOperation,
    controller: int,
    players: list,
) -> list[GraveyardCard]:
    return _filter_graveyard_candidates(
        list(players[controller].graveyard),
        operation,
    )


def target_candidates(
    operation: EffectOperation,
    controller: int,
    players: list,
) -> list[BoardCard]:
    target = operation.target
    own_board = players[controller].board
    enemy_board = players[1 - controller].board

    candidates: list[BoardCard]
    if target == TargetKind.OWN_UNIT:
        candidates = _board_entities(own_board, units_only=True, amulets_only=False)
    elif target == TargetKind.ENEMY_UNIT:
        candidates = _board_entities(enemy_board, units_only=True, amulets_only=False)
    elif target == TargetKind.ANY_UNIT:
        candidates = _board_entities(own_board, units_only=True, amulets_only=False) + _board_entities(enemy_board, units_only=True, amulets_only=False)
    elif target == TargetKind.OWN_UNIT_OR_LEADER:
        candidates = _board_entities(own_board, units_only=True, amulets_only=False)
    elif target == TargetKind.ENEMY_UNIT_OR_LEADER:
        candidates = _board_entities(enemy_board, units_only=True, amulets_only=False)
    elif target == TargetKind.ANY_UNIT_OR_LEADER:
        candidates = _board_entities(own_board, units_only=True, amulets_only=False) + _board_entities(enemy_board, units_only=True, amulets_only=False)
    elif target == TargetKind.OWN_AMULET:
        candidates = _board_entities(own_board, units_only=False, amulets_only=True)
    elif target == TargetKind.ENEMY_AMULET:
        candidates = _board_entities(enemy_board, units_only=False, amulets_only=True)
    elif target == TargetKind.ANY_AMULET:
        candidates = _board_entities(own_board, units_only=False, amulets_only=True) + _board_entities(enemy_board, units_only=False, amulets_only=True)
    elif target == TargetKind.OWN_BOARD:
        candidates = list(own_board)
    elif target == TargetKind.ENEMY_BOARD:
        candidates = list(enemy_board)
    elif target == TargetKind.ANY_BOARD:
        candidates = list(own_board) + list(enemy_board)
    elif target == TargetKind.RANDOM_OWN_UNIT:
        candidates = _board_entities(own_board, units_only=True, amulets_only=False)
    elif target == TargetKind.RANDOM_ENEMY_UNIT:
        candidates = _board_entities(enemy_board, units_only=True, amulets_only=False)
    elif target == TargetKind.RANDOM_OWN_BOARD:
        candidates = list(own_board)
    elif target == TargetKind.RANDOM_ENEMY_BOARD:
        candidates = list(enemy_board)
    elif target == TargetKind.ALL_OWN_UNITS:
        candidates = _board_entities(own_board, units_only=True, amulets_only=False)
    elif target == TargetKind.ALL_ENEMY_UNITS:
        candidates = _board_entities(enemy_board, units_only=True, amulets_only=False)
    elif target == TargetKind.ALL_UNITS:
        candidates = _board_entities(own_board, units_only=True, amulets_only=False) + _board_entities(enemy_board, units_only=True, amulets_only=False)
    elif target == TargetKind.ALL_OWN_BOARD:
        candidates = list(own_board)
    elif target == TargetKind.ALL_ENEMY_BOARD:
        candidates = list(enemy_board)
    elif target == TargetKind.ALL_BOARD:
        candidates = list(own_board) + list(enemy_board)
    elif target == TargetKind.ALL_OWN_AMULETS:
        candidates = _board_entities(own_board, units_only=False, amulets_only=True)
    elif target == TargetKind.ALL_ENEMY_AMULETS:
        candidates = _board_entities(enemy_board, units_only=False, amulets_only=True)
    else:
        return []

    candidates = [e for e in candidates if _effect_compatible(e, operation.kind)]
    if operation.board_filter is not None:
        candidates = [
            e for e in candidates
            if operation.board_filter.matches_entity(e)
        ]

    if is_manual_target(operation.target):
        candidates = [
            e for e in candidates
            if not _is_unselectable_by_enemy_effects(e, controller, players)
        ]

    return candidates


def _is_unselectable_by_enemy_effects(
    entity: BoardCard, controller: int, players: list
) -> bool:
    protected = (
        isinstance(entity, Unit) and entity.cannot_be_enemy_targeted
    ) or (
        isinstance(entity, Amulet)
        and AbilityKeyword.EARTH_SIGIL in entity.definition.abilities
    )
    if not protected:
        return False
    for idx, player in enumerate(players):
        if entity in player.board:
            return idx != controller
    return False


def build_choice_options(candidates: list[BoardCard]) -> list[ChoiceOption]:
    from swb.engine.commands import ChoiceOption

    return [
        ChoiceOption(
            option_id=f"entity:{entity.entity_id}",
            label=entity.definition.name,
            entity_id=entity.entity_id,
        )
        for entity in candidates
    ]


def leader_choice_options(target: TargetKind, controller: int) -> list[ChoiceOption]:
    from swb.engine.commands import ChoiceOption

    if target == TargetKind.OWN_UNIT_OR_LEADER:
        leader_indexes = (controller,)
    elif target == TargetKind.ENEMY_UNIT_OR_LEADER:
        leader_indexes = (1 - controller,)
    elif target == TargetKind.ANY_UNIT_OR_LEADER:
        leader_indexes = (controller, 1 - controller)
    else:
        leader_indexes = ()
    return [
        ChoiceOption(
            option_id=f"leader:{player_index}",
            label=("己方主战者" if player_index == controller else "对方主战者"),
            leader_player_index=player_index,
        )
        for player_index in leader_indexes
    ]


def has_leader_choice(target: TargetKind) -> bool:
    return target in {
        TargetKind.OWN_UNIT_OR_LEADER,
        TargetKind.ENEMY_UNIT_OR_LEADER,
        TargetKind.ANY_UNIT_OR_LEADER,
    }


def build_graveyard_choice_options(
    candidates: list[GraveyardCard],
) -> list[ChoiceOption]:
    from swb.engine.commands import ChoiceOption

    return [
        ChoiceOption(
            option_id=f"entity:{gc.entity_id}",
            label=gc.definition.name,
            entity_id=gc.entity_id,
        )
        for gc in candidates
    ]


def pick_random(candidates: list[BoardCard], rng: random.Random) -> BoardCard | None:
    if not candidates:
        return None
    return rng.choice(candidates)


def pick_random_graveyard(
    candidates: list[GraveyardCard], rng: random.Random,
) -> GraveyardCard | None:
    if not candidates:
        return None
    return rng.choice(candidates)


def candidate_entity_ids(candidates: list[BoardCard]) -> list[int]:
    return [entity.entity_id for entity in candidates]


def hand_choice_options(player) -> list:
    from swb.engine.commands import ChoiceOption

    options = []
    for idx, card in enumerate(player.hand):
        eid = getattr(card, "entity_id", player.hand_entity_ids[idx])
        options.append(
            ChoiceOption(
                option_id=f"hand:{eid}",
                label=card.name,
                entity_id=eid,
            )
        )
    return options
