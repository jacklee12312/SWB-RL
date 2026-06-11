from __future__ import annotations

import random
from typing import TYPE_CHECKING

from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.state import Amulet, BoardCard, BoardEntity, Unit

if TYPE_CHECKING:
    from swb.engine.commands import ChoiceOption


_UNIT_TARGETS = frozenset({
    TargetKind.OWN_UNIT,
    TargetKind.ENEMY_UNIT,
    TargetKind.ANY_UNIT,
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
    TargetKind.OWN_AMULET,
    TargetKind.ENEMY_AMULET,
    TargetKind.ANY_AMULET,
    TargetKind.OWN_BOARD,
    TargetKind.ENEMY_BOARD,
    TargetKind.ANY_BOARD,
    TargetKind.OWN_HAND,
})

_RANDOM_TARGETS = frozenset({
    TargetKind.RANDOM_OWN_UNIT,
    TargetKind.RANDOM_ENEMY_UNIT,
    TargetKind.RANDOM_OWN_BOARD,
    TargetKind.RANDOM_ENEMY_BOARD,
})

_ALL_TARGETS = frozenset({
    TargetKind.ALL_OWN_UNITS,
    TargetKind.ALL_ENEMY_UNITS,
    TargetKind.ALL_UNITS,
    TargetKind.ALL_OWN_BOARD,
    TargetKind.ALL_ENEMY_BOARD,
    TargetKind.ALL_OWN_AMULETS,
    TargetKind.ALL_ENEMY_AMULETS,
})

_IMPLICIT_TARGETS = frozenset({
    TargetKind.SELF,
    TargetKind.OWN_LEADER,
    TargetKind.ENEMY_LEADER,
})


def is_manual_target(kind: TargetKind) -> bool:
    return kind in _MANUAL_TARGETS


def is_random_target(kind: TargetKind) -> bool:
    return kind in _RANDOM_TARGETS


def is_all_target(kind: TargetKind) -> bool:
    return kind in _ALL_TARGETS


def is_choice_target(kind: TargetKind) -> bool:
    return is_manual_target(kind)


_EFFECT_UNIT_ONLY = frozenset({
    EffectKind.DAMAGE_UNIT,
    EffectKind.BUFF_UNIT,
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
    elif target == TargetKind.ALL_OWN_AMULETS:
        candidates = _board_entities(own_board, units_only=False, amulets_only=True)
    elif target == TargetKind.ALL_ENEMY_AMULETS:
        candidates = _board_entities(enemy_board, units_only=False, amulets_only=True)
    else:
        return []

    return [e for e in candidates if _effect_compatible(e, operation.kind)]


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


def pick_random(candidates: list[BoardCard], rng: random.Random) -> BoardCard | None:
    if not candidates:
        return None
    return rng.choice(candidates)


def candidate_entity_ids(candidates: list[BoardCard]) -> list[int]:
    return [entity.entity_id for entity in candidates]


def hand_choice_options(player) -> list:
    from swb.engine.commands import ChoiceOption

    options = []
    for idx, (card, eid) in enumerate(zip(player.hand, player.hand_entity_ids)):
        options.append(
            ChoiceOption(
                option_id=f"hand:{eid}",
                label=card.name,
                entity_id=eid,
            )
        )
    return options

