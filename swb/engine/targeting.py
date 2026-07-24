from __future__ import annotations

import random
from typing import TYPE_CHECKING

from swb.engine.abilities import AbilityKeyword
from swb.engine.effects import CandidateExtreme, EffectKind, EffectOperation, TargetKind
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
    TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER,
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
    TargetKind.ALL_BOARD,
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
    TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER,
    TargetKind.RANDOM_OWN_BOARD,
    TargetKind.RANDOM_ENEMY_BOARD,
    TargetKind.RANDOM_OWN_HAND,
    TargetKind.RANDOM_ENEMY_HAND,
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
    TargetKind.ALL_OWN_EMBLEMS,
    TargetKind.ALL_OWN_HAND,
    TargetKind.ALL_ENEMY_HAND,
    TargetKind.ALL_OWN_GRAVEYARD_CARDS,
    TargetKind.ALL_LEADERS,
})

_IMPLICIT_TARGETS = frozenset({
    TargetKind.SELF,
    TargetKind.ATTACK_TARGET,
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
    EffectKind.SUMMON_COPY,
    EffectKind.GRANT_TURN_END_DESTROY,
    EffectKind.BUFF_UNIT,
    EffectKind.ADD_KEYWORD,
    EffectKind.REMOVE_KEYWORD,
    EffectKind.SET_STATS,
    EffectKind.EVOLVE_UNIT,
    EffectKind.SUPER_EVOLVE_UNIT,
    EffectKind.ADD_ATTACK_RESTRICTION,
    EffectKind.REMOVE_ATTACK_RESTRICTION,
    EffectKind.ADD_TARGETING_RESTRICTION,
    EffectKind.REMOVE_TARGETING_RESTRICTION,
})

_EFFECT_AMULET_ONLY = frozenset()


def _effect_compatible(entity: BoardCard, kind: EffectKind) -> bool:
    if kind in _EFFECT_UNIT_ONLY:
        if not isinstance(entity, Unit):
            return False
        if kind in {EffectKind.EVOLVE_UNIT, EffectKind.SUPER_EVOLVE_UNIT}:
            return not entity.evolved
        return True
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


def hand_candidates(
    operation: EffectOperation,
    controller: int,
    players: list,
    *,
    source_entity_id: int | None = None,
) -> list:
    """Return legal hand targets, excluding the resolving source by identity."""

    hand_owner = (
        1 - controller
        if operation.target in {
            TargetKind.RANDOM_ENEMY_HAND,
            TargetKind.ALL_ENEMY_HAND,
        }
        else controller
    )
    candidates = list(players[hand_owner].hand)
    if source_entity_id is not None:
        candidates = [
            card
            for card in candidates
            if getattr(card, "entity_id", None) != source_entity_id
        ]
    if operation.hand_filter is not None:
        candidates = [
            card
            for card in candidates
            if operation.hand_filter.matches(card)
        ]
    return candidates


def target_candidates(
    operation: EffectOperation,
    controller: int,
    players: list,
    *,
    source_entity_id: int | None = None,
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
    elif target == TargetKind.RANDOM_ENEMY_UNIT_OR_LEADER:
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

    if operation.exclude_source and source_entity_id is not None:
        candidates = [
            entity for entity in candidates
            if entity.entity_id != source_entity_id
        ]

    if is_manual_target(operation.target):
        candidates = [
            e for e in candidates
            if not _is_unselectable_by_enemy_effects(e, controller, players)
        ]
        if enemy_has_forced_ability_target(controller, players):
            candidates = [
                entity
                for entity in candidates
                if entity not in enemy_board
                or (
                    isinstance(entity, Unit)
                    and entity.forces_enemy_ability_target
                )
            ]

    candidates = apply_candidate_extreme(candidates, operation.candidate_extreme)

    return candidates


def apply_candidate_extreme(
    candidates: list,
    extreme: CandidateExtreme | None,
) -> list:
    """Keep every candidate tied at the requested current-state extreme."""

    if extreme is None or not candidates:
        return candidates
    if extreme is CandidateExtreme.LEFTMOST:
        return candidates[:1]
    attribute = (
        "attack"
        if extreme in {CandidateExtreme.HIGHEST_ATTACK, CandidateExtreme.LOWEST_ATTACK}
        else "health"
    )
    values = [getattr(candidate, attribute, None) for candidate in candidates]
    comparable = [value for value in values if value is not None]
    if not comparable:
        return []
    choose = (
        max
        if extreme in {CandidateExtreme.HIGHEST_ATTACK, CandidateExtreme.HIGHEST_HEALTH}
        else min
    )
    selected_value = choose(comparable)
    return [
        candidate
        for candidate in candidates
        if getattr(candidate, attribute, None) == selected_value
    ]


def leader_target_ids(
    operation: EffectOperation,
    controller: int,
    players: list,
) -> list[int]:
    if operation.target is not TargetKind.ALL_LEADERS:
        return []
    player_indexes = [0, 1]
    if operation.candidate_extreme is not None:
        healths = [players[index].health for index in player_indexes]
        choose = (
            max
            if operation.candidate_extreme is CandidateExtreme.HIGHEST_HEALTH
            else min
        )
        selected_health = choose(healths)
        player_indexes = [
            index for index in player_indexes if players[index].health == selected_health
        ]
    return [-1 - player_index for player_index in player_indexes]


def _is_unselectable_by_enemy_effects(
    entity: BoardCard, controller: int, players: list
) -> bool:
    protected = (
        isinstance(entity, Unit) and entity.cannot_be_enemy_targeted
    ) or (
        isinstance(entity, Amulet)
        and (
            AbilityKeyword.EARTH_SIGIL in entity.definition.abilities
            or AbilityKeyword.AURA in entity.definition.abilities
        )
    )
    if not protected:
        return False
    for idx, player in enumerate(players):
        if entity in player.board:
            return idx != controller
    return False


def enemy_has_forced_ability_target(controller: int, players: list) -> bool:
    """Return whether the opposing field currently forces ability selections."""

    return any(
        isinstance(entity, Unit) and entity.forces_enemy_ability_target
        for entity in players[1 - controller].board
    )


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


def leader_choice_options(
    target: TargetKind,
    controller: int,
    players: list | None = None,
) -> list[ChoiceOption]:
    from swb.engine.commands import ChoiceOption

    if target == TargetKind.OWN_UNIT_OR_LEADER:
        leader_indexes = (controller,)
    elif target == TargetKind.ENEMY_UNIT_OR_LEADER:
        leader_indexes = (1 - controller,)
    elif target == TargetKind.ANY_UNIT_OR_LEADER:
        leader_indexes = (controller, 1 - controller)
    else:
        leader_indexes = ()
    if (
        players is not None
        and 1 - controller in leader_indexes
        and enemy_has_forced_ability_target(controller, players)
    ):
        leader_indexes = tuple(
            index for index in leader_indexes if index == controller
        )
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


def hand_choice_options(candidates: list) -> list:
    from swb.engine.commands import ChoiceOption

    return [
        ChoiceOption(
            option_id=f"hand:{card.entity_id}",
            label=getattr(card, "definition", card).name,
            entity_id=card.entity_id,
        )
        for card in candidates
    ]
