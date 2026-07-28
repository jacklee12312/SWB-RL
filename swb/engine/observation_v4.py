from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from gymnasium import spaces

from swb.engine.abilities import RUNTIME_UNIT_KEYWORDS
from swb.engine.commands import ChoiceKind
from swb.engine.effects import ModifierDuration, TurnEndDestroyTiming
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.state import Amulet, DeckCard, GraveyardCard, HandCard, Unit

if TYPE_CHECKING:
    from swb.engine.environment import ShadowverseEnv


HISTORY_LENGTH = 32
HISTORY_RECORDS_PER_PLAYER = 16
MAX_LEADER_AREA_SLOTS = 5
MAX_LEADER_DAMAGE_MODIFIERS = 8
MAX_EMBLEM_TRIGGERS = 8
MAX_FAITH_GRANTED_ABILITIES = 8
MAX_LISTENERS_PER_SOURCE = 8
MAX_COST_MODIFIERS = 10
MAX_STAT_MODIFIERS = 10
MAX_KEYWORD_MODIFIERS = 18
MAX_ATTACK_CAPACITY_MODIFIERS = 4
MAX_ATTACK_RESTRICTIONS = 3
MAX_TARGETING_RESTRICTIONS = 3
MAX_GRANTED_ABILITIES = 4
SEMANTIC_BITS = 32

RUNTIME_KEYWORDS = tuple(sorted(RUNTIME_UNIT_KEYWORDS))
KEYWORD_INDEX = {keyword: index for index, keyword in enumerate(RUNTIME_KEYWORDS)}
ORIGIN_VALUES = (None, *tuple(CardOrigin))
ORIGIN_INDEX = {origin: index for index, origin in enumerate(ORIGIN_VALUES)}
DURATION_VALUES = ("unknown", *tuple(duration.value for duration in ModifierDuration))
DURATION_INDEX = {duration: index for index, duration in enumerate(DURATION_VALUES)}
COST_MODE_VALUES = ("unknown", "set", "add", "subtract", "halve_round_up")
COST_MODE_INDEX = {mode: index for index, mode in enumerate(COST_MODE_VALUES)}
ATTACK_RESTRICTION_VALUES = (
    "unknown",
    "cannot_attack",
    "cannot_attack_leader",
    "cannot_attack_units",
)
TARGETING_RESTRICTION_VALUES = (
    "unknown",
    "cannot_be_targeted_by_enemy_effects",
    "forces_enemy_ability_target",
)
CHOICE_KIND_VALUES = tuple(ChoiceKind)
CHOICE_KIND_INDEX = {kind: index for index, kind in enumerate(CHOICE_KIND_VALUES)}
EVENT_VALUES = tuple(EventType)
EVENT_INDEX = {event: index for index, event in enumerate(EVENT_VALUES)}

ORIGIN_BITS = len(ORIGIN_VALUES)
DURATION_BITS = len(DURATION_VALUES)
RELATION_BITS = 3
CHOICE_REFERENCE_COUNT = 1 + 2 * 5 + 9
LEADER_SOURCE_REFERENCE_COUNT = (
    1 + 9 + 2 * 5 + 4 * MAX_LEADER_AREA_SLOTS
)

HAND_STATE_SIZE = 22
HAND_ORIGIN_SIZE = 2 * ORIGIN_BITS
HAND_KEYWORD_SIZE = len(RUNTIME_KEYWORDS)
HAND_MODIFIER_SIZE = (
    2 + MAX_COST_MODIFIERS * (len(COST_MODE_VALUES) + 1 + DURATION_BITS + RELATION_BITS)
    + 2 + MAX_STAT_MODIFIERS * (2 + DURATION_BITS + RELATION_BITS)
    + 2 * len(RUNTIME_KEYWORDS)
    + 2 + MAX_KEYWORD_MODIFIERS * (
        len(RUNTIME_KEYWORDS) + DURATION_BITS + RELATION_BITS
    )
    + 2 + MAX_KEYWORD_MODIFIERS * (
        len(RUNTIME_KEYWORDS) + DURATION_BITS + RELATION_BITS + 2
    )
)
HAND_EFFECT_SIZE = 2 + MAX_GRANTED_ABILITIES * SEMANTIC_BITS

BOARD_STATE_SIZE = 28
BOARD_ORIGIN_SIZE = 2 * ORIGIN_BITS
BOARD_KEYWORD_SIZE = len(RUNTIME_KEYWORDS)
BOARD_MODIFIER_SIZE = (
    2 + MAX_STAT_MODIFIERS * (2 + DURATION_BITS + RELATION_BITS)
    + 2 + MAX_ATTACK_CAPACITY_MODIFIERS * (1 + DURATION_BITS + RELATION_BITS)
    + 2 + MAX_ATTACK_RESTRICTIONS * (
        len(ATTACK_RESTRICTION_VALUES) + DURATION_BITS + RELATION_BITS
    )
    + 2 + MAX_TARGETING_RESTRICTIONS * (
        len(TARGETING_RESTRICTION_VALUES) + DURATION_BITS + RELATION_BITS
    )
    + 2 * len(RUNTIME_KEYWORDS)
    + 2 + MAX_KEYWORD_MODIFIERS * (
        len(RUNTIME_KEYWORDS) + DURATION_BITS + RELATION_BITS
    )
    + 2 + MAX_KEYWORD_MODIFIERS * (
        len(RUNTIME_KEYWORDS) + DURATION_BITS + RELATION_BITS + 2
    )
)
BOARD_EFFECT_SIZE = (
    2 * (2 + MAX_GRANTED_ABILITIES * SEMANTIC_BITS)
    + SEMANTIC_BITS
)

PLAYER_STATE_SIZE = 28
MATCH_STATE_SIZE = 14
FAITH_STATE_SIZE = (
    4 + MAX_FAITH_GRANTED_ABILITIES * SEMANTIC_BITS
)
EMBLEM_STATE_SIZE = (
    3 + 2 * MAX_EMBLEM_TRIGGERS + SEMANTIC_BITS
)
LEADER_MODIFIER_ROW_SIZE = (
    1 + 3 + DURATION_BITS + RELATION_BITS + RELATION_BITS
    + LEADER_SOURCE_REFERENCE_COUNT
)
LEADER_AREA_STATE_SIZE = (
    2 * MAX_LEADER_AREA_SLOTS * FAITH_STATE_SIZE
    + 2 * MAX_LEADER_AREA_SLOTS * EMBLEM_STATE_SIZE
    + 2 * 3
    + 2 * MAX_LEADER_DAMAGE_MODIFIERS * LEADER_MODIFIER_ROW_SIZE
)
LISTENER_SOURCE_COUNT = 9 + 2 * 5 + 4 * MAX_LEADER_AREA_SLOTS
LISTENER_STATE_SIZE = LISTENER_SOURCE_COUNT * (
    2 * MAX_LISTENERS_PER_SOURCE + 1
)

CHOICE_STATE_SIZE = 1 + len(CHOICE_KIND_VALUES) + 4
CHOICE_OPTION_STATE_SIZE = (
    CHOICE_REFERENCE_COUNT + RELATION_BITS + 1 + SEMANTIC_BITS
)
GRAVEYARD_OPTION_STATE_SIZE = (
    1 + 2 * ORIGIN_BITS + 2 + SEMANTIC_BITS
)
HISTORY_EVENT_SIZE = 1 + len(EVENT_VALUES)
HISTORY_REFERENCE_SIZE = 2 * CHOICE_REFERENCE_COUNT

PUBLIC_CARD_ID_EVENTS = frozenset({
    EventType.CARD_PLAYED,
    EventType.FOLLOWER_SUMMONED,
    EventType.AMULET_ENTERED,
    EventType.SPELL_RESOLVED,
    EventType.FOLLOWER_EVOLVED,
    EventType.FOLLOWER_SUPER_EVOLVED,
    EventType.FOLLOWER_DESTROYED,
    EventType.AMULET_DESTROYED,
    EventType.CARD_BANISHED,
    EventType.CARD_RETURNED_TO_HAND,
    EventType.CARD_RETURNED_TO_DECK,
    EventType.CARD_DISCARDED,
    EventType.GRAVEYARD_ENTERED,
    EventType.GRAVEYARD_CARD_RETURNED,
    EventType.GRAVEYARD_CARD_SUMMONED,
    EventType.GRAVEYARD_CARD_BANISHED,
    EventType.BOARD_CARD_TRANSFORMED,
    EventType.CARD_INVOKED,
    EventType.AMULET_ACTIVATED,
    EventType.EMBLEM_GAINED,
    EventType.EMBLEM_REMOVED,
})
PUBLIC_METADATA_KEYS = frozenset({
    "cause",
    "entry_cause",
    "mode_id",
    "origin",
    "source_origin",
    "trigger",
    "choice_kind",
})


def _int_array(values, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = tuple(values)
    result = np.asarray(values, dtype=np.int32)
    return result if shape is None else result.reshape(shape)


def _float_array(values, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = tuple(values)
    result = np.asarray(values, dtype=np.float32)
    return result if shape is None else result.reshape(shape)


def _one_hot(index: int, size: int) -> tuple[float, ...]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return tuple(values)


def _relation(player_index: int | None, perspective: int) -> tuple[float, ...]:
    if player_index not in (0, 1):
        return (1.0, 0.0, 0.0)
    if player_index == perspective:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def _duration(duration: str | None) -> tuple[float, ...]:
    return _one_hot(DURATION_INDEX.get(duration or "unknown", 0), DURATION_BITS)


def _origin(origin: CardOrigin | None) -> tuple[float, ...]:
    return _one_hot(ORIGIN_INDEX.get(origin, 0), ORIGIN_BITS)


def _origin_pair(
    origin: CardOrigin | None,
    source_origin: CardOrigin | None,
) -> tuple[float, ...]:
    return (*_origin(origin), *_origin(source_origin))


def _normalize_semantic(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _normalize_semantic(getattr(value, field.name))
            for field in fields(value)
            if field.name not in {"entity_id", "source_entity_id", "target_entity_id"}
        }
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_semantic(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in {"entity_id", "source_id", "target_id", "request_id"}
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_semantic(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_normalize_semantic(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return type(value).__name__


def _semantic_bits(value) -> tuple[float, ...]:
    payload = json.dumps(
        _normalize_semantic(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    values = []
    for byte in digest[: SEMANTIC_BITS // 8]:
        values.extend(float((byte >> shift) & 1) for shift in range(7, -1, -1))
    return tuple(values)


def _list_rows(items, maximum: int, row_encoder) -> tuple[float, ...]:
    values = [
        min(len(items), maximum) / max(1, maximum),
        float(len(items) > maximum),
    ]
    row_size = len(row_encoder(items[0])) if items else len(row_encoder(None))
    for item in items[:maximum]:
        values.extend(row_encoder(item))
    values.extend([0.0] * (maximum - min(len(items), maximum)) * row_size)
    return tuple(values)


def _card_definition(entry):
    return entry.definition if isinstance(entry, DeckCard) else entry


def _card_index(env: ShadowverseEnv, definition) -> int:
    if definition is None:
        return 0
    return env._v2_card_index.get(definition.card_id, 0)


def _histogram(env: ShadowverseEnv, definitions) -> tuple[int, ...]:
    counts = [0] * len(env.card_vocabulary)
    for definition in definitions:
        index = _card_index(env, definition)
        if index:
            counts[index - 1] += 1
    return tuple(counts)


def _keyword_modifier_state(entity, perspective: int) -> tuple[float, ...]:
    permanent = tuple(
        float(keyword in entity.permanent_keywords) for keyword in RUNTIME_KEYWORDS
    )
    removed = tuple(
        float(keyword in entity.removed_keywords) for keyword in RUNTIME_KEYWORDS
    )

    def grant_row(modifier):
        if modifier is None:
            return (0.0,) * (len(RUNTIME_KEYWORDS) + DURATION_BITS + RELATION_BITS)
        return (
            *_one_hot(KEYWORD_INDEX.get(modifier.keyword, -1), len(RUNTIME_KEYWORDS)),
            *_duration(modifier.duration),
            *_relation(modifier.expires_for_player, perspective),
        )

    def removal_row(modifier):
        if modifier is None:
            return (0.0,) * (
                len(RUNTIME_KEYWORDS) + DURATION_BITS + RELATION_BITS + 2
            )
        return (
            *_one_hot(KEYWORD_INDEX.get(modifier.keyword, -1), len(RUNTIME_KEYWORDS)),
            *_duration(modifier.duration),
            *_relation(modifier.expires_for_player, perspective),
            float(modifier.restore_barrier_charge),
            float(modifier.restore_ambush),
        )

    return (
        *permanent,
        *removed,
        *_list_rows(entity.temporary_keywords, MAX_KEYWORD_MODIFIERS, grant_row),
        *_list_rows(
            entity.temporary_keyword_removals,
            MAX_KEYWORD_MODIFIERS,
            removal_row,
        ),
    )


def _stat_modifier_rows(modifiers, perspective: int) -> tuple[float, ...]:
    def row(modifier):
        if modifier is None:
            return (0.0,) * (2 + DURATION_BITS + RELATION_BITS)
        return (
            modifier.attack_delta / 40,
            modifier.health_delta / 40,
            *_duration(modifier.duration),
            *_relation(modifier.expires_for_player, perspective),
        )

    return _list_rows(modifiers, MAX_STAT_MODIFIERS, row)


def _granted_effect_rows(effects) -> tuple[float, ...]:
    return _list_rows(
        effects,
        MAX_GRANTED_ABILITIES,
        lambda item: (
            (0.0,) * SEMANTIC_BITS if item is None else _semantic_bits(item)
        ),
    )


def _hand_state(
    env: ShadowverseEnv,
    card: HandCard | None,
    turns_started: int,
    turn: int,
) -> tuple[float, ...]:
    if card is None:
        return (0.0,) * HAND_STATE_SIZE
    gauge = card.union_burst_gauge(turns_started)
    union_bursts = env._core.rulebook.union_bursts_for(
        card.definition.card_id
    )
    has_union_burst = any(
        definition.kind.value == "union_burst"
        for definition in union_bursts
    )
    has_super_skybound_art = any(
        definition.kind.value == "super_skybound_art"
        for definition in union_bursts
    )
    return (
        1.0,
        card.current_cost / 20,
        card.spellboost_count / 20,
        card.spellboost_cost_reduction / 10,
        len(card.fused_material_ids) / 9,
        float(card.fusion_used_turn == turn),
        card.evolutions_while_in_hand / 15,
        float(card.cannot_be_played),
        float(card.definition.is_collectible),
        float(card.effect_destroy_immunity),
        (card.attack or 0) / 40,
        (card.life or 0) / 40,
        gauge / 15,
        len(card.cost_modifiers) / MAX_COST_MODIFIERS,
        len(card.stat_modifiers) / MAX_STAT_MODIFIERS,
        len(card.granted_last_words) / MAX_GRANTED_ABILITIES,
        float(has_union_burst),
        float(has_super_skybound_art),
        10 / 15 if has_union_burst else 0.0,
        1.0 if has_super_skybound_art else 0.0,
        float(has_union_burst and gauge >= 10),
        float(has_super_skybound_art and gauge >= 15),
    )


def _hand_modifier_state(card: HandCard | None, perspective: int) -> tuple[float, ...]:
    if card is None:
        return (0.0,) * HAND_MODIFIER_SIZE

    def cost_row(modifier):
        if modifier is None:
            return (0.0,) * (
                len(COST_MODE_VALUES) + 1 + DURATION_BITS + RELATION_BITS
            )
        return (
            *_one_hot(COST_MODE_INDEX.get(modifier.mode, 0), len(COST_MODE_VALUES)),
            modifier.amount / 20,
            *_duration(modifier.duration),
            *_relation(modifier.expires_for_player, perspective),
        )

    values = (
        *_list_rows(card.cost_modifiers, MAX_COST_MODIFIERS, cost_row),
        *_stat_modifier_rows(card.stat_modifiers, perspective),
        *_keyword_modifier_state(card, perspective),
    )
    if len(values) != HAND_MODIFIER_SIZE:
        raise AssertionError((len(values), HAND_MODIFIER_SIZE))
    return values


def _hand_effect_state(card: HandCard | None) -> tuple[float, ...]:
    if card is None:
        return (0.0,) * HAND_EFFECT_SIZE
    return _granted_effect_rows(card.granted_last_words)


def _board_state(entity, turn: int) -> tuple[float, ...]:
    if entity is None:
        return (0.0,) * BOARD_STATE_SIZE
    if isinstance(entity, Amulet):
        return (
            1.0, 0.0, 1.0,
            0.0, 0.0, 0.0,
            (entity.countdown or 0) / 20,
            entity.earth_sigil_count / 20,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
            float(entity.pending_destroy),
            float(entity.activated_turn == turn),
            float(entity.entered_turn == turn),
            len(entity.fused_material_ids) / 9,
            0.0, 0.0, 0.0, 0.0, 0.0,
        )
    return (
        1.0, 1.0, 0.0,
        entity.attack / 40,
        max(entity.health, 0) / 40,
        entity.max_health / 40,
        0.0, 0.0,
        entity.attacks_remaining / 4,
        entity.attacks_per_turn / 4,
        float(entity.can_attack),
        float(entity.rush_only),
        float(entity.evolved),
        float(entity.super_evolved),
        entity.barrier_charges / 4,
        float(entity.ambush_active),
        float(entity.summoned_this_turn),
        float(entity.printed_abilities_removed),
        float(entity.last_words_removed),
        0.0, 0.0, 0.0,
        len(entity.fused_material_ids) / 9,
        float(entity.effect_destroy_immunity),
        float(TurnEndDestroyTiming.OWNER_TURN in entity.turn_end_destroy_timings),
        float(TurnEndDestroyTiming.OWNER_TURN in entity.turn_end_banish_timings),
        float(TurnEndDestroyTiming.OPPONENT_TURN in entity.turn_end_destroy_timings),
        float(TurnEndDestroyTiming.OPPONENT_TURN in entity.turn_end_banish_timings),
    )


def _restriction_rows(items, maximum: int, values: tuple[str, ...], perspective: int):
    def row(modifier):
        if modifier is None:
            return (0.0,) * (len(values) + DURATION_BITS + RELATION_BITS)
        restriction = getattr(modifier.restriction, "value", modifier.restriction)
        index = values.index(restriction) if restriction in values else 0
        return (
            *_one_hot(index, len(values)),
            *_duration(modifier.duration),
            *_relation(modifier.expires_for_player, perspective),
        )

    return _list_rows(items, maximum, row)


def _board_modifier_state(entity, perspective: int) -> tuple[float, ...]:
    if not isinstance(entity, Unit):
        return (0.0,) * BOARD_MODIFIER_SIZE

    def capacity_row(modifier):
        if modifier is None:
            return (0.0,) * (1 + DURATION_BITS + RELATION_BITS)
        return (
            modifier.attacks_per_turn / 4,
            *_duration(modifier.duration),
            *_relation(modifier.expires_for_player, perspective),
        )

    values = (
        *_stat_modifier_rows(entity.stat_modifiers, perspective),
        *_list_rows(
            entity.attack_capacity_modifiers,
            MAX_ATTACK_CAPACITY_MODIFIERS,
            capacity_row,
        ),
        *_restriction_rows(
            entity.attack_restrictions,
            MAX_ATTACK_RESTRICTIONS,
            ATTACK_RESTRICTION_VALUES,
            perspective,
        ),
        *_restriction_rows(
            entity.targeting_restrictions,
            MAX_TARGETING_RESTRICTIONS,
            TARGETING_RESTRICTION_VALUES,
            perspective,
        ),
        *_keyword_modifier_state(entity, perspective),
    )
    if len(values) != BOARD_MODIFIER_SIZE:
        raise AssertionError((len(values), BOARD_MODIFIER_SIZE))
    return values


def _board_effect_state(entity) -> tuple[float, ...]:
    if not isinstance(entity, Unit):
        return (0.0,) * BOARD_EFFECT_SIZE
    values = (
        *_granted_effect_rows(entity.granted_last_words),
        *_granted_effect_rows(entity.granted_turn_end_abilities),
        *_semantic_bits(entity.random_choice_history),
    )
    if len(values) != BOARD_EFFECT_SIZE:
        raise AssertionError((len(values), BOARD_EFFECT_SIZE))
    return values


def _relative_players(env: ShadowverseEnv, perspective: int):
    return env._core.players[perspective], env._core.players[1 - perspective]


def _artifact_entry_kind_count(env: ShadowverseEnv, player_index: int) -> int:
    return len({
        record.definition.name
        for record in env._core.state.follower_entries
        if record.owner == player_index
        and record.definition.card_type == "随从"
        and record.definition.tribe_name == "创造物"
    })


def _player_state(env: ShadowverseEnv, player_index: int) -> tuple[float, ...]:
    player = env._core.players[player_index]
    return (
        player.health / 40,
        player.max_health / 40,
        player.mana / 10,
        player.max_mana / 10,
        len(player.deck) / 40,
        len(player.hand) / 9,
        player.evolution_points / 4,
        player.super_evolution_points / 4,
        player.turns_started / 200,
        float(player.evolved_this_turn),
        float(player.super_evolved_this_turn),
        player.followers_evolved_this_match / 20,
        player.cards_played_this_turn / 20,
        player.follower_attacks_this_turn / 10,
        player.followers_destroyed_this_turn / 10,
        player.cooperation / 40,
        player.shadows / 40,
        player.earth_sigils / 20,
        len(player.faiths) / MAX_LEADER_AREA_SLOTS,
        len(player.emblems) / MAX_LEADER_AREA_SLOTS,
        float(player.extra_pp_available),
        player.extra_pp_uses / 2,
        float(player.extra_pp_refresh_done),
        float(player.extra_pp_active_turn == env.turn),
        player.fatigue / 20,
        player.leader_barrier_charges / 4,
        float(player.empty_deck_outcome.value == "victory"),
        _artifact_entry_kind_count(env, player_index) / 40,
    )


def _match_state(env: ShadowverseEnv, perspective: int) -> tuple[float, ...]:
    total_pages = env._graveyard_total_pages()
    phase_values = ("mulligan", "main", "awaiting_choice", "finished")
    return (
        env.turn / 200,
        float(env._core.state.active_player == perspective),
        float(env._core.state.active_player == 1 - perspective),
        float(env._core.state.first_player == perspective),
        float(env._core.state.first_player == 1 - perspective),
        *(
            float(env._core.state.phase.value == phase)
            for phase in phase_values
        ),
        float(env._core.state.mulligan_completed[perspective]),
        float(env._core.state.mulligan_completed[1 - perspective]),
        float(env._core.state.pending_choice is not None),
        (
            env._graveyard_page / max(1, total_pages - 1)
            if total_pages > 1 else 0.0
        ),
        total_pages / 16,
    )


def _public_entity_reference(
    env: ShadowverseEnv,
    entity_id: int | None,
    perspective: int,
) -> int:
    if entity_id is None:
        return 0
    me, opponent = _relative_players(env, perspective)
    for index, entity in enumerate(me.board):
        if entity.entity_id == entity_id:
            return 1 + index
    for index, entity in enumerate(opponent.board):
        if entity.entity_id == entity_id:
            return 1 + env.MAX_BOARD + index
    for index, card in enumerate(me.hand):
        if card.entity_id == entity_id:
            return 1 + 2 * env.MAX_BOARD + index
    return 0


def _public_definition_for_entity(
    env: ShadowverseEnv,
    entity_id: int | None,
    perspective: int,
):
    if entity_id is None:
        return None
    me, opponent = _relative_players(env, perspective)
    for entity in (*me.board, *opponent.board):
        if entity.entity_id == entity_id:
            return entity.definition
    for card in me.hand:
        if card.entity_id == entity_id:
            return card.definition
    for player in (me, opponent):
        for card in player.graveyard:
            if card.entity_id == entity_id:
                return card.definition
    return None


def _choice_option_definition(
    env: ShadowverseEnv,
    option,
    perspective: int,
):
    return _public_definition_for_entity(env, option.entity_id, perspective)


def _choice_option_semantics(request, option) -> tuple[float, ...]:
    if option.entity_id is not None or option.leader_player_index is not None:
        value = (request.choice_kind.value, "entity-or-leader")
    else:
        value = (
            request.choice_kind.value,
            option.option_id,
            option.label,
        )
    return _semantic_bits(value)


def _choice_fields(env: ShadowverseEnv, perspective: int):
    request = env._core.state.pending_choice
    empty_options = (0,) * env.MAX_CHOICE_OPTIONS
    if request is None or perspective != env.decision_player:
        return {
            "state": (0.0,) * CHOICE_STATE_SIZE,
            "cards": empty_options,
            "options": (0.0,) * (
                env.MAX_CHOICE_OPTIONS * CHOICE_OPTION_STATE_SIZE
            ),
            "graveyard_cards": (0,) * env.GRAVEYARD_PAGE_SIZE,
            "graveyard_options": (0.0,) * (
                env.GRAVEYARD_PAGE_SIZE * GRAVEYARD_OPTION_STATE_SIZE
            ),
        }
    state = (
        1.0,
        *_one_hot(CHOICE_KIND_INDEX[request.choice_kind], len(CHOICE_KIND_VALUES)),
        len(request.options) / 64,
        request.target_count / env.MAX_CHOICE_OPTIONS,
        len(request.selected_options) / env.MAX_CHOICE_OPTIONS,
        float(request.allow_duplicate_targets),
    )
    selected_ids = {option.option_id for option in request.selected_options}
    option_cards = []
    option_state = []
    if request.choice_kind is not ChoiceKind.GRAVEYARD:
        for option in request.options[: env.MAX_CHOICE_OPTIONS]:
            definition = _choice_option_definition(env, option, perspective)
            option_cards.append(_card_index(env, definition))
            reference = _public_entity_reference(env, option.entity_id, perspective)
            if option.leader_player_index is None:
                leader_relation = (1.0, 0.0, 0.0)
            else:
                leader_relation = _relation(
                    option.leader_player_index,
                    perspective,
                )
            option_state.extend((
                *_one_hot(reference, CHOICE_REFERENCE_COUNT),
                *leader_relation,
                float(option.option_id in selected_ids),
                *_choice_option_semantics(request, option),
            ))
    option_cards.extend([0] * (env.MAX_CHOICE_OPTIONS - len(option_cards)))
    option_state.extend(
        [0.0] * (
            env.MAX_CHOICE_OPTIONS * CHOICE_OPTION_STATE_SIZE
            - len(option_state)
        )
    )

    graveyard_cards = []
    graveyard_state = []
    if request.choice_kind is ChoiceKind.GRAVEYARD:
        start = env._graveyard_page * env.GRAVEYARD_PAGE_SIZE
        options = request.options[start : start + env.GRAVEYARD_PAGE_SIZE]
        me = env._core.players[perspective]
        opponent = env._core.players[1 - perspective]
        by_entity = {
            card.entity_id: card
            for player in (me, opponent)
            for card in player.graveyard
        }
        for option in options:
            card = by_entity.get(option.entity_id)
            definition = None if card is None else card.definition
            graveyard_cards.append(_card_index(env, definition))
            if card is None:
                graveyard_state.extend([0.0] * GRAVEYARD_OPTION_STATE_SIZE)
            else:
                graveyard_state.extend((
                    1.0,
                    *_origin_pair(card.origin, card.source_origin),
                    float(card.derived),
                    float(card.token),
                    *_semantic_bits((card.entry_cause, option.label)),
                ))
    graveyard_cards.extend([0] * (env.GRAVEYARD_PAGE_SIZE - len(graveyard_cards)))
    graveyard_state.extend(
        [0.0] * (
            env.GRAVEYARD_PAGE_SIZE * GRAVEYARD_OPTION_STATE_SIZE
            - len(graveyard_state)
        )
    )
    return {
        "state": state,
        "cards": tuple(option_cards),
        "options": tuple(option_state),
        "graveyard_cards": tuple(graveyard_cards),
        "graveyard_options": tuple(graveyard_state),
    }


def _event_public_card_id(
    env: ShadowverseEnv,
    event,
    perspective: int,
    *,
    target: bool,
) -> int:
    entity_id = event.target_id if target else event.source_id
    definition = _public_definition_for_entity(env, entity_id, perspective)
    if definition is not None:
        return _card_index(env, definition)
    if target:
        keys = ("target_card_id",)
    else:
        keys = ("card_id", "source_card_id")
    can_reveal = (
        event.player_index == perspective
        or event.type in PUBLIC_CARD_ID_EVENTS
    )
    if can_reveal:
        for key in keys:
            card_id = event.metadata.get(key)
            if isinstance(card_id, int) and not isinstance(card_id, bool):
                return env._v2_card_index.get(card_id, 0)
    return 0


def _history_fields(env: ShadowverseEnv, perspective: int):
    events = env._core.event_history[-HISTORY_LENGTH:]
    padding = HISTORY_LENGTH - len(events)
    event_bits = [0.0] * padding * HISTORY_EVENT_SIZE
    actor_bits = [0.0] * padding * RELATION_BITS
    amounts = [0.0] * padding
    references = [0.0] * padding * HISTORY_REFERENCE_SIZE
    semantics = [0.0] * padding * SEMANTIC_BITS
    source_cards = [0] * padding
    target_cards = [0] * padding
    for event in events:
        event_bits.extend((
            1.0,
            *_one_hot(EVENT_INDEX[event.type], len(EVENT_VALUES)),
        ))
        actor_bits.extend(_relation(event.player_index, perspective))
        amounts.append(max(-40, min(40, event.amount)) / 40)
        source_reference = _public_entity_reference(
            env, event.source_id, perspective
        )
        target_reference = _public_entity_reference(
            env, event.target_id, perspective
        )
        references.extend((
            *_one_hot(source_reference, CHOICE_REFERENCE_COUNT),
            *_one_hot(target_reference, CHOICE_REFERENCE_COUNT),
        ))
        can_reveal_metadata = (
            event.player_index == perspective
            or event.type in PUBLIC_CARD_ID_EVENTS
        )
        public_metadata = (
            {
                key: value
                for key, value in event.metadata.items()
                if key in PUBLIC_METADATA_KEYS
            }
            if can_reveal_metadata
            else {}
        )
        semantics.extend(_semantic_bits((event.type.value, public_metadata)))
        source_cards.append(
            _event_public_card_id(env, event, perspective, target=False)
        )
        target_cards.append(
            _event_public_card_id(env, event, perspective, target=True)
        )
    return {
        "event_bits": tuple(event_bits),
        "actor_bits": tuple(actor_bits),
        "amounts": tuple(amounts),
        "references": tuple(references),
        "semantics": tuple(semantics),
        "source_cards": tuple(source_cards),
        "target_cards": tuple(target_cards),
    }


def _leader_source_reference(
    env: ShadowverseEnv,
    entity_id: int | None,
    perspective: int,
) -> int:
    if entity_id is None:
        return 0
    me, opponent = _relative_players(env, perspective)
    offset = 1
    for collection in (
        me.hand,
        me.board,
        opponent.board,
        me.faiths,
        opponent.faiths,
        me.emblems,
        opponent.emblems,
    ):
        for index, entity in enumerate(collection):
            if entity.entity_id == entity_id:
                return offset + index
        if collection is me.hand:
            offset += 9
        else:
            offset += 5
    return 0


def _leader_area_fields(env: ShadowverseEnv, perspective: int):
    me, opponent = _relative_players(env, perspective)
    faiths = []
    emblems = []
    for player in (me, opponent):
        faiths.extend([*player.faiths[:MAX_LEADER_AREA_SLOTS]])
        faiths.extend([None] * (MAX_LEADER_AREA_SLOTS - len(player.faiths)))
        emblems.extend([*player.emblems[:MAX_LEADER_AREA_SLOTS]])
        emblems.extend([None] * (MAX_LEADER_AREA_SLOTS - len(player.emblems)))

    cards = []
    state = []
    for faith in faiths:
        cards.append(
            0 if faith is None else env._v2_card_index.get(faith.source_card_id, 0)
        )
        if faith is None:
            state.extend([0.0] * FAITH_STATE_SIZE)
            continue
        state.extend((
            1.0,
            faith.value / 50,
            len(faith.granted_abilities) / MAX_FAITH_GRANTED_ABILITIES,
            faith.mode_selection_bonus / 8,
        ))
        for ability in faith.granted_abilities[:MAX_FAITH_GRANTED_ABILITIES]:
            state.extend(_semantic_bits(ability))
        state.extend(
            [0.0]
            * (MAX_FAITH_GRANTED_ABILITIES - min(
                len(faith.granted_abilities),
                MAX_FAITH_GRANTED_ABILITIES,
            ))
            * SEMANTIC_BITS
        )
    for emblem in emblems:
        cards.append(
            0 if emblem is None else env._v2_card_index.get(emblem.source_card_id, 0)
        )
        if emblem is None:
            state.extend([0.0] * EMBLEM_STATE_SIZE)
            continue
        state.extend((
            1.0,
            (emblem.countdown or 0) / 20,
            (emblem.countdown_before or 0) / 20,
        ))
        state.extend(
            emblem.activation_counts.get(index, 0) / 10
            for index in range(MAX_EMBLEM_TRIGGERS)
        )
        state.extend(
            float(index in emblem._once_per_turn_used)
            for index in range(MAX_EMBLEM_TRIGGERS)
        )
        state.extend(_semantic_bits(emblem.random_choice_history))

    modifier_source_cards = []
    for player in (me, opponent):
        modifiers = player.leader_damage_modifiers
        state.extend((
            len(modifiers) / MAX_LEADER_DAMAGE_MODIFIERS,
            float(len(modifiers) > MAX_LEADER_DAMAGE_MODIFIERS),
            sum(modifier.amount for modifier in modifiers) / 20,
        ))
        for modifier in modifiers[:MAX_LEADER_DAMAGE_MODIFIERS]:
            source_reference = _leader_source_reference(
                env, modifier.source_entity_id, perspective
            )
            state.extend((
                modifier.amount / 10,
                *_one_hot(
                    {
                        "additive": 1,
                        "set_zero_if_positive": 2,
                    }.get(modifier.mode, 0),
                    3,
                ),
                *_duration(modifier.duration),
                *_relation(modifier.expires_for_player, perspective),
                *_relation(modifier.source_controller, perspective),
                *_one_hot(
                    source_reference,
                    LEADER_SOURCE_REFERENCE_COUNT,
                ),
            ))
            modifier_source_cards.append(
                env._v2_card_index.get(modifier.source_card_id or 0, 0)
            )
        missing = MAX_LEADER_DAMAGE_MODIFIERS - min(
            len(modifiers), MAX_LEADER_DAMAGE_MODIFIERS
        )
        state.extend([0.0] * missing * LEADER_MODIFIER_ROW_SIZE)
        modifier_source_cards.extend([0] * missing)

    if len(state) != LEADER_AREA_STATE_SIZE:
        raise AssertionError((len(state), LEADER_AREA_STATE_SIZE))
    return tuple(cards), tuple(state), tuple(modifier_source_cards)


def _listener_state(env: ShadowverseEnv, perspective: int) -> tuple[float, ...]:
    me, opponent = _relative_players(env, perspective)
    sources = [
        *me.hand[:9],
        *([None] * (9 - len(me.hand[:9]))),
        *me.board[:5],
        *([None] * (5 - len(me.board))),
        *opponent.board[:5],
        *([None] * (5 - len(opponent.board))),
    ]
    for collection in (me.faiths, opponent.faiths, me.emblems, opponent.emblems):
        sources.extend(collection[:MAX_LEADER_AREA_SLOTS])
        sources.extend([None] * (MAX_LEADER_AREA_SLOTS - len(collection)))
    values = []
    counts = env._core.state.listener_activation_counts
    used = env._core.state.listener_once_per_turn_used
    for source in sources:
        if source is None:
            values.extend(
                [0.0] * (2 * MAX_LISTENERS_PER_SOURCE + 1)
            )
            continue
        if hasattr(source, "definition") and hasattr(source.definition, "card_id"):
            card_id = source.definition.card_id
        else:
            card_id = source.source_card_id
        for definition_index in range(MAX_LISTENERS_PER_SOURCE):
            key = (source.entity_id, card_id, definition_index)
            values.append(counts.get(key, 0) / 10)
        for definition_index in range(MAX_LISTENERS_PER_SOURCE):
            key = (source.entity_id, card_id, definition_index)
            values.append(float(key in used))
        values.append(float(any(
            key[0] == source.entity_id
            and key[1] == card_id
            and key[2] >= MAX_LISTENERS_PER_SOURCE
            for key in counts
        )))
    if len(values) != LISTENER_STATE_SIZE:
        raise AssertionError((len(values), LISTENER_STATE_SIZE))
    return tuple(values)


def _fusion_card_indices(env: ShadowverseEnv, entities, player) -> tuple[int, ...]:
    material_by_id = {
        material.entity_id: material.definition
        for material in player.fusion_materials
    }
    values = []
    for entity in entities:
        material_ids = () if entity is None else entity.fused_material_ids
        for material_id in material_ids[:9]:
            values.append(_card_index(env, material_by_id.get(material_id)))
        values.extend([0] * (9 - min(len(material_ids), 9)))
    return tuple(values)


def _history_record_fields(env: ShadowverseEnv, perspective: int):
    state = env._core.state
    card_fields = {
        "destroyed_follower_cards": [],
        "destroyed_amulet_cards": [],
        "follower_entry_cards": [],
    }
    runtime_fields = {
        "destroyed_follower_state": [],
        "destroyed_amulet_state": [],
        "follower_entry_state": [],
    }
    for relative_owner, owner in enumerate((perspective, 1 - perspective)):
        follower_records = [
            record for record in state.destroyed_followers if record.owner == owner
        ][-HISTORY_RECORDS_PER_PLAYER:]
        amulet_records = [
            record for record in state.destroyed_amulets if record.owner == owner
        ][-HISTORY_RECORDS_PER_PLAYER:]
        entry_records = [
            record for record in state.follower_entries if record.owner == owner
        ][-HISTORY_RECORDS_PER_PLAYER:]
        for name, records in (
            ("destroyed_follower", follower_records),
            ("destroyed_amulet", amulet_records),
            ("follower_entry", entry_records),
        ):
            pad = HISTORY_RECORDS_PER_PLAYER - len(records)
            card_fields[f"{name}_cards"].extend([0] * pad)
            row_size = 1 + 2 * ORIGIN_BITS + 3 + SEMANTIC_BITS
            if name == "follower_entry":
                row_size = 2 + SEMANTIC_BITS
            runtime_fields[f"{name}_state"].extend([0.0] * pad * row_size)
            for record in records:
                card_fields[f"{name}_cards"].append(
                    _card_index(env, record.definition)
                )
                if name == "follower_entry":
                    runtime_fields[f"{name}_state"].extend((
                        1.0,
                        record.entered_turn / 200,
                        *_semantic_bits(record.entry_cause),
                    ))
                else:
                    runtime_fields[f"{name}_state"].extend((
                        1.0,
                        *_origin_pair(record.origin, record.source_origin),
                        float(record.derived),
                        float(record.token),
                        record.destroyed_turn / 200,
                        *_semantic_bits((
                            record.cause.value,
                            getattr(record, "play_mode_id", None),
                            getattr(record, "summon_countdown", None),
                        )),
                    ))
    return card_fields, runtime_fields


def _deck_runtime(env: ShadowverseEnv, deck) -> tuple[float, ...]:
    rows = [[0.0, 0.0, 0.0, 0.0] for _ in env.card_vocabulary]
    for entry in deck:
        definition = _card_definition(entry)
        index = _card_index(env, definition)
        if not index:
            continue
        row = rows[index - 1]
        row[0] += float(isinstance(entry, DeckCard))
        row[1] += entry.cost / 60
        row[2] += (entry.attack or 0) / 120
        row[3] += (entry.life or 0) / 120
    return tuple(value for row in rows for value in row)


def encode_observation_v4(
    env: ShadowverseEnv,
    *,
    perspective: int | None = None,
    action_mask: Sequence[bool] | None = None,
    open_decklists: bool = False,
) -> dict[str, np.ndarray]:
    """Encode a fixed-shape public observation with semantic runtime state."""
    perspective = env.decision_player if perspective is None else perspective
    if action_mask is None and perspective != env.decision_player:
        action_mask = [False] * env.ACTION_SIZE
    me, opponent = _relative_players(env, perspective)
    hand = [*me.hand[: env.MAX_HAND]]
    hand.extend([None] * (env.MAX_HAND - len(hand)))
    boards = []
    for board in (me.board, opponent.board):
        slots = [*board[: env.MAX_BOARD]]
        slots.extend([None] * (env.MAX_BOARD - len(slots)))
        boards.extend(slots)

    choice = _choice_fields(env, perspective)
    history = _history_fields(env, perspective)
    leader_cards, leader_state, modifier_source_cards = _leader_area_fields(
        env, perspective
    )
    record_cards, record_runtime = _history_record_fields(env, perspective)
    initial_opponent = (
        env._initial_deck_histograms[1 - perspective]
        if open_decklists
        else (0,) * len(env.card_vocabulary)
    )
    result = {
        "player_state": _float_array((
            *_player_state(env, perspective),
            *_player_state(env, 1 - perspective),
        )),
        "player_class_bits": _float_array((
            *(
                float(me.class_id == class_id)
                for class_id in range(1, env.CLASS_COUNT + 1)
            ),
            *(
                float(opponent.class_id == class_id)
                for class_id in range(1, env.CLASS_COUNT + 1)
            ),
        )),
        "match_state": _float_array(_match_state(env, perspective)),
        "own_hand_cards": _int_array(
            0 if card is None else _card_index(env, card.definition)
            for card in hand
        ),
        "public_board_cards": _int_array(
            0 if entity is None else _card_index(env, entity.definition)
            for entity in boards
        ),
        "leader_area_cards": _int_array(leader_cards),
        "graveyard_page_cards": _int_array(choice["graveyard_cards"]),
        "choice_option_cards": _int_array(choice["cards"]),
        "history_source_cards": _int_array(history["source_cards"]),
        "history_target_cards": _int_array(history["target_cards"]),
        "destroyed_follower_cards": _int_array(
            record_cards["destroyed_follower_cards"]
        ),
        "destroyed_amulet_cards": _int_array(
            record_cards["destroyed_amulet_cards"]
        ),
        "follower_entry_cards": _int_array(
            record_cards["follower_entry_cards"]
        ),
        "own_hand_fusion_cards": _int_array(
            _fusion_card_indices(env, hand, me)
        ),
        "public_board_fusion_cards": _int_array((
            *_fusion_card_indices(env, boards[:env.MAX_BOARD], me),
            *_fusion_card_indices(env, boards[env.MAX_BOARD:], opponent),
        )),
        "leader_modifier_source_cards": _int_array(modifier_source_cards),
        "own_initial_deck": _int_array(
            env._initial_deck_histograms[perspective]
        ),
        "opponent_initial_deck": _int_array(initial_opponent),
        "own_current_deck": _int_array(
            _histogram(env, (_card_definition(entry) for entry in me.deck))
        ),
        "own_current_deck_runtime": _float_array(
            _deck_runtime(env, me.deck)
        ),
        "public_graveyards": _int_array((
            _histogram(env, (card.definition for card in me.graveyard)),
            _histogram(env, (card.definition for card in opponent.graveyard)),
        ), shape=(2, -1)),
        "public_banished": _int_array((
            _histogram(env, me.banished),
            _histogram(env, opponent.banished),
        ), shape=(2, -1)),
        "destroyed_follower_histograms": _int_array((
            _histogram(
                env,
                (
                    record.definition
                    for record in env._core.state.destroyed_followers
                    if record.owner == perspective
                ),
            ),
            _histogram(
                env,
                (
                    record.definition
                    for record in env._core.state.destroyed_followers
                    if record.owner == 1 - perspective
                ),
            ),
        ), shape=(2, -1)),
        "destroyed_amulet_histograms": _int_array((
            _histogram(
                env,
                (
                    record.definition
                    for record in env._core.state.destroyed_amulets
                    if record.owner == perspective
                ),
            ),
            _histogram(
                env,
                (
                    record.definition
                    for record in env._core.state.destroyed_amulets
                    if record.owner == 1 - perspective
                ),
            ),
        ), shape=(2, -1)),
        "follower_entry_histograms": _int_array((
            _histogram(
                env,
                (
                    record.definition
                    for record in env._core.state.follower_entries
                    if record.owner == perspective
                ),
            ),
            _histogram(
                env,
                (
                    record.definition
                    for record in env._core.state.follower_entries
                    if record.owner == 1 - perspective
                ),
            ),
        ), shape=(2, -1)),
        "own_hand_origin_bits": _float_array(
            value
            for card in hand
            for value in (
                (0.0,) * HAND_ORIGIN_SIZE
                if card is None
                else _origin_pair(card.origin, card.source_origin)
            )
        ),
        "public_board_origin_bits": _float_array(
            value
            for entity in boards
            for value in (
                (0.0,) * BOARD_ORIGIN_SIZE
                if entity is None
                else _origin_pair(entity.origin, entity.source_origin)
            )
        ),
        "own_hand_state": _float_array(
            value
            for card in hand
            for value in _hand_state(env, card, me.turns_started, env.turn)
        ),
        "own_hand_keyword_bits": _float_array(
            value
            for card in hand
            for value in (
                (0.0,) * HAND_KEYWORD_SIZE
                if card is None
                else tuple(
                    float(keyword in card.effective_keywords)
                    for keyword in RUNTIME_KEYWORDS
                )
            )
        ),
        "own_hand_modifier_state": _float_array(
            value
            for card in hand
            for value in _hand_modifier_state(card, perspective)
        ),
        "own_hand_effect_bits": _float_array(
            value for card in hand for value in _hand_effect_state(card)
        ),
        "public_board_state": _float_array(
            value for entity in boards for value in _board_state(entity, env.turn)
        ),
        "public_board_keyword_bits": _float_array(
            value
            for entity in boards
            for value in (
                (0.0,) * BOARD_KEYWORD_SIZE
                if entity is None
                else tuple(
                    float(
                        entity.has_keyword(keyword)
                        if isinstance(entity, Unit)
                        else keyword in entity.definition.keywords
                    )
                    for keyword in RUNTIME_KEYWORDS
                )
            )
        ),
        "public_board_modifier_state": _float_array(
            value
            for entity in boards
            for value in _board_modifier_state(entity, perspective)
        ),
        "public_board_effect_bits": _float_array(
            value for entity in boards for value in _board_effect_state(entity)
        ),
        "leader_area_state": _float_array(leader_state),
        "listener_state": _float_array(_listener_state(env, perspective)),
        "choice_state": _float_array(choice["state"]),
        "choice_option_state": _float_array(choice["options"]),
        "graveyard_option_state": _float_array(choice["graveyard_options"]),
        "history_event_bits": _float_array(history["event_bits"]),
        "history_actor_bits": _float_array(history["actor_bits"]),
        "history_amounts": _float_array(history["amounts"]),
        "history_reference_bits": _float_array(history["references"]),
        "history_semantic_bits": _float_array(history["semantics"]),
        "destroyed_follower_state": _float_array(
            record_runtime["destroyed_follower_state"]
        ),
        "destroyed_amulet_state": _float_array(
            record_runtime["destroyed_amulet_state"]
        ),
        "follower_entry_state": _float_array(
            record_runtime["follower_entry_state"]
        ),
        "action_mask": np.asarray(
            env.action_mask() if action_mask is None else action_mask,
            dtype=np.int8,
        ),
    }
    return result


def observation_v4_space(env: ShadowverseEnv) -> spaces.Dict:
    vocabulary_size = len(env.card_vocabulary)
    max_int = np.iinfo(np.int32).max

    def card_indices(shape):
        return spaces.Box(0, vocabulary_size, shape=shape, dtype=np.int32)

    def histogram(shape):
        return spaces.Box(0, max_int, shape=shape, dtype=np.int32)

    def continuous(shape):
        return spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

    record_count = 2 * HISTORY_RECORDS_PER_PLAYER
    destroyed_record_size = 1 + 2 * ORIGIN_BITS + 3 + SEMANTIC_BITS
    entry_record_size = 2 + SEMANTIC_BITS
    return spaces.Dict({
        "player_state": continuous((2 * PLAYER_STATE_SIZE,)),
        "player_class_bits": continuous((2 * env.CLASS_COUNT,)),
        "match_state": continuous((MATCH_STATE_SIZE,)),
        "own_hand_cards": card_indices((env.MAX_HAND,)),
        "public_board_cards": card_indices((2 * env.MAX_BOARD,)),
        "leader_area_cards": card_indices((4 * MAX_LEADER_AREA_SLOTS,)),
        "graveyard_page_cards": card_indices((env.GRAVEYARD_PAGE_SIZE,)),
        "choice_option_cards": card_indices((env.MAX_CHOICE_OPTIONS,)),
        "history_source_cards": card_indices((HISTORY_LENGTH,)),
        "history_target_cards": card_indices((HISTORY_LENGTH,)),
        "destroyed_follower_cards": card_indices((record_count,)),
        "destroyed_amulet_cards": card_indices((record_count,)),
        "follower_entry_cards": card_indices((record_count,)),
        "own_hand_fusion_cards": card_indices((env.MAX_HAND * 9,)),
        "public_board_fusion_cards": card_indices((2 * env.MAX_BOARD * 9,)),
        "leader_modifier_source_cards": card_indices(
            (2 * MAX_LEADER_DAMAGE_MODIFIERS,)
        ),
        "own_initial_deck": histogram((vocabulary_size,)),
        "opponent_initial_deck": histogram((vocabulary_size,)),
        "own_current_deck": histogram((vocabulary_size,)),
        "own_current_deck_runtime": continuous((vocabulary_size * 4,)),
        "public_graveyards": histogram((2, vocabulary_size)),
        "public_banished": histogram((2, vocabulary_size)),
        "destroyed_follower_histograms": histogram((2, vocabulary_size)),
        "destroyed_amulet_histograms": histogram((2, vocabulary_size)),
        "follower_entry_histograms": histogram((2, vocabulary_size)),
        "own_hand_origin_bits": continuous((env.MAX_HAND * HAND_ORIGIN_SIZE,)),
        "public_board_origin_bits": continuous(
            (2 * env.MAX_BOARD * BOARD_ORIGIN_SIZE,)
        ),
        "own_hand_state": continuous((env.MAX_HAND * HAND_STATE_SIZE,)),
        "own_hand_keyword_bits": continuous(
            (env.MAX_HAND * HAND_KEYWORD_SIZE,)
        ),
        "own_hand_modifier_state": continuous(
            (env.MAX_HAND * HAND_MODIFIER_SIZE,)
        ),
        "own_hand_effect_bits": continuous(
            (env.MAX_HAND * HAND_EFFECT_SIZE,)
        ),
        "public_board_state": continuous(
            (2 * env.MAX_BOARD * BOARD_STATE_SIZE,)
        ),
        "public_board_keyword_bits": continuous(
            (2 * env.MAX_BOARD * BOARD_KEYWORD_SIZE,)
        ),
        "public_board_modifier_state": continuous(
            (2 * env.MAX_BOARD * BOARD_MODIFIER_SIZE,)
        ),
        "public_board_effect_bits": continuous(
            (2 * env.MAX_BOARD * BOARD_EFFECT_SIZE,)
        ),
        "leader_area_state": continuous((LEADER_AREA_STATE_SIZE,)),
        "listener_state": continuous((LISTENER_STATE_SIZE,)),
        "choice_state": continuous((CHOICE_STATE_SIZE,)),
        "choice_option_state": continuous(
            (env.MAX_CHOICE_OPTIONS * CHOICE_OPTION_STATE_SIZE,)
        ),
        "graveyard_option_state": continuous(
            (env.GRAVEYARD_PAGE_SIZE * GRAVEYARD_OPTION_STATE_SIZE,)
        ),
        "history_event_bits": continuous(
            (HISTORY_LENGTH * HISTORY_EVENT_SIZE,)
        ),
        "history_actor_bits": continuous((HISTORY_LENGTH * RELATION_BITS,)),
        "history_amounts": continuous((HISTORY_LENGTH,)),
        "history_reference_bits": continuous(
            (HISTORY_LENGTH * HISTORY_REFERENCE_SIZE,)
        ),
        "history_semantic_bits": continuous(
            (HISTORY_LENGTH * SEMANTIC_BITS,)
        ),
        "destroyed_follower_state": continuous(
            (record_count * destroyed_record_size,)
        ),
        "destroyed_amulet_state": continuous(
            (record_count * destroyed_record_size,)
        ),
        "follower_entry_state": continuous(
            (record_count * entry_record_size,)
        ),
        "action_mask": spaces.MultiBinary(env.ACTION_SIZE),
    })
