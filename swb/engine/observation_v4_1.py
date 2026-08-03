from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from gymnasium import spaces

from swb.engine.commands import ChoiceKind
from swb.engine.origin import CardOrigin
from swb.engine.state import Amulet, DeckCard, Unit
from swb.engine import observation_v4 as v4

if TYPE_CHECKING:
    from swb.engine.environment import ShadowverseEnv


HISTORY_LENGTH = 32
HISTORY_RECORDS_PER_GROUP = 16
ZONE_GROUPS = 13
MAX_ZONE_CARD_KINDS = 64
MAX_DECK_CARDS = 40
MAX_HAND_MODIFIERS = (
    v4.MAX_COST_MODIFIERS
    + v4.MAX_STAT_MODIFIERS
    + 2 * v4.MAX_KEYWORD_MODIFIERS
)
MAX_BOARD_MODIFIERS = (
    v4.MAX_STAT_MODIFIERS
    + v4.MAX_ATTACK_CAPACITY_MODIFIERS
    + v4.MAX_ATTACK_RESTRICTIONS
    + v4.MAX_TARGETING_RESTRICTIONS
    + 2 * v4.MAX_KEYWORD_MODIFIERS
)
MAX_ENTITY_EFFECTS = 9
HAND_EFFECT_SUMMARY_SIZE = 2
BOARD_EFFECT_SUMMARY_SIZE = 4
LEADER_AREA_SLOTS = 4 * v4.MAX_LEADER_AREA_SLOTS
LEADER_EFFECTS_PER_SLOT = v4.MAX_FAITH_GRANTED_ABILITIES + 1
LEADER_EFFECT_SUMMARY_SIZE = 2
ENTITY_SLOTS = 9 + 2 * 5
LISTENER_SOURCE_COUNT = ENTITY_SLOTS + LEADER_AREA_SLOTS
LISTENER_FEATURE_SIZE = 2 * v4.MAX_LISTENERS_PER_SOURCE + 1
RECORD_GROUPS = 6
RECORD_COUNT = RECORD_GROUPS * HISTORY_RECORDS_PER_GROUP
STRUCTURED_TOKEN_COUNT = (
    1
    + 2
    + ENTITY_SLOTS
    + LEADER_AREA_SLOTS
    + ZONE_GROUPS
    + HISTORY_LENGTH
    + RECORD_GROUPS
)
SEMANTIC_TOKEN_SIZE = 5
EVENT_TYPE_COUNT = len(v4.EVENT_VALUES)
ORIGIN_COUNT = len(v4.ORIGIN_VALUES)
DURATION_COUNT = len(v4.DURATION_VALUES)
CHOICE_REFERENCE_COUNT = v4.CHOICE_REFERENCE_COUNT
ENTITY_BASE_SIZE = 3 + 2 * len(v4.RUNTIME_KEYWORDS)
HAND_BASE_SIZE = ENTITY_BASE_SIZE + v4.HAND_STATE_SIZE
BOARD_BASE_SIZE = ENTITY_BASE_SIZE + v4.BOARD_STATE_SIZE
MODIFIER_VALUE_SIZE = 4
LEADER_AREA_STATE_SIZE = 24
LEADER_MODIFIER_STATE_SIZE = 7
RECORD_STATE_SIZE = 9

MODIFIER_KINDS = {
    "cost": 1,
    "stat": 2,
    "keyword_grant": 3,
    "keyword_remove": 4,
    "attack_capacity": 5,
    "attack_restriction": 6,
    "targeting_restriction": 7,
}
MAX_MODIFIER_KIND = max(MODIFIER_KINDS.values())
MODIFIER_SUMMARY_SIZE = 2 * MAX_MODIFIER_KIND
MAX_MODIFIER_SUBTYPE = max(
    len(v4.RUNTIME_KEYWORDS),
    len(v4.COST_MODE_VALUES),
    len(v4.ATTACK_RESTRICTION_VALUES),
    len(v4.TARGETING_RESTRICTION_VALUES),
)


def _int(values, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = tuple(values)
    result = np.asarray(values, dtype=np.int32)
    return result if shape is None else result.reshape(shape)


def _float(values, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = tuple(values)
    result = np.asarray(values, dtype=np.float32)
    return result if shape is None else result.reshape(shape)


def _relation(value: int | None, perspective: int) -> int:
    if value not in (0, 1):
        return 0
    return 1 if value == perspective else 2


def _origin(value: CardOrigin | None) -> int:
    return int(v4.ORIGIN_INDEX.get(value, 0))


def _duration(value: str | None) -> int:
    return int(v4.DURATION_INDEX.get(value or "unknown", 0))


def _card_index(env: ShadowverseEnv, definition) -> int:
    return 0 if definition is None else env._v2_card_index.get(
        definition.card_id, 0
    )


def _normalize_semantic(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_semantic(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in {
                "entity_id",
                "source_id",
                "target_id",
                "request_id",
            }
        }
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _normalize_semantic(getattr(value, name))
            for name in value.__dataclass_fields__
            if name not in {"entity_id", "source_entity_id", "target_entity_id"}
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_semantic(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_normalize_semantic(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return type(value).__name__


def _semantic_token(value, *, kind: int = 1) -> tuple[int, ...]:
    payload = json.dumps(
        _normalize_semantic(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return (kind, digest[0], digest[1], digest[2], digest[3])


def _empty_semantic_tokens(count: int) -> list[int]:
    return [0] * count * SEMANTIC_TOKEN_SIZE


def _modifier_rows(entity, perspective: int, *, board: bool):
    rows: list[tuple[int, int, int, int, tuple[float, ...]]] = []

    def append(
        kind: str,
        subtype: int,
        duration: str | None,
        expiry: int | None,
        values: tuple[float, float, float, float],
    ) -> None:
        rows.append((
            MODIFIER_KINDS[kind],
            subtype,
            _duration(duration),
            _relation(expiry, perspective),
            values,
        ))

    if not board:
        for modifier in entity.cost_modifiers:
            append(
                "cost",
                v4.COST_MODE_INDEX.get(modifier.mode, 0),
                modifier.duration,
                modifier.expires_for_player,
                (modifier.amount / 20, 0.0, 0.0, 0.0),
            )
    for modifier in entity.stat_modifiers:
        append(
            "stat",
            0,
            modifier.duration,
            modifier.expires_for_player,
            (
                modifier.attack_delta / 40,
                modifier.health_delta / 40,
                0.0,
                0.0,
            ),
        )
    if board:
        for modifier in entity.attack_capacity_modifiers:
            append(
                "attack_capacity",
                0,
                modifier.duration,
                modifier.expires_for_player,
                (modifier.attacks_per_turn / 4, 0.0, 0.0, 0.0),
            )
        for modifier in entity.attack_restrictions:
            restriction = getattr(
                modifier.restriction, "value", modifier.restriction
            )
            append(
                "attack_restriction",
                (
                    v4.ATTACK_RESTRICTION_VALUES.index(restriction)
                    if restriction in v4.ATTACK_RESTRICTION_VALUES
                    else 0
                ),
                modifier.duration,
                modifier.expires_for_player,
                (0.0, 0.0, 0.0, 0.0),
            )
        for modifier in entity.targeting_restrictions:
            restriction = getattr(
                modifier.restriction, "value", modifier.restriction
            )
            append(
                "targeting_restriction",
                (
                    v4.TARGETING_RESTRICTION_VALUES.index(restriction)
                    if restriction in v4.TARGETING_RESTRICTION_VALUES
                    else 0
                ),
                modifier.duration,
                modifier.expires_for_player,
                (0.0, 0.0, 0.0, 0.0),
            )
    for modifier in entity.temporary_keywords:
        append(
            "keyword_grant",
            v4.KEYWORD_INDEX.get(modifier.keyword, -1) + 1,
            modifier.duration,
            modifier.expires_for_player,
            (0.0, 0.0, 0.0, 0.0),
        )
    for modifier in entity.temporary_keyword_removals:
        append(
            "keyword_remove",
            v4.KEYWORD_INDEX.get(modifier.keyword, -1) + 1,
            modifier.duration,
            modifier.expires_for_player,
            (
                0.0,
                0.0,
                float(modifier.restore_barrier_charge),
                float(modifier.restore_ambush),
            ),
        )
    return rows


def _modifier_fields(entities, perspective: int, *, board: bool):
    maximum = MAX_BOARD_MODIFIERS if board else MAX_HAND_MODIFIERS
    kinds: list[int] = []
    subtypes: list[int] = []
    durations: list[int] = []
    expiries: list[int] = []
    values: list[float] = []
    summaries: list[float] = []
    capacities = {
        MODIFIER_KINDS["cost"]: v4.MAX_COST_MODIFIERS,
        MODIFIER_KINDS["stat"]: v4.MAX_STAT_MODIFIERS,
        MODIFIER_KINDS["keyword_grant"]: v4.MAX_KEYWORD_MODIFIERS,
        MODIFIER_KINDS["keyword_remove"]: v4.MAX_KEYWORD_MODIFIERS,
        MODIFIER_KINDS["attack_capacity"]: v4.MAX_ATTACK_CAPACITY_MODIFIERS,
        MODIFIER_KINDS["attack_restriction"]: v4.MAX_ATTACK_RESTRICTIONS,
        MODIFIER_KINDS["targeting_restriction"]: (
            v4.MAX_TARGETING_RESTRICTIONS
        ),
    }
    for entity in entities:
        if entity is None or (board and not isinstance(entity, Unit)):
            rows = []
        else:
            rows = _modifier_rows(entity, perspective, board=board)
        selected_rows = []
        for kind in range(1, MAX_MODIFIER_KIND + 1):
            kind_rows = [row for row in rows if row[0] == kind]
            count = len(kind_rows)
            capacity = capacities[kind]
            summaries.extend((
                min(count, capacity) / max(1, capacity),
                float(count > capacity),
            ))
            selected_rows.extend(kind_rows[:capacity])
        for kind, subtype, duration, expiry, row_values in selected_rows:
            kinds.append(kind)
            subtypes.append(subtype)
            durations.append(duration)
            expiries.append(expiry)
            values.extend(row_values)
        padding = maximum - len(selected_rows)
        kinds.extend([0] * padding)
        subtypes.extend([0] * padding)
        durations.extend([0] * padding)
        expiries.extend([0] * padding)
        values.extend([0.0] * padding * MODIFIER_VALUE_SIZE)
    return {
        "kind": kinds,
        "subtype": subtypes,
        "duration": durations,
        "expiry": expiries,
        "values": values,
        "summary": summaries,
    }


def _entity_base(entity, state_values: Sequence[float]) -> tuple[float, ...]:
    if entity is None:
        return (0.0,) * (len(state_values) + ENTITY_BASE_SIZE)
    permanent_keywords = getattr(
        entity,
        "permanent_keywords",
        entity.definition.keywords,
    )
    removed_keywords = getattr(entity, "removed_keywords", ())
    permanent = tuple(
        float(keyword in permanent_keywords)
        for keyword in v4.RUNTIME_KEYWORDS
    )
    removed = tuple(
        float(keyword in removed_keywords)
        for keyword in v4.RUNTIME_KEYWORDS
    )
    return (
        *_origin_values(entity.origin, entity.source_origin),
        *state_values,
        *permanent,
        *removed,
    )


def _origin_values(
    origin: CardOrigin | None,
    source_origin: CardOrigin | None,
) -> tuple[float, float, float]:
    return (
        float(_origin(origin)),
        float(_origin(source_origin)),
        float(source_origin is not None),
    )


def _entity_effect_fields(hand, boards):
    hand_values: list[int] = []
    hand_summary: list[float] = []
    for card in hand:
        effects = () if card is None else card.granted_last_words
        hand_summary.extend((
            min(len(effects), v4.MAX_GRANTED_ABILITIES)
            / v4.MAX_GRANTED_ABILITIES,
            float(len(effects) > v4.MAX_GRANTED_ABILITIES),
        ))
        for effect in effects[:v4.MAX_GRANTED_ABILITIES]:
            hand_values.extend(_semantic_token(effect, kind=1))
        hand_values.extend(_empty_semantic_tokens(
            v4.MAX_GRANTED_ABILITIES
            - min(len(effects), v4.MAX_GRANTED_ABILITIES)
        ))

    board_values: list[int] = []
    board_summary: list[float] = []
    for entity in boards:
        tokens: list[tuple[int, ...]] = []
        if isinstance(entity, Unit):
            board_summary.extend((
                min(
                    len(entity.granted_last_words),
                    v4.MAX_GRANTED_ABILITIES,
                ) / v4.MAX_GRANTED_ABILITIES,
                float(
                    len(entity.granted_last_words)
                    > v4.MAX_GRANTED_ABILITIES
                ),
                min(
                    len(entity.granted_turn_end_abilities),
                    v4.MAX_GRANTED_ABILITIES,
                ) / v4.MAX_GRANTED_ABILITIES,
                float(
                    len(entity.granted_turn_end_abilities)
                    > v4.MAX_GRANTED_ABILITIES
                ),
            ))
            tokens.extend(
                _semantic_token(effect, kind=1)
                for effect in entity.granted_last_words[
                    :v4.MAX_GRANTED_ABILITIES
                ]
            )
            tokens.extend(
                _semantic_token(effect, kind=2)
                for effect in entity.granted_turn_end_abilities[
                    :v4.MAX_GRANTED_ABILITIES
                ]
            )
            tokens.append(_semantic_token(
                entity.random_choice_history,
                kind=3,
            ))
        else:
            board_summary.extend(
                (0.0,) * BOARD_EFFECT_SUMMARY_SIZE
            )
        for token in tokens[:MAX_ENTITY_EFFECTS]:
            board_values.extend(token)
        board_values.extend(_empty_semantic_tokens(
            MAX_ENTITY_EFFECTS - min(len(tokens), MAX_ENTITY_EFFECTS)
        ))
    return hand_values, hand_summary, board_values, board_summary


def _leader_area_fields(env: ShadowverseEnv, perspective: int):
    me = env._core.players[perspective]
    opponent = env._core.players[1 - perspective]
    collections = (
        me.faiths,
        opponent.faiths,
        me.emblems,
        opponent.emblems,
    )
    slots = []
    for collection in collections:
        slots.extend(collection[:v4.MAX_LEADER_AREA_SLOTS])
        slots.extend(
            [None] * (v4.MAX_LEADER_AREA_SLOTS - len(collection))
        )
    cards: list[int] = []
    state: list[float] = []
    effects: list[int] = []
    effect_summary: list[float] = []
    for slot_index, source in enumerate(slots):
        source_type = 1 if slot_index < 2 * v4.MAX_LEADER_AREA_SLOTS else 2
        owner_relation = 1 if (
            slot_index < v4.MAX_LEADER_AREA_SLOTS
            or 2 * v4.MAX_LEADER_AREA_SLOTS
            <= slot_index < 3 * v4.MAX_LEADER_AREA_SLOTS
        ) else 2
        cards.append(
            0 if source is None else env._v2_card_index.get(
                source.source_card_id, 0
            )
        )
        row = [0.0] * LEADER_AREA_STATE_SIZE
        if source is not None:
            row[0] = 1.0
            row[1] = float(source_type)
            row[2] = float(owner_relation)
            if source_type == 1:
                row[3] = source.value / 50
                row[4] = len(source.granted_abilities) / max(
                    1, v4.MAX_FAITH_GRANTED_ABILITIES
                )
                row[5] = source.mode_selection_bonus / 8
                tokens = [
                    _semantic_token(ability, kind=1)
                    for ability in source.granted_abilities[
                        :v4.MAX_FAITH_GRANTED_ABILITIES
                    ]
                ]
                effect_summary.extend((
                    min(
                        len(source.granted_abilities),
                        v4.MAX_FAITH_GRANTED_ABILITIES,
                    ) / v4.MAX_FAITH_GRANTED_ABILITIES,
                    float(
                        len(source.granted_abilities)
                        > v4.MAX_FAITH_GRANTED_ABILITIES
                    ),
                ))
            else:
                row[6] = (source.countdown or 0) / 20
                row[7] = (source.countdown_before or 0) / 20
                for index in range(v4.MAX_EMBLEM_TRIGGERS):
                    row[8 + index] = (
                        source.activation_counts.get(index, 0) / 10
                    )
                    row[16 + index] = float(
                        index in source._once_per_turn_used
                    )
                tokens = [
                    _semantic_token(source.random_choice_history, kind=2)
                ]
                effect_summary.extend(
                    (0.0,) * LEADER_EFFECT_SUMMARY_SIZE
                )
        else:
            tokens = []
            effect_summary.extend(
                (0.0,) * LEADER_EFFECT_SUMMARY_SIZE
            )
        state.extend(row)
        for token in tokens[:LEADER_EFFECTS_PER_SLOT]:
            effects.extend(token)
        effects.extend(_empty_semantic_tokens(
            LEADER_EFFECTS_PER_SLOT
            - min(len(tokens), LEADER_EFFECTS_PER_SLOT)
        ))
    return slots, cards, state, effects, effect_summary


def _listener_state(env: ShadowverseEnv, perspective: int, leader_slots):
    me = env._core.players[perspective]
    opponent = env._core.players[1 - perspective]
    sources = [
        *me.hand[:9],
        *([None] * (9 - len(me.hand[:9]))),
        *me.board[:5],
        *([None] * (5 - len(me.board[:5]))),
        *opponent.board[:5],
        *([None] * (5 - len(opponent.board[:5]))),
        *leader_slots,
    ]
    values: list[float] = []
    counts = env._core.state.listener_activation_counts
    used = env._core.state.listener_once_per_turn_used
    for source in sources:
        if source is None:
            values.extend([0.0] * LISTENER_FEATURE_SIZE)
            continue
        card_id = (
            source.definition.card_id
            if hasattr(source, "definition")
            and hasattr(source.definition, "card_id")
            else source.source_card_id
        )
        for index in range(v4.MAX_LISTENERS_PER_SOURCE):
            values.append(
                counts.get((source.entity_id, card_id, index), 0) / 10
            )
        for index in range(v4.MAX_LISTENERS_PER_SOURCE):
            values.append(float(
                (source.entity_id, card_id, index) in used
            ))
        values.append(float(any(
            key[0] == source.entity_id
            and key[1] == card_id
            and key[2] >= v4.MAX_LISTENERS_PER_SOURCE
            for key in counts
        )))
    return values


def _leader_modifier_fields(env: ShadowverseEnv, perspective: int):
    me = env._core.players[perspective]
    opponent = env._core.players[1 - perspective]
    source_cards: list[int] = []
    state: list[float] = []
    for player in (me, opponent):
        modifiers = player.leader_damage_modifiers
        for modifier in modifiers[:v4.MAX_LEADER_DAMAGE_MODIFIERS]:
            source_cards.append(
                env._v2_card_index.get(modifier.source_card_id or 0, 0)
            )
            state.extend((
                1.0,
                float({
                    "additive": 1,
                    "set_zero_if_positive": 2,
                }.get(modifier.mode, 0)),
                float(_duration(modifier.duration)),
                float(_relation(modifier.expires_for_player, perspective)),
                float(_relation(modifier.source_controller, perspective)),
                float(v4._leader_source_reference(
                    env, modifier.source_entity_id, perspective
                )),
                modifier.amount / 10,
            ))
        padding = v4.MAX_LEADER_DAMAGE_MODIFIERS - min(
            len(modifiers), v4.MAX_LEADER_DAMAGE_MODIFIERS
        )
        source_cards.extend([0] * padding)
        state.extend([0.0] * padding * LEADER_MODIFIER_STATE_SIZE)
    return source_cards, state


def _sparse_histogram(
    counts: Sequence[int],
) -> tuple[list[int], list[float], tuple[float, float]]:
    nonzero = [
        (index + 1, int(count))
        for index, count in enumerate(counts)
        if count
    ]
    kept = nonzero[:MAX_ZONE_CARD_KINDS]
    cards = [index for index, _ in kept]
    values = [count / 40 for _, count in kept]
    cards.extend([0] * (MAX_ZONE_CARD_KINDS - len(kept)))
    values.extend([0.0] * (MAX_ZONE_CARD_KINDS - len(kept)))
    dropped = nonzero[MAX_ZONE_CARD_KINDS:]
    return cards, values, (
        len(dropped) / max(1, len(counts)),
        sum(count for _, count in dropped) / 40,
    )


def _zone_fields(env: ShadowverseEnv, perspective: int, open_decklists: bool):
    me = env._core.players[perspective]
    opponent = env._core.players[1 - perspective]
    state = env._core.state
    hidden_opponent_deck = (0,) * len(env.card_vocabulary)
    histograms = [
        env._initial_deck_histograms[perspective],
        (
            env._initial_deck_histograms[1 - perspective]
            if open_decklists
            else hidden_opponent_deck
        ),
        v4._histogram(
            env, (v4._card_definition(entry) for entry in me.deck)
        ),
        v4._histogram(env, (card.definition for card in me.graveyard)),
        v4._histogram(
            env, (card.definition for card in opponent.graveyard)
        ),
        v4._histogram(env, me.banished),
        v4._histogram(env, opponent.banished),
        *(
            v4._histogram(
                env,
                (
                    record.definition
                    for record in state.destroyed_followers
                    if record.owner == owner
                ),
            )
            for owner in (perspective, 1 - perspective)
        ),
        *(
            v4._histogram(
                env,
                (
                    record.definition
                    for record in state.destroyed_amulets
                    if record.owner == owner
                ),
            )
            for owner in (perspective, 1 - perspective)
        ),
        *(
            v4._histogram(
                env,
                (
                    record.definition
                    for record in state.follower_entries
                    if record.owner == owner
                ),
            )
            for owner in (perspective, 1 - perspective)
        ),
    ]
    if len(histograms) != ZONE_GROUPS:
        raise AssertionError((len(histograms), ZONE_GROUPS))
    cards: list[int] = []
    counts: list[float] = []
    overflow: list[float] = []
    for histogram in histograms:
        group_cards, group_counts, group_overflow = _sparse_histogram(
            histogram
        )
        cards.extend(group_cards)
        counts.extend(group_counts)
        overflow.extend(group_overflow)
    return cards, counts, overflow


def _deck_fields(env: ShadowverseEnv, perspective: int):
    player = env._core.players[perspective]
    rows = []
    for entry in player.deck:
        definition = entry.definition if isinstance(entry, DeckCard) else entry
        rows.append((
            _card_index(env, definition),
            (
                entry.current_cost if isinstance(entry, DeckCard)
                else definition.cost
            ),
            (
                entry.attack if isinstance(entry, DeckCard)
                else definition.attack
            ) or 0,
            (
                entry.life if isinstance(entry, DeckCard)
                else definition.life
            ) or 0,
        ))
    rows.sort()
    cards: list[int] = []
    state: list[float] = []
    for card_index, cost, attack, life in rows[:MAX_DECK_CARDS]:
        cards.append(card_index)
        state.extend((1.0, cost / 20, attack / 40, life / 40))
    padding = MAX_DECK_CARDS - min(len(rows), MAX_DECK_CARDS)
    cards.extend([0] * padding)
    state.extend([0.0] * padding * 4)
    return cards, state


def _choice_fields(env: ShadowverseEnv, perspective: int):
    request = env._core.state.pending_choice
    normal_cards = [0] * env.MAX_CHOICE_OPTIONS
    references = [0] * env.MAX_CHOICE_OPTIONS
    relations = [0] * env.MAX_CHOICE_OPTIONS
    selected = [0.0] * env.MAX_CHOICE_OPTIONS
    semantics = _empty_semantic_tokens(env.MAX_CHOICE_OPTIONS)
    grave_cards = [0] * env.GRAVEYARD_PAGE_SIZE
    grave_state = [0.0] * env.GRAVEYARD_PAGE_SIZE * 5
    grave_semantics = _empty_semantic_tokens(env.GRAVEYARD_PAGE_SIZE)
    if request is None or perspective != env.decision_player:
        return {
            "kind": 0,
            "state": (0.0,) * 5,
            "cards": normal_cards,
            "references": references,
            "relations": relations,
            "selected": selected,
            "semantics": semantics,
            "grave_cards": grave_cards,
            "grave_state": grave_state,
            "grave_semantics": grave_semantics,
        }
    state = (
        1.0,
        len(request.options) / 64,
        request.target_count / env.MAX_CHOICE_OPTIONS,
        len(request.selected_options) / env.MAX_CHOICE_OPTIONS,
        float(request.allow_duplicate_targets),
    )
    selected_ids = {option.option_id for option in request.selected_options}
    if request.choice_kind is not ChoiceKind.GRAVEYARD:
        semantics = []
        for index, option in enumerate(
            request.options[:env.MAX_CHOICE_OPTIONS]
        ):
            normal_cards[index] = _card_index(
                env,
                v4._choice_option_definition(env, option, perspective),
            )
            references[index] = v4._public_entity_reference(
                env, option.entity_id, perspective
            )
            relations[index] = _relation(
                option.leader_player_index, perspective
            )
            selected[index] = float(option.option_id in selected_ids)
            semantics.extend(_semantic_token(
                (
                    request.choice_kind.value,
                    "entity-or-leader",
                )
                if option.entity_id is not None
                or option.leader_player_index is not None
                else (
                    request.choice_kind.value,
                    option.option_id,
                    option.label,
                )
            ))
        semantics.extend(_empty_semantic_tokens(
            env.MAX_CHOICE_OPTIONS
            - min(len(request.options), env.MAX_CHOICE_OPTIONS)
        ))
    else:
        start = env._graveyard_page * env.GRAVEYARD_PAGE_SIZE
        options = request.options[start:start + env.GRAVEYARD_PAGE_SIZE]
        by_entity = {
            card.entity_id: card
            for player in env._core.players
            for card in player.graveyard
        }
        grave_semantics = []
        for index, option in enumerate(options):
            card = by_entity.get(option.entity_id)
            if card is None:
                grave_semantics.extend((0,) * SEMANTIC_TOKEN_SIZE)
                continue
            grave_cards[index] = _card_index(env, card.definition)
            row = (
                1.0,
                float(_origin(card.origin)),
                float(_origin(card.source_origin)),
                float(card.derived),
                float(card.token),
            )
            row_start = index * 5
            grave_state[row_start:row_start + 5] = row
            grave_semantics.extend(_semantic_token(
                (card.entry_cause, option.label)
            ))
        grave_semantics.extend(_empty_semantic_tokens(
            env.GRAVEYARD_PAGE_SIZE - len(options)
        ))
    return {
        "kind": v4.CHOICE_KIND_INDEX[request.choice_kind] + 1,
        "state": state,
        "cards": normal_cards,
        "references": references,
        "relations": relations,
        "selected": selected,
        "semantics": semantics,
        "grave_cards": grave_cards,
        "grave_state": grave_state,
        "grave_semantics": grave_semantics,
    }


def _history_fields(env: ShadowverseEnv, perspective: int):
    events = [
        event
        for event in env._core.event_history
        if event.type in v4.EVENT_INDEX
    ][-HISTORY_LENGTH:]
    padding = HISTORY_LENGTH - len(events)
    types = [0] * padding
    actors = [0] * padding
    amounts = [0.0] * padding
    source_references = [0] * padding
    target_references = [0] * padding
    semantics = _empty_semantic_tokens(padding)
    source_cards = [0] * padding
    target_cards = [0] * padding
    for event in events:
        types.append(v4.EVENT_INDEX[event.type] + 1)
        actors.append(_relation(event.player_index, perspective))
        amounts.append(max(-40, min(40, event.amount)) / 40)
        source_references.append(v4._public_entity_reference(
            env, event.source_id, perspective
        ))
        target_references.append(v4._public_entity_reference(
            env, event.target_id, perspective
        ))
        can_reveal = (
            event.player_index == perspective
            or event.type in v4.PUBLIC_CARD_ID_EVENTS
        )
        metadata = (
            {
                key: value
                for key, value in event.metadata.items()
                if key in v4.PUBLIC_METADATA_KEYS
            }
            if can_reveal else {}
        )
        semantics.extend(_semantic_token((event.type.value, metadata)))
        source_cards.append(v4._event_public_card_id(
            env, event, perspective, target=False
        ))
        target_cards.append(v4._event_public_card_id(
            env, event, perspective, target=True
        ))
    return {
        "types": types,
        "actors": actors,
        "amounts": amounts,
        "source_references": source_references,
        "target_references": target_references,
        "semantics": semantics,
        "source_cards": source_cards,
        "target_cards": target_cards,
    }


def _record_fields(env: ShadowverseEnv, perspective: int):
    state = env._core.state
    groups = []
    for owner in (perspective, 1 - perspective):
        groups.append((
            1,
            [
                record for record in state.destroyed_followers
                if record.owner == owner
            ][-HISTORY_RECORDS_PER_GROUP:],
        ))
    for owner in (perspective, 1 - perspective):
        groups.append((
            2,
            [
                record for record in state.destroyed_amulets
                if record.owner == owner
            ][-HISTORY_RECORDS_PER_GROUP:],
        ))
    for owner in (perspective, 1 - perspective):
        groups.append((
            3,
            [
                record for record in state.follower_entries
                if record.owner == owner
            ][-HISTORY_RECORDS_PER_GROUP:],
        ))
    cards: list[int] = []
    values: list[float] = []
    semantics: list[int] = []
    for group_index, (record_kind, records) in enumerate(groups):
        padding = HISTORY_RECORDS_PER_GROUP - len(records)
        cards.extend([0] * padding)
        values.extend([0.0] * padding * RECORD_STATE_SIZE)
        semantics.extend(_empty_semantic_tokens(padding))
        owner_relation = 1 if group_index % 2 == 0 else 2
        for record in records:
            cards.append(_card_index(env, record.definition))
            if record_kind == 3:
                values.extend((
                    1.0,
                    float(record_kind),
                    float(owner_relation),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    record.entered_turn / 200,
                    0.0,
                ))
                semantics.extend(_semantic_token(record.entry_cause))
            else:
                values.extend((
                    1.0,
                    float(record_kind),
                    float(owner_relation),
                    float(_origin(record.origin)),
                    float(_origin(record.source_origin)),
                    float(record.derived),
                    float(record.token),
                    record.destroyed_turn / 200,
                    0.0,
                ))
                semantics.extend(_semantic_token((
                    record.cause.value,
                    getattr(record, "play_mode_id", None),
                    getattr(record, "summon_countdown", None),
                )))
    return cards, values, semantics


def _fusion_cards(env: ShadowverseEnv, entities, player):
    return v4._fusion_card_indices(env, entities, player)


def encode_observation_v4_1(
    env: ShadowverseEnv,
    *,
    perspective: int | None = None,
    action_mask: Sequence[bool] | None = None,
    open_decklists: bool = False,
) -> dict[str, np.ndarray]:
    perspective = env.decision_player if perspective is None else perspective
    if action_mask is None and perspective != env.decision_player:
        action_mask = [False] * env.ACTION_SIZE
    me = env._core.players[perspective]
    opponent = env._core.players[1 - perspective]
    hand = [*me.hand[:env.MAX_HAND]]
    hand.extend([None] * (env.MAX_HAND - len(hand)))
    boards = []
    for board in (me.board, opponent.board):
        slots = [*board[:env.MAX_BOARD]]
        slots.extend([None] * (env.MAX_BOARD - len(slots)))
        boards.extend(slots)

    hand_modifiers = _modifier_fields(hand, perspective, board=False)
    board_modifiers = _modifier_fields(boards, perspective, board=True)
    (
        hand_effects,
        hand_effect_summary,
        board_effects,
        board_effect_summary,
    ) = _entity_effect_fields(hand, boards)
    (
        leader_slots,
        leader_cards,
        leader_state,
        leader_effects,
        leader_effect_summary,
    ) = (
        _leader_area_fields(env, perspective)
    )
    leader_modifier_cards, leader_modifier_state = _leader_modifier_fields(
        env, perspective
    )
    zone_cards, zone_counts, zone_overflow = _zone_fields(
        env, perspective, open_decklists
    )
    deck_cards, deck_state = _deck_fields(env, perspective)
    choice = _choice_fields(env, perspective)
    history = _history_fields(env, perspective)
    record_cards, record_state, record_semantics = _record_fields(
        env, perspective
    )

    return {
        "player_state": _float((
            *v4._player_state(env, perspective),
            *v4._player_state(env, 1 - perspective),
        )),
        "player_class": _int((me.class_id, opponent.class_id)),
        "match_state": _float(v4._match_state(env, perspective)),
        "own_hand_cards": _int(
            0 if card is None else _card_index(env, card.definition)
            for card in hand
        ),
        "public_board_cards": _int(
            0 if entity is None else _card_index(env, entity.definition)
            for entity in boards
        ),
        "own_hand_base": _float(
            value
            for card in hand
            for value in _entity_base(
                card,
                v4._hand_state(env, card, me.turns_started, env.turn),
            )
        ),
        "public_board_base": _float(
            value
            for entity in boards
            for value in _entity_base(
                entity,
                v4._board_state(entity, env.turn),
            )
        ),
        "own_hand_keywords": _float(
            value
            for card in hand
            for value in (
                (0.0,) * len(v4.RUNTIME_KEYWORDS)
                if card is None
                else tuple(
                    float(keyword in card.effective_keywords)
                    for keyword in v4.RUNTIME_KEYWORDS
                )
            )
        ),
        "public_board_keywords": _float(
            value
            for entity in boards
            for value in (
                (0.0,) * len(v4.RUNTIME_KEYWORDS)
                if entity is None
                else tuple(
                    float(
                        entity.has_keyword(keyword)
                        if isinstance(entity, Unit)
                        else keyword in entity.definition.keywords
                    )
                    for keyword in v4.RUNTIME_KEYWORDS
                )
            )
        ),
        "hand_modifier_kind": _int(hand_modifiers["kind"]),
        "hand_modifier_subtype": _int(hand_modifiers["subtype"]),
        "hand_modifier_duration": _int(hand_modifiers["duration"]),
        "hand_modifier_expiry": _int(hand_modifiers["expiry"]),
        "hand_modifier_values": _float(hand_modifiers["values"]),
        "hand_modifier_summary": _float(hand_modifiers["summary"]),
        "board_modifier_kind": _int(board_modifiers["kind"]),
        "board_modifier_subtype": _int(board_modifiers["subtype"]),
        "board_modifier_duration": _int(board_modifiers["duration"]),
        "board_modifier_expiry": _int(board_modifiers["expiry"]),
        "board_modifier_values": _float(board_modifiers["values"]),
        "board_modifier_summary": _float(board_modifiers["summary"]),
        "hand_effect_tokens": _int(hand_effects),
        "hand_effect_summary": _float(hand_effect_summary),
        "board_effect_tokens": _int(board_effects),
        "board_effect_summary": _float(board_effect_summary),
        "own_hand_fusion_cards": _int(
            _fusion_cards(env, hand, me)
        ),
        "public_board_fusion_cards": _int((
            *_fusion_cards(env, boards[:env.MAX_BOARD], me),
            *_fusion_cards(env, boards[env.MAX_BOARD:], opponent),
        )),
        "leader_area_cards": _int(leader_cards),
        "leader_area_state": _float(leader_state),
        "leader_effect_tokens": _int(leader_effects),
        "leader_effect_summary": _float(leader_effect_summary),
        "listener_state": _float(_listener_state(
            env, perspective, leader_slots
        )),
        "leader_modifier_source_cards": _int(leader_modifier_cards),
        "leader_modifier_state": _float(leader_modifier_state),
        "zone_cards": _int(zone_cards),
        "zone_counts": _float(zone_counts),
        "zone_overflow": _float(zone_overflow),
        "own_deck_cards": _int(deck_cards),
        "own_deck_state": _float(deck_state),
        "choice_kind": _int((choice["kind"],)),
        "choice_state": _float(choice["state"]),
        "choice_option_cards": _int(choice["cards"]),
        "choice_option_references": _int(choice["references"]),
        "choice_option_relations": _int(choice["relations"]),
        "choice_option_selected": _float(choice["selected"]),
        "choice_option_semantics": _int(choice["semantics"]),
        "graveyard_page_cards": _int(choice["grave_cards"]),
        "graveyard_option_state": _float(choice["grave_state"]),
        "graveyard_option_semantics": _int(choice["grave_semantics"]),
        "history_event_types": _int(history["types"]),
        "history_actors": _int(history["actors"]),
        "history_amounts": _float(history["amounts"]),
        "history_source_references": _int(history["source_references"]),
        "history_target_references": _int(history["target_references"]),
        "history_semantics": _int(history["semantics"]),
        "history_source_cards": _int(history["source_cards"]),
        "history_target_cards": _int(history["target_cards"]),
        "record_cards": _int(record_cards),
        "record_state": _float(record_state),
        "record_semantics": _int(record_semantics),
        "action_mask": np.asarray(
            env.action_mask() if action_mask is None else action_mask,
            dtype=np.int8,
        ),
    }


def observation_v4_1_space(env: ShadowverseEnv) -> spaces.Dict:
    vocabulary_size = len(env.card_vocabulary)

    def card(shape):
        return spaces.Box(0, vocabulary_size, shape=shape, dtype=np.int32)

    def categorical(high, shape):
        return spaces.Box(0, high, shape=shape, dtype=np.int32)

    def continuous(shape):
        return spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

    hand_modifier_count = env.MAX_HAND * MAX_HAND_MODIFIERS
    board_modifier_count = 2 * env.MAX_BOARD * MAX_BOARD_MODIFIERS
    return spaces.Dict({
        "player_state": continuous((2 * v4.PLAYER_STATE_SIZE,)),
        "player_class": categorical(env.CLASS_COUNT, (2,)),
        "match_state": continuous((v4.MATCH_STATE_SIZE,)),
        "own_hand_cards": card((env.MAX_HAND,)),
        "public_board_cards": card((2 * env.MAX_BOARD,)),
        "own_hand_base": continuous((env.MAX_HAND * HAND_BASE_SIZE,)),
        "public_board_base": continuous(
            (2 * env.MAX_BOARD * BOARD_BASE_SIZE,)
        ),
        "own_hand_keywords": continuous(
            (env.MAX_HAND * len(v4.RUNTIME_KEYWORDS),)
        ),
        "public_board_keywords": continuous(
            (2 * env.MAX_BOARD * len(v4.RUNTIME_KEYWORDS),)
        ),
        "hand_modifier_kind": categorical(
            MAX_MODIFIER_KIND, (hand_modifier_count,)
        ),
        "hand_modifier_subtype": categorical(
            MAX_MODIFIER_SUBTYPE, (hand_modifier_count,)
        ),
        "hand_modifier_duration": categorical(
            len(v4.DURATION_VALUES), (hand_modifier_count,)
        ),
        "hand_modifier_expiry": categorical(2, (hand_modifier_count,)),
        "hand_modifier_values": continuous(
            (hand_modifier_count * MODIFIER_VALUE_SIZE,)
        ),
        "hand_modifier_summary": continuous(
            (env.MAX_HAND * MODIFIER_SUMMARY_SIZE,)
        ),
        "board_modifier_kind": categorical(
            MAX_MODIFIER_KIND, (board_modifier_count,)
        ),
        "board_modifier_subtype": categorical(
            MAX_MODIFIER_SUBTYPE, (board_modifier_count,)
        ),
        "board_modifier_duration": categorical(
            len(v4.DURATION_VALUES), (board_modifier_count,)
        ),
        "board_modifier_expiry": categorical(2, (board_modifier_count,)),
        "board_modifier_values": continuous(
            (board_modifier_count * MODIFIER_VALUE_SIZE,)
        ),
        "board_modifier_summary": continuous(
            (2 * env.MAX_BOARD * MODIFIER_SUMMARY_SIZE,)
        ),
        "hand_effect_tokens": categorical(
            255,
            (
                env.MAX_HAND
                * v4.MAX_GRANTED_ABILITIES
                * SEMANTIC_TOKEN_SIZE,
            ),
        ),
        "hand_effect_summary": continuous(
            (env.MAX_HAND * HAND_EFFECT_SUMMARY_SIZE,)
        ),
        "board_effect_tokens": categorical(
            255,
            (
                2 * env.MAX_BOARD
                * MAX_ENTITY_EFFECTS
                * SEMANTIC_TOKEN_SIZE,
            ),
        ),
        "board_effect_summary": continuous(
            (2 * env.MAX_BOARD * BOARD_EFFECT_SUMMARY_SIZE,)
        ),
        "own_hand_fusion_cards": card((env.MAX_HAND * 9,)),
        "public_board_fusion_cards": card((2 * env.MAX_BOARD * 9,)),
        "leader_area_cards": card((LEADER_AREA_SLOTS,)),
        "leader_area_state": continuous(
            (LEADER_AREA_SLOTS * LEADER_AREA_STATE_SIZE,)
        ),
        "leader_effect_tokens": categorical(
            255,
            (
                LEADER_AREA_SLOTS
                * LEADER_EFFECTS_PER_SLOT
                * SEMANTIC_TOKEN_SIZE,
            ),
        ),
        "leader_effect_summary": continuous(
            (LEADER_AREA_SLOTS * LEADER_EFFECT_SUMMARY_SIZE,)
        ),
        "listener_state": continuous(
            (LISTENER_SOURCE_COUNT * LISTENER_FEATURE_SIZE,)
        ),
        "leader_modifier_source_cards": card(
            (2 * v4.MAX_LEADER_DAMAGE_MODIFIERS,)
        ),
        "leader_modifier_state": continuous((
            2 * v4.MAX_LEADER_DAMAGE_MODIFIERS
            * LEADER_MODIFIER_STATE_SIZE,
        )),
        "zone_cards": card((ZONE_GROUPS * MAX_ZONE_CARD_KINDS,)),
        "zone_counts": continuous((ZONE_GROUPS * MAX_ZONE_CARD_KINDS,)),
        "zone_overflow": continuous((ZONE_GROUPS * 2,)),
        "own_deck_cards": card((MAX_DECK_CARDS,)),
        "own_deck_state": continuous((MAX_DECK_CARDS * 4,)),
        "choice_kind": categorical(len(v4.CHOICE_KIND_VALUES), (1,)),
        "choice_state": continuous((5,)),
        "choice_option_cards": card((env.MAX_CHOICE_OPTIONS,)),
        "choice_option_references": categorical(
            v4.CHOICE_REFERENCE_COUNT - 1,
            (env.MAX_CHOICE_OPTIONS,),
        ),
        "choice_option_relations": categorical(
            2, (env.MAX_CHOICE_OPTIONS,)
        ),
        "choice_option_selected": continuous((env.MAX_CHOICE_OPTIONS,)),
        "choice_option_semantics": categorical(
            255, (env.MAX_CHOICE_OPTIONS * SEMANTIC_TOKEN_SIZE,)
        ),
        "graveyard_page_cards": card((env.GRAVEYARD_PAGE_SIZE,)),
        "graveyard_option_state": continuous(
            (env.GRAVEYARD_PAGE_SIZE * 5,)
        ),
        "graveyard_option_semantics": categorical(
            255, (env.GRAVEYARD_PAGE_SIZE * SEMANTIC_TOKEN_SIZE,)
        ),
        "history_event_types": categorical(
            len(v4.EVENT_VALUES), (HISTORY_LENGTH,)
        ),
        "history_actors": categorical(2, (HISTORY_LENGTH,)),
        "history_amounts": continuous((HISTORY_LENGTH,)),
        "history_source_references": categorical(
            v4.CHOICE_REFERENCE_COUNT - 1, (HISTORY_LENGTH,)
        ),
        "history_target_references": categorical(
            v4.CHOICE_REFERENCE_COUNT - 1, (HISTORY_LENGTH,)
        ),
        "history_semantics": categorical(
            255, (HISTORY_LENGTH * SEMANTIC_TOKEN_SIZE,)
        ),
        "history_source_cards": card((HISTORY_LENGTH,)),
        "history_target_cards": card((HISTORY_LENGTH,)),
        "record_cards": card((RECORD_COUNT,)),
        "record_state": continuous((RECORD_COUNT * RECORD_STATE_SIZE,)),
        "record_semantics": categorical(
            255, (RECORD_COUNT * SEMANTIC_TOKEN_SIZE,)
        ),
        "action_mask": spaces.MultiBinary(env.ACTION_SIZE),
    })
