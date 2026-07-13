from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from swb.engine.abilities import RUNTIME_UNIT_KEYWORDS
from swb.engine.commands import ChoiceKind
from swb.engine.events import EventType
from swb.engine.effects import ModifierDuration
from swb.engine.origin import CardOrigin
from swb.engine.state import Amulet, HandCard, Unit

if TYPE_CHECKING:
    from swb.engine.environment import ShadowverseEnv


HISTORY_LENGTH = 16
MAX_LEADER_AREA_SLOTS = 5
MAX_LEADER_DAMAGE_MODIFIERS = 8
ORIGIN_INDEX = {origin: index + 1 for index, origin in enumerate(CardOrigin)}
EVENT_INDEX = {event: index + 1 for index, event in enumerate(EventType)}
CHOICE_KIND_INDEX = {kind: index + 1 for index, kind in enumerate(ChoiceKind)}
RUNTIME_KEYWORDS = tuple(sorted(RUNTIME_UNIT_KEYWORDS))
MODIFIER_DURATION_INDEX = {
    duration.value: index + 1 for index, duration in enumerate(ModifierDuration)
}


def _card_index(env: ShadowverseEnv, card_id: int) -> int:
    return env._v2_card_index.get(card_id, 0)


def _origin_index(origin: CardOrigin | None) -> int:
    return 0 if origin is None else ORIGIN_INDEX.get(origin, 0)


def _histogram(env: ShadowverseEnv, definitions) -> tuple[int, ...]:
    counts = Counter(_card_index(env, definition.card_id) for definition in definitions)
    counts.pop(0, None)
    return tuple(counts.get(index, 0) for index in range(1, len(env.card_vocabulary) + 1))


def _hand_runtime(card: HandCard | None, turn: int) -> tuple[float, ...]:
    if card is None:
        return (0.0,) * 10
    return (
        1.0,
        min(card.current_cost, 20) / 20,
        min(len(card.cost_modifiers), 10) / 10,
        min(card.spellboost_count, 20) / 20,
        min(card.spellboost_cost_reduction, 10) / 10,
        min(len(card.fused_material_ids), 9) / 9,
        float(card.fusion_used_turn == turn),
        min(card.evolutions_while_in_hand, 15) / 15,
        float(card.cannot_be_played),
        float(card.definition.is_collectible),
    )


def _board_runtime(entity) -> tuple[float, ...]:
    if entity is None:
        return (0.0,) * 12
    if isinstance(entity, Amulet):
        return (
            1.0,
            1.0,
            min(entity.countdown or 0, 20) / 20,
            min(entity.earth_sigil_count, 20) / 20,
            float(entity.pending_destroy),
            min(len(entity.fused_material_ids), 9) / 9,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    return (
        1.0,
        0.0,
        min(entity.attack, 40) / 40,
        min(max(entity.health, 0), 40) / 40,
        min(entity.max_health, 40) / 40,
        min(entity.attacks_remaining, 4) / 4,
        min(len(entity.stat_modifiers), 10) / 10,
        min(len(entity.attack_restrictions), 3) / 3,
        min(len(entity.targeting_restrictions), 3) / 3,
        float(entity.printed_abilities_removed),
        float(entity.evolved),
        float(entity.super_evolved),
    )


def _option_reference(env: ShadowverseEnv, entity_id: int | None) -> int:
    if entity_id is None:
        return 0
    perspective = env.decision_player
    me = env.players[perspective]
    opponent = env.players[1 - perspective]
    for index, entity in enumerate(me.board):
        if entity.entity_id == entity_id:
            return 1 + index
    for index, entity in enumerate(opponent.board):
        if entity.entity_id == entity_id:
            return 1 + env.MAX_BOARD + index
    for index, card in enumerate(me.hand):
        if card.entity_id == entity_id:
            return 1 + 2 * env.MAX_BOARD + index
    for index, card in enumerate(me.graveyard[: env.GRAVEYARD_PAGE_SIZE]):
        if card.entity_id == entity_id:
            return 1 + 2 * env.MAX_BOARD + env.MAX_HAND + index
    return 0


def _choice_features(env: ShadowverseEnv) -> dict[str, object]:
    request = env.core.state.pending_choice
    if request is None:
        return {
            "kind": 0,
            "option_count": 0,
            "target_count": 0,
            "selected_count": 0,
            "option_references": (0,) * env.MAX_CHOICE_OPTIONS,
            "option_leader_relations": (0,) * env.MAX_CHOICE_OPTIONS,
        }
    references = []
    leader_relations = []
    for option in request.options[: env.MAX_CHOICE_OPTIONS]:
        references.append(_option_reference(env, option.entity_id))
        if option.leader_player_index is None:
            leader_relations.append(0)
        elif option.leader_player_index == env.decision_player:
            leader_relations.append(1)
        else:
            leader_relations.append(2)
    padding = env.MAX_CHOICE_OPTIONS - len(references)
    return {
        "kind": CHOICE_KIND_INDEX.get(request.choice_kind, 0),
        "option_count": min(len(request.options), env.MAX_CHOICE_OPTIONS),
        "target_count": request.target_count,
        "selected_count": len(request.selected_options),
        "option_references": tuple(references + [0] * padding),
        "option_leader_relations": tuple(leader_relations + [0] * padding),
    }


def _history_features(env: ShadowverseEnv) -> dict[str, tuple]:
    events = env.core.event_history[-HISTORY_LENGTH:]
    event_types = []
    actor_relations = []
    amounts = []
    for event in events:
        event_types.append(EVENT_INDEX[event.type])
        if event.player_index not in (0, 1):
            actor_relations.append(0)
        elif event.player_index == env.decision_player:
            actor_relations.append(1)
        else:
            actor_relations.append(2)
        amounts.append(max(-20, min(20, event.amount)) / 20)
    padding = HISTORY_LENGTH - len(events)
    return {
        "event_types": tuple([0] * padding + event_types),
        "actor_relations": tuple([0] * padding + actor_relations),
        "amounts": tuple([0.0] * padding + amounts),
    }


def _leader_modifier_runtime(env: ShadowverseEnv, player) -> tuple[float, ...]:
    values = []
    public_entity_ids = {
        entity.entity_id
        for board_owner in env.players
        for entity in board_owner.board
    }
    for modifier in player.leader_damage_modifiers[:MAX_LEADER_DAMAGE_MODIFIERS]:
        if modifier.expires_for_player not in (0, 1):
            expiry_relation = 0
        elif modifier.expires_for_player == env.decision_player:
            expiry_relation = 1
        else:
            expiry_relation = 2
        if modifier.source_controller not in (0, 1):
            source_relation = 0
        elif modifier.source_controller == env.decision_player:
            source_relation = 1
        else:
            source_relation = 2
        values.extend((
            max(-10, min(10, modifier.amount)) / 10,
            float(MODIFIER_DURATION_INDEX.get(modifier.duration, 0)),
            float(expiry_relation),
            float(source_relation),
            float(
                modifier.source_entity_id is not None
                and modifier.source_entity_id in public_entity_ids
            ),
        ))
    padding = MAX_LEADER_DAMAGE_MODIFIERS - min(
        len(player.leader_damage_modifiers), MAX_LEADER_DAMAGE_MODIFIERS
    )
    values.extend([0.0] * padding * 5)
    return tuple(values)


def encode_observation_v2(env: ShadowverseEnv) -> dict[str, object]:
    perspective = env.decision_player
    me = env.players[perspective]
    opponent = env.players[1 - perspective]
    hand = [*me.hand[: env.MAX_HAND]]
    hand.extend([None] * (env.MAX_HAND - len(hand)))
    boards = []
    for board in (me.board, opponent.board):
        slots = [*board[: env.MAX_BOARD]]
        slots.extend([None] * (env.MAX_BOARD - len(slots)))
        boards.extend(slots)
    faiths = []
    emblems = []
    for player in (me, opponent):
        faith_slots = [*player.faiths[:MAX_LEADER_AREA_SLOTS]]
        faith_slots.extend([None] * (MAX_LEADER_AREA_SLOTS - len(faith_slots)))
        faiths.extend(faith_slots)
        emblem_slots = [*player.emblems[:MAX_LEADER_AREA_SLOTS]]
        emblem_slots.extend([None] * (MAX_LEADER_AREA_SLOTS - len(emblem_slots)))
        emblems.extend(emblem_slots)

    keyword_rows = []
    for entity in boards:
        if isinstance(entity, Unit):
            keyword_rows.extend(
                float(entity.has_keyword(keyword)) for keyword in RUNTIME_KEYWORDS
            )
        elif isinstance(entity, Amulet):
            keyword_rows.extend(
                float(keyword in entity.definition.keywords)
                for keyword in RUNTIME_KEYWORDS
            )
        else:
            keyword_rows.extend([0.0] * len(RUNTIME_KEYWORDS))

    return {
        "version": 2,
        "continuous_v1": tuple(env._observation_v1()),
        "card_indices": {
            "own_hand": tuple(
                0 if card is None else _card_index(env, card.card_id)
                for card in hand
            ),
            "public_board": tuple(
                0 if entity is None else _card_index(env, entity.definition.card_id)
                for entity in boards
            ),
            "initial_decks": (
                _histogram(env, env.deck_lists[perspective]),
                _histogram(env, env.deck_lists[1 - perspective]),
            ),
            "public_graveyards": (
                _histogram(env, (card.definition for card in me.graveyard)),
                _histogram(env, (card.definition for card in opponent.graveyard)),
            ),
            "public_banished": (
                _histogram(env, me.banished),
                _histogram(env, opponent.banished),
            ),
        },
        "origins": {
            "own_hand": tuple(
                0 if card is None else _origin_index(card.origin)
                for card in hand
            ),
            "public_board": tuple(
                0 if entity is None else _origin_index(entity.origin)
                for entity in boards
            ),
        },
        "own_hand_runtime": tuple(
            value for card in hand for value in _hand_runtime(card, env.turn)
        ),
        "public_board_runtime": tuple(
            value for entity in boards for value in _board_runtime(entity)
        ),
        "public_board_keywords": tuple(keyword_rows),
        "leader_area": {
            "faith_ids": tuple(
                0 if faith is None else env._v2_faith_index.get(faith.faith_id, 0)
                for faith in faiths
            ),
            "faith_values": tuple(
                0.0 if faith is None else min(faith.value, 50) / 50
                for faith in faiths
            ),
            "faith_granted_ability_counts": tuple(
                0 if faith is None else len(faith.granted_abilities)
                for faith in faiths
            ),
            "emblem_ids": tuple(
                0 if emblem is None else env._v2_emblem_index.get(emblem.emblem_id, 0)
                for emblem in emblems
            ),
            "emblem_countdowns": tuple(
                0.0 if emblem is None else min(emblem.countdown or 0, 20) / 20
                for emblem in emblems
            ),
            "leader_damage_modifier_counts": (
                len(me.leader_damage_modifiers),
                len(opponent.leader_damage_modifiers),
            ),
            "leader_damage_modifier_totals": (
                sum(modifier.amount for modifier in me.leader_damage_modifiers),
                sum(modifier.amount for modifier in opponent.leader_damage_modifiers),
            ),
            "leader_damage_modifier_runtime": (
                *_leader_modifier_runtime(env, me),
                *_leader_modifier_runtime(env, opponent),
            ),
        },
        "choice": _choice_features(env),
        "public_history": _history_features(env),
        "action_mask": tuple(env.action_mask()),
    }


def observation_v2_spec(env: ShadowverseEnv) -> dict[str, object]:
    faith_vocabulary = tuple(
        faith_id
        for faith_id, _ in sorted(
            env._v2_faith_index.items(), key=lambda item: item[1]
        )
    )
    emblem_vocabulary = tuple(
        emblem_id
        for emblem_id, _ in sorted(
            env._v2_emblem_index.items(), key=lambda item: item[1]
        )
    )
    return {
        "version": 2,
        "action_size": env.ACTION_SIZE,
        "card_vocabulary_size": len(env.card_vocabulary),
        "card_padding_index": 0,
        "continuous_v1": env.OBSERVATION_V1_SIZE,
        "own_hand_slots": env.MAX_HAND,
        "public_board_slots": 2 * env.MAX_BOARD,
        "deck_histograms": (2, len(env.card_vocabulary)),
        "graveyard_histograms": (2, len(env.card_vocabulary)),
        "banished_histograms": (2, len(env.card_vocabulary)),
        "own_hand_runtime": env.MAX_HAND * 10,
        "public_board_runtime": 2 * env.MAX_BOARD * 12,
        "public_board_keyword_bits": 2 * env.MAX_BOARD * len(RUNTIME_KEYWORDS),
        "leader_area_slots_per_player": MAX_LEADER_AREA_SLOTS,
        "leader_damage_modifier_runtime": (
            2 * MAX_LEADER_DAMAGE_MODIFIERS * 5
        ),
        "choice_options": env.MAX_CHOICE_OPTIONS,
        "public_history_length": HISTORY_LENGTH,
        "categorical_vocabulary": {
            "cards": env.card_vocabulary,
            "faiths": faith_vocabulary,
            "emblems": emblem_vocabulary,
            "origins": tuple(origin.value for origin in CardOrigin),
            "events": tuple(event.value for event in EventType),
            "choice_kinds": tuple(kind.value for kind in ChoiceKind),
            "runtime_keywords": RUNTIME_KEYWORDS,
        },
        "recurrent_state": "model-owned; carry public_history and hidden state externally",
    }
