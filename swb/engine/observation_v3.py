from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from gymnasium import spaces

from swb.engine.observation_v2 import (
    HISTORY_LENGTH,
    MAX_LEADER_AREA_SLOTS,
    MAX_LEADER_DAMAGE_MODIFIERS,
    RUNTIME_KEYWORDS,
    encode_observation_v2,
)

if TYPE_CHECKING:
    from swb.engine.environment import ShadowverseEnv


def _int_array(values, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.int32)
    return array if shape is None else array.reshape(shape)


def _float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def encode_observation_v3(
    env: ShadowverseEnv,
    *,
    perspective: int | None = None,
    action_mask: Sequence[bool] | None = None,
    open_decklists: bool = False,
) -> dict[str, np.ndarray]:
    """Encode a NumPy-only, fixed-shape, hidden-information-safe observation."""
    perspective = env.decision_player if perspective is None else perspective
    if action_mask is None and perspective != env.decision_player:
        action_mask = [False] * env.ACTION_SIZE
    legacy = encode_observation_v2(
        env,
        perspective=perspective,
        action_mask=action_mask,
        open_decklists=open_decklists,
    )
    cards = legacy["card_indices"]
    origins = legacy["origins"]
    leader = legacy["leader_area"]
    choice = legacy["choice"]
    history = legacy["public_history"]

    leader_categorical = (
        *leader["faith_ids"],
        *leader["emblem_ids"],
        *leader["empty_deck_outcomes"],
    )
    leader_continuous = (
        *leader["faith_values"],
        *leader["faith_granted_ability_counts"],
        *leader["faith_mode_selection_bonuses"],
        *leader["emblem_countdowns"],
        *leader["emblem_random_choice_runtime"],
        *leader["leader_damage_modifier_counts"],
        *leader["leader_damage_modifier_totals"],
        *leader["leader_damage_modifier_runtime"],
        *leader["leader_barrier_charges"],
        *leader["leader_max_healths"],
    )
    choice_categorical = (
        choice["kind"],
        choice["option_count"],
        choice["target_count"],
        choice["selected_count"],
        *choice["option_references"],
        *choice["option_leader_relations"],
    )
    if perspective != env.decision_player:
        choice_categorical = (0,) * len(choice_categorical)
    history_categorical = (
        *history["event_types"],
        *history["actor_relations"],
    )

    return {
        "continuous": _float_array(legacy["continuous_v1"]),
        "own_hand_cards": _int_array(cards["own_hand"]),
        "public_board_cards": _int_array(cards["public_board"]),
        "own_initial_deck": _int_array(cards["initial_decks"][0]),
        "opponent_initial_deck": _int_array(cards["initial_decks"][1]),
        "public_graveyards": _int_array(cards["public_graveyards"], shape=(2, -1)),
        "public_banished": _int_array(cards["public_banished"], shape=(2, -1)),
        "own_hand_origins": _int_array(origins["own_hand"]),
        "public_board_origins": _int_array(origins["public_board"]),
        "own_hand_runtime": _float_array(legacy["own_hand_runtime"]),
        "public_board_runtime": _float_array(legacy["public_board_runtime"]),
        "public_board_keywords": _float_array(legacy["public_board_keywords"]),
        "leader_categorical": _int_array(leader_categorical),
        "leader_continuous": _float_array(leader_continuous),
        "choice_categorical": _int_array(choice_categorical),
        "history_categorical": _int_array(history_categorical),
        "history_amounts": _float_array(history["amounts"]),
        "action_mask": np.asarray(legacy["action_mask"], dtype=np.int8),
    }


def observation_v3_space(env: ShadowverseEnv) -> spaces.Dict:
    vocabulary_size = len(env.card_vocabulary)
    max_int = np.iinfo(np.int32).max

    def categorical(shape: tuple[int, ...]) -> spaces.Box:
        return spaces.Box(0, max_int, shape=shape, dtype=np.int32)

    def continuous(shape: tuple[int, ...]) -> spaces.Box:
        return spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

    leader_categorical_size = 4 * MAX_LEADER_AREA_SLOTS + 2
    leader_continuous_size = (
        4 * 2 * MAX_LEADER_AREA_SLOTS
        + 2 * MAX_LEADER_AREA_SLOTS * 5
        + 2
        + 2
        + 2 * MAX_LEADER_DAMAGE_MODIFIERS * 6
        + 2
        + 2
    )
    choice_size = 4 + 2 * env.MAX_CHOICE_OPTIONS
    return spaces.Dict({
        "continuous": continuous((env.OBSERVATION_V1_SIZE,)),
        "own_hand_cards": categorical((env.MAX_HAND,)),
        "public_board_cards": categorical((2 * env.MAX_BOARD,)),
        "own_initial_deck": categorical((vocabulary_size,)),
        "opponent_initial_deck": categorical((vocabulary_size,)),
        "public_graveyards": categorical((2, vocabulary_size)),
        "public_banished": categorical((2, vocabulary_size)),
        "own_hand_origins": categorical((env.MAX_HAND,)),
        "public_board_origins": categorical((2 * env.MAX_BOARD,)),
        "own_hand_runtime": continuous((env.MAX_HAND * 14,)),
        "public_board_runtime": continuous((2 * env.MAX_BOARD * 23,)),
        "public_board_keywords": continuous(
            (2 * env.MAX_BOARD * len(RUNTIME_KEYWORDS),)
        ),
        "leader_categorical": categorical((leader_categorical_size,)),
        "leader_continuous": continuous((leader_continuous_size,)),
        "choice_categorical": categorical((choice_size,)),
        "history_categorical": categorical((2 * HISTORY_LENGTH,)),
        "history_amounts": continuous((HISTORY_LENGTH,)),
        "action_mask": spaces.MultiBinary(env.ACTION_SIZE),
    })
