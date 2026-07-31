from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from swb.engine.environment import ShadowverseEnv
from swb.engine.observation_v4 import (
    CHOICE_OPTION_STATE_SIZE,
    CHOICE_REFERENCE_COUNT,
    GRAVEYARD_OPTION_STATE_SIZE,
)
from swb.engine import observation_v4_1 as v4_1
from swb.engine.play_modes import MAX_SPECIAL_MODES_PER_CARD


LEGACY_POLICY_ARCHITECTURE = "legacy_gru_v1"
ENTITY_ACTION_POLICY_ARCHITECTURE = "entity_action_v1"
POLICY_ARCHITECTURES = frozenset({
    LEGACY_POLICY_ARCHITECTURE,
    ENTITY_ACTION_POLICY_ARCHITECTURE,
})


class MaskedPolicyNetwork(nn.Module):
    architecture: str
    input_size: int
    action_size: int
    hidden_size: int
    card_vocabulary_size: int
    card_slot_count: int
    card_embedding_dim: int

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    @staticmethod
    def masked_logits(
        logits: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        validate_legal_rows: bool = True,
    ) -> torch.Tensor:
        mask = action_mask.to(dtype=torch.bool)
        if mask.ndim != logits.ndim or mask.shape != logits.shape:
            raise ValueError("action mask shape must match policy logits")
        if validate_legal_rows and not bool(mask.any(dim=-1).all()):
            raise ValueError("every live policy row must contain a legal action")
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def specification(self) -> dict[str, object]:
        raise NotImplementedError


class RecurrentMaskedActorCritic(MaskedPolicyNetwork):
    """Legacy flattened-observation baseline kept for checkpoint compatibility."""

    architecture = LEGACY_POLICY_ARCHITECTURE

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_size: int = 64,
        *,
        card_vocabulary_size: int = 0,
        card_slot_count: int = 0,
        card_embedding_dim: int = 16,
    ):
        super().__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.card_vocabulary_size = card_vocabulary_size
        self.card_slot_count = card_slot_count
        self.card_embedding_dim = card_embedding_dim
        if (card_vocabulary_size == 0) != (card_slot_count == 0):
            raise ValueError(
                "card_vocabulary_size and card_slot_count must both be zero or positive"
            )
        self.card_embedding = (
            nn.Embedding(
                card_vocabulary_size + 1,
                card_embedding_dim,
                padding_idx=0,
            )
            if card_vocabulary_size > 0
            else None
        )
        encoder_size = input_size + card_slot_count * card_embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_size, hidden_size),
            nn.Tanh(),
        )
        self.recurrent = nn.GRUCell(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward_step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        card_indices: torch.Tensor | None = None,
        *,
        validate_card_indices: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_input = observation
        if self.card_embedding is not None:
            if card_indices is None:
                raise ValueError("card indices are required by this policy")
            if card_indices.shape[:-1] != observation.shape[:-1]:
                raise ValueError("card index batch shape must match observations")
            if card_indices.shape[-1] != self.card_slot_count:
                raise ValueError(
                    f"expected {self.card_slot_count} card slots, "
                    f"got {card_indices.shape[-1]}"
                )
            if validate_card_indices and (
                bool((card_indices < 0).any())
                or bool(
                    (card_indices > self.card_vocabulary_size).any()
                )
            ):
                raise ValueError("card index is outside the policy vocabulary")
            embedded = self.card_embedding(card_indices.to(dtype=torch.long))
            model_input = torch.cat(
                (observation, embedded.flatten(start_dim=-2)), dim=-1
            )
        encoded = self.encoder(model_input)
        next_hidden = self.recurrent(encoded, hidden)
        return (
            self.policy_head(next_hidden),
            self.value_head(next_hidden).squeeze(-1),
            next_hidden,
        )

    def specification(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "input_size": self.input_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "card_vocabulary_size": self.card_vocabulary_size,
            "card_slot_count": self.card_slot_count,
            "card_embedding_dim": self.card_embedding_dim,
        }


class EntityActionRecurrentActorCritic(MaskedPolicyNetwork):
    """Entity encoder plus a source/target-conditioned legal-action scorer.

    Card identities and their public runtime fields are encoded as entity
    tokens. The policy head scores an action from its semantic kind and the
    contextual source/target tokens instead of assigning one unrelated output
    weight to every integer action ID.
    """

    architecture = ENTITY_ACTION_POLICY_ARCHITECTURE
    _ENTITY_FEATURE_SIZE = 33
    _ACTION_KIND_COUNT = 11
    _LEADER_RELATION_COUNT = 3
    _OPTION_ORDINAL_COUNT = max(
        ShadowverseEnv.MAX_CHOICE_OPTIONS,
        ShadowverseEnv.GRAVEYARD_PAGE_SIZE,
    ) + 1

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_size: int = 512,
        *,
        card_vocabulary_size: int,
        card_slot_count: int,
        card_embedding_dim: int = 128,
        model_dim: int = 256,
        transformer_layers: int = 4,
        attention_heads: int = 8,
        feedforward_dim: int = 1024,
        field_layout: Mapping[str, tuple[int, int]],
        card_field_layout: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        if action_size != ShadowverseEnv.ACTION_SIZE:
            raise ValueError(
                "entity-action-v1 requires the versioned 112-action layout"
            )
        minimum_card_slots = ShadowverseEnv.MAX_HAND + 2 * ShadowverseEnv.MAX_BOARD
        if card_slot_count < minimum_card_slots:
            raise ValueError(
                f"entity-action-v1 requires at least {minimum_card_slots} card slots"
            )
        if model_dim <= 0 or transformer_layers <= 0 or attention_heads <= 0:
            raise ValueError("entity-action dimensions must be positive")
        if model_dim % attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        self.v4_1_observation = "hand_modifier_kind" in field_layout
        self.v4_observation = (
            "own_hand_state" in field_layout
            and not self.v4_1_observation
        )
        if self.v4_1_observation:
            required_fields = {
                "player_state",
                "player_class",
                "match_state",
                "own_hand_base",
                "public_board_base",
                "own_hand_keywords",
                "public_board_keywords",
                "hand_modifier_kind",
                "hand_modifier_subtype",
                "hand_modifier_duration",
                "hand_modifier_expiry",
                "hand_modifier_values",
                "hand_modifier_summary",
                "board_modifier_kind",
                "board_modifier_subtype",
                "board_modifier_duration",
                "board_modifier_expiry",
                "board_modifier_values",
                "board_modifier_summary",
                "hand_effect_tokens",
                "hand_effect_summary",
                "board_effect_tokens",
                "board_effect_summary",
                "leader_area_state",
                "leader_effect_tokens",
                "leader_effect_summary",
                "listener_state",
                "leader_modifier_state",
                "zone_counts",
                "zone_overflow",
                "own_deck_state",
                "choice_kind",
                "choice_state",
                "choice_option_references",
                "choice_option_relations",
                "choice_option_selected",
                "choice_option_semantics",
                "graveyard_option_state",
                "graveyard_option_semantics",
                "history_event_types",
                "history_actors",
                "history_amounts",
                "history_source_references",
                "history_target_references",
                "history_semantics",
                "record_state",
                "record_semantics",
            }
        elif self.v4_observation:
            required_fields = {
                "own_hand_origin_bits",
                "public_board_origin_bits",
                "own_hand_state",
                "public_board_state",
                "own_hand_keyword_bits",
                "public_board_keyword_bits",
                "own_hand_modifier_state",
                "public_board_modifier_state",
                "own_hand_effect_bits",
                "public_board_effect_bits",
                "choice_option_state",
                "graveyard_option_state",
            }
        else:
            required_fields = {
                "own_hand_origins",
                "public_board_origins",
                "own_hand_runtime",
                "public_board_runtime",
                "public_board_keywords",
                "choice_categorical",
            }
        missing = required_fields - set(field_layout)
        if missing:
            raise ValueError(
                f"entity-action-v1 is missing observation fields {sorted(missing)}"
            )

        self.input_size = input_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.card_vocabulary_size = card_vocabulary_size
        self.card_slot_count = card_slot_count
        self.entity_slot_count = minimum_card_slots
        self.card_embedding_dim = card_embedding_dim
        self.model_dim = model_dim
        self.transformer_layers = transformer_layers
        self.attention_heads = attention_heads
        self.feedforward_dim = feedforward_dim
        self.field_layout = {
            name: (int(offset), int(size))
            for name, (offset, size) in field_layout.items()
        }
        if card_field_layout is None:
            card_field_layout = {
                "own_hand_cards": (0, ShadowverseEnv.MAX_HAND),
                "public_board_cards": (
                    ShadowverseEnv.MAX_HAND,
                    2 * ShadowverseEnv.MAX_BOARD,
                ),
            }
        self.card_field_layout = {
            name: (int(offset), int(size))
            for name, (offset, size) in card_field_layout.items()
        }
        for name, expected_size in (
            ("own_hand_cards", ShadowverseEnv.MAX_HAND),
            ("public_board_cards", 2 * ShadowverseEnv.MAX_BOARD),
        ):
            if name not in self.card_field_layout:
                raise ValueError(f"entity-action-v1 is missing card field {name!r}")
            if self.card_field_layout[name][1] != expected_size:
                raise ValueError(
                    f"entity-action-v1 card field {name!r} must have "
                    f"{expected_size} slots"
                )
        if sum(size for _, size in self.card_field_layout.values()) != card_slot_count:
            raise ValueError("card field layout does not match card_slot_count")

        global_keep = torch.ones(input_size, dtype=torch.float32)
        if self.v4_1_observation:
            global_fields = {"match_state", "choice_state"}
            for name, (offset, size) in self.field_layout.items():
                if name not in global_fields:
                    global_keep[offset : offset + size] = 0.0
        else:
            choice_route_field = (
                "choice_option_state"
                if self.v4_observation
                else "choice_categorical"
            )
            routed_fields = (
                required_fields
                if self.v4_observation
                else required_fields - {choice_route_field}
            )
            for name in routed_fields:
                offset, size = self.field_layout[name]
                global_keep[offset : offset + size] = 0.0
        if not self.v4_observation and not self.v4_1_observation:
            choice_offset, choice_size = self.field_layout["choice_categorical"]
            global_keep[choice_offset + 4 : choice_offset + choice_size] = 0.0
        self.register_buffer("_global_keep", global_keep, persistent=False)
        if self.v4_observation or self.v4_1_observation:
            global_indices = torch.nonzero(
                global_keep, as_tuple=False
            ).flatten()
            global_input_size = int(global_indices.numel())
        else:
            # Preserve the exact v3 parameter shape so existing checkpoints
            # continue to load. V4 is new and can avoid allocating weights for
            # entity/option rows that are routed through dedicated encoders.
            global_indices = torch.arange(input_size, dtype=torch.long)
            global_input_size = input_size
        self.register_buffer(
            "_global_indices",
            global_indices,
            persistent=False,
        )
        self.global_input_size = global_input_size

        self.card_embedding = nn.Embedding(
            card_vocabulary_size + 1,
            card_embedding_dim,
            padding_idx=0,
        )
        self.card_projection = nn.Linear(card_embedding_dim, model_dim)
        if self.v4_1_observation:
            self._v4_hand_fields = ()
            self._v4_board_fields = ()
            hand_feature_size = (
                self.field_layout["own_hand_base"][1]
                // ShadowverseEnv.MAX_HAND
                + self.field_layout["own_hand_keywords"][1]
                // ShadowverseEnv.MAX_HAND
                + self.field_layout["hand_modifier_summary"][1]
                // ShadowverseEnv.MAX_HAND
            )
            board_count = 2 * ShadowverseEnv.MAX_BOARD
            board_feature_size = (
                self.field_layout["public_board_base"][1] // board_count
                + self.field_layout["public_board_keywords"][1] // board_count
                + self.field_layout["board_modifier_summary"][1] // board_count
            )
            self.entity_feature_size = max(
                hand_feature_size, board_feature_size
            )
        elif self.v4_observation:
            hand_fields = (
                "own_hand_origin_bits",
                "own_hand_state",
                "own_hand_keyword_bits",
                "own_hand_modifier_state",
                "own_hand_effect_bits",
            )
            board_fields = (
                "public_board_origin_bits",
                "public_board_state",
                "public_board_keyword_bits",
                "public_board_modifier_state",
                "public_board_effect_bits",
            )
            hand_feature_size = sum(
                self.field_layout[name][1] // ShadowverseEnv.MAX_HAND
                for name in hand_fields
            )
            board_feature_size = sum(
                self.field_layout[name][1] // (2 * ShadowverseEnv.MAX_BOARD)
                for name in board_fields
            )
            self._v4_hand_fields = hand_fields
            self._v4_board_fields = board_fields
            self.entity_feature_size = max(hand_feature_size, board_feature_size)
        else:
            self._v4_hand_fields = ()
            self._v4_board_fields = ()
            self.entity_feature_size = self._ENTITY_FEATURE_SIZE
        self.entity_feature_projection = nn.Linear(
            self.entity_feature_size, model_dim, bias=False
        )
        self.zone_embedding = nn.Embedding(3, model_dim)
        self.zone_slot_embedding = nn.Embedding(
            max(ShadowverseEnv.MAX_HAND, ShadowverseEnv.MAX_BOARD),
            model_dim,
        )
        self.global_projection = nn.Sequential(
            nn.Linear(global_input_size, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
        auxiliary_fields = tuple(
            name
            for name in self.card_field_layout
            if not self.v4_1_observation and name not in {
                "own_hand_cards",
                "public_board_cards",
                "choice_option_cards",
                "graveyard_page_cards",
            }
        )
        self.auxiliary_card_fields = auxiliary_fields
        if auxiliary_fields:
            max_auxiliary_slots = max(
                self.card_field_layout[name][1] for name in auxiliary_fields
            )
            self.auxiliary_field_embedding = nn.Embedding(
                len(auxiliary_fields), model_dim
            )
            self.auxiliary_position_embedding = nn.Embedding(
                max_auxiliary_slots, model_dim
            )
            self.auxiliary_token_encoder = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, model_dim),
                nn.GELU(),
            )
        else:
            self.auxiliary_field_embedding = None
            self.auxiliary_position_embedding = None
            self.auxiliary_token_encoder = None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.entity_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.recurrent = nn.GRUCell(model_dim, hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, model_dim)

        self.action_kind_embedding = nn.Embedding(
            self._ACTION_KIND_COUNT, model_dim
        )
        self.mode_embedding = nn.Embedding(
            MAX_SPECIAL_MODES_PER_CARD + 1, model_dim
        )
        self.leader_relation_embedding = nn.Embedding(
            self._LEADER_RELATION_COUNT, model_dim
        )
        self.option_ordinal_embedding = nn.Embedding(
            self._OPTION_ORDINAL_COUNT, model_dim
        )
        self.source_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.target_projection = nn.Linear(model_dim, model_dim, bias=False)
        if self.v4_observation:
            self.choice_option_projection = nn.Linear(
                CHOICE_OPTION_STATE_SIZE, model_dim, bias=False
            )
            self.graveyard_option_projection = nn.Linear(
                GRAVEYARD_OPTION_STATE_SIZE, model_dim, bias=False
            )
        else:
            self.choice_option_projection = None
            self.graveyard_option_projection = None
        if self.v4_1_observation:
            self._init_v4_1_modules()
            self.structured_token_count = v4_1.STRUCTURED_TOKEN_COUNT
        else:
            self.structured_token_count = 1 + self.entity_slot_count
        self.policy_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

        action_kind, source_slot, target_slot, mode, leader_relation, ordinal = (
            self._build_action_layout()
        )
        self.register_buffer("_action_kind", action_kind, persistent=False)
        self.register_buffer("_source_slot", source_slot, persistent=False)
        self.register_buffer("_target_slot", target_slot, persistent=False)
        self.register_buffer("_action_mode", mode, persistent=False)
        self.register_buffer(
            "_leader_relation", leader_relation, persistent=False
        )
        self.register_buffer("_option_ordinal", ordinal, persistent=False)

        zones = (
            [0] * ShadowverseEnv.MAX_HAND
            + [1] * ShadowverseEnv.MAX_BOARD
            + [2] * ShadowverseEnv.MAX_BOARD
        )
        zone_slots = (
            list(range(ShadowverseEnv.MAX_HAND))
            + list(range(ShadowverseEnv.MAX_BOARD))
            + list(range(ShadowverseEnv.MAX_BOARD))
        )
        self.register_buffer(
            "_entity_zones", torch.tensor(zones, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_entity_zone_slots",
            torch.tensor(zone_slots, dtype=torch.long),
            persistent=False,
        )

    def _init_v4_1_modules(self) -> None:
        dim = self.model_dim
        self.v41_player_projection = nn.Linear(
            v4_1.v4.PLAYER_STATE_SIZE, dim, bias=False
        )
        self.v41_player_class_embedding = nn.Embedding(
            ShadowverseEnv.CLASS_COUNT + 1, dim
        )
        self.v41_entity_base_projection = nn.Linear(
            self.entity_feature_size, dim, bias=False
        )
        self.v41_origin_embedding = nn.Embedding(
            v4_1.ORIGIN_COUNT, dim, padding_idx=0
        )
        self.v41_source_origin_embedding = nn.Embedding(
            v4_1.ORIGIN_COUNT, dim, padding_idx=0
        )
        self.v41_modifier_kind_embedding = nn.Embedding(
            v4_1.MAX_MODIFIER_KIND + 1, dim, padding_idx=0
        )
        self.v41_modifier_subtype_embedding = nn.Embedding(
            v4_1.MAX_MODIFIER_SUBTYPE + 1, dim, padding_idx=0
        )
        self.v41_modifier_duration_embedding = nn.Embedding(
            v4_1.DURATION_COUNT + 1, dim, padding_idx=0
        )
        self.v41_expiry_relation_embedding = nn.Embedding(3, dim)
        self.v41_modifier_value_projection = nn.Linear(
            v4_1.MODIFIER_VALUE_SIZE, dim, bias=False
        )
        self.v41_semantic_kind_embedding = nn.Embedding(
            16, dim, padding_idx=0
        )
        self.v41_semantic_byte_embedding = nn.Embedding(4 * 256, dim)
        self.v41_hand_effect_summary_projection = nn.Linear(
            v4_1.HAND_EFFECT_SUMMARY_SIZE, dim, bias=False
        )
        self.v41_board_effect_summary_projection = nn.Linear(
            v4_1.BOARD_EFFECT_SUMMARY_SIZE, dim, bias=False
        )
        self.v41_leader_state_projection = nn.Linear(
            v4_1.LEADER_AREA_STATE_SIZE, dim, bias=False
        )
        self.v41_leader_slot_embedding = nn.Embedding(
            v4_1.LEADER_AREA_SLOTS, dim
        )
        self.v41_leader_area_type_embedding = nn.Embedding(3, dim)
        self.v41_leader_effect_summary_projection = nn.Linear(
            v4_1.LEADER_EFFECT_SUMMARY_SIZE, dim, bias=False
        )
        self.v41_listener_projection = nn.Linear(
            v4_1.LISTENER_FEATURE_SIZE, dim, bias=False
        )
        self.v41_leader_modifier_mode_embedding = nn.Embedding(3, dim)
        self.v41_source_controller_embedding = nn.Embedding(3, dim)
        self.v41_leader_source_reference_embedding = nn.Embedding(
            v4_1.v4.LEADER_SOURCE_REFERENCE_COUNT, dim
        )
        self.v41_leader_modifier_amount_projection = nn.Linear(
            1, dim, bias=False
        )
        self.v41_leader_modifier_count_projection = nn.Linear(
            1, dim, bias=False
        )
        self.v41_leader_modifier_row_encoder = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )
        self.v41_scalar_projection = nn.Linear(1, dim, bias=False)
        self.v41_zone_embedding = nn.Embedding(v4_1.ZONE_GROUPS, dim)
        self.v41_zone_card_count_projection = nn.Linear(
            1, dim, bias=False
        )
        self.v41_zone_kind_count_projection = nn.Linear(
            1, dim, bias=False
        )
        self.v41_zone_overflow_projection = nn.Linear(2, dim, bias=False)
        self.v41_deck_state_projection = nn.Linear(4, dim, bias=False)
        self.v41_history_event_embedding = nn.Embedding(
            v4_1.EVENT_TYPE_COUNT + 1, dim, padding_idx=0
        )
        self.v41_reference_embedding = nn.Embedding(
            v4_1.CHOICE_REFERENCE_COUNT, dim
        )
        self.v41_history_source_reference_embedding = nn.Embedding(
            v4_1.CHOICE_REFERENCE_COUNT, dim
        )
        self.v41_history_target_reference_embedding = nn.Embedding(
            v4_1.CHOICE_REFERENCE_COUNT, dim
        )
        self.v41_history_source_card_projection = nn.Linear(
            dim, dim, bias=False
        )
        self.v41_history_target_card_projection = nn.Linear(
            dim, dim, bias=False
        )
        self.v41_history_position_embedding = nn.Embedding(
            v4_1.HISTORY_LENGTH, dim
        )
        self.v41_record_state_projection = nn.Linear(
            v4_1.RECORD_STATE_SIZE, dim, bias=False
        )
        self.v41_record_position_embedding = nn.Embedding(
            v4_1.HISTORY_RECORDS_PER_GROUP, dim
        )
        self.v41_record_group_embedding = nn.Embedding(
            v4_1.RECORD_GROUPS, dim
        )
        self.v41_record_kind_embedding = nn.Embedding(4, dim)
        self.v41_choice_kind_embedding = nn.Embedding(
            len(v4_1.v4.CHOICE_KIND_VALUES) + 1,
            dim,
            padding_idx=0,
        )
        self.v41_choice_state_projection = nn.Linear(5, dim, bias=False)
        self.v41_graveyard_state_projection = nn.Linear(5, dim, bias=False)

    @staticmethod
    def _build_action_layout() -> tuple[torch.Tensor, ...]:
        action_size = ShadowverseEnv.ACTION_SIZE
        kind = torch.zeros(action_size, dtype=torch.long)
        source = torch.full((action_size,), -1, dtype=torch.long)
        target = torch.full((action_size,), -1, dtype=torch.long)
        mode = torch.zeros(action_size, dtype=torch.long)
        leader_relation = torch.zeros(action_size, dtype=torch.long)
        ordinal = torch.zeros(action_size, dtype=torch.long)

        hand_start = 0
        own_board_start = ShadowverseEnv.MAX_HAND
        enemy_board_start = own_board_start + ShadowverseEnv.MAX_BOARD

        for hand_index in range(ShadowverseEnv.MAX_HAND):
            action = ShadowverseEnv.PLAY_OFFSET + hand_index
            kind[action] = 1
            source[action] = hand_start + hand_index

        for attacker in range(ShadowverseEnv.MAX_BOARD):
            for target_index in range(ShadowverseEnv.TARGETS_PER_ATTACKER):
                action = (
                    ShadowverseEnv.ATTACK_OFFSET
                    + attacker * ShadowverseEnv.TARGETS_PER_ATTACKER
                    + target_index
                )
                kind[action] = 2
                source[action] = own_board_start + attacker
                if target_index == 0:
                    leader_relation[action] = 2
                else:
                    target[action] = enemy_board_start + target_index - 1

        for board_index in range(ShadowverseEnv.MAX_BOARD):
            action = ShadowverseEnv.EVOLVE_OFFSET + board_index
            kind[action] = 3
            source[action] = own_board_start + board_index

        for option_index in range(ShadowverseEnv.MAX_CHOICE_OPTIONS):
            action = ShadowverseEnv.CHOICE_OFFSET + option_index
            kind[action] = 4
            ordinal[action] = option_index + 1

        kind[ShadowverseEnv.GRAVEYARD_PREV_PAGE] = 5
        kind[ShadowverseEnv.GRAVEYARD_NEXT_PAGE] = 6
        for option_index in range(ShadowverseEnv.GRAVEYARD_PAGE_SIZE):
            action = ShadowverseEnv.GRAVEYARD_SLOT_OFFSET + option_index
            kind[action] = 7
            ordinal[action] = option_index + 1

        for hand_index in range(ShadowverseEnv.MAX_HAND):
            for mode_index in range(MAX_SPECIAL_MODES_PER_CARD):
                action = (
                    ShadowverseEnv.MODE_PLAY_OFFSET
                    + hand_index * MAX_SPECIAL_MODES_PER_CARD
                    + mode_index
                )
                kind[action] = 8
                source[action] = hand_start + hand_index
                mode[action] = mode_index + 1

        for board_index in range(ShadowverseEnv.MAX_BOARD):
            action = ShadowverseEnv.SUPER_EVOLVE_OFFSET + board_index
            kind[action] = 9
            source[action] = own_board_start + board_index

        kind[ShadowverseEnv.USE_EXTRA_PP] = 10
        return kind, source, target, mode, leader_relation, ordinal

    def _field(self, observation: torch.Tensor, name: str) -> torch.Tensor:
        offset, size = self.field_layout[name]
        return observation[:, offset : offset + size]

    def _v41_semantic_context(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        rows = torch.round(values).to(dtype=torch.long)
        if rows.shape[-1] != v4_1.SEMANTIC_TOKEN_SIZE:
            raise ValueError("v4.1 semantic rows must have five values")
        kind = rows[..., 0].clamp(
            min=0,
            max=self.v41_semantic_kind_embedding.num_embeddings - 1,
        )
        byte_values = rows[..., 1:].clamp(min=0, max=255)
        positions = torch.arange(4, device=values.device)
        positioned_bytes = byte_values + 256 * positions
        byte_context = self.v41_semantic_byte_embedding(
            positioned_bytes
        ).mean(dim=-2)
        context = self.v41_semantic_kind_embedding(kind) + byte_context
        return context * (kind != 0).unsqueeze(-1).to(dtype=values.dtype)

    @staticmethod
    def _v41_masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        dim: int,
    ) -> torch.Tensor:
        expanded = mask.unsqueeze(-1).to(dtype=values.dtype)
        total = (values * expanded).sum(dim=dim)
        count = expanded.sum(dim=dim).clamp(min=1)
        return total / count

    def _v41_modifier_context(
        self,
        observation: torch.Tensor,
        *,
        prefix: str,
        entity_count: int,
        modifier_count: int,
    ) -> torch.Tensor:
        batch = observation.shape[0]
        shape = (batch, entity_count, modifier_count)
        kind = torch.round(
            self._field(observation, f"{prefix}_modifier_kind")
        ).to(dtype=torch.long).reshape(shape)
        subtype = torch.round(
            self._field(observation, f"{prefix}_modifier_subtype")
        ).to(dtype=torch.long).reshape(shape)
        duration = torch.round(
            self._field(observation, f"{prefix}_modifier_duration")
        ).to(dtype=torch.long).reshape(shape)
        expiry = torch.round(
            self._field(observation, f"{prefix}_modifier_expiry")
        ).to(dtype=torch.long).reshape(shape)
        values = self._field(
            observation, f"{prefix}_modifier_values"
        ).reshape(
            batch,
            entity_count,
            modifier_count,
            v4_1.MODIFIER_VALUE_SIZE,
        )
        kind_context = self.v41_modifier_kind_embedding(kind.clamp(
            min=0,
            max=self.v41_modifier_kind_embedding.num_embeddings - 1,
        ))
        subtype_context = self.v41_modifier_subtype_embedding(
            subtype.clamp(
                min=0,
                max=self.v41_modifier_subtype_embedding.num_embeddings - 1,
            )
        )
        timing_context = (
            self.v41_modifier_duration_embedding(duration.clamp(
                min=0,
                max=self.v41_modifier_duration_embedding.num_embeddings - 1,
            ))
            + self.v41_expiry_relation_embedding(
                expiry.clamp(min=0, max=2)
            )
        )
        value_context = self.v41_modifier_value_projection(values)
        paired_identity = kind_context * (
            1.0 + torch.tanh(subtype_context)
        )
        paired_timing = paired_identity * (
            1.0 + torch.tanh(timing_context)
        )
        tokens = (
            paired_timing * (1.0 + torch.tanh(value_context))
            + subtype_context
            + timing_context
            + value_context
        )
        return self._v41_masked_mean(
            tokens, kind != 0, dim=2
        )

    def _entity_features_v4_1(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        batch = observation.shape[0]
        hand_count = ShadowverseEnv.MAX_HAND
        board_count = 2 * ShadowverseEnv.MAX_BOARD
        hand = torch.cat((
            self._field(observation, "own_hand_base").reshape(
                batch, hand_count, -1
            ),
            self._field(observation, "own_hand_keywords").reshape(
                batch, hand_count, -1
            ),
            self._field(observation, "hand_modifier_summary").reshape(
                batch, hand_count, -1
            ),
        ), dim=-1)
        board = torch.cat((
            self._field(observation, "public_board_base").reshape(
                batch, board_count, -1
            ),
            self._field(observation, "public_board_keywords").reshape(
                batch, board_count, -1
            ),
            self._field(observation, "board_modifier_summary").reshape(
                batch, board_count, -1
            ),
        ), dim=-1)
        features = observation.new_zeros(
            batch,
            hand_count + board_count,
            self.entity_feature_size,
        )
        features[:, :hand_count, :hand.shape[-1]] = hand
        features[:, hand_count:, :board.shape[-1]] = board
        return features

    def _entity_features(self, observation: torch.Tensor) -> torch.Tensor:
        if self.v4_1_observation:
            return self._entity_features_v4_1(observation)
        if self.v4_observation:
            return self._entity_features_v4(observation)
        batch = observation.shape[0]
        features = observation.new_zeros(
            batch, self.card_slot_count, self._ENTITY_FEATURE_SIZE
        )
        hand_count = ShadowverseEnv.MAX_HAND
        board_count = 2 * ShadowverseEnv.MAX_BOARD
        hand_runtime = self._field(observation, "own_hand_runtime").reshape(
            batch, hand_count, -1
        )
        board_runtime = self._field(observation, "public_board_runtime").reshape(
            batch, board_count, -1
        )
        board_keywords = self._field(
            observation, "public_board_keywords"
        ).reshape(batch, board_count, -1)
        features[:, :hand_count, 0] = self._field(
            observation, "own_hand_origins"
        )
        features[:, :hand_count, 1 : 1 + hand_runtime.shape[-1]] = hand_runtime
        features[:, hand_count:, 0] = self._field(
            observation, "public_board_origins"
        )
        features[
            :, hand_count:, 1 : 1 + board_runtime.shape[-1]
        ] = board_runtime
        keyword_start = 1 + board_runtime.shape[-1]
        features[
            :, hand_count:, keyword_start : keyword_start + board_keywords.shape[-1]
        ] = board_keywords
        return features

    def _entity_features_v4(self, observation: torch.Tensor) -> torch.Tensor:
        batch = observation.shape[0]
        hand_count = ShadowverseEnv.MAX_HAND
        board_count = 2 * ShadowverseEnv.MAX_BOARD
        hand_parts = [
            self._field(observation, name).reshape(batch, hand_count, -1)
            for name in self._v4_hand_fields
        ]
        board_parts = [
            self._field(observation, name).reshape(batch, board_count, -1)
            for name in self._v4_board_fields
        ]
        hand = torch.cat(hand_parts, dim=-1)
        board = torch.cat(board_parts, dim=-1)
        features = observation.new_zeros(
            batch,
            hand_count + board_count,
            self.entity_feature_size,
        )
        features[:, :hand_count, : hand.shape[-1]] = hand
        features[:, hand_count:, : board.shape[-1]] = board
        return features

    def _card_field(
        self,
        card_tokens: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        offset, size = self.card_field_layout[name]
        return card_tokens[:, offset : offset + size]

    def _card_index_field(
        self,
        card_indices: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        offset, size = self.card_field_layout[name]
        return card_indices[:, offset : offset + size]

    def _entity_card_tokens(
        self,
        card_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                self._card_field(card_tokens, "own_hand_cards"),
                self._card_field(card_tokens, "public_board_cards"),
            ),
            dim=1,
        )

    def _entity_card_indices(
        self,
        card_indices: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                self._card_index_field(card_indices, "own_hand_cards"),
                self._card_index_field(card_indices, "public_board_cards"),
            ),
            dim=1,
        )

    @staticmethod
    def _masked_group_mean(
        tokens: torch.Tensor,
        indices: torch.Tensor,
        groups: int,
    ) -> torch.Tensor:
        batch, slots, model_dim = tokens.shape
        if slots % groups:
            raise ValueError("grouped card field size is not divisible by groups")
        width = slots // groups
        grouped_tokens = tokens.reshape(batch, groups, width, model_dim)
        grouped_mask = (indices != 0).reshape(batch, groups, width, 1)
        total = (grouped_tokens * grouped_mask).sum(dim=2)
        count = grouped_mask.sum(dim=2).clamp(min=1)
        return total / count

    def _auxiliary_context(
        self,
        card_tokens: torch.Tensor,
        card_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self.auxiliary_card_fields:
            return card_tokens.new_zeros(card_tokens.shape[0], self.model_dim)
        encoded_rows = []
        masks = []
        for field_index, name in enumerate(self.auxiliary_card_fields):
            tokens = self._card_field(card_tokens, name)
            indices = self._card_index_field(card_indices, name)
            positions = torch.arange(
                tokens.shape[1], device=tokens.device
            )
            enriched = (
                tokens
                + self.auxiliary_field_embedding.weight[field_index]
                + self.auxiliary_position_embedding(positions).unsqueeze(0)
            )
            encoded_rows.append(self.auxiliary_token_encoder(enriched))
            masks.append(indices != 0)
        encoded = torch.cat(encoded_rows, dim=1)
        mask = torch.cat(masks, dim=1).unsqueeze(-1)
        return (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def _v41_effect_context(
        self,
        observation: torch.Tensor,
        *,
        field: str,
        entity_count: int,
        effects_per_entity: int,
    ) -> torch.Tensor:
        values = self._field(observation, field).reshape(
            observation.shape[0],
            entity_count,
            effects_per_entity,
            v4_1.SEMANTIC_TOKEN_SIZE,
        )
        tokens = self._v41_semantic_context(values)
        return self._v41_masked_mean(
            tokens,
            torch.round(values[..., 0]).to(dtype=torch.long) != 0,
            dim=2,
        )

    def _v41_leader_modifier_context(
        self,
        observation: torch.Tensor,
        card_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch = observation.shape[0]
        modifiers_per_player = v4_1.v4.MAX_LEADER_DAMAGE_MODIFIERS
        state = self._field(
            observation, "leader_modifier_state"
        ).reshape(
            batch,
            2,
            modifiers_per_player,
            v4_1.LEADER_MODIFIER_STATE_SIZE,
        )
        sources = self._card_field(
            card_tokens, "leader_modifier_source_cards"
        ).reshape(batch, 2, modifiers_per_player, self.model_dim)
        mode = torch.round(state[..., 1]).to(dtype=torch.long).clamp(
            min=0,
            max=self.v41_leader_modifier_mode_embedding.num_embeddings - 1,
        )
        duration = torch.round(state[..., 2]).to(dtype=torch.long).clamp(
            min=0,
            max=self.v41_modifier_duration_embedding.num_embeddings - 1,
        )
        expiry = torch.round(state[..., 3]).to(dtype=torch.long).clamp(
            min=0, max=2
        )
        source_relation = torch.round(state[..., 4]).to(
            dtype=torch.long
        ).clamp(min=0, max=2)
        source_reference = torch.round(state[..., 5]).to(
            dtype=torch.long
        ).clamp(
            min=0,
            max=self.v41_leader_source_reference_embedding.num_embeddings - 1,
        )
        tokens = (
            sources
            + self.v41_leader_modifier_mode_embedding(mode)
            + self.v41_modifier_duration_embedding(duration)
            + self.v41_expiry_relation_embedding(expiry)
            + self.v41_source_controller_embedding(source_relation)
            + self.v41_leader_source_reference_embedding(source_reference)
            + self.v41_leader_modifier_amount_projection(
                state[..., 6:7]
            )
        )
        tokens = self.v41_leader_modifier_row_encoder(tokens)
        return self._v41_masked_mean(
            tokens, state[..., 0] > 0, dim=2
        ) + self.v41_leader_modifier_count_projection(
            (state[..., 0] > 0).sum(dim=2, keepdim=True)
            / modifiers_per_player
        )

    def _forward_step_v4_1(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        card_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = observation.shape[0]
        card_tokens = self.card_projection(
            self.card_embedding(card_indices.to(dtype=torch.long))
        )

        global_observation = (
            observation
            * self._global_keep.to(dtype=observation.dtype)
        ).index_select(-1, self._global_indices)
        global_token = self.global_projection(global_observation)
        choice_kind = torch.round(
            self._field(observation, "choice_kind")
        ).to(dtype=torch.long).squeeze(-1).clamp(
            min=0,
            max=self.v41_choice_kind_embedding.num_embeddings - 1,
        )
        global_token = (
            global_token + self.v41_choice_kind_embedding(choice_kind)
        )

        player_state = self._field(
            observation, "player_state"
        ).reshape(batch, 2, -1)
        player_class = torch.round(
            self._field(observation, "player_class")
        ).to(dtype=torch.long).clamp(
            min=0,
            max=self.v41_player_class_embedding.num_embeddings - 1,
        )
        player_relations = torch.tensor(
            (1, 2), dtype=torch.long, device=observation.device
        )
        player_tokens = (
            self.v41_player_projection(player_state)
            + self.v41_player_class_embedding(player_class)
            + self.leader_relation_embedding(player_relations).unsqueeze(0)
        )
        player_tokens = player_tokens + self._v41_leader_modifier_context(
            observation, card_tokens
        )

        entity_card_tokens = self._entity_card_tokens(card_tokens)
        entity_card_indices = self._entity_card_indices(card_indices)
        entity_features = self._entity_features_v4_1(observation)
        entity_origins = torch.round(
            entity_features[..., :2]
        ).to(dtype=torch.long).clamp(
            min=0,
            max=self.v41_origin_embedding.num_embeddings - 1,
        )
        projected_entity_features = entity_features.clone()
        projected_entity_features[..., :2] = 0
        entity_tokens = (
            entity_card_tokens
            + self.v41_entity_base_projection(
                projected_entity_features
            )
            + self.v41_origin_embedding(entity_origins[..., 0])
            + self.v41_source_origin_embedding(entity_origins[..., 1])
            + self.zone_embedding(self._entity_zones).unsqueeze(0)
            + self.zone_slot_embedding(
                self._entity_zone_slots
            ).unsqueeze(0)
        )
        hand_modifiers = self._v41_modifier_context(
            observation,
            prefix="hand",
            entity_count=ShadowverseEnv.MAX_HAND,
            modifier_count=v4_1.MAX_HAND_MODIFIERS,
        )
        board_modifiers = self._v41_modifier_context(
            observation,
            prefix="board",
            entity_count=2 * ShadowverseEnv.MAX_BOARD,
            modifier_count=v4_1.MAX_BOARD_MODIFIERS,
        )
        hand_effects = self._v41_effect_context(
            observation,
            field="hand_effect_tokens",
            entity_count=ShadowverseEnv.MAX_HAND,
            effects_per_entity=v4_1.v4.MAX_GRANTED_ABILITIES,
        )
        board_effects = self._v41_effect_context(
            observation,
            field="board_effect_tokens",
            entity_count=2 * ShadowverseEnv.MAX_BOARD,
            effects_per_entity=v4_1.MAX_ENTITY_EFFECTS,
        )
        hand_fusion = self._masked_group_mean(
            self._card_field(card_tokens, "own_hand_fusion_cards"),
            self._card_index_field(
                card_indices, "own_hand_fusion_cards"
            ),
            ShadowverseEnv.MAX_HAND,
        )
        board_fusion = self._masked_group_mean(
            self._card_field(card_tokens, "public_board_fusion_cards"),
            self._card_index_field(
                card_indices, "public_board_fusion_cards"
            ),
            2 * ShadowverseEnv.MAX_BOARD,
        )
        listeners = self._field(
            observation, "listener_state"
        ).reshape(
            batch,
            v4_1.LISTENER_SOURCE_COUNT,
            v4_1.LISTENER_FEATURE_SIZE,
        )
        entity_tokens = entity_tokens + torch.cat(
            (
                hand_modifiers,
                board_modifiers,
            ),
            dim=1,
        )
        entity_tokens = entity_tokens + torch.cat(
            (
                hand_effects
                + self.v41_hand_effect_summary_projection(
                    self._field(
                        observation, "hand_effect_summary"
                    ).reshape(
                        batch,
                        ShadowverseEnv.MAX_HAND,
                        v4_1.HAND_EFFECT_SUMMARY_SIZE,
                    )
                ),
                board_effects
                + self.v41_board_effect_summary_projection(
                    self._field(
                        observation, "board_effect_summary"
                    ).reshape(
                        batch,
                        2 * ShadowverseEnv.MAX_BOARD,
                        v4_1.BOARD_EFFECT_SUMMARY_SIZE,
                    )
                ),
            ),
            dim=1,
        )
        entity_tokens = entity_tokens + torch.cat(
            (hand_fusion, board_fusion), dim=1
        )
        entity_tokens = entity_tokens + self.v41_listener_projection(
            listeners[:, : self.entity_slot_count]
        )

        leader_card_indices = self._card_index_field(
            card_indices, "leader_area_cards"
        )
        leader_tokens = self._card_field(
            card_tokens, "leader_area_cards"
        )
        leader_state = self._field(
            observation, "leader_area_state"
        ).reshape(
            batch,
            v4_1.LEADER_AREA_SLOTS,
            v4_1.LEADER_AREA_STATE_SIZE,
        )
        leader_types = torch.round(
            leader_state[..., 1]
        ).to(dtype=torch.long).clamp(min=0, max=2)
        leader_relations = torch.round(
            leader_state[..., 2]
        ).to(dtype=torch.long).clamp(min=0, max=2)
        projected_leader_state = leader_state.clone()
        projected_leader_state[..., 1:3] = 0
        leader_effects = self._v41_effect_context(
            observation,
            field="leader_effect_tokens",
            entity_count=v4_1.LEADER_AREA_SLOTS,
            effects_per_entity=v4_1.LEADER_EFFECTS_PER_SLOT,
        )
        leader_positions = torch.arange(
            v4_1.LEADER_AREA_SLOTS, device=observation.device
        )
        leader_tokens = (
            leader_tokens
            + self.v41_leader_state_projection(projected_leader_state)
            + self.v41_leader_area_type_embedding(leader_types)
            + self.leader_relation_embedding(leader_relations)
            + self.v41_leader_slot_embedding(
                leader_positions
            ).unsqueeze(0)
            + leader_effects
            + self.v41_leader_effect_summary_projection(
                self._field(
                    observation, "leader_effect_summary"
                ).reshape(
                    batch,
                    v4_1.LEADER_AREA_SLOTS,
                    v4_1.LEADER_EFFECT_SUMMARY_SIZE,
                )
            )
            + self.v41_listener_projection(
                listeners[:, self.entity_slot_count :]
            )
        )

        zone_card_indices = self._card_index_field(
            card_indices, "zone_cards"
        ).reshape(
            batch, v4_1.ZONE_GROUPS, v4_1.MAX_ZONE_CARD_KINDS
        )
        zone_card_tokens = self._card_field(
            card_tokens, "zone_cards"
        ).reshape(
            batch,
            v4_1.ZONE_GROUPS,
            v4_1.MAX_ZONE_CARD_KINDS,
            self.model_dim,
        )
        zone_counts = self._field(
            observation, "zone_counts"
        ).reshape(
            batch, v4_1.ZONE_GROUPS, v4_1.MAX_ZONE_CARD_KINDS, 1
        )
        zone_positions = torch.arange(
            v4_1.ZONE_GROUPS, device=observation.device
        )
        zone_count_context = self.v41_zone_card_count_projection(
            zone_counts
        )
        zone_rows = (
            zone_card_tokens * (1.0 + torch.tanh(zone_count_context))
            + zone_count_context
            + self.v41_zone_embedding(zone_positions)[None, :, None, :]
        )
        zone_tokens = self._v41_masked_mean(
            zone_rows, zone_card_indices != 0, dim=2
        )
        zone_tokens = (
            zone_tokens
            + self.v41_zone_embedding(zone_positions).unsqueeze(0)
            + self.v41_zone_overflow_projection(
                self._field(observation, "zone_overflow").reshape(
                    batch, v4_1.ZONE_GROUPS, 2
                )
            )
            + self.v41_zone_kind_count_projection(
                (zone_card_indices != 0).sum(
                    dim=2, keepdim=True
                ) / v4_1.MAX_ZONE_CARD_KINDS
            )
        )
        deck_card_indices = self._card_index_field(
            card_indices, "own_deck_cards"
        )
        deck_state_context = self.v41_deck_state_projection(
            self._field(observation, "own_deck_state").reshape(
                batch, v4_1.MAX_DECK_CARDS, 4
            )
        )
        deck_rows = (
            self._card_field(card_tokens, "own_deck_cards")
            * (1.0 + torch.tanh(deck_state_context))
            + deck_state_context
        )
        deck_token = self._v41_masked_mean(
            deck_rows, deck_card_indices != 0, dim=1
        )
        zone_tokens[:, 2] = zone_tokens[:, 2] + deck_token

        event_types = torch.round(
            self._field(observation, "history_event_types")
        ).to(dtype=torch.long).clamp(
            min=0,
            max=self.v41_history_event_embedding.num_embeddings - 1,
        )
        event_actors = torch.round(
            self._field(observation, "history_actors")
        ).to(dtype=torch.long).clamp(min=0, max=2)
        source_references = torch.round(
            self._field(observation, "history_source_references")
        ).to(dtype=torch.long).clamp(
            min=0, max=v4_1.CHOICE_REFERENCE_COUNT - 1
        )
        target_references = torch.round(
            self._field(observation, "history_target_references")
        ).to(dtype=torch.long).clamp(
            min=0, max=v4_1.CHOICE_REFERENCE_COUNT - 1
        )
        history_positions = torch.arange(
            v4_1.HISTORY_LENGTH, device=observation.device
        )
        history_semantics = self._v41_semantic_context(
            self._field(observation, "history_semantics").reshape(
                batch,
                v4_1.HISTORY_LENGTH,
                v4_1.SEMANTIC_TOKEN_SIZE,
            )
        )
        history_tokens = (
            self.v41_history_event_embedding(event_types)
            + self.leader_relation_embedding(event_actors)
            + self.v41_scalar_projection(
                self._field(observation, "history_amounts").unsqueeze(-1)
            )
            + self.v41_history_source_reference_embedding(
                source_references
            )
            + self.v41_history_target_reference_embedding(
                target_references
            )
            + self.v41_history_source_card_projection(
                self._card_field(card_tokens, "history_source_cards")
            )
            + self.v41_history_target_card_projection(
                self._card_field(card_tokens, "history_target_cards")
            )
            + history_semantics
            + self.v41_history_position_embedding(
                history_positions
            ).unsqueeze(0)
        )

        record_card_indices = self._card_index_field(
            card_indices, "record_cards"
        ).reshape(
            batch,
            v4_1.RECORD_GROUPS,
            v4_1.HISTORY_RECORDS_PER_GROUP,
        )
        record_state = self._field(
            observation, "record_state"
        ).reshape(
            batch,
            v4_1.RECORD_GROUPS,
            v4_1.HISTORY_RECORDS_PER_GROUP,
            v4_1.RECORD_STATE_SIZE,
        )
        record_kind = torch.round(
            record_state[..., 1]
        ).to(dtype=torch.long).clamp(min=0, max=3)
        record_relation = torch.round(
            record_state[..., 2]
        ).to(dtype=torch.long).clamp(min=0, max=2)
        record_origins = torch.round(
            record_state[..., 3:5]
        ).to(dtype=torch.long).clamp(
            min=0,
            max=self.v41_origin_embedding.num_embeddings - 1,
        )
        projected_record_state = record_state.clone()
        projected_record_state[..., 1:5] = 0
        record_semantics = self._v41_semantic_context(
            self._field(observation, "record_semantics").reshape(
                batch,
                v4_1.RECORD_GROUPS,
                v4_1.HISTORY_RECORDS_PER_GROUP,
                v4_1.SEMANTIC_TOKEN_SIZE,
            )
        )
        record_positions = torch.arange(
            v4_1.HISTORY_RECORDS_PER_GROUP,
            device=observation.device,
        )
        record_context = (
            self.v41_record_state_projection(projected_record_state)
            + self.v41_record_kind_embedding(record_kind)
            + self.leader_relation_embedding(record_relation)
            + self.v41_origin_embedding(record_origins[..., 0])
            + self.v41_source_origin_embedding(record_origins[..., 1])
            + record_semantics
            + self.v41_record_position_embedding(
                record_positions
            )[None, None, :, :]
        )
        record_rows = (
            self._card_field(card_tokens, "record_cards").reshape(
                batch,
                v4_1.RECORD_GROUPS,
                v4_1.HISTORY_RECORDS_PER_GROUP,
                self.model_dim,
            )
            * (1.0 + torch.tanh(record_context))
            + record_context
        )
        record_tokens = self._v41_masked_mean(
            record_rows, record_card_indices != 0, dim=2
        )
        record_groups = torch.arange(
            v4_1.RECORD_GROUPS, device=observation.device
        )
        record_tokens = record_tokens + self.v41_record_group_embedding(
            record_groups
        ).unsqueeze(0) + self.v41_scalar_projection(
            (record_card_indices != 0).sum(
                dim=2, keepdim=True
            ) / v4_1.HISTORY_RECORDS_PER_GROUP
        )

        tokens = torch.cat(
            (
                global_token.unsqueeze(1),
                player_tokens,
                entity_tokens,
                leader_tokens,
                zone_tokens,
                history_tokens,
                record_tokens,
            ),
            dim=1,
        )
        padding = torch.cat(
            (
                torch.zeros(
                    batch,
                    3,
                    dtype=torch.bool,
                    device=observation.device,
                ),
                entity_card_indices == 0,
                leader_card_indices == 0,
                torch.zeros(
                    batch,
                    v4_1.ZONE_GROUPS,
                    dtype=torch.bool,
                    device=observation.device,
                ),
                event_types == 0,
                torch.zeros(
                    batch,
                    v4_1.RECORD_GROUPS,
                    dtype=torch.bool,
                    device=observation.device,
                ),
            ),
            dim=1,
        )
        if tokens.shape[1] != self.structured_token_count:
            raise AssertionError(
                (tokens.shape[1], self.structured_token_count)
            )
        contextual = self.entity_encoder(
            tokens, src_key_padding_mask=padding
        )
        next_hidden = self.recurrent(contextual[:, 0], hidden)
        contextual_entities = contextual[:, 3 : 3 + self.entity_slot_count]

        source = self.source_projection(
            self._gather_entities(contextual_entities, self._source_slot)
        )
        target_slots = self._target_slot.unsqueeze(0).expand(
            batch, -1
        ).clone()
        leader_relations = self._leader_relation.unsqueeze(0).expand(
            batch, -1
        ).clone()
        fallback_ordinals = self._option_ordinal.unsqueeze(0).expand(
            batch, -1
        ).clone()
        choice_slots, choice_relations, choice_fallback = (
            self._choice_targets(observation)
        )
        choice_slice = slice(
            ShadowverseEnv.CHOICE_OFFSET,
            ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
        )
        target_slots[:, choice_slice] = choice_slots
        leader_relations[:, choice_slice] = choice_relations
        fallback_ordinals[:, choice_slice] = torch.where(
            choice_fallback,
            fallback_ordinals[:, choice_slice],
            torch.zeros_like(fallback_ordinals[:, choice_slice]),
        )
        target = self.target_projection(
            self._gather_entities(contextual_entities, target_slots)
        )

        choice_semantics = self._v41_semantic_context(
            self._field(
                observation, "choice_option_semantics"
            ).reshape(
                batch,
                ShadowverseEnv.MAX_CHOICE_OPTIONS,
                v4_1.SEMANTIC_TOKEN_SIZE,
            )
        )
        choice_references = torch.round(
            self._field(observation, "choice_option_references")
        ).to(dtype=torch.long).clamp(
            min=0, max=v4_1.CHOICE_REFERENCE_COUNT - 1
        )
        choice_relations_direct = torch.round(
            self._field(observation, "choice_option_relations")
        ).to(dtype=torch.long).clamp(min=0, max=2)
        choice_action_features = (
            self._card_field(card_tokens, "choice_option_cards")
            + self.v41_reference_embedding(choice_references)
            + self.leader_relation_embedding(choice_relations_direct)
            + self.v41_scalar_projection(
                self._field(
                    observation, "choice_option_selected"
                ).unsqueeze(-1)
            )
            + choice_semantics
        )
        graveyard_state = self._field(
            observation, "graveyard_option_state"
        ).reshape(
            batch,
            ShadowverseEnv.GRAVEYARD_PAGE_SIZE,
            5,
        )
        graveyard_origins = torch.round(
            graveyard_state[..., 1:3]
        ).to(dtype=torch.long).clamp(
            min=0,
            max=self.v41_origin_embedding.num_embeddings - 1,
        )
        projected_graveyard_state = graveyard_state.clone()
        projected_graveyard_state[..., 1:3] = 0
        graveyard_action_features = (
            self._card_field(card_tokens, "graveyard_page_cards")
            + self.v41_graveyard_state_projection(
                projected_graveyard_state
            )
            + self.v41_origin_embedding(graveyard_origins[..., 0])
            + self.v41_source_origin_embedding(graveyard_origins[..., 1])
            + self._v41_semantic_context(
                self._field(
                    observation, "graveyard_option_semantics"
                ).reshape(
                    batch,
                    ShadowverseEnv.GRAVEYARD_PAGE_SIZE,
                    v4_1.SEMANTIC_TOKEN_SIZE,
                )
            )
        )
        action_features = (
            self.action_kind_embedding(self._action_kind).unsqueeze(0)
            + self.mode_embedding(self._action_mode).unsqueeze(0)
            + self.leader_relation_embedding(leader_relations)
            + self.option_ordinal_embedding(fallback_ordinals)
            + source
            + target
            + self.hidden_projection(next_hidden).unsqueeze(1)
        )
        action_features[
            :,
            ShadowverseEnv.CHOICE_OFFSET :
            ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
        ] += choice_action_features
        action_features[
            :,
            ShadowverseEnv.GRAVEYARD_SLOT_OFFSET :
            ShadowverseEnv.MODE_PLAY_OFFSET,
        ] += graveyard_action_features
        logits = self.policy_head(action_features).squeeze(-1)
        value = self.value_head(next_hidden).squeeze(-1)
        return logits, value, next_hidden

    @staticmethod
    def _gather_entities(
        entities: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, model_dim = entities.shape
        if slots.ndim == 1:
            slots = slots.unsqueeze(0).expand(batch, -1)
        safe_slots = slots.clamp(min=0)
        gathered = torch.gather(
            entities,
            1,
            safe_slots.unsqueeze(-1).expand(-1, -1, model_dim),
        )
        return gathered * (slots >= 0).unsqueeze(-1).to(dtype=entities.dtype)

    def _choice_targets(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.v4_1_observation:
            references = torch.round(
                self._field(observation, "choice_option_references")
            ).to(dtype=torch.long)
            relations = torch.round(
                self._field(observation, "choice_option_relations")
            ).to(dtype=torch.long).clamp(min=0, max=2)
        elif self.v4_observation:
            choice = self._field(
                observation, "choice_option_state"
            ).reshape(
                observation.shape[0],
                ShadowverseEnv.MAX_CHOICE_OPTIONS,
                CHOICE_OPTION_STATE_SIZE,
            )
            reference_bits = choice[:, :, :CHOICE_REFERENCE_COUNT]
            relation_bits = choice[
                :,
                :,
                CHOICE_REFERENCE_COUNT :
                CHOICE_REFERENCE_COUNT + self._LEADER_RELATION_COUNT,
            ]
            references = reference_bits.argmax(dim=-1)
            relations = relation_bits.argmax(dim=-1)
            present = choice.abs().sum(dim=-1) > 0
            references = torch.where(
                present, references, torch.zeros_like(references)
            )
            relations = torch.where(
                present, relations, torch.zeros_like(relations)
            )
        else:
            choice = torch.round(
                self._field(observation, "choice_categorical") * 1024.0
            ).to(dtype=torch.long)
            references = choice[:, 4 : 4 + ShadowverseEnv.MAX_CHOICE_OPTIONS]
            relations = choice[
                :,
                4 + ShadowverseEnv.MAX_CHOICE_OPTIONS :
                4 + 2 * ShadowverseEnv.MAX_CHOICE_OPTIONS,
            ].clamp(min=0, max=2)
        slots = torch.full_like(references, -1)
        own_board = (references >= 1) & (
            references <= ShadowverseEnv.MAX_BOARD
        )
        enemy_board = (references > ShadowverseEnv.MAX_BOARD) & (
            references <= 2 * ShadowverseEnv.MAX_BOARD
        )
        own_hand = (references > 2 * ShadowverseEnv.MAX_BOARD) & (
            references <= 2 * ShadowverseEnv.MAX_BOARD + ShadowverseEnv.MAX_HAND
        )
        slots = torch.where(
            own_board,
            ShadowverseEnv.MAX_HAND + references - 1,
            slots,
        )
        slots = torch.where(
            enemy_board,
            ShadowverseEnv.MAX_HAND + references - 1,
            slots,
        )
        slots = torch.where(
            own_hand,
            references - (2 * ShadowverseEnv.MAX_BOARD + 1),
            slots,
        )
        fallback = (slots < 0) & (relations == 0)
        return slots, relations, fallback

    def forward_step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor,
        card_indices: torch.Tensor | None = None,
        *,
        validate_card_indices: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != self.input_size:
            raise ValueError("entity-action observations must be [batch, input_size]")
        if card_indices is None:
            raise ValueError("card indices are required by this policy")
        if card_indices.shape != (observation.shape[0], self.card_slot_count):
            raise ValueError("card index batch shape must match entity slots")
        if validate_card_indices and (
            bool((card_indices < 0).any())
            or bool(
                (card_indices > self.card_vocabulary_size).any()
            )
        ):
            raise ValueError("card index is outside the policy vocabulary")
        if self.v4_1_observation:
            return self._forward_step_v4_1(
                observation, hidden, card_indices
            )

        global_observation = observation * self._global_keep.to(
            dtype=observation.dtype
        )
        if self.v4_observation:
            global_observation = global_observation.index_select(
                -1, self._global_indices
            )
        global_token = self.global_projection(global_observation)
        card_tokens = self.card_projection(
            self.card_embedding(card_indices.to(dtype=torch.long))
        )
        global_token = global_token + self._auxiliary_context(
            card_tokens, card_indices
        )
        entity_card_tokens = self._entity_card_tokens(card_tokens)
        entity_card_indices = self._entity_card_indices(card_indices)
        entity_tokens = (
            entity_card_tokens
            + self.entity_feature_projection(self._entity_features(observation))
            + self.zone_embedding(self._entity_zones).unsqueeze(0)
            + self.zone_slot_embedding(self._entity_zone_slots).unsqueeze(0)
        )
        if self.v4_observation:
            hand_fusion = self._masked_group_mean(
                self._card_field(card_tokens, "own_hand_fusion_cards"),
                self._card_index_field(card_indices, "own_hand_fusion_cards"),
                ShadowverseEnv.MAX_HAND,
            )
            board_fusion = self._masked_group_mean(
                self._card_field(card_tokens, "public_board_fusion_cards"),
                self._card_index_field(card_indices, "public_board_fusion_cards"),
                2 * ShadowverseEnv.MAX_BOARD,
            )
            entity_tokens = entity_tokens + torch.cat(
                (hand_fusion, board_fusion), dim=1
            )
        tokens = torch.cat((global_token.unsqueeze(1), entity_tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros(
                    observation.shape[0],
                    1,
                    dtype=torch.bool,
                    device=observation.device,
                ),
                entity_card_indices == 0,
            ),
            dim=1,
        )
        contextual = self.entity_encoder(
            tokens, src_key_padding_mask=padding
        )
        next_hidden = self.recurrent(contextual[:, 0], hidden)
        contextual_entities = contextual[:, 1:]

        source = self.source_projection(
            self._gather_entities(contextual_entities, self._source_slot)
        )
        target_slots = self._target_slot.unsqueeze(0).expand(
            observation.shape[0], -1
        ).clone()
        leader_relations = self._leader_relation.unsqueeze(0).expand(
            observation.shape[0], -1
        ).clone()
        fallback_ordinals = self._option_ordinal.unsqueeze(0).expand(
            observation.shape[0], -1
        ).clone()
        choice_slots, choice_relations, choice_fallback = self._choice_targets(
            observation
        )
        choice_slice = slice(
            ShadowverseEnv.CHOICE_OFFSET,
            ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
        )
        target_slots[:, choice_slice] = choice_slots
        leader_relations[:, choice_slice] = choice_relations
        fallback_ordinals[:, choice_slice] = torch.where(
            choice_fallback,
            fallback_ordinals[:, choice_slice],
            torch.zeros_like(fallback_ordinals[:, choice_slice]),
        )
        target = self.target_projection(
            self._gather_entities(contextual_entities, target_slots)
        )
        choice_action_features = None
        graveyard_action_features = None
        if self.v4_observation:
            choice_options = self._field(
                observation, "choice_option_state"
            ).reshape(
                observation.shape[0],
                ShadowverseEnv.MAX_CHOICE_OPTIONS,
                CHOICE_OPTION_STATE_SIZE,
            )
            choice_action_features = self.choice_option_projection(
                choice_options
            )
            choice_action_features = choice_action_features + self.target_projection(
                self._card_field(card_tokens, "choice_option_cards")
            )
            graveyard_options = self._field(
                observation, "graveyard_option_state"
            ).reshape(
                observation.shape[0],
                ShadowverseEnv.GRAVEYARD_PAGE_SIZE,
                GRAVEYARD_OPTION_STATE_SIZE,
            )
            graveyard_action_features = self.graveyard_option_projection(
                graveyard_options
            )
            graveyard_action_features = (
                graveyard_action_features
                + self.target_projection(
                    self._card_field(card_tokens, "graveyard_page_cards")
                )
            )
        action_features = (
            self.action_kind_embedding(self._action_kind).unsqueeze(0)
            + self.mode_embedding(self._action_mode).unsqueeze(0)
            + self.leader_relation_embedding(leader_relations)
            + self.option_ordinal_embedding(fallback_ordinals)
            + source
            + target
            + self.hidden_projection(next_hidden).unsqueeze(1)
        )
        if choice_action_features is not None:
            action_features[
                :,
                ShadowverseEnv.CHOICE_OFFSET :
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
            ] += choice_action_features
            action_features[
                :,
                ShadowverseEnv.GRAVEYARD_SLOT_OFFSET :
                ShadowverseEnv.MODE_PLAY_OFFSET,
            ] += graveyard_action_features
        logits = self.policy_head(action_features).squeeze(-1)
        value = self.value_head(next_hidden).squeeze(-1)
        return logits, value, next_hidden

    def specification(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "input_size": self.input_size,
            "action_size": self.action_size,
            "hidden_size": self.hidden_size,
            "card_vocabulary_size": self.card_vocabulary_size,
            "card_slot_count": self.card_slot_count,
            "card_embedding_dim": self.card_embedding_dim,
            "model_dim": self.model_dim,
            "transformer_layers": self.transformer_layers,
            "attention_heads": self.attention_heads,
            "feedforward_dim": self.feedforward_dim,
            "structured_token_count": self.structured_token_count,
            "field_layout": dict(self.field_layout),
            "card_field_layout": dict(self.card_field_layout),
        }


def build_policy_from_specification(
    specification: Mapping[str, object],
) -> MaskedPolicyNetwork:
    values = dict(specification)
    values.pop("structured_token_count", None)
    architecture = str(
        values.pop("architecture", LEGACY_POLICY_ARCHITECTURE)
    )
    if architecture == LEGACY_POLICY_ARCHITECTURE:
        return RecurrentMaskedActorCritic(**values)
    if architecture == ENTITY_ACTION_POLICY_ARCHITECTURE:
        return EntityActionRecurrentActorCritic(**values)
    raise ValueError(f"unsupported policy architecture {architecture!r}")
