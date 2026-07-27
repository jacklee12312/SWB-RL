from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from swb.engine.environment import ShadowverseEnv
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
    def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        mask = action_mask.to(dtype=torch.bool)
        if mask.ndim != logits.ndim or mask.shape != logits.shape:
            raise ValueError("action mask shape must match policy logits")
        if not bool(mask.any(dim=-1).all()):
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
            if bool((card_indices < 0).any()) or bool(
                (card_indices > self.card_vocabulary_size).any()
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
    ) -> None:
        super().__init__()
        if action_size != ShadowverseEnv.ACTION_SIZE:
            raise ValueError(
                "entity-action-v1 requires the versioned 112-action layout"
            )
        expected_card_slots = ShadowverseEnv.MAX_HAND + 2 * ShadowverseEnv.MAX_BOARD
        if card_slot_count != expected_card_slots:
            raise ValueError(
                f"entity-action-v1 requires {expected_card_slots} card slots"
            )
        if model_dim <= 0 or transformer_layers <= 0 or attention_heads <= 0:
            raise ValueError("entity-action dimensions must be positive")
        if model_dim % attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
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
        self.card_embedding_dim = card_embedding_dim
        self.model_dim = model_dim
        self.transformer_layers = transformer_layers
        self.attention_heads = attention_heads
        self.feedforward_dim = feedforward_dim
        self.field_layout = {
            name: (int(offset), int(size))
            for name, (offset, size) in field_layout.items()
        }

        global_keep = torch.ones(input_size, dtype=torch.float32)
        for name in required_fields - {"choice_categorical"}:
            offset, size = self.field_layout[name]
            global_keep[offset : offset + size] = 0.0
        choice_offset, choice_size = self.field_layout["choice_categorical"]
        global_keep[choice_offset + 4 : choice_offset + choice_size] = 0.0
        self.register_buffer("_global_keep", global_keep, persistent=False)

        self.card_embedding = nn.Embedding(
            card_vocabulary_size + 1,
            card_embedding_dim,
            padding_idx=0,
        )
        self.card_projection = nn.Linear(card_embedding_dim, model_dim)
        self.entity_feature_projection = nn.Linear(
            self._ENTITY_FEATURE_SIZE, model_dim, bias=False
        )
        self.zone_embedding = nn.Embedding(3, model_dim)
        self.zone_slot_embedding = nn.Embedding(
            max(ShadowverseEnv.MAX_HAND, ShadowverseEnv.MAX_BOARD),
            model_dim,
        )
        self.global_projection = nn.Sequential(
            nn.Linear(input_size, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
        )
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

    def _entity_features(self, observation: torch.Tensor) -> torch.Tensor:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.ndim != 2 or observation.shape[-1] != self.input_size:
            raise ValueError("entity-action observations must be [batch, input_size]")
        if card_indices is None:
            raise ValueError("card indices are required by this policy")
        if card_indices.shape != (observation.shape[0], self.card_slot_count):
            raise ValueError("card index batch shape must match entity slots")
        if bool((card_indices < 0).any()) or bool(
            (card_indices > self.card_vocabulary_size).any()
        ):
            raise ValueError("card index is outside the policy vocabulary")

        global_token = self.global_projection(
            observation * self._global_keep.to(dtype=observation.dtype)
        )
        card_tokens = self.card_projection(
            self.card_embedding(card_indices.to(dtype=torch.long))
        )
        entity_tokens = (
            card_tokens
            + self.entity_feature_projection(self._entity_features(observation))
            + self.zone_embedding(self._entity_zones).unsqueeze(0)
            + self.zone_slot_embedding(self._entity_zone_slots).unsqueeze(0)
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
                card_indices == 0,
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
        action_features = (
            self.action_kind_embedding(self._action_kind).unsqueeze(0)
            + self.mode_embedding(self._action_mode).unsqueeze(0)
            + self.leader_relation_embedding(leader_relations)
            + self.option_ordinal_embedding(fallback_ordinals)
            + source
            + target
            + self.hidden_projection(next_hidden).unsqueeze(1)
        )
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
            "field_layout": dict(self.field_layout),
        }


def build_policy_from_specification(
    specification: Mapping[str, object],
) -> MaskedPolicyNetwork:
    values = dict(specification)
    architecture = str(
        values.pop("architecture", LEGACY_POLICY_ARCHITECTURE)
    )
    if architecture == LEGACY_POLICY_ARCHITECTURE:
        return RecurrentMaskedActorCritic(**values)
    if architecture == ENTITY_ACTION_POLICY_ARCHITECTURE:
        return EntityActionRecurrentActorCritic(**values)
    raise ValueError(f"unsupported policy architecture {architecture!r}")
