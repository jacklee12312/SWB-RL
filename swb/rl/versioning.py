from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from swb.engine.environment import MATCH_SETUP_LEGACY
from swb.rl.class_schedule import CLASS_SCHEDULE_VERSION

if TYPE_CHECKING:
    from swb.engine.environment import ShadowverseEnv
    from swb.rl.catalog import TrainableCardCatalog


OBSERVATION_SCHEMA_VERSIONS = {
    "v3": "observation-v3.6",
    "v4": "observation-v4.0",
    "v4.1": "observation-v4.1",
}
# Historical compatibility alias. New code should index
# OBSERVATION_SCHEMA_VERSIONS by the selected environment version.
OBSERVATION_SCHEMA_VERSION = OBSERVATION_SCHEMA_VERSIONS["v4"]
ACTION_LAYOUT_VERSION = "action-112-v2"
SEED_DERIVATION_VERSION = 1


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def observation_schema_manifest(env: ShadowverseEnv) -> dict[str, object]:
    if env.observation_version not in OBSERVATION_SCHEMA_VERSIONS:
        raise ValueError(
            "versioned observation manifest requires observation_version "
            "'v3', 'v4', or 'v4.1'"
        )
    if env.observation_version == "v4.1":
        space = env.observation_v4_1_space()
    elif env.observation_version == "v4":
        space = env.observation_v4_space()
    else:
        space = env.observation_v3_space()
    fields = []
    for name, field in space.spaces.items():
        fields.append({
            "name": name,
            "shape": list(field.shape),
            "dtype": str(field.dtype),
            "space_type": type(field).__name__,
        })
    manifest = {
        "version": OBSERVATION_SCHEMA_VERSIONS[env.observation_version],
        "fields": fields,
        "card_vocabulary_size": len(env.card_vocabulary),
        "privacy": {
            "opponent_hand_identity": "hidden",
            "opponent_deck_order": "hidden",
            "opponent_initial_deck": (
                "explicit-open-decklists" if env.open_decklists else "hidden"
            ),
            "raw_entity_ids": "excluded",
            "non_decision_action_mask": "all-zero",
        },
    }
    if env.observation_version == "v4":
        from swb.engine import observation_v4

        manifest["encoding"] = {
            "categorical_values": "one-hot",
            "card_identity": "shared-card-vocabulary-index",
            "structured_effect_identity": (
                f"stable-sha256-{observation_v4.SEMANTIC_BITS}-bits"
            ),
            "histogram_scaling": "divide-by-40-in-policy-flattener",
            "raw_entity_ids": "never-encoded",
        }
        manifest["fixed_limits"] = {
            "public_history": observation_v4.HISTORY_LENGTH,
            "history_records_per_player": (
                observation_v4.HISTORY_RECORDS_PER_PLAYER
            ),
            "leader_area_slots": observation_v4.MAX_LEADER_AREA_SLOTS,
            "cost_modifiers_per_card": observation_v4.MAX_COST_MODIFIERS,
            "stat_modifiers_per_card": observation_v4.MAX_STAT_MODIFIERS,
            "keyword_modifiers_per_card": (
                observation_v4.MAX_KEYWORD_MODIFIERS
            ),
            "listeners_per_source": observation_v4.MAX_LISTENERS_PER_SOURCE,
            "granted_abilities_per_source": (
                observation_v4.MAX_GRANTED_ABILITIES
            ),
        }
    elif env.observation_version == "v4.1":
        from swb.engine import observation_v4_1

        manifest["encoding"] = {
            "categorical_values": "typed-indices",
            "card_identity": "shared-card-vocabulary-index",
            "structured_effect_identity": (
                "stable-sha256-32-bits-as-four-byte-tokens"
            ),
            "zone_identity": "sparse-card-count-pairs-with-overflow",
            "current_own_deck": "order-independent-physical-card-rows",
            "raw_entity_ids": "never-encoded",
        }
        manifest["fixed_limits"] = {
            "transformer_tokens": observation_v4_1.STRUCTURED_TOKEN_COUNT,
            "public_history": observation_v4_1.HISTORY_LENGTH,
            "history_records_per_group": (
                observation_v4_1.HISTORY_RECORDS_PER_GROUP
            ),
            "record_groups": observation_v4_1.RECORD_GROUPS,
            "leader_area_slots": observation_v4_1.LEADER_AREA_SLOTS,
            "zone_groups": observation_v4_1.ZONE_GROUPS,
            "zone_card_kinds_per_group": (
                observation_v4_1.MAX_ZONE_CARD_KINDS
            ),
            "hand_modifiers_per_card": (
                observation_v4_1.MAX_HAND_MODIFIERS
            ),
            "board_modifiers_per_card": (
                observation_v4_1.MAX_BOARD_MODIFIERS
            ),
            "listeners_per_source": (
                observation_v4_1.v4.MAX_LISTENERS_PER_SOURCE
            ),
            "granted_abilities_per_source": (
                observation_v4_1.v4.MAX_GRANTED_ABILITIES
            ),
        }
    return manifest


def action_layout_manifest(env: ShadowverseEnv) -> dict[str, object]:
    return {
        "version": ACTION_LAYOUT_VERSION,
        "size": env.ACTION_SIZE,
        "ranges": [
            {"name": "end_turn", "start": env.END_TURN, "stop": env.PLAY_OFFSET},
            {"name": "play_normal", "start": env.PLAY_OFFSET, "stop": env.ATTACK_OFFSET},
            {"name": "attack", "start": env.ATTACK_OFFSET, "stop": env.EVOLVE_OFFSET},
            {"name": "evolve_or_activate", "start": env.EVOLVE_OFFSET, "stop": env.CHOICE_OFFSET},
            {"name": "choice", "start": env.CHOICE_OFFSET, "stop": env.GRAVEYARD_CHOICE_OFFSET},
            {"name": "graveyard_navigation", "start": env.GRAVEYARD_CHOICE_OFFSET, "stop": env.GRAVEYARD_SLOT_OFFSET},
            {"name": "graveyard_slots", "start": env.GRAVEYARD_SLOT_OFFSET, "stop": env.MODE_PLAY_OFFSET},
            {"name": "fusion_or_special_mode", "start": env.MODE_PLAY_OFFSET, "stop": env.SUPER_EVOLVE_OFFSET},
            {"name": "super_evolve", "start": env.SUPER_EVOLVE_OFFSET, "stop": env.USE_EXTRA_PP},
            {"name": "use_extra_pp", "start": env.USE_EXTRA_PP, "stop": env.ACTION_SIZE},
        ],
        "max_hand": env.MAX_HAND,
        "max_board": env.MAX_BOARD,
        "max_choice_options": env.MAX_CHOICE_OPTIONS,
        "graveyard_page_size": env.GRAVEYARD_PAGE_SIZE,
    }


@dataclass(frozen=True)
class ExperimentVersions:
    observation_version: str
    observation_schema_sha256: str
    action_layout_version: str
    action_layout_sha256: str
    catalog_sha256: str
    card_vocabulary_sha256: str
    rulebook_sha256: str
    coverage_report_sha256: str
    training_pool_sha256: str
    seed_derivation_version: int = SEED_DERIVATION_VERSION
    class_schedule_version: int = CLASS_SCHEDULE_VERSION
    match_setup: str = MATCH_SETUP_LEGACY

    @classmethod
    def capture(
        cls,
        env: ShadowverseEnv,
        catalog: TrainableCardCatalog,
        *,
        rulebook_sha256: str,
    ) -> ExperimentVersions:
        observation = observation_schema_manifest(env)
        action = action_layout_manifest(env)
        return cls(
            observation_version=str(observation["version"]),
            observation_schema_sha256=stable_json_sha256(observation),
            action_layout_version=ACTION_LAYOUT_VERSION,
            action_layout_sha256=stable_json_sha256(action),
            catalog_sha256=catalog.catalog_sha256,
            card_vocabulary_sha256=catalog.card_vocabulary_sha256,
            rulebook_sha256=rulebook_sha256,
            coverage_report_sha256=catalog.coverage_report_sha256,
            training_pool_sha256=catalog.training_pool_sha256,
            match_setup=env.match_setup,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def assert_compatible(self, actual: ExperimentVersions) -> None:
        expected_values = self.to_dict()
        actual_values = actual.to_dict()
        mismatches = {
            key: {"checkpoint": expected_values[key], "runtime": actual_values[key]}
            for key in expected_values
            if expected_values[key] != actual_values[key]
        }
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={values['checkpoint']!r}, "
                f"runtime={values['runtime']!r}"
                for key, values in mismatches.items()
            )
            raise ValueError(f"Incompatible experiment versions: {details}")
