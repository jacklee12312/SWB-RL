from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from swb.rl.class_schedule import CLASS_SCHEDULE_VERSION

if TYPE_CHECKING:
    from swb.engine.environment import ShadowverseEnv
    from swb.rl.catalog import TrainableCardCatalog


OBSERVATION_SCHEMA_VERSION = "observation-v3.5"
ACTION_LAYOUT_VERSION = "action-111-v1"
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
    space = env.observation_v3_space()
    fields = []
    for name, field in space.spaces.items():
        fields.append({
            "name": name,
            "shape": list(field.shape),
            "dtype": str(field.dtype),
            "space_type": type(field).__name__,
        })
    return {
        "version": OBSERVATION_SCHEMA_VERSION,
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
            {"name": "super_evolve", "start": env.SUPER_EVOLVE_OFFSET, "stop": env.ACTION_SIZE},
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
            observation_version=OBSERVATION_SCHEMA_VERSION,
            observation_schema_sha256=stable_json_sha256(observation),
            action_layout_version=ACTION_LAYOUT_VERSION,
            action_layout_sha256=stable_json_sha256(action),
            catalog_sha256=catalog.catalog_sha256,
            card_vocabulary_sha256=catalog.card_vocabulary_sha256,
            rulebook_sha256=rulebook_sha256,
            coverage_report_sha256=catalog.coverage_report_sha256,
            training_pool_sha256=catalog.training_pool_sha256,
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
