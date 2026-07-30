"""Audit zone capacity, ownership, leader-area, and class-resource contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.effects import ConditionType, EffectKind, ExprType


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_RULES = Path("data/rules")
DEFAULT_CLOSURE = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)
DEFAULT_COVERAGE = Path("data/reports/rule_coverage.json")
DEFAULT_EVIDENCE = Path("data/audits/zone_resource_evidence.json")
DEFAULT_OUTPUT = Path(
    "data/reports/card_bug_audit/zone_resource_audit.json"
)
DEFAULT_MARKDOWN = Path(
    "data/reports/card_bug_audit/zone_resource_audit.md"
)

ZONE_CATEGORIES = (
    "draw",
    "add_to_hand",
    "discard",
    "return_to_hand",
    "banish",
    "transform",
    "return_to_deck",
    "summon",
    "destroy",
    "countdown_or_activate",
    "leader_area",
    "empty_deck",
)
RESOURCE_CATEGORIES = (
    "combo",
    "cooperation",
    "shadows_necromancy",
    "overflow",
    "earth_sigils",
    "spellboost",
    "fusion",
    "union_burst",
    "super_skybound_art",
)
CATEGORIES = ZONE_CATEGORIES + RESOURCE_CATEGORIES

EFFECT_CATEGORIES = {
    EffectKind.DRAW: ("draw",),
    EffectKind.DRAW_FILTERED: ("draw",),
    EffectKind.REDRAW_HAND: ("draw", "discard"),
    EffectKind.ADD_CARD: ("add_to_hand",),
    EffectKind.COPY_TO_HAND: ("add_to_hand",),
    EffectKind.COPY_DESTROYED_FOLLOWERS_TO_HAND: ("add_to_hand",),
    EffectKind.COPY_RANDOM_ENEMY_DECK_TO_HAND: ("add_to_hand",),
    EffectKind.COPY_LEFTMOST_HAND_TO_HAND: ("add_to_hand",),
    EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND: (
        "add_to_hand",
        "return_to_hand",
    ),
    EffectKind.DISCARD: ("discard",),
    EffectKind.RETURN_TO_HAND: ("return_to_hand",),
    EffectKind.BANISH: ("banish",),
    EffectKind.BANISH_SAME_NAME: ("banish",),
    EffectKind.BANISH_DECK_FILTERED: ("banish",),
    EffectKind.BANISH_DECK_DUPLICATES: ("banish",),
    EffectKind.BANISH_FROM_GRAVEYARD: ("banish",),
    EffectKind.TRANSFORM: ("transform",),
    EffectKind.TRANSFORM_BOARD_FROM_RANDOM_OWN_DECK: ("transform",),
    EffectKind.TRANSFORM_DECK_CARDS: ("transform",),
    EffectKind.TRANSFORM_HAND_FROM_RANDOM_ENEMY_DECK: ("transform",),
    EffectKind.RETURN_TO_DECK: ("return_to_deck",),
    EffectKind.ADD_CARD_TO_DECK: ("return_to_deck",),
    EffectKind.REPLACE_DECK: ("return_to_deck", "empty_deck"),
    EffectKind.SET_EMPTY_DECK_OUTCOME: ("empty_deck",),
    EffectKind.SUMMON: ("summon",),
    EffectKind.SUMMON_COPY: ("summon",),
    EffectKind.SUMMON_EXACT_COPY: ("summon",),
    EffectKind.SUMMON_HAND_COPY: ("summon",),
    EffectKind.SUMMON_FROM_HAND: ("summon",),
    EffectKind.SUMMON_FROM_DECK: ("summon",),
    EffectKind.SUMMON_DESTROYED_AMULETS: ("summon",),
    EffectKind.SUMMON_FROM_GRAVEYARD: ("summon",),
    EffectKind.REANIMATE: ("summon", "shadows_necromancy"),
    EffectKind.DESTROY: ("destroy",),
    EffectKind.REDUCE_COUNTDOWN: ("countdown_or_activate",),
    EffectKind.INCREASE_COUNTDOWN: ("countdown_or_activate",),
    EffectKind.GAIN_EMBLEM: ("leader_area",),
    EffectKind.ADD_EMBLEM: ("leader_area",),
    EffectKind.REMOVE_EMBLEM: ("leader_area",),
    EffectKind.REMOVE_ALL_EMBLEMS: ("leader_area",),
    EffectKind.CONSUME_FAITH: ("leader_area",),
    EffectKind.GRANT_FAITH_ABILITY: ("leader_area",),
    EffectKind.GRANT_FAITH_MODE_SELECTION_BONUS: ("leader_area",),
    EffectKind.ADD_COMBO: ("combo",),
    EffectKind.ADD_SHADOWS: ("shadows_necromancy",),
    EffectKind.NECROMANCY: ("shadows_necromancy",),
    EffectKind.ADD_EARTH_SIGILS: ("earth_sigils",),
    EffectKind.EARTH_RITE: ("earth_sigils",),
    EffectKind.SPELLBOOST_HAND: ("spellboost",),
    EffectKind.ADD_UNION_BURST_GAUGE: (
        "union_burst",
        "super_skybound_art",
    ),
}

CONDITION_CATEGORIES = {
    ConditionType.CONTROLLER_SHADOWS_AT_LEAST: "shadows_necromancy",
    ConditionType.OPPONENT_SHADOWS_AT_LEAST: "shadows_necromancy",
    ConditionType.CONTROLLER_COOPERATION_AT_LEAST: "cooperation",
    ConditionType.OPPONENT_COOPERATION_AT_LEAST: "cooperation",
    ConditionType.CONTROLLER_OVERFLOW: "overflow",
    ConditionType.OPPONENT_OVERFLOW: "overflow",
    ConditionType.CONTROLLER_COMBO_AT_LEAST: "combo",
    ConditionType.OPPONENT_COMBO_AT_LEAST: "combo",
    ConditionType.CONTROLLER_EARTH_SIGILS_AT_LEAST: "earth_sigils",
    ConditionType.OPPONENT_EARTH_SIGILS_AT_LEAST: "earth_sigils",
    ConditionType.SOURCE_FUSION_COUNT_AT_LEAST: "fusion",
    ConditionType.SOURCE_SPELLBOOST_COUNT_AT_LEAST: "spellboost",
}

EXPR_CATEGORIES = {
    ExprType.CONTROLLER_SHADOWS: "shadows_necromancy",
    ExprType.OPPONENT_SHADOWS: "shadows_necromancy",
    ExprType.CONTROLLER_COOPERATION: "cooperation",
    ExprType.OPPONENT_COOPERATION: "cooperation",
    ExprType.CONTROLLER_OVERFLOW: "overflow",
    ExprType.OPPONENT_OVERFLOW: "overflow",
    ExprType.CONTROLLER_COMBO: "combo",
    ExprType.OPPONENT_COMBO: "combo",
    ExprType.CONTROLLER_EARTH_SIGILS: "earth_sigils",
    ExprType.OPPONENT_EARTH_SIGILS: "earth_sigils",
    ExprType.SOURCE_FUSION_DISTINCT_NAME_COUNT: "fusion",
    ExprType.SOURCE_SPELLBOOST_COUNT: "spellboost",
}

MATRIX_EVIDENCE = {
    "draw": (
        ("tests/test_zone_resource_audit.py", "test_hand_zero_eight_nine_and_overdraw_boundaries"),
        ("tests/test_filtered_draw.py", "test_draw_filtered_full_hand_overdraws_to_graveyard"),
    ),
    "add_to_hand": (
        ("tests/test_targeting_and_zones.py", "test_add_card_full_hand_discards_to_graveyard"),
    ),
    "discard": (
        ("tests/test_targeting_and_zones.py", "test_discard_moves_hand_to_graveyard"),
    ),
    "return_to_hand": (
        ("tests/test_card_origin.py", "test_return_to_hand_preserves_deck_origin"),
        ("tests/test_targeting_and_zones.py", "test_return_to_hand_full_hand_banishes"),
    ),
    "banish": (
        ("tests/test_targeting_and_zones.py", "test_banish_does_not_trigger_last_words"),
        ("tests/test_graveyard.py", "test_banish_moves_to_banished_zone"),
    ),
    "transform": (
        ("tests/test_card_origin.py", "test_transform_sets_transformed_origin"),
        ("tests/test_card_origin.py", "test_transform_does_not_increment_cooperation"),
    ),
    "return_to_deck": (
        ("tests/test_card_origin.py", "test_return_to_deck_then_draw_resets_origin_to_deck"),
        ("tests/test_targeting_and_zones.py", "test_return_to_deck"),
    ),
    "summon": (
        ("tests/test_zone_resource_audit.py", "test_board_zero_four_five_and_death_reopens_slot"),
        ("tests/test_cooperation.py", "test_failed_summon_on_full_board_does_not_increment"),
    ),
    "destroy": (
        ("tests/test_core_engine.py", "test_stabilization_moves_destroyed_units_to_graveyards"),
    ),
    "countdown_or_activate": (
        ("tests/test_zone_resource_audit.py", "test_amulet_countdown_destroy_banish_and_activate_are_distinct"),
        ("tests/test_activate.py", "test_reduce_countdown_clamps_at_zero_and_expires_source"),
    ),
    "leader_area": (
        ("tests/test_zone_resource_audit.py", "test_leader_area_shares_five_slots_and_rejects_sixth"),
        ("tests/test_emblems.py", "test_same_emblem_does_not_stack_under_legacy_allow_policy"),
        ("tests/test_faith.py", "test_same_named_faith_definitions_are_deduplicated"),
    ),
    "empty_deck": (
        ("tests/test_real_mjerrabaine_deck_batch.py", "test_empty_deck_draw_is_official_immediate_defeat_not_fatigue"),
        ("tests/test_real_mjerrabaine_deck_batch.py", "test_victory_card_result_stops_turn_end_resolution_immediately"),
    ),
    "combo": (
        ("tests/test_combo.py", "test_combo_threshold_includes_currently_played_card"),
        ("tests/test_combo.py", "test_cards_played_this_turn_increments_and_resets_at_end_turn"),
    ),
    "cooperation": (
        ("tests/test_cooperation.py", "test_cooperation_event_precedes_follower_summoned"),
        ("tests/test_cooperation.py", "test_failed_summon_on_full_board_does_not_increment"),
    ),
    "shadows_necromancy": (
        ("tests/test_necromancy.py", "test_destroyed_follower_increments_shadows_once"),
        ("tests/test_necromancy.py", "test_sufficient_shadows_spends_once_and_executes"),
    ),
    "overflow": (
        ("tests/test_overflow.py", "test_non_overflow_boundary_deals_base_damage"),
        ("tests/test_overflow.py", "test_overflow_boundary_deals_upgraded_damage"),
    ),
    "earth_sigils": (
        ("tests/test_earth_rite.py", "test_sufficient_sigils_pay_before_effect_and_zero_destroys_stack"),
        ("tests/test_earth_rite.py", "test_add_sigils_without_stack_fails_cleanly_on_full_board"),
    ),
    "spellboost": (
        ("tests/test_unit_state.py", "test_playing_spell_boosts_other_hand_spells"),
        ("tests/test_unit_state.py", "test_spell_does_not_boost_self"),
    ),
    "fusion": (
        ("tests/test_fusion.py", "test_fusion_consumes_material_without_graveyard_shadows_or_banish"),
        ("tests/test_fusion.py", "test_each_fusion_card_can_fuse_once_per_turn_and_again_next_turn"),
    ),
    "union_burst": (
        ("tests/test_union_burst.py", "test_exact_threshold_uses_turn_and_hand_evolution_bonus"),
        ("tests/test_union_burst.py", "test_successful_normal_evolution_increments_current_own_hand_only"),
    ),
    "super_skybound_art": (
        ("tests/test_union_burst.py", "test_gauge_fifteen_activates_union_then_super_skybound_art"),
        ("tests/test_union_burst.py", "test_card_entering_hand_after_evolution_starts_with_zero_bonus"),
    ),
}

BEHAVIOR_CONTRACTS = {
    "hand_capacity_0_8_9_overdraw": (
        ("tests/test_zone_resource_audit.py", "test_hand_zero_eight_nine_and_overdraw_boundaries"),
    ),
    "board_capacity_0_4_5_death_slot": (
        ("tests/test_zone_resource_audit.py", "test_board_zero_four_five_and_death_reopens_slot"),
    ),
    "zone_ownership_and_uniqueness": (
        ("tests/test_zone_resource_audit.py", "test_zone_transitions_keep_single_entity_ownership"),
        ("tests/test_engine_invariants.py", "test_duplicate_entity_across_zones_is_rejected"),
        ("tests/test_graveyard_audit.py", "test_duplicate_entity_across_hand_and_graveyard_is_rejected"),
    ),
    "empty_deck_default_and_victory_card": MATRIX_EVIDENCE["empty_deck"],
    "amulet_exit_and_activation_modes": (
        ("tests/test_zone_resource_audit.py", "test_amulet_countdown_destroy_banish_and_activate_are_distinct"),
    ),
    "shared_leader_area_capacity": (
        ("tests/test_zone_resource_audit.py", "test_leader_area_shares_five_slots_and_rejects_sixth"),
    ),
    "resource_increment_and_consumption_timing": tuple(
        reference
        for category in RESOURCE_CATEGORIES
        for reference in MATRIX_EVIDENCE[category]
    ),
    "overdraw_is_not_successful_draw": (
        ("tests/test_turn_timing_p1_corner_cases.py", "test_overdraw_does_not_trigger_desperate_shrinemouse"),
        ("tests/test_turn_timing_p1_corner_cases.py", "test_overdraw_does_not_trigger_mistbloom_emblem"),
        ("tests/test_real_hand_runtime_existing_eighth_batch.py", "test_rusty_overdraw_binds_only_successful_draws"),
    ),
    "public_zone_histograms_match_state": (
        ("tests/test_zone_resource_audit.py", "test_public_zone_histograms_match_real_zones"),
    ),
}

DEMO_TEST_EVIDENCE = {
    "conditional_demo.json": "tests/test_conditions.py",
    "decisions_demo.json": "tests/test_decisions.py",
    "emblems_advanced_demo.json": "tests/test_emblems_advanced.py",
    "emblems_demo.json": "tests/test_emblems.py",
    "graveyard_demo.json": "tests/test_graveyard.py",
    "play_modes_demo.json": "tests/test_play_modes.py",
}


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> object:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _test_reference_status(path: str, test_name: str) -> dict[str, object]:
    full_path = _repo_path(Path(path))
    exists = full_path.is_file()
    text = full_path.read_text(encoding="utf-8") if exists else ""
    test_exists = f"def {test_name}(" in text
    return {
        "path": path,
        "test_name": test_name,
        "file_exists": exists,
        "test_exists": test_exists,
        "passed": exists and test_exists,
    }


def _coverage_entry(
    coverage: Mapping[str, object],
    card_id: int,
) -> Mapping[str, object]:
    classifications = coverage.get("classifications", {})
    if not isinstance(classifications, Mapping):
        return {}
    entry = classifications.get(str(card_id), {})
    return entry if isinstance(entry, Mapping) else {}


def _test_evidence(entry: Mapping[str, object]) -> list[str]:
    for key in ("clause_audit", "rule_metadata"):
        nested = entry.get(key, {})
        if not isinstance(nested, Mapping):
            continue
        evidence = nested.get("test_evidence", [])
        if isinstance(evidence, list) and evidence:
            return sorted(str(path) for path in evidence)
    return []


def _walk(value: object) -> Iterable[object]:
    yield value
    if is_dataclass(value):
        for field in fields(value):
            yield from _walk(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _walk(item)


def _categories_for(value: object) -> set[str]:
    categories: set[str] = set()
    for item in _walk(value):
        if isinstance(item, EffectKind):
            categories.update(EFFECT_CATEGORIES.get(item, ()))
        elif isinstance(item, ConditionType):
            category = CONDITION_CATEGORIES.get(item)
            if category is not None:
                categories.add(category)
        elif isinstance(item, ExprType):
            category = EXPR_CATEGORIES.get(item)
            if category is not None:
                categories.add(category)
    return categories


def _add_source(
    sources: dict[int, list[dict[str, str]]],
    card_id: int,
    root: str,
    value: object,
    extra_categories: Iterable[str] = (),
) -> None:
    categories = _categories_for(value)
    categories.update(extra_categories)
    for category in sorted(categories):
        sources[card_id].append({"category": category, "root": root})


def _source_records(rulebook: RuleBook) -> dict[int, list[dict[str, str]]]:
    sources: dict[int, list[dict[str, str]]] = defaultdict(list)
    for (card_id, trigger), operations in sorted(
        rulebook._rules.items(),
        key=lambda item: (item[0][0], item[0][1].value),
    ):
        extra = ("countdown_or_activate",) if trigger.value == "activate" else ()
        _add_source(
            sources,
            card_id,
            f"rule:{trigger.value}",
            operations,
            extra,
        )
    for card_id, passives in sorted(rulebook._passives.items()):
        extra = (
            ("spellboost",)
            if any(passive.kind == "spellboost_cost_reduction" for passive in passives)
            else ()
        )
        _add_source(sources, card_id, "passives", passives, extra)
    for card_id, modes in sorted(rulebook._play_modes.items()):
        _add_source(sources, card_id, "play_modes", modes)
    for card_id, definition in sorted(rulebook._fusion_defs.items()):
        _add_source(sources, card_id, "fusion", definition, ("fusion",))
    for card_id, definition in sorted(rulebook._activation_defs.items()):
        _add_source(
            sources,
            card_id,
            "activation",
            definition,
            ("countdown_or_activate",),
        )
    for card_id, definition in sorted(rulebook._faith_defs.items()):
        _add_source(
            sources,
            card_id,
            f"faith:{definition.faith_id}",
            definition,
            ("leader_area",),
        )
    for card_id, definitions in sorted(rulebook._union_burst_defs.items()):
        extra = {
            definition.kind.value
            for definition in definitions
        }
        _add_source(
            sources,
            card_id,
            "union_bursts",
            definitions,
            extra,
        )
    for card_id, definitions in sorted(rulebook._listener_defs.items()):
        _add_source(sources, card_id, "listeners", definitions)
    for card_id, countdown in sorted(rulebook._countdowns.items()):
        _add_source(
            sources,
            card_id,
            f"countdown:{countdown}",
            countdown,
            ("countdown_or_activate",),
        )
    for emblem_id, definition in sorted(rulebook._emblem_defs.items()):
        _add_source(
            sources,
            definition.source_card_id,
            f"emblem:{emblem_id}",
            definition,
            ("leader_area",),
        )
    return {
        card_id: sorted(
            records,
            key=lambda row: (row["category"], row["root"]),
        )
        for card_id, records in sorted(sources.items())
        if records
    }


def _demo_rule_files(card_id: int, rules_path: Path) -> list[str]:
    needle = str(card_id)
    paths: list[str] = []
    for path in sorted(_repo_path(rules_path).rglob("*.json")):
        if needle not in path.read_text(encoding="utf-8"):
            continue
        paths.append(path.relative_to(ROOT).as_posix())
    return paths


def _inventory(
    cards: tuple[CardDefinition, ...],
    sources: Mapping[int, list[dict[str, str]]],
    closure_ids: set[int],
    coverage: Mapping[str, object],
    rules_path: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    cards_by_id = {card.card_id: card for card in cards}
    rows: list[dict[str, object]] = []
    issues: list[str] = []
    for card_id, records in sorted(sources.items()):
        card = cards_by_id.get(card_id)
        categories = sorted(
            {record["category"] for record in records},
            key=lambda category: CATEGORIES.index(category),
        )
        row_issues: list[str] = []
        if card is None:
            demo_files = _demo_rule_files(card_id, rules_path)
            synthetic_demo = bool(demo_files) and all(
                path.endswith("_demo.json") for path in demo_files
            )
            evidence = sorted(
                {
                    DEMO_TEST_EVIDENCE.get(Path(path).name, "")
                    for path in demo_files
                }
                - {""}
            )
            if not synthetic_demo:
                row_issues.append(
                    "source is absent from the database and not isolated to *_demo.json"
                )
            if not evidence:
                row_issues.append("synthetic demo source lacks test evidence")
            name = f"synthetic-demo-{card_id}"
            collectible = False
            training = False
        else:
            synthetic_demo = False
            demo_files = []
            entry = _coverage_entry(coverage, card_id)
            evidence = _test_evidence(entry)
            accepted = (
                {"covered_exact"}
                if card.is_collectible
                else {"token_or_non_collectible"}
            )
            if entry.get("coverage") not in accepted:
                row_issues.append(
                    "source lacks required collectible/generated coverage"
                )
            if not evidence:
                row_issues.append("source lacks permanent test evidence")
            name = card.name
            collectible = card.is_collectible
            training = card_id in closure_ids
        for path in evidence:
            if not _repo_path(Path(path)).is_file():
                row_issues.append(f"missing test evidence file: {path}")
        if row_issues:
            issues.extend(f"card {card_id}: {issue}" for issue in row_issues)
        rows.append(
            {
                "card_id": card_id,
                "name": name,
                "collectible": collectible,
                "training_closure": training,
                "synthetic_demo": synthetic_demo,
                "demo_rule_files": demo_files,
                "categories": categories,
                "record_count": len(records),
                "roots": sorted({record["root"] for record in records}),
                "test_evidence": evidence,
                "issues": row_issues,
                "passed": not row_issues,
            }
        )
    return rows, sorted(set(issues))


def _matrix(inventory: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category in CATEGORIES:
        production = [
            row
            for row in inventory
            if not row["synthetic_demo"] and category in row["categories"]
        ]
        demo = [
            row
            for row in inventory
            if row["synthetic_demo"] and category in row["categories"]
        ]
        evidence = [
            _test_reference_status(path, test_name)
            for path, test_name in MATRIX_EVIDENCE[category]
        ]
        rows.append(
            {
                "category": category,
                "source_card_count": len(production),
                "collectible_source_count": sum(
                    bool(row["collectible"]) for row in production
                ),
                "generated_source_count": sum(
                    not bool(row["collectible"]) for row in production
                ),
                "training_source_count": sum(
                    bool(row["training_closure"]) for row in production
                ),
                "synthetic_demo_source_count": len(demo),
                "test_evidence": evidence,
                "passed": bool(production)
                and all(bool(row["passed"]) for row in production)
                and all(bool(item["passed"]) for item in evidence),
            }
        )
    return rows


def _behavior_contracts() -> list[dict[str, object]]:
    return [
        {
            "contract_id": contract_id,
            "test_evidence": [
                _test_reference_status(path, test_name)
                for path, test_name in references
            ],
            "passed": all(
                _test_reference_status(path, test_name)["passed"]
                for path, test_name in references
            ),
        }
        for contract_id, references in BEHAVIOR_CONTRACTS.items()
    ]


def build_report(
    *,
    database_path: Path = DEFAULT_DATABASE,
    rules_path: Path = DEFAULT_RULES,
    closure_path: Path = DEFAULT_CLOSURE,
    coverage_path: Path = DEFAULT_COVERAGE,
    evidence_path: Path = DEFAULT_EVIDENCE,
) -> dict[str, object]:
    repository = CardRepository(_repo_path(database_path))
    cards = tuple(repository.all_cards())
    rulebook = RuleBook.from_directory(_repo_path(rules_path))
    closure = _load_json(closure_path)
    coverage = _load_json(coverage_path)
    external_evidence = _load_json(evidence_path)
    if not isinstance(closure, Mapping):
        raise ValueError("closure report must be an object")
    if not isinstance(coverage, Mapping):
        raise ValueError("coverage report must be an object")
    if not isinstance(external_evidence, Mapping):
        raise ValueError("external evidence must be an object")
    closure_rows = closure.get("cards", [])
    if not isinstance(closure_rows, list):
        raise ValueError("closure report cards must be a list")
    closure_ids = {
        int(row["card_id"])
        for row in closure_rows
        if isinstance(row, Mapping) and "card_id" in row
    }
    sources = _source_records(rulebook)
    inventory, inventory_issues = _inventory(
        cards,
        sources,
        closure_ids,
        coverage,
        rules_path,
    )
    matrix = _matrix(inventory)
    contracts = _behavior_contracts()
    evidence_rows = external_evidence.get("sources", [])
    evidence_ok = (
        isinstance(evidence_rows, list)
        and len(evidence_rows) >= 3
        and all(
            isinstance(row, Mapping)
            and row.get("authority")
            and row.get("url")
            and row.get("conclusion")
            for row in evidence_rows
        )
    )
    failures = list(inventory_issues)
    failures.extend(
        f"matrix {row['category']} failed"
        for row in matrix
        if not row["passed"]
    )
    failures.extend(
        f"contract {row['contract_id']} failed"
        for row in contracts
        if not row["passed"]
    )
    if not evidence_ok:
        failures.append("official evidence is incomplete")
    production = [row for row in inventory if not row["synthetic_demo"]]
    return {
        "schema_version": 1,
        "report_kind": "swb_zone_resource_audit",
        "inputs": {
            "database": {
                "path": database_path.as_posix(),
                "sha256": _sha256(database_path),
            },
            "rules": {
                "path": rules_path.as_posix(),
                "json_file_count": len(
                    tuple(_repo_path(rules_path).rglob("*.json"))
                ),
            },
            "closure": {
                "path": closure_path.as_posix(),
                "sha256": _sha256(closure_path),
            },
            "coverage": {
                "path": coverage_path.as_posix(),
                "sha256": _sha256(coverage_path),
            },
            "external_evidence": {
                "path": evidence_path.as_posix(),
                "sha256": _sha256(evidence_path),
            },
        },
        "scope": {
            "card_count": len(cards),
            "collectible_card_count": sum(card.is_collectible for card in cards),
            "generated_card_count": sum(not card.is_collectible for card in cards),
            "training_closure_card_count": len(closure_ids),
            "leader_area_limit": 5,
            "hand_limit": 9,
            "board_limit": 5,
        },
        "summary": {
            "production_source_card_count": len(production),
            "collectible_source_card_count": sum(
                bool(row["collectible"]) for row in production
            ),
            "generated_source_card_count": sum(
                not bool(row["collectible"]) for row in production
            ),
            "training_source_card_count": sum(
                bool(row["training_closure"]) for row in production
            ),
            "synthetic_demo_source_count": sum(
                bool(row["synthetic_demo"]) for row in inventory
            ),
            "category_count": len(matrix),
            "behavior_contract_count": len(contracts),
            "official_evidence_count": len(evidence_rows),
            "inventory_issue_count": len(inventory_issues),
            "failure_count": len(failures),
            "failures": failures,
            "passed": not failures,
        },
        "official_evidence": evidence_rows,
        "category_matrix": matrix,
        "behavior_contracts": contracts,
        "inventory": inventory,
    }


def render_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def render_markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    scope = report["scope"]
    lines = [
        "# Zone, Capacity, and Resource Audit",
        "",
        "This report inventories structured zone/resource sources and preserves "
        "the direct executable contracts used by checklist section 1.8.",
        "",
        "## Summary",
        "",
        f"- Cards: {scope['card_count']} "
        f"({scope['collectible_card_count']} collectible, "
        f"{scope['generated_card_count']} generated)",
        f"- Training closure: {scope['training_closure_card_count']}",
        f"- Production source cards: {summary['production_source_card_count']}",
        f"- Synthetic demo sources: {summary['synthetic_demo_source_count']}",
        f"- Behavioral contracts: {summary['behavior_contract_count']}",
        f"- Failures: {summary['failure_count']}",
        f"- Result: {'PASS' if summary['passed'] else 'FAIL'}",
        "",
        "## Official evidence",
        "",
        "| Evidence | Authority | Conclusion |",
        "|---|---|---|",
    ]
    for row in report["official_evidence"]:
        lines.append(
            f"| [{row['evidence_id']}]({row['url']}) | "
            f"{row['authority']} | {row['summary']} |"
        )
    lines.extend(
        [
            "",
            "## Category matrix",
            "",
            "| Category | Sources | Collectible | Generated | Training | Demo | Result |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in report["category_matrix"]:
        lines.append(
            f"| {row['category']} | {row['source_card_count']} | "
            f"{row['collectible_source_count']} | "
            f"{row['generated_source_count']} | "
            f"{row['training_source_count']} | "
            f"{row['synthetic_demo_source_count']} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Behavioral contracts",
            "",
            "| Contract | Evidence tests | Result |",
            "|---|---:|:---:|",
        ]
    )
    for row in report["behavior_contracts"]:
        lines.append(
            f"| {row['contract_id']} | {len(row['test_evidence'])} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Source inventory",
            "",
            "| Card | Categories | Records | Training | Result |",
            "|---|---|---:|:---:|:---:|",
        ]
    )
    for row in report["inventory"]:
        lines.append(
            f"| {row['card_id']} {row['name']} | "
            f"{', '.join(row['categories'])} | {row['record_count']} | "
            f"{'yes' if row['training_closure'] else 'no'} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["failures"])
    return "\n".join(lines) + "\n"


def write_report(
    report: Mapping[str, object],
    output_path: Path,
    markdown_path: Path,
) -> None:
    output = _repo_path(output_path)
    markdown = _repo_path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_json(report), encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report(
        database_path=args.database,
        rules_path=args.rules,
        closure_path=args.closure,
        coverage_path=args.coverage,
        evidence_path=args.evidence,
    )
    write_report(report, args.output, args.markdown)
    summary = report["summary"]
    print(
        f"cards={report['scope']['card_count']} "
        f"sources={summary['production_source_card_count']} "
        f"training_sources={summary['training_source_card_count']} "
        f"categories={summary['category_count']} "
        f"contracts={summary['behavior_contract_count']} "
        f"failures={summary['failure_count']} "
        f"passed={summary['passed']}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
