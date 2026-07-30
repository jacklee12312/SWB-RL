"""Audit trigger sources, timing priority, batching, and loop protection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.events import EventType
from swb.engine.resolution import MAX_RESOLUTION_STEPS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_RULES = Path("data/rules")
DEFAULT_CLOSURE = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)
DEFAULT_COVERAGE = Path("data/reports/rule_coverage.json")
DEFAULT_EVIDENCE = Path("data/audits/timing_priority_evidence.json")
DEFAULT_OUTPUT = Path(
    "data/reports/card_bug_audit/trigger_timing_audit.json"
)
DEFAULT_MARKDOWN = Path(
    "data/reports/card_bug_audit/trigger_timing_audit.md"
)

REQUIRED_TRIGGER_CATEGORIES = (
    "turn_start",
    "turn_end",
    "attack",
    "clash",
    "damage_survived",
    "entry",
    "evolve",
    "super_evolve",
    "last_words",
    "countdown",
    "emblem",
    "faith",
)
EXTRA_TRIGGER_CATEGORIES = (
    "other_event_listener",
    "other_card_trigger",
)
TRIGGER_CATEGORIES = REQUIRED_TRIGGER_CATEGORIES + EXTRA_TRIGGER_CATEGORIES

DEMO_TEST_EVIDENCE = {
    "conditional_demo.json": "tests/test_conditions.py",
    "decisions_demo.json": "tests/test_decisions.py",
    "emblems_advanced_demo.json": "tests/test_emblems_advanced.py",
    "emblems_demo.json": "tests/test_emblems.py",
    "graveyard_demo.json": "tests/test_graveyard.py",
    "play_modes_demo.json": "tests/test_play_modes.py",
}

MATRIX_EVIDENCE = {
    "turn_start": (
        (
            "tests/test_triggers.py",
            "test_turn_start_trigger",
        ),
        (
            "tests/test_turn_timing_corner_cases.py",
            "test_turn_start_invocation_precedes_countdown_last_words",
        ),
        (
            "tests/test_emblems.py",
            "test_turn_start_emblem_fires_on_controller_turn",
        ),
    ),
    "turn_end": (
        (
            "tests/test_triggers.py",
            "test_turn_end_trigger",
        ),
        (
            "tests/test_turn_timing_corner_cases.py",
            "test_marwynn_crest_precedes_dark_dimension",
        ),
        (
            "tests/test_faith.py",
            "test_lyanthoth_turn_end_pays_and_generates_depths_with_token_origin",
        ),
    ),
    "attack": (
        (
            "tests/test_triggers.py",
            "test_attack_trigger_preserves_combat",
        ),
        (
            "tests/test_real_attack_history_emblem_countdown_token_batch.py",
            "test_attack_still_counts_when_on_attack_effect_removes_attacker",
        ),
    ),
    "clash": (
        (
            "tests/test_triggers.py",
            "test_clash_triggers_both_sides",
        ),
        (
            "tests/test_triggers.py",
            "test_bilateral_choice_clash_both_sides",
        ),
    ),
    "damage_survived": (
        (
            "tests/test_real_generated_damage_countdown_batch.py",
            "test_galmieux_self_survival_triggers_follower_and_crest_once_each",
        ),
        (
            "tests/test_real_crest_entry_source_health_fourteenth_batch.py",
            "test_angela_repeats_sequential_damage_and_self_listener",
        ),
    ),
    "entry": (
        (
            "tests/test_emblems.py",
            "test_follower_summoned_emblem_fires",
        ),
        (
            "tests/test_card_listeners.py",
            "test_board_hand_and_leader_area_sources_all_activate",
        ),
        (
            "tests/test_triggers.py",
            "test_fanfare_still_works",
        ),
    ),
    "evolve": (
        (
            "tests/test_triggers.py",
            "test_evolve_trigger_fires_after_stat_change",
        ),
        (
            "tests/test_faith.py",
            "test_faith_increments_before_evolve_trigger_pending_choice",
        ),
    ),
    "super_evolve": (
        (
            "tests/test_triggers.py",
            "test_super_evolve_trigger_fires_after_stat_change",
        ),
        (
            "tests/test_faith.py",
            "test_super_evolution_counts_as_follower_evolution",
        ),
    ),
    "last_words": (
        (
            "tests/test_last_words.py",
            "test_mixed_follower_amulet_batch_events_share_batch_id_before_lw",
        ),
        (
            "tests/test_last_words.py",
            "test_same_batch_last_words_complete_before_new_death_batch",
        ),
    ),
    "countdown": (
        (
            "tests/test_last_words.py",
            "test_countdown_amulet_last_words_triggers",
        ),
        (
            "tests/test_emblems.py",
            "test_countdown_emblem_expires_no_trigger",
        ),
    ),
    "emblem": (
        (
            "tests/test_emblems.py",
            "test_emblems_fire_in_creation_order",
        ),
        (
            "tests/test_emblems_advanced.py",
            "test_new_emblem_not_in_current_batch",
        ),
    ),
    "faith": (
        (
            "tests/test_faith.py",
            "test_simultaneous_amulet_faith_progression_is_active_player_first",
        ),
        (
            "tests/test_faith.py",
            "test_pending_choice_pauses_after_progress_and_resumes_event",
        ),
    ),
    "other_event_listener": (
        (
            "tests/test_card_listeners.py",
            "test_event_source_is_revalidated_after_it_leaves",
        ),
    ),
    "other_card_trigger": (
        (
            "tests/test_activate.py",
            "test_source_can_destroy_itself_before_targeted_effect_resolves",
        ),
    ),
}

CHECKLIST_CONTRACTS = (
    {
        "contract_id": "trigger_matrix",
        "conclusion": (
            "Every required timing family has production sources, full-pool "
            "inventory, and direct executable test evidence."
        ),
        "tests": tuple(
            evidence
            for category in REQUIRED_TRIGGER_CATEGORIES
            for evidence in MATRIX_EVIDENCE[category]
        ),
        "external_evidence_ids": (),
    },
    {
        "contract_id": "official_turn_boundary_priority",
        "conclusion": (
            "Official Sandalphon Q&A order is retained for crest, countdown, "
            "Invocation, Last Words, invoked effects, and ordinary draw."
        ),
        "tests": (
            (
                "tests/test_turn_timing_corner_cases.py",
                "test_turn_start_invocation_precedes_countdown_last_words",
            ),
        ),
        "external_evidence_ids": ("SWB-TIMING-OFFICIAL-001",),
    },
    {
        "contract_id": "marwynn_crest_before_board",
        "conclusion": (
            "The real Marwynn crest resolves before field-card turn-end "
            "abilities."
        ),
        "tests": (
            (
                "tests/test_turn_timing_corner_cases.py",
                "test_marwynn_crest_precedes_dark_dimension",
            ),
        ),
        "external_evidence_ids": ("SWB-TIMING-CARD-003",),
    },
    {
        "contract_id": "simultaneous_death_batch_before_last_words",
        "conclusion": (
            "All simultaneous follower and amulet deaths are collected and "
            "leave play before the first Last Words starts."
        ),
        "tests": (
            (
                "tests/test_last_words.py",
                "test_mixed_follower_amulet_batch_events_share_batch_id_before_lw",
            ),
            (
                "tests/test_last_words.py",
                "test_cross_player_mixed_batch_metadata_counts_followers_and_amulets",
            ),
        ),
        "external_evidence_ids": (),
    },
    {
        "contract_id": "new_sources_wait_for_next_batch",
        "conclusion": (
            "A newly gained crest, summoned follower, or newly granted "
            "turn-end ability does not join the current trigger batch."
        ),
        "tests": (
            (
                "tests/test_emblems_advanced.py",
                "test_new_emblem_not_in_current_batch",
            ),
            (
                "tests/test_turn_timing_p1_corner_cases.py",
                "test_follower_summoned_mid_turn_end_waits_for_next_batch",
            ),
            (
                "tests/test_turn_timing_p1_corner_cases.py",
                "test_turn_end_ability_granted_mid_batch_waits_for_next_batch",
            ),
        ),
        "external_evidence_ids": (),
    },
    {
        "contract_id": "conditions_snapshot_at_batch_start",
        "conclusion": (
            "Turn-boundary eligibility and operation branches are frozen at "
            "the start of the timing batch."
        ),
        "tests": (
            (
                "tests/test_turn_timing_corner_cases.py",
                "test_turn_end_conditions_are_snapshotted",
            ),
            (
                "tests/test_turn_timing_corner_cases.py",
                "test_newly_eligible_turn_end_effect_does_not_join_batch",
            ),
        ),
        "external_evidence_ids": (),
    },
    {
        "contract_id": "queued_source_leaves_play",
        "conclusion": (
            "An already queued timing ability continues from its source "
            "snapshot after the source leaves play."
        ),
        "tests": (
            (
                "tests/test_trigger_timing_audit.py",
                "test_queued_turn_end_source_continues_after_leaving_play",
            ),
            (
                "tests/test_last_words.py",
                "test_last_words_source_keyword_and_attack_expression_use_snapshot",
            ),
        ),
        "external_evidence_ids": (),
    },
    {
        "contract_id": "pending_choice_preserves_queue_order",
        "conclusion": (
            "Attack, clash, turn-end, event, crest, faith, and Last Words "
            "queues pause and resume without skipping or reordering."
        ),
        "tests": (
            (
                "tests/test_triggers.py",
                "test_bilateral_choice_clash_both_sides",
            ),
            (
                "tests/test_emblems.py",
                "test_choice_event_emblem_resumes_remaining_emblems",
            ),
            (
                "tests/test_faith.py",
                "test_pending_choice_pauses_after_progress_and_resumes_event",
            ),
            (
                "tests/test_last_words.py",
                "test_choice_lw_then_draw_lw_same_batch_ordered",
            ),
        ),
        "external_evidence_ids": (),
    },
    {
        "contract_id": "terminal_result_stops_remaining_queue",
        "conclusion": (
            "Once lethal determines the result, remaining damage, healing, "
            "Last Words, and crest records do not change the result."
        ),
        "tests": (
            (
                "tests/test_turn_timing_p1_corner_cases.py",
                "test_super_evolution_lethal_stops_mimi_counter_damage",
            ),
            (
                "tests/test_turn_timing_p1_corner_cases.py",
                "test_super_evolution_lethal_stops_coco_last_words_heal",
            ),
            (
                "tests/test_emblems_advanced.py",
                "test_game_end_stops_remaining_emblem_batch",
            ),
        ),
        "external_evidence_ids": (),
    },
    {
        "contract_id": "all_leader_damage_winner",
        "conclusion": (
            "Balt's official all-leader lethal case awards the opponent and "
            "does not continue to damage the opponent leader."
        ),
        "tests": (
            (
                "tests/test_turn_timing_p1_corner_cases.py",
                "test_balt_lethal_to_both_leaders_awards_opponent",
            ),
        ),
        "external_evidence_ids": ("SWB-TIMING-OFFICIAL-002",),
    },
    {
        "contract_id": "recursive_trigger_step_limit",
        "conclusion": (
            "Recursive death/crest loops raise ResolutionLoopError at the "
            "configured step limit with deterministic JSON diagnostics."
        ),
        "tests": (
            (
                "tests/test_last_words.py",
                "test_loop_detection_throws_resolution_loop_error",
            ),
            (
                "tests/test_last_words.py",
                "test_loop_error_includes_structured_diagnostics",
            ),
            (
                "tests/test_last_words.py",
                "test_loop_diagnostics_are_seed_deterministic",
            ),
            (
                "tests/test_last_words.py",
                "test_death_batch_end_emblem_loop_diagnostics_identify_trigger_batches",
            ),
        ),
        "external_evidence_ids": (),
    },
)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> object:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


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


def _rule_trigger_categories(trigger: Trigger) -> tuple[str, ...]:
    return {
        Trigger.PLAY: ("entry",),
        Trigger.FANFARE: ("entry",),
        Trigger.LAST_WORDS: ("last_words",),
        Trigger.EVOLVE: ("evolve",),
        Trigger.SELF_EVOLVED: ("evolve",),
        Trigger.SUPER_EVOLVE: ("super_evolve",),
        Trigger.SELF_SUPER_EVOLVED: ("super_evolve",),
        Trigger.ATTACK: ("attack",),
        Trigger.CLASH: ("clash",),
        Trigger.TURN_START: ("turn_start",),
        Trigger.TURN_END: ("turn_end",),
        Trigger.COUNTDOWN_EXPIRED: ("countdown", "last_words"),
        Trigger.INVOKE: ("turn_start", "entry"),
    }.get(trigger, ("other_card_trigger",))


def _event_categories(event: EventType) -> tuple[str, ...]:
    return {
        EventType.TURN_STARTED: ("turn_start",),
        EventType.TURN_ENDED: ("turn_end",),
        EventType.ATTACK_DECLARED: ("attack",),
        EventType.COMBAT_STARTED: ("clash",),
        EventType.FOLLOWER_DAMAGED_SURVIVED: ("damage_survived",),
        EventType.FOLLOWER_SUMMONED: ("entry",),
        EventType.CARD_PLAYED: ("entry",),
        EventType.FOLLOWER_EVOLVED: ("evolve",),
        EventType.FOLLOWER_SUPER_EVOLVED: ("super_evolve",),
        EventType.FOLLOWER_DESTROYED: ("last_words",),
        EventType.AMULET_DESTROYED: ("last_words", "faith"),
        EventType.DEATH_BATCH_END: ("last_words",),
    }.get(event, ("other_event_listener",))


def _emblem_categories(trigger: str) -> tuple[str, ...]:
    mapped = {
        "turn_start": ("turn_start",),
        "turn_end": ("turn_end",),
        "attack_declared": ("attack",),
        "follower_summoned": ("entry",),
        "card_played": ("entry",),
        "follower_evolved": ("evolve",),
        "follower_destroyed": ("last_words",),
        "amulet_destroyed": ("last_words",),
        "death_batch_end": ("last_words",),
    }.get(trigger, ("other_event_listener",))
    return ("emblem",) + mapped


def _faith_categories(trigger: str) -> tuple[str, ...]:
    mapped = {
        "follower_evolved": ("evolve",),
        "follower_summoned": ("entry",),
        "amulet_destroyed": ("last_words",),
    }.get(trigger, ("other_event_listener",))
    return ("faith",) + mapped


def _add_source(
    sources: dict[int, list[dict[str, object]]],
    card_id: int,
    root: str,
    categories: Iterable[str],
) -> None:
    for category in categories:
        sources[card_id].append(
            {
                "category": category,
                "root": root,
            }
        )


def _source_records(rulebook: RuleBook) -> dict[int, list[dict[str, object]]]:
    sources: dict[int, list[dict[str, object]]] = defaultdict(list)
    for (card_id, trigger), operations in sorted(
        rulebook._rules.items(),
        key=lambda item: (item[0][0], item[0][1].value),
    ):
        _add_source(
            sources,
            card_id,
            f"rule:{trigger.value}",
            _rule_trigger_categories(trigger),
        )
        if not operations:
            sources[card_id].append(
                {
                    "category": "other_card_trigger",
                    "root": f"empty_rule:{trigger.value}",
                }
            )
    for card_id, definitions in sorted(rulebook._listener_defs.items()):
        for index, definition in enumerate(definitions):
            _add_source(
                sources,
                card_id,
                f"listener:{definition.event.value}:{index}",
                _event_categories(definition.event),
            )
    for emblem_id, definition in sorted(rulebook._emblem_defs.items()):
        for index, trigger in enumerate(definition.triggers):
            _add_source(
                sources,
                definition.source_card_id,
                f"emblem:{emblem_id}:{trigger.trigger}:{index}",
                _emblem_categories(trigger.trigger),
            )
        if definition.countdown is not None:
            _add_source(
                sources,
                definition.source_card_id,
                f"emblem:{emblem_id}:countdown",
                ("emblem", "countdown"),
            )
        if definition.on_expire:
            _add_source(
                sources,
                definition.source_card_id,
                f"emblem:{emblem_id}:on_expire",
                ("emblem", "countdown"),
            )
    for card_id, definition in sorted(rulebook._faith_defs.items()):
        for index, trigger in enumerate(definition.triggers):
            _add_source(
                sources,
                card_id,
                f"faith:{definition.faith_id}:{trigger.trigger.value}:{index}",
                _faith_categories(trigger.trigger.value),
            )
        _add_source(
            sources,
            card_id,
            f"faith:{definition.faith_id}",
            ("faith",),
        )
    for card_id, countdown in sorted(rulebook._countdowns.items()):
        _add_source(
            sources,
            card_id,
            f"countdown:{countdown}",
            ("countdown",),
        )
    for card_id, definition in sorted(rulebook._invocation_defs.items()):
        _add_source(
            sources,
            card_id,
            f"invocation:{definition.trigger.value}",
            ("turn_start", "entry"),
        )
    return {
        card_id: sorted(
            records,
            key=lambda row: (str(row["category"]), str(row["root"])),
        )
        for card_id, records in sorted(sources.items())
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
    sources: Mapping[int, list[dict[str, object]]],
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
            {str(record["category"]) for record in records},
            key=lambda value: (
                TRIGGER_CATEGORIES.index(value)
                if value in TRIGGER_CATEGORIES
                else len(TRIGGER_CATEGORIES),
                value,
            ),
        )
        category_record_counts = {
            category: sum(
                str(record["category"]) == category for record in records
            )
            for category in categories
        }
        roots = sorted({str(record["root"]) for record in records})
        row_issues: list[str] = []
        if card is None:
            demo_files = _demo_rule_files(card_id, rules_path)
            demo = bool(demo_files) and all(
                path.endswith("_demo.json") for path in demo_files
            )
            evidence = sorted(
                {
                    DEMO_TEST_EVIDENCE.get(Path(path).name, "")
                    for path in demo_files
                }
                - {""}
            )
            if not demo:
                row_issues.append(
                    "source is absent from the database and is not isolated "
                    "to explicitly named *_demo.json files"
                )
            if not evidence:
                row_issues.append("synthetic demo source lacks test evidence")
            for path in evidence:
                if not _repo_path(Path(path)).is_file():
                    row_issues.append(f"missing demo test evidence: {path}")
            row = {
                "card_id": card_id,
                "name": f"synthetic-demo-{card_id}",
                "collectible": False,
                "training_closure": False,
                "synthetic_demo": demo,
                "demo_rule_files": demo_files,
                "categories": categories,
                "category_record_counts": category_record_counts,
                "roots": roots,
                "record_count": len(records),
                "test_evidence": evidence,
                "issues": row_issues,
                "passed": not row_issues,
            }
        else:
            entry = _coverage_entry(coverage, card_id)
            evidence = _test_evidence(entry)
            accepted = (
                {"covered_exact"}
                if card.is_collectible
                else {"token_or_non_collectible"}
            )
            if entry.get("coverage") not in accepted:
                row_issues.append(
                    "source lacks the required collectible/generated coverage"
                )
            if not evidence:
                row_issues.append("source lacks permanent test evidence")
            for path in evidence:
                if not _repo_path(Path(path)).is_file():
                    row_issues.append(f"missing test evidence file: {path}")
            row = {
                "card_id": card_id,
                "name": card.name,
                "collectible": card.is_collectible,
                "training_closure": card_id in closure_ids,
                "synthetic_demo": False,
                "demo_rule_files": [],
                "categories": categories,
                "category_record_counts": category_record_counts,
                "roots": roots,
                "record_count": len(records),
                "test_evidence": evidence,
                "issues": row_issues,
                "passed": not row_issues,
            }
        if row_issues:
            issues.extend(
                f"card {card_id}: {message}" for message in row_issues
            )
        rows.append(row)
    return rows, sorted(set(issues))


def _matrix(
    inventory: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category in TRIGGER_CATEGORIES:
        production = [
            row
            for row in inventory
            if not row["synthetic_demo"] and category in row["categories"]
        ]
        demos = [
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
                "source_record_count": sum(
                    int(row["category_record_counts"].get(category, 0))
                    for row in production
                ),
                "synthetic_demo_source_count": len(demos),
                "test_evidence": evidence,
                "passed": bool(production)
                and all(bool(item["passed"]) for item in evidence),
            }
        )
    return rows


def _external_evidence(
    evidence_path: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    raw = _load_json(evidence_path)
    issues: list[str] = []
    if not isinstance(raw, Mapping):
        return [], ["timing priority evidence is not a JSON object"]
    entries = raw.get("entries", [])
    if not isinstance(entries, list):
        return [], ["timing priority evidence entries must be a list"]
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            issues.append(f"evidence entry {index} is not an object")
            continue
        row = dict(entry)
        evidence_id = row.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            issues.append(f"evidence entry {index} lacks evidence_id")
            continue
        if evidence_id in seen:
            issues.append(f"duplicate evidence_id: {evidence_id}")
        seen.add(evidence_id)
        for field in (
            "authority",
            "url",
            "accessed_on",
            "conclusion",
            "summary",
        ):
            if not isinstance(row.get(field), str) or not row[field]:
                issues.append(f"{evidence_id} lacks {field}")
        if not str(row.get("url", "")).startswith("https://shadowverse-wb.com/"):
            issues.append(f"{evidence_id} is not an official SWB URL")
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["evidence_id"])), sorted(
        set(issues)
    )


def _contracts(
    evidence_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    evidence_ids = {
        str(row["evidence_id"]) for row in evidence_rows
    }
    rows: list[dict[str, object]] = []
    for definition in CHECKLIST_CONTRACTS:
        test_rows = [
            _test_reference_status(path, test_name)
            for path, test_name in definition["tests"]
        ]
        required_external = list(definition["external_evidence_ids"])
        missing_external = sorted(
            set(required_external) - evidence_ids
        )
        rows.append(
            {
                "contract_id": definition["contract_id"],
                "conclusion": definition["conclusion"],
                "test_evidence": test_rows,
                "external_evidence_ids": required_external,
                "missing_external_evidence_ids": missing_external,
                "passed": (
                    bool(test_rows)
                    and all(bool(row["passed"]) for row in test_rows)
                    and not missing_external
                ),
            }
        )
    return rows


def build_report(
    *,
    database_path: Path = DEFAULT_DATABASE,
    rules_path: Path = DEFAULT_RULES,
    closure_path: Path = DEFAULT_CLOSURE,
    coverage_path: Path = DEFAULT_COVERAGE,
    evidence_path: Path = DEFAULT_EVIDENCE,
) -> dict[str, object]:
    database = _repo_path(database_path)
    rules = _repo_path(rules_path)
    closure_file = _repo_path(closure_path)
    coverage_file = _repo_path(coverage_path)
    evidence_file = _repo_path(evidence_path)
    cards = tuple(CardRepository(database).all_cards())
    rulebook = RuleBook.from_directory(rules)
    closure = _load_json(closure_path)
    coverage = _load_json(coverage_path)
    if not isinstance(closure, Mapping) or not isinstance(coverage, Mapping):
        raise ValueError("closure and coverage reports must be JSON objects")
    closure_ids = {
        int(row["card_id"])
        for row in closure.get("cards", [])
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
    evidence, evidence_issues = _external_evidence(evidence_path)
    contracts = _contracts(evidence)
    unsupported = {
        "mechanism": "death_batch_start_emblem_trigger",
        "status": "explicitly_unsupported_not_applicable",
        "production_source_count": sum(
            1
            for row in inventory
            if not row["synthetic_demo"]
            and any(
                "death_batch_start" in str(root)
                for root in row["roots"]
            )
        ),
        "test_evidence": _test_reference_status(
            "tests/test_emblems.py",
            "test_death_batch_start_emblem_trigger_remains_unsupported",
        ),
    }
    unsupported["passed"] = (
        unsupported["production_source_count"] == 0
        and unsupported["test_evidence"]["passed"]
    )

    failures = list(inventory_issues) + list(evidence_issues)
    failures.extend(
        f"trigger matrix failed: {row['category']}"
        for row in matrix
        if not row["passed"]
    )
    failures.extend(
        f"behavior contract failed: {row['contract_id']}"
        for row in contracts
        if not row["passed"]
    )
    if not unsupported["passed"]:
        failures.append(
            "death_batch_start unsupported boundary is not safely isolated"
        )
    production = [
        row for row in inventory if not row["synthetic_demo"]
    ]
    demos = [row for row in inventory if row["synthetic_demo"]]
    return {
        "schema_version": 1,
        "report_kind": "swb_trigger_timing_audit",
        "inputs": {
            "database": {
                "path": database.relative_to(ROOT).as_posix(),
                "sha256": _sha256(database),
            },
            "rules": {
                "path": rules.relative_to(ROOT).as_posix(),
                "json_file_count": len(tuple(rules.rglob("*.json"))),
            },
            "closure": {
                "path": closure_file.relative_to(ROOT).as_posix(),
                "sha256": _sha256(closure_file),
            },
            "coverage": {
                "path": coverage_file.relative_to(ROOT).as_posix(),
                "sha256": _sha256(coverage_file),
            },
            "external_evidence": {
                "path": evidence_file.relative_to(ROOT).as_posix(),
                "sha256": _sha256(evidence_file),
            },
        },
        "scope": {
            "card_count": len(cards),
            "collectible_card_count": sum(
                card.is_collectible for card in cards
            ),
            "generated_card_count": sum(
                not card.is_collectible for card in cards
            ),
            "training_closure_card_count": len(closure_ids),
            "rule_trigger_definition_count": len(rulebook._rules),
            "listener_definition_count": sum(
                len(definitions)
                for definitions in rulebook._listener_defs.values()
            ),
            "emblem_definition_count": len(rulebook._emblem_defs),
            "emblem_trigger_definition_count": sum(
                len(definition.triggers)
                for definition in rulebook._emblem_defs.values()
            ),
            "faith_definition_count": len(rulebook._faith_defs),
            "faith_trigger_definition_count": sum(
                len(definition.triggers)
                for definition in rulebook._faith_defs.values()
            ),
            "max_resolution_steps": MAX_RESOLUTION_STEPS,
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
            "synthetic_demo_source_count": len(demos),
            "required_trigger_category_count": len(
                REQUIRED_TRIGGER_CATEGORIES
            ),
            "behavior_contract_count": len(contracts),
            "inventory_issue_count": len(inventory_issues),
            "evidence_issue_count": len(evidence_issues),
            "failure_count": len(failures),
            "failures": sorted(set(failures)),
            "passed": not failures,
        },
        "external_evidence": evidence,
        "trigger_matrix": matrix,
        "behavior_contracts": contracts,
        "explicit_unsupported": [unsupported],
        "inventory": inventory,
    }


def render_json(report: Mapping[str, object]) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    ) + "\n"


def render_markdown(report: Mapping[str, object]) -> str:
    scope = report["scope"]
    summary = report["summary"]
    lines = [
        "# Trigger timing, priority, and batch audit",
        "",
        f"- Result: **{'PASS' if summary['passed'] else 'FAIL'}**; "
        f"{summary['failure_count']} failures.",
        f"- Snapshot: {scope['card_count']} cards "
        f"({scope['collectible_card_count']} collectible / "
        f"{scope['generated_card_count']} generated).",
        f"- Trigger sources: {summary['production_source_card_count']} cards; "
        f"{summary['training_source_card_count']} in the training closure; "
        f"{summary['synthetic_demo_source_count']} isolated demo sources.",
        f"- Resolution loop guard: {scope['max_resolution_steps']} steps.",
        "",
        "## External timing evidence",
        "",
        "| Evidence | Authority | Card | Accessed | Conclusion |",
        "|---|---|---:|---|---|",
    ]
    for row in report["external_evidence"]:
        conclusion = str(row["summary"]).replace("|", "\\|")
        lines.append(
            f"| {row['evidence_id']} | {row['authority']} | "
            f"{row.get('card_id', '')} | {row['accessed_on']} | "
            f"{conclusion} |"
        )
    lines.extend(
        [
            "",
            "## Trigger matrix",
            "",
            "| Category | Sources | Collectible | Generated | Training | "
            "Records | Demo | Result |",
            "|---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in report["trigger_matrix"]:
        lines.append(
            f"| {row['category']} | {row['source_card_count']} | "
            f"{row['collectible_source_count']} | "
            f"{row['generated_source_count']} | "
            f"{row['training_source_count']} | "
            f"{row['source_record_count']} | "
            f"{row['synthetic_demo_source_count']} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Behavioral contracts",
            "",
            "| Contract | Evidence tests | External evidence | Result |",
            "|---|---:|---:|:---:|",
        ]
    )
    for row in report["behavior_contracts"]:
        lines.append(
            f"| {row['contract_id']} | {len(row['test_evidence'])} | "
            f"{len(row['external_evidence_ids'])} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Explicit unsupported boundaries",
            "",
            "| Mechanism | Status | Production sources | Result |",
            "|---|---|---:|:---:|",
        ]
    )
    for row in report["explicit_unsupported"]:
        lines.append(
            f"| {row['mechanism']} | {row['status']} | "
            f"{row['production_source_count']} | "
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
        f"contracts={summary['behavior_contract_count']} "
        f"failures={summary['failure_count']} "
        f"passed={summary['passed']}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
