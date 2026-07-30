# -*- coding: utf-8 -*-
"""Build the deterministic per-card, per-clause runtime audit matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from scripts.report_card_bug_audit_baseline import render_json
from scripts.report_rule_coverage import (
    _load_source_text_map,
    _source_text_sha256,
)


SCHEMA_VERSION = 1
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_RULES = Path("data/rules")
DEFAULT_COVERAGE = Path("data/reports/rule_coverage.json")
DEFAULT_TOKEN_AUDIT = Path("data/reports/token_audit.json")
DEFAULT_BASELINE = Path("data/reports/card_bug_audit/baseline.json")
DEFAULT_CLOSURE = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)
DEFAULT_JSON_OUTPUT = Path(
    "data/reports/card_bug_audit/card_clause_matrix.json"
)
DEFAULT_MARKDOWN_OUTPUT = Path(
    "data/reports/card_bug_audit/card_clause_matrix.md"
)

AUDIT_STATUSES = (
    "not_applicable",
    "passed",
    "not_tested",
    "ruling_uncertain",
    "failed",
)

STATUS_DEFINITIONS = (
    {
        "status": "not_applicable",
        "meaning": "该卡或条款不使用此机制；报告必须保存判定理由。",
    },
    {
        "status": "passed",
        "meaning": "有当前 source hash、可解析结构和命名测试/证据支持。",
    },
    {
        "status": "not_tested",
        "meaning": "适用但尚未完成本轮运行时边界验证，不能推定通过。",
    },
    {
        "status": "ruling_uncertain",
        "meaning": "规则证据不足，等待官方裁定或客户端复现。",
    },
    {
        "status": "failed",
        "meaning": "已保存与预期不符的复现，必须进入 Bug 台账。",
    },
)

AUDIT_DIMENSIONS = (
    "source_mapping",
    "normal_path",
    "cost_or_threshold_boundary",
    "alternate_modes",
    "entry_methods",
    "targeting_and_choices",
    "timing_and_priority",
    "zones_and_capacity",
    "randomness_and_determinism",
    "command_action_mask_consistency",
    "runtime_execution_coverage",
    "official_ruling_or_client_reproduction",
)

ZONE_OPERATION_KINDS = frozenset(
    {
        "add_card",
        "add_card_to_deck",
        "banish_unit",
        "copy_card",
        "destroy_unit",
        "discard",
        "draw",
        "draw_filtered",
        "replace_deck",
        "return_to_deck",
        "return_to_hand",
        "summon",
        "transform",
        "transform_deck_cards",
    }
)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _normalize_repo_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("test evidence path must be a non-empty string")
    path = PurePosixPath(value.strip().replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"test evidence must be repository-relative: {value}")
    return path.as_posix()


def validate_test_references(paths: Iterable[object], root: Path) -> list[str]:
    normalized = sorted({_normalize_repo_path(path) for path in paths})
    missing = [path for path in normalized if not (root / path).is_file()]
    if missing:
        raise ValueError(f"broken test evidence references: {missing}")
    return normalized


def validate_source_hash(
    card_id: int,
    source_texts: list[str],
    expected_sha256: object,
) -> str:
    actual = _source_text_sha256(source_texts)
    if not isinstance(expected_sha256, str):
        raise ValueError(f"card {card_id}: source_text_sha256 is missing")
    if actual != expected_sha256:
        raise ValueError(
            f"card {card_id}: stale source_text_sha256; "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


def _walk_with_path(
    value: object,
    path: str = "",
) -> Iterable[tuple[str, dict[str, object]]]:
    if isinstance(value, dict):
        yield path or "/", value
        for key, nested in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from _walk_with_path(nested, f"{path}/{escaped}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _walk_with_path(nested, f"{path}/{index}")


def _group_entries(payload: object) -> Iterable[tuple[str, int, object]]:
    if isinstance(payload, list):
        yield from (
            ("rules", index, entry) for index, entry in enumerate(payload)
        )
        return
    if not isinstance(payload, dict):
        return
    for group, entries in payload.items():
        if isinstance(entries, list):
            yield from (
                (group, index, entry) for index, entry in enumerate(entries)
            )


def _positive_card_id(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _deduplicate_records(
    records: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    unique = {
        json.dumps(record, ensure_ascii=False, sort_keys=True): record
        for record in records
    }
    return [unique[key] for key in sorted(unique)]


def _structured_entries(
    rules_directory: Path,
    root: Path,
) -> dict[int, list[dict[str, object]]]:
    by_card: dict[int, list[dict[str, object]]] = defaultdict(list)
    for file_path in sorted(rules_directory.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        relative_path = _relative(file_path, root)
        for group, index, entry in _group_entries(payload):
            if group == "vanilla_cards":
                card_id = _positive_card_id(entry)
                if card_id is not None:
                    by_card[card_id].append(
                        {
                            "entry_id": (
                                f"{relative_path}#{group}/{index}"
                            ),
                            "source_path": relative_path,
                            "rule_group": group,
                            "entry_index": index,
                            "trigger": "audited_vanilla_declaration",
                            "conditions": [],
                            "targets": [],
                            "operations": [],
                        }
                    )
                continue
            if not isinstance(entry, dict):
                continue
            card_id = _positive_card_id(
                entry.get("card_id", entry.get("source_card_id"))
            )
            if card_id is None:
                continue
            conditions: list[dict[str, object]] = []
            targets: list[dict[str, object]] = []
            operations: list[dict[str, object]] = []
            for node_path, node in _walk_with_path(entry):
                for condition_key in ("condition", "conditions"):
                    if condition_key in node:
                        conditions.append(
                            {
                                "path": f"{node_path}/{condition_key}",
                                "value": node[condition_key],
                            }
                        )
                if "target" in node:
                    targets.append(
                        {
                            "path": node_path,
                            "target": node["target"],
                            "requires_target": node.get("requires_target"),
                            "target_key": node.get("target_key"),
                            "filter": {
                                key: value
                                for key, value in node.items()
                                if key
                                not in {
                                    "kind",
                                    "target",
                                    "requires_target",
                                    "target_key",
                                    "operations",
                                    "then_operations",
                                    "else_operations",
                                }
                            },
                        }
                    )
                kind = node.get("kind")
                if isinstance(kind, str):
                    operations.append(
                        {
                            "path": node_path,
                            "kind": kind,
                            "target": node.get("target"),
                            "payload": {
                                key: value
                                for key, value in node.items()
                                if key
                                not in {
                                    "operations",
                                    "then_operations",
                                    "else_operations",
                                    "triggers",
                                }
                            },
                        }
                    )
            trigger = entry.get("trigger")
            if not isinstance(trigger, str):
                trigger = {
                    "activations": "activate_definition",
                    "emblems": "emblem_definition",
                    "faiths": "faith_definition",
                    "fusions": "fusion_definition",
                    "intrinsic_keywords": "intrinsic_keyword_declaration",
                    "invocations": "invocation_definition",
                    "listeners": "listener_definition",
                    "passives": "passive_definition",
                    "play_modes": "play_mode_definition",
                    "union_bursts": "union_burst_definition",
                }.get(group, f"{group}_definition")
            by_card[card_id].append(
                {
                    "entry_id": f"{relative_path}#{group}/{index}",
                    "source_path": relative_path,
                    "rule_group": group,
                    "entry_index": index,
                    "trigger": trigger,
                    "conditions": _deduplicate_records(conditions),
                    "targets": _deduplicate_records(targets),
                    "operations": _deduplicate_records(operations),
                }
            )
    for card_id, entries in by_card.items():
        by_card[card_id] = sorted(
            entries,
            key=lambda item: (
                item["source_path"],
                item["rule_group"],
                item["entry_index"],
            ),
        )
    return by_card


def _source_clauses(
    connection: sqlite3.Connection,
    closure_ids: set[int],
) -> dict[int, list[dict[str, object]]]:
    by_card: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT card_id, position, text_chs, text_eng, text_jpn
        FROM skill_texts
        ORDER BY card_id, position
        """
    ):
        card_id = int(row[0])
        if card_id not in closure_ids:
            continue
        position = int(row[1])
        clause = {
            "clause_id": f"{card_id}:main:{position}",
            "source_kind": "main_skill",
            "position": position,
            "mode_type": None,
            "mode_cost": None,
            "texts": {
                "zh_CN": row[2],
                "en": row[3],
                "ja": row[4],
            },
        }
        clause["source_clause_sha256"] = hashlib.sha256(
            json.dumps(
                clause,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        by_card[card_id].append(clause)
    for row in connection.execute(
        """
        SELECT card_id, position, mode_type, cost,
               text_chs, text_eng, text_jpn
        FROM alt_modes
        ORDER BY card_id, position
        """
    ):
        card_id = int(row[0])
        if card_id not in closure_ids:
            continue
        position = int(row[1])
        clause = {
            "clause_id": f"{card_id}:alternate_mode:{position}",
            "source_kind": "alternate_mode",
            "position": position,
            "mode_type": row[2],
            "mode_cost": row[3],
            "texts": {
                "zh_CN": row[4],
                "en": row[5],
                "ja": row[6],
            },
        }
        clause["source_clause_sha256"] = hashlib.sha256(
            json.dumps(
                clause,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        by_card[card_id].append(clause)
    for card_id, clauses in by_card.items():
        by_card[card_id] = sorted(
            clauses,
            key=lambda item: (
                0 if item["source_kind"] == "main_skill" else 1,
                item["position"],
            ),
        )
    return by_card


def _status(
    status: str,
    reason: str,
    evidence: Iterable[str] = (),
) -> dict[str, object]:
    if status not in AUDIT_STATUSES:
        raise ValueError(f"invalid audit status {status!r}")
    return {
        "status": status,
        "reason": reason,
        "evidence": sorted(set(evidence)),
    }


def _candidate_summary(
    entries: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "candidate_entry_ids": [entry["entry_id"] for entry in entries],
        "triggers": sorted({str(entry["trigger"]) for entry in entries}),
        "conditions": [
            {
                "entry_id": entry["entry_id"],
                **condition,
            }
            for entry in entries
            for condition in entry["conditions"]
        ],
        "targets": [
            {
                "entry_id": entry["entry_id"],
                **target,
            }
            for entry in entries
            for target in entry["targets"]
        ],
        "operations": [
            {
                "entry_id": entry["entry_id"],
                "path": operation["path"],
                "kind": operation["kind"],
                "target": operation["target"],
            }
            for entry in entries
            for operation in entry["operations"]
        ],
    }


def _dimension_states(
    *,
    card: Mapping[str, object],
    clauses: list[dict[str, object]],
    structured_entries: list[dict[str, object]],
    source_mapping_passed: bool,
    direct_tests: list[str],
    official_evidence: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    candidate = _candidate_summary(structured_entries)
    operation_kinds = {
        operation["kind"] for operation in candidate["operations"]
    }
    target_values = [
        str(target["target"]).lower() for target in candidate["targets"]
    ]
    serialized_structure = json.dumps(
        structured_entries,
        ensure_ascii=False,
        sort_keys=True,
    ).lower()
    has_alternate_modes = any(
        clause["source_kind"] == "alternate_mode" for clause in clauses
    ) or any(
        entry["rule_group"] in {"play_modes", "fusions", "invocations"}
        for entry in structured_entries
    )
    has_cost_or_threshold = has_alternate_modes or any(
        token in serialized_structure
        for token in (
            "cost",
            "mana",
            "threshold",
            "combo",
            "overflow",
            "spellboost",
            "skybound",
        )
    )
    has_targets = bool(candidate["targets"])
    has_randomness = any("random" in value for value in target_values) or any(
        "random" in kind or "distribute" in kind for kind in operation_kinds
    )
    has_zone_behavior = bool(operation_kinds & ZONE_OPERATION_KINDS)
    has_player_action = bool(card["is_collectible"]) or any(
        edge["relation"] in {"add_to_hand", "add_to_deck"}
        for edge in card["references"]["incoming"]
    )
    official_complete = bool(
        official_evidence.get("url")
        and official_evidence.get("retrieved_at")
        and official_evidence.get("ruling")
    )
    source_evidence = direct_tests if source_mapping_passed else ()
    return {
        "source_mapping": _status(
            "passed" if source_mapping_passed else "failed",
            (
                "Current source hash and exact collectible/token audit agree."
                if source_mapping_passed
                else "Source hash or structural audit does not resolve."
            ),
            source_evidence,
        ),
        "normal_path": _status(
            "not_tested",
            "Existing direct tests are indexed, but normal-path results have "
            "not yet been re-executed and attributed in this runtime audit.",
            direct_tests,
        ),
        "cost_or_threshold_boundary": _status(
            "not_tested" if has_cost_or_threshold else "not_applicable",
            (
                "Structured candidates contain a cost, mode, or threshold."
                if has_cost_or_threshold
                else "No cost, mode, or threshold candidate was found."
            ),
        ),
        "alternate_modes": _status(
            "not_tested" if has_alternate_modes else "not_applicable",
            (
                "Alternate-mode or special-play definitions are present."
                if has_alternate_modes
                else "No alternate-mode or special-play definition is present."
            ),
        ),
        "entry_methods": _status(
            "not_tested",
            "Normal play and every referenced/generated entry path require the "
            "entry-method matrix.",
        ),
        "targeting_and_choices": _status(
            "not_tested" if has_targets else "not_applicable",
            (
                "Structured candidates contain target specifications."
                if has_targets
                else "No structured target specification was found."
            ),
        ),
        "timing_and_priority": _status(
            "not_tested" if structured_entries else "not_applicable",
            (
                "Structured triggers or declarations require timing review."
                if structured_entries
                else "No card-specific trigger is present."
            ),
        ),
        "zones_and_capacity": _status(
            "not_tested" if has_zone_behavior or has_player_action else "not_applicable",
            (
                "The card is playable/generated or has a zone-changing operation."
                if has_zone_behavior or has_player_action
                else "No playable or zone-changing path was found."
            ),
        ),
        "randomness_and_determinism": _status(
            "not_tested" if has_randomness else "not_applicable",
            (
                "Random/distributed structured behavior is present."
                if has_randomness
                else "No random structured behavior was found."
            ),
        ),
        "command_action_mask_consistency": _status(
            "not_tested" if has_player_action else "not_applicable",
            (
                "The card can participate in a player decision."
                if has_player_action
                else "No player-command entry path was found."
            ),
        ),
        "runtime_execution_coverage": _status(
            "not_tested",
            "Structural exact coverage is not runtime branch coverage.",
        ),
        "official_ruling_or_client_reproduction": _status(
            "passed" if official_complete else "not_tested",
            (
                "URL, retrieval date, and ruling are all recorded."
                if official_complete
                else "No complete official ruling/client reproduction is "
                "attributed for this audit row."
            ),
        ),
    }


def build_card_clause_matrix(
    *,
    root: Path,
    database: Path,
    rules_directory: Path,
    coverage_report: Path,
    token_audit: Path,
    baseline_report: Path,
    closure_report: Path,
) -> dict[str, object]:
    baseline = _load_json(baseline_report)
    closure = _load_json(closure_report)
    coverage = _load_json(coverage_report)
    token_report = _load_json(token_audit)
    classifications = coverage.get("classifications")
    if not isinstance(classifications, dict):
        raise ValueError("coverage report has no classifications mapping")
    closure_cards = closure.get("cards")
    if not isinstance(closure_cards, list):
        raise ValueError("closure report has no cards array")
    closure_ids = {int(card["card_id"]) for card in closure_cards}
    structured_by_card = _structured_entries(rules_directory, root)
    token_by_id = {
        int(card["card_id"]): card
        for card in token_report.get("cards", [])
        if isinstance(card, dict) and "card_id" in card
    }
    with closing(sqlite3.connect(database)) as connection:
        source_clauses = _source_clauses(connection, closure_ids)
        source_text_map = _load_source_text_map(connection)

    outgoing: dict[int, list[dict[str, object]]] = defaultdict(list)
    for edge in closure.get("reference_edges", []):
        outgoing[int(edge["source_card_id"])].append(edge)
    for edges in outgoing.values():
        edges.sort(
            key=lambda edge: (
                edge["target_card_id"],
                edge["relation"],
                edge["evidence_path"],
            )
        )

    baseline_commit = baseline["git_audit_start"]["commit"]
    report_cards: list[dict[str, object]] = []
    all_test_evidence: set[str] = set()
    source_hash_registry_validated = 0
    source_hash_frozen_by_matrix = 0
    source_hash_not_applicable = 0
    for closure_card in sorted(
        closure_cards,
        key=lambda item: int(item["card_id"]),
    ):
        card_id = int(closure_card["card_id"])
        classification = classifications.get(str(card_id))
        if not isinstance(classification, dict):
            raise ValueError(f"card {card_id}: missing coverage classification")
        clause_audit = classification.get("clause_audit")
        if not isinstance(clause_audit, dict):
            raise ValueError(f"card {card_id}: missing clause audit")
        direct_tests = validate_test_references(
            clause_audit.get("test_evidence", []),
            root,
        )
        all_test_evidence.update(direct_tests)
        expected_source_hash = clause_audit.get("source_text_sha256")
        card_source_texts = source_text_map.get(card_id, [])
        if expected_source_hash is None:
            if card_source_texts:
                source_hash_actual = _source_text_sha256(card_source_texts)
                source_hash_status = "passed"
                source_hash_provenance = "card_clause_matrix"
                source_hash_frozen_by_matrix += 1
            else:
                source_hash_status = "not_applicable"
                source_hash_actual = _source_text_sha256([])
                source_hash_provenance = "no_source_text"
                source_hash_not_applicable += 1
        else:
            source_hash_actual = validate_source_hash(
                card_id,
                card_source_texts,
                expected_source_hash,
            )
            source_hash_status = "passed"
            source_hash_provenance = "clause_audit_registry"
            source_hash_registry_validated += 1

        token_entry = token_by_id.get(card_id, {})
        source_mapping_passed = (
            classification.get("coverage") == "covered_exact"
            if closure_card["is_collectible"]
            else token_entry.get("category") == "entry_behavior_complete"
        ) and source_hash_status in {"passed", "not_applicable"}
        metadata = classification.get("rule_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        official_evidence = {
            "url": metadata.get("official_source_url"),
            "retrieved_at": metadata.get("official_source_retrieved_at"),
            "ruling": metadata.get("official_ruling"),
            "client_reproduction": None,
        }
        structured_entries = structured_by_card.get(card_id, [])
        candidate_mapping = _candidate_summary(structured_entries)
        clauses: list[dict[str, object]] = []
        for source_clause in source_clauses.get(card_id, []):
            clauses.append(
                {
                    **source_clause,
                    "mapping_status": "not_tested",
                    "mapping_note": (
                        "Card-level structured candidates are recorded below; "
                        "this clause still requires explicit trigger/condition/"
                        "target/operation attribution during the runtime audit."
                    ),
                    "structured_mapping": candidate_mapping,
                    "direct_tests": direct_tests,
                    "mechanic_tests": [],
                    "official_evidence": official_evidence,
                    "last_verified_commit": baseline_commit,
                }
            )
        references = {
            "incoming": closure_card.get("incoming_references", []),
            "outgoing": outgoing.get(card_id, []),
        }
        status_card = {
            **closure_card,
            "deck_total_copies": sum(
                int(item["copies"])
                for item in closure_card.get("deck_membership", [])
            ),
            "references": references,
        }
        dimensions = _dimension_states(
            card=status_card,
            clauses=clauses,
            structured_entries=structured_entries,
            source_mapping_passed=source_mapping_passed,
            direct_tests=direct_tests,
            official_evidence=official_evidence,
        )
        report_cards.append(
            {
                "audit_id": closure_card["audit_id"],
                "card_id": card_id,
                "name": closure_card["name"],
                "class_id": closure_card["class_id"],
                "class_name": closure_card["class_name"],
                "card_type": closure_card["card_type"],
                "is_collectible": closure_card["is_collectible"],
                "origin": closure_card["origin"],
                "deck_membership": closure_card.get("deck_membership", []),
                "deck_total_copies": status_card["deck_total_copies"],
                "references": references,
                "source_validation": {
                    "status": source_hash_status,
                    "provenance": source_hash_provenance,
                    "expected_source_text_sha256": expected_source_hash,
                    "actual_source_text_sha256": source_hash_actual,
                },
                "coverage_status": classification.get("coverage"),
                "token_audit_category": token_entry.get("category"),
                "structured_entries": structured_entries,
                "direct_tests": direct_tests,
                "mechanic_tests": [],
                "official_evidence": official_evidence,
                "last_verified_commit": baseline_commit,
                "dimensions": dimensions,
                "clauses": clauses,
            }
        )

    clause_counts = Counter(
        clause["source_kind"]
        for card in report_cards
        for clause in card["clauses"]
    )
    dimension_counts = {
        dimension: {
            status: sum(
                card["dimensions"][dimension]["status"] == status
                for card in report_cards
            )
            for status in AUDIT_STATUSES
        }
        for dimension in AUDIT_DIMENSIONS
    }
    open_runtime_rows = sum(
        state["status"] in {"not_tested", "ruling_uncertain", "failed"}
        for card in report_cards
        for dimension, state in card["dimensions"].items()
        if dimension != "source_mapping"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "swb_card_clause_runtime_audit_matrix",
        "status_definitions": list(STATUS_DEFINITIONS),
        "dimension_order": list(AUDIT_DIMENSIONS),
        "generated_from": {
            "database": _relative(database, root),
            "database_sha256": _sha256_file(database),
            "rules_directory": _relative(rules_directory, root),
            "coverage_report": _relative(coverage_report, root),
            "coverage_report_sha256": _sha256_file(coverage_report),
            "token_audit": _relative(token_audit, root),
            "token_audit_sha256": _sha256_file(token_audit),
            "baseline": _relative(baseline_report, root),
            "baseline_sha256": _sha256_file(baseline_report),
            "closure": _relative(closure_report, root),
            "closure_sha256": _sha256_file(closure_report),
            "last_verified_commit": baseline_commit,
        },
        "summary": {
            "card_count": len(report_cards),
            "collectible_count": sum(
                bool(card["is_collectible"]) for card in report_cards
            ),
            "non_collectible_count": sum(
                not bool(card["is_collectible"]) for card in report_cards
            ),
            "clause_count": sum(
                len(card["clauses"]) for card in report_cards
            ),
            "clause_kinds": {
                "main_skill": clause_counts.get("main_skill", 0),
                "alternate_mode": clause_counts.get("alternate_mode", 0),
            },
            "cards_without_source_clauses": sum(
                not card["clauses"] for card in report_cards
            ),
            "source_hash_registry_validated": source_hash_registry_validated,
            "source_hash_frozen_by_matrix": source_hash_frozen_by_matrix,
            "source_hash_not_applicable": source_hash_not_applicable,
            "unique_test_evidence_files": len(all_test_evidence),
            "dimension_status_counts": dimension_counts,
            "open_runtime_audit_rows": open_runtime_rows,
            "training_runtime_gate_ready": open_runtime_rows == 0,
            "validation_issues": [],
        },
        "cards": report_cards,
    }


def validate_matrix_shape(report: Mapping[str, object]) -> None:
    cards = report.get("cards")
    if not isinstance(cards, list):
        raise ValueError("matrix cards must be a list")
    card_ids = [card.get("card_id") for card in cards]
    if card_ids != sorted(card_ids) or len(card_ids) != len(set(card_ids)):
        raise ValueError("matrix cards must have unique sorted card IDs")
    audit_ids = [card.get("audit_id") for card in cards]
    if len(audit_ids) != len(set(audit_ids)):
        raise ValueError("matrix card audit IDs must be unique")
    clause_ids: list[object] = []
    for card in cards:
        dimensions = card.get("dimensions")
        if not isinstance(dimensions, dict) or tuple(dimensions) != AUDIT_DIMENSIONS:
            raise ValueError(
                f"card {card.get('card_id')}: incomplete audit dimensions"
            )
        for state in dimensions.values():
            if state.get("status") not in AUDIT_STATUSES:
                raise ValueError(
                    f"card {card.get('card_id')}: invalid dimension status"
                )
        for clause in card.get("clauses", []):
            clause_ids.append(clause.get("clause_id"))
            required = {
                "clause_id",
                "source_kind",
                "source_clause_sha256",
                "texts",
                "mapping_status",
                "structured_mapping",
                "direct_tests",
                "mechanic_tests",
                "official_evidence",
                "last_verified_commit",
            }
            if not required.issubset(clause):
                raise ValueError(
                    f"clause {clause.get('clause_id')}: incomplete fields"
                )
    if len(clause_ids) != len(set(clause_ids)):
        raise ValueError("matrix clause IDs must be unique")


def render_markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# Card / Clause Runtime Audit Matrix",
        "",
        f"- Cards: **{summary['card_count']}** "
        f"({summary['collectible_count']} collectible, "
        f"{summary['non_collectible_count']} generated)",
        f"- Source clauses: **{summary['clause_count']}** "
        f"({summary['clause_kinds']['main_skill']} main, "
        f"{summary['clause_kinds']['alternate_mode']} alternate mode)",
        f"- Open runtime audit rows: **{summary['open_runtime_audit_rows']}**",
        f"- Training runtime gate ready: "
        f"**{str(summary['training_runtime_gate_ready']).lower()}**",
        "",
        "Structural exact coverage is recorded only in `source_mapping`. "
        "Every other applicable runtime dimension remains `not_tested` until "
        "its checklist scan produces direct evidence.",
        "",
        "## Dimension Status",
        "",
        "| Dimension | N/A | Passed | Not tested | Ruling uncertain | Failed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dimension in report["dimension_order"]:
        counts = summary["dimension_status_counts"][dimension]
        lines.append(
            f"| `{dimension}` | {counts['not_applicable']} | "
            f"{counts['passed']} | {counts['not_tested']} | "
            f"{counts['ruling_uncertain']} | {counts['failed']} |"
        )
    lines.extend(
        [
            "",
            "## Cards",
            "",
            "| Card | Class / Type | Deck copies | Clauses | "
            "Structured entries | Open dimensions |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for card in report["cards"]:
        open_dimensions = sum(
            state["status"]
            in {"not_tested", "ruling_uncertain", "failed"}
            for name, state in card["dimensions"].items()
            if name != "source_mapping"
        )
        lines.append(
            f"| {card['card_id']} {card['name']} | "
            f"{card['class_name']} / {card['card_type']} | "
            f"{card['deck_total_copies']} | {len(card['clauses'])} | "
            f"{len(card['structured_entries'])} | {open_dimensions} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--token-audit", type=Path, default=DEFAULT_TOKEN_AUDIT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--markdown",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = build_card_clause_matrix(
        root=root,
        database=args.database,
        rules_directory=args.rules,
        coverage_report=args.coverage,
        token_audit=args.token_audit,
        baseline_report=args.baseline,
        closure_report=args.closure,
    )
    validate_matrix_shape(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_json(report), encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"JSON matrix written to {args.output}")
    print(f"Markdown matrix written to {args.markdown}")


if __name__ == "__main__":
    main()
