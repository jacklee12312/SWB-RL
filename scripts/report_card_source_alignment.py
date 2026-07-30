# -*- coding: utf-8 -*-
"""Audit training-closure card sources against normalized DB and rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Iterable, Mapping

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from scripts.report_card_bug_audit_baseline import render_json


SCHEMA_VERSION = 1
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_RULES = Path("data/rules")
DEFAULT_BASELINE = Path("data/reports/card_bug_audit/baseline.json")
DEFAULT_CLOSURE = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)
DEFAULT_MATRIX = Path(
    "data/reports/card_bug_audit/card_clause_matrix.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/card_bug_audit/source_alignment.json"
)
DEFAULT_MARKDOWN = Path(
    "data/reports/card_bug_audit/source_alignment.md"
)
DEFAULT_RULING_QUEUE = Path(
    "data/reports/card_bug_audit/ruling_queue.json"
)

EVIDENCE_HIERARCHY = (
    {
        "tier": 1,
        "kind": "official_qa_or_rules",
        "description": "官方 Q&A、综合规则或官方规则说明。",
    },
    {
        "tier": 2,
        "kind": "reproducible_client_result",
        "description": "记录版本、初始状态和操作序列的客户端可重复复现。",
    },
    {
        "tier": 3,
        "kind": "multiple_reliable_recordings_or_tests",
        "description": "多个互相独立且可核查的录像或测试。",
    },
    {
        "tier": 4,
        "kind": "single_guide_source",
        "description": "单个攻略或非官方资料来源。",
    },
    {
        "tier": 5,
        "kind": "text_only_inference",
        "description": "仅依据卡牌文字的推断；不能关闭裁定项。",
    },
)

SEMANTIC_SIGNALS = (
    {
        "signal": "condition",
        "pattern": r"若|如果",
        "rule_markers": (
            '"condition"',
            '"conditions"',
            '"target":"own_board"',
            "min_distinct_material_cards",
        ),
        "meaning": "条件及其取值时刻",
    },
    {
        "signal": "whenever",
        "pattern": r"每当",
        "rule_markers": (
            '"listeners"',
            '"event"',
            '"trigger"',
            '"triggers"',
        ),
        "meaning": "事件触发时机",
    },
    {
        "signal": "until",
        "pattern": r"直到|为止|结束前",
        "rule_markers": (
            "until",
            "end_of_turn",
            "turn_end",
            "duration",
            "expires",
        ),
        "meaning": "持续时间和失效时机",
    },
    {
        "signal": "this_turn",
        "pattern": r"本回合|这个回合",
        "rule_markers": (
            "until_end_of_turn",
            "end_of_turn",
            "turn_end",
            "duration",
            "temporary",
        ),
        "meaning": "本回合持续时间",
    },
    {
        "signal": "owner_turn",
        "pattern": r"自己的回合|己方回合",
        "rule_markers": (
            "owner_turn",
            "controller_turn",
            "turn_scope",
            '"trigger":"turn_end"',
            '"trigger":"turn_start"',
            '"event":"turn_end"',
            '"event":"turn_start"',
        ),
        "meaning": "己方回合范围",
    },
    {
        "signal": "random",
        "pattern": r"随机",
        "rule_markers": (
            "random",
            "summon_from_deck",
        ),
        "meaning": "随机候选和抽样语义",
    },
    {
        "signal": "all",
        "pattern": r"所有|全部",
        "rule_markers": (
            "all_",
            '"all"',
            "choose_one",
            "conditional",
        ),
        "meaning": "全体目标或全部分支语义",
    },
    {
        "signal": "one",
        "pattern": r"(?<!\d)1(?:个|张|点|次|种|名|回合)?",
        "rule_markers": (
            ":1",
            '"amount":1',
            '"value":1',
            '"count":1',
            '"max_count":1',
            '"min_count":1',
            '"requires_target":true',
            "random_enemy_unit",
            "random_own_unit",
        ),
        "meaning": "单一数量或单目标语义",
    },
    {
        "signal": "other",
        "pattern": r"其他",
        "rule_markers": (
            '"source_relation":"other"',
            '"exclude_source":true',
            '"other"',
        ),
        "meaning": "排除来源自身",
    },
    {
        "signal": "different_names",
        "pattern": (
            r"不同名|不同名称|名字各不相同|名称各不相同|"
            r"种类为\d+种"
        ),
        "rule_markers": (
            "distinct",
            "different_name",
            "unique_name",
        ),
        "meaning": "名称去重语义",
    },
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


def _group_entries(payload: object) -> Iterable[tuple[str, int, object]]:
    if isinstance(payload, list):
        for index, entry in enumerate(payload):
            yield "rules", index, entry
        return
    if not isinstance(payload, dict):
        return
    for group, entries in payload.items():
        if isinstance(entries, list):
            for index, entry in enumerate(entries):
                yield group, index, entry


def _rule_records(
    rules_directory: Path,
    root: Path,
) -> dict[int, list[dict[str, object]]]:
    records: dict[int, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(rules_directory.glob("*.json")):
        relative = _relative(path, root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for group, index, entry in _group_entries(payload):
            if group == "vanilla_cards":
                card_id = entry if isinstance(entry, int) else None
            elif isinstance(entry, dict):
                card_id = entry.get("card_id", entry.get("source_card_id"))
            else:
                card_id = None
            if not isinstance(card_id, int) or isinstance(card_id, bool):
                continue
            records[card_id].append(
                {
                    "entry_id": f"{relative}#{group}/{index}",
                    "group": group,
                    "value": entry,
                }
            )
    return {
        card_id: sorted(items, key=lambda item: item["entry_id"])
        for card_id, items in records.items()
    }


def _compact_rules(records: list[dict[str, object]]) -> str:
    return json.dumps(
        [record["value"] for record in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).lower()


def _strip_markup(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _semantic_alignment(
    clauses: list[dict[str, object]],
    raw_rules: list[dict[str, object]],
) -> list[dict[str, object]]:
    rules_text = _compact_rules(raw_rules)
    entry_ids = [record["entry_id"] for record in raw_rules]
    results: list[dict[str, object]] = []
    for definition in SEMANTIC_SIGNALS:
        matched_clauses = []
        for clause in clauses:
            text = _strip_markup(str(clause["texts"]["zh_CN"]))
            if re.search(str(definition["pattern"]), text):
                matched_clauses.append(
                    {
                        "clause_id": clause["clause_id"],
                        "text_sha256": clause["source_clause_sha256"],
                    }
                )
        if not matched_clauses:
            continue
        matching_markers = sorted(
            marker
            for marker in definition["rule_markers"]
            if marker.lower() in rules_text
        )
        results.append(
            {
                "signal": definition["signal"],
                "meaning": definition["meaning"],
                "source_clauses": matched_clauses,
                "status": "passed" if matching_markers else "ruling_uncertain",
                "rule_markers": matching_markers,
                "candidate_entry_ids": entry_ids,
            }
        )
    return results


def _raw_attribute_issues(
    row: sqlite3.Row,
    raw: Mapping[str, object],
    names: Mapping[str, str],
) -> list[str]:
    comparisons = {
        "card_id": (row["card_id"], raw.get("card_id")),
        "base_card_id": (row["base_card_id"], raw.get("base_card_id")),
        "card_set_id": (row["card_set_id"], raw.get("card_set_id")),
        "class_id": (row["class_id"], raw.get("class")),
        "type_id": (row["type_id"], raw.get("type")),
        "cost": (row["cost"], raw.get("cost")),
        "attack": (row["attack"], raw.get("atk")),
        "life": (row["life"], raw.get("life")),
        "tribe_id": (row["tribe_id"], raw.get("tribe")),
        "tribe_name": (row["tribe_name"], raw.get("tribe_name", "")),
        "name.zh_CN": (names["zh_CN"], raw.get("name_chs")),
        "name.en": (names["en"], raw.get("name_eng")),
        "name.ja": (names["ja"], raw.get("name_jpn")),
    }
    return [
        f"{field}: normalized={normalized!r}, raw={source!r}"
        for field, (normalized, source) in comparisons.items()
        if normalized != source
    ]


def _raw_clause_issues(
    clauses: list[dict[str, object]],
    raw: Mapping[str, object],
) -> list[str]:
    source_by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    for index, item in enumerate(raw.get("skill_texts", [])):
        if isinstance(item, dict):
            source_by_key[("main_skill", index)] = item
    for index, item in enumerate(raw.get("alt_modes", [])):
        if isinstance(item, dict):
            source_by_key[("alternate_mode", index)] = item
    issues: list[str] = []
    if len(clauses) != len(source_by_key):
        issues.append(
            f"clause count: normalized={len(clauses)}, raw={len(source_by_key)}"
        )
    language_keys = {"zh_CN": "text_chs", "en": "text_eng", "ja": "text_jpn"}
    for clause in clauses:
        key = (str(clause["source_kind"]), int(clause["position"]))
        source = source_by_key.get(key)
        if source is None:
            issues.append(f"{clause['clause_id']}: missing raw clause")
            continue
        for language, raw_key in language_keys.items():
            normalized = clause["texts"][language]
            if normalized != source.get(raw_key):
                issues.append(
                    f"{clause['clause_id']} {language}: raw text mismatch"
                )
        if clause["source_kind"] == "alternate_mode":
            if clause["mode_type"] != source.get("type"):
                issues.append(f"{clause['clause_id']}: mode_type mismatch")
            if clause["mode_cost"] != source.get("cost"):
                issues.append(f"{clause['clause_id']}: mode_cost mismatch")
    return issues


def _base_keyword_issues(
    card_abilities: list[dict[str, str]],
    clauses: list[dict[str, object]],
) -> list[str]:
    source_text = "\n".join(
        _strip_markup(str(clause["texts"]["zh_CN"])) for clause in clauses
    )
    issues = []
    for ability in card_abilities:
        if ability["registry_status"] not in {
            "implemented",
            "partial",
            "placeholder",
        }:
            issues.append(
                f"{ability['ability_keyword']}: invalid ability registry status"
            )
        if not ability["raw_keyword"]:
            issues.append(
                f"{ability['ability_keyword']}: missing raw keyword"
            )
        elif ability["raw_keyword"] not in source_text:
            issues.append(
                f"{ability['ability_keyword']}: raw keyword is absent from "
                "the Chinese source clauses"
            )
    return issues


def _card_names(
    connection: sqlite3.Connection,
) -> dict[int, dict[str, str]]:
    names: dict[int, dict[str, str]] = defaultdict(dict)
    language_map = {"zh-CN": "zh_CN", "en": "en", "ja": "ja"}
    for card_id, language, name in connection.execute(
        """
        SELECT card_id, language, name
        FROM card_names
        WHERE language IN ('zh-CN', 'en', 'ja')
        ORDER BY card_id, language
        """
    ):
        names[int(card_id)][language_map[str(language)]] = str(name)
    return dict(names)


def _card_abilities(
    connection: sqlite3.Connection,
) -> dict[int, list[dict[str, str]]]:
    abilities: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in connection.execute(
        """
        SELECT ca.card_id, ca.ability_keyword, ca.raw_keyword, a.status
        FROM card_abilities ca
        JOIN abilities a ON a.keyword = ca.ability_keyword
        ORDER BY ca.card_id, ca.ability_keyword
        """
    ):
        abilities[int(row[0])].append(
            {
                "ability_keyword": str(row[1]),
                "raw_keyword": str(row[2]),
                "registry_status": str(row[3]),
            }
        )
    return dict(abilities)


def _references(
    connection: sqlite3.Connection,
    closure_ids: set[int],
    names: Mapping[int, Mapping[str, str]],
) -> dict[int, list[dict[str, object]]]:
    output: dict[int, list[dict[str, object]]] = defaultdict(list)
    placeholders = ",".join("?" for _ in closure_ids)
    for row in connection.execute(
        f"""
        SELECT card_id, position, referenced_card_id, referenced_name
        FROM card_references
        WHERE card_id IN ({placeholders})
        ORDER BY card_id, position
        """,
        tuple(sorted(closure_ids)),
    ):
        card_id = int(row[0])
        referenced_card_id = row[2]
        target_names = (
            names.get(int(referenced_card_id), {})
            if referenced_card_id is not None
            else {}
        )
        named_target_matches = row[3] in set(target_names.values())
        output[card_id].append(
            {
                "position": int(row[1]),
                "referenced_card_id": referenced_card_id,
                "referenced_name": row[3],
                "target_names": target_names,
                "target_exists": bool(target_names),
                "referenced_name_matches_target": named_target_matches,
                "status": (
                    "passed"
                    if target_names and named_target_matches
                    else "failed"
                ),
            }
        )
    return dict(output)


def _validate_external_ruling(entry: Mapping[str, object]) -> None:
    if entry.get("status") not in {"ruling_uncertain", "confirmed", "rejected"}:
        raise ValueError(f"invalid ruling status: {entry.get('status')!r}")
    if entry.get("status") == "ruling_uncertain":
        return
    required = ("source_url", "retrieved_at", "conclusion", "summary")
    missing = [field for field in required if not entry.get(field)]
    if missing:
        raise ValueError(
            f"closed ruling {entry.get('ruling_id')}: missing {missing}"
        )
    if not str(entry["source_url"]).startswith(("https://", "http://")):
        raise ValueError(
            f"closed ruling {entry.get('ruling_id')}: invalid source URL"
        )


def validate_ruling_queue(queue: Mapping[str, object]) -> None:
    entries = queue.get("entries")
    if not isinstance(entries, list):
        raise ValueError("ruling queue entries must be a list")
    ruling_ids = [entry.get("ruling_id") for entry in entries]
    if ruling_ids != sorted(ruling_ids) or len(ruling_ids) != len(set(ruling_ids)):
        raise ValueError("ruling queue IDs must be unique and sorted")
    for entry in entries:
        _validate_external_ruling(entry)


def build_source_alignment(
    *,
    root: Path,
    database: Path,
    rules_directory: Path,
    baseline_report: Path,
    closure_report: Path,
    matrix_report: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    baseline = _load_json(baseline_report)
    closure = _load_json(closure_report)
    matrix = _load_json(matrix_report)
    expected_database_sha = matrix["generated_from"]["database_sha256"]
    actual_database_sha = _sha256_file(database)
    if actual_database_sha != expected_database_sha:
        raise ValueError(
            "source database changed since card clause matrix generation; "
            "regenerate baseline, closure, and matrix before source alignment"
        )
    closure_cards = {
        int(card["card_id"]): card for card in closure["cards"]
    }
    matrix_cards = {
        int(card["card_id"]): card for card in matrix["cards"]
    }
    if set(closure_cards) != set(matrix_cards):
        raise ValueError("closure and matrix card IDs differ")
    rules = _rule_records(rules_directory, root)

    report_cards: list[dict[str, object]] = []
    ruling_entries: list[dict[str, object]] = []
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        names = _card_names(connection)
        abilities = _card_abilities(connection)
        references = _references(connection, set(closure_cards), names)
        placeholders = ",".join("?" for _ in closure_cards)
        rows = connection.execute(
            f"""
            SELECT c.*, cl.name AS class_name, ct.name AS type_name,
                   rs.support_level, rs.keywords AS support_keywords
            FROM cards c
            JOIN classes cl ON cl.id = c.class_id
            JOIN card_types ct ON ct.id = c.type_id
            LEFT JOIN rule_support rs ON rs.card_id = c.card_id
            WHERE c.card_id IN ({placeholders})
            ORDER BY c.card_id
            """,
            tuple(sorted(closure_cards)),
        ).fetchall()
        if len(rows) != len(closure_cards):
            raise ValueError("not every closure card exists in the database")

        for row in rows:
            card_id = int(row["card_id"])
            matrix_card = matrix_cards[card_id]
            card_names = names.get(card_id, {})
            raw = json.loads(row["raw_json"])
            attribute_issues = _raw_attribute_issues(row, raw, card_names)
            clause_issues = _raw_clause_issues(matrix_card["clauses"], raw)
            card_abilities = abilities.get(card_id, [])
            keyword_issues = _base_keyword_issues(
                card_abilities,
                matrix_card["clauses"],
            )
            multilingual_issues = []
            for language in ("zh_CN", "en", "ja"):
                if not card_names.get(language):
                    multilingual_issues.append(f"missing {language} card name")
            for clause in matrix_card["clauses"]:
                for language in ("zh_CN", "en", "ja"):
                    if not clause["texts"].get(language):
                        multilingual_issues.append(
                            f"{clause['clause_id']}: missing {language} text"
                        )
            card_references = references.get(card_id, [])
            reference_issues = [
                (
                    f"reference {item['position']} does not resolve to the "
                    "named target"
                )
                for item in card_references
                if item["status"] != "passed"
            ]
            normalized_reference_projection = [
                {
                    "card_id": item["referenced_card_id"],
                    "name": item["referenced_name"],
                }
                for item in card_references
            ]
            raw_reference_projection = [
                {
                    "card_id": item.get("card_id"),
                    "name": item.get("name"),
                }
                for item in raw.get("references", [])
                if isinstance(item, dict)
            ]
            if normalized_reference_projection != raw_reference_projection:
                reference_issues.append(
                    "normalized references differ from the preserved raw import"
                )
            semantic = _semantic_alignment(
                matrix_card["clauses"],
                rules.get(card_id, []),
            )
            semantic_issues = [
                item for item in semantic if item["status"] != "passed"
            ]
            for item in semantic_issues:
                ruling_entries.append(
                    {
                        "ruling_id": (
                            f"card:{card_id}:signal:{item['signal']}"
                        ),
                        "card_id": card_id,
                        "card_name": card_names.get("zh_CN"),
                        "clause_ids": [
                            clause["clause_id"]
                            for clause in item["source_clauses"]
                        ],
                        "question": (
                            f"确认“{item['meaning']}”与结构化规则的准确对应。"
                        ),
                        "status": "ruling_uncertain",
                        "evidence_tier": None,
                        "source_url": None,
                        "retrieved_at": None,
                        "conclusion": None,
                        "summary": None,
                        "reason": "Source phrase has no explicit rule marker.",
                    }
                )
            validation_issues = (
                attribute_issues
                + clause_issues
                + keyword_issues
                + multilingual_issues
                + reference_issues
            )
            source_hash_state = matrix_card["source_validation"]["status"]
            source_hash_ok = source_hash_state in {"passed", "not_applicable"}
            status = (
                "passed"
                if not validation_issues
                and not semantic_issues
                and source_hash_ok
                else (
                    "ruling_uncertain"
                    if semantic_issues and not validation_issues
                    else "failed"
                )
            )
            support_keywords = json.loads(row["support_keywords"] or "[]")
            report_cards.append(
                {
                    "audit_id": matrix_card["audit_id"],
                    "card_id": card_id,
                    "names": card_names,
                    "origin": matrix_card["origin"],
                    "is_collectible": matrix_card["is_collectible"],
                    "printed_fields": {
                        "base_card_id": row["base_card_id"],
                        "card_set_id": row["card_set_id"],
                        "class_id": row["class_id"],
                        "class_name": row["class_name"],
                        "type_id": row["type_id"],
                        "type_name": row["type_name"],
                        "cost": row["cost"],
                        "attack": row["attack"],
                        "life": row["life"],
                        "tribe_id": row["tribe_id"],
                        "tribe_name": row["tribe_name"],
                    },
                    "base_keywords": {
                        "rule_support_level": row["support_level"],
                        "declared_support_keywords": support_keywords,
                        "normalized_abilities": card_abilities,
                        "status": (
                            "passed" if not keyword_issues else "failed"
                        ),
                        "issues": keyword_issues,
                    },
                    "source_texts": {
                        "clause_count": len(matrix_card["clauses"]),
                        "clauses": [
                            {
                                "clause_id": clause["clause_id"],
                                "source_kind": clause["source_kind"],
                                "position": clause["position"],
                                "source_clause_sha256": (
                                    clause["source_clause_sha256"]
                                ),
                                "texts": clause["texts"],
                            }
                            for clause in matrix_card["clauses"]
                        ],
                        "source_text_sha256": (
                            matrix_card["source_validation"][
                                "actual_source_text_sha256"
                            ]
                        ),
                        "source_hash_status": source_hash_state,
                        "raw_import_alignment": (
                            "passed" if not clause_issues else "failed"
                        ),
                    },
                    "printed_field_alignment": {
                        "status": (
                            "passed" if not attribute_issues else "failed"
                        ),
                        "issues": attribute_issues,
                    },
                    "multilingual_completeness": {
                        "status": (
                            "passed" if not multilingual_issues else "failed"
                        ),
                        "issues": multilingual_issues,
                    },
                    "references": card_references,
                    "reference_alignment": {
                        "status": (
                            "passed" if not reference_issues else "failed"
                        ),
                        "issues": reference_issues,
                    },
                    "semantic_signals": semantic,
                    "semantic_alignment": {
                        "status": (
                            "passed"
                            if not semantic_issues
                            else "ruling_uncertain"
                        ),
                        "issues": [
                            item["signal"] for item in semantic_issues
                        ],
                    },
                    "direct_tests": matrix_card["direct_tests"],
                    "status": status,
                    "issues": validation_issues,
                }
            )

    report_cards.sort(key=lambda card: card["card_id"])
    ruling_entries.sort(key=lambda entry: entry["ruling_id"])
    status_counts = Counter(card["status"] for card in report_cards)
    signal_counts = Counter(
        signal["signal"]
        for card in report_cards
        for signal in card["semantic_signals"]
    )
    uncertain_signal_counts = Counter(
        signal["signal"]
        for card in report_cards
        for signal in card["semantic_signals"]
        if signal["status"] != "passed"
    )
    signal_coverage = {
        str(definition["signal"]): {
            "source_occurrences": signal_counts[str(definition["signal"])],
            "status": (
                "not_applicable"
                if signal_counts[str(definition["signal"])] == 0
                else (
                    "ruling_uncertain"
                    if uncertain_signal_counts[str(definition["signal"])]
                    else "passed"
                )
            ),
        }
        for definition in SEMANTIC_SIGNALS
    }
    generated_from = {
        "database": _relative(database, root),
        "database_sha256": actual_database_sha,
        "rules_directory": _relative(rules_directory, root),
        "rules_snapshot_sha256": baseline["audit_artifacts"]["rulebook"][
            "sha256"
        ],
        "baseline": _relative(baseline_report, root),
        "baseline_sha256": _sha256_file(baseline_report),
        "closure": _relative(closure_report, root),
        "closure_sha256": _sha256_file(closure_report),
        "matrix": _relative(matrix_report, root),
        "matrix_sha256": _sha256_file(matrix_report),
        "last_verified_commit": matrix["generated_from"][
            "last_verified_commit"
        ],
    }
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "swb_card_source_alignment",
        "generated_from": generated_from,
        "summary": {
            "card_count": len(report_cards),
            "collectible_count": sum(
                bool(card["is_collectible"]) for card in report_cards
            ),
            "generated_count": sum(
                not bool(card["is_collectible"]) for card in report_cards
            ),
            "source_clause_count": sum(
                card["source_texts"]["clause_count"]
                for card in report_cards
            ),
            "reference_count": sum(
                len(card["references"]) for card in report_cards
            ),
            "semantic_signal_counts": dict(sorted(signal_counts.items())),
            "semantic_signal_coverage": signal_coverage,
            "passed": status_counts["passed"],
            "ruling_uncertain": status_counts["ruling_uncertain"],
            "failed": status_counts["failed"],
            "source_alignment_gate_ready": (
                status_counts["ruling_uncertain"] == 0
                and status_counts["failed"] == 0
            ),
        },
        "cards": report_cards,
    }
    queue: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "swb_card_ruling_queue",
        "generated_from": generated_from,
        "evidence_hierarchy": list(EVIDENCE_HIERARCHY),
        "external_evidence_contract": {
            "required_for_closed_entries": [
                "source_url",
                "retrieved_at",
                "conclusion",
                "summary",
            ],
            "summary_rule": (
                "Store only the minimum summary needed to support the ruling."
            ),
            "text_only_inference_can_close_entry": False,
        },
        "summary": {
            "entry_count": len(ruling_entries),
            "open_count": sum(
                entry["status"] == "ruling_uncertain"
                for entry in ruling_entries
            ),
            "source_alignment_blocks_training": bool(ruling_entries),
            "scope_note": (
                "Zero entries means no unresolved source-to-structure signal "
                "was found in this 147-card closure. It does not substitute "
                "for later official/client or runtime audit dimensions."
            ),
        },
        "entries": ruling_entries,
    }
    validate_ruling_queue(queue)
    return report, queue


def render_markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# Training Closure Source Alignment",
        "",
        f"- Cards: **{summary['card_count']}** "
        f"({summary['collectible_count']} collectible, "
        f"{summary['generated_count']} generated)",
        f"- Source clauses: **{summary['source_clause_count']}**",
        f"- Card references: **{summary['reference_count']}**",
        f"- Passed: **{summary['passed']}**",
        f"- Ruling uncertain: **{summary['ruling_uncertain']}**",
        f"- Failed: **{summary['failed']}**",
        f"- Source alignment gate ready: "
        f"**{str(summary['source_alignment_gate_ready']).lower()}**",
        "",
        "This report compares normalized SQLite fields and multilingual clauses "
        "to each card's preserved import record, validates referenced IDs and "
        "names, and maps audited source phrases to full structured rule entries.",
        "",
        "## Semantic Signals",
        "",
        "| Signal | Cards / clauses matched |",
        "|---|---:|",
    ]
    for signal, coverage in summary["semantic_signal_coverage"].items():
        lines.append(
            f"| `{signal}` | {coverage['source_occurrences']} "
            f"(`{coverage['status']}`) |"
        )
    lines.extend(
        [
            "",
            "## Cards",
            "",
            "| Card | Status | Clauses | References | Semantic signals |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for card in report["cards"]:
        lines.append(
            f"| {card['card_id']} {card['names'].get('zh_CN', '')} | "
            f"`{card['status']}` | "
            f"{card['source_texts']['clause_count']} | "
            f"{len(card['references'])} | "
            f"{len(card['semantic_signals'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--ruling-queue",
        type=Path,
        default=DEFAULT_RULING_QUEUE,
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report, queue = build_source_alignment(
        root=root,
        database=args.database,
        rules_directory=args.rules,
        baseline_report=args.baseline,
        closure_report=args.closure,
        matrix_report=args.matrix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_json(report), encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    args.ruling_queue.write_text(render_json(queue), encoding="utf-8")
    print(f"JSON source alignment written to {args.output}")
    print(f"Markdown source alignment written to {args.markdown}")
    print(f"Ruling queue written to {args.ruling_queue}")


if __name__ == "__main__":
    main()
