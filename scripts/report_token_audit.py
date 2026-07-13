# -*- coding: utf-8 -*-
"""Audit generated/non-collectible cards and their executable entry paths."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from scripts.report_rule_coverage import _load_rule_metadata


PRODUCER_KINDS = frozenset({"add_card", "summon", "transform"})
AUDIT_CATEGORIES = (
    "entry_behavior_complete",
    "entry_behavior_partial",
    "database_only_no_entry",
    "text_unclear",
    "external_blocker",
)


def _rule_entries(payload: object) -> list[tuple[str, dict]]:
    if isinstance(payload, list):
        return [("rules", entry) for entry in payload if isinstance(entry, dict)]
    if not isinstance(payload, dict):
        return []
    result = []
    for group, entries in payload.items():
        if not isinstance(entries, list):
            continue
        result.extend(
            (group, entry)
            for entry in entries
            if isinstance(entry, dict)
        )
    return result


def _walk_dicts(value: object):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_dicts(nested)


def _load_authored_evidence(
    rules_dir: str,
    overrides: dict,
) -> tuple[set[int], dict[int, list[dict]]]:
    authored_cards: set[int] = set()
    producers: dict[int, list[dict]] = defaultdict(list)
    for file_path in sorted(Path(rules_dir).glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        for group, entry in _rule_entries(payload):
            raw_source_id = entry.get("card_id", entry.get("source_card_id"))
            if raw_source_id is None:
                continue
            source_id = int(raw_source_id)
            authored_cards.add(source_id)
            for operation in _walk_dicts(entry):
                kind = operation.get("kind")
                target_id = operation.get("card_id")
                if kind not in PRODUCER_KINDS or target_id is None:
                    continue
                producers[int(target_id)].append({
                    "source_card_id": source_id,
                    "entry_kind": kind,
                    "rule_file": file_path.name,
                    "rule_group": group,
                })
            if group == "fusions":
                for result in entry.get("transform_results", []):
                    if not isinstance(result, dict) or result.get("card_id") is None:
                        continue
                    producers[int(result["card_id"])].append({
                        "source_card_id": source_id,
                        "entry_kind": "fusion_transform",
                        "rule_file": file_path.name,
                        "rule_group": group,
                    })

    for entry in overrides.get("entries", []):
        target_id = int(entry["card_id"])
        producers[target_id].append({
            "source_card_id": entry.get("source_card_id"),
            "entry_kind": entry["entry_kind"],
            "rule_file": entry.get("rule_file"),
            "rule_group": entry.get("rule_group", "audited_override"),
            "notes": entry.get("notes"),
        })
    for target_id, records in producers.items():
        producers[target_id] = sorted(
            {json.dumps(record, sort_keys=True): record for record in records}.values(),
            key=lambda record: (
                record["source_card_id"] is None,
                record["source_card_id"] or 0,
                record["entry_kind"],
                record["rule_file"] or "",
            ),
        )
    return authored_cards, producers


def _is_keyword_only(skill_texts: list[str]) -> bool:
    if not skill_texts:
        return False
    for text in skill_texts:
        remaining = re.sub(
            r"【\s*<color=Keyword>.*?</color>(?:_[0-9]+)?\s*】",
            "",
            text,
            flags=re.DOTALL,
        )
        remaining = re.sub(r"<hr\s*/?>", "", remaining)
        remaining = re.sub(r"<[^>]+>", "", remaining)
        if remaining.strip():
            return False
    return True


def _build_token_audit(
    db_path: str,
    rules_dir: str,
    overrides_path: str = "data/audits/token_overrides.json",
) -> dict:
    overrides = json.loads(Path(overrides_path).read_text(encoding="utf-8"))
    classification_overrides = {
        int(item["card_id"]): item
        for item in overrides.get("classifications", [])
    }
    for item in classification_overrides.values():
        if item["category"] not in AUDIT_CATEGORIES:
            raise ValueError(
                f"Invalid token audit category {item['category']!r}"
            )
    metadata = _load_rule_metadata(rules_dir)
    authored_cards, authored_producers = _load_authored_evidence(
        rules_dir,
        overrides,
    )
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        token_rows = conn.execute(
            """
            SELECT c.card_id, c.card_set_id, cs.is_collectible,
                   n.name, cl.class_name, ct.name AS card_type,
                   COALESCE(rs.support_level, 'unsupported') AS support_level
            FROM cards c
            JOIN card_sets cs ON cs.id = c.card_set_id
            JOIN card_names n ON n.card_id = c.card_id AND n.language = 'zh-CN'
            JOIN card_localizations cl
              ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
            JOIN card_types ct ON ct.id = c.type_id
            LEFT JOIN rule_support rs ON rs.card_id = c.card_id
            WHERE c.card_set_id = 90000 OR cs.is_collectible = 0
            ORDER BY c.card_id
            """
        ).fetchall()
        token_ids = {row["card_id"] for row in token_rows}
        skill_texts: dict[int, list[str]] = defaultdict(list)
        for row in conn.execute(
            "SELECT card_id, text_chs FROM skill_texts ORDER BY card_id, position"
        ):
            if row["card_id"] in token_ids and row["text_chs"]:
                skill_texts[row["card_id"]].append(row["text_chs"])
        abilities: dict[int, list[dict]] = defaultdict(list)
        for row in conn.execute(
            """
            SELECT ca.card_id, ca.ability_keyword, a.status
            FROM card_abilities ca
            JOIN abilities a ON a.keyword = ca.ability_keyword
            ORDER BY ca.card_id, ca.ability_keyword
            """
        ):
            if row["card_id"] in token_ids:
                abilities[row["card_id"]].append({
                    "keyword": row["ability_keyword"],
                    "status": row["status"],
                })
        database_sources: dict[int, list[dict]] = defaultdict(list)
        for row in conn.execute(
            """
            SELECT r.card_id AS source_card_id, r.referenced_card_id,
                   sn.name AS source_name, r.referenced_name
            FROM card_references r
            LEFT JOIN card_names sn
              ON sn.card_id = r.card_id AND sn.language = 'zh-CN'
            WHERE r.referenced_card_id IS NOT NULL
            ORDER BY r.referenced_card_id, r.card_id, r.position
            """
        ):
            target_id = row["referenced_card_id"]
            if target_id in token_ids:
                database_sources[target_id].append({
                    "source_card_id": row["source_card_id"],
                    "source_name": row["source_name"],
                    "referenced_name": row["referenced_name"],
                })

    cards = []
    counts: Counter[str] = Counter()
    for row in token_rows:
        card_id = row["card_id"]
        texts = skill_texts.get(card_id, [])
        card_abilities = abilities.get(card_id, [])
        producers = authored_producers.get(card_id, [])
        explicit_coverage = metadata.get(card_id, {}).get("coverage")
        keyword_only = _is_keyword_only(texts)
        keywords_complete = bool(card_abilities) and all(
            ability["status"] == "implemented" for ability in card_abilities
        )
        manual = classification_overrides.get(card_id)
        if manual is not None:
            category = manual["category"]
            reason = manual["reason"]
        elif producers:
            behavior_complete = (
                explicit_coverage == "exact"
                or not texts
                or (keyword_only and keywords_complete)
            )
            category = (
                "entry_behavior_complete"
                if behavior_complete
                else "entry_behavior_partial"
            )
            reason = (
                "Executable entry exists and behavior is vanilla, fully implemented "
                "keyword-only, or explicitly audited exact."
                if behavior_complete
                else "Executable entry exists, but nontrivial token text lacks an exact audit."
            )
        else:
            category = "database_only_no_entry"
            reason = (
                "Database source references exist, but no executable producer is authored."
                if database_sources.get(card_id)
                else "Token exists in the database without a resolved executable producer."
            )
        counts[category] += 1
        cards.append({
            "card_id": card_id,
            "name": row["name"],
            "class_name": row["class_name"],
            "card_type": row["card_type"],
            "card_set_id": row["card_set_id"],
            "is_collectible": bool(row["is_collectible"]),
            "support_level": row["support_level"],
            "category": category,
            "reason": reason,
            "database_sources": database_sources.get(card_id, []),
            "authored_producers": producers,
            "has_authored_behavior": card_id in authored_cards,
            "explicit_coverage": explicit_coverage,
            "keyword_only_text": keyword_only,
            "abilities": card_abilities,
            "skill_texts": texts,
        })

    return {
        "generated_from": {
            "database": db_path,
            "rules": rules_dir,
            "overrides": overrides_path,
        },
        "summary": {
            "total": len(cards),
            "categories": {
                category: counts.get(category, 0)
                for category in AUDIT_CATEGORIES
            },
        },
        "cards": cards,
    }


def _render_markdown(report: dict) -> str:
    lines = [
        "# Token / Non-Collectible Audit",
        "",
        f"**Database**: `{report['generated_from']['database']}`  ",
        f"**Rules**: `{report['generated_from']['rules']}`  ",
        f"**Overrides**: `{report['generated_from']['overrides']}`",
        "",
        "## Summary",
        "",
        f"Total audited cards: **{report['summary']['total']}**",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category, count in report["summary"]["categories"].items():
        lines.append(f"| {category} | {count} |")
    lines.extend([
        "",
        "## Cards",
        "",
        "| Card | Class / Type | Category | DB sources | Executable producers | Behavior |",
        "|---|---|---|---:|---:|---|",
    ])
    for card in report["cards"]:
        behavior = (
            "exact"
            if card["explicit_coverage"] == "exact"
            else "authored" if card["has_authored_behavior"] else card["support_level"]
        )
        lines.append(
            f"| {card['card_id']} {card['name']} | {card['class_name']} / "
            f"{card['card_type']} | {card['category']} | "
            f"{len(card['database_sources'])} | {len(card['authored_producers'])} | "
            f"{behavior} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/cards.sqlite3")
    parser.add_argument("--rules", default="data/rules")
    parser.add_argument(
        "--overrides",
        default="data/audits/token_overrides.json",
    )
    parser.add_argument("--output", default="data/reports/token_audit.json")
    parser.add_argument("--markdown", default="data/reports/token_audit.md")
    args = parser.parse_args()
    report = _build_token_audit(args.db, args.rules, args.overrides)
    output = Path(args.output)
    markdown = Path(args.markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown.write_text(_render_markdown(report), encoding="utf-8")
    print(f"JSON report written to {output}")
    print(f"Markdown report written to {markdown}")


if __name__ == "__main__":
    main()
