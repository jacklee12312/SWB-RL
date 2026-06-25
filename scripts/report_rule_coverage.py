# -*- coding: utf-8 -*-
"""Coverage report tool: maps DB cards to existing rules and primitive support."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import OrderedDict
from contextlib import closing
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, _iter_nested_operations


# ---------------------------------------------------------------------------
# Keywords → primitive mapping (ordered for deterministic output)
# ---------------------------------------------------------------------------

PRIMITIVE_KEYWORD_MAP = OrderedDict([
    ("入场曲", {"primitive": "FANFARE trigger", "covered": True}),
    ("谢幕曲", {"primitive": "LAST_WORDS trigger", "covered": True}),
    ("进化时", {"primitive": "EVOLVE trigger", "covered": True}),
    ("超进化", {"primitive": "SUPER_EVOLVE trigger", "covered": True}),
    ("攻击时", {"primitive": "ATTACK trigger", "covered": True}),
    ("交战时", {"primitive": "CLASH trigger", "covered": True}),
    ("连击", {"primitive": "COMBO (placeholder)", "covered": False}),
    ("觉醒", {"primitive": "OVERFLOW (placeholder)", "covered": False}),
    ("策动", {"primitive": "ACTIVATE (placeholder)", "covered": False}),
    ("威慑", {"primitive": "INTIMIDATE (placeholder)", "covered": False}),
    ("灵气", {"primitive": "AURA (placeholder)", "covered": False}),
    ("瞬念召唤", {"primitive": "INVOCATION (placeholder)", "covered": False}),
    ("奥义", {"primitive": "UNION_BURST (placeholder)", "covered": False}),
    ("回合开始", {"primitive": "TURN_START trigger / Emblem", "covered": True}),
    ("回合结束", {"primitive": "TURN_END trigger / Emblem", "covered": True}),
    ("倒数", {"primitive": "COUNTDOWN / countdown", "covered": True}),
    ("抽取", {"primitive": "DRAW / DRAW_FILTERED", "covered": True}),
    ("将.*加入手牌", {"primitive": "ADD_CARD", "covered": True}),
    ("回复", {"primitive": "HEAL_LEADER", "covered": True}),
    ("造成.*伤害", {"primitive": "DAMAGE_LEADER / DAMAGE_UNIT", "covered": True}),
    ("破坏", {"primitive": "DESTROY", "covered": True}),
    ("消失", {"primitive": "BANISH", "covered": True}),
    ("召唤", {"primitive": "SUMMON", "covered": True}),
    ("返回手牌", {"primitive": "RETURN_TO_HAND", "covered": True}),
    ("返回牌", {"primitive": "RETURN_TO_DECK", "covered": True}),
    ("亡者召还", {"primitive": "REANIMATE", "covered": True}),
    ("舍弃", {"primitive": "DISCARD", "covered": True}),
    ("死灵术|唤灵", {"primitive": "NECROMANCY", "covered": True}),
    ("魔力增幅", {"primitive": "SPELLBOOST_HAND / passive", "covered": True}),
    ("无法使用", {"primitive": "cannot_be_played passive", "covered": True}),
    ("协作", {"primitive": "COOPERATION value / conditions", "covered": True}),
    ("纹章", {"primitive": "GAIN_EMBLEM / EMBLEM system", "covered": True}),
    ("土之秘术|土之印", {"primitive": "EARTH_RITE (placeholder)", "covered": False}),
    ("融合", {"primitive": "FUSION (placeholder)", "covered": False}),
    ("信仰", {"primitive": "FAITH (placeholder)", "covered": False}),
    ("必杀", {"primitive": "BANE keyword", "covered": True}),
    ("吸血", {"primitive": "DRAIN keyword", "covered": True}),
    ("屏障", {"primitive": "BARRIER keyword", "covered": True}),
    ("不能攻击|无法攻击", {"primitive": "ADD_ATTACK_RESTRICTION", "covered": True}),
    ("不能被指定|无法被能力指定", {"primitive": "ADD_TARGETING_RESTRICTION", "covered": True}),
    ("变形", {"primitive": "TRANSFORM", "covered": True}),
    ("爆能强化", {"primitive": "ENHANCE play mode", "covered": True}),
    ("激奏", {"primitive": "ACCELERATE play mode", "covered": True}),
    ("结晶", {"primitive": "CRYSTALLIZE play mode", "covered": True}),
    ("选择一项|模式", {"primitive": "CHOOSE_ONE / OPTIONAL", "covered": True}),
])


def _classify_card(
    card: CardDefinition,
    ruled_cards: set[int],
    ruled_ops: dict[int, dict],
    rule_metadata: dict[int, dict],
    ability_map: dict[int, list[str]],
    skill_text_map: dict[int, list[str]],
    support_map: dict[int, str],
) -> dict:
    """Classify a single card's coverage status."""
    card_id = card.card_id
    result = OrderedDict()
    result["card_id"] = card_id
    result["name"] = card.name
    result["class_name"] = card.class_name
    result["card_type"] = card.card_type
    result["cost"] = card.cost
    result["is_collectible"] = card.is_collectible
    result["card_set_id"] = card.card_set_id

    if not card.is_collectible:
        result["coverage"] = "token_or_non_collectible"
        result["reason"] = f"card_set_id={card.card_set_id}, is_collectible=False"
        return result

    if card.card_set_id == 90000:
        result["coverage"] = "token_or_non_collectible"
        result["reason"] = "card_set_id=90000 (token)"
        return result

    is_test_id = 999000 <= card_id <= 999999
    if is_test_id:
        result["coverage"] = "test_only_rule" if card_id in ruled_cards else "token_or_non_collectible"
        result["reason"] = "synthetic TEST ID"
        return result

    support = support_map.get(card_id, "unsupported")
    abilities = ability_map.get(card_id, [])
    skill_texts = skill_text_map.get(card_id, [])
    search_text = "\n".join([*abilities, *skill_texts])

    hit_keywords = []
    missing_keywords = []
    for pattern, info in PRIMITIVE_KEYWORD_MAP.items():
        found = re.search(pattern, search_text) is not None
        if found:
            if info["covered"]:
                hit_keywords.append(pattern)
            else:
                missing_keywords.append(pattern)

    has_rule = card_id in ruled_cards
    metadata = rule_metadata.get(card_id, {})
    explicit_coverage = metadata.get("coverage")

    if has_rule:
        rule_info = ruled_ops[card_id]
        triggers = rule_info.get("triggers", [])
        ops = rule_info.get("effect_kinds", [])
        if explicit_coverage == "partial":
            result["coverage"] = "covered_partial"
            unsupported = metadata.get("unsupported_text")
            result["reason"] = (
                f"Partial rule. Triggers: {triggers}, Ops: {ops}"
                + (f"; unsupported: {unsupported}" if unsupported else "")
            )
        else:
            result["coverage"] = "covered_partial" if missing_keywords else "covered_exact"
            result["reason"] = f"Triggers: {triggers}, Ops: {ops}"
    else:
        if missing_keywords:
            result["coverage"] = "missing_primitive"
            result["reason"] = f"Missing primitives: {missing_keywords}"
        elif hit_keywords:
            result["coverage"] = "supported_missing_rule"
            result["reason"] = f"Covered keywords: {hit_keywords}"
        elif support == "unsupported":
            result["coverage"] = "text_unclear"
            result["reason"] = "No ability keywords in database"
        else:
            result["coverage"] = "text_unclear"
            result["reason"] = f"Support: {support}"

    result["hit_keywords"] = hit_keywords
    result["missing_primitives"] = missing_keywords
    result["ability_keywords"] = abilities
    result["skill_texts"] = skill_texts
    result["support_level"] = support
    if metadata:
        result["rule_metadata"] = metadata
    return result


def _build_coverage_report(db_path: str, rules_dir: str) -> dict:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    repo = CardRepository(db_path)
    rulebook = RuleBook.from_directory(rules_dir)

    all_cards: dict[int, CardDefinition] = {}
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT card_id FROM cards"):
                cid = row["card_id"]
                try:
                    all_cards[cid] = repo.get(cid)
                except Exception:
                    continue
    except Exception as e:
        raise RuntimeError(f"Failed to read database: {e}")

    ruled_cards: set[int] = set()
    ruled_ops: dict[int, dict] = {}
    rule_metadata = _load_rule_metadata(rules_dir)
    for (cid, trigger), ops in rulebook._rules.items():
        ruled_cards.add(cid)
        ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
        ruled_ops[cid]["triggers"].append(trigger.value)
        for op in _iter_nested_operations(ops):
            ruled_ops[cid]["effect_kinds"].append(op.kind.value)
    for cid in rulebook._play_modes:
        ruled_cards.add(cid)
        ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
        ruled_ops[cid]["triggers"].append("play_modes")
        ruled_ops[cid]["effect_kinds"].append("play_mode")
        for mode in rulebook._play_modes[cid]:
            for op in _iter_nested_operations(mode.operations):
                ruled_ops[cid]["effect_kinds"].append(op.kind.value)
    for cid, passives in rulebook._passives.items():
        ruled_cards.add(cid)
        ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
        ruled_ops[cid]["triggers"].append("passive")
        for passive in passives:
            ruled_ops[cid]["effect_kinds"].append(passive.kind)
    for emblem_id, ed in rulebook._emblem_defs.items():
        ruled_cards.add(ed.source_card_id)
        ruled_ops.setdefault(ed.source_card_id, {"triggers": [], "effect_kinds": []})
        ruled_ops[ed.source_card_id]["triggers"].append("emblem_source")
        for tr in ed.triggers:
            for op in _iter_nested_operations(tr.operations):
                ruled_ops[ed.source_card_id]["effect_kinds"].append(op.kind.value)
        for op in _iter_nested_operations(ed.on_expire):
            ruled_ops[ed.source_card_id]["effect_kinds"].append(op.kind.value)

    ability_map: dict[int, list[str]] = {}
    skill_text_map: dict[int, list[str]] = {}
    support_map: dict[int, str] = {}
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute(
                "SELECT ca.card_id, ca.ability_keyword, rs.support_level "
                "FROM card_abilities ca LEFT JOIN rule_support rs ON ca.card_id=rs.card_id "
                "ORDER BY ca.card_id, ca.ability_keyword"
            ).fetchall()
            for cid, ab, sl in rows:
                ability_map.setdefault(cid, []).append(ab)
                if cid not in support_map:
                    support_map[cid] = sl or "unsupported"
            text_rows = conn.execute(
                "SELECT card_id, text_chs, text FROM skill_texts "
                "ORDER BY card_id, position"
            ).fetchall()
            for cid, text_chs, text in text_rows:
                text_value = text_chs or text or ""
                if text_value:
                    skill_text_map.setdefault(cid, []).append(text_value)
    except Exception:
        pass

    classifications = OrderedDict()
    for cid in sorted(all_cards):
        classifications[str(cid)] = _classify_card(
            all_cards[cid],
            ruled_cards,
            ruled_ops,
            rule_metadata,
            ability_map,
            skill_text_map,
            support_map,
        )

    total = len(classifications)
    counts = {}
    for v in classifications.values():
        cat = v["coverage"]
        counts[cat] = counts.get(cat, 0) + 1

    test_ids = sum(1 for cid in ruled_cards if 999000 <= int(cid) <= 999999)

    rule_issues = []
    unknown_rules = sorted(ruled_cards - set(all_cards))
    for cid in unknown_rules:
        rule_issues.append({
            "card_id": cid,
            "issue": "card_id_not_in_database",
            "detail": f"Card {cid} has rules but is not in the database",
        })

    recommendations = _generate_recommendations(classifications, ability_map, ruled_cards)

    return OrderedDict([
        ("generated_from", {
            "database": db_path,
            "rules_directory": rules_dir,
        }),
        ("summary", OrderedDict([
            ("total_cards", total),
            ("total_with_rules", len(ruled_cards)),
            ("test_or_synthetic_ids_with_rules", test_ids),
            ("coverage_counts", counts),
        ])),
        ("rule_consistency_issues", rule_issues),
        ("primitive_keyword_map", OrderedDict([
            (kw, info) for kw, info in PRIMITIVE_KEYWORD_MAP.items()
        ])),
        ("classifications", classifications),
        ("top_20_recommendations", recommendations),
    ])


def _load_rule_metadata(rules_dir: str) -> dict[int, dict]:
    """Read optional coverage annotations from rule JSON files."""
    metadata: dict[int, dict] = {}
    path = Path(rules_dir)
    if not path.exists():
        return metadata
    for file_path in sorted(path.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else payload.get("rules", [])
        for entry in entries:
            if not isinstance(entry, dict) or "card_id" not in entry:
                continue
            cid = int(entry["card_id"])
            item = {
                key: entry[key]
                for key in ("coverage", "implemented_text", "unsupported_text", "notes")
                if key in entry
            }
            if item:
                metadata.setdefault(cid, {}).update(item)
    return metadata


def _generate_recommendations(
    classifications: OrderedDict,
    ability_map: dict[int, list[str]],
    ruled_cards: set[int],
) -> list[dict]:
    candidates = []
    for cid_str, info in classifications.items():
        cid = int(cid_str)
        if info["coverage"] == "supported_missing_rule" and info["is_collectible"]:
            keywords = ability_map.get(cid, [])
            candidates.append(OrderedDict([
                ("card_id", cid),
                ("name", info["name"]),
                ("class_name", info["class_name"]),
                ("card_type", info["card_type"]),
                ("cost", info["cost"]),
                ("ability_keywords", keywords),
                ("hit_keywords", info["hit_keywords"]),
                ("confidence", "high" if len(keywords) <= 3 else "medium"),
                ("why_recommended", f"Covered keywords: {', '.join(info['hit_keywords'][:3])}"),
                ("required_primitives", info["hit_keywords"][:5]),
                ("suggested_rule_file", f"batch_{info['class_name']}.json"),
            ]))
    candidates.sort(key=lambda x: (
        len(x["ability_keywords"]),
        x["cost"],
    ))
    return candidates[:20]


def write_json_report(report: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON report written to {output_path}")


def write_markdown_report(report: dict, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    s = report["summary"]
    lines = [
        "# Rule Coverage Report",
        "",
        f"**Database**: `{report['generated_from']['database']}`",
        f"**Rules**: `{report['generated_from']['rules_directory']}`",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|---|---|",
        f"| Total cards in DB | {s['total_cards']} |",
        f"| Cards with rules | {s['total_with_rules']} |",
        f"| Test/synthetic IDs with rules | {s['test_or_synthetic_ids_with_rules']} |",
        "",
        "### Coverage Categories",
        "",
        "| Category | Count |",
        "|---|---|",
    ]
    for cat, cnt in s["coverage_counts"].items():
        lines.append(f"| {cat} | {cnt} |")

    if report["rule_consistency_issues"]:
        lines.append("")
        lines.append("## Rule Consistency Issues")
        lines.append("")
        for issue in report["rule_consistency_issues"]:
            lines.append(f"- **{issue['card_id']}**: {issue['issue']} — {issue['detail']}")

    lines.append("")
    lines.append("## Primitive Keyword Map")
    lines.append("")
    lines.append("| Keyword | Primitive | Covered |")
    lines.append("|---|---|---|")
    for kw, info in report["primitive_keyword_map"].items():
        lines.append(f"| {kw} | {info['primitive']} | {info['covered']} |")

    recs = report["top_20_recommendations"]
    if recs:
        lines.append("")
        lines.append("## Top 20 Recommended Cards")
        lines.append("")
        lines.append("| # | Card ID | Name | Class | Cost | Type | Confidence | Why |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r['card_id']} | {r['name']} | {r['class_name']} "
                f"| {r['cost']} | {r['card_type']} | {r['confidence']} "
                f"| {r['why_recommended'][:60]} |"
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Markdown report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SWB Rule Coverage Report")
    parser.add_argument("--db", default="data/cards.sqlite3", help="Path to cards SQLite database")
    parser.add_argument("--rules", default="data/rules", help="Path to rules directory")
    parser.add_argument("--output", help="JSON output path")
    parser.add_argument("--markdown", help="Markdown output path")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        print("Run with --db <path> to specify the SQLite database", file=sys.stderr)
        sys.exit(1)

    report = _build_coverage_report(args.db, args.rules)

    if args.output:
        write_json_report(report, args.output)
    if args.markdown:
        write_markdown_report(report, args.markdown)

    if not args.output and not args.markdown:
        print(json.dumps({"summary": report["summary"]}, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
