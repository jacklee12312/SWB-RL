# -*- coding: utf-8 -*-
"""Coverage report tool: maps DB cards to existing rules and primitive support."""

from __future__ import annotations

import argparse
import hashlib
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
    ("连击", {"primitive": "COMBO condition / expression / add_combo", "covered": True}),
    ("觉醒", {"primitive": "OVERFLOW condition / expression", "covered": True}),
    (
        "策动",
        {
            "primitive": "ActivateAmulet command / ACTIVATE trigger",
            "covered": True,
        },
    ),
    (
        "威慑",
        {
            "primitive": "INTIMIDATE attack-target legality",
            "covered": True,
        },
    ),
    (
        "灵气",
        {
            "primitive": "AURA manual enemy-effect target legality",
            "covered": True,
        },
    ),
    (
        "瞬念召唤",
        {
            "primitive": "Invocation deck scan / INVOKE trigger",
            "covered": True,
        },
    ),
    (
        "奥义",
        {
            "primitive": "Union Burst hand gauge / threshold operations",
            "covered": True,
        },
    ),
    ("回合开始", {"primitive": "TURN_START trigger / Emblem", "covered": True}),
    ("回合结束", {"primitive": "TURN_END trigger / Emblem", "covered": True}),
    ("倒数", {"primitive": "COUNTDOWN / countdown", "covered": True}),
    ("抽取", {"primitive": "DRAW / DRAW_FILTERED", "covered": True}),
    ("将.*加入手牌", {"primitive": "ADD_CARD", "covered": True}),
    (
        "回复自己\\d+点超进化点",
        {"primitive": "RESTORE_SUPER_EVOLUTION_POINTS", "covered": True},
    ),
    (
        "回复自己\\d+点进化点",
        {"primitive": "RESTORE_EVOLUTION_POINTS", "covered": True},
    ),
    (
        "回复自己\\d+点能量点",
        {"primitive": "RESTORE_MANA", "covered": True},
    ),
    (
        "回复(?!自己\\d+点(?:超进化点|进化点|能量点))",
        {"primitive": "HEAL_LEADER / HEAL_UNIT", "covered": True},
    ),
    ("造成.*伤害", {"primitive": "DAMAGE_LEADER / DAMAGE_UNIT", "covered": True}),
    ("失去所有能力", {"primitive": "REMOVE_ALL_ABILITIES", "covered": True}),
    (
        "受到的伤害[+＋]",
        {"primitive": "ADD_LEADER_DAMAGE_MODIFIER", "covered": True},
    ),
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
    (
        "土之秘术|土之印",
        {
            "primitive": "EARTH_RITE / ADD_EARTH_SIGILS / Earth Sigil board state",
            "covered": True,
        },
    ),
    (
        "融合",
        {
            "primitive": "BeginFusion command / Fusion material state",
            "covered": True,
        },
    ),
    (
        "信仰",
        {
            "primitive": "Faith leader-area state / evolution trigger",
            "covered": True,
        },
    ),
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

BLOCKER_TYPES = (
    "missing_rule",
    "missing_schema",
    "missing_primitive",
    "missing_targeting",
    "timing_unclear",
    "text_unclear",
    "external_blocker",
    "audit_unverified",
)

CLAUSE_AUDIT_STATUSES = (
    "mapped_exact",
    "unverified_exact",
    "partial",
    "missing_rule",
    "missing_primitive",
    "text_unclear",
    "token_separate_audit",
)


def _classify_card(
    card: CardDefinition,
    ruled_cards: set[int],
    ruled_ops: dict[int, dict],
    rule_metadata: dict[int, dict],
    ability_map: dict[int, list[str]],
    skill_text_map: dict[int, list[str]],
    support_map: dict[int, str],
    activation_cards: set[int] | None = None,
    faith_cards: set[int] | None = None,
    union_burst_cards: set[int] | None = None,
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

    missing_rule_mechanics = []
    if (
        re.search("策动", search_text) is not None
        and card_id not in (activation_cards or set())
    ):
        missing_rule_mechanics.append("策动")
    if (
        re.search("信仰", search_text) is not None
        and card_id not in (faith_cards or set())
    ):
        missing_rule_mechanics.append("信仰")
    if (
        re.search("奥义", search_text) is not None
        and card_id not in (union_burst_cards or set())
    ):
        missing_rule_mechanics.append("奥义")

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
            result["coverage"] = (
                "covered_partial"
                if missing_keywords or missing_rule_mechanics
                else "covered_exact"
            )
            result["reason"] = f"Triggers: {triggers}, Ops: {ops}"
            if missing_rule_mechanics:
                result["reason"] += (
                    "; missing structured rules: "
                    f"{missing_rule_mechanics}"
                )
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
    if missing_rule_mechanics:
        result["missing_rule_mechanics"] = missing_rule_mechanics
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
    source_snapshot: dict[str, object] = {}
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT card_id FROM cards"):
                cid = row["card_id"]
                try:
                    all_cards[cid] = repo.get(cid)
                except Exception:
                    continue
            source_row = conn.execute(
                "SELECT source_url, fetched_at, sha256, card_count "
                "FROM source_imports ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if source_row is not None:
                source_snapshot = dict(source_row)
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
    for cid in rulebook._fusion_defs:
        ruled_cards.add(cid)
        ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
        ruled_ops[cid]["triggers"].append("fusion")
        ruled_ops[cid]["effect_kinds"].append("fusion")
    for cid in rulebook._invocation_defs:
        ruled_cards.add(cid)
        ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
        ruled_ops[cid]["triggers"].append("invocation")
        ruled_ops[cid]["effect_kinds"].append("invocation")
    for cid in rulebook._activation_defs:
        ruled_cards.add(cid)
        ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
        ruled_ops[cid]["triggers"].append("activation")
        ruled_ops[cid]["effect_kinds"].append("activation")
    for cid in rulebook._faith_defs:
        ruled_cards.add(cid)
        ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
        ruled_ops[cid]["triggers"].append("faith")
        ruled_ops[cid]["effect_kinds"].append("faith_value")
    for cid, definitions in rulebook._union_burst_defs.items():
        ruled_cards.add(cid)
        ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
        for definition in definitions:
            ruled_ops[cid]["triggers"].append(definition.kind.value)
            for op in _iter_nested_operations(definition.operations):
                ruled_ops[cid]["effect_kinds"].append(op.kind.value)
    for cid, definitions in rulebook._listener_defs.items():
        ruled_cards.add(cid)
        ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
        for definition in definitions:
            ruled_ops[cid]["triggers"].append(
                f"listener:{definition.zone.value}:{definition.event.value}"
            )
            for op in _iter_nested_operations(definition.operations):
                ruled_ops[cid]["effect_kinds"].append(op.kind.value)
    for cid, passives in rulebook._passives.items():
        for passive in passives:
            if passive.kind == "non_intrinsic_keyword":
                continue
            ruled_cards.add(cid)
            ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
            if "passive" not in ruled_ops[cid]["triggers"]:
                ruled_ops[cid]["triggers"].append("passive")
            ruled_ops[cid]["effect_kinds"].append(passive.kind)
    for cid, keywords in rulebook._intrinsic_keyword_defs.items():
        ruled_cards.add(cid)
        ruled_ops.setdefault(cid, {"triggers": [], "effect_kinds": []})
        ruled_ops[cid]["triggers"].append("intrinsic_keywords")
        ruled_ops[cid]["effect_kinds"].extend(
            f"keyword:{keyword}" for keyword in keywords
        )
    for emblem_id, ed in rulebook._emblem_defs.items():
        ruled_cards.add(ed.source_card_id)
        ruled_ops.setdefault(ed.source_card_id, {"triggers": [], "effect_kinds": []})
        ruled_ops[ed.source_card_id]["triggers"].append("emblem_source")
        for op in _iter_nested_operations(ed.on_gain):
            ruled_ops[ed.source_card_id]["effect_kinds"].append(op.kind.value)
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
            skill_text_map = _load_source_text_map(conn)
    except Exception:
        pass

    _validate_rule_metadata_source_hashes(rule_metadata, skill_text_map)

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
            activation_cards=(
                set(rulebook._activation_defs)
                | {
                    card_id
                    for card_id, definitions in rulebook._listener_defs.items()
                    if any(
                        definition.event.value == "amulet_activated"
                        for definition in definitions
                    )
                }
            ),
            faith_cards=set(rulebook._faith_defs),
            union_burst_cards=set(rulebook._union_burst_defs),
        )

    test_sources = {
        path.as_posix(): path.read_text(encoding="utf-8", errors="ignore")
        for path in sorted(Path("tests").glob("test_*.py"))
    }
    clause_counts: dict[str, int] = {
        status: 0 for status in CLAUSE_AUDIT_STATUSES
    }
    blocker_counts: dict[str, int] = {kind: 0 for kind in BLOCKER_TYPES}
    clause_issues: list[dict] = []
    for cid_text, classification in classifications.items():
        cid = int(cid_text)
        metadata = rule_metadata.get(cid, {})
        discovered_evidence = [
            path for path, source in test_sources.items()
            if cid_text in source
        ]
        configured_evidence = metadata.get("test_evidence")
        if configured_evidence is None:
            evidence = discovered_evidence
        else:
            evidence = [
                path
                for path in configured_evidence
                if path in test_sources and cid_text in test_sources[path]
            ]
            if len(evidence) != len(configured_evidence):
                metadata["audit_validation_error"] = (
                    "configured test_evidence is missing or does not reference "
                    f"card {cid}"
                )
        coverage = classification["coverage"]
        if coverage == "token_or_non_collectible":
            audit_status = "token_separate_audit"
        elif coverage == "covered_exact":
            if (
                metadata.get("coverage") == "exact"
                and metadata.get("implemented_text")
                and evidence
                and not metadata.get("audit_validation_error")
            ):
                audit_status = "mapped_exact"
            else:
                audit_status = "unverified_exact"
                missing = []
                if metadata.get("coverage") != "exact":
                    missing.append("explicit_exact_annotation")
                if not metadata.get("implemented_text"):
                    missing.append("implemented_text")
                if not evidence:
                    missing.append("test_evidence")
                if metadata.get("audit_validation_error"):
                    missing.append("audit_validation")
                clause_issues.append({
                    "card_id": cid,
                    "issue": "covered_exact_without_clause_evidence",
                    "missing": missing,
                })
        elif coverage == "covered_partial":
            audit_status = "partial"
        elif coverage == "supported_missing_rule":
            audit_status = "missing_rule"
        elif coverage == "missing_primitive":
            audit_status = "missing_primitive"
        else:
            audit_status = "text_unclear"
        clause_counts[audit_status] += 1
        classification["clause_audit"] = {
            "status": audit_status,
            "rule_version": metadata.get("rule_version", 1),
            "errata": metadata.get("errata", []),
            "source_clauses": [
                {
                    "clause_id": f"{cid}:{index}",
                    "source_text": text,
                    "mapping_status": (
                        "implemented"
                        if audit_status == "mapped_exact"
                        else audit_status
                    ),
                }
                for index, text in enumerate(classification.get("skill_texts", []))
            ],
            "implemented_text": metadata.get("implemented_text"),
            "unsupported_text": metadata.get("unsupported_text"),
            "source_text_sha256": metadata.get("source_text_sha256"),
            "structured_evidence": ruled_ops.get(cid, {"triggers": [], "effect_kinds": []}),
            "test_evidence": evidence,
            "audit_validation_error": metadata.get("audit_validation_error"),
            "blocker_type": metadata.get(
                "blocker_type",
                (
                    "audit_unverified"
                    if audit_status == "unverified_exact"
                    else {
                    "covered_partial": "missing_schema",
                    "supported_missing_rule": "missing_rule",
                    "missing_primitive": "missing_primitive",
                    "text_unclear": "text_unclear",
                    }.get(coverage)
                ),
            ),
        }
        blocker = classification["clause_audit"]["blocker_type"]
        if blocker is not None:
            if blocker not in blocker_counts:
                raise ValueError(
                    f"card {cid}: unsupported blocker_type {blocker!r}"
                )
            blocker_counts[blocker] += 1

    total = len(classifications)
    counts = {}
    for v in classifications.values():
        cat = v["coverage"]
        counts[cat] = counts.get(cat, 0) + 1

    test_ids = sum(1 for cid in ruled_cards if 999000 <= int(cid) <= 999999)

    rule_issues = [
        {
            "card_id": cid,
            "issue": "clause_audit_validation_failed",
            "detail": metadata["audit_validation_error"],
        }
        for cid, metadata in sorted(rule_metadata.items())
        if metadata.get("audit_validation_error")
    ]
    unknown_rules = sorted(
        cid for cid in ruled_cards - set(all_cards)
        if not 999000 <= cid <= 999999
    )
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
            "clause_audit_registry": str(
                Path(rules_dir).parent / "audits" / "rule_clauses.json"
            ),
            "source_snapshot": source_snapshot,
        }),
        ("summary", OrderedDict([
            ("total_cards", total),
            ("total_with_rules", len(ruled_cards)),
            ("test_or_synthetic_ids_with_rules", test_ids),
            ("coverage_counts", counts),
            ("clause_audit_counts", clause_counts),
            ("blocker_counts", blocker_counts),
        ])),
        ("rule_consistency_issues", rule_issues),
        ("clause_audit_issues", clause_issues),
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
        if isinstance(payload, list):
            entries = payload
        else:
            entries = []
            for key in (
                "rules",
                "activations",
                "fusions",
                "invocations",
                "faiths",
                "union_bursts",
                "listeners",
            ):
                raw_entries = payload.get(key, [])
                if isinstance(raw_entries, list):
                    entries.extend(raw_entries)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_card_id = entry.get("card_id", entry.get("source_card_id"))
            if raw_card_id is None:
                continue
            cid = int(raw_card_id)
            item = {
                key: entry[key]
                for key in (
                    "coverage",
                    "implemented_text",
                    "unsupported_text",
                    "notes",
                    "rule_version",
                    "errata",
                    "blocker_type",
                )
                if key in entry
            }
            if item:
                metadata.setdefault(cid, {}).update(item)
    audit_path = path.parent / "audits" / "rule_clauses.json"
    if audit_path.exists():
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        entries = payload.get("cards", []) if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError(f"{audit_path}: cards must be a list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{audit_path}/cards[{index}]: must be an object")
            if "card_id" not in entry:
                raise ValueError(f"{audit_path}/cards[{index}]: card_id is required")
            cid = int(entry["card_id"])
            item = {
                key: entry[key]
                for key in (
                    "coverage",
                    "implemented_text",
                    "unsupported_text",
                    "notes",
                    "rule_version",
                    "errata",
                    "blocker_type",
                    "source_text_sha256",
                    "test_evidence",
                )
                if key in entry
            }
            duplicate_keys = sorted(set(metadata.get(cid, {})) & set(item))
            if duplicate_keys:
                raise ValueError(
                    f"{audit_path}/cards[{index}]: card {cid} duplicates rule "
                    f"metadata keys {duplicate_keys}"
                )
            if item:
                metadata.setdefault(cid, {}).update(item)
    return metadata


def _source_text_sha256(texts: list[str]) -> str:
    payload = json.dumps(
        texts,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_source_text_map(conn: sqlite3.Connection) -> dict[int, list[str]]:
    """Load every rules-bearing text clause, including alternate modes."""
    source_text_map: dict[int, list[str]] = {}
    text_rows = conn.execute(
        "SELECT card_id, text_chs, text FROM skill_texts "
        "ORDER BY card_id, position"
    ).fetchall()
    for cid, text_chs, text in text_rows:
        text_value = text_chs or text or ""
        if text_value:
            source_text_map.setdefault(cid, []).append(text_value)

    mode_rows = conn.execute(
        "SELECT card_id, mode_type, cost, text_chs FROM alt_modes "
        "ORDER BY card_id, position"
    ).fetchall()
    for cid, mode_type, cost, text_chs in mode_rows:
        mode_label = mode_type if cost is None else f"{mode_type}_{cost}"
        text_value = f"【{mode_label}】{text_chs}" if text_chs else f"【{mode_label}】"
        clauses = source_text_map.setdefault(cid, [])
        if text_value not in clauses:
            clauses.append(text_value)
    return source_text_map


def _validate_rule_metadata_source_hashes(
    rule_metadata: dict[int, dict],
    source_text_map: dict[int, list[str]],
) -> None:
    """Invalidate audit entries when imported source text changes."""
    for cid, metadata in rule_metadata.items():
        expected_hash = metadata.get("source_text_sha256")
        if expected_hash is None:
            continue
        actual_hash = _source_text_sha256(source_text_map.get(cid, []))
        if actual_hash != expected_hash:
            metadata["audit_validation_error"] = (
                f"source_text_sha256 mismatch: expected {expected_hash}, "
                f"got {actual_hash}"
            )


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

    lines.extend([
        "",
        "### Clause Audit",
        "",
        "| Clause status | Count |",
        "|---|---:|",
    ])
    for status, count in s["clause_audit_counts"].items():
        lines.append(f"| {status} | {count} |")

    lines.extend([
        "",
        "### Blocker Types",
        "",
        "| Blocker | Count |",
        "|---|---:|",
    ])
    for blocker, count in s["blocker_counts"].items():
        lines.append(f"| {blocker} | {count} |")

    if report["clause_audit_issues"]:
        lines.extend([
            "",
            "## Exact-Coverage Clause Audit Issues",
            "",
        ])
        for issue in report["clause_audit_issues"]:
            lines.append(
                f"- **{issue['card_id']}**: {issue['issue']} — missing "
                f"{', '.join(issue['missing'])}"
            )

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
