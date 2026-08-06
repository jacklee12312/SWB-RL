from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from swb.engine.abilities import ABILITY_DEFINITIONS, AbilityKeyword
from scripts.report_rule_coverage import PRIMITIVE_KEYWORD_MAP


_MANUAL_PRIMITIVE_STATUS = {
    AbilityKeyword.STORM: "covered",
    AbilityKeyword.RUSH: "covered",
    AbilityKeyword.WARD: "covered",
    AbilityKeyword.AMBUSH: "covered",
    AbilityKeyword.INTIMIDATE: "covered",
    AbilityKeyword.AURA: "covered",
    AbilityKeyword.ACTIVATE: "covered",
    AbilityKeyword.UNION_BURST: "covered",
}

_PRIMITIVE_PATTERN_BY_KEYWORD = {
    AbilityKeyword.COMBO: "连击",
    AbilityKeyword.COOPERATION: "协作",
    AbilityKeyword.OVERFLOW: "觉醒",
    AbilityKeyword.SPELLBOOST: "魔力增幅",
    AbilityKeyword.EARTH_RITE: "土之秘术|土之印",
    AbilityKeyword.EARTH_SIGIL: "土之秘术|土之印",
    AbilityKeyword.NECROMANCY: "死灵术|唤灵",
    AbilityKeyword.REANIMATE: "亡者召还",
    AbilityKeyword.STORM: "疾驰",
    AbilityKeyword.RUSH: "突进",
    AbilityKeyword.WARD: "守护",
    AbilityKeyword.BANE: "必杀",
    AbilityKeyword.AMBUSH: "潜行",
    AbilityKeyword.DRAIN: "吸血",
    AbilityKeyword.COUNTDOWN: "倒数",
    AbilityKeyword.BARRIER: "屏障",
    AbilityKeyword.FANFARE: "入场曲",
    AbilityKeyword.LAST_WORDS: "谢幕曲",
    AbilityKeyword.ON_EVOLVE: "进化时",
    AbilityKeyword.ON_SUPER_EVOLVE: "超进化",
    AbilityKeyword.ON_ATTACK: "攻击时",
    AbilityKeyword.ON_CLASH: "交战时",
    AbilityKeyword.ENHANCE: "爆能强化",
    AbilityKeyword.ACCELERATE: "激奏",
    AbilityKeyword.CRYSTALLIZE: "结晶",
    AbilityKeyword.CHOOSE: "选择一项|模式",
    AbilityKeyword.FUSION: "融合",
    AbilityKeyword.INVOCATION: "瞬念召唤",
    AbilityKeyword.EMBLEM: "纹章",
    AbilityKeyword.FAITH: "信仰",
}


def primitive_status(keyword: AbilityKeyword) -> str:
    manual = _MANUAL_PRIMITIVE_STATUS.get(keyword)
    if manual is not None:
        return manual
    pattern = _PRIMITIVE_PATTERN_BY_KEYWORD.get(keyword)
    if pattern is None:
        return "unmapped"
    info = PRIMITIVE_KEYWORD_MAP.get(pattern)
    if info is None:
        return "unmapped"
    return "covered" if info["covered"] else "missing"


def build_ability_audit(
    audit_path: str = "data/audits/ability_registry.json",
) -> dict:
    payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    audited = {entry["keyword"]: entry for entry in payload.get("abilities", [])}
    expected = {definition.keyword.value for definition in ABILITY_DEFINITIONS}
    missing = sorted(expected - set(audited))
    extra = sorted(set(audited) - expected)
    if missing or extra:
        raise ValueError(
            f"Ability audit mismatch: missing={missing}, extra={extra}"
        )
    rows = []
    counts: Counter[str] = Counter()
    for definition in ABILITY_DEFINITIONS:
        item = audited[definition.keyword.value]
        if item.get("status") != definition.status.value:
            raise ValueError(
                f"Ability {definition.keyword.value!r} audit status "
                f"{item.get('status')!r} != registry {definition.status.value!r}"
            )
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError(
                f"Ability {definition.keyword.value!r} requires a non-empty reason"
            )
        evidence = item.get("test_evidence", [])
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(
                f"Ability {definition.keyword.value!r} requires test_evidence"
            )
        counts[definition.status.value] += 1
        rows.append({
            "keyword": definition.keyword.value,
            "status": definition.status.value,
            "handler_name": definition.handler_name,
            "primitive_status": primitive_status(definition.keyword),
            "events": [event.value for event in sorted(definition.events, key=lambda e: e.value)],
            "aliases": list(definition.aliases),
            "reason": item["reason"],
            "test_evidence": evidence,
        })
    return {
        "audit_source": audit_path,
        "summary": {
            "total": len(rows),
            "statuses": {
                status: counts.get(status, 0)
                for status in ("implemented", "partial", "placeholder")
            },
            "primitive_statuses": dict(sorted(Counter(
                row["primitive_status"] for row in rows
            ).items())),
        },
        "abilities": rows,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Ability Registry Audit",
        "",
        f"Audit source: `{report['audit_source']}`",
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in report["summary"]["statuses"].items():
        lines.append(f"| {status} | {count} |")
    lines.extend([
        "",
        "## Abilities",
        "",
        "| Ability | Registry | Primitive | Handler | Audit reason |",
        "|---|---|---|---|---|",
    ])
    for row in report["abilities"]:
        lines.append(
            f"| {row['keyword']} | {row['status']} | {row['primitive_status']} | "
            f"`{row['handler_name']}` | {row['reason']} |"
        )
        lines.append(
            f"| ↳ tests |  |  |  | {', '.join(f'`{path}`' for path in row['test_evidence'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report audited ability status")
    parser.add_argument("--audit", default="data/audits/ability_registry.json")
    parser.add_argument("--output")
    parser.add_argument("--markdown")
    args = parser.parse_args()
    report = build_ability_audit(args.audit)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown:
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"{'能力':<10} {'Handler':<12} {'Primitive':<10} 触发事件")
    print("-" * 76)
    for definition in ABILITY_DEFINITIONS:
        events = ", ".join(event.value for event in definition.events) or "static"
        print(
            f"{definition.keyword.value:<10} "
            f"{definition.status.value:<12} "
            f"{primitive_status(definition.keyword):<10} "
            f"{events}"
        )


if __name__ == "__main__":
    main()
