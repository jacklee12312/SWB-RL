# -*- coding: utf-8 -*-
"""Build and validate the deterministic card-bug audit ledger."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))


SCHEMA_VERSION = 1
BUG_ID_PATTERN = re.compile(r"^SWB-CARD-\d{4}$")
SEVERITY_ORDER = ("P0", "P1", "P2", "P3")
STATUS_ORDER = ("open", "ruling_uncertain", "fixed", "closed_not_bug")

SEVERITY_DEFINITIONS = (
    {
        "severity": "P0",
        "definition": (
            "非法动作、费用或替代模式合法性、攻击权限、隐藏信息泄漏、胜负、"
            "伤害、奖励，或 action mask 与 command 不一致。"
        ),
        "training_policy": "立即暂停正式训练，修复并通过完整门禁前不得恢复。",
        "blocks_formal_training": True,
        "blocks_long_training": True,
    },
    {
        "severity": "P1",
        "definition": (
            "八套训练卡组及其递归衍生闭包中常见卡牌的效果、目标、时机、"
            "区域移动或职业资源结算错误。"
        ),
        "training_policy": "允许最小复现和修复验证；正式长训练前必须清零。",
        "blocks_formal_training": False,
        "blocks_long_training": True,
    },
    {
        "severity": "P2",
        "definition": (
            "低频卡牌或罕见组合的状态或规则错误；当前训练轨迹中出现概率低，"
            "但仍须登记、保存复现并增加回归测试。"
        ),
        "training_policy": "不阻塞小规模试验，可与修复工作并行。",
        "blocks_formal_training": False,
        "blocks_long_training": False,
    },
    {
        "severity": "P3",
        "definition": (
            "只影响 UI、动画建议或文字日志，不改变引擎状态、合法动作、奖励"
            "或策略输入。"
        ),
        "training_policy": "不阻塞训练，但仍须登记影响和处置结果。",
        "blocks_formal_training": False,
        "blocks_long_training": False,
    },
)

STATUS_DEFINITIONS = (
    {
        "status": "open",
        "meaning": "已确认且尚未修复。",
        "counts_as_closed": False,
    },
    {
        "status": "ruling_uncertain",
        "meaning": "规则证据不足，保持显式待确认；不得按现有引擎行为自行关闭。",
        "counts_as_closed": False,
    },
    {
        "status": "fixed",
        "meaning": (
            "已修复，并有修复版本标识和永久回归测试；用户禁止提交时，"
            "允许使用明确标注的工作树 diff hash。"
        ),
        "counts_as_closed": True,
    },
    {
        "status": "closed_not_bug",
        "meaning": "经外部裁定或可重复客户端证据确认不是 Bug，并记录结论。",
        "counts_as_closed": True,
    },
)

ENTRY_FIELD_DEFINITIONS = (
    {
        "name": "bug_id",
        "type": "string",
        "required": True,
        "description": "稳定编号，格式为 SWB-CARD-0001。",
    },
    {
        "name": "severity",
        "type": "enum[P0,P1,P2,P3]",
        "required": True,
        "description": "按本报告固化定义分级。",
    },
    {
        "name": "status",
        "type": "enum[open,ruling_uncertain,fixed,closed_not_bug]",
        "required": True,
        "description": "当前处理状态。",
    },
    {
        "name": "card",
        "type": "object{card_id: integer|null, name: string}",
        "required": True,
        "description": "受影响卡牌；通用机制缺陷允许 card_id 为 null。",
    },
    {
        "name": "mechanic",
        "type": "string",
        "required": True,
        "description": "受影响的通用机制或规则族。",
    },
    {
        "name": "discovery_commit",
        "type": "string",
        "required": True,
        "description": "保存最小复现时的 Git HEAD。",
    },
    {
        "name": "minimal_seed",
        "type": "integer|null",
        "required": True,
        "description": "最小复现 seed；确定性 fixture 无随机性时为 null。",
    },
    {
        "name": "reproduction_file",
        "type": "string",
        "required": True,
        "description": "仓库相对路径，指向可移植最小复现包。",
    },
    {
        "name": "expected",
        "type": "string",
        "required": True,
        "description": "由卡牌文字和外部证据支持的预期结果。",
    },
    {
        "name": "actual",
        "type": "string",
        "required": True,
        "description": "修复前实际结果。",
    },
    {
        "name": "impact",
        "type": "string",
        "required": True,
        "description": "对规则、轨迹、训练或展示的明确影响。",
    },
    {
        "name": "affected_decks",
        "type": "array[string]",
        "required": True,
        "description": "受影响的固定训练卡组名称，稳定排序且不得重复。",
    },
    {
        "name": "fix_commit",
        "type": "string|null",
        "required": True,
        "description": (
            "兼容字段：修复提交，或用户禁止提交时明确标注的工作树 diff hash；"
            "fixed 状态必须填写。"
        ),
    },
    {
        "name": "regression_tests",
        "type": "array[string]",
        "required": True,
        "description": "永久回归测试路径；fixed 状态必须非空。",
    },
    {
        "name": "notes",
        "type": "string",
        "required": True,
        "description": "裁定、关闭理由、checkpoint 影响或其他审计说明。",
    },
)

ENTRY_FIELDS = tuple(field["name"] for field in ENTRY_FIELD_DEFINITIONS)
REQUIRED_CHECKLIST_FIELDS = (
    "bug_id",
    "severity",
    "card",
    "mechanic",
    "discovery_commit",
    "minimal_seed",
    "reproduction_file",
    "expected",
    "actual",
    "affected_decks",
    "fix_commit",
    "regression_tests",
)


def _require_nonempty_string(entry: Mapping[str, object], field: str) -> str:
    value = entry[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _normalize_repo_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay inside the repository")
    return path.as_posix()


def normalize_bug_entry(raw_entry: Mapping[str, object]) -> dict[str, object]:
    """Validate one ledger entry and return it in deterministic field order."""

    if not isinstance(raw_entry, Mapping):
        raise ValueError("bug ledger entries must be objects")
    missing = [field for field in ENTRY_FIELDS if field not in raw_entry]
    extra = sorted(set(raw_entry) - set(ENTRY_FIELDS))
    if missing or extra:
        raise ValueError(f"invalid bug entry fields: missing={missing}, extra={extra}")

    bug_id = _require_nonempty_string(raw_entry, "bug_id")
    if BUG_ID_PATTERN.fullmatch(bug_id) is None:
        raise ValueError("bug_id must match SWB-CARD-0001")

    severity = raw_entry["severity"]
    if severity not in SEVERITY_ORDER:
        raise ValueError(f"invalid severity {severity!r}")
    status = raw_entry["status"]
    if status not in STATUS_ORDER:
        raise ValueError(f"invalid status {status!r}")

    raw_card = raw_entry["card"]
    if not isinstance(raw_card, Mapping) or set(raw_card) != {"card_id", "name"}:
        raise ValueError("card must contain exactly card_id and name")
    card_id = raw_card["card_id"]
    if card_id is not None and (
        not isinstance(card_id, int) or isinstance(card_id, bool) or card_id <= 0
    ):
        raise ValueError("card.card_id must be a positive integer or null")
    card_name = raw_card["name"]
    if not isinstance(card_name, str) or not card_name.strip():
        raise ValueError("card.name must be a non-empty string")

    minimal_seed = raw_entry["minimal_seed"]
    if minimal_seed is not None and (
        not isinstance(minimal_seed, int)
        or isinstance(minimal_seed, bool)
        or minimal_seed < 0
    ):
        raise ValueError("minimal_seed must be a non-negative integer or null")

    affected_decks = raw_entry["affected_decks"]
    if (
        not isinstance(affected_decks, list)
        or any(not isinstance(item, str) or not item.strip() for item in affected_decks)
    ):
        raise ValueError("affected_decks must be an array of non-empty strings")
    normalized_decks = sorted({item.strip() for item in affected_decks})
    if len(normalized_decks) != len(affected_decks):
        raise ValueError("affected_decks must not contain duplicates")

    regression_tests = raw_entry["regression_tests"]
    if not isinstance(regression_tests, list):
        raise ValueError("regression_tests must be an array")
    normalized_tests = sorted(
        _normalize_repo_path(item, "regression_tests")
        for item in regression_tests
    )
    if len(set(normalized_tests)) != len(normalized_tests):
        raise ValueError("regression_tests must not contain duplicates")

    fix_commit = raw_entry["fix_commit"]
    if fix_commit is not None and (
        not isinstance(fix_commit, str) or not fix_commit.strip()
    ):
        raise ValueError("fix_commit must be a non-empty string or null")
    if status == "fixed":
        if fix_commit is None:
            raise ValueError("fixed entries require fix_commit")
        if not normalized_tests:
            raise ValueError("fixed entries require regression_tests")
    elif fix_commit is not None:
        raise ValueError("only fixed entries may set fix_commit")

    notes = raw_entry["notes"]
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")
    if status == "closed_not_bug" and not notes.strip():
        raise ValueError("closed_not_bug entries require a recorded ruling")

    return {
        "bug_id": bug_id,
        "severity": severity,
        "status": status,
        "card": {
            "card_id": card_id,
            "name": card_name.strip(),
        },
        "mechanic": _require_nonempty_string(raw_entry, "mechanic"),
        "discovery_commit": _require_nonempty_string(
            raw_entry, "discovery_commit"
        ),
        "minimal_seed": minimal_seed,
        "reproduction_file": _normalize_repo_path(
            raw_entry["reproduction_file"], "reproduction_file"
        ),
        "expected": _require_nonempty_string(raw_entry, "expected"),
        "actual": _require_nonempty_string(raw_entry, "actual"),
        "impact": _require_nonempty_string(raw_entry, "impact"),
        "affected_decks": normalized_decks,
        "fix_commit": fix_commit.strip() if isinstance(fix_commit, str) else None,
        "regression_tests": normalized_tests,
        "notes": notes.strip(),
    }


def build_bug_ledger(
    entries: Iterable[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return a normalized, deterministically ordered ledger report."""

    normalized = [normalize_bug_entry(entry) for entry in entries]
    normalized.sort(key=lambda entry: entry["bug_id"])
    bug_ids = [entry["bug_id"] for entry in normalized]
    if len(set(bug_ids)) != len(bug_ids):
        raise ValueError("bug_id values must be unique")

    by_severity = Counter(str(entry["severity"]) for entry in normalized)
    by_status = Counter(str(entry["status"]) for entry in normalized)
    closed_statuses = {
        item["status"] for item in STATUS_DEFINITIONS if item["counts_as_closed"]
    }
    open_blockers = {
        severity: sum(
            entry["severity"] == severity and entry["status"] not in closed_statuses
            for entry in normalized
        )
        for severity in ("P0", "P1")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "swb_card_bug_ledger",
        "severity_definitions": list(SEVERITY_DEFINITIONS),
        "status_definitions": list(STATUS_DEFINITIONS),
        "entry_field_definitions": list(ENTRY_FIELD_DEFINITIONS),
        "summary": {
            "total": len(normalized),
            "by_severity": {
                severity: by_severity.get(severity, 0)
                for severity in SEVERITY_ORDER
            },
            "by_status": {
                status: by_status.get(status, 0) for status in STATUS_ORDER
            },
            "open_training_blockers": open_blockers,
            "ledger_p0_clear": open_blockers["P0"] == 0,
            "ledger_p0_p1_clear": (
                open_blockers["P0"] == 0 and open_blockers["P1"] == 0
            ),
        },
        "entries": normalized,
    }


def render_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def render_markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# Card Bug Audit Ledger",
        "",
        f"Schema version: `{report['schema_version']}`",
        "",
        "## Severity And Training Policy",
        "",
        "| Severity | Definition | Training policy |",
        "|---|---|---|",
    ]
    for row in report["severity_definitions"]:
        lines.append(
            f"| {row['severity']} | {row['definition']} | "
            f"{row['training_policy']} |"
        )
    lines.extend(
        [
            "",
            "P0 立即暂停正式训练；P1 必须在正式长训练前清零；"
            "P2 可与小规模试验并行；P3 不阻塞训练。",
            "",
            "## Ledger Contract",
            "",
            "| Field | Type | Required | Meaning |",
            "|---|---|---|---|",
        ]
    )
    for row in report["entry_field_definitions"]:
        lines.append(
            f"| `{row['name']}` | `{row['type']}` | "
            f"{'yes' if row['required'] else 'no'} | {row['description']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Total entries: {summary['total']}",
            f"- Open P0 blockers: {summary['open_training_blockers']['P0']}",
            f"- Open P1 blockers: {summary['open_training_blockers']['P1']}",
            f"- Ledger P0 clear: {str(summary['ledger_p0_clear']).lower()}",
            f"- Ledger P0/P1 clear: {str(summary['ledger_p0_p1_clear']).lower()}",
            "",
            "These ledger flags do not authorize training by themselves; the "
            "eight-deck and full-catalog checklist gates must also pass.",
            "",
            "## Entries",
            "",
        ]
    )
    if not report["entries"]:
        lines.append("No card bugs have been recorded at this audit baseline.")
    else:
        lines.extend(
            [
                "| ID | Severity | Status | Card | Mechanic | Reproduction |",
                "|---|---|---|---|---|---|",
            ]
        )
        for entry in report["entries"]:
            card = entry["card"]
            card_label = (
                f"{card['card_id']} {card['name']}"
                if card["card_id"] is not None
                else card["name"]
            )
            lines.append(
                f"| {entry['bug_id']} | {entry['severity']} | "
                f"{entry['status']} | {card_label} | {entry['mechanic']} | "
                f"`{entry['reproduction_file']}` |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/reports/card_bug_audit/bug_ledger.json",
    )
    parser.add_argument(
        "--markdown",
        default="data/reports/card_bug_audit/bug_ledger.md",
    )
    args = parser.parse_args()

    output = Path(args.output)
    markdown = Path(args.markdown)
    existing_entries: Iterable[Mapping[str, object]] = ()
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        existing_entries = existing.get("entries", ())

    report = build_bug_ledger(existing_entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_json(report), encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"JSON ledger written to {output}")
    print(f"Markdown ledger written to {markdown}")


if __name__ == "__main__":
    main()
