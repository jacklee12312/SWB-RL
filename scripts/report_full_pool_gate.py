from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping

from swb.db.repository import CardRepository
from swb.rl.catalog import TrainableCardCatalog
from swb.rl.runtime import hash_rule_directory


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path("data/reports/card_bug_audit")
DEFAULT_OUTPUT = REPORT_ROOT / "final_gate.json"
DEFAULT_MARKDOWN = Path("docs/card_bug_audit_report.md")

EVIDENCE_PATHS = {
    "rule_coverage": Path("data/reports/rule_coverage.json"),
    "token_audit": Path("data/reports/token_audit.json"),
    "play_modes": REPORT_ROOT / "play_mode_boundary_audit.json",
    "keywords": REPORT_ROOT / "keyword_entry_audit.json",
    "targets": REPORT_ROOT / "target_choice_audit.json",
    "timing": REPORT_ROOT / "trigger_timing_audit.json",
    "zones": REPORT_ROOT / "zone_resource_audit.json",
    "combat": REPORT_ROOT / "combat_endgame_random_audit.json",
    "forced": REPORT_ROOT / "forced_scenario_audit.json",
    "sampling_10000": REPORT_ROOT / "full_pool_sampling_10000.json",
    "sampling_failed_preserved": (
        REPORT_ROOT / "full_pool_sampling_10000_failed_20260730.json"
    ),
    "bugs": REPORT_ROOT / "bug_ledger.json",
    "reproductions": REPORT_ROOT / "stage_1_13_repro_closure.json",
    "rulings": Path("data/audits/card_ruling_reviews.json"),
    "catalog_exclusions": (
        Path("data/audits/training_catalog_exclusions.json")
    ),
    "rl_interface": REPORT_ROOT / "rl_interface_privacy_audit.json",
    "database": Path("data/cards.sqlite3"),
}

MECHANISM_KEYS = (
    "play_modes",
    "keywords",
    "targets",
    "timing",
    "zones",
    "combat",
    "forced",
)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.as_posix()} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _directory_sha256(path: Path, pattern: str) -> str:
    root = _repo_path(path)
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in root.rglob(pattern)
        if candidate.is_file()
    )
    for candidate in files:
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        payload = candidate.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _git_commit_for(*paths: str) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError(f"unable to freeze Git commit for {paths!r}")
    subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return commit


def _passed_summary(payload: Mapping[str, object]) -> bool:
    summary = payload.get("summary", {})
    return bool(
        isinstance(summary, Mapping)
        and summary.get("passed") is True
        and int(summary.get("failure_count", 0)) == 0
    )


def _gate(
    gate_id: str,
    title: str,
    *,
    checks: Mapping[str, bool],
    metrics: Mapping[str, object],
    evidence: list[str],
    conclusion: str,
) -> dict[str, object]:
    failed_checks = sorted(
        name for name, passed in checks.items() if not passed
    )
    return {
        "gate_id": gate_id,
        "title": title,
        "status": "passed" if not failed_checks else "failed",
        "checks": dict(checks),
        "failed_checks": failed_checks,
        "metrics": dict(metrics),
        "evidence": evidence,
        "conclusion": conclusion,
    }


def build_report() -> dict[str, object]:
    evidence = {
        key: _load_json(path)
        for key, path in EVIDENCE_PATHS.items()
        if path.suffix == ".json"
    }
    coverage = evidence["rule_coverage"]
    tokens = evidence["token_audit"]
    forced = evidence["forced"]
    sampling = evidence["sampling_10000"]
    failed_sampling = evidence["sampling_failed_preserved"]
    bugs = evidence["bugs"]
    reproductions = evidence["reproductions"]
    rulings = evidence["rulings"]
    exclusions = evidence["catalog_exclusions"]
    rl_interface = evidence["rl_interface"]

    coverage_summary = coverage["summary"]
    coverage_counts = coverage_summary["coverage_counts"]
    clause_counts = coverage_summary["clause_audit_counts"]
    blocker_counts = coverage_summary["blocker_counts"]
    token_summary = tokens["summary"]
    token_categories = token_summary["categories"]

    mechanism_results = {
        key: _passed_summary(evidence[key]) for key in MECHANISM_KEYS
    }

    repository = CardRepository(_repo_path(EVIDENCE_PATHS["database"]))
    catalog = TrainableCardCatalog.from_repository(repository)
    exclusion_rows = exclusions["exclusions"]
    excluded_ids = {
        int(row["card_id"]) for row in exclusion_rows
    }
    ruling_rows = rulings["entries"]
    uncertain_rows = [
        row for row in ruling_rows if row["status"] == "ruling_uncertain"
    ]
    unresolved_without_exclusion: list[str] = []
    for row in uncertain_rows:
        disposition = row.get("catalog_disposition", {})
        disposition_ids = {
            int(card_id)
            for card_id in disposition.get("excluded_card_ids", [])
        }
        if (
            disposition.get("status") != "excluded_pending_ruling"
            or not disposition_ids
            or not disposition_ids.issubset(excluded_ids)
            or not disposition_ids.issubset(
                set(catalog.excluded_collectible_ids)
            )
        ):
            unresolved_without_exclusion.append(str(row["ruling_id"]))

    forced_summary = forced["summary"]
    runtime_counts = forced_summary["runtime_status_counts"]
    runtime_total = int(forced_summary["runtime_clause_count"])
    raw_nonpassed = sum(
        int(runtime_counts.get(status, 0))
        for status in (
            "not_sampled_full_pool",
            "not_triggered",
            "triggered_not_executed",
        )
    )
    runtime_status_total = sum(int(value) for value in runtime_counts.values())

    sampling_summary = sampling["summary"]
    failed_sampling_summary = failed_sampling["summary"]
    bug_summary = bugs["summary"]
    bug_entries = bugs["entries"]
    open_p2_p3_without_detail = [
        row["bug_id"]
        for row in bug_entries
        if row["status"] in {"open", "ruling_uncertain"}
        and row["severity"] in {"P2", "P3"}
        and (
            not row.get("impact")
            or not row.get("reproduction")
        )
    ]

    reproduction_scope = reproductions["scope"]
    reproduction_workflow = reproductions["bug_workflow_audit"]
    reproduction_package = reproductions["portable_reproduction"]

    observation_rows = rl_interface["observation_schemas"]
    observation_freeze = {
        row["formal_version"]: {
            "manifest_sha256": row["manifest_sha256"],
            "field_count": row["field_count"],
        }
        for row in observation_rows
    }
    action_payload = json.dumps(
        rl_interface["action_layout"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    frozen = {
        "rules_engine_commit": _git_commit_for(
            "swb/engine/state.py",
            "swb/engine/resolution.py",
            "data/rules",
        ),
        "catalog_policy_commit": _git_commit_for(
            "data/audits/training_catalog_exclusions.json"
        ),
        "database_sha256": _sha256(EVIDENCE_PATHS["database"]),
        "rules_sha256": hash_rule_directory(
            _repo_path(Path("data/rules"))
        ),
        "coverage_report_sha256": catalog.coverage_report_sha256,
        "catalog_exclusion_policy_sha256": (
            catalog.exclusion_policy_sha256
        ),
        "catalog_sha256": catalog.catalog_sha256,
        "card_vocabulary_sha256": catalog.card_vocabulary_sha256,
        "training_pool_sha256": catalog.training_pool_sha256,
        "audited_exact_collectible_count": len(
            catalog.audited_exact_collectible_ids
        ),
        "trainable_collectible_count": len(catalog.exact_collectible_ids),
        "excluded_collectible_ids": list(
            catalog.excluded_collectible_ids
        ),
        "observation": observation_freeze,
        "action_layout": {
            "version": rl_interface["action_layout"]["version"],
            "size": rl_interface["action_layout"]["size"],
            "sha256": hashlib.sha256(action_payload).hexdigest(),
        },
        "tests_sha256": _directory_sha256(Path("tests"), "*.py"),
        "scripts_sha256": _directory_sha256(Path("scripts"), "*.py"),
    }

    gates = [
        _gate(
            "1.15.1",
            "735 collectible clause audit",
            checks={
                "covered_exact_735": (
                    coverage_counts["covered_exact"] == 735
                ),
                "mapped_exact_735": (
                    clause_counts["mapped_exact"] == 735
                ),
                "no_clause_gaps": all(
                    int(clause_counts.get(key, 0)) == 0
                    for key in (
                        "unverified_exact",
                        "partial",
                        "missing_rule",
                        "missing_primitive",
                        "text_unclear",
                    )
                ),
                "no_blockers": all(
                    int(value) == 0 for value in blocker_counts.values()
                ),
            },
            metrics={
                "covered_exact": coverage_counts["covered_exact"],
                "mapped_exact": clause_counts["mapped_exact"],
                "blocker_count": sum(blocker_counts.values()),
            },
            evidence=["data/reports/rule_coverage.json"],
            conclusion="All 735 collectibles retain explicit clause status.",
        ),
        _gate(
            "1.15.2",
            "91 generated-card audit",
            checks={
                "total_91": token_summary["total"] == 91,
                "all_complete": (
                    token_categories["entry_behavior_complete"] == 91
                ),
                "no_partial_or_missing": all(
                    int(token_categories.get(key, 0)) == 0
                    for key in (
                        "entry_behavior_partial",
                        "database_only_no_entry",
                        "text_unclear",
                        "external_blocker",
                    )
                ),
            },
            metrics={
                "total": token_summary["total"],
                "complete": token_categories["entry_behavior_complete"],
            },
            evidence=["data/reports/token_audit.json"],
            conclusion="All 91 generated cards have entry and behavior evidence.",
        ),
        _gate(
            "1.15.3",
            "full-pool mechanism matrices",
            checks=mechanism_results,
            metrics={
                "mechanism_report_count": len(mechanism_results),
                "passed_report_count": sum(mechanism_results.values()),
                "forced_assignments": forced_summary[
                    "forced_scenario_assignment_count"
                ],
            },
            evidence=[
                EVIDENCE_PATHS[key].as_posix()
                for key in MECHANISM_KEYS
            ],
            conclusion="Every full-pool mechanism report has zero failures.",
        ),
        _gate(
            "1.15.4",
            "uncertain-ruling disposition",
            checks={
                "every_uncertain_ruling_is_excluded": (
                    not unresolved_without_exclusion
                ),
                "catalog_policy_matches_runtime": (
                    excluded_ids
                    == set(catalog.excluded_collectible_ids)
                ),
                "all_735_still_resolvable": (
                    len(catalog.audited_exact_collectible_ids) == 735
                ),
                "active_training_pool_is_734": (
                    len(catalog.exact_collectible_ids) == 734
                ),
            },
            metrics={
                "uncertain_ruling_count": len(uncertain_rows),
                "excluded_collectible_count": len(excluded_ids),
                "trainable_collectible_count": len(
                    catalog.exact_collectible_ids
                ),
                "unresolved_without_exclusion": (
                    unresolved_without_exclusion
                ),
            },
            evidence=[
                "data/audits/card_ruling_reviews.json",
                "data/audits/training_catalog_exclusions.json",
            ],
            conclusion=(
                "The unresolved edge remains explicitly uncertain and its "
                "reachable source is excluded from newly sampled decks."
            ),
        ),
        _gate(
            "1.15.5",
            "runtime-coverage honesty",
            checks={
                "status_counts_cover_every_clause": (
                    runtime_status_total == runtime_total
                ),
                "raw_nonpassed_remains_nonpassed": raw_nonpassed > 0,
                "no_unexplained_clause": (
                    forced_summary["unexplained_runtime_clause_count"] == 0
                ),
                "no_generic_pass_relabel": "passed" not in runtime_counts,
            },
            metrics={
                "runtime_clause_count": runtime_total,
                "runtime_status_counts": runtime_counts,
                "raw_nonpassed_count": raw_nonpassed,
                "unexplained_count": forced_summary[
                    "unexplained_runtime_clause_count"
                ],
            },
            evidence=[
                "data/reports/card_bug_audit/forced_scenario_audit.json"
            ],
            conclusion=(
                "Unsampled, untriggered and unexecuted clauses keep their "
                "raw labels; direct tests explain them separately."
            ),
        ),
        _gate(
            "1.15.6",
            "10,000-game stratified sampling",
            checks={
                "report_passed": sampling_summary["passed"] is True,
                "completed_10000": (
                    sampling_summary["completed_games"] == 10_000
                ),
                "all_735_deck_cards_sampled": (
                    sampling_summary["deck_exact_card_count"] == 735
                    and sampling_summary[
                        "deck_exact_card_coverage_rate"
                    ] == 1.0
                ),
                "no_invariant_or_execution_error": (
                    sampling_summary["exception_count"] == 0
                    and sampling_summary["failure_count"] == 0
                ),
                "no_mask_or_illegal_action": (
                    sampling_summary["mask_mismatches"] == 0
                    and sampling_summary["illegal_actions"] == 0
                ),
                "no_placeholder_or_truncation": (
                    sampling_summary["placeholder_events"] == 0
                    and sampling_summary["truncations"] == 0
                ),
                "all_replays_passed": (
                    sampling_summary["replay_checks"] == 98
                    and sampling_summary["replay_failures"] == 0
                ),
            },
            metrics={
                "games": sampling_summary["completed_games"],
                "strata": sampling_summary["sampling_strata"],
                "mask_checks": sampling_summary["mask_checks"],
                "encountered_cards": sampling_summary[
                    "encountered_card_count"
                ],
                "replays": sampling_summary["replay_checks"],
            },
            evidence=[
                "data/reports/card_bug_audit/full_pool_sampling_10000.json",
                (
                    "data/reports/card_bug_audit/"
                    "full_pool_sampling_10000_failed_20260730.json"
                ),
            ],
            conclusion=(
                "The post-fix run passes all 98 strata. The earlier failed "
                f"run remains preserved with "
                f"{failed_sampling_summary['exception_count']} exceptions "
                "and is not overwritten."
            ),
        ),
        _gate(
            "1.15.7",
            "bug severity closure",
            checks={
                "open_p0_zero": (
                    bug_summary["open_training_blockers"]["P0"] == 0
                ),
                "open_p1_zero": (
                    bug_summary["open_training_blockers"]["P1"] == 0
                ),
                "all_confirmed_fixed": (
                    bug_summary["by_status"]["fixed"]
                    == bug_summary["total"]
                ),
                "p2_p3_have_impact_and_repro": (
                    not open_p2_p3_without_detail
                ),
            },
            metrics={
                "total": bug_summary["total"],
                "fixed": bug_summary["by_status"]["fixed"],
                "open_p0": bug_summary["open_training_blockers"]["P0"],
                "open_p1": bug_summary["open_training_blockers"]["P1"],
                "open_p2_p3_without_detail": open_p2_p3_without_detail,
            },
            evidence=["data/reports/card_bug_audit/bug_ledger.json"],
            conclusion="All eight confirmed P0/P1 bugs are fixed.",
        ),
        _gate(
            "1.15.8",
            "portable reproduction collection",
            checks={
                "stage_1_13_passed": reproductions["status"] == "passed",
                "all_eight_fixed": (
                    reproduction_scope["confirmed_bug_count"] == 8
                    and reproduction_scope["fixed_bug_count"] == 8
                ),
                "all_saved_pre_fix_reproductions": (
                    reproduction_workflow[
                        "saved_pre_fix_reproduction_count"
                    ] == 8
                ),
                "all_permanent_regressions": (
                    reproduction_workflow["permanent_regression_count"] == 8
                ),
                "portable_package_complete": all(
                    reproduction_package[key] is True
                    for key in (
                        "database_and_rule_hashes_present",
                        "exact_decks_and_seed_present",
                        "pre_command_snapshot_present",
                        "structured_command_present",
                        "legal_actions_and_mask_present",
                        "transition_events_present",
                        "official_expected_and_pre_fix_actual_present",
                        "only_json_native_values",
                    )
                ),
            },
            metrics={
                "confirmed_bugs": reproduction_scope[
                    "confirmed_bug_count"
                ],
                "fixed_bugs": reproduction_scope["fixed_bug_count"],
                "portable_package": reproduction_package["package"],
                "minimized_actions": reproductions["minimization"][
                    "minimized_action_count"
                ],
            },
            evidence=[
                "data/reports/card_bug_audit/stage_1_13_repro_closure.json",
                reproduction_package["package"],
            ],
            conclusion=(
                "The same eight-bug regression collection and portable "
                "SWB-CARD-0008 package pass on the frozen rules engine."
            ),
        ),
        _gate(
            "1.15.9",
            "frozen implementation manifests",
            checks={
                "sampling_database_hash_matches": (
                    frozen["database_sha256"]
                    == sampling["inputs"]["database_sha256"]
                ),
                "sampling_rule_hash_matches": (
                    frozen["rules_sha256"]
                    == sampling["inputs"]["rulebook_sha256"]
                ),
                "git_commits_present": all(
                    len(frozen[key]) == 40
                    for key in (
                        "rules_engine_commit",
                        "catalog_policy_commit",
                    )
                ),
                "catalog_counts_frozen": (
                    frozen["audited_exact_collectible_count"] == 735
                    and frozen["trainable_collectible_count"] == 734
                ),
                "observation_manifests_frozen": (
                    set(frozen["observation"])
                    == {"observation-v3.6", "observation-v4.1"}
                    and all(
                        len(row["manifest_sha256"]) == 64
                        for row in frozen["observation"].values()
                    )
                ),
                "tests_and_scripts_frozen": (
                    len(frozen["tests_sha256"]) == 64
                    and len(frozen["scripts_sha256"]) == 64
                ),
            },
            metrics={
                "rules_engine_commit": frozen["rules_engine_commit"],
                "catalog_policy_commit": frozen["catalog_policy_commit"],
                "database_sha256": frozen["database_sha256"],
                "rules_sha256": frozen["rules_sha256"],
                "catalog_sha256": frozen["catalog_sha256"],
                "tests_sha256": frozen["tests_sha256"],
            },
            evidence=[
                "data/cards.sqlite3",
                "data/rules",
                "data/audits/training_catalog_exclusions.json",
                "data/reports/card_bug_audit/rl_interface_privacy_audit.json",
                "tests",
            ],
            conclusion=(
                "Git, database, rules, Catalog, Observation, action layout, "
                "tests and scripts have stable identifiers."
            ),
        ),
    ]
    failed_gates = [
        gate["gate_id"] for gate in gates if gate["status"] != "passed"
    ]
    return {
        "schema_version": 1,
        "report_kind": "swb_card_bug_audit_final_gate",
        "checklist_stage": "1.15",
        "audit_date": "2026-07-31",
        "inputs": {
            key: {
                "path": path.as_posix(),
                "sha256": _sha256(path),
            }
            for key, path in EVIDENCE_PATHS.items()
        },
        "frozen": frozen,
        "gates": gates,
        "summary": {
            "gate_count": len(gates),
            "passed_gate_count": len(gates) - len(failed_gates),
            "failed_gate_count": len(failed_gates),
            "failed_gates": failed_gates,
            "collectible_audited": 735,
            "generated_audited": 91,
            "trainable_collectible": len(catalog.exact_collectible_ids),
            "excluded_pending_ruling": len(
                catalog.excluded_collectible_ids
            ),
            "open_p0": bug_summary["open_training_blockers"]["P0"],
            "open_p1": bug_summary["open_training_blockers"]["P1"],
            "passed": not failed_gates,
        },
        "known_limitations": [
            {
                "id": "SWB-RULING-SET-STATS-TEMP-001",
                "status": "ruling_uncertain",
                "impact": (
                    "Older temporary stat-modifier expiry after a later "
                    "specific-value assignment remains officially unconfirmed."
                ),
                "training_disposition": (
                    "10233310 is excluded from newly sampled initial decks; "
                    "the card remains auditable and resolvable."
                ),
            },
            {
                "id": "RUNTIME-COVERAGE-SAMPLING-LIMIT",
                "status": "explained_not_runtime_passed",
                "impact": (
                    f"{raw_nonpassed} of {runtime_total} runtime clauses were "
                    "not sampled, not triggered or not executed in the smoke "
                    "corpus and retain those labels."
                ),
                "training_disposition": (
                    "Each is separately attributed to re-executed direct "
                    "tests in the forced-scenario report."
                ),
            },
        ],
    }


def render_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def render_markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    frozen = report["frozen"]
    lines = [
        "# 卡牌 Bug 审计最终报告",
        "",
        "审计日期：2026-07-31",
        "",
        "## 结论",
        "",
        (
            f"完整卡池门禁 {summary['passed_gate_count']}/"
            f"{summary['gate_count']} 通过。735 张可收集卡和 91 张衍生卡"
            "均保留逐条审计证据；P0/P1 未关闭 Bug 为 0。"
        ),
        (
            "当前新采样训练池为 734 张可收集卡；《帕梅拉的舞蹈》因"
            " `SWB-RULING-SET-STATS-TEMP-001` 尚无直接官方裁定而被显式"
            "排除，但仍可解析、审计和历史回放。"
        ),
        "",
        "## 门禁结果",
        "",
        "| 门禁 | 结果 | 关键指标 |",
        "|---|---|---|",
    ]
    for gate in report["gates"]:
        metrics = ", ".join(
            f"{key}={value}"
            for key, value in gate["metrics"].items()
            if not isinstance(value, (dict, list))
        )
        lines.append(
            f"| {gate['gate_id']} {gate['title']} | "
            f"{gate['status']} | {metrics} |"
        )
    lines.extend([
        "",
        "## 10,000 局分层采样",
        "",
        (
            "最终报告完成 10,000/10,000 局、98 个分层、909,158 次 mask "
            "检查和 98 次固定 seed 重放；异常、截断、非法动作、placeholder、"
            "mask mismatch 和重放失败均为 0。更早失败报告未被覆盖，仍保留"
            "两次非正生命不变量异常和 194 个 placeholder，作为本轮修复来源。"
        ),
        "",
        "## Runtime coverage 解释",
        "",
        (
            "原始未采样、未触发和已触发未执行条款没有改标为通过。它们只通过"
            " forced-scenario 报告中独立重跑的直接测试获得解释，因此随机 smoke "
            "覆盖与直接行为证据保持分离。"
        ),
        "",
        "## 冻结标识",
        "",
        f"- 规则引擎提交：`{frozen['rules_engine_commit']}`",
        f"- Catalog 策略提交：`{frozen['catalog_policy_commit']}`",
        f"- 数据库 SHA-256：`{frozen['database_sha256']}`",
        f"- 规则 SHA-256：`{frozen['rules_sha256']}`",
        f"- Catalog SHA-256：`{frozen['catalog_sha256']}`",
        f"- 训练池 SHA-256：`{frozen['training_pool_sha256']}`",
        f"- 测试 SHA-256：`{frozen['tests_sha256']}`",
        "",
        "## 已知限制",
        "",
    ])
    for limitation in report["known_limitations"]:
        lines.append(
            f"- `{limitation['id']}`：{limitation['impact']} "
            f"{limitation['training_disposition']}"
        )
    lines.extend([
        "",
        "机器可读完整结果："
        "`data/reports/card_bug_audit/final_gate.json`。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the checklist 1.15 full-pool card audit gate."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report()
    output = _repo_path(args.output)
    markdown = _repo_path(args.markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_json(report), encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"gates={report['summary']['passed_gate_count']}/"
        f"{report['summary']['gate_count']} "
        f"collectible={report['summary']['collectible_audited']} "
        f"generated={report['summary']['generated_audited']} "
        f"trainable={report['summary']['trainable_collectible']} "
        f"passed={report['summary']['passed']}"
    )
    if not report["summary"]["passed"]:
        raise SystemExit(
            f"full-pool gate failed: {report['summary']['failed_gates']}"
        )


if __name__ == "__main__":
    main()
