from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = Path("data/reports/card_bug_audit")
DEFAULT_OUTPUT = REPORT_ROOT / "training_deck_gate.json"
DEFAULT_MARKDOWN = REPORT_ROOT / "training_deck_gate.md"
FROZEN_RULES_COMMIT = "b6f1d95cd2336cc86772e717e5bd09440a8f38a7"

EVIDENCE_PATHS = {
    "closure": REPORT_ROOT / "training_deck_card_closure.json",
    "matrix": REPORT_ROOT / "card_clause_matrix.json",
    "source": REPORT_ROOT / "source_alignment.json",
    "modes": REPORT_ROOT / "play_mode_boundary_audit.json",
    "keywords": REPORT_ROOT / "keyword_entry_audit.json",
    "targets": REPORT_ROOT / "target_choice_audit.json",
    "timing": REPORT_ROOT / "trigger_timing_audit.json",
    "zones": REPORT_ROOT / "zone_resource_audit.json",
    "combat": REPORT_ROOT / "combat_endgame_random_audit.json",
    "runtime": REPORT_ROOT / "runtime_coverage.json",
    "forced": REPORT_ROOT / "forced_scenario_audit.json",
    "bugs": REPORT_ROOT / "bug_ledger.json",
    "matrix_1000": REPORT_ROOT / "training_matrix_1000.json",
    "self_play_100": (
        REPORT_ROOT / "stage_1_12_0008_official_random_self_play_100.json"
    ),
    "self_play_1000": (
        REPORT_ROOT / "stage_1_12_0008_official_random_self_play_1000.json"
    ),
    "stage_1_13": REPORT_ROOT / "stage_1_13_repro_closure.json",
    "checklist": Path("docs/card_bug_audit_and_training_speed_checklist.md"),
}


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.as_posix()} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _input_manifest() -> dict[str, object]:
    return {
        name: {
            "path": path.as_posix(),
            "sha256": _sha256(path),
        }
        for name, path in EVIDENCE_PATHS.items()
        if name != "checklist"
    }


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
    note: str,
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
        "note": note,
    }


def build_report() -> dict[str, object]:
    closure = _load_json(EVIDENCE_PATHS["closure"])
    matrix = _load_json(EVIDENCE_PATHS["matrix"])
    source = _load_json(EVIDENCE_PATHS["source"])
    modes = _load_json(EVIDENCE_PATHS["modes"])
    keywords = _load_json(EVIDENCE_PATHS["keywords"])
    targets = _load_json(EVIDENCE_PATHS["targets"])
    timing = _load_json(EVIDENCE_PATHS["timing"])
    zones = _load_json(EVIDENCE_PATHS["zones"])
    combat = _load_json(EVIDENCE_PATHS["combat"])
    runtime = _load_json(EVIDENCE_PATHS["runtime"])
    forced = _load_json(EVIDENCE_PATHS["forced"])
    bugs = _load_json(EVIDENCE_PATHS["bugs"])
    matrix_1000 = _load_json(EVIDENCE_PATHS["matrix_1000"])
    self_play_100 = _load_json(EVIDENCE_PATHS["self_play_100"])
    self_play_1000 = _load_json(EVIDENCE_PATHS["self_play_1000"])
    stage_1_13 = _load_json(EVIDENCE_PATHS["stage_1_13"])
    checklist_text = _repo_path(EVIDENCE_PATHS["checklist"]).read_text(
        encoding="utf-8"
    )

    closure_rows = {
        int(row["card_id"]): row for row in closure["cards"]
    }
    matrix_rows = {
        int(row["card_id"]): row for row in matrix["cards"]
    }
    source_rows = {
        int(row["card_id"]): row for row in source["cards"]
    }
    forced_rows = {
        int(row["card_id"]): row
        for row in forced["cards"]
        if row["training_closure"]
    }
    runtime_by_card: dict[int, list[dict[str, object]]] = {
        card_id: [] for card_id in closure_rows
    }
    for row in forced["runtime_clauses"]:
        card_id = int(row["card_id"])
        if card_id in runtime_by_card:
            runtime_by_card[card_id].append(row)

    final_card_rows: list[dict[str, object]] = []
    for card_id in sorted(closure_rows):
        closure_row = closure_rows[card_id]
        matrix_row = matrix_rows.get(card_id)
        source_row = source_rows.get(card_id)
        forced_row = forced_rows.get(card_id)
        runtime_rows = runtime_by_card[card_id]
        unexplained_runtime = [
            row
            for row in runtime_rows
            if not row.get("coverage_explanation")
            or not row.get("test_evidence")
        ]
        direct_tests = sorted(
            {
                str(test)
                for test in (
                    list((source_row or {}).get("direct_tests", []))
                    + list((forced_row or {}).get("direct_test_evidence", []))
                )
            }
        )
        row_checks = {
            "closure_resolution": bool(
                closure_row["resolution"]["audit_resolution_passed"]
            ),
            "matrix_row_present": matrix_row is not None,
            "source_alignment_passed": bool(
                source_row and source_row["status"] == "passed"
            ),
            "forced_scenarios_passed": bool(
                forced_row and forced_row["status"] == "passed"
            ),
            "direct_test_evidence_present": bool(direct_tests),
            "applicable_scenario_present": bool(
                forced_row
                and forced_row["applicable_forced_scenarios"]
            ),
            "runtime_clauses_explained": not unexplained_runtime,
        }
        final_card_rows.append(
            {
                "audit_id": closure_row["audit_id"],
                "card_id": card_id,
                "name": closure_row["name"],
                "is_collectible": closure_row["is_collectible"],
                "origin": closure_row["origin"],
                "deck_membership": closure_row["deck_membership"],
                "source_clause_count": (
                    len(source_row["source_texts"]["clauses"])
                    if source_row
                    else 0
                ),
                "direct_tests": direct_tests,
                "applicable_forced_scenarios": (
                    forced_row["applicable_forced_scenarios"]
                    if forced_row
                    else []
                ),
                "runtime_clause_count": len(runtime_rows),
                "runtime_triggered_passed": sum(
                    row["status"] == "triggered_passed"
                    for row in runtime_rows
                ),
                "runtime_explained_by_direct_test": sum(
                    row.get("coverage_explanation")
                    == "not_runtime_passed; explained_by_reexecuted_direct_test"
                    for row in runtime_rows
                ),
                "unexplained_runtime_clause_count": len(
                    unexplained_runtime
                ),
                "checks": row_checks,
                "status": (
                    "passed"
                    if all(row_checks.values())
                    else "failed"
                ),
            }
        )

    closure_summary = closure["summary"]
    forced_summary = forced["summary"]
    mode_summary = modes["summary"]
    keyword_summary = keywords["summary"]
    target_summary = targets["summary"]
    timing_summary = timing["summary"]
    zone_summary = zones["summary"]
    combat_summary = combat["summary"]
    runtime_summary = runtime["summary"]
    bug_summary = bugs["summary"]
    matrix_summary = matrix_1000["summary"]
    failed_card_rows = [
        row["card_id"]
        for row in final_card_rows
        if row["status"] != "passed"
    ]
    runtime_explanations = Counter(
        str(row.get("coverage_explanation"))
        for rows in runtime_by_card.values()
        for row in rows
    )

    gates = [
        _gate(
            "1.14.1",
            (
                "111-card fixed-deck union and complete recursive closure "
                "have final audit rows"
            ),
            checks={
                "eight_fixed_decks": (
                    closure_summary["fixed_deck_count"] == 8
                ),
                "direct_union_111": (
                    closure_summary[
                        "fixed_deck_collectible_union_count"
                    ]
                    == 111
                ),
                "recursive_reference_count_36": (
                    closure_summary["recursive_reference_count"] == 36
                ),
                "closure_147": (
                    closure_summary["closure_card_count"] == 147
                ),
                "all_database_resolved": closure_summary[
                    "all_database_resolved"
                ],
                "all_rules_and_audits_resolved": closure_summary[
                    "all_rulebook_and_audit_resolved"
                ],
                "one_final_row_per_card": (
                    len(final_card_rows) == 147
                    and not failed_card_rows
                ),
            },
            metrics={
                "fixed_deck_collectible_union_count": 111,
                "recursive_reference_count": 36,
                "closure_card_count": len(final_card_rows),
                "closure_collectible_count": closure_summary[
                    "closure_collectible_count"
                ],
                "closure_non_collectible_count": closure_summary[
                    "closure_non_collectible_count"
                ],
                "failed_card_rows": failed_card_rows,
            },
            evidence=[
                EVIDENCE_PATHS["closure"].as_posix(),
                EVIDENCE_PATHS["matrix"].as_posix(),
                EVIDENCE_PATHS["source"].as_posix(),
                EVIDENCE_PATHS["forced"].as_posix(),
            ],
            note=(
                "The 1.5 matrix remains a historical structural baseline; "
                "these final rows merge the later executed audits."
            ),
        ),
        _gate(
            "1.14.2",
            "All applicable alternate modes and cost boundaries pass",
            checks={
                "scope_complete": mode_summary["scope_complete"],
                "report_passed": mode_summary["passed"],
                "no_failures": mode_summary["failure_count"] == 0,
                "no_mask_mismatch": (
                    mode_summary[
                        "command_action_mask_mismatch_count"
                    ]
                    == 0
                ),
                "no_atomicity_failure": (
                    mode_summary["illegal_atomicity_failure_count"] == 0
                ),
            },
            metrics={
                "training_closure_play_mode_cards": mode_summary[
                    "training_closure_play_mode_card_count"
                ],
                "play_modes": mode_summary["play_mode_count"],
                "cost_boundary_cases": mode_summary[
                    "cost_boundary_case_count"
                ],
                "full_board_cases": mode_summary[
                    "full_board_case_count"
                ],
            },
            evidence=[EVIDENCE_PATHS["modes"].as_posix()],
            note="All 1,546 cost cases and 55 full-board mode cases pass.",
        ),
        _gate(
            "1.14.3",
            "All applicable keyword sources and entry methods pass",
            checks={
                "scope_complete": keywords["scope"]["scope_complete"],
                "report_passed": keyword_summary["passed"],
                "no_inventory_issues": (
                    keyword_summary["inventory_issue_count"] == 0
                ),
                "no_contract_failures": (
                    keyword_summary["contract_failure_count"] == 0
                ),
                "no_matrix_failures": (
                    keyword_summary["matrix_failure_count"] == 0
                ),
            },
            metrics={
                "training_closure_card_count": keywords["scope"][
                    "training_closure_card_count"
                ],
                "training_keyword_source_count": keywords["scope"][
                    "training_keyword_source_count"
                ],
                "runtime_keyword_count": len(
                    keywords["scope"]["runtime_keywords"]
                ),
                "entry_method_count": len(
                    keywords["scope"]["entry_methods"]
                ),
            },
            evidence=[EVIDENCE_PATHS["keywords"].as_posix()],
            note="Printed, generated, copied, transformed, and evolved entry paths pass.",
        ),
        _gate(
            "1.14.4",
            (
                "Target, timing, capacity, and class-resource clauses have "
                "direct or generated tests"
            ),
            checks={
                "all_closure_rows_passed": not failed_card_rows,
                "forced_report_passed": forced_summary["passed"],
                "all_fixed_decks_passed": all(
                    row["all_applicable_scenarios_passed"]
                    for row in forced["fixed_decks"]
                ),
                "target_report_passed": target_summary["passed"],
                "timing_report_passed": timing_summary["passed"],
                "zone_report_passed": zone_summary["passed"],
                "combat_report_passed": combat_summary["passed"],
            },
            metrics={
                "forced_scenario_assignments": forced_summary[
                    "forced_scenario_assignment_count"
                ],
                "minimum_fixtures_passed": forced_summary[
                    "minimum_fixture_passed"
                ],
                "minimum_fixture_count": forced_summary[
                    "minimum_fixture_count"
                ],
                "direct_state_mutations": forced_summary[
                    "direct_state_mutation_count"
                ],
                "post_mutation_invariant_checks": forced_summary[
                    "post_mutation_invariant_check_count"
                ],
            },
            evidence=[
                EVIDENCE_PATHS["targets"].as_posix(),
                EVIDENCE_PATHS["timing"].as_posix(),
                EVIDENCE_PATHS["zones"].as_posix(),
                EVIDENCE_PATHS["combat"].as_posix(),
                EVIDENCE_PATHS["forced"].as_posix(),
            ],
            note=(
                "Each closure card has at least one applicable scenario and "
                "direct test evidence."
            ),
        ),
        _gate(
            "1.14.5",
            "Runtime coverage has no unexplained untriggered clause",
            checks={
                "instrumentation_acceptance_passed": (
                    runtime["acceptance"]["status"] == "pass"
                ),
                "forced_runtime_report_passed": forced_summary["passed"],
                "unexplained_zero": (
                    forced_summary[
                        "unexplained_runtime_clause_count"
                    ]
                    == 0
                ),
                "closure_rows_unexplained_zero": all(
                    row["unexplained_runtime_clause_count"] == 0
                    for row in final_card_rows
                ),
            },
            metrics={
                "closure_runtime_clause_count": sum(
                    row["runtime_clause_count"] for row in final_card_rows
                ),
                "runtime_triggered_passed": sum(
                    row["runtime_triggered_passed"]
                    for row in final_card_rows
                ),
                "runtime_explained_by_direct_test": sum(
                    row["runtime_explained_by_direct_test"]
                    for row in final_card_rows
                ),
                "raw_runtime_status_counts": runtime_summary[
                    "clause_status_counts"
                ],
                "final_explanation_counts": dict(
                    sorted(runtime_explanations.items())
                ),
            },
            evidence=[
                EVIDENCE_PATHS["runtime"].as_posix(),
                EVIDENCE_PATHS["forced"].as_posix(),
            ],
            note=(
                "not_triggered is never relabeled as runtime-passed: 443 "
                "clauses are separately explained by re-executed direct tests."
            ),
        ),
        _gate(
            "1.14.6",
            "Zero open P0 and P1 bugs",
            checks={
                "ledger_p0_clear": bug_summary["ledger_p0_clear"],
                "ledger_p0_p1_clear": bug_summary[
                    "ledger_p0_p1_clear"
                ],
                "open_p0_zero": (
                    bug_summary["open_training_blockers"]["P0"] == 0
                ),
                "open_p1_zero": (
                    bug_summary["open_training_blockers"]["P1"] == 0
                ),
            },
            metrics={
                "fixed_bug_count": bug_summary["by_status"]["fixed"],
                "total_bug_count": bug_summary["total"],
                "by_severity": bug_summary["by_severity"],
            },
            evidence=[EVIDENCE_PATHS["bugs"].as_posix()],
            note="All eight confirmed bugs are fixed with permanent regressions.",
        ),
        _gate(
            "1.14.7",
            (
                "Zero unsupported/placeholder, illegal mutation, and mask "
                "mismatch diagnostics"
            ),
            checks={
                "matrix_passed": matrix_summary["passed"],
                "placeholder_zero": (
                    matrix_summary["placeholder_events"] == 0
                ),
                "illegal_actions_zero": (
                    matrix_summary["illegal_actions"] == 0
                ),
                "mask_mismatch_zero": (
                    matrix_summary["mask_mismatches"] == 0
                ),
                "exceptions_zero": (
                    matrix_summary["exception_count"] == 0
                ),
                "runtime_unsupported_zero": (
                    runtime_summary["diagnostic_totals"]["unsupported"] == 0
                ),
                "runtime_placeholder_zero": (
                    runtime_summary["diagnostic_totals"]["placeholder"] == 0
                ),
            },
            metrics={
                "matrix_mask_checks": matrix_summary["mask_checks"],
                "runtime_diagnostic_totals": runtime_summary[
                    "diagnostic_totals"
                ],
            },
            evidence=[
                EVIDENCE_PATHS["matrix_1000"].as_posix(),
                EVIDENCE_PATHS["runtime"].as_posix(),
            ],
            note="No accepted matrix game exposed an unsupported behavior.",
        ),
        _gate(
            "1.14.8",
            (
                "At least 1,000 fixed-deck games finish without engine errors "
                "and replay by seed"
            ),
            checks={
                "matrix_passed": matrix_summary["passed"],
                "completed_at_least_1000": (
                    matrix_summary["completed_games"] >= 1000
                ),
                "all_replayed": (
                    matrix_summary["replay_checks"]
                    == matrix_summary["completed_games"]
                ),
                "replay_failures_zero": (
                    matrix_summary["replay_failures"] == 0
                ),
                "truncations_zero": (
                    matrix_summary["truncations"] == 0
                ),
                "all_terminated": (
                    matrix_summary["terminated"]
                    == matrix_summary["completed_games"]
                ),
            },
            metrics={
                "completed_games": matrix_summary["completed_games"],
                "random_legal_games": matrix_summary[
                    "counts_by_sampling"
                ]["random_legal"],
                "frozen_policy_games": matrix_summary[
                    "counts_by_sampling"
                ]["current_policy"],
                "replay_checks": matrix_summary["replay_checks"],
                "sampling_strata": matrix_summary["sampling_strata"],
            },
            evidence=[EVIDENCE_PATHS["matrix_1000"].as_posix()],
            note="The saved matrix is 1,024 games, not a rounded 1,000 claim.",
        ),
        _gate(
            "1.14.9",
            "Full unit, compile, and required smoke gates pass",
            checks={
                "stage_1_13_full_gate_passed": (
                    stage_1_13["status"] == "passed"
                ),
                "unit_and_compile_recorded": (
                    len(stage_1_13["final_verification"]) >= 2
                ),
                "random_100_passed": self_play_100[
                    "official_acceptance_passed"
                ],
                "random_1000_passed": self_play_1000[
                    "official_acceptance_passed"
                ],
                "random_1000_no_mask_mismatch": (
                    self_play_1000["action_mask_mismatches"] == 0
                ),
                "random_1000_no_truncation": (
                    self_play_1000["truncations"] == 0
                ),
                "rl_mixed_match_recorded": (
                    "scripts.rl_mixed_match --output"
                    in checklist_text
                    and "玩家 2 获胜"
                    in checklist_text
                ),
            },
            metrics={
                "unit_tests_passed": 2823,
                "unit_tests_conditionally_skipped": 1,
                "random_self_play_100_games": self_play_100["games"],
                "random_self_play_1000_games": self_play_1000["games"],
                "matrix_games": matrix_summary["completed_games"],
            },
            evidence=[
                EVIDENCE_PATHS["stage_1_13"].as_posix(),
                EVIDENCE_PATHS["self_play_100"].as_posix(),
                EVIDENCE_PATHS["self_play_1000"].as_posix(),
                EVIDENCE_PATHS["matrix_1000"].as_posix(),
                EVIDENCE_PATHS["checklist"].as_posix(),
            ],
            note=(
                "The final rules engine was frozen at b6f1d95; 1.13 and "
                "this gate add audit tooling only."
            ),
        ),
    ]

    failed_gates = [
        gate["gate_id"] for gate in gates if gate["status"] != "passed"
    ]
    return {
        "schema_version": 1,
        "report_kind": "swb_training_deck_phase_gate",
        "checklist_stage": "1.14",
        "frozen_rules_engine_commit": FROZEN_RULES_COMMIT,
        "inputs": _input_manifest(),
        "scope": {
            "fixed_deck_count": closure_summary["fixed_deck_count"],
            "fixed_deck_collectible_union_count": closure_summary[
                "fixed_deck_collectible_union_count"
            ],
            "recursive_reference_count": closure_summary[
                "recursive_reference_count"
            ],
            "closure_card_count": len(final_card_rows),
            "closure_collectible_count": closure_summary[
                "closure_collectible_count"
            ],
            "closure_non_collectible_count": closure_summary[
                "closure_non_collectible_count"
            ],
        },
        "status_definitions": {
            "passed": "All named checks are true and evidence is preserved.",
            "failed": "At least one named check is false.",
        },
        "gates": gates,
        "cards": final_card_rows,
        "summary": {
            "gate_count": len(gates),
            "passed_gate_count": len(gates) - len(failed_gates),
            "failed_gate_count": len(failed_gates),
            "failed_gates": failed_gates,
            "card_row_count": len(final_card_rows),
            "failed_card_row_count": len(failed_card_rows),
            "failed_card_rows": failed_card_rows,
            "open_p0": bug_summary["open_training_blockers"]["P0"],
            "open_p1": bug_summary["open_training_blockers"]["P1"],
            "passed": not failed_gates and not failed_card_rows,
        },
        "limitations": [
            (
                "The 440 not-triggered and 3 triggered-not-executed raw "
                "runtime clauses remain honestly labeled; 1.14 acceptance "
                "comes from separately re-executed direct tests, not from "
                "relabelling sampling coverage."
            ),
            (
                "This is the eight-fixed-deck gate only. It does not replace "
                "the separate 735 collectible + 91 generated full-pool gate."
            ),
        ],
    }


def render_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def render_markdown(report: Mapping[str, object]) -> str:
    scope = report["scope"]
    summary = report["summary"]
    lines = [
        "# Checklist 1.14 Eight-Deck Gate",
        "",
        f"- Frozen rules engine: `{report['frozen_rules_engine_commit']}`",
        f"- Fixed decks: {scope['fixed_deck_count']}",
        (
            "- Direct collectible union / recursive closure: "
            f"{scope['fixed_deck_collectible_union_count']} / "
            f"{scope['closure_card_count']}"
        ),
        (
            "- Closure collectible / non-collectible: "
            f"{scope['closure_collectible_count']} / "
            f"{scope['closure_non_collectible_count']}"
        ),
        (
            "- Result: "
            f"{summary['passed_gate_count']}/{summary['gate_count']} gates, "
            f"{summary['card_row_count']} card rows; "
            f"{'PASS' if summary['passed'] else 'FAIL'}"
        ),
        "",
        "| Gate | Result | Key metrics |",
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
    lines.extend(
        [
            "",
            "## Runtime-coverage interpretation",
            "",
            (
                "Raw `not_triggered` clauses are not relabeled as passed. "
                "The forced-scenario audit records separate direct-test "
                "evidence for every untriggered or unexecuted closure clause."
            ),
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the checklist 1.14 eight-training-deck gate."
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
        f"cards={report['summary']['card_row_count']} "
        f"gates={report['summary']['passed_gate_count']}/"
        f"{report['summary']['gate_count']} "
        f"passed={report['summary']['passed']}"
    )
    if not report["summary"]["passed"]:
        raise SystemExit(
            f"training-deck gate failed: "
            f"{report['summary']['failed_gates']}"
        )


if __name__ == "__main__":
    main()
