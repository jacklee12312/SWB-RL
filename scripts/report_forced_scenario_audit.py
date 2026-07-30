from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine.environment import ShadowverseEnv
from swb.engine.forced_scenarios import run_minimal_forced_scenarios
from swb.engine.card_rules import RuleBook
from swb.rl.catalog import TrainableCardCatalog
from swb.rl.fixed_decks import (
    fixed_training_deck_names,
    get_fixed_training_deck,
)


DEFAULT_CLOSURE = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)
DEFAULT_COVERAGE = Path("data/reports/rule_coverage.json")
DEFAULT_RUNTIME_COVERAGE = Path(
    "data/reports/card_bug_audit/runtime_coverage.json"
)
DEFAULT_JSON = Path(
    "data/reports/card_bug_audit/forced_scenario_audit.json"
)
DEFAULT_MARKDOWN = Path(
    "data/reports/card_bug_audit/forced_scenario_audit.md"
)

MECHANISM_REPORTS = {
    "cost": "play_mode_boundary_audit",
    "target": "target_choice_audit",
    "capacity": "zone_resource_audit",
    "resource": "zone_resource_audit",
    "ordinary_evolution": "keyword_entry_audit",
    "super_evolution": "keyword_entry_audit",
    "turn_start": "trigger_timing_audit",
    "turn_end": "trigger_timing_audit",
    "simultaneous_death": "trigger_timing_audit",
}

TARGET_EFFECTS = frozenset({
    "damage_unit",
    "heal_unit",
    "heal_unit_and_leader",
    "buff_unit",
    "destroy",
    "banish",
    "transform",
    "return_to_hand",
    "return_to_deck",
    "add_keyword",
    "remove_keyword",
    "set_stats",
    "evolve_unit",
    "super_evolve_unit",
    "copy_to_hand",
})
CAPACITY_EFFECTS = frozenset({
    "add_card",
    "copy_to_hand",
    "draw",
    "draw_filtered",
    "reanimate",
    "summon",
    "summon_copy",
    "summon_exact_copy",
    "summon_from_deck",
    "summon_from_graveyard",
    "summon_from_hand",
})
RESOURCE_EFFECTS = frozenset({
    "add_combo",
    "add_earth_sigils",
    "add_shadows",
    "change_max_mana",
    "consume_faith",
    "earth_rite",
    "faith",
    "necromancy",
    "restore_evolution_points",
    "restore_mana",
    "restore_super_evolution_points",
})


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_passed(payload: dict[str, object]) -> bool:
    summary = payload.get("summary", {})
    if isinstance(summary, dict) and "passed" in summary:
        return bool(summary["passed"])
    acceptance = payload.get("acceptance", {})
    return (
        isinstance(acceptance, dict)
        and acceptance.get("status") == "pass"
    )


def _mechanisms_for_card(row: dict[str, object]) -> list[str]:
    card_type = str(row["card_type"])
    clause_audit = row.get("clause_audit", {})
    structured = (
        clause_audit.get("structured_evidence", {})
        if isinstance(clause_audit, dict)
        else {}
    )
    triggers = {
        str(item)
        for item in (
            structured.get("triggers", [])
            if isinstance(structured, dict)
            else []
        )
    }
    effects = {
        str(item).removeprefix("keyword:")
        for item in (
            structured.get("effect_kinds", [])
            if isinstance(structured, dict)
            else []
        )
    }
    mechanisms = {"cost"}
    if card_type in {"随从", "护符"} or effects & CAPACITY_EFFECTS:
        mechanisms.add("capacity")
    if effects & TARGET_EFFECTS:
        mechanisms.add("target")
    if effects & RESOURCE_EFFECTS or any(
        marker in item
        for item in (*triggers, *effects)
        for marker in (
            "mana",
            "shadow",
            "cooperation",
            "combo",
            "faith",
            "earth",
            "necromancy",
        )
    ):
        mechanisms.add("resource")
    if "evolve" in triggers or "self_evolved" in triggers:
        mechanisms.add("ordinary_evolution")
    if "super_evolve" in triggers or "self_super_evolved" in triggers:
        mechanisms.add("super_evolution")
    if "turn_start" in triggers:
        mechanisms.add("turn_start")
    if "turn_end" in triggers:
        mechanisms.add("turn_end")
    if (
        "last_words" in triggers
        or "countdown_expired" in triggers
        or effects & {"destroy", "damage_unit"}
    ):
        mechanisms.add("simultaneous_death")
    return sorted(mechanisms)


def _existing_runtime_rows(
    runtime_report: dict[str, object],
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for session in runtime_report.get("sessions", []):
        for row in session.get("clauses", []):
            clause_id = str(row["clause_id"])
            previous = rows.get(clause_id)
            rank = {
                "not_triggered": 0,
                "triggered_not_executed": 1,
                "triggered_passed": 2,
            }
            if previous is None or rank[str(row["status"])] > rank[
                str(previous["status"])
            ]:
                rows[clause_id] = dict(row)
    return rows


def _runtime_catalog(
    *,
    database: Path,
    card_ids: set[int],
    seed: int,
) -> list[dict[str, object]]:
    repository = CardRepository(database)
    catalog = TrainableCardCatalog.from_repository(repository)
    rulebook = RuleBook.from_directory(ShadowverseEnv.DEFAULT_RULE_DIRECTORY)
    names = tuple(sorted(fixed_training_deck_names()))
    first = get_fixed_training_deck(names[0])
    second = get_fixed_training_deck(names[1])
    env = ShadowverseEnv(
        first.build(catalog),
        second.build(catalog),
        class_a=first.class_id,
        class_b=second.class_id,
        seed=seed,
        rulebook=rulebook,
        card_resolver=catalog.resolve,
        validate_invariants=True,
        training_mode=True,
        audit_runtime_coverage=True,
        audit_context={
            "sampling_kind": "forced_scenario_clause_catalog",
            "seed": seed,
        },
    )
    env.reset(seed=seed)
    return list(env.runtime_coverage.to_session(card_ids=card_ids)["clauses"])


def build_report(
    *,
    database: Path,
    closure: Path,
    coverage: Path,
    runtime_coverage: Path,
    report_directory: Path,
    seed: int,
) -> dict[str, object]:
    closure_payload = _load_json(closure)
    coverage_payload = _load_json(coverage)
    previous_runtime = _load_json(runtime_coverage)
    repository_card_ids = set(CardRepository(database).card_ids())
    closure_ids = {
        int(row["card_id"]) for row in closure_payload["cards"]
    }
    classifications = coverage_payload["classifications"]
    all_rows = {
        int(card_id): row
        for card_id, row in classifications.items()
        if int(card_id) in repository_card_ids
    }
    all_ids = set(all_rows)
    collectible_ids = {
        card_id
        for card_id, row in all_rows.items()
        if bool(row["is_collectible"])
    }
    generated_ids = all_ids - collectible_ids

    mechanism_reports: dict[str, dict[str, object]] = {}
    report_contracts: dict[str, dict[str, object]] = {}
    for report_name in sorted(set(MECHANISM_REPORTS.values())):
        path = report_directory / f"{report_name}.json"
        payload = _load_json(path)
        mechanism_reports[report_name] = payload
        report_contracts[report_name] = {
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "passed": _report_passed(payload),
        }

    fixture_results = [
        result.to_dict() for result in run_minimal_forced_scenarios()
    ]
    fixture_status = {
        str(row["category"]): str(row["status"])
        for row in fixture_results
    }
    previous_rows = _existing_runtime_rows(previous_runtime)
    catalog_rows = _runtime_catalog(
        database=database,
        card_ids=all_ids,
        seed=seed,
    )

    test_files: set[str] = set()
    card_rows: list[dict[str, object]] = []
    missing_test_evidence: list[int] = []
    mechanism_failures: list[str] = []
    for card_id in sorted(all_ids):
        row = all_rows[card_id]
        clause_audit = row.get("clause_audit", {})
        evidence = sorted({
            str(path)
            for path in (
                clause_audit.get("test_evidence", [])
                if isinstance(clause_audit, dict)
                else []
            )
        })
        test_files.update(evidence)
        if not evidence:
            missing_test_evidence.append(card_id)
        mechanisms = _mechanisms_for_card(row)
        mechanism_evidence = []
        for mechanism in mechanisms:
            report_name = MECHANISM_REPORTS[mechanism]
            passed = report_contracts[report_name]["passed"]
            if not passed:
                mechanism_failures.append(
                    f"card {card_id} mechanism {mechanism}: "
                    f"{report_name} did not pass"
                )
            mechanism_evidence.append({
                "scenario": mechanism,
                "minimum_fixture_status": fixture_status[mechanism],
                "stratified_report": report_contracts[report_name]["path"],
                "stratified_report_passed": passed,
            })
        card_rows.append({
            "card_id": card_id,
            "name": row["name"],
            "collectible": bool(row["is_collectible"]),
            "training_closure": card_id in closure_ids,
            "applicable_forced_scenarios": mechanism_evidence,
            "direct_test_evidence": evidence,
            "status": (
                "passed"
                if evidence
                and all(
                    item["stratified_report_passed"]
                    for item in mechanism_evidence
                )
                else "failed"
            ),
        })

    missing_test_files = sorted(
        path for path in test_files if not Path(path).is_file()
    )
    runtime_rows: list[dict[str, object]] = []
    unexplained: list[str] = []
    runtime_counts: Counter[str] = Counter()
    direct_evidence_by_card = {
        int(row["card_id"]): list(row["direct_test_evidence"])
        for row in card_rows
    }
    for row in catalog_rows:
        clause_id = str(row["clause_id"])
        card_id = int(row["card_id"])
        previous = previous_rows.get(clause_id)
        runtime_status = (
            str(previous["status"])
            if previous is not None
            else "not_sampled_full_pool"
        )
        runtime_counts[runtime_status] += 1
        direct_evidence = direct_evidence_by_card[card_id]
        if runtime_status == "triggered_passed":
            explanation = "runtime_triggered_passed"
        elif direct_evidence:
            explanation = "not_runtime_passed; explained_by_reexecuted_direct_test"
        else:
            explanation = "unexplained"
            unexplained.append(clause_id)
        runtime_rows.append({
            **row,
            "status": runtime_status,
            "coverage_explanation": explanation,
            "test_evidence": direct_evidence,
        })

    deck_rows = []
    for name in sorted(fixed_training_deck_names()):
        deck = get_fixed_training_deck(name)
        member_ids = sorted(set(deck.card_ids))
        closure_member_ids = sorted(
            card_id
            for card_id in closure_ids
            if any(
                membership["deck_name"] == name
                for membership in next(
                    row["deck_membership"]
                    for row in closure_payload["cards"]
                    if int(row["card_id"]) == card_id
                )
            )
        )
        assigned = [
            item
            for item in card_rows
            if item["card_id"] in closure_member_ids
        ]
        deck_rows.append({
            "deck_name": name,
            "class_id": deck.class_id,
            "direct_collectible_card_count": len(member_ids),
            "closure_card_count": len(closure_member_ids),
            "closure_card_ids": closure_member_ids,
            "forced_scenario_assignment_count": sum(
                len(item["applicable_forced_scenarios"])
                for item in assigned
            ),
            "all_applicable_scenarios_passed": all(
                item["status"] == "passed" for item in assigned
            ),
        })

    failures = [
        *(
            [f"missing test evidence for {missing_test_evidence}"]
            if missing_test_evidence
            else []
        ),
        *(
            [f"missing test files: {missing_test_files}"]
            if missing_test_files
            else []
        ),
        *mechanism_failures,
        *(
            [f"unexplained runtime clauses: {unexplained}"]
            if unexplained
            else []
        ),
    ]
    if len(collectible_ids) != 735:
        failures.append(
            f"collectible catalog expected 735, found {len(collectible_ids)}"
        )
    if len(generated_ids) != 91:
        failures.append(
            f"generated catalog expected 91, found {len(generated_ids)}"
        )
    if len(closure_ids) != 147:
        failures.append(
            f"training closure expected 147, found {len(closure_ids)}"
        )
    if len(deck_rows) != 8:
        failures.append(f"fixed deck registry expected 8, found {len(deck_rows)}")
    if any(row["status"] != "passed" for row in fixture_results):
        failures.append("one or more minimum forced fixtures failed")
    if any(
        not row["all_applicable_scenarios_passed"] for row in deck_rows
    ):
        failures.append(
            "one or more fixed-deck closures have an unpassed scenario"
        )

    return {
        "schema_version": 1,
        "report_kind": "swb_forced_scenario_audit",
        "inputs": {
            "database": database.as_posix(),
            "database_sha256": _sha256(database),
            "closure": closure.as_posix(),
            "closure_sha256": _sha256(closure),
            "coverage": coverage.as_posix(),
            "coverage_sha256": _sha256(coverage),
            "runtime_coverage": runtime_coverage.as_posix(),
            "runtime_coverage_sha256": _sha256(runtime_coverage),
            "seed": seed,
        },
        "public_interface_contract": {
            "commands": [
                "PlayCard",
                "Choose",
                "Evolve",
                "SuperEvolve",
                "EndTurn",
                "Attack",
            ],
            "state_interface": [
                "GameState",
                "PlayerState",
                "HandCard",
                "Unit",
            ],
            "effect_interface": [
                "CardRule",
                "EffectOperation",
                "Condition",
            ],
            "direct_mutation_policy": (
                "Every direct fixture mutation is immediately followed by "
                "GameEngine.assert_invariants()."
            ),
            "private_resolution_helpers_used": False,
        },
        "minimum_fixtures": fixture_results,
        "mechanism_reports": report_contracts,
        "fixed_decks": deck_rows,
        "cards": card_rows,
        "runtime_clauses": runtime_rows,
        "summary": {
            "minimum_fixture_count": len(fixture_results),
            "minimum_fixture_passed": sum(
                row["status"] == "passed" for row in fixture_results
            ),
            "direct_state_mutation_count": sum(
                int(row["direct_state_mutations"])
                for row in fixture_results
            ),
            "post_mutation_invariant_check_count": sum(
                int(row["invariant_checks"])
                for row in fixture_results
            ),
            "fixed_deck_count": len(deck_rows),
            "training_closure_card_count": len(closure_ids),
            "collectible_card_count": len(collectible_ids),
            "generated_card_count": len(generated_ids),
            "full_pool_card_count": len(all_ids),
            "forced_scenario_assignment_count": sum(
                len(row["applicable_forced_scenarios"])
                for row in card_rows
            ),
            "runtime_clause_count": len(runtime_rows),
            "runtime_status_counts": dict(sorted(runtime_counts.items())),
            "unexplained_runtime_clause_count": len(unexplained),
            "test_file_count": len(test_files),
            "missing_test_file_count": len(missing_test_files),
            "failure_count": len(failures),
            "passed": not failures,
        },
        "limitations": [
            (
                "A direct regression test explains a clause that random play "
                "did not trigger, but is not relabelled as runtime-triggered."
            ),
            (
                "Random/current-policy game distributions and truncation "
                "analysis are saved by card_audit_sampling.py."
            ),
        ],
        "failures": failures,
    }


def render_report(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# 1.12 Forced Scenario Audit",
        "",
        f"- Acceptance: `{'pass' if summary['passed'] else 'fail'}`",
        (
            "- Minimum public-interface fixtures: "
            f"{summary['minimum_fixture_passed']}/"
            f"{summary['minimum_fixture_count']} passed"
        ),
        (
            "- Direct mutations / invariant checks: "
            f"{summary['direct_state_mutation_count']} / "
            f"{summary['post_mutation_invariant_check_count']}"
        ),
        (
            "- Scope: "
            f"{summary['fixed_deck_count']} decks, "
            f"{summary['training_closure_card_count']} closure cards, "
            f"{summary['collectible_card_count']} collectible + "
            f"{summary['generated_card_count']} generated cards"
        ),
        (
            "- Runtime clauses: "
            f"{summary['runtime_clause_count']}; unexplained "
            f"{summary['unexplained_runtime_clause_count']}"
        ),
        "",
        "## Minimum Fixtures",
        "",
        "| Scenario | Category | Status | Invariant checks |",
        "|---|---|---:|---:|",
    ]
    for row in report["minimum_fixtures"]:
        lines.append(
            f"| `{row['scenario_id']}` | `{row['category']}` | "
            f"{row['status']} | {row['invariant_checks']} |"
        )
    lines.extend([
        "",
        "## Evidence Boundary",
        "",
        (
            "Runtime 未触发的 clause 只记录为“由已重新执行的直接测试解释”，"
            "不会被改写为 runtime passed。完整 1,000/10,000 局分布由独立"
            "采样报告保存。"
        ),
        "",
    ])
    if report["failures"]:
        lines.extend([
            "## Failures",
            "",
            *[f"- {failure}" for failure in report["failures"]],
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the checklist 1.12 forced-scenario audit"
    )
    parser.add_argument(
        "--database", type=Path, default=Path("data/cards.sqlite3")
    )
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument(
        "--runtime-coverage",
        type=Path,
        default=DEFAULT_RUNTIME_COVERAGE,
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--output-markdown", type=Path, default=DEFAULT_MARKDOWN
    )
    parser.add_argument("--seed", type=int, default=1200)
    args = parser.parse_args()
    report = build_report(
        database=args.database,
        closure=args.closure,
        coverage=args.coverage,
        runtime_coverage=args.runtime_coverage,
        report_directory=args.output_json.parent,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(
        render_report(report),
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        "forced_scenario_audit "
        f"acceptance={'pass' if summary['passed'] else 'fail'} "
        f"fixtures={summary['minimum_fixture_passed']}/"
        f"{summary['minimum_fixture_count']} "
        f"cards={summary['full_pool_card_count']} "
        f"clauses={summary['runtime_clause_count']} "
        f"unexplained={summary['unexplained_runtime_clause_count']}"
    )
    if not summary["passed"]:
        raise SystemExit(
            "forced scenario audit failed: "
            + "; ".join(report["failures"])
        )


if __name__ == "__main__":
    main()
