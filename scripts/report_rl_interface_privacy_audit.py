"""Audit RL action routing, observation schemas, and privacy boundaries."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swb.db.repository import CardDefinition
from swb.engine.card_rules import RuleBook
from swb.engine.commands import CommandType
from swb.engine.environment import ShadowverseEnv
from swb.rl.versioning import (
    ACTION_LAYOUT_VERSION,
    OBSERVATION_SCHEMA_VERSIONS,
    action_layout_manifest,
    observation_schema_manifest,
    stable_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = Path("data/audits/rl_interface_privacy_evidence.json")
DEFAULT_PLAY_MODE_REPORT = Path(
    "data/reports/card_bug_audit/play_mode_boundary_audit.json"
)
DEFAULT_BUG_LEDGER = Path("data/reports/card_bug_audit/bug_ledger.json")
DEFAULT_OUTPUT = Path(
    "data/reports/card_bug_audit/rl_interface_privacy_audit.json"
)
DEFAULT_MARKDOWN = Path(
    "data/reports/card_bug_audit/rl_interface_privacy_audit.md"
)

SOURCE_INPUTS = (
    Path("swb/engine/environment.py"),
    Path("swb/engine/observation_v3.py"),
    Path("swb/engine/observation_v4_1.py"),
    Path("swb/rl/versioning.py"),
    Path("swb/simulator/history.py"),
    Path("swb/simulator/service.py"),
)

COMMAND_EVIDENCE = {
    "end_turn": (
        "tests/test_rl_interface_privacy_audit.py::"
        "RLInterfacePrivacyAuditTests::test_base_command_round_trips_are_unique",
    ),
    "play_card": (
        "tests/test_rl_interface_privacy_audit.py::"
        "RLInterfacePrivacyAuditTests::test_base_command_round_trips_are_unique",
        "tests/test_play_modes_audit.py::"
        "RLEncodingTests::test_normal_play_round_trip",
    ),
    "attack": (
        "tests/test_rl_interface_privacy_audit.py::"
        "RLInterfacePrivacyAuditTests::test_base_command_round_trips_are_unique",
        "tests/test_environment.py::"
        "EnvironmentTests::test_play_and_attack_leader",
    ),
    "evolve": (
        "tests/test_rl_interface_privacy_audit.py::"
        "RLInterfacePrivacyAuditTests::test_base_command_round_trips_are_unique",
    ),
    "super_evolve": (
        "tests/test_rl_interface_privacy_audit.py::"
        "RLInterfacePrivacyAuditTests::test_base_command_round_trips_are_unique",
        "tests/test_environment.py::"
        "EnvironmentTests::test_super_evolution_uses_appended_action_slots",
    ),
    "choose": (
        "tests/test_rl_interface_privacy_audit.py::"
        "RLInterfacePrivacyAuditTests::"
        "test_graveyard_pagination_is_complete_unique_and_bounded",
        "tests/test_environment.py::"
        "EnvironmentTests::test_rl_choice_actions_resume_targeted_spell",
    ),
    "begin_fusion": (
        "tests/test_fusion.py::FusionEnvironmentTests::"
        "test_rl_reuses_special_and_choice_actions_for_fusion",
    ),
    "activate_amulet": (
        "tests/test_activate.py::ActivateEnvironmentTests::"
        "test_rl_mask_exposes_play_then_activation_for_activation_only_amulet",
    ),
    "use_extra_pp": (
        "tests/test_official_match_setup.py::OfficialCapacityAndRLTests::"
        "test_rl_reuses_choice_slots_for_mulligan_and_appends_extra_pp_action",
    ),
}

CHECKLIST_CONTRACTS = (
    {
        "contract_id": "command_has_one_expected_action",
        "tests": (
            "tests/test_rl_interface_privacy_audit.py::"
            "RLInterfacePrivacyAuditTests::"
            "test_base_command_round_trips_are_unique",
            "tests/test_play_mode_boundary_audit.py::"
            "PlayModeBoundaryAuditTests::"
            "test_commands_masks_execution_and_illegal_atomicity_all_pass",
        ),
    },
    {
        "contract_id": "true_mask_executes_expected_command",
        "tests": (
            "tests/test_rl_interface_privacy_audit.py::"
            "RLInterfacePrivacyAuditTests::"
            "test_every_true_action_executes_expected_command",
        ),
    },
    {
        "contract_id": "false_mask_is_atomic",
        "tests": (
            "tests/test_rl_interface_privacy_audit.py::"
            "RLInterfacePrivacyAuditTests::"
            "test_illegal_mask_samples_are_atomic_across_layout_ranges",
            "tests/test_environment.py::EnvironmentTests::"
            "test_illegal_rl_action_does_not_mutate_core_state",
        ),
    },
    {
        "contract_id": "pagination_complete_unique_bounded",
        "tests": (
            "tests/test_rl_interface_privacy_audit.py::"
            "RLInterfacePrivacyAuditTests::"
            "test_graveyard_pagination_is_complete_unique_and_bounded",
            "tests/test_environment_limits.py::EnvironmentLimitTests::"
            "test_page_navigation_counts_toward_agent_step_limit",
        ),
    },
    {
        "contract_id": "opponent_hand_identity_hidden",
        "tests": (
            "tests/test_rl_interface_privacy_audit.py::"
            "RLInterfacePrivacyAuditTests::"
            "test_v3_6_and_v4_1_hide_hand_and_deck_identity",
        ),
    },
    {
        "contract_id": "opponent_deck_identity_and_order_hidden",
        "tests": (
            "tests/test_rl_interface_privacy_audit.py::"
            "RLInterfacePrivacyAuditTests::"
            "test_v3_6_and_v4_1_hide_hand_and_deck_identity",
        ),
    },
    {
        "contract_id": "public_history_tracks_actions_targets_and_zones",
        "tests": (
            "tests/test_rl_interface_privacy_audit.py::"
            "RLInterfacePrivacyAuditTests::"
            "test_public_history_tracks_play_attack_target_and_zone_change",
        ),
    },
    {
        "contract_id": "persistent_private_online_redacted",
        "tests": (
            "tests/test_match_simulator.py::MatchSimulatorTests::"
            "test_online_history_redacts_private_persistent_state",
            "tests/test_match_simulator.py::MatchSimulatorTests::"
            "test_history_persists_private_state_and_complete_policy_decisions",
        ),
    },
    {
        "contract_id": "v3_6_and_v4_1_shape_dtype_version",
        "tests": (
            "tests/test_rl_interface_privacy_audit.py::"
            "RLInterfacePrivacyAuditTests::"
            "test_formal_observation_manifests_match_live_spaces",
            "tests/test_observation_v4_1.py::ObservationV41Tests::"
            "test_schema_is_fixed_compact_and_space_checked",
        ),
    },
    {
        "contract_id": "observation_migration_decision_explicit",
        "tests": (
            "tests/test_rl_interface_privacy_audit.py::"
            "RLInterfacePrivacyAuditTests::"
            "test_migration_decisions_are_explicit_and_non_schema_changes",
        ),
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _card(card_id: int) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"audit-card-{card_id}",
        cost=1,
        card_type="随从",
        attack=1,
        life=2,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def _environment(version: str) -> ShadowverseEnv:
    first = [_card(100 + index) for index in range(40)]
    second = [_card(200 + index) for index in range(40)]
    env = ShadowverseEnv(
        first,
        second,
        class_a=1,
        class_b=1,
        seed=110,
        rulebook=RuleBook(),
        observation_version=version,
        card_vocabulary=tuple(range(100, 240)),
    )
    env.reset(seed=110)
    return env


def discover_tests(root: Path) -> set[str]:
    discovered: set[str] = set()
    for path in sorted((root / "tests").glob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    discovered.add(f"{relative}::{node.name}")
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name.startswith("test_")
                    ):
                        discovered.add(
                            f"{relative}::{node.name}::{child.name}"
                        )
    return discovered


def _schema_row(env: ShadowverseEnv) -> dict[str, object]:
    observation = env.observation()
    space = (
        env.observation_v3_space()
        if env.observation_version == "v3"
        else env.observation_v4_1_space()
    )
    manifest = observation_schema_manifest(env)
    return {
        "environment_version": env.observation_version,
        "formal_version": manifest["version"],
        "manifest_sha256": stable_json_sha256(manifest),
        "field_count": len(manifest["fields"]),
        "space_contains_observation": bool(space.contains(observation)),
        "fields": manifest["fields"],
        "privacy": manifest["privacy"],
    }


def _layout_analysis(layout: dict[str, object]) -> dict[str, object]:
    owners: dict[int, str] = {}
    overlaps: list[dict[str, object]] = []
    for row in layout["ranges"]:
        for action in range(int(row["start"]), int(row["stop"])):
            if action in owners:
                overlaps.append({
                    "action": action,
                    "first": owners[action],
                    "second": row["name"],
                })
            owners[action] = str(row["name"])
    size = int(layout["size"])
    gaps = [action for action in range(size) if action not in owners]
    out_of_range = sorted(action for action in owners if not 0 <= action < size)
    return {
        "covered_action_count": len(owners),
        "gaps": gaps,
        "overlaps": overlaps,
        "out_of_range": out_of_range,
        "passed": not gaps and not overlaps and not out_of_range,
    }


def build_report(root: Path = ROOT) -> dict[str, object]:
    evidence_path = root / DEFAULT_EVIDENCE
    play_mode_path = root / DEFAULT_PLAY_MODE_REPORT
    ledger_path = root / DEFAULT_BUG_LEDGER
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    play_mode = json.loads(play_mode_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    tests = discover_tests(root)

    env_v3 = _environment("v3")
    env_v41 = _environment("v4.1")
    layout = action_layout_manifest(env_v3)
    layout_analysis = _layout_analysis(layout)
    schemas = [_schema_row(env_v3), _schema_row(env_v41)]
    failures: list[dict[str, object]] = []

    command_rows = []
    enum_values = {command.value for command in CommandType}
    if enum_values != set(COMMAND_EVIDENCE):
        failures.append({
            "kind": "command_inventory",
            "expected": sorted(enum_values),
            "actual": sorted(COMMAND_EVIDENCE),
        })
    for command in sorted(enum_values):
        references = COMMAND_EVIDENCE.get(command, ())
        missing = sorted(reference for reference in references if reference not in tests)
        row = {
            "command_type": command,
            "tests": list(references),
            "missing_tests": missing,
            "passed": bool(references) and not missing,
        }
        command_rows.append(row)
        if not row["passed"]:
            failures.append({"kind": "command_evidence", **row})

    contract_rows = []
    for contract in CHECKLIST_CONTRACTS:
        missing = sorted(
            reference
            for reference in contract["tests"]
            if reference not in tests
        )
        row = {
            "contract_id": contract["contract_id"],
            "tests": list(contract["tests"]),
            "missing_tests": missing,
            "passed": not missing,
        }
        contract_rows.append(row)
        if not row["passed"]:
            failures.append({"kind": "checklist_contract", **row})

    if layout["version"] != ACTION_LAYOUT_VERSION or not layout_analysis["passed"]:
        failures.append({
            "kind": "action_layout",
            "version": layout["version"],
            "analysis": layout_analysis,
        })
    expected_schema_versions = {
        "v3": OBSERVATION_SCHEMA_VERSIONS["v3"],
        "v4.1": OBSERVATION_SCHEMA_VERSIONS["v4.1"],
    }
    for row in schemas:
        if (
            row["formal_version"]
            != expected_schema_versions[row["environment_version"]]
            or not row["space_contains_observation"]
        ):
            failures.append({"kind": "observation_schema", **row})

    play_summary = play_mode["summary"]
    if (
        not play_summary["passed"]
        or play_summary["command_action_mask_mismatch_count"] != 0
        or play_summary["illegal_atomicity_failure_count"] != 0
    ):
        failures.append({
            "kind": "full_pool_play_mode_gate",
            "summary": play_summary,
        })

    source_rows = []
    for source in evidence["sources"]:
        path = root / source["path"]
        exists = path.is_file()
        source_rows.append({**source, "exists": exists})
        if not exists:
            failures.append({"kind": "missing_source", "path": source["path"]})

    migrations = evidence["migration_decisions"]
    if any(
        row["migration_required"]
        or row["observation_fields_changed"]
        or row["action_layout_changed"]
        for row in migrations
    ):
        failures.append({
            "kind": "migration_decision",
            "decisions": migrations,
        })

    privacy_bug = next(
        (
            entry for entry in ledger["entries"]
            if entry["bug_id"] == "SWB-CARD-0004"
        ),
        None,
    )
    if privacy_bug is None or privacy_bug["status"] != "fixed":
        failures.append({
            "kind": "privacy_bug_not_closed",
            "bug_id": "SWB-CARD-0004",
            "status": None if privacy_bug is None else privacy_bug["status"],
        })

    inputs = {
        path.as_posix(): _sha256(root / path)
        for path in SOURCE_INPUTS
    }
    inputs[DEFAULT_EVIDENCE.as_posix()] = _sha256(evidence_path)
    inputs[DEFAULT_PLAY_MODE_REPORT.as_posix()] = _sha256(play_mode_path)
    inputs[DEFAULT_BUG_LEDGER.as_posix()] = _sha256(ledger_path)

    summary = {
        "action_layout_version": layout["version"],
        "action_size": layout["size"],
        "command_type_count": len(command_rows),
        "checklist_contract_count": len(contract_rows),
        "observation_schema_count": len(schemas),
        "v3_6_field_count": schemas[0]["field_count"],
        "v4_1_field_count": schemas[1]["field_count"],
        "database_card_count": play_summary["database_card_count"],
        "full_pool_play_mode_card_count": (
            play_summary["full_pool_play_mode_card_count"]
        ),
        "full_pool_cost_boundary_case_count": (
            play_summary["cost_boundary_case_count"]
        ),
        "command_action_mask_mismatch_count": (
            play_summary["command_action_mask_mismatch_count"]
        ),
        "illegal_atomicity_failure_count": (
            play_summary["illegal_atomicity_failure_count"]
        ),
        "observation_migration_required": False,
        "failure_count": len(failures),
        "passed": not failures,
    }
    return {
        "schema_version": 1,
        "report_kind": "swb_rl_interface_privacy_audit",
        "inputs": inputs,
        "summary": summary,
        "action_layout": layout,
        "action_layout_analysis": layout_analysis,
        "command_matrix": command_rows,
        "checklist_contracts": contract_rows,
        "observation_schemas": schemas,
        "privacy_bug": privacy_bug,
        "migration_decisions": migrations,
        "evidence_sources": source_rows,
        "full_pool_play_mode_summary": play_summary,
        "failures": failures,
    }


def render_json(report: dict[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def render_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# RL Interface, Observation, and Privacy Audit",
        "",
        "This report is the executable evidence index for checklist section 1.10.",
        "",
        "## Summary",
        "",
        f"- Action layout: `{summary['action_layout_version']}` "
        f"({summary['action_size']} actions)",
        f"- Command types: {summary['command_type_count']}",
        f"- Checklist contracts: {summary['checklist_contract_count']}",
        f"- Observation schemas: {summary['observation_schema_count']}",
        f"- v3.6 fields: {summary['v3_6_field_count']}",
        f"- v4.1 fields: {summary['v4_1_field_count']}",
        f"- Full-pool cost boundary cases: "
        f"{summary['full_pool_cost_boundary_case_count']}",
        f"- Command/mask mismatches: "
        f"{summary['command_action_mask_mismatch_count']}",
        f"- Illegal atomicity failures: "
        f"{summary['illegal_atomicity_failure_count']}",
        f"- Observation migration required: "
        f"{str(summary['observation_migration_required']).lower()}",
        f"- Failures: {summary['failure_count']}",
        f"- Result: {'PASS' if summary['passed'] else 'FAIL'}",
        "",
        "## Command matrix",
        "",
        "| Command | Tests | Result |",
        "|---|---:|:---:|",
    ]
    for row in report["command_matrix"]:
        lines.append(
            f"| `{row['command_type']}` | {len(row['tests'])} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Checklist contracts",
        "",
        "| Contract | Tests | Result |",
        "|---|---:|:---:|",
    ])
    for row in report["checklist_contracts"]:
        lines.append(
            f"| `{row['contract_id']}` | {len(row['tests'])} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Observation schemas",
        "",
        "| Environment | Formal version | Fields | Manifest SHA-256 | Result |",
        "|---|---|---:|---|:---:|",
    ])
    for row in report["observation_schemas"]:
        lines.append(
            f"| `{row['environment_version']}` | `{row['formal_version']}` | "
            f"{row['field_count']} | `{row['manifest_sha256']}` | "
            f"{'PASS' if row['space_contains_observation'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Migration decisions",
        "",
        "| Change | Scope | Migration | Reason |",
        "|---|---|:---:|---|",
    ])
    for row in report["migration_decisions"]:
        lines.append(
            f"| `{row['change']}` | {row['scope']} | "
            f"{'yes' if row['migration_required'] else 'no'} | "
            f"{row['reason']} |"
        )
    bug = report["privacy_bug"]
    lines.extend([
        "",
        "## Privacy finding",
        "",
        (
            "No `SWB-CARD-0004` ledger entry was found."
            if bug is None
            else (
                f"- `{bug['bug_id']}`: {bug['severity']} / "
                f"{bug['status']} / fix `{bug['fix_commit']}`"
            )
        ),
        "",
    ])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_report(ROOT)
    output = ROOT / args.output
    markdown = ROOT / args.markdown
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_json(report), encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    summary = report["summary"]
    print(
        "actions={action_size} commands={command_type_count} "
        "contracts={checklist_contract_count} v3_fields={v3_6_field_count} "
        "v4_1_fields={v4_1_field_count} cases="
        "{full_pool_cost_boundary_case_count} failures={failure_count} "
        "passed={passed}".format(**summary)
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
