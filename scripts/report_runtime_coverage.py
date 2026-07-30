from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.environment import ShadowverseEnv
from swb.engine.runtime_coverage import (
    aggregate_runtime_coverage,
    render_runtime_coverage_markdown,
)
from swb.rl.catalog import TrainableCardCatalog
from swb.rl.fixed_decks import (
    fixed_training_deck_names,
    get_fixed_training_deck,
)


DEFAULT_CLOSURE = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)
DEFAULT_JSON = Path(
    "data/reports/card_bug_audit/runtime_coverage.json"
)
DEFAULT_MARKDOWN = Path(
    "data/reports/card_bug_audit/runtime_coverage.md"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _closure_inputs(
    path: Path,
) -> tuple[set[int], dict[int, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cards = payload.get("cards")
    if not isinstance(cards, list):
        raise ValueError(f"{path}: cards must be a list")
    card_ids: set[int] = set()
    memberships: dict[int, list[str]] = {}
    for row in cards:
        card_id = int(row["card_id"])
        card_ids.add(card_id)
        memberships[card_id] = sorted({
            str(entry["deck_name"])
            for entry in row.get("deck_membership", [])
        })
    return card_ids, memberships


def build_report(
    *,
    database: Path,
    closure: Path,
    seed: int,
    max_agent_steps: int,
) -> dict[str, object]:
    repository = CardRepository(database)
    catalog = TrainableCardCatalog.from_repository(repository)
    rulebook = RuleBook.from_directory(ShadowverseEnv.DEFAULT_RULE_DIRECTORY)
    deck_names = tuple(sorted(fixed_training_deck_names()))
    if len(deck_names) < 2:
        raise ValueError("runtime coverage smoke requires at least two fixed decks")
    deck_a_def = get_fixed_training_deck(deck_names[0])
    deck_b_def = get_fixed_training_deck(deck_names[1])
    matchup_id = f"{deck_a_def.name}__vs__{deck_b_def.name}"
    env = ShadowverseEnv(
        deck_a_def.build(catalog),
        deck_b_def.build(catalog),
        class_a=deck_a_def.class_id,
        class_b=deck_b_def.class_id,
        seed=seed,
        rulebook=rulebook,
        card_resolver=catalog.resolve,
        validate_invariants=True,
        training_mode=True,
        audit_runtime_coverage=True,
        audit_context={
            "deck_a": deck_a_def.name,
            "deck_b": deck_b_def.name,
            "matchup_id": matchup_id,
            "seed": seed,
            "sampling_kind": "deterministic_random_legal_smoke",
        },
        max_agent_steps=max_agent_steps,
    )
    _, info = env.reset(seed=seed)
    rng = random.Random(seed)
    while not (env.terminated or env.truncated):
        mask = list(info["action_mask"])
        legal_actions = [
            action for action, allowed in enumerate(mask) if allowed
        ]
        if not legal_actions:
            raise RuntimeError("runtime coverage smoke found no legal action")
        result = env.step(rng.choice(legal_actions))
        info = result.info

    closure_card_ids, memberships = _closure_inputs(closure)
    session = env.runtime_coverage.to_session(card_ids=closure_card_ids)
    report = aggregate_runtime_coverage(
        [session],
        deck_memberships=memberships,
    )
    diagnostics = report["summary"]["diagnostic_totals"]
    required_diagnostics = {
        "placeholder",
        "unsupported",
        "resolution_step_limit",
        "illegal_command",
        "illegal_action",
        "action_mask_mismatch",
    }
    clause_ids = [
        str(row["clause_id"])
        for row in session["clauses"]
    ]
    aggregation_keys = set(report["aggregations"])
    failures: list[str] = []
    if not clause_ids:
        failures.append("no structured operation clauses were catalogued")
    if any(not clause_id.isascii() for clause_id in clause_ids):
        failures.append("one or more clause IDs are not ASCII stable IDs")
    if set(diagnostics) != required_diagnostics:
        failures.append("required diagnostic counters are incomplete")
    if aggregation_keys != {
        "by_card",
        "by_mechanic",
        "by_deck",
        "by_matchup",
    }:
        failures.append("required aggregation dimensions are incomplete")
    if int(diagnostics["action_mask_mismatch"]) != 0:
        failures.append("instrumentation smoke found an action-mask mismatch")

    report["inputs"] = {
        "database": str(database).replace("\\", "/"),
        "database_sha256": _sha256(database),
        "rules_directory": str(
            ShadowverseEnv.DEFAULT_RULE_DIRECTORY
        ).replace("\\", "/"),
        "closure": str(closure).replace("\\", "/"),
        "closure_sha256": _sha256(closure),
    }
    report["scope"] = {
        "checklist_section": "1.11",
        "training_closure_card_count": len(closure_card_ids),
        "fixed_deck_count": len(deck_names),
        "smoke_matchup": matchup_id,
        "smoke_seed": seed,
        "smoke_steps": env.agent_steps,
        "winner": env.winner,
        "terminated": env.terminated,
        "truncated": env.truncated,
        "structured_operation_clause_count": len(clause_ids),
        "sampling_scope": (
            "1.11 instrumentation contract and one deterministic smoke; "
            "forced scenarios and 1,000-game coverage belong to 1.12"
        ),
        "phase_1_12_sampling_complete": False,
    }
    report["instrumentation_contract"] = {
        "structured_events_only": True,
        "localized_log_parsing": False,
        "stable_card_and_clause_ids": not any(
            not clause_id.isascii() for clause_id in clause_ids
        ),
        "lifecycle_categories": [
            "drawn",
            "played",
            "evolved",
            "super_evolved",
            "attacked",
            "left_play",
        ],
        "alternate_mode_categories": [
            "play",
            "fusion",
            "invocation",
            "activation",
            "union_burst",
            "choose",
        ],
        "target_metrics": [
            "target_kind",
            "no_target",
            "capacity_shortage",
            "random_candidate_count",
        ],
        "clause_metrics": [
            "entered",
            "condition_evaluated",
            "condition_true",
            "condition_false",
            "operation_executed",
        ],
    }
    report["acceptance"] = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "note": (
            "This accepts the 1.11 instrumentation contract only. "
            "not_triggered clauses remain open evidence for 1.12."
        ),
    }
    return report


def render_report(report: dict[str, object]) -> str:
    scope = report["scope"]
    acceptance = report["acceptance"]
    base = render_runtime_coverage_markdown(report)
    return "\n".join([
        base.rstrip(),
        "",
        "## 1.11 Instrumentation Smoke",
        "",
        f"- Acceptance: `{acceptance['status']}`",
        f"- Matchup: `{scope['smoke_matchup']}`",
        f"- Seed: {scope['smoke_seed']}",
        f"- Agent steps: {scope['smoke_steps']}",
        f"- Terminated: {str(scope['terminated']).lower()}",
        f"- Truncated: {str(scope['truncated']).lower()}",
        (
            "- Structured operation clauses in the training closure: "
            f"{scope['structured_operation_clause_count']}"
        ),
        "",
        "该 smoke 只验证 1.11 采集链路；强制场景、八套卡组对阵矩阵和"
        " 1,000 局采样属于 1.12，因此当前 `not_triggered` 不能解释为通过。",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate structured runtime card-ability coverage"
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/cards.sqlite3"),
    )
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_MARKDOWN,
    )
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--max-agent-steps", type=int, default=1000)
    args = parser.parse_args()
    report = build_report(
        database=args.database,
        closure=args.closure,
        seed=args.seed,
        max_agent_steps=args.max_agent_steps,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(
        render_report(report),
        encoding="utf-8",
    )
    print(
        "runtime_coverage "
        f"acceptance={report['acceptance']['status']} "
        f"clauses={report['scope']['structured_operation_clause_count']} "
        f"steps={report['scope']['smoke_steps']} "
        f"truncated={report['scope']['truncated']}"
    )
    if report["acceptance"]["status"] != "pass":
        raise SystemExit(
            "runtime coverage acceptance failed: "
            + "; ".join(report["acceptance"]["failures"])
        )


if __name__ == "__main__":
    main()
