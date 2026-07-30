# -*- coding: utf-8 -*-
"""Freeze the deterministic card-audit baseline and training-deck closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from collections import Counter, defaultdict, deque
from contextlib import closing
from pathlib import Path
from typing import Iterable, Mapping

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.rl.catalog import TrainableCardCatalog
from swb.rl.fixed_decks import (
    fixed_training_deck_names,
    get_fixed_training_deck,
)
from swb.rl.runtime import hash_rule_directory


SCHEMA_VERSION = 1
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_SOURCE_JSON = Path("shadowverse_cards.json")
DEFAULT_RULES = Path("data/rules")
DEFAULT_AUDIT_START = Path("data/audits/card_bug_audit_start.json")
DEFAULT_CLAUSE_AUDIT = Path("data/audits/rule_clauses.json")
DEFAULT_TOKEN_AUDIT = Path("data/reports/token_audit.json")
DEFAULT_ABILITY_AUDIT = Path("data/reports/ability_audit.json")
DEFAULT_COVERAGE_REPORT = Path("data/reports/rule_coverage.json")
DEFAULT_BASELINE_OUTPUT = Path(
    "data/reports/card_bug_audit/baseline.json"
)
DEFAULT_CLOSURE_OUTPUT = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)

REFERENCE_OPERATION_KINDS = {
    "add_card": "add_to_hand",
    "add_card_to_deck": "add_to_deck",
    "summon": "summon",
    "transform": "transform",
    "transform_deck_cards": "transform_deck_cards",
}


def render_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _run_git(root: Path, *arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.rstrip()


def capture_git_state(root: Path) -> dict[str, object]:
    """Capture path-level workspace state for a new frozen start manifest."""

    status_lines = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    filtered_status = [line.replace("\\", "/") for line in status_lines]

    upstream = _run_git(
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        check=False,
    )
    ahead = None
    behind = None
    if upstream:
        counts = _run_git(
            root,
            "rev-list",
            "--left-right",
            "--count",
            f"HEAD...{upstream}",
        ).split()
        if len(counts) == 2:
            ahead, behind = (int(counts[0]), int(counts[1]))

    return {
        "commit": _run_git(root, "rev-parse", "HEAD"),
        "branch": _run_git(root, "branch", "--show-current"),
        "upstream": upstream or None,
        "ahead": ahead,
        "behind": behind,
        "is_clean": not filtered_status,
        "status_porcelain": filtered_status,
    }


def load_audit_start(
    path: Path,
    *,
    root: Path,
) -> dict[str, object]:
    payload = _load_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported audit-start schema")
    if payload.get("report_kind") != "swb_card_bug_audit_start":
        raise ValueError(f"{path}: invalid audit-start report kind")
    git_state = payload.get("git_state")
    required = {
        "commit",
        "branch",
        "upstream",
        "ahead",
        "behind",
        "is_clean",
        "status_porcelain",
    }
    if not isinstance(git_state, dict) or set(git_state) != required:
        raise ValueError(f"{path}: invalid frozen git_state fields")
    commit = git_state["commit"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(
            f"{path}: invalid frozen audit-start commit"
        )
    try:
        _run_git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    except RuntimeError as error:
        raise ValueError(
            f"{path}: frozen audit-start commit is not available"
        ) from error
    return {
        "manifest_path": _relative(path, root),
        "manifest_sha256": _sha256_file(path),
        **git_state,
    }


def _walk_dicts(value: object) -> Iterable[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_dicts(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_dicts(nested)


def _group_entries(payload: object) -> Iterable[tuple[str, object]]:
    if isinstance(payload, list):
        yield from (("rules", entry) for entry in payload)
        return
    if not isinstance(payload, dict):
        return
    for group, entries in payload.items():
        if isinstance(entries, list):
            yield from ((group, entry) for entry in entries)


def _positive_card_id(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _rule_reference_edges(
    rules_directory: Path,
    root: Path,
) -> tuple[set[int], list[dict[str, object]]]:
    authored_card_ids: set[int] = set()
    edges: list[dict[str, object]] = []
    for path in sorted(rules_directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rule_file = _relative(path, root)
        for group, entry in _group_entries(payload):
            if group == "vanilla_cards":
                vanilla_id = _positive_card_id(entry)
                if vanilla_id is not None:
                    authored_card_ids.add(vanilla_id)
                continue
            if not isinstance(entry, dict):
                continue
            source_id = _positive_card_id(
                entry.get("card_id", entry.get("source_card_id"))
            )
            if source_id is None:
                continue
            authored_card_ids.add(source_id)
            for operation in _walk_dicts(entry):
                kind = operation.get("kind")
                if kind == "replace_deck":
                    target_ids = operation.get("card_ids")
                    if isinstance(target_ids, list):
                        for raw_target_id in target_ids:
                            target_id = _positive_card_id(raw_target_id)
                            if target_id is not None:
                                edges.append(
                                    {
                                        "source_card_id": source_id,
                                        "target_card_id": target_id,
                                        "relation": "replace_deck",
                                        "evidence_path": rule_file,
                                        "rule_group": group,
                                    }
                                )
                    continue
                relation = REFERENCE_OPERATION_KINDS.get(str(kind))
                target_id = _positive_card_id(operation.get("card_id"))
                if relation is not None and target_id is not None:
                    edges.append(
                        {
                            "source_card_id": source_id,
                            "target_card_id": target_id,
                            "relation": relation,
                            "evidence_path": rule_file,
                            "rule_group": group,
                        }
                    )
            if group == "fusions":
                results = entry.get("transform_results")
                if isinstance(results, list):
                    for result in results:
                        if not isinstance(result, dict):
                            continue
                        target_id = _positive_card_id(result.get("card_id"))
                        if target_id is not None:
                            edges.append(
                                {
                                    "source_card_id": source_id,
                                    "target_card_id": target_id,
                                    "relation": "fusion_transform",
                                    "evidence_path": rule_file,
                                    "rule_group": group,
                                }
                            )
    return authored_card_ids, _deduplicate_edges(edges)


def _database_reference_edges(
    database: Path,
    root: Path,
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute(
            """
            SELECT card_id, position, referenced_card_id, referenced_name
            FROM card_references
            WHERE referenced_card_id IS NOT NULL
            ORDER BY card_id, position, referenced_card_id
            """
        )
        for source_id, position, target_id, referenced_name in rows:
            edges.append(
                {
                    "source_card_id": int(source_id),
                    "target_card_id": int(target_id),
                    "relation": "database_reference",
                    "evidence_path": _relative(database, root),
                    "reference_position": int(position),
                    "referenced_name": referenced_name,
                }
            )
    return _deduplicate_edges(edges)


def _edge_sort_key(edge: Mapping[str, object]) -> tuple[object, ...]:
    return (
        edge["source_card_id"],
        edge["target_card_id"],
        edge["relation"],
        edge["evidence_path"],
        edge.get("rule_group", ""),
        edge.get("reference_position", -1),
        edge.get("referenced_name", ""),
    )


def _deduplicate_edges(
    edges: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    unique = {
        json.dumps(edge, ensure_ascii=False, sort_keys=True): edge
        for edge in edges
    }
    return sorted(unique.values(), key=_edge_sort_key)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _rulebook_lookup_succeeds(rulebook: RuleBook, card_id: int) -> bool:
    """Exercise every card-keyed RuleBook lookup used by the runtime."""

    for trigger in Trigger:
        rulebook.operations_for(card_id, trigger)
    rulebook.countdown_for(card_id)
    rulebook.modes_for(card_id)
    rulebook.fusion_for(card_id)
    rulebook.invocation_for(card_id)
    rulebook.activation_for(card_id)
    rulebook.faith_for(card_id)
    rulebook.union_bursts_for(card_id)
    rulebook.listeners_for(card_id)
    rulebook.intrinsic_keywords_for(card_id)
    rulebook.is_explicit_vanilla(card_id)
    rulebook.spellboost_cost_reduction(card_id)
    rulebook.attacks_per_turn(card_id)
    rulebook.cannot_be_played(card_id)
    rulebook.banish_on_leave(card_id)
    rulebook.cannot_be_destroyed_by_effects(card_id)
    rulebook.incoming_damage_replacement(card_id)
    rulebook.forces_enemy_ability_target(card_id)
    rulebook.ignores_ward(card_id)
    rulebook.non_intrinsic_keywords(card_id)
    return True


def _card_record(card: CardDefinition) -> dict[str, object]:
    return {
        "card_id": card.card_id,
        "name": card.name,
        "class_id": card.class_id,
        "class_name": card.class_name,
        "card_type": card.card_type,
        "is_collectible": card.is_collectible,
    }


def build_training_deck_closure(
    *,
    root: Path,
    database: Path,
    rules_directory: Path,
    coverage_report: Path,
    token_audit: Path,
) -> dict[str, object]:
    repository = CardRepository(database)
    cards = {card.card_id: card for card in repository.all_cards()}
    rulebook = RuleBook.from_directory(rules_directory)
    coverage = _load_json(coverage_report)
    classifications = coverage.get("classifications", {})
    if not isinstance(classifications, dict):
        raise ValueError("coverage report has no classifications mapping")
    token_report = _load_json(token_audit)
    token_by_id = {
        int(entry["card_id"]): entry
        for entry in token_report.get("cards", [])
        if isinstance(entry, dict) and "card_id" in entry
    }

    deck_manifests = [
        get_fixed_training_deck(name).manifest()
        for name in fixed_training_deck_names()
    ]
    if len(deck_manifests) != 8:
        raise ValueError(
            f"expected eight fixed training decks, got {len(deck_manifests)}"
        )
    deck_membership: dict[int, list[dict[str, object]]] = defaultdict(list)
    for manifest in deck_manifests:
        counts = Counter(int(card_id) for card_id in manifest["card_ids"])
        for card_id, count in sorted(counts.items()):
            deck_membership[card_id].append(
                {"deck_name": manifest["name"], "copies": count}
            )
    base_ids = sorted(deck_membership)

    authored_ids, rule_edges = _rule_reference_edges(rules_directory, root)
    all_edges = _deduplicate_edges(
        [*_database_reference_edges(database, root), *rule_edges]
    )
    outgoing: dict[int, list[dict[str, object]]] = defaultdict(list)
    incoming: dict[int, list[dict[str, object]]] = defaultdict(list)
    for edge in all_edges:
        outgoing[int(edge["source_card_id"])].append(edge)
        incoming[int(edge["target_card_id"])].append(edge)

    closure_ids = set(base_ids)
    discovery_paths = {card_id: [card_id] for card_id in base_ids}
    queue = deque(base_ids)
    while queue:
        source_id = queue.popleft()
        for edge in outgoing.get(source_id, ()):
            target_id = int(edge["target_card_id"])
            if target_id in closure_ids:
                continue
            closure_ids.add(target_id)
            discovery_paths[target_id] = [
                *discovery_paths[source_id],
                target_id,
            ]
            queue.append(target_id)

    missing_database_ids = sorted(closure_ids - set(cards))
    if missing_database_ids:
        raise ValueError(
            "training closure references cards absent from the database: "
            f"{missing_database_ids}"
        )

    closure_cards: list[dict[str, object]] = []
    unresolved_audits: list[int] = []
    for card_id in sorted(closure_ids):
        card = cards[card_id]
        coverage_entry = classifications.get(str(card_id), {})
        if not isinstance(coverage_entry, dict):
            coverage_entry = {}
        coverage_status = coverage_entry.get("coverage")
        token_entry = token_by_id.get(card_id, {})
        token_status = token_entry.get("category")
        audit_resolution = (
            coverage_status == "covered_exact"
            if card.is_collectible
            else token_status == "entry_behavior_complete"
        )
        lookup_succeeded = _rulebook_lookup_succeeds(rulebook, card_id)
        if not lookup_succeeded or not audit_resolution:
            unresolved_audits.append(card_id)
        closure_cards.append(
            {
                "audit_id": f"card:{card_id}",
                **_card_record(card),
                "origin": (
                    "fixed_training_deck"
                    if card_id in deck_membership
                    else "recursive_reference"
                ),
                "deck_membership": deck_membership.get(card_id, []),
                "discovery_path": discovery_paths[card_id],
                "incoming_references": incoming.get(card_id, []),
                "resolution": {
                    "database": True,
                    "rulebook_lookup_succeeded": lookup_succeeded,
                    "has_authored_rule_entry": card_id in authored_ids,
                    "collectible_coverage": coverage_status,
                    "token_audit_category": token_status,
                    "audit_resolution_passed": audit_resolution,
                },
            }
        )
    if unresolved_audits:
        raise ValueError(
            "training closure contains unresolved RuleBook/audit cards: "
            f"{unresolved_audits}"
        )

    relevant_edges = [
        edge
        for edge in all_edges
        if int(edge["source_card_id"]) in closure_ids
        and int(edge["target_card_id"]) in closure_ids
    ]
    collectible_count = sum(cards[card_id].is_collectible for card_id in closure_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "swb_training_deck_card_closure",
        "inputs": {
            "database": _relative(database, root),
            "rules_directory": _relative(rules_directory, root),
            "coverage_report": _relative(coverage_report, root),
            "token_audit": _relative(token_audit, root),
        },
        "summary": {
            "fixed_deck_count": len(deck_manifests),
            "fixed_deck_collectible_union_count": len(base_ids),
            "closure_card_count": len(closure_ids),
            "recursive_reference_count": len(closure_ids) - len(base_ids),
            "closure_collectible_count": collectible_count,
            "closure_non_collectible_count": len(closure_ids) - collectible_count,
            "all_database_resolved": not missing_database_ids,
            "all_rulebook_and_audit_resolved": not unresolved_audits,
        },
        "fixed_deck_collectible_ids": base_ids,
        "cards": closure_cards,
        "reference_edges": relevant_edges,
    }


def _database_counts(database: Path) -> dict[str, int]:
    with closing(sqlite3.connect(database)) as connection:
        total, collectible, non_collectible = connection.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN cs.is_collectible = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN cs.is_collectible = 0 THEN 1 ELSE 0 END)
            FROM cards c
            JOIN card_sets cs ON cs.id = c.card_set_id
            """
        ).fetchone()
    return {
        "total": int(total),
        "collectible": int(collectible),
        "non_collectible_or_derived": int(non_collectible),
    }


def build_baseline(
    *,
    root: Path,
    audit_start: Path,
    database: Path,
    source_json: Path,
    rules_directory: Path,
    clause_audit: Path,
    token_audit: Path,
    ability_audit: Path,
    coverage_report: Path,
    closure: Mapping[str, object],
) -> dict[str, object]:
    repository = CardRepository(database)
    catalog = TrainableCardCatalog.from_repository(
        repository,
        coverage_report=coverage_report,
    )
    deck_manifests = [
        get_fixed_training_deck(name).manifest()
        for name in fixed_training_deck_names()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "swb_card_bug_audit_baseline",
        "git_audit_start": load_audit_start(audit_start, root=root),
        "database": {
            "path": _relative(database, root),
            "sha256": _sha256_file(database),
            "counts": _database_counts(database),
            "source_snapshot": repository.source_snapshot(),
            "source_json": {
                "path": _relative(source_json, root),
                "sha256": _sha256_file(source_json),
            },
        },
        "audit_artifacts": {
            "rulebook": {
                "path": _relative(rules_directory, root),
                "sha256": hash_rule_directory(rules_directory),
            },
            "clause_audit": {
                "path": _relative(clause_audit, root),
                "sha256": _sha256_file(clause_audit),
            },
            "token_audit": {
                "path": _relative(token_audit, root),
                "sha256": _sha256_file(token_audit),
            },
            "ability_audit": {
                "path": _relative(ability_audit, root),
                "sha256": _sha256_file(ability_audit),
            },
            "coverage_report": {
                "path": _relative(coverage_report, root),
                "sha256": catalog.coverage_report_sha256,
            },
            "catalog": {
                "sha256": catalog.catalog_sha256,
                "card_vocabulary_sha256": catalog.card_vocabulary_sha256,
                "training_pool_sha256": catalog.training_pool_sha256,
                "card_count": len(catalog.cards_by_id),
                "exact_collectible_count": len(catalog.exact_collectible_ids),
            },
        },
        "fixed_training_decks": deck_manifests,
        "training_deck_closure": {
            "path": _relative(DEFAULT_CLOSURE_OUTPUT, root),
            "sha256": hashlib.sha256(render_json(closure).encode("utf-8")).hexdigest(),
            **dict(closure["summary"]),
        },
    }


def build_reports(
    *,
    root: Path,
    audit_start: Path = DEFAULT_AUDIT_START,
    database: Path = DEFAULT_DATABASE,
    source_json: Path = DEFAULT_SOURCE_JSON,
    rules_directory: Path = DEFAULT_RULES,
    clause_audit: Path = DEFAULT_CLAUSE_AUDIT,
    token_audit: Path = DEFAULT_TOKEN_AUDIT,
    ability_audit: Path = DEFAULT_ABILITY_AUDIT,
    coverage_report: Path = DEFAULT_COVERAGE_REPORT,
) -> tuple[dict[str, object], dict[str, object]]:
    closure = build_training_deck_closure(
        root=root,
        database=database,
        rules_directory=rules_directory,
        coverage_report=coverage_report,
        token_audit=token_audit,
    )
    baseline = build_baseline(
        root=root,
        audit_start=audit_start,
        database=database,
        source_json=source_json,
        rules_directory=rules_directory,
        clause_audit=clause_audit,
        token_audit=token_audit,
        ability_audit=ability_audit,
        coverage_report=coverage_report,
        closure=closure,
    )
    return baseline, closure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-start",
        type=Path,
        default=DEFAULT_AUDIT_START,
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--clause-audit", type=Path, default=DEFAULT_CLAUSE_AUDIT)
    parser.add_argument("--token-audit", type=Path, default=DEFAULT_TOKEN_AUDIT)
    parser.add_argument("--ability-audit", type=Path, default=DEFAULT_ABILITY_AUDIT)
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=DEFAULT_COVERAGE_REPORT,
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=DEFAULT_BASELINE_OUTPUT,
    )
    parser.add_argument(
        "--closure-output",
        type=Path,
        default=DEFAULT_CLOSURE_OUTPUT,
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    baseline, closure = build_reports(
        root=root,
        audit_start=args.audit_start,
        database=args.database,
        source_json=args.source_json,
        rules_directory=args.rules,
        clause_audit=args.clause_audit,
        token_audit=args.token_audit,
        ability_audit=args.ability_audit,
        coverage_report=args.coverage_report,
    )
    args.baseline_output.parent.mkdir(parents=True, exist_ok=True)
    args.closure_output.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_output.write_text(render_json(baseline), encoding="utf-8")
    args.closure_output.write_text(render_json(closure), encoding="utf-8")
    print(f"Baseline written to {args.baseline_output}")
    print(f"Training-deck closure written to {args.closure_output}")


if __name__ == "__main__":
    main()
