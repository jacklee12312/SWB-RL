"""Audit combat, damage, endgame, and deterministic RNG contracts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.effects import ConditionType, EffectKind, TargetKind


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_RULES = Path("data/rules")
DEFAULT_CLOSURE = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)
DEFAULT_COVERAGE = Path("data/reports/rule_coverage.json")
DEFAULT_EVIDENCE = Path("data/audits/combat_endgame_random_evidence.json")
DEFAULT_OUTPUT = Path(
    "data/reports/card_bug_audit/combat_endgame_random_audit.json"
)
DEFAULT_MARKDOWN = Path(
    "data/reports/card_bug_audit/combat_endgame_random_audit.md"
)

CATEGORIES = (
    "combat_targeting",
    "combat_timing",
    "super_evolution",
    "damage_and_healing",
    "leave_play",
    "endgame",
    "randomness",
)

COMBAT_TARGETING_KEYWORDS = {
    "守护",
    "疾驰",
    "突进",
    "潜行",
    "威慑",
}
COMBAT_DAMAGE_KEYWORDS = {
    "必杀",
    "毁灭",
    "吸血",
    "虹吸",
    "屏障",
}
DAMAGE_KINDS = {
    EffectKind.DAMAGE_LEADER,
    EffectKind.DAMAGE_UNIT,
    EffectKind.DISTRIBUTE_DAMAGE,
    EffectKind.ADD_LEADER_BARRIER,
    EffectKind.ADD_LEADER_DAMAGE_MODIFIER,
}
HEAL_KINDS = {
    EffectKind.HEAL_LEADER,
    EffectKind.HEAL_UNIT,
    EffectKind.HEAL_UNIT_AND_LEADER,
    EffectKind.SET_LEADER_MAX_HEALTH,
    EffectKind.CHANGE_LEADER_MAX_HEALTH,
}
LEAVE_PLAY_KINDS = {
    EffectKind.DESTROY,
    EffectKind.BANISH,
    EffectKind.BANISH_SAME_NAME,
    EffectKind.RETURN_TO_HAND,
    EffectKind.RETURN_TO_DECK,
    EffectKind.TRANSFORM,
}
RANDOM_KINDS = {
    EffectKind.ADD_RANDOM_KEYWORDS,
    EffectKind.TRANSFORM_BOARD_FROM_RANDOM_OWN_DECK,
    EffectKind.TRANSFORM_HAND_FROM_RANDOM_ENEMY_DECK,
    EffectKind.COPY_RANDOM_ENEMY_DECK_TO_HAND,
    EffectKind.RANDOM_CHOICE,
    EffectKind.RANDOM_DISTRIBUTE,
}

MATRIX_EVIDENCE = {
    "combat_targeting": (
        (
            "tests/test_environment.py",
            "test_guard_blocks_leader_and_other_targets",
        ),
        (
            "tests/test_real_official_basic_completion_twenty_first_batch.py",
            "test_all_real_ward_followers_force_combat_targeting",
        ),
        (
            "tests/test_real_official_complex_completion_twenty_second_batch.py",
            "test_anisage_storm_can_attack_any_target_while_ward_exists",
        ),
        (
            "tests/test_intimidate.py",
            "test_guard_is_inactive_while_same_follower_has_intimidate",
        ),
    ),
    "combat_timing": (
        ("tests/test_triggers.py", "test_attack_trigger_preserves_combat"),
        ("tests/test_triggers.py", "test_clash_triggers_both_sides"),
        (
            "tests/test_triggers.py",
            "test_defender_clash_fires_after_attacker_clash",
        ),
        (
            "tests/test_super_evolution.py",
            "test_attack_trigger_destroy_also_deals_bonus",
        ),
    ),
    "super_evolution": (
        (
            "tests/test_unit_state.py",
            "test_super_evolved_follower_prevents_combat_damage_on_own_turn",
        ),
        (
            "tests/test_unit_state.py",
            "test_super_evolved_follower_prevents_effect_destroy_on_later_own_turn",
        ),
        (
            "tests/test_super_evolution.py",
            "test_combat_destroy_deals_one_to_enemy_leader",
        ),
        (
            "tests/test_unit_state.py",
            "test_super_evolved_follower_can_take_damage_on_opponents_turn",
        ),
    ),
    "damage_and_healing": (
        (
            "tests/test_keywords.py",
            "test_bane_still_destroys_after_barrier_prevents_damage",
        ),
        ("tests/test_keywords.py", "test_drain_overkill_capped"),
        (
            "tests/test_ability_removal_and_leader_modifiers.py",
            "test_stacking_applies_to_effect_combat_and_self_damage_with_floor_zero",
        ),
        (
            "tests/test_real_damage_replacement_binding_tenth_batch.py",
            "test_damage_replacement_threshold_barrier_and_ability_removal",
        ),
        (
            "tests/test_real_apocalypse_deck_batch.py",
            "test_astaroth_sets_and_clamps_max_health_without_damage_then_caps_healing",
        ),
    ),
    "leave_play": (
        ("tests/test_keywords.py", "test_bane_respects_effect_destroy_immunity"),
        ("tests/test_last_words.py", "test_banish_does_not_trigger_last_words"),
        (
            "tests/test_runtime_modifiers.py",
            "test_transform_keeps_entity_id_and_clears_old_state",
        ),
        (
            "tests/test_real_generated_spell_and_follower_chain_batch.py",
            "test_destroy_immunity_does_not_block_banish_or_zero_health_and_removal_disables_it",
        ),
    ),
    "endgame": (
        ("tests/test_environment.py", "test_terminal_reward_belongs_to_actor"),
        (
            "tests/test_turn_timing_p1_corner_cases.py",
            "test_balt_lethal_to_both_leaders_awards_opponent",
        ),
        (
            "tests/test_real_mjerrabaine_deck_batch.py",
            "test_empty_deck_draw_is_official_immediate_defeat_not_fatigue",
        ),
        (
            "tests/test_real_mjerrabaine_deck_batch.py",
            "test_victory_card_result_stops_turn_end_resolution_immediately",
        ),
    ),
    "randomness": (
        (
            "tests/test_combat_endgame_random_audit.py",
            "test_random_choice_is_event_visible_and_full_replay_is_identical",
        ),
        (
            "tests/test_combat_endgame_random_audit.py",
            "test_no_candidate_skip_and_illegal_branch_do_not_consume_rng",
        ),
        (
            "tests/test_official_match_setup.py",
            "test_seeded_random_starting_player_is_reproducible",
        ),
        (
            "tests/test_core_engine.py",
            "test_same_seed_command_replay_fingerprint_matches_real_rules",
        ),
    ),
}

CHECKLIST_CONTRACTS = (
    {
        "contract_id": "attack_follower_leader_ward_and_ignore",
        "tests": MATRIX_EVIDENCE["combat_targeting"]
        + (("tests/test_environment.py", "test_play_and_attack_leader"),),
        "external_evidence_ids": ("SWB-COMBAT-OFFICIAL-001",),
    },
    {
        "contract_id": "attack_clash_damage_and_post_attack_order",
        "tests": MATRIX_EVIDENCE["combat_timing"],
        "external_evidence_ids": ("SWB-COMBAT-OFFICIAL-001",),
    },
    {
        "contract_id": "super_evolution_protection_bonus_and_expiry",
        "tests": MATRIX_EVIDENCE["super_evolution"],
        "external_evidence_ids": ("SWB-COMBAT-OFFICIAL-002",),
    },
    {
        "contract_id": "damage_counter_heal_replacement_and_cap",
        "tests": MATRIX_EVIDENCE["damage_and_healing"],
        "external_evidence_ids": (
            "SWB-COMBAT-OFFICIAL-001",
            "SWB-COMBAT-OFFICIAL-003",
        ),
    },
    {
        "contract_id": "bane_destroy_banish_transform_and_zero_health",
        "tests": MATRIX_EVIDENCE["leave_play"],
        "external_evidence_ids": ("SWB-COMBAT-OFFICIAL-001",),
    },
    {
        "contract_id": "single_dual_deck_and_special_victory",
        "tests": MATRIX_EVIDENCE["endgame"],
        "external_evidence_ids": (
            "SWB-COMBAT-OFFICIAL-001",
            "SWB-COMBAT-OFFICIAL-004",
            "SWB-COMBAT-OFFICIAL-005",
        ),
    },
    {
        "contract_id": "game_over_stops_queue_and_future_commands",
        "tests": (
            (
                "tests/test_turn_timing_p1_corner_cases.py",
                "test_super_evolution_lethal_stops_mimi_counter_damage",
            ),
            (
                "tests/test_turn_timing_p1_corner_cases.py",
                "test_super_evolution_lethal_stops_coco_last_words_heal",
            ),
            (
                "tests/test_emblems_advanced.py",
                "test_game_end_stops_remaining_emblem_batch",
            ),
            (
                "tests/test_combat_endgame_random_audit.py",
                "test_terminated_match_rejects_commands_without_visible_mutation",
            ),
        ),
        "external_evidence_ids": ("SWB-COMBAT-OFFICIAL-004",),
    },
    {
        "contract_id": "engine_rng_and_event_choice_evidence",
        "tests": (
            (
                "tests/test_combat_endgame_random_audit.py",
                "test_engine_random_callsites_use_owned_rng",
            ),
            (
                "tests/test_combat_endgame_random_audit.py",
                "test_random_choice_is_event_visible_and_full_replay_is_identical",
            ),
            (
                "tests/test_real_turn_end_draw_random_seventeenth_batch.py",
                "test_random_and_draw_events_are_explicit",
            ),
        ),
        "external_evidence_ids": (),
    },
    {
        "contract_id": "same_seed_fingerprint_events_and_winner",
        "tests": (
            (
                "tests/test_combat_endgame_random_audit.py",
                "test_random_choice_is_event_visible_and_full_replay_is_identical",
            ),
            (
                "tests/test_real_official_basic_completion_twenty_first_batch.py",
                "test_same_seed_and_command_sequence_reproduces_identical_state",
            ),
        ),
        "external_evidence_ids": (),
    },
    {
        "contract_id": "no_candidate_skip_and_illegal_rng_neutrality",
        "tests": (
            (
                "tests/test_combat_endgame_random_audit.py",
                "test_no_candidate_skip_and_illegal_branch_do_not_consume_rng",
            ),
            (
                "tests/test_conditions.py",
                "test_mixed_all_condition_does_not_block_play_or_consume_rng",
            ),
            (
                "tests/test_real_apocalypse_deck_batch.py",
                "test_illegal_source_play_preserves_state_rng_events_and_logs",
            ),
        ),
        "external_evidence_ids": (),
    },
)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> object:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk(value: object) -> Iterable[object]:
    yield value
    if is_dataclass(value):
        for field in fields(value):
            yield from _walk(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _walk(item)


def _categories_for(value: object) -> set[str]:
    categories: set[str] = set()
    for item in _walk(value):
        if isinstance(item, EffectKind):
            if item in DAMAGE_KINDS or item in HEAL_KINDS:
                categories.add("damage_and_healing")
            if item in LEAVE_PLAY_KINDS:
                categories.add("leave_play")
            if item is EffectKind.SET_EMPTY_DECK_OUTCOME:
                categories.add("endgame")
            if item is EffectKind.SUPER_EVOLVE_UNIT:
                categories.add("super_evolution")
            if item in RANDOM_KINDS:
                categories.add("randomness")
        elif isinstance(item, TargetKind):
            if item.value.startswith("random_"):
                categories.add("randomness")
            if item is TargetKind.ATTACK_TARGET:
                categories.add("combat_timing")
        elif isinstance(item, ConditionType):
            if item in {
                ConditionType.ATTACK_TARGET_EXISTS,
                ConditionType.CONTROLLER_FOLLOWER_ATTACKS_THIS_TURN_AT_MOST,
            }:
                categories.add("combat_timing")
            if item in {
                ConditionType.SOURCE_SUPER_EVOLVED,
                ConditionType.CONTROLLER_SUPER_EVOLUTION_UNLOCKED,
                ConditionType.OPPONENT_SUPER_EVOLUTION_UNLOCKED,
            }:
                categories.add("super_evolution")
        elif isinstance(item, str):
            if item in COMBAT_TARGETING_KEYWORDS:
                categories.add("combat_targeting")
            if item in COMBAT_DAMAGE_KEYWORDS:
                categories.add("damage_and_healing")
            if item == "ignores_ward":
                categories.add("combat_targeting")
            if item in {
                "incoming_damage_replacement",
                "cannot_be_destroyed_by_effects",
            }:
                categories.add("damage_and_healing")
            if item == "banish_on_leave":
                categories.add("leave_play")
    return categories


def _add_source(
    sources: dict[int, list[dict[str, str]]],
    card_id: int,
    root: str,
    value: object,
    extra_categories: Iterable[str] = (),
) -> None:
    categories = _categories_for(value)
    categories.update(extra_categories)
    for category in sorted(categories):
        sources[card_id].append({"category": category, "root": root})


def _source_records(
    cards: tuple[CardDefinition, ...],
    rulebook: RuleBook,
) -> dict[int, list[dict[str, str]]]:
    sources: dict[int, list[dict[str, str]]] = defaultdict(list)
    for card in cards:
        _add_source(sources, card.card_id, "database", card)
    for (card_id, trigger), operations in rulebook._rules.items():
        extra = ()
        if trigger in {Trigger.ATTACK, Trigger.CLASH}:
            extra = ("combat_timing",)
        elif trigger is Trigger.SUPER_EVOLVE:
            extra = ("super_evolution",)
        _add_source(
            sources,
            card_id,
            f"rule:{trigger.value}",
            operations,
            extra,
        )
    mappings = (
        ("passives", rulebook._passives),
        ("play_modes", rulebook._play_modes),
        ("fusion", rulebook._fusion_defs),
        ("activation", rulebook._activation_defs),
        ("faith", rulebook._faith_defs),
        ("union_burst", rulebook._union_burst_defs),
        ("listeners", rulebook._listener_defs),
    )
    for root, mapping in mappings:
        for card_id, value in mapping.items():
            _add_source(sources, card_id, root, value)
    for definition in rulebook._emblem_defs.values():
        _add_source(
            sources,
            definition.source_card_id,
            f"emblem:{definition.emblem_id}",
            definition,
        )
    return {
        card_id: sorted(
            records,
            key=lambda row: (row["category"], row["root"]),
        )
        for card_id, records in sorted(sources.items())
        if records
    }


def _coverage_entry(
    coverage: Mapping[str, object],
    card_id: int,
) -> Mapping[str, object]:
    classifications = coverage.get("classifications", {})
    if not isinstance(classifications, Mapping):
        return {}
    entry = classifications.get(str(card_id), {})
    return entry if isinstance(entry, Mapping) else {}


def _test_evidence(entry: Mapping[str, object]) -> list[str]:
    for key in ("clause_audit", "rule_metadata"):
        nested = entry.get(key, {})
        if not isinstance(nested, Mapping):
            continue
        evidence = nested.get("test_evidence", [])
        if isinstance(evidence, list) and evidence:
            return sorted(str(path) for path in evidence)
    return []


def _test_reference_status(path: str, test_name: str) -> dict[str, object]:
    full_path = _repo_path(Path(path))
    exists = full_path.is_file()
    text = full_path.read_text(encoding="utf-8") if exists else ""
    test_exists = f"def {test_name}(" in text
    return {
        "path": path,
        "test_name": test_name,
        "file_exists": exists,
        "test_exists": test_exists,
        "passed": exists and test_exists,
    }


def _inventory(
    cards: tuple[CardDefinition, ...],
    sources: Mapping[int, list[dict[str, str]]],
    closure_ids: set[int],
    coverage: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    issues: list[str] = []
    for card in cards:
        records = sources.get(card.card_id, [])
        if not records:
            continue
        entry = _coverage_entry(coverage, card.card_id)
        evidence = _test_evidence(entry)
        accepted = (
            {"covered_exact"}
            if card.is_collectible
            else {"token_or_non_collectible"}
        )
        row_issues: list[str] = []
        if entry.get("coverage") not in accepted:
            row_issues.append("card lacks required exact/generated coverage")
        if not evidence:
            row_issues.append("card lacks permanent test evidence")
        for path in evidence:
            if not _repo_path(Path(path)).is_file():
                row_issues.append(f"missing test evidence file: {path}")
        categories = sorted(
            {record["category"] for record in records},
            key=CATEGORIES.index,
        )
        if row_issues:
            issues.extend(
                f"card {card.card_id}: {issue}" for issue in row_issues
            )
        rows.append(
            {
                "card_id": card.card_id,
                "name": card.name,
                "collectible": card.is_collectible,
                "training_closure": card.card_id in closure_ids,
                "categories": categories,
                "record_count": len(records),
                "roots": sorted({record["root"] for record in records}),
                "test_evidence": evidence,
                "issues": row_issues,
                "passed": not row_issues,
            }
        )
    return rows, sorted(set(issues))


def _matrix(inventory: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category in CATEGORIES:
        category_sources = [
            row for row in inventory if category in row["categories"]
        ]
        evidence = [
            _test_reference_status(path, test_name)
            for path, test_name in MATRIX_EVIDENCE[category]
        ]
        rows.append(
            {
                "category": category,
                "source_card_count": len(category_sources),
                "collectible_source_count": sum(
                    bool(row["collectible"]) for row in category_sources
                ),
                "generated_source_count": sum(
                    not bool(row["collectible"]) for row in category_sources
                ),
                "training_source_count": sum(
                    bool(row["training_closure"]) for row in category_sources
                ),
                "test_evidence": evidence,
                "passed": bool(category_sources)
                and all(bool(row["passed"]) for row in category_sources)
                and all(bool(item["passed"]) for item in evidence),
            }
        )
    return rows


RANDOM_METHODS = {
    "choice",
    "choices",
    "randint",
    "randrange",
    "sample",
    "shuffle",
}


class _RandomCallVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.functions: list[str] = []
        self.callsites: list[dict[str, object]] = []
        self.violations: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        function = self.functions[-1] if self.functions else "<module>"
        call = node.func
        if isinstance(call, ast.Attribute) and call.attr in RANDOM_METHODS:
            owner = ast.unparse(call.value)
            allowed = (
                owner == "self.random"
                or (
                    self.path.name == "targeting.py"
                    and owner == "rng"
                    and function in {
                        "pick_random",
                        "pick_random_graveyard",
                    }
                )
            )
            row = {
                "path": self.path.relative_to(ROOT).as_posix(),
                "line": node.lineno,
                "function": function,
                "owner": owner,
                "method": call.attr,
                "evidence": (
                    "seeded state/fingerprint"
                    if function in {"reset", "_mulligan"}
                    else "transition event and resulting state"
                ),
                "passed": allowed,
            }
            self.callsites.append(row)
            if not allowed:
                self.violations.append(
                    f"{row['path']}:{node.lineno} uses {owner}.{call.attr}"
                )
        elif isinstance(call, ast.Name) and call.id in RANDOM_METHODS:
            self.violations.append(
                f"{self.path.relative_to(ROOT).as_posix()}:{node.lineno} "
                f"calls unowned {call.id}"
            )
        self.generic_visit(node)


def _engine_rng_audit() -> dict[str, object]:
    callsites: list[dict[str, object]] = []
    violations: list[str] = []
    for path in sorted((ROOT / "swb/engine").glob("*.py")):
        visitor = _RandomCallVisitor(path)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        callsites.extend(visitor.callsites)
        violations.extend(visitor.violations)
    return {
        "audited_path": "swb/engine/*.py",
        "callsite_count": len(callsites),
        "callsites": sorted(
            callsites,
            key=lambda row: (str(row["path"]), int(row["line"])),
        ),
        "violation_count": len(violations),
        "violations": sorted(set(violations)),
        "passed": bool(callsites) and not violations,
    }


def _contracts(
    evidence_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    evidence_ids = {
        str(row["evidence_id"])
        for row in evidence_rows
        if isinstance(row, Mapping) and "evidence_id" in row
    }
    rows: list[dict[str, object]] = []
    for definition in CHECKLIST_CONTRACTS:
        tests = [
            _test_reference_status(path, test_name)
            for path, test_name in definition["tests"]
        ]
        missing_external = sorted(
            set(definition["external_evidence_ids"]) - evidence_ids
        )
        rows.append(
            {
                "contract_id": definition["contract_id"],
                "test_evidence": tests,
                "external_evidence_ids": list(
                    definition["external_evidence_ids"]
                ),
                "missing_external_evidence_ids": missing_external,
                "passed": all(bool(row["passed"]) for row in tests)
                and not missing_external,
            }
        )
    return rows


def build_report(
    *,
    database_path: Path = DEFAULT_DATABASE,
    rules_path: Path = DEFAULT_RULES,
    closure_path: Path = DEFAULT_CLOSURE,
    coverage_path: Path = DEFAULT_COVERAGE,
    evidence_path: Path = DEFAULT_EVIDENCE,
) -> dict[str, object]:
    repository = CardRepository(_repo_path(database_path))
    cards = tuple(repository.all_cards())
    rulebook = RuleBook.from_directory(_repo_path(rules_path))
    closure = _load_json(closure_path)
    coverage = _load_json(coverage_path)
    evidence = _load_json(evidence_path)
    if not isinstance(closure, Mapping):
        raise ValueError("closure report must be an object")
    if not isinstance(coverage, Mapping):
        raise ValueError("coverage report must be an object")
    if not isinstance(evidence, Mapping):
        raise ValueError("official evidence must be an object")
    closure_rows = closure.get("cards", [])
    if not isinstance(closure_rows, list):
        raise ValueError("closure cards must be a list")
    closure_ids = {
        int(row["card_id"])
        for row in closure_rows
        if isinstance(row, Mapping) and "card_id" in row
    }
    evidence_rows = evidence.get("sources", [])
    if not isinstance(evidence_rows, list):
        raise ValueError("official evidence sources must be a list")
    evidence_issues = [
        f"official evidence row {index} is incomplete"
        for index, row in enumerate(evidence_rows)
        if not isinstance(row, Mapping)
        or not row.get("evidence_id")
        or not row.get("authority")
        or not str(row.get("url", "")).startswith(
            "https://shadowverse-wb.com/"
        )
        or not row.get("conclusion")
    ]
    sources = _source_records(cards, rulebook)
    inventory, inventory_issues = _inventory(
        cards,
        sources,
        closure_ids,
        coverage,
    )
    matrix = _matrix(inventory)
    contracts = _contracts(evidence_rows)
    rng_audit = _engine_rng_audit()
    failures = list(inventory_issues) + evidence_issues
    failures.extend(
        f"matrix {row['category']} failed"
        for row in matrix
        if not row["passed"]
    )
    failures.extend(
        f"contract {row['contract_id']} failed"
        for row in contracts
        if not row["passed"]
    )
    if not rng_audit["passed"]:
        failures.extend(rng_audit["violations"])
        if not rng_audit["callsites"]:
            failures.append("engine RNG audit found no callsites")
    return {
        "schema_version": 1,
        "report_kind": "swb_combat_endgame_random_audit",
        "inputs": {
            "database": {
                "path": database_path.as_posix(),
                "sha256": _sha256(database_path),
            },
            "rules": {
                "path": rules_path.as_posix(),
                "json_file_count": len(
                    tuple(_repo_path(rules_path).rglob("*.json"))
                ),
            },
            "closure": {
                "path": closure_path.as_posix(),
                "sha256": _sha256(closure_path),
            },
            "coverage": {
                "path": coverage_path.as_posix(),
                "sha256": _sha256(coverage_path),
            },
            "official_evidence": {
                "path": evidence_path.as_posix(),
                "sha256": _sha256(evidence_path),
            },
        },
        "scope": {
            "card_count": len(cards),
            "collectible_card_count": sum(
                card.is_collectible for card in cards
            ),
            "generated_card_count": sum(
                not card.is_collectible for card in cards
            ),
            "training_closure_card_count": len(closure_ids),
        },
        "summary": {
            "source_card_count": len(inventory),
            "training_source_card_count": sum(
                bool(row["training_closure"]) for row in inventory
            ),
            "category_count": len(matrix),
            "behavior_contract_count": len(contracts),
            "official_evidence_count": len(evidence_rows),
            "rng_callsite_count": rng_audit["callsite_count"],
            "failure_count": len(failures),
            "failures": failures,
            "passed": not failures,
        },
        "official_evidence": evidence_rows,
        "category_matrix": matrix,
        "behavior_contracts": contracts,
        "engine_rng_audit": rng_audit,
        "inventory": inventory,
    }


def render_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"


def render_markdown(report: Mapping[str, object]) -> str:
    scope = report["scope"]
    summary = report["summary"]
    lines = [
        "# Combat, Damage, Endgame, and RNG Audit",
        "",
        "This report preserves the full-pool source inventory and executable "
        "contracts for checklist section 1.9.",
        "",
        "## Summary",
        "",
        f"- Cards: {scope['card_count']} "
        f"({scope['collectible_card_count']} collectible, "
        f"{scope['generated_card_count']} generated)",
        f"- Training closure: {scope['training_closure_card_count']}",
        f"- Relevant source cards: {summary['source_card_count']}",
        f"- Training source cards: {summary['training_source_card_count']}",
        f"- Engine RNG callsites: {summary['rng_callsite_count']}",
        f"- Behavioral contracts: {summary['behavior_contract_count']}",
        f"- Failures: {summary['failure_count']}",
        f"- Result: {'PASS' if summary['passed'] else 'FAIL'}",
        "",
        "## Official evidence",
        "",
        "| Evidence | Authority | Conclusion |",
        "|---|---|---|",
    ]
    for row in report["official_evidence"]:
        lines.append(
            f"| [{row['evidence_id']}]({row['url']}) | "
            f"{row['authority']} | {row['summary']} |"
        )
    lines.extend(
        [
            "",
            "## Category matrix",
            "",
            "| Category | Sources | Collectible | Generated | Training | Result |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in report["category_matrix"]:
        lines.append(
            f"| {row['category']} | {row['source_card_count']} | "
            f"{row['collectible_source_count']} | "
            f"{row['generated_source_count']} | "
            f"{row['training_source_count']} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Behavioral contracts",
            "",
            "| Contract | Tests | Official evidence | Result |",
            "|---|---:|---:|:---:|",
        ]
    )
    for row in report["behavior_contracts"]:
        lines.append(
            f"| {row['contract_id']} | {len(row['test_evidence'])} | "
            f"{len(row['external_evidence_ids'])} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    rng = report["engine_rng_audit"]
    lines.extend(
        [
            "",
            "## Engine RNG audit",
            "",
            f"- Audited: `{rng['audited_path']}`",
            f"- Callsites: {rng['callsite_count']}",
            f"- Violations: {rng['violation_count']}",
            f"- Result: {'PASS' if rng['passed'] else 'FAIL'}",
            "",
            "## Source inventory",
            "",
            "| Card | Categories | Records | Training | Result |",
            "|---|---|---:|:---:|:---:|",
        ]
    )
    for row in report["inventory"]:
        lines.append(
            f"| {row['card_id']} {row['name']} | "
            f"{', '.join(row['categories'])} | {row['record_count']} | "
            f"{'yes' if row['training_closure'] else 'no'} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    if summary["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in summary["failures"])
    return "\n".join(lines) + "\n"


def write_report(
    report: Mapping[str, object],
    output_path: Path,
    markdown_path: Path,
) -> None:
    output = _repo_path(output_path)
    markdown = _repo_path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_json(report), encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report(
        database_path=args.database,
        rules_path=args.rules,
        closure_path=args.closure,
        coverage_path=args.coverage,
        evidence_path=args.evidence,
    )
    write_report(report, args.output, args.markdown)
    summary = report["summary"]
    print(
        f"cards={report['scope']['card_count']} "
        f"sources={summary['source_card_count']} "
        f"training_sources={summary['training_source_card_count']} "
        f"rng_callsites={summary['rng_callsite_count']} "
        f"contracts={summary['behavior_contract_count']} "
        f"failures={summary['failure_count']} "
        f"passed={summary['passed']}"
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
