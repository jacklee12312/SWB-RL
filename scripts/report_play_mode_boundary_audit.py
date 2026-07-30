# -*- coding: utf-8 -*-
"""Generate deterministic full-pool play-mode cost-boundary evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import (
    CardRule,
    PlayModeDefinition,
    RuleBook,
    Trigger,
)
from swb.engine.commands import PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import Amulet, CostModifier, HandCard, Unit
from swb.rl.runtime import hash_rule_directory


SCHEMA_VERSION = 1
MAX_REMAINING_PP = 10
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_RULES = Path("data/rules")
DEFAULT_CLOSURE = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/card_bug_audit/play_mode_boundary_audit.json"
)
DEFAULT_MARKDOWN = Path(
    "data/reports/card_bug_audit/play_mode_boundary_audit.md"
)

FILLER_CARD = CardDefinition(
    card_id=99000001,
    card_set_id=99000,
    class_id=0,
    class_name="中立",
    name="play-mode boundary filler",
    cost=1,
    card_type="随从",
    attack=1,
    life=1,
    keywords=frozenset(),
    support_level="IMPLEMENTED",
    is_collectible=True,
)
FILLER_CARDS = tuple(
    replace(
        FILLER_CARD,
        card_id=FILLER_CARD.card_id + index,
        name=f"play-mode boundary filler {index}",
    )
    for index in range(40)
)
FILLER_BY_ID = {card.card_id: card for card in FILLER_CARDS}

MODIFIER_SCENARIOS = (
    {
        "id": "printed",
        "modifier": None,
        "duration": None,
    },
    {
        "id": "temporary_discount",
        "modifier": "discount",
        "duration": "until_end_of_turn",
    },
    {
        "id": "permanent_discount",
        "modifier": "discount",
        "duration": "permanent",
    },
    {
        "id": "temporary_increase",
        "modifier": "increase",
        "duration": "until_end_of_turn",
    },
    {
        "id": "permanent_increase",
        "modifier": "increase",
        "duration": "permanent",
    },
)


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


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _scenario_current_cost(printed_cost: int, scenario: Mapping[str, object]) -> int:
    modifier = scenario["modifier"]
    if modifier is None:
        return printed_cost
    if modifier == "discount":
        return max(0, printed_cost - 1)
    if modifier == "increase":
        return printed_cost + 1
    raise ValueError(f"unknown modifier scenario {modifier!r}")


def _boundary_points(
    *,
    printed_cost: int,
    current_cost: int,
    modes: Sequence[PlayModeDefinition],
) -> tuple[list[dict[str, object]], list[str]]:
    labels_by_pp: dict[int, set[str]] = defaultdict(set)
    unavailable: list[str] = []

    def add(label: str, value: int) -> None:
        if 0 <= value <= MAX_REMAINING_PP:
            labels_by_pp[value].add(label)
        else:
            unavailable.append(label)

    add("zero", 0)
    add("printed_cost_minus_one", printed_cost - 1)
    add("printed_cost_exact", printed_cost)
    add("above_printed_cost", printed_cost + 1)
    add("current_cost_minus_one", current_cost - 1)
    add("current_cost_exact", current_cost)
    add("above_current_cost", current_cost + 1)
    for mode in modes:
        add(f"{mode.mode_id}_cost_minus_one", mode.cost - 1)
        add(f"{mode.mode_id}_cost_exact", mode.cost)

    return (
        [
            {
                "remaining_pp": remaining_pp,
                "boundary_labels": sorted(labels),
            }
            for remaining_pp, labels in sorted(labels_by_pp.items())
        ],
        sorted(set(unavailable)),
    )


def _expected_mode_ids(
    *,
    current_cost: int,
    remaining_pp: int,
    modes: Sequence[PlayModeDefinition],
) -> list[str]:
    affordable_enhances = [
        mode
        for mode in modes
        if mode.is_enhance and mode.cost <= remaining_pp
    ]
    if affordable_enhances:
        highest_cost = max(mode.cost for mode in affordable_enhances)
        selected = next(
            mode
            for mode in modes
            if mode.is_enhance and mode.cost == highest_cost
        )
        return [selected.mode_id]

    expected: list[str] = []
    if current_cost <= remaining_pp:
        expected.append("normal")
    expected.extend(
        mode.mode_id
        for mode in modes
        if (
            (mode.is_accelerate or mode.is_crystallize)
            and mode.cost <= remaining_pp
            and current_cost > remaining_pp
        )
    )
    return expected


def _cost_only_rulebook(
    card_id: int,
    modes: Sequence[PlayModeDefinition],
    card_type: str,
) -> RuleBook:
    cost_only_modes = tuple(
        replace(mode, conditions=(), operations=())
        for mode in modes
    )
    rules = ()
    if card_type in {"法术", "护符"}:
        rules = (
            CardRule(
                card_id=card_id,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DRAW,
                        target=TargetKind.OWN_LEADER,
                        amount=0,
                    ),
                ),
            ),
        )
    return RuleBook(
        rules=rules,
        play_modes={card_id: cost_only_modes},
    )


def _build_environment(
    *,
    definition: CardDefinition,
    modes: Sequence[PlayModeDefinition],
    cards_by_id: Mapping[int, CardDefinition],
    scenario: Mapping[str, object],
    remaining_pp: int,
    seed: int,
    full_board: bool = False,
) -> tuple[ShadowverseEnv, HandCard]:
    rulebook = _cost_only_rulebook(
        definition.card_id,
        modes,
        definition.card_type,
    )

    def resolve(card_id: int) -> CardDefinition | None:
        return FILLER_BY_ID.get(card_id) or cards_by_id.get(card_id)

    deck = list(FILLER_CARDS)
    player_class_id = (
        definition.class_id if definition.class_id in range(1, 8) else 1
    )
    env = ShadowverseEnv(
        deck,
        deck,
        class_a=player_class_id,
        class_b=player_class_id,
        seed=seed,
        rulebook=rulebook,
        card_resolver=resolve,
        validate_invariants=True,
    )
    env.reset(seed=seed)
    for player in env.players:
        player.hand.clear()
        player.hand_entity_ids.clear()
        player.board.clear()
        player.graveyard.clear()
        player.deck.clear()
    player = env.players[0]
    player.max_mana = MAX_REMAINING_PP
    player.mana = remaining_pp
    player.cards_played_this_turn = 0

    hand_card = HandCard(
        definition=definition,
        entity_id=env.core.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    modifier = scenario["modifier"]
    if modifier is not None:
        hand_card.cost_modifiers.append(
            CostModifier(
                modifier_id=definition.card_id * 10
                + list(MODIFIER_SCENARIOS).index(scenario),
                mode="set",
                amount=_scenario_current_cost(definition.cost, scenario),
                duration=str(scenario["duration"]),
            )
        )
    player.hand.append(hand_card)
    player.hand_entity_ids.append(hand_card.entity_id)

    if full_board:
        for _ in range(env.core.config.max_board):
            player.board.append(
                Unit.summon(
                    FILLER_CARD,
                    entity_id=env.core.state.allocate_entity_id(),
                )
            )
    env.invalidate_cache(reason="play-mode boundary audit setup")
    env.core.assert_invariants()
    return env, hand_card


def _play_mode_ids(commands: Sequence[object]) -> list[str]:
    return sorted(
        command.mode_id
        for command in commands
        if isinstance(command, PlayCard) and command.hand_index == 0
    )


def _mask_mode_ids(env: ShadowverseEnv) -> list[str]:
    commands = []
    for action, allowed in enumerate(env.action_mask()):
        if not allowed:
            continue
        command = env._decode_action(action)
        if isinstance(command, PlayCard) and command.hand_index == 0:
            commands.append(command)
    return _play_mode_ids(commands)


def _route_expected_zone(
    definition: CardDefinition,
    mode: PlayModeDefinition | None,
) -> str:
    if mode is not None and mode.is_accelerate:
        return "graveyard"
    if mode is not None and mode.is_crystallize:
        return "board_amulet"
    if definition.card_type == "法术":
        return "graveyard"
    if definition.card_type == "护符":
        return "board_amulet"
    return "board_unit"


def _entity_zone(
    env: ShadowverseEnv,
    entity_id: int,
    card_id: int,
) -> str:
    if any(
        entity.entity_id == entity_id
        or entity.definition.card_id == card_id
        for entity in env.players[0].graveyard
    ):
        return "graveyard"
    for entity in env.players[0].board:
        if (
            entity.entity_id != entity_id
            and entity.definition.card_id != card_id
        ):
            continue
        if isinstance(entity, Amulet):
            return "board_amulet"
        if isinstance(entity, Unit):
            return "board_unit"
    return "missing"


def _scan_case(
    *,
    definition: CardDefinition,
    modes: Sequence[PlayModeDefinition],
    cards_by_id: Mapping[int, CardDefinition],
    scenario: Mapping[str, object],
    point: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    remaining_pp = int(point["remaining_pp"])
    env, hand_card = _build_environment(
        definition=definition,
        modes=modes,
        cards_by_id=cards_by_id,
        scenario=scenario,
        remaining_pp=remaining_pp,
        seed=seed,
    )
    current_cost = hand_card.current_cost
    expected = _expected_mode_ids(
        current_cost=current_cost,
        remaining_pp=remaining_pp,
        modes=modes,
    )
    legal = _play_mode_ids(env.core.legal_commands())
    masked = _mask_mode_ids(env)

    mode_by_id = {mode.mode_id: mode for mode in modes}
    all_routes = ["normal", *(mode.mode_id for mode in modes)]
    illegal_atomicity: dict[str, bool] = {}
    for mode_id in all_routes:
        if mode_id in expected:
            continue
        before = env.core.deterministic_fingerprint()
        rejected = False
        try:
            env.core.apply(PlayCard(0, 0, mode_id=mode_id))
        except IllegalCommand:
            rejected = True
        illegal_atomicity[mode_id] = (
            rejected
            and env.core.deterministic_fingerprint() == before
        )

    execution: dict[str, object] | None = None
    if len(expected) == 1:
        mode_id = expected[0]
        mode = mode_by_id.get(mode_id)
        expected_cost = current_cost if mode is None else mode.cost
        expected_zone = _route_expected_zone(definition, mode)
        combo_before = env.players[0].cards_played_this_turn
        rejection = None
        try:
            env.core.apply(PlayCard(0, 0, mode_id=mode_id))
        except IllegalCommand as error:
            rejection = str(error)
        actual_zone = _entity_zone(
            env,
            hand_card.entity_id,
            definition.card_id,
        )
        execution = {
            "mode_id": mode_id,
            "expected_cost": expected_cost,
            "mana_after": env.players[0].mana,
            "expected_mana_after": remaining_pp - expected_cost,
            "expected_zone": expected_zone,
            "actual_zone": actual_zone,
            "combo_delta": (
                env.players[0].cards_played_this_turn - combo_before
            ),
            "rejection": rejection,
            "passed": (
                rejection is None
                and env.players[0].mana == remaining_pp - expected_cost
                and actual_zone == expected_zone
                and env.players[0].cards_played_this_turn - combo_before == 1
            ),
        }

    command_mask_match = expected == legal == masked
    atomic = all(illegal_atomicity.values())
    execution_passed = execution is None or bool(execution["passed"])
    return {
        "remaining_pp": remaining_pp,
        "boundary_labels": point["boundary_labels"],
        "printed_cost": definition.cost,
        "current_cost": current_cost,
        "expected_mode_ids": expected,
        "legal_command_mode_ids": legal,
        "action_mask_mode_ids": masked,
        "command_mask_match": command_mask_match,
        "illegal_atomicity": illegal_atomicity,
        "illegal_atomicity_passed": atomic,
        "execution": execution,
        "passed": command_mask_match and atomic and execution_passed,
    }


def _scan_full_board(
    *,
    definition: CardDefinition,
    modes: Sequence[PlayModeDefinition],
    cards_by_id: Mapping[int, CardDefinition],
    mode: PlayModeDefinition,
    seed: int,
) -> dict[str, object]:
    scenario = MODIFIER_SCENARIOS[0]
    env, _ = _build_environment(
        definition=definition,
        modes=modes,
        cards_by_id=cards_by_id,
        scenario=scenario,
        remaining_pp=mode.cost,
        seed=seed,
        full_board=True,
    )
    legal = _play_mode_ids(env.core.legal_commands())
    masked = _mask_mode_ids(env)
    effective_type = (
        "法术"
        if mode.is_accelerate
        else "护符"
        if mode.is_crystallize
        else definition.card_type
    )
    expected_legal = effective_type == "法术"
    present = mode.mode_id in legal
    masked_present = mode.mode_id in masked
    before = env.core.deterministic_fingerprint()
    rejected_atomically = True
    if not expected_legal:
        try:
            env.core.apply(PlayCard(0, 0, mode_id=mode.mode_id))
            rejected_atomically = False
        except IllegalCommand:
            rejected_atomically = (
                env.core.deterministic_fingerprint() == before
            )
    return {
        "mode_id": mode.mode_id,
        "remaining_pp": mode.cost,
        "effective_card_type": effective_type,
        "expected_legal": expected_legal,
        "command_legal": present,
        "action_mask_legal": masked_present,
        "illegal_atomicity_passed": rejected_atomically,
        "passed": (
            present == expected_legal
            and masked_present == expected_legal
            and rejected_atomically
        ),
    }


def build_report(
    *,
    root: Path,
    database: Path,
    rules_directory: Path,
    closure_path: Path,
) -> dict[str, object]:
    repository = CardRepository(database)
    all_cards = repository.all_cards()
    cards_by_id = {card.card_id: card for card in all_cards}
    rulebook = RuleBook.from_directory(rules_directory)
    closure = _load_json(closure_path)
    closure_ids = {
        int(card["card_id"])
        for card in closure.get("cards", [])
        if isinstance(card, dict) and "card_id" in card
    }

    mode_cards = [
        card
        for card in all_cards
        if rulebook.modes_for(card.card_id)
    ]
    card_reports: list[dict[str, object]] = []
    failure_count = 0
    case_count = 0
    for card_index, definition in enumerate(mode_cards):
        modes = rulebook.modes_for(definition.card_id)
        scenarios: list[dict[str, object]] = []
        unavailable_boundaries: set[str] = set()
        for scenario_index, scenario in enumerate(MODIFIER_SCENARIOS):
            current_cost = _scenario_current_cost(
                definition.cost,
                scenario,
            )
            points, unavailable = _boundary_points(
                printed_cost=definition.cost,
                current_cost=current_cost,
                modes=modes,
            )
            unavailable_boundaries.update(unavailable)
            cases = [
                _scan_case(
                    definition=definition,
                    modes=modes,
                    cards_by_id=cards_by_id,
                    scenario=scenario,
                    point=point,
                    seed=(
                        definition.card_id * 100
                        + scenario_index * 11
                        + int(point["remaining_pp"])
                    ),
                )
                for point in points
            ]
            case_count += len(cases)
            failure_count += sum(not case["passed"] for case in cases)
            scenarios.append(
                {
                    **scenario,
                    "current_cost": current_cost,
                    "cases": cases,
                    "passed": all(case["passed"] for case in cases),
                }
            )

        full_board = [
            _scan_full_board(
                definition=definition,
                modes=modes,
                cards_by_id=cards_by_id,
                mode=mode,
                seed=definition.card_id * 1000 + mode_index,
            )
            for mode_index, mode in enumerate(modes)
        ]
        failure_count += sum(not result["passed"] for result in full_board)
        card_reports.append(
            {
                "card_id": definition.card_id,
                "name": definition.name,
                "collectible": definition.is_collectible,
                "in_training_deck_closure": definition.card_id in closure_ids,
                "card_type": definition.card_type,
                "printed_cost": definition.cost,
                "modes": [
                    {
                        "mode_id": mode.mode_id,
                        "mode_type": mode.mode_type,
                        "mode_cost": mode.cost,
                        "resulting_card_type": mode.resulting_card_type,
                        "countdown": mode.countdown,
                    }
                    for mode in modes
                ],
                "modifier_scenarios": scenarios,
                "full_board_cases": full_board,
                "unavailable_boundaries": sorted(unavailable_boundaries),
                "passed": (
                    all(scenario["passed"] for scenario in scenarios)
                    and all(result["passed"] for result in full_board)
                ),
            }
        )

    invocation_cards = [
        card.card_id
        for card in all_cards
        if rulebook.invocation_for(card.card_id) is not None
    ]
    mode_type_counts = Counter(
        mode.mode_type
        for card in mode_cards
        for mode in rulebook.modes_for(card.card_id)
    )
    training_mode_cards = sum(
        card.card_id in closure_ids for card in mode_cards
    )
    full_pool_ids = {card.card_id for card in all_cards}
    scope_complete = (
        {card["card_id"] for card in card_reports}
        == {card.card_id for card in mode_cards}
        and all(card["passed"] for card in card_reports)
        and all(card["card_id"] in full_pool_ids for card in card_reports)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "swb_play_mode_boundary_audit",
        "inputs": {
            "database": {
                "path": _relative(database, root),
                "sha256": _sha256_file(database),
            },
            "rules": {
                "path": _relative(rules_directory, root),
                "sha256": hash_rule_directory(rules_directory),
            },
            "training_closure": {
                "path": _relative(closure_path, root),
                "sha256": _sha256_file(closure_path),
            },
            "resolution_source_sha256": _sha256_file(
                root / "swb/engine/resolution.py"
            ),
            "environment_source_sha256": _sha256_file(
                root / "swb/engine/environment.py"
            ),
        },
        "policy": {
            "maximum_remaining_pp": MAX_REMAINING_PP,
            "enhance": (
                "mandatory highest affordable Enhance; normal is suppressed "
                "once an applicable Enhance threshold is reached"
            ),
            "accelerate_and_crystallize": (
                "available only while the card's current body cost cannot be "
                "paid; printed cost and current cost remain distinct"
            ),
            "illegal_command_atomicity": (
                "deterministic_fingerprint includes RNG and all public/private "
                "zones and must remain identical"
            ),
            "official_evidence": [
                {
                    "publisher": "Cygames",
                    "url": "https://shadowverse-wb.com/ja/help?tab=tab0",
                    "accessed": "2026-07-29",
                    "section": "エンハンス",
                    "conclusion": (
                        "Enhance is mandatory at its threshold and the highest "
                        "affordable Enhance value is selected."
                    ),
                },
                {
                    "publisher": "Cygames",
                    "url": (
                        "https://shadowverse.com/cards/cardpack/"
                        "verdantconflict/?lang=en"
                    ),
                    "accessed": "2026-07-29",
                    "section": "Crystallize",
                    "conclusion": (
                        "Crystallize is an alternate route used when the body "
                        "cost cannot be paid."
                    ),
                },
            ],
        },
        "summary": {
            "database_card_count": len(all_cards),
            "training_closure_card_count": len(closure_ids),
            "play_mode_card_count": len(mode_cards),
            "training_closure_play_mode_card_count": training_mode_cards,
            "full_pool_play_mode_card_count": len(mode_cards),
            "play_mode_count": sum(mode_type_counts.values()),
            "mode_type_counts": dict(sorted(mode_type_counts.items())),
            "modifier_scenario_count": len(MODIFIER_SCENARIOS),
            "cost_boundary_case_count": case_count,
            "full_board_case_count": sum(
                len(card["full_board_cases"]) for card in card_reports
            ),
            "command_action_mask_mismatch_count": sum(
                not case["command_mask_match"]
                for card in card_reports
                for scenario in card["modifier_scenarios"]
                for case in scenario["cases"]
            ),
            "illegal_atomicity_failure_count": sum(
                not case["illegal_atomicity_passed"]
                for card in card_reports
                for scenario in card["modifier_scenarios"]
                for case in scenario["cases"]
            ),
            "execution_failure_count": sum(
                case["execution"] is not None
                and not case["execution"]["passed"]
                for card in card_reports
                for scenario in card["modifier_scenarios"]
                for case in scenario["cases"]
            ),
            "failure_count": failure_count,
            "scope_complete": scope_complete,
            "passed": failure_count == 0 and scope_complete,
        },
        "non_pp_alternate_routes": {
            "invocation": {
                "card_ids": invocation_cards,
                "status": "covered_by_dedicated_trigger_and_capacity tests",
                "evidence": ["tests/test_invocation.py"],
            },
            "manual_mode_choices": {
                "status": (
                    "not a play-point route; covered by choose-one command, "
                    "target, stale-choice, and action-mask tests"
                ),
                "evidence": [
                    "tests/test_choices.py",
                    "tests/test_real_multi_mode_batch.py",
                    "tests/test_real_spell_modes_and_earth_listener_batch.py",
                ],
            },
        },
        "cross_cutting_evidence": {
            "no_legal_target": [
                "tests/test_play_modes_audit.py",
                "tests/test_real_selected_hand_grants_twelfth_batch.py",
            ],
            "hand_capacity": [
                "tests/test_real_generated_hand_last_words_batch.py",
                "tests/test_real_spell_enhance_batch.py",
            ],
            "real_card_zone_and_resource_semantics": [
                "tests/test_play_modes.py",
                "tests/test_real_balanced_trigger_resource_batch.py",
                "tests/test_real_ward_marine_crest_listener_batch.py",
            ],
            "real_high_pp_regressions": [
                "tests/test_play_modes_audit.py",
                "data/reports/card_bug_audit/reproductions/SWB-CARD-0001.json",
                "data/reports/card_bug_audit/reproductions/SWB-CARD-0002.json",
            ],
        },
        "cards": card_reports,
    }


def render_markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    rows = [
        "# Play-mode boundary audit",
        "",
        (
            f"- Result: **{'PASS' if summary['passed'] else 'FAIL'}**; "
            f"{summary['failure_count']} failures."
        ),
        (
            f"- Full pool: {summary['full_pool_play_mode_card_count']} cards / "
            f"{summary['play_mode_count']} modes; training closure: "
            f"{summary['training_closure_play_mode_card_count']} cards."
        ),
        (
            f"- Cost cases: {summary['cost_boundary_case_count']}; full-board "
            f"cases: {summary['full_board_case_count']}."
        ),
        (
            "- Command/action-mask mismatches: "
            f"{summary['command_action_mask_mismatch_count']}; illegal "
            "atomicity failures: "
            f"{summary['illegal_atomicity_failure_count']}; execution "
            f"failures: {summary['execution_failure_count']}."
        ),
        "",
        "| Card | Type | Printed cost | Modes | Training closure | Result |",
        "|---:|---|---:|---|:---:|:---:|",
    ]
    for card in report["cards"]:
        modes = ", ".join(
            f"{mode['mode_id']}={mode['mode_cost']}"
            for mode in card["modes"]
        )
        rows.append(
            f"| {card['card_id']} {card['name']} | {card['card_type']} | "
            f"{card['printed_cost']} | {modes} | "
            f"{'yes' if card['in_training_deck_closure'] else 'no'} | "
            f"{'PASS' if card['passed'] else 'FAIL'} |"
        )
    return "\n".join(rows) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    report = build_report(
        root=root,
        database=(root / args.database).resolve(),
        rules_directory=(root / args.rules).resolve(),
        closure_path=(root / args.closure).resolve(),
    )
    output = (root / args.output).resolve()
    markdown = (root / args.markdown).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_json(report), encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(render_json(report["summary"]), end="")
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
