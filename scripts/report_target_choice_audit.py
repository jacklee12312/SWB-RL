"""Audit target provenance, pending choices, and RL choice ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardPassive, CardRule, RuleBook, Trigger
from swb.engine.commands import Attack, ChoiceKind, Choose, PlayCard
from swb.engine.effects import (
    BoardFilter,
    EffectKind,
    EffectOperation,
    TargetKind,
    ValueExpression,
    ExprType,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import (
    Amulet,
    HandCard,
    Phase,
    TargetingRestriction,
    Unit,
)
from swb.engine.targeting import (
    build_choice_options,
    build_graveyard_choice_options,
    graveyard_candidates,
    hand_candidates,
    hand_choice_options,
    is_all_target,
    is_choice_target,
    is_random_target,
    leader_choice_options,
    target_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_RULES = Path("data/rules")
DEFAULT_CLOSURE = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)
DEFAULT_COVERAGE = Path("data/reports/rule_coverage.json")
DEFAULT_OUTPUT = Path(
    "data/reports/card_bug_audit/target_choice_audit.json"
)
DEFAULT_MARKDOWN = Path(
    "data/reports/card_bug_audit/target_choice_audit.md"
)

MANUAL_TARGETS = tuple(
    target for target in TargetKind if is_choice_target(target)
)
DECISION_KINDS = frozenset(
    {
        EffectKind.CHOOSE_ONE,
        EffectKind.OPTIONAL,
        EffectKind.TARGET_EXISTS,
        EffectKind.SELECT_TARGETS,
    }
)
TARGET_CATEGORIES = (
    "manual",
    "random",
    "all",
    "implicit_or_bound",
    "decision",
)
DEMO_TEST_EVIDENCE = {
    "conditional_demo.json": "tests/test_conditions.py",
    "decisions_demo.json": "tests/test_decisions.py",
    "graveyard_demo.json": "tests/test_graveyard.py",
}


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _card(
    card_id: int,
    *,
    card_type: str = "随从",
    cost: int = 1,
    attack: int = 2,
    life: int = 4,
    keywords: Iterable[str] = (),
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"target-audit-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack if card_type == "随从" else None,
        life=life if card_type == "随从" else None,
        keywords=frozenset(keywords),
        support_level="basic",
        is_collectible=True,
    )


def _engine(
    *,
    rulebook: RuleBook | None = None,
    resolver=None,
    seed: int = 16001,
) -> GameEngine:
    engine = GameEngine(
        [_card(860000 + index) for index in range(40)],
        [_card(870000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook or RuleBook(),
        card_resolver=resolver,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    engine.state.phase = Phase.MAIN
    engine.state.active_player = 0
    for player in engine.players:
        player.hand.clear()
        player.hand_entity_ids.clear()
        player.board.clear()
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _put_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int = 0,
) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
        printed_keyword_overrides=set(
            engine.rulebook.non_intrinsic_keywords(definition.card_id)
        ),
    )
    player = engine.players[player_index]
    player.hand.append(card)
    player.hand_entity_ids.append(card.entity_id)
    return card


def _put_unit(
    engine: GameEngine,
    player_index: int,
    definition: CardDefinition,
) -> Unit:
    unit = Unit.summon(
        definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    unit.summoned_this_turn = False
    unit.can_attack = True
    unit.attacks_remaining = unit.attacks_per_turn
    engine.players[player_index].board.append(unit)
    return unit


def _put_amulet(
    engine: GameEngine,
    player_index: int,
    definition: CardDefinition,
) -> Amulet:
    amulet = Amulet(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[player_index].board.append(amulet)
    return amulet


def _play(engine: GameEngine, definition: CardDefinition) -> None:
    _put_hand(engine, definition)
    engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))


def _choose_entity(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    if request is None:
        raise AssertionError("expected a pending target choice")
    option = next(
        option for option in request.options if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _manual_options(
    engine: GameEngine,
    operation: EffectOperation,
    *,
    source_entity_id: int | None = None,
) -> list:
    if operation.target is TargetKind.OWN_HAND:
        # Hand options intentionally preserve hand order.
        return hand_choice_options(
            hand_candidates(
                operation,
                0,
                engine.players,
                source_entity_id=source_entity_id,
            )
        )
    if operation.target is TargetKind.OWN_GRAVEYARD_CARD:
        return build_graveyard_choice_options(
            graveyard_candidates(operation, 0, engine.players)
        )
    options = build_choice_options(
        target_candidates(
            operation,
            0,
            engine.players,
            source_entity_id=source_entity_id,
        )
    )
    options.extend(
        leader_choice_options(operation.target, 0, engine.players)
    )
    return options


def _add_graveyard_card(
    engine: GameEngine,
    definition: CardDefinition,
) -> int:
    entity_id = engine.state.allocate_entity_id()
    engine._send_to_graveyard(
        0,
        definition,
        "target_audit_fixture",
        source_entity_id=entity_id,
    )
    return entity_id


def _expected_option_ids(
    target: TargetKind,
    ids: Mapping[str, list[int]],
) -> list[str]:
    lookup = {
        TargetKind.OWN_UNIT: ids["own_units"],
        TargetKind.ENEMY_UNIT: ids["enemy_units"],
        TargetKind.ANY_UNIT: ids["own_units"] + ids["enemy_units"],
        TargetKind.OWN_UNIT_OR_LEADER: ids["own_units"] + [-1],
        TargetKind.ENEMY_UNIT_OR_LEADER: ids["enemy_units"] + [-2],
        TargetKind.ANY_UNIT_OR_LEADER: (
            ids["own_units"] + ids["enemy_units"] + [-1, -2]
        ),
        TargetKind.OWN_AMULET: ids["own_amulets"],
        TargetKind.ENEMY_AMULET: ids["enemy_amulets"],
        TargetKind.ANY_AMULET: (
            ids["own_amulets"] + ids["enemy_amulets"]
        ),
        TargetKind.OWN_BOARD: ids["own_board"],
        TargetKind.ENEMY_BOARD: ids["enemy_board"],
        TargetKind.ANY_BOARD: ids["own_board"] + ids["enemy_board"],
        TargetKind.OWN_HAND: ids["hand"],
        TargetKind.OWN_GRAVEYARD_CARD: ids["graveyard"],
    }
    return [
        (
            f"leader:{-1 - value}"
            if value < 0
            else (
                f"hand:{value}"
                if target is TargetKind.OWN_HAND
                else f"entity:{value}"
            )
        )
        for value in lookup[target]
    ]


def _candidate_domain_contracts() -> list[dict[str, object]]:
    empty_engine = _engine()
    populated = _engine()
    own_units = [
        _put_unit(populated, 0, _card(861100 + index))
        for index in range(2)
    ]
    own_amulets = [
        _put_amulet(
            populated,
            0,
            _card(861200 + index, card_type="护符"),
        )
        for index in range(2)
    ]
    enemy_units = [
        _put_unit(populated, 1, _card(861300 + index))
        for index in range(2)
    ]
    enemy_amulets = [
        _put_amulet(
            populated,
            1,
            _card(861400 + index, card_type="护符"),
        )
        for index in range(2)
    ]
    hand = [
        _put_hand(populated, _card(861500 + index)).entity_id
        for index in range(2)
    ]
    graveyard = [
        _add_graveyard_card(populated, _card(861600 + index))
        for index in range(2)
    ]
    ids = {
        "own_units": [entity.entity_id for entity in own_units],
        "enemy_units": [entity.entity_id for entity in enemy_units],
        "own_amulets": [entity.entity_id for entity in own_amulets],
        "enemy_amulets": [entity.entity_id for entity in enemy_amulets],
        "own_board": [
            entity.entity_id for entity in populated.players[0].board
        ],
        "enemy_board": [
            entity.entity_id for entity in populated.players[1].board
        ],
        "hand": hand,
        "graveyard": graveyard,
    }
    expected_empty = {
        TargetKind.OWN_UNIT: 0,
        TargetKind.ENEMY_UNIT: 0,
        TargetKind.ANY_UNIT: 0,
        TargetKind.OWN_UNIT_OR_LEADER: 1,
        TargetKind.ENEMY_UNIT_OR_LEADER: 1,
        TargetKind.ANY_UNIT_OR_LEADER: 2,
        TargetKind.OWN_AMULET: 0,
        TargetKind.ENEMY_AMULET: 0,
        TargetKind.ANY_AMULET: 0,
        TargetKind.OWN_BOARD: 0,
        TargetKind.ENEMY_BOARD: 0,
        TargetKind.ANY_BOARD: 0,
        TargetKind.OWN_HAND: 0,
        TargetKind.OWN_GRAVEYARD_CARD: 0,
    }
    rows: list[dict[str, object]] = []
    for target in MANUAL_TARGETS:
        operation = EffectOperation(EffectKind.BANISH, target)
        empty_options = _manual_options(empty_engine, operation)
        populated_options = _manual_options(populated, operation)
        actual_ids = [option.option_id for option in populated_options]
        expected_ids = _expected_option_ids(target, ids)
        rows.append(
            {
                "target": target.value,
                "empty_candidate_count": len(empty_options),
                "expected_empty_candidate_count": expected_empty[target],
                "populated_candidate_count": len(populated_options),
                "expected_option_ids": expected_ids,
                "actual_option_ids": actual_ids,
                "passed": (
                    len(empty_options) == expected_empty[target]
                    and actual_ids == expected_ids
                ),
            }
        )
    return rows


def _cardinality_contracts() -> list[dict[str, object]]:
    source = _card(862000, card_type="法术")
    operation = EffectOperation(
        EffectKind.DAMAGE_UNIT,
        TargetKind.ENEMY_UNIT,
        amount=1,
        requires_target=True,
    )
    rulebook = RuleBook((CardRule(source.card_id, Trigger.PLAY, (operation,)),))
    rows: list[dict[str, object]] = []
    for count in range(3):
        engine = _engine(rulebook=rulebook)
        targets = [
            _put_unit(engine, 1, _card(862100 + index))
            for index in range(count)
        ]
        _put_hand(engine, source)
        fingerprint = engine.deterministic_fingerprint()
        playable = any(
            isinstance(command, PlayCard)
            for command in engine.legal_commands()
        )
        if playable:
            engine.apply(PlayCard(0, 0))
        option_count = (
            len(engine.state.pending_choice.options)
            if engine.state.pending_choice is not None
            else 0
        )
        rows.append(
            {
                "legal_candidate_count": count,
                "playable": playable,
                "pending_option_count": option_count,
                "zero_candidate_atomic": (
                    engine.deterministic_fingerprint() == fingerprint
                    if count == 0
                    else True
                ),
                "passed": (
                    (not playable and option_count == 0)
                    if count == 0
                    else (
                        playable
                        and option_count == len(targets)
                    )
                ),
            }
        )
    return rows


def _source_exclusion_contract() -> dict[str, object]:
    source = _card(862200)
    operation = EffectOperation(
        EffectKind.BANISH,
        TargetKind.OWN_BOARD,
        exclude_source=True,
    )
    rulebook = RuleBook(
        (CardRule(source.card_id, Trigger.FANFARE, (operation,)),)
    )
    engine = _engine(rulebook=rulebook)
    other_unit = _put_unit(engine, 0, _card(862201))
    other_amulet = _put_amulet(
        engine,
        0,
        _card(862202, card_type="护符"),
    )
    _play(engine, source)
    played_source = next(
        entity
        for entity in engine.players[0].board
        if entity.definition.card_id == source.card_id
    )
    option_ids = [
        option.entity_id for option in engine.state.pending_choice.options
    ]
    return {
        "source_entity_id": played_source.entity_id,
        "actual_option_ids": option_ids,
        "expected_option_ids": [other_unit.entity_id, other_amulet.entity_id],
        "passed": (
            option_ids == [other_unit.entity_id, other_amulet.entity_id]
            and played_source.entity_id not in option_ids
        ),
    }


def _restriction_contracts() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    engine = _engine()
    protected = _put_unit(engine, 1, _card(862300))
    ordinary = _put_unit(engine, 1, _card(862301))
    protected.add_targeting_restriction(
        TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS,
        duration="permanent",
    )
    manual = EffectOperation(EffectKind.DAMAGE_UNIT, TargetKind.ENEMY_UNIT)
    random_op = EffectOperation(
        EffectKind.DAMAGE_UNIT,
        TargetKind.RANDOM_ENEMY_UNIT,
    )
    all_op = EffectOperation(
        EffectKind.DAMAGE_UNIT,
        TargetKind.ALL_ENEMY_UNITS,
    )
    manual_ids = [
        option.entity_id for option in engine._target_options(manual, 0)
    ]
    random_ids = [
        entity.entity_id
        for entity in target_candidates(random_op, 0, engine.players)
    ]
    all_ids = [
        entity.entity_id
        for entity in target_candidates(all_op, 0, engine.players)
    ]
    rows.append(
        {
            "case": "cannot_be_targeted_affects_manual_enemy_effect_only",
            "passed": (
                manual_ids == [ordinary.entity_id]
                and random_ids == [protected.entity_id, ordinary.entity_id]
                and all_ids == [protected.entity_id, ordinary.entity_id]
            ),
        }
    )

    ambush_engine = _engine()
    ambush = _put_unit(
        ambush_engine,
        1,
        _card(862310, keywords=("潜行",)),
    )
    visible = _put_unit(ambush_engine, 1, _card(862311))
    ambush_manual = [
        option.entity_id
        for option in ambush_engine._target_options(manual, 0)
    ]
    ambush_random = [
        entity.entity_id
        for entity in target_candidates(
            random_op,
            0,
            ambush_engine.players,
        )
    ]
    rows.append(
        {
            "case": "ambush_affects_manual_enemy_effect_only",
            "passed": (
                ambush_manual == [visible.entity_id]
                and ambush_random == [ambush.entity_id, visible.entity_id]
            ),
        }
    )

    effect_engine = _engine()
    ward = _put_unit(
        effect_engine,
        1,
        _card(862320, keywords=("守护",)),
    )
    non_ward = _put_unit(effect_engine, 1, _card(862321))
    effect_ids = [
        option.entity_id
        for option in effect_engine._target_options(manual, 0)
    ]

    attacker_definition = _card(862322)
    passives = (
        CardPassive(
            attacker_definition.card_id,
            "ignores_ward",
            0,
        ),
    )
    attack_engine = _engine(rulebook=RuleBook(passives=passives))
    attacker = _put_unit(attack_engine, 0, attacker_definition)
    attack_ward = _put_unit(
        attack_engine,
        1,
        _card(862323, keywords=("守护",)),
    )
    attack_other = _put_unit(attack_engine, 1, _card(862324))
    attack_targets = {
        command.target_id
        for command in attack_engine.legal_commands()
        if isinstance(command, Attack)
        and command.attacker_id == attacker.entity_id
    }
    rows.append(
        {
            "case": "ward_and_ignore_ward_affect_combat_not_effect_choice",
            "passed": (
                effect_ids == [ward.entity_id, non_ward.entity_id]
                and attack_targets
                == {None, attack_ward.entity_id, attack_other.entity_id}
            ),
        }
    )
    return rows


def _multi_target_contracts() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    source = _card(862400, card_type="法术")
    distinct = EffectOperation(
        EffectKind.DAMAGE_UNIT,
        TargetKind.ENEMY_UNIT,
        amount=1,
        target_count=3,
    )
    engine = _engine(
        rulebook=RuleBook(
            (CardRule(source.card_id, Trigger.PLAY, (distinct,)),)
        )
    )
    targets = [
        _put_unit(engine, 1, _card(862410 + index))
        for index in range(3)
    ]
    _play(engine, source)
    selection_order = [targets[2], targets[0], targets[1]]
    observed_prefixes: list[list[int]] = []
    duplicate_atomic = True
    for index, target in enumerate(selection_order):
        request = engine.state.pending_choice
        if request is None:
            break
        if index == 1:
            fingerprint = engine.deterministic_fingerprint()
            try:
                engine.apply(
                    Choose(0, f"entity:{selection_order[0].entity_id}")
                )
            except IllegalCommand:
                duplicate_atomic = (
                    engine.deterministic_fingerprint() == fingerprint
                )
            else:
                duplicate_atomic = False
        _choose_entity(engine, target.entity_id)
        if engine.state.pending_choice is not None:
            observed_prefixes.append(
                [
                    option.entity_id
                    for option
                    in engine.state.pending_choice.selected_options
                ]
            )
    rows.append(
        {
            "case": "distinct_targets_preserve_selection_order",
            "observed_selected_prefixes": observed_prefixes,
            "duplicate_rejection_atomic": duplicate_atomic,
            "passed": (
                observed_prefixes
                == [
                    [selection_order[0].entity_id],
                    [
                        selection_order[0].entity_id,
                        selection_order[1].entity_id,
                    ],
                ]
                and duplicate_atomic
                and [target.health for target in targets] == [3, 3, 3]
            ),
        }
    )

    duplicate_source = _card(862401, card_type="法术")
    duplicate = EffectOperation(
        EffectKind.DAMAGE_UNIT,
        TargetKind.ENEMY_UNIT,
        amount=1,
        target_count=2,
        allow_duplicate_targets=True,
    )
    duplicate_engine = _engine(
        rulebook=RuleBook(
            (
                CardRule(
                    duplicate_source.card_id,
                    Trigger.PLAY,
                    (duplicate,),
                ),
            )
        )
    )
    duplicate_target = _put_unit(
        duplicate_engine,
        1,
        _card(862420),
    )
    _play(duplicate_engine, duplicate_source)
    _choose_entity(duplicate_engine, duplicate_target.entity_id)
    _choose_entity(duplicate_engine, duplicate_target.entity_id)
    rows.append(
        {
            "case": "duplicate_targets_apply_in_order_when_allowed",
            "passed": duplicate_target.health == 2,
        }
    )

    shortage_source = _card(862402, card_type="法术")
    shortage = EffectOperation(
        EffectKind.DAMAGE_UNIT,
        TargetKind.ENEMY_UNIT,
        amount=1,
        target_count=3,
    )
    shortage_engine = _engine(
        rulebook=RuleBook(
            (
                CardRule(
                    shortage_source.card_id,
                    Trigger.PLAY,
                    (shortage,),
                ),
            )
        )
    )
    shortage_targets = [
        _put_unit(shortage_engine, 1, _card(862430 + index))
        for index in range(2)
    ]
    _play(shortage_engine, shortage_source)
    effective_count = shortage_engine.state.pending_choice.target_count
    for target in shortage_targets:
        _choose_entity(shortage_engine, target.entity_id)
    rows.append(
        {
            "case": "candidate_shortage_truncates_without_duplicates",
            "effective_target_count": effective_count,
            "passed": (
                effective_count == 2
                and [target.health for target in shortage_targets] == [3, 3]
            ),
        }
    )
    return rows


def _stale_target_case(
    case: str,
    mutate,
) -> dict[str, object]:
    source = _card(862500, card_type="法术")
    operation = EffectOperation(
        EffectKind.DAMAGE_UNIT,
        TargetKind.ENEMY_UNIT,
        amount=2,
        board_filter=BoardFilter(cost_max=2),
    )
    draw = EffectOperation(
        EffectKind.DRAW,
        TargetKind.OWN_LEADER,
        amount=1,
    )
    engine = _engine(
        rulebook=RuleBook(
            (CardRule(source.card_id, Trigger.PLAY, (operation, draw)),)
        )
    )
    target = _put_unit(engine, 1, _card(862501, cost=1))
    _play(engine, source)
    request = engine.state.pending_choice
    option_id = request.options[0].option_id
    deck_before = len(engine.players[0].deck)
    mutate(engine, target)
    engine.apply(Choose(0, option_id))
    return {
        "case": case,
        "target_health": target.health,
        "draw_continued": len(engine.players[0].deck) == deck_before - 1,
        "pending_cleared": engine.state.pending_choice is None,
        "passed": (
            target.health == 4
            and len(engine.players[0].deck) == deck_before - 1
            and engine.state.pending_choice is None
        ),
    }


def _stale_target_contracts() -> list[dict[str, object]]:
    def leave(engine: GameEngine, target: Unit) -> None:
        engine.players[1].board.remove(target)
        engine._send_to_graveyard(
            1,
            target.definition,
            "target_audit_left_play",
            source_entity_id=target.entity_id,
        )

    def die(engine: GameEngine, target: Unit) -> None:
        engine.players[1].board.remove(target)
        engine._send_to_graveyard(
            1,
            target.definition,
            "zero_health",
            source_entity_id=target.entity_id,
        )

    def transform(engine: GameEngine, target: Unit) -> None:
        replacement = Amulet(
            definition=_card(862502, card_type="护符"),
            entity_id=target.entity_id,
        )
        board = engine.players[1].board
        board[board.index(target)] = replacement

    def change_controller(engine: GameEngine, target: Unit) -> None:
        engine.players[1].board.remove(target)
        engine.players[0].board.append(target)

    def fail_filter(engine: GameEngine, target: Unit) -> None:
        target.definition = _card(862503, cost=5)

    return [
        _stale_target_case("target_died", die),
        _stale_target_case("target_left_play", leave),
        _stale_target_case("target_transformed", transform),
        _stale_target_case("target_changed_controller", change_controller),
        _stale_target_case("target_failed_filter", fail_filter),
    ]


def _source_leave_contracts() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    source = _card(862600, attack=3)
    operations = (
        EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.ENEMY_UNIT,
            amount=1,
            target_key="selected",
        ),
        EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.PREVIOUS_TARGET,
            amount=1,
            target_key="selected",
        ),
    )
    engine = _engine(
        rulebook=RuleBook(
            (CardRule(source.card_id, Trigger.FANFARE, operations),)
        )
    )
    target = _put_unit(engine, 1, _card(862601, life=5))
    _play(engine, source)
    played_source = next(
        entity
        for entity in engine.players[0].board
        if entity.definition.card_id == source.card_id
    )
    engine.players[0].board.remove(played_source)
    engine._send_to_graveyard(
        0,
        played_source.definition,
        "target_audit_source_left",
        source_entity_id=played_source.entity_id,
    )
    _choose_entity(engine, target.entity_id)
    rows.append(
        {
            "case": "bound_target_survives_source_leaving",
            "passed": target.health == 3,
        }
    )

    dependent_source = _card(862610, attack=3)
    dependent_ops = (
        EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.ENEMY_UNIT,
            amount_expr=ValueExpression(ExprType.SOURCE_ATTACK),
        ),
        EffectOperation(
            EffectKind.DRAW,
            TargetKind.OWN_LEADER,
            amount=1,
        ),
    )
    dependent = _engine(
        rulebook=RuleBook(
            (
                CardRule(
                    dependent_source.card_id,
                    Trigger.FANFARE,
                    dependent_ops,
                ),
            )
        )
    )
    dependent_target = _put_unit(
        dependent,
        1,
        _card(862611, life=5),
    )
    _play(dependent, dependent_source)
    source_entity = next(
        entity
        for entity in dependent.players[0].board
        if entity.definition.card_id == dependent_source.card_id
    )
    dependent.players[0].board.remove(source_entity)
    dependent._send_to_graveyard(
        0,
        source_entity.definition,
        "target_audit_source_left",
        source_entity_id=source_entity.entity_id,
    )
    deck_before = len(dependent.players[0].deck)
    _choose_entity(dependent, dependent_target.entity_id)
    rows.append(
        {
            "case": "source_dependent_operation_skips_but_queue_continues",
            "passed": (
                dependent_target.health == 5
                and len(dependent.players[0].deck) == deck_before - 1
            ),
        }
    )
    return rows


def _mixed_target_order_contract() -> dict[str, object]:
    source = _card(862700, card_type="法术")
    operations = (
        EffectOperation(
            EffectKind.DESTROY,
            TargetKind.ENEMY_UNIT,
        ),
        EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.RANDOM_ENEMY_UNIT,
            amount=1,
        ),
        EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.ALL_ENEMY_UNITS,
            amount=2,
        ),
    )
    engine = _engine(
        rulebook=RuleBook(
            (CardRule(source.card_id, Trigger.PLAY, operations),)
        ),
        seed=16002,
    )
    selected = _put_unit(engine, 1, _card(862701, life=4))
    survivor = _put_unit(engine, 1, _card(862702, life=6))
    _play(engine, source)
    _choose_entity(engine, selected.entity_id)
    return {
        "selected_left_play": selected not in engine.players[1].board,
        "survivor_health": survivor.health,
        "expected_survivor_health": 3,
        "passed": (
            selected not in engine.players[1].board
            and survivor.health == 3
        ),
    }


def _no_candidate_policy_contracts() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    prohibit_source = _card(862800, card_type="法术")
    prohibit_op = EffectOperation(
        EffectKind.DAMAGE_UNIT,
        TargetKind.ENEMY_UNIT,
        amount=1,
        requires_target=True,
    )
    prohibit = _engine(
        rulebook=RuleBook(
            (
                CardRule(
                    prohibit_source.card_id,
                    Trigger.PLAY,
                    (prohibit_op,),
                ),
            )
        )
    )
    _put_hand(prohibit, prohibit_source)
    before = prohibit.deterministic_fingerprint()
    playable = any(
        isinstance(command, PlayCard)
        for command in prohibit.legal_commands()
    )
    rows.append(
        {
            "case": "required_selected_target_prohibits_play",
            "passed": (
                not playable
                and prohibit.deterministic_fingerprint() == before
            ),
        }
    )

    skip_source = _card(862801, card_type="法术")
    skip_ops = (
        EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.ENEMY_UNIT,
            amount=1,
        ),
        EffectOperation(
            EffectKind.DRAW,
            TargetKind.OWN_LEADER,
            amount=1,
        ),
    )
    skip = _engine(
        rulebook=RuleBook(
            (CardRule(skip_source.card_id, Trigger.PLAY, skip_ops),)
        )
    )
    deck_before = len(skip.players[0].deck)
    _play(skip, skip_source)
    rows.append(
        {
            "case": "unavailable_selected_operation_skips",
            "passed": (
                skip.state.pending_choice is None
                and len(skip.players[0].deck) == deck_before - 1
            ),
        }
    )

    random_all_source = _card(862802, card_type="法术")
    random_all_ops = (
        EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.RANDOM_ENEMY_UNIT,
            amount=1,
        ),
        EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.ALL_ENEMY_UNITS,
            amount=1,
        ),
        EffectOperation(
            EffectKind.DRAW,
            TargetKind.OWN_LEADER,
            amount=1,
        ),
    )
    random_all = _engine(
        rulebook=RuleBook(
            (
                CardRule(
                    random_all_source.card_id,
                    Trigger.PLAY,
                    random_all_ops,
                ),
            )
        )
    )
    deck_before = len(random_all.players[0].deck)
    _play(random_all, random_all_source)
    rows.append(
        {
            "case": "random_and_all_no_candidates_are_safe_noops",
            "passed": (
                random_all.state.pending_choice is None
                and len(random_all.players[0].deck) == deck_before - 1
            ),
        }
    )

    branch_source = _card(862803, card_type="法术")
    target_exists = EffectOperation(
        EffectKind.TARGET_EXISTS,
        TargetKind.ENEMY_UNIT,
        then_operations=(
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                amount=5,
            ),
        ),
        else_operations=(
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                amount=2,
            ),
        ),
    )
    branch = _engine(
        rulebook=RuleBook(
            (
                CardRule(
                    branch_source.card_id,
                    Trigger.PLAY,
                    (target_exists,),
                ),
            )
        )
    )
    _play(branch, branch_source)
    rows.append(
        {
            "case": "target_exists_executes_else_branch",
            "passed": branch.players[1].health == 18,
        }
    )
    return rows


def _snapshot_restore_contract() -> dict[str, object]:
    source = _card(862900, card_type="法术")
    operations = (
        EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.ENEMY_UNIT,
            amount=1,
        ),
        EffectOperation(
            EffectKind.DAMAGE_UNIT,
            TargetKind.RANDOM_ENEMY_UNIT,
            amount=1,
        ),
    )
    engine = _engine(
        rulebook=RuleBook(
            (CardRule(source.card_id, Trigger.PLAY, operations),)
        ),
        seed=16003,
    )
    targets = [
        _put_unit(engine, 1, _card(862910 + index, life=5))
        for index in range(3)
    ]
    _play(engine, source)
    pending_saved = engine.state.pending_choice is not None
    snapshot = engine.snapshot()
    request = engine.state.pending_choice
    choice = Choose(request.player_index, request.options[1].option_id)
    first = engine.apply(choice)
    first_events = first.events
    first_fingerprint = engine.deterministic_fingerprint()
    engine.restore(snapshot)
    pending_restored = engine.state.pending_choice is not None
    second = engine.apply(choice)
    return {
        "pending_choice_saved": pending_saved,
        "pending_choice_restored": pending_restored,
        "events_equal": first_events == second.events,
        "fingerprints_equal": (
            first_fingerprint == engine.deterministic_fingerprint()
        ),
        "target_healths": [target.health for target in engine.players[1].board],
        "passed": (
            pending_saved
            and pending_restored
            and first_events == second.events
            and first_fingerprint == engine.deterministic_fingerprint()
        ),
    }


def _action_order_contract() -> dict[str, object]:
    source = _card(863000, card_type="法术")
    operation = EffectOperation(
        EffectKind.BANISH,
        TargetKind.ANY_BOARD,
    )
    rulebook = RuleBook(
        (CardRule(source.card_id, Trigger.PLAY, (operation,)),)
    )
    deck_a = [_card(863100 + index) for index in range(40)]
    deck_b = [_card(863200 + index) for index in range(40)]
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=1,
        class_b=1,
        seed=16004,
        rulebook=rulebook,
    )
    env.reset(seed=16004)
    env.core.state.phase = Phase.MAIN
    env.core.state.active_player = 0
    for player in env.core.players:
        player.hand.clear()
        player.hand_entity_ids.clear()
        player.board.clear()
    env.core.players[0].max_mana = 10
    env.core.players[0].mana = 10
    _put_hand(env.core, source)
    own_unit = _put_unit(env.core, 0, _card(863001))
    own_amulet = _put_amulet(
        env.core,
        0,
        _card(863002, card_type="护符"),
    )
    enemy_unit = _put_unit(env.core, 1, _card(863003))
    enemy_amulet = _put_amulet(
        env.core,
        1,
        _card(863004, card_type="护符"),
    )
    env.core.apply(PlayCard(0, 0))
    env._invalidate_caches(
        advance_transition=False,
        reason="target_audit_fixture",
    )
    request = env.core.state.pending_choice
    expected_ids = [
        own_unit.entity_id,
        own_amulet.entity_id,
        enemy_unit.entity_id,
        enemy_amulet.entity_id,
    ]
    option_ids = [option.entity_id for option in request.options]
    mask = env.action_mask()
    enabled = [
        action
        for action in range(
            env.CHOICE_OFFSET,
            env.GRAVEYARD_CHOICE_OFFSET,
        )
        if mask[action]
    ]
    decoded_ids = [
        env._decode_action(action).option_id
        for action in enabled
    ]
    option_strings = [option.option_id for option in request.options]
    legal_strings = [
        command.option_id
        for command in env.core.legal_commands()
        if isinstance(command, Choose)
    ]
    return {
        "action_size": env.ACTION_SIZE,
        "ui_option_entity_ids": option_ids,
        "enabled_choice_actions": enabled,
        "decoded_option_ids": decoded_ids,
        "legal_command_option_ids": legal_strings,
        "passed": (
            env.ACTION_SIZE == 112
            and option_ids == expected_ids
            and enabled
            == [
                env.CHOICE_OFFSET + index
                for index in range(len(expected_ids))
            ]
            and decoded_ids == option_strings == legal_strings
        ),
    }


def _iter_operations(
    value: object,
    path: str,
) -> Iterable[tuple[str, EffectOperation]]:
    if isinstance(value, EffectOperation):
        yield path, value
        for field in fields(value):
            child = getattr(value, field.name)
            if child is value:
                continue
            yield from _iter_operations(child, f"{path}/{field.name}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_operations(child, f"{path}/{key}")
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for index, child in enumerate(value):
            yield from _iter_operations(child, f"{path}/{index}")
        return
    if is_dataclass(value):
        for field in fields(value):
            yield from _iter_operations(
                getattr(value, field.name),
                f"{path}/{field.name}",
            )


def _rule_roots(
    rulebook: RuleBook,
) -> Iterable[tuple[int | None, str, object]]:
    for (card_id, trigger), operations in sorted(
        rulebook._rules.items(),
        key=lambda item: (item[0][0], item[0][1].value),
    ):
        yield card_id, f"rule:{trigger.value}", operations
    for card_id, modes in sorted(rulebook._play_modes.items()):
        yield card_id, "play_modes", modes
    for card_id, definitions in sorted(rulebook._listener_defs.items()):
        yield card_id, "listeners", definitions
    for card_id, definitions in sorted(rulebook._union_burst_defs.items()):
        yield card_id, "union_burst", definitions
    for card_id, definition in sorted(rulebook._activation_defs.items()):
        yield card_id, "activation", definition
    for card_id, definition in sorted(rulebook._faith_defs.items()):
        yield card_id, "faith", definition
    for card_id, definition in sorted(rulebook._fusion_defs.items()):
        yield card_id, "fusion", definition
    for card_id, definition in sorted(rulebook._invocation_defs.items()):
        yield card_id, "invocation", definition
    for emblem_id, definition in sorted(rulebook._emblem_defs.items()):
        yield None, f"emblem:{emblem_id}", definition


def _target_category(operation: EffectOperation) -> str:
    if operation.kind in DECISION_KINDS:
        return "decision"
    if is_choice_target(operation.target):
        return "manual"
    if is_random_target(operation.target):
        return "random"
    if is_all_target(operation.target):
        return "all"
    return "implicit_or_bound"


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
        block = entry.get(key, {})
        if isinstance(block, Mapping):
            evidence = block.get("test_evidence", [])
            if isinstance(evidence, list) and evidence:
                return sorted(str(path) for path in evidence)
    return []


def _inventory(
    cards: tuple[CardDefinition, ...],
    rulebook: RuleBook,
    closure_ids: set[int],
    coverage: Mapping[str, object],
    demo_sources: Mapping[int, Mapping[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    sources: dict[int, list[dict[str, object]]] = {}
    global_sources: list[dict[str, object]] = []
    issues: list[str] = []
    for card_id, root, value in _rule_roots(rulebook):
        for path, operation in _iter_operations(value, root):
            category = _target_category(operation)
            if (
                category == "implicit_or_bound"
                and operation.kind not in DECISION_KINDS
                and operation.target
                in {
                    TargetKind.SELF,
                    TargetKind.EMBLEM_SELF,
                    TargetKind.EVENT_SOURCE,
                    TargetKind.ATTACK_TARGET,
                    TargetKind.OWN_LEADER,
                    TargetKind.ENEMY_LEADER,
                }
            ):
                continue
            record = {
                "root": root,
                "path": path,
                "kind": operation.kind.value,
                "target": operation.target.value,
                "category": category,
                "requires_target": operation.requires_target,
                "target_count": operation.target_count,
                "allow_duplicate_targets": (
                    operation.allow_duplicate_targets
                ),
                "requires_full_target_count": (
                    operation.requires_full_target_count
                ),
                "exclude_source": operation.exclude_source,
                "target_key": operation.target_key,
            }
            if card_id is None:
                global_sources.append(record)
            else:
                sources.setdefault(card_id, []).append(record)

    rows: list[dict[str, object]] = []
    cards_by_id = {card.card_id: card for card in cards}
    for card_id, operations in sorted(sources.items()):
        card = cards_by_id.get(card_id)
        row_issues: list[str] = []
        demo_source = demo_sources.get(card_id)
        synthetic_demo = card is None and demo_source is not None
        if card is None:
            collectible = False
            if synthetic_demo:
                name = f"synthetic-demo-{card_id}"
                card_type = "synthetic"
            else:
                name = f"unknown-{card_id}"
                card_type = "unknown"
                row_issues.append(
                    "rule source card is absent from the database and is "
                    "not declared by an explicit *_demo.json fixture"
                )
        else:
            collectible = card.is_collectible
            name = card.name
            card_type = card.card_type
        entry = _coverage_entry(coverage, card_id)
        evidence = (
            [str(demo_source["test_evidence"])]
            if synthetic_demo
            else _test_evidence(entry)
        )
        if not synthetic_demo:
            accepted_coverage = (
                {"covered_exact"}
                if collectible
                else {"token_or_non_collectible"}
            )
            if entry.get("coverage") not in accepted_coverage:
                row_issues.append(
                    "card lacks the required collectible/generated coverage status"
                )
        if not evidence:
            row_issues.append("card lacks permanent test evidence")
        for path in evidence:
            if not _repo_path(Path(path)).is_file():
                row_issues.append(f"missing test evidence file: {path}")
        if row_issues:
            issues.extend(
                f"card {card_id}: {message}" for message in row_issues
            )
        rows.append(
            {
                "card_id": card_id,
                "name": name,
                "card_type": card_type,
                "collectible": collectible,
                "synthetic_demo": synthetic_demo,
                "demo_rule_file": (
                    str(demo_source["rule_file"])
                    if synthetic_demo
                    else None
                ),
                "training_closure": card_id in closure_ids,
                "operations": sorted(
                    operations,
                    key=lambda row: (
                        str(row["root"]),
                        str(row["path"]),
                        str(row["kind"]),
                    ),
                ),
                "test_evidence": evidence,
                "issues": row_issues,
                "passed": not row_issues,
            }
        )
    return rows, global_sources, sorted(set(issues))


def _demo_source_index(rules_path: Path) -> dict[int, dict[str, str]]:
    """Index explicitly named synthetic fixtures without masking bad real IDs."""

    result: dict[int, dict[str, str]] = {}
    for path in sorted(rules_path.glob("*_demo.json")):
        evidence = DEMO_TEST_EVIDENCE.get(path.name)
        if evidence is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            continue
        for value in payload.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, Mapping):
                    continue
                card_id = item.get("card_id")
                if isinstance(card_id, int):
                    result[card_id] = {
                        "rule_file": str(path.relative_to(ROOT)),
                        "test_evidence": evidence,
                    }
    return result


def _contract_groups() -> dict[str, object]:
    return {
        "candidate_domains": _candidate_domain_contracts(),
        "candidate_cardinalities": _cardinality_contracts(),
        "source_exclusion": _source_exclusion_contract(),
        "restrictions": _restriction_contracts(),
        "multi_target": _multi_target_contracts(),
        "stale_targets": _stale_target_contracts(),
        "source_leaving": _source_leave_contracts(),
        "mixed_target_order": _mixed_target_order_contract(),
        "no_candidate_policies": _no_candidate_policy_contracts(),
        "snapshot_restore": _snapshot_restore_contract(),
        "action_order": _action_order_contract(),
    }


def _contract_failures(
    groups: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    for group_name, payload in groups.items():
        rows = payload if isinstance(payload, list) else [payload]
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or not row.get("passed"):
                case = (
                    row.get("case", row.get("target", index))
                    if isinstance(row, Mapping)
                    else index
                )
                failures.append(f"{group_name}:{case}")
    return failures


def build_report(
    *,
    database: Path = DEFAULT_DATABASE,
    rules: Path = DEFAULT_RULES,
    closure: Path = DEFAULT_CLOSURE,
    coverage_report: Path = DEFAULT_COVERAGE,
) -> dict[str, object]:
    database_path = _repo_path(database)
    rules_path = _repo_path(rules)
    closure_path = _repo_path(closure)
    coverage_path = _repo_path(coverage_report)

    repository = CardRepository(database_path)
    cards = repository.all_cards()
    rulebook = RuleBook.from_directory(rules_path)
    closure_payload = json.loads(closure_path.read_text(encoding="utf-8"))
    closure_ids = {
        int(card_id)
        for card_id in closure_payload.get("closure_card_ids", [])
    }
    if not closure_ids:
        closure_ids = {
            int(row["card_id"])
            for row in closure_payload.get("cards", [])
            if isinstance(row, Mapping) and "card_id" in row
        }
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))

    inventory, global_sources, inventory_issues = _inventory(
        cards,
        rulebook,
        closure_ids,
        coverage,
        _demo_source_index(rules_path),
    )
    contracts = _contract_groups()
    contract_failures = _contract_failures(contracts)

    operation_counts = {category: 0 for category in TARGET_CATEGORIES}
    target_counts: dict[str, int] = {}
    for row in inventory:
        for operation in row["operations"]:
            category = str(operation["category"])
            operation_counts[category] += 1
            target = str(operation["target"])
            target_counts[target] = target_counts.get(target, 0) + 1
    for operation in global_sources:
        category = str(operation["category"])
        operation_counts[category] += 1
        target = str(operation["target"])
        target_counts[target] = target_counts.get(target, 0) + 1

    failures = inventory_issues + contract_failures
    production_inventory = [
        row for row in inventory if not row["synthetic_demo"]
    ]
    synthetic_demo_inventory = [
        row for row in inventory if row["synthetic_demo"]
    ]
    collectible_count = sum(card.is_collectible for card in cards)
    generated_count = len(cards) - collectible_count
    return {
        "schema_version": 1,
        "inputs": {
            "database": str(database),
            "database_sha256": _sha256(database_path),
            "rules": str(rules),
            "closure": str(closure),
            "closure_sha256": _sha256(closure_path),
            "coverage_report": str(coverage_report),
            "coverage_sha256": _sha256(coverage_path),
        },
        "scope": {
            "card_count": len(cards),
            "collectible_card_count": collectible_count,
            "generated_card_count": generated_count,
            "manual_target_kinds": [
                target.value for target in MANUAL_TARGETS
            ],
            "target_categories": list(TARGET_CATEGORIES),
        },
        "inventory": inventory,
        "global_sources": global_sources,
        "operation_counts_by_category": operation_counts,
        "operation_counts_by_target": dict(sorted(target_counts.items())),
        "contracts": contracts,
        "summary": {
            "source_card_count": len(production_inventory),
            "synthetic_demo_source_count": len(synthetic_demo_inventory),
            "collectible_source_card_count": sum(
                row["collectible"] for row in production_inventory
            ),
            "generated_source_card_count": sum(
                not row["collectible"] for row in production_inventory
            ),
            "training_source_card_count": sum(
                row["training_closure"] for row in production_inventory
            ),
            "global_source_count": len(global_sources),
            "inventory_issue_count": len(inventory_issues),
            "contract_failure_count": len(contract_failures),
            "failure_count": len(failures),
            "failures": failures,
            "passed": not failures,
        },
    }


def render_markdown(report: Mapping[str, object]) -> str:
    scope = report["scope"]
    summary = report["summary"]
    lines = [
        "# Target, choice, and pending-state audit",
        "",
        (
            f"- Result: **{'PASS' if summary['passed'] else 'FAIL'}**; "
            f"{summary['failure_count']} failures."
        ),
        (
            f"- Snapshot: {scope['card_count']} cards "
            f"({scope['collectible_card_count']} collectible / "
            f"{scope['generated_card_count']} generated)."
        ),
        (
            f"- Target/choice sources: {summary['source_card_count']} cards; "
            f"{summary['training_source_card_count']} in the training closure; "
            f"{summary['global_source_count']} global sources."
        ),
        (
            f"- Manual target kinds: "
            f"{len(scope['manual_target_kinds'])}; "
            f"contract failures: {summary['contract_failure_count']}."
        ),
        "",
        "## Operation categories",
        "",
        "| Category | Operations |",
        "|---|---:|",
    ]
    for category, count in report["operation_counts_by_category"].items():
        lines.append(f"| {category} | {count} |")

    lines.extend(
        [
            "",
            "## Manual target-domain matrix",
            "",
            "| Target | Empty | Populated | Result |",
            "|---|---:|---:|:---:|",
        ]
    )
    for row in report["contracts"]["candidate_domains"]:
        lines.append(
            f"| {row['target']} | {row['empty_candidate_count']} | "
            f"{row['populated_candidate_count']} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Behavioral contracts",
            "",
            "| Group | Case | Result |",
            "|---|---|:---:|",
        ]
    )
    for group_name, payload in report["contracts"].items():
        if group_name == "candidate_domains":
            continue
        rows = payload if isinstance(payload, list) else [payload]
        for index, row in enumerate(rows):
            case = row.get("case", row.get("target", index))
            lines.append(
                f"| {group_name} | {case} | "
                f"{'PASS' if row['passed'] else 'FAIL'} |"
            )

    lines.extend(
        [
            "",
            "## Source inventory",
            "",
            "| Card | Categories | Operations | Training | Result |",
            "|---|---|---:|:---:|:---:|",
        ]
    )
    for row in report["inventory"]:
        categories = sorted(
            {
                operation["category"]
                for operation in row["operations"]
            }
        )
        lines.append(
            f"| {row['card_id']} {row['name']} | "
            f"{', '.join(categories)} | {len(row['operations'])} | "
            f"{'yes' if row['training_closure'] else 'no'} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=DEFAULT_COVERAGE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    report = build_report(
        database=args.database,
        rules=args.rules,
        closure=args.closure,
        coverage_report=args.coverage_report,
    )
    output = _repo_path(args.output)
    markdown = _repo_path(args.markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown.write_text(render_markdown(report), encoding="utf-8")
    summary = report["summary"]
    print(
        "cards={cards} sources={sources} training_sources={training} "
        "failures={failures} passed={passed}".format(
            cards=report["scope"]["card_count"],
            sources=summary["source_card_count"],
            training=summary["training_source_card_count"],
            failures=summary["failure_count"],
            passed=summary["passed"],
        )
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
