from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import GameCommand, SuperEvolve
from swb.engine.environment import MATCH_SETUP_OFFICIAL, ShadowverseEnv
from swb.engine.state import StatModifier, Unit
from swb.rl.catalog import TrainableCardCatalog
from swb.rl.runtime import hash_rule_directory
from swb.rl.versioning import stable_json_sha256


DEFAULT_SOURCE_REPORT = Path(
    "data/reports/card_bug_audit/reproductions/"
    "SWB-CARD-0008-random-self-play.json"
)
DEFAULT_BUG_REPORT = Path(
    "data/reports/card_bug_audit/reproductions/SWB-CARD-0008.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/card_bug_audit/repros/SWB-CARD-0008.json"
)
DEFAULT_SYNTHETIC_OUTPUT = Path(
    "data/reports/card_bug_audit/repros/SWB-CARD-0008-synthetic.json"
)
DEFAULT_DATABASE = Path("data/cards.sqlite3")
TARGET_CARD_ID = 10154120
TARGET_ENTITY_ID = 69


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_native(value: object) -> object:
    """Convert supported audit values to JSON-native objects without repr."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(_json_native(key)): _json_native(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_json_native(item) for item in value),
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
    raise TypeError(
        "portable reproduction contains unsupported in-process object "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _json_round_trip(value: object) -> object:
    native = _json_native(value)
    return json.loads(
        json.dumps(
            native,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _command_manifest(action: int, command: GameCommand) -> dict[str, object]:
    return {
        "action": action,
        "command_type": type(command).__name__,
        "parameters": _json_round_trip(asdict(command)),
    }


def _unit_projection(
    env: ShadowverseEnv,
    *,
    entity_id: int = TARGET_ENTITY_ID,
) -> dict[str, object] | None:
    for owner, player in enumerate(env.core.players):
        for board_index, entity in enumerate(player.board):
            if entity.entity_id != entity_id or not isinstance(entity, Unit):
                continue
            return {
                "owner": owner,
                "board_index": board_index,
                "entity_id": entity.entity_id,
                "card_id": entity.definition.card_id,
                "card_name": entity.definition.name,
                "attack": entity.attack,
                "health": entity.health,
                "max_health": entity.max_health,
                "base_attack": entity.base_attack,
                "base_health": entity.base_health,
                "evolved": entity.evolved,
                "super_evolved": entity.super_evolved,
                "stat_modifiers": [
                    _json_round_trip(asdict(modifier))
                    for modifier in entity.stat_modifiers
                ],
            }
    return None


def _load_failure(source_report: Path) -> tuple[dict[str, object], bytes]:
    source_bytes = source_report.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    failure = source.get("failure")
    if not isinstance(failure, dict):
        raise ValueError(f"{source_report} has no failure object")
    required = (
        "deck_a",
        "deck_b",
        "class_a",
        "class_b",
        "game_seed",
        "actions",
    )
    missing = [field for field in required if field not in failure]
    if missing:
        raise ValueError(
            f"{source_report} failure is missing: {', '.join(missing)}"
        )
    return source, source_bytes


@dataclass(frozen=True)
class ReplayAssets:
    catalog: TrainableCardCatalog
    rulebook: RuleBook
    class_a: int
    class_b: int
    deck_a: tuple[CardDefinition, ...]
    deck_b: tuple[CardDefinition, ...]
    game_seed: int
    match_setup: str

    @classmethod
    def from_source(
        cls,
        source: dict[str, object],
        *,
        database: Path,
    ) -> ReplayAssets:
        failure = source["failure"]
        assert isinstance(failure, dict)
        repository = CardRepository(database)
        catalog = TrainableCardCatalog.from_repository(repository)

        def resolve_deck(field: str) -> tuple[CardDefinition, ...]:
            result = []
            for raw_card_id in failure[field]:
                card_id = int(raw_card_id)
                definition = catalog.resolve(card_id)
                if definition is None:
                    raise ValueError(
                        f"{field} references unknown card {card_id}"
                    )
                result.append(definition)
            return tuple(result)

        return cls(
            catalog=catalog,
            rulebook=RuleBook.from_directory(
                ShadowverseEnv.DEFAULT_RULE_DIRECTORY
            ),
            class_a=int(failure["class_a"]),
            class_b=int(failure["class_b"]),
            deck_a=resolve_deck("deck_a"),
            deck_b=resolve_deck("deck_b"),
            game_seed=int(failure["game_seed"]),
            match_setup=str(
                source.get("match_setup", MATCH_SETUP_OFFICIAL)
            ),
        )

    def environment(self) -> ShadowverseEnv:
        env = ShadowverseEnv(
            self.deck_a,
            self.deck_b,
            class_a=self.class_a,
            class_b=self.class_b,
            seed=self.game_seed,
            rulebook=self.rulebook,
            card_resolver=self.catalog.resolve,
            validate_invariants=True,
            match_setup=self.match_setup,
        )
        env.reset()
        return env


@dataclass
class MinimizationResult:
    original_action_count: int
    minimized_action_count: int
    attempts: int
    actions: list[int]
    deletions: list[dict[str, object]]
    natural_reduction_found: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def minimize_action_sequence(
    actions: Sequence[int],
    reproduces: Callable[[Sequence[int]], bool],
) -> MinimizationResult:
    """Delta-debug an action trace, trying early chunks before later chunks."""

    current = list(actions)
    attempts = 0
    accepted_deletions: list[dict[str, object]] = []
    granularity = 2
    while len(current) >= 2:
        chunk_size = (len(current) + granularity - 1) // granularity
        reduced = False
        starts = list(range(0, len(current), chunk_size))
        for start in starts:
            stop = min(len(current), start + chunk_size)
            candidate = current[:start] + current[stop:]
            attempts += 1
            if not candidate or not reproduces(candidate):
                continue
            accepted_deletions.append(
                {
                    "start": start,
                    "stop": stop,
                    "removed_count": stop - start,
                    "remaining_count": len(candidate),
                    "was_early_prefix": start == 0,
                }
            )
            current = candidate
            granularity = max(2, granularity - 1)
            reduced = True
            break
        if reduced:
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)
    return MinimizationResult(
        original_action_count=len(actions),
        minimized_action_count=len(current),
        attempts=attempts,
        actions=current,
        deletions=accepted_deletions,
        natural_reduction_found=len(current) < len(actions),
    )


def _replay_prefix(
    assets: ReplayAssets,
    actions: Sequence[int],
    *,
    stop_before_last: bool,
) -> tuple[ShadowverseEnv, list[int]]:
    env = assets.environment()
    illegal_indices: list[int] = []
    replay_actions = actions[:-1] if stop_before_last else actions
    for index, raw_action in enumerate(replay_actions):
        action = int(raw_action)
        mask = env.action_mask()
        if action < 0 or action >= len(mask) or not bool(mask[action]):
            illegal_indices.append(index)
            break
        env.step(action)
    return env, illegal_indices


def _scenario_reproduces(
    assets: ReplayAssets,
    actions: Sequence[int],
) -> bool:
    if not actions or int(actions[-1]) != 107:
        return False
    try:
        env, illegal = _replay_prefix(
            assets,
            actions,
            stop_before_last=True,
        )
        if illegal:
            return False
        final_action = int(actions[-1])
        mask = env.action_mask()
        if final_action >= len(mask) or not bool(mask[final_action]):
            return False
        command = env._decode_action(final_action)
        before = _unit_projection(env)
        if (
            not isinstance(command, SuperEvolve)
            or command.unit_id != TARGET_ENTITY_ID
            or before is None
            or before["card_id"] != TARGET_CARD_ID
            or (
                before["attack"],
                before["health"],
                before["max_health"],
            )
            != (5, 1, 1)
        ):
            return False
        env.step(final_action)
        after = _unit_projection(env)
        return (
            after is not None
            and (
                after["attack"],
                after["health"],
                after["max_health"],
            )
            == (8, 4, 4)
            and after["super_evolved"] is True
        )
    except Exception:
        return False


def synthetic_fixture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "report_kind": "swb_card_bug_synthetic_fixture",
        "bug_id": "SWB-CARD-0008",
        "purpose": (
            "Minimal fallback after the state-dependent natural action trace "
            "cannot be shortened while remaining a legal match."
        ),
        "fixture": {
            "target": {
                "card_id": TARGET_CARD_ID,
                "printed_attack": 3,
                "printed_health": 7,
            },
            "operations": [
                {
                    "kind": "add_stat_modifier",
                    "attack_delta": 2,
                    "health_delta": -2,
                    "duration": "permanent",
                },
                {
                    "kind": "set_stats",
                    "attack": None,
                    "health": 1,
                },
                {
                    "kind": "add_stat_modifier",
                    "attack_delta": 3,
                    "health_delta": 3,
                    "duration": "permanent",
                    "represents": "super_evolution",
                },
            ],
            "expected": {
                "attack": 8,
                "health": 4,
                "max_health": 4,
                "remaining_prior_modifier": {
                    "attack_delta": 2,
                    "health_delta": 0,
                },
            },
        },
        "would_fail_before_fix": {
            "engine_commit": "bb5635b58709e0c3e6cf5486f6708530f47be3f2",
            "result": {
                "attack": 8,
                "health": 4,
                "max_health": 2,
            },
            "exception": (
                "Invariant failed: player 1 board[1] health exceeds max_health"
            ),
        },
        "permanent_regression": (
            "tests/test_card_bug_repro_package.py::"
            "CardBugReproPackageTests::test_synthetic_fixture_executes"
        ),
    }


def execute_synthetic_fixture(
    fixture: dict[str, object],
) -> dict[str, object]:
    target = fixture["fixture"]["target"]
    unit = Unit.summon(
        CardDefinition(
            card_id=int(target["card_id"]),
            card_set_id=0,
            class_id=0,
            class_name="synthetic",
            name="SET_STATS synthetic target",
            cost=0,
            card_type="随从",
            attack=int(target["printed_attack"]),
            life=int(target["printed_health"]),
            keywords=frozenset(),
            support_level="synthetic",
            is_collectible=False,
        ),
        entity_id=1,
    )
    for operation in fixture["fixture"]["operations"]:
        if operation["kind"] == "add_stat_modifier":
            unit.add_stat_modifier(
                StatModifier(
                    modifier_id=len(unit.stat_modifiers) + 1,
                    attack_delta=int(operation["attack_delta"]),
                    health_delta=int(operation["health_delta"]),
                    duration=str(operation["duration"]),
                )
            )
        elif operation["kind"] == "set_stats":
            unit.set_stats(
                attack=operation["attack"],
                health=int(operation["health"]),
            )
        else:
            raise ValueError(f"unsupported fixture operation {operation!r}")
    return {
        "attack": unit.attack,
        "health": unit.health,
        "max_health": unit.max_health,
        "remaining_prior_modifier": {
            "attack_delta": unit.stat_modifiers[0].attack_delta,
            "health_delta": unit.stat_modifiers[0].health_delta,
        },
    }


def build_package(
    *,
    source_report: Path,
    bug_report: Path,
    database: Path,
    synthetic_output: Path,
) -> dict[str, object]:
    source, source_bytes = _load_failure(source_report)
    failure = source["failure"]
    assert isinstance(failure, dict)
    source_actions = [int(action) for action in failure["actions"]]
    if not source_actions:
        raise ValueError("source failure action trace is empty")
    assets = ReplayAssets.from_source(source, database=database)
    minimization = minimize_action_sequence(
        source_actions,
        lambda candidate: _scenario_reproduces(assets, candidate),
    )
    actions = minimization.actions

    env, illegal = _replay_prefix(
        assets,
        actions,
        stop_before_last=True,
    )
    if illegal:
        raise AssertionError(
            f"saved pre-command trace became illegal at {illegal}"
        )
    pre_fingerprint = _json_round_trip(
        env.core.deterministic_fingerprint()
    )
    pre_fingerprint_sha256 = stable_json_sha256(pre_fingerprint)
    action = actions[-1]
    mask = [bool(value) for value in env.action_mask()]
    if not mask[action]:
        raise AssertionError(f"saved final action {action} is not legal")
    command = env._decode_action(action)
    legal_actions = [
        _command_manifest(action_id, env._decode_action(action_id))
        for action_id, legal in enumerate(mask)
        if legal
    ]
    before = _unit_projection(env)
    event_count_before = len(env.core.event_history)
    env.step(action)
    after = _unit_projection(env)
    transition_events = [
        _json_round_trip(env.core._event_fingerprint(event))
        for event in env.core.event_history[event_count_before:]
    ]
    final_fingerprint = _json_round_trip(
        env.core.deterministic_fingerprint()
    )
    if (
        before is None
        or after is None
        or (
            after["attack"],
            after["health"],
            after["max_health"],
        )
        != (8, 4, 4)
    ):
        raise AssertionError(
            f"exact replay did not reach official result: {after}"
        )

    fixture = synthetic_fixture()
    synthetic_output.parent.mkdir(parents=True, exist_ok=True)
    synthetic_output.write_text(
        json.dumps(fixture, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fixture_result = execute_synthetic_fixture(fixture)
    if fixture_result != fixture["fixture"]["expected"]:
        raise AssertionError(
            "synthetic fixture does not reach expected result: "
            f"{fixture_result}"
        )

    bug = json.loads(bug_report.read_text(encoding="utf-8"))
    action_bytes = b"".join(
        item.to_bytes(2, "big") for item in actions
    )
    source_action_bytes = b"".join(
        item.to_bytes(2, "big") for item in source_actions
    )
    database_snapshot = _json_round_trip(assets.catalog.source_snapshot)
    package = {
        "schema_version": 1,
        "report_kind": "swb_card_bug_portable_reproduction",
        "bug_id": "SWB-CARD-0008",
        "portability": {
            "encoding": "UTF-8 JSON",
            "only_json_native_values": True,
            "requires_ui": False,
            "requires_original_process": False,
            "replay_source": (
                "database + structured rules + exact deck order + seed + "
                "integer action sequence"
            ),
        },
        "provenance": {
            "database": {
                "path": database.as_posix(),
                "sha256": file_sha256(database),
                "source_snapshot": database_snapshot,
            },
            "rules": {
                "path": Path(
                    ShadowverseEnv.DEFAULT_RULE_DIRECTORY
                ).as_posix(),
                "sha256": hash_rule_directory(
                    ShadowverseEnv.DEFAULT_RULE_DIRECTORY
                ),
            },
            "source_failure_report": {
                "path": source_report.as_posix(),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "discovery_commit": bug["discovery_commit"],
                "action_count": len(source_actions),
                "action_trace_sha256": hashlib.sha256(
                    source_action_bytes
                ).hexdigest(),
            },
            "official_fix_commit": (
                "b6f1d95cd2336cc86772e717e5bd09440a8f38a7"
            ),
        },
        "setup": {
            "match_setup": assets.match_setup,
            "class_a": assets.class_a,
            "class_b": assets.class_b,
            "game_seed": assets.game_seed,
            "decks": {
                "player_1": [
                    definition.card_id for definition in assets.deck_a
                ],
                "player_2": [
                    definition.card_id for definition in assets.deck_b
                ],
            },
        },
        "action_trace": {
            "actions": actions,
            "action_count": len(actions),
            "sha256": hashlib.sha256(action_bytes).hexdigest(),
            "pre_command_actions": actions[:-1],
            "command_action": action,
        },
        "pre_command": {
            "snapshot": pre_fingerprint,
            "snapshot_sha256": pre_fingerprint_sha256,
            "command": _command_manifest(action, command),
            "action_mask": mask,
            "action_mask_size": len(mask),
            "legal_action_count": sum(mask),
            "legal_actions": legal_actions,
            "target_before": before,
            "event_history_count": event_count_before,
        },
        "transition": {
            "events": transition_events,
            "event_count": len(transition_events),
            "target_after": after,
            "final_snapshot_sha256": stable_json_sha256(
                final_fingerprint
            ),
        },
        "expectation": {
            "official_expected": {
                "attack": 8,
                "health": 4,
                "max_health": 4,
                "super_evolved": True,
            },
            "pre_fix_actual": bug["pre_fix_actual"],
            "post_fix_actual": after,
            "official_evidence": bug["official_evidence"],
        },
        "minimization": {
            **minimization.to_dict(),
            "algorithm": (
                "delta debugging with prefix-first contiguous chunk deletion"
            ),
            "oracle": (
                "all retained actions legal; final command super-evolves "
                "entity 69/card 10154120 from 5/1/1 to 8/4/4"
            ),
            "fallback_required": not minimization.natural_reduction_found,
            "synthetic_fixture": synthetic_output.as_posix(),
        },
        "regression": {
            "package_replay_test": (
                "tests/test_card_bug_repro_package.py::"
                "CardBugReproPackageTests::test_portable_package_replays"
            ),
            "synthetic_fixture_test": fixture["permanent_regression"],
            "same_mechanic_real_card_tests": [
                (
                    "tests/test_real_generated_burst_spell_batch.py::"
                    "RealGeneratedBurstSpellTests::"
                    "test_ordinary_evolution_after_set_health_uses_"
                    "official_visible_plus_two"
                ),
                (
                    "tests/test_real_generated_burst_spell_batch.py::"
                    "RealGeneratedBurstSpellTests::"
                    "test_super_evolution_after_set_health_uses_"
                    "official_visible_plus_three"
                ),
                (
                    "tests/test_real_fusion_combo_cost_repose_"
                    "sixteenth_batch.py::"
                    "FusionComboCostReposeSixteenthBehaviorTests::"
                    "test_himeka_super_evolve_sets_attack_only_and_"
                    "seeded_target_is_reproducible"
                ),
                (
                    "tests/test_real_listener_spellboost_followup_"
                    "batch.py::"
                    "RealListenerSpellboostFollowupTests::"
                    "test_holy_knight_heals_per_positive_buff_but_not_"
                    "set_stats"
                ),
                (
                    "tests/test_real_spell_amulet_crest_batch.py::"
                    "RealSpellAmuletCrestBehaviorTests::"
                    "test_pascales_dance_draws_then_pays_ten_and_"
                    "doubles_current_stats"
                ),
            ],
            "card_id_special_cases_added": 0,
        },
    }
    validate_package(package)
    return package


def validate_package(package: dict[str, object]) -> None:
    if package.get("schema_version") != 1:
        raise ValueError("unsupported reproduction package schema")
    if package.get("report_kind") != "swb_card_bug_portable_reproduction":
        raise ValueError("unexpected reproduction package kind")
    _json_round_trip(package)
    portability = package["portability"]
    if (
        portability["requires_ui"]
        or portability["requires_original_process"]
        or not portability["only_json_native_values"]
    ):
        raise ValueError("reproduction package is not process-independent")
    trace = package["action_trace"]
    pre = package["pre_command"]
    if trace["action_count"] != len(trace["actions"]):
        raise ValueError("action count does not match action trace")
    if trace["command_action"] != trace["actions"][-1]:
        raise ValueError("command action is not the last trace action")
    mask = pre["action_mask"]
    if pre["action_mask_size"] != len(mask):
        raise ValueError("action mask size does not match mask")
    legal_ids = [row["action"] for row in pre["legal_actions"]]
    if legal_ids != [index for index, legal in enumerate(mask) if legal]:
        raise ValueError("legal action list disagrees with mask")
    if not mask[trace["command_action"]]:
        raise ValueError("saved command action is masked illegal")
    if pre["snapshot_sha256"] != stable_json_sha256(pre["snapshot"]):
        raise ValueError("pre-command snapshot hash mismatch")


def replay_package(
    package: dict[str, object],
    *,
    database: Path,
) -> dict[str, object]:
    validate_package(package)
    database_record = package["provenance"]["database"]
    rules_record = package["provenance"]["rules"]
    if file_sha256(database) != database_record["sha256"]:
        raise ValueError("database hash differs from reproduction package")
    if (
        hash_rule_directory(ShadowverseEnv.DEFAULT_RULE_DIRECTORY)
        != rules_record["sha256"]
    ):
        raise ValueError("rule hash differs from reproduction package")

    setup = package["setup"]
    repository = CardRepository(database)
    catalog = TrainableCardCatalog.from_repository(repository)

    def resolve(card_ids: Sequence[int]) -> tuple[CardDefinition, ...]:
        result = []
        for card_id in card_ids:
            definition = catalog.resolve(int(card_id))
            if definition is None:
                raise ValueError(f"unknown replay card {card_id}")
            result.append(definition)
        return tuple(result)

    assets = ReplayAssets(
        catalog=catalog,
        rulebook=RuleBook.from_directory(
            ShadowverseEnv.DEFAULT_RULE_DIRECTORY
        ),
        class_a=int(setup["class_a"]),
        class_b=int(setup["class_b"]),
        deck_a=resolve(setup["decks"]["player_1"]),
        deck_b=resolve(setup["decks"]["player_2"]),
        game_seed=int(setup["game_seed"]),
        match_setup=str(setup["match_setup"]),
    )
    actions = [int(action) for action in package["action_trace"]["actions"]]
    env, illegal = _replay_prefix(
        assets,
        actions,
        stop_before_last=True,
    )
    if illegal:
        raise AssertionError(f"pre-command replay became illegal: {illegal}")
    fingerprint = _json_round_trip(env.core.deterministic_fingerprint())
    snapshot_sha256 = stable_json_sha256(fingerprint)
    mask = [bool(value) for value in env.action_mask()]
    final_action = actions[-1]
    command = _command_manifest(
        final_action,
        env._decode_action(final_action),
    )
    if snapshot_sha256 != package["pre_command"]["snapshot_sha256"]:
        raise AssertionError("pre-command snapshot is not deterministic")
    if mask != package["pre_command"]["action_mask"]:
        raise AssertionError("pre-command action mask changed")
    if command != package["pre_command"]["command"]:
        raise AssertionError("pre-command decoded command changed")
    event_count = len(env.core.event_history)
    env.step(final_action)
    return {
        "illegal_action_indices": illegal,
        "pre_command_snapshot_sha256": snapshot_sha256,
        "command": command,
        "target_before": package["pre_command"]["target_before"],
        "target_after": _unit_projection(env),
        "events": [
            _json_round_trip(env.core._event_fingerprint(event))
            for event in env.core.event_history[event_count:]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build or validate the checklist 1.13 portable reproduction "
            "package."
        )
    )
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--bug-report", type=Path, default=DEFAULT_BUG_REPORT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--synthetic-output",
        type=Path,
        default=DEFAULT_SYNTHETIC_OUTPUT,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and replay an existing --output package.",
    )
    args = parser.parse_args()
    if args.validate_only:
        package = json.loads(args.output.read_text(encoding="utf-8"))
        replay = replay_package(package, database=args.database)
        print(
            f"validated={args.output.as_posix()} "
            f"snapshot={replay['pre_command_snapshot_sha256']} "
            f"target_after={replay['target_after']}"
        )
        return
    package = build_package(
        source_report=args.source_report,
        bug_report=args.bug_report,
        database=args.database,
        synthetic_output=args.synthetic_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(package, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"output={args.output.as_posix()} "
        f"actions={package['action_trace']['action_count']} "
        f"minimized={package['minimization']['minimized_action_count']} "
        f"attempts={package['minimization']['attempts']} "
        f"fallback={package['minimization']['fallback_required']}"
    )


if __name__ == "__main__":
    main()
