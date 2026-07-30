from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.card_audit_sampling import (
    DEFAULT_CHECKPOINT,
    MULLIGAN_POLICY_CURVE,
    SAMPLING_RANDOM,
    _game_decks,
    _run_game,
    build_full_pool_specs,
)
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.environment import MATCH_SETUP_OFFICIAL, ShadowverseEnv
from swb.engine.resolution import GameEngine
from swb.rl.catalog import TrainableCardCatalog
from swb.rl.checkpoint import load_checkpoint
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.versioning import stable_json_sha256


def _entity_row(entity: Any) -> dict[str, object]:
    return {
        "entity_id": entity.entity_id,
        "card_id": entity.definition.card_id,
        "name": entity.definition.name,
        "attack": getattr(entity, "attack", None),
        "health": getattr(entity, "health", None),
        "max_health": getattr(entity, "max_health", None),
        "base_attack": getattr(entity, "base_attack", None),
        "base_health": getattr(entity, "base_health", None),
        "stat_modifiers": [
            asdict(modifier)
            for modifier in getattr(entity, "stat_modifiers", ())
        ],
        "evolved": getattr(entity, "evolved", False),
        "super_evolved": getattr(entity, "super_evolved", False),
    }


def _state_row(env: ShadowverseEnv) -> dict[str, object]:
    core = env.core
    return {
        "turn": core.turn,
        "active_player": core.state.active_player,
        "decision_player": env.decision_player,
        "agent_steps": env.agent_steps,
        "players": [
            {
                "health": player.health,
                "mana": player.mana,
                "max_mana": player.max_mana,
                "hand": [
                    {
                        "entity_id": card.entity_id,
                        "card_id": card.definition.card_id,
                        "name": card.definition.name,
                        "current_cost": card.current_cost,
                    }
                    for card in player.hand
                ],
                "board": [_entity_row(entity) for entity in player.board],
                "graveyard_tail": [
                    {
                        "card_id": item.definition.card_id,
                        "name": item.definition.name,
                    }
                    for item in player.graveyard[-10:]
                ],
            }
            for player in core.players
        ],
        "event_history_tail": [
            {
                "type": event.type.value,
                "player_index": event.player_index,
                "source_id": event.source_id,
                "target_id": event.target_id,
                "amount": event.amount,
                "metadata": event.metadata,
            }
            for event in core.event_history[-50:]
        ],
        "pending_choice": repr(core.state.pending_choice),
        "effect_stack": repr(core.state.effect_stack),
        "fingerprint_sha256": stable_json_sha256(
            core.deterministic_fingerprint()
        ),
    }


def reproduce_failure(
    *,
    game_id: int,
    database: Path,
    checkpoint: Path,
    master_seed: int,
    max_game_turns: int,
    max_agent_steps: int,
    capture_after_action_count: int | None = None,
) -> dict[str, object]:
    specs = build_full_pool_specs(master_seed=master_seed)
    if game_id < 0 or game_id >= len(specs):
        raise ValueError(f"game_id must be between 0 and {len(specs) - 1}")
    spec = specs[game_id]
    snapshot = WorkerAssetsSnapshot.build(CardRepository(database))
    assets = snapshot.load()
    trainer = (
        None
        if spec.sampling_kind == SAMPLING_RANDOM
        else load_checkpoint(
            checkpoint,
            snapshot,
            device="cpu",
            restore_rng_state=False,
        )
    )
    deck_a, deck_b = _game_decks(spec, snapshot)
    actions: list[int] = []
    command_trace: list[dict[str, object]] = []
    current_command: dict[str, object] = {}
    failure_state: dict[str, object] | None = None
    captured_state_after_action: dict[str, object] | None = None
    active_env: ShadowverseEnv | None = None
    original_step = ShadowverseEnv.step
    original_assert_invariants = GameEngine.assert_invariants

    def capture_step(env: ShadowverseEnv, action: int):
        nonlocal active_env, captured_state_after_action
        active_env = env
        command = env._decode_action(action)
        current_command.clear()
        current_command.update({
            "step": env.agent_steps,
            "turn": env.turn,
            "decision_player": env.decision_player,
            "action": action,
            "command_type": type(command).__name__,
            "command": repr(command),
        })
        actions.append(action)
        command_trace.append(dict(current_command))
        result = original_step(env, action)
        if len(actions) == capture_after_action_count:
            captured_state_after_action = {
                "action_count": len(actions),
                "command": dict(current_command),
                "state": _state_row(env),
            }
        return result

    def capture_failing_invariants(engine) -> None:
        nonlocal failure_state
        try:
            original_assert_invariants(engine)
        except Exception:
            failure_state = {
                "command": dict(current_command),
                "state": (
                    _state_row(active_env)
                    if active_env is not None
                    else None
                ),
            }
            raise

    ShadowverseEnv.step = capture_step
    GameEngine.assert_invariants = capture_failing_invariants
    exception: dict[str, object] | None = None
    try:
        _run_game(
            spec,
            snapshot,
            rulebook=assets.rulebook,
            trainer=trainer,
            max_game_turns=max_game_turns,
            max_agent_steps=max_agent_steps,
        )
    except Exception as exc:
        exception = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        ShadowverseEnv.step = original_step
        GameEngine.assert_invariants = original_assert_invariants
        if trainer is not None:
            trainer.close()

    action_bytes = b"".join(action.to_bytes(2, "big") for action in actions)
    return {
        "schema_version": 1,
        "report_kind": "swb_card_audit_failure_reproduction",
        "spec": asdict(spec),
        "configuration": {
            "master_seed": master_seed,
            "max_game_turns": max_game_turns,
            "max_agent_steps": max_agent_steps,
            "capture_after_action_count": capture_after_action_count,
            "match_setup": MATCH_SETUP_OFFICIAL,
            "mulligan_policy": MULLIGAN_POLICY_CURVE,
            "validate_invariants": True,
        },
        "inputs": {
            "database": database.as_posix(),
            "checkpoint": checkpoint.as_posix(),
            "rulebook_sha256": snapshot.rulebook_sha256,
        },
        "decks": {
            "player_1": [card.card_id for card in deck_a],
            "player_2": [card.card_id for card in deck_b],
        },
        "exception": exception,
        "action_count": len(actions),
        "actions": actions,
        "action_trace_sha256": hashlib.sha256(action_bytes).hexdigest(),
        "command_trace": command_trace,
        "failure_state_before_rollback": failure_state,
        "captured_state_after_action": captured_state_after_action,
        "reproduced": exception is not None and failure_state is not None,
    }


def replay_saved_failure(
    *,
    source_report: Path,
    database: Path,
    capture_after_action_count: int | None = None,
) -> dict[str, object]:
    """Replay the exact deck order, seed, and action prefix from a failure."""

    source_bytes = source_report.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    failure = source.get("failure")
    if not isinstance(failure, dict):
        raise ValueError("source report has no failure object")
    required = ("deck_a", "deck_b", "class_a", "class_b", "game_seed", "actions")
    missing_fields = [field for field in required if field not in failure]
    if missing_fields:
        raise ValueError(
            "source failure is missing fields: " + ", ".join(missing_fields)
        )

    repository = CardRepository(database)
    catalog = TrainableCardCatalog.from_repository(repository)

    def resolve_deck(field: str) -> list[object]:
        deck = []
        for card_id in failure[field]:
            card = catalog.resolve(int(card_id))
            if card is None:
                raise ValueError(f"{field} references unknown card {card_id}")
            deck.append(card)
        return deck

    deck_a = resolve_deck("deck_a")
    deck_b = resolve_deck("deck_b")
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=int(failure["class_a"]),
        class_b=int(failure["class_b"]),
        seed=int(failure["game_seed"]),
        rulebook=RuleBook.from_directory(ShadowverseEnv.DEFAULT_RULE_DIRECTORY),
        card_resolver=catalog.resolve,
        validate_invariants=True,
        match_setup=str(source.get("match_setup", MATCH_SETUP_OFFICIAL)),
    )
    env.reset()
    command_trace: list[dict[str, object]] = []
    illegal_action_indices: list[int] = []
    exception: dict[str, object] | None = None
    failure_state: dict[str, object] | None = None
    captured_state_after_action: dict[str, object] | None = None
    actions_replayed: list[int] = []

    for action_index, raw_action in enumerate(failure["actions"]):
        action = int(raw_action)
        mask = env.action_mask()
        if action < 0 or action >= len(mask) or not mask[action]:
            illegal_action_indices.append(action_index)
        command = env._decode_action(action)
        command_row = {
            "action_index": action_index,
            "turn": env.turn,
            "decision_player": env.decision_player,
            "action": action,
            "command_type": type(command).__name__,
            "command": repr(command),
        }
        command_trace.append(command_row)
        actions_replayed.append(action)
        try:
            env.step(action)
        except Exception as exc:
            exception = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            failure_state = {
                "command": command_row,
                "state": _state_row(env),
            }
            break
        if len(actions_replayed) == capture_after_action_count:
            captured_state_after_action = {
                "action_count": len(actions_replayed),
                "command": command_row,
                "state": _state_row(env),
            }

    action_bytes = b"".join(
        action.to_bytes(2, "big") for action in actions_replayed
    )
    return {
        "schema_version": 1,
        "report_kind": "swb_card_audit_saved_failure_replay",
        "source": {
            "path": source_report.as_posix(),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "game_index": failure.get("game_index"),
            "game_seed": failure["game_seed"],
            "source_action_index": failure.get("action_index"),
            "source_exception_type": failure.get("exception_type"),
            "source_exception": failure.get("exception"),
        },
        "configuration": {
            "database": database.as_posix(),
            "match_setup": source.get("match_setup", MATCH_SETUP_OFFICIAL),
            "validate_invariants": True,
            "capture_after_action_count": capture_after_action_count,
        },
        "decks": {
            "class_a": failure["class_a"],
            "class_b": failure["class_b"],
            "player_1": failure["deck_a"],
            "player_2": failure["deck_b"],
        },
        "exception": exception,
        "action_count": len(actions_replayed),
        "actions": actions_replayed,
        "action_trace_sha256": hashlib.sha256(action_bytes).hexdigest(),
        "command_trace": command_trace,
        "illegal_action_indices": illegal_action_indices,
        "failure_state_before_rollback": failure_state,
        "captured_state_after_action": captured_state_after_action,
        "final_state": _state_row(env),
        "reproduced": exception is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce one deterministic checklist 1.12 sampling failure"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--game-id", type=int)
    source.add_argument(
        "--source-report",
        type=Path,
        help="Replay the exact saved deck order, seed, and action prefix.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/cards.sqlite3"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--master-seed", type=int, default=120012)
    parser.add_argument("--max-game-turns", type=int, default=200)
    parser.add_argument("--max-agent-steps", type=int, default=2000)
    parser.add_argument(
        "--capture-after-action-count",
        type=int,
        help="Save the complete state immediately after this many actions.",
    )
    parser.add_argument(
        "--expect",
        choices=("failure", "success", "either"),
        default="failure",
        help="Expected replay outcome (default: failure).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source_report is not None:
        report = replay_saved_failure(
            source_report=args.source_report,
            database=args.database,
            capture_after_action_count=args.capture_after_action_count,
        )
        source_label = args.source_report.as_posix()
    else:
        report = reproduce_failure(
            game_id=args.game_id,
            database=args.database,
            checkpoint=args.checkpoint,
            master_seed=args.master_seed,
            max_game_turns=args.max_game_turns,
            max_agent_steps=args.max_agent_steps,
            capture_after_action_count=args.capture_after_action_count,
        )
        source_label = f"game_id={args.game_id}"
    expectation_met = (
        args.expect == "either"
        or (args.expect == "failure" and report["reproduced"])
        or (args.expect == "success" and report["exception"] is None)
    )
    report["expected_outcome"] = args.expect
    report["expectation_met"] = expectation_met
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"source={source_label} reproduced={report['reproduced']} "
        f"actions={report['action_count']} expectation_met={expectation_met} "
        f"output={args.output.as_posix()}"
    )
    if not expectation_met:
        raise SystemExit(
            f"expected {args.expect} outcome, got "
            f"{'failure' if report['exception'] else 'success'}"
        )


if __name__ == "__main__":
    main()
