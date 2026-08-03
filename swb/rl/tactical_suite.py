from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from swb.rl.versioning import ACTION_LAYOUT_VERSION, stable_json_sha256
from swb.simulator.service import MatchSimulator, PolicyDecision


TACTICAL_CASE_SCHEMA_VERSION = 1
TACTICAL_REPORT_SCHEMA_VERSION = 1


class TacticalCaseError(ValueError):
    """Raised when a tactical case is malformed or cannot be replayed."""


class _PrefixReplaySimulator(MatchSimulator):
    """Match simulator that never auto-advances the checkpoint policy."""

    def _advance_ai(self, max_actions: int = 512) -> None:
        del max_actions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TacticalCaseError(f"{label} must be an object")
    return value


def _require_non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TacticalCaseError(f"{label} must be a non-empty string")
    return value


def _require_player(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
        raise TacticalCaseError(f"{label} must be player 0 or 1")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalized_prefix_actions(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    prefix = _require_mapping(case.get("prefix"), "prefix")
    raw_actions = prefix.get("actions")
    if not isinstance(raw_actions, list):
        raise TacticalCaseError("prefix.actions must be an array")
    normalized: list[dict[str, Any]] = []
    for index, raw_action in enumerate(raw_actions, start=1):
        action = _require_mapping(raw_action, f"prefix.actions[{index - 1}]")
        sequence = action.get("sequence")
        player_index = action.get("player_index")
        action_id = action.get("action_id")
        actor_role = action.get("actor_role")
        kind = action.get("kind")
        if sequence != index:
            raise TacticalCaseError(
                "prefix action sequences must be contiguous and start at 1"
            )
        _require_player(player_index, f"prefix.actions[{index - 1}].player_index")
        if not isinstance(action_id, int) or isinstance(action_id, bool) or action_id < 0:
            raise TacticalCaseError(
                f"prefix.actions[{index - 1}].action_id must be a non-negative integer"
            )
        _require_non_empty_string(
            actor_role,
            f"prefix.actions[{index - 1}].actor_role",
        )
        _require_non_empty_string(kind, f"prefix.actions[{index - 1}].kind")
        normalized.append(
            {
                "sequence": sequence,
                "player_index": player_index,
                "actor_role": actor_role,
                "action_id": action_id,
                "kind": kind,
            }
        )
    return normalized


def _validate_selector(selector: object, label: str) -> Mapping[str, Any]:
    item = _require_mapping(selector, label)
    _require_non_empty_string(item.get("kind"), f"{label}.kind")
    if "source_card_id" in item:
        card_id = item["source_card_id"]
        if not isinstance(card_id, int) or isinstance(card_id, bool) or card_id <= 0:
            raise TacticalCaseError(f"{label}.source_card_id must be a positive integer")
        occurrence = item.get("source_occurrence", 0)
        if (
            not isinstance(occurrence, int)
            or isinstance(occurrence, bool)
            or occurrence < 0
        ):
            raise TacticalCaseError(
                f"{label}.source_occurrence must be a non-negative integer"
            )
    if "target" in item and item["target"] not in ("leader", "board"):
        raise TacticalCaseError(f"{label}.target must be 'leader' or 'board'")
    return item


def validate_tactical_case(case: Mapping[str, Any]) -> None:
    if case.get("schema_version") != TACTICAL_CASE_SCHEMA_VERSION:
        raise TacticalCaseError(
            f"unsupported tactical case schema: {case.get('schema_version')!r}"
        )
    _require_non_empty_string(case.get("case_id"), "case_id")
    _require_non_empty_string(case.get("title"), "title")
    _require_non_empty_string(case.get("category"), "category")
    _require_non_empty_string(case.get("objective"), "objective")

    setup = _require_mapping(case.get("setup"), "setup")
    seed = setup.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TacticalCaseError("setup.seed must be an integer")
    human_player = _require_player(setup.get("human_player"), "setup.human_player")
    evaluated_player = _require_player(
        setup.get("evaluated_player"),
        "setup.evaluated_player",
    )
    if human_player == evaluated_player:
        raise TacticalCaseError("evaluated_player must be the non-human player")
    _require_non_empty_string(setup.get("human_deck"), "setup.human_deck")
    _require_non_empty_string(setup.get("ai_deck"), "setup.ai_deck")
    if setup.get("match_setup") != "official":
        raise TacticalCaseError("tactical cases currently require official match setup")

    actions = _normalized_prefix_actions(case)
    prefix = _require_mapping(case.get("prefix"), "prefix")
    expected_trace_hash = prefix.get("action_trace_sha256")
    if not _is_sha256(expected_trace_hash):
        raise TacticalCaseError("prefix.action_trace_sha256 must be a SHA-256 digest")
    actual_trace_hash = stable_json_sha256(actions)
    if actual_trace_hash != expected_trace_hash:
        raise TacticalCaseError(
            "prefix action trace hash mismatch: "
            f"expected {expected_trace_hash}, got {actual_trace_hash}"
        )
    if not _is_sha256(prefix.get("expected_state_sha256")):
        raise TacticalCaseError("prefix.expected_state_sha256 must be a SHA-256 digest")

    decision = _require_mapping(case.get("decision"), "decision")
    target_sequence = decision.get("target_sequence")
    if target_sequence != len(actions) + 1:
        raise TacticalCaseError(
            "decision.target_sequence must immediately follow the action prefix"
        )
    _require_player(decision.get("player_index"), "decision.player_index")
    if decision.get("player_index") != evaluated_player:
        raise TacticalCaseError("decision.player_index must equal setup.evaluated_player")
    if decision.get("grading") != "pairwise_preference":
        raise TacticalCaseError(
            "tactical cases currently require pairwise_preference grading"
        )
    preferred = decision.get("preferred")
    disfavored = decision.get("disfavored")
    if not isinstance(preferred, list) or not preferred:
        raise TacticalCaseError("decision.preferred must be a non-empty array")
    if not isinstance(disfavored, list) or not disfavored:
        raise TacticalCaseError("decision.disfavored must be a non-empty array")
    for index, selector in enumerate(preferred):
        _validate_selector(selector, f"decision.preferred[{index}]")
    for index, selector in enumerate(disfavored):
        _validate_selector(selector, f"decision.disfavored[{index}]")


def load_tactical_case(path: str | Path) -> dict[str, Any]:
    case_path = Path(path)
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TacticalCaseError(f"{case_path} must contain a JSON object")
    validate_tactical_case(payload)
    return payload


def _board_projection(simulator: MatchSimulator, player_index: int) -> list[dict[str, Any]]:
    player = simulator._require_env().core.players[player_index]
    return [
        {
            "entity_id": entity.entity_id,
            "card_id": entity.definition.card_id,
            "name": entity.definition.name,
            "board_index": board_index,
        }
        for board_index, entity in enumerate(player.board)
    ]


def resolve_action_selector(
    selector: Mapping[str, Any],
    *,
    legal_actions: Sequence[Mapping[str, Any]],
    own_board: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Resolve a semantic selector to exactly one current legal action."""

    _validate_selector(selector, "selector")
    source_entity_id: int | None = None
    if "source_card_id" in selector:
        matching_sources = [
            entity
            for entity in own_board
            if int(entity["card_id"]) == int(selector["source_card_id"])
        ]
        occurrence = int(selector.get("source_occurrence", 0))
        if occurrence >= len(matching_sources):
            raise TacticalCaseError(
                "selector source is absent from the evaluated player's board: "
                f"card {selector['source_card_id']} occurrence {occurrence}"
            )
        source_entity_id = int(matching_sources[occurrence]["entity_id"])

    matches: list[Mapping[str, Any]] = []
    for action in legal_actions:
        if action.get("kind") != selector["kind"]:
            continue
        if source_entity_id is not None and action.get("source_entity_id") != source_entity_id:
            continue
        if "mode_id" in selector and action.get("mode_id") != selector["mode_id"]:
            continue
        if "option_id" in selector and action.get("option_id") != selector["option_id"]:
            continue
        if selector.get("target") == "leader" and action.get("target_entity_id") is not None:
            continue
        if selector.get("target") == "board" and action.get("target_entity_id") is None:
            continue
        matches.append(action)
    if len(matches) != 1:
        raise TacticalCaseError(
            f"selector resolved to {len(matches)} legal actions instead of one: "
            f"{dict(selector)!r}"
        )
    return matches[0]


def _selector_result(
    selector: Mapping[str, Any],
    *,
    legal_actions: Sequence[Mapping[str, Any]],
    own_board: Sequence[Mapping[str, Any]],
    policy_decision: PolicyDecision,
) -> dict[str, Any]:
    action = resolve_action_selector(
        selector,
        legal_actions=legal_actions,
        own_board=own_board,
    )
    action_id = int(action["id"])
    return {
        "selector": dict(selector),
        "action_id": action_id,
        "label": action.get("label"),
        "logit": policy_decision.logits[action_id],
        "probability": policy_decision.probabilities[action_id],
    }


def evaluate_tactical_case(
    case: Mapping[str, Any],
    *,
    checkpoint: str | Path,
    database: str | Path = Path("data/cards.sqlite3"),
    card_catalog: str | Path = Path("shadowverse_cards.json"),
    image_directory: str | Path = Path("data/card_images"),
    device: str = "cpu",
) -> dict[str, Any]:
    """Teacher-force one replay prefix and score a checkpoint at its target."""

    validate_tactical_case(case)
    checkpoint_path = Path(checkpoint)
    setup = _require_mapping(case["setup"], "setup")
    decision_spec = _require_mapping(case["decision"], "decision")
    prefix = _require_mapping(case["prefix"], "prefix")
    actions = _normalized_prefix_actions(case)
    evaluated_player = int(setup["evaluated_player"])
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="swb-tactical-") as history_directory:
        load_started = time.perf_counter()
        simulator = _PrefixReplaySimulator(
            database=database,
            checkpoint=checkpoint_path,
            checkpoint_directory=checkpoint_path.parent,
            card_catalog=card_catalog,
            image_directory=image_directory,
            history_directory=history_directory,
            device=device,
        )
        simulator.new_match(
            seed=int(setup["seed"]),
            human_player=int(setup["human_player"]),
            human_deck=str(setup["human_deck"]),
            ai_deck=str(setup["ai_deck"]),
        )
        load_seconds = time.perf_counter() - load_started

        replay_started = time.perf_counter()
        prefix_policy_decisions = 0
        prefix_top1_matches = 0
        divergences: list[dict[str, Any]] = []
        for action_record in actions:
            env = simulator._require_env()
            sequence = int(action_record["sequence"])
            expected_player = int(action_record["player_index"])
            action_id = int(action_record["action_id"])
            if env.decision_player != expected_player:
                raise TacticalCaseError(
                    f"case {case['case_id']} sequence {sequence} expected player "
                    f"{expected_player}, got {env.decision_player}"
                )
            mask = env.action_mask()
            if action_id >= len(mask) or not bool(mask[action_id]):
                raise TacticalCaseError(
                    f"case {case['case_id']} sequence {sequence} action "
                    f"{action_id} is no longer legal"
                )
            if expected_player == evaluated_player:
                prefix_decision = simulator.policy.decision(env, evaluated_player)
                prefix_policy_decisions += 1
                if prefix_decision.action == action_id:
                    prefix_top1_matches += 1
                else:
                    divergences.append(
                        {
                            "sequence": sequence,
                            "recorded_action_id": action_id,
                            "checkpoint_top1_action_id": prefix_decision.action,
                        }
                    )
            env.step(action_id)

        actual_state = simulator._history_snapshot()
        actual_state_hash = stable_json_sha256(actual_state)
        expected_state_hash = str(prefix["expected_state_sha256"])
        if actual_state_hash != expected_state_hash:
            raise TacticalCaseError(
                f"case {case['case_id']} replay state hash mismatch: expected "
                f"{expected_state_hash}, got {actual_state_hash}"
            )
        env = simulator._require_env()
        if env.decision_player != evaluated_player:
            raise TacticalCaseError(
                f"case {case['case_id']} target belongs to player "
                f"{env.decision_player}, not evaluated player {evaluated_player}"
            )
        legal_actions = simulator._legal_actions()
        own_board = _board_projection(simulator, evaluated_player)
        target_decision = simulator.policy.decision(env, evaluated_player)
        replay_seconds = time.perf_counter() - replay_started

        preferred = [
            _selector_result(
                selector,
                legal_actions=legal_actions,
                own_board=own_board,
                policy_decision=target_decision,
            )
            for selector in decision_spec["preferred"]
        ]
        disfavored = [
            _selector_result(
                selector,
                legal_actions=legal_actions,
                own_board=own_board,
                policy_decision=target_decision,
            )
            for selector in decision_spec["disfavored"]
        ]
        preferred_ids = {int(item["action_id"]) for item in preferred}
        best_preferred = max(preferred, key=lambda item: float(item["logit"]))
        best_disfavored = max(disfavored, key=lambda item: float(item["logit"]))
        preferred_probability = sum(float(item["probability"]) for item in preferred)
        disfavored_probability = sum(
            float(item["probability"]) for item in disfavored
        )
        comparison_probability = preferred_probability / (
            preferred_probability + disfavored_probability
        )
        selected_action = next(
            action
            for action in legal_actions
            if int(action["id"]) == target_decision.action
        )
        pairwise_margin = float(best_preferred["logit"]) - float(
            best_disfavored["logit"]
        )
        pairwise_pass = pairwise_margin > 0.0
        top1_preferred = target_decision.action in preferred_ids

        return {
            "case_id": case["case_id"],
            "title": case["title"],
            "checkpoint": {
                "path": checkpoint_path.as_posix(),
                "sha256": file_sha256(checkpoint_path),
            },
            "replay": {
                "valid": True,
                "state_sha256": actual_state_hash,
                "prefix_action_count": len(actions),
                "prefix_policy_decisions": prefix_policy_decisions,
                "prefix_top1_matches_recorded": prefix_top1_matches,
                "prefix_top1_match_rate": (
                    1.0
                    if prefix_policy_decisions == 0
                    else prefix_top1_matches / prefix_policy_decisions
                ),
                "teacher_forced_divergences": divergences,
            },
            "target": {
                "sequence": int(decision_spec["target_sequence"]),
                "value": target_decision.value,
                "selected_action": {
                    "action_id": target_decision.action,
                    "kind": selected_action.get("kind"),
                    "label": selected_action.get("label"),
                    "probability": target_decision.probabilities[
                        target_decision.action
                    ],
                },
                "preferred": preferred,
                "disfavored": disfavored,
                "preferred_probability": preferred_probability,
                "disfavored_probability": disfavored_probability,
                "comparison_preferred_probability": comparison_probability,
                "pairwise_logit_margin": pairwise_margin,
                "pairwise_pass": pairwise_pass,
                "top1_preferred": top1_preferred,
                "pass": pairwise_pass and top1_preferred,
            },
            "timing_seconds": {
                "load_and_setup": load_seconds,
                "prefix_and_target": replay_seconds,
                "total": time.perf_counter() - started,
            },
        }


def build_tactical_report(
    cases: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[str | Path],
    *,
    database: str | Path = Path("data/cards.sqlite3"),
    card_catalog: str | Path = Path("shadowverse_cards.json"),
    image_directory: str | Path = Path("data/card_images"),
    device: str = "cpu",
) -> dict[str, Any]:
    results = [
        evaluate_tactical_case(
            case,
            checkpoint=checkpoint,
            database=database,
            card_catalog=card_catalog,
            image_directory=image_directory,
            device=device,
        )
        for checkpoint in checkpoints
        for case in cases
    ]
    passed = sum(bool(result["target"]["pass"]) for result in results)
    return {
        "schema_version": TACTICAL_REPORT_SCHEMA_VERSION,
        "action_layout_version": ACTION_LAYOUT_VERSION,
        "device": device,
        "summary": {
            "case_count": len(cases),
            "checkpoint_count": len(checkpoints),
            "evaluation_count": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": 1.0 if not results else passed / len(results),
        },
        "results": results,
    }
