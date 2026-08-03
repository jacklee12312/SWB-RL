from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from swb.rl.tactical_suite import (
    TACTICAL_CASE_SCHEMA_VERSION,
    TacticalCaseError,
    file_sha256,
    validate_tactical_case,
)
from swb.rl.versioning import ACTION_LAYOUT_VERSION, stable_json_sha256


def _source_card(
    before: Mapping[str, Any],
    *,
    player_index: int,
    entity_id: int,
) -> Mapping[str, Any]:
    player = before["players"][player_index]
    for card in player["board"]:
        if int(card["entity_id"]) == entity_id:
            return card
    raise TacticalCaseError(f"board entity {entity_id} is absent at the target")


def _selector_from_candidate(
    candidate: Mapping[str, Any],
    *,
    before: Mapping[str, Any],
    player_index: int,
) -> dict[str, Any]:
    selector: dict[str, Any] = {"kind": str(candidate["kind"])}
    if candidate.get("source_entity_id") is not None:
        source = _source_card(
            before,
            player_index=player_index,
            entity_id=int(candidate["source_entity_id"]),
        )
        same_card_sources = [
            card
            for card in before["players"][player_index]["board"]
            if int(card["card_id"]) == int(source["card_id"])
        ]
        occurrence = next(
            index
            for index, card in enumerate(same_card_sources)
            if int(card["entity_id"]) == int(source["entity_id"])
        )
        selector.update(
            {
                "source_card_id": int(source["card_id"]),
                "source_name": str(source["name"]),
                "source_occurrence": occurrence,
            }
        )
    if "mode_id" in candidate:
        selector["mode_id"] = candidate["mode_id"]
    if "option_id" in candidate:
        selector["option_id"] = candidate["option_id"]
    if candidate.get("kind") == "attack":
        selector["target"] = (
            "leader" if candidate.get("target_entity_id") is None else "board"
        )
    return selector


def extract_case(
    history: Mapping[str, Any],
    *,
    history_path: Path,
    target_sequence: int,
    case_id: str,
    title: str,
    category: str,
    objective: str,
    preferred_action_id: int,
    disfavored_action_id: int,
    rationale: str,
) -> dict[str, Any]:
    if history.get("schema_version") != 2:
        raise TacticalCaseError("source history must use schema version 2")
    target = next(
        (
            action
            for action in history.get("actions", [])
            if int(action["sequence"]) == target_sequence
        ),
        None,
    )
    if target is None:
        raise TacticalCaseError(f"history has no action sequence {target_sequence}")
    if target.get("actor_role") != "ai":
        raise TacticalCaseError("the first tactical extractor requires an AI target")
    decision = target.get("decision")
    if not isinstance(decision, Mapping):
        raise TacticalCaseError("target action has no policy decision trace")
    candidates = {
        int(candidate["id"]): candidate
        for candidate in decision.get("legal_actions", [])
    }
    missing = [
        action_id
        for action_id in (preferred_action_id, disfavored_action_id)
        if action_id not in candidates
    ]
    if missing:
        raise TacticalCaseError(f"target legal actions do not contain {missing}")
    prefix_actions = [
        {
            "sequence": int(action["sequence"]),
            "player_index": int(action["player_index"]),
            "actor_role": str(action["actor_role"]),
            "action_id": int(action["action_id"]),
            "kind": str(action["action"]["kind"]),
        }
        for action in history["actions"]
        if int(action["sequence"]) < target_sequence
    ]
    before = target["before"]
    evaluated_player = int(target["player_index"])
    human_player = int(history["human_player"])
    setup_deck = history["deck"]
    case = {
        "schema_version": TACTICAL_CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "title": title,
        "category": category,
        "objective": objective,
        "provenance": {
            "source_history": history_path.as_posix(),
            "source_history_sha256": file_sha256(history_path),
            "match_id": history["match_id"],
            "source_checkpoint": history["checkpoint"],
        },
        "setup": {
            "seed": int(history["seed"]),
            "human_player": human_player,
            "evaluated_player": evaluated_player,
            "human_deck": setup_deck["human"]["name"],
            "human_deck_sha256": setup_deck["human"]["sha256"],
            "ai_deck": setup_deck["ai"]["name"],
            "ai_deck_sha256": setup_deck["ai"]["sha256"],
            "match_setup": "official",
            "action_layout_version": ACTION_LAYOUT_VERSION,
        },
        "prefix": {
            "actions": prefix_actions,
            "action_trace_sha256": stable_json_sha256(prefix_actions),
            "expected_state_sha256": stable_json_sha256(before),
        },
        "decision": {
            "target_sequence": target_sequence,
            "player_index": evaluated_player,
            "grading": "pairwise_preference",
            "preferred": [
                _selector_from_candidate(
                    candidates[preferred_action_id],
                    before=before,
                    player_index=evaluated_player,
                )
            ],
            "disfavored": [
                _selector_from_candidate(
                    candidates[disfavored_action_id],
                    before=before,
                    player_index=evaluated_player,
                )
            ],
            "confidence": "high",
            "rationale": rationale,
        },
        "state_summary": {
            "turn": int(before["turn"]),
            "evaluated_leader_health": int(
                before["players"][evaluated_player]["health"]
            ),
            "opponent_leader_health": int(
                before["players"][1 - evaluated_player]["health"]
            ),
            "evaluated_super_evolution_points": int(
                before["players"][evaluated_player]["super_evolution_points"]
            ),
            "opponent_board_count": len(before["players"][1 - evaluated_player]["board"]),
            "evaluated_board": before["players"][evaluated_player]["board"],
        },
        "reference_policy": {
            "selected_action_id": int(decision["selected_action_id"]),
            "selected_probability": float(decision["selected_probability"]),
            "value": float(decision["value"]),
            "preferred": {
                "action_id": preferred_action_id,
                "logit": float(candidates[preferred_action_id]["logit"]),
                "probability": float(candidates[preferred_action_id]["probability"]),
            },
            "disfavored": {
                "action_id": disfavored_action_id,
                "logit": float(candidates[disfavored_action_id]["logit"]),
                "probability": float(candidates[disfavored_action_id]["probability"]),
            },
        },
    }
    validate_tactical_case(case)
    return case


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a portable tactical policy case from schema-v2 history"
    )
    parser.add_argument("history", type=Path)
    parser.add_argument("--target-sequence", type=int, required=True)
    parser.add_argument("--preferred-action-id", type=int, required=True)
    parser.add_argument("--disfavored-action-id", type=int, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    history = json.loads(args.history.read_text(encoding="utf-8"))
    case = extract_case(
        history,
        history_path=args.history,
        target_sequence=args.target_sequence,
        case_id=args.case_id,
        title=args.title,
        category=args.category,
        objective=args.objective,
        preferred_action_id=args.preferred_action_id,
        disfavored_action_id=args.disfavored_action_id,
        rationale=args.rationale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(case, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(case, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
