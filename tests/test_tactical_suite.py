from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_tactical_scenario import extract_case
from swb.rl.tactical_suite import (
    TacticalCaseError,
    load_tactical_case,
    resolve_action_selector,
    validate_tactical_case,
)
from swb.rl.versioning import stable_json_sha256


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = (
    ROOT
    / "data"
    / "tactical_scenarios"
    / "TACT-SE-0001-empty-board-storm.json"
)
SOURCE_HISTORY = (
    ROOT
    / "data"
    / "match_history"
    / "20260802T081708672593Z-0ca7034d.json"
)


class TacticalSuiteTests(unittest.TestCase):
    def test_semantic_selector_uses_card_identity_not_entity_identity(self) -> None:
        legal_actions = [
            {
                "id": 106,
                "kind": "super_evolve",
                "source_entity_id": 991,
                "label": "preferred",
            },
            {
                "id": 107,
                "kind": "super_evolve",
                "source_entity_id": 992,
                "label": "other",
            },
        ]
        own_board = [
            {"entity_id": 991, "card_id": 10461110, "name": "托路"},
            {"entity_id": 992, "card_id": 10404110, "name": "圣德芬"},
        ]

        resolved = resolve_action_selector(
            {
                "kind": "super_evolve",
                "source_card_id": 10461110,
                "source_occurrence": 0,
            },
            legal_actions=legal_actions,
            own_board=own_board,
        )

        self.assertEqual(resolved["id"], 106)

    def test_semantic_selector_rejects_ambiguous_action(self) -> None:
        with self.assertRaisesRegex(TacticalCaseError, "resolved to 2"):
            resolve_action_selector(
                {"kind": "end_turn"},
                legal_actions=[
                    {"id": 0, "kind": "end_turn"},
                    {"id": 1, "kind": "end_turn"},
                ],
                own_board=[],
            )

    def test_prefix_hash_detects_action_mutation(self) -> None:
        case = {
            "schema_version": 1,
            "case_id": "test",
            "title": "test",
            "category": "test",
            "objective": "test",
            "setup": {
                "seed": 1,
                "human_player": 0,
                "evaluated_player": 1,
                "human_deck": "a",
                "ai_deck": "b",
                "match_setup": "official",
            },
            "prefix": {
                "actions": [
                    {
                        "sequence": 1,
                        "player_index": 0,
                        "actor_role": "human",
                        "action_id": 0,
                        "kind": "end_turn",
                    }
                ],
                "expected_state_sha256": "0" * 64,
            },
            "decision": {
                "target_sequence": 2,
                "player_index": 1,
                "grading": "pairwise_preference",
                "preferred": [{"kind": "end_turn"}],
                "disfavored": [{"kind": "extra_pp"}],
            },
        }
        case["prefix"]["action_trace_sha256"] = stable_json_sha256(
            case["prefix"]["actions"]
        )
        validate_tactical_case(case)
        case["prefix"]["actions"][0]["action_id"] = 1

        with self.assertRaisesRegex(TacticalCaseError, "trace hash mismatch"):
            validate_tactical_case(case)

    @unittest.skipUnless(CASE_PATH.is_file(), "tactical case has not been extracted")
    def test_first_case_is_self_validating(self) -> None:
        case = load_tactical_case(CASE_PATH)

        self.assertEqual(case["case_id"], "TACT-SE-0001")
        self.assertEqual(case["decision"]["target_sequence"], 73)
        self.assertEqual(len(case["prefix"]["actions"]), 72)

    @unittest.skipUnless(
        SOURCE_HISTORY.is_file(),
        "source match history is unavailable",
    )
    def test_extractor_uses_semantic_board_selectors(self) -> None:
        history = json.loads(SOURCE_HISTORY.read_text(encoding="utf-8"))
        case = extract_case(
            history,
            history_path=SOURCE_HISTORY,
            target_sequence=73,
            case_id="TACT-SE-0001",
            title="empty board",
            category="super_evolution_target",
            objective="prefer_immediate_leader_pressure",
            preferred_action_id=106,
            disfavored_action_id=107,
            rationale="test",
        )

        self.assertEqual(
            case["decision"]["preferred"][0]["source_card_id"],
            10461110,
        )
        self.assertEqual(
            case["decision"]["disfavored"][0]["source_card_id"],
            10404110,
        )
        self.assertNotIn("source_entity_id", case["decision"]["preferred"][0])
        self.assertEqual(case["reference_policy"]["selected_action_id"], 107)

    def test_load_rejects_non_object_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(TacticalCaseError, "JSON object"):
                load_tactical_case(path)


if __name__ == "__main__":
    unittest.main()
