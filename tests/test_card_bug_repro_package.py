# -*- coding: utf-8 -*-
"""Checklist 1.13 contracts for portable card-bug reproductions."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.card_bug_repro_package import (
    execute_synthetic_fixture,
    minimize_action_sequence,
    replay_package,
    validate_package,
)
from swb.engine.environment import ShadowverseEnv


PACKAGE_PATH = Path(
    "data/reports/card_bug_audit/repros/SWB-CARD-0008.json"
)
SYNTHETIC_PATH = Path(
    "data/reports/card_bug_audit/repros/SWB-CARD-0008-synthetic.json"
)
LEDGER_PATH = Path("data/reports/card_bug_audit/bug_ledger.json")
CLOSURE_PATH = Path(
    "data/reports/card_bug_audit/stage_1_13_repro_closure.json"
)
CHECKPOINT_IMPACT_PATH = Path(
    "data/reports/card_bug_audit/repros/checkpoint_impact.json"
)
DATABASE_PATH = Path("data/cards.sqlite3")


class CardBugReproPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.package = json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))
        cls.synthetic = json.loads(
            SYNTHETIC_PATH.read_text(encoding="utf-8")
        )

    def test_package_is_json_native_and_process_independent(self) -> None:
        validate_package(self.package)
        portability = self.package["portability"]
        self.assertTrue(portability["only_json_native_values"])
        self.assertFalse(portability["requires_ui"])
        self.assertFalse(portability["requires_original_process"])
        self.assertEqual(portability["encoding"], "UTF-8 JSON")
        self.assertEqual(
            json.loads(json.dumps(self.package, ensure_ascii=False)),
            self.package,
        )

    def test_portable_package_replays(self) -> None:
        replay = replay_package(self.package, database=DATABASE_PATH)

        self.assertEqual(replay["illegal_action_indices"], [])
        self.assertEqual(
            replay["pre_command_snapshot_sha256"],
            self.package["pre_command"]["snapshot_sha256"],
        )
        self.assertEqual(
            replay["command"],
            {
                "action": 107,
                "command_type": "SuperEvolve",
                "parameters": {
                    "player_index": 0,
                    "type": "super_evolve",
                    "unit_id": 69,
                },
            },
        )
        target = replay["target_after"]
        self.assertEqual(target["card_id"], 10154120)
        self.assertEqual(
            (target["attack"], target["health"], target["max_health"]),
            (8, 4, 4),
        )
        self.assertTrue(target["super_evolved"])
        self.assertEqual(
            replay["events"],
            self.package["transition"]["events"],
        )

    def test_natural_trace_is_reduced_and_mask_matches_legal_actions(self) -> None:
        minimization = self.package["minimization"]
        self.assertEqual(minimization["original_action_count"], 107)
        self.assertEqual(minimization["minimized_action_count"], 86)
        self.assertTrue(minimization["natural_reduction_found"])
        self.assertFalse(minimization["fallback_required"])
        self.assertGreater(minimization["attempts"], 0)
        self.assertEqual(self.package["action_trace"]["action_count"], 86)

        mask = self.package["pre_command"]["action_mask"]
        self.assertEqual(len(mask), ShadowverseEnv.ACTION_SIZE)
        self.assertEqual(
            len(mask),
            self.package["pre_command"]["action_mask_size"],
        )
        legal_ids = [
            row["action"]
            for row in self.package["pre_command"]["legal_actions"]
        ]
        self.assertEqual(
            legal_ids,
            [index for index, legal in enumerate(mask) if legal],
        )
        self.assertTrue(mask[107])

    def test_delta_debugger_prioritizes_and_removes_irrelevant_actions(self) -> None:
        result = minimize_action_sequence(
            [0, 1, 2, 3, 4, 5],
            lambda candidate: 1 in candidate and 4 in candidate,
        )

        self.assertEqual(result.actions, [1, 4])
        self.assertTrue(result.natural_reduction_found)
        self.assertEqual(result.minimized_action_count, 2)
        self.assertGreater(result.attempts, 0)

    def test_synthetic_fixture_executes(self) -> None:
        self.assertEqual(
            execute_synthetic_fixture(self.synthetic),
            self.synthetic["fixture"]["expected"],
        )
        self.assertEqual(
            (
                self.synthetic["fixture"]["expected"]["attack"],
                self.synthetic["fixture"]["expected"]["health"],
                self.synthetic["fixture"]["expected"]["max_health"],
            ),
            (8, 4, 4),
        )

    def test_every_fixed_bug_has_saved_reproduction_and_regression(self) -> None:
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        fixed = [
            entry
            for entry in ledger["entries"]
            if entry["status"] == "fixed"
        ]

        self.assertEqual(len(fixed), ledger["summary"]["total"])
        for entry in fixed:
            with self.subTest(bug_id=entry["bug_id"]):
                self.assertTrue(
                    Path(entry["reproduction_file"]).is_file(),
                    entry["reproduction_file"],
                )
                self.assertTrue(entry["regression_tests"])
                self.assertTrue(entry["actual"])
                self.assertTrue(entry["expected"])
                self.assertTrue(entry["fix_commit"])

    def test_same_mechanic_real_card_regressions_are_recorded(self) -> None:
        tests = self.package["regression"]["same_mechanic_real_card_tests"]

        self.assertEqual(len(tests), 5)
        self.assertTrue(any("ordinary_evolution" in name for name in tests))
        self.assertTrue(any("super_evolution" in name for name in tests))
        self.assertTrue(any("himeka" in name for name in tests))
        self.assertTrue(any("holy_knight" in name for name in tests))
        self.assertTrue(any("pascales_dance" in name for name in tests))
        self.assertEqual(
            self.package["regression"]["card_id_special_cases_added"],
            0,
        )

    def test_closure_report_hashes_all_portable_artifacts(self) -> None:
        closure = json.loads(CLOSURE_PATH.read_text(encoding="utf-8"))

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        self.assertEqual(closure["status"], "passed")
        self.assertEqual(
            closure["portable_reproduction"]["package_sha256"],
            digest(PACKAGE_PATH),
        )
        self.assertEqual(
            closure["minimization"]["additional_synthetic_fixture"][
                "sha256"
            ],
            digest(SYNTHETIC_PATH),
        )
        self.assertEqual(
            closure["checkpoint_impact"]["report_sha256"],
            digest(CHECKPOINT_IMPACT_PATH),
        )
        self.assertEqual(
            closure["minimization"]["minimized_action_count"],
            self.package["minimization"]["minimized_action_count"],
        )


if __name__ == "__main__":
    unittest.main()
