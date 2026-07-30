# -*- coding: utf-8 -*-
"""Checklist 1.14 eight-training-deck phase-gate contracts."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.report_training_deck_gate import (
    DEFAULT_MARKDOWN,
    DEFAULT_OUTPUT,
    build_report,
    render_json,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]


class TrainingDeckGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_saved_reports_match_deterministic_generation(self) -> None:
        self.assertEqual(
            (ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"),
            render_json(self.report),
        )
        self.assertEqual(
            (ROOT / DEFAULT_MARKDOWN).read_text(encoding="utf-8"),
            render_markdown(self.report),
        )

    def test_all_nine_checklist_gates_pass(self) -> None:
        summary = self.report["summary"]

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["gate_count"], 9)
        self.assertEqual(summary["passed_gate_count"], 9)
        self.assertEqual(summary["failed_gate_count"], 0)
        self.assertEqual(summary["failed_gates"], [])

    def test_closure_has_one_complete_row_per_card(self) -> None:
        scope = self.report["scope"]
        rows = self.report["cards"]

        self.assertEqual(scope["fixed_deck_count"], 8)
        self.assertEqual(scope["fixed_deck_collectible_union_count"], 111)
        self.assertEqual(scope["recursive_reference_count"], 36)
        self.assertEqual(scope["closure_card_count"], 147)
        self.assertEqual(scope["closure_collectible_count"], 116)
        self.assertEqual(scope["closure_non_collectible_count"], 31)
        self.assertEqual(len(rows), 147)
        self.assertEqual(len({row["card_id"] for row in rows}), 147)
        self.assertTrue(all(row["status"] == "passed" for row in rows))
        self.assertTrue(all(row["direct_tests"] for row in rows))
        self.assertTrue(
            all(row["applicable_forced_scenarios"] for row in rows)
        )

    def test_runtime_untriggered_clauses_remain_honestly_labeled(self) -> None:
        gate = next(
            gate
            for gate in self.report["gates"]
            if gate["gate_id"] == "1.14.5"
        )

        self.assertEqual(gate["status"], "passed")
        self.assertEqual(
            gate["metrics"]["closure_runtime_clause_count"],
            458,
        )
        self.assertEqual(gate["metrics"]["runtime_triggered_passed"], 15)
        self.assertEqual(
            gate["metrics"]["runtime_explained_by_direct_test"],
            443,
        )
        self.assertEqual(
            gate["metrics"]["raw_runtime_status_counts"],
            {
                "not_triggered": 440,
                "triggered_not_executed": 3,
                "triggered_passed": 15,
            },
        )
        self.assertIn(
            "not_triggered is never relabeled",
            gate["note"],
        )

    def test_matrix_and_bug_blockers_are_zero(self) -> None:
        by_id = {
            gate["gate_id"]: gate for gate in self.report["gates"]
        }

        self.assertEqual(
            by_id["1.14.6"]["metrics"]["fixed_bug_count"],
            8,
        )
        self.assertEqual(self.report["summary"]["open_p0"], 0)
        self.assertEqual(self.report["summary"]["open_p1"], 0)
        self.assertEqual(
            by_id["1.14.7"]["metrics"]["matrix_mask_checks"],
            95230,
        )
        self.assertEqual(
            by_id["1.14.8"]["metrics"]["completed_games"],
            1024,
        )
        self.assertEqual(
            by_id["1.14.8"]["metrics"]["replay_checks"],
            1024,
        )

    def test_full_pool_gate_remains_explicitly_separate(self) -> None:
        self.assertTrue(
            any(
                "does not replace" in limitation
                and "full-pool gate" in limitation
                for limitation in self.report["limitations"]
            )
        )


if __name__ == "__main__":
    unittest.main()
