from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.report_card_bug_audit import (
    ENTRY_FIELDS,
    REQUIRED_CHECKLIST_FIELDS,
    SEVERITY_DEFINITIONS,
    build_bug_ledger,
    normalize_bug_entry,
    render_json,
    render_markdown,
)


JSON_REPORT = Path("data/reports/card_bug_audit/bug_ledger.json")
MARKDOWN_REPORT = Path("data/reports/card_bug_audit/bug_ledger.md")


def _entry(**overrides):
    entry = {
        "bug_id": "SWB-CARD-0001",
        "severity": "P1",
        "status": "open",
        "card": {"card_id": 10001110, "name": "测试卡牌"},
        "mechanic": "目标",
        "discovery_commit": "0123456789abcdef",
        "minimal_seed": 7,
        "reproduction_file": (
            "data/reports/card_bug_audit/repros/SWB-CARD-0001.json"
        ),
        "expected": "应选择一个合法目标。",
        "actual": "未产生待决选择。",
        "impact": "影响训练卡组中的目标决策。",
        "affected_decks": ["deck_b", "deck_a"],
        "fix_commit": None,
        "regression_tests": [],
        "notes": "",
    }
    entry.update(overrides)
    return entry


class CardBugAuditLedgerTests(unittest.TestCase):
    def test_severity_definitions_and_training_gates_are_frozen(self):
        definitions = {
            row["severity"]: row for row in SEVERITY_DEFINITIONS
        }
        self.assertEqual(tuple(definitions), ("P0", "P1", "P2", "P3"))
        self.assertTrue(definitions["P0"]["blocks_formal_training"])
        self.assertTrue(definitions["P0"]["blocks_long_training"])
        self.assertFalse(definitions["P1"]["blocks_formal_training"])
        self.assertTrue(definitions["P1"]["blocks_long_training"])
        self.assertFalse(definitions["P2"]["blocks_long_training"])
        self.assertFalse(definitions["P3"]["blocks_long_training"])
        self.assertIn("隐藏信息泄漏", definitions["P0"]["definition"])
        self.assertIn("职业资源", definitions["P1"]["definition"])
        self.assertIn("罕见组合", definitions["P2"]["definition"])
        self.assertIn("UI", definitions["P3"]["definition"])

    def test_ledger_contract_contains_every_checklist_field(self):
        self.assertTrue(set(REQUIRED_CHECKLIST_FIELDS).issubset(ENTRY_FIELDS))
        self.assertIn("status", ENTRY_FIELDS)
        self.assertIn("impact", ENTRY_FIELDS)
        self.assertIn("notes", ENTRY_FIELDS)

    def test_entries_are_normalized_and_sorted_deterministically(self):
        second = _entry(
            bug_id="SWB-CARD-0002",
            severity="P2",
            affected_decks=[],
        )
        first = _entry(affected_decks=["deck_b", "deck_a"])
        report = build_bug_ledger([second, first])
        self.assertEqual(
            [entry["bug_id"] for entry in report["entries"]],
            ["SWB-CARD-0001", "SWB-CARD-0002"],
        )
        self.assertEqual(
            report["entries"][0]["affected_decks"],
            ["deck_a", "deck_b"],
        )
        self.assertEqual(report["summary"]["open_training_blockers"]["P0"], 0)
        self.assertEqual(report["summary"]["open_training_blockers"]["P1"], 1)
        self.assertFalse(report["summary"]["ledger_p0_p1_clear"])
        self.assertEqual(
            render_json(report),
            render_json(build_bug_ledger([second, first])),
        )

    def test_fixed_entries_require_fix_commit_and_regression_test(self):
        with self.assertRaisesRegex(ValueError, "require fix_commit"):
            normalize_bug_entry(_entry(status="fixed"))
        with self.assertRaisesRegex(ValueError, "require regression_tests"):
            normalize_bug_entry(
                _entry(status="fixed", fix_commit="fedcba9876543210")
            )
        normalized = normalize_bug_entry(
            _entry(
                status="fixed",
                fix_commit="fedcba9876543210",
                regression_tests=["tests/test_targeting.py"],
            )
        )
        self.assertEqual(normalized["status"], "fixed")

    def test_closed_not_bug_requires_a_recorded_ruling(self):
        with self.assertRaisesRegex(ValueError, "recorded ruling"):
            normalize_bug_entry(_entry(status="closed_not_bug"))

    def test_paths_must_be_repository_relative(self):
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            normalize_bug_entry(_entry(reproduction_file="../outside.json"))

    def test_saved_reports_match_the_deterministic_current_ledger(self):
        saved = json.loads(JSON_REPORT.read_text(encoding="utf-8"))
        expected = build_bug_ledger(saved["entries"])
        self.assertEqual(saved, expected)
        self.assertEqual(JSON_REPORT.read_text(encoding="utf-8"), render_json(expected))
        self.assertEqual(
            MARKDOWN_REPORT.read_text(encoding="utf-8"),
            render_markdown(expected),
        )


if __name__ == "__main__":
    unittest.main()
