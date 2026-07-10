# -*- coding: utf-8 -*-
"""Tests for rule coverage report script."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import RuleBook


class CoverageReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = os.path.dirname(os.path.dirname(__file__))
        cls.db_path = os.path.join(cls.project_root, "data", "cards.sqlite3")
        cls.rules_dir = os.path.join(cls.project_root, "data", "rules")
        cls.has_real_db = os.path.exists(cls.db_path)

    def setUp(self):
        if not self.has_real_db:
            self.skipTest("cards.sqlite3 not found")

    def test_script_runs_and_outputs_json(self):
        """Script produces valid JSON output."""
        from scripts.report_rule_coverage import _build_coverage_report
        report = _build_coverage_report(self.db_path, self.rules_dir)
        self.assertIsInstance(report, dict)
        self.assertIn("summary", report)
        self.assertIn("classifications", report)
        self.assertIn("top_20_recommendations", report)

    def test_script_can_run_by_file_path(self):
        """Direct script execution works without manual PYTHONPATH setup."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            output_path = f.name
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    os.path.join(self.project_root, "scripts", "report_rule_coverage.py"),
                    "--db",
                    self.db_path,
                    "--rules",
                    self.rules_dir,
                    "--output",
                    output_path,
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            with open(output_path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertIn("summary", loaded)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_json_output_is_stable_sort(self):
        """Same input produces identical output (stable sort)."""
        from scripts.report_rule_coverage import _build_coverage_report
        report1 = _build_coverage_report(self.db_path, self.rules_dir)
        report2 = _build_coverage_report(self.db_path, self.rules_dir)
        keys1 = list(report1["classifications"].keys())
        keys2 = list(report2["classifications"].keys())
        self.assertEqual(keys1, keys2)

    def test_test_ids_recognized(self):
        """Synthetic card IDs in rules (999xxx) are detected in issues."""
        from scripts.report_rule_coverage import _build_coverage_report
        report = _build_coverage_report(self.db_path, self.rules_dir)
        issues = report.get("rule_consistency_issues", [])
        unknown_ids = [i["card_id"] for i in issues]
        found = any(999000 <= cid <= 999999 for cid in unknown_ids)
        self.assertTrue(found, f"No synthetic IDs found in rule issues. Issues: {issues[:3]}")

    def test_real_card_ids_exist(self):
        """Cards in rules that exist in DB are properly classified."""
        from scripts.report_rule_coverage import _build_coverage_report
        report = _build_coverage_report(self.db_path, self.rules_dir)
        known_real = [10001110, 10041130, 10214110, 10671110]
        for cid in known_real:
            self.assertIn(str(cid), report["classifications"], f"Card {cid} not in report")

    def test_missing_database_error(self):
        """Script errors on missing database."""
        from scripts.report_rule_coverage import _build_coverage_report
        with self.assertRaises(FileNotFoundError):
            _build_coverage_report("nonexistent.db", self.rules_dir)

    def test_rule_cards_not_in_db_reported(self):
        """Cards with rules but not in DB appear in issues."""
        from scripts.report_rule_coverage import _build_coverage_report
        report = _build_coverage_report(self.db_path, self.rules_dir)
        issues = report.get("rule_consistency_issues", [])
        for issue in issues:
            self.assertIn("card_id", issue)
            self.assertIn("issue", issue)

    def test_existing_rules_counted(self):
        """Cards with rules appear in summary count."""
        from scripts.report_rule_coverage import _build_coverage_report
        report = _build_coverage_report(self.db_path, self.rules_dir)
        self.assertGreater(report["summary"]["total_with_rules"], 0)

    def test_primitive_keyword_map_covers_10_plus(self):
        """Primitive keyword map has at least 10 entries."""
        from scripts.report_rule_coverage import PRIMITIVE_KEYWORD_MAP
        self.assertGreaterEqual(len(PRIMITIVE_KEYWORD_MAP), 10)

    def test_regex_keyword_map_detects_alternatives(self):
        """Regex-style primitive patterns detect either alternative keyword."""
        from scripts.report_rule_coverage import _classify_card
        card = CardDefinition(
            card_id=123456,
            card_set_id=10000,
            class_id=1,
            class_name="梦魇",
            name="测试唤灵",
            cost=2,
            card_type="随从",
            attack=2,
            life=2,
            keywords=frozenset(),
            support_level="unsupported",
            is_collectible=True,
        )
        result = _classify_card(
            card,
            ruled_cards=set(),
            ruled_ops={},
            rule_metadata={},
            ability_map={123456: ["唤灵"]},
            skill_text_map={},
            support_map={},
        )
        self.assertEqual(result["coverage"], "supported_missing_rule")
        self.assertIn("死灵术|唤灵", result["hit_keywords"])

    def test_placeholder_keyword_with_rule_is_not_covered_exact(self):
        """A real rule does not hide missing primitives such as activate."""
        from scripts.report_rule_coverage import _classify_card
        card = CardDefinition(
            card_id=123457,
            card_set_id=10000,
            class_id=1,
            class_name="精灵",
            name="测试策动",
            cost=2,
            card_type="随从",
            attack=2,
            life=2,
            keywords=frozenset(),
            support_level="unsupported",
            is_collectible=True,
        )
        result = _classify_card(
            card,
            ruled_cards={123457},
            ruled_ops={123457: {"triggers": ["attack"], "effect_kinds": ["heal_leader"]}},
            rule_metadata={},
            ability_map={123457: ["策动"]},
            skill_text_map={},
            support_map={},
        )
        self.assertEqual(result["coverage"], "covered_partial")
        self.assertIn("策动", result["missing_primitives"])

    def test_top_20_recommendations_fields(self):
        """Top 20 recommendations have complete fields."""
        from scripts.report_rule_coverage import _build_coverage_report
        report = _build_coverage_report(self.db_path, self.rules_dir)
        recs = report["top_20_recommendations"]
        self.assertLessEqual(len(recs), 20)
        for rec in recs:
            for field in ("card_id", "name", "class_name", "card_type", "cost", "confidence", "why_recommended"):
                self.assertIn(field, rec, f"Missing field {field} in recommendation")

    def test_markdown_output_contains_summary_table(self):
        """Markdown output contains the summary table."""
        import tempfile
        from scripts.report_rule_coverage import write_markdown_report, _build_coverage_report
        report = _build_coverage_report(self.db_path, self.rules_dir)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            write_markdown_report(report, f.name)
            md_path = f.name
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        os.unlink(md_path)
        self.assertIn("## Summary", content)
        self.assertIn("|", content)

    def test_rulebook_loads_all_rules(self):
        """RuleBook.from_directory loads without error."""
        rb = RuleBook.from_directory(self.rules_dir)
        self.assertIsInstance(rb, RuleBook)

    def test_ability_status_reports_handler_and_primitive_columns(self):
        """Ability status distinguishes handler state from primitive support."""
        result = subprocess.run(
            [sys.executable, os.path.join(self.project_root, "scripts", "ability_status.py")],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("Handler", result.stdout)
        self.assertIn("Primitive", result.stdout)
        self.assertRegex(result.stdout, r"爆能强化\s+placeholder\s+covered")
        self.assertRegex(result.stdout, r"融合\s+partial\s+covered")

    def test_json_output_writes_file(self):
        """JSON output file is created."""
        from scripts.report_rule_coverage import write_json_report, _build_coverage_report
        report = _build_coverage_report(self.db_path, self.rules_dir)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            write_json_report(report, f.name)
            json_path = f.name
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        os.unlink(json_path)
        self.assertIn("summary", loaded)

    def test_real_cards_covered(self):
        """Real cards with rules have covered_exact or covered_partial."""
        from scripts.report_rule_coverage import _build_coverage_report
        report = _build_coverage_report(self.db_path, self.rules_dir)
        covered = [v for v in report["classifications"].values()
                   if v["coverage"] in ("covered_exact", "covered_partial")]
        self.assertGreater(len(covered), 0, "No real cards covered by rules")


if __name__ == "__main__":
    unittest.main()
