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
from contextlib import closing

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
        """Synthetic IDs are counted without becoming consistency issues."""
        from scripts.report_rule_coverage import _build_coverage_report
        report = _build_coverage_report(self.db_path, self.rules_dir)
        issues = report.get("rule_consistency_issues", [])
        unknown_ids = [i["card_id"] for i in issues]
        self.assertFalse(any(999000 <= cid <= 999999 for cid in unknown_ids))
        self.assertGreater(
            report["summary"]["test_or_synthetic_ids_with_rules"],
            0,
        )

    def test_clause_audit_has_no_unverified_exact_rules(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report(self.db_path, self.rules_dir)
        unverified = [
            info for info in report["classifications"].values()
            if info["clause_audit"]["status"] == "unverified_exact"
        ]
        self.assertEqual(unverified, [])
        self.assertEqual(
            report["summary"]["clause_audit_counts"].get("unverified_exact", 0),
            0,
        )
        self.assertEqual(report["clause_audit_issues"], [])
        self.assertFalse(any(
            issue["issue"] == "clause_audit_validation_failed"
            for issue in report["rule_consistency_issues"]
        ))

    def test_explicit_exact_rules_map_text_rules_and_tests(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report(self.db_path, self.rules_dir)
        mapped = [
            info for info in report["classifications"].values()
            if info["clause_audit"]["status"] == "mapped_exact"
        ]
        self.assertTrue(mapped)
        for info in mapped:
            with self.subTest(card_id=info["card_id"]):
                audit = info["clause_audit"]
                self.assertTrue(audit["implemented_text"])
                self.assertTrue(audit["test_evidence"])
                self.assertIn("triggers", audit["structured_evidence"])
                self.assertIn("effect_kinds", audit["structured_evidence"])
                self.assertEqual(audit["rule_version"], 1)
                self.assertEqual(len(audit["source_text_sha256"]), 64)
                self.assertIsNone(audit["audit_validation_error"])

    def test_source_snapshot_detects_database_refreshes(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report(self.db_path, self.rules_dir)
        snapshot = report["generated_from"]["source_snapshot"]
        self.assertEqual(snapshot["card_count"], 826)
        self.assertEqual(len(snapshot["sha256"]), 64)

    def test_blocker_taxonomy_is_explicit_and_complete(self):
        from scripts.report_rule_coverage import BLOCKER_TYPES, _build_coverage_report

        report = _build_coverage_report(self.db_path, self.rules_dir)
        self.assertEqual(
            tuple(report["summary"]["blocker_counts"]),
            BLOCKER_TYPES,
        )
        self.assertEqual(
            report["summary"]["blocker_counts"]["audit_unverified"],
            report["summary"]["clause_audit_counts"]["unverified_exact"],
        )
        self.assertEqual(
            report["summary"]["blocker_counts"]["missing_rule"],
            report["summary"]["coverage_counts"]["supported_missing_rule"],
        )

    def test_metadata_supports_rule_versions_errata_and_blockers(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.report_rule_coverage import _load_rule_metadata

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "versioned.json").write_text(
                json.dumps({
                    "rules": [{
                        "card_id": 123,
                        "trigger": "play",
                        "coverage": "partial",
                        "rule_version": 2,
                        "errata": ["2026-07 official wording update"],
                        "blocker_type": "timing_unclear",
                        "unsupported_text": "resolve order pending ruling",
                        "operations": [],
                    }]
                }),
                encoding="utf-8",
            )
            metadata = _load_rule_metadata(tmp)[123]

        self.assertEqual(metadata["rule_version"], 2)
        self.assertEqual(metadata["errata"], ["2026-07 official wording update"])
        self.assertEqual(metadata["blocker_type"], "timing_unclear")

    def test_clause_audit_registry_merges_hash_and_explicit_test_evidence(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.report_rule_coverage import (
            _load_rule_metadata,
            _source_text_sha256,
        )

        with tempfile.TemporaryDirectory() as tmp:
            rules = Path(tmp, "rules")
            audits = Path(tmp, "audits")
            rules.mkdir()
            audits.mkdir()
            Path(rules, "rule.json").write_text(
                json.dumps({"rules": [{
                    "card_id": 123,
                    "trigger": "play",
                    "operations": [],
                }]}),
                encoding="utf-8",
            )
            Path(audits, "rule_clauses.json").write_text(
                json.dumps({"cards": [{
                    "card_id": 123,
                    "coverage": "exact",
                    "implemented_text": "draw one",
                    "source_text_sha256": _source_text_sha256(["draw one"]),
                    "test_evidence": ["tests/test_draw.py"],
                }]}),
                encoding="utf-8",
            )
            metadata = _load_rule_metadata(str(rules))[123]

        self.assertEqual(metadata["coverage"], "exact")
        self.assertEqual(metadata["implemented_text"], "draw one")
        self.assertEqual(metadata["test_evidence"], ["tests/test_draw.py"])
        self.assertEqual(
            metadata["source_text_sha256"],
            _source_text_sha256(["draw one"]),
        )

    def test_source_hash_validation_detects_changed_text(self):
        from scripts.report_rule_coverage import (
            _source_text_sha256,
            _validate_rule_metadata_source_hashes,
        )

        metadata = {
            123: {"source_text_sha256": _source_text_sha256(["old text"])}
        }
        _validate_rule_metadata_source_hashes(metadata, {123: ["new text"]})

        self.assertIn("mismatch", metadata[123]["audit_validation_error"])

    def test_source_text_map_includes_deduplicated_alternate_modes(self):
        from scripts.report_rule_coverage import _load_source_text_map

        with closing(sqlite3.connect(self.db_path)) as conn:
            source_texts = _load_source_text_map(conn)

        self.assertIn(
            "【激奏_2】召唤1个『<color=Keyword>低劣的玩具</color>』。",
            source_texts[10671110],
        )
        emblem_clauses = [
            text for text in source_texts[10153140]
            if text.startswith("【纹章_0】")
        ]
        self.assertEqual(len(emblem_clauses), 1)

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

    def test_synthetic_missing_primitive_with_rule_is_not_covered_exact(self):
        """A real rule does not hide a still-missing primitive."""
        from unittest.mock import patch

        from scripts.report_rule_coverage import (
            PRIMITIVE_KEYWORD_MAP,
            _classify_card,
        )
        card = CardDefinition(
            card_id=123457,
            card_set_id=10000,
            class_id=1,
            class_name="精灵",
            name="测试灵气",
            cost=2,
            card_type="随从",
            attack=2,
            life=2,
            keywords=frozenset(),
            support_level="unsupported",
            is_collectible=True,
        )
        with patch.dict(
            PRIMITIVE_KEYWORD_MAP,
            {
                "灵气": {
                    "primitive": "synthetic unavailable primitive",
                    "covered": False,
                }
            },
        ):
            result = _classify_card(
                card,
                ruled_cards={123457},
                ruled_ops={123457: {
                    "triggers": ["attack"],
                    "effect_kinds": ["heal_leader"],
                }},
                rule_metadata={},
                ability_map={123457: ["灵气"]},
                skill_text_map={},
                support_map={},
            )
        self.assertEqual(result["coverage"], "covered_partial")
        self.assertIn("灵气", result["missing_primitives"])

    def test_activate_primitive_still_requires_a_per_card_definition(self):
        """Generic Activate support must not make unrelated partial rules exact."""
        from scripts.report_rule_coverage import _classify_card
        card = CardDefinition(
            card_id=123458,
            card_set_id=10000,
            class_id=1,
            class_name="精灵",
            name="测试策动护符",
            cost=2,
            card_type="护符",
            attack=None,
            life=None,
            keywords=frozenset(),
            support_level="unsupported",
            is_collectible=True,
        )
        kwargs = {
            "card": card,
            "ruled_cards": {123458},
            "ruled_ops": {
                123458: {
                    "triggers": ["play"],
                    "effect_kinds": ["draw"],
                }
            },
            "rule_metadata": {},
            "ability_map": {123458: ["策动"]},
            "skill_text_map": {},
            "support_map": {},
        }

        partial = _classify_card(**kwargs, activation_cards=set())
        exact = _classify_card(**kwargs, activation_cards={123458})

        self.assertEqual(partial["coverage"], "covered_partial")
        self.assertEqual(partial["missing_rule_mechanics"], ["策动"])
        self.assertNotIn("策动", partial["missing_primitives"])
        self.assertEqual(exact["coverage"], "covered_exact")

    def test_faith_primitive_still_requires_a_per_card_definition(self):
        """Faith state support must not hide a missing card-specific Faith."""
        from scripts.report_rule_coverage import _classify_card
        card = CardDefinition(
            card_id=123459,
            card_set_id=10000,
            class_id=1,
            class_name="精灵",
            name="测试信仰",
            cost=2,
            card_type="随从",
            attack=2,
            life=2,
            keywords=frozenset(),
            support_level="unsupported",
            is_collectible=True,
        )
        kwargs = {
            "card": card,
            "ruled_cards": {123459},
            "ruled_ops": {
                123459: {
                    "triggers": ["play"],
                    "effect_kinds": ["draw"],
                }
            },
            "rule_metadata": {},
            "ability_map": {123459: ["信仰"]},
            "skill_text_map": {},
            "support_map": {},
        }

        partial = _classify_card(**kwargs, faith_cards=set())
        exact = _classify_card(**kwargs, faith_cards={123459})

        self.assertEqual(partial["coverage"], "covered_partial")
        self.assertEqual(partial["missing_rule_mechanics"], ["信仰"])
        self.assertNotIn("信仰", partial["missing_primitives"])
        self.assertEqual(exact["coverage"], "covered_exact")

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
        self.assertRegex(result.stdout, r"瞬念召唤\s+implemented\s+covered")

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
