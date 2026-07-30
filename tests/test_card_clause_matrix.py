from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.report_card_clause_matrix import (
    AUDIT_DIMENSIONS,
    DEFAULT_BASELINE,
    DEFAULT_CLOSURE,
    DEFAULT_COVERAGE,
    DEFAULT_DATABASE,
    DEFAULT_RULES,
    DEFAULT_TOKEN_AUDIT,
    build_card_clause_matrix,
    render_json,
    render_markdown,
    validate_matrix_shape,
    validate_source_hash,
    validate_test_references,
)
from scripts.report_rule_coverage import _source_text_sha256


ROOT = Path(__file__).resolve().parents[1]
JSON_REPORT = ROOT / "data/reports/card_bug_audit/card_clause_matrix.json"
MARKDOWN_REPORT = ROOT / "data/reports/card_bug_audit/card_clause_matrix.md"


def _build():
    return build_card_clause_matrix(
        root=ROOT,
        database=ROOT / DEFAULT_DATABASE,
        rules_directory=ROOT / DEFAULT_RULES,
        coverage_report=ROOT / DEFAULT_COVERAGE,
        token_audit=ROOT / DEFAULT_TOKEN_AUDIT,
        baseline_report=ROOT / DEFAULT_BASELINE,
        closure_report=ROOT / DEFAULT_CLOSURE,
    )


class CardClauseMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _build()

    def test_matrix_has_one_stable_card_row_per_training_closure_card(self):
        summary = self.report["summary"]
        self.assertEqual(summary["card_count"], 147)
        self.assertEqual(summary["collectible_count"], 116)
        self.assertEqual(summary["non_collectible_count"], 31)
        self.assertEqual(
            [card["card_id"] for card in self.report["cards"]],
            sorted(card["card_id"] for card in self.report["cards"]),
        )
        self.assertEqual(
            len({card["audit_id"] for card in self.report["cards"]}),
            147,
        )

    def test_main_and_alternate_mode_texts_are_independent_clause_rows(self):
        summary = self.report["summary"]
        self.assertEqual(summary["clause_count"], 161)
        self.assertEqual(summary["clause_kinds"]["main_skill"], 143)
        self.assertEqual(summary["clause_kinds"]["alternate_mode"], 18)
        self.assertEqual(summary["source_hash_registry_validated"], 145)
        self.assertEqual(summary["source_hash_frozen_by_matrix"], 1)
        self.assertEqual(summary["source_hash_not_applicable"], 1)
        clause_ids = [
            clause["clause_id"]
            for card in self.report["cards"]
            for clause in card["clauses"]
        ]
        self.assertEqual(len(clause_ids), len(set(clause_ids)))

    def test_every_clause_has_mapping_evidence_and_verification_fields(self):
        for card in self.report["cards"]:
            for clause in card["clauses"]:
                with self.subTest(clause_id=clause["clause_id"]):
                    mapping = clause["structured_mapping"]
                    self.assertIn("triggers", mapping)
                    self.assertIn("conditions", mapping)
                    self.assertIn("targets", mapping)
                    self.assertIn("operations", mapping)
                    self.assertTrue(clause["direct_tests"])
                    self.assertEqual(clause["mechanic_tests"], [])
                    self.assertIn("url", clause["official_evidence"])
                    self.assertEqual(
                        clause["last_verified_commit"],
                        self.report["generated_from"]["last_verified_commit"],
                    )

    def test_dimensions_distinguish_passed_not_tested_and_not_applicable(self):
        validate_matrix_shape(self.report)
        for card in self.report["cards"]:
            self.assertEqual(tuple(card["dimensions"]), AUDIT_DIMENSIONS)
            self.assertEqual(
                card["dimensions"]["source_mapping"]["status"],
                "passed",
            )
        status_counts = self.report["summary"]["dimension_status_counts"]
        self.assertGreater(
            status_counts["alternate_modes"]["not_applicable"],
            0,
        )
        self.assertGreater(status_counts["alternate_modes"]["not_tested"], 0)
        self.assertGreater(
            self.report["summary"]["open_runtime_audit_rows"],
            0,
        )
        self.assertFalse(self.report["summary"]["training_runtime_gate_ready"])

    def test_broken_test_references_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "tests").mkdir()
            (root / "tests/present.py").write_text("", encoding="utf-8")
            self.assertEqual(
                validate_test_references(["tests/present.py"], root),
                ["tests/present.py"],
            )
            with self.assertRaisesRegex(ValueError, "broken test evidence"):
                validate_test_references(["tests/missing.py"], root)

    def test_stale_source_hash_is_rejected(self):
        expected = _source_text_sha256(["原始文字"])
        self.assertEqual(
            validate_source_hash(1, ["原始文字"], expected),
            expected,
        )
        with self.assertRaisesRegex(ValueError, "stale source_text_sha256"):
            validate_source_hash(1, ["已变更文字"], expected)

    def test_generation_and_rendering_are_byte_deterministic(self):
        second = _build()
        self.assertEqual(render_json(self.report), render_json(second))
        self.assertEqual(
            render_markdown(self.report),
            render_markdown(second),
        )

    def test_saved_reports_match_deterministic_generation(self):
        self.assertEqual(
            JSON_REPORT.read_text(encoding="utf-8"),
            render_json(self.report),
        )
        self.assertEqual(
            MARKDOWN_REPORT.read_text(encoding="utf-8"),
            render_markdown(self.report),
        )
        self.assertEqual(
            json.loads(JSON_REPORT.read_text(encoding="utf-8")),
            self.report,
        )


if __name__ == "__main__":
    unittest.main()
