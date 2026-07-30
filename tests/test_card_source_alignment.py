from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.report_card_bug_audit_baseline import render_json
from scripts.report_card_source_alignment import (
    DEFAULT_BASELINE,
    DEFAULT_CLOSURE,
    DEFAULT_DATABASE,
    DEFAULT_MATRIX,
    DEFAULT_RULES,
    build_source_alignment,
    render_markdown,
    validate_ruling_queue,
)


ROOT = Path(__file__).resolve().parents[1]
JSON_REPORT = ROOT / "data/reports/card_bug_audit/source_alignment.json"
MARKDOWN_REPORT = ROOT / "data/reports/card_bug_audit/source_alignment.md"
RULING_QUEUE = ROOT / "data/reports/card_bug_audit/ruling_queue.json"


def _build():
    return build_source_alignment(
        root=ROOT,
        database=ROOT / DEFAULT_DATABASE,
        rules_directory=ROOT / DEFAULT_RULES,
        baseline_report=ROOT / DEFAULT_BASELINE,
        closure_report=ROOT / DEFAULT_CLOSURE,
        matrix_report=ROOT / DEFAULT_MATRIX,
    )


class CardSourceAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.queue = _build()

    def test_every_training_closure_card_passes_source_alignment(self):
        summary = self.report["summary"]
        self.assertEqual(summary["card_count"], 147)
        self.assertEqual(summary["collectible_count"], 116)
        self.assertEqual(summary["generated_count"], 31)
        self.assertEqual(summary["passed"], 147)
        self.assertEqual(summary["ruling_uncertain"], 0)
        self.assertEqual(summary["failed"], 0)
        self.assertTrue(summary["source_alignment_gate_ready"])

    def test_multilingual_texts_and_printed_fields_match_raw_imports(self):
        self.assertEqual(
            self.report["summary"]["source_clause_count"],
            161,
        )
        for card in self.report["cards"]:
            with self.subTest(card_id=card["card_id"]):
                self.assertEqual(
                    card["printed_field_alignment"]["status"],
                    "passed",
                )
                self.assertEqual(
                    card["multilingual_completeness"]["status"],
                    "passed",
                )
                self.assertEqual(
                    card["source_texts"]["raw_import_alignment"],
                    "passed",
                )
                self.assertIn(
                    card["source_texts"]["source_hash_status"],
                    {"passed", "not_applicable"},
                )
                self.assertEqual(card["base_keywords"]["status"], "passed")

    def test_every_card_reference_resolves_to_named_version(self):
        self.assertEqual(self.report["summary"]["reference_count"], 76)
        for card in self.report["cards"]:
            self.assertEqual(card["reference_alignment"]["status"], "passed")
            for reference in card["references"]:
                with self.subTest(
                    card_id=card["card_id"],
                    position=reference["position"],
                ):
                    self.assertTrue(reference["target_exists"])
                    self.assertTrue(
                        reference["referenced_name_matches_target"]
                    )
                    self.assertEqual(reference["status"], "passed")

    def test_timing_quantity_and_dedup_signals_have_rule_markers(self):
        expected_signals = {
            "condition",
            "whenever",
            "until",
            "this_turn",
            "owner_turn",
            "random",
            "all",
            "one",
            "other",
            "different_names",
        }
        coverage = self.report["summary"]["semantic_signal_coverage"]
        self.assertEqual(set(coverage), expected_signals)
        self.assertEqual(coverage["whenever"]["status"], "not_applicable")
        self.assertEqual(coverage["this_turn"]["status"], "not_applicable")
        self.assertEqual(coverage["different_names"]["status"], "passed")
        self.assertTrue(
            all(
                item["status"] in {"passed", "not_applicable"}
                for item in coverage.values()
            )
        )
        for card in self.report["cards"]:
            self.assertEqual(card["semantic_alignment"]["status"], "passed")
            for signal in card["semantic_signals"]:
                with self.subTest(
                    card_id=card["card_id"],
                    signal=signal["signal"],
                ):
                    self.assertEqual(signal["status"], "passed")
                    self.assertTrue(signal["rule_markers"])
                    self.assertTrue(signal["candidate_entry_ids"])

    def test_empty_ruling_queue_preserves_evidence_policy(self):
        validate_ruling_queue(self.queue)
        self.assertEqual(self.queue["summary"]["entry_count"], 0)
        self.assertEqual(self.queue["summary"]["open_count"], 0)
        self.assertFalse(
            self.queue["summary"]["source_alignment_blocks_training"]
        )
        hierarchy = self.queue["evidence_hierarchy"]
        self.assertEqual([item["tier"] for item in hierarchy], [1, 2, 3, 4, 5])
        self.assertEqual(
            hierarchy[0]["kind"],
            "official_qa_or_rules",
        )
        self.assertFalse(
            self.queue["external_evidence_contract"][
                "text_only_inference_can_close_entry"
            ]
        )

    def test_closed_external_ruling_requires_complete_evidence(self):
        incomplete = {
            "entries": [
                {
                    "ruling_id": "card:1:test",
                    "status": "confirmed",
                    "source_url": "https://example.invalid/rule",
                    "retrieved_at": None,
                    "conclusion": "test",
                    "summary": "test",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_ruling_queue(incomplete)
        incomplete["entries"][0]["retrieved_at"] = "2026-07-29"
        validate_ruling_queue(incomplete)

    def test_database_refresh_invalidates_saved_matrix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            changed_database = temporary / "cards.sqlite3"
            changed_database.write_bytes((ROOT / DEFAULT_DATABASE).read_bytes())
            with changed_database.open("ab") as destination:
                destination.write(b"source-refresh")
            with self.assertRaisesRegex(
                ValueError,
                "source database changed",
            ):
                build_source_alignment(
                    root=ROOT,
                    database=changed_database,
                    rules_directory=ROOT / DEFAULT_RULES,
                    baseline_report=ROOT / DEFAULT_BASELINE,
                    closure_report=ROOT / DEFAULT_CLOSURE,
                    matrix_report=ROOT / DEFAULT_MATRIX,
                )

    def test_generation_and_saved_reports_are_byte_deterministic(self):
        second_report, second_queue = _build()
        self.assertEqual(render_json(self.report), render_json(second_report))
        self.assertEqual(render_json(self.queue), render_json(second_queue))
        self.assertEqual(
            render_markdown(self.report),
            render_markdown(second_report),
        )
        self.assertEqual(
            JSON_REPORT.read_text(encoding="utf-8"),
            render_json(self.report),
        )
        self.assertEqual(
            MARKDOWN_REPORT.read_text(encoding="utf-8"),
            render_markdown(self.report),
        )
        self.assertEqual(
            RULING_QUEUE.read_text(encoding="utf-8"),
            render_json(self.queue),
        )
        self.assertEqual(
            json.loads(JSON_REPORT.read_text(encoding="utf-8")),
            self.report,
        )


if __name__ == "__main__":
    unittest.main()
