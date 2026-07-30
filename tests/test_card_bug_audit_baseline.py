from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from scripts.report_card_bug_audit_baseline import (
    DEFAULT_AUDIT_START,
    DEFAULT_ABILITY_AUDIT,
    DEFAULT_CLAUSE_AUDIT,
    DEFAULT_COVERAGE_REPORT,
    DEFAULT_DATABASE,
    DEFAULT_RULES,
    DEFAULT_SOURCE_JSON,
    DEFAULT_TOKEN_AUDIT,
    _rule_reference_edges,
    build_reports,
    load_audit_start,
    render_json,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_REPORT = ROOT / "data/reports/card_bug_audit/baseline.json"
CLOSURE_REPORT = (
    ROOT / "data/reports/card_bug_audit/training_deck_card_closure.json"
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _build():
    return build_reports(
        root=ROOT,
        audit_start=ROOT / DEFAULT_AUDIT_START,
        database=ROOT / DEFAULT_DATABASE,
        source_json=ROOT / DEFAULT_SOURCE_JSON,
        rules_directory=ROOT / DEFAULT_RULES,
        clause_audit=ROOT / DEFAULT_CLAUSE_AUDIT,
        token_audit=ROOT / DEFAULT_TOKEN_AUDIT,
        ability_audit=ROOT / DEFAULT_ABILITY_AUDIT,
        coverage_report=ROOT / DEFAULT_COVERAGE_REPORT,
    )


class CardBugAuditBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline, cls.closure = _build()

    def test_database_and_audit_hashes_are_frozen(self):
        counts = self.baseline["database"]["counts"]
        self.assertEqual(
            counts,
            {
                "total": 826,
                "collectible": 735,
                "non_collectible_or_derived": 91,
            },
        )
        self.assertEqual(
            self.baseline["database"]["source_snapshot"]["card_count"],
            826,
        )
        self.assertTrue(
            SHA256_PATTERN.fullmatch(self.baseline["database"]["sha256"])
        )
        for artifact in self.baseline["audit_artifacts"].values():
            self.assertTrue(SHA256_PATTERN.fullmatch(artifact["sha256"]))

    def test_git_audit_start_records_commit_branch_and_workspace(self):
        git_state = self.baseline["git_audit_start"]
        self.assertTrue(re.fullmatch(r"[0-9a-f]{40}", git_state["commit"]))
        self.assertTrue(git_state["branch"])
        self.assertIsInstance(git_state["is_clean"], bool)
        self.assertIsInstance(git_state["status_porcelain"], list)
        self.assertEqual(
            git_state["manifest_path"],
            "data/audits/card_bug_audit_start.json",
        )
        self.assertTrue(
            SHA256_PATTERN.fullmatch(git_state["manifest_sha256"])
        )
        self.assertNotIn(
            "report_card_bug_audit_baseline.py",
            "\n".join(git_state["status_porcelain"]),
        )
        self.assertNotIn(
            "training_deck_card_closure.json",
            "\n".join(git_state["status_porcelain"]),
        )

    def test_frozen_audit_start_accepts_later_checkpoint_heads(self):
        frozen = load_audit_start(
            ROOT / DEFAULT_AUDIT_START,
            root=ROOT,
        )
        self.assertEqual(
            frozen["commit"],
            json.loads(
                (ROOT / DEFAULT_AUDIT_START).read_text(encoding="utf-8")
            )["git_state"]["commit"],
        )

    def test_frozen_audit_start_rejects_missing_commit(self):
        payload = json.loads(
            (ROOT / DEFAULT_AUDIT_START).read_text(encoding="utf-8")
        )
        payload["git_state"]["commit"] = "0" * 40
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary_directory:
            path = Path(temporary_directory) / "audit_start.json"
            path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "frozen audit-start commit is not available",
            ):
                load_audit_start(path, root=ROOT)

    def test_eight_decks_have_frozen_forty_card_manifests(self):
        decks = self.baseline["fixed_training_decks"]
        self.assertEqual(len(decks), 8)
        self.assertEqual(len({deck["name"] for deck in decks}), 8)
        self.assertEqual({deck["class_id"] for deck in decks}, set(range(1, 8)))
        for deck in decks:
            with self.subTest(deck=deck["name"]):
                self.assertEqual(len(deck["card_ids"]), 40)
                self.assertTrue(SHA256_PATTERN.fullmatch(deck["sha256"]))
                self.assertEqual(sum(deck["card_counts"].values()), 40)

    def test_recursive_closure_is_stable_and_fully_resolved(self):
        summary = self.closure["summary"]
        self.assertEqual(summary["fixed_deck_count"], 8)
        self.assertEqual(summary["fixed_deck_collectible_union_count"], 111)
        self.assertEqual(summary["closure_card_count"], 147)
        self.assertEqual(summary["recursive_reference_count"], 36)
        self.assertTrue(summary["all_database_resolved"])
        self.assertTrue(summary["all_rulebook_and_audit_resolved"])
        cards = self.closure["cards"]
        self.assertEqual(len({card["audit_id"] for card in cards}), len(cards))
        self.assertEqual(
            [card["card_id"] for card in cards],
            sorted(card["card_id"] for card in cards),
        )
        for card in cards:
            with self.subTest(card_id=card["card_id"]):
                self.assertTrue(card["resolution"]["database"])
                self.assertTrue(
                    card["resolution"]["rulebook_lookup_succeeded"]
                )
                self.assertTrue(
                    card["resolution"]["audit_resolution_passed"]
                )

    def test_closure_includes_database_and_structured_rule_edges(self):
        relations = {
            edge["relation"] for edge in self.closure["reference_edges"]
        }
        self.assertIn("database_reference", relations)
        self.assertIn("summon", relations)
        self.assertIn("add_to_hand", relations)
        self.assertIn("transform", relations)

    def test_rule_edge_parser_covers_all_required_producer_definitions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rules = root / "rules"
            rules.mkdir()
            payload = {
                "rules": [
                    {
                        "card_id": 1,
                        "operations": [
                            {"kind": "add_card", "card_id": 2},
                            {"kind": "add_card_to_deck", "card_id": 3},
                            {"kind": "transform_deck_cards", "card_id": 4},
                            {"kind": "replace_deck", "card_ids": [5, 6]},
                        ],
                    }
                ],
                "emblems": [
                    {
                        "source_card_id": 7,
                        "triggers": [
                            {
                                "operations": [
                                    {"kind": "summon", "card_id": 8}
                                ]
                            }
                        ],
                    }
                ],
                "faiths": [
                    {
                        "card_id": 9,
                        "operations": [
                            {"kind": "transform", "card_id": 10}
                        ],
                    }
                ],
                "fusions": [
                    {
                        "card_id": 11,
                        "transform_results": [{"card_id": 12}],
                    }
                ],
            }
            (rules / "fixture.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            authored, edges = _rule_reference_edges(rules, root)

        self.assertEqual(authored, {1, 7, 9, 11})
        self.assertEqual(
            {
                (
                    edge["source_card_id"],
                    edge["target_card_id"],
                    edge["relation"],
                )
                for edge in edges
            },
            {
                (1, 2, "add_to_hand"),
                (1, 3, "add_to_deck"),
                (1, 4, "transform_deck_cards"),
                (1, 5, "replace_deck"),
                (1, 6, "replace_deck"),
                (7, 8, "summon"),
                (9, 10, "transform"),
                (11, 12, "fusion_transform"),
            },
        )

    def test_generation_is_byte_deterministic(self):
        second_baseline, second_closure = _build()
        self.assertEqual(
            render_json(self.baseline),
            render_json(second_baseline),
        )
        self.assertEqual(
            render_json(self.closure),
            render_json(second_closure),
        )

    def test_saved_reports_match_current_deterministic_generation(self):
        self.assertEqual(
            BASELINE_REPORT.read_text(encoding="utf-8"),
            render_json(self.baseline),
        )
        self.assertEqual(
            CLOSURE_REPORT.read_text(encoding="utf-8"),
            render_json(self.closure),
        )
        self.assertEqual(
            json.loads(BASELINE_REPORT.read_text(encoding="utf-8")),
            self.baseline,
        )


if __name__ == "__main__":
    unittest.main()
