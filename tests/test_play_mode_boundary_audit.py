from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.report_play_mode_boundary_audit import (
    DEFAULT_CLOSURE,
    DEFAULT_DATABASE,
    DEFAULT_MARKDOWN,
    DEFAULT_OUTPUT,
    DEFAULT_RULES,
    MODIFIER_SCENARIOS,
    build_report,
    render_json,
    render_markdown,
)
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook


ROOT = Path(__file__).resolve().parents[1]


def _build():
    return build_report(
        root=ROOT,
        database=ROOT / DEFAULT_DATABASE,
        rules_directory=ROOT / DEFAULT_RULES,
        closure_path=ROOT / DEFAULT_CLOSURE,
    )


class PlayModeBoundaryAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _build()

    def test_full_pool_and_training_closure_scopes_are_complete(self):
        repository = CardRepository(ROOT / DEFAULT_DATABASE)
        rulebook = RuleBook.from_directory(ROOT / DEFAULT_RULES)
        expected = {
            card.card_id
            for card in repository.all_cards()
            if rulebook.modes_for(card.card_id)
        }
        actual = {card["card_id"] for card in self.report["cards"]}
        self.assertEqual(actual, expected)
        self.assertTrue(self.report["summary"]["scope_complete"])
        self.assertTrue(all(card["passed"] for card in self.report["cards"]))
        self.assertTrue(
            any(card["in_training_deck_closure"] for card in self.report["cards"])
        )

    def test_all_required_cost_and_modifier_boundaries_are_scanned(self):
        expected_scenarios = {
            scenario["id"] for scenario in MODIFIER_SCENARIOS
        }
        for card in self.report["cards"]:
            with self.subTest(card_id=card["card_id"]):
                scenarios = {
                    scenario["id"]: scenario
                    for scenario in card["modifier_scenarios"]
                }
                self.assertEqual(set(scenarios), expected_scenarios)
                for scenario in scenarios.values():
                    labels = {
                        label
                        for case in scenario["cases"]
                        for label in case["boundary_labels"]
                    }
                    self.assertIn("zero", labels)
                    self.assertIn("printed_cost_exact", labels)
                    self.assertTrue(
                        "current_cost_exact" in labels
                        or "current_cost_exact"
                        in card["unavailable_boundaries"]
                    )
                    for mode in card["modes"]:
                        self.assertIn(
                            f"{mode['mode_id']}_cost_exact",
                            labels,
                        )

    def test_commands_masks_execution_and_illegal_atomicity_all_pass(self):
        summary = self.report["summary"]
        self.assertEqual(summary["command_action_mask_mismatch_count"], 0)
        self.assertEqual(summary["illegal_atomicity_failure_count"], 0)
        self.assertEqual(summary["execution_failure_count"], 0)
        self.assertEqual(summary["failure_count"], 0)
        self.assertTrue(summary["passed"])
        for card in self.report["cards"]:
            for scenario in card["modifier_scenarios"]:
                for case in scenario["cases"]:
                    with self.subTest(
                        card_id=card["card_id"],
                        scenario=scenario["id"],
                        remaining_pp=case["remaining_pp"],
                    ):
                        self.assertEqual(
                            case["expected_mode_ids"],
                            case["legal_command_mode_ids"],
                        )
                        self.assertEqual(
                            case["expected_mode_ids"],
                            case["action_mask_mode_ids"],
                        )
                        self.assertTrue(case["illegal_atomicity_passed"])
                        if case["execution"] is not None:
                            self.assertTrue(case["execution"]["passed"])
                            self.assertEqual(
                                case["execution"]["combo_delta"],
                                1,
                            )

    def test_full_board_uses_effective_mode_card_type(self):
        for card in self.report["cards"]:
            for case in card["full_board_cases"]:
                with self.subTest(
                    card_id=card["card_id"],
                    mode_id=case["mode_id"],
                ):
                    self.assertTrue(case["passed"])
                    self.assertEqual(
                        case["command_legal"],
                        case["action_mask_legal"],
                    )
                    self.assertEqual(
                        case["command_legal"],
                        case["effective_card_type"] == "法术",
                    )

    def test_real_regression_cards_have_high_pp_exclusivity_evidence(self):
        by_id = {card["card_id"]: card for card in self.report["cards"]}
        for card_id, mode_id in (
            (10424110, "enhance_6"),
            (10661110, "crystallize_2"),
            (10671110, "accelerate_2"),
        ):
            card = by_id[card_id]
            printed = next(
                scenario
                for scenario in card["modifier_scenarios"]
                if scenario["id"] == "printed"
            )
            high_pp = next(
                case
                for case in printed["cases"]
                if case["remaining_pp"] == card["printed_cost"]
            )
            with self.subTest(card_id=card_id):
                self.assertNotIn(mode_id, high_pp["legal_command_mode_ids"])
                if mode_id.startswith("enhance"):
                    threshold = next(
                        case
                        for case in printed["cases"]
                        if case["remaining_pp"]
                        == next(
                            mode["mode_cost"]
                            for mode in card["modes"]
                            if mode["mode_id"] == mode_id
                        )
                    )
                    self.assertEqual(threshold["expected_mode_ids"], [mode_id])
                    self.assertNotIn(
                        "normal",
                        threshold["legal_command_mode_ids"],
                    )

    def test_saved_reports_match_deterministic_generation(self):
        saved = json.loads(
            (ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8")
        )
        normalized = json.loads(render_json(self.report))
        # Stage 1 reports are frozen evidence. Later non-semantic timing
        # instrumentation may change this source hash without changing the
        # report inputs or conclusions.
        normalized["inputs"]["environment_source_sha256"] = (
            saved["inputs"]["environment_source_sha256"]
        )
        self.assertEqual(
            saved,
            normalized,
        )
        self.assertEqual(
            (ROOT / DEFAULT_MARKDOWN).read_text(encoding="utf-8"),
            render_markdown(normalized),
        )

    def test_repeated_generation_is_byte_deterministic(self):
        second = _build()
        self.assertEqual(render_json(self.report), render_json(second))
        self.assertEqual(
            render_markdown(self.report),
            render_markdown(second),
        )


if __name__ == "__main__":
    unittest.main()
