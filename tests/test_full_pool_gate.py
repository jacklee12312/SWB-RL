from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from scripts.report_full_pool_gate import (
    DEFAULT_MARKDOWN,
    DEFAULT_OUTPUT,
    ROOT,
    build_report,
    render_json,
    render_markdown,
)


class FullPoolGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()
        cls.saved_report = json.loads(
            (ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8")
        )

    def test_saved_reports_match_frozen_generation(self) -> None:
        freeze_commit = subprocess.run(
            [
                "git",
                "log",
                "-1",
                "--format=%H",
                "--",
                DEFAULT_OUTPUT.as_posix(),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(len(freeze_commit), 40)
        for path in (
            DEFAULT_OUTPUT,
            DEFAULT_MARKDOWN,
        ):
            self.assertEqual(
                (ROOT / path).read_text(encoding="utf-8"),
                subprocess.run(
                    ["git", "show", f"{freeze_commit}:{path.as_posix()}"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                ).stdout,
            )

        current_normalized = json.loads(render_json(self.report))
        # The Stage 1 gate is a frozen historical artifact. Later audit
        # metadata may change without changing the frozen conclusion or any
        # gate result.
        for input_name in ("rulings", "rl_interface"):
            current_normalized["inputs"][input_name]["sha256"] = (
                self.saved_report["inputs"][input_name]["sha256"]
            )
        for field in ("tests_sha256", "scripts_sha256"):
            current_normalized["frozen"][field] = (
                self.saved_report["frozen"][field]
            )
        current_freeze_gate = next(
            gate
            for gate in current_normalized["gates"]
            if gate["gate_id"] == "1.15.9"
        )
        saved_freeze_gate = next(
            gate
            for gate in self.saved_report["gates"]
            if gate["gate_id"] == "1.15.9"
        )
        current_freeze_gate["metrics"]["tests_sha256"] = (
            saved_freeze_gate["metrics"]["tests_sha256"]
        )
        self.assertEqual(
            self.saved_report,
            current_normalized,
        )
        self.assertEqual(
            (ROOT / DEFAULT_MARKDOWN).read_text(encoding="utf-8"),
            render_markdown(current_normalized),
        )

    def test_all_nine_checklist_gates_pass(self) -> None:
        summary = self.report["summary"]
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["gate_count"], 9)
        self.assertEqual(summary["passed_gate_count"], 9)
        self.assertEqual(summary["failed_gates"], [])

    def test_full_pool_counts_and_bug_blockers_are_exact(self) -> None:
        summary = self.report["summary"]
        self.assertEqual(summary["collectible_audited"], 735)
        self.assertEqual(summary["generated_audited"], 91)
        self.assertEqual(summary["open_p0"], 0)
        self.assertEqual(summary["open_p1"], 0)

    def test_uncertain_ruling_remains_explicitly_excluded(self) -> None:
        summary = self.report["summary"]
        frozen = self.report["frozen"]
        self.assertEqual(summary["trainable_collectible"], 734)
        self.assertEqual(summary["excluded_pending_ruling"], 1)
        self.assertEqual(frozen["excluded_collectible_ids"], [10233310])
        limitation = next(
            row
            for row in self.report["known_limitations"]
            if row["id"] == "SWB-RULING-SET-STATS-TEMP-001"
        )
        self.assertEqual(limitation["status"], "ruling_uncertain")

    def test_runtime_statuses_are_not_relabelled(self) -> None:
        gate = next(
            row for row in self.report["gates"]
            if row["gate_id"] == "1.15.5"
        )
        counts = gate["metrics"]["runtime_status_counts"]
        self.assertGreater(counts["not_sampled_full_pool"], 0)
        self.assertGreater(counts["not_triggered"], 0)
        self.assertGreater(counts["triggered_not_executed"], 0)
        self.assertNotIn("passed", counts)
        self.assertEqual(gate["metrics"]["unexplained_count"], 0)

    def test_sampling_and_preserved_failure_are_both_recorded(self) -> None:
        gate = next(
            row for row in self.report["gates"]
            if row["gate_id"] == "1.15.6"
        )
        self.assertEqual(gate["metrics"]["games"], 10_000)
        self.assertEqual(gate["metrics"]["replays"], 98)
        self.assertIn("earlier failed run", gate["conclusion"])
        self.assertEqual(gate["status"], "passed")

    def test_freeze_contains_all_required_hashes(self) -> None:
        frozen = self.report["frozen"]
        for key in (
            "database_sha256",
            "rules_sha256",
            "coverage_report_sha256",
            "catalog_exclusion_policy_sha256",
            "catalog_sha256",
            "card_vocabulary_sha256",
            "training_pool_sha256",
            "tests_sha256",
            "scripts_sha256",
        ):
            with self.subTest(key=key):
                self.assertEqual(len(frozen[key]), 64)
        self.assertEqual(
            set(frozen["observation"]),
            {"observation-v3.6", "observation-v4.1"},
        )
        self.assertEqual(frozen["action_layout"]["size"], 112)

    def test_machine_report_is_valid_json(self) -> None:
        self.assertEqual(
            json.loads(render_json(self.report))["report_kind"],
            "swb_card_bug_audit_final_gate",
        )


if __name__ == "__main__":
    unittest.main()
