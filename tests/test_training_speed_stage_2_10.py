from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


class TrainingSpeedStage210AcceptanceTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REPORT = (
        ROOT
        / "data/reports/training_speed/final_comparison.json"
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            cls.REPORT.read_text(encoding="utf-8")
        )

    def test_every_final_gate_passes(self) -> None:
        self.assertTrue(self.report["passed"])
        self.assertTrue(all(self.report["gates"].values()))
        self.assertEqual(
            self.report["checklist"]["unchecked_items"],
            [],
        )

    def test_baseline_a_stack_and_final_stack_use_three_runs(
        self,
    ) -> None:
        baseline = self.report["baseline"]
        a_stack = self.report["a_class_stack"]
        final_stack = self.report["final_adopted_stack"]
        self.assertEqual(baseline["run_count"], 3)
        self.assertEqual(
            len(baseline["runs_agent_steps_per_second"]),
            3,
        )
        self.assertEqual(a_stack["run_count"], 3)
        self.assertEqual(
            len(a_stack["runs_agent_steps_per_second"]),
            3,
        )
        self.assertGreaterEqual(
            a_stack["relative_gain_vs_baseline"],
            0.25,
        )
        self.assertTrue(a_stack["target_met"])
        self.assertEqual(
            len(final_stack["runs_agent_steps_per_second"]),
            3,
        )
        self.assertGreater(
            final_stack["relative_gain_vs_frozen_baseline"],
            a_stack["relative_gain_vs_baseline"],
        )

    def test_long_run_is_monitored_and_stable(self) -> None:
        stability = self.report["stability_100k"]
        result = stability["result"]
        monitor = stability["system_monitor_summary"]
        self.assertGreaterEqual(
            result["completed_additional_agent_steps"],
            100_000,
        )
        self.assertTrue(stability["checkpoint_unchanged"])
        self.assertTrue(stability["finite_update_metrics"])
        self.assertTrue(monitor["no_page_in_or_page_out"])
        self.assertGreaterEqual(monitor["sample_count"], 100)
        self.assertLess(
            monitor["gpu_memory_peak_mib"],
            stability["gpu_memory_total_mib"],
        )
        self.assertTrue(
            stability["truncations"]["no_abnormal_increase"]
        )

    def test_contract_and_all_candidate_dispositions_are_closed(
        self,
    ) -> None:
        contract = self.report["contract_preservation"]
        self.assertTrue(contract["preserved"])
        self.assertEqual(
            contract["changed_training_paths_since_stage_2_7_final"],
            [],
        )
        outcomes = self.report["candidate_outcomes"]
        self.assertEqual(outcomes["candidate_count"], 24)
        self.assertEqual(len(outcomes["rows"]), 24)
        self.assertEqual(
            set(outcomes["groups"]["adopted"]),
            {
                "A-BATCH-WAIT-001",
                "A-WORKERS-001",
                "B-BATCHED-LEARNER-001",
            },
        )

    def test_final_report_sources_match_saved_evidence(self) -> None:
        for source in self.report["sources"].values():
            content = (self.ROOT / source["path"]).read_bytes()
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                source["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
