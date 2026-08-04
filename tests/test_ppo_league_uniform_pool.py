from __future__ import annotations

import json
import unittest

from scripts.report_ppo_league_baseline import render_json
from scripts.report_ppo_league_uniform_pool import (
    DEFAULT_OUTPUT,
    HUNDRED_K_REPORT,
    PAIRED_BASELINE_REPORT,
    ROOT,
    build_uniform_pool_report,
)


@unittest.skipUnless(
    (ROOT / HUNDRED_K_REPORT).is_file()
    and (ROOT / PAIRED_BASELINE_REPORT).is_file(),
    "paired 100k smoke reports have not completed",
)
class PPOLeagueUniformPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.saved_path = ROOT / DEFAULT_OUTPUT
        cls.saved = json.loads(cls.saved_path.read_text(encoding="utf-8"))

    def test_saved_report_is_byte_stable(self) -> None:
        self.assertEqual(
            self.saved_path.read_bytes(),
            render_json(build_uniform_pool_report()),
        )

    def test_uniform_pool_passes_registered_gates(self) -> None:
        self.assertTrue(self.saved["passed"])
        self.assertTrue(all(
            gate["passed"] for gate in self.saved["gates"].values()
        ))
        self.assertEqual(
            self.saved["contract"]["semantic_config_mismatches"],
            {},
        )
        self.assertEqual(
            self.saved["contract"][
                "paired_baseline_semantic_config_mismatches"
            ],
            {},
        )
        self.assertEqual(
            self.saved["contract"][
                "paired_baseline_runtime_config_mismatches"
            ],
            {},
        )
        self.assertEqual(
            len(self.saved["population"]["trainable_opponent_ids"]),
            24,
        )
        self.assertEqual(
            len(self.saved["population"]["anchor_only_opponent_ids"]),
            3,
        )

    def test_smokes_are_safe_and_cover_the_full_runtime_pool(self) -> None:
        for smoke in self.saved["smoke_runs"].values():
            self.assertEqual(smoke["illegal_action_errors"], 0)
            self.assertEqual(smoke["action_mask_mismatch_errors"], 0)
            self.assertTrue(smoke["completed_without_exception"])
        runtime_gate = self.saved["gates"]["runtime_selection"]
        self.assertEqual(runtime_gate["missing_trainable_models"], [])
        self.assertEqual(runtime_gate["selected_anchor_models"], [])

    def test_throughput_gate_uses_paired_same_runtime_baseline(self) -> None:
        gate = self.saved["gates"]["throughput"]
        self.assertEqual(
            gate["baseline_kind"],
            "same_runtime_100k_own_history_pool",
        )
        self.assertGreaterEqual(
            gate["uniform_to_paired_baseline_ratio"],
            gate["minimum_ratio"],
        )
        self.assertFalse(
            self.saved["historical_scaling_reference"][
                "used_as_decision_gate"
            ]
        )


if __name__ == "__main__":
    unittest.main()
