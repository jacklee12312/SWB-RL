from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.report_ppo_league_meta_game import (
    BOOTSTRAP_REPLICATES,
    OUTPUT_DIRECTORY,
    build_outputs,
    diagnose_payoff_matrix,
    interquartile_mean,
    solve_zero_sum_meta_strategy,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY = ROOT / OUTPUT_DIRECTORY


class ZeroSumMetaGameSolverTests(unittest.TestCase):
    def test_rock_paper_scissors_is_uniform(self) -> None:
        matrix = [
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
        ]
        result = solve_zero_sum_meta_strategy(matrix)
        for weight in result["weights"]:
            self.assertAlmostEqual(weight, 1.0 / 3.0, places=10)
        self.assertAlmostEqual(result["exploitability_proxy"], 0.0, places=10)

    def test_transitive_game_selects_top_strategy(self) -> None:
        matrix = [
            [0.0, -1.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
        ]
        result = solve_zero_sum_meta_strategy(matrix)
        self.assertEqual(result["weights"], [0.0, 0.0, 1.0])
        self.assertEqual(result["support"], [2])

    def test_duplicate_strategies_are_explicitly_diagnosed(self) -> None:
        matrix = [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
        ]
        diagnostics = diagnose_payoff_matrix(matrix)
        self.assertIn("matrix_has_duplicate_strategies", diagnostics["warnings"])
        self.assertEqual(diagnostics["duplicate_strategy_groups"], [[0, 1]])
        result = solve_zero_sum_meta_strategy(matrix)
        self.assertEqual(result["weights"], [0.0, 0.0, 1.0])

    def test_all_draw_game_uses_documented_canonical_uniform_solution(self) -> None:
        result = solve_zero_sum_meta_strategy([[0.0] * 4 for _ in range(4)])
        self.assertEqual(result["weights"], [0.25] * 4)
        self.assertEqual(result["solver_path"], "all_draw_uniform_canonical_solution")
        self.assertIn("matrix_is_all_draw", result["diagnostics"]["warnings"])

    def test_incomplete_and_non_antisymmetric_matrices_fail_loudly(self) -> None:
        incomplete = [[0.0, None], [None, 0.0]]
        self.assertIn(
            "matrix_has_missing_entries",
            diagnose_payoff_matrix(incomplete)["warnings"],
        )
        with self.assertRaisesRegex(ValueError, "matrix_has_missing_entries"):
            solve_zero_sum_meta_strategy(incomplete)

        invalid = [[0.0, 0.2], [-0.1, 0.0]]
        self.assertIn(
            "matrix_is_not_antisymmetric",
            diagnose_payoff_matrix(invalid)["warnings"],
        )
        with self.assertRaisesRegex(ValueError, "matrix_is_not_antisymmetric"):
            solve_zero_sum_meta_strategy(invalid)

    def test_iqm_uses_fractional_quartile_boundaries(self) -> None:
        self.assertEqual(interquartile_mean([0.0, 1.0]), 0.5)
        self.assertAlmostEqual(
            interquartile_mean([0.0, 0.25, 0.5, 0.75, 1.0]),
            0.5,
        )


class PPOLeagueMetaGameReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.outputs = build_outputs()
        cls.report = json.loads(cls.outputs["meta_game.json"].decode("utf-8"))

    def test_real_candidate_matrix_is_complete_and_valid(self) -> None:
        diagnostics = self.report["matrix_diagnostics"]
        self.assertTrue(diagnostics["valid_zero_sum_antisymmetric"])
        self.assertEqual(diagnostics["missing_entries"], [])
        self.assertEqual(diagnostics["max_antisymmetry_error"], 0.0)
        self.assertEqual(len(self.report["input"]["candidate_model_ids"]), 6)
        self.assertEqual(
            len(self.report["input"]["historical_anchors_excluded_from_meta_game"]),
            3,
        )

    def test_real_nash_support_and_weights_are_deterministic(self) -> None:
        nash = self.report["nash_mixture"]
        self.assertEqual(nash["support"], [0, 3, 4])
        expected = [8 / 19, 0.0, 0.0, 8 / 19, 3 / 19, 0.0]
        for actual, wanted in zip(nash["weights"], expected):
            self.assertAlmostEqual(actual, wanted, places=12)
        self.assertLess(nash["exploitability_proxy"], 1e-12)
        self.assertAlmostEqual(
            nash["effective_population_size"],
            2.635036496350365,
            places=12,
        )

    def test_uniform_mixture_has_measured_internal_weakness(self) -> None:
        uniform = self.report["uniform_mixture"]
        self.assertEqual(uniform["weights"], [1.0 / 6.0] * 6)
        self.assertAlmostEqual(
            uniform["exploitability_proxy"],
            0.054421768707483,
            places=12,
        )
        self.assertLess(uniform["worst_expected_payoff"], 0.0)

    def test_bootstrap_is_paired_and_reports_population_intervals(self) -> None:
        bootstrap = self.report["bootstrap"]
        self.assertEqual(bootstrap["seed"], 20261001)
        self.assertEqual(bootstrap["replicates"], BOOTSTRAP_REPLICATES)
        self.assertTrue(
            bootstrap["resampling_units"]["common_resample_across_all_model_pairs"]
        )
        self.assertEqual(
            set(bootstrap["nash_weight_95_ci"]),
            set(self.report["input"]["candidate_model_ids"]),
        )
        self.assertEqual(
            set(bootstrap["nash_metric_95_ci"]),
            {
                "effective_population_size",
                "exploitability_proxy",
                "worst_expected_payoff",
            },
        )

    def test_cycles_iqm_profiles_and_selection_evidence_are_saved(self) -> None:
        cycles = self.report["cycle_graphs"]
        self.assertEqual(len(cycles["global_models"]["point_estimate"]), 2)
        self.assertEqual(cycles["global_models"]["strong_55_percent"], [])
        self.assertGreaterEqual(len(cycles["classes"]["point_estimate"]), 1)

        robust = self.report["robust_aggregates"]
        self.assertAlmostEqual(robust["population_directed_score_iqm"], 0.5)
        self.assertEqual(len(robust["per_model"]), 6)
        for model in robust["per_model"].values():
            self.assertEqual(len(model["performance_profile"]), 5)

        evidence = self.report["population_selection_evidence"]
        self.assertEqual(len(evidence), 9)
        self.assertEqual(
            sum(item["generation_0_disposition"] == "include_candidate" for item in evidence),
            6,
        )
        self.assertEqual(
            sum(item["generation_0_disposition"] == "include_anchor_only" for item in evidence),
            3,
        )

    def test_saved_outputs_are_byte_stable(self) -> None:
        for name, payload in self.outputs.items():
            with self.subTest(name=name):
                self.assertEqual((REPORT_DIRECTORY / name).read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
