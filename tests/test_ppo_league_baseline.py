from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from scripts.report_ppo_league_baseline import (
    BASELINE_GIT_COMMIT,
    CONTRACT_PATHS,
    LEGACY_ANCHOR_SEEDS,
    NEW_RULE_SEEDS,
    build_reports,
    contract_differences,
    render_json,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIRECTORY = ROOT / "data/reports/league_training"


class PPOLeagueBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reports = build_reports()
        cls.baseline = cls.reports["baseline_manifest.json"]
        cls.protocol = cls.reports["evaluation_protocol.json"]
        cls.registry = cls.reports["checkpoint_registry.json"]

    def test_saved_reports_are_byte_identical_to_fresh_build(self) -> None:
        for name, payload in self.reports.items():
            with self.subTest(name=name):
                self.assertEqual(
                    (REPORT_DIRECTORY / name).read_bytes(),
                    render_json(payload),
                )

    def test_saved_reports_are_valid_canonical_json(self) -> None:
        for name, payload in self.reports.items():
            with self.subTest(name=name):
                saved = json.loads(
                    (REPORT_DIRECTORY / name).read_text(encoding="utf-8")
                )
                self.assertEqual(saved, payload)
                self.assertTrue(render_json(payload).endswith(b"\n"))

    def test_baseline_freezes_required_runtime_contract(self) -> None:
        self.assertEqual(
            self.baseline["source_control"]["baseline_commit"],
            BASELINE_GIT_COMMIT,
        )
        artifacts = self.baseline["artifacts"]
        self.assertEqual(artifacts["fixed_decks"]["count"], 8)
        self.assertEqual(artifacts["class_schedule"]["cycle_length"], 49)
        self.assertEqual(
            len(artifacts["class_schedule"]["ordered_pairs"]),
            49,
        )
        self.assertEqual(
            self.baseline["interfaces"]["observation"]["version"],
            "observation-v4.1",
        )
        self.assertEqual(
            self.baseline["interfaces"]["action"]["version"],
            "action-112-v2",
        )
        self.assertEqual(
            self.baseline["interfaces"]["action"]["manifest"]["size"],
            112,
        )
        self.assertEqual(
            self.baseline["policy_architecture"]["name"],
            "entity_action_v1",
        )
        self.assertEqual(
            self.baseline["policy_architecture"]["model_parameters"],
            5_581_698,
        )
        self.assertEqual(
            self.baseline["interfaces"]["seed_derivation"]["version"],
            1,
        )

    def test_checkpoint_registry_has_six_candidates_and_three_anchors(self) -> None:
        entries = self.registry["entries"]
        self.assertEqual(len(entries), 9)
        self.assertEqual(len({entry["checkpoint_id"] for entry in entries}), 9)
        self.assertEqual(len({entry["checkpoint"]["sha256"] for entry in entries}), 9)

        by_seed = {entry["seed"]: entry for entry in entries}
        self.assertEqual(set(by_seed), {*NEW_RULE_SEEDS, *LEGACY_ANCHOR_SEEDS})
        for seed in NEW_RULE_SEEDS:
            entry = by_seed[seed]
            self.assertEqual(entry["role"], "training_candidate")
            self.assertGreaterEqual(entry["training"]["agent_steps"], 1_000_000)
            self.assertEqual(
                entry["rules_semantics"]["extra_pp"],
                "refundable_until_base_pp_is_exceeded",
            )
        for seed in LEGACY_ANCHOR_SEEDS:
            entry = by_seed[seed]
            self.assertEqual(entry["role"], "anchor_only")
            self.assertGreaterEqual(entry["training"]["agent_steps"], 3_000_000)
            self.assertEqual(
                entry["rules_semantics"]["comparison_scope"],
                "cross_rule_historical_anchor_not_steps_ablation",
            )

    def test_seed_partitions_are_disjoint_and_final_is_held_out(self) -> None:
        partitions = self.protocol["seed_partitions"]
        training = set(partitions["training_model_seeds"])
        tuning = set(partitions["pfsp_tuning_match_master_seeds"])
        final = set(partitions["final_evaluation_match_master_seeds"])
        self.assertFalse(training & tuning)
        self.assertFalse(training & final)
        self.assertFalse(tuning & final)
        self.assertFalse(partitions["final_seeds_allowed_for_pfsp"])

    def test_primary_and_diagnostic_metrics_are_preregistered(self) -> None:
        primary_names = {
            metric["name"] for metric in self.protocol["primary_metrics"]
        }
        self.assertEqual(
            primary_names,
            {
                "frozen_anchor_paired_mean_win_rate",
                "meta_strategy_worst_expected_payoff",
                "class_matrix_worst_cell",
                "class_matrix_p10_cell",
                "tactical_replay_preference",
                "safety",
            },
        )
        diagnostics = set(
            self.protocol["diagnostic_metrics_not_standalone_success"]
        )
        self.assertTrue(
            {
                "agent_steps_per_second",
                "policy_entropy",
                "grad_norm",
                "opponent_selection_distribution",
            }.issubset(diagnostics)
        )

    def test_each_frozen_input_explicitly_invalidates_contract(self) -> None:
        self.assertEqual(
            tuple(self.baseline["contract_invalidation_fields"]),
            CONTRACT_PATHS,
        )
        self.assertEqual(contract_differences(self.baseline, self.baseline), [])
        for path in CONTRACT_PATHS:
            with self.subTest(path=path):
                mutated = deepcopy(self.baseline)
                current = mutated
                keys = path.split(".")
                for key in keys[:-1]:
                    current = current[key]
                current[keys[-1]] = "mutated"
                differences = contract_differences(self.baseline, mutated)
                self.assertEqual(
                    [difference["field"] for difference in differences],
                    [path],
                )


if __name__ == "__main__":
    unittest.main()
