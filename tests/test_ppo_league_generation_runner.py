from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from scripts.report_ppo_league_evolving import (
    ACTIVE_FRACTION,
    ACTIVE_SEEDS,
    ARCHIVE_FRACTION,
    build_generation_zero_active_matrix,
)
from scripts.run_ppo_league_generations import (
    GENERATION_ZERO_MANIFEST,
    LINEAGE_MANIFEST_DIRECTORY,
    ROOT,
    VALIDATION_NONDEGRADATION_SCORE,
    _archive_confirmation_required,
    _build_population_and_lineage_manifests,
    _build_population_summary,
    _generation_gate,
    _is_forgotten,
    _lineage_manifest_path,
    _periodic_checkpoint_candidates,
    _read_json,
    _training_command,
)


class PPOLeagueGenerationRunnerTests(unittest.TestCase):
    def test_training_command_replaces_pool_only_at_generation_boundary(self) -> None:
        command = _training_command(
            resume=Path("parent.pt"),
            checkpoint=Path("child.pt"),
            metrics=Path("metrics.json"),
            target_agent_steps=1_250_000,
            lineage_manifest=Path("manifest.json"),
            replace_opponent_pool=True,
        )
        self.assertIn("--resume-opponent-pool-overrides", command)
        self.assertIn("--opponent-external-manifest", command)
        self.assertIn("episode_seed_clustered", command)
        resumed = _training_command(
            resume=Path("child.pt"),
            checkpoint=Path("child.pt"),
            metrics=Path("metrics.json"),
            target_agent_steps=1_250_000,
            lineage_manifest=None,
            replace_opponent_pool=False,
        )
        self.assertNotIn("--resume-opponent-pool-overrides", resumed)
        self.assertNotIn("--opponent-external-manifest", resumed)

    def test_lineage_order_does_not_change_source_manifest(self) -> None:
        for seed in ACTIVE_SEEDS:
            forward = _lineage_manifest_path(
                Path("data/reports/league_training/generations"), 0, seed
            )
            reverse = _lineage_manifest_path(
                Path("data/reports/league_training/generations"), 0, seed
            )
            self.assertEqual(forward, reverse)
            self.assertTrue((ROOT / forward).is_file())
            custom_root = _lineage_manifest_path(
                Path("data/reports/league_training/custom_smoke"), 0, seed
            )
            self.assertEqual(
                custom_root,
                LINEAGE_MANIFEST_DIRECTORY / f"seed_{seed}_1m.json",
            )

    def test_next_population_keeps_six_active_and_bounded_archive(self) -> None:
        source = _read_json(GENERATION_ZERO_MANIFEST)
        matrix = build_generation_zero_active_matrix()
        active = []
        for row in source["entries"]:
            if row["role"] != "candidate_final":
                continue
            entry = dict(row)
            entry["opponent_id"] = f"seed_{entry['policy_seed']}_g001"
            entry["generation"] = 1
            active.append(entry)
        old_ids = matrix["active_policy_ids"]
        new_ids = [
            next(
                entry["opponent_id"]
                for entry in active
                if entry["policy_seed"] == int(old_id.split("_")[1])
            )
            for old_id in old_ids
        ]
        matrix["generation"] = 1
        matrix["active_policy_ids"] = new_ids
        matrix["score_matrix"] = {
            new_left: {
                new_right: matrix["score_matrix"][old_left][old_right]
                for old_right, new_right in zip(old_ids, new_ids)
            }
            for old_left, new_left in zip(old_ids, new_ids)
        }
        matrix["confidence_interval_95_matrix"] = {
            new_left: {
                new_right: matrix["confidence_interval_95_matrix"][old_left][old_right]
                for old_right, new_right in zip(old_ids, new_ids)
            }
            for old_left, new_left in zip(old_ids, new_ids)
        }
        population, manifests, archive = _build_population_and_lineage_manifests(
            source_population=source,
            source_generation=0,
            target_generation=1,
            active_entries=active,
            active_payoff=matrix,
        )
        self.assertEqual(population["summary"]["active_latest_count"], 6)
        self.assertEqual(population["summary"]["selected_historical_archive_count"], 24)
        self.assertEqual(len(archive["selected_archive_ids"]), 24)
        self.assertEqual(set(manifests), set(ACTIVE_SEEDS))
        for manifest in manifests.values():
            positive = [
                entry for entry in manifest["entries"]
                if entry["sampling_weight"] > 0
            ]
            active_rows = [row for row in positive if row["role"] == "active_latest"]
            archive_rows = [
                row for row in positive if row["role"] == "historical_archive"
            ]
            self.assertEqual((len(active_rows), len(archive_rows)), (6, 24))
            self.assertTrue(math.isclose(
                sum(row["sampling_weight"] for row in active_rows),
                ACTIVE_FRACTION,
                abs_tol=1e-12,
            ))
            self.assertTrue(math.isclose(
                sum(row["sampling_weight"] for row in archive_rows),
                ARCHIVE_FRACTION,
                abs_tol=1e-12,
            ))

    def test_generation_gate_stops_safety_regression_and_two_low_gains(self) -> None:
        validation = {
            "summary": {
                "mean_score_rate": 0.55,
                "minimum_score_rate": VALIDATION_NONDEGRADATION_SCORE,
                "nondegraded_lineages": 6,
            }
        }
        first = _generation_gate(validation, previous_training_report=None)
        self.assertTrue(first["passed"])
        previous = {"gate": first}
        validation["summary"]["mean_score_rate"] = 0.56
        second = _generation_gate(validation, previous_training_report=previous)
        self.assertTrue(second["passed"])
        self.assertEqual(second["consecutive_low_gain_generations"], 1)
        validation["summary"]["mean_score_rate"] = 0.57
        third = _generation_gate(
            validation, previous_training_report={"gate": second}
        )
        self.assertFalse(third["passed"])
        self.assertIn(
            "two_consecutive_generations_below_minimum_gain",
            third["stop_reasons"],
        )
        validation["summary"]["minimum_score_rate"] = 0.47
        regressed = _generation_gate(validation, previous_training_report=None)
        self.assertFalse(regressed["passed"])

    def test_archive_forgetting_requires_screen_and_confirmation(self) -> None:
        self.assertTrue(_archive_confirmation_required(0.70, 0.45))
        self.assertFalse(_archive_confirmation_required(0.69, 0.30))
        self.assertFalse(_archive_confirmation_required(0.90, 0.46))
        self.assertTrue(_is_forgotten(0.70, 0.399, confirmed=True))
        self.assertFalse(_is_forgotten(0.70, 0.40, confirmed=True))
        self.assertFalse(_is_forgotten(0.70, 0.20, confirmed=False))

    def test_periodic_checkpoint_candidates_are_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "final.pt"
            periodic = Path(temporary) / "final_checkpoints"
            periodic.mkdir()
            for steps in (1_100_000, 1_200_000, 1_150_000):
                (periodic / f"step_{steps:012d}.pt").touch()
            self.assertEqual(
                [path.name for path in _periodic_checkpoint_candidates(checkpoint)],
                [
                    "step_000001200000.pt",
                    "step_000001150000.pt",
                    "step_000001100000.pt",
                ],
            )

    def test_population_summary_reports_registered_metrics(self) -> None:
        matrix = build_generation_zero_active_matrix()
        validation = {
            "lineages": [
                {"score_rate": 0.48 + index * 0.01}
                for index in range(6)
            ]
        }
        summary = _build_population_summary(
            active_payoff=matrix,
            validation=validation,
            bootstrap_replicates=5,
        )
        self.assertEqual(len(summary["per_model"]), 6)
        self.assertEqual(len(summary["worst_class_cells"]), 12)
        self.assertEqual(
            summary["paired_bootstrap"]["replicates"],
            5,
        )
        self.assertIn("exploitability_proxy", summary["nash_mixture"])
        self.assertGreaterEqual(
            summary["payoff_profile_diversity"]["mean"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
