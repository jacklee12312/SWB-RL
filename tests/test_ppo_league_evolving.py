from __future__ import annotations

import math
import unittest

from scripts.report_ppo_league_baseline import render_json
from scripts.report_ppo_league_evolving import (
    ACTIVE_FRACTION,
    ACTIVE_MATRIX_OUTPUT,
    ARCHIVE_FRACTION,
    ARCHIVE_SELECTION_OUTPUT,
    CONTRACT_OUTPUT,
    GENERATION_QUEUE_SCHEMA_OUTPUT,
    LINEAGE_MANIFEST_DIRECTORY,
    ROOT,
    SCHEDULE_OUTPUT,
    _artifacts,
    build_archive_selection_report,
    build_evolving_contract,
    build_generation_schedule,
    build_generation_zero_active_matrix,
    build_generation_zero_lineage_manifests,
)


class PPOLeagueEvolvingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_evolving_contract()
        cls.schedule = build_generation_schedule()
        cls.matrix = build_generation_zero_active_matrix()
        cls.archive = build_archive_selection_report()
        cls.manifests = build_generation_zero_lineage_manifests()

    def test_contract_and_schedule_freeze_six_lineages_to_three_million(self) -> None:
        self.assertEqual(self.contract["selected_sampler"], "hard")
        self.assertEqual(len(self.contract["lineages"]), 6)
        self.assertFalse(
            self.contract["generation_barrier"]["same_generation_visibility"]
        )
        self.assertTrue(
            self.contract["generation_barrier"][
                "publish_requires_all_six_lineages"
            ]
        )
        transitions = self.schedule["transitions"]
        self.assertEqual(len(transitions), 8)
        self.assertEqual(transitions[0]["nominal_millions"], 1.25)
        self.assertEqual(transitions[-1]["nominal_millions"], 3.0)
        self.assertEqual(
            self.schedule["budget"][
                "total_additional_agent_steps_to_generation_008"
            ],
            12_000_000,
        )

    def test_generation_zero_active_matrix_is_complete_and_antisymmetric(self) -> None:
        self.assertEqual(self.matrix["unique_pair_count"], 15)
        self.assertEqual(self.matrix["total_games"], 2_940)
        self.assertEqual(self.matrix["audit"]["truncated"], 0)
        ids = self.matrix["active_policy_ids"]
        for left in ids:
            for right in ids:
                self.assertTrue(math.isclose(
                    self.matrix["score_matrix"][left][right]
                    + self.matrix["score_matrix"][right][left],
                    1.0,
                    abs_tol=1e-12,
                ))

    def test_archive_and_lineage_manifests_use_hard_seventy_thirty_split(self) -> None:
        self.assertEqual(self.archive["audit"]["pair_count"], 108)
        self.assertEqual(self.archive["audit"]["total_games"], 21_168)
        self.assertEqual(len(self.archive["selected_archive_ids"]), 18)
        self.assertEqual(len(self.manifests), 6)
        for learner_id, manifest in self.manifests.items():
            with self.subTest(learner_id=learner_id):
                self.assertEqual(manifest["selection_mode"], "hard")
                active = [
                    entry for entry in manifest["entries"]
                    if entry["role"] == "active_latest"
                ]
                archive = [
                    entry for entry in manifest["entries"]
                    if entry["role"] == "historical_archive"
                ]
                anchors = [
                    entry for entry in manifest["entries"]
                    if entry["role"] == "evaluation_anchor"
                ]
                self.assertEqual((len(active), len(archive), len(anchors)), (6, 18, 3))
                self.assertTrue(math.isclose(
                    sum(entry["sampling_weight"] for entry in active),
                    ACTIVE_FRACTION,
                    abs_tol=1e-12,
                ))
                self.assertTrue(math.isclose(
                    sum(entry["sampling_weight"] for entry in archive),
                    ARCHIVE_FRACTION,
                    abs_tol=1e-12,
                ))
                self.assertEqual(
                    {entry["sampling_weight"] for entry in anchors}, {0.0}
                )
                self.assertLessEqual(manifest["summary"]["positive_entry_count"], 32)
                self.assertTrue(
                    manifest["selection_audit"]["all_trainable_entries_hit"]
                )

    def test_checked_in_artifacts_are_byte_stable(self) -> None:
        expected_paths = {
            CONTRACT_OUTPUT,
            SCHEDULE_OUTPUT,
            ACTIVE_MATRIX_OUTPUT,
            ARCHIVE_SELECTION_OUTPUT,
            GENERATION_QUEUE_SCHEMA_OUTPUT,
            *(
                LINEAGE_MANIFEST_DIRECTORY / f"{learner_id}.json"
                for learner_id in self.manifests
            ),
        }
        artifacts = _artifacts()
        self.assertEqual(set(artifacts), expected_paths)
        for path, expected in artifacts.items():
            self.assertEqual((ROOT / path).read_bytes(), expected)
        self.assertEqual(
            render_json(self.contract), render_json(build_evolving_contract())
        )


if __name__ == "__main__":
    unittest.main()
