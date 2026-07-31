from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.profile_v4_1_inference_breakdown import (
    BATCH_SIZES,
    EPISODE_LENGTHS,
    LEGAL_ACTION_COUNTS,
    summarize_samples,
    validate_report,
)


class TrainingSpeedStage23Tests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    REPORT = (
        ROOT
        / "data/reports/training_speed/v4_1_inference_breakdown.json"
    )

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(cls.REPORT.read_text(encoding="utf-8"))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def test_summary_handles_empty_and_nearest_rank_p95(self) -> None:
        self.assertEqual(summarize_samples([])["sample_count"], 0)
        summary = summarize_samples([3.0, 1.0, 2.0])
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["p95"], 3.0)
        self.assertEqual(summary["samples"], [3.0, 1.0, 2.0])

    def test_saved_report_has_complete_fixed_input_comparison(self) -> None:
        validate_report(self.report)
        self.assertEqual(
            self.report["methodology"]["v3_6_scope"],
            "pure_forward_reference_only",
        )
        self.assertNotIn("v3_6_component_profile", self.report)
        expected = {str(value) for value in BATCH_SIZES}
        for version in ("v4.1", "v3.6"):
            batches = self.report["fixed_input_forward"][version]
            self.assertEqual(set(batches), expected)
            for batch in batches.values():
                self.assertEqual(
                    batch["device_milliseconds_per_call"]["sample_count"],
                    3,
                )
                self.assertGreater(batch["samples_per_second"], 0.0)

    def test_checkpoint_and_profiler_artifacts_are_integrity_checked(self) -> None:
        checkpoints = self.report["checkpoints"]
        self.assertEqual(
            checkpoints["v4.1"]["observation_schema"],
            "observation-v4.1",
        )
        self.assertEqual(
            checkpoints["v3.6"]["observation_schema"],
            "observation-v3.6",
        )
        for checkpoint in checkpoints.values():
            path = self.ROOT / checkpoint["path"]
            self.assertEqual(self._sha256(path), checkpoint["sha256"])

        profiler = self.report["profiler"]
        trace_path = self.ROOT / profiler["compressed_trace_path"]
        self.assertTrue(trace_path.is_file())
        self.assertEqual(
            self._sha256(trace_path),
            profiler["compressed_trace_sha256"],
        )
        self.assertGreater(profiler["trace"]["kernel_event_count"], 0)
        self.assertGreater(
            profiler["trace"]["kernel_launch_event_count"],
            0,
        )
        self.assertGreater(
            profiler["trace"]["synchronization_event_count"],
            0,
        )
        self.assertGreater(len(profiler["trace"]["top_kernels"]), 0)

    def test_v4_1_diagnostics_cover_every_refined_checklist_dimension(
        self,
    ) -> None:
        component_fields = {
            "structured_token_construction_milliseconds",
            "card_embedding_lookup_milliseconds",
            "card_projection_milliseconds",
            "non_card_numeric_projection_milliseconds",
            "transformer_milliseconds",
            "gru_milliseconds",
            "action_value_stage_milliseconds",
            "policy_head_milliseconds",
            "value_head_milliseconds",
        }
        for batch in self.report["v4_1_component_profile"].values():
            self.assertTrue(
                component_fields.issubset(batch["fields"])
            )
            self.assertGreater(
                batch["fields"]["policy_head_milliseconds"]["median"],
                0.0,
            )
            self.assertGreater(
                batch["fields"]["value_head_milliseconds"]["median"],
                0.0,
            )

        packing = self.report["v4_1_input_packing"]
        self.assertEqual(set(packing), {str(value) for value in BATCH_SIZES})
        per_request = {
            item["input_bytes_per_request"]
            for item in packing.values()
        }
        self.assertEqual(len(per_request), 1)
        self.assertEqual(
            {
                int(value)
                for value in self.report[
                    "v4_1_recurrent_episode_lengths"
                ]
            },
            set(EPISODE_LENGTHS),
        )
        legal = self.report["v4_1_legal_action_counts"]
        self.assertTrue(legal["dense_action_scoring_is_mask_independent"])
        self.assertEqual(
            {
                int(value)
                for value in legal[
                    "distribution_by_legal_action_count"
                ]
            },
            set(LEGAL_ACTION_COUNTS),
        )

    def test_duplicate_audit_and_ranking_are_explicit(self) -> None:
        audit = self.report["duplicate_work_audit"]
        self.assertEqual(
            audit["observation_conversion"]["status"],
            "confirmed_duplicate_candidate",
        )
        self.assertEqual(
            audit["card_embedding"]["status"],
            "no_duplicate_within_forward",
        )
        self.assertEqual(
            audit["device_validation_sync"]["status"],
            "confirmed_host_sync_candidate",
        )
        ranking = self.report["bottleneck_ranking"]
        self.assertGreaterEqual(len(ranking), 3)
        self.assertEqual(
            [item["rank"] for item in ranking],
            list(range(1, len(ranking) + 1)),
        )
        self.assertTrue(
            all(item["target_stage"].startswith("2.") for item in ranking)
        )


if __name__ == "__main__":
    unittest.main()
