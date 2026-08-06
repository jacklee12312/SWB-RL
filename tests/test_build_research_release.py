from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_research_release import build_state_tree, split_by_size


class ResearchReleaseBuilderTests(unittest.TestCase):
    def test_split_by_size_is_deterministic_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, size in (("c.pt", 4), ("a.pt", 4), ("b.pt", 4)):
                (root / name).write_bytes(b"x" * size)

            groups = split_by_size(
                root,
                [Path("c.pt"), Path("a.pt"), Path("b.pt")],
                maximum_bytes=8,
            )

            self.assertEqual(
                groups,
                [[Path("a.pt"), Path("b.pt")], [Path("c.pt")]],
            )

    def test_split_by_size_rejects_oversized_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "large.pt").write_bytes(b"x" * 9)

            with self.assertRaisesRegex(ValueError, "exceeds asset limit"):
                split_by_size(
                    root,
                    [Path("large.pt")],
                    maximum_bytes=8,
                )

    def test_state_tree_includes_release_only_research_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            destination = Path(temporary) / "state"
            report_root = Path("data/reports/league_training/generations")
            generation = root / report_root / "generation_001"
            generation.mkdir(parents=True)
            (generation / "population_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            generation_zero = (
                root
                / "data/reports/league_training/"
                "generation_000_payoff_evaluations"
            )
            generation_zero.mkdir(parents=True)
            (generation_zero / "pair.json").write_text("{}\n", encoding="utf-8")

            sampler = (
                root
                / "data/reports/league_training/sampler_screen_20260804"
            )
            for directory in (
                "archive_baseline",
                "candidate_evaluations",
                "generation_000_active_matrix",
                "training",
            ):
                path = sampler / directory / "sample.json"
                path.parent.mkdir(parents=True)
                path.write_text("{}\n", encoding="utf-8")
            profiler = (
                root / "data/reports/training_speed/v4_1_profiler_trace.json.gz"
            )
            profiler.parent.mkdir(parents=True)
            profiler.write_bytes(b"trace")

            copied = build_state_tree(
                root,
                report_root,
                generation=1,
                destination=destination,
            )

            self.assertIn(
                Path(
                    "data/reports/league_training/sampler_screen_20260804/"
                    "generation_000_active_matrix/sample.json"
                ),
                copied,
            )
            self.assertIn(
                Path("data/reports/training_speed/v4_1_profiler_trace.json.gz"),
                copied,
            )
            self.assertTrue(
                (destination / "data/checkpoints/RELEASE_INSTALLED.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
