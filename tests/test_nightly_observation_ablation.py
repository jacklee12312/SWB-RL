from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.nightly_observation_ablation import (
    CANDIDATES,
    NightlyRun,
    _wilson,
)


class NightlyObservationAblationTests(unittest.TestCase):
    def test_candidate_matrix_keeps_model_and_reward_constants(self) -> None:
        self.assertEqual(len(CANDIDATES), 6)
        self.assertEqual(
            {candidate.entropy_coefficient for candidate in CANDIDATES},
            {0.01},
        )
        self.assertEqual(
            {candidate.clip_ratio for candidate in CANDIDATES},
            {0.2},
        )
        self.assertEqual(
            {candidate.learning_rate for candidate in CANDIDATES},
            {1e-4, 2e-4, 3e-4},
        )

    def test_wilson_interval_contains_observed_rate(self) -> None:
        lower, upper = _wilson(64, 200)
        self.assertLess(lower, 0.32)
        self.assertGreater(upper, 0.32)

    def test_initial_training_command_has_reproducible_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.pt"
            reference.touch()
            args = Namespace(
                deadline="2026-07-29T08:20:00",
                output_root=root / "reports",
                checkpoint_root=root / "checkpoints",
                reference_checkpoint=reference,
                pilot_steps=40_000,
                finalist_steps=120_000,
                final_steps=500_000,
                device="cuda",
                python=Path("python"),
            )
            run = NightlyRun(args)
            command = list(run.initial_train_arguments(
                observation="v4.1",
                seed=123,
                steps=40_000,
                checkpoint=root / "candidate.pt",
                report=root / "candidate.json",
                candidate=CANDIDATES[0],
            ))
            self.assertIn("--training-deck", command)
            self.assertIn("--opponent-decks", command)
            self.assertEqual(
                command[command.index("--master-seed") + 1],
                123,
            )
            self.assertEqual(
                command[command.index("--opponent-current-weight") + 1],
                1,
            )


if __name__ == "__main__":
    unittest.main()
