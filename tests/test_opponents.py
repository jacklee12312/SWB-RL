from __future__ import annotations

import unittest

from swb.rl.opponents import OpponentPool


class OpponentPoolTests(unittest.TestCase):
    def make_pool(self) -> OpponentPool:
        return OpponentPool(
            42,
            current_weight=1.0,
            random_weight=1.0,
            fixed_weight=1.0,
            historical_weight=1.0,
            max_history=2,
            snapshot_interval_steps=100,
        )

    def test_selection_is_reproducible_by_episode_and_side(self) -> None:
        first = self.make_pool()
        second = self.make_pool()
        selected_first = [
            first.select(episode_id=episode, learner_player=episode % 2).opponent_id
            for episode in range(20)
        ]
        selected_second = [
            second.select(episode_id=episode, learner_player=episode % 2).opponent_id
            for episode in range(20)
        ]
        self.assertEqual(selected_first, selected_second)
        self.assertGreater(len(set(selected_first)), 1)

    def test_snapshot_retention_is_bounded_and_monotonic(self) -> None:
        pool = self.make_pool()
        self.assertTrue(pool.snapshot_due(100))
        pool.register_snapshot("one.pt", agent_steps=100)
        pool.register_snapshot("two.pt", agent_steps=200)
        pool.register_snapshot("three.pt", agent_steps=300)
        history = [entry for entry in pool.entries if entry.kind == "historical"]
        self.assertEqual(
            [entry.opponent_id for entry in history],
            ["historical_000000000200", "historical_000000000300"],
        )
        with self.assertRaisesRegex(ValueError, "increase monotonically"):
            pool.register_snapshot("stale.pt", agent_steps=300)

    def test_zero_weight_history_does_not_write_unused_snapshots(self) -> None:
        pool = OpponentPool(
            42,
            current_weight=1.0,
            historical_weight=0.0,
            snapshot_interval_steps=100,
        )
        self.assertFalse(pool.snapshot_due(100))
        self.assertFalse(pool.snapshot_due(1_000_000))

    def test_state_round_trip_preserves_selection_progress(self) -> None:
        pool = self.make_pool()
        pool.register_snapshot("one.pt", agent_steps=100)
        pool.select(episode_id=1, learner_player=1)
        restored = OpponentPool.from_state_dict(pool.state_dict())
        self.assertEqual(restored.state_dict(), pool.state_dict())
        self.assertEqual(sum(restored.selection_counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
