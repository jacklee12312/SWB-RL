from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from swb.db.repository import CardRepository
from swb.rl.class_schedule import (
    ALL_CLASS_IDS,
    class_pair_for_episode,
    normalize_class_ids,
)
from swb.rl.distribution import build_training_distribution_audit
from swb.rl.runtime import WorkerAssetsSnapshot


DATABASE = Path("data/cards.sqlite3")


class ClassScheduleTests(unittest.TestCase):
    def test_round_robin_cycle_visits_every_ordered_matchup_once(self) -> None:
        pairs = [
            class_pair_for_episode(ALL_CLASS_IDS, episode_id)
            for episode_id in range(49)
        ]
        self.assertEqual(len(set(pairs)), 49)
        self.assertEqual(Counter(a for a, _ in pairs), Counter({cid: 7 for cid in ALL_CLASS_IDS}))
        self.assertEqual(Counter(b for _, b in pairs), Counter({cid: 7 for cid in ALL_CLASS_IDS}))
        self.assertEqual(
            class_pair_for_episode(ALL_CLASS_IDS, 49),
            class_pair_for_episode(ALL_CLASS_IDS, 0),
        )

    def test_invalid_class_schedules_are_rejected(self) -> None:
        for classes in ((), (0,), (1, 1), (8,)):
            with self.subTest(classes=classes):
                with self.assertRaises(ValueError):
                    normalize_class_ids(classes)
        with self.assertRaises(ValueError):
            class_pair_for_episode((1,), -1)


@unittest.skipUnless(DATABASE.exists(), "real card database is unavailable")
class TrainingDistributionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = WorkerAssetsSnapshot.build(CardRepository(DATABASE))

    def test_seven_class_two_cycle_audit_is_balanced_and_reproducible(self) -> None:
        kwargs = {
            "master_seed": 77,
            "episode_count": 98,
            "worker_count": 2,
            "class_ids": ALL_CLASS_IDS,
            "rulebook_sha256": self.snapshot.rulebook_sha256,
        }
        first = build_training_distribution_audit(self.snapshot.catalog, **kwargs)
        second = build_training_distribution_audit(self.snapshot.catalog, **kwargs)
        self.assertEqual(first, second)
        distribution = first["distribution"]
        self.assertEqual(set(distribution["class_pair_counts"].values()), {2})
        self.assertEqual(
            set(distribution["learner_class_counts"].values()),
            {14},
        )
        self.assertEqual(
            set(distribution["opponent_class_counts"].values()),
            {14},
        )
        self.assertEqual(
            set(distribution["card_type_slot_counts"]),
            {"随从", "法术", "护符"},
        )
        self.assertEqual(len(distribution["per_class"]), 7)
        self.assertGreater(distribution["unique_exact_cards_sampled"], 0)
        self.assertEqual(len(first["audit_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
