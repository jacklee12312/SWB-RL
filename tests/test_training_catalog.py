from __future__ import annotations

import json
import pickle
import random
import sqlite3
import tempfile
import time
import tracemalloc
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from swb.db.repository import CardRepository
from swb.engine.deck import DECK_SIZE, validate_deck
from swb.engine.triggers import Trigger
from swb.rl.catalog import TrainableCardCatalog
from swb.rl.runtime import WorkerAssetsSnapshot, hash_rule_directory


DATABASE = Path("data/cards.sqlite3")


@unittest.skipUnless(DATABASE.exists(), "real card database is unavailable")
class TrainableCardCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = CardRepository(DATABASE)
        cls.catalog = TrainableCardCatalog.from_repository(cls.repository)

    def test_pool_is_exact_collectible_and_includes_all_card_types(self) -> None:
        pool = self.catalog.pool()
        report = json.loads(
            Path("data/reports/rule_coverage.json").read_text(encoding="utf-8")
        )
        expected_ids = {
            int(card_id)
            for card_id, classification in report["classifications"].items()
            if classification["coverage"] == "covered_exact"
            and classification["is_collectible"]
        }
        self.assertEqual({card.card_id for card in pool}, expected_ids)
        self.assertEqual({card.card_type for card in pool}, {"随从", "法术", "护符"})
        self.assertTrue(all(card.is_collectible for card in pool))

    def test_class_pool_contains_only_neutral_or_requested_class(self) -> None:
        for class_id in range(1, 8):
            pool = self.catalog.pool(class_id=class_id)
            self.assertTrue(pool)
            self.assertTrue(all(card.class_id in (0, class_id) for card in pool))

    def test_seeded_deck_sampling_is_reproducible_and_copy_bounded(self) -> None:
        first = self.catalog.sample_deck(6, random.Random(17))
        second = self.catalog.sample_deck(6, random.Random(17))
        self.assertEqual(
            [card.card_id for card in first],
            [card.card_id for card in second],
        )
        self.assertEqual(len(first), DECK_SIZE)
        self.assertLessEqual(max(Counter(card.card_id for card in first).values()), 3)
        validate_deck(first, class_id=6, player_index=0)

    def test_resolver_performs_no_sqlite_access_after_catalog_construction(self) -> None:
        card_id = self.catalog.exact_collectible_ids[0]
        with patch("sqlite3.connect", side_effect=AssertionError("hot-path SQL")):
            self.assertEqual(self.catalog.resolve(card_id).card_id, card_id)
            self.assertIsNone(self.catalog.resolve(-1))

    def test_all_cards_uses_one_sqlite_connection(self) -> None:
        with patch("sqlite3.connect", wraps=sqlite3.connect) as connect:
            cards = self.repository.all_cards()
        self.assertEqual(len(cards), len(self.catalog.cards_by_id))
        self.assertEqual(connect.call_count, 1)

    def test_stale_coverage_snapshot_is_rejected(self) -> None:
        report_path = Path("data/reports/rule_coverage.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["generated_from"]["source_snapshot"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            stale_report = Path(directory) / "rule_coverage.json"
            stale_report.write_text(
                json.dumps(report, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                TrainableCardCatalog.from_repository(
                    self.repository,
                    coverage_report=stale_report,
                )

    def test_catalog_is_pickleable_and_remains_immutable(self) -> None:
        restored = pickle.loads(pickle.dumps(self.catalog))
        self.assertEqual(restored.catalog_sha256, self.catalog.catalog_sha256)
        self.assertEqual(
            restored.card_vocabulary_sha256,
            self.catalog.card_vocabulary_sha256,
        )
        with self.assertRaises(TypeError):
            restored.cards_by_id[-1] = restored.cards_by_id[restored.card_vocabulary[0]]

    def test_worker_snapshot_load_performs_no_database_or_rule_io(self) -> None:
        snapshot = WorkerAssetsSnapshot.build(self.repository)
        restored = pickle.loads(pickle.dumps(snapshot))
        with (
            patch("sqlite3.connect", side_effect=AssertionError("worker SQL")),
            patch.object(Path, "read_bytes", side_effect=AssertionError("worker rule I/O")),
        ):
            assets = restored.load()
        self.assertEqual(assets.catalog.catalog_sha256, self.catalog.catalog_sha256)
        self.assertEqual(assets.rulebook_sha256, hash_rule_directory("data/rules"))
        self.assertTrue(assets.rulebook.operations_for(10031310, Trigger.PLAY))

    def test_worker_snapshot_startup_time_and_memory_are_bounded(self) -> None:
        tracemalloc.start()
        started = time.perf_counter()
        snapshot = WorkerAssetsSnapshot.build(self.repository)
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        serialized_bytes = len(pickle.dumps(snapshot))

        self.assertLess(elapsed, 30.0)
        self.assertLess(peak_bytes, 100 * 1024 * 1024)
        self.assertLess(serialized_bytes, 50 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
