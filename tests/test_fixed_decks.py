from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from swb.db.repository import CardRepository
from swb.rl.fixed_decks import (
    OFFICIAL_QR_EVOLVE_HAVEN,
    fixed_training_deck_names,
    get_fixed_training_deck,
)
from swb.rl.runtime import WorkerAssetsSnapshot


DATABASE = Path("data/cards.sqlite3")
IMPORTED_QR_DECKS = {
    "international_qr_forest_20260728": (1, "国际服二维码·精灵样本"),
    "international_qr_sword_20260728": (2, "国际服二维码·皇家护卫"),
    "international_qr_runecraft_20260728": (3, "国际服二维码·巫师"),
    "international_qr_dragon_20260728": (4, "国际服二维码·龙族"),
    "international_qr_nightmare_20260728": (5, "国际服二维码·梦魇"),
    "international_qr_portal_myuu_20260728": (
        7,
        "国际服二维码·超越者·米乌",
    ),
    "international_qr_portal_lishenna_20260728": (
        7,
        "国际服二维码·超越者·莉洁纳",
    ),
}

EXPECTED_COUNTS = {
    10403120: 2,
    10861110: 3,
    10863210: 3,
    10304120: 3,
    10461110: 2,
    10863110: 3,
    10362220: 3,
    10463210: 2,
    10564110: 3,
    10661110: 1,
    10864120: 3,
    10404110: 3,
    10663110: 1,
    10862120: 3,
    10864110: 3,
    10804120: 2,
}


class FixedTrainingDeckRegistryTests(unittest.TestCase):
    def test_official_qr_deck_is_named_and_immutable(self) -> None:
        self.assertIn(OFFICIAL_QR_EVOLVE_HAVEN, fixed_training_deck_names())
        deck = get_fixed_training_deck(OFFICIAL_QR_EVOLVE_HAVEN)
        self.assertEqual(deck.class_id, 6)
        self.assertEqual(len(deck.card_ids), 40)
        self.assertEqual(Counter(deck.card_ids), Counter(EXPECTED_COUNTS))
        self.assertEqual(len(deck.sha256), 64)
        self.assertEqual(deck.manifest()["sha256"], deck.sha256)

    def test_unknown_fixed_deck_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fixed training deck"):
            get_fixed_training_deck("missing")

    def test_trainable_qr_manifests_are_discovered(self) -> None:
        names = fixed_training_deck_names()
        self.assertTrue(set(IMPORTED_QR_DECKS).issubset(names))
        self.assertEqual(
            {get_fixed_training_deck(name).class_id for name in names},
            set(range(1, 8)),
        )
        for name, (class_id, display_name) in IMPORTED_QR_DECKS.items():
            with self.subTest(name=name):
                deck = get_fixed_training_deck(name)
                self.assertEqual(deck.class_id, class_id)
                self.assertEqual(len(deck.card_ids), 40)
                self.assertEqual(deck.display_name, display_name)


@unittest.skipUnless(DATABASE.exists(), "real card database is unavailable")
class FixedTrainingDeckCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.snapshot = WorkerAssetsSnapshot.build(CardRepository(DATABASE))

    def test_official_qr_deck_resolves_to_exact_collectible_cards(self) -> None:
        recipe = get_fixed_training_deck(OFFICIAL_QR_EVOLVE_HAVEN)
        deck = recipe.build(self.snapshot.catalog)
        self.assertEqual(len(deck), 40)
        self.assertEqual(Counter(card.card_id for card in deck), EXPECTED_COUNTS)
        self.assertTrue(all(card.class_id in (0, 6) for card in deck))
        self.assertTrue(all(card.is_collectible for card in deck))
        exact_ids = frozenset(self.snapshot.catalog.exact_collectible_ids)
        self.assertTrue(all(card.card_id in exact_ids for card in deck))

    def test_imported_qr_decks_resolve_to_exact_collectible_cards(self) -> None:
        exact_ids = frozenset(self.snapshot.catalog.exact_collectible_ids)
        for name, (class_id, _) in IMPORTED_QR_DECKS.items():
            with self.subTest(name=name):
                recipe = get_fixed_training_deck(name)
                deck = recipe.build(self.snapshot.catalog)
                self.assertEqual(len(deck), 40)
                self.assertTrue(
                    all(card.class_id in (0, class_id) for card in deck)
                )
                self.assertTrue(all(card.is_collectible for card in deck))
                self.assertTrue(
                    all(card.card_id in exact_ids for card in deck)
                )


if __name__ == "__main__":
    unittest.main()
