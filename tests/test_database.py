from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from swb.db.import_cards import import_cards
from swb.db.repository import CardRepository
from swb.engine.abilities import AbilityKeyword


ROOT = Path(__file__).resolve().parents[1]


class DatabaseTests(unittest.TestCase):
    def test_import_preserves_all_cards_and_relations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "cards.sqlite3"
            count = import_cards(ROOT / "shadowverse_cards.json", database)
            self.assertEqual(count, 740)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM cards").fetchone()[0], 740)
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM card_names").fetchone()[0],
                    740 * 5,
                )
                violations = connection.execute("PRAGMA foreign_key_check").fetchall()
                self.assertEqual(violations, [])
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM abilities").fetchone()[0],
                    34,
                )
                self.assertGreater(
                    connection.execute(
                        "SELECT COUNT(*) FROM card_abilities"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM card_localizations"
                    ).fetchone()[0],
                    740 * 5,
                )
                self.assertGreater(
                    connection.execute(
                        "SELECT COUNT(*) FROM flavor_texts"
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM cards
                        WHERE type_id = 1 AND attack IS NULL
                        """
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM card_abilities
                        WHERE ability_keyword IN ('妖精', '骑士', '铁甲骑士')
                        """
                    ).fetchone()[0],
                    0,
                )

            repository = CardRepository(database)
            card = repository.get(10001110)
            self.assertEqual(card.name, "不屈的剑斗士")
            self.assertEqual(card.cost, 2)
            self.assertEqual(card.card_type, "随从")
            self.assertEqual(card.class_name, "中立")
            self.assertTrue(card.is_collectible)
            token = repository.get(90044110)
            self.assertEqual(token.name, "霸道之金龙")
            self.assertFalse(token.is_collectible)
            pool = repository.training_pool()
            self.assertGreater(len(pool), 0)
            self.assertTrue(all(item.card_set_id != 90000 for item in pool))
            bishop_pool = repository.training_pool(class_id=6)
            self.assertTrue(
                all(item.class_id in {0, 6} for item in bishop_pool)
            )
            guard = repository.get(10001130)
            self.assertIn(AbilityKeyword.WARD, guard.abilities)
            self.assertTrue(repository.cards_with_ability(AbilityKeyword.WARD))
            bane_cards = repository.cards_with_ability(AbilityKeyword.BANE)
            self.assertTrue(bane_cards)
            self.assertTrue(
                all(AbilityKeyword.BANE in item.abilities for item in bane_cards)
            )
            emblem = repository.get(10114110)
            self.assertIn(AbilityKeyword.EMBLEM, emblem.abilities)
            faith = repository.get(10354110)
            self.assertIn(AbilityKeyword.FAITH, faith.abilities)


if __name__ == "__main__":
    unittest.main()
