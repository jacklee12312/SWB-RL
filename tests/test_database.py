from __future__ import annotations

import json
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
        expected_count = len(
            json.loads((ROOT / "shadowverse_cards.json").read_text(encoding="utf-8"))
        )
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "cards.sqlite3"
            count = import_cards(ROOT / "shadowverse_cards.json", database)
            self.assertEqual(count, expected_count)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM cards").fetchone()[0],
                    expected_count,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM card_names").fetchone()[0],
                    expected_count * 5,
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
                    expected_count * 5,
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
            fairy = repository.get(90011110)
            self.assertEqual((fairy.tribe_id, fairy.tribe_name), (5, "妖精"))
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
            alt_mode_only_faith = repository.get(10634120)
            self.assertIn(AbilityKeyword.FAITH, alt_mode_only_faith.abilities)

    def test_import_accepts_compact_sva_records(self) -> None:
        card = {
            "card_id": 99999910,
            "base_card_id": 99999910,
            "card_set_id": 10008,
            "class": 1,
            "rarity": 2,
            "type": 1,
            "cost": 2,
            "atk": 2,
            "life": 2,
            "is_evolution": False,
            "evolves_to": None,
            "tribe": 0,
            "tribe_name": "",
            "name_chs": "紧凑测试随从",
            "name_cht": "緊湊測試隨從",
            "name_eng": "Compact Test Follower",
            "name_jpn": "コンパクトテストフォロワー",
            "name_kor": "컴팩트 테스트 추종자",
            "name_pinyin": "jincouceshisuicong",
            "name_romaji": "konpakutotesutoforowa",
            "skills": [],
            "skill_texts": [],
            "alt_modes": [],
            "references": [],
            "textures": {},
            "skin_names": {},
            "flavor_chs": "新版数据使用扁平 flavor 字段。",
            "flavor_cht": "新版資料使用扁平 flavor 欄位。",
            "flavor_eng": "New data uses flat flavor fields.",
            "flavor_jpn": "新しいデータは平坦な flavor フィールドを使う。",
            "flavor_kor": "새 데이터는 평면 flavor 필드를 사용한다.",
            "cv_chs": "测试声优",
            "cv_cht": "測試聲優",
            "cv_eng": "Test Voice",
            "cv_jpn": "テスト声優",
            "cv_kor": "테스트 성우",
            "voice_variants": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cards.json"
            database = Path(directory) / "cards.sqlite3"
            source.write_text(json.dumps([card], ensure_ascii=False), encoding="utf-8")
            self.assertEqual(import_cards(source, database), 1)
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(
                    connection.execute("SELECT name FROM classes WHERE id = 1").fetchone()[0],
                    "精灵",
                )
                self.assertEqual(
                    connection.execute("SELECT name FROM rarities WHERE id = 2").fetchone()[0],
                    "银",
                )
                self.assertEqual(
                    connection.execute("SELECT name FROM card_types WHERE id = 1").fetchone()[0],
                    "随从",
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT text_chs FROM flavor_texts WHERE card_id = 99999910"
                    ).fetchone()[0],
                    "新版数据使用扁平 flavor 字段。",
                )
                voices = json.loads(
                    connection.execute(
                        "SELECT voices FROM card_extra_data WHERE card_id = 99999910"
                    ).fetchone()[0]
                )
                self.assertEqual(voices["chs"], "测试声优")


if __name__ == "__main__":
    unittest.main()
