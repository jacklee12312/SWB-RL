from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from swb.engine.abilities import (
    ABILITY_DEFINITIONS,
    ABILITY_NAME_MAP,
)
from swb.rules import KEYWORD_RE, clean_printed_text, parse_fanfare


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SUPPORTED_KEYWORDS = {"守护", "疾驰", "突进"}

LOCALIZED_LOOKUPS = {
    "class": {
        0: {"chs": "中立", "cht": "中立", "eng": "Neutral", "jpn": "ニュートラル", "kor": "중립"},
        1: {"chs": "精灵", "cht": "精靈", "eng": "Forestcraft", "jpn": "エルフ", "kor": "엘프"},
        2: {"chs": "皇家护卫", "cht": "皇家護衛", "eng": "Swordcraft", "jpn": "ロイヤル", "kor": "로얄"},
        3: {"chs": "巫师", "cht": "巫師", "eng": "Runecraft", "jpn": "ウィッチ", "kor": "위치"},
        4: {"chs": "龙族", "cht": "龍族", "eng": "Dragoncraft", "jpn": "ドラゴン", "kor": "드래곤"},
        5: {"chs": "梦魇", "cht": "夢魘", "eng": "Abysscraft", "jpn": "ナイトメア", "kor": "나이트메어"},
        6: {"chs": "主教", "cht": "主教", "eng": "Havencraft", "jpn": "ビショップ", "kor": "비숍"},
        7: {"chs": "超越者", "cht": "超越者", "eng": "Portalcraft", "jpn": "ネメシス", "kor": "네메시스"},
    },
    "rarity": {
        1: {"chs": "铜", "cht": "銅", "eng": "Bronze", "jpn": "ブロンズ", "kor": "브론즈"},
        2: {"chs": "银", "cht": "銀", "eng": "Silver", "jpn": "シルバー", "kor": "실버"},
        3: {"chs": "金", "cht": "金", "eng": "Gold", "jpn": "ゴールド", "kor": "골드"},
        4: {"chs": "虹", "cht": "虹", "eng": "Legendary", "jpn": "レジェンド", "kor": "레전드"},
    },
    "type": {
        1: {"chs": "随从", "cht": "隨從", "eng": "Follower", "jpn": "フォロワー", "kor": "추종자"},
        2: {"chs": "护符", "cht": "護符", "eng": "Amulet", "jpn": "アミュレット", "kor": "부적"},
        3: {"chs": "法术", "cht": "法術", "eng": "Spell", "jpn": "スペル", "kor": "마법"},
    },
}


def _localized_lookup(card: dict[str, Any], prefix: str, suffix: str) -> str:
    value = card.get(f"{prefix}_name_{suffix}")
    if value:
        return value
    if suffix == "chs":
        value = card.get(f"{prefix}_name")
        if value:
            return value
    item_id = card[prefix]
    return LOCALIZED_LOOKUPS.get(prefix, {}).get(item_id, {}).get(suffix, "")


def _localized_primary(card: dict[str, Any], prefix: str) -> str:
    return _localized_lookup(card, prefix, "chs")


def _flavor_texts(card: dict[str, Any]) -> list[dict[str, str]]:
    existing = card.get("flavor_texts")
    if isinstance(existing, list):
        return existing
    keys = {
        "text_chs": card.get("flavor_chs", ""),
        "text_cht": card.get("flavor_cht", ""),
        "text_eng": card.get("flavor_eng", ""),
        "text_jpn": card.get("flavor_jpn", ""),
        "text_kor": card.get("flavor_kor", ""),
    }
    if not any(keys.values()):
        return []
    return [{"key": f"FT_{card['card_id']}_01", **keys}]


def _voice_data(card: dict[str, Any]) -> dict[str, Any]:
    voices = card.get("voices")
    if isinstance(voices, dict):
        return voices
    keys = {
        suffix: card.get(field, "")
        for suffix, field in (
            ("chs", "cv_chs"),
            ("cht", "cv_cht"),
            ("eng", "cv_eng"),
            ("jpn", "cv_jpn"),
            ("kor", "cv_kor"),
            ("pinyin", "cv_pinyin"),
            ("romaji", "cv_romaji"),
            ("engcv", "engcv"),
        )
        if card.get(field)
    }
    return keys


def extract_raw_keywords(skill_texts: list[str]) -> list[str]:
    joined = "\n".join(skill_texts)
    keywords = set(KEYWORD_RE.findall(joined))
    if "纹章：" in joined or "纹章:" in joined:
        keywords.add("纹章")
    if "信仰值" in joined or "自己的信仰" in joined:
        keywords.add("信仰")
    return sorted(keywords)


def classify_support(card: dict[str, Any]) -> tuple[str, list[str], str]:
    texts = [
        entry.get("text_chs", entry.get("text", ""))
        for entry in card.get("skill_texts", [])
    ]
    joined = "\n".join(texts)
    keywords = extract_raw_keywords(texts)
    clean = clean_printed_text(joined)
    fanfare_effects, fanfare_supported = parse_fanfare(texts)

    if not clean:
        return "basic", [], "No printed ability."
    if fanfare_supported and set(keywords).issubset(SUPPORTED_KEYWORDS | {"入场曲"}):
        return (
            "keyword",
            keywords,
            f"Supported fanfare with {len(fanfare_effects)} effect(s).",
        )
    if keywords and set(keywords).issubset(SUPPORTED_KEYWORDS):
        remaining = clean
        for keyword in keywords:
            remaining = remaining.replace(f"【{keyword}】", "")
        if not remaining.strip():
            return "keyword", keywords, "Supported keyword-only card."
    return "unsupported", keywords, "Printed effects are preserved but not executed."


def import_cards(source: Path, database: Path) -> int:
    cards = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(cards, list):
        raise ValueError("Expected a top-level JSON array")

    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        card_set_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(card_sets)")
        }
        if "is_collectible" not in card_set_columns:
            connection.execute(
                "ALTER TABLE card_sets ADD COLUMN is_collectible INTEGER NOT NULL DEFAULT 1"
            )
        required_columns = {
            "cards": {
                "tribe_id": "INTEGER NOT NULL DEFAULT 0",
                "tribe_name": "TEXT NOT NULL DEFAULT ''",
                "name_pinyin": "TEXT NOT NULL DEFAULT ''",
                "name_romaji": "TEXT NOT NULL DEFAULT ''",
            },
            "skill_texts": {
                "text_chs": "TEXT NOT NULL DEFAULT ''",
                "text_cht": "TEXT NOT NULL DEFAULT ''",
                "text_eng": "TEXT NOT NULL DEFAULT ''",
                "text_jpn": "TEXT NOT NULL DEFAULT ''",
                "text_kor": "TEXT NOT NULL DEFAULT ''",
            },
        }
        for table, columns in required_columns.items():
            existing = {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for column, declaration in columns.items():
                if column not in existing:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                    )
        with connection:
            for table in (
                "card_abilities",
                "abilities",
                "rule_support",
                "card_extra_data",
                "card_references",
                "alt_modes",
                "flavor_texts",
                "card_localizations",
                "textures",
                "skill_texts",
                "skills",
                "card_names",
                "cards",
                "card_sets",
                "classes",
                "rarities",
                "card_types",
            ):
                connection.execute(f"DELETE FROM {table}")

            for definition in ABILITY_DEFINITIONS:
                connection.execute(
                    """
                    INSERT INTO abilities(keyword, status, events, aliases)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        definition.keyword.value,
                        definition.status.value,
                        json.dumps(
                            sorted(event.value for event in definition.events),
                            ensure_ascii=False,
                        ),
                        json.dumps(definition.aliases, ensure_ascii=False),
                    ),
                )

            for card in cards:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO card_sets(id, is_collectible)
                    VALUES (?, ?)
                    """,
                    (card["card_set_id"], int(card["card_set_id"] != 90000)),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO classes(id, name) VALUES (?, ?)",
                    (card["class"], _localized_primary(card, "class")),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO rarities(id, name) VALUES (?, ?)",
                    (card["rarity"], _localized_primary(card, "rarity")),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO card_types(id, name) VALUES (?, ?)",
                    (card["type"], _localized_primary(card, "type")),
                )
                connection.execute(
                    """
                    INSERT INTO cards(
                        card_id, base_card_id, card_set_id, class_id, rarity_id,
                        type_id, cost, attack, life, is_evolution, evolves_to,
                        tribe_id, tribe_name, name_pinyin, name_romaji, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card["card_id"],
                        card["base_card_id"],
                        card["card_set_id"],
                        card["class"],
                        card["rarity"],
                        card["type"],
                        card["cost"],
                        card["atk"],
                        card["life"],
                        int(card["is_evolution"]),
                        card.get("evolves_to"),
                        card.get("tribe", 0),
                        card.get("tribe_name", ""),
                        card.get("name_pinyin", ""),
                        card.get("name_romaji", ""),
                        json.dumps(card, ensure_ascii=False, separators=(",", ":")),
                    ),
                )
                for language, field in (
                    ("zh-CN", "name_chs"),
                    ("zh-TW", "name_cht"),
                    ("en", "name_eng"),
                    ("ja", "name_jpn"),
                    ("ko", "name_kor"),
                ):
                    connection.execute(
                        "INSERT INTO card_names(card_id, language, name) VALUES (?, ?, ?)",
                        (card["card_id"], language, card[field]),
                    )
                for position, skill in enumerate(card.get("skills", [])):
                    connection.execute(
                        """
                        INSERT INTO skills(card_id, position, skill_id, type, subtype)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            card["card_id"],
                            position,
                            skill["skill_id"],
                            skill["type"],
                            skill["subtype"],
                        ),
                    )
                for position, text in enumerate(card.get("skill_texts", [])):
                    text_chs = text.get("text_chs", text.get("text", ""))
                    connection.execute(
                        """
                        INSERT INTO skill_texts(
                            card_id, position, text_key, text, text_chs, text_cht,
                            text_eng, text_jpn, text_kor
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            card["card_id"],
                            position,
                            text["key"],
                            text_chs,
                            text_chs,
                            text.get("text_cht", ""),
                            text.get("text_eng", ""),
                            text.get("text_jpn", ""),
                            text.get("text_kor", ""),
                        ),
                    )
                for language, suffix in (
                    ("zh-CN", "chs"),
                    ("zh-TW", "cht"),
                    ("en", "eng"),
                    ("ja", "jpn"),
                    ("ko", "kor"),
                ):
                    connection.execute(
                        """
                        INSERT INTO card_localizations(
                            card_id, language, class_name, rarity_name,
                            type_name, tribe_name
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            card["card_id"],
                            language,
                            _localized_lookup(card, "class", suffix),
                            _localized_lookup(card, "rarity", suffix),
                            _localized_lookup(card, "type", suffix),
                            card.get(f"tribe_name_{suffix}", card.get("tribe_name", "")),
                        ),
                    )
                for position, text in enumerate(_flavor_texts(card)):
                    connection.execute(
                        """
                        INSERT INTO flavor_texts(
                            card_id, position, text_key, text_chs, text_cht,
                            text_eng, text_jpn, text_kor
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            card["card_id"],
                            position,
                            text["key"],
                            text.get("text_chs", ""),
                            text.get("text_cht", ""),
                            text.get("text_eng", ""),
                            text.get("text_jpn", ""),
                            text.get("text_kor", ""),
                        ),
                    )
                for position, mode in enumerate(card.get("alt_modes", [])):
                    connection.execute(
                        """
                        INSERT INTO alt_modes(
                            card_id, position, mode_type, cost, text_chs,
                            text_cht, text_eng, text_jpn, text_kor
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            card["card_id"],
                            position,
                            mode.get("type", ""),
                            mode.get("cost"),
                            mode.get("text_chs", ""),
                            mode.get("text_cht", ""),
                            mode.get("text_eng", ""),
                            mode.get("text_jpn", ""),
                            mode.get("text_kor", ""),
                        ),
                    )
                for position, reference in enumerate(card.get("references", [])):
                    connection.execute(
                        """
                        INSERT INTO card_references(
                            card_id, position, referenced_card_id, referenced_name
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            card["card_id"],
                            position,
                            reference.get("card_id"),
                            reference.get("name", ""),
                        ),
                    )
                connection.execute(
                    """
                    INSERT INTO card_extra_data(
                        card_id, skin_names, voices, voice_variants
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        card["card_id"],
                        json.dumps(card.get("skin_names", {}), ensure_ascii=False),
                        json.dumps(_voice_data(card), ensure_ascii=False),
                        json.dumps(
                            card.get("voice_variants") or {}, ensure_ascii=False
                        ),
                    ),
                )
                for variant, path in card.get("textures", {}).items():
                    connection.execute(
                        "INSERT INTO textures(card_id, variant, path) VALUES (?, ?, ?)",
                        (card["card_id"], variant, path),
                    )
                level, keywords, notes = classify_support(card)
                connection.execute(
                    """
                    INSERT INTO rule_support(card_id, support_level, keywords, notes)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        card["card_id"],
                        level,
                        json.dumps(keywords, ensure_ascii=False),
                        notes,
                    ),
                )
                normalized: dict[str, str] = {}
                for raw_keyword in keywords:
                    ability = ABILITY_NAME_MAP.get(raw_keyword)
                    if ability is not None:
                        normalized.setdefault(ability.value, raw_keyword)
                connection.executemany(
                    """
                    INSERT INTO card_abilities(
                        card_id, ability_keyword, raw_keyword
                    ) VALUES (?, ?, ?)
                    """,
                    (
                        (card["card_id"], ability_keyword, raw_keyword)
                        for ability_keyword, raw_keyword in normalized.items()
                    ),
                )
        return len(cards)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import card JSON into SQLite")
    parser.add_argument("--source", type=Path, default=ROOT / "shadowverse_cards.json")
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "cards.sqlite3")
    args = parser.parse_args()
    count = import_cards(args.source, args.database)
    print(f"Imported {count} cards into {args.database}")


if __name__ == "__main__":
    main()
