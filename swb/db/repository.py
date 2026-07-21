from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from swb.rules import EffectDefinition, parse_fanfare

if TYPE_CHECKING:
    from swb.engine.abilities import AbilityKeyword


@dataclass(frozen=True)
class CardDefinition:
    card_id: int
    card_set_id: int
    class_id: int
    class_name: str
    name: str
    cost: int
    card_type: str
    attack: int | None
    life: int | None
    keywords: frozenset[str]
    support_level: str
    is_collectible: bool
    fanfare_effects: tuple[EffectDefinition, ...] = ()
    ability_keywords: frozenset[AbilityKeyword] = frozenset()
    tribe_id: int = 0
    tribe_name: str = ""

    @property
    def abilities(self) -> frozenset[AbilityKeyword]:
        if self.ability_keywords:
            return self.ability_keywords
        from swb.engine.abilities import normalize_abilities

        return normalize_abilities(self.keywords)


class CardRepository:
    _CARD_SELECT = """
        SELECT c.card_id, c.card_set_id, c.class_id, cl.name,
               n.name, c.cost, t.name, c.attack, c.life,
               r.keywords, r.support_level, cs.is_collectible,
               (
                   SELECT GROUP_CONCAT(ordered.text, char(30))
                   FROM (
                       SELECT st.text
                       FROM skill_texts st
                       WHERE st.card_id = c.card_id
                       ORDER BY st.position
                   ) AS ordered
               ),
               c.tribe_id, c.tribe_name,
               (
                   SELECT GROUP_CONCAT(ordered.ability_keyword, char(30))
                   FROM (
                       SELECT ca.ability_keyword
                       FROM card_abilities ca
                       WHERE ca.card_id = c.card_id
                       ORDER BY ca.ability_keyword
                   ) AS ordered
               )
        FROM cards c
        JOIN card_sets cs ON cs.id = c.card_set_id
        JOIN classes cl ON cl.id = c.class_id
        JOIN card_names n ON n.card_id = c.card_id AND n.language = 'zh-CN'
        JOIN card_types t ON t.id = c.type_id
        JOIN rule_support r ON r.card_id = c.card_id
    """

    def __init__(self, database: str | Path):
        self.database = str(database)

    def get(self, card_id: int) -> CardDefinition:
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute(
                self._CARD_SELECT + " WHERE c.card_id = ?",
                (card_id,),
            ).fetchone()
        if row is None:
            raise KeyError(card_id)
        return self._card_from_row(row)

    @staticmethod
    def _card_from_row(row) -> CardDefinition:
        texts = row[12].split(chr(30)) if row[12] else []
        fanfare_effects, fanfare_supported = parse_fanfare(texts)
        from swb.engine.abilities import AbilityKeyword

        abilities = row[15].split(chr(30)) if row[15] else []

        return CardDefinition(
            card_id=row[0],
            card_set_id=row[1],
            class_id=row[2],
            class_name=row[3],
            name=row[4],
            cost=row[5],
            card_type=row[6],
            attack=row[7],
            life=row[8],
            keywords=frozenset(json.loads(row[9])),
            support_level=row[10],
            is_collectible=bool(row[11]),
            fanfare_effects=fanfare_effects if fanfare_supported else (),
            ability_keywords=frozenset(
                AbilityKeyword(ability) for ability in abilities
            ),
            tribe_id=row[13],
            tribe_name=row[14],
        )

    def cards_with_ability(
        self,
        ability: AbilityKeyword | str,
        *,
        collectible_only: bool = False,
    ) -> list[CardDefinition]:
        keyword = ability.value if hasattr(ability, "value") else str(ability)
        query = """
            SELECT ca.card_id
            FROM card_abilities ca
            JOIN cards c ON c.card_id = ca.card_id
            JOIN card_sets cs ON cs.id = c.card_set_id
            WHERE ca.ability_keyword = ?
        """
        if collectible_only:
            query += " AND cs.is_collectible = 1"
        query += " ORDER BY ca.card_id"
        with closing(sqlite3.connect(self.database)) as connection:
            ids = [row[0] for row in connection.execute(query, (keyword,))]
        return [self.get(card_id) for card_id in ids]

    def card_ids(self) -> tuple[int, ...]:
        """Return every database card ID in deterministic order."""
        with closing(sqlite3.connect(self.database)) as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT card_id FROM cards ORDER BY card_id"
                )
            )

    def all_cards(self) -> tuple[CardDefinition, ...]:
        """Load all definitions with one SQLite connection and one query."""
        with closing(sqlite3.connect(self.database)) as connection:
            rows = connection.execute(
                self._CARD_SELECT + " ORDER BY c.card_id"
            ).fetchall()
        return tuple(self._card_from_row(row) for row in rows)

    def source_snapshot(self) -> dict[str, object]:
        """Return metadata for the most recent database source import."""
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute(
                """
                SELECT source_url, fetched_at, sha256, card_count
                FROM source_imports
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return {}
        return {
            "source_url": row[0],
            "fetched_at": row[1],
            "sha256": row[2],
            "card_count": row[3],
        }

    def training_pool(
        self,
        limit: int | None = None,
        *,
        class_id: int | None = None,
    ) -> list[CardDefinition]:
        query = """
            SELECT c.card_id
            FROM cards c
            JOIN card_sets cs ON cs.id = c.card_set_id
            JOIN rule_support r ON r.card_id = c.card_id
            WHERE c.type_id = 1
              AND cs.is_collectible = 1
              AND c.attack IS NOT NULL
              AND c.life IS NOT NULL
              AND r.support_level IN ('basic', 'keyword')
        """
        params: tuple[int, ...] = ()
        if class_id is not None:
            query += " AND c.class_id IN (0, ?)"
            params = (class_id,)
        query += " ORDER BY c.cost, c.card_id"
        if limit is not None:
            query += " LIMIT ?"
            params += (limit,)
        with closing(sqlite3.connect(self.database)) as connection:
            ids = [row[0] for row in connection.execute(query, params)]
        return [self.get(card_id) for card_id in ids]
