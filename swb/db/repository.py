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
    def __init__(self, database: str | Path):
        self.database = str(database)

    def get(self, card_id: int) -> CardDefinition:
        with closing(sqlite3.connect(self.database)) as connection:
            row = connection.execute(
                """
                SELECT c.card_id, c.card_set_id, c.class_id, cl.name,
                       n.name, c.cost, t.name, c.attack, c.life,
                       r.keywords, r.support_level, cs.is_collectible,
                       GROUP_CONCAT(st.text, char(30)),
                       c.tribe_id, c.tribe_name
                FROM cards c
                JOIN card_sets cs ON cs.id = c.card_set_id
                JOIN classes cl ON cl.id = c.class_id
                JOIN card_names n ON n.card_id = c.card_id AND n.language = 'zh-CN'
                JOIN card_types t ON t.id = c.type_id
                JOIN rule_support r ON r.card_id = c.card_id
                LEFT JOIN skill_texts st ON st.card_id = c.card_id
                WHERE c.card_id = ?
                GROUP BY c.card_id
                """,
                (card_id,),
            ).fetchone()
            ability_rows = connection.execute(
                """
                SELECT ability_keyword
                FROM card_abilities
                WHERE card_id = ?
                ORDER BY ability_keyword
                """,
                (card_id,),
            ).fetchall()
        if row is None:
            raise KeyError(card_id)
        texts = row[12].split(chr(30)) if row[12] else []
        fanfare_effects, fanfare_supported = parse_fanfare(texts)
        from swb.engine.abilities import AbilityKeyword

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
                AbilityKeyword(ability_row[0]) for ability_row in ability_rows
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
