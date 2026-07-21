from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.deck import DECK_SIZE


DEFAULT_COVERAGE_REPORT = (
    Path(__file__).resolve().parents[2] / "data" / "reports" / "rule_coverage.json"
)


@dataclass(frozen=True)
class TrainableCardCatalog:
    """Frozen card snapshot whose trainable subset passed the exact-rule audit.

    Database access is confined to construction. Matches use ``resolve`` and
    ``pool`` against immutable in-memory data, so workers do not query SQLite
    while resolving effects.
    """

    cards_by_id: Mapping[int, CardDefinition]
    exact_collectible_ids: tuple[int, ...]
    coverage_report_sha256: str
    source_snapshot: Mapping[str, object]
    catalog_sha256: str
    card_vocabulary_sha256: str
    training_pool_sha256: str

    @classmethod
    def from_repository(
        cls,
        repository: CardRepository,
        *,
        coverage_report: str | Path = DEFAULT_COVERAGE_REPORT,
    ) -> TrainableCardCatalog:
        report_path = Path(coverage_report)
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes.decode("utf-8"))
        classifications = report.get("classifications")
        if not isinstance(classifications, dict):
            raise ValueError("Coverage report has no classifications mapping")

        cards = {card.card_id: card for card in repository.all_cards()}
        exact_ids = tuple(sorted(
            int(card_id)
            for card_id, classification in classifications.items()
            if classification.get("coverage") == "covered_exact"
            and classification.get("is_collectible") is True
        ))
        missing = [card_id for card_id in exact_ids if card_id not in cards]
        if missing:
            raise ValueError(
                "Coverage report references cards absent from the database: "
                f"{missing[:5]}"
            )
        non_collectible = [
            card_id for card_id in exact_ids if not cards[card_id].is_collectible
        ]
        if non_collectible:
            raise ValueError(
                "Coverage report marks non-collectible cards as trainable: "
                f"{non_collectible[:5]}"
            )
        if not exact_ids:
            raise ValueError("Coverage report contains no exact collectible cards")

        generated_from = report.get("generated_from", {})
        source_snapshot = generated_from.get("source_snapshot", {})
        if not isinstance(source_snapshot, dict):
            source_snapshot = {}
        expected_count = source_snapshot.get("card_count")
        if expected_count is not None and expected_count != len(cards):
            raise ValueError(
                "Coverage report/database snapshot mismatch: "
                f"report={expected_count}, database={len(cards)}"
            )
        database_snapshot = repository.source_snapshot()
        for field in ("sha256", "card_count"):
            expected = source_snapshot.get(field)
            actual = database_snapshot.get(field)
            if expected is not None and actual is not None and expected != actual:
                raise ValueError(
                    "Coverage report/database source mismatch for "
                    f"{field}: report={expected!r}, database={actual!r}"
                )

        coverage_sha256 = hashlib.sha256(report_bytes).hexdigest()
        vocabulary_bytes = json.dumps(
            sorted(cards), separators=(",", ":")
        ).encode("ascii")
        training_pool_bytes = json.dumps(
            exact_ids, separators=(",", ":")
        ).encode("ascii")
        catalog_payload = {
            "source": source_snapshot,
            "coverage_sha256": coverage_sha256,
            "cards": [
                {
                    "card_id": card.card_id,
                    "card_set_id": card.card_set_id,
                    "class_id": card.class_id,
                    "name": card.name,
                    "cost": card.cost,
                    "card_type": card.card_type,
                    "attack": card.attack,
                    "life": card.life,
                    "keywords": sorted(card.keywords),
                    "abilities": sorted(ability.value for ability in card.abilities),
                    "support_level": card.support_level,
                    "is_collectible": card.is_collectible,
                    "tribe_id": card.tribe_id,
                    "tribe_name": card.tribe_name,
                }
                for card in cards.values()
            ],
        }
        catalog_bytes = json.dumps(
            catalog_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            cards_by_id=MappingProxyType(cards),
            exact_collectible_ids=exact_ids,
            coverage_report_sha256=coverage_sha256,
            source_snapshot=MappingProxyType(dict(source_snapshot)),
            catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
            card_vocabulary_sha256=hashlib.sha256(vocabulary_bytes).hexdigest(),
            training_pool_sha256=hashlib.sha256(training_pool_bytes).hexdigest(),
        )

    def __reduce__(self):
        """Keep the immutable mapping facade while supporting spawn pickling."""
        return (
            _restore_catalog,
            (
                tuple(self.cards_by_id.values()),
                self.exact_collectible_ids,
                self.coverage_report_sha256,
                tuple(self.source_snapshot.items()),
                self.catalog_sha256,
                self.card_vocabulary_sha256,
                self.training_pool_sha256,
            ),
        )

    def resolve(self, card_id: int) -> CardDefinition | None:
        """Resolve any collectible or generated card without database I/O."""
        return self.cards_by_id.get(card_id)

    @property
    def card_vocabulary(self) -> tuple[int, ...]:
        return tuple(self.cards_by_id)

    def pool(
        self,
        *,
        class_id: int | None = None,
        limit: int | None = None,
    ) -> tuple[CardDefinition, ...]:
        if class_id is not None and class_id not in range(1, 8):
            raise ValueError(f"class_id must be in 1..7, got {class_id}")
        cards = tuple(
            self.cards_by_id[card_id]
            for card_id in self.exact_collectible_ids
            if class_id is None
            or self.cards_by_id[card_id].class_id in (0, class_id)
        )
        cards = tuple(sorted(cards, key=lambda card: (card.cost, card.card_id)))
        return cards if limit is None else cards[:limit]

    def sample_deck(
        self,
        class_id: int,
        rng: random.Random,
        *,
        deck_size: int = DECK_SIZE,
        max_copies: int = 3,
    ) -> list[CardDefinition]:
        """Sample a reproducible class-valid deck from exact-audited cards."""
        if deck_size <= 0:
            raise ValueError("deck_size must be positive")
        if max_copies <= 0:
            raise ValueError("max_copies must be positive")
        pool = self.pool(class_id=class_id)
        if len(pool) * max_copies < deck_size:
            raise ValueError(
                f"Exact card pool for class {class_id} cannot fill {deck_size} "
                f"slots with max_copies={max_copies}"
            )
        candidates = [card for card in pool for _ in range(max_copies)]
        return rng.sample(candidates, deck_size)


def _restore_catalog(
    cards: tuple[CardDefinition, ...],
    exact_collectible_ids: tuple[int, ...],
    coverage_report_sha256: str,
    source_snapshot_items: tuple[tuple[str, object], ...],
    catalog_sha256: str,
    card_vocabulary_sha256: str,
    training_pool_sha256: str,
) -> TrainableCardCatalog:
    return TrainableCardCatalog(
        cards_by_id=MappingProxyType({card.card_id: card for card in cards}),
        exact_collectible_ids=exact_collectible_ids,
        coverage_report_sha256=coverage_report_sha256,
        source_snapshot=MappingProxyType(dict(source_snapshot_items)),
        catalog_sha256=catalog_sha256,
        card_vocabulary_sha256=card_vocabulary_sha256,
        training_pool_sha256=training_pool_sha256,
    )
