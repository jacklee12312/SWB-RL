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
DEFAULT_EXCLUSION_POLICY = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "audits"
    / "training_catalog_exclusions.json"
)


@dataclass(frozen=True)
class TrainableCardCatalog:
    """Frozen card snapshot whose trainable subset passed the exact-rule audit.

    Database access is confined to construction. Matches use ``resolve`` and
    ``pool`` against immutable in-memory data, so workers do not query SQLite
    while resolving effects.
    """

    cards_by_id: Mapping[int, CardDefinition]
    audited_exact_collectible_ids: tuple[int, ...]
    exact_collectible_ids: tuple[int, ...]
    excluded_collectible_ids: tuple[int, ...]
    coverage_report_sha256: str
    exclusion_policy_sha256: str
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
        exclusion_policy: str | Path | None = DEFAULT_EXCLUSION_POLICY,
    ) -> TrainableCardCatalog:
        report_path = Path(coverage_report)
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes.decode("utf-8"))
        classifications = report.get("classifications")
        if not isinstance(classifications, dict):
            raise ValueError("Coverage report has no classifications mapping")

        cards = {card.card_id: card for card in repository.all_cards()}
        audited_exact_ids = tuple(sorted(
            int(card_id)
            for card_id, classification in classifications.items()
            if classification.get("coverage") == "covered_exact"
            and classification.get("is_collectible") is True
        ))
        missing = [
            card_id for card_id in audited_exact_ids if card_id not in cards
        ]
        if missing:
            raise ValueError(
                "Coverage report references cards absent from the database: "
                f"{missing[:5]}"
            )
        non_collectible = [
            card_id
            for card_id in audited_exact_ids
            if not cards[card_id].is_collectible
        ]
        if non_collectible:
            raise ValueError(
                "Coverage report marks non-collectible cards as trainable: "
                f"{non_collectible[:5]}"
            )
        if not audited_exact_ids:
            raise ValueError("Coverage report contains no exact collectible cards")

        exclusion_policy_sha256 = hashlib.sha256(b"disabled").hexdigest()
        excluded_ids: tuple[int, ...] = ()
        if exclusion_policy is not None:
            policy_path = Path(exclusion_policy)
            policy_bytes = policy_path.read_bytes()
            policy = json.loads(policy_bytes.decode("utf-8"))
            entries = policy.get("exclusions")
            if not isinstance(entries, list):
                raise ValueError(
                    "Catalog exclusion policy has no exclusions list"
                )
            raw_ids = [
                entry.get("card_id")
                for entry in entries
                if isinstance(entry, dict)
            ]
            if len(raw_ids) != len(entries) or any(
                not isinstance(card_id, int) for card_id in raw_ids
            ):
                raise ValueError(
                    "Catalog exclusions must each contain an integer card_id"
                )
            if len(set(raw_ids)) != len(raw_ids):
                raise ValueError(
                    "Catalog exclusion policy contains duplicate card IDs"
                )
            invalid = sorted(set(raw_ids) - set(audited_exact_ids))
            if invalid:
                raise ValueError(
                    "Catalog exclusions are not exact collectible cards: "
                    f"{invalid[:5]}"
                )
            missing_rulings = [
                entry["card_id"]
                for entry in entries
                if not isinstance(entry.get("ruling_ids"), list)
                or not entry["ruling_ids"]
                or not all(
                    isinstance(ruling_id, str) and ruling_id
                    for ruling_id in entry["ruling_ids"]
                )
            ]
            if missing_rulings:
                raise ValueError(
                    "Catalog exclusions require non-empty ruling_ids: "
                    f"{missing_rulings[:5]}"
                )
            excluded_ids = tuple(sorted(raw_ids))
            exclusion_policy_sha256 = hashlib.sha256(
                policy_bytes
            ).hexdigest()

        exact_ids = tuple(
            card_id
            for card_id in audited_exact_ids
            if card_id not in set(excluded_ids)
        )
        if not exact_ids:
            raise ValueError(
                "Catalog exclusion policy removed every exact collectible card"
            )

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
        if exclusion_policy is not None:
            catalog_payload.update({
                "exclusion_policy_sha256": exclusion_policy_sha256,
                "audited_exact_collectible_ids": audited_exact_ids,
                "excluded_collectible_ids": excluded_ids,
            })
        catalog_bytes = json.dumps(
            catalog_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return cls(
            cards_by_id=MappingProxyType(cards),
            audited_exact_collectible_ids=audited_exact_ids,
            exact_collectible_ids=exact_ids,
            excluded_collectible_ids=excluded_ids,
            coverage_report_sha256=coverage_sha256,
            exclusion_policy_sha256=exclusion_policy_sha256,
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
                self.audited_exact_collectible_ids,
                self.exact_collectible_ids,
                self.excluded_collectible_ids,
                self.coverage_report_sha256,
                self.exclusion_policy_sha256,
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
    audited_exact_collectible_ids: tuple[int, ...],
    exact_collectible_ids: tuple[int, ...],
    excluded_collectible_ids: tuple[int, ...],
    coverage_report_sha256: str,
    exclusion_policy_sha256: str,
    source_snapshot_items: tuple[tuple[str, object], ...],
    catalog_sha256: str,
    card_vocabulary_sha256: str,
    training_pool_sha256: str,
) -> TrainableCardCatalog:
    return TrainableCardCatalog(
        cards_by_id=MappingProxyType({card.card_id: card for card in cards}),
        audited_exact_collectible_ids=audited_exact_collectible_ids,
        exact_collectible_ids=exact_collectible_ids,
        excluded_collectible_ids=excluded_collectible_ids,
        coverage_report_sha256=coverage_report_sha256,
        exclusion_policy_sha256=exclusion_policy_sha256,
        source_snapshot=MappingProxyType(dict(source_snapshot_items)),
        catalog_sha256=catalog_sha256,
        card_vocabulary_sha256=card_vocabulary_sha256,
        training_pool_sha256=training_pool_sha256,
    )
