from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from swb.db.repository import CardDefinition
from swb.engine.deck import DECK_SIZE, validate_deck
from swb.rl.catalog import TrainableCardCatalog


OFFICIAL_QR_EVOLVE_HAVEN = "official_qr_evolve_haven_20260727"


@dataclass(frozen=True)
class FixedTrainingDeck:
    """Immutable, source-attributed deck recipe for reproducible training."""

    name: str
    display_name: str
    class_id: int
    card_ids: tuple[int, ...]
    source_deck_hash: str

    def __post_init__(self) -> None:
        if len(self.card_ids) != DECK_SIZE:
            raise ValueError(
                f"fixed deck {self.name!r} must contain {DECK_SIZE} cards"
            )
        excessive = {
            card_id: count
            for card_id, count in Counter(self.card_ids).items()
            if count > 3
        }
        if excessive:
            raise ValueError(
                f"fixed deck {self.name!r} exceeds the three-copy limit: "
                f"{excessive}"
            )

    @property
    def sha256(self) -> str:
        payload = {
            "name": self.name,
            "display_name": self.display_name,
            "class_id": self.class_id,
            "card_ids": self.card_ids,
            "source_deck_hash": self.source_deck_hash,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def build(self, catalog: TrainableCardCatalog) -> list[CardDefinition]:
        exact_ids = frozenset(catalog.exact_collectible_ids)
        deck: list[CardDefinition] = []
        for card_id in self.card_ids:
            definition = catalog.resolve(card_id)
            if definition is None:
                raise ValueError(
                    f"fixed deck {self.name!r} references missing card {card_id}"
                )
            if card_id not in exact_ids:
                raise ValueError(
                    f"fixed deck {self.name!r} references non-exact card "
                    f"{card_id}"
                )
            deck.append(definition)
        validate_deck(deck, self.class_id, player_index=0)
        return deck

    def manifest(self) -> dict[str, object]:
        counts = Counter(self.card_ids)
        return {
            "name": self.name,
            "display_name": self.display_name,
            "class_id": self.class_id,
            "card_ids": list(self.card_ids),
            "card_counts": {
                str(card_id): count for card_id, count in counts.items()
            },
            "source_deck_hash": self.source_deck_hash,
            "sha256": self.sha256,
        }


_OFFICIAL_QR_EVOLVE_HAVEN_HASH = (
    "1.6.dhqm.dhqm.fRes.fRes.fRes.fS9g.fS9g.fS9g.dJfu.dJfu.dJfu."
    "dv-s.dv-s.fS86.fS86.fS86.dXri.dXri.dXri.dwVg.dwVg.eJ8E.eJ8E."
    "eJ8E.egps.fSNu.fSNu.fSNu.di4E.di4E.di4E.ehJ6.fRue.fRue.fRue."
    "fSNk.fSNk.fSNk.fDkO.fDkO"
)

_OFFICIAL_QR_EVOLVE_HAVEN_CARD_IDS = (
    10403120,
    10403120,
    10861110,
    10861110,
    10861110,
    10863210,
    10863210,
    10863210,
    10304120,
    10304120,
    10304120,
    10461110,
    10461110,
    10863110,
    10863110,
    10863110,
    10362220,
    10362220,
    10362220,
    10463210,
    10463210,
    10564110,
    10564110,
    10564110,
    10661110,
    10864120,
    10864120,
    10864120,
    10404110,
    10404110,
    10404110,
    10663110,
    10862120,
    10862120,
    10862120,
    10864110,
    10864110,
    10864110,
    10804120,
    10804120,
)

FIXED_TRAINING_DECKS: Mapping[str, FixedTrainingDeck] = MappingProxyType({
    OFFICIAL_QR_EVOLVE_HAVEN: FixedTrainingDeck(
        name=OFFICIAL_QR_EVOLVE_HAVEN,
        display_name="官方二维码·超进化主教",
        class_id=6,
        card_ids=_OFFICIAL_QR_EVOLVE_HAVEN_CARD_IDS,
        source_deck_hash=_OFFICIAL_QR_EVOLVE_HAVEN_HASH,
    ),
})


def fixed_training_deck_names() -> tuple[str, ...]:
    return tuple(FIXED_TRAINING_DECKS)


def get_fixed_training_deck(name: str) -> FixedTrainingDeck:
    try:
        return FIXED_TRAINING_DECKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown fixed training deck {name!r}") from exc
