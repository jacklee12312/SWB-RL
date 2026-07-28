from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from swb.db.repository import CardDefinition
from swb.deck_import import (
    DEFAULT_DECK_DIRECTORY,
    DeckImportError,
    parse_official_deck_hash,
)
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


def load_registered_training_decks(
    directory: str | Path = DEFAULT_DECK_DIRECTORY,
) -> Mapping[str, FixedTrainingDeck]:
    root = Path(directory)
    if not root.exists():
        return MappingProxyType({})
    if not root.is_dir():
        raise ValueError(f"deck registry path is not a directory: {root}")

    loaded: dict[str, FixedTrainingDeck] = {}
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"unable to load registered deck manifest {path}"
            ) from exc
        if payload.get("schema_version") != 1:
            raise ValueError(
                f"unsupported registered deck schema in {path}"
            )
        validation = payload.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(
                f"registered deck manifest has no validation object: {path}"
            )
        if validation.get("trainable") is not True:
            continue
        if validation.get("issues") not in ([], ()):
            raise ValueError(
                f"registered trainable deck still has validation issues: {path}"
            )

        name = payload.get("name")
        display_name = payload.get("display_name")
        source_hash = payload.get("source_deck_hash")
        card_ids = payload.get("card_ids")
        if (
            not isinstance(name, str)
            or not isinstance(display_name, str)
            or not isinstance(source_hash, str)
            or not isinstance(card_ids, list)
            or any(
                not isinstance(card_id, int) or isinstance(card_id, bool)
                for card_id in card_ids
            )
        ):
            raise ValueError(
                f"registered deck manifest has invalid identity fields: {path}"
            )
        if path.stem != name:
            raise ValueError(
                f"registered deck filename/name mismatch: {path}"
            )
        try:
            decoded = parse_official_deck_hash(source_hash)
        except DeckImportError as exc:
            raise ValueError(
                f"registered deck contains an invalid official hash: {path}"
            ) from exc
        if (
            int(payload.get("format_id", -1)) != decoded.format_id
            or int(payload.get("class_id", -1)) != decoded.class_id
            or tuple(card_ids) != decoded.card_ids
            or payload.get("content_sha256") != decoded.content_sha256
        ):
            raise ValueError(
                f"registered deck manifest/hash mismatch: {path}"
            )
        if name in loaded or name in FIXED_TRAINING_DECKS:
            raise ValueError(f"duplicate registered deck name {name!r}")
        loaded[name] = FixedTrainingDeck(
            name=name,
            display_name=display_name,
            class_id=decoded.class_id,
            card_ids=decoded.card_ids,
            source_deck_hash=decoded.source_hash,
        )
    return MappingProxyType(loaded)


@lru_cache(maxsize=1)
def _registered_training_decks() -> Mapping[str, FixedTrainingDeck]:
    return load_registered_training_decks()


def clear_registered_training_deck_cache() -> None:
    """Expose an explicit refresh point for long-lived UI/backend processes."""
    _registered_training_decks.cache_clear()


def fixed_training_deck_names() -> tuple[str, ...]:
    return tuple(FIXED_TRAINING_DECKS) + tuple(
        name
        for name in _registered_training_decks()
        if name not in FIXED_TRAINING_DECKS
    )


def get_fixed_training_deck(name: str) -> FixedTrainingDeck:
    try:
        return FIXED_TRAINING_DECKS[name]
    except KeyError:
        try:
            return _registered_training_decks()[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown fixed training deck {name!r}"
            ) from exc
