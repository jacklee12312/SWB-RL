from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote

from swb.db.repository import CardRepository
from swb.engine.deck import CLASS_NAMES, DECK_SIZE, PLAYABLE_CLASS_IDS, validate_deck


DEFAULT_COVERAGE_REPORT = (
    Path(__file__).resolve().parents[1] / "data" / "reports" / "rule_coverage.json"
)
DEFAULT_DECK_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "decks"
OFFICIAL_DECK_ALPHABET = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
)
MAX_QR_PAYLOAD_LENGTH = 32_768
MAX_QR_IMAGE_PIXELS = 25_000_000

_CARD_TOKEN_LENGTH = 4
_CARD_TOKEN_BASE = len(OFFICIAL_DECK_ALPHABET)
_CARD_TOKEN_LIMIT = _CARD_TOKEN_BASE**_CARD_TOKEN_LENGTH
_DECK_HASH_PATTERN = re.compile(
    rf"(?<![0-9A-Za-z_.-])"
    rf"(?P<hash>[0-9]+\.[1-7]"
    rf"(?:\.[0-9A-Za-z_-]{{{_CARD_TOKEN_LENGTH}}}){{{DECK_SIZE}}})"
    rf"(?!(?:[0-9A-Za-z_-]|\.[0-9A-Za-z_-]))"
)
_DECK_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_ALPHABET_INDEX = {
    character: index
    for index, character in enumerate(OFFICIAL_DECK_ALPHABET)
}


class DeckImportError(ValueError):
    """An official deck payload, image, or manifest could not be imported."""


class QRDependencyError(DeckImportError):
    """The optional QR image dependencies are unavailable."""


@dataclass(frozen=True)
class OfficialDeckHash:
    format_id: int
    class_id: int
    card_tokens: tuple[str, ...]
    card_ids: tuple[int, ...]
    source_hash: str

    @property
    def card_counts(self) -> Counter[int]:
        return Counter(self.card_ids)

    @property
    def content_sha256(self) -> str:
        payload = {
            "format_id": self.format_id,
            "class_id": self.class_id,
            "card_counts": sorted(self.card_counts.items()),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ImportedCard:
    card_id: int
    count: int
    name: str | None
    class_id: int | None
    class_name: str | None
    cost: int | None
    card_type: str | None
    is_collectible: bool | None
    coverage: str

    def manifest(self) -> dict[str, object]:
        return {
            "card_id": self.card_id,
            "count": self.count,
            "name": self.name,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "cost": self.cost,
            "card_type": self.card_type,
            "is_collectible": self.is_collectible,
            "coverage": self.coverage,
        }


@dataclass(frozen=True)
class DeckImportAudit:
    deck: OfficialDeckHash
    source_payload: str
    source_kind: str
    cards: tuple[ImportedCard, ...]
    issues: tuple[str, ...]
    database_source: dict[str, object]
    coverage_report_sha256: str

    @property
    def trainable(self) -> bool:
        return not self.issues

    def default_name(self) -> str:
        return (
            f"qr_class_{self.deck.class_id}_"
            f"{self.deck.content_sha256[:12]}"
        )

    def manifest(
        self,
        *,
        name: str | None = None,
        display_name: str | None = None,
    ) -> dict[str, object]:
        resolved_name = self.default_name() if name is None else name
        _validate_deck_name(resolved_name)
        resolved_display_name = (
            f"二维码导入·{CLASS_NAMES[self.deck.class_id]}·"
            f"{self.deck.content_sha256[:8]}"
            if display_name is None
            else display_name
        )
        return {
            "schema_version": 1,
            "name": resolved_name,
            "display_name": resolved_display_name,
            "format_id": self.deck.format_id,
            "class_id": self.deck.class_id,
            "class_name": CLASS_NAMES[self.deck.class_id],
            "card_ids": list(self.deck.card_ids),
            "card_counts": {
                str(card_id): count
                for card_id, count in sorted(self.deck.card_counts.items())
            },
            "cards": [card.manifest() for card in self.cards],
            "source_deck_hash": self.deck.source_hash,
            "content_sha256": self.deck.content_sha256,
            "source": {
                "kind": self.source_kind,
                "payload": self.source_payload,
            },
            "validation": {
                "trainable": self.trainable,
                "issues": list(self.issues),
                "coverage_report_sha256": self.coverage_report_sha256,
                "database_source": self.database_source,
            },
        }


def decode_card_token(token: str) -> int:
    if len(token) != _CARD_TOKEN_LENGTH:
        raise DeckImportError(
            f"official card token must contain {_CARD_TOKEN_LENGTH} characters"
        )
    value = 0
    for character in token:
        try:
            digit = _ALPHABET_INDEX[character]
        except KeyError as exc:
            raise DeckImportError(
                f"official card token contains invalid character {character!r}"
            ) from exc
        value = value * _CARD_TOKEN_BASE + digit
    if value <= 0:
        raise DeckImportError("official card token decodes to an invalid card ID")
    return value


def encode_card_id(card_id: int) -> str:
    if (
        not isinstance(card_id, int)
        or isinstance(card_id, bool)
        or not 0 < card_id < _CARD_TOKEN_LIMIT
    ):
        raise DeckImportError(
            f"card_id must be in 1..{_CARD_TOKEN_LIMIT - 1}"
        )
    remaining = card_id
    characters = ["0"] * _CARD_TOKEN_LENGTH
    for index in range(_CARD_TOKEN_LENGTH - 1, -1, -1):
        remaining, digit = divmod(remaining, _CARD_TOKEN_BASE)
        characters[index] = OFFICIAL_DECK_ALPHABET[digit]
    return "".join(characters)


def extract_official_deck_hash(payload: str) -> str:
    if not isinstance(payload, str):
        raise DeckImportError("QR payload must be text")
    candidate = payload.strip()
    if not candidate:
        raise DeckImportError("QR payload is empty")
    if len(candidate) > MAX_QR_PAYLOAD_LENGTH:
        raise DeckImportError("QR payload is too large")

    decoded = candidate
    for _ in range(2):
        unquoted = unquote(decoded)
        if unquoted == decoded:
            break
        decoded = unquoted

    matches = list(dict.fromkeys(
        match.group("hash") for match in _DECK_HASH_PATTERN.finditer(decoded)
    ))
    if not matches:
        raise DeckImportError("QR payload does not contain an official deck hash")
    if len(matches) > 1:
        raise DeckImportError("QR payload contains multiple official deck hashes")
    return matches[0]


def parse_official_deck_hash(source_hash: str) -> OfficialDeckHash:
    extracted = extract_official_deck_hash(source_hash)
    if extracted != source_hash.strip():
        raise DeckImportError(
            "parse_official_deck_hash requires a bare official deck hash"
        )
    parts = extracted.split(".")
    expected_parts = DECK_SIZE + 2
    if len(parts) != expected_parts:
        raise DeckImportError(
            f"official deck hash must contain {expected_parts} fields"
        )
    try:
        format_id = int(parts[0])
        class_id = int(parts[1])
    except ValueError as exc:
        raise DeckImportError(
            "official deck format and class fields must be integers"
        ) from exc
    if format_id <= 0:
        raise DeckImportError("official deck format field must be positive")
    if class_id not in PLAYABLE_CLASS_IDS:
        raise DeckImportError(
            f"official deck class must be one of {sorted(PLAYABLE_CLASS_IDS)}"
        )

    tokens = tuple(parts[2:])
    card_ids = tuple(decode_card_token(token) for token in tokens)
    excessive = {
        card_id: count
        for card_id, count in Counter(card_ids).items()
        if count > 3
    }
    if excessive:
        raise DeckImportError(
            f"official deck exceeds the three-copy limit: {excessive}"
        )
    if any(
        encode_card_id(card_id) != token
        for card_id, token in zip(card_ids, tokens)
    ):
        raise DeckImportError("official deck card token is not canonical")
    return OfficialDeckHash(
        format_id=format_id,
        class_id=class_id,
        card_tokens=tokens,
        card_ids=card_ids,
        source_hash=extracted,
    )


def parse_official_deck_payload(payload: str) -> OfficialDeckHash:
    return parse_official_deck_hash(extract_official_deck_hash(payload))


def _source_kind(payload: str) -> str:
    lowered = payload.lower()
    if "shadowverse-wb.com" in lowered:
        return "international_official_qr"
    if "163.com" in lowered:
        return "netease_official_qr"
    if payload.strip() == extract_official_deck_hash(payload):
        return "official_deck_hash"
    return "official_qr"


def _load_coverage_report(
    repository: CardRepository,
    coverage_report: str | Path,
) -> tuple[dict[str, object], str]:
    path = Path(coverage_report)
    try:
        report_bytes = path.read_bytes()
        report = json.loads(report_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeckImportError(
            f"unable to load rule coverage report {path}"
        ) from exc
    classifications = report.get("classifications")
    if not isinstance(classifications, dict):
        raise DeckImportError("rule coverage report has no classifications")

    expected_source = report.get("generated_from", {}).get(
        "source_snapshot", {}
    )
    actual_source = repository.source_snapshot()
    if isinstance(expected_source, dict):
        for field in ("sha256", "card_count"):
            expected = expected_source.get(field)
            actual = actual_source.get(field)
            if (
                expected is not None
                and actual is not None
                and expected != actual
            ):
                raise DeckImportError(
                    "rule coverage report/database mismatch for "
                    f"{field}: report={expected!r}, database={actual!r}"
                )
    return classifications, hashlib.sha256(report_bytes).hexdigest()


def audit_official_deck(
    deck: OfficialDeckHash,
    repository: CardRepository,
    *,
    source_payload: str | None = None,
    coverage_report: str | Path = DEFAULT_COVERAGE_REPORT,
) -> DeckImportAudit:
    classifications, coverage_sha256 = _load_coverage_report(
        repository, coverage_report
    )
    counts = deck.card_counts
    unique_ids = tuple(dict.fromkeys(deck.card_ids))
    definitions = []
    cards = []
    issues: list[str] = []
    for card_id in unique_ids:
        classification = classifications.get(str(card_id))
        coverage = (
            str(classification.get("coverage", "missing"))
            if isinstance(classification, dict)
            else "missing"
        )
        try:
            definition = repository.get(card_id)
        except KeyError:
            issues.append(f"missing_card:{card_id}")
            cards.append(ImportedCard(
                card_id=card_id,
                count=counts[card_id],
                name=None,
                class_id=None,
                class_name=None,
                cost=None,
                card_type=None,
                is_collectible=None,
                coverage=coverage,
            ))
            continue

        definitions.extend([definition] * counts[card_id])
        if not definition.is_collectible:
            issues.append(f"non_collectible_card:{card_id}")
        if definition.class_id not in (0, deck.class_id):
            issues.append(f"off_class_card:{card_id}")
        if coverage != "covered_exact":
            issues.append(f"rule_not_exact:{card_id}:{coverage}")
        cards.append(ImportedCard(
            card_id=card_id,
            count=counts[card_id],
            name=definition.name,
            class_id=definition.class_id,
            class_name=definition.class_name,
            cost=definition.cost,
            card_type=definition.card_type,
            is_collectible=definition.is_collectible,
            coverage=coverage,
        ))

    if len(definitions) == DECK_SIZE:
        try:
            validate_deck(definitions, deck.class_id, player_index=0)
        except ValueError as exc:
            issues.append(f"illegal_deck:{exc}")
    else:
        issues.append(
            f"unresolved_deck_cards:{DECK_SIZE - len(definitions)}"
        )

    payload = deck.source_hash if source_payload is None else source_payload.strip()
    return DeckImportAudit(
        deck=deck,
        source_payload=payload,
        source_kind=_source_kind(payload),
        cards=tuple(cards),
        issues=tuple(dict.fromkeys(issues)),
        database_source=dict(repository.source_snapshot()),
        coverage_report_sha256=coverage_sha256,
    )


def audit_official_deck_payload(
    payload: str,
    repository: CardRepository,
    *,
    coverage_report: str | Path = DEFAULT_COVERAGE_REPORT,
) -> DeckImportAudit:
    return audit_official_deck(
        parse_official_deck_payload(payload),
        repository,
        source_payload=payload,
        coverage_report=coverage_report,
    )


def read_qr_payloads(
    image: str | Path | bytes | bytearray | BinaryIO,
) -> tuple[str, ...]:
    try:
        from PIL import Image
        import zxingcpp
    except ImportError as exc:
        raise QRDependencyError(
            "QR image import requires the optional 'qr' dependencies; "
            "install with `pip install -e .[qr]`"
        ) from exc

    source: object
    if isinstance(image, (bytes, bytearray)):
        source = BytesIO(bytes(image))
    else:
        source = image
    try:
        with Image.open(source) as opened:
            width, height = opened.size
            if (
                width <= 0
                or height <= 0
                or width * height > MAX_QR_IMAGE_PIXELS
            ):
                raise DeckImportError(
                    f"QR image dimensions are outside the supported limit: "
                    f"{width}x{height}"
                )
            opened.load()
            results = zxingcpp.read_barcodes(
                opened,
                formats=zxingcpp.BarcodeFormat.QRCode,
                try_rotate=True,
                try_downscale=True,
                try_invert=True,
            )
    except DeckImportError:
        raise
    except Exception as exc:
        raise DeckImportError("unable to read QR image") from exc

    payloads = tuple(dict.fromkeys(
        result.text.strip()
        for result in results
        if result.text and result.text.strip()
    ))
    if not payloads:
        raise DeckImportError("QR image contains no readable QR code")
    return payloads


def import_official_deck_qr(
    image: str | Path | bytes | bytearray | BinaryIO,
    repository: CardRepository,
    *,
    coverage_report: str | Path = DEFAULT_COVERAGE_REPORT,
) -> DeckImportAudit:
    candidates: dict[str, tuple[OfficialDeckHash, str]] = {}
    for payload in read_qr_payloads(image):
        try:
            deck = parse_official_deck_payload(payload)
        except DeckImportError:
            continue
        candidates.setdefault(deck.source_hash, (deck, payload))
    if not candidates:
        raise DeckImportError(
            "QR image contains no supported official constructed deck"
        )
    if len(candidates) > 1:
        raise DeckImportError(
            "QR image contains multiple different official decks"
        )
    deck, payload = next(iter(candidates.values()))
    return audit_official_deck(
        deck,
        repository,
        source_payload=payload,
        coverage_report=coverage_report,
    )


def _validate_deck_name(name: str) -> None:
    if not isinstance(name, str) or not _DECK_NAME_PATTERN.fullmatch(name):
        raise DeckImportError(
            "deck name must contain 1-64 lowercase letters, digits, "
            "underscores, or hyphens and must start with a letter or digit"
        )


def save_deck_manifest(
    audit: DeckImportAudit,
    *,
    directory: str | Path = DEFAULT_DECK_DIRECTORY,
    name: str | None = None,
    display_name: str | None = None,
    overwrite: bool = False,
) -> Path:
    manifest = audit.manifest(name=name, display_name=display_name)
    destination_directory = Path(directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / f"{manifest['name']}.json"
    if destination.exists() and not overwrite:
        raise DeckImportError(
            f"deck manifest already exists: {destination}"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination_directory,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(
                manifest,
                temporary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination
