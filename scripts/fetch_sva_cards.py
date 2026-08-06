from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from swb.db.import_cards import import_cards


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://sva.hypd.asia/data/cards.json"


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SWB-RL-card-importer/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"Download failed with HTTP {response.status}")
        return response.read()


def validate(payload: bytes) -> list[dict[str, object]]:
    cards = json.loads(payload.decode("utf-8"))
    if not isinstance(cards, list) or not cards:
        raise ValueError("Expected a non-empty JSON card array")
    required = {
        "card_id",
        "card_set_id",
        "name_chs",
        "cost",
        "class",
        "rarity",
        "type",
        "skill_texts",
    }
    ids = []
    for index, card in enumerate(cards):
        missing = required - card.keys()
        if missing:
            raise ValueError(f"Card at index {index} is missing {sorted(missing)}")
        ids.append(card["card_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Downloaded data contains duplicate card IDs")
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch cards from SVA and rebuild SQLite")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=ROOT / "shadowverse_cards.json")
    parser.add_argument("--database", type=Path, default=ROOT / "data" / "cards.sqlite3")
    parser.add_argument("--backup-dir", type=Path, default=ROOT / "data" / "backups")
    args = parser.parse_args()

    payload = download(args.url)
    cards = validate(payload)
    digest = hashlib.sha256(payload).hexdigest()
    fetched_at = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.output, args.database):
        if path.exists():
            shutil.copy2(path, args.backup_dir / f"{path.stem}_{stamp}{path.suffix}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.output.parent, delete=False, suffix=".json"
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    temporary_path.replace(args.output)

    import_cards(args.output, args.database)
    with sqlite3.connect(args.database) as connection:
        connection.execute(
            """
            INSERT INTO source_imports(source_url, fetched_at, sha256, card_count)
            VALUES (?, ?, ?, ?)
            """,
            (args.url, fetched_at, digest, len(cards)),
        )
        connection.commit()

    print(f"Fetched {len(cards)} cards from {args.url}")
    print(f"SHA-256: {digest}")
    print(f"JSON: {args.output}")
    print(f"SQLite: {args.database}")
    print(f"Backups: {args.backup_dir}")


if __name__ == "__main__":
    main()
