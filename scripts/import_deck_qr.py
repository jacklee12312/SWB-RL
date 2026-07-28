from __future__ import annotations

import argparse
import json
from pathlib import Path

from swb.db.repository import CardRepository
from swb.deck_import import (
    DEFAULT_COVERAGE_REPORT,
    DEFAULT_DECK_DIRECTORY,
    DeckImportError,
    import_official_deck_qr,
    save_deck_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode an official Shadowverse: Worlds Beyond deck QR image, "
            "audit it against the local card/rule database, and register it."
        )
    )
    parser.add_argument("image", type=Path, help="PNG/JPEG QR image")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/cards.sqlite3"),
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=DEFAULT_COVERAGE_REPORT,
    )
    parser.add_argument(
        "--deck-directory",
        type=Path,
        default=DEFAULT_DECK_DIRECTORY,
    )
    parser.add_argument(
        "--name",
        help="stable lowercase registry name; generated from deck content by default",
    )
    parser.add_argument("--display-name")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing manifest with the same registry name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="decode and audit without saving a registry manifest",
    )
    parser.add_argument(
        "--require-trainable",
        action="store_true",
        help="return an error instead of saving if any card lacks exact coverage",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit = import_official_deck_qr(
            args.image,
            CardRepository(args.database),
            coverage_report=args.coverage_report,
        )
        manifest = audit.manifest(
            name=args.name,
            display_name=args.display_name,
        )
        if args.require_trainable and not audit.trainable:
            raise DeckImportError(
                "deck is not trainable: " + ", ".join(audit.issues)
            )
        destination = None
        if not args.dry_run:
            destination = save_deck_manifest(
                audit,
                directory=args.deck_directory,
                name=args.name,
                display_name=args.display_name,
                overwrite=args.overwrite,
            )
    except DeckImportError as exc:
        print(json.dumps(
            {"ok": False, "error": str(exc)},
            ensure_ascii=False,
            indent=2,
        ))
        return 2

    print(json.dumps(
        {
            "ok": True,
            "saved_to": None if destination is None else str(destination),
            "manifest": manifest,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
