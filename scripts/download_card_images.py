from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_SITE_URL = "https://sva.hypd.asia/"
DEFAULT_CATALOG_URL = urljoin(DEFAULT_SITE_URL, "data/cards.json")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
USER_AGENT = "SWB-project-card-image-downloader/1.0"


def fetch_json(url: str, *, timeout: float) -> list[dict[str, Any]]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError("card catalog must be a JSON array")
    return payload


def build_asset_index(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source_path: dict[str, dict[str, Any]] = {}
    filenames: dict[str, str] = {}
    for card in cards:
        card_id = int(card["card_id"])
        textures = card.get("textures") or {}
        if not isinstance(textures, dict):
            raise ValueError(f"card {card_id} textures must be an object")
        for variant, source_path in textures.items():
            if not isinstance(source_path, str):
                raise ValueError(f"card {card_id} texture {variant!r} must be a path")
            normalized = PurePosixPath(source_path).as_posix()
            if not normalized.startswith("data/textures/") or not normalized.endswith(".png"):
                raise ValueError(
                    f"card {card_id} texture {variant!r} has unsafe path {source_path!r}"
                )
            filename = PurePosixPath(normalized).name
            previous_path = filenames.setdefault(filename, normalized)
            if previous_path != normalized:
                raise ValueError(
                    f"texture filename collision: {previous_path!r} and {normalized!r}"
                )
            asset = by_source_path.setdefault(
                normalized,
                {
                    "source_path": normalized,
                    "filename": filename,
                    "references": [],
                },
            )
            asset["references"].append({"card_id": card_id, "variant": str(variant)})

    assets = sorted(by_source_path.values(), key=lambda item: item["source_path"])
    for asset in assets:
        asset["references"].sort(key=lambda item: (item["card_id"], item["variant"]))
    return assets


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_png(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


def download_asset(
    asset: dict[str, Any],
    *,
    site_url: str,
    output_dir: Path,
    timeout: float,
    retries: int,
    force: bool,
) -> dict[str, Any]:
    destination = output_dir / asset["filename"]
    source_url = urljoin(site_url, asset["source_path"])
    if destination.is_file() and not force and is_png(destination):
        return {
            "status": "skipped",
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
            "source_url": source_url,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(source_url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response, temporary.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
            if not is_png(temporary):
                raise ValueError("response is not a PNG image")
            temporary.replace(destination)
            return {
                "status": "downloaded",
                "bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
                "source_url": source_url,
            }
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"failed to download {source_url}: {last_error}") from last_error


def write_manifest(
    path: Path,
    *,
    site_url: str,
    catalog_url: str,
    cards: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> None:
    manifest_assets = []
    for asset in assets:
        result = results[asset["source_path"]]
        manifest_assets.append(
            {
                "source_path": asset["source_path"],
                "source_url": result["source_url"],
                "local_path": asset["filename"],
                "bytes": result["bytes"],
                "sha256": result["sha256"],
                "references": asset["references"],
            }
        )
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_site_url": site_url,
        "source_catalog_url": catalog_url,
        "card_count": len(cards),
        "asset_count": len(assets),
        "total_bytes": sum(item["bytes"] for item in manifest_assets),
        "assets": manifest_assets,
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all original SWB card textures listed by WBArts."
    )
    parser.add_argument("--site-url", default=DEFAULT_SITE_URL)
    parser.add_argument("--catalog-url", default=DEFAULT_CATALOG_URL)
    parser.add_argument("--output", type=Path, default=Path("data/card_images"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers must be positive")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.retries <= 0:
        parser.error("retries must be positive")

    site_url = args.site_url.rstrip("/") + "/"
    cards = fetch_json(args.catalog_url, timeout=args.timeout)
    assets = build_asset_index(cards)
    print(f"catalog_cards={len(cards)} texture_assets={len(assets)}")
    if args.dry_run:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    downloaded = 0
    skipped = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_asset,
                asset,
                site_url=site_url,
                output_dir=args.output,
                timeout=args.timeout,
                retries=args.retries,
                force=args.force,
            ): asset
            for asset in assets
        }
        for future in as_completed(futures):
            asset = futures[future]
            try:
                result = future.result()
                results[asset["source_path"]] = result
                if result["status"] == "downloaded":
                    downloaded += 1
                else:
                    skipped += 1
            except Exception as error:
                failures.append(str(error))
            completed += 1
            if completed % 25 == 0 or completed == len(assets):
                print(
                    f"progress={completed}/{len(assets)} "
                    f"downloaded={downloaded} skipped={skipped} failures={len(failures)}",
                    flush=True,
                )

    if failures:
        failure_path = args.output / "failures.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(
            f"{len(failures)} downloads failed; details written to {failure_path}"
        )

    write_manifest(
        args.output / "manifest.json",
        site_url=site_url,
        catalog_url=args.catalog_url,
        cards=cards,
        assets=assets,
        results=results,
    )
    total_bytes = sum(result["bytes"] for result in results.values())
    print(
        f"complete downloaded={downloaded} skipped={skipped} "
        f"total_gib={total_bytes / (1024 ** 3):.2f} "
        f"manifest={args.output / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
