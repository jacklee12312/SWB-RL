from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "jacklee12312/SWB-RL"
LATEST_MANIFEST = ROOT / "releases" / "latest.json"
REQUIRED_PATHS = (
    ROOT / "pyproject.toml",
    ROOT / "shadowverse_cards.json",
    ROOT / "data" / "cards.sqlite3",
    ROOT / "data" / "rules",
    ROOT / "data" / "decks",
    ROOT / "swb" / "engine",
    ROOT / "tests",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def safe_extract_zip(archive: Path, destination: Path) -> None:
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if not _is_within(resolved_destination, target):
                raise ValueError(f"unsafe release path: {member.filename}")
        if "filter" in inspect.signature(package.extractall).parameters:
            package.extractall(destination, filter="data")
        else:
            package.extractall(destination)


def safe_extract_tar(archive: Path, destination: Path) -> None:
    resolved_destination = destination.resolve()
    with tarfile.open(archive) as package:
        for member in package.getmembers():
            target = (destination / member.name).resolve()
            if not _is_within(resolved_destination, target):
                raise ValueError(f"unsafe release path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"release links are not allowed: {member.name}")
        if "filter" in inspect.signature(package.extractall).parameters:
            package.extractall(destination, filter="data")
        else:
            package.extractall(destination)


def safe_extract_archive(archive: Path, destination: Path) -> None:
    if archive.suffix.lower() == ".zip":
        safe_extract_zip(archive, destination)
        return
    if archive.suffix.lower() == ".tar":
        safe_extract_tar(archive, destination)
        return
    raise ValueError(f"unsupported release archive: {archive.name}")


def verify_checkout(required_paths: Iterable[Path] = REQUIRED_PATHS) -> dict[str, object]:
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required checkout paths: " + ", ".join(missing))

    catalog = json.loads((ROOT / "shadowverse_cards.json").read_text(encoding="utf-8"))
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("shadowverse_cards.json must contain a non-empty card list")

    database = ROOT / "data" / "cards.sqlite3"
    with sqlite3.connect(database) as connection:
        card_count = int(connection.execute("SELECT COUNT(*) FROM cards").fetchone()[0])
    if card_count <= 0:
        raise ValueError("data/cards.sqlite3 contains no cards")

    rule_files = tuple((ROOT / "data" / "rules").glob("*.json"))
    if not rule_files:
        raise ValueError("data/rules contains no JSON rules")

    return {
        "catalog_records": len(catalog),
        "database_cards": card_count,
        "database_sha256": sha256_file(database),
        "rule_files": len(rule_files),
    }


def _run(command: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def install_dependencies(with_ui: bool) -> None:
    _run([sys.executable, "-m", "pip", "install", "-e", ".[rl,train,qr]"])
    if with_ui:
        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError("npm is required for --with-ui")
        _run([npm, "ci"], cwd=ROOT / "simulator-ui")


def load_release_manifest(tag: str) -> dict[str, object]:
    manifest_path = LATEST_MANIFEST if tag == "latest" else ROOT / "releases" / f"{tag}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"release manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("tag", "assets"):
        if key not in manifest:
            raise ValueError(f"release manifest missing {key!r}: {manifest_path}")
    assets = manifest["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"release manifest has no assets: {manifest_path}")
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ValueError(f"release asset {index} must be an object: {manifest_path}")
        for key in ("name", "sha256", "size_bytes"):
            if key not in asset:
                raise ValueError(
                    f"release asset {index} missing {key!r}: {manifest_path}"
                )
    return manifest


def download_release(tag: str) -> Path:
    manifest = load_release_manifest(tag)
    release_tag = str(manifest["tag"])
    with tempfile.TemporaryDirectory(prefix="swb-release-") as temporary:
        for asset_metadata in manifest["assets"]:
            asset = str(asset_metadata["name"])
            expected_sha256 = str(asset_metadata["sha256"])
            expected_size = int(asset_metadata["size_bytes"])
            url = (
                f"https://github.com/{REPOSITORY}/releases/download/"
                f"{release_tag}/{asset}"
            )
            archive = Path(temporary) / asset
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, archive)
            if archive.stat().st_size != expected_size:
                raise ValueError(
                    f"release size mismatch for {asset}: expected {expected_size}, "
                    f"got {archive.stat().st_size}"
                )
            actual_sha256 = sha256_file(archive)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"release SHA-256 mismatch for {asset}: expected "
                    f"{expected_sha256}, got {actual_sha256}"
                )
            safe_extract_archive(archive, ROOT)

    marker = ROOT / "data" / "checkpoints" / "RELEASE_INSTALLED.json"
    if not marker.exists():
        raise FileNotFoundError("release extracted without RELEASE_INSTALLED.json")
    print(f"Installed research snapshot {release_tag}")
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a checkout, install dependencies, and fetch research snapshots."
    )
    parser.add_argument("--install", action="store_true", help="install Python dependencies")
    parser.add_argument(
        "--with-ui",
        action="store_true",
        help="run npm ci for simulator-ui (implies --install)",
    )
    parser.add_argument(
        "--release",
        metavar="TAG",
        help="download and verify a research release; use 'latest' for the recommended snapshot",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="run the complete Python unit suite and compile check after setup",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if sys.version_info < (3, 11):
        raise RuntimeError("SWB RL requires Python 3.11 or newer")

    if arguments.install or arguments.with_ui:
        install_dependencies(arguments.with_ui)
    verification = verify_checkout()
    if arguments.release:
        download_release(arguments.release)
    if arguments.run_tests:
        _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        _run([sys.executable, "-m", "compileall", "-q", "swb", "scripts", "tests"])

    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
