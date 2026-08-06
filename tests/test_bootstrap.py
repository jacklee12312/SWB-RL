from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts import bootstrap


class BootstrapTests(unittest.TestCase):
    def test_verify_checkout_reports_database_and_catalog(self) -> None:
        result = bootstrap.verify_checkout()
        self.assertGreater(result["catalog_records"], 0)
        self.assertGreater(result["database_cards"], 0)
        self.assertEqual(len(result["database_sha256"]), 64)
        self.assertGreater(result["rule_files"], 0)

    def test_safe_extract_zip_accepts_nested_release_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "release.zip"
            destination = root / "checkout"
            destination.mkdir()
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("data/checkpoints/example.pt", b"checkpoint")

            bootstrap.safe_extract_zip(archive, destination)

            self.assertEqual(
                (destination / "data/checkpoints/example.pt").read_bytes(),
                b"checkpoint",
            )

    def test_safe_extract_zip_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "release.zip"
            destination = root / "checkout"
            destination.mkdir()
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("../escape.txt", b"bad")

            with self.assertRaisesRegex(ValueError, "unsafe release path"):
                bootstrap.safe_extract_zip(archive, destination)
            self.assertFalse((root / "escape.txt").exists())

    def test_safe_extract_tar_accepts_nested_release_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "release.tar"
            source = root / "example.pt"
            source.write_bytes(b"checkpoint")
            destination = root / "checkout"
            destination.mkdir()
            with tarfile.open(archive, "w") as package:
                package.add(source, arcname="data/checkpoints/example.pt")

            bootstrap.safe_extract_tar(archive, destination)

            self.assertEqual(
                (destination / "data/checkpoints/example.pt").read_bytes(),
                b"checkpoint",
            )

    def test_safe_extract_tar_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "release.tar"
            source = root / "escape.txt"
            source.write_bytes(b"bad")
            destination = root / "checkout"
            destination.mkdir()
            with tarfile.open(archive, "w") as package:
                package.add(source, arcname="../escape.txt")

            with self.assertRaisesRegex(ValueError, "unsafe release path"):
                bootstrap.safe_extract_tar(archive, destination)

    def test_load_release_manifest_requires_complete_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps({"tag": "test"}), encoding="utf-8")
            original = bootstrap.LATEST_MANIFEST
            try:
                bootstrap.LATEST_MANIFEST = path
                with self.assertRaisesRegex(ValueError, "missing 'assets'"):
                    bootstrap.load_release_manifest("latest")
            finally:
                bootstrap.LATEST_MANIFEST = original


if __name__ == "__main__":
    unittest.main()
