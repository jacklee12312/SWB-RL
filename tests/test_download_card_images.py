from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.download_card_images import (
    PNG_SIGNATURE,
    build_asset_index,
    file_sha256,
    is_png,
)


class CardImageCatalogTests(unittest.TestCase):
    def test_build_asset_index_collects_all_variants_and_shared_references(self) -> None:
        assets = build_asset_index(
            [
                {
                    "card_id": 2,
                    "textures": {
                        "base": "data/textures/200.png",
                        "evo": "data/textures/201.png",
                    },
                },
                {
                    "card_id": 1,
                    "textures": {"base": "data/textures/200.png"},
                },
            ]
        )

        self.assertEqual(
            [asset["source_path"] for asset in assets],
            ["data/textures/200.png", "data/textures/201.png"],
        )
        self.assertEqual(
            assets[0]["references"],
            [
                {"card_id": 1, "variant": "base"},
                {"card_id": 2, "variant": "base"},
            ],
        )

    def test_build_asset_index_rejects_paths_outside_texture_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            build_asset_index(
                [{"card_id": 1, "textures": {"base": "../secret.png"}}]
            )

    def test_png_validation_and_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.png"
            path.write_bytes(PNG_SIGNATURE + b"fixture")

            self.assertTrue(is_png(path))
            self.assertEqual(
                file_sha256(path),
                "bd54b02fae14b6b9ed73887ded339b8ef846fbcba0d4e5f9d95470ac23ade242",
            )


if __name__ == "__main__":
    unittest.main()
