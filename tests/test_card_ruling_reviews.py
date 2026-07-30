# -*- coding: utf-8 -*-
"""Contracts for the official-source-first card ruling review ledger."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from urllib.parse import urlparse


REVIEW_PATH = Path("data/audits/card_ruling_reviews.json")
OFFICIAL_HOST_SUFFIXES = ("shadowverse-wb.com", "shadowverse.com")


class CardRulingReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    def test_search_policy_requires_official_sources_before_client_evidence(
        self,
    ) -> None:
        required_order = self.payload["search_policy"]["required_order"]
        self.assertEqual(
            [row["kind"] for row in required_order],
            [
                "official_card_qa",
                "official_help_and_rules",
                "official_notices_and_errata",
                "official_other_languages",
                "reproducible_client_evidence",
            ],
        )
        self.assertEqual(
            [row["order"] for row in required_order],
            list(range(1, len(required_order) + 1)),
        )

        fallback = self.payload["search_policy"]["no_official_match_policy"]
        self.assertTrue(fallback["allow_provisional_implementation"])
        self.assertEqual(fallback["required_status"], "ruling_uncertain")
        self.assertFalse(fallback["may_claim_official_confirmation"])
        self.assertFalse(fallback["may_close_from_tests_or_invariants_alone"])

    def test_every_review_records_queries_checked_pages_and_status_evidence(
        self,
    ) -> None:
        entries = self.payload["entries"]
        ruling_ids = [entry["ruling_id"] for entry in entries]
        self.assertEqual(len(ruling_ids), len(set(ruling_ids)))

        fallback = self.payload["search_policy"]["no_official_match_policy"]
        required_uncertain_fields = set(fallback["must_record"])
        for entry in entries:
            with self.subTest(ruling_id=entry["ruling_id"]):
                self.assertIn(entry["status"], {"confirmed", "ruling_uncertain"})
                self.assertTrue(entry["question"])
                self.assertTrue(entry["search_queries"])
                self.assertTrue(entry["official_pages_checked"])
                for page in entry["official_pages_checked"]:
                    self.assertTrue(page["authority"])
                    self.assertTrue(page["accessed_on"])
                    self.assertTrue(page["result"])
                    hostname = urlparse(page["url"]).hostname or ""
                    self.assertTrue(
                        any(
                            hostname == suffix
                            or hostname.endswith(f".{suffix}")
                            for suffix in OFFICIAL_HOST_SUFFIXES
                        ),
                        page["url"],
                    )

                if entry["status"] == "confirmed":
                    self.assertTrue(entry["conclusion"])
                    self.assertTrue(entry["evidence_scope"])
                else:
                    self.assertTrue(
                        required_uncertain_fields.issubset(entry),
                        required_uncertain_fields - set(entry),
                    )
                    self.assertGreaterEqual(len(entry["alternatives"]), 1)

    def test_swb_card_0008_has_direct_snow_awake_official_qa(self) -> None:
        entry = next(
            row
            for row in self.payload["entries"]
            if row["ruling_id"] == "SWB-RULING-0008-A"
        )
        self.assertEqual(entry["status"], "confirmed")
        self.assertTrue(
            any(
                page["authority"] == "official_card_qa"
                and "card_id=10132320" in page["url"]
                for page in entry["official_pages_checked"]
            )
        )
        self.assertIn("攻击 8、当前生命 4、生命上限 4", entry["conclusion"])


if __name__ == "__main__":
    unittest.main()
