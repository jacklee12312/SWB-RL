from __future__ import annotations

import unittest

from scripts.report_token_audit import AUDIT_CATEGORIES, _build_token_audit


class TokenAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        cls.cards = {
            card["card_id"]: card
            for card in cls.report["cards"]
        }

    def test_audit_covers_every_non_collectible_card_once(self):
        self.assertEqual(self.report["summary"]["total"], 91)
        self.assertEqual(len(self.cards), 91)
        self.assertEqual(
            sum(self.report["summary"]["categories"].values()),
            91,
        )
        self.assertEqual(
            tuple(self.report["summary"]["categories"]),
            AUDIT_CATEGORIES,
        )

    def test_exact_generated_faith_spells_have_complete_entries(self):
        for card_id in (90014330, 90064320):
            with self.subTest(card_id=card_id):
                card = self.cards[card_id]
                self.assertEqual(card["category"], "entry_behavior_complete")
                self.assertEqual(card["explicit_coverage"], "exact")
                self.assertTrue(card["authored_producers"])

    def test_mjerrabaine_testimony_has_a_complete_executable_producer(self):
        proof = self.cards[90004320]
        self.assertTrue(proof["database_sources"])
        self.assertEqual(
            proof["authored_producers"],
            [
                {
                    "source_card_id": 10304110,
                    "entry_kind": "add_card",
                    "rule_file": "real_mjerrabaine_deck_batch.json",
                    "rule_group": "rules",
                }
            ],
        )
        self.assertEqual(proof["category"], "entry_behavior_complete")
        self.assertEqual(proof["explicit_coverage"], "exact")

    def test_goblin_has_a_complete_executable_producer(self):
        goblin = self.cards[90001110]
        self.assertEqual(goblin["category"], "entry_behavior_complete")
        self.assertEqual(
            goblin["authored_producers"],
            [
                {
                    "source_card_id": 10101310,
                    "entry_kind": "summon",
                    "rule_file": "real_basic_spells_batch.json",
                    "rule_group": "rules",
                }
            ],
        )

    def test_generic_earth_sigil_entry_is_explicitly_audited(self):
        sigil = self.cards[90031210]
        self.assertEqual(
            sigil["authored_producers"][0]["entry_kind"],
            "engine_earth_sigil",
        )
        self.assertEqual(sigil["category"], "entry_behavior_complete")
        self.assertEqual(sigil["explicit_coverage"], "exact")

    def test_report_is_deterministic(self):
        again = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        self.assertEqual(again, self.report)


if __name__ == "__main__":
    unittest.main()
