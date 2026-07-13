from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path

from scripts.ability_status import build_ability_audit


class AbilityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_ability_audit("data/audits/ability_registry.json")

    def test_every_registry_ability_has_reason_and_test_evidence(self):
        self.assertEqual(self.report["summary"]["total"], 34)
        self.assertEqual(
            self.report["summary"]["statuses"],
            {"implemented": 18, "partial": 5, "placeholder": 11},
        )
        for row in self.report["abilities"]:
            with self.subTest(keyword=row["keyword"]):
                self.assertTrue(row["reason"].strip())
                self.assertTrue(row["test_evidence"])
                for path in row["test_evidence"]:
                    self.assertTrue(Path(path).is_file(), path)

    def test_primitive_support_does_not_upgrade_registry_status(self):
        self.assertEqual(
            self.report["summary"]["primitive_statuses"],
            {"covered": 34},
        )
        statuses = {
            row["keyword"]: row["status"]
            for row in self.report["abilities"]
        }
        self.assertEqual(statuses["爆能强化"], "placeholder")
        self.assertEqual(statuses["融合"], "partial")
        self.assertEqual(statuses["瞬念召唤"], "implemented")

    def test_database_registry_matches_runtime_audit(self):
        with sqlite3.connect("data/cards.sqlite3") as conn:
            database = dict(conn.execute(
                "SELECT keyword, status FROM abilities ORDER BY keyword"
            ))
        audited = {
            row["keyword"]: row["status"]
            for row in self.report["abilities"]
        }
        self.assertEqual(database, audited)

    def test_report_is_deterministic(self):
        self.assertEqual(
            build_ability_audit("data/audits/ability_registry.json"),
            self.report,
        )


if __name__ == "__main__":
    unittest.main()
