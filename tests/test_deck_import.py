from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from swb.db.repository import CardRepository
from swb.deck_import import (
    DEFAULT_COVERAGE_REPORT,
    DeckImportError,
    audit_official_deck_payload,
    decode_card_token,
    encode_card_id,
    extract_official_deck_hash,
    import_official_deck_qr,
    parse_official_deck_hash,
    parse_official_deck_payload,
    save_deck_manifest,
)
from swb.rl.fixed_decks import load_registered_training_decks


DATABASE = Path("data/cards.sqlite3")
INTERNATIONAL_QR_FIXTURE = Path(
    "tests/fixtures/official_deck_qr_international_forest.png"
)
INTERNATIONAL_HASH = (
    "1.1.dhqm.dhqm.e6C6.e6C6.e6C6.fFhE.fFhE.fFhE.fGAe.fGAe.fGAe."
    "e6x8.e6x8.e6x8.fFRm.fFRm.dM5-.dM5-.dM5-.dkGs.dkGs.dkGs.dkWe."
    "dkWe.e6hM.e6hM.e6hM.fFz-.fFz-.fFz-.di4E.di4E.fFws.fFws.fGAU."
    "fGAU.fGAU.eVLU.eVLU.eVLU"
)
INTERNATIONAL_PAYLOAD = (
    "https://shadowverse-wb.com/ja/deck/detail/?hash="
    + INTERNATIONAL_HASH
)
NETEASE_HASH = (
    "1.6.dhqm.dhqm.fRes.fRes.fRes.fS9g.fS9g.fS9g.dJfu.dJfu.dJfu."
    "dv-s.dv-s.fS86.fS86.fS86.dXri.dXri.dXri.dwVg.dwVg.eJ8E.eJ8E."
    "eJ8E.egps.fSNu.fSNu.fSNu.di4E.di4E.di4E.ehJ6.fRue.fRue.fRue."
    "fSNk.fSNk.fSNk.fDkO.fDkO"
)
NETEASE_PAYLOAD = (
    "https://sv.163.com/?code"
    "https://ma68sv2.game.163.com/index.phpchs/deck/detail/?hash="
    + NETEASE_HASH
)
EXPECTED_INTERNATIONAL_COUNTS = {
    10403120: 2,
    10511110: 3,
    10812110: 3,
    10814120: 3,
    10514120: 3,
    10811120: 2,
    10314110: 3,
    10413110: 3,
    10414120: 2,
    10513110: 3,
    10813310: 3,
    10404110: 2,
    10813110: 2,
    10814110: 3,
    10614110: 3,
}


class OfficialDeckHashTests(unittest.TestCase):
    def test_card_tokens_use_the_official_custom_base64_alphabet(self) -> None:
        examples = {
            "dhqm": 10403120,
            "fRes": 10861110,
            "fS9g": 10863210,
            "dJfu": 10304120,
            "dv-s": 10461110,
        }
        for token, card_id in examples.items():
            with self.subTest(token=token):
                self.assertEqual(decode_card_token(token), card_id)
                self.assertEqual(encode_card_id(card_id), token)

    def test_international_payload_decodes_all_forty_cards(self) -> None:
        deck = parse_official_deck_payload(INTERNATIONAL_PAYLOAD)
        self.assertEqual(deck.format_id, 1)
        self.assertEqual(deck.class_id, 1)
        self.assertEqual(len(deck.card_ids), 40)
        self.assertEqual(Counter(deck.card_ids), EXPECTED_INTERNATIONAL_COUNTS)
        self.assertEqual(deck.source_hash, INTERNATIONAL_HASH)
        self.assertEqual(len(deck.content_sha256), 64)

    def test_netease_wrapper_uses_the_same_hash_decoder(self) -> None:
        deck = parse_official_deck_payload(NETEASE_PAYLOAD)
        self.assertEqual(deck.format_id, 1)
        self.assertEqual(deck.class_id, 6)
        self.assertEqual(len(deck.card_ids), 40)
        self.assertEqual(deck.source_hash, NETEASE_HASH)
        self.assertEqual(deck.card_ids[:5], (
            10403120,
            10403120,
            10861110,
            10861110,
            10861110,
        ))

    def test_percent_encoded_official_url_is_supported(self) -> None:
        encoded = INTERNATIONAL_PAYLOAD.replace(":", "%3A").replace("/", "%2F")
        self.assertEqual(
            extract_official_deck_hash(encoded),
            INTERNATIONAL_HASH,
        )

    def test_multiple_deck_hashes_are_rejected(self) -> None:
        with self.assertRaisesRegex(DeckImportError, "multiple"):
            extract_official_deck_hash(
                INTERNATIONAL_PAYLOAD + " " + NETEASE_PAYLOAD
            )

    def test_wrong_card_count_is_rejected(self) -> None:
        short_hash = ".".join(INTERNATIONAL_HASH.split(".")[:-1])
        with self.assertRaisesRegex(DeckImportError, "does not contain"):
            parse_official_deck_payload(short_hash)

    def test_five_character_token_cannot_be_partially_accepted(self) -> None:
        malformed = INTERNATIONAL_HASH + "x"
        with self.assertRaisesRegex(DeckImportError, "does not contain"):
            parse_official_deck_payload(malformed)

    def test_more_than_three_copies_is_rejected(self) -> None:
        parts = INTERNATIONAL_HASH.split(".")
        parts[2:6] = [parts[2]] * 4
        with self.assertRaisesRegex(DeckImportError, "three-copy"):
            parse_official_deck_hash(".".join(parts))

    def test_invalid_card_token_character_is_rejected(self) -> None:
        with self.assertRaisesRegex(DeckImportError, "invalid character"):
            decode_card_token("ab*c")


@unittest.skipUnless(DATABASE.exists(), "real card database is unavailable")
class OfficialDeckImportIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = CardRepository(DATABASE)

    def test_international_sample_is_exact_and_trainable(self) -> None:
        audit = audit_official_deck_payload(
            INTERNATIONAL_PAYLOAD,
            self.repository,
        )
        self.assertTrue(audit.trainable)
        self.assertEqual(audit.issues, ())
        self.assertEqual(audit.source_kind, "international_official_qr")
        self.assertTrue(all(
            card.coverage == "covered_exact" for card in audit.cards
        ))

    def test_netease_sample_is_exact_and_trainable(self) -> None:
        audit = audit_official_deck_payload(
            NETEASE_PAYLOAD,
            self.repository,
        )
        self.assertTrue(audit.trainable)
        self.assertEqual(audit.source_kind, "netease_official_qr")

    def test_non_exact_card_remains_visible_and_blocks_training(self) -> None:
        report = json.loads(
            DEFAULT_COVERAGE_REPORT.read_text(encoding="utf-8")
        )
        report["classifications"]["10403120"]["coverage"] = "partial"
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "coverage.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False),
                encoding="utf-8",
            )
            audit = audit_official_deck_payload(
                INTERNATIONAL_PAYLOAD,
                self.repository,
                coverage_report=report_path,
            )
        self.assertFalse(audit.trainable)
        self.assertIn(
            "rule_not_exact:10403120:partial",
            audit.issues,
        )

    def test_saved_manifest_loads_as_a_training_deck(self) -> None:
        audit = audit_official_deck_payload(
            INTERNATIONAL_PAYLOAD,
            self.repository,
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = save_deck_manifest(
                audit,
                directory=temporary,
                name="international_forest_fixture",
                display_name="国际服二维码·精灵测试",
            )
            registered = load_registered_training_decks(temporary)
            recipe = registered["international_forest_fixture"]
            self.assertEqual(destination.name, "international_forest_fixture.json")
            self.assertEqual(recipe.class_id, 1)
            self.assertEqual(
                Counter(recipe.card_ids),
                EXPECTED_INTERNATIONAL_COUNTS,
            )
            self.assertEqual(recipe.source_deck_hash, INTERNATIONAL_HASH)


HAS_QR_DEPENDENCIES = (
    importlib.util.find_spec("PIL") is not None
    and importlib.util.find_spec("zxingcpp") is not None
)


@unittest.skipUnless(HAS_QR_DEPENDENCIES, "optional QR dependencies unavailable")
@unittest.skipUnless(INTERNATIONAL_QR_FIXTURE.exists(), "QR fixture unavailable")
class OfficialDeckQRImageTests(unittest.TestCase):
    @unittest.skipUnless(DATABASE.exists(), "real card database is unavailable")
    def test_user_supplied_international_qr_image_imports_offline(self) -> None:
        audit = import_official_deck_qr(
            INTERNATIONAL_QR_FIXTURE,
            CardRepository(DATABASE),
        )
        self.assertEqual(audit.deck.source_hash, INTERNATIONAL_HASH)
        self.assertEqual(
            Counter(audit.deck.card_ids),
            EXPECTED_INTERNATIONAL_COUNTS,
        )
        self.assertTrue(audit.trainable)
        self.assertEqual(audit.source_kind, "international_official_qr")


if __name__ == "__main__":
    unittest.main()
