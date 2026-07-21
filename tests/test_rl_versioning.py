from __future__ import annotations

import random
import unittest
from dataclasses import replace
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine.environment import ShadowverseEnv
from swb.rl.catalog import TrainableCardCatalog
from swb.rl.runtime import hash_rule_directory
from swb.rl.versioning import (
    ACTION_LAYOUT_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    ExperimentVersions,
    action_layout_manifest,
    observation_schema_manifest,
    stable_json_sha256,
)


DATABASE = Path("data/cards.sqlite3")


@unittest.skipUnless(DATABASE.exists(), "real card database is unavailable")
class RLVersioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = TrainableCardCatalog.from_repository(CardRepository(DATABASE))
        rng = random.Random(11)
        cls.deck_a = cls.catalog.sample_deck(1, rng)
        cls.deck_b = cls.catalog.sample_deck(1, rng)
        cls.rulebook_sha256 = hash_rule_directory("data/rules")

    def make_env(self, **kwargs) -> ShadowverseEnv:
        env = ShadowverseEnv(
            self.deck_a,
            self.deck_b,
            class_a=1,
            class_b=1,
            seed=11,
            observation_version="v3",
            card_vocabulary=self.catalog.card_vocabulary,
            **kwargs,
        )
        env.reset(seed=11)
        return env

    def test_observation_v3_and_action_111_have_named_stable_versions(self) -> None:
        env = self.make_env()
        observation = observation_schema_manifest(env)
        action = action_layout_manifest(env)
        self.assertEqual(observation["version"], OBSERVATION_SCHEMA_VERSION)
        self.assertEqual(action["version"], ACTION_LAYOUT_VERSION)
        self.assertEqual(action["size"], 111)
        self.assertEqual(action["ranges"][-1]["stop"], 111)
        self.assertEqual(len(stable_json_sha256(observation)), 64)
        self.assertEqual(len(stable_json_sha256(action)), 64)

    def test_versions_are_deck_and_reset_independent(self) -> None:
        first = ExperimentVersions.capture(
            self.make_env(),
            self.catalog,
            rulebook_sha256=self.rulebook_sha256,
        )
        rng = random.Random(99)
        other = ShadowverseEnv(
            self.catalog.sample_deck(2, rng),
            self.catalog.sample_deck(2, rng),
            class_a=2,
            class_b=2,
            seed=99,
            observation_version="v3",
            card_vocabulary=self.catalog.card_vocabulary,
        )
        other.reset(seed=101)
        second = ExperimentVersions.capture(
            other,
            self.catalog,
            rulebook_sha256=self.rulebook_sha256,
        )
        self.assertEqual(first, second)

    def test_open_decklist_schema_is_explicitly_incompatible(self) -> None:
        closed = ExperimentVersions.capture(
            self.make_env(),
            self.catalog,
            rulebook_sha256=self.rulebook_sha256,
        )
        opened = ExperimentVersions.capture(
            self.make_env(open_decklists=True),
            self.catalog,
            rulebook_sha256=self.rulebook_sha256,
        )
        with self.assertRaisesRegex(ValueError, "observation_schema_sha256"):
            closed.assert_compatible(opened)

    def test_mismatch_error_names_every_incompatible_component(self) -> None:
        current = ExperimentVersions.capture(
            self.make_env(),
            self.catalog,
            rulebook_sha256=self.rulebook_sha256,
        )
        stale = replace(
            current,
            catalog_sha256="0" * 64,
            action_layout_sha256="1" * 64,
        )
        with self.assertRaisesRegex(
            ValueError,
            "action_layout_sha256.*catalog_sha256",
        ):
            stale.assert_compatible(current)


if __name__ == "__main__":
    unittest.main()
