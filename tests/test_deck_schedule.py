from __future__ import annotations

import unittest
from collections import Counter

from swb.rl.deck_schedule import (
    DECK_MATCHUP_SCHEDULE_VERSION,
    deck_matchup_for_episode,
    normalize_opponent_decks,
)
from swb.rl.fixed_decks import OFFICIAL_QR_EVOLVE_HAVEN


OPPONENT_DECKS = (
    "international_qr_forest_20260728",
    "international_qr_sword_20260728",
    "international_qr_runecraft_20260728",
    "international_qr_dragon_20260728",
    "international_qr_nightmare_20260728",
    "international_qr_portal_myuu_20260728",
    "international_qr_portal_lishenna_20260728",
)


class DeckMatchupScheduleTests(unittest.TestCase):
    def test_cycle_balances_every_opponent_and_learner_side(self) -> None:
        matchups = [
            deck_matchup_for_episode(
                OFFICIAL_QR_EVOLVE_HAVEN,
                OPPONENT_DECKS,
                episode_id,
            )
            for episode_id in range(14)
        ]
        self.assertEqual(DECK_MATCHUP_SCHEDULE_VERSION, 1)
        self.assertEqual(
            Counter(matchup.opponent_deck.name for matchup in matchups),
            Counter({name: 2 for name in OPPONENT_DECKS}),
        )
        self.assertEqual(
            Counter(matchup.learner_player for matchup in matchups),
            Counter({0: 7, 1: 7}),
        )
        for matchup in matchups:
            with self.subTest(episode_id=matchup.episode_id):
                self.assertEqual(
                    matchup.decks[matchup.learner_player].name,
                    OFFICIAL_QR_EVOLVE_HAVEN,
                )
                self.assertEqual(
                    matchup.decks[1 - matchup.learner_player].name,
                    matchup.opponent_deck.name,
                )
        self.assertEqual(
            deck_matchup_for_episode(
                OFFICIAL_QR_EVOLVE_HAVEN,
                OPPONENT_DECKS,
                14,
            ).manifest(),
            matchups[0].manifest() | {"episode_id": 14},
        )

    def test_invalid_opponent_pools_are_rejected(self) -> None:
        for opponents in (
            (),
            (OPPONENT_DECKS[0], OPPONENT_DECKS[0]),
            (OFFICIAL_QR_EVOLVE_HAVEN,),
            ("missing",),
        ):
            with self.subTest(opponents=opponents):
                with self.assertRaises(ValueError):
                    normalize_opponent_decks(
                        OFFICIAL_QR_EVOLVE_HAVEN,
                        opponents,
                    )
        with self.assertRaises(ValueError):
            deck_matchup_for_episode(
                OFFICIAL_QR_EVOLVE_HAVEN,
                OPPONENT_DECKS,
                -1,
            )


if __name__ == "__main__":
    unittest.main()
