from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.commands import EndTurn
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import GameConfig, GameEngine


def card(card_id: int) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=1,
        card_type="随从",
        attack=1,
        life=2,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def decks() -> tuple[list[CardDefinition], list[CardDefinition]]:
    return (
        [card(100 + index) for index in range(40)],
        [card(200 + index) for index in range(40)],
    )


class TrainingModeTests(unittest.TestCase):
    def test_core_transition_keeps_all_events_when_history_is_bounded(self) -> None:
        deck_a, deck_b = decks()
        engine = GameEngine(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=4,
            config=GameConfig(
                retain_text_logs=False,
                event_history_limit=1,
            ),
        )
        engine.reset(seed=4)
        transition = engine.apply(EndTurn(0))
        self.assertGreater(len(transition.events), len(engine.event_history))
        self.assertLessEqual(len(engine.event_history), 1)
        self.assertEqual(engine.logs, [])

    def test_environment_training_mode_bounds_diagnostics(self) -> None:
        deck_a, deck_b = decks()
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=5,
            training_mode=True,
            training_event_history_limit=8,
            max_agent_steps=40,
        )
        env.reset(seed=5)
        for _ in range(12):
            env.step(env.END_TURN)
        self.assertEqual(env.logs, [])
        self.assertLessEqual(len(env._core.event_history), 8)

    def test_normal_mode_retains_text_log_and_unbounded_history(self) -> None:
        deck_a, deck_b = decks()
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=6,
            training_mode=False,
        )
        env.reset(seed=6)
        env.step(env.END_TURN)
        self.assertTrue(env.logs)
        self.assertIsNone(env._core.config.event_history_limit)

    def test_training_mode_replay_remains_deterministic(self) -> None:
        fingerprints = []
        for _ in range(2):
            deck_a, deck_b = decks()
            engine = GameEngine(
                deck_a,
                deck_b,
                class_a=1,
                class_b=1,
                seed=9,
                config=GameConfig(
                    retain_text_logs=False,
                    event_history_limit=16,
                ),
            )
            engine.reset(seed=9)
            for player in (0, 1, 0, 1):
                engine.apply(EndTurn(player))
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])


if __name__ == "__main__":
    unittest.main()
