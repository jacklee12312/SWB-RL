from __future__ import annotations

import pickle
import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import Unit


def card(
    card_id: int,
    *,
    attack: int | None = 1,
    life: int | None = 2,
    cost: int = 1,
    card_type: str = "随从",
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"card-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack,
        life=life,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def decks() -> tuple[list[CardDefinition], list[CardDefinition]]:
    return (
        [card(100 + index) for index in range(40)],
        [card(200 + index) for index in range(40)],
    )


class SnapshotCloneTests(unittest.TestCase):
    def make_engine(self, *, rulebook=None) -> GameEngine:
        deck_a, deck_b = decks()
        engine = GameEngine(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=17,
            rulebook=rulebook,
        )
        engine.reset(seed=17)
        return engine

    def test_snapshot_is_serializable_and_restore_replays_exactly(self) -> None:
        engine = self.make_engine()
        snapshot = pickle.loads(pickle.dumps(engine.snapshot()))
        first = engine.apply(EndTurn(0))
        first_fingerprint = engine.deterministic_fingerprint()
        engine.restore(snapshot)
        second = engine.apply(EndTurn(0))
        self.assertEqual(first.events, second.events)
        self.assertEqual(first.winner, second.winner)
        self.assertEqual(first_fingerprint, engine.deterministic_fingerprint())

    def test_clone_has_no_shared_mutable_state_and_illegal_branch_isolated(self) -> None:
        engine = self.make_engine()
        original = engine.deterministic_fingerprint()
        branch = engine.clone()
        branch.players[0].health = 3
        branch.players[0].hand.clear()
        self.assertEqual(engine.deterministic_fingerprint(), original)
        with self.assertRaises(IllegalCommand):
            branch.apply(EndTurn(1))
        self.assertEqual(engine.deterministic_fingerprint(), original)

    def test_pending_choice_effect_stack_and_rng_resume_identically(self) -> None:
        rulebook = RuleBook.from_directory("data/rules")
        spell = card(
            10041310,
            attack=None,
            life=None,
            card_type="法术",
        )
        engine = GameEngine(
            [spell] * 40,
            [card(2)] * 40,
            class_a=1,
            class_b=1,
            seed=1,
            rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(
            card(99),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = spell
        engine.apply(PlayCard(0, 0))
        self.assertTrue(engine.state.effect_stack)
        self.assertIsNotNone(engine.state.pending_choice)

        branch = engine.clone()
        original_choice = next(
            command for command in engine.legal_commands() if isinstance(command, Choose)
        )
        branch_choice = next(
            command for command in branch.legal_commands() if isinstance(command, Choose)
        )
        first = engine.apply(original_choice)
        second = branch.apply(branch_choice)
        self.assertEqual(first.events, second.events)
        self.assertEqual(
            engine.deterministic_fingerprint(),
            branch.deterministic_fingerprint(),
        )

    def test_environment_clone_preserves_page_limits_and_cache_independence(self) -> None:
        deck_a, deck_b = decks()
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=5,
            observation_version="v3",
            card_vocabulary=tuple((*range(100, 140), *range(200, 240))),
            training_mode=True,
            max_agent_steps=10,
        )
        env.reset(seed=5)
        branch = env.clone()
        self.assertEqual(
            env._core.deterministic_fingerprint(),
            branch._core.deterministic_fingerprint(),
        )
        branch.step(branch.END_TURN)
        self.assertNotEqual(branch.state_version, env.state_version)
        self.assertNotEqual(
            env._core.deterministic_fingerprint(),
            branch._core.deterministic_fingerprint(),
        )


if __name__ == "__main__":
    unittest.main()
