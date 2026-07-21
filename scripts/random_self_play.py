from __future__ import annotations

import argparse
import random
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine import ShadowverseEnv
from swb.engine.card_rules import RuleBook
from swb.rl.catalog import TrainableCardCatalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random legal-action self play")
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--validate-invariants",
        action="store_true",
        help="Run the engine's runtime invariant checks after each command.",
    )
    args = parser.parse_args()

    repository = CardRepository(args.database)
    catalog = TrainableCardCatalog.from_repository(repository)
    rulebook = RuleBook.from_directory(ShadowverseEnv.DEFAULT_RULE_DIRECTORY)
    rng = random.Random(args.seed)
    wins = [0, 0]
    draws = 0
    truncations = 0
    turns = []
    for game in range(args.games):
        class_a = rng.randint(1, 7)
        class_b = rng.randint(1, 7)
        deck_a = catalog.sample_deck(class_a, rng)
        deck_b = catalog.sample_deck(class_b, rng)
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=class_a,
            class_b=class_b,
            seed=args.seed + game,
            rulebook=rulebook,
            card_resolver=catalog.resolve,
            validate_invariants=args.validate_invariants,
        )
        env.reset()
        while not (env.terminated or env.truncated):
            legal = [i for i, allowed in enumerate(env.action_mask()) if allowed]
            result = env.step(rng.choice(legal))
        turns.append(result.info["turn"])
        if env.truncated:
            truncations += 1
        elif env.winner is None:
            draws += 1
        else:
            wins[env.winner] += 1
    print(
        f"games={args.games} wins={wins} draws={draws} truncations={truncations} "
        f"mean_turns={sum(turns) / len(turns):.1f}"
    )


if __name__ == "__main__":
    main()
