from __future__ import annotations

import argparse
import random
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine import DECK_SIZE, ShadowverseEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random legal-action self play")
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    repository = CardRepository(args.database)
    rng = random.Random(args.seed)
    wins = [0, 0]
    draws = 0
    turns = []
    for game in range(args.games):
        class_a = rng.randint(1, 7)
        class_b = rng.randint(1, 7)
        pool_a = repository.training_pool(class_id=class_a)
        pool_b = repository.training_pool(class_id=class_b)
        if not pool_a or not pool_b:
            raise RuntimeError("No supported collectible cards are available")
        deck_a = [rng.choice(pool_a) for _ in range(DECK_SIZE)]
        deck_b = [rng.choice(pool_b) for _ in range(DECK_SIZE)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=class_a,
            class_b=class_b,
            seed=args.seed + game,
            card_resolver=repository.get,
        )
        env.reset()
        while not env.terminated:
            legal = [i for i, allowed in enumerate(env.action_mask()) if allowed]
            result = env.step(rng.choice(legal))
        turns.append(result.info["turn"])
        if env.winner is None:
            draws += 1
        else:
            wins[env.winner] += 1
    print(
        f"games={args.games} wins={wins} draws={draws} "
        f"mean_turns={sum(turns) / len(turns):.1f}"
    )


if __name__ == "__main__":
    main()
