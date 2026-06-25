from __future__ import annotations

import argparse
import random
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine import DECK_SIZE, ShadowverseEnv


CARD_IDS = (
    10001130,
    10061120,
    10461110,
    10662110,
    10161210,
    10563210,
)
PLAYER_CLASS_ID = 6


def choose_action(env: ShadowverseEnv, rng: random.Random) -> int:
    mask = env.action_mask()
    legal = [index for index, allowed in enumerate(mask) if allowed]
    if env.core.state.pending_choice is not None:
        return rng.choice(legal)
    plays = [
        action
        for action in legal
        if env.PLAY_OFFSET <= action < env.ATTACK_OFFSET
    ]
    if plays:
        return rng.choice(plays)
    mode_plays = [
        action
        for action in legal
        if env.MODE_PLAY_OFFSET <= action < env.SUPER_EVOLVE_OFFSET
    ]
    if mode_plays:
        return rng.choice(mode_plays)
    evolutions = [
        action
        for action in legal
        if env.EVOLVE_OFFSET <= action < env.CHOICE_OFFSET
    ]
    if evolutions:
        return rng.choice(evolutions)
    super_evolutions = [
        action
        for action in legal
        if env.SUPER_EVOLVE_OFFSET <= action < env.ACTION_SIZE
    ]
    if super_evolutions:
        return rng.choice(super_evolutions)
    attacks = [
        action
        for action in legal
        if env.ATTACK_OFFSET <= action < env.EVOLVE_OFFSET
    ]
    if attacks:
        return rng.choice(attacks)
    return env.END_TURN


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mixed-card RL smoke match")
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rl_mixed_match.log"),
    )
    parser.add_argument(
        "--validate-invariants",
        action="store_true",
        help="Run the engine's runtime invariant checks after each command.",
    )
    args = parser.parse_args()

    repository = CardRepository(args.database)
    cards = [repository.get(card_id) for card_id in CARD_IDS]
    deck_a = [cards[index % len(cards)] for index in range(DECK_SIZE)]
    deck_b = [cards[-1 - (index % len(cards))] for index in range(DECK_SIZE)]
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=PLAYER_CLASS_ID,
        class_b=PLAYER_CLASS_ID,
        seed=args.seed,
        card_resolver=repository.get,
        validate_invariants=args.validate_invariants,
    )
    env.reset(seed=args.seed)
    rng = random.Random(args.seed)

    while not env.terminated:
        env.step(choose_action(env, rng))

    transcript = "\n".join(env.logs)
    summary = (
        f"\n最终状态：玩家1生命={env.players[0].health}，"
        f"玩家2生命={env.players[1].health}，"
        f"胜者={env.winner + 1 if env.winner is not None else '平局'}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{transcript}\n{summary}\n", encoding="utf-8")
    print(transcript)
    print(summary)


if __name__ == "__main__":
    main()
