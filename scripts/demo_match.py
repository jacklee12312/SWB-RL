from __future__ import annotations

import argparse
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine import CLASS_NAMES, DECK_SIZE, ShadowverseEnv


def choose_action(env: ShadowverseEnv) -> int:
    mask = env.action_mask()

    choices = [
        action
        for action in range(env.CHOICE_OFFSET, env.ACTION_SIZE)
        if mask[action]
    ]
    if choices:
        return choices[0]

    playable = [
        action
        for action in range(env.PLAY_OFFSET, env.ATTACK_OFFSET)
        if mask[action]
    ]
    if playable:
        player = env.players[env.current_player]
        return max(playable, key=lambda action: player.hand[action - env.PLAY_OFFSET].cost)

    evolutions = [
        action
        for action in range(env.EVOLVE_OFFSET, env.ACTION_SIZE)
        if mask[action]
    ]
    if evolutions:
        player = env.players[env.current_player]
        return max(
            evolutions,
            key=lambda action: player.board[action - env.EVOLVE_OFFSET].attack,
        )

    attacks = [
        action
        for action in range(env.ATTACK_OFFSET, env.EVOLVE_OFFSET)
        if mask[action]
    ]
    if attacks:
        leader_attacks = [
            action
            for action in attacks
            if (action - env.ATTACK_OFFSET) % env.TARGETS_PER_ATTACKER == 0
        ]
        return leader_attacks[0] if leader_attacks else attacks[0]
    return env.END_TURN


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and print one deterministic match")
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--class-a", type=int, choices=range(1, 8), default=1)
    parser.add_argument("--class-b", type=int, choices=range(1, 8), default=1)
    args = parser.parse_args()

    repository = CardRepository(args.database)
    pool_a = repository.training_pool(class_id=args.class_a)
    pool_b = repository.training_pool(class_id=args.class_b)
    deck_a = [pool_a[index % len(pool_a)] for index in range(DECK_SIZE)]
    deck_b = [pool_b[-1 - (index % len(pool_b))] for index in range(DECK_SIZE)]
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=args.class_a,
        class_b=args.class_b,
        seed=args.seed,
    )
    env.reset(seed=args.seed)

    while not env.terminated:
        env.step(choose_action(env))

    transcript = "\n".join(env.logs)
    summary = (
        f"\n最终状态：玩家1({CLASS_NAMES[args.class_a]})生命={env.players[0].health}，"
        f"玩家2({CLASS_NAMES[args.class_b]})生命={env.players[1].health}，"
        f"胜者={env.winner + 1 if env.winner is not None else '平局'}"
    )
    print(transcript)
    print(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{transcript}\n{summary}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
