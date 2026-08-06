from __future__ import annotations

import argparse
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine import (
    CLASS_NAMES,
    DECK_SIZE,
    MATCH_SETUP_OFFICIAL,
    ShadowverseEnv,
)
from swb.engine.card_rules import RuleBook
from swb.rl.catalog import TrainableCardCatalog


def choose_action(env: ShadowverseEnv) -> int:
    mask = env.action_mask()

    if env.core.state.pending_choice is not None:
        choices = [action for action, allowed in enumerate(mask) if allowed]
        if choices:
            return choices[0]
    if mask[env.USE_EXTRA_PP]:
        return env.USE_EXTRA_PP

    playable = [
        action
        for action in range(env.PLAY_OFFSET, env.ATTACK_OFFSET)
        if mask[action]
    ]
    if playable:
        player = env.players[env.current_player]
        return max(playable, key=lambda action: player.hand[action - env.PLAY_OFFSET].cost)

    mode_plays = [
        action
        for action in range(env.MODE_PLAY_OFFSET, env.SUPER_EVOLVE_OFFSET)
        if mask[action]
    ]
    if mode_plays:
        return mode_plays[0]

    evolutions = [
        action
        for action in range(env.EVOLVE_OFFSET, env.CHOICE_OFFSET)
        if mask[action]
    ]
    if evolutions:
        player = env.players[env.current_player]
        return max(
            evolutions,
            key=lambda action: player.board[action - env.EVOLVE_OFFSET].attack,
        )

    super_evolutions = [
        action
        for action in range(env.SUPER_EVOLVE_OFFSET, env.USE_EXTRA_PP)
        if mask[action]
    ]
    if super_evolutions:
        player = env.players[env.current_player]
        return max(
            super_evolutions,
            key=lambda action: player.board[action - env.SUPER_EVOLVE_OFFSET].attack,
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
    catalog = TrainableCardCatalog.from_repository(repository)
    rulebook = RuleBook.from_directory(ShadowverseEnv.DEFAULT_RULE_DIRECTORY)
    pool_a = catalog.pool(class_id=args.class_a)
    pool_b = catalog.pool(class_id=args.class_b)
    deck_a = [pool_a[index % len(pool_a)] for index in range(DECK_SIZE)]
    deck_b = [pool_b[-1 - (index % len(pool_b))] for index in range(DECK_SIZE)]
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=args.class_a,
        class_b=args.class_b,
        seed=args.seed,
        rulebook=rulebook,
        card_resolver=catalog.resolve,
        match_setup=MATCH_SETUP_OFFICIAL,
    )
    env.reset(seed=args.seed)

    while not (env.terminated or env.truncated):
        env.step(choose_action(env))

    transcript = "\n".join(env.logs)
    summary = (
        f"\n最终状态：玩家1({CLASS_NAMES[args.class_a]})生命={env.players[0].health}，"
        f"玩家2({CLASS_NAMES[args.class_b]})生命={env.players[1].health}，"
        f"结果={'截断' if env.truncated else (env.winner + 1 if env.winner is not None else '平局')}"
    )
    print(transcript)
    print(summary)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{transcript}\n{summary}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
