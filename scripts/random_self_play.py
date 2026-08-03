from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine import ShadowverseEnv
from swb.engine.card_rules import RuleBook
from swb.engine.environment import (
    MATCH_SETUP_OFFICIAL,
    MATCH_SETUP_VALUES,
)
from swb.rl.catalog import TrainableCardCatalog
from swb.rl.baseline_policy import (
    MULLIGAN_POLICY_RANDOM,
    MULLIGAN_POLICY_VALUES,
    select_baseline_action,
)


def official_acceptance_failures(report: dict[str, object]) -> list[str]:
    games = int(report["games"])
    first_players = list(report["first_players"])
    failures: list[str] = []
    if report["match_setup"] != MATCH_SETUP_OFFICIAL:
        failures.append("match_setup is not official")
    if int(report["mulligan_games_entered"]) != games:
        failures.append("not every game entered mulligan")
    if int(report["mulligan_games_completed"]) != games:
        failures.append("not every game completed mulligan")
    if int(report["mulligan_decisions"]) != games * 2:
        failures.append("each game must resolve exactly two mulligan decisions")
    if games >= 2 and (first_players[0] == 0 or first_players[1] == 0):
        failures.append("both players were not sampled as first player")
    if abs(first_players[0] - first_players[1]) > max(2, games // 10):
        failures.append("first-player distribution exceeds 10% imbalance")
    if int(report["extra_pp_uses"]) == 0:
        failures.append("Extra PP was never exercised")
    if int(report["illegal_actions"]) != 0:
        failures.append("illegal actions occurred")
    if int(report["action_mask_mismatches"]) != 0:
        failures.append("reported and executable action masks disagreed")
    if int(report["truncations"]) != 0:
        failures.append("one or more games truncated")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random legal-action self play")
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--match-setup",
        choices=sorted(MATCH_SETUP_VALUES),
        default=MATCH_SETUP_OFFICIAL,
    )
    parser.add_argument(
        "--mulligan-policy",
        choices=sorted(MULLIGAN_POLICY_VALUES),
        default=MULLIGAN_POLICY_RANDOM,
    )
    parser.add_argument("--curve-keep-cost", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--assert-official-acceptance",
        action="store_true",
        help="Fail unless the official setup acceptance criteria all pass.",
    )
    parser.add_argument(
        "--validate-invariants",
        action="store_true",
        help="Run the engine's runtime invariant checks after each command.",
    )
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("games must be positive")
    if args.curve_keep_cost < 0:
        parser.error("curve-keep-cost must be non-negative")

    repository = CardRepository(args.database)
    catalog = TrainableCardCatalog.from_repository(repository)
    rulebook = RuleBook.from_directory(ShadowverseEnv.DEFAULT_RULE_DIRECTORY)
    rng = random.Random(args.seed)
    wins = [0, 0]
    draws = 0
    truncations = 0
    turns = []
    first_players = [0, 0]
    mulligan_games_entered = 0
    mulligan_games_completed = 0
    mulligan_decisions = 0
    mulligan_cards_replaced = 0
    extra_pp_uses = 0
    extra_pp_games = 0
    extra_pp_use_turns: list[int] = []
    action_mask_mismatches = 0
    illegal_actions = 0
    for game in range(args.games):
        class_a = rng.randint(1, 7)
        class_b = rng.randint(1, 7)
        deck_a = catalog.sample_deck(class_a, rng)
        deck_b = catalog.sample_deck(class_b, rng)
        actions: list[int] = []
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=class_a,
            class_b=class_b,
            seed=args.seed + game,
            rulebook=rulebook,
            card_resolver=catalog.resolve,
            validate_invariants=args.validate_invariants,
            match_setup=args.match_setup,
        )
        _, info = env.reset()
        first_players[int(info["first_player"])] += 1
        entered_mulligan = info["phase"] == "mulligan"
        mulligan_games_entered += int(entered_mulligan)
        game_extra_pp_uses = 0
        while not (env.terminated or env.truncated):
            reported_mask = list(info["action_mask"])
            executable_mask = env.action_mask()
            if reported_mask != executable_mask:
                action_mask_mismatches += 1
            action = select_baseline_action(
                env,
                reported_mask,
                rng,
                mulligan_policy=args.mulligan_policy,
                curve_keep_cost=args.curve_keep_cost,
            )
            if info["phase"] == "mulligan":
                mulligan_decisions += 1
                mulligan_cards_replaced += (
                    action - env.CHOICE_OFFSET
                ).bit_count()
            if not reported_mask[action]:
                illegal_actions += 1
            actions.append(action)
            extra_pp_uses_before = sum(
                player.extra_pp_uses for player in env.players
            )
            turn_before_action = env.turn
            try:
                result = env.step(action)
            except Exception as exc:
                failure = {
                    "schema_version": 1,
                    "status": "failed",
                    "games_requested": args.games,
                    "seed": args.seed,
                    "match_setup": args.match_setup,
                    "mulligan_policy": args.mulligan_policy,
                    "failure": {
                        "game_index": game,
                        "game_seed": args.seed + game,
                        "class_a": class_a,
                        "class_b": class_b,
                        "deck_a": [card.card_id for card in deck_a],
                        "deck_b": [card.card_id for card in deck_b],
                        "action_index": len(actions) - 1,
                        "action": action,
                        "actions": actions,
                        "turn": env.turn,
                        "active_player": env.current_player,
                        "phase": info["phase"],
                        "reported_action_mask": reported_mask,
                        "state_fingerprint": (
                            env.core.deterministic_fingerprint()
                        ),
                        "boards": [
                            [
                                {
                                    "entity_id": entity.entity_id,
                                    "card_id": entity.definition.card_id,
                                    "attack": getattr(entity, "attack", None),
                                    "health": getattr(entity, "health", None),
                                    "max_health": getattr(
                                        entity, "max_health", None
                                    ),
                                }
                                for entity in player.board
                            ]
                            for player in env.players
                        ],
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    },
                }
                if args.output is not None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(
                            failure,
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                raise RuntimeError(
                    "self-play failure at "
                    f"game={game} game_seed={args.seed + game} "
                    f"action_index={len(actions) - 1} action={action}: {exc}"
                ) from exc
            committed_extra_pp = (
                sum(player.extra_pp_uses for player in env.players)
                - extra_pp_uses_before
            )
            if committed_extra_pp:
                extra_pp_uses += committed_extra_pp
                game_extra_pp_uses += committed_extra_pp
                extra_pp_use_turns.extend(
                    [turn_before_action] * committed_extra_pp
                )
            info = result.info
        mulligan_games_completed += int(
            entered_mulligan
            and all(info["mulligan_completed"])
            and info["phase"] != "mulligan"
        )
        extra_pp_games += int(game_extra_pp_uses > 0)
        turns.append(result.info["turn"])
        if env.truncated:
            truncations += 1
        elif env.winner is None:
            draws += 1
        else:
            wins[env.winner] += 1
    report = {
        "schema_version": 1,
        "games": args.games,
        "seed": args.seed,
        "match_setup": args.match_setup,
        "mulligan_policy": args.mulligan_policy,
        "curve_keep_cost": args.curve_keep_cost,
        "wins": wins,
        "draws": draws,
        "truncations": truncations,
        "mean_turns": sum(turns) / len(turns),
        "first_players": first_players,
        "mulligan_games_entered": mulligan_games_entered,
        "mulligan_games_completed": mulligan_games_completed,
        "mulligan_decisions": mulligan_decisions,
        "mulligan_cards_replaced": mulligan_cards_replaced,
        "extra_pp_uses": extra_pp_uses,
        "extra_pp_games": extra_pp_games,
        "extra_pp_use_turns": extra_pp_use_turns,
        "illegal_actions": illegal_actions,
        "action_mask_mismatches": action_mask_mismatches,
    }
    failures = official_acceptance_failures(report)
    report["official_acceptance_passed"] = not failures
    report["official_acceptance_failures"] = failures
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"games={args.games} wins={wins} draws={draws} "
        f"truncations={truncations} mean_turns={report['mean_turns']:.1f} "
        f"first_players={first_players} mulligans={mulligan_decisions} "
        f"replaced={mulligan_cards_replaced} extra_pp_uses={extra_pp_uses} "
        f"mask_mismatches={action_mask_mismatches} "
        f"acceptance={'pass' if not failures else 'fail'}"
    )
    if args.assert_official_acceptance and failures:
        raise SystemExit(
            "official self-play acceptance failed: " + "; ".join(failures)
        )


if __name__ == "__main__":
    main()
