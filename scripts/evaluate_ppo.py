from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine.environment import MATCH_SETUP_OFFICIAL, MATCH_SETUP_VALUES
from swb.rl.checkpoint import load_checkpoint
from swb.rl.class_schedule import ALL_CLASS_IDS
from swb.rl.evaluation import EvaluationConfig, evaluate
from swb.rl.fixed_decks import (
    fixed_training_deck_names,
    get_fixed_training_deck,
)
from swb.rl.runtime import WorkerAssetsSnapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-seed mirrored evaluation for a PPO checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--max-agent-steps", type=int, default=512)
    parser.add_argument("--master-seed", type=int, default=20260721)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--match-setup",
        choices=sorted(MATCH_SETUP_VALUES),
        default=MATCH_SETUP_OFFICIAL,
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="playable class IDs; fixed-deck evaluation selects its class",
    )
    parser.add_argument(
        "--training-deck",
        choices=fixed_training_deck_names(),
        help="mirror one named fixed training deck on both sides",
    )
    parser.add_argument(
        "--full-matchup-matrix",
        action="store_true",
        help=(
            "evaluate every ordered learner/opponent class pair and mirror "
            "both player positions"
        ),
    )
    parser.add_argument(
        "--opponent",
        choices=("current", "random_legal", "fixed", "historical"),
        default="random_legal",
    )
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/ppo_evaluation.json"),
    )
    args = parser.parse_args()
    class_ids = (
        tuple(ALL_CLASS_IDS)
        if args.classes is None
        else tuple(args.classes)
    )
    if args.training_deck is not None:
        if args.full_matchup_matrix:
            parser.error(
                "--full-matchup-matrix cannot use one --training-deck"
            )
        fixed_deck = get_fixed_training_deck(args.training_deck)
        if (
            args.classes is not None
            and class_ids != (fixed_deck.class_id,)
        ):
            parser.error(
                f"--training-deck {args.training_deck} requires "
                f"--classes {fixed_deck.class_id}"
            )
        class_ids = (fixed_deck.class_id,)
    snapshot = WorkerAssetsSnapshot.build(CardRepository(args.database))
    trainer = load_checkpoint(
        args.checkpoint,
        snapshot,
        device=args.device,
        restore_rng_state=False,
    )
    report = evaluate(
        trainer,
        snapshot,
        EvaluationConfig(
            master_seed=args.master_seed,
            seed_count=args.seed_count,
            max_agent_steps=args.max_agent_steps,
            opponent_kind=args.opponent,
            opponent_checkpoint=(
                None
                if args.opponent_checkpoint is None
                else str(args.opponent_checkpoint)
            ),
            class_ids=class_ids,
            match_setup=args.match_setup,
            training_deck=args.training_deck,
            full_matchup_matrix=args.full_matchup_matrix,
        ),
    )
    report["checkpoint"] = {
        "path": str(args.checkpoint),
        "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
