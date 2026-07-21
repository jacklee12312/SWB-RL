from __future__ import annotations

import argparse
import json
from pathlib import Path

from swb.db.repository import CardRepository
from swb.rl.checkpoint import load_checkpoint
from swb.rl.evaluation import EvaluationConfig, evaluate
from swb.rl.runtime import WorkerAssetsSnapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-seed mirrored evaluation for a PPO checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--max-agent-steps", type=int, default=512)
    parser.add_argument("--master-seed", type=int, default=20260721)
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
    snapshot = WorkerAssetsSnapshot.build(CardRepository(args.database))
    trainer = load_checkpoint(args.checkpoint, snapshot)
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
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
