from __future__ import annotations

import argparse
import json
from pathlib import Path

from swb.db.repository import CardRepository
from swb.rl.class_schedule import ALL_CLASS_IDS
from swb.rl.distribution import build_training_distribution_audit
from swb.rl.runtime import WorkerAssetsSnapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the deterministic class and exact-card distribution used by "
            "PPO episode sampling"
        )
    )
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--master-seed", type=int, default=20260721)
    parser.add_argument("--episodes", type=int, default=98)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(ALL_CLASS_IDS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/rl_training_distribution.json"),
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.workers <= 0:
        parser.error("episodes and workers must be positive")

    snapshot = WorkerAssetsSnapshot.build(CardRepository(args.database))
    report = build_training_distribution_audit(
        snapshot.catalog,
        master_seed=args.master_seed,
        episode_count=args.episodes,
        worker_count=args.workers,
        class_ids=tuple(args.classes),
        rulebook_sha256=snapshot.rulebook_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
