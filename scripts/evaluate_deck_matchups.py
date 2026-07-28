from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from swb.db.repository import CardRepository
from swb.engine.environment import (
    MATCH_SETUP_OFFICIAL,
    MATCH_SETUP_VALUES,
    ShadowverseEnv,
)
from swb.rl.checkpoint import load_checkpoint
from swb.rl.evaluation import _RecurrentPolicy, _aggregate_metrics
from swb.rl.fixed_decks import (
    fixed_training_deck_names,
    get_fixed_training_deck,
)
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.seeding import derive_seed
from swb.rl.versioning import ExperimentVersions, stable_json_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one fixed learner deck against named fixed opponent "
            "decks with mirrored policy sides and held-out seeds"
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--learner-deck",
        choices=fixed_training_deck_names(),
        required=True,
    )
    parser.add_argument(
        "--opponent-decks",
        nargs="+",
        choices=fixed_training_deck_names(),
        required=True,
    )
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--max-agent-steps", type=int, default=512)
    parser.add_argument("--master-seed", type=int, default=20260728)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--match-setup",
        choices=sorted(MATCH_SETUP_VALUES),
        default=MATCH_SETUP_OFFICIAL,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/fixed_deck_matchups.json"),
    )
    args = parser.parse_args()
    if args.seed_count <= 0 or args.max_agent_steps <= 0:
        parser.error("--seed-count and --max-agent-steps must be positive")
    if len(set(args.opponent_decks)) != len(args.opponent_decks):
        parser.error("--opponent-decks must not contain duplicates")
    if args.learner_deck in args.opponent_decks:
        parser.error("--learner-deck must not appear in --opponent-decks")
    return args


def _checkpoint_manifest(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    args = parse_args()
    snapshot = WorkerAssetsSnapshot.build(CardRepository(args.database))
    learner_trainer = load_checkpoint(
        args.checkpoint,
        snapshot,
        device=args.device,
        restore_rng_state=False,
    )
    opponent_trainer = load_checkpoint(
        args.opponent_checkpoint,
        snapshot,
        device=args.device,
        restore_rng_state=False,
    )
    learner_recipe = get_fixed_training_deck(args.learner_deck)
    opponent_recipes = tuple(
        get_fixed_training_deck(name)
        for name in args.opponent_decks
    )
    learner_trainer.model.eval()
    opponent_trainer.model.eval()
    results: list[dict[str, object]] = []
    try:
        for opponent_index, opponent_recipe in enumerate(opponent_recipes):
            for seed_index in range(args.seed_count):
                learner_deck = learner_recipe.build(snapshot.catalog)
                opponent_deck = opponent_recipe.build(snapshot.catalog)
                for learner_player in (0, 1):
                    recipes = (
                        (learner_recipe, opponent_recipe)
                        if learner_player == 0
                        else (opponent_recipe, learner_recipe)
                    )
                    decks = (
                        (learner_deck, opponent_deck)
                        if learner_player == 0
                        else (opponent_deck, learner_deck)
                    )
                    engine_seed = derive_seed(
                        args.master_seed,
                        1,
                        opponent_index,
                        seed_index,
                        learner_player,
                    )
                    env = ShadowverseEnv(
                        decks[0],
                        decks[1],
                        class_a=recipes[0].class_id,
                        class_b=recipes[1].class_id,
                        seed=engine_seed,
                        rulebook=learner_trainer.assets.rulebook,
                        card_resolver=learner_trainer.assets.catalog.resolve,
                        observation_version=(
                            learner_trainer.config.observation_version
                        ),
                        card_vocabulary=(
                            learner_trainer.assets.catalog.card_vocabulary
                        ),
                        max_agent_steps=args.max_agent_steps,
                        training_mode=True,
                        training_event_history_limit=4096,
                        validate_invariants=True,
                        match_setup=args.match_setup,
                    )
                    _, info = env.reset(seed=engine_seed)
                    learner_policy = _RecurrentPolicy(
                        learner_trainer.model,
                        learner_trainer.flattener,
                        learner_trainer.device,
                    )
                    opponent_policy = _RecurrentPolicy(
                        opponent_trainer.model,
                        opponent_trainer.flattener,
                        opponent_trainer.device,
                    )
                    learner_policy.reset()
                    opponent_policy.reset()
                    steps = 0
                    mask_checks = 0
                    mask_mismatches = 0
                    illegal_actions = 0
                    while not env.terminated and not env.truncated:
                        player = env.decision_player
                        reported_mask = np.asarray(
                            info["action_mask"],
                            dtype=np.int8,
                        )
                        executable_mask = np.asarray(
                            env.action_mask(),
                            dtype=np.int8,
                        )
                        mask_checks += 1
                        if not np.array_equal(
                            reported_mask,
                            executable_mask,
                        ):
                            mask_mismatches += 1
                        policy = (
                            learner_policy
                            if player == learner_player
                            else opponent_policy
                        )
                        action = policy.action(env, player, reported_mask)
                        if (
                            action < 0
                            or action >= env.ACTION_SIZE
                            or not reported_mask[action]
                        ):
                            illegal_actions += 1
                            raise RuntimeError(
                                "evaluation policy selected an illegal action"
                            )
                        transition = env.step(action)
                        info = transition.info
                        steps += 1
                    score = (
                        0.5
                        if env.winner is None
                        else (
                            1.0
                            if env.winner == learner_player
                            else 0.0
                        )
                    )
                    results.append({
                        "learner_deck": learner_recipe.name,
                        "opponent_deck": opponent_recipe.name,
                        "seed_index": seed_index,
                        "learner_player": learner_player,
                        "score": score,
                        "winner": env.winner,
                        "turn": env.turn,
                        "agent_steps": steps,
                        "terminated": env.terminated,
                        "truncated": env.truncated,
                        "engine_seed": engine_seed,
                        "action_mask_checks": mask_checks,
                        "action_mask_mismatches": mask_mismatches,
                        "illegal_actions": illegal_actions,
                    })
    finally:
        learner_trainer.close()
        opponent_trainer.close()

    per_opponent = {}
    for recipe in opponent_recipes:
        deck_results = [
            result
            for result in results
            if result["opponent_deck"] == recipe.name
        ]
        per_opponent[recipe.name] = {
            "display_name": recipe.display_name,
            **_aggregate_metrics(deck_results),
        }
    versions = ExperimentVersions.capture(
        learner_trainer.env,
        snapshot.catalog,
        rulebook_sha256=snapshot.rulebook_sha256,
    ).to_dict()
    configuration = {
        "master_seed": args.master_seed,
        "seed_count": args.seed_count,
        "mirrored_games": len(results),
        "max_agent_steps": args.max_agent_steps,
        "match_setup": args.match_setup,
        "learner_deck": learner_recipe.manifest(),
        "opponent_decks": [
            recipe.manifest()
            for recipe in opponent_recipes
        ],
    }
    report = {
        "schema_version": 1,
        "purpose": (
            "held-out mirrored fixed-deck matchup evaluation between two "
            "frozen policy checkpoints"
        ),
        "configuration": configuration,
        "versions": versions,
        "learner_checkpoint": _checkpoint_manifest(args.checkpoint),
        "opponent_checkpoint": _checkpoint_manifest(
            args.opponent_checkpoint
        ),
        "metrics": {
            **_aggregate_metrics(results),
            "per_opponent": per_opponent,
        },
        "games": results,
    }
    report["evaluation_suite_sha256"] = stable_json_sha256({
        "configuration": configuration,
        "versions": versions,
        "learner_checkpoint": report["learner_checkpoint"],
        "opponent_checkpoint": report["opponent_checkpoint"],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
