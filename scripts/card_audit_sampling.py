from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.environment import MATCH_SETUP_OFFICIAL, ShadowverseEnv
from swb.rl.baseline_policy import (
    MULLIGAN_POLICY_CURVE,
    select_baseline_action,
)
from swb.rl.checkpoint import load_checkpoint
from swb.rl.fixed_decks import (
    fixed_training_deck_names,
    get_fixed_training_deck,
)
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.seeding import derive_seed
from swb.rl.versioning import stable_json_sha256


DEFAULT_CHECKPOINT = Path(
    "data/checkpoints/observation_nightly_20260729/final/"
    "v4_1_seed_20260801_500k.pt"
)
DEFAULT_MATRIX_OUTPUT = Path(
    "data/reports/card_bug_audit/training_matrix_1000.json"
)
DEFAULT_FULL_OUTPUT = Path(
    "data/reports/card_bug_audit/full_pool_sampling_10000.json"
)
SAMPLING_RANDOM = "random_legal"
SAMPLING_POLICY = "current_policy"


@dataclass(frozen=True)
class GameSpec:
    game_id: int
    sampling_kind: str
    class_a: int
    class_b: int
    deck_a_name: str | None
    deck_b_name: str | None
    deck_seed_a: int
    deck_seed_b: int
    engine_seed: int
    policy_seed: int
    verify_replay: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixed_matrix_specs(
    *,
    master_seed: int,
    random_repeats: int = 15,
    policy_repeats: int = 1,
    verify_replays: bool = True,
) -> list[GameSpec]:
    if random_repeats <= 0 or policy_repeats <= 0:
        raise ValueError("fixed-matrix sampling repeats must be positive")
    names = tuple(sorted(fixed_training_deck_names()))
    specs: list[GameSpec] = []
    for sampling_kind, repeats in (
        (SAMPLING_RANDOM, random_repeats),
        (SAMPLING_POLICY, policy_repeats),
    ):
        for deck_a_name in names:
            deck_a = get_fixed_training_deck(deck_a_name)
            for deck_b_name in names:
                deck_b = get_fixed_training_deck(deck_b_name)
                for repeat in range(repeats):
                    game_id = len(specs)
                    episode_seed = derive_seed(
                        master_seed,
                        "card_audit_fixed_matrix",
                        0 if sampling_kind == SAMPLING_RANDOM else 1,
                        names.index(deck_a_name),
                        names.index(deck_b_name),
                        repeat,
                    )
                    specs.append(GameSpec(
                        game_id=game_id,
                        sampling_kind=sampling_kind,
                        class_a=deck_a.class_id,
                        class_b=deck_b.class_id,
                        deck_a_name=deck_a_name,
                        deck_b_name=deck_b_name,
                        deck_seed_a=derive_seed(episode_seed, "deck", 0),
                        deck_seed_b=derive_seed(episode_seed, "deck", 1),
                        engine_seed=derive_seed(episode_seed, "engine"),
                        policy_seed=derive_seed(episode_seed, "policy"),
                        verify_replay=verify_replays,
                    ))
    return specs


def build_full_pool_specs(
    *,
    master_seed: int,
    games: int = 10_000,
    policy_games_per_matchup: int = 4,
) -> list[GameSpec]:
    if games < 1:
        raise ValueError("full-pool games must be positive")
    if policy_games_per_matchup < 1:
        raise ValueError("policy_games_per_matchup must be positive")
    class_matchups = [
        (class_a, class_b)
        for class_a in range(1, 8)
        for class_b in range(1, 8)
    ]
    policy_games = policy_games_per_matchup * len(class_matchups)
    if policy_games >= games:
        raise ValueError(
            "full-pool game count must exceed the policy stratum"
        )
    random_games = games - policy_games
    random_base, random_extra = divmod(random_games, len(class_matchups))
    specs: list[GameSpec] = []

    def append_spec(
        sampling_kind: str,
        class_a: int,
        class_b: int,
        repeat: int,
        *,
        verify_replay: bool,
    ) -> None:
        game_id = len(specs)
        episode_seed = derive_seed(
            master_seed,
            "card_audit_full_pool",
            0 if sampling_kind == SAMPLING_RANDOM else 1,
            class_a,
            class_b,
            repeat,
        )
        specs.append(GameSpec(
            game_id=game_id,
            sampling_kind=sampling_kind,
            class_a=class_a,
            class_b=class_b,
            deck_a_name=None,
            deck_b_name=None,
            deck_seed_a=derive_seed(episode_seed, "deck", 0),
            deck_seed_b=derive_seed(episode_seed, "deck", 1),
            engine_seed=derive_seed(episode_seed, "engine"),
            policy_seed=derive_seed(episode_seed, "policy"),
            verify_replay=verify_replay,
        ))

    for matchup_index, (class_a, class_b) in enumerate(class_matchups):
        repeats = random_base + int(matchup_index < random_extra)
        for repeat in range(repeats):
            append_spec(
                SAMPLING_RANDOM,
                class_a,
                class_b,
                repeat,
                verify_replay=(repeat == 0),
            )
    for class_a, class_b in class_matchups:
        for repeat in range(policy_games_per_matchup):
            append_spec(
                SAMPLING_POLICY,
                class_a,
                class_b,
                repeat,
                verify_replay=(repeat == 0),
            )
    return specs


class _CurrentPolicySampler:
    def __init__(self, trainer, *, seed: int) -> None:
        self.model = trainer.model
        self.flattener = trainer.flattener
        self.device = trainer.device
        self.observation_version = trainer.config.observation_version
        self.generator = torch.Generator(device=self.device.type)
        self.generator.manual_seed(seed % (2**63 - 1))
        self.hidden = {
            player: self.model.initial_state(1, device=self.device)
            for player in (0, 1)
        }
        self.model.eval()

    def action(
        self,
        env: ShadowverseEnv,
        player_id: int,
        action_mask: Sequence[bool],
    ) -> int:
        observation = env.observation(
            perspective=player_id,
            action_mask=action_mask,
        )
        vector = torch.from_numpy(self.flattener.encode(observation)).to(
            self.device
        ).unsqueeze(0)
        card_indices = torch.from_numpy(
            self.flattener.encode_cards(observation)
        ).to(self.device).unsqueeze(0)
        mask = torch.as_tensor(
            action_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _, hidden = self.model.forward_step(
                vector,
                self.hidden[player_id],
                card_indices,
            )
            probabilities = torch.softmax(
                self.model.masked_logits(logits, mask),
                dim=-1,
            )
            action = torch.multinomial(
                probabilities,
                1,
                generator=self.generator,
            )
        self.hidden[player_id] = hidden.detach()
        return int(action.item())


def _deck_sha256(deck: Sequence[CardDefinition]) -> str:
    return stable_json_sha256(
        tuple(sorted(card.card_id for card in deck))
    )


def _game_decks(
    spec: GameSpec,
    snapshot: WorkerAssetsSnapshot,
) -> tuple[list[CardDefinition], list[CardDefinition]]:
    if spec.deck_a_name is not None:
        deck_a = get_fixed_training_deck(spec.deck_a_name).build(
            snapshot.catalog
        )
        deck_b = get_fixed_training_deck(spec.deck_b_name).build(
            snapshot.catalog
        )
        return deck_a, deck_b
    return (
        snapshot.catalog.sample_deck(
            spec.class_a, random.Random(spec.deck_seed_a)
        ),
        snapshot.catalog.sample_deck(
            spec.class_b, random.Random(spec.deck_seed_b)
        ),
    )


def _run_game(
    spec: GameSpec,
    snapshot: WorkerAssetsSnapshot,
    *,
    rulebook,
    trainer,
    max_game_turns: int,
    max_agent_steps: int,
) -> dict[str, object]:
    deck_a, deck_b = _game_decks(spec, snapshot)
    observation_version = (
        "v1"
        if spec.sampling_kind == SAMPLING_RANDOM
        else trainer.config.observation_version
    )
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=spec.class_a,
        class_b=spec.class_b,
        seed=spec.engine_seed,
        rulebook=rulebook,
        card_resolver=snapshot.catalog.resolve,
        observation_version=observation_version,
        card_vocabulary=(
            None
            if observation_version == "v1"
            else snapshot.catalog.card_vocabulary
        ),
        validate_invariants=True,
        training_mode=True,
        training_event_history_limit=4096,
        max_game_turns=max_game_turns,
        max_agent_steps=max_agent_steps,
        match_setup=MATCH_SETUP_OFFICIAL,
    )
    _, info = env.reset(seed=spec.engine_seed)
    random_policy = random.Random(spec.policy_seed)
    current_policy = (
        None
        if spec.sampling_kind == SAMPLING_RANDOM
        else _CurrentPolicySampler(trainer, seed=spec.policy_seed)
    )
    action_digest = hashlib.sha256()
    action_counts: Counter[str] = Counter()
    mask_checks = 0
    mask_mismatches = 0
    illegal_actions = 0
    while not (env.terminated or env.truncated):
        reported_mask = list(info["action_mask"])
        executable_mask = env.action_mask()
        mask_checks += 1
        if reported_mask != executable_mask:
            mask_mismatches += 1
        if current_policy is None:
            action = select_baseline_action(
                env,
                reported_mask,
                random_policy,
                mulligan_policy=MULLIGAN_POLICY_CURVE,
            )
        else:
            action = current_policy.action(
                env,
                env.decision_player,
                reported_mask,
            )
        if (
            action < 0
            or action >= env.ACTION_SIZE
            or not reported_mask[action]
        ):
            illegal_actions += 1
            raise RuntimeError(
                f"sampling selected illegal action {action}"
            )
        command = env._decode_action(action)
        action_counts[type(command).__name__] += 1
        action_digest.update(action.to_bytes(2, "big"))
        result = env.step(action)
        info = result.info

    core = env.core
    event_counts = Counter(
        event.type.value for event in core.event_history
    )
    placeholder_rows = [
        {
            "turn": event.turn,
            "player_index": event.player_index,
            "card_id": event.card_id,
            "card_name": event.card_name,
            "ability": event.ability.value,
            "event": event.event.value,
        }
        for event in env.placeholder_ability_events
    ]
    final_cards = {
        card.card_id
        for player in core.players
        for card in player.deck
    }
    for player in core.players:
        final_cards.update(card.card_id for card in player.hand)
        final_cards.update(
            entity.definition.card_id for entity in player.board
        )
        final_cards.update(
            item.definition.card_id for item in player.graveyard
        )
        final_cards.update(card.card_id for card in player.banished)
    truncation_reason = None
    if env.truncated:
        if env.turn > max_game_turns:
            truncation_reason = "max_game_turns"
        elif env.agent_steps >= max_agent_steps:
            truncation_reason = "max_agent_steps"
        else:
            truncation_reason = "unknown"
    return {
        **asdict(spec),
        "deck_a_sha256": _deck_sha256(deck_a),
        "deck_b_sha256": _deck_sha256(deck_b),
        "deck_card_ids": sorted({
            card.card_id for card in (*deck_a, *deck_b)
        }),
        "encountered_card_ids": sorted(final_cards),
        "first_player": core.state.first_player,
        "winner": env.winner,
        "turn": env.turn,
        "agent_steps": env.agent_steps,
        "terminated": env.terminated,
        "truncated": env.truncated,
        "truncation_reason": truncation_reason,
        "mask_checks": mask_checks,
        "mask_mismatches": mask_mismatches,
        "illegal_actions": illegal_actions,
        "placeholder_events": len(placeholder_rows),
        "placeholder_event_details": placeholder_rows,
        "action_counts": dict(sorted(action_counts.items())),
        "action_trace_sha256": action_digest.hexdigest(),
        "event_counts": dict(sorted(event_counts.items())),
        "final_fingerprint_sha256": stable_json_sha256(
            core.deterministic_fingerprint()
        ),
    }


def _replay_signature(row: dict[str, object]) -> dict[str, object]:
    keys = (
        "deck_a_sha256",
        "deck_b_sha256",
        "first_player",
        "winner",
        "turn",
        "agent_steps",
        "terminated",
        "truncated",
        "truncation_reason",
        "mask_checks",
        "mask_mismatches",
        "illegal_actions",
        "placeholder_events",
        "action_counts",
        "action_trace_sha256",
        "event_counts",
        "final_fingerprint_sha256",
    )
    return {key: row[key] for key in keys}


def _distribution(rows: list[dict[str, object]]) -> dict[str, object]:
    turn_histogram = Counter(str(row["turn"]) for row in rows)
    step_buckets = Counter()
    for row in rows:
        steps = int(row["agent_steps"])
        bucket = (
            "<=64"
            if steps <= 64
            else (
                "65-128"
                if steps <= 128
                else (
                    "129-256"
                    if steps <= 256
                    else (
                        "257-512"
                        if steps <= 512
                        else (
                            "513-1024"
                            if steps <= 1024
                            else ">1024"
                        )
                    )
                )
            )
        )
        step_buckets[bucket] += 1
    sorted_turns = sorted(int(row["turn"]) for row in rows)

    def percentile(fraction: float) -> int:
        index = min(
            len(sorted_turns) - 1,
            max(0, int((len(sorted_turns) - 1) * fraction)),
        )
        return sorted_turns[index]

    return {
        "turn_histogram": dict(
            sorted(turn_histogram.items(), key=lambda item: int(item[0]))
        ),
        "agent_step_buckets": dict(sorted(step_buckets.items())),
        "turn_min": min(sorted_turns),
        "turn_median": percentile(0.5),
        "turn_p95": percentile(0.95),
        "turn_p99": percentile(0.99),
        "turn_max": max(sorted_turns),
        "agent_steps_mean": (
            sum(int(row["agent_steps"]) for row in rows) / len(rows)
        ),
    }


def run_sampling(
    *,
    database: Path,
    checkpoint: Path,
    specs: list[GameSpec],
    scope: str,
    master_seed: int,
    max_game_turns: int = 200,
    max_agent_steps: int = 2000,
    progress_interval: int = 100,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    if scope not in {"fixed_matrix", "full_pool"}:
        raise ValueError("scope must be fixed_matrix or full_pool")
    if not specs:
        raise ValueError("sampling specs must not be empty")
    snapshot = WorkerAssetsSnapshot.build(CardRepository(database))
    assets = snapshot.load()
    needs_policy = any(
        spec.sampling_kind == SAMPLING_POLICY for spec in specs
    )
    trainer = (
        load_checkpoint(
            checkpoint,
            snapshot,
            device="cpu",
            restore_rng_state=False,
        )
        if needs_policy
        else None
    )
    rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    exceptions: list[dict[str, object]] = []
    replay_failures: list[dict[str, object]] = []
    try:
        for index, spec in enumerate(specs, start=1):
            try:
                row = _run_game(
                    spec,
                    snapshot,
                    rulebook=assets.rulebook,
                    trainer=trainer,
                    max_game_turns=max_game_turns,
                    max_agent_steps=max_agent_steps,
                )
                rows.append(row)
                if spec.verify_replay:
                    replay = _run_game(
                        spec,
                        snapshot,
                        rulebook=assets.rulebook,
                        trainer=trainer,
                        max_game_turns=max_game_turns,
                        max_agent_steps=max_agent_steps,
                    )
                    matched = (
                        _replay_signature(row) == _replay_signature(replay)
                    )
                    replay_rows.append({
                        "game_id": spec.game_id,
                        "sampling_kind": spec.sampling_kind,
                        "engine_seed": spec.engine_seed,
                        "matched": matched,
                    })
                    if not matched:
                        replay_failures.append({
                            "game_id": spec.game_id,
                            "first": _replay_signature(row),
                            "replay": _replay_signature(replay),
                        })
            except Exception as exc:
                exceptions.append({
                    "game_id": spec.game_id,
                    "spec": asdict(spec),
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                })
            if progress_interval > 0 and (
                index % progress_interval == 0 or index == len(specs)
            ):
                progress(
                    f"{scope} progress={index}/{len(specs)} "
                    f"completed={len(rows)} exceptions={len(exceptions)}"
                )
    finally:
        if trainer is not None:
            trainer.close()

    counts_by_sampling = Counter(
        str(row["sampling_kind"]) for row in rows
    )
    ordered_matchups = {
        (
            str(row["deck_a_name"])
            if scope == "fixed_matrix"
            else str(row["class_a"]),
            str(row["deck_b_name"])
            if scope == "fixed_matrix"
            else str(row["class_b"]),
            str(row["sampling_kind"]),
        )
        for row in rows
    }
    matchup_summaries = []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["deck_a_name"])
            if scope == "fixed_matrix"
            else str(row["class_a"]),
            str(row["deck_b_name"])
            if scope == "fixed_matrix"
            else str(row["class_b"]),
            str(row["sampling_kind"]),
        )
        grouped[key].append(row)
    for key in sorted(grouped):
        group = grouped[key]
        matchup_summaries.append({
            "side_a": key[0],
            "side_b": key[1],
            "sampling_kind": key[2],
            "games": len(group),
            "wins": [
                sum(row["winner"] == 0 for row in group),
                sum(row["winner"] == 1 for row in group),
            ],
            "draws": sum(
                row["winner"] is None and not row["truncated"]
                for row in group
            ),
            "truncations": sum(bool(row["truncated"]) for row in group),
            "mean_turn": sum(int(row["turn"]) for row in group) / len(group),
        })

    deck_card_ids = {
        int(card_id)
        for row in rows
        for card_id in row["deck_card_ids"]
    }
    encountered_card_ids = {
        int(card_id)
        for row in rows
        for card_id in row["encountered_card_ids"]
    }
    exact_ids = set(snapshot.catalog.exact_collectible_ids)
    fixed_names = tuple(sorted(fixed_training_deck_names()))
    required_strata = (
        len(fixed_names) ** 2 * 2
        if scope == "fixed_matrix"
        else 7 * 7 * 2
    )
    totals = {
        "mask_checks": sum(int(row["mask_checks"]) for row in rows),
        "mask_mismatches": sum(
            int(row["mask_mismatches"]) for row in rows
        ),
        "illegal_actions": sum(
            int(row["illegal_actions"]) for row in rows
        ),
        "placeholder_events": sum(
            int(row["placeholder_events"]) for row in rows
        ),
        "truncations": sum(bool(row["truncated"]) for row in rows),
        "terminated": sum(bool(row["terminated"]) for row in rows),
    }
    failures = []
    minimum_games = 1000 if scope == "fixed_matrix" else 10_000
    if len(rows) < minimum_games:
        failures.append(
            f"completed games {len(rows)} below required {minimum_games}"
        )
    if set(counts_by_sampling) != {SAMPLING_RANDOM, SAMPLING_POLICY}:
        failures.append("both random_legal and current_policy are required")
    if len(ordered_matchups) != required_strata:
        failures.append(
            f"sampling strata {len(ordered_matchups)} != {required_strata}"
        )
    if scope == "full_pool" and deck_card_ids != exact_ids:
        failures.append(
            f"full-pool decks missed {len(exact_ids - deck_card_ids)} exact cards"
        )
    for key in (
        "mask_mismatches",
        "illegal_actions",
        "placeholder_events",
    ):
        if totals[key]:
            failures.append(f"{key}={totals[key]}")
    if exceptions:
        failures.append(f"game exceptions={len(exceptions)}")
    if replay_failures:
        failures.append(
            f"deterministic replay failures={len(replay_failures)}"
        )

    return {
        "schema_version": 1,
        "report_kind": "swb_card_audit_sampling",
        "scope": scope,
        "configuration": {
            "master_seed": master_seed,
            "requested_games": len(specs),
            "max_game_turns": max_game_turns,
            "max_agent_steps": max_agent_steps,
            "match_setup": MATCH_SETUP_OFFICIAL,
            "mulligan_policy": MULLIGAN_POLICY_CURVE,
            "validate_invariants": True,
            "sampling_kinds": [SAMPLING_RANDOM, SAMPLING_POLICY],
            "current_policy_action_selection": (
                "seeded categorical sample from masked PPO distribution"
            ),
            "policy_device": "cpu",
        },
        "inputs": {
            "database": database.as_posix(),
            "database_sha256": _sha256(database),
            "checkpoint": checkpoint.as_posix(),
            "checkpoint_sha256": _sha256(checkpoint),
            "rulebook_sha256": snapshot.rulebook_sha256,
            "catalog_exact_collectible_count": len(exact_ids),
        },
        "summary": {
            "completed_games": len(rows),
            "counts_by_sampling": dict(sorted(counts_by_sampling.items())),
            "sampling_strata": len(ordered_matchups),
            "required_sampling_strata": required_strata,
            "replay_checks": len(replay_rows),
            "replay_failures": len(replay_failures),
            "deck_exact_card_count": len(deck_card_ids & exact_ids),
            "deck_exact_card_coverage_rate": (
                len(deck_card_ids & exact_ids) / len(exact_ids)
            ),
            "encountered_card_count": len(encountered_card_ids),
            **totals,
            "exception_count": len(exceptions),
            "failure_count": len(failures),
            "passed": not failures,
        },
        "distribution": _distribution(rows),
        "matchups": matchup_summaries,
        "replay_checks": replay_rows,
        "replay_failures": replay_failures,
        "exceptions": exceptions,
        "games": rows,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run checklist 1.12 deterministic card-audit sampling"
    )
    parser.add_argument(
        "--database", type=Path, default=Path("data/cards.sqlite3")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--scope",
        choices=("fixed_matrix", "full_pool"),
        required=True,
    )
    parser.add_argument("--master-seed", type=int, default=120012)
    parser.add_argument("--max-game-turns", type=int, default=200)
    parser.add_argument("--max-agent-steps", type=int, default=2000)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument(
        "--policy-games-per-matchup", type=int, default=4
    )
    parser.add_argument("--random-repeats", type=int, default=15)
    parser.add_argument("--policy-repeats", type=int, default=1)
    args = parser.parse_args()
    if args.scope == "fixed_matrix":
        specs = build_fixed_matrix_specs(
            master_seed=args.master_seed,
            random_repeats=args.random_repeats,
            policy_repeats=args.policy_repeats,
        )
        output = args.output or DEFAULT_MATRIX_OUTPUT
    else:
        specs = build_full_pool_specs(
            master_seed=args.master_seed,
            games=args.games,
            policy_games_per_matchup=args.policy_games_per_matchup,
        )
        output = args.output or DEFAULT_FULL_OUTPUT
    report = run_sampling(
        database=args.database,
        checkpoint=args.checkpoint,
        specs=specs,
        scope=args.scope,
        master_seed=args.master_seed,
        max_game_turns=args.max_game_turns,
        max_agent_steps=args.max_agent_steps,
        progress_interval=args.progress_interval,
        progress=lambda message: print(message, flush=True),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = report["summary"]
    print(
        f"{args.scope} "
        f"acceptance={'pass' if summary['passed'] else 'fail'} "
        f"games={summary['completed_games']} "
        f"random={summary['counts_by_sampling'].get(SAMPLING_RANDOM, 0)} "
        f"policy={summary['counts_by_sampling'].get(SAMPLING_POLICY, 0)} "
        f"truncations={summary['truncations']} "
        f"replays={summary['replay_checks']} "
        f"replay_failures={summary['replay_failures']}"
    )
    if not summary["passed"]:
        raise SystemExit(
            "card-audit sampling failed: "
            + "; ".join(report["failures"])
        )


if __name__ == "__main__":
    main()
