from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from scripts.scan_training_speed_stage_2_4 import DEFAULT_CHECKPOINT
from swb.db.repository import CardRepository
from swb.rl.evaluation import EvaluationConfig, evaluate
from swb.rl.fixed_decks import OFFICIAL_QR_EVOLVE_HAVEN
from swb.rl.ppo import PPOConfig, PPOTrainer
from swb.rl.runtime import WorkerAssetsSnapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_A_GATE = Path(
    "data/reports/training_speed/"
    "stage_2_7_a_padded_compute_001_gate.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/"
    "stage_2_7_b_batched_learner_001_learning.json"
)
SEEDS = (20260811, 20260812, 20260813)
TARGET_AGENT_STEPS = 4096


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _training_config(checkpoint: Path) -> PPOConfig:
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    config = PPOConfig(**payload["trainer"]["config"])
    return replace(
        config,
        rollout_workers=6,
        rollout_worker_torch_threads=2,
        central_inference_batch_wait_seconds=0.001,
        profile_ipc_timing=False,
        profile_central_timing=False,
        profile_learner_timing=False,
    )


def _run_seed(
    snapshot: WorkerAssetsSnapshot,
    config: PPOConfig,
    *,
    seed: int,
    batched: bool,
) -> dict[str, object]:
    trainer = PPOTrainer(
        snapshot,
        master_seed=seed,
        config=config,
        device="cuda",
    )
    trainer._batched_v41_learner = batched
    initial_sha256 = _state_sha256(trainer.model)
    updates = []
    truncated_episodes = 0
    abnormal_exits = 0
    started = time.perf_counter()
    try:
        while trainer.agent_steps < TARGET_AGENT_STEPS:
            records, bootstrap, _ = trainer.collect_rollout()
            truncated_episodes += int(
                trainer.last_collect_timing[
                    "worker_truncated_episode_count"
                ]
            )
            update_started = time.perf_counter()
            metrics = trainer.update(records, bootstrap)
            update_seconds = time.perf_counter() - update_started
            finite_metrics = all(
                math.isfinite(float(value))
                for value in metrics.values()
            )
            if not finite_metrics:
                abnormal_exits += 1
            updates.append({
                "agent_steps": int(trainer.agent_steps),
                "records": len(records),
                "update_seconds": update_seconds,
                "metrics": metrics,
                "finite_metrics": finite_metrics,
            })
        evaluation = evaluate(
            trainer,
            snapshot,
            EvaluationConfig(
                master_seed=20260821,
                seed_count=2,
                max_agent_steps=256,
                opponent_kind="random_legal",
                class_ids=(6,),
                training_deck=OFFICIAL_QR_EVOLVE_HAVEN,
            ),
        )
        final_finite = all(
            bool(torch.isfinite(parameter).all())
            for parameter in trainer.model.parameters()
        )
        return {
            "seed": seed,
            "batched": batched,
            "initial_model_sha256": initial_sha256,
            "final_model_sha256": _state_sha256(trainer.model),
            "completed_agent_steps": int(trainer.agent_steps),
            "updates": updates,
            "elapsed_seconds": time.perf_counter() - started,
            "finite_model": final_finite,
            "truncated_training_episodes": truncated_episodes,
            "abnormal_exit_count": abnormal_exits,
            "evaluation": {
                "games": evaluation["metrics"]["games"],
                "win_rate": evaluation["metrics"]["win_rate"],
                "terminated": evaluation["metrics"]["terminated"],
                "truncated": evaluation["metrics"]["truncated"],
                "illegal_actions": evaluation["metrics"][
                    "illegal_actions"
                ],
                "action_mask_mismatches": evaluation["metrics"][
                    "action_mask_mismatches"
                ],
                "suite_sha256": evaluation[
                    "evaluation_suite_sha256"
                ],
            },
        }
    finally:
        trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess 2.7 B-BATCHED-LEARNER-001 learning"
    )
    parser.add_argument(
        "--database", type=Path, default=DEFAULT_DATABASE
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--a-gate", type=Path, default=DEFAULT_A_GATE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    checkpoint = _path(args.checkpoint)
    checkpoint_before = {
        "sha256": _sha256(checkpoint),
        "size": checkpoint.stat().st_size,
        "mtime_ns": checkpoint.stat().st_mtime_ns,
    }
    snapshot = WorkerAssetsSnapshot.build(
        CardRepository(_path(args.database))
    )
    config = _training_config(checkpoint)
    runs = []
    for seed in SEEDS:
        for batched in (False, True):
            run = _run_seed(
                snapshot,
                config,
                seed=seed,
                batched=batched,
            )
            runs.append(run)
            print(json.dumps({
                "seed": seed,
                "batched": batched,
                "agent_steps": run["completed_agent_steps"],
                "elapsed_seconds": run["elapsed_seconds"],
                "win_rate": run["evaluation"]["win_rate"],
            }, sort_keys=True), flush=True)

    baseline = [run for run in runs if not run["batched"]]
    candidate = [run for run in runs if run["batched"]]
    paired_initial_state = all(
        next(
            run["initial_model_sha256"]
            for run in baseline if run["seed"] == seed
        )
        == next(
            run["initial_model_sha256"]
            for run in candidate if run["seed"] == seed
        )
        for seed in SEEDS
    )
    baseline_win_rates = [
        float(run["evaluation"]["win_rate"]) for run in baseline
    ]
    candidate_win_rates = [
        float(run["evaluation"]["win_rate"]) for run in candidate
    ]
    all_stable = all(
        run["finite_model"]
        and run["abnormal_exit_count"] == 0
        and run["truncated_training_episodes"] == 0
        and run["evaluation"]["truncated"] == 0
        and run["evaluation"]["illegal_actions"] == 0
        and run["evaluation"]["action_mask_mismatches"] == 0
        and all(
            update["finite_metrics"] for update in run["updates"]
        )
        for run in runs
    )
    learning_non_degradation = (
        statistics.median(candidate_win_rates)
        >= statistics.median(baseline_win_rates) - 0.25
    )
    baseline_update_times = [
        float(update["update_seconds"])
        for run in baseline for update in run["updates"]
    ]
    candidate_update_times = [
        float(update["update_seconds"])
        for run in candidate for update in run["updates"]
    ]
    checkpoint_after = {
        "sha256": _sha256(checkpoint),
        "size": checkpoint.stat().st_size,
        "mtime_ns": checkpoint.stat().st_mtime_ns,
    }
    report = {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_7_"
            "b_batched_learner_001_learning"
        ),
        "checklist_section": "2.7",
        "candidate": "B-BATCHED-LEARNER-001",
        "classification": "B",
        "configuration": {
            "seeds": list(SEEDS),
            "target_agent_steps": TARGET_AGENT_STEPS,
            "rollout_steps": config.rollout_steps,
            "rollout_workers": config.rollout_workers,
            "worker_torch_threads": (
                config.rollout_worker_torch_threads
            ),
            "central_inference_batch_wait_seconds": (
                config.central_inference_batch_wait_seconds
            ),
            "sequence_length": config.sequence_length,
            "minibatch_sequences": config.minibatch_sequences,
            "update_epochs": config.update_epochs,
            "policy_architecture": config.policy_architecture,
            "observation_version": config.observation_version,
            "model_dim": config.model_dim,
            "transformer_layers": config.transformer_layers,
            "evaluation_seed_count": 2,
            "evaluation_max_agent_steps": 256,
            "evaluation_opponent": "random_legal",
        },
        "runs": runs,
        "summary": {
            "paired_initial_model_state": paired_initial_state,
            "all_numeric_and_runtime_stable": all_stable,
            "baseline_evaluation_win_rates": baseline_win_rates,
            "candidate_evaluation_win_rates": candidate_win_rates,
            "baseline_evaluation_median": statistics.median(
                baseline_win_rates
            ),
            "candidate_evaluation_median": statistics.median(
                candidate_win_rates
            ),
            "allowed_median_regression": 0.25,
            "learning_non_degradation_passed": (
                learning_non_degradation
            ),
            "baseline_update_median_seconds": statistics.median(
                baseline_update_times
            ),
            "candidate_update_median_seconds": statistics.median(
                candidate_update_times
            ),
            "update_relative_reduction": (
                statistics.median(baseline_update_times)
                - statistics.median(candidate_update_times)
            ) / statistics.median(baseline_update_times),
        },
        "decision": {
            "advance_to_end_to_end": (
                paired_initial_state
                and all_stable
                and learning_non_degradation
            ),
            "default_enabled": False,
            "checkpoint_resume_test": (
                "tests.test_checkpoint.CheckpointTests."
                "test_batched_v41_learner_resume_next_update_"
                "drift_is_bounded"
            ),
        },
        "checkpoint": {
            "before": checkpoint_before,
            "after": checkpoint_after,
            "unchanged": checkpoint_before == checkpoint_after,
        },
        "sources": {
            "a_gate": {
                "path": args.a_gate.as_posix(),
                "sha256": _sha256(args.a_gate),
            },
            "checkpoint": {
                "path": args.checkpoint.as_posix(),
                "sha256": _sha256(args.checkpoint),
            },
        },
        "passed": (
            checkpoint_before == checkpoint_after
            and paired_initial_state
            and all_stable
            and learning_non_degradation
        ),
    }
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": args.output.as_posix(),
        "passed": report["passed"],
        "advance_to_end_to_end": report["decision"][
            "advance_to_end_to_end"
        ],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
