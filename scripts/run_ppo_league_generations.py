from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

import torch

from scripts.report_ppo_league_baseline import ROOT, render_json
from scripts.report_ppo_league_evolving import (
    ACTIVE_FRACTION,
    ACTIVE_MATRIX_OUTPUT,
    ACTIVE_PAIR_GAMES,
    ACTIVE_SEEDS,
    ARCHIVE_FRACTION,
    ARCHIVE_SELECTION_OUTPUT,
    CONTRACT_OUTPUT,
    GENERATION_ZERO_MANIFEST,
    LINEAGE_MANIFEST_DIRECTORY,
    SCHEDULE_OUTPUT,
    _relative,
    _selection_audit,
    _sha256_file,
    _source,
    _validate_evaluation,
)
from scripts.report_ppo_league_generation import _model_values_sha256
from scripts.report_ppo_league_meta_game import (
    PROFILE_THRESHOLDS,
    evaluate_mixture,
    interquartile_mean,
    paired_bootstrap,
    solve_zero_sum_meta_strategy,
)
from swb.rl.pfsp import PayoffEstimate, compute_pfsp_distribution
from swb.rl.versioning import stable_json_sha256


DEFAULT_REPORT_ROOT = Path("data/reports/league_training/generations")
DEFAULT_CHECKPOINT_ROOT = Path("data/checkpoints/ppo_evolving_league")
GENERATION_ZERO_ACTIVE = tuple(f"seed_{seed}_1m" for seed in ACTIVE_SEEDS)
META_SUPPORT_SEEDS = (20260903, 20260906, 20260907)
MAX_POSITIVE_ARCHIVE = 26
VALIDATION_NONDEGRADATION_SCORE = 0.48
MINIMUM_GENERATION_GAIN = 0.03
MINIMUM_EXPLOITABILITY_PROXY_IMPROVEMENT = 0.10
GENERATION_BOOTSTRAP_REPLICATES = 2000
ARCHIVE_SCREEN_SEED_COUNT = 1
ARCHIVE_CONFIRM_SEED_COUNT = 2
ARCHIVE_CONFIRM_SCREEN_SCORE = 0.45
FORGOTTEN_PREVIOUS_SCORE = 0.70
FORGOTTEN_CURRENT_SCORE = 0.40


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_json(path: str | Path) -> dict[str, object]:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _atomic_json(path: str | Path, payload: Mapping[str, object]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_bytes(render_json(dict(payload)))
    os.replace(temporary, output)


def _append_log(path: str | Path, message: str) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with output.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def _generation_directory(report_root: Path, generation: int) -> Path:
    return report_root / f"generation_{generation:03d}"


def _checkpoint_directory(checkpoint_root: Path, generation: int) -> Path:
    return checkpoint_root / f"generation_{generation:03d}"


def _lineage_manifest_path(
    report_root: Path,
    source_generation: int,
    seed: int,
) -> Path:
    if source_generation == 0:
        return LINEAGE_MANIFEST_DIRECTORY / f"seed_{seed}_1m.json"
    return (
        _generation_directory(report_root, source_generation)
        / "lineage_manifests"
        / f"seed_{seed}.json"
    )


def _population_path(report_root: Path, generation: int) -> Path:
    if generation == 0:
        return GENERATION_ZERO_MANIFEST
    return _generation_directory(report_root, generation) / "population_manifest.json"


def _role_entries(
    manifest: Mapping[str, object],
    generation: int,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]], list[Mapping[str, object]]]:
    rows = list(manifest["entries"])  # type: ignore[arg-type]
    if generation == 0:
        active = [row for row in rows if row["role"] == "candidate_final"]
        archive = [row for row in rows if row["role"] == "self_history"]
        anchors = [row for row in rows if row["role"] == "anchor_only"]
    else:
        active = [row for row in rows if row["role"] == "active_latest"]
        archive = [row for row in rows if row["role"] == "historical_archive"]
        anchors = [row for row in rows if row["role"] == "evaluation_anchor"]
    active.sort(key=lambda row: int(row["policy_seed"]))
    archive.sort(key=lambda row: str(row["opponent_id"]))
    anchors.sort(key=lambda row: str(row["opponent_id"]))
    if len(active) != 6:
        raise ValueError(f"generation {generation} requires six active entries")
    return active, archive, anchors


def _checkpoint_entry(
    checkpoint: str | Path,
    *,
    generation: int,
    role: str,
    rules_version: str,
) -> dict[str, object]:
    path = _repo_path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    trainer = payload["trainer"]
    versions = payload["versions"]
    seed = int(trainer["master_seed"])
    result = {
        "opponent_id": f"seed_{seed}_g{generation:03d}",
        "checkpoint_path": _relative(path),
        "checkpoint_sha256": _sha256_file(path),
        "model_values_sha256": _model_values_sha256(payload["model_state"]),
        "policy_seed": seed,
        "training_steps": int(trainer["agent_steps"]),
        "generation": generation,
        "role": role,
        "rules_version": rules_version,
        "policy_architecture": str(trainer["config"]["policy_architecture"]),
        "versions_sha256": stable_json_sha256(versions),
        "training_eligible": False,
        "sampling_weight": 0.0,
    }
    del payload
    return result


def _checkpoint_agent_steps(path: str | Path) -> int:
    payload = torch.load(_repo_path(path), map_location="cpu", weights_only=False)
    steps = int(payload["trainer"]["agent_steps"])
    del payload
    return steps


def _periodic_checkpoint_candidates(checkpoint: Path) -> list[Path]:
    directory = checkpoint.parent / f"{checkpoint.stem}_checkpoints"
    resolved_directory = _repo_path(directory)
    if not resolved_directory.is_dir():
        return []
    return sorted(
        resolved_directory.glob("step_*.pt"),
        key=lambda path: path.name,
        reverse=True,
    )


def _latest_resume_checkpoint(checkpoint: Path) -> tuple[Path | None, int]:
    candidates = []
    if _repo_path(checkpoint).is_file():
        candidates.append(checkpoint)
    candidates.extend(_periodic_checkpoint_candidates(checkpoint))
    best_path = None
    best_steps = -1
    for candidate in candidates:
        steps = _checkpoint_agent_steps(candidate)
        if steps > best_steps:
            best_path = candidate
            best_steps = steps
    return best_path, best_steps


def _training_command(
    *,
    resume: Path,
    checkpoint: Path,
    metrics: Path,
    target_agent_steps: int,
    lineage_manifest: Path | None,
    replace_opponent_pool: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.train_ppo",
        "--resume",
        str(_repo_path(resume)),
        "--total-agent-steps",
        str(target_agent_steps),
        "--device",
        "cuda",
        "--rollout-workers",
        "7",
        "--rollout-worker-threads",
        "2",
        "--central-inference-batch-wait-ms",
        "3.0",
        "--opponent-model-cache-size",
        "7",
        "--opponent-model-cache-max-mib",
        "512",
        "--checkpoint",
        str(_repo_path(checkpoint)),
        "--checkpoint-interval-agent-steps",
        "100000",
        "--metrics-output",
        str(_repo_path(metrics)),
        "--monitor-system",
        "--resume-runtime-overrides",
    ]
    if replace_opponent_pool:
        if lineage_manifest is None:
            raise ValueError("fresh generation training requires lineage manifest")
        command.extend([
            "--resume-opponent-pool-overrides",
            "--opponent-current-weight",
            "0",
            "--opponent-random-weight",
            "0",
            "--opponent-fixed-weight",
            "0",
            "--opponent-historical-weight",
            "0",
            "--opponent-external-manifest",
            str(_repo_path(lineage_manifest)),
            "--opponent-external-weight",
            "1",
            "--opponent-batching-mode",
            "episode_seed_clustered",
            "--opponent-max-history",
            "4",
            "--opponent-snapshot-interval-steps",
            "250000",
        ])
    return command


def _evaluation_command(
    *,
    learner: Path,
    opponent: Path,
    output: Path,
    master_seed: int,
    seed_count: int = 2,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.evaluate_ppo",
        str(_repo_path(learner)),
        "--seed-count",
        str(seed_count),
        "--max-agent-steps",
        "512",
        "--master-seed",
        str(master_seed),
        "--device",
        "cuda",
        "--classes",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "--full-matchup-matrix",
        "--opponent",
        "historical",
        "--opponent-checkpoint",
        str(_repo_path(opponent)),
        "--output",
        str(_repo_path(output)),
    ]


def _archive_candidates(
    population: Mapping[str, object],
    generation: int,
) -> list[Mapping[str, object]]:
    active, archive, _ = _role_entries(population, generation)
    by_hash: dict[str, Mapping[str, object]] = {}
    for entry in (*archive, *active):
        by_hash[str(entry["checkpoint_sha256"])] = entry
    return sorted(
        by_hash.values(),
        key=lambda entry: (
            int(entry["generation"]),
            int(entry["policy_seed"]),
            int(entry["training_steps"]),
            str(entry["opponent_id"]),
        ),
    )


def _generation_zero_archive_scores() -> dict[tuple[int, str], float]:
    population = _read_json(GENERATION_ZERO_MANIFEST)
    entries = {
        str(entry["opponent_id"]): entry
        for entry in population["entries"]  # type: ignore[union-attr]
    }
    report = _read_json(ARCHIVE_SELECTION_OUTPUT)
    scores: dict[tuple[int, str], float] = {}
    for row in report["active_archive_baselines"]:  # type: ignore[union-attr]
        active = entries[str(row["active_policy_id"])]
        archive = entries[str(row["archive_policy_id"])]
        key = (
            int(active["policy_seed"]),
            str(archive["checkpoint_sha256"]),
        )
        scores[key] = max(scores.get(key, 0.0), float(row["score_rate"]))
    return scores


def _previous_archive_scores(
    report_root: Path,
    target_generation: int,
) -> dict[tuple[int, str], float]:
    scores = _generation_zero_archive_scores()
    for generation in range(1, target_generation):
        path = _generation_directory(report_root, generation) / "archive_audit.json"
        if not _repo_path(path).is_file():
            continue
        report = _read_json(path)
        for row in report["pairs"]:  # type: ignore[union-attr]
            key = (
                int(row["policy_seed"]),
                str(row["archive_checkpoint_sha256"]),
            )
            scores[key] = max(
                scores.get(key, 0.0),
                float(row["final_score_rate"]),
            )
    return scores


def _previous_forgotten_hashes(
    report_root: Path,
    target_generation: int,
) -> set[str]:
    forgotten: set[str] = set()
    for generation in range(1, target_generation):
        path = _generation_directory(report_root, generation) / "archive_audit.json"
        if not _repo_path(path).is_file():
            continue
        report = _read_json(path)
        forgotten.update(
            str(value)
            for value in report["summary"]["forgotten_checkpoint_sha256"]
        )
    return forgotten


def _archive_evaluation_path(
    audit_directory: Path,
    stage: str,
    policy_seed: int,
    opponent_id: str,
) -> Path:
    safe_id = opponent_id.replace("/", "_").replace("\\", "_")
    return audit_directory / stage / f"seed_{policy_seed}__vs__{safe_id}.json"


def _archive_confirmation_required(
    previous_score: float | None,
    screen_score: float,
) -> bool:
    return bool(
        previous_score is not None
        and previous_score >= FORGOTTEN_PREVIOUS_SCORE
        and screen_score <= ARCHIVE_CONFIRM_SCREEN_SCORE
    )


def _is_forgotten(
    previous_score: float | None,
    current_score: float,
    *,
    confirmed: bool,
) -> bool:
    return bool(
        confirmed
        and previous_score is not None
        and previous_score >= FORGOTTEN_PREVIOUS_SCORE
        and current_score < FORGOTTEN_CURRENT_SCORE
    )


def _build_archive_audit_report(
    *,
    generation: int,
    master_seed: int,
    active_entries: Sequence[Mapping[str, object]],
    archive_entries: Sequence[Mapping[str, object]],
    audit_directory: Path,
    previous_scores: Mapping[tuple[int, str], float],
) -> dict[str, object]:
    pairs = []
    forgotten_hashes: set[str] = set()
    for active in sorted(active_entries, key=lambda row: int(row["policy_seed"])):
        seed = int(active["policy_seed"])
        for archive in archive_entries:
            opponent_id = str(archive["opponent_id"])
            checkpoint_hash = str(archive["checkpoint_sha256"])
            screen_path = _archive_evaluation_path(
                audit_directory,
                "screen",
                seed,
                opponent_id,
            )
            screen, screen_source = _validate_evaluation(
                screen_path,
                learner_entry=active,
                opponent_entry=archive,
                master_seed=master_seed,
                games=98,
            )
            previous = previous_scores.get((seed, checkpoint_hash))
            screen_score = float(screen["score_rate"])
            confirmation_source = None
            final = screen
            confirmation_required = _archive_confirmation_required(
                previous,
                screen_score,
            )
            if confirmation_required:
                confirm_path = _archive_evaluation_path(
                    audit_directory,
                    "confirm",
                    seed,
                    opponent_id,
                )
                final, confirmation_source = _validate_evaluation(
                    confirm_path,
                    learner_entry=active,
                    opponent_entry=archive,
                    master_seed=master_seed,
                    games=ACTIVE_PAIR_GAMES,
                )
            final_score = float(final["score_rate"])
            forgotten = _is_forgotten(
                previous,
                final_score,
                confirmed=confirmation_required,
            )
            if forgotten:
                forgotten_hashes.add(checkpoint_hash)
            pairs.append({
                "policy_seed": seed,
                "learner_opponent_id": active["opponent_id"],
                "archive_opponent_id": opponent_id,
                "archive_checkpoint_sha256": checkpoint_hash,
                "previous_best_score_rate": previous,
                "screen_score_rate": screen_score,
                "screen_report": screen_source,
                "confirmation_required": confirmation_required,
                "confirmation_report": confirmation_source,
                "final_score_rate": final_score,
                "forgotten": forgotten,
            })
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_archive_forgetting_audit",
        "immutable": True,
        "data_partition": "pfsp_tuning",
        "generation": generation,
        "master_seed": master_seed,
        "protocol": {
            "screen_games_per_pair": 98,
            "confirmation_games_per_pair": ACTIVE_PAIR_GAMES,
            "confirmation_rule": (
                f"previous >= {FORGOTTEN_PREVIOUS_SCORE:.2f} and "
                f"screen <= {ARCHIVE_CONFIRM_SCREEN_SCORE:.2f}"
            ),
            "forgotten_rule": (
                f"previous >= {FORGOTTEN_PREVIOUS_SCORE:.2f} and "
                f"confirmed current < {FORGOTTEN_CURRENT_SCORE:.2f}"
            ),
        },
        "pairs": pairs,
        "summary": {
            "active_lineage_count": len(active_entries),
            "archive_checkpoint_count": len(archive_entries),
            "screen_pair_count": len(pairs),
            "confirmation_pair_count": sum(
                bool(row["confirmation_required"]) for row in pairs
            ),
            "forgotten_pair_count": sum(bool(row["forgotten"]) for row in pairs),
            "forgotten_checkpoint_sha256": sorted(forgotten_hashes),
        },
        "audit": {
            "illegal_actions": 0,
            "action_mask_mismatches": 0,
            "truncated": 0,
        },
    }


def _run_command(command: Sequence[str], stdout: Path, stderr: Path) -> None:
    stdout_path = _repo_path(stdout)
    stderr_path = _repo_path(stderr)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open(
        "a", encoding="utf-8"
    ) as err:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            stdout=out,
            stderr=err,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            + " ".join(command)
        )


def _validate_training_report(
    path: Path,
    *,
    target_agent_steps: int,
    manifest_path: Path,
) -> dict[str, object]:
    payload = _read_json(path)
    diagnostics = payload["league_diagnostics"]
    if not diagnostics["completed_without_exception"]:
        raise ValueError(f"training did not complete: {path}")
    if int(payload["completed_agent_steps"]) < target_agent_steps:
        raise ValueError(f"training missed target steps: {path}")
    for name in (
        "truncated_episodes",
        "illegal_action_errors",
        "action_mask_mismatch_errors",
    ):
        if int(diagnostics[name]) != 0:
            raise ValueError(f"training safety failure {name}: {path}")
    external = diagnostics["external_opponent_manifest"]
    if external["file_sha256"] != _sha256_file(manifest_path):
        raise ValueError(f"training used wrong lineage manifest: {path}")
    checkpoint = _repo_path(payload["checkpoint"])
    if not checkpoint.is_file():
        raise ValueError(f"training checkpoint is missing: {checkpoint}")
    if _checkpoint_agent_steps(checkpoint) < target_agent_steps:
        raise ValueError(f"training checkpoint missed target steps: {checkpoint}")
    for name, value in payload["final_metrics"].items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise ValueError(f"non-finite training metric {name}: {path}")
    return payload


def _source_active_entry_map(
    population: Mapping[str, object],
    generation: int,
) -> dict[int, Mapping[str, object]]:
    active, _, _ = _role_entries(population, generation)
    return {int(row["policy_seed"]): row for row in active}


def _build_active_payoff(
    *,
    generation: int,
    active_entries: Sequence[Mapping[str, object]],
    evaluation_directory: Path,
    master_seed: int,
) -> dict[str, object]:
    by_seed = {int(row["policy_seed"]): row for row in active_entries}
    ids = [str(by_seed[seed]["opponent_id"]) for seed in ACTIVE_SEEDS]
    matrix = {
        left: {right: (0.5 if left == right else None) for right in ids}
        for left in ids
    }
    intervals = {
        left: {right: ([0.5, 0.5] if left == right else None) for right in ids}
        for left in ids
    }
    pairs = []
    for left_index, left_seed in enumerate(ACTIVE_SEEDS):
        left = by_seed[left_seed]
        left_id = str(left["opponent_id"])
        for right_seed in ACTIVE_SEEDS[left_index + 1 :]:
            right = by_seed[right_seed]
            right_id = str(right["opponent_id"])
            path = evaluation_directory / f"seed_{left_seed}__vs__seed_{right_seed}.json"
            result, source = _validate_evaluation(
                path,
                learner_entry=left,
                opponent_entry=right,
                master_seed=master_seed,
                games=ACTIVE_PAIR_GAMES,
            )
            score = float(result["score_rate"])
            interval = result["confidence_interval_95"]
            matrix[left_id][right_id] = score
            matrix[right_id][left_id] = 1.0 - score
            intervals[left_id][right_id] = interval
            intervals[right_id][left_id] = [
                1.0 - interval[1],
                1.0 - interval[0],
            ]
            pairs.append({
                "left": left_id,
                "right": right_id,
                **result,
                "report": source,
            })
    uniform_worst_case = min(
        sum(float(matrix[left][right]) for left in ids) / len(ids)
        for right in ids
    )
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_active_payoff_matrix",
        "immutable": True,
        "data_partition": "pfsp_tuning",
        "generation": generation,
        "master_seed": master_seed,
        "active_policy_ids": ids,
        "games_per_unique_pair": ACTIVE_PAIR_GAMES,
        "unique_pair_count": len(pairs),
        "total_games": len(pairs) * ACTIVE_PAIR_GAMES,
        "score_matrix": matrix,
        "confidence_interval_95_matrix": intervals,
        "pairs": pairs,
        "population_metrics": {
            "uniform_mixture_worst_case_score": uniform_worst_case,
        },
        "audit": {
            "complete": len(pairs) == 15,
            "antisymmetric": True,
            "illegal_actions": 0,
            "action_mask_mismatches": 0,
            "truncated": 0,
        },
    }


def _build_validation_report(
    *,
    generation: int,
    active_entries: Sequence[Mapping[str, object]],
    validation_directory: Path,
    master_seed: int,
) -> dict[str, object]:
    generation_zero = _read_json(GENERATION_ZERO_MANIFEST)
    base_active, _, _ = _role_entries(generation_zero, 0)
    base_by_seed = {int(row["policy_seed"]): row for row in base_active}
    current_by_seed = {int(row["policy_seed"]): row for row in active_entries}
    lineages = []
    for seed in ACTIVE_SEEDS:
        current = current_by_seed[seed]
        baseline = base_by_seed[seed]
        path = validation_directory / f"seed_{seed}__vs__generation_000_parent.json"
        result, source = _validate_evaluation(
            path,
            learner_entry=current,
            opponent_entry=baseline,
            master_seed=master_seed,
            games=ACTIVE_PAIR_GAMES,
        )
        lineages.append({
            "policy_seed": seed,
            "current_policy_id": current["opponent_id"],
            "generation_000_parent_id": baseline["opponent_id"],
            **result,
            "report": source,
        })
    scores = [float(row["score_rate"]) for row in lineages]
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_generation_validation",
        "immutable": True,
        "data_partition": "pfsp_tuning",
        "generation": generation,
        "master_seed": master_seed,
        "lineages": lineages,
        "summary": {
            "mean_score_rate": sum(scores) / len(scores),
            "minimum_score_rate": min(scores),
            "nondegraded_lineages": sum(
                score >= VALIDATION_NONDEGRADATION_SCORE for score in scores
            ),
            "required_nondegraded_lineages": 4,
            "minimum_allowed_score_rate": VALIDATION_NONDEGRADATION_SCORE,
        },
        "audit": {
            "illegal_actions": 0,
            "action_mask_mismatches": 0,
            "truncated": 0,
        },
    }


def _build_population_summary(
    *,
    active_payoff: Mapping[str, object],
    validation: Mapping[str, object],
    bootstrap_replicates: int = GENERATION_BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    model_ids = list(active_payoff["active_policy_ids"])
    score_matrix = [
        [
            float(active_payoff["score_matrix"][left][right])
            for right in model_ids
        ]
        for left in model_ids
    ]
    payoff_matrix = [
        [2.0 * score - 1.0 for score in row]
        for row in score_matrix
    ]
    nash = solve_zero_sum_meta_strategy(payoff_matrix)
    uniform = evaluate_mixture(
        payoff_matrix,
        [1.0 / len(model_ids)] * len(model_ids),
    )
    directed_scores = [
        score_matrix[row][column]
        for row in range(len(model_ids))
        for column in range(len(model_ids))
        if row != column
    ]
    per_model = {}
    for row, model_id in enumerate(model_ids):
        scores = [
            score_matrix[row][column]
            for column in range(len(model_ids))
            if column != row
        ]
        per_model[model_id] = {
            "median_score_rate": statistics.median(scores),
            "iqm_score_rate": interquartile_mean(scores),
            "worst_score_rate": min(scores),
            "performance_profile": {
                f"score_at_least_{threshold:.2f}": (
                    sum(score >= threshold for score in scores) / len(scores)
                )
                for threshold in PROFILE_THRESHOLDS
            },
        }
    pairwise_profile_distances = []
    for left in range(len(model_ids)):
        for right in range(left + 1, len(model_ids)):
            distance = sum(
                abs(score_matrix[left][column] - score_matrix[right][column])
                for column in range(len(model_ids))
            ) / len(model_ids)
            pairwise_profile_distances.append({
                "left": model_ids[left],
                "right": model_ids[right],
                "mean_absolute_payoff_profile_distance": distance,
            })
    class_cells: dict[tuple[str, int, int], list[float]] = {}
    pairwise_results = []
    for pair in active_payoff["pairs"]:
        left = str(pair["left"])
        right = str(pair["right"])
        pairwise_results.append({
            "learner_model": left,
            "opponent_model": right,
            "report": pair["report"],
        })
        raw = _read_json(pair["report"]["path"])
        for game in raw["games"]:
            learner_class = int(game["learner_class_id"])
            opponent_class = int(game["opponent_class_id"])
            score = float(game["score"])
            class_cells.setdefault(
                (left, learner_class, opponent_class), []
            ).append(score)
            class_cells.setdefault(
                (right, opponent_class, learner_class), []
            ).append(1.0 - score)
    worst_class_cells = sorted(
        (
            {
                "model_id": key[0],
                "learner_class_id": key[1],
                "opponent_class_id": key[2],
                "games": len(scores),
                "score_rate": sum(scores) / len(scores),
            }
            for key, scores in class_cells.items()
        ),
        key=lambda row: (
            float(row["score_rate"]),
            str(row["model_id"]),
            int(row["learner_class_id"]),
            int(row["opponent_class_id"]),
        ),
    )[:12]
    bootstrap_input = {
        "models": {"candidate_ids": model_ids},
        "payoff_matrix": {
            "model_ids": model_ids,
            "antisymmetric_payoff_matrix": payoff_matrix,
        },
        "pairwise_results": pairwise_results,
    }
    validation_scores = [
        float(row["score_rate"])
        for row in validation["lineages"]
    ]
    return {
        "model_ids": model_ids,
        "per_model": per_model,
        "population_directed_score_median": statistics.median(directed_scores),
        "population_directed_score_iqm": interquartile_mean(directed_scores),
        "population_worst_case_score": min(
            sum(score_matrix[row][column] for row in range(len(model_ids)))
            / len(model_ids)
            for column in range(len(model_ids))
        ),
        "validation_score_median": statistics.median(validation_scores),
        "validation_score_iqm": interquartile_mean(validation_scores),
        "nash_mixture": nash,
        "uniform_mixture": uniform,
        "paired_bootstrap": paired_bootstrap(
            bootstrap_input,
            model_ids,
            seed=int(active_payoff["master_seed"]),
            replicates=bootstrap_replicates,
        ),
        "payoff_profile_diversity": {
            "metric": "mean_absolute_score_profile_distance",
            "pairwise": pairwise_profile_distances,
            "mean": statistics.mean(
                float(row["mean_absolute_payoff_profile_distance"])
                for row in pairwise_profile_distances
            ),
        },
        "worst_class_cells": worst_class_cells,
    }


def _archive_priority(
    entry: Mapping[str, object],
    *,
    previous_active_hashes: set[str],
    forgotten_hashes: set[str],
) -> tuple[object, ...]:
    checkpoint_hash = str(entry["checkpoint_sha256"])
    seed = int(entry["policy_seed"])
    if checkpoint_hash in previous_active_hashes:
        return (0, seed, str(entry["opponent_id"]))
    if checkpoint_hash in forgotten_hashes:
        return (1, -int(entry["generation"]), seed, str(entry["opponent_id"]))
    if int(entry["generation"]) == 0 and "step_" in str(entry["opponent_id"]):
        return (2, seed, int(entry["training_steps"]), str(entry["opponent_id"]))
    support_rank = (
        META_SUPPORT_SEEDS.index(seed)
        if seed in META_SUPPORT_SEEDS
        else len(META_SUPPORT_SEEDS)
    )
    return (
        3,
        support_rank,
        -int(entry["generation"]),
        seed,
        str(entry["opponent_id"]),
    )


def _build_population_and_lineage_manifests(
    *,
    source_population: Mapping[str, object],
    source_generation: int,
    target_generation: int,
    active_entries: Sequence[Mapping[str, object]],
    active_payoff: Mapping[str, object],
    forgotten_hashes: set[str] | None = None,
) -> tuple[dict[str, object], dict[int, dict[str, object]], dict[str, object]]:
    source_active, source_archive, source_anchors = _role_entries(
        source_population, source_generation
    )
    forgotten = set() if forgotten_hashes is None else set(forgotten_hashes)
    archive_by_hash: dict[str, dict[str, object]] = {}
    for row in (*source_archive, *source_active):
        entry = dict(row)
        entry["role"] = "historical_archive"
        entry["training_eligible"] = False
        entry["sampling_weight"] = 0.0
        archive_by_hash[str(entry["checkpoint_sha256"])] = entry
    previous_active_hashes = {
        str(entry["checkpoint_sha256"]) for entry in source_active
    }
    archive_candidates = sorted(
        archive_by_hash.values(),
        key=lambda entry: _archive_priority(
            entry,
            previous_active_hashes=previous_active_hashes,
            forgotten_hashes=forgotten,
        ),
    )
    selected_archive = archive_candidates[:MAX_POSITIVE_ARCHIVE]
    selected_hashes = {
        str(entry["checkpoint_sha256"]) for entry in selected_archive
    }
    canonical_active = []
    for row in active_entries:
        entry = dict(row)
        entry["role"] = "active_latest"
        entry["training_eligible"] = False
        entry["sampling_weight"] = 0.0
        canonical_active.append(entry)
    canonical_anchors = []
    for row in source_anchors:
        entry = dict(row)
        entry["role"] = "evaluation_anchor"
        entry["training_eligible"] = False
        entry["sampling_weight"] = 0.0
        canonical_anchors.append(entry)
    canonical_entries = [
        *canonical_active,
        *archive_candidates,
        *canonical_anchors,
    ]
    canonical_entries.sort(key=lambda row: str(row["opponent_id"]))
    population = {
        "schema_version": 1,
        "report_kind": "ppo_league_population_manifest",
        "immutable": True,
        "path_base": "repository_root",
        "generation": target_generation,
        "selection_mode": "learner_specific_hard",
        "contract": source_population["contract"],
        "source_generation": source_generation,
        "entries": canonical_entries,
        "summary": {
            "active_latest_count": len(canonical_active),
            "historical_archive_count": len(archive_candidates),
            "selected_historical_archive_count": len(selected_archive),
            "evaluation_anchor_count": len(canonical_anchors),
            "unique_checkpoint_count": len({
                str(entry["checkpoint_sha256"]) for entry in canonical_entries
            }),
        },
    }
    active_by_id = {
        str(entry["opponent_id"]): entry for entry in canonical_active
    }
    lineage_manifests = {}
    for seed in ACTIVE_SEEDS:
        learner_id = str(next(
            entry["opponent_id"]
            for entry in canonical_active
            if int(entry["policy_seed"]) == seed
        ))
        estimates = tuple(
            PayoffEstimate(
                opponent_id=opponent_id,
                checkpoint_sha256=str(entry["checkpoint_sha256"]),
                games=ACTIVE_PAIR_GAMES,
                score_rate=float(active_payoff["score_matrix"][learner_id][opponent_id]),
                confidence_interval_95=tuple(
                    float(value)
                    for value in active_payoff[
                        "confidence_interval_95_matrix"
                    ][learner_id][opponent_id]
                ),
            )
            for opponent_id, entry in sorted(active_by_id.items())
        )
        distribution = compute_pfsp_distribution(
            estimates,
            sampler="hard",
            epsilon_floor=0.02,
            maximum_probability=0.35,
            hard_alpha=1.0,
        )
        rows = []
        for canonical in canonical_entries:
            entry = dict(canonical)
            opponent_id = str(entry["opponent_id"])
            checkpoint_hash = str(entry["checkpoint_sha256"])
            if entry["role"] == "active_latest":
                entry["training_eligible"] = True
                entry["sampling_weight"] = (
                    ACTIVE_FRACTION * distribution.probabilities[opponent_id]
                )
            elif (
                entry["role"] == "historical_archive"
                and checkpoint_hash in selected_hashes
            ):
                entry["training_eligible"] = True
                entry["sampling_weight"] = (
                    ARCHIVE_FRACTION / len(selected_archive)
                )
            else:
                entry["training_eligible"] = False
                entry["sampling_weight"] = 0.0
            rows.append(entry)
        lineage_manifests[seed] = {
            "schema_version": 1,
            "report_kind": "ppo_league_generation_manifest",
            "immutable": True,
            "path_base": "repository_root",
            "generation": target_generation,
            "selection_mode": "hard",
            "learner": {
                "lineage_id": f"lineage_{seed}",
                "policy_seed": seed,
                "parent_opponent_id": learner_id,
                "parent_checkpoint_sha256": active_by_id[learner_id][
                    "checkpoint_sha256"
                ],
            },
            "contract": source_population["contract"],
            "sources": {
                "population_payload_sha256": stable_json_sha256(population),
                "active_payoff_payload_sha256": stable_json_sha256(active_payoff),
            },
            "two_stage_sampling": {
                "active_latest_mass": ACTIVE_FRACTION,
                "historical_archive_mass": ARCHIVE_FRACTION,
                "active_latest_distribution": distribution.report(),
                "historical_archive_distribution": "uniform",
            },
            "entries": rows,
            "selection_audit": _selection_audit(rows),
            "summary": {
                "entry_count": len(rows),
                "positive_entry_count": sum(
                    float(entry["sampling_weight"]) > 0 for entry in rows
                ),
                "active_latest_count": 6,
                "historical_archive_count": len(selected_archive),
                "sampling_weight_total": sum(
                    float(entry["sampling_weight"]) for entry in rows
                ),
            },
        }
    archive_selection = {
        "schema_version": 1,
        "report_kind": "ppo_league_archive_selection",
        "generation": target_generation,
        "maximum_positive_archive_entries": MAX_POSITIVE_ARCHIVE,
        "selected_checkpoint_sha256": sorted(selected_hashes),
        "selected_archive_ids": [
            str(entry["opponent_id"]) for entry in selected_archive
        ],
        "zero_weight_archive_ids": [
            str(entry["opponent_id"])
            for entry in archive_candidates
            if str(entry["checkpoint_sha256"]) not in selected_hashes
        ],
        "forgotten_checkpoint_sha256": sorted(forgotten),
        "priority_order": (
            "previous_active, forgotten, generation_0_histories, "
            "meta_support_and_generation_recency"
        ),
    }
    return population, lineage_manifests, archive_selection


def _generation_gate(
    validation: Mapping[str, object],
    *,
    previous_training_report: Mapping[str, object] | None,
    population_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    summary = validation["summary"]
    mean_score = float(summary["mean_score_rate"])
    minimum_score = float(summary["minimum_score_rate"])
    nondegraded = int(summary["nondegraded_lineages"])
    safety_pass = (
        nondegraded >= 4
        and minimum_score >= VALIDATION_NONDEGRADATION_SCORE
    )
    previous_mean = None
    gain = None
    previous_low_gain_count = 0
    exploitability_proxy = None
    previous_exploitability_proxy = None
    exploitability_proxy_improvement = None
    proxy_improved = False
    if population_summary is not None:
        exploitability_proxy = float(
            population_summary["uniform_mixture"]["exploitability_proxy"]
        )
    if previous_training_report is not None:
        previous_gate = previous_training_report["gate"]
        previous_mean = float(previous_gate["mean_validation_score"])
        gain = mean_score - previous_mean
        previous_low_gain_count = int(
            previous_gate.get("consecutive_low_gain_generations", 0)
        )
        previous_summary = previous_training_report.get("population_summary")
        if previous_summary is not None and exploitability_proxy is not None:
            previous_exploitability_proxy = float(
                previous_summary["uniform_mixture"]["exploitability_proxy"]
            )
            if previous_exploitability_proxy > 0.0:
                exploitability_proxy_improvement = (
                    previous_exploitability_proxy - exploitability_proxy
                ) / previous_exploitability_proxy
                proxy_improved = (
                    exploitability_proxy_improvement
                    >= MINIMUM_EXPLOITABILITY_PROXY_IMPROVEMENT
                )
    low_gain_count = (
        0
        if gain is None or gain >= MINIMUM_GENERATION_GAIN or proxy_improved
        else previous_low_gain_count + 1
    )
    plateau_stop = low_gain_count >= 2
    passed = safety_pass and not plateau_stop
    reasons = []
    if nondegraded < 4:
        reasons.append("fewer_than_four_nondegraded_lineages")
    if minimum_score < VALIDATION_NONDEGRADATION_SCORE:
        reasons.append("population_worst_lineage_below_minus_two_pp_gate")
    if plateau_stop:
        reasons.append("two_consecutive_generations_below_minimum_gain")
    return {
        "passed": passed,
        "stop_reasons": reasons,
        "mean_validation_score": mean_score,
        "minimum_validation_score": minimum_score,
        "nondegraded_lineages": nondegraded,
        "previous_mean_validation_score": previous_mean,
        "generation_gain": gain,
        "minimum_generation_gain": MINIMUM_GENERATION_GAIN,
        "uniform_exploitability_proxy": exploitability_proxy,
        "previous_uniform_exploitability_proxy": previous_exploitability_proxy,
        "uniform_exploitability_proxy_relative_improvement": (
            exploitability_proxy_improvement
        ),
        "minimum_exploitability_proxy_relative_improvement": (
            MINIMUM_EXPLOITABILITY_PROXY_IMPROVEMENT
        ),
        "exploitability_proxy_improved": proxy_improved,
        "consecutive_low_gain_generations": low_gain_count,
    }


class GenerationRunner:
    def __init__(
        self,
        *,
        report_root: Path,
        checkpoint_root: Path,
        max_target_generation: int,
        lineage_order: Sequence[int],
        smoke_agent_steps: int | None = None,
        smoke_interrupt_seed: int | None = None,
    ) -> None:
        self.report_root = report_root
        self.checkpoint_root = checkpoint_root
        self.max_target_generation = max_target_generation
        self.lineage_order = tuple(lineage_order)
        self.smoke_agent_steps = smoke_agent_steps
        self.smoke_interrupt_seed = smoke_interrupt_seed
        self.schedule = _read_json(SCHEDULE_OUTPUT)
        self.contract = _read_json(CONTRACT_OUTPUT)
        self.state_path = report_root / "generation_queue_status.json"
        self.log_path = report_root / "generation_queue.log"
        if smoke_agent_steps is not None:
            pending_jobs = [f"seed_{seed}" for seed in self.lineage_order]
        else:
            pending_jobs = [
                f"g{int(transition['target_generation']):03d}/seed_{seed}"
                for transition in self.schedule["transitions"]
                if int(transition["target_generation"])
                <= self.max_target_generation
                for seed in self.lineage_order
            ]
        self.state: dict[str, object] = {
            "schema_version": 1,
            "state": "running",
            "runner_pid": os.getpid(),
            "mode": "smoke" if smoke_agent_steps is not None else "full",
            "active_stage": "initializing",
            "active_job": "",
            "completed_generations": [],
            "pending_training_jobs": pending_jobs,
            "completed_training_jobs": [],
            "stop_reason": "",
            "error": "",
        }
        self._write_state()

    def _write_state(self) -> None:
        self.state["updated_at_epoch_seconds"] = time.time()
        _atomic_json(self.state_path, self.state)

    def _set_active(self, stage: str, job: str) -> None:
        self.state["active_stage"] = stage
        self.state["active_job"] = job
        self._write_state()
        _append_log(self.log_path, f"Starting {stage} / {job}.")

    def _complete_job(self, stage: str, job: str, elapsed: float) -> None:
        _append_log(
            self.log_path,
            f"Completed {stage} / {job}: elapsed={elapsed:.1f}s.",
        )

    def _mark_training_complete(self, job: str) -> None:
        pending = self.state["pending_training_jobs"]
        if job in pending:
            pending.remove(job)
        if job not in self.state["completed_training_jobs"]:
            self.state["completed_training_jobs"].append(job)
        self._write_state()

    def _training_paths(self, generation: int, seed: int) -> dict[str, Path]:
        report_directory = _generation_directory(self.report_root, generation)
        checkpoint = (
            _checkpoint_directory(self.checkpoint_root, generation)
            / f"seed_{seed}"
            / "final.pt"
        )
        return {
            "checkpoint": checkpoint,
            "metrics": report_directory / "training" / f"seed_{seed}.json",
            "stdout": report_directory / "training" / f"seed_{seed}.log",
            "stderr": report_directory / "training" / f"seed_{seed}.stderr.log",
        }

    def _train_once(
        self,
        *,
        source_generation: int,
        target_generation: int,
        seed: int,
        parent_checkpoint: Path,
        target_agent_steps: int,
        metrics_suffix: str = "",
    ) -> dict[str, object]:
        paths = self._training_paths(target_generation, seed)
        if metrics_suffix:
            paths["metrics"] = paths["metrics"].with_name(
                f"seed_{seed}{metrics_suffix}.json"
            )
        manifest = _lineage_manifest_path(
            self.report_root, source_generation, seed
        )
        checkpoint = paths["checkpoint"]
        metrics = paths["metrics"]
        if metrics.is_file():
            return _validate_training_report(
                metrics,
                target_agent_steps=target_agent_steps,
                manifest_path=manifest,
            )
        resume = parent_checkpoint
        replace_pool = True
        partial_checkpoint, progress = _latest_resume_checkpoint(checkpoint)
        if partial_checkpoint is not None:
            resume = partial_checkpoint
            replace_pool = False
        command = _training_command(
            resume=resume,
            checkpoint=checkpoint,
            metrics=metrics,
            target_agent_steps=target_agent_steps,
            lineage_manifest=manifest if replace_pool else None,
            replace_opponent_pool=replace_pool,
        )
        started = time.perf_counter()
        _run_command(command, paths["stdout"], paths["stderr"])
        self._complete_job("training", f"g{target_generation:03d}/seed_{seed}", time.perf_counter() - started)
        return _validate_training_report(
            metrics,
            target_agent_steps=target_agent_steps,
            manifest_path=manifest,
        )

    def _run_smoke(self) -> None:
        source_population = _read_json(GENERATION_ZERO_MANIFEST)
        source_by_seed = _source_active_entry_map(source_population, 0)
        reports = []
        for seed in self.lineage_order:
            parent = _repo_path(source_by_seed[seed]["checkpoint_path"])
            base_steps = int(source_by_seed[seed]["training_steps"])
            final_target = base_steps + int(self.smoke_agent_steps)
            self._set_active("smoke_training", f"seed_{seed}")
            if seed == self.smoke_interrupt_seed:
                first_target = base_steps + max(1, int(self.smoke_agent_steps) // 2)
                self._train_once(
                    source_generation=0,
                    target_generation=1,
                    seed=seed,
                    parent_checkpoint=parent,
                    target_agent_steps=first_target,
                    metrics_suffix="_interrupted_part",
                )
            report = self._train_once(
                source_generation=0,
                target_generation=1,
                seed=seed,
                parent_checkpoint=parent,
                target_agent_steps=final_target,
            )
            reports.append(report)
            self._mark_training_complete(f"seed_{seed}")
        summary = {
            "schema_version": 1,
            "report_kind": "ppo_league_generation_runner_smoke",
            "state": "completed",
            "lineage_order": list(self.lineage_order),
            "smoke_agent_steps": self.smoke_agent_steps,
            "interrupted_and_resumed_seed": self.smoke_interrupt_seed,
            "lineages": [
                {
                    "policy_seed": int(report["master_seed"]),
                    "trained_agent_steps": int(report["trained_agent_steps"]),
                    "completed_agent_steps": int(report["completed_agent_steps"]),
                    "agent_steps_per_second": float(report["agent_steps_per_second"]),
                    "truncated_episodes": int(
                        report["league_diagnostics"]["truncated_episodes"]
                    ),
                    "illegal_action_errors": int(
                        report["league_diagnostics"]["illegal_action_errors"]
                    ),
                    "action_mask_mismatch_errors": int(
                        report["league_diagnostics"]["action_mask_mismatch_errors"]
                    ),
                    "manifest_sha256": report["league_diagnostics"][
                        "external_opponent_manifest"
                    ]["file_sha256"],
                }
                for report in reports
            ],
        }
        _atomic_json(self.report_root / "generation_runner_smoke.json", summary)
        self.state.update({
            "state": "completed",
            "active_stage": "",
            "active_job": "",
        })
        self._write_state()
        _append_log(self.log_path, "Generation runner smoke completed.")

    def _ensure_evaluation(
        self,
        *,
        stage: str,
        job_id: str,
        learner: Mapping[str, object],
        opponent: Mapping[str, object],
        output: Path,
        master_seed: int,
        seed_count: int = 2,
    ) -> None:
        games = 98 * seed_count
        if output.is_file():
            _validate_evaluation(
                output,
                learner_entry=learner,
                opponent_entry=opponent,
                master_seed=master_seed,
                games=games,
            )
            return
        self._set_active(stage, job_id)
        started = time.perf_counter()
        _run_command(
            _evaluation_command(
                learner=_repo_path(learner["checkpoint_path"]),
                opponent=_repo_path(opponent["checkpoint_path"]),
                output=output,
                master_seed=master_seed,
                seed_count=seed_count,
            ),
            output.with_suffix(".log"),
            output.with_suffix(".stderr.log"),
        )
        self._complete_job(stage, job_id, time.perf_counter() - started)

    def _run_archive_audit(
        self,
        *,
        generation: int,
        source_generation: int,
        source_population: Mapping[str, object],
        active_entries: Sequence[Mapping[str, object]],
        master_seed: int,
    ) -> dict[str, object]:
        audit_directory = (
            _generation_directory(self.report_root, generation)
            / "archive_audit"
        )
        archive_entries = _archive_candidates(
            source_population,
            source_generation,
        )
        previous_scores = _previous_archive_scores(
            self.report_root,
            generation,
        )
        for active in sorted(
            active_entries,
            key=lambda row: int(row["policy_seed"]),
        ):
            seed = int(active["policy_seed"])
            for archive in archive_entries:
                opponent_id = str(archive["opponent_id"])
                checkpoint_hash = str(archive["checkpoint_sha256"])
                screen_path = _archive_evaluation_path(
                    audit_directory,
                    "screen",
                    seed,
                    opponent_id,
                )
                self._ensure_evaluation(
                    stage="archive_screen",
                    job_id=(
                        f"g{generation:03d}/seed_{seed}_vs_{opponent_id}"
                    ),
                    learner=active,
                    opponent=archive,
                    output=screen_path,
                    master_seed=master_seed,
                    seed_count=ARCHIVE_SCREEN_SEED_COUNT,
                )
                screen, _ = _validate_evaluation(
                    screen_path,
                    learner_entry=active,
                    opponent_entry=archive,
                    master_seed=master_seed,
                    games=98,
                )
                previous = previous_scores.get((seed, checkpoint_hash))
                if not _archive_confirmation_required(
                    previous,
                    float(screen["score_rate"]),
                ):
                    continue
                confirm_path = _archive_evaluation_path(
                    audit_directory,
                    "confirm",
                    seed,
                    opponent_id,
                )
                self._ensure_evaluation(
                    stage="archive_confirmation",
                    job_id=(
                        f"g{generation:03d}/seed_{seed}_vs_{opponent_id}"
                    ),
                    learner=active,
                    opponent=archive,
                    output=confirm_path,
                    master_seed=master_seed,
                    seed_count=ARCHIVE_CONFIRM_SEED_COUNT,
                )
        report = _build_archive_audit_report(
            generation=generation,
            master_seed=master_seed,
            active_entries=active_entries,
            archive_entries=archive_entries,
            audit_directory=audit_directory,
            previous_scores=previous_scores,
        )
        _atomic_json(
            _generation_directory(self.report_root, generation)
            / "archive_audit.json",
            report,
        )
        return report

    def _run_full(self) -> None:
        transitions = self.schedule["transitions"]
        previous_training_report = None
        for transition in transitions:
            target_generation = int(transition["target_generation"])
            if target_generation > self.max_target_generation:
                break
            source_generation = target_generation - 1
            source_population_path = _population_path(
                self.report_root, source_generation
            )
            source_population = _read_json(source_population_path)
            source_by_seed = _source_active_entry_map(
                source_population, source_generation
            )
            training_reports = []
            for seed in self.lineage_order:
                target = next(
                    int(row["target_agent_steps"])
                    for row in transition["lineages"]
                    if int(row["policy_seed"]) == seed
                )
                self._set_active("training", f"g{target_generation:03d}/seed_{seed}")
                report = self._train_once(
                    source_generation=source_generation,
                    target_generation=target_generation,
                    seed=seed,
                    parent_checkpoint=_repo_path(
                        source_by_seed[seed]["checkpoint_path"]
                    ),
                    target_agent_steps=target,
                )
                training_reports.append(report)
                self._mark_training_complete(
                    f"g{target_generation:03d}/seed_{seed}"
                )
            source_rules = str(next(iter(source_by_seed.values()))["rules_version"])
            active_entries = [
                _checkpoint_entry(
                    self._training_paths(target_generation, seed)["checkpoint"],
                    generation=target_generation,
                    role="active_latest",
                    rules_version=source_rules,
                )
                for seed in ACTIVE_SEEDS
            ]
            active_by_seed = {
                int(entry["policy_seed"]): entry for entry in active_entries
            }
            generation_directory = _generation_directory(
                self.report_root, target_generation
            )
            active_evaluations = generation_directory / "active_evaluations"
            validation_evaluations = generation_directory / "validation_evaluations"
            master_seed = int(transition["payoff_master_seed"])
            for left_index, left_seed in enumerate(ACTIVE_SEEDS):
                for right_seed in ACTIVE_SEEDS[left_index + 1 :]:
                    self._ensure_evaluation(
                        stage="active_evaluation",
                        job_id=f"g{target_generation:03d}/seed_{left_seed}_vs_{right_seed}",
                        learner=active_by_seed[left_seed],
                        opponent=active_by_seed[right_seed],
                        output=active_evaluations / f"seed_{left_seed}__vs__seed_{right_seed}.json",
                        master_seed=master_seed,
                    )
            generation_zero = _read_json(GENERATION_ZERO_MANIFEST)
            base_by_seed = _source_active_entry_map(generation_zero, 0)
            for seed in ACTIVE_SEEDS:
                self._ensure_evaluation(
                    stage="validation_evaluation",
                    job_id=f"g{target_generation:03d}/seed_{seed}_vs_g0_parent",
                    learner=active_by_seed[seed],
                    opponent=base_by_seed[seed],
                    output=validation_evaluations / f"seed_{seed}__vs__generation_000_parent.json",
                    master_seed=master_seed,
                )
            active_payoff = _build_active_payoff(
                generation=target_generation,
                active_entries=active_entries,
                evaluation_directory=active_evaluations,
                master_seed=master_seed,
            )
            validation = _build_validation_report(
                generation=target_generation,
                active_entries=active_entries,
                validation_directory=validation_evaluations,
                master_seed=master_seed,
            )
            population_summary = _build_population_summary(
                active_payoff=active_payoff,
                validation=validation,
            )
            gate = _generation_gate(
                validation,
                previous_training_report=previous_training_report,
                population_summary=population_summary,
            )
            forgotten_hashes = _previous_forgotten_hashes(
                self.report_root,
                target_generation,
            )
            archive_audit = None
            if bool(transition["archive_audit_due"]):
                archive_audit = self._run_archive_audit(
                    generation=target_generation,
                    source_generation=source_generation,
                    source_population=source_population,
                    active_entries=active_entries,
                    master_seed=master_seed,
                )
                forgotten_hashes.update(
                    str(value)
                    for value in archive_audit["summary"][
                        "forgotten_checkpoint_sha256"
                    ]
                )
            population, manifests, archive_selection = (
                _build_population_and_lineage_manifests(
                    source_population=source_population,
                    source_generation=source_generation,
                    target_generation=target_generation,
                    active_entries=active_entries,
                    active_payoff=active_payoff,
                    forgotten_hashes=forgotten_hashes,
                )
            )
            training_report = {
                "schema_version": 1,
                "report_kind": "ppo_evolving_league_generation_training",
                "generation": target_generation,
                "source_generation": source_generation,
                "source_population_manifest": _source(source_population_path),
                "training_reports": [
                    _source(
                        self._training_paths(target_generation, seed)["metrics"]
                    )
                    for seed in ACTIVE_SEEDS
                ],
                "active_payoff_payload_sha256": stable_json_sha256(active_payoff),
                "validation_payload_sha256": stable_json_sha256(validation),
                "population_summary": population_summary,
                "archive_audit_due": bool(transition["archive_audit_due"]),
                "archive_audit_status": (
                    "completed_before_generation_publish"
                    if transition["archive_audit_due"]
                    else "not_due"
                ),
                "archive_audit_payload_sha256": (
                    stable_json_sha256(archive_audit)
                    if archive_audit is not None
                    else None
                ),
                "gate": gate,
                "safety": {
                    "illegal_actions": 0,
                    "action_mask_mismatches": 0,
                    "truncated": 0,
                    "nan_or_inf": 0,
                    "completed_lineages": len(training_reports),
                },
            }
            _atomic_json(generation_directory / "active_payoff.json", active_payoff)
            _atomic_json(generation_directory / "validation_report.json", validation)
            _atomic_json(generation_directory / "archive_selection.json", archive_selection)
            _atomic_json(generation_directory / "training_report.json", training_report)
            if not gate["passed"]:
                self.state.update({
                    "state": "stopped_by_gate",
                    "active_stage": "",
                    "active_job": "",
                    "stop_reason": ",".join(gate["stop_reasons"]),
                })
                self._write_state()
                _append_log(
                    self.log_path,
                    f"Generation {target_generation} stopped by gate: {self.state['stop_reason']}.",
                )
                return
            _atomic_json(generation_directory / "population_manifest.json", population)
            for seed, manifest in manifests.items():
                _atomic_json(
                    generation_directory / "lineage_manifests" / f"seed_{seed}.json",
                    manifest,
                )
            self.state["completed_generations"].append(target_generation)
            self._write_state()
            _append_log(
                self.log_path,
                f"Published generation {target_generation:03d} after all gates passed.",
            )
            previous_training_report = training_report
        self.state.update({
            "state": "completed",
            "active_stage": "",
            "active_job": "",
        })
        self._write_state()
        _append_log(self.log_path, "Evolving League queue completed.")

    def run(self) -> None:
        try:
            if self.smoke_agent_steps is not None:
                self._run_smoke()
            else:
                self._run_full()
        except BaseException as exc:
            self.state.update({
                "state": "failed",
                "active_stage": "",
                "active_job": "",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            self._write_state()
            _append_log(self.log_path, f"Queue failed: {type(exc).__name__}: {exc}")
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the serial, resumable six-lineage PFSP League generation queue."
        )
    )
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT
    )
    parser.add_argument(
        "--max-target-generation", type=int, default=8, choices=range(1, 9)
    )
    parser.add_argument(
        "--lineage-order",
        choices=("forward", "reverse"),
        default="forward",
    )
    parser.add_argument("--smoke-agent-steps", type=int)
    parser.add_argument("--smoke-interrupt-seed", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_agent_steps is not None and args.smoke_agent_steps <= 0:
        raise SystemExit("--smoke-agent-steps must be positive")
    if (
        args.smoke_interrupt_seed is not None
        and args.smoke_interrupt_seed not in ACTIVE_SEEDS
    ):
        raise SystemExit("--smoke-interrupt-seed must identify an active lineage")
    order = ACTIVE_SEEDS if args.lineage_order == "forward" else tuple(reversed(ACTIVE_SEEDS))
    runner = GenerationRunner(
        report_root=args.report_root,
        checkpoint_root=args.checkpoint_root,
        max_target_generation=args.max_target_generation,
        lineage_order=order,
        smoke_agent_steps=args.smoke_agent_steps,
        smoke_interrupt_seed=args.smoke_interrupt_seed,
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
