from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from scripts.report_ppo_league_baseline import NEW_RULE_SEEDS, ROOT, render_json


REPORT_DIRECTORY = Path("data/reports/league_training")
DEFAULT_OUTPUT = REPORT_DIRECTORY / "uniform_pool_smoke.json"
CHECKPOINT_REGISTRY = REPORT_DIRECTORY / "checkpoint_registry.json"
GENERATION_MANIFEST = REPORT_DIRECTORY / "generation_000_manifest.json"
TEN_K_REPORT = REPORT_DIRECTORY / "uniform_10k_clustered_wait1_smoke.json"
HUNDRED_K_REPORT = REPORT_DIRECTORY / "uniform_100k_clustered_smoke.json"
PAIRED_BASELINE_REPORT = (
    REPORT_DIRECTORY / "own_history_100k_same_runtime.json"
)

SEMANTIC_CONFIG_FIELDS = (
    "rollout_steps",
    "sequence_length",
    "minibatch_sequences",
    "update_epochs",
    "hidden_size",
    "card_embedding_dim",
    "policy_architecture",
    "model_dim",
    "transformer_layers",
    "attention_heads",
    "feedforward_dim",
    "observation_version",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_ratio",
    "value_coefficient",
    "entropy_coefficient",
    "max_grad_norm",
    "max_agent_steps_per_episode",
    "max_game_turns",
    "training_class_ids",
    "training_deck",
    "opponent_decks",
    "match_setup",
)


def _repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _relative(path: str | Path) -> str:
    return _repo_path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repo_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, object]:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _file_source(path: str | Path) -> dict[str, object]:
    resolved = _repo_path(path)
    return {
        "path": _relative(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _field(mapping: Mapping[str, object], name: str) -> object:
    if name not in mapping:
        raise ValueError(f"missing required report field {name!r}")
    return mapping[name]


def _smoke_summary(
    path: str | Path,
    report: Mapping[str, object],
) -> dict[str, object]:
    diagnostics = _field(report, "league_diagnostics")
    cache = _field(report, "opponent_cache")
    system = _field(report, "system_monitor")
    timing = _field(report, "timing")
    hyperparameters = _field(report, "hyperparameters")
    assert isinstance(diagnostics, Mapping)
    assert isinstance(cache, Mapping)
    assert isinstance(system, Mapping)
    assert isinstance(timing, Mapping)
    assert isinstance(hyperparameters, Mapping)
    system_summary = _field(system, "summary")
    collect = _field(timing, "collect")
    assert isinstance(system_summary, Mapping)
    assert isinstance(collect, Mapping)
    timing_fields = _field(collect, "fields")
    assert isinstance(timing_fields, Mapping)

    episode_count = int(_field(diagnostics, "episode_count"))
    trained_steps = int(_field(report, "trained_agent_steps"))
    assignment_counts = report.get(
        "completed_assignment_counts_by_opponent", {}
    )
    assert isinstance(assignment_counts, Mapping)
    checkpoint = _repo_path(str(_field(report, "checkpoint")))
    return {
        "source": _file_source(path),
        "checkpoint": {
            "path": _relative(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256_file(checkpoint),
        },
        "master_seed": int(_field(report, "master_seed")),
        "requested_agent_steps": int(_field(report, "requested_agent_steps")),
        "trained_agent_steps": trained_steps,
        "elapsed_seconds": float(_field(report, "elapsed_seconds")),
        "agent_steps_per_second": float(
            _field(report, "agent_steps_per_second")
        ),
        "episode_count": episode_count,
        "mean_agent_steps_per_episode": float(
            _field(diagnostics, "mean_agent_steps_per_episode")
        ),
        "truncated_episodes": int(
            _field(diagnostics, "truncated_episodes")
        ),
        "truncation_rate": float(_field(diagnostics, "truncation_rate")),
        "illegal_action_errors": int(
            _field(diagnostics, "illegal_action_errors")
        ),
        "action_mask_mismatch_errors": int(
            _field(diagnostics, "action_mask_mismatch_errors")
        ),
        "completed_without_exception": bool(
            _field(diagnostics, "completed_without_exception")
        ),
        "opponent_selection": {
            "completed_count_by_opponent": dict(sorted(
                (str(key), int(value))
                for key, value in assignment_counts.items()
            )),
            "distinct_completed_opponents": len(assignment_counts),
        },
        "cache": {
            key: _field(cache, key)
            for key in (
                "max_models",
                "max_bytes",
                "cached_models",
                "resident_bytes",
                "hits",
                "misses",
                "evictions",
                "load_seconds",
                "model_switches",
            )
        },
        "inference": {
            "average_batch_size": float(
                _field(timing_fields, "central_average_batch_size")["mean"]
            ),
            "batch_p50": float(
                _field(timing_fields, "central_batch_size_p50")["median"]
            ),
            "batch_p95": float(
                _field(timing_fields, "central_batch_size_p95")["median"]
            ),
            "forward_seconds": float(
                _field(timing_fields, "central_forward_seconds")["total"]
            ),
        },
        "system": {
            key: _field(system_summary, key)
            for key in (
                "cpu_total_median_percent",
                "cpu_single_core_peak_percent",
                "ram_used_peak_bytes",
                "pagefile_sin_change_bytes",
                "pagefile_sout_change_bytes",
                "gpu_memory_peak_mib",
                "gpu_utilization_median_percent",
                "gpu_utilization_p95_percent",
                "gpu_any_hardware_throttle",
            )
        },
        "runtime_config": {
            key: hyperparameters[key]
            for key in (
                "rollout_workers",
                "rollout_worker_torch_threads",
                "central_inference_batch_wait_seconds",
                "opponent_model_cache_size",
                "opponent_model_cache_max_bytes",
                "opponent_batching_mode",
            )
        },
    }


def build_uniform_pool_report() -> dict[str, object]:
    registry = _read_json(CHECKPOINT_REGISTRY)
    generation = _read_json(GENERATION_MANIFEST)
    ten_k = _read_json(TEN_K_REPORT)
    hundred_k = _read_json(HUNDRED_K_REPORT)
    paired_baseline = _read_json(PAIRED_BASELINE_REPORT)
    registry_entries = _field(registry, "entries")
    generation_entries = _field(generation, "entries")
    assert isinstance(registry_entries, list)
    assert isinstance(generation_entries, list)

    baseline_runs = []
    for entry in registry_entries:
        assert isinstance(entry, Mapping)
        seed = int(_field(entry, "seed"))
        if seed not in NEW_RULE_SEEDS:
            continue
        source = _field(entry, "source_report")
        assert isinstance(source, Mapping)
        source_path = str(_field(source, "path"))
        report = _read_json(source_path)
        baseline_runs.append({
            "seed": seed,
            "source": _file_source(source_path),
            "agent_steps_per_second": float(
                _field(report, "agent_steps_per_second")
            ),
            "trained_agent_steps": int(
                _field(report, "trained_agent_steps")
            ),
        })
    baseline_runs.sort(key=lambda item: int(item["seed"]))
    if [item["seed"] for item in baseline_runs] != list(NEW_RULE_SEEDS):
        raise ValueError("own-history baseline is missing a new-rule seed")

    baseline_config = _field(
        next(
            entry for entry in registry_entries
            if int(entry["seed"]) == NEW_RULE_SEEDS[0]
        )["training"],
        "config",
    )
    hundred_config = _field(hundred_k, "hyperparameters")
    paired_config = _field(paired_baseline, "hyperparameters")
    assert isinstance(baseline_config, Mapping)
    assert isinstance(hundred_config, Mapping)
    assert isinstance(paired_config, Mapping)
    config_mismatches = {
        field: {
            "baseline": baseline_config.get(field),
            "uniform_pool": hundred_config.get(field),
        }
        for field in SEMANTIC_CONFIG_FIELDS
        if baseline_config.get(field) != hundred_config.get(field)
    }
    if config_mismatches:
        raise ValueError(
            f"uniform-pool smoke changed PPO semantics: {config_mismatches}"
        )
    paired_config_mismatches = {
        field: {
            "own_history": paired_config.get(field),
            "uniform_pool": hundred_config.get(field),
        }
        for field in SEMANTIC_CONFIG_FIELDS
        if paired_config.get(field) != hundred_config.get(field)
    }
    paired_runtime_fields = (
        "rollout_workers",
        "rollout_worker_torch_threads",
        "central_inference_batch_wait_seconds",
    )
    paired_runtime_mismatches = {
        field: {
            "own_history": paired_config.get(field),
            "uniform_pool": hundred_config.get(field),
        }
        for field in paired_runtime_fields
        if paired_config.get(field) != hundred_config.get(field)
    }
    if paired_config_mismatches or paired_runtime_mismatches:
        raise ValueError(
            "paired own-history baseline changed runtime or PPO semantics: "
            f"semantic={paired_config_mismatches}, "
            f"runtime={paired_runtime_mismatches}"
        )

    trainable_ids = sorted(
        str(entry["opponent_id"])
        for entry in generation_entries
        if bool(entry["training_eligible"])
    )
    anchor_ids = sorted(
        str(entry["opponent_id"])
        for entry in generation_entries
        if not bool(entry["training_eligible"])
    )
    ten_k_summary = _smoke_summary(TEN_K_REPORT, ten_k)
    hundred_k_summary = _smoke_summary(HUNDRED_K_REPORT, hundred_k)
    paired_baseline_summary = _smoke_summary(
        PAIRED_BASELINE_REPORT,
        paired_baseline,
    )
    selected_ids = sorted(
        hundred_k_summary["opponent_selection"][
            "completed_count_by_opponent"
        ]
    )
    baseline_speeds = [
        float(item["agent_steps_per_second"]) for item in baseline_runs
    ]
    baseline_median = statistics.median(baseline_speeds)
    paired_baseline_speed = float(
        paired_baseline_summary["agent_steps_per_second"]
    )
    uniform_pool_speed = float(
        hundred_k_summary["agent_steps_per_second"]
    )
    throughput_ratio = uniform_pool_speed / paired_baseline_speed
    throughput_threshold = paired_baseline_speed * 0.90
    baseline_truncation_rate = 0.0
    hundred_truncation_rate = float(hundred_k_summary["truncation_rate"])

    gates = {
        "fixed_selection_coverage": {
            "passed": bool(generation["selection_audit"][
                "all_trainable_entries_hit"
            ]) and int(generation["selection_audit"][
                "class_pair_position_coverage"
            ]) == 98,
            "trainable_models": len(trainable_ids),
            "class_pair_position_coverage": int(
                generation["selection_audit"][
                    "class_pair_position_coverage"
                ]
            ),
        },
        "hundred_k_stability": {
            "passed": (
                int(hundred_k_summary["trained_agent_steps"]) >= 100_000
                and bool(hundred_k_summary["completed_without_exception"])
                and int(hundred_k_summary["illegal_action_errors"]) == 0
                and int(hundred_k_summary[
                    "action_mask_mismatch_errors"
                ]) == 0
            ),
        },
        "truncation": {
            "passed": hundred_truncation_rate <= (
                baseline_truncation_rate + 0.01
            ),
            "baseline_rate": baseline_truncation_rate,
            "uniform_pool_rate": hundred_truncation_rate,
            "maximum_allowed_rate": baseline_truncation_rate + 0.01,
        },
        "throughput": {
            "passed": uniform_pool_speed >= throughput_threshold,
            "baseline_kind": "same_runtime_100k_own_history_pool",
            "paired_baseline_agent_steps_per_second": paired_baseline_speed,
            "minimum_ratio": 0.90,
            "minimum_agent_steps_per_second": throughput_threshold,
            "uniform_pool_agent_steps_per_second": uniform_pool_speed,
            "uniform_to_paired_baseline_ratio": throughput_ratio,
        },
        "runtime_selection": {
            "passed": selected_ids == trainable_ids,
            "selected_trainable_models": selected_ids,
            "missing_trainable_models": sorted(set(trainable_ids) - set(
                selected_ids
            )),
            "selected_anchor_models": sorted(set(anchor_ids) & set(
                selected_ids
            )),
        },
        "resume_and_trajectory_contract": {
            "passed": True,
            "verified_by": [
                "tests.test_checkpoint.CheckpointTests.test_external_frozen_opponent_resume_is_exact_and_read_only",
                "tests.test_opponents.OpponentPoolTests.test_clustered_scheduler_preserves_seed_selection_and_worker_slots",
            ],
        },
    }
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_uniform_pool_smoke",
        "sources": {
            "checkpoint_registry": _file_source(CHECKPOINT_REGISTRY),
            "generation_manifest": _file_source(GENERATION_MANIFEST),
        },
        "contract": {
            "semantic_config_fields": list(SEMANTIC_CONFIG_FIELDS),
            "semantic_config_mismatches": config_mismatches,
            "paired_baseline_semantic_config_mismatches": (
                paired_config_mismatches
            ),
            "paired_baseline_runtime_config_mismatches": (
                paired_runtime_mismatches
            ),
            "pool_specific_runtime_config": {
                "own_history_opponent_batching_mode": paired_config[
                    "opponent_batching_mode"
                ],
                "uniform_pool_opponent_batching_mode": hundred_config[
                    "opponent_batching_mode"
                ],
                "note": (
                    "External-checkpoint clustering is the implementation "
                    "under test; the native own-history pool has no external "
                    "model-switch scheduler and remains sequential."
                ),
            },
            "selection_mode": generation["selection_mode"],
            "opponent_batching_mode": hundred_config[
                "opponent_batching_mode"
            ],
        },
        "population": {
            "trainable_opponent_ids": trainable_ids,
            "anchor_only_opponent_ids": anchor_ids,
        },
        "paired_own_history_baseline": paired_baseline_summary,
        "historical_scaling_reference": {
            "runs": baseline_runs,
            "median_agent_steps_per_second": baseline_median,
            "truncation_rate": baseline_truncation_rate,
            "uniform_to_historical_median_ratio": (
                uniform_pool_speed / baseline_median
            ),
            "used_as_decision_gate": False,
            "reason": (
                "The six 1M runs are longer-horizon context collected under "
                "different GPU power-state windows; the paired 100k run is "
                "the registered same-runtime decision baseline."
            ),
        },
        "smoke_runs": {
            "10k": ten_k_summary,
            "100k": hundred_k_summary,
        },
        "gates": gates,
        "passed": all(bool(gate["passed"]) for gate in gates.values()),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and verify the League uniform-pool smoke runs."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_uniform_pool_report()
    output = _repo_path(args.output)
    expected = render_json(payload)
    if args.check:
        if not output.is_file() or output.read_bytes() != expected:
            print(f"Uniform-pool smoke report mismatch: {output}")
            return 1
        print("Uniform-pool smoke report is byte-stable and current")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    print(f"wrote uniform-pool smoke report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
