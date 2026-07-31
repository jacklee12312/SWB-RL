from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import torch

from scripts.scan_training_speed_stage_2_4 import DEFAULT_CHECKPOINT
from swb.db.repository import CardRepository
from swb.rl.checkpoint import load_checkpoint
from swb.rl.runtime import WorkerAssetsSnapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_COMPARISON = Path(
    "data/reports/training_speed/"
    "stage_2_7_b_batched_learner_001_end_to_end.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/"
    "stage_2_7_b_learner_amp_001_gate.json"
)
REPEATS = 3
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _optimizer_tensors(
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    result = {}
    state = optimizer.state_dict()["state"]
    for parameter_id in sorted(state):
        for name, value in sorted(state[parameter_id].items()):
            if isinstance(value, torch.Tensor):
                result[f"{parameter_id}:{name}"] = (
                    value.detach().cpu().clone()
                )
    return result


def _tensor_drift(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    maximum = 0.0
    mean_errors = []
    finite = True
    worst_name = None
    for name, expected in reference.items():
        value = candidate[name]
        difference = (expected.float() - value.float()).abs()
        current = float(difference.max())
        if current > maximum:
            maximum = current
            worst_name = name
        mean_errors.append(float(difference.mean()))
        finite = finite and bool(torch.isfinite(value).all())
    return {
        "tensor_count": len(reference),
        "finite": finite,
        "maximum_absolute_error": maximum,
        "mean_of_tensor_mean_absolute_errors": statistics.mean(
            mean_errors
        ),
        "worst_tensor": worst_name,
    }


def _run_update(
    snapshot: WorkerAssetsSnapshot,
    checkpoint: Path,
    records,
    bootstrap,
    *,
    dtype_name: str | None,
    repeat: int,
    capture_state: bool,
) -> dict[str, object]:
    trainer = load_checkpoint(
        checkpoint,
        snapshot,
        device="cuda",
        restore_rng_state=False,
    )
    try:
        trainer._batched_v41_learner = True
        dtype = None if dtype_name is None else DTYPES[dtype_name]
        trainer.configure_experimental_learner_amp(dtype)
        trainer.torch_generator.manual_seed(20260831 + repeat)
        scaler_before = (
            None
            if trainer._learner_grad_scaler is None
            else trainer._learner_grad_scaler.get_scale()
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            metrics = trainer.update(records, bootstrap)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            finite_metrics = all(
                math.isfinite(float(value))
                for value in metrics.values()
            )
            finite_model = all(
                bool(torch.isfinite(parameter).all())
                for parameter in trainer.model.parameters()
            )
            result: dict[str, object] = {
                "seconds": elapsed,
                "metrics": metrics,
                "finite_metrics": finite_metrics,
                "finite_model": finite_model,
                "exception": None,
                "stable": finite_metrics and finite_model,
                "grad_scaler": {
                    "enabled": (
                        trainer._learner_grad_scaler is not None
                    ),
                    "scale_before": scaler_before,
                    "scale_after": (
                        None
                        if trainer._learner_grad_scaler is None
                        else trainer._learner_grad_scaler.get_scale()
                    ),
                },
            }
            if capture_state:
                result["model"] = {
                    name: value.detach().cpu().clone()
                    for name, value in trainer.model.state_dict().items()
                }
                result["optimizer"] = _optimizer_tensors(
                    trainer.optimizer
                )
            return result
        except Exception as error:
            torch.cuda.synchronize()
            return {
                "seconds": time.perf_counter() - started,
                "metrics": None,
                "finite_metrics": False,
                "finite_model": all(
                    bool(torch.isfinite(parameter).all())
                    for parameter in trainer.model.parameters()
                ),
                "exception": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "stable": False,
                "grad_scaler": {
                    "enabled": (
                        trainer._learner_grad_scaler is not None
                    ),
                    "scale_before": scaler_before,
                    "scale_after": (
                        None
                        if trainer._learner_grad_scaler is None
                        else trainer._learner_grad_scaler.get_scale()
                    ),
                },
            }
    finally:
        trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess checklist 2.7 B-LEARNER-AMP-001"
    )
    parser.add_argument(
        "--database", type=Path, default=DEFAULT_DATABASE
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--comparison", type=Path, default=DEFAULT_COMPARISON
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
    collector = load_checkpoint(
        checkpoint,
        snapshot,
        device="cuda",
        restore_rng_state=False,
    )
    try:
        collector.config = replace(
            collector.config,
            rollout_workers=6,
            rollout_worker_torch_threads=2,
            central_inference_batch_wait_seconds=0.001,
        )
        records, bootstrap, _ = collector.collect_rollout()
    finally:
        collector.close()

    runs: dict[str, list[dict[str, object]]] = {
        "float32": [],
        **{name: [] for name in DTYPES},
    }
    captured: dict[str, dict[str, object]] = {}
    for repeat in range(REPEATS):
        for dtype_name in (None, *DTYPES):
            name = "float32" if dtype_name is None else dtype_name
            result = _run_update(
                snapshot,
                checkpoint,
                records,
                bootstrap,
                dtype_name=dtype_name,
                repeat=repeat,
                capture_state=repeat == 0,
            )
            if repeat == 0 and result["stable"]:
                captured[name] = result
            runs[name].append({
                "repeat": repeat,
                "seconds": result["seconds"],
                "metrics": result["metrics"],
                "finite_metrics": result["finite_metrics"],
                "finite_model": result["finite_model"],
                "exception": result["exception"],
                "stable": result["stable"],
                "grad_scaler": result["grad_scaler"],
            })
            print(json.dumps({
                "dtype": name,
                "repeat": repeat,
                "seconds": result["seconds"],
                "stable": result["stable"],
                "exception": result["exception"],
            }, sort_keys=True), flush=True)

    comparison = _json(args.comparison)
    comparison_speeds = [
        float(value)
        for value in comparison["end_to_end"][
            "runs_agent_steps_per_second"
        ]
    ]
    variability = (
        max(comparison_speeds) - min(comparison_speeds)
    ) / statistics.median(comparison_speeds)
    baseline_times = [
        float(run["seconds"]) for run in runs["float32"]
    ]
    baseline_median = statistics.median(baseline_times)
    variants = {}
    advancing = []
    for name in DTYPES:
        candidate_runs = runs[name]
        candidate_times = [
            float(run["seconds"]) for run in candidate_runs
        ]
        candidate_median = statistics.median(candidate_times)
        reduction = (
            baseline_median - candidate_median
        ) / baseline_median
        stable = all(bool(run["stable"]) for run in candidate_runs)
        model_drift = (
            _tensor_drift(
                captured["float32"]["model"],
                captured[name]["model"],
            )
            if name in captured and "float32" in captured
            else None
        )
        optimizer_drift = (
            _tensor_drift(
                captured["float32"]["optimizer"],
                captured[name]["optimizer"],
            )
            if name in captured and "float32" in captured
            else None
        )
        advance = stable and reduction > variability
        if advance:
            advancing.append(name)
        variants[name] = {
            "runs": candidate_runs,
            "median_seconds": candidate_median,
            "relative_reduction": reduction,
            "stable": stable,
            "speed_exceeds_current_three_run_variability": (
                reduction > variability
            ),
            "model_after_one_update_drift": model_drift,
            "optimizer_after_one_update_drift": optimizer_drift,
            "advance_to_three_seed_learning": advance,
        }
    selected = (
        min(
            advancing,
            key=lambda name: variants[name]["median_seconds"],
        )
        if advancing
        else None
    )
    checkpoint_after = {
        "sha256": _sha256(checkpoint),
        "size": checkpoint.stat().st_size,
        "mtime_ns": checkpoint.stat().st_mtime_ns,
    }
    report = {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_7_b_learner_amp_001_gate"
        ),
        "checklist_section": "2.7",
        "candidate": "B-LEARNER-AMP-001",
        "classification": "B",
        "methodology": {
            "same_collected_rollout": True,
            "same_minibatch_permutation_seed_per_pair": True,
            "repeats": REPEATS,
            "records": len(records),
            "rollout_workers": 6,
            "worker_torch_threads": 2,
            "central_inference_batch_wait_seconds": 0.001,
            "batched_v41_learner": True,
            "amp_variants": list(DTYPES),
            "grad_scaler_initial_scale": 16.0,
        },
        "baseline": {
            "runs": runs["float32"],
            "seconds": baseline_times,
            "median_seconds": baseline_median,
        },
        "variants": variants,
        "decision": {
            "advancing_variants": advancing,
            "selected_variant": selected,
            "advance_to_three_seed_learning": selected is not None,
            "run_end_to_end": False,
            "default_enabled": False,
            "reason": (
                "stable_speed_gate_passed"
                if selected is not None
                else "numeric_stability_or_speed_gate_failed"
            ),
        },
        "speed_gate": {
            "current_batched_learner_runs_agent_steps_per_second": (
                comparison_speeds
            ),
            "current_three_run_relative_range": variability,
        },
        "checkpoint": {
            "before": checkpoint_before,
            "after": checkpoint_after,
            "unchanged": checkpoint_before == checkpoint_after,
        },
        "sources": {
            "comparison": {
                "path": args.comparison.as_posix(),
                "sha256": _sha256(args.comparison),
            },
            "checkpoint": {
                "path": args.checkpoint.as_posix(),
                "sha256": _sha256(args.checkpoint),
            },
        },
        "passed": (
            checkpoint_before == checkpoint_after
            and all(
                bool(run["stable"]) for run in runs["float32"]
            )
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
        "decision": report["decision"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
