from __future__ import annotations

import argparse
import hashlib
import json
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
DEFAULT_BASELINE = Path(
    "data/reports/training_speed/"
    "stage_2_7_learner_baseline_summary.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/"
    "stage_2_7_a_padded_compute_001_gate.json"
)
REPEATS = 3
RTOL = 1e-5
ATOL = 1e-6


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


def _tensor_contract(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    maximum = 0.0
    mean_errors = []
    allclose = True
    finite = True
    worst_name = None
    for name, expected in reference.items():
        value = candidate[name]
        value_finite = bool(torch.isfinite(value).all())
        difference = (expected.float() - value.float()).abs()
        current = float(difference.max())
        if current > maximum:
            maximum = current
            worst_name = name
        mean_errors.append(float(difference.mean()))
        finite = finite and value_finite
        allclose = allclose and torch.allclose(
            expected, value, rtol=RTOL, atol=ATOL
        )
    return {
        "tensor_count": len(reference),
        "finite": finite,
        "allclose": allclose,
        "rtol": RTOL,
        "atol": ATOL,
        "maximum_absolute_error": maximum,
        "mean_of_tensor_mean_absolute_errors": (
            statistics.mean(mean_errors)
        ),
        "worst_tensor": worst_name,
    }


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


def _run_update(
    snapshot: WorkerAssetsSnapshot,
    checkpoint: Path,
    records,
    bootstrap,
    *,
    batched: bool,
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
        trainer._batched_v41_learner = batched
        trainer.torch_generator.manual_seed(20260810 + repeat)
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = trainer.update(records, bootstrap)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result: dict[str, object] = {
            "seconds": elapsed,
            "metrics": metrics,
            "finite_metrics": all(
                torch.isfinite(torch.tensor(value))
                for value in metrics.values()
            ),
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
    finally:
        trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess checklist 2.7 A-PADDED-COMPUTE-001"
    )
    parser.add_argument(
        "--database", type=Path, default=DEFAULT_DATABASE
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_BASELINE
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

    baseline_runs = []
    candidate_runs = []
    numeric_reference = None
    numeric_candidate = None
    for repeat in range(REPEATS):
        for batched, destination in (
            (False, baseline_runs),
            (True, candidate_runs),
        ):
            result = _run_update(
                snapshot,
                checkpoint,
                records,
                bootstrap,
                batched=batched,
                repeat=repeat,
                capture_state=repeat == 0,
            )
            if repeat == 0:
                if batched:
                    numeric_candidate = result
                else:
                    numeric_reference = result
            destination.append({
                "repeat": repeat,
                "seconds": result["seconds"],
                "metrics": result["metrics"],
                "finite_metrics": result["finite_metrics"],
            })
            print(json.dumps({
                "batched": batched,
                "repeat": repeat,
                "seconds": result["seconds"],
            }, sort_keys=True), flush=True)

    assert numeric_reference is not None
    assert numeric_candidate is not None
    model_contract = _tensor_contract(
        numeric_reference["model"],
        numeric_candidate["model"],
    )
    optimizer_contract = _tensor_contract(
        numeric_reference["optimizer"],
        numeric_candidate["optimizer"],
    )
    metric_errors = {
        name: abs(
            float(numeric_reference["metrics"][name])
            - float(numeric_candidate["metrics"][name])
        )
        for name in numeric_reference["metrics"]
    }
    baseline_times = [float(run["seconds"]) for run in baseline_runs]
    candidate_times = [
        float(run["seconds"]) for run in candidate_runs
    ]
    baseline_median = statistics.median(baseline_times)
    candidate_median = statistics.median(candidate_times)
    relative_reduction = (
        baseline_median - candidate_median
    ) / baseline_median
    baseline_report = _json(args.baseline)
    variability = float(
        baseline_report["materiality_gates"][
            "comparison_three_run_relative_range"
        ]
    )
    numeric_gate = (
        model_contract["allclose"]
        and optimizer_contract["allclose"]
        and model_contract["finite"]
        and optimizer_contract["finite"]
        and all(run["finite_metrics"] for run in candidate_runs)
    )
    speed_gate = relative_reduction > variability
    checkpoint_after = {
        "sha256": _sha256(checkpoint),
        "size": checkpoint.stat().st_size,
        "mtime_ns": checkpoint.stat().st_mtime_ns,
    }
    report = {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_7_"
            "a_padded_compute_001_gate"
        ),
        "checklist_section": "2.7",
        "candidate": "A-PADDED-COMPUTE-001",
        "classification": "A",
        "methodology": {
            "same_collected_rollout": True,
            "same_minibatch_permutation_seed_per_pair": True,
            "repeats": REPEATS,
            "records": len(records),
            "rollout_workers": 6,
            "worker_torch_threads": 2,
            "central_inference_batch_wait_seconds": 0.001,
        },
        "timing": {
            "baseline_seconds": baseline_times,
            "candidate_seconds": candidate_times,
            "baseline_median_seconds": baseline_median,
            "candidate_median_seconds": candidate_median,
            "relative_reduction": relative_reduction,
            "comparison_three_run_relative_range": variability,
            "speed_gate_passed": speed_gate,
        },
        "numeric": {
            "model_after_one_update": model_contract,
            "optimizer_after_one_update": optimizer_contract,
            "metric_absolute_errors": metric_errors,
            "strict_a_gate_passed": numeric_gate,
        },
        "decision": {
            "adopt_as_a": speed_gate and numeric_gate,
            "reclassify_as_b": (
                speed_gate
                and not numeric_gate
                and model_contract["finite"]
                and optimizer_contract["finite"]
            ),
            "run_end_to_end_as_a": speed_gate and numeric_gate,
            "reason": (
                "a_speed_and_numeric_gates_passed"
                if speed_gate and numeric_gate
                else (
                    "strict_a_numeric_gate_failed"
                    if speed_gate
                    else "speed_gate_failed"
                )
            ),
        },
        "checkpoint": {
            "before": checkpoint_before,
            "after": checkpoint_after,
            "unchanged": checkpoint_before == checkpoint_after,
        },
        "sources": {
            "baseline": {
                "path": args.baseline.as_posix(),
                "sha256": _sha256(args.baseline),
            },
            "checkpoint": {
                "path": args.checkpoint.as_posix(),
                "sha256": _sha256(args.checkpoint),
            },
        },
        "passed": checkpoint_before == checkpoint_after,
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
