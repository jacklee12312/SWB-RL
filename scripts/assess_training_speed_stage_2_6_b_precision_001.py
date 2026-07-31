from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Iterator

import torch

from scripts.profile_v4_1_inference_breakdown import (
    _checkpoint_contract,
    _device_fixture,
    _fixed_fixture,
    summarize_samples,
)
from scripts.scan_training_speed_stage_2_4 import (
    DEFAULT_CHECKPOINT,
    _json,
    _repo_path,
    _sha256,
)
from swb.db.repository import CardRepository
from swb.rl.checkpoint import load_checkpoint
from swb.rl.runtime import WorkerAssetsSnapshot


BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
MODES = ("tf32", "fp16_autocast", "bf16_autocast")
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_COMPARISON = Path(
    "data/reports/training_speed/stage_2_5_a_obs_001.json"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/"
    "stage_2_6_b_precision_001_gate.json"
)
RECURRENT_STEPS = 128


@contextlib.contextmanager
def _precision_mode(name: str) -> Iterator[None]:
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        if name == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            with contextlib.nullcontext():
                yield
        elif name == "fp16_autocast":
            with torch.autocast("cuda", dtype=torch.float16):
                yield
        elif name == "bf16_autocast":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yield
        elif name == "fp32":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            with contextlib.nullcontext():
                yield
        else:
            raise ValueError(f"unknown precision mode: {name}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_precision)


def _forward(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    mode: str,
) -> tuple[torch.Tensor, ...]:
    with torch.no_grad(), _precision_mode(mode):
        return model.forward_step(
            tensors["observation"],
            tensors["hidden"],
            tensors["card_indices"],
        )


def _benchmark(
    model: torch.nn.Module,
    fixture: dict[str, object],
    *,
    mode: str,
    device: torch.device,
) -> dict[str, object]:
    results = {}
    for batch_size in BATCH_SIZES:
        tensors = _device_fixture(
            fixture,
            batch_size=batch_size,
            device=device,
        )
        for _ in range(8):
            _forward(model, tensors, mode)
        torch.cuda.synchronize()
        host_samples = []
        device_samples = []
        for _ in range(3):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            start_event.record()
            with torch.no_grad(), _precision_mode(mode):
                for _ in range(20):
                    model.forward_step(
                        tensors["observation"],
                        tensors["hidden"],
                        tensors["card_indices"],
                    )
            end_event.record()
            torch.cuda.synchronize()
            host_samples.append(
                (time.perf_counter() - started) * 1000.0 / 20
            )
            device_samples.append(
                start_event.elapsed_time(end_event) / 20
            )
        results[str(batch_size)] = {
            "host_milliseconds_per_call": summarize_samples(
                host_samples
            ),
            "device_milliseconds_per_call": summarize_samples(
                device_samples
            ),
        }
    return results


def _numeric_contract(
    model: torch.nn.Module,
    fixture: dict[str, object],
    *,
    mode: str,
    device: torch.device,
) -> dict[str, object]:
    tensors = _device_fixture(
        fixture, batch_size=64, device=device
    )
    baseline = _forward(model, tensors, "fp32")
    candidate = _forward(model, tensors, mode)
    tensor_contracts = {}
    finite = True
    for name, reference, value in zip(
        ("logits", "value", "hidden"),
        baseline,
        candidate,
    ):
        value_finite = bool(torch.isfinite(value).all())
        finite = finite and value_finite
        difference = (
            reference.float() - value.float()
        ).abs()
        tensor_contracts[name] = {
            "dtype": str(value.dtype),
            "finite": value_finite,
            "max_absolute_error": float(difference.max()),
            "mean_absolute_error": float(difference.mean()),
        }
    mask = tensors["action_mask"].to(dtype=torch.bool)
    with torch.no_grad():
        baseline_masked = model.masked_logits(
            baseline[0].float(), mask
        )
        candidate_masked = model.masked_logits(
            candidate[0].float(), mask
        )
        baseline_probabilities = torch.softmax(
            baseline_masked, dim=-1
        )
        candidate_probabilities = torch.softmax(
            candidate_masked, dim=-1
        )
        baseline_log_probs = torch.log_softmax(
            baseline_masked, dim=-1
        )
        candidate_log_probs = torch.log_softmax(
            candidate_masked, dim=-1
        )
        baseline_actions = baseline_masked.argmax(dim=-1)
        candidate_actions = candidate_masked.argmax(dim=-1)
        selected_baseline = baseline_log_probs.gather(
            1, baseline_actions.unsqueeze(-1)
        )
        selected_candidate = candidate_log_probs.gather(
            1, baseline_actions.unsqueeze(-1)
        )
    probability_difference = (
        baseline_probabilities - candidate_probabilities
    ).abs()
    argmax_flips = int(
        (baseline_actions != candidate_actions).sum()
    )
    probability_contract = {
        "finite": (
            bool(torch.isfinite(candidate_probabilities).all())
            and bool(torch.isfinite(candidate_log_probs).all())
        ),
        "max_absolute_error": float(probability_difference.max()),
        "mean_absolute_error": float(probability_difference.mean()),
        "argmax_flips": argmax_flips,
        "argmax_flip_rate": argmax_flips / baseline_actions.numel(),
        "selected_log_probability_max_absolute_error": float(
            (selected_baseline - selected_candidate).abs().max()
        ),
    }

    recurrent_tensors = _device_fixture(
        fixture, batch_size=4, device=device
    )
    baseline_hidden = recurrent_tensors["hidden"].clone()
    candidate_hidden = recurrent_tensors["hidden"].clone()
    recurrent_finite = True
    recurrent_argmax_flips = 0
    max_hidden_error = 0.0
    max_probability_error = 0.0
    for _ in range(RECURRENT_STEPS):
        baseline_inputs = {
            **recurrent_tensors,
            "hidden": baseline_hidden,
        }
        candidate_inputs = {
            **recurrent_tensors,
            "hidden": candidate_hidden,
        }
        baseline_step = _forward(
            model, baseline_inputs, "fp32"
        )
        candidate_step = _forward(
            model, candidate_inputs, mode
        )
        baseline_hidden = baseline_step[2]
        candidate_hidden = candidate_step[2]
        recurrent_finite = (
            recurrent_finite
            and all(
                bool(torch.isfinite(value).all())
                for value in candidate_step
            )
        )
        max_hidden_error = max(
            max_hidden_error,
            float(
                (
                    baseline_hidden.float()
                    - candidate_hidden.float()
                ).abs().max()
            ),
        )
        baseline_probability = torch.softmax(
            baseline_step[0].float(), dim=-1
        )
        candidate_probability = torch.softmax(
            candidate_step[0].float(), dim=-1
        )
        max_probability_error = max(
            max_probability_error,
            float(
                (
                    baseline_probability
                    - candidate_probability
                ).abs().max()
            ),
        )
        recurrent_argmax_flips += int(
            (
                baseline_step[0].argmax(dim=-1)
                != candidate_step[0].argmax(dim=-1)
            ).sum()
        )
    recurrent_contract = {
        "steps": RECURRENT_STEPS,
        "finite": recurrent_finite,
        "max_hidden_absolute_error": max_hidden_error,
        "max_probability_absolute_error": max_probability_error,
        "argmax_flips": recurrent_argmax_flips,
        "argmax_flip_rate": (
            recurrent_argmax_flips
            / (RECURRENT_STEPS * recurrent_tensors["hidden"].shape[0])
        ),
    }
    numeric_passed = (
        finite
        and probability_contract["finite"]
        and recurrent_finite
        and probability_contract["max_absolute_error"] <= 1e-3
        and probability_contract["argmax_flip_rate"] <= 0.01
        and recurrent_contract["max_hidden_absolute_error"] <= 1e-2
        and recurrent_contract["argmax_flip_rate"] <= 0.01
    )
    return {
        "tensors": tensor_contracts,
        "probabilities": probability_contract,
        "recurrent_drift": recurrent_contract,
        "thresholds": {
            "probability_max_absolute_error": 1e-3,
            "argmax_flip_rate": 0.01,
            "recurrent_hidden_max_absolute_error": 1e-2,
        },
        "passed": numeric_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess checklist 2.6 B-PRECISION-001"
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--comparison", type=Path, default=DEFAULT_COMPARISON
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    device = torch.device("cuda")
    checkpoint_path = _repo_path(args.checkpoint)
    checkpoint_before = _checkpoint_contract(checkpoint_path)
    snapshot = WorkerAssetsSnapshot.build(
        CardRepository(_repo_path(args.database))
    )
    trainer = load_checkpoint(
        checkpoint_path,
        snapshot,
        device=str(device),
        restore_rng_state=False,
    )
    try:
        model = trainer.model
        model.eval()
        fixture = _fixed_fixture(
            model,
            maximum_batch_size=max(BATCH_SIZES),
            seed=20260801,
        )
        timings = {
            "fp32": _benchmark(
                model, fixture, mode="fp32", device=device
            )
        }
        modes = {}
        for mode in MODES:
            try:
                timing = _benchmark(
                    model, fixture, mode=mode, device=device
                )
                numeric = _numeric_contract(
                    model, fixture, mode=mode, device=device
                )
                modes[mode] = {
                    "supported": True,
                    "timing": timing,
                    "numeric": numeric,
                }
            except Exception as exc:
                modes[mode] = {
                    "supported": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
    finally:
        trainer.close()
    checkpoint_after = _checkpoint_contract(checkpoint_path)

    comparison = _json(args.comparison)
    comparison_speeds = [
        float(value)
        for value in comparison["end_to_end"][
            "runs_agent_steps_per_second"
        ]
    ]
    comparison_median = statistics.median(comparison_speeds)
    variability = (
        max(comparison_speeds) - min(comparison_speeds)
    ) / comparison_median
    baseline_batch_4 = float(
        timings["fp32"]["4"][
            "device_milliseconds_per_call"
        ]["median"]
    )
    for mode, result in modes.items():
        if not result["supported"]:
            result["decision"] = {
                "advance_to_end_to_end": False,
                "advance_to_learning_seeds": False,
                "reason": "unsupported",
            }
            continue
        candidate_batch_4 = float(
            result["timing"]["4"][
                "device_milliseconds_per_call"
            ]["median"]
        )
        relative_reduction = (
            baseline_batch_4 - candidate_batch_4
        ) / baseline_batch_4
        speed_gate = relative_reduction > variability
        numeric_gate = bool(result["numeric"]["passed"])
        result["comparison"] = {
            "baseline_batch_4_device_milliseconds": (
                baseline_batch_4
            ),
            "candidate_batch_4_device_milliseconds": (
                candidate_batch_4
            ),
            "relative_reduction": relative_reduction,
            "comparison_three_run_relative_range": variability,
            "speed_gate_passed": speed_gate,
        }
        result["decision"] = {
            "advance_to_end_to_end": speed_gate and numeric_gate,
            "advance_to_learning_seeds": False,
            "reason": (
                "micro_and_numeric_gates_passed"
                if speed_gate and numeric_gate
                else "micro_or_numeric_gate_failed"
            ),
        }
    advancing = [
        mode
        for mode, result in modes.items()
        if result["decision"]["advance_to_end_to_end"]
    ]
    report = {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_6_b_precision_001_gate"
        ),
        "checklist_section": "2.6",
        "candidate": "B-PRECISION-001",
        "classification": "B",
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "methodology": {
            "fixture_seed": 20260801,
            "batch_sizes": list(BATCH_SIZES),
            "warmup_iterations": 8,
            "measured_iterations": 20,
            "timing_repeats": 3,
            "recurrent_drift_steps": RECURRENT_STEPS,
        },
        "baseline": {
            "mode": "fp32_tf32_disabled",
            "timing": timings["fp32"],
        },
        "modes": modes,
        "decision": {
            "advancing_modes": advancing,
            "run_end_to_end": bool(advancing),
            "run_three_seed_learning": False,
            "disposition": (
                "pending_end_to_end"
                if advancing
                else "rejected_micro_or_numeric_gate"
            ),
        },
        "checkpoint": checkpoint_after,
        "sources": {
            "script": {
                "path": (
                    "scripts/"
                    "assess_training_speed_stage_2_6_b_precision_001.py"
                ),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "comparison": {
                "path": str(args.comparison).replace("\\", "/"),
                "sha256": _sha256(args.comparison),
            },
            "checkpoint": {
                "path": str(args.checkpoint).replace("\\", "/"),
                "sha256": _sha256(args.checkpoint),
            },
        },
        "passed": checkpoint_before == checkpoint_after,
    }
    output = _repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output).replace("\\", "/"),
        "passed": report["passed"],
        "advancing_modes": advancing,
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
