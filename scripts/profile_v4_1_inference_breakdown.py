from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from swb.db.repository import CardRepository
from swb.rl.checkpoint import load_checkpoint
from swb.rl.runtime import WorkerAssetsSnapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_V4_CHECKPOINT = Path(
    "data/checkpoints/training_speed/"
    "frozen_v4_1_seed_20260801_500k.pt"
)
DEFAULT_V3_CHECKPOINT = Path(
    "data/checkpoints/training_speed/"
    "frozen_v3_6_seed_20260801_500k.pt"
)
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/v4_1_inference_breakdown.json"
)
DEFAULT_TRACE = Path(
    "data/reports/training_speed/v4_1_profiler_trace.json.gz"
)
DEFAULT_STAGE_2_2_CENTRAL = Path(
    "data/reports/training_speed/stage_2_2_central_inference_smoke.json"
)
DEFAULT_STAGE_2_2_LEARNER = Path(
    "data/reports/training_speed/stage_2_2_learner_timing_smoke.json"
)
BATCH_SIZES = (1, 4, 8, 16, 32, 64)
EPISODE_LENGTHS = (1, 16, 64, 256)
LEGAL_ACTION_COUNTS = (1, 8, 32, 64, 112)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1),
    )
    return ordered[index]


def summarize_samples(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {
            "samples": [],
            "sample_count": 0,
            "median": 0.0,
            "p95": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
    samples = [float(value) for value in values]
    return {
        "samples": samples,
        "sample_count": len(samples),
        "median": statistics.median(samples),
        "p95": _percentile(samples, 0.95),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fixed_fixture(
    model: nn.Module,
    *,
    maximum_batch_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    observation = rng.uniform(
        0.0,
        1.0,
        size=(maximum_batch_size, int(model.input_size)),
    ).astype(np.float32)
    cards = rng.integers(
        0,
        int(model.card_vocabulary_size) + 1,
        size=(maximum_batch_size, int(model.card_slot_count)),
        dtype=np.int64,
    )
    # Keep a deterministic mix of padding and populated slots.
    cards[:, ::5] = 0
    action_mask = np.ones(
        (maximum_batch_size, int(model.action_size)),
        dtype=np.bool_,
    )
    hidden = np.zeros(
        (maximum_batch_size, int(model.hidden_size)),
        dtype=np.float32,
    )
    return {
        "observation": np.ascontiguousarray(observation),
        "card_indices": np.ascontiguousarray(cards),
        "action_mask": np.ascontiguousarray(action_mask),
        "hidden": np.ascontiguousarray(hidden),
    }


def _device_fixture(
    fixture: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(
            values[:batch_size],
            device=device,
        )
        for name, values in fixture.items()
    }


def _timed_repeats(
    function: Callable[[], object],
    *,
    device: torch.device,
    warmup_iterations: int,
    measured_iterations: int,
    repeats: int,
) -> dict[str, object]:
    with torch.no_grad():
        for _ in range(warmup_iterations):
            function()
    _synchronize(device)
    host_samples = []
    device_samples = []
    peak_memory = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        host_started = time.perf_counter()
        with torch.no_grad():
            for _ in range(measured_iterations):
                function()
        if device.type == "cuda":
            end_event.record()
        _synchronize(device)
        host_ms = (
            (time.perf_counter() - host_started)
            * 1000.0
            / measured_iterations
        )
        if device.type == "cuda":
            device_ms = (
                start_event.elapsed_time(end_event) / measured_iterations
            )
            peak_memory.append(float(torch.cuda.max_memory_allocated(device)))
        else:
            device_ms = host_ms
            peak_memory.append(0.0)
        host_samples.append(host_ms)
        device_samples.append(device_ms)
    return {
        "host_milliseconds_per_call": summarize_samples(host_samples),
        "device_milliseconds_per_call": summarize_samples(device_samples),
        "peak_allocated_bytes": summarize_samples(peak_memory),
    }


def benchmark_forward_batches(
    model: nn.Module,
    fixture: Mapping[str, np.ndarray],
    *,
    batch_sizes: Sequence[int],
    device: torch.device,
    warmup_iterations: int,
    measured_iterations: int,
    repeats: int,
) -> dict[str, object]:
    results: dict[str, object] = {}
    model.eval()
    for batch_size in batch_sizes:
        tensors = _device_fixture(
            fixture,
            batch_size=batch_size,
            device=device,
        )

        def forward() -> object:
            return model.forward_step(
                tensors["observation"],
                tensors["hidden"],
                tensors["card_indices"],
            )

        timing = _timed_repeats(
            forward,
            device=device,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            repeats=repeats,
        )
        device_ms = timing["device_milliseconds_per_call"]
        assert isinstance(device_ms, Mapping)
        median_ms = float(device_ms["median"])
        timing["batch_size"] = int(batch_size)
        timing["samples_per_second"] = (
            float(batch_size) * 1000.0 / median_ms
            if median_ms
            else 0.0
        )
        results[str(batch_size)] = timing
    return results


def _profile_forward_components_once(
    model: nn.Module,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    device = tensors["observation"].device
    use_cuda = device.type == "cuda"
    markers: dict[str, Any] = {}
    handles = []
    phase = {"value": "input_encoding"}
    module_calls: list[dict[str, object]] = []
    active_calls: dict[str, list[int]] = defaultdict(list)

    def mark(name: str) -> None:
        if use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            markers[name] = event
        else:
            markers[name] = time.perf_counter()

    def stage_pre(name: str, next_phase: str):
        def hook(_module, _inputs) -> None:
            mark(f"{name}_start")
            phase["value"] = next_phase

        return hook

    def stage_post(name: str, next_phase: str):
        def hook(_module, _inputs, _output) -> None:
            mark(f"{name}_end")
            phase["value"] = next_phase

        return hook

    def marker_pre(name: str):
        def hook(_module, _inputs) -> None:
            mark(f"{name}_start")

        return hook

    def marker_post(name: str):
        def hook(_module, _inputs, _output) -> None:
            mark(f"{name}_end")

        return hook

    encoder = getattr(model, "entity_encoder", None)
    recurrent = getattr(model, "recurrent", None)
    if encoder is None or recurrent is None:
        raise ValueError(
            "component profiling requires entity_encoder and recurrent"
        )
    handles.append(
        encoder.register_forward_pre_hook(
            stage_pre("transformer", "transformer")
        )
    )
    handles.append(
        encoder.register_forward_hook(
            stage_post("transformer", "transformer_to_gru")
        )
    )
    handles.append(
        recurrent.register_forward_pre_hook(stage_pre("gru", "gru"))
    )
    handles.append(
        recurrent.register_forward_hook(stage_post("gru", "action"))
    )
    for head_name in ("policy_head", "value_head"):
        head = getattr(model, head_name, None)
        if head is not None:
            handles.append(
                head.register_forward_pre_hook(marker_pre(head_name))
            )
            handles.append(
                head.register_forward_hook(marker_post(head_name))
            )

    def module_pre(module_name: str):
        def hook(_module, _inputs) -> None:
            call_index = len(module_calls)
            module_calls.append({
                "name": module_name,
                "phase": phase["value"],
                "start": f"module_{call_index}_start",
                "end": f"module_{call_index}_end",
            })
            active_calls[module_name].append(call_index)
            mark(f"module_{call_index}_start")

        return hook

    def module_post(module_name: str):
        def hook(_module, _inputs, _output) -> None:
            call_index = active_calls[module_name].pop()
            mark(f"module_{call_index}_end")

        return hook

    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            handles.append(
                module.register_forward_pre_hook(module_pre(module_name))
            )
            handles.append(
                module.register_forward_hook(module_post(module_name))
            )

    try:
        mark("model_start")
        with torch.no_grad():
            model.forward_step(
                tensors["observation"],
                tensors["hidden"],
                tensors["card_indices"],
            )
        mark("model_end")
    finally:
        for handle in handles:
            handle.remove()
    _synchronize(device)

    def elapsed(start: str, end: str) -> float:
        if use_cuda:
            return markers[start].elapsed_time(markers[end])
        return float(markers[end] - markers[start]) * 1000.0

    call_times: list[dict[str, object]] = []
    for call in module_calls:
        call_times.append({
            **call,
            "milliseconds": elapsed(
                str(call["start"]),
                str(call["end"]),
            ),
        })

    def sum_calls(
        *,
        selected_phase: str | None = None,
        exact_name: str | None = None,
        module_type: type[nn.Module] | None = None,
        excluded_names: frozenset[str] = frozenset(),
    ) -> float:
        module_lookup = dict(model.named_modules())
        return sum(
            float(call["milliseconds"])
            for call in call_times
            if (
                (selected_phase is None or call["phase"] == selected_phase)
                and (exact_name is None or call["name"] == exact_name)
                and call["name"] not in excluded_names
                and (
                    module_type is None
                    or isinstance(module_lookup[str(call["name"])], module_type)
                )
            )
        )

    input_total = elapsed("model_start", "transformer_start")
    card_embedding = sum_calls(exact_name="card_embedding")
    card_projection = sum_calls(exact_name="card_projection")
    non_card_projection = sum_calls(
        selected_phase="input_encoding",
        module_type=nn.Linear,
        excluded_names=frozenset({"card_projection"}),
    )
    other_input_embedding = sum_calls(
        selected_phase="input_encoding",
        module_type=nn.Embedding,
        excluded_names=frozenset({"card_embedding"}),
    )
    known_input = (
        card_embedding
        + card_projection
        + non_card_projection
        + other_input_embedding
    )
    return {
        "model_forward_milliseconds": elapsed("model_start", "model_end"),
        "structured_token_construction_milliseconds": input_total,
        "card_embedding_lookup_milliseconds": card_embedding,
        "card_projection_milliseconds": card_projection,
        "non_card_numeric_projection_milliseconds": non_card_projection,
        "other_input_embedding_milliseconds": other_input_embedding,
        "token_tensor_ops_and_launch_gap_residual_milliseconds": max(
            0.0,
            input_total - known_input,
        ),
        "transformer_milliseconds": elapsed(
            "transformer_start", "transformer_end"
        ),
        "transformer_to_gru_milliseconds": elapsed(
            "transformer_end", "gru_start"
        ),
        "gru_milliseconds": elapsed("gru_start", "gru_end"),
        "action_value_stage_milliseconds": elapsed(
            "gru_end", "model_end"
        ),
        "policy_head_milliseconds": elapsed(
            "policy_head_start", "policy_head_end"
        ),
        "value_head_milliseconds": elapsed(
            "value_head_start", "value_head_end"
        ),
    }


def profile_forward_components(
    model: nn.Module,
    fixture: Mapping[str, np.ndarray],
    *,
    batch_sizes: Sequence[int],
    repeats: int,
    device: torch.device,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for batch_size in batch_sizes:
        tensors = _device_fixture(
            fixture,
            batch_size=batch_size,
            device=device,
        )
        with torch.no_grad():
            model.forward_step(
                tensors["observation"],
                tensors["hidden"],
                tensors["card_indices"],
            )
        _synchronize(device)
        samples = [
            _profile_forward_components_once(model, tensors)
            for _ in range(repeats)
        ]
        fields: dict[str, object] = {}
        for field in samples[0]:
            fields[field] = summarize_samples([
                sample[field] for sample in samples
            ])
        input_median = float(
            fields["structured_token_construction_milliseconds"]["median"]
        )
        forward_median = float(
            fields["model_forward_milliseconds"]["median"]
        )
        fields["structured_token_fraction_of_forward"] = (
            input_median / forward_median if forward_median else 0.0
        )
        results[str(batch_size)] = {
            "batch_size": int(batch_size),
            "fields": fields,
        }
    return results


def benchmark_input_packing(
    fixture: Mapping[str, np.ndarray],
    *,
    batch_sizes: Sequence[int],
    device: torch.device,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for batch_size in batch_sizes:
        rows = {
            name: [array[index] for index in range(batch_size)]
            for name, array in fixture.items()
            if name != "hidden"
        }
        cpu_stack_samples = []
        tensor_samples = []
        h2d_samples = []
        input_bytes = 0
        for _ in range(repeats):
            stack_started = time.perf_counter()
            stacked = {}
            for _ in range(iterations):
                stacked = {
                    name: np.stack(values)
                    for name, values in rows.items()
                }
            cpu_stack_samples.append(
                (time.perf_counter() - stack_started)
                * 1000.0
                / iterations
            )
            input_bytes = sum(array.nbytes for array in stacked.values())

            tensor_started = time.perf_counter()
            cpu_tensors = {}
            for _ in range(iterations):
                cpu_tensors = {
                    "observation": torch.from_numpy(
                        stacked["observation"]
                    ),
                    "card_indices": torch.from_numpy(
                        stacked["card_indices"]
                    ).to(dtype=torch.long),
                    "action_mask": torch.from_numpy(
                        stacked["action_mask"]
                    ).to(dtype=torch.bool),
                }
            tensor_samples.append(
                (time.perf_counter() - tensor_started)
                * 1000.0
                / iterations
            )

            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(iterations):
                    for value in cpu_tensors.values():
                        value.to(device=device)
                end_event.record()
                _synchronize(device)
                h2d_samples.append(
                    start_event.elapsed_time(end_event) / iterations
                )
            else:
                h2d_started = time.perf_counter()
                for _ in range(iterations):
                    for value in cpu_tensors.values():
                        value.to(device=device)
                h2d_samples.append(
                    (time.perf_counter() - h2d_started)
                    * 1000.0
                    / iterations
                )
        results[str(batch_size)] = {
            "batch_size": int(batch_size),
            "stack_copy_operations_per_batch": 3,
            "input_bytes_per_batch": int(input_bytes),
            "input_bytes_per_request": input_bytes / batch_size,
            "cpu_numpy_stack_milliseconds": summarize_samples(
                cpu_stack_samples
            ),
            "cpu_tensor_construction_milliseconds": summarize_samples(
                tensor_samples
            ),
            "host_to_device_milliseconds": summarize_samples(h2d_samples),
            "pinned_memory": False,
            "non_blocking": False,
        }
    return results


def benchmark_recurrent_lengths(
    model: nn.Module,
    *,
    episode_lengths: Sequence[int],
    batch_size: int,
    total_steps: int,
    repeats: int,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    recurrent_input = torch.randn(
        batch_size,
        int(model.recurrent.input_size),
        generator=generator,
    ).to(device)
    results: dict[str, object] = {}
    for episode_length in episode_lengths:
        episode_count = math.ceil(total_steps / episode_length)
        executed_steps = episode_count * episode_length
        host_samples = []
        device_samples = []
        for _ in range(repeats):
            _synchronize(device)
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            host_started = time.perf_counter()
            with torch.no_grad():
                for _ in range(episode_count):
                    hidden = model.initial_state(
                        batch_size,
                        device=device,
                    )
                    for _ in range(episode_length):
                        hidden = model.recurrent(recurrent_input, hidden)
            if device.type == "cuda":
                end_event.record()
            _synchronize(device)
            host_samples.append(
                (time.perf_counter() - host_started)
                * 1000.0
                / executed_steps
            )
            if device.type == "cuda":
                device_samples.append(
                    start_event.elapsed_time(end_event) / executed_steps
                )
            else:
                device_samples.append(host_samples[-1])
        results[str(episode_length)] = {
            "episode_length": int(episode_length),
            "episode_count": int(episode_count),
            "executed_recurrent_steps": int(executed_steps),
            "batch_size": int(batch_size),
            "host_milliseconds_per_recurrent_step": summarize_samples(
                host_samples
            ),
            "device_milliseconds_per_recurrent_step": summarize_samples(
                device_samples
            ),
        }
    return results


def benchmark_legal_action_counts(
    model: nn.Module,
    fixture: Mapping[str, np.ndarray],
    *,
    legal_counts: Sequence[int],
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    repeats: int,
    device: torch.device,
) -> dict[str, object]:
    tensors = _device_fixture(
        fixture,
        batch_size=batch_size,
        device=device,
    )
    with torch.no_grad():
        logits, _, _ = model.forward_step(
            tensors["observation"],
            tensors["hidden"],
            tensors["card_indices"],
        )
    _synchronize(device)
    results: dict[str, object] = {}
    for legal_count in legal_counts:
        if legal_count > int(model.action_size):
            continue
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, :legal_count] = True

        def distribution() -> object:
            masked = model.masked_logits(logits, mask)
            return (
                torch.softmax(masked, dim=-1),
                torch.log_softmax(masked, dim=-1),
            )

        timing = _timed_repeats(
            distribution,
            device=device,
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            repeats=repeats,
        )
        results[str(legal_count)] = {
            "legal_action_count": int(legal_count),
            **timing,
        }
    return {
        "action_size": int(model.action_size),
        "batch_size": int(batch_size),
        "dense_action_scoring_is_mask_independent": True,
        "mask_is_applied_after_forward_step": True,
        "distribution_by_legal_action_count": results,
    }


def _trace_summary(trace_path: Path) -> dict[str, object]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", [])
    kernels = [
        event
        for event in events
        if (
            event.get("ph") == "X"
            and float(event.get("dur", 0.0)) >= 0.0
            and "kernel" in str(event.get("cat", "")).lower()
        )
    ]
    kernel_groups: dict[tuple[object, object], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    kernel_totals: dict[str, float] = defaultdict(float)
    for event in kernels:
        kernel_groups[(event.get("pid"), event.get("tid"))].append(event)
        kernel_totals[str(event.get("name", "unknown"))] += float(
            event.get("dur", 0.0)
        )
    gaps = []
    for group in kernel_groups.values():
        ordered = sorted(group, key=lambda event: float(event.get("ts", 0.0)))
        for previous, current in zip(ordered, ordered[1:]):
            previous_end = float(previous.get("ts", 0.0)) + float(
                previous.get("dur", 0.0)
            )
            gaps.append(max(0.0, float(current.get("ts", 0.0)) - previous_end))
    launches = [
        event
        for event in events
        if "launchkernel" in str(event.get("name", "")).lower()
    ]
    synchronizations = [
        event
        for event in events
        if "synchroniz" in str(event.get("name", "")).lower()
    ]
    top_kernels = sorted(
        (
            {"name": name, "total_microseconds": duration}
            for name, duration in kernel_totals.items()
        ),
        key=lambda item: float(item["total_microseconds"]),
        reverse=True,
    )[:25]
    return {
        "kernel_event_count": len(kernels),
        "kernel_launch_event_count": len(launches),
        "synchronization_event_count": len(synchronizations),
        "kernel_gap_microseconds": summarize_samples(gaps),
        "top_kernels": top_kernels,
        "synchronization_events": [
            {
                "name": str(event.get("name", "")),
                "duration_microseconds": float(event.get("dur", 0.0)),
            }
            for event in synchronizations[:50]
        ],
    }


def run_torch_profiler(
    model: nn.Module,
    fixture: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    iterations: int,
    device: torch.device,
    trace_path: Path,
) -> dict[str, object]:
    tensors = _device_fixture(
        fixture,
        batch_size=batch_size,
        device=device,
    )
    with torch.no_grad():
        for _ in range(3):
            model.forward_step(
                tensors["observation"],
                tensors["hidden"],
                tensors["card_indices"],
            )
    _synchronize(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        with torch.no_grad():
            for _ in range(iterations):
                model.forward_step(
                    tensors["observation"],
                    tensors["hidden"],
                    tensors["card_indices"],
                )
        _synchronize(device)

    operator_rows = []
    for event in profiler.key_averages():
        self_device = float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
        )
        device_total = float(
            getattr(
                event,
                "device_time_total",
                getattr(event, "cuda_time_total", 0.0),
            )
        )
        operator_rows.append({
            "name": str(event.key),
            "calls": int(event.count),
            "self_cpu_microseconds": float(event.self_cpu_time_total),
            "cpu_total_microseconds": float(event.cpu_time_total),
            "self_device_microseconds": self_device,
            "device_total_microseconds": device_total,
            "cpu_memory_bytes": int(event.cpu_memory_usage),
            "device_memory_bytes": int(
                getattr(
                    event,
                    "device_memory_usage",
                    getattr(event, "cuda_memory_usage", 0),
                )
            ),
        })
    operator_rows.sort(
        key=lambda row: (
            float(row["self_device_microseconds"]),
            float(row["self_cpu_microseconds"]),
        ),
        reverse=True,
    )

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = trace_path.with_suffix("")
    if raw_path.suffix == ".json":
        pass
    else:
        raw_path = trace_path.parent / f"{trace_path.stem}.json"
    profiler.export_chrome_trace(str(raw_path))
    trace = _trace_summary(raw_path)
    with raw_path.open("rb") as source, gzip.open(
        trace_path, "wb"
    ) as destination:
        shutil.copyfileobj(source, destination)
    raw_path.unlink()
    return {
        "batch_size": int(batch_size),
        "iterations": int(iterations),
        "activities": [
            "CPU",
            *(["CUDA"] if device.type == "cuda" else []),
        ],
        "top_operators": operator_rows[:50],
        "trace": trace,
        "compressed_trace_path": _relative(trace_path),
        "compressed_trace_sha256": _sha256(trace_path),
    }


def _source_line(
    path: Path,
    needle: str,
    *,
    after: str | None = None,
    use_last: bool = False,
) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = 0
    if after is not None:
        for index, line in enumerate(lines):
            if after in line:
                start = index + 1
                break
        else:
            raise ValueError(f"{after!r} not found in {path}")
    matches = [
        index + 1
        for index, line in enumerate(lines[start:], start=start)
        if needle in line
    ]
    if matches:
        return matches[-1] if use_last else matches[0]
    raise ValueError(f"{needle!r} not found in {path}")


def duplicate_work_audit() -> dict[str, object]:
    rollout = ROOT / "swb/rl/vector_rollout.py"
    policy = ROOT / "swb/rl/policy.py"
    return {
        "observation_conversion": {
            "status": "confirmed_duplicate_candidate",
            "finding": (
                "env.step constructs the next observation, but the worker "
                "does not reuse result.observation and calls env.observation "
                "again at the next decision"
            ),
            "evidence": [
                {
                    "path": _relative(rollout),
                    "line": _source_line(
                        rollout,
                        "result = env.step(action, timing=step_timing)",
                    ),
                },
                {
                    "path": _relative(rollout),
                    "line": _source_line(
                        rollout,
                        "observation = env.observation(",
                        after="observation_started = time.perf_counter()",
                    ),
                },
            ],
            "deferred_to": "2.5 Observation v4.1 hot-path optimization",
        },
        "central_stack_and_mask_copy": {
            "status": "confirmed_repeated_copy_candidate",
            "finding": (
                "each central batch allocates three np.stack outputs and "
                "copies observation, card indices, and action mask to device"
            ),
            "evidence": [{
                "path": _relative(rollout),
                "line": _source_line(
                    rollout,
                    "observations_np = np.stack([",
                ),
            }],
            "deferred_to": "2.4 central batching/buffer reuse",
        },
        "card_embedding": {
            "status": "no_duplicate_within_forward",
            "finding": (
                "v4.1 computes card_embedding/card_projection once per "
                "forward and reuses the resulting card_tokens"
            ),
            "evidence": [{
                "path": _relative(policy),
                "line": _source_line(
                    policy,
                    "card_tokens = self.card_projection(",
                ),
            }],
        },
        "semantic_position_tensor": {
            "status": "confirmed_reconstruction_candidate",
            "finding": (
                "_v41_semantic_context recreates torch.arange(4) on every "
                "call instead of reusing a registered static buffer"
            ),
            "evidence": [{
                "path": _relative(policy),
                "line": _source_line(
                    policy,
                    "positions = torch.arange(4, device=values.device)",
                ),
            }],
            "deferred_to": "2.6 v4.1 network forward optimization",
        },
        "device_validation_sync": {
            "status": "confirmed_host_sync_candidate",
            "finding": (
                "Python bool conversions validate device tensors in "
                "forward_step and masked_logits, forcing host-visible scalar "
                "results in the inference hot path"
            ),
            "evidence": [
                {
                    "path": _relative(policy),
                    "line": _source_line(
                        policy,
                        "if bool((card_indices < 0).any()) or bool(",
                        use_last=True,
                    ),
                },
                {
                    "path": _relative(policy),
                    "line": _source_line(
                        policy,
                        "if not bool(mask.any(dim=-1).all()):",
                    ),
                },
            ],
            "deferred_to": "2.6 equivalent hot-path validation optimization",
        },
    }


def build_bottleneck_ranking(
    *,
    component_profile: Mapping[str, object],
    forward_batches: Mapping[str, object],
    live_central_report: Mapping[str, object],
    live_learner_report: Mapping[str, object],
) -> list[dict[str, object]]:
    batch_one = component_profile["1"]
    assert isinstance(batch_one, Mapping)
    fields = batch_one["fields"]
    assert isinstance(fields, Mapping)
    input_fraction = float(fields["structured_token_fraction_of_forward"])
    batch_one_throughput = float(forward_batches["1"]["samples_per_second"])
    batch_four_throughput = float(forward_batches["4"]["samples_per_second"])
    batching_scale = (
        batch_four_throughput / batch_one_throughput
        if batch_one_throughput
        else 0.0
    )

    live_iteration = live_central_report.get("iterations", [{}])[0]
    live_collect = (
        live_iteration.get("collect", {})
        if isinstance(live_iteration, Mapping)
        else {}
    )
    empty_fraction = float(
        live_collect.get("central_batch_empty_slot_fraction", 0.0)
    )
    live_learner_iterations = live_learner_report.get("iterations", [])
    learner_update_seconds = sum(
        float(item.get("update", {}).get("update_total_seconds", 0.0))
        for item in live_learner_iterations
        if isinstance(item, Mapping)
    )
    learner_forward_backward = sum(
        float(item.get("update", {}).get("learner_forward_seconds", 0.0))
        + float(
            item.get("update", {}).get("learner_backward_seconds", 0.0)
        )
        for item in live_learner_iterations
        if isinstance(item, Mapping)
    )
    learner_fraction = (
        learner_forward_backward / learner_update_seconds
        if learner_update_seconds
        else 0.0
    )
    candidates = [
        {
            "rank": 1,
            "candidate": "v4.1 structured-token construction and launch/sync",
            "target_stage": "2.6",
            "classification": "A candidates first",
            "ranking_basis": (
                "shared root cause in central inference and learner forward"
            ),
            "evidence": {
                "synthetic_batch_1_fraction_of_forward": input_fraction,
                "live_stage_2_2_input_encoding_fraction": 0.684,
            },
            "next_experiment": (
                "remove repeated static tensor creation/device scalar sync "
                "one candidate at a time, then remeasure fixed batches"
            ),
        },
        {
            "rank": 2,
            "candidate": "central batch occupancy and reusable input buffers",
            "target_stage": "2.4",
            "classification": "A",
            "ranking_basis": (
                "independent central opportunity after the shared model "
                "hot path; fixed-input latency stays nearly flat as batch grows"
            ),
            "evidence": {
                "live_empty_slot_fraction": empty_fraction,
                "synthetic_batch_4_throughput_scale_vs_batch_1": batching_scale,
            },
            "next_experiment": (
                "scan batch wait/workers before changing execution semantics"
            ),
        },
        {
            "rank": 3,
            "candidate": "learner model forward/backward",
            "target_stage": "2.7",
            "classification": "A before numerical changes",
            "ranking_basis": (
                "large pipeline consumer, but partly overlaps the first "
                "shared-network candidate and must not be double-counted"
            ),
            "evidence": {
                "live_learner_forward_backward_fraction": learner_fraction,
            },
            "next_experiment": (
                "profile minibatch kernels and test buffer/layout reuse "
                "without changing PPO epochs or sequence boundaries"
            ),
        },
        {
            "rank": 4,
            "candidate": "duplicate next-observation construction",
            "target_stage": "2.5",
            "classification": "A",
            "ranking_basis": (
                "confirmed semantic-preserving candidate with smaller "
                "measured wall-time scope"
            ),
            "evidence": {
                "live_v4_1_observation_step_return_seconds": 3.7995651024393737,
                "live_profile_agent_steps": 2282,
            },
            "next_experiment": (
                "reuse the exact next-decision observation only after a "
                "trajectory/action-mask equivalence test"
            ),
        },
    ]
    return candidates


def _checkpoint_contract(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "observation_schema": payload["versions"]["observation_version"],
        "configuration_observation_version": payload["trainer"]["config"][
            "observation_version"
        ],
        "policy_architecture": payload["trainer"]["config"][
            "policy_architecture"
        ],
    }


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != 1:
        raise ValueError("unexpected report schema")
    methodology = report.get("methodology", {})
    if not isinstance(methodology, Mapping):
        raise ValueError("methodology must be an object")
    if methodology.get("v3_6_scope") != "pure_forward_reference_only":
        raise ValueError("v3.6 scope must remain pure-forward only")
    expected = [str(value) for value in BATCH_SIZES]
    forward = report.get("fixed_input_forward", {})
    if not isinstance(forward, Mapping):
        raise ValueError("fixed_input_forward must be an object")
    for version in ("v4.1", "v3.6"):
        values = forward.get(version, {})
        if not isinstance(values, Mapping) or sorted(
            values, key=int
        ) != expected:
            raise ValueError(f"{version} batch sweep is incomplete")
        for batch in values.values():
            samples = batch["device_milliseconds_per_call"]
            if int(samples["sample_count"]) < 3:
                raise ValueError("every fixed batch requires three repeats")
    components = report.get("v4_1_component_profile", {})
    if not isinstance(components, Mapping):
        raise ValueError("v4.1 component profile missing")
    required_components = {
        "structured_token_construction_milliseconds",
        "card_embedding_lookup_milliseconds",
        "non_card_numeric_projection_milliseconds",
        "transformer_milliseconds",
        "gru_milliseconds",
        "action_value_stage_milliseconds",
    }
    for batch in components.values():
        if not required_components <= set(batch["fields"]):
            raise ValueError("component profile is incomplete")
    if not report.get("profiler", {}).get("trace", {}).get(
        "kernel_event_count"
    ):
        raise ValueError("CUDA profiler kernel evidence is missing")
    if len(report.get("bottleneck_ranking", [])) < 3:
        raise ValueError("bottleneck ranking is incomplete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile actionable v4.1 inference bottlenecks with v3.6 used "
            "only as a fixed-input pure-forward reference."
        )
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--v4-checkpoint", type=Path, default=DEFAULT_V4_CHECKPOINT
    )
    parser.add_argument(
        "--v3-checkpoint", type=Path, default=DEFAULT_V3_CHECKPOINT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(BATCH_SIZES),
    )
    parser.add_argument("--warmup-iterations", type=int, default=8)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--packing-iterations", type=int, default=50)
    parser.add_argument("--recurrent-total-steps", type=int, default=512)
    parser.add_argument("--profiler-batch-size", type=int, default=4)
    parser.add_argument("--profiler-iterations", type=int, default=3)
    parser.add_argument("--fixture-seed", type=int, default=20260801)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-output", type=Path, default=DEFAULT_TRACE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if sorted(args.batch_sizes) != list(BATCH_SIZES):
        raise SystemExit(
            "--batch-sizes must be exactly 1 4 8 16 32 64 for checklist 2.3"
        )
    for name in (
        "warmup_iterations",
        "measured_iterations",
        "repeats",
        "packing_iterations",
        "recurrent_total_steps",
        "profiler_batch_size",
        "profiler_iterations",
    ):
        if int(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    database = _repo_path(args.database)
    v4_checkpoint = _repo_path(args.v4_checkpoint)
    v3_checkpoint = _repo_path(args.v3_checkpoint)
    output = _repo_path(args.output)
    trace_output = _repo_path(args.trace_output)
    central_report_path = _repo_path(DEFAULT_STAGE_2_2_CENTRAL)
    learner_report_path = _repo_path(DEFAULT_STAGE_2_2_LEARNER)
    checkpoint_before = {
        "v4.1": _checkpoint_contract(v4_checkpoint),
        "v3.6": _checkpoint_contract(v3_checkpoint),
    }

    snapshot = WorkerAssetsSnapshot.build(CardRepository(database))
    v4_trainer = load_checkpoint(
        v4_checkpoint,
        snapshot,
        device=str(device),
        restore_rng_state=False,
    )
    v3_trainer = load_checkpoint(
        v3_checkpoint,
        snapshot,
        device=str(device),
        restore_rng_state=False,
    )
    try:
        v4_model = v4_trainer.model
        v3_model = v3_trainer.model
        maximum_batch = max(args.batch_sizes)
        v4_fixture = _fixed_fixture(
            v4_model,
            maximum_batch_size=maximum_batch,
            seed=args.fixture_seed,
        )
        v3_fixture = _fixed_fixture(
            v3_model,
            maximum_batch_size=maximum_batch,
            seed=args.fixture_seed,
        )

        forward = {
            "v4.1": benchmark_forward_batches(
                v4_model,
                v4_fixture,
                batch_sizes=args.batch_sizes,
                device=device,
                warmup_iterations=args.warmup_iterations,
                measured_iterations=args.measured_iterations,
                repeats=args.repeats,
            ),
            "v3.6": benchmark_forward_batches(
                v3_model,
                v3_fixture,
                batch_sizes=args.batch_sizes,
                device=device,
                warmup_iterations=args.warmup_iterations,
                measured_iterations=args.measured_iterations,
                repeats=args.repeats,
            ),
        }
        component_profile = profile_forward_components(
            v4_model,
            v4_fixture,
            batch_sizes=args.batch_sizes,
            repeats=args.repeats,
            device=device,
        )
        packing = benchmark_input_packing(
            v4_fixture,
            batch_sizes=args.batch_sizes,
            device=device,
            iterations=args.packing_iterations,
            repeats=args.repeats,
        )
        recurrent = benchmark_recurrent_lengths(
            v4_model,
            episode_lengths=EPISODE_LENGTHS,
            batch_size=4,
            total_steps=args.recurrent_total_steps,
            repeats=args.repeats,
            device=device,
            seed=args.fixture_seed,
        )
        legal_actions = benchmark_legal_action_counts(
            v4_model,
            v4_fixture,
            legal_counts=LEGAL_ACTION_COUNTS,
            batch_size=args.profiler_batch_size,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
            repeats=args.repeats,
            device=device,
        )
        profiler = run_torch_profiler(
            v4_model,
            v4_fixture,
            batch_size=args.profiler_batch_size,
            iterations=args.profiler_iterations,
            device=device,
            trace_path=trace_output,
        )
    finally:
        v4_trainer.close()
        v3_trainer.close()

    checkpoint_after = {
        "v4.1": _checkpoint_contract(v4_checkpoint),
        "v3.6": _checkpoint_contract(v3_checkpoint),
    }
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("a frozen checkpoint changed during profiling")

    live_central = _load_json(central_report_path)
    live_learner = _load_json(learner_report_path)
    report: dict[str, object] = {
        "schema_version": 1,
        "checklist_section": "2.3",
        "purpose": (
            "identify actionable v4.1 bottlenecks; v3.6 is a lightweight "
            "fixed-input pure-forward reference and is not an optimization "
            "target"
        ),
        "methodology": {
            "classification": "A-PROFILE-002",
            "v4_1_scope": (
                "fixed-input forward scaling, component events, CPU/GPU "
                "packing, recurrent length, legal-count distribution, "
                "PyTorch Profiler, and duplicate-work audit"
            ),
            "v3_6_scope": "pure_forward_reference_only",
            "synthetic_fixture": (
                "shape/dtype-valid deterministic tensors generated once at "
                "maximum batch and sliced so each batch uses the same prefix; "
                "not a learning or gameplay-quality input"
            ),
            "timing": (
                "three repeat samples; CUDA events measure device work and "
                "host wall time is retained separately; component hooks do "
                "not replace the executed model graph"
            ),
            "interpretation_limit": (
                "microbenchmarks choose later optimization experiments but "
                "do not establish end-to-end training speedup"
            ),
        },
        "configuration": {
            "device": str(device),
            "batch_sizes": list(args.batch_sizes),
            "warmup_iterations": args.warmup_iterations,
            "measured_iterations": args.measured_iterations,
            "repeats": args.repeats,
            "packing_iterations": args.packing_iterations,
            "recurrent_total_steps": args.recurrent_total_steps,
            "episode_lengths": list(EPISODE_LENGTHS),
            "legal_action_counts": list(LEGAL_ACTION_COUNTS),
            "profiler_batch_size": args.profiler_batch_size,
            "profiler_iterations": args.profiler_iterations,
            "fixture_seed": args.fixture_seed,
        },
        "hardware": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "checkpoints": checkpoint_after,
        "source_reports": {
            "stage_2_2_central": {
                "path": _relative(central_report_path),
                "sha256": _sha256(central_report_path),
            },
            "stage_2_2_learner": {
                "path": _relative(learner_report_path),
                "sha256": _sha256(learner_report_path),
            },
        },
        "fixed_input_forward": forward,
        "v4_1_component_profile": component_profile,
        "v4_1_input_packing": packing,
        "v4_1_recurrent_episode_lengths": recurrent,
        "v4_1_legal_action_counts": legal_actions,
        "profiler": profiler,
        "duplicate_work_audit": duplicate_work_audit(),
        "bottleneck_ranking": build_bottleneck_ranking(
            component_profile=component_profile,
            forward_batches=forward["v4.1"],
            live_central_report=live_central,
            live_learner_report=live_learner,
        ),
    }
    validate_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"v4.1 inference breakdown written to {output}")
    print(f"compressed PyTorch Profiler trace written to {trace_output}")


if __name__ == "__main__":
    main()
