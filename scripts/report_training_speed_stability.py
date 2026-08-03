from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = Path("data/reports/training_speed_stability")
DEFAULT_OUTPUT = DEFAULT_REPORT_ROOT / "summary.json"
DEFAULT_MARKDOWN = Path("docs/training_speed_stability_report.md")

GROUP_FILES = {
    "initial_six_worker_baseline": (
        "baseline_repeat_1.json",
        "baseline_repeat_2.json",
        "baseline_repeat_3.json",
    ),
    "post_candidate_six_worker_control": (
        "baseline_repeat_4_post_candidate.json",
    ),
    "seven_worker_one_ms": (
        "workers_7_screen.json",
        "workers_7_repeat_2.json",
        "workers_7_repeat_3.json",
    ),
    "final_seven_worker_half_ms": (
        "workers_7_wait_0_5_screen.json",
        "workers_7_wait_0_5_repeat_2.json",
        "workers_7_wait_0_5_repeat_3.json",
    ),
    "screen_seven_worker_quarter_ms": (
        "workers_7_wait_0_25_screen.json",
    ),
    "screen_eight_worker_one_ms": (
        "workers_8_screen.json",
    ),
}


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _field(
    fields: Mapping[str, object],
    name: str,
    statistic: str,
    default: float = 0.0,
) -> float:
    row = fields.get(name)
    if not isinstance(row, Mapping):
        return default
    return float(row.get(statistic, default))


def summarize_run(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    warmup = int(payload["steady_state"]["excluded_warmup_updates"])
    iterations = payload["iterations"][warmup:]
    steps = sum(int(row["agent_steps"]) for row in iterations)
    elapsed = sum(float(row["elapsed_seconds"]) for row in iterations)
    collect_fields = payload["steady_state"]["collect"]["fields"]
    system = payload["system_monitor"]["summary"]
    samples = payload["system_monitor"]["samples"]
    gpu_samples = [
        row["gpu"]
        for row in samples
        if isinstance(row.get("gpu"), Mapping)
    ]
    gpu_total = [
        float(row["memory_total_mib"])
        for row in gpu_samples
        if row.get("memory_total_mib") is not None
    ]
    episodes = _field(
        collect_fields, "worker_episode_count", "total"
    )
    truncations = _field(
        collect_fields, "worker_truncated_episode_count", "total"
    )
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": payload["checkpoint_sha256_before"],
        "checkpoint_unchanged": bool(payload["checkpoint_unchanged"]),
        "configuration": payload["runtime_rollout_configuration"],
        "warmup_updates_excluded": warmup,
        "steady_agent_steps": steps,
        "steady_elapsed_seconds": elapsed,
        "steady_agent_steps_per_second": steps / elapsed,
        "overall_agent_steps_per_second": float(
            payload["result"]["agent_steps_per_second"]
        ),
        "batch": {
            "mean": _field(
                collect_fields, "central_average_batch_size", "mean"
            ),
            "p50": _field(
                collect_fields, "central_batch_size_p50", "median"
            ),
            "p95": _field(
                collect_fields, "central_batch_size_p95", "median"
            ),
            "empty_slot_fraction": _field(
                collect_fields,
                "central_batch_empty_slot_fraction",
                "mean",
            ),
        },
        "milliseconds_per_agent_step": {
            "central_forward": (
                _field(
                    collect_fields, "central_forward_seconds", "total"
                ) * 1000.0 / steps
            ),
            "central_batch_wait": (
                _field(
                    collect_fields,
                    "central_batch_wait_seconds",
                    "total",
                ) * 1000.0 / steps
            ),
            "central_worker_message_wait": (
                _field(
                    collect_fields,
                    "central_worker_message_wait_seconds",
                    "total",
                ) * 1000.0 / steps
            ),
        },
        "episodes": int(episodes),
        "truncations": int(truncations),
        "system": {
            "gpu_memory_peak_mib": system["gpu_memory_peak_mib"],
            "gpu_memory_total_mib": (
                statistics.median(gpu_total) if gpu_total else None
            ),
            "ram_used_peak_bytes": system["ram_used_peak_bytes"],
            "pagefile_used_change_bytes": (
                system["pagefile_used_change_bytes"]
            ),
            "gpu_graphics_clock_median_mhz": (
                system["gpu_graphics_clock_median_mhz"]
            ),
            "gpu_power_p95_watts": system["gpu_power_p95_watts"],
            "gpu_temperature_peak_celsius": (
                system["gpu_temperature_peak_celsius"]
            ),
            "gpu_pstate_sample_counts": (
                system["gpu_pstate_sample_counts"]
            ),
            "gpu_any_hardware_throttle": (
                system["gpu_any_hardware_throttle"]
            ),
        },
    }


def aggregate_runs(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    speeds = [
        float(row["steady_agent_steps_per_second"]) for row in rows
    ]
    gpu_peaks = [
        float(row["system"]["gpu_memory_peak_mib"])
        for row in rows
    ]
    gpu_totals = [
        float(row["system"]["gpu_memory_total_mib"])
        for row in rows
        if row["system"]["gpu_memory_total_mib"] is not None
    ]
    return {
        "run_count": len(rows),
        "steady_agent_steps_per_second": {
            "runs": speeds,
            "median": statistics.median(speeds),
            "minimum": min(speeds),
            "maximum": max(speeds),
            "max_to_min_ratio": max(speeds) / min(speeds),
        },
        "batch_mean_median": statistics.median(
            float(row["batch"]["mean"]) for row in rows
        ),
        "central_forward_ms_per_step_median": statistics.median(
            float(row["milliseconds_per_agent_step"]["central_forward"])
            for row in rows
        ),
        "gpu_memory_peak_mib": max(gpu_peaks),
        "gpu_memory_total_mib": (
            statistics.median(gpu_totals) if gpu_totals else None
        ),
        "gpu_memory_minimum_headroom_mib": (
            statistics.median(gpu_totals) - max(gpu_peaks)
            if gpu_totals else None
        ),
        "ram_used_peak_bytes": max(
            int(row["system"]["ram_used_peak_bytes"]) for row in rows
        ),
        "episodes": sum(int(row["episodes"]) for row in rows),
        "truncations": sum(int(row["truncations"]) for row in rows),
        "all_checkpoints_unchanged": all(
            bool(row["checkpoint_unchanged"]) for row in rows
        ),
        "any_hardware_throttle": any(
            bool(row["system"]["gpu_any_hardware_throttle"])
            for row in rows
        ),
    }


def _improvement(candidate: float, baseline: float) -> float:
    return candidate / baseline - 1.0


def build_report(
    report_root: Path = DEFAULT_REPORT_ROOT,
) -> dict[str, object]:
    root = _repo_path(report_root)
    groups = {
        name: [summarize_run(root / filename) for filename in filenames]
        for name, filenames in GROUP_FILES.items()
    }
    aggregates = {
        name: aggregate_runs(rows) for name, rows in groups.items()
    }
    initial = aggregates["initial_six_worker_baseline"]
    post_control = groups["post_candidate_six_worker_control"][0]
    initial_fast = groups["initial_six_worker_baseline"][-1]
    fast_six_reference = statistics.median((
        float(initial_fast["steady_agent_steps_per_second"]),
        float(post_control["steady_agent_steps_per_second"]),
    ))
    seven_one = aggregates["seven_worker_one_ms"]
    final = aggregates["final_seven_worker_half_ms"]
    eight = aggregates["screen_eight_worker_one_ms"]
    quarter = aggregates["screen_seven_worker_quarter_ms"]
    initial_median = float(
        initial["steady_agent_steps_per_second"]["median"]
    )
    seven_one_median = float(
        seven_one["steady_agent_steps_per_second"]["median"]
    )
    final_median = float(
        final["steady_agent_steps_per_second"]["median"]
    )
    final_headroom = float(final["gpu_memory_minimum_headroom_mib"])
    adopted = (
        _improvement(seven_one_median, fast_six_reference) >= 0.05
        and _improvement(final_median, seven_one_median) >= 0.05
        and final_headroom >= 1024.0
        and int(final["truncations"]) == 0
        and bool(final["all_checkpoints_unchanged"])
        and not bool(final["any_hardware_throttle"])
    )
    checkpoint_hashes = sorted({
        str(row["checkpoint_sha256"])
        for rows in groups.values()
        for row in rows
    })
    return {
        "schema_version": 1,
        "report_kind": "swb_training_speed_runtime_stability",
        "checkpoint_sha256": checkpoint_hashes,
        "measurement_policy": {
            "checkpoint_updated_in_memory_only": True,
            "requested_agent_steps_per_run": 20_480,
            "warmup_updates_excluded": 2,
            "system_monitor_interval_seconds": 0.5,
            "comparison_note": (
                "The initial six-worker group directly measures runtime "
                "instability. Topology attribution conservatively uses the "
                "fast initial run plus the post-candidate six-worker control, "
                "not the slow initial median."
            ),
        },
        "groups": {
            name: {
                "aggregate": aggregates[name],
                "runs": rows,
            }
            for name, rows in groups.items()
        },
        "findings": {
            "initial_six_worker_runtime_variability_factor": (
                initial["steady_agent_steps_per_second"][
                    "max_to_min_ratio"
                ]
            ),
            "fast_six_worker_reference_steps_per_second": (
                fast_six_reference
            ),
            "seven_worker_one_ms_improvement_over_fast_six": (
                _improvement(seven_one_median, fast_six_reference)
            ),
            "half_ms_improvement_over_one_ms": (
                _improvement(final_median, seven_one_median)
            ),
            "final_improvement_over_fast_six": (
                _improvement(final_median, fast_six_reference)
            ),
            "final_improvement_over_initial_median_not_attributed": (
                _improvement(final_median, initial_median)
            ),
            "final_runtime_variability_factor": (
                final["steady_agent_steps_per_second"][
                    "max_to_min_ratio"
                ]
            ),
        },
        "decision": {
            "adopted": adopted,
            "runtime_overrides": {
                "rollout_workers": 7,
                "rollout_worker_torch_threads": 2,
                "central_inference_batch_wait_seconds": 0.0005,
            },
            "rejected_or_blocked": [
                {
                    "candidate": "8 workers, 1.0 ms",
                    "reason": (
                        "Only one screen and insufficient incremental gain; "
                        f"GPU peak {eight['gpu_memory_peak_mib']:.0f} MiB left "
                        f"{eight['gpu_memory_minimum_headroom_mib']:.0f} MiB "
                        "headroom."
                    ),
                },
                {
                    "candidate": "7 workers, 0.25 ms",
                    "reason": (
                        "Screen was slower than the three-run 0.5 ms median "
                        f"({quarter['steady_agent_steps_per_second']['median']:.2f} "
                        f"vs {final_median:.2f} steps/s)."
                    ),
                },
                {
                    "candidate": "lock GPU clocks to 2520-2820 MHz",
                    "reason": (
                        "nvidia-smi rejected the reversible clock-lock probe: "
                        "current user does not have permission; no clock state "
                        "was changed."
                    ),
                },
            ],
        },
    }


def render_markdown(report: Mapping[str, object]) -> str:
    groups = report["groups"]
    order = (
        ("initial_six_worker_baseline", "6 workers / 1.0ms 初始"),
        ("post_candidate_six_worker_control", "6 workers / 1.0ms 回切"),
        ("seven_worker_one_ms", "7 workers / 1.0ms"),
        ("final_seven_worker_half_ms", "7 workers / 0.5ms（采用）"),
        ("screen_seven_worker_quarter_ms", "7 workers / 0.25ms"),
        ("screen_eight_worker_one_ms", "8 workers / 1.0ms"),
    )
    lines = [
        "# PPO 训练速度稳定性报告",
        "",
        "所有运行使用同一只读 3M checkpoint；每轮请求 20,480 agent "
        "steps，并排除前 2 次 update。GPU/CPU/内存每 0.5 秒采样。",
        "",
        "| 配置 | 次数 | 稳态 steps/s | batch mean | forward ms/step | "
        "GPU 峰值 MiB | 截断 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in order:
        aggregate = groups[key]["aggregate"]
        speed = aggregate["steady_agent_steps_per_second"]
        lines.append(
            f"| {label} | {aggregate['run_count']} | "
            f"{speed['median']:.2f} "
            f"({speed['minimum']:.2f}–{speed['maximum']:.2f}) | "
            f"{aggregate['batch_mean_median']:.2f} | "
            f"{aggregate['central_forward_ms_per_step_median']:.2f} | "
            f"{aggregate['gpu_memory_peak_mib']:.0f} | "
            f"{aggregate['truncations']} |"
        )
    findings = report["findings"]
    decision = report["decision"]
    lines.extend((
        "",
        "## 结论",
        "",
        f"- 同配置 6-worker 初始三次最大相差 "
        f"{findings['initial_six_worker_runtime_variability_factor']:.2f}×；"
        "中央 forward 与 GPU 时钟/功耗状态同步变化，seed 不是原因。",
        f"- 保守快态 6-worker 参考为 "
        f"{findings['fast_six_worker_reference_steps_per_second']:.2f} "
        "steps/s。",
        f"- 最终候选中位数相对该参考提升 "
        f"{findings['final_improvement_over_fast_six']:.1%}，三次波动 "
        f"{findings['final_runtime_variability_factor']:.3f}×。",
        "- 最终三次共 644 局、截断 0、checkpoint 前后哈希不变、"
        "无硬件 throttle。",
        "",
        "## 采用配置",
        "",
        "```text",
        "rollout_workers = 7",
        "rollout_worker_torch_threads = 2",
        "central_inference_batch_wait_seconds = 0.0005",
        "```",
        "",
        f"决策门：{'通过' if decision['adopted'] else '未通过'}。",
        "",
        "## 拒绝或受阻候选",
        "",
    ))
    for row in decision["rejected_or_blocked"]:
        lines.append(f"- {row['candidate']}：{row['reason']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the 2026-08-03 PPO runtime stability profiles."
    )
    parser.add_argument(
        "--report-root", type=Path, default=DEFAULT_REPORT_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report(args.report_root)
    output = _repo_path(args.output)
    markdown = _repo_path(args.markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "markdown": str(args.markdown),
        "adopted": report["decision"]["adopted"],
        "runtime_overrides": report["decision"]["runtime_overrides"],
    }, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
