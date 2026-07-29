from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable


LEARNER_DECK = "official_qr_evolve_haven_20260727"
OPPONENT_DECKS = (
    "international_qr_forest_20260728",
    "international_qr_sword_20260728",
    "international_qr_runecraft_20260728",
    "international_qr_dragon_20260728",
    "international_qr_nightmare_20260728",
    "international_qr_portal_myuu_20260728",
    "international_qr_portal_lishenna_20260728",
)
FINAL_SEEDS = (20260730, 20260731, 20260801)
TUNING_SEED = 20260802
TUNING_EVALUATION_SEED = 20261001
FINAL_COMMON_EVALUATION_SEED = 20261002
FINAL_HEAD_TO_HEAD_SEEDS = (20261101, 20261102, 20261103)


@dataclass(frozen=True)
class Candidate:
    name: str
    learning_rate: float
    update_epochs: int
    minibatch_sequences: int
    rollout_steps: int
    entropy_coefficient: float = 0.01
    clip_ratio: float = 0.2


CANDIDATES = (
    Candidate("a_baseline", 3e-4, 2, 8, 2048),
    Candidate("b_lr2e4", 2e-4, 2, 8, 2048),
    Candidate("c_lr1e4", 1e-4, 2, 8, 2048),
    Candidate("d_lr2e4_e3_mb16", 2e-4, 3, 16, 2048),
    Candidate("e_lr1e4_e4_mb16", 1e-4, 4, 16, 2048),
    Candidate("f_lr2e4_e3_mb16_r4096", 2e-4, 3, 16, 4096),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded overnight v3.6/v4.1 learning ablation"
    )
    parser.add_argument(
        "--deadline",
        default="2026-07-29T08:20:00",
        help="local ISO timestamp; new optional evaluations stop after this time",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/reports/observation_nightly_20260729"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("data/checkpoints/observation_nightly_20260729"),
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=Path(
            "data/checkpoints/observation_ablation/"
            "v3_6_seed_20260729_100k.pt"
        ),
    )
    parser.add_argument("--pilot-steps", type=int, default=40_000)
    parser.add_argument("--finalist-steps", type=int, default=120_000)
    parser.add_argument("--final-steps", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wilson(successes: float, games: int) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    proportion = successes / games
    denominator = 1 + z * z / games
    center = (proportion + z * z / (2 * games)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / games
            + z * z / (4 * games * games)
        )
        / denominator
    )
    return (center - radius, center + radius)


class NightlyRun:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.output_root.resolve()
        self.checkpoints = args.checkpoint_root.resolve()
        self.reference = args.reference_checkpoint.resolve()
        self.deadline = datetime.fromisoformat(args.deadline)
        self.status_path = self.root / "status.json"
        self.started = datetime.now()
        self.completed_commands: list[dict[str, object]] = []
        self.failures: list[dict[str, object]] = []
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoints.mkdir(parents=True, exist_ok=True)

    def status(self, phase: str, **values: object) -> None:
        _atomic_json(
            self.status_path,
            {
                "schema_version": 1,
                "started_at": self.started.isoformat(timespec="seconds"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "deadline": self.deadline.isoformat(timespec="seconds"),
                "phase": phase,
                "completed_commands": len(self.completed_commands),
                "failures": list(self.failures),
                **values,
            },
        )

    def run_command(
        self,
        name: str,
        arguments: Iterable[object],
        *,
        optional: bool = False,
    ) -> bool:
        if optional and datetime.now() >= self.deadline:
            record = {
                "name": name,
                "skipped": True,
                "reason": "nightly deadline reached",
            }
            self.completed_commands.append(record)
            self.status("optional_command_skipped", skipped_command=record)
            return False
        command = [str(self.args.python), *map(str, arguments)]
        log_path = self.root / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self.status(
            "running_command",
            command_name=name,
            command=command,
            log=str(log_path),
        )
        with log_path.open("w", encoding="utf-8") as log:
            log.write(
                f"started={datetime.now().isoformat(timespec='seconds')}\n"
            )
            log.write(f"command={json.dumps(command, ensure_ascii=False)}\n")
            log.flush()
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                text=True,
            )
        record = {
            "name": name,
            "returncode": result.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "log": str(log_path),
        }
        self.completed_commands.append(record)
        if result.returncode:
            self.failures.append(record)
            self.status("command_failed", failed_command=record)
            if not optional:
                raise RuntimeError(
                    f"{name} failed with exit code {result.returncode}; "
                    f"see {log_path}"
                )
            return False
        return True

    def initial_train_arguments(
        self,
        *,
        observation: str,
        seed: int,
        steps: int,
        checkpoint: Path,
        report: Path,
        candidate: Candidate,
    ) -> list[object]:
        return [
            "-m",
            "scripts.train_ppo",
            "--total-agent-steps",
            steps,
            "--rollout-steps",
            candidate.rollout_steps,
            "--rollout-workers",
            4,
            "--rollout-worker-threads",
            2,
            "--central-inference-batch-wait-ms",
            0.5,
            "--max-episode-steps",
            256,
            "--sequence-length",
            32,
            "--minibatch-sequences",
            candidate.minibatch_sequences,
            "--update-epochs",
            candidate.update_epochs,
            "--policy-architecture",
            "entity_action_v1",
            "--observation-version",
            observation,
            "--hidden-size",
            512,
            "--card-embedding-dim",
            128,
            "--model-dim",
            256,
            "--transformer-layers",
            4,
            "--attention-heads",
            8,
            "--feedforward-dim",
            1024,
            "--learning-rate",
            candidate.learning_rate,
            "--entropy-coefficient",
            candidate.entropy_coefficient,
            "--clip-ratio",
            candidate.clip_ratio,
            "--master-seed",
            seed,
            "--training-deck",
            LEARNER_DECK,
            "--opponent-decks",
            *OPPONENT_DECKS,
            "--device",
            self.args.device,
            "--match-setup",
            "official",
            "--opponent-current-weight",
            1,
            "--opponent-random-weight",
            0,
            "--opponent-fixed-weight",
            0,
            "--opponent-historical-weight",
            0,
            "--checkpoint",
            checkpoint,
            "--metrics-output",
            report,
        ]

    def resume_train_arguments(
        self,
        *,
        source: Path,
        steps: int,
        checkpoint: Path,
        report: Path,
    ) -> list[object]:
        return [
            "-m",
            "scripts.train_ppo",
            "--total-agent-steps",
            steps,
            "--resume",
            source,
            "--device",
            self.args.device,
            "--checkpoint",
            checkpoint,
            "--metrics-output",
            report,
        ]

    def matchup_arguments(
        self,
        *,
        checkpoint: Path,
        seed_count: int,
        master_seed: int,
        output: Path,
    ) -> list[object]:
        return [
            "-m",
            "scripts.evaluate_deck_matchups",
            checkpoint,
            "--opponent-checkpoint",
            self.reference,
            "--learner-deck",
            LEARNER_DECK,
            "--opponent-decks",
            *OPPONENT_DECKS,
            "--seed-count",
            seed_count,
            "--max-agent-steps",
            512,
            "--master-seed",
            master_seed,
            "--device",
            self.args.device,
            "--match-setup",
            "official",
            "--output",
            output,
        ]

    def head_to_head_arguments(
        self,
        *,
        learner: Path,
        opponent: Path,
        seed_count: int,
        master_seed: int,
        output: Path,
    ) -> list[object]:
        return [
            "-m",
            "scripts.evaluate_ppo",
            learner,
            "--seed-count",
            seed_count,
            "--max-agent-steps",
            512,
            "--master-seed",
            master_seed,
            "--device",
            self.args.device,
            "--match-setup",
            "official",
            "--training-deck",
            LEARNER_DECK,
            "--opponent",
            "historical",
            "--opponent-checkpoint",
            opponent,
            "--output",
            output,
        ]

    @staticmethod
    def evaluation_score(path: Path) -> tuple[float, float]:
        report = _read_json(path)
        metrics = report["metrics"]
        return (
            float(metrics["win_rate"]),
            -float(metrics["truncated_rate"]),
        )

    def tuning(self) -> Candidate:
        tuning_root = self.root / "tuning"
        tuning_checkpoints = self.checkpoints / "tuning"
        pilot_results: list[dict[str, object]] = []
        for candidate in CANDIDATES:
            checkpoint = tuning_checkpoints / f"{candidate.name}_40k.pt"
            report = tuning_root / f"{candidate.name}_40k_training.json"
            evaluation = tuning_root / f"{candidate.name}_40k_eval.json"
            self.run_command(
                f"tune_{candidate.name}_40k_train",
                self.initial_train_arguments(
                    observation="v4.1",
                    seed=TUNING_SEED,
                    steps=self.args.pilot_steps,
                    checkpoint=checkpoint,
                    report=report,
                    candidate=candidate,
                ),
            )
            self.run_command(
                f"tune_{candidate.name}_40k_eval",
                self.matchup_arguments(
                    checkpoint=checkpoint,
                    seed_count=5,
                    master_seed=TUNING_EVALUATION_SEED,
                    output=evaluation,
                ),
            )
            score = self.evaluation_score(evaluation)
            pilot_results.append({
                "candidate": asdict(candidate),
                "checkpoint": str(checkpoint),
                "training_report": str(report),
                "evaluation_report": str(evaluation),
                "score": list(score),
            })
            _atomic_json(tuning_root / "pilot_ranking.json", pilot_results)
        pilot_results.sort(
            key=lambda row: tuple(row["score"]),
            reverse=True,
        )
        finalists = [
            next(
                candidate
                for candidate in CANDIDATES
                if candidate.name == row["candidate"]["name"]
            )
            for row in pilot_results[:2]
        ]
        finalist_results: list[dict[str, object]] = []
        for candidate in finalists:
            source = tuning_checkpoints / f"{candidate.name}_40k.pt"
            checkpoint = tuning_checkpoints / f"{candidate.name}_120k.pt"
            report = tuning_root / f"{candidate.name}_120k_training.json"
            evaluation = tuning_root / f"{candidate.name}_120k_eval.json"
            self.run_command(
                f"tune_{candidate.name}_120k_train",
                self.resume_train_arguments(
                    source=source,
                    steps=self.args.finalist_steps,
                    checkpoint=checkpoint,
                    report=report,
                ),
            )
            self.run_command(
                f"tune_{candidate.name}_120k_eval",
                self.matchup_arguments(
                    checkpoint=checkpoint,
                    seed_count=10,
                    master_seed=TUNING_EVALUATION_SEED + 1,
                    output=evaluation,
                ),
            )
            finalist_results.append({
                "candidate": asdict(candidate),
                "checkpoint": str(checkpoint),
                "training_report": str(report),
                "evaluation_report": str(evaluation),
                "score": list(self.evaluation_score(evaluation)),
            })
        finalist_results.sort(
            key=lambda row: tuple(row["score"]),
            reverse=True,
        )
        winner_name = finalist_results[0]["candidate"]["name"]
        winner = next(
            candidate for candidate in CANDIDATES
            if candidate.name == winner_name
        )
        _atomic_json(
            tuning_root / "selection.json",
            {
                "pilot_results": pilot_results,
                "finalist_results": finalist_results,
                "selected": asdict(winner),
            },
        )
        return winner

    def train_final_pair(
        self,
        seed: int,
        selected_v41: Candidate,
    ) -> dict[str, dict[int, str]]:
        standard_v3 = CANDIDATES[0]
        outputs: dict[str, dict[int, str]] = {}
        for observation, candidate in (
            ("v4.1", selected_v41),
            ("v3", standard_v3),
        ):
            label = "v4_1" if observation == "v4.1" else "v3_6"
            stage_sources: dict[int, str] = {}
            previous: Path | None = None
            for steps in (100_000, 300_000, self.args.final_steps):
                checkpoint = (
                    self.checkpoints / "final"
                    / f"{label}_seed_{seed}_{steps // 1000}k.pt"
                )
                report = (
                    self.root / "training"
                    / f"{label}_seed_{seed}_{steps // 1000}k.json"
                )
                if previous is None:
                    arguments = self.initial_train_arguments(
                        observation=observation,
                        seed=seed,
                        steps=steps,
                        checkpoint=checkpoint,
                        report=report,
                        candidate=candidate,
                    )
                else:
                    arguments = self.resume_train_arguments(
                        source=previous,
                        steps=steps,
                        checkpoint=checkpoint,
                        report=report,
                    )
                self.run_command(
                    f"final_{label}_seed_{seed}_{steps // 1000}k",
                    arguments,
                )
                stage_sources[steps] = str(checkpoint)
                previous = checkpoint
            outputs[label] = stage_sources
        return outputs

    def final_evaluations(
        self,
        final_models: dict[str, dict[str, dict[int, str]]],
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        evaluation_root = self.root / "evaluation"
        for seed_text, versions in final_models.items():
            seed = int(seed_text)
            for label, stages in versions.items():
                checkpoint = Path(stages[str(self.args.final_steps)])
                output = (
                    evaluation_root
                    / f"common_{label}_seed_{seed}.json"
                )
                ok = self.run_command(
                    f"eval_common_{label}_seed_{seed}",
                    self.matchup_arguments(
                        checkpoint=checkpoint,
                        seed_count=10,
                        master_seed=FINAL_COMMON_EVALUATION_SEED,
                        output=output,
                    ),
                    optional=True,
                )
                if ok:
                    results.append({
                        "kind": "common_reference",
                        "seed": seed,
                        "version": label,
                        "report": str(output),
                    })
            v41 = Path(
                versions["v4_1"][str(self.args.final_steps)]
            )
            v3 = Path(
                versions["v3_6"][str(self.args.final_steps)]
            )
            output = evaluation_root / f"head_to_head_seed_{seed}.json"
            ok = self.run_command(
                f"eval_head_to_head_seed_{seed}",
                self.head_to_head_arguments(
                    learner=v41,
                    opponent=v3,
                    seed_count=100,
                    master_seed=FINAL_HEAD_TO_HEAD_SEEDS[
                        FINAL_SEEDS.index(seed)
                    ],
                    output=output,
                ),
                optional=True,
            )
            if ok:
                results.append({
                    "kind": "head_to_head",
                    "seed": seed,
                    "learner_version": "v4_1",
                    "opponent_version": "v3_6",
                    "report": str(output),
                })
        return results

    def summarize(
        self,
        selected: Candidate,
        final_models: dict[str, dict[str, dict[int, str]]],
        evaluations: list[dict[str, object]],
    ) -> None:
        common: dict[str, list[float]] = {"v3_6": [], "v4_1": []}
        head_to_head: list[float] = []
        integrity = {
            "illegal_actions": 0,
            "action_mask_mismatches": 0,
            "truncated": 0,
        }
        detailed: list[dict[str, object]] = []
        for item in evaluations:
            report = _read_json(Path(item["report"]))
            metrics = report["metrics"]
            record = {
                **item,
                "win_rate": metrics["win_rate"],
                "confidence_interval_95": metrics["confidence_interval_95"],
                "games": metrics["games"],
                "truncated": metrics["truncated"],
                "illegal_actions": metrics["illegal_actions"],
                "action_mask_mismatches": metrics[
                    "action_mask_mismatches"
                ],
            }
            detailed.append(record)
            integrity["illegal_actions"] += int(metrics["illegal_actions"])
            integrity["action_mask_mismatches"] += int(
                metrics["action_mask_mismatches"]
            )
            integrity["truncated"] += int(metrics["truncated"])
            if item["kind"] == "common_reference":
                common[item["version"]].append(float(metrics["win_rate"]))
            else:
                head_to_head.append(float(metrics["win_rate"]))

        aggregates: dict[str, object] = {}
        for label, values in common.items():
            if values:
                aggregates[label] = {
                    "seed_win_rates": values,
                    "mean_win_rate": fmean(values),
                    "population_stddev": pstdev(values),
                }
        if head_to_head:
            total_games = sum(
                int(record["games"])
                for record in detailed
                if record["kind"] == "head_to_head"
            )
            successes = sum(
                float(record["win_rate"]) * int(record["games"])
                for record in detailed
                if record["kind"] == "head_to_head"
            )
            aggregates["v4_1_head_to_head"] = {
                "seed_win_rates": head_to_head,
                "mean_win_rate": fmean(head_to_head),
                "population_stddev": pstdev(head_to_head),
                "pooled_win_rate": successes / total_games,
                "pooled_wilson_95": list(_wilson(successes, total_games)),
            }
        summary = {
            "schema_version": 1,
            "started_at": self.started.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "deadline": self.deadline.isoformat(timespec="seconds"),
            "selected_v4_1_hyperparameters": asdict(selected),
            "final_models": final_models,
            "evaluations": detailed,
            "aggregates": aggregates,
            "integrity": integrity,
            "completed_commands": self.completed_commands,
            "failures": self.failures,
        }
        _atomic_json(self.root / "summary.json", summary)
        lines = [
            "# Observation v3.6 / v4.1 夜间实验",
            "",
            f"- 开始：{summary['started_at']}",
            f"- 完成：{summary['finished_at']}",
            f"- v4.1 参数：`{json.dumps(asdict(selected), ensure_ascii=False)}`",
            f"- 非法动作：{integrity['illegal_actions']}",
            f"- 动作掩码不一致：{integrity['action_mask_mismatches']}",
            "",
            "## 汇总",
            "",
            "```json",
            json.dumps(aggregates, ensure_ascii=False, indent=2),
            "```",
            "",
            "完整逐局结果和命令日志位于本目录。",
        ]
        (self.root / "summary.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def execute(self) -> None:
        if not self.reference.is_file():
            raise FileNotFoundError(self.reference)
        self.status("starting")
        selected = self.tuning()
        self.status(
            "tuning_complete",
            selected_v4_1_hyperparameters=asdict(selected),
        )
        final_models: dict[str, dict[str, dict[int, str]]] = {}
        for seed in FINAL_SEEDS:
            final_models[str(seed)] = self.train_final_pair(seed, selected)
            _atomic_json(self.root / "final_models.json", final_models)
        self.status("training_complete", final_models=final_models)
        normalized_models = {
            seed: {
                version: {
                    str(steps): path for steps, path in stages.items()
                }
                for version, stages in versions.items()
            }
            for seed, versions in final_models.items()
        }
        evaluations = self.final_evaluations(normalized_models)
        _atomic_json(self.root / "evaluations.json", evaluations)
        self.summarize(selected, normalized_models, evaluations)
        self.status(
            "complete",
            summary=str(self.root / "summary.json"),
        )


def _prevent_sleep(enable: bool) -> None:
    if os.name != "nt":
        return
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    flags = es_continuous | (es_system_required if enable else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def main() -> None:
    args = parse_args()
    run = NightlyRun(args)
    _prevent_sleep(True)
    try:
        run.execute()
    except BaseException as exc:
        run.status(
            "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        _prevent_sleep(False)


if __name__ == "__main__":
    main()
