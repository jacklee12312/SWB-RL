from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from scripts.profile_v4_1_inference_breakdown import (
    _checkpoint_contract,
    _device_fixture,
    _fixed_fixture,
)
from scripts.scan_training_speed_stage_2_4 import (
    DEFAULT_CHECKPOINT,
    _repo_path,
    _sha256,
)
from swb.db.repository import CardRepository
from swb.rl.checkpoint import load_checkpoint
from swb.rl.runtime import WorkerAssetsSnapshot


DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_OUTPUT = Path(
    "data/reports/training_speed/"
    "stage_2_6_b_compile_001_gate.json"
)


def _probe(
    *,
    backend: str,
    database: Path,
    checkpoint: Path,
) -> None:
    device = torch.device("cuda")
    checkpoint_path = _repo_path(checkpoint)
    checkpoint_before = _checkpoint_contract(checkpoint_path)
    snapshot = WorkerAssetsSnapshot.build(
        CardRepository(_repo_path(database))
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
            model, maximum_batch_size=4, seed=20260801
        )
        inputs = _device_fixture(
            fixture, batch_size=4, device=device
        )
        with torch.no_grad():
            native = model.forward_step(
                inputs["observation"],
                inputs["hidden"],
                inputs["card_indices"],
            )
        state_keys_before = list(model.state_dict())
        compiled = torch.compile(
            model.forward_step,
            backend=backend,
            dynamic=False,
        )
        torch.cuda.synchronize()
        first_started = time.perf_counter()
        with torch.no_grad():
            compiled_outputs = compiled(
                inputs["observation"],
                inputs["hidden"],
                inputs["card_indices"],
            )
        torch.cuda.synchronize()
        first_seconds = time.perf_counter() - first_started
        steady_started = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                compiled(
                    inputs["observation"],
                    inputs["hidden"],
                    inputs["card_indices"],
                )
        torch.cuda.synchronize()
        steady_seconds = (
            time.perf_counter() - steady_started
        ) / 10.0
        exact = [
            torch.equal(first, second)
            for first, second in zip(native, compiled_outputs)
        ]
        state_keys_after = list(model.state_dict())
    finally:
        trainer.close()
    checkpoint_after = _checkpoint_contract(checkpoint_path)
    print(json.dumps({
        "backend": backend,
        "first_call_seconds": first_seconds,
        "steady_call_seconds": steady_seconds,
        "exact_outputs": exact,
        "state_dict_keys_unchanged": (
            state_keys_before == state_keys_after
        ),
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
    }, sort_keys=True))


def _run_child(
    *,
    backend: str,
    utf8: bool,
    database: Path,
    checkpoint: Path,
) -> dict[str, object]:
    environment = dict(os.environ)
    if utf8:
        environment["PYTHONUTF8"] = "1"
    else:
        environment.pop("PYTHONUTF8", None)
        environment.pop("PYTHONIOENCODING", None)
    environment["TORCH_LOGS"] = "graph_breaks"
    command = [
        sys.executable,
        "-m",
        "scripts.assess_training_speed_stage_2_6_b_compile_001",
        "--probe-backend",
        backend,
        "--database",
        str(database),
        "--checkpoint",
        str(checkpoint),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=_repo_path(Path(".")),
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    elapsed = time.perf_counter() - started
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        break
    stderr = completed.stderr
    return {
        "backend": backend,
        "python_utf8": utf8,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "payload": payload,
        "signals": {
            "unicode_decode_error": "UnicodeDecodeError" in stderr,
            "triton_missing": "TritonMissing" in stderr,
            "graph_break": "Graph break" in stderr,
            "card_index_bool_site": (
                "card_indices < 0" in stderr
                or "policy.py\", line 1753" in stderr
            ),
        },
        "stderr_tail": stderr[-6000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assess checklist 2.6 B-COMPILE-001"
    )
    parser.add_argument(
        "--probe-backend", choices=("inductor", "eager")
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.probe_backend is not None:
        _probe(
            backend=args.probe_backend,
            database=args.database,
            checkpoint=args.checkpoint,
        )
        return

    checkpoint_path = _repo_path(args.checkpoint)
    checkpoint_before = _checkpoint_contract(checkpoint_path)
    probes = {
        "system_locale_inductor": _run_child(
            backend="inductor",
            utf8=False,
            database=args.database,
            checkpoint=args.checkpoint,
        ),
        "utf8_inductor": _run_child(
            backend="inductor",
            utf8=True,
            database=args.database,
            checkpoint=args.checkpoint,
        ),
        "utf8_eager_control": _run_child(
            backend="eager",
            utf8=True,
            database=args.database,
            checkpoint=args.checkpoint,
        ),
    }
    checkpoint_after = _checkpoint_contract(checkpoint_path)
    locale_failure = probes["system_locale_inductor"]["signals"][
        "unicode_decode_error"
    ]
    inductor_failure = probes["utf8_inductor"]["signals"][
        "triton_missing"
    ]
    graph_break = probes["utf8_inductor"]["signals"]["graph_break"]
    eager_payload = probes["utf8_eager_control"]["payload"]
    eager_control_passed = (
        probes["utf8_eager_control"]["returncode"] == 0
        and eager_payload is not None
        and all(eager_payload["exact_outputs"])
        and eager_payload["state_dict_keys_unchanged"]
        and eager_payload["checkpoint_unchanged"]
    )
    assessment_passed = (
        locale_failure
        and inductor_failure
        and graph_break
        and eager_control_passed
        and checkpoint_before == checkpoint_after
    )
    report = {
        "schema_version": 1,
        "report_kind": (
            "swb_training_speed_stage_2_6_b_compile_001_gate"
        ),
        "checklist_section": "2.6",
        "candidate": "B-COMPILE-001",
        "classification": "B",
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton_module_available": (
                importlib.util.find_spec("triton") is not None
            ),
        },
        "probes": probes,
        "checkpoint": checkpoint_after,
        "decision": {
            "implement": False,
            "run_end_to_end": False,
            "run_learning_seeds": False,
            "disposition": (
                "blocked_current_environment_missing_triton"
            ),
            "reason": (
                "the default locale fails while loading an inductor "
                "template; UTF-8 reaches compilation but the installed "
                "PyTorch environment has no working Triton, and the "
                "policy validation also introduces a graph break"
            ),
            "eager_backend_is_not_optimization_candidate": True,
            "reopen_when": (
                "a project-supported Windows Triton/inductor toolchain "
                "is declared and the compile graph-break contract is "
                "resolved without weakening illegal-input rejection"
            ),
        },
        "sources": {
            "checkpoint": {
                "path": str(args.checkpoint).replace("\\", "/"),
                "sha256": _sha256(args.checkpoint),
            },
        },
        "passed": assessment_passed,
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
        "disposition": report["decision"]["disposition"],
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
