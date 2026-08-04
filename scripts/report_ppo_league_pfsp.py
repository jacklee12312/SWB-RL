from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from scripts.report_ppo_league_baseline import ROOT, render_json
from scripts.report_ppo_league_generation import _selection_audit
from swb.rl.pfsp import (
    DEFAULT_EPSILON_FLOOR,
    DEFAULT_FORGOTTEN_SCORE_THRESHOLD,
    DEFAULT_HARD_ALPHA,
    DEFAULT_MAX_PROBABILITY,
    DEFAULT_RETAINED_SCORE_THRESHOLD,
    PFSP_SAMPLERS,
    compute_pfsp_distribution,
    load_training_payoff_snapshot,
)
from swb.rl.versioning import stable_json_sha256


REPORT_DIRECTORY = Path("data/reports/league_training")
GENERATION_ZERO_MANIFEST = REPORT_DIRECTORY / "generation_000_manifest.json"
EVALUATION_PROTOCOL = REPORT_DIRECTORY / "evaluation_protocol.json"
DEFAULT_PAYOFF_SNAPSHOT = (
    REPORT_DIRECTORY / "generation_000_training_payoff_snapshot.json"
)
DEFAULT_SCAN_OUTPUT = REPORT_DIRECTORY / "pfsp_sampler_scan.json"
DEFAULT_MANIFEST_OUTPUT = REPORT_DIRECTORY / "generation_001_manifest.json"


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


def _file_source(path: str | Path) -> dict[str, object]:
    resolved = _repo_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    return {
        "path": _relative(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "payload_sha256": stable_json_sha256(payload),
    }


def _load_contract(
    payoff_snapshot_path: str | Path,
    generation_manifest_path: str | Path,
    evaluation_protocol_path: str | Path,
):
    protocol_path = _repo_path(evaluation_protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    partitions = protocol["seed_partitions"]
    snapshot = load_training_payoff_snapshot(
        _repo_path(payoff_snapshot_path),
        generation_manifest_path=_repo_path(generation_manifest_path),
        allowed_tuning_seeds=partitions["pfsp_tuning_match_master_seeds"],
        forbidden_final_seeds=(
            partitions["final_evaluation_match_master_seeds"]
        ),
        repository_root=ROOT,
    )
    generation = json.loads(
        _repo_path(generation_manifest_path).read_text(encoding="utf-8")
    )
    return protocol, snapshot, generation


def build_pfsp_sampler_scan(
    *,
    payoff_snapshot_path: str | Path = DEFAULT_PAYOFF_SNAPSHOT,
    generation_manifest_path: str | Path = GENERATION_ZERO_MANIFEST,
    evaluation_protocol_path: str | Path = EVALUATION_PROTOCOL,
) -> dict[str, object]:
    protocol, snapshot, generation = _load_contract(
        payoff_snapshot_path,
        generation_manifest_path,
        evaluation_protocol_path,
    )
    distributions = {
        sampler: compute_pfsp_distribution(
            snapshot.estimates,
            sampler=sampler,
            epsilon_floor=DEFAULT_EPSILON_FLOOR,
            maximum_probability=DEFAULT_MAX_PROBABILITY,
            hard_alpha=DEFAULT_HARD_ALPHA,
        ).report()
        for sampler in ("uniform", "variance", "hard")
    }
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_pfsp_sampler_scan",
        "status": "implementation_ready_training_screen_pending",
        "sources": {
            "generation_manifest": _file_source(generation_manifest_path),
            "training_payoff_snapshot": _file_source(payoff_snapshot_path),
            "evaluation_protocol": _file_source(evaluation_protocol_path),
        },
        "payoff_contract": snapshot.summary(),
        "data_isolation": {
            "partition": "pfsp_tuning",
            "used_match_master_seeds": list(snapshot.match_master_seeds),
            "registered_tuning_match_master_seeds": protocol[
                "seed_partitions"
            ]["pfsp_tuning_match_master_seeds"],
            "final_evaluation_match_master_seeds_used": [],
            "payoff_updates": "generation_boundary_only",
            "source_generation": snapshot.source_generation,
            "target_generation": snapshot.target_generation,
        },
        "preregistered_parameters": {
            "sampler_order": ["uniform", "variance", "hard"],
            "epsilon_floor": DEFAULT_EPSILON_FLOOR,
            "maximum_single_opponent_probability": (
                DEFAULT_MAX_PROBABILITY
            ),
            "hard_alpha": DEFAULT_HARD_ALPHA,
            "forgotten_previous_score_at_least": (
                DEFAULT_RETAINED_SCORE_THRESHOLD
            ),
            "forgotten_current_score_below": (
                DEFAULT_FORGOTTEN_SCORE_THRESHOLD
            ),
            "missing_confidence_interval_policy": (
                "unreliable_for_payoff_aware_raw_weight; epsilon floor only "
                "unless every raw weight is zero"
            ),
            "probability_cap_exception": (
                "allowed only when candidate_count * cap < 1"
            ),
        },
        "population": {
            "source_generation": int(generation["generation"]),
            "trainable_opponent_ids": [
                str(entry["opponent_id"])
                for entry in generation["entries"]
                if bool(entry["training_eligible"])
            ],
            "anchor_only_opponent_ids": [
                str(entry["opponent_id"])
                for entry in generation["entries"]
                if not bool(entry["training_eligible"])
            ],
        },
        "samplers": distributions,
        "forgotten_priority_queue": {
            "status": "no_previous_generation_payoff_snapshot",
            "entries": [],
        },
        "screening_plan": {
            "training_seed_count": 3,
            "agent_steps_per_configuration_per_seed": 100_000,
            "samplers_in_order": ["uniform", "variance", "hard"],
            "advance_to_500k_limit": 2,
            "required_500k_pair": [
                "uniform",
                "one_stable_payoff_aware_sampler",
            ],
            "final_evaluation_seeds_forbidden": True,
        },
    }


def build_generation_manifest(
    *,
    sampler: str,
    payoff_snapshot_path: str | Path = DEFAULT_PAYOFF_SNAPSHOT,
    generation_manifest_path: str | Path = GENERATION_ZERO_MANIFEST,
    evaluation_protocol_path: str | Path = EVALUATION_PROTOCOL,
) -> dict[str, object]:
    if sampler not in PFSP_SAMPLERS:
        raise ValueError(f"unsupported PFSP sampler {sampler!r}")
    _, snapshot, source_generation = _load_contract(
        payoff_snapshot_path,
        generation_manifest_path,
        evaluation_protocol_path,
    )
    distribution = compute_pfsp_distribution(
        snapshot.estimates,
        sampler=sampler,
        epsilon_floor=DEFAULT_EPSILON_FLOOR,
        maximum_probability=DEFAULT_MAX_PROBABILITY,
        hard_alpha=DEFAULT_HARD_ALPHA,
    )
    entries = []
    for source_entry in source_generation["entries"]:
        entry = dict(source_entry)
        entry["sampling_weight"] = (
            distribution.probabilities[str(entry["opponent_id"])]
            if bool(entry["training_eligible"])
            else 0.0
        )
        entries.append(entry)
    audit = _selection_audit(entries)
    trainable = [entry for entry in entries if entry["training_eligible"]]
    anchors = [entry for entry in entries if not entry["training_eligible"]]
    return {
        "schema_version": 1,
        "report_kind": "ppo_league_generation_manifest",
        "immutable": True,
        "path_base": "repository_root",
        "generation": snapshot.target_generation,
        "selection_mode": sampler,
        "contract": source_generation["contract"],
        "sources": {
            "previous_generation_manifest": _file_source(
                generation_manifest_path
            ),
            "training_payoff_snapshot": _file_source(payoff_snapshot_path),
            "evaluation_protocol": _file_source(evaluation_protocol_path),
        },
        "pfsp": distribution.report(),
        "forgotten_priority_queue": {
            "status": "no_previous_generation_payoff_snapshot",
            "entries": [],
        },
        "deduplication": source_generation["deduplication"],
        "entries": entries,
        "selection_audit": audit,
        "summary": {
            "entry_count": len(entries),
            "trainable_entry_count": len(trainable),
            "anchor_only_count": len(anchors),
            "source_model_generation_counts": {
                str(model_generation): sum(
                    int(entry["generation"]) == model_generation
                    for entry in entries
                )
                for model_generation in sorted({
                    int(entry["generation"]) for entry in entries
                })
            },
            "trainable_sampling_weight_total": sum(
                float(entry["sampling_weight"]) for entry in trainable
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify PFSP sampler scan and Generation 1 manifest."
    )
    parser.add_argument(
        "--payoff-snapshot",
        type=Path,
        default=DEFAULT_PAYOFF_SNAPSHOT,
    )
    parser.add_argument(
        "--generation-manifest",
        type=Path,
        default=GENERATION_ZERO_MANIFEST,
    )
    parser.add_argument(
        "--evaluation-protocol",
        type=Path,
        default=EVALUATION_PROTOCOL,
    )
    parser.add_argument(
        "--sampler",
        choices=("uniform", "variance", "hard"),
        required=True,
    )
    parser.add_argument("--scan-output", type=Path, default=DEFAULT_SCAN_OUTPUT)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_OUTPUT,
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scan = render_json(build_pfsp_sampler_scan(
        payoff_snapshot_path=args.payoff_snapshot,
        generation_manifest_path=args.generation_manifest,
        evaluation_protocol_path=args.evaluation_protocol,
    ))
    manifest = render_json(build_generation_manifest(
        sampler=args.sampler,
        payoff_snapshot_path=args.payoff_snapshot,
        generation_manifest_path=args.generation_manifest,
        evaluation_protocol_path=args.evaluation_protocol,
    ))
    scan_output = _repo_path(args.scan_output)
    manifest_output = _repo_path(args.manifest_output)
    if args.check:
        mismatches = []
        if not scan_output.is_file() or scan_output.read_bytes() != scan:
            mismatches.append(str(scan_output))
        if not manifest_output.is_file() or manifest_output.read_bytes() != manifest:
            mismatches.append(str(manifest_output))
        if mismatches:
            print("PFSP artifacts mismatch: " + ", ".join(mismatches))
            return 1
        print("PFSP artifacts are byte-stable and current")
        return 0
    scan_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    scan_output.write_bytes(scan)
    manifest_output.write_bytes(manifest)
    print(f"wrote PFSP sampler scan to {scan_output}")
    print(f"wrote Generation 1 manifest to {manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
