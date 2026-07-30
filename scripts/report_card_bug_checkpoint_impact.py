from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from swb.engine.environment import ShadowverseEnv
from swb.rl.runtime import hash_rule_directory


FIX_COMMIT = "b6f1d95cd2336cc86772e717e5bd09440a8f38a7"
BUG_ID = "SWB-CARD-0008"
DEFAULT_CHECKPOINT_DIRECTORY = Path("data/checkpoints")
DEFAULT_OUTPUT = Path(
    "data/reports/card_bug_audit/repros/checkpoint_impact.json"
)


def _commit_timestamp(commit: str) -> datetime:
    result = subprocess.run(
        ("git", "show", "-s", "--format=%cI", commit),
        check=True,
        capture_output=True,
        text=True,
    )
    return datetime.fromisoformat(result.stdout.strip()).astimezone(
        timezone.utc
    )


def _relative(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _inspect_checkpoint(
    path: Path,
    *,
    fix_timestamp: datetime,
) -> dict[str, object]:
    stat = path.stat()
    modified = datetime.fromtimestamp(
        stat.st_mtime,
        tz=timezone.utc,
    )
    row: dict[str, object] = {
        "path": _relative(path),
        "size_bytes": stat.st_size,
        "modified_at_utc": modified.isoformat(),
        "readable": False,
    }
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        versions = payload.get("versions", {})
        trainer = payload.get("trainer", {})
        manifest = payload.get("experiment_manifest", {})
        git = manifest.get("git", {})
        row.update(
            {
                "readable": True,
                "checkpoint_schema_version": payload.get(
                    "checkpoint_schema_version"
                ),
                "agent_steps": trainer.get("agent_steps"),
                "update_count": trainer.get("update_count"),
                "rulebook_sha256": versions.get(
                    "rulebook_sha256",
                    manifest.get("rulebook_sha256"),
                ),
                "observation_version": versions.get(
                    "observation_version",
                    manifest.get("observation_version"),
                ),
                "action_layout_version": versions.get(
                    "action_layout_version"
                ),
                "training_git_commit": git.get("commit"),
                "training_git_dirty": git.get("dirty"),
            }
        )
        del payload
    except Exception as exc:
        row["load_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    is_pre_fix = modified < fix_timestamp
    row["classification"] = (
        "pre_fix_historical_only"
        if is_pre_fix
        else "post_fix_timestamp_requires_manifest_review"
    )
    row["potentially_affected"] = is_pre_fix
    row["fair_strength_comparison_with_post_fix_models"] = False
    row["preservation"] = "keep_read_only_historical_artifact"
    return row


def build_report(
    *,
    checkpoint_directory: Path,
) -> dict[str, object]:
    fix_timestamp = _commit_timestamp(FIX_COMMIT)
    checkpoints = sorted(checkpoint_directory.rglob("*.pt"))
    rows = [
        _inspect_checkpoint(path, fix_timestamp=fix_timestamp)
        for path in checkpoints
    ]
    classifications = Counter(
        str(row["classification"]) for row in rows
    )
    rule_hashes = Counter(
        str(row.get("rulebook_sha256"))
        for row in rows
        if row.get("rulebook_sha256")
    )
    readable = sum(bool(row["readable"]) for row in rows)
    potentially_affected = sum(
        bool(row["potentially_affected"]) for row in rows
    )
    return {
        "schema_version": 1,
        "report_kind": "swb_card_bug_checkpoint_impact",
        "bug_id": BUG_ID,
        "fix": {
            "commit": FIX_COMMIT,
            "committed_at_utc": fix_timestamp.isoformat(),
            "mechanic": (
                "SET_STATS assignment after prior stat modifiers, followed "
                "by evolution"
            ),
            "current_rulebook_sha256": hash_rule_directory(
                ShadowverseEnv.DEFAULT_RULE_DIRECTORY
            ),
            "compatibility_caveat": (
                "The fix changed Python engine semantics, not data/rules. "
                "The rulebook hash can therefore match while behavior differs."
            ),
        },
        "scope": {
            "checkpoint_directory": checkpoint_directory.as_posix(),
            "scan_mode": "read-only metadata inspection",
            "files_modified_or_deleted": 0,
        },
        "summary": {
            "checkpoint_count": len(rows),
            "readable_count": readable,
            "unreadable_count": len(rows) - readable,
            "potentially_affected_pre_fix_count": potentially_affected,
            "classification_counts": dict(
                sorted(classifications.items())
            ),
            "rulebook_hash_counts": dict(sorted(rule_hashes.items())),
        },
        "policy": {
            "preserve_old_checkpoints": True,
            "allowed_use": [
                "historical reconstruction",
                "same-checkpoint replay under its frozen historical engine",
            ],
            "forbidden_conclusion": (
                "Do not mix pre-fix and post-fix models in a fair strength "
                "comparison because legal trajectories and terminal outcomes "
                "can differ."
            ),
            "resume_policy": (
                "Do not resume a pre-fix checkpoint as a continuation of a "
                "post-fix experiment. Start a new experiment baseline from "
                "the frozen rules-engine commit."
            ),
        },
        "checkpoints": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory old PPO checkpoints affected by a card-rule engine fix."
        )
    )
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIRECTORY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        checkpoint_directory=args.checkpoint_directory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"output={args.output.as_posix()} "
        f"checkpoints={report['summary']['checkpoint_count']} "
        f"readable={report['summary']['readable_count']} "
        "potentially_affected="
        f"{report['summary']['potentially_affected_pre_fix_count']}"
    )


if __name__ == "__main__":
    main()
