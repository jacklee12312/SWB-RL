from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = Path("data/reports/league_training/generations")
SUPPORT_CHECKPOINTS = (
    Path("data/checkpoints/league_smoke/uniform_10k_clustered_wait1.pt"),
    Path("data/checkpoints/league_smoke/uniform_100k_clustered.pt"),
    Path("data/checkpoints/league_smoke/own_history_100k_same_runtime.pt"),
    Path("data/checkpoints/training_speed/frozen_v3_6_seed_20260801_500k.pt"),
    Path("data/checkpoints/training_speed/frozen_v4_1_seed_20260801_500k.pt"),
)
SUPPORT_CHECKPOINT_GLOBS = (
    "data/checkpoints/league_sampler_screen_20260804/**/final_100k.pt",
)
SUPPORT_STATE_PATHS = (
    Path(
        "data/reports/league_training/sampler_screen_20260804/"
        "archive_baseline"
    ),
    Path(
        "data/reports/league_training/sampler_screen_20260804/"
        "candidate_evaluations"
    ),
    Path(
        "data/reports/league_training/sampler_screen_20260804/"
        "generation_000_active_matrix"
    ),
    Path(
        "data/reports/league_training/sampler_screen_20260804/training"
    ),
    Path("data/reports/training_speed/v4_1_profiler_trace.json.gz"),
)
MAX_ARCHIVE_ASSET_BYTES = 1_600_000_000


def render_json(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _relative(root: Path, path: str | Path) -> Path:
    return _repo_path(root, path).resolve().relative_to(root.resolve())


def split_by_size(
    root: Path,
    paths: Iterable[Path],
    maximum_bytes: int = MAX_ARCHIVE_ASSET_BYTES,
) -> list[list[Path]]:
    groups: list[list[Path]] = []
    current: list[Path] = []
    current_size = 0
    for path in sorted(set(paths), key=lambda value: value.as_posix()):
        size = (root / path).stat().st_size
        if size > maximum_bytes:
            raise ValueError(f"single release file exceeds asset limit: {path}")
        if current and current_size + size > maximum_bytes:
            groups.append(current)
            current = []
            current_size = 0
        current.append(path)
        current_size += size
    if current:
        groups.append(current)
    return groups


def create_tar(root: Path, output: Path, paths: Iterable[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as package:
        for relative in sorted(set(paths), key=lambda value: value.as_posix()):
            source = (root / relative).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"release source is missing: {relative}")
            source.relative_to(root.resolve())
            package.add(source, arcname=relative.as_posix(), recursive=False)


def checkpoint_groups(
    root: Path,
    population: Mapping[str, object],
) -> tuple[list[Path], list[Path], list[Path]]:
    active_and_anchors: list[Path] = []
    archive: list[Path] = []
    support: list[Path] = []
    entries = population.get("entries")
    if not isinstance(entries, list):
        raise ValueError("population manifest has no entries")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("population entry must be an object")
        path = _relative(root, str(entry["checkpoint_path"]))
        expected = str(entry["checkpoint_sha256"])
        actual = sha256_file(root / path)
        if actual != expected:
            raise ValueError(
                f"checkpoint hash mismatch for {path}: expected {expected}, got {actual}"
            )
        if entry.get("role") == "historical_archive":
            archive.append(path)
        else:
            active_and_anchors.append(path)
    for support_path in SUPPORT_CHECKPOINTS:
        if not (root / support_path).is_file():
            raise FileNotFoundError(
                f"release support checkpoint is missing: {support_path}"
            )
    support.extend(SUPPORT_CHECKPOINTS)
    for pattern in SUPPORT_CHECKPOINT_GLOBS:
        matches = [
            path.relative_to(root)
            for path in root.glob(pattern)
            if path.is_file()
        ]
        if not matches:
            raise FileNotFoundError(
                f"release support checkpoint glob has no matches: {pattern}"
            )
        support.extend(matches)
    return (
        sorted(set(active_and_anchors)),
        sorted(set(archive)),
        sorted(set(support)),
    )


def _portable_training_report(root: Path, source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, str):
        payload["checkpoint"] = _relative(root, checkpoint).as_posix()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(render_json(payload))


def build_state_tree(
    root: Path,
    report_root: Path,
    generation: int,
    destination: Path,
) -> list[Path]:
    copied: list[Path] = []
    for current in range(1, generation + 1):
        source_generation = root / report_root / f"generation_{current:03d}"
        if not (source_generation / "population_manifest.json").is_file():
            raise FileNotFoundError(f"generation {current} is not published")
        for source in source_generation.rglob("*.json"):
            relative = source.relative_to(root)
            target = destination / relative
            if source.parent.name == "training" and source.name.startswith("seed_"):
                _portable_training_report(root, source, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            copied.append(relative)

    generation_zero = (
        root / "data/reports/league_training/generation_000_payoff_evaluations"
    )
    if generation_zero.is_dir():
        for source in generation_zero.glob("*.json"):
            relative = source.relative_to(root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(relative)

    for support in SUPPORT_STATE_PATHS:
        source_path = root / support
        sources = (
            [source_path]
            if source_path.is_file()
            else sorted(path for path in source_path.rglob("*") if path.is_file())
        )
        if not sources:
            raise FileNotFoundError(f"release support path is missing: {support}")
        for source in sources:
            relative = source.relative_to(root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(relative)

    marker = Path("data/checkpoints/RELEASE_INSTALLED.json")
    marker_path = destination / marker
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_bytes(
        render_json(
            {
                "schema_version": 1,
                "generation": generation,
                "report_root": report_root.as_posix(),
            }
        )
    )
    copied.append(marker)
    return sorted(set(copied))


def asset_metadata(path: Path) -> dict[str, object]:
    return {
        "name": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_release(
    *,
    root: Path,
    report_root: Path,
    generation: int,
    output_directory: Path,
) -> dict[str, object]:
    generation_directory = root / report_root / f"generation_{generation:03d}"
    population_path = generation_directory / "population_manifest.json"
    population = json.loads(population_path.read_text(encoding="utf-8"))
    active, archive, support = checkpoint_groups(root, population)
    tag = f"league-g{generation:03d}-resume"
    output_directory.mkdir(parents=True, exist_ok=True)
    assets: list[Path] = []

    active_asset = output_directory / f"{tag}-active.tar"
    create_tar(root, active_asset, active)
    assets.append(active_asset)

    for index, group in enumerate(split_by_size(root, support), start=1):
        support_asset = output_directory / f"{tag}-support-{index:02d}.tar"
        create_tar(root, support_asset, group)
        assets.append(support_asset)

    for index, group in enumerate(split_by_size(root, archive), start=1):
        archive_asset = output_directory / f"{tag}-archive-{index:02d}.tar"
        create_tar(root, archive_asset, group)
        assets.append(archive_asset)

    with tempfile.TemporaryDirectory(prefix="swb-release-state-") as temporary:
        staging = Path(temporary)
        state_paths = build_state_tree(root, report_root, generation, staging)
        state_asset = output_directory / f"{tag}-state.tar"
        create_tar(staging, state_asset, state_paths)
        assets.append(state_asset)

    manifest = {
        "schema_version": 1,
        "tag": tag,
        "generation": generation,
        "population_manifest_sha256": sha256_file(population_path),
        "assets": [asset_metadata(path) for path in assets],
    }
    (output_directory / f"{tag}.json").write_bytes(render_json(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build portable GitHub Release assets for a published League generation."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest = build_release(
        root=arguments.root.resolve(),
        report_root=arguments.report_root,
        generation=arguments.generation,
        output_directory=arguments.output_directory.resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
