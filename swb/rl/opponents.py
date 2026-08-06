from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from swb.rl.seeding import derive_seed
from swb.rl.versioning import stable_json_sha256


EXTERNAL_OPPONENT_MANIFEST_SCHEMA_VERSION = 1
EXTERNAL_OPPONENT_SELECTION_MODES = frozenset({
    "uniform",
    "variance",
    "hard",
})
OPPONENT_BATCHING_MODES = frozenset({"sequential", "episode_seed_clustered"})


@dataclass(frozen=True)
class OpponentEntry:
    opponent_id: str
    kind: str
    weight: float
    checkpoint_path: str | None = None
    created_agent_steps: int = 0
    checkpoint_sha256: str | None = None
    policy_seed: int | None = None
    training_steps: int = 0
    generation: int | None = None
    role: str | None = None
    rules_version: str | None = None
    policy_architecture: str | None = None
    versions_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "current",
            "historical",
            "external",
            "random_legal",
            "fixed",
        }:
            raise ValueError(f"unsupported opponent kind {self.kind!r}")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("opponent weight must be non-negative")
        if self.kind == "historical" and not self.checkpoint_path:
            raise ValueError("historical opponent requires checkpoint_path")
        if self.kind == "external":
            missing = [
                name
                for name, value in (
                    ("checkpoint_path", self.checkpoint_path),
                    ("checkpoint_sha256", self.checkpoint_sha256),
                    ("policy_seed", self.policy_seed),
                    ("generation", self.generation),
                    ("role", self.role),
                    ("rules_version", self.rules_version),
                    ("policy_architecture", self.policy_architecture),
                    ("versions_sha256", self.versions_sha256),
                )
                if value is None or value == ""
            ]
            if missing:
                raise ValueError(
                    "external opponent is missing " + ", ".join(missing)
                )
            if (
                len(str(self.checkpoint_sha256)) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(self.checkpoint_sha256)
                )
            ):
                raise ValueError("external opponent checkpoint_sha256 is invalid")
            if (
                len(str(self.versions_sha256)) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(self.versions_sha256)
                )
            ):
                raise ValueError("external opponent versions_sha256 is invalid")
            if self.training_steps <= 0:
                raise ValueError("external opponent training_steps must be positive")
            if self.generation is None or self.generation < 0:
                raise ValueError("external opponent generation must be non-negative")

    def assignment_metadata(self) -> dict[str, object]:
        return {
            "opponent_role": self.role,
            "opponent_policy_seed": self.policy_seed,
            "opponent_training_steps": self.training_steps,
            "opponent_generation": self.generation,
            "opponent_checkpoint_sha256": self.checkpoint_sha256,
            "opponent_rules_version": self.rules_version,
        }


@dataclass(frozen=True)
class ExternalOpponentManifest:
    path: str
    file_sha256: str
    payload_sha256: str
    generation: int
    selection_mode: str
    entries: tuple[OpponentEntry, ...]
    contract: Mapping[str, object]

    @property
    def trainable_entries(self) -> tuple[OpponentEntry, ...]:
        return tuple(entry for entry in self.entries if entry.weight > 0)

    @property
    def reference_entries(self) -> tuple[OpponentEntry, ...]:
        return tuple(entry for entry in self.entries if entry.weight == 0)

    def summary(self) -> dict[str, object]:
        return {
            "path": self.path,
            "file_sha256": self.file_sha256,
            "payload_sha256": self.payload_sha256,
            "generation": self.generation,
            "selection_mode": self.selection_mode,
            "entry_count": len(self.entries),
            "trainable_entry_count": len(self.trainable_entries),
            "reference_entry_count": len(self.reference_entries),
            "contract": dict(self.contract),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_nonempty_string(
    mapping: Mapping[str, object],
    name: str,
) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"external opponent {name} must be a non-empty string")
    return value


def _required_integer(
    mapping: Mapping[str, object],
    name: str,
    *,
    minimum: int,
) -> int:
    value = mapping.get(name)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(
            f"external opponent {name} must be an integer >= {minimum}"
        )
    return value


def load_external_opponent_manifest(
    path: str | Path,
    *,
    external_weight: float,
    repository_root: str | Path | None = None,
) -> ExternalOpponentManifest:
    if not math.isfinite(external_weight) or external_weight < 0:
        raise ValueError("external_weight must be finite and non-negative")
    manifest_path = Path(path).resolve()
    raw = manifest_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("external opponent manifest must be a JSON object")
    if payload.get("schema_version") != EXTERNAL_OPPONENT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported external opponent manifest schema")
    if payload.get("report_kind") != "ppo_league_generation_manifest":
        raise ValueError("external opponent manifest has the wrong report_kind")
    if payload.get("immutable") is not True:
        raise ValueError("external opponent manifest must declare immutable=true")
    if payload.get("path_base") != "repository_root":
        raise ValueError("external opponent manifest path_base must be repository_root")
    generation = payload.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise ValueError("external opponent manifest generation is invalid")
    selection_mode = payload.get("selection_mode")
    if selection_mode not in EXTERNAL_OPPONENT_SELECTION_MODES:
        raise ValueError(
            f"unsupported external opponent selection mode {selection_mode!r}"
        )
    contract = payload.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("external opponent manifest has no contract object")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("external opponent manifest entries must be non-empty")
    base = (
        Path.cwd().resolve()
        if repository_root is None
        else Path(repository_root).resolve()
    )
    required_contract_fields = {
        "experiment_versions",
        "policy_architecture",
        "model_structure_sha256",
        "catalog_sha256",
        "rulebook_sha256",
        "observation_schema_sha256",
        "action_layout_sha256",
    }
    missing_contract_fields = sorted(required_contract_fields - set(contract))
    if missing_contract_fields:
        raise ValueError(
            "external opponent manifest contract is missing "
            + ", ".join(missing_contract_fields)
        )
    if not isinstance(contract["experiment_versions"], dict):
        raise ValueError(
            "external opponent manifest experiment_versions must be an object"
        )
    contract_architecture = contract["policy_architecture"]
    if not isinstance(contract_architecture, str) or not contract_architecture:
        raise ValueError(
            "external opponent manifest policy_architecture must be non-empty"
        )
    for name in (
        "model_structure_sha256",
        "catalog_sha256",
        "rulebook_sha256",
        "observation_schema_sha256",
        "action_layout_sha256",
    ):
        if not _is_sha256(contract[name]):
            raise ValueError(
                f"external opponent manifest {name} must be lowercase SHA-256"
            )
    sampling_weights = []
    training_eligibility = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("external opponent manifest entry must be an object")
        eligible = raw_entry.get("training_eligible")
        if not isinstance(eligible, bool):
            raise ValueError(
                "external opponent training_eligible must be boolean"
            )
        value = raw_entry.get("sampling_weight")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(
                "external opponent sampling_weight must be finite and non-negative"
            )
        if eligible != (float(value) > 0):
            raise ValueError(
                "trainable external opponents need positive sampling weight; "
                "reference-only opponents need zero sampling weight"
            )
        sampling_weights.append(float(value))
        training_eligibility.append(eligible)
    weight_total = sum(
        weight
        for weight, eligible in zip(sampling_weights, training_eligibility)
        if eligible
    )
    if external_weight > 0 and weight_total <= 0:
        raise ValueError(
            "positive external_weight requires at least one trainable entry"
        )
    entries = []
    ids = set()
    hashes = set()
    for raw_entry, sampling_weight, eligible in zip(
        raw_entries,
        sampling_weights,
        training_eligibility,
    ):
        opponent_id = raw_entry.get("opponent_id")
        if not isinstance(opponent_id, str) or not opponent_id:
            raise ValueError("external opponent_id must be non-empty")
        if opponent_id in ids:
            raise ValueError(f"duplicate external opponent_id {opponent_id!r}")
        ids.add(opponent_id)
        relative_path = raw_entry.get("checkpoint_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("external checkpoint_path must be non-empty")
        checkpoint = (base / relative_path).resolve()
        try:
            checkpoint.relative_to(base)
        except ValueError as exc:
            raise ValueError(
                f"external checkpoint escapes repository root: {relative_path}"
            ) from exc
        if not checkpoint.is_file():
            raise ValueError(f"external checkpoint does not exist: {checkpoint}")
        expected_hash = raw_entry.get("checkpoint_sha256")
        if not _is_sha256(expected_hash):
            raise ValueError(
                f"external opponent {opponent_id} checkpoint_sha256 is invalid"
            )
        actual_hash = _sha256_file(checkpoint)
        if expected_hash != actual_hash:
            raise ValueError(
                f"external checkpoint hash mismatch for {opponent_id}: "
                f"manifest={expected_hash!r}, actual={actual_hash!r}"
            )
        if actual_hash in hashes:
            raise ValueError(
                f"duplicate external checkpoint content for {opponent_id}"
            )
        hashes.add(actual_hash)
        entry_generation = raw_entry.get("generation")
        if (
            not isinstance(entry_generation, int)
            or isinstance(entry_generation, bool)
            or entry_generation < 0
            or entry_generation > generation
        ):
            raise ValueError(
                f"external opponent {opponent_id} has invalid model generation "
                f"{entry_generation!r} for pool generation {generation}"
            )
        policy_seed = _required_integer(
            raw_entry,
            "policy_seed",
            minimum=0,
        )
        training_steps = _required_integer(
            raw_entry,
            "training_steps",
            minimum=1,
        )
        role = _required_nonempty_string(raw_entry, "role")
        rules_version = _required_nonempty_string(
            raw_entry,
            "rules_version",
        )
        policy_architecture = _required_nonempty_string(
            raw_entry,
            "policy_architecture",
        )
        if policy_architecture != contract_architecture:
            raise ValueError(
                f"external opponent {opponent_id} architecture does not match "
                "the manifest contract"
            )
        versions_sha256 = raw_entry.get("versions_sha256")
        if not _is_sha256(versions_sha256):
            raise ValueError(
                f"external opponent {opponent_id} versions_sha256 is invalid"
            )
        entries.append(OpponentEntry(
            opponent_id=opponent_id,
            kind="external",
            weight=(
                external_weight * sampling_weight / weight_total
                if eligible and weight_total > 0
                else 0.0
            ),
            checkpoint_path=str(checkpoint),
            created_agent_steps=training_steps,
            checkpoint_sha256=actual_hash,
            policy_seed=policy_seed,
            training_steps=training_steps,
            generation=entry_generation,
            role=role,
            rules_version=rules_version,
            policy_architecture=policy_architecture,
            versions_sha256=versions_sha256,
        ))
    return ExternalOpponentManifest(
        path=str(manifest_path),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        payload_sha256=stable_json_sha256(payload),
        generation=generation,
        selection_mode=str(selection_mode),
        entries=tuple(entries),
        contract=dict(contract),
    )


class OpponentPool:
    """Reproducible current/history/random/fixed opponent registry."""

    def __init__(
        self,
        master_seed: int,
        *,
        current_weight: float = 1.0,
        random_weight: float = 0.0,
        fixed_weight: float = 0.0,
        historical_weight: float = 0.0,
        max_history: int = 8,
        snapshot_interval_steps: int = 50_000,
        external_entries: Sequence[OpponentEntry] = (),
        external_manifest_path: str | None = None,
        external_manifest_sha256: str | None = None,
        external_generation: int | None = None,
    ) -> None:
        if max_history <= 0 or snapshot_interval_steps <= 0:
            raise ValueError("max_history and snapshot_interval_steps must be positive")
        self.master_seed = master_seed
        self.max_history = max_history
        self.snapshot_interval_steps = snapshot_interval_steps
        self.historical_weight = historical_weight
        self.last_snapshot_steps = 0
        self.selection_count = 0
        self.selection_counts: dict[str, int] = {}
        self.selection_counts_by_opponent: dict[str, int] = {}
        self._base_entries = (
            OpponentEntry("current", "current", current_weight),
            OpponentEntry("random_legal", "random_legal", random_weight),
            OpponentEntry("fixed_first_legal", "fixed", fixed_weight),
        )
        if any(entry.kind != "external" for entry in external_entries):
            raise ValueError("external_entries must contain only external opponents")
        if external_entries and (
            external_manifest_path is None
            or external_manifest_sha256 is None
            or external_generation is None
        ):
            raise ValueError("external opponents require manifest identity")
        self._external = tuple(external_entries)
        self.external_manifest_path = external_manifest_path
        self.external_manifest_sha256 = external_manifest_sha256
        self.external_generation = external_generation
        self._history: list[OpponentEntry] = []

    @property
    def entries(self) -> tuple[OpponentEntry, ...]:
        return (*self._base_entries, *self._external, *self._history)

    def select(self, *, episode_id: int, learner_player: int) -> OpponentEntry:
        candidates = [entry for entry in self.entries if entry.weight > 0]
        if not candidates:
            raise ValueError("opponent pool has no positive selection weights")
        rng = random.Random(
            derive_seed(self.master_seed, "opponent", episode_id, learner_player)
        )
        selected = rng.choices(
            candidates,
            weights=[entry.weight for entry in candidates],
            k=1,
        )[0]
        self.selection_count += 1
        self.selection_counts[selected.kind] = (
            self.selection_counts.get(selected.kind, 0) + 1
        )
        self.selection_counts_by_opponent[selected.opponent_id] = (
            self.selection_counts_by_opponent.get(selected.opponent_id, 0) + 1
        )
        return selected

    def snapshot_due(self, agent_steps: int) -> bool:
        return (
            self.historical_weight > 0
            and agent_steps - self.last_snapshot_steps
            >= self.snapshot_interval_steps
        )

    def register_snapshot(self, path: str | Path, *, agent_steps: int) -> OpponentEntry:
        if agent_steps <= self.last_snapshot_steps:
            raise ValueError("snapshot agent_steps must increase monotonically")
        entry = OpponentEntry(
            opponent_id=f"historical_{agent_steps:012d}",
            kind="historical",
            weight=self.historical_weight,
            checkpoint_path=str(path),
            created_agent_steps=agent_steps,
        )
        self._history.append(entry)
        self._history = self._history[-self.max_history :]
        self.last_snapshot_steps = agent_steps
        return entry

    def state_dict(self) -> dict[str, object]:
        return {
            "master_seed": self.master_seed,
            "max_history": self.max_history,
            "snapshot_interval_steps": self.snapshot_interval_steps,
            "historical_weight": self.historical_weight,
            "last_snapshot_steps": self.last_snapshot_steps,
            "selection_count": self.selection_count,
            "selection_counts": dict(sorted(self.selection_counts.items())),
            "selection_counts_by_opponent": dict(sorted(
                self.selection_counts_by_opponent.items()
            )),
            "base_entries": [asdict(entry) for entry in self._base_entries],
            "external_entries": [asdict(entry) for entry in self._external],
            "external_manifest_path": self.external_manifest_path,
            "external_manifest_sha256": self.external_manifest_sha256,
            "external_generation": self.external_generation,
            "history": [asdict(entry) for entry in self._history],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, object],
        *,
        expected_external_manifest_sha256: str | None = None,
        expected_external_entries: Sequence[OpponentEntry] | None = None,
    ) -> OpponentPool:
        base = state["base_entries"]
        weights = {entry["kind"]: entry["weight"] for entry in base}
        stored_manifest_hash = state.get("external_manifest_sha256")
        if expected_external_manifest_sha256 != stored_manifest_hash:
            raise ValueError(
                "external opponent manifest hash changed across resume: "
                f"checkpoint={stored_manifest_hash!r}, "
                f"runtime={expected_external_manifest_sha256!r}"
            )
        stored_external_entries = tuple(
            OpponentEntry(**entry)
            for entry in state.get("external_entries", [])
        )
        if (
            expected_external_entries is not None
            and stored_external_entries != tuple(expected_external_entries)
        ):
            raise ValueError(
                "external opponent entries changed across resume"
            )
        pool = cls(
            int(state["master_seed"]),
            current_weight=float(weights["current"]),
            random_weight=float(weights["random_legal"]),
            fixed_weight=float(weights["fixed"]),
            historical_weight=float(state["historical_weight"]),
            max_history=int(state["max_history"]),
            snapshot_interval_steps=int(state["snapshot_interval_steps"]),
            external_entries=stored_external_entries,
            external_manifest_path=state.get("external_manifest_path"),
            external_manifest_sha256=stored_manifest_hash,
            external_generation=state.get("external_generation"),
        )
        pool._history = [OpponentEntry(**entry) for entry in state["history"]]
        pool.last_snapshot_steps = int(state["last_snapshot_steps"])
        pool.selection_count = int(state["selection_count"])
        pool.selection_counts = {
            str(kind): int(count)
            for kind, count in state.get("selection_counts", {}).items()
        }
        pool.selection_counts_by_opponent = {
            str(opponent_id): int(count)
            for opponent_id, count in state.get(
                "selection_counts_by_opponent", {}
            ).items()
        }
        return pool


class OpponentEpisodeScheduler:
    """Deterministically batch episode IDs without changing their opponents."""

    def __init__(
        self,
        pool: OpponentPool,
        *,
        worker_count: int,
        mode: str = "sequential",
    ) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if mode not in OPPONENT_BATCHING_MODES:
            raise ValueError(f"unsupported opponent batching mode {mode!r}")
        self.pool = pool
        self.worker_count = worker_count
        self.mode = mode
        self._pending: dict[int, OpponentEntry] = {}

    @property
    def pending(self) -> tuple[tuple[int, OpponentEntry], ...]:
        return tuple(sorted(self._pending.items()))

    def next_wave(
        self,
        next_episode_id: int,
        *,
        learner_player_for_episode,
    ) -> tuple[int, tuple[tuple[int, OpponentEntry], ...]]:
        if self.mode == "sequential":
            selected = []
            for episode_id in range(
                next_episode_id,
                next_episode_id + self.worker_count,
            ):
                learner_player = learner_player_for_episode(episode_id)
                selected.append((episode_id, self.pool.select(
                    episode_id=episode_id,
                    learner_player=learner_player,
                )))
            return next_episode_id + self.worker_count, tuple(selected)

        while True:
            slots_by_opponent: dict[str, dict[int, int]] = {}
            for episode_id, entry in self.pending:
                slots = slots_by_opponent.setdefault(entry.opponent_id, {})
                slots.setdefault(episode_id % self.worker_count, episode_id)
            ready = [
                (
                    max(slots.values()),
                    opponent_id,
                    tuple(slots[slot] for slot in range(self.worker_count)),
                )
                for opponent_id, slots in slots_by_opponent.items()
                if len(slots) == self.worker_count
            ]
            if ready:
                _, _, episode_ids = min(ready)
                wave = tuple(
                    (episode_id, self._pending.pop(episode_id))
                    for episode_id in episode_ids
                )
                return next_episode_id, wave
            learner_player = learner_player_for_episode(next_episode_id)
            self._pending[next_episode_id] = self.pool.select(
                episode_id=next_episode_id,
                learner_player=learner_player,
            )
            next_episode_id += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "worker_count": self.worker_count,
            "mode": self.mode,
            "pending": [
                {
                    "episode_id": episode_id,
                    "opponent": asdict(entry),
                }
                for episode_id, entry in self.pending
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state.get("worker_count", self.worker_count)) != self.worker_count:
            raise ValueError(
                "opponent episode scheduler worker_count changed across resume"
            )
        if str(state.get("mode", self.mode)) != self.mode:
            raise ValueError(
                "opponent episode scheduler mode changed across resume"
            )
        pending: dict[int, OpponentEntry] = {}
        for row in state.get("pending", []):
            episode_id = int(row["episode_id"])
            if episode_id in pending:
                raise ValueError("duplicate pending opponent episode ID")
            entry = OpponentEntry(**row["opponent"])
            registered = {
                candidate.opponent_id: candidate
                for candidate in self.pool.entries
            }.get(entry.opponent_id)
            if registered != entry:
                raise ValueError(
                    "pending opponent entry does not match resumed pool"
                )
            pending[episode_id] = entry
        self._pending = pending
