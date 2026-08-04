from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


PFSP_SAMPLERS = frozenset({"uniform", "variance", "hard"})
TRAINING_PAYOFF_SNAPSHOT_SCHEMA_VERSION = 1
DEFAULT_EPSILON_FLOOR = 0.02
DEFAULT_MAX_PROBABILITY = 0.35
DEFAULT_HARD_ALPHA = 1.0
DEFAULT_RETAINED_SCORE_THRESHOLD = 0.70
DEFAULT_FORGOTTEN_SCORE_THRESHOLD = 0.40
_TOLERANCE = 1e-12


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


@dataclass(frozen=True)
class PayoffEstimate:
    opponent_id: str
    checkpoint_sha256: str
    games: int
    score_rate: float | None
    confidence_interval_95: tuple[float, float] | None

    def __post_init__(self) -> None:
        if not self.opponent_id:
            raise ValueError("payoff opponent_id must be non-empty")
        if not _is_sha256(self.checkpoint_sha256):
            raise ValueError("payoff checkpoint_sha256 must be lowercase SHA-256")
        if (
            not isinstance(self.games, int)
            or isinstance(self.games, bool)
            or self.games < 0
        ):
            raise ValueError("payoff games must be a non-negative integer")
        if self.score_rate is not None and (
            not math.isfinite(self.score_rate)
            or not 0.0 <= self.score_rate <= 1.0
        ):
            raise ValueError("payoff score_rate must be in [0, 1]")
        if self.confidence_interval_95 is not None:
            lower, upper = self.confidence_interval_95
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or not 0.0 <= lower <= upper <= 1.0
            ):
                raise ValueError(
                    "payoff confidence_interval_95 must be ordered in [0, 1]"
                )
            if self.score_rate is None or not lower <= self.score_rate <= upper:
                raise ValueError(
                    "payoff confidence interval must contain score_rate"
                )
        if self.games == 0 and (
            self.score_rate is not None
            or self.confidence_interval_95 is not None
        ):
            raise ValueError("unevaluated payoff rows must have no estimate")
        if self.games > 0 and self.score_rate is None:
            raise ValueError("evaluated payoff rows require score_rate")

    @property
    def reliable(self) -> bool:
        return (
            self.games > 0
            and self.score_rate is not None
            and self.confidence_interval_95 is not None
        )


@dataclass(frozen=True)
class TrainingPayoffSnapshot:
    path: str
    file_sha256: str
    source_generation: int
    target_generation: int
    source_generation_manifest_path: str
    source_generation_manifest_sha256: str
    match_master_seeds: tuple[int, ...]
    focal_policy_ids: tuple[str, ...]
    aggregation: str
    estimates: tuple[PayoffEstimate, ...]

    def summary(self) -> dict[str, object]:
        return {
            "path": self.path,
            "file_sha256": self.file_sha256,
            "source_generation": self.source_generation,
            "target_generation": self.target_generation,
            "source_generation_manifest_path": (
                self.source_generation_manifest_path
            ),
            "source_generation_manifest_sha256": (
                self.source_generation_manifest_sha256
            ),
            "match_master_seeds": list(self.match_master_seeds),
            "focal_policy_ids": list(self.focal_policy_ids),
            "aggregation": self.aggregation,
            "opponent_count": len(self.estimates),
            "reliable_estimate_count": sum(
                estimate.reliable for estimate in self.estimates
            ),
        }


@dataclass(frozen=True)
class PFSPDistribution:
    sampler: str
    epsilon_floor: float
    maximum_probability: float
    hard_alpha: float
    probabilities: Mapping[str, float]
    raw_weights: Mapping[str, float]
    raw_probabilities: Mapping[str, float]
    unreliable_opponent_ids: tuple[str, ...]
    epsilon_bound_opponent_ids: tuple[str, ...]
    cap_bound_opponent_ids: tuple[str, ...]
    cap_exception_required: bool

    def report(self) -> dict[str, object]:
        return {
            "sampler": self.sampler,
            "formula": {
                "uniform": "1",
                "variance": "p * (1 - p)",
                "hard": "(1 - p) ** alpha",
            }[self.sampler],
            "epsilon_floor": self.epsilon_floor,
            "maximum_probability": self.maximum_probability,
            "hard_alpha": self.hard_alpha,
            "raw_weights": dict(self.raw_weights),
            "raw_probabilities": dict(self.raw_probabilities),
            "probabilities": dict(self.probabilities),
            "unreliable_opponent_ids": list(self.unreliable_opponent_ids),
            "epsilon_bound_opponent_ids": list(
                self.epsilon_bound_opponent_ids
            ),
            "cap_bound_opponent_ids": list(self.cap_bound_opponent_ids),
            "cap_exception_required": self.cap_exception_required,
            "probability_sum": sum(self.probabilities.values()),
        }


def load_training_payoff_snapshot(
    path: str | Path,
    *,
    generation_manifest_path: str | Path,
    allowed_tuning_seeds: Sequence[int],
    forbidden_final_seeds: Sequence[int],
    repository_root: str | Path | None = None,
) -> TrainingPayoffSnapshot:
    root = (
        Path.cwd().resolve()
        if repository_root is None
        else Path(repository_root).resolve()
    )
    snapshot_path = Path(path).resolve()
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training payoff snapshot must be a JSON object")
    if payload.get("schema_version") != TRAINING_PAYOFF_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported training payoff snapshot schema")
    if payload.get("report_kind") != "ppo_league_training_payoff_snapshot":
        raise ValueError("training payoff snapshot has the wrong report_kind")
    if payload.get("immutable") is not True:
        raise ValueError("training payoff snapshot must declare immutable=true")
    evaluator = payload.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("training payoff snapshot has no evaluator contract")
    if evaluator.get("data_partition") != "pfsp_tuning":
        raise ValueError("PFSP requires the pfsp_tuning evaluator partition")
    if evaluator.get("payoff_update_boundary") != "generation_end":
        raise ValueError("PFSP payoff updates must occur at generation_end")
    match_seeds = evaluator.get("match_master_seeds")
    if (
        not isinstance(match_seeds, list)
        or not match_seeds
        or any(
            not isinstance(seed, int) or isinstance(seed, bool)
            for seed in match_seeds
        )
        or len(set(match_seeds)) != len(match_seeds)
    ):
        raise ValueError("PFSP tuning match seeds must be unique integers")
    allowed = set(allowed_tuning_seeds)
    forbidden = set(forbidden_final_seeds)
    unexpected = sorted(set(match_seeds) - allowed)
    leaked = sorted(set(match_seeds) & forbidden)
    if unexpected or leaked:
        raise ValueError(
            "PFSP payoff snapshot uses unregistered or final evaluation seeds: "
            f"unexpected={unexpected}, leaked={leaked}"
        )

    manifest_path = Path(generation_manifest_path).resolve()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = _sha256_file(manifest_path)
    source = payload.get("source_generation_manifest")
    if not isinstance(source, dict):
        raise ValueError("payoff snapshot has no source generation manifest")
    declared_path = source.get("path")
    if not isinstance(declared_path, str):
        raise ValueError("payoff snapshot generation path must be a string")
    declared_resolved = (root / declared_path).resolve()
    if declared_resolved != manifest_path:
        raise ValueError("payoff snapshot generation manifest path changed")
    if source.get("sha256") != manifest_hash:
        raise ValueError("payoff snapshot generation manifest hash changed")
    source_generation = payload.get("source_generation")
    target_generation = payload.get("target_generation")
    if (
        not isinstance(source_generation, int)
        or isinstance(source_generation, bool)
        or source_generation != manifest_payload.get("generation")
        or target_generation != source_generation + 1
    ):
        raise ValueError("payoff snapshot generation boundary is invalid")

    focal_policy_ids = payload.get("focal_policy_ids")
    if (
        not isinstance(focal_policy_ids, list)
        or not focal_policy_ids
        or any(not isinstance(value, str) or not value for value in focal_policy_ids)
        or len(set(focal_policy_ids)) != len(focal_policy_ids)
    ):
        raise ValueError("payoff snapshot focal_policy_ids are invalid")
    aggregation = payload.get("aggregation")
    if aggregation != "equal_weight_mean_across_focal_policies":
        raise ValueError("unsupported PFSP payoff aggregation")

    manifest_entries = {
        str(entry["opponent_id"]): entry
        for entry in manifest_payload.get("entries", [])
        if bool(entry.get("training_eligible"))
    }
    raw_estimates = payload.get("opponents")
    if not isinstance(raw_estimates, list):
        raise ValueError("payoff snapshot opponents must be a list")
    estimates = []
    seen_ids = set()
    seen_hashes = set()
    for raw in raw_estimates:
        if not isinstance(raw, dict):
            raise ValueError("payoff snapshot opponent row must be an object")
        opponent_id = raw.get("opponent_id")
        checkpoint_hash = raw.get("checkpoint_sha256")
        if opponent_id in seen_ids:
            raise ValueError(f"duplicate payoff opponent {opponent_id!r}")
        if checkpoint_hash in seen_hashes:
            raise ValueError("duplicate payoff checkpoint content")
        seen_ids.add(opponent_id)
        seen_hashes.add(checkpoint_hash)
        interval = raw.get("confidence_interval_95")
        if interval is not None and (
            not isinstance(interval, list) or len(interval) != 2
        ):
            raise ValueError(
                "payoff confidence_interval_95 must contain two values"
            )
        estimate = PayoffEstimate(
            opponent_id=str(opponent_id or ""),
            checkpoint_sha256=str(checkpoint_hash or ""),
            games=raw.get("games"),
            score_rate=raw.get("score_rate"),
            confidence_interval_95=(
                None
                if interval is None
                else (float(interval[0]), float(interval[1]))
            ),
        )
        manifest_entry = manifest_entries.get(estimate.opponent_id)
        if manifest_entry is None:
            raise ValueError(
                f"payoff opponent {estimate.opponent_id!r} is not trainable"
            )
        if manifest_entry.get("checkpoint_sha256") != estimate.checkpoint_sha256:
            raise ValueError(
                f"payoff checkpoint hash changed for {estimate.opponent_id}"
            )
        estimates.append(estimate)
    if set(seen_ids) != set(manifest_entries):
        raise ValueError("payoff snapshot must cover every trainable opponent")

    return TrainingPayoffSnapshot(
        path=str(snapshot_path),
        file_sha256=_sha256_file(snapshot_path),
        source_generation=int(source_generation),
        target_generation=int(target_generation),
        source_generation_manifest_path=str(manifest_path),
        source_generation_manifest_sha256=manifest_hash,
        match_master_seeds=tuple(match_seeds),
        focal_policy_ids=tuple(focal_policy_ids),
        aggregation=aggregation,
        estimates=tuple(sorted(estimates, key=lambda row: row.opponent_id)),
    )


def _raw_weight(estimate: PayoffEstimate, sampler: str, alpha: float) -> float:
    if sampler == "uniform":
        return 1.0
    if not estimate.reliable:
        return 0.0
    assert estimate.score_rate is not None
    if sampler == "variance":
        return estimate.score_rate * (1.0 - estimate.score_rate)
    if sampler == "hard":
        return (1.0 - estimate.score_rate) ** alpha
    raise ValueError(f"unsupported PFSP sampler {sampler!r}")


def _bounded_probabilities(
    raw_weights: Sequence[float],
    *,
    epsilon_floor: float,
    maximum_probability: float,
) -> tuple[list[float], bool]:
    count = len(raw_weights)
    if count == 0:
        raise ValueError("PFSP requires at least one opponent")
    if count == 1:
        return [1.0], maximum_probability < 1.0
    if epsilon_floor * count > 1.0 + _TOLERANCE:
        raise ValueError("epsilon floor is infeasible for opponent count")
    if maximum_probability < epsilon_floor:
        raise ValueError("maximum probability must not be below epsilon floor")
    cap_exception = maximum_probability * count < 1.0 - _TOLERANCE
    upper = 1.0 if cap_exception else maximum_probability
    probabilities = [epsilon_floor for _ in raw_weights]
    remaining = 1.0 - epsilon_floor * count
    active = set(range(count))
    while remaining > _TOLERANCE and active:
        weight_total = sum(raw_weights[index] for index in active)
        shares = {
            index: (
                remaining / len(active)
                if weight_total <= _TOLERANCE
                else remaining * raw_weights[index] / weight_total
            )
            for index in active
        }
        capped = [
            index
            for index, share in shares.items()
            if probabilities[index] + share > upper + _TOLERANCE
        ]
        if not capped:
            for index, share in shares.items():
                probabilities[index] += share
            remaining = 0.0
            break
        for index in sorted(capped):
            available = upper - probabilities[index]
            probabilities[index] = upper
            remaining -= available
            active.remove(index)
    if remaining > 1e-9:
        raise ValueError("probability constraints cannot fill the simplex")
    correction = 1.0 - sum(probabilities)
    if abs(correction) > _TOLERANCE:
        candidates = [
            index
            for index, probability in enumerate(probabilities)
            if epsilon_floor - _TOLERANCE
            <= probability + correction
            <= upper + _TOLERANCE
        ]
        if not candidates:
            raise ValueError("probability rounding correction is infeasible")
        probabilities[candidates[-1]] += correction
    return probabilities, cap_exception


def compute_pfsp_distribution(
    estimates: Sequence[PayoffEstimate],
    *,
    sampler: str,
    epsilon_floor: float = DEFAULT_EPSILON_FLOOR,
    maximum_probability: float = DEFAULT_MAX_PROBABILITY,
    hard_alpha: float = DEFAULT_HARD_ALPHA,
) -> PFSPDistribution:
    if sampler not in PFSP_SAMPLERS:
        raise ValueError(f"unsupported PFSP sampler {sampler!r}")
    if (
        not math.isfinite(epsilon_floor)
        or not 0.0 <= epsilon_floor < 1.0
    ):
        raise ValueError("epsilon_floor must be finite and in [0, 1)")
    if (
        not math.isfinite(maximum_probability)
        or not 0.0 < maximum_probability <= 1.0
    ):
        raise ValueError("maximum_probability must be finite and in (0, 1]")
    if not math.isfinite(hard_alpha) or hard_alpha <= 0.0:
        raise ValueError("hard_alpha must be finite and positive")
    ordered = tuple(sorted(estimates, key=lambda row: row.opponent_id))
    if not ordered:
        raise ValueError("PFSP requires at least one opponent")
    ids = [estimate.opponent_id for estimate in ordered]
    hashes = [estimate.checkpoint_sha256 for estimate in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("PFSP opponents must have unique IDs")
    if len(set(hashes)) != len(hashes):
        raise ValueError("PFSP opponents must have unique checkpoint content")
    raw = [_raw_weight(estimate, sampler, hard_alpha) for estimate in ordered]
    raw_total = sum(raw)
    raw_probabilities = (
        [1.0 / len(raw) for _ in raw]
        if raw_total <= _TOLERANCE
        else [weight / raw_total for weight in raw]
    )
    probabilities, cap_exception = _bounded_probabilities(
        raw,
        epsilon_floor=epsilon_floor,
        maximum_probability=maximum_probability,
    )
    return PFSPDistribution(
        sampler=sampler,
        epsilon_floor=epsilon_floor,
        maximum_probability=maximum_probability,
        hard_alpha=hard_alpha,
        probabilities=dict(zip(ids, probabilities)),
        raw_weights=dict(zip(ids, raw)),
        raw_probabilities=dict(zip(ids, raw_probabilities)),
        unreliable_opponent_ids=tuple(
            estimate.opponent_id
            for estimate in ordered
            if not estimate.reliable
        ),
        epsilon_bound_opponent_ids=tuple(
            opponent_id
            for opponent_id, probability in zip(ids, probabilities)
            if probability <= epsilon_floor + _TOLERANCE
        ),
        cap_bound_opponent_ids=tuple(
            opponent_id
            for opponent_id, probability in zip(ids, probabilities)
            if (
                not cap_exception
                and probability >= maximum_probability - _TOLERANCE
            )
        ),
        cap_exception_required=cap_exception,
    )


def forgotten_opponent_queue(
    previous: Sequence[PayoffEstimate],
    current: Sequence[PayoffEstimate],
    *,
    retained_score_threshold: float = DEFAULT_RETAINED_SCORE_THRESHOLD,
    forgotten_score_threshold: float = DEFAULT_FORGOTTEN_SCORE_THRESHOLD,
) -> tuple[dict[str, object], ...]:
    if not 0.0 <= forgotten_score_threshold < retained_score_threshold <= 1.0:
        raise ValueError("forgotten thresholds must satisfy 0 <= low < high <= 1")
    previous_by_id = {estimate.opponent_id: estimate for estimate in previous}
    current_by_id = {estimate.opponent_id: estimate for estimate in current}
    if len(previous_by_id) != len(previous) or len(current_by_id) != len(current):
        raise ValueError("forgotten queue inputs require unique opponent IDs")
    queue = []
    for opponent_id in sorted(set(previous_by_id) & set(current_by_id)):
        before = previous_by_id[opponent_id]
        after = current_by_id[opponent_id]
        if not before.reliable or not after.reliable:
            continue
        assert before.score_rate is not None and after.score_rate is not None
        if (
            before.score_rate >= retained_score_threshold
            and after.score_rate < forgotten_score_threshold
        ):
            queue.append({
                "opponent_id": opponent_id,
                "previous_score_rate": before.score_rate,
                "current_score_rate": after.score_rate,
                "drop": before.score_rate - after.score_rate,
                "checkpoint_sha256": after.checkpoint_sha256,
            })
    queue.sort(key=lambda row: (-float(row["drop"]), str(row["opponent_id"])))
    return tuple(queue)
