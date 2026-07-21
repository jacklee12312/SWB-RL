from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

from swb.rl.seeding import derive_seed


@dataclass(frozen=True)
class OpponentEntry:
    opponent_id: str
    kind: str
    weight: float
    checkpoint_path: str | None = None
    created_agent_steps: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"current", "historical", "random_legal", "fixed"}:
            raise ValueError(f"unsupported opponent kind {self.kind!r}")
        if self.weight < 0:
            raise ValueError("opponent weight must be non-negative")
        if self.kind == "historical" and not self.checkpoint_path:
            raise ValueError("historical opponent requires checkpoint_path")


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
    ) -> None:
        if max_history <= 0 or snapshot_interval_steps <= 0:
            raise ValueError("max_history and snapshot_interval_steps must be positive")
        self.master_seed = master_seed
        self.max_history = max_history
        self.snapshot_interval_steps = snapshot_interval_steps
        self.historical_weight = historical_weight
        self.last_snapshot_steps = 0
        self.selection_count = 0
        self._base_entries = (
            OpponentEntry("current", "current", current_weight),
            OpponentEntry("random_legal", "random_legal", random_weight),
            OpponentEntry("fixed_first_legal", "fixed", fixed_weight),
        )
        self._history: list[OpponentEntry] = []

    @property
    def entries(self) -> tuple[OpponentEntry, ...]:
        return (*self._base_entries, *self._history)

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
        return selected

    def snapshot_due(self, agent_steps: int) -> bool:
        return agent_steps - self.last_snapshot_steps >= self.snapshot_interval_steps

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
            "base_entries": [asdict(entry) for entry in self._base_entries],
            "history": [asdict(entry) for entry in self._history],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> OpponentPool:
        base = state["base_entries"]
        weights = {entry["kind"]: entry["weight"] for entry in base}
        pool = cls(
            int(state["master_seed"]),
            current_weight=float(weights["current"]),
            random_weight=float(weights["random_legal"]),
            fixed_weight=float(weights["fixed"]),
            historical_weight=float(state["historical_weight"]),
            max_history=int(state["max_history"]),
            snapshot_interval_steps=int(state["snapshot_interval_steps"]),
        )
        pool._history = [OpponentEntry(**entry) for entry in state["history"]]
        pool.last_snapshot_steps = int(state["last_snapshot_steps"])
        pool.selection_count = int(state["selection_count"])
        return pool
