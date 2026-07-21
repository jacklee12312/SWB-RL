from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.environment import ShadowverseEnv
from swb.rl.catalog import DEFAULT_COVERAGE_REPORT, TrainableCardCatalog


@dataclass(frozen=True)
class WorkerAssets:
    """Process-local runtime assets loaded without filesystem access."""

    catalog: TrainableCardCatalog
    rulebook: RuleBook
    rulebook_sha256: str


@dataclass(frozen=True)
class WorkerAssetsSnapshot:
    """Serializable parent-built snapshot for Windows/Linux spawn workers."""

    catalog: TrainableCardCatalog
    rulebook_pickle: bytes
    rulebook_sha256: str

    @classmethod
    def build(
        cls,
        repository: CardRepository,
        *,
        rule_directory: str | Path = ShadowverseEnv.DEFAULT_RULE_DIRECTORY,
        coverage_report: str | Path = DEFAULT_COVERAGE_REPORT,
    ) -> WorkerAssetsSnapshot:
        rule_path = Path(rule_directory)
        rulebook = RuleBook.from_directory(rule_path)
        return cls(
            catalog=TrainableCardCatalog.from_repository(
                repository,
                coverage_report=coverage_report,
            ),
            rulebook_pickle=pickle.dumps(rulebook, protocol=pickle.HIGHEST_PROTOCOL),
            rulebook_sha256=hash_rule_directory(rule_path),
        )

    def load(self) -> WorkerAssets:
        """Materialize one worker-local RuleBook; performs no I/O."""
        return WorkerAssets(
            catalog=self.catalog,
            rulebook=pickle.loads(self.rulebook_pickle),
            rulebook_sha256=self.rulebook_sha256,
        )


def hash_rule_directory(directory: str | Path) -> str:
    """Hash rule relative paths and bytes in deterministic order."""
    root = Path(directory)
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.json") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
