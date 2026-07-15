from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from swb.engine.effects import EffectOperation


class UnionBurstKind(str, Enum):
    UNION_BURST = "union_burst"
    SUPER_SKYBOUND_ART = "super_skybound_art"

    @property
    def threshold(self) -> int:
        return 10 if self is UnionBurstKind.UNION_BURST else 15


@dataclass(frozen=True)
class UnionBurstDefinition:
    card_id: int
    kind: UnionBurstKind
    operations: tuple[EffectOperation, ...]
    replace_base_operations: bool = False

    @property
    def threshold(self) -> int:
        return self.kind.threshold
