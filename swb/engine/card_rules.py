from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from swb.engine.effects import EffectKind, EffectOperation, TargetKind


class Trigger(str, Enum):
    PLAY = "play"
    FANFARE = "fanfare"
    LAST_WORDS = "last_words"
    EVOLVE = "evolve"
    SUPER_EVOLVE = "super_evolve"
    ATTACK = "attack"
    CLASH = "clash"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    COUNTDOWN_EXPIRED = "countdown_expired"


@dataclass(frozen=True)
class CardRule:
    card_id: int
    trigger: Trigger
    operations: tuple[EffectOperation, ...]
    countdown: int | None = None


class RuleBook:
    def __init__(self, rules: tuple[CardRule, ...] = ()):
        self._rules: dict[tuple[int, Trigger], tuple[EffectOperation, ...]] = {
            (rule.card_id, rule.trigger): rule.operations for rule in rules
        }
        self._countdowns = {
            rule.card_id: rule.countdown
            for rule in rules
            if rule.countdown is not None
        }

    def operations_for(
        self, card_id: int, trigger: Trigger
    ) -> tuple[EffectOperation, ...]:
        return self._rules.get((card_id, trigger), ())

    def countdown_for(self, card_id: int) -> int | None:
        return self._countdowns.get(card_id)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "RuleBook":
        path = Path(directory)
        if not path.exists():
            return cls()
        rules: list[CardRule] = []
        for file_path in sorted(path.glob("*.json")):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            entries = payload if isinstance(payload, list) else payload["rules"]
            for entry in entries:
                rules.append(
                    CardRule(
                        card_id=int(entry["card_id"]),
                        trigger=Trigger(entry["trigger"]),
                        operations=tuple(
                            EffectOperation(
                                kind=EffectKind(operation["kind"]),
                                target=TargetKind(operation["target"]),
                                amount=int(operation.get("amount", 0)),
                                secondary_amount=int(
                                    operation.get("secondary_amount", 0)
                                ),
                                card_id=operation.get("card_id"),
                                keyword=operation.get("keyword"),
                            )
                            for operation in entry.get("operations", [])
                        ),
                        countdown=entry.get("countdown"),
                    )
                )
        return cls(tuple(rules))
