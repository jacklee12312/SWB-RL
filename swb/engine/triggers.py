from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.effects import EffectOperation

if TYPE_CHECKING:
    from swb.engine.state import Unit
    from swb.db.repository import CardDefinition


@dataclass
class TriggerRecord:
    card_id: int
    trigger: Trigger
    operations: tuple[EffectOperation, ...]
    source_entity_id: int
    source_name: str
    definition: CardDefinition
    owner: int
    board_position: int

    def __lt__(self, other: TriggerRecord) -> bool:
        return (self.owner, self.board_position) < (other.owner, other.board_position)


@dataclass
class TriggerBatch:
    records: list[TriggerRecord] = field(default_factory=list)

    def sort_by_active_player(self, active_player: int) -> None:
        self.records.sort(key=lambda r: (0 if r.owner == active_player else 1, r.owner, r.board_position))


def collect_triggers(
    rulebook: RuleBook,
    entities: list[tuple[int, Unit, int]],  # (player_index, unit, board_position)
    trigger: Trigger,
    active_player: int,
) -> TriggerBatch:
    batch = TriggerBatch()
    for player_index, unit, board_pos in entities:
        operations = rulebook.operations_for(unit.definition.card_id, trigger)
        if operations:
            batch.records.append(TriggerRecord(
                card_id=unit.definition.card_id,
                trigger=trigger,
                operations=operations,
                source_entity_id=unit.entity_id,
                source_name=unit.definition.name,
                definition=unit.definition,
                owner=player_index,
                board_position=board_pos,
            ))
    batch.sort_by_active_player(active_player)
    return batch
