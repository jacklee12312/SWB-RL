from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Iterable

from swb.engine.effects import EffectOperation, TargetKind
from swb.engine.events import EventType, GameEvent

if TYPE_CHECKING:
    from swb.engine.card_rules import RuleBook


_LIFECYCLE_EVENTS = {
    EventType.CARD_DRAWN: "drawn",
    EventType.CARD_PLAYED: "played",
    EventType.FOLLOWER_EVOLVED: "evolved",
    EventType.FOLLOWER_SUPER_EVOLVED: "super_evolved",
    EventType.ATTACK_DECLARED: "attacked",
    EventType.ENTITY_LEFT_PLAY: "left_play",
}

_DIAGNOSTIC_KINDS = (
    "placeholder",
    "unsupported",
    "resolution_step_limit",
    "illegal_command",
    "illegal_action",
    "action_mask_mismatch",
)


def _stable_component(value: object) -> str:
    encoded = str(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _event_card_id(event: GameEvent) -> int | None:
    for key in ("card_id", "source_card_id", "fusion_card_id"):
        value = event.metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    for key in ("source", "definition", "card"):
        value = event.metadata.get(key)
        definition = getattr(value, "definition", value)
        card_id = getattr(definition, "card_id", None)
        if (
            isinstance(card_id, int)
            and not isinstance(card_id, bool)
            and card_id > 0
        ):
            return card_id
    return None


def _nested_operation_groups(
    operation: EffectOperation,
) -> Iterable[tuple[str, tuple[EffectOperation, ...]]]:
    for name in (
        "earth_rite_operations",
        "necromancy_operations",
        "faith_operations",
        "then_operations",
        "else_operations",
        "optional_operations",
        "repeat_operations",
        "granted_operations",
    ):
        operations = getattr(operation, name)
        if operations:
            yield name.removesuffix("_operations"), operations
    for option_index, option in enumerate(operation.choose_one_options):
        if option.operations:
            yield (
                f"choose:{_stable_component(option.option_id)}:{option_index}",
                option.operations,
            )
    for option_index, option in enumerate(operation.random_choice_options):
        if option.operations:
            yield (
                f"random_choice:{_stable_component(option.option_id)}:"
                f"{option_index}",
                option.operations,
            )
    for bucket_index, operations in enumerate(
        operation.random_distribution_operations
    ):
        if operations:
            yield f"random_distribution:{bucket_index}", operations


class RuntimeCoverageRecorder:
    """Non-semantic counters for card-rule runtime audit.

    The recorder is only created when audit mode is explicitly enabled. Its
    stable identities come from the structured rule tree and never from
    localized text logs or effect-frame labels.
    """

    schema_version = 1

    def __init__(self, rulebook: RuleBook):
        self._rulebook = rulebook
        self._operation_refs: dict[int, list[str]] = defaultdict(list)
        self._operation_groups: dict[int, str] = {}
        self._structural_groups: dict[
            int,
            list[tuple[str, tuple[EffectOperation, ...]]],
        ] = defaultdict(list)
        self._catalog: dict[str, dict[str, object]] = {}
        self._build_catalog()
        self.context: dict[str, object] = {}
        self.reset()

    def _build_catalog(self) -> None:
        for (card_id, trigger), operations in sorted(
            self._rulebook._rules.items(),
            key=lambda item: (item[0][0], item[0][1].value),
        ):
            self._register_group(
                card_id,
                f"card:{card_id}/trigger:{trigger.value}",
                operations,
            )
        for card_id, modes in sorted(self._rulebook._play_modes.items()):
            for mode in sorted(modes, key=lambda item: item.mode_id):
                self._register_group(
                    card_id,
                    f"card:{card_id}/mode:{_stable_component(mode.mode_id)}",
                    mode.operations,
                )
        for emblem_id, definition in sorted(
            self._rulebook._emblem_defs.items()
        ):
            prefix = (
                f"card:{definition.source_card_id}/emblem:"
                f"{_stable_component(emblem_id)}"
            )
            for index, trigger in enumerate(definition.triggers):
                self._register_group(
                    definition.source_card_id,
                    f"{prefix}/trigger:{_stable_component(trigger.trigger)}:"
                    f"{index}",
                    trigger.operations,
                )
            for name in ("on_gain", "on_expire", "last_words"):
                self._register_group(
                    definition.source_card_id,
                    f"{prefix}/{name}",
                    getattr(definition, name),
                )
        for card_id, definitions in sorted(
            self._rulebook._union_burst_defs.items()
        ):
            for definition in definitions:
                self._register_group(
                    card_id,
                    f"card:{card_id}/union_burst:{definition.kind.value}",
                    definition.operations,
                )
        for card_id, definitions in sorted(
            self._rulebook._listener_defs.items()
        ):
            for index, definition in enumerate(definitions):
                self._register_group(
                    card_id,
                    f"card:{card_id}/listener:{definition.event.value}:"
                    f"{definition.zone.value}:{index}",
                    definition.operations,
                )

    def _register_group(
        self,
        card_id: int,
        ability_id: str,
        operations: tuple[EffectOperation, ...],
    ) -> None:
        if not operations:
            return
        self._operation_groups[id(operations)] = ability_id
        self._register_operations(card_id, ability_id, operations, ())

    def _register_operations(
        self,
        card_id: int,
        ability_id: str,
        operations: tuple[EffectOperation, ...],
        parent_path: tuple[str, ...],
    ) -> None:
        group_id = "/".join((ability_id, *parent_path))
        self._operation_groups[id(operations)] = group_id
        self._structural_groups[card_id].append((group_id, operations))
        for index, operation in enumerate(operations):
            path = (*parent_path, f"operation:{index}")
            clause_id = f"{ability_id}/{'/'.join(path)}"
            self._operation_refs[id(operation)].append(clause_id)
            self._catalog[clause_id] = {
                "card_id": card_id,
                "ability_id": ability_id,
                "clause_id": clause_id,
                "operation_kind": operation.kind.value,
                "target_kind": operation.target.value,
                "condition_types": [
                    condition.type.value for condition in operation.conditions
                ],
            }
            for branch, nested in _nested_operation_groups(operation):
                self._register_operations(
                    card_id,
                    ability_id,
                    nested,
                    (*path, branch),
                )

    def reset(self) -> None:
        self.lifecycle: dict[tuple[int, str], int] = defaultdict(int)
        self.modes: dict[tuple[int, str], int] = defaultdict(int)
        self.clauses: dict[str, Counter[str]] = defaultdict(Counter)
        self.targets: dict[str, Counter[str]] = defaultdict(Counter)
        self.diagnostics: dict[str, Counter[str]] = {
            kind: Counter() for kind in _DIAGNOSTIC_KINDS
        }

    def set_context(self, **context: object) -> None:
        self.context = {
            key: value for key, value in context.items() if value is not None
        }

    def resolve_clause_id(
        self,
        card_id: int,
        operation: EffectOperation,
        operations: tuple[EffectOperation, ...],
    ) -> str:
        candidates = self._operation_refs.get(id(operation), ())
        group = self._operation_groups.get(id(operations))
        if group is None:
            matching_groups = sorted(
                [
                    (
                        candidate_group,
                        candidate_operations,
                    )
                    for candidate_group, candidate_operations in (
                        self._structural_groups.get(card_id, ())
                    )
                    if candidate_operations == operations
                ],
                key=lambda item: item[0],
            )
            if matching_groups:
                group = matching_groups[0][0]
        if group is not None:
            operation_indexes = [
                index
                for index, candidate in enumerate(operations)
                if candidate is operation
            ]
            if not operation_indexes:
                operation_indexes = [
                    index
                    for index, candidate in enumerate(operations)
                    if candidate == operation
                ]
            for index in operation_indexes:
                candidate = f"{group}/operation:{index}"
                if candidate in self._catalog:
                    return candidate
        for candidate in candidates:
            if group is None or candidate.startswith(f"{group}/"):
                return candidate
        if candidates:
            return sorted(candidates)[0]
        signature = json.dumps(
            {
                "kind": operation.kind.value,
                "target": operation.target.value,
                "card_id": operation.card_id,
                "amount": operation.amount,
                "secondary_amount": operation.secondary_amount,
                "index": next(
                    (
                        index
                        for index, candidate in enumerate(operations)
                        if candidate is operation
                    ),
                    -1,
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        clause_id = (
            f"card:{card_id}/dynamic:{_stable_component(signature)}"
        )
        self._catalog.setdefault(
            clause_id,
            {
                "card_id": card_id,
                "ability_id": f"card:{card_id}/dynamic",
                "clause_id": clause_id,
                "operation_kind": operation.kind.value,
                "target_kind": operation.target.value,
                "condition_types": [
                    condition.type.value for condition in operation.conditions
                ],
            },
        )
        return clause_id

    def record_event(self, event: GameEvent) -> None:
        card_id = _event_card_id(event)
        lifecycle = _LIFECYCLE_EVENTS.get(event.type)
        if lifecycle is not None and card_id is not None:
            self.lifecycle[(card_id, lifecycle)] += 1

        mode_id: str | None = None
        if event.type is EventType.CARD_PLAYED:
            value = event.metadata.get("mode_id")
            if isinstance(value, str) and value != "normal":
                mode_id = f"play:{_stable_component(value)}"
        elif event.type is EventType.CARD_FUSED:
            mode_id = "fusion"
        elif event.type is EventType.CARD_INVOKED:
            mode_id = "invocation"
        elif event.type is EventType.AMULET_ACTIVATED:
            mode_id = "activation"
        elif event.type is EventType.UNION_BURST_ACTIVATED:
            mode_id = (
                "union_burst:"
                f"{_stable_component(event.metadata.get('kind', 'unknown'))}"
            )
        elif event.type is EventType.MODE_SELECTED:
            mode_id = "choose"
        if mode_id is not None and card_id is not None:
            self.modes[(card_id, mode_id)] += 1

    def record_clause(
        self,
        clause_id: str,
        stage: str,
        *,
        amount: int = 1,
    ) -> None:
        self.clauses[clause_id][stage] += amount

    def record_target(
        self,
        clause_id: str,
        target: TargetKind | str,
        *,
        candidate_count: int | None = None,
        random: bool = False,
        no_target: bool = False,
    ) -> None:
        target_kind = target.value if isinstance(target, TargetKind) else target
        counter = self.targets[target_kind]
        counter["entered"] += 1
        if no_target:
            counter["no_target"] += 1
            self.clauses[clause_id]["no_target"] += 1
        if candidate_count is not None:
            counter["candidate_samples"] += 1
            counter["candidate_total"] += candidate_count
            minimum = counter.get("candidate_min")
            maximum = counter.get("candidate_max")
            counter["candidate_min"] = (
                candidate_count
                if minimum is None
                else min(minimum, candidate_count)
            )
            counter["candidate_max"] = (
                candidate_count
                if maximum is None
                else max(maximum, candidate_count)
            )
            if random:
                counter["random_candidate_samples"] += 1
                counter["random_candidate_total"] += candidate_count

    def record_capacity_shortage(self, clause_id: str, zone: str) -> None:
        self.clauses[clause_id]["capacity_shortage"] += 1
        self.targets[f"capacity:{zone}"]["capacity_shortage"] += 1

    def record_diagnostic(
        self,
        kind: str,
        *,
        card_id: int | None = None,
        clause_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        counter = self.diagnostics.setdefault(kind, Counter())
        counter["total"] += 1
        if card_id is not None:
            counter[f"card:{card_id}"] += 1
        if clause_id is not None:
            counter[f"clause:{clause_id}"] += 1
        if detail is not None:
            stable_detail = (
                detail
                if detail.isascii()
                and all(
                    character.isalnum()
                    or character in "._:/|=-"
                    for character in detail
                )
                else _stable_component(detail)
            )
            counter[f"detail:{stable_detail}"] += 1

    def to_session(
        self,
        *,
        card_ids: set[int] | frozenset[int] | None = None,
    ) -> dict[str, object]:
        catalog = [
            row
            for _, row in sorted(self._catalog.items())
            if card_ids is None or int(row["card_id"]) in card_ids
        ]
        clause_rows: list[dict[str, object]] = []
        for row in catalog:
            counters = self.clauses.get(str(row["clause_id"]), Counter())
            entered = int(counters["entered"])
            executed = int(counters["operation_executed"])
            status = (
                "not_triggered"
                if entered == 0
                else (
                    "triggered_passed"
                    if executed > 0
                    else "triggered_not_executed"
                )
            )
            clause_rows.append({
                **row,
                "status": status,
                "counts": dict(sorted(counters.items())),
            })
        return {
            "schema_version": self.schema_version,
            "context": dict(sorted(self.context.items())),
            "lifecycle": [
                {
                    "card_id": card_id,
                    "event": event,
                    "count": count,
                }
                for (card_id, event), count in sorted(self.lifecycle.items())
                if card_ids is None or card_id in card_ids
            ],
            "alternate_modes": [
                {
                    "card_id": card_id,
                    "mode_id": mode_id,
                    "count": count,
                }
                for (card_id, mode_id), count in sorted(self.modes.items())
                if card_ids is None or card_id in card_ids
            ],
            "clauses": clause_rows,
            "targets": [
                {
                    "target_kind": target_kind,
                    "counts": dict(sorted(counts.items())),
                }
                for target_kind, counts in sorted(self.targets.items())
            ],
            "diagnostics": {
                kind: dict(sorted(counts.items()))
                for kind, counts in sorted(self.diagnostics.items())
            },
        }


def aggregate_runtime_coverage(
    sessions: Iterable[dict[str, object]],
    *,
    deck_memberships: dict[int, list[str]] | None = None,
) -> dict[str, object]:
    session_list = list(sessions)
    by_card: dict[str, Counter[str]] = defaultdict(Counter)
    by_mechanic: dict[str, Counter[str]] = defaultdict(Counter)
    by_deck: dict[str, Counter[str]] = defaultdict(Counter)
    by_matchup: dict[str, Counter[str]] = defaultdict(Counter)
    status_counts: Counter[str] = Counter()
    global_clauses: dict[str, dict[str, object]] = {}
    matchup_clauses: dict[
        str,
        dict[str, dict[str, object]],
    ] = defaultdict(dict)
    status_rank = {
        "not_triggered": 0,
        "triggered_not_executed": 1,
        "triggered_passed": 2,
    }

    def merge_clause(
        destination: dict[str, dict[str, object]],
        clause: dict[str, object],
    ) -> None:
        clause_id = str(clause["clause_id"])
        previous = destination.get(clause_id)
        if (
            previous is None
            or status_rank[str(clause["status"])]
            > status_rank[str(previous["status"])]
        ):
            destination[clause_id] = clause

    for card_id, decks in sorted((deck_memberships or {}).items()):
        by_card[str(card_id)]
        for deck in decks:
            by_deck[deck]

    for session in session_list:
        context = session.get("context", {})
        matchup_id = str(context.get("matchup_id", "unassigned"))
        by_matchup[matchup_id]
        for lifecycle in session.get("lifecycle", []):
            card_id = int(lifecycle["card_id"])
            count = int(lifecycle["count"])
            event = str(lifecycle["event"])
            by_card[str(card_id)][f"lifecycle:{event}"] += count
            by_mechanic[f"lifecycle:{event}"]["events"] += count
            by_matchup[matchup_id][f"lifecycle:{event}"] += count
            for deck in (deck_memberships or {}).get(card_id, []):
                by_deck[deck][f"lifecycle:{event}"] += count
        for mode in session.get("alternate_modes", []):
            card_id = int(mode["card_id"])
            count = int(mode["count"])
            by_card[str(card_id)]["alternate_modes"] += count
            by_mechanic["alternate_mode"]["executed"] += count
            by_matchup[matchup_id]["alternate_modes"] += count
            for deck in (deck_memberships or {}).get(card_id, []):
                by_deck[deck]["alternate_modes"] += count
        for clause in session.get("clauses", []):
            merge_clause(global_clauses, clause)
            merge_clause(matchup_clauses[matchup_id], clause)

    for clause in global_clauses.values():
        card_id = int(clause["card_id"])
        status = str(clause["status"])
        operation_kind = str(clause["operation_kind"])
        status_counts[status] += 1
        by_card[str(card_id)][status] += 1
        by_mechanic[operation_kind][status] += 1
        for deck in (deck_memberships or {}).get(card_id, []):
            by_deck[deck][status] += 1

    for matchup_id, clauses in matchup_clauses.items():
        for clause in clauses.values():
            by_matchup[matchup_id][str(clause["status"])] += 1

    def rows(values: dict[str, Counter[str]]) -> list[dict[str, object]]:
        return [
            {"id": key, "counts": dict(sorted(counter.items()))}
            for key, counter in sorted(values.items())
        ]

    return {
        "schema_version": 1,
        "report_kind": "swb_runtime_coverage",
        "summary": {
            "session_count": len(session_list),
            "clause_status_counts": dict(sorted(status_counts.items())),
            "diagnostic_totals": {
                kind: sum(
                    int(session.get("diagnostics", {}).get(kind, {}).get("total", 0))
                    for session in session_list
                )
                for kind in _DIAGNOSTIC_KINDS
            },
        },
        "status_definitions": {
            "not_triggered": "能力从未进入运行时 clause。",
            "triggered_passed": "能力已进入并至少执行一次操作。",
            "triggered_not_executed": (
                "能力已进入，但因条件、目标、容量或安全分支未执行操作。"
            ),
        },
        "sessions": session_list,
        "aggregations": {
            "by_card": rows(by_card),
            "by_mechanic": rows(by_mechanic),
            "by_deck": rows(by_deck),
            "by_matchup": rows(by_matchup),
        },
    }


def render_runtime_coverage_markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    statuses = summary["clause_status_counts"]
    diagnostics = summary["diagnostic_totals"]
    lines = [
        "# Runtime Coverage",
        "",
        "本报告只读取结构化事件和稳定 card/clause ID，不解析中文运行日志。",
        "",
        "## Summary",
        "",
        f"- Sessions: {summary['session_count']}",
        f"- Not triggered: {statuses.get('not_triggered', 0)}",
        f"- Triggered and passed: {statuses.get('triggered_passed', 0)}",
        (
            "- Triggered but not executed: "
            f"{statuses.get('triggered_not_executed', 0)}"
        ),
        "",
        "## Diagnostics",
        "",
        "| Kind | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{kind}` | {count} |"
        for kind, count in sorted(diagnostics.items())
    )
    lines.extend([
        "",
        "## Aggregations",
        "",
        "JSON 产物包含 `by_card`、`by_mechanic`、`by_deck` 和 "
        "`by_matchup` 四个机器可读聚合。",
        "",
        "## Interpretation",
        "",
        "`not_triggered` 与 `triggered_passed` 是不同状态；前者不能作为"
        "能力通过的证据。1.12 将使用强制场景和随机对局继续消除未解释空白。",
        "",
    ])
    return "\n".join(lines)
