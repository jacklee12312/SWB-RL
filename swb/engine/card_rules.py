from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from swb.engine.abilities import RUNTIME_UNIT_KEYWORDS, normalize_keyword_name
from swb.engine.effects import (
    BoardFilter,
    ChooseOneOption,
    Condition,
    ConditionType,
    CostChangeMode,
    DeckFilter,
    EffectKind,
    EffectOperation,
    ExprType,
    ModifierDuration,
    TargetKind,
    ValueExpression,
)
from swb.engine.emblem import (
    EmblemDefinition,
    EmblemStacking,
    EmblemTriggerRule,
    EventScope,
    TurnScope,
)
from swb.engine.play_modes import (
    MAX_SPECIAL_MODES_PER_CARD,
    PlayModeDefinition,
    validate_play_mode_definition,
    validate_runtime_play_mode,
)

_VALID_CARD_TYPES = frozenset({"随从", "法术", "护符"})

_GRAVEYARD_EFFECT_KINDS = frozenset({
    EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND,
    EffectKind.SUMMON_FROM_GRAVEYARD,
    EffectKind.BANISH_FROM_GRAVEYARD,
})

_GRAVEYARD_TARGETS = frozenset({
    TargetKind.OWN_GRAVEYARD_CARD,
    TargetKind.RANDOM_OWN_GRAVEYARD_CARD,
    TargetKind.ALL_OWN_GRAVEYARD_CARDS,
})

_TARGET_DEPENDENT_CONDITIONS = frozenset({
    ConditionType.TARGET_ATTACK_AT_MOST,
    ConditionType.TARGET_ATTACK_AT_LEAST,
    ConditionType.TARGET_HEALTH_AT_MOST,
    ConditionType.TARGET_HEALTH_AT_LEAST,
    ConditionType.TARGET_HAS_KEYWORD,
})

_SOURCE_DEPENDENT_CONDITIONS = frozenset({
    ConditionType.SOURCE_EVOLVED,
    ConditionType.SOURCE_HAS_KEYWORD,
    ConditionType.SOURCE_FUSION_COUNT_AT_LEAST,
})

_UNSUPPORTED_PRESELECTED_TARGET_FIELDS = frozenset({
    "targets",
    "target_ids",
    "duplicate_targets",
})


def _reject_unsupported_preselected_target_fields(
    raw: dict,
    source_file: str,
    card_id: int,
) -> None:
    fields = sorted(set(raw) & _UNSUPPORTED_PRESELECTED_TARGET_FIELDS)
    if fields:
        raise ValueError(
            f"{source_file} card {card_id}: preselected multi-target payloads "
            f"are unsupported; fields {fields} cannot replace command-level "
            f"target selection"
        )


def _expression_depends_on_target(expression: ValueExpression) -> bool:
    return (
        expression.type in {ExprType.TARGET_ATTACK, ExprType.TARGET_HEALTH}
        or any(_expression_depends_on_target(value) for value in expression.values)
    )


def _check_target_conditions(conditions: tuple[Condition, ...], source: str) -> set[str]:
    invalid: set[str] = set()
    for cond in conditions:
        if cond.type in _TARGET_DEPENDENT_CONDITIONS:
            invalid.add(cond.type.value)
        invalid.update(_check_target_conditions(cond.conditions, source))
    return invalid


def _check_source_conditions(conditions: tuple[Condition, ...]) -> set[str]:
    invalid: set[str] = set()
    for condition in conditions:
        if condition.type in _SOURCE_DEPENDENT_CONDITIONS:
            invalid.add(condition.type.value)
        invalid.update(_check_source_conditions(tuple(condition.conditions)))
    return invalid


def _parse_board_filter(
    raw: dict,
    *,
    source_path: str,
    card_id: int,
    prefix: str = "",
    allow_evolved: bool = False,
) -> BoardFilter | None:
    card_type_key = f"{prefix}card_type_filter"
    cost_min_key = f"{prefix}cost_min"
    cost_max_key = f"{prefix}cost_max"
    card_id_key = f"{prefix}card_id_filter"
    card_name_key = f"{prefix}card_name_filter"
    evolved_key = f"{prefix}evolved_filter"

    card_type = raw.get(card_type_key)
    if card_type is not None:
        if not isinstance(card_type, str):
            raise ValueError(
                f"{source_path}/{card_type_key} card {card_id}: must be a string"
            )
        if card_type not in _VALID_CARD_TYPES:
            raise ValueError(
                f"{source_path}/{card_type_key} card {card_id}: "
                f"unknown card type {card_type!r}; valid: {sorted(_VALID_CARD_TYPES)}"
            )

    cost_min = raw.get(cost_min_key)
    if cost_min is not None:
        cost_min = _parse_non_negative_int(cost_min, f"{source_path}/{cost_min_key}", card_id)
    cost_max = raw.get(cost_max_key)
    if cost_max is not None:
        cost_max = _parse_non_negative_int(cost_max, f"{source_path}/{cost_max_key}", card_id)
    if cost_min is not None and cost_max is not None and cost_min > cost_max:
        raise ValueError(
            f"{source_path} card {card_id}: {cost_min_key} ({cost_min}) "
            f"must not exceed {cost_max_key} ({cost_max})"
        )

    filter_card_id = raw.get(card_id_key)
    if filter_card_id is not None:
        filter_card_id = _parse_non_negative_int(
            filter_card_id,
            f"{source_path}/{card_id_key}",
            card_id,
        )

    card_name = raw.get(card_name_key)
    if card_name is not None and not isinstance(card_name, str):
        raise ValueError(
            f"{source_path}/{card_name_key} card {card_id}: must be a string"
        )

    evolved = raw.get(evolved_key)
    if evolved is not None:
        if not allow_evolved:
            raise ValueError(
                f"{source_path}/{evolved_key} card {card_id}: evolved filter is not valid here"
            )
        if not isinstance(evolved, bool):
            raise ValueError(
                f"{source_path}/{evolved_key} card {card_id}: must be boolean"
            )

    if not any(v is not None for v in (card_type, cost_min, cost_max, filter_card_id, card_name, evolved)):
        return None
    return BoardFilter(
        card_type=card_type,
        cost_min=cost_min,
        cost_max=cost_max,
        card_id=filter_card_id,
        card_name=card_name,
        evolved=evolved,
    )


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
    INVOKE = "invoke"
    ACTIVATE = "activate"


@dataclass(frozen=True)
class CardRule:
    card_id: int
    trigger: Trigger
    operations: tuple[EffectOperation, ...]
    countdown: int | None = None


@dataclass(frozen=True)
class CardPassive:
    card_id: int
    kind: str
    amount: int


@dataclass(frozen=True)
class FusionDefinition:
    card_id: int
    material_filter: DeckFilter
    min_materials: int = 1
    max_materials: int | None = None


@dataclass(frozen=True)
class InvocationDefinition:
    card_id: int
    trigger: Trigger
    conditions: tuple[Condition, ...] = ()


@dataclass(frozen=True)
class ActivationDefinition:
    card_id: int
    cost: int = 0


class RuleBook:
    def __init__(
        self,
        rules: tuple[CardRule, ...] = (),
        passives: tuple[CardPassive, ...] = (),
        play_modes: dict[int, tuple[PlayModeDefinition, ...]] | None = None,
        emblem_defs: dict[str, EmblemDefinition] | None = None,
        fusion_defs: dict[int, FusionDefinition] | None = None,
        invocation_defs: dict[int, InvocationDefinition] | None = None,
        activation_defs: dict[int, ActivationDefinition] | None = None,
    ):
        self._rules: dict[tuple[int, Trigger], tuple[EffectOperation, ...]] = {}
        for rule in rules:
            key = (rule.card_id, rule.trigger)
            if key in self._rules:
                raise ValueError(
                    f"duplicate rule for card {rule.card_id} trigger "
                    f"{rule.trigger.value!r}"
                )
            self._rules[key] = rule.operations
        self._countdowns = {
            rule.card_id: rule.countdown
            for rule in rules
            if rule.countdown is not None
        }
        self._passives: dict[int, list[CardPassive]] = {}
        for p in passives:
            self._passives.setdefault(p.card_id, []).append(p)
        self._play_modes: dict[int, tuple[PlayModeDefinition, ...]] = play_modes or {}
        for card_id, modes in self._play_modes.items():
            if len(modes) > MAX_SPECIAL_MODES_PER_CARD:
                raise ValueError(
                    f"card {card_id}: has {len(modes)} play modes, "
                    f"maximum is {MAX_SPECIAL_MODES_PER_CARD}"
                )
            seen: set[str] = set()
            for m in modes:
                validate_runtime_play_mode(
                    m, f"card {card_id}/play_modes/{m.mode_id}"
                )
                if m.mode_id in seen:
                    raise ValueError(
                        f"card {card_id}: duplicate play mode id {m.mode_id!r}"
                    )
                seen.add(m.mode_id)
        self._emblem_defs: dict[str, EmblemDefinition] = emblem_defs or {}
        self._fusion_defs: dict[int, FusionDefinition] = fusion_defs or {}
        self._invocation_defs: dict[int, InvocationDefinition] = invocation_defs or {}
        self._activation_defs: dict[int, ActivationDefinition] = activation_defs or {}
        for card_id in self._fusion_defs:
            if len(self._play_modes.get(card_id, ())) + 1 > MAX_SPECIAL_MODES_PER_CARD:
                raise ValueError(
                    f"card {card_id}: fusion plus play modes exceeds "
                    f"{MAX_SPECIAL_MODES_PER_CARD} special actions"
                )
        for card_id in self._activation_defs:
            if not self.operations_for(card_id, Trigger.ACTIVATE):
                raise ValueError(
                    f"card {card_id}: activation definition requires a non-empty "
                    f"{Trigger.ACTIVATE.value!r} rule"
                )
        for card_id, trigger in self._rules:
            if trigger is Trigger.ACTIVATE and card_id not in self._activation_defs:
                raise ValueError(
                    f"card {card_id}: {Trigger.ACTIVATE.value!r} rule requires an "
                    "activation definition"
                )

    def operations_for(
        self, card_id: int, trigger: Trigger
    ) -> tuple[EffectOperation, ...]:
        return self._rules.get((card_id, trigger), ())

    def countdown_for(self, card_id: int) -> int | None:
        return self._countdowns.get(card_id)

    def modes_for(self, card_id: int) -> tuple[PlayModeDefinition, ...]:
        return self._play_modes.get(card_id, ())

    def emblem_def(self, emblem_id: str) -> EmblemDefinition | None:
        return self._emblem_defs.get(emblem_id)

    def fusion_for(self, card_id: int) -> FusionDefinition | None:
        return self._fusion_defs.get(card_id)

    def invocation_for(self, card_id: int) -> InvocationDefinition | None:
        return self._invocation_defs.get(card_id)

    def activation_for(self, card_id: int) -> ActivationDefinition | None:
        return self._activation_defs.get(card_id)

    def emblem_trigger_ops_for(self, emblem_id: str, trigger: str) -> tuple[EffectOperation, ...]:
        from swb.engine.emblem import EmblemTriggerRule
        ed = self._emblem_defs.get(emblem_id)
        if ed is None:
            return ()
        for tr in ed.triggers:
            if tr.trigger == trigger:
                return tr.operations
        return ()

    def spellboost_cost_reduction(self, card_id: int) -> int:
        for p in self._passives.get(card_id, []):
            if p.kind == "spellboost_cost_reduction":
                return p.amount
        return 0

    def cannot_be_played(self, card_id: int) -> bool:
        return any(
            p.kind == "cannot_be_played"
            for p in self._passives.get(card_id, [])
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> "RuleBook":
        path = Path(directory)
        if not path.exists():
            return cls()
        rules: list[CardRule] = []
        seen_rules: dict[tuple[int, Trigger], str] = {}
        passives: list[tuple[CardPassive, str]] = []
        all_play_modes: dict[int, list[PlayModeDefinition]] = {}
        all_emblem_defs: dict[str, EmblemDefinition] = {}
        all_fusion_defs: dict[int, FusionDefinition] = {}
        all_invocation_defs: dict[int, InvocationDefinition] = {}
        all_activation_defs: dict[int, ActivationDefinition] = {}
        for file_path in sorted(path.glob("*.json")):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                entries = payload
                raw_passives = []
                raw_fusions = []
                raw_invocations = []
                raw_activations = []
            else:
                entries = payload.get("rules", [])
                raw_passives = payload.get("passives", [])
                raw_fusions = payload.get("fusions", [])
                raw_invocations = payload.get("invocations", [])
                raw_activations = payload.get("activations", [])
                if not isinstance(raw_fusions, list):
                    raise ValueError(f"{file_path.name}: 'fusions' must be a list")
                if not isinstance(raw_invocations, list):
                    raise ValueError(
                        f"{file_path.name}: 'invocations' must be a list"
                    )
                if not isinstance(raw_activations, list):
                    raise ValueError(
                        f"{file_path.name}: 'activations' must be a list"
                    )
                raw_emblems = payload.get("emblems")
                if raw_emblems is not None:
                    if not isinstance(raw_emblems, list):
                        raise ValueError(
                            f"{file_path.name}: 'emblems' must be a list"
                        )
                    for index, raw_emblem in enumerate(raw_emblems):
                        ed = _parse_emblem_definition(
                            raw_emblem,
                            f"{file_path.name}/emblems[{index}]",
                            _parse_operation,
                        )
                        if ed.emblem_id in all_emblem_defs:
                            raise ValueError(
                                f"{file_path.name}/emblems[{index}]: "
                                f"duplicate emblem id {ed.emblem_id!r}"
                            )
                        all_emblem_defs[ed.emblem_id] = ed
            for entry in entries:
                operations = tuple(
                    _parse_operation(
                        operation,
                        f"{file_path.name}/operations[{index}]",
                        entry["card_id"],
                    )
                    for index, operation in enumerate(
                        entry.get("operations", [])
                    )
                )
                _validate_target_keys(operations, f"{file_path.name} card {entry['card_id']}")
                card_id = int(entry["card_id"])
                trigger = Trigger(entry["trigger"])
                source = f"{file_path.name} card {card_id} trigger {trigger.value!r}"
                key = (card_id, trigger)
                if key in seen_rules:
                    raise ValueError(
                        f"{source}: duplicate rule; first defined at "
                        f"{seen_rules[key]}"
                    )
                seen_rules[key] = source
                rules.append(
                    CardRule(
                        card_id=card_id,
                        trigger=trigger,
                        operations=operations,
                        countdown=entry.get("countdown"),
                    )
                )
                raw_modes = entry.get("play_modes")
                if raw_modes is not None:
                    if not isinstance(raw_modes, list):
                        raise ValueError(
                            f"{file_path.name} card {entry['card_id']}: "
                            f"'play_modes' must be a list"
                        )
                    if len(raw_modes) > MAX_SPECIAL_MODES_PER_CARD:
                        raise ValueError(
                            f"{file_path.name} card {entry['card_id']}: "
                            f"play_modes has {len(raw_modes)} entries, "
                            f"maximum is {MAX_SPECIAL_MODES_PER_CARD}"
                        )
                    mode_ids_seen: set[str] = set()
                    for index, raw_mode in enumerate(raw_modes):
                        mode_def = validate_play_mode_definition(
                            raw_mode,
                            file_path.name,
                            entry["card_id"],
                            _parse_operation,
                        )
                        if mode_def.mode_id in mode_ids_seen:
                            raise ValueError(
                                f"{file_path.name} card {entry['card_id']}: "
                                f"duplicate play mode id {mode_def.mode_id!r}"
                            )
                        mode_ids_seen.add(mode_def.mode_id)
                        _validate_target_keys(
                            mode_def.operations,
                            f"{file_path.name} card {entry['card_id']}/play_modes/{mode_def.mode_id}",
                        )
                        card_modes = all_play_modes.setdefault(int(entry["card_id"]), [])
                        if len(card_modes) >= MAX_SPECIAL_MODES_PER_CARD:
                            raise ValueError(
                                f"{file_path.name} card {entry['card_id']}: "
                                f"total play_modes across entries exceeds "
                                f"maximum of {MAX_SPECIAL_MODES_PER_CARD}"
                            )
                        card_modes.append(mode_def)
            for index, raw_passive in enumerate(raw_passives):
                source_path = f"{file_path.name}/passives[{index}]"
                passives.append(
                    (_parse_passive(raw_passive, source_path), source_path)
                )
            for index, raw_fusion in enumerate(raw_fusions):
                source_path = f"{file_path.name}/fusions[{index}]"
                fusion = _parse_fusion_definition(raw_fusion, source_path)
                if fusion.card_id in all_fusion_defs:
                    raise ValueError(
                        f"{source_path}: duplicate fusion definition for card "
                        f"{fusion.card_id}"
                    )
                all_fusion_defs[fusion.card_id] = fusion
            for index, raw_invocation in enumerate(raw_invocations):
                source_path = f"{file_path.name}/invocations[{index}]"
                invocation = _parse_invocation_definition(
                    raw_invocation,
                    source_path,
                )
                if invocation.card_id in all_invocation_defs:
                    raise ValueError(
                        f"{source_path}: duplicate invocation definition for "
                        f"card {invocation.card_id}"
                    )
                all_invocation_defs[invocation.card_id] = invocation
            for index, raw_activation in enumerate(raw_activations):
                source_path = f"{file_path.name}/activations[{index}]"
                activation = _parse_activation_definition(
                    raw_activation,
                    source_path,
                )
                if activation.card_id in all_activation_defs:
                    raise ValueError(
                        f"{source_path}: duplicate activation definition for "
                        f"card {activation.card_id}"
                    )
                all_activation_defs[activation.card_id] = activation
        _validate_passives(passives)
        frozen_modes = {
            cid: tuple(modes) for cid, modes in all_play_modes.items()
        }
        _validate_emblem_references(rules, frozen_modes, all_emblem_defs)
        return cls(
            rules=tuple(rules),
            passives=tuple(passive for passive, _ in passives),
            play_modes=frozen_modes,
            emblem_defs=all_emblem_defs,
            fusion_defs=all_fusion_defs,
            invocation_defs=all_invocation_defs,
            activation_defs=all_activation_defs,
        )


def _parse_emblem_definition(raw: dict, source_file: str, ops_parser) -> EmblemDefinition:
    error_prefix = source_file
    if not isinstance(raw, dict):
        raise ValueError(f"{error_prefix}: emblem definition must be an object")
    unknown_keys = set(raw) - {
        "id", "source_card_id", "stacking", "countdown", "triggers", "on_expire",
    }
    if unknown_keys:
        raise ValueError(
            f"{error_prefix}: unknown fields {sorted(unknown_keys)}"
        )
    emblem_id = raw.get("id")
    if not isinstance(emblem_id, str) or not emblem_id:
        raise ValueError(f"{error_prefix}: emblem 'id' must be a non-empty string")
    source_card_id = raw.get("source_card_id")
    if source_card_id is None:
        raise ValueError(f"{error_prefix}/source_card_id: required")
    if not isinstance(source_card_id, int) or isinstance(source_card_id, bool):
        raise ValueError(f"{error_prefix}/source_card_id: must be an integer")
    if source_card_id <= 0:
        raise ValueError(f"{error_prefix}/source_card_id: must be positive")
    stacking_raw = raw.get("stacking", "allow")
    try:
        stacking = EmblemStacking(stacking_raw)
    except ValueError:
        raise ValueError(
            f"{error_prefix}/stacking: must be one of {sorted(s.value for s in EmblemStacking)}, "
            f"got {stacking_raw!r}"
        )
    countdown = raw.get("countdown")
    if countdown is not None:
        if isinstance(countdown, bool):
            raise ValueError(f"{error_prefix}/countdown: must be an integer, got bool")
        if not isinstance(countdown, int) or countdown < 0:
            raise ValueError(f"{error_prefix}/countdown: must be a non-negative integer")

    raw_triggers = raw.get("triggers", [])
    if not isinstance(raw_triggers, list):
        raise ValueError(f"{error_prefix}: 'triggers' must be a list")
    triggers: list[EmblemTriggerRule] = []
    _VALID_EMBLEM_TRIGGERS = frozenset({
        "turn_start", "turn_end", "follower_summoned",
        "follower_evolved", "follower_destroyed", "amulet_destroyed",
        "card_played", "leader_healed", "death_batch_end",
        "amulet_activated",
    })
    for i, rt in enumerate(raw_triggers):
        t_source = f"{error_prefix}/triggers[{i}]"
        if not isinstance(rt, dict):
            raise ValueError(f"{t_source}: trigger must be an object")
        unknown_trigger_keys = set(rt) - {
            "trigger", "operations", "conditions",
            "turn_scope", "event_scope", "once_per_turn", "max_activations",
        }
        if unknown_trigger_keys:
            raise ValueError(
                f"{t_source}: unknown fields {sorted(unknown_trigger_keys)}"
            )
        trigger_name = rt.get("trigger")
        if trigger_name not in _VALID_EMBLEM_TRIGGERS:
            raise ValueError(
                f"{t_source}/trigger: must be one of {sorted(_VALID_EMBLEM_TRIGGERS)}, "
                f"got {trigger_name!r}"
            )

        turn_scope = None
        if "turn_scope" in rt:
            turn_scope_raw = rt["turn_scope"]
            try:
                turn_scope = TurnScope(turn_scope_raw)
            except ValueError:
                raise ValueError(
                    f"{t_source}/turn_scope: must be one of {sorted(s.value for s in TurnScope)}, "
                    f"got {turn_scope_raw!r}"
                )

        event_scope = None
        if "event_scope" in rt:
            event_scope_raw = rt["event_scope"]
            try:
                event_scope = EventScope(event_scope_raw)
            except ValueError:
                raise ValueError(
                    f"{t_source}/event_scope: must be one of {sorted(s.value for s in EventScope)}, "
                    f"got {event_scope_raw!r}"
                )

        once_per_turn = rt.get("once_per_turn", False)
        if not isinstance(once_per_turn, bool):
            raise ValueError(f"{t_source}/once_per_turn: must be a boolean, got {type(once_per_turn).__name__}")

        max_activations = rt.get("max_activations")
        if max_activations is not None:
            if isinstance(max_activations, bool):
                raise ValueError(f"{t_source}/max_activations: must be a positive integer, got bool")
            if not isinstance(max_activations, int) or max_activations <= 0:
                raise ValueError(
                    f"{t_source}/max_activations: must be a positive integer, got {max_activations!r}"
                )

        raw_ops = rt.get("operations", [])
        if not isinstance(raw_ops, list):
            raise ValueError(f"{t_source}: 'operations' must be a list")
        operations = tuple(
            ops_parser(op, f"{t_source}/operations[{idx}]", source_card_id)
            for idx, op in enumerate(raw_ops)
        )
        _validate_target_keys(operations, t_source)
        conditions: tuple = ()
        raw_conds = rt.get("conditions")
        if raw_conds is not None:
            if not isinstance(raw_conds, list):
                raise ValueError(f"{t_source}: 'conditions' must be a list")
            conditions = tuple(
                _parse_condition(c, f"{t_source}/conditions[{j}]", source_card_id)
                for j, c in enumerate(raw_conds)
            )
        invalid_conditions = _check_target_conditions(conditions, t_source)
        if invalid_conditions:
            raise ValueError(
                f"{t_source}/conditions: emblem trigger conditions cannot "
                f"depend on a selected target: {sorted(invalid_conditions)}"
            )
        triggers.append(EmblemTriggerRule(
            trigger=trigger_name,
            operations=operations,
            conditions=conditions,
            turn_scope=turn_scope,
            event_scope=event_scope,
            once_per_turn=once_per_turn,
            max_activations=max_activations,
        ))

    on_expire: tuple = ()
    raw_on_expire = raw.get("on_expire")
    if raw_on_expire is not None:
        if not isinstance(raw_on_expire, list):
            raise ValueError(f"{error_prefix}/on_expire: must be a list")
        on_expire = tuple(
            ops_parser(op, f"{error_prefix}/on_expire[{idx}]", source_card_id)
            for idx, op in enumerate(raw_on_expire)
        )
        _validate_target_keys(on_expire, f"{error_prefix}/on_expire")

    return EmblemDefinition(
        emblem_id=emblem_id,
        source_card_id=source_card_id,
        stacking=stacking,
        countdown=countdown,
        triggers=tuple(triggers),
        on_expire=on_expire,
    )


def _iter_nested_operations(
    operations: tuple[EffectOperation, ...],
):
    for operation in operations:
        yield operation
        if operation.necromancy_operations:
            yield from _iter_nested_operations(operation.necromancy_operations)
        if operation.earth_rite_operations:
            yield from _iter_nested_operations(operation.earth_rite_operations)
        if operation.then_operations:
            yield from _iter_nested_operations(operation.then_operations)
        if operation.else_operations:
            yield from _iter_nested_operations(operation.else_operations)
        if operation.optional_operations:
            yield from _iter_nested_operations(operation.optional_operations)
        for option in operation.choose_one_options:
            yield from _iter_nested_operations(option.operations)


def _validate_emblem_references(
    rules: list[CardRule],
    play_modes: dict[int, tuple[PlayModeDefinition, ...]],
    emblem_defs: dict[str, EmblemDefinition],
) -> None:
    references: list[tuple[int, str]] = []

    def collect(card_id: int, operations: tuple[EffectOperation, ...]) -> None:
        for operation in _iter_nested_operations(operations):
            if operation.kind in {
                EffectKind.GAIN_EMBLEM,
                EffectKind.ADD_EMBLEM,
                EffectKind.REMOVE_EMBLEM,
            }:
                references.append((card_id, operation.emblem_id or ""))

    for rule in rules:
        collect(rule.card_id, rule.operations)
    for card_id, modes in play_modes.items():
        for mode in modes:
            collect(card_id, mode.operations)
    for definition in emblem_defs.values():
        for trigger in definition.triggers:
            collect(definition.source_card_id, trigger.operations)
        collect(definition.source_card_id, definition.on_expire)

    for card_id, emblem_id in references:
        if emblem_id not in emblem_defs:
            raise ValueError(
                f"card {card_id}: unknown emblem_id {emblem_id!r}"
            )


def _validate_passives(passives: list[tuple[CardPassive, str]]) -> None:
    seen: dict[tuple[int, str], str] = {}
    for passive, source_path in passives:
        p = passive
        key = (p.card_id, p.kind)
        if key in seen:
            raise ValueError(
                f"{source_path}: duplicate passive {p.kind!r} for card "
                f"{p.card_id}; first defined at {seen[key]}"
            )
        seen[key] = source_path


def _parse_fusion_definition(raw: dict, source_path: str) -> FusionDefinition:
    if not isinstance(raw, dict):
        raise ValueError(f"{source_path}: fusion definition must be an object")
    unknown = set(raw) - {
        "card_id", "material_filter", "min_materials", "max_materials",
    }
    if unknown:
        raise ValueError(f"{source_path}: unknown fields {sorted(unknown)}")
    card_id = raw.get("card_id")
    if isinstance(card_id, bool) or not isinstance(card_id, int) or card_id <= 0:
        raise ValueError(f"{source_path}/card_id: must be a positive integer")

    raw_filter = raw.get("material_filter")
    if not isinstance(raw_filter, dict):
        raise ValueError(f"{source_path}/material_filter: must be an object")
    filter_unknown = set(raw_filter) - {
        "card_type", "class_id", "class_name", "cost_min", "cost_max",
        "card_id", "card_name",
    }
    if filter_unknown:
        raise ValueError(
            f"{source_path}/material_filter: unknown fields "
            f"{sorted(filter_unknown)}"
        )
    card_type = raw_filter.get("card_type")
    if card_type is not None and card_type not in _VALID_CARD_TYPES:
        raise ValueError(
            f"{source_path}/material_filter/card_type: must be one of "
            f"{sorted(_VALID_CARD_TYPES)}"
        )
    class_id = raw_filter.get("class_id")
    if class_id is not None:
        class_id = _parse_non_negative_int(
            class_id, f"{source_path}/material_filter/class_id", card_id
        )
    class_name = raw_filter.get("class_name")
    if class_name is not None and (not isinstance(class_name, str) or not class_name):
        raise ValueError(
            f"{source_path}/material_filter/class_name: must be a non-empty string"
        )
    cost_min = raw_filter.get("cost_min")
    if cost_min is not None:
        cost_min = _parse_non_negative_int(
            cost_min, f"{source_path}/material_filter/cost_min", card_id
        )
    cost_max = raw_filter.get("cost_max")
    if cost_max is not None:
        cost_max = _parse_non_negative_int(
            cost_max, f"{source_path}/material_filter/cost_max", card_id
        )
    if cost_min is not None and cost_max is not None and cost_min > cost_max:
        raise ValueError(
            f"{source_path}/material_filter: cost_min must not exceed cost_max"
        )
    filter_card_id = raw_filter.get("card_id")
    if filter_card_id is not None:
        filter_card_id = _parse_non_negative_int(
            filter_card_id, f"{source_path}/material_filter/card_id", card_id
        )
    card_name = raw_filter.get("card_name")
    if card_name is not None and (not isinstance(card_name, str) or not card_name):
        raise ValueError(
            f"{source_path}/material_filter/card_name: must be a non-empty string"
        )

    min_materials = raw.get("min_materials", 1)
    min_materials = _parse_non_negative_int(
        min_materials, f"{source_path}/min_materials", card_id
    )
    if min_materials < 1:
        raise ValueError(f"{source_path}/min_materials: must be positive")
    max_materials = raw.get("max_materials")
    if max_materials is not None:
        max_materials = _parse_non_negative_int(
            max_materials, f"{source_path}/max_materials", card_id
        )
        if max_materials < min_materials:
            raise ValueError(
                f"{source_path}/max_materials: must be at least min_materials"
            )

    return FusionDefinition(
        card_id=card_id,
        material_filter=DeckFilter(
            card_type=card_type,
            class_id=class_id,
            class_name=class_name,
            cost_min=cost_min,
            cost_max=cost_max,
            card_id=filter_card_id,
            card_name=card_name,
        ),
        min_materials=min_materials,
        max_materials=max_materials,
    )


def _parse_invocation_definition(
    raw: dict,
    source_path: str,
) -> InvocationDefinition:
    if not isinstance(raw, dict):
        raise ValueError(f"{source_path}: invocation definition must be an object")
    unknown = set(raw) - {"card_id", "trigger", "conditions"}
    if unknown:
        raise ValueError(f"{source_path}: unknown fields {sorted(unknown)}")

    card_id = raw.get("card_id")
    if isinstance(card_id, bool) or not isinstance(card_id, int) or card_id <= 0:
        raise ValueError(f"{source_path}/card_id: must be a positive integer")

    trigger_raw = raw.get("trigger", Trigger.TURN_START.value)
    try:
        trigger = Trigger(trigger_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_path}/trigger: invalid trigger {trigger_raw!r}"
        ) from exc
    if trigger is not Trigger.TURN_START:
        raise ValueError(
            f"{source_path}/trigger: invocation currently requires "
            f"{Trigger.TURN_START.value!r}"
        )

    raw_conditions = raw.get("conditions", [])
    if not isinstance(raw_conditions, list):
        raise ValueError(f"{source_path}/conditions: must be a list")
    conditions = tuple(
        _parse_condition(
            condition,
            f"{source_path}/conditions[{index}]",
            card_id,
        )
        for index, condition in enumerate(raw_conditions)
    )
    invalid = _check_target_conditions(conditions, source_path)
    invalid.update(_check_source_conditions(conditions))
    if invalid:
        raise ValueError(
            f"{source_path}/conditions: deck invocation conditions cannot "
            f"depend on a source or selected target: {sorted(invalid)}"
        )
    return InvocationDefinition(
        card_id=card_id,
        trigger=trigger,
        conditions=conditions,
    )


def _parse_activation_definition(
    raw: dict,
    source_path: str,
) -> ActivationDefinition:
    if not isinstance(raw, dict):
        raise ValueError(f"{source_path}: activation definition must be an object")
    unknown = set(raw) - {"card_id", "cost"}
    if unknown:
        raise ValueError(f"{source_path}: unknown fields {sorted(unknown)}")

    card_id = raw.get("card_id")
    if isinstance(card_id, bool) or not isinstance(card_id, int) or card_id <= 0:
        raise ValueError(f"{source_path}/card_id: must be a positive integer")
    cost = _parse_non_negative_int(
        raw.get("cost", 0),
        f"{source_path}/cost",
        card_id,
    )
    return ActivationDefinition(card_id=card_id, cost=cost)


def _parse_passive(raw: dict, source_file: str) -> CardPassive:
    if not isinstance(raw, dict):
        raise ValueError(f"{source_file}: passive must be an object")
    card_id = raw.get("card_id")
    if card_id is None:
        raise ValueError(f"{source_file}: passive requires card_id")
    if not isinstance(card_id, int) or isinstance(card_id, bool) or card_id <= 0:
        raise ValueError(
            f"{source_file}/card_id: must be a positive integer, got {card_id!r}"
        )
    kind = raw.get("kind")
    if kind not in ("spellboost_cost_reduction", "cannot_be_played"):
        raise ValueError(f"{source_file} card {card_id}: unknown passive kind {kind!r}")
    if kind == "cannot_be_played":
        amount = raw.get("amount")
        if amount is None:
            return CardPassive(card_id=int(card_id), kind=kind, amount=0)
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise ValueError(
                f"{source_file} card {card_id}/amount: must be 0 or omitted for {kind!r}"
            )
        if amount != 0:
            raise ValueError(
                f"{source_file} card {card_id}/amount: must be 0 or omitted for {kind!r}"
            )
        return CardPassive(card_id=int(card_id), kind=kind, amount=0)
    amount = raw.get("amount")
    if amount is None:
        raise ValueError(f"{source_file} card {card_id}/amount: required for {kind!r}")
    if isinstance(amount, bool):
        raise ValueError(
            f"{source_file} card {card_id}/amount: must be an integer, got bool"
        )
    if not isinstance(amount, int):
        raise ValueError(
            f"{source_file} card {card_id}/amount: must be an integer, got {type(amount).__name__}"
        )
    if amount < 0:
        raise ValueError(
            f"{source_file} card {card_id}/amount: must be non-negative, got {amount}"
        )
    return CardPassive(card_id=int(card_id), kind=kind, amount=amount)


_BINDABLE_TARGETS = frozenset({
    TargetKind.OWN_UNIT,
    TargetKind.ENEMY_UNIT,
    TargetKind.ANY_UNIT,
    TargetKind.OWN_AMULET,
    TargetKind.ENEMY_AMULET,
    TargetKind.ANY_AMULET,
    TargetKind.OWN_BOARD,
    TargetKind.ENEMY_BOARD,
    TargetKind.ANY_BOARD,
    TargetKind.RANDOM_OWN_UNIT,
    TargetKind.RANDOM_ENEMY_UNIT,
    TargetKind.RANDOM_OWN_BOARD,
    TargetKind.RANDOM_ENEMY_BOARD,
})

_REQUIRES_TARGET_TARGETS = frozenset({
    TargetKind.OWN_UNIT,
    TargetKind.ENEMY_UNIT,
    TargetKind.ANY_UNIT,
    TargetKind.OWN_AMULET,
    TargetKind.ENEMY_AMULET,
    TargetKind.ANY_AMULET,
    TargetKind.OWN_BOARD,
    TargetKind.ENEMY_BOARD,
    TargetKind.ANY_BOARD,
    TargetKind.OWN_HAND,
    TargetKind.OWN_GRAVEYARD_CARD,
    TargetKind.RANDOM_OWN_UNIT,
    TargetKind.RANDOM_ENEMY_UNIT,
    TargetKind.RANDOM_OWN_BOARD,
    TargetKind.RANDOM_ENEMY_BOARD,
    TargetKind.RANDOM_OWN_HAND,
    TargetKind.RANDOM_OWN_GRAVEYARD_CARD,
    TargetKind.ALL_OWN_UNITS,
    TargetKind.ALL_ENEMY_UNITS,
    TargetKind.ALL_UNITS,
    TargetKind.ALL_OWN_BOARD,
    TargetKind.ALL_ENEMY_BOARD,
    TargetKind.ALL_BOARD,
    TargetKind.ALL_OWN_AMULETS,
    TargetKind.ALL_ENEMY_AMULETS,
    TargetKind.ALL_OWN_HAND,
    TargetKind.ALL_OWN_GRAVEYARD_CARDS,
})

_TARGET_EXISTS_TARGETS = _REQUIRES_TARGET_TARGETS | frozenset({
    TargetKind.OWN_UNIT_OR_LEADER,
    TargetKind.ENEMY_UNIT_OR_LEADER,
    TargetKind.ANY_UNIT_OR_LEADER,
})

_MULTI_TARGET_TARGETS = frozenset({
    TargetKind.OWN_UNIT,
    TargetKind.ENEMY_UNIT,
    TargetKind.ANY_UNIT,
    TargetKind.OWN_UNIT_OR_LEADER,
    TargetKind.ENEMY_UNIT_OR_LEADER,
    TargetKind.ANY_UNIT_OR_LEADER,
    TargetKind.OWN_AMULET,
    TargetKind.ENEMY_AMULET,
    TargetKind.ANY_AMULET,
    TargetKind.OWN_BOARD,
    TargetKind.ENEMY_BOARD,
    TargetKind.ANY_BOARD,
    TargetKind.OWN_HAND,
    TargetKind.OWN_GRAVEYARD_CARD,
})


def _validate_target_keys(operations: tuple[EffectOperation, ...], source: str) -> None:
    defined: set[str] = set()
    for i, op in enumerate(operations):
        if op.target is TargetKind.PREVIOUS_TARGET:
            if not op.target_key:
                raise ValueError(
                    f"{source}/operations[{i}]: PREVIOUS_TARGET requires target_key"
                )
            if op.target_key not in defined:
                raise ValueError(
                    f"{source}/operations[{i}]: target_key "
                    f"{op.target_key!r} was not defined by a previous operation"
                )
        elif op.target_key:
            if op.target not in _BINDABLE_TARGETS:
                raise ValueError(
                    f"{source}/operations[{i}]: target {op.target.value!r} "
                    f"cannot define target_key {op.target_key!r}; "
                    f"target_key requires selected board-entity targets"
                )
            if op.target_key in defined:
                raise ValueError(
                    f"{source}/operations[{i}]: duplicate target_key "
                    f"{op.target_key!r}"
                )
            defined.add(op.target_key)
        if op.necromancy_operations:
            _validate_target_keys(
                op.necromancy_operations,
                f"{source}/operations[{i}]/necromancy",
            )
        if op.earth_rite_operations:
            _validate_target_keys(
                op.earth_rite_operations,
                f"{source}/operations[{i}]/earth_rite",
            )
        if op.then_operations:
            _validate_target_keys(
                op.then_operations,
                f"{source}/operations[{i}]/then",
            )
        if op.else_operations:
            _validate_target_keys(
                op.else_operations,
                f"{source}/operations[{i}]/else",
            )
        if op.optional_operations:
            _validate_target_keys(
                op.optional_operations,
                f"{source}/operations[{i}]/optional",
            )
        for option_index, option in enumerate(op.choose_one_options):
            _validate_target_keys(
                option.operations,
                f"{source}/operations[{i}]/options[{option_index}]",
            )


def _parse_operation(raw: dict, source_file: str, card_id: int, _depth: int = 0) -> EffectOperation:
    if _depth > 16:
        raise ValueError(f"{source_file} card {card_id}: nested effect depth exceeds maximum of 16")
    error_prefix = f"{source_file} card {card_id}"
    if not isinstance(raw, dict):
        raise ValueError(
            f"{error_prefix}: operation must be an object, "
            f"got {type(raw).__name__}"
        )
    _reject_unsupported_preselected_target_fields(raw, source_file, card_id)
    try:
        kind = EffectKind(raw["kind"])
        raw_target = raw.get("target", "own_leader" if kind in (
            EffectKind.ADD_EARTH_SIGILS,
            EffectKind.EARTH_RITE,
            EffectKind.NECROMANCY,
            EffectKind.REANIMATE,
            EffectKind.CONDITIONAL,
            EffectKind.CHOOSE_ONE,
            EffectKind.OPTIONAL,
        ) else None)
        if raw_target is None:
            raise KeyError("target")
        target = TargetKind(raw_target)
    except (KeyError, ValueError) as e:
        raise ValueError(f"{error_prefix}: invalid kind/target: {e}") from e

    conditions = ()
    raw_conds = raw.get("conditions")
    if raw_conds is not None:
        if not isinstance(raw_conds, list):
            raise ValueError(
                f"{error_prefix}: 'conditions' must be a list, "
                f"got {type(raw_conds).__name__}"
            )
        conditions = tuple(
            _parse_condition(c, f"{source_file}/conditions[{i}]", card_id)
            for i, c in enumerate(raw_conds)
        )

    amount_expr = None
    raw_amount = raw.get("amount")
    if kind is EffectKind.SET_STATS and raw_amount is None:
        raw_amount = raw.get("attack")
    if isinstance(raw_amount, dict):
        amount_expr = _parse_expression(raw_amount, f"{source_file}/amount", card_id)

    secondary_expr = None
    raw_secondary = raw.get("secondary_amount")
    if kind is EffectKind.SET_STATS and raw_secondary is None:
        raw_secondary = raw.get("health")
    if isinstance(raw_secondary, dict):
        secondary_expr = _parse_expression(raw_secondary, f"{source_file}/secondary_amount", card_id)

    raw_target_count = raw.get("target_count")
    raw_target_count_expr = raw.get("target_count_expr")
    if raw_target_count is not None and raw_target_count_expr is not None:
        raise ValueError(
            f"{error_prefix}: target_count and target_count_expr are mutually exclusive"
        )
    if raw_target_count is None:
        target_count = 1
    else:
        target_count = _parse_non_negative_int(
            raw_target_count,
            f"{source_file}/target_count",
            card_id,
        )
        if target_count < 1:
            raise ValueError(
                f"{source_file}/target_count card {card_id}: must be positive"
            )
    target_count_expr = None
    if raw_target_count_expr is not None:
        if not isinstance(raw_target_count_expr, dict):
            raise ValueError(
                f"{source_file}/target_count_expr card {card_id}: must be an expression object"
            )
        target_count_expr = _parse_expression(
            raw_target_count_expr,
            f"{source_file}/target_count_expr",
            card_id,
        )
        if _expression_depends_on_target(target_count_expr):
            raise ValueError(
                f"{source_file}/target_count_expr card {card_id}: "
                "cannot depend on a target selected by the same operation"
            )

    duplicate_fields = [
        field
        for field in ("allow_duplicate_targets", "allow_duplicates")
        if field in raw
    ]
    duplicate_values: list[bool] = []
    for field in duplicate_fields:
        value = raw[field]
        if not isinstance(value, bool):
            raise ValueError(
                f"{source_file}/{field} card {card_id}: must be boolean"
            )
        duplicate_values.append(value)
    if len(set(duplicate_values)) > 1:
        raise ValueError(
            f"{error_prefix}: allow_duplicate_targets and allow_duplicates conflict"
        )
    allow_duplicate_targets = duplicate_values[0] if duplicate_values else False
    has_multi_target_fields = (
        raw_target_count is not None
        or raw_target_count_expr is not None
        or bool(duplicate_fields)
    )
    if has_multi_target_fields and target not in _MULTI_TARGET_TARGETS:
        raise ValueError(
            f"{error_prefix}: multi-target fields require a selected target, "
            f"got {target.value!r}"
        )

    if kind is EffectKind.SET_STATS:
        if raw_amount is None and raw_secondary is None:
            raise ValueError(
                f"{source_file} card {card_id}: "
                f"SET_STATS requires at least one of 'attack'/'amount' or 'health'/'secondary_amount'"
            )

    keyword = raw.get("keyword")
    if kind in (EffectKind.ADD_KEYWORD, EffectKind.REMOVE_KEYWORD):
        if not isinstance(keyword, str) or not keyword:
            raise ValueError(
                f"{source_file}/keyword card {card_id}: "
                f"'{kind.value}' requires a keyword"
            )
        try:
            keyword = normalize_keyword_name(keyword, strict=True)
        except ValueError as exc:
            raise ValueError(
                f"{source_file}/keyword card {card_id}: {exc}"
            ) from exc
        if keyword not in RUNTIME_UNIT_KEYWORDS:
            raise ValueError(
                f"{source_file}/keyword card {card_id}: keyword "
                f"{keyword!r} is not a supported runtime unit keyword"
            )

    restriction = raw.get("restriction")
    if kind in (
        EffectKind.ADD_ATTACK_RESTRICTION,
        EffectKind.REMOVE_ATTACK_RESTRICTION,
        EffectKind.ADD_TARGETING_RESTRICTION,
        EffectKind.REMOVE_TARGETING_RESTRICTION,
    ):
        if not isinstance(restriction, str) or not restriction:
            raise ValueError(
                f"{source_file}/restriction card {card_id}: "
                f"'{kind.value}' requires a non-empty restriction"
            )
        from swb.engine.state import AttackRestriction, TargetingRestriction
        valid_attack = set(r.value for r in AttackRestriction)
        valid_targeting = set(r.value for r in TargetingRestriction)
        if kind in (EffectKind.ADD_ATTACK_RESTRICTION, EffectKind.REMOVE_ATTACK_RESTRICTION):
            if restriction not in valid_attack:
                raise ValueError(
                    f"{source_file}/restriction card {card_id}: "
                    f"unknown attack restriction {restriction!r}"
                )
        else:
            if restriction not in valid_targeting:
                raise ValueError(
                    f"{source_file}/restriction card {card_id}: "
                    f"unknown targeting restriction {restriction!r}"
                )

    mode = None
    if kind is EffectKind.CHANGE_COST:
        raw_mode = raw.get("mode", CostChangeMode.ADD.value)
        try:
            mode = CostChangeMode(raw_mode)
        except ValueError as exc:
            raise ValueError(
                f"{source_file}/mode card {card_id}: invalid cost mode "
                f"{raw_mode!r}"
            ) from exc

    raw_duration = raw.get("duration", ModifierDuration.PERMANENT.value)
    try:
        duration = ModifierDuration(raw_duration)
    except ValueError as exc:
        raise ValueError(
            f"{source_file}/duration card {card_id}: invalid duration "
            f"{raw_duration!r}"
        ) from exc

    operation_card_id = raw.get("card_id")
    if kind in (EffectKind.SUMMON, EffectKind.ADD_CARD, EffectKind.TRANSFORM):
        if operation_card_id is None:
            raise ValueError(
                f"{source_file}/card_id card {card_id}: "
                f"'{kind.value}' requires card_id"
            )
        try:
            operation_card_id = int(operation_card_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source_file}/card_id card {card_id}: expected an integer"
            ) from exc

    emblem_id = raw.get("emblem_id")
    if kind in (EffectKind.GAIN_EMBLEM, EffectKind.ADD_EMBLEM, EffectKind.REMOVE_EMBLEM):
        if not isinstance(emblem_id, str) or not emblem_id:
            raise ValueError(
                f"{source_file}/emblem_id card {card_id}: "
                f"'{kind.value}' requires a non-empty emblem_id string"
            )
    else:
        if emblem_id is not None:
            raise ValueError(
                f"{source_file}/emblem_id card {card_id}: "
                f"emblem_id is only valid for GAIN_EMBLEM/ADD_EMBLEM/REMOVE_EMBLEM"
            )
    emblem_remove_mode = raw.get("remove_mode", "first")
    if kind is EffectKind.REMOVE_EMBLEM:
        if emblem_remove_mode not in {"first", "all"}:
            raise ValueError(
                f"{source_file}/remove_mode card {card_id}: must be "
                f"'first' or 'all', got {emblem_remove_mode!r}"
            )
    elif "remove_mode" in raw:
        raise ValueError(
            f"{source_file}/remove_mode card {card_id}: remove_mode is only "
            f"valid for remove_emblem"
        )

    target_key = raw.get("target_key")
    if target_key is not None and not isinstance(target_key, str):
        raise ValueError(
            f"{source_file}/target_key card {card_id}: must be a string"
        )
    if target is TargetKind.PREVIOUS_TARGET:
        if not target_key:
            raise ValueError(
                f"{source_file} card {card_id}: "
                f"PREVIOUS_TARGET requires a non-empty target_key"
            )
    if kind is EffectKind.SELECT_TARGETS:
        if not target_key:
            raise ValueError(
                f"{source_file}/target_key card {card_id}: "
                "select_targets requires a non-empty target_key"
            )
        if target not in _BINDABLE_TARGETS:
            raise ValueError(
                f"{source_file}/target card {card_id}: select_targets requires "
                "selected board-entity targets"
            )

    requires_target = raw.get("requires_target", False)
    if not isinstance(requires_target, bool):
        raise ValueError(
            f"{source_file}/requires_target card {card_id}: must be boolean"
        )
    if requires_target and target not in _REQUIRES_TARGET_TARGETS:
        raise ValueError(
            f"{source_file}/requires_target card {card_id}: requires_target "
            f"is only valid for targets with explicit candidate sets, "
            f"got {target.value!r}"
        )
    if kind is EffectKind.TARGET_EXISTS:
        if requires_target:
            raise ValueError(
                f"{source_file}/requires_target card {card_id}: "
                "target_exists defines its own no-target branch semantics"
            )
        if target not in _TARGET_EXISTS_TARGETS:
            raise ValueError(
                f"{source_file}/target card {card_id}: target_exists requires "
                f"a target with explicit candidate sets, got {target.value!r}"
            )

    earth_rite_ops: tuple = ()
    if kind is EffectKind.EARTH_RITE:
        raw_inner = raw.get("operations")
        if not isinstance(raw_inner, list) or len(raw_inner) == 0:
            raise ValueError(
                f"{source_file} card {card_id}: "
                "EARTH_RITE requires non-empty 'operations' list"
            )
        earth_rite_ops = tuple(
            _parse_operation(op, f"{source_file}/operations[{i}]", card_id)
            for i, op in enumerate(raw_inner)
        )
        _validate_target_keys(
            earth_rite_ops,
            f"{source_file} card {card_id} (earth_rite)",
        )

    if kind in (EffectKind.ADD_EARTH_SIGILS, EffectKind.EARTH_RITE):
        if (
            raw_amount is None
            or isinstance(raw_amount, bool)
            or not isinstance(raw_amount, int)
            or raw_amount <= 0
        ):
            raise ValueError(
                f"{source_file}/amount card {card_id}: {kind.value} "
                f"requires a positive integer amount, got {raw_amount!r}"
            )
        if target is not TargetKind.OWN_LEADER:
            raise ValueError(
                f"{source_file}/target card {card_id}: {kind.value} "
                "requires target 'own_leader'"
            )

    necromancy_ops: tuple = ()
    if kind is EffectKind.NECROMANCY:
        raw_inner = raw.get("operations")
        if not isinstance(raw_inner, list) or len(raw_inner) == 0:
            raise ValueError(
                f"{source_file} card {card_id}: "
                f"NECROMANCY requires non-empty 'operations' list"
            )
        necromancy_ops = tuple(
            _parse_operation(op, f"{source_file}/operations[{i}]", card_id)
            for i, op in enumerate(raw_inner)
        )
        # validate nested target bindings
        _validate_target_keys(necromancy_ops, f"{source_file} card {card_id} (necromancy)")
        if raw_amount is None or raw_amount is True or raw_amount is False:
            raise ValueError(
                f"{source_file}/amount card {card_id}: "
                f"NECROMANCY requires a non-negative integer amount, got {raw_amount!r}"
            )
        if not isinstance(raw_amount, (int, float)) or raw_amount < 0:
            raise ValueError(
                f"{source_file}/amount card {card_id}: "
                f"NECROMANCY amount must be a non-negative integer, got {raw_amount!r}"
            )
        if isinstance(raw_amount, float) and raw_amount != int(raw_amount):
            raise ValueError(
                f"{source_file}/amount card {card_id}: "
                f"NECROMANCY amount must be an integer, got {raw_amount!r}"
            )

    if kind is EffectKind.REANIMATE:
        if raw_amount is None or isinstance(raw_amount, bool):
            raise ValueError(
                f"{source_file}/amount card {card_id}: "
                f"REANIMATE requires a non-negative integer amount, got {raw_amount!r}"
            )
        if not isinstance(raw_amount, (int, float)) or raw_amount < 0:
            raise ValueError(
                f"{source_file}/amount card {card_id}: "
                f"REANIMATE amount must be a non-negative integer, got {raw_amount!r}"
            )
        if isinstance(raw_amount, float) and raw_amount != int(raw_amount):
            raise ValueError(
                f"{source_file}/amount card {card_id}: "
                f"REANIMATE amount must be an integer, got {raw_amount!r}"
            )

    if kind is EffectKind.ADD_COMBO:
        if target not in (TargetKind.OWN_LEADER, TargetKind.ENEMY_LEADER):
            raise ValueError(
                f"{source_file} card {card_id}: add_combo requires own_leader "
                f"or enemy_leader target, got {target.value!r}"
            )
        if raw_amount is None or isinstance(raw_amount, dict):
            raise ValueError(
                f"{source_file}/amount card {card_id}: "
                f"add_combo requires a non-negative integer amount"
            )
        _parse_non_negative_int(raw_amount, f"{source_file}/amount", card_id)

    raw_cost_max = raw.get("cost_max")
    raw_cost_min = raw.get("cost_min")
    graveyard_cost_max = raw_cost_max if kind in _GRAVEYARD_EFFECT_KINDS else None
    if graveyard_cost_max is not None:
        graveyard_cost_max = _parse_non_negative_int(graveyard_cost_max, f"{source_file}/cost_max", card_id)
    graveyard_cost_min = raw_cost_min if kind in _GRAVEYARD_EFFECT_KINDS else None
    if graveyard_cost_min is not None:
        graveyard_cost_min = _parse_non_negative_int(graveyard_cost_min, f"{source_file}/cost_min", card_id)
    if graveyard_cost_min is not None and graveyard_cost_max is not None and graveyard_cost_min > graveyard_cost_max:
        raise ValueError(
            f"{source_file} card {card_id}: cost_min ({graveyard_cost_min}) "
            f"must not exceed cost_max ({graveyard_cost_max})"
        )
    raw_follower_only = raw.get("follower_only")
    if raw_follower_only is not None and not isinstance(raw_follower_only, bool):
        raise ValueError(
            f"{source_file}/follower_only card {card_id}: must be boolean, "
            f"got {type(raw_follower_only).__name__}"
        )
    graveyard_follower_only = bool(raw_follower_only) if raw_follower_only is not None else False
    card_type_filter = raw.get("card_type_filter")
    if card_type_filter is not None:
        if not isinstance(card_type_filter, str):
            raise ValueError(f"{source_file}/card_type_filter card {card_id}: must be a string")
        if card_type_filter not in _VALID_CARD_TYPES:
            raise ValueError(
                f"{source_file}/card_type_filter card {card_id}: "
                f"unknown card type {card_type_filter!r}; valid: {sorted(_VALID_CARD_TYPES)}"
            )
    graveyard_card_type = card_type_filter if kind in _GRAVEYARD_EFFECT_KINDS else None
    deck_class_id = raw.get("class_id_filter")
    if deck_class_id is not None:
        deck_class_id = _parse_non_negative_int(
            deck_class_id,
            f"{source_file}/class_id_filter",
            card_id,
        )
    deck_class_name = raw.get("class_name_filter")
    if deck_class_name is not None and not isinstance(deck_class_name, str):
        raise ValueError(
            f"{source_file}/class_name_filter card {card_id}: must be a string"
        )
    deck_cost_min = raw_cost_min if kind is EffectKind.DRAW_FILTERED else None
    deck_cost_max = raw_cost_max if kind is EffectKind.DRAW_FILTERED else None
    deck_card_id = raw.get("card_id_filter")
    if deck_card_id is not None:
        deck_card_id = _parse_non_negative_int(
            deck_card_id,
            f"{source_file}/card_id_filter",
            card_id,
        )
    deck_card_name = raw.get("card_name_filter")
    if deck_card_name is not None and not isinstance(deck_card_name, str):
        raise ValueError(
            f"{source_file}/card_name_filter card {card_id}: must be a string"
        )
    operation_board_filter = _parse_board_filter(
        raw,
        source_path=source_file,
        card_id=card_id,
        prefix="target_",
        allow_evolved=True,
    )

    _is_graveyard_kind = kind in _GRAVEYARD_EFFECT_KINDS
    _is_deck_filter_kind = kind is EffectKind.DRAW_FILTERED
    deck_filter: DeckFilter | None = None
    board_filter: BoardFilter | None = None
    if _is_deck_filter_kind and target not in (TargetKind.OWN_LEADER, TargetKind.ENEMY_LEADER):
        raise ValueError(
            f"{source_file} card {card_id}: draw_filtered requires own_leader "
            f"or enemy_leader target, got {target.value!r}"
        )
    if _is_deck_filter_kind:
        if deck_cost_min is not None:
            deck_cost_min = _parse_non_negative_int(
                deck_cost_min,
                f"{source_file}/cost_min",
                card_id,
            )
        if deck_cost_max is not None:
            deck_cost_max = _parse_non_negative_int(
                deck_cost_max,
                f"{source_file}/cost_max",
                card_id,
            )
        if deck_cost_min is not None and deck_cost_max is not None and deck_cost_min > deck_cost_max:
            raise ValueError(
                f"{source_file} card {card_id}: cost_min ({deck_cost_min}) "
                f"must not exceed cost_max ({deck_cost_max})"
            )
        deck_filter = DeckFilter(
            card_type=card_type_filter,
            class_id=deck_class_id,
            class_name=deck_class_name,
            cost_min=deck_cost_min,
            cost_max=deck_cost_max,
            card_id=deck_card_id,
            card_name=deck_card_name,
        )
    if not _is_deck_filter_kind and any([
        raw.get("class_id_filter") is not None,
        raw.get("class_name_filter") is not None,
        raw.get("card_id_filter") is not None,
        raw.get("card_name_filter") is not None,
    ]):
        raise ValueError(
            f"{source_file} card {card_id}: deck filter fields "
            f"are only valid with draw_filtered"
        )
    if _is_graveyard_kind:
        if target not in _GRAVEYARD_TARGETS:
            raise ValueError(
                f"{source_file} card {card_id}: {kind.value} requires a graveyard target, "
                f"got {target.value!r}"
            )
    board_filter_fields_present = operation_board_filter is not None
    if board_filter_fields_present:
        if target in _GRAVEYARD_TARGETS or target in (
            TargetKind.OWN_HAND,
            TargetKind.RANDOM_OWN_HAND,
            TargetKind.ALL_OWN_HAND,
            TargetKind.OWN_LEADER,
            TargetKind.ENEMY_LEADER,
            TargetKind.OWN_UNIT_OR_LEADER,
            TargetKind.ENEMY_UNIT_OR_LEADER,
            TargetKind.ANY_UNIT_OR_LEADER,
            TargetKind.SELF,
            TargetKind.PREVIOUS_TARGET,
        ):
            raise ValueError(
                f"{source_file} card {card_id}: target_*_filter fields require "
                f"a board target, got {target.value!r}"
            )
        board_filter = operation_board_filter
    elif not _is_graveyard_kind:
        has_graveyard_filter = any([
            raw.get("cost_max") is not None and not _is_deck_filter_kind,
            raw.get("cost_min") is not None and not _is_deck_filter_kind,
            raw.get("follower_only") is not None,
            raw.get("card_type_filter") is not None and not _is_deck_filter_kind,
        ])
        if has_graveyard_filter:
            raise ValueError(
                f"{source_file} card {card_id}: graveyard filter fields "
                f"(cost_max/cost_min/follower_only/card_type_filter) are only valid with graveyard effect kinds"
            )

    if kind is EffectKind.SUMMON_FROM_GRAVEYARD:
        graveyard_follower_only = True
        if graveyard_card_type is not None and graveyard_card_type != "随从":
            raise ValueError(
                f"{source_file}/card_type_filter card {card_id}: "
                f"SUMMON_FROM_GRAVEYARD only supports 随从, got {graveyard_card_type!r}"
            )
        if raw_follower_only is not None and raw_follower_only is not True:
            raise ValueError(
                f"{source_file}/follower_only card {card_id}: "
                f"SUMMON_FROM_GRAVEYARD requires follower_only=true or omit it"
            )

    if _is_graveyard_kind:
        invalid_conds = _check_target_conditions(conditions, f"{source_file} card {card_id}")
        if invalid_conds:
            raise ValueError(
                f"{source_file} card {card_id}: graveyard operations do not support "
                f"target-dependent conditions: {sorted(invalid_conds)}"
            )

    then_ops: tuple = ()
    else_ops: tuple = ()
    choose_one_options: tuple = ()
    optional_prompt: str | None = None
    optional_ops: tuple = ()

    if kind is EffectKind.CONDITIONAL:
        unknown_conditional_keys = set(raw) - {
            "kind", "target", "conditions", "then", "else",
        }
        if unknown_conditional_keys:
            raise ValueError(
                f"{error_prefix}: unknown fields {sorted(unknown_conditional_keys)}"
            )
        if not conditions:
            raise ValueError(f"{error_prefix}: CONDITIONAL requires non-empty 'conditions'")
        invalid_conditional_conditions = _check_target_conditions(
            conditions,
            f"{source_file} card {card_id}",
        )
        if invalid_conditional_conditions:
            raise ValueError(
                f"{error_prefix}/conditions: conditional conditions cannot "
                f"depend on a selected target: {sorted(invalid_conditional_conditions)}"
            )
        raw_then = raw.get("then")
        if raw_then is None:
            raise ValueError(f"{error_prefix}: CONDITIONAL requires 'then'")
        if not isinstance(raw_then, list):
            raise ValueError(f"{error_prefix}/then: must be a list")
        then_ops = tuple(
            _parse_operation(op, f"{source_file}/then[{idx}]", card_id, _depth + 1)
            for idx, op in enumerate(raw_then)
        )
        _validate_target_keys(then_ops, f"{source_file}/then (card {card_id})")
        raw_else = raw.get("else")
        if raw_else is not None:
            if not isinstance(raw_else, list):
                raise ValueError(f"{error_prefix}/else: must be a list")
            else_ops = tuple(
                _parse_operation(op, f"{source_file}/else[{idx}]", card_id, _depth + 1)
                for idx, op in enumerate(raw_else)
            )
            _validate_target_keys(else_ops, f"{source_file}/else (card {card_id})")

    elif kind is EffectKind.TARGET_EXISTS:
        unknown_target_exists_keys = set(raw) - {
            "kind", "target", "conditions", "then", "else",
            "target_card_type_filter", "target_cost_min", "target_cost_max",
            "target_card_id_filter", "target_card_name_filter",
            "target_evolved_filter",
        }
        if unknown_target_exists_keys:
            raise ValueError(
                f"{error_prefix}: unknown fields {sorted(unknown_target_exists_keys)}"
            )
        if not ("then" in raw or "else" in raw):
            raise ValueError(
                f"{error_prefix}: TARGET_EXISTS requires at least one of 'then' or 'else'"
            )
        invalid_target_conditions = _check_target_conditions(
            conditions,
            f"{source_file} card {card_id}",
        )
        if invalid_target_conditions and (
            target in _GRAVEYARD_TARGETS
            or target in {
                TargetKind.OWN_HAND,
                TargetKind.RANDOM_OWN_HAND,
                TargetKind.ALL_OWN_HAND,
            }
        ):
            raise ValueError(
                f"{error_prefix}/conditions: target_exists conditions cannot "
                f"depend on non-board targets: {sorted(invalid_target_conditions)}"
            )
        raw_then = raw.get("then", [])
        if not isinstance(raw_then, list):
            raise ValueError(f"{error_prefix}/then: must be a list")
        then_ops = tuple(
            _parse_operation(op, f"{source_file}/then[{idx}]", card_id, _depth + 1)
            for idx, op in enumerate(raw_then)
        )
        _validate_target_keys(then_ops, f"{source_file}/then (card {card_id})")
        raw_else = raw.get("else", [])
        if not isinstance(raw_else, list):
            raise ValueError(f"{error_prefix}/else: must be a list")
        else_ops = tuple(
            _parse_operation(op, f"{source_file}/else[{idx}]", card_id, _depth + 1)
            for idx, op in enumerate(raw_else)
        )
        _validate_target_keys(else_ops, f"{source_file}/else (card {card_id})")
        if not then_ops and not else_ops:
            raise ValueError(
                f"{error_prefix}: TARGET_EXISTS requires a non-empty branch"
            )

    elif kind is EffectKind.CHOOSE_ONE:
        unknown_choose_keys = set(raw) - {
            "kind", "target", "options",
        }
        if unknown_choose_keys:
            raise ValueError(
                f"{error_prefix}: unknown fields {sorted(unknown_choose_keys)}"
            )
        raw_options = raw.get("options")
        if not isinstance(raw_options, list) or len(raw_options) == 0:
            raise ValueError(f"{error_prefix}: CHOOSE_ONE requires non-empty 'options' list")
        choose_ops: list = []
        seen_ids: set[str] = set()
        for idx, raw_opt in enumerate(raw_options):
            opt_source = f"{source_file}/options[{idx}]"
            if not isinstance(raw_opt, dict):
                raise ValueError(f"{opt_source}: option must be an object")
            unknown_opt_keys = set(raw_opt) - {"id", "label", "conditions", "operations"}
            if unknown_opt_keys:
                raise ValueError(
                    f"{opt_source}: unknown fields {sorted(unknown_opt_keys)}"
                )
            opt_id = raw_opt.get("id")
            if not isinstance(opt_id, str) or not opt_id:
                raise ValueError(f"{opt_source}/id: must be a non-empty string")
            if opt_id in seen_ids:
                raise ValueError(f"{opt_source}/id: duplicate option id {opt_id!r}")
            seen_ids.add(opt_id)
            opt_label = raw_opt.get("label", opt_id)
            if not isinstance(opt_label, str):
                raise ValueError(f"{opt_source}/label: must be a string")
            opt_conditions: tuple = ()
            raw_opt_conds = raw_opt.get("conditions")
            if raw_opt_conds is not None:
                if not isinstance(raw_opt_conds, list):
                    raise ValueError(f"{opt_source}/conditions: must be a list")
                opt_conditions = tuple(
                    _parse_condition(c, f"{opt_source}/conditions[{j}]", card_id)
                    for j, c in enumerate(raw_opt_conds)
                )
                invalid_option_conditions = _check_target_conditions(
                    opt_conditions,
                    opt_source,
                )
                if invalid_option_conditions:
                    raise ValueError(
                        f"{opt_source}/conditions: choose_one option conditions cannot "
                        f"depend on a selected target: {sorted(invalid_option_conditions)}"
                    )
            raw_opt_ops = raw_opt.get("operations", [])
            if not isinstance(raw_opt_ops, list):
                raise ValueError(f"{opt_source}/operations: must be a list")
            opt_operations = tuple(
                _parse_operation(op, f"{opt_source}/operations[{j}]", card_id, _depth + 1)
                for j, op in enumerate(raw_opt_ops)
            )
            _validate_target_keys(opt_operations, opt_source)
            choose_ops.append(ChooseOneOption(
                option_id=opt_id,
                label=opt_label,
                conditions=opt_conditions,
                operations=opt_operations,
            ))
        choose_one_options = tuple(choose_ops)

    elif kind is EffectKind.OPTIONAL:
        unknown_optional_keys = set(raw) - {
            "kind", "target", "prompt", "operations",
        }
        if unknown_optional_keys:
            raise ValueError(
                f"{error_prefix}: unknown fields {sorted(unknown_optional_keys)}"
            )
        optional_prompt = raw.get("prompt", "是否发动？")
        if not isinstance(optional_prompt, str) or not optional_prompt:
            raise ValueError(f"{error_prefix}/prompt: must be a non-empty string")
        raw_ops = raw.get("operations", [])
        if not isinstance(raw_ops, list):
            raise ValueError(f"{error_prefix}/operations: must be a list")
        optional_ops = tuple(
            _parse_operation(op, f"{source_file}/operations[{idx}]", card_id, _depth + 1)
            for idx, op in enumerate(raw_ops)
        )
        _validate_target_keys(optional_ops, f"{source_file}/operations (card {card_id})")

    return EffectOperation(
        kind=kind,
        target=target,
        amount=_parse_optional_int(raw_amount, f"{source_file}/amount", card_id),
        secondary_amount=_parse_optional_int(
            raw_secondary,
            f"{source_file}/secondary_amount",
            card_id,
        ),
        card_id=operation_card_id,
        emblem_id=emblem_id,
        emblem_remove_mode=emblem_remove_mode if kind is EffectKind.REMOVE_EMBLEM else "first",
        keyword=keyword,
        restriction=restriction,
        conditions=conditions,
        amount_expr=amount_expr,
        secondary_expr=secondary_expr,
        mode=mode,
        duration=duration,
        set_attack=(
            kind is EffectKind.SET_STATS
            and (raw.get("amount") is not None or raw.get("attack") is not None or amount_expr is not None)
        ),
        set_health=(
            kind is EffectKind.SET_STATS
            and (raw.get("secondary_amount") is not None or raw.get("health") is not None or secondary_expr is not None)
        ),
        target_key=target_key,
        earth_rite_operations=earth_rite_ops,
        necromancy_operations=necromancy_ops,
        graveyard_cost_max=graveyard_cost_max,
        graveyard_cost_min=graveyard_cost_min,
        graveyard_follower_only=graveyard_follower_only,
        graveyard_card_type=graveyard_card_type,
        deck_filter=deck_filter,
        board_filter=board_filter,
        then_operations=then_ops,
        else_operations=else_ops,
        choose_one_options=choose_one_options,
        optional_prompt=optional_prompt,
        optional_operations=optional_ops,
        requires_target=requires_target,
        target_count=target_count,
        target_count_expr=target_count_expr,
        allow_duplicate_targets=allow_duplicate_targets,
    )


def _parse_condition(raw: dict, source_path: str, card_id: int) -> Condition:
    error_prefix = f"{source_path} card {card_id}"
    if not isinstance(raw, dict):
        raise ValueError(
            f"{error_prefix}: condition must be an object, "
            f"got {type(raw).__name__}"
        )
    try:
        t = ConditionType(raw["type"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"{error_prefix}: invalid condition type: {e}") from e

    sub_raws = raw.get("conditions", [])
    if not isinstance(sub_raws, list):
        raise ValueError(f"{error_prefix}: 'conditions' must be a list")
    sub = [
        _parse_condition(c, f"{source_path}/conditions[{i}]", card_id)
        for i, c in enumerate(sub_raws)
    ]

    if t in (ConditionType.ALL, ConditionType.ANY) and not sub:
        raise ValueError(
            f"{error_prefix}: '{t.value}' requires at least one sub-condition"
        )
    if t == ConditionType.NOT and len(sub) != 1:
        raise ValueError(
            f"{error_prefix}: 'not' requires exactly one sub-condition"
        )

    cooperation_threshold_types = (
        ConditionType.CONTROLLER_COOPERATION_AT_LEAST,
        ConditionType.OPPONENT_COOPERATION_AT_LEAST,
        ConditionType.CONTROLLER_COMBO_AT_LEAST,
        ConditionType.OPPONENT_COMBO_AT_LEAST,
        ConditionType.CONTROLLER_EARTH_SIGILS_AT_LEAST,
        ConditionType.OPPONENT_EARTH_SIGILS_AT_LEAST,
        ConditionType.CONTROLLER_EVOLUTIONS_THIS_MATCH_AT_LEAST,
        ConditionType.OPPONENT_EVOLUTIONS_THIS_MATCH_AT_LEAST,
        ConditionType.SOURCE_FUSION_COUNT_AT_LEAST,
    )
    if t in cooperation_threshold_types:
        if "value" not in raw:
            raise ValueError(
                f"{source_path}/value card {card_id}: required for {t.value!r}"
            )
        value = _parse_non_negative_int(
            raw["value"], f"{source_path}/value", card_id
        )
    else:
        value = _parse_optional_int(
            raw.get("value"), f"{source_path}/value", card_id
        )

    keyword = raw.get("keyword")
    if t in (ConditionType.SOURCE_HAS_KEYWORD, ConditionType.TARGET_HAS_KEYWORD):
        if not isinstance(keyword, str) or not keyword:
            raise ValueError(f"{source_path}/keyword card {card_id}: keyword required")
        try:
            keyword = normalize_keyword_name(keyword, strict=True)
        except ValueError as exc:
            raise ValueError(
                f"{source_path}/keyword card {card_id}: {exc}"
            ) from exc

    board_filter = None
    if t in (ConditionType.CONTROLLER_BOARD_HAS, ConditionType.OPPONENT_BOARD_HAS):
        board_filter = _parse_board_filter(
            raw,
            source_path=source_path,
            card_id=card_id,
            allow_evolved=True,
        )

    return Condition(
        type=t,
        value=value,
        keyword=keyword,
        board_filter=board_filter,
        conditions=sub,
    )


def _parse_expression(raw: dict, source_path: str, card_id: int) -> ValueExpression:
    error_prefix = f"{source_path} card {card_id}"
    if not isinstance(raw, dict):
        raise ValueError(
            f"{error_prefix}: expression must be an object, "
            f"got {type(raw).__name__}"
        )
    try:
        t = ExprType(raw["type"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"{error_prefix}: invalid expression type: {e}") from e

    sub_raws = raw.get("values", [])
    if not isinstance(sub_raws, list):
        raise ValueError(f"{error_prefix}: 'values' must be a list")
    sub = [
        _parse_expression(v, f"{source_path}/values[{i}]", card_id)
        for i, v in enumerate(sub_raws)
    ]

    if t in (ExprType.ADD, ExprType.SUBTRACT, ExprType.MULTIPLY, ExprType.MIN, ExprType.MAX):
        if len(sub) < 1:
            raise ValueError(
                f"{error_prefix}: '{t.value}' requires at least one value"
            )

    if t == ExprType.CONSTANT:
        if sub:
            raise ValueError(
                f"{error_prefix}: 'constant' must not have 'values'"
            )
    elif t in (ExprType.CONTROLLER_BOARD_COUNT, ExprType.OPPONENT_BOARD_COUNT,
               ExprType.CONTROLLER_HAND_COUNT, ExprType.SOURCE_ATTACK, ExprType.SOURCE_HEALTH,
               ExprType.TARGET_ATTACK, ExprType.TARGET_HEALTH,
               ExprType.CONTROLLER_SHADOWS, ExprType.OPPONENT_SHADOWS,
               ExprType.CONTROLLER_COOPERATION, ExprType.OPPONENT_COOPERATION,
               ExprType.CONTROLLER_OVERFLOW, ExprType.OPPONENT_OVERFLOW,
               ExprType.CONTROLLER_COMBO, ExprType.OPPONENT_COMBO,
               ExprType.CONTROLLER_EARTH_SIGILS, ExprType.OPPONENT_EARTH_SIGILS):
        if sub:
            raise ValueError(
                f"{error_prefix}: '{t.value}' must not have 'values'"
            )

    return ValueExpression(
        type=t,
        value=_parse_optional_int(raw.get("value"), f"{source_path}/value", card_id),
        values=sub,
    )


def _parse_optional_int(raw, source_path: str, card_id: int) -> int:
    if raw is None or isinstance(raw, dict):
        return 0
    if isinstance(raw, bool):
        raise ValueError(
            f"{source_path} card {card_id}: expected an integer, got boolean"
        )
    if not isinstance(raw, int) and not isinstance(raw, str):
        raise ValueError(
            f"{source_path} card {card_id}: expected an integer, got {type(raw).__name__}"
        )
    try:
        result = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source_path} card {card_id}: expected an integer, got {raw!r}"
        ) from exc
    if isinstance(raw, str) and str(result) != raw.strip():
        raise ValueError(
            f"{source_path} card {card_id}: expected an integer, got string {raw!r}"
        )
    return result


def _parse_non_negative_int(raw, source_path: str, card_id: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            f"{source_path} card {card_id}: expected an integer, "
            f"got {type(raw).__name__} ({raw!r})"
        )
    if raw < 0:
        raise ValueError(
            f"{source_path} card {card_id}: must be non-negative, got {raw}"
        )
    return raw
