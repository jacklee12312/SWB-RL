from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from swb.engine.events import EventType, GameEvent


INTERESTING_EVENTS = {
    EventType.CARD_PLAYED,
    EventType.CARD_DRAWN,
    EventType.FOLLOWER_SUMMONED,
    EventType.SPELL_RESOLVED,
    EventType.AMULET_ACTIVATED,
    EventType.ATTACK_DECLARED,
    EventType.DAMAGE_APPLIED,
    EventType.DAMAGE_PREVENTED,
    EventType.FOLLOWER_DESTROYED,
    EventType.AMULET_DESTROYED,
    EventType.CARD_BANISHED,
    EventType.CARD_RETURNED_TO_HAND,
    EventType.CARD_RETURNED_TO_DECK,
    EventType.CARD_DISCARDED,
    EventType.FOLLOWER_EVOLVED,
    EventType.FOLLOWER_SUPER_EVOLVED,
    EventType.LEADER_HEALED,
    EventType.FOLLOWER_HEALED,
    EventType.FOLLOWER_STATS_INCREASED,
    EventType.FOLLOWER_STATS_DECREASED,
    EventType.LAST_WORDS_START,
    EventType.EMBLEM_GAINED,
    EventType.EMBLEM_TRIGGERED,
    EventType.FAITH_PLACED,
    EventType.FAITH_VALUE_CHANGED,
    EventType.FAITH_ABILITY_TRIGGERED,
    EventType.GAME_ENDED,
}


def _identity(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    definition = getattr(value, "definition", value)
    return {
        key: result
        for key, result in (
            ("name", getattr(definition, "name", None)),
            ("card_id", getattr(definition, "card_id", None)),
            ("card_type", getattr(definition, "card_type", None)),
            ("entity_id", getattr(value, "entity_id", None)),
        )
        if result is not None
    }


def _safe_value(value: object, *, depth: int = 0) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if depth >= 3:
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item, depth=depth + 1)
            for key, item in value.items()
            if key not in {"source", "target", "card", "definition"}
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_value(item, depth=depth + 1) for item in value]
    identity = _identity(value)
    return identity or str(value)


def serialize_event(
    event: GameEvent,
    *,
    entity_names: Mapping[int, str],
    card_lookup: Callable[[int], object | None],
) -> dict[str, Any]:
    metadata = event.metadata
    source = (
        _identity(metadata.get("source"))
        or _identity(metadata.get("card"))
        or _identity(metadata.get("definition"))
    )
    target = _identity(metadata.get("target"))

    source_card_id = source.get("card_id") or metadata.get("card_id")
    if source_card_id is not None and not source.get("name"):
        source.update(_identity(card_lookup(int(source_card_id))))
    if event.source_id is not None:
        source.setdefault("entity_id", event.source_id)
        source.setdefault("name", entity_names.get(event.source_id))

    if event.target_id is not None:
        target.setdefault("entity_id", event.target_id)
        target.setdefault("name", entity_names.get(event.target_id))
    target_player = metadata.get("target_player")
    if not target and isinstance(target_player, int):
        target = {
            "name": f"玩家 {target_player + 1} 主战者",
            "player_index": target_player,
        }

    return {
        "type": event.type.value,
        "player_index": event.player_index,
        "source_id": event.source_id,
        "target_id": event.target_id,
        "amount": event.amount,
        "source": source,
        "target": target,
        "metadata": _safe_value(metadata),
    }


def _closest_log(
    logs: list[str],
    *,
    source_name: str | None,
    target_name: str | None,
) -> str | None:
    for line in reversed(logs):
        if source_name and source_name not in line:
            continue
        if target_name and target_name not in line:
            continue
        return line
    if source_name:
        for line in reversed(logs):
            if source_name in line:
                return line
    return logs[-1] if logs else None


def build_animation_cues(
    events: Iterable[dict[str, Any]],
    *,
    logs: list[str],
    action_label: str,
) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for event in events:
        event_type = EventType(event["type"])
        if event_type not in INTERESTING_EVENTS:
            continue
        source = event.get("source") or {}
        target = event.get("target") or {}
        source_name = source.get("name")
        target_name = target.get("name")
        amount = int(event.get("amount") or 0)
        actor = int(event["player_index"])
        kind = "effect"
        title = event_type.value
        detail: str | None = None

        if event_type is EventType.CARD_PLAYED:
            card_type = source.get("card_type")
            kind = {
                "法术": "spell",
                "护符": "amulet",
                "随从": "play",
            }.get(card_type, "play")
            title = f"玩家 {actor + 1} 打出 {source_name or '卡牌'}"
            detail = (
                f"{card_type or '卡牌'} · "
                f"{event['metadata'].get('source_cost', event['metadata'].get('base_cost', '?'))} PP"
            )
        elif event_type is EventType.FOLLOWER_SUMMONED:
            if event["metadata"].get("via") == "play":
                continue
            kind = "summon"
            title = f"{source_name or '随从'} 被召唤"
        elif event_type is EventType.CARD_DRAWN:
            kind = "draw"
            title = f"玩家 {actor + 1} 抽取 {source_name or '一张卡'}"
        elif event_type is EventType.SPELL_RESOLVED:
            kind = "spell"
            title = f"{source_name or '法术'} 结算完成"
        elif event_type is EventType.AMULET_ACTIVATED:
            kind = "amulet"
            title = f"策动 {source_name or '护符'}"
        elif event_type is EventType.ATTACK_DECLARED:
            kind = "attack"
            target_name = target_name or f"玩家 {2 - actor} 主战者"
            title = f"{source_name or '攻击者'} → {target_name}"
            detail = "宣告攻击"
        elif event_type is EventType.DAMAGE_APPLIED:
            kind = "damage"
            title = f"{target_name or '目标'}受到 {amount} 点伤害"
            detail = f"来源：{source_name or '效果'}"
        elif event_type is EventType.DAMAGE_PREVENTED:
            kind = "prevent"
            title = f"{target_name or '目标'}防止了 {amount} 点伤害"
        elif event_type in {
            EventType.FOLLOWER_DESTROYED,
            EventType.AMULET_DESTROYED,
        }:
            kind = "destroy"
            title = f"{source_name or '场上卡牌'}被破坏"
        elif event_type is EventType.CARD_BANISHED:
            kind = "banish"
            title = f"{source_name or target_name or '卡牌'}被消失"
        elif event_type is EventType.CARD_RETURNED_TO_HAND:
            kind = "move"
            title = f"{source_name or target_name or '卡牌'}返回手牌"
        elif event_type is EventType.CARD_RETURNED_TO_DECK:
            kind = "move"
            title = f"{source_name or target_name or '卡牌'}返回牌库"
        elif event_type is EventType.CARD_DISCARDED:
            kind = "move"
            title = f"{source_name or target_name or '卡牌'}被舍弃"
        elif event_type is EventType.FOLLOWER_EVOLVED:
            kind = "evolve"
            title = f"{source_name or '随从'}进化"
        elif event_type is EventType.FOLLOWER_SUPER_EVOLVED:
            kind = "super_evolve"
            title = f"{source_name or '随从'}超进化"
        elif event_type in {EventType.LEADER_HEALED, EventType.FOLLOWER_HEALED}:
            kind = "heal"
            title = f"{target_name or source_name or '目标'}回复 {amount} 点生命"
        elif event_type in {
            EventType.FOLLOWER_STATS_INCREASED,
            EventType.FOLLOWER_STATS_DECREASED,
        }:
            kind = "stats"
            direction = "获得强化" if amount >= 0 else "属性降低"
            title = f"{target_name or source_name or '随从'}{direction}"
        elif event_type is EventType.LAST_WORDS_START:
            kind = "last_words"
            title = f"{source_name or '卡牌'}发动谢幕曲"
        elif event_type in {EventType.EMBLEM_GAINED, EventType.FAITH_PLACED}:
            kind = "leader_area"
            title = "主战者区域新增纹章或信仰"
        elif event_type in {
            EventType.EMBLEM_TRIGGERED,
            EventType.FAITH_ABILITY_TRIGGERED,
            EventType.FAITH_VALUE_CHANGED,
        }:
            kind = "leader_area"
            title = f"{source_name or '主战者区域能力'}发动"
        elif event_type is EventType.GAME_ENDED:
            kind = "result"
            winner = event["metadata"].get("winner")
            title = "对局以平局结束" if winner is None else f"玩家 {int(winner) + 1} 获胜"

        if detail is None:
            detail = _closest_log(
                logs,
                source_name=source_name,
                target_name=target_name,
            )
        cues.append(
            {
                "kind": kind,
                "title": title,
                "detail": detail,
                "actor_player": actor,
                "source_entity_id": event.get("source_id"),
                "source_name": source_name,
                "target_entity_id": event.get("target_id"),
                "target_name": target_name,
                "amount": amount,
                "duration_ms": 1050 if kind in {"attack", "spell", "amulet"} else 780,
            }
        )

    if not cues:
        cues.append(
            {
                "kind": "action",
                "title": action_label,
                "detail": logs[-1] if logs else None,
                "actor_player": None,
                "source_entity_id": None,
                "source_name": None,
                "target_entity_id": None,
                "target_name": None,
                "amount": 0,
                "duration_ms": 720,
            }
        )
    return cues
