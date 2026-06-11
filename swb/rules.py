from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable


TAG_RE = re.compile(r"<[^>]+>")
KEYWORD_RE = re.compile(r"<color=Keyword>(.*?)</color>")


@dataclass(frozen=True)
class EffectDefinition:
    kind: str
    amount: int
    secondary_amount: int = 0


def clean_printed_text(value: str) -> str:
    value = value.replace("<hr>", "\n").replace("<ev>", "").replace("</ev>", "")
    return html.unescape(TAG_RE.sub("", value)).strip()


def parse_fanfare(skill_texts: Iterable[str]) -> tuple[tuple[EffectDefinition, ...], bool]:
    """Parse only simple, unconditional fanfares.

    The boolean indicates whether a printed fanfare was present and completely
    understood. Unknown or targeted clauses make the whole fanfare unsupported.
    """
    joined = "\n".join(skill_texts)
    if "入场曲" not in joined:
        return (), False

    base_section = joined.split("<hr>", 1)[0]
    clean = clean_printed_text(base_section)
    marker = "【入场曲】"
    if marker not in clean:
        return (), False
    body = clean.split(marker, 1)[1].strip()
    clauses = [part.strip() for part in re.split(r"[。\n]+", body) if part.strip()]
    effects: list[EffectDefinition] = []

    patterns = (
        (r"抽取(\d+)张卡牌", lambda m: EffectDefinition("draw", int(m.group(1)))),
        (
            r"回复自己的主战者(\d+)点生命值",
            lambda m: EffectDefinition("heal_leader", int(m.group(1))),
        ),
        (
            r"对对手的主战者造成(\d+)点伤害",
            lambda m: EffectDefinition("damage_enemy_leader", int(m.group(1))),
        ),
        (
            r"对自己的主战者造成(\d+)点伤害",
            lambda m: EffectDefinition("damage_own_leader", int(m.group(1))),
        ),
        (
            r"回复自己(\d+)点能量点",
            lambda m: EffectDefinition("restore_mana", int(m.group(1))),
        ),
        (
            r"本随从\+(\d+)/\+(\d+)",
            lambda m: EffectDefinition("buff_self", int(m.group(1)), int(m.group(2))),
        ),
    )
    for clause in clauses:
        if re.fullmatch(r"【(?:守护|疾驰|突进)】", clause):
            continue
        for pattern, factory in patterns:
            match = re.fullmatch(pattern, clause)
            if match:
                effects.append(factory(match))
                break
        else:
            return (), False
    return tuple(effects), True

