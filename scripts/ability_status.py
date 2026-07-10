from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from swb.engine.abilities import ABILITY_DEFINITIONS, AbilityKeyword
from scripts.report_rule_coverage import PRIMITIVE_KEYWORD_MAP


_MANUAL_PRIMITIVE_STATUS = {
    AbilityKeyword.STORM: "covered",
    AbilityKeyword.RUSH: "covered",
    AbilityKeyword.WARD: "covered",
    AbilityKeyword.AMBUSH: "covered",
}

_PRIMITIVE_PATTERN_BY_KEYWORD = {
    AbilityKeyword.COMBO: "连击",
    AbilityKeyword.COOPERATION: "协作",
    AbilityKeyword.OVERFLOW: "觉醒",
    AbilityKeyword.SPELLBOOST: "魔力增幅",
    AbilityKeyword.EARTH_RITE: "土之秘术|土之印",
    AbilityKeyword.EARTH_SIGIL: "土之秘术|土之印",
    AbilityKeyword.NECROMANCY: "死灵术|唤灵",
    AbilityKeyword.REANIMATE: "亡者召还",
    AbilityKeyword.STORM: "疾驰",
    AbilityKeyword.RUSH: "突进",
    AbilityKeyword.WARD: "守护",
    AbilityKeyword.BANE: "必杀",
    AbilityKeyword.AMBUSH: "潜行",
    AbilityKeyword.DRAIN: "吸血",
    AbilityKeyword.COUNTDOWN: "倒数",
    AbilityKeyword.BARRIER: "屏障",
    AbilityKeyword.FANFARE: "入场曲",
    AbilityKeyword.LAST_WORDS: "谢幕曲",
    AbilityKeyword.ON_EVOLVE: "进化时",
    AbilityKeyword.ON_SUPER_EVOLVE: "超进化",
    AbilityKeyword.ON_ATTACK: "攻击时",
    AbilityKeyword.ON_CLASH: "交战时",
    AbilityKeyword.ENHANCE: "爆能强化",
    AbilityKeyword.ACCELERATE: "激奏",
    AbilityKeyword.CRYSTALLIZE: "结晶",
    AbilityKeyword.CHOOSE: "选择一项|模式",
    AbilityKeyword.FUSION: "融合",
    AbilityKeyword.INVOCATION: "瞬念召唤",
    AbilityKeyword.EMBLEM: "纹章",
    AbilityKeyword.FAITH: "信仰",
}


def primitive_status(keyword: AbilityKeyword) -> str:
    manual = _MANUAL_PRIMITIVE_STATUS.get(keyword)
    if manual is not None:
        return manual
    pattern = _PRIMITIVE_PATTERN_BY_KEYWORD.get(keyword)
    if pattern is None:
        return "unmapped"
    info = PRIMITIVE_KEYWORD_MAP.get(pattern)
    if info is None:
        return "unmapped"
    return "covered" if info["covered"] else "missing"


def main() -> None:
    print(f"{'能力':<10} {'Handler':<12} {'Primitive':<10} 触发事件")
    print("-" * 76)
    for definition in ABILITY_DEFINITIONS:
        events = ", ".join(event.value for event in definition.events) or "static"
        print(
            f"{definition.keyword.value:<10} "
            f"{definition.status.value:<12} "
            f"{primitive_status(definition.keyword):<10} "
            f"{events}"
        )


if __name__ == "__main__":
    main()
