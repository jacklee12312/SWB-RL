from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from swb.db.repository import CardDefinition
    from swb.engine.model import Unit
    from swb.engine.card_rules import Trigger


class AbilityKeyword(str, Enum):
    COMBO = "连击"
    COOPERATION = "协作"
    SPELLBOOST = "魔力增幅"
    EARTH_RITE = "土之秘术"
    EARTH_SIGIL = "土之印"
    OVERFLOW = "觉醒"
    NECROMANCY = "死灵术"
    REANIMATE = "亡者召还"

    STORM = "疾驰"
    RUSH = "突进"
    WARD = "守护"
    BANE = "必杀"
    AMBUSH = "潜行"
    DRAIN = "吸血"
    COUNTDOWN = "倒数"
    INTIMIDATE = "威慑"
    AURA = "灵气"
    BARRIER = "屏障"
    INVOCATION = "瞬念召唤"

    FANFARE = "入场曲"
    LAST_WORDS = "谢幕曲"
    ON_EVOLVE = "进化时"
    ON_SUPER_EVOLVE = "超进化时"
    ON_ATTACK = "攻击时"
    ON_CLASH = "交战时"
    ENHANCE = "爆能强化"
    ACCELERATE = "激奏"
    CRYSTALLIZE = "结晶"
    CHOOSE = "模式"
    FUSION = "融合"
    ACTIVATE = "策动"
    EMBLEM = "纹章"
    FAITH = "信仰"
    UNION_BURST = "奥义"


class AbilityEvent(str, Enum):
    CHECK_PLAY = "check_play"
    CARD_PLAYED = "card_played"
    FOLLOWER_SUMMONED = "follower_summoned"
    FOLLOWER_EVOLVED = "follower_evolved"
    FOLLOWER_SUPER_EVOLVED = "follower_super_evolved"
    BEFORE_ATTACK = "before_attack"
    BEFORE_COMBAT = "before_combat"
    AFTER_DAMAGE = "after_damage"
    FOLLOWER_DESTROYED = "follower_destroyed"
    TURN_STARTED = "turn_started"
    TURN_ENDED = "turn_ended"
    CARD_DRAWN = "card_drawn"


class AbilityStatus(str, Enum):
    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    PLACEHOLDER = "placeholder"


@dataclass(frozen=True)
class AbilityDefinition:
    keyword: AbilityKeyword
    events: frozenset[AbilityEvent]
    status: AbilityStatus
    handler_name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class AbilityContext:
    event: AbilityEvent
    player_index: int
    source: Unit | CardDefinition | None = None
    target: Unit | None = None


@dataclass(frozen=True)
class PlaceholderAbilityEvent:
    turn: int
    player_index: int
    card_id: int
    card_name: str
    ability: AbilityKeyword
    event: AbilityEvent


class AbilityHost(Protocol):
    turn: int
    placeholder_ability_events: list[PlaceholderAbilityEvent]


def _definition(
    keyword: AbilityKeyword,
    events: tuple[AbilityEvent, ...],
    handler_name: str,
    *,
    status: AbilityStatus = AbilityStatus.PLACEHOLDER,
    aliases: tuple[str, ...] = (),
) -> AbilityDefinition:
    return AbilityDefinition(keyword, frozenset(events), status, handler_name, aliases)


ABILITY_DEFINITIONS = (
    _definition(AbilityKeyword.COMBO, (AbilityEvent.CHECK_PLAY,), "handle_combo"),
    _definition(
        AbilityKeyword.COOPERATION, (AbilityEvent.CHECK_PLAY,), "handle_cooperation"
    ),
    _definition(
        AbilityKeyword.SPELLBOOST,
        (AbilityEvent.CARD_PLAYED,),
        "handle_spellboost",
        aliases=("魔力增幅时",),
    ),
    _definition(
        AbilityKeyword.EARTH_RITE,
        (AbilityEvent.CARD_PLAYED,),
        "handle_earth_rite",
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.EARTH_SIGIL,
        (),
        "handle_earth_sigil",
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(AbilityKeyword.OVERFLOW, (AbilityEvent.CHECK_PLAY,), "handle_overflow"),
    _definition(
        AbilityKeyword.NECROMANCY,
        (AbilityEvent.CARD_PLAYED,),
        "handle_necromancy",
        aliases=("唤灵",),
    ),
    _definition(
        AbilityKeyword.REANIMATE, (AbilityEvent.CARD_PLAYED,), "handle_reanimate"
    ),
    _definition(
        AbilityKeyword.STORM,
        (),
        "handle_storm",
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.RUSH, (), "handle_rush", status=AbilityStatus.IMPLEMENTED
    ),
    _definition(
        AbilityKeyword.WARD, (), "handle_ward", status=AbilityStatus.IMPLEMENTED
    ),
    _definition(
        AbilityKeyword.BANE,
        (AbilityEvent.AFTER_DAMAGE,),
        "handle_bane",
        aliases=("毁灭",),
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.AMBUSH, (AbilityEvent.BEFORE_ATTACK,), "handle_ambush",
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.DRAIN,
        (AbilityEvent.AFTER_DAMAGE,),
        "handle_drain",
        aliases=("虹吸",),
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.COUNTDOWN,
        (AbilityEvent.TURN_STARTED,),
        "handle_countdown",
        status=AbilityStatus.PARTIAL,
        aliases=("吟唱",),
    ),
    _definition(
        AbilityKeyword.INTIMIDATE,
        (),
        "handle_intimidate",
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.AURA,
        (),
        "handle_aura",
        aliases=("无敌", "光纹", "光紋"),
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.BARRIER,
        (AbilityEvent.AFTER_DAMAGE,),
        "handle_barrier",
        aliases=("障壁",),
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.INVOCATION,
        (AbilityEvent.TURN_STARTED,),
        "handle_invocation",
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.FANFARE,
        (AbilityEvent.FOLLOWER_SUMMONED,),
        "handle_fanfare",
        status=AbilityStatus.PARTIAL,
    ),
    _definition(
        AbilityKeyword.LAST_WORDS,
        (AbilityEvent.FOLLOWER_DESTROYED,),
        "handle_last_words",
        status=AbilityStatus.PARTIAL,
    ),
    _definition(
        AbilityKeyword.ON_EVOLVE,
        (AbilityEvent.FOLLOWER_EVOLVED,),
        "handle_on_evolve",
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.ON_SUPER_EVOLVE,
        (AbilityEvent.FOLLOWER_SUPER_EVOLVED,),
        "handle_on_super_evolve",
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.ON_ATTACK, (AbilityEvent.BEFORE_ATTACK,), "handle_on_attack",
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(
        AbilityKeyword.ON_CLASH, (AbilityEvent.BEFORE_COMBAT,), "handle_on_clash",
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(AbilityKeyword.ENHANCE, (AbilityEvent.CHECK_PLAY,), "handle_enhance"),
    _definition(AbilityKeyword.ACCELERATE, (AbilityEvent.CHECK_PLAY,), "handle_accelerate"),
    _definition(
        AbilityKeyword.CRYSTALLIZE, (AbilityEvent.CHECK_PLAY,), "handle_crystallize"
    ),
    _definition(AbilityKeyword.CHOOSE, (AbilityEvent.CHECK_PLAY,), "handle_choose"),
    _definition(
        AbilityKeyword.FUSION,
        (AbilityEvent.CHECK_PLAY,),
        "handle_fusion",
        status=AbilityStatus.PARTIAL,
    ),
    _definition(
        AbilityKeyword.ACTIVATE,
        (AbilityEvent.CARD_PLAYED,),
        "handle_activate",
        aliases=("启动",),
        status=AbilityStatus.IMPLEMENTED,
    ),
    _definition(AbilityKeyword.EMBLEM, (AbilityEvent.CARD_PLAYED,), "handle_emblem"),
    _definition(
        AbilityKeyword.FAITH,
        (AbilityEvent.CARD_PLAYED,),
        "handle_faith",
        status=AbilityStatus.PARTIAL,
    ),
    _definition(
        AbilityKeyword.UNION_BURST,
        (AbilityEvent.CHECK_PLAY,),
        "handle_union_burst",
        aliases=("解放奥义",),
        status=AbilityStatus.IMPLEMENTED,
    ),
)

ABILITY_REGISTRY = {definition.keyword: definition for definition in ABILITY_DEFINITIONS}
ABILITY_NAME_MAP = {
    name: definition.keyword
    for definition in ABILITY_DEFINITIONS
    for name in (definition.keyword.value, *definition.aliases)
}

RUNTIME_UNIT_KEYWORDS = frozenset(
    {
        AbilityKeyword.STORM.value,
        AbilityKeyword.RUSH.value,
        AbilityKeyword.WARD.value,
        AbilityKeyword.BANE.value,
        AbilityKeyword.AMBUSH.value,
        AbilityKeyword.DRAIN.value,
        AbilityKeyword.BARRIER.value,
        AbilityKeyword.INTIMIDATE.value,
        AbilityKeyword.AURA.value,
    }
)


def normalize_keyword_name(name: str, *, strict: bool = False) -> str:
    keyword = ABILITY_NAME_MAP.get(name)
    if keyword is not None:
        return keyword.value
    if strict:
        raise ValueError(f"Unknown ability keyword: {name!r}")
    return name


def normalize_abilities(names: frozenset[str] | set[str] | list[str]) -> frozenset[AbilityKeyword]:
    return frozenset(
        ability
        for name in names
        if (ability := ABILITY_NAME_MAP.get(name)) is not None
    )


class AbilityHandlers:
    """Stable extension points for every documented SWB ability.

    Placeholder handlers deliberately do not mutate game state. They record
    that a relevant event occurred so unsupported mechanics remain visible.
    """

    def __init__(self, environment: AbilityHost):
        self.environment = environment

    def dispatch(self, context: AbilityContext) -> None:
        if context.source is None:
            return
        card = (
            context.source.definition
            if hasattr(context.source, "definition")
            else context.source
        )
        for ability in card.abilities:
            definition = ABILITY_REGISTRY[ability]
            if context.event not in definition.events:
                continue
            handler: Callable[[AbilityContext], None] = getattr(
                self, definition.handler_name
            )
            handler(context)

    def _placeholder(self, context: AbilityContext, ability: AbilityKeyword) -> None:
        source = context.source
        if source is None:
            return
        card = source.definition if hasattr(source, "definition") else source
        self.environment.placeholder_ability_events.append(
            PlaceholderAbilityEvent(
                turn=self.environment.turn,
                player_index=context.player_index,
                card_id=card.card_id,
                card_name=card.name,
                ability=ability,
                event=context.event,
            )
        )
        recorder = getattr(
            self.environment,
            "_record_runtime_diagnostic",
            None,
        )
        if callable(recorder):
            recorder(
                "placeholder",
                card_id=card.card_id,
                detail=f"{ability.name}:{context.event.value}",
            )
            recorder(
                "unsupported",
                card_id=card.card_id,
                detail=f"{ability.name}:{context.event.value}",
            )

    def handle_combo(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.COMBO):
            self._placeholder(context, AbilityKeyword.COMBO)

    def handle_cooperation(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.COOPERATION):
            self._placeholder(context, AbilityKeyword.COOPERATION)

    def handle_spellboost(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.SPELLBOOST):
            self._placeholder(context, AbilityKeyword.SPELLBOOST)

    def handle_earth_rite(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.EARTH_RITE):
            self._placeholder(context, AbilityKeyword.EARTH_RITE)

    def handle_earth_sigil(self, context: AbilityContext) -> None:
        pass

    def handle_overflow(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.OVERFLOW):
            self._placeholder(context, AbilityKeyword.OVERFLOW)

    def handle_necromancy(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.NECROMANCY):
            self._placeholder(context, AbilityKeyword.NECROMANCY)

    def handle_reanimate(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.REANIMATE):
            self._placeholder(context, AbilityKeyword.REANIMATE)

    def handle_storm(self, context: AbilityContext) -> None:
        pass

    def handle_rush(self, context: AbilityContext) -> None:
        pass

    def handle_ward(self, context: AbilityContext) -> None:
        pass

    def handle_bane(self, context: AbilityContext) -> None:
        pass

    def handle_ambush(self, context: AbilityContext) -> None:
        pass

    def handle_drain(self, context: AbilityContext) -> None:
        pass

    def handle_countdown(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.COUNTDOWN):
            self._placeholder(context, AbilityKeyword.COUNTDOWN)

    def handle_intimidate(self, context: AbilityContext) -> None:
        pass

    def handle_aura(self, context: AbilityContext) -> None:
        pass

    def handle_barrier(self, context: AbilityContext) -> None:
        pass

    def handle_invocation(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.INVOCATION):
            self._placeholder(context, AbilityKeyword.INVOCATION)

    def handle_fanfare(self, context: AbilityContext) -> None:
        source = context.source
        card = (
            source.definition
            if source is not None and hasattr(source, "definition")
            else source
        )
        covered = getattr(self.environment, "_is_ability_covered", None)
        if (
            card is not None
            and not card.fanfare_effects
            and (covered is None or not covered(context, AbilityKeyword.FANFARE))
        ):
            self._placeholder(context, AbilityKeyword.FANFARE)

    def handle_last_words(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.LAST_WORDS):
            self._placeholder(context, AbilityKeyword.LAST_WORDS)

    def handle_on_evolve(self, context: AbilityContext) -> None:
        from swb.engine.card_rules import Trigger
        self._dispatch_trigger(context, Trigger.EVOLVE)

    def handle_on_super_evolve(self, context: AbilityContext) -> None:
        from swb.engine.card_rules import Trigger
        self._dispatch_trigger(context, Trigger.SUPER_EVOLVE)

    def handle_on_attack(self, context: AbilityContext) -> None:
        from swb.engine.card_rules import Trigger
        self._dispatch_trigger(context, Trigger.ATTACK)

    def handle_on_clash(self, context: AbilityContext) -> None:
        from swb.engine.card_rules import Trigger
        self._dispatch_trigger(context, Trigger.CLASH)

    def _dispatch_trigger(self, context: AbilityContext, trigger: str) -> None:
        engine = getattr(self.environment, '_execute_trigger_rules', None)
        if engine is not None:
            engine(trigger, context)

    def handle_enhance(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.ENHANCE):
            self._placeholder(context, AbilityKeyword.ENHANCE)

    def handle_accelerate(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.ACCELERATE):
            self._placeholder(context, AbilityKeyword.ACCELERATE)

    def handle_crystallize(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.CRYSTALLIZE):
            self._placeholder(context, AbilityKeyword.CRYSTALLIZE)

    def handle_choose(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.CHOOSE):
            self._placeholder(context, AbilityKeyword.CHOOSE)

    def handle_fusion(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.FUSION):
            self._placeholder(context, AbilityKeyword.FUSION)

    def handle_activate(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.ACTIVATE):
            self._placeholder(context, AbilityKeyword.ACTIVATE)

    def handle_emblem(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.EMBLEM):
            self._placeholder(context, AbilityKeyword.EMBLEM)

    def handle_faith(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.FAITH):
            self._placeholder(context, AbilityKeyword.FAITH)

    def handle_union_burst(self, context: AbilityContext) -> None:
        covered = getattr(self.environment, "_is_ability_covered", None)
        if covered is None or not covered(context, AbilityKeyword.UNION_BURST):
            self._placeholder(context, AbilityKeyword.UNION_BURST)
