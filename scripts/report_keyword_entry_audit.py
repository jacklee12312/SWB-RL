"""Audit runtime keyword provenance, entry methods, and attack-mask contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import RUNTIME_UNIT_KEYWORDS, normalize_keyword_name
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import (
    Attack,
    Choose,
    Evolve,
    PlayCard,
    SuperEvolve,
)
from swb.engine.effects import (
    DeckFilter,
    EffectKind,
    EffectOperation,
    ModifierDuration,
    TargetKind,
)
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.play_modes import PlayModeDefinition
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import (
    AttackRestriction,
    DeathCause,
    DestroyedFollowerRecord,
    HandCard,
    Phase,
    TargetingRestriction,
    Unit,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = Path("data/cards.sqlite3")
DEFAULT_RULES = Path("data/rules")
DEFAULT_CLOSURE = Path(
    "data/reports/card_bug_audit/training_deck_card_closure.json"
)
DEFAULT_COVERAGE = Path("data/reports/rule_coverage.json")
DEFAULT_OUTPUT = Path(
    "data/reports/card_bug_audit/keyword_entry_audit.json"
)
DEFAULT_MARKDOWN = Path(
    "data/reports/card_bug_audit/keyword_entry_audit.md"
)

KEYWORDS = tuple(sorted(RUNTIME_UNIT_KEYWORDS))
KEYWORD_EFFECT_KINDS = frozenset(
    {
        EffectKind.ADD_KEYWORD,
        EffectKind.ADD_RANDOM_KEYWORDS,
        EffectKind.REMOVE_KEYWORD,
        EffectKind.REMOVE_ALL_ABILITIES,
        EffectKind.GRANT_ATTACKS_PER_TURN,
        EffectKind.ADD_ATTACK_RESTRICTION,
        EffectKind.REMOVE_ATTACK_RESTRICTION,
        EffectKind.ADD_TARGETING_RESTRICTION,
        EffectKind.REMOVE_TARGETING_RESTRICTION,
    }
)
ENTRY_METHODS = (
    "normal_play",
    "enhance_play",
    "direct_summon",
    "summon_from_deck",
    "reanimate",
    "definition_copy",
    "exact_copy",
    "transform",
    "normal_evolution",
    "super_evolution",
    "effect_evolution",
    "effect_super_evolution",
)
SOURCE_CATEGORIES = (
    "printed_intrinsic",
    "conditional_declared",
    "runtime_permanent",
    "runtime_temporary",
    "evolution_grant",
    "attack_capacity",
    "attack_restriction",
    "targeting_restriction",
)


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_card(
    card_id: int,
    *,
    keywords: Iterable[str] = (),
    card_type: str = "随从",
    cost: int = 1,
    class_id: int = 1,
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=class_id,
        class_name="精灵",
        name=f"audit-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=2 if card_type == "随从" else None,
        life=3 if card_type == "随从" else None,
        keywords=frozenset(keywords),
        support_level="basic",
        is_collectible=True,
    )


def _engine(
    *,
    rulebook: RuleBook | None = None,
    resolver=None,
    seed: int = 15001,
) -> GameEngine:
    engine = GameEngine(
        [_synthetic_card(810000 + index) for index in range(40)],
        [_synthetic_card(820000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook or RuleBook(),
        card_resolver=resolver,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    engine.state.phase = Phase.MAIN
    engine.state.active_player = 0
    engine.players[0].hand.clear()
    engine.players[0].hand_entity_ids.clear()
    engine.players[1].hand.clear()
    engine.players[1].hand_entity_ids.clear()
    engine.players[0].board.clear()
    engine.players[1].board.clear()
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _put_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
        printed_keyword_overrides=set(
            engine.rulebook.non_intrinsic_keywords(definition.card_id)
        ),
    )
    engine.players[0].hand.append(hand_card)
    engine.players[0].hand_entity_ids.append(hand_card.entity_id)
    return hand_card


def _choose_entity(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    if request is None:
        raise AssertionError("expected a pending target choice")
    option = next(
        candidate
        for candidate in request.options
        if candidate.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _state_contracts() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for index, keyword in enumerate(KEYWORDS):
        printed = Unit.summon(
            _synthetic_card(830000 + index, keywords=(keyword,))
        )
        printed_ok = printed.has_keyword(keyword)

        permanent = Unit.summon(_synthetic_card(831000 + index))
        permanent.add_keyword(keyword)
        added_ok = permanent.has_keyword(keyword)
        permanent.remove_keyword(keyword)
        removed_ok = not permanent.has_keyword(keyword)

        temporary = Unit.summon(_synthetic_card(832000 + index))
        temporary.add_keyword(
            keyword,
            duration=ModifierDuration.UNTIL_END_OF_TURN.value,
            expires_for_player=0,
        )
        temporary_added = temporary.has_keyword(keyword)
        temporary.expire_keywords(
            ModifierDuration.UNTIL_END_OF_TURN.value,
            0,
        )
        temporary_expired = not temporary.has_keyword(keyword)

        silenced = Unit.summon(
            _synthetic_card(833000 + index, keywords=(keyword,))
        )
        silenced.add_keyword(keyword)
        silenced.remove_all_abilities()
        silence_ok = not silenced.has_keyword(keyword)

        passed = all(
            (
                printed_ok,
                added_ok,
                removed_ok,
                temporary_added,
                temporary_expired,
                silence_ok,
            )
        )
        cases.append(
            {
                "keyword": keyword,
                "printed_state": printed_ok,
                "permanent_add": added_ok,
                "permanent_remove": removed_ok,
                "temporary_add": temporary_added,
                "temporary_expiry": temporary_expired,
                "silence_clears": silence_ok,
                "passed": passed,
            }
        )
    return cases


def _entry_contracts() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    summon_causes = (
        "direct_summon",
        "summon_from_deck",
        "reanimate",
        "copy_summon",
    )
    for index, keyword in enumerate(KEYWORDS):
        definition = _synthetic_card(
            840000 + index,
            keywords=(keyword,),
        )
        fanfare_rule = CardRule(
            definition.card_id,
            Trigger.FANFARE,
            (
                EffectOperation(
                    kind=EffectKind.DAMAGE_LEADER,
                    target=TargetKind.ENEMY_LEADER,
                    amount=1,
                ),
            ),
        )

        normal = _engine(
            rulebook=RuleBook((fanfare_rule,)),
            seed=15100 + index,
        )
        _put_hand(normal, definition)
        normal.apply(PlayCard(0, 0, "normal"))
        normal_unit = normal.players[0].board[0]
        normal_ok = (
            normal_unit.has_keyword(keyword)
            and normal.players[1].health == 19
        )

        enhance_rulebook = RuleBook((fanfare_rule,))
        enhance_rulebook._play_modes = {
            definition.card_id: (
                PlayModeDefinition(
                    mode_id="enhance_2",
                    mode_type="enhance",
                    cost=2,
                ),
            )
        }
        enhanced = _engine(
            rulebook=enhance_rulebook,
            seed=15200 + index,
        )
        _put_hand(enhanced, definition)
        enhanced.apply(PlayCard(0, 0, "enhance_2"))
        enhance_ok = (
            enhanced.players[0].board[0].has_keyword(keyword)
            and enhanced.players[1].health == 19
        )

        cause_results: dict[str, bool] = {}
        for offset, cause in enumerate(summon_causes):
            summoned = _engine(
                rulebook=RuleBook((fanfare_rule,)),
                seed=15300 + index * 10 + offset,
            )
            unit = summoned._summon_follower_to_board(
                0,
                definition,
                summon_cause=cause,
                origin=(
                    CardOrigin.REANIMATED
                    if cause == "reanimate"
                    else CardOrigin.GENERATED
                ),
            )
            cause_results[cause] = bool(
                unit is not None and unit.has_keyword(keyword)
                and summoned.players[1].health == 20
            )

        replacement = _synthetic_card(
            841000 + index,
            keywords=(keyword,),
        )
        transform_spell = _synthetic_card(
            842000 + index,
            card_type="法术",
        )
        transform_rule = CardRule(
            transform_spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    kind=EffectKind.TRANSFORM,
                    target=TargetKind.OWN_UNIT,
                    card_id=replacement.card_id,
                    requires_target=True,
                ),
            ),
        )
        replacement_fanfare = CardRule(
            replacement.card_id,
            Trigger.FANFARE,
            (
                EffectOperation(
                    kind=EffectKind.DAMAGE_LEADER,
                    target=TargetKind.ENEMY_LEADER,
                    amount=1,
                ),
            ),
        )
        transformed = _engine(
            rulebook=RuleBook((transform_rule, replacement_fanfare)),
            resolver=lambda card_id, replacement=replacement: (
                replacement if card_id == replacement.card_id else None
            ),
            seed=15400 + index,
        )
        target = transformed._summon_follower_to_board(
            0,
            _synthetic_card(843000 + index),
            summon_cause="direct_summon",
        )
        if target is None:
            raise AssertionError("failed to create transform target")
        _put_hand(transformed, transform_spell)
        transformed.apply(PlayCard(0, 0))
        _choose_entity(transformed, target.entity_id)
        transform_ok = (
            target.has_keyword(keyword)
            and transformed.players[1].health == 20
        )

        passed = (
            normal_ok
            and enhance_ok
            and transform_ok
            and all(cause_results.values())
        )
        cases.append(
            {
                "keyword": keyword,
                "normal_play": normal_ok,
                "enhance_play": enhance_ok,
                "summon_causes": cause_results,
                "transform": transform_ok,
                "passed": passed,
            }
        )
    return cases


def _special_entry_contracts() -> list[dict[str, object]]:
    """Execute zone-backed summon and copy paths for every runtime keyword."""

    cases: list[dict[str, object]] = []
    for index, keyword in enumerate(KEYWORDS):
        definition = _synthetic_card(
            844000 + index,
            keywords=(keyword,),
            cost=2,
        )
        fanfare_rule = CardRule(
            definition.card_id,
            Trigger.FANFARE,
            (
                EffectOperation(
                    kind=EffectKind.DAMAGE_LEADER,
                    target=TargetKind.ENEMY_LEADER,
                    amount=1,
                ),
            ),
        )

        deck_spell = _synthetic_card(
            845000 + index,
            card_type="法术",
        )
        deck_rule = CardRule(
            deck_spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    kind=EffectKind.SUMMON_FROM_DECK,
                    target=TargetKind.OWN_LEADER,
                    amount=1,
                    deck_filter=DeckFilter(
                        card_type="随从",
                        card_id=definition.card_id,
                    ),
                ),
            ),
        )
        deck_engine = _engine(
            rulebook=RuleBook((deck_rule, fanfare_rule)),
            seed=16000 + index,
        )
        deck_engine.players[0].deck = [definition]
        _put_hand(deck_engine, deck_spell)
        deck_engine.apply(PlayCard(0, 0))
        deck_summoned = deck_engine.players[0].board[0]
        deck_ok = (
            deck_summoned.has_keyword(keyword)
            and deck_summoned.origin is CardOrigin.DECK
            and not deck_engine.players[0].deck
            and deck_engine.players[1].health == 20
        )

        reanimate_spell = _synthetic_card(
            846000 + index,
            card_type="法术",
        )
        reanimate_rule = CardRule(
            reanimate_spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    kind=EffectKind.REANIMATE,
                    target=TargetKind.OWN_LEADER,
                    amount=10,
                ),
            ),
        )
        reanimate_engine = _engine(
            rulebook=RuleBook((reanimate_rule, fanfare_rule)),
            seed=16100 + index,
        )
        reanimate_engine.state.destroyed_followers.append(
            DestroyedFollowerRecord(
                definition=definition,
                owner=0,
                death_sequence=1,
                cause=DeathCause.COMBAT,
            )
        )
        _put_hand(reanimate_engine, reanimate_spell)
        reanimate_engine.apply(PlayCard(0, 0))
        reanimated = reanimate_engine.players[0].board[0]
        reanimate_ok = (
            reanimated.has_keyword(keyword)
            and reanimated.origin is CardOrigin.REANIMATED
            and reanimate_engine.players[1].health == 20
        )

        definition_copy_spell = _synthetic_card(
            847000 + index,
            card_type="法术",
        )
        definition_copy_rule = CardRule(
            definition_copy_spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    kind=EffectKind.SUMMON_COPY,
                    target=TargetKind.OWN_UNIT,
                    requires_target=True,
                ),
            ),
        )
        definition_copy_engine = _engine(
            rulebook=RuleBook((definition_copy_rule, fanfare_rule)),
            seed=16200 + index,
        )
        definition_source = (
            definition_copy_engine._summon_follower_to_board(
                0,
                definition,
                summon_cause="direct_summon",
            )
        )
        if definition_source is None:
            raise AssertionError("failed to create definition-copy source")
        _put_hand(definition_copy_engine, definition_copy_spell)
        definition_copy_engine.apply(PlayCard(0, 0))
        _choose_entity(
            definition_copy_engine,
            definition_source.entity_id,
        )
        definition_copy = next(
            unit
            for unit in definition_copy_engine.players[0].board
            if unit.entity_id != definition_source.entity_id
        )
        definition_copy_ok = (
            definition_copy.has_keyword(keyword)
            and definition_copy_engine.players[1].health == 20
        )

        exact_source_definition = _synthetic_card(848000 + index)
        exact_copy_spell = _synthetic_card(
            849000 + index,
            card_type="法术",
        )
        exact_copy_rule = CardRule(
            exact_copy_spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    kind=EffectKind.SUMMON_EXACT_COPY,
                    target=TargetKind.OWN_UNIT,
                    requires_target=True,
                ),
            ),
        )
        exact_copy_engine = _engine(
            rulebook=RuleBook((exact_copy_rule,)),
            seed=16300 + index,
        )
        exact_source = exact_copy_engine._summon_follower_to_board(
            0,
            exact_source_definition,
            summon_cause="direct_summon",
        )
        if exact_source is None:
            raise AssertionError("failed to create exact-copy source")
        exact_source.add_keyword(keyword)
        _put_hand(exact_copy_engine, exact_copy_spell)
        exact_copy_engine.apply(PlayCard(0, 0))
        _choose_entity(exact_copy_engine, exact_source.entity_id)
        exact_copy = next(
            unit
            for unit in exact_copy_engine.players[0].board
            if unit.entity_id != exact_source.entity_id
        )
        exact_copy_ok = (
            exact_copy.has_keyword(keyword)
            and keyword in exact_copy.permanent_keywords
        )

        cases.append(
            {
                "keyword": keyword,
                "summon_from_deck": deck_ok,
                "reanimate": reanimate_ok,
                "definition_copy": definition_copy_ok,
                "exact_copy_preserves_dynamic": exact_copy_ok,
                "fanfare_skipped_for_non_play_entries": (
                    deck_engine.players[1].health == 20
                    and reanimate_engine.players[1].health == 20
                    and definition_copy_engine.players[1].health == 20
                ),
                "passed": (
                    deck_ok
                    and reanimate_ok
                    and definition_copy_ok
                    and exact_copy_ok
                ),
            }
        )
    return cases


def _evolution_contracts() -> list[dict[str, object]]:
    """Distinguish keyword abilities from self-evolved state triggers."""

    follower = _synthetic_card(
        854001,
        keywords=("进化时", "超进化时"),
    )
    normal_effect_spell = _synthetic_card(854002, card_type="法术")
    super_effect_spell = _synthetic_card(854003, card_type="法术")
    rules = (
        CardRule(
            follower.card_id,
            Trigger.EVOLVE,
            (
                EffectOperation(
                    kind=EffectKind.ADD_KEYWORD,
                    target=TargetKind.SELF,
                    keyword="疾驰",
                ),
            ),
        ),
        CardRule(
            follower.card_id,
            Trigger.SUPER_EVOLVE,
            (
                EffectOperation(
                    kind=EffectKind.ADD_KEYWORD,
                    target=TargetKind.SELF,
                    keyword="守护",
                ),
            ),
        ),
        CardRule(
            follower.card_id,
            Trigger.SELF_EVOLVED,
            (
                EffectOperation(
                    kind=EffectKind.ADD_KEYWORD,
                    target=TargetKind.SELF,
                    keyword="必杀",
                ),
            ),
        ),
        CardRule(
            follower.card_id,
            Trigger.SELF_SUPER_EVOLVED,
            (
                EffectOperation(
                    kind=EffectKind.ADD_KEYWORD,
                    target=TargetKind.SELF,
                    keyword="屏障",
                ),
            ),
        ),
        CardRule(
            normal_effect_spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    kind=EffectKind.EVOLVE_UNIT,
                    target=TargetKind.OWN_UNIT,
                    requires_target=True,
                ),
            ),
        ),
        CardRule(
            super_effect_spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    kind=EffectKind.SUPER_EVOLVE_UNIT,
                    target=TargetKind.OWN_UNIT,
                    requires_target=True,
                ),
            ),
        ),
    )
    rulebook = RuleBook(rules)

    manual_normal = _engine(rulebook=rulebook, seed=16401)
    normal_unit = manual_normal._summon_follower_to_board(
        0,
        follower,
        summon_cause="direct_summon",
    )
    if normal_unit is None:
        raise AssertionError("failed to create normal evolution source")
    manual_normal.players[0].turns_started = (
        manual_normal.config.evolution_unlock_turn
    )
    manual_normal.players[0].evolution_points = 1
    manual_normal.apply(Evolve(0, normal_unit.entity_id))
    manual_normal_ok = (
        normal_unit.evolved
        and not normal_unit.super_evolved
        and normal_unit.has_keyword("疾驰")
        and normal_unit.has_keyword("必杀")
        and not normal_unit.has_keyword("守护")
        and not normal_unit.has_keyword("屏障")
    )

    manual_super = _engine(rulebook=rulebook, seed=16402)
    super_unit = manual_super._summon_follower_to_board(
        0,
        follower,
        summon_cause="direct_summon",
    )
    if super_unit is None:
        raise AssertionError("failed to create super evolution source")
    manual_super.players[0].turns_started = (
        manual_super.config.first_player_super_evolution_unlock_turn
    )
    manual_super.players[0].super_evolution_points = 1
    manual_super.apply(SuperEvolve(0, super_unit.entity_id))
    manual_super_ok = (
        super_unit.evolved
        and super_unit.super_evolved
        and all(
            super_unit.has_keyword(keyword)
            for keyword in ("疾驰", "守护", "必杀", "屏障")
        )
    )

    effect_normal = _engine(rulebook=rulebook, seed=16403)
    effect_normal_unit = effect_normal._summon_follower_to_board(
        0,
        follower,
        summon_cause="direct_summon",
    )
    if effect_normal_unit is None:
        raise AssertionError("failed to create effect evolution source")
    _put_hand(effect_normal, normal_effect_spell)
    effect_normal.apply(PlayCard(0, 0))
    _choose_entity(effect_normal, effect_normal_unit.entity_id)
    effect_normal_ok = (
        effect_normal_unit.evolved
        and not effect_normal_unit.super_evolved
        and not effect_normal_unit.has_keyword("疾驰")
        and effect_normal_unit.has_keyword("必杀")
        and not effect_normal_unit.has_keyword("守护")
        and not effect_normal_unit.has_keyword("屏障")
    )

    effect_super = _engine(rulebook=rulebook, seed=16404)
    effect_super_unit = effect_super._summon_follower_to_board(
        0,
        follower,
        summon_cause="direct_summon",
    )
    if effect_super_unit is None:
        raise AssertionError("failed to create effect super evolution source")
    _put_hand(effect_super, super_effect_spell)
    effect_super.apply(PlayCard(0, 0))
    _choose_entity(effect_super, effect_super_unit.entity_id)
    effect_super_ok = (
        effect_super_unit.evolved
        and effect_super_unit.super_evolved
        and not effect_super_unit.has_keyword("疾驰")
        and not effect_super_unit.has_keyword("守护")
        and effect_super_unit.has_keyword("必杀")
        and effect_super_unit.has_keyword("屏障")
    )

    return [
        {
            "evolution_method": "normal_evolution",
            "keyword_evolve_trigger": True,
            "keyword_super_evolve_trigger": False,
            "self_evolved_trigger": True,
            "self_super_evolved_trigger": False,
            "passed": manual_normal_ok,
        },
        {
            "evolution_method": "super_evolution",
            "keyword_evolve_trigger": True,
            "keyword_super_evolve_trigger": True,
            "self_evolved_trigger": True,
            "self_super_evolved_trigger": True,
            "passed": manual_super_ok,
        },
        {
            "evolution_method": "effect_evolution",
            "keyword_evolve_trigger": False,
            "keyword_super_evolve_trigger": False,
            "self_evolved_trigger": True,
            "self_super_evolved_trigger": False,
            "passed": effect_normal_ok,
        },
        {
            "evolution_method": "effect_super_evolution",
            "keyword_evolve_trigger": False,
            "keyword_super_evolve_trigger": False,
            "self_evolved_trigger": True,
            "self_super_evolved_trigger": True,
            "passed": effect_super_ok,
        },
    ]


def _zone_reset_contracts() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for index, keyword in enumerate(KEYWORDS):
        source = _synthetic_card(850000 + index)
        return_spell = _synthetic_card(
            851000 + index,
            card_type="法术",
        )
        rule = CardRule(
            return_spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    kind=EffectKind.RETURN_TO_HAND,
                    target=TargetKind.OWN_UNIT,
                    requires_target=True,
                ),
            ),
        )
        returned = _engine(
            rulebook=RuleBook((rule,)),
            seed=15500 + index,
        )
        unit = returned._summon_follower_to_board(
            0,
            source,
            summon_cause="direct_summon",
        )
        if unit is None:
            raise AssertionError("failed to create return target")
        unit.add_keyword(keyword)
        _put_hand(returned, return_spell)
        returned.apply(PlayCard(0, 0))
        _choose_entity(returned, unit.entity_id)
        hand_card = next(
            card
            for card in returned.players[0].hand
            if isinstance(card, HandCard) and card.card_id == source.card_id
        )
        return_resets = not hand_card.has_keyword(keyword)

        replacement = _synthetic_card(852000 + index)
        transform_spell = _synthetic_card(
            853000 + index,
            card_type="法术",
        )
        transform_rule = CardRule(
            transform_spell.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    kind=EffectKind.TRANSFORM,
                    target=TargetKind.OWN_UNIT,
                    card_id=replacement.card_id,
                    requires_target=True,
                ),
            ),
        )
        transformed = _engine(
            rulebook=RuleBook((transform_rule,)),
            resolver=lambda card_id, replacement=replacement: (
                replacement if card_id == replacement.card_id else None
            ),
            seed=15600 + index,
        )
        transform_target = transformed._summon_follower_to_board(
            0,
            source,
            summon_cause="direct_summon",
        )
        if transform_target is None:
            raise AssertionError("failed to create transform reset target")
        transform_target.add_keyword(keyword)
        _put_hand(transformed, transform_spell)
        transformed.apply(PlayCard(0, 0))
        _choose_entity(transformed, transform_target.entity_id)
        transform_resets = not transform_target.has_keyword(keyword)

        cases.append(
            {
                "keyword": keyword,
                "return_to_hand_resets_dynamic": return_resets,
                "transform_resets_dynamic": transform_resets,
                "passed": return_resets and transform_resets,
            }
        )
    return cases


def _attack_mask_contracts() -> list[dict[str, object]]:
    def fresh_env(
        seed: int,
        rulebook: RuleBook | None = None,
    ) -> ShadowverseEnv:
        deck_a = [
            _synthetic_card(860000 + index) for index in range(40)
        ]
        deck_b = [
            _synthetic_card(861000 + index) for index in range(40)
        ]
        result = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=seed,
            rulebook=rulebook or RuleBook(),
        )
        result.reset(seed=seed)
        result.core.state.phase = Phase.MAIN
        result.core.state.active_player = 0
        result.core.players[0].board.clear()
        result.core.players[1].board.clear()
        result.core.players[0].hand.clear()
        result.core.players[0].hand_entity_ids.clear()
        result.core.players[0].max_mana = 10
        result.core.players[0].mana = 10
        return result

    def record(
        result: list[dict[str, object]],
        env: ShadowverseEnv,
        name: str,
        command,
        expected: bool,
    ) -> None:
        command_legal = command in set(env.core.legal_commands())
        action_mask_legal = bool(
            env.action_mask()[env._encode_command(command)]
        )
        result.append(
            {
                "case": name,
                "expected": expected,
                "command_legal": command_legal,
                "action_mask_legal": action_mask_legal,
                "passed": (
                    command_legal == expected
                    and action_mask_legal == expected
                ),
            }
        )

    env = fresh_env(15701)

    storm = Unit.summon(
        _synthetic_card(862001, keywords=("疾驰",)),
        entity_id=env.core.state.allocate_entity_id(),
    )
    rush = Unit.summon(
        _synthetic_card(862002, keywords=("突进",)),
        entity_id=env.core.state.allocate_entity_id(),
    )
    enemy = Unit.summon(
        _synthetic_card(862003),
        entity_id=env.core.state.allocate_entity_id(),
    )
    env.core.players[0].board.extend((storm, rush))
    env.core.players[1].board.append(enemy)

    commands = (
        ("storm_leader", Attack(0, storm.entity_id, None), True),
        ("storm_follower", Attack(0, storm.entity_id, enemy.entity_id), True),
        ("rush_leader", Attack(0, rush.entity_id, None), False),
        ("rush_follower", Attack(0, rush.entity_id, enemy.entity_id), True),
    )
    cases: list[dict[str, object]] = []
    for name, command, expected in commands:
        record(cases, env, name, command, expected)

    ward_env = fresh_env(15702)
    ward_attacker = Unit.summon(
        _synthetic_card(862010, keywords=("疾驰",)),
        entity_id=ward_env.core.state.allocate_entity_id(),
    )
    ward = Unit.summon(
        _synthetic_card(862011, keywords=("守护",)),
        entity_id=ward_env.core.state.allocate_entity_id(),
    )
    unguarded = Unit.summon(
        _synthetic_card(862012),
        entity_id=ward_env.core.state.allocate_entity_id(),
    )
    ward_env.core.players[0].board.append(ward_attacker)
    ward_env.core.players[1].board.extend((ward, unguarded))
    record(
        cases,
        ward_env,
        "ward_blocks_leader",
        Attack(0, ward_attacker.entity_id, None),
        False,
    )
    record(
        cases,
        ward_env,
        "ward_blocks_other_follower",
        Attack(0, ward_attacker.entity_id, unguarded.entity_id),
        False,
    )
    record(
        cases,
        ward_env,
        "ward_is_required_target",
        Attack(0, ward_attacker.entity_id, ward.entity_id),
        True,
    )

    hidden_env = fresh_env(15703)
    hidden_attacker = Unit.summon(
        _synthetic_card(862020, keywords=("疾驰",)),
        entity_id=hidden_env.core.state.allocate_entity_id(),
    )
    ambush = Unit.summon(
        _synthetic_card(862021, keywords=("潜行",)),
        entity_id=hidden_env.core.state.allocate_entity_id(),
    )
    intimidate = Unit.summon(
        _synthetic_card(862022, keywords=("威慑",)),
        entity_id=hidden_env.core.state.allocate_entity_id(),
    )
    ordinary = Unit.summon(
        _synthetic_card(862023),
        entity_id=hidden_env.core.state.allocate_entity_id(),
    )
    hidden_env.core.players[0].board.append(hidden_attacker)
    hidden_env.core.players[1].board.extend(
        (ambush, intimidate, ordinary)
    )
    record(
        cases,
        hidden_env,
        "ambush_is_not_attack_target",
        Attack(0, hidden_attacker.entity_id, ambush.entity_id),
        False,
    )
    record(
        cases,
        hidden_env,
        "intimidate_is_not_attack_target",
        Attack(0, hidden_attacker.entity_id, intimidate.entity_id),
        False,
    )
    record(
        cases,
        hidden_env,
        "ordinary_follower_remains_target",
        Attack(0, hidden_attacker.entity_id, ordinary.entity_id),
        True,
    )

    capacity_env = fresh_env(15704)
    capacity = Unit.summon(
        _synthetic_card(862030, keywords=("疾驰",)),
        entity_id=capacity_env.core.state.allocate_entity_id(),
    )
    capacity.grant_attacks_per_turn(2)
    capacity_env.core.players[0].board.append(capacity)
    first_attack = Attack(0, capacity.entity_id, None)
    record(
        cases,
        capacity_env,
        "two_attack_capacity_first",
        first_attack,
        True,
    )
    capacity_env.step(capacity_env._encode_command(first_attack))
    record(
        cases,
        capacity_env,
        "two_attack_capacity_second",
        Attack(0, capacity.entity_id, None),
        True,
    )

    restricted_env = fresh_env(15705)
    restricted = Unit.summon(
        _synthetic_card(862040, keywords=("疾驰",)),
        entity_id=restricted_env.core.state.allocate_entity_id(),
    )
    restricted.add_attack_restriction(
        AttackRestriction.CANNOT_ATTACK,
        duration=ModifierDuration.PERMANENT.value,
    )
    restricted_env.core.players[0].board.append(restricted)
    record(
        cases,
        restricted_env,
        "cannot_attack_restriction",
        Attack(0, restricted.entity_id, None),
        False,
    )

    targeting_spell = _synthetic_card(862050, card_type="法术")
    targeting_rule = CardRule(
        targeting_spell.card_id,
        Trigger.PLAY,
        (
            EffectOperation(
                kind=EffectKind.DAMAGE_UNIT,
                target=TargetKind.ENEMY_UNIT,
                amount=1,
                requires_target=True,
            ),
        ),
    )
    targeting_env = fresh_env(
        15706,
        rulebook=RuleBook((targeting_rule,)),
    )
    untargetable = Unit.summon(
        _synthetic_card(862051),
        entity_id=targeting_env.core.state.allocate_entity_id(),
    )
    untargetable.add_targeting_restriction(
        TargetingRestriction.CANNOT_BE_TARGETED_BY_ENEMY_EFFECTS,
        duration=ModifierDuration.PERMANENT.value,
    )
    targeting_env.core.players[1].board.append(untargetable)
    _put_hand(targeting_env.core, targeting_spell)
    record(
        cases,
        targeting_env,
        "cannot_be_targeted_blocks_play",
        PlayCard(0, 0),
        False,
    )
    return cases


def _iter_operations(
    value: object,
    path: str,
) -> Iterable[tuple[str, EffectOperation]]:
    if isinstance(value, EffectOperation):
        yield path, value
        for field in fields(value):
            nested = getattr(value, field.name)
            if nested is value:
                continue
            yield from _iter_operations(nested, f"{path}/{field.name}")
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            yield from _iter_operations(value[key], f"{path}/{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_operations(item, f"{path}/{index}")
        return
    if is_dataclass(value):
        for field in fields(value):
            yield from _iter_operations(
                getattr(value, field.name),
                f"{path}/{field.name}",
            )


def _rule_roots(
    rulebook: RuleBook,
) -> Iterable[tuple[int | None, str, object]]:
    for (card_id, trigger), operations in sorted(
        rulebook._rules.items(),
        key=lambda item: (item[0][0], item[0][1].value),
    ):
        yield card_id, f"rule:{trigger.value}", operations
    for card_id, modes in sorted(rulebook._play_modes.items()):
        yield card_id, "play_modes", modes
    for card_id, definitions in sorted(rulebook._listener_defs.items()):
        yield card_id, "listeners", definitions
    for card_id, definitions in sorted(rulebook._union_burst_defs.items()):
        yield card_id, "union_burst", definitions
    for card_id, definition in sorted(rulebook._activation_defs.items()):
        yield card_id, "activation", definition
    for card_id, definition in sorted(rulebook._faith_defs.items()):
        yield card_id, "faith", definition
    for card_id, definition in sorted(rulebook._fusion_defs.items()):
        yield card_id, "fusion", definition
    for card_id, definition in sorted(rulebook._invocation_defs.items()):
        yield card_id, "invocation", definition
    for emblem_id, definition in sorted(rulebook._emblem_defs.items()):
        yield None, f"emblem:{emblem_id}", definition


def _operation_record(
    root: str,
    path: str,
    operation: EffectOperation,
) -> dict[str, object]:
    keywords: list[str] = []
    if operation.keyword is not None:
        keywords.append(operation.keyword)
    keywords.extend(operation.keywords)
    return {
        "root": root,
        "path": path,
        "kind": operation.kind.value,
        "keywords": sorted(set(keywords)),
        "target": operation.target.value,
        "duration": operation.duration.value,
        "restriction": operation.restriction,
    }


def _coverage_entry(
    coverage: Mapping[str, object],
    card_id: int,
) -> Mapping[str, object]:
    classifications = coverage.get("classifications", {})
    if not isinstance(classifications, Mapping):
        return {}
    entry = classifications.get(str(card_id), {})
    return entry if isinstance(entry, Mapping) else {}


def _test_evidence(entry: Mapping[str, object]) -> list[str]:
    audit = entry.get("clause_audit", {})
    if isinstance(audit, Mapping):
        evidence = audit.get("test_evidence", [])
        if isinstance(evidence, list):
            return sorted(str(path) for path in evidence)
    metadata = entry.get("rule_metadata", {})
    if isinstance(metadata, Mapping):
        evidence = metadata.get("test_evidence", [])
        if isinstance(evidence, list):
            return sorted(str(path) for path in evidence)
    return []


def _inventory(
    cards: tuple[CardDefinition, ...],
    rulebook: RuleBook,
    closure_ids: set[int],
    coverage: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    sources: dict[int, list[dict[str, object]]] = {}
    global_sources: list[dict[str, object]] = []
    issues: list[str] = []
    for card_id, root, value in _rule_roots(rulebook):
        for path, operation in _iter_operations(value, root):
            if operation.kind not in KEYWORD_EFFECT_KINDS:
                continue
            record = _operation_record(root, path, operation)
            invalid_keywords = [
                keyword
                for keyword in record["keywords"]
                if normalize_keyword_name(keyword) not in RUNTIME_UNIT_KEYWORDS
            ]
            if invalid_keywords:
                issues.append(
                    f"{root}: unsupported runtime keywords {invalid_keywords}"
                )
            if card_id is None:
                global_sources.append(record)
            else:
                sources.setdefault(card_id, []).append(record)
    for card_id, passives in sorted(rulebook._passives.items()):
        for index, passive in enumerate(passives):
            if passive.kind not in {
                "non_intrinsic_keyword",
                "attacks_per_turn",
                "forces_enemy_ability_target",
                "ignores_ward",
            }:
                continue
            sources.setdefault(card_id, []).append(
                {
                    "root": "passives",
                    "path": f"passives/{index}",
                    "kind": f"passive:{passive.kind}",
                    "keywords": (
                        [passive.keyword]
                        if passive.keyword is not None
                        else []
                    ),
                    "target": "self",
                    "duration": "permanent",
                    "restriction": None,
                    "amount": passive.amount,
                }
            )

    rows: list[dict[str, object]] = []
    for card in cards:
        declared = sorted(
            normalize_keyword_name(keyword)
            for keyword in card.keywords
            if normalize_keyword_name(keyword) in RUNTIME_UNIT_KEYWORDS
        )
        conditional = sorted(rulebook.non_intrinsic_keywords(card.card_id))
        intrinsic = sorted(set(declared) - set(conditional))
        operations = sorted(
            sources.get(card.card_id, []),
            key=lambda item: (
                str(item["root"]),
                str(item["path"]),
                str(item["kind"]),
            ),
        )
        if not declared and not conditional and not operations:
            continue
        entry = _coverage_entry(coverage, card.card_id)
        evidence = _test_evidence(entry)
        row_issues: list[str] = []
        if set(conditional) - set(declared):
            row_issues.append(
                "non-intrinsic keyword is absent from declared card keywords"
            )
        accepted_coverage = (
            {"covered_exact"}
            if card.is_collectible
            else {"token_or_non_collectible"}
        )
        if entry.get("coverage") not in accepted_coverage:
            row_issues.append(
                "card does not have the required collectible/generated "
                "coverage classification"
            )
        if not evidence:
            row_issues.append("card lacks permanent test evidence")
        for path in evidence:
            if not (_repo_path(Path(path))).is_file():
                row_issues.append(f"missing test evidence file: {path}")
        if row_issues:
            issues.extend(
                f"card {card.card_id}: {message}"
                for message in row_issues
            )
        rows.append(
            {
                "card_id": card.card_id,
                "name": card.name,
                "card_type": card.card_type,
                "collectible": card.is_collectible,
                "training_closure": card.card_id in closure_ids,
                "declared_keywords": declared,
                "intrinsic_keywords": intrinsic,
                "conditional_keywords": conditional,
                "runtime_sources": operations,
                "test_evidence": evidence,
                "issues": row_issues,
                "passed": not row_issues,
            }
        )
    return rows, global_sources, sorted(set(issues))


def _matrix(
    entry_contracts: list[dict[str, object]],
    special_entry_contracts: list[dict[str, object]],
    evolution_contracts: list[dict[str, object]],
) -> list[dict[str, object]]:
    evidence = {
        "normal_play": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_real_intrinsic_keyword_pairs_batch.py",
        ],
        "enhance_play": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_real_listener_context_leader_runtime_nineteenth_batch.py",
        ],
        "direct_summon": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_real_generated_entry_listener_batch.py",
        ],
        "summon_from_deck": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_card_origin.py",
        ],
        "reanimate": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_card_origin.py",
        ],
        "definition_copy": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_card_origin.py",
        ],
        "exact_copy": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_snapshot_clone.py",
        ],
        "transform": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_runtime_modifiers.py",
        ],
        "normal_evolution": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_super_evolution.py",
        ],
        "super_evolution": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_super_evolution.py",
        ],
        "effect_evolution": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_super_evolution.py",
        ],
        "effect_super_evolution": [
            "tests/test_keyword_entry_audit.py",
            "tests/test_super_evolution.py",
        ],
    }
    evolution_by_method = {
        str(row["evolution_method"]): row
        for row in evolution_contracts
    }
    actual_results = {
        "normal_play": [
            bool(row["normal_play"]) for row in entry_contracts
        ],
        "enhance_play": [
            bool(row["enhance_play"]) for row in entry_contracts
        ],
        "direct_summon": [
            bool(row["summon_causes"]["direct_summon"])
            for row in entry_contracts
        ],
        "summon_from_deck": [
            bool(row["summon_from_deck"])
            for row in special_entry_contracts
        ],
        "reanimate": [
            bool(row["reanimate"])
            for row in special_entry_contracts
        ],
        "definition_copy": [
            bool(row["definition_copy"])
            for row in special_entry_contracts
        ],
        "exact_copy": [
            bool(row["exact_copy_preserves_dynamic"])
            for row in special_entry_contracts
        ],
        "transform": [
            bool(row["transform"]) for row in entry_contracts
        ],
        "normal_evolution": [
            bool(evolution_by_method["normal_evolution"]["passed"])
        ],
        "super_evolution": [
            bool(evolution_by_method["super_evolution"]["passed"])
        ],
        "effect_evolution": [
            bool(evolution_by_method["effect_evolution"]["passed"])
        ],
        "effect_super_evolution": [
            bool(
                evolution_by_method["effect_super_evolution"]["passed"]
            )
        ],
    }
    rows: list[dict[str, object]] = []
    for method in ENTRY_METHODS:
        paths = evidence[method]
        evidence_files_exist = all(
            _repo_path(Path(path)).is_file() for path in paths
        )
        actual_contract_passed = all(actual_results[method])
        rows.append(
            {
                "entry_method": method,
                "source_categories": list(SOURCE_CATEGORIES),
                "fanfare_policy": (
                    "fires"
                    if method in {"normal_play", "enhance_play"}
                    else (
                        "not_applicable"
                        if "evolution" in method
                        else "does_not_fire"
                    )
                ),
                "test_evidence": paths,
                "evidence_files_exist": evidence_files_exist,
                "actual_contract_case_count": len(
                    actual_results[method]
                ),
                "actual_contract_passed": actual_contract_passed,
                "passed": (
                    evidence_files_exist and actual_contract_passed
                ),
            }
        )
    return rows


def build_report(
    *,
    database: Path = DEFAULT_DATABASE,
    rules: Path = DEFAULT_RULES,
    closure: Path = DEFAULT_CLOSURE,
    coverage_report: Path = DEFAULT_COVERAGE,
) -> dict[str, object]:
    database_path = _repo_path(database)
    rules_path = _repo_path(rules)
    closure_path = _repo_path(closure)
    coverage_path = _repo_path(coverage_report)

    repository = CardRepository(database_path)
    cards = repository.all_cards()
    rulebook = RuleBook.from_directory(rules_path)
    closure_payload = json.loads(closure_path.read_text(encoding="utf-8"))
    closure_ids = {
        int(card_id)
        for card_id in closure_payload.get("closure_card_ids", [])
    }
    if not closure_ids:
        closure_ids = {
            int(row["card_id"])
            for row in closure_payload.get("cards", [])
            if isinstance(row, Mapping) and "card_id" in row
        }
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))

    inventory, global_sources, inventory_issues = _inventory(
        cards,
        rulebook,
        closure_ids,
        coverage,
    )
    state_contracts = _state_contracts()
    entry_contracts = _entry_contracts()
    special_entry_contracts = _special_entry_contracts()
    evolution_contracts = _evolution_contracts()
    zone_reset_contracts = _zone_reset_contracts()
    attack_mask_contracts = _attack_mask_contracts()
    matrix = _matrix(
        entry_contracts,
        special_entry_contracts,
        evolution_contracts,
    )

    contract_failures = [
        f"state:{row['keyword']}"
        for row in state_contracts
        if not row["passed"]
    ]
    contract_failures.extend(
        f"entry:{row['keyword']}"
        for row in entry_contracts
        if not row["passed"]
    )
    contract_failures.extend(
        f"special_entry:{row['keyword']}"
        for row in special_entry_contracts
        if not row["passed"]
    )
    contract_failures.extend(
        f"evolution:{row['evolution_method']}"
        for row in evolution_contracts
        if not row["passed"]
    )
    contract_failures.extend(
        f"zone_reset:{row['keyword']}"
        for row in zone_reset_contracts
        if not row["passed"]
    )
    contract_failures.extend(
        f"attack_mask:{row['case']}"
        for row in attack_mask_contracts
        if not row["passed"]
    )
    matrix_failures = [
        str(row["entry_method"])
        for row in matrix
        if not row["passed"]
    ]
    source_card_ids = {int(row["card_id"]) for row in inventory}
    training_source_ids = source_card_ids & closure_ids
    collectible_source_count = sum(
        bool(row["collectible"]) for row in inventory
    )
    generated_source_count = len(inventory) - collectible_source_count
    zooey = next(
        row for row in inventory if row["card_id"] == 10444120
    )
    zooey_regression = {
        "card_id": 10444120,
        "normal_has_storm": False,
        "enhance_10_has_storm": True,
        "conditional_source_recorded": (
            "疾驰" in zooey["conditional_keywords"]
        ),
        "official_evidence": {
            "url": (
                "https://shadowverse-wb.com/ja/deck/cardslist/card/"
                "?card_id=10444120"
            ),
            "accessed_on": "2026-07-30",
            "conclusion": (
                "The official card page grants Storm only in the "
                "Enhance 10 clause; normal play does not grant Storm."
            ),
        },
        "test_evidence": [
            "tests/test_real_listener_context_leader_runtime_nineteenth_batch.py::ListenerContextLeaderRuntimeNineteenthBehaviorTests::test_zooey_only_gains_storm_from_enhance_ten",
            "tests/test_play_mode_boundary_audit.py::PlayModeBoundaryAuditTests::test_real_regression_cards_have_high_pp_exclusivity_evidence",
        ],
    }
    zooey_regression["passed"] = bool(
        zooey_regression["conditional_source_recorded"]
        and zooey["passed"]
    )

    failures = (
        inventory_issues
        + contract_failures
        + matrix_failures
        + ([] if zooey_regression["passed"] else ["zooey_regression"])
    )
    return {
        "schema_version": 1,
        "report_kind": "swb_keyword_entry_audit",
        "inputs": {
            "database": str(database).replace("\\", "/"),
            "database_sha256": _sha256(database_path),
            "rules": str(rules).replace("\\", "/"),
            "closure": str(closure).replace("\\", "/"),
            "closure_sha256": _sha256(closure_path),
            "coverage_report": str(coverage_report).replace("\\", "/"),
            "coverage_report_sha256": _sha256(coverage_path),
        },
        "scope": {
            "database_card_count": len(cards),
            "collectible_card_count": sum(card.is_collectible for card in cards),
            "generated_card_count": sum(
                not card.is_collectible for card in cards
            ),
            "training_closure_card_count": len(closure_ids),
            "runtime_keywords": list(KEYWORDS),
            "entry_methods": list(ENTRY_METHODS),
            "source_categories": list(SOURCE_CATEGORIES),
            "keyword_source_card_count": len(inventory),
            "collectible_keyword_source_count": collectible_source_count,
            "generated_keyword_source_count": generated_source_count,
            "training_keyword_source_count": len(training_source_ids),
            "global_keyword_source_count": len(global_sources),
            "scope_complete": (
                bool(cards)
                and sum(card.is_collectible for card in cards)
                + sum(not card.is_collectible for card in cards)
                == len(cards)
                and closure_ids <= {card.card_id for card in cards}
            ),
        },
        "entry_method_matrix": matrix,
        "state_contracts": state_contracts,
        "entry_contracts": entry_contracts,
        "special_entry_contracts": special_entry_contracts,
        "evolution_contracts": evolution_contracts,
        "zone_reset_contracts": zone_reset_contracts,
        "attack_mask_contracts": attack_mask_contracts,
        "zooey_regression": zooey_regression,
        "global_runtime_sources": global_sources,
        "cards": inventory,
        "summary": {
            "inventory_issue_count": len(inventory_issues),
            "contract_failure_count": len(contract_failures),
            "matrix_failure_count": len(matrix_failures),
            "failure_count": len(failures),
            "failures": failures,
            "passed": (
                not failures
                and all(row["passed"] for row in inventory)
                and bool(cards)
                and closure_ids <= {card.card_id for card in cards}
            ),
        },
    }


def render_json(report: Mapping[str, object]) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    ) + "\n"


def render_markdown(report: Mapping[str, object]) -> str:
    scope = report["scope"]
    summary = report["summary"]
    lines = [
        "# Keyword provenance and entry-method audit",
        "",
        f"- Result: **{'PASS' if summary['passed'] else 'FAIL'}**; "
        f"{summary['failure_count']} failures.",
        f"- Snapshot: {scope['database_card_count']} cards "
        f"({scope['collectible_card_count']} collectible / "
        f"{scope['generated_card_count']} generated).",
        f"- Keyword sources: {scope['keyword_source_card_count']} cards; "
        f"{scope['training_keyword_source_count']} in the training closure.",
        f"- Runtime keywords: {len(scope['runtime_keywords'])}; "
        f"entry methods: {len(scope['entry_methods'])}.",
        "",
        "## Entry-method matrix",
        "",
        "| Entry method | Fanfare | Evidence | Result |",
        "|---|---|---:|:---:|",
    ]
    for row in report["entry_method_matrix"]:
        lines.append(
            f"| {row['entry_method']} | {row['fanfare_policy']} | "
            f"{len(row['test_evidence'])} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Source inventory",
            "",
            "| Card | Intrinsic | Conditional | Runtime grants | Training | Result |",
            "|---|---|---|---:|:---:|:---:|",
        ]
    )
    for row in report["cards"]:
        intrinsic = ", ".join(row["intrinsic_keywords"]) or "-"
        conditional = ", ".join(row["conditional_keywords"]) or "-"
        lines.append(
            f"| {row['card_id']} {row['name']} | {intrinsic} | "
            f"{conditional} | {len(row['runtime_sources'])} | "
            f"{'yes' if row['training_closure'] else 'no'} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=DEFAULT_COVERAGE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    report = build_report(
        database=args.database,
        rules=args.rules,
        closure=args.closure,
        coverage_report=args.coverage_report,
    )
    output = _repo_path(args.output)
    markdown = _repo_path(args.markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_json(report), encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"JSON report written to {args.output}")
    print(f"Markdown report written to {args.markdown}")
    print(
        "cards={cards} sources={sources} training_sources={training} "
        "failures={failures} passed={passed}".format(
            cards=report["scope"]["database_card_count"],
            sources=report["scope"]["keyword_source_card_count"],
            training=report["scope"]["training_keyword_source_count"],
            failures=report["summary"]["failure_count"],
            passed=report["summary"]["passed"],
        )
    )


if __name__ == "__main__":
    main()
