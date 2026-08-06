from __future__ import annotations

import json

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import (
    ChoiceKind,
    ChoiceOption,
    ChoiceRequest,
    Choose,
    PlayCard,
)
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.resolution import GameEngine
from swb.engine.state import GraveyardCard, HandCard


def _card(
    card_id: int,
    *,
    name: str,
    card_type: str = "随从",
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=name,
        cost=0,
        card_type=card_type,
        attack=1 if card_type == "随从" else None,
        life=1 if card_type == "随从" else None,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def _choose_card(engine: GameEngine, card_id: int) -> None:
    request = engine.state.pending_choice
    assert request is not None
    option = next(
        option
        for option in request.options
        if next(
            graveyard_card
            for graveyard_card in engine.players[0].graveyard
            if graveyard_card.entity_id == option.entity_id
        ).definition.card_id
        == card_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _run_zone_operations(seed: int) -> dict[str, object]:
    return_spell = _card(100, name="Return", card_type="法术")
    summon_spell = _card(101, name="Summon", card_type="法术")
    banish_spell = _card(102, name="Banish", card_type="法术")
    return_target = _card(200, name="ReturnTarget", card_type="法术")
    summon_target = _card(201, name="SummonTarget")
    banish_target = _card(202, name="BanishTarget")

    rules = RuleBook((
        CardRule(
            100,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND,
                    TargetKind.OWN_GRAVEYARD_CARD,
                ),
            ),
        ),
        CardRule(
            101,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.SUMMON_FROM_GRAVEYARD,
                    TargetKind.OWN_GRAVEYARD_CARD,
                ),
            ),
        ),
        CardRule(
            102,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.BANISH_FROM_GRAVEYARD,
                    TargetKind.OWN_GRAVEYARD_CARD,
                ),
            ),
        ),
    ))
    definitions = {
        card.card_id: card
        for card in (
            return_spell,
            summon_spell,
            banish_spell,
            return_target,
            summon_target,
            banish_target,
        )
    }
    deck_a = [_card(1000 + index, name=f"A{index}") for index in range(40)]
    deck_b = [_card(2000 + index, name=f"B{index}") for index in range(40)]
    engine = GameEngine(
        deck_a,
        deck_b,
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rules,
        card_resolver=definitions.get,
    )
    engine.reset(seed=seed)
    player = engine.players[0]
    player.mana = 10
    player.hand = [
        HandCard(return_spell, engine.state.allocate_entity_id()),
        HandCard(summon_spell, engine.state.allocate_entity_id()),
        HandCard(banish_spell, engine.state.allocate_entity_id()),
    ]
    player.hand_entity_ids = [card.entity_id for card in player.hand]
    spell_entity_ids = {
        card.definition.card_id: card.entity_id for card in player.hand
    }
    player.graveyard = [
        GraveyardCard(return_target, 50001, 0, 1, "scenario"),
        GraveyardCard(summon_target, 50002, 0, 2, "scenario"),
        GraveyardCard(banish_target, 50003, 0, 3, "scenario"),
    ]
    shadows_before = player.shadows

    engine.apply(PlayCard(0, 0))
    _choose_card(engine, return_target.card_id)
    returned = next(
        card for card in player.hand if card.definition.card_id == return_target.card_id
    )
    assert returned.entity_id == 50001
    assert any(
        card.entity_id == spell_entity_ids[100] for card in player.graveyard
    )

    summon_index = next(
        index
        for index, card in enumerate(player.hand)
        if card.definition.card_id == summon_spell.card_id
    )
    engine.apply(PlayCard(0, summon_index))
    _choose_card(engine, summon_target.card_id)
    assert any(entity.entity_id == 50002 for entity in player.board)

    banish_index = next(
        index
        for index, card in enumerate(player.hand)
        if card.definition.card_id == banish_spell.card_id
    )
    engine.apply(PlayCard(0, banish_index))
    _choose_card(engine, banish_target.card_id)
    assert any(card.card_id == banish_target.card_id for card in player.banished)
    assert player.shadows == shadows_before + 3

    zone_ids = (
        [card.entity_id for card in player.hand]
        + [entity.entity_id for entity in player.board]
        + [card.entity_id for card in player.graveyard]
    )
    assert len(zone_ids) == len(set(zone_ids))

    return {
        "hand": [(card.definition.card_id, card.entity_id) for card in player.hand],
        "board": [
            (entity.definition.card_id, entity.entity_id) for entity in player.board
        ],
        "graveyard": [
            (card.definition.card_id, card.entity_id) for card in player.graveyard
        ],
        "banished": [card.card_id for card in player.banished],
        "shadows": player.shadows,
        "events": [event.type.value for event in engine.event_history],
    }


def _run_pagination(seed: int) -> dict[str, object]:
    deck_a = [_card(3000 + index, name=f"P{index}") for index in range(40)]
    deck_b = [_card(4000 + index, name=f"Q{index}") for index in range(40)]
    env = ShadowverseEnv(
        deck_a,
        deck_b,
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=RuleBook(),
    )
    env.reset(seed=seed)
    env.core.state.pending_choice = ChoiceRequest(
        player_index=0,
        prompt="选择第 41 张墓地卡",
        options=tuple(
            ChoiceOption(f"entity:{60000 + index}", f"G{index}", 60000 + index)
            for index in range(41)
        ),
        continuation_id="scenario-pagination",
        choice_kind=ChoiceKind.GRAVEYARD,
        request_id=1,
    )

    env.step(env.GRAVEYARD_NEXT_PAGE)
    env.step(env.GRAVEYARD_NEXT_PAGE)
    mask = env.action_mask()
    assert mask[env.GRAVEYARD_SLOT_OFFSET + 8]
    decoded = env._decode_action(env.GRAVEYARD_SLOT_OFFSET + 8)
    assert isinstance(decoded, Choose)
    assert decoded.option_id == "entity:60040"
    env.step(env.GRAVEYARD_SLOT_OFFSET + 8)
    assert env.core.state.pending_choice is None

    return {
        "selected": decoded.option_id,
        "page_after": env.info()["graveyard_page"],
        "action_size": env.ACTION_SIZE,
        "observation_size": len(env.observation()),
    }


def run(seed: int = 17) -> dict[str, object]:
    return {
        "zones": _run_zone_operations(seed),
        "pagination": _run_pagination(seed),
    }


def main() -> None:
    first = run()
    second = run()
    assert first == second
    print(json.dumps(first, ensure_ascii=False, sort_keys=True))
    print("DETERMINISTIC: OK")


if __name__ == "__main__":
    main()
