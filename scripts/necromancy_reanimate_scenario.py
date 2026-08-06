#!/usr/bin/env python
"""Deterministic scenario exercising Necromancy and Reanimate."""
from __future__ import annotations

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard, EndTurn
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine
from swb.engine.state import DeathCause, DestroyedFollowerRecord, HandCard, Unit


def _card(cid, **kw):
    return CardDefinition(
        card_id=cid, card_set_id=10000, class_id=1, class_name="精灵",
        name=kw.get("name", f"card-{cid}"),
        cost=kw.get("cost", 1), card_type=kw.get("card_type", "随从"),
        attack=kw.get("attack", 1), life=kw.get("life", 1),
        keywords=frozenset(), support_level="basic", is_collectible=True,
    )


def build_engine():
    # Cards:
    # 100: Necromancy spell: consumes 3 shadows to damage enemy leader by 3
    # 200: Reanimate spell: reanimate up to cost 4
    # 300: target follower (cost 3, attack 3, life 3)
    # 400: target follower (cost 4, attack 4, life 4)

    necro_op = EffectOperation(
        kind=EffectKind.NECROMANCY, amount=3, target=TargetKind.OWN_LEADER,
        necromancy_operations=(
            EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=3),
        ),
    )
    reanimate_op = EffectOperation(
        kind=EffectKind.REANIMATE, amount=4, target=TargetKind.OWN_LEADER,
    )

    rules = (
        CardRule(card_id=100, trigger=Trigger.PLAY, operations=(necro_op,)),
        CardRule(card_id=200, trigger=Trigger.PLAY, operations=(reanimate_op,)),
    )

    all_defs = {
        100: _card(100, name="NecroSpell", card_type="法术", cost=2),
        200: _card(200, name="ReanimateSpell", card_type="法术", cost=2),
        300: _card(300, name="Follower3", cost=3, attack=3, life=3),
        400: _card(400, name="Follower4", cost=4, attack=4, life=4),
    }

    deck_a = [_card(i, name=f"dA-{i}") for i in range(1000, 1040)]
    deck_b = [_card(i, name=f"dB-{i}") for i in range(2000, 2040)]

    def resolver(cid):
        return all_defs.get(cid)

    engine = GameEngine(
        deck_a=deck_a, deck_b=deck_b,
        class_a=1, class_b=1,
        seed=42,
        rulebook=RuleBook(rules),
        card_resolver=resolver,
    )
    return engine, all_defs


def run_scenario():
    engine, defs = build_engine()
    engine.reset(seed=42)

    # Give p0 10 shadows
    engine.players[0].shadows = 10
    # Add destroyed followers for reanimate
    engine.state.destroyed_followers = [
        DestroyedFollowerRecord(definition=defs[300], owner=0, death_sequence=1, cause=DeathCause.COMBAT),
        DestroyedFollowerRecord(definition=defs[400], owner=0, death_sequence=2, cause=DeathCause.EFFECT_DESTROY),
    ]

    # Give p0 mana and the spell cards in hand
    engine.players[0].mana = 10
    engine.players[0].hand = [
        HandCard(
            definition=card_def,
            entity_id=engine.state.allocate_entity_id(),
        )
        for card_def in (defs[100], defs[200])
    ]
    engine.players[0].hand_entity_ids = [
        card.entity_id for card in engine.players[0].hand
    ]

    print("=== Step 1: Play Necromancy spell (card 100) ===")
    print(f"  Shadows before: {engine.players[0].shadows}")
    print(f"  Enemy HP before: {engine.players[1].health}")
    engine.apply(PlayCard(0, 0))
    print(f"  Shadows after: {engine.players[0].shadows}")
    print(f"  Enemy HP after: {engine.players[1].health}")

    necro_events = [e for e in engine.event_history if e.type is EventType.NECROMANCY_ACTIVATED]
    print(f"  NECROMANCY_ACTIVATED events: {len(necro_events)}")
    for e in necro_events:
        print(f"    amount={e.amount} before={e.metadata.get('shadows_before')} after={e.metadata.get('shadows_after')}")

    shadows_events = [e for e in engine.event_history if e.type is EventType.SHADOWS_CHANGED]
    print(f"  SHADOWS_CHANGED events: {len(shadows_events)}")
    for e in shadows_events:
        print(f"    change={e.metadata.get('change')} before={e.metadata.get('shadows_before')} after={e.metadata.get('shadows_after')}")
    assert engine.players[0].shadows == 8
    assert engine.players[1].health == 17
    assert len(necro_events) == 1

    print("\n=== Step 2: Play Reanimate spell (card 200) ===")
    board_before = len(engine.players[0].board)
    coor_before = engine.players[0].cooperation
    engine.apply(PlayCard(0, 0))
    board_after = len(engine.players[0].board)
    coor_after = engine.players[0].cooperation
    print(f"  Board before: {board_before}, after: {board_after}")
    print(f"  Cooperation before: {coor_before}, after: {coor_after}")
    if engine.players[0].board:
        unit = engine.players[0].board[0]
        assert isinstance(unit, Unit)
        print(f"  Reanimated: {unit.definition.name} ({unit.definition.card_id}) entity_id={unit.entity_id}")
        print(f"  Stats: {unit.attack}/{unit.health}  evolved={unit.evolved}")

    reanimate_events = [e for e in engine.event_history if e.type is EventType.REANIMATE_RESOLVED]
    print(f"  REANIMATE_RESOLVED events: {len(reanimate_events)}")
    for e in reanimate_events:
        print(f"    card_id={e.metadata.get('reanimated_card_id')} entity_id={e.metadata.get('new_entity_id')}")

    summon_events = [e for e in engine.event_history if e.type is EventType.FOLLOWER_SUMMONED]
    print(f"  FOLLOWER_SUMMONED events: {len(summon_events)}")

    # Verify key properties
    print("\n=== Verification ===")
    print(f"  Shadows: {engine.players[0].shadows} (expected: 9)")
    print(f"  Board followers: {len(engine.players[0].board)} (expected: 1)")
    print(f"  Cooperation: {coor_after} (expected: 1)")
    assert engine.players[0].shadows == 9
    assert len(engine.players[0].board) == 1
    assert coor_after == 1
    assert coor_after == coor_before + 1

    # Determinism check: run again with same seed
    print("\n=== Determinism Check (same seed, second run) ===")
    engine2, defs2 = build_engine()
    engine2.reset(seed=42)
    engine2.players[0].shadows = 10
    engine2.state.destroyed_followers = [
        DestroyedFollowerRecord(definition=defs2[300], owner=0, death_sequence=1, cause=DeathCause.COMBAT),
        DestroyedFollowerRecord(definition=defs2[400], owner=0, death_sequence=2, cause=DeathCause.EFFECT_DESTROY),
    ]
    engine2.players[0].mana = 10
    engine2.players[0].hand = [
        HandCard(
            definition=card_def,
            entity_id=engine2.state.allocate_entity_id(),
        )
        for card_def in (defs2[100], defs2[200])
    ]
    engine2.players[0].hand_entity_ids = [
        card.entity_id for card in engine2.players[0].hand
    ]
    engine2.apply(PlayCard(0, 0))
    engine2.apply(PlayCard(0, 0))

    uid1 = engine.players[0].board[0].entity_id if engine.players[0].board else None
    uid2 = engine2.players[0].board[0].entity_id if engine2.players[0].board else None
    card1 = engine.players[0].board[0].definition.card_id if engine.players[0].board else None
    card2 = engine2.players[0].board[0].definition.card_id if engine2.players[0].board else None
    print(f"  Run 1: entity_id={uid1}, card_id={card1}, shadows={engine.players[0].shadows}")
    print(f"  Run 2: entity_id={uid2}, card_id={card2}, shadows={engine2.players[0].shadows}")
    assert uid1 == uid2
    assert card1 == card2
    assert engine.players[0].shadows == engine2.players[0].shadows
    print("  DETERMINISTIC: OK")


if __name__ == "__main__":
    run_scenario()
