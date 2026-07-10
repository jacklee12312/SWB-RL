# -*- coding: utf-8 -*-
"""Tests for unified card origin, token identity, and zone eligibility."""

from __future__ import annotations

import os
import sqlite3
import unittest
from unittest.mock import patch

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import (
    CardOrigin,
    is_derived,
    is_generated_card,
    is_initial_deck_eligible,
    is_reanimate_eligible,
    is_token,
    is_token_definition,
    origin_for_added_card,
    origin_for_summoned_card,
)
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import (
    DeathCause,
    DestroyedFollowerRecord,
    GraveyardCard,
    HandCard,
    Phase,
    Unit,
)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _card(
    card_id: int,
    *,
    attack: int | None = 1,
    life: int | None = 1,
    cost: int = 1,
    card_type: str = "随从",
    name: str | None = None,
    is_collectible: bool = True,
    card_set_id: int = 10000,
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=card_set_id,
        class_id=1,
        class_name="精灵",
        name=name or f"card-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack if card_type == "随从" else None,
        life=life if card_type == "随从" else None,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=is_collectible,
    )


def _token_card(card_id: int, **kwargs) -> CardDefinition:
    return _card(card_id, card_set_id=90000, is_collectible=False, **kwargs)


def _make_resolver(defs: dict[int, CardDefinition]):
    def resolve(cid: int) -> CardDefinition | None:
        return defs.get(cid)
    return resolve


def _spell_rule(card_id: int, *ops: EffectOperation) -> CardRule:
    return CardRule(card_id=card_id, trigger=Trigger.PLAY, operations=ops)


def _make_engine(
    *rules: CardRule,
    card_defs: dict[int, CardDefinition] | None = None,
    seed: int = 42,
) -> GameEngine:
    all_defs = dict(card_defs) if card_defs else {}
    deck_a = [_card(i, name=f"dA-{i}") for i in range(1000, 1040)]
    deck_b = [_card(i, name=f"dB-{i}") for i in range(2000, 2040)]
    return GameEngine(
        deck_a=deck_a, deck_b=deck_b,
        class_a=1, class_b=1,
        seed=seed,
        rulebook=RuleBook(tuple(rules)),
        card_resolver=_make_resolver(all_defs),
    )


def _insert_spell(
    engine,
    spell_def,
    *,
    origin: CardOrigin = CardOrigin.DECK,
    source_origin: CardOrigin | None = None,
):
    """Insert a spell card into hand[0] and ensure enough mana."""
    engine.players[0].mana = 10
    hc = HandCard(
        definition=spell_def,
        entity_id=engine.state.allocate_entity_id(),
        origin=origin,
        source_origin=source_origin,
    )
    engine.players[0].hand.insert(0, hc)
    engine.players[0].hand_entity_ids.insert(0, hc.entity_id)


def _play(engine):
    """Play the first card in hand (should be the inserted spell)."""
    engine.apply(PlayCard(0, 0))


# ---------------------------------------------------------------------------
# CardOrigin model
# ---------------------------------------------------------------------------

class OriginModelTests(unittest.TestCase):
    def test_all_enum_values(self):
        expected = {"deck", "generated", "token", "reanimated", "returned", "transformed", "unknown"}
        self.assertEqual(set(v.value for v in CardOrigin), expected)

    def test_is_derived(self):
        self.assertFalse(is_derived(CardOrigin.DECK))
        self.assertTrue(is_derived(CardOrigin.GENERATED))
        self.assertTrue(is_derived(CardOrigin.TOKEN))
        self.assertTrue(is_derived(CardOrigin.REANIMATED))
        self.assertTrue(is_derived(CardOrigin.TRANSFORMED))
        self.assertFalse(is_derived(CardOrigin.RETURNED))
        self.assertFalse(is_derived(CardOrigin.UNKNOWN))

    def test_is_generated_card(self):
        self.assertTrue(is_generated_card(CardOrigin.GENERATED))
        self.assertTrue(is_generated_card(CardOrigin.TOKEN))
        self.assertTrue(is_generated_card(CardOrigin.REANIMATED))
        self.assertFalse(is_generated_card(CardOrigin.DECK))
        self.assertFalse(is_generated_card(CardOrigin.TRANSFORMED))
        self.assertFalse(is_generated_card(CardOrigin.RETURNED))

    def test_is_token_definition(self):
        self.assertTrue(is_token_definition(_token_card(90001)))
        self.assertFalse(is_token_definition(_card(10001)))

    def test_is_token(self):
        token_c = _token_card(90001)
        normal_c = _card(10001)
        self.assertTrue(is_token(token_c))
        self.assertTrue(is_token(token_c, CardOrigin.DECK))
        self.assertTrue(is_token(normal_c, CardOrigin.TOKEN))
        self.assertFalse(is_token(normal_c))
        self.assertFalse(is_token(normal_c, CardOrigin.DECK))

    def test_origin_for_added_card(self):
        self.assertEqual(origin_for_added_card(_token_card(90001)), CardOrigin.TOKEN)
        self.assertEqual(origin_for_added_card(_card(10001)), CardOrigin.GENERATED)

    def test_origin_for_summoned_card(self):
        self.assertEqual(origin_for_summoned_card(_token_card(90001)), CardOrigin.TOKEN)
        self.assertEqual(origin_for_summoned_card(_card(10001)), CardOrigin.GENERATED)

    def test_is_initial_deck_eligible(self):
        self.assertTrue(is_initial_deck_eligible(_card(10001)))
        self.assertFalse(is_initial_deck_eligible(_token_card(90001)))
        malformed = _card(
            90002,
            card_set_id=90000,
            is_collectible=True,
        )
        self.assertFalse(is_initial_deck_eligible(malformed))

    def test_graveyard_flags_are_normalized_without_losing_explicit_truth(self):
        explicit = GraveyardCard(
            definition=_card(10001),
            entity_id=1,
            owner=0,
            entered_sequence=1,
            entry_cause="test",
            derived=True,
            token=True,
        )
        self.assertTrue(explicit.derived)
        self.assertTrue(explicit.token)

        inherited = GraveyardCard(
            definition=_card(10002),
            entity_id=2,
            owner=0,
            entered_sequence=2,
            entry_cause="test",
            origin=CardOrigin.REANIMATED,
            source_origin=CardOrigin.TOKEN,
        )
        self.assertTrue(inherited.derived)
        self.assertTrue(inherited.token)

    def test_reanimate_eligible_follower(self):
        record = DestroyedFollowerRecord(
            definition=_card(10001),
            owner=0,
            death_sequence=1,
            cause=DeathCause.COMBAT,
        )
        self.assertTrue(is_reanimate_eligible(record))

    def test_graveyard_return_always_eligible(self):
        from swb.engine.origin import is_graveyard_return_eligible
        gc = GraveyardCard(
            definition=_token_card(90001),
            entity_id=1, owner=0, entered_sequence=1,
            entry_cause="combat",
        )
        self.assertTrue(is_graveyard_return_eligible(gc))


# ---------------------------------------------------------------------------
# Deck eligibility
# ---------------------------------------------------------------------------

class DeckEligibilityTests(unittest.TestCase):
    def test_collectible_card_accepted_in_deck(self):
        from swb.engine.deck import validate_deck
        deck = [_card(100 + i) for i in range(40)]
        validate_deck(deck, 1, player_index=0)

    def test_token_card_rejected_from_deck(self):
        from swb.engine.deck import validate_deck
        token_deck = [_token_card(90001 + i) for i in range(40)]
        with self.assertRaises(ValueError):
            validate_deck(token_deck, 1, player_index=0)

    def test_mixed_bag_has_token_rejected(self):
        from swb.engine.deck import validate_deck
        deck = [_card(100 + i) for i in range(39)] + [_token_card(90001)]
        with self.assertRaises(ValueError):
            validate_deck(deck, 1, player_index=0)

    def test_collectible_flag_cannot_hide_token_set(self):
        from swb.engine.deck import validate_deck
        deck = [_card(100 + i) for i in range(39)] + [
            _card(90001, card_set_id=90000, is_collectible=True)
        ]
        with self.assertRaises(ValueError):
            validate_deck(deck, 1, player_index=0)


# ---------------------------------------------------------------------------
# Deck origin (DECK)
# ---------------------------------------------------------------------------

class DeckOriginTests(unittest.TestCase):
    def test_initial_deck_hand_cards_have_deck_origin(self):
        engine = _make_engine()
        engine.reset(seed=10)
        hand = [c for c in engine.players[0].hand if hasattr(c, 'origin')]
        self.assertTrue(len(hand) > 0)
        self.assertTrue(all(c.origin == CardOrigin.DECK for c in hand))

    def test_play_follower_keeps_deck_origin(self):
        engine = _make_engine()
        engine.reset(seed=10)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        board = engine.players[0].board
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0].origin, CardOrigin.DECK)

    def test_death_from_play_has_deck_origin_in_graveyard(self):
        engine = _make_engine()
        engine.reset(seed=10)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(0))
        gy = [g for g in engine.players[0].graveyard
              if g.definition.card_id == unit.definition.card_id]
        self.assertEqual(len(gy), 1)
        self.assertEqual(gy[0].origin, CardOrigin.DECK)
        self.assertFalse(gy[0].derived)
        self.assertFalse(gy[0].token)

    def test_destroyed_follower_record_has_no_derived_flag_for_deck_card(self):
        engine = _make_engine()
        engine.reset(seed=10)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(0))
        dfr = engine.state.destroyed_followers[-1]
        self.assertFalse(dfr.derived)
        self.assertFalse(dfr.token)

    def test_overdraw_to_graveyard_has_deck_origin(self):
        engine = _make_engine()
        engine.reset(seed=10)
        for _ in range(9):
            engine._draw(0, reason="fill")
        engine._draw(0, reason="overdraw")
        gy = engine.players[0].graveyard
        self.assertTrue(len(gy) > 0)
        self.assertEqual(gy[-1].origin, CardOrigin.DECK)

    def test_generated_spell_keeps_origin_when_resolved_to_graveyard(self):
        spell = _card(100, card_type="法术", name="GeneratedSpell", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.HEAL_LEADER,
                target=TargetKind.OWN_LEADER,
                amount=1,
            )),
            card_defs={100: spell},
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell, origin=CardOrigin.GENERATED)
        source_entity_id = engine.players[0].hand[0].entity_id

        _play(engine)

        graveyard_card = next(
            card
            for card in engine.players[0].graveyard
            if card.entity_id == source_entity_id
        )
        self.assertEqual(graveyard_card.origin, CardOrigin.GENERATED)
        self.assertTrue(graveyard_card.derived)


# ---------------------------------------------------------------------------
# ADD_CARD origin
# ---------------------------------------------------------------------------

class AddCardOriginTests(unittest.TestCase):
    def test_add_card_to_hand_has_generated_origin(self):
        spell = _card(100, card_type="法术", name="AddCard", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.ADD_CARD,
                target=TargetKind.OWN_LEADER,
                card_id=900,
            )),
            card_defs={
                100: spell,
                900: _card(900, name="Added"),
            },
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell)
        _play(engine)
        hand = engine.players[0].hand
        added = [c for c in hand if hasattr(c, 'card_id') and c.card_id == 900]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].origin, CardOrigin.GENERATED)

    def test_add_token_card_to_hand_has_token_origin(self):
        spell = _card(100, card_type="法术", name="AddToken", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.ADD_CARD,
                target=TargetKind.OWN_LEADER,
                card_id=90099,
            )),
            card_defs={
                100: spell,
                90099: _token_card(90099, name="Token"),
            },
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell)
        _play(engine)
        hand = engine.players[0].hand
        added = [c for c in hand if hasattr(c, 'card_id') and c.card_id == 90099]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].origin, CardOrigin.TOKEN)

    def test_add_card_full_hand_discards_to_graveyard_with_origin(self):
        spell = _card(100, card_type="法术", name="AddCard", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.ADD_CARD,
                target=TargetKind.OWN_LEADER,
                card_id=900,
            )),
            card_defs={
                100: spell,
                900: _card(900, name="Added"),
            },
        )
        engine.reset(seed=10)
        for _ in range(9):
            engine._draw(0, reason="fill")
        _insert_spell(engine, spell)
        _play(engine)
        gy = [g for g in engine.players[0].graveyard if g.definition.card_id == 900]
        self.assertEqual(len(gy), 1)
        self.assertEqual(gy[0].origin, CardOrigin.GENERATED)
        self.assertTrue(gy[0].derived)


# ---------------------------------------------------------------------------
# SUMMON origin
# ---------------------------------------------------------------------------

class SummonOriginTests(unittest.TestCase):
    def test_effect_summon_follower_has_generated_origin(self):
        spell = _card(100, card_type="法术", name="Summon", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=901,
            )),
            card_defs={
                100: spell,
                901: _card(901, name="Summoned"),
            },
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell)
        _play(engine)
        board = engine.players[0].board
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0].origin, CardOrigin.GENERATED)

    def test_effect_summon_token_has_token_origin(self):
        spell = _card(100, card_type="法术", name="SummonToken", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=90098,
            )),
            card_defs={
                100: spell,
                90098: _token_card(90098, name="SummonedToken"),
            },
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell)
        _play(engine)
        board = engine.players[0].board
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0].origin, CardOrigin.TOKEN)

    def test_full_board_summon_noop_does_not_change_cooperation(self):
        spell = _card(100, card_type="法术", name="Summon", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=901,
            )),
            card_defs={
                100: spell,
                901: _card(901, name="Summoned"),
            },
        )
        engine.reset(seed=10)
        for i in range(5):
            engine.players[0].board.append(Unit.summon(
                _card(400 + i), entity_id=400 + i
            ))
        _insert_spell(engine, spell)
        _play(engine)
        self.assertEqual(len(engine.players[0].board), 5)
        self.assertEqual(engine.players[0].cooperation, 0)


# ---------------------------------------------------------------------------
# Token death propagation
# ---------------------------------------------------------------------------

class TokenDeathPropagationTests(unittest.TestCase):
    def test_token_death_has_derived_in_graveyard(self):
        spell = _card(100, card_type="法术", name="SummonToken", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=90098,
            )),
            card_defs={
                100: spell,
                90098: _token_card(90098, name="SummonedToken"),
            },
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell)
        _play(engine)
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(0))
        gy = engine.players[0].graveyard
        token_entry = [g for g in gy if g.definition.card_id == 90098]
        self.assertEqual(len(token_entry), 1)
        self.assertEqual(token_entry[0].origin, CardOrigin.TOKEN)
        self.assertTrue(token_entry[0].derived)
        self.assertTrue(token_entry[0].token)

    def test_token_death_in_destroyed_follower_record(self):
        spell = _card(100, card_type="法术", name="SummonToken", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=90098,
            )),
            card_defs={
                100: spell,
                90098: _token_card(90098, name="SummonedToken"),
            },
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell)
        _play(engine)
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(0))
        dfr = engine.state.destroyed_followers[-1]
        self.assertTrue(dfr.derived)
        self.assertTrue(dfr.token)


# ---------------------------------------------------------------------------
# Reanimate origin
# ---------------------------------------------------------------------------

class ReanimateOriginTests(unittest.TestCase):
    def test_reanimate_has_reanimated_origin(self):
        spell = _card(100, card_type="法术", name="Reanimate", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.REANIMATE,
                target=TargetKind.OWN_LEADER,
                amount=10,
            )),
            card_defs={
                100: spell,
                300: _card(300, name="Huge", cost=8),
            },
        )
        engine.reset(seed=10)
        engine.state.destroyed_followers.append(
            DestroyedFollowerRecord(
                definition=_card(300, name="Huge", cost=8),
                owner=0,
                death_sequence=1,
                cause=DeathCause.COMBAT,
            )
        )
        _insert_spell(engine, spell)
        _play(engine)
        board = engine.players[0].board
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0].origin, CardOrigin.REANIMATED)
        self.assertTrue(is_derived(board[0].origin))

    def test_reanimate_creates_new_entity_id(self):
        spell = _card(100, card_type="法术", name="Reanimate", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.REANIMATE,
                target=TargetKind.OWN_LEADER,
                amount=10,
            )),
            card_defs={
                100: spell,
                300: _card(300, name="Huge", cost=8),
            },
        )
        engine.reset(seed=10)
        engine.state.destroyed_followers.append(
            DestroyedFollowerRecord(
                definition=_card(300, name="Huge", cost=8),
                owner=0,
                death_sequence=1,
                cause=DeathCause.COMBAT,
            )
        )
        _insert_spell(engine, spell)
        _play(engine)
        unit = engine.players[0].board[0]
        self.assertTrue(unit.entity_id > 0)

    def test_reanimate_no_candidates_skips(self):
        spell = _card(100, card_type="法术", name="Reanimate", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.REANIMATE,
                target=TargetKind.OWN_LEADER,
                amount=10,
            )),
            card_defs={
                100: spell,
            },
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell)
        _play(engine)
        self.assertEqual(len(engine.players[0].board), 0)

    def test_reanimate_board_full_skips(self):
        spell = _card(100, card_type="法术", name="Reanimate", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.REANIMATE,
                target=TargetKind.OWN_LEADER,
                amount=10,
            )),
            card_defs={
                100: spell,
                300: _card(300, name="Huge", cost=8),
            },
        )
        engine.reset(seed=10)
        for i in range(5):
            engine.players[0].board.append(Unit.summon(
                _card(400 + i), entity_id=400 + i
            ))
        engine.state.destroyed_followers.append(
            DestroyedFollowerRecord(
                definition=_card(300, name="Huge", cost=8),
                owner=0,
                death_sequence=1,
                cause=DeathCause.COMBAT,
            )
        )
        _insert_spell(engine, spell)
        _play(engine)
        self.assertEqual(len(engine.players[0].board), 5)

    def test_reanimate_preserves_root_source_origin(self):
        spell = _card(100, card_type="法术", name="Reanimate", cost=1)
        follower = _card(300, name="GeneratedFollower", cost=3)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.REANIMATE,
                target=TargetKind.OWN_LEADER,
                amount=10,
            )),
            card_defs={100: spell, 300: follower},
        )
        engine.reset(seed=10)
        engine.state.destroyed_followers.append(
            DestroyedFollowerRecord(
                definition=follower,
                owner=0,
                death_sequence=1,
                cause=DeathCause.COMBAT,
                origin=CardOrigin.REANIMATED,
                source_origin=CardOrigin.GENERATED,
            )
        )
        _insert_spell(engine, spell)

        _play(engine)

        unit = engine.players[0].board[0]
        self.assertEqual(unit.origin, CardOrigin.REANIMATED)
        self.assertEqual(unit.source_origin, CardOrigin.GENERATED)

    def test_reanimate_candidate_policy_is_used_by_resolution(self):
        spell = _card(100, card_type="法术", name="Reanimate", cost=1)
        follower = _card(300, name="Candidate", cost=3)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.REANIMATE,
                target=TargetKind.OWN_LEADER,
                amount=10,
            )),
            card_defs={100: spell, 300: follower},
        )
        engine.reset(seed=10)
        engine.state.destroyed_followers.append(
            DestroyedFollowerRecord(
                definition=follower,
                owner=0,
                death_sequence=1,
                cause=DeathCause.COMBAT,
            )
        )
        _insert_spell(engine, spell)

        with patch(
            "swb.engine.resolution.is_reanimate_eligible",
            return_value=False,
        ):
            _play(engine)

        self.assertEqual(engine.players[0].board, [])


# ---------------------------------------------------------------------------
# Transform origin
# ---------------------------------------------------------------------------

class TransformOriginTests(unittest.TestCase):
    def test_transform_sets_transformed_origin(self):
        engine = _make_engine()
        engine.reset(seed=10)
        engine.players[0].mana = 10
        engine.card_resolver = lambda cid: _card(cid, name="Result")
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        old_eid = unit.entity_id
        self.assertEqual(unit.origin, CardOrigin.DECK)
        op = EffectOperation(
            kind=EffectKind.TRANSFORM,
            target=TargetKind.SELF,
            card_id=999,
        )
        engine._start_effects(unit.definition, unit.entity_id, (op,), label="test")
        engine._continue_effects()
        self.assertEqual(unit.entity_id, old_eid)
        self.assertEqual(unit.origin, CardOrigin.TRANSFORMED)

    def test_transform_death_has_transformed_origin_in_graveyard(self):
        engine = _make_engine()
        engine.reset(seed=10)
        engine.players[0].mana = 10
        engine.card_resolver = lambda cid: _card(cid, name="Result")
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        op = EffectOperation(
            kind=EffectKind.TRANSFORM,
            target=TargetKind.SELF,
            card_id=999,
        )
        engine._start_effects(unit.definition, unit.entity_id, (op,), label="test")
        engine._continue_effects()
        unit.health = 0
        engine.apply(EndTurn(0))
        gy = engine.players[0].graveyard
        transformed = [g for g in gy if g.definition.card_id == 999]
        self.assertEqual(len(transformed), 1)
        self.assertEqual(transformed[0].origin, CardOrigin.TRANSFORMED)
        self.assertTrue(transformed[0].derived)


# ---------------------------------------------------------------------------
# Graveyard return / summon preserves origin
# ---------------------------------------------------------------------------

class GraveyardPreservationTests(unittest.TestCase):
    def test_summon_from_graveyard_preserves_origin(self):
        spell = _card(100, card_type="法术", name="GraveSummon", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON_FROM_GRAVEYARD,
                target=TargetKind.OWN_GRAVEYARD_CARD,
                graveyard_card_type="随从",
            )),
            card_defs={
                100: spell,
            },
        )
        engine.reset(seed=10)
        engine.players[0].graveyard.append(
            GraveyardCard(
                definition=_card(777, name="GraveCard"),
                entity_id=7777,
                owner=0,
                entered_sequence=1,
                entry_cause="combat",
                origin=CardOrigin.TOKEN,
                derived=True,
                token=True,
            )
        )
        _insert_spell(engine, spell)
        _play(engine)
        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        option = request.options[0]
        engine.apply(Choose(0, option.option_id))
        board = engine.players[0].board
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0].origin, CardOrigin.TOKEN)

    def test_return_from_graveyard_preserves_origin(self):
        spell = _card(100, card_type="法术", name="GraveReturn", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND,
                target=TargetKind.OWN_GRAVEYARD_CARD,
                graveyard_card_type="随从",
            )),
            card_defs={
                100: spell,
            },
        )
        engine.reset(seed=10)
        engine.players[0].graveyard.append(
            GraveyardCard(
                definition=_card(777, name="GraveCard"),
                entity_id=7777,
                owner=0,
                entered_sequence=1,
                entry_cause="combat",
                origin=CardOrigin.GENERATED,
                derived=True,
                token=False,
            )
        )
        _insert_spell(engine, spell)
        _play(engine)
        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        option = request.options[0]
        engine.apply(Choose(0, option.option_id))
        hand = [c for c in engine.players[0].hand
                if hasattr(c, 'entity_id') and c.entity_id == 7777]
        self.assertEqual(len(hand), 1)
        self.assertEqual(hand[0].origin, CardOrigin.GENERATED)

    def test_return_candidate_policy_is_used_before_choice(self):
        spell = _card(100, card_type="法术", name="GraveReturn", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: spell},
        )
        engine.reset(seed=10)
        engine.players[0].graveyard.append(
            GraveyardCard(
                definition=_card(777, name="GraveCard"),
                entity_id=7777,
                owner=0,
                entered_sequence=1,
                entry_cause="combat",
            )
        )
        _insert_spell(engine, spell)

        with patch(
            "swb.engine.targeting.is_graveyard_return_eligible",
            return_value=False,
        ):
            with self.assertRaises(IllegalCommand):
                _play(engine)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[0].graveyard[0].entity_id, 7777)


# ---------------------------------------------------------------------------
# Return to hand / replay
# ---------------------------------------------------------------------------

class ReturnReplayTests(unittest.TestCase):
    def test_return_to_hand_preserves_deck_origin(self):
        engine = _make_engine()
        engine.reset(seed=10)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        self.assertEqual(unit.origin, CardOrigin.DECK)
        card_id = unit.definition.card_id
        op = EffectOperation(
            kind=EffectKind.RETURN_TO_HAND,
            target=TargetKind.SELF,
        )
        engine._start_effects(unit.definition, unit.entity_id, (op,), label="test")
        engine._continue_effects()
        hand = engine.players[0].hand
        returned = [c for c in hand
                    if hasattr(c, 'card_id') and c.card_id == card_id]
        self.assertEqual(len(returned), 1)
        self.assertEqual(returned[0].origin, CardOrigin.DECK)

    def test_replay_returned_card_keeps_deck_origin(self):
        engine = _make_engine()
        engine.reset(seed=10)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        card_id = unit.definition.card_id
        op = EffectOperation(
            kind=EffectKind.RETURN_TO_HAND,
            target=TargetKind.SELF,
        )
        engine._start_effects(unit.definition, unit.entity_id, (op,), label="test")
        engine._continue_effects()
        hand = engine.players[0].hand
        returned = [c for c in hand
                    if hasattr(c, 'card_id') and c.card_id == card_id]
        self.assertEqual(len(returned), 1)
        self.assertEqual(returned[0].origin, CardOrigin.DECK)
        engine.players[0].mana = 10
        idx = hand.index(returned[0])
        engine.apply(PlayCard(0, idx))
        board = engine.players[0].board
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0].origin, CardOrigin.DECK)

    def test_return_to_deck_then_draw_resets_origin_to_deck(self):
        engine = _make_engine()
        engine.reset(seed=10)
        generated = _card(777, name="Generated")
        unit = Unit.summon(
            generated,
            entity_id=engine.state.allocate_entity_id(),
            origin=CardOrigin.GENERATED,
        )
        engine.players[0].board.append(unit)
        op = EffectOperation(
            kind=EffectKind.RETURN_TO_DECK,
            target=TargetKind.SELF,
        )
        engine._start_effects(generated, unit.entity_id, (op,), label="test")
        engine._continue_effects()
        engine.players[0].deck = [generated]
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()

        engine._draw(0, reason="test")

        self.assertEqual(engine.players[0].hand[0].origin, CardOrigin.DECK)


# ---------------------------------------------------------------------------
# Cooperation: single increment
# ---------------------------------------------------------------------------

class CooperationOriginTests(unittest.TestCase):
    def test_play_follower_increments_cooperation_once(self):
        engine = _make_engine()
        engine.reset(seed=10)
        engine.players[0].mana = 10
        self.assertEqual(engine.players[0].cooperation, 0)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].cooperation, 1)

    def test_effect_summon_increments_cooperation_once(self):
        spell = _card(100, card_type="法术", name="Summon", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=901,
            )),
            card_defs={
                100: spell,
                901: _card(901, name="Summoned"),
            },
        )
        engine.reset(seed=10)
        self.assertEqual(engine.players[0].cooperation, 0)
        _insert_spell(engine, spell)
        _play(engine)
        self.assertEqual(engine.players[0].cooperation, 1)

    def test_full_board_summon_no_cooperation(self):
        spell = _card(100, card_type="法术", name="Summon", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=901,
            )),
            card_defs={
                100: spell,
                901: _card(901, name="Summoned"),
            },
        )
        engine.reset(seed=10)
        for i in range(5):
            engine.players[0].board.append(Unit.summon(
                _card(400 + i), entity_id=400 + i
            ))
        _insert_spell(engine, spell)
        _play(engine)
        self.assertEqual(engine.players[0].cooperation, 0)

    def test_reanimate_increments_cooperation_once(self):
        spell = _card(100, card_type="法术", name="Reanimate", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.REANIMATE,
                target=TargetKind.OWN_LEADER,
                amount=10,
            )),
            card_defs={
                100: spell,
                300: _card(300, name="Huge", cost=8),
            },
        )
        engine.reset(seed=10)
        engine.state.destroyed_followers.append(
            DestroyedFollowerRecord(
                definition=_card(300, name="Huge", cost=8),
                owner=0,
                death_sequence=1,
                cause=DeathCause.COMBAT,
            )
        )
        self.assertEqual(engine.players[0].cooperation, 0)
        _insert_spell(engine, spell)
        _play(engine)
        self.assertEqual(engine.players[0].cooperation, 1)


# ---------------------------------------------------------------------------
# Transform does not increment cooperation
# ---------------------------------------------------------------------------

class TransformCooperationTests(unittest.TestCase):
    def test_transform_does_not_increment_cooperation(self):
        engine = _make_engine()
        engine.reset(seed=10)
        engine.players[0].mana = 10
        engine.card_resolver = lambda cid: _card(cid, name="Result")
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].cooperation, 1)
        unit = engine.players[0].board[0]
        op = EffectOperation(
            kind=EffectKind.TRANSFORM,
            target=TargetKind.SELF,
            card_id=999,
        )
        engine._start_effects(unit.definition, unit.entity_id, (op,), label="test")
        engine._continue_effects()
        self.assertEqual(engine.players[0].cooperation, 1)


# ---------------------------------------------------------------------------
# Event metadata
# ---------------------------------------------------------------------------

class EventMetadataTests(unittest.TestCase):
    def test_follower_summoned_event_has_origin_metadata(self):
        spell = _card(100, card_type="法术", name="Summon", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=901,
            )),
            card_defs={
                100: spell,
                901: _card(901, name="Summoned"),
            },
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell)
        _play(engine)
        events = engine.event_history
        summoned = [e for e in events if e.type == EventType.FOLLOWER_SUMMONED]
        self.assertTrue(len(summoned) > 0)
        evt = summoned[-1]
        self.assertIn("origin", evt.metadata)
        self.assertIn("derived", evt.metadata)
        self.assertIn("token", evt.metadata)
        self.assertEqual(evt.metadata["origin"], "generated")

    def test_graveyard_entered_has_origin_metadata(self):
        spell = _card(100, card_type="法术", name="Summon", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON,
                target=TargetKind.OWN_LEADER,
                card_id=901,
            )),
            card_defs={
                100: spell,
                901: _card(901, name="Summoned"),
            },
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell)
        _play(engine)
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(0))
        events = engine.event_history
        gy_events = [e for e in events if e.type == EventType.GRAVEYARD_ENTERED]
        self.assertTrue(len(gy_events) > 0)
        evt = gy_events[-1]
        self.assertIn("origin", evt.metadata)
        self.assertIn("derived", evt.metadata)
        self.assertIn("token", evt.metadata)

    def test_card_added_to_hand_has_origin_metadata(self):
        spell = _card(100, card_type="法术", name="AddCard", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.ADD_CARD,
                target=TargetKind.OWN_LEADER,
                card_id=900,
            )),
            card_defs={
                100: spell,
                900: _card(900, name="Added"),
            },
        )
        engine.reset(seed=10)
        _insert_spell(engine, spell)
        _play(engine)
        events = engine.event_history
        added = [e for e in events if e.type == EventType.CARD_ADDED_TO_HAND]
        self.assertTrue(len(added) > 0)
        evt = added[-1]
        self.assertIn("origin", evt.metadata)
        self.assertIn("derived", evt.metadata)
        self.assertIn("token", evt.metadata)

    def test_reanimate_resolved_has_origin_metadata(self):
        spell = _card(100, card_type="法术", name="Reanimate", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.REANIMATE,
                target=TargetKind.OWN_LEADER,
                amount=10,
            )),
            card_defs={
                100: spell,
                300: _card(300, name="Huge", cost=8),
            },
        )
        engine.reset(seed=10)
        engine.state.destroyed_followers.append(
            DestroyedFollowerRecord(
                definition=_card(300, name="Huge", cost=8),
                owner=0,
                death_sequence=1,
                cause=DeathCause.COMBAT,
            )
        )
        _insert_spell(engine, spell)
        _play(engine)
        events = engine.event_history
        resolved = [e for e in events if e.type == EventType.REANIMATE_RESOLVED]
        self.assertTrue(len(resolved) > 0)
        evt = resolved[-1]
        self.assertIn("origin", evt.metadata)
        self.assertEqual(evt.metadata["origin"], "reanimated")

    def test_graveyard_card_summoned_has_origin_metadata(self):
        spell = _card(100, card_type="法术", name="GraveSummon", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON_FROM_GRAVEYARD,
                target=TargetKind.OWN_GRAVEYARD_CARD,
                graveyard_card_type="随从",
            )),
            card_defs={
                100: spell,
            },
        )
        engine.reset(seed=10)
        engine.players[0].graveyard.append(
            GraveyardCard(
                definition=_card(777, name="GraveCard"),
                entity_id=7777,
                owner=0,
                entered_sequence=1,
                entry_cause="combat",
                origin=CardOrigin.TOKEN,
                derived=True,
                token=True,
            )
        )
        _insert_spell(engine, spell)
        _play(engine)
        request = engine.state.pending_choice
        option = request.options[0]
        engine.apply(Choose(0, option.option_id))
        events = engine.event_history
        gcs = [e for e in events if e.type == EventType.GRAVEYARD_CARD_SUMMONED]
        self.assertTrue(len(gcs) > 0)
        evt = gcs[-1]
        self.assertIn("origin", evt.metadata)
        self.assertIn("derived", evt.metadata)
        self.assertIn("token", evt.metadata)

    def test_graveyard_card_returned_has_origin_metadata(self):
        spell = _card(100, card_type="法术", name="GraveReturn", cost=1)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND,
                target=TargetKind.OWN_GRAVEYARD_CARD,
                graveyard_card_type="随从",
            )),
            card_defs={
                100: spell,
            },
        )
        engine.reset(seed=10)
        engine.players[0].graveyard.append(
            GraveyardCard(
                definition=_card(777, name="GraveCard"),
                entity_id=7777,
                owner=0,
                entered_sequence=1,
                entry_cause="combat",
                origin=CardOrigin.GENERATED,
                derived=True,
                token=False,
            )
        )
        _insert_spell(engine, spell)
        _play(engine)
        request = engine.state.pending_choice
        option = request.options[0]
        engine.apply(Choose(0, option.option_id))
        events = engine.event_history
        gcr = [e for e in events if e.type == EventType.GRAVEYARD_CARD_RETURNED]
        self.assertTrue(len(gcr) > 0)
        evt = gcr[-1]
        self.assertIn("origin", evt.metadata)
        self.assertIn("derived", evt.metadata)
        self.assertIn("token", evt.metadata)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_origin(self):
        engine1 = _make_engine(seed=42)
        state1 = engine1.reset(seed=42)
        engine2 = _make_engine(seed=42)
        state2 = engine2.reset(seed=42)
        origins1 = [c.origin for c in engine1.players[0].hand if hasattr(c, 'origin')]
        origins2 = [c.origin for c in engine2.players[0].hand if hasattr(c, 'origin')]
        self.assertEqual(origins1, origins2)
        self.assertTrue(all(o == CardOrigin.DECK for o in origins1))

    def test_entity_id_uniqueness(self):
        engine = _make_engine(seed=42)
        engine.reset(seed=42)
        seen: set[int] = set()
        for player in engine.state.players:
            for c in player.hand:
                if hasattr(c, 'entity_id'):
                    self.assertNotIn(c.entity_id, seen)
                    seen.add(c.entity_id)


# ---------------------------------------------------------------------------
# Observation / RL invariants
# ---------------------------------------------------------------------------

class ObservationInvariantTests(unittest.TestCase):
    def test_observation_dimension_is_227(self):
        env = ShadowverseEnv(
            deck_a=[_card(100 + i) for i in range(40)],
            deck_b=[_card(200 + i) for i in range(40)],
            class_a=1, class_b=1,
            seed=42,
        )
        obs, _ = env.reset(seed=42)
        self.assertEqual(len(obs), 227)

    def test_action_size_is_111(self):
        self.assertEqual(ShadowverseEnv.ACTION_SIZE, 111)

    def test_opponent_hand_not_leaked(self):
        env = ShadowverseEnv(
            deck_a=[_card(100 + i) for i in range(40)],
            deck_b=[_card(200 + i) for i in range(40)],
            class_a=1, class_b=1,
            seed=42,
        )
        obs, _ = env.reset(seed=42)
        self.assertEqual(len(obs), 227)


# ---------------------------------------------------------------------------
# Real database verification
# ---------------------------------------------------------------------------

class RealCardVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from swb.db.repository import CardRepository
        db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(str(db_path)):
            raise unittest.SkipTest("cards.sqlite3 not found")
        cls.repo = CardRepository(str(db_path))

    def test_token_card_90044110_is_non_collectible(self):
        try:
            card = self.repo.get(90044110)
        except KeyError:
            self.skipTest("Card 90044110 not in database")
        self.assertFalse(card.is_collectible)
        self.assertEqual(card.card_set_id, 90000)
        self.assertFalse(is_initial_deck_eligible(card))
        self.assertTrue(is_token_definition(card))
        self.assertIn("霸道", card.name)

    def test_other_token_cards_from_set_90000(self):
        token_ids = [90043110, 90042110]
        found = False
        for cid in token_ids:
            try:
                card = self.repo.get(cid)
                found = True
                self.assertEqual(card.card_set_id, 90000)
                self.assertFalse(card.is_collectible)
                self.assertTrue(is_token_definition(card))
            except KeyError:
                continue
        self.assertTrue(found, "No token cards from the test list were found")

    def test_normal_collectible_card_is_deck_eligible(self):
        try:
            card = self.repo.get(10001110)
        except KeyError:
            self.skipTest("Card 10001110 not in database")
        self.assertTrue(card.is_collectible)
        self.assertNotEqual(card.card_set_id, 90000)
        self.assertTrue(is_initial_deck_eligible(card))
        self.assertFalse(is_token_definition(card))


if __name__ == "__main__":
    unittest.main()
