# -*- coding: utf-8 -*-
"""Tests for universal graveyard card interaction system."""

from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, PlayCard, EndTurn
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import (
    DeathCause,
    DestroyedFollowerRecord,
    GraveyardCard,
    HandCard,
    Phase,
    Unit,
)


def _card(
    card_id: int,
    *,
    attack: int | None = 1,
    life: int | None = 1,
    cost: int = 1,
    card_type: str = "随从",
    name: str | None = None,
    is_collectible: bool = True,
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=name or f"card-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack,
        life=life,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=is_collectible,
    )


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


# ---------------------------------------------------------------------------
# GraveyardCard entity model
# ---------------------------------------------------------------------------


class GraveyardEntityTests(unittest.TestCase):
    def test_graveyard_card_has_stable_entity_id(self):
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON, target=TargetKind.OWN_BOARD, card_id=1001,
            )),
            card_defs={100: _card(100, card_type="法术", name="Summon", cost=1)},
        )
        engine.reset(seed=42)
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        assert isinstance(unit, Unit)
        eid_before = unit.entity_id
        unit.health = 0
        engine.apply(EndTurn(0))
        gc = engine.players[0].graveyard[-1]
        self.assertEqual(gc.entity_id, eid_before)
        self.assertEqual(gc.owner, 0)
        self.assertGreater(gc.entered_sequence, 0)

    def test_card_not_in_multiple_zones(self):
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON, target=TargetKind.OWN_BOARD, card_id=1001,
            )),
            card_defs={100: _card(100, card_type="法术", name="Summon", cost=1)},
        )
        engine.reset(seed=42)
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        assert isinstance(unit, Unit)
        eid = unit.entity_id
        unit.health = 0
        engine.apply(EndTurn(0))
        self.assertNotIn(eid, [e.entity_id for e in engine.players[0].board])
        self.assertIn(eid, [g.entity_id for g in engine.players[0].graveyard])

    def test_destroyed_followers_separate_from_graveyard(self):
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON, target=TargetKind.OWN_BOARD, card_id=1001,
            )),
            card_defs={100: _card(100, card_type="法术", name="Summon", cost=1)},
        )
        engine.reset(seed=42)
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        assert isinstance(unit, Unit)
        unit.health = 0
        engine.apply(EndTurn(0))
        self.assertGreater(len(engine.players[0].graveyard), 0)
        self.assertGreater(len(engine.state.destroyed_followers), 0)
        gc = engine.players[0].graveyard[-1]
        df = engine.state.destroyed_followers[-1]
        self.assertIsNot(gc, df)
        self.assertIs(gc.definition, df.definition)

    def test_graveyard_entries_have_unique_sequences(self):
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON, target=TargetKind.OWN_BOARD, card_id=1001,
            )),
            card_defs={100: _card(100, card_type="法术", name="Summon", cost=1)},
        )
        engine.reset(seed=42)
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        assert isinstance(unit, Unit)
        unit.health = 0
        engine.apply(EndTurn(0))
        seqs = [g.entered_sequence for g in engine.players[0].graveyard]
        self.assertEqual(len(seqs), len(set(seqs)))

    def test_reset_clears_graveyard_state(self):
        engine = _make_engine()
        engine.reset(seed=42)
        engine.players[0].graveyard.append(
            GraveyardCard(definition=_card(1), entity_id=99, owner=0, entered_sequence=1, entry_cause="test")
        )
        engine.players[0].shadows = 10
        engine.reset(seed=42)
        self.assertEqual(len(engine.players[0].graveyard), 0)
        self.assertEqual(engine.players[0].shadows, 0)

    def test_entry_cause_preserved(self):
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON, target=TargetKind.OWN_BOARD, card_id=1001,
            )),
            card_defs={100: _card(100, card_type="法术", name="Summon", cost=1)},
        )
        engine.reset(seed=42)
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        assert isinstance(unit, Unit)
        unit.health = 0
        engine.apply(EndTurn(0))
        gc = engine.players[0].graveyard[-1]
        self.assertIn(gc.entry_cause, {"zero_health", "combat"})


# ---------------------------------------------------------------------------
# Graveyard targeting
# ---------------------------------------------------------------------------


class GraveyardTargetingTests(unittest.TestCase):
    def _setup_graveyard(self, engine, entries):
        engine.reset(seed=42)
        for i, card in enumerate(entries):
            engine.players[0].graveyard.append(
                GraveyardCard(
                    definition=card, entity_id=card.card_id * 100,
                    owner=0, entered_sequence=i + 1, entry_cause="test",
                )
            )

    def test_own_graveyard_choice_produces_options(self):
        f1 = _card(200, name="F1", cost=2)
        f2 = _card(201, name="F2", cost=3)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="ReturnG", cost=4), 200: f1, 201: f2},
        )
        self._setup_graveyard(engine, [f1, f2])
        engine.players[0].mana = 10
        sp = engine.card_resolver(100)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        opts = engine.state.pending_choice.options
        self.assertEqual(len(opts), 2)

    def test_own_graveyard_choice_selects_and_returns_to_hand(self):
        f1 = _card(200, name="F1", cost=2)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="ReturnG", cost=4), 200: f1},
        )
        self._setup_graveyard(engine, [f1])
        engine.players[0].mana = 10
        sp = engine.card_resolver(100)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        hand_before = len(engine.players[0].hand)
        grave_before = len(engine.players[0].graveyard)
        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))
        self.assertEqual(len(engine.players[0].hand), hand_before)
        self.assertEqual(len(engine.players[0].graveyard), grave_before)

    def test_random_graveyard_deterministic(self):
        f1 = _card(200, name="F1", cost=2)
        f2 = _card(201, name="F2", cost=3)
        results = []
        for trial_seed in (42, 42):
            engine = _make_engine(
                _spell_rule(100, EffectOperation(
                    kind=EffectKind.BANISH_FROM_GRAVEYARD,
                    target=TargetKind.RANDOM_OWN_GRAVEYARD_CARD,
                )),
                card_defs={100: _card(100, card_type="法术", name="BanishG", cost=4), 200: f1, 201: f2},
                seed=trial_seed,
            )
            self._setup_graveyard(engine, [f1, f2])
            engine.players[0].mana = 10
            sp = engine.card_resolver(100)
            engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
            engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
            engine.apply(PlayCard(0, 0))
            results.append(len(engine.players[0].graveyard))
        self.assertEqual(results[0], results[1])

    def test_all_graveyard_banishes_all(self):
        f1 = _card(200, name="F1", cost=2)
        f2 = _card(201, name="F2", cost=3)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.BANISH_FROM_GRAVEYARD,
                target=TargetKind.ALL_OWN_GRAVEYARD_CARDS,
            )),
            card_defs={100: _card(100, card_type="法术", name="BanishAllG", cost=4)},
        )
        self._setup_graveyard(engine, [f1, f2])
        engine.players[0].mana = 10
        sp = engine.card_resolver(100)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].graveyard), 1)
        self.assertEqual(len(engine.players[0].banished), 2)

    def test_choice_target_leaving_graveyard_safe(self):
        f1 = _card(200, name="F1", cost=2)
        f2 = _card(201, name="F2", cost=3)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.BANISH_FROM_GRAVEYARD,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="BanishG", cost=4)},
        )
        self._setup_graveyard(engine, [f1, f2])
        engine.players[0].mana = 10
        sp = engine.card_resolver(100)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        # remove the chosen target to simulate it leaving
        chosen_id = engine.state.pending_choice.options[0].entity_id
        engine.players[0].graveyard = [g for g in engine.players[0].graveyard if g.entity_id != chosen_id]
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))
        self.assertIsNone(engine.state.pending_choice)

    def test_graveyard_filter_card_type(self):
        f1 = _card(200, name="F1", cost=2, card_type="随从")
        s1 = _card(201, name="S1", cost=2, card_type="法术")
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND,
                target=TargetKind.RANDOM_OWN_GRAVEYARD_CARD,
                graveyard_card_type="随从",
            )),
            card_defs={100: _card(100, card_type="法术", name="Filter", cost=4), 200: f1, 201: s1},
        )
        self._setup_graveyard(engine, [f1, s1])
        engine.players[0].mana = 10
        sp = engine.card_resolver(100)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        remaining = engine.players[0].graveyard
        self.assertTrue(any(g.definition.card_type == "法术" for g in remaining))


# ---------------------------------------------------------------------------
# Graveyard effect operations
# ---------------------------------------------------------------------------


class GraveyardOperationTests(unittest.TestCase):
    def _setup_graveyard(self, engine, entries):
        engine.reset(seed=42)
        for i, card in enumerate(entries):
            engine.players[0].graveyard.append(
                GraveyardCard(
                    definition=card, entity_id=card.card_id * 100,
                    owner=0, entered_sequence=i + 1, entry_cause="test",
                )
            )

    def _play_spell(self, engine, spell_card_id, spell_def):
        engine.players[0].mana = 10
        engine.players[0].hand.insert(0, HandCard(definition=spell_def, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))

    def test_successful_summon_from_graveyard(self):
        f1 = _card(200, name="F1", cost=2, attack=2, life=2)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON_FROM_GRAVEYARD,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="SummonG", cost=4), 200: f1},
        )
        self._setup_graveyard(engine, [f1])
        sp = _card(100, card_type="法术", name="SummonG", cost=4)
        self._play_spell(engine, 100, sp)
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))
        self.assertEqual(len(engine.players[0].board), 1)
        self.assertEqual(len(engine.players[0].graveyard), 1)
        unit = engine.players[0].board[0]
        assert isinstance(unit, Unit)
        self.assertEqual(unit.attack, 2)
        self.assertEqual(unit.health, 2)
        self.assertFalse(unit.can_attack)

    def test_summon_from_graveyard_board_full_skips(self):
        f1 = _card(200, name="F1", cost=2, attack=2, life=2)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON_FROM_GRAVEYARD,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="SummonG", cost=4), 200: f1},
        )
        self._setup_graveyard(engine, [f1])
        for i in range(engine.config.max_board):
            engine.players[0].board.append(Unit.summon(_card(900 + i), entity_id=engine.state.allocate_entity_id()))
        sp = _card(100, card_type="法术", name="SummonG", cost=4)
        self._play_spell(engine, 100, sp)
        self.assertEqual(len(engine.players[0].board), engine.config.max_board)

    def test_return_to_hand_full_hand_safely_skips(self):
        f1 = _card(200, name="F1", cost=2)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="ReturnG", cost=4), 200: f1},
        )
        self._setup_graveyard(engine, [f1])
        for i in range(engine.config.max_hand):
            engine.players[0].hand.append(HandCard(definition=_card(300 + i), entity_id=engine.state.allocate_entity_id()))
            engine.players[0].hand_entity_ids.append(engine.players[0].hand[-1].entity_id)
        sp = _card(100, card_type="法术", name="ReturnG", cost=4)
        self._play_spell(engine, 100, sp)
        self.assertEqual(len(engine.players[0].graveyard), 1)

    def test_banish_from_graveyard_no_shadow_increase(self):
        f1 = _card(200, name="F1", cost=2)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.BANISH_FROM_GRAVEYARD,
                target=TargetKind.RANDOM_OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="BanishG", cost=4), 200: f1},
        )
        self._setup_graveyard(engine, [f1])
        shadows_before = engine.players[0].shadows
        sp = _card(100, card_type="法术", name="BanishG", cost=4)
        self._play_spell(engine, 100, sp)
        shadows_after = engine.players[0].shadows
        self.assertEqual(shadows_after, shadows_before + 1)

    def test_entity_gone_after_removal(self):
        f1 = _card(200, name="F1", cost=2)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON_FROM_GRAVEYARD,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="SummonG", cost=4), 200: f1},
        )
        self._setup_graveyard(engine, [f1])
        gc_eid = engine.players[0].graveyard[0].entity_id
        sp = _card(100, card_type="法术", name="SummonG", cost=4)
        self._play_spell(engine, 100, sp)
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))
        self.assertFalse(any(g.entity_id == gc_eid for g in engine.players[0].graveyard))

    def test_events_emitted(self):
        f1 = _card(200, name="F1", cost=2)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON_FROM_GRAVEYARD,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="SummonG", cost=4), 200: f1},
        )
        self._setup_graveyard(engine, [f1])
        sp = _card(100, card_type="法术", name="SummonG", cost=4)
        self._play_spell(engine, 100, sp)
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))
        summoned = [e for e in engine.event_history if e.type is EventType.GRAVEYARD_CARD_SUMMONED]
        self.assertGreater(len(summoned), 0)
        self.assertEqual(summoned[0].metadata["from_zone"], "graveyard")
        self.assertEqual(summoned[0].metadata["to_zone"], "board")

    def test_no_fanfare_from_graveyard_summon(self):
        f1 = _card(200, name="F1", cost=2, attack=2, life=2)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.SUMMON_FROM_GRAVEYARD,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="SummonG", cost=4), 200: f1},
        )
        self._setup_graveyard(engine, [f1])
        sp = _card(100, card_type="法术", name="SummonG", cost=4)
        self._play_spell(engine, 100, sp)
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))
        card_played = [e for e in engine.event_history if e.type is EventType.CARD_PLAYED and e.metadata.get("card_id") == 200]
        self.assertEqual(len(card_played), 0)

    def test_banish_moves_to_banished_zone(self):
        f1 = _card(200, name="F1", cost=2)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.BANISH_FROM_GRAVEYARD,
                target=TargetKind.OWN_GRAVEYARD_CARD,
            )),
            card_defs={100: _card(100, card_type="法术", name="BanishG", cost=4), 200: f1},
        )
        self._setup_graveyard(engine, [f1])
        sp = _card(100, card_type="法术", name="BanishG", cost=4)
        self._play_spell(engine, 100, sp)
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))
        self.assertEqual(len(engine.players[0].graveyard), 1)
        self.assertEqual(len(engine.players[0].banished), 1)


# ---------------------------------------------------------------------------
# JSON Schema
# ---------------------------------------------------------------------------


class GraveyardSchemaTests(unittest.TestCase):
    def test_parse_return_from_graveyard(self):
        op = _parse_operation({
            "kind": "return_from_graveyard_to_hand",
            "target": "own_graveyard_card",
        }, "test.json", 1)
        self.assertEqual(op.kind, EffectKind.RETURN_FROM_GRAVEYARD_TO_HAND)
        self.assertEqual(op.target, TargetKind.OWN_GRAVEYARD_CARD)

    def test_parse_summon_from_graveyard(self):
        op = _parse_operation({
            "kind": "summon_from_graveyard",
            "target": "random_own_graveyard_card",
        }, "test.json", 1)
        self.assertEqual(op.kind, EffectKind.SUMMON_FROM_GRAVEYARD)

    def test_parse_banish_from_graveyard(self):
        op = _parse_operation({
            "kind": "banish_from_graveyard",
            "target": "all_own_graveyard_cards",
        }, "test.json", 1)
        self.assertEqual(op.kind, EffectKind.BANISH_FROM_GRAVEYARD)

    def test_parse_graveyard_filter_params(self):
        op = _parse_operation({
            "kind": "return_from_graveyard_to_hand",
            "target": "random_own_graveyard_card",
            "cost_max": 4,
            "cost_min": 2,
            "follower_only": True,
            "card_type_filter": "随从",
        }, "test.json", 1)
        self.assertEqual(op.graveyard_cost_max, 4)
        self.assertEqual(op.graveyard_cost_min, 2)
        self.assertTrue(op.graveyard_follower_only)
        self.assertEqual(op.graveyard_card_type, "随从")

    def test_unknown_graveyard_target_rejected(self):
        with self.assertRaises(ValueError):
            _parse_operation({
                "kind": "return_from_graveyard_to_hand",
                "target": "enemy_graveyard_card",
            }, "test.json", 1)

    def test_old_rulebook_remains_compatible(self):
        from swb.engine.card_rules import RuleBook
        rulebook = RuleBook()
        self.assertIsInstance(rulebook, RuleBook)


if __name__ == "__main__":
    unittest.main()
