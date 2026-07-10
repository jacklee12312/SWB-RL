# -*- coding: utf-8 -*-
"""Tests for graveyard, shadows, Necromancy, and Reanimate."""

from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard, EndTurn
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import (
    DeathCause,
    DestroyedFollowerRecord,
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
    deck_card_id_start: int = 1000,
    seed: int = 42,
) -> GameEngine:
    """Create a GameEngine with the given rules and card defs."""
    all_defs = dict(card_defs) if card_defs else {}
    deck_a = [_card(i, name=f"deck-a-{i}") for i in range(deck_card_id_start, deck_card_id_start + 40)]
    deck_b = [_card(i, name=f"deck-b-{i}") for i in range(deck_card_id_start + 100, deck_card_id_start + 140)]
    return GameEngine(
        deck_a=deck_a,
        deck_b=deck_b,
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=RuleBook(tuple(rules)),
        card_resolver=_make_resolver(all_defs),
    )


# ---------------------------------------------------------------------------
# Graveyard entry and shadow counters
# ---------------------------------------------------------------------------


class GraveyardEntryTests(unittest.TestCase):
    def test_destroyed_follower_increments_shadows_once(self):
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
        before = engine.players[0].shadows
        engine.apply(EndTurn(0))
        after = engine.players[0].shadows
        self.assertEqual(after - before, 1)

    def test_return_to_hand_does_not_increment_shadows(self):
        target = _card(200, name="Target", cost=2)
        engine = _make_engine(
            _spell_rule(300, EffectOperation(
                kind=EffectKind.RETURN_TO_HAND, target=TargetKind.ENEMY_UNIT,
            )),
            card_defs={200: target, 300: _card(300, card_type="法术", name="Return", cost=4)},
        )
        engine.reset(seed=42)
        engine.apply(EndTurn(0))
        engine.apply(PlayCard(1, 0))
        engine.apply(EndTurn(1))
        # p0 now at 2 mana - need to cheat hand
        engine.players[0].mana = 10
        sp = _card(300, card_type="法术", name="Return", cost=4)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        before = engine.players[0].shadows
        engine.apply(PlayCard(0, 0))
        if engine.state.pending_choice:
            engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))
        # The return spell itself goes to graveyard (+1 shadow), but
        # the returned follower does NOT increment shadows.
        self.assertEqual(engine.players[0].shadows, before + 1,
                         "only the spell should give shadow, not the returned follower")

    def test_shadows_changed_emitted_for_gain(self):
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
        shadows_events = [
            e for e in engine.event_history
            if e.type is EventType.SHADOWS_CHANGED
        ]
        self.assertGreater(len(shadows_events), 0)

    def test_shadows_changed_emitted_for_consumption(self):
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.NECROMANCY, amount=2, target=TargetKind.OWN_LEADER,
                necromancy_operations=(
                    EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
                ),
            )),
            card_defs={100: _card(100, card_type="法术", name="NSpell", cost=4)},
        )
        engine.reset(seed=42)
        engine.players[0].shadows = 5
        engine.players[0].mana = 10
        sp = engine.card_resolver(100)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        spend_events = [
            e for e in engine.event_history
            if e.type is EventType.SHADOWS_CHANGED and e.metadata.get("change") == "spend"
        ]
        self.assertGreater(len(spend_events), 0)
        # start 5, necro consume 2, spell burial +1 = 4
        self.assertEqual(engine.players[0].shadows, 4)


# ---------------------------------------------------------------------------
# Necromancy behavior
# ---------------------------------------------------------------------------


class NecromancyTests(unittest.TestCase):
    def _make_necromancy_engine(self, necro_op: EffectOperation, *, seed: int = 42) -> GameEngine:
        return _make_engine(
            _spell_rule(100, necro_op),
            card_defs={100: _card(100, card_type="法术", name="NSpell", cost=4)},
            seed=seed,
        )

    def _setup_necromancy(self, engine: GameEngine, shadows: int) -> None:
        engine.reset(seed=42)
        engine.players[0].shadows = shadows
        engine.players[0].mana = 10
        sp = engine.card_resolver(100)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)

    def test_insufficient_shadows_skips(self):
        necro_op = EffectOperation(
            kind=EffectKind.NECROMANCY, amount=4, target=TargetKind.OWN_LEADER,
            necromancy_operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=3),
            ),
        )
        engine = self._make_necromancy_engine(necro_op)
        self._setup_necromancy(engine, shadows=2)
        opp_hp = engine.players[1].health
        engine.apply(PlayCard(0, 0))
        # start 2 shadows, insufficient so necro skipped, spell burial +1 = 3
        self.assertEqual(engine.players[0].shadows, 3)
        self.assertEqual(engine.players[1].health, opp_hp)

    def test_sufficient_shadows_spends_once_and_executes(self):
        necro_op = EffectOperation(
            kind=EffectKind.NECROMANCY, amount=3, target=TargetKind.OWN_LEADER,
            necromancy_operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=4),
            ),
        )
        engine = self._make_necromancy_engine(necro_op)
        self._setup_necromancy(engine, shadows=5)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].shadows, 3)
        self.assertEqual(engine.players[1].health, 16)

    def test_necromancy_multi_op_executes_all_in_order(self):
        necro_op = EffectOperation(
            kind=EffectKind.NECROMANCY, amount=3, target=TargetKind.OWN_LEADER,
            necromancy_operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=2),
            ),
        )
        engine = self._make_necromancy_engine(necro_op)
        self._setup_necromancy(engine, shadows=5)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].shadows, 3)
        self.assertEqual(engine.players[1].health, 17)

    def test_necromancy_choice_pauses_and_resumes_without_repeat_payment(self):
        target_card = _card(200, name="Target", cost=2, attack=1, life=3)
        engine = _make_engine(
            _spell_rule(100, EffectOperation(
                kind=EffectKind.NECROMANCY, amount=1, target=TargetKind.OWN_LEADER,
                necromancy_operations=(
                    EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=2),
                    EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
                ),
            )),
            card_defs={100: _card(100, card_type="法术", name="NSpell", cost=4), 200: target_card},
        )
        engine.reset(seed=42)
        engine.players[0].shadows = 5
        # give p1 a target
        engine.apply(EndTurn(0))
        engine.players[1].mana = 10
        engine.players[1].hand.insert(0, HandCard(definition=target_card, entity_id=engine.state.allocate_entity_id()))
        engine.players[1].hand_entity_ids.insert(0, engine.players[1].hand[0].entity_id)
        engine.apply(PlayCard(1, 0))
        engine.apply(EndTurn(1))
        # now p0's turn, give them the necro spell
        engine.players[0].mana = 10
        sp = engine.card_resolver(100)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)
        engine.apply(PlayCard(0, 0))
        # should have paused
        self.assertIsNotNone(engine.state.pending_choice)
        shadows_after_pause = engine.players[0].shadows
        self.assertEqual(shadows_after_pause, 4, "shadows consumed exactly once during pause")
        engine.apply(Choose(0, engine.state.pending_choice.options[0].option_id))
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[1].health, 19, "second op: enemy leader took 1")
        self.assertEqual(engine.players[0].shadows, 5, "no second payment")

    def test_opponent_last_words_necromancy_keeps_controller(self):
        necro_op = EffectOperation(
            kind=EffectKind.NECROMANCY,
            amount=1,
            target=TargetKind.OWN_LEADER,
            necromancy_operations=(
                EffectOperation(
                    kind=EffectKind.DAMAGE_LEADER,
                    target=TargetKind.ENEMY_LEADER,
                    amount=3,
                ),
            ),
        )
        engine = _make_engine(
            CardRule(900, Trigger.LAST_WORDS, (necro_op,))
        )
        engine.reset(seed=42)
        last_words_unit = Unit.summon(
            _card(900, name="Opponent LW"),
            entity_id=engine.state.allocate_entity_id(),
        )
        last_words_unit.health = 0
        engine.players[1].board = [last_words_unit]
        engine.players[1].shadows = 2

        engine._stabilize()

        self.assertEqual(engine.players[0].health, 17)
        self.assertEqual(engine.players[1].health, 20)
        self.assertEqual(engine.players[1].shadows, 2)
        activated = [
            event
            for event in engine.event_history
            if event.type is EventType.NECROMANCY_ACTIVATED
        ]
        self.assertEqual(len(activated), 1)
        self.assertEqual(activated[0].player_index, 1)


# ---------------------------------------------------------------------------
# Reanimate behavior
# ---------------------------------------------------------------------------


class ReanimateTests(unittest.TestCase):
    def _make_reanimate_engine(self, reanimate_op: EffectOperation, *, seed: int = 42) -> GameEngine:
        return _make_engine(
            _spell_rule(100, reanimate_op),
            card_defs={100: _card(100, card_type="法术", name="Reanimate", cost=1)},
            seed=seed,
        )

    def _setup_reanimate(self, engine: GameEngine, records: list[DestroyedFollowerRecord]) -> None:
        engine.reset(seed=42)
        engine.state.destroyed_followers = list(records)
        engine.players[0].mana = 10
        sp = engine.card_resolver(100)
        engine.players[0].hand.insert(0, HandCard(definition=sp, entity_id=engine.state.allocate_entity_id()))
        engine.players[0].hand_entity_ids.insert(0, engine.players[0].hand[0].entity_id)

    def test_reanimate_chooses_highest_cost(self):
        f_low = _card(200, name="LowCost", cost=2, attack=2, life=2)
        f_high = _card(201, name="HighCost", cost=3, attack=3, life=3)
        engine = self._make_reanimate_engine(EffectOperation(
            kind=EffectKind.REANIMATE, amount=5, target=TargetKind.OWN_LEADER,
        ))
        self._setup_reanimate(engine, [
            DestroyedFollowerRecord(definition=f_low, owner=0, death_sequence=1, cause=DeathCause.COMBAT),
            DestroyedFollowerRecord(definition=f_high, owner=0, death_sequence=2, cause=DeathCause.COMBAT),
        ])
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        assert isinstance(unit, Unit)
        self.assertEqual(unit.definition.card_id, 201)

    def test_reanimate_deterministic_tie_breaking(self):
        f_a = _card(200, name="TieA", cost=2, attack=2, life=2)
        f_b = _card(201, name="TieB", cost=2, attack=2, life=2)
        results = []
        for trial_seed in (42, 42):
            engine = self._make_reanimate_engine(EffectOperation(
                kind=EffectKind.REANIMATE, amount=2, target=TargetKind.OWN_LEADER,
            ), seed=trial_seed)
            self._setup_reanimate(engine, [
                DestroyedFollowerRecord(definition=f_a, owner=0, death_sequence=1, cause=DeathCause.COMBAT),
                DestroyedFollowerRecord(definition=f_b, owner=0, death_sequence=2, cause=DeathCause.COMBAT),
            ])
            engine.apply(PlayCard(0, 0))
            u = engine.players[0].board[0]
            assert isinstance(u, Unit)
            results.append(u.definition.card_id)
        self.assertEqual(results[0], results[1])

    def test_reanimate_resets_state(self):
        f_d = _card(200, name="Dead", cost=3, attack=3, life=4)
        engine = self._make_reanimate_engine(EffectOperation(
            kind=EffectKind.REANIMATE, amount=3, target=TargetKind.OWN_LEADER,
        ))
        self._setup_reanimate(engine, [
            DestroyedFollowerRecord(definition=f_d, owner=0, death_sequence=1, cause=DeathCause.COMBAT),
        ])
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        assert isinstance(unit, Unit)
        self.assertGreater(unit.entity_id, 0)
        self.assertEqual(unit.attack, 3)
        self.assertEqual(unit.health, 4)
        self.assertFalse(unit.evolved)

    def test_reanimate_no_fanfare(self):
        f_d = _card(200, name="Dead", cost=2, attack=2, life=2)
        engine = self._make_reanimate_engine(EffectOperation(
            kind=EffectKind.REANIMATE, amount=2, target=TargetKind.OWN_LEADER,
        ))
        self._setup_reanimate(engine, [
            DestroyedFollowerRecord(definition=f_d, owner=0, death_sequence=1, cause=DeathCause.COMBAT),
        ])
        engine.apply(PlayCard(0, 0))
        # Reanimate summons via effect, not play; no card_played event should fire
        card_played_events = [
            e for e in engine.event_history
            if e.type is EventType.CARD_PLAYED
            and e.metadata.get("card_id") == 200
        ]
        self.assertEqual(len(card_played_events), 0)

    def test_reanimate_board_full_skips(self):
        f_d = _card(200, name="Dead", cost=3, attack=3, life=3)
        engine = self._make_reanimate_engine(EffectOperation(
            kind=EffectKind.REANIMATE, amount=3, target=TargetKind.OWN_LEADER,
        ))
        self._setup_reanimate(engine, [
            DestroyedFollowerRecord(definition=f_d, owner=0, death_sequence=1, cause=DeathCause.COMBAT),
        ])
        for i in range(engine.config.max_board):
            engine.players[0].board.append(
                Unit.summon(_card(900 + i), entity_id=engine.state.allocate_entity_id())
            )
        board_before = len(engine.players[0].board)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].board), board_before)

    def test_reanimate_no_candidates_skips(self):
        engine = self._make_reanimate_engine(EffectOperation(
            kind=EffectKind.REANIMATE, amount=3, target=TargetKind.OWN_LEADER,
        ))
        self._setup_reanimate(engine, [])
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].board), 0)

    def test_reanimate_updates_cooperation(self):
        f_d = _card(200, name="Dead", cost=2, attack=2, life=2)
        engine = self._make_reanimate_engine(EffectOperation(
            kind=EffectKind.REANIMATE, amount=2, target=TargetKind.OWN_LEADER,
        ))
        self._setup_reanimate(engine, [
            DestroyedFollowerRecord(definition=f_d, owner=0, death_sequence=1, cause=DeathCause.COMBAT),
        ])
        before = engine.players[0].cooperation
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].cooperation, before + 1)

    def test_reanimate_event_emitted(self):
        f_d = _card(200, name="Dead", cost=2, attack=2, life=2)
        engine = self._make_reanimate_engine(EffectOperation(
            kind=EffectKind.REANIMATE, amount=2, target=TargetKind.OWN_LEADER,
        ))
        self._setup_reanimate(engine, [
            DestroyedFollowerRecord(definition=f_d, owner=0, death_sequence=1, cause=DeathCause.COMBAT),
        ])
        engine.apply(PlayCard(0, 0))
        reanimate_events = [
            e for e in engine.event_history
            if e.type is EventType.REANIMATE_RESOLVED
        ]
        self.assertGreater(len(reanimate_events), 0)


# ---------------------------------------------------------------------------
# DeathCause preservation
# ---------------------------------------------------------------------------


class DeathCauseTests(unittest.TestCase):
    def test_death_cause_preserved_in_graveyard_event(self):
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
        grave_events = [
            e for e in engine.event_history
            if e.type is EventType.GRAVEYARD_ENTERED
        ]
        self.assertGreater(len(grave_events), 0)
        self.assertIn("cause", grave_events[-1].metadata)
        cause = grave_events[-1].metadata["cause"]
        self.assertNotEqual(cause, "destroyed", f"expected real DeathCause, got {cause}")

    def test_destroyed_follower_record_preserves_cause(self):
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
        self.assertGreater(len(engine.state.destroyed_followers), 0)
        for rec in engine.state.destroyed_followers:
            self.assertIn(rec.cause.value, {"combat", "zero_health", "effect_destroy"})


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class SchemaValidationTests(unittest.TestCase):
    def test_necromancy_amount_rejects_bool(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaises(ValueError):
            _parse_operation({
                "kind": "necromancy",
                "amount": True,
                "operations": [{"kind": "damage_leader", "target": "enemy_leader", "amount": 1}],
            }, "test.json", 1)

    def test_necromancy_amount_rejects_string(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaises(ValueError):
            _parse_operation({
                "kind": "necromancy",
                "amount": "4",
                "operations": [{"kind": "damage_leader", "target": "enemy_leader", "amount": 1}],
            }, "test.json", 1)

    def test_necromancy_amount_rejects_float(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaises(ValueError):
            _parse_operation({
                "kind": "necromancy",
                "amount": 3.5,
                "operations": [{"kind": "damage_leader", "target": "enemy_leader", "amount": 1}],
            }, "test.json", 1)

    def test_necromancy_amount_rejects_negative(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaises(ValueError):
            _parse_operation({
                "kind": "necromancy",
                "amount": -1,
                "operations": [{"kind": "damage_leader", "target": "enemy_leader", "amount": 1}],
            }, "test.json", 1)

    def test_necromancy_requires_operations(self):
        from swb.engine.card_rules import _parse_operation
        with self.assertRaises(ValueError):
            _parse_operation({
                "kind": "necromancy",
                "amount": 2,
            }, "test.json", 1)

    def test_nested_target_key_validated(self):
        from swb.engine.card_rules import _validate_target_keys
        with self.assertRaises(ValueError):
            _validate_target_keys((
                EffectOperation(
                    kind=EffectKind.NECROMANCY, target=TargetKind.OWN_LEADER, amount=1,
                    necromancy_operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT, target=TargetKind.PREVIOUS_TARGET,
                            target_key="nonexistent",
                        ),
                    ),
                ),
            ), "test.json")

    def test_shadow_leaf_expression_rejects_values(self):
        from swb.engine.card_rules import _parse_expression
        with self.assertRaises(ValueError):
            _parse_expression(
                {
                    "type": "controller_shadows",
                    "values": [{"type": "constant", "value": 99}],
                },
                "test.json/amount",
                1,
            )


# ---------------------------------------------------------------------------
# Reset and observation
# ---------------------------------------------------------------------------


class ResetAndObservationTests(unittest.TestCase):
    def test_reset_clears_shadows_and_destroyed_followers(self):
        engine = _make_engine()
        engine.reset(seed=42)
        engine.players[0].shadows = 10
        engine.state.destroyed_followers = [
            DestroyedFollowerRecord(definition=_card(1), owner=0, death_sequence=1, cause=DeathCause.COMBAT),
        ]
        engine.reset(seed=42)
        self.assertEqual(engine.players[0].shadows, 0)
        self.assertEqual(len(engine.state.destroyed_followers), 0)

    def test_observation_includes_public_shadows(self):
        from swb.engine.environment import ShadowverseEnv
        deck_a = [_card(i) for i in range(1000, 1040)]
        deck_b = [_card(i) for i in range(1100, 1140)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=RuleBook(),
        )
        env.reset(seed=42)
        env.players[0].shadows = 7
        env.players[1].shadows = 3

        observation = env.observation()
        self.assertEqual(len(observation), 225)
        self.assertEqual(observation[16], 7 / 20)
        self.assertEqual(observation[17], 3 / 20)

        env.core.state.active_player = 1
        reversed_observation = env.observation()
        self.assertEqual(reversed_observation[16], 3 / 20)
        self.assertEqual(reversed_observation[17], 7 / 20)


class RealCardNecromancyTests(unittest.TestCase):
    def test_database_cards_match_structured_rules(self):
        repository = CardRepository("data/cards.sqlite3")
        mummy = repository.get(10051130)
        soulmancer = repository.get(10551120)
        rulebook = RuleBook.from_directory("data/rules")

        self.assertEqual(mummy.name, "恶毒的小木乃伊")
        self.assertEqual(soulmancer.name, "红符的魂魄道士")

        mummy_fanfare = rulebook.operations_for(10051130, Trigger.FANFARE)
        self.assertEqual(len(mummy_fanfare), 1)
        self.assertEqual(mummy_fanfare[0].kind, EffectKind.NECROMANCY)
        self.assertEqual(mummy_fanfare[0].amount, 4)
        self.assertEqual(
            mummy_fanfare[0].necromancy_operations[0].kind,
            EffectKind.ADD_KEYWORD,
        )

        for trigger in (Trigger.FANFARE, Trigger.EVOLVE):
            operations = rulebook.operations_for(10551120, trigger)
            self.assertEqual(len(operations), 1)
            self.assertEqual(operations[0].kind, EffectKind.REANIMATE)
            self.assertEqual(operations[0].amount, 2)

    def test_real_necromancy_card_executes_without_placeholder(self):
        repository = CardRepository("data/cards.sqlite3")
        mummy = repository.get(10051130)
        rulebook = RuleBook.from_directory("data/rules")
        deck = [mummy] * 40
        engine = GameEngine(
            deck,
            deck,
            class_a=mummy.class_id,
            class_b=mummy.class_id,
            seed=42,
            rulebook=rulebook,
            card_resolver=repository.get,
        )
        engine.reset(seed=42)
        engine.players[0].shadows = 4
        engine.players[0].mana = 10
        engine.players[0].hand = [
            HandCard(
                definition=mummy,
                entity_id=engine.state.allocate_entity_id(),
            )
        ]
        engine.players[0].hand_entity_ids = [
            engine.players[0].hand[0].entity_id
        ]

        engine.apply(PlayCard(0, 0))

        unit = engine.players[0].board[0]
        self.assertTrue(unit.has_keyword("疾驰"))
        unsupported = [
            event
            for event in engine.placeholder_ability_events
            if event.card_id == mummy.card_id
            and event.ability.value == "死灵术"
        ]
        self.assertEqual(unsupported, [])


if __name__ == "__main__":
    unittest.main()
