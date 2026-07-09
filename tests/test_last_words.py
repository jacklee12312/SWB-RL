from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard, EndTurn
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.emblem import EmblemDefinition, EmblemTriggerRule, EventScope
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand, MAX_RESOLUTION_STEPS, ResolutionLoopError
from swb.engine.state import Amulet, Phase, Unit


def card(
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


class AtomicAllTargetTests(unittest.TestCase):
    """Tests that all-target damage hits all before any death."""

    def test_all_target_damage_hits_all_before_any_death(self):
        """3 damage to units with 1/1/2 health: all hit before any dies."""
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=3),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        a = Unit.summon(card(900, attack=1, life=1))
        b = Unit.summon(card(901, attack=1, life=1))
        c = Unit.summon(card(902, attack=1, life=2))
        engine.players[1].board = [a, b, c]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(len(engine.players[1].graveyard), 3)

    def test_simultaneous_death_both_sides(self):
        """Units on both sides die together, not one side then the other."""
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        a = Unit.summon(card(900, attack=1, life=2))
        b = Unit.summon(card(901, attack=1, life=3))
        engine.players[0].board = [a]
        engine.players[1].board = [b]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].board, [])
        self.assertEqual(engine.players[1].board, [])
        self.assertTrue(len(engine.players[0].graveyard) >= 1)
        self.assertTrue(len(engine.players[1].graveyard) >= 1)

    def test_all_target_accounts_for_target_leaving_early(self):
        """If a target is gone by the time all-target resolves, skip it safely."""
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_OWN_UNITS, amount=1),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        a = Unit.summon(card(900, attack=1, life=1))
        engine.players[0].board = [a]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].board, [])
        self.assertTrue(any(g.definition.card_id == 900 for g in engine.players[0].graveyard))


class DeathBatchTests(unittest.TestCase):
    """Tests for the death batch model."""

    def test_death_batch_events_in_correct_order(self):
        """DEATH_BATCH_START → FOLLOWER_DESTROYED → ENTITY_LEFT_PLAY → ... → DEATH_BATCH_END."""
        rulebook = RuleBook((
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = [Unit.summon(card(900, attack=1, life=2))]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        transition = engine.apply(PlayCard(0, 0))
        types = [e.type for e in transition.events]
        self.assertIn(EventType.DEATH_BATCH_START, types)
        self.assertIn(EventType.FOLLOWER_DESTROYED, types)
        self.assertIn(EventType.DEATH_BATCH_END, types)
        start_idx = types.index(EventType.DEATH_BATCH_START)
        end_idx = types.index(EventType.DEATH_BATCH_END)
        destroyed_idx = types.index(EventType.FOLLOWER_DESTROYED)
        self.assertLess(start_idx, destroyed_idx)
        self.assertLess(destroyed_idx, end_idx)

    def test_mixed_follower_amulet_batch_events_share_batch_id_before_lw(self):
        """Mixed destroyed followers/amulets emit batch diagnostics before LW."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DESTROY, target=TargetKind.ALL_ENEMY_BOARD),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        follower = Unit.summon(card(900, attack=1, life=2), entity_id=900)
        amulet = Amulet(
            definition=card(
                901, card_type="护符", attack=None, life=None,
            ),
            entity_id=901,
        )
        engine.players[1].board = [follower, amulet]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        transition = engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900, 901]],
        )
        events = transition.events
        indexed = list(enumerate(events))
        batch_start = next(
            i for i, event in indexed
            if event.type == EventType.DEATH_BATCH_START
            and event.metadata["batch_id"] == 1
        )
        first_lw = next(
            i for i, event in indexed
            if event.type == EventType.LAST_WORDS_START
        )
        batch_end = next(
            i for i, event in indexed
            if event.type == EventType.DEATH_BATCH_END
            and event.metadata["batch_id"] == 1
        )
        death_event_indexes = [
            i for i, event in indexed
            if event.type in {
                EventType.FOLLOWER_DESTROYED,
                EventType.AMULET_DESTROYED,
                EventType.ENTITY_LEFT_PLAY,
            }
            and event.metadata["batch_id"] == 1
        ]
        self.assertEqual(len(death_event_indexes), 4)
        self.assertTrue(all(batch_start < i < first_lw for i in death_event_indexes))

        lw_events = [
            event for event in events
            if event.type in {
                EventType.LAST_WORDS_START,
                EventType.LAST_WORDS_COMPLETE,
            }
            and event.metadata["card_id"] in {900, 901}
        ]
        self.assertEqual(len(lw_events), 4)
        self.assertTrue(all(event.metadata["batch_id"] == 1 for event in lw_events))
        self.assertLess(first_lw, batch_end)

    def test_same_batch_death_order_deterministic(self):
        """Same board state with same seed produces identical death order."""
        def run_and_get_death_order():
            rulebook = RuleBook((
                CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                    EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=10),
                ),),
            ))
            engine = GameEngine(
                [card(i) for i in range(100, 140)],
                [card(i) for i in range(200, 240)],
                class_a=1, class_b=1, seed=3, rulebook=rulebook,
            )
            engine.reset(seed=3)
            engine.players[1].board = [
                Unit.summon(card(900, attack=1, life=3)),
                Unit.summon(card(901, attack=1, life=3)),
                Unit.summon(card(902, attack=1, life=3)),
            ]
            engine.players[0].mana = 10
            engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
            transition = engine.apply(PlayCard(0, 0))
            return [e.source_id for e in transition.events if e.type == EventType.FOLLOWER_DESTROYED]

        order1 = run_and_get_death_order()
        order2 = run_and_get_death_order()
        self.assertEqual(order1, order2)

    def test_death_batch_metadata_exposes_active_player_order(self):
        """Death-batch diagnostics expose the exact order used for LW resolution."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=902, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=903, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=3, rulebook=rulebook,
        )
        engine.reset(seed=3)
        engine.players[0].board = [
            Unit.summon(card(900, attack=1, life=3), entity_id=9000),
            Unit.summon(card(901, attack=1, life=3), entity_id=9001),
        ]
        engine.players[1].board = [
            Unit.summon(card(902, attack=1, life=3), entity_id=9002),
            Unit.summon(card(903, attack=1, life=3), entity_id=9003),
        ]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        transition = engine.apply(PlayCard(0, 0))

        batch_start = next(
            event for event in transition.events
            if event.type == EventType.DEATH_BATCH_START
        )
        self.assertEqual(batch_start.metadata["batch_id"], 1)
        self.assertEqual(batch_start.metadata["active_player"], 0)
        self.assertEqual(batch_start.metadata["count"], 4)
        self.assertEqual(
            [
                (
                    record["batch_order_index"],
                    record["owner"],
                    record["card_id"],
                    record["board_position"],
                )
                for record in batch_start.metadata["ordered_records"]
            ],
            [
                (0, 0, 900, 0),
                (1, 0, 901, 1),
                (2, 1, 902, 0),
                (3, 1, 903, 1),
            ],
        )

        lw_starts = [
            event for event in transition.events
            if event.type == EventType.LAST_WORDS_START
        ]
        self.assertEqual([event.metadata["card_id"] for event in lw_starts], [900, 901, 902, 903])
        self.assertEqual([event.metadata["batch_order_index"] for event in lw_starts], [0, 1, 2, 3])
        self.assertTrue(all(event.metadata["batch_record_count"] == 4 for event in lw_starts))
        self.assertTrue(all(event.metadata["active_player"] == 0 for event in lw_starts))

        destroyed = [
            event for event in transition.events
            if event.type == EventType.FOLLOWER_DESTROYED
        ]
        left_play = [
            event for event in transition.events
            if event.type == EventType.ENTITY_LEFT_PLAY
            and event.metadata["batch_id"] == 1
        ]
        self.assertEqual([event.metadata["batch_order_index"] for event in destroyed], [0, 1, 2, 3])
        self.assertEqual([event.metadata["batch_order_index"] for event in left_play], [0, 1, 2, 3])

    def test_cross_player_mixed_batch_metadata_counts_followers_and_amulets(self):
        """Mixed follower/amulet deaths across both players expose composition."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=902, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=903, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DESTROY, target=TargetKind.ALL_BOARD),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=4, rulebook=rulebook,
        )
        engine.reset(seed=4)
        engine.players[0].board = [
            Unit.summon(card(900, attack=1, life=3), entity_id=9000),
            Amulet(
                definition=card(901, card_type="护符", attack=None, life=None),
                entity_id=9001,
            ),
        ]
        engine.players[1].board = [
            Unit.summon(card(902, attack=1, life=3), entity_id=9002),
            Amulet(
                definition=card(903, card_type="护符", attack=None, life=None),
                entity_id=9003,
            ),
        ]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        transition = engine.apply(PlayCard(0, 0))

        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900, 901, 902, 903]],
        )
        events = transition.events
        batch_events = [
            event for event in events
            if event.type in {EventType.DEATH_BATCH_START, EventType.DEATH_BATCH_END}
            and event.metadata["batch_id"] == 1
        ]
        self.assertEqual(len(batch_events), 2)
        for event in batch_events:
            self.assertEqual(event.metadata["follower_count"], 2)
            self.assertEqual(event.metadata["amulet_count"], 2)
            self.assertEqual(
                event.metadata["owner_counts"],
                [
                    {"owner": 0, "record_count": 2, "follower_count": 1, "amulet_count": 1},
                    {"owner": 1, "record_count": 2, "follower_count": 1, "amulet_count": 1},
                ],
            )

        batch_start = batch_events[0]
        self.assertEqual(
            [
                (record["batch_order_index"], record["owner"], record["card_id"], record["card_type"])
                for record in batch_start.metadata["ordered_records"]
            ],
            [
                (0, 0, 900, "随从"),
                (1, 0, 901, "护符"),
                (2, 1, 902, "随从"),
                (3, 1, 903, "护符"),
            ],
        )
        lifecycle_events = [
            event for event in events
            if event.type in {
                EventType.FOLLOWER_DESTROYED,
                EventType.AMULET_DESTROYED,
                EventType.ENTITY_LEFT_PLAY,
                EventType.LAST_WORDS_START,
                EventType.LAST_WORDS_COMPLETE,
            }
            and event.metadata["batch_id"] == 1
        ]
        self.assertTrue(lifecycle_events)
        self.assertTrue(all(event.metadata["batch_follower_count"] == 2 for event in lifecycle_events))
        self.assertTrue(all(event.metadata["batch_amulet_count"] == 2 for event in lifecycle_events))


class LastWordsTests(unittest.TestCase):
    """Tests for Last Words execution."""

    def test_last_words_draw_on_follower_death(self):
        """Follower with Last Words: draw 1 card."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        lw_unit = Unit.summon(card(900, attack=1, life=2))
        engine.players[1].board = [lw_unit]
        hand_before = len(engine.players[1].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        transition = engine.apply(choice)
        self.assertIn(EventType.LAST_WORDS_START, [e.type for e in transition.events])
        self.assertEqual(len(engine.players[1].hand), hand_before + 1)

    def test_last_words_summon_follower(self):
        """Last Words: summon a follower."""
        token = card(700, attack=2, life=2, is_collectible=False)
        def resolver(cid):
            return token if cid == 700 else None
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.SUMMON, target=TargetKind.SELF, card_id=700),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook, card_resolver=resolver,
        )
        engine.reset(seed=1)
        lw_unit = Unit.summon(card(900, attack=1, life=2))
        engine.players[1].board = [lw_unit]
        board_before = len(engine.players[1].board)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertTrue(any(u.definition.card_id == 700 for u in engine.players[1].board if hasattr(u, 'attack')))

    def test_last_words_causes_second_death_batch(self):
        """Last Words that damage all units creates a second death batch."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_UNITS, amount=2),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        lw_unit = Unit.summon(card(900, attack=1, life=2))
        nearby = Unit.summon(card(901, attack=1, life=1))
        engine.players[1].board = [lw_unit, nearby]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(len(engine.players[1].board), 0)
        batch_sizes = [len(b.records) for b in engine.state.death_queue]
        self.assertEqual(batch_sizes, [1, 1])

    def test_same_batch_last_words_complete_before_new_death_batch(self):
        """Deaths caused by an early LW wait until the original batch completes."""
        token = card(902, attack=0, life=1, is_collectible=False)

        def resolver(cid):
            return token if cid == 902 else None

        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.SUMMON, target=TargetKind.OWN_LEADER, card_id=902),
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_OWN_UNITS, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=902, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=2),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
            card_resolver=resolver,
        )
        engine.reset(seed=1)
        engine.players[1].board = [
            Unit.summon(card(900, attack=1, life=2)),
            Unit.summon(card(901, attack=1, life=2)),
        ]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        transition = engine.apply(PlayCard(0, 0))

        batch_card_ids = [[record.card_id for record in batch.records] for batch in engine.state.death_queue]
        self.assertEqual(batch_card_ids, [[900, 901], [902]])
        self.assertEqual(engine.players[0].health, 17)

        events = transition.events
        batch_starts = [i for i, event in enumerate(events) if event.type == EventType.DEATH_BATCH_START]
        batch_ends = [i for i, event in enumerate(events) if event.type == EventType.DEATH_BATCH_END]
        lw_starts = {
            event.metadata["card_id"]: i
            for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
        }

        self.assertEqual(len(batch_starts), 2)
        self.assertEqual(len(batch_ends), 2)
        self.assertLess(lw_starts[900], lw_starts[901])
        self.assertLess(lw_starts[901], batch_ends[0])
        self.assertLess(batch_ends[0], batch_starts[1])
        self.assertLess(batch_starts[1], lw_starts[902])

    def test_banish_does_not_trigger_last_words(self):
        """Banish moves to banished zone without Last Words."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.BANISH, target=TargetKind.ENEMY_UNIT),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        lw_unit = Unit.summon(card(900, attack=1, life=2))
        engine.players[1].board = [lw_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        transition = engine.apply(choice)
        self.assertNotIn(EventType.LAST_WORDS_START, [e.type for e in transition.events])
        self.assertEqual(len(engine.players[1].banished), 1)

    def test_return_to_hand_does_not_trigger_last_words(self):
        """Return to hand does not trigger Last Words."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.RETURN_TO_HAND, target=TargetKind.ENEMY_UNIT),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        lw_unit = Unit.summon(card(900, attack=1, life=2))
        engine.players[1].board = [lw_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        transition = engine.apply(choice)
        self.assertNotIn(EventType.LAST_WORDS_START, [e.type for e in transition.events])

    def test_return_to_deck_does_not_trigger_last_words(self):
        """Return to deck does not trigger Last Words."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.RETURN_TO_DECK, target=TargetKind.ENEMY_UNIT),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        lw_unit = Unit.summon(card(900, attack=1, life=2))
        engine.players[1].board = [lw_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        transition = engine.apply(choice)
        self.assertNotIn(EventType.LAST_WORDS_START, [e.type for e in transition.events])

    def test_two_last_words_execute_in_stable_order(self):
        """Two followers die together: active player's first, left-to-right."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.HEAL_LEADER, target=TargetKind.OWN_LEADER, amount=2),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        a = Unit.summon(card(900, attack=1, life=2))
        b = Unit.summon(card(901, attack=1, life=2))
        engine.players[0].board = [a, b]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        transition = engine.apply(PlayCard(0, 0))
        lw_starts = [e for e in transition.events if e.type == EventType.LAST_WORDS_START]
        self.assertEqual(len(lw_starts), 2)
        self.assertEqual(lw_starts[0].metadata["card_id"], 900)
        self.assertEqual(lw_starts[1].metadata["card_id"], 901)

    def test_last_words_cannot_see_other_same_batch_dead(self):
        """Last Words of one unit cannot target another that died in same batch."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        a = Unit.summon(card(900, attack=1, life=2))
        b = Unit.summon(card(901, attack=1, life=2))
        engine.players[0].board = [a, b]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].board), 0)

    def test_countdown_amulet_last_words_triggers(self):
        """Countdown amulet expiry triggers Last Words via death batch."""
        rulebook = RuleBook.from_directory("data/rules")
        amulet = card(10161210, attack=None, life=None, card_type="护符")
        filler = card(2)
        engine = GameEngine(
            [amulet] * 40, [filler] * 40,
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = amulet
        engine.apply(PlayCard(0, 0))
        deck_before = len(engine.players[0].deck)
        for _ in range(6):
            engine.apply(EndTurn(engine.current_player))
        self.assertEqual(engine.players[0].board, [])
        self.assertTrue(any(g.definition is amulet for g in engine.players[0].graveyard))
        self.assertLessEqual(len(engine.players[0].deck), deck_before - 5)

    def test_destroy_amulet_via_effect_triggers_last_words(self):
        """Destroying an amulet via DESTROY effect triggers Last Words."""
        rulebook = RuleBook((
            CardRule(card_id=700, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DESTROY, target=TargetKind.OWN_AMULET),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        amulet = Amulet(definition=card(700, card_type="护符", attack=None, life=None), entity_id=1000)
        engine.players[0].board = [amulet]
        hand_before = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertTrue(any(g.definition is amulet.definition for g in engine.players[0].graveyard))
        self.assertEqual(len(engine.players[0].hand), hand_before)

    def test_amulet_destroyed_emblem_choice_waits_before_amulet_last_words(self):
        """Destroyed-amulet emblem choices resolve before that amulet's LW."""
        rulebook = RuleBook((
            CardRule(card_id=700, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DESTROY, target=TargetKind.ALL_OWN_AMULETS),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(902, attack=1, life=3), entity_id=902)
        amulet = Amulet(
            definition=card(700, card_type="护符", attack=None, life=None),
            entity_id=700,
        )
        engine.players[0].board = [amulet]
        engine.players[1].board = [target]
        emblem = EmblemDefinition(
            "amulet_death_choice",
            999964,
            triggers=(
                EmblemTriggerRule(
                    "amulet_destroyed",
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.ENEMY_UNIT,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )
        engine._add_emblem_to_player(0, emblem, emblem.source_card_id)
        hand_before = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        first = engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        self.assertEqual(target.health, 3)
        self.assertNotIn(EventType.LAST_WORDS_START, [event.type for event in first.events])
        self.assertEqual(len(engine.players[0].hand), hand_before - 1)
        destroyed_idx = next(
            i for i, event in enumerate(first.events)
            if event.type == EventType.AMULET_DESTROYED
            and event.metadata["card_id"] == 700
        )
        trigger_idx = next(
            i for i, event in enumerate(first.events)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "amulet_death_choice"
        )
        self.assertLess(destroyed_idx, trigger_idx)

        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        transition = engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(target.health, 2)
        self.assertEqual(len(engine.players[0].hand), hand_before)

        history = engine.event_history
        trigger_history = next(
            i for i, event in enumerate(history)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "amulet_death_choice"
        )
        damage_history = next(
            i for i, event in enumerate(history)
            if event.type == EventType.DAMAGE_APPLIED
            and event.target_id == target.entity_id
        )
        lw_history = next(
            i for i, event in enumerate(history)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 700
        )
        batch_end = next(
            i for i, event in enumerate(transition.events)
            if event.type == EventType.DEATH_BATCH_END
        )
        self.assertLess(trigger_history, damage_history)
        self.assertLess(damage_history, lw_history)
        self.assertLess(lw_history, len(history))
        self.assertLess(
            next(i for i, event in enumerate(transition.events) if event.type == EventType.LAST_WORDS_START),
            batch_end,
        )

    def test_amulet_destroyed_emblem_with_no_targets_skips_before_last_words(self):
        """A no-target amulet-destroyed emblem does not block amulet LW."""
        rulebook = RuleBook((
            CardRule(card_id=700, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DESTROY, target=TargetKind.ALL_OWN_AMULETS),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        amulet = Amulet(
            definition=card(700, card_type="护符", attack=None, life=None),
            entity_id=700,
        )
        engine.players[0].board = [amulet]
        emblem = EmblemDefinition(
            "amulet_no_targets",
            999965,
            triggers=(
                EmblemTriggerRule(
                    "amulet_destroyed",
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.ENEMY_UNIT,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )
        engine._add_emblem_to_player(0, emblem, emblem.source_card_id)
        hand_before = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        transition = engine.apply(PlayCard(0, 0))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(len(engine.players[0].hand), hand_before)
        self.assertTrue(
            any(
                event.type == EventType.LAST_WORDS_START
                and event.metadata["card_id"] == 700
                for event in transition.events
            )
        )
        self.assertFalse(
            any(
                event.type == EventType.EMBLEM_TRIGGERED
                and event.metadata["emblem_id"] == "amulet_no_targets"
                for event in engine.event_history
            )
        )

    def test_all_destroy_amulets_unified_batch(self):
        """Destroy all amulets: all removed first, then Last Words."""
        rulebook = RuleBook((
            CardRule(card_id=700, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=701, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DESTROY, target=TargetKind.ALL_OWN_AMULETS),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        a1 = Amulet(definition=card(700, card_type="护符", attack=None, life=None), entity_id=1000)
        a2 = Amulet(definition=card(701, card_type="护符", attack=None, life=None), entity_id=1001)
        engine.players[0].board = [a1, a2]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        batch_sizes = [len(b.records) for b in engine.state.death_queue]
        self.assertEqual(batch_sizes, [2])
        self.assertTrue(any(g.definition.card_id == 700 for g in engine.players[0].graveyard))
        self.assertTrue(any(g.definition.card_id == 701 for g in engine.players[0].graveyard))

    def test_last_words_heal_leader(self):
        """Last Words: heal leader."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.HEAL_LEADER, target=TargetKind.OWN_LEADER, amount=3),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].health = 10
        lw_unit = Unit.summon(card(900, attack=1, life=2))
        engine.players[1].board = [lw_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(engine.players[1].health, 13)


class LoopProtectionTests(unittest.TestCase):
    """Tests for resolution loop protection."""

    def _make_loop_engine(self):
        lw_card = card(900, attack=1, life=1, is_collectible=False)
        def resolver(cid):
            return lw_card if cid == 900 else None
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.SUMMON, target=TargetKind.SELF, card_id=900),
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_OWN_UNITS, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook, card_resolver=resolver,
        )
        engine.reset(seed=1)
        lw_unit = Unit.summon(lw_card, entity_id=engine.state.allocate_entity_id())
        engine.players[1].board = [lw_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        return engine

    def _make_emblem_loop_engine(self):
        lw_card = card(900, attack=1, life=1, is_collectible=False)

        def resolver(cid):
            return lw_card if cid == 900 else None

        payload = {
            "emblems": [
                {
                    "id": "recursive_batch_end",
                    "source_card_id": 999970,
                    "triggers": [
                        {
                            "trigger": "death_batch_end",
                            "event_scope": "any_event",
                            "operations": [
                                {
                                    "kind": "damage_unit",
                                    "target": "all_own_units",
                                    "amount": 1,
                                },
                            ],
                        },
                    ],
                },
            ],
            "rules": [
                {
                    "card_id": 900,
                    "trigger": "last_words",
                    "operations": [
                        {
                            "kind": "summon",
                            "target": "own_leader",
                            "card_id": 900,
                        },
                    ],
                },
                {
                    "card_id": 1,
                    "trigger": "play",
                    "operations": [
                        {
                            "kind": "damage_unit",
                            "target": "enemy_unit",
                            "amount": 5,
                        },
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "loop.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            rulebook = RuleBook.from_directory(tmp)

        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
            card_resolver=resolver,
        )
        engine.reset(seed=1)
        engine._add_emblem_to_player(
            1,
            rulebook.emblem_def("recursive_batch_end"),
            999970,
        )
        lw_unit = Unit.summon(lw_card, entity_id=engine.state.allocate_entity_id())
        engine.players[1].board = [lw_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        return engine

    def _raise_loop(self, *, limit: int = 80) -> ResolutionLoopError:
        from swb.engine import resolution as resolution_module

        original_limit = resolution_module.MAX_RESOLUTION_STEPS
        resolution_module.MAX_RESOLUTION_STEPS = limit
        try:
            engine = self._make_loop_engine()
            engine.apply(PlayCard(0, 0))
            choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
            with self.assertRaises(ResolutionLoopError) as raised:
                engine.apply(choice)
        finally:
            resolution_module.MAX_RESOLUTION_STEPS = original_limit
        return raised.exception

    def _raise_emblem_loop(self, *, limit: int = 100) -> ResolutionLoopError:
        from swb.engine import resolution as resolution_module

        original_limit = resolution_module.MAX_RESOLUTION_STEPS
        resolution_module.MAX_RESOLUTION_STEPS = limit
        try:
            engine = self._make_emblem_loop_engine()
            engine.apply(PlayCard(0, 0))
            choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
            with self.assertRaises(ResolutionLoopError) as raised:
                engine.apply(choice)
        finally:
            resolution_module.MAX_RESOLUTION_STEPS = original_limit
        return raised.exception

    def test_loop_detection_throws_resolution_loop_error(self):
        """A deliberately constructed loop raises ResolutionLoopError."""
        error = self._raise_loop()
        self.assertIn("Resolution step limit exceeded", str(error))

    def test_loop_error_includes_structured_diagnostics(self):
        error = self._raise_loop(limit=60)
        diagnostics = error.diagnostics

        self.assertEqual(diagnostics["limit"], 60)
        self.assertGreater(diagnostics["resolution_steps"], diagnostics["limit"])
        self.assertEqual(diagnostics["turn"], 1)
        self.assertIn("recent_events", diagnostics)
        self.assertIn("event_queue", diagnostics)
        self.assertIn("effect_stack", diagnostics)
        self.assertIn("death_queue", diagnostics)
        self.assertIn("emblem_batches", diagnostics)
        self.assertIn("recent_emblem_triggers", diagnostics)
        self.assertIn("suspended_event", diagnostics)
        self.assertIn("suspended_death_batch", diagnostics)
        self.assertIn("logs_tail", diagnostics)
        self.assertTrue(
            any(
                event["type"] == EventType.FOLLOWER_DESTROYED.value
                for event in diagnostics["recent_events"] + diagnostics["event_queue"]
            )
        )
        json.dumps(diagnostics, ensure_ascii=False)

    def test_loop_diagnostics_are_seed_deterministic(self):
        first = self._raise_loop(limit=60).diagnostics
        second = self._raise_loop(limit=60).diagnostics

        self.assertEqual(first["recent_events"], second["recent_events"])
        self.assertEqual(first["event_queue"], second["event_queue"])
        self.assertEqual(first["effect_stack"], second["effect_stack"])
        self.assertEqual(first["death_queue"], second["death_queue"])
        self.assertEqual(first["emblem_batches"], second["emblem_batches"])
        self.assertEqual(first["recent_emblem_triggers"], second["recent_emblem_triggers"])
        self.assertEqual(first["suspended_event"], second["suspended_event"])
        self.assertEqual(first["suspended_death_batch"], second["suspended_death_batch"])
        self.assertEqual(first["logs_tail"], second["logs_tail"])

    def test_death_batch_end_emblem_loop_diagnostics_identify_trigger_batches(self):
        """JSON death-batch-end emblem loops report their trigger batch source."""
        error = self._raise_emblem_loop(limit=100)
        diagnostics = error.diagnostics

        self.assertIn("Emblem batches", str(error))
        self.assertIn("emblem_batches", diagnostics)
        self.assertIn("recent_emblem_triggers", diagnostics)
        recursive_triggers = [
            event for event in diagnostics["recent_emblem_triggers"]
            if event["metadata"]["emblem_id"] == "recursive_batch_end"
        ]
        self.assertTrue(recursive_triggers)
        self.assertTrue(
            all(event["metadata"]["trigger"] == "death_batch_end" for event in recursive_triggers)
        )
        self.assertTrue(
            any(event["metadata"]["trigger_batch_id"] is not None for event in recursive_triggers)
        )
        self.assertTrue(
            any(batch["record_count"] >= 1 for batch in diagnostics["death_queue"])
        )
        json.dumps(diagnostics, ensure_ascii=False)

    def test_loop_detection_default_limit_uses_configured_constant(self):
        """The production loop guard still reports the configured limit."""
        engine = self._make_loop_engine()
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        with self.assertRaises(ResolutionLoopError) as raised:
            engine.apply(choice)
        self.assertEqual(raised.exception.diagnostics["limit"], MAX_RESOLUTION_STEPS)

    def test_normal_match_does_not_trigger_loop_limit(self):
        """A normal long-playing game stays under MAX_RESOLUTION_STEPS."""
        rulebook = RuleBook.from_directory("data/rules")
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        for _ in range(20):
            if engine.terminated:
                break
            cmd = engine.legal_commands()
            if cmd:
                engine.apply(cmd[0])
        self.assertLess(engine.state.resolution_steps, MAX_RESOLUTION_STEPS)


class DeterminismTests(unittest.TestCase):
    """Tests for deterministic behavior with new death batch system."""

    def test_same_seed_produces_identical_logs(self):
        """Same seed, decks, and action sequence → identical logs."""
        import hashlib

        def get_log_hash():
            rulebook = RuleBook((
                CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                    EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
                ),),
                CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                    EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
                ),),
            ))
            engine = GameEngine(
                [card(i) for i in range(100, 140)],
                [card(i) for i in range(200, 240)],
                class_a=1, class_b=1, seed=7, rulebook=rulebook,
            )
            engine.reset(seed=7)
            a = Unit.summon(card(900, attack=1, life=2))
            b = Unit.summon(card(901, attack=1, life=3))
            engine.players[1].board = [a, b]
            engine.players[0].mana = 10
            engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
            engine.apply(PlayCard(0, 0))
            return hashlib.sha256("\n".join(engine.logs).encode()).hexdigest()

        h1 = get_log_hash()
        h2 = get_log_hash()
        self.assertEqual(h1, h2)


class LastWordsContinuationTests(unittest.TestCase):
    """Regression tests for choice-target Last Words suspension/resumption."""

    def test_single_choice_lw_emits_lw_complete_and_batch_end(self):
        """Single choice-target LW: choose -> damage -> LW_COMPLETE -> BATCH_END."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ANY_UNIT, amount=2),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        lw_unit = Unit.summon(card(900, attack=1, life=2))
        enemy = Unit.summon(card(901, attack=1, life=3))
        engine.players[0].board = [enemy]
        engine.players[1].board = [lw_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        # Play spell, choose lw_unit (player 1's only unit) as target
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        spell_choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(spell_choice)

        # LW triggers, needs choice for its DAMAGE_UNIT ANY_UNIT
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        lw_choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        transition = engine.apply(lw_choice)

        # Verify damage was applied
        self.assertEqual(enemy.health, 1)
        # Verify events are in correct order
        event_types = [e.type for e in transition.events]
        self.assertIn(EventType.LAST_WORDS_COMPLETE, event_types)
        self.assertIn(EventType.DEATH_BATCH_END, event_types)
        lwc_idx = event_types.index(EventType.LAST_WORDS_COMPLETE)
        dbe_idx = event_types.index(EventType.DEATH_BATCH_END)
        self.assertLess(lwc_idx, dbe_idx)
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)

    def test_choice_lw_then_draw_lw_same_batch_ordered(self):
        """Same-batch choice LW + draw LW: must not skip, repeat, or reorder."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ANY_UNIT, amount=2),
            ),),
            CardRule(card_id=901, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        a = Unit.summon(card(900, attack=1, life=2))
        b = Unit.summon(card(901, attack=1, life=2))
        enemy = Unit.summon(card(902, attack=1, life=3))
        engine.players[0].board = [enemy]
        engine.players[1].board = [a, b]
        engine.players[0].mana = 10
        hand_before = len(engine.players[1].hand)
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        # Play all-damage spell -> kills a and b on player 1's board
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)

        # First choice: LW of 900 (ANY_UNIT) - choose enemy
        choice1 = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice1)

        # After first choice: 900's LW fires damage, then 901's LW draws
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(enemy.health, 1)
        self.assertEqual(len(engine.players[1].hand), hand_before + 1)

        # Verify events: two LW_COMPLETE, one DEATH_BATCH_END
        all_events = [e.type for e in engine.event_history]
        lw_completes = [e for e in all_events if e == EventType.LAST_WORDS_COMPLETE]
        self.assertEqual(len(lw_completes), 2)
        batch_ends = [e for e in all_events if e == EventType.DEATH_BATCH_END]
        self.assertEqual(len(batch_ends), 1)

        # Verify no duplicate choices pending
        legal = engine.legal_commands()
        self.assertFalse(any(isinstance(c, Choose) for c in legal))

    def test_follower_destroyed_emblem_choice_waits_before_last_words(self):
        """Destroyed-event emblem choices resolve before same-batch Last Words."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(902, attack=1, life=3), entity_id=902)
        engine.players[0].board = [target]
        engine.players[1].board = [
            Unit.summon(card(900, attack=1, life=2), entity_id=900),
        ]
        emblem = EmblemDefinition(
            "death_choice",
            999960,
            triggers=(
                EmblemTriggerRule(
                    "follower_destroyed",
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.ENEMY_UNIT,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )
        engine._add_emblem_to_player(1, emblem, emblem.source_card_id)
        engine.players[0].mana = 10
        hand_before = len(engine.players[1].hand)
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        first = engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        self.assertEqual(target.health, 3)
        self.assertEqual(len(engine.players[1].hand), hand_before)
        first_types = [event.type for event in first.events]
        self.assertIn(EventType.FOLLOWER_DESTROYED, first_types)
        self.assertNotIn(EventType.LAST_WORDS_START, first_types)

        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        transition = engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(target.health, 2)
        self.assertEqual(len(engine.players[1].hand), hand_before + 1)
        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900]],
        )

        events = transition.events
        damage_idx = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DAMAGE_APPLIED
            and event.target_id == target.entity_id
        )
        lw_start = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 900
        )
        batch_end = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_END
            and event.metadata["batch_id"] == 1
        )
        self.assertLess(damage_idx, lw_start)
        self.assertLess(lw_start, batch_end)

    def test_multiple_follower_destroyed_emblems_finish_before_last_words(self):
        """A pending destroyed-event emblem does not let later emblems trail LW."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(902, attack=1, life=5), entity_id=902)
        engine.players[0].board = [target]
        engine.players[1].board = [
            Unit.summon(card(900, attack=1, life=2), entity_id=900),
        ]
        first_emblem = EmblemDefinition(
            "death_choice",
            999960,
            triggers=(
                EmblemTriggerRule(
                    "follower_destroyed",
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.ENEMY_UNIT,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )
        second_emblem = EmblemDefinition(
            "death_leader",
            999961,
            triggers=(
                EmblemTriggerRule(
                    "follower_destroyed",
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_LEADER,
                            target=TargetKind.ENEMY_LEADER,
                            amount=2,
                        ),
                    ),
                ),
            ),
        )
        engine._add_emblem_to_player(1, first_emblem, first_emblem.source_card_id)
        engine._add_emblem_to_player(1, second_emblem, second_emblem.source_card_id)
        engine.players[0].mana = 10
        hand_before = len(engine.players[1].hand)
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        first = engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        self.assertNotIn(EventType.LAST_WORDS_START, [event.type for event in first.events])
        choice_emblem = next(
            i for i, event in enumerate(first.events)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "death_choice"
        )
        first_destroyed = next(
            i for i, event in enumerate(first.events)
            if event.type == EventType.FOLLOWER_DESTROYED
            and event.metadata["card_id"] == 900
        )
        self.assertLess(first_destroyed, choice_emblem)

        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        transition = engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(target.health, 4)
        self.assertEqual(engine.players[0].health, 18)
        self.assertEqual(len(engine.players[1].hand), hand_before + 1)

        events = transition.events
        choice_damage = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DAMAGE_APPLIED
            and event.target_id == target.entity_id
        )
        leader_emblem = next(
            i for i, event in enumerate(events)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "death_leader"
        )
        leader_damage = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DAMAGE_APPLIED
            and event.metadata.get("target_player") == 0
            and event.amount == 2
        )
        lw_start = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 900
        )
        batch_end = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_END
            and event.metadata["batch_id"] == 1
        )
        history = engine.event_history
        choice_emblem_history = next(
            i for i, event in enumerate(history)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "death_choice"
        )
        lw_start_history = next(
            i for i, event in enumerate(history)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 900
        )
        self.assertLess(choice_damage, lw_start)
        self.assertLess(leader_emblem, leader_damage)
        self.assertLess(leader_damage, lw_start)
        self.assertLess(lw_start, batch_end)
        self.assertLess(choice_emblem_history, lw_start_history)

    def test_cross_player_follower_destroyed_emblems_for_batch_precede_last_words(self):
        """All destroyed-event emblems for a death batch are recorded before LW."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(902, attack=1, life=10), entity_id=902)
        engine.players[0].board = [
            Unit.summon(card(900, attack=1, life=1), entity_id=900),
            target,
        ]
        engine.players[1].board = [
            Unit.summon(card(901, attack=1, life=1), entity_id=901),
        ]
        choice_emblem = EmblemDefinition(
            "cross_death_choice",
            999962,
            triggers=(
                EmblemTriggerRule(
                    "follower_destroyed",
                    event_scope=EventScope.ANY_EVENT,
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.OWN_UNIT,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )
        leader_emblem = EmblemDefinition(
            "cross_death_leader",
            999963,
            triggers=(
                EmblemTriggerRule(
                    "follower_destroyed",
                    event_scope=EventScope.ANY_EVENT,
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_LEADER,
                            target=TargetKind.ENEMY_LEADER,
                            amount=2,
                        ),
                    ),
                ),
            ),
        )
        engine._add_emblem_to_player(0, choice_emblem, choice_emblem.source_card_id)
        engine._add_emblem_to_player(1, leader_emblem, leader_emblem.source_card_id)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        first = engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        self.assertEqual(target.health, 5)
        self.assertNotIn(EventType.LAST_WORDS_START, [event.type for event in first.events])
        first_choice_trigger = next(
            i for i, event in enumerate(first.events)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "cross_death_choice"
            and event.metadata["source_entity_id"] == 900
        )
        first_destroyed = next(
            i for i, event in enumerate(first.events)
            if event.type == EventType.FOLLOWER_DESTROYED
            and event.metadata["card_id"] == 900
        )
        self.assertLess(first_destroyed, first_choice_trigger)

        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        second = engine.apply(choice)

        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        self.assertNotIn(EventType.LAST_WORDS_START, [event.type for event in second.events])

        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        transition = engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(target.health, 3)
        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900, 901]],
        )
        self.assertEqual(engine.players[0].health, 15)

        events = engine.event_history
        first_lw = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] in {900, 901}
        )
        batch_emblems = [
            (i, event)
            for i, event in enumerate(events)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["trigger"] == "follower_destroyed"
            and event.metadata["source_entity_id"] in {900, 901}
        ]
        self.assertEqual(len(batch_emblems), 4)
        self.assertTrue(all(i < first_lw for i, _ in batch_emblems))
        first_choice_trigger = next(
            i for i, event in batch_emblems
            if event.metadata["emblem_id"] == "cross_death_choice"
            and event.metadata["source_entity_id"] == 900
        )
        first_choice_damage = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DAMAGE_APPLIED
            and event.target_id == target.entity_id
            and i > first_choice_trigger
        )
        self.assertLess(first_choice_trigger, first_choice_damage)

    def test_death_batch_end_emblem_fires_after_batch_lw_and_defers_new_death(self):
        """A death-batch-end emblem fires after LW and its kills form a new batch."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=902, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=2),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        delayed_victim = Unit.summon(card(902, attack=1, life=1), entity_id=902)
        engine.players[0].board = [delayed_victim]
        engine.players[1].board = [
            Unit.summon(card(900, attack=1, life=1), entity_id=900),
            Unit.summon(card(901, attack=1, life=1), entity_id=901),
        ]
        emblem = EmblemDefinition(
            "batch_end_cleanup",
            999964,
            triggers=(
                EmblemTriggerRule(
                    "death_batch_end",
                    event_scope=EventScope.ANY_EVENT,
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.ALL_OWN_UNITS,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )
        engine._add_emblem_to_player(0, emblem, emblem.source_card_id)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        transition = engine.apply(PlayCard(0, 0))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900, 901], [902]],
        )

        events = transition.events
        complete_indexes = [
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_COMPLETE
            and event.metadata["card_id"] in {900, 901}
        ]
        batch_1_end = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_END
            and event.metadata["batch_id"] == 1
        )
        trigger = next(
            i for i, event in enumerate(events)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "batch_end_cleanup"
        )
        trigger_event = events[trigger]
        victim_damage = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DAMAGE_APPLIED
            and event.target_id == delayed_victim.entity_id
        )
        batch_2_start = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_START
            and event.metadata["batch_id"] == 2
        )
        start_902 = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 902
        )

        self.assertLess(max(complete_indexes), batch_1_end)
        self.assertLess(batch_1_end, trigger)
        self.assertLess(trigger, victim_damage)
        self.assertLess(victim_damage, batch_2_start)
        self.assertLess(batch_2_start, start_902)
        self.assertEqual(trigger_event.metadata["trigger"], "death_batch_end")
        self.assertEqual(trigger_event.metadata["trigger_batch_id"], 1)
        self.assertEqual(trigger_event.metadata["trigger_batch_record_count"], 2)
        self.assertEqual(trigger_event.metadata["event_player"], 0)

    def test_death_batch_end_emblem_with_no_legal_targets_skips(self):
        """A boundary emblem with no legal targets does not fabricate a trigger."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = [
            Unit.summon(card(900, attack=1, life=1), entity_id=900),
        ]
        emblem = EmblemDefinition(
            "empty_batch_end",
            999965,
            triggers=(
                EmblemTriggerRule(
                    "death_batch_end",
                    event_scope=EventScope.ANY_EVENT,
                    operations=(
                        EffectOperation(
                            kind=EffectKind.DAMAGE_UNIT,
                            target=TargetKind.ALL_ENEMY_UNITS,
                            amount=1,
                        ),
                    ),
                ),
            ),
        )
        engine._add_emblem_to_player(0, emblem, emblem.source_card_id)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        transition = engine.apply(PlayCard(0, 0))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900]],
        )
        self.assertTrue(
            any(
                event.type == EventType.DEATH_BATCH_END
                and event.metadata["batch_id"] == 1
                for event in transition.events
            )
        )
        self.assertFalse(
            any(
                event.type == EventType.EMBLEM_TRIGGERED
                and event.metadata["emblem_id"] == "empty_batch_end"
                for event in transition.events
            )
        )

    def test_choice_lw_kill_waits_for_remaining_lw_before_new_batch(self):
        """A choice LW may kill a unit, but that death waits for same-batch LWs."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ANY_UNIT, amount=2),
            ),),
            CardRule(card_id=901, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=902, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=2),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        victim = Unit.summon(card(902, attack=1, life=2), entity_id=902)
        engine.players[0].board = [victim]
        engine.players[1].board = [
            Unit.summon(card(900, attack=1, life=2)),
            Unit.summon(card(901, attack=1, life=2)),
        ]
        engine.players[0].mana = 10
        hand_before = len(engine.players[1].hand)
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900, 901]],
        )

        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose) and command.option_id == f"entity:{victim.entity_id}"
        )
        transition = engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(len(engine.players[1].hand), hand_before + 1)
        self.assertEqual(engine.players[1].health, 18)
        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900, 901], [902]],
        )

        events = transition.events
        complete_900_event = next(
            event for event in events
            if event.type == EventType.LAST_WORDS_COMPLETE
            and event.metadata["card_id"] == 900
        )
        complete_900 = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_COMPLETE
            and event.metadata["card_id"] == 900
        )
        start_901 = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 901
        )
        batch_1_end = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_END
            and event.metadata["batch_id"] == 1
        )
        batch_2_start = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_START
            and event.metadata["batch_id"] == 2
        )
        start_902 = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 902
        )
        self.assertLess(complete_900, start_901)
        self.assertLess(start_901, batch_1_end)
        self.assertLess(batch_1_end, batch_2_start)
        self.assertLess(batch_2_start, start_902)
        self.assertEqual(complete_900_event.metadata["batch_order_index"], 0)
        self.assertEqual(complete_900_event.metadata["batch_record_count"], 2)
        self.assertEqual(complete_900_event.metadata["active_player"], 0)

    def test_stale_choice_lw_target_skips_and_continues_same_batch(self):
        """If a LW choice target leaves play, skip it and keep resolving batch LWs."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ANY_UNIT, amount=2),
            ),),
            CardRule(card_id=901, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
            CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        stale_target = Unit.summon(card(902, attack=1, life=2), entity_id=902)
        engine.players[0].board = [stale_target]
        engine.players[1].board = [
            Unit.summon(card(900, attack=1, life=2)),
            Unit.summon(card(901, attack=1, life=2)),
        ]
        engine.players[0].mana = 10
        hand_before = len(engine.players[1].hand)
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose) and command.option_id == f"entity:{stale_target.entity_id}"
        )
        engine.players[0].board.remove(stale_target)
        engine._send_to_graveyard(
            0,
            stale_target.definition,
            "test_stale_lw_target",
            source_entity_id=stale_target.entity_id,
        )

        transition = engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(len(engine.players[1].hand), hand_before + 1)
        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900, 901]],
        )
        self.assertTrue(any("已离场，跳过" in log for log in engine.logs))

        events = transition.events
        complete_900 = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_COMPLETE
            and event.metadata["card_id"] == 900
        )
        start_901 = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 901
        )
        batch_1_end = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_END
            and event.metadata["batch_id"] == 1
        )
        self.assertLess(complete_900, start_901)
        self.assertLess(start_901, batch_1_end)


if __name__ == "__main__":
    unittest.main()
