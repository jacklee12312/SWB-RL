from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard, EndTurn
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
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

    def test_loop_detection_throws_resolution_loop_error(self):
        """A deliberately constructed loop raises ResolutionLoopError."""
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
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        with self.assertRaises(ResolutionLoopError):
            engine.apply(choice)

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


if __name__ == "__main__":
    unittest.main()
