from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard, EndTurn
from swb.engine.effects import BoardFilter, EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
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


def make_resolver(defs: dict[int, CardDefinition]):
    def resolve(cid: int) -> CardDefinition | None:
        return defs.get(cid)
    return resolve


def spell_rule(card_id: int, kind: EffectKind, target: TargetKind, **kwargs) -> CardRule:
    return CardRule(
        card_id=card_id,
        trigger=Trigger.PLAY,
        operations=(EffectOperation(kind=kind, target=target, **kwargs),),
    )


class TargetingTests(unittest.TestCase):
    """Tests for the unified targeting system."""

    def test_banish_enemy_unit_moves_to_banished_zone(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=2, life=3))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)

        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(len(engine.players[1].graveyard), 0)
        self.assertEqual(len(engine.players[1].banished), 1)
        self.assertEqual(engine.players[1].banished[0].card_id, 900)

    def test_banish_does_not_trigger_last_words(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=2))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        transition = engine.apply(choice)

        destroyed_events = [e for e in transition.events if e.type == EventType.FOLLOWER_DESTROYED]
        banished_events = [e for e in transition.events if e.type == EventType.CARD_BANISHED]
        self.assertEqual(len(destroyed_events), 0)
        self.assertEqual(len(banished_events), 1)

    def test_random_target_reproducible_with_fixed_seed(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.DAMAGE_UNIT, TargetKind.RANDOM_ENEMY_UNIT, amount=5),
        ))
        results = []
        for trial in range(3):
            engine = GameEngine(
                [card(i) for i in range(100, 140)],
                [card(i) for i in range(200, 240)],
                class_a=1, class_b=1, seed=42, rulebook=rulebook,
            )
            engine.reset(seed=42)
            for i in range(3):
                unit = Unit.summon(card(900 + i, attack=1, life=10))
                engine.players[1].board.append(unit)
            engine.players[0].mana = 10
            engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
            engine.apply(PlayCard(0, 0))
            results.append([u.health for u in engine.players[1].board])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])

    def test_all_enemy_units_hits_all_targets(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.DAMAGE_UNIT, TargetKind.ALL_ENEMY_UNITS, amount=3),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        for i in range(3):
            unit = Unit.summon(card(900 + i, attack=1, life=5))
            engine.players[1].board.append(unit)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))

        for unit in engine.players[1].board:
            self.assertEqual(unit.health, 2)

    def test_no_legal_target_skips_operation(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(kind=EffectKind.BANISH, target=TargetKind.ENEMY_UNIT),
                    EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        hand_before = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertEqual(len(engine.players[0].hand), hand_before)

    def test_random_no_target_safe_skip(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.RANDOM_ENEMY_UNIT, amount=2),
                    EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertIsNone(engine.state.pending_choice)

    def test_all_no_target_safe_skip(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=2),
                    EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertIsNone(engine.state.pending_choice)

    def test_card_unplayable_when_all_choice_ops_have_no_targets(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[1].board = []
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        commands = engine.legal_commands()
        play_cmds = [c for c in commands if isinstance(c, PlayCard) and c.hand_index == 0]
        self.assertEqual(len(play_cmds), 0)

    def test_target_leaves_play_during_choice_does_not_crash(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.ANY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=2))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)

        engine.players[1].board = []
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(engine.state.phase, Phase.MAIN)
        self.assertIsNone(engine.state.pending_choice)

    def test_manual_board_filter_limits_choice_candidates(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DESTROY,
                        target=TargetKind.OWN_UNIT,
                        board_filter=BoardFilter(card_id=900),
                    ),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        valid = Unit.summon(card(900, name="valid"))
        invalid = Unit.summon(card(901, name="invalid"))
        engine.players[0].board = [valid, invalid]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))

        choices = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertEqual([c.option_id for c in choices], [f"entity:{valid.entity_id}"])

    def test_manual_board_filter_can_require_evolved_unit(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DAMAGE_UNIT,
                        target=TargetKind.ENEMY_UNIT,
                        amount=2,
                        board_filter=BoardFilter(evolved=True),
                    ),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        evolved = Unit.summon(card(900, life=5), entity_id=900)
        evolved.evolved = True
        unevolved = Unit.summon(card(901, life=5), entity_id=901)
        engine.players[1].board = [unevolved, evolved]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))

        choices = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertEqual([c.option_id for c in choices], [f"entity:{evolved.entity_id}"])

    def test_random_and_all_board_filters_share_candidate_logic(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DAMAGE_UNIT,
                        target=TargetKind.RANDOM_ENEMY_UNIT,
                        amount=2,
                        board_filter=BoardFilter(cost_min=3, cost_max=3),
                    ),
                    EffectOperation(
                        kind=EffectKind.DAMAGE_UNIT,
                        target=TargetKind.ALL_ENEMY_UNITS,
                        amount=1,
                        board_filter=BoardFilter(card_name="target"),
                    ),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        low = Unit.summon(card(900, cost=1, life=10, name="skip"))
        target = Unit.summon(card(901, cost=3, life=10, name="target"))
        high = Unit.summon(card(902, cost=5, life=10, name="skip"))
        engine.players[1].board = [low, target, high]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(low.health, 10)
        self.assertEqual(target.health, 7)
        self.assertEqual(high.health, 10)


class ZoneChangeTests(unittest.TestCase):
    """Tests for zone change effects."""

    def test_summon_follower_to_board(self):
        summoned_card = card(700, attack=3, life=4, is_collectible=False)
        resolver = make_resolver({700: summoned_card})
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.SUMMON, target=TargetKind.SELF, card_id=700,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
            card_resolver=resolver,
        )
        engine.reset(seed=1)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].board), 1)
        unit = engine.players[0].board[0]
        self.assertEqual(unit.attack, 3)
        self.assertEqual(unit.health, 4)

    def test_summon_fails_when_board_full(self):
        summoned_card = card(700, attack=1, life=1, is_collectible=False)
        resolver = make_resolver({700: summoned_card})
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.SUMMON, target=TargetKind.SELF, card_id=700,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
            card_resolver=resolver,
        )
        engine.reset(seed=1)
        for i in range(5):
            engine.players[0].board.append(Unit.summon(card(800 + i)))
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].board), 5)

    def test_add_card_to_hand(self):
        added_card = card(700, attack=None, life=None, card_type="法术", is_collectible=False)
        resolver = make_resolver({700: added_card})
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.ADD_CARD, target=TargetKind.OWN_LEADER, card_id=700,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
            card_resolver=resolver,
        )
        engine.reset(seed=1)
        hand_size_before = len(engine.players[0].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].hand), hand_size_before)
        self.assertEqual(engine.players[0].hand[-1].card_id, 700)

    def test_add_card_full_hand_discards_to_graveyard(self):
        added_card = card(700, attack=None, life=None, card_type="法术", is_collectible=False)
        resolver = make_resolver({700: added_card})
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.ADD_CARD, target=TargetKind.OWN_LEADER, card_id=700,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
            card_resolver=resolver,
        )
        engine.reset(seed=1)
        engine.players[0].hand = [card(800 + i) for i in range(engine.config.max_hand + 1)]
        engine.players[0].hand_entity_ids = [engine.state.allocate_entity_id() for _ in range(engine.config.max_hand + 1)]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].hand), engine.config.max_hand)
        self.assertTrue(any(g.definition.card_id == 700 for g in engine.players[0].graveyard))

    def test_return_unit_to_hand(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.RETURN_TO_HAND, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=2, life=3))
        engine.players[1].board = [target]
        hand_before = len(engine.players[1].hand)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(len(engine.players[1].hand), hand_before + 1)
        self.assertEqual(engine.players[1].hand[-1].card_id, 900)

    def test_return_to_hand_full_hand_banishes(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.RETURN_TO_HAND, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        for i in range(engine.config.max_hand):
            engine.players[1].hand.append(card(800 + i))
            engine.players[1].hand_entity_ids.append(engine.state.allocate_entity_id())
        target = Unit.summon(card(900, attack=2, life=3))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(len(engine.players[1].banished), 1)
        self.assertEqual(engine.players[1].banished[0].card_id, 900)

    def test_return_to_deck(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.RETURN_TO_DECK, TargetKind.ENEMY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target_card = card(900, attack=2, life=3)
        target = Unit.summon(target_card)
        engine.players[1].board = [target]
        deck_before = len(engine.players[1].deck)
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(len(engine.players[1].deck), deck_before + 1)
        self.assertTrue(any(c.card_id == 900 for c in engine.players[1].deck))

    def test_discard_moves_hand_to_graveyard(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.DISCARD, target=TargetKind.OWN_HAND,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        known = card(999, card_type="法术", attack=None, life=None, name="discard-target")
        engine.players[0].hand.append(known)
        engine.players[0].hand_entity_ids.append(engine.state.allocate_entity_id())
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        hand_before = len(engine.players[0].hand)

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        options = engine.state.pending_choice.options
        known_option = next(o for o in options if "discard-target" in o.label)
        choice = Choose(engine.current_player, known_option.option_id)
        engine.apply(choice)
        self.assertEqual(len(engine.players[0].hand), hand_before - 2)
        self.assertTrue(any(g.definition.card_id == 999 for g in engine.players[0].graveyard))

    def test_old_rulebook_remains_compatible(self):
        rulebook = RuleBook.from_directory("data/rules")
        operations = rulebook.operations_for(10041130, Trigger.FANFARE)
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0].amount, 6)

        shark = card(10041130, attack=4, life=3)
        engine = GameEngine(
            [shark] * 40, [card(2)] * 40,
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[1].health, 14)

    def test_spell_with_both_choice_and_implicit_targets(self):
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=3),
                    EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
                ),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        target = Unit.summon(card(900, attack=1, life=5))
        engine.players[1].board = [target]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        choice = [c for c in engine.legal_commands() if isinstance(c, Choose)][0]
        engine.apply(choice)
        self.assertEqual(target.health, 2)

    def test_any_unit_target_includes_both_sides(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.ANY_UNIT),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        own_unit = Unit.summon(card(700, attack=1, life=1))
        enemy_unit = Unit.summon(card(800, attack=1, life=1))
        engine.players[0].board = [own_unit]
        engine.players[1].board = [enemy_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        options = engine.state.pending_choice.options
        option_labels = [o.label for o in options]
        self.assertIn("card-700", option_labels)
        self.assertIn("card-800", option_labels)

    def test_own_amulet_target_correctly_filters(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.OWN_AMULET),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        unit = Unit.summon(card(700, attack=1, life=1))
        amulet = Amulet(definition=card(701, card_type="护符", attack=None, life=None), entity_id=1000)
        engine.players[0].board = [unit, amulet]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        options = engine.state.pending_choice.options
        option_ids = [o.entity_id for o in options]
        self.assertNotIn(unit.entity_id, option_ids)
        self.assertIn(amulet.entity_id, option_ids)

    def test_all_units_hits_both_sides(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.DAMAGE_UNIT, TargetKind.ALL_UNITS, amount=1),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        own_unit = Unit.summon(card(700, attack=1, life=3))
        enemy_unit = Unit.summon(card(800, attack=1, life=3))
        engine.players[0].board = [own_unit]
        engine.players[1].board = [enemy_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(own_unit.health, 2)
        self.assertEqual(enemy_unit.health, 2)

    def test_random_own_board_picks_from_own_side(self):
        rulebook = RuleBook((
            spell_rule(1, EffectKind.BANISH, TargetKind.RANDOM_OWN_BOARD),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        own_unit = Unit.summon(card(700, attack=1, life=1))
        engine.players[0].board = [own_unit]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.players[0].board, [])
        self.assertEqual(len(engine.players[0].banished), 1)

    def test_hand_entity_ids_are_consistent(self):
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1,
        )
        engine.reset(seed=1)
        player = engine.players[0]
        self.assertEqual(len(player.hand), len(player.hand_entity_ids))
        self.assertTrue(all(eid > 0 for eid in player.hand_entity_ids))
        entity_set = set(player.hand_entity_ids)
        self.assertEqual(len(entity_set), len(player.hand_entity_ids))

    def test_damage_unit_with_any_board_filters_out_amulets(self):
        """Regression: DAMAGE_UNIT + ANY_BOARD must not allow selecting amulets."""
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ANY_BOARD, amount=3,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        unit = Unit.summon(card(700, attack=1, life=3))
        amulet = Amulet(definition=card(701, card_type="护符", attack=None, life=None), entity_id=1000)
        engine.players[0].board = [unit, amulet]
        engine.players[0].mana = 10
        engine.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase, Phase.AWAITING_CHOICE)
        options = engine.state.pending_choice.options
        option_eids = [o.entity_id for o in options]
        self.assertIn(unit.entity_id, option_eids)
        self.assertNotIn(amulet.entity_id, option_eids)

    def test_amulet_entity_ids_are_validated_and_unique(self):
        """Regression: _ensure_entity_ids must handle Amulet, not just Unit."""
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1,
        )
        engine.reset(seed=1)
        a1 = Amulet(definition=card(701, card_type="护符", attack=None, life=None))
        a2 = Amulet(definition=card(702, card_type="护符", attack=None, life=None))
        engine.players[0].board = [a1, a2]
        rules = list(engine.legal_commands())
        self.assertGreater(a1.entity_id, 0)
        self.assertGreater(a2.entity_id, 0)
        self.assertNotEqual(a1.entity_id, a2.entity_id)

    def test_discard_spell_unplayable_when_only_self_in_hand(self):
        """Regression: a discard spell must not count itself as a valid target."""
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.DISCARD, target=TargetKind.OWN_HAND,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].hand = [card(1, card_type="法术", attack=None, life=None)]
        engine.players[0].hand_entity_ids = [engine.state.allocate_entity_id()]
        engine.players[0].mana = 10
        commands = engine.legal_commands()
        play_cmds = [c for c in commands if isinstance(c, PlayCard)]
        self.assertEqual(len(play_cmds), 0)

    def test_discard_spell_playable_when_other_card_in_hand(self):
        """Discard spell is playable when at least one other card exists in hand."""
        rulebook = RuleBook((
            CardRule(
                card_id=1, trigger=Trigger.PLAY,
                operations=(EffectOperation(
                    kind=EffectKind.DISCARD, target=TargetKind.OWN_HAND,
                ),),
            ),
        ))
        engine = GameEngine(
            [card(i) for i in range(100, 140)],
            [card(i) for i in range(200, 240)],
            class_a=1, class_b=1, seed=1, rulebook=rulebook,
        )
        engine.reset(seed=1)
        engine.players[0].hand = [
            card(1, card_type="法术", attack=None, life=None),
            card(800, card_type="法术", attack=None, life=None),
        ]
        engine.players[0].hand_entity_ids = [
            engine.state.allocate_entity_id(),
            engine.state.allocate_entity_id(),
        ]
        engine.players[0].mana = 10
        commands = engine.legal_commands()
        play_cmds = [c for c in commands if isinstance(c, PlayCard)]
        self.assertEqual(len(play_cmds), 1)


if __name__ == "__main__":
    unittest.main()
