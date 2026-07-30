from __future__ import annotations

import unittest

from scripts.report_zone_resource_audit import build_report
from swb.db.repository import CardDefinition
from swb.engine.card_rules import (
    ActivationDefinition,
    CardRule,
    RuleBook,
    Trigger,
)
from swb.engine.commands import ActivateAmulet, Choose, EndTurn, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.emblem import EmblemDefinition
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.faith import FaithDefinition
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, GraveyardCard, HandCard, Phase, Unit


def _card(
    card_id: int,
    *,
    card_type: str = "随从",
    cost: int = 1,
    attack: int = 1,
    life: int = 3,
    name: str | None = None,
    collectible: bool = True,
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=name or f"zone-resource-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack if card_type == "随从" else None,
        life=life if card_type == "随从" else None,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=collectible,
    )


def _deck(start: int) -> list[CardDefinition]:
    return [_card(start + index) for index in range(40)]


def _resolver(definitions: dict[int, CardDefinition]):
    def resolve(card_id: int) -> CardDefinition:
        return definitions[card_id]

    return resolve


def _engine(
    *,
    rulebook: RuleBook | None = None,
    resolver=None,
    deck_a: list[CardDefinition] | None = None,
    seed: int = 18001,
) -> GameEngine:
    engine = GameEngine(
        deck_a or _deck(810000),
        _deck(820000),
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
    engine.state.pending_choice = None
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _put_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int = 0,
) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    player = engine.players[player_index]
    player.hand.append(hand_card)
    player.hand_entity_ids.append(hand_card.entity_id)
    return hand_card


def _put_unit(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int,
) -> Unit:
    unit = Unit.summon(
        definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    unit.summoned_this_turn = False
    engine.players[player_index].board.append(unit)
    return unit


def _put_amulet(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int,
    countdown: int | None = None,
) -> Amulet:
    amulet = Amulet(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        countdown=countdown,
    )
    engine.players[player_index].board.append(amulet)
    return amulet


def _choose_entity(engine: GameEngine, entity_id: int) -> None:
    request = engine.state.pending_choice
    if request is None:
        raise AssertionError("expected pending choice")
    option = next(
        option for option in request.options if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _entity_zones(engine: GameEngine, entity_id: int) -> list[str]:
    zones: list[str] = []
    for player_index, player in enumerate(engine.players):
        if any(card.entity_id == entity_id for card in player.hand):
            zones.append(f"p{player_index}.hand")
        if any(entity.entity_id == entity_id for entity in player.board):
            zones.append(f"p{player_index}.board")
        if any(card.entity_id == entity_id for card in player.graveyard):
            zones.append(f"p{player_index}.graveyard")
        if any(
            material.entity_id == entity_id
            for material in player.fusion_materials
        ):
            zones.append(f"p{player_index}.fusion_materials")
    return zones


class ZoneCapacityBoundaryTests(unittest.TestCase):
    def test_hand_zero_eight_nine_and_overdraw_boundaries(self):
        for initial_size, expected_hand, expected_graveyard, successful in (
            (0, 1, 0, True),
            (8, 9, 0, True),
            (9, 9, 1, False),
        ):
            with self.subTest(initial_size=initial_size):
                engine = _engine(seed=18100 + initial_size)
                player = engine.players[1]
                player.hand.clear()
                player.hand_entity_ids.clear()
                player.graveyard.clear()
                for index in range(initial_size):
                    _put_hand(
                        engine,
                        _card(830000 + index),
                        player_index=1,
                    )
                drawn = _card(831000 + initial_size)
                player.deck[:] = [drawn]

                transition = engine.apply(EndTurn(0))

                self.assertEqual(len(player.hand), expected_hand)
                self.assertEqual(len(player.graveyard), expected_graveyard)
                drawn_events = [
                    event
                    for event in transition.events
                    if event.type is EventType.CARD_DRAWN
                    and event.metadata.get("card_id") == drawn.card_id
                ]
                self.assertEqual(bool(drawn_events), successful)
                if successful:
                    drawn_card = next(
                        card for card in player.hand
                        if card.card_id == drawn.card_id
                    )
                    self.assertEqual(
                        _entity_zones(engine, drawn_card.entity_id),
                        ["p1.hand"],
                    )
                else:
                    overdrawn = player.graveyard[-1]
                    self.assertEqual(overdrawn.definition.card_id, drawn.card_id)
                    self.assertEqual(overdrawn.entry_cause, "overdraw")
                    self.assertEqual(
                        _entity_zones(engine, overdrawn.entity_id),
                        ["p1.graveyard"],
                    )
                engine.assert_invariants()

    def test_board_zero_four_five_and_death_reopens_slot(self):
        source = _card(840001, card_type="法术")
        token = _card(840002, collectible=False)
        resolver = _resolver({token.card_id: token})
        summon_rulebook = RuleBook(
            rules=(
                CardRule(
                    source.card_id,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.SUMMON,
                            TargetKind.SELF,
                            card_id=token.card_id,
                        ),
                    ),
                ),
            )
        )
        for initial_size, expected_size in ((0, 1), (4, 5), (5, 5)):
            with self.subTest(initial_size=initial_size):
                engine = _engine(
                    rulebook=summon_rulebook,
                    resolver=resolver,
                    seed=18200 + initial_size,
                )
                player = engine.players[0]
                player.hand.clear()
                player.hand_entity_ids.clear()
                player.board.clear()
                for index in range(initial_size):
                    _put_unit(
                        engine,
                        _card(841000 + index),
                        player_index=0,
                    )
                _put_hand(engine, source)

                engine.apply(PlayCard(0, 0))

                self.assertEqual(len(player.board), expected_size)
                self.assertEqual(
                    sum(entity.definition.card_id == token.card_id for entity in player.board),
                    int(initial_size < engine.config.max_board),
                )
                engine.assert_invariants()

        replace_source = _card(840003, card_type="法术")
        replace_rulebook = RuleBook(
            rules=(
                CardRule(
                    replace_source.card_id,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.DESTROY,
                            TargetKind.OWN_UNIT,
                        ),
                        EffectOperation(
                            EffectKind.SUMMON,
                            TargetKind.SELF,
                            card_id=token.card_id,
                        ),
                    ),
                ),
            )
        )
        engine = _engine(
            rulebook=replace_rulebook,
            resolver=resolver,
            seed=18299,
        )
        player = engine.players[0]
        player.hand.clear()
        player.hand_entity_ids.clear()
        player.board.clear()
        targets = [
            _put_unit(
                engine,
                _card(842000 + index),
                player_index=0,
            )
            for index in range(5)
        ]
        _put_hand(engine, replace_source)

        engine.apply(PlayCard(0, 0))
        _choose_entity(engine, targets[2].entity_id)

        self.assertEqual(len(player.board), 5)
        self.assertEqual(
            _entity_zones(engine, targets[2].entity_id),
            ["p0.graveyard"],
        )
        summoned = next(
            entity
            for entity in player.board
            if entity.definition.card_id == token.card_id
        )
        self.assertEqual(_entity_zones(engine, summoned.entity_id), ["p0.board"])
        engine.assert_invariants()


class ZoneOwnershipTests(unittest.TestCase):
    def _target_engine(
        self,
        kind: EffectKind,
        target: TargetKind,
        *,
        result_card: CardDefinition | None = None,
    ) -> tuple[GameEngine, CardDefinition]:
        source = _card(850000 + list(EffectKind).index(kind), card_type="法术")
        operation = EffectOperation(
            kind,
            target,
            card_id=None if result_card is None else result_card.card_id,
        )
        resolver = (
            None
            if result_card is None
            else _resolver({result_card.card_id: result_card})
        )
        engine = _engine(
            rulebook=RuleBook(
                rules=(CardRule(source.card_id, Trigger.PLAY, (operation,)),)
            ),
            resolver=resolver,
            seed=18300 + list(EffectKind).index(kind),
        )
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        _put_hand(engine, source)
        return engine, source

    def test_zone_transitions_keep_single_entity_ownership(self):
        add_result = _card(851001, card_type="法术", collectible=False)
        add_engine, _ = self._target_engine(
            EffectKind.ADD_CARD,
            TargetKind.OWN_LEADER,
            result_card=add_result,
        )
        add_engine.apply(PlayCard(0, 0))
        added = next(
            card
            for card in add_engine.players[0].hand
            if card.card_id == add_result.card_id
        )
        self.assertEqual(_entity_zones(add_engine, added.entity_id), ["p0.hand"])
        add_engine.assert_invariants()

        discard_engine, _ = self._target_engine(
            EffectKind.DISCARD,
            TargetKind.OWN_HAND,
        )
        discarded = _put_hand(
            discard_engine,
            _card(851002, card_type="法术", name="discard-me"),
        )
        discard_engine.apply(PlayCard(0, 0))
        _choose_entity(discard_engine, discarded.entity_id)
        self.assertEqual(
            _entity_zones(discard_engine, discarded.entity_id),
            ["p0.graveyard"],
        )
        discard_engine.assert_invariants()

        return_engine, _ = self._target_engine(
            EffectKind.RETURN_TO_HAND,
            TargetKind.ENEMY_UNIT,
        )
        returned = _put_unit(
            return_engine,
            _card(851003),
            player_index=1,
        )
        return_engine.apply(PlayCard(0, 0))
        _choose_entity(return_engine, returned.entity_id)
        self.assertEqual(_entity_zones(return_engine, returned.entity_id), [])
        returned_hand_card = next(
            card
            for card in return_engine.players[1].hand
            if card.card_id == returned.definition.card_id
        )
        self.assertEqual(
            _entity_zones(return_engine, returned_hand_card.entity_id),
            ["p1.hand"],
        )
        return_engine.assert_invariants()

        banish_engine, _ = self._target_engine(
            EffectKind.BANISH,
            TargetKind.ENEMY_UNIT,
        )
        banished = _put_unit(
            banish_engine,
            _card(851004),
            player_index=1,
        )
        banish_engine.apply(PlayCard(0, 0))
        _choose_entity(banish_engine, banished.entity_id)
        self.assertEqual(_entity_zones(banish_engine, banished.entity_id), [])
        self.assertEqual(
            [card.card_id for card in banish_engine.players[1].banished],
            [banished.definition.card_id],
        )
        self.assertFalse(banish_engine.players[1].graveyard)
        banish_engine.assert_invariants()

        transformed_definition = _card(851006, collectible=False)
        transform_engine, _ = self._target_engine(
            EffectKind.TRANSFORM,
            TargetKind.ENEMY_UNIT,
            result_card=transformed_definition,
        )
        transformed = _put_unit(
            transform_engine,
            _card(851005),
            player_index=1,
        )
        transform_engine.apply(PlayCard(0, 0))
        _choose_entity(transform_engine, transformed.entity_id)
        self.assertEqual(
            _entity_zones(transform_engine, transformed.entity_id),
            ["p1.board"],
        )
        self.assertEqual(
            transform_engine.players[1].board[0].definition.card_id,
            transformed_definition.card_id,
        )
        self.assertFalse(transform_engine.players[1].graveyard)
        transform_engine.assert_invariants()

        deck_engine, _ = self._target_engine(
            EffectKind.RETURN_TO_DECK,
            TargetKind.ENEMY_UNIT,
        )
        deck_target = _put_unit(
            deck_engine,
            _card(851007),
            player_index=1,
        )
        deck_size = len(deck_engine.players[1].deck)
        deck_engine.apply(PlayCard(0, 0))
        _choose_entity(deck_engine, deck_target.entity_id)
        self.assertEqual(_entity_zones(deck_engine, deck_target.entity_id), [])
        self.assertEqual(len(deck_engine.players[1].deck), deck_size + 1)
        self.assertTrue(
            any(
                card.card_id == deck_target.definition.card_id
                for card in deck_engine.players[1].deck
            )
        )
        self.assertFalse(deck_engine.players[1].graveyard)
        deck_engine.assert_invariants()


class AmuletAndLeaderAreaTests(unittest.TestCase):
    def test_amulet_countdown_destroy_banish_and_activate_are_distinct(self):
        countdown_card = _card(860001, card_type="护符")
        countdown_engine = _engine(
            rulebook=RuleBook(
                rules=(
                    CardRule(
                        countdown_card.card_id,
                        Trigger.COUNTDOWN_EXPIRED,
                        (
                            EffectOperation(
                                EffectKind.HEAL_LEADER,
                                TargetKind.OWN_LEADER,
                                amount=2,
                            ),
                        ),
                        countdown=1,
                    ),
                )
            ),
            seed=18401,
        )
        countdown_engine.players[0].hand.clear()
        countdown_engine.players[0].hand_entity_ids.clear()
        countdown_engine.players[0].health = 10
        _put_hand(countdown_engine, countdown_card)
        countdown_engine.apply(PlayCard(0, 0))
        countdown_engine.apply(EndTurn(0))
        countdown_engine.apply(EndTurn(1))
        self.assertFalse(countdown_engine.players[0].board)
        self.assertEqual(countdown_engine.players[0].health, 12)
        self.assertEqual(
            countdown_engine.players[0].graveyard[-1].entry_cause,
            "countdown_expired",
        )

        last_words_card = _card(860002, card_type="护符")
        destroy_spell = _card(860003, card_type="法术")
        banish_spell = _card(860004, card_type="法术")
        shared_rules = (
            CardRule(
                last_words_card.card_id,
                Trigger.LAST_WORDS,
                (
                    EffectOperation(
                        EffectKind.HEAL_LEADER,
                        TargetKind.OWN_LEADER,
                        amount=2,
                    ),
                ),
            ),
            CardRule(
                destroy_spell.card_id,
                Trigger.PLAY,
                (
                    EffectOperation(
                        EffectKind.DESTROY,
                        TargetKind.ENEMY_AMULET,
                    ),
                ),
            ),
            CardRule(
                banish_spell.card_id,
                Trigger.PLAY,
                (
                    EffectOperation(
                        EffectKind.BANISH,
                        TargetKind.ENEMY_AMULET,
                    ),
                ),
            ),
        )
        for source, expected_health, expected_zone in (
            (destroy_spell, 12, "graveyard"),
            (banish_spell, 10, "banished"),
        ):
            with self.subTest(exit=expected_zone):
                engine = _engine(
                    rulebook=RuleBook(rules=shared_rules),
                    seed=18410 + source.card_id,
                )
                engine.players[0].hand.clear()
                engine.players[0].hand_entity_ids.clear()
                engine.players[1].board.clear()
                engine.players[1].graveyard.clear()
                engine.players[1].banished.clear()
                engine.players[1].health = 10
                target = _put_amulet(
                    engine,
                    last_words_card,
                    player_index=1,
                )
                _put_hand(engine, source)
                engine.apply(PlayCard(0, 0))
                _choose_entity(engine, target.entity_id)
                self.assertEqual(engine.players[1].health, expected_health)
                self.assertFalse(engine.players[1].board)
                self.assertEqual(
                    bool(engine.players[1].graveyard),
                    expected_zone == "graveyard",
                )
                self.assertEqual(
                    bool(engine.players[1].banished),
                    expected_zone == "banished",
                )
                engine.assert_invariants()

        activation_card = _card(860005, card_type="护符")
        activation_engine = _engine(
            rulebook=RuleBook(
                rules=(
                    CardRule(
                        activation_card.card_id,
                        Trigger.ACTIVATE,
                        (
                            EffectOperation(
                                EffectKind.HEAL_LEADER,
                                TargetKind.OWN_LEADER,
                                amount=1,
                            ),
                        ),
                    ),
                ),
                activation_defs={
                    activation_card.card_id: ActivationDefinition(
                        activation_card.card_id,
                        cost=0,
                    )
                },
            ),
            seed=18499,
        )
        activation_engine.players[0].hand.clear()
        activation_engine.players[0].hand_entity_ids.clear()
        activation_engine.players[0].health = 10
        _put_hand(activation_engine, activation_card)
        activation_engine.apply(PlayCard(0, 0))
        activated = activation_engine.players[0].board[0]
        activation_engine.apply(ActivateAmulet(0, activated.entity_id))
        self.assertIn(activated, activation_engine.players[0].board)
        self.assertEqual(activation_engine.players[0].health, 11)
        self.assertEqual(activated.activated_turn, activation_engine.turn)
        self.assertTrue(
            any(
                event.type is EventType.AMULET_ACTIVATED
                for event in activation_engine.event_history
            )
        )
        activation_engine.assert_invariants()

    def test_leader_area_shares_five_slots_and_rejects_sixth(self):
        faith_cards = [
            _card(870000 + index, name=f"faith-{index}")
            for index in range(4)
        ]
        faith_defs = {
            card.card_id: FaithDefinition(
                faith_id=f"faith-{index}",
                source_card_id=card.card_id,
            )
            for index, card in enumerate(faith_cards)
        }
        first_spell = _card(870100, card_type="法术")
        second_spell = _card(870101, card_type="法术")
        first_emblem = EmblemDefinition("first-emblem", first_spell.card_id)
        second_emblem = EmblemDefinition("second-emblem", second_spell.card_id)
        rulebook = RuleBook(
            rules=(
                CardRule(
                    first_spell.card_id,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.GAIN_EMBLEM,
                            TargetKind.OWN_LEADER,
                            emblem_id=first_emblem.emblem_id,
                        ),
                    ),
                ),
                CardRule(
                    second_spell.card_id,
                    Trigger.PLAY,
                    (
                        EffectOperation(
                            EffectKind.GAIN_EMBLEM,
                            TargetKind.OWN_LEADER,
                            emblem_id=second_emblem.emblem_id,
                        ),
                    ),
                ),
            ),
            faith_defs=faith_defs,
            emblem_defs={
                first_emblem.emblem_id: first_emblem,
                second_emblem.emblem_id: second_emblem,
            },
        )
        deck = [*faith_cards, *_deck(871000)[:36]]
        engine = _engine(rulebook=rulebook, deck_a=deck, seed=18501)
        player = engine.players[0]
        self.assertEqual(len(player.faiths), 4)
        player.hand.clear()
        player.hand_entity_ids.clear()

        _put_hand(engine, first_spell)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(player.faiths) + len(player.emblems), 5)
        self.assertEqual(
            [emblem.emblem_id for emblem in player.emblems],
            [first_emblem.emblem_id],
        )

        _put_hand(engine, first_spell)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(player.emblems), 1)

        _put_hand(engine, second_spell)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(player.faiths) + len(player.emblems), 5)
        self.assertEqual(
            [emblem.emblem_id for emblem in player.emblems],
            [first_emblem.emblem_id],
        )
        self.assertEqual(
            sum(
                event.type is EventType.EMBLEM_GAINED
                for event in engine.event_history
            ),
            1,
        )
        engine.assert_invariants()


class ObservationZoneTests(unittest.TestCase):
    def test_public_zone_histograms_match_real_zones(self):
        deck_a = _deck(880000)
        deck_b = _deck(881000)
        grave_a = _card(882001)
        grave_b = _card(882002)
        banish_a = _card(882003)
        banish_b = _card(882004)
        vocabulary = tuple(
            card.card_id
            for card in (*deck_a, *deck_b, grave_a, grave_b, banish_a, banish_b)
        )
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=1,
            class_b=1,
            seed=18601,
            observation_version="v2",
            card_vocabulary=vocabulary,
        )
        env.reset(seed=18601)
        for player in env.players:
            player.graveyard.clear()
            player.banished.clear()
        env.players[0].graveyard.extend(
            (
                GraveyardCard(
                    grave_a,
                    env.core.state.allocate_entity_id(),
                    0,
                    1,
                    "audit",
                    origin=CardOrigin.DECK,
                ),
                GraveyardCard(
                    grave_a,
                    env.core.state.allocate_entity_id(),
                    0,
                    2,
                    "audit",
                    origin=CardOrigin.DECK,
                ),
                GraveyardCard(
                    grave_b,
                    env.core.state.allocate_entity_id(),
                    0,
                    3,
                    "audit",
                    origin=CardOrigin.DECK,
                ),
            )
        )
        env.players[1].graveyard.append(
            GraveyardCard(
                grave_b,
                env.core.state.allocate_entity_id(),
                1,
                1,
                "audit",
                origin=CardOrigin.DECK,
            )
        )
        env.players[0].banished.extend((banish_a, banish_a, banish_b))
        env.players[1].banished.extend((banish_b, banish_b))
        env.invalidate_cache(reason="zone resource audit setup")

        observation_0 = env.observation(perspective=0)
        observation_1 = env.observation(perspective=1)
        index = {card_id: offset for offset, card_id in enumerate(vocabulary)}

        graveyards_0 = observation_0["card_indices"]["public_graveyards"]
        banished_0 = observation_0["card_indices"]["public_banished"]
        self.assertEqual(sum(graveyards_0[0]), len(env.players[0].graveyard))
        self.assertEqual(sum(graveyards_0[1]), len(env.players[1].graveyard))
        self.assertEqual(sum(banished_0[0]), len(env.players[0].banished))
        self.assertEqual(sum(banished_0[1]), len(env.players[1].banished))
        self.assertEqual(graveyards_0[0][index[grave_a.card_id]], 2)
        self.assertEqual(graveyards_0[0][index[grave_b.card_id]], 1)
        self.assertEqual(graveyards_0[1][index[grave_b.card_id]], 1)
        self.assertEqual(banished_0[0][index[banish_a.card_id]], 2)
        self.assertEqual(banished_0[0][index[banish_b.card_id]], 1)
        self.assertEqual(banished_0[1][index[banish_b.card_id]], 2)
        self.assertEqual(
            observation_1["card_indices"]["public_graveyards"],
            (graveyards_0[1], graveyards_0[0]),
        )
        self.assertEqual(
            observation_1["card_indices"]["public_banished"],
            (banished_0[1], banished_0[0]),
        )
        env.core.assert_invariants()


class ZoneResourceReportTests(unittest.TestCase):
    def test_full_pool_zone_resource_report_has_zero_failures(self):
        report = build_report()
        self.assertEqual(report["scope"]["card_count"], 826)
        self.assertEqual(report["scope"]["collectible_card_count"], 735)
        self.assertEqual(report["scope"]["generated_card_count"], 91)
        self.assertEqual(report["scope"]["training_closure_card_count"], 147)
        self.assertTrue(report["summary"]["passed"], report["summary"]["failures"])
        self.assertEqual(report["summary"]["failure_count"], 0)
        self.assertEqual(
            {row["category"] for row in report["category_matrix"]},
            set(
                (
                    "draw",
                    "add_to_hand",
                    "discard",
                    "return_to_hand",
                    "banish",
                    "transform",
                    "return_to_deck",
                    "summon",
                    "destroy",
                    "countdown_or_activate",
                    "leader_area",
                    "empty_deck",
                    "combo",
                    "cooperation",
                    "shadows_necromancy",
                    "overflow",
                    "earth_sigils",
                    "spellboost",
                    "fusion",
                    "union_burst",
                    "super_skybound_art",
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
