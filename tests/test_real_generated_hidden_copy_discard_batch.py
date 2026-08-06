# -*- coding: utf-8 -*-
"""Exact hidden-copy, discard-trigger, and board-card transform token chains."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import (
    CostChangeMode,
    EffectKind,
    EffectOperation,
    ModifierDuration,
    TargetKind,
)
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import Amulet, HandCard, Unit
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (10244110, 10644120, 10573310, 10674120)
TOKEN_IDS = (90044310, 90044330, 90074140, 90074320)
SOURCE_HASHES = {
    10244110: "2438f2a33579a4cc61f767064e847ce74c0e7292cd48139a8102783ee7e90910",
    90044310: "d98548a36e8b78c25df0c743bd6d9cdaecda932965647d613214fd18aeb471ce",
    10644120: "537b8a4c886a417e8e876c5e2499f53939ea0005cdb46c8fb99c0f7062cfde04",
    90044330: "ef8b99f1c0fe7e4111b1ccecc5076f35b144f549fe587d5b662a17929c767996",
    10573310: "a8c80be2808eb2a16efe708fba8d86f38f9635fd585545842eb53369d6f7c92d",
    90074140: "7abde1fa8f2c436a6805d99abe026718965b9776b4b5a2dffe05f57cd5b2776f",
    10674120: "4fc858e9c92faf34a84a0202606d69102a76f3ac57ab4d2de9eba2b70b41a09f",
    90074320: "bab9bb5cfc4c06662746e6993600123c38cd56a65943593d76a00238d4abf53d",
}


def _choose_id(engine, option_id: str) -> None:
    request = engine.state.pending_choice
    engine.apply(Choose(request.player_index, option_id))


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _put_owner_hand(engine, owner: int, definition) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[owner].hand.append(card)
    engine.players[owner].hand_entity_ids.append(card.entity_id)
    return card


def _discard(engine, hand_card: HandCard) -> None:
    source = _card(999990, name="测试弃牌源", card_type="法术", attack=None, life=None)
    engine._start_effects(
        source,
        None,
        (EffectOperation(EffectKind.DISCARD, TargetKind.OWN_HAND),),
        controller=0,
        label="测试弃牌",
    )
    _choose_id(engine, f"hand:{hand_card.entity_id}")


def _enable_evolution(engine, *, super_evolve: bool = False) -> None:
    player = engine.players[0]
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False
    if super_evolve:
        player.turns_started = engine.config.first_player_super_evolution_unlock_turn
        player.super_evolution_points = max(1, player.super_evolution_points)
    else:
        player.turns_started = engine.config.evolution_unlock_turn
        player.evolution_points = max(1, player.evolution_points)


class HiddenCopyDiscardSchemaTests(unittest.TestCase):
    def test_schema_accepts_enemy_hand_cost_and_copy_cost_modifier(self):
        cost = _parse_operation(
            {
                "kind": "change_cost",
                "target": "all_enemy_hand",
                "amount": 1,
                "mode": "add",
                "duration": "until_end_of_opponent_turn",
            },
            "test",
            123,
        )
        self.assertIs(cost.target, TargetKind.ALL_ENEMY_HAND)
        self.assertIs(cost.duration, ModifierDuration.UNTIL_END_OF_OPPONENT_TURN)

        copy = _parse_operation(
            {
                "kind": "copy_to_hand",
                "target": "own_unit",
                "amount": 3,
                "mode": "subtract",
                "target_cost_min": 5,
            },
            "test",
            123,
        )
        self.assertIs(copy.kind, EffectKind.COPY_TO_HAND)
        self.assertIs(copy.mode, CostChangeMode.SUBTRACT)
        self.assertEqual(copy.board_filter.cost_min, 5)

    def test_copy_schema_rejects_non_board_target_and_invalid_amount(self):
        for raw in (
            {"kind": "copy_to_hand", "target": "own_leader", "amount": 3},
            {"kind": "copy_to_hand", "target": "own_unit"},
            {"kind": "copy_to_hand", "target": "own_unit", "amount": True},
            {"kind": "copy_to_hand", "target": "own_unit", "amount": -1},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test", 123)

    def test_discarded_is_a_structured_trigger(self):
        self.assertIs(Trigger("discarded"), Trigger.DISCARDED)


class RealGeneratedHiddenCopyDiscardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1601):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_filin_overflow_produces_whisper_and_evolve_hits_all_enemies(self):
        below = self.fresh(seed=3)
        below.players[0].max_mana = below.players[0].mana = 6
        source = _play(below, self.repository, 10244110)
        self.assertTrue(source.has_keyword("必杀"))
        self.assertFalse(any(card.card_id == 90044310 for card in below.players[0].hand))

        active = self.fresh(seed=5)
        active.players[0].max_mana = active.players[0].mana = 7
        enemies = [_put_unit(active, 1, _card(100 + index, life=4)) for index in range(2)]
        source = _play(active, self.repository, 10244110)
        self.assertTrue(any(card.card_id == 90044310 for card in active.players[0].hand))
        _enable_evolution(active)
        active.apply(Evolve(0, source.entity_id))
        self.assertEqual([enemy.health for enemy in enemies], [3, 3])

    def test_whitefrost_destroy_mode_only_destroys_damaged_followers(self):
        engine = self.fresh(seed=7)
        damaged = _put_unit(engine, 1, _card(200, life=5))
        damaged.health = 4
        healthy = _put_unit(engine, 1, _card(201, life=5))
        _put_hand(engine, self.repository.get(90044310))

        engine.apply(PlayCard(0, 0))
        _choose_id(engine, "choose_one:destroy_damaged")

        self.assertNotIn(damaged, engine.players[1].board)
        self.assertIn(healthy, engine.players[1].board)

    def test_whitefrost_enemy_hand_cost_expires_after_opponent_turn(self):
        engine = self.fresh(seed=11)
        first = _put_owner_hand(engine, 1, _card(300, cost=2))
        second = _put_owner_hand(engine, 1, _card(301, cost=5))
        _put_hand(engine, self.repository.get(90044310))

        engine.apply(PlayCard(0, 0))
        _choose_id(engine, "choose_one:freeze_enemy_hand")
        self.assertEqual((first.current_cost, second.current_cost), (3, 6))
        engine.apply(EndTurn(0))
        self.assertEqual((first.current_cost, second.current_cost), (3, 6))
        engine.apply(EndTurn(1))
        self.assertEqual((first.current_cost, second.current_cost), (2, 5))

    def test_vorlalai_discard_summons_copy_and_emits_identity(self):
        engine = self.fresh(seed=13)
        discarded = _put_hand(engine, self.repository.get(10644120))

        _discard(engine, discarded)

        summoned = next(unit for unit in engine.players[0].board if unit.definition.card_id == 10644120)
        self.assertTrue(summoned.has_keyword("必杀"))
        self.assertTrue(any(card.entity_id == discarded.entity_id for card in engine.players[0].graveyard))
        event = next(event for event in engine.event_history if event.type is EventType.CARD_DISCARDED)
        self.assertEqual(event.source_id, discarded.entity_id)
        self.assertEqual(event.metadata["card_id"], 10644120)

    def test_vorlalai_discard_on_full_board_safely_skips_summon(self):
        engine = self.fresh(seed=17)
        for index in range(engine.config.max_board):
            _put_unit(engine, 0, _card(400 + index))
        discarded = _put_hand(engine, self.repository.get(10644120))

        _discard(engine, discarded)

        self.assertEqual(len(engine.players[0].board), engine.config.max_board)
        self.assertEqual(
            sum(card.definition.card_id == 10644120 for card in engine.players[0].graveyard),
            1,
        )

    def test_eld_blades_spell_resolves_when_played_or_discarded(self):
        played = self.fresh(seed=19)
        played.players[0].health = 18
        _put_hand(played, self.repository.get(90044330))
        played.apply(PlayCard(0, 0))
        self.assertEqual((played.players[0].health, played.players[1].health), (19, 19))

        discarded_engine = self.fresh(seed=23)
        discarded_engine.players[0].health = 18
        discarded = _put_hand(discarded_engine, self.repository.get(90044330))
        _discard(discarded_engine, discarded)
        self.assertEqual(
            (discarded_engine.players[0].health, discarded_engine.players[1].health),
            (19, 19),
        )

    def test_vorlalai_evolve_adds_one_and_super_evolve_adds_three(self):
        evolved = self.fresh(seed=29)
        source = _play(evolved, self.repository, 10644120)
        _enable_evolution(evolved)
        evolved.apply(Evolve(0, source.entity_id))
        self.assertEqual(sum(card.card_id == 90044330 for card in evolved.players[0].hand), 1)

        super_evolved = self.fresh(seed=31)
        source = _play(super_evolved, self.repository, 10644120)
        _enable_evolution(super_evolved, super_evolve=True)
        super_evolved.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(
            sum(card.card_id == 90044330 for card in super_evolved.players[0].hand),
            3,
        )

    def test_honest_blossom_transforms_amulet_with_stable_identity(self):
        engine = self.fresh(seed=37)
        amulet = Amulet(
            definition=_card(500, name="测试护符", card_type="护符", attack=None, life=None),
            entity_id=engine.state.allocate_entity_id(),
            entered_turn=engine.turn,
        )
        engine.players[1].board.append(amulet)
        _put_hand(engine, self.repository.get(10573310))

        engine.apply(PlayCard(0, 0))
        self.assertTrue(any(option.entity_id == amulet.entity_id for option in engine.state.pending_choice.options))
        _choose_entity(engine, amulet.entity_id)

        transformed = next(entity for entity in engine.players[1].board if entity.entity_id == amulet.entity_id)
        self.assertIsInstance(transformed, Unit)
        self.assertEqual(transformed.definition.card_id, 90074140)
        self.assertTrue(transformed.has_keyword("突进"))
        self.assertIs(transformed.origin, CardOrigin.TRANSFORMED)
        event = next(event for event in engine.event_history if event.type is EventType.BOARD_CARD_TRANSFORMED)
        self.assertEqual((event.metadata["old_card_type"], event.metadata["new_card_type"]), ("护符", "随从"))

    def test_honest_blossom_requires_a_board_target_without_mutation(self):
        engine = self.fresh(seed=41)
        _put_hand(engine, self.repository.get(10573310))
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_yuzeta_produces_axe_only_with_high_cost_follower(self):
        below = self.fresh(seed=43)
        source = _play(below, self.repository, 10674120)
        self.assertTrue(source.has_keyword("突进"))
        self.assertFalse(any(card.card_id == 90074320 for card in below.players[0].hand))

        active = self.fresh(seed=47)
        _put_unit(active, 0, _card(600, cost=5))
        _play(active, self.repository, 10674120)
        self.assertTrue(any(card.card_id == 90074320 for card in active.players[0].hand))

    def test_eld_axe_copies_selected_follower_hidden_and_reduces_cost(self):
        engine = self.fresh(seed=53)
        target = _put_unit(engine, 0, _card(700, name="秘密高费随从", cost=8))
        _put_hand(engine, self.repository.get(90074320))

        engine.apply(PlayCard(0, 0))
        _choose_entity(engine, target.entity_id)

        copied = next(card for card in engine.players[0].hand if card.card_id == 700)
        self.assertEqual(copied.current_cost, 5)
        self.assertIs(copied.origin, CardOrigin.GENERATED)
        event = next(
            event for event in engine.event_history
            if event.type is EventType.CARD_ADDED_TO_HAND
            and event.metadata.get("copied_from_entity_id") == target.entity_id
        )
        self.assertFalse(event.metadata["revealed"])
        self.assertEqual(event.metadata["cost_after"], 5)
        self.assertNotIn("秘密高费随从", engine.logs[-1])

    def test_eld_axe_full_hand_burns_copy_without_revealing(self):
        engine = self.fresh(seed=59)
        target = _put_unit(engine, 0, _card(800, name="满手复制目标", cost=6))
        _put_hand(engine, self.repository.get(90074320))
        for index in range(engine.config.max_hand - 1):
            _put_owner_hand(engine, 0, _card(810 + index))
        engine.apply(PlayCard(0, 0))
        _put_owner_hand(engine, 0, _card(899))
        self.assertEqual(len(engine.players[0].hand), engine.config.max_hand)

        _choose_entity(engine, target.entity_id)

        self.assertFalse(any(card.card_id == 800 for card in engine.players[0].hand))
        self.assertTrue(any(card.definition.card_id == 800 for card in engine.players[0].graveyard))
        self.assertNotIn("满手复制目标", engine.logs[-1])

    def test_all_eight_cards_are_exact_and_tokens_have_real_producers(self):
        coverage = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in COLLECTIBLE_IDS:
            with self.subTest(card_id=card_id):
                info = coverage["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )

        audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        tokens = {card["card_id"]: card for card in audit["cards"]}
        for card_id in TOKEN_IDS:
            with self.subTest(card_id=card_id):
                info = tokens[card_id]
                self.assertEqual(info["category"], "entry_behavior_complete")
                self.assertEqual(info["explicit_coverage"], "exact")
                self.assertTrue(info["authored_producers"])
                self.assertEqual(len(SOURCE_HASHES[card_id]), 64)


if __name__ == "__main__":
    unittest.main()
