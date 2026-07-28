# -*- coding: utf-8 -*-
"""Exact oldest-first distributed damage and Octrice crest/token chains."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, _parse_operation
from swb.engine.commands import (
    ActivateAmulet,
    BeginFusion,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
)
from swb.engine.effects import EffectKind, EffectOperation, ExprType, TargetKind
from swb.engine.emblem import EmblemDefinition, EmblemStacking
from swb.engine.events import EventType
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (
    10113120,
    10154130,
    10324120,
    10363210,
    10511310,
    10514120,
    10673110,
    10753310,
)
TOKEN_IDS = (90024310,)
SOURCE_HASHES = {
    10113120: "721d6ad795c04207e134eb384b060a9bf12b5ca115b66a4130799c4240ac2e27",
    10154130: "27839cc83dba2c316f2b3a4e4794ada297f7e4423c5b1b6ab8159d8824e3686c",
    10324120: "a5b66cee06015986df92c03d0a325eb8ff11665ce73051a804bde3e9ef90310f",
    10363210: "fd74a641ffe6f63617a4e89047992330eddc3296170af670117403189c938a08",
    10511310: "0810d42507753370e129117a9c5f18c2499ce5294b55c6cca0d6d6963dc4180b",
    10514120: "4dddc1f7ad882db73f05a2c7288b2636863edb7c6935bada6dec3a88c9371fec",
    10673110: "3f3610e64dc8983ca2cef7c4f9870e5c5864370cef871bc7a17dffbaf62d2e05",
    10753310: "671cf9f554f3e924ca095e545b27de1f72c353c417c241a0e35e784b5a383398",
    90024310: "4c6a54931490f57a6dd74ed8179fb84a5f0b833998437453360a9d96610db00d",
}


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _choose_option(engine, option_id: str) -> None:
    request = engine.state.pending_choice
    assert request is not None
    engine.apply(Choose(request.player_index, option_id))


def _fuse(engine, destination, *materials) -> None:
    engine.apply(BeginFusion(0, destination.entity_id))
    for material in materials:
        _choose_option(engine, f"hand:{material.entity_id}")
    _choose_option(engine, "fusion:confirm")


class RealGeneratedDistributedDamageCrestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1701):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_schema_supports_distribution_dynamic_countdown_and_fusion_filter(self):
        operation = _parse_operation(
            {
                "kind": "distribute_damage",
                "target": "all_enemy_units",
                "amount": {"type": "controller_hand_count"},
                "include_leader": True,
            },
            "test.json/operations[0]",
            1,
        )
        self.assertIs(operation.kind, EffectKind.DISTRIBUTE_DAMAGE)
        self.assertIs(operation.amount_expr.type, ExprType.CONTROLLER_HAND_COUNT)
        self.assertTrue(operation.include_leader)

        countdown = _parse_operation(
            {
                "kind": "reduce_countdown",
                "target": "self",
                "amount": {"type": "controller_emblem_count"},
            },
            "test.json/operations[0]",
            1,
        )
        self.assertIs(countdown.amount_expr.type, ExprType.CONTROLLER_EMBLEM_COUNT)

        emblem = self.rulebook.emblem_def("hollowness_manifest_octrice")
        self.assertEqual([trigger.trigger for trigger in emblem.triggers], [
            "card_played",
            "card_fused",
        ])
        self.assertEqual(
            [trigger.event_filter.tribe_name for trigger in emblem.triggers],
            ["财宝", "财宝"],
        )

    def test_distribution_schema_rejects_ambiguous_shapes(self):
        invalid = (
            {
                "kind": "distribute_damage",
                "target": "enemy_leader",
                "amount": 3,
            },
            {
                "kind": "damage_unit",
                "target": "all_enemy_units",
                "amount": 3,
                "include_leader": True,
            },
            {
                "kind": "distribute_damage",
                "target": "all_enemy_units",
                "amount": 3,
                "target_count": 2,
            },
            {
                "kind": "damage_unit",
                "target": "emblem_self",
                "amount": 1,
            },
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test.json/operations[0]", 1)

    def test_oldest_first_distribution_caps_early_and_overkills_last(self):
        engine = self.fresh(seed=3)
        oldest = _put_unit(engine, 1, _card(990001, life=2))
        newest = _put_unit(engine, 1, _card(990002, life=1))
        newest.add_keyword("屏障")

        engine._start_effects(
            _card(990000, card_type="法术", attack=None, life=None),
            None,
            (
                EffectOperation(
                    EffectKind.DISTRIBUTE_DAMAGE,
                    TargetKind.ALL_ENEMY_UNITS,
                    amount=7,
                ),
            ),
            controller=0,
            label="分配测试",
        )

        self.assertEqual(oldest.health, 0)
        self.assertEqual(newest.health, 1)
        self.assertEqual(newest.barrier_charges, 0)
        self.assertEqual(engine.players[1].health, 20)
        prevented = next(
            event
            for event in engine.event_history
            if event.type is EventType.DAMAGE_PREVENTED
            and event.target_id == newest.entity_id
        )
        self.assertEqual(prevented.amount, 5)

    def test_including_leader_uses_health_caps_even_when_barrier_prevents_damage(self):
        engine = self.fresh(seed=5)
        oldest = _put_unit(engine, 1, _card(990011, life=2))
        oldest.add_keyword("屏障")
        newest = _put_unit(engine, 1, _card(990012, life=1))

        engine._start_effects(
            _card(990010, card_type="法术", attack=None, life=None),
            None,
            (
                EffectOperation(
                    EffectKind.DISTRIBUTE_DAMAGE,
                    TargetKind.ALL_ENEMY_UNITS,
                    amount=5,
                    include_leader=True,
                ),
            ),
            controller=0,
            label="含主战者分配测试",
        )

        self.assertEqual(oldest.health, 2)
        self.assertEqual(newest.health, 0)
        self.assertEqual(engine.players[1].health, 18)

    def test_octrice_gains_one_nonduplicating_crest_and_evolve_adds_loot(self):
        engine = self.fresh(seed=7)
        source = _play(engine, self.repository, 10324120)
        self.assertEqual(len(engine.players[0].emblems), 1)
        emblem = engine.players[0].emblems[0]
        self.assertEqual(emblem.countdown, 8)

        _play(engine, self.repository, 10324120)
        self.assertEqual(len(engine.players[0].emblems), 1)
        self.assertIs(engine.players[0].emblems[0], emblem)
        self.assertEqual(emblem.countdown, 8)

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            sorted(card.card_id for card in engine.players[0].hand),
            [90021310, 90021340],
        )

    def test_octrice_crest_filters_plays_and_multi_material_fusion_ticks_once(self):
        engine = self.fresh(seed=11)
        _play(engine, self.repository, 10324120)
        emblem = engine.players[0].emblems[0]

        _play(engine, self.repository, 10511310)
        self.assertEqual(emblem.countdown, 8)

        _put_hand(engine, self.repository.get(90021320))
        engine.players[0].health = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(emblem.countdown, 7)

        destination = _put_hand(engine, self.repository.get(10323310))
        dagger = _put_hand(engine, self.repository.get(90021310))
        necklace = _put_hand(engine, self.repository.get(90021340))
        _fuse(engine, destination, dagger, necklace)
        self.assertEqual(emblem.countdown, 6)
        fused = next(
            event
            for event in engine.event_history
            if event.type is EventType.CARD_FUSED
        )
        self.assertEqual(fused.amount, 2)
        self.assertEqual(
            tuple(card.card_id for card in fused.metadata["material_definitions"]),
            (90021310, 90021340),
        )

    def test_octrice_fusion_expiration_adds_executable_remnant(self):
        engine = self.fresh(seed=13)
        _play(engine, self.repository, 10324120)
        emblem = engine.players[0].emblems[0]
        emblem.countdown = 1
        destination = _put_hand(engine, self.repository.get(10323310))
        material = _put_hand(engine, self.repository.get(90021310))

        _fuse(engine, destination, material)

        self.assertFalse(engine.players[0].emblems)
        remnant = next(
            card for card in engine.players[0].hand if card.card_id == 90024310
        )
        target = _put_unit(engine, 1, _card(990020, life=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, engine.players[0].hand.index(remnant)))
        self.assertEqual(target.health, 0)
        self.assertEqual(engine.players[1].health, 17)

        no_followers = self.fresh(seed=14)
        _put_hand(no_followers, self.repository.get(90024310))
        no_followers.apply(PlayCard(0, 0))
        self.assertEqual(no_followers.players[1].health, 16)

    def test_backwood_draws_then_evolve_uses_current_hand_count(self):
        engine = self.fresh(seed=17)
        oldest = _put_unit(engine, 1, _card(990031, life=1))
        newest = _put_unit(engine, 1, _card(990032, life=5))
        source = _play(engine, self.repository, 10113120)
        self.assertEqual(len(engine.players[0].hand), 2)

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(oldest.health, 0)
        self.assertEqual(newest.health, 4)

    def test_aragavy_distributes_seven_and_evolve_damages_both_leaders(self):
        engine = self.fresh(seed=19)
        oldest = _put_unit(engine, 1, _card(990041, life=2))
        newest = _put_unit(engine, 1, _card(990042, life=8))
        source = _play(engine, self.repository, 10154130)
        self.assertEqual(oldest.health, 0)
        self.assertEqual(newest.health, 3)

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            (engine.players[0].health, engine.players[1].health),
            (17, 17),
        )

    def test_shining_disappointment_act_uses_emblem_count_then_resolves_last_words(self):
        engine = self.fresh(seed=23)
        source = _play(engine, self.repository, 10363210)
        source.countdown = 2
        for index in range(2):
            definition = EmblemDefinition(
                f"counted-{index}",
                990050 + index,
                stacking=EmblemStacking.ALLOW,
            )
            engine._add_emblem_to_player(0, definition, definition.source_card_id)
        engine.players[0].health = 10
        target = _put_unit(engine, 1, _card(990052, life=1))

        engine.apply(ActivateAmulet(0, source.entity_id))

        self.assertNotIn(source, engine.players[0].board)
        self.assertEqual(target.health, 0)
        self.assertEqual(engine.players[1].health, 17)
        self.assertEqual(engine.players[0].health, 14)

    def test_insect_flight_and_all_miroku_modes(self):
        flight = self.fresh(seed=29)
        target = _put_unit(flight, 1, _card(990061, life=5))
        _play(flight, self.repository, 10511310)
        self.assertEqual(target.health, 2)
        self.assertEqual(
            [card.card_id for card in flight.players[0].hand],
            [90011110],
        )

        fairies = self.fresh(seed=31)
        _play(fairies, self.repository, 10514120)
        _choose_option(fairies, "choose_one:add_fairies")
        self.assertEqual(
            [card.card_id for card in fairies.players[0].hand],
            [90011110, 90011110],
        )

        mana = self.fresh(seed=37)
        _play(mana, self.repository, 10514120)
        self.assertEqual(mana.players[0].mana, 7)
        _choose_option(mana, "choose_one:restore_mana")
        self.assertEqual(mana.players[0].mana, 9)

        damage = self.fresh(seed=41)
        source = _play(damage, self.repository, 10514120)
        _choose_option(damage, "choose_one:restore_mana")
        target = _put_unit(damage, 1, _card(990062, life=5))
        _enable_evolution(damage)
        damage.apply(Evolve(0, source.entity_id))
        _choose_option(damage, "choose_one:distribute_damage")
        self.assertEqual(target.health, 2)

    def test_foolish_weapon_copies_do_not_recurse_and_each_turn_end_triggers(self):
        engine = self.fresh(seed=43)
        target = _put_unit(engine, 1, _card(990071, life=30))
        source = _play(engine, self.repository, 10673110)
        self.assertEqual(
            [unit.definition.card_id for unit in engine.players[0].board],
            [10673110, 10673110, 10673110],
        )

        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(target.health, 27)
        engine.apply(EndTurn(0))
        self.assertEqual(target.health, 18)

        accelerated = self.fresh(seed=44)
        accelerated.players[0].mana = 7
        _play(
            accelerated,
            self.repository,
            10673110,
            mode_id="accelerate_4",
        )
        self.assertEqual(
            [unit.definition.card_id for unit in accelerated.players[0].board],
            [10673110],
        )
        self.assertEqual(accelerated.players[0].mana, 3)

    def test_concert_necromancy_is_optional_by_resource_availability(self):
        enough = self.fresh(seed=47)
        target = _put_unit(enough, 1, _card(990081, life=8))
        enough.players[0].shadows = 6
        _play(enough, self.repository, 10753310)
        self.assertEqual(target.health, 2)
        self.assertEqual(enough.players[1].health, 18)
        self.assertEqual(enough.players[0].shadows, 1)

        short = self.fresh(seed=53)
        short.players[0].shadows = 5
        _play(short, self.repository, 10753310)
        self.assertEqual(short.players[1].health, 20)
        self.assertEqual(short.players[0].shadows, 6)

    def test_crest_fusion_sequence_is_seed_reproducible(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=59)
            _play(engine, self.repository, 10324120)
            destination = _put_hand(engine, self.repository.get(10323310))
            first = _put_hand(engine, self.repository.get(90021310))
            second = _put_hand(engine, self.repository.get(90021340))
            _fuse(engine, destination, first, second)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_all_cards_are_exact_and_remnant_has_real_producer(self):
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
        remnant = tokens[90024310]
        self.assertEqual(remnant["category"], "entry_behavior_complete")
        self.assertEqual(remnant["explicit_coverage"], "exact")
        self.assertTrue(remnant["authored_producers"])


if __name__ == "__main__":
    unittest.main()
