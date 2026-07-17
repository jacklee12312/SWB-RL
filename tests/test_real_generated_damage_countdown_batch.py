# -*- coding: utf-8 -*-
"""Exact survived-damage, Burst crest, and amulet-countdown token chains."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import DamageType
from swb.engine.state import Amulet
from swb.engine.targeting import target_candidates
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (10344120, 10434120, 10764110)
TOKEN_IDS = (90044320, 90034320, 90064210)
SOURCE_HASHES = {
    10344120: "02c75f4a0c9747489ff4ae5282aafd380db637ff6fe3c9cb052e45872d0a014e",
    90044320: "014bd0aa67c48094be95aab6b9064c04d2c438a2c83517f96ad83b0dad08803c",
    10434120: "5107cc463a5ff6fec38fdaac05a7e780ba7403a805d2715853004073e30ff036",
    90034320: "3e2a4e2725fd530bac7d8caadbe5416de30e8cc6f272a60ccf4cb60ed261e670",
    10764110: "c653df9a1d602fb409b01d64f5645d8f5b3d7c3b748eda1a990c910750d121e3",
    90064210: "1a76774b368b49782668d83504e101eba2f28fe777f77fdaf2355a4edf512133",
}


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _damage_self(engine, source, amount: int) -> None:
    engine._start_effects(
        source.definition,
        source.entity_id,
        (
            EffectOperation(
                kind=EffectKind.DAMAGE_UNIT,
                target=TargetKind.SELF,
                amount=amount,
            ),
        ),
        controller=0,
        label="测试自伤",
    )


def _put_amulet(engine, card_id: int) -> Amulet:
    amulet = Amulet(
        definition=_card(
            card_id,
            card_type="护符",
            attack=None,
            life=None,
        ),
        entity_id=engine.state.allocate_entity_id(),
        entered_turn=engine.turn,
        origin=CardOrigin.TOKEN,
    )
    engine.players[0].board.append(amulet)
    return amulet


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options
        if option.entity_id == entity_id
    )
    engine.apply(Choose(request.player_index, option.option_id))


def _choose_enemy_leader(engine) -> None:
    request = engine.state.pending_choice
    option = next(
        option for option in request.options
        if option.leader_player_index == 1
    )
    engine.apply(Choose(request.player_index, option.option_id))


class RealGeneratedDamageCountdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1501):
        return _fresh(self.rulebook, self.repository, seed=seed)

    @staticmethod
    def _fang_count(engine) -> int:
        return sum(
            card.card_id == 90044320
            for card in engine.players[0].hand
        )

    def test_galmieux_self_survival_triggers_follower_and_crest_once_each(self):
        engine = self.fresh(seed=3)
        source = _play(engine, self.repository, 10344120)
        self.assertFalse(source.has_keyword("疾驰"))
        self.assertTrue(any(
            emblem.emblem_id == "manifestation_of_ardent_destruction_galmieux"
            for emblem in engine.players[0].emblems
        ))
        enemy = _put_unit(engine, 1, _card(999200, life=6))

        _damage_self(engine, source, 1)
        self.assertEqual(source.health, 4)
        self.assertEqual(enemy.health, 3)
        self.assertEqual(self._fang_count(engine), 1)
        _damage_self(engine, source, 1)
        self.assertEqual(enemy.health, 3)
        self.assertEqual(self._fang_count(engine), 1)
        survived = [
            event for event in engine.event_history
            if event.type is EventType.FOLLOWER_DAMAGED_SURVIVED
            and event.source_id == source.entity_id
        ]
        self.assertEqual(len(survived), 2)
        self.assertEqual(survived[0].metadata["health_after"], 4)

    def test_galmieux_crest_tracks_other_survivor_but_not_lethal_or_opponent_turn(self):
        other = self.fresh(seed=5)
        source = _play(other, self.repository, 10344120)
        ally = _put_unit(other, 0, _card(999201, life=3))
        other.apply_damage(
            source,
            ally,
            1,
            DamageType.EFFECT,
            0,
        )
        other._resolve_event_queue()
        other._stabilize()
        self.assertEqual(self._fang_count(other), 1)

        lethal = self.fresh(seed=7)
        source = _play(lethal, self.repository, 10344120)
        _damage_self(lethal, source, 5)
        self.assertEqual(self._fang_count(lethal), 0)
        self.assertNotIn(source, lethal.players[0].board)

        opponent_turn = self.fresh(seed=11)
        source = _play(opponent_turn, self.repository, 10344120)
        opponent_turn.apply(EndTurn(0))
        _damage_self(opponent_turn, source, 1)
        self.assertEqual(self._fang_count(opponent_turn), 0)

    def test_galmieux_enhance_grants_storm_and_fang_hits_all_followers(self):
        engine = self.fresh(seed=13)
        source = _play(
            engine,
            self.repository,
            10344120,
            mode_id="enhance_7",
        )
        ally = _put_unit(engine, 0, _card(999202, life=4))
        enemy = _put_unit(engine, 1, _card(999203, life=6))
        self.assertTrue(source.has_keyword("疾驰"))

        fang = _put_hand(engine, self.repository.get(90044320))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, engine.players[0].hand.index(fang)))
        self.assertEqual((source.health, ally.health, enemy.health), (4, 3, 2))
        self.assertEqual(self._fang_count(engine), 1)

    def test_cagliostro_fanfare_and_burst_thresholds_are_additive(self):
        below = self.fresh(seed=17)
        below.players[0].turns_started = 9
        source = _play(below, self.repository, 10434120)
        self.assertFalse(source.evolved)
        self.assertEqual(below.players[0].earth_sigils, 2)
        self.assertEqual(sum(
            card.card_id == 90034320 for card in below.players[0].hand
        ), 1)

        union = self.fresh(seed=19)
        union.players[0].turns_started = 10
        source = _play(union, self.repository, 10434120)
        self.assertTrue(source.evolved)
        self.assertFalse(any(
            emblem.emblem_id == "genius_alchemist_cagliostro"
            for emblem in union.players[0].emblems
        ))

        liberation = self.fresh(seed=23)
        liberation.players[0].turns_started = 15
        source = _play(liberation, self.repository, 10434120)
        self.assertTrue(source.evolved)
        self.assertTrue(any(
            emblem.emblem_id == "genius_alchemist_cagliostro"
            for emblem in liberation.players[0].emblems
        ))

    def test_cagliostro_crest_spends_earth_rite_and_stops_when_empty(self):
        engine = self.fresh(seed=29)
        engine.players[0].turns_started = 15
        _play(engine, self.repository, 10434120)
        self.assertEqual(engine.players[0].earth_sigils, 2)
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        self.assertEqual(engine.players[0].earth_sigils, 1)
        self.assertEqual(sum(
            card.card_id == 90034320 for card in engine.players[0].hand
        ), 2)

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        self.assertEqual(engine.players[0].earth_sigils, 0)
        before = sum(
            card.card_id == 90034320 for card in engine.players[0].hand
        )
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        self.assertEqual(sum(
            card.card_id == 90034320 for card in engine.players[0].hand
        ), before)

    def test_great_magic_selects_follower_or_leader_then_heals(self):
        follower = self.fresh(seed=31)
        follower.players[0].health = 18
        enemy = _put_unit(follower, 1, _card(999204, life=5))
        spell = _put_hand(follower, self.repository.get(90034320))
        follower.apply(PlayCard(0, follower.players[0].hand.index(spell)))
        _choose_entity(follower, enemy.entity_id)
        self.assertEqual(enemy.health, 3)
        self.assertEqual(follower.players[0].health, 19)

        leader = self.fresh(seed=37)
        leader.players[0].health = 18
        spell = _put_hand(leader, self.repository.get(90034320))
        leader.apply(PlayCard(0, leader.players[0].hand.index(spell)))
        _choose_enemy_leader(leader)
        self.assertEqual(leader.players[1].health, 18)
        self.assertEqual(leader.players[0].health, 19)

    def test_rodeo_summons_ring_and_evolve_increases_only_its_countdown(self):
        engine = self.fresh(seed=41)
        other = _put_amulet(engine, 999205)
        source = _play(engine, self.repository, 10764110)
        ring = next(
            entity for entity in engine.players[0].board
            if isinstance(entity, Amulet)
            and entity.definition.card_id == 90064210
        )
        self.assertEqual(ring.countdown, 1)
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(ring.countdown, 2)
        self.assertIsNone(other.countdown)
        event = next(
            event for event in engine.event_history
            if event.type is EventType.AMULET_COUNTDOWN_CHANGED
        )
        self.assertEqual(
            (event.source_id, event.amount),
            (ring.entity_id, 1),
        )

    def test_moon_ring_turn_end_requires_three_amulets_and_has_aura(self):
        active = self.fresh(seed=43)
        _play(active, self.repository, 10764110)
        ring = next(
            entity for entity in active.players[0].board
            if isinstance(entity, Amulet)
            and entity.definition.card_id == 90064210
        )
        _put_amulet(active, 999206)
        _put_amulet(active, 999207)
        enemy = _put_unit(active, 1, _card(999208, life=5))
        candidates = target_candidates(
            EffectOperation(
                kind=EffectKind.DESTROY,
                target=TargetKind.ENEMY_AMULET,
            ),
            1,
            active.players,
        )
        self.assertNotIn(ring, candidates)
        active.apply(EndTurn(0))
        self.assertEqual(enemy.health, 2)
        self.assertEqual(active.players[1].health, 17)

        below = self.fresh(seed=47)
        _play(below, self.repository, 10764110)
        enemy = _put_unit(below, 1, _card(999209, life=5))
        below.apply(EndTurn(0))
        self.assertEqual(enemy.health, 5)
        self.assertEqual(below.players[1].health, 20)

    def test_rodeo_full_board_skips_ring_and_safe_evolution(self):
        engine = self.fresh(seed=53)
        for index in range(4):
            _put_unit(engine, 0, _card(999220 + index))
        source = _play(engine, self.repository, 10764110)
        self.assertFalse(any(
            isinstance(entity, Amulet)
            and entity.definition.card_id == 90064210
            for entity in engine.players[0].board
        ))
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertIsNone(engine.state.pending_choice)
        self.assertTrue(source.evolved)

    def test_all_six_cards_are_exact_and_tokens_have_real_producers(self):
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
