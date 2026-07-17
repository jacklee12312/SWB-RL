# -*- coding: utf-8 -*-
"""Exact Spellboost growth, stat-increase listeners, and generated-token chains."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Attack, EndTurn, Evolve, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (10814120, 10232110, 10133310)
TOKEN_IDS = (90014110, 90032110, 90033310)
SOURCE_HASHES = {
    10814120: "b4b579354c1c14e4d63add1f09dff4ed747e799e024638978a242062d32f9444",
    90014110: "aa52b7161a6c63106daab4129b5846597d52b722f34aad71e5973667f1f3affa",
    10232110: "f9f0b155ded66e9d8d1702aeaf4154495c4297fa51f4e56aa8ac84d6feb71c7d",
    90032110: "225926651cbb723e29109208bf7b14056dd05465a196358f4ca7d626449ea933",
    10133310: "558ee927becae510471e229c7b485012a8ee17d0388d50a0b84ea6023d13ae87",
    90033310: "e2748181f7691bdefe9f1611800f020ee44da0748056eb7e27d1d676854c6575",
}


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _buff_self(engine, source, attack: int, health: int) -> None:
    engine._start_effects(
        source.definition,
        source.entity_id,
        (
            EffectOperation(
                kind=EffectKind.BUFF_UNIT,
                target=TargetKind.SELF,
                amount=attack,
                secondary_amount=health,
            ),
        ),
        controller=0,
        label="测试属性变化",
    )


class RealGeneratedSpellboostGrowthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1401):
        return _fresh(self.rulebook, self.repository, seed=seed)

    @staticmethod
    def _eve_count(engine) -> int:
        return sum(
            card.card_id == 90014110
            for card in engine.players[0].hand
        )

    def test_tia_enhance_buffs_all_and_its_own_increase_adds_eve(self):
        engine = self.fresh(seed=3)
        ally = _put_unit(engine, 0, _card(998901, attack=2, life=3))
        source = _play(
            engine,
            self.repository,
            10814120,
            mode_id="enhance_4",
        )

        self.assertEqual((ally.attack, ally.max_health), (3, 4))
        self.assertEqual((source.attack, source.max_health), (3, 3))
        self.assertTrue(source.has_keyword("突进"))
        self.assertEqual(self._eve_count(engine), 1)
        event = next(
            event for event in engine.event_history
            if event.type is EventType.FOLLOWER_STATS_INCREASED
            and event.source_id == source.entity_id
        )
        self.assertEqual(
            (event.metadata["attack_delta"], event.metadata["health_delta"]),
            (1, 1),
        )

    def test_tia_triggers_only_for_positive_own_turn_change_once_per_turn(self):
        engine = self.fresh(seed=5)
        source = _play(engine, self.repository, 10814120)
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(self._eve_count(engine), 0)

        _buff_self(engine, source, -1, 0)
        self.assertEqual(self._eve_count(engine), 0)
        _buff_self(engine, source, 0, 1)
        _buff_self(engine, source, 1, 0)
        self.assertEqual(self._eve_count(engine), 1)

        engine.apply(EndTurn(0))
        _buff_self(engine, source, 1, 1)
        self.assertEqual(self._eve_count(engine), 1)
        engine.apply(EndTurn(1))
        _buff_self(engine, source, 1, 0)
        self.assertEqual(self._eve_count(engine), 2)

    def test_tia_removed_abilities_do_not_trigger_and_eve_has_both_keywords(self):
        engine = self.fresh(seed=7)
        source = _play(engine, self.repository, 10814120)
        source.remove_all_abilities()
        _buff_self(engine, source, 1, 1)
        self.assertEqual(self._eve_count(engine), 0)

        complete = self.fresh(seed=11)
        _play(complete, self.repository, 10814120, mode_id="enhance_4")
        eve = next(card for card in complete.players[0].hand if card.card_id == 90014110)
        complete.players[0].mana = 10
        complete.apply(PlayCard(0, complete.players[0].hand.index(eve)))
        unit = next(
            unit for unit in complete.players[0].board
            if unit.definition.card_id == 90014110
        )
        self.assertTrue(unit.has_keyword("疾驰"))
        self.assertTrue(unit.has_keyword("守护"))

    def test_homework_transforms_exactly_on_fifth_spellboost_with_stable_id(self):
        engine = self.fresh(seed=13)
        homework = _put_hand(engine, self.repository.get(10133310))
        entity_id = homework.entity_id

        for count in range(1, 6):
            _put_hand(engine, self.repository.get(10031310))
            engine.players[0].mana = 10
            engine.apply(PlayCard(0, 0))
            if count < 5:
                self.assertEqual(homework.card_id, 10133310)
                self.assertEqual(homework.spellboost_count, count)

        self.assertEqual((homework.card_id, homework.entity_id), (90033310, entity_id))
        self.assertEqual(homework.spellboost_count, 0)
        transformed = [
            event for event in engine.event_history
            if event.type is EventType.HAND_CARD_TRANSFORMED
            and event.source_id == entity_id
        ]
        self.assertEqual(len(transformed), 1)

        _put_hand(engine, self.repository.get(10031310))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(homework.card_id, 90033310)
        self.assertEqual(homework.spellboost_count, 1)

    def test_homework_and_grown_spell_resolve_draw_and_empty_enemy_paths(self):
        homework_game = self.fresh(seed=17)
        homework = _put_hand(homework_game, self.repository.get(10133310))
        deck_before = len(homework_game.players[0].deck)
        homework_game.apply(PlayCard(0, homework_game.players[0].hand.index(homework)))
        self.assertEqual(len(homework_game.players[0].deck), deck_before - 2)

        grown_game = self.fresh(seed=19)
        grown = _put_hand(grown_game, self.repository.get(90033310))
        enemy = _put_unit(grown_game, 1, _card(998902, attack=0, life=5))
        deck_before = len(grown_game.players[0].deck)
        grown_game.apply(PlayCard(0, grown_game.players[0].hand.index(grown)))
        self.assertEqual(len(grown_game.players[0].deck), deck_before - 2)
        self.assertEqual(enemy.health, 3)

        no_enemy = self.fresh(seed=23)
        grown = _put_hand(no_enemy, self.repository.get(90033310))
        deck_before = len(no_enemy.players[0].deck)
        no_enemy.apply(PlayCard(0, no_enemy.players[0].hand.index(grown)))
        self.assertEqual(len(no_enemy.players[0].deck), deck_before - 2)
        self.assertIsNone(no_enemy.state.pending_choice)

    def test_basset_summons_onions_whose_attack_spellboosts_the_hand(self):
        engine = self.fresh(seed=29)
        source = _play(engine, self.repository, 10232110)
        onions = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90032110
        ]
        self.assertEqual(len(onions), 2)
        self.assertTrue(all(unit.has_keyword("突进") for unit in onions))
        tracked = _put_hand(
            engine,
            _card(998903, card_type="法术", attack=None, life=None),
        )
        enemy = _put_unit(engine, 1, _card(998904, attack=0, life=5))
        engine.apply(Attack(0, onions[0].entity_id, enemy.entity_id))
        self.assertEqual(tracked.spellboost_count, 1)
        self.assertIn(source, engine.players[0].board)

    def test_basset_crest_summons_at_owner_turn_start_and_respects_capacity(self):
        engine = self.fresh(seed=31)
        source = _play(engine, self.repository, 10232110)
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertTrue(any(
            emblem.emblem_id == "rejected_artes_bergent"
            for emblem in engine.players[0].emblems
        ))
        before = sum(
            unit.definition.card_id == 90032110
            for unit in engine.players[0].board
        )
        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        after = sum(
            unit.definition.card_id == 90032110
            for unit in engine.players[0].board
        )
        self.assertEqual(after, before + 1)

        full = self.fresh(seed=37)
        for index in range(2):
            _put_unit(full, 0, _card(998910 + index))
        source = _play(full, self.repository, 10232110)
        self.assertEqual(len(full.players[0].board), 5)
        _enable_evolution(full)
        full.apply(Evolve(0, source.entity_id))
        full.apply(EndTurn(0))
        full.apply(EndTurn(1))
        self.assertEqual(len(full.players[0].board), 5)
        self.assertEqual(sum(
            unit.definition.card_id == 90032110
            for unit in full.players[0].board
        ), 2)

    def test_tia_enhance_has_rl_action_mask_parity(self):
        env = ShadowverseEnv(
            [_card(999000 + index) for index in range(40)],
            [_card(999100 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=41,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=41)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].board.clear()
        _put_hand(env.core, self.repository.get(10814120))
        env.players[0].max_mana = env.players[0].mana = 4
        command = PlayCard(0, 0, mode_id="enhance_4")
        action = env._encode_command(command)
        self.assertIsNotNone(action)
        self.assertTrue(env.action_mask()[action])
        env.step(action)
        self.assertTrue(any(
            card.card_id == 90014110
            for card in env.players[0].hand
        ))

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
