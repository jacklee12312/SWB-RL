# -*- coding: utf-8 -*-
"""Tests for batch of real card rules."""

from __future__ import annotations

import os
import sqlite3
import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


def _card(cid, **kw):
    return CardDefinition(
        card_id=cid, card_set_id=kw.get("card_set_id", 10000),
        class_id=1, class_name="\u7cbe\u7075",
        name=kw.get("name", f"c{cid}"), cost=kw.get("cost", 1),
        card_type=kw.get("card_type", "\u968f\u4ece"),
        attack=kw.get("attack", 1), life=kw.get("life", 1),
        keywords=frozenset(), support_level="basic",
        is_collectible=kw.get("is_collectible", True),
    )


def _resolver(defs):
    return lambda cid: defs.get(cid)


def _make_engine(rulebook=None):
    return GameEngine(
        deck_a=[_card(i) for i in range(1000, 1040)],
        deck_b=[_card(i) for i in range(2000, 2040)],
        class_a=1, class_b=1, seed=42,
        rulebook=rulebook or RuleBook(()),
        card_resolver=_resolver({
            90011110: _card(90011110, name="\u5996\u7cbe", card_set_id=90000, is_collectible=False),
            90051120: _card(90051120, name="\u8760\u8760", card_set_id=90000, is_collectible=False),
            90051130: _card(90051130, name="\u6028\u7075", card_set_id=90000, is_collectible=False),
            90071220: _card(90071220, name="\u8fc7\u5f80\u6838\u5fc3", card_type="\u62a4\u7b26", attack=None, life=None, card_set_id=90000, is_collectible=False),
            90021110: _card(90021110, name="\u9a91\u58eb", cost=0, card_set_id=90000, is_collectible=False),
        }),
    )


def _insert_card(engine, card_def, origin=CardOrigin.DECK):
    hc = HandCard(definition=card_def, entity_id=engine.state.allocate_entity_id(), origin=origin)
    engine.players[0].hand.insert(0, hc)
    engine.players[0].hand_entity_ids.insert(0, hc.entity_id)
    return hc


class DatabaseVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(str(cls.db_path)):
            raise unittest.SkipTest("cards.sqlite3 not found")
        cls.repo = CardRepository(str(cls.db_path))

    def _skill_texts(self, card_id):
        with sqlite3.connect(str(self.db_path)) as conn:
            return [
                row[0]
                for row in conn.execute(
                    "SELECT text_chs FROM skill_texts WHERE card_id=? ORDER BY position",
                    (card_id,),
                )
            ]

    def test_10031310_exists(self):
        card = self.repo.get(10031310)
        self.assertEqual(card.name, "\u667a\u6167\u5149\u8f89")
        self.assertEqual(card.cost, 1)
        self.assertEqual(card.card_type, "\u6cd5\u672f")
        self.assertEqual(self._skill_texts(10031310), ["抽取1张卡牌。"])

    def test_10041310_exists(self):
        card = self.repo.get(10041310)
        self.assertIn("\u9f99", card.name)
        self.assertEqual(card.cost, 1)
        text = "\n".join(self._skill_texts(10041310))
        self.assertIn("对其造成2点伤害", text)
        self.assertIn("觉醒", text)
        self.assertIn("改为4点伤害", text)

    def test_10041110_exists(self):
        card = self.repo.get(10041110)
        self.assertIn("\u706b\u8725\u8734", card.name)
        self.assertEqual(card.cost, 2)
        self.assertIn("入场曲", "\n".join(self._skill_texts(10041110)))

    def test_10052310_exists(self):
        card = self.repo.get(10052310)
        self.assertIn("\u6355\u98df", card.name)
        self.assertEqual(card.cost, 2)
        text = "\n".join(self._skill_texts(10052310))
        self.assertIn("破坏该随从", text)
        self.assertIn("抽取2张卡牌", text)

    def test_10111310_exists(self):
        card = self.repo.get(10111310)
        self.assertIn("\u5996\u7cbe", card.name)
        self.assertEqual(card.cost, 1)
        self.assertIn("将2张", "\n".join(self._skill_texts(10111310)))

    def test_10642310_exists(self):
        card = self.repo.get(10642310)
        self.assertIn("\u8d64", card.name)
        self.assertEqual(card.cost, 1)
        text = "\n".join(self._skill_texts(10642310))
        self.assertIn("\u820d\u5f03", text)
        self.assertIn("\u7834\u574f", text)


class DatabaseVerificationBatch2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db_path = os.path.join(os.path.dirname(__file__), "..", "data", "cards.sqlite3")
        if not os.path.exists(str(cls.db_path)):
            raise unittest.SkipTest("cards.sqlite3 not found")
        cls.repo = CardRepository(str(cls.db_path))

    def _skill_texts(self, card_id):
        with sqlite3.connect(str(self.db_path)) as conn:
            return [row[0] for row in conn.execute(
                "SELECT text_chs FROM skill_texts WHERE card_id=? ORDER BY position", (card_id,))]

    def test_10052110_has_lw_add_bat(self):
        card = self.repo.get(10052110)
        self.assertIn("\u9b45\u9b54", card.name)
        self.assertEqual(card.cost, 1)
        self.assertEqual(card.card_type, "\u968f\u4ece")
        text = "\n".join(self._skill_texts(10052110))
        self.assertIn("\u8c22\u5e55\u66f2", text)
        self.assertIn("\u8759\u8760", text)

    def test_10112120_has_lw_add_fairy(self):
        card = self.repo.get(10112120)
        self.assertEqual(card.cost, 1)
        self.assertEqual(card.card_type, "\u968f\u4ece")
        text = "\n".join(self._skill_texts(10112120))
        self.assertIn("\u8c22\u5e55\u66f2", text)
        self.assertIn("\u5996\u7cbe", text)

    def test_10251120_has_lw_add_ghost(self):
        card = self.repo.get(10251120)
        self.assertEqual(card.cost, 1)
        text = "\n".join(self._skill_texts(10251120))
        self.assertIn("\u8c22\u5e55\u66f2", text)
        self.assertIn("\u6028\u7075", text)

    def test_10171110_has_lw_add_core(self):
        card = self.repo.get(10171110)
        self.assertIn("\u4f0a\u8389\u65af", card.name)
        self.assertEqual(card.cost, 1)
        text = "\n".join(self._skill_texts(10171110))
        self.assertIn("\u8c22\u5e55\u66f2", text)
        self.assertIn("\u8fc7\u5f80\u6838\u5fc3", text)
        token_text = "\n".join(self._skill_texts(90071220))
        self.assertIn("\u65e0\u6cd5\u4f7f\u7528", token_text)
        self.assertIn("\u878d\u5408", token_text)

    def test_10601110_has_lw_damage_leader(self):
        card = self.repo.get(10601110)
        self.assertIn("\u6d51\u6d4a", card.name)
        self.assertEqual(card.cost, 2)
        text = "\n".join(self._skill_texts(10601110))
        self.assertIn("\u8c22\u5e55\u66f2", text)
        self.assertIn("\u5bf9\u624b\u7684\u4e3b\u6218\u8005", text)
        self.assertIn("1\u70b9\u4f24\u5bb3", text)

    def test_10651110_has_lw_draw_self_damage(self):
        card = self.repo.get(10651110)
        self.assertIn("\u6e34\u671b", card.name)
        self.assertEqual(card.cost, 2)
        text = "\n".join(self._skill_texts(10651110))
        self.assertIn("\u8c22\u5e55\u66f2", text)
        self.assertIn("\u62bd\u53d61\u5f20\u5361\u724c", text)
        self.assertIn("\u81ea\u5df1\u7684\u4e3b\u6218\u8005", text)

    def test_10571310_is_draw_spell(self):
        card = self.repo.get(10571310)
        self.assertEqual(card.cost, 1)
        self.assertEqual(card.card_type, "\u6cd5\u672f")
        self.assertEqual(self._skill_texts(10571310), ["\u62bd\u53d61\u5f20\u5361\u724c\u3002"])

    def test_10022110_has_lw_summon_knight(self):
        card = self.repo.get(10022110)
        self.assertIn("\u8f66\u592b", card.name)
        self.assertEqual(card.cost, 2)
        text = "\n".join(self._skill_texts(10022110))
        self.assertIn("\u8c22\u5e55\u66f2", text)
        self.assertIn("\u53ec\u55241\u4e2a", text)
        self.assertIn("\u9a91\u58eb", text)


class BehaviorBatch2Tests(unittest.TestCase):
    def setUp(self):
        self.rb = RuleBook.from_directory("data/rules")

    def _make_engine(self):
        return GameEngine(
            deck_a=[_card(i) for i in range(1000, 1040)],
            deck_b=[_card(i) for i in range(2000, 2040)],
            class_a=1, class_b=1, seed=42,
            rulebook=self.rb,
            card_resolver=_resolver({
                90011110: _card(90011110, name="x", card_set_id=90000, is_collectible=False),
                90051120: _card(90051120, name="x", card_set_id=90000, is_collectible=False),
                90051130: _card(90051130, name="x", card_set_id=90000, is_collectible=False),
                90071220: _card(90071220, name="x", card_type="\u62a4\u7b26", attack=None, life=None, card_set_id=90000, is_collectible=False),
                90021110: _card(90021110, name="x", cost=0, card_set_id=90000, is_collectible=False),
            }),
        )

    def test_10052110_lw_adds_bat(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10052110, cost=1, attack=1, life=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(engine.current_player))
        bats = sum(1 for h in engine.players[0].hand if hasattr(h, 'card_id') and h.card_id == 90051120)
        self.assertEqual(bats, 1)

    def test_10112120_lw_adds_fairy(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10112120, cost=1, attack=1, life=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(engine.current_player))
        fairies = sum(1 for h in engine.players[0].hand if hasattr(h, 'card_id') and h.card_id == 90011110)
        self.assertEqual(fairies, 1)

    def test_10251120_lw_adds_ghost(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10251120, cost=1, attack=1, life=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(engine.current_player))
        ghosts = sum(1 for h in engine.players[0].hand if hasattr(h, 'card_id') and h.card_id == 90051130)
        self.assertEqual(ghosts, 1)

    def test_10171110_lw_adds_past_core_as_amulet(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10171110, cost=1, attack=1, life=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(engine.current_player))
        cores = [
            h for h in engine.players[0].hand
            if hasattr(h, 'card_id') and h.card_id == 90071220
        ]
        self.assertEqual(len(cores), 1)
        self.assertEqual(cores[0].definition.card_type, "\u62a4\u7b26")

    def test_10601110_lw_damages_enemy_leader(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10601110, cost=2, attack=2, life=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(engine.current_player))
        self.assertEqual(engine.players[1].health, 19)

    def test_10651110_lw_draws_and_self_damages(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10651110, cost=2, attack=2, life=2))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        unit.health = 0
        deck_before = len(engine.players[0].deck)
        engine.apply(EndTurn(engine.current_player))
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)
        self.assertEqual(engine.players[0].health, 19)

    def test_10571310_draws_one(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10571310, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        deck_before = len(engine.players[0].deck)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_10022110_lw_summons_knight(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10022110, cost=2, attack=1, life=2))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        unit = engine.players[0].board[0]
        unit.health = 0
        engine.apply(EndTurn(engine.current_player))
        knights = sum(
            1 for u in engine.players[0].board
            if hasattr(u, 'definition') and u.definition.card_id == 90021110
        )
        self.assertEqual(knights, 1)


class DeterminismBatch2Tests(unittest.TestCase):
    def test_same_seed_same_result(self):
        rb = RuleBook.from_directory("data/rules")
        for _ in range(2):
            engine = GameEngine(
                deck_a=[_card(i) for i in range(1000, 1040)],
                deck_b=[_card(i) for i in range(2000, 2040)],
                class_a=1, class_b=1, seed=42, rulebook=rb,
            )
            engine.reset(seed=42)
            _insert_card(engine, _card(10571310, card_type="\u6cd5\u672f", cost=1))
            engine.players[0].mana = 10
            deck_before = len(engine.players[0].deck)
            engine.apply(PlayCard(0, 0))
            self.assertEqual(len(engine.players[0].deck), deck_before - 1)


class RulesLoadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rb = RuleBook.from_directory("data/rules")

    def test_all_new_cards_have_rules(self):
        for cid in (
            10031310, 10041310, 10041110, 10052310, 10111310, 10642310,
            10052110, 10112120, 10251120, 10171110, 10601110, 10651110,
            10571310, 10022110,
        ):
            ops_play = self.rb.operations_for(cid, Trigger.PLAY)
            ops_fanfare = self.rb.operations_for(cid, Trigger.FANFARE)
            ops_last_words = self.rb.operations_for(cid, Trigger.LAST_WORDS)
            self.assertTrue(
                len(ops_play) > 0 or len(ops_fanfare) > 0 or len(ops_last_words) > 0,
                f"Card {cid} has no rules in RuleBook",
            )

    def test_partial_rule_is_reported_as_partial(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10041310"]
        self.assertEqual(info["coverage"], "covered_partial")
        self.assertIn("觉醒", info["rule_metadata"]["unsupported_text"])

    def test_generated_unplayable_token_rule_is_reported_as_partial(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10171110"]
        self.assertEqual(info["coverage"], "covered_partial")
        self.assertIn("无法使用", info["rule_metadata"]["unsupported_text"])


class BehaviorTests(unittest.TestCase):
    def setUp(self):
        self.rb = RuleBook.from_directory("data/rules")

    def test_10031310_draws_one(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        _insert_card(engine, _card(10031310, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        deck_before = len(engine.players[0].deck)
        engine.apply(PlayCard(0, 0))
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_10041310_damages_enemy_follower(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        target = Unit.summon(_card(999, attack=1, life=3), entity_id=999)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10041310, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        cmds = engine.legal_commands()
        choose_cmds = [c for c in cmds if isinstance(c, Choose)]
        self.assertTrue(len(choose_cmds) > 0)
        engine.apply(choose_cmds[0])
        self.assertEqual(target.health, 1)

    def test_10041310_no_target_unplayable(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        _insert_card(engine, _card(10041310, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

    def test_10041110_fanfare_damages_enemy(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        target = Unit.summon(_card(999, attack=1, life=2), entity_id=999)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10041110, cost=2, attack=2, life=2))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        cmds = engine.legal_commands()
        choose_cmds = [c for c in cmds if isinstance(c, Choose)]
        self.assertTrue(len(choose_cmds) > 0)
        engine.apply(choose_cmds[0])
        self.assertEqual(target.health, 1)

    def test_10052310_destroys_own_and_draws(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        sac = Unit.summon(_card(500, attack=1, life=1), entity_id=500)
        engine.players[0].board.append(sac)
        _insert_card(engine, _card(10052310, card_type="\u6cd5\u672f", cost=2))
        engine.players[0].mana = 10
        deck_before = len(engine.players[0].deck)
        engine.apply(PlayCard(0, 0))
        cmds = engine.legal_commands()
        choose_cmds = [c for c in cmds if isinstance(c, Choose)]
        self.assertTrue(len(choose_cmds) > 0)
        engine.apply(choose_cmds[0])
        self.assertEqual(len(engine.players[0].board), 0)
        self.assertEqual(len(engine.players[0].deck), deck_before - 2)

    def test_10642310_playable_when_other_card_in_hand(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        target = Unit.summon(_card(999, attack=1, life=1), entity_id=999)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10642310, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        cmds = engine.legal_commands()
        choose_cmds = [c for c in cmds if isinstance(c, Choose)]
        self.assertTrue(len(choose_cmds) > 0)
        engine.apply(choose_cmds[0])
        self.assertIsNotNone(engine.state.pending_choice)
        cmds2 = engine.legal_commands()
        choose_cmds2 = [c for c in cmds2 if isinstance(c, Choose)]
        self.assertTrue(len(choose_cmds2) > 0)
        engine.apply(choose_cmds2[0])
        self.assertEqual(len(engine.players[1].board), 0)

    def test_10111310_adds_two_fairies(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        _insert_card(engine, _card(10111310, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        fairies = sum(
            1 for h in engine.players[0].hand
            if hasattr(h, 'card_id') and h.card_id == 90011110
        )
        self.assertEqual(fairies, 2)


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_result(self):
        rb = RuleBook.from_directory("data/rules")
        for _ in range(2):
            engine = _make_engine(rb)
            engine.reset(seed=42)
            _insert_card(engine, _card(10031310, card_type="\u6cd5\u672f", cost=1))
            engine.players[0].mana = 10
            deck_before = len(engine.players[0].deck)
            engine.apply(PlayCard(0, 0))
            self.assertEqual(len(engine.players[0].deck), deck_before - 1)


if __name__ == "__main__":
    unittest.main()
