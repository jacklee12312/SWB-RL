# -*- coding: utf-8 -*-
"""Tests for batch of real card rules."""

from __future__ import annotations

import copy
import os
import sqlite3
import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Attack, Choose, EndTurn, PlayCard, SuperEvolve
from swb.engine.emblem import EmblemDefinition, EmblemTriggerRule, EventScope
from swb.engine.environment import ShadowverseEnv
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import DamageType, GameEngine, IllegalCommand
from swb.engine.state import AttackRestriction, Amulet, DeathCause, DestroyedFollowerRecord, HandCard, Unit


def _card(cid, **kw):
    return CardDefinition(
        card_id=cid, card_set_id=kw.get("card_set_id", 10000),
        class_id=kw.get("class_id", 1), class_name=kw.get("class_name", "\u7cbe\u7075"),
        name=kw.get("name", f"c{cid}"), cost=kw.get("cost", 1),
        card_type=kw.get("card_type", "\u968f\u4ece"),
        attack=kw.get("attack", 1), life=kw.get("life", 1),
        keywords=kw.get("keywords", frozenset()), support_level="basic",
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
            90051110: _card(90051110, name="\u9ab8\u9aa8\u58eb\u5175", cost=0, card_set_id=90000, is_collectible=False),
            90051140: _card(90051140, name="\u8150\u81ed\u7684\u50f5\u5c38", cost=3, attack=2, life=2, card_set_id=90000, is_collectible=False),
            90071120: _card(90071120, name="\u6539\u826f\u578b\u00b7\u60ac\u4e1d\u5080\u5121", cost=1, attack=3, life=3, card_set_id=90000, is_collectible=False),
            90071210: _card(90071210, name="\u672a\u6765\u6838\u5fc3", card_type="\u62a4\u7b26", attack=None, life=None, card_set_id=90000, is_collectible=False),
            90071220: _card(90071220, name="\u8fc7\u5f80\u6838\u5fc3", card_type="\u62a4\u7b26", attack=None, life=None, card_set_id=90000, is_collectible=False),
            90021110: _card(90021110, name="\u9a91\u58eb", cost=0, card_set_id=90000, is_collectible=False),
            90021120: _card(90021120, name="\u94c1\u7532\u9a91\u58eb", cost=1, attack=2, life=2, card_set_id=90000, is_collectible=False),
            90031110: _card(90031110, name="\u6ce5\u5c18\u5de8\u50cf", cost=1, attack=2, life=2, card_set_id=90000, is_collectible=False),
            10631110: _card(10631110, name="\u5929\u6676\u9b54\u624b", class_id=3, class_name="\u5deb\u5e08", cost=1, attack=1, life=1),
        }),
    )


def _engine_snapshot(engine):
    return (
        copy.deepcopy(engine.state),
        tuple(engine.logs),
        tuple(engine.event_history),
        tuple(engine.placeholder_ability_events),
        engine.random.getstate(),
        dict(engine._death_causes),
        copy.deepcopy(engine._suspended_batch),
        copy.deepcopy(engine._suspended_record),
        copy.deepcopy(engine._suspended_lw_records),
        engine._suspended_action,
        copy.deepcopy(engine._suspended_action_state),
        copy.deepcopy(engine._suspended_event_state),
        engine._spellboost_pending,
        engine._pending_spellboost_player,
        engine._pending_spellboost_source_card_id,
        engine._pending_spellboost_source_entity_id,
        copy.deepcopy(engine._emblem_batches),
        engine._next_emblem_batch_id,
        copy.deepcopy(engine._emblem_expiration_batches),
        engine._next_emblem_expiration_batch_id,
        engine._stabilizing,
        engine._next_modifier_id,
        engine._next_choice_request_id,
    )


def _insert_card(engine, card_def, origin=CardOrigin.DECK):
    hc = HandCard(definition=card_def, entity_id=engine.state.allocate_entity_id(), origin=origin)
    engine.players[0].hand.insert(0, hc)
    engine.players[0].hand_entity_ids.insert(0, hc.entity_id)
    return hc


def _choose_hand_entity(engine, entity_id):
    for command in engine.legal_commands():
        if isinstance(command, Choose) and command.option_id == f"hand:{entity_id}":
            return command
    raise AssertionError(f"choice for hand entity {entity_id} not found")


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

    def test_10351120_exists(self):
        card = self.repo.get(10351120)
        self.assertEqual(card.name, "泡沫鬼姬")
        self.assertEqual(card.cost, 8)
        text = "\n".join(self._skill_texts(10351120))
        self.assertIn("选择对手的战场上的2个随从", text)
        self.assertIn("对自己的主战者造成4点伤害", text)

    def test_10474120_exists(self):
        card = self.repo.get(10474120)
        self.assertEqual(card.name, "唯一王者·别西卜")
        text = "\n".join(self._skill_texts(10474120))
        self.assertIn("选择对手的战场上的2个随从", text)
        self.assertIn("使其失去所有能力", text)
        self.assertIn("对其造成9点伤害", text)

    def test_10051310_exists(self):
        card = self.repo.get(10051310)
        self.assertEqual(card.name, "\u6df7\u6c8c\u8bc5\u5492")
        self.assertEqual(card.cost, 2)
        self.assertEqual(card.card_type, "\u6cd5\u672f")
        text = "\n".join(self._skill_texts(10051310))
        self.assertIn("\u6a21\u5f0f", text)
        self.assertIn("\u62bd\u53d61\u5f20\u968f\u4ece", text)
        self.assertIn("\u4ea1\u8005\u53ec\u8fd8", text)
        self.assertIn("_2", text)

    def test_10041110_exists(self):
        card = self.repo.get(10041110)
        self.assertIn("\u706b\u8725\u8734", card.name)
        self.assertEqual(card.cost, 2)
        self.assertIn("入场曲", "\n".join(self._skill_texts(10041110)))

    def test_10713110_exists(self):
        card = self.repo.get(10713110)
        self.assertEqual(card.name, "\u51b0\u7bad\u5c04\u624b")
        self.assertEqual(card.cost, 3)
        text = "\n".join(self._skill_texts(10713110))
        self.assertIn("\u5165\u573a\u66f2", text)
        self.assertIn("\u672c\u968f\u4ece\u7684\u653b\u51fb\u529b", text)
        self.assertIn("\u8fde\u51fb", text)

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


class DatabaseVerificationBatch3Tests(unittest.TestCase):
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

    def test_batch3_database_texts_match_supported_rules(self):
        expected = {
            10012310: ("\u6606\u866b\u7684\u5fe0\u544a", ["\u8fd4\u56de\u624b\u724c", "\u968f\u673a1\u4e2a\u968f\u4ece", "2\u70b9\u4f24\u5bb3"]),
            10151310: ("\u6b7b\u795e\u6325\u5200", ["\u81ea\u5df1\u7684\u6218\u573a", "\u5bf9\u624b\u7684\u6218\u573a", "\u7834\u574f"]),
            10171320: ("\u521b\u9020\u7269\u5145\u80fd", ["\u672a\u6765\u6838\u5fc3", "\u8fc7\u5f80\u6838\u5fc3"]),
            10031320: ("\u53ec\u5524\u771f\u7406", ["\u53ec\u55241\u4e2a", "\u6ce5\u5c18\u5de8\u50cf"]),
            10172310: ("\u751f\u547d\u7684\u5954\u6d41", ["3\u70b9\u4f24\u5bb3", "\u8fc7\u5f80\u6838\u5fc3"]),
            10221310: ("\u5546\u8c08\u6210\u7acb", ["\u62bd\u53d62\u5f20", "\u5bf9\u624b\u62bd\u53d61\u5f20"]),
            10252310: ("\u4f7f\u5524\u8759\u8760", ["\u53ec\u55242\u4e2a", "\u8759\u8760"]),
            10442310: ("\u81f3\u7231\u72c2\u8f70", ["3\u70b9\u4f24\u5bb3", "\u65e0\u6cd5\u653b\u51fb"]),
            10021310: ("\u5973\u4ec6\u7684\u793c\u4eea", ["\u8fd4\u56de\u724c\u7ec4", "\u62bd\u53d62\u5f20\u7687\u5bb6\u62a4\u536b\u00b7\u968f\u4ece"]),
            10711310: ("\u4eba\u683c\u5207\u6362", ["\u9009\u62e9\u81ea\u5df1\u76842\u5f20\u624b\u724c", "\u8fd4\u56de\u724c\u7ec4", "\u62bd\u53d62\u5f20"]),
            10661310: ("\u5929\u4e66\u6388\u4e88", ["\u62bd\u53d62\u5f20\u62a4\u7b26"]),
            10231120: ("\u9b54\u5bfc\u56fe\u4e66\u7ba1\u7406\u5458", ["\u8fd4\u56de\u724c\u7ec4", "\u62bd\u53d61\u5f20\u6cd5\u672f"]),
            10632310: ("\u6b63\u5e38\u7684\u4fb5\u8680", ["\u5929\u6676\u9b54\u624b", "\u7834\u574f\u8be5\u968f\u4ece", "\u62bd\u53d62\u5f20\u5deb\u5e08\u00b7\u968f\u4ece"]),
            10411310: ("\u5f57\u661f", ["4\u70b9\u4f24\u5bb3", "\u8fdb\u5316\u540e\u7684\u968f\u4ece", "\u62bd\u53d61\u5f20\u5361\u724c"]),
            10671310: ("\u5929\u65a7\u6388\u4e88", ["4\u70b9\u4f24\u5bb3", "\u539f\u59cb\u8d39\u7528\u4e3a5\u6216\u4ee5\u4e0a", "\u62bd\u53d61\u5f20\u5361\u724c"]),
        }
        for card_id, (name_part, substrings) in expected.items():
            with self.subTest(card_id=card_id):
                card = self.repo.get(card_id)
                self.assertIn(name_part, card.name)
                text = "\n".join(self._skill_texts(card_id))
                for substring in substrings:
                    self.assertIn(substring, text)


class DatabaseVerificationBatch4Tests(unittest.TestCase):
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

    def test_batch4_database_texts_match_supported_rules(self):
        expected = {
            10251310: ("\u8bc5\u5492\u6d3e\u5bf9", ["\u6028\u7075", "\u9ab8\u9aa8\u58eb\u5175", "\u8150\u81ed\u7684\u50f5\u5c38", "\u52a0\u5165\u624b\u724c"]),
            10531310: ("\u660e\u8d8a\u82b1\u7684\u8f6c\u53d8", ["\u820d\u5f03\u8be5\u624b\u724c", "\u62bd\u53d62\u5f20\u5361\u724c"]),
            10521310: ("\u4e3d\u91d1\u82b1\u7684\u6325\u970d", ["\u624b\u724c\u4e2d\u76841\u5f20\u6cd5\u672f", "\u968f\u673a1\u4e2a\u968f\u4ece", "3\u70b9\u4f24\u5bb3"]),
            10631310: ("\u5929\u6676\u6388\u4e88", ["\u53ec\u55242\u4e2a", "\u5929\u6676\u9b54\u624b"]),
            10171310: ("\u4eba\u5076\u66ff\u8eab", ["\u53ec\u55242\u4e2a", "\u6539\u826f\u578b\u00b7\u60ac\u4e1d\u5080\u5121"]),
            10472310: ("\u8eab\u65e0\u957f\u7269\u552f\u6709\u77f3", ["\u53d1\u52a86\u6b21", "\u968f\u673a1\u4e2a\u968f\u4ece", "1\u70b9\u4f24\u5bb3"]),
            10153310: ("\u86c7\u795e\u4e4b\u6012", ["\u968f\u4ece\u6216\u5bf9\u624b\u7684\u4e3b\u6218\u8005", "3\u70b9\u4f24\u5bb3", "\u81ea\u5df1\u7684\u4e3b\u6218\u8005", "2\u70b9\u4f24\u5bb3"]),
            10121310: ("\u5251\u58eb\u7684\u65a9\u51fb", ["\u7834\u574f\u8be5\u968f\u4ece", "\u53ec\u55241\u4e2a", "\u94c1\u7532\u9a91\u58eb"]),
            10301310: ("\u81f3\u9ad8\u7684\u51cc\u9a7e", ["\u62bd\u53d63\u5f20\u5361\u724c", "\u81ea\u5df1\u7684\u724c\u7ec4\u4e2d\u6ca1\u6709\u91cd\u590d\u5361\u724c", "\u56de\u590d\u81ea\u5df13\u70b9\u80fd\u91cf\u70b9"]),
        }
        for card_id, (name_part, substrings) in expected.items():
            with self.subTest(card_id=card_id):
                card = self.repo.get(card_id)
                self.assertIn(name_part, card.name)
                text = "\n".join(self._skill_texts(card_id))
                for substring in substrings:
                    self.assertIn(substring, text)


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
                90071210: _card(90071210, name="x", card_type="\u62a4\u7b26", attack=None, life=None, card_set_id=90000, is_collectible=False),
                90071220: _card(90071220, name="x", card_type="\u62a4\u7b26", attack=None, life=None, card_set_id=90000, is_collectible=False),
                90021110: _card(90021110, name="x", cost=0, card_set_id=90000, is_collectible=False),
                90031110: _card(90031110, name="x", cost=1, attack=2, life=2, card_set_id=90000, is_collectible=False),
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

    def test_10012120_super_evolve_returns_enemy_unit_from_json(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        source = Unit.summon(
            _card(
                10012120,
                cost=3,
                attack=2,
                life=3,
                keywords=frozenset({"超进化时"}),
            ),
            entity_id=engine.state.allocate_entity_id(),
        )
        target = Unit.summon(
            _card(801, cost=2, attack=2, life=2),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board = [source]
        engine.players[1].board = [target]
        engine.players[0].turns_started = (
            engine.config.first_player_super_evolution_unlock_turn
        )

        engine.apply(SuperEvolve(0, source.entity_id))
        choices = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertEqual([c.option_id for c in choices], [f"entity:{target.entity_id}"])
        engine.apply(choices[0])

        self.assertTrue(source.super_evolved)
        self.assertNotIn(target, engine.players[1].board)
        self.assertTrue(any(h.card_id == 801 for h in engine.players[1].hand))

    def test_10012120_super_evolve_prevents_combat_damage(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        source = Unit.summon(
            _card(
                10012120,
                cost=3,
                attack=2,
                life=5,
                keywords=frozenset({"超进化时"}),
            ),
            entity_id=engine.state.allocate_entity_id(),
        )
        returned = Unit.summon(
            _card(801, cost=2, attack=1, life=2),
            entity_id=engine.state.allocate_entity_id(),
        )
        defender = Unit.summon(
            _card(802, cost=2, attack=3, life=7),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board = [source]
        engine.players[1].board = [returned, defender]
        engine.players[0].turns_started = (
            engine.config.first_player_super_evolution_unlock_turn
        )

        engine.apply(SuperEvolve(0, source.entity_id))
        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{returned.entity_id}"
        )
        engine.apply(choice)
        self.assertTrue(source.super_evolved)
        self.assertEqual(source.health, source.max_health)

        engine.apply(Attack(0, source.entity_id, defender.entity_id))

        self.assertEqual(source.health, source.max_health)
        self.assertTrue(any(
            event.type is EventType.DAMAGE_PREVENTED
            and event.target_id == source.entity_id
            and event.metadata.get("damage_type") == "combat"
            for event in engine.event_history
        ))

    def test_10012120_super_evolve_protection_persists_on_later_own_turn(self):
        rulebook = RuleBook((
            CardRule(
                card_id=10012120,
                trigger=Trigger.SUPER_EVOLVE,
                operations=self.rb.operations_for(10012120, Trigger.SUPER_EVOLVE),
            ),
            CardRule(
                card_id=99000002,
                trigger=Trigger.PLAY,
                operations=(
                    EffectOperation(
                        kind=EffectKind.DAMAGE_UNIT,
                        target=TargetKind.OWN_UNIT,
                        amount=3,
                    ),
                ),
            ),
        ))
        engine = _make_engine(rulebook)
        engine.reset(seed=42)
        source = Unit.summon(
            _card(
                10012120,
                cost=3,
                attack=2,
                life=5,
                keywords=frozenset({"超进化时"}),
            ),
            entity_id=engine.state.allocate_entity_id(),
        )
        returned = Unit.summon(
            _card(801, cost=2, attack=1, life=2),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].board = [source]
        engine.players[1].board = [returned]
        engine.players[0].turns_started = (
            engine.config.first_player_super_evolution_unlock_turn
        )

        engine.apply(SuperEvolve(0, source.entity_id))
        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
        )
        engine.apply(choice)
        self.assertTrue(source.super_evolved)
        self.assertEqual(source.super_evolved_turn, engine.turn)
        self.assertEqual(source.health, source.max_health)

        engine.apply(EndTurn(0))
        engine.apply(EndTurn(1))
        self.assertGreater(engine.turn, source.super_evolved_turn)
        _insert_card(
            engine,
            _card(99000002, card_type="法术", attack=None, life=None),
        )
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        damage_choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{source.entity_id}"
        )
        engine.apply(damage_choice)

        self.assertEqual(source.health, source.max_health)
        self.assertTrue(any(
            event.type is EventType.DAMAGE_PREVENTED
            and event.target_id == source.entity_id
            for event in engine.event_history
        ))
        self.assertFalse(any(
            event.type is EventType.DAMAGE_APPLIED
            and event.target_id == source.entity_id
            and event.metadata.get("damage_type") == "effect"
            for event in engine.event_history
        ))

    def test_real_lw_cards_share_one_simultaneous_death_batch(self):
        rulebook = RuleBook((
            CardRule(
                card_id=10052110,
                trigger=Trigger.LAST_WORDS,
                operations=self.rb.operations_for(10052110, Trigger.LAST_WORDS),
            ),
            CardRule(
                card_id=10112120,
                trigger=Trigger.LAST_WORDS,
                operations=self.rb.operations_for(10112120, Trigger.LAST_WORDS),
            ),
            CardRule(card_id=99000001, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = _make_engine(rulebook)
        engine.reset(seed=42)
        engine.players[1].board = [
            Unit.summon(_card(10052110, cost=1, attack=1, life=1)),
            Unit.summon(_card(10112120, cost=1, attack=1, life=1)),
        ]
        engine.players[0].mana = 10
        _insert_card(
            engine,
            _card(99000001, card_type="\u6cd5\u672f", cost=1, attack=None, life=None),
        )

        transition = engine.apply(PlayCard(0, 0))

        first_batch = engine.state.death_queue[0]
        self.assertEqual([record.card_id for record in first_batch.records], [10052110, 10112120])
        opponent_hand_ids = [h.card_id for h in engine.players[1].hand]
        self.assertIn(90051120, opponent_hand_ids)
        self.assertIn(90011110, opponent_hand_ids)

        events = transition.events
        batch_start = next(
            event for event in events
            if event.type == EventType.DEATH_BATCH_START
            and event.metadata["batch_id"] == 1
        )
        self.assertEqual(batch_start.metadata["active_player"], 0)
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
                (0, 1, 10052110, 0),
                (1, 1, 10112120, 1),
            ],
        )
        destroyed_indices = [
            i for i, event in enumerate(events)
            if event.type == EventType.FOLLOWER_DESTROYED
        ]
        lw_indices = [
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
        ]
        lw_events = [
            event for event in events
            if event.type == EventType.LAST_WORDS_START
        ]
        batch_end = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_END
        )
        self.assertEqual(len(destroyed_indices), 2)
        self.assertEqual(len(lw_indices), 2)
        self.assertEqual([event.metadata["batch_order_index"] for event in lw_events], [0, 1])
        self.assertTrue(all(event.metadata["batch_record_count"] == 2 for event in lw_events))
        self.assertTrue(all(event.metadata["active_player"] == 0 for event in lw_events))
        self.assertLess(max(destroyed_indices), min(lw_indices))
        self.assertLess(max(lw_indices), batch_end)

    def test_real_death_batch_end_emblem_defers_real_lw_to_next_batch(self):
        rulebook = RuleBook((
            CardRule(
                card_id=10052110,
                trigger=Trigger.LAST_WORDS,
                operations=self.rb.operations_for(10052110, Trigger.LAST_WORDS),
            ),
            CardRule(
                card_id=10022110,
                trigger=Trigger.LAST_WORDS,
                operations=self.rb.operations_for(10022110, Trigger.LAST_WORDS),
            ),
            CardRule(card_id=99000007, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = _make_engine(rulebook)
        engine.reset(seed=42)
        delayed_real_lw = Unit.summon(
            _card(10022110, cost=1, attack=1, life=1),
            entity_id=10022110,
        )
        engine.players[0].board = [delayed_real_lw]
        engine.players[1].board = [
            Unit.summon(_card(10052110, cost=1, attack=1, life=1), entity_id=10052110),
        ]
        emblem = EmblemDefinition(
            "real_batch_end_cleanup",
            999967,
            triggers=(
                EmblemTriggerRule(
                    "death_batch_end",
                    event_scope=EventScope.ANY_EVENT,
                    max_activations=1,
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
        _insert_card(
            engine,
            _card(99000007, card_type="\u6cd5\u672f", cost=1, attack=None, life=None),
        )

        transition = engine.apply(PlayCard(0, 0))

        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[10052110], [10022110]],
        )
        self.assertIn(90051120, [h.card_id for h in engine.players[1].hand])
        self.assertTrue(
            any(
                isinstance(entity, Unit)
                and entity.definition.card_id == 90021110
                for entity in engine.players[0].board
            )
        )

        events = transition.events
        real_lw_batch_1 = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 10052110
        )
        batch_1_end = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_END
            and event.metadata["batch_id"] == 1
        )
        trigger = next(
            i for i, event in enumerate(events)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "real_batch_end_cleanup"
        )
        trigger_event = events[trigger]
        batch_2_start = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_START
            and event.metadata["batch_id"] == 2
        )
        real_lw_batch_2 = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 10022110
        )
        self.assertLess(real_lw_batch_1, batch_1_end)
        self.assertLess(batch_1_end, trigger)
        self.assertLess(trigger, batch_2_start)
        self.assertLess(batch_2_start, real_lw_batch_2)
        self.assertEqual(trigger_event.metadata["trigger"], "death_batch_end")
        self.assertEqual(trigger_event.metadata["trigger_batch_id"], 1)
        self.assertEqual(trigger_event.metadata["trigger_batch_record_count"], 1)
        diagnostics = engine._loop_diagnostics()
        real_triggers = [
            event for event in diagnostics["recent_emblem_triggers"]
            if event["metadata"]["emblem_id"] == "real_batch_end_cleanup"
        ]
        self.assertTrue(real_triggers)
        self.assertEqual(real_triggers[-1]["metadata"]["trigger"], "death_batch_end")
        self.assertEqual(real_triggers[-1]["metadata"]["trigger_batch_id"], 1)

    def test_real_mixed_follower_amulet_cross_player_batch_metadata(self):
        rulebook = RuleBook((
            CardRule(
                card_id=10052110,
                trigger=Trigger.LAST_WORDS,
                operations=self.rb.operations_for(10052110, Trigger.LAST_WORDS),
            ),
            CardRule(
                card_id=10161210,
                trigger=Trigger.COUNTDOWN_EXPIRED,
                operations=self.rb.operations_for(10161210, Trigger.COUNTDOWN_EXPIRED),
            ),
            CardRule(card_id=99000008, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DESTROY, target=TargetKind.ALL_BOARD),
            ),),
        ))
        engine = _make_engine(rulebook)
        engine.reset(seed=42)
        church = Amulet(
            definition=_card(
                10161210,
                card_type="\u62a4\u7b26",
                cost=1,
                attack=None,
                life=None,
            ),
            entity_id=10161210,
        )
        engine.players[0].board = [church]
        engine.players[1].board = [
            Unit.summon(_card(10052110, cost=1, attack=1, life=1), entity_id=10052110),
        ]
        engine.players[0].mana = 10
        deck_before = len(engine.players[0].deck)
        _insert_card(
            engine,
            _card(99000008, card_type="\u6cd5\u672f", cost=1, attack=None, life=None),
        )

        transition = engine.apply(PlayCard(0, 0))

        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[10161210, 10052110]],
        )
        self.assertLessEqual(len(engine.players[0].deck), deck_before - 2)
        self.assertIn(90051120, [h.card_id for h in engine.players[1].hand])

        batch_start = next(
            event for event in transition.events
            if event.type == EventType.DEATH_BATCH_START
            and event.metadata["batch_id"] == 1
        )
        batch_end = next(
            event for event in transition.events
            if event.type == EventType.DEATH_BATCH_END
            and event.metadata["batch_id"] == 1
        )
        for event in (batch_start, batch_end):
            self.assertEqual(event.metadata["follower_count"], 1)
            self.assertEqual(event.metadata["amulet_count"], 1)
            self.assertEqual(
                event.metadata["owner_counts"],
                [
                    {"owner": 0, "record_count": 1, "follower_count": 0, "amulet_count": 1},
                    {"owner": 1, "record_count": 1, "follower_count": 1, "amulet_count": 0},
                ],
            )
        self.assertEqual(
            [
                (record["batch_order_index"], record["owner"], record["card_id"], record["card_type"])
                for record in batch_start.metadata["ordered_records"]
            ],
            [
                (0, 0, 10161210, "\u62a4\u7b26"),
                (1, 1, 10052110, "\u968f\u4ece"),
            ],
        )
        lw_events = [
            event for event in transition.events
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] in {10161210, 10052110}
        ]
        self.assertEqual([event.metadata["batch_follower_count"] for event in lw_events], [1, 1])
        self.assertEqual([event.metadata["batch_amulet_count"] for event in lw_events], [1, 1])

    def test_real_lw_continues_after_choice_lw_kills_new_unit(self):
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ANY_UNIT, amount=2),
            ),),
            CardRule(
                card_id=10052110,
                trigger=Trigger.LAST_WORDS,
                operations=self.rb.operations_for(10052110, Trigger.LAST_WORDS),
            ),
            CardRule(card_id=902, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=2),
            ),),
            CardRule(card_id=99000002, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = _make_engine(rulebook)
        engine.reset(seed=42)
        victim = Unit.summon(_card(902, attack=1, life=2), entity_id=902)
        engine.players[0].board = [victim]
        engine.players[1].board = [
            Unit.summon(_card(900, cost=1, attack=1, life=1)),
            Unit.summon(_card(10052110, cost=1, attack=1, life=1)),
        ]
        engine.players[0].mana = 10
        _insert_card(
            engine,
            _card(99000002, card_type="\u6cd5\u672f", cost=1, attack=None, life=None),
        )

        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.phase.name, "AWAITING_CHOICE")
        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose) and command.option_id == f"entity:{victim.entity_id}"
        )
        transition = engine.apply(choice)

        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900, 10052110], [902]],
        )
        self.assertIn(90051120, [h.card_id for h in engine.players[1].hand])
        self.assertEqual(engine.players[1].health, 18)

        events = transition.events
        real_lw_start = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 10052110
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
        self.assertLess(real_lw_start, batch_1_end)
        self.assertLess(batch_1_end, batch_2_start)

    def test_real_lw_continues_after_stale_choice_lw_target(self):
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.LAST_WORDS, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ANY_UNIT, amount=2),
            ),),
            CardRule(
                card_id=10052110,
                trigger=Trigger.LAST_WORDS,
                operations=self.rb.operations_for(10052110, Trigger.LAST_WORDS),
            ),
            CardRule(card_id=99000003, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = _make_engine(rulebook)
        engine.reset(seed=42)
        stale_target = Unit.summon(_card(903, attack=1, life=2), entity_id=903)
        engine.players[0].board = [stale_target]
        engine.players[1].board = [
            Unit.summon(_card(900, cost=1, attack=1, life=1)),
            Unit.summon(_card(10052110, cost=1, attack=1, life=1)),
        ]
        engine.players[0].mana = 10
        _insert_card(
            engine,
            _card(99000003, card_type="\u6cd5\u672f", cost=1, attack=None, life=None),
        )

        engine.apply(PlayCard(0, 0))
        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose) and command.option_id == f"entity:{stale_target.entity_id}"
        )
        engine.players[0].board.remove(stale_target)
        engine._send_to_graveyard(
            0,
            stale_target.definition,
            "test_stale_real_lw_target",
            source_entity_id=stale_target.entity_id,
        )
        transition = engine.apply(choice)

        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[900, 10052110]],
        )
        self.assertIn(90051120, [h.card_id for h in engine.players[1].hand])
        self.assertTrue(any("已离场，跳过" in log for log in engine.logs))

        events = transition.events
        real_lw_start = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 10052110
        )
        batch_1_end = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_END
            and event.metadata["batch_id"] == 1
        )
        self.assertLess(real_lw_start, batch_1_end)
        self.assertFalse(
            any(event.type == EventType.DEATH_BATCH_START and event.metadata["batch_id"] == 2 for event in events)
        )

    def test_real_lw_waits_for_follower_destroyed_emblem_choice(self):
        rulebook = RuleBook((
            CardRule(
                card_id=10052110,
                trigger=Trigger.LAST_WORDS,
                operations=self.rb.operations_for(10052110, Trigger.LAST_WORDS),
            ),
            CardRule(card_id=99000004, trigger=Trigger.PLAY, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
            ),),
        ))
        engine = _make_engine(rulebook)
        engine.reset(seed=42)
        target = Unit.summon(_card(904, attack=1, life=3), entity_id=904)
        engine.players[0].board = [target]
        engine.players[1].board = [
            Unit.summon(_card(10052110, cost=1, attack=1, life=1), entity_id=10052110),
        ]
        emblem = EmblemDefinition(
            "real_death_choice",
            999961,
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
            "real_death_leader",
            999962,
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
        engine._add_emblem_to_player(1, emblem, emblem.source_card_id)
        engine._add_emblem_to_player(
            1,
            second_emblem,
            second_emblem.source_card_id,
        )
        bat_count = lambda: sum(
            1 for hand_card in engine.players[1].hand
            if hasattr(hand_card, "card_id") and hand_card.card_id == 90051120
        )
        bats_before = bat_count()
        engine.players[0].mana = 10
        _insert_card(
            engine,
            _card(99000004, card_type="\u6cd5\u672f", cost=1, attack=None, life=None),
        )

        first = engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.state.phase.name, "AWAITING_CHOICE")
        self.assertEqual(target.health, 3)
        self.assertEqual(bat_count(), bats_before)
        self.assertNotIn(
            EventType.LAST_WORDS_START,
            [event.type for event in first.events],
        )
        first_trigger = next(
            i for i, event in enumerate(first.events)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "real_death_choice"
        )
        destroyed_event = next(
            i for i, event in enumerate(first.events)
            if event.type == EventType.FOLLOWER_DESTROYED
            and event.metadata["card_id"] == 10052110
        )
        self.assertLess(destroyed_event, first_trigger)

        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        transition = engine.apply(choice)

        self.assertEqual(target.health, 2)
        self.assertEqual(engine.players[0].health, 18)
        self.assertEqual(bat_count(), bats_before + 1)
        self.assertEqual(
            [[record.card_id for record in batch.records] for batch in engine.state.death_queue],
            [[10052110]],
        )

        history = engine.event_history
        real_choice_trigger = next(
            i for i, event in enumerate(history)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "real_death_choice"
        )
        real_choice_damage = next(
            i for i, event in enumerate(history)
            if event.type == EventType.DAMAGE_APPLIED
            and event.target_id == target.entity_id
            and i > real_choice_trigger
        )
        self.assertLess(real_choice_trigger, real_choice_damage)

        events = transition.events
        damage_idx = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DAMAGE_APPLIED
            and event.target_id == target.entity_id
        )
        leader_emblem = next(
            i for i, event in enumerate(events)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "real_death_leader"
        )
        leader_damage = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DAMAGE_APPLIED
            and event.metadata.get("target_player") == 0
            and event.amount == 2
        )
        real_lw_start = next(
            i for i, event in enumerate(events)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 10052110
        )
        batch_end = next(
            i for i, event in enumerate(events)
            if event.type == EventType.DEATH_BATCH_END
            and event.metadata["batch_id"] == 1
        )
        self.assertLess(damage_idx, real_lw_start)
        self.assertLess(leader_emblem, leader_damage)
        self.assertLess(leader_damage, real_lw_start)
        self.assertLess(real_lw_start, batch_end)

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
        self.assertTrue(cores[0].cannot_be_played)

    def test_90071220_cannot_be_played_passive_blocks_core_play(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(
            engine,
            _card(
                90071220,
                name="\u8fc7\u5f80\u6838\u5fc3",
                card_type="\u62a4\u7b26",
                attack=None,
                life=None,
                card_set_id=90000,
                is_collectible=False,
            ),
        )
        engine._ensure_entity_ids()
        engine.players[0].mana = 10
        core = engine.players[0].hand[0]

        self.assertTrue(core.cannot_be_played)
        self.assertFalse(
            any(
                isinstance(command, PlayCard) and command.hand_index == 0
                for command in engine.legal_commands()
            )
        )
        before = (
            len(engine.players[0].hand),
            len(engine.players[0].board),
            engine.players[0].mana,
        )
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))
        after = (
            len(engine.players[0].hand),
            len(engine.players[0].board),
            engine.players[0].mana,
        )
        self.assertEqual(after, before)

    def test_90071220_cannot_be_played_passive_blocks_rl_action_mask(self):
        rb = RuleBook.from_directory("data/rules")
        core = _card(
            90071220,
            name="\u8fc7\u5f80\u6838\u5fc3",
            card_type="\u62a4\u7b26",
            attack=None,
            life=None,
            card_set_id=90000,
            is_collectible=False,
        )
        resolver = _resolver({90071220: core})
        env = ShadowverseEnv(
            [_card(i) for i in range(1000, 1040)],
            [_card(i) for i in range(2000, 2040)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=rb,
            card_resolver=resolver,
        )
        env.reset(seed=42)
        _insert_card(env.core, core)
        env.core._ensure_entity_ids()
        env.core.players[0].mana = 10

        self.assertFalse(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])

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

    def test_real_countdown_amulet_destroyed_emblem_precedes_countdown_lw(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        emblem = EmblemDefinition(
            "real_amulet_destroyed",
            999966,
            triggers=(
                EmblemTriggerRule(
                    "amulet_destroyed",
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
        engine._add_emblem_to_player(0, emblem, emblem.source_card_id)
        _insert_card(
            engine,
            _card(
                10161210,
                card_type="\u62a4\u7b26",
                cost=1,
                attack=None,
                life=None,
            ),
        )
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertTrue(
            any(
                isinstance(entity, Amulet)
                and entity.definition.card_id == 10161210
                and entity.countdown == 3
                for entity in engine.players[0].board
            )
        )
        deck_before_expiry = len(engine.players[0].deck)

        for _ in range(6):
            engine.apply(EndTurn(engine.current_player))

        self.assertFalse(
            any(
                entity.definition.card_id == 10161210
                for entity in engine.players[0].board
            )
        )
        self.assertEqual(engine.players[1].health, 18)
        self.assertLessEqual(len(engine.players[0].deck), deck_before_expiry - 2)

        history = engine.event_history
        destroyed = next(
            i for i, event in enumerate(history)
            if event.type == EventType.AMULET_DESTROYED
            and event.metadata["card_id"] == 10161210
        )
        trigger = next(
            i for i, event in enumerate(history)
            if event.type == EventType.EMBLEM_TRIGGERED
            and event.metadata["emblem_id"] == "real_amulet_destroyed"
        )
        lw_start = next(
            i for i, event in enumerate(history)
            if event.type == EventType.LAST_WORDS_START
            and event.metadata["card_id"] == 10161210
        )
        batch_id = history[destroyed].metadata["batch_id"]
        self.assertEqual(history[lw_start].metadata["batch_id"], batch_id)
        self.assertTrue(
            any(
                event.type == EventType.DEATH_BATCH_START
                and event.metadata["batch_id"] == batch_id
                for event in history
            )
        )
        self.assertTrue(
            any(
                event.type == EventType.DEATH_BATCH_END
                and event.metadata["batch_id"] == batch_id
                for event in history
            )
        )
        draw_after_lw = [
            i for i, event in enumerate(history)
            if i > lw_start
            and event.type == EventType.CARD_DRAWN
            and event.player_index == 0
        ]
        self.assertLess(destroyed, trigger)
        self.assertLess(trigger, lw_start)
        self.assertGreaterEqual(len(draw_after_lw), 2)


class BehaviorBatch3Tests(unittest.TestCase):
    def setUp(self):
        self.rb = RuleBook.from_directory("data/rules")

    def _make_engine(self):
        return _make_engine(self.rb)

    def _play_and_choose_first(self, engine, card_id, *, card_type="\u6cd5\u672f", cost=1):
        _insert_card(engine, _card(card_id, card_type=card_type, cost=cost, attack=None, life=None))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        choose_cmds = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertTrue(choose_cmds)
        engine.apply(choose_cmds[0])

    def test_10012310_returns_own_board_and_damages_random_enemy(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        own = Unit.summon(_card(500, attack=1, life=1), entity_id=500)
        enemy = Unit.summon(_card(600, attack=1, life=5), entity_id=600)
        engine.players[0].board.append(own)
        engine.players[1].board.append(enemy)

        self._play_and_choose_first(engine, 10012310, cost=1)

        self.assertEqual(len(engine.players[0].board), 0)
        self.assertTrue(any(h.card_id == 500 for h in engine.players[0].hand))
        self.assertEqual(enemy.health, 3)

    def test_10012310_no_enemy_target_still_returns_own_board(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        own = Unit.summon(_card(500, attack=1, life=1), entity_id=500)
        engine.players[0].board.append(own)
        _insert_card(engine, _card(10012310, card_type="\u6cd5\u672f", cost=1, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{own.entity_id}"
        )
        engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[0].board, [])
        self.assertTrue(any(h.card_id == 500 for h in engine.players[0].hand))
        self.assertEqual(engine.players[1].health, 20)

    def test_10151310_destroys_one_own_and_one_enemy_follower(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        own = Unit.summon(_card(501, attack=1, life=1), entity_id=501)
        enemy = Unit.summon(_card(601, attack=1, life=1), entity_id=601)
        engine.players[0].board.append(own)
        engine.players[1].board.append(enemy)
        _insert_card(engine, _card(10151310, card_type="\u6cd5\u672f", cost=1, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply([c for c in engine.legal_commands() if isinstance(c, Choose)][0])
        engine.apply([c for c in engine.legal_commands() if isinstance(c, Choose)][0])

        self.assertNotIn(own, engine.players[0].board)
        self.assertNotIn(enemy, engine.players[1].board)

    def test_10171320_adds_future_and_past_cores_as_unplayable_cards(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10171320, card_type="\u6cd5\u672f", cost=1, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        core_ids = {h.card_id for h in engine.players[0].hand}
        self.assertIn(90071210, core_ids)
        self.assertIn(90071220, core_ids)
        cores = [h for h in engine.players[0].hand if h.card_id in (90071210, 90071220)]
        self.assertTrue(all(h.cannot_be_played for h in cores))

    def test_10031320_summons_mud_golem(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10031320, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertTrue(any(u.definition.card_id == 90031110 for u in engine.players[0].board))

    def test_10172310_damages_enemy_unit_and_adds_past_core(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(602, attack=1, life=5), entity_id=602)
        engine.players[1].board.append(target)

        self._play_and_choose_first(engine, 10172310, cost=2)

        self.assertEqual(target.health, 2)
        self.assertTrue(any(h.card_id == 90071220 for h in engine.players[0].hand))

    def test_10221310_draws_two_for_self_and_one_for_opponent(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10221310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10
        p0_before = len(engine.players[0].deck)
        p1_before = len(engine.players[1].deck)

        engine.apply(PlayCard(0, 0))

        self.assertEqual(len(engine.players[0].deck), p0_before - 2)
        self.assertEqual(len(engine.players[1].deck), p1_before - 1)

    def test_10252310_summons_two_bats(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10252310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        bats = [u for u in engine.players[0].board if u.definition.card_id == 90051120]
        self.assertEqual(len(bats), 2)

    def test_10442310_damages_and_restricts_same_target(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(603, attack=3, life=5), entity_id=603)
        engine.players[1].board.append(target)

        self._play_and_choose_first(engine, 10442310, cost=2)

        self.assertEqual(target.health, 2)
        self.assertTrue(
            any(r.restriction is AttackRestriction.CANNOT_ATTACK for r in target.attack_restrictions)
        )
        self.assertFalse(target.can_attack_leader)
        self.assertFalse(target.can_attack_units)

    def test_10442310_previous_target_skips_when_damage_kills_target(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(604, attack=3, life=3), entity_id=604)
        engine.players[1].board.append(target)

        self._play_and_choose_first(engine, 10442310, cost=2)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[1].board, [])
        self.assertTrue(
            any(card.entity_id == target.entity_id for card in engine.players[1].graveyard)
        )

    def test_10021310_returns_hand_and_draws_two_royal_followers(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        returned = HandCard(
            definition=_card(700, class_id=1, class_name="\u7cbe\u7075", card_type="\u6cd5\u672f", attack=None, life=None),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].hand.append(returned)
        engine.players[0].hand_entity_ids.append(returned.entity_id)
        engine.players[0].deck = [
            _card(701, class_id=2, class_name="\u7687\u5bb6\u62a4\u536b", card_type="\u6cd5\u672f", attack=None, life=None),
            _card(702, class_id=1, class_name="\u7cbe\u7075", card_type="\u968f\u4ece"),
            _card(703, class_id=2, class_name="\u7687\u5bb6\u62a4\u536b", card_type="\u968f\u4ece"),
            _card(704, class_id=2, class_name="\u7687\u5bb6\u62a4\u536b", card_type="\u968f\u4ece"),
        ]
        _insert_card(engine, _card(10021310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply(_choose_hand_entity(engine, returned.entity_id))

        hand_ids = {h.card_id for h in engine.players[0].hand}
        self.assertIn(703, hand_ids)
        self.assertIn(704, hand_ids)
        self.assertNotIn(701, hand_ids)
        self.assertNotIn(702, hand_ids)

    def test_10711310_returns_two_hand_cards_then_draws_two(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        first = HandCard(definition=_card(710), entity_id=engine.state.allocate_entity_id())
        second = HandCard(definition=_card(711), entity_id=engine.state.allocate_entity_id())
        engine.players[0].hand.extend([first, second])
        engine.players[0].hand_entity_ids.extend([first.entity_id, second.entity_id])
        engine.players[0].deck = [_card(712), _card(713)]
        _insert_card(engine, _card(10711310, card_type="\u6cd5\u672f", cost=1, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply(_choose_hand_entity(engine, first.entity_id))
        engine.apply(_choose_hand_entity(engine, second.entity_id))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(len(engine.players[0].hand), 7)
        self.assertEqual(len(engine.players[0].deck), 2)

    def test_10661310_draws_two_amulets(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        engine.players[0].deck = [
            _card(720, card_type="\u968f\u4ece"),
            _card(721, card_type="\u62a4\u7b26", attack=None, life=None),
            _card(722, card_type="\u6cd5\u672f", attack=None, life=None),
            _card(723, card_type="\u62a4\u7b26", attack=None, life=None),
        ]
        _insert_card(engine, _card(10661310, card_type="\u6cd5\u672f", cost=3, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        hand_ids = {h.card_id for h in engine.players[0].hand}
        self.assertIn(721, hand_ids)
        self.assertIn(723, hand_ids)
        self.assertNotIn(720, hand_ids)
        self.assertNotIn(722, hand_ids)

    def test_10231120_fanfare_returns_hand_and_draws_spell(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        returned = HandCard(
            definition=_card(730, card_type="\u968f\u4ece"),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].hand.append(returned)
        engine.players[0].hand_entity_ids.append(returned.entity_id)
        engine.players[0].deck = [
            _card(731, card_type="\u968f\u4ece"),
            _card(732, card_type="\u6cd5\u672f", attack=None, life=None),
            _card(733, card_type="\u6cd5\u672f", attack=None, life=None),
        ]
        _insert_card(engine, _card(10231120, card_type="\u968f\u4ece", cost=2))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply(_choose_hand_entity(engine, returned.entity_id))

        hand_ids = {h.card_id for h in engine.players[0].hand}
        self.assertTrue({732, 733} & hand_ids)
        self.assertNotIn(731, hand_ids)

    def test_10632310_only_targets_crystal_hand_and_draws_witch_followers(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        valid = Unit.summon(_card(10631110, name="\u5929\u6676\u9b54\u624b", class_id=3, class_name="\u5deb\u5e08"))
        invalid = Unit.summon(_card(740, name="\u5176\u4ed6\u968f\u4ece", class_id=3, class_name="\u5deb\u5e08"))
        engine.players[0].board = [invalid, valid]
        engine.players[0].deck = [
            _card(741, class_id=3, class_name="\u5deb\u5e08", card_type="\u6cd5\u672f", attack=None, life=None),
            _card(742, class_id=1, class_name="\u7cbe\u7075", card_type="\u968f\u4ece"),
            _card(743, class_id=3, class_name="\u5deb\u5e08", card_type="\u968f\u4ece"),
            _card(744, class_id=3, class_name="\u5deb\u5e08", card_type="\u968f\u4ece"),
        ]
        _insert_card(engine, _card(10632310, card_type="\u6cd5\u672f", cost=1, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        choices = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertEqual([c.option_id for c in choices], [f"entity:{valid.entity_id}"])
        engine.apply(choices[0])

        self.assertIn(invalid, engine.players[0].board)
        self.assertNotIn(valid, engine.players[0].board)
        hand_ids = {h.card_id for h in engine.players[0].hand}
        self.assertIn(743, hand_ids)
        self.assertIn(744, hand_ids)
        self.assertNotIn(741, hand_ids)
        self.assertNotIn(742, hand_ids)

    def test_10411310_damages_and_draws_with_evolved_follower(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        evolved = Unit.summon(_card(750, cost=1), entity_id=750)
        evolved.evolved = True
        target = Unit.summon(_card(751, attack=1, life=5), entity_id=751)
        engine.players[0].board.append(evolved)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10411310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10
        hand_before = len(engine.players[0].hand)

        engine.apply(PlayCard(0, 0))
        choices = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertEqual([c.option_id for c in choices], [f"entity:{target.entity_id}"])
        engine.apply(choices[0])

        self.assertEqual(target.health, 1)
        self.assertEqual(len(engine.players[0].hand), hand_before)

    def test_10411310_no_evolved_follower_skips_draw(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        unevolved = Unit.summon(_card(752, cost=1), entity_id=752)
        target = Unit.summon(_card(753, attack=1, life=5), entity_id=753)
        engine.players[0].board.append(unevolved)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10411310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10
        hand_before = len(engine.players[0].hand)

        engine.apply(PlayCard(0, 0))
        engine.apply([c for c in engine.legal_commands() if isinstance(c, Choose)][0])

        self.assertEqual(target.health, 1)
        self.assertEqual(len(engine.players[0].hand), hand_before - 1)

    def test_10671310_damages_and_draws_with_cost_five_follower(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        expensive = Unit.summon(_card(754, cost=5), entity_id=754)
        target = Unit.summon(_card(755, attack=1, life=5), entity_id=755)
        engine.players[0].board.append(expensive)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10671310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10
        hand_before = len(engine.players[0].hand)

        engine.apply(PlayCard(0, 0))
        choices = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertEqual([c.option_id for c in choices], [f"entity:{target.entity_id}"])
        engine.apply(choices[0])

        self.assertEqual(target.health, 1)
        self.assertEqual(len(engine.players[0].hand), hand_before)

    def test_10671310_no_cost_five_follower_skips_draw(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        cheap = Unit.summon(_card(756, cost=4), entity_id=756)
        target = Unit.summon(_card(757, attack=1, life=5), entity_id=757)
        engine.players[0].board.append(cheap)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10671310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10
        hand_before = len(engine.players[0].hand)

        engine.apply(PlayCard(0, 0))
        engine.apply([c for c in engine.legal_commands() if isinstance(c, Choose)][0])

        self.assertEqual(target.health, 1)
        self.assertEqual(len(engine.players[0].hand), hand_before - 1)


class BehaviorBatch4Tests(unittest.TestCase):
    def setUp(self):
        self.rb = RuleBook.from_directory("data/rules")

    def _make_engine(self):
        return _make_engine(self.rb)

    def test_10251310_adds_three_nightmare_tokens_to_hand(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10251310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        hand_ids = {h.card_id for h in engine.players[0].hand}
        self.assertIn(90051130, hand_ids)
        self.assertIn(90051110, hand_ids)
        self.assertIn(90051140, hand_ids)

    def test_10531310_discards_one_hand_card_then_draws_two(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        discarded = HandCard(definition=_card(760), entity_id=engine.state.allocate_entity_id())
        engine.players[0].hand.append(discarded)
        engine.players[0].hand_entity_ids.append(discarded.entity_id)
        _insert_card(engine, _card(10531310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10
        deck_before = len(engine.players[0].deck)

        engine.apply(PlayCard(0, 0))
        engine.apply(_choose_hand_entity(engine, discarded.entity_id))

        self.assertFalse(any(h.card_id == 760 for h in engine.players[0].hand))
        self.assertEqual(len(engine.players[0].deck), deck_before - 2)
        self.assertTrue(any(g.definition.card_id == 760 for g in engine.players[0].graveyard))

    def test_10531310_stale_hand_choice_skips_discard_then_draws_two(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        discarded = HandCard(
            definition=_card(760),
            entity_id=engine.state.allocate_entity_id(),
        )
        engine.players[0].hand.append(discarded)
        engine.players[0].hand_entity_ids.append(discarded.entity_id)
        _insert_card(engine, _card(10531310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        choice = _choose_hand_entity(engine, discarded.entity_id)
        hand_index = next(
            index
            for index, hand_card in enumerate(engine.players[0].hand)
            if hand_card.entity_id == discarded.entity_id
        )
        removed = engine.players[0].hand.pop(hand_index)
        engine.players[0].hand_entity_ids.pop(hand_index)
        engine._send_to_graveyard(
            0,
            removed.definition,
            "test_hand_target_left_hand",
            source_entity_id=removed.entity_id,
            origin=removed.origin,
            source_origin=removed.source_origin,
        )
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "hand:999999"))
        self.assertEqual(engine.deterministic_fingerprint(), before)
        deck_before = len(engine.players[0].deck)

        transition = engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(len(engine.players[0].deck), deck_before - 2)
        self.assertFalse(
            any(
                getattr(hand_card, "entity_id", None) == discarded.entity_id
                for hand_card in engine.players[0].hand
            )
        )
        self.assertEqual(
            sum(
                graveyard_card.entity_id == discarded.entity_id
                for graveyard_card in engine.players[0].graveyard
            ),
            1,
        )
        self.assertFalse(
            any(
                event.type is EventType.CARD_DISCARDED
                and event.metadata.get("card_id") == discarded.card_id
                for event in transition.events
            )
        )
        self.assertTrue(any("\u5df2\u79bb\u624b\uff0c\u8df3\u8fc7" in log for log in engine.logs))

    def test_10521310_discards_spell_then_deals_random_damage_twice(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        discarded = HandCard(
            definition=_card(761, card_type="\u6cd5\u672f", attack=None, life=None),
            entity_id=engine.state.allocate_entity_id(),
        )
        target = Unit.summon(_card(762, attack=1, life=7), entity_id=762)
        engine.players[0].hand.append(discarded)
        engine.players[0].hand_entity_ids.append(discarded.entity_id)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10521310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply(_choose_hand_entity(engine, discarded.entity_id))

        self.assertEqual(target.health, 1)
        self.assertFalse(any(h.card_id == 761 for h in engine.players[0].hand))

    def test_10631310_summons_two_crystal_hands(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10631310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        summons = [u for u in engine.players[0].board if u.definition.card_id == 10631110]
        self.assertEqual(len(summons), 2)

    def test_10171310_summons_only_until_board_limit(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        engine.players[0].board = [
            Unit.summon(_card(770 + i), entity_id=770 + i)
            for i in range(4)
        ]
        _insert_card(engine, _card(10171310, card_type="\u6cd5\u672f", cost=3, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        puppets = [u for u in engine.players[0].board if u.definition.card_id == 90071120]
        self.assertEqual(len(engine.players[0].board), 5)
        self.assertEqual(len(puppets), 1)

    def test_10472310_deals_one_damage_six_times_to_random_enemy(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(780, attack=1, life=7), entity_id=780)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10472310, card_type="\u6cd5\u672f", cost=3, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(target.health, 1)

    def test_10472310_no_enemy_target_is_unplayable(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10472310, card_type="\u6cd5\u672f", cost=3, attack=None, life=None))
        engine.players[0].mana = 10

        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

    def test_10153310_damages_enemy_unit_then_own_leader(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(781, attack=1, life=5), entity_id=781)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10153310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertEqual(target.health, 2)
        self.assertEqual(engine.players[1].health, 20)
        self.assertEqual(engine.players[0].health, 18)

    def test_10153310_damages_enemy_leader_when_selected(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(782, attack=1, life=5), entity_id=782)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10153310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, "leader:1"))

        self.assertEqual(target.health, 5)
        self.assertEqual(engine.players[1].health, 17)
        self.assertEqual(engine.players[0].health, 18)

    def test_10153310_targets_enemy_leader_without_enemy_units(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        engine.players[1].board = []
        _insert_card(engine, _card(10153310, card_type="\u6cd5\u672f", cost=2, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(
            [option.option_id for option in engine.state.pending_choice.options],
            ["leader:1"],
        )
        engine.apply(Choose(0, "leader:1"))

        self.assertEqual(engine.players[1].health, 17)
        self.assertEqual(engine.players[0].health, 18)

    def test_10121310_destroys_enemy_unit_and_summons_steelclad_knight(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        target = Unit.summon(_card(783, attack=1, life=5), entity_id=783)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10121310, card_type="\u6cd5\u672f", cost=4, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertNotIn(target, engine.players[1].board)
        steelclads = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90021120
        ]
        self.assertEqual(len(steelclads), 1)
        self.assertEqual((steelclads[0].attack, steelclads[0].health), (2, 2))

    def test_10121310_requires_enemy_unit_target(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        _insert_card(engine, _card(10121310, card_type="\u6cd5\u672f", cost=4, attack=None, life=None))
        engine.players[0].mana = 10

        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

    def test_10121310_full_board_still_destroys_but_skips_summon(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        engine.players[0].board = [
            Unit.summon(_card(790 + index), entity_id=790 + index)
            for index in range(5)
        ]
        target = Unit.summon(_card(784, attack=1, life=5), entity_id=784)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10121310, card_type="\u6cd5\u672f", cost=4, attack=None, life=None))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(len(engine.players[0].board), 5)
        self.assertFalse(
            any(unit.definition.card_id == 90021120 for unit in engine.players[0].board)
        )

    def test_10301310_draws_three_and_restores_mana_with_unique_deck(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        engine.players[0].deck = [_card(810 + index) for index in range(5)]
        _insert_card(engine, _card(10301310, card_type="\u6cd5\u672f", cost=4, attack=None, life=None))
        engine.players[0].mana = 10
        engine.players[0].max_mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(len(engine.players[0].deck), 2)
        self.assertEqual(engine.players[0].mana, 9)

    def test_10301310_draws_three_without_restore_for_duplicate_deck(self):
        engine = self._make_engine()
        engine.reset(seed=42)
        engine.players[0].deck = [
            _card(823),
            _card(823),
            _card(820),
            _card(821),
            _card(822),
        ]
        _insert_card(engine, _card(10301310, card_type="\u6cd5\u672f", cost=4, attack=None, life=None))
        engine.players[0].mana = 10
        engine.players[0].max_mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(len(engine.players[0].deck), 2)
        self.assertEqual(engine.players[0].mana, 6)


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
            10571310, 10022110, 10012310, 10151310, 10171320, 10031320,
            10172310, 10221310, 10252310, 10442310, 10021310, 10711310,
            10661310, 10231120, 10632310, 10411310, 10671310,
            10251310, 10531310, 10521310, 10631310, 10171310, 10472310,
            10153310, 10121310, 10301310, 10713110, 10051310,
            10351120,
            10474120,
        ):
            ops_play = self.rb.operations_for(cid, Trigger.PLAY)
            ops_fanfare = self.rb.operations_for(cid, Trigger.FANFARE)
            ops_last_words = self.rb.operations_for(cid, Trigger.LAST_WORDS)
            self.assertTrue(
                len(ops_play) > 0 or len(ops_fanfare) > 0 or len(ops_last_words) > 0,
                f"Card {cid} has no rules in RuleBook",
            )

    def test_overflow_rule_is_reported_as_exact(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10041310"]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertNotIn("rule_metadata", info)

    def test_multi_target_real_rule_is_reported_as_exact(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10351120"]
        self.assertEqual(info["coverage"], "covered_exact")

    def test_multi_target_binding_real_rule_is_reported_as_exact(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10474120"]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertNotIn("unsupported_text", info.get("rule_metadata", {}))

    def test_generated_core_rule_is_reported_as_exact(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10171110"]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertNotIn("unsupported_text", info.get("rule_metadata", {}))

    def test_10713110_rule_covers_combo_turn_end_exactly(self):
        from scripts.report_rule_coverage import _build_coverage_report

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10713110"]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertNotIn("unsupported_text", info["rule_metadata"])


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

    def test_10351120_selects_and_destroys_two_enemy_followers(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        targets = [
            Unit.summon(_card(990 + index, life=5), entity_id=990 + index)
            for index in range(2)
        ]
        engine.players[1].board = targets
        _insert_card(
            engine,
            _card(
                10351120,
                name="泡沫鬼姬",
                cost=8,
                attack=4,
                life=6,
                keywords=frozenset({"疾驰"}),
            ),
        )
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))

        self.assertEqual(engine.state.pending_choice.target_count, 2)
        for target in targets:
            engine.apply(Choose(0, f"entity:{target.entity_id}"))
        self.assertEqual(engine.players[1].board, [])
        self.assertEqual(engine.players[0].health, 16)
        self.assertTrue(
            any(unit.definition.card_id == 10351120 for unit in engine.players[0].board)
        )

    def test_10474120_removes_abilities_damages_targets_and_marks_leader(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        targets = [
            Unit.summon(
                _card(980 + index, life=12, keywords=frozenset({"守护"})),
                entity_id=980 + index,
            )
            for index in range(2)
        ]
        engine.players[1].board = list(targets)
        _insert_card(
            engine,
            _card(
                10474120,
                name="唯一王者·别西卜",
                cost=9,
                attack=9,
                life=9,
            ),
        )
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        for target in reversed(targets):
            engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertEqual([target.health for target in targets], [3, 3])
        self.assertTrue(all(target.printed_abilities_removed for target in targets))
        self.assertTrue(all(not target.has_keyword("守护") for target in targets))
        self.assertEqual(
            [modifier.amount for modifier in engine.players[1].leader_damage_modifiers],
            [1],
        )
        engine.apply_damage(
            None,
            None,
            2,
            DamageType.EFFECT,
            0,
            target_player_index=1,
        )
        self.assertEqual(engine.players[1].health, 17)

    def test_10474120_uses_single_available_target_when_targets_are_short(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        engine.players[1].board = [
            Unit.summon(_card(980, life=12), entity_id=980)
        ]
        _insert_card(
            engine,
            _card(10474120, name="唯一王者·别西卜", cost=9, attack=9, life=9),
        )
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.state.pending_choice.target_count, 1)
        engine.apply(Choose(0, "entity:980"))
        target = engine.players[1].board[0]
        self.assertEqual(target.health, 3)
        self.assertTrue(target.printed_abilities_removed)

    def test_10474120_rl_mask_matches_two_target_legality(self):
        beelzebub = _card(
            10474120,
            name="唯一王者·别西卜",
            cost=9,
            attack=9,
            life=9,
        )
        env = ShadowverseEnv(
            [_card(i) for i in range(1000, 1040)],
            [_card(i) for i in range(2000, 2040)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=self.rb,
        )
        env.reset(seed=42)
        _insert_card(env.core, beelzebub)
        env.players[0].mana = 10
        env.players[1].board = [
            Unit.summon(_card(980), entity_id=980)
        ]
        self.assertTrue(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])
        env.players[1].board.append(
            Unit.summon(_card(981), entity_id=981)
        )
        self.assertTrue(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])

    def test_10474120_uses_remaining_valid_target_if_first_leaves_play(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        first = Unit.summon(
            _card(980, life=12, keywords=frozenset({"守护"})), entity_id=980
        )
        second = Unit.summon(
            _card(981, life=12, keywords=frozenset({"守护"})), entity_id=981
        )
        engine.players[1].board = [first, second]
        _insert_card(
            engine,
            _card(10474120, name="唯一王者·别西卜", cost=9, attack=9, life=9),
        )
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        engine.apply(Choose(0, "entity:980"))
        engine.players[1].board.remove(first)
        before_invalid = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "entity:980"))
        self.assertEqual(engine.deterministic_fingerprint(), before_invalid)
        engine.apply(Choose(0, "entity:981"))

        self.assertEqual(second.health, 3)
        self.assertTrue(second.printed_abilities_removed)
        self.assertEqual(engine.players[1].leader_damage_modifiers[0].amount, 1)

    def test_10041310_target_changed_controller_before_choice_skips_damage(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        target = Unit.summon(_card(998, attack=1, life=5), entity_id=998)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10041310, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        choice = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{target.entity_id}"
        )
        engine.players[1].board.remove(target)
        engine.players[0].board.append(target)

        engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(target.health, 5)
        self.assertEqual(engine.players[0].board, [target])
        self.assertTrue(any("已不再是合法目标，跳过" in log for log in engine.logs))

    def test_10041310_no_target_unplayable(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        _insert_card(engine, _card(10041310, card_type="\u6cd5\u672f", cost=1))
        engine.players[0].mana = 10
        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))

    def test_10051310_mode_draws_follower(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        spell = _card(601, card_type="\u6cd5\u672f", cost=1)
        follower = _card(602, card_type="\u968f\u4ece", cost=2, attack=2, life=2)
        engine.players[0].deck = [spell, follower]
        _insert_card(engine, _card(10051310, card_type="\u6cd5\u672f", cost=2))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == "choose_one:draw_follower"
        )
        engine.apply(choice)

        self.assertIn(
            follower.card_id,
            [card.definition.card_id for card in engine.players[0].hand],
        )
        self.assertEqual([card.card_id for card in engine.players[0].deck], [spell.card_id])

    def test_10051310_mode_reanimates_cost_two_follower(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        dead = _card(603, card_type="\u968f\u4ece", cost=2, attack=2, life=3)
        engine.state.destroyed_followers = [
            DestroyedFollowerRecord(
                definition=dead,
                owner=0,
                death_sequence=1,
                cause=DeathCause.COMBAT,
            )
        ]
        _insert_card(engine, _card(10051310, card_type="\u6cd5\u672f", cost=2))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        choice = next(
            command for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == "choose_one:reanimate_2"
        )
        engine.apply(choice)

        self.assertEqual(len(engine.players[0].board), 1)
        self.assertEqual(engine.players[0].board[0].definition.card_id, dead.card_id)
        self.assertTrue(
            any(event.type is EventType.REANIMATE_RESOLVED for event in engine.event_history)
        )

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

    def test_10713110_fanfare_uses_source_attack(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        target = Unit.summon(_card(997, attack=1, life=5), entity_id=997)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10713110, cost=3, attack=3, life=1))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        choose_cmds = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertTrue(choose_cmds)
        engine.apply(choose_cmds[0])

        self.assertEqual(target.health, 2)

    def test_10713110_source_left_before_choice_skips_damage(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        target = Unit.summon(_card(998, attack=1, life=5), entity_id=998)
        engine.players[1].board.append(target)
        _insert_card(engine, _card(10713110, cost=3, attack=3, life=1))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        source = engine.players[0].board[0]
        engine.players[0].board.remove(source)
        engine._send_to_graveyard(
            0,
            source.definition,
            "test_source_left_play",
            source_entity_id=source.entity_id,
        )
        choose_cmds = [c for c in engine.legal_commands() if isinstance(c, Choose)]
        self.assertTrue(choose_cmds)
        engine.apply(choose_cmds[0])

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(target.health, 5)

    def test_10713110_turn_end_draw_requires_combo_three_and_source_in_play(self):
        for combo, source_present, expected_draws in (
            (2, True, 0),
            (3, True, 1),
            (3, False, 0),
        ):
            with self.subTest(combo=combo, source_present=source_present):
                engine = _make_engine(self.rb)
                engine.reset(seed=42)
                if source_present:
                    engine.players[0].board.append(Unit.summon(
                        _card(10713110, cost=3, attack=3, life=1),
                        entity_id=engine.state.allocate_entity_id(),
                    ))
                engine.players[0].cards_played_this_turn = combo
                deck_before = len(engine.players[0].deck)

                engine.apply(EndTurn(0))

                self.assertEqual(
                    deck_before - len(engine.players[0].deck),
                    expected_draws,
                )

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

    def test_10052310_target_left_before_choice_skips_destroy_and_draws(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        sac = Unit.summon(_card(501, attack=1, life=1), entity_id=501)
        engine.players[0].board.append(sac)
        _insert_card(engine, _card(10052310, card_type="\u6cd5\u672f", cost=2))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        choice = next(
            command
            for command in engine.legal_commands()
            if isinstance(command, Choose)
            and command.option_id == f"entity:{sac.entity_id}"
        )
        engine.players[0].board.remove(sac)
        engine._send_to_graveyard(
            0,
            sac.definition,
            "test_target_left_play",
            source_entity_id=sac.entity_id,
        )
        deck_before = len(engine.players[0].deck)

        engine.apply(choice)

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(engine.players[0].board, [])
        self.assertEqual(len(engine.players[0].deck), deck_before - 2)

    def test_10052310_invalid_choice_option_does_not_mutate_pending_effect(self):
        engine = _make_engine(self.rb)
        engine.reset(seed=42)
        sac = Unit.summon(_card(502, attack=1, life=1), entity_id=502)
        engine.players[0].board.append(sac)
        _insert_card(engine, _card(10052310, card_type="\u6cd5\u672f", cost=2))
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, 0))
        self.assertIsNotNone(engine.state.pending_choice)
        before = _engine_snapshot(engine)

        with self.assertRaisesRegex(IllegalCommand, "Choice option is invalid"):
            engine.apply(Choose(0, "entity:999999"))

        after = _engine_snapshot(engine)
        self.assertEqual(after, before)

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
