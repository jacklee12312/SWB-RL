# -*- coding: utf-8 -*-
"""Tests for batch of real card rules."""

from __future__ import annotations

import os
import sqlite3
import unittest

from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.environment import ShadowverseEnv
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import AttackRestriction, HandCard, Unit


def _card(cid, **kw):
    return CardDefinition(
        card_id=cid, card_set_id=kw.get("card_set_id", 10000),
        class_id=kw.get("class_id", 1), class_name=kw.get("class_name", "\u7cbe\u7075"),
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
            10153310, 10121310,
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
        self.assertIn("融合", info["rule_metadata"]["unsupported_text"])
        self.assertNotIn("无法使用", info["rule_metadata"]["unsupported_text"])


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
