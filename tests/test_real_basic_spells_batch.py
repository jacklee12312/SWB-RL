# -*- coding: utf-8 -*-
"""Direct behavior audits for the basic real-spell coverage batch."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Choose, PlayCard
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import Amulet, HandCard, Unit


BATCH_CARD_IDS = (
    10101310,
    10102310,
    10123310,
    10141310,
    10431310,
    10561310,
    10701310,
    10743310,
    10751310,
    10812310,
    10832310,
    10842310,
)


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 1),
        class_name=overrides.get("class_name", "精灵"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 1),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
        tribe_id=overrides.get("tribe_id", 0),
        tribe_name=overrides.get("tribe_name", ""),
    )


def _spell(card_id: int, *, cost: int = 1) -> CardDefinition:
    return _card(
        card_id,
        cost=cost,
        card_type="法术",
        attack=None,
        life=None,
    )


def _make_engine(rulebook: RuleBook, *, seed: int = 42) -> GameEngine:
    goblin = _card(
        90001110,
        card_set_id=90000,
        class_id=0,
        class_name="中立",
        name="哥布林",
        cost=1,
        attack=1,
        life=2,
        is_collectible=False,
    )
    return GameEngine(
        deck_a=[_card(card_id) for card_id in range(1000, 1040)],
        deck_b=[_card(card_id) for card_id in range(2000, 2040)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
        card_resolver=lambda card_id: goblin if card_id == goblin.card_id else None,
    )


def _insert_hand_card(
    engine: GameEngine,
    definition: CardDefinition,
) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


def _add_unit(
    engine: GameEngine,
    owner: int,
    card_id: int,
    *,
    life: int = 5,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, attack=1, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[owner].board.append(unit)
    return unit


def _play(engine: GameEngine, card_id: int, *, cost: int = 1) -> None:
    _insert_hand_card(engine, _spell(card_id, cost=cost))
    engine.players[0].mana = 10
    engine.apply(PlayCard(0, 0))


class DatabaseClauseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_texts_modes_references_and_token_match_audit(self):
        expected = {
            10101310: ("中立", "法术", 5, "召唤5个『<color=Keyword>哥布林</color>』。"),
            10102310: ("中立", "法术", 3, "抽取2张卡牌。"),
            10123310: (
                "皇家护卫",
                "法术",
                7,
                "选择对手的战场上的1个随从或对手的主战者，对其造成5点伤害。"
                "回复自己的主战者5点生命值。",
            ),
            10141310: ("龙族", "法术", 6, "对战场上的所有随从造成5点伤害。"),
            10431310: (
                "巫师",
                "法术",
                7,
                "对对手的战场上的所有随从造成6点伤害。"
                "回复自己的主战者3点生命值。",
            ),
            10561310: ("主教", "法术", 3, "使自己的随机1张手牌返回牌组。抽取3张卡牌。"),
            10701310: ("中立", "法术", 4, "选择对手的战场上的1张卡牌，使其消失。"),
            10743310: (
                "龙族",
                "法术",
                7,
                "对对手的战场上的所有随从造成5点伤害。"
                "对对手的主战者造成3点伤害。",
            ),
            10751310: ("梦魇", "法术", 1, "选择自己的战场上的2个随从，使其+0/+1。抽取1张卡牌。"),
            10812310: (
                "精灵",
                "法术",
                4,
                "选择对手的战场上的1个随从，破坏该随从。"
                "回复自己的主战者2点生命值。",
            ),
            10832310: ("巫师", "法术", 2, "回复自己的主战者2点生命值。使自己的所有手牌发动1次魔力增幅。"),
            10842310: (
                "龙族",
                "法术",
                6,
                "对对手的战场上的随机1个随从造成4点伤害。"
                "对对手的主战者造成4点伤害。",
            ),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, (class_name, card_type, cost, text) in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT cl.class_name, cl.type_name, c.cost, st.text_chs
                        FROM cards c
                        JOIN card_localizations cl
                          ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                        JOIN skill_texts st ON st.card_id = c.card_id
                        WHERE c.card_id = ?
                        ORDER BY st.position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(row, [(class_name, card_type, cost, text)])
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

            references = connection.execute(
                """
                SELECT card_id, referenced_card_id, referenced_name
                FROM card_references
                WHERE card_id IN ({})
                ORDER BY card_id, position
                """.format(",".join("?" for _ in BATCH_CARD_IDS)),
                BATCH_CARD_IDS,
            ).fetchall()
            self.assertEqual(references, [(10101310, 90001110, "哥布林")])
            token = connection.execute(
                """
                SELECT cl.class_name, cl.type_name, c.cost, c.attack, c.life,
                       cs.is_collectible,
                       (SELECT COUNT(*) FROM skill_texts st WHERE st.card_id = c.card_id)
                FROM cards c
                JOIN card_sets cs ON cs.id = c.card_set_id
                JOIN card_localizations cl
                  ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                WHERE c.card_id = 90001110
                """
            ).fetchone()
            self.assertEqual(token, ("中立", "随从", 1, 1, 2, 0, 0))

    def test_all_batch_cards_have_mapped_exact_clause_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in BATCH_CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_basic_spells_batch.py"],
                )


class RealBasicSpellBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")

    def fresh_engine(self, *, seed: int = 42) -> GameEngine:
        engine = _make_engine(self.rulebook, seed=seed)
        engine.reset(seed=seed)
        return engine

    def test_10101310_summons_five_token_origin_goblins(self):
        engine = self.fresh_engine()
        _play(engine, 10101310, cost=5)

        self.assertEqual(len(engine.players[0].board), 5)
        self.assertEqual(
            [
                (unit.definition.card_id, unit.attack, unit.health)
                for unit in engine.players[0].board
            ],
            [(90001110, 1, 2)] * 5,
        )
        self.assertTrue(
            all(unit.origin is CardOrigin.TOKEN for unit in engine.players[0].board)
        )
        self.assertEqual(len({unit.entity_id for unit in engine.players[0].board}), 5)

    def test_10101310_respects_remaining_board_capacity(self):
        engine = self.fresh_engine()
        existing = [_add_unit(engine, 0, 300 + index) for index in range(3)]
        _play(engine, 10101310, cost=5)

        self.assertEqual(engine.players[0].board[:3], existing)
        self.assertEqual(
            [unit.definition.card_id for unit in engine.players[0].board[3:]],
            [90001110, 90001110],
        )

    def test_10102310_draws_exactly_two_cards(self):
        engine = self.fresh_engine()
        deck_before = len(engine.players[0].deck)
        hand_before = len(engine.players[0].hand)
        _play(engine, 10102310, cost=3)

        self.assertEqual(len(engine.players[0].deck), deck_before - 2)
        self.assertEqual(len(engine.players[0].hand), hand_before + 2)

    def test_10123310_can_hit_follower_or_leader_then_heals(self):
        for target_kind in ("follower", "leader"):
            with self.subTest(target_kind=target_kind):
                engine = self.fresh_engine()
                engine.players[0].health = 10
                target = _add_unit(engine, 1, 401, life=6)
                _play(engine, 10123310, cost=7)
                option_id = (
                    f"entity:{target.entity_id}"
                    if target_kind == "follower"
                    else "leader:1"
                )
                engine.apply(Choose(0, option_id))

                self.assertEqual(engine.players[0].health, 15)
                if target_kind == "follower":
                    self.assertEqual(target.health, 1)
                    self.assertEqual(engine.players[1].health, 20)
                else:
                    self.assertEqual(target.health, 6)
                    self.assertEqual(engine.players[1].health, 15)

    def test_10123310_remains_playable_with_only_the_leader_target(self):
        engine = self.fresh_engine()
        _insert_hand_card(engine, _spell(10123310, cost=7))
        engine.players[0].mana = 10

        self.assertIn(PlayCard(0, 0), engine.legal_commands())
        engine.apply(PlayCard(0, 0))
        self.assertEqual(
            [option.option_id for option in engine.state.pending_choice.options],
            ["leader:1"],
        )

    def test_10141310_damages_both_sides_in_one_death_batch(self):
        engine = self.fresh_engine()
        own = _add_unit(engine, 0, 501, life=5)
        enemy = _add_unit(engine, 1, 502, life=5)
        _play(engine, 10141310, cost=6)

        self.assertNotIn(own, engine.players[0].board)
        self.assertNotIn(enemy, engine.players[1].board)
        destroyed = [
            event
            for event in engine.event_history
            if event.type is EventType.FOLLOWER_DESTROYED
            and event.source_id in {own.entity_id, enemy.entity_id}
        ]
        self.assertEqual(len(destroyed), 2)
        self.assertEqual(len({event.metadata["batch_id"] for event in destroyed}), 1)

    def test_10431310_damages_only_enemy_followers_and_heals(self):
        engine = self.fresh_engine()
        own = _add_unit(engine, 0, 601, life=7)
        enemies = [_add_unit(engine, 1, 610 + index, life=7) for index in range(2)]
        engine.players[0].health = 15
        _play(engine, 10431310, cost=7)

        self.assertEqual(own.health, 7)
        self.assertEqual([unit.health for unit in enemies], [1, 1])
        self.assertEqual(engine.players[0].health, 18)

    def test_10561310_random_return_and_draw_are_seeded(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh_engine(seed=17)
            deck_before = len(engine.players[0].deck)
            hand_before = len(engine.players[0].hand)
            _play(engine, 10561310, cost=3)
            returned = [
                event
                for event in engine.event_history
                if event.type is EventType.CARD_RETURNED_TO_DECK
            ]
            self.assertEqual(len(returned), 1)
            self.assertEqual(len(engine.players[0].deck), deck_before - 2)
            self.assertEqual(len(engine.players[0].hand), hand_before + 2)
            outcomes.append(
                (
                    returned[0].metadata["source"].card_id,
                    engine.deterministic_fingerprint(),
                )
            )
        self.assertEqual(outcomes[0], outcomes[1])

    def test_10561310_skips_random_return_with_no_other_hand_card(self):
        engine = self.fresh_engine(seed=17)
        engine.players[0].hand.clear()
        engine.players[0].hand_entity_ids.clear()
        deck_before = len(engine.players[0].deck)
        _play(engine, 10561310, cost=3)

        self.assertEqual(len(engine.players[0].deck), deck_before - 3)
        self.assertEqual(len(engine.players[0].hand), 3)
        self.assertFalse(
            any(
                event.type is EventType.CARD_RETURNED_TO_DECK
                for event in engine.event_history
            )
        )

    def test_10701310_banishes_a_selected_follower_or_amulet(self):
        for card_type in ("随从", "护符"):
            with self.subTest(card_type=card_type):
                engine = self.fresh_engine()
                if card_type == "随从":
                    target = _add_unit(engine, 1, 701)
                else:
                    target = Amulet(
                        definition=_card(
                            702,
                            card_type="护符",
                            attack=None,
                            life=None,
                        ),
                        entity_id=engine.state.allocate_entity_id(),
                    )
                    engine.players[1].board.append(target)
                _play(engine, 10701310, cost=4)
                engine.apply(Choose(0, f"entity:{target.entity_id}"))

                self.assertNotIn(target, engine.players[1].board)
                self.assertEqual(
                    [card.card_id for card in engine.players[1].banished],
                    [target.definition.card_id],
                )

    def test_10701310_no_target_is_illegal_without_mutation(self):
        engine = self.fresh_engine()
        _insert_hand_card(engine, _spell(10701310, cost=4))
        engine.players[0].mana = 10
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_10743310_damages_all_enemy_followers_and_leader(self):
        engine = self.fresh_engine()
        own = _add_unit(engine, 0, 801, life=6)
        enemies = [_add_unit(engine, 1, 810 + index, life=5) for index in range(2)]
        _play(engine, 10743310, cost=7)

        self.assertEqual(own.health, 6)
        self.assertTrue(all(unit not in engine.players[1].board for unit in enemies))
        self.assertEqual(engine.players[1].health, 17)

    def test_10751310_buffs_two_selected_followers_and_draws(self):
        engine = self.fresh_engine()
        targets = [_add_unit(engine, 0, 901 + index, life=3) for index in range(2)]
        deck_before = len(engine.players[0].deck)
        _play(engine, 10751310)

        self.assertEqual(engine.state.pending_choice.target_count, 2)
        for target in reversed(targets):
            engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertEqual(
            [(unit.attack, unit.health, unit.max_health) for unit in targets],
            [(1, 4, 4), (1, 4, 4)],
        )
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)

    def test_10751310_revalidates_a_selected_follower_that_left_play(self):
        engine = self.fresh_engine()
        first = _add_unit(engine, 0, 911, life=3)
        second = _add_unit(engine, 0, 912, life=3)
        _play(engine, 10751310)
        engine.apply(Choose(0, f"entity:{first.entity_id}"))
        engine.players[0].board.remove(first)
        engine._send_to_graveyard(
            0,
            first.definition,
            "test_target_left_play",
            source_entity_id=first.entity_id,
        )
        engine.apply(Choose(0, f"entity:{second.entity_id}"))

        self.assertEqual(
            (second.attack, second.health, second.max_health),
            (1, 4, 4),
        )
        self.assertTrue(any("已不再合法" in log for log in engine.logs))

    def test_10751310_legality_and_rl_mask_share_target_candidates(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(1000, 1040)],
            [_card(card_id) for card_id in range(2000, 2040)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=self.rulebook,
        )
        env.reset(seed=42)
        _insert_hand_card(env.core, _spell(10751310))
        env.players[0].mana = 10

        self.assertNotIn(PlayCard(0, 0), env.core.legal_commands())
        self.assertFalse(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])
        target = _add_unit(env.core, 0, 921, life=3)
        self.assertIn(PlayCard(0, 0), env.core.legal_commands())
        self.assertTrue(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])

        env.step(ShadowverseEnv.PLAY_OFFSET)
        self.assertEqual(env.core.state.pending_choice.target_count, 1)
        choice_mask = env.action_mask()[
            ShadowverseEnv.CHOICE_OFFSET : ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET
        ]
        self.assertEqual(sum(choice_mask), 1)
        self.assertEqual(
            env.core.state.pending_choice.options[0].entity_id,
            target.entity_id,
        )

    def test_10812310_destroys_selected_enemy_and_heals(self):
        engine = self.fresh_engine()
        target = _add_unit(engine, 1, 1001, life=8)
        engine.players[0].health = 17
        _play(engine, 10812310, cost=4)
        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[0].health, 19)

    def test_10812310_no_target_is_illegal_without_mutation(self):
        engine = self.fresh_engine()
        _insert_hand_card(engine, _spell(10812310, cost=4))
        engine.players[0].mana = 10
        before = engine.deterministic_fingerprint()

        with self.assertRaises(IllegalCommand):
            engine.apply(PlayCard(0, 0))
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_10832310_heals_and_explicitly_spellboosts_the_whole_hand(self):
        engine = self.fresh_engine()
        engine.players[0].health = 17
        _insert_hand_card(engine, _card(1101, cost=4))
        _play(engine, 10832310, cost=2)

        self.assertEqual(engine.players[0].health, 19)
        self.assertTrue(
            all(card.spellboost_count == 2 for card in engine.players[0].hand)
        )

    def test_10842310_random_follower_damage_is_seeded(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh_engine(seed=73)
            targets = [
                _add_unit(engine, 1, 1200 + index, life=10)
                for index in range(3)
            ]
            _play(engine, 10842310, cost=6)
            outcomes.append(
                ([unit.health for unit in targets], engine.players[1].health)
            )

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(sorted(outcomes[0][0]), [6, 10, 10])
        self.assertEqual(outcomes[0][1], 16)

    def test_10842310_still_damages_leader_with_empty_enemy_board(self):
        engine = self.fresh_engine()
        _play(engine, 10842310, cost=6)
        self.assertEqual(engine.players[1].health, 16)


if __name__ == "__main__":
    unittest.main()
