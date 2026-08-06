# -*- coding: utf-8 -*-
"""Direct audits for the random-effect follow-up real-card batch."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, PlayCard
from swb.engine.effects import ConditionType, EffectKind, ExprType, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine
from swb.engine.state import Amulet, HandCard, Unit


BATCH_CARD_IDS = (
    10254110,
    10262110,
    10373110,
    10452110,
    10472120,
    10863110,
)
TOKEN_CARD_ID = 90054130


def _card(card_id: int, **overrides) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=overrides.get("card_set_id", 10000),
        class_id=overrides.get("class_id", 2),
        class_name=overrides.get("class_name", "皇家护卫"),
        name=overrides.get("name", f"card-{card_id}"),
        cost=overrides.get("cost", 1),
        card_type=overrides.get("card_type", "随从"),
        attack=overrides.get("attack", 1),
        life=overrides.get("life", 3),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _make_engine(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 241,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=2,
        class_b=2,
        seed=seed,
        rulebook=rulebook,
        card_resolver=repository.get,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _clear_hand(engine: GameEngine) -> None:
    engine.players[0].hand.clear()
    engine.players[0].hand_entity_ids.clear()


def _put_in_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    origin: CardOrigin = CardOrigin.DECK,
) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=origin,
    )
    engine.players[0].hand.insert(0, hand_card)
    engine.players[0].hand_entity_ids.insert(0, hand_card.entity_id)
    return hand_card


def _add_unit(
    engine: GameEngine,
    player_index: int,
    card_id: int,
    *,
    life: int = 3,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[player_index].board.append(unit)
    return unit


def _add_amulet(engine: GameEngine, player_index: int, card_id: int) -> Amulet:
    amulet = Amulet(
        definition=_card(
            card_id,
            card_type="护符",
            attack=None,
            life=None,
        ),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[player_index].board.append(amulet)
    return amulet


class DatabaseClauseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_texts_keywords_modes_and_references_match_audit(self):
        expected = {
            10254110: (
                "双轮夜行·吟雪&夕月", "梦魇", "随从", 9,
                "【<color=Keyword>入场曲</color>】【<color=Keyword>模式</color>】选择1个能力发动。\n"
                "<ridx=0>（1）召唤4个『<color=Keyword>一尾狐</color>』。</ridx>\n"
                "<ridx=1>（2）破坏对手的战场上的随机4个随从。回复自己的主战者4点生命值。</ridx>",
                {("入场曲", "入场曲"), ("模式", "模式")},
                {(0, 90054130, "一尾狐")},
            ),
            10262110: (
                "弹幕驱魔人·珂蕾特", "主教", "随从", 3,
                "【<color=Keyword>入场曲</color>】发动1次「对对手的战场上的随机1个随从造成2点伤害」。"
                "若自己的战场上的护符的张数为2张或以上，则改为发动2次。",
                {("入场曲", "入场曲")}, set(),
            ),
            10373110: (
                "破坏的团结者", "超越者", "随从", 6,
                "【<color=Keyword>入场曲</color>】破坏对手的战场上的随机X个随从。"
                "X为自己的战场上的其他卡牌张数。破坏自己的战场上的其他所有卡牌。",
                {("入场曲", "入场曲")}, set(),
            ),
            10452110: (
                "霸空武神·哪吒", "梦魇", "随从", 6,
                "【<color=Keyword>突进</color>】\n"
                "自己的回合结束时，对对手的战场上的随机1个随从造成4点伤害。"
                "对对手的战场上的随机1个随从造成2点伤害。",
                {("突进", "突进")}, set(),
            ),
            10472120: (
                "严厉的教官·伊尔莎", "超越者", "随从", 7,
                "【<color=Keyword>入场曲</color>】【<color=Keyword>模式</color>】选择1个能力发动。\n"
                "<ridx=0>（1）发动3次「对对手的战场上的随机1个随从造成4点伤害」。</ridx>\n"
                "<ridx=1>（2）对对手的主战者造成4点伤害。</ridx>",
                {("入场曲", "入场曲"), ("模式", "模式")}, set(),
            ),
            10863110: (
                "圣洁驱魔人·珂蕾特", "主教", "随从", 3,
                "【<color=Keyword>入场曲</color>】若自己的战场上有进化后的随从，则本随从进化。\n"
                "<hr>【<color=Keyword>守护</color>】\n"
                "<hr><ev>本随从进化时，发动2次「对对手的战场上的随机1个随从造成1点伤害」。</ev>",
                {("入场曲", "入场曲"), ("守护", "守护")}, set(),
            ),
            90054130: (
                "一尾狐", "梦魇", "随从", 2,
                "【<color=Keyword>突进</color>】\n"
                "【<color=Keyword>守护</color>】\n"
                "【<color=Keyword>谢幕曲</color>】使自己的战场上的随机1个"
                "『<color=Keyword>双轮夜行·吟雪&夕月</color>』+1/+0。",
                {("守护", "守护"), ("突进", "突进"), ("谢幕曲", "谢幕曲")},
                {(0, 10254110, "双轮夜行·吟雪&夕月")},
            ),
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    rows = connection.execute(
                        """
                        SELECT json_extract(c.raw_json, '$.name_chs'),
                               cl.class_name, cl.type_name, c.cost,
                               st.text_chs, st.text_eng, st.text_jpn, st.text_cht
                        FROM cards c
                        JOIN card_localizations cl
                          ON cl.card_id = c.card_id AND cl.language = 'zh-CN'
                        JOIN skill_texts st ON st.card_id = c.card_id
                        WHERE c.card_id = ? ORDER BY st.position
                        """,
                        (card_id,),
                    ).fetchall()
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0][:5], values[:5])
                    self.assertTrue(all(rows[0][index] for index in (5, 6, 7)))
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT ability_keyword, raw_keyword "
                            "FROM card_abilities WHERE card_id = ?",
                            (card_id,),
                        )),
                        values[5],
                    )
                    self.assertEqual(
                        set(connection.execute(
                            "SELECT position, referenced_card_id, referenced_name "
                            "FROM card_references WHERE card_id = ?",
                            (card_id,),
                        )),
                        values[6],
                    )
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id = ?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )

    def test_collectibles_and_token_have_exact_clause_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in BATCH_CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_random_effect_followup_batch.py"],
                )

        token_report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        token = next(
            card
            for card in token_report["cards"]
            if card["card_id"] == TOKEN_CARD_ID
        )
        self.assertEqual(token["category"], "entry_behavior_complete")
        self.assertEqual(token["explicit_coverage"], "exact")
        self.assertEqual(
            token["authored_producers"],
            [{
                "source_card_id": 10254110,
                "entry_kind": "summon",
                "rule_file": "real_random_effect_followup_batch.json",
                "rule_group": "rules",
            }],
        )


class RealRandomEffectFollowupBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 241) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def play_real(self, engine: GameEngine, card_id: int) -> Unit:
        _put_in_hand(engine, self.repository.get(card_id))
        engine.apply(PlayCard(0, 0))
        return next(
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == card_id
        )

    def test_rule_shapes_preserve_batches_repetitions_filters_and_exclusions(self):
        twin = self.rulebook.operations_for(10254110, Trigger.FANFARE)[0]
        self.assertEqual(twin.kind, EffectKind.CHOOSE_ONE)
        self.assertEqual(len(twin.choose_one_options), 2)
        destroy = twin.choose_one_options[1].operations[0]
        self.assertEqual(
            (destroy.target, destroy.target_count),
            (TargetKind.RANDOM_ENEMY_UNIT, 4),
        )

        barrage = self.rulebook.operations_for(10262110, Trigger.FANFARE)
        self.assertEqual(len(barrage), 2)
        self.assertEqual(
            barrage[1].conditions[0].type,
            ConditionType.CONTROLLER_BOARD_HAS,
        )
        self.assertEqual(barrage[1].conditions[0].value, 2)
        self.assertEqual(barrage[1].conditions[0].board_filter.card_type, "护符")

        congregant = self.rulebook.operations_for(10373110, Trigger.FANFARE)
        self.assertEqual(
            congregant[0].target_count_expr.type,
            ExprType.SUBTRACT,
        )
        self.assertTrue(congregant[1].exclude_source)
        self.assertEqual(congregant[1].target, TargetKind.ALL_OWN_BOARD)

        ilsa = self.rulebook.operations_for(10472120, Trigger.FANFARE)[0]
        triple = ilsa.choose_one_options[0].operations
        self.assertEqual(len(triple), 3)
        self.assertTrue(all(operation.target_count == 1 for operation in triple))

        fox = self.rulebook.operations_for(TOKEN_CARD_ID, Trigger.LAST_WORDS)[0]
        self.assertEqual(fox.board_filter.card_id, 10254110)
        self.assertEqual((fox.amount, fox.secondary_amount), (1, 0))

    def test_twin_calamities_summons_four_complete_foxes_and_fox_buffs_source(self):
        engine = self.fresh_engine(seed=251)
        _clear_hand(engine)
        source = self.play_real(engine, 10254110)
        engine.apply(Choose(0, "choose_one:summon_foxes"))

        foxes = [
            unit
            for unit in engine.players[0].board
            if unit.definition.card_id == TOKEN_CARD_ID
        ]
        self.assertEqual(len(foxes), 4)
        self.assertTrue(all(unit.origin is CardOrigin.TOKEN for unit in foxes))
        self.assertTrue(all(unit.has_keyword("突进") and unit.has_guard for unit in foxes))

        attack_before = source.attack
        foxes[0].health = 0
        engine._stabilize()

        self.assertEqual(source.attack, attack_before + 1)
        self.assertEqual(len(engine.players[0].board), 4)

    def test_twin_calamities_destroy_branch_is_distinct_batch_then_heals(self):
        engine = self.fresh_engine(seed=257)
        _clear_hand(engine)
        engine.players[0].health = 15
        targets = [_add_unit(engine, 1, 3100 + index) for index in range(5)]
        self.play_real(engine, 10254110)
        engine.apply(Choose(0, "choose_one:destroy_and_heal"))

        self.assertEqual(sum(target in engine.players[1].board for target in targets), 1)
        self.assertEqual(engine.players[0].health, 19)
        self.assertEqual([len(batch.records) for batch in engine.state.death_queue], [4])

    def test_barrage_exorcist_requires_two_matching_amulets_for_second_hit(self):
        one = self.fresh_engine(seed=263)
        _clear_hand(one)
        _add_amulet(one, 0, 3200)
        one_target = _add_unit(one, 1, 3201, life=4)
        self.play_real(one, 10262110)
        self.assertEqual(one_target.health, 2)

        two = self.fresh_engine(seed=263)
        _clear_hand(two)
        _add_amulet(two, 0, 3210)
        _add_amulet(two, 0, 3211)
        two_target = _add_unit(two, 1, 3212, life=4)
        self.play_real(two, 10262110)
        self.assertNotIn(two_target, two.players[1].board)
        self.assertEqual(len(two.state.death_queue[-1].records), 1)

    def test_congregant_counts_other_allied_cards_then_destroys_them_all(self):
        engine = self.fresh_engine(seed=269)
        _clear_hand(engine)
        own_cards = [
            _add_unit(engine, 0, 3300),
            _add_unit(engine, 0, 3301),
            _add_amulet(engine, 0, 3302),
        ]
        enemies = [_add_unit(engine, 1, 3310 + index) for index in range(4)]

        source = self.play_real(engine, 10373110)

        self.assertEqual(sum(enemy in engine.players[1].board for enemy in enemies), 1)
        self.assertTrue(all(card not in engine.players[0].board for card in own_cards))
        self.assertEqual(engine.players[0].board, [source])
        self.assertEqual(
            [len(batch.records) for batch in engine.state.death_queue],
            [3, 3],
        )

    def test_nezha_turn_end_hits_in_sequence_and_skips_after_target_dies(self):
        engine = self.fresh_engine(seed=271)
        _clear_hand(engine)
        target = _add_unit(engine, 1, 3400, life=4)
        source = self.play_real(engine, 10452110)
        self.assertTrue(source.has_keyword("突进"))

        engine.apply(EndTurn(0))

        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(len(engine.state.death_queue), 1)
        self.assertEqual(engine.players[1].health, 20)

    def test_ilsa_modes_keep_triple_hits_repeated_and_leader_branch_separate(self):
        def triple_once():
            engine = self.fresh_engine(seed=277)
            _clear_hand(engine)
            targets = [_add_unit(engine, 1, 3500 + index, life=20) for index in range(2)]
            self.play_real(engine, 10472120)
            engine.apply(Choose(0, "choose_one:triple_damage"))
            return tuple(target.health for target in targets)

        first = triple_once()
        second = triple_once()
        self.assertEqual(first, second)
        self.assertEqual(sum(first), 28)

        leader = self.fresh_engine(seed=281)
        _clear_hand(leader)
        target = _add_unit(leader, 1, 3510, life=20)
        self.play_real(leader, 10472120)
        leader.apply(Choose(0, "choose_one:leader_damage"))
        self.assertEqual(leader.players[1].health, 16)
        self.assertEqual(target.health, 20)

    def test_holy_exorcist_effect_evolves_and_fires_two_independent_hits(self):
        normal = self.fresh_engine(seed=283)
        _clear_hand(normal)
        untouched = _add_unit(normal, 1, 3600, life=3)
        normal_source = self.play_real(normal, 10863110)
        self.assertFalse(normal_source.evolved)
        self.assertEqual(untouched.health, 3)

        engine = self.fresh_engine(seed=283)
        _clear_hand(engine)
        ally = _add_unit(engine, 0, 3610)
        ally.evolved = True
        target = _add_unit(engine, 1, 3611, life=2)
        ep_before = engine.players[0].evolution_points

        source = self.play_real(engine, 10863110)

        self.assertTrue(source.evolved)
        self.assertEqual((source.attack, source.health), (4, 6))
        self.assertTrue(source.has_guard)
        self.assertNotIn(target, engine.players[1].board)
        self.assertEqual(engine.players[0].evolution_points, ep_before)
        self.assertFalse(engine.players[0].evolved_this_turn)

    def test_choose_one_follower_has_rl_mask_parity_and_resumes(self):
        deck_a = [_card(card_id) for card_id in range(4000, 4040)]
        deck_b = [_card(card_id) for card_id in range(5000, 5040)]
        env = ShadowverseEnv(
            deck_a,
            deck_b,
            class_a=2,
            class_b=2,
            seed=293,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
        )
        env.reset(seed=293)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        _put_in_hand(env.core, self.repository.get(10472120))
        env.players[0].max_mana = env.players[0].mana = 10

        self.assertTrue(env.action_mask()[ShadowverseEnv.PLAY_OFFSET])
        env.step(ShadowverseEnv.PLAY_OFFSET)
        choice = Choose(0, "choose_one:leader_damage")
        action = env._encode_command(choice)
        self.assertIsNotNone(action)
        self.assertTrue(env.action_mask()[action])

        result = env.step(action)

        self.assertIsNone(env.core.state.pending_choice)
        self.assertEqual(env.players[1].health, 16)
        self.assertFalse(result.terminated)


if __name__ == "__main__":
    unittest.main()
