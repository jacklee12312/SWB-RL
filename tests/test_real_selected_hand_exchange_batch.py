# -*- coding: utf-8 -*-
"""Exact audits for eight real cards that select another card in hand."""

from __future__ import annotations

import os
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Attack, Choose, Evolve, PlayCard
from swb.engine.effects import EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import DamageType, GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


CARD_IDS = (
    10131120,
    10621120,
    10641110,
    10642110,
    10703110,
    10711120,
    10811130,
    10842120,
)
SOURCE_HASHES = {
    10131120: "da40cf4064ee5bc9ddc5b0be0d555076ce6052bcc50a189c1ed15624307e15c3",
    10621120: "3d569e34737fb9ae6dc7e33115e31a038a324be8d50b22940de004b87b5e77b2",
    10641110: "cc0eaf6e72e030bf45d7cbc7bfc7750950118079ab82a88c2664aaac97d9a960",
    10642110: "0b8201eb36e96718e064dabab105e5ae966b7faf29653230c0532c966b028a63",
    10703110: "5acc560190081e6be75e3dabb1b6295a355eadda00ab2ca38f996025b6e3cc9f",
    10711120: "fc5d58ee4d4c80e6d90203a66c76fd0a0cd815ab1d5eb5de007f198f70afbde6",
    10811130: "644fd8fc4d74153e78468e496d661b8abaaf5d8c65f0e016cc3f144b236b1a16",
    10842120: "5b08ca8bb8a8dc71c309be9561ecf7847a50badd28c54f5f16e0234afddfd645",
}
STRUCTURED_EVIDENCE = {
    10131120: {
        "triggers": ["fanfare"],
        "effect_kinds": ["return_to_deck", "draw", "add_earth_sigils"],
    },
    10621120: {
        "triggers": ["fanfare"],
        "effect_kinds": ["return_to_deck", "draw_filtered"],
    },
    10641110: {
        "triggers": ["fanfare", "last_words"],
        "effect_kinds": ["discard", "draw"],
    },
    10642110: {
        "triggers": ["fanfare", "intrinsic_keywords"],
        "effect_kinds": ["discard", "keyword:突进", "keyword:屏障"],
    },
    10703110: {
        "triggers": ["fanfare"],
        "effect_kinds": ["discard", "damage_unit"],
    },
    10711120: {
        "triggers": ["fanfare"],
        "effect_kinds": ["return_to_deck", "add_card", "add_card"],
    },
    10811130: {
        "triggers": ["fanfare", "intrinsic_keywords"],
        "effect_kinds": ["return_to_deck", "draw", "keyword:守护"],
    },
    10842120: {
        "triggers": ["fanfare", "evolve"],
        "effect_kinds": [
            "discard", "draw", "heal_leader",
            "discard", "draw", "heal_leader",
        ],
    },
}


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
        life=overrides.get("life", 3),
        keywords=overrides.get("keywords", frozenset()),
        support_level="basic",
        is_collectible=overrides.get("is_collectible", True),
    )


def _spell(card_id: int, **overrides) -> CardDefinition:
    return _card(card_id, card_type="法术", attack=None, life=None, **overrides)


def _make_engine(
    rulebook: RuleBook,
    repository: CardRepository,
    *,
    seed: int = 1421,
) -> GameEngine:
    engine = GameEngine(
        [_card(card_id) for card_id in range(1000, 1040)],
        [_card(card_id) for card_id in range(2000, 2040)],
        class_a=1,
        class_b=1,
        seed=seed,
        rulebook=rulebook,
        card_resolver=repository.get,
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=seed)
    for player in engine.players:
        player.hand.clear()
        player.hand_entity_ids.clear()
        player.max_mana = player.mana = 10
    return engine


def _put_in_hand(engine: GameEngine, definition: CardDefinition) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[0].hand.insert(0, card)
    engine.players[0].hand_entity_ids.insert(0, card.entity_id)
    return card


def _play_real(
    engine: GameEngine,
    repository: CardRepository,
    card_id: int,
) -> Unit:
    _put_in_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, 0))
    return next(
        unit for unit in engine.players[0].board
        if unit.definition.card_id == card_id
    )


def _choose_hand(engine: GameEngine, hand_card: HandCard) -> None:
    engine.apply(Choose(0, f"hand:{hand_card.entity_id}"))


def _add_unit(
    engine: GameEngine,
    owner: int,
    card_id: int,
    *,
    attack: int = 1,
    life: int = 3,
) -> Unit:
    unit = Unit.summon(
        _card(card_id, attack=attack, life=life),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[owner].board.append(unit)
    return unit


class DatabaseAndClauseAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "cards.sqlite3"
        )
        if not os.path.exists(cls.db_path):
            raise unittest.SkipTest("cards.sqlite3 not found")

    def test_database_cards_match_audited_stats_text_and_keywords(self):
        expected = {
            10131120: ("见习占星术师", "巫师", 2, 2, 2),
            10621120: ("懒惰女仆", "皇家护卫", 4, 3, 5),
            10641110: ("决断的龙人", "龙族", 3, 2, 3),
            10642110: ("果断的剑圣", "龙族", 4, 4, 5),
            10703110: ("享乐的上级市民", "中立", 3, 1, 1),
            10711120: ("精灵陷阱师", "精灵", 1, 1, 1),
            10811130: ("忧郁少女·莫埃尔", "精灵", 1, 1, 1),
            10842120: ("满面笑容的烹饪·琪米卡", "龙族", 2, 2, 1),
        }
        expected_abilities = {
            10131120: {"入场曲"},
            10621120: {"入场曲"},
            10641110: {"入场曲", "谢幕曲"},
            10642110: {"入场曲", "突进", "屏障"},
            10703110: {"入场曲"},
            10711120: {"入场曲"},
            10811130: {"入场曲", "守护"},
            10842120: {"入场曲", "进化时"},
        }
        with closing(sqlite3.connect(self.db_path)) as connection:
            for card_id, values in expected.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        """
                        SELECT json_extract(c.raw_json, '$.name_chs'),
                               cl.class_name, c.cost, c.attack, c.life
                        FROM cards c
                        JOIN card_localizations cl
                          ON cl.card_id=c.card_id AND cl.language='zh-CN'
                        WHERE c.card_id=?
                        """,
                        (card_id,),
                    ).fetchone()
                    self.assertEqual(row, values)
                    abilities = {
                        row[0] for row in connection.execute(
                            "SELECT ability_keyword FROM card_abilities WHERE card_id=?",
                            (card_id,),
                        )
                    }
                    self.assertEqual(abilities, expected_abilities[card_id])
                    self.assertEqual(
                        connection.execute(
                            "SELECT COUNT(*) FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchone()[0],
                        0,
                    )
            self.assertEqual(
                connection.execute(
                    "SELECT referenced_card_id FROM card_references WHERE card_id=10711120"
                ).fetchall(),
                [(90011110,)],
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM card_references WHERE card_id IN "
                    "(10131120,10621120,10641110,10642110,10703110,10811130,10842120)"
                ).fetchone()[0],
                0,
            )

    def test_all_eight_cards_are_exact_with_hash_and_evidence(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["clause_audit_issues"], [])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                audit = info["clause_audit"]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(audit["status"], "mapped_exact")
                self.assertEqual(audit["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(
                    audit["structured_evidence"], STRUCTURED_EVIDENCE[card_id]
                )
                self.assertEqual(
                    audit["test_evidence"],
                    ["tests/test_real_selected_hand_exchange_batch.py"],
                )


class RealSelectedHandExchangeBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh_engine(self, *, seed: int = 1421) -> GameEngine:
        return _make_engine(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_filters_and_intrinsic_keywords_are_exact(self):
        for card_id in CARD_IDS:
            operations = self.rulebook.operations_for(card_id, Trigger.FANFARE)
            self.assertIn(
                operations[0].kind,
                {EffectKind.RETURN_TO_DECK, EffectKind.DISCARD},
            )
            self.assertEqual(operations[0].target, TargetKind.OWN_HAND)
            self.assertFalse(operations[0].requires_target)
        filtered = self.rulebook.operations_for(10621120, Trigger.FANFARE)[1]
        self.assertEqual(filtered.kind, EffectKind.DRAW_FILTERED)
        self.assertEqual(
            (filtered.deck_filter.card_type, filtered.deck_filter.class_id),
            ("随从", 2),
        )
        self.assertEqual(self.rulebook.intrinsic_keywords_for(10642110), ("突进", "屏障"))
        self.assertEqual(self.rulebook.intrinsic_keywords_for(10811130), ("守护",))
        self.assertEqual(
            self.rulebook.operations_for(10842120, Trigger.FANFARE),
            self.rulebook.operations_for(10842120, Trigger.EVOLVE),
        )

    def test_astrologer_returns_then_draws_then_adds_earth_sigil(self):
        engine = self.fresh_engine()
        engine.players[0].deck = [_card(3000), _card(3001)]
        returned = _put_in_hand(engine, _card(3010))
        _play_real(engine, self.repository, 10131120)
        _choose_hand(engine, returned)
        self.assertTrue(any(card.card_id == 3010 for card in engine.players[0].deck))
        self.assertEqual(engine.players[0].earth_sigils, 1)
        relevant = [
            event.type for event in engine.event_history
            if event.type in {
                EventType.CARD_RETURNED_TO_DECK,
                EventType.CARD_DRAWN,
                EventType.EARTH_SIGILS_CHANGED,
            }
        ]
        self.assertEqual(relevant[-3:], [
            EventType.CARD_RETURNED_TO_DECK,
            EventType.CARD_DRAWN,
            EventType.EARTH_SIGILS_CHANGED,
        ])

        no_target = self.fresh_engine()
        no_target.players[0].deck = [_card(3020)]
        _play_real(no_target, self.repository, 10131120)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertEqual((len(no_target.players[0].hand), no_target.players[0].earth_sigils), (1, 1))

    def test_maid_returns_selected_card_and_draws_only_two_royal_followers(self):
        engine = self.fresh_engine(seed=99)
        engine.players[0].deck = [
            _card(3100, class_id=2),
            _spell(3101, class_id=2),
            _card(3102, class_id=2),
            _card(3103, class_id=1),
        ]
        returned = _put_in_hand(engine, _card(3110))
        _play_real(engine, self.repository, 10621120)
        _choose_hand(engine, returned)
        self.assertEqual(
            {card.card_id for card in engine.players[0].hand},
            {3100, 3102},
        )
        self.assertTrue(any(card.card_id == 3110 for card in engine.players[0].deck))

    def test_dragon_discards_and_last_words_draws_two(self):
        engine = self.fresh_engine()
        engine.players[0].deck = [_card(3200), _card(3201), _card(3202)]
        discarded = _put_in_hand(engine, _card(3210))
        source = _play_real(engine, self.repository, 10641110)
        _choose_hand(engine, discarded)
        self.assertTrue(any(card.entity_id == discarded.entity_id for card in engine.players[0].graveyard))
        hand_before = len(engine.players[0].hand)
        engine.apply_damage(None, source, 99, DamageType.EFFECT, controller=1)
        engine._stabilize()
        self.assertEqual(len(engine.players[0].hand), hand_before + 2)

        no_target = self.fresh_engine()
        _play_real(no_target, self.repository, 10641110)
        self.assertIsNone(no_target.state.pending_choice)

    def test_swordsman_has_rush_and_barrier_after_optional_discard(self):
        engine = self.fresh_engine()
        target = _add_unit(engine, 1, 3300, life=5)
        source = _play_real(engine, self.repository, 10642110)
        self.assertIsNone(engine.state.pending_choice)
        self.assertTrue(source.has_keyword("突进"))
        self.assertTrue(source.has_keyword("屏障"))
        self.assertIn(Attack(0, source.entity_id, target.entity_id), engine.legal_commands())
        self.assertNotIn(Attack(0, source.entity_id, None), engine.legal_commands())
        health_before = source.health
        engine.apply_damage(None, source, 3, DamageType.EFFECT, controller=1)
        self.assertEqual(source.health, health_before)
        self.assertEqual(source.barrier_charges, 0)

    def test_citizen_discards_then_damages_all_enemies_even_without_hand_target(self):
        engine = self.fresh_engine()
        doomed = _add_unit(engine, 1, 3400, life=2)
        survivor = _add_unit(engine, 1, 3401, life=4)
        discarded = _put_in_hand(engine, _card(3410))
        _play_real(engine, self.repository, 10703110)
        _choose_hand(engine, discarded)
        self.assertNotIn(doomed, engine.players[1].board)
        self.assertEqual(survivor.health, 2)

        no_target = self.fresh_engine()
        doomed = _add_unit(no_target, 1, 3420, life=2)
        _play_real(no_target, self.repository, 10703110)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertNotIn(doomed, no_target.players[1].board)

    def test_trapper_returns_first_then_adds_two_exact_fairies_with_hand_cap(self):
        engine = self.fresh_engine()
        returned = _put_in_hand(engine, _card(3500))
        _play_real(engine, self.repository, 10711120)
        _choose_hand(engine, returned)
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [90011110, 90011110],
        )
        self.assertTrue(any(card.card_id == 3500 for card in engine.players[0].deck))

        capped = self.fresh_engine()
        for card_id in range(3510, 3518):
            _put_in_hand(capped, _card(card_id))
        returned = capped.players[0].hand[-1]
        _play_real(capped, self.repository, 10711120)
        _choose_hand(capped, returned)
        self.assertEqual(len(capped.players[0].hand), capped.config.max_hand)
        self.assertEqual(
            sum(card.card_id == 90011110 for card in capped.players[0].hand),
            2,
        )

    def test_moel_returns_draws_and_ward_blocks_leader_attack(self):
        engine = self.fresh_engine()
        engine.players[0].deck = [_card(3600)]
        returned = _put_in_hand(engine, _card(3610))
        source = _play_real(engine, self.repository, 10811130)
        _choose_hand(engine, returned)
        self.assertTrue(source.has_guard)
        attacker = _add_unit(engine, 1, 3620, attack=3)
        attacker.can_attack = True
        attacker.attacks_remaining = 1
        attacker.rush_only = False
        engine.state.active_player = 1
        self.assertIn(Attack(1, attacker.entity_id, source.entity_id), engine.legal_commands())
        self.assertNotIn(Attack(1, attacker.entity_id, None), engine.legal_commands())

    def test_kimika_fanfare_and_evolve_repeat_discard_draw_heal(self):
        engine = self.fresh_engine()
        engine.players[0].deck = [_card(3700), _card(3701), _card(3702)]
        engine.players[0].health = 17
        discarded = _put_in_hand(engine, _card(3710))
        source = _play_real(engine, self.repository, 10842120)
        _choose_hand(engine, discarded)
        self.assertEqual(engine.players[0].health, 18)
        drawn = engine.players[0].hand[0]
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        ep_before = engine.players[0].evolution_points
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(engine.state.pending_choice.options[0].entity_id, drawn.entity_id)
        _choose_hand(engine, drawn)
        self.assertEqual(engine.players[0].health, 19)
        self.assertEqual(engine.players[0].evolution_points, ep_before - 1)
        self.assertEqual(len(engine.players[0].hand), 1)

        capped = self.fresh_engine()
        capped.players[0].health = capped.config.starting_health
        capped.players[0].deck = [_card(3720)]
        _play_real(capped, self.repository, 10842120)
        self.assertIsNone(capped.state.pending_choice)
        self.assertEqual(capped.players[0].health, capped.config.starting_health)
        self.assertEqual(len(capped.players[0].hand), 1)

    def test_pending_hand_target_that_left_hand_skips_and_continues_deterministically(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh_engine(seed=2718)
            engine.players[0].deck = [_card(3800), _card(3801)]
            target = _put_in_hand(engine, _card(3810))
            _play_real(engine, self.repository, 10131120)
            choice = Choose(0, f"hand:{target.entity_id}")
            index = engine.players[0].hand.index(target)
            engine.players[0].hand.pop(index)
            engine.players[0].hand_entity_ids.pop(index)
            engine._send_to_graveyard(
                0,
                target.definition,
                "test_pending_hand_target_left",
                source_entity_id=target.entity_id,
            )
            engine.apply(choice)
            self.assertIsNone(engine.state.pending_choice)
            self.assertEqual((len(engine.players[0].hand), engine.players[0].earth_sigils), (1, 1))
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_illegal_hand_choice_does_not_mutate_pending_state(self):
        engine = self.fresh_engine()
        target = _put_in_hand(engine, _card(3900))
        _play_real(engine, self.repository, 10641110)
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "hand:999999"))
        self.assertEqual(engine.deterministic_fingerprint(), before)
        _choose_hand(engine, target)

    def test_rl_mask_matches_play_and_selected_hand_choice(self):
        env = ShadowverseEnv(
            [_card(card_id) for card_id in range(4000, 4040)],
            [_card(card_id) for card_id in range(4100, 4140)],
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
        env.players[0].max_mana = env.players[0].mana = 10
        target = _put_in_hand(env.core, _card(4200))
        _put_in_hand(env.core, self.repository.get(10641110))
        self.assertTrue(env.action_mask()[env.PLAY_OFFSET])
        env.step(env.PLAY_OFFSET)
        enabled = [
            action for action in range(env.CHOICE_OFFSET, env.GRAVEYARD_CHOICE_OFFSET)
            if env.action_mask()[action]
        ]
        self.assertEqual(enabled, [env.CHOICE_OFFSET])
        self.assertEqual(env.core.state.pending_choice.options[0].entity_id, target.entity_id)

    def test_fairy_token_audit_includes_trapper_as_complete_producer(self):
        report = _build_token_audit(
            "data/cards.sqlite3",
            "data/rules",
            "data/audits/token_overrides.json",
        )
        fairy = next(
            card for card in report["cards"] if card["card_id"] == 90011110
        )
        self.assertEqual(fairy["category"], "entry_behavior_complete")
        self.assertIn(
            10711120,
            {
                producer["source_card_id"]
                for producer in fairy["authored_producers"]
            },
        )


if __name__ == "__main__":
    unittest.main()
