# -*- coding: utf-8 -*-
"""Exact attack-history crest and generated countdown spell chain."""

from __future__ import annotations

import copy
import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import (
    RuleBook,
    Trigger,
    _parse_condition,
    _parse_operation,
)
from swb.engine.commands import Attack, EndTurn, Evolve, PlayCard
from swb.engine.effects import (
    ConditionType,
    EffectKind,
    EffectOperation,
    ExprType,
    TargetKind,
)
from swb.engine.emblem import EmblemDefinition
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


SOURCE_ID = 10364120
TOKEN_ID = 90064310
EMBLEM_ID = "despair_manifest_marwynn"
SOURCE_HASHES = {
    SOURCE_ID: "459769cb5d49c35cdc03808b4beb8d82de628ff24b98e1f93a0f4ab35dcf3ec2",
    TOKEN_ID: "81413d0a6ed82a18dbc112cf8a1c129697522d38cb3f81d5aacf76b99b5ab48a",
}


def _ready_attacker(engine, owner: int, card_id: int = 99700):
    unit = _put_unit(engine, owner, _card(card_id, attack=1, life=5))
    unit.summoned_this_turn = False
    unit.can_attack = True
    unit.attacks_remaining = 1
    unit.rush_only = False
    return unit


def _add_emblem(
    engine,
    owner: int,
    emblem_id: str,
    *,
    countdown: int | None,
    source_card_id: int,
):
    definition = EmblemDefinition(
        emblem_id=emblem_id,
        source_card_id=source_card_id,
        countdown=countdown,
    )
    engine._add_emblem_to_player(owner, definition, source_card_id)
    return engine.players[owner].emblems[-1]


class RealAttackHistoryEmblemCountdownTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 6412, rulebook: RuleBook | None = None):
        return _fresh(rulebook or self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_schema_validation(self):
        source_fanfare = self.rulebook.operations_for(SOURCE_ID, Trigger.FANFARE)
        source_evolve = self.rulebook.operations_for(SOURCE_ID, Trigger.EVOLVE)
        token_ops = self.rulebook.operations_for(TOKEN_ID, Trigger.PLAY)
        emblem = self.rulebook.emblem_def(EMBLEM_ID)

        self.assertEqual(source_fanfare[0].kind, EffectKind.ADD_CARD)
        self.assertEqual(source_fanfare[0].card_id, TOKEN_ID)
        self.assertEqual(source_evolve[0].kind, EffectKind.GAIN_EMBLEM)
        self.assertEqual(source_evolve[0].emblem_id, EMBLEM_ID)
        self.assertEqual(
            [(op.kind, op.target) for op in token_ops],
            [
                (EffectKind.BANISH, TargetKind.RANDOM_ENEMY_UNIT),
                (EffectKind.INCREASE_COUNTDOWN, TargetKind.ALL_OWN_EMBLEMS),
            ],
        )
        trigger = emblem.triggers[0]
        self.assertEqual(trigger.trigger, "turn_end")
        self.assertEqual(
            trigger.conditions[0].type,
            ConditionType.CONTROLLER_FOLLOWER_ATTACKS_THIS_TURN_AT_MOST,
        )
        self.assertEqual(trigger.conditions[0].value, 0)
        self.assertIs(trigger.operations[0].kind, EffectKind.DISTRIBUTE_DAMAGE)
        self.assertIs(
            trigger.operations[0].amount_expr.type,
            ExprType.CONTROLLER_EMBLEM_COUNT,
        )
        self.assertTrue(trigger.operations[0].include_leader)

        condition = _parse_condition(
            {
                "type": "controller_follower_attacks_this_turn_at_most",
                "value": 0,
            },
            "test.json/conditions[0]",
            SOURCE_ID,
        )
        self.assertEqual(condition.value, 0)
        for raw in (
            {"type": "controller_follower_attacks_this_turn_at_most"},
            {
                "type": "controller_follower_attacks_this_turn_at_most",
                "value": -1,
            },
            {
                "type": "controller_follower_attacks_this_turn_at_most",
                "value": True,
            },
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_condition(raw, "test.json/conditions[0]", SOURCE_ID)

        valid = _parse_operation(
            {
                "kind": "increase_countdown",
                "target": "all_own_emblems",
                "amount": 1,
            },
            "test.json/operations[0]",
            TOKEN_ID,
        )
        self.assertIs(valid.target, TargetKind.ALL_OWN_EMBLEMS)
        for raw in (
            {
                "kind": "increase_countdown",
                "target": "all_own_emblems",
                "amount": 0,
            },
            {
                "kind": "reduce_countdown",
                "target": "all_own_emblems",
                "amount": 1,
            },
            {
                "kind": "damage_unit",
                "target": "all_own_emblems",
                "amount": 1,
            },
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                _parse_operation(raw, "test.json/operations[0]", TOKEN_ID)

    def test_source_fanfare_produces_token_and_evolve_gains_nonstacking_crest(self):
        engine = self.fresh(seed=3)
        source = _play(engine, self.repository, SOURCE_ID)
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [TOKEN_ID],
        )

        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [emblem.emblem_id for emblem in engine.players[0].emblems],
            [EMBLEM_ID],
        )
        definition = self.rulebook.emblem_def(EMBLEM_ID)
        engine._add_emblem_to_player(0, definition, SOURCE_ID)
        self.assertEqual(len(engine.players[0].emblems), 1)

    def test_crest_distributes_emblem_count_when_no_follower_attacked(self):
        engine = self.fresh(seed=5)
        source = _play(engine, self.repository, SOURCE_ID)
        engine.players[0].turns_started = engine.config.evolution_unlock_turn
        engine.apply(Evolve(0, source.entity_id))
        _add_emblem(engine, 0, "countdown-a", countdown=2, source_card_id=99001)
        _add_emblem(engine, 0, "permanent-b", countdown=None, source_card_id=99002)
        enemy = _put_unit(engine, 1, _card(99710, attack=0, life=1))

        engine.apply(EndTurn(0))

        self.assertNotIn(enemy, engine.players[1].board)
        self.assertEqual(engine.players[1].health, 18)
        self.assertEqual(engine.players[0].follower_attacks_this_turn, 0)

    def test_legal_attack_suppresses_crest_and_counter_resets_after_turn(self):
        engine = self.fresh(seed=7)
        engine._add_emblem_to_player(
            0,
            self.rulebook.emblem_def(EMBLEM_ID),
            SOURCE_ID,
        )
        attacker = _ready_attacker(engine, 0, 99720)
        target = _put_unit(engine, 1, _card(99721, attack=0, life=5))

        engine.apply(Attack(0, attacker.entity_id, target.entity_id))

        self.assertEqual(engine.players[0].follower_attacks_this_turn, 1)
        declared = next(
            event
            for event in engine.event_history
            if event.type is EventType.ATTACK_DECLARED
        )
        self.assertEqual(declared.metadata["follower_attacks_this_turn"], 1)
        self.assertEqual(target.health, 4)
        engine.apply(EndTurn(0))
        self.assertEqual(target.health, 4)
        self.assertEqual(engine.players[1].health, 20)
        self.assertEqual(engine.players[0].follower_attacks_this_turn, 0)

    def test_attack_still_counts_when_on_attack_effect_removes_attacker(self):
        rulebook = copy.deepcopy(self.rulebook)
        attacker_card_id = 99730
        rulebook._rules[(attacker_card_id, Trigger.ATTACK)] = (
            EffectOperation(
                EffectKind.DESTROY,
                TargetKind.SELF,
            ),
        )
        engine = self.fresh(seed=11, rulebook=rulebook)
        engine._add_emblem_to_player(
            0,
            rulebook.emblem_def(EMBLEM_ID),
            SOURCE_ID,
        )
        attacker = _ready_attacker(engine, 0, attacker_card_id)

        engine.apply(Attack(0, attacker.entity_id, None))

        self.assertNotIn(attacker, engine.players[0].board)
        self.assertEqual(engine.players[0].follower_attacks_this_turn, 1)
        self.assertEqual(engine.players[1].health, 20)
        engine.apply(EndTurn(0))
        self.assertEqual(engine.players[1].health, 20)

    def test_illegal_attack_preserves_counter_state_rng_events_and_log(self):
        engine = self.fresh(seed=13)
        attacker = _put_unit(engine, 0, _card(99740))
        before = engine.deterministic_fingerprint()
        before_rng = engine.random.getstate()
        before_events = tuple(engine.event_history)
        before_log = tuple(engine.logs)

        with self.assertRaises(IllegalCommand):
            engine.apply(Attack(0, attacker.entity_id, None))

        self.assertEqual(engine.deterministic_fingerprint(), before)
        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertEqual(tuple(engine.event_history), before_events)
        self.assertEqual(tuple(engine.logs), before_log)
        self.assertEqual(engine.players[0].follower_attacks_this_turn, 0)

    def test_token_seeded_banish_and_all_countdown_crest_increase(self):
        outcomes = []
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=17)
            first = _add_emblem(
                engine, 0, "first", countdown=2, source_card_id=99011
            )
            permanent = _add_emblem(
                engine, 0, "permanent", countdown=None, source_card_id=99012
            )
            second = _add_emblem(
                engine, 0, "second", countdown=1, source_card_id=99013
            )
            opposing = _add_emblem(
                engine, 1, "opposing", countdown=4, source_card_id=99014
            )
            enemies = [
                _put_unit(engine, 1, _card(99750 + index))
                for index in range(3)
            ]

            _play(engine, self.repository, TOKEN_ID)

            surviving_ids = tuple(
                unit.definition.card_id for unit in engine.players[1].board
            )
            outcomes.append(surviving_ids)
            fingerprints.append(engine.deterministic_fingerprint())
            self.assertEqual((first.countdown, permanent.countdown, second.countdown), (3, None, 2))
            self.assertEqual(opposing.countdown, 4)
            self.assertEqual(len(surviving_ids), 2)
            banished_ids = [card.card_id for card in engine.players[1].banished]
            self.assertEqual(len(banished_ids), 1)
            self.assertIn(banished_ids[0], {unit.definition.card_id for unit in enemies})
            changed = [
                event
                for event in engine.event_history
                if event.type is EventType.EMBLEM_COUNTDOWN_CHANGED
                and event.metadata.get("effect_source_card_id") == TOKEN_ID
            ]
            self.assertEqual(
                [event.source_id for event in changed],
                [first.entity_id, second.entity_id],
            )
            self.assertTrue(all(event.amount == 1 for event in changed))

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_token_without_enemy_skips_rng_but_still_increases_countdowns(self):
        engine = self.fresh(seed=19)
        countdown = _add_emblem(
            engine, 0, "only-countdown", countdown=3, source_card_id=99021
        )
        _add_emblem(
            engine, 0, "only-permanent", countdown=None, source_card_id=99022
        )
        before_rng = engine.random.getstate()

        _play(engine, self.repository, TOKEN_ID)

        self.assertEqual(engine.random.getstate(), before_rng)
        self.assertEqual(countdown.countdown, 4)
        self.assertFalse(engine.players[1].banished)

    def test_invariant_rejects_negative_attack_history(self):
        engine = self.fresh(seed=23)
        engine.players[0].follower_attacks_this_turn = -1
        with self.assertRaisesRegex(IllegalCommand, "follower_attacks_this_turn"):
            engine.assert_invariants()

    def test_rl_observations_expose_attack_history_without_action_id_migration(self):
        deck = [_card(99800 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=29,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=29)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        attacker = _ready_attacker(env.core, 0, 99850)
        action = env._encode_command(Attack(0, attacker.entity_id, None))
        self.assertEqual(ShadowverseEnv.ACTION_SIZE, 112)
        self.assertEqual(ShadowverseEnv.OBSERVATION_V1_SIZE, 304)
        self.assertTrue(env.action_mask()[action])
        self.assertEqual(env.observation()[22:24], [0.0, 0.0])

        result = env.step(action)

        self.assertEqual(
            len(result.observation),
            ShadowverseEnv.OBSERVATION_V1_SIZE,
        )
        self.assertEqual(result.observation[22:24], [0.1, 0.0])

        env.observation_version = "v2"
        structured = env.observation()
        self.assertEqual(
            len(structured["continuous_v1"]),
            ShadowverseEnv.OBSERVATION_V1_SIZE,
        )
        self.assertEqual(structured["continuous_v1"][22:24], (0.1, 0.0))


class AttackHistoryEmblemCountdownDatabaseAuditTests(unittest.TestCase):
    def test_database_text_alt_mode_and_reference_match_reviewed_chain(self):
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            source_text = connection.execute(
                "SELECT text_eng FROM skill_texts WHERE card_id=?",
                (SOURCE_ID,),
            ).fetchone()[0]
            token_text = connection.execute(
                "SELECT text_eng FROM skill_texts WHERE card_id=?",
                (TOKEN_ID,),
            ).fetchone()[0]
            crest_text = connection.execute(
                "SELECT text_eng FROM alt_modes WHERE card_id=? AND mode_type='纹章'",
                (SOURCE_ID,),
            ).fetchone()[0]
            self.assertIn("Torrent of Despair", re.sub(r"<[^>]+>", "", source_text))
            self.assertIn("Gain Crest", re.sub(r"<[^>]+>", "", source_text))
            self.assertIn("didn't attack this turn", crest_text)
            self.assertIn("number of crests", crest_text)
            self.assertIn("random enemy follower", token_text)
            self.assertIn("counts of all your crests by 1", token_text)
            self.assertEqual(
                connection.execute(
                    "SELECT referenced_card_id FROM card_references WHERE card_id=?",
                    (SOURCE_ID,),
                ).fetchall(),
                [(TOKEN_ID,)],
            )

    def test_both_cards_are_exact_and_token_has_executable_producer(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        for card_id in (SOURCE_ID, TOKEN_ID):
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                if card_id == SOURCE_ID:
                    self.assertEqual(info["coverage"], "covered_exact")
                    self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )

        audit = _build_token_audit("data/cards.sqlite3", "data/rules")
        token = next(card for card in audit["cards"] if card["card_id"] == TOKEN_ID)
        self.assertEqual(token["category"], "entry_behavior_complete")
        self.assertEqual(token["explicit_coverage"], "exact")
        self.assertEqual(
            token["authored_producers"],
            [{
                "source_card_id": SOURCE_ID,
                "entry_kind": "add_card",
                "rule_file": "real_attack_history_emblem_countdown_token_batch.json",
                "rule_group": "rules",
            }],
        )


if __name__ == "__main__":
    unittest.main()
