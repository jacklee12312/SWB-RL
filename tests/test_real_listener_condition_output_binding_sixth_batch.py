# -*- coding: utf-8 -*-
"""Direct contracts for the sixth exact listener/condition/output slice."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Attack, Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import ConditionType, EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.listeners import LISTENER_EVENT_TYPES
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import DeathCause
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _destroy_units,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10363110,
    10454110,
    10544120,
    10553110,
    10744110,
    10843110,
    10851130,
)
SOURCE_HASHES = {
    10363110: "dbddc8894b5bf8e8b58210e780714dfb5e94de478818cc0fb2ad96823465d5c9",
    10454110: "4ce06718c8798a47c7beaf2516260d87135695e339fa072fa4e5d128ac929c82",
    10544120: "abf4fc74c773b0acab2682ac51929ce7f6f9d873895af60bffb405b34fad21d4",
    10553110: "56d3c2febbebe1a28f50218f635c9acca5070d53f34e6dc281d169d5a0b653e9",
    10744110: "188e5cdd98e6e43262ac69791405a9515d399e5229d5981bb2bc2ea423d89c4b",
    10843110: "1c132af6507a49c2efea1b764805b1f105fc1676646abe6409aae12d6d2ef915",
    10851130: "5ec56f4e333c6e2db635d2f7ec3a55e30ec3bba778c5d033a703f21d89107c18",
}
TEST_EVIDENCE = "tests/test_real_listener_condition_output_binding_sixth_batch.py"
RULE_FILE = "real_listener_condition_output_binding_sixth_batch.json"


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolution(engine, *, super_evolve: bool = False) -> None:
    player = engine.players[0]
    if super_evolve:
        player.turns_started = engine.config.first_player_super_evolution_unlock_turn
        player.super_evolution_points = max(1, player.super_evolution_points)
        player.super_evolved_this_turn = False
    else:
        player.turns_started = engine.config.evolution_unlock_turn
        player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


def _heal_leader(engine, controller: int, amount: int) -> None:
    engine._start_effects(
        _card(
            998800 + controller,
            name="批次治疗来源",
            card_type="法术",
            attack=None,
            life=None,
        ),
        None,
        (
            EffectOperation(
                EffectKind.HEAL_LEADER,
                TargetKind.OWN_LEADER,
                amount=amount,
            ),
        ),
        controller=controller,
        label="批次治疗",
    )


class ListenerConditionOutputBindingSixthBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 7501):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def fresh_env(self, *, seed: int = 7601) -> ShadowverseEnv:
        deck = [_card(996000 + index) for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=seed,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=seed)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
            player.max_mana = player.mana = 10
        return env

    def test_rule_shapes_expose_generic_primitives_and_all_seven_cards(self):
        repose = self.rulebook.emblem_def("congregant_of_repose")
        draw = repose.triggers[0].operations[0]
        self.assertEqual(repose.countdown, 4)
        self.assertEqual((draw.deck_filter.life_min, draw.deck_filter.life_max), (4, 4))

        yube = self.rulebook.emblem_def("yube_crestpetal")
        self.assertEqual([trigger.trigger for trigger in yube.triggers], [
            "attack_declared",
            "attack_declared",
        ])
        self.assertIs(yube.triggers[0].operations[0].target, TargetKind.EVENT_SOURCE)
        self.assertTrue(yube.triggers[1].once_per_turn)

        fediel = self.rulebook.operations_for(10454110, Trigger.FANFARE)[0]
        self.assertEqual(
            [operation.target_key for operation in fediel.necromancy_operations],
            ["fediel_reanimate_two", "fediel_reanimate_two", "fediel_reanimate_one", "fediel_reanimate_one"],
        )
        self.assertEqual(
            self.rulebook.non_intrinsic_keywords(10843110),
            frozenset({"屏障"}),
        )
        giada_conditional = self.rulebook.operations_for(10843110, Trigger.ATTACK)[1]
        self.assertIs(giada_conditional.conditions[0].type, ConditionType.ATTACK_TARGET_EXISTS)
        limil_conditional = self.rulebook.operations_for(10851130, Trigger.FANFARE)[0]
        self.assertIs(
            limil_conditional.conditions[0].type,
            ConditionType.CONTROLLER_HEALTH_GREATER_THAN_OPPONENT,
        )
        self.assertIn(EventType.ATTACK_DECLARED, LISTENER_EVENT_TYPES)
        ruled = {
            card_id
            for card_id in CARD_IDS
            if any(self.rulebook.operations_for(card_id, trigger) for trigger in Trigger)
            or self.rulebook.listeners_for(card_id)
        }
        self.assertEqual(ruled, set(CARD_IDS))

        with self.assertRaisesRegex(ValueError, "life_min_filter"):
            _parse_operation(
                {
                    "kind": "draw_filtered",
                    "target": "own_leader",
                    "amount": 1,
                    "life_min_filter": 5,
                    "life_max_filter": 4,
                },
                "test.json",
                1,
            )
        with self.assertRaisesRegex(ValueError, "requires own_leader or enemy_leader"):
            _parse_operation(
                {
                    "kind": "gain_emblem",
                    "target": "self",
                    "emblem_id": "bad",
                },
                "test.json",
                1,
            )

    def test_congregant_targeting_crest_filter_attack_gate_and_no_target_path(self):
        engine = self.fresh(seed=3)
        enemy = _put_unit(engine, 1, _card(995001, attack=1, life=5))
        hand_card = _put_hand(engine, self.repository.get(10363110))
        command = PlayCard(0, engine.players[0].hand.index(hand_card))
        self.assertIn(command, engine.legal_commands())
        engine.apply(command)
        self.assertIsNotNone(engine.state.pending_choice)
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "entity:999999"))
        self.assertEqual(engine.deterministic_fingerprint(), before)
        _choose_entity(engine, enemy.entity_id)
        self.assertNotIn(enemy, engine.players[1].board)

        source = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10363110
        )
        engine.players[0].deck = [
            _card(995010, life=3),
            _card(995011, life=4),
            _card(995012, card_type="法术", attack=None, life=None),
        ]
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            [emblem.emblem_id for emblem in engine.players[0].emblems],
            ["congregant_of_repose"],
        )
        engine.apply(EndTurn(0))
        self.assertEqual([card.card_id for card in engine.players[0].hand], [995011])

        attacked = self.fresh(seed=5)
        attacked._add_emblem_to_player(
            0,
            self.rulebook.emblem_def("congregant_of_repose"),
            source_card_id=10363110,
        )
        attacked.players[0].deck = [_card(995020, life=4)]
        attacker = _put_unit(attacked, 0, _card(995021, attack=1, life=5))
        defender = _put_unit(attacked, 1, _card(995022, attack=0, life=5))
        attacker.can_attack = True
        attacker.rush_only = False
        attacked.apply(Attack(0, attacker.entity_id, defender.entity_id))
        attacked.apply(EndTurn(0))
        self.assertFalse(attacked.players[0].hand)

        no_target = self.fresh(seed=7)
        no_target_card = _put_hand(no_target, self.repository.get(10363110))
        no_target_command = PlayCard(0, no_target.players[0].hand.index(no_target_card))
        self.assertIn(no_target_command, no_target.legal_commands())
        no_target.apply(no_target_command)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertTrue(any(
            unit.definition.card_id == 10363110 for unit in no_target.players[0].board
        ))
        no_target.assert_invariants()

    def _setup_fediel(self, seed: int):
        engine = self.fresh(seed=seed)
        engine.players[0].shadows = 6
        for definition in (
            _card(995101, cost=2, attack=2, life=2),
            _card(995102, cost=2, attack=3, life=3),
            _card(995103, cost=1, attack=1, life=1),
        ):
            engine._record_destroyed_follower(0, definition, DeathCause.EFFECT_DESTROY)
        return engine

    def test_fediel_reanimate_bindings_capacity_no_candidate_turn_end_and_seed(self):
        engine = self._setup_fediel(11)
        source = _play(engine, self.repository, 10454110)
        outputs = [unit for unit in engine.players[0].board if unit is not source]
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(unit.evolved for unit in outputs))
        self.assertEqual({unit.definition.cost for unit in outputs}, {1, 2})
        self.assertEqual(engine.players[0].shadows, 0)

        first = self._setup_fediel(13)
        second = self._setup_fediel(13)
        _play(first, self.repository, 10454110)
        _play(second, self.repository, 10454110)
        self.assertEqual(first.deterministic_fingerprint(), second.deterministic_fingerprint())

        capped = self._setup_fediel(17)
        for index in range(3):
            _put_unit(capped, 0, _card(995110 + index))
        source = _play(capped, self.repository, 10454110)
        outputs = [
            unit for unit in capped.players[0].board
            if unit is not source and unit.definition.card_id in {995101, 995102, 995103}
        ]
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0].evolved)

        empty = self.fresh(seed=19)
        empty.players[0].shadows = 6
        _play(empty, self.repository, 10454110)
        self.assertEqual(empty.players[0].shadows, 0)
        self.assertEqual(len(empty.players[0].board), 1)

        debuff = self.fresh(seed=23)
        source = _play(debuff, self.repository, 10454110)
        enemy = _put_unit(debuff, 1, _card(995120, attack=5, life=5))
        debuff.apply(EndTurn(0))
        self.assertEqual((enemy.attack, enemy.health), (3, 3))

        departed = self.fresh(seed=29)
        source = _play(departed, self.repository, 10454110)
        enemy = _put_unit(departed, 1, _card(995121, attack=5, life=5))
        _destroy_units(departed, source)
        departed.apply(EndTurn(0))
        self.assertEqual((enemy.attack, enemy.health), (5, 5))
        departed.assert_invariants()

    def test_yube_discard_crest_attack_listener_once_capacity_and_departure(self):
        engine = self.fresh(seed=31)
        source = _play(engine, self.repository, 10544120)
        discard = _put_hand(engine, _card(995201, card_type="法术", attack=None, life=None))
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertIsNotNone(engine.state.pending_choice)
        before = engine.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "entity:999999"))
        self.assertEqual(engine.deterministic_fingerprint(), before)
        _choose_entity(engine, discard.entity_id)
        self.assertEqual(
            [emblem.emblem_id for emblem in engine.players[0].emblems],
            ["yube_crestpetal"],
        )
        _destroy_units(engine, source)

        marine = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90041130
        )
        defender = _put_unit(engine, 1, _card(995202, attack=0, life=30))
        base_attack = marine.attack
        marine.grant_attacks_per_turn(2)
        marine.summoned_this_turn = False
        marine.can_attack = True
        marine.rush_only = False
        marine.attacks_remaining = 2
        engine.apply(Attack(0, marine.entity_id, defender.entity_id))
        engine.apply(Attack(0, marine.entity_id, defender.entity_id))
        self.assertEqual(marine.attack, base_attack + 2)
        self.assertEqual(
            sum(card.card_id == 90041130 for card in engine.players[0].hand),
            1,
        )
        engine.apply(EndTurn(0))
        self.assertEqual(marine.attack, base_attack)

        capped = self.fresh(seed=37)
        capped._add_emblem_to_player(
            0,
            self.rulebook.emblem_def("yube_crestpetal"),
            source_card_id=10544120,
        )
        for index in range(capped.config.max_hand):
            _put_hand(capped, _card(995210 + index))
        marine = _put_unit(capped, 0, self.repository.get(90041130))
        marine.summoned_this_turn = False
        marine.can_attack = True
        marine.rush_only = False
        capped.apply(Attack(0, marine.entity_id, None))
        self.assertEqual(len(capped.players[0].hand), capped.config.max_hand)
        self.assertTrue(any(
            card.definition.card_id == 90041130 for card in capped.players[0].graveyard
        ))
        capped.assert_invariants()

    def test_lifestealer_transform_listener_evolve_and_simultaneous_source_death(self):
        engine = self.fresh(seed=41)
        _put_unit(engine, 0, _card(995301, attack=2, life=3))
        _put_unit(engine, 1, _card(995302, attack=2, life=3))
        _put_unit(engine, 1, _card(995303, attack=2, life=3))
        source = _play(engine, self.repository, 10553110)
        transformed = [
            unit for player in engine.players for unit in player.board if unit is not source
        ]
        self.assertEqual(len(transformed), 3)
        self.assertTrue(all(unit.definition.card_id == 90051110 for unit in transformed))
        engine.players[0].health = 10
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(engine.players[0].health, 13)
        self.assertTrue(all(
            unit.definition.card_id != 90051110
            for player in engine.players
            for unit in player.board
        ))

        simultaneous = self.fresh(seed=43)
        source = _play(simultaneous, self.repository, 10553110)
        skeleton = _put_unit(simultaneous, 0, self.repository.get(90051110))
        simultaneous.players[0].health = 10
        _destroy_units(simultaneous, source, skeleton)
        self.assertEqual(simultaneous.players[0].health, 10)
        simultaneous.assert_invariants()

    def test_burnite_enemy_crest_turn_start_and_once_per_turn_heal_listener(self):
        engine = self.fresh(seed=47)
        enemy = _put_unit(engine, 1, _card(995401, attack=1, life=12))
        source = _play(engine, self.repository, 10744110)
        self.assertEqual(enemy.health, 3)
        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertFalse(engine.players[0].emblems)
        self.assertEqual(
            [emblem.emblem_id for emblem in engine.players[1].emblems],
            ["burnite_anathema_of_ash"],
        )
        engine.apply(EndTurn(0))
        self.assertEqual(engine.current_player, 1)
        self.assertEqual(engine.players[1].health, 18)
        _heal_leader(engine, 1, 1)
        self.assertEqual(engine.players[1].health, 18)
        _heal_leader(engine, 1, 1)
        self.assertEqual(engine.players[1].health, 19)
        engine.assert_invariants()

    def test_giada_barrier_follower_attack_capacity_and_leader_attack_branch(self):
        engine = self.fresh(seed=53)
        source = _play(engine, self.repository, 10843110)
        self.assertTrue(source.has_keyword("突进"))
        self.assertTrue(source.has_keyword("威慑"))
        self.assertFalse(source.has_keyword("屏障"))
        defender = _put_unit(engine, 1, _card(995501, attack=0, life=30))
        engine.apply(Attack(0, source.entity_id, defender.entity_id))
        self.assertTrue(source.has_keyword("屏障"))
        self.assertEqual((source.attacks_per_turn, source.attacks_remaining), (2, 1))
        engine.apply(Attack(0, source.entity_id, defender.entity_id))
        self.assertEqual(source.attacks_remaining, 0)
        engine.apply(EndTurn(0))
        self.assertEqual(source.attacks_per_turn, 1)
        self.assertTrue(source.has_keyword("屏障"))

        leader = self.fresh(seed=59)
        source = _play(leader, self.repository, 10843110)
        source.summoned_this_turn = False
        source.can_attack = True
        source.rush_only = False
        source.attacks_remaining = 1
        leader.apply(Attack(0, source.entity_id, None))
        self.assertTrue(source.has_keyword("屏障"))
        self.assertEqual(source.attacks_per_turn, 1)
        leader.assert_invariants()

    def test_limil_health_comparison_evolve_replay_equal_and_board_capacity(self):
        engine = self.fresh(seed=61)
        engine.players[0].health = 20
        engine.players[1].health = 10
        source = _play(engine, self.repository, 10851130)
        self.assertEqual(
            sum(unit.definition.card_id == 90051120 for unit in engine.players[0].board),
            2,
        )
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual(
            sum(unit.definition.card_id == 90051120 for unit in engine.players[0].board),
            4,
        )

        equal = self.fresh(seed=67)
        equal.players[0].health = equal.players[1].health = 10
        _play(equal, self.repository, 10851130)
        self.assertFalse(any(
            unit.definition.card_id == 90051120 for unit in equal.players[0].board
        ))

        capped = self.fresh(seed=71)
        capped.players[0].health = 20
        capped.players[1].health = 10
        for index in range(3):
            _put_unit(capped, 0, _card(995510 + index))
        _play(capped, self.repository, 10851130)
        self.assertEqual(
            sum(unit.definition.card_id == 90051120 for unit in capped.players[0].board),
            1,
        )
        capped.assert_invariants()

    def test_action_masks_match_congregant_and_yube_pending_choices(self):
        congregant = self.fresh_env(seed=73)
        enemy = _put_unit(congregant.core, 1, _card(995601, attack=1, life=4))
        _put_hand(congregant.core, self.repository.get(10363110))
        play = PlayCard(0, 0)
        self.assertTrue(congregant.action_mask()[congregant._encode_command(play)])
        congregant.core.apply(play)
        congregant.invalidate_cache(reason="congregant pending target")
        decoded = {
            congregant._decode_action(index)
            for index, allowed in enumerate(congregant.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(congregant.core.legal_commands()))
        self.assertEqual(decoded, {Choose(0, f"entity:{enemy.entity_id}")})
        before = congregant.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            congregant.core.apply(Choose(0, "entity:999999"))
        self.assertEqual(congregant.core.deterministic_fingerprint(), before)

        yube = self.fresh_env(seed=79)
        source = _play(yube.core, self.repository, 10544120)
        discard = _put_hand(
            yube.core,
            _card(995602, card_type="法术", attack=None, life=None),
        )
        _enable_evolution(yube.core)
        yube.core.apply(Evolve(0, source.entity_id))
        yube.invalidate_cache(reason="yube pending discard")
        decoded = {
            yube._decode_action(index)
            for index, allowed in enumerate(yube.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(yube.core.legal_commands()))
        self.assertEqual(decoded, {Choose(0, f"hand:{discard.entity_id}")})


class ListenerConditionOutputBindingSixthAuditTests(unittest.TestCase):
    def test_database_multilingual_text_modes_and_references_are_reviewed(self):
        expected_phrases = {
            10363110: ("Select an enemy follower", "destroy it", "Evolve"),
            10454110: ("Necromancy", "Reanimate", "-2/-2"),
            10544120: ("Majestic Megalorca", "discard", "Crest"),
            10553110: ("Transform all other followers", "Skeleton", "Evolve"),
            10744110: ("9 damage", "Super-Evolve", "Crest"),
            10843110: ("Barrier", "2 times per turn", "Intimidate"),
            10851130: ("higher than", "2 copies", "Evolve"),
        }
        expected_references = {
            10363110: (),
            10454110: (),
            10544120: (90041130,),
            10553110: (90051110,),
            10744110: (),
            10843110: (),
            10851130: (90051120,),
        }
        expected_modes = {
            10363110: ("Countdown", "didn't attack", "4 defense"),
            10544120: ("Marine follower attacks", "+1/+0", "once"),
            10744110: ("start of your turn", "restored", "Once"),
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id, phrases in expected_phrases.items():
                with self.subTest(card_id=card_id):
                    rows = connection.execute(
                        "SELECT text_chs, text_cht, text_eng, text_jpn, text_kor "
                        "FROM skill_texts WHERE card_id=? ORDER BY position",
                        (card_id,),
                    ).fetchall()
                    self.assertTrue(rows)
                    self.assertTrue(all(all(row) for row in rows))
                    english = "\n".join(re.sub(r"<[^>]+>", "", row[2]) for row in rows)
                    for phrase in phrases:
                        self.assertIn(phrase, english)
                    references = tuple(
                        row[0]
                        for row in connection.execute(
                            "SELECT referenced_card_id FROM card_references "
                            "WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    )
                    self.assertEqual(references, expected_references[card_id])

            for card_id, phrases in expected_modes.items():
                mode_rows = connection.execute(
                    "SELECT text_chs, text_cht, text_eng, text_jpn, text_kor "
                    "FROM alt_modes WHERE card_id=? ORDER BY position",
                    (card_id,),
                ).fetchall()
                self.assertEqual(len(mode_rows), 1)
                self.assertTrue(all(mode_rows[0]))
                english = re.sub(r"<[^>]+>", "", mode_rows[0][2])
                for phrase in phrases:
                    self.assertIn(phrase, english)

    def test_coverage_clause_hashes_and_token_audit_are_exact(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["summary"]["coverage_counts"]["covered_exact"], 640)
        self.assertEqual(
            report["summary"]["coverage_counts"]["supported_missing_rule"],
            79,
        )
        self.assertFalse(report["rule_consistency_issues"])
        self.assertFalse(report["clause_audit_issues"])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    info["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(info["clause_audit"]["test_evidence"], [TEST_EVIDENCE])
                self.assertTrue(info["clause_audit"]["source_clauses"])
                self.assertTrue(all(
                    clause["mapping_status"] == "implemented"
                    for clause in info["clause_audit"]["source_clauses"]
                ))

        token_report = _build_token_audit("data/cards.sqlite3", "data/rules")
        categories = token_report["summary"]["categories"]
        self.assertEqual(categories["entry_behavior_complete"], 91)
        self.assertEqual(sum(categories.values()), 91)
        tokens = {entry["card_id"]: entry for entry in token_report["cards"]}
        expected = {
            90041130: {(10544120, "summon"), (10544120, "add_card")},
            90051110: {(10553110, "transform")},
            90051120: {(10851130, "summon")},
        }
        for token_id, pairs in expected.items():
            actual = {
                (producer["source_card_id"], producer["entry_kind"])
                for producer in tokens[token_id]["authored_producers"]
                if producer["rule_file"] == RULE_FILE
            }
            self.assertEqual(actual, pairs)


if __name__ == "__main__":
    unittest.main()
