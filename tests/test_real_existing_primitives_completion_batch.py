# -*- coding: utf-8 -*-
"""Direct contracts for seven exact cards composed from existing primitives."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import ConditionType, EffectKind, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.resolution import IllegalCommand
from swb.engine.state import HandCard, Unit
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10052120,
    10153110,
    10211120,
    10504110,
    10812120,
    10813110,
    10861120,
)
SOURCE_HASHES = {
    10052120: "ab89fcd5f438770b024224a1ff1614bf4c0a27f526d5723e6318e7d2b77e4a78",
    10153110: "e09c6c96661b632bef332efd4e0434cc1cd21e12a629158b8395a32201c5e623",
    10211120: "f956037f1c90463dbe7f498cbb454e13c7c34b1f03543949a0820d6763297b68",
    10504110: "4b662efadf0987635bad729b66e1dfdc8df4c48e3bc4db63c25328d43260525a",
    10812120: "022fc17933bdab3eff29d0ae261f4456d7322f74986b52b7866efff6764d794d",
    10813110: "a4956c77c12a5bbce6926eb3152392b7cf635d2b3dbf2bb3c67ba7ad94e47e2a",
    10861120: "4590d34d1bd8efc2484f4b866f31da38ac6cc9c4a9ea785d61660d3f536c4ad8",
}


def _play(engine, repository: CardRepository, card_id: int) -> Unit:
    source = _put_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
    return next(
        unit
        for unit in engine.players[0].board
        if isinstance(unit, Unit) and unit.definition.card_id == card_id
    )


def _choose_label(engine, text: str) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if text in option.label)
    engine.apply(Choose(request.player_index, option.option_id))


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _enable_evolve(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = 2
    player.evolved_this_turn = False


def _enable_super_evolve(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.first_player_super_evolution_unlock_turn
    player.super_evolution_points = 2
    player.super_evolved_this_turn = False


class RealExistingPrimitivesCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 6101):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_and_keyword_provenance(self):
        mallet = self.rulebook.operations_for(10211120, Trigger.FANFARE)[0]
        self.assertIs(mallet.kind, EffectKind.CONDITIONAL)
        self.assertEqual(mallet.conditions[0].type, ConditionType.CONTROLLER_COMBO_AT_LEAST)
        self.assertEqual(
            (mallet.then_operations[0].target, mallet.else_operations[0].target),
            (TargetKind.ALL_ENEMY_UNITS, TargetKind.ENEMY_UNIT),
        )

        getenou = self.rulebook.operations_for(10504110, Trigger.FANFARE)
        self.assertEqual([operation.kind for operation in getenou], [EffectKind.DISCARD, EffectKind.CHOOSE_ONE])
        discount = getenou[1].choose_one_options[1].operations
        self.assertEqual(discount[0].target_key, "drawn_cards")
        self.assertIs(discount[1].target, TargetKind.PREVIOUS_TARGET)

        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10153110)), {"必杀"})
        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10812120)), {"突进", "必杀"})
        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10813110)), {"守护"})

    def test_malletman_selects_one_below_combo_and_stale_target_safely_skips(self):
        engine = self.fresh(seed=11)
        first = _put_unit(engine, 1, _card(991001, life=7))
        second = _put_unit(engine, 1, _card(991002, life=7))
        _play(engine, self.repository, 10211120)
        self.assertEqual(len(engine.state.pending_choice.options), 2)
        _choose_entity(engine, first.entity_id)
        self.assertEqual((first.health, second.health), (3, 7))

        stale = self.fresh(seed=13)
        target = _put_unit(stale, 1, _card(991003, life=7))
        _play(stale, self.repository, 10211120)
        stale.players[1].board.remove(target)
        _choose_entity(stale, target.entity_id)
        self.assertIsNone(stale.state.pending_choice)
        stale.assert_invariants()

        empty = self.fresh(seed=17)
        source = _play(empty, self.repository, 10211120)
        self.assertIn(source, empty.players[0].board)
        self.assertIsNone(empty.state.pending_choice)

    def test_malletman_combo_replaces_selection_with_simultaneous_area_damage(self):
        engine = self.fresh(seed=19)
        engine.players[0].cards_played_this_turn = 2
        enemies = [_put_unit(engine, 1, _card(991010 + index, life=4)) for index in range(3)]
        _play(engine, self.repository, 10211120)

        self.assertIsNone(engine.state.pending_choice)
        self.assertTrue(all(enemy not in engine.players[1].board for enemy in enemies))
        death_events = [event for event in engine.event_history if event.type is EventType.FOLLOWER_DESTROYED]
        self.assertGreaterEqual(len(death_events), 3)
        self.assertEqual(len(engine.players[1].graveyard), 3)

    def test_necromancer_normal_and_super_evolution_respect_capacity_and_outputs(self):
        normal = self.fresh(seed=23)
        source = _play(normal, self.repository, 10052120)
        _enable_evolve(normal)
        normal.apply(Evolve(0, source.entity_id))
        ghosts = [unit for unit in normal.players[0].board if unit.definition.card_id == 90051130]
        self.assertEqual(len(ghosts), 2)
        self.assertTrue(all(not ghost.has_keyword("虹吸") for ghost in ghosts))

        limited = self.fresh(seed=29)
        source = _play(limited, self.repository, 10052120)
        for index in range(3):
            _put_unit(limited, 0, _card(991020 + index))
        _enable_super_evolve(limited)
        limited.apply(SuperEvolve(0, source.entity_id))
        ghosts = [unit for unit in limited.players[0].board if unit.definition.card_id == 90051130]
        self.assertEqual(len(ghosts), 1)
        self.assertTrue(ghosts[0].has_keyword("虹吸"))
        self.assertEqual(len(limited.players[0].board), limited.config.max_board)

    def test_ceres_turn_end_uses_owner_scope_and_super_branch(self):
        normal = self.fresh(seed=31)
        normal.players[0].health = 10
        source = _play(normal, self.repository, 10153110)
        self.assertTrue(source.has_keyword("必杀"))
        normal.apply(EndTurn(0))
        self.assertEqual(normal.players[0].health, 12)
        self.assertFalse(source.has_keyword("屏障"))
        normal.players[0].health = 8
        normal.apply(EndTurn(1))
        self.assertEqual(normal.players[0].health, 8)

        evolved = self.fresh(seed=37)
        evolved.players[0].health = 10
        source = _play(evolved, self.repository, 10153110)
        _enable_super_evolve(evolved)
        evolved.apply(SuperEvolve(0, source.entity_id))
        evolved.apply(EndTurn(0))
        self.assertEqual(evolved.players[0].health, 14)
        self.assertTrue(source.has_keyword("屏障"))

    def test_lycoris_and_michelle_summon_each_other_without_recursive_fanfare(self):
        lycoris = self.fresh(seed=41)
        source = _play(lycoris, self.repository, 10812120)
        self.assertTrue(source.has_keyword("突进"))
        self.assertTrue(source.has_keyword("必杀"))
        summoned = [unit for unit in lycoris.players[0].board if unit.definition.card_id == 10813110]
        self.assertEqual(len(summoned), 1)
        self.assertTrue(summoned[0].has_keyword("守护"))
        self.assertFalse(summoned[0].has_keyword("屏障"))

        michelle = self.fresh(seed=43)
        ally = _put_unit(michelle, 0, _card(991030))
        source = _play(michelle, self.repository, 10813110)
        summoned = next(unit for unit in michelle.players[0].board if unit.definition.card_id == 10812120)
        self.assertTrue(all(unit.has_keyword("屏障") for unit in (ally, source, summoned)))
        self.assertEqual(len(michelle.players[0].board), 3)

        full = self.fresh(seed=47)
        existing = [_put_unit(full, 0, _card(991040 + index)) for index in range(4)]
        source = _play(full, self.repository, 10813110)
        self.assertEqual(len(full.players[0].board), full.config.max_board)
        self.assertFalse(any(unit.definition.card_id == 10812120 for unit in full.players[0].board))
        self.assertTrue(all(unit.has_keyword("屏障") for unit in existing + [source]))

    def test_theresa_modifies_only_the_summoned_sister_and_skips_failed_output(self):
        engine = self.fresh(seed=53)
        old_sister = _put_unit(engine, 0, self.repository.get(10061110))
        printed = self.repository.get(10061110)
        _play(engine, self.repository, 10861120)
        sisters = [unit for unit in engine.players[0].board if unit.definition.card_id == 10061110]
        new_sister = next(unit for unit in sisters if unit is not old_sister)
        self.assertEqual((old_sister.attack, old_sister.max_health), (printed.attack, printed.life))
        self.assertFalse(old_sister.has_keyword("疾驰"))
        self.assertEqual((new_sister.attack, new_sister.max_health), (printed.attack + 2, printed.life - 2))
        self.assertTrue(new_sister.has_keyword("疾驰"))

        full = self.fresh(seed=59)
        existing = [_put_unit(full, 0, _card(991050 + index)) for index in range(4)]
        _play(full, self.repository, 10861120)
        self.assertFalse(any(unit.definition.card_id == 10061110 for unit in full.players[0].board))
        self.assertTrue(all((unit.attack, unit.max_health) == (1, 5) for unit in existing))

    def test_getenou_discards_first_then_draws_eight(self):
        engine = self.fresh(seed=61)
        discarded = [_put_hand(engine, _card(991060 + index, card_type="法术", attack=None, life=None)) for index in range(3)]
        engine.players[0].deck = [_card(991100 + index, cost=3) for index in range(8)]
        _play(engine, self.repository, 10504110)
        self.assertFalse(engine.players[0].hand)
        self.assertEqual(len(engine.state.pending_choice.options), 2)
        _choose_label(engine, "抽取8张")
        self.assertEqual(len(engine.players[0].hand), 8)
        self.assertTrue(all(card.definition in [entry.definition for entry in engine.players[0].graveyard] for card in discarded))

    def test_getenou_discount_binds_both_drawn_cards_and_illegal_choice_rolls_back(self):
        engine = self.fresh(seed=67)
        engine.players[0].deck = [_card(991201, cost=5), _card(991202, cost=10)]
        _put_hand(engine, _card(991200, cost=2))
        _play(engine, self.repository, 10504110)
        before = (
            engine.deterministic_fingerprint(),
            engine.random.getstate(),
            tuple(engine.event_history),
            tuple(engine.logs),
        )
        with self.assertRaises(IllegalCommand):
            engine.apply(Choose(0, "choose_one:not_a_mode"))
        self.assertEqual(
            (
                engine.deterministic_fingerprint(),
                engine.random.getstate(),
                tuple(engine.event_history),
                tuple(engine.logs),
            ),
            before,
        )
        _choose_label(engine, "费用-8")
        self.assertEqual(len(engine.players[0].hand), 2)
        self.assertEqual(
            sorted((card.definition.cost, card.current_cost) for card in engine.players[0].hand),
            [(5, 0), (10, 2)],
        )

    def test_getenou_mode_action_mask_matches_commands_and_illegal_rl_action_preserves_core(self):
        deck = [_card(992000 + index, class_id=0, class_name="中立") for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=71,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=71)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        env.players[0].deck = [_card(992100, cost=9), _card(992101, cost=7)]
        _put_hand(env.core, self.repository.get(10504110))
        play_action = env._encode_command(PlayCard(0, 0))
        self.assertTrue(env.action_mask()[play_action])
        env.step(play_action)

        mask = env.action_mask()
        legal_actions = [index for index, allowed in enumerate(mask) if allowed]
        legal_commands = env.core.legal_commands()
        self.assertEqual(len(legal_actions), 2)
        self.assertEqual(
            {env._decode_action(action) for action in legal_actions},
            set(legal_commands),
        )
        self.assertFalse(mask[env.END_TURN])
        fingerprint = env.core.deterministic_fingerprint()
        with self.assertRaises(ValueError):
            env.step(env.END_TURN)
        self.assertEqual(env.core.deterministic_fingerprint(), fingerprint)

        discounted_action = next(
            action
            for action in legal_actions
            if "费用-8" in env.core.state.pending_choice.options[action - env.CHOICE_OFFSET].label
        )
        env.step(discounted_action)
        self.assertEqual(sorted(card.current_cost for card in env.players[0].hand), [0, 1])

    def test_seeded_real_sequence_is_reproducible(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=73)
            engine.players[0].cards_played_this_turn = 2
            for index in range(2):
                _put_unit(engine, 1, _card(992200 + index, life=6))
            _play(engine, self.repository, 10211120)
            outcomes.append((engine.deterministic_fingerprint(), tuple(engine.event_history)))
        self.assertEqual(outcomes[0], outcomes[1])


class ExistingPrimitivesCompletionAuditTests(unittest.TestCase):
    def test_database_multilingual_text_references_and_embedded_mode_are_reviewed(self):
        expected_phrases = {
            10052120: ("Summon 2 copies", "Super-Evolve", "Drain"),
            10153110: ("end of your turn", "restore 4 instead", "Barrier"),
            10211120: ("Combo", "all enemy followers instead"),
            10504110: ("Discard your hand", "Draw 8 cards", "reduce their costs by 8"),
            10812120: ("Michelle", "Rush", "Bane"),
            10813110: ("Lycoris", "all allied followers", "Ward"),
            10861120: ("Soulcure Sister", "+2/-2", "Storm"),
        }
        expected_references = {
            10052120: (90051130,),
            10153110: (),
            10211120: (),
            10504110: (),
            10812120: (10813110,),
            10813110: (10812120,),
            10861120: (10061110,),
        }
        with closing(sqlite3.connect("data/cards.sqlite3")) as connection:
            for card_id, phrases in expected_phrases.items():
                with self.subTest(card_id=card_id):
                    row = connection.execute(
                        "SELECT text_chs, text_cht, text_eng, text_jpn, text_kor FROM skill_texts WHERE card_id=?",
                        (card_id,),
                    ).fetchone()
                    self.assertIsNotNone(row)
                    self.assertTrue(all(row))
                    english = re.sub(r"<[^>]+>", "", row[2])
                    for phrase in phrases:
                        self.assertIn(phrase, english)
                    refs = tuple(
                        entry[0]
                        for entry in connection.execute(
                            "SELECT referenced_card_id FROM card_references WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    )
                    self.assertEqual(refs, expected_references[card_id])
                    self.assertEqual(
                        connection.execute("SELECT COUNT(*) FROM alt_modes WHERE card_id=?", (card_id,)).fetchone()[0],
                        0,
                    )

    def test_all_seven_cards_have_exact_clause_evidence_and_reports_remain_consistent(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        coverage_counts = report["summary"]["coverage_counts"]
        clause_counts = report["summary"]["clause_audit_counts"]
        self.assertEqual(coverage_counts["covered_exact"], clause_counts["mapped_exact"])
        self.assertEqual(coverage_counts["supported_missing_rule"], clause_counts["missing_rule"])
        self.assertEqual(sum(coverage_counts.values()), report["summary"]["total_cards"])
        self.assertFalse(report["rule_consistency_issues"])
        self.assertFalse(report["clause_audit_issues"])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                info = report["classifications"][str(card_id)]
                self.assertEqual(info["coverage"], "covered_exact")
                self.assertEqual(info["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(info["clause_audit"]["source_text_sha256"], SOURCE_HASHES[card_id])
                self.assertEqual(
                    info["clause_audit"]["test_evidence"],
                    ["tests/test_real_existing_primitives_completion_batch.py"],
                )

        token_report = _build_token_audit("data/cards.sqlite3", "data/rules")
        token_total = token_report["summary"]["total"]
        token_categories = token_report["summary"]["categories"]
        self.assertEqual(token_categories["entry_behavior_complete"], token_total)
        self.assertEqual(sum(token_categories.values()), token_total)
        self.assertTrue(all(card["category"] == "entry_behavior_complete" for card in token_report["cards"]))


if __name__ == "__main__":
    unittest.main()
