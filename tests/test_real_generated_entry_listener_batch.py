# -*- coding: utf-8 -*-
"""Exact generated entry-listener, hand-transform, and crest token chains."""

from __future__ import annotations

import unittest

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook
from swb.engine.commands import Choose, Evolve, PlayCard
from swb.engine.environment import ShadowverseEnv
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _choose,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


COLLECTIBLE_IDS = (10874120, 10773110, 10272110, 10264120)
TOKEN_IDS = (90071130, 90072130, 90064110)
SOURCE_HASHES = {
    10874120: "67300e982e599559b187afce4674c54bdb725a3edeb7003e22bbddecd1a72e2d",
    10773110: "0729a062e58bfe4b65f490e4b1cd5b9dfff0eeb5cdaa2d5582bf1510a1803d02",
    90071130: "12ea140ffd423deafdf28dab0d6d348160d3e1df0ed3cc75caa4b374df28b9d6",
    10272110: "950a6dd1f99569c7ce648fc3fc6e811228a8603ea3aa44418296c06bb6ae34b5",
    90072130: "d2292474b71a9be06a903f08368c98b491203c0d7394559e76ed1546fce82faa",
    10264120: "8847ee69d01cf4d3ef07217af3d15c72ebcf3ef015d8131c11dcc9c3e2ac1b40",
    90064110: "fd22b46c04522f13395a9cb2a5444e83007a05cf6121d94ba170126602ed58c5",
}


def _enable_evolution(engine) -> None:
    player = engine.players[0]
    player.turns_started = engine.config.evolution_unlock_turn
    player.evolution_points = max(1, player.evolution_points)
    player.evolved_this_turn = False


class RealGeneratedEntryListenerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 1301):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_orth_adds_analyzing_artifact_whose_own_entry_draws(self):
        engine = self.fresh(seed=3)
        source = _play(engine, self.repository, 10874120)
        artifact_card = next(
            card for card in engine.players[0].hand
            if card.card_id == 90071130
        )
        deck_before = len(engine.players[0].deck)
        engine.apply(
            PlayCard(0, engine.players[0].hand.index(artifact_card))
        )
        artifact = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90071130
        )
        self.assertIn(source, engine.players[0].board)
        self.assertEqual(len(engine.players[0].deck), deck_before - 1)
        self.assertTrue(any(
            event.type.value == "card_listener_triggered"
            and event.source_id == artifact.entity_id
            for event in engine.event_history
        ))

    def test_orth_evolution_selects_only_other_unevolved_follower(self):
        engine = self.fresh(seed=5)
        source = _play(engine, self.repository, 10874120)
        eligible = _put_unit(engine, 0, _card(998601, attack=2, life=3))
        already_evolved = _put_unit(engine, 0, _card(998602, life=4))
        already_evolved.evolved = True
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        request = engine.state.pending_choice
        self.assertEqual(
            [option.entity_id for option in request.options],
            [eligible.entity_id],
        )
        _choose(engine, eligible.entity_id)
        self.assertTrue(eligible.evolved)
        self.assertEqual((eligible.attack, eligible.max_health), (4, 5))
        self.assertTrue(already_evolved.evolved)

        no_target = self.fresh(seed=7)
        lone_source = _play(no_target, self.repository, 10874120)
        _enable_evolution(no_target)
        no_target.apply(Evolve(0, lone_source.entity_id))
        self.assertIsNone(no_target.state.pending_choice)
        self.assertTrue(lone_source.evolved)

    def test_wild_announcer_normal_and_enhance_outputs_draw_and_gain_rush(self):
        normal = self.fresh(seed=11)
        deck_before = len(normal.players[0].deck)
        source = _play(normal, self.repository, 10773110)
        artifact = next(
            unit for unit in normal.players[0].board
            if unit.definition.card_id == 90071130
        )
        self.assertFalse(source.has_keyword("突进"))
        self.assertTrue(artifact.has_keyword("突进"))
        self.assertEqual(len(normal.players[0].deck), deck_before - 1)

        enhanced = self.fresh(seed=13)
        source = _play(
            enhanced,
            self.repository,
            10773110,
            mode_id="enhance_5",
        )
        artifacts = {
            unit.definition.card_id: unit
            for unit in enhanced.players[0].board
            if unit.definition.card_id in {90071130, 90071150}
        }
        self.assertEqual(set(artifacts), {90071130, 90071150})
        self.assertTrue(all(unit.has_keyword("突进") for unit in artifacts.values()))
        self.assertTrue(artifacts[90071150].has_keyword("守护"))
        self.assertFalse(source.has_keyword("突进"))

    def test_wild_announcer_board_shortage_preserves_order(self):
        one_slot_after_source = self.fresh(seed=17)
        for index in range(3):
            _put_unit(one_slot_after_source, 0, _card(998610 + index))
        _play(
            one_slot_after_source,
            self.repository,
            10773110,
            mode_id="enhance_5",
        )
        self.assertTrue(any(
            unit.definition.card_id == 90071130
            for unit in one_slot_after_source.players[0].board
        ))
        self.assertFalse(any(
            unit.definition.card_id == 90071150
            for unit in one_slot_after_source.players[0].board
        ))

        full_after_source = self.fresh(seed=19)
        for index in range(4):
            _put_unit(full_after_source, 0, _card(998620 + index))
        deck_before = len(full_after_source.players[0].deck)
        _play(
            full_after_source,
            self.repository,
            10773110,
            mode_id="enhance_5",
        )
        self.assertFalse(any(
            unit.definition.card_id in {90071130, 90071150}
            for unit in full_after_source.players[0].board
        ))
        self.assertEqual(len(full_after_source.players[0].deck), deck_before)

    def test_wild_announcer_enhance_has_rl_mask_parity(self):
        env = ShadowverseEnv(
            [
                _card(998700 + index, class_id=7, class_name="超越者")
                for index in range(40)
            ],
            [
                _card(998800 + index, class_id=7, class_name="超越者")
                for index in range(40)
            ],
            class_a=7,
            class_b=7,
            seed=23,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=23)
        env.players[0].hand.clear()
        env.players[0].hand_entity_ids.clear()
        env.players[0].board.clear()
        _put_hand(env.core, self.repository.get(10773110))
        env.players[0].max_mana = env.players[0].mana = 5
        command = PlayCard(0, 0, mode_id="enhance_5")
        action = env._encode_command(command)
        self.assertIsNotNone(action)
        self.assertTrue(env.action_mask()[action])
        env.step(action)
        self.assertTrue(any(
            unit.definition.card_id == 90071150
            for unit in env.players[0].board
        ))

    def test_fia_transforms_only_selected_puppet_and_token_returns_fia(self):
        engine = self.fresh(seed=29)
        puppet = _put_hand(engine, self.repository.get(90071120))
        non_puppet = _put_hand(engine, self.repository.get(90071130))
        puppet_entity_id = puppet.entity_id
        source = _play(engine, self.repository, 10272110)
        self.assertTrue(source.has_keyword("必杀"))
        request = engine.state.pending_choice
        self.assertEqual(
            [option.entity_id for option in request.options],
            [puppet.entity_id],
        )
        _choose(engine, puppet.entity_id)
        self.assertEqual((puppet.card_id, puppet.entity_id), (90072130, puppet_entity_id))
        self.assertEqual(non_puppet.card_id, 90071130)

        engine.players[0].mana = 10
        engine.apply(PlayCard(0, engine.players[0].hand.index(puppet)))
        slaughter = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90072130
        )
        self.assertTrue(slaughter.has_keyword("潜行"))
        slaughter.health = 0
        engine._stabilize()
        fias = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 10272110
        ]
        self.assertEqual(len(fias), 2)

    def test_fia_without_puppet_skips_fanfare_but_remains_playable(self):
        engine = self.fresh(seed=31)
        _put_hand(engine, self.repository.get(90071130))
        source = _play(engine, self.repository, 10272110)
        self.assertIsNone(engine.state.pending_choice)
        self.assertTrue(source.has_keyword("必杀"))
        self.assertEqual(
            [card.card_id for card in engine.players[0].hand],
            [90071130],
        )

    def test_wilbert_crest_buffs_ward_entries_and_last_words_tokens(self):
        engine = self.fresh(seed=37)
        source = _play(engine, self.repository, 10264120)
        self.assertTrue(source.has_keyword("守护"))
        _enable_evolution(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertTrue(any(
            emblem.emblem_id == "weeping_paladin_wilbert"
            for emblem in engine.players[0].emblems
        ))

        ward = _put_hand(engine, self.repository.get(90064110))
        engine.players[0].mana = 10
        engine.apply(PlayCard(0, engine.players[0].hand.index(ward)))
        paladin = next(
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90064110
        )
        self.assertEqual((paladin.attack, paladin.max_health), (2, 4))
        plain = _put_unit(engine, 0, _card(998630, attack=2, life=3))
        self.assertEqual((plain.attack, plain.max_health), (2, 3))

        source.health = 0
        engine._stabilize()
        paladins = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90064110
        ]
        self.assertEqual(len(paladins), 3)
        self.assertTrue(all(unit.has_keyword("守护") for unit in paladins))
        self.assertTrue(all(
            (unit.attack, unit.max_health) == (2, 4)
            for unit in paladins
        ))

    def test_wilbert_without_crest_and_board_shortage_are_exact(self):
        engine = self.fresh(seed=41)
        source = _play(engine, self.repository, 10264120)
        for index in range(4):
            _put_unit(engine, 0, _card(998640 + index))
        source.health = 0
        engine._stabilize()
        paladins = [
            unit for unit in engine.players[0].board
            if unit.definition.card_id == 90064110
        ]
        self.assertEqual(len(paladins), 1)
        self.assertEqual((paladins[0].attack, paladins[0].max_health), (1, 2))
        self.assertTrue(paladins[0].has_keyword("守护"))

    def test_all_seven_cards_are_exact_and_tokens_have_producers(self):
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
