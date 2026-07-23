# -*- coding: utf-8 -*-
"""Direct contracts for the seventh keyed-crest and binding exact slice."""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger, _parse_operation
from swb.engine.commands import Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import Amulet, CostModifier
from tests.test_real_low_coverage_token_amulet_batch import (
    _card,
    _fresh,
    _play,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10333110,
    10432120,
    10453310,
    10454120,
    10521110,
    10734110,
    10744120,
)
SOURCE_HASHES = {
    10333110: "87d3d5b84f95289bcc3984d839d031a5c4309f85b303ea77b7495e53ede453da",
    10432120: "f1ece335bf36f94d490815bb3454dfecbe31a365b5a50ce8e0d30373a9bb2847",
    10453310: "708858f5fe413dcb368bb393b52c313c10e645f6b53c6c005a50cfed60b79f00",
    10454120: "fdcbaa6ab1b285f8bec0b7289953888dbe310119d0c143cd6853a592f60b4cf9",
    10521110: "a9432569ca6b127b3b9fa7dfb2b35c79a8553c6e6db2e65661d643c14925c20e",
    10734110: "3575fec154ba700653db55e0055528a236a8e755bd2389983c2a07e2605ab2d4",
    10744120: "364f2ca6add1f150a1aa320141756fee6f5641b6b457c1ef959cf14a1167a7eb",
}
TEST_EVIDENCE = "tests/test_real_keyed_crest_existing_seventh_batch.py"
RULE_FILE = "real_keyed_crest_existing_seventh_batch.json"


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


def _put_sigil(engine, repository: CardRepository, count: int) -> Amulet:
    sigil = Amulet(
        definition=repository.get(90031210),
        entity_id=engine.state.allocate_entity_id(),
        earth_sigil_count=count,
        entered_turn=engine.turn,
        origin=CardOrigin.TOKEN,
    )
    engine.players[0].board.append(sigil)
    return sigil


def _gain_emblem(engine, emblem_id: str, *, controller: int = 0) -> None:
    engine._start_effects(
        _card(
            998700 + controller,
            name="纹章测试来源",
            card_type="法术",
            attack=None,
            life=None,
        ),
        None,
        (
            EffectOperation(
                EffectKind.GAIN_EMBLEM,
                TargetKind.OWN_LEADER,
                emblem_id=emblem_id,
            ),
        ),
        controller=controller,
        label="获得测试纹章",
    )


class KeyedCrestExistingSeventhBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 7701):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def fresh_env(self, *, seed: int = 7801) -> ShadowverseEnv:
        deck = [_card(996500 + index) for index in range(40)]
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

    def test_rule_shapes_and_generic_schema_contracts(self):
        aristocrat = self.rulebook.operations_for(10521110, Trigger.FANFARE)
        self.assertIs(aristocrat[0].target, TargetKind.OWN_HAND)
        self.assertEqual(aristocrat[0].target_key, "altruistic_aristocrat_discard")
        self.assertEqual(
            aristocrat[1].condition_target_key,
            "altruistic_aristocrat_discard",
        )

        congregant = self.rulebook.operations_for(10333110, Trigger.FANFARE)[0]
        self.assertTrue(all(
            operation.kind is EffectKind.SUMMON_COPY
            and operation.target is TargetKind.SELF
            for operation in congregant.then_operations
        ))

        mireille = self.rulebook.operations_for(10432120, Trigger.FANFARE)
        self.assertEqual(mireille[0].target_key, "mireille_risette_summon")
        self.assertIs(
            mireille[1].earth_rite_operations[0].target,
            TargetKind.PREVIOUS_TARGET,
        )

        elder = self.rulebook.operations_for(10744120, Trigger.SUPER_EVOLVE)[0]
        belial = self.rulebook.operations_for(10454120, Trigger.SUPER_EVOLVE)[0]
        self.assertEqual(
            (elder.target, elder.emblem_id, elder.amount),
            (TargetKind.ALL_OWN_EMBLEMS, "dragons_vale_elder", 2),
        )
        self.assertEqual(
            (belial.target, belial.emblem_id, belial.amount),
            (TargetKind.ALL_OWN_EMBLEMS, "belial_archangel_of_cunning", 1),
        )
        self.assertFalse(
            self.rulebook.emblem_def("belial_archangel_of_cunning").on_expire
        )
        self.assertEqual(
            self.rulebook.emblem_def("belial_archangel_of_cunning")
            .last_words[0].kind,
            EffectKind.DAMAGE_LEADER,
        )

        with self.assertRaisesRegex(ValueError, "requires a specific emblem_id"):
            _parse_operation(
                {
                    "kind": "reduce_countdown",
                    "target": "all_own_emblems",
                    "amount": 1,
                },
                "test.json",
                1,
            )

        payload = {
            "rules": [{
                "card_id": 990001,
                "trigger": "play",
                "operations": [{
                    "kind": "earth_rite",
                    "target": "own_leader",
                    "amount": 1,
                    "operations": [{
                        "kind": "evolve_unit",
                        "target": "previous_target",
                        "target_key": "missing_outer_binding",
                    }],
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "bad.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "was not defined"):
                RuleBook.from_directory(directory)

    def test_aristocrat_binds_discarded_hand_snapshot_for_both_heal_branches(self):
        for card_type, expected_health in (("法术", 16), ("随从", 13)):
            with self.subTest(card_type=card_type):
                engine = self.fresh(seed=3)
                engine.players[0].health = 10
                source = _put_hand(engine, self.repository.get(10521110))
                discarded = _put_hand(
                    engine,
                    _card(
                        997000,
                        card_type=card_type,
                        attack=None if card_type == "法术" else 1,
                        life=None if card_type == "法术" else 3,
                    ),
                )
                engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
                _choose_entity(engine, discarded.entity_id)
                self.assertEqual(engine.players[0].health, expected_health)
                self.assertIn(
                    discarded.definition.card_id,
                    [card.definition.card_id for card in engine.players[0].graveyard],
                )
                engine.assert_invariants()

        capped = self.fresh(seed=4)
        capped.players[0].health = 19
        source = _put_hand(capped, self.repository.get(10521110))
        discarded = _put_hand(
            capped,
            _card(997001, card_type="法术", attack=None, life=None),
        )
        capped.apply(PlayCard(0, capped.players[0].hand.index(source)))
        _choose_entity(capped, discarded.entity_id)
        self.assertEqual(capped.players[0].health, 20)

        automatic = self.fresh(seed=4)
        automatic.players[0].health = 10
        _put_hand(
            automatic,
            _card(997002, card_type="法术", attack=None, life=None),
        )
        frame = automatic._queue_effects(
            self.repository.get(10521110),
            None,
            self.rulebook.operations_for(10521110, Trigger.FANFARE),
            controller=0,
            label="自动手牌目标绑定",
        )
        frame.auto_resolve_choices = True
        automatic._continue_effects()
        self.assertEqual(automatic.players[0].health, 16)
        self.assertFalse(automatic.players[0].hand)

    def test_aristocrat_no_target_stale_target_and_rl_choice_are_safe(self):
        no_target = self.fresh(seed=5)
        source = _put_hand(no_target, self.repository.get(10521110))
        command = PlayCard(0, no_target.players[0].hand.index(source))
        self.assertIn(command, no_target.legal_commands())
        no_target.apply(command)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertEqual(no_target.players[0].health, 20)

        stale = self.fresh(seed=7)
        stale.players[0].health = 10
        source = _put_hand(stale, self.repository.get(10521110))
        discarded = _put_hand(
            stale,
            _card(997010, card_type="法术", attack=None, life=None),
        )
        stale.apply(PlayCard(0, stale.players[0].hand.index(source)))
        stale.players[0].hand.remove(discarded)
        stale.players[0].hand_entity_ids.remove(discarded.entity_id)
        _choose_entity(stale, discarded.entity_id)
        self.assertEqual(stale.players[0].health, 10)
        self.assertIsNone(stale.state.pending_choice)
        stale.assert_invariants()

        env = self.fresh_env(seed=11)
        source = _put_hand(env.core, self.repository.get(10521110))
        discarded = _put_hand(
            env.core,
            _card(997011, card_type="法术", attack=None, life=None),
        )
        env.core.apply(PlayCard(0, env.players[0].hand.index(source)))
        env.invalidate_cache(reason="aristocrat pending hand target")
        decoded = {
            env._decode_action(index)
            for index, allowed in enumerate(env.action_mask())
            if allowed
        }
        self.assertEqual(decoded, set(env.core.legal_commands()))
        self.assertEqual(decoded, {Choose(0, f"hand:{discarded.entity_id}")})
        before = env.core.deterministic_fingerprint()
        with self.assertRaises(IllegalCommand):
            env.core.apply(Choose(0, "hand:999999"))
        self.assertEqual(env.core.deterministic_fingerprint(), before)

    def test_congregant_uses_physical_play_cost_and_copy_capacity(self):
        normal = self.fresh(seed=13)
        source = _put_hand(normal, self.repository.get(10333110))
        normal.apply(PlayCard(0, normal.players[0].hand.index(source)))
        self.assertEqual(len(normal.players[0].board), 1)
        self.assertTrue(normal.players[0].board[0].has_keyword("突进"))

        changed = self.fresh(seed=17)
        source = _put_hand(changed, self.repository.get(10333110))
        source.cost_modifiers.append(CostModifier(
            modifier_id=changed._allocate_modifier_id(),
            mode="set",
            amount=2,
            duration="permanent",
        ))
        changed.apply(PlayCard(0, changed.players[0].hand.index(source)))
        copies = [
            unit for unit in changed.players[0].board
            if unit.definition.card_id == 10333110
        ]
        self.assertEqual(len(copies), 3)
        self.assertTrue(all(unit.has_keyword("突进") for unit in copies))

        capped = self.fresh(seed=19)
        for index in range(3):
            _put_unit(capped, 0, _card(997020 + index))
        source = _put_hand(capped, self.repository.get(10333110))
        source.cost_modifiers.append(CostModifier(
            modifier_id=capped._allocate_modifier_id(),
            mode="set",
            amount=1,
            duration="permanent",
        ))
        capped.apply(PlayCard(0, capped.players[0].hand.index(source)))
        self.assertEqual(len(capped.players[0].board), capped.config.max_board)
        self.assertEqual(
            sum(unit.definition.card_id == 10333110 for unit in capped.players[0].board),
            2,
        )

        evolved = self.fresh(seed=23)
        source = _play(evolved, self.repository, 10333110)
        _enable_evolution(evolved, super_evolve=True)
        evolved.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(
            sum(unit.definition.card_id == 10333110 for unit in evolved.players[0].board),
            2,
        )

    def test_mireille_cross_scope_summon_binding_and_capacity(self):
        no_rite = self.fresh(seed=29)
        _play(no_rite, self.repository, 10432120)
        self.assertEqual(
            [(unit.definition.card_id, unit.evolved) for unit in no_rite.players[0].board],
            [(10432120, False), (10432120, False)],
        )

        rite = self.fresh(seed=31)
        _put_sigil(rite, self.repository, 2)
        _play(rite, self.repository, 10432120)
        duo = [
            unit for unit in rite.players[0].board
            if unit.definition.card_id == 10432120
        ]
        self.assertEqual(len(duo), 2)
        self.assertTrue(all(unit.evolved for unit in duo))
        self.assertEqual(rite.players[0].earth_sigils, 0)

        capped = self.fresh(seed=37)
        for index in range(3):
            _put_unit(capped, 0, _card(997030 + index))
        _put_sigil(capped, self.repository, 2)
        source = _play(capped, self.repository, 10432120)
        self.assertTrue(source.evolved)
        self.assertEqual(len(capped.players[0].board), 4)
        self.assertEqual(
            sum(unit.definition.card_id == 10432120 for unit in capped.players[0].board),
            1,
        )
        self.assertEqual(capped.players[0].earth_sigils, 0)
        capped.assert_invariants()

    def test_lilanthim_earth_rite_targeting_and_stale_choice(self):
        no_rite = self.fresh(seed=41)
        enemy = _put_unit(no_rite, 1, _card(997040))
        source = _play(no_rite, self.repository, 10734110)
        _enable_evolution(no_rite)
        no_rite.apply(Evolve(0, source.entity_id))
        self.assertIn(enemy, no_rite.players[1].board)

        targeted = self.fresh(seed=43)
        _put_sigil(targeted, self.repository, 2)
        enemy = _put_unit(targeted, 1, _card(997041))
        source = _play(targeted, self.repository, 10734110)
        self.assertEqual(
            [emblem.emblem_id for emblem in targeted.players[0].emblems],
            ["laralem_anathema_of_gluttony"],
        )
        self.assertTrue(source.has_keyword("灵气"))
        _enable_evolution(targeted)
        targeted.apply(Evolve(0, source.entity_id))
        _choose_entity(targeted, enemy.entity_id)
        self.assertNotIn(enemy, targeted.players[1].board)
        self.assertEqual(targeted.players[0].earth_sigils, 0)

        stale = self.fresh(seed=47)
        _put_sigil(stale, self.repository, 2)
        enemy = _put_unit(stale, 1, _card(997042))
        source = _play(stale, self.repository, 10734110)
        _enable_evolution(stale)
        stale.apply(Evolve(0, source.entity_id))
        stale.players[1].board.remove(enemy)
        _choose_entity(stale, enemy.entity_id)
        self.assertIsNone(stale.state.pending_choice)
        stale.assert_invariants()

    def test_lilanthim_crest_opponent_turn_end_output_binding_and_capacity(self):
        fingerprints = []
        for _ in range(2):
            engine = self.fresh(seed=53)
            _put_sigil(engine, self.repository, 1)
            _play(engine, self.repository, 10734110)
            engine.apply(EndTurn(0))
            engine.apply(EndTurn(1))
            summoned = [
                unit for unit in engine.players[0].board
                if unit.definition.card_id == 10734110
            ]
            self.assertEqual(len(summoned), 2)
            self.assertTrue(summoned[-1].evolved)
            self.assertFalse(engine.players[0].emblems)
            fingerprints.append(engine.deterministic_fingerprint())
        self.assertEqual(fingerprints[0], fingerprints[1])

        capped = self.fresh(seed=59)
        _put_sigil(capped, self.repository, 1)
        _play(capped, self.repository, 10734110)
        while len(capped.players[0].board) < capped.config.max_board:
            _put_unit(capped, 0, _card(997050 + len(capped.players[0].board)))
        capped.apply(EndTurn(0))
        capped.apply(EndTurn(1))
        self.assertEqual(
            sum(unit.definition.card_id == 10734110 for unit in capped.players[0].board),
            1,
        )
        capped.assert_invariants()

    def test_dragons_vale_token_turn_end_and_keyed_countdown_only(self):
        engine = self.fresh(seed=61)
        source = _play(engine, self.repository, 10744120)
        self.assertTrue(source.has_keyword("守护"))
        self.assertEqual(
            sum(unit.definition.card_id == 90041120 for unit in engine.players[0].board),
            1,
        )
        elder = engine.players[0].emblems[0]
        _gain_emblem(engine, "corruption")
        corruption = next(
            emblem for emblem in engine.players[0].emblems
            if emblem.emblem_id == "corruption"
        )
        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(elder.countdown, 4)
        self.assertEqual(corruption.countdown, 4)
        engine.apply(EndTurn(0))
        self.assertEqual(
            sum(unit.definition.card_id == 90041120 for unit in engine.players[0].board),
            2,
        )

        capped = self.fresh(seed=67)
        for index in range(4):
            _put_unit(capped, 0, _card(997060 + index))
        _play(capped, self.repository, 10744120)
        self.assertFalse(any(
            unit.definition.card_id == 90041120 for unit in capped.players[0].board
        ))
        capped.apply(EndTurn(0))
        self.assertEqual(len(capped.players[0].board), capped.config.max_board)

    def test_corruption_simultaneous_stats_owner_turn_damage_and_super_art(self):
        engine = self.fresh(seed=71)
        doomed = _put_unit(engine, 0, _card(997070, life=2))
        survivor = _put_unit(engine, 1, _card(997071, life=3))
        _play(engine, self.repository, 10453310)
        self.assertNotIn(doomed, engine.players[0].board)
        self.assertEqual(survivor.health, 1)
        self.assertEqual(
            [emblem.emblem_id for emblem in engine.players[0].emblems],
            ["corruption"],
        )
        self.assertEqual(
            [emblem.emblem_id for emblem in engine.players[1].emblems],
            ["corruption"],
        )
        engine.apply(EndTurn(0))
        self.assertEqual((engine.players[0].health, engine.players[1].health), (18, 20))
        engine.apply(EndTurn(1))
        self.assertEqual((engine.players[0].health, engine.players[1].health), (18, 18))

        burst = self.fresh(seed=73)
        burst.players[0].turns_started = 15
        _play(burst, self.repository, 10453310)
        self.assertFalse(burst.players[0].emblems)
        self.assertEqual(
            [emblem.emblem_id for emblem in burst.players[1].emblems],
            ["corruption"],
        )
        removed = [
            event for event in burst.event_history
            if event.type is EventType.EMBLEM_REMOVED
        ]
        self.assertEqual(removed[-1].metadata["cause"], "effect")

    def test_belial_damage_super_art_keyed_advance_and_crest_last_words(self):
        engine = self.fresh(seed=79)
        engine.players[0].turns_started = 15
        ally = _put_unit(engine, 0, _card(997080, life=10))
        enemy = _put_unit(engine, 1, _card(997081, life=10))
        source = _play(engine, self.repository, 10454120)
        self.assertNotIn(ally, engine.players[0].board)
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertIn(source, engine.players[0].board)
        belial = next(
            emblem for emblem in engine.players[0].emblems
            if emblem.emblem_id == "belial_archangel_of_cunning"
        )
        belial.countdown = 1
        _gain_emblem(engine, "corruption")
        corruption = next(
            emblem for emblem in engine.players[0].emblems
            if emblem.emblem_id == "corruption"
        )
        _enable_evolution(engine, super_evolve=True)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(engine.state.winner, 0)
        self.assertFalse(any(
            emblem.emblem_id == "belial_archangel_of_cunning"
            for emblem in engine.players[0].emblems
        ))
        self.assertEqual(corruption.countdown, 4)

        expire_only = self.fresh(seed=81)
        _gain_emblem(expire_only, "kagemitsu_enduring_warrior")
        expire_only._start_effects(
            _card(997089, card_type="法术", attack=None, life=None),
            None,
            (
                EffectOperation(
                    EffectKind.REMOVE_EMBLEM,
                    TargetKind.OWN_LEADER,
                    emblem_id="kagemitsu_enduring_warrior",
                ),
            ),
            controller=0,
            label="移除仅到期纹章",
        )
        self.assertFalse(any(
            unit.definition.card_id == 10124130
            for unit in expire_only.players[0].board
        ))

        destroyed = self.fresh(seed=83)
        _gain_emblem(destroyed, "belial_archangel_of_cunning")
        destroyed._start_effects(
            _card(997090, card_type="法术", attack=None, life=None),
            None,
            (
                EffectOperation(
                    EffectKind.REMOVE_EMBLEM,
                    TargetKind.OWN_LEADER,
                    emblem_id="belial_archangel_of_cunning",
                ),
            ),
            controller=0,
            label="破坏彼列纹章",
        )
        self.assertEqual(destroyed.state.winner, 0)
        removed_index = next(
            index for index, event in enumerate(destroyed.event_history)
            if event.type is EventType.EMBLEM_REMOVED
        )
        damage_index = next(
            index for index, event in enumerate(destroyed.event_history)
            if event.type is EventType.DAMAGE_APPLIED
        )
        self.assertLess(removed_index, damage_index)


class KeyedCrestExistingSeventhAuditTests(unittest.TestCase):
    def test_database_multilingual_text_modes_and_references_are_reviewed(self):
        expected_phrases = {
            10333110: ("cost isn't 3", "2 exact copies", "Super-Evolve"),
            10432120: ("Mireille & Risette", "Earth Rite (2)", "Evolve it"),
            10453310: ("-2/-2", "Crest: Corruption", "Super Skybound Art"),
            10454120: ("all other followers", "Super Skybound Art", "Advance the count"),
            10521110: ("card in your hand", "discard it", "spell"),
            10734110: ("Earth Rite (1)", "Aura", "destroy it"),
            10744120: ("Vastwing Dragon", "Ward", "Delay the count"),
        }
        expected_references = {
            10333110: (),
            10432120: (10432120,),
            10453310: (),
            10454120: (),
            10521110: (),
            10734110: (10734110,),
            10744120: (90041120,),
        }
        expected_modes = {
            10453310: ("Countdown (4)", "end of your turn", "2 damage"),
            10454120: ("Countdown (4)", "Last Words", "20 damage"),
            10734110: ("Countdown (1)", "opponent's turn", "evolve it"),
            10744120: ("Countdown (2)", "end of your turn", "Vastwing Dragon"),
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
                    english = "\n".join(
                        re.sub(r"<[^>]+>", "", row[2]) for row in rows
                    )
                    for phrase in phrases:
                        self.assertIn(phrase, english)
                    references = tuple(
                        row[0] for row in connection.execute(
                            "SELECT referenced_card_id FROM card_references "
                            "WHERE card_id=? ORDER BY position",
                            (card_id,),
                        )
                    )
                    self.assertEqual(references, expected_references[card_id])

            for card_id, phrases in expected_modes.items():
                rows = connection.execute(
                    "SELECT text_chs, text_cht, text_eng, text_jpn, text_kor "
                    "FROM alt_modes WHERE card_id=? ORDER BY position",
                    (card_id,),
                ).fetchall()
                self.assertEqual(len(rows), 1)
                self.assertTrue(all(rows[0]))
                english = re.sub(r"<[^>]+>", "", rows[0][2])
                for phrase in phrases:
                    self.assertIn(phrase, english)

    def test_coverage_clause_hashes_and_token_audit_are_exact(self):
        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        self.assertEqual(report["summary"]["coverage_counts"]["covered_exact"], 647)
        self.assertEqual(
            report["summary"]["coverage_counts"]["supported_missing_rule"],
            72,
        )
        self.assertFalse(report["rule_consistency_issues"])
        self.assertFalse(report["clause_audit_issues"])
        for card_id in CARD_IDS:
            with self.subTest(card_id=card_id):
                card = report["classifications"][str(card_id)]
                self.assertEqual(card["coverage"], "covered_exact")
                self.assertEqual(card["clause_audit"]["status"], "mapped_exact")
                self.assertEqual(
                    card["clause_audit"]["source_text_sha256"],
                    SOURCE_HASHES[card_id],
                )
                self.assertEqual(
                    card["clause_audit"]["test_evidence"],
                    [TEST_EVIDENCE],
                )
                self.assertTrue(all(
                    clause["mapping_status"] == "implemented"
                    for clause in card["clause_audit"]["source_clauses"]
                ))

        token = _build_token_audit("data/cards.sqlite3", "data/rules")
        categories = token["summary"]["categories"]
        self.assertEqual(categories["entry_behavior_complete"], 91)
        self.assertEqual(sum(categories.values()), 91)
        vastwing = next(card for card in token["cards"] if card["card_id"] == 90041120)
        self.assertIn(
            (10744120, "summon"),
            {
                (producer["source_card_id"], producer["entry_kind"])
                for producer in vastwing["authored_producers"]
                if producer["rule_file"] == RULE_FILE
            },
        )


if __name__ == "__main__":
    unittest.main()
