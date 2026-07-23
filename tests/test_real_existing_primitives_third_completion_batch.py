# -*- coding: utf-8 -*-
"""Direct contracts for the third existing-primitive exact-card slice."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import Attack, Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import Amulet, HandCard
from tests.test_real_basic_existing_primitives_batch import _card, _fresh, _put_hand, _put_unit


CARD_IDS = (
    10234110,
    10234120,
    10352110,
    10353110,
    10534120,
    10544110,
    10564110,
    10764120,
    10824120,
    10854120,
)
SOURCE_HASHES = {
    10234110: "d07c1399dc81449a2c9973996a42917a4eb3b4b6ca61c8053adf180a8968d600",
    10234120: "f7c7bd02f65ca5c0a070bb7a1151d5f0857e22c97d54de14f7dd7652e7bdb79b",
    10352110: "1a766d0a4ccf1e3cbc61e0a5b15f6ad276896f31cd3bfc974bea176684b0c374",
    10353110: "d832231aa10967873f5ec29af76234007ed49d2ae1e6733c1c92c3018a5676f5",
    10534120: "e708299120d79a072156d63cdbb727d481d2db7789cbdbe9b96a5a3af466a824",
    10544110: "a05c3775dfc17662c22fe2ff9947bbcecf0c23db5f47eb8bbae654f845ab5561",
    10564110: "3809a45f13a3ed0208b914b1d6c48afd49e0436e69780f6b1b994dc8b0b420db",
    10764120: "9c77f424f7ad3d4e339a57af7443b742936dfd85067a4442de322bd507f7e9e3",
    10824120: "f2509623d6c89a960c9ecb6c669e341189a73b230ad4c8321b81a00bcf80c119",
    10854120: "df424d87ad7949b48c77e6be058a3e1a39ce5b35c03cf0a388cd81ef0a795b9e",
}


def _play(engine, repository: CardRepository, card_id: int):
    source = _put_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, engine.players[0].hand.index(source)))
    return next(
        (
            entity
            for entity in reversed(engine.players[0].board)
            if entity.definition.card_id == card_id
        ),
        None,
    )


def _put_hand_for(engine, owner: int, definition) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    engine.players[owner].hand.append(card)
    engine.players[owner].hand_entity_ids.append(card.entity_id)
    return card


def _put_amulet(engine, repository: CardRepository, *, sigils: int = 0) -> Amulet:
    entity_id = engine.state.allocate_entity_id()
    definition = (
        repository.get(90031210)
        if sigils
        else _card(
            997800 + entity_id,
            card_type="护符",
            attack=None,
            life=None,
        )
    )
    amulet = Amulet(
        definition=definition,
        entity_id=entity_id,
        earth_sigil_count=sigils,
        entered_turn=engine.turn,
        origin=CardOrigin.TOKEN,
    )
    engine.players[0].board.append(amulet)
    return amulet


def _choose_entity(engine, entity_id: int) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if option.entity_id == entity_id)
    engine.apply(Choose(request.player_index, option.option_id))


def _choose_label(engine, text: str) -> None:
    request = engine.state.pending_choice
    option = next(option for option in request.options if text in option.label)
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
    player.evolved_this_turn = False
    player.super_evolved_this_turn = False


class RealExistingPrimitivesThirdCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 7201):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_static_traits_filters_modes_and_listener(self):
        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10234110)), {"灵气"})
        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10544110)), {"守护"})
        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10564110)), {"守护"})
        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10764120)), {"守护", "灵气"})
        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10824120)), {"疾驰", "必杀"})
        self.assertEqual(self.rulebook.spellboost_cost_reduction(10534120), 1)

        lilanthim = self.rulebook.operations_for(10234110, Trigger.SUPER_EVOLVE)[0]
        self.assertEqual(lilanthim.target_count, 2)
        self.assertIs(lilanthim.target, TargetKind.ENEMY_UNIT)
        sofina = self.rulebook.operations_for(10564110, Trigger.FANFARE)[0]
        random_mode = sofina.choose_one_options[1].operations[0]
        self.assertTrue(random_mode.exclude_source)
        self.assertFalse(random_mode.board_filter.evolved)
        self.assertEqual(random_mode.board_filter.keyword, "守护")
        mars = self.rulebook.listeners_for(10824120)[0]
        self.assertEqual((mars.event_filter.card_type, mars.event_filter.tribe_name), ("随从", "士兵"))

    def test_lilanthim_last_words_earth_rite_and_multi_target_super_are_exact(self):
        engine = self.fresh(seed=11)
        _put_amulet(engine, self.repository, sigils=2)
        source = _play(engine, self.repository, 10234110)
        self.assertTrue(source.has_keyword("灵气"))
        engine._start_effects(
            source.definition,
            source.entity_id,
            (EffectOperation(EffectKind.DESTROY, TargetKind.SELF),),
            controller=0,
        )
        copies = [unit for unit in engine.players[0].board if unit.definition.card_id == 10234110]
        self.assertEqual(len(copies), 1)
        self.assertFalse(any(isinstance(card, Amulet) for card in engine.players[0].board))

        no_rite = self.fresh(seed=13)
        source = _play(no_rite, self.repository, 10234110)
        no_rite._start_effects(
            source.definition,
            source.entity_id,
            (EffectOperation(EffectKind.DESTROY, TargetKind.SELF),),
            controller=0,
        )
        self.assertFalse(any(unit.definition.card_id == 10234110 for unit in no_rite.players[0].board))

        multi = self.fresh(seed=17)
        targets = [_put_unit(multi, 1, _card(997010 + index, life=8)) for index in range(2)]
        source = _play(multi, self.repository, 10234110)
        _enable_super_evolve(multi)
        multi.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(multi.state.pending_choice.target_count, 2)
        choice = Choose(0, f"entity:{targets[0].entity_id}")
        multi.apply(choice)
        before = multi.deterministic_fingerprint()
        with self.assertRaisesRegex(IllegalCommand, "Choice option is invalid"):
            multi.apply(choice)
        self.assertEqual(multi.deterministic_fingerprint(), before)
        multi.players[1].board.remove(targets[0])
        multi._send_to_graveyard(
            1,
            targets[0].definition,
            "test_target_left_play",
            source_entity_id=targets[0].entity_id,
        )
        _choose_entity(multi, targets[1].entity_id)
        self.assertNotIn(targets[1], multi.players[1].board)
        multi.assert_invariants()

        shortage = self.fresh(seed=19)
        only = _put_unit(shortage, 1, _card(997020))
        source = _play(shortage, self.repository, 10234110)
        _enable_super_evolve(shortage)
        shortage.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(shortage.state.pending_choice.target_count, 1)
        _choose_entity(shortage, only.entity_id)
        self.assertNotIn(only, shortage.players[1].board)

    def test_norman_earth_rite_modes_repeat_on_evolve_and_skip_without_sigils(self):
        engine = self.fresh(seed=23)
        _put_amulet(engine, self.repository, sigils=1)
        source = _play(engine, self.repository, 10234120)
        _choose_label(engine, "守护者巨像")
        golem = next(unit for unit in engine.players[0].board if unit.definition.card_id == 90031120)
        self.assertTrue(golem.has_keyword("屏障"))

        engine.players[0].deck = [_card(997030 + index) for index in range(3)]
        _put_amulet(engine, self.repository, sigils=1)
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        _choose_label(engine, "抽取3张")
        self.assertEqual(len(engine.players[0].hand), 3)

        healed = self.fresh(seed=29)
        healed.players[0].health = 10
        _put_amulet(healed, self.repository, sigils=1)
        _play(healed, self.repository, 10234120)
        _choose_label(healed, "4点生命值")
        self.assertEqual(healed.players[0].health, 14)

        no_rite = self.fresh(seed=31)
        _play(no_rite, self.repository, 10234120)
        self.assertIsNone(no_rite.state.pending_choice)

    def test_supplicant_modes_repeat_on_evolve_and_random_choice_is_seeded(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=37)
            enemies = [_put_unit(engine, 1, _card(997040 + index, life=7)) for index in range(2)]
            source = _play(engine, self.repository, 10352110)
            _choose_label(engine, "随机1个")
            damaged = [unit.definition.card_id for unit in enemies if unit.health == 4]
            engine.players[0].health = 15
            _enable_evolve(engine)
            engine.apply(Evolve(0, source.entity_id))
            _choose_label(engine, "2点生命值")
            self.assertEqual(engine.players[0].health, 17)
            outcomes.append((damaged, engine.deterministic_fingerprint()))
        self.assertEqual(outcomes[0], outcomes[1])

        empty = self.fresh(seed=41)
        _play(empty, self.repository, 10352110)
        _choose_label(empty, "随机1个")
        self.assertIsNone(empty.state.pending_choice)

    def test_congregant_modes_bind_summoned_copy_repeat_and_handle_full_board(self):
        engine = self.fresh(seed=43)
        source = _play(engine, self.repository, 10353110)
        _choose_label(engine, "突进")
        copies = [unit for unit in engine.players[0].board if unit.definition.card_id == 10353110]
        self.assertEqual(len(copies), 2)
        self.assertTrue(all(unit.has_keyword("突进") for unit in copies))
        before = {unit.entity_id: (unit.attack, unit.max_health) for unit in copies}
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose_label(engine, "+2/+2")
        for unit in copies:
            old_attack, old_health = before[unit.entity_id]
            if unit is source:
                self.assertGreaterEqual(unit.attack, old_attack + 4)
                self.assertGreaterEqual(unit.max_health, old_health + 4)
            else:
                self.assertEqual((unit.attack, unit.max_health), (old_attack + 2, old_health + 2))

        full = self.fresh(seed=47)
        for index in range(4):
            _put_unit(full, 0, _card(997050 + index))
        source = _play(full, self.repository, 10353110)
        _choose_label(full, "守护")
        self.assertEqual(len(full.players[0].board), full.config.max_board)
        self.assertTrue(source.has_keyword("守护"))
        self.assertEqual(sum(unit.definition.card_id == 10353110 for unit in full.players[0].board), 1)

    def test_arow_spellboost_damage_transform_and_illegal_play_are_exact(self):
        no_target = self.fresh(seed=53)
        hand_card = _put_hand(no_target, self.repository.get(10534120))
        command = PlayCard(0, no_target.players[0].hand.index(hand_card))
        self.assertIn(command, no_target.legal_commands())
        no_target.apply(command)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertTrue(any(unit.definition.card_id == 10534120 for unit in no_target.players[0].board))

        engine = self.fresh(seed=59)
        hand_card = _put_hand(engine, self.repository.get(10534120))
        engine._start_effects(
            _card(997060, card_type="法术", attack=None, life=None),
            None,
            (EffectOperation(EffectKind.SPELLBOOST_HAND, TargetKind.ALL_OWN_HAND, 1),),
            controller=0,
        )
        self.assertEqual(hand_card.current_cost, 9)
        target = _put_unit(engine, 1, _card(997061, life=12))
        engine.apply(PlayCard(0, engine.players[0].hand.index(hand_card)))
        _choose_entity(engine, target.entity_id)
        self.assertEqual(target.health, 2)
        source = next(unit for unit in engine.players[0].board if unit.definition.card_id == 10534120)
        target_entity_id = target.entity_id
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        _choose_entity(engine, target.entity_id)
        transformed = next(unit for unit in engine.players[1].board if unit.entity_id == target_entity_id)
        self.assertEqual(transformed.definition.card_id, 90061130)
        self.assertNotEqual(transformed.entity_id, source.entity_id)

        stale = self.fresh(seed=61)
        other = _put_unit(stale, 1, _card(997062))
        source = _play(stale, self.repository, 10534120)
        _choose_entity(stale, other.entity_id)
        other = _put_unit(stale, 1, _card(997063))
        _enable_evolve(stale)
        stale.apply(Evolve(0, source.entity_id))
        stale.players[1].board.remove(other)
        _choose_entity(stale, other.entity_id)
        self.assertFalse(any(unit.definition.card_id == 90061130 for unit in stale.players[1].board))
        stale.assert_invariants()

    def test_ilantry_turn_end_branches_keywords_and_randomness(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=67)
            source = _play(engine, self.repository, 10544110)
            self.assertTrue(source.has_keyword("守护"))
            engine.players[0].health = 5
            enemies = [_put_unit(engine, 1, _card(997070 + index, life=9)) for index in range(3)]
            engine.apply(EndTurn(0))
            damaged = [unit.definition.card_id for unit in enemies if unit.health == 1]
            self.assertEqual(len(damaged), 2)
            self.assertEqual(engine.players[0].health, 13)
            outcomes.append((damaged, engine.deterministic_fingerprint()))
        self.assertEqual(outcomes[0], outcomes[1])

        evolved = self.fresh(seed=71)
        source = _play(evolved, self.repository, 10544110)
        _enable_evolve(evolved)
        evolved.apply(Evolve(0, source.entity_id))
        self.assertFalse(source.has_keyword("守护"))
        self.assertTrue(source.has_keyword("威慑"))
        evolved.players[0].health = 9
        evolved.apply(EndTurn(0))
        self.assertEqual((evolved.players[0].health, evolved.players[1].health), (9, 12))

    def test_sofina_modes_filter_random_ward_and_turn_end_kills_simultaneously(self):
        engine = self.fresh(seed=73)
        ward = _put_unit(engine, 0, _card(997080))
        ward.add_keyword("守护")
        plain = _put_unit(engine, 0, _card(997081))
        before = (ward.attack, ward.max_health)
        source = _play(engine, self.repository, 10564110)
        _choose_label(engine, "其他进化前守护")
        self.assertTrue(ward.evolved)
        self.assertGreater(ward.attack, before[0])
        self.assertGreater(ward.max_health, before[1])
        self.assertFalse(plain.evolved)
        self.assertFalse(source.evolved)

        no_target = self.fresh(seed=79)
        source = _play(no_target, self.repository, 10564110)
        _choose_label(no_target, "其他进化前守护")
        self.assertFalse(source.evolved)

        sweep = self.fresh(seed=83)
        allied = _put_unit(sweep, 0, _card(997082, life=1))
        enemy = _put_unit(sweep, 1, _card(997083, life=1))
        source = _play(sweep, self.repository, 10564110)
        _choose_label(sweep, "本随从进化")
        self.assertTrue(source.evolved)
        sweep.apply(EndTurn(0))
        self.assertNotIn(allied, sweep.players[0].board)
        self.assertNotIn(enemy, sweep.players[1].board)
        self.assertIn(source, sweep.players[0].board)
        sweep.assert_invariants()

    def test_inishia_banish_heal_threshold_no_target_and_super_repeat(self):
        engine = self.fresh(seed=89)
        for _ in range(3):
            _put_amulet(engine, self.repository)
        engine.players[0].health = 10
        target = _put_unit(engine, 1, _card(997090))
        source = _play(engine, self.repository, 10764120)
        self.assertTrue(source.has_keyword("守护"))
        self.assertTrue(source.has_keyword("灵气"))
        _choose_entity(engine, target.entity_id)
        self.assertIn(target.definition, engine.players[1].banished)
        self.assertEqual(engine.players[0].health, 13)

        second = _put_unit(engine, 1, _card(997091))
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        _choose_entity(engine, second.entity_id)
        self.assertIn(second.definition, engine.players[1].banished)
        self.assertEqual(engine.players[0].health, 16)

        no_target = self.fresh(seed=97)
        for _ in range(3):
            _put_amulet(no_target, self.repository)
        no_target.players[0].health = 10
        _play(no_target, self.repository, 10764120)
        self.assertIsNone(no_target.state.pending_choice)
        self.assertEqual(no_target.players[0].health, 13)

    def test_mars_summons_and_buffs_each_soldier_with_capacity_and_super_evolve(self):
        engine = self.fresh(seed=101)
        source = _play(engine, self.repository, 10824120)
        knights = [unit for unit in engine.players[0].board if unit.definition.card_id == 90021110]
        self.assertEqual(len(knights), 3)
        self.assertEqual(source.attack, self.repository.get(10824120).attack + 3)
        self.assertTrue(source.has_keyword("疾驰"))
        self.assertTrue(source.has_keyword("必杀"))
        self.assertTrue(all(unit.attack == unit.definition.attack + 2 for unit in knights))
        self.assertTrue(all(unit.has_keyword("突进") for unit in knights))

        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        knights = [unit for unit in engine.players[0].board if unit.definition.card_id == 90021110]
        self.assertEqual(len(knights), 4)
        self.assertEqual(len(engine.players[0].board), engine.config.max_board)
        self.assertEqual(source.attack, self.repository.get(10824120).attack + 7)

        capped = self.fresh(seed=103)
        for index in range(3):
            _put_unit(capped, 0, _card(997100 + index))
        source = _play(capped, self.repository, 10824120)
        self.assertEqual(sum(unit.definition.card_id == 90021110 for unit in capped.players[0].board), 1)
        self.assertEqual(source.attack, self.repository.get(10824120).attack + 1)

    def test_ceres_necromancy_filters_hand_clash_and_turn_end(self):
        engine = self.fresh(seed=107)
        abyss = _put_hand_for(engine, 0, _card(997110, class_id=5, class_name="梦魇", cost=4))
        neutral = _put_hand_for(engine, 0, _card(997111, class_id=0, class_name="中立", cost=4))
        engine.players[0].shadows = 20
        source = _play(engine, self.repository, 10854120)
        self.assertEqual(engine.players[0].shadows, 0)
        self.assertEqual((abyss.current_cost, neutral.current_cost), (2, 4))

        target = _put_unit(engine, 1, _card(997112, attack=0, life=10))
        source.summoned_this_turn = False
        source.can_attack = True
        engine.apply(Attack(0, source.entity_id, target.entity_id))
        self.assertEqual(target.health, 5)
        engine.players[0].health = 12
        engine.apply(EndTurn(0))
        self.assertEqual(engine.players[0].health, 16)

        insufficient = self.fresh(seed=109)
        abyss = _put_hand_for(insufficient, 0, _card(997113, class_id=5, class_name="梦魇", cost=4))
        _play(insufficient, self.repository, 10854120)
        self.assertEqual(abyss.current_cost, 4)

        removed = self.fresh(seed=113)
        source = _play(removed, self.repository, 10854120)
        removed.players[0].health = 10
        removed._start_effects(
            source.definition,
            source.entity_id,
            (EffectOperation(EffectKind.DESTROY, TargetKind.SELF),),
            controller=0,
        )
        removed.apply(EndTurn(0))
        self.assertEqual(removed.players[0].health, 10)

    def test_mode_and_target_action_masks_match_commands_and_illegal_actions_are_atomic(self):
        deck = [_card(997200 + index, class_id=0, class_name="中立") for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=127,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=127)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        _put_hand(env.core, self.repository.get(10352110))
        env.core.apply(PlayCard(0, 0))
        mask = env.action_mask()
        legal = env.core.legal_commands()
        actions = [index for index, allowed in enumerate(mask) if allowed]
        self.assertEqual({env._decode_action(action) for action in actions}, set(legal))
        self.assertEqual(len(actions), 2)
        fingerprint = env.core.deterministic_fingerprint()
        with self.assertRaises(ValueError):
            env.step(env.END_TURN)
        self.assertEqual(env.core.deterministic_fingerprint(), fingerprint)

        target_env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=131,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        target_env.reset(seed=131)
        for player in target_env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        target_env.players[0].mana = target_env.players[0].max_mana = 10
        target = _put_unit(target_env.core, 1, _card(997300, life=12))
        _put_hand(target_env.core, self.repository.get(10534120))
        play = PlayCard(0, 0)
        self.assertIn(play, target_env.core.legal_commands())
        self.assertTrue(target_env.action_mask()[target_env._encode_command(play)])
        target_env.core.apply(play)
        choices = [index for index, allowed in enumerate(target_env.action_mask()) if allowed]
        self.assertEqual(len(choices), 1)
        self.assertEqual(target_env._decode_action(choices[0]).option_id, f"entity:{target.entity_id}")


class ExistingPrimitivesThirdCompletionAuditTests(unittest.TestCase):
    def test_database_multilingual_text_references_and_modes_are_reviewed(self):
        expected_phrases = {
            10234110: ("Aura", "Earth Rite", "Select 2 enemy followers"),
            10234120: ("Select a Mode", "Guardian Golem", "Replicate"),
            10352110: ("random enemy follower", "Restore 2 defense", "Replicate"),
            10353110: ("Congregant of Entwining", "Rush", "Super-Evolve"),
            10534120: ("On Spellboost", "deal it 10 damage", "Regal Falcon"),
            10544110: ("2 random enemy followers", "Intimidate", "enemy leader"),
            10564110: ("random unevolved allied follower", "Ward", "all other followers"),
            10764120: ("banish it", "at least 3 allied amulets", "Aura"),
            10824120: ("Summon 3 copies", "Officer follower", "Knight"),
            10854120: ("Necromancy", "all Abysscraft cards", "Clash"),
        }
        expected_references = {
            10234110: (10234110,),
            10234120: (90031120,),
            10352110: (),
            10353110: (10353110,),
            10534120: (90061130,),
            10544110: (),
            10564110: (),
            10764120: (),
            10824120: (90021110,),
            10854120: (),
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
                    self.assertFalse(
                        connection.execute(
                            "SELECT 1 FROM alt_modes WHERE card_id=?",
                            (card_id,),
                        ).fetchall()
                    )

    def test_all_ten_cards_have_exact_clause_evidence_and_token_audit_stays_complete(self):
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
                    ["tests/test_real_existing_primitives_third_completion_batch.py"],
                )

        token_report = _build_token_audit("data/cards.sqlite3", "data/rules")
        token_total = token_report["summary"]["total"]
        token_categories = token_report["summary"]["categories"]
        self.assertEqual(token_categories["entry_behavior_complete"], token_total)
        self.assertEqual(sum(token_categories.values()), token_total)
        self.assertTrue(all(card["category"] == "entry_behavior_complete" for card in token_report["cards"]))


if __name__ == "__main__":
    unittest.main()
