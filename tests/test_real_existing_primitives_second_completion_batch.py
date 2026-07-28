# -*- coding: utf-8 -*-
"""Direct contracts for the second existing-primitive exact-card slice."""

from __future__ import annotations

import re
import sqlite3
import unittest
from contextlib import closing

from scripts.report_rule_coverage import _build_coverage_report
from scripts.report_token_audit import _build_token_audit
from swb.db.repository import CardRepository
from swb.engine.card_rules import RuleBook, Trigger
from swb.engine.commands import (
    ActivateAmulet,
    Attack,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
    SuperEvolve,
)
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import IllegalCommand
from swb.engine.state import (
    Amulet,
    AttackRestriction,
    DeathCause,
    DestroyedFollowerRecord,
    HandCard,
    Unit,
)
from tests.test_real_basic_existing_primitives_batch import (
    _card,
    _fresh,
    _put_hand,
    _put_unit,
)


CARD_IDS = (
    10113130,
    10133130,
    10272120,
    10351110,
    10461210,
    10523110,
    10654110,
    10733110,
    10734120,
    10803110,
    10842110,
    10844120,
)
SOURCE_HASHES = {
    10113130: "f272c627f88f6f0786626abf93103c62f990e47b26039f176d83ed6f4db6943f",
    10133130: "7f3e57aab85ed5d57e1f77fb4dcb95e2a4e88f2b3f9c4d53272ed0c0b0bad1ec",
    10272120: "bda4ad0a050b32dc94aef7bbe925c80d850c19444769dda75da7a3afa7701c51",
    10351110: "2dbcbc478e569ee4d965ae9399a5df7294bf9f125da26e329f4a6e38dc3cd2b0",
    10461210: "41928ef92875f4823d1d2aedb15dbeb27c4cc6cf38f9ac393a47b42dfa8b1a0b",
    10523110: "b09eda2cc334cb901b2b84dd8ffb9a1b1dd77e1e058ad6edefc0538bea90e387",
    10654110: "a66db86d0d27011134d0708ddc88348d2fe69be1378f54d1ba0d569cdffde370",
    10733110: "88bbe9ddb0333abbf0964695ab325313742cb5c763613a62bfe96943d25fa663",
    10734120: "23916ea38e8ae3496f179d840e068a828268ab0be14a8a34834f64fbed7915f6",
    10803110: "0540d93eca308f58dfe315bf79fb298d3854584bbbfa6ffa61c361bbaebe9a02",
    10842110: "677b3d019705d4197cfd7bcea09877e1b3fe584fd04fffce63fff0af39626f24",
    10844120: "36d247cc628bb3b1a1ca01d422d2a6cfc9fb15b0bb6cbd6381ca0c7e7e078b69",
}


def _play(engine, repository: CardRepository, card_id: int, *, mode_id: str = "normal"):
    source = _put_hand(engine, repository.get(card_id))
    engine.apply(PlayCard(0, engine.players[0].hand.index(source), mode_id=mode_id))
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
    player.super_evolved_this_turn = False


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


def _destroyed(definition, sequence: int) -> DestroyedFollowerRecord:
    return DestroyedFollowerRecord(
        definition=definition,
        owner=0,
        death_sequence=sequence,
        cause=DeathCause.EFFECT_DESTROY,
    )


class RealExistingPrimitivesSecondCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rulebook = RuleBook.from_directory("data/rules")
        cls.repository = CardRepository("data/cards.sqlite3")

    def fresh(self, *, seed: int = 6201):
        return _fresh(self.rulebook, self.repository, seed=seed)

    def test_rule_shapes_cover_triggers_modes_activation_and_static_traits(self):
        akhim = self.rulebook.operations_for(10272120, Trigger.EVOLVE)
        self.assertEqual([operation.kind for operation in akhim], [EffectKind.BANISH, EffectKind.SUMMON_COPY])
        self.assertEqual(akhim[0].conditions[0].value, 4)
        self.assertEqual(akhim[0].target_key, "banished_follower")
        self.assertIs(akhim[1].target, TargetKind.PREVIOUS_TARGET)

        self.assertEqual(self.rulebook.activation_for(10461210).cost, 2)
        modes = self.rulebook.modes_for(10844120)
        self.assertEqual([(mode.mode_id, mode.mode_type, mode.cost) for mode in modes], [("accelerate_3", "accelerate", 3)])
        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10654110)), {"灵气"})
        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10734120)), {"守护"})
        self.assertEqual(set(self.rulebook.intrinsic_keywords_for(10842110)), {"突进"})
        self.assertTrue(self.rulebook.cannot_be_destroyed_by_effects(10654110))

    def test_belle_hand_listener_counts_only_allied_followers_leaving_then_fanfare_targets(self):
        engine = self.fresh(seed=11)
        belle = _put_hand(engine, self.repository.get(10113130))
        allies = [_put_unit(engine, 0, _card(993000 + index)) for index in range(2)]
        enemy = _put_unit(engine, 1, _card(993010))
        engine._start_effects(
            _card(999001, card_type="法术", attack=None, life=None),
            None,
            (EffectOperation(EffectKind.DESTROY, TargetKind.ALL_OWN_UNITS),),
            controller=0,
        )
        self.assertTrue(all(unit not in engine.players[0].board for unit in allies))
        self.assertEqual(belle.current_cost, 6)

        engine._start_effects(
            _card(999002, card_type="法术", attack=None, life=None),
            None,
            (EffectOperation(EffectKind.DESTROY, TargetKind.ALL_ENEMY_UNITS),),
            controller=0,
        )
        self.assertNotIn(enemy, engine.players[1].board)
        self.assertEqual(belle.current_cost, 6)

        target = _put_unit(engine, 1, _card(993011, life=8))
        engine.apply(PlayCard(0, engine.players[0].hand.index(belle)))
        _choose_entity(engine, target.entity_id)
        self.assertEqual(target.health, 4)

    def test_edelweiss_earth_rite_evolves_then_random_damage_and_restores_mana(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=17)
            engine.players[0].mana = 4
            sigil = _put_sigil(engine, self.repository, 2)
            enemies = [_put_unit(engine, 1, _card(993020 + index, life=8)) for index in range(2)]
            source = _play(engine, self.repository, 10133130)
            self.assertTrue(source.evolved)
            self.assertNotIn(sigil, engine.players[0].board)
            self.assertEqual(engine.players[0].mana, 2)
            damaged = [unit.definition.card_id for unit in enemies if unit.health == 4]
            self.assertEqual(len(damaged), 1)
            outcomes.append((damaged[0], engine.deterministic_fingerprint()))
        self.assertEqual(outcomes[0], outcomes[1])

        no_rite = self.fresh(seed=19)
        source = _play(no_rite, self.repository, 10133130)
        self.assertFalse(source.evolved)
        self.assertEqual(no_rite.players[0].mana, 6)

        empty = self.fresh(seed=23)
        empty.players[0].mana = 4
        _put_sigil(empty, self.repository, 2)
        source = _play(empty, self.repository, 10133130)
        self.assertTrue(source.evolved)
        self.assertEqual(empty.players[0].mana, 2)

    def test_akhim_filters_attack_banishes_then_summons_base_copy_and_handles_stale_target(self):
        engine = self.fresh(seed=29)
        low = _put_unit(engine, 1, _card(993030, attack=4, life=7))
        high = _put_unit(engine, 1, _card(993031, attack=5, life=7))
        source = _play(engine, self.repository, 10272120)
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        self.assertEqual([option.entity_id for option in engine.state.pending_choice.options], [low.entity_id])
        _choose_entity(engine, low.entity_id)
        self.assertIn(low.definition, engine.players[1].banished)
        self.assertIn(high, engine.players[1].board)
        copied = next(unit for unit in engine.players[0].board if unit.definition.card_id == low.definition.card_id)
        self.assertEqual((copied.attack, copied.max_health), (4, 7))

        stale = self.fresh(seed=31)
        target = _put_unit(stale, 1, _card(993032, attack=4))
        source = _play(stale, self.repository, 10272120)
        _enable_evolve(stale)
        stale.apply(Evolve(0, source.entity_id))
        stale.players[1].board.remove(target)
        _choose_entity(stale, target.entity_id)
        self.assertFalse(any(unit.definition.card_id == target.definition.card_id for unit in stale.players[0].board))
        stale.assert_invariants()

        no_target = self.fresh(seed=37)
        _put_unit(no_target, 1, _card(993033, attack=5))
        source = _play(no_target, self.repository, 10272120)
        _enable_evolve(no_target)
        no_target.apply(Evolve(0, source.entity_id))
        self.assertTrue(source.evolved)
        self.assertIsNone(no_target.state.pending_choice)

    def test_affirmator_repeats_fanfare_mode_on_evolve(self):
        engine = self.fresh(seed=41)
        enemy = _put_unit(engine, 1, _card(993040, life=6))
        source = _play(engine, self.repository, 10351110)
        _choose_label(engine, "+2/+2")
        definition = self.repository.get(10351110)
        self.assertEqual((source.attack, source.max_health), (definition.attack + 2, definition.life + 2))
        self.assertEqual(enemy.health, 6)

        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        _choose_label(engine, "造成1点")
        self.assertEqual(enemy.health, 5)

    def test_liliales_engage_is_atomic_transforms_target_and_continues_after_stale_target(self):
        illegal = self.fresh(seed=43)
        amulet = _play(illegal, self.repository, 10461210)
        command = ActivateAmulet(0, amulet.entity_id)
        before = (
            illegal.deterministic_fingerprint(),
            illegal.random.getstate(),
            tuple(illegal.event_history),
            tuple(illegal.logs),
        )
        with self.assertRaises(IllegalCommand):
            illegal.apply(command)
        self.assertEqual(
            (
                illegal.deterministic_fingerprint(),
                illegal.random.getstate(),
                tuple(illegal.event_history),
                tuple(illegal.logs),
            ),
            before,
        )

        engine = self.fresh(seed=47)
        engine.players[0].deck = [_card(993051)]
        target = _put_unit(engine, 0, _card(993050))
        amulet = _play(engine, self.repository, 10461210)
        mana_before = engine.players[0].mana
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertNotIn(amulet, engine.players[0].board)
        _choose_entity(engine, target.entity_id)
        transformed = next(card for card in engine.players[0].board if card.definition.card_id == 10461210)
        self.assertIsInstance(transformed, Amulet)
        self.assertNotEqual(transformed.entity_id, amulet.entity_id)
        self.assertEqual(engine.players[0].mana, mana_before - 2)
        self.assertEqual([card.card_id for card in engine.players[0].hand], [993051])

        stale = self.fresh(seed=53)
        stale.players[0].deck = [_card(993053)]
        target = _put_unit(stale, 0, _card(993052))
        amulet = _play(stale, self.repository, 10461210)
        stale.apply(ActivateAmulet(0, amulet.entity_id))
        stale.players[0].board.remove(target)
        _choose_entity(stale, target.entity_id)
        self.assertEqual([card.card_id for card in stale.players[0].hand], [993053])
        self.assertIsNone(stale.state.pending_choice)

    def test_officer_cannot_attack_on_any_entry_and_owner_turn_end_summons_knight(self):
        engine = self.fresh(seed=59)
        source = _play(engine, self.repository, 10523110)
        self.assertIn(AttackRestriction.CANNOT_ATTACK, {entry.restriction for entry in source.attack_restrictions})
        self.assertFalse(any(isinstance(command, Attack) and command.attacker_id == source.entity_id for command in engine.legal_commands()))

        summoned = self.fresh(seed=61)
        summoned._start_effects(
            _card(999003, card_type="法术", attack=None, life=None),
            None,
            (EffectOperation(EffectKind.SUMMON, TargetKind.OWN_LEADER, card_id=10523110),),
            controller=0,
        )
        source = next(unit for unit in summoned.players[0].board if unit.definition.card_id == 10523110)
        self.assertIn(AttackRestriction.CANNOT_ATTACK, {entry.restriction for entry in source.attack_restrictions})

        engine.apply(EndTurn(0))
        self.assertEqual(sum(unit.definition.card_id == 90021120 for unit in engine.players[0].board), 1)
        engine.apply(EndTurn(1))
        self.assertEqual(sum(unit.definition.card_id == 90021120 for unit in engine.players[0].board), 1)

    def test_officer_super_evolve_buffs_only_others_and_capacity_is_respected(self):
        engine = self.fresh(seed=67)
        ally = _put_unit(engine, 0, _card(993060))
        source = _play(engine, self.repository, 10523110)
        printed = self.repository.get(10523110)
        _enable_super_evolve(engine)
        engine.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual((ally.attack, ally.max_health), (4, 8))
        self.assertEqual((source.attack, source.max_health), (printed.attack + 3, printed.life + 3))

        full = self.fresh(seed=71)
        for index in range(4):
            _put_unit(full, 0, _card(993070 + index))
        _play(full, self.repository, 10523110)
        full.apply(EndTurn(0))
        self.assertFalse(any(unit.definition.card_id == 90021120 for unit in full.players[0].board))

    def test_almyess_aura_effect_protection_clash_and_three_attack_super_evolution(self):
        protected = self.fresh(seed=73)
        source = _play(protected, self.repository, 10654110)
        self.assertTrue(source.has_keyword("灵气"))
        protected._start_effects(
            _card(999004, card_type="法术", attack=None, life=None),
            None,
            (EffectOperation(EffectKind.DESTROY, TargetKind.ALL_ENEMY_UNITS),),
            controller=1,
        )
        self.assertIn(source, protected.players[0].board)
        protected._start_effects(
            _card(999005, card_type="法术", attack=None, life=None),
            None,
            (EffectOperation(EffectKind.DESTROY, TargetKind.ENEMY_UNIT),),
            controller=1,
        )
        self.assertIsNone(protected.state.pending_choice)
        self.assertIn(source, protected.players[0].board)

        combat = self.fresh(seed=79)
        source = _play(combat, self.repository, 10654110)
        target = _put_unit(combat, 1, _card(993080, attack=1, life=20))
        source.summoned_this_turn = False
        source.can_attack = True
        combat.apply(Attack(0, source.entity_id, target.entity_id))
        self.assertNotIn(target, combat.players[1].board)
        self.assertEqual(source.health, source.max_health)

        evolved = self.fresh(seed=83)
        source = _play(evolved, self.repository, 10654110)
        _enable_super_evolve(evolved)
        evolved.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual((source.attacks_per_turn, source.attacks_remaining), (3, 3))

    def test_sweetness_earth_rite_modes_repeat_on_evolve_and_skip_when_insufficient(self):
        engine = self.fresh(seed=89)
        _put_sigil(engine, self.repository, 1)
        enemy = _put_unit(engine, 1, _card(993090, life=7))
        source = _play(engine, self.repository, 10733110)
        _choose_label(engine, "造成3点")
        self.assertEqual(enemy.health, 4)

        engine.players[0].deck = [_card(993091), _card(993092)]
        _put_sigil(engine, self.repository, 1)
        _enable_evolve(engine)
        engine.apply(Evolve(0, source.entity_id))
        _choose_label(engine, "抽取2张")
        self.assertEqual(len(engine.players[0].hand), 2)

        insufficient = self.fresh(seed=97)
        _play(insufficient, self.repository, 10733110)
        self.assertIsNone(insufficient.state.pending_choice)

    def test_beloved_masterpiece_fanfare_last_words_and_super_evolve(self):
        engine = self.fresh(seed=101)
        enemies = [_put_unit(engine, 1, _card(993100 + index, life=6)) for index in range(2)]
        source = _play(engine, self.repository, 10734120)
        self.assertTrue(source.has_keyword("守护"))
        self.assertTrue(all(enemy not in engine.players[1].board for enemy in enemies))
        _put_sigil(engine, self.repository, 2)
        enemy_health = engine.players[1].health
        engine._start_effects(
            source.definition,
            source.entity_id,
            (EffectOperation(EffectKind.DESTROY, TargetKind.SELF),),
            controller=0,
        )
        self.assertEqual(engine.players[1].health, enemy_health - 3)
        self.assertNotIn(source, engine.players[0].board)

        evolved = self.fresh(seed=103)
        source = _play(evolved, self.repository, 10734120)
        for index in range(3):
            _put_unit(evolved, 0, _card(993110 + index))
        _enable_super_evolve(evolved)
        evolved.apply(SuperEvolve(0, source.entity_id))
        copies = [unit for unit in evolved.players[0].board if unit.definition.card_id == 10734120]
        self.assertEqual(len(copies), 2)
        self.assertEqual(len(evolved.players[0].board), evolved.config.max_board)

    def test_aika_copies_seeded_destroyed_history_on_fanfare_and_evolve(self):
        outcomes = []
        for _ in range(2):
            engine = self.fresh(seed=107)
            engine.state.destroyed_followers = [
                _destroyed(_card(993120), 1),
                _destroyed(_card(993121), 2),
            ]
            source = _play(engine, self.repository, 10803110)
            first = engine.players[0].hand[0].card_id
            _enable_evolve(engine)
            engine.apply(Evolve(0, source.entity_id))
            self.assertEqual(len(engine.players[0].hand), 2)
            outcomes.append((first, tuple(card.card_id for card in engine.players[0].hand), engine.deterministic_fingerprint()))
        self.assertEqual(outcomes[0], outcomes[1])

        empty = self.fresh(seed=109)
        _play(empty, self.repository, 10803110)
        self.assertFalse(empty.players[0].hand)

    def test_aika_hidden_random_copy_does_not_leak_identity_to_opponent_observation(self):
        deck = [_card(994000 + index, class_id=0, class_name="中立") for index in range(40)]
        vocabulary = tuple(card.card_id for card in deck) + (10803110, 994100, 994101)

        def resolved(seed: int):
            env = ShadowverseEnv(
                deck,
                deck,
                class_a=1,
                class_b=1,
                seed=113,
                rulebook=self.rulebook,
                card_resolver=self.repository.get,
                observation_version="v2",
                card_vocabulary=vocabulary,
                validate_invariants=True,
            )
            env.reset(seed=113)
            for player in env.players:
                player.hand.clear()
                player.hand_entity_ids.clear()
                player.board.clear()
            env.players[0].mana = env.players[0].max_mana = 10
            env.core.state.destroyed_followers = [
                _destroyed(_card(994100), 1),
                _destroyed(_card(994101), 2),
            ]
            _put_hand(env.core, self.repository.get(10803110))
            env.core.random.seed(seed)
            env.core.apply(PlayCard(0, 0))
            copied = env.players[0].hand[0].card_id
            env.core.state.active_player = 1
            return copied, env.observation()

        first_id, first_obs = resolved(0)
        second_id, second_obs = resolved(1)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(first_obs, second_obs)

    def test_reef_and_lolo_self_summon_has_no_fanfare_loop_and_both_add_orcas(self):
        engine = self.fresh(seed=127)
        source = _play(engine, self.repository, 10842110)
        copies = [unit for unit in engine.players[0].board if unit.definition.card_id == 10842110]
        self.assertEqual(len(copies), 2)
        self.assertTrue(all(unit.has_keyword("突进") for unit in copies))
        engine.apply(EndTurn(0))
        self.assertEqual([card.card_id for card in engine.players[0].hand].count(90041130), 2)
        self.assertIn(source, engine.players[0].board)

        capped = self.fresh(seed=131)
        for index in range(8):
            _put_hand(capped, _card(994200 + index))
        _play(capped, self.repository, 10842110)
        capped.apply(EndTurn(0))
        self.assertEqual(len(capped.players[0].hand), capped.config.max_hand)
        self.assertEqual([card.card_id for card in capped.players[0].hand].count(90041130), 1)

    def test_luminous_dragon_fanfare_discards_available_two_then_damages_all(self):
        engine = self.fresh(seed=137)
        first = _put_hand(engine, _card(994300))
        second = _put_hand(engine, _card(994301))
        enemies = [_put_unit(engine, 1, _card(994310 + index, life=4)) for index in range(2)]
        _play(engine, self.repository, 10844120)
        _choose_entity(engine, first.entity_id)
        _choose_entity(engine, second.entity_id)
        self.assertTrue(all(enemy not in engine.players[1].board for enemy in enemies))
        self.assertEqual(engine.players[1].health, 16)
        self.assertEqual({entry.definition.card_id for entry in engine.players[0].graveyard}, {994300, 994301})

        empty = self.fresh(seed=139)
        enemy = _put_unit(empty, 1, _card(994320, life=5))
        _play(empty, self.repository, 10844120)
        self.assertIsNone(empty.state.pending_choice)
        self.assertEqual((enemy.health, empty.players[1].health), (1, 16))

    def test_luminous_dragon_super_draws_three_and_accelerate_only_gains_max_mana(self):
        evolved = self.fresh(seed=149)
        evolved.players[0].deck = [_card(994400 + index) for index in range(3)]
        source = _play(evolved, self.repository, 10844120)
        _enable_super_evolve(evolved)
        evolved.apply(SuperEvolve(0, source.entity_id))
        self.assertEqual(len(evolved.players[0].hand), 3)

        accelerated = self.fresh(seed=151)
        accelerated.players[0].max_mana = accelerated.players[0].mana = 5
        enemy = _put_unit(accelerated, 1, _card(994410, life=8))
        _put_hand(accelerated, self.repository.get(10844120))
        accelerated.apply(PlayCard(0, 0, mode_id="accelerate_3"))
        self.assertEqual((accelerated.players[0].max_mana, accelerated.players[0].mana), (6, 2))
        self.assertFalse(any(unit.definition.card_id == 10844120 for unit in accelerated.players[0].board))
        self.assertEqual((enemy.health, accelerated.players[1].health), (8, 20))
        self.assertEqual(accelerated.players[0].graveyard[-1].definition.card_id, 10844120)

    def test_all_real_accelerate_cards_require_body_to_be_unaffordable(self):
        expected = {
            10671110: ("accelerate_2", 2),
            10672110: ("accelerate_3", 3),
            10673110: ("accelerate_4", 4),
            10844120: ("accelerate_3", 3),
        }
        for card_id, (mode_id, accelerate_cost) in expected.items():
            with self.subTest(card_id=card_id):
                card = self.repository.get(card_id)
                mode = next(
                    mode
                    for mode in self.rulebook.modes_for(card_id)
                    if mode.mode_id == mode_id
                )
                self.assertEqual(mode.cost, accelerate_cost)

                affordable = self.fresh(seed=card_id)
                affordable.players[0].mana = affordable.players[0].max_mana = card.cost
                _put_hand(affordable, card)
                self.assertNotIn(
                    PlayCard(0, 0, mode_id=mode_id),
                    affordable.legal_commands(),
                )

                unaffordable = self.fresh(seed=card_id + 1)
                unaffordable.players[0].mana = (
                    unaffordable.players[0].max_mana
                ) = accelerate_cost
                _put_hand(unaffordable, card)
                self.assertIn(
                    PlayCard(0, 0, mode_id=mode_id),
                    unaffordable.legal_commands(),
                )

    def test_real_mode_and_activate_action_masks_match_command_layer(self):
        deck = [_card(995000 + index, class_id=0, class_name="中立") for index in range(40)]
        env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=157,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        env.reset(seed=157)
        for player in env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        env.players[0].mana = env.players[0].max_mana = 10
        _put_hand(env.core, self.repository.get(10351110))
        env.core.apply(PlayCard(0, 0))
        mask = env.action_mask()
        legal = env.core.legal_commands()
        actions = [index for index, allowed in enumerate(mask) if allowed]
        self.assertEqual(len(actions), 2)
        self.assertEqual({env._decode_action(action) for action in actions}, set(legal))
        fingerprint = env.core.deterministic_fingerprint()
        with self.assertRaises(ValueError):
            env.step(env.END_TURN)
        self.assertEqual(env.core.deterministic_fingerprint(), fingerprint)

        activate_env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=163,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        activate_env.reset(seed=163)
        for player in activate_env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        activate_env.players[0].mana = activate_env.players[0].max_mana = 10
        target = _put_unit(activate_env.core, 0, _card(995100))
        amulet = _play(activate_env.core, self.repository, 10461210)
        command = ActivateAmulet(0, amulet.entity_id)
        action = activate_env._encode_command(command)
        self.assertTrue(activate_env.action_mask()[action])
        activate_env.step(action)
        choice_actions = [index for index, allowed in enumerate(activate_env.action_mask()) if allowed]
        self.assertEqual(len(choice_actions), 1)
        self.assertEqual(activate_env._decode_action(choice_actions[0]).option_id, f"entity:{target.entity_id}")

        accelerate_env = ShadowverseEnv(
            deck,
            deck,
            class_a=1,
            class_b=1,
            seed=167,
            rulebook=self.rulebook,
            card_resolver=self.repository.get,
            validate_invariants=True,
        )
        accelerate_env.reset(seed=167)
        for player in accelerate_env.players:
            player.hand.clear()
            player.hand_entity_ids.clear()
            player.board.clear()
        accelerate_env.players[0].mana = accelerate_env.players[0].max_mana = 3
        _put_hand(accelerate_env.core, self.repository.get(10844120))
        accelerate = PlayCard(0, 0, mode_id="accelerate_3")
        normal = PlayCard(0, 0)
        self.assertIn(accelerate, accelerate_env.core.legal_commands())
        self.assertNotIn(normal, accelerate_env.core.legal_commands())
        accelerate_action = accelerate_env._encode_command(accelerate)
        self.assertIsNotNone(accelerate_action)
        self.assertTrue(accelerate_env.action_mask()[accelerate_action])
        accelerate_env.players[0].mana = accelerate_env.players[0].max_mana = 8
        self.assertNotIn(accelerate, accelerate_env.core.legal_commands())
        self.assertIn(normal, accelerate_env.core.legal_commands())
        self.assertFalse(accelerate_env.action_mask()[accelerate_action])
        self.assertTrue(accelerate_env.action_mask()[accelerate_env._encode_command(normal)])


class ExistingPrimitivesSecondCompletionAuditTests(unittest.TestCase):
    def test_database_multilingual_text_references_and_accelerate_are_reviewed(self):
        expected_phrases = {
            10113130: ("Activates in hand", "leaves the field", "deal it 4 damage"),
            10133130: ("Earth Rite", "Evolve this follower", "recover 2 play points"),
            10272120: ("4 attack or less", "banish it", "exact copy"),
            10351110: ("Select a Mode", "Give this follower +2/+2", "Replicate"),
            10461210: ("Engage", "transform it", "Draw a card"),
            10523110: ("Can't attack", "Steelclad Knight", "all other allied followers"),
            10654110: ("Aura", "Can't be destroyed", "3 times per turn"),
            10733110: ("Earth Rite", "Select a Mode", "Replicate"),
            10734120: ("Deal 6 damage", "Earth Rite", "Beloved Masterpiece"),
            10803110: ("destroyed this match", "without revealing", "Replicate"),
            10842110: ("Serene Sirens", "Rush", "Majestic Megalorca"),
            10844120: ("Select 2 cards", "all enemies", "Draw 3 cards"),
        }
        expected_references = {
            10113130: (),
            10133130: (),
            10272120: (),
            10351110: (),
            10461210: (10461210,),
            10523110: (90021120,),
            10654110: (),
            10733110: (),
            10734120: (10734120,),
            10803110: (),
            10842110: (10842110, 90041130),
            10844120: (),
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

            modes = connection.execute(
                "SELECT mode_type, cost, text_chs, text_cht, text_eng, text_jpn, text_kor FROM alt_modes WHERE card_id=10844120"
            ).fetchall()
            self.assertEqual(len(modes), 1)
            self.assertEqual((modes[0][0], modes[0][1]), ("激奏", 3))
            self.assertTrue(all(modes[0][index] for index in range(2, 7)))
            self.assertIn("Gain 1 max play point", modes[0][4])

    def test_all_twelve_cards_have_exact_clause_evidence_and_token_audit_stays_complete(self):
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
                    ["tests/test_real_existing_primitives_second_completion_batch.py"],
                )

        token_report = _build_token_audit("data/cards.sqlite3", "data/rules")
        token_total = token_report["summary"]["total"]
        token_categories = token_report["summary"]["categories"]
        self.assertEqual(token_categories["entry_behavior_complete"], token_total)
        self.assertEqual(sum(token_categories.values()), token_total)
        self.assertTrue(all(card["category"] == "entry_behavior_complete" for card in token_report["cards"]))


if __name__ == "__main__":
    unittest.main()
