from __future__ import annotations

import os
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import ActivateAmulet, Attack, Choose, EndTurn, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import Amulet, HandCard, Unit


def _card(card_id: int, **overrides) -> CardDefinition:
    values = {
        "card_id": card_id,
        "card_set_id": 10000,
        "class_id": 1,
        "class_name": "精灵",
        "name": f"card-{card_id}",
        "cost": 1,
        "card_type": "随从",
        "attack": 2,
        "life": 5,
        "keywords": frozenset(),
        "support_level": "basic",
        "is_collectible": True,
    }
    values.update(overrides)
    return CardDefinition(**values)


def _engine(
    rulebook: RuleBook | None = None,
    *,
    catalog: dict[int, CardDefinition] | None = None,
) -> GameEngine:
    cards = catalog or {}
    engine = GameEngine(
        [_card(1000 + index) for index in range(40)],
        [_card(2000 + index) for index in range(40)],
        class_a=1,
        class_b=1,
        seed=42,
        rulebook=rulebook or RuleBook(),
        card_resolver=lambda card_id: cards.get(card_id),
        config=GameConfig(validate_invariants=True),
    )
    engine.reset(seed=42)
    engine.players[0].max_mana = 10
    engine.players[0].mana = 10
    return engine


def _place_unit(
    engine: GameEngine,
    player_index: int,
    definition: CardDefinition,
) -> Unit:
    unit = engine._summon_follower_to_board(
        player_index,
        definition,
        summon_cause="test_setup",
    )
    assert unit is not None
    return unit


def _place_amulet(
    engine: GameEngine,
    player_index: int,
    definition: CardDefinition,
    *,
    earth_sigil_count: int = 0,
) -> Amulet:
    amulet = Amulet(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
        entered_turn=engine.turn,
        earth_sigil_count=earth_sigil_count,
    )
    engine.players[player_index].board.append(amulet)
    return amulet


def _insert_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int = 0,
) -> HandCard:
    hand_card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    player = engine.players[player_index]
    player.hand.append(hand_card)
    player.hand_entity_ids.append(hand_card.entity_id)
    return hand_card


def _target_spell(
    target: TargetKind,
    *,
    effect: EffectKind = EffectKind.DAMAGE_UNIT,
    requires_target: bool = True,
) -> tuple[CardDefinition, RuleBook]:
    spell = _card(
        600,
        card_type="法术",
        attack=None,
        life=None,
    )
    operation = EffectOperation(
        effect,
        target,
        amount=2,
        requires_target=requires_target,
    )
    return spell, RuleBook((CardRule(600, Trigger.PLAY, (operation,)),))


class AuraTargetLegalityTests(unittest.TestCase):
    def test_enemy_manual_effect_cannot_select_aura_and_illegal_play_is_atomic(self):
        spell, rulebook = _target_spell(TargetKind.ENEMY_UNIT)
        engine = _engine(rulebook)
        target = _place_unit(
            engine,
            1,
            _card(400, keywords=frozenset({"灵气"})),
        )
        _insert_hand(engine, spell)
        command = PlayCard(0, len(engine.players[0].hand) - 1)
        before = engine.deterministic_fingerprint()

        self.assertTrue(target.has_aura)
        self.assertNotIn(command, engine.legal_commands())
        with self.assertRaises(IllegalCommand):
            engine.apply(command)
        self.assertEqual(engine.deterministic_fingerprint(), before)

    def test_manual_enemy_choice_excludes_only_aura_target(self):
        spell, rulebook = _target_spell(TargetKind.ENEMY_UNIT)
        engine = _engine(rulebook)
        aura = _place_unit(
            engine,
            1,
            _card(400, keywords=frozenset({"灵气"})),
        )
        ordinary = _place_unit(engine, 1, _card(401))
        _insert_hand(engine, spell)

        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        request = engine.state.pending_choice
        self.assertIsNotNone(request)
        self.assertEqual(
            [option.entity_id for option in request.options],
            [ordinary.entity_id],
        )
        engine.apply(Choose(0, f"entity:{ordinary.entity_id}"))

        self.assertEqual(ordinary.health, 3)
        self.assertEqual(aura.health, 5)

    def test_controller_can_select_own_aura_with_an_ability(self):
        spell, rulebook = _target_spell(TargetKind.OWN_UNIT)
        engine = _engine(rulebook)
        aura = _place_unit(
            engine,
            0,
            _card(400, keywords=frozenset({"灵气"})),
        )
        _insert_hand(engine, spell)

        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        request = engine.state.pending_choice
        self.assertIn(aura.entity_id, [option.entity_id for option in request.options])
        engine.apply(Choose(0, f"entity:{aura.entity_id}"))

        self.assertEqual(aura.health, 3)

    def test_random_and_all_enemy_effects_still_affect_aura(self):
        for target in (TargetKind.RANDOM_ENEMY_UNIT, TargetKind.ALL_ENEMY_UNITS):
            with self.subTest(target=target.value):
                spell, rulebook = _target_spell(
                    target,
                    requires_target=False,
                )
                engine = _engine(rulebook)
                aura = _place_unit(
                    engine,
                    1,
                    _card(400, keywords=frozenset({"灵气"})),
                )
                _insert_hand(engine, spell)

                engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))

                self.assertEqual(aura.health, 3)

    def test_aura_does_not_prevent_follower_attacks(self):
        engine = _engine()
        attacker = _place_unit(engine, 0, _card(300))
        attacker.can_attack = True
        aura = _place_unit(
            engine,
            1,
            _card(400, keywords=frozenset({"灵气"})),
        )
        command = Attack(0, attacker.entity_id, aura.entity_id)

        self.assertIn(command, engine.legal_commands())
        engine.apply(command)
        self.assertEqual(aura.health, 3)

    def test_intrinsic_aura_amulet_blocks_manual_but_not_random_effects(self):
        aura_amulet = _card(
            400,
            card_type="护符",
            attack=None,
            life=None,
            keywords=frozenset({"灵气"}),
        )
        manual_spell, manual_rules = _target_spell(
            TargetKind.ENEMY_BOARD,
            effect=EffectKind.BANISH,
        )
        manual_engine = _engine(manual_rules)
        _place_amulet(manual_engine, 1, aura_amulet)
        _insert_hand(manual_engine, manual_spell)
        manual_command = PlayCard(0, len(manual_engine.players[0].hand) - 1)
        self.assertNotIn(manual_command, manual_engine.legal_commands())

        random_spell, random_rules = _target_spell(
            TargetKind.RANDOM_ENEMY_BOARD,
            effect=EffectKind.BANISH,
            requires_target=False,
        )
        random_engine = _engine(random_rules)
        placed = _place_amulet(random_engine, 1, aura_amulet)
        _insert_hand(random_engine, random_spell)
        random_engine.apply(
            PlayCard(0, len(random_engine.players[0].hand) - 1)
        )
        self.assertNotIn(placed, random_engine.players[1].board)


class AuraKeywordLifecycleTests(unittest.TestCase):
    def test_schema_and_runtime_add_remove_aura(self):
        operation = _parse_operation(
            {
                "kind": "add_keyword",
                "target": "own_unit",
                "keyword": "灵气",
            },
            "test/operations[0]",
            600,
        )
        self.assertEqual(operation.keyword, "灵气")

        engine = _engine()
        target = _place_unit(engine, 1, _card(400))
        target.add_keyword("灵气")
        self.assertTrue(target.has_aura)
        self.assertTrue(target.cannot_be_enemy_targeted)
        target.remove_keyword("灵气")
        self.assertFalse(target.has_aura)
        self.assertFalse(target.cannot_be_enemy_targeted)

    def test_temporary_removal_restores_printed_aura(self):
        engine = _engine()
        target = _place_unit(
            engine,
            1,
            _card(400, keywords=frozenset({"灵气"})),
        )
        target.remove_keyword(
            "灵气",
            duration="until_end_of_turn",
            expires_for_player=0,
        )
        self.assertFalse(target.has_aura)

        target.expire_keywords("until_end_of_turn", 0)

        self.assertTrue(target.has_aura)

    def test_transform_replaces_aura_with_new_printed_keywords(self):
        aura = _card(400, keywords=frozenset({"灵气"}))
        ordinary = _card(401)
        transform = _card(600, card_type="法术", attack=None, life=None)
        rulebook = RuleBook((CardRule(
            600,
            Trigger.PLAY,
            (EffectOperation(
                EffectKind.TRANSFORM,
                TargetKind.ENEMY_UNIT,
                card_id=401,
                requires_target=True,
            ),),
        ),))
        engine = _engine(rulebook, catalog={401: ordinary})
        target = _place_unit(engine, 1, aura)
        _insert_hand(engine, transform)

        # Temporarily remove Aura so the opponent may choose the transform target.
        target.remove_keyword(
            "灵气",
            duration="until_end_of_turn",
            expires_for_player=0,
        )
        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertEqual(target.definition.card_id, 401)
        self.assertFalse(target.has_aura)

    def test_pending_manual_target_is_revalidated_after_gaining_aura(self):
        spell, rulebook = _target_spell(TargetKind.ENEMY_UNIT)
        engine = _engine(rulebook)
        target = _place_unit(engine, 1, _card(400))
        _insert_hand(engine, spell)
        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        target.add_keyword("灵气")

        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertIsNone(engine.state.pending_choice)
        self.assertEqual(target.health, 5)


class AuraEnvironmentTests(unittest.TestCase):
    def test_choice_mask_and_observation_expose_public_aura(self):
        spell, rulebook = _target_spell(TargetKind.ENEMY_UNIT)
        env = ShadowverseEnv(
            [_card(1000 + index) for index in range(40)],
            [_card(2000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=rulebook,
        )
        env.reset(seed=42)
        aura = _place_unit(
            env.core,
            1,
            _card(400, keywords=frozenset({"灵气"})),
        )
        ordinary = _place_unit(env.core, 1, _card(401))
        env.players[0].hand[0] = HandCard(
            spell,
            entity_id=env.players[0].hand_entity_ids[0],
        )
        env.players[0].mana = 10

        env.step(ShadowverseEnv.PLAY_OFFSET)
        request = env.core.state.pending_choice
        self.assertIsNotNone(request)
        self.assertEqual(
            [option.entity_id for option in request.options],
            [ordinary.entity_id],
        )
        choice_actions = [
            action
            for action in range(
                ShadowverseEnv.CHOICE_OFFSET,
                ShadowverseEnv.GRAVEYARD_CHOICE_OFFSET,
            )
            if env.action_mask()[action]
        ]
        self.assertEqual(len(choice_actions), 1)
        self.assertEqual(env._board_features(aura)[13], 1.0)
        self.assertEqual(
            len(env.observation()),
            ShadowverseEnv.OBSERVATION_V1_SIZE,
        )

        aura_amulet = Amulet(
            definition=_card(
                402,
                card_type="护符",
                attack=None,
                life=None,
                keywords=frozenset({"灵气"}),
            ),
            entity_id=env.core.state.allocate_entity_id(),
        )
        self.assertEqual(env._board_features(aura_amulet)[13], 1.0)


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealAuraCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def _real_engine(self) -> GameEngine:
        engine = GameEngine(
            [_card(1000 + index) for index in range(40)],
            [_card(2000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=42,
            rulebook=self.rulebook,
            card_resolver=self.repo.get,
            config=GameConfig(validate_invariants=True),
        )
        engine.reset(seed=42)
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10
        return engine

    def test_real_maniou_aura_and_activation_listener_are_exact(self):
        engine = self._real_engine()
        maniou_card = self.repo.get(10161140)
        _insert_hand(engine, maniou_card)
        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        maniou = next(
            unit
            for unit in engine.players[0].board
            if isinstance(unit, Unit) and unit.definition.card_id == 10161140
        )
        activation_card = self.repo.get(10031210)
        amulet = _place_amulet(
            engine,
            0,
            activation_card,
            earth_sigil_count=1,
        )

        self.assertTrue(maniou.has_aura)
        self.assertEqual(maniou.attack, 2)
        engine.apply(ActivateAmulet(0, amulet.entity_id))
        self.assertEqual(maniou.attack, 3)
        engine.apply(EndTurn(0))
        self.assertEqual(maniou.attack, 2)
        self.assertFalse(any(
            event.ability in {AbilityKeyword.AURA, AbilityKeyword.ACTIVATE}
            and event.card_id == 10161140
            for event in engine.placeholder_ability_events
        ))

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10161140"]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertIn("listener:board:amulet_activated", info["reason"])

    def test_non_intrinsic_mentions_do_not_grant_initial_aura(self):
        engine = self._real_engine()
        aether = _place_unit(engine, 0, self.repo.get(10264110))
        beast = _place_unit(engine, 0, self.repo.get(10603110))
        joel = _place_unit(engine, 0, self.repo.get(10441110))

        self.assertFalse(aether.has_aura)
        self.assertFalse(beast.has_aura)
        self.assertTrue(joel.has_aura)


if __name__ == "__main__":
    unittest.main()
