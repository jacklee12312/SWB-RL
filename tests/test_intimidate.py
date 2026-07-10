# -*- coding: utf-8 -*-
"""Official Intimidate attack-legality semantics."""

from __future__ import annotations

import os
import unittest

from scripts.report_rule_coverage import _build_coverage_report
from swb.db.repository import CardDefinition, CardRepository
from swb.engine.abilities import AbilityKeyword
from swb.engine.card_rules import CardRule, RuleBook, Trigger, _parse_operation
from swb.engine.commands import Attack, Choose, PlayCard
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.environment import ShadowverseEnv
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


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
        "life": 3,
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
    return engine


def _place(
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


def _insert_hand(
    engine: GameEngine,
    definition: CardDefinition,
    *,
    player_index: int = 0,
) -> HandCard:
    card = HandCard(
        definition=definition,
        entity_id=engine.state.allocate_entity_id(),
        origin=CardOrigin.DECK,
    )
    player = engine.players[player_index]
    player.hand.append(card)
    player.hand_entity_ids.append(card.entity_id)
    return card


def _attack_commands(engine: GameEngine, attacker: Unit) -> list[Attack]:
    return [
        command
        for command in engine.legal_commands()
        if isinstance(command, Attack)
        and command.attacker_id == attacker.entity_id
    ]


class IntimidateAttackLegalityTests(unittest.TestCase):
    def test_intimidate_blocks_only_that_follower_as_attack_target(self):
        engine = _engine()
        attacker = _place(engine, 0, _card(300))
        attacker.can_attack = True
        intimidate = _place(
            engine,
            1,
            _card(400, keywords=frozenset({"威慑"})),
        )
        other = _place(engine, 1, _card(401))

        commands = _attack_commands(engine, attacker)

        self.assertIn(Attack(0, attacker.entity_id, None), commands)
        self.assertNotIn(
            Attack(0, attacker.entity_id, intimidate.entity_id),
            commands,
        )
        self.assertIn(Attack(0, attacker.entity_id, other.entity_id), commands)

    def test_guard_is_inactive_while_same_follower_has_intimidate(self):
        engine = _engine()
        attacker = _place(engine, 0, _card(300))
        attacker.can_attack = True
        hidden_guard = _place(
            engine,
            1,
            _card(400, keywords=frozenset({"守护", "威慑"})),
        )
        other = _place(engine, 1, _card(401))

        commands = _attack_commands(engine, attacker)

        self.assertIn(Attack(0, attacker.entity_id, None), commands)
        self.assertIn(Attack(0, attacker.entity_id, other.entity_id), commands)
        self.assertNotIn(
            Attack(0, attacker.entity_id, hidden_guard.entity_id),
            commands,
        )

    def test_another_visible_guard_still_must_be_attacked(self):
        engine = _engine()
        attacker = _place(engine, 0, _card(300))
        attacker.can_attack = True
        hidden_guard = _place(
            engine,
            1,
            _card(400, keywords=frozenset({"守护", "威慑"})),
        )
        visible_guard = _place(
            engine,
            1,
            _card(401, keywords=frozenset({"守护"})),
        )
        other = _place(engine, 1, _card(402))

        commands = _attack_commands(engine, attacker)

        self.assertEqual(
            commands,
            [Attack(0, attacker.entity_id, visible_guard.entity_id)],
        )
        self.assertNotIn(
            Attack(0, attacker.entity_id, hidden_guard.entity_id),
            commands,
        )
        self.assertNotIn(Attack(0, attacker.entity_id, other.entity_id), commands)

    def test_rush_storm_and_evolution_do_not_bypass_intimidate(self):
        for keyword in ("突进", "疾驰"):
            with self.subTest(keyword=keyword):
                engine = _engine()
                attacker = _place(
                    engine,
                    0,
                    _card(300, keywords=frozenset({keyword})),
                )
                target = _place(
                    engine,
                    1,
                    _card(400, keywords=frozenset({"威慑"})),
                )
                self.assertNotIn(
                    Attack(0, attacker.entity_id, target.entity_id),
                    _attack_commands(engine, attacker),
                )
                engine._apply_evolution_state(
                    attacker,
                    0,
                    super_evolve=True,
                    cause="test",
                    trigger_abilities=False,
                )
                self.assertNotIn(
                    Attack(0, attacker.entity_id, target.entity_id),
                    _attack_commands(engine, attacker),
                )

    def test_intimidate_follower_can_attack_normally(self):
        engine = _engine()
        attacker = _place(
            engine,
            0,
            _card(300, keywords=frozenset({"威慑"})),
        )
        attacker.can_attack = True
        target = _place(engine, 1, _card(400))

        commands = _attack_commands(engine, attacker)

        self.assertIn(Attack(0, attacker.entity_id, None), commands)
        self.assertIn(Attack(0, attacker.entity_id, target.entity_id), commands)

    def test_illegal_direct_attack_preserves_full_fingerprint(self):
        engine = _engine()
        attacker = _place(engine, 0, _card(300))
        attacker.can_attack = True
        target = _place(
            engine,
            1,
            _card(400, keywords=frozenset({"威慑"})),
        )
        before = engine.deterministic_fingerprint()

        with self.assertRaisesRegex(IllegalCommand, "intimidate"):
            engine.apply(Attack(0, attacker.entity_id, target.entity_id))

        self.assertEqual(engine.deterministic_fingerprint(), before)


class IntimidateKeywordLifecycleTests(unittest.TestCase):
    def test_add_and_remove_keyword_update_attack_legality(self):
        operation = _parse_operation(
            {
                "kind": "add_keyword",
                "target": "enemy_unit",
                "keyword": "威慑",
            },
            "test/operations[0]",
            600,
        )
        self.assertEqual(operation.keyword, "威慑")

        engine = _engine()
        attacker = _place(engine, 0, _card(300))
        attacker.can_attack = True
        target = _place(engine, 1, _card(400))
        target.add_keyword("威慑")
        self.assertNotIn(
            Attack(0, attacker.entity_id, target.entity_id),
            _attack_commands(engine, attacker),
        )

        target.remove_keyword("威慑")
        self.assertIn(
            Attack(0, attacker.entity_id, target.entity_id),
            _attack_commands(engine, attacker),
        )

    def test_temporary_removal_restores_printed_intimidate(self):
        engine = _engine()
        target = _place(
            engine,
            1,
            _card(400, keywords=frozenset({"威慑"})),
        )
        target.remove_keyword(
            "威慑",
            duration="until_end_of_turn",
            expires_for_player=0,
        )
        self.assertFalse(target.has_intimidate)

        target.expire_keywords("until_end_of_turn", 0)

        self.assertTrue(target.has_intimidate)

    def test_transform_replaces_intimidate_with_new_printed_keywords(self):
        intimidate = _card(400, keywords=frozenset({"威慑"}))
        ordinary = _card(401)
        transform = _card(600, card_type="法术", attack=None, life=None)
        rulebook = RuleBook((CardRule(
            600,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.TRANSFORM,
                    TargetKind.ENEMY_UNIT,
                    card_id=401,
                    requires_target=True,
                ),
            ),
        ),))
        engine = _engine(rulebook, catalog={401: ordinary})
        target = _place(engine, 1, intimidate)
        _insert_hand(engine, transform)
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertEqual(target.definition.card_id, 401)
        self.assertFalse(target.has_intimidate)

    def test_selected_ability_damage_can_target_intimidate(self):
        spell = _card(600, card_type="法术", attack=None, life=None)
        rulebook = RuleBook((CardRule(
            600,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.DAMAGE_UNIT,
                    TargetKind.ENEMY_UNIT,
                    2,
                    requires_target=True,
                ),
            ),
        ),))
        engine = _engine(rulebook)
        target = _place(
            engine,
            1,
            _card(400, life=5, keywords=frozenset({"威慑"})),
        )
        _insert_hand(engine, spell)
        engine.players[0].max_mana = 10
        engine.players[0].mana = 10

        engine.apply(PlayCard(0, len(engine.players[0].hand) - 1))
        request = engine.state.pending_choice
        self.assertIn(target.entity_id, [option.entity_id for option in request.options])
        engine.apply(Choose(0, f"entity:{target.entity_id}"))

        self.assertEqual(target.health, 3)


class IntimidateEnvironmentTests(unittest.TestCase):
    def test_action_mask_and_observation_expose_intimidate(self):
        env = ShadowverseEnv(
            [_card(1000 + index) for index in range(40)],
            [_card(2000 + index) for index in range(40)],
            class_a=1,
            class_b=1,
            seed=42,
        )
        env.reset(seed=42)
        attacker = Unit.summon(
            _card(300),
            entity_id=env.core.state.allocate_entity_id(),
        )
        attacker.can_attack = True
        target = Unit.summon(
            _card(400, keywords=frozenset({"威慑"})),
            entity_id=env.core.state.allocate_entity_id(),
        )
        env.players[0].board = [attacker]
        env.players[1].board = [target]

        mask = env.action_mask()
        base = ShadowverseEnv.ATTACK_OFFSET

        self.assertTrue(mask[base])
        self.assertFalse(mask[base + 1])
        self.assertEqual(env._board_features(target)[12], 1.0)
        self.assertEqual(len(env.observation()), 280)


@unittest.skipUnless(os.path.exists("data/cards.sqlite3"), "card database unavailable")
class RealBazarragaIntimidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = CardRepository("data/cards.sqlite3")
        cls.rulebook = RuleBook.from_directory("data/rules")

    def test_real_bazarraga_is_unattackable_and_last_words_is_exact(self):
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
        attacker = _place(engine, 0, _card(300))
        attacker.can_attack = True
        bazarraga = _place(engine, 1, self.repo.get(10451120))

        self.assertTrue(bazarraga.has_intimidate)
        self.assertNotIn(
            Attack(0, attacker.entity_id, bazarraga.entity_id),
            _attack_commands(engine, attacker),
        )

        bazarraga.health = 0
        engine._stabilize()
        engine._resolve_event_queue()
        engine._stabilize()

        replacements = [
            unit
            for unit in engine.players[1].board
            if isinstance(unit, Unit)
            and unit.definition.card_id == bazarraga.definition.card_id
        ]
        self.assertEqual(len(replacements), 1)
        self.assertTrue(replacements[0].has_intimidate)
        self.assertEqual(engine.players[1].health, 18)
        self.assertFalse(any(
            event.ability is AbilityKeyword.INTIMIDATE
            for event in engine.placeholder_ability_events
        ))

        report = _build_coverage_report("data/cards.sqlite3", "data/rules")
        info = report["classifications"]["10451120"]
        self.assertEqual(info["coverage"], "covered_exact")
        self.assertIn("summon", info["reason"])

    def test_conditional_keyword_mentions_are_not_initial_keywords(self):
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

        wind = _place(engine, 0, self.repo.get(10142120))
        justice = _place(engine, 0, self.repo.get(10544110))
        beast = _place(engine, 0, self.repo.get(10603110))

        self.assertFalse(wind.has_intimidate)
        self.assertTrue(wind.has_keyword("疾驰"))
        self.assertFalse(justice.has_intimidate)
        self.assertTrue(justice.has_guard)
        for keyword in ("疾驰", "必杀", "威慑", "吸血", "屏障"):
            self.assertFalse(beast.has_keyword(keyword))


if __name__ == "__main__":
    unittest.main()
