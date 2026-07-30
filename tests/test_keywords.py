from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard, Attack
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine, IllegalCommand
from swb.engine.state import Unit, DeathCause
from swb.engine.events import EventType


def card(cid, **kw):
    defaults = dict(card_id=cid, card_set_id=10000, class_id=1, class_name="elf",
                    name="c%d" % cid, cost=1, card_type="随从", attack=1, life=1,
                    keywords=frozenset(), support_level="basic", is_collectible=True)
    defaults.update(kw)
    return CardDefinition(**defaults)


def mkunit(eng, cid, **kw):
    return Unit.summon(card(cid, **kw), entity_id=eng.state.allocate_entity_id())


def mkengine(**kw):
    e = GameEngine([card(i) for i in range(100,140)], [card(i) for i in range(200,240)], class_a=1, class_b=1, seed=1, **kw)
    e.reset(seed=1)
    return e


class BaneTests(unittest.TestCase):
    def test_bane_destroys_target(self):
        eng = mkengine()
        b = mkunit(eng, 900, keywords=frozenset({"必杀"}), attack=3, life=5)
        v = mkunit(eng, 901, attack=1, life=3)
        b.can_attack = True
        eng.players[0].board = [b]; eng.players[1].board = [v]
        eng.apply(Attack(0, b.entity_id, v.entity_id))
        self.assertEqual(v.health, 0)
        # death cause recorded in death queue
        causes = [r.cause for batch in eng.state.death_queue for r in batch.records]
        self.assertIn(DeathCause.EFFECT_DESTROY, causes)

    def test_bane_vs_leader(self):
        eng = mkengine()
        b = mkunit(eng, 900, keywords=frozenset({"必杀"}), attack=3, life=5)
        b.can_attack = True
        eng.players[0].board = [b]
        eng.apply(Attack(0, b.entity_id, None))
        self.assertEqual(eng.players[1].health, 17)

    def test_zero_attack_bane_still_destroys(self):
        eng = mkengine()
        b = mkunit(eng, 900, keywords=frozenset({"必杀"}), attack=0, life=5)
        v = mkunit(eng, 901, attack=1, life=3)
        b.can_attack = True
        eng.players[0].board = [b]; eng.players[1].board = [v]
        transition = eng.apply(Attack(0, b.entity_id, v.entity_id))
        self.assertNotIn(v, eng.players[1].board)
        self.assertTrue(any(
            event.type is EventType.BANE_TRIGGERED
            and event.source_id == b.entity_id
            and event.target_id == v.entity_id
            for event in transition.events
        ))

    def test_bane_still_destroys_after_barrier_prevents_damage(self):
        eng = mkengine()
        b = mkunit(eng, 900, keywords=frozenset({"必杀"}), attack=3, life=5)
        v = mkunit(eng, 901, keywords=frozenset({"屏障"}), attack=1, life=3)
        b.can_attack = True
        eng.players[0].board = [b]; eng.players[1].board = [v]
        transition = eng.apply(Attack(0, b.entity_id, v.entity_id))
        self.assertNotIn(v, eng.players[1].board)
        self.assertEqual(v.barrier_charges, 0)
        self.assertTrue(any(
            event.type is EventType.BANE_TRIGGERED
            and event.source_id == b.entity_id
            and event.target_id == v.entity_id
            for event in transition.events
        ))

    def test_bane_respects_effect_destroy_immunity(self):
        eng = mkengine()
        b = mkunit(eng, 900, keywords=frozenset({"必杀"}), attack=1, life=5)
        v = mkunit(eng, 901, attack=1, life=5)
        v.effect_destroy_immunity = True
        b.can_attack = True
        eng.players[0].board = [b]; eng.players[1].board = [v]

        transition = eng.apply(Attack(0, b.entity_id, v.entity_id))

        self.assertIn(v, eng.players[1].board)
        self.assertEqual(v.health, 4)
        self.assertTrue(any(
            event.type is EventType.BANE_TRIGGERED
            and event.target_id == v.entity_id
            for event in transition.events
        ))
        self.assertTrue(any(
            event.type is EventType.EFFECT_DESTROY_PREVENTED
            and event.target_id == v.entity_id
            for event in transition.events
        ))

    def test_bane_counter_respects_super_evolution_protection(self):
        eng = mkengine()
        attacker = mkunit(eng, 900, attack=3, life=5)
        defender = mkunit(
            eng,
            901,
            keywords=frozenset({"必杀"}),
            attack=1,
            life=5,
        )
        attacker.can_attack = True
        attacker.super_evolved = True
        attacker.super_evolved_turn = eng.turn
        eng.players[0].board = [attacker]
        eng.players[1].board = [defender]

        transition = eng.apply(
            Attack(0, attacker.entity_id, defender.entity_id)
        )

        self.assertIn(attacker, eng.players[0].board)
        self.assertEqual(attacker.health, 5)
        self.assertTrue(any(
            event.type is EventType.BANE_TRIGGERED
            and event.source_id == defender.entity_id
            and event.target_id == attacker.entity_id
            for event in transition.events
        ))

    def test_mutual_death_bane_applies(self):
        eng = mkengine()
        b = mkunit(eng, 900, keywords=frozenset({"必杀"}), attack=5, life=1)
        v = mkunit(eng, 901, attack=5, life=5)
        b.can_attack = True
        eng.players[0].board = [b]; eng.players[1].board = [v]
        eng.apply(Attack(0, b.entity_id, v.entity_id))
        self.assertEqual(v.health, 0)
        self.assertEqual(b.health, 0)


class DrainTests(unittest.TestCase):
    def test_drain_vs_leader(self):
        eng = mkengine()
        d = mkunit(eng, 900, keywords=frozenset({"吸血"}), attack=3, life=5)
        d.can_attack = True
        eng.players[0].health = 15
        eng.players[0].board = [d]
        eng.apply(Attack(0, d.entity_id, None))
        self.assertEqual(eng.players[0].health, 18)

    def test_drain_vs_follower(self):
        eng = mkengine()
        d = mkunit(eng, 900, keywords=frozenset({"吸血"}), attack=3, life=5)
        v = mkunit(eng, 901, attack=1, life=2)
        d.can_attack = True
        eng.players[0].health = 15
        eng.players[0].board = [d]; eng.players[1].board = [v]
        eng.apply(Attack(0, d.entity_id, v.entity_id))
        self.assertEqual(eng.players[0].health, 17)

    def test_drain_overkill_capped(self):
        eng = mkengine()
        d = mkunit(eng, 900, keywords=frozenset({"吸血"}), attack=10, life=5)
        v = mkunit(eng, 901, attack=1, life=1)
        d.can_attack = True
        eng.players[0].health = 15
        eng.players[0].board = [d]; eng.players[1].board = [v]
        eng.apply(Attack(0, d.entity_id, v.entity_id))
        self.assertEqual(eng.players[0].health, 16)

    def test_drain_barrier_blocked(self):
        eng = mkengine()
        d = mkunit(eng, 900, keywords=frozenset({"吸血"}), attack=3, life=5)
        v = mkunit(eng, 901, keywords=frozenset({"屏障"}), attack=1, life=3)
        d.can_attack = True
        eng.players[0].health = 15
        eng.players[0].board = [d]; eng.players[1].board = [v]
        eng.apply(Attack(0, d.entity_id, v.entity_id))
        self.assertEqual(eng.players[0].health, 15)

    def test_drain_not_exceed_max(self):
        eng = mkengine()
        d = mkunit(eng, 900, keywords=frozenset({"吸血"}), attack=3, life=5)
        d.can_attack = True
        eng.players[0].health = 19
        eng.players[0].board = [d]
        eng.apply(Attack(0, d.entity_id, None))
        self.assertEqual(eng.players[0].health, 20)


class BarrierTests(unittest.TestCase):
    def test_barrier_blocks_one(self):
        eng = mkengine()
        a = mkunit(eng, 900, attack=3, life=5); a.can_attack = True
        bu = mkunit(eng, 901, keywords=frozenset({"屏障"}), attack=1, life=3)
        eng.players[0].board = [a]; eng.players[1].board = [bu]
        eng.apply(Attack(0, a.entity_id, bu.entity_id))
        self.assertEqual(bu.health, 3)
        self.assertEqual(bu.barrier_charges, 0)

    def test_barrier_second_hits(self):
        eng = mkengine()
        a = mkunit(eng, 900, attack=3, life=5); a.can_attack = True
        bu = mkunit(eng, 901, attack=1, life=5)
        bu.barrier_charges = 0
        eng.players[0].board = [a]; eng.players[1].board = [bu]
        eng.apply(Attack(0, a.entity_id, bu.entity_id))
        self.assertEqual(bu.health, 2)

    def test_zero_no_barrier_consume(self):
        eng = mkengine()
        a = mkunit(eng, 900, attack=0, life=5); a.can_attack = True
        bu = mkunit(eng, 901, keywords=frozenset({"屏障"}), attack=1, life=3)
        eng.players[0].board = [a]; eng.players[1].board = [bu]
        eng.apply(Attack(0, a.entity_id, bu.entity_id))
        self.assertEqual(bu.barrier_charges, 1)

    def test_destroy_bypasses_barrier(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.DESTROY, target=TargetKind.ENEMY_UNIT),)),))
        eng = mkengine(rulebook=rulebook)
        bu = mkunit(eng, 900, keywords=frozenset({"屏障"}), attack=1, life=5)
        eng.players[1].board = [bu]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        eng.apply([c for c in eng.legal_commands() if isinstance(c, Choose)][0])
        self.assertTrue(any(g.definition.card_id == 900 for g in eng.players[1].graveyard))

    def test_all_target_per_unit_barrier(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=3),)),))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, keywords=frozenset({"屏障"}), attack=1, life=3)
        b = mkunit(eng, 901, attack=1, life=5)
        eng.players[1].board = [a, b]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(a.health, 3); self.assertEqual(a.barrier_charges, 0)
        self.assertEqual(b.health, 2)


class AmbushTests(unittest.TestCase):
    def test_ambush_not_attack_target(self):
        eng = mkengine()
        at = mkunit(eng, 900, attack=3, life=5); at.can_attack = True
        am = mkunit(eng, 901, keywords=frozenset({"潜行"}), attack=1, life=3)
        eng.players[0].board = [at]; eng.players[1].board = [am]
        cmd_attacks = [c for c in eng.legal_commands() if isinstance(c, Attack) and c.target_id is not None]
        self.assertFalse(any(a.target_id == am.entity_id for a in cmd_attacks))

    def test_ambush_not_enemy_choice_target(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_UNIT, amount=2),)),))
        eng = mkengine(rulebook=rulebook)
        am = mkunit(eng, 900, keywords=frozenset({"潜行"}), attack=1, life=3)
        eng.players[1].board = [am]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        playable = [c for c in eng.legal_commands() if isinstance(c, PlayCard) and c.hand_index == 0]
        self.assertEqual(len(playable), 0)

    def test_own_effect_targets_own_ambush(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.BUFF_UNIT, target=TargetKind.OWN_UNIT, amount=2, secondary_amount=2),)),))
        eng = mkengine(rulebook=rulebook)
        am = mkunit(eng, 900, keywords=frozenset({"潜行"}), attack=1, life=3)
        eng.players[0].board = [am]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        eng.apply([c for c in eng.legal_commands() if isinstance(c, Choose)][0])
        self.assertEqual(am.attack, 3)

    def test_ambush_lost_on_attack(self):
        eng = mkengine()
        am = mkunit(eng, 900, keywords=frozenset({"潜行"}), attack=3, life=5)
        v = mkunit(eng, 901, attack=1, life=3)
        am.can_attack = True
        eng.players[0].board = [am]; eng.players[1].board = [v]
        eng.apply(Attack(0, am.entity_id, v.entity_id))
        self.assertFalse(am.ambush_active)

    def test_ambush_guard_no_block(self):
        eng = mkengine()
        at = mkunit(eng, 900, attack=3, life=5); at.can_attack = True
        ag = mkunit(eng, 901, keywords=frozenset({"潜行", "守护"}), attack=1, life=3)
        eng.players[0].board = [at]; eng.players[1].board = [ag]
        leader_attacks = [c for c in eng.legal_commands() if isinstance(c, Attack) and c.target_id is None]
        self.assertTrue(any(leader_attacks))

    def test_random_hits_ambush(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.RANDOM_ENEMY_UNIT, amount=3),)),))
        eng = mkengine(rulebook=rulebook)
        am = mkunit(eng, 900, keywords=frozenset({"潜行"}), attack=1, life=5)
        eng.players[1].board = [am]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(am.health, 2)

    def test_all_hits_ambush(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=3),)),))
        eng = mkengine(rulebook=rulebook)
        am = mkunit(eng, 900, keywords=frozenset({"潜行"}), attack=1, life=5)
        eng.players[1].board = [am]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))
        self.assertEqual(am.health, 2)


class RegressionTests(unittest.TestCase):
    """Regression: attack validation and barrier counter-damage."""

    def test_direct_attack_on_ambush_raises_illegal(self):
        eng = mkengine()
        at = mkunit(eng, 900, attack=3, life=5); at.can_attack = True
        am = mkunit(eng, 901, keywords=frozenset({"潜行"}), attack=1, life=3)
        eng.players[0].board = [at]; eng.players[1].board = [am]
        hp0 = eng.players[0].health; hp1 = eng.players[1].health
        with self.assertRaises(IllegalCommand):
            eng.apply(Attack(0, at.entity_id, am.entity_id))
        self.assertEqual(eng.players[0].health, hp0)
        self.assertEqual(eng.players[1].health, hp1)
        self.assertEqual(am.health, 3)

    def test_counter_damage_blocked_by_barrier_zero_in_event(self):
        eng = mkengine()
        at = mkunit(eng, 900, attack=3, life=5); at.can_attack = True
        at.barrier_charges = 1
        df = mkunit(eng, 901, attack=4, life=5)
        eng.players[0].board = [at]; eng.players[1].board = [df]
        transition = eng.apply(Attack(0, at.entity_id, df.entity_id))
        counter_events = [e for e in transition.events if e.type == EventType.DAMAGE_DEALT and e.source_id == df.entity_id]
        self.assertTrue(any(e.amount == 0 for e in counter_events))
        self.assertEqual(at.health, 5)


if __name__ == "__main__":
    unittest.main()
