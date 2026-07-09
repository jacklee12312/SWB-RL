from __future__ import annotations

import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Attack, Choose, EndTurn, Evolve, PlayCard, SuperEvolve
from swb.engine.effects import EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine
from swb.engine.state import Unit


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


def mkunit(eng, cid, **kw):
    kws = kw.pop("keywords", frozenset())
    return Unit.summon(card(cid, keywords=kws, **kw), entity_id=eng.state.allocate_entity_id())


class AttackTriggerTests(unittest.TestCase):
    def test_attack_trigger_fires_only_for_attacker(self):
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.ATTACK, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=2),
        ),),))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"攻击时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=3)
        eng.players[0].board = [a]; eng.players[1].board = [d]
        hp_before = eng.players[1].health
        eng.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertEqual(eng.players[1].health, hp_before - 2)

    def test_attack_trigger_preserves_combat(self):
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.ATTACK, operations=(
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"攻击时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=3)
        eng.players[0].board = [a]; eng.players[1].board = [d]
        eng.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertEqual(d.health, 0)

    def test_attack_trigger_with_conditions(self):
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.ATTACK, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=2),
        ),),))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"攻击时"})); a.can_attack = True
        eng.players[0].board = [a]
        eng.apply(Attack(0, a.entity_id, None))
        self.assertEqual(eng.players[1].health, 15)


class ClashTriggerTests(unittest.TestCase):
    def test_clash_triggers_both_sides(self):
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.CLASH, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.CLASH, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"交战时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=3, keywords=frozenset({"交战时"}))
        eng.players[0].board = [a]; eng.players[1].board = [d]
        hp_before = eng.players[1].health
        hand_before = len(eng.players[1].hand)
        eng.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertEqual(eng.players[1].health, hp_before - 1)
        self.assertEqual(len(eng.players[1].hand), hand_before + 1)


class EvolveTriggerTests(unittest.TestCase):
    def test_evolve_trigger_fires_after_stat_change(self):
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.EVOLVE, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=3),
        ),),))
        eng = mkengine(rulebook=rulebook)
        u = mkunit(eng, 900, attack=2, life=3, keywords=frozenset({"进化时"}))
        eng.players[0].board = [u]
        eng.players[0].turns_started = 4
        hp_before = eng.players[1].health
        eng.apply(Evolve(0, u.entity_id))
        self.assertEqual(eng.players[1].health, hp_before - 3)
        self.assertEqual(u.attack, 4)
        self.assertEqual(u.health, 5)

    def test_super_evolve_trigger_fires_after_stat_change(self):
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.SUPER_EVOLVE, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=3),
        ),),))
        eng = mkengine(rulebook=rulebook)
        u = mkunit(eng, 900, attack=2, life=3, keywords=frozenset({"超进化时"}))
        eng.players[0].board = [u]
        eng.players[0].turns_started = eng.config.first_player_super_evolution_unlock_turn
        hp_before = eng.players[1].health
        eng.apply(SuperEvolve(0, u.entity_id))
        self.assertTrue(u.evolved)
        self.assertTrue(u.super_evolved)
        self.assertEqual(eng.players[1].health, hp_before - 3)
        self.assertEqual(u.attack, 4)
        self.assertEqual(u.health, 5)
        self.assertTrue(any(e.type is EventType.FOLLOWER_SUPER_EVOLVED for e in eng.event_history))


class TurnTriggerTests(unittest.TestCase):
    """Tests for TURN_START and TURN_END triggers via rulebook."""

    def test_turn_start_trigger(self):
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.TURN_START, operations=(
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        u = mkunit(eng, 900, attack=1, life=3)
        eng.players[0].board = [u]
        hand_before = len(eng.players[0].hand)
        eng.apply(EndTurn(0))
        eng.apply(EndTurn(1))
        self.assertGreater(len(eng.players[0].hand), hand_before)

    def test_turn_end_trigger(self):
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.TURN_END, operations=(
            EffectOperation(kind=EffectKind.HEAL_LEADER, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        u = mkunit(eng, 900, attack=1, life=3)
        eng.players[0].board = [u]
        eng.players[0].health = 15
        eng.apply(EndTurn(0))
        self.assertEqual(eng.players[0].health, 16)


class TriggerDeterminismTests(unittest.TestCase):
    def test_same_seed_same_rules_produces_identical_logs(self):
        import hashlib
        def hash_run():
            rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.ATTACK, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_LEADER, target=TargetKind.ENEMY_LEADER, amount=2),
            ),),))
            eng = GameEngine([card(i) for i in range(100,140)], [card(i) for i in range(200,240)], class_a=1, class_b=1, seed=7, rulebook=rulebook)
            eng.reset(seed=7)
            a = mkunit(eng, 900, attack=3, life=5); a.can_attack = True
            eng.players[0].board = [a]
            eng.apply(Attack(0, a.entity_id, None))
            return hashlib.sha256("\n".join(eng.logs).encode()).hexdigest()
        self.assertEqual(hash_run(), hash_run())


class TriggerCompatTests(unittest.TestCase):
    """Existing triggers (PLAY, FANFARE, LAST_WORDS, COUNTDOWN_EXPIRED) still work."""

    def test_play_trigger_still_works(self):
        rulebook = RuleBook((CardRule(card_id=1, trigger=Trigger.PLAY, operations=(
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(1, card_type="法术", attack=None, life=None)
        eng.apply(PlayCard(0, 0))

    def test_fanfare_still_works(self):
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.FANFARE, operations=(
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        u = mkunit(eng, 900, attack=1, life=3)
        eng.players[0].board = [u]
        eng.players[0].mana = 10
        eng.players[0].hand[0] = card(900)
        eng.apply(PlayCard(0, 0))


class TriggerContinuationTests(unittest.TestCase):
    """Trigger effects that require a target choice must pause and resume."""

    def test_attack_trigger_choice_pauses_combat_then_resumes(self):
        """Attack trigger with choice: combat damage must wait for choice."""
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.ATTACK, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_BOARD, amount=2),
        ),),))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"攻击时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=5)
        eng.players[0].board = [a]; eng.players[1].board = [d]
        hp_before = eng.players[1].health
        eng.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertIsNotNone(eng.state.pending_choice)
        self.assertEqual(d.health, 5)
        choice = next(cmd for cmd in eng.legal_commands() if isinstance(cmd, Choose))
        eng.apply(choice)
        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(d.health, 0)

    def test_attack_leader_trigger_choice_pauses_then_resumes(self):
        """Attack leader with choice trigger: leader damage must wait for choice."""
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.ATTACK, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_BOARD, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"攻击时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=2)
        eng.players[0].board = [a]; eng.players[1].board = [d]
        hp_before = eng.players[1].health
        eng.apply(Attack(0, a.entity_id, None))
        self.assertIsNotNone(eng.state.pending_choice)
        self.assertEqual(eng.players[1].health, hp_before)
        choice = next(cmd for cmd in eng.legal_commands() if isinstance(cmd, Choose))
        eng.apply(choice)
        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(eng.players[1].health, hp_before - 3)

    def test_clash_trigger_choice_pauses_combat(self):
        """Clash trigger with choice: combat damage must wait for choice."""
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.CLASH, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_BOARD, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"交战时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=3)
        eng.players[0].board = [a]; eng.players[1].board = [d]
        eng.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertIsNotNone(eng.state.pending_choice)
        self.assertEqual(d.health, 3)
        choice = next(cmd for cmd in eng.legal_commands() if isinstance(cmd, Choose))
        eng.apply(choice)
        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(d.health, 0)

    def test_evolve_trigger_choice_pauses_then_resumes(self):
        """Evolve trigger with choice: effect resolves after evolution stats."""
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.EVOLVE, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_BOARD, amount=2),
        ),),))
        eng = mkengine(rulebook=rulebook)
        u = mkunit(eng, 900, attack=2, life=3, keywords=frozenset({"进化时"}))
        eng.players[0].board = [u]
        eng.players[0].turns_started = 4
        d = mkunit(eng, 901, attack=1, life=2)
        eng.players[1].board = [d]
        eng.apply(Evolve(0, u.entity_id))
        self.assertIsNotNone(eng.state.pending_choice)
        self.assertEqual(u.attack, 4)
        self.assertEqual(d.health, 2)
        choice = next(cmd for cmd in eng.legal_commands() if isinstance(cmd, Choose))
        eng.apply(choice)
        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(d.health, 0)

    def test_turn_end_trigger_choice_pauses_turn_transition(self):
        """Turn end trigger with choice: turn end effects wait for choice."""
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.TURN_END, operations=(
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_BOARD, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        u = mkunit(eng, 900, attack=1, life=3)
        eng.players[0].board = [u]
        hand_before = len(eng.players[0].hand)
        eng.apply(EndTurn(0))
        self.assertIsNotNone(eng.state.pending_choice)
        self.assertEqual(eng.state.active_player, 0)
        choice = next(cmd for cmd in eng.legal_commands() if isinstance(cmd, Choose))
        eng.apply(choice)
        self.assertIsNone(eng.state.pending_choice)

    def test_turn_end_choice_target_entered_own_graveyard_skips_and_resumes(self):
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.TURN_END, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.OWN_UNIT, amount=1),
            EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
        ),),))
        eng = mkengine(rulebook=rulebook)
        source = mkunit(eng, 900, attack=1, life=3)
        stale_target = mkunit(eng, 901, attack=1, life=3)
        eng.players[0].board = [source, stale_target]
        eng.apply(EndTurn(0))
        self.assertIsNotNone(eng.state.pending_choice)
        choice = next(
            cmd for cmd in eng.legal_commands()
            if isinstance(cmd, Choose)
            and cmd.option_id == f"entity:{stale_target.entity_id}"
        )
        eng.players[0].board.remove(stale_target)
        eng._send_to_graveyard(
            0,
            stale_target.definition,
            "test_target_left_play",
            source_entity_id=stale_target.entity_id,
        )
        deck_before = len(eng.players[0].deck)

        eng.apply(choice)

        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(eng.state.active_player, 1)
        self.assertEqual(stale_target.health, 3)
        self.assertEqual(len(eng.players[0].deck), deck_before - 1)

    def test_target_leaves_play_during_trigger_choice(self):
        """When attack target is destroyed by trigger, combat is cancelled silently."""
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.ATTACK, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_ENEMY_UNITS, amount=5),
        ),),))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=2, life=5, keywords=frozenset({"攻击时"})); a.can_attack = True
        d1 = mkunit(eng, 901, attack=1, life=2)
        eng.players[0].board = [a]; eng.players[1].board = [d1]
        hp_before = eng.players[1].health
        eng.apply(Attack(0, a.entity_id, d1.entity_id))
        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(eng.players[1].health, hp_before)

    def test_attacker_destroyed_by_trigger_cancels_combat(self):
        """When attacker is destroyed by own non-choice trigger, combat cancels."""
        rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.ATTACK, operations=(
            EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_OWN_UNITS, amount=5),
        ),),))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=2, life=2, keywords=frozenset({"攻击时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=5)
        eng.players[0].board = [a]; eng.players[1].board = [d]
        eng.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(d.health, 5)

    def test_bilateral_clash_choice_pauses_both_sides(self):
        """Both sides CLASH: attacker choice first, then defender auto-resolve."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.CLASH, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_BOARD, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.CLASH, operations=(
                EffectOperation(kind=EffectKind.DRAW, target=TargetKind.OWN_LEADER, amount=1),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"交战时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=5, keywords=frozenset({"交战时"}))
        eng.players[0].board = [a]; eng.players[1].board = [d]
        hand_before = len(eng.players[1].hand)
        eng.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertIsNotNone(eng.state.pending_choice)
        choice = next(cmd for cmd in eng.legal_commands() if isinstance(cmd, Choose))
        eng.apply(choice)
        self.assertIsNone(eng.state.pending_choice)
        self.assertGreater(len(eng.players[1].hand), hand_before)
        self.assertEqual(d.health, 1)

    def test_defender_clash_fires_after_attacker_clash(self):
        """After attacker CLASH choice resolves, defender's non-choice CLASH fires."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.CLASH, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_BOARD, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.CLASH, operations=(
                EffectOperation(kind=EffectKind.HEAL_LEADER, target=TargetKind.OWN_LEADER, amount=2),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"交战时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=5, keywords=frozenset({"交战时"}))
        eng.players[0].board = [a]; eng.players[1].board = [d]
        eng.players[1].health = 10
        eng.apply(Attack(0, a.entity_id, d.entity_id))
        choice = next(cmd for cmd in eng.legal_commands() if isinstance(cmd, Choose))
        eng.apply(choice)
        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(eng.players[1].health, 12)

    def test_bilateral_choice_clash_both_sides(self):
        """Both sides have choice CLASH: both pause and resolve in order."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.CLASH, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_BOARD, amount=1),
            ),),
            CardRule(card_id=901, trigger=Trigger.CLASH, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_BOARD, amount=2),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"交战时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=5, keywords=frozenset({"交战时"}))
        eng.players[0].board = [a]; eng.players[1].board = [d]
        eng.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertIsNotNone(eng.state.pending_choice)
        c1 = next(cmd for cmd in eng.legal_commands() if isinstance(cmd, Choose))
        eng.apply(c1)
        self.assertIsNotNone(eng.state.pending_choice)
        c2 = next(cmd for cmd in eng.legal_commands() if isinstance(cmd, Choose))
        eng.apply(c2)
        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(a.health, 2)
        self.assertEqual(d.health, 1)

    def test_clash_kills_attacker_cancels_combat(self):
        """When CLASH effect kills attacker, combat is cancelled entirely."""
        rulebook = RuleBook((
            CardRule(card_id=900, trigger=Trigger.CLASH, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ALL_OWN_UNITS, amount=5),
            ),),
        ))
        eng = mkengine(rulebook=rulebook)
        a = mkunit(eng, 900, attack=3, life=2, keywords=frozenset({"交战时"})); a.can_attack = True
        d = mkunit(eng, 901, attack=1, life=5)
        eng.players[0].board = [a]; eng.players[1].board = [d]
        eng.apply(Attack(0, a.entity_id, d.entity_id))
        self.assertIsNone(eng.state.pending_choice)
        self.assertEqual(d.health, 5)

    def test_same_seed_choice_trigger_reproducible(self):
        """Deterministic choice-trigger: same seed + same choice → identical logs."""
        import hashlib
        def hash_run():
            rulebook = RuleBook((CardRule(card_id=900, trigger=Trigger.ATTACK, operations=(
                EffectOperation(kind=EffectKind.DAMAGE_UNIT, target=TargetKind.ENEMY_BOARD, amount=1),
            ),),))
            eng = GameEngine([card(i) for i in range(100,140)], [card(i) for i in range(200,240)], class_a=1, class_b=1, seed=7, rulebook=rulebook)
            eng.reset(seed=7)
            a = mkunit(eng, 900, attack=3, life=5, keywords=frozenset({"攻击时"})); a.can_attack = True
            d = mkunit(eng, 901, attack=1, life=3)
            eng.players[0].board = [a]; eng.players[1].board = [d]
            eng.apply(Attack(0, a.entity_id, d.entity_id))
            choice = next(cmd for cmd in eng.legal_commands() if isinstance(cmd, Choose))
            eng.apply(choice)
            return hashlib.sha256("\n".join(eng.logs).encode()).hexdigest()
        self.assertEqual(hash_run(), hash_run())


if __name__ == "__main__":
    unittest.main()
