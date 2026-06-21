from __future__ import annotations

import json
import os
import tempfile
import unittest

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import Choose, PlayCard
from swb.engine.effects import DeckFilter, EffectKind, EffectOperation, TargetKind
from swb.engine.events import EventType
from swb.engine.resolution import GameEngine
from swb.engine.state import HandCard


def card(cid: int, **kw) -> CardDefinition:
    return CardDefinition(
        card_id=cid,
        card_set_id=kw.get("card_set_id", 10000),
        class_id=kw.get("class_id", 1),
        class_name=kw.get("class_name", "精灵"),
        name=kw.get("name", f"c{cid}"),
        cost=kw.get("cost", 1),
        card_type=kw.get("card_type", "随从"),
        attack=kw.get("attack", 1),
        life=kw.get("life", 1),
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def spell_rule(card_id: int, *ops: EffectOperation) -> CardRule:
    return CardRule(card_id=card_id, trigger=Trigger.PLAY, operations=ops)


def engine_with(rulebook: RuleBook) -> GameEngine:
    eng = GameEngine(
        [card(i) for i in range(100, 140)],
        [card(i) for i in range(200, 240)],
        class_a=1,
        class_b=1,
        seed=7,
        rulebook=rulebook,
    )
    eng.reset(seed=7)
    return eng


def put_spell_in_hand(engine: GameEngine, card_id: int) -> None:
    hc = HandCard(
        definition=card(card_id, card_type="法术", attack=None, life=None),
        entity_id=engine.state.allocate_entity_id(),
    )
    engine.players[0].hand.insert(0, hc)
    engine.players[0].hand_entity_ids.insert(0, hc.entity_id)
    engine.players[0].mana = 10


def choose_entity(engine: GameEngine, entity_id: int) -> Choose:
    for command in engine.legal_commands():
        if isinstance(command, Choose) and command.option_id == f"hand:{entity_id}":
            return command
    raise AssertionError(f"choice for hand entity {entity_id} not found")


class FilteredDrawTests(unittest.TestCase):
    def test_draw_filtered_pulls_matching_type_and_class(self):
        rb = RuleBook((
            spell_rule(
                1,
                EffectOperation(
                    EffectKind.DRAW_FILTERED,
                    TargetKind.OWN_LEADER,
                    amount=2,
                    deck_filter=DeckFilter(card_type="随从", class_id=2),
                ),
            ),
        ))
        eng = engine_with(rb)
        eng.players[0].deck = [
            card(10, class_id=2, class_name="皇家护卫", card_type="法术", attack=None, life=None),
            card(11, class_id=1, class_name="精灵", card_type="随从"),
            card(12, class_id=2, class_name="皇家护卫", card_type="随从"),
            card(13, class_id=2, class_name="皇家护卫", card_type="随从"),
        ]
        put_spell_in_hand(eng, 1)

        eng.apply(PlayCard(0, 0))

        drawn_ids = {h.card_id for h in eng.players[0].hand}
        self.assertIn(12, drawn_ids)
        self.assertIn(13, drawn_ids)
        self.assertNotIn(10, drawn_ids)
        self.assertNotIn(11, drawn_ids)

    def test_draw_filtered_enemy_leader_draws_from_opponent_deck(self):
        rb = RuleBook((
            spell_rule(
                1,
                EffectOperation(
                    EffectKind.DRAW_FILTERED,
                    TargetKind.ENEMY_LEADER,
                    amount=1,
                    deck_filter=DeckFilter(card_type="随从", class_name="皇家护卫"),
                ),
            ),
        ))
        eng = engine_with(rb)
        eng.players[1].deck = [
            card(20, class_id=2, class_name="皇家护卫", card_type="随从"),
        ]
        put_spell_in_hand(eng, 1)

        eng.apply(PlayCard(0, 0))

        self.assertTrue(any(h.card_id == 20 for h in eng.players[1].hand))
        self.assertFalse(any(h.card_id == 20 for h in eng.players[0].hand))

    def test_draw_filtered_no_candidates_skips_without_fatigue(self):
        rb = RuleBook((
            spell_rule(
                1,
                EffectOperation(
                    EffectKind.DRAW_FILTERED,
                    TargetKind.OWN_LEADER,
                    amount=1,
                    deck_filter=DeckFilter(card_type="护符"),
                ),
            ),
        ))
        eng = engine_with(rb)
        eng.players[0].deck = [card(30, card_type="随从")]
        put_spell_in_hand(eng, 1)

        eng.apply(PlayCard(0, 0))

        self.assertEqual(eng.players[0].fatigue, 0)
        self.assertFalse(any(h.card_id == 30 for h in eng.players[0].hand))
        self.assertEqual(len(eng.players[0].deck), 1)

    def test_draw_filtered_full_hand_overdraws_to_graveyard(self):
        rb = RuleBook((
            spell_rule(
                1,
                EffectOperation(
                    EffectKind.DRAW_FILTERED,
                    TargetKind.OWN_LEADER,
                    amount=1,
                    deck_filter=DeckFilter(card_type="随从"),
                ),
            ),
        ))
        eng = engine_with(rb)
        eng.players[0].deck = [card(40, card_type="随从")]
        put_spell_in_hand(eng, 1)
        while len(eng.players[0].hand) < eng.config.max_hand + 1:
            hc = HandCard(
                definition=card(500 + len(eng.players[0].hand)),
                entity_id=eng.state.allocate_entity_id(),
            )
            eng.players[0].hand.append(hc)
            eng.players[0].hand_entity_ids.append(hc.entity_id)

        eng.apply(PlayCard(0, 0))

        self.assertTrue(any(g.definition.card_id == 40 for g in eng.players[0].graveyard))
        self.assertEqual(eng.players[0].shadows, 2)

    def test_return_own_hand_to_deck_then_draw_filtered(self):
        rb = RuleBook((
            spell_rule(
                1,
                EffectOperation(EffectKind.RETURN_TO_DECK, TargetKind.OWN_HAND),
                EffectOperation(
                    EffectKind.DRAW_FILTERED,
                    TargetKind.OWN_LEADER,
                    amount=1,
                    deck_filter=DeckFilter(card_type="随从", class_id=2),
                ),
            ),
        ))
        eng = engine_with(rb)
        returned = HandCard(
            definition=card(60, class_id=2, class_name="皇家护卫", card_type="随从"),
            entity_id=eng.state.allocate_entity_id(),
        )
        eng.players[0].hand.append(returned)
        eng.players[0].hand_entity_ids.append(returned.entity_id)
        put_spell_in_hand(eng, 1)
        eng.players[0].deck = []

        eng.apply(PlayCard(0, 0))
        eng.apply(choose_entity(eng, returned.entity_id))

        self.assertTrue(any(h.card_id == 60 for h in eng.players[0].hand))
        self.assertEqual(len(eng.players[0].deck), 0)

    def test_draw_filtered_supports_cost_card_id_and_name_filters(self):
        rb = RuleBook((
            spell_rule(
                1,
                EffectOperation(
                    EffectKind.DRAW_FILTERED,
                    TargetKind.OWN_LEADER,
                    amount=1,
                    deck_filter=DeckFilter(cost_min=2, cost_max=4, card_id=91, card_name="target"),
                ),
            ),
        ))
        eng = engine_with(rb)
        eng.players[0].deck = [
            card(90, name="target", cost=1),
            card(91, name="wrong", cost=3),
            card(91, name="target", cost=5),
            card(91, name="target", cost=3),
        ]
        put_spell_in_hand(eng, 1)

        eng.apply(PlayCard(0, 0))

        self.assertTrue(any(h.card_id == 91 and h.definition.name == "target" for h in eng.players[0].hand))
        self.assertEqual(len(eng.players[0].deck), 3)


class FilteredDrawSchemaTests(unittest.TestCase):
    def _load_payload(self, payload):
        d = tempfile.mkdtemp()
        fp = os.path.join(d, "rules.json")
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            return RuleBook.from_directory(d)
        finally:
            os.remove(fp)
            os.rmdir(d)

    def test_json_parses_draw_filtered(self):
        rb = self._load_payload({
            "rules": [{
                "card_id": 1,
                "trigger": "play",
                "operations": [{
                    "kind": "draw_filtered",
                    "target": "own_leader",
                    "amount": 2,
                    "card_type_filter": "随从",
                    "class_id_filter": 2,
                    "class_name_filter": "皇家护卫",
                }],
            }],
        })
        op = rb.operations_for(1, Trigger.PLAY)[0]
        self.assertEqual(op.kind, EffectKind.DRAW_FILTERED)
        self.assertEqual(op.deck_filter.card_type, "随从")
        self.assertEqual(op.deck_filter.class_id, 2)
        self.assertEqual(op.deck_filter.class_name, "皇家护卫")

    def test_json_parses_extended_draw_filtered_filters(self):
        rb = self._load_payload({
            "rules": [{
                "card_id": 1,
                "trigger": "play",
                "operations": [{
                    "kind": "draw_filtered",
                    "target": "own_leader",
                    "amount": 1,
                    "cost_min": 2,
                    "cost_max": 4,
                    "card_id_filter": 123,
                    "card_name_filter": "目标",
                }],
            }],
        })
        op = rb.operations_for(1, Trigger.PLAY)[0]
        self.assertEqual(op.deck_filter.cost_min, 2)
        self.assertEqual(op.deck_filter.cost_max, 4)
        self.assertEqual(op.deck_filter.card_id, 123)
        self.assertEqual(op.deck_filter.card_name, "目标")

    def test_json_rejects_draw_filtered_board_target(self):
        payload = {
            "rules": [{
                "card_id": 1,
                "trigger": "play",
                "operations": [{
                    "kind": "draw_filtered",
                    "target": "own_unit",
                    "amount": 1,
                    "card_type_filter": "随从",
                }],
            }],
        }
        with self.assertRaises(ValueError) as ctx:
            self._load_payload(payload)
        self.assertIn("draw_filtered requires", str(ctx.exception))

    def test_json_rejects_deck_filters_on_other_effects(self):
        payload = {
            "rules": [{
                "card_id": 1,
                "trigger": "play",
                "operations": [{
                    "kind": "draw",
                    "target": "own_leader",
                    "amount": 1,
                    "class_id_filter": 2,
                }],
            }],
        }
        with self.assertRaises(ValueError) as ctx:
            self._load_payload(payload)
        self.assertIn("only valid with draw_filtered", str(ctx.exception))

    def test_json_rejects_invalid_extended_filters(self):
        payload = {
            "rules": [{
                "card_id": 1,
                "trigger": "play",
                "operations": [{
                    "kind": "draw_filtered",
                    "target": "own_leader",
                    "amount": 1,
                    "cost_min": 5,
                    "cost_max": 4,
                }],
            }],
        }
        with self.assertRaises(ValueError) as ctx:
            self._load_payload(payload)
        self.assertIn("cost_min", str(ctx.exception))

    def test_json_parses_board_target_filters(self):
        rb = self._load_payload({
            "rules": [{
                "card_id": 1,
                "trigger": "play",
                "operations": [{
                    "kind": "destroy",
                    "target": "own_unit",
                    "target_card_type_filter": "随从",
                    "target_cost_min": 1,
                    "target_cost_max": 3,
                    "target_card_id_filter": 900,
                    "target_card_name_filter": "目标",
                    "target_evolved_filter": True,
                }],
            }],
        })
        op = rb.operations_for(1, Trigger.PLAY)[0]
        self.assertEqual(op.board_filter.card_type, "随从")
        self.assertEqual(op.board_filter.cost_min, 1)
        self.assertEqual(op.board_filter.cost_max, 3)
        self.assertEqual(op.board_filter.card_id, 900)
        self.assertEqual(op.board_filter.card_name, "目标")
        self.assertTrue(op.board_filter.evolved)

    def test_json_rejects_board_filters_on_non_board_targets(self):
        payload = {
            "rules": [{
                "card_id": 1,
                "trigger": "play",
                "operations": [{
                    "kind": "draw",
                    "target": "own_leader",
                    "amount": 1,
                    "target_card_id_filter": 900,
                }],
            }],
        }
        with self.assertRaises(ValueError) as ctx:
            self._load_payload(payload)
        self.assertIn("board target", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
