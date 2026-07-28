from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from swb.engine.events import EventType, GameEvent
from swb.engine.state import HandCard
from swb.simulator import MatchSimulator
from swb.simulator.history import MatchHistoryStore
from swb.simulator.timeline import build_animation_cues, serialize_event


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    (PROJECT_ROOT / "data" / "checkpoints" / "ppo_evolve_haven_100k.pt").is_file()
    and (PROJECT_ROOT / "data" / "card_images").is_dir(),
    "local simulator checkpoint and card images are required",
)
class MatchSimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history_directory = tempfile.TemporaryDirectory()
        cls.simulator = MatchSimulator(
            database=PROJECT_ROOT / "data" / "cards.sqlite3",
            checkpoint=PROJECT_ROOT
            / "data"
            / "checkpoints"
            / "ppo_evolve_haven_100k.pt",
            card_catalog=PROJECT_ROOT / "shadowverse_cards.json",
            image_directory=PROJECT_ROOT / "data" / "card_images",
            history_directory=cls.history_directory.name,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.history_directory.cleanup()

    def test_new_match_hides_ai_hand_and_waits_for_human(self) -> None:
        state = self.simulator.new_match(seed=7, human_player=0)

        self.assertEqual(state["deck"]["name"], "official_qr_evolve_haven_20260727")
        self.assertTrue(state["match_id"])
        self.assertTrue(state["human_turn"])
        self.assertIsNotNone(state["players"][0]["hand"])
        self.assertIsNone(state["players"][1]["hand"])
        self.assertTrue(state["actions"])
        self.assertEqual(state["players"][0]["leader_area_limit"], 5)
        self.assertIn("cards_played_this_turn", state["players"][0])
        self.assertIn("overflow_active", state["players"][0])
        ai_marker = f"[玩家 {state['ai_player'] + 1}]"
        ai_opening_logs = [
            line
            for line in state["logs"]
            if ai_marker in line and "起手：" in line
        ]
        self.assertTrue(ai_opening_logs)
        self.assertTrue(
            all(line.endswith("起手：隐藏卡牌") for line in ai_opening_logs)
        )

    def test_legal_human_action_advances_without_mask_mismatch(self) -> None:
        state = self.simulator.new_match(seed=11, human_player=0)
        result = self.simulator.apply_human_action(state["actions"][0]["id"])

        self.assertIn("players", result)
        self.assertTrue(
            result["human_turn"] or result["terminated"] or result["truncated"]
        )
        self.assertTrue(result["animation_batch"])
        record = self.simulator.match_history(result["match_id"])
        self.assertGreaterEqual(len(record["actions"]), 1)
        self.assertTrue(record["actions"][-1]["animations"])
        self.assertTrue(
            (Path(self.history_directory.name) / f"{result['match_id']}.json").is_file()
        )

    def test_history_persists_private_state_and_complete_policy_decisions(self) -> None:
        state = self.simulator.new_match(seed=23, human_player=0)
        record = self.simulator.match_history(state["match_id"])

        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(record["privacy"]["persistence"], "full")
        self.assertIsNotNone(
            record["initial_state"]["players"][state["ai_player"]]["hand"]
        )
        ai_marker = f"[玩家 {state['ai_player'] + 1}]"
        ai_opening_logs = [
            line
            for line in record["logs"]
            if ai_marker in line and "起手：" in line
        ]
        self.assertTrue(ai_opening_logs)
        self.assertTrue(
            all("隐藏卡牌" not in line for line in ai_opening_logs)
        )

        ai_action = next(
            action
            for action in record["actions"]
            if action["actor_role"] == "ai"
        )
        decision = ai_action["decision"]
        self.assertEqual(decision["type"], "ppo_argmax")
        self.assertIsInstance(decision["value"], float)
        self.assertEqual(
            decision["selected_action_id"],
            ai_action["action_id"],
        )
        self.assertGreater(len(decision["legal_actions"]), 1)
        self.assertAlmostEqual(
            sum(
                candidate["probability"]
                for candidate in decision["legal_actions"]
            ),
            1.0,
            places=6,
        )
        selected = [
            candidate
            for candidate in decision["legal_actions"]
            if candidate["selected"]
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], ai_action["action_id"])
        self.assertAlmostEqual(
            selected[0]["probability"],
            decision["selected_probability"],
        )

    def test_human_decision_records_all_legal_actions_without_fake_probabilities(
        self,
    ) -> None:
        state = self.simulator.new_match(seed=29, human_player=0)
        self.simulator.apply_human_action(state["actions"][0]["id"])
        record = self.simulator.match_history(state["match_id"])
        human_action = next(
            action
            for action in record["actions"]
            if action["actor_role"] == "human"
        )

        self.assertEqual(human_action["decision"]["type"], "human")
        self.assertIsNone(human_action["decision"]["value"])
        self.assertTrue(human_action["decision"]["legal_actions"])
        self.assertTrue(
            all(
                candidate["probability"] is None
                and candidate["logit"] is None
                for candidate in human_action["decision"]["legal_actions"]
            )
        )

    def test_image_lookup_rejects_traversal(self) -> None:
        self.assertIsNone(self.simulator.image_path("../shadowverse_cards.json"))
        state = self.simulator.new_match(seed=13, human_player=0)
        filename = Path(state["players"][0]["hand"][0]["image_url"]).name
        self.assertTrue(self.simulator.image_path(filename).is_file())

    def test_hand_card_serializes_union_burst_progress(self) -> None:
        definition = self.simulator.assets.catalog.resolve(10413110)
        self.assertIsNotNone(definition)
        card = HandCard(
            definition=definition,
            entity_id=123,
            evolutions_while_in_hand=3,
        )

        serialized = self.simulator._serialize_hand_card(
            0,
            card,
            turns_started=7,
        )

        self.assertEqual(
            serialized["union_bursts"],
            [
                {
                    "kind": "union_burst",
                    "label": "奥义",
                    "gauge": 10,
                    "threshold": 10,
                    "remaining": 0,
                    "ready": True,
                },
                {
                    "kind": "super_skybound_art",
                    "label": "解放奥义",
                    "gauge": 10,
                    "threshold": 15,
                    "remaining": 5,
                    "ready": False,
                },
            ],
        )

    def test_starting_new_match_marks_previous_record_abandoned(self) -> None:
        previous = self.simulator.new_match(seed=17, human_player=0)
        current = self.simulator.new_match(seed=19, human_player=0)

        self.assertNotEqual(previous["match_id"], current["match_id"])
        record = self.simulator.match_history(previous["match_id"])
        self.assertEqual(record["status"], "abandoned")
        summaries = self.simulator.list_history()["matches"]
        self.assertIn(current["match_id"], {item["match_id"] for item in summaries})

    def test_history_lookup_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            self.simulator.match_history("../outside")


class MatchTimelineTests(unittest.TestCase):
    def test_attack_and_damage_events_create_readable_animation_cues(self) -> None:
        events = [
            GameEvent(
                EventType.ATTACK_DECLARED,
                0,
                source_id=10,
                target_id=20,
            ),
            GameEvent(
                EventType.DAMAGE_APPLIED,
                0,
                source_id=10,
                target_id=20,
                amount=3,
            ),
        ]
        serialized = [
            serialize_event(
                event,
                entity_names={10: "攻击者", 20: "防守者"},
                card_lookup=lambda _: None,
            )
            for event in events
        ]

        cues = build_animation_cues(
            serialized,
            logs=["攻击者 攻击 防守者，造成 3 点伤害"],
            action_label="攻击",
        )

        self.assertEqual([cue["kind"] for cue in cues], ["attack", "damage"])
        self.assertEqual(cues[0]["title"], "攻击者 → 防守者")
        self.assertIn("3 点伤害", cues[1]["title"])

    def test_spell_play_and_resolution_identify_the_card(self) -> None:
        spell = SimpleNamespace(
            name="测试法术",
            card_id=900001,
            card_type="法术",
        )
        serialized = [
            serialize_event(
                GameEvent(
                    EventType.CARD_PLAYED,
                    1,
                    source_id=30,
                    metadata={
                        "card": spell,
                        "card_id": spell.card_id,
                        "source_cost": 2,
                    },
                ),
                entity_names={},
                card_lookup=lambda _: spell,
            )
        ]

        cues = build_animation_cues(
            serialized,
            logs=["玩家 2 使用法术 测试法术"],
            action_label="打出测试法术",
        )

        self.assertEqual(cues[0]["kind"], "spell")
        self.assertIn("测试法术", cues[0]["title"])
        self.assertEqual(cues[0]["detail"], "法术 · 2 PP")


class MatchHistoryStoreTests(unittest.TestCase):
    def test_schema_v1_record_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = MatchHistoryStore(directory)
            match_id = "20260727T143423334011Z-e416cb43"
            payload = {
                "schema_version": 1,
                "match_id": match_id,
            }
            (Path(directory) / f"{match_id}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            self.assertEqual(store.load(match_id), payload)


if __name__ == "__main__":
    unittest.main()
