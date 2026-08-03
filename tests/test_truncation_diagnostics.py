from __future__ import annotations

import unittest

from swb.rl.truncation_diagnostics import analyze_truncated_trace


def _step(
    before: str,
    after: str,
    signature: str,
    kind: str,
    *,
    player: int = 0,
) -> dict[str, object]:
    return {
        "before_state_sha256": before,
        "after_state_sha256": after,
        "action_signature": signature,
        "action_kind": kind,
        "action_label": signature,
        "player_id": player,
    }


class TruncationDiagnosticsTests(unittest.TestCase):
    def test_classifies_alternating_graveyard_pages_as_navigation_loop(self) -> None:
        steps = []
        for _ in range(12):
            steps.append(
                _step(
                    "page-0",
                    "page-1",
                    "graveyard_page_next",
                    "graveyard_page_next",
                )
            )
            steps.append(
                _step(
                    "page-1",
                    "page-0",
                    "graveyard_page_prev",
                    "graveyard_page_prev",
                )
            )

        analysis = analyze_truncated_trace(steps, tail_window=16)

        self.assertEqual(
            analysis["classification"],
            "graveyard_page_navigation_loop",
        )
        self.assertEqual(analysis["max_state_visits"], 13)
        self.assertEqual(analysis["longest_page_navigation_streak"], 24)
        self.assertEqual(analysis["dominant_cycle"]["period"], 2)

    def test_classifies_repeated_choice_cycle(self) -> None:
        steps = []
        for _ in range(5):
            steps.extend((
                _step("ready", "choosing", "fusion:100", "fusion"),
                _step("choosing", "ready", "choice:cancel", "choice"),
            ))

        analysis = analyze_truncated_trace(steps)

        self.assertEqual(
            analysis["classification"],
            "fusion_or_choice_state_cycle",
        )
        self.assertGreaterEqual(
            analysis["dominant_cycle"]["repeat_evidence"],
            3,
        )

    def test_reports_step_budget_without_repeated_state(self) -> None:
        steps = [
            _step(f"state-{index}", f"state-{index + 1}", f"play:{index}", "play")
            for index in range(40)
        ]

        analysis = analyze_truncated_trace(steps)

        self.assertEqual(
            analysis["classification"],
            "step_budget_without_dominant_cycle",
        )
        self.assertEqual(analysis["revisited_strategic_states"], 0)

    def test_rejects_empty_trace_and_invalid_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            analyze_truncated_trace([])
        with self.assertRaisesRegex(ValueError, "positive"):
            analyze_truncated_trace(
                [_step("a", "b", "end_turn", "end_turn")],
                tail_window=0,
            )


if __name__ == "__main__":
    unittest.main()
