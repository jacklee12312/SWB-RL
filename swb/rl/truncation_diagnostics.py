from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence


PAGE_ACTION_KINDS = frozenset({
    "graveyard_page_next",
    "graveyard_page_prev",
})


def _canonical_cycle(signatures: Sequence[str]) -> tuple[str, ...]:
    """Return one rotation-independent representation of a cycle."""

    values = tuple(signatures)
    if not values:
        return ()
    rotations = tuple(
        values[index:] + values[:index]
        for index in range(len(values))
    )
    return min(rotations)


def _longest_kind_streak(
    steps: Sequence[Mapping[str, object]],
    kinds: frozenset[str],
) -> int:
    longest = 0
    current = 0
    for step in steps:
        if str(step["action_kind"]) in kinds:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def analyze_truncated_trace(
    steps: Sequence[Mapping[str, object]],
    *,
    tail_window: int = 256,
    maximum_cycle_period: int = 32,
) -> dict[str, object]:
    """Classify repeated actions and strategic-state cycles in one trace.

    A trace step must contain the strategic state hashes before and after the
    action plus a normalized action signature/kind.  The strategic hash is
    deliberately expected to exclude counters such as total agent steps and
    append-only event history, otherwise a genuine policy loop could never
    revisit an earlier state.
    """

    if tail_window <= 0 or maximum_cycle_period <= 0:
        raise ValueError("tail_window and maximum_cycle_period must be positive")
    if not steps:
        raise ValueError("trace must contain at least one step")

    state_sequence = [str(steps[0]["before_state_sha256"])]
    state_sequence.extend(str(step["after_state_sha256"]) for step in steps)
    occurrences: dict[str, list[int]] = defaultdict(list)
    for index, state_hash in enumerate(state_sequence):
        occurrences[state_hash].append(index)

    revisited_states = {
        state_hash: indices
        for state_hash, indices in occurrences.items()
        if len(indices) > 1
    }
    max_state_visits = max(len(indices) for indices in occurrences.values())

    cycle_counts: Counter[tuple[str, ...]] = Counter()
    cycle_samples: dict[tuple[str, ...], tuple[int, int]] = {}
    for indices in revisited_states.values():
        for start, stop in zip(indices, indices[1:]):
            period = stop - start
            if period <= 0 or period > maximum_cycle_period:
                continue
            signatures = [
                str(step["action_signature"])
                for step in steps[start:stop]
            ]
            cycle = _canonical_cycle(signatures)
            cycle_counts[cycle] += 1
            cycle_samples.setdefault(cycle, (start, stop))

    ranked_cycles = sorted(
        cycle_counts.items(),
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )
    cycle_summaries: list[dict[str, object]] = []
    for signatures, repeat_evidence in ranked_cycles[:8]:
        start, stop = cycle_samples[signatures]
        sample = steps[start:stop]
        sample_signatures = [
            str(step["action_signature"])
            for step in sample
        ]
        cycle_summaries.append({
            "period": len(signatures),
            "repeat_evidence": repeat_evidence,
            "canonical_action_cycle": list(signatures),
            "action_sequence": sample_signatures,
            "action_kinds": [str(step["action_kind"]) for step in sample],
            "action_labels": [str(step["action_label"]) for step in sample],
            "players": [int(step["player_id"]) for step in sample],
            "sample_step_range": [start, stop],
        })

    tail = list(steps[-min(tail_window, len(steps)) :])
    tail_kind_counts = Counter(str(step["action_kind"]) for step in tail)
    tail_signature_counts = Counter(
        str(step["action_signature"]) for step in tail
    )
    page_steps = sum(
        count for kind, count in tail_kind_counts.items() if kind in PAGE_ACTION_KINDS
    )
    tail_page_fraction = page_steps / len(tail)
    top_signature, top_signature_count = tail_signature_counts.most_common(1)[0]
    top_signature_fraction = top_signature_count / len(tail)
    dominant_cycle = cycle_summaries[0] if cycle_summaries else None

    if (
        tail_page_fraction >= 0.8
        or (
            dominant_cycle is not None
            and int(dominant_cycle["repeat_evidence"]) >= 3
            and set(dominant_cycle["action_kinds"]).issubset(PAGE_ACTION_KINDS)
        )
    ):
        classification = "graveyard_page_navigation_loop"
    elif dominant_cycle is not None and int(dominant_cycle["repeat_evidence"]) >= 3:
        cycle_kinds = set(dominant_cycle["action_kinds"])
        if cycle_kinds.issubset({"choice", "fusion"}):
            classification = "fusion_or_choice_state_cycle"
        else:
            classification = "exact_strategic_state_cycle"
    elif top_signature_fraction >= 0.8:
        classification = "repeated_action_without_exact_state_cycle"
    else:
        classification = "step_budget_without_dominant_cycle"

    return {
        "classification": classification,
        "steps": len(steps),
        "unique_strategic_states": len(occurrences),
        "revisited_strategic_states": len(revisited_states),
        "max_state_visits": max_state_visits,
        "unchanged_state_actions": sum(
            str(step["before_state_sha256"])
            == str(step["after_state_sha256"])
            for step in steps
        ),
        "tail_window": len(tail),
        "tail_page_fraction": tail_page_fraction,
        "longest_page_navigation_streak": _longest_kind_streak(
            steps, PAGE_ACTION_KINDS
        ),
        "tail_action_kinds": dict(tail_kind_counts.most_common()),
        "tail_action_signatures": dict(tail_signature_counts.most_common(12)),
        "top_tail_action_signature": top_signature,
        "top_tail_action_fraction": top_signature_fraction,
        "dominant_cycle": dominant_cycle,
        "cycle_candidates": cycle_summaries,
    }
