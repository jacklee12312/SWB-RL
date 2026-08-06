from __future__ import annotations

from collections.abc import Iterable

from swb.engine.deck import PLAYABLE_CLASS_IDS


ALL_CLASS_IDS = tuple(sorted(PLAYABLE_CLASS_IDS))
CLASS_SCHEDULE_VERSION = 1


def normalize_class_ids(class_ids: Iterable[int]) -> tuple[int, ...]:
    """Validate and freeze an ordered set of playable class IDs."""
    normalized = tuple(class_ids)
    if not normalized:
        raise ValueError("class_ids must contain at least one playable class")
    if any(
        not isinstance(class_id, int)
        or isinstance(class_id, bool)
        or class_id not in PLAYABLE_CLASS_IDS
        for class_id in normalized
    ):
        raise ValueError(
            f"class_ids must contain only {list(ALL_CLASS_IDS)}, got "
            f"{list(normalized)}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("class_ids must not contain duplicates")
    return normalized


def class_pair_for_episode(
    class_ids: Iterable[int],
    episode_id: int,
) -> tuple[int, int]:
    """Return the deterministic ordered round-robin pair for an episode.

    A complete cycle contains ``len(class_ids) ** 2`` episodes and visits every
    ordered matchup exactly once.  The schedule depends only on episode ID, so
    worker completion order cannot bias the class distribution.
    """
    if not isinstance(episode_id, int) or isinstance(episode_id, bool):
        raise ValueError("episode_id must be a non-negative integer")
    if episode_id < 0:
        raise ValueError("episode_id must be a non-negative integer")
    classes = normalize_class_ids(class_ids)
    pair_index = episode_id % (len(classes) ** 2)
    return (
        classes[pair_index // len(classes)],
        classes[pair_index % len(classes)],
    )
