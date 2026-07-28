from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from swb.rl.fixed_decks import (
    FixedTrainingDeck,
    get_fixed_training_deck,
)


DECK_MATCHUP_SCHEDULE_VERSION = 1


@dataclass(frozen=True)
class DeckMatchup:
    episode_id: int
    learner_player: int
    learner_deck: FixedTrainingDeck
    opponent_deck: FixedTrainingDeck

    @property
    def decks(self) -> tuple[FixedTrainingDeck, FixedTrainingDeck]:
        if self.learner_player == 0:
            return self.learner_deck, self.opponent_deck
        return self.opponent_deck, self.learner_deck

    @property
    def class_ids(self) -> tuple[int, int]:
        return tuple(deck.class_id for deck in self.decks)

    def manifest(self) -> dict[str, object]:
        class_a, class_b = self.class_ids
        return {
            "episode_id": self.episode_id,
            "learner_player": self.learner_player,
            "learner_deck": self.learner_deck.name,
            "opponent_deck": self.opponent_deck.name,
            "class_a": class_a,
            "class_b": class_b,
            "learner_class": self.learner_deck.class_id,
            "opponent_class": self.opponent_deck.class_id,
        }


def normalize_opponent_decks(
    learner_deck: str,
    opponent_decks: Iterable[str],
) -> tuple[str, ...]:
    learner = get_fixed_training_deck(learner_deck)
    normalized = tuple(opponent_decks)
    if not normalized:
        raise ValueError("opponent_decks must contain at least one deck")
    if any(not isinstance(name, str) or not name for name in normalized):
        raise ValueError("opponent_decks must contain non-empty deck names")
    if len(set(normalized)) != len(normalized):
        raise ValueError("opponent_decks must not contain duplicates")
    if learner.name in normalized:
        raise ValueError("learner deck must not appear in opponent_decks")
    for name in normalized:
        get_fixed_training_deck(name)
    return normalized


def deck_matchup_for_episode(
    learner_deck: str,
    opponent_decks: Iterable[str],
    episode_id: int,
) -> DeckMatchup:
    if (
        not isinstance(episode_id, int)
        or isinstance(episode_id, bool)
        or episode_id < 0
    ):
        raise ValueError("episode_id must be a non-negative integer")
    opponents = normalize_opponent_decks(learner_deck, opponent_decks)
    opponent_index = (episode_id // 2) % len(opponents)
    return DeckMatchup(
        episode_id=episode_id,
        learner_player=episode_id % 2,
        learner_deck=get_fixed_training_deck(learner_deck),
        opponent_deck=get_fixed_training_deck(opponents[opponent_index]),
    )
