from __future__ import annotations

from collections.abc import Sequence

from swb.db.repository import CardDefinition
from swb.engine.origin import is_initial_deck_eligible


DECK_SIZE = 40
NEUTRAL_CLASS_ID = 0
CLASS_NAMES = {
    0: "中立",
    1: "精灵",
    2: "皇家护卫",
    3: "巫师",
    4: "龙族",
    5: "梦魇",
    6: "主教",
    7: "超越者",
}
PLAYABLE_CLASS_IDS = frozenset(CLASS_NAMES) - {NEUTRAL_CLASS_ID}


def validate_deck(
    deck: Sequence[CardDefinition],
    class_id: int,
    *,
    player_index: int,
) -> None:
    if class_id not in PLAYABLE_CLASS_IDS:
        raise ValueError(
            f"Player {player_index} class_id must be one of "
            f"{sorted(PLAYABLE_CLASS_IDS)}"
        )

    non_collectible = [
        card for card in deck if not is_initial_deck_eligible(card)
    ]
    if non_collectible:
        names = ", ".join(
            f"{card.name}({card.card_id})" for card in non_collectible[:5]
        )
        raise ValueError(
            f"Player {player_index} deck contains non-collectible cards: {names}"
        )

    off_class = [
        card
        for card in deck
        if card.class_id not in {NEUTRAL_CLASS_ID, class_id}
    ]
    if off_class:
        names = ", ".join(
            f"{card.name}({card.class_name})" for card in off_class[:5]
        )
        raise ValueError(
            f"Player {player_index} {CLASS_NAMES[class_id]} deck contains "
            f"off-class cards: {names}"
        )

    if len(deck) != DECK_SIZE:
        raise ValueError(
            f"Player {player_index} deck must contain exactly "
            f"{DECK_SIZE} cards, got {len(deck)}"
        )
