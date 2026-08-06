from __future__ import annotations

from enum import Enum

from swb.db.repository import CardDefinition

TOKEN_CARD_SET_ID = 90000


class CardOrigin(str, Enum):
    DECK = "deck"
    GENERATED = "generated"
    TOKEN = "token"
    REANIMATED = "reanimated"
    RETURNED = "returned"
    TRANSFORMED = "transformed"
    UNKNOWN = "unknown"


def is_token_definition(definition: CardDefinition) -> bool:
    return definition.card_set_id == TOKEN_CARD_SET_ID


def is_derived(origin: CardOrigin) -> bool:
    """True when the card was not originally in the deck.

    Cards generated, summoned, reanimated, or transformed are derived.
    Cards drawn from the initial deck or returned from board are not.
    """
    return origin in {
        CardOrigin.GENERATED,
        CardOrigin.TOKEN,
        CardOrigin.REANIMATED,
        CardOrigin.TRANSFORMED,
    }


def is_token(definition: CardDefinition, origin: CardOrigin | None = None) -> bool:
    """True when either the definition is a token or the origin is TOKEN."""
    if is_token_definition(definition):
        return True
    return origin is CardOrigin.TOKEN


def is_generated_card(origin: CardOrigin) -> bool:
    """True when the card was generated during play (GENERATED, TOKEN, or REANIMATED)."""
    return origin in {
        CardOrigin.GENERATED,
        CardOrigin.TOKEN,
        CardOrigin.REANIMATED,
    }


def origin_for_added_card(definition: CardDefinition) -> CardOrigin:
    """Determine the origin for a card added to hand via ADD_CARD."""
    return CardOrigin.TOKEN if is_token_definition(definition) else CardOrigin.GENERATED


def origin_for_summoned_card(definition: CardDefinition) -> CardOrigin:
    """Determine the origin for a card summoned via SUMMON."""
    return CardOrigin.TOKEN if is_token_definition(definition) else CardOrigin.GENERATED


def is_initial_deck_eligible(definition: CardDefinition) -> bool:
    """A card is eligible for an initial deck if it is collectible.

    Non-collectible cards (including card_set_id=90000 tokens) cannot
    appear in initial decks.
    """
    return definition.is_collectible and not is_token_definition(definition)


def is_reanimate_eligible(record: "DestroyedFollowerRecord") -> bool:
    """Whether a destroyed follower record is eligible for reanimate.

    The current implementation does not filter out tokens or derived
    followers.  This matches the observed behavior where SWB allows
    reanimating any destroyed follower regardless of origin.
    """
    return record.definition.card_type == "随从"


def is_graveyard_return_eligible(card: "GraveyardCard") -> bool:
    """Whether a graveyard card can be returned to hand.

    Currently no additional restrictions beyond card type.
    """
    return True
