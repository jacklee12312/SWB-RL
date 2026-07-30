"""Test-fixture helpers for mandatory Enhance mode selection."""

from __future__ import annotations


def prepare_mana_for_play_mode(engine, hand_card, mode_id: str) -> None:
    """Place a real-card fixture at PP where the requested mode is selectable.

    Enhance is mandatory once its threshold is reached. Normal-mode behavior
    fixtures therefore use the largest PP below the first Enhance threshold,
    while an explicitly requested Enhance preserves the fixture's PP whenever
    it is already the highest affordable mode. Multi-Enhance fixtures are
    lowered only far enough to keep a higher mode from replacing the requested
    one.
    """

    modes = engine.rulebook.modes_for(hand_card.card_id)
    enhance_modes = tuple(mode for mode in modes if mode.is_enhance)
    if not enhance_modes:
        return

    player = engine.players[engine.current_player]
    if mode_id == "normal":
        first_enhance_cost = min(mode.cost for mode in enhance_modes)
        mana = first_enhance_cost - 1
        normal_cost = engine.effective_play_cost(hand_card, None)
        if normal_cost > mana:
            raise AssertionError(
                f"card {hand_card.card_id} has no reachable normal PP: "
                f"current cost {normal_cost}, first Enhance "
                f"{first_enhance_cost}"
            )
        player.mana = mana
        return

    selected = next(
        (mode for mode in enhance_modes if mode.mode_id == mode_id),
        None,
    )
    if selected is not None:
        if player.mana < selected.cost:
            player.mana = selected.cost
        higher_costs = sorted(
            mode.cost
            for mode in enhance_modes
            if mode.cost > selected.cost
        )
        if higher_costs and player.mana >= higher_costs[0]:
            player.mana = higher_costs[0] - 1
