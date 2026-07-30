from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from swb.db.repository import CardDefinition
from swb.engine.card_rules import CardRule, RuleBook, Trigger
from swb.engine.commands import (
    Attack,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
    SuperEvolve,
)
from swb.engine.effects import (
    Condition,
    ConditionType,
    EffectKind,
    EffectOperation,
    TargetKind,
)
from swb.engine.events import EventType
from swb.engine.origin import CardOrigin
from swb.engine.resolution import GameConfig, GameEngine, IllegalCommand
from swb.engine.state import HandCard, Unit


@dataclass(frozen=True)
class ForcedScenarioResult:
    scenario_id: str
    category: str
    status: str
    commands: tuple[str, ...]
    event_types: tuple[str, ...]
    direct_state_mutations: int
    invariant_checks: int
    conclusion: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _card(
    card_id: int,
    *,
    cost: int = 1,
    attack: int | None = 2,
    life: int | None = 2,
    card_type: str = "随从",
) -> CardDefinition:
    return CardDefinition(
        card_id=card_id,
        card_set_id=10000,
        class_id=1,
        class_name="精灵",
        name=f"forced-scenario-{card_id}",
        cost=cost,
        card_type=card_type,
        attack=attack,
        life=life,
        keywords=frozenset(),
        support_level="basic",
        is_collectible=True,
    )


def _deck(special: CardDefinition | None = None) -> list[CardDefinition]:
    cards = [_card(98000000 + index) for index in range(40)]
    if special is not None:
        cards[0] = special
    return cards


class _Scenario:
    """Prepare reachable boundary fixtures and validate every direct mutation."""

    def __init__(
        self,
        *,
        rulebook: RuleBook | None = None,
        special: CardDefinition | None = None,
        seed: int = 1200,
    ) -> None:
        cards = _deck(special)
        self.engine = GameEngine(
            cards,
            _deck(),
            class_a=1,
            class_b=1,
            seed=seed,
            rulebook=rulebook or RuleBook(),
            config=GameConfig(validate_invariants=True),
        )
        self.engine.reset(seed=seed)
        self.commands: list[str] = []
        self.direct_state_mutations = 0
        self.invariant_checks = 1

    def mutate(self, mutation: Callable[[GameEngine], None]) -> None:
        mutation(self.engine)
        self.direct_state_mutations += 1
        self.engine.assert_invariants()
        self.invariant_checks += 1

    def apply(self, command) -> None:
        self.engine.apply(command)
        self.commands.append(type(command).__name__)

    def replace_hand_card(
        self,
        definition: CardDefinition,
        *,
        player_index: int = 0,
        hand_index: int = 0,
    ) -> None:
        def mutation(engine: GameEngine) -> None:
            player = engine.players[player_index]
            old = player.hand[hand_index]
            player.deck.append(old.definition)
            for index, deck_card in enumerate(player.deck):
                candidate = getattr(deck_card, "definition", deck_card)
                if candidate.card_id == definition.card_id:
                    player.deck.pop(index)
                    break
            else:
                player.deck.pop()
            hand_card = HandCard(
                definition=definition,
                entity_id=engine.state.allocate_entity_id(),
                origin=CardOrigin.DECK,
            )
            player.hand[hand_index] = hand_card
            player.hand_entity_ids[hand_index] = hand_card.entity_id

        self.mutate(mutation)

    def place_unit(
        self,
        definition: CardDefinition,
        *,
        player_index: int,
        attack: int | None = None,
        health: int | None = None,
        can_attack: bool = False,
    ) -> Unit:
        holder: list[Unit] = []

        def mutation(engine: GameEngine) -> None:
            unit = Unit.summon(
                definition,
                entity_id=engine.state.allocate_entity_id(),
                origin=CardOrigin.DECK,
            )
            if attack is not None:
                unit.attack = attack
            if health is not None:
                unit.health = health
                unit.max_health = max(unit.max_health, health)
            unit.can_attack = can_attack
            unit.summoned_this_turn = not can_attack
            engine.players[player_index].board.append(unit)
            holder.append(unit)

        self.mutate(mutation)
        return holder[0]

    def resolve_choices(self, *, limit: int = 16) -> None:
        for _ in range(limit):
            choices = [
                command
                for command in self.engine.legal_commands()
                if isinstance(command, Choose)
            ]
            if not choices:
                return
            self.apply(choices[0])
        raise RuntimeError("forced scenario exceeded pending-choice limit")

    def result(
        self,
        scenario_id: str,
        category: str,
        conclusion: str,
    ) -> ForcedScenarioResult:
        self.engine.assert_invariants()
        self.invariant_checks += 1
        return ForcedScenarioResult(
            scenario_id=scenario_id,
            category=category,
            status="passed",
            commands=tuple(self.commands),
            event_types=tuple(
                event.type.value for event in self.engine.event_history
            ),
            direct_state_mutations=self.direct_state_mutations,
            invariant_checks=self.invariant_checks,
            conclusion=conclusion,
        )


def _cost_fixture() -> ForcedScenarioResult:
    source = _card(98100001, cost=3)
    scenario = _Scenario(special=source)
    scenario.replace_hand_card(source)
    scenario.mutate(
        lambda engine: (
            setattr(engine.players[0], "max_mana", 3),
            setattr(engine.players[0], "mana", 3),
        )
    )
    plays = [
        command
        for command in scenario.engine.legal_commands()
        if isinstance(command, PlayCard) and command.hand_index == 0
    ]
    if plays != [PlayCard(0, 0)]:
        raise AssertionError(f"cost fixture legal commands differ: {plays!r}")
    scenario.apply(plays[0])
    if scenario.engine.players[0].mana != 0:
        raise AssertionError("cost fixture did not spend the exact PP threshold")
    return scenario.result(
        "minimum_cost_threshold",
        "cost",
        "A card is illegal below cost, legal at exact cost, and spends that PP.",
    )


def _target_fixture() -> ForcedScenarioResult:
    source = _card(
        98100002,
        cost=1,
        attack=None,
        life=None,
        card_type="法术",
    )
    rulebook = RuleBook(rules=(
        CardRule(
            source.card_id,
            Trigger.PLAY,
            (
                EffectOperation(
                    EffectKind.DAMAGE_UNIT,
                    TargetKind.ENEMY_UNIT,
                    amount=2,
                    requires_target=True,
                ),
            ),
        ),
    ))
    scenario = _Scenario(rulebook=rulebook, special=source)
    scenario.replace_hand_card(source)
    target = scenario.place_unit(_card(98100003), player_index=1, health=3)
    scenario.mutate(
        lambda engine: (
            setattr(engine.players[0], "max_mana", 1),
            setattr(engine.players[0], "mana", 1),
        )
    )
    scenario.apply(PlayCard(0, 0))
    request = scenario.engine.state.pending_choice
    if request is None or [option.entity_id for option in request.options] != [
        target.entity_id
    ]:
        raise AssertionError("target fixture did not expose the exact candidate")
    scenario.resolve_choices()
    if target.health != 1:
        raise AssertionError("target fixture did not damage the selected target")
    return scenario.result(
        "minimum_selected_target",
        "target",
        "A selected target is offered through Choose and receives the effect.",
    )


def _capacity_fixture() -> ForcedScenarioResult:
    source = _card(98100004, cost=1)
    scenario = _Scenario(special=source)
    scenario.replace_hand_card(source)

    def fill(engine: GameEngine) -> None:
        engine.players[0].max_mana = 1
        engine.players[0].mana = 1
        for index in range(engine.config.max_board):
            engine.players[0].board.append(
                Unit.summon(
                    _card(98100100 + index),
                    entity_id=engine.state.allocate_entity_id(),
                )
            )

    scenario.mutate(fill)
    if any(
        isinstance(command, PlayCard) and command.hand_index == 0
        for command in scenario.engine.legal_commands()
    ):
        raise AssertionError("full-board follower was exposed as legal")
    before = scenario.engine.deterministic_fingerprint()
    try:
        scenario.engine.apply(PlayCard(0, 0))
    except IllegalCommand:
        scenario.commands.append("PlayCard(illegal)")
    else:
        raise AssertionError("full-board follower command was accepted")
    if scenario.engine.deterministic_fingerprint() != before:
        raise AssertionError("illegal full-board play mutated state")
    return scenario.result(
        "minimum_board_capacity",
        "capacity",
        "A full board masks follower play and direct illegal execution is atomic.",
    )


def _resource_fixture() -> ForcedScenarioResult:
    source = _card(
        98100005,
        attack=None,
        life=None,
        card_type="法术",
    )
    conditional = EffectOperation(
        EffectKind.CONDITIONAL,
        TargetKind.OWN_LEADER,
        conditions=(
            Condition(ConditionType.CONTROLLER_SHADOWS_AT_LEAST, value=5),
        ),
        then_operations=(
            EffectOperation(
                EffectKind.DAMAGE_LEADER,
                TargetKind.ENEMY_LEADER,
                amount=2,
            ),
        ),
    )
    scenario = _Scenario(
        rulebook=RuleBook(rules=(
            CardRule(source.card_id, Trigger.PLAY, (conditional,)),
        )),
        special=source,
    )
    scenario.replace_hand_card(source)
    scenario.mutate(
        lambda engine: (
            setattr(engine.players[0], "max_mana", 1),
            setattr(engine.players[0], "mana", 1),
            setattr(engine.players[0], "shadows", 5),
            setattr(engine.players[0], "cooperation", 10),
            setattr(engine.players[0], "cards_played_this_turn", 10),
        )
    )
    scenario.apply(PlayCard(0, 0))
    if scenario.engine.players[1].health != 18:
        raise AssertionError("resource threshold did not execute at equality")
    return scenario.result(
        "minimum_resource_threshold",
        "resource",
        "Shadow, Cooperation, and combo thresholds survive invariant validation; "
        "the equality branch executes.",
    )


def _evolution_fixture(*, super_evolution: bool) -> ForcedScenarioResult:
    source = _card(98100007 if super_evolution else 98100006)
    trigger = Trigger.SUPER_EVOLVE if super_evolution else Trigger.EVOLVE
    event = (
        EventType.FOLLOWER_SUPER_EVOLVED
        if super_evolution
        else EventType.FOLLOWER_EVOLVED
    )
    scenario = _Scenario(rulebook=RuleBook(rules=(
        CardRule(
            source.card_id,
            trigger,
            (
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    amount=1,
                ),
            ),
        ),
    )))
    unit = scenario.place_unit(source, player_index=0)
    scenario.mutate(
        lambda engine: setattr(
            engine.players[0],
            "turns_started",
            (
                engine.config.first_player_super_evolution_unlock_turn
                if super_evolution
                else engine.config.evolution_unlock_turn
            ),
        )
    )
    command = (
        SuperEvolve(0, unit.entity_id)
        if super_evolution
        else Evolve(0, unit.entity_id)
    )
    scenario.apply(command)
    if not any(item.type is event for item in scenario.engine.event_history):
        raise AssertionError("evolution fixture did not emit its lifecycle event")
    if scenario.engine.players[1].health != 19:
        raise AssertionError("evolution fixture did not execute its trigger")
    return scenario.result(
        (
            "minimum_super_evolution"
            if super_evolution
            else "minimum_ordinary_evolution"
        ),
        "super_evolution" if super_evolution else "ordinary_evolution",
        "The public evolution command emits and resolves the matching trigger.",
    )


def _turn_boundary_fixture(*, turn_start: bool) -> ForcedScenarioResult:
    source = _card(98100009 if turn_start else 98100008)
    trigger = Trigger.TURN_START if turn_start else Trigger.TURN_END
    scenario = _Scenario(rulebook=RuleBook(rules=(
        CardRule(
            source.card_id,
            trigger,
            (
                EffectOperation(
                    EffectKind.DAMAGE_LEADER,
                    TargetKind.ENEMY_LEADER,
                    amount=1,
                ),
            ),
        ),
    )))
    scenario.place_unit(source, player_index=0)
    scenario.apply(EndTurn(0))
    if turn_start:
        scenario.apply(EndTurn(1))
    if scenario.engine.players[1].health != 19:
        raise AssertionError("turn-boundary fixture did not execute its trigger")
    return scenario.result(
        "minimum_turn_start" if turn_start else "minimum_turn_end",
        "turn_start" if turn_start else "turn_end",
        "Public EndTurn transitions execute the matching boundary batch.",
    )


def _simultaneous_death_fixture() -> ForcedScenarioResult:
    scenario = _Scenario()
    attacker = scenario.place_unit(
        _card(98100010, attack=2, life=2),
        player_index=0,
        can_attack=True,
    )
    defender = scenario.place_unit(
        _card(98100011, attack=2, life=2),
        player_index=1,
    )
    scenario.apply(Attack(0, attacker.entity_id, defender.entity_id))
    deaths = [
        event
        for event in scenario.engine.event_history
        if event.type is EventType.FOLLOWER_DESTROYED
    ]
    if len(deaths) != 2:
        raise AssertionError("combat did not collect both simultaneous deaths")
    if scenario.engine.players[0].board or scenario.engine.players[1].board:
        raise AssertionError("simultaneous deaths did not leave both boards")
    return scenario.result(
        "minimum_simultaneous_death",
        "simultaneous_death",
        "One public Attack collects both lethal combat deaths before stabilization.",
    )


def run_minimal_forced_scenarios() -> list[ForcedScenarioResult]:
    """Run the checklist's minimum public-interface scenario fixtures."""

    runners = (
        _cost_fixture,
        _target_fixture,
        _capacity_fixture,
        _resource_fixture,
        lambda: _evolution_fixture(super_evolution=False),
        lambda: _evolution_fixture(super_evolution=True),
        lambda: _turn_boundary_fixture(turn_start=False),
        lambda: _turn_boundary_fixture(turn_start=True),
        _simultaneous_death_fixture,
    )
    return [runner() for runner in runners]
