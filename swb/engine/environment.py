from __future__ import annotations

import copy
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from swb.db.repository import CardDefinition
from swb.engine.abilities import AbilityKeyword
from swb.engine.commands import ActivateAmulet, Attack, BeginFusion, ChoiceKind, Choose, EndTurn, Evolve, GameCommand, PlayCard, SuperEvolve, UseExtraPP
from swb.engine.conditions import OVERFLOW_MAX_MANA_THRESHOLD
from swb.engine.card_rules import RuleBook
from swb.engine.play_modes import MAX_SPECIAL_MODES_PER_CARD
from swb.engine.resolution import GameConfig, GameEngine, GameEngineSnapshot
from swb.engine.state import Amulet, BoardCard, HandCard, PlayerState, Unit

MATCH_SETUP_LEGACY = "legacy"
MATCH_SETUP_OFFICIAL = "official"
MATCH_SETUP_VALUES = frozenset({MATCH_SETUP_LEGACY, MATCH_SETUP_OFFICIAL})
_MATCH_SETUP_DEFAULT = object()


@dataclass(frozen=True)
class StepResult:
    observation: object
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


@dataclass(frozen=True)
class EnvironmentSnapshot:
    core: GameEngineSnapshot
    graveyard_page: int
    last_choice_request_key: tuple[str, int] | None
    truncated: bool
    agent_steps: int
    transition_version: int


class ShadowverseEnv:
    """RL adapter around the deterministic command-based game engine."""

    OBSERVATION_V1_SIZE = 304
    MAX_HAND = 9
    MAX_BOARD = 5
    MAX_MANA = 10
    MAX_TURNS = 200
    STARTING_HAND = 4
    STARTING_EVOLUTION_POINTS = 2
    EVOLUTION_UNLOCK_TURN = 5
    SECOND_PLAYER_EVOLUTION_UNLOCK_TURN = 4
    STARTING_SUPER_EVOLUTION_POINTS = 2
    FIRST_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN = 7
    SECOND_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN = 6
    CLASS_COUNT = 7

    END_TURN = 0
    PLAY_OFFSET = 1
    ATTACK_OFFSET = PLAY_OFFSET + MAX_HAND
    TARGETS_PER_ATTACKER = 1 + MAX_BOARD
    EVOLVE_OFFSET = ATTACK_OFFSET + MAX_BOARD * TARGETS_PER_ATTACKER
    CHOICE_OFFSET = EVOLVE_OFFSET + MAX_BOARD
    MAX_CHOICE_OPTIONS = 16

    GRAVEYARD_PAGE_SIZE = 16
    GRAVEYARD_CHOICE_OFFSET = CHOICE_OFFSET + MAX_CHOICE_OPTIONS
    GRAVEYARD_PREV_PAGE = GRAVEYARD_CHOICE_OFFSET
    GRAVEYARD_NEXT_PAGE = GRAVEYARD_CHOICE_OFFSET + 1
    GRAVEYARD_SLOT_OFFSET = GRAVEYARD_CHOICE_OFFSET + 2

    MODE_PLAY_OFFSET = GRAVEYARD_SLOT_OFFSET + GRAVEYARD_PAGE_SIZE

    SUPER_EVOLVE_OFFSET = MODE_PLAY_OFFSET + MAX_HAND * MAX_SPECIAL_MODES_PER_CARD

    USE_EXTRA_PP = SUPER_EVOLVE_OFFSET + MAX_BOARD
    ACTION_SIZE = USE_EXTRA_PP + 1

    DEFAULT_RULE_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "rules"

    def __init__(
        self,
        deck_a: Sequence[CardDefinition],
        deck_b: Sequence[CardDefinition],
        *,
        class_a: int,
        class_b: int,
        seed: int | None = None,
        rulebook: RuleBook | None = None,
        card_resolver: Callable[[int], CardDefinition | None] | None = None,
        debug_info: bool = False,
        validate_invariants: bool = False,
        observation_version: str = "v1",
        card_vocabulary: Sequence[int] | None = None,
        open_decklists: bool = False,
        max_game_turns: int | None = MAX_TURNS,
        max_agent_steps: int | None = 2000,
        debug_cache_validation: bool = False,
        training_mode: bool = False,
        training_event_history_limit: int = 256,
        audit_runtime_coverage: bool = False,
        audit_context: dict[str, object] | None = None,
        match_setup: str = MATCH_SETUP_LEGACY,
        starting_player: int | None | object = _MATCH_SETUP_DEFAULT,
        enable_mulligan: bool | None = None,
    ):
        if observation_version not in {"v1", "v2", "v3", "v4", "v4.1"}:
            raise ValueError(
                "observation_version must be 'v1', 'v2', 'v3', 'v4', "
                "or 'v4.1', "
                f"got {observation_version!r}"
            )
        for name, value in (
            ("max_game_turns", max_game_turns),
            ("max_agent_steps", max_agent_steps),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if (
            not isinstance(training_event_history_limit, int)
            or isinstance(training_event_history_limit, bool)
            or training_event_history_limit <= 0
        ):
            raise ValueError(
                "training_event_history_limit must be a positive integer"
            )
        if match_setup not in MATCH_SETUP_VALUES:
            raise ValueError(
                "match_setup must be 'legacy' or 'official', "
                f"got {match_setup!r}"
            )
        if starting_player is _MATCH_SETUP_DEFAULT:
            resolved_starting_player = (
                None if match_setup == MATCH_SETUP_OFFICIAL else 0
            )
        else:
            resolved_starting_player = starting_player
        if resolved_starting_player not in (None, 0, 1):
            raise ValueError("starting_player must be 0, 1, or None")
        resolved_enable_mulligan = (
            match_setup == MATCH_SETUP_OFFICIAL
            if enable_mulligan is None
            else bool(enable_mulligan)
        )
        self.observation_version = observation_version
        self.open_decklists = bool(open_decklists)
        self.max_game_turns = max_game_turns
        self.max_agent_steps = max_agent_steps
        self.debug_cache_validation = bool(debug_cache_validation)
        self.training_mode = bool(training_mode)
        self.training_event_history_limit = training_event_history_limit
        self.audit_runtime_coverage = bool(audit_runtime_coverage)
        self.audit_context = dict(audit_context or {})
        self.match_setup = match_setup
        self.starting_player = resolved_starting_player
        self.enable_mulligan = resolved_enable_mulligan
        vocabulary = (
            []
            if observation_version == "v1" and card_vocabulary is None
            else (
                sorted({card.card_id for card in (*deck_a, *deck_b)})
                if card_vocabulary is None
                else list(card_vocabulary)
            )
        )
        if (
            any(
                not isinstance(card_id, int)
                or isinstance(card_id, bool)
                or card_id <= 0
                for card_id in vocabulary
            )
            or len(vocabulary) != len(set(vocabulary))
        ):
            raise ValueError(
                "card_vocabulary must contain unique positive integer card IDs"
            )
        self.card_vocabulary = tuple(vocabulary)
        self._v2_card_index = {
            card_id: index + 1 for index, card_id in enumerate(self.card_vocabulary)
        }
        resolved_rulebook = rulebook or RuleBook.from_directory(
            self.DEFAULT_RULE_DIRECTORY
        )
        self.debug_info = debug_info
        self._core = GameEngine(
            deck_a,
            deck_b,
            class_a=class_a,
            class_b=class_b,
            seed=seed,
            rulebook=resolved_rulebook,
            card_resolver=card_resolver,
            config=GameConfig(
                max_hand=self.MAX_HAND,
                max_board=self.MAX_BOARD,
                max_mana=self.MAX_MANA,
                # Time limits belong to the RL adapter and must truncate rather
                # than manufacture a rules-level winner from current health.
                max_turns=None,
                starting_hand=self.STARTING_HAND,
                starting_evolution_points=self.STARTING_EVOLUTION_POINTS,
                evolution_unlock_turn=self.EVOLUTION_UNLOCK_TURN,
                second_player_evolution_unlock_turn=(
                    self.SECOND_PLAYER_EVOLUTION_UNLOCK_TURN
                ),
                starting_super_evolution_points=self.STARTING_SUPER_EVOLUTION_POINTS,
                first_player_super_evolution_unlock_turn=(
                    self.FIRST_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN
                ),
                second_player_super_evolution_unlock_turn=(
                    self.SECOND_PLAYER_SUPER_EVOLUTION_UNLOCK_TURN
                ),
                validate_invariants=validate_invariants,
                retain_text_logs=not self.training_mode,
                event_history_limit=(
                    self.training_event_history_limit
                    if self.training_mode
                    else None
                ),
                audit_runtime_coverage=self.audit_runtime_coverage,
                starting_player=self.starting_player,
                enable_mulligan=self.enable_mulligan,
            ),
        )
        if self._core.runtime_coverage is not None:
            self._core.runtime_coverage.set_context(**self.audit_context)
        self._graveyard_page: int = 0
        self._last_choice_request_key: tuple[str, int] | None = None
        self._truncated = False
        self._agent_steps = 0
        self._v2_faith_index = {
            faith_id: index + 1
            for index, faith_id in enumerate(
                sorted({
                    definition.faith_id
                    for definition in resolved_rulebook._faith_defs.values()
                })
            )
        }
        self._v2_emblem_index = {
            emblem_id: index + 1
            for index, emblem_id in enumerate(sorted(resolved_rulebook._emblem_defs))
        }
        self._transition_version = 0
        self._legal_commands_cache: tuple[GameCommand, ...] | None = None
        self._legal_action_map_cache: dict[int, GameCommand] | None = None
        self._action_mask_cache: tuple[bool, ...] | None = None
        self._observation_cache: dict[tuple[object, ...], object] = {}
        self._zone_histogram_cache: dict[tuple[object, ...], tuple[int, ...]] = {}
        self._debug_cache_fingerprint: dict[str, object] | None = None
        self._cache_stats = Counter()
        self._initial_deck_histograms = tuple(
            self._build_card_histogram(deck) for deck in self._core.deck_lists
        )

    @property
    def core(self) -> GameEngine:
        """Expose mutable core state through an explicit cache boundary.

        Holding this object and mutating it later still requires callers to use
        :meth:`invalidate_cache`; ``debug_cache_validation=True`` detects a
        missed boundary before a cached result can be returned.
        """
        self.invalidate_cache(reason="mutable core access")
        return self._core

    @core.setter
    def core(self, value: GameEngine) -> None:
        """Replace the compatibility core and rebuild deck-derived caches."""
        if not isinstance(value, GameEngine):
            raise TypeError("core must be a GameEngine")
        self._core = value
        self._graveyard_page = 0
        self._last_choice_request_key = None
        self._initial_deck_histograms = tuple(
            self._build_card_histogram(deck) for deck in self._core.deck_lists
        )
        self.invalidate_cache(reason="mutable core replacement")

    @property
    def deck_lists(self) -> tuple[list[CardDefinition], list[CardDefinition]]:
        return self._core.deck_lists

    @property
    def players(self) -> list[PlayerState]:
        self.invalidate_cache(reason="mutable player access")
        return self._core.players

    @property
    def state_version(self) -> int:
        return self._core.state_version

    @property
    def transition_version(self) -> int:
        return self._transition_version

    @property
    def cache_stats(self) -> dict[str, int]:
        return dict(self._cache_stats)

    def invalidate_cache(self, *, reason: str = "external state mutation") -> None:
        """Invalidate all derived RL state after direct/debug mutation."""
        self._invalidate_caches(advance_transition=True, reason=reason)

    def snapshot(self) -> EnvironmentSnapshot:
        return EnvironmentSnapshot(
            core=self._core.snapshot(),
            graveyard_page=self._graveyard_page,
            last_choice_request_key=self._last_choice_request_key,
            truncated=self._truncated,
            agent_steps=self._agent_steps,
            transition_version=self._transition_version,
        )

    def restore(self, snapshot: EnvironmentSnapshot) -> None:
        previous_version = self._transition_version
        self._core.restore(snapshot.core)
        self._graveyard_page = snapshot.graveyard_page
        self._last_choice_request_key = snapshot.last_choice_request_key
        self._truncated = snapshot.truncated
        self._agent_steps = snapshot.agent_steps
        self._transition_version = max(
            previous_version,
            snapshot.transition_version,
        ) + 1
        self._invalidate_caches(
            advance_transition=False,
            reason="snapshot restore",
        )

    def clone(self) -> ShadowverseEnv:
        clone = ShadowverseEnv(
            self._core.deck_lists[0],
            self._core.deck_lists[1],
            class_a=self._core.player_classes[0],
            class_b=self._core.player_classes[1],
            seed=0,
            rulebook=self._core.rulebook,
            card_resolver=self._core.card_resolver,
            debug_info=self.debug_info,
            validate_invariants=self._core.config.validate_invariants,
            observation_version=self.observation_version,
            card_vocabulary=self.card_vocabulary,
            open_decklists=self.open_decklists,
            max_game_turns=self.max_game_turns,
            max_agent_steps=self.max_agent_steps,
            debug_cache_validation=self.debug_cache_validation,
            training_mode=self.training_mode,
            training_event_history_limit=self.training_event_history_limit,
            audit_runtime_coverage=self.audit_runtime_coverage,
            audit_context=self.audit_context,
            match_setup=self.match_setup,
            starting_player=self.starting_player,
            enable_mulligan=self.enable_mulligan,
        )
        clone.restore(self.snapshot())
        return clone

    def _invalidate_caches(
        self,
        *,
        advance_transition: bool,
        reason: str,
    ) -> None:
        if advance_transition:
            self._transition_version += 1
        self._legal_commands_cache = None
        self._legal_action_map_cache = None
        self._action_mask_cache = None
        self._observation_cache.clear()
        self._zone_histogram_cache.clear()
        self._debug_cache_fingerprint = None
        self._cache_stats["invalidations"] += 1
        self._cache_stats[f"invalidation:{reason}"] += 1

    def _record_debug_cache_fingerprint(self) -> None:
        if self.debug_cache_validation:
            self._debug_cache_fingerprint = self._core.deterministic_fingerprint()

    def _assert_cache_coherent(self) -> None:
        if not self.debug_cache_validation or self._debug_cache_fingerprint is None:
            return
        if self._debug_cache_fingerprint != self._core.deterministic_fingerprint():
            self._invalidate_caches(
                advance_transition=True,
                reason="debug mutation detected",
            )
            raise RuntimeError(
                "Engine state changed outside an official transition; "
                "call env.invalidate_cache() after direct debug/test mutation"
            )

    def _build_card_histogram(self, definitions) -> tuple[int, ...]:
        counts = Counter(
            self._v2_card_index.get(definition.card_id, 0)
            for definition in definitions
        )
        counts.pop(0, None)
        return tuple(
            counts.get(index, 0)
            for index in range(1, len(self.card_vocabulary) + 1)
        )

    def cached_card_histogram(
        self,
        zone_key: tuple[object, ...],
        definitions,
    ) -> tuple[int, ...]:
        """Return one zone histogram per environment transition."""
        key = (self._transition_version, self._core.state_version, *zone_key)
        cached = self._zone_histogram_cache.get(key)
        if cached is not None:
            self._assert_cache_coherent()
            self._cache_stats["histogram_hits"] += 1
            return cached
        result = self._build_card_histogram(definitions)
        self._zone_histogram_cache[key] = result
        self._cache_stats["histogram_misses"] += 1
        self._record_debug_cache_fingerprint()
        return result

    @property
    def current_player(self) -> int:
        return self._core.current_player

    @property
    def decision_player(self) -> int:
        pending = self._core.state.pending_choice
        if pending is not None:
            return pending.player_index
        return self._core.current_player

    @property
    def turn(self) -> int:
        return self._core.turn

    @property
    def terminated(self) -> bool:
        return self._core.terminated

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def agent_steps(self) -> int:
        return self._agent_steps

    @property
    def winner(self) -> int | None:
        return self._core.winner

    @property
    def logs(self) -> list[str]:
        return self._core.logs

    @property
    def placeholder_ability_events(self) -> list[object]:
        return self._core.placeholder_ability_events

    @property
    def runtime_coverage(self):
        return self._core.runtime_coverage

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[object, dict[str, object]]:
        self._core.reset(seed=seed)
        self._graveyard_page = 0
        self._last_choice_request_key = None
        self._truncated = False
        self._agent_steps = 0
        self._invalidate_caches(advance_transition=True, reason="reset")
        mask = self.action_mask()
        return self.observation(action_mask=mask), self.info(action_mask=mask)

    def step(
        self,
        action: int,
        *,
        timing: dict[str, float] | None = None,
    ) -> StepResult:
        step_started = time.perf_counter() if timing is not None else 0.0
        if self.terminated or self.truncated:
            raise ValueError("Cannot step a finished environment; call reset()")
        page_before = self._graveyard_page
        choice_key_before = self._last_choice_request_key
        if action < 0 or action >= self.ACTION_SIZE:
            self._core._record_runtime_diagnostic(
                "illegal_action",
                detail=str(action),
            )
            raise ValueError(f"Illegal action: {action}")
        action_mask_started = (
            time.perf_counter() if timing is not None else 0.0
        )
        mask = self.action_mask()
        if timing is not None:
            timing["action_mask_seconds"] = (
                time.perf_counter() - action_mask_started
            )
        if not mask[action]:
            self._graveyard_page = page_before
            self._last_choice_request_key = choice_key_before
            self._core._record_runtime_diagnostic(
                "illegal_action",
                detail=str(action),
            )
            raise ValueError(f"Illegal action: {action}")
        acting_player = self.decision_player
        command_started = time.perf_counter() if timing is not None else 0.0
        try:
            command = self._decode_action(action)
        except _PageTurn:
            if timing is not None:
                timing["command_decode_seconds"] = (
                    time.perf_counter() - command_started
                )
                timing["resolution_seconds"] = 0.0
            post_started = (
                time.perf_counter() if timing is not None else 0.0
            )
            self._agent_steps += 1
            self._update_truncation()
            self._invalidate_caches(
                advance_transition=True,
                reason="graveyard page transition",
            )
            action_mask_started = (
                time.perf_counter() if timing is not None else 0.0
            )
            next_mask = self.action_mask()
            if timing is not None:
                timing["action_mask_seconds"] += (
                    time.perf_counter() - action_mask_started
                )
            observation_started = (
                time.perf_counter() if timing is not None else 0.0
            )
            next_observation = self.observation(action_mask=next_mask)
            if timing is not None:
                timing["observation_seconds"] = (
                    time.perf_counter() - observation_started
                )
            result = StepResult(
                observation=next_observation,
                reward=0.0,
                terminated=False,
                truncated=self.truncated,
                info=self.info(action_mask=next_mask),
            )
            if timing is not None:
                timing["post_step_seconds"] = (
                    time.perf_counter() - post_started
                )
                timing["step_total_seconds"] = (
                    time.perf_counter() - step_started
                )
            return result
        except ValueError:
            self._graveyard_page = page_before
            self._last_choice_request_key = choice_key_before
            self._core._record_runtime_diagnostic(
                "illegal_action",
                detail=str(action),
            )
            raise
        if timing is not None:
            timing["command_decode_seconds"] = (
                time.perf_counter() - command_started
            )
        resolution_started = (
            time.perf_counter() if timing is not None else 0.0
        )
        result = self._core.apply(command)
        if timing is not None:
            timing["resolution_seconds"] = (
                time.perf_counter() - resolution_started
            )
        post_started = time.perf_counter() if timing is not None else 0.0
        self._agent_steps += 1
        self._update_truncation()
        reward = 0.0 if result.winner is None else (
            1.0 if result.winner == acting_player else -1.0
        )
        self._sync_choice_page(invalidate=False)
        self._invalidate_caches(advance_transition=True, reason="legal step")
        action_mask_started = (
            time.perf_counter() if timing is not None else 0.0
        )
        next_mask = self.action_mask()
        if timing is not None:
            timing["action_mask_seconds"] += (
                time.perf_counter() - action_mask_started
            )
        observation_started = (
            time.perf_counter() if timing is not None else 0.0
        )
        next_observation = self.observation(action_mask=next_mask)
        if timing is not None:
            timing["observation_seconds"] = (
                time.perf_counter() - observation_started
            )
        step_result = StepResult(
            observation=next_observation,
            reward=reward,
            terminated=self._core.terminated,
            truncated=self.truncated,
            info=self.info(action_mask=next_mask),
        )
        if timing is not None:
            timing["post_step_seconds"] = (
                time.perf_counter() - post_started
            )
            timing["step_total_seconds"] = (
                time.perf_counter() - step_started
            )
        return step_result

    def _update_truncation(self) -> None:
        if self._core.terminated:
            self._truncated = False
            return
        turn_limit = (
            self.max_game_turns is not None and self.turn > self.max_game_turns
        )
        step_limit = (
            self.max_agent_steps is not None
            and self._agent_steps >= self.max_agent_steps
        )
        self._truncated = turn_limit or step_limit

    def observation(
        self,
        *,
        perspective: int | None = None,
        action_mask: Sequence[bool] | None = None,
    ) -> object:
        if perspective is not None and perspective not in (0, 1):
            raise ValueError("perspective must be 0 or 1")
        self._sync_choice_page()
        resolved_perspective = (
            self.decision_player if perspective is None else perspective
        )
        mask_key = None if action_mask is None else tuple(bool(x) for x in action_mask)
        cache_key = (
            self._transition_version,
            self._core.state_version,
            self.observation_version,
            resolved_perspective,
            mask_key,
            self.open_decklists,
        )
        # V1/v2 are compatibility formats and historically reflect direct
        # mutation through retained entity references. Keep that behavior;
        # formal NumPy-only v3 training observations use the versioned cache.
        cacheable = self.observation_version in {"v3", "v4", "v4.1"}
        cached = self._observation_cache.get(cache_key) if cacheable else None
        if cached is not None:
            self._assert_cache_coherent()
            self._cache_stats["observation_hits"] += 1
            return copy.deepcopy(cached)
        if self.observation_version == "v2":
            from swb.engine.observation_v2 import encode_observation_v2

            result = encode_observation_v2(
                self,
                perspective=resolved_perspective,
                action_mask=action_mask,
            )
        elif self.observation_version == "v3":
            from swb.engine.observation_v3 import encode_observation_v3

            result = encode_observation_v3(
                self,
                perspective=resolved_perspective,
                action_mask=action_mask,
                open_decklists=self.open_decklists,
            )
        elif self.observation_version == "v4":
            from swb.engine.observation_v4 import encode_observation_v4

            result = encode_observation_v4(
                self,
                perspective=resolved_perspective,
                action_mask=action_mask,
                open_decklists=self.open_decklists,
            )
        elif self.observation_version == "v4.1":
            from swb.engine.observation_v4_1 import encode_observation_v4_1

            result = encode_observation_v4_1(
                self,
                perspective=resolved_perspective,
                action_mask=action_mask,
                open_decklists=self.open_decklists,
            )
        else:
            result = self._observation_v1(perspective=resolved_perspective)
        if cacheable:
            self._observation_cache[cache_key] = copy.deepcopy(result)
        self._cache_stats["observation_misses"] += 1
        self._record_debug_cache_fingerprint()
        return result

    def observation_v2_spec(self) -> dict[str, object]:
        if self.observation_version != "v2":
            raise ValueError("observation_v2_spec requires observation_version='v2'")
        from swb.engine.observation_v2 import observation_v2_spec

        return observation_v2_spec(self)

    def recurrent_observation(self) -> dict[str, object]:
        """Return v2 public history for a caller-owned recurrent/belief state."""
        if self.observation_version != "v2":
            raise ValueError("recurrent_observation requires observation_version='v2'")
        from swb.engine.observation_v2 import encode_observation_v2

        return encode_observation_v2(self)

    def observation_v3_space(self):
        if self.observation_version != "v3":
            raise ValueError("observation_v3_space requires observation_version='v3'")
        from swb.engine.observation_v3 import observation_v3_space

        return observation_v3_space(self)

    def observation_v4_space(self):
        if self.observation_version != "v4":
            raise ValueError("observation_v4_space requires observation_version='v4'")
        from swb.engine.observation_v4 import observation_v4_space

        return observation_v4_space(self)

    def observation_v4_1_space(self):
        if self.observation_version != "v4.1":
            raise ValueError(
                "observation_v4_1_space requires "
                "observation_version='v4.1'"
            )
        from swb.engine.observation_v4_1 import observation_v4_1_space

        return observation_v4_1_space(self)

    def _observation_v1(self, perspective: int | None = None) -> list[float]:
        self._sync_choice_page()
        perspective = self.decision_player if perspective is None else perspective
        me = self._core.players[perspective]
        opponent = self._core.players[1 - perspective]
        values = [
            me.health / 20, opponent.health / 20,
            me.mana / self.MAX_MANA, me.max_mana / self.MAX_MANA,
            len(me.deck) / max(1, len(self.deck_lists[perspective])),
            len(opponent.deck) / max(1, len(self.deck_lists[1 - perspective])),
            len(me.hand) / self.MAX_HAND, len(opponent.hand) / self.MAX_HAND,
            self.turn / self.MAX_TURNS,
            me.evolution_points / self.STARTING_EVOLUTION_POINTS,
            opponent.evolution_points / self.STARTING_EVOLUTION_POINTS,
            me.turns_started / self.MAX_TURNS,
            opponent.turns_started / self.MAX_TURNS,
            float(me.evolved_this_turn),
            float(opponent.evolved_this_turn),
            float(self._core.state.pending_choice is not None),
            me.shadows / 20, opponent.shadows / 20,
            me.cooperation / 10, opponent.cooperation / 10,
            me.cards_played_this_turn / 10,
            opponent.cards_played_this_turn / 10,
            me.follower_attacks_this_turn / 10,
            opponent.follower_attacks_this_turn / 10,
        ]
        total_pages = self._graveyard_total_pages()
        values.extend([
            (
                self._graveyard_page / max(1, total_pages - 1)
                if total_pages > 0
                else 0.0
            ),
            float(total_pages),
        ])
        me_countdowns = [
            emblem.countdown
            for emblem in me.emblems
            if emblem.countdown is not None
        ]
        opponent_countdowns = [
            emblem.countdown
            for emblem in opponent.emblems
            if emblem.countdown is not None
        ]
        values.extend([
            len(me.emblems) / 10,
            len(opponent.emblems) / 10,
            float(bool(me_countdowns)),
            float(bool(opponent_countdowns)),
            min(me_countdowns, default=0) / 10,
            min(opponent_countdowns, default=0) / 10,
        ])
        values.extend(float(me.class_id == cid) for cid in range(1, self.CLASS_COUNT + 1))
        values.extend(float(opponent.class_id == cid) for cid in range(1, self.CLASS_COUNT + 1))
        values.extend([
            me.super_evolution_points / self.STARTING_SUPER_EVOLUTION_POINTS,
            opponent.super_evolution_points / self.STARTING_SUPER_EVOLUTION_POINTS,
            float(me.super_evolved_this_turn),
            float(opponent.super_evolved_this_turn),
        ])
        values.extend([
            float(me.max_mana >= OVERFLOW_MAX_MANA_THRESHOLD),
            float(opponent.max_mana >= OVERFLOW_MAX_MANA_THRESHOLD),
        ])
        for index in range(self.MAX_HAND):
            card = me.hand[index] if index < len(me.hand) else None
            values.extend(self._card_features(card, perspective=perspective))
        for board in (me.board, opponent.board):
            for index in range(self.MAX_BOARD):
                unit = board[index] if index < len(board) else None
                values.extend(self._board_features(unit))
        values.extend([
            me.followers_evolved_this_match / 10,
            opponent.followers_evolved_this_match / 10,
        ])
        pending = self._core.state.pending_choice
        pending_target_count = 0 if pending is None else pending.target_count
        pending_selected_count = (
            0 if pending is None else len(pending.selected_options)
        )
        values.extend([
            min(pending_target_count, self.MAX_CHOICE_OPTIONS)
            / self.MAX_CHOICE_OPTIONS,
            (
                pending_selected_count / pending_target_count
                if pending_target_count > 0
                else 0.0
            ),
        ])
        values.extend([
            me.earth_sigils / 20,
            opponent.earth_sigils / 20,
        ])
        values.extend([
            len(me.faiths) / 5,
            len(opponent.faiths) / 5,
            sum(faith.value for faith in me.faiths) / 50,
            sum(faith.value for faith in opponent.faiths) / 50,
        ])
        values.extend([
            min(self._artifact_entry_kind_count(perspective), 40) / 40,
            min(self._artifact_entry_kind_count(1 - perspective), 40) / 40,
        ])
        values.extend([
            float(perspective == self._core.state.first_player),
            float(self._core.state.phase.value == "mulligan"),
            float(self._core.state.mulligan_completed[perspective]),
            float(self._core.state.mulligan_completed[1 - perspective]),
            float(me.extra_pp_available),
            float(opponent.extra_pp_available),
            me.extra_pp_uses / 2,
            opponent.extra_pp_uses / 2,
            float(me.extra_pp_active_turn == self.turn),
            float(opponent.extra_pp_active_turn == self.turn),
        ])
        return values

    def _artifact_entry_kind_count(self, player_index: int) -> int:
        return len({
            record.definition.name
            for record in self._core.state.follower_entries
            if record.owner == player_index
            and record.definition.card_type == "随从"
            and record.definition.tribe_name == "创造物"
        })

    def info(
        self,
        *,
        debug: bool | None = None,
        action_mask: Sequence[bool] | None = None,
    ) -> dict[str, object]:
        self._sync_choice_page()
        total_pages = self._graveyard_total_pages()
        include_debug = self.debug_info if debug is None else debug
        info: dict[str, object] = {
            "current_player": self.current_player,
            "decision_player": self.decision_player,
            "turn": self.turn,
            "winner": self.winner,
            "first_player": self._core.state.first_player,
            "phase": self._core.state.phase.value,
            "mulligan_completed": tuple(
                self._core.state.mulligan_completed
            ),
            "player_classes": self._core.player_classes,
            "action_mask": list(
                self.action_mask() if action_mask is None else action_mask
            ),
            "super_evolution_points": (
                self._core.players[0].super_evolution_points,
                self._core.players[1].super_evolution_points,
            ),
            "extra_pp": tuple(
                {
                    "available": player.extra_pp_available,
                    "uses": player.extra_pp_uses,
                    "active": player.extra_pp_active_turn == self.turn,
                }
                for player in self._core.players
            ),
            "placeholder_ability_count": len(self.placeholder_ability_events),
            "graveyard_page": self._graveyard_page,
            "graveyard_total_pages": total_pages,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "agent_steps": self.agent_steps,
        }
        if include_debug:
            info.update({
                "log": tuple(self.logs),
                "events": tuple(self._core.event_history),
                "placeholder_ability_events": tuple(self.placeholder_ability_events),
            })
        return info

    def _decode_action(self, action: int) -> GameCommand:
        if action == self.END_TURN:
            return EndTurn(self.current_player)
        if action == self.USE_EXTRA_PP:
            return UseExtraPP(self.current_player)
        if action == self.GRAVEYARD_PREV_PAGE:
            self._graveyard_page = max(0, self._graveyard_page - 1)
            raise _PageTurn()
        if action == self.GRAVEYARD_NEXT_PAGE:
            self._graveyard_page += 1
            raise _PageTurn()
        if action >= self.GRAVEYARD_SLOT_OFFSET and action < self.MODE_PLAY_OFFSET:
            request = self._core.state.pending_choice
            if request is None or request.choice_kind is not ChoiceKind.GRAVEYARD:
                raise ValueError("No graveyard choice is pending")
            slot_idx = action - self.GRAVEYARD_SLOT_OFFSET
            global_idx = self._graveyard_page * self.GRAVEYARD_PAGE_SIZE + slot_idx
            if global_idx >= len(request.options):
                raise ValueError(f"Graveyard choice index {global_idx} out of range")
            return Choose(request.player_index, request.options[global_idx].option_id)
        if action >= self.CHOICE_OFFSET and action < self.GRAVEYARD_CHOICE_OFFSET:
            request = self._core.state.pending_choice
            if request is None:
                raise ValueError("No choice is pending")
            option_index = action - self.CHOICE_OFFSET
            if option_index >= len(request.options):
                raise ValueError(f"Choice index {option_index} out of range")
            return Choose(request.player_index, request.options[option_index].option_id)
        if action >= self.MODE_PLAY_OFFSET:
            if action >= self.SUPER_EVOLVE_OFFSET:
                board_index = action - self.SUPER_EVOLVE_OFFSET
                return SuperEvolve(
                    self.current_player,
                    self._core.players[self.current_player].board[board_index].entity_id,
                )
            return self._decode_mode_play(action)
        if action < self.ATTACK_OFFSET:
            return PlayCard(self.current_player, action - self.PLAY_OFFSET)
        if action >= self.EVOLVE_OFFSET:
            board_index = action - self.EVOLVE_OFFSET
            entity = self._core.players[self.current_player].board[board_index]
            if isinstance(entity, Amulet):
                return ActivateAmulet(self.current_player, entity.entity_id)
            return Evolve(self.current_player, entity.entity_id)
        relative = action - self.ATTACK_OFFSET
        attacker_index = relative // self.TARGETS_PER_ATTACKER
        target_slot = relative % self.TARGETS_PER_ATTACKER
        attacker = self._core.players[self.current_player].board[attacker_index]
        target_id = None
        if target_slot:
            target_id = self._core.players[1 - self.current_player].board[target_slot - 1].entity_id
        return Attack(self.current_player, attacker.entity_id, target_id)

    def _special_hand_commands(self, hand_index: int) -> list[GameCommand]:
        player = self._core.players[self.current_player]
        if hand_index >= len(player.hand):
            return []
        entity_id = player.hand[hand_index].entity_id
        return [
            command
            for command in self._cached_legal_commands()
            if (
                isinstance(command, BeginFusion)
                and command.fusion_entity_id == entity_id
            )
            or (
                isinstance(command, PlayCard)
                and command.hand_index == hand_index
                and command.mode_id != "normal"
            )
        ]

    def _decode_mode_play(self, action: int) -> GameCommand:
        relative = action - self.MODE_PLAY_OFFSET
        hand_index = relative // MAX_SPECIAL_MODES_PER_CARD
        mode_slot = relative % MAX_SPECIAL_MODES_PER_CARD
        if hand_index >= len(self._core.players[self.current_player].hand):
            raise ValueError(f"Hand index {hand_index} out of range")
        commands = self._special_hand_commands(hand_index)
        if mode_slot >= len(commands):
            raise ValueError(f"Special slot {mode_slot} out of range for hand index {hand_index}")
        return commands[mode_slot]

    def _encode_command(self, command: GameCommand) -> int | None:
        if isinstance(command, EndTurn):
            return self.END_TURN
        if isinstance(command, UseExtraPP):
            return self.USE_EXTRA_PP
        if isinstance(command, Choose):
            request = self._core.state.pending_choice
            if request is None:
                return None
            if request.choice_kind is ChoiceKind.GRAVEYARD:
                for index, option in enumerate(request.options):
                    if option.option_id == command.option_id:
                        page_start = self._graveyard_page * self.GRAVEYARD_PAGE_SIZE
                        page_end = page_start + self.GRAVEYARD_PAGE_SIZE
                        if page_start <= index < page_end:
                            slot = index - page_start
                            return self.GRAVEYARD_SLOT_OFFSET + slot
                        return None
                return None
            for index, option in enumerate(request.options[: self.MAX_CHOICE_OPTIONS]):
                if option.option_id == command.option_id:
                    return self.CHOICE_OFFSET + index
            return None
        if isinstance(command, PlayCard):
            if command.mode_id == "normal":
                return self.PLAY_OFFSET + command.hand_index
            for idx, special in enumerate(self._special_hand_commands(command.hand_index)):
                if special == command and idx < MAX_SPECIAL_MODES_PER_CARD:
                    return self.MODE_PLAY_OFFSET + command.hand_index * MAX_SPECIAL_MODES_PER_CARD + idx
            return None
        if isinstance(command, BeginFusion):
            hand_index = next(
                (
                    index
                    for index, card in enumerate(self._core.players[self.current_player].hand)
                    if card.entity_id == command.fusion_entity_id
                ),
                None,
            )
            if hand_index is None:
                return None
            for idx, special in enumerate(self._special_hand_commands(hand_index)):
                if special == command and idx < MAX_SPECIAL_MODES_PER_CARD:
                    return self.MODE_PLAY_OFFSET + hand_index * MAX_SPECIAL_MODES_PER_CARD + idx
            return None
        if isinstance(command, ActivateAmulet):
            index = self._unit_index(
                self._core.players[self.current_player].board,
                command.amulet_id,
            )
            return self.EVOLVE_OFFSET + index
        if isinstance(command, Evolve):
            index = self._unit_index(self._core.players[self.current_player].board, command.unit_id)
            return self.EVOLVE_OFFSET + index
        if isinstance(command, SuperEvolve):
            index = self._unit_index(self._core.players[self.current_player].board, command.unit_id)
            return self.SUPER_EVOLVE_OFFSET + index
        if isinstance(command, Attack):
            attacker_index = self._unit_index(self._core.players[self.current_player].board, command.attacker_id)
            base = self.ATTACK_OFFSET + attacker_index * self.TARGETS_PER_ATTACKER
            if command.target_id is None:
                return base
            target_index = self._unit_index(self._core.players[1 - self.current_player].board, command.target_id)
            return base + 1 + target_index
        return None

    def _build_grave_page_mask(self) -> list[bool]:
        mask = [False] * self.ACTION_SIZE
        request = self._core.state.pending_choice
        if request is None or request.choice_kind is not ChoiceKind.GRAVEYARD:
            return mask
        total = len(request.options)
        page_start = self._graveyard_page * self.GRAVEYARD_PAGE_SIZE
        page_end = min(page_start + self.GRAVEYARD_PAGE_SIZE, total)
        for i in range(page_start, page_end):
            mask[self.GRAVEYARD_SLOT_OFFSET + (i - page_start)] = True
        if self._graveyard_page > 0:
            mask[self.GRAVEYARD_PREV_PAGE] = True
        if page_end < total:
            mask[self.GRAVEYARD_NEXT_PAGE] = True
        return mask

    def _legal_non_choice_mask(
        self, commands: Sequence[GameCommand] | None = None
    ) -> list[bool]:
        mask = [False] * self.ACTION_SIZE
        for command in self._core.legal_commands() if commands is None else commands:
            if not isinstance(command, Choose):
                action = self._encode_command(command)
                if action is not None:
                    mask[action] = True
        return mask

    def _legal_choice_mask(
        self, commands: Sequence[GameCommand] | None = None
    ) -> list[bool]:
        mask = [False] * self.ACTION_SIZE
        request = self._core.state.pending_choice
        if request is None:
            return mask
        if request.choice_kind is ChoiceKind.GRAVEYARD:
            return self._build_grave_page_mask()
        for command in self._core.legal_commands() if commands is None else commands:
            if isinstance(command, Choose) and request.choice_kind is not ChoiceKind.GRAVEYARD:
                action = self._encode_command(command)
                if action is not None:
                    mask[action] = True
        return mask

    def _cached_legal_commands(self) -> tuple[GameCommand, ...]:
        cached = self._legal_commands_cache
        if cached is not None:
            self._assert_cache_coherent()
            self._cache_stats["legal_command_hits"] += 1
            return cached
        commands = tuple(self._core.legal_commands())
        self._legal_commands_cache = commands
        self._cache_stats["legal_command_misses"] += 1
        self._record_debug_cache_fingerprint()
        return commands

    def action_mask(self) -> list[bool]:
        self._sync_choice_page()
        if self._action_mask_cache is not None:
            self._assert_cache_coherent()
            self._cache_stats["action_mask_hits"] += 1
            return list(self._action_mask_cache)
        if self.terminated or self.truncated:
            mask = [False] * self.ACTION_SIZE
            self._action_mask_cache = tuple(mask)
            self._cache_stats["action_mask_misses"] += 1
            self._record_debug_cache_fingerprint()
            return mask
        request = self._core.state.pending_choice
        if request is not None and request.choice_kind is ChoiceKind.GRAVEYARD:
            mask = self._build_grave_page_mask()
        else:
            commands = self._cached_legal_commands()
            if request is not None:
                mask = self._legal_choice_mask(commands)
            else:
                mask = self._legal_non_choice_mask(commands)
        self._action_mask_cache = tuple(mask)
        self._audit_action_mask(mask)
        self._cache_stats["action_mask_misses"] += 1
        self._record_debug_cache_fingerprint()
        return mask

    def _audit_action_mask(self, mask: Sequence[bool]) -> None:
        recorder = self._core.runtime_coverage
        if recorder is None:
            return
        commands = self._cached_legal_commands()
        mismatches: list[str] = []
        encoded: dict[int, GameCommand] = {}
        for command in commands:
            action = self._encode_command(command)
            if action is None:
                mismatches.append(f"unencoded:{type(command).__name__}")
                continue
            encoded[action] = command
            if action >= len(mask) or not mask[action]:
                mismatches.append(
                    f"masked:{type(command).__name__}:{action}"
                )
        page_actions = {self.GRAVEYARD_PREV_PAGE, self.GRAVEYARD_NEXT_PAGE}
        for action, allowed in enumerate(mask):
            if not allowed or action in page_actions:
                continue
            try:
                decoded = self._decode_action(action)
            except (IndexError, ValueError):
                mismatches.append(f"undecodable:{action}")
                continue
            if encoded.get(action) != decoded:
                mismatches.append(f"decode_disagrees:{action}")
        if mismatches:
            recorder.record_diagnostic(
                "action_mask_mismatch",
                detail="|".join(sorted(mismatches)),
            )

    def _graveyard_total_pages(self) -> int:
        request = self._core.state.pending_choice
        if request is None or request.choice_kind is not ChoiceKind.GRAVEYARD:
            return 0
        return max(
            1,
            (len(request.options) + self.GRAVEYARD_PAGE_SIZE - 1)
            // self.GRAVEYARD_PAGE_SIZE,
        )

    def _sync_choice_page(self, *, invalidate: bool = True) -> None:
        request = self._core.state.pending_choice
        if request is None:
            request_key = None
        else:
            identity = request.request_id if request.request_id > 0 else id(request)
            request_key = (request.continuation_id, identity)
        if request_key != self._last_choice_request_key:
            self._graveyard_page = 0
            self._last_choice_request_key = request_key
            if invalidate:
                self._invalidate_caches(
                    advance_transition=True,
                    reason="pending choice changed",
                )

    @staticmethod
    def _unit_index(board: list[BoardCard], entity_id: int) -> int:
        for index, unit in enumerate(board):
            if unit.entity_id == entity_id:
                return index
        raise ValueError(f"Unit {entity_id} is not on the board")

    def _card_features(
        self,
        card: CardDefinition | HandCard | None,
        *,
        perspective: int | None = None,
    ) -> list[float]:
        if card is None:
            return [0.0] * 10
        fused_count = len(card.fused_material_ids) if isinstance(card, HandCard) else 0
        fusion_used = (
            isinstance(card, HandCard) and card.fusion_used_turn == self.turn
        )
        union_burst_gauge = 0
        if (
            isinstance(card, HandCard)
            and self._core.rulebook.union_bursts_for(card.card_id)
        ):
            player_index = self.decision_player if perspective is None else perspective
            union_burst_gauge = card.union_burst_gauge(
                self._core.players[player_index].turns_started
            )
        return [
            1.0, card.cost / ShadowverseEnv.MAX_MANA,
            (card.attack or 0) / 20, (card.life or 0) / 20,
            float(card.card_type == "随从"),
            float(card.card_type == "护符"),
            float(card.card_type == "法术"),
            min(union_burst_gauge, 15) / 15,
            fused_count / ShadowverseEnv.MAX_HAND,
            float(fusion_used),
        ]

    def _board_features(self, entity: BoardCard | None) -> list[float]:
        if entity is None:
            return [0.0] * 14
        if isinstance(entity, Amulet):
            return [1.0, 0.0, (entity.countdown or 0) / 10, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, float(entity.activated_turn == self.turn), len(entity.fused_material_ids) / ShadowverseEnv.MAX_HAND, 0.0, float(AbilityKeyword.AURA in entity.definition.abilities)]
        unit = entity
        return [
            1.0, unit.attack / 20, unit.health / 20,
            float(unit.can_attack), float(unit.has_guard),
            float(unit.has_keyword("疾驰")),
            float(unit.evolved), float(unit.rush_only), float(unit.super_evolved),
            float(unit.barrier_charges > 0), float(unit.ambush_active),
            len(unit.fused_material_ids) / ShadowverseEnv.MAX_HAND,
            float(unit.has_intimidate),
            float(unit.has_aura),
        ]


class _PageTurn(Exception):
    """Signal that a page-turn action was executed; step() catches this."""
    pass
