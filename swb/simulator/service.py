from __future__ import annotations

import copy
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from swb.db.repository import CardRepository
from swb.engine.commands import (
    ActivateAmulet,
    Attack,
    BeginFusion,
    Choose,
    EndTurn,
    Evolve,
    PlayCard,
    SuperEvolve,
    UseExtraPP,
)
from swb.engine.environment import MATCH_SETUP_OFFICIAL, ShadowverseEnv
from swb.engine.state import Amulet, HandCard, Unit
from swb.engine.union_burst import UnionBurstKind
from swb.rl.action_guard import FusionCancelActionGuard
from swb.rl.checkpoint import CHECKPOINT_SCHEMA_VERSION
from swb.rl.fixed_decks import (
    FixedTrainingDeck,
    fixed_training_deck_names,
    get_fixed_training_deck,
)
from swb.rl.ppo import PPOConfig, PPOTrainer
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.versioning import ExperimentVersions, stable_json_sha256
from swb.simulator.history import MatchHistoryStore
from swb.simulator.timeline import build_animation_cues, serialize_event


IMAGE_FILENAME_PATTERN = re.compile(r"^[0-9]+\.png$")
PRIVATE_DRAW_EVENT_TYPES = {
    "card_drawn",
    "card_added_to_hand",
    "hand_card_transformed",
    "hand_follower_stats_increased",
    "spellboosted",
}


@dataclass
class PolicyDecision:
    action: int
    value: float
    logits: dict[int, float]
    probabilities: dict[int, float]
    suppressed_actions: tuple[int, ...] = ()


@dataclass(frozen=True)
class SimulatorModelOption:
    model_id: str
    path: Path
    display_name: str
    group: str

    def manifest(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "display_name": self.display_name,
            "group": self.group,
            "filename": self.path.name,
            "size_bytes": self.path.stat().st_size,
        }


@dataclass
class InferenceModelBundle:
    option: SimulatorModelOption
    trainer: PPOTrainer
    warnings: list[str]
    specialist_deck: FixedTrainingDeck | None
    policy: "DeterministicPPOPolicy"


@dataclass
class DeterministicPPOPolicy:
    model: Any
    flattener: Any
    device: torch.device
    hidden: torch.Tensor
    fusion_cancel_guard: FusionCancelActionGuard | None = field(
        default_factory=FusionCancelActionGuard
    )

    @classmethod
    def from_trainer(
        cls,
        trainer,
        *,
        enable_fusion_cancel_guard: bool = True,
    ) -> "DeterministicPPOPolicy":
        trainer.model.eval()
        return cls(
            model=trainer.model,
            flattener=trainer.flattener,
            device=trainer.device,
            hidden=trainer.model.initial_state(1, device=trainer.device),
            fusion_cancel_guard=(
                FusionCancelActionGuard()
                if enable_fusion_cancel_guard
                else None
            ),
        )

    def reset(self) -> None:
        self.hidden = self.model.initial_state(1, device=self.device)
        if self.fusion_cancel_guard is not None:
            self.fusion_cancel_guard.reset()

    def decision(self, env: ShadowverseEnv, player_id: int) -> PolicyDecision:
        legal_action_mask = np.asarray(env.action_mask(), dtype=np.bool_)
        action_mask = (
            legal_action_mask.copy()
            if self.fusion_cancel_guard is None
            else self.fusion_cancel_guard.policy_mask(
                env,
                player_id,
                legal_action_mask,
            )
        )
        observation = env.observation(
            perspective=player_id,
            action_mask=action_mask,
        )
        vector = torch.from_numpy(self.flattener.encode(observation)).to(
            self.device
        ).unsqueeze(0)
        card_indices = torch.from_numpy(
            self.flattener.encode_cards(observation)
        ).to(self.device).unsqueeze(0)
        mask = torch.from_numpy(action_mask).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value, hidden = self.model.forward_step(
                vector,
                self.hidden,
                card_indices,
            )
            masked = self.model.masked_logits(logits, mask)
            probabilities = torch.softmax(masked, dim=-1)
        self.hidden = hidden.detach()
        action = int(masked.argmax(dim=-1).item())
        legal_actions = np.flatnonzero(legal_action_mask).tolist()
        suppressed_actions = tuple(
            int(action_id)
            for action_id in np.flatnonzero(
                legal_action_mask & ~action_mask
            )
        )
        if self.fusion_cancel_guard is not None:
            self.fusion_cancel_guard.record_selected_action(
                env,
                player_id,
                action,
            )
        return PolicyDecision(
            action=action,
            value=float(value.reshape(-1)[0].item()),
            logits={
                action_id: float(logits[0, action_id].item())
                for action_id in legal_actions
            },
            probabilities={
                action_id: float(probabilities[0, action_id].item())
                for action_id in legal_actions
            },
            suppressed_actions=suppressed_actions,
        )

    def action(self, env: ShadowverseEnv, player_id: int) -> int:
        return self.decision(env, player_id).action


class MatchSimulator:
    """Single-user local human-vs-checkpoint match service."""

    def __init__(
        self,
        *,
        database: str | Path,
        checkpoint: str | Path,
        card_catalog: str | Path,
        image_directory: str | Path,
        history_directory: str | Path,
        checkpoint_directory: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.database_path = Path(database)
        self.checkpoint_path = Path(checkpoint)
        self.checkpoint_directory = (
            self.checkpoint_path.parent
            if checkpoint_directory is None
            else Path(checkpoint_directory)
        )
        self.card_catalog_path = Path(card_catalog)
        self.image_directory = Path(image_directory)
        for label, path in (
            ("database", self.database_path),
            ("checkpoint", self.checkpoint_path),
            ("checkpoint directory", self.checkpoint_directory),
            ("card catalog", self.card_catalog_path),
            ("image directory", self.image_directory),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        self.device = device
        self.snapshot = WorkerAssetsSnapshot.build(
            CardRepository(self.database_path)
        )
        self.assets = self.snapshot.load()
        self.available_models = self._discover_models(
            self.checkpoint_path,
            self.checkpoint_directory,
        )
        self._available_models_by_id = {
            option.model_id: option
            for option in self.available_models
        }
        current_option = next(
            option
            for option in self.available_models
            if option.path.resolve() == self.checkpoint_path.resolve()
        )
        initial_model = self._load_model_bundle(current_option)
        self.current_model = initial_model.option
        self.trainer = initial_model.trainer
        self.compatibility_warnings = initial_model.warnings
        self.specialist_deck_recipe = initial_model.specialist_deck
        self.policy = initial_model.policy
        self.available_deck_recipes = tuple(
            get_fixed_training_deck(name)
            for name in fixed_training_deck_names()
        )
        self._available_decks_by_name = {
            recipe.name: recipe
            for recipe in self.available_deck_recipes
        }
        default_deck_recipe = (
            self.specialist_deck_recipe
            or self.available_deck_recipes[0]
        )
        self.human_deck_recipe = default_deck_recipe
        self.ai_deck_recipe = default_deck_recipe
        self.texture_paths = self._load_texture_paths()
        self.history_store = MatchHistoryStore(history_directory)
        self.env: ShadowverseEnv | None = None
        self.human_player = 0
        self.seed = 0
        self.last_ai_actions: list[str] = []
        self.current_record: dict[str, Any] | None = None
        self.animation_batch: list[dict[str, Any]] = []
        self.animation_generation = 0
        self._lock = threading.RLock()

    @staticmethod
    def _discover_models(
        checkpoint_path: Path,
        checkpoint_directory: Path,
    ) -> tuple[SimulatorModelOption, ...]:
        root = checkpoint_directory.resolve()
        default = checkpoint_path.resolve()
        candidates = {default}
        candidates.update(path.resolve() for path in root.rglob("*.pt"))
        options: list[SimulatorModelOption] = []
        used_ids: set[str] = set()
        for candidate in sorted(candidates, key=lambda path: str(path).lower()):
            if candidate != default:
                try:
                    relative = candidate.relative_to(root)
                except ValueError:
                    continue
                auxiliary_directory = any(
                    part == "tuning" or part.endswith("_history")
                    for part in relative.parts[:-1]
                )
                auxiliary_file = (
                    "preflight" in candidate.stem.lower()
                    or candidate.stem.lower().endswith("_init")
                )
                if auxiliary_directory or auxiliary_file:
                    continue
            try:
                relative = candidate.relative_to(root)
                model_id = relative.as_posix()
            except ValueError:
                model_id = f"default/{candidate.name}"
            if model_id in used_ids:
                raise ValueError(f"duplicate simulator model id: {model_id}")
            used_ids.add(model_id)
            parent = Path(model_id).parent.as_posix()
            group = "根目录" if parent == "." else parent
            label = Path(model_id).with_suffix("").as_posix()
            options.append(SimulatorModelOption(
                model_id=model_id,
                path=candidate,
                display_name=label.replace("/", " · "),
                group=group,
            ))
        options.sort(
            key=lambda option: (
                option.path.resolve() != default,
                option.model_id.lower(),
            )
        )
        return tuple(options)

    def _load_model_bundle(
        self,
        option: SimulatorModelOption,
    ) -> InferenceModelBundle:
        trainer, warnings = self._load_inference_checkpoint(
            option.path,
            self.snapshot,
            device=self.device,
        )
        training_deck_name = trainer.config.training_deck
        specialist_deck = (
            None
            if training_deck_name is None
            else get_fixed_training_deck(training_deck_name)
        )
        return InferenceModelBundle(
            option=option,
            trainer=trainer,
            warnings=warnings,
            specialist_deck=specialist_deck,
            policy=DeterministicPPOPolicy.from_trainer(trainer),
        )

    def _resolve_model(
        self,
        model_id: str | None,
    ) -> SimulatorModelOption:
        if model_id is None:
            return self.current_model
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model must be a non-empty model id")
        try:
            return self._available_models_by_id[model_id]
        except KeyError as error:
            raise ValueError(f"unknown model: {model_id!r}") from error

    def _activate_model(self, bundle: InferenceModelBundle) -> None:
        self.current_model = bundle.option
        self.checkpoint_path = bundle.option.path
        self.trainer = bundle.trainer
        self.compatibility_warnings = bundle.warnings
        self.specialist_deck_recipe = bundle.specialist_deck
        self.policy = bundle.policy

    @staticmethod
    def _load_inference_checkpoint(
        checkpoint_path: Path,
        snapshot: WorkerAssetsSnapshot,
        *,
        device: str,
    ) -> tuple[PPOTrainer, list[str]]:
        try:
            payload = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as error:
            raise ValueError(
                f"unable to load simulator checkpoint {checkpoint_path}: {error}"
            ) from error
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported simulator checkpoint schema version: "
                f"{payload.get('checkpoint_schema_version')!r}"
            )
        trainer_state = payload["trainer"]
        config_payload = dict(trainer_state["config"])
        config_payload.setdefault("match_setup", "legacy")
        checkpoint_observation = str(
            payload.get("versions", {}).get("observation_version", "")
        )
        config_payload.setdefault(
            "observation_version",
            (
                "v3"
                if checkpoint_observation.startswith("observation-v3")
                else (
                    "v4.1"
                    if checkpoint_observation.startswith(
                        "observation-v4.1"
                    )
                    else "v4"
                )
            ),
        )
        trainer = PPOTrainer(
            snapshot,
            master_seed=int(trainer_state["master_seed"]),
            config=PPOConfig(**config_payload),
            device=device,
        )
        checkpoint_versions = ExperimentVersions(**payload["versions"])
        if trainer.env is None:
            raise RuntimeError("checkpoint trainer did not create an environment")
        runtime_versions = ExperimentVersions.capture(
            trainer.env,
            trainer.assets.catalog,
            rulebook_sha256=snapshot.rulebook_sha256,
        )
        expected = checkpoint_versions.to_dict()
        actual = runtime_versions.to_dict()
        mismatches = {
            key: {"checkpoint": expected[key], "runtime": actual[key]}
            for key in expected
            if expected[key] != actual[key]
        }
        inference_only_mismatches = {
            "rulebook_sha256",
            "catalog_sha256",
            "training_pool_sha256",
        }
        unsupported = set(mismatches) - inference_only_mismatches
        if unsupported:
            details = ", ".join(sorted(unsupported))
            raise ValueError(
                "checkpoint is incompatible with the simulator runtime: "
                f"{details}"
            )
        warnings: list[str] = []
        if "rulebook_sha256" in mismatches:
            values = mismatches["rulebook_sha256"]
            warnings.append(
                "该模型训练时的规则库与当前 main 不同；动作与观察版本一致，"
                "本模拟器允许纯推理加载，但对局结果不能视为原训练环境的严格复现。"
                f" checkpoint={values['checkpoint']} runtime={values['runtime']}"
            )
        if {
            "catalog_sha256",
            "training_pool_sha256",
        } & mismatches.keys():
            warnings.append(
                "该模型训练时的 Catalog/训练卡池与当前运行时不同；"
                "卡牌词表、动作和观察版本一致，因此固定卡组纯推理可以加载，"
                "但不能据此恢复训练、重现原训练分布或进行公平强度比较。"
            )
        trainer.model.load_state_dict(payload["model_state"])
        trainer.model.eval()
        return trainer, warnings

    def _load_texture_paths(self) -> dict[int, dict[str, str]]:
        payload = json.loads(self.card_catalog_path.read_text(encoding="utf-8"))
        return {
            int(card["card_id"]): {
                str(variant): Path(source).name
                for variant, source in (card.get("textures") or {}).items()
            }
            for card in payload
        }

    def new_match(
        self,
        *,
        seed: int | None = None,
        human_player: int = 0,
        human_deck: str | None = None,
        ai_deck: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        if human_player not in (0, 1):
            raise ValueError("human_player must be 0 or 1")
        with self._lock:
            selected_model = self._resolve_model(model)
            loaded_model = (
                None
                if selected_model.model_id == self.current_model.model_id
                else self._load_model_bundle(selected_model)
            )
            active_specialist = (
                self.specialist_deck_recipe
                if loaded_model is None
                else loaded_model.specialist_deck
            )
            default_recipe = (
                active_specialist
                or self.available_deck_recipes[0]
            )
            human_recipe = self._resolve_deck(
                human_deck,
                default=default_recipe,
                label="human_deck",
            )
            ai_recipe = self._resolve_deck(
                ai_deck,
                default=default_recipe,
                label="ai_deck",
            )
            recipes_by_player: list[FixedTrainingDeck] = [
                human_recipe,
                ai_recipe,
            ]
            recipes_by_player[human_player] = human_recipe
            recipes_by_player[1 - human_player] = ai_recipe
            deck_a = recipes_by_player[0].build(self.assets.catalog)
            deck_b = recipes_by_player[1].build(self.assets.catalog)

            self._finalize_current_record("abandoned")
            if loaded_model is not None:
                self._activate_model(loaded_model)
            self.seed = (
                int(time.time_ns() & 0x7FFFFFFF)
                if seed is None
                else int(seed)
            )
            self.human_player = human_player
            self.human_deck_recipe = human_recipe
            self.ai_deck_recipe = ai_recipe
            self.env = ShadowverseEnv(
                deck_a,
                deck_b,
                class_a=recipes_by_player[0].class_id,
                class_b=recipes_by_player[1].class_id,
                seed=self.seed,
                rulebook=self.assets.rulebook,
                card_resolver=self.assets.catalog.resolve,
                observation_version=self.trainer.config.observation_version,
                card_vocabulary=self.assets.catalog.card_vocabulary,
                max_game_turns=self.trainer.config.max_game_turns,
                max_agent_steps=self.trainer.config.max_agent_steps_per_episode,
                training_mode=False,
                validate_invariants=True,
                match_setup=MATCH_SETUP_OFFICIAL,
            )
            self.env.reset(seed=self.seed)
            self.policy.reset()
            self.last_ai_actions = []
            self.animation_batch = []
            self.animation_generation = 0
            self.current_record = self.history_store.new_record(
                seed=self.seed,
                human_player=self.human_player,
                deck=self._match_deck_manifest(),
                checkpoint=self.current_model.model_id,
                warnings=self._active_warnings(),
                initial_state=self._history_snapshot(),
                initial_logs=list(self.env.logs),
            )
            self._advance_ai()
            return self.state()

    def apply_human_action(self, action: int) -> dict[str, Any]:
        with self._lock:
            env = self._require_env()
            if env.terminated or env.truncated:
                raise ValueError("match has already ended")
            if env.decision_player != self.human_player:
                raise ValueError("it is not the human player's decision")
            mask = env.action_mask()
            if action < 0 or action >= env.ACTION_SIZE or not mask[action]:
                raise ValueError(f"illegal action: {action}")
            self.animation_generation += 1
            self.animation_batch = []
            self._step_with_recording(action, actor_role="human")
            self.last_ai_actions = []
            self._advance_ai()
            return self.state()

    def list_history(self, *, limit: int = 50) -> dict[str, Any]:
        with self._lock:
            return {
                "matches": self.history_store.list_summaries(limit=limit),
                "current_match_id": (
                    None
                    if self.current_record is None
                    else self.current_record["match_id"]
                ),
            }

    def match_history(self, match_id: str) -> dict[str, Any]:
        with self._lock:
            record = self.history_store.load(match_id)
            if record is None:
                raise FileNotFoundError(f"match history not found: {match_id}")
            return self._public_history_record(record)

    def state(self) -> dict[str, Any]:
        with self._lock:
            env = self._require_env()
            state = env._core.state
            players = [
                self._serialize_player(
                    player_index,
                    reveal_hand=player_index == self.human_player,
                )
                for player_index in (0, 1)
            ]
            actions = (
                self._legal_actions()
                if (
                    not env.terminated
                    and not env.truncated
                    and env.decision_player == self.human_player
                )
                else []
            )
            pending = state.pending_choice
            return {
                "seed": self.seed,
                "match_id": (
                    None
                    if self.current_record is None
                    else self.current_record["match_id"]
                ),
                "deck": self._match_deck_manifest(),
                "human_deck": self._deck_manifest(
                    self.human_deck_recipe
                ),
                "ai_deck": self._deck_manifest(self.ai_deck_recipe),
                "specialist_deck": (
                    None
                    if self.specialist_deck_recipe is None
                    else self._deck_manifest(self.specialist_deck_recipe)
                ),
                "available_decks": [
                    self._deck_manifest(recipe)
                    for recipe in self.available_deck_recipes
                ],
                "model": self.current_model.manifest(),
                "available_models": [
                    option.manifest()
                    for option in self.available_models
                ],
                "checkpoint": self.checkpoint_path.name,
                "warnings": self._active_warnings(),
                "human_player": self.human_player,
                "ai_player": 1 - self.human_player,
                "current_player": env.current_player,
                "decision_player": env.decision_player,
                "first_player": state.first_player,
                "turn": env.turn,
                "phase": state.phase.value,
                "terminated": env.terminated,
                "truncated": env.truncated,
                "winner": env.winner,
                "human_turn": (
                    not env.terminated
                    and not env.truncated
                    and env.decision_player == self.human_player
                ),
                "players": players,
                "actions": actions,
                "pending_choice": (
                    None
                    if pending is None
                    else {
                        "prompt": pending.prompt,
                        "kind": pending.choice_kind.value,
                        "target_count": pending.target_count,
                        "selected_count": len(pending.selected_options),
                    }
                ),
                "last_ai_actions": list(self.last_ai_actions),
                "animation_batch_id": (
                    None
                    if self.current_record is None
                    else (
                        f"{self.current_record['match_id']}:"
                        f"{self.animation_generation}"
                    )
                ),
                "animation_batch": list(self.animation_batch),
                "logs": self._public_logs(env.logs[-80:]),
            }

    @staticmethod
    def _deck_manifest(recipe: FixedTrainingDeck) -> dict[str, Any]:
        return {
            "name": recipe.name,
            "display_name": recipe.display_name,
            "class_id": recipe.class_id,
            "sha256": recipe.sha256,
        }

    def _match_deck_manifest(self) -> dict[str, Any]:
        human = self._deck_manifest(self.human_deck_recipe)
        ai = self._deck_manifest(self.ai_deck_recipe)
        display_name = (
            human["display_name"]
            if human["name"] == ai["name"]
            else f"{human['display_name']} vs {ai['display_name']}"
        )
        return {
            "name": human["name"],
            "display_name": display_name,
            "sha256": stable_json_sha256({
                "human": human["sha256"],
                "ai": ai["sha256"],
            }),
            "human": human,
            "ai": ai,
        }

    def _resolve_deck(
        self,
        name: str | None,
        *,
        default: FixedTrainingDeck,
        label: str,
    ) -> FixedTrainingDeck:
        if name is None:
            return default
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} must be a non-empty deck name")
        try:
            return self._available_decks_by_name[name]
        except KeyError as error:
            raise ValueError(f"unknown {label}: {name!r}") from error

    def _active_warnings(self) -> list[str]:
        warnings = list(self.compatibility_warnings)
        if (
            self.specialist_deck_recipe is not None
            and self.ai_deck_recipe.name != self.specialist_deck_recipe.name
        ):
            warnings.append(
                "当前模型声明了固定专精卡组；让 AI 使用其他卡组仅适合"
                "测试通用动作能力，不代表该卡组的训练强度。"
            )
        return warnings

    def image_path(self, filename: str) -> Path | None:
        if not IMAGE_FILENAME_PATTERN.fullmatch(filename):
            return None
        path = self.image_directory / filename
        return path if path.is_file() else None

    def _advance_ai(self, max_actions: int = 512) -> None:
        env = self._require_env()
        ai_player = 1 - self.human_player
        for _ in range(max_actions):
            if env.terminated or env.truncated or env.decision_player == self.human_player:
                return
            if env.decision_player != ai_player:
                raise RuntimeError("simulator reached an unknown decision owner")
            policy_decision = self.policy.decision(env, ai_player)
            action = policy_decision.action
            raw_description = self._describe_action(action)
            public_description = self._public_action_description(
                raw_description,
                actor_role="ai",
            )
            self._step_with_recording(
                action,
                actor_role="ai",
                description=raw_description,
                policy_decision=policy_decision,
            )
            self.last_ai_actions.append(public_description["label"])
            self.last_ai_actions = self.last_ai_actions[-24:]
        raise RuntimeError("AI exceeded the simulator action limit")

    def _step_with_recording(
        self,
        action: int,
        *,
        actor_role: str,
        description: dict[str, Any] | None = None,
        policy_decision: PolicyDecision | None = None,
    ) -> None:
        env = self._require_env()
        if self.current_record is None:
            raise RuntimeError("simulator match history was not initialized")
        description = (
            self._describe_action(action)
            if description is None
            else description
        )
        acting_player = env.decision_player
        before = self._history_snapshot()
        decision = self._decision_record(
            action,
            actor_role=actor_role,
            policy_decision=policy_decision,
        )
        public_description = self._public_action_description(
            description,
            actor_role=actor_role,
        )
        event_offset = len(env._core.event_history)
        log_offset = len(env.logs)
        entity_names = self._entity_names()

        env.step(action)

        entity_names.update(self._entity_names())
        logs = list(env.logs[log_offset:])
        events = [
            serialize_event(
                event,
                entity_names=entity_names,
                card_lookup=self.assets.catalog.resolve,
            )
            for event in env._core.event_history[event_offset:]
        ]
        cues = build_animation_cues(
            events,
            logs=logs,
            action_label=description["label"],
        )
        public_logs = self._public_logs(logs)
        public_events = [
            self._public_event(event)
            for event in events
        ]
        public_cues = build_animation_cues(
            public_events,
            logs=public_logs,
            action_label=public_description["label"],
        )
        sequence = len(self.current_record["actions"]) + 1
        self._identify_cues(cues, sequence=sequence)
        self._identify_cues(public_cues, sequence=sequence)
        self.animation_batch.extend(public_cues)
        self.current_record["actions"].append(
            {
                "sequence": sequence,
                "actor_role": actor_role,
                "player_index": acting_player,
                "action_id": action,
                "action": description,
                "decision": decision,
                "before": before,
                "after": self._history_snapshot(),
                "logs": logs,
                "events": events,
                "animations": cues,
            }
        )
        self._sync_current_record()

    def _sync_current_record(self, *, status: str | None = None) -> None:
        if self.current_record is None or self.env is None:
            return
        env = self.env
        if status is None:
            if env.terminated:
                status = "completed"
            elif env.truncated:
                status = "truncated"
            else:
                status = "ongoing"
        snapshot = self._history_snapshot()
        self.current_record.update(
            {
                "status": status,
                "winner": env.winner,
                "turn": env.turn,
                "phase": env._core.state.phase.value,
                "latest_state": snapshot,
                "logs": list(env.logs),
            }
        )
        self.history_store.write(self.current_record)

    def _finalize_current_record(self, status: str) -> None:
        if (
            self.current_record is None
            or self.current_record.get("status") != "ongoing"
        ):
            return
        self._sync_current_record(status=status)

    def _history_snapshot(self) -> dict[str, Any]:
        env = self._require_env()
        return {
            "turn": env.turn,
            "phase": env._core.state.phase.value,
            "current_player": env.current_player,
            "decision_player": env.decision_player,
            "first_player": env._core.state.first_player,
            "terminated": env.terminated,
            "truncated": env.truncated,
            "winner": env.winner,
            "players": [
                self._serialize_player(
                    player_index,
                    reveal_hand=True,
                )
                for player_index in (0, 1)
            ],
        }

    def _decision_record(
        self,
        action: int,
        *,
        actor_role: str,
        policy_decision: PolicyDecision | None,
    ) -> dict[str, Any]:
        legal_actions = self._legal_actions()
        legal_ids = {int(candidate["id"]) for candidate in legal_actions}
        if action not in legal_ids:
            raise RuntimeError("selected action is missing from the legal action set")
        if actor_role == "ai" and policy_decision is None:
            raise RuntimeError("AI action is missing its policy decision trace")
        if policy_decision is not None:
            if policy_decision.action != action:
                raise RuntimeError("policy trace action does not match selected action")
            if set(policy_decision.probabilities) != legal_ids:
                raise RuntimeError("policy trace does not match the legal action set")
            if not set(policy_decision.suppressed_actions) <= legal_ids:
                raise RuntimeError(
                    "policy trace suppressed an engine-illegal action"
                )

        candidates: list[dict[str, Any]] = []
        for candidate in legal_actions:
            action_id = int(candidate["id"])
            candidates.append(
                {
                    **candidate,
                    "selected": action_id == action,
                    "logit": (
                        None
                        if policy_decision is None
                        else policy_decision.logits[action_id]
                    ),
                    "probability": (
                        None
                        if policy_decision is None
                        else policy_decision.probabilities[action_id]
                    ),
                    "policy_suppressed": (
                        False
                        if policy_decision is None
                        else action_id in policy_decision.suppressed_actions
                    ),
                }
            )
        return {
            "type": "ppo_argmax" if policy_decision is not None else "human",
            "policy_architecture": (
                getattr(self.policy.model, "architecture", None)
                if policy_decision is not None
                else None
            ),
            "selected_action_id": action,
            "selected_probability": (
                None
                if policy_decision is None
                else policy_decision.probabilities[action]
            ),
            "value": (
                None if policy_decision is None else policy_decision.value
            ),
            "policy_suppressed_action_ids": (
                []
                if policy_decision is None
                else list(policy_decision.suppressed_actions)
            ),
            "legal_actions": candidates,
        }

    def _identify_cues(
        self,
        cues: list[dict[str, Any]],
        *,
        sequence: int,
    ) -> None:
        if self.current_record is None:
            raise RuntimeError("simulator match history was not initialized")
        for cue_index, cue in enumerate(cues, start=1):
            cue["id"] = (
                f"{self.current_record['match_id']}:{sequence}:{cue_index}"
            )
            cue["action_sequence"] = sequence

    def _entity_names(self) -> dict[int, str]:
        env = self._require_env()
        names: dict[int, str] = {}
        for player in env._core.players:
            names.update(
                (card.entity_id, card.name)
                for card in player.hand
            )
            names.update(
                (card.entity_id, card.definition.name)
                for card in player.board
            )
            names.update(
                (card.entity_id, card.definition.name)
                for card in player.graveyard
            )
            for instance in (*player.faiths, *player.emblems):
                definition = self.assets.catalog.resolve(
                    instance.source_card_id
                )
                names[instance.entity_id] = (
                    definition.name
                    if definition is not None
                    else str(instance.source_card_id)
                )
        return names

    def _public_history_record(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        public = copy.deepcopy(record)
        human_player = int(public["human_player"])
        hidden_player = 1 - human_player
        privacy = dict(public.get("privacy") or {})
        privacy["persistence"] = "full"
        privacy["online_history"] = "redacted"
        public["privacy"] = privacy

        def redact_snapshot(snapshot: object) -> None:
            if not isinstance(snapshot, dict):
                return
            players = snapshot.get("players")
            if not isinstance(players, list):
                return
            for player in players:
                if (
                    isinstance(player, dict)
                    and player.get("player_index") == hidden_player
                ):
                    player["hand"] = None

        redact_snapshot(public.get("initial_state"))
        redact_snapshot(public.get("latest_state"))
        public["logs"] = self._public_logs(
            public.get("logs", ()),
            human_player=human_player,
        )

        for action in public.get("actions", ()):
            if not isinstance(action, dict):
                continue
            redact_snapshot(action.get("before"))
            redact_snapshot(action.get("after"))
            action["logs"] = self._public_logs(
                action.get("logs", ()),
                human_player=human_player,
            )
            action["events"] = [
                self._public_event(
                    event,
                    human_player=human_player,
                )
                for event in action.get("events", ())
                if isinstance(event, dict)
            ]
            actor_role = action.get("actor_role")
            description = action.get("action")
            if actor_role == "ai" and isinstance(description, dict):
                action["action"] = self._redact_ai_history_action(description)
            if actor_role == "ai" and isinstance(action.get("decision"), dict):
                action["decision"] = {
                    "type": action["decision"].get("type"),
                    "privacy": "redacted",
                    "selected_action_id": action.get("action_id"),
                }

            action_label = (
                action.get("action", {}).get("label", "公开动作")
                if isinstance(action.get("action"), dict)
                else "公开动作"
            )
            cues = build_animation_cues(
                action["events"],
                logs=action["logs"],
                action_label=action_label,
            )
            sequence = int(action.get("sequence", 0))
            for cue_index, cue in enumerate(cues, start=1):
                cue["id"] = (
                    f"{public['match_id']}:{sequence}:{cue_index}"
                )
                cue["action_sequence"] = sequence
            action["animations"] = cues
        return public

    @staticmethod
    def _redact_ai_history_action(
        description: dict[str, Any],
    ) -> dict[str, Any]:
        kind = description.get("kind")
        if kind == "choice":
            return {
                "id": description.get("id"),
                "kind": "choice",
                "label": "完成隐藏选择",
            }
        if kind == "fusion":
            return {
                "id": description.get("id"),
                "kind": "fusion",
                "label": "进行融合",
            }
        return description

    def _public_event(
        self,
        event: dict[str, Any],
        *,
        human_player: int | None = None,
    ) -> dict[str, Any]:
        visible_player = (
            self.human_player
            if human_player is None
            else human_player
        )
        if (
            event["player_index"] != visible_player
            and event["type"] in PRIVATE_DRAW_EVENT_TYPES
        ):
            event = dict(event)
            event["source"] = {}
            event["target"] = {}
            event["source_id"] = None
            event["target_id"] = None
            event["metadata"] = {}
        return event

    def _public_action_description(
        self,
        description: dict[str, Any],
        *,
        actor_role: str,
    ) -> dict[str, Any]:
        env = self._require_env()
        if (
            actor_role == "ai"
            and env._core.state.phase.value == "mulligan"
            and description.get("kind") == "choice"
        ):
            return {
                "id": description["id"],
                "kind": "choice",
                "label": "完成起手换牌",
            }
        return description

    def _public_logs(
        self,
        lines: Any,
        *,
        human_player: int | None = None,
    ) -> list[str]:
        visible_player = (
            self.human_player
            if human_player is None
            else human_player
        )
        hidden_player = 1 - visible_player
        marker = f"[玩家 {hidden_player + 1}]"
        public: list[str] = []
        for original in lines:
            line = str(original)
            if marker in line and "起手：" in line:
                line = line.split("起手：", 1)[0] + "起手：隐藏卡牌"
            elif marker in line and "回合抽牌：" in line:
                line = line.split("回合抽牌：", 1)[0] + "回合抽牌：1 张"
            elif marker in line:
                continue
            public.append(line)
        return public

    def _require_env(self) -> ShadowverseEnv:
        if self.env is None:
            raise ValueError("start a match first")
        return self.env

    def _image_url(self, card_id: int, *, evolved: bool = False) -> str | None:
        textures = self.texture_paths.get(card_id, {})
        filename = textures.get("evo") if evolved else None
        filename = filename or textures.get("base")
        return None if filename is None else f"/api/images/{filename}"

    def _serialize_hand_card(
        self,
        index: int,
        card: HandCard,
        *,
        turns_started: int,
    ) -> dict[str, Any]:
        gauge = card.union_burst_gauge(turns_started)
        burst_labels = {
            UnionBurstKind.UNION_BURST: "奥义",
            UnionBurstKind.SUPER_SKYBOUND_ART: "解放奥义",
        }
        union_bursts = [
            {
                "kind": definition.kind.value,
                "label": burst_labels[definition.kind],
                "gauge": gauge,
                "threshold": definition.threshold,
                "remaining": max(0, definition.threshold - gauge),
                "ready": gauge >= definition.threshold,
            }
            for definition in sorted(
                self.assets.rulebook.union_bursts_for(card.card_id),
                key=lambda definition: (
                    definition.threshold,
                    definition.kind.value,
                ),
            )
        ]
        return {
            "index": index,
            "entity_id": card.entity_id,
            "card_id": card.card_id,
            "name": card.name,
            "type": card.card_type,
            "cost": card.current_cost,
            "printed_cost": card.definition.cost,
            "attack": card.attack,
            "health": card.life,
            "keywords": sorted(card.effective_keywords),
            "spellboost": card.spellboost_count,
            "union_bursts": union_bursts,
            "image_url": self._image_url(card.card_id),
        }

    def _serialize_board_card(self, card: Unit | Amulet) -> dict[str, Any]:
        definition = card.definition
        common = {
            "entity_id": card.entity_id,
            "card_id": definition.card_id,
            "name": definition.name,
            "type": definition.card_type,
            "cost": definition.cost,
            "image_url": self._image_url(
                definition.card_id,
                evolved=isinstance(card, Unit) and card.evolved,
            ),
        }
        if isinstance(card, Unit):
            common.update(
                {
                    "attack": card.attack,
                    "health": card.health,
                    "max_health": card.max_health,
                    "evolved": card.evolved,
                    "super_evolved": card.super_evolved,
                    "keywords": sorted(card.effective_keywords),
                    "can_attack": card.can_attack,
                    "attacks_remaining": card.attacks_remaining,
                    "barrier_charges": card.barrier_charges,
                }
            )
        else:
            common.update(
                {
                    "countdown": card.countdown,
                    "earth_sigils": card.earth_sigil_count,
                    "keywords": sorted(definition.keywords),
                }
            )
        return common

    def _serialize_player(
        self,
        player_index: int,
        *,
        reveal_hand: bool,
    ) -> dict[str, Any]:
        env = self._require_env()
        player = env._core.players[player_index]
        leader_area_used = len(player.faiths) + len(player.emblems)
        return {
            "player_index": player_index,
            "role": "human" if player_index == self.human_player else "ai",
            "class_id": player.class_id,
            "class_name": player.class_name,
            "health": player.health,
            "max_health": player.max_health,
            "mana": player.mana,
            "max_mana": player.max_mana,
            "extra_pp_available": player.extra_pp_available,
            "extra_pp_uses": player.extra_pp_uses,
            "extra_pp_active": player.extra_pp_active_turn == env.turn,
            "extra_pp_pending": bool(
                getattr(player, "extra_pp_pending", False)
            ),
            "evolution_points": player.evolution_points,
            "super_evolution_points": player.super_evolution_points,
            "shadows": player.shadows,
            "cooperation": player.cooperation,
            "cards_played_this_turn": player.cards_played_this_turn,
            "overflow_active": player.max_mana >= 7,
            "earth_sigils": player.earth_sigils,
            "leader_area_used": leader_area_used,
            "leader_area_limit": env._core.config.leader_area_limit,
            "deck_count": len(player.deck),
            "hand_count": len(player.hand),
            "graveyard_count": len(player.graveyard),
            "banished_count": len(player.banished),
            "board": [self._serialize_board_card(card) for card in player.board],
            "hand": (
                [
                    self._serialize_hand_card(
                        index,
                        card,
                        turns_started=player.turns_started,
                    )
                    for index, card in enumerate(player.hand)
                ]
                if reveal_hand
                else None
            ),
            "faiths": [
                {
                    "entity_id": faith.entity_id,
                    "faith_id": faith.faith_id,
                    "source_card_id": faith.source_card_id,
                    "source_name": self._source_card_name(
                        faith.source_card_id
                    ),
                    "image_url": self._image_url(faith.source_card_id),
                    "value": faith.value,
                }
                for faith in player.faiths
            ],
            "emblems": [
                {
                    "entity_id": emblem.entity_id,
                    "emblem_id": emblem.emblem_id,
                    "source_card_id": emblem.source_card_id,
                    "source_name": self._source_card_name(
                        emblem.source_card_id
                    ),
                    "image_url": self._image_url(emblem.source_card_id),
                    "countdown": emblem.countdown,
                }
                for emblem in player.emblems
            ],
        }

    def _source_card_name(self, card_id: int) -> str:
        definition = self.assets.catalog.resolve(card_id)
        return str(card_id) if definition is None else definition.name

    def _legal_actions(self) -> list[dict[str, Any]]:
        env = self._require_env()
        return [
            self._describe_action(action)
            for action, legal in enumerate(env.action_mask())
            if legal
        ]

    def _describe_action(self, action: int) -> dict[str, Any]:
        env = self._require_env()
        if action == env.GRAVEYARD_PREV_PAGE:
            return {"id": action, "kind": "page", "label": "上一页墓场"}
        if action == env.GRAVEYARD_NEXT_PAGE:
            return {"id": action, "kind": "page", "label": "下一页墓场"}
        command = env._decode_action(action)
        base: dict[str, Any] = {"id": action}
        if isinstance(command, EndTurn):
            return {**base, "kind": "end_turn", "label": "结束回合"}
        if isinstance(command, UseExtraPP):
            return {**base, "kind": "extra_pp", "label": "启用 Extra PP"}
        if isinstance(command, PlayCard):
            card = env._core.players[command.player_index].hand[command.hand_index]
            mode = "" if command.mode_id == "normal" else f" · {command.mode_id}"
            return {
                **base,
                "kind": "play",
                "label": f"打出 {card.name}{mode}",
                "source_entity_id": card.entity_id,
                "hand_index": command.hand_index,
                "mode_id": command.mode_id,
            }
        if isinstance(command, BeginFusion):
            card = next(
                card
                for card in env._core.players[command.player_index].hand
                if card.entity_id == command.fusion_entity_id
            )
            return {
                **base,
                "kind": "fusion",
                "label": f"融合 {card.name}",
                "source_entity_id": card.entity_id,
            }
        if isinstance(command, Attack):
            attacker = self._board_entity(command.attacker_id)
            target = (
                None
                if command.target_id is None
                else self._board_entity(command.target_id)
            )
            return {
                **base,
                "kind": "attack",
                "label": (
                    f"{attacker.definition.name} 攻击主战者"
                    if target is None
                    else f"{attacker.definition.name} 攻击 {target.definition.name}"
                ),
                "source_entity_id": command.attacker_id,
                "target_entity_id": command.target_id,
            }
        if isinstance(command, Evolve):
            card = self._board_entity(command.unit_id)
            return {
                **base,
                "kind": "evolve",
                "label": f"进化 {card.definition.name}",
                "source_entity_id": command.unit_id,
            }
        if isinstance(command, SuperEvolve):
            card = self._board_entity(command.unit_id)
            return {
                **base,
                "kind": "super_evolve",
                "label": f"超进化 {card.definition.name}",
                "source_entity_id": command.unit_id,
            }
        if isinstance(command, ActivateAmulet):
            card = self._board_entity(command.amulet_id)
            return {
                **base,
                "kind": "activate",
                "label": f"发动 {card.definition.name}",
                "source_entity_id": command.amulet_id,
            }
        if isinstance(command, Choose):
            request = env._core.state.pending_choice
            option = next(
                option
                for option in request.options
                if option.option_id == command.option_id
            )
            return {
                **base,
                "kind": "choice",
                "label": option.label,
                "option_id": option.option_id,
                "target_entity_id": option.entity_id,
                "target_player": option.leader_player_index,
            }
        return {**base, "kind": "unknown", "label": str(command)}

    def _board_entity(self, entity_id: int):
        env = self._require_env()
        for player in env._core.players:
            for card in player.board:
                if card.entity_id == entity_id:
                    return card
        raise ValueError(f"board entity not found: {entity_id}")
