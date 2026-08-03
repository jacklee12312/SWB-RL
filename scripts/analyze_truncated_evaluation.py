from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

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
from swb.engine.environment import ShadowverseEnv
from swb.rl.checkpoint import load_checkpoint
from swb.rl.runtime import WorkerAssetsSnapshot
from swb.rl.truncation_diagnostics import analyze_truncated_trace
from swb.rl.versioning import stable_json_sha256
from swb.simulator.service import DeterministicPPOPolicy


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_checkpoint(path: Path, expected_sha256: object, *, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    actual = _file_sha256(path)
    if expected_sha256 is not None and actual != str(expected_sha256):
        raise ValueError(
            f"{label} checkpoint hash mismatch: expected {expected_sha256}, "
            f"got {actual}"
        )
    return actual


def _strategic_state_sha256(env: ShadowverseEnv) -> str:
    full = env.core.deterministic_fingerprint()
    state = dict(full["state"])
    # This counter is diagnostic bookkeeping, not a decision-relevant game
    # state.  Keeping it would hide a BeginFusion -> CancelFusion loop because
    # every otherwise identical retry increments the counter.
    state.pop("resolution_steps", None)
    return stable_json_sha256({
        "state": state,
        "graveyard_page": env._graveyard_page,
        "decision_player": env.decision_player,
    })


def _board_card(env: ShadowverseEnv, entity_id: int):
    for player in env.core.players:
        for card in player.board:
            if card.entity_id == entity_id:
                return card
    raise ValueError(f"board entity not found: {entity_id}")


def _hand_card(env: ShadowverseEnv, player_id: int, entity_id: int):
    for card in env.core.players[player_id].hand:
        if card.entity_id == entity_id:
            return card
    raise ValueError(f"hand entity not found: {entity_id}")


def _describe_action(env: ShadowverseEnv, action: int) -> dict[str, object]:
    if action == env.GRAVEYARD_PREV_PAGE:
        return {
            "kind": "graveyard_page_prev",
            "label": "墓场上一页",
            "signature": "graveyard_page_prev",
            "page_before": env._graveyard_page,
        }
    if action == env.GRAVEYARD_NEXT_PAGE:
        return {
            "kind": "graveyard_page_next",
            "label": "墓场下一页",
            "signature": "graveyard_page_next",
            "page_before": env._graveyard_page,
        }

    command = env._decode_action(action)
    if isinstance(command, EndTurn):
        return {
            "kind": "end_turn",
            "label": "结束回合",
            "signature": "end_turn",
        }
    if isinstance(command, UseExtraPP):
        return {
            "kind": "extra_pp",
            "label": "使用 Extra PP",
            "signature": "extra_pp",
        }
    if isinstance(command, PlayCard):
        card = env.core.players[command.player_index].hand[command.hand_index]
        return {
            "kind": "play",
            "label": f"打出 {card.name} · {command.mode_id}",
            "signature": f"play:{card.card_id}:{command.mode_id}",
            "source_card_id": card.card_id,
            "source_card_name": card.name,
            "source_entity_id": card.entity_id,
            "mode_id": command.mode_id,
        }
    if isinstance(command, BeginFusion):
        card = _hand_card(env, command.player_index, command.fusion_entity_id)
        return {
            "kind": "fusion",
            "label": f"开始融合 {card.name}",
            "signature": f"fusion:{card.card_id}",
            "source_card_id": card.card_id,
            "source_card_name": card.name,
            "source_entity_id": card.entity_id,
        }
    if isinstance(command, Attack):
        attacker = _board_card(env, command.attacker_id)
        target = (
            None
            if command.target_id is None
            else _board_card(env, command.target_id)
        )
        target_signature = (
            "leader" if target is None else str(target.definition.card_id)
        )
        return {
            "kind": "attack",
            "label": (
                f"{attacker.definition.name} 攻击主战者"
                if target is None
                else f"{attacker.definition.name} 攻击 {target.definition.name}"
            ),
            "signature": (
                f"attack:{attacker.definition.card_id}:{target_signature}"
            ),
            "source_card_id": attacker.definition.card_id,
            "source_card_name": attacker.definition.name,
            "source_entity_id": attacker.entity_id,
            "target_card_id": (
                None if target is None else target.definition.card_id
            ),
            "target_entity_id": command.target_id,
        }
    if isinstance(command, (Evolve, SuperEvolve, ActivateAmulet)):
        entity_id = (
            command.unit_id
            if isinstance(command, (Evolve, SuperEvolve))
            else command.amulet_id
        )
        card = _board_card(env, entity_id)
        if isinstance(command, SuperEvolve):
            kind = "super_evolve"
            verb = "超进化"
        elif isinstance(command, Evolve):
            kind = "evolve"
            verb = "进化"
        else:
            kind = "activate"
            verb = "发动"
        return {
            "kind": kind,
            "label": f"{verb} {card.definition.name}",
            "signature": f"{kind}:{card.definition.card_id}",
            "source_card_id": card.definition.card_id,
            "source_card_name": card.definition.name,
            "source_entity_id": entity_id,
        }
    if isinstance(command, Choose):
        request = env.core.state.pending_choice
        if request is None:
            raise ValueError("choice action has no pending request")
        option = next(
            option
            for option in request.options
            if option.option_id == command.option_id
        )
        choice_kind = request.choice_kind.value
        return {
            "kind": "choice",
            "label": f"{request.prompt}: {option.label}",
            "signature": f"choice:{choice_kind}:{option.label}",
            "choice_kind": choice_kind,
            "choice_prompt": request.prompt,
            "option_id": option.option_id,
            "option_label": option.label,
            "target_entity_id": option.entity_id,
            "target_player": option.leader_player_index,
        }
    return {
        "kind": "unknown",
        "label": str(command),
        "signature": f"unknown:{type(command).__name__}",
    }


def _deck_from_manifest(manifest: dict[str, object], snapshot: WorkerAssetsSnapshot):
    deck = []
    for raw_card_id in manifest["card_ids"]:
        card_id = int(raw_card_id)
        definition = snapshot.catalog.resolve(card_id)
        if definition is None:
            raise ValueError(f"deck references unknown card {card_id}")
        deck.append(definition)
    if len(deck) != 40:
        raise ValueError(
            f"deck {manifest['composition_sha256']} contains {len(deck)} cards"
        )
    return deck


def _checkpoint_path(
    argument: Path | None,
    report_value: object,
    *,
    label: str,
) -> Path:
    path = argument if argument is not None else Path(str(report_value))
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    return path


def _source_game_summary(game: dict[str, object], index: int) -> dict[str, object]:
    fields = (
        "learner_class_id",
        "learner_class_name",
        "opponent_class_id",
        "opponent_class_name",
        "deck_index",
        "learner_player",
        "score",
        "winner",
        "turn",
        "agent_steps",
        "terminated",
        "truncated",
        "engine_seed",
        "learner_deck_sha256",
        "opponent_deck_sha256",
    )
    return {
        "source_game_index": index,
        **{field: game[field] for field in fields},
    }


def _replay_game(
    *,
    source_game: dict[str, object],
    source_game_index: int,
    deck_manifests: dict[str, dict[str, object]],
    learner_trainer,
    opponent_trainer,
    snapshot: WorkerAssetsSnapshot,
    configuration: dict[str, object],
    learner_model_id: str,
    opponent_model_id: str,
    enable_fusion_cancel_guard: bool,
) -> dict[str, object]:
    learner_manifest = deck_manifests[str(source_game["learner_deck_sha256"])]
    opponent_manifest = deck_manifests[str(source_game["opponent_deck_sha256"])]
    learner_deck = _deck_from_manifest(learner_manifest, snapshot)
    opponent_deck = _deck_from_manifest(opponent_manifest, snapshot)
    learner_player = int(source_game["learner_player"])
    decks = (
        (learner_deck, opponent_deck)
        if learner_player == 0
        else (opponent_deck, learner_deck)
    )
    classes = (
        (
            int(source_game["learner_class_id"]),
            int(source_game["opponent_class_id"]),
        )
        if learner_player == 0
        else (
            int(source_game["opponent_class_id"]),
            int(source_game["learner_class_id"]),
        )
    )
    engine_seed = int(source_game["engine_seed"])
    env = ShadowverseEnv(
        *decks,
        class_a=classes[0],
        class_b=classes[1],
        seed=engine_seed,
        rulebook=learner_trainer.assets.rulebook,
        card_resolver=learner_trainer.assets.catalog.resolve,
        observation_version=learner_trainer.config.observation_version,
        card_vocabulary=learner_trainer.assets.catalog.card_vocabulary,
        max_agent_steps=int(configuration["max_agent_steps"]),
        max_game_turns=learner_trainer.config.max_game_turns,
        training_mode=True,
        training_event_history_limit=4096,
        validate_invariants=bool(configuration["validate_invariants"]),
        match_setup=str(configuration["match_setup"]),
    )
    env.reset(seed=engine_seed)
    learner_policy = DeterministicPPOPolicy.from_trainer(
        learner_trainer,
        enable_fusion_cancel_guard=enable_fusion_cancel_guard,
    )
    opponent_policy = DeterministicPPOPolicy.from_trainer(
        opponent_trainer,
        enable_fusion_cancel_guard=enable_fusion_cancel_guard,
    )
    learner_policy.reset()
    opponent_policy.reset()
    steps: list[dict[str, object]] = []

    while not env.terminated and not env.truncated:
        player_id = env.decision_player
        policy = learner_policy if player_id == learner_player else opponent_policy
        decision = policy.decision(env, player_id)
        action = decision.action
        mask = env.action_mask()
        if action < 0 or action >= len(mask) or not mask[action]:
            raise RuntimeError(
                f"replay game {source_game_index} selected illegal action {action}"
            )
        description = _describe_action(env, action)
        before_state = _strategic_state_sha256(env)
        turn_before = env.turn
        page_before = env._graveyard_page
        result = env.step(action)
        after_state = _strategic_state_sha256(env)
        actor_model = (
            learner_model_id if player_id == learner_player else opponent_model_id
        )
        selected_probability = float(decision.probabilities[action])
        steps.append({
            "step_index": len(steps),
            "turn_before": turn_before,
            "turn_after": env.turn,
            "player_id": player_id,
            "actor_model": actor_model,
            "action_id": action,
            "action_kind": description["kind"],
            "action_label": description["label"],
            "action_signature": description["signature"],
            "action": description,
            "selected_probability": selected_probability,
            "value": decision.value,
            "legal_action_count": sum(bool(value) for value in mask),
            "graveyard_page_before": page_before,
            "graveyard_page_after": env._graveyard_page,
            "before_state_sha256": before_state,
            "after_state_sha256": after_state,
            "strategic_state_changed": before_state != after_state,
            "terminated": result.terminated,
            "truncated": result.truncated,
        })

    expected = {
        "agent_steps": int(source_game["agent_steps"]),
        "turn": int(source_game["turn"]),
        "winner": source_game["winner"],
        "terminated": bool(source_game["terminated"]),
        "truncated": bool(source_game["truncated"]),
    }
    actual = {
        "agent_steps": env.agent_steps,
        "turn": env.turn,
        "winner": env.winner,
        "terminated": env.terminated,
        "truncated": env.truncated,
    }
    analysis = analyze_truncated_trace(steps)
    dominant_cycle = analysis["dominant_cycle"]
    if dominant_cycle is not None:
        start, stop = dominant_cycle["sample_step_range"]
        actor_counts = Counter(
            str(step["actor_model"]) for step in steps[int(start) : int(stop)]
        )
        dominant_cycle["actor_models"] = dict(actor_counts.most_common())

    return {
        "source": _source_game_summary(source_game, source_game_index),
        "replay": actual,
        "exact_replay_match": actual == expected,
        "analysis": analysis,
        "steps": steps,
    }


def _aggregate(replays: list[dict[str, object]]) -> dict[str, object]:
    classifications = Counter(
        str(replay["analysis"]["classification"]) for replay in replays
    )
    action_kinds: Counter[str] = Counter()
    tail_action_kinds: Counter[str] = Counter()
    actor_models: Counter[str] = Counter()
    pattern_members: dict[str, list[dict[str, object]]] = {}
    cycle_periods: Counter[int] = Counter()
    loop_start_steps: list[int] = []
    loop_remaining_budget_fractions: list[float] = []
    loop_entry_cards: dict[tuple[int, str], Counter[str]] = {}
    loop_probability_samples: dict[tuple[str, str], list[float]] = {}
    portal_involved = 0
    immediate_fusion_retries = 0
    for replay in replays:
        source = replay["source"]
        if source["learner_class_id"] == 7 or source["opponent_class_id"] == 7:
            portal_involved += 1
        steps = replay["steps"]
        immediate_fusion_retries += sum(
            1
            for current, following in zip(steps, steps[1:])
            if (
                current["action"].get("option_id") == "fusion:cancel"
                and following["action_kind"] == "fusion"
            )
        )
        action_kinds.update(str(step["action_kind"]) for step in steps)
        tail_action_kinds.update(
            str(step["action_kind"]) for step in steps[-min(256, len(steps)) :]
        )
        for step in steps[-min(256, len(steps)) :]:
            key = (str(step["actor_model"]), str(step["action_kind"]))
            loop_probability_samples.setdefault(key, []).append(
                float(step["selected_probability"])
            )
        dominant = replay["analysis"]["dominant_cycle"]
        if dominant is None:
            pattern_key = "<no exact strategic-state cycle>"
        else:
            pattern_key = " -> ".join(
                dominant.get(
                    "canonical_action_cycle",
                    dominant["action_sequence"],
                )
            )
            cycle_actor_counts = dominant.get("actor_models", {})
            dominant_actor = "<unknown>"
            if cycle_actor_counts:
                dominant_actor = max(
                    cycle_actor_counts,
                    key=lambda model_id: cycle_actor_counts[model_id],
                )
                actor_models[dominant_actor] += 1
            cycle_periods[int(dominant["period"])] += 1
            start, stop = dominant["sample_step_range"]
            loop_start_steps.append(int(start))
            loop_remaining_budget_fractions.append(
                (len(steps) - int(start)) / len(steps)
            )
            sample = steps[int(start) : int(stop)]
            fusion_step = next(
                (
                    step
                    for step in sample
                    if step["action_kind"] == "fusion"
                ),
                None,
            )
            if fusion_step is not None:
                action = fusion_step["action"]
                card_key = (
                    int(action["source_card_id"]),
                    str(action["source_card_name"]),
                )
                card_actors = loop_entry_cards.setdefault(card_key, Counter())
                card_actors[dominant_actor] += 1
        pattern_members.setdefault(pattern_key, []).append(replay)

    clusters = []
    for pattern_key, members in sorted(
        pattern_members.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        example = members[0]
        dominant = example["analysis"]["dominant_cycle"]
        if dominant is None:
            display_pattern = "<no exact strategic-state cycle>"
        else:
            start, stop = dominant["sample_step_range"]
            display_pattern = " -> ".join(
                str(step["action_signature"])
                for step in example["steps"][int(start) : int(stop)]
            )
        cluster_actors: Counter[str] = Counter()
        for member in members:
            candidate = member["analysis"]["dominant_cycle"]
            if candidate is None or not candidate.get("actor_models"):
                continue
            counts = candidate["actor_models"]
            actor = max(counts, key=lambda model_id: counts[model_id])
            cluster_actors[actor] += 1
        clusters.append({
            "pattern_key": pattern_key,
            "pattern": display_pattern,
            "games": len(members),
            "source_game_indices": [
                member["source"]["source_game_index"]
                for member in members
            ],
            "source_game_index": example["source"]["source_game_index"],
            "classification": example["analysis"]["classification"],
            "action_labels": (
                [] if dominant is None else dominant["action_labels"]
            ),
            "actor_model_games": dict(cluster_actors.most_common()),
        })
    return {
        "source_truncated_games": len(replays),
        "exact_replay_matches": sum(
            bool(replay["exact_replay_match"]) for replay in replays
        ),
        "replay_terminated_games": sum(
            bool(replay["replay"]["terminated"]) for replay in replays
        ),
        "replay_truncated_games": sum(
            bool(replay["replay"]["truncated"]) for replay in replays
        ),
        "immediate_fusion_retries_after_cancel": immediate_fusion_retries,
        "portal_involved_games": portal_involved,
        "total_replayed_steps": sum(len(replay["steps"]) for replay in replays),
        "maximum_replayed_steps": max(
            (len(replay["steps"]) for replay in replays),
            default=0,
        ),
        "classification_counts": dict(classifications.most_common()),
        "dominant_cycle_actor_model_games": dict(actor_models.most_common()),
        "cycle_period_games": {
            str(period): count
            for period, count in sorted(cycle_periods.items())
        },
        "loop_start_step": (
            None
            if not loop_start_steps
            else {
                "minimum": min(loop_start_steps),
                "median": median(loop_start_steps),
                "maximum": max(loop_start_steps),
                "mean_remaining_budget_fraction": sum(
                    loop_remaining_budget_fractions
                ) / len(loop_remaining_budget_fractions),
            }
        ),
        "loop_entry_card_games": [
            {
                "card_id": card_id,
                "card_name": card_name,
                "games": sum(actors.values()),
                "actor_model_games": dict(actors.most_common()),
            }
            for (card_id, card_name), actors in sorted(
                loop_entry_cards.items(),
                key=lambda item: (-sum(item[1].values()), item[0]),
            )
        ],
        "tail_selected_probability_by_actor_kind": [
            {
                "actor_model": actor_model,
                "action_kind": action_kind,
                "actions": len(values),
                "mean": sum(values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for (actor_model, action_kind), values in sorted(
                loop_probability_samples.items()
            )
            if action_kind in {"fusion", "choice"}
        ],
        "action_kind_counts": dict(action_kinds.most_common()),
        "tail_action_kind_counts": dict(tail_action_kinds.most_common()),
        "dominant_cycle_clusters": clusters,
    }


def _markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# PPO 截断对局动作循环诊断",
        "",
        "## 摘要",
        "",
        f"- 来源截断局：{summary['source_truncated_games']}",
        f"- 精确重放一致：{summary['exact_replay_matches']}",
        f"- 重放正常终局：{summary['replay_terminated_games']}",
        f"- 重放仍截断：{summary['replay_truncated_games']}",
        "- 取消融合后立即重开："
        f"{summary['immediate_fusion_retries_after_cancel']}",
        f"- 涉及超越者：{summary['portal_involved_games']}",
        f"- 重放 agent steps：{summary['total_replayed_steps']}",
        f"- 最长单局 agent steps：{summary['maximum_replayed_steps']}",
        "- 分类：`"
        + json.dumps(summary["classification_counts"], ensure_ascii=False)
        + "`",
        f"- 两步/三步状态环：{summary['cycle_period_games']}",
        "- 循环入口牌：`"
        + json.dumps(summary["loop_entry_card_games"], ensure_ascii=False)
        + "`",
        "- 主循环执行模型：`"
        + json.dumps(
            summary["dominant_cycle_actor_model_games"],
            ensure_ascii=False,
        )
        + "`",
        "",
        "## 主循环聚类",
        "",
        "| 局数 | 分类 | 动作循环 | 示例源局 | 可读动作 |",
        "|---:|---|---|---:|---|",
    ]
    for cluster in summary["dominant_cycle_clusters"]:
        pattern = str(cluster["pattern"]).replace("|", "\\|")
        labels = " → ".join(cluster["action_labels"]).replace("|", "\\|")
        lines.append(
            f"| {cluster['games']} | {cluster['classification']} | "
            f"`{pattern}` | {cluster['source_game_index']} | {labels} |"
        )
    lines.extend((
        "",
        "## 逐局结果",
        "",
        "| 源局 | 对阵 | 截断回合 | 分类 | 最大同状态访问 | 尾部翻页占比 | 主循环执行模型 |",
        "|---:|---|---:|---|---:|---:|---|",
    ))
    for replay in report["games"]:
        source = replay["source"]
        analysis = replay["analysis"]
        dominant = analysis["dominant_cycle"]
        actors = {} if dominant is None else dominant.get("actor_models", {})
        lines.append(
            f"| {source['source_game_index']} | "
            f"{source['learner_class_name']} vs {source['opponent_class_name']} | "
            f"{source['turn']} | {analysis['classification']} | "
            f"{analysis['max_state_visits']} | "
            f"{analysis['tail_page_fraction']:.2%} | "
            f"{json.dumps(actors, ensure_ascii=False)} |"
        )
    lines.extend((
        "",
        "完整逐步动作、状态哈希、选择概率和卡牌信息见同名 JSON。",
        "",
    ))
    return "\n".join(lines)


def build_report(
    *,
    source_report: Path,
    database: Path,
    learner_checkpoint: Path | None,
    opponent_checkpoint: Path | None,
    device: str,
    enable_fusion_cancel_guard: bool = False,
) -> dict[str, object]:
    source = _read_json(source_report)
    configuration = source["configuration"]
    if configuration["opponent_kind"] != "historical":
        raise ValueError("source evaluation must use a historical checkpoint opponent")
    learner_info = source["checkpoint"]
    learner_path = _checkpoint_path(
        learner_checkpoint,
        learner_info["path"],
        label="learner",
    )
    opponent_path = _checkpoint_path(
        opponent_checkpoint,
        configuration["opponent_checkpoint"],
        label="opponent",
    )
    learner_sha256 = _verify_checkpoint(
        learner_path, learner_info.get("sha256"), label="learner"
    )
    opponent_sha256 = _verify_checkpoint(
        opponent_path,
        configuration.get("opponent_checkpoint_sha256"),
        label="opponent",
    )
    snapshot = WorkerAssetsSnapshot.build(CardRepository(database))
    learner_trainer = load_checkpoint(
        learner_path, snapshot, device=device, restore_rng_state=False
    )
    opponent_trainer = load_checkpoint(
        opponent_path, snapshot, device=device, restore_rng_state=False
    )
    if (
        learner_trainer.config.observation_version
        != opponent_trainer.config.observation_version
    ):
        raise ValueError("checkpoint observation versions do not match")
    deck_manifests = {
        str(manifest["composition_sha256"]): manifest
        for manifest in source["decks"]
    }
    learner_model_id = learner_path.parent.name
    opponent_model_id = opponent_path.parent.name
    truncated_games = [
        (index, game)
        for index, game in enumerate(source["games"])
        if bool(game["truncated"])
    ]
    try:
        replays = [
            _replay_game(
                source_game=game,
                source_game_index=index,
                deck_manifests=deck_manifests,
                learner_trainer=learner_trainer,
                opponent_trainer=opponent_trainer,
                snapshot=snapshot,
                configuration=configuration,
                learner_model_id=learner_model_id,
                opponent_model_id=opponent_model_id,
                enable_fusion_cancel_guard=enable_fusion_cancel_guard,
            )
            for index, game in truncated_games
        ]
    finally:
        learner_trainer.close()
        opponent_trainer.close()
    summary = _aggregate(replays)
    if (
        not enable_fusion_cancel_guard
        and summary["exact_replay_matches"] != len(replays)
    ):
        mismatches = [
            replay["source"]["source_game_index"]
            for replay in replays
            if not replay["exact_replay_match"]
        ]
        raise AssertionError(f"truncated replay mismatches: {mismatches}")
    return {
        "schema_version": 1,
        "purpose": (
            "exact deterministic replay and repeated-action/strategic-state "
            "classification for truncated PPO evaluation games"
        ),
        "source_report": {
            "path": str(source_report.resolve()),
            "sha256": _file_sha256(source_report),
            "evaluation_suite_sha256": source.get("evaluation_suite_sha256"),
        },
        "checkpoints": {
            "learner": {"path": str(learner_path), "sha256": learner_sha256},
            "opponent": {"path": str(opponent_path), "sha256": opponent_sha256},
        },
        "configuration": configuration,
        "policy_action_guard": (
            "fusion-cancel-retry-v1"
            if enable_fusion_cancel_guard
            else None
        ),
        "summary": summary,
        "games": replays,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay and classify truncated games from one PPO evaluation"
    )
    parser.add_argument("source_report", type=Path)
    parser.add_argument("--database", type=Path, default=Path("data/cards.sqlite3"))
    parser.add_argument("--learner-checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--enable-fusion-cancel-guard",
        action="store_true",
        help=(
            "replay with the production fusion-cancel retry guard; endpoint "
            "equality with the historical source is then not required"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/ppo_truncation_diagnostics.json"),
    )
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--reuse-existing-traces",
        action="store_true",
        help="recompute summary/Markdown from an existing --output JSON",
    )
    args = parser.parse_args()
    if args.reuse_existing_traces:
        report = _read_json(args.output)
        report["summary"] = _aggregate(report["games"])
    else:
        report = build_report(
            source_report=args.source_report,
            database=args.database,
            learner_checkpoint=args.learner_checkpoint,
            opponent_checkpoint=args.opponent_checkpoint,
            device=args.device,
            enable_fusion_cancel_guard=args.enable_fusion_cancel_guard,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = (
        args.markdown
        if args.markdown is not None
        else args.output.with_suffix(".md")
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "markdown": str(markdown_path),
        "summary": report["summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
