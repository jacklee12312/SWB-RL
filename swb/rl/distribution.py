from __future__ import annotations

import random
from collections import Counter
from typing import TYPE_CHECKING

from swb.engine.deck import CLASS_NAMES
from swb.rl.class_schedule import (
    CLASS_SCHEDULE_VERSION,
    class_pair_for_episode,
    normalize_class_ids,
)
from swb.rl.seeding import episode_seeds
from swb.rl.versioning import stable_json_sha256

if TYPE_CHECKING:
    from swb.rl.catalog import TrainableCardCatalog


def _deck_manifest(deck) -> dict[str, object]:
    card_ids = tuple(card.card_id for card in deck)
    composition = tuple(sorted(card_ids))
    return {
        "card_ids": list(card_ids),
        "composition_sha256": stable_json_sha256(composition),
    }


def build_training_distribution_audit(
    catalog: TrainableCardCatalog,
    *,
    master_seed: int,
    episode_count: int,
    worker_count: int,
    class_ids: tuple[int, ...],
    rulebook_sha256: str | None = None,
) -> dict[str, object]:
    """Materialize the deterministic class/deck distribution before training."""
    if episode_count <= 0 or worker_count <= 0:
        raise ValueError("episode_count and worker_count must be positive")
    classes = normalize_class_ids(class_ids)
    class_pair_counts: Counter[tuple[int, int]] = Counter()
    slot_class_counts = {0: Counter(), 1: Counter()}
    learner_class_counts: Counter[int] = Counter()
    opponent_class_counts: Counter[int] = Counter()
    card_slot_counts: Counter[int] = Counter()
    card_type_slot_counts: Counter[str] = Counter()
    sampled_card_ids: set[int] = set()
    sampled_by_class = {class_id: set() for class_id in classes}
    episodes: list[dict[str, object]] = []

    for episode_id in range(episode_count):
        worker_id = episode_id % worker_count
        seeds = episode_seeds(master_seed, worker_id, episode_id)
        class_a, class_b = class_pair_for_episode(classes, episode_id)
        decks = (
            catalog.sample_deck(
                class_a,
                random.Random(seeds.deck_seed_a),
            ),
            catalog.sample_deck(
                class_b,
                random.Random(seeds.deck_seed_b),
            ),
        )
        learner_player = episode_id % 2
        class_pair_counts[(class_a, class_b)] += 1
        learner_class_counts[(class_a, class_b)[learner_player]] += 1
        opponent_class_counts[(class_a, class_b)[1 - learner_player]] += 1
        deck_manifests = []
        for player, (class_id, deck) in enumerate(
            zip((class_a, class_b), decks)
        ):
            slot_class_counts[player][class_id] += 1
            ids = {card.card_id for card in deck}
            sampled_card_ids.update(ids)
            sampled_by_class[class_id].update(ids)
            card_slot_counts.update(card.card_id for card in deck)
            card_type_slot_counts.update(card.card_type for card in deck)
            deck_manifests.append(_deck_manifest(deck))
        episodes.append({
            "episode_id": episode_id,
            "worker_id": worker_id,
            "learner_player": learner_player,
            "classes": [class_a, class_b],
            "deck_seeds": [seeds.deck_seed_a, seeds.deck_seed_b],
            "decks": deck_manifests,
        })

    exact_ids = set(catalog.exact_collectible_ids)
    sampled_exact = sampled_card_ids & exact_ids
    per_class = {}
    for class_id in classes:
        eligible = {card.card_id for card in catalog.pool(class_id=class_id)}
        sampled = sampled_by_class[class_id] & eligible
        per_class[str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "eligible_exact_count": len(eligible),
            "sampled_exact_count": len(sampled),
            "sampled_exact_rate": len(sampled) / max(1, len(eligible)),
            "unsampled_exact_card_ids": sorted(eligible - sampled),
        }

    configuration = {
        "master_seed": master_seed,
        "episode_count": episode_count,
        "worker_count": worker_count,
        "class_ids": list(classes),
        "class_names": [CLASS_NAMES[class_id] for class_id in classes],
        "schedule": "ordered_round_robin_by_episode_id",
        "class_schedule_version": CLASS_SCHEDULE_VERSION,
        "schedule_cycle_episodes": len(classes) ** 2,
        "learner_side": "episode_id_parity",
    }
    report = {
        "schema_version": 1,
        "purpose": (
            "deterministic pre-training distribution audit; not a policy-strength "
            "or gameplay-balance claim"
        ),
        "configuration": configuration,
        "versions": {
            "catalog_sha256": catalog.catalog_sha256,
            "coverage_report_sha256": catalog.coverage_report_sha256,
            "training_pool_sha256": catalog.training_pool_sha256,
            "class_schedule_version": CLASS_SCHEDULE_VERSION,
            **(
                {}
                if rulebook_sha256 is None
                else {"rulebook_sha256": rulebook_sha256}
            ),
        },
        "distribution": {
            "class_pair_counts": {
                f"{class_a}:{class_b}": count
                for (class_a, class_b), count in sorted(class_pair_counts.items())
            },
            "player_slot_class_counts": {
                str(player): {
                    str(class_id): counter[class_id]
                    for class_id in classes
                }
                for player, counter in slot_class_counts.items()
            },
            "learner_class_counts": {
                str(class_id): learner_class_counts[class_id]
                for class_id in classes
            },
            "opponent_class_counts": {
                str(class_id): opponent_class_counts[class_id]
                for class_id in classes
            },
            "card_type_slot_counts": dict(sorted(card_type_slot_counts.items())),
            "unique_exact_cards_sampled": len(sampled_exact),
            "exact_card_sampling_rate": (
                len(sampled_exact) / max(1, len(exact_ids))
            ),
            "unsampled_exact_card_ids": sorted(exact_ids - sampled_exact),
            "per_class": per_class,
            "most_frequent_card_slots": [
                {"card_id": card_id, "slots": count}
                for card_id, count in card_slot_counts.most_common(25)
            ],
        },
        "episodes": episodes,
    }
    report["audit_sha256"] = stable_json_sha256(report)
    return report
