from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.report_ppo_league_baseline import render_json
from scripts.report_ppo_league_pfsp import (
    build_generation_manifest,
    build_pfsp_sampler_scan,
)
from scripts.report_ppo_league_training_payoff import (
    build_training_payoff_plan,
    build_training_payoff_snapshot,
)
from swb.rl.opponents import OpponentEntry, OpponentPool
from swb.rl.pfsp import (
    PayoffEstimate,
    compute_pfsp_distribution,
    forgotten_opponent_queue,
    load_training_payoff_snapshot,
)


class PFSPSamplingTests(unittest.TestCase):
    @staticmethod
    def estimate(
        index: int,
        score: float | None,
        *,
        with_interval: bool = True,
        checkpoint_hash: str | None = None,
    ) -> PayoffEstimate:
        interval = None
        games = 0 if score is None else 196
        if score is not None and with_interval:
            interval = (max(0.0, score - 0.05), min(1.0, score + 0.05))
        return PayoffEstimate(
            opponent_id=f"opponent_{index}",
            checkpoint_sha256=(
                checkpoint_hash
                if checkpoint_hash is not None
                else f"{index + 1:064x}"
            ),
            games=games,
            score_rate=score,
            confidence_interval_95=interval,
        )

    def test_all_win_loss_and_tie_inputs_have_safe_distributions(self) -> None:
        for score in (0.0, 0.5, 1.0):
            estimates = tuple(self.estimate(index, score) for index in range(4))
            for sampler in ("uniform", "variance", "hard"):
                with self.subTest(score=score, sampler=sampler):
                    result = compute_pfsp_distribution(
                        estimates,
                        sampler=sampler,
                    )
                    self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)
                    self.assertTrue(all(
                        probability == 0.25
                        for probability in result.probabilities.values()
                    ))

    def test_variance_and_hard_match_preregistered_formulas(self) -> None:
        estimates = (
            self.estimate(0, 0.0),
            self.estimate(1, 0.5),
            self.estimate(2, 0.9),
            self.estimate(3, 1.0),
        )
        variance = compute_pfsp_distribution(estimates, sampler="variance")
        hard = compute_pfsp_distribution(estimates, sampler="hard")
        for opponent_id, expected in {
            "opponent_0": 0.0,
            "opponent_1": 0.25,
            "opponent_2": 0.09,
            "opponent_3": 0.0,
        }.items():
            self.assertAlmostEqual(variance.raw_weights[opponent_id], expected)
        for opponent_id, expected in {
            "opponent_0": 1.0,
            "opponent_1": 0.5,
            "opponent_2": 0.1,
            "opponent_3": 0.0,
        }.items():
            self.assertAlmostEqual(hard.raw_weights[opponent_id], expected)
        for distribution in (variance, hard):
            self.assertAlmostEqual(sum(distribution.probabilities.values()), 1.0)
            self.assertGreaterEqual(min(distribution.probabilities.values()), 0.02)
            self.assertLessEqual(max(distribution.probabilities.values()), 0.35)

    def test_missing_ci_is_explicit_and_receives_only_floor_when_possible(self) -> None:
        estimates = (
            self.estimate(0, 0.2),
            self.estimate(1, 0.5, with_interval=False),
            self.estimate(2, 0.8),
            self.estimate(3, 0.9),
        )
        result = compute_pfsp_distribution(estimates, sampler="hard")
        self.assertEqual(result.unreliable_opponent_ids, ("opponent_1",))
        self.assertAlmostEqual(result.probabilities["opponent_1"], 0.02)

    def test_duplicate_models_and_infeasible_floor_are_rejected(self) -> None:
        duplicate_hash = "a" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint content"):
            compute_pfsp_distribution(
                (
                    self.estimate(0, 0.2, checkpoint_hash=duplicate_hash),
                    self.estimate(1, 0.8, checkpoint_hash=duplicate_hash),
                ),
                sampler="hard",
            )
        with self.assertRaisesRegex(ValueError, "epsilon floor is infeasible"):
            compute_pfsp_distribution(
                tuple(self.estimate(index, 0.5) for index in range(51)),
                sampler="variance",
            )

    def test_single_or_too_few_candidates_record_cap_exception(self) -> None:
        single = compute_pfsp_distribution(
            (self.estimate(0, 0.1),),
            sampler="hard",
        )
        self.assertEqual(single.probabilities, {"opponent_0": 1.0})
        self.assertTrue(single.cap_exception_required)
        pair = compute_pfsp_distribution(
            (self.estimate(0, 0.1), self.estimate(1, 0.9)),
            sampler="hard",
        )
        self.assertTrue(pair.cap_exception_required)
        self.assertAlmostEqual(sum(pair.probabilities.values()), 1.0)

    def test_fixed_seed_selection_is_reproducible(self) -> None:
        estimates = tuple(self.estimate(index, index / 5) for index in range(5))
        distribution = compute_pfsp_distribution(estimates, sampler="hard")
        entries = tuple(
            OpponentEntry(
                opponent_id=estimate.opponent_id,
                kind="external",
                weight=distribution.probabilities[estimate.opponent_id],
                checkpoint_path=f"{estimate.opponent_id}.pt",
                checkpoint_sha256=estimate.checkpoint_sha256,
                policy_seed=index,
                training_steps=1000,
                generation=1,
                role="candidate",
                rules_version="rules-v1",
                policy_architecture="policy-v1",
                versions_sha256=f"{index + 20:064x}",
            )
            for index, estimate in enumerate(estimates)
        )

        def pool() -> OpponentPool:
            return OpponentPool(
                1234,
                current_weight=0.0,
                external_entries=entries,
                external_manifest_path="generation_001.json",
                external_manifest_sha256="f" * 64,
                external_generation=1,
            )

        first = pool()
        second = pool()
        expected = [
            first.select(episode_id=index, learner_player=index % 2).opponent_id
            for index in range(1000)
        ]
        actual = [
            second.select(episode_id=index, learner_player=index % 2).opponent_id
            for index in range(1000)
        ]
        self.assertEqual(actual, expected)

    def test_forgotten_queue_uses_registered_thresholds(self) -> None:
        previous = (
            self.estimate(0, 0.75),
            self.estimate(1, 0.69),
            self.estimate(2, 0.80),
        )
        current = (
            self.estimate(0, 0.39),
            self.estimate(1, 0.10),
            self.estimate(2, 0.40),
        )
        queue = forgotten_opponent_queue(previous, current)
        self.assertEqual([row["opponent_id"] for row in queue], ["opponent_0"])
        self.assertAlmostEqual(queue[0]["drop"], 0.36)

    def _write_snapshot_fixture(
        self,
        directory: Path,
    ) -> tuple[Path, Path, Path]:
        entries = []
        for index in range(3):
            entries.append({
                "opponent_id": f"opponent_{index}",
                "checkpoint_sha256": f"{index + 1:064x}",
                "training_eligible": True,
                "sampling_weight": 1.0,
                "generation": 0,
                "checkpoint_path": f"opponent_{index}.pt",
            })
        entries.append({
            "opponent_id": "anchor_0",
            "checkpoint_sha256": "f" * 64,
            "training_eligible": False,
            "sampling_weight": 0.0,
            "generation": 0,
            "checkpoint_path": "anchor_0.pt",
        })
        manifest = directory / "generation.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "report_kind": "ppo_league_generation_manifest",
            "immutable": True,
            "path_base": "repository_root",
            "generation": 0,
            "selection_mode": "uniform",
            "contract": {
                "test": "contract",
                "experiment_versions": {"test": "versions"},
            },
            "deduplication": {
                "key": "checkpoint_sha256",
                "duplicate_groups": [],
                "unique_model_count": 4,
            },
            "entries": entries,
        }), encoding="utf-8")
        snapshot = directory / "payoff.json"
        snapshot.write_text(json.dumps({
            "schema_version": 1,
            "report_kind": "ppo_league_training_payoff_snapshot",
            "immutable": True,
            "source_generation": 0,
            "target_generation": 1,
            "source_generation_manifest": {
                "path": manifest.resolve().relative_to(
                    Path.cwd().resolve()
                ).as_posix(),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
            "evaluator": {
                "data_partition": "pfsp_tuning",
                "payoff_update_boundary": "generation_end",
                "match_master_seeds": [101, 102],
            },
            "focal_policy_ids": ["learner_a", "learner_b"],
            "aggregation": "equal_weight_mean_across_focal_policies",
            "opponents": [
                {
                    "opponent_id": f"opponent_{index}",
                    "checkpoint_sha256": f"{index + 1:064x}",
                    "games": 196,
                    "score_rate": 0.2 + index * 0.3,
                    "confidence_interval_95": (
                        None if index == 1 else [0.15 + index * 0.3, 0.25 + index * 0.3]
                    ),
                }
                for index in range(3)
            ],
        }), encoding="utf-8")
        protocol = directory / "evaluation_protocol.json"
        protocol.write_text(json.dumps({
            "seed_partitions": {
                "pfsp_tuning_match_master_seeds": [101, 102, 103],
                "final_evaluation_match_master_seeds": [201, 202],
            },
        }), encoding="utf-8")
        return manifest, snapshot, protocol

    def test_training_snapshot_enforces_seed_and_generation_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory_name:
            directory = Path(directory_name)
            manifest, snapshot, _ = self._write_snapshot_fixture(directory)
            loaded = load_training_payoff_snapshot(
                snapshot,
                generation_manifest_path=manifest,
                allowed_tuning_seeds=(101, 102, 103),
                forbidden_final_seeds=(201, 202),
                repository_root=Path.cwd(),
            )
            self.assertEqual(loaded.source_generation, 0)
            self.assertEqual(loaded.target_generation, 1)
            self.assertEqual(len(loaded.estimates), 3)
            self.assertEqual(
                sum(estimate.reliable for estimate in loaded.estimates),
                2,
            )

            payload = json.loads(snapshot.read_text(encoding="utf-8"))
            payload["evaluator"]["match_master_seeds"] = [101, 201]
            snapshot.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "final evaluation seeds"):
                load_training_payoff_snapshot(
                    snapshot,
                    generation_manifest_path=manifest,
                    allowed_tuning_seeds=(101, 102, 103),
                    forbidden_final_seeds=(201, 202),
                    repository_root=Path.cwd(),
                )

    def test_sampler_scan_and_next_generation_are_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory_name:
            directory = Path(directory_name)
            manifest, snapshot, protocol = self._write_snapshot_fixture(directory)
            scan = build_pfsp_sampler_scan(
                payoff_snapshot_path=snapshot,
                generation_manifest_path=manifest,
                evaluation_protocol_path=protocol,
            )
            next_generation = build_generation_manifest(
                sampler="variance",
                payoff_snapshot_path=snapshot,
                generation_manifest_path=manifest,
                evaluation_protocol_path=protocol,
            )
            self.assertEqual(
                render_json(scan),
                render_json(build_pfsp_sampler_scan(
                    payoff_snapshot_path=snapshot,
                    generation_manifest_path=manifest,
                    evaluation_protocol_path=protocol,
                )),
            )
            self.assertEqual(next_generation["generation"], 1)
            self.assertEqual(next_generation["selection_mode"], "variance")
            self.assertEqual(
                next_generation["summary"]["source_model_generation_counts"],
                {"0": 4},
            )
            trainable_weights = [
                float(entry["sampling_weight"])
                for entry in next_generation["entries"]
                if entry["training_eligible"]
            ]
            anchor_weights = [
                float(entry["sampling_weight"])
                for entry in next_generation["entries"]
                if not entry["training_eligible"]
            ]
            self.assertAlmostEqual(sum(trainable_weights), 1.0)
            self.assertEqual(anchor_weights, [0.0])
            self.assertEqual(
                next_generation["selection_audit"][
                    "class_pair_position_coverage"
                ],
                98,
            )

    def test_training_payoff_report_aggregates_only_tuning_seed_data(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory_name:
            directory = Path(directory_name)
            manifest, _, protocol = self._write_snapshot_fixture(directory)
            source = directory / "reports"
            source.mkdir()
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            entries = {
                entry["opponent_id"]: entry
                for entry in manifest_payload["entries"]
            }
            scores_by_opponent = {
                "opponent_0": [1.0] * 98 + [0.0] * 98,
                "opponent_1": [1.0] * 147 + [0.0] * 49,
                "opponent_2": [1.0] * 49 + [0.0] * 147,
            }
            for opponent_id, scores in scores_by_opponent.items():
                report = {
                    "checkpoint": {
                        "sha256": entries["opponent_0"]["checkpoint_sha256"],
                    },
                    "configuration": {
                        "opponent_checkpoint_sha256": entries[opponent_id][
                            "checkpoint_sha256"
                        ],
                        "master_seed": 101,
                        "seed_count": 2,
                        "max_agent_steps": 512,
                        "full_matchup_matrix": True,
                        "match_setup": "official",
                        "opponent_kind": "historical",
                        "class_ids": list(range(1, 8)),
                    },
                    "versions": {"test": "versions"},
                    "metrics": {
                        "games": 196,
                        "terminated": 196,
                        "truncated": 0,
                        "illegal_actions": 0,
                        "action_mask_mismatches": 0,
                    },
                    "games": [{"score": score} for score in scores],
                }
                path = source / f"opponent_0__vs__{opponent_id}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
            snapshot = build_training_payoff_snapshot(
                generation_manifest_path=manifest,
                evaluation_protocol_path=protocol,
                source_directory=source,
                focal_policy_ids=("opponent_0",),
                match_master_seeds=(101,),
                seed_count=2,
                games_per_pair=196,
            )
            self.assertEqual(snapshot["audit"]["total_games"], 588)
            self.assertEqual(snapshot["audit"]["truncated"], 0)
            self.assertEqual(
                [row["score_rate"] for row in snapshot["opponents"]],
                [0.5, 0.75, 0.25],
            )
            self.assertEqual(
                render_json(snapshot),
                render_json(build_training_payoff_snapshot(
                    generation_manifest_path=manifest,
                    evaluation_protocol_path=protocol,
                    source_directory=source,
                    focal_policy_ids=("opponent_0",),
                    match_master_seeds=(101,),
                    seed_count=2,
                    games_per_pair=196,
                )),
            )
            plan = build_training_payoff_plan(
                generation_manifest_path=manifest,
                evaluation_protocol_path=protocol,
                focal_policy_ids=("opponent_0",),
                match_master_seeds=(101,),
                seed_count=2,
                games_per_pair=196,
            )
            self.assertEqual(plan["data_partition"], "pfsp_tuning")
            self.assertEqual(plan["expected"]["pair_count"], 3)
            self.assertEqual(plan["expected"]["game_count"], 588)
            self.assertEqual(
                plan["evaluation"]["final_evaluation_match_master_seeds_used"],
                [],
            )


if __name__ == "__main__":
    unittest.main()
