from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from swb.rl.opponents import (
    OpponentEntry,
    OpponentEpisodeScheduler,
    OpponentPool,
    load_external_opponent_manifest,
)


class OpponentPoolTests(unittest.TestCase):
    @staticmethod
    def _contract() -> dict[str, object]:
        digest = "a" * 64
        return {
            "experiment_versions": {},
            "policy_architecture": "test_policy",
            "model_structure_sha256": digest,
            "catalog_sha256": digest,
            "rulebook_sha256": digest,
            "observation_schema_sha256": digest,
            "action_layout_sha256": digest,
        }

    def _write_external_manifest(self, directory: Path) -> Path:
        entries = []
        for index, (name, eligible) in enumerate(
            (("candidate.pt", True), ("anchor.pt", False))
        ):
            checkpoint = directory / name
            checkpoint.write_bytes(f"checkpoint-{index}".encode("ascii"))
            entries.append({
                "opponent_id": f"opponent_{index}",
                "checkpoint_path": name,
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "policy_seed": 100 + index,
                "training_steps": 1000 + index,
                "generation": 0,
                "role": "candidate" if eligible else "anchor_only",
                "rules_version": "rules-v1",
                "policy_architecture": "test_policy",
                "versions_sha256": "b" * 64,
                "training_eligible": eligible,
                "sampling_weight": 1.0 if eligible else 0.0,
            })
        manifest = directory / "generation.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "report_kind": "ppo_league_generation_manifest",
            "immutable": True,
            "path_base": "repository_root",
            "generation": 0,
            "selection_mode": "uniform",
            "contract": self._contract(),
            "entries": entries,
        }), encoding="utf-8")
        return manifest

    def make_pool(self) -> OpponentPool:
        return OpponentPool(
            42,
            current_weight=1.0,
            random_weight=1.0,
            fixed_weight=1.0,
            historical_weight=1.0,
            max_history=2,
            snapshot_interval_steps=100,
        )

    def test_selection_is_reproducible_by_episode_and_side(self) -> None:
        first = self.make_pool()
        second = self.make_pool()
        selected_first = [
            first.select(episode_id=episode, learner_player=episode % 2).opponent_id
            for episode in range(20)
        ]
        selected_second = [
            second.select(episode_id=episode, learner_player=episode % 2).opponent_id
            for episode in range(20)
        ]
        self.assertEqual(selected_first, selected_second)
        self.assertGreater(len(set(selected_first)), 1)

    def test_snapshot_retention_is_bounded_and_monotonic(self) -> None:
        pool = self.make_pool()
        self.assertTrue(pool.snapshot_due(100))
        pool.register_snapshot("one.pt", agent_steps=100)
        pool.register_snapshot("two.pt", agent_steps=200)
        pool.register_snapshot("three.pt", agent_steps=300)
        history = [entry for entry in pool.entries if entry.kind == "historical"]
        self.assertEqual(
            [entry.opponent_id for entry in history],
            ["historical_000000000200", "historical_000000000300"],
        )
        with self.assertRaisesRegex(ValueError, "increase monotonically"):
            pool.register_snapshot("stale.pt", agent_steps=300)

    def test_zero_weight_history_does_not_write_unused_snapshots(self) -> None:
        pool = OpponentPool(
            42,
            current_weight=1.0,
            historical_weight=0.0,
            snapshot_interval_steps=100,
        )
        self.assertFalse(pool.snapshot_due(100))
        self.assertFalse(pool.snapshot_due(1_000_000))

    def test_state_round_trip_preserves_selection_progress(self) -> None:
        pool = self.make_pool()
        pool.register_snapshot("one.pt", agent_steps=100)
        pool.select(episode_id=1, learner_player=1)
        restored = OpponentPool.from_state_dict(pool.state_dict())
        self.assertEqual(restored.state_dict(), pool.state_dict())
        self.assertEqual(sum(restored.selection_counts.values()), 1)

    def test_resume_preserves_future_selection_sequence_and_counts(self) -> None:
        uninterrupted = self.make_pool()
        uninterrupted.register_snapshot("one.pt", agent_steps=100)
        for episode_id in range(37):
            uninterrupted.select(
                episode_id=episode_id,
                learner_player=episode_id % 2,
            )
        resumed = OpponentPool.from_state_dict(uninterrupted.state_dict())
        expected = [
            uninterrupted.select(
                episode_id=episode_id,
                learner_player=episode_id % 2,
            ).opponent_id
            for episode_id in range(37, 137)
        ]
        actual = [
            resumed.select(
                episode_id=episode_id,
                learner_player=episode_id % 2,
            ).opponent_id
            for episode_id in range(37, 137)
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(resumed.selection_count, uninterrupted.selection_count)
        self.assertEqual(resumed.selection_counts, uninterrupted.selection_counts)
        self.assertEqual(
            resumed.selection_counts_by_opponent,
            uninterrupted.selection_counts_by_opponent,
        )

    def test_external_manifest_keeps_anchors_auditable_but_unselectable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manifest_path = self._write_external_manifest(directory)
            manifest = load_external_opponent_manifest(
                manifest_path,
                external_weight=2.0,
                repository_root=directory,
            )
            self.assertEqual(len(manifest.entries), 2)
            self.assertEqual(len(manifest.trainable_entries), 1)
            self.assertEqual(len(manifest.reference_entries), 1)
            self.assertEqual(manifest.trainable_entries[0].weight, 2.0)
            self.assertEqual(manifest.reference_entries[0].weight, 0.0)
            pool = OpponentPool(
                42,
                current_weight=0.0,
                external_entries=manifest.entries,
                external_manifest_path=manifest.path,
                external_manifest_sha256=manifest.file_sha256,
                external_generation=manifest.generation,
            )
            selected = {
                pool.select(
                    episode_id=episode_id,
                    learner_player=episode_id % 2,
                ).opponent_id
                for episode_id in range(100)
            }
            self.assertEqual(selected, {"opponent_0"})
            restored = OpponentPool.from_state_dict(
                pool.state_dict(),
                expected_external_manifest_sha256=manifest.file_sha256,
                expected_external_entries=manifest.entries,
            )
            self.assertEqual(restored.state_dict(), pool.state_dict())

    def test_external_manifest_rejects_hash_and_resume_entry_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manifest_path = self._write_external_manifest(directory)
            manifest = load_external_opponent_manifest(
                manifest_path,
                external_weight=1.0,
                repository_root=directory,
            )
            (directory / "candidate.pt").write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_external_opponent_manifest(
                    manifest_path,
                    external_weight=1.0,
                    repository_root=directory,
                )
            pool = OpponentPool(
                42,
                current_weight=0.0,
                external_entries=manifest.entries,
                external_manifest_path=manifest.path,
                external_manifest_sha256=manifest.file_sha256,
                external_generation=manifest.generation,
            )
            changed = list(manifest.entries)
            changed[0] = OpponentEntry(
                **{
                    **changed[0].__dict__,
                    "weight": changed[0].weight + 1.0,
                }
            )
            with self.assertRaisesRegex(ValueError, "entries changed"):
                OpponentPool.from_state_dict(
                    pool.state_dict(),
                    expected_external_manifest_sha256=manifest.file_sha256,
                    expected_external_entries=changed,
                )

    def test_external_manifest_rejects_malformed_metadata_and_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manifest_path = self._write_external_manifest(directory)
            original = json.loads(manifest_path.read_text(encoding="utf-8"))

            malformed = json.loads(json.dumps(original))
            malformed["entries"][0]["training_steps"] = "1000"
            manifest_path.write_text(
                json.dumps(malformed),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "training_steps"):
                load_external_opponent_manifest(
                    manifest_path,
                    external_weight=1.0,
                    repository_root=directory,
                )

            malformed = json.loads(json.dumps(original))
            malformed["entries"][0]["policy_architecture"] = "other"
            manifest_path.write_text(
                json.dumps(malformed),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_external_opponent_manifest(
                    manifest_path,
                    external_weight=1.0,
                    repository_root=directory,
                )

            malformed = json.loads(json.dumps(original))
            malformed["contract"]["rulebook_sha256"] = "not-a-hash"
            manifest_path.write_text(
                json.dumps(malformed),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "rulebook_sha256"):
                load_external_opponent_manifest(
                    manifest_path,
                    external_weight=1.0,
                    repository_root=directory,
                )

    def test_clustered_scheduler_preserves_seed_selection_and_worker_slots(
        self,
    ) -> None:
        entries = tuple(
            OpponentEntry(
                opponent_id=f"external_{index}",
                kind="external",
                weight=1.0,
                checkpoint_path=f"model_{index}.pt",
                checkpoint_sha256=f"{index + 1:064x}",
                policy_seed=100 + index,
                training_steps=1000,
                generation=0,
                role="candidate",
                rules_version="rules-v1",
                policy_architecture="policy-v1",
                versions_sha256=f"{index + 11:064x}",
            )
            for index in range(3)
        )

        def make_pool() -> OpponentPool:
            return OpponentPool(
                42,
                current_weight=0.0,
                external_entries=entries,
                external_manifest_path="generation.json",
                external_manifest_sha256="f" * 64,
                external_generation=0,
            )

        pool = make_pool()
        scheduler = OpponentEpisodeScheduler(
            pool,
            worker_count=3,
            mode="episode_seed_clustered",
        )
        next_episode_id, wave = scheduler.next_wave(
            0,
            learner_player_for_episode=lambda episode_id: episode_id % 2,
        )
        self.assertGreaterEqual(next_episode_id, 3)
        self.assertEqual(
            {episode_id % 3 for episode_id, _ in wave},
            {0, 1, 2},
        )
        self.assertEqual(len({entry.opponent_id for _, entry in wave}), 1)
        reference = make_pool()
        for episode_id, entry in (*scheduler.pending, *wave):
            expected = reference.select(
                episode_id=episode_id,
                learner_player=episode_id % 2,
            )
            self.assertEqual(entry, expected)
        state = scheduler.state_dict()
        restored = OpponentEpisodeScheduler(
            OpponentPool.from_state_dict(
                pool.state_dict(),
                expected_external_manifest_sha256="f" * 64,
                expected_external_entries=entries,
            ),
            worker_count=3,
            mode="episode_seed_clustered",
        )
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)


if __name__ == "__main__":
    unittest.main()
