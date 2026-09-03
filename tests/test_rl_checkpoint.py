"""SPICE-free tests for rl/checkpoint.py -- Required change 5."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from rl.autockt_state import STATE_DIM
from rl.checkpoint import (
    CheckpointIncompatibleError,
    export_inference_only,
    load_full_checkpoint,
    save_full_checkpoint,
)
from rl.parameter_grid import PARAMETER_NAMES
from rl.ppo_agent import PPOAgent


def _agent(seed=0, state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES)):
    return PPOAgent(state_dim=state_dim, num_heads=num_heads, seed=seed)


class FullCheckpointRoundTripTests(unittest.TestCase):
    def test_saves_and_loads_policy_weights_correctly(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            source = _agent(seed=1)
            save_full_checkpoint(
                path, agent=source, update_count=3, evaluation_count=42,
                state_schema="v1", reward_schema="v1", grid_points=21, grid_spacing="log",
                target_ids=("trivial", "hard"),
            )
            target = _agent(seed=999)  # different init -- weights must come from the checkpoint, not this seed
            metadata = load_full_checkpoint(path, agent=target, state_schema="v1", reward_schema="v1")

            for p_source, p_target in zip(source.policy.parameters(), target.policy.parameters()):
                self.assertTrue(torch.equal(p_source, p_target))
            self.assertEqual(metadata.update_count, 3)
            self.assertEqual(metadata.evaluation_count, 42)
            self.assertEqual(metadata.target_ids, ("trivial", "hard"))

    def test_restores_optimizer_state(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            source = _agent(seed=1)
            # Take one optimizer step so its internal state (Adam moments) is non-trivial.
            dummy_loss = sum(p.sum() for p in source.policy.parameters())
            dummy_loss.backward()
            source.optimizer.step()
            save_full_checkpoint(
                path, agent=source, update_count=1, evaluation_count=1,
                state_schema="v1", reward_schema="v1", grid_points=21, grid_spacing="log", target_ids=(),
            )
            target = _agent(seed=2)
            load_full_checkpoint(path, agent=target, state_schema="v1", reward_schema="v1")
            self.assertEqual(
                len(source.optimizer.state_dict()["state"]), len(target.optimizer.state_dict()["state"]),
            )

    def test_restores_rng_state_reproducibly(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            source = _agent(seed=1)
            import random
            random.seed(12345)
            random.random()  # advance the stream so it's not fresh
            save_full_checkpoint(
                path, agent=source, update_count=0, evaluation_count=0,
                state_schema="v1", reward_schema="v1", grid_points=21, grid_spacing="log", target_ids=(),
            )
            expected_next = random.random()

            random.seed(0)  # perturb the global stream
            target = _agent(seed=2)
            load_full_checkpoint(path, agent=target, state_schema="v1", reward_schema="v1", restore_rng=True)
            self.assertEqual(random.random(), expected_next)

    def test_metadata_carries_git_identity_and_schema_versions(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_full_checkpoint(
                path, agent=_agent(), update_count=0, evaluation_count=0,
                state_schema="v2", reward_schema="v2", grid_points=21, grid_spacing="log", target_ids=(),
            )
            payload = torch.load(path, weights_only=False)
            metadata = payload["metadata"]
            self.assertEqual(metadata["state_schema"], "v2")
            self.assertEqual(metadata["reward_schema"], "v2")
            self.assertIn("action_schema_version", metadata)
            self.assertIn("git_commit", metadata)  # may be None outside a git repo, but the key must exist


class IncompatibleCheckpointRejectionTests(unittest.TestCase):
    def test_state_dim_mismatch_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_full_checkpoint(
                path, agent=_agent(state_dim=STATE_DIM), update_count=0, evaluation_count=0,
                state_schema="v1", reward_schema="v1", grid_points=21, grid_spacing="log", target_ids=(),
            )
            mismatched = _agent(state_dim=STATE_DIM + 5)
            with self.assertRaises(CheckpointIncompatibleError):
                load_full_checkpoint(path, agent=mismatched, state_schema="v1", reward_schema="v1")

    def test_state_schema_mismatch_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_full_checkpoint(
                path, agent=_agent(), update_count=0, evaluation_count=0,
                state_schema="v1", reward_schema="v1", grid_points=21, grid_spacing="log", target_ids=(),
            )
            with self.assertRaises(CheckpointIncompatibleError):
                load_full_checkpoint(path, agent=_agent(), state_schema="v2", reward_schema="v1")

    def test_reward_schema_mismatch_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_full_checkpoint(
                path, agent=_agent(), update_count=0, evaluation_count=0,
                state_schema="v1", reward_schema="v1", grid_points=21, grid_spacing="log", target_ids=(),
            )
            with self.assertRaises(CheckpointIncompatibleError):
                load_full_checkpoint(path, agent=_agent(), state_schema="v1", reward_schema="v2")

    def test_rejection_does_not_mutate_the_agent(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_full_checkpoint(
                path, agent=_agent(seed=1), update_count=0, evaluation_count=0,
                state_schema="v1", reward_schema="v1", grid_points=21, grid_spacing="log", target_ids=(),
            )
            target = _agent(seed=2)
            original_params = [p.clone() for p in target.policy.parameters()]
            with self.assertRaises(CheckpointIncompatibleError):
                load_full_checkpoint(path, agent=target, state_schema="v2", reward_schema="v1")
            for original, current in zip(original_params, target.policy.parameters()):
                self.assertTrue(torch.equal(original, current))

    def test_invalid_state_schema_argument_rejected_at_save_time(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            with self.assertRaises(ValueError):
                save_full_checkpoint(
                    path, agent=_agent(), update_count=0, evaluation_count=0,
                    state_schema="v99", reward_schema="v1", grid_points=21, grid_spacing="log", target_ids=(),
                )


class InferenceOnlyExportTests(unittest.TestCase):
    def test_export_is_a_bare_state_dict_loadable_with_weights_only_true(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference.pt"
            source = _agent(seed=1)
            export_inference_only(path, agent=source)

            # This is EXACTLY the loading path the existing pipeline uses
            # (experiments/run_autockt_pipeline.py::generate_candidates,
            # experiments/train_autockt.py) -- must keep working unchanged.
            state_dict = torch.load(path, weights_only=True)
            target = _agent(seed=2)
            target.policy.load_state_dict(state_dict)
            for p_source, p_target in zip(source.policy.parameters(), target.policy.parameters()):
                self.assertTrue(torch.equal(p_source, p_target))

    def test_export_contains_only_policy_weights_no_metadata(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference.pt"
            export_inference_only(path, agent=_agent())
            state_dict = torch.load(path, weights_only=True)
            self.assertIsInstance(state_dict, dict)
            self.assertTrue(all(torch.is_tensor(v) for v in state_dict.values()))


if __name__ == "__main__":
    unittest.main()
