"""Required change 5: resumable full checkpoints + inference-only export.

Two distinct file formats, deliberately not interchangeable:

  - Full checkpoint (save_full_checkpoint/load_full_checkpoint): policy +
    value-network weights, optimizer state, RNG states (Python/NumPy/
    PyTorch), update/evaluation counters, hyperparameters, parameter-grid
    definition, target IDs, schema versions, git identity. Loaded with
    weights_only=False (it carries non-tensor metadata) -- NOT the format
    the existing pipeline (experiments/run_autockt_pipeline.py,
    experiments/train_autockt.py) loads.

  - Inference-only export (export_inference_only): a bare
    policy.state_dict(), nothing else -- byte-for-byte what
    torch.save(agent.policy.state_dict(), path) has always produced (see
    experiments/train_autockt.py::--save-final-policy, unmodified). This
    is what the existing pipeline's `torch.load(path, weights_only=True)`
    call sites already expect and continue to work with unchanged.

Reads simulator.provenance.git_identity read-only (not modified).
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from simulator.provenance import git_identity
from simulator.rl_adapter import ACTION_SCHEMA_VERSION

from .autockt_reward import AUTOCKT_REWARD_VERSION
from .autockt_reward_v2 import REWARD_V2_VERSION
from .autockt_state import STATE_DIM as STATE_DIM_V1
from .autockt_state_v2 import STATE_DIM_V2
from .autockt_v3 import STATE_DIM_V3
from .ppo_agent import PPOAgent

try:
    import numpy
except ImportError:  # pragma: no cover -- numpy is a hard project dependency in practice
    numpy = None

CHECKPOINT_SCHEMA_VERSION = 1

REWARD_SCHEMA_VERSIONS = {"v1": AUTOCKT_REWARD_VERSION, "v2": REWARD_V2_VERSION, "v3": "reward_v3"}
STATE_SCHEMA_DIMS = {"v1": STATE_DIM_V1, "v2": STATE_DIM_V2, "v3": STATE_DIM_V3}


class CheckpointIncompatibleError(ValueError):
    """Raised by load_full_checkpoint when the stored checkpoint's schema/
    config does not match the agent/config it's being loaded into --
    Required change 5's "reject incompatible checkpoint/configuration
    combinations." Never silently loads mismatched weights.
    """


@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint_schema_version: int
    state_schema: str  # "v1" or "v2"
    action_schema_version: int
    reward_schema: str  # "v1" or "v2"
    update_count: int
    evaluation_count: int
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    parameter_grid: dict[str, Any] = field(default_factory=dict)
    target_ids: tuple[str, ...] = ()
    git_commit: Optional[str] = None
    working_tree_dirty: Optional[bool] = None


def _rng_states() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }
    if numpy is not None:
        name, keys, pos, has_gauss, cached = numpy.random.get_state()
        state["numpy_random_state"] = (name, keys.tolist(), pos, has_gauss, cached)
    return state


def _restore_rng_states(state: dict[str, Any]) -> None:
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "torch_random_state" in state:
        torch.set_rng_state(state["torch_random_state"])
    if numpy is not None and "numpy_random_state" in state:
        name, keys, pos, has_gauss, cached = state["numpy_random_state"]
        numpy.random.set_state((name, numpy.asarray(keys, dtype=numpy.uint32), pos, has_gauss, cached))


def save_full_checkpoint(
    path: str | Path,
    *,
    agent: PPOAgent,
    update_count: int,
    evaluation_count: int,
    state_schema: str,
    reward_schema: str,
    grid_points: int,
    grid_spacing: str,
    target_ids: tuple[str, ...],
    extra_hyperparameters: Optional[dict[str, Any]] = None,
    repo_root: str | Path = ".",
) -> None:
    if state_schema not in STATE_SCHEMA_DIMS:
        raise ValueError(f"state_schema must be one of {tuple(STATE_SCHEMA_DIMS)}, got {state_schema!r}")
    if reward_schema not in REWARD_SCHEMA_VERSIONS:
        raise ValueError(f"reward_schema must be one of {tuple(REWARD_SCHEMA_VERSIONS)}, got {reward_schema!r}")

    hyperparameters = {
        "state_dim": agent.state_dim, "num_heads": agent.num_heads,
        "gamma": agent.gamma, "gae_lambda": agent.gae_lambda, "clip_eps": agent.clip_eps,
        "entropy_coef": agent.entropy_coef, "value_coef": agent.value_coef,
        **(extra_hyperparameters or {}),
    }
    metadata = CheckpointMetadata(
        checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        state_schema=state_schema, action_schema_version=3 if state_schema == "v3" else ACTION_SCHEMA_VERSION, reward_schema=reward_schema,
        update_count=update_count, evaluation_count=evaluation_count,
        hyperparameters=hyperparameters,
        parameter_grid={"grid_points": grid_points, "grid_spacing": grid_spacing},
        target_ids=tuple(target_ids),
        **git_identity(repo_root),
    )
    payload = {
        "metadata": asdict(metadata),
        "policy_state_dict": agent.policy.state_dict(),
        "value_state_dict": agent.value_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        **_rng_states(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_full_checkpoint(
    path: str | Path, *, agent: PPOAgent, state_schema: str, reward_schema: str, restore_rng: bool = True,
    expected_contract=None,
) -> CheckpointMetadata:
    """Loads a full checkpoint INTO `agent` in place (policy, value net,
    optimizer, and -- unless restore_rng=False -- Python/NumPy/PyTorch RNG
    state). Rejects the load (CheckpointIncompatibleError) BEFORE mutating
    `agent` at all if state_dim/num_heads/state_schema/reward_schema don't
    match what the checkpoint was saved with -- never silently loads
    weights shaped for a different state/action/reward configuration.
    """

    payload = torch.load(path, weights_only=state_schema == "v3")
    metadata = CheckpointMetadata(**payload["metadata"])

    if metadata.hyperparameters.get("state_dim") != agent.state_dim:
        raise CheckpointIncompatibleError(
            f"checkpoint state_dim={metadata.hyperparameters.get('state_dim')} "
            f"does not match agent.state_dim={agent.state_dim}"
        )
    if metadata.hyperparameters.get("num_heads") != agent.num_heads:
        raise CheckpointIncompatibleError(
            f"checkpoint num_heads={metadata.hyperparameters.get('num_heads')} "
            f"does not match agent.num_heads={agent.num_heads}"
        )
    if metadata.state_schema != state_schema:
        raise CheckpointIncompatibleError(
            f"checkpoint was saved with state_schema={metadata.state_schema!r}, "
            f"but state_schema={state_schema!r} was requested"
        )
    if metadata.reward_schema != reward_schema:
        raise CheckpointIncompatibleError(
            f"checkpoint was saved with reward_schema={metadata.reward_schema!r}, "
            f"but reward_schema={reward_schema!r} was requested"
        )

    if state_schema == "v3":
        from .runtime_contract import validate_inference_contract
        contract = metadata.hyperparameters.get("runtime_contract")
        if expected_contract is None or contract != expected_contract:
            raise CheckpointIncompatibleError("v3 resume requires identical runtime contract (grids, backend and channel)")
        validate_inference_contract(contract, expected_contract)
    # Validate both weight sets before mutating either live network.
    for network, key in ((agent.policy, "policy_state_dict"), (agent.value_net, "value_state_dict")):
        current = network.state_dict()
        saved = payload.get(key, {})
        if current.keys() != saved.keys() or any(current[k].shape != saved[k].shape for k in current):
            raise CheckpointIncompatibleError(f"Invalid {key} shapes/keys")
    import copy
    staged = copy.deepcopy(agent)
    staged.policy.load_state_dict(payload["policy_state_dict"])
    staged.value_net.load_state_dict(payload["value_state_dict"])
    staged.optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        if "python_random_state" in payload:
            random.Random().setstate(payload["python_random_state"])
        if "torch_random_state" in payload:
            torch.Generator().set_state(payload["torch_random_state"])
        if numpy is not None and "numpy_random_state" in payload:
            name, keys, pos, has_gauss, cached = payload["numpy_random_state"]
            numpy.random.RandomState().set_state((name, numpy.asarray(keys, dtype=numpy.uint32), pos, has_gauss, cached))
    agent.policy.load_state_dict(payload["policy_state_dict"])
    agent.value_net.load_state_dict(payload["value_state_dict"])
    agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        _restore_rng_states(payload)
    return metadata


def export_inference_only(path: str | Path, *, agent: PPOAgent, metadata=None) -> None:
    """A bare policy.state_dict() -- byte-for-byte the same format
    experiments/train_autockt.py::--save-final-policy already produces
    (torch.save(agent.policy.state_dict(), path)), compatible with the
    EXISTING pipeline's weights_only=True loading path, unchanged.
    """

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = agent.policy.state_dict()
    if metadata is not None:
        payload = {"policy_state_dict": payload, "metadata": metadata}
    torch.save(payload, path)
