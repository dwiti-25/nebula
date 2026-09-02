"""ML-side environment wrapper. [NEBULA ADAPTATION -- the "ML environment /
adapter" layer between the PPO agent and the untouched simulator.]

This is the only file in this package that talks to
simulator.rl_adapter.ReceiverRLAdapter, and it only calls that class's
existing public reset()/step()/seed() -- simulator/rl_adapter.py itself is
never modified. It composes parameter_grid + target_spec + autockt_state +
autockt_action + autockt_reward into a single AutoCkt-shaped episode
contract (reset() -> state; step(choices) -> AutoCktStep).

[AUTOCKT-REPLICATED] episode/termination shape, verified from
autockt/envs/ngspice_vanilla_opamp.py (github.com/ksettaluri6/AutoCkt):
  - one TargetSpec per episode (AutoCkt: random per-episode goal spec when
    `generalize=True`)
  - a FIXED initial parameter-grid index every episode, not randomized
    (AutoCkt: `self.cur_params_idx = np.array([33,33,33,33,33,14,20])`)
  - `done=True` exactly when the reward hits the terminal bonus (AutoCkt:
    `if reward >= 10: done = True`), never otherwise from inside the env

[ROUGH-SCALE ADAPTATION] horizon truncation: AutoCkt's own env.step() never
checks step count -- horizon=30 is enforced by the outer Ray RLlib trainer
config, not the environment. This class mirrors that split by returning a
separate `truncated` flag (computed from `horizon`) alongside `done`, rather
than folding truncation into `done` -- matching NEBULA's own existing
convention of a `terminated`/`truncated` split in
simulator.rl_adapter.RLStep. The `horizon` value itself is caller-supplied;
see docs/autockt-mapping.md for why the first NEBULA milestone uses a much
smaller horizon than AutoCkt's 30 (SPICE evaluation cost, measured at
~15-65s/step in this session, vs. AutoCkt's fast opamp SPICE).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Mapping, Optional, Sequence

from simulator.rl_adapter import METRIC_OBSERVATION_NAMES, ReceiverRLAdapter

from .autockt_action import apply_action, indices_to_normalized_action
from .autockt_reward import TERMINAL_BONUS, autockt_reward
from .autockt_state import STATE_DIM, build_state
from .parameter_grid import ParameterGrid, PARAMETER_NAMES, build_parameter_grids
from .target_spec import TargetSpec

RewardFn = Callable[..., float]


def _default_reward_fn(
    metrics: Mapping[str, float], target: TargetSpec, *, success: bool, failure_stage: Optional[str],
) -> float:
    """[NEBULA ADAPTATION] wraps the locked, unmodified autockt_reward in the
    richer (metrics, target, success, failure_stage) signature every
    reward_fn must accept, so AutoCktReceiverEnv.step() can call whichever
    reward function is plugged in identically -- ignores failure_stage,
    exactly reproducing autockt_reward's own unchanged behavior.
    """

    return autockt_reward(metrics, target, success=success)


def metrics_from_observation(observation: Sequence[float]) -> dict[str, float]:
    """Extracts the metric-name -> value subset of the existing, untouched
    ReceiverRLAdapter observation tuple (the first len(METRIC_OBSERVATION_NAMES)
    entries; the remaining half are the adapter's own validity flags -- see
    simulator/rl_adapter.py::observation_from_evaluation). Values are already
    zero-filled by the adapter itself when a metric was missing/non-finite.

    NOTE: the observation slot is *labeled* "dfe_eye_height_v" in
    METRIC_OBSERVATION_NAMES, but simulator/rl_adapter.py::observation_from_evaluation
    documents (and this session's inspection confirmed) that it is populated
    from the raw evaluation metric "dfe_locked_phase_eye_height_v", not a
    metric literally named "dfe_eye_height_v". This function re-keys that one
    slot back to "dfe_locked_phase_eye_height_v" so downstream code
    (target_spec.py, autockt_reward.py) can use the same metric name that
    simulator/rl_adapter.py's own reward_v1/constraints_from_evaluation use,
    without re-deriving the alias in multiple places.
    """

    metrics = dict(zip(METRIC_OBSERVATION_NAMES, observation[: len(METRIC_OBSERVATION_NAMES)]))
    metrics["dfe_locked_phase_eye_height_v"] = metrics.pop("dfe_eye_height_v")
    return metrics


# [NEBULA ADAPTATION] see AutoCktReceiverEnv's `randomize_initial_state`
# docstring below. Deliberately NOT parameter_grid.ACTION_DELTAS
# (= (-1, 0, +2), the asymmetric mechanic the *policy's* actions use, see
# rl/autockt_action.py -- unmodified, untouched by this feature): this is a
# separate, symmetric {-1, 0, +1} perturbation applied only once, at
# episode-reset time, to pick the episode's starting point -- never fed
# through rl/autockt_action.py::apply_action.
INITIAL_STATE_PERTURBATION_CHOICES: tuple[int, int, int] = (-1, 0, 1)


@dataclass(frozen=True)
class AutoCktStep:
    state: tuple[float, ...]
    reward: float
    done: bool
    truncated: bool
    info: dict[str, object]


class AutoCktReceiverEnv:
    """AutoCkt-shaped episode wrapper around the untouched ReceiverRLAdapter.

    [NEBULA ADAPTATION] `randomize_initial_state` (default False, preserving
    exact prior behavior): AutoCkt's own env.reset() always returns to the
    identical fixed initial parameter-grid index every episode
    ([AUTOCKT-REPLICATED] methodology, see rl/parameter_grid.py). Real-SPICE
    evidence collected this session showed that fixed point -- verified-good
    by construction (VERIFIED_INITIAL_PARAMETERS) -- already satisfies every
    target this repo can currently express, so nearly every episode's very
    first evaluated step already hits the terminal bonus, leaving little
    room for a real bad/near-good-to-good learning curve to appear. When
    enabled, each reset() perturbs the anchor `initial_indices` by an
    independently-sampled amount in INITIAL_STATE_PERTURBATION_CHOICES =
    {-1, 0, +1} per parameter (at most one grid index away, symmetric,
    clipped to grid bounds via the existing, unmodified
    ParameterGrid.clip_index) -- deliberately *not* the policy's own
    {-1, 0, +2} action-delta mechanic (rl/autockt_action.py, untouched),
    since this is a one-time initialization choice, not an RL action. The
    anchor itself (`initial_indices`, i.e. VERIFIED_INITIAL_PARAMETERS) is
    unchanged -- only where each episode *starts* relative to it varies.
    Uses its own `random.Random(seed)` stream, independent of the
    target-selection RNG, so enabling this cannot change which target an
    existing seeded run selects per episode.
    """

    def __init__(
        self,
        *,
        target_pool: Sequence[TargetSpec],
        initial_indices: Sequence[int],
        horizon: int,
        adapter: ReceiverRLAdapter | None = None,
        grids: Mapping[str, ParameterGrid] | None = None,
        seed: int = 0,
        randomize_initial_state: bool = False,
        reward_fn: Optional[RewardFn] = None,
    ):
        if not target_pool:
            raise ValueError("target_pool must be non-empty")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.target_pool = tuple(target_pool)
        self.initial_indices = tuple(initial_indices)
        if len(self.initial_indices) != len(PARAMETER_NAMES):
            raise ValueError(f"initial_indices must have {len(PARAMETER_NAMES)} entries")
        self.horizon = horizon
        self.grids: dict[str, ParameterGrid] = dict(grids) if grids is not None else build_parameter_grids()
        self.adapter = adapter if adapter is not None else ReceiverRLAdapter(seed=seed)
        self._episode_rng = random.Random(seed)
        self.randomize_initial_state = randomize_initial_state
        self._init_rng = random.Random(seed)  # separate stream, see class docstring
        # [NEBULA ADAPTATION] optional, additive: defaults to the exact prior
        # behavior (autockt_reward, unmodified) via _default_reward_fn.
        # Passing e.g. rl.autockt_reward.graded_autockt_reward here is the
        # only way episode reward computation changes -- nothing else in
        # this class does, and no existing caller that omits reward_fn sees
        # any behavior change.
        self.reward_fn: RewardFn = reward_fn if reward_fn is not None else _default_reward_fn
        self.target: TargetSpec | None = None
        self.indices: tuple[int, ...] = self.initial_indices
        self.step_count = 0

    def _sample_initial_indices(self) -> tuple[int, ...]:
        """[NEBULA ADAPTATION] see class docstring."""

        return tuple(
            self.grids[name].clip_index(anchor + self._init_rng.choice(INITIAL_STATE_PERTURBATION_CHOICES))
            for name, anchor in zip(PARAMETER_NAMES, self.initial_indices)
        )

    def _normalized_indices(self) -> tuple[float, ...]:
        """[NEBULA ADAPTATION] see ParameterGrid.normalized_index. Feeds
        build_state's `param_indices` argument with values scaled to
        [-1, 1] instead of raw grid indices (0..grid_points-1) -- a
        real-SPICE-diagnosed fix, not a stylistic preference: a direct
        forward-pass check (see docs/autockt-mapping.md) showed raw indices
        of magnitude up to 20 alongside the other ~[-1,1]-bounded state
        components pushed ~40% of the policy/value networks' first-hidden-layer
        tanh units into near-saturation (|pre-activation| > 3) at
        initialization, on a representative sample of real states. This is
        a change to what AutoCktReceiverEnv *feeds* build_state, not to
        build_state itself (rl/autockt_state.py is unmodified and still
        accepts/documents raw indices for direct callers).
        """

        return tuple(self.grids[name].normalized_index(i) for name, i in zip(PARAMETER_NAMES, self.indices))

    def reset(self, *, target: TargetSpec | None = None) -> tuple[tuple[float, ...], dict[str, object]]:
        self.target = target if target is not None else self._episode_rng.choice(self.target_pool)
        self.indices = self._sample_initial_indices() if self.randomize_initial_state else self.initial_indices
        self.step_count = 0
        # [AUTOCKT-REPLICATED] AutoCkt's reset() does not re-run a
        # simulation either; cur_spec_norm is computed against whatever
        # `self.specs` last held (post-__init__ zeros on the very first
        # reset). Mirrored here with an all-zero metrics dict.
        state = build_state({}, self.target, self._normalized_indices())
        return state, {"target": self.target.as_dict()}

    def step(self, choices: Sequence[int]) -> AutoCktStep:
        if self.target is None:
            raise RuntimeError("call reset() before step()")
        self.indices = apply_action(self.indices, choices, self.grids)
        normalized_action = indices_to_normalized_action(self.indices, self.grids)
        rl_step = self.adapter.step(normalized_action)
        metrics = metrics_from_observation(rl_step.observation)
        success = rl_step.info["failure_stage"] is None
        reward = self.reward_fn(
            metrics, self.target, success=success, failure_stage=rl_step.info["failure_stage"],
        )
        self.step_count += 1
        done = reward >= TERMINAL_BONUS
        truncated = self.step_count >= self.horizon
        state = build_state(metrics, self.target, self._normalized_indices())
        info = {
            "success": success,
            "spec_satisfied": done,
            "step_count": self.step_count,
            "failure_stage": rl_step.info["failure_stage"],
            "total_evaluation_count": rl_step.info["total_evaluation_count"],
            "parameters": rl_step.info["parameters"],
            "indices": self.indices,
            "metrics": metrics,
        }
        return AutoCktStep(state, reward, done, truncated, info)


__all__ = ["AutoCktReceiverEnv", "AutoCktStep", "STATE_DIM", "metrics_from_observation"]
