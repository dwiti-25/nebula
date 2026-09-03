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
from .autockt_reward_v2 import RewardResult, reward_v2
from .autockt_state import STATE_DIM, build_state
from .autockt_state_v2 import STATE_DIM_V2, build_state_v2, validity_mask_from_observation
from .parameter_grid import ACTION_DELTAS, ParameterGrid, PARAMETER_NAMES, build_parameter_grids
from .target_spec import SPEC_NAMES, TargetSpec

RewardFn = Callable[..., float]

STATE_SCHEMAS = ("v1", "v2")

# SPEC_NAMES metrics as they're keyed in METRIC_OBSERVATION_NAMES's
# validity-flag half (see simulator/rl_adapter.py::observation_from_evaluation
# -- one name, dfe_locked_phase_eye_height_v, is aliased to dfe_eye_height_v
# in the observation vector).
_SPEC_TO_OBSERVATION_NAME = {name: name for name in SPEC_NAMES}
_SPEC_TO_OBSERVATION_NAME["dfe_locked_phase_eye_height_v"] = "dfe_eye_height_v"


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


def spec_metrics_valid(observation: Sequence[float]) -> bool:
    """[NEBULA ADAPTATION] whether the 4 SPEC_NAMES metrics were genuinely
    measured this step -- reads the adapter's OWN validity flags (the
    second half of its observation tuple, simulator/rl_adapter.py::
    observation_from_evaluation), never fabricated. True only when ALL 4
    are valid, since this project's specific 4 tracked specs are all
    computed together at the transient stage (see
    rl/autockt_state_v2.py module docstring) -- in practice this is an
    all-or-nothing flag for this repo's current metric set, not an
    approximation.
    """

    mask = validity_mask_from_observation(observation, METRIC_OBSERVATION_NAMES)
    return all(mask[_SPEC_TO_OBSERVATION_NAME[name]] for name in SPEC_NAMES)


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
        evaluate_on_reset: bool = False,
        state_schema: str = "v1",
        use_reward_v2: bool = False,
        action_deltas: Sequence[int] = ACTION_DELTAS,
        max_total_evaluations: Optional[int] = None,
    ):
        if not target_pool:
            raise ValueError("target_pool must be non-empty")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if state_schema not in STATE_SCHEMAS:
            raise ValueError(f"state_schema must be one of {STATE_SCHEMAS}, got {state_schema!r}")
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
        # any behavior change. Ignored when use_reward_v2=True.
        self.reward_fn: RewardFn = reward_fn if reward_fn is not None else _default_reward_fn
        # [NEBULA ADAPTATION -- PPO model-improvement study] all four below
        # default to exact prior behavior; every combination is versioned
        # and independently selectable (Required changes 1/2/3/7). See
        # module docstring additions and docs/RL_PPO_V2_STUDY.md.
        self.evaluate_on_reset = evaluate_on_reset
        self.state_schema = state_schema
        self.use_reward_v2 = use_reward_v2
        self.action_deltas = tuple(action_deltas)
        # [NEBULA ADAPTATION -- Required change 4] the experiment-wide/
        # adapter evaluation budget, checked BEFORE any step (or the reset-
        # time evaluation) is attempted, so a caller sharing one adapter's
        # budget across many episodes (as experiments/train_autockt.py
        # already does) never triggers ReceiverRLAdapter.step()'s own
        # RuntimeError guard -- truncates cleanly instead. Defaults to the
        # adapter's own configured budget when omitted.
        self.max_total_evaluations = (
            max_total_evaluations if max_total_evaluations is not None else self.adapter.budget.max_evaluations
        )
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

    def budget_exhausted(self) -> bool:
        """[NEBULA ADAPTATION -- Required change 4] checked before ANY
        adapter call (reset-time evaluation or step) -- see
        max_total_evaluations docstring above.
        """

        return self.adapter.total_evaluations >= self.max_total_evaluations

    def _build_state(
        self, metrics: Mapping[str, float], *, metrics_valid: bool, failure_stage: Optional[str],
    ) -> tuple[float, ...]:
        if self.state_schema == "v2":
            return build_state_v2(
                metrics, self.target, self._normalized_indices(),
                metrics_valid=metrics_valid, failure_stage=failure_stage,
            )
        return build_state(metrics, self.target, self._normalized_indices())

    def reset(self, *, target: TargetSpec | None = None) -> tuple[tuple[float, ...], dict[str, object]]:
        self.target = target if target is not None else self._episode_rng.choice(self.target_pool)
        self.indices = self._sample_initial_indices() if self.randomize_initial_state else self.initial_indices
        self.step_count = 0

        # [NEBULA ADAPTATION -- Required change 1] "corrected reset":
        # opt-in, evaluate_on_reset=False preserves the exact prior
        # behavior (AutoCkt's own reset() also never re-runs a simulation;
        # mirrored here with an all-zero/unevaluated metrics dict -- see
        # module docstring). When enabled, evaluates the ACTUAL starting
        # indices once via the existing, unmodified adapter (spends one
        # real evaluation from the shared budget -- reported in
        # reset_info["reset_evaluation"] for separate cost accounting, per
        # instruction) instead of feeding the state a fabricated zero
        # reading. Checks the budget first (Required change 4) so a
        # caller near budget exhaustion never has this optional call
        # trigger the adapter's own RuntimeError guard.
        metrics: Mapping[str, float] = {}
        failure_stage: Optional[str] = "unevaluated"
        metrics_valid = False
        reset_evaluation_info: dict[str, object] = {"evaluated": False}
        if self.evaluate_on_reset and not self.budget_exhausted():
            normalized_action = indices_to_normalized_action(self.indices, self.grids)
            rl_step = self.adapter.step(normalized_action)
            metrics = metrics_from_observation(rl_step.observation)
            failure_stage = rl_step.info["failure_stage"]
            metrics_valid = spec_metrics_valid(rl_step.observation)
            reset_evaluation_info = {
                "evaluated": True, "failure_stage": failure_stage,
                "total_evaluation_count": rl_step.info["total_evaluation_count"],
            }
        elif self.evaluate_on_reset:
            reset_evaluation_info = {"evaluated": False, "reason": "adapter budget exhausted"}

        state = self._build_state(metrics, metrics_valid=metrics_valid, failure_stage=failure_stage)
        return state, {"target": self.target.as_dict(), "reset_evaluation": reset_evaluation_info}

    def step(self, choices: Sequence[int]) -> AutoCktStep:
        if self.target is None:
            raise RuntimeError("call reset() before step()")
        if self.budget_exhausted():
            # [NEBULA ADAPTATION -- Required change 4] "Do not attempt one
            # additional simulator call after exhaustion": truncate
            # cleanly with no adapter call at all, rather than letting
            # ReceiverRLAdapter.step()'s own RuntimeError guard fire.
            state = self._build_state({}, metrics_valid=False, failure_stage="unevaluated")
            info = {
                "success": False, "spec_satisfied": False, "step_count": self.step_count,
                "failure_stage": "budget_exhausted", "total_evaluation_count": self.adapter.total_evaluations,
                "parameters": None, "indices": self.indices, "metrics": {},
            }
            return AutoCktStep(state, 0.0, False, True, info)

        self.indices = apply_action(self.indices, choices, self.grids, deltas=self.action_deltas)
        normalized_action = indices_to_normalized_action(self.indices, self.grids)
        rl_step = self.adapter.step(normalized_action)
        metrics = metrics_from_observation(rl_step.observation)
        success = rl_step.info["failure_stage"] is None
        failure_stage = rl_step.info["failure_stage"]
        metrics_valid = spec_metrics_valid(rl_step.observation)

        reward_result: Optional[RewardResult] = None
        if self.use_reward_v2:
            reward_result = reward_v2(
                metrics, self.target, metrics_valid=metrics_valid, success=success, failure_stage=failure_stage,
            )
            reward = reward_result.total
            done = reward_result.autockt_terminal_success
        else:
            reward = self.reward_fn(metrics, self.target, success=success, failure_stage=failure_stage)
            done = reward >= TERMINAL_BONUS

        self.step_count += 1
        # [NEBULA ADAPTATION -- Required change 4] truncate on horizon OR
        # the adapter's own budget-exhaustion signal (rl_step.truncated,
        # simulator/rl_adapter.py -- previously computed but silently
        # ignored here). Fixes a real bug: without this, the NEXT step()
        # call (this episode or the next one) would reach the adapter
        # already at budget and crash with a RuntimeError instead of
        # truncating.
        truncated = self.step_count >= self.horizon or rl_step.truncated
        state = self._build_state(metrics, metrics_valid=metrics_valid, failure_stage=failure_stage)
        info = {
            "success": success,
            "spec_satisfied": done,
            "step_count": self.step_count,
            "failure_stage": failure_stage,
            "total_evaluation_count": rl_step.info["total_evaluation_count"],
            "parameters": rl_step.info["parameters"],
            "indices": self.indices,
            "metrics": metrics,
            "metrics_valid": metrics_valid,
            "metrics_valid_mask": validity_mask_from_observation(rl_step.observation, METRIC_OBSERVATION_NAMES),
            "reward_result": reward_result,  # None unless use_reward_v2=True
        }
        return AutoCktStep(state, reward, done, truncated, info)


__all__ = [
    "AutoCktReceiverEnv", "AutoCktStep", "STATE_DIM", "STATE_DIM_V2", "metrics_from_observation",
    "spec_metrics_valid",
]
