"""AutoCkt-style reward. Completely separate from simulator/rl_adapter.py's
reward_v1 -- reward_v1 is never imported or reused anywhere in this file, by
explicit instruction.

[AUTOCKT-REPLICATED] matches autockt/envs/ngspice_vanilla_opamp.py::reward()
(github.com/ksettaluri6/AutoCkt), verified verbatim from source:

    def reward(self, spec, goal_spec):
        rel_specs = self.lookup(spec, goal_spec)
        reward = 0.0
        for i, rel_spec in enumerate(rel_specs):
            if self.specs_id[i] == 'ibias_max':
                rel_spec = rel_spec * -1.0
            if rel_spec < 0:
                reward += rel_spec
            # else: satisfied -- contributes exactly 0, no overshoot credit
        return reward if reward < -0.02 else 10

i.e.: per-spec relative error via `lookup`, sign-flipped for
smaller-is-better specs, only unsatisfied (negative) contributions are
summed, and a flat terminal bonus of 10 replaces the summed penalty once it
is >= -0.02 (never penalizing overshoot beyond a satisfied spec).
"""

from __future__ import annotations

from typing import Mapping

from .autockt_state import lookup, signed_relative_error
from .target_spec import SPEC_NAMES, TargetSpec

# Versioned independently from simulator/rl_adapter.py's REWARD_VERSION
# ("receiver_reward_v1") -- this is a distinct, ML-side reward, never a
# variant of reward_v1.
AUTOCKT_REWARD_VERSION = "autockt_reward_v1"

# [AUTOCKT-REPLICATED] verified constants from ngspice_vanilla_opamp.py::reward().
UNSATISFIED_THRESHOLD = -0.02
TERMINAL_BONUS = 10.0

# [NEBULA ADAPTATION] AutoCkt's official repo has no verified graceful
# handling of a failed SPICE simulation: NgSpiceWrapper.simulate()'s `info`
# failure flag is read but discarded by TwoStageAmp.update(), and
# parse_output() proceeds to np.genfromtxt() on the (possibly-missing)
# result files regardless -- i.e. a failed simulation would propagate an
# exception rather than return a defined reward (see
# docs/autockt-mapping.md, "AutoCkt failed-simulation handling"). NEBULA's
# simulator instead returns a structured ReceiverEvaluation(success=False,
# failed_stage=...), so this reward defines an explicit, dominant failure
# penalty instead of crashing. FAILURE_REWARD is a new NEBULA-side constant,
# chosen to sit below every value the AutoCkt-replicated branch below can
# produce (the unsatisfied branch is a sum of `lookup()` terms in [-1, 0), so
# it is bounded below by -len(SPEC_NAMES) = -4; -1.0 sits inside that range
# rather than dominating it by construction -- unlike reward_v1's -100.0,
# which is deliberately far outside its own success range. This asymmetry is
# intentional: unlike reward_v1, this reward's failure penalty is NOT
# claimed to dominate every possible partial-credit score, since AutoCkt's
# own reward has no such "failures cannot beat successes" design goal to
# replicate. This value and its rationale are documented here, not
# borrowed from reward_v1's -100.0 sentinel.
FAILURE_REWARD = -1.0


def autockt_reward(
    current_metrics: Mapping[str, float],
    target: TargetSpec,
    *,
    success: bool,
) -> float:
    """[AUTOCKT-REPLICATED] reward mechanics, applied to NEBULA's 4-spec
    subset (dfe_locked_phase_eye_height_v, dfe_eye_width_ui,
    dfe_min_margin_v, ctle_power_w -- see target_spec.py). `success` should
    be derived from the evaluation's own failure_stage (e.g.
    `rl_step.info["failure_stage"] is None`), not from any reward_v1 value.
    """

    if not success:  # [NEBULA ADAPTATION] -- see FAILURE_REWARD docstring above.
        return FAILURE_REWARD
    reward = 0.0
    for name in SPEC_NAMES:
        relative_error = signed_relative_error(name, lookup(current_metrics.get(name, 0.0), getattr(target, name)))
        if relative_error < 0:
            reward += relative_error
    return reward if reward < UNSATISFIED_THRESHOLD else TERMINAL_BONUS


def is_spec_satisfied(current_metrics: Mapping[str, float], target: TargetSpec, *, success: bool) -> bool:
    return success and autockt_reward(current_metrics, target, success=success) >= TERMINAL_BONUS


# [NEBULA ADAPTATION] additive alternative reward -- autockt_reward above is
# NEVER modified, and every existing caller/historical result (all PPO
# training/benchmark logs in results/) stays on it, byte-for-byte unchanged
# behavior. This addresses a confirmed collapse, not a hypothesized one:
# docs/autockt-mapping.md sec 20's fair, no-warm-start PPO trial had every
# one of 20 episodes fail at the `dc` stage, so autockt_reward's own flat
# FAILURE_REWARD was returned every single time -- mean_episode_reward was
# exactly -1.0 across all 4 updates, zero variance, meaning the policy-
# gradient update had nothing to differentiate any of the 20 sampled actions
# by, regardless of how close (or far) each one actually came.
#
# Root cause, diagnosed from simulator/rl_adapter.py::observation_from_evaluation
# and rl/autockt_env.py::metrics_from_observation: the adapter always
# zero-fills SPEC_NAMES-shaped metrics that were never measured. So naively
# grading EVERY failure the way autockt_reward's success branch does would
# NOT fix the collapse -- it would silently replace one constant (-1.0) with
# a DIFFERENT constant (the relative-error sum against four zero-filled,
# fabricated "achieved" values), since a dc/ac/ctle_transient/channel
# failure never reaches the stage that actually computes
# dfe_locked_phase_eye_height_v/eye_width_ui/min_margin_v/ctle_power_w. Only
# a failure AT the `transient` stage itself (which computes all four before
# failing one of their hard gates) carries real, non-fabricated per-spec
# distance information. This mirrors experiments/train_cem.py's own,
# already-validated graded_cem_fitness and its GRADED_FITNESS_RELIABLE_STAGES
# gate, for the identical reason -- not a new idea, the same fix applied to
# PPO's reward instead of CEM's fitness.
GRADED_RELIABLE_STAGES: tuple[str | None, ...] = (None, "transient")

# Strictly below the graded branch's own attainable range (a sum of <= 4
# negative signed_relative_error terms, each individually bounded by
# lookup()'s own ~[-1, 1] range -- so at most roughly -4 in practice) --
# chosen so a no-information failure (dc/ac/ctle_transient/channel) can
# never be mistaken for, or numerically outrank, a graded transient-stage
# failure that came closer. This is a structural bound, not a tuned weight:
# the same reasoning and role as experiments/train_cem.py's own
# GRADED_FITNESS_NO_INFORMATION_FLOOR (-10.0), reused here as the identical
# value for the identical reason.
GRADED_NO_INFORMATION_FLOOR = -10.0


def graded_autockt_reward(
    current_metrics: Mapping[str, float],
    target: TargetSpec,
    *,
    success: bool,
    failure_stage: str | None,
) -> float:
    """[NEBULA ADAPTATION] Identical to autockt_reward when success=True
    (same formula, same TERMINAL_BONUS -- success handling is untouched).
    Differs only in how a FAILURE is scored: autockt_reward always returns
    the flat FAILURE_REWARD; this function additionally grades a
    `transient`-stage failure (real per-spec distances are available, see
    module comment above GRADED_RELIABLE_STAGES) by the same relative-
    error-sum formula the success branch uses, and falls back to the fixed,
    deliberately-dominated GRADED_NO_INFORMATION_FLOOR only for failures at
    any earlier stage, where no real per-spec measurement exists to grade.
    """

    if success:
        return autockt_reward(current_metrics, target, success=True)
    if failure_stage not in GRADED_RELIABLE_STAGES:
        return GRADED_NO_INFORMATION_FLOOR
    total = 0.0
    for name in SPEC_NAMES:
        relative_error = signed_relative_error(name, lookup(current_metrics.get(name, 0.0), getattr(target, name)))
        if relative_error < 0:
            total += relative_error
    return total  # <= 0.0 always; never reaches TERMINAL_BONUS, so `done` semantics are unaffected
