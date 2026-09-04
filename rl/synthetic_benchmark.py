"""SYNTHETIC PPO-TRAINING-MECHANICS BENCHMARK -- NOT A CIRCUIT SIMULATOR.

============================================================================
READ THIS BEFORE USING ANYTHING IN THIS FILE
============================================================================

This module is NOT:
  - a circuit simulator
  - a physics model of the CTLE/DFE receiver
  - a learned surrogate (nothing here is fit to SPICE data; no training
    data from simulator/ or results/*.jsonl was used to derive any constant
    below)
  - evidence of circuit performance, feasibility, or optimization quality

This module IS:
  - a small, deliberately simple, fully deterministic, hand-specified
    function of the same 5 continuous parameters
    (rload_ohm, rdeg_ohm, cdeg_f, itail_a, dfe_tap_v) that
    simulator.receiver.ReceiverParameters already exposes
  - a drop-in replacement for the `evaluator` argument of
    simulator.rl_adapter.ReceiverRLAdapter (matching evaluate_receiver's own
    call signature and ReceiverEvaluation return type exactly), used ONLY to
    exercise PPO's training mechanics (batch composition, advantage
    variance, gradient flow across many updates) at near-zero cost per
    "evaluation" -- seconds instead of ~15-65s of real ngspice per step.
  - a landscape with a declared "good" region (near a fixed center point)
    and "bad" regions (far from it, including a synthetic failure zone),
    so PPO has an actual, cheap learning problem to solve before ever
    touching real SPICE again.

Real ngspice (via simulator.receiver.evaluate_receiver, the default
`ReceiverRLAdapter` evaluator) remains the ONLY authoritative source for any
claim about circuit performance. Nothing trained or measured against this
synthetic benchmark should be cited as evidence about the receiver itself --
only about whether the PPO implementation can learn at all. See
experiments/train_autockt.py's `--backend synthetic` flag, which defaults to
`real` (unchanged existing behavior) and must be explicitly requested.

============================================================================
WHY THIS IS ARCHITECTURALLY SAFE
============================================================================

simulator/rl_adapter.py already accepts any callable matching
evaluate_receiver's signature as its `evaluator` constructor argument (this
is the pre-existing, unmodified "stable API" boundary between the ML side
and the simulator wrapper). This module supplies exactly one such callable.
No other file needs to change, and nothing above the evaluator boundary
(rl/autockt_env.py, rl/autockt_state.py, rl/autockt_action.py,
rl/autockt_reward.py, rl/ppo_agent.py, rl/trainer.py) has any dependency on
whether the evaluator behind ReceiverRLAdapter is real ngspice or this
synthetic function -- they only ever consume the public
ReceiverRLAdapter.step() contract.
"""

from __future__ import annotations

import math
from typing import Mapping

from simulator.config import SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters, StageResult
from simulator.rl_adapter import ACTION_BOUNDS

# A declared label distinct from any real ngspice failure stage name
# ("dc", "ac", "ctle_transient", "channel", "noise", "hd3", "transient",
# "setup", "internal") so synthetic failures can never be mistaken for a
# real SPICE failure mode when reading logs.
SYNTHETIC_FAILURE_STAGE = "synthetic_out_of_region"

# Arbitrary, disclosed "good region" center in NORMALIZED parameter space
# (each of the 5 parameters mapped to [0, 1] via its own ACTION_BOUNDS
# lower/upper/scale -- reusing the existing, unmodified bounds purely as a
# coordinate system, not simulating anything). This point is not derived
# from any real successful design; it is simply the midpoint of the grid.
SYNTHETIC_CENTER: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 0.5)

# Beyond this Euclidean distance from the center (max possible distance in
# [0,1]^5 from the midpoint is sqrt(5 * 0.5**2) = sqrt(1.25) =~ 1.118), the
# synthetic evaluator reports a failure, mirroring the real simulator's
# fail-fast behavior without claiming to model any particular SPICE failure
# mechanism.
SYNTHETIC_FAILURE_DISTANCE = 0.9

# Linear interpolation endpoints for each of the 4 AutoCkt reward specs
# (rl/target_spec.py::SPEC_NAMES), as a function of "goodness" in [0, 1]
# (1.0 exactly at the center, decaying to 0.0 at distance 1.0). Chosen only
# so that the interpolated values cross HARD_TARGET_THRESHOLDS
# (height=0.8, width=0.6, margin=0.35, power=0.015) somewhere in the middle
# of the goodness range -- i.e. purely to make the existing, unmodified
# hard target reachable near the center and unreachable far from it. These
# numbers are not measurements of anything.
_HEIGHT_RANGE = (0.05, 1.20)
_WIDTH_RANGE = (0.10, 0.90)
_MARGIN_RANGE = (-0.10, 0.60)
_POWER_RANGE = (0.030, 0.002)  # smaller-is-better: HIGH goodness -> LOW power


def _lerp(bounds: tuple[float, float], t: float) -> float:
    low, high = bounds
    return low + (high - low) * t


def _normalized_position(parameters: ReceiverParameters) -> list[float]:
    """Maps the 5 physical parameters into [0, 1]^5 using the existing,
    unmodified simulator.rl_adapter.ACTION_BOUNDS (lower, upper, scale) --
    read-only reuse of published bounds, not a model of circuit behavior.
    """

    position = []
    for name, lower, upper, scale in ACTION_BOUNDS:
        value = getattr(parameters, name)
        if scale == "log":
            fraction = math.log(value / lower) / math.log(upper / lower)
        else:
            fraction = (value - lower) / (upper - lower)
        position.append(min(1.0, max(0.0, fraction)))
    return position


def _distance_from_center(position: list[float]) -> float:
    return math.sqrt(sum((p - c) ** 2 for p, c in zip(position, SYNTHETIC_CENTER)))


def synthetic_goodness(parameters: ReceiverParameters) -> float:
    """Public helper (used by tests) returning the [0, 1] "goodness" score:
    1.0 exactly at SYNTHETIC_CENTER, decaying linearly to 0.0 at distance
    1.0 or beyond. Deterministic and a pure function of `parameters`.
    """

    distance = _distance_from_center(_normalized_position(parameters))
    return max(0.0, 1.0 - distance)


def synthetic_evaluate_receiver(
    parameters: ReceiverParameters,
    conditions: SimulationConditions = SimulationConditions(),
    fidelity: EvaluationFidelity = EvaluationFidelity.TRAINING,
    **_kwargs: Mapping[str, object],
) -> ReceiverEvaluation:
    """Drop-in synthetic replacement for simulator.receiver.evaluate_receiver.

    Same call signature and ReceiverEvaluation return type; NOT a circuit
    simulation. See module docstring. `**_kwargs` absorbs any keyword
    arguments the real evaluator accepts (e.g. channel_path, cache) so this
    function can be substituted via ReceiverRLAdapter(evaluator=...,
    evaluator_kwargs=...) without those kwargs causing a TypeError; they are
    otherwise ignored.
    """

    position = _normalized_position(parameters)
    distance = _distance_from_center(position)
    evaluation_id = "synthetic:" + ",".join(f"{p:.6f}" for p in position)

    if distance > SYNTHETIC_FAILURE_DISTANCE:
        stage = StageResult(SYNTHETIC_FAILURE_STAGE, False, 0.0, metrics={})
        return ReceiverEvaluation(
            False, parameters, conditions, fidelity, (stage,), {},
            SYNTHETIC_FAILURE_STAGE, 0.0, evaluation_id, {}, False,
        )

    goodness = max(0.0, 1.0 - distance)
    metrics = {
        "dfe_locked_phase_eye_height_v": _lerp(_HEIGHT_RANGE, goodness),
        "dfe_eye_width_ui": _lerp(_WIDTH_RANGE, goodness),
        "dfe_min_margin_v": _lerp(_MARGIN_RANGE, goodness),
        "ctle_power_w": _lerp(_POWER_RANGE, goodness),
    }
    stage = StageResult("synthetic", True, 0.0, metrics=metrics)
    return ReceiverEvaluation(
        True, parameters, conditions, fidelity, (stage,), metrics,
        None, 0.0, evaluation_id, {}, False,
    )


# [NEBULA ADAPTATION -- PPO model-improvement study, "Synthetic RL
# ablations," in-scope] a second, additive synthetic evaluator alongside
# synthetic_evaluate_receiver (unchanged, byte-for-byte, above): adds a
# THIRD region -- reached transient, real (possibly poor) metrics
# computed, but a hard gate failed (failure_stage="transient", the same
# label the real simulator uses) -- between the "good" and "no
# information at all" regions. The original function only ever produces a
# binary good/no-information outcome, which cannot exercise reward_v2's
# graded-vs-no-information distinction (Required change 3) at all. Not a
# claim about circuit behavior, same as the module docstring above.
SYNTHETIC_TRANSIENT_REACHED_DISTANCE = 1.05  # > SYNTHETIC_FAILURE_DISTANCE (0.9)


def synthetic_evaluate_receiver_graded(
    parameters: ReceiverParameters,
    conditions: SimulationConditions = SimulationConditions(),
    fidelity: EvaluationFidelity = EvaluationFidelity.TRAINING,
    **_kwargs: Mapping[str, object],
) -> ReceiverEvaluation:
    """Same landscape as synthetic_evaluate_receiver, with one addition:
    the distance band (SYNTHETIC_FAILURE_DISTANCE, SYNTHETIC_TRANSIENT_REACHED_DISTANCE]
    reports success=False, failure_stage="transient" (matching the real
    simulator's own label so reward_v2/graded_autockt_reward's existing
    transient-stage gating activates unmodified), WITH real (lerp'd, just
    not good enough) metrics attached -- unlike the pure no-information
    region beyond it.
    """

    position = _normalized_position(parameters)
    distance = _distance_from_center(position)
    evaluation_id = "synthetic-graded:" + ",".join(f"{p:.6f}" for p in position)
    goodness = max(0.0, 1.0 - distance)
    metrics = {
        "dfe_locked_phase_eye_height_v": _lerp(_HEIGHT_RANGE, goodness),
        "dfe_eye_width_ui": _lerp(_WIDTH_RANGE, goodness),
        "dfe_min_margin_v": _lerp(_MARGIN_RANGE, goodness),
        "ctle_power_w": _lerp(_POWER_RANGE, goodness),
    }

    if distance > SYNTHETIC_TRANSIENT_REACHED_DISTANCE:
        stage = StageResult(SYNTHETIC_FAILURE_STAGE, False, 0.0, metrics={})
        return ReceiverEvaluation(
            False, parameters, conditions, fidelity, (stage,), {},
            SYNTHETIC_FAILURE_STAGE, 0.0, evaluation_id, {}, False,
        )
    if distance > SYNTHETIC_FAILURE_DISTANCE:
        stage = StageResult("transient", False, 0.0, metrics=metrics)
        return ReceiverEvaluation(
            False, parameters, conditions, fidelity, (stage,), metrics,
            "transient", 0.0, evaluation_id, {}, False,
        )
    stage = StageResult("synthetic", True, 0.0, metrics=metrics)
    return ReceiverEvaluation(
        True, parameters, conditions, fidelity, (stage,), metrics,
        None, 0.0, evaluation_id, {}, False,
    )
