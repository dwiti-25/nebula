"""State schema v2 -- additive, versioned. rl/autockt_state.py (v1) is
completely unmodified; PPO v1 checkpoints/pipeline behavior are unaffected.

Fixes the "artificial zeros" problem (Required change 1) and adds
validity/failure information (Required change 2), by properly consuming
signals simulator/rl_adapter.py ALREADY computes and exposes but the RL
layer previously discarded:

  - OBSERVATION_NAMES already includes a `valid_<metric>` flag per metric
    (simulator/rl_adapter.py::observation_from_evaluation) -- computed
    correctly today (e.g. `valid_ctle_power_w` is 1.0 even on a dc-stage
    failure, since _dc_metrics computes ctle_power_w before checking
    violations), but rl/autockt_env.py::metrics_from_observation only
    ever reads the first half of the observation tuple (the values),
    never the second half (the validity flags). This module reads both.
  - ReceiverEvaluation.failed_stage is already a stable, well-defined
    string (verified against every StageResult(...) call site in
    simulator/receiver.py: dc, ac, ctle_transient, channel, transient,
    noise, hd3, setup, internal, or None for success) -- already threaded
    into AutoCktStep.info["failure_stage"] but never encoded into state.

No new simulator/adapter code; this only reads existing fields.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from .autockt_state import STATE_DIM as STATE_DIM_V1, build_state
from .target_spec import TargetSpec

STATE_SCHEMA_VERSION = 2

# Every StageResult(...) name actually used in simulator/receiver.py,
# confirmed by inspection, plus None for success and "unevaluated" for a
# reset() that performed no evaluation at all (evaluate_on_reset=False) --
# a deliberately DISTINCT category from None/success: conflating "never
# measured" with "measured and passed" is exactly the artificial-zeros
# problem this state schema exists to fix. Order is fixed (one-hot index =
# vocab index) and must never be reordered once a checkpoint using it
# exists -- append-only if a new stage name is ever added.
FAILURE_STAGE_VOCAB: tuple[Optional[str], ...] = (
    None, "unevaluated", "setup", "dc", "ac", "ctle_transient", "channel", "transient", "noise", "hd3", "internal",
    # rl/synthetic_benchmark.py::SYNTHETIC_FAILURE_STAGE -- the synthetic,
    # SPICE-free RL-mechanics benchmark's own out-of-region failure label
    # (never produced by the real simulator; included so synthetic ablations
    # can use this state schema without a vocabulary error).
    "synthetic_out_of_region",
)

# Metrics that ARE genuinely available before the transient stage (the one
# that computes the 4 SPEC_NAMES targets) -- both confirmed by inspection:
# ctle_power_w is computed by _dc_metrics (simulator/receiver.py) before any
# DC violation check; peaking_db is computed by ac_metrics
# (simulator/waveform.py) before any AC violation check. Used here only as
# EXISTING-metric availability flags/margins, not new derived quantities.
EARLY_STAGE_MARGIN_METRICS: tuple[str, ...] = ("ctle_power_w", "peaking_db")

STATE_DIM_V2 = STATE_DIM_V1 + 1 + len(EARLY_STAGE_MARGIN_METRICS) + len(FAILURE_STAGE_VOCAB)


def validity_mask_from_observation(
    observation: Sequence[float], metric_observation_names: Sequence[str],
) -> dict[str, bool]:
    """Reads the `valid_<name>` half of the adapter's own observation tuple
    (simulator/rl_adapter.py::OBSERVATION_NAMES = METRIC_OBSERVATION_NAMES +
    valid_* flags) -- does not recompute anything, just decodes what the
    adapter already returned.
    """

    n = len(metric_observation_names)
    valid_slice = observation[n : 2 * n]
    return {name: bool(valid_slice[i]) for i, name in enumerate(metric_observation_names)}


def _one_hot_failure_stage(failure_stage: Optional[str]) -> tuple[float, ...]:
    if failure_stage not in FAILURE_STAGE_VOCAB:
        # An unrecognized stage name would be a genuine, actionable bug (a
        # new StageResult name added to the simulator without updating
        # FAILURE_STAGE_VOCAB) -- fail loudly rather than silently miscode it.
        raise ValueError(f"unrecognized failure_stage {failure_stage!r}; update FAILURE_STAGE_VOCAB")
    index = FAILURE_STAGE_VOCAB.index(failure_stage)
    return tuple(1.0 if i == index else 0.0 for i in range(len(FAILURE_STAGE_VOCAB)))


def build_state_v2(
    current_metrics: Mapping[str, float],
    target: TargetSpec,
    param_indices: Sequence[int],
    *,
    metrics_valid: bool,
    failure_stage: Optional[str],
) -> tuple[float, ...]:
    """Extends build_state (v1, unmodified) with:
      - 1 dim: metrics_valid (1.0/0.0) -- were the 4 SPEC_NAMES values in
        the v1 slice genuinely measured this step, or is this a reset /
        early-stage-failure state where they are NOT real circuit
        measurements (do not train as if they were)?
      - len(EARLY_STAGE_MARGIN_METRICS) dims: margin proxies for metrics
        that CAN be genuinely available even when metrics_valid=False
        (e.g. ctle_power_w survives a dc-stage failure) -- 0.0 when truly
        unavailable, never fabricated as a passing/failing value.
      - len(FAILURE_STAGE_VOCAB) dims: one-hot failure-stage encoding.

    The v1 slice itself is computed by the SAME, unmodified build_state --
    when metrics_valid=False, callers are expected to pass `current_metrics`
    unchanged from whatever they already had (typically `{}`, matching v1's
    existing reset() convention exactly) so the numeric VALUES in that slice
    are byte-identical to v1's; only the new validity/margin/stage dims
    disambiguate "this is real" from "this is not."
    """

    v1_slice = build_state(current_metrics, target, param_indices)
    margins = tuple(
        current_metrics.get(name, 0.0) if current_metrics.get(name) is not None else 0.0
        for name in EARLY_STAGE_MARGIN_METRICS
    )
    return v1_slice + (1.0 if metrics_valid else 0.0,) + margins + _one_hot_failure_stage(failure_stage)
