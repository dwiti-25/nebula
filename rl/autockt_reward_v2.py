"""Structured reward v2 -- additive, versioned. rl/autockt_reward.py
(autockt_reward, the v1 formula PPO's historical results were trained
with) is completely unmodified; every existing caller keeps its exact
behavior. reward_v2 is a new, richer, opt-in alternative.

Key differences from v1 (see RewardResult below for the full breakdown):
  - Never interprets a missing/unmeasured metric as zero -- components and
    constraint_margins only ever contain metrics that were genuinely
    measured this step (per rl/autockt_state_v2.py's validity mask).
  - A simulator failure (metrics_valid=False) is scored strictly below
    every valid-design total (structural floor, see
    NO_INFORMATION_FLOOR_V2), never conflated with a graded but valid
    near-miss.
  - DC/AC-stage metrics (ctle_power_w, peaking_db) contribute graded
    constraint_margins even when the 4 target specs were never reached.
  - Distinguishes autockt_terminal_success (the existing -0.02 relative-
    error tolerance v1 already uses) from strict_target_pass (an exact,
    zero-tolerance boundary check against the raw achieved values) --
    these are DIFFERENT criteria and must not be conflated.
  - A strict pass gets a bounded overshoot bonus on top of TERMINAL_BONUS,
    so near-feasible designs are not all collapsed to the identical score
    the way v1's own (tested, locked) "overshoot is not penalized beyond
    satisfaction" behavior does -- a deliberate, disclosed DIFFERENCE from
    v1, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from .autockt_reward import TERMINAL_BONUS, autockt_reward
from .autockt_state import lookup, signed_relative_error
from .autockt_state_v2 import EARLY_STAGE_MARGIN_METRICS
from .target_spec import EXISTING_THRESHOLDS, SPEC_DIRECTIONS, SPEC_NAMES, TargetSpec

REWARD_V2_VERSION = "autockt_reward_v2"

# The official AC peaking gate (simulator/receiver.py::_run_ac, unmodified,
# 3-12 dB required) -- reused verbatim as the reference band for a peaking
# reward component, since TargetSpec itself deliberately has no peaking
# field (rl/target_spec.py's own documented reason: peaking is banded, not
# monotone larger/smaller-is-better). Not a new number.
PEAKING_BAND_DB: tuple[float, float] = (3.0, 12.0)
PEAKING_BAND_CENTER_DB = sum(PEAKING_BAND_DB) / 2.0
PEAKING_BAND_HALF_WIDTH_DB = (PEAKING_BAND_DB[1] - PEAKING_BAND_DB[0]) / 2.0

# Structural floor: strictly below any reachable valid-design total. The
# valid-failing branch sums <= 4 SPEC_NAMES components (each roughly
# bounded via lookup()'s ~[-1,1] range) plus one peaking component bounded
# to [-2, 0] below -- worst case around -6. -20.0 leaves a wide, deliberate
# safety margin, the same structural role (not a tuned weight) as
# rl/autockt_reward.py::GRADED_NO_INFORMATION_FLOOR.
NO_INFORMATION_FLOOR_V2 = -20.0

# Bounded overshoot bonus on a strict pass -- capped so it can never
# approach a second full TERMINAL_BONUS, only order successes by margin
# quality. A simple, symmetric, undisclosed-weight-free formula: each of
# the 4 SPEC_NAMES contributes its own (already-computed, not re-tuned)
# positive relative-error term, summed and scaled, then capped.
OVERSHOOT_BONUS_SCALE = 0.1
OVERSHOOT_BONUS_CAP = 2.0


@dataclass(frozen=True)
class RewardResult:
    total: float
    autockt_terminal_success: bool  # v1's own -0.02-tolerant criterion (autockt_reward >= TERMINAL_BONUS)
    strict_target_pass: bool        # exact boundary check, zero tolerance, independent of the above
    components: dict[str, float] = field(default_factory=dict)
    constraint_margins: dict[str, float] = field(default_factory=dict)
    available_metrics: tuple[str, ...] = ()
    failure_stage: Optional[str] = None


def _early_stage_margins(metrics: Mapping[str, float]) -> dict[str, float]:
    """DC/AC-stage margins, present only for metrics genuinely in `metrics`
    (never fabricated). ctle_power_w and peaking_db are both computed by
    their own stage before that stage's own violation check (see
    rl/autockt_state_v2.py docstring) -- so they can be present even when
    the 4 SPEC_NAMES targets were never reached.
    """

    margins: dict[str, float] = {}
    if "ctle_power_w" in metrics:
        margins["ctle_power_w_margin"] = EXISTING_THRESHOLDS["ctle_power_w"] - metrics["ctle_power_w"]
    if "peaking_db" in metrics:
        distance = abs(metrics["peaking_db"] - PEAKING_BAND_CENTER_DB) - PEAKING_BAND_HALF_WIDTH_DB
        margins["peaking_db_margin"] = -distance  # >= 0 inside the band, negative outside
    return margins


def _peaking_component(metrics: Mapping[str, float]) -> Optional[float]:
    if "peaking_db" not in metrics:
        return None
    distance = abs(metrics["peaking_db"] - PEAKING_BAND_CENTER_DB) - PEAKING_BAND_HALF_WIDTH_DB
    if distance <= 0:
        return 0.0  # inside the band -- no penalty, matching autockt_reward's own "no overshoot credit" convention
    return -min(2.0, distance / PEAKING_BAND_HALF_WIDTH_DB)


def reward_v2(
    metrics: Mapping[str, float],
    target: TargetSpec,
    *,
    metrics_valid: bool,
    success: bool,
    failure_stage: Optional[str],
) -> RewardResult:
    constraint_margins = _early_stage_margins(metrics)

    if not metrics_valid:
        return RewardResult(
            total=NO_INFORMATION_FLOOR_V2, autockt_terminal_success=False, strict_target_pass=False,
            components={}, constraint_margins=constraint_margins, available_metrics=(), failure_stage=failure_stage,
        )

    components: dict[str, float] = {}
    strict_pass = True
    for name in SPEC_NAMES:
        achieved = metrics.get(name, 0.0)
        goal = getattr(target, name)
        relative_error = signed_relative_error(name, lookup(achieved, goal))
        components[name] = relative_error
        if SPEC_DIRECTIONS[name] == "larger_is_better":
            strict_pass = strict_pass and achieved >= goal
        else:
            strict_pass = strict_pass and achieved <= goal
    available_metrics = tuple(SPEC_NAMES)

    peaking_component = _peaking_component(metrics)
    if peaking_component is not None:
        components["peaking_db"] = peaking_component
        available_metrics = available_metrics + ("peaking_db",)

    autockt_terminal_success = autockt_reward(metrics, target, success=True) >= TERMINAL_BONUS

    if strict_pass:
        overshoot = sum(value for name, value in components.items() if name in SPEC_NAMES and value > 0)
        bonus = min(OVERSHOOT_BONUS_CAP, overshoot * OVERSHOOT_BONUS_SCALE)
        total = TERMINAL_BONUS + bonus
    else:
        total = sum(value for value in components.values() if value < 0)
        # A failure at the `transient` stage that nonetheless has no
        # negative component (e.g. failed only on dfe_error_rate, which
        # this reward does not track) must still read as a failure, not a
        # 0.0 that could be misread as "as good as passing."
        if total >= 0.0:
            total = NO_INFORMATION_FLOOR_V2 / 2.0  # worse than any graded failure, better than no-information

    return RewardResult(
        total=total, autockt_terminal_success=autockt_terminal_success, strict_target_pass=strict_pass,
        components=components, constraint_margins=constraint_margins, available_metrics=available_metrics,
        failure_stage=failure_stage,
    )
