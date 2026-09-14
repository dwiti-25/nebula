"""Discrete parameter grid over NEBULA's five continuous RL parameters.

[AUTOCKT-REPLICATED] grid construction and action-delta mechanics:
autockt/envs/ngspice_vanilla_opamp.py (github.com/ksettaluri6/AutoCkt) builds
each parameter's grid as

    param_vec = np.arange(value[0], value[1], value[2])

i.e. fixed-step LINEAR spacing from a (start, stop, step) triple, and steps
an index by one of `self.action_meaning = [-1, 0, 2]`, clipped to
`[0, len(param_vec) - 1]`. This module replicates that mechanic exactly
(linear numpy.arange grid, {-1, 0, +2} index deltas, clip-to-bounds) for
NEBULA's five parameters. Per explicit direction, this first implementation
does NOT use log-spaced grids even though CDEG/RLOAD/RDEG/ITAIL span 2-3
orders of magnitude and simulator/rl_adapter.py's own *continuous* action
encoding uses log spacing for exactly those four (see ACTION_BOUNDS below,
`scale="log"`). That `scale` flag is read here only to stay a single source
of truth for (lower, upper) bounds; it is otherwise ignored -- the grid
itself is linear for all five parameters. If CDEG's resulting grid
resolution proves too coarse near its low end, a log-spaced grid variant
will be added later as an explicitly labeled [NEBULA ADAPTATION], not by
silently changing this file's semantics.

[ROUGH-SCALE ADAPTATION] grid resolution: AutoCkt hand-picks per-parameter
steps in its YAML (e.g. 6 transistor multipliers on 1-99 step 1 -> 99 points
each; `cc` on 0.1pF-10pF step 0.1pF -> 99 points). NEBULA has no equivalent
hand-tuned step. DEFAULT_GRID_POINTS=21 below is a deliberately small,
explicitly-labeled choice for the first end-to-end milestone: with 5
parameters x 3 discrete choices each, the joint action space is 3**5=243
combinations regardless of per-parameter grid resolution, so grid size only
affects how finely a single step moves a single parameter. 21 points keeps
grid construction/inspection trivial while each single-parameter step still
covers a meaningful fraction of that parameter's range, appropriate for a
rough smoke-scale milestone where each SPICE evaluation costs ~15-65s
wall-clock (measured directly in this session; see
docs/autockt-mapping.md, "SPICE timing measurement"). This is not a claim of
AutoCkt's own (~99-point) resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from simulator.rl_adapter import ACTION_BOUNDS

# [AUTOCKT-REPLICATED] action_meaning = [-1, 0, 2] in ngspice_vanilla_opamp.py.
ACTION_DELTAS: tuple[int, int, int] = (-1, 0, 2)

# [ROUGH-SCALE ADAPTATION] see module docstring.
DEFAULT_GRID_POINTS = 21

# [NEBULA ADAPTATION] `spacing="log"` option (see build_parameter_grids
# below) -- additive, not part of the AutoCkt replication above. AutoCkt's
# own repo only ever builds linear np.arange grids; this is a NEBULA-only
# alternative added because real-SPICE evidence (a 57-evaluation hard-target
# training run, see docs/autockt-mapping.md) showed the linear grid's fixed
# absolute step size is disproportionately large near the low end of the
# log-scale parameters, reliably knocking a near-optimal design out of its
# narrow feasible band with a single action. "linear" remains the default so
# existing behavior/tests are unchanged unless this is explicitly requested.
GRID_SPACINGS: tuple[str, ...] = ("linear", "log")
DEFAULT_GRID_SPACING = "linear"

PARAMETER_NAMES: tuple[str, ...] = tuple(name for name, *_ in ACTION_BOUNDS)


@dataclass(frozen=True)
class ParameterGrid:
    """One parameter's discrete, linearly spaced physical-value grid."""

    name: str
    values: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.values)

    def clip_index(self, index: int) -> int:
        """[AUTOCKT-REPLICATED] mirrors `np.clip(idx, 0, len(param_vec) - 1)`."""
        return max(0, min(len(self.values) - 1, index))

    def value_at(self, index: int) -> float:
        return self.values[self.clip_index(index)]

    def nearest_index(self, physical_value: float) -> int:
        array = np.asarray(self.values)
        return int(np.argmin(np.abs(array - physical_value)))

    def normalized_index(self, index: int) -> float:
        """[NEBULA ADAPTATION] maps a grid index to [-1, 1] (endpoints exactly
        -1.0/+1.0, 0 at the midpoint), for use as a PPO state feature scaled
        comparably to the other (roughly [-1, 1]-bounded) state components --
        see rl/autockt_env.py, which uses this instead of the raw index when
        building the AutoCkt-shaped state. AutoCkt's own state leaves
        cur_params_idx un-normalized (verified from source, see
        rl/autockt_state.py); this method is not used by rl/autockt_state.py
        itself, which is unchanged and still documents/accepts raw indices
        exactly as before -- only rl/autockt_env.py's *caller* of build_state
        was changed to feed this instead.
        """

        if len(self.values) <= 1:
            return 0.0
        return 2.0 * self.clip_index(index) / (len(self.values) - 1) - 1.0


def build_parameter_grids(
    points_per_parameter: int = DEFAULT_GRID_POINTS,
    *,
    spacing: str = DEFAULT_GRID_SPACING,
    version: str = "v1",
) -> dict[str, ParameterGrid]:
    """[AUTOCKT-REPLICATED] `np.arange(lower, upper, step)` per parameter
    when `spacing="linear"` (the default -- byte-for-byte the original,
    unchanged behavior).

    [NEBULA ADAPTATION] `spacing="log"` builds a geometrically-spaced grid
    (`np.geomspace`) instead, but *only* for the four parameters
    ACTION_BOUNDS itself already marks `scale="log"` (rload_ohm, rdeg_ohm,
    cdeg_f, itail_a). `dfe_tap_v` (`scale="linear"`, and its range spans
    negative to positive, where geometric spacing is undefined) always keeps
    the original linear `np.arange` grid regardless of `spacing`. Note
    `np.geomspace` includes both endpoints, unlike `np.arange`'s
    exclusive-upper-bound linear grid -- a deliberate, minor difference, not
    an oversight.

    Bounds (lower, upper) are read verbatim from the existing, untouched
    simulator.rl_adapter.ACTION_BOUNDS -- not duplicated or re-derived here,
    so this grid can never drift out of sync with the simulator's own
    validated parameter limits.
    """

    if spacing not in GRID_SPACINGS:
        raise ValueError(f"spacing must be one of {GRID_SPACINGS}, got {spacing!r}")

    from simulator.design_schema import parameter_bounds
    if points_per_parameter < 2:
        raise ValueError("grid requires at least two points")
    grids: dict[str, ParameterGrid] = {}
    for name, lower, upper, scale in parameter_bounds(version):
        if scale == "integer":
            values = tuple(range(int(lower), int(upper) + 1))
        elif name.startswith("mos_"):
            # SKY130 dimensions in microns, snapped to a 1 nm sizing grid.
            values = tuple(sorted(set(round(float(v), 3) for v in np.linspace(lower, upper, points_per_parameter)) | {10.0 if name == "mos_width_um" else 0.15}))
        elif spacing == "log" and scale == "log":
            values = tuple(float(v) for v in np.geomspace(lower, upper, points_per_parameter))
        else:
            step = (upper - lower) / points_per_parameter
            values = tuple(float(v) for v in np.arange(lower, upper, step))
        grids[name] = ParameterGrid(name=name, values=values)
    return grids


# [NEBULA ADAPTATION] initial parameter-grid state.
#
# AutoCkt's own env.reset() uses a fixed, hardcoded starting index array
# (`self.cur_params_idx = np.array([33,33,33,33,33,14,20])` in
# ngspice_vanilla_opamp.py) -- i.e. a *fixed, non-random* starting point is
# the AutoCkt methodology [AUTOCKT-REPLICATED]. NEBULA has different
# parameters/grids, so the literal indices cannot be ported; a NEBULA-specific
# fixed starting point must be chosen.
#
# The README's documented "baseline circuit" (RLOAD=1kOhm, RDEG=1kOhm,
# CDEG=0.5pF, ITAIL=100uA) is also the literal default of
# simulator.receiver.ReceiverParameters() (receiver.py:54-58), so it exists
# in-repo, not just in prose. However, it was directly measured in this
# session via ReceiverRLAdapter.step() at TRAINING fidelity and FAILS at the
# DC stage (does not clear the DC rail-headroom/offset gates), so it is not
# a feasible starting point despite being the in-code default -- do not
# assume documented defaults are feasible without checking.
#
# Instead, VERIFIED_INITIAL_PARAMETERS below is a real, feasible design
# already on record in results/receiver_random_search_20_seed123.jsonl
# (candidate_index 8, success=True, reward=100.0), re-confirmed by a fresh
# SPICE re-evaluation in this session (see docs/autockt-mapping.md, "SPICE
# timing measurement"): dfe_locked_phase_eye_height_v=1.522V,
# dfe_eye_width_ui=0.87, dfe_min_margin_v=0.517V, ctle_power_w=1.085mW,
# dfe_error_count=0.
VERIFIED_INITIAL_PARAMETERS: Mapping[str, float] = {
    "rload_ohm": 2342.472156058411,
    "rdeg_ohm": 822.3558626926603,
    "cdeg_f": 9.999375862792168e-13,
    "itail_a": 0.0006028331705063624,
    "dfe_tap_v": -0.011090823885148815,
}


def verified_initial_indices(grids: Mapping[str, ParameterGrid]) -> tuple[int, ...]:
    """Nearest-grid-index encoding of VERIFIED_INITIAL_PARAMETERS."""

    from simulator.receiver import ReceiverParameters
    defaults = ReceiverParameters()
    return tuple(grids[name].nearest_index(VERIFIED_INITIAL_PARAMETERS.get(name, getattr(defaults, name))) for name in grids)


def quantize_parameters(parameters, grids):
    """Use identical physical grids for PPO, Random Search and CEM comparisons."""
    from dataclasses import replace
    return replace(parameters, **{name: int(grid.value_at(grid.nearest_index(getattr(parameters, name))))
        if name == "mos_multiplier" else grid.value_at(grid.nearest_index(getattr(parameters, name)))
        for name, grid in grids.items()})
