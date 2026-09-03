"""AutoCkt-style discrete action decoding.

[AUTOCKT-REPLICATED] five independent Discrete(3) heads (one per NEBULA
parameter: RLOAD, RDEG, CDEG, ITAIL, DFE_TAP), each selecting an index delta
from parameter_grid.ACTION_DELTAS = (-1, 0, +2). Matches
autockt/envs/ngspice_vanilla_opamp.py (github.com/ksettaluri6/AutoCkt):

    self.action_meaning = [-1, 0, 2]
    self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_meaning))] * len(self.params_id))
    ...
    self.cur_params_idx = self.cur_params_idx + np.array([self.action_meaning[a] for a in action])
    self.cur_params_idx = np.clip(self.cur_params_idx, 0, len(param_vec) - 1)

AutoCkt uses 7 parameter heads (6 transistor multipliers + 1 cap); NEBULA
uses 5 (RLOAD, RDEG, CDEG, ITAIL, DFE_TAP) [NEBULA ADAPTATION -- parameter
count/identity only, delta mechanics are unchanged].
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from simulator.rl_adapter import ACTION_BOUNDS

from .parameter_grid import ACTION_DELTAS, ParameterGrid, PARAMETER_NAMES


def apply_action(
    indices: Sequence[int],
    choices: Sequence[int],
    grids: Mapping[str, ParameterGrid],
    *,
    deltas: Sequence[int] = ACTION_DELTAS,
) -> tuple[int, ...]:
    """[AUTOCKT-REPLICATED] index update + clip.

    `choices[i]` selects an entry of `deltas` (ACTION_DELTAS = (-1, 0, +2)
    by default, unchanged) for parameter `PARAMETER_NAMES[i]`.

    [NEBULA ADAPTATION] `deltas` is additive, optional, and defaults to the
    exact AutoCkt-replicated ACTION_DELTAS -- every existing call site is
    byte-for-byte unchanged. Only the PPO model-improvement study's
    optional symmetric-action ablation (Required change 7, variant 5)
    passes a different tuple (e.g. (-1, 0, +1)), instrumented and compared
    against, never silently substituted for the AutoCkt-replicated default.
    """

    if len(indices) != len(PARAMETER_NAMES):
        raise ValueError(f"expected {len(PARAMETER_NAMES)} indices, got {len(indices)}")
    if len(choices) != len(PARAMETER_NAMES):
        raise ValueError(f"expected {len(PARAMETER_NAMES)} choices, got {len(choices)}")
    if len(deltas) != 3:
        raise ValueError("deltas must have exactly 3 entries (one per Discrete(3) choice)")
    new_indices = []
    for name, index, choice in zip(PARAMETER_NAMES, indices, choices):
        if choice not in (0, 1, 2):
            raise ValueError("each choice must be 0, 1, or 2 (an index into deltas)")
        delta = deltas[choice]
        new_indices.append(grids[name].clip_index(index + delta))
    return tuple(new_indices)


def indices_to_normalized_action(
    indices: Sequence[int],
    grids: Mapping[str, ParameterGrid],
) -> tuple[float, ...]:
    """ML-side glue, not part of the AutoCkt replication: converts discrete
    grid indices -> physical values -> the *existing, untouched*
    simulator.rl_adapter continuous [-1, 1] action encoding required by
    ReceiverRLAdapter.step(). This inverts
    simulator.rl_adapter.normalized_action_to_parameters's forward formula
    (same log/linear `scale` per ACTION_BOUNDS) purely so the discrete grid
    can drive the unmodified adapter -- simulator/rl_adapter.py is only
    read, never changed.
    """

    if len(indices) != len(PARAMETER_NAMES):
        raise ValueError(f"expected {len(PARAMETER_NAMES)} indices, got {len(indices)}")
    action = []
    for (name, lower, upper, scale), index in zip(ACTION_BOUNDS, indices):
        physical = grids[name].value_at(index)
        if scale == "log":
            fraction = math.log(physical / lower) / math.log(upper / lower)
        else:
            fraction = (physical - lower) / (upper - lower)
        fraction = min(1.0, max(0.0, fraction))
        action.append(2.0 * fraction - 1.0)
    return tuple(action)
