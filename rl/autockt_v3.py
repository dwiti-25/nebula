"""Eight-head PPO state/reward. Missing measurements never earn feasibility."""
from functools import lru_cache
from pathlib import Path
import math

from .autockt_state import spec_error_vector, target_reference_vector
from .autockt_state_v2 import FAILURE_STAGE_VOCAB
from .autockt_reward_v2 import RewardResult, reward_v2
from .target_spec import SPEC_NAMES
from simulator.channel import ChannelPortMap, load_s4p
from simulator.receiver import SYNTHETIC_CHANNEL

METRICS_V3 = tuple(SPEC_NAMES) + ("peaking_db", "gain_100mhz_db", "gain_2p5ghz_db", "output_common_mode_v", "dfe_error_count")
SCALES_V3 = (1.0, 1.0, 1.0, 0.015, 12.0, 30.0, 30.0, 1.8, 1.0)
STATE_DIM_V3 = 8 + 8 + 2 * len(METRICS_V3) + len(FAILURE_STAGE_VOCAB) + 4 + 2
REWARD_V3_VERSION = "reward_v3"

def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

@lru_cache(maxsize=16)
def _channel_features(path, ports, modified_ns):
    channel = load_s4p(path, port_map=ChannelPortMap(*ports))
    return tuple(channel.insertion_loss_db(f) / 30.0 for f in (1.25e9, 2.5e9, 5e9)) + (channel.bulk_delay_s() / 1e-8,)

def build_state_v3(metrics, target, indices, *, failure_stage, parameters, channel_kwargs):
    if len(indices) != 8:
        raise ValueError("v3 state requires eight parameter indices")
    clean = {k: float(v) for k, v in metrics.items() if finite(v)}
    path = Path(channel_kwargs.get("channel_path", SYNTHETIC_CHANNEL)).resolve()
    port_map = channel_kwargs.get("channel_port_map", ChannelPortMap())
    from dataclasses import astuple
    channel = _channel_features(str(path), astuple(port_map), path.stat().st_mtime_ns)
    stage = failure_stage if failure_stage in FAILURE_STAGE_VOCAB else "internal"
    # Matched pair channel area only, deliberately not total layout area.
    area = 2 * parameters["mos_width_um"] * parameters["mos_length_um"] * parameters["mos_multiplier"]
    state = (spec_error_vector(clean, target) + target_reference_vector(target) + tuple(indices)
        + tuple(max(-10.0, min(10.0, clean.get(k, 0.0) / s)) for k, s in zip(METRICS_V3, SCALES_V3))
        + tuple(float(k in clean) for k in METRICS_V3)
        + tuple(float(stage == entry) for entry in FAILURE_STAGE_VOCAB)
        + channel + (math.log1p(area) / math.log1p(3200.0), 0.0))
    assert len(state) == STATE_DIM_V3
    return state

def reward_v3(metrics, target, *, success, failure_stage):
    clean = {k: float(v) for k, v in metrics.items() if finite(v)}
    valid = all(k in clean for k in SPEC_NAMES)
    base = reward_v2(clean, target, metrics_valid=valid, success=success, failure_stage=failure_stage)
    peaking_ok = "peaking_db" in clean and 3.0 <= clean["peaking_db"] <= 12.0
    errors_ok = "dfe_error_count" in clean and clean["dfe_error_count"] == 0
    passed = success and valid and base.strict_target_pass and peaking_ok and errors_ok
    components = dict(base.components)
    if not peaking_ok:
        components["peaking_compliance"] = -1.0
    if not errors_ok:
        components["zero_errors"] = -1.0
    total = base.total if passed else min(-0.01, sum(min(0.0, v) for v in components.values()))
    if not valid:
        total = min(total, base.total)
    # A small, explicitly PARTIAL area regularizer, never a feasibility gate.
    if "partial_mos_channel_area_um2" in clean:
        penalty = -0.02 * math.log1p(clean["partial_mos_channel_area_um2"]) / math.log1p(3200.0)
        components["partial_mos_area_regularizer"] = penalty
        total += penalty
    return RewardResult(total=total, autockt_terminal_success=passed, strict_target_pass=passed,
        components=components, constraint_margins=base.constraint_margins,
        available_metrics=tuple(clean), failure_stage=failure_stage)
