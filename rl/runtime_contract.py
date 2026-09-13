"""Shared channel and version contract for training and inference."""
from dataclasses import asdict
from pathlib import Path
from simulator.channel import ChannelPortMap, load_s4p, validate_s4p_channel
from simulator.receiver import SYNTHETIC_CHANNEL
from simulator.design_schema import parameter_names
from .autockt_state import STATE_DIM
from .autockt_state_v2 import STATE_DIM_V2
from .autockt_v3 import STATE_DIM_V3

STATE_DIMS = {"v1": STATE_DIM, "v2": STATE_DIM_V2, "v3": STATE_DIM_V3}

def add_channel_arguments(parser):
    parser.add_argument("--channel", type=Path, default=SYNTHETIC_CHANNEL)
    parser.add_argument("--channel-ports", type=int, nargs=4, default=(1, 2, 3, 4), metavar=("TXP", "TXN", "RXP", "RXN"))

def channel_kwargs(args):
    return {"channel_path": args.channel, "channel_port_map": ChannelPortMap(*args.channel_ports)}

def runtime_contract(version, grids, channel_path=SYNTHETIC_CHANNEL, channel_port_map=ChannelPortMap(), backend="real"):
    channel = load_s4p(channel_path, port_map=channel_port_map)
    validate_s4p_channel(channel)
    contract = {"state_schema": version, "state_dim": STATE_DIMS[version],
        "reward_schema": f"reward_{version}" if version != "v1" else "autockt_reward_v1",
        "action_schema_version": 3 if version == "v3" else 1,
        "parameter_names": list(parameter_names(version)),
        "parameter_grid": {k: list(v.values) for k, v in grids.items()},
        "action_deltas": [-1, 0, 2], "backend": backend,
        "channel_checksum": channel.checksum, "channel_port_map": asdict(channel_port_map),
        "area_status": "partial_mos_channel_only", "partial_area_reward_weight": 0.02,
        "dfe_implementation": "behavioral"}
    if version == "v3":
        import sys
        import numpy
        import torch
        contract["toolchain"] = {"python": sys.version.split()[0], "numpy": numpy.__version__, "torch": str(torch.__version__)}
        if backend == "real":
            from simulator.config import Sky130Config
            from simulator.ngspice import ngspice_identity
            from simulator.provenance import spice_dependency_fingerprint
            contract["toolchain"].update(ngspice_identity())
            contract["toolchain"]["pdk_checksum"] = spice_dependency_fingerprint(Sky130Config().resolve_model_library())
    return contract

def validate_inference_contract(metadata, expected):
    if expected["state_schema"] != "v3":
        if metadata and metadata.get("state_schema") != expected["state_schema"]:
            raise ValueError("Checkpoint state schema does not match requested PPO version")
        return
    if not metadata:
        raise ValueError("PPO v3 requires a metadata-bearing v3 policy export, not a legacy bare checkpoint")
    contract = metadata.get("hyperparameters", {}).get("runtime_contract", metadata.get("runtime_contract", metadata))
    for key in ("state_schema", "state_dim", "reward_schema", "action_schema_version", "parameter_names", "parameter_grid", "action_deltas"):
        if contract.get(key) != expected[key]:
            raise ValueError(f"Incompatible v3 checkpoint: {key} differs")
    # A different channel is allowed for explicit inference/generalization;
    # both identities are retained in the run graph. Resume is stricter.

def grids_from_contract(metadata):
    from .parameter_grid import ParameterGrid
    from simulator.design_schema import parameter_bounds
    import math
    contract = metadata.get("hyperparameters", {}).get("runtime_contract", metadata.get("runtime_contract", metadata))
    if contract.get("parameter_names") != list(parameter_names("v3")):
        raise ValueError("Invalid v3 checkpoint parameter order")
    grids = {}
    for name, lower, upper, scale in parameter_bounds("v3"):
        values = contract.get("parameter_grid", {}).get(name, [])
        if (len(values) < 2 or any(not math.isfinite(v) or not lower <= v <= upper for v in values)
                or any(a >= b for a, b in zip(values, values[1:]))
                or (scale == "integer" and any(int(v) != v for v in values))):
            raise ValueError(f"Invalid v3 checkpoint grid: {name}")
        grids[name] = ParameterGrid(name, tuple(values))
    return grids
