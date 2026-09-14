"""Version-aware frozen-policy setup shared by generalization tools."""
from .runtime_contract import STATE_DIMS, grids_from_contract, runtime_contract, validate_inference_contract
from .parameter_grid import build_parameter_grids
from .ppo_agent import PPOAgent

def load_evaluation_policy(path, *, version, grid_points, grid_spacing, seed, channel_kwargs):
    from experiments.run_autockt_pipeline import _load_checkpoint_payload
    state, metadata = _load_checkpoint_payload(path)
    grids = grids_from_contract(metadata) if version == "v3" and metadata else build_parameter_grids(grid_points, spacing=grid_spacing, version=version)
    validate_inference_contract(metadata, runtime_contract(version, grids, **channel_kwargs))
    agent = PPOAgent(state_dim=STATE_DIMS[version], num_heads=len(grids), seed=seed)
    agent.policy.load_state_dict(state)
    return state, grids, agent
