"""Controlled, matched initial-vs-final checkpoint evaluation against the
unseen midpoint target -- the generalization-verification milestone.

Reuses experiments.train_autockt._evaluate_checkpoint UNMODIFIED (the same
matched-seed, deterministic-policy methodology already validated in
docs/autockt-mapping.md sections 15 and 17): both checkpoints get a FRESH
AutoCktReceiverEnv built with the identical seed, so both see byte-identical
sequences of (randomized, since randomize_initial_state=True) starting
states and the identical target -- only the loaded policy weights differ.

Passing training_pool=(midpoint,) -- a single-element pool -- makes every
episode's target resolve to the midpoint on every reset(), since
AutoCktReceiverEnv.reset() samples via self._episode_rng.choice(target_pool)
and a 1-element pool has only one possible choice. No new code path is
introduced; this is the exact mechanism recommended (but not implemented)
in docs/autockt-mapping.md sections 16/17.

KNOWN LIMITATION (unchanged, not fixed here per "reuse the EXISTING
checkpoint machinery" instruction): _evaluate_checkpoint's returned episode
rows carry checkpoint/episode/episode_reward/spec_satisfied/steps/target
only -- no per-step parameter trajectory. "parameter trajectory if
available" in the per-case results below is therefore None for every case;
this is not a bug introduced here, it is the pre-existing row shape of the
function this script is instructed to reuse unmodified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from simulator.rl_adapter import ReceiverRLAdapter, RLBudget

from rl.parameter_grid import PARAMETER_NAMES, build_parameter_grids, verified_initial_indices
from rl.ppo_agent import PPOAgent
from rl.autockt_state import STATE_DIM
from rl.target_spec import TargetSpec

from experiments.train_autockt import _evaluate_checkpoint

# Requirement 1: the exact unseen midpoint target already defined (matches
# TargetSpec's arithmetic midpoint of from_existing_thresholds()/
# from_hard_target(), spelled out verbatim rather than re-derived, per
# instruction).
UNSEEN_MIDPOINT_TARGET = TargetSpec(
    dfe_locked_phase_eye_height_v=0.45,
    dfe_eye_width_ui=0.5,
    dfe_min_margin_v=0.175,
    ctle_power_w=0.015,
)

MATCHED_EPISODES = 10


def run_controlled_evaluation(
    *,
    initial_policy_state: dict[str, Any],
    final_policy_state: dict[str, Any],
    adapter: ReceiverRLAdapter,
    grids: dict,
    initial_indices: tuple[int, ...],
    horizon: int,
    eval_seed: int,
    agent_seed: int,
    episodes: int = MATCHED_EPISODES,
    rl_version: str = "v1",
) -> dict[str, Any]:
    """Runs the matched initial-vs-final comparison and returns the full
    numerical result (per-case rows plus aggregate win/tie/loss counts).
    Does not write to disk; callers handle persistence.
    """

    from rl.runtime_contract import STATE_DIMS
    agent = PPOAgent(state_dim=STATE_DIMS[rl_version], num_heads=len(grids), seed=agent_seed)

    common_kwargs = dict(
        adapter=adapter,
        training_pool=(UNSEEN_MIDPOINT_TARGET,),
        initial_indices=initial_indices,
        grids=grids,
        horizon=horizon,
        randomize_initial_state=True,
        seed=eval_seed,
        episodes=episodes, rl_version=rl_version,
    )

    initial_summary, initial_rows = _evaluate_checkpoint(
        label="initial", policy_state=initial_policy_state, agent=agent, **common_kwargs,
    )
    final_summary, final_rows = _evaluate_checkpoint(
        label="final", policy_state=final_policy_state, agent=agent, **common_kwargs,
    )

    if len(initial_rows) != len(final_rows):
        raise RuntimeError(
            f"matched evaluation produced different episode counts "
            f"({len(initial_rows)} vs {len(final_rows)}) -- the matched-seed "
            f"guarantee has been violated; do not trust these results"
        )

    cases: list[dict[str, Any]] = []
    wins = ties = losses = 0
    for initial_row, final_row in zip(initial_rows, final_rows):
        if initial_row["episode"] != final_row["episode"]:
            raise RuntimeError("episode index mismatch between initial/final rows")
        if initial_row["target"] != final_row["target"]:
            raise RuntimeError("target mismatch between initial/final rows -- matching failed")

        initial_reward = initial_row["episode_reward"]
        final_reward = final_row["episode_reward"]
        if final_reward > initial_reward:
            outcome = "win"
            wins += 1
        elif final_reward < initial_reward:
            outcome = "loss"
            losses += 1
        else:
            outcome = "tie"
            ties += 1

        cases.append({
            "episode": initial_row["episode"],
            "target": initial_row["target"],
            "initial_reward": initial_reward,
            "final_reward": final_reward,
            "initial_spec_satisfied": initial_row["spec_satisfied"],
            "final_spec_satisfied": final_row["spec_satisfied"],
            "initial_steps_to_satisfaction": initial_row["steps"] if initial_row["spec_satisfied"] else None,
            "final_steps_to_satisfaction": final_row["steps"] if final_row["spec_satisfied"] else None,
            "initial_steps_used": initial_row["steps"],
            "final_steps_used": final_row["steps"],
            "outcome": outcome,
            "parameter_trajectory": None,  # see module docstring: not available from _evaluate_checkpoint
        })

    n = len(cases)
    initial_satisfaction_rate = initial_summary["satisfaction_rate"]
    final_satisfaction_rate = final_summary["satisfaction_rate"]

    return {
        "target": UNSEEN_MIDPOINT_TARGET.as_dict(),
        "matched_episodes": n,
        "eval_seed": eval_seed,
        "horizon": horizon,
        "initial_satisfaction_rate": initial_satisfaction_rate,
        "final_satisfaction_rate": final_satisfaction_rate,
        "initial_mean_reward": initial_summary["mean_episode_reward"],
        "final_mean_reward": final_summary["mean_episode_reward"],
        "initial_mean_steps": sum(c["initial_steps_used"] for c in cases) / n,
        "final_mean_steps": sum(c["final_steps_used"] for c in cases) / n,
        "wins": wins, "ties": ties, "losses": losses,
        "cases": cases,
        "initial_summary_raw": initial_summary,
        "final_summary_raw": final_summary,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Controlled matched initial-vs-final checkpoint evaluation "
        "on the unseen midpoint target, via the existing _evaluate_checkpoint machinery."
    )
    parser.add_argument(
        "--initial-checkpoint", type=Path, default=None,
        help="torch.save'd initial (pre-training) policy state_dict. If omitted (the "
        "default), the initial/untrained policy is instead a FRESH PPOAgent(seed=--agent-seed) "
        "constructed on the spot -- deterministic and reproducible on demand, exactly what "
        "the pre-training policy state_dict always was, so nothing needs to have been saved "
        "for this side of the comparison.",
    )
    parser.add_argument("--final-checkpoint", type=Path, required=True, help="torch.save'd final (post-training) policy state_dict")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--grid-points", type=int, default=21)
    parser.add_argument("--grid-spacing", choices=("linear", "log"), default="log")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=MATCHED_EPISODES)
    parser.add_argument("--max-evaluations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    from rl.runtime_contract import add_channel_arguments, channel_kwargs, STATE_DIMS
    parser.add_argument("--rl-version", choices=("v1", "v2", "v3"), default="v1")
    add_channel_arguments(parser)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    from rl.evaluation_runtime import load_evaluation_policy
    final_policy_state, grids, _ = load_evaluation_policy(args.final_checkpoint, version=args.rl_version,
        grid_points=args.grid_points, grid_spacing=args.grid_spacing, seed=args.agent_seed, channel_kwargs=channel_kwargs(args))
    if args.initial_checkpoint is not None:
        initial_policy_state, initial_grids, _ = load_evaluation_policy(args.initial_checkpoint, version=args.rl_version,
            grid_points=args.grid_points, grid_spacing=args.grid_spacing, seed=args.agent_seed, channel_kwargs=channel_kwargs(args))
        if initial_grids != grids:
            raise ValueError("Matched checkpoint evaluation requires identical physical grids")
    else:
        # Deterministic, reproducible pre-training policy -- identical to
        # what experiments/train_autockt.py's own initial_policy_state was
        # before any PPO update, for the same --agent-seed.
        initial_policy_state = PPOAgent(
            state_dim=STATE_DIMS[args.rl_version], num_heads=len(grids), seed=args.agent_seed,
        ).policy.state_dict()
    from analysis.run_graph import RunGraph, ACTIVE
    graph = RunGraph({"backend": "real", "algorithm": "matched_checkpoint_evaluation", "version": args.rl_version}, args.output.with_suffix(".graph.json"))
    token = ACTIVE.set(graph)
    initial_indices = verified_initial_indices(grids)
    adapter = ReceiverRLAdapter(budget=RLBudget(args.max_evaluations), seed=args.eval_seed, version=args.rl_version, evaluator_kwargs=channel_kwargs(args))

    result = run_controlled_evaluation(
        initial_policy_state=initial_policy_state, final_policy_state=final_policy_state,
        adapter=adapter, grids=grids, initial_indices=initial_indices, horizon=args.horizon,
        eval_seed=args.eval_seed, agent_seed=args.agent_seed, episodes=args.episodes, rl_version=args.rl_version,
    )
    result["total_evaluations"] = graph.count
    result["initial_checkpoint_source"] = (
        str(args.initial_checkpoint) if args.initial_checkpoint is not None
        else f"fresh PPOAgent(seed={args.agent_seed}) -- not loaded from any file"
    )
    result["final_checkpoint_source"] = str(args.final_checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for case in result["cases"]:
            stream.write(json.dumps({"row_type": "case", **case}) + "\n")
        summary_row = {k: v for k, v in result.items() if k != "cases"}
        stream.write(json.dumps({"row_type": "summary", **summary_row}) + "\n")

    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    graph.finish(result)
    ACTIVE.reset(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
