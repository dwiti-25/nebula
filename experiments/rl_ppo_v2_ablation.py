"""Required change 7: controlled synthetic ablations, PPO model-improvement
study. SPICE-free throughout (rl.synthetic_benchmark.
synthetic_evaluate_receiver_graded -- verified deterministic, pure-Python,
no subprocess/SPICE calls; see its module docstring and
tests/test_synthetic_benchmark.py). Establishes SOFTWARE behavior only --
does not prove circuit improvement (see module docstring of
rl/synthetic_benchmark.py and the report this script's output feeds).

5 variants, incremental/nested per the required ladder:
  1. v1_baseline            -- exact existing behavior, unchanged
  2. v2_corrected_reset     -- + evaluate_on_reset (Required change 1)
  3. v3_validity_state      -- + state_schema="v2" (Required change 2)
  4. v4_reward_v2           -- + use_reward_v2 (Required change 3)
  5. v5_symmetric_actions   -- + symmetric {-1,0,+1} action deltas (optional)

All 20+ seeds share IDENTICAL starting states (grid-center, unbiased --
the same reachability-testing setup the fair real-SPICE trial used),
target sets, budgets, and network seeds; only the variant configuration
differs -- this is what makes cross-variant differences attributable.

A uniform "strict success" criterion (reward_v2's own strict_target_pass,
recomputed post-hoc from each step's raw metrics via the SAME reward_v2
function for every variant regardless of what it trained with) is used
for the headline success-rate comparison, mirroring this project's
established practice (analysis/fair_comparison.py) of never comparing
methods by their own different native criteria.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from simulator.rl_adapter import ReceiverRLAdapter, RLBudget

from rl.autockt_env import AutoCktReceiverEnv
from rl.autockt_reward_v2 import reward_v2
from rl.autockt_state import STATE_DIM as STATE_DIM_V1
from rl.autockt_state_v2 import STATE_DIM_V2
from rl.parameter_grid import ACTION_DELTAS, PARAMETER_NAMES, build_parameter_grids
from rl.ppo_agent import PPOAgent
from rl.synthetic_benchmark import synthetic_evaluate_receiver_graded
from rl.target_spec import TargetSpec
from rl.trainer import train

GRID_POINTS = 21
GRID_SPACING = "log"
UPDATES = 3
EPISODES_PER_UPDATE = 4
HORIZON = 4
HELD_OUT_EPISODES = 5
TRAIN_TARGET = TargetSpec.from_existing_thresholds()
HELD_OUT_TARGET = TargetSpec.from_hard_target()
DEFAULT_SEEDS = tuple(range(20))

SYMMETRIC_DELTAS = (-1, 0, 1)

VARIANTS: dict[str, dict[str, Any]] = {
    "v1_baseline": dict(evaluate_on_reset=False, state_schema="v1", use_reward_v2=False, action_deltas=ACTION_DELTAS),
    "v2_corrected_reset": dict(evaluate_on_reset=True, state_schema="v1", use_reward_v2=False, action_deltas=ACTION_DELTAS),
    "v3_validity_state": dict(evaluate_on_reset=True, state_schema="v2", use_reward_v2=False, action_deltas=ACTION_DELTAS),
    "v4_reward_v2": dict(evaluate_on_reset=True, state_schema="v2", use_reward_v2=True, action_deltas=ACTION_DELTAS),
    "v5_symmetric_actions": dict(evaluate_on_reset=True, state_schema="v2", use_reward_v2=True, action_deltas=SYMMETRIC_DELTAS),
}


def _infeasible_corner_indices(grids) -> tuple[int, ...]:
    """Deliberately NOT grid-center: grid-center coincides almost exactly
    with SYNTHETIC_CENTER (the synthetic landscape's own "good" point, at
    normalized fraction 0.5 on every dimension) since both are each
    parameter's own midpoint -- starting there made every ablation variant
    succeed trivially in 1 evaluation regardless of configuration (verified
    directly), giving zero differentiation and defeating the entire
    hypothesis under test ("...improves PPO's ability to move from
    non-feasible starting regions"). Index 0 (every parameter's lower
    bound, normalized fraction 0.0) sits at synthetic distance ~1.118 from
    center, solidly inside SYNTHETIC_FAILURE_STAGE's "no information"
    region (> SYNTHETIC_TRANSIENT_REACHED_DISTANCE=1.05) -- a genuinely
    infeasible starting corner, giving the policy real work to do.
    """

    return tuple(0 for _ in PARAMETER_NAMES)


def _state_dim(variant_config: dict[str, Any]) -> int:
    return STATE_DIM_V2 if variant_config["state_schema"] == "v2" else STATE_DIM_V1


def _uniform_strict_pass(event: dict[str, Any], target: TargetSpec) -> bool:
    """Recomputes strict_target_pass via reward_v2 from the event's own raw
    metrics/failure_stage, REGARDLESS of which reward the variant actually
    trained with -- the uniform, cross-variant-comparable criterion.
    """

    metrics_valid = all(event["metrics_valid_mask"].get(name, False) for name in
                         ("dfe_eye_height_v", "dfe_eye_width_ui", "dfe_min_margin_v", "ctle_power_w"))
    success = event["failure_stage"] is None
    result = reward_v2(
        event["raw_metrics"], target, metrics_valid=metrics_valid, success=success,
        failure_stage=event["failure_stage"],
    )
    return result.strict_target_pass


def _distance_travelled(events: list[dict[str, Any]], grids) -> float:
    """Sum of per-step normalized-space step sizes. ParameterGrid.
    normalized_index maps [0, len-1] -> [-1, 1] linearly, so an index
    DELTA's normalized-space magnitude is 2*delta/(len-1) -- computed
    directly from applied_delta (already a delta, not an absolute index),
    without needing an absolute-index round trip.
    """

    total = 0.0
    for event in events:
        total += math.sqrt(sum(
            (delta / (len(grids[name]) - 1) * 2.0) ** 2
            for name, delta in zip(PARAMETER_NAMES, event["applied_delta"])
        ))
    return total


def run_one(variant_name: str, seed: int) -> dict[str, Any]:
    variant_config = VARIANTS[variant_name]
    grids = build_parameter_grids(GRID_POINTS, spacing=GRID_SPACING)
    initial_indices = _infeasible_corner_indices(grids)  # genuinely infeasible start -- tests the hypothesis directly
    adapter = ReceiverRLAdapter(evaluator=synthetic_evaluate_receiver_graded, budget=RLBudget(10_000), seed=seed)
    env = AutoCktReceiverEnv(
        target_pool=(TRAIN_TARGET,), initial_indices=initial_indices, horizon=HORIZON,
        adapter=adapter, grids=grids, seed=seed, randomize_initial_state=True, **variant_config,
    )
    agent = PPOAgent(state_dim=_state_dim(variant_config), num_heads=len(PARAMETER_NAMES), seed=seed)

    events: list[dict[str, Any]] = []
    t0 = time.monotonic()
    train_result = train(
        env, agent, num_updates=UPDATES, episodes_per_update=EPISODES_PER_UPDATE, on_event=events.append,
        run_id=f"{variant_name}-seed{seed}",
    )
    wall_clock_s = time.monotonic() - t0

    step_events = [e for e in events if e["event_type"] == "step"]
    strict_pass_flags = [_uniform_strict_pass(e, TRAIN_TARGET) for e in step_events]
    first_success_index = next((i for i, ok in enumerate(strict_pass_flags) if ok), None)
    failure_stage_counts: dict[str, int] = {}
    for e in step_events:
        key = e["failure_stage"] or "success"
        failure_stage_counts[key] = failure_stage_counts.get(key, 0) + 1
    boundary_hits = sum(1 for e in step_events if any(e["boundary_clipped"]))

    # Held-out, zero-shot, deterministic evaluation -- same starting
    # distribution, DIFFERENT target, no further training.
    held_out_events: list[dict[str, Any]] = []
    eval_adapter = ReceiverRLAdapter(evaluator=synthetic_evaluate_receiver_graded, budget=RLBudget(10_000), seed=seed + 500_000)
    eval_env = AutoCktReceiverEnv(
        target_pool=(HELD_OUT_TARGET,), initial_indices=initial_indices, horizon=HORIZON,
        adapter=eval_adapter, grids=grids, seed=seed + 500_000, randomize_initial_state=True, **variant_config,
    )
    for episode in range(HELD_OUT_EPISODES):
        state, reset_info = eval_env.reset()
        done = truncated = False
        while not done and not truncated:
            choices, log_prob, value = agent.act(state, deterministic=True)
            indices_before = eval_env.indices
            step_out = eval_env.step(choices)
            from rl.events import build_step_event
            held_out_events.append(build_step_event(
                run_id=f"{variant_name}-seed{seed}-heldout", step=step_out.info["step_count"], episode=episode,
                update=-1, target_id=str(reset_info["target"]), target_values=reset_info["target"],
                state_before=state, state_after=step_out.state, metrics_valid_mask=step_out.info["metrics_valid_mask"],
                action_choice=tuple(choices), action_deltas=variant_config["action_deltas"],
                indices_before=indices_before, indices_after=step_out.info["indices"],
                parameters_before={}, parameters_after=step_out.info["parameters"] or {},
                raw_metrics=step_out.info["metrics"], reward_components={}, reward_total=step_out.reward,
                strict_pass=None, failure_stage=step_out.info["failure_stage"], log_prob=log_prob,
                value_estimate=value, done=step_out.done, truncated=step_out.truncated,
                evaluation_count=step_out.info["total_evaluation_count"],
            ))
            state = step_out.state
            done, truncated = step_out.done, step_out.truncated
    held_out_strict = [_uniform_strict_pass(e, HELD_OUT_TARGET) for e in held_out_events]

    loose_successes = failure_stage_counts.get("success", 0)  # simulator-level evaluation.success, looser than strict_pass
    return {
        "variant": variant_name, "seed": seed,
        "n_evaluations": adapter.total_evaluations,
        "n_steps": len(step_events),
        "strict_success_rate": sum(strict_pass_flags) / max(1, len(strict_pass_flags)),
        "loose_success_rate": loose_successes / max(1, len(step_events)),
        "evaluations_to_first_strict_success": first_success_index + 1 if first_success_index is not None else None,
        "success_within_budget": first_success_index is not None,
        "mean_reward": sum(e["reward_total"] for e in step_events) / max(1, len(step_events)),
        "final_update_mean_reward": train_result.updates[-1]["mean_episode_reward"] if train_result.updates else None,
        "failure_stage_distribution": failure_stage_counts,
        "boundary_hit_rate": boundary_hits / max(1, len(step_events)),
        "distance_travelled": _distance_travelled(step_events, grids),
        "held_out_strict_success_rate": sum(held_out_strict) / max(1, len(held_out_strict)),
        "held_out_n_episodes": HELD_OUT_EPISODES,
        "wall_clock_s": wall_clock_s,
    }


def run_all(seeds: tuple[int, ...] = DEFAULT_SEEDS) -> list[dict[str, Any]]:
    results = []
    for variant_name in VARIANTS:
        for seed in seeds:
            results.append(run_one(variant_name, seed))
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant_name in VARIANTS:
        rows = [r for r in results if r["variant"] == variant_name]
        n = len(rows)
        successes = [r["evaluations_to_first_strict_success"] for r in rows
                     if r["evaluations_to_first_strict_success"] is not None]
        summary[variant_name] = {
            "n_seeds": n,
            "mean_strict_success_rate": sum(r["strict_success_rate"] for r in rows) / n,
            "mean_loose_success_rate": sum(r["loose_success_rate"] for r in rows) / n,
            "success_within_budget_rate": sum(r["success_within_budget"] for r in rows) / n,
            # None (not 0.0) when no seed ever succeeded -- 0.0 would
            # falsely read as "succeeded instantly," exactly the
            # fabricated-zero mistake this study exists to avoid.
            "mean_evaluations_to_first_success": (sum(successes) / len(successes)) if successes else None,
            "n_seeds_that_succeeded": len(successes),
            "mean_reward": sum(r["mean_reward"] for r in rows) / n,
            "mean_boundary_hit_rate": sum(r["boundary_hit_rate"] for r in rows) / n,
            "mean_distance_travelled": sum(r["distance_travelled"] for r in rows) / n,
            "mean_held_out_strict_success_rate": sum(r["held_out_strict_success_rate"] for r in rows) / n,
            "mean_wall_clock_s": sum(r["wall_clock_s"] for r in rows) / n,
        }
    return summary


def _main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic, SPICE-free PPO v2 ablation study.")
    parser.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    seeds = tuple(range(args.seeds))
    results = run_all(seeds)
    summary = summarize(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps({"row_type": "seed_result", **row}) + "\n")
        stream.write(json.dumps({"row_type": "summary", "seeds": list(seeds), "summary": summary}) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
