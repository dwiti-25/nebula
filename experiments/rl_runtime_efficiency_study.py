"""Phase 7: controlled runtime/simulation-efficiency comparison.

baseline v4_reward_v2 vs optimized v4_reward_v2 (same config + the
opt-in evaluation cache from rl/evaluation_cache.py wrapped around the
SAME evaluator callable). Everything else -- config, targets, starting
state, seeds, network init, horizon, budget, stopping criteria -- is
imported directly from experiments/rl_ppo_v2_ablation.py (the frozen,
already-validated v4_reward_v2 study) rather than redefined, so
"identical everything" is structural, not a promise.

This is a RUNTIME/EFFICIENCY comparison only. It does not claim PPO
beats Random Search or CEM, and synthetic-mode wall-clock numbers are
NOT representative of real-SPICE runtime (the synthetic evaluator body
is ~microseconds; real SPICE evaluations run 15-300s depending on
fidelity per this project's own historical measurements). The
evaluation-COUNT reduction is the metric that transfers to real-SPICE
deployment; wall-clock reduction in synthetic mode is reported as
measured, not extrapolated, and may be negligible or slightly negative
due to the cache wrapper's own (tiny) Python overhead.

Correctness invariant checked directly, not assumed: for a fixed seed,
baseline and optimized MUST produce bit-for-bit identical action
trajectories and rewards -- caching only changes whether the
underlying evaluator body actually runs, never what it returns. Any
divergence would mean the cache wrapper altered algorithmic behavior
(forbidden by Phase 6) and is asserted against in run_pair().
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from simulator.rl_adapter import ReceiverRLAdapter, RLBudget

from experiments.rl_ppo_v2_ablation import (
    GRID_POINTS,
    GRID_SPACING,
    HORIZON,
    TRAIN_TARGET,
    UPDATES,
    EPISODES_PER_UPDATE,
    VARIANTS,
    _distance_travelled,
    _infeasible_corner_indices,
    _uniform_strict_pass,
)
from rl.autockt_env import AutoCktReceiverEnv
from rl.autockt_state_v2 import STATE_DIM_V2
from rl.evaluation_cache import make_cached_evaluator
from rl.parameter_grid import PARAMETER_NAMES, build_parameter_grids
from rl.ppo_agent import PPOAgent
from rl.synthetic_benchmark import synthetic_evaluate_receiver_graded
from rl.trainer import train

VARIANT_CONFIG = VARIANTS["v4_reward_v2"]  # frozen; imported, not redefined
DEFAULT_SEEDS = tuple(range(20))


def _no_op_count(events: list[dict[str, Any]]) -> int:
    return sum(1 for e in events if all(d == 0 for d in e["applied_delta"]))


def _reversal_count(events: list[dict[str, Any]]) -> int:
    """Count sign flips on any single parameter's applied_delta between
    consecutive steps of the SAME episode -- a directional-oscillation
    signal (Phase 4)."""

    reversals = 0
    for prev, cur in zip(events, events[1:]):
        if prev["episode"] != cur["episode"] or prev["run_id"] != cur["run_id"]:
            continue
        for d_prev, d_cur in zip(prev["applied_delta"], cur["applied_delta"]):
            if d_prev != 0 and d_cur != 0 and (d_prev > 0) != (d_cur > 0):
                reversals += 1
                break
    return reversals


def _evaluations_per_improvement(events: list[dict[str, Any]]) -> float | None:
    """Mean number of step-evaluations between successive strict-pass-rate
    improvements in cumulative reward within a run; None if no improvement
    ever occurred."""

    best = float("-inf")
    improvement_gaps: list[int] = []
    last_improvement_index = -1
    for i, e in enumerate(events):
        if e["reward_total"] > best:
            best = e["reward_total"]
            improvement_gaps.append(i - last_improvement_index)
            last_improvement_index = i
    gaps_after_first = improvement_gaps[1:]  # first is from run start, not a "gap between improvements"
    return sum(gaps_after_first) / len(gaps_after_first) if gaps_after_first else None


def _run_config(seed: int, evaluator) -> tuple[dict[str, Any], list[dict[str, Any]], float, ReceiverRLAdapter]:
    grids = build_parameter_grids(GRID_POINTS, spacing=GRID_SPACING)
    initial_indices = _infeasible_corner_indices(grids)
    adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(10_000), seed=seed)
    env = AutoCktReceiverEnv(
        target_pool=(TRAIN_TARGET,), initial_indices=initial_indices, horizon=HORIZON,
        adapter=adapter, grids=grids, seed=seed, randomize_initial_state=True, **VARIANT_CONFIG,
    )
    agent = PPOAgent(state_dim=STATE_DIM_V2, num_heads=len(PARAMETER_NAMES), seed=seed)

    events: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    train_result = train(
        env, agent, num_updates=UPDATES, episodes_per_update=EPISODES_PER_UPDATE,
        on_event=events.append, run_id=f"seed{seed}",
    )
    wall_clock_s = time.perf_counter() - t0
    step_events = [e for e in events if e["event_type"] == "step"]
    return {"train_result": train_result}, step_events, wall_clock_s, adapter


def _metrics_for(step_events, wall_clock_s, adapter, grids) -> dict[str, Any]:
    strict_pass_flags = [_uniform_strict_pass(e, TRAIN_TARGET) for e in step_events]
    first_success_index = next((i for i, ok in enumerate(strict_pass_flags) if ok), None)
    boundary_hits = sum(1 for e in step_events if any(e["boundary_clipped"]))
    return {
        "n_steps": len(step_events),
        "n_evaluations": adapter.total_evaluations,
        "wall_clock_s": wall_clock_s,
        "strict_success_rate": sum(strict_pass_flags) / max(1, len(strict_pass_flags)),
        "evaluations_to_first_strict_success": first_success_index + 1 if first_success_index is not None else None,
        "success_within_budget": first_success_index is not None,
        "mean_reward": sum(e["reward_total"] for e in step_events) / max(1, len(step_events)),
        "boundary_hit_rate": boundary_hits / max(1, len(step_events)),
        "distance_travelled": _distance_travelled(step_events, grids),
        "no_op_action_count": _no_op_count(step_events),
        "directional_reversal_count": _reversal_count(step_events),
        "evaluations_per_improvement": _evaluations_per_improvement(step_events),
    }


def run_pair(seed: int) -> dict[str, Any]:
    grids = build_parameter_grids(GRID_POINTS, spacing=GRID_SPACING)

    _, baseline_events, baseline_wall_s, baseline_adapter = _run_config(seed, synthetic_evaluate_receiver_graded)
    cached_evaluator = make_cached_evaluator(synthetic_evaluate_receiver_graded)
    _, optimized_events, optimized_wall_s, optimized_adapter = _run_config(seed, cached_evaluator)

    # Correctness invariant: caching must not change any RL decision.
    baseline_actions = [e["action_choice"] for e in baseline_events]
    optimized_actions = [e["action_choice"] for e in optimized_events]
    baseline_rewards = [e["reward_total"] for e in baseline_events]
    optimized_rewards = [e["reward_total"] for e in optimized_events]
    if baseline_actions != optimized_actions or baseline_rewards != optimized_rewards:
        raise AssertionError(
            f"seed {seed}: cached evaluator produced a DIFFERENT trajectory than baseline -- "
            "caching altered algorithmic behavior, which Phase 6 forbids"
        )

    baseline_metrics = _metrics_for(baseline_events, baseline_wall_s, baseline_adapter, grids)
    optimized_metrics = _metrics_for(optimized_events, optimized_wall_s, optimized_adapter, grids)

    # "Expensive evaluator calls" -- for baseline every adapter call is
    # uncached (real work); for optimized only cache MISSES did real work.
    baseline_expensive_calls = baseline_adapter.total_evaluations
    optimized_expensive_calls = cached_evaluator.stats.misses

    return {
        "seed": seed,
        "trajectories_identical": True,  # only reached if the assertion above passed
        "baseline": baseline_metrics,
        "optimized": optimized_metrics,
        "baseline_expensive_evaluator_calls": baseline_expensive_calls,
        "optimized_expensive_evaluator_calls": optimized_expensive_calls,
        "cache_hits": cached_evaluator.stats.hits,
        "cache_misses": cached_evaluator.stats.misses,
        "cache_hit_rate": cached_evaluator.stats.hit_rate,
        "cache_unique_entries": len(cached_evaluator),
        "cache_approx_memory_bytes": cached_evaluator.approx_memory_bytes(),
        "evaluation_reduction_frac": (
            (baseline_expensive_calls - optimized_expensive_calls) / baseline_expensive_calls
            if baseline_expensive_calls else None
        ),
        "wall_clock_reduction_frac": (
            (baseline_wall_s - optimized_wall_s) / baseline_wall_s if baseline_wall_s else None
        ),
    }


def run_all(seeds: tuple[int, ...] = DEFAULT_SEEDS) -> list[dict[str, Any]]:
    """Runs one discarded warm-up pass before any timed pair.

    Measured directly, not assumed: without this, whichever config
    (baseline or optimized) happens to run FIRST in-process measures
    ~5-6x slower wall-clock than the one that runs second, regardless
    of which one is cached -- a PyTorch/interpreter/CPU warm-up
    artifact, not a caching effect (confirmed by swapping run order:
    the "second" config is always the fast one). One throwaway
    train() call before any timed run removes the effect (both configs
    then measure within ~2% of each other on the SAME uncached
    evaluator run twice). Skipping this warm-up would have reported a
    fabricated ~79% synthetic wall-clock "speedup" that had nothing to
    do with the cache.
    """

    _run_config(-1, synthetic_evaluate_receiver_graded)
    return [run_pair(seed) for seed in seeds]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    total_baseline_calls = sum(r["baseline_expensive_evaluator_calls"] for r in results)
    total_optimized_calls = sum(r["optimized_expensive_evaluator_calls"] for r in results)
    total_baseline_wall = sum(r["baseline"]["wall_clock_s"] for r in results)
    total_optimized_wall = sum(r["optimized"]["wall_clock_s"] for r in results)
    return {
        "n_seeds": n,
        "all_trajectories_identical": all(r["trajectories_identical"] for r in results),
        "mean_cache_hit_rate": sum(r["cache_hit_rate"] for r in results) / n,
        "total_baseline_expensive_evaluator_calls": total_baseline_calls,
        "total_optimized_expensive_evaluator_calls": total_optimized_calls,
        "evaluation_reduction_pct": (
            100.0 * (total_baseline_calls - total_optimized_calls) / total_baseline_calls
            if total_baseline_calls else None
        ),
        "total_baseline_wall_clock_s": total_baseline_wall,
        "total_optimized_wall_clock_s": total_optimized_wall,
        "wall_clock_reduction_pct": (
            100.0 * (total_baseline_wall - total_optimized_wall) / total_baseline_wall
            if total_baseline_wall else None
        ),
        "baseline_mean_strict_success_rate": sum(r["baseline"]["strict_success_rate"] for r in results) / n,
        "optimized_mean_strict_success_rate": sum(r["optimized"]["strict_success_rate"] for r in results) / n,
        "baseline_mean_reward": sum(r["baseline"]["mean_reward"] for r in results) / n,
        "optimized_mean_reward": sum(r["optimized"]["mean_reward"] for r in results) / n,
        "mean_cache_unique_entries": sum(r["cache_unique_entries"] for r in results) / n,
        "mean_cache_approx_memory_bytes": sum(r["cache_approx_memory_bytes"] for r in results) / n,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7 controlled runtime-efficiency comparison (synthetic, SPICE-free).")
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
