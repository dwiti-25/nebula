"""PPO model-improvement study, improvement #2: a broader held-out
generalization check.

The project's only existing unseen-target generalization result (docs/
autockt-mapping.md sec 18) evaluates exactly ONE held-out target -- the
arithmetic midpoint of the two training targets (trivial, hard) -- giving
n=1 distinct target (many episodes against that one target, but only one
difficulty level). This script broadens that to several genuinely
different held-out targets, using the EXISTING trained checkpoint
(results/autockt_mixed_target_confirmation_policy.pt) with NO new
training: pure deterministic rollout (agent.act(..., deterministic=True))
via experiments.train_autockt._evaluate_checkpoint, reused unmodified.

Held-out targets, each independently justified (never invented numbers):

  moderate_25pct / moderate_75pct: two additional points on the SAME
    trivial->hard interpolation line the existing midpoint (50%) came
    from -- broadens coverage along the one axis already explored,
    without introducing a new axis.

  power_focused: trivial and hard BOTH fix ctle_power_w at the existing
    0.015 W threshold -- no prior generalization check (midpoint
    included) has ever varied power at all. This target keeps
    height/width/margin close to the trivial target but tightens power
    to 0.002 W (still comfortably above Design A's own real measured
    power, ~0.00109 W, so it is plausibly achievable, not extrapolated
    beyond demonstrated behavior) -- a genuinely orthogonal probe, not
    another point on the same line.

No historical result file is touched. Output goes to a new file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from simulator.rl_adapter import ReceiverRLAdapter, RLBudget

from rl.parameter_grid import PARAMETER_NAMES, build_parameter_grids, verified_initial_indices
from rl.ppo_agent import PPOAgent
from rl.autockt_state import STATE_DIM
from rl.target_spec import EXISTING_THRESHOLDS, HARD_TARGET_THRESHOLDS, SPEC_NAMES, TargetSpec

from experiments.train_autockt import _evaluate_checkpoint

trivial = TargetSpec.from_existing_thresholds()
hard = TargetSpec.from_hard_target()


def _interpolate(fraction: float) -> TargetSpec:
    return TargetSpec(**{
        name: getattr(trivial, name) + fraction * (getattr(hard, name) - getattr(trivial, name))
        for name in SPEC_NAMES
    })


HELD_OUT_TARGETS: dict[str, TargetSpec] = {
    "moderate_25pct": _interpolate(0.25),
    "moderate_75pct": _interpolate(0.75),
    "power_focused": TargetSpec(
        dfe_locked_phase_eye_height_v=EXISTING_THRESHOLDS["dfe_locked_phase_eye_height_v"],
        dfe_eye_width_ui=EXISTING_THRESHOLDS["dfe_eye_width_ui"] + 0.02,
        dfe_min_margin_v=EXISTING_THRESHOLDS["dfe_min_margin_v"] + 0.02,
        ctle_power_w=0.002,
    ),
}


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Broader held-out generalization check: existing final checkpoint, "
        "several new held-out targets, zero new training."
    )
    parser.add_argument("--checkpoint", type=Path,
                         default=Path("results/autockt_mixed_target_confirmation_policy.pt"))
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--grid-points", type=int, default=21)
    parser.add_argument("--grid-spacing", choices=("linear", "log"), default="log")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--randomize-initial-state", action="store_true", default=True)
    parser.add_argument("--max-evaluations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    from rl.runtime_contract import add_channel_arguments, channel_kwargs
    parser.add_argument("--rl-version", choices=("v1", "v2", "v3"), default="v1")
    add_channel_arguments(parser)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    from rl.evaluation_runtime import load_evaluation_policy
    policy_state, grids, agent = load_evaluation_policy(args.checkpoint, version=args.rl_version,
        grid_points=args.grid_points, grid_spacing=args.grid_spacing, seed=args.agent_seed, channel_kwargs=channel_kwargs(args))
    initial_indices = verified_initial_indices(grids)
    adapter = ReceiverRLAdapter(budget=RLBudget(args.max_evaluations), seed=args.eval_seed, version=args.rl_version, evaluator_kwargs=channel_kwargs(args))
    from analysis.run_graph import RunGraph, ACTIVE
    graph = RunGraph({"backend": "real", "algorithm": "held_out_evaluation", "version": args.rl_version, "checkpoint": str(args.checkpoint)}, args.output.with_suffix(".graph.json"))
    token = ACTIVE.set(graph)

    results: dict[str, object] = {}
    output_rows: list[dict[str, object]] = []
    t0 = time.monotonic()
    for name, target in HELD_OUT_TARGETS.items():
        summary, rows = _evaluate_checkpoint(
            label=name, policy_state=policy_state, agent=agent, adapter=adapter,
            training_pool=(target,), initial_indices=initial_indices, grids=grids,
            horizon=args.horizon, randomize_initial_state=args.randomize_initial_state,
            seed=args.eval_seed, episodes=args.episodes, rl_version=args.rl_version,
        )
        results[name] = {
            "target": target.as_dict(), "satisfaction_rate": summary["satisfaction_rate"],
            "mean_episode_reward": summary["mean_episode_reward"], "episodes": summary["episodes"],
        }
        output_rows.extend({"target_label": name, **row} for row in rows)
    elapsed_s = time.monotonic() - t0

    summary_record = {
        "row_type": "summary",
        "checkpoint": str(args.checkpoint),
        "eval_seed": args.eval_seed, "agent_seed": args.agent_seed,
        "horizon": args.horizon, "episodes_per_target": args.episodes,
        "randomize_initial_state": args.randomize_initial_state,
        "grid_points": args.grid_points, "grid_spacing": args.grid_spacing,
        "targets": {name: t.as_dict() for name, t in HELD_OUT_TARGETS.items()},
        "results": results,
        "total_evaluations": graph.count,
        "wall_clock_s": elapsed_s,
        # comparison point: the existing single-target generalization result (docs sec 18)
        "existing_midpoint_generalization_reference": {
            "wins": 1, "ties": 6, "losses": 3,
            "source": "results/controlled_unseen_target_generalization.jsonl",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in output_rows:
            stream.write(json.dumps(row) + "\n")
        stream.write(json.dumps(summary_record) + "\n")

    print(json.dumps(summary_record, indent=2))
    graph.finish(summary_record)
    ACTIVE.reset(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
