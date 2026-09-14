"""First learning baseline for the NEBULA receiver."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from simulator.rl_adapter import ReceiverRLAdapter
from rl.autockt_env import metrics_from_observation
from rl.autockt_state import lookup, signed_relative_error
from rl.target_spec import SPEC_NAMES, TargetSpec

# [NEBULA ADAPTATION] reward_v1 (simulator.rl_adapter.reward_v1, what CEM's
# elite selection ranks candidates by, below) returns a FLAT -100.0 for
# ANY simulator-stage failure, regardless of how close the candidate came
# to passing -- see docs/autockt-mapping.md sec 19 for the confirmed
# consequence (an unbiased-init CEM run's logged best_reward stayed at
# exactly -100.0 across all iterations, elite selection choosing among
# ties, std collapsing to its floor with no real gradient followed).
#
# GRADED_FITNESS_NO_INFORMATION_FLOOR / graded_cem_fitness below are an
# OPTIONAL, additive alternative objective ONLY for elite selection
# (--fitness graded) -- reward_v1 remains the default and is still what
# gets logged as "reward"/used for reported success in every row either
# way; this does not change what "success" means anywhere in the repo.
# Mirrors rl.autockt_reward.autockt_reward's own unsatisfied-sum formula
# (same lookup()/signed_relative_error() machinery, reused read-only, not
# modified) but does NOT gate on `success` first, so a candidate that
# fails only because of the TRAINING-fidelity margin>0 hard gate (i.e. its
# `failure_stage` is "transient" -- the stage that actually computes eye
# height/width/margin/receiver power, per simulator/receiver.py
# ::_transient_violations) still contributes a graded score reflecting how
# close it came, instead of the same flat floor as an early dc/ac/setup
# failure. A design that never reaches the stage where those four metrics
# are computed genuinely has no principled continuous measure of them
# available -- GRADED_FITNESS_NO_INFORMATION_FLOOR is a fixed, disclosed
# floor for that case, not an invented proxy.
GRADED_FITNESS_NO_INFORMATION_FLOOR = -10.0
# failure_stage values whose metrics dict actually contains real, non-zero-
# filled height/width/margin/power measurements (see simulator/receiver.py
# ::_transient_violations -- both the passing and the violating outcome of
# that same stage compute and attach the full metrics dict).
GRADED_FITNESS_RELIABLE_STAGES = (None, "transient")


def graded_cem_fitness(metrics: dict, target: TargetSpec, failure_stage: str | None) -> float:
    if failure_stage not in GRADED_FITNESS_RELIABLE_STAGES:
        return GRADED_FITNESS_NO_INFORMATION_FLOOR
    total = 0.0
    for name in SPEC_NAMES:
        relative_error = signed_relative_error(name, lookup(metrics.get(name, 0.0), getattr(target, name)))
        if relative_error < 0:
            total += relative_error
    return total  # <= 0.0 always; 0.0 means every one of the 4 specs is at/above its target


# [NEBULA ADAPTATION] the original, warm-started initialization: a Gaussian
# centered near a known-good design (not the center of the normalized
# action space) with a narrow std. Kept as the CLI default so existing
# invocations/behavior are unchanged; --init-mean/--init-std (below) make
# this overridable for an unbiased, uniform-search-comparable CEM run
# without altering the algorithm itself (elite selection, mean/std update,
# and the reward_v1 objective CEM ranks candidates by are all unchanged
# either way).
DEFAULT_INIT_MEAN = (
    0.369674437322012,
    0.27670652861099754,
    0.33331526214636176,
    0.7801971410738049,
    -0.027727059712872038,
)
DEFAULT_INIT_STD = 0.15


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a simple CEM optimizer on the NEBULA receiver"
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--population", type=int, default=5)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init-mean", type=float, nargs="+", default=None,
        help="[NEBULA ADAPTATION] initial Gaussian mean over the 5 normalized "
        "action dimensions. Default matches the original warm-started values "
        "(unchanged behavior); pass '0 0 0 0 0' for an unbiased, centered start.",
    )
    parser.add_argument(
        "--init-std", type=float, nargs="+", default=None,
        help="[NEBULA ADAPTATION] initial per-dimension std. Default matches the "
        "original narrow value (0.15); pass e.g. '0.577 ...' (Uniform(-1,1)'s own "
        "std, 2/sqrt(12)) for first-generation coverage comparable to uniform sampling.",
    )
    parser.add_argument(
        "--fitness", choices=("reward_v1", "graded"), default="reward_v1",
        help="[NEBULA ADAPTATION] objective elite selection ranks candidates by. "
        "'reward_v1' (default, unchanged behavior): simulator.rl_adapter.reward_v1, "
        "flat -100.0 on any failure. 'graded': graded_cem_fitness (see module docstring) "
        "-- gives informative signal among candidates that reach the transient stage. "
        "Does NOT change what is logged as 'reward' or what counts as 'success' in the "
        "output row (both remain reward_v1-based, unchanged) -- only what elite selection "
        "optimizes toward.",
    )
    parser.add_argument(
        "--target", choices=("trivial", "hard"), default="trivial",
        help="[NEBULA ADAPTATION] only used when --fitness graded: which TargetSpec "
        "graded_cem_fitness scores against.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cem_training.jsonl"),
    )
    from rl.runtime_contract import add_channel_arguments, channel_kwargs
    from simulator.design_schema import parameter_names
    parser.add_argument("--rl-version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--backend", choices=("real", "synthetic"), default="real")
    add_channel_arguments(parser)
    parser.add_argument("--grid-points", type=int, default=21)
    parser.add_argument("--grid-spacing", choices=("linear", "log"), default="linear")
    args = parser.parse_args()

    if args.elite > args.population:
        raise ValueError("--elite cannot exceed --population")

    target = TargetSpec.from_hard_target() if args.target == "hard" else TargetSpec.from_existing_thresholds()

    rng = random.Random(args.seed)
    from simulator.rl_adapter import RLBudget
    from simulator.receiver import evaluate_receiver
    from rl.synthetic_v3 import synthetic_evaluate_receiver_v3
    from rl.synthetic_benchmark import synthetic_evaluate_receiver
    n_parameters = len(parameter_names(args.rl_version))
    from rl.parameter_grid import build_parameter_grids
    grids = build_parameter_grids(args.grid_points, spacing=args.grid_spacing, version=args.rl_version) if args.rl_version == "v3" else None
    evaluator = evaluate_receiver if args.backend == "real" else (synthetic_evaluate_receiver_v3 if args.rl_version == "v3" else synthetic_evaluate_receiver)
    env = ReceiverRLAdapter(seed=args.seed, version=args.rl_version, evaluator=evaluator,
        budget=RLBudget(args.iterations * args.population), evaluator_kwargs=channel_kwargs(args), grids=grids)

    mean = list(args.init_mean if args.init_mean is not None else DEFAULT_INIT_MEAN + ((-0.807, -1.0, -1.0) if args.rl_version == "v3" else ()))
    std = list(args.init_std if args.init_std is not None else [DEFAULT_INIT_STD] * n_parameters)
    if len(mean) != n_parameters or len(std) != n_parameters or any(v <= 0 for v in std):
        raise ValueError("CEM mean/std must match the selected parameter count and std must be positive")
    from analysis.run_graph import RunGraph, ACTIVE
    graph = RunGraph({"backend": args.backend, "algorithm": "cem", "version": args.rl_version, "channel": str(args.channel)}, args.output.with_suffix(".graph.json"))
    token = ACTIVE.set(graph)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    best_reward = -math.inf
    best_action = None

    with args.output.open("w", encoding="utf-8") as output:
        evaluation_number = 0

        for iteration in range(args.iterations):
            candidates = []

            for _ in range(args.population):
                action = tuple(
                    max(-1.0, min(1.0, rng.gauss(mean[i], std[i])))
                    for i in range(n_parameters)
                )

                start_time = time.perf_counter()
                step = env.step(action)
                wall_clock_s = time.perf_counter() - start_time
                evaluation_number += 1

                metrics = metrics_from_observation(step.observation)
                failure_stage = step.info.get("failure_stage")
                if args.rl_version == "v3":
                    from rl.autockt_v3 import reward_v3
                    metrics = dict(step.info["raw_metrics"])
                    p = step.info["parameters"]
                    metrics["partial_mos_channel_area_um2"] = 2 * p["mos_width_um"] * p["mos_length_um"] * p["mos_multiplier"]
                    fitness_value = reward_v3(metrics, target, success=failure_stage is None, failure_stage=failure_stage).total
                elif args.fitness == "graded":
                    fitness_value = graded_cem_fitness(metrics, target, failure_stage)
                else:
                    fitness_value = step.reward
                candidates.append((fitness_value, action))

                reported_reward = fitness_value if args.rl_version == "v3" else step.reward
                if reported_reward > best_reward:
                    best_reward = reported_reward
                    best_action = action

                row = {
                    "iteration": iteration + 1,
                    "evaluation": evaluation_number,
                    "reward": reported_reward,
                    "action": action,
                    "success": step.info.get("failure_stage") is None
                    and step.reward > -100.0,
                    "failure_stage": failure_stage,
                    "wall_clock_s": wall_clock_s,
                    "parameters": step.info.get("parameters"),
                    "evaluation_id": step.info.get("evaluation_id"),
                    # [NEBULA ADAPTATION] reused, unmodified rl.autockt_env
                    # helper -- reconstructs the metric-name -> value dict
                    # from the adapter's own public observation tuple, the
                    # same read-only reuse pattern rl/autockt_env.py already
                    # uses. Does not change what CEM optimizes by default
                    # (elite selection ranks by reward_v1 unless --fitness
                    # graded is passed); this only makes post-hoc, uniform
                    # re-scoring (e.g. against a TargetSpec threshold set)
                    # possible from the logged output, which it previously
                    # was not.
                    "metrics": metrics,
                    "fitness_kind": "reward_v3" if args.rl_version == "v3" else args.fitness,
                    "fitness_value": fitness_value,
                }

                output.write(json.dumps(row) + "\n")
                output.flush()

                print(json.dumps(row), flush=True)

            # Highest-FITNESS actions become the next generation's center
            # (fitness_value == reward_v1 unless --fitness graded).
            candidates.sort(key=lambda item: item[0], reverse=True)
            elites = candidates[: args.elite]

            for i in range(n_parameters):
                values = [action[i] for _, action in elites]
                mean[i] = sum(values) / len(values)

                if len(values) > 1:
                    variance = sum(
                        (value - mean[i]) ** 2 for value in values
                    ) / len(values)
                    std[i] = max(0.15, math.sqrt(variance))
                else:
                    std[i] = max(0.15, std[i] * 0.7)

            print(
                json.dumps(
                    {
                        "iteration_complete": iteration + 1,
                        "best_reward": best_reward,
                        "best_action": best_action,
                        "mean": mean,
                        "std": std,
                    },
                    indent=2,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "evaluations": evaluation_number,
                "best_reward": best_reward,
                "best_action": best_action,
                "output": str(args.output),
            },
            indent=2,
        )
    )

    graph.finish({"algorithm": "cem", "best_reward": best_reward, "best_action": best_action})
    ACTIVE.reset(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
