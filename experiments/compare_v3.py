"""Small budget-matched real-SPICE comparison of frozen v3 and search baselines."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from analysis.target_assessment import assess_target, final_violations
from rl.autockt_env import AutoCktReceiverEnv
from rl.autockt_v3 import reward_v3
from rl.evaluation_runtime import load_evaluation_policy
from rl.ppo_agent import PPOAgent
from rl.runtime_contract import STATE_DIMS
from rl.target_spec import TargetSpec
from simulator.receiver import ReceiverParameters, EvaluationFidelity, evaluate_receiver
from simulator.rl_adapter import ReceiverRLAdapter, RLBudget
from simulator.runtime import time_budget, expired


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seconds-per-method", type=float, default=180)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--target-mode", choices=("trivial", "hard", "midpoint"), default="trivial")
    parser.add_argument("--random-start", action="store_true", help="Shared seed-specific starting point for all methods")
    args = parser.parse_args(argv)
    if args.budget < 2:
        parser.error("budget must be at least two")
    if args.output.exists():
        parser.error("output already exists")
    state, grids, agent = load_evaluation_policy(args.checkpoint, version="v3", grid_points=21,
        grid_spacing="log", seed=101, channel_kwargs={})
    target = TargetSpec.from_existing_thresholds()
    if args.target_mode == "hard":
        target = TargetSpec.from_hard_target()
    elif args.target_mode == "midpoint":
        from experiments.controlled_unseen_target_eval import UNSEEN_MIDPOINT_TARGET
        target = UNSEEN_MIDPOINT_TARGET
    center = tuple(len(grid.values) // 2 for grid in grids.values())
    report = {"schema_version": 1, "version": "v3", "backend": "real",
        "checkpoint": str(args.checkpoint), "target": target.as_dict(), "budget_per_method_seed": args.budget,
        "fidelity": "TRAINING", "initial_state": "seeded-random" if args.random_start else "grid-center", "runs": [],
        "seeds": args.seeds, "horizon": args.horizon, "target_mode": args.target_mode,
        "limitations": [f"Small {len(args.seeds)}-seed evaluation; pretrained PPO training cost is excluded.",
                         "TRAINING fidelity is nominal screening, not full RF/PVT validation.",
                         "Fresh policy is a seed-101 baseline; not a saved pretraining checkpoint.",
                         "Random/CEM share PPO's discrete grids and initial center; no evaluation cache."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        if args.random_start:
            start_rng = np.random.default_rng(seed)
            center = tuple(int(start_rng.integers(len(grid.values))) for grid in grids.values())
        for method in ("random_search", "cem", "fresh_v3", "trained_v3"):
            rows = []
            def evaluator(parameters, conditions=None, fidelity=EvaluationFidelity.TRAINING, **kwargs):
                from simulator.config import SimulationConditions
                started = time.perf_counter()
                result = evaluate_receiver(parameters, conditions or SimulationConditions(), fidelity, **kwargs)
                passed = assess_target(result.metrics, target, simulator_success=result.success).passed
                passed = passed and not any(k in ("peaking_db", "dfe_error_count") for k in final_violations(result.metrics, target))
                rows.append({"parameters": asdict(parameters), "metrics": result.metrics,
                    "success": passed, "simulator_success": result.success, "failed_stage": result.failed_stage,
                    "evaluation_id": result.evaluation_id, "elapsed_s": time.perf_counter() - started})
                return result
            started = time.perf_counter()
            with time_budget(args.seconds_per_method):
                if method.endswith("v3"):
                    policy = PPOAgent(STATE_DIMS["v3"], len(grids), seed=101)
                    if method == "trained_v3":
                        policy.policy.load_state_dict(state)
                    env = AutoCktReceiverEnv(target_pool=(target,), grids=grids, initial_indices=center,
                        horizon=args.horizon, seed=seed, evaluate_on_reset=True, state_schema="v3", use_reward_v2=True,
                        adapter=ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(args.budget), version="v3"))
                    while not env.budget_exhausted():
                        observation, _ = env.reset()
                        while not env.budget_exhausted():
                            step = env.step(policy.act(observation, deterministic=True)[0])
                            observation = step.state
                            if step.done or step.truncated:
                                break
                else:
                    rng = np.random.default_rng(seed)
                    sizes = np.array([len(grid.values) for grid in grids.values()])
                    mean, sigma = np.array(center, dtype=float), sizes / 3
                    population = []
                    for index in range(args.budget):
                        if expired():
                            break
                        indices = np.array(center) if index == 0 else (
                            rng.integers(0, sizes) if method == "random_search" else
                            np.clip(np.rint(rng.normal(mean, sigma)), 0, sizes - 1).astype(int))
                        parameters = ReceiverParameters(**{name: grid.value_at(int(i)) for (name, grid), i in zip(grids.items(), indices)})
                        result = evaluator(parameters)
                        score = reward_v3(result.metrics, target, success=result.success, failure_stage=result.failed_stage).total
                        population.append((score, indices))
                        if method == "cem" and (index + 1) % 4 == 0:
                            elite = np.array([entry[1] for entry in sorted(population, key=lambda entry: entry[0], reverse=True)[:2]])
                            mean, sigma = elite.mean(axis=0), np.maximum(elite.std(axis=0), 1.0)
                            population = []
            run = {"method": method, "seed": seed, "evaluations": len(rows), "rows": rows,
                   "initial_indices": list(center),
                   "complete": len(rows) == args.budget, "elapsed_s": time.perf_counter() - started,
                   "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else None}
            report["runs"].append(run)
            args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            print(json.dumps({k: v for k, v in run.items() if k != "rows"}), flush=True)
    return 0 if all(run["complete"] for run in report["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
