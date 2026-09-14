"""Lightweight sequential RL baseline for the NEBULA receiver."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from experiments.receiver_search import _sample_actions
from simulator.config import SimulationConditions
from simulator.rl_adapter import ReceiverRLAdapter


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def perturb(
    action: tuple[float, ...],
    rng: random.Random,
    sigma: float,
) -> tuple[float, ...]:
    return tuple(clamp(x + rng.gauss(0.0, sigma)) for x in action)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential learning baseline for the NEBULA receiver"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rl_baseline.jsonl"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    conditions = SimulationConditions()

    # Start from physically sensible, constraint-aware designs.
    initial_actions = _sample_actions(
        max(args.episodes, 10),
        args.seed,
        "constraint_aware_v1",
        conditions,
    )

    env = ReceiverRLAdapter(
        conditions=conditions,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    best_reward = -math.inf
    best_action = None
    best_parameters = None
    total_evaluations = 0

    with args.output.open("w", encoding="utf-8") as output:
        for episode in range(args.episodes):
            # Seed each episode with a feasible design.
            current_action = initial_actions[episode]

            observation, reset_info = env.reset(
                seed=args.seed + episode
            )

            episode_best = -math.inf

            for step_number in range(args.steps):
                # First step evaluates the feasible seed.
                # Later steps make a local sequential update around
                # the previous design.
                if step_number > 0:
                    current_action = perturb(
                        current_action,
                        rng,
                        args.sigma,
                    )

                step = env.step(current_action)
                total_evaluations += 1

                reward = step.reward

                if reward > best_reward:
                    best_reward = reward
                    best_action = current_action
                    best_parameters = step.info["parameters"]

                episode_best = max(episode_best, reward)

                row = {
                    "episode": episode + 1,
                    "step": step_number + 1,
                    "evaluation": total_evaluations,
                    "reward": reward,
                    "action": list(current_action),
                    "parameters": step.info["parameters"],
                    "success": reward > -100.0,
                    "failure_stage": step.info.get("failure_stage"),
                    "episode_best_reward": episode_best,
                    "observation_length": len(step.observation),
                    "constraints": list(step.constraints),
                    "reward_version": step.info["reward_version"],
                    "reset_seed": reset_info["seed"],
                }

                output.write(json.dumps(row) + "\n")
                output.flush()

                print(json.dumps(row), flush=True)

                observation = step.observation

            print(
                json.dumps(
                    {
                        "episode_complete": episode + 1,
                        "episode_best_reward": episode_best,
                        "global_best_reward": best_reward,
                        "global_best_action": best_action,
                    },
                    indent=2,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "episodes": args.episodes,
                "steps_per_episode": args.steps,
                "evaluations": total_evaluations,
                "best_reward": best_reward,
                "best_action": best_action,
                "best_parameters": best_parameters,
                "output": str(args.output),
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
