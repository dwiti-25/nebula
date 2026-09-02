"""AutoCkt-style PPO training entry point for the NEBULA receiver.

[NEBULA ADAPTATION] This is the first real AutoCkt-methodology RL algorithm
in the repo. It is deliberately separate from experiments/train_rl.py
(a reward-gated local-search baseline -- not RL/PPO, see git history) and
experiments/train_cem.py (Cross-Entropy Method baseline); both remain
unmodified and continue to exist as independent baselines.

Every step here is one real evaluation through the untouched
simulator.rl_adapter.ReceiverRLAdapter -> simulator.receiver.evaluate_receiver
-> real ngspice. Nothing is mocked. Real SPICE evaluation is expensive
(measured ~15-65s/evaluation at TRAINING fidelity in this session -- see
docs/autockt-mapping.md), so the defaults below are intentionally small: a
few PPO updates over a handful of short episodes, enough to prove the full
loop (target spec -> state -> PPO -> discrete action -> parameters ->
ReceiverRLAdapter -> real SPICE -> metrics -> AutoCkt reward -> PPO update)
actually closes end to end, not to produce a converged policy. See
docs/autockt-mapping.md, section P, for what is realistically achievable in
the first milestone versus later scale-up.

Full component-by-component [AUTOCKT-REPLICATED] / [NEBULA ADAPTATION] /
[ROUGH-SCALE ADAPTATION] / [UNVERIFIED-FROM-SOURCE] labeling lives in
docs/autockt-mapping.md and in the docstrings of rl/parameter_grid.py,
rl/target_spec.py, rl/autockt_state.py, rl/autockt_action.py,
rl/autockt_reward.py, rl/autockt_env.py, and rl/ppo_agent.py.
"""

from __future__ import annotations

# Must be set before `import torch`: this machine's conda-forge numpy/scipy
# stack and the pip-installed torch wheel each ship their own libomp.dylib,
# which OpenMP refuses to double-initialize by default. This is a
# environment workaround, not a NEBULA/AutoCkt methodology choice -- see
# docs/autockt-mapping.md, "Environment notes".
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import copy
import json
from pathlib import Path

import torch

from simulator.rl_adapter import ACTION_SCHEMA_VERSION, ReceiverRLAdapter, RLBudget

from rl.autockt_env import AutoCktReceiverEnv
from rl.autockt_reward import AUTOCKT_REWARD_VERSION, graded_autockt_reward
from rl.autockt_state import STATE_DIM
from rl.parameter_grid import (
    DEFAULT_GRID_POINTS,
    PARAMETER_NAMES,
    ParameterGrid,
    build_parameter_grids,
    verified_initial_indices,
)
from rl.ppo_agent import PPOAgent
from rl.synthetic_benchmark import synthetic_evaluate_receiver
from rl.target_spec import EXISTING_THRESHOLDS, SPEC_NAMES, TargetSpec, sample_target_pool
from rl.trainer import train


def _build_target_pools(args: argparse.Namespace) -> tuple[tuple[TargetSpec, ...], tuple[TargetSpec, ...]]:
    """Training and (optionally) validation target pools.

    [ROUGH-SCALE ADAPTATION] default (--num-training-specs 1,
    --target-mode trivial) uses exactly ONE deliberately achievable target
    (TargetSpec.from_existing_thresholds()) for the first smoke milestone,
    per explicit instruction. --target-mode hard swaps in the deliberately
    harder, still real-design-verified target
    (TargetSpec.from_hard_target() -- see rl/target_spec.py and
    docs/autockt-mapping.md, section 14) for the single-target case;
    from_existing_thresholds() itself is unchanged either way. Passing
    --num-training-specs > 1 samples additional targets
    [AUTOCKT-REPLICATED sampling methodology, see target_spec.sample_target_pool]
    from ranges anchored at/above EXISTING_THRESHOLDS -- i.e. every sampled
    target is at least as demanding as the existing repo thresholds, never
    an arbitrarily invented easier or harder number. (--target-mode has no
    effect once --num-training-specs > 1; it only selects which single
    fixed target is used in the default single-target case.)

    [NEBULA ADAPTATION] --target-mode mixed is a third, separate case,
    checked first and independent of --num-training-specs/--num-validation-specs
    (both are ignored when it is selected): training pool =
    (TargetSpec.from_existing_thresholds(), TargetSpec.from_hard_target()) --
    the same two already real-SPICE-verified fixed targets used elsewhere in
    this file, nothing new. Validation pool = exactly one target, the
    per-dimension arithmetic midpoint of those two, held out (never appears
    in the training pool). This deliberately avoids sample_target_pool's
    continuous ranges above: inspection showed those ranges are degenerate
    for this repo's fixed topology -- every dimension's bound sits at or
    below Design A's (results/receiver_random_search_20_seed123.jsonl
    candidate_index 8, also the source of VERIFIED_INITIAL_PARAMETERS)
    already-achieved value, so Design A satisfies literally every target
    those ranges can produce, with no code change needed to demonstrate it
    (verified during inspection, not assumed). The midpoint uses only
    arithmetic on the two already-grounded constants (EXISTING_THRESHOLDS,
    HARD_TARGET_THRESHOLDS) -- no new numbers are invented. See
    docs/autockt-mapping.md for the full inspection record.
    """

    if args.target_mode == "mixed":
        trivial = TargetSpec.from_existing_thresholds()
        hard = TargetSpec.from_hard_target()
        midpoint = TargetSpec(**{
            name: (getattr(trivial, name) + getattr(hard, name)) / 2.0 for name in SPEC_NAMES
        })
        return (trivial, hard), (midpoint,)

    if args.num_training_specs <= 1 and args.num_validation_specs == 0:
        target = TargetSpec.from_hard_target() if args.target_mode == "hard" else TargetSpec.from_existing_thresholds()
        return (target,), ()

    # Ranges anchored at EXISTING_THRESHOLDS on the low end; the high end is
    # a modest multiple, wide enough to vary the target across episodes
    # without asking for anything unverified against the simulator's own
    # observed range (see results/receiver_random_search_20_seed123.jsonl
    # candidate_index 8, which reaches roughly 15x/2x/lots/0.07x these
    # thresholds on height/width/margin/power respectively).
    ranges = {
        "dfe_locked_phase_eye_height_v": (EXISTING_THRESHOLDS["dfe_locked_phase_eye_height_v"], 0.5),
        "dfe_eye_width_ui": (EXISTING_THRESHOLDS["dfe_eye_width_ui"], 0.8),
        "dfe_min_margin_v": (EXISTING_THRESHOLDS["dfe_min_margin_v"], 0.2),
        "ctle_power_w": (0.001, EXISTING_THRESHOLDS["ctle_power_w"]),
    }
    training_count = max(1, args.num_training_specs)
    training_pool = sample_target_pool(training_count, seed=args.seed, ranges=ranges)
    validation_pool: tuple[TargetSpec, ...] = ()
    if args.num_validation_specs > 0:
        # distinct seed => disjoint sampling stream from the training pool,
        # matching AutoCkt's own (manual, non-code-enforced) train/val split.
        validation_pool = sample_target_pool(
            args.num_validation_specs, seed=args.seed + 1_000_000, ranges=ranges
        )
    return training_pool, validation_pool


def _evaluate_checkpoint(
    *,
    label: str,
    policy_state: dict[str, object],
    agent: PPOAgent,
    adapter: ReceiverRLAdapter,
    training_pool: tuple[TargetSpec, ...],
    initial_indices: tuple[int, ...],
    grids: dict[str, ParameterGrid],
    horizon: int,
    randomize_initial_state: bool,
    seed: int,
    episodes: int,
    reward_fn=None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """[NEBULA ADAPTATION] Frozen-checkpoint evaluation, not part of the
    AutoCkt-replicated training loop itself. Runs `episodes` fully
    deterministic (agent.act(..., deterministic=True)) episodes through the
    real, untouched ReceiverRLAdapter with the given policy weights loaded,
    against a FRESH AutoCktReceiverEnv built with a fixed `seed` shared
    across every checkpoint evaluated in the same run -- so "initial" and
    "final" checkpoints see the identical sequence of (randomized, if
    enabled) starting points and targets, a matched, apples-to-apples
    comparison per docs/autockt-mapping.md's before/after methodology.
    Mutates `agent.policy` in place (loads `policy_state`); callers must
    restore whichever state they want left in `agent` afterward.
    """

    agent.policy.load_state_dict(policy_state)
    eval_env = AutoCktReceiverEnv(
        target_pool=training_pool,
        initial_indices=initial_indices,
        horizon=horizon,
        adapter=adapter,
        grids=grids,
        seed=seed,
        randomize_initial_state=randomize_initial_state,
        reward_fn=reward_fn,
    )
    rows: list[dict[str, object]] = []
    episode_rewards = []
    satisfied_count = 0
    for episode_index in range(episodes):
        state, reset_info = eval_env.reset()
        done = truncated = False
        episode_reward = 0.0
        steps = 0
        while not done and not truncated:
            choices, _log_prob, _value = agent.act(state, deterministic=True)
            step_out = eval_env.step(choices)
            episode_reward += step_out.reward
            state = step_out.state
            done, truncated = step_out.done, step_out.truncated
            steps += 1
        satisfied_count += int(done)
        episode_rewards.append(episode_reward)
        rows.append({
            "checkpoint": label,
            "episode": episode_index,
            "episode_reward": episode_reward,
            "spec_satisfied": done,
            "steps": steps,
            "target": reset_info["target"],
        })
    summary = {
        "checkpoint": label,
        "episodes": episodes,
        "mean_episode_reward": sum(episode_rewards) / max(1, len(episode_rewards)),
        "satisfaction_rate": satisfied_count / max(1, episodes),
        "total_evaluations_after": adapter.total_evaluations,
    }
    return summary, rows


def _evaluate_validation_pool(
    *,
    env: AutoCktReceiverEnv,
    agent: PPOAgent,
    validation_pool: tuple[TargetSpec, ...],
) -> list[dict[str, object]]:
    """[NEBULA ADAPTATION] Deterministic, zero-shot evaluation of each target
    in `validation_pool` against the CURRENT policy weights -- no PPO update
    happens in this function; `agent.act(..., deterministic=True)` only reads
    the policy. Extracted from the body of main() so it is independently
    testable (see tests/test_autockt_rl.py::ValidationPoolEvaluationTests)
    without invoking the CLI or real SPICE.

    Runs on the given `env` (same target_pool/adapter/grids/initial-state
    settings training used) but forces each episode's target via
    AutoCktReceiverEnv.reset(target=...), which bypasses target_pool sampling
    entirely -- this is how a target that is NOT a member of
    `env.target_pool` (e.g. --target-mode mixed's held-out midpoint) is
    evaluated: `reset()`'s own `target` keyword takes precedence over
    sampling from the pool (see rl/autockt_env.py::AutoCktReceiverEnv.reset),
    so this is a genuine zero-shot check, not an accidental re-sample from
    the training targets.
    """

    validation_rows: list[dict[str, object]] = []
    for index, target in enumerate(validation_pool):
        state, _ = env.reset(target=target)
        done = truncated = False
        episode_reward = 0.0
        while not done and not truncated:
            choices, _, _ = agent.act(state, deterministic=True)
            step_out = env.step(choices)
            episode_reward += step_out.reward
            state = step_out.state
            done, truncated = step_out.done, step_out.truncated
        validation_rows.append(
            {
                "validation_spec_index": index,
                "target": target.as_dict(),
                "episode_reward": episode_reward,
                "spec_satisfied": done,
            }
        )
    return validation_rows


def _resolve_initial_indices(
    grids: dict[str, ParameterGrid], source: str,
) -> tuple[int, ...]:
    """[NEBULA ADAPTATION] 'verified' (default): VERIFIED_INITIAL_PARAMETERS,
    a design seeded from Random Search's own results (see
    rl/parameter_grid.py) -- unchanged existing behavior. 'grid-center': the
    geometric/arithmetic midpoint index (`len(grid) // 2`) of every
    parameter's own grid, independent of any prior search result -- for
    experiments (e.g. docs/autockt-mapping.md sec 20) that must not
    warm-start PPO from a design Random Search already found.
    """

    if source == "grid-center":
        return tuple(len(grids[name]) // 2 for name in PARAMETER_NAMES)
    if source == "verified":
        return verified_initial_indices(grids)
    raise ValueError(f"unknown initial-indices source: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AutoCkt-style PPO training on the NEBULA receiver (see docs/autockt-mapping.md)"
    )
    parser.add_argument("--updates", type=int, default=2, help="number of PPO updates")
    parser.add_argument("--episodes-per-update", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=6, help="[ROUGH-SCALE ADAPTATION] AutoCkt uses 30")
    parser.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS)
    parser.add_argument(
        "--grid-spacing",
        choices=("linear", "log"),
        default="linear",
        help="[NEBULA ADAPTATION] 'linear' (default, unchanged AutoCkt-replicated behavior) "
        "uses np.arange for all 5 parameters. 'log' uses np.geomspace for the four parameters "
        "ACTION_BOUNDS already marks scale='log' (rload_ohm, rdeg_ohm, cdeg_f, itail_a); "
        "dfe_tap_v always stays linear. See rl/parameter_grid.py::build_parameter_grids.",
    )
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=16)
    parser.add_argument("--num-training-specs", type=int, default=1)
    parser.add_argument("--num-validation-specs", type=int, default=0)
    parser.add_argument(
        "--target-mode",
        choices=("trivial", "hard", "mixed"),
        default="trivial",
        help="'trivial'/'hard': which single fixed target to use when --num-training-specs <= 1 "
        "(see rl/target_spec.py TargetSpec.from_existing_thresholds / from_hard_target). "
        "[NEBULA ADAPTATION] 'mixed': ignores --num-training-specs/--num-validation-specs; "
        "trains on (trivial, hard) and validates zero-shot on their arithmetic midpoint, "
        "an unseen target never in the training pool. See _build_target_pools docstring.",
    )
    parser.add_argument(
        "--backend",
        choices=("real", "synthetic"),
        default="real",
        help="'real' (default, unchanged existing behavior) evaluates through the untouched "
        "simulator.receiver.evaluate_receiver -> real ngspice. 'synthetic' swaps in "
        "rl.synthetic_benchmark.synthetic_evaluate_receiver -- a fast, deterministic, "
        "NON-circuit-simulating stand-in used ONLY to validate PPO training mechanics; "
        "see rl/synthetic_benchmark.py for why this is never evidence about the receiver "
        "itself. Nothing above the ReceiverRLAdapter evaluator boundary changes either way.",
    )
    parser.add_argument(
        "--randomize-initial-state",
        action="store_true",
        default=False,
        help="[NEBULA ADAPTATION] default disabled (unchanged prior behavior: every episode "
        "starts at the exact fixed VERIFIED_INITIAL_PARAMETERS indices). When set, each "
        "episode's starting indices are independently perturbed by at most 1 grid index per "
        "parameter (symmetric {-1,0,+1}, clipped to grid bounds), deterministic under --seed. "
        "See rl/autockt_env.py::AutoCktReceiverEnv docstring.",
    )
    parser.add_argument(
        "--checkpoint-eval-episodes",
        type=int,
        default=0,
        help="[NEBULA ADAPTATION] default 0 (disabled, unchanged prior behavior). When > 0, "
        "freezes the policy's weights before training (the 'initial' checkpoint) and after "
        "training (the 'final' checkpoint), then runs this many fully deterministic real-SPICE "
        "episodes for each against a matched (same seed) fresh environment -- a cleaner "
        "trained-vs-untrained comparison than noisy training-time rewards. Adds "
        "2 * --checkpoint-eval-episodes more real evaluations (worst case: also * --horizon).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-evaluations", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("results/autockt_training.jsonl"))
    parser.add_argument(
        "--initial-indices-source", choices=("verified", "grid-center"), default="verified",
        help="[NEBULA ADAPTATION] 'verified' (default, unchanged existing behavior): "
        "VERIFIED_INITIAL_PARAMETERS, a design seeded from Random Search's own "
        "results/receiver_random_search_20_seed123.jsonl candidate_index 8 (see "
        "rl/parameter_grid.py). 'grid-center': the geometric/arithmetic midpoint index "
        "(len(grid)//2) of every parameter's grid -- independent of any prior search "
        "result, for experiments that must not warm-start PPO from a known-good design.",
    )
    parser.add_argument(
        "--reward-mode", choices=("terminal", "graded"), default="terminal",
        help="[NEBULA ADAPTATION] 'terminal' (default, unchanged existing behavior): "
        "rl.autockt_reward.autockt_reward, a flat FAILURE_REWARD on any failure. "
        "'graded': rl.autockt_reward.graded_autockt_reward -- identical on success; "
        "additionally grades a `transient`-stage failure (real per-spec distances are "
        "available) by the same relative-error-sum formula the success branch uses, "
        "instead of the same flat penalty every failure gets under 'terminal'. Earlier-"
        "stage failures (dc/ac/ctle_transient/channel, no real per-spec measurement) "
        "still get a fixed floor either way. See rl/autockt_reward.py for the full "
        "diagnosis this addresses (docs/autockt-mapping.md sec 20's zero-variance "
        "reward collapse).",
    )
    parser.add_argument(
        "--save-final-policy", type=Path, default=None,
        help="[NEBULA ADAPTATION -- pure I/O, no formulation change] optional path to "
        "torch.save the trained policy's state_dict after training completes. Default "
        "None (unchanged existing behavior: nothing is persisted to disk). Does not "
        "affect training in any way -- the save happens strictly after train() returns.",
    )
    args = parser.parse_args()

    training_pool, validation_pool = _build_target_pools(args)
    reward_fn = graded_autockt_reward if args.reward_mode == "graded" else None

    grids = build_parameter_grids(args.grid_points, spacing=args.grid_spacing)
    initial_indices = _resolve_initial_indices(grids, args.initial_indices_source)

    if args.backend == "synthetic":
        adapter = ReceiverRLAdapter(
            evaluator=synthetic_evaluate_receiver, budget=RLBudget(args.max_evaluations), seed=args.seed
        )
    else:
        adapter = ReceiverRLAdapter(budget=RLBudget(args.max_evaluations), seed=args.seed)
    env = AutoCktReceiverEnv(
        target_pool=training_pool,
        initial_indices=initial_indices,
        horizon=args.horizon,
        adapter=adapter,
        grids=grids,
        seed=args.seed,
        randomize_initial_state=args.randomize_initial_state,
        reward_fn=reward_fn,
    )
    agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=args.seed)
    initial_policy_state = copy.deepcopy(agent.policy.state_dict()) if args.checkpoint_eval_episodes > 0 else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_file = args.output.open("w", encoding="utf-8")

    def on_step(row: dict[str, object]) -> None:
        record = {
            **row,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "reward_version": AUTOCKT_REWARD_VERSION,
            "seed": args.seed,
        }
        output_file.write(json.dumps(record) + "\n")
        output_file.flush()
        print(json.dumps(record), flush=True)

    def on_update(row: dict[str, object]) -> None:
        print(json.dumps({"update_complete": True, **row}, indent=2), flush=True)

    print(
        json.dumps(
            {
                "run_start": True,
                "seed": args.seed,
                "reward_version": AUTOCKT_REWARD_VERSION,
                "action_schema_version": ACTION_SCHEMA_VERSION,
                "spec_names": SPEC_NAMES,
                "training_targets": [spec.as_dict() for spec in training_pool],
                "validation_targets": [spec.as_dict() for spec in validation_pool],
                "initial_indices": initial_indices,
                "initial_parameters": {
                    name: grids[name].value_at(index) for name, index in zip(PARAMETER_NAMES, initial_indices)
                },
                "horizon": args.horizon,
                "grid_points": args.grid_points,
                "grid_spacing": args.grid_spacing,
                "randomize_initial_state": args.randomize_initial_state,
                "reward_mode": args.reward_mode,
            },
            indent=2,
            default=list,
        ),
        flush=True,
    )

    result = train(
        env,
        agent,
        num_updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        on_step=on_step,
        on_update=on_update,
    )
    if args.save_final_policy is not None:
        args.save_final_policy.parent.mkdir(parents=True, exist_ok=True)
        torch.save(agent.policy.state_dict(), args.save_final_policy)
    checkpoint_summaries: list[dict[str, object]] = []
    if args.checkpoint_eval_episodes > 0:
        final_policy_state = copy.deepcopy(agent.policy.state_dict())
        checkpoint_seed = args.seed + 500_000
        for label, policy_state in (("initial", initial_policy_state), ("final", final_policy_state)):
            summary_row, episode_rows = _evaluate_checkpoint(
                label=label,
                policy_state=policy_state,
                agent=agent,
                adapter=adapter,
                training_pool=training_pool,
                initial_indices=initial_indices,
                grids=grids,
                horizon=args.horizon,
                randomize_initial_state=args.randomize_initial_state,
                seed=checkpoint_seed,
                episodes=args.checkpoint_eval_episodes,
                reward_fn=reward_fn,
            )
            checkpoint_summaries.append(summary_row)
            print(json.dumps({"checkpoint_complete": True, **summary_row}, indent=2), flush=True)
            with args.output.open("a", encoding="utf-8") as checkpoint_file:
                for row in episode_rows:
                    checkpoint_file.write(json.dumps(row) + "\n")
        # Leave the agent holding the fully-trained final policy regardless
        # of which checkpoint was evaluated last.
        agent.policy.load_state_dict(final_policy_state)

    validation_rows = _evaluate_validation_pool(env=env, agent=agent, validation_pool=validation_pool)

    summary = {
        "run_complete": True,
        "updates": args.updates,
        "total_evaluations": result.total_evaluations,
        "best_reward": result.best_reward,
        "best_parameters": result.best_parameters,
        "any_spec_satisfied_in_training": result.any_spec_satisfied,
        "validation_results": validation_rows,
        "checkpoint_evaluations": checkpoint_summaries,
        "seed": args.seed,
        "reward_mode": args.reward_mode,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
