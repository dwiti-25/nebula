"""Rollout collection + PPO update loop, on top of the untouched
simulator.rl_adapter.ReceiverRLAdapter via rl/autockt_env.py.

[AUTOCKT-REPLICATED] rollout/update shape: collect a batch of on-policy
transitions against the AutoCkt-shaped environment, then run several PPO
epochs over that batch -- the standard on-policy PPO loop AutoCkt trains
with via Ray RLlib (`train_batch_size` in autockt/val_autobag_ray.py,
github.com/ksettaluri6/AutoCkt).

[ROUGH-SCALE ADAPTATION] batch size / episodes-per-update / epochs:
AutoCkt's own `train_batch_size=1200` assumes fast opamp SPICE. NEBULA's
receiver evaluation, measured directly in this session, costs ~15s (fast
DC-stage failure) to ~63s (a full successful TRAINING-fidelity evaluation)
per step -- see docs/autockt-mapping.md, "SPICE timing measurement". The
DEFAULT_* constants below are deliberately small so a full PPO update
(collect + learn) completes in minutes, not the hours AutoCkt-scale batches
would take here; they are explicitly not a claim of matching AutoCkt's own
batch size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

from .autockt_env import AutoCktReceiverEnv
from .events import build_step_event, build_update_event
from .parameter_grid import PARAMETER_NAMES
from .ppo_agent import PPOAgent, Transition

# [ROUGH-SCALE ADAPTATION] see module docstring.
DEFAULT_EPISODES_PER_UPDATE = 2
DEFAULT_PPO_EPOCHS = 4
DEFAULT_MINIBATCH_SIZE = 16


@dataclass
class EpisodeLog:
    episode_index: int
    target: dict[str, float]
    steps: list[dict[str, object]] = field(default_factory=list)
    episode_reward: float = 0.0
    spec_satisfied: bool = False


def collect_rollout(
    env: AutoCktReceiverEnv,
    agent: PPOAgent,
    *,
    episodes: int,
    episode_index_start: int = 0,
    on_step: Callable[[dict[str, object]], None] | None = None,
    on_event: Callable[[dict[str, object]], None] | None = None,
    run_id: str = "",
    update_index: int = 0,
) -> tuple[list[Transition], float, list[EpisodeLog]]:
    """Runs `episodes` full episodes (each up to env.horizon steps, or ending
    earlier via the AutoCkt-replicated `done` condition in
    rl/autockt_env.py), collecting on-policy transitions. Every step is one
    real, expensive evaluation through the untouched ReceiverRLAdapter --
    nothing here is mocked. Returns (transitions, bootstrap value of the
    final reached state, per-episode logs).
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    transitions: list[Transition] = []
    episode_logs: list[EpisodeLog] = []
    last_state: tuple[float, ...] = ()

    for offset in range(episodes):
        episode_index = episode_index_start + offset
        if env.budget_exhausted():
            # [NEBULA ADAPTATION -- Required change 4] the env's own
            # step()/reset() already refuse to make a further simulator
            # call once the budget is exhausted (see rl/autockt_env.py),
            # so this is a pure efficiency stop, not a correctness fix --
            # avoids collecting further reset+truncated-step cycles that
            # can never make real progress once no evaluations remain.
            break
        state, reset_info = env.reset()
        episode_log = EpisodeLog(episode_index=episode_index, target=reset_info["target"])
        done = False
        truncated = False
        while not done and not truncated:
            choices, log_prob, value = agent.act(state)
            indices_before = env.indices
            parameters_before = {
                name: env.grids[name].value_at(index) for name, index in zip(PARAMETER_NAMES, indices_before)
            }
            step_out = env.step(choices)
            if on_event is not None:
                # [NEBULA ADAPTATION -- Required change 6] purely additive:
                # existing on_step/on_update logging (below, and in
                # train()) is completely unaffected either way.
                reward_result = step_out.info.get("reward_result")
                on_event(build_step_event(
                    run_id=run_id, step=step_out.info["step_count"], episode=episode_index,
                    update=update_index, target_id=str(reset_info["target"]), target_values=reset_info["target"],
                    state_before=state, state_after=step_out.state,
                    metrics_valid_mask=step_out.info["metrics_valid_mask"],
                    action_choice=tuple(choices), action_deltas=env.action_deltas,
                    indices_before=indices_before, indices_after=step_out.info["indices"],
                    parameters_before=parameters_before, parameters_after=step_out.info["parameters"] or {},
                    raw_metrics=step_out.info["metrics"],
                    reward_components=reward_result.components if reward_result else {},
                    reward_total=step_out.reward,
                    strict_pass=reward_result.strict_target_pass if reward_result else None,
                    failure_stage=step_out.info["failure_stage"],
                    log_prob=log_prob, value_estimate=value,
                    done=step_out.done, truncated=step_out.truncated,
                    evaluation_count=step_out.info["total_evaluation_count"],
                ))
            bootstrap_value = 0.0
            if step_out.truncated and not step_out.done:
                # [NEBULA ADAPTATION -- repair #2] Horizon cutoff, not a
                # true episode end: capture the value net's own estimate of
                # the actual post-episode state so PPOAgent.compute_gae can
                # bootstrap this transition correctly instead of treating it
                # as terminal (see rl/ppo_agent.py::Transition docstring).
                bootstrap_value = agent.act(step_out.state)[2]
            transitions.append(
                Transition(
                    state=state,
                    choices=choices,
                    log_prob=log_prob,
                    value=value,
                    reward=step_out.reward,
                    terminated=step_out.done,
                    truncated=step_out.truncated,
                    bootstrap_value=bootstrap_value,
                )
            )
            row = {
                "episode": episode_index,
                "step": step_out.info["step_count"],
                "total_evaluation_count": step_out.info["total_evaluation_count"],
                "reward": step_out.reward,
                "success": step_out.info["success"],
                "failure_stage": step_out.info["failure_stage"],
                "spec_satisfied": step_out.info["spec_satisfied"],
                "parameters": step_out.info["parameters"],
                "indices": step_out.info["indices"],
                "target": reset_info["target"],
            }
            episode_log.steps.append(row)
            episode_log.episode_reward += step_out.reward
            if on_step is not None:
                on_step(row)
            state = step_out.state
            done = step_out.done
            truncated = step_out.truncated
        episode_log.spec_satisfied = done
        episode_logs.append(episode_log)
        last_state = state

    bootstrap_value = agent.act(last_state)[2] if last_state else 0.0
    return transitions, bootstrap_value, episode_logs


@dataclass
class TrainResult:
    updates: list[dict[str, object]] = field(default_factory=list)
    total_evaluations: int = 0
    best_reward: float = float("-inf")
    best_parameters: dict[str, float] | None = None
    any_spec_satisfied: bool = False


def train(
    env: AutoCktReceiverEnv,
    agent: PPOAgent,
    *,
    num_updates: int,
    episodes_per_update: int = DEFAULT_EPISODES_PER_UPDATE,
    ppo_epochs: int = DEFAULT_PPO_EPOCHS,
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    on_step: Callable[[dict[str, object]], None] | None = None,
    on_update: Callable[[dict[str, object]], None] | None = None,
    on_event: Callable[[dict[str, object]], None] | None = None,
    run_id: str = "",
) -> TrainResult:
    result = TrainResult()
    episode_index = 0
    for update_index in range(num_updates):
        t0 = time.perf_counter()
        transitions, bootstrap_value, episode_logs = collect_rollout(
            env,
            agent,
            episodes=episodes_per_update,
            episode_index_start=episode_index,
            on_step=on_step,
            on_event=on_event,
            run_id=run_id,
            update_index=update_index,
        )
        episode_index += len(episode_logs)
        stats = agent.update(transitions, bootstrap_value, epochs=ppo_epochs, minibatch_size=minibatch_size)
        wall_clock_s = time.perf_counter() - t0

        for log in episode_logs:
            for row in log.steps:
                result.total_evaluations = max(result.total_evaluations, row["total_evaluation_count"])
                if row["reward"] > result.best_reward:
                    result.best_reward = row["reward"]
                    result.best_parameters = row["parameters"]
                if row["spec_satisfied"]:
                    result.any_spec_satisfied = True

        update_row = {
            "update": update_index,
            "episodes": len(episode_logs),
            "transitions": len(transitions),
            "mean_episode_reward": sum(log.episode_reward for log in episode_logs) / max(1, len(episode_logs)),
            "any_spec_satisfied_this_update": any(log.spec_satisfied for log in episode_logs),
            "total_evaluations": result.total_evaluations,
            "wall_clock_s": wall_clock_s,
            **stats,
        }
        result.updates.append(update_row)
        if on_update is not None:
            on_update(update_row)
        if on_event is not None:
            on_event(build_update_event(run_id=run_id, update_row=update_row))
    return result
