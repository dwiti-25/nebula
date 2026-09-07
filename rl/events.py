"""Required change 6: complete PPO event data, as typed dictionaries.

This module defines the schema and a builder function only -- durable
storage and visualization belong to another team ("The other team will
own durable storage and visualization," per the RL subsystem's scope
boundary). rl/trainer.py gains an optional `on_event` hook
(collect_rollout/train) that emits one PPOStepEvent per step and one
PPOUpdateEvent per update when provided -- purely additive, default
None, the existing `on_step`/`on_update` callbacks (already used by
experiments/train_autockt.py's JSONL logging) are completely unchanged
and continue to fire exactly as before, independent of this hook.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

EVENT_SCHEMA_VERSION = 1


class PPOStepEvent(TypedDict):
    schema_version: int
    event_type: str  # always "step" -- discriminator for a mixed on_event stream
    run_id: str
    step: int
    episode: int
    update: int
    target_id: str
    target_values: dict[str, float]
    state_before: tuple[float, ...]
    state_after: tuple[float, ...]
    metrics_valid_mask: dict[str, bool]
    failure_stage: Optional[str]
    action_choice: tuple[int, ...]
    requested_delta: tuple[int, ...]
    applied_delta: tuple[int, ...]
    boundary_clipped: tuple[bool, ...]
    parameters_before: dict[str, float]
    parameters_after: dict[str, float]
    raw_metrics: dict[str, float]
    reward_components: dict[str, float]
    reward_total: float
    strict_pass: Optional[bool]
    log_prob: float
    value_estimate: float
    termination_reason: str  # "terminal_success" | "horizon_truncated" | "budget_truncated" | "ongoing"
    evaluation_count: int


class PPOUpdateEvent(TypedDict):
    schema_version: int
    event_type: str  # always "update" -- discriminator for a mixed on_event stream
    run_id: str
    update: int
    episodes: int
    transitions: int
    policy_loss: float
    value_loss: float
    entropy: float
    mean_episode_reward: float
    any_spec_satisfied_this_update: bool
    total_evaluations: int
    wall_clock_s: float


def _termination_reason(done: bool, truncated: bool, failure_stage: Optional[str]) -> str:
    if done:
        return "terminal_success"
    if failure_stage == "budget_exhausted":
        return "budget_truncated"
    if truncated:
        return "horizon_truncated"
    return "ongoing"


def build_step_event(
    *,
    run_id: str,
    step: int,
    episode: int,
    update: int,
    target_id: str,
    target_values: dict[str, float],
    state_before: tuple[float, ...],
    state_after: tuple[float, ...],
    metrics_valid_mask: dict[str, bool],
    action_choice: tuple[int, ...],
    action_deltas: tuple[int, int, int],
    indices_before: tuple[int, ...],
    indices_after: tuple[int, ...],
    parameters_before: dict[str, float],
    parameters_after: dict[str, float],
    raw_metrics: dict[str, float],
    reward_components: dict[str, float],
    reward_total: float,
    strict_pass: Optional[bool],
    failure_stage: Optional[str],
    log_prob: float,
    value_estimate: float,
    done: bool,
    truncated: bool,
    evaluation_count: int,
) -> PPOStepEvent:
    """Assembles one PPOStepEvent from pieces the caller (typically
    rl/trainer.py::collect_rollout, or an ablation harness driving the env
    directly) already has on hand -- computes nothing new about the
    circuit or the policy, just packages existing values into the typed
    shape. `requested_delta` is what `action_choice` selects from
    `action_deltas` before clipping; `applied_delta` is the actual index
    change after ParameterGrid.clip_index; `boundary_clipped[i]` is True
    exactly when they differ (the grid edge was hit).
    """

    requested_delta = tuple(action_deltas[choice] for choice in action_choice)
    applied_delta = tuple(after - before for after, before in zip(indices_after, indices_before))
    boundary_clipped = tuple(req != app for req, app in zip(requested_delta, applied_delta))

    return PPOStepEvent(
        schema_version=EVENT_SCHEMA_VERSION, event_type="step", run_id=run_id,
        step=step, episode=episode, update=update,
        target_id=target_id, target_values=target_values,
        state_before=state_before, state_after=state_after,
        metrics_valid_mask=metrics_valid_mask, failure_stage=failure_stage,
        action_choice=action_choice, requested_delta=requested_delta, applied_delta=applied_delta,
        boundary_clipped=boundary_clipped,
        parameters_before=parameters_before, parameters_after=parameters_after,
        raw_metrics=raw_metrics, reward_components=reward_components, reward_total=reward_total,
        strict_pass=strict_pass, log_prob=log_prob, value_estimate=value_estimate,
        termination_reason=_termination_reason(done, truncated, failure_stage),
        evaluation_count=evaluation_count,
    )


def build_update_event(
    *, run_id: str, update_row: dict[str, Any],
) -> PPOUpdateEvent:
    """Wraps an existing rl/trainer.py::train() update_row (already
    computed, unmodified) into the typed PPOUpdateEvent shape.
    """

    return PPOUpdateEvent(
        schema_version=EVENT_SCHEMA_VERSION, event_type="update", run_id=run_id, update=update_row["update"],
        episodes=update_row["episodes"], transitions=update_row["transitions"],
        policy_loss=update_row["policy_loss"], value_loss=update_row["value_loss"],
        entropy=update_row["entropy"], mean_episode_reward=update_row["mean_episode_reward"],
        any_spec_satisfied_this_update=update_row["any_spec_satisfied_this_update"],
        total_evaluations=update_row["total_evaluations"], wall_clock_s=update_row["wall_clock_s"],
    )
