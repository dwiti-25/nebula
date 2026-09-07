"""SPICE-free tests for rl/events.py and its opt-in on_event wiring into
rl/trainer.py -- Required change 6.
"""

from __future__ import annotations

import unittest

from simulator.config import SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters, StageResult
from simulator.rl_adapter import METRIC_OBSERVATION_NAMES, ReceiverRLAdapter, RLBudget

from rl.autockt_env import AutoCktReceiverEnv
from rl.autockt_state import STATE_DIM
from rl.events import EVENT_SCHEMA_VERSION, build_step_event, build_update_event
from rl.parameter_grid import PARAMETER_NAMES, build_parameter_grids, verified_initial_indices
from rl.ppo_agent import PPOAgent
from rl.target_spec import TargetSpec
from rl.trainer import collect_rollout, train

KNOWN_GOOD_METRICS = {
    "dfe_locked_phase_eye_height_v": 1.5215334399802445,
    "dfe_eye_width_ui": 0.8699999999999999,
    "dfe_min_margin_v": 0.5174474651666459,
    "ctle_power_w": 0.0010850994,
}


def _fake_evaluation(success: bool = True, *, metrics: dict | None = None, failure_stage="dc") -> ReceiverEvaluation:
    values = dict(metrics or {})
    stage = StageResult("fake", success, 0.0, metrics=values)
    return ReceiverEvaluation(
        success, ReceiverParameters(), SimulationConditions(), EvaluationFidelity.TRAINING,
        (stage,), values, None if success else failure_stage, 0.0, "fake", {},
    )


class BuildStepEventTests(unittest.TestCase):
    def test_requested_vs_applied_delta_and_boundary_clip_flag(self):
        event = build_step_event(
            run_id="r1", step=1, episode=0, update=0, target_id="t", target_values={},
            state_before=(0.0,), state_after=(0.0,), metrics_valid_mask={}, action_choice=(2, 1, 0, 1, 1),
            action_deltas=(-1, 0, 2), indices_before=(20, 5, 5, 5, 5), indices_after=(20, 5, 4, 5, 5),
            parameters_before={}, parameters_after={}, raw_metrics={}, reward_components={},
            reward_total=-1.0, strict_pass=None, failure_stage="dc", log_prob=0.0, value_estimate=0.0,
            done=False, truncated=False, evaluation_count=1,
        )
        # choice 2 -> delta +2 requested for param 0, but index 20 is grid-edge -> applied 0, clipped True
        self.assertEqual(event["requested_delta"][0], 2)
        self.assertEqual(event["applied_delta"][0], 0)
        self.assertTrue(event["boundary_clipped"][0])
        # choice 0 -> delta -1 requested for param 2, applied -1, not clipped
        self.assertEqual(event["requested_delta"][2], -1)
        self.assertEqual(event["applied_delta"][2], -1)
        self.assertFalse(event["boundary_clipped"][2])

    def test_termination_reason_terminal_success(self):
        event = build_step_event(
            run_id="r", step=1, episode=0, update=0, target_id="t", target_values={},
            state_before=(), state_after=(), metrics_valid_mask={}, action_choice=(1,) * 5,
            action_deltas=(-1, 0, 2), indices_before=(0,) * 5, indices_after=(0,) * 5,
            parameters_before={}, parameters_after={}, raw_metrics={}, reward_components={},
            reward_total=10.0, strict_pass=True, failure_stage=None, log_prob=0.0, value_estimate=0.0,
            done=True, truncated=False, evaluation_count=1,
        )
        self.assertEqual(event["termination_reason"], "terminal_success")

    def test_termination_reason_horizon_truncated(self):
        event = build_step_event(
            run_id="r", step=1, episode=0, update=0, target_id="t", target_values={},
            state_before=(), state_after=(), metrics_valid_mask={}, action_choice=(1,) * 5,
            action_deltas=(-1, 0, 2), indices_before=(0,) * 5, indices_after=(0,) * 5,
            parameters_before={}, parameters_after={}, raw_metrics={}, reward_components={},
            reward_total=-1.0, strict_pass=None, failure_stage="dc", log_prob=0.0, value_estimate=0.0,
            done=False, truncated=True, evaluation_count=1,
        )
        self.assertEqual(event["termination_reason"], "horizon_truncated")

    def test_termination_reason_budget_truncated(self):
        event = build_step_event(
            run_id="r", step=1, episode=0, update=0, target_id="t", target_values={},
            state_before=(), state_after=(), metrics_valid_mask={}, action_choice=(1,) * 5,
            action_deltas=(-1, 0, 2), indices_before=(0,) * 5, indices_after=(0,) * 5,
            parameters_before={}, parameters_after={}, raw_metrics={}, reward_components={},
            reward_total=0.0, strict_pass=None, failure_stage="budget_exhausted", log_prob=0.0, value_estimate=0.0,
            done=False, truncated=True, evaluation_count=1,
        )
        self.assertEqual(event["termination_reason"], "budget_truncated")

    def test_event_type_discriminator(self):
        step_event = build_step_event(
            run_id="r", step=1, episode=0, update=0, target_id="t", target_values={},
            state_before=(), state_after=(), metrics_valid_mask={}, action_choice=(1,) * 5,
            action_deltas=(-1, 0, 2), indices_before=(0,) * 5, indices_after=(0,) * 5,
            parameters_before={}, parameters_after={}, raw_metrics={}, reward_components={},
            reward_total=0.0, strict_pass=None, failure_stage=None, log_prob=0.0, value_estimate=0.0,
            done=False, truncated=False, evaluation_count=1,
        )
        update_event = build_update_event(run_id="r", update_row={
            "update": 0, "episodes": 1, "transitions": 1, "policy_loss": 0.0, "value_loss": 0.0,
            "entropy": 0.0, "mean_episode_reward": 0.0, "any_spec_satisfied_this_update": False,
            "total_evaluations": 1, "wall_clock_s": 0.1,
        })
        self.assertEqual(step_event["event_type"], "step")
        self.assertEqual(update_event["event_type"], "update")
        self.assertEqual(step_event["schema_version"], EVENT_SCHEMA_VERSION)


class TrainerOnEventWiringTests(unittest.TestCase):
    def _make_env(self, evaluator, *, horizon=4, budget=50):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target_pool = (TargetSpec.from_existing_thresholds(),)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(budget), seed=0)
        return AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=horizon,
            adapter=adapter, grids=grids, seed=0,
        )

    def test_on_event_none_by_default_no_crash(self):
        env = self._make_env(lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS))
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)
        collect_rollout(env, agent, episodes=1)  # no on_event -- must not raise

    def test_collect_rollout_emits_one_step_event_per_step(self):
        env = self._make_env(lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS))
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)
        events = []
        collect_rollout(env, agent, episodes=1, on_event=events.append, run_id="run-1", update_index=7)
        self.assertGreaterEqual(len(events), 1)
        self.assertTrue(all(e["event_type"] == "step" for e in events))
        self.assertTrue(all(e["run_id"] == "run-1" for e in events))
        self.assertTrue(all(e["update"] == 7 for e in events))

    def test_train_emits_step_and_update_events(self):
        env = self._make_env(lambda *a, **k: _fake_evaluation(False, failure_stage="dc"), horizon=1)
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)
        events = []
        train(env, agent, num_updates=1, episodes_per_update=2, on_event=events.append, run_id="run-2")
        step_events = [e for e in events if e["event_type"] == "step"]
        update_events = [e for e in events if e["event_type"] == "update"]
        self.assertEqual(len(step_events), 2)  # 2 episodes x horizon 1
        self.assertEqual(len(update_events), 1)
        self.assertTrue(all(e["run_id"] == "run-2" for e in events))

    def test_existing_on_step_on_update_unaffected_by_on_event(self):
        env = self._make_env(lambda *a, **k: _fake_evaluation(False, failure_stage="dc"), horizon=1)
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)
        on_step_rows = []
        on_update_rows = []
        train(
            env, agent, num_updates=1, episodes_per_update=1,
            on_step=on_step_rows.append, on_update=on_update_rows.append, on_event=lambda e: None,
        )
        self.assertEqual(len(on_step_rows), 1)
        self.assertEqual(len(on_update_rows), 1)
        # unchanged shape -- these are NOT typed events, still the original plain rows
        self.assertIn("total_evaluation_count", on_step_rows[0])
        self.assertIn("policy_loss", on_update_rows[0])


if __name__ == "__main__":
    unittest.main()
