"""Tests for rl/synthetic_benchmark.py -- the fast, deterministic,
NON-circuit-simulating PPO-training-mechanics benchmark.

These tests verify two separate things, kept clearly apart:
  1. The synthetic landscape itself behaves as documented (deterministic,
     has a good region and a bad/failure region, plugs cleanly into the
     unmodified ReceiverRLAdapter/AutoCktReceiverEnv contract).
  2. The PPO implementation can demonstrably learn something in this cheap,
     controlled environment.

Nothing here calls real ngspice, and nothing here is used to draw any
conclusion about actual receiver circuit performance -- see the module
docstring of rl/synthetic_benchmark.py.
"""

from __future__ import annotations

import math
import unittest

from simulator.receiver import EvaluationFidelity, ReceiverParameters
from simulator.rl_adapter import ACTION_BOUNDS, ReceiverRLAdapter, RLBudget

from rl.autockt_env import AutoCktReceiverEnv
from rl.autockt_reward import TERMINAL_BONUS
from rl.autockt_state import STATE_DIM
from rl.parameter_grid import PARAMETER_NAMES, build_parameter_grids, verified_initial_indices
from rl.ppo_agent import PPOAgent
from rl.synthetic_benchmark import (
    SYNTHETIC_CENTER,
    SYNTHETIC_FAILURE_DISTANCE,
    SYNTHETIC_FAILURE_STAGE,
    synthetic_evaluate_receiver,
    synthetic_goodness,
)
from rl.target_spec import TargetSpec
from rl.trainer import train


def _parameters_at_fraction(fraction: float) -> ReceiverParameters:
    """Builds a ReceiverParameters instance whose normalized position is
    `fraction` along every one of the 5 ACTION_BOUNDS dimensions (0.0 = the
    lower bound corner, 0.5 = SYNTHETIC_CENTER, 1.0 = the upper bound corner).
    """

    values = {}
    for name, lower, upper, scale in ACTION_BOUNDS:
        if scale == "log":
            values[name] = math.exp(math.log(lower) + fraction * math.log(upper / lower))
        else:
            values[name] = lower + fraction * (upper - lower)
    return ReceiverParameters(**values)


class SyntheticLandscapeTests(unittest.TestCase):
    def test_center_point_has_maximal_goodness_and_succeeds(self):
        center_params = _parameters_at_fraction(0.5)
        self.assertAlmostEqual(synthetic_goodness(center_params), 1.0, places=6)
        evaluation = synthetic_evaluate_receiver(center_params, fidelity=EvaluationFidelity.TRAINING)
        self.assertTrue(evaluation.success)
        self.assertIsNone(evaluation.failed_stage)

    def test_corner_points_are_far_and_report_synthetic_failure(self):
        low_corner = _parameters_at_fraction(0.0)
        high_corner = _parameters_at_fraction(1.0)
        for params in (low_corner, high_corner):
            self.assertLess(synthetic_goodness(params), 0.2)
            evaluation = synthetic_evaluate_receiver(params, fidelity=EvaluationFidelity.TRAINING)
            self.assertFalse(evaluation.success)
            self.assertEqual(evaluation.failed_stage, SYNTHETIC_FAILURE_STAGE)
            # never collides with a real ngspice failure_stage string
            self.assertNotIn(evaluation.failed_stage, ("dc", "ac", "ctle_transient", "channel", "noise", "hd3", "transient", "setup", "internal"))

    def test_goodness_decreases_monotonically_with_distance_from_center(self):
        fractions = [0.5, 0.6, 0.7, 0.8, 0.9]
        goodness_values = [synthetic_goodness(_parameters_at_fraction(f)) for f in fractions]
        for earlier, later in zip(goodness_values, goodness_values[1:]):
            self.assertLessEqual(later, earlier)

    def test_deterministic_repeated_calls_are_identical(self):
        params = _parameters_at_fraction(0.65)
        first = synthetic_evaluate_receiver(params, fidelity=EvaluationFidelity.TRAINING)
        second = synthetic_evaluate_receiver(params, fidelity=EvaluationFidelity.TRAINING)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.success, second.success)
        self.assertEqual(first.evaluation_id, second.evaluation_id)

    def test_has_both_a_success_and_a_failure_region(self):
        self.assertLess(SYNTHETIC_FAILURE_DISTANCE, math.sqrt(len(SYNTHETIC_CENTER) * 0.5**2))
        near = synthetic_evaluate_receiver(_parameters_at_fraction(0.5), fidelity=EvaluationFidelity.TRAINING)
        far = synthetic_evaluate_receiver(_parameters_at_fraction(1.0), fidelity=EvaluationFidelity.TRAINING)
        self.assertTrue(near.success)
        self.assertFalse(far.success)

    def test_plugs_cleanly_into_the_unmodified_receiver_rl_adapter(self):
        # Confirms the synthetic evaluator satisfies ReceiverRLAdapter's
        # existing, unmodified evaluator contract -- same shapes, same
        # finiteness guarantees as the real evaluator.
        adapter = ReceiverRLAdapter(evaluator=synthetic_evaluate_receiver, budget=RLBudget(5), seed=0)
        step = adapter.step([0.0] * len(ACTION_BOUNDS))
        self.assertEqual(len(step.observation), 20)
        self.assertTrue(all(math.isfinite(v) for v in step.observation))
        self.assertTrue(math.isfinite(step.reward))

    def test_plugs_cleanly_into_the_unmodified_autockt_env(self):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target = TargetSpec.from_hard_target()
        adapter = ReceiverRLAdapter(evaluator=synthetic_evaluate_receiver, budget=RLBudget(20), seed=0)
        env = AutoCktReceiverEnv(
            target_pool=(target,), initial_indices=indices, horizon=5, adapter=adapter, grids=grids, seed=0
        )
        state, _ = env.reset()
        self.assertEqual(len(state), STATE_DIM)
        step_out = env.step([1, 1, 1, 1, 1])
        self.assertTrue(math.isfinite(step_out.reward))


class SyntheticGradedEvaluatorTests(unittest.TestCase):
    """rl/synthetic_benchmark.py::synthetic_evaluate_receiver_graded --
    additive, PPO model-improvement study only. synthetic_evaluate_receiver
    itself (tested above) is completely unchanged.
    """

    def test_near_center_still_succeeds(self):
        from rl.synthetic_benchmark import synthetic_evaluate_receiver_graded
        result = synthetic_evaluate_receiver_graded(_parameters_at_fraction(0.5))
        self.assertTrue(result.success)
        self.assertIsNone(result.failed_stage)

    def test_middle_band_reports_transient_failure_with_real_metrics(self):
        from rl.synthetic_benchmark import synthetic_evaluate_receiver_graded
        # fraction=0.07 -> distance = sqrt(5)*0.43 ~= 0.962, inside (0.9, 1.05]
        result = synthetic_evaluate_receiver_graded(_parameters_at_fraction(0.07))
        self.assertFalse(result.success)
        self.assertEqual(result.failed_stage, "transient")
        self.assertIn("dfe_locked_phase_eye_height_v", result.metrics)

    def test_far_corner_reports_no_information_failure(self):
        from rl.synthetic_benchmark import synthetic_evaluate_receiver_graded
        result = synthetic_evaluate_receiver_graded(_parameters_at_fraction(1.0))
        self.assertFalse(result.success)
        self.assertEqual(result.failed_stage, "synthetic_out_of_region")
        self.assertEqual(result.metrics, {})

    def test_original_synthetic_evaluator_is_unaffected(self):
        # The original function's own behavior at the SAME far-corner point
        # must be byte-for-byte unchanged (binary success/no-info only).
        far = synthetic_evaluate_receiver(_parameters_at_fraction(1.0))
        self.assertEqual(far.failed_stage, SYNTHETIC_FAILURE_STAGE)
        self.assertEqual(far.metrics, {})


class SyntheticPPOLearningTests(unittest.TestCase):
    """Establishes whether the PPO implementation can learn at all, using
    the cheap synthetic backend. This says NOTHING about circuit
    optimization -- only about the PPO training loop's own mechanics.
    """

    def test_ppo_improves_mean_reward_over_updates_on_the_synthetic_landscape(self):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target = TargetSpec.from_hard_target()
        adapter = ReceiverRLAdapter(evaluator=synthetic_evaluate_receiver, budget=RLBudget(5000), seed=7)
        env = AutoCktReceiverEnv(
            target_pool=(target,), initial_indices=indices, horizon=5, adapter=adapter, grids=grids, seed=7
        )
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=7)

        result = train(env, agent, num_updates=15, episodes_per_update=12, ppo_epochs=4, minibatch_size=32)

        self.assertEqual(len(result.updates), 15)
        first_five = [row["mean_episode_reward"] for row in result.updates[:5]]
        last_five = [row["mean_episode_reward"] for row in result.updates[-5:]]
        mean_first = sum(first_five) / len(first_five)
        mean_last = sum(last_five) / len(last_five)
        # A soft, seeded statistical check (not a single-update comparison,
        # which the real-SPICE run already showed is too noisy to trust):
        # averaged over 5 updates at each end of training, later updates
        # should score at least as well as the earliest ones.
        self.assertGreaterEqual(mean_last, mean_first)

        # At least one update must show a nonzero policy gradient -- proof
        # the zero-advantage-variance stall seen in the real-SPICE run is
        # not a permanent property of this PPO implementation once batches
        # are large enough to (almost) always contain reward variance.
        nonzero_policy_loss_updates = sum(1 for row in result.updates if row["policy_loss"] != 0.0)
        self.assertGreater(nonzero_policy_loss_updates, len(result.updates) // 2)


if __name__ == "__main__":
    unittest.main()
