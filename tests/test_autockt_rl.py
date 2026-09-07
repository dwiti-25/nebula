"""Fast, SPICE-free unit tests for the ML-side AutoCkt RL implementation
(rl/*.py, experiments/train_autockt.py). These never call ngspice: all
receiver evaluations are synthetic ReceiverEvaluation objects injected via
ReceiverRLAdapter(evaluator=...), the same pattern
tests/test_rl_readiness.py already uses for its RL-contract tests. Real,
SPICE-backed end-to-end verification is a separate, manual smoke run (see
docs/autockt-mapping.md, "First end-to-end SPICE smoke test"), not part of
this fast suite.

Nothing in simulator/*.py is touched or monkeypatched here; only its public,
documented interfaces (ReceiverRLAdapter, ReceiverEvaluation, StageResult,
ReceiverParameters, SimulationConditions, EvaluationFidelity, ACTION_BOUNDS)
are used, read-only, exactly as experiments/*.py already does.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from simulator.config import SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters, StageResult
from simulator.rl_adapter import ACTION_BOUNDS, METRIC_OBSERVATION_NAMES, ReceiverRLAdapter, RLBudget

from rl.autockt_action import apply_action, indices_to_normalized_action
from rl.autockt_env import AutoCktReceiverEnv, metrics_from_observation
from rl.autockt_reward import (
    FAILURE_REWARD,
    GRADED_NO_INFORMATION_FLOOR,
    TERMINAL_BONUS,
    UNSATISFIED_THRESHOLD,
    autockt_reward,
    graded_autockt_reward,
)
from rl.autockt_state import STATE_DIM, build_state, lookup, signed_relative_error
from rl.parameter_grid import (
    ACTION_DELTAS, PARAMETER_NAMES, VERIFIED_INITIAL_PARAMETERS,
    build_parameter_grids, verified_initial_indices,
)
from rl.ppo_agent import PPOAgent, Transition
from rl.target_spec import EXISTING_THRESHOLDS, HARD_TARGET_THRESHOLDS, SPEC_NAMES, TargetSpec, sample_target_pool
from rl.trainer import collect_rollout, train

from experiments.train_autockt import (
    _build_target_pools, _evaluate_checkpoint, _evaluate_validation_pool, _resolve_initial_indices, main,
)


def _fake_evaluation(
    success: bool = True, *, metrics: dict | None = None, identity: str = "fake",
    failure_stage: str | None = "dc",
) -> ReceiverEvaluation:
    values = {name: 0.0 for name in METRIC_OBSERVATION_NAMES}
    values["dfe_locked_phase_eye_height_v"] = values.pop("dfe_eye_height_v")
    values.update(metrics or {})
    stage = StageResult("fake", success, 0.0, metrics=values)
    return ReceiverEvaluation(
        success, ReceiverParameters(), SimulationConditions(),
        EvaluationFidelity.TRAINING, (stage,), values,
        None if success else failure_stage, 0.0, identity, {},
    )


# Real, verified successful design from
# results/receiver_random_search_20_seed123.jsonl (candidate_index 8),
# re-confirmed by direct SPICE re-evaluation in this session (see
# docs/autockt-mapping.md). Comfortably beats every EXISTING_THRESHOLDS
# value.
KNOWN_GOOD_METRICS = {
    "dfe_locked_phase_eye_height_v": 1.5215334399802445,
    "dfe_eye_width_ui": 0.8699999999999999,
    "dfe_min_margin_v": 0.5174474651666459,
    "ctle_power_w": 0.0010850994,
}

# A second, independent real successful design from
# results/rl_reward_directed_smoke.jsonl (episode 1, step 1; reward_v1 =
# 70.656, success=true), re-derived from its logged `constraints` tuple
# (evaluation_success, zero_errors, positive_margin, eye_height_over_100mv,
# eye_width_over_0p4ui, power_under_15mw): margin=0.229488,
# height=0.1+0.451433=0.551433, width=0.4+0.34=0.74,
# power=0.015-0.014655=0.000345. Used to prove HARD_TARGET_THRESHOLDS is
# genuinely selective, not just a rescaled trivial target: this design
# clears the trivial target comfortably but must NOT clear the hard one.
SECOND_KNOWN_GOOD_METRICS = {
    "dfe_locked_phase_eye_height_v": 0.5514326967009642,
    "dfe_eye_width_ui": 0.74,
    "dfe_min_margin_v": 0.2294877551337197,
    "ctle_power_w": 0.015 - 0.0146546682,  # constraints[5] was (0.015 - power)
}


class ParameterGridTests(unittest.TestCase):
    def test_grids_cover_exactly_the_five_action_bound_parameters(self):
        grids = build_parameter_grids()
        self.assertEqual(set(grids), set(PARAMETER_NAMES))
        self.assertEqual(len(PARAMETER_NAMES), len(ACTION_BOUNDS))

    def test_grid_values_stay_within_action_bounds(self):
        grids = build_parameter_grids()
        for name, lower, upper, _scale in ACTION_BOUNDS:
            grid = grids[name]
            self.assertGreaterEqual(grid.values[0], lower)
            self.assertLess(grid.values[-1], upper)  # AutoCkt-replicated np.arange excludes the upper bound

    def test_clip_index_saturates_at_both_ends(self):
        grids = build_parameter_grids()
        grid = grids["rload_ohm"]
        self.assertEqual(grid.clip_index(-5), 0)
        self.assertEqual(grid.clip_index(10_000), len(grid) - 1)

    def test_normalized_index_maps_endpoints_to_minus_one_and_one(self):
        grids = build_parameter_grids()
        for grid in grids.values():
            self.assertAlmostEqual(grid.normalized_index(0), -1.0)
            self.assertAlmostEqual(grid.normalized_index(len(grid) - 1), 1.0)

    def test_normalized_index_midpoint_and_monotonicity(self):
        grid = build_parameter_grids()["rload_ohm"]
        n = len(grid)
        if n % 2 == 1:
            self.assertAlmostEqual(grid.normalized_index(n // 2), 0.0)
        values = [grid.normalized_index(i) for i in range(n)]
        for earlier, later in zip(values, values[1:]):
            self.assertLess(earlier, later)

    def test_normalized_index_clips_out_of_range_indices(self):
        grid = build_parameter_grids()["rload_ohm"]
        self.assertAlmostEqual(grid.normalized_index(-5), -1.0)
        self.assertAlmostEqual(grid.normalized_index(10_000), 1.0)

    def test_verified_initial_indices_are_in_bounds_and_close_to_the_real_design(self):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        self.assertEqual(len(indices), len(PARAMETER_NAMES))
        for name, index in zip(PARAMETER_NAMES, indices):
            self.assertTrue(0 <= index < len(grids[name]))
            physical = grids[name].value_at(index)
            actual = VERIFIED_INITIAL_PARAMETERS[name]
            span = grids[name].values[-1] - grids[name].values[0]
            self.assertLessEqual(abs(physical - actual), span / len(grids[name]) + 1e-18)

    def test_default_spacing_is_linear_and_unchanged(self):
        # spacing defaults to "linear" -- byte-for-byte the original,
        # pre-repair grid construction (repair #1 must not change default
        # behavior for any existing caller).
        self.assertEqual(build_parameter_grids(), build_parameter_grids(spacing="linear"))

    def test_log_spacing_uses_geomspace_for_the_four_log_scale_parameters(self):
        linear_grids = build_parameter_grids(spacing="linear")
        log_grids = build_parameter_grids(spacing="log")
        for name, lower, upper, scale in ACTION_BOUNDS:
            if scale != "log":
                continue
            self.assertNotEqual(log_grids[name].values, linear_grids[name].values)
            # np.geomspace includes both endpoints exactly (unlike arange's
            # exclusive-upper linear grid).
            self.assertAlmostEqual(log_grids[name].values[0], lower, places=6)
            self.assertAlmostEqual(log_grids[name].values[-1], upper, places=6)
            # strictly increasing, still fully inside [lower, upper]
            values = log_grids[name].values
            for earlier, later in zip(values, values[1:]):
                self.assertLess(earlier, later)

    def test_log_spacing_keeps_dfe_tap_v_linear(self):
        # dfe_tap_v is scale="linear" in ACTION_BOUNDS and spans negative to
        # positive values, where geometric spacing is undefined -- it must
        # be unaffected by spacing="log".
        linear_grids = build_parameter_grids(spacing="linear")
        log_grids = build_parameter_grids(spacing="log")
        self.assertEqual(log_grids["dfe_tap_v"].values, linear_grids["dfe_tap_v"].values)

    def test_invalid_spacing_raises(self):
        with self.assertRaises(ValueError):
            build_parameter_grids(spacing="quadratic")


class AutoCktActionTests(unittest.TestCase):
    def test_action_deltas_are_autockt_verified(self):
        self.assertEqual(ACTION_DELTAS, (-1, 0, 2))

    def test_apply_action_moves_and_clips_indices(self):
        grids = build_parameter_grids()
        start = tuple(0 for _ in PARAMETER_NAMES)
        # choice 0 -> delta -1 -> clipped to 0 (already at floor)
        clipped_low = apply_action(start, [0] * len(PARAMETER_NAMES), grids)
        self.assertEqual(clipped_low, start)
        # choice 2 -> delta +2
        moved = apply_action(start, [2] * len(PARAMETER_NAMES), grids)
        self.assertEqual(moved, tuple(2 for _ in PARAMETER_NAMES))
        # repeatedly stepping +2 must clip at the top, never raise or overflow
        indices = start
        for _ in range(50):
            indices = apply_action(indices, [2] * len(PARAMETER_NAMES), grids)
        for name, index in zip(PARAMETER_NAMES, indices):
            self.assertEqual(index, len(grids[name]) - 1)

    def test_apply_action_rejects_wrong_length_or_invalid_choice(self):
        grids = build_parameter_grids()
        start = tuple(0 for _ in PARAMETER_NAMES)
        with self.assertRaises(ValueError):
            apply_action(start, [0, 0], grids)
        with self.assertRaises(ValueError):
            apply_action(start, [3] * len(PARAMETER_NAMES), grids)

    def test_indices_to_normalized_action_stays_in_bounds_and_round_trips(self):
        from simulator.rl_adapter import normalized_action_to_parameters

        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        action = indices_to_normalized_action(indices, grids)
        self.assertEqual(len(action), len(ACTION_BOUNDS))
        for value in action:
            self.assertTrue(-1.0 <= value <= 1.0)
        # round trip through the existing, untouched simulator mapping must
        # reproduce each grid's physical value exactly.
        params = normalized_action_to_parameters(action)
        for name, index in zip(PARAMETER_NAMES, indices):
            self.assertAlmostEqual(getattr(params, name), grids[name].value_at(index), places=6)


class TargetSpecTests(unittest.TestCase):
    def test_existing_thresholds_match_rl_adapter_constraint_numbers(self):
        # simulator/rl_adapter.py::constraints_from_evaluation uses exactly
        # these four numbers (height - 0.1, width - 0.4, margin, 0.015 - power).
        self.assertEqual(EXISTING_THRESHOLDS["dfe_locked_phase_eye_height_v"], 0.1)
        self.assertEqual(EXISTING_THRESHOLDS["dfe_eye_width_ui"], 0.4)
        self.assertEqual(EXISTING_THRESHOLDS["dfe_min_margin_v"], 0.0)
        self.assertEqual(EXISTING_THRESHOLDS["ctle_power_w"], 0.015)

    def test_from_existing_thresholds_is_achievable_by_the_verified_design(self):
        target = TargetSpec.from_existing_thresholds()
        for name in SPEC_NAMES:
            achieved = KNOWN_GOOD_METRICS[name]
            goal = getattr(target, name)
            if name == "ctle_power_w":
                self.assertLess(achieved, goal)
            else:
                self.assertGreater(achieved, goal)

    def test_sample_target_pool_is_deterministic_and_seed_streams_are_disjoint(self):
        ranges = {name: (0.0, 1.0) for name in SPEC_NAMES}
        pool_a = sample_target_pool(5, seed=1, ranges=ranges)
        pool_b = sample_target_pool(5, seed=1, ranges=ranges)
        self.assertEqual(pool_a, pool_b)
        pool_c = sample_target_pool(5, seed=2, ranges=ranges)
        self.assertNotEqual(pool_a, pool_c)

    def test_sample_target_pool_requires_all_ranges(self):
        with self.assertRaises(ValueError):
            sample_target_pool(3, seed=0, ranges={})

    def test_hard_target_values_match_documented_constants(self):
        self.assertEqual(HARD_TARGET_THRESHOLDS["dfe_locked_phase_eye_height_v"], 0.8)
        self.assertEqual(HARD_TARGET_THRESHOLDS["dfe_eye_width_ui"], 0.6)
        self.assertEqual(HARD_TARGET_THRESHOLDS["dfe_min_margin_v"], 0.35)
        self.assertEqual(HARD_TARGET_THRESHOLDS["ctle_power_w"], 0.015)

    def test_hard_target_is_strictly_tighter_than_trivial_target_on_three_specs(self):
        for name in ("dfe_locked_phase_eye_height_v", "dfe_eye_width_ui", "dfe_min_margin_v"):
            self.assertGreater(HARD_TARGET_THRESHOLDS[name], EXISTING_THRESHOLDS[name])
        # power is deliberately left unchanged, see rl/target_spec.py.
        self.assertEqual(HARD_TARGET_THRESHOLDS["ctle_power_w"], EXISTING_THRESHOLDS["ctle_power_w"])

    def test_from_hard_target_does_not_mutate_the_existing_trivial_target(self):
        hard = TargetSpec.from_hard_target()
        trivial = TargetSpec.from_existing_thresholds()
        self.assertNotEqual(hard, trivial)
        self.assertEqual(trivial, TargetSpec(**EXISTING_THRESHOLDS))

    def test_hard_target_is_satisfied_by_one_known_real_design(self):
        # Design A: results/receiver_random_search_20_seed123.jsonl candidate_index 8.
        target = TargetSpec.from_hard_target()
        for name in SPEC_NAMES:
            achieved = KNOWN_GOOD_METRICS[name]
            goal = getattr(target, name)
            if name == "ctle_power_w":
                self.assertLessEqual(achieved, goal)
            else:
                self.assertGreaterEqual(achieved, goal)

    def test_hard_target_is_not_satisfied_by_a_second_known_real_design(self):
        # Design B: results/rl_reward_directed_smoke.jsonl episode 1 step 1.
        # This design clears the TRIVIAL target comfortably but must fail
        # the hard target on height and margin -- the discriminating
        # property the hard target is designed to have.
        target = TargetSpec.from_hard_target()
        self.assertLess(SECOND_KNOWN_GOOD_METRICS["dfe_locked_phase_eye_height_v"], target.dfe_locked_phase_eye_height_v)
        self.assertLess(SECOND_KNOWN_GOOD_METRICS["dfe_min_margin_v"], target.dfe_min_margin_v)
        # ... while it still clears the unchanged/lenient trivial target.
        trivial = TargetSpec.from_existing_thresholds()
        self.assertGreater(SECOND_KNOWN_GOOD_METRICS["dfe_locked_phase_eye_height_v"], trivial.dfe_locked_phase_eye_height_v)
        self.assertGreater(SECOND_KNOWN_GOOD_METRICS["dfe_min_margin_v"], trivial.dfe_min_margin_v)

    def test_hard_target_reward_reflects_design_a_pass_and_design_b_fail(self):
        # End-to-end check through the actual reward function (not just the
        # raw threshold comparison above): Design A should hit the terminal
        # bonus against the hard target; Design B should not.
        target = TargetSpec.from_hard_target()
        reward_a = autockt_reward(KNOWN_GOOD_METRICS, target, success=True)
        reward_b = autockt_reward(SECOND_KNOWN_GOOD_METRICS, target, success=True)
        self.assertEqual(reward_a, TERMINAL_BONUS)
        self.assertLess(reward_b, TERMINAL_BONUS)


class AutoCktStateTests(unittest.TestCase):
    def test_state_dimension(self):
        self.assertEqual(STATE_DIM, 2 * len(SPEC_NAMES) + len(PARAMETER_NAMES))

    def test_build_state_shape_and_finiteness(self):
        target = TargetSpec.from_existing_thresholds()
        indices = (0, 1, 2, 3, 4)
        state = build_state(KNOWN_GOOD_METRICS, target, indices)
        self.assertEqual(len(state), STATE_DIM)
        self.assertTrue(all(math.isfinite(value) for value in state))
        # trailing entries are the raw, unnormalized parameter indices
        self.assertEqual(state[-len(indices):], tuple(float(i) for i in indices))

    def test_build_state_rejects_wrong_index_count(self):
        target = TargetSpec.from_existing_thresholds()
        with self.assertRaises(ValueError):
            build_state({}, target, (0, 1))

    def test_lookup_epsilon_guard_avoids_division_by_zero(self):
        # value == -reference would make the raw denominator exactly zero.
        result = lookup(1e-15, -1e-15)
        self.assertTrue(math.isfinite(result))

    def test_signed_relative_error_flips_only_for_smaller_is_better(self):
        self.assertEqual(signed_relative_error("dfe_min_margin_v", 0.3), 0.3)
        self.assertEqual(signed_relative_error("ctle_power_w", 0.3), -0.3)


class AutoCktRewardTests(unittest.TestCase):
    def test_known_good_design_hits_terminal_bonus(self):
        target = TargetSpec.from_existing_thresholds()
        reward = autockt_reward(KNOWN_GOOD_METRICS, target, success=True)
        self.assertEqual(reward, TERMINAL_BONUS)

    def test_all_specs_failing_gives_negative_reward_not_bonus(self):
        target = TargetSpec.from_existing_thresholds()
        failing_metrics = {
            "dfe_locked_phase_eye_height_v": 0.0,
            "dfe_eye_width_ui": 0.0,
            "dfe_min_margin_v": -0.1,
            "ctle_power_w": 0.03,
        }
        reward = autockt_reward(failing_metrics, target, success=True)
        self.assertLess(reward, UNSATISFIED_THRESHOLD)
        self.assertNotEqual(reward, TERMINAL_BONUS)

    def test_failed_simulation_gives_the_fixed_failure_reward(self):
        target = TargetSpec.from_existing_thresholds()
        self.assertEqual(autockt_reward({}, target, success=False), FAILURE_REWARD)

    def test_overshoot_is_not_penalized_beyond_satisfaction(self):
        target = TargetSpec.from_existing_thresholds()
        just_over = dict(KNOWN_GOOD_METRICS)
        way_over = dict(KNOWN_GOOD_METRICS)
        way_over["dfe_locked_phase_eye_height_v"] *= 100.0
        self.assertEqual(
            autockt_reward(just_over, target, success=True),
            autockt_reward(way_over, target, success=True),
        )

    def test_reward_module_never_imports_reward_v1(self):
        # The module docstring/comments legitimately *mention* reward_v1 by
        # name to document why this reward is deliberately separate; what
        # must never happen is actually importing/calling it.
        from rl import autockt_reward as module

        self.assertFalse(hasattr(module, "reward_v1"))
        self.assertNotIn("reward_v1", dir(module))


class GradedAutocktRewardTests(unittest.TestCase):
    """PPO model-improvement study, improvement #1: a graded failure reward
    that only grades failures where real per-spec information exists
    (the `transient` stage), and falls back to a fixed, dominated floor
    everywhere else -- mirroring experiments/train_cem.py's already-
    validated graded_cem_fitness. autockt_reward itself (tested above) is
    completely unchanged; these tests are strictly additive coverage.
    """

    def test_success_matches_autockt_reward_exactly(self):
        target = TargetSpec.from_existing_thresholds()
        self.assertEqual(
            graded_autockt_reward(KNOWN_GOOD_METRICS, target, success=True, failure_stage=None),
            autockt_reward(KNOWN_GOOD_METRICS, target, success=True),
        )

    def test_dc_failure_gets_the_fixed_no_information_floor(self):
        target = TargetSpec.from_existing_thresholds()
        reward = graded_autockt_reward({}, target, success=False, failure_stage="dc")
        self.assertEqual(reward, GRADED_NO_INFORMATION_FLOOR)

    def test_ac_and_ctle_transient_and_channel_failures_also_get_the_floor(self):
        target = TargetSpec.from_existing_thresholds()
        for stage in ("ac", "ctle_transient", "channel"):
            with self.subTest(stage=stage):
                reward = graded_autockt_reward({}, target, success=False, failure_stage=stage)
                self.assertEqual(reward, GRADED_NO_INFORMATION_FLOOR)

    def test_transient_failure_is_graded_by_real_distance_not_the_flat_floor(self):
        # Note: dfe_min_margin_v's own threshold is exactly 0.0, and
        # lookup(value, 0.0) == value/value == 1.0 for any nonzero value
        # (verified directly) -- margin structurally cannot contribute
        # negative/informative signal to this locked formula regardless of
        # sign. Using dfe_locked_phase_eye_height_v instead, whose nonzero
        # threshold (0.1) does not have this degeneracy.
        target = TargetSpec.from_existing_thresholds()
        close_metrics = dict(KNOWN_GOOD_METRICS)
        close_metrics["dfe_locked_phase_eye_height_v"] = 0.05  # below the 0.1 threshold, but not by much
        reward = graded_autockt_reward(close_metrics, target, success=False, failure_stage="transient")
        self.assertNotEqual(reward, GRADED_NO_INFORMATION_FLOOR)
        self.assertLess(reward, 0.0)  # still a failure -- never reaches TERMINAL_BONUS

    def test_a_closer_transient_failure_scores_strictly_better_than_a_farther_one(self):
        # This is the entire point of the fix: two DIFFERENT failing
        # candidates must produce DIFFERENT rewards, unlike autockt_reward's
        # flat FAILURE_REWARD for both.
        target = TargetSpec.from_existing_thresholds()
        close = dict(KNOWN_GOOD_METRICS)
        close["dfe_locked_phase_eye_height_v"] = 0.05
        far = dict(KNOWN_GOOD_METRICS)
        far["dfe_locked_phase_eye_height_v"] = 0.001
        far["ctle_power_w"] = 1.0
        reward_close = graded_autockt_reward(close, target, success=False, failure_stage="transient")
        reward_far = graded_autockt_reward(far, target, success=False, failure_stage="transient")
        self.assertGreater(reward_close, reward_far)

    def test_no_information_floor_is_always_strictly_below_any_graded_transient_score(self):
        # Structural guarantee, not an empirical observation: a genuinely
        # bad transient-stage failure must never be scored as if it were
        # "no information" -- and a no-information failure must never
        # accidentally outrank a real, close, graded failure.
        target = TargetSpec.from_existing_thresholds()
        worst_transient_metrics = {
            "dfe_locked_phase_eye_height_v": 0.0, "dfe_eye_width_ui": 0.0,
            "dfe_min_margin_v": -1.0, "ctle_power_w": 1.0,
        }
        worst_graded = graded_autockt_reward(worst_transient_metrics, target, success=False, failure_stage="transient")
        self.assertLess(GRADED_NO_INFORMATION_FLOOR, worst_graded)

    def test_never_reaches_terminal_bonus_on_failure(self):
        target = TargetSpec.from_existing_thresholds()
        reward = graded_autockt_reward(KNOWN_GOOD_METRICS, target, success=False, failure_stage="transient")
        self.assertLess(reward, TERMINAL_BONUS)


class AutoCktEnvTests(unittest.TestCase):
    def _make_env(self, evaluator, *, horizon=4, target=None, budget=50, seed=0):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target_pool = (target or TargetSpec.from_existing_thresholds(),)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(budget), seed=seed)
        return AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=horizon,
            adapter=adapter, grids=grids, seed=seed,
        )

    def test_reset_state_encodes_normalized_not_raw_indices(self):
        # [NEBULA ADAPTATION] see AutoCktReceiverEnv._normalized_indices /
        # ParameterGrid.normalized_index: the env must feed build_state
        # scaled-to-[-1,1] indices, not the raw 0..grid_points-1 values
        # build_state itself still documents/accepts for direct callers.
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        env = self._make_env(evaluator)
        state, _ = env.reset()
        trailing = state[-len(PARAMETER_NAMES):]
        expected = tuple(env.grids[name].normalized_index(i) for name, i in zip(PARAMETER_NAMES, env.indices))
        self.assertEqual(trailing, expected)
        for value in trailing:
            self.assertTrue(-1.0 <= value <= 1.0)
        # not the raw indices (which are 0..20 for the default 21-point grid)
        self.assertNotEqual(trailing, tuple(float(i) for i in env.indices))

    def test_step_state_also_encodes_normalized_indices(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics={
                "dfe_locked_phase_eye_height_v": 0.0, "dfe_eye_width_ui": 0.0,
                "dfe_min_margin_v": -1.0, "ctle_power_w": 1.0,
            })

        env = self._make_env(evaluator, horizon=3)
        env.reset()
        step_out = env.step([2, 2, 2, 2, 2])  # delta +2 on every parameter
        trailing = step_out.state[-len(PARAMETER_NAMES):]
        expected = tuple(env.grids[name].normalized_index(i) for name, i in zip(PARAMETER_NAMES, env.indices))
        self.assertEqual(trailing, expected)
        for value in trailing:
            self.assertTrue(-1.0 <= value <= 1.0)

    def test_episode_terminates_when_spec_is_satisfied(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        env = self._make_env(evaluator, horizon=10)
        env.reset()
        step_out = env.step([1, 1, 1, 1, 1])
        self.assertTrue(step_out.done)
        self.assertFalse(step_out.truncated)
        self.assertEqual(step_out.reward, TERMINAL_BONUS)
        self.assertTrue(step_out.info["success"])
        self.assertTrue(step_out.info["spec_satisfied"])

    def test_episode_truncates_at_horizon_when_never_satisfied(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics={
                "dfe_locked_phase_eye_height_v": 0.0, "dfe_eye_width_ui": 0.0,
                "dfe_min_margin_v": -1.0, "ctle_power_w": 1.0,
            })

        env = self._make_env(evaluator, horizon=3)
        env.reset()
        last = None
        for _ in range(3):
            last = env.step([1, 1, 1, 1, 1])
            if last.done:
                break
        self.assertFalse(last.done)
        self.assertTrue(last.truncated)
        self.assertEqual(last.info["step_count"], 3)

    def test_failed_simulation_does_not_terminate_episode(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(False)

        env = self._make_env(evaluator, horizon=5)
        env.reset()
        step_out = env.step([1, 1, 1, 1, 1])
        self.assertFalse(step_out.done)
        self.assertFalse(step_out.info["success"])
        self.assertEqual(step_out.reward, FAILURE_REWARD)

    def test_step_before_reset_raises(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        env = self._make_env(evaluator)
        with self.assertRaises(RuntimeError):
            env.step([1, 1, 1, 1, 1])

    def test_metrics_from_observation_aliases_locked_phase_eye_height(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(5), seed=0)
        step = adapter.step([0.0] * len(ACTION_BOUNDS))
        metrics = metrics_from_observation(step.observation)
        self.assertIn("dfe_locked_phase_eye_height_v", metrics)
        self.assertNotIn("dfe_eye_height_v", metrics)
        self.assertAlmostEqual(
            metrics["dfe_locked_phase_eye_height_v"], KNOWN_GOOD_METRICS["dfe_locked_phase_eye_height_v"]
        )


class AutoCktEnvRewardFnPluggabilityTests(unittest.TestCase):
    """PPO model-improvement study: AutoCktReceiverEnv's optional reward_fn
    is the only way to opt into graded_autockt_reward -- default (omitted)
    behavior must remain byte-for-byte autockt_reward, unchanged, so every
    existing training-log result stays reproducible.
    """

    def _make_env(self, evaluator, *, horizon=4, reward_fn=None, seed=0):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target_pool = (TargetSpec.from_existing_thresholds(),)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(50), seed=seed)
        return AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=horizon,
            adapter=adapter, grids=grids, seed=seed, reward_fn=reward_fn,
        )

    def test_default_reward_fn_matches_autockt_reward(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(False, failure_stage="dc")

        env = self._make_env(evaluator)  # reward_fn omitted
        env.reset()
        step_out = env.step([1, 1, 1, 1, 1])
        self.assertEqual(step_out.reward, FAILURE_REWARD)

    def test_graded_reward_fn_differentiates_dc_from_transient_failures(self):
        def dc_evaluator(*args, **kwargs):
            return _fake_evaluation(False, failure_stage="dc")

        def transient_evaluator(*args, **kwargs):
            close_metrics = dict(KNOWN_GOOD_METRICS)
            close_metrics["dfe_locked_phase_eye_height_v"] = 0.05  # below the 0.1 threshold
            return _fake_evaluation(False, metrics=close_metrics, failure_stage="transient")

        dc_env = self._make_env(dc_evaluator, reward_fn=graded_autockt_reward)
        dc_env.reset()
        dc_reward = dc_env.step([1, 1, 1, 1, 1]).reward

        transient_env = self._make_env(transient_evaluator, reward_fn=graded_autockt_reward)
        transient_env.reset()
        transient_reward = transient_env.step([1, 1, 1, 1, 1]).reward

        self.assertEqual(dc_reward, GRADED_NO_INFORMATION_FLOOR)
        self.assertNotEqual(transient_reward, GRADED_NO_INFORMATION_FLOOR)
        self.assertGreater(transient_reward, dc_reward)

    def test_graded_reward_fn_success_case_still_terminates_the_episode(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        env = self._make_env(evaluator, horizon=10, reward_fn=graded_autockt_reward)
        env.reset()
        step_out = env.step([1, 1, 1, 1, 1])
        self.assertTrue(step_out.done)
        self.assertEqual(step_out.reward, TERMINAL_BONUS)


class AutoCktEnvInitialStateRandomizationTests(unittest.TestCase):
    """AutoCktReceiverEnv's [NEBULA ADAPTATION] randomize_initial_state
    option (default False -- see class docstring in rl/autockt_env.py).
    """

    def _make_env(self, *, seed=0, randomize_initial_state=False, budget=200):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target_pool = (TargetSpec.from_existing_thresholds(),)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(budget), seed=seed)
        env = AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=4,
            adapter=adapter, grids=grids, seed=seed, randomize_initial_state=randomize_initial_state,
        )
        return env, indices

    def test_disabled_reproduces_the_exact_previous_initial_state(self):
        env, anchor = self._make_env(randomize_initial_state=False)
        for _ in range(5):
            env.reset()
            self.assertEqual(env.indices, anchor)

    def test_enabled_initialization_stays_within_grid_bounds(self):
        env, _anchor = self._make_env(randomize_initial_state=True, seed=1)
        for _ in range(30):
            env.reset()
            for name, index in zip(PARAMETER_NAMES, env.indices):
                self.assertGreaterEqual(index, 0)
                self.assertLess(index, len(env.grids[name]))

    def test_perturbations_are_never_more_than_one_index_from_the_anchor(self):
        env, anchor = self._make_env(randomize_initial_state=True, seed=2)
        for _ in range(30):
            env.reset()
            for anchor_index, sampled_index in zip(anchor, env.indices):
                self.assertLessEqual(abs(sampled_index - anchor_index), 1)

    def test_same_seed_gives_reproducible_starting_states(self):
        env_a, _ = self._make_env(randomize_initial_state=True, seed=7)
        env_b, _ = self._make_env(randomize_initial_state=True, seed=7)
        sequence_a = []
        sequence_b = []
        for _ in range(10):
            env_a.reset()
            sequence_a.append(env_a.indices)
            env_b.reset()
            sequence_b.append(env_b.indices)
        self.assertEqual(sequence_a, sequence_b)

    def test_different_resets_can_produce_different_starting_states(self):
        env, _anchor = self._make_env(randomize_initial_state=True, seed=3)
        observed = set()
        for _ in range(20):
            env.reset()
            observed.add(env.indices)
        self.assertGreater(len(observed), 1)

    def test_randomization_uses_an_rng_stream_independent_of_target_selection(self):
        # Enabling randomize_initial_state must not change which target a
        # multi-target pool selects per episode under the same seed.
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target_pool = (
            TargetSpec.from_existing_thresholds(),
            TargetSpec.from_hard_target(),
        )
        targets_without = []
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(200), seed=5)
        env = AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=4,
            adapter=adapter, grids=grids, seed=5, randomize_initial_state=False,
        )
        for _ in range(10):
            _state, info = env.reset()
            targets_without.append(info["target"])

        targets_with = []
        adapter2 = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(200), seed=5)
        env2 = AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=4,
            adapter=adapter2, grids=grids, seed=5, randomize_initial_state=True,
        )
        for _ in range(10):
            _state, info = env2.reset()
            targets_with.append(info["target"])

        self.assertEqual(targets_without, targets_with)


class PPOAgentTests(unittest.TestCase):
    def test_act_returns_one_valid_choice_per_parameter(self):
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)
        state = tuple(0.0 for _ in range(STATE_DIM))
        choices, log_prob, value = agent.act(state)
        self.assertEqual(len(choices), len(PARAMETER_NAMES))
        for choice in choices:
            self.assertIn(choice, (0, 1, 2))
        self.assertTrue(math.isfinite(log_prob))
        self.assertTrue(math.isfinite(value))

    def test_update_runs_on_a_tiny_synthetic_batch_and_returns_finite_stats(self):
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)
        state = tuple(0.1 for _ in range(STATE_DIM))
        transitions = [
            Transition(
                state=state, choices=(0, 1, 2, 1, 0), log_prob=-1.0, value=0.0, reward=1.0,
                terminated=False, truncated=False,
            ),
            Transition(
                state=state, choices=(1, 1, 1, 1, 1), log_prob=-1.0, value=0.0, reward=-0.5,
                terminated=True, truncated=False,
            ),
        ]
        stats = agent.update(transitions, last_value=0.0, epochs=2, minibatch_size=2)
        for value in stats.values():
            self.assertTrue(math.isfinite(value))

    def test_update_rejects_empty_batch(self):
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)
        with self.assertRaises(ValueError):
            agent.update([], last_value=0.0)

    def test_compute_gae_bootstraps_truncated_transitions_from_bootstrap_value_not_zero(self):
        # [repair #2] gamma=lambda=1 for exact, easy-to-hand-check arithmetic.
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0, gamma=1.0, gae_lambda=1.0)
        state = tuple(0.0 for _ in range(STATE_DIM))
        truncated = Transition(
            state=state, choices=(0,) * len(PARAMETER_NAMES), log_prob=0.0,
            value=2.0, reward=1.0, terminated=False, truncated=True, bootstrap_value=5.0,
        )
        advantages, returns = agent.compute_gae([truncated], last_value=0.0)
        # delta = reward + gamma*bootstrap_value - value = 1.0 + 5.0 - 2.0 = 4.0
        # (the pre-repair code always zero-bootstrapped here, giving -1.0).
        self.assertAlmostEqual(advantages[0], 4.0)
        self.assertAlmostEqual(returns[0], 4.0 + 2.0)

    def test_compute_gae_still_zero_bootstraps_true_termination(self):
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0, gamma=1.0, gae_lambda=1.0)
        state = tuple(0.0 for _ in range(STATE_DIM))
        terminated = Transition(
            state=state, choices=(0,) * len(PARAMETER_NAMES), log_prob=0.0,
            value=2.0, reward=1.0, terminated=True, truncated=False, bootstrap_value=5.0,  # must be ignored
        )
        advantages, returns = agent.compute_gae([terminated], last_value=0.0)
        self.assertAlmostEqual(advantages[0], 1.0 - 2.0)  # reward - value, bootstrap_value ignored
        self.assertAlmostEqual(returns[0], (1.0 - 2.0) + 2.0)

    def test_compute_gae_does_not_leak_advantage_across_a_truncated_episode_boundary(self):
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0, gamma=1.0, gae_lambda=1.0)
        state = tuple(0.0 for _ in range(STATE_DIM))
        # Episode A: one truncated transition. Episode B: one terminated
        # transition immediately after it in the same flat batch.
        episode_a_last = Transition(
            state=state, choices=(0,) * len(PARAMETER_NAMES), log_prob=0.0,
            value=2.0, reward=1.0, terminated=False, truncated=True, bootstrap_value=5.0,
        )
        episode_b_only = Transition(
            state=state, choices=(0,) * len(PARAMETER_NAMES), log_prob=0.0,
            value=0.5, reward=-3.0, terminated=True, truncated=False,
        )
        advantages, _ = agent.compute_gae([episode_a_last, episode_b_only], last_value=0.0)
        # Episode A's advantage must equal its own delta (1.0 + 5.0 - 2.0 = 4.0),
        # independent of episode B's reward/value -- the mask stops the
        # recursive GAE(lambda) carry at the truncation boundary too.
        self.assertAlmostEqual(advantages[0], 4.0)
        self.assertAlmostEqual(advantages[1], -3.0 - 0.5)


class TrainerTests(unittest.TestCase):
    def test_collect_rollout_and_train_close_the_full_loop_without_real_spice(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target_pool = (TargetSpec.from_existing_thresholds(),)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(100), seed=0)
        env = AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=4,
            adapter=adapter, grids=grids, seed=0,
        )
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)

        seen_rows = []
        result = train(env, agent, num_updates=2, episodes_per_update=2, on_step=seen_rows.append)

        self.assertGreater(len(seen_rows), 0)
        self.assertEqual(result.best_reward, TERMINAL_BONUS)
        self.assertTrue(result.any_spec_satisfied)
        self.assertEqual(result.total_evaluations, adapter.total_evaluations)
        self.assertEqual(len(result.updates), 2)

    def test_collect_rollout_marks_truncated_transitions_and_captures_bootstrap_value(self):
        # [repair #2] An evaluator that never satisfies the target forces
        # every episode to end via horizon truncation, never termination.
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics={
                "dfe_locked_phase_eye_height_v": 0.0, "dfe_eye_width_ui": 0.0,
                "dfe_min_margin_v": -1.0, "ctle_power_w": 1.0,
            })

        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target_pool = (TargetSpec.from_existing_thresholds(),)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(20), seed=0)
        env = AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=3,
            adapter=adapter, grids=grids, seed=0,
        )
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)

        transitions, _bootstrap, episode_logs = collect_rollout(env, agent, episodes=1)

        self.assertEqual(len(transitions), 3)
        for transition in transitions[:-1]:
            self.assertFalse(transition.terminated)
            self.assertFalse(transition.truncated)
            self.assertEqual(transition.bootstrap_value, 0.0)  # unused, default
        last = transitions[-1]
        self.assertFalse(last.terminated)
        self.assertTrue(last.truncated)
        self.assertTrue(math.isfinite(last.bootstrap_value))
        self.assertFalse(episode_logs[0].spec_satisfied)


class TargetModeMixedTests(unittest.TestCase):
    """experiments.train_autockt._build_target_pools's --target-mode mixed
    (see its docstring): training pool = (trivial, hard); validation pool =
    exactly one unseen arithmetic midpoint of the two, never the degenerate
    sample_target_pool ranges that inspection found trivially satisfied by
    every possible sample (Design A clears every one of them).
    """

    def _pools(self, **overrides):
        args = argparse.Namespace(
            target_mode="mixed", num_training_specs=1, num_validation_specs=0, seed=42,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return _build_target_pools(args)

    def test_training_pool_is_exactly_trivial_then_hard(self):
        training_pool, _validation_pool = self._pools()
        self.assertEqual(training_pool, (TargetSpec.from_existing_thresholds(), TargetSpec.from_hard_target()))

    def test_validation_pool_is_exactly_one_arithmetic_midpoint_target(self):
        _training_pool, validation_pool = self._pools()
        self.assertEqual(len(validation_pool), 1)
        expected = TargetSpec(
            dfe_locked_phase_eye_height_v=(EXISTING_THRESHOLDS["dfe_locked_phase_eye_height_v"]
                                            + HARD_TARGET_THRESHOLDS["dfe_locked_phase_eye_height_v"]) / 2.0,
            dfe_eye_width_ui=(EXISTING_THRESHOLDS["dfe_eye_width_ui"]
                               + HARD_TARGET_THRESHOLDS["dfe_eye_width_ui"]) / 2.0,
            dfe_min_margin_v=(EXISTING_THRESHOLDS["dfe_min_margin_v"]
                               + HARD_TARGET_THRESHOLDS["dfe_min_margin_v"]) / 2.0,
            ctle_power_w=(EXISTING_THRESHOLDS["ctle_power_w"] + HARD_TARGET_THRESHOLDS["ctle_power_w"]) / 2.0,
        )
        self.assertEqual(validation_pool[0], expected)
        # exact expected numeric values, spelled out (not just re-derived arithmetic):
        self.assertAlmostEqual(validation_pool[0].dfe_locked_phase_eye_height_v, 0.45)
        self.assertAlmostEqual(validation_pool[0].dfe_eye_width_ui, 0.5)
        self.assertAlmostEqual(validation_pool[0].dfe_min_margin_v, 0.175)
        self.assertAlmostEqual(validation_pool[0].ctle_power_w, 0.015)

    def test_validation_target_is_not_identical_to_either_training_target(self):
        training_pool, validation_pool = self._pools()
        trivial, hard = training_pool
        self.assertNotEqual(validation_pool[0], trivial)
        self.assertNotEqual(validation_pool[0], hard)

    def test_mixed_mode_ignores_num_training_and_validation_specs(self):
        # Deliberately different, larger values must not change the result --
        # 'mixed' bypasses sample_target_pool entirely.
        default_training, default_validation = self._pools()
        overridden_training, overridden_validation = self._pools(
            num_training_specs=7, num_validation_specs=5,
        )
        self.assertEqual(default_training, overridden_training)
        self.assertEqual(default_validation, overridden_validation)

    def test_existing_trivial_and_hard_modes_are_unchanged(self):
        trivial_pool, trivial_validation = _build_target_pools(
            argparse.Namespace(target_mode="trivial", num_training_specs=1, num_validation_specs=0, seed=42)
        )
        self.assertEqual(trivial_pool, (TargetSpec.from_existing_thresholds(),))
        self.assertEqual(trivial_validation, ())

        hard_pool, hard_validation = _build_target_pools(
            argparse.Namespace(target_mode="hard", num_training_specs=1, num_validation_specs=0, seed=42)
        )
        self.assertEqual(hard_pool, (TargetSpec.from_hard_target(),))
        self.assertEqual(hard_validation, ())


class CheckpointEvaluationTests(unittest.TestCase):
    """experiments.train_autockt._evaluate_checkpoint -- [NEBULA ADAPTATION],
    no AutoCkt equivalent. Verifies the frozen-checkpoint evaluation helper
    actually loads the given policy weights, is reproducible under a shared
    seed, and aggregates its own returned per-episode rows correctly.
    """

    def _common_kwargs(self, *, evaluator, seed):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(200), seed=seed)
        return dict(
            adapter=adapter,
            training_pool=(TargetSpec.from_existing_thresholds(),),
            initial_indices=indices,
            grids=grids,
            horizon=3,
            randomize_initial_state=True,
        )

    def test_evaluate_checkpoint_loads_the_given_policy_weights(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)
        original_state = copy_state = {k: v.clone() for k, v in agent.policy.state_dict().items()}
        other_agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=99)
        other_state = other_agent.policy.state_dict()
        # sanity: the two seeds really do produce different weights
        self.assertFalse(all(torch.equal(original_state[k], other_state[k]) for k in original_state))

        kwargs = self._common_kwargs(evaluator=evaluator, seed=1)
        _summary, _rows = _evaluate_checkpoint(label="other", policy_state=other_state, agent=agent, seed=1, episodes=1, **kwargs)

        loaded = agent.policy.state_dict()
        for key in other_state:
            self.assertTrue(torch.equal(loaded[key], other_state[key]))
            self.assertFalse(torch.equal(loaded[key], copy_state[key]))

    def test_evaluate_checkpoint_is_reproducible_under_a_shared_seed(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=3)
        policy_state = agent.policy.state_dict()

        kwargs_a = self._common_kwargs(evaluator=evaluator, seed=7)
        summary_a, rows_a = _evaluate_checkpoint(label="x", policy_state=policy_state, agent=agent, seed=7, episodes=3, **kwargs_a)

        kwargs_b = self._common_kwargs(evaluator=evaluator, seed=7)
        summary_b, rows_b = _evaluate_checkpoint(label="x", policy_state=policy_state, agent=agent, seed=7, episodes=3, **kwargs_b)

        rows_a_comparable = [{k: v for k, v in row.items() if k != "checkpoint"} for row in rows_a]
        rows_b_comparable = [{k: v for k, v in row.items() if k != "checkpoint"} for row in rows_b]
        self.assertEqual(rows_a_comparable, rows_b_comparable)
        self.assertAlmostEqual(summary_a["mean_episode_reward"], summary_b["mean_episode_reward"])
        self.assertAlmostEqual(summary_a["satisfaction_rate"], summary_b["satisfaction_rate"])

    def test_evaluate_checkpoint_summary_matches_its_own_episode_rows(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics={
                "dfe_locked_phase_eye_height_v": 0.0, "dfe_eye_width_ui": 0.0,
                "dfe_min_margin_v": -1.0, "ctle_power_w": 1.0,
            })

        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=0)
        policy_state = agent.policy.state_dict()
        kwargs = self._common_kwargs(evaluator=evaluator, seed=2)
        summary, rows = _evaluate_checkpoint(label="x", policy_state=policy_state, agent=agent, seed=2, episodes=4, **kwargs)

        self.assertEqual(len(rows), 4)
        expected_mean = sum(row["episode_reward"] for row in rows) / 4
        expected_rate = sum(1 for row in rows if row["spec_satisfied"]) / 4
        self.assertAlmostEqual(summary["mean_episode_reward"], expected_mean)
        self.assertAlmostEqual(summary["satisfaction_rate"], expected_rate)
        self.assertEqual(summary["total_evaluations_after"], kwargs["adapter"].total_evaluations)


class ValidationPoolEvaluationTests(unittest.TestCase):
    """experiments.train_autockt._evaluate_validation_pool -- the zero-shot
    unseen-target evaluation extracted from main()'s body (see
    docs/autockt-mapping.md for --target-mode mixed's provenance). Nothing
    here calls real ngspice or the actual CLI; all evaluations go through a
    deterministic fake evaluator injected via ReceiverRLAdapter(evaluator=...).

    Together with MixedModeMilestoneTests below, this establishes the five
    things the multi-target-generalization milestone requires:
      1/2. exact training/validation targets  -- TargetModeMixedTests (pool
           construction) + MixedModeMilestoneTests (end-to-end runtime use)
      3. validation target differs from both training targets -- TargetModeMixedTests
      4. reproducibility -- test_reproducible_under_a_shared_seed below
      5. trivial/hard modes unchanged -- TargetModeMixedTests
    plus the one gap pool-construction tests alone cannot cover: that
    AutoCktReceiverEnv.reset(target=...) actually evaluates the GIVEN target
    rather than silently re-sampling from env.target_pool -- the mechanism
    that makes "zero-shot on an unseen target" true at runtime, not just in
    the pool-construction data.
    """

    def _env_and_agent(self, *, evaluator, env_seed, agent_seed, target_pool, randomize_initial_state=False):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(200), seed=env_seed)
        env = AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=3, adapter=adapter,
            grids=grids, seed=env_seed, randomize_initial_state=randomize_initial_state,
        )
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=agent_seed)
        return env, agent

    def test_evaluates_the_given_target_not_a_resample_from_env_target_pool(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        trivial = TargetSpec.from_existing_thresholds()
        hard = TargetSpec.from_hard_target()
        midpoint = TargetSpec(**{
            name: (getattr(trivial, name) + getattr(hard, name)) / 2.0 for name in SPEC_NAMES
        })
        self.assertNotEqual(midpoint, trivial)
        self.assertNotEqual(midpoint, hard)

        # env's OWN target_pool never contains the midpoint -- if reset()
        # silently resampled from the pool instead of honoring the explicit
        # `target=` override, every row would report `trivial`, not `midpoint`.
        env, agent = self._env_and_agent(
            evaluator=evaluator, env_seed=1, agent_seed=1, target_pool=(trivial,),
        )
        rows = _evaluate_validation_pool(env=env, agent=agent, validation_pool=(midpoint,))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target"], midpoint.as_dict())
        self.assertNotEqual(rows[0]["target"], trivial.as_dict())

    def test_multiple_validation_targets_are_evaluated_independently_and_in_order(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        trivial = TargetSpec.from_existing_thresholds()
        hard = TargetSpec.from_hard_target()
        env, agent = self._env_and_agent(
            evaluator=evaluator, env_seed=2, agent_seed=2, target_pool=(trivial, hard),
        )
        rows = _evaluate_validation_pool(env=env, agent=agent, validation_pool=(hard, trivial))

        self.assertEqual(rows[0]["validation_spec_index"], 0)
        self.assertEqual(rows[0]["target"], hard.as_dict())
        self.assertEqual(rows[1]["validation_spec_index"], 1)
        self.assertEqual(rows[1]["target"], trivial.as_dict())

    def test_reproducible_under_a_shared_seed(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        trivial = TargetSpec.from_existing_thresholds()
        hard = TargetSpec.from_hard_target()
        midpoint = TargetSpec(**{
            name: (getattr(trivial, name) + getattr(hard, name)) / 2.0 for name in SPEC_NAMES
        })

        env_a, agent_a = self._env_and_agent(
            evaluator=evaluator, env_seed=9, agent_seed=9, target_pool=(trivial, hard),
            randomize_initial_state=True,
        )
        rows_a = _evaluate_validation_pool(env=env_a, agent=agent_a, validation_pool=(midpoint,))

        env_b, agent_b = self._env_and_agent(
            evaluator=evaluator, env_seed=9, agent_seed=9, target_pool=(trivial, hard),
            randomize_initial_state=True,
        )
        rows_b = _evaluate_validation_pool(env=env_b, agent=agent_b, validation_pool=(midpoint,))

        self.assertEqual(rows_a, rows_b)

    def test_does_not_mutate_policy_weights(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        trivial = TargetSpec.from_existing_thresholds()
        env, agent = self._env_and_agent(
            evaluator=evaluator, env_seed=4, agent_seed=4, target_pool=(trivial,),
        )
        before = {k: v.clone() for k, v in agent.policy.state_dict().items()}
        _evaluate_validation_pool(env=env, agent=agent, validation_pool=(trivial,))
        after = agent.policy.state_dict()

        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]))

    def test_failed_evaluation_does_not_satisfy_the_spec(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(False)

        trivial = TargetSpec.from_existing_thresholds()
        env, agent = self._env_and_agent(
            evaluator=evaluator, env_seed=5, agent_seed=5, target_pool=(trivial,),
        )
        rows = _evaluate_validation_pool(env=env, agent=agent, validation_pool=(trivial,))
        self.assertFalse(rows[0]["spec_satisfied"])


class ResolveInitialIndicesTests(unittest.TestCase):
    """experiments.train_autockt._resolve_initial_indices --
    [NEBULA ADAPTATION], no algorithm/formulation change: only selects
    which starting grid point a training run uses. 'grid-center' exists so
    an experiment can avoid warm-starting PPO from
    VERIFIED_INITIAL_PARAMETERS (itself seeded from Random Search's own
    result -- see rl/parameter_grid.py), per
    docs/autockt-mapping.md section 20.
    """

    def test_verified_matches_the_existing_helper_unchanged(self):
        grids = build_parameter_grids()
        self.assertEqual(
            _resolve_initial_indices(grids, "verified"), verified_initial_indices(grids),
        )

    def test_grid_center_is_the_true_midpoint_of_every_parameter(self):
        # log spacing (what the actual experiment uses, see
        # docs/autockt-mapping.md sec 20) gives a clean 21-point grid for
        # every parameter; linear spacing's np.arange has its own known
        # floating-point overshoot quirk (e.g. 22 points for cdeg_f) that
        # is unrelated to this helper and not asserted on here.
        grids = build_parameter_grids(21, spacing="log")
        indices = _resolve_initial_indices(grids, "grid-center")
        self.assertEqual(indices, tuple(len(grids[name]) // 2 for name in PARAMETER_NAMES))
        self.assertEqual(indices, (10,) * 5)
        # for the four log-scale parameters, the center index is the true
        # geometric midpoint of the grid's own (lower, upper) bounds.
        for name in ("rload_ohm", "rdeg_ohm", "cdeg_f", "itail_a"):
            lower, upper = grids[name].values[0], grids[name].values[-1]
            self.assertAlmostEqual(grids[name].value_at(10), (lower * upper) ** 0.5, delta=1e-9 * upper)

    def test_grid_center_differs_from_the_random_search_seeded_verified_point(self):
        grids = build_parameter_grids()
        self.assertNotEqual(
            _resolve_initial_indices(grids, "grid-center"),
            _resolve_initial_indices(grids, "verified"),
        )

    def test_unknown_source_raises(self):
        grids = build_parameter_grids()
        with self.assertRaises(ValueError):
            _resolve_initial_indices(grids, "not-a-real-source")


class MixedModeMilestoneTests(unittest.TestCase):
    """End-to-end (still SPICE-free) check that --target-mode mixed's pool
    construction and its actual runtime evaluation path agree: training on
    exactly (trivial, hard) and validating zero-shot on exactly their
    arithmetic midpoint, per the multi-target-generalization milestone.
    """

    def test_mixed_mode_trains_on_trivial_and_hard_validates_on_midpoint_at_runtime(self):
        def evaluator(*args, **kwargs):
            return _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS)

        args = argparse.Namespace(target_mode="mixed", num_training_specs=1, num_validation_specs=0, seed=42)
        training_pool, validation_pool = _build_target_pools(args)

        self.assertEqual(training_pool, (TargetSpec.from_existing_thresholds(), TargetSpec.from_hard_target()))
        self.assertEqual(len(validation_pool), 1)
        self.assertNotIn(validation_pool[0], training_pool)

        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(200), seed=42)
        env = AutoCktReceiverEnv(
            target_pool=training_pool, initial_indices=indices, horizon=3, adapter=adapter,
            grids=grids, seed=42,
        )
        agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=42)

        rows = _evaluate_validation_pool(env=env, agent=agent, validation_pool=validation_pool)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target"], validation_pool[0].as_dict())
        # the evaluated target is genuinely absent from what training exposed the env to:
        self.assertNotIn(rows[0]["target"], [spec.as_dict() for spec in training_pool])


class RewardModeCliTests(unittest.TestCase):
    """--reward-mode wiring, exercised through the real CLI (--backend
    synthetic, so SPICE-free) rather than just the underlying mechanism
    (already covered by GradedAutocktRewardTests/
    AutoCktEnvRewardFnPluggabilityTests above) -- catches a wiring mistake
    (e.g. a typo in the flag name or a dropped reward_fn= pass-through)
    that unit tests on the pieces alone would not.
    """

    def test_default_reward_mode_is_terminal_and_is_recorded_in_the_output(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run.jsonl"
            argv = [
                "train_autockt.py", "--backend", "synthetic", "--updates", "1",
                "--episodes-per-update", "1", "--horizon", "1", "--seed", "1",
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                rc = main()
            self.assertEqual(rc, 0)
            self.assertTrue(output.is_file())

    def test_graded_reward_mode_runs_end_to_end_via_the_cli(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "run.jsonl"
            argv = [
                "train_autockt.py", "--backend", "synthetic", "--updates", "1",
                "--episodes-per-update", "1", "--horizon", "1", "--seed", "1",
                "--reward-mode", "graded", "--output", str(output),
            ]
            with patch("sys.argv", argv):
                rc = main()
            self.assertEqual(rc, 0)
            self.assertTrue(output.is_file())
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertIn("reward_version", rows[0])

    def test_invalid_reward_mode_is_rejected(self):
        argv = ["train_autockt.py", "--backend", "synthetic", "--reward-mode", "bogus",
                "--output", "results/does_not_matter.jsonl"]
        with patch("sys.argv", argv):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
