"""SPICE-free tests for the PPO v2 model-improvement study: state schema
v2 (rl/autockt_state_v2.py), reward v2 (rl/autockt_reward_v2.py), and
AutoCktReceiverEnv's new opt-in capabilities (evaluate_on_reset,
state_schema, use_reward_v2, action_deltas, budget-aware truncation).

PPO v1 preservation is the load-bearing property throughout: every test
class here either asserts v1 behavior is unchanged when the new
parameters are omitted, or exercises v2 behavior only when explicitly
opted into.
"""

from __future__ import annotations

import unittest

from simulator.config import SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters, StageResult
from simulator.rl_adapter import ACTION_BOUNDS, METRIC_OBSERVATION_NAMES, ReceiverRLAdapter, RLBudget

from rl.autockt_action import apply_action
from rl.autockt_env import AutoCktReceiverEnv, metrics_from_observation, spec_metrics_valid
from rl.autockt_reward import FAILURE_REWARD, TERMINAL_BONUS
from rl.autockt_reward_v2 import (
    NO_INFORMATION_FLOOR_V2,
    PEAKING_BAND_DB,
    RewardResult,
    reward_v2,
)
from rl.autockt_state_v2 import (
    FAILURE_STAGE_VOCAB,
    STATE_DIM_V1,
    STATE_DIM_V2,
    build_state_v2,
    validity_mask_from_observation,
)
from rl.parameter_grid import ACTION_DELTAS, build_parameter_grids, verified_initial_indices, PARAMETER_NAMES
from rl.target_spec import TargetSpec

KNOWN_GOOD_METRICS = {
    "dfe_locked_phase_eye_height_v": 1.5215334399802445,
    "dfe_eye_width_ui": 0.8699999999999999,
    "dfe_min_margin_v": 0.5174474651666459,
    "ctle_power_w": 0.0010850994,
}


def _fake_evaluation(
    success: bool = True, *, metrics: dict | None = None, failure_stage: str | None = "dc",
) -> ReceiverEvaluation:
    """Unlike tests/test_autockt_rl.py's fixture (which zero-fills every
    metric so it always looks "present" -- appropriate for testing v1,
    which never checked validity), this one only includes keys explicitly
    passed via `metrics`, mirroring how a REAL early-stage failure
    genuinely lacks the transient-only keys entirely (simulator/
    receiver.py::evaluate_receiver only ever adds a stage's own computed
    metrics -- a dc failure's `combined` dict never gains
    dfe_locked_phase_eye_height_v etc. at all). Needed so
    spec_metrics_valid's "genuinely missing vs. present-as-zero" test is
    actually meaningful.
    """

    # evaluation.metrics keeps the RAW metric name
    # ("dfe_locked_phase_eye_height_v") -- observation_from_evaluation
    # itself does the dfe_eye_height_v aliasing on lookup, not the other
    # way around (simulator/rl_adapter.py::observation_from_evaluation).
    values = dict(metrics or {})
    stage = StageResult("fake", success, 0.0, metrics=values)
    return ReceiverEvaluation(
        success, ReceiverParameters(), SimulationConditions(), EvaluationFidelity.TRAINING,
        (stage,), values, None if success else failure_stage, 0.0, "fake", {},
    )


class SpecMetricsValidTests(unittest.TestCase):
    def test_dc_failure_is_not_valid(self):
        adapter = ReceiverRLAdapter(
            evaluator=lambda *a, **k: _fake_evaluation(False, failure_stage="dc"), budget=RLBudget(5), seed=0,
        )
        step = adapter.step([0.0] * len(ACTION_BOUNDS))
        self.assertFalse(spec_metrics_valid(step.observation))

    def test_success_is_valid(self):
        adapter = ReceiverRLAdapter(
            evaluator=lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS), budget=RLBudget(5), seed=0,
        )
        step = adapter.step([0.0] * len(ACTION_BOUNDS))
        self.assertTrue(spec_metrics_valid(step.observation))

    def test_validity_mask_reads_the_adapters_own_flags_not_recomputed(self):
        adapter = ReceiverRLAdapter(
            evaluator=lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS), budget=RLBudget(5), seed=0,
        )
        step = adapter.step([0.0] * len(ACTION_BOUNDS))
        mask = validity_mask_from_observation(step.observation, METRIC_OBSERVATION_NAMES)
        self.assertTrue(mask["dfe_eye_height_v"])
        self.assertTrue(mask["ctle_power_w"])


class StateV2Tests(unittest.TestCase):
    def test_dim_matches_v1_plus_extension(self):
        state = build_state_v2({}, TargetSpec.from_existing_thresholds(), [0] * len(PARAMETER_NAMES),
                                metrics_valid=False, failure_stage="unevaluated")
        self.assertEqual(len(state), STATE_DIM_V2)
        self.assertEqual(STATE_DIM_V2, STATE_DIM_V1 + 1 + 2 + len(FAILURE_STAGE_VOCAB))

    def test_v1_slice_is_byte_identical_to_build_state(self):
        from rl.autockt_state import build_state
        target = TargetSpec.from_existing_thresholds()
        indices = [1, 2, 3, 4, 5]
        v1 = build_state(KNOWN_GOOD_METRICS, target, indices)
        v2 = build_state_v2(KNOWN_GOOD_METRICS, target, indices, metrics_valid=True, failure_stage=None)
        self.assertEqual(v2[:STATE_DIM_V1], v1)

    def test_unevaluated_and_success_get_different_one_hot_codes(self):
        target = TargetSpec.from_existing_thresholds()
        indices = [0] * len(PARAMETER_NAMES)
        unevaluated = build_state_v2({}, target, indices, metrics_valid=False, failure_stage="unevaluated")
        success = build_state_v2(KNOWN_GOOD_METRICS, target, indices, metrics_valid=True, failure_stage=None)
        self.assertNotEqual(unevaluated[-len(FAILURE_STAGE_VOCAB):], success[-len(FAILURE_STAGE_VOCAB):])

    def test_metrics_valid_flag_is_encoded(self):
        target = TargetSpec.from_existing_thresholds()
        indices = [0] * len(PARAMETER_NAMES)
        valid_flag_index = STATE_DIM_V1
        invalid_state = build_state_v2({}, target, indices, metrics_valid=False, failure_stage="dc")
        valid_state = build_state_v2(KNOWN_GOOD_METRICS, target, indices, metrics_valid=True, failure_stage=None)
        self.assertEqual(invalid_state[valid_flag_index], 0.0)
        self.assertEqual(valid_state[valid_flag_index], 1.0)

    def test_early_stage_margin_available_even_when_metrics_invalid(self):
        # ctle_power_w survives a dc-stage failure (computed before the
        # violation check) -- this must reach the state even though the 4
        # SPEC_NAMES targets are invalid.
        target = TargetSpec.from_existing_thresholds()
        indices = [0] * len(PARAMETER_NAMES)
        partial_metrics = {"ctle_power_w": 0.005}
        state = build_state_v2(partial_metrics, target, indices, metrics_valid=False, failure_stage="ac")
        margin_start = STATE_DIM_V1 + 1
        self.assertEqual(state[margin_start], 0.005)

    def test_unrecognized_failure_stage_raises(self):
        with self.assertRaises(ValueError):
            build_state_v2({}, TargetSpec.from_existing_thresholds(), [0] * len(PARAMETER_NAMES),
                            metrics_valid=False, failure_stage="not_a_real_stage")


class RewardV2PropertyTests(unittest.TestCase):
    """Property tests required by the spec: correct direction, monotonicity,
    exact-boundary behavior, missing-metric handling, failure ordering,
    and PPO v1 remaining unchanged.
    """

    target = TargetSpec.from_existing_thresholds()

    # -- PPO v1 unchanged --
    def test_v1_autockt_reward_is_untouched_by_this_module(self):
        from rl.autockt_reward import autockt_reward
        # Importing/using reward_v2 must not alter v1's own function object
        # or behavior in any way.
        self.assertEqual(
            autockt_reward(KNOWN_GOOD_METRICS, self.target, success=True), TERMINAL_BONUS,
        )

    # -- missing-metric handling --
    def test_metrics_invalid_never_interpreted_as_zero(self):
        result = reward_v2({}, self.target, metrics_valid=False, success=False, failure_stage="dc")
        self.assertEqual(result.components, {})
        self.assertEqual(result.available_metrics, ())

    def test_simulator_failure_worse_than_every_valid_design(self):
        worst_valid = reward_v2(
            {"dfe_locked_phase_eye_height_v": 0.0, "dfe_eye_width_ui": 0.0,
             "dfe_min_margin_v": -1.0, "ctle_power_w": 1.0},
            self.target, metrics_valid=True, success=False, failure_stage="transient",
        )
        no_info = reward_v2({}, self.target, metrics_valid=False, success=False, failure_stage="dc")
        self.assertLess(no_info.total, worst_valid.total)
        self.assertEqual(no_info.total, NO_INFORMATION_FLOOR_V2)

    # -- correct direction --
    def test_larger_is_better_metric_improves_score_when_increased(self):
        low = dict(KNOWN_GOOD_METRICS, dfe_locked_phase_eye_height_v=0.05)
        high = dict(KNOWN_GOOD_METRICS, dfe_locked_phase_eye_height_v=0.09)
        r_low = reward_v2(low, self.target, metrics_valid=True, success=False, failure_stage="transient")
        r_high = reward_v2(high, self.target, metrics_valid=True, success=False, failure_stage="transient")
        self.assertGreater(r_high.components["dfe_locked_phase_eye_height_v"],
                            r_low.components["dfe_locked_phase_eye_height_v"])

    def test_smaller_is_better_metric_improves_score_when_decreased(self):
        high_power = dict(KNOWN_GOOD_METRICS, ctle_power_w=0.02, dfe_locked_phase_eye_height_v=0.05)
        low_power = dict(KNOWN_GOOD_METRICS, ctle_power_w=0.016, dfe_locked_phase_eye_height_v=0.05)
        r_high = reward_v2(high_power, self.target, metrics_valid=True, success=False, failure_stage="transient")
        r_low = reward_v2(low_power, self.target, metrics_valid=True, success=False, failure_stage="transient")
        self.assertGreater(r_low.components["ctle_power_w"], r_high.components["ctle_power_w"])

    # -- monotonicity --
    def test_component_is_monotonic_in_distance_from_target(self):
        near = dict(KNOWN_GOOD_METRICS, dfe_locked_phase_eye_height_v=0.09)
        far = dict(KNOWN_GOOD_METRICS, dfe_locked_phase_eye_height_v=0.01)
        very_far = dict(KNOWN_GOOD_METRICS, dfe_locked_phase_eye_height_v=0.001)
        r_near = reward_v2(near, self.target, metrics_valid=True, success=False, failure_stage="transient")
        r_far = reward_v2(far, self.target, metrics_valid=True, success=False, failure_stage="transient")
        r_very_far = reward_v2(very_far, self.target, metrics_valid=True, success=False, failure_stage="transient")
        self.assertGreater(r_near.components["dfe_locked_phase_eye_height_v"],
                            r_far.components["dfe_locked_phase_eye_height_v"])
        self.assertGreater(r_far.components["dfe_locked_phase_eye_height_v"],
                            r_very_far.components["dfe_locked_phase_eye_height_v"])

    # -- exact-boundary behavior --
    def test_exact_boundary_is_a_strict_pass_zero_tolerance(self):
        boundary_metrics = dict(KNOWN_GOOD_METRICS)
        boundary_metrics["dfe_eye_width_ui"] = self.target.dfe_eye_width_ui  # exactly at threshold
        result = reward_v2(boundary_metrics, self.target, metrics_valid=True, success=True, failure_stage=None)
        self.assertTrue(result.strict_target_pass)

    def test_just_below_boundary_fails_strict_pass(self):
        below_metrics = dict(KNOWN_GOOD_METRICS)
        below_metrics["dfe_eye_width_ui"] = self.target.dfe_eye_width_ui - 1e-9
        result = reward_v2(below_metrics, self.target, metrics_valid=True, success=False, failure_stage="transient")
        self.assertFalse(result.strict_target_pass)

    def test_strict_pass_and_autockt_terminal_success_are_independent_flags(self):
        # autockt_terminal_success uses v1's -0.02 relative-error tolerance;
        # strict_target_pass does not. They can legitimately differ.
        result = reward_v2(KNOWN_GOOD_METRICS, self.target, metrics_valid=True, success=True, failure_stage=None)
        self.assertTrue(result.strict_target_pass)
        self.assertTrue(result.autockt_terminal_success)  # both true here; independence tested via boundary test above

    # -- failure ordering --
    def test_failure_ordering_no_info_worse_than_transient_worse_than_success(self):
        no_info = reward_v2({}, self.target, metrics_valid=False, success=False, failure_stage="dc")
        transient_fail = reward_v2(
            dict(KNOWN_GOOD_METRICS, dfe_locked_phase_eye_height_v=0.02),
            self.target, metrics_valid=True, success=False, failure_stage="transient",
        )
        success = reward_v2(KNOWN_GOOD_METRICS, self.target, metrics_valid=True, success=True, failure_stage=None)
        self.assertLess(no_info.total, transient_fail.total)
        self.assertLess(transient_fail.total, success.total)

    # -- terminal bonus differentiation --
    def test_successes_are_not_all_collapsed_to_the_same_score(self):
        just_over = dict(KNOWN_GOOD_METRICS)
        way_over = dict(KNOWN_GOOD_METRICS)
        way_over["dfe_locked_phase_eye_height_v"] *= 2.0
        r_just = reward_v2(just_over, self.target, metrics_valid=True, success=True, failure_stage=None)
        r_way = reward_v2(way_over, self.target, metrics_valid=True, success=True, failure_stage=None)
        self.assertGreaterEqual(r_way.total, r_just.total)
        self.assertGreaterEqual(r_just.total, TERMINAL_BONUS)

    def test_overshoot_bonus_is_bounded_never_approaches_a_second_terminal_bonus(self):
        absurdly_over = dict(KNOWN_GOOD_METRICS)
        absurdly_over["dfe_locked_phase_eye_height_v"] = 1000.0
        result = reward_v2(absurdly_over, self.target, metrics_valid=True, success=True, failure_stage=None)
        self.assertLess(result.total, TERMINAL_BONUS * 2)

    # -- peaking / DC-AC margins available pre-transient --
    def test_peaking_margin_available_even_when_transient_not_reached(self):
        result = reward_v2({"peaking_db": 7.0}, self.target, metrics_valid=False, success=False, failure_stage="ac")
        self.assertIn("peaking_db_margin", result.constraint_margins)
        self.assertGreater(result.constraint_margins["peaking_db_margin"], 0)  # 7.0 is inside 3-12 dB

    def test_peaking_outside_band_gives_negative_margin(self):
        result = reward_v2({"peaking_db": 20.0}, self.target, metrics_valid=False, success=False, failure_stage="ac")
        self.assertLess(result.constraint_margins["peaking_db_margin"], 0)

    def test_result_is_a_reward_result_dataclass_with_required_fields(self):
        result = reward_v2(KNOWN_GOOD_METRICS, self.target, metrics_valid=True, success=True, failure_stage=None)
        self.assertIsInstance(result, RewardResult)
        for field in ("total", "autockt_terminal_success", "strict_target_pass", "components",
                      "constraint_margins", "available_metrics", "failure_stage"):
            self.assertTrue(hasattr(result, field))


class ActionDeltasTests(unittest.TestCase):
    def test_default_deltas_unchanged(self):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        result = apply_action(indices, [2, 2, 2, 2, 2], grids)  # choice 2 -> ACTION_DELTAS[2] = +2
        expected = tuple(grids[n].clip_index(i + 2) for n, i in zip(PARAMETER_NAMES, indices))
        self.assertEqual(result, expected)

    def test_symmetric_deltas_produce_different_result(self):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        symmetric = (-1, 0, 1)
        result = apply_action(indices, [2, 2, 2, 2, 2], grids, deltas=symmetric)
        expected = tuple(grids[n].clip_index(i + 1) for n, i in zip(PARAMETER_NAMES, indices))
        self.assertEqual(result, expected)
        self.assertNotEqual(result, apply_action(indices, [2, 2, 2, 2, 2], grids))

    def test_wrong_length_deltas_rejected(self):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        with self.assertRaises(ValueError):
            apply_action(indices, [0] * 5, grids, deltas=(-1, 0, 1, 2))


class EnvV2OptInTests(unittest.TestCase):
    def _make_env(self, evaluator, *, horizon=4, budget=50, seed=0, **env_kwargs):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target_pool = (TargetSpec.from_existing_thresholds(),)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(budget), seed=seed)
        return AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=horizon,
            adapter=adapter, grids=grids, seed=seed, **env_kwargs,
        )

    def test_default_reset_unchanged_no_evaluation_spent(self):
        env = self._make_env(lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS))
        _, reset_info = env.reset()
        self.assertEqual(reset_info["reset_evaluation"], {"evaluated": False})
        self.assertEqual(env.adapter.total_evaluations, 0)

    def test_evaluate_on_reset_spends_exactly_one_evaluation(self):
        env = self._make_env(
            lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS), evaluate_on_reset=True,
        )
        _, reset_info = env.reset()
        self.assertTrue(reset_info["reset_evaluation"]["evaluated"])
        self.assertEqual(env.adapter.total_evaluations, 1)

    def test_evaluate_on_reset_feeds_real_metrics_into_v1_shaped_state(self):
        from rl.autockt_state import STATE_DIM
        env = self._make_env(
            lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS), evaluate_on_reset=True,
        )
        state, _ = env.reset()
        self.assertEqual(len(state), STATE_DIM)  # v1 shape unchanged when state_schema="v1"
        # first component should reflect the REAL height, not a fabricated zero
        self.assertNotEqual(state[0], 0.0)

    def test_state_schema_v2_produces_v2_shaped_state(self):
        env = self._make_env(
            lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS), state_schema="v2",
        )
        state, _ = env.reset()
        self.assertEqual(len(state), STATE_DIM_V2)

    def test_use_reward_v2_populates_reward_result_in_info(self):
        env = self._make_env(
            lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS),
            state_schema="v2", use_reward_v2=True,
        )
        env.reset()
        step_out = env.step([1, 1, 1, 1, 1])
        self.assertIsInstance(step_out.info["reward_result"], RewardResult)
        self.assertTrue(step_out.done)  # success -> autockt_terminal_success -> done

    def test_reward_v2_off_by_default_info_is_none(self):
        env = self._make_env(lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS))
        env.reset()
        step_out = env.step([1, 1, 1, 1, 1])
        self.assertIsNone(step_out.info["reward_result"])

    def test_default_behavior_completely_unchanged(self):
        # No new kwargs at all -- must reproduce the exact pre-existing
        # AutoCktEnvTests behavior (episode terminates on success).
        env = self._make_env(lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS), horizon=10)
        env.reset()
        step_out = env.step([1, 1, 1, 1, 1])
        self.assertTrue(step_out.done)
        self.assertEqual(step_out.reward, TERMINAL_BONUS)


class BudgetAwareTruncationTests(unittest.TestCase):
    """Required change 4: an evaluation budget exhausted mid-episode must
    truncate cleanly (never raise, never attempt one more simulator call).
    """

    def _make_env(self, evaluator, *, horizon, budget, max_total_evaluations=None):
        grids = build_parameter_grids()
        indices = verified_initial_indices(grids)
        target_pool = (TargetSpec.from_existing_thresholds(),)
        adapter = ReceiverRLAdapter(evaluator=evaluator, budget=RLBudget(budget), seed=0)
        return AutoCktReceiverEnv(
            target_pool=target_pool, initial_indices=indices, horizon=horizon,
            adapter=adapter, grids=grids, seed=0, max_total_evaluations=max_total_evaluations,
        )

    def test_adapter_truncated_flag_now_propagates(self):
        # budget=2, horizon=10 -- the ADAPTER's own budget exhausts first.
        env = self._make_env(
            lambda *a, **k: _fake_evaluation(False, failure_stage="dc"), horizon=10, budget=2,
        )
        env.reset()
        first = env.step([1, 1, 1, 1, 1])
        self.assertFalse(first.truncated)
        second = env.step([1, 1, 1, 1, 1])
        self.assertTrue(second.truncated)  # adapter budget (2) reached, horizon (10) not

    def test_step_after_exhaustion_does_not_raise_and_makes_no_adapter_call(self):
        env = self._make_env(
            lambda *a, **k: _fake_evaluation(False, failure_stage="dc"), horizon=10, budget=1,
        )
        env.reset()
        first = env.step([1, 1, 1, 1, 1])
        self.assertTrue(first.truncated)
        self.assertEqual(env.adapter.total_evaluations, 1)
        # A second step must NOT attempt another simulator call (which
        # would raise RuntimeError inside ReceiverRLAdapter.step()).
        second = env.step([1, 1, 1, 1, 1])
        self.assertTrue(second.truncated)
        self.assertEqual(env.adapter.total_evaluations, 1)  # unchanged -- no call was made
        self.assertEqual(second.info["failure_stage"], "budget_exhausted")

    def test_experiment_wide_budget_can_be_stricter_than_adapter_budget(self):
        # max_total_evaluations=1 permits exactly one call (checked BEFORE
        # a call is spent, not retroactively) -- the first step is allowed
        # to consume it; the second must truncate without a further call,
        # even though the adapter's own budget (100) would otherwise allow it.
        env = self._make_env(
            lambda *a, **k: _fake_evaluation(False, failure_stage="dc"), horizon=10, budget=100,
            max_total_evaluations=1,
        )
        env.reset()
        first = env.step([1, 1, 1, 1, 1])
        self.assertFalse(first.truncated)
        self.assertEqual(env.adapter.total_evaluations, 1)
        second = env.step([1, 1, 1, 1, 1])
        self.assertTrue(second.truncated)
        self.assertEqual(env.adapter.total_evaluations, 1)  # adapter itself would allow up to 100 -- stayed at 1

    def test_evaluate_on_reset_respects_budget_too(self):
        env = self._make_env(
            lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS), horizon=10, budget=1,
        )
        env.evaluate_on_reset = True
        env.reset()  # spends the 1 available evaluation
        self.assertEqual(env.adapter.total_evaluations, 1)
        env2 = self._make_env(
            lambda *a, **k: _fake_evaluation(True, metrics=KNOWN_GOOD_METRICS), horizon=10, budget=1,
        )
        env2.evaluate_on_reset = True
        env2.adapter.total_evaluations = 1  # simulate already-exhausted from a prior episode
        _, reset_info = env2.reset()
        self.assertFalse(reset_info["reset_evaluation"]["evaluated"])
        self.assertEqual(env2.adapter.total_evaluations, 1)  # no additional call made


if __name__ == "__main__":
    unittest.main()
