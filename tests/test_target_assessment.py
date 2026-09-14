from __future__ import annotations

import unittest

from analysis.target_assessment import assess_target
from rl.autockt_reward import TERMINAL_BONUS, autockt_reward
from rl.target_spec import TargetSpec


class StrictTargetAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.target = TargetSpec(
            dfe_locked_phase_eye_height_v=0.8,
            dfe_eye_width_ui=0.6,
            dfe_min_margin_v=0.35,
            ctle_power_w=0.015,
        )
        self.passing = {
            "dfe_locked_phase_eye_height_v": 0.8,
            "dfe_eye_width_ui": 0.6,
            "dfe_min_margin_v": 0.35,
            "ctle_power_w": 0.015,
        }

    def test_exact_boundaries_pass(self):
        result = assess_target(self.passing, self.target, simulator_success=True)
        self.assertTrue(result.passed)
        self.assertTrue(all(metric.margin == 0.0 for metric in result.metrics))

    def test_each_metric_must_pass_independently(self):
        metrics = dict(self.passing)
        metrics["dfe_locked_phase_eye_height_v"] = 0.79
        result = assess_target(metrics, self.target, simulator_success=True)
        self.assertFalse(result.passed)
        failed = [metric.name for metric in result.metrics if not metric.passed]
        self.assertEqual(failed, ["dfe_locked_phase_eye_height_v"])

    def test_reward_terminal_tolerance_is_not_engineering_pass(self):
        metrics = dict(self.passing)
        metrics["dfe_locked_phase_eye_height_v"] = 0.79
        self.assertEqual(autockt_reward(metrics, self.target, success=True), TERMINAL_BONUS)
        self.assertFalse(assess_target(metrics, self.target, simulator_success=True).passed)

    def test_missing_metric_is_not_zero_and_fails_closed(self):
        metrics = dict(self.passing)
        del metrics["dfe_min_margin_v"]
        result = assess_target(metrics, self.target, simulator_success=True)
        self.assertFalse(result.passed)
        self.assertEqual(result.missing_metrics, ("dfe_min_margin_v",))
        row = next(metric for metric in result.metrics if metric.name == "dfe_min_margin_v")
        self.assertIsNone(row.measured)

    def test_simulator_failure_fails_even_when_metrics_clear_target(self):
        result = assess_target(self.passing, self.target, simulator_success=False)
        self.assertFalse(result.passed)

    def test_zero_margin_target_has_finite_normalization(self):
        target = TargetSpec(0.1, 0.4, 0.0, 0.015)
        metrics = dict(self.passing)
        metrics.update({"dfe_locked_phase_eye_height_v": 0.1, "dfe_eye_width_ui": 0.4,
                        "dfe_min_margin_v": 0.05})
        result = assess_target(metrics, target, simulator_success=True)
        row = next(metric for metric in result.metrics if metric.name == "dfe_min_margin_v")
        self.assertEqual(row.normalized_margin, 0.5)


if __name__ == "__main__":
    unittest.main()
