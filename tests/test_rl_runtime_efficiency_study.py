"""SPICE-free sanity tests for experiments/rl_runtime_efficiency_study.py."""

from __future__ import annotations

import unittest

from experiments.rl_runtime_efficiency_study import run_all, run_pair, summarize


class RunPairTests(unittest.TestCase):
    def test_trajectories_are_identical_between_baseline_and_optimized(self):
        result = run_pair(0)
        self.assertTrue(result["trajectories_identical"])

    def test_evaluation_counts_match_between_configs_pre_cache_dedup(self):
        result = run_pair(0)
        self.assertEqual(result["baseline"]["n_evaluations"], result["optimized"]["n_evaluations"])

    def test_optimized_expensive_calls_never_exceed_baseline(self):
        result = run_pair(0)
        self.assertLessEqual(result["optimized_expensive_evaluator_calls"], result["baseline_expensive_evaluator_calls"])

    def test_cache_hits_plus_misses_equals_total_calls(self):
        result = run_pair(0)
        self.assertEqual(result["cache_hits"] + result["cache_misses"], result["baseline"]["n_evaluations"])

    def test_reward_and_success_metrics_identical_between_configs(self):
        result = run_pair(0)
        self.assertEqual(result["baseline"]["mean_reward"], result["optimized"]["mean_reward"])
        self.assertEqual(result["baseline"]["strict_success_rate"], result["optimized"]["strict_success_rate"])
        self.assertEqual(result["baseline"]["distance_travelled"], result["optimized"]["distance_travelled"])


class RunAllSummarizeTests(unittest.TestCase):
    def test_summary_reports_non_negative_evaluation_reduction(self):
        results = run_all(seeds=(0, 1))
        summary = summarize(results)
        self.assertGreaterEqual(summary["evaluation_reduction_pct"], 0.0)

    def test_summary_confirms_all_trajectories_identical(self):
        results = run_all(seeds=(0, 1, 2))
        summary = summarize(results)
        self.assertTrue(summary["all_trajectories_identical"])

    def test_summary_has_required_seed_count(self):
        results = run_all(seeds=(0, 1, 2))
        summary = summarize(results)
        self.assertEqual(summary["n_seeds"], 3)


if __name__ == "__main__":
    unittest.main()
