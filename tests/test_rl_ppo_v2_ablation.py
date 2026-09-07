"""SPICE-free sanity tests for experiments/rl_ppo_v2_ablation.py.
Uses a tiny seed count -- the real 20-seed study is a separate, already-
run experiment (results/rl_ppo_v2_ablation_20seeds.jsonl), not repeated
here.
"""

from __future__ import annotations

import unittest

from experiments.rl_ppo_v2_ablation import VARIANTS, run_all, run_one, summarize


class RunOneTests(unittest.TestCase):
    def test_all_five_variants_run_without_error(self):
        for variant in VARIANTS:
            result = run_one(variant, seed=0)
            self.assertEqual(result["variant"], variant)
            self.assertGreater(result["n_evaluations"], 0)

    def test_result_has_every_required_metric(self):
        result = run_one("v1_baseline", seed=0)
        for key in (
            "strict_success_rate", "loose_success_rate", "evaluations_to_first_strict_success",
            "success_within_budget", "mean_reward", "failure_stage_distribution", "boundary_hit_rate",
            "distance_travelled", "held_out_strict_success_rate", "wall_clock_s",
        ):
            self.assertIn(key, result)

    def test_v5_uses_symmetric_deltas_not_default(self):
        from rl.parameter_grid import ACTION_DELTAS
        self.assertNotEqual(VARIANTS["v5_symmetric_actions"]["action_deltas"], ACTION_DELTAS)
        self.assertEqual(VARIANTS["v5_symmetric_actions"]["action_deltas"], (-1, 0, 1))

    def test_variants_are_incrementally_nested_per_the_required_ladder(self):
        self.assertFalse(VARIANTS["v1_baseline"]["evaluate_on_reset"])
        self.assertTrue(VARIANTS["v2_corrected_reset"]["evaluate_on_reset"])
        self.assertEqual(VARIANTS["v2_corrected_reset"]["state_schema"], "v1")
        self.assertEqual(VARIANTS["v3_validity_state"]["state_schema"], "v2")
        self.assertFalse(VARIANTS["v3_validity_state"]["use_reward_v2"])
        self.assertTrue(VARIANTS["v4_reward_v2"]["use_reward_v2"])


class SummarizeTests(unittest.TestCase):
    def test_no_successes_reports_none_not_a_fabricated_zero(self):
        results = run_all(seeds=(0, 1))
        summary = summarize(results)
        for variant_summary in summary.values():
            if variant_summary["n_seeds_that_succeeded"] == 0:
                self.assertIsNone(variant_summary["mean_evaluations_to_first_success"])

    def test_summary_covers_every_variant(self):
        results = run_all(seeds=(0,))
        summary = summarize(results)
        self.assertEqual(set(summary), set(VARIANTS))


if __name__ == "__main__":
    unittest.main()
