"""Tests for analysis/benchmark_report.py -- Task 5 (NEXT IMPLEMENTATION
CHUNK). Anchors against the REAL, already-existing result files and the
ground-truth numbers independently established in
docs/autockt-mapping.md sec 19/sec 20. Computes nothing new; this module
is a consolidation layer, so these tests mainly guard that the
consolidation reproduces the underlying (already-tested)
analysis/fair_comparison.py numbers correctly and that the structural
metadata fields are actually populated and honest (e.g. the warm-started
trial must flag itself as non-comparable; the head-to-head trial must
not).
"""

from __future__ import annotations

import unittest
from pathlib import Path

from analysis.benchmark_report import (
    build_benchmark_report,
    no_warm_start_headtohead_trial,
    warm_started_trial,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "results/receiver_random_search_20_seed123.jsonl",
    "results/cem_unbiased_seed123.jsonl",
    "results/autockt_mixed_target_confirmation.jsonl",
    "results/cem_graded_unbiased_seed123.jsonl",
    "results/autockt_fair_headtohead_seed123.jsonl",
]


def _fixtures_present() -> bool:
    return all((REPOSITORY_ROOT / f).is_file() for f in REQUIRED_FILES)


class WarmStartedTrialGroundTruthTests(unittest.TestCase):
    def test_matches_sec19_headline_numbers(self):
        if not _fixtures_present():
            self.skipTest("fixtures not present")
        trial = warm_started_trial()
        self.assertEqual(trial.methods["random_search"].n_evaluations, 20)
        self.assertAlmostEqual(trial.methods["random_search"].uniform_success_rate, 1 / 20)
        self.assertEqual(trial.methods["cem"].n_evaluations, 20)
        self.assertEqual(trial.methods["cem"].uniform_success_rate, 0.0)
        self.assertEqual(trial.methods["ppo"].n_evaluations, 65)

    def test_flagged_as_not_comparable(self):
        if not _fixtures_present():
            self.skipTest("fixtures not present")
        trial = warm_started_trial()
        self.assertFalse(trial.comparable_across_methods)
        self.assertIn("VERIFIED_INITIAL_PARAMETERS", trial.initialization)


class HeadToHeadTrialGroundTruthTests(unittest.TestCase):
    def test_matches_sec20_headline_numbers(self):
        if not _fixtures_present():
            self.skipTest("fixtures not present")
        trial = no_warm_start_headtohead_trial()
        # sec 20: Random Search 1/20, CEM 0/20, PPO 0/20 -- all n=20.
        for method in ("random_search", "cem", "ppo"):
            self.assertEqual(trial.methods[method].n_evaluations, 20, method)
        self.assertAlmostEqual(trial.methods["random_search"].uniform_success_rate, 1 / 20)
        self.assertEqual(trial.methods["cem"].uniform_success_rate, 0.0)
        # This training log predates per-step `metrics` logging (same KNOWN
        # GAP as sec 19, never closed for this file either), so the uniform
        # recomputation honestly reports None rather than a fabricated 0.0
        # -- PPO's own NATIVE reward already targets the trivial target
        # directly here (target=trivial by construction), so native is the
        # right field to check for this file.
        self.assertIsNone(trial.methods["ppo"].uniform_success_rate)
        self.assertEqual(trial.methods["ppo"].uniform_scoreable_fraction, 0.0)
        self.assertEqual(trial.methods["ppo"].native_success_rate, 0.0)

    def test_flagged_as_comparable_but_still_carries_a_caveat(self):
        if not _fixtures_present():
            self.skipTest("fixtures not present")
        trial = no_warm_start_headtohead_trial()
        self.assertTrue(trial.comparable_across_methods)
        # comparable does not mean unconditionally conclusive -- the n=1
        # seed / structural-difference caveat must still be present.
        self.assertIn("n=1 seed", trial.comparability_caveat)


class BuildReportTests(unittest.TestCase):
    def test_report_contains_both_trials_and_a_conclusion(self):
        if not _fixtures_present():
            self.skipTest("fixtures not present")
        report = build_benchmark_report()
        self.assertIn("warm_started_ppo_sec19", report["trials"])
        self.assertIn("no_warm_start_headtohead_sec20", report["trials"])
        self.assertIn("conclusion", report)

    def test_conclusion_does_not_claim_a_ppo_advantage(self):
        if not _fixtures_present():
            self.skipTest("fixtures not present")
        report = build_benchmark_report()
        conclusion = report["conclusion"].lower()
        self.assertIn("no demonstrated", conclusion)

    def test_every_method_summary_has_no_fabricated_wall_clock(self):
        # Random Search's own file predates wall-clock instrumentation --
        # the report must say so, not invent a number.
        if not _fixtures_present():
            self.skipTest("fixtures not present")
        report = build_benchmark_report()
        rs = report["trials"]["no_warm_start_headtohead_sec20"]["methods"]["random_search"]
        self.assertFalse(rs["wall_clock_available"])
        self.assertIsNone(rs["wall_clock_total_s"])

    def test_strict_rate_distinguishes_recomputed_and_logged_evidence(self):
        if not _fixtures_present():
            self.skipTest("fixtures not present")
        methods = build_benchmark_report()["trials"]["no_warm_start_headtohead_sec20"]["methods"]
        self.assertEqual(methods["random_search"]["strict_success_evidence_grade"], "raw_metrics_recomputed")
        self.assertEqual(
            methods["ppo"]["strict_success_evidence_grade"],
            "logged_same_target_outcome_raw_metrics_unavailable",
        )
        self.assertEqual(methods["ppo"]["reported_strict_success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
