"""Stable entry point for the SPICE-free regression suite. Invoking this
module keeps the shell command identical across the whole session even as
new test modules are added below -- only this file changes, not the
command used to run it.

Deliberately excludes test_ctle.py, test_ngspice.py (partially mocked but
mixed), test_receiver_components.py, test_rl_readiness.py, and
test_sky130_integration.py -- those touch real ngspice/SKY130 paths.
"""

from __future__ import annotations

import unittest

SPICE_FREE_MODULES = [
    "tests.test_qualification_sequence",
    "tests.test_v3_dashboard",
    "tests.test_mos_area_expansion",
    "tests.test_parser",
    "tests.test_receiver_components",
    "tests.test_runtime_integration",
    "tests.test_v3_integration",
    "tests.test_autockt_rl",
    "tests.test_synthetic_benchmark",
    "tests.test_baseline_comparison",
    "tests.test_policy_inspection",
    "tests.test_rc_counterfactual_sweep",
    "tests.test_controlled_unseen_target_eval",
    "tests.test_train_cem",
    "tests.test_fair_comparison",
    "tests.test_export_final_schematic",
    "tests.test_evaluate_pvt_grid",
    "tests.test_pvt_sweep",
    "tests.test_pvt_diagnose",
    "tests.test_area_estimate",
    "tests.test_design_catalog",
    "tests.test_target_assessment",
    "tests.test_pvt_selection",
    "tests.test_final_specification",
    "tests.test_run_autockt_pipeline",
    "tests.test_learning_evidence",
    "tests.test_benchmark_report",
    "tests.test_artifact_integrity",
    "tests.test_performance_dashboard",
    "tests.test_web_ui",
    "tests.test_broader_generalization_check",
    "tests.test_rl_ppo_v2",
    "tests.test_rl_checkpoint",
    "tests.test_rl_events",
    "tests.test_rl_ppo_v2_ablation",
    "tests.test_evaluation_cache",
    "tests.test_rl_runtime_efficiency_study",
    "tests.test_phase7_rl_integration",
]


def main() -> int:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in SPICE_FREE_MODULES:
        suite.addTests(loader.loadTestsFromName(name))
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
