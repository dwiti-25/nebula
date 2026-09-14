"""Tests for analysis/pvt_selection.py -- Task 4 (PVT-aware candidate
selection). SPICE-free: run_pvt_evaluation is tested by mocking
evaluate_pvt_grid; the rest is pure aggregation/ranking over already-loaded
data (including the REAL results/design_a_pvt_minimal27.jsonl).
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from simulator.config import ProcessCorner, SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters

from analysis.design_catalog import FeasibleDesign
from analysis.pvt_selection import (
    PVTPointResult,
    load_pvt_results_from_jsonl,
    rank_by_robustness,
    run_pvt_evaluation,
    select_final_designs,
    select_with_trade_off_preference,
    summarize_pvt_results,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SummarizePvtResultsTests(unittest.TestCase):
    def test_pass_rate_and_worst_case_conditions(self):
        points = [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, False, "transient"),
            PVTPointResult("ss", 1.89, 0.0, True, None),
        ]
        result = summarize_pvt_results("x", points)
        self.assertEqual(result.n_conditions, 3)
        self.assertEqual(result.n_passing, 2)
        self.assertAlmostEqual(result.pass_rate, 2 / 3)
        self.assertEqual(len(result.worst_case_conditions), 1)
        self.assertEqual(result.worst_case_conditions[0].process_corner, "ff")

    def test_empty_points_gives_zero_pass_rate_not_a_crash(self):
        result = summarize_pvt_results("x", [])
        self.assertEqual(result.pass_rate, 0.0)
        self.assertEqual(result.n_conditions, 0)


class LoadFromRealDataTests(unittest.TestCase):
    def test_matches_the_known_23_of_27_result(self):
        path = REPOSITORY_ROOT / "results" / "design_a_pvt_minimal27.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        result = load_pvt_results_from_jsonl("design_a", path)
        self.assertEqual(result.n_conditions, 27)
        self.assertEqual(result.n_passing, 23)
        self.assertAlmostEqual(result.pass_rate, 23 / 27)
        self.assertEqual(len(result.worst_case_conditions), 4)
        for point in result.worst_case_conditions:
            self.assertEqual(point.process_corner, "ff")
            self.assertEqual(point.failed_stage, "transient")

    def test_matches_the_reproducibility_rerun_27_of_27_result(self):
        # docs/autockt-mapping.md sec 22 Task 1: the original 23/27 above
        # was found to reflect simulation-level non-reproducibility at 4
        # points, not a design defect -- Design A itself was never changed.
        # Both results are preserved; this is the current, most-verified one.
        path = REPOSITORY_ROOT / "results" / "design_a_pvt_minimal27_rerun.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        result = load_pvt_results_from_jsonl("design_a", path)
        self.assertEqual(result.n_conditions, 27)
        self.assertEqual(result.n_passing, 27)
        self.assertEqual(result.pass_rate, 1.0)
        self.assertEqual(len(result.worst_case_conditions), 0)


class RunPvtEvaluationTests(unittest.TestCase):
    def test_calls_evaluate_pvt_grid_with_the_given_conditions_and_summarizes(self):
        design = FeasibleDesign(
            design_id="test_design", source_file="f", source_description="d",
            parameters={"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                        "itail_a": 1e-4, "dfe_tap_v": 0.0},
            metrics={}, native_reward=10.0, native_reward_scale="autockt_reward",
        )
        conditions = (
            SimulationConditions(ProcessCorner.TT, 27.0, 1.8),
            SimulationConditions(ProcessCorner.FF, 125.0, 1.71),
        )

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            results = []
            for i, c in enumerate(conditions):
                success = i == 0  # first passes, second fails
                results.append(ReceiverEvaluation(
                    success, parameters, c, fidelity, (), {},
                    None if success else "transient", 0.0, f"id-{i}", {},
                ))
            return tuple(results)

        with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
            result = run_pvt_evaluation(design, conditions)

        self.assertEqual(result.design_id, "test_design")
        self.assertEqual(result.n_conditions, 2)
        self.assertEqual(result.n_passing, 1)
        self.assertEqual(len(result.worst_case_conditions), 1)
        self.assertEqual(result.worst_case_conditions[0].process_corner, "ff")

    def test_requested_target_is_checked_at_every_corner(self):
        from rl.target_spec import TargetSpec

        design = FeasibleDesign(
            design_id="test_design", source_file="f", source_description="d",
            parameters={"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                        "itail_a": 1e-4, "dfe_tap_v": 0.0},
            metrics={}, native_reward=10.0, native_reward_scale="autockt_reward",
        )
        conditions = (SimulationConditions(ProcessCorner.TT, 27.0, 1.8),)
        target = TargetSpec(0.8, 0.6, 0.35, 0.015)
        below_target = {
            "dfe_locked_phase_eye_height_v": 0.79, "dfe_eye_width_ui": 0.6,
            "dfe_min_margin_v": 0.35, "ctle_power_w": 0.001,
        }

        def fake_grid(parameters, *, conditions, fidelity):
            return tuple(ReceiverEvaluation(
                True, parameters, condition, fidelity, (), below_target, None, 0.0, "id", {},
            ) for condition in conditions)

        with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_grid):
            result = run_pvt_evaluation(design, conditions, target=target)
        self.assertEqual(result.n_passing, 0)
        self.assertTrue(result.points[0].simulator_success)
        self.assertFalse(result.points[0].target_assessment.passed)


class RankAndSelectTests(unittest.TestCase):
    def test_rank_by_robustness_orders_by_pass_rate_descending(self):
        low = summarize_pvt_results("low", [PVTPointResult("tt", 1.8, 27.0, False, "ac")])
        high = summarize_pvt_results("high", [PVTPointResult("tt", 1.8, 27.0, True, None)])
        ranked = rank_by_robustness([low, high])
        self.assertEqual([r.design_id for r in ranked], ["high", "low"])

    def test_select_final_designs_respects_minimum_pass_rate(self):
        partial = summarize_pvt_results("partial", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, False, "transient"),
        ])
        full = summarize_pvt_results("full", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, True, None),
        ])
        selected = select_final_designs([partial, full], minimum_pass_rate=1.0, top_n=1)
        self.assertEqual(selected[0].design_id, "full")

    def test_select_final_designs_fails_closed_when_no_candidate_meets_the_bar(self):
        partial = summarize_pvt_results("partial", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, False, "transient"),
        ])
        selected = select_final_designs([partial], minimum_pass_rate=1.0, top_n=1)
        self.assertEqual(selected, [])


class SelectWithTradeOffPreferenceTests(unittest.TestCase):
    """The full 4-level priority: nominal feasibility (guaranteed by only
    passing already-feasible designs) -> PVT pass rate -> robustness
    tie-break -> secondary trade-off preference among ties only.
    """

    def _design(self, design_id, power=0.001, height=1.0, width=0.5, margin=0.3):
        return FeasibleDesign(
            design_id=design_id, source_file="f", source_description="d",
            parameters={"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                        "itail_a": 1e-4, "dfe_tap_v": 0.0},
            metrics={"ctle_power_w": power, "dfe_locked_phase_eye_height_v": height,
                     "dfe_eye_width_ui": width, "dfe_min_margin_v": margin},
            native_reward=10.0, native_reward_scale="autockt_reward",
        )

    def test_higher_pass_rate_always_wins_regardless_of_preference(self):
        low_power_low_robustness = summarize_pvt_results("low_power", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, False, "transient"),
        ])
        high_power_high_robustness = summarize_pvt_results("high_power", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, True, None),
        ])
        designs = [self._design("low_power", power=0.001), self._design("high_power", power=0.01)]
        selected = select_with_trade_off_preference(
            [low_power_low_robustness, high_power_high_robustness], designs, preference="lowest_power",
        )
        # high_power_high_robustness has the higher pass_rate -- preference
        # must NOT override that, even though it asks for lowest_power.
        self.assertEqual(selected.design_id, "high_power")

    def test_trade_off_preference_breaks_a_genuine_pvt_tie(self):
        tie_a = summarize_pvt_results("a", [PVTPointResult("tt", 1.8, 27.0, True, None)])
        tie_b = summarize_pvt_results("b", [PVTPointResult("tt", 1.8, 27.0, True, None)])
        designs = [self._design("a", power=0.01), self._design("b", power=0.001)]
        selected = select_with_trade_off_preference([tie_a, tie_b], designs, preference="lowest_power")
        self.assertEqual(selected.design_id, "b")

    def test_most_robust_preference_ignores_trade_offs(self):
        tie_a = summarize_pvt_results("a", [PVTPointResult("tt", 1.8, 27.0, True, None)])
        tie_b = summarize_pvt_results("b", [PVTPointResult("tt", 1.8, 27.0, True, None)])
        designs = [self._design("a", power=0.01), self._design("b", power=0.001)]
        selected = select_with_trade_off_preference([tie_a, tie_b], designs, preference="most_robust")
        self.assertEqual(selected.design_id, "a")  # first by robustness order, trade-offs never consulted

    def test_empty_results_returns_none(self):
        self.assertIsNone(select_with_trade_off_preference([], [], preference="most_robust"))

    def test_trade_off_selection_fails_closed_when_no_result_meets_minimum(self):
        partial = summarize_pvt_results("partial", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, False, "transient"),
        ])
        self.assertIsNone(select_with_trade_off_preference(
            [partial], [self._design("partial")], minimum_pass_rate=1.0,
        ))

    def test_unknown_preference_raises(self):
        with self.assertRaises(ValueError):
            select_with_trade_off_preference(
                [summarize_pvt_results("a", [PVTPointResult("tt", 1.8, 27.0, True, None)])],
                [self._design("a")], preference="not_a_real_preference",
            )


if __name__ == "__main__":
    unittest.main()
