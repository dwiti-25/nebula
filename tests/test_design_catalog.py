"""Tests for analysis/design_catalog.py -- Task 3 (multiple feasible
designs / trade-offs). SPICE-free: parses existing results/*.jsonl and
small in-memory fixtures only.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.design_catalog import (
    FeasibleDesign,
    build_catalog,
    deduplicate,
    load_all_known_feasible_designs,
    load_rc_counterfactual_survivors,
    load_reward_directed_smoke_designs,
    rank_by_measured_trade_offs,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DeduplicateTests(unittest.TestCase):
    def test_drops_exact_duplicate_parameter_sets(self):
        base_params = {"rload_ohm": 1000.0, "rdeg_ohm": 500.0, "cdeg_f": 1e-13,
                        "itail_a": 1e-4, "dfe_tap_v": 0.0}
        a = FeasibleDesign("a", "f", "d", base_params, {}, 10.0, "autockt_reward")
        b = FeasibleDesign("b", "f", "d", dict(base_params), {}, 10.0, "autockt_reward")
        unique = deduplicate([a, b])
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0].design_id, "a")  # keeps first occurrence

    def test_keeps_genuinely_different_designs(self):
        a = FeasibleDesign("a", "f", "d", {"rload_ohm": 1000.0, "rdeg_ohm": 500.0,
                            "cdeg_f": 1e-13, "itail_a": 1e-4, "dfe_tap_v": 0.0}, {}, 10.0, "autockt_reward")
        b = FeasibleDesign("b", "f", "d", {"rload_ohm": 2000.0, "rdeg_ohm": 500.0,
                            "cdeg_f": 1e-13, "itail_a": 1e-4, "dfe_tap_v": 0.0}, {}, 10.0, "autockt_reward")
        unique = deduplicate([a, b])
        self.assertEqual(len(unique), 2)

    def test_keeps_distinct_pico_scale_capacitances(self):
        base = {"rload_ohm": 1000.0, "rdeg_ohm": 500.0,
                "cdeg_f": 1e-13, "itail_a": 1e-4, "dfe_tap_v": 0.0}
        changed = dict(base, cdeg_f=2e-13)
        a = FeasibleDesign("a", "f", "d", base, {}, 10.0, "autockt_reward")
        b = FeasibleDesign("b", "f", "d", changed, {}, 10.0, "autockt_reward")
        self.assertEqual(len(deduplicate([a, b])), 2)


class RankByMeasuredTradeOffsTests(unittest.TestCase):
    def test_label_only_assigned_to_the_actual_best_measured_value(self):
        low_power = FeasibleDesign("low_power", "f", "d", {}, {"ctle_power_w": 0.001}, 10.0, "autockt_reward")
        high_power = FeasibleDesign("high_power", "f", "d", {}, {"ctle_power_w": 0.010}, 10.0, "autockt_reward")
        ranked = rank_by_measured_trade_offs([low_power, high_power])
        by_id = {r.design.design_id: r.trade_off_labels for r in ranked}
        self.assertIn("lowest_power", by_id["low_power"])
        self.assertNotIn("lowest_power", by_id["high_power"])

    def test_a_design_that_is_not_best_on_any_axis_is_balanced(self):
        best = FeasibleDesign("best", "f", "d", {}, {
            "ctle_power_w": 0.001, "dfe_locked_phase_eye_height_v": 2.0,
            "dfe_eye_width_ui": 0.9, "dfe_min_margin_v": 0.6,
        }, 10.0, "autockt_reward")
        middling = FeasibleDesign("middling", "f", "d", {}, {
            "ctle_power_w": 0.005, "dfe_locked_phase_eye_height_v": 1.0,
            "dfe_eye_width_ui": 0.5, "dfe_min_margin_v": 0.3,
        }, 10.0, "autockt_reward")
        ranked = rank_by_measured_trade_offs([best, middling])
        by_id = {r.design.design_id: r.trade_off_labels for r in ranked}
        self.assertEqual(by_id["middling"], ("balanced",))

    def test_empty_pool_returns_empty(self):
        self.assertEqual(rank_by_measured_trade_offs([]), [])

    def test_ties_can_produce_multiple_winners(self):
        a = FeasibleDesign("a", "f", "d", {}, {"ctle_power_w": 0.001}, 10.0, "autockt_reward")
        b = FeasibleDesign("b", "f", "d", {}, {"ctle_power_w": 0.001}, 10.0, "autockt_reward")
        ranked = rank_by_measured_trade_offs([a, b])
        for entry in ranked:
            self.assertIn("lowest_power", entry.trade_off_labels)


class LoadRealDataGroundTruthTests(unittest.TestCase):
    def test_reward_directed_smoke_designs_are_recovered_correctly(self):
        path = REPOSITORY_ROOT / "results" / "rl_reward_directed_smoke.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        designs = load_reward_directed_smoke_designs(path)
        self.assertEqual(len(designs), 2)
        first = designs[0]
        self.assertAlmostEqual(first.metrics["dfe_locked_phase_eye_height_v"], 0.5514326967009642, places=6)
        self.assertAlmostEqual(first.metrics["dfe_eye_width_ui"], 0.74, places=6)
        self.assertAlmostEqual(first.metrics["dfe_min_margin_v"], 0.2294877551337197, places=6)
        # all rows in this file are success=true and satisfy the trivial target
        for design in designs:
            self.assertGreater(design.metrics["dfe_locked_phase_eye_height_v"], 0.1)
            self.assertGreater(design.metrics["dfe_eye_width_ui"], 0.4)
            self.assertGreater(design.metrics["dfe_min_margin_v"], 0.0)
            self.assertLess(design.metrics["ctle_power_w"], 0.015)

    def test_rc_counterfactual_survivors_are_all_genuinely_spec_satisfied(self):
        path = REPOSITORY_ROOT / "results" / "rc_counterfactual_sweep_mixed_target.jsonl"
        if not path.is_file():
            self.skipTest("fixture not present")
        designs = load_rc_counterfactual_survivors(path)
        self.assertEqual(len(designs), 7)  # confirmed count from docs sec 17
        for design in designs:
            self.assertEqual(design.native_reward, 10.0)
            self.assertEqual(design.native_reward_scale, "autockt_reward")
            self.assertEqual(design.uniform_reward, 10.0)

    def test_missing_source_file_returns_empty_not_an_error(self):
        self.assertEqual(load_reward_directed_smoke_designs("results/does_not_exist.jsonl"), [])
        self.assertEqual(load_rc_counterfactual_survivors("results/does_not_exist.jsonl"), [])


class BuildCatalogTests(unittest.TestCase):
    def test_catalog_has_at_least_design_a(self):
        catalog = build_catalog()
        ids = {entry.design.design_id for entry in catalog}
        self.assertIn("design_a", ids)

    def test_every_catalog_entry_has_at_least_one_label(self):
        catalog = build_catalog()
        for entry in catalog:
            self.assertGreaterEqual(len(entry.trade_off_labels), 1)

    def test_catalog_has_no_duplicate_parameter_sets(self):
        catalog = build_catalog()
        keys = [entry.design.parameter_key() for entry in catalog]
        self.assertEqual(len(keys), len(set(keys)))

    def test_native_reward_scale_is_disclosed_and_not_uniformly_comparable(self):
        # Regression guard for a real bug caught and fixed: sources use
        # different reward functions (autockt_reward, max 10.0, vs
        # reward_v1, a different scale) -- native_reward must never be
        # silently treated as comparable across differing scales.
        catalog = build_catalog()
        scales = {entry.design.native_reward_scale for entry in catalog}
        self.assertTrue(scales)  # at least one scale present
        for entry in catalog:
            self.assertIn(entry.design.native_reward_scale, ("autockt_reward", "reward_v1", "receiver_reward_v1"))

    def test_uniform_reward_is_comparable_across_the_whole_catalog(self):
        # Unlike native_reward, uniform_reward is recomputed the same way
        # for every entry and must agree (every catalog member is already
        # uniform-criterion feasible, so this is always the terminal bonus).
        catalog = build_catalog()
        for entry in catalog:
            self.assertEqual(entry.design.uniform_reward, 10.0)


if __name__ == "__main__":
    unittest.main()
