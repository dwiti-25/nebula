"""Tests for analysis/final_specification.py -- Task 2 (overnight chunk,
sec 22), the authoritative final-design specification report. Consumes
already-known data (or the pipeline's own output); no simulation here.
"""

from __future__ import annotations

import unittest

from analysis.final_specification import build_final_specification_report, format_report
from analysis.pvt_selection import PVTPointResult, summarize_pvt_results
from rl.target_spec import TargetSpec

DESIGN_A_PARAMS = {
    "rload_ohm": 2342.472156058411, "rdeg_ohm": 822.3558626926603,
    "cdeg_f": 9.999375862792168e-13, "itail_a": 0.0006028331705063624,
    "dfe_tap_v": -0.011090823885148815,
}
DESIGN_A_NOMINAL_METRICS = {
    "peaking_db": 7.070704101946522,
    "hd3_db": -79.92810113972858,
    "input_referred_noise_vrms": 0.0003613874185493741,
    "ctle_power_w": 0.0010850994,
    "dfe_locked_phase_eye_height_v": 1.5548473759143944,
    "dfe_eye_width_ui": 0.8699999999999999,
    "dfe_min_margin_v": 0.525491567711839,
}


class BuildReportTests(unittest.TestCase):
    def test_all_measured_specs_pass_for_the_known_good_design(self):
        report = build_final_specification_report(
            design_id="design_a", parameters=DESIGN_A_PARAMS,
            nominal_metrics=DESIGN_A_NOMINAL_METRICS, nominal_source="test fixture",
        )
        by_metric = {row["metric"]: row for row in report["rows"]}
        for metric in ("Eye width (UI)", "Eye height (V)", "Margin (V)", "Power (W)",
                        "Peaking (dB)", "HD3 (dB)", "Input-referred noise (Vrms)"):
            self.assertEqual(by_metric[metric]["verdict"], "PASS", metric)

    def test_missing_metric_is_not_claimed_not_fabricated(self):
        report = build_final_specification_report(
            design_id="design_a", parameters=DESIGN_A_PARAMS,
            nominal_metrics={}, nominal_source="test fixture (empty)",
        )
        by_metric = {row["metric"]: row for row in report["rows"]}
        for row in by_metric.values():
            if row["metric"].startswith(("Eye", "Margin", "Power", "Peaking", "HD3", "Input")):
                self.assertEqual(row["verdict"], "NOT CLAIMED")
                self.assertIsNone(row["measured"])

    def test_area_is_always_partial_never_pass_fail_against_budget(self):
        report = build_final_specification_report(
            design_id="design_a", parameters=DESIGN_A_PARAMS,
            nominal_metrics=DESIGN_A_NOMINAL_METRICS, nominal_source="test fixture",
        )
        by_metric = {row["metric"]: row for row in report["rows"]}
        area_row = by_metric["Transistor channel area (mm^2)"]
        self.assertEqual(area_row["verdict"], "NOT CLAIMED")
        self.assertIsNotNone(area_row["measured"])  # the partial number IS reported
        self.assertIn("PARTIAL", area_row["requirement"])

    def test_failing_metric_is_marked_fail_not_pass(self):
        weak_metrics = dict(DESIGN_A_NOMINAL_METRICS)
        weak_metrics["ctle_power_w"] = 0.02  # exceeds 15mW budget
        report = build_final_specification_report(
            design_id="design_a", parameters=DESIGN_A_PARAMS,
            nominal_metrics=weak_metrics, nominal_source="test fixture",
        )
        by_metric = {row["metric"]: row for row in report["rows"]}
        self.assertEqual(by_metric["Power (W)"]["verdict"], "FAIL")

    def test_pvt_row_reflects_a_supplied_pvt_result(self):
        pvt = summarize_pvt_results("design_a", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, False, "transient"),
        ])
        report = build_final_specification_report(
            design_id="design_a", parameters=DESIGN_A_PARAMS,
            nominal_metrics=DESIGN_A_NOMINAL_METRICS, nominal_source="test fixture",
            pvt_result=pvt,
        )
        by_metric = {row["metric"]: row for row in report["rows"]}
        pvt_row = by_metric["PVT (pass/total)"]
        self.assertEqual(pvt_row["measured"], "1/2")
        self.assertEqual(pvt_row["verdict"], "PARTIAL")
        self.assertIn("ff", pvt_row["source"])

    def test_pvt_row_is_pass_at_full_pass_rate(self):
        pvt = summarize_pvt_results("design_a", [
            PVTPointResult("tt", 1.8, 27.0, True, None),
            PVTPointResult("ff", 1.71, 125.0, True, None),
        ])
        report = build_final_specification_report(
            design_id="design_a", parameters=DESIGN_A_PARAMS,
            nominal_metrics=DESIGN_A_NOMINAL_METRICS, nominal_source="test fixture",
            pvt_result=pvt,
        )
        by_metric = {row["metric"]: row for row in report["rows"]}
        self.assertEqual(by_metric["PVT (pass/total)"]["verdict"], "PASS")

    def test_no_pvt_result_is_not_claimed(self):
        report = build_final_specification_report(
            design_id="design_a", parameters=DESIGN_A_PARAMS,
            nominal_metrics=DESIGN_A_NOMINAL_METRICS, nominal_source="test fixture",
            pvt_result=None,
        )
        by_metric = {row["metric"]: row for row in report["rows"]}
        self.assertEqual(by_metric["PVT (pass/total)"]["verdict"], "NOT CLAIMED")

    def test_report_qualifies_against_the_requested_target(self):
        target = TargetSpec(2.0, 0.4, 0.0, 0.015)
        report = build_final_specification_report(
            design_id="design_a", parameters=DESIGN_A_PARAMS,
            nominal_metrics=DESIGN_A_NOMINAL_METRICS, nominal_source="test fixture",
            target=target,
        )
        self.assertEqual(report["requested_target"], target.as_dict())
        self.assertFalse(report["target_qualification"]["passed"])
        failed = [m["name"] for m in report["target_qualification"]["metrics"] if not m["passed"]]
        self.assertEqual(failed, ["dfe_locked_phase_eye_height_v"])


class FormatReportTests(unittest.TestCase):
    def test_format_report_includes_parameters_and_all_rows(self):
        report = build_final_specification_report(
            design_id="design_a", parameters=DESIGN_A_PARAMS,
            nominal_metrics=DESIGN_A_NOMINAL_METRICS, nominal_source="test fixture",
        )
        text = format_report(report)
        self.assertIn("design_a", text)
        self.assertIn("rload_ohm", text)
        for row in report["rows"]:
            self.assertIn(row["metric"], text)


if __name__ == "__main__":
    unittest.main()
