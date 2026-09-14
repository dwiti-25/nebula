import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from analysis.v3_dashboard import build_dashboard
from analysis.final_specification import build_final_specification_report
from simulator.cache import EvaluationCache
from simulator.provenance import spice_dependency_manifest


class ReviewRegressionTests(unittest.TestCase):
    def test_malformed_cache_is_a_miss(self):
        for value in ("[]", "null", "42"):
            with patch.object(Path, "is_file", return_value=True), patch.object(Path, "read_text", return_value=value):
                self.assertIsNone(EvaluationCache("unused").get("abc"))

    def test_report_uses_actual_area_and_strict_rf_limits(self):
        report = build_final_specification_report(design_id="x", parameters={
            "mos_width_um": 50, "mos_length_um": 1, "mos_multiplier": 4},
            nominal_metrics={"hd3_db": -30, "input_referred_noise_vrms": 0.0015}, nominal_source="test")
        rows = {r["metric"]: r for r in report["rows"]}
        self.assertEqual(rows["HD3 (dB)"]["verdict"], "FAIL")
        self.assertEqual(rows["Input-referred noise (Vrms)"]["verdict"], "FAIL")
        self.assertAlmostEqual(float(rows["Transistor channel area (mm^2)"]["measured"]), 0.0004)

    def test_lib_and_inc_dependencies_are_hashed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.spice").write_text('.lib "models.spice" tt\n.inc extra.spice\n')
            (root / "models.spice").write_text('.lib tt\n.endl tt\n')
            (root / "extra.spice").write_text('* extra\n')
            self.assertEqual(len(spice_dependency_manifest(root / "main.spice")), 3)

    def test_empty_dashboard_never_falls_back_to_old_models(self):
        with TemporaryDirectory() as tmp:
            dashboard = build_dashboard(tmp)
            self.assertFalse(dashboard["charts"])
            self.assertEqual(dashboard["summary"]["pvt_total"], 0)

    def test_local_v3_charts_render(self):
        from analysis.plot_renderer import render_chart
        dashboard = build_dashboard(Path(__file__).resolve().parents[1])
        ids = [c["id"] for c in dashboard["charts"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("optimizer_comparison", ids)
        for chart in dashboard["charts"]:
            result = render_chart(chart)
            self.assertIn(b"<svg", result)
