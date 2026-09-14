from __future__ import annotations

import unittest
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from analysis.performance_dashboard import build_dashboard


class PerformanceDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dashboard = build_dashboard(Path(__file__).resolve().parents[1])

    def test_dashboard_has_multiple_evidence_backed_graphs(self):
        self.assertGreaterEqual(len(self.dashboard["charts"]), 6)
        self.assertTrue(self.dashboard["sources"])

    def test_chart_ids_are_unique(self):
        ids = [chart["id"] for chart in self.dashboard["charts"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_summary_matches_archived_evidence(self):
        summary = self.dashboard["summary"]
        self.assertGreater(summary["ppo_evaluations"], 0)
        self.assertEqual(summary["pvt_passes"], summary["pvt_total"])
        self.assertEqual(summary["pvt_total"], 27)

    def test_historical_metric_limitation_is_explicit(self):
        self.assertTrue(any("raw metrics" in item for item in self.dashboard["limitations"]))

    def test_clean_checkout_reads_evidence_directly_from_the_artifact_zip(self):
        repository = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as temp:
            clean_root = Path(temp)
            (clean_root / "artifacts").mkdir()
            copyfile(
                repository / "artifacts" / "NEBULA_phase2_experimental_artifacts.zip",
                clean_root / "artifacts" / "NEBULA_phase2_experimental_artifacts.zip",
            )
            dashboard = build_dashboard(clean_root)
        self.assertGreaterEqual(len(dashboard["charts"]), 6)
        self.assertEqual(dashboard["summary"]["pvt_total"], 27)


if __name__ == "__main__":
    unittest.main()
