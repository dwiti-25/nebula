from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from analysis.plot_renderer import evidence_hash, render_chart
from analysis.rl_statistics import bootstrap_interval, interquartile_mean, performance_profile, probability_of_improvement
from analysis.run_evidence import JsonlEventWriter, build_run_dashboard
from experiments import web_ui
from experiments.run_autockt_pipeline import generate_candidates
from rl.target_spec import TargetSpec


class VersionedPipelineTests(unittest.TestCase):
    def test_v1_and_v2_are_both_runnable(self):
        for version in ("v1", "v2"):
            candidates = generate_candidates(
                target=TargetSpec.from_existing_thresholds(), checkpoint_path=None,
                backend="synthetic", episodes=1, horizon=1, rl_version=version,
            )
            self.assertEqual(len(candidates), 1)

    def test_ui_exposes_and_forwards_both_versions(self):
        self.assertIn('id="rlVersion"', web_ui.INDEX_HTML)
        base = {"target_mode": "trivial", "backend": "synthetic", "episodes": 1, "horizon": 1,
                "pvt_condition_set": "none", "trade_off_preference": "most_robust", "rl_version": "v2"}
        argv = web_ui._build_argv(base, output_path=Path("x.json"), schematic_path=Path("x.spice"))
        self.assertEqual(argv[argv.index("--rl-version") + 1], "v2")


class PlotRendererTests(unittest.TestCase):
    CHART = {"id": "sample", "title": "Sample", "type": "line", "x_label": "evaluation",
             "y_label": "reward", "series": [{"name": "v1", "values": [0, 1, 2]}]}

    def test_svg_and_png_render_with_expected_signatures(self):
        self.assertIn(b"<svg", render_chart(self.CHART, format="svg")[:500])
        self.assertTrue(render_chart(self.CHART, format="png").startswith(b"\x89PNG"))

    def test_hash_is_order_independent(self):
        self.assertEqual(evidence_hash(self.CHART), evidence_hash(dict(reversed(list(self.CHART.items())))))


class EventPersistenceTests(unittest.TestCase):
    def test_writer_flushes_enriched_strict_assessment_and_dashboard(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.events.jsonl"
            writer = JsonlEventWriter(path, metadata={"configuration_id": "ppo_v2"})
            writer({"event_type": "step", "target_values": TargetSpec.from_existing_thresholds().as_dict(),
                    "raw_metrics": {"dfe_locked_phase_eye_height_v": .2, "dfe_eye_width_ui": .5,
                                    "dfe_min_margin_v": .1, "ctle_power_w": .01},
                    "failure_stage": None, "reward_total": 1.0})
            row = json.loads(path.read_text().strip())
            self.assertTrue(row["strict_target_assessment"]["passed"])
            self.assertGreaterEqual(len(build_run_dashboard(path)["charts"]), 3)


class StatisticsTests(unittest.TestCase):
    def test_hand_computable_statistics(self):
        self.assertEqual(interquartile_mean([1, 2, 3, 100]), 2.5)
        self.assertEqual(probability_of_improvement([2], [1]), 1.0)
        self.assertEqual(performance_profile([0, 1], [0, 1, 2]), [1.0, .5, 0.0])
        low, high = bootstrap_interval([1, 2, 3, 4], samples=100, seed=7)
        self.assertLessEqual(low, high)


if __name__ == "__main__":
    unittest.main()
