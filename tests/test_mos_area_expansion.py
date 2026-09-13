from pathlib import Path
import unittest

from analysis.area_estimate import estimate_ctle_area
from experiments.export_final_schematic import render_final_schematic
from experiments.web_ui import INDEX_HTML
from simulator.config import SimulationConditions
from simulator.receiver import BENCHES, ReceiverParameters


class GroupedMosSizingTests(unittest.TestCase):
    def test_legacy_defaults_are_preserved(self):
        parameters = ReceiverParameters()
        self.assertEqual(parameters.mos_width_um, 10.0)
        self.assertEqual(parameters.mos_length_um, 0.15)
        self.assertEqual(parameters.mos_multiplier, 1)

    def test_geometry_is_validated(self):
        for parameters in (
            ReceiverParameters(mos_width_um=0.1),
            ReceiverParameters(mos_length_um=0.1),
            ReceiverParameters(mos_multiplier=0),
            ReceiverParameters(mos_multiplier=1.5),
        ):
            with self.assertRaises(ValueError):
                parameters.validate()

    def test_spice_parameters_include_grouped_geometry(self):
        values = ReceiverParameters(mos_width_um=24, mos_length_um=0.3,
                                    mos_multiplier=4).spice_parameters(SimulationConditions())
        self.assertEqual(values["MOS_W"], 24)
        self.assertEqual(values["MOS_L"], 0.3)
        self.assertEqual(values["MOS_M"], 4)

    def test_every_receiver_bench_forwards_grouped_geometry(self):
        for path in Path(BENCHES).glob("*.cir"):
            source = path.read_text(encoding="utf-8")
            if "XCTLE" in source:
                self.assertIn("MOS_W={MOS_W}", source, path.name)
                self.assertIn("MOS_L={MOS_L}", source, path.name)
                self.assertIn("MOS_M={MOS_M}", source, path.name)

    def test_matched_pair_uses_one_shared_group(self):
        source = (Path(BENCHES).parent / "blocks" / "ctle.spice").read_text(encoding="utf-8")
        self.assertEqual(source.count("W={MOS_W} L={MOS_L} m={MOS_M}"), 2)

    def test_partial_channel_area_tracks_geometry_and_multiplicity(self):
        estimate = estimate_ctle_area(parameters=ReceiverParameters(
            mos_width_um=20, mos_length_um=0.3, mos_multiplier=4,
        ))
        self.assertAlmostEqual(estimate.transistor_channel_area_um2, 48.0)
        self.assertFalse(estimate.total_area_computable)

    def test_export_contains_selected_geometry(self):
        text = render_final_schematic(ReceiverParameters(
            mos_width_um=20, mos_length_um=0.3, mos_multiplier=4,
        ))
        self.assertIn("MOS_W=20", text)
        self.assertIn("MOS_L=0.3", text)
        self.assertIn("MOS_M=4", text)

    def test_ui_names_and_explains_all_three_ppo_versions(self):
        self.assertIn('value="v1"', INDEX_HTML)
        self.assertIn('value="v2"', INDEX_HTML)
        self.assertIn('value="v3"', INDEX_HTML)
        self.assertNotIn('value="v3" disabled', INDEX_HTML)
        self.assertIn("Eight-head PPO", INDEX_HTML)
        self.assertIn("matched MOS width, length and integer multiplier", INDEX_HTML)


if __name__ == "__main__":
    unittest.main()
