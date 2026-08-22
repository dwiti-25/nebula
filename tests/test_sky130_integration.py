import os
from pathlib import Path
import unittest

from simulator import (
    EvaluationFidelity, NgSpiceConfig, ReceiverParameters,
    SimulationConditions, evaluate_ctle, evaluate_receiver,
)
from simulator.config import SKY130_MODEL_ENV, Sky130Config


BASELINE = {"RLOAD": "1k", "RDEG": "1k", "CDEG": "0.5p", "ITAIL_VAL": "100u"}
MODEL = os.environ.get(SKY130_MODEL_ENV)


@unittest.skipUnless(MODEL and Path(MODEL).is_file(), f"set {SKY130_MODEL_ENV} to run SKY130 integration")
class Sky130IntegrationTest(unittest.TestCase):
    def test_a_baseline(self):
        result = evaluate_ctle(BASELINE, sky130=Sky130Config(MODEL))
        self.assertTrue(result.success, result.errors)
        self.assertAlmostEqual(result.metrics["gain_1mhz_db"], -7.18, delta=1.0)
        self.assertAlmostEqual(result.metrics["gain_2p5ghz_db"], -1.47, delta=1.0)
        self.assertAlmostEqual(result.metrics["peaking_2p5ghz_db"], 5.71, delta=1.0)
        self.assertAlmostEqual(result.metrics["gain_100ghz_db"], 1.37, delta=1.5)

    def test_hierarchical_candidate_baseline(self):
        result = evaluate_receiver(
            ReceiverParameters(), SimulationConditions(),
            EvaluationFidelity.CANDIDATE,
            sky130=Sky130Config(MODEL),
            ngspice=NgSpiceConfig(timeout_s=180),
            cache=None,
        )
        self.assertTrue(result.success, result.to_dict())
        self.assertEqual(
            [stage.name for stage in result.stages],
            ["dc", "ac", "ctle_transient", "noise", "hd3", "channel", "transient"],
        )
        self.assertGreaterEqual(result.metrics["peaking_db"], 3.0)
        self.assertLessEqual(result.metrics["peaking_db"], 12.0)
        self.assertGreaterEqual(result.metrics["peak_frequency_hz"], 1.25e9)
        self.assertLessEqual(result.metrics["peak_frequency_hz"], 2.5e9)
        self.assertLess(result.metrics["input_referred_noise_vrms"], 1.5e-3)
        self.assertLess(result.metrics["hd3_db"], -30.0)
        self.assertGreater(result.metrics["dfe_locked_phase_eye_height_v"], 0.1)
        self.assertGreater(result.metrics["dfe_eye_width_ui"], 0.4)
