import os
from pathlib import Path
import unittest

from simulator import evaluate_ctle
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
