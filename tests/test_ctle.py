from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.records import append_jsonl, flatten_evaluation, write_csv
from experiments.sweep import grid_parameters, random_parameters
from simulator.config import Sky130Config
from simulator.ctle import CTLEEvaluation, evaluate_ctle, spice_number, validate_ctle_parameters
from simulator.metrics import ctle_constraints, derive_ctle_metrics
from simulator.models import FailureCode, SimulationResult, SimulationStatus
from simulator.ngspice import render_template


BASELINE = {"RLOAD": "1k", "RDEG": "1k", "CDEG": "0.5p", "ITAIL_VAL": "100u"}


class CTLEParameterTests(unittest.TestCase):
    def test_parses_spice_suffixes(self):
        self.assertEqual(spice_number("1k"), 1000)
        self.assertAlmostEqual(spice_number("0.5p"), 0.5e-12)
        self.assertAlmostEqual(spice_number("100u"), 100e-6)

    def test_validates_complete_bounded_parameter_set(self):
        validate_ctle_parameters(BASELINE)
        with self.assertRaises(ValueError):
            validate_ctle_parameters({**BASELINE, "RDEG": -1})
        with self.assertRaises(ValueError):
            validate_ctle_parameters({"RLOAD": "1k"})

    def test_template_requires_all_explicit_values(self):
        self.assertEqual(render_template('.lib "@@MODEL@@" tt', {"MODEL": "x/y"}), '.lib "x/y" tt')
        with self.assertRaises(KeyError):
            render_template("@@MODEL@@", {})

    def test_missing_model_has_specific_failure(self):
        result = evaluate_ctle(BASELINE, sky130=Sky130Config("missing-model.spice"))
        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, FailureCode.MODEL_LIBRARY_NOT_FOUND.value)


class CTLEMetricTests(unittest.TestCase):
    def test_derives_baseline_metrics_and_constraints(self):
        measurements = {
            "gain_1mhz_mag": 10 ** (-7.18 / 20), "gain_2p5ghz_mag": 10 ** (-1.47 / 20),
            "gain_5ghz_mag": 10 ** (-0.5 / 20), "gain_100ghz_mag": 10 ** (1.37 / 20),
            "outp_dc_v": 1.75, "outn_dc_v": 1.75,
            "vdd_current_a": -100e-6,
        }
        metrics = derive_ctle_metrics(measurements)
        self.assertAlmostEqual(metrics["peaking_2p5ghz_db"], 5.71)
        self.assertAlmostEqual(metrics["power_w"], 180e-6)
        self.assertTrue(ctle_constraints(metrics)[0])

    @patch("simulator.ctle.run_simulation")
    def test_evaluator_returns_derived_metrics(self, run):
        run.return_value = SimulationResult(
            SimulationStatus.SUCCESS,
            {"gain_1mhz_mag": 10 ** (-7.18 / 20), "gain_2p5ghz_mag": 10 ** (-1.47 / 20),
             "gain_5ghz_mag": 10 ** (-0.5 / 20), "gain_100ghz_mag": 10 ** (1.37 / 20),
             "outp_dc_v": 1.75, "outn_dc_v": 1.75, "vdd_current_a": -100e-6},
            0.2,
        )
        with tempfile.NamedTemporaryFile() as model:
            result = evaluate_ctle(BASELINE, sky130=Sky130Config(model.name))
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.metrics["peaking_2p5ghz_db"], 5.71)

    def test_automation_netlist_runs_with_minimal_model_library(self):
        model_text = """.lib tt
.subckt sky130_fd_pr__nfet_01v8 d g s b W=1 L=1
Gm d s value={1m*v(g,s)}
Rds d s 100k
.ends sky130_fd_pr__nfet_01v8
.endl tt
"""
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "minimal_sky130.spice"
            model.write_text(model_text, encoding="utf-8")
            result = evaluate_ctle(BASELINE, sky130=Sky130Config(model))
        self.assertTrue(result.success, result.errors)
        self.assertEqual(set(result.measurements), {
            "gain_1mhz_mag", "gain_2p5ghz_mag", "gain_5ghz_mag",
            "gain_100ghz_mag", "outp_dc_v", "outn_dc_v", "vdd_current_a",
        })


class ExperimentTests(unittest.TestCase):
    def test_grid_and_random_generators_are_reproducible(self):
        grid = list(grid_parameters({"a": (1, 2), "b": (3, 4)}))
        self.assertEqual(len(grid), 4)
        self.assertEqual(list(random_parameters(2, 7)), list(random_parameters(2, 7)))

    def test_flattens_evaluation(self):
        evaluation = CTLEEvaluation(BASELINE, True, {"gain": 1}, {"power_w": 1e-3}, True, (), 0.1)
        row = flatten_evaluation(evaluation, 3)
        self.assertEqual(row["run_id"], 3)
        self.assertEqual(row["RLOAD"], "1k")
        self.assertEqual(row["power_w"], 1e-3)

    def test_writes_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            jsonl = Path(directory) / "runs.jsonl"
            csv_path = Path(directory) / "runs.csv"
            rows = [{"run_id": 1, "success": True}, {"run_id": 2, "success": False}]
            for row in rows:
                append_jsonl(jsonl, row)
            write_csv(csv_path, rows)
            self.assertEqual(len(jsonl.read_text(encoding="utf-8").splitlines()), 2)
            self.assertIn("run_id", csv_path.read_text(encoding="utf-8"))
