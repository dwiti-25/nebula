from pathlib import Path
import os
import stat
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from simulator import FailureCode, FailureStage, SimulationRequest, SimulationStatus
from simulator.ngspice import NgSpiceConfig, parameterize_netlist, run_ngspice, run_simulation


ROOT = Path(__file__).resolve().parents[1]
DIVIDER = ROOT / "circuits" / "test" / "divider.cir"


class ParameterizationTests(unittest.TestCase):
    def test_replaces_declared_parameter_only(self):
        source = ".param UNUSED=2 RVAL=1k\nR1 in out {RVAL}\n"
        rendered = parameterize_netlist(source, {"RVAL": 3000})
        self.assertIn("RVAL=3000", rendered)
        self.assertIn("UNUSED=2", rendered)
        self.assertIn("{RVAL}", rendered)

    def test_rejects_unknown_parameter(self):
        with self.assertRaises(KeyError):
            parameterize_netlist(".param RVAL=1k\n", {"OTHER": 3})


class WrapperTests(unittest.TestCase):
    @patch("simulator.ngspice.subprocess.run")
    def test_returns_typed_success_for_expected_measurement(self, run):
        run.return_value = subprocess.CompletedProcess(
            [], 0, stdout="vout = 2.5e-1\n", stderr=""
        )
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "ngspice"
            fake.write_text("fake", encoding="utf-8")
            result = run_ngspice(
                DIVIDER, {"RVAL": 3000}, expected_measurements=("vout",),
                config=NgSpiceConfig(fake),
            )
        self.assertTrue(result.success)
        self.assertEqual(result.status, SimulationStatus.SUCCESS)
        self.assertEqual(result.measurements, {"vout": 0.25})
        self.assertIsNone(result.failure_code)
        self.assertEqual(result.to_dict()["status"], "success")

    @unittest.skipIf(os.name == "nt", "POSIX fake executable test")
    def test_runs_in_temp_directory_and_parses_log(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "fake-ngspice"
            fake.write_text("#!/bin/sh\nprintf 'vout = 2.5e-1\\n' > \"$3\"\n", encoding="utf-8")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            result = run_ngspice(
                DIVIDER, {"RVAL": 3000}, expected_measurements=("vout",),
                config=NgSpiceConfig(fake),
            )
        self.assertTrue(result.success, result)
        self.assertEqual(result.measurements["vout"], 0.25)

    def test_missing_executable_is_structured_failure(self):
        missing = ROOT / "does-not-exist" / "ngspice"
        result = run_ngspice(DIVIDER, {"RVAL": 3000}, config=NgSpiceConfig(missing))
        self.assertFalse(result.success)
        self.assertEqual(result.status, SimulationStatus.FAILED)
        self.assertEqual(result.failure_stage, FailureStage.SETUP)
        self.assertEqual(result.failure_code, FailureCode.NGSPICE_NOT_FOUND)
        self.assertFalse(result.retryable)
        self.assertIn("does not exist", result.stderr)

    def test_invalid_parameter_has_specific_failure_code(self):
        result = run_ngspice(DIVIDER, {"UNKNOWN": 3000})
        self.assertEqual(result.failure_code, FailureCode.INVALID_PARAMETER)
        self.assertEqual(result.failure_stage, FailureStage.SETUP)

    @patch("simulator.ngspice.subprocess.run")
    def test_missing_expected_measurement_is_parsing_failure(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout="other = 1\n", stderr="")
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "ngspice"
            fake.write_text("fake", encoding="utf-8")
            request = SimulationRequest(
                DIVIDER, {"RVAL": 3000}, expected_measurements=("vout",)
            )
            result = run_simulation(request, config=NgSpiceConfig(fake))
        self.assertEqual(result.failure_code, FailureCode.MEASUREMENT_MISSING)
        self.assertEqual(result.failure_stage, FailureStage.PARSING)

    @patch("simulator.ngspice.subprocess.run")
    def test_timeout_is_retryable_execution_failure(self, run):
        run.side_effect = subprocess.TimeoutExpired("ngspice", 0.01, output="partial")
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "ngspice"
            fake.write_text("fake", encoding="utf-8")
            result = run_ngspice(
                DIVIDER, {"RVAL": 3000},
                config=NgSpiceConfig(fake, timeout_s=0.01),
            )
        self.assertEqual(result.failure_code, FailureCode.NGSPICE_TIMEOUT)
        self.assertEqual(result.failure_stage, FailureStage.EXECUTION)
        self.assertTrue(result.retryable)


class RealNgSpiceIntegrationTest(unittest.TestCase):
    def test_divider(self):
        try:
            NgSpiceConfig().resolve_executable()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        result = run_ngspice(
            DIVIDER, {"RVAL": 3000}, expected_measurements=("vout",)
        )
        self.assertTrue(result.success, result)
        self.assertAlmostEqual(result.measurements["vout"], 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
