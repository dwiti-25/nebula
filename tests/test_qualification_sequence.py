import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from dataclasses import replace
from experiments.qualification_sequence import run_sequence
from rl.synthetic_benchmark import synthetic_evaluate_receiver_graded
from rl.target_spec import TargetSpec
from simulator.receiver import ReceiverParameters
from simulator.config import QUALIFICATION_PVT_GRID


class QualificationTests(unittest.TestCase):
    def test_timeout_identity_is_canonical(self):
        from simulator.ngspice import NgSpiceConfig
        from simulator.provenance import stable_fingerprint
        from dataclasses import asdict
        self.assertEqual(stable_fingerprint(asdict(NgSpiceConfig(timeout_s=180))),
                         stable_fingerprint(asdict(NgSpiceConfig(timeout_s=180.0))))
    def test_agreed_grid(self):
        self.assertEqual(len(QUALIFICATION_PVT_GRID), 36)
        self.assertEqual({c.process_corner.value for c in QUALIFICATION_PVT_GRID}, {"tt", "ss", "ff"})

    def run_case(self, nominal_ok):
        calls = []
        base = synthetic_evaluate_receiver_graded(ReceiverParameters())
        metrics = {**TargetSpec.from_existing_thresholds().as_dict(), "peaking_db": 6,
                   "dfe_error_count": 0, "hd3_db": -40, "input_referred_noise_vrms": 0.0005}
        def evaluate(parameters, conditions, fidelity, **kwargs):
            calls.append(conditions)
            return replace(base, parameters=parameters, conditions=conditions, fidelity=fidelity,
                           success=nominal_ok, metrics=metrics if nominal_ok else {})
        with TemporaryDirectory() as tmp:
            result = run_sequence({"selection": {"selected": {"design_id": "x", "parameters": {"rload_ohm": 1000}}}},
                                  output=Path(tmp)/"report.json", cache=None, evaluator=evaluate)
        return result, calls

    def test_nominal_failure_never_spends_pvt(self):
        result, calls = self.run_case(False)
        self.assertEqual(len(calls), 1)
        self.assertFalse(result["qualified_designs"])

    def test_nominal_pass_unlocks_only_agreed_grid(self):
        result, calls = self.run_case(True)
        self.assertEqual(len(calls), 38)
        self.assertEqual(result["qualified_designs"], ["x"])
        self.assertEqual(result["candidates"][0]["specification"]["rows"][-1]["verdict"], "PASS")
