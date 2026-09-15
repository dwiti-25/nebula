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
    def test_1024_corner_selection_does_not_reuse_512_corner_measurements(self):
        from simulator.receiver import EvaluationFidelity
        from simulator.config import SimulationConditions
        base = synthetic_evaluate_receiver_graded(ReceiverParameters())
        metrics = {**TargetSpec.from_existing_thresholds().as_dict(), "peaking_db": 6,
                   "dfe_error_count": 0, "hd3_db": -40, "input_referred_noise_vrms": 0.0005}
        calls = []
        def evaluate(parameters, condition, fidelity, **kwargs):
            calls.append(fidelity)
            return replace(base, parameters=parameters, conditions=condition, fidelity=fidelity,
                           success=True, stages=(), metrics=metrics)
        data = {"saved_candidates": [{"design_id": "x", "parameters": {"rload_ohm": 1000}}]}
        with TemporaryDirectory() as tmp:
            first = run_sequence(data, output=Path(tmp)/"512.json", cache=None, evaluator=evaluate,
                                 conditions=(SimulationConditions(temperature_c=125),), runtime_identity="same")
            calls.clear()
            second = run_sequence(first, output=Path(tmp)/"1024.json", cache=None, evaluator=evaluate,
                                  conditions=(SimulationConditions(temperature_c=125),), runtime_identity="same",
                                  pvt_pattern_bits=1024)
            self.assertEqual(calls, [EvaluationFidelity.FINAL])
            self.assertEqual(second["pvt_pattern_bits"], 1024)
            self.assertEqual(second["pvt_fidelity"], "FINAL")
            self.assertTrue(all(r.get("reused") for r in second["candidates"][0]["nominal"]))
            self.assertNotIn("reused", second["candidates"][0]["pvt"][0])
            with self.assertRaises(ValueError):
                run_sequence(data, output=Path(tmp)/"bad.json", cache=None, pvt_pattern_bits=256)

    def test_restart_reuses_completed_measurements_but_retries_timeout(self):
        from simulator.receiver import EvaluationFidelity, StageResult
        from simulator.config import SimulationConditions
        base = synthetic_evaluate_receiver_graded(ReceiverParameters())
        metrics = {**TargetSpec.from_existing_thresholds().as_dict(), "peaking_db": 6,
                   "dfe_error_count": 0, "hd3_db": -40, "input_referred_noise_vrms": 0.0005}
        conditions = (SimulationConditions(temperature_c=0), SimulationConditions(temperature_c=125))
        calls = []
        def evaluate(parameters, condition, fidelity, **kwargs):
            calls.append((condition, fidelity, kwargs["ngspice"].timeout_s))
            timeout = condition.temperature_c == 125 and kwargs["ngspice"].timeout_s == 180
            stages = (StageResult("transient", False, 180, retryable=True, failure_code="ngspice_timeout"),) if timeout else ()
            return replace(base, parameters=parameters, conditions=condition, fidelity=fidelity,
                           success=not timeout, stages=stages, metrics={} if timeout else metrics)
        data = {"saved_candidates": [{"design_id": "x", "parameters": {"rload_ohm": 1000}}]}
        with TemporaryDirectory() as tmp:
            first = run_sequence(data, output=Path(tmp)/"first.json", cache=None, evaluator=evaluate,
                                 conditions=conditions, timeout_s=180, runtime_identity="same")
            self.assertEqual(first["candidates"][0]["status"], "incomplete_execution")
            original = (Path(tmp)/"first.json").read_bytes()
            calls.clear()
            second = run_sequence(first, output=Path(tmp)/"second.json", cache=None, evaluator=evaluate,
                                  conditions=conditions, timeout_s=360, runtime_identity="same")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1:], (EvaluationFidelity.CANDIDATE, 360))
            self.assertEqual(second["qualified_designs"], ["x"])
            self.assertEqual((Path(tmp)/"first.json").read_bytes(), original)
            calls.clear()
            run_sequence(second, output=Path(tmp)/"third.json", cache=None, evaluator=evaluate,
                         conditions=conditions, runtime_identity="changed")
            self.assertEqual(len(calls), 4)

    def test_invalid_timeout_and_existing_output_fail_before_evaluation(self):
        from unittest.mock import Mock
        evaluator = Mock()
        data = {"saved_candidates": [{"parameters": {"rload_ohm": 1000}}]}
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/"existing.json"
            output.write_text("original")
            for timeout in (0, -1, float("nan"), float("inf"), True):
                with self.assertRaises((TypeError, ValueError)):
                    run_sequence(data, output=output, cache=None, evaluator=evaluator, timeout_s=timeout)
            with self.assertRaises(FileExistsError):
                run_sequence(data, output=output, cache=None, evaluator=evaluator)
        evaluator.assert_not_called()

    def test_recovers_failed_ui_candidates_and_saved_conditions(self):
        from experiments.qualification_sequence import load_restart_input
        from simulator.config import SimulationConditions
        from simulator.receiver import SYNTHETIC_CHANNEL
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"run.json"
            path.write_text(json.dumps({"backend": "real", "channel": {"path": str(SYNTHETIC_CHANNEL)},
                "selection": {"selected": None}, "execution_graph": {"nodes": [{"kind": "filter", "detail": {
                    "accepted_episodes": [2], "candidates": [
                        {"episode": 1, "parameters": {"rload_ohm": 500}},
                        {"episode": 2, "parameters": {"rload_ohm": 1000}}]}}]}}))
            path.with_suffix(".events.jsonl").write_text(json.dumps({"phase": "pvt", "conditions": SimulationConditions().to_dict()}))
            data, conditions = load_restart_input(path)
            self.assertEqual([c["design_id"] for c in data["saved_candidates"]], ["pipeline_ep2"])
            self.assertEqual(len(conditions), 1)

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
