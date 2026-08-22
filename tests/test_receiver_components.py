from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from simulator.cache import EvaluationCache
from simulator.channel import filter_channel, load_s4p
from simulator.config import PVT_GRID, SimulationConditions, Sky130Config
from simulator.ngspice import NgSpiceConfig, run_simulation
from simulator.models import SimulationRequest
from simulator.receiver import (
    EvaluationFidelity, ReceiverParameters, _run_ac, _run_ctle_transient,
    _evaluation_is_cacheable, _run_channel_diagnostics, _run_dc, _run_hd3,
    _run_noise, _run_transient, _transient_violations, ReceiverEvaluation,
    StageResult, evaluate_receiver,
)
from simulator.receiver_metrics import (
    DFEMode, _contiguous_open_width, apply_hybrid_one_tap_dfe,
    apply_one_tap_dfe, dfe_eye_metrics,
)
from simulator.stimulus import NRZStimulusConfig, generate_bits, generate_nrz, stimulus_include
from simulator.waveform import Trace, integrated_input_noise, parse_wrdata


ROOT = Path(__file__).resolve().parents[1]
CHANNEL = ROOT / "channels" / "synthetic_regression.s4p"
WRDATA = ROOT / "circuits" / "test" / "wrdata.cir"


class StimulusTests(unittest.TestCase):
    def test_prbs7_has_127_bit_period(self):
        bits = generate_bits("prbs7", 254, seed=7)
        self.assertEqual(bits[:127], bits[127:])
        self.assertEqual(sum(bits[:127]), 64)

    def test_pwl_is_wrapped_and_uses_requested_compensation(self):
        stimulus = generate_nrz(NRZStimulusConfig(bit_count=8, warmup_bits=1, tail_bits=1))
        text = stimulus_include(stimulus.time_s, stimulus.differential_v, 1.2, 2.0)
        self.assertIn("VCHP chp_src 0 PWL(", text)
        self.assertTrue(all(len(line) < 120 for line in text.splitlines()))


class ChannelTests(unittest.TestCase):
    def test_synthetic_channel_parses_and_filters(self):
        channel = load_s4p(CHANNEL)
        self.assertAlmostEqual(channel.insertion_loss_db(2.5e9), -3.098, places=2)
        stimulus = generate_nrz(NRZStimulusConfig(bit_count=8, warmup_bits=1, tail_bits=1))
        output = filter_channel(channel, stimulus.time_s, stimulus.differential_v)
        self.assertEqual(len(output), len(stimulus.differential_v))
        self.assertTrue(np.isfinite(output).all())


class ReceiverMetricTests(unittest.TestCase):
    def test_dfe_training_and_decision_directed_modes(self):
        bits = np.asarray([0, 1, 1, 0, 1, 0])
        signs = np.where(bits > 0, 1.0, -1.0)
        previous = np.r_[-1.0, signs[:-1]]
        samples = 0.2 * signs + 0.06 * previous
        for mode in (DFEMode.TRAINING, DFEMode.DECISION_DIRECTED):
            result = apply_one_tap_dfe(samples, bits, 0.06, mode)
            self.assertEqual(result.error_count, 0)
            np.testing.assert_allclose(result.corrected_samples_v, 0.2 * signs)

    def test_dfe_switches_after_training_without_resetting_feedback_state(self):
        bits = np.asarray([1, 1, 0, 1])
        signs = np.where(bits > 0, 1.0, -1.0)
        previous = np.r_[-1.0, signs[:-1]]
        samples = 0.2 * signs + 0.06 * previous
        samples[0] = -0.4
        result = apply_hybrid_one_tap_dfe(samples, bits, 0.06, training_stop=2)
        self.assertEqual(result.decisions[1], bits[1])
        self.assertAlmostEqual(result.corrected_samples_v[1], 0.2)
        self.assertAlmostEqual(result.corrected_samples_v[2], -0.2)

    def test_eye_width_uses_one_contiguous_locked_region(self):
        mask = np.asarray([True, True, False, True, True, False, True])
        self.assertAlmostEqual(_contiguous_open_width(mask, 0, 0.1, 1.0), 0.3)

    def test_candidate_vertical_limit_uses_locked_phase_height(self):
        metrics = {
            "dfe_eye_height_v": 0.2,
            "dfe_locked_phase_eye_height_v": 0.08,
            "dfe_eye_width_ui": 0.5,
        }
        violations = _transient_violations(metrics, EvaluationFidelity.CANDIDATE)
        self.assertTrue(any("locked phase" in item for item in violations))

    def test_dfe_eye_width_is_measured_across_phase(self):
        config = NRZStimulusConfig(bit_count=32, warmup_bits=4, tail_bits=4)
        stimulus = generate_nrz(config)
        metrics = dfe_eye_metrics(
            stimulus.time_s, stimulus.differential_v, stimulus.bits, config.ui_s,
            config.time_step_s, 4, 28, 0.0,
        )
        self.assertGreater(metrics["dfe_eye_height_v"], 0.1)
        self.assertGreater(metrics["dfe_eye_width_ui"], 0.4)

    def test_noise_integration_supports_numpy_without_trapz(self):
        class ModernNumpy:
            trapezoid = staticmethod(np.trapezoid)

        trace = Trace("frequency", np.asarray([1.0, 2.0]), {
            "inoise_spectrum": np.asarray([2.0, 2.0]),
        })
        with patch("simulator.waveform.np", ModernNumpy):
            self.assertAlmostEqual(integrated_input_noise(trace), 2.0)


class ConfigurationAndCacheTests(unittest.TestCase):
    def test_full_pvt_grid_has_60_unique_conditions(self):
        self.assertEqual(len(PVT_GRID), 60)
        identities = {(c.process_corner, c.supply_v, c.temperature_c) for c in PVT_GRID}
        self.assertEqual(len(identities), 60)

    def test_cache_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = EvaluationCache(directory)
            cache.put("abcd", {"value": 3})
            self.assertEqual(cache.get("abcd")["value"], 3)
            path = cache._path("abcd")
            path.write_text('{"evaluation_id":"different"}', encoding="utf-8")
            self.assertIsNone(cache.get("abcd"))

    def test_retryable_failures_are_not_cacheable(self):
        evaluation = ReceiverEvaluation(
            False, ReceiverParameters(), SimulationConditions(),
            EvaluationFidelity.SCREENING,
            (StageResult("dc", False, 1.0, retryable=True),),
            {}, "dc", 1.0, "id", {},
        )
        self.assertFalse(_evaluation_is_cacheable(evaluation))

    def test_deterministic_failures_are_cacheable(self):
        evaluation = ReceiverEvaluation(
            False, ReceiverParameters(), SimulationConditions(),
            EvaluationFidelity.SCREENING,
            (StageResult("dc", False, 1.0),),
            {}, "dc", 1.0, "id", {},
        )
        self.assertTrue(_evaluation_is_cacheable(evaluation))

    def test_timeout_must_be_positive_and_finite(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                NgSpiceConfig(timeout_s=timeout)

    def test_hd3_preserves_retryable_simulator_failure(self):
        timeout = StageResult(
            "hd3", False, 1.0,
            failure_code="ngspice_timeout", retryable=True,
        )
        with patch("simulator.receiver._run_trace_stage", return_value=timeout):
            result = _run_hd3(
                ReceiverParameters(), SimulationConditions(), Path("model"),
                NgSpiceConfig(executable=Path(__file__)),
            )
        self.assertTrue(result.retryable)
        evaluation = ReceiverEvaluation(
            False, ReceiverParameters(), SimulationConditions(),
            EvaluationFidelity.CANDIDATE, (result,), {}, "hd3", 1.0, "id", {},
        )
        self.assertFalse(_evaluation_is_cacheable(evaluation))


class FailureContainmentTests(unittest.TestCase):
    def test_empty_channel_returns_structured_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            channel = Path(directory) / "empty.s4p"
            channel.write_text("", encoding="utf-8")
            result = _run_channel_diagnostics(channel)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, "channel_error")
        self.assertTrue(result.errors)

    def test_unexpected_runner_exception_becomes_structured_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.spice"
            model.write_text("* model", encoding="utf-8")
            cache = EvaluationCache(Path(directory) / "cache")
            with patch("simulator.receiver.ngspice_identity", return_value={}), \
                    patch("simulator.receiver._run_dc", side_effect=RuntimeError("unexpected")), \
                    patch.object(cache, "put", wraps=cache.put) as put:
                result = evaluate_receiver(
                    ReceiverParameters(), SimulationConditions(),
                    EvaluationFidelity.SCREENING,
                    sky130=Sky130Config(model),
                    ngspice=NgSpiceConfig(executable=Path(__file__)),
                    cache=cache,
                )
                put.assert_not_called()
        self.assertFalse(result.success)
        self.assertEqual(result.failed_stage, "internal")
        self.assertEqual(result.stages[-1].failure_code, "internal_error")
        self.assertIn("RuntimeError: unexpected", result.stages[-1].errors)

    def test_retryable_stage_is_not_written_to_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.spice"
            model.write_text("* model", encoding="utf-8")
            cache = EvaluationCache(Path(directory) / "cache")
            retryable = StageResult(
                "dc", False, 1.0, failure_code="ngspice_timeout", retryable=True,
            )
            with patch("simulator.receiver.ngspice_identity", return_value={}), \
                    patch("simulator.receiver._run_dc", return_value=retryable), \
                    patch.object(cache, "put", wraps=cache.put) as put:
                result = evaluate_receiver(
                    ReceiverParameters(), SimulationConditions(),
                    EvaluationFidelity.SCREENING,
                    sky130=Sky130Config(model),
                    ngspice=NgSpiceConfig(executable=Path(__file__)),
                    cache=cache,
                )
                put.assert_not_called()
        self.assertFalse(result.success)

    def test_raw_process_corner_returns_structured_setup_failure(self):
        conditions = SimulationConditions(process_corner="tt")
        result = evaluate_receiver(ReceiverParameters(), conditions)
        self.assertFalse(result.success)
        self.assertEqual(result.failed_stage, "setup")
        self.assertEqual(result.stages[0].failure_code, "invalid_parameter")
        self.assertEqual(result.to_dict()["conditions"]["process_corner"], "tt")

    def test_corrupt_identity_matching_cache_entry_is_a_cache_miss(self):
        class CorruptCache:
            def __init__(self):
                self.put_count = 0

            def get(self, evaluation_id):
                return {"evaluation_id": evaluation_id, "success": True}

            def put(self, evaluation_id, value):
                self.put_count += 1

        cache = CorruptCache()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.spice"
            model.write_text("* model", encoding="utf-8")
            with patch("simulator.receiver.ngspice_identity", return_value={}), \
                    patch("simulator.receiver._run_dc", return_value=StageResult("dc", False, 0.0)):
                result = evaluate_receiver(
                    ReceiverParameters(), SimulationConditions(),
                    EvaluationFidelity.SCREENING,
                    sky130=Sky130Config(model),
                    ngspice=NgSpiceConfig(executable=Path(__file__)),
                    cache=cache,
                )
        self.assertFalse(result.cache_hit)
        self.assertEqual(result.failed_stage, "dc")
        self.assertEqual(cache.put_count, 1)

    def test_final_default_timeout_is_180_seconds(self):
        observed = []
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.spice"
            model.write_text("* model", encoding="utf-8")

            def identity(config):
                observed.append(config.timeout_s)
                return {}

            with patch("simulator.receiver.ngspice_identity", side_effect=identity), \
                    patch("simulator.receiver._run_dc", return_value=StageResult("dc", False, 0.0)):
                evaluate_receiver(
                    ReceiverParameters(), SimulationConditions(),
                    EvaluationFidelity.FINAL,
                    sky130=Sky130Config(model),
                )
        self.assertEqual(observed, [180.0])


class WrdataIntegrationTest(unittest.TestCase):
    def test_artifact_round_trip(self):
        try:
            NgSpiceConfig().resolve_executable()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        result = run_simulation(SimulationRequest(
            WRDATA, output_files={"OUTPUT_FILE": "trace.dat"},
        ))
        self.assertTrue(result.success, result.errors)
        trace = parse_wrdata(result.artifacts["OUTPUT_FILE"], ("v(in)", "v(out)"))
        self.assertGreater(len(trace.scale), 10)
        self.assertGreater(float(np.max(trace.column("v(out)"))), 0.5)


class MinimalBenchIntegrationTest(unittest.TestCase):
    def test_all_waveform_benches_execute_without_sky130_installation(self):
        try:
            NgSpiceConfig().resolve_executable()
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        model_text = """.lib tt
.subckt sky130_fd_pr__nfet_01v8 d g s b W=1 L=1
Gm d s value={1m*v(g,s)}
Rds d s 100k
.ends sky130_fd_pr__nfet_01v8
.endl tt
"""
        parameters = ReceiverParameters()
        conditions = SimulationConditions()
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "minimal_sky130.spice"
            model.write_text(model_text, encoding="utf-8")
            stages = (
                _run_dc(parameters, conditions, model, None),
                _run_ac(parameters, conditions, model, None),
                _run_ctle_transient(parameters, conditions, model, None),
                _run_noise(parameters, conditions, model, None),
                _run_hd3(parameters, conditions, model, None),
                _run_transient(parameters, conditions, model, None,
                               EvaluationFidelity.TRAINING, CHANNEL),
            )
        for stage in stages:
            self.assertNotEqual(stage.failure_code, "waveform_parse_error", stage)
            self.assertFalse(stage.errors, stage)
        transient = next(stage for stage in stages if stage.name == "transient")
        self.assertLess(transient.metrics["dfe_error_rate"], 0.5)
        self.assertGreater(transient.metrics["dfe_eye_height_v"], 0.0)


if __name__ == "__main__":
    unittest.main()
