"""Hierarchical Stage 1 receiver evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
import math
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

from .channel import filter_channel, load_s4p
from .cache import EvaluationCache
from .config import PVT_GRID, ProcessCorner, SimulationConditions, Sky130Config
from .ctle import spice_number, validate_ctle_parameters
from .models import FailureCode, ParameterValue, SimulationRequest, SimulationResult
from .ngspice import NgSpiceConfig, ngspice_identity, run_simulation
from .provenance import git_identity, sha256_file, stable_fingerprint
from .receiver_metrics import (
    apply_hybrid_one_tap_dfe, choose_sampling_phase, dfe_eye_metrics,
    eye_metrics, expected_signs,
)
from .stimulus import NRZStimulusConfig, generate_nrz, stimulus_include
from .waveform import ac_metrics, hd3_db, integrated_input_noise, parse_wrdata, require_numpy


ROOT = Path(__file__).resolve().parents[1]
BLOCK = ROOT / "circuits" / "blocks" / "ctle.spice"
BENCHES = ROOT / "circuits" / "benches"
SYNTHETIC_CHANNEL = ROOT / "channels" / "synthetic_regression.s4p"

DC_MEASUREMENTS = (
    "inp_dc_v", "inn_dc_v", "outp_dc_v", "outn_dc_v",
    "srcp_dc_v", "srcn_dc_v", "tail_dc_v", "vdd_current_a",
)


class EvaluationFidelity(IntEnum):
    SCREENING = 1
    TRAINING = 2
    CANDIDATE = 3
    FINAL = 4


@dataclass(frozen=True)
class ReceiverParameters:
    rload_ohm: float = 1000.0
    rdeg_ohm: float = 1000.0
    cdeg_f: float = 0.5e-12
    itail_a: float = 100e-6
    dfe_tap_v: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, ParameterValue]) -> "ReceiverParameters":
        aliases = {
            "RLOAD": "rload_ohm", "RDEG": "rdeg_ohm", "CDEG": "cdeg_f",
            "ITAIL_VAL": "itail_a", "DFE_TAP": "dfe_tap_v",
        }
        normalized = {aliases.get(name, name.lower()): spice_number(value) for name, value in values.items()}
        return cls(**normalized)

    def spice_parameters(self, conditions: SimulationConditions) -> dict[str, float]:
        return {
            "RLOAD": self.rload_ohm,
            "RDEG": self.rdeg_ohm,
            "CDEG": self.cdeg_f,
            "ITAIL_VAL": self.itail_a,
            "VDD_VAL": conditions.supply_v,
            "VIN_CM": conditions.input_common_mode_v,
            "CLOAD": conditions.output_load_f,
        }

    def validate(self) -> None:
        validate_ctle_parameters({
            "RLOAD": self.rload_ohm, "RDEG": self.rdeg_ohm,
            "CDEG": self.cdeg_f, "ITAIL_VAL": self.itail_a,
        })
        if not -0.4 <= self.dfe_tap_v <= 0.4:
            raise ValueError("DFE tap must be between -0.4 V and 0.4 V")


@dataclass(frozen=True)
class StageResult:
    name: str
    success: bool
    runtime_s: float
    measurements: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    violations: tuple[str, ...] = ()
    failure_code: str | None = None
    errors: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    retryable: bool = False


@dataclass(frozen=True)
class ReceiverEvaluation:
    success: bool
    parameters: ReceiverParameters
    conditions: SimulationConditions
    fidelity: EvaluationFidelity
    stages: tuple[StageResult, ...]
    metrics: dict[str, float]
    failed_stage: str | None
    runtime_s: float
    evaluation_id: str
    provenance: dict[str, object]
    cache_hit: bool = False

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["fidelity"] = self.fidelity.name.lower()
        value["conditions"]["process_corner"] = (
            self.conditions.process_corner.value
            if isinstance(self.conditions.process_corner, ProcessCorner)
            else str(self.conditions.process_corner)
        )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ReceiverEvaluation":
        conditions_value = dict(value["conditions"])
        conditions_value["process_corner"] = ProcessCorner(conditions_value["process_corner"])
        return cls(
            bool(value["success"]), ReceiverParameters(**value["parameters"]),
            SimulationConditions(**conditions_value),
            EvaluationFidelity[str(value["fidelity"]).upper()],
            tuple(StageResult(**stage) for stage in value["stages"]),
            dict(value["metrics"]), value.get("failed_stage"), float(value["runtime_s"]),
            str(value["evaluation_id"]), dict(value["provenance"]), True,
        )


def _templates(model: Path, conditions: SimulationConditions) -> dict[str, str]:
    return {
        "SKY130_MODEL_LIBRARY": model.as_posix(),
        "PROCESS_CORNER": conditions.process_corner.value,
        "CTLE_BLOCK_FILE": BLOCK.as_posix(),
        "TEMPERATURE_C": format(conditions.temperature_c, ".15g"),
    }


def _stage_from_simulation(name: str, result: SimulationResult) -> StageResult:
    retryable_codes = {
        FailureCode.NGSPICE_TIMEOUT,
        FailureCode.NGSPICE_PROCESS_FAILURE,
        FailureCode.PROCESS_LAUNCH_ERROR,
        FailureCode.CONVERGENCE_FAILURE,
    }
    return StageResult(
        name, result.success, result.runtime_s, result.measurements, {}, (),
        result.failure_code.value if result.failure_code else None,
        result.errors, result.provenance, (),
        result.retryable or result.failure_code in retryable_codes,
    )


def _dc_metrics(measurements: Mapping[str, float], parameters: ReceiverParameters, conditions: SimulationConditions):
    outp = measurements["outp_dc_v"]
    outn = measurements["outn_dc_v"]
    srcp = measurements["srcp_dc_v"]
    srcn = measurements["srcn_dc_v"]
    branch_p = (conditions.supply_v - outp) / parameters.rload_ohm
    branch_n = (conditions.supply_v - outn) / parameters.rload_ohm
    metrics = {
        "output_common_mode_v": (outp + outn) / 2,
        "output_offset_v": outp - outn,
        "source_mismatch_v": abs(srcp - srcn),
        "vds_p_v": outp - srcp,
        "vds_n_v": outn - srcn,
        "branch_current_p_a": branch_p,
        "branch_current_n_a": branch_n,
        "branch_current_sum_a": branch_p + branch_n,
        "ctle_power_w": abs(measurements["vdd_current_a"]) * conditions.supply_v,
    }
    violations: list[str] = []
    warnings: list[str] = []
    if not 0 < outp < conditions.supply_v or not 0 < outn < conditions.supply_v:
        violations.append("CTLE output lies outside the supply rails")
    if metrics["vds_p_v"] <= 0 or metrics["vds_n_v"] <= 0:
        violations.append("an input transistor has non-positive VDS")
    if abs(metrics["output_offset_v"]) > 0.05:
        violations.append("DC differential output offset exceeds 50 mV")
    if abs(metrics["branch_current_sum_a"] - parameters.itail_a) > max(parameters.itail_a * 0.1, 1e-9):
        violations.append("load branch currents do not match tail current within 10 percent")
    if metrics["ctle_power_w"] <= 0 or metrics["ctle_power_w"] >= 15e-3:
        violations.append("Stage 1 CTLE power is outside (0, 15 mW)")
    if not 0.2 <= metrics["output_common_mode_v"] <= conditions.supply_v - 0.2:
        warnings.append("output common mode has less than 200 mV rail headroom")
    if metrics["source_mismatch_v"] > 0.01:
        warnings.append("source-node mismatch exceeds 10 mV")
    return metrics, tuple(violations), tuple(warnings)


def _run_dc(parameters, conditions, model, ngspice) -> StageResult:
    request = SimulationRequest(
        BENCHES / "ctle_dc.cir", parameters.spice_parameters(conditions),
        DC_MEASUREMENTS, _templates(model, conditions),
    )
    result = run_simulation(request, config=ngspice)
    stage = _stage_from_simulation("dc", result)
    if not result.success:
        return stage
    metrics, violations, warnings = _dc_metrics(result.measurements, parameters, conditions)
    return StageResult("dc", not violations, result.runtime_s, result.measurements, metrics, violations,
                       None if not violations else FailureCode.SIMULATION_ERROR.value, (), result.provenance,
                       warnings)


def _run_trace_stage(
    name, bench, artifact_marker, artifact_filename, parameters, conditions,
    model, ngspice, metric_function, parameter_overrides: Mapping[str, ParameterValue] | None = None,
) -> StageResult:
    values = parameters.spice_parameters(conditions)
    values.update(parameter_overrides or {})
    request = SimulationRequest(
        BENCHES / bench,
        values, (), _templates(model, conditions),
        {artifact_marker: artifact_filename},
    )
    result = run_simulation(request, config=ngspice)
    if not result.success:
        return _stage_from_simulation(name, result)
    try:
        trace = parse_wrdata(result.artifacts[artifact_marker])
        metrics = metric_function(trace)
    except (KeyError, ValueError, RuntimeError) as exc:
        return StageResult(name, False, result.runtime_s, failure_code=FailureCode.WAVEFORM_PARSE_ERROR.value,
                           errors=(str(exc),), provenance=result.provenance)
    return StageResult(name, True, result.runtime_s, metrics=metrics, provenance=result.provenance)


def _run_ac(parameters, conditions, model, ngspice) -> StageResult:
    stage = _run_trace_stage("ac", "ctle_ac.cir", "AC_OUTPUT_FILE", "ac.dat",
                             parameters, conditions, model, ngspice, ac_metrics)
    if not stage.success:
        return stage
    violations: list[str] = []
    if not 3 <= stage.metrics["peaking_db"] <= 12:
        violations.append("CTLE peaking is outside 3 dB to 12 dB")
    if stage.metrics["out_of_band_excess_peak_db"] > 3:
        violations.append("an out-of-band peak exceeds the intended peak by more than 3 dB")
    warnings = ()
    if stage.metrics["group_delay_span_s"] > 40e-12:
        warnings = ("group-delay span exceeds 0.2 UI over 100 MHz to 5 GHz",)
    return StageResult("ac", not violations, stage.runtime_s, metrics=stage.metrics,
                       violations=tuple(violations),
                       failure_code=None if not violations else FailureCode.SIMULATION_ERROR.value,
                       provenance=stage.provenance, warnings=warnings)


def _run_noise(parameters, conditions, model, ngspice) -> StageResult:
    def metrics(trace):
        value = integrated_input_noise(trace)
        return {"input_referred_noise_vrms": value}
    sparse = NgSpiceConfig(
        executable=ngspice.executable if ngspice else None,
        timeout_s=ngspice.timeout_s if ngspice else NgSpiceConfig().timeout_s,
        solver="sparse",
    )
    stage = _run_trace_stage("noise", "ctle_noise.cir", "NOISE_OUTPUT_FILE", "noise.dat",
                             parameters, conditions, model, sparse, metrics)
    if stage.success and stage.metrics["input_referred_noise_vrms"] >= 1.5e-3:
        return StageResult("noise", False, stage.runtime_s, metrics=stage.metrics,
                           violations=("input-referred noise is at least 1.5 mVrms",),
                           failure_code=FailureCode.SIMULATION_ERROR.value, provenance=stage.provenance)
    return stage


def _run_hd3(parameters, conditions, model, ngspice, characterize: bool = False) -> StageResult:
    amplitudes = (0.05, 0.1, 0.2) if characterize else (0.1,)
    stages: list[StageResult] = []
    metrics: dict[str, float] = {}
    for amplitude in amplitudes:
        label_mv = int(round(amplitude * 1000))
        stage = _run_trace_stage(
            "hd3", "ctle_hd3.cir", "HD3_OUTPUT_FILE", f"hd3_{label_mv}mv.dat",
            parameters, conditions, model, ngspice,
            lambda trace, key=f"hd3_{label_mv}mv_db": {key: hd3_db(trace)},
            {"HD3_DIFF_PP": amplitude},
        )
        stages.append(stage)
        metrics.update(stage.metrics)
        if not stage.success:
            return StageResult("hd3", False, sum(item.runtime_s for item in stages),
                               metrics=metrics, failure_code=stage.failure_code,
                               errors=stage.errors, provenance=stage.provenance,
                               retryable=stage.retryable)
    metrics["hd3_db"] = metrics["hd3_100mv_db"]
    violations = () if metrics["hd3_db"] < -30 else ("HD3 is not below -30 dB at 100 mVpp differential",)
    return StageResult(
        "hd3", not violations, sum(item.runtime_s for item in stages), metrics=metrics,
        violations=violations,
        failure_code=None if not violations else FailureCode.SIMULATION_ERROR.value,
        provenance=stages[-1].provenance,
    )


def _transient_config(fidelity: EvaluationFidelity, conditions: SimulationConditions) -> NRZStimulusConfig:
    counts = {
        EvaluationFidelity.SCREENING: 32,
        EvaluationFidelity.TRAINING: 128,
        EvaluationFidelity.CANDIDATE: 512,
        EvaluationFidelity.FINAL: 1024,
    }
    count = counts[fidelity]
    warmup = 4 if count == 32 else (16 if count == 128 else 32)
    tail = warmup
    return NRZStimulusConfig(
        common_mode_v=conditions.input_common_mode_v, bit_count=count,
        warmup_bits=warmup, tail_bits=tail,
    )


def _source_division_compensation(conditions: SimulationConditions) -> float:
    source_diff = 2 * conditions.source_resistance_per_leg_ohm
    termination = conditions.receiver_termination_diff_ohm
    return (source_diff + termination) / termination


def _write_stimulus_include(path: Path, stimulus, waveform, conditions: SimulationConditions) -> None:
    path.write_text(
        stimulus_include(
            stimulus.time_s, waveform, conditions.input_common_mode_v,
            _source_division_compensation(conditions),
        ),
        encoding="utf-8",
    )


def _run_ctle_transient(parameters, conditions, model, ngspice) -> StageResult:
    numpy = require_numpy()
    stimulus = generate_nrz(_transient_config(EvaluationFidelity.SCREENING, conditions))
    with tempfile.TemporaryDirectory(prefix="nebula_ctle_stimulus_") as directory:
        include_path = Path(directory) / "stimulus.inc"
        _write_stimulus_include(include_path, stimulus, stimulus.differential_v, conditions)
        templates = _templates(model, conditions)
        templates.update({
            "STIMULUS_INCLUDE_FILE": include_path.as_posix(),
            "TRANSIENT_STEP": format(stimulus.config.time_step_s, ".15g"),
            "TRANSIENT_END": format(float(stimulus.time_s[-1]), ".15g"),
        })
        values = parameters.spice_parameters(conditions)
        values.update({
            "SOURCE_LEG_R": conditions.source_resistance_per_leg_ohm,
            "RTERM_DIFF": conditions.receiver_termination_diff_ohm,
            "CIN": conditions.input_parasitic_f,
        })
        result = run_simulation(SimulationRequest(
            BENCHES / "ctle_transient.cir", values, (), templates,
            {"WAVEFORM_OUTPUT_FILE": "ctle_transient.dat"},
        ), config=ngspice)
    if not result.success:
        return _stage_from_simulation("ctle_transient", result)
    try:
        trace = parse_wrdata(result.artifacts["WAVEFORM_OUTPUT_FILE"], ("vin_diff", "vout_diff"))
        vin = trace.column("vin_diff")
        vout = trace.column("vout_diff")
        if not numpy.ptp(vin) > 0.5 * stimulus.config.differential_pp_v:
            raise ValueError("terminated CTLE input amplitude is unexpectedly small")
        metrics = {
            "ctle_transient_input_pp_v": float(numpy.ptp(vin)),
            "ctle_transient_output_pp_v": float(numpy.ptp(vout)),
        }
        if not metrics["ctle_transient_output_pp_v"] > 0:
            raise ValueError("CTLE transient output has zero swing")
    except (KeyError, ValueError, RuntimeError) as exc:
        return StageResult("ctle_transient", False, result.runtime_s,
                           failure_code=FailureCode.WAVEFORM_PARSE_ERROR.value,
                           errors=(str(exc),), provenance=result.provenance)
    return StageResult("ctle_transient", True, result.runtime_s, metrics=metrics,
                       provenance={**result.provenance, "stimulus_checksum": stimulus.checksum})


def _run_channel_diagnostics(channel_path: str | Path) -> StageResult:
    import time
    started = time.perf_counter()
    try:
        channel = load_s4p(channel_path)
        numpy = require_numpy()
        transfer = channel.differential_transfer()
        if not math.isclose(channel.reference_ohm, 50.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Stage 1 requires a 50-ohm single-ended Touchstone reference")
        if channel.frequency_hz[-1] < 5e9:
            raise ValueError("channel data must extend through at least 5 GHz")
        if not numpy.isfinite(transfer).all():
            raise ValueError("channel transfer contains non-finite values")
        metrics = {
            "channel_reference_ohm": channel.reference_ohm,
            "channel_loss_1p25ghz_db": channel.insertion_loss_db(1.25e9),
            "channel_loss_2p5ghz_db": channel.insertion_loss_db(2.5e9),
            "channel_loss_5ghz_db": channel.insertion_loss_db(5e9),
        }
        return StageResult("channel", True, time.perf_counter() - started,
                           metrics=metrics, provenance={"channel_checksum": channel.checksum})
    except Exception as exc:
        return StageResult("channel", False, time.perf_counter() - started,
                           failure_code=FailureCode.CHANNEL_ERROR.value, errors=(str(exc),))


def _run_transient(parameters, conditions, model, ngspice, fidelity, channel_path) -> StageResult:
    numpy = require_numpy()
    stimulus = generate_nrz(_transient_config(fidelity, conditions))
    channel = load_s4p(channel_path)
    channel_output = filter_channel(channel, stimulus.time_s, stimulus.differential_v)
    with tempfile.TemporaryDirectory(prefix="nebula_stimulus_") as directory:
        include_path = Path(directory) / "stimulus.inc"
        _write_stimulus_include(include_path, stimulus, channel_output, conditions)
        templates = _templates(model, conditions)
        templates.update({
            "STIMULUS_INCLUDE_FILE": include_path.as_posix(),
            "TRANSIENT_STEP": format(stimulus.config.time_step_s, ".15g"),
            "TRANSIENT_END": format(float(stimulus.time_s[-1]), ".15g"),
        })
        values = parameters.spice_parameters(conditions)
        values.update({
            "SOURCE_LEG_R": conditions.source_resistance_per_leg_ohm,
            "RTERM_DIFF": conditions.receiver_termination_diff_ohm,
            "CIN": conditions.input_parasitic_f,
        })
        request = SimulationRequest(
            BENCHES / "receiver_transient.cir", values, (), templates,
            {"WAVEFORM_OUTPUT_FILE": "receiver.dat"},
        )
        result = run_simulation(request, config=ngspice)
    if not result.success:
        return _stage_from_simulation("transient", result)
    try:
        trace = parse_wrdata(result.artifacts["WAVEFORM_OUTPUT_FILE"])
        vout = trace.column("vout_diff")
        training_stop = stimulus.config.warmup_bits
        sampling = choose_sampling_phase(
            trace.scale, vout, stimulus.bits, stimulus.config.ui_s,
            0, training_stop, stimulus.config.time_step_s,
        )
        dfe = apply_hybrid_one_tap_dfe(
            sampling.raw_samples_v, stimulus.bits, parameters.dfe_tap_v,
            training_stop,
        )
        measurement_start = stimulus.config.warmup_bits
        measurement_stop = stimulus.config.bit_count - stimulus.config.tail_bits
        eye = eye_metrics(trace.scale, vout, stimulus.bits, stimulus.config.ui_s,
                          stimulus.config.time_step_s, measurement_start, measurement_stop)
        dfe_eye = dfe_eye_metrics(
            trace.scale, vout, stimulus.bits, stimulus.config.ui_s,
            stimulus.config.time_step_s, measurement_start, measurement_stop,
            parameters.dfe_tap_v,
            training_stop=training_stop,
            locked_phase_s=sampling.phase_s,
        )
        section = slice(measurement_start, measurement_stop)
        signs = expected_signs(stimulus.bits[section])
        corrected = dfe.corrected_samples_v[section]
        ones = corrected[stimulus.bits[section] == 1]
        zeros = corrected[stimulus.bits[section] == 0]
        metrics = {
            **eye,
            **dfe_eye,
            "dfe_tap_v": parameters.dfe_tap_v,
            "dfe_error_count": int(numpy.count_nonzero(dfe.decisions[section] != stimulus.bits[section])),
            "dfe_error_rate": float(numpy.mean(dfe.decisions[section] != stimulus.bits[section])),
            "dfe_min_margin_v": float(numpy.min(signs * corrected)),
            "sampling_phase_ui": sampling.phase_ui,
            "channel_loss_2p5ghz_db": channel.insertion_loss_db(2.5e9),
            "simulated_bits": measurement_stop - measurement_start,
        }
    except (KeyError, ValueError, RuntimeError) as exc:
        return StageResult("transient", False, result.runtime_s,
                           failure_code=FailureCode.WAVEFORM_PARSE_ERROR.value,
                           errors=(str(exc),), provenance=result.provenance)
    violations = _transient_violations(metrics, fidelity)
    return StageResult("transient", not violations, result.runtime_s, metrics=metrics,
                       violations=tuple(violations),
                       failure_code=None if not violations else FailureCode.SIMULATION_ERROR.value,
                       provenance={**result.provenance, "channel_checksum": channel.checksum,
                                   "stimulus_checksum": stimulus.checksum})


def _transient_violations(metrics: Mapping[str, float], fidelity: EvaluationFidelity) -> list[str]:
    violations: list[str] = []
    if fidelity >= EvaluationFidelity.CANDIDATE:
        if metrics["dfe_locked_phase_eye_height_v"] <= 0.1:
            violations.append("DFE vertical eye opening at the locked phase does not exceed 100 mV")
        if metrics["dfe_eye_width_ui"] <= 0.4:
            violations.append("horizontal eye opening does not exceed 0.4 UI")
    return violations


def _evaluation_is_cacheable(evaluation: ReceiverEvaluation) -> bool:
    return not any(
        stage.retryable or stage.failure_code == FailureCode.INTERNAL_ERROR.value
        for stage in evaluation.stages
    )


def evaluate_receiver(
    parameters: ReceiverParameters,
    conditions: SimulationConditions = SimulationConditions(),
    fidelity: EvaluationFidelity = EvaluationFidelity.TRAINING,
    *,
    channel_path: str | Path = SYNTHETIC_CHANNEL,
    sky130: Sky130Config | None = None,
    ngspice: NgSpiceConfig | None = None,
    cache: EvaluationCache | None = None,
) -> ReceiverEvaluation:
    started_stages: list[StageResult] = []
    effective_ngspice = ngspice or NgSpiceConfig(
        timeout_s=180.0 if fidelity >= EvaluationFidelity.FINAL else 60.0,
    )
    try:
        parameters.validate()
        conditions.validate()
        model = (sky130 or Sky130Config()).resolve_model_library()
    except Exception as exc:
        code = FailureCode.MODEL_LIBRARY_NOT_FOUND if isinstance(exc, FileNotFoundError) else FailureCode.INVALID_PARAMETER
        stage = StageResult("setup", False, 0.0, failure_code=code.value,
                            errors=(str(exc),))
        identity = stable_fingerprint({"parameters": asdict(parameters), "conditions": conditions.to_dict()})
        return ReceiverEvaluation(False, parameters, conditions, fidelity, (stage,), {}, "setup", 0.0,
                                  identity, git_identity(ROOT))

    try:
        simulator_identity = ngspice_identity(effective_ngspice)
    except Exception as exc:
        code = FailureCode.NGSPICE_NOT_FOUND if isinstance(exc, (FileNotFoundError, OSError)) else FailureCode.INVALID_PARAMETER
        stage = StageResult("setup", False, 0.0, failure_code=code.value,
                            errors=(str(exc),))
        identity = stable_fingerprint({"parameters": asdict(parameters), "conditions": conditions.to_dict()})
        return ReceiverEvaluation(False, parameters, conditions, fidelity, (stage,), {}, "setup", 0.0,
                                  identity, git_identity(ROOT))
    try:
        channel_checksum = sha256_file(channel_path)
    except (FileNotFoundError, OSError) as exc:
        stage = StageResult("setup", False, 0.0, failure_code=FailureCode.CHANNEL_ERROR.value,
                            errors=(str(exc),))
        identity = stable_fingerprint({"parameters": asdict(parameters), "conditions": conditions.to_dict()})
        return ReceiverEvaluation(False, parameters, conditions, fidelity, (stage,), {}, "setup", 0.0,
                                  identity, git_identity(ROOT))
    try:
        provenance = {
            **git_identity(ROOT),
            **simulator_identity,
            "model_library_checksum": sha256_file(model),
            "ctle_block_checksum": sha256_file(BLOCK),
            "channel_checksum": channel_checksum,
            "bench_checksum": stable_fingerprint({
                path.name: sha256_file(path) for path in sorted(BENCHES.glob("*.cir"))
            }),
            "implementation_checksum": stable_fingerprint({
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("receiver.py", "receiver_metrics.py", "channel.py", "stimulus.py", "waveform.py")
            }),
            "metric_schema_version": 1,
            "stimulus_schema_version": 1,
            }
    except Exception as exc:
        stage = StageResult("setup", False, 0.0, failure_code=FailureCode.INVALID_PARAMETER.value,
                            errors=(str(exc),))
        identity = stable_fingerprint({"parameters": asdict(parameters), "conditions": conditions.to_dict()})
        return ReceiverEvaluation(False, parameters, conditions, fidelity, (stage,), {}, "setup", 0.0,
                                  identity, git_identity(ROOT))
    evaluation_id = stable_fingerprint({
        "parameters": asdict(parameters), "conditions": conditions.to_dict(),
        "fidelity": fidelity.name, **provenance,
    })
    if cache:
        cached = cache.get(evaluation_id)
        if cached is not None:
            try:
                return ReceiverEvaluation.from_dict(cached)
            except (KeyError, TypeError, ValueError):
                # A corrupt or obsolete entry is a miss, never a fatal evaluation error.
                pass

    runners = [lambda: _run_dc(parameters, conditions, model, effective_ngspice),
               lambda: _run_ac(parameters, conditions, model, effective_ngspice),
               lambda: _run_ctle_transient(parameters, conditions, model, effective_ngspice)]
    if fidelity >= EvaluationFidelity.CANDIDATE:
        runners.extend((lambda: _run_noise(parameters, conditions, model, effective_ngspice),
                        lambda: _run_hd3(parameters, conditions, model, effective_ngspice,
                                        fidelity >= EvaluationFidelity.FINAL)))
    if fidelity >= EvaluationFidelity.TRAINING:
        runners.append(lambda: _run_channel_diagnostics(channel_path))
        runners.append(lambda: _run_transient(parameters, conditions, model, effective_ngspice,
                                              fidelity, channel_path))
    combined: dict[str, float] = {}
    failed_stage: str | None = None
    for runner in runners:
        try:
            stage = runner()
        except Exception as exc:
            stage = StageResult(
                "internal", False, 0.0,
                failure_code=FailureCode.INTERNAL_ERROR.value,
                errors=(f"{type(exc).__name__}: {exc}",),
            )
        started_stages.append(stage)
        combined.update(stage.metrics)
        if not stage.success:
            failed_stage = stage.name
            break
    evaluation = ReceiverEvaluation(
        failed_stage is None, parameters, conditions, fidelity, tuple(started_stages),
        combined, failed_stage, sum(stage.runtime_s for stage in started_stages),
        evaluation_id, provenance,
    )
    if cache and _evaluation_is_cacheable(evaluation):
        cache.put(evaluation_id, evaluation.to_dict())
    return evaluation


def evaluate_pvt_grid(
    parameters: ReceiverParameters,
    *,
    conditions: Iterable[SimulationConditions] = PVT_GRID,
    fidelity: EvaluationFidelity = EvaluationFidelity.FINAL,
    channel_path: str | Path = SYNTHETIC_CHANNEL,
    sky130: Sky130Config | None = None,
    ngspice: NgSpiceConfig | None = None,
    cache: EvaluationCache | None = None,
    stop_on_failure: bool = False,
) -> tuple[ReceiverEvaluation, ...]:
    """Evaluate an explicit PVT set; the default is the approved full 60-point grid."""
    results: list[ReceiverEvaluation] = []
    for item in conditions:
        result = evaluate_receiver(
            parameters, item, fidelity, channel_path=channel_path, sky130=sky130,
            ngspice=ngspice, cache=cache,
        )
        results.append(result)
        if stop_on_failure and not result.success:
            break
    return tuple(results)
