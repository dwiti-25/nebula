"""High-level evaluator for Person A's parameterized SKY130 CTLE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Mapping

from .config import Sky130Config
from .metrics import REQUIRED_CTLE_MEASUREMENTS, ctle_constraints, derive_ctle_metrics
from .models import FailureCode, FailureStage, ParameterValue, SimulationRequest
from .ngspice import NgSpiceConfig, run_simulation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CTLE_NETLIST = ROOT / "circuits" / "ctle_automation.cir"
PARAMETER_LIMITS = {
    "RLOAD": (100.0, 10_000.0),
    "RDEG": (10.0, 10_000.0),
    "CDEG": (10e-15, 10e-12),
    "ITAIL_VAL": (10e-6, 1e-3),
}
_VALUE_RE = re.compile(r"^([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)([a-z]+)?$", re.I)
_SCALE = {"": 1.0, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9}


@dataclass(frozen=True)
class CTLEEvaluation:
    parameters: dict[str, ParameterValue]
    success: bool
    measurements: dict[str, float]
    metrics: dict[str, float]
    constraints_passed: bool
    constraint_violations: tuple[str, ...]
    runtime_s: float
    failure_stage: str | None = None
    failure_code: str | None = None
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def spice_number(value: ParameterValue) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean values are not circuit parameters")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = _VALUE_RE.fullmatch(value.strip()) if isinstance(value, str) else None
        if not match or match.group(2).lower() not in _SCALE:
            raise ValueError(f"unsupported SPICE value: {value!r}")
        number = float(match.group(1)) * _SCALE[match.group(2).lower()]
    if not math.isfinite(number):
        raise ValueError("circuit values must be finite")
    return number


def validate_ctle_parameters(parameters: Mapping[str, ParameterValue]) -> None:
    missing = sorted(set(PARAMETER_LIMITS) - parameters.keys())
    unknown = sorted(set(parameters) - set(PARAMETER_LIMITS))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise ValueError("invalid CTLE parameter set (" + "; ".join(details) + ")")
    for name, (minimum, maximum) in PARAMETER_LIMITS.items():
        value = spice_number(parameters[name])
        if not minimum <= value <= maximum:
            raise ValueError(f"{name}={parameters[name]!r} is outside [{minimum:g}, {maximum:g}]")


def evaluate_ctle(
    parameters: Mapping[str, ParameterValue],
    *,
    sky130: Sky130Config | None = None,
    ngspice: NgSpiceConfig | None = None,
    netlist: str | Path = DEFAULT_CTLE_NETLIST,
) -> CTLEEvaluation:
    values = dict(parameters)
    try:
        validate_ctle_parameters(values)
        model_library = (sky130 or Sky130Config()).resolve_model_library()
    except (TypeError, ValueError) as exc:
        return CTLEEvaluation(
            values, False, {}, {}, False, (), 0.0,
            FailureStage.SETUP.value, FailureCode.INVALID_PARAMETER.value,
            (str(exc),),
        )
    except FileNotFoundError as exc:
        return CTLEEvaluation(
            values, False, {}, {}, False, (), 0.0,
            FailureStage.SETUP.value, FailureCode.MODEL_LIBRARY_NOT_FOUND.value,
            (str(exc),),
        )
    request = SimulationRequest(
        netlist=netlist,
        parameters=values,
        expected_measurements=REQUIRED_CTLE_MEASUREMENTS,
        template_values={"SKY130_MODEL_LIBRARY": model_library.as_posix()},
    )
    result = run_simulation(request, config=ngspice)
    if not result.success:
        return CTLEEvaluation(
            values, False, result.measurements, {}, False, (), result.runtime_s,
            result.failure_stage.value if result.failure_stage else None,
            result.failure_code.value if result.failure_code else None,
            result.errors,
        )
    metrics = derive_ctle_metrics(result.measurements)
    valid, violations = ctle_constraints(metrics)
    return CTLEEvaluation(
        values, valid, result.measurements, metrics, valid, violations,
        result.runtime_s,
        None if valid else FailureStage.SIMULATION.value,
        None if valid else FailureCode.SIMULATION_ERROR.value,
        violations,
    )
