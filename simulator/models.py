"""Typed request and result contracts for circuit simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping


ParameterValue = int | float | str


class SimulationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


class FailureStage(str, Enum):
    SETUP = "setup"
    EXECUTION = "execution"
    SIMULATION = "simulation"
    PARSING = "parsing"


class FailureCode(str, Enum):
    INTERNAL_ERROR = "internal_error"
    INVALID_PARAMETER = "invalid_parameter"
    NETLIST_NOT_FOUND = "netlist_not_found"
    NETLIST_READ_ERROR = "netlist_read_error"
    NGSPICE_NOT_FOUND = "ngspice_not_found"
    MODEL_LIBRARY_NOT_FOUND = "model_library_not_found"
    PROCESS_LAUNCH_ERROR = "process_launch_error"
    NGSPICE_TIMEOUT = "ngspice_timeout"
    NGSPICE_PROCESS_FAILURE = "ngspice_process_failure"
    CONVERGENCE_FAILURE = "convergence_failure"
    MEASUREMENT_FAILED = "measurement_failed"
    MEASUREMENT_MISSING = "measurement_missing"
    MEASUREMENT_PARSE_ERROR = "measurement_parse_error"
    SIMULATION_ERROR = "simulation_error"
    OUTPUT_FILE_MISSING = "output_file_missing"
    WAVEFORM_PARSE_ERROR = "waveform_parse_error"
    CHANNEL_ERROR = "channel_error"


@dataclass(frozen=True)
class SimulationRequest:
    """Everything needed to perform one deterministic NGSpice evaluation."""

    netlist: str | Path
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)
    expected_measurements: tuple[str, ...] = ()
    template_values: Mapping[str, str] = field(default_factory=dict)
    output_files: Mapping[str, str] = field(default_factory=dict)
    initialization_text: str | None = None

    def normalized_expected_measurements(self) -> tuple[str, ...]:
        return tuple(name.lower() for name in self.expected_measurements)


@dataclass(frozen=True)
class SimulationResult:
    """Typed outcome returned for both successful and failed simulations."""

    status: SimulationStatus
    measurements: dict[str, float]
    runtime_s: float
    stdout: str = ""
    stderr: str = ""
    errors: tuple[str, ...] = ()
    returncode: int | None = None
    failure_stage: FailureStage | None = None
    failure_code: FailureCode | None = None
    retryable: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status is SimulationStatus.SUCCESS

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the result."""

        value = asdict(self)
        value["status"] = self.status.value
        value["failure_stage"] = self.failure_stage.value if self.failure_stage else None
        value["failure_code"] = self.failure_code.value if self.failure_code else None
        value["success"] = self.success
        return value
