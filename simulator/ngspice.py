"""Robust subprocess wrapper for parameterized NGSpice simulations."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Mapping

from .parser import find_simulation_errors, parse_measurements
from .models import (
    FailureCode,
    FailureStage,
    ParameterValue,
    SimulationRequest,
    SimulationResult,
    SimulationStatus,
)
from .provenance import sha256_text, stable_fingerprint


_PARAM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TEMPLATE_NAME_RE = _PARAM_NAME_RE


@dataclass(frozen=True)
class NgSpiceConfig:
    """Process-level NGSpice configuration kept in one place."""

    executable: str | Path | None = None
    timeout_s: float = 60.0
    solver: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.timeout_s, (int, float)) or isinstance(self.timeout_s, bool):
            raise TypeError("NGSpice timeout must be a number")
        if not math.isfinite(float(self.timeout_s)) or self.timeout_s <= 0:
            raise ValueError("NGSpice timeout must be finite and greater than zero")

    def resolve_executable(self) -> Path:
        requested = self.executable or os.environ.get("NGSPICE_EXECUTABLE")
        if requested:
            path = Path(requested).expanduser()
            if path.is_file():
                return path.resolve()
            raise FileNotFoundError(f"NGSpice executable does not exist: {path}")

        for name in ("ngspice_con.exe", "ngspice.exe", "ngspice"):
            found = shutil.which(name)
            if found:
                return Path(found).resolve()
        raise FileNotFoundError(
            "NGSpice was not found. Put ngspice_con.exe on PATH or set "
            "NGSPICE_EXECUTABLE to its full path."
        )


_RESOURCE_INIT = Path(__file__).resolve().parent / "resources" / "sky130.spiceinit"


@lru_cache(maxsize=8)
def _version_text(executable: str) -> str:
    try:
        process = subprocess.run(
            [executable, "--version"], capture_output=True, text=True,
            errors="replace", timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return ((process.stdout or "") + (process.stderr or "")).strip() or "unknown"


def _supports_klu(executable: str) -> bool:
    return "klu" in _version_text(executable).lower()


def ngspice_identity(config: NgSpiceConfig | None = None) -> dict[str, object]:
    """Return stable simulator inputs used by evaluation fingerprints."""
    settings = config or NgSpiceConfig()
    executable = settings.resolve_executable()
    init_text, solver = _initialization_text(
        SimulationRequest(netlist="identity-only"), settings, executable,
    )
    return {
        "ngspice_executable": str(executable),
        "ngspice_version": _version_text(str(executable)),
        "solver": solver,
        "initialization_checksum": sha256_text(init_text),
        "timeout_s": settings.timeout_s,
    }


def _initialization_text(request: SimulationRequest, settings: NgSpiceConfig, executable: Path) -> tuple[str, str]:
    if request.initialization_text is not None:
        base = request.initialization_text
    else:
        base = _RESOURCE_INIT.read_text(encoding="utf-8")
    solver = settings.solver.lower()
    if solver not in {"auto", "klu", "sparse"}:
        raise ValueError("solver must be auto, klu, or sparse")
    use_klu = solver == "klu" or (solver == "auto" and _supports_klu(str(executable)))
    if use_klu:
        base = base.rstrip() + "\noption klu\n"
    return base, "klu" if use_klu else "sparse"


def _spice_value(value: int | float | str) -> str:
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid circuit parameters")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("Circuit parameters must be finite")
        return format(value, ".15g")
    if not isinstance(value, str) or not value.strip():
        raise TypeError("Circuit parameters must be numbers or explicit SPICE strings")
    if any(character.isspace() for character in value):
        raise ValueError("SPICE parameter strings may not contain whitespace")
    return value


def parameterize_netlist(source: str, parameters: Mapping[str, ParameterValue]) -> str:
    """Replace declared ``.param`` values without changing the master file."""

    rendered = source
    for name, value in parameters.items():
        if not _PARAM_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid SPICE parameter name: {name!r}")
        pattern = re.compile(
            rf"(?im)^(\s*\.param\b[^\r\n]*?\b{re.escape(name)}\s*=\s*)([^\s;$]+)"
        )
        rendered, replacements = pattern.subn(
            lambda match: match.group(1) + _spice_value(value), rendered
        )
        if replacements == 0:
            raise KeyError(f"Parameter {name!r} is not declared in the netlist")
    return rendered


def render_template(source: str, values: Mapping[str, str]) -> str:
    """Replace explicit ``@@NAME@@`` placeholders in a netlist."""

    rendered = source
    for name, value in values.items():
        if not _TEMPLATE_NAME_RE.fullmatch(name):
            raise ValueError(f"Invalid netlist template name: {name!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Template value {name!r} must be a non-empty string")
        if any(character in value for character in ('\n', '\r', '"')):
            raise ValueError(f"Template value {name!r} contains unsafe characters")
        marker = f"@@{name}@@"
        if marker not in rendered:
            raise KeyError(f"Template marker {marker!r} is not declared in the netlist")
        rendered = rendered.replace(marker, value)
    unresolved = sorted(set(re.findall(r"@@([A-Za-z_][A-Za-z0-9_]*)@@", rendered)))
    if unresolved:
        raise KeyError("Missing netlist template values: " + ", ".join(unresolved))
    return rendered


def _failure_result(
    message: str,
    runtime_s: float,
    *,
    stage: FailureStage,
    code: FailureCode,
    retryable: bool = False,
    stdout: str = "",
    stderr: str | None = None,
    errors: tuple[str, ...] | None = None,
    returncode: int | None = None,
    measurements: dict[str, float] | None = None,
    artifacts: dict[str, str] | None = None,
    provenance: dict[str, object] | None = None,
) -> SimulationResult:
    return SimulationResult(
        status=SimulationStatus.FAILED,
        measurements=measurements or {},
        runtime_s=runtime_s,
        stdout=stdout,
        stderr=message if stderr is None else stderr,
        errors=errors or (message,),
        returncode=returncode,
        failure_stage=stage,
        failure_code=code,
        retryable=retryable,
        artifacts=artifacts or {},
        provenance=provenance or {},
    )


def _classify_simulation_error(errors: list[str]) -> FailureCode:
    combined = "\n".join(errors).lower()
    if any(term in combined for term in ("no convergence", "timestep too small", "singular matrix")):
        return FailureCode.CONVERGENCE_FAILURE
    if "measure" in combined and "failed" in combined:
        return FailureCode.MEASUREMENT_FAILED
    return FailureCode.SIMULATION_ERROR


def run_ngspice(
    netlist: str | Path,
    parameters: Mapping[str, ParameterValue] | None = None,
    *,
    expected_measurements: tuple[str, ...] = (),
    config: NgSpiceConfig | None = None,
) -> SimulationResult:
    """Run a copied, parameterized netlist and return structured results.

    Setup and simulator failures are represented by ``success=False`` rather
    than escaping into an optimization loop as exceptions.
    """

    request = SimulationRequest(
        netlist=netlist,
        parameters=parameters or {},
        expected_measurements=expected_measurements,
    )
    return run_simulation(request, config=config)


def run_simulation(
    request: SimulationRequest, *, config: NgSpiceConfig | None = None
) -> SimulationResult:
    """Execute a typed simulation request."""

    started = time.perf_counter()
    source_path = Path(request.netlist).expanduser().resolve()
    try:
        source = source_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        return _failure_result(
            str(exc), time.perf_counter() - started,
            stage=FailureStage.SETUP, code=FailureCode.NETLIST_NOT_FOUND,
        )
    except OSError as exc:
        return _failure_result(
            str(exc), time.perf_counter() - started,
            stage=FailureStage.SETUP, code=FailureCode.NETLIST_READ_ERROR,
        )
    try:
        rendered_parameters = parameterize_netlist(source, request.parameters)
    except (TypeError, ValueError, KeyError) as exc:
        return _failure_result(
            str(exc), time.perf_counter() - started,
            stage=FailureStage.SETUP, code=FailureCode.INVALID_PARAMETER,
        )
    settings = config or NgSpiceConfig()
    try:
        executable = settings.resolve_executable()
    except FileNotFoundError as exc:
        return _failure_result(
            str(exc), time.perf_counter() - started,
            stage=FailureStage.SETUP, code=FailureCode.NGSPICE_NOT_FOUND,
        )

    try:
        init_text, selected_solver = _initialization_text(request, settings, executable)
    except (OSError, ValueError) as exc:
        return _failure_result(
            str(exc), time.perf_counter() - started,
            stage=FailureStage.SETUP, code=FailureCode.NETLIST_READ_ERROR,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="analog_rl_ngspice_") as temp_name:
            temp_dir = Path(temp_name)
            generated = temp_dir / source_path.name
            log_path = temp_dir / "ngspice.log"
            output_paths: dict[str, Path] = {}
            for marker, filename in request.output_files.items():
                if Path(filename).name != filename or not filename:
                    return _failure_result(
                        f"Output filename must be a simple relative name: {filename!r}",
                        time.perf_counter() - started,
                        stage=FailureStage.SETUP, code=FailureCode.INVALID_PARAMETER,
                    )
                output_paths[marker] = temp_dir / filename
            template_values = dict(request.template_values)
            # ngspice on Windows rejects some absolute paths passed to wrdata.
            # The process runs in temp_dir, so a simple relative filename is deterministic.
            template_values.update({name: path.name for name, path in output_paths.items()})
            try:
                rendered = render_template(rendered_parameters, template_values)
            except (TypeError, ValueError, KeyError) as exc:
                return _failure_result(
                    str(exc), time.perf_counter() - started,
                    stage=FailureStage.SETUP, code=FailureCode.INVALID_PARAMETER,
                )
            generated.write_text(rendered, encoding="utf-8")
            (temp_dir / ".spiceinit").write_text(init_text, encoding="utf-8")
            provenance = {
                "netlist_source": str(source_path),
                "source_checksum": sha256_text(source),
                "rendered_checksum": sha256_text(rendered),
                "initialization_checksum": sha256_text(init_text),
                "solver": selected_solver,
                "ngspice_executable": str(executable),
                "ngspice_version": _version_text(str(executable)),
            }
            provenance["evaluation_id"] = stable_fingerprint({
                **provenance,
                "parameters": dict(request.parameters),
                "templates": dict(request.template_values),
                "outputs": dict(request.output_files),
            })
            child_environment = os.environ.copy()
            child_environment["HOME"] = str(temp_dir)
            if os.name == "nt":
                child_environment["USERPROFILE"] = str(temp_dir)
            try:
                process = subprocess.run(
                    [str(executable), "-b", "-o", str(log_path), str(generated)],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=settings.timeout_s,
                    check=False,
                    env=child_environment,
                )
            except subprocess.TimeoutExpired as exc:
                runtime = time.perf_counter() - started
                return _failure_result(
                    f"NGSpice timed out after {settings.timeout_s:g} seconds",
                    runtime,
                    stage=FailureStage.EXECUTION,
                    code=FailureCode.NGSPICE_TIMEOUT,
                    retryable=True,
                    stdout=exc.stdout or "",
                    provenance=provenance,
                )

            log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            stdout = process.stdout or ""
            stderr = process.stderr or ""
            combined = "\n".join((stdout, stderr, log))
            artifacts = {
                marker: path.read_text(encoding="utf-8", errors="replace")
                for marker, path in output_paths.items() if path.is_file()
            }
            parsed = parse_measurements(
                combined, request.normalized_expected_measurements()
            )
            errors = find_simulation_errors(combined)
            if process.returncode != 0:
                message = f"NGSpice exited with status {process.returncode}"
                return _failure_result(
                    message, time.perf_counter() - started,
                    stage=FailureStage.EXECUTION,
                    code=FailureCode.NGSPICE_PROCESS_FAILURE,
                    stdout=stdout + log, stderr=stderr,
                    errors=tuple([message, *errors, *parsed.errors]),
                    returncode=process.returncode,
                    measurements=parsed.measurements,
                    artifacts=artifacts, provenance=provenance,
                )
            if errors:
                return _failure_result(
                    errors[0], time.perf_counter() - started,
                    stage=FailureStage.SIMULATION,
                    code=_classify_simulation_error(errors),
                    stdout=stdout + log, stderr=stderr,
                    errors=tuple(errors), returncode=process.returncode,
                    measurements=parsed.measurements,
                    artifacts=artifacts, provenance=provenance,
                )
            if parsed.errors:
                return _failure_result(
                    parsed.errors[0], time.perf_counter() - started,
                    stage=FailureStage.PARSING,
                    code=FailureCode.MEASUREMENT_PARSE_ERROR,
                    stdout=stdout + log, stderr=stderr,
                    errors=parsed.errors, returncode=process.returncode,
                    measurements=parsed.measurements,
                    artifacts=artifacts, provenance=provenance,
                )
            missing = sorted(
                set(request.normalized_expected_measurements()) - parsed.measurements.keys()
            )
            if missing:
                message = "Missing expected measurements: " + ", ".join(missing)
                return _failure_result(
                    message, time.perf_counter() - started,
                    stage=FailureStage.PARSING,
                    code=FailureCode.MEASUREMENT_MISSING,
                    stdout=stdout + log, stderr=stderr,
                    returncode=process.returncode,
                    measurements=parsed.measurements,
                    artifacts=artifacts, provenance=provenance,
                )
            missing_artifacts = sorted(set(output_paths) - artifacts.keys())
            if missing_artifacts:
                message = "Missing expected output files: " + ", ".join(missing_artifacts)
                return _failure_result(
                    message, time.perf_counter() - started,
                    stage=FailureStage.PARSING,
                    code=FailureCode.OUTPUT_FILE_MISSING,
                    stdout=stdout + log, stderr=stderr,
                    returncode=process.returncode,
                    measurements=parsed.measurements,
                    artifacts=artifacts, provenance=provenance,
                )

            return SimulationResult(
                status=SimulationStatus.SUCCESS,
                measurements=parsed.measurements,
                runtime_s=time.perf_counter() - started,
                stdout=stdout + log,
                stderr=stderr,
                returncode=process.returncode,
                artifacts=artifacts,
                provenance=provenance,
            )
    except OSError as exc:
        return _failure_result(
            str(exc), time.perf_counter() - started,
            stage=FailureStage.EXECUTION,
            code=FailureCode.PROCESS_LAUNCH_ERROR,
            retryable=True,
        )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a parameterized NGSpice netlist")
    parser.add_argument("netlist", type=Path)
    parser.add_argument("parameters", nargs="*", metavar="NAME=VALUE")
    args = parser.parse_args()
    values: dict[str, str] = {}
    for item in args.parameters:
        name, separator, value = item.partition("=")
        if not separator:
            parser.error(f"parameter must be NAME=VALUE: {item}")
        values[name] = value
    result = run_ngspice(args.netlist, values)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
