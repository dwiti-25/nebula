"""Strict measurement parsing and error detection for NGSpice output."""

from __future__ import annotations

import re
from dataclasses import dataclass
import math


_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*=\s*(.*?)\s*$")
_NUMBER_RE = re.compile(
    r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?$", re.IGNORECASE
)
_NONFINITE_RE = re.compile(r"^[-+]?(?:nan|inf(?:inity)?)$", re.IGNORECASE)

_ERROR_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\berror:\s*",
        r"\bfatal error\b",
        r"\bmeasure(?:ment)?\s+\S+\s+failed\b",
        r"\bno convergence\b",
        r"\btimestep too small\b",
        r"\bsingular matrix\b",
        r"\bdoAnalyses:.*failed\b",
    )
)


@dataclass(frozen=True)
class MeasurementParseResult:
    measurements: dict[str, float]
    errors: tuple[str, ...]


def parse_measurements(
    output: str, expected: tuple[str, ...] = ()
) -> MeasurementParseResult:
    """Parse scalar measurements and report duplicate or invalid values.

    When *expected* is supplied, assignments for those names are treated as
    measurements even when their values are malformed. Other arbitrary NGSpice
    assignments are ignored unless their value looks numeric.
    """

    expected_names = {name.lower() for name in expected}
    measurements: dict[str, float] = {}
    errors: list[str] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        match = _ASSIGNMENT_RE.match(line)
        if not match:
            continue
        original_name, raw_value = match.groups()
        name = original_name.lower()
        if expected_names and name not in expected_names:
            continue
        numeric = bool(_NUMBER_RE.fullmatch(raw_value))
        nonfinite = bool(_NONFINITE_RE.fullmatch(raw_value))
        if name not in expected_names and not numeric and not nonfinite:
            continue
        if name in measurements:
            errors.append(f"Duplicate measurement {name!r} on line {line_number}")
            continue
        if not numeric:
            description = "non-finite" if nonfinite else "invalid"
            errors.append(
                f"Measurement {name!r} has {description} value {raw_value!r} "
                f"on line {line_number}"
            )
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            errors.append(
                f"Measurement {name!r} has non-finite value on line {line_number}"
            )
            continue
        measurements[name] = value
    return MeasurementParseResult(measurements, tuple(errors))


def find_simulation_errors(output: str) -> list[str]:
    """Return concise output lines that indicate a failed simulation."""

    return [
        line.strip()
        for line in output.splitlines()
        if any(pattern.search(line) for pattern in _ERROR_PATTERNS)
    ]
