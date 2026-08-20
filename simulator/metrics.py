"""Circuit metric calculations kept separate from simulator output parsing."""

from __future__ import annotations

import math
from typing import Mapping


REQUIRED_CTLE_MEASUREMENTS = (
    "gain_1mhz_mag",
    "gain_2p5ghz_mag",
    "gain_5ghz_mag",
    "gain_100ghz_mag",
    "outp_dc_v",
    "outn_dc_v",
    "vdd_current_a",
)


def derive_ctle_metrics(measurements: Mapping[str, float], vdd_v: float = 1.8) -> dict[str, float]:
    values = {name: float(measurements[name]) for name in REQUIRED_CTLE_MEASUREMENTS}
    for frequency in ("1mhz", "2p5ghz", "5ghz", "100ghz"):
        magnitude = values[f"gain_{frequency}_mag"]
        if magnitude <= 0:
            raise ValueError(f"gain magnitude at {frequency} must be positive")
        values[f"gain_{frequency}_db"] = 20.0 * math.log10(magnitude)
    values["peaking_2p5ghz_db"] = values["gain_2p5ghz_db"] - values["gain_1mhz_db"]
    values["output_common_mode_v"] = (values["outp_dc_v"] + values["outn_dc_v"]) / 2
    values["output_mismatch_v"] = abs(values["outp_dc_v"] - values["outn_dc_v"])
    values["power_w"] = abs(values["vdd_current_a"]) * vdd_v
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("CTLE metrics must all be finite")
    return values


def ctle_constraints(metrics: Mapping[str, float]) -> tuple[bool, tuple[str, ...]]:
    violations: list[str] = []
    common_mode = metrics["output_common_mode_v"]
    if not 0.1 <= common_mode <= 1.79:
        violations.append("output common-mode voltage is outside 0.1 V to 1.79 V")
    if metrics["output_mismatch_v"] > 0.05:
        violations.append("DC differential output mismatch exceeds 50 mV")
    if metrics["power_w"] <= 0:
        violations.append("computed power is not positive")
    return not violations, tuple(violations)
