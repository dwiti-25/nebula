"""Parsing and analysis of ngspice wrdata traces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by installation checks
    np = None


def require_numpy():
    if np is None:
        raise RuntimeError("NumPy is required; install dependencies from requirements.txt")
    return np


@dataclass(frozen=True)
class Trace:
    scale_name: str
    scale: object
    columns: dict[str, object]

    def column(self, name: str):
        try:
            return self.columns[name.lower()]
        except KeyError as exc:
            raise KeyError(f"trace does not contain column {name!r}") from exc


def parse_wrdata(text: str, expected: Iterable[str] = ()) -> Trace:
    numpy = require_numpy()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("wrdata output is empty or truncated")
    header = lines[0].split()
    rows: list[list[float]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        try:
            values = [float(token) for token in line.split()]
        except ValueError as exc:
            raise ValueError(f"non-numeric wrdata row {line_number}") from exc
        if len(values) != len(header):
            raise ValueError(
                f"wrdata row {line_number} has {len(values)} values; expected {len(header)}"
            )
        rows.append(values)
    data = numpy.asarray(rows, dtype=float)
    if not numpy.isfinite(data).all():
        raise ValueError("wrdata contains non-finite values")
    scale = data[:, 0]
    if len(scale) < 2 or not numpy.all(numpy.diff(scale) > 0):
        raise ValueError("wrdata scale must be finite and strictly increasing")
    columns = {name.lower(): data[:, index] for index, name in enumerate(header[1:], start=1)}
    missing = sorted({name.lower() for name in expected} - columns.keys())
    if missing:
        raise ValueError("missing trace columns: " + ", ".join(missing))
    return Trace(header[0].lower(), scale, columns)


def _at(frequency, values, target: float) -> float:
    numpy = require_numpy()
    return float(numpy.interp(target, frequency, values))


def ac_metrics(trace: Trace) -> dict[str, float]:
    numpy = require_numpy()
    frequency = trace.scale
    magnitude = trace.column("transfer_mag")
    phase_deg = trace.column("transfer_phase_deg")
    if numpy.any(magnitude <= 0):
        raise ValueError("AC transfer magnitude must be positive")
    gain_db = 20 * numpy.log10(magnitude)
    metrics = {
        "gain_1mhz_db": _at(frequency, gain_db, 1e6),
        "gain_10mhz_db": _at(frequency, gain_db, 10e6),
        "gain_100mhz_db": _at(frequency, gain_db, 100e6),
        "gain_1p25ghz_db": _at(frequency, gain_db, 1.25e9),
        "gain_2p5ghz_db": _at(frequency, gain_db, 2.5e9),
        "gain_5ghz_db": _at(frequency, gain_db, 5e9),
    }
    intended = (frequency >= 1.25e9) & (frequency <= 2.5e9)
    if not numpy.any(intended):
        raise ValueError("AC trace does not cover the intended peaking band")
    intended_indices = numpy.flatnonzero(intended)
    peak_index = int(intended_indices[numpy.argmax(gain_db[intended])])
    global_index = int(numpy.argmax(gain_db))
    metrics.update({
        "peak_gain_db": float(gain_db[peak_index]),
        "peak_frequency_hz": float(frequency[peak_index]),
        "peaking_db": float(gain_db[peak_index] - metrics["gain_100mhz_db"]),
        "global_peak_gain_db": float(gain_db[global_index]),
        "global_peak_frequency_hz": float(frequency[global_index]),
        "out_of_band_excess_peak_db": float(gain_db[global_index] - gain_db[peak_index]),
    })
    phase_rad = numpy.unwrap(numpy.deg2rad(phase_deg))
    group_delay = -numpy.gradient(phase_rad, 2 * math.pi * frequency)
    valid_delay = (frequency >= 100e6) & (frequency <= 5e9)
    metrics["group_delay_mean_s"] = float(numpy.mean(group_delay[valid_delay]))
    metrics["group_delay_span_s"] = float(numpy.ptp(group_delay[valid_delay]))
    return metrics


def integrated_input_noise(trace: Trace) -> float:
    numpy = require_numpy()
    density = trace.column("inoise_spectrum")
    integrate = getattr(numpy, "trapezoid", None)
    if integrate is None:  # NumPy versions before trapezoid was introduced
        integrate = numpy.trapz
    power = integrate(density * density, trace.scale)
    if power < 0 or not math.isfinite(float(power)):
        raise ValueError("invalid integrated noise power")
    return math.sqrt(float(power))


def hd3_db(trace: Trace, fundamental_hz: float = 100e6, discard_before_s: float = 100e-9) -> float:
    numpy = require_numpy()
    time = trace.scale
    signal = trace.column("vout_diff")
    keep = time >= discard_before_s
    time = time[keep]
    signal = signal[keep]
    if len(time) < 16:
        raise ValueError("insufficient steady-state samples for HD3")
    step = float(numpy.median(numpy.diff(time)))
    signal = signal - numpy.mean(signal)
    spectrum = numpy.fft.rfft(signal)
    frequencies = numpy.fft.rfftfreq(len(signal), step)
    fundamental = abs(spectrum[int(numpy.argmin(abs(frequencies - fundamental_hz)))])
    third = abs(spectrum[int(numpy.argmin(abs(frequencies - 3 * fundamental_hz)))])
    if fundamental <= 0:
        raise ValueError("HD3 fundamental is zero")
    return 20 * math.log10(max(float(third / fundamental), 1e-300))
