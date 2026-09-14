"""Touchstone four-port parsing and differential channel filtering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

from .provenance import sha256_file
from .waveform import require_numpy


_FREQUENCY_SCALE = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def _interpolate_transfer(frequency, source_frequency, transfer):
    """Interpolate magnitude and unwrapped phase, preserving long bulk delay."""
    numpy = require_numpy()
    magnitude = numpy.abs(transfer)
    phase = numpy.unwrap(numpy.angle(transfer))
    interpolated_magnitude = numpy.interp(
        frequency, source_frequency, magnitude, left=magnitude[0], right=0.0,
    )
    interpolated_phase = numpy.interp(frequency, source_frequency, phase)
    if len(source_frequency) >= 2:
        low_slope = (phase[1] - phase[0]) / (source_frequency[1] - source_frequency[0])
        below = frequency < source_frequency[0]
        interpolated_phase[below] = phase[0] + low_slope * (frequency[below] - source_frequency[0])
    return interpolated_magnitude * numpy.exp(1j * interpolated_phase)


def sampling_offset_bits(bulk_delay_s: float, ui_s: float) -> int:
    """Integer offset that keeps the delayed bit center in fractional phase [0, UI)."""
    if not math.isfinite(bulk_delay_s) or bulk_delay_s < 0 or not math.isfinite(ui_s) or ui_s <= 0:
        raise ValueError("bulk delay must be nonnegative and UI must be positive")
    ratio = bulk_delay_s / ui_s
    # Bias only the floating-point tie itself upward (round-half-up).  This
    # prevents a fitted 0.5 UI represented as 0.49999999999999994 from choosing
    # the previous symbol while leaving physically distinct delays unchanged.
    return int(math.floor(ratio + 0.5 + 8 * math.ulp(max(abs(ratio), 1.0))))


def sampling_reference_phase_s(bulk_delay_s: float, ui_s: float,
                               offset_bits: int, phase_step_s: float) -> float:
    """Return a grid-stable fractional bit-center phase in ``[0, UI)``."""
    if not math.isfinite(phase_step_s) or phase_step_s <= 0 or phase_step_s >= ui_s:
        raise ValueError("phase step must be finite, positive, and smaller than one UI")
    unwrapped = bulk_delay_s + ui_s / 2 - offset_bits * ui_s
    phase = unwrapped % ui_s
    # Fitted delays near a half-UI boundary can produce UI-epsilon rather than
    # zero.  Those points are the same grid phase and must share bit labels.
    if min(phase, ui_s - phase) <= phase_step_s / 2:
        return 0.0
    return phase


@dataclass(frozen=True)
class ChannelPortMap:
    """One-based physical port assignment used to form mixed-mode Sdd21."""

    tx_positive: int = 1
    tx_negative: int = 2
    rx_positive: int = 3
    rx_negative: int = 4

    @property
    def tx_ports(self) -> tuple[int, int]:
        return self.tx_positive, self.tx_negative

    @property
    def rx_ports(self) -> tuple[int, int]:
        return self.rx_positive, self.rx_negative

    def validate(self) -> None:
        if sorted((*self.tx_ports, *self.rx_ports)) != [1, 2, 3, 4]:
            raise ValueError("port mapping must use each of ports 1, 2, 3, and 4 exactly once")


@dataclass(frozen=True)
class S4PChannel:
    path: Path
    frequency_hz: object
    matrix: object
    reference_ohm: float
    data_format: str
    checksum: str
    tx_ports: tuple[int, int] = (1, 2)
    rx_ports: tuple[int, int] = (3, 4)

    def differential_transfer(self):
        txp, txn = (port - 1 for port in self.tx_ports)
        rxp, rxn = (port - 1 for port in self.rx_ports)
        s = self.matrix
        return 0.5 * (s[:, rxp, txp] - s[:, rxp, txn] - s[:, rxn, txp] + s[:, rxn, txn])

    def insertion_loss_db(self, target_hz: float) -> float:
        transfer = self.differential_transfer()
        value = _interpolate_transfer(
            require_numpy().asarray([target_hz], dtype=float), self.frequency_hz, transfer,
        )[0]
        return 20 * math.log10(max(abs(value), 1e-300))

    @property
    def port_map(self) -> ChannelPortMap:
        return ChannelPortMap(*self.tx_ports, *self.rx_ports)

    def bulk_delay_s(self, low_hz: float = 100e6, high_hz: float = 5e9) -> float:
        """Estimate causal bulk delay from a robust fit of unwrapped Sdd21 phase."""
        numpy = require_numpy()
        transfer = self.differential_transfer()
        selected = (
            (self.frequency_hz >= low_hz) & (self.frequency_hz <= high_hz)
            & (numpy.abs(transfer) > 1e-9)
        )
        if numpy.count_nonzero(selected) < 3:
            raise ValueError("channel has too few in-band points for a delay estimate")
        frequency = self.frequency_hz[selected]
        phase = numpy.unwrap(numpy.angle(transfer[selected]))
        slope = float(numpy.polyfit(frequency, phase, 1)[0])
        delay = -slope / (2 * math.pi)
        # Small negative values are normally numerical phase-fit noise.  A
        # material negative delay is rejected by diagnostics as noncausal.
        return 0.0 if abs(delay) < 1e-15 else delay

    def validation_metrics(self, high_hz: float = 5e9) -> dict[str, float]:
        """Return quantitative passivity and time-domain causality diagnostics.

        This is intentionally a qualification gate, not a full rational-model
        proof.  Real signoff should additionally use a trusted VNA/model tool.
        """
        numpy = require_numpy()
        selected = self.frequency_hz <= high_hz
        if numpy.count_nonzero(selected) < 3:
            raise ValueError("channel has too few points in its validation band")
        singular_max = max(
            float(numpy.linalg.svd(matrix, compute_uv=False)[0])
            for matrix in self.matrix[selected]
        )
        transfer = self.differential_transfer()
        bulk_delay = self.bulk_delay_s()
        bins = 4097
        frequency = numpy.linspace(0.0, min(high_hz, float(self.frequency_hz[-1])), bins)
        interpolated = _interpolate_transfer(frequency, self.frequency_hz, transfer)
        # Remove fitted transport delay before measuring wrapped negative-time
        # energy; otherwise a perfectly causal fractional delay is penalized by
        # the finite-band IFFT representation itself.
        interpolated *= numpy.exp(2j * math.pi * frequency * bulk_delay)
        impulse = numpy.fft.irfft(interpolated)
        energy = numpy.abs(impulse) ** 2
        # irfft index zero is t=0; wrapped energy in the latter half represents
        # negative-time content for this band-limited causality screen.
        negative_ratio = float(numpy.sum(energy[len(energy) // 2:]) / max(numpy.sum(energy), 1e-300))
        return {
            "channel_max_singular_value": singular_max,
            "channel_bulk_delay_s": bulk_delay,
            "channel_negative_time_energy_ratio": negative_ratio,
        }


def _complex(first: float, second: float, data_format: str) -> complex:
    if data_format == "ri":
        return complex(first, second)
    if data_format == "ma":
        return first * complex(math.cos(math.radians(second)), math.sin(math.radians(second)))
    if data_format == "db":
        magnitude = 10 ** (first / 20)
        return magnitude * complex(math.cos(math.radians(second)), math.sin(math.radians(second)))
    raise ValueError(f"unsupported Touchstone format: {data_format}")


def load_s4p(path: str | Path, tx_ports: tuple[int, int] = (1, 2), rx_ports: tuple[int, int] = (3, 4),
             *, port_map: ChannelPortMap | None = None) -> S4PChannel:
    numpy = require_numpy()
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".s4p" or not source.is_file():
        raise FileNotFoundError(f"four-port Touchstone file not found: {source}")
    option: list[str] | None = None
    tokens: list[str] = []
    for raw_line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            raise ValueError(
                "Touchstone 2 bracketed metadata/per-port references are not supported; "
                "export a Touchstone 1.x S4P with one reference impedance"
            )
        if line.startswith("#"):
            option = line[1:].lower().split()
            continue
        tokens.extend(line.replace(",", " ").split())
    if not option or len(option) < 5 or option[1] != "s":
        raise ValueError("Touchstone option line must declare S-parameters")
    unit, _, data_format = option[:3]
    try:
        r_index = option.index("r")
        reference = float(option[r_index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError("Touchstone option line must declare reference impedance") from exc
    width = 1 + 2 * 16
    if len(tokens) % width:
        raise ValueError("Touchstone data is truncated or has unsupported continuation syntax")
    rows = len(tokens) // width
    frequency = numpy.empty(rows, dtype=float)
    matrix = numpy.empty((rows, 4, 4), dtype=complex)
    for row in range(rows):
        values = tokens[row * width:(row + 1) * width]
        frequency[row] = float(values[0]) * _FREQUENCY_SCALE[unit]
        offset = 1
        # Touchstone full-matrix data is ordered one destination row at a
        # time: S11, S12, ... S14, S21, ... S44.  Keeping the loops in that
        # order matters for asymmetric measured channels; reciprocal synthetic
        # fixtures can otherwise hide an accidental matrix transpose.
        for destination_port in range(4):
            for source_port in range(4):
                matrix[row, destination_port, source_port] = _complex(
                    float(values[offset]), float(values[offset + 1]), data_format,
                )
                offset += 2
    if not numpy.all(numpy.diff(frequency) > 0):
        raise ValueError("Touchstone frequencies must be strictly increasing")
    if port_map is not None:
        port_map.validate()
        tx_ports, rx_ports = port_map.tx_ports, port_map.rx_ports
    else:
        ChannelPortMap(*tx_ports, *rx_ports).validate()
    return S4PChannel(source, frequency, matrix, reference, data_format, sha256_file(source), tx_ports, rx_ports)


def validate_s4p_channel(channel: S4PChannel) -> dict[str, float]:
    """Apply the receiver's deterministic preflight gates to a parsed channel."""
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
        **channel.validation_metrics(),
    }
    if metrics["channel_max_singular_value"] > 1.01:
        raise ValueError("channel fails passivity screen: maximum singular value exceeds 1.01")
    if metrics["channel_bulk_delay_s"] < -1e-12:
        raise ValueError("channel has a material negative fitted bulk delay")
    if metrics["channel_negative_time_energy_ratio"] > 0.05:
        raise ValueError("channel fails band-limited causality screen")
    return metrics


def filter_channel(channel: S4PChannel, time_s, differential_v):
    numpy = require_numpy()
    if len(time_s) != len(differential_v) or len(time_s) < 2:
        raise ValueError("channel input time and waveform lengths must match")
    step = float(numpy.median(numpy.diff(time_s)))
    intervals = numpy.diff(time_s)
    if not numpy.allclose(intervals, step, rtol=1e-3, atol=step * 1e-6):
        raise ValueError("channel filtering requires a uniform input timebase")
    padded_length = 1 << (2 * len(differential_v) - 1).bit_length()
    input_spectrum = numpy.fft.rfft(differential_v, padded_length)
    frequencies = numpy.fft.rfftfreq(padded_length, step)
    transfer = channel.differential_transfer()
    interpolated = _interpolate_transfer(frequencies, channel.frequency_hz, transfer)
    output = numpy.fft.irfft(input_spectrum * interpolated, padded_length)
    return output[:len(differential_v)]


def impulse_response(channel: S4PChannel, time_step_s: float, sample_count: int):
    numpy = require_numpy()
    impulse = numpy.zeros(sample_count)
    impulse[0] = 1.0
    time = numpy.arange(sample_count) * time_step_s
    return time, filter_channel(channel, time, impulse)
