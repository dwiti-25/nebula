"""Touchstone four-port parsing and differential channel filtering."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

from .provenance import sha256_file
from .waveform import require_numpy


_FREQUENCY_SCALE = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


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
        numpy = require_numpy()
        transfer = self.differential_transfer()
        real = numpy.interp(target_hz, self.frequency_hz, transfer.real)
        imag = numpy.interp(target_hz, self.frequency_hz, transfer.imag)
        return 20 * math.log10(max(abs(complex(real, imag)), 1e-300))


def _complex(first: float, second: float, data_format: str) -> complex:
    if data_format == "ri":
        return complex(first, second)
    if data_format == "ma":
        return first * complex(math.cos(math.radians(second)), math.sin(math.radians(second)))
    if data_format == "db":
        magnitude = 10 ** (first / 20)
        return magnitude * complex(math.cos(math.radians(second)), math.sin(math.radians(second)))
    raise ValueError(f"unsupported Touchstone format: {data_format}")


def load_s4p(path: str | Path, tx_ports: tuple[int, int] = (1, 2), rx_ports: tuple[int, int] = (3, 4)) -> S4PChannel:
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
            continue
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
        for source_port in range(4):
            for destination_port in range(4):
                matrix[row, destination_port, source_port] = _complex(
                    float(values[offset]), float(values[offset + 1]), data_format,
                )
                offset += 2
    if not numpy.all(numpy.diff(frequency) > 0):
        raise ValueError("Touchstone frequencies must be strictly increasing")
    if sorted((*tx_ports, *rx_ports)) != [1, 2, 3, 4]:
        raise ValueError("port mapping must use each port exactly once")
    return S4PChannel(source, frequency, matrix, reference, data_format, sha256_file(source), tx_ports, rx_ports)


def filter_channel(channel: S4PChannel, time_s, differential_v):
    numpy = require_numpy()
    if len(time_s) != len(differential_v) or len(time_s) < 2:
        raise ValueError("channel input time and waveform lengths must match")
    step = float(numpy.median(numpy.diff(time_s)))
    padded_length = 1 << (2 * len(differential_v) - 1).bit_length()
    input_spectrum = numpy.fft.rfft(differential_v, padded_length)
    frequencies = numpy.fft.rfftfreq(padded_length, step)
    transfer = channel.differential_transfer()
    real = numpy.interp(frequencies, channel.frequency_hz, transfer.real, left=transfer.real[0], right=0.0)
    imag = numpy.interp(frequencies, channel.frequency_hz, transfer.imag, left=transfer.imag[0], right=0.0)
    output = numpy.fft.irfft(input_spectrum * (real + 1j * imag), padded_length)
    return output[:len(differential_v)]


def impulse_response(channel: S4PChannel, time_step_s: float, sample_count: int):
    numpy = require_numpy()
    impulse = numpy.zeros(sample_count)
    impulse[0] = 1.0
    time = numpy.arange(sample_count) * time_step_s
    return time, filter_channel(channel, time, impulse)
