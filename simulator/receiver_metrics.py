"""Behavioral sampling, one-tap DFE, and eye measurements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .waveform import require_numpy


class DFEMode(str, Enum):
    TRAINING = "training"
    DECISION_DIRECTED = "decision_directed"


@dataclass(frozen=True)
class SamplingResult:
    phase_s: float
    phase_ui: float
    sample_times_s: object
    raw_samples_v: object


@dataclass(frozen=True)
class DFEResult:
    mode: DFEMode
    tap_v: float
    corrected_samples_v: object
    decisions: object
    expected_bits: object
    margins_v: object
    error_indices: object
    error_count: int
    error_propagation_events: int


def _one_tap_dfe(samples_v, expected_bits_value, tap_v, threshold_v,
                 initial_previous_bit, training_stop: int | None) -> DFEResult:
    """Run one stateful DFE, optionally switching from training to decisions."""
    numpy = require_numpy()
    samples = numpy.asarray(samples_v, dtype=float)
    expected = numpy.asarray(expected_bits_value, dtype=int)
    if len(samples) != len(expected):
        raise ValueError("samples and expected bits must have the same length")
    if initial_previous_bit not in (0, 1):
        raise ValueError("initial previous bit must be zero or one")
    if training_stop is not None and not 0 <= training_stop <= len(samples):
        raise ValueError("training stop must lie within the sample sequence")
    corrected = numpy.empty_like(samples)
    decisions = numpy.empty_like(expected)
    previous_sign = 1.0 if initial_previous_bit else -1.0
    propagation = 0
    previous_error = False
    for index, sample in enumerate(samples):
        corrected[index] = sample - tap_v * previous_sign
        decisions[index] = int(corrected[index] >= threshold_v)
        error = decisions[index] != expected[index]
        if error and previous_error:
            propagation += 1
        previous_error = bool(error)
        known_bit = training_stop is None or index < training_stop
        feedback_bit = expected[index] if known_bit else decisions[index]
        previous_sign = 1.0 if feedback_bit else -1.0
    signs = expected_signs(expected)
    margins = signs * (corrected - threshold_v)
    errors = numpy.flatnonzero(decisions != expected)
    mode = DFEMode.TRAINING if training_stop is None else DFEMode.DECISION_DIRECTED
    return DFEResult(mode, float(tap_v), corrected, decisions, expected, margins,
                     errors, int(len(errors)), propagation)


def expected_signs(bits):
    numpy = require_numpy()
    values = numpy.asarray(bits, dtype=int)
    return numpy.where(values > 0, 1.0, -1.0)


def sample_at_phase(time_s, waveform_v, bit_count: int, ui_s: float, phase_s: float) -> SamplingResult:
    numpy = require_numpy()
    sample_times = numpy.arange(bit_count, dtype=float) * ui_s + phase_s
    if sample_times[-1] > time_s[-1]:
        raise ValueError("waveform does not cover all requested sample times")
    samples = numpy.interp(sample_times, time_s, waveform_v)
    return SamplingResult(phase_s, phase_s / ui_s, sample_times, samples)


def choose_sampling_phase(
    time_s,
    waveform_v,
    bits,
    ui_s: float,
    training_start: int,
    training_stop: int,
    phase_step_s: float,
) -> SamplingResult:
    numpy = require_numpy()
    signs = expected_signs(bits)
    best: SamplingResult | None = None
    best_score = -float("inf")
    phases = numpy.arange(phase_step_s, ui_s, phase_step_s)
    for phase in phases:
        candidate = sample_at_phase(time_s, waveform_v, len(bits), ui_s, float(phase))
        margins = signs[training_start:training_stop] * candidate.raw_samples_v[training_start:training_stop]
        score = float(numpy.percentile(margins, 10))
        if score > best_score:
            best_score = score
            best = candidate
    if best is None:
        raise ValueError("no valid sampling phase was evaluated")
    return best


def apply_one_tap_dfe(
    samples_v,
    expected_bits_value,
    tap_v: float,
    mode: DFEMode = DFEMode.DECISION_DIRECTED,
    threshold_v: float = 0.0,
    initial_previous_bit: int = 0,
) -> DFEResult:
    training_stop = None if mode is DFEMode.TRAINING else 0
    return _one_tap_dfe(samples_v, expected_bits_value, tap_v, threshold_v,
                        initial_previous_bit, training_stop)


def apply_hybrid_one_tap_dfe(
    samples_v, expected_bits_value, tap_v: float, training_stop: int,
    threshold_v: float = 0.0, initial_previous_bit: int = 0,
) -> DFEResult:
    """Use known bits through ``training_stop``, then preserve state for DD mode."""
    return _one_tap_dfe(samples_v, expected_bits_value, tap_v, threshold_v,
                        initial_previous_bit, training_stop)


def _contiguous_open_width(open_mask, anchor: int, phase_step_s: float, ui_s: float) -> float:
    """Return the circular contiguous opening containing the selected phase."""
    numpy = require_numpy()
    mask = numpy.asarray(open_mask, dtype=bool)
    count = len(mask)
    if not count or not mask[anchor]:
        return 0.0
    if numpy.all(mask):
        return 1.0
    width = 1
    index = (anchor - 1) % count
    while index != anchor and mask[index]:
        width += 1
        index = (index - 1) % count
    index = (anchor + 1) % count
    while index != anchor and mask[index] and width < count:
        width += 1
        index = (index + 1) % count
    return min(float(width * phase_step_s / ui_s), 1.0)


def dfe_eye_metrics(
    time_s,
    waveform_v,
    bits,
    ui_s: float,
    phase_step_s: float,
    start_bit: int,
    stop_bit: int,
    tap_v: float,
    minimum_height_v: float = 0.1,
    training_stop: int = 0,
    locked_phase_s: float | None = None,
) -> dict[str, float]:
    """Measure horizontal and vertical opening after a one-tap DFE."""
    numpy = require_numpy()
    bits_array = numpy.asarray(bits, dtype=int)
    phases = numpy.arange(phase_step_s, ui_s, phase_step_s)
    heights: list[float] = []
    for phase in phases:
        raw = sample_at_phase(time_s, waveform_v, len(bits_array), ui_s, float(phase)).raw_samples_v
        corrected = apply_hybrid_one_tap_dfe(
            raw, bits_array, tap_v, training_stop,
        ).corrected_samples_v[start_bit:stop_bit]
        section_bits = bits_array[start_bit:stop_bit]
        ones = corrected[section_bits == 1]
        zeros = corrected[section_bits == 0]
        if not len(ones) or not len(zeros):
            heights.append(float("nan"))
        else:
            heights.append(float(numpy.percentile(ones, 10) - numpy.percentile(zeros, 90)))
    values = numpy.asarray(heights)
    valid = numpy.isfinite(values)
    if not numpy.any(valid):
        raise ValueError("DFE eye calculation has no phases containing both symbols")
    best_index = int(numpy.nanargmax(values))
    open_mask = valid & (values >= minimum_height_v)
    anchor = best_index if locked_phase_s is None else int(numpy.argmin(numpy.abs(phases - locked_phase_s)))
    return {
        "dfe_eye_height_v": float(values[best_index]),
        "dfe_eye_width_ui": _contiguous_open_width(open_mask, anchor, phase_step_s, ui_s),
        "dfe_eye_best_phase_ui": float(phases[best_index] / ui_s),
        "dfe_locked_phase_eye_height_v": float(values[anchor]),
    }


def eye_metrics(
    time_s,
    waveform_v,
    bits,
    ui_s: float,
    phase_step_s: float,
    start_bit: int,
    stop_bit: int,
    minimum_height_v: float = 0.1,
) -> dict[str, float]:
    numpy = require_numpy()
    bits_array = numpy.asarray(bits, dtype=int)
    phase_values = numpy.arange(phase_step_s, ui_s, phase_step_s)
    heights: list[float] = []
    centers: list[float] = []
    for phase in phase_values:
        sampled = sample_at_phase(time_s, waveform_v, len(bits_array), ui_s, float(phase)).raw_samples_v
        section = sampled[start_bit:stop_bit]
        section_bits = bits_array[start_bit:stop_bit]
        ones = section[section_bits == 1]
        zeros = section[section_bits == 0]
        if not len(ones) or not len(zeros):
            heights.append(float("nan"))
            centers.append(float("nan"))
            continue
        lower_one = float(numpy.percentile(ones, 10))
        upper_zero = float(numpy.percentile(zeros, 90))
        heights.append(lower_one - upper_zero)
        centers.append((lower_one + upper_zero) / 2)
    height_values = numpy.asarray(heights)
    valid = numpy.isfinite(height_values)
    if not numpy.any(valid):
        raise ValueError("eye calculation has no phases containing both symbols")
    best_index = int(numpy.nanargmax(height_values))
    open_mask = valid & (height_values >= minimum_height_v)
    width_ui = float(numpy.count_nonzero(open_mask) * phase_step_s / ui_s)
    return {
        "eye_height_v": float(height_values[best_index]),
        "eye_width_ui": min(width_ui, 1.0),
        "eye_best_phase_s": float(phase_values[best_index]),
        "eye_best_phase_ui": float(phase_values[best_index] / ui_s),
        "eye_vertical_center_v": float(centers[best_index]),
    }
