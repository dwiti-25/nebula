"""Small, deterministic statistical primitives for multi-seed RL reports."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def interquartile_mean(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("at least one value is required")
    trim = len(ordered) // 4
    core = ordered[trim:len(ordered) - trim] if trim else ordered
    return sum(core) / len(core)


def bootstrap_interval(values: Sequence[float], *, seed: int = 0,
                       samples: int = 2000, confidence: float = 0.95) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("at least two independent seeds are required for an interval")
    rng = random.Random(seed)
    draws = sorted(interquartile_mean(rng.choices(values, k=len(values))) for _ in range(samples))
    alpha = (1.0 - confidence) / 2.0
    return draws[max(0, math.floor(alpha * samples))], draws[min(samples - 1, math.ceil((1 - alpha) * samples) - 1)]


def probability_of_improvement(candidate: Sequence[float], baseline: Sequence[float]) -> float:
    if not candidate or not baseline:
        raise ValueError("both samples are required")
    wins = sum(a > b for a in candidate for b in baseline)
    ties = sum(a == b for a in candidate for b in baseline)
    return (wins + 0.5 * ties) / (len(candidate) * len(baseline))


def performance_profile(runs: Sequence[float], thresholds: Sequence[float]) -> list[float]:
    if not runs:
        raise ValueError("at least one run is required")
    return [sum(value >= threshold for value in runs) / len(runs) for threshold in thresholds]

