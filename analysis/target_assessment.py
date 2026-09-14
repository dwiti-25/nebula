"""Strict, reward-independent assessment of a design against a target.

The AutoCkt reward intentionally grants its terminal bonus when the summed
relative violation is within a tolerance.  That is useful training behavior,
but it is not an engineering PASS predicate.  Reporting, candidate filtering,
PVT qualification, and the UI use this module instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping

from rl.target_spec import SPEC_DIRECTIONS, SPEC_NAMES, TargetSpec


@dataclass(frozen=True)
class MetricAssessment:
    name: str
    measured: float | None
    target: float
    direction: str
    margin: float | None
    normalized_margin: float | None
    passed: bool


@dataclass(frozen=True)
class TargetAssessment:
    passed: bool
    simulator_success: bool
    metrics: tuple[MetricAssessment, ...]
    missing_metrics: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "simulator_success": self.simulator_success,
            "missing_metrics": list(self.missing_metrics),
            "metrics": [
                {
                    "name": metric.name,
                    "measured": metric.measured,
                    "target": metric.target,
                    "direction": metric.direction,
                    "margin": metric.margin,
                    "normalized_margin": metric.normalized_margin,
                    "passed": metric.passed,
                }
                for metric in self.metrics
            ],
        }


def _normalization_scale(name: str, target: float) -> float:
    """Return a stable engineering scale, including for a zero-margin target."""

    if name == "dfe_min_margin_v":
        return max(abs(target), 0.1)
    return max(abs(target), 1e-12)


def final_violations(metrics, target):
    """Final measurements must independently satisfy the target and RF limits."""
    assessment = assess_target(metrics, target, simulator_success=True)
    failures = [m.name for m in assessment.metrics if not m.passed]
    checks = {
        "hd3_db": lambda x: x < -30.0,
        "input_referred_noise_vrms": lambda x: 0 < x < 0.0015,
        "peaking_db": lambda x: 3 <= x <= 12,
        "dfe_error_count": lambda x: x == 0,
    }
    for name, predicate in checks.items():
        value = metrics.get(name)
        if (not isinstance(value, Real) or isinstance(value, bool)
                or not math.isfinite(value) or not predicate(value)):
            failures.append(name)
    return failures


def assess_target(
    current_metrics: Mapping[str, float],
    target: TargetSpec,
    *,
    simulator_success: bool,
) -> TargetAssessment:
    """Require every target independently and require simulator success.

    A margin is positive when the requirement is exceeded, zero on the exact
    boundary, and negative when violated.  Boundary equality is a strict pass
    because TargetSpec represents inclusive minimum/maximum thresholds.
    """

    assessed: list[MetricAssessment] = []
    missing: list[str] = []
    for name in SPEC_NAMES:
        raw = current_metrics.get(name)
        target_value = float(getattr(target, name))
        direction = SPEC_DIRECTIONS[name]
        if not isinstance(raw, Real) or isinstance(raw, bool) or not math.isfinite(raw):
            missing.append(name)
            assessed.append(MetricAssessment(name, None, target_value, direction, None, None, False))
            continue

        measured = float(raw)
        if direction == "larger_is_better":
            margin = measured - target_value
        elif direction == "smaller_is_better":
            margin = target_value - measured
        else:  # defensive guard against a silently unsupported target schema
            raise ValueError(f"unsupported target direction for {name}: {direction!r}")
        assessed.append(MetricAssessment(
            name=name,
            measured=measured,
            target=target_value,
            direction=direction,
            margin=margin,
            normalized_margin=margin / _normalization_scale(name, target_value),
            passed=margin >= 0.0,
        ))

    passed = simulator_success and not missing and all(metric.passed for metric in assessed)
    return TargetAssessment(passed, simulator_success, tuple(assessed), tuple(missing))
