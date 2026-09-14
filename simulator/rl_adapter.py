"""Dependency-free, deterministic RL contract for the receiver evaluator.

This module deliberately does not depend on Gymnasium.  It provides a stable
contract that can be wrapped by the training framework chosen later without
letting framework-specific behavior leak into circuit evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Callable, Mapping, Sequence

from .config import SimulationConditions
from .receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters, evaluate_receiver


ACTION_SCHEMA_VERSION = 1
OBSERVATION_SCHEMA_VERSION = 1
REWARD_VERSION = "receiver_reward_v1"

ACTION_BOUNDS = (
    ("rload_ohm", 100.0, 10_000.0, "log"),
    ("rdeg_ohm", 10.0, 10_000.0, "log"),
    ("cdeg_f", 10e-15, 10e-12, "log"),
    ("itail_a", 10e-6, 1e-3, "log"),
    ("dfe_tap_v", -0.4, 0.4, "linear"),
)
LEGACY_PARAMETER_NAMES = tuple(name for name, *_ in ACTION_BOUNDS)

METRIC_OBSERVATION_NAMES = (
    "gain_100mhz_db", "gain_2p5ghz_db", "peaking_db",
    "ctle_power_w", "output_common_mode_v", "dfe_eye_height_v",
    "dfe_eye_width_ui", "dfe_min_margin_v", "dfe_error_rate",
    "channel_loss_2p5ghz_db",
)
OBSERVATION_NAMES = METRIC_OBSERVATION_NAMES + tuple(
    f"valid_{name}" for name in METRIC_OBSERVATION_NAMES
)

CONSTRAINT_NAMES = (
    "evaluation_success", "zero_errors", "positive_margin",
    "eye_height_over_100mv", "eye_width_over_0p4ui", "power_under_15mw",
)


@dataclass(frozen=True)
class RLBudget:
    max_evaluations: int = 1000

    def __post_init__(self) -> None:
        if self.max_evaluations <= 0:
            raise ValueError("RL evaluation budget must be positive")


@dataclass(frozen=True)
class RLStep:
    observation: tuple[float, ...]
    constraints: tuple[float, ...]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


def normalized_action_to_parameters(action: Sequence[float], *, version: str = "v1") -> ReceiverParameters:
    from .design_schema import parameter_bounds
    bounds = parameter_bounds(version)
    if len(action) != len(bounds):
        raise ValueError(f"action must contain exactly {len(bounds)} values")
    values: dict[str, float] = {}
    for raw, (name, lower, upper, scale) in zip(action, bounds):
        value = float(raw)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("normalized actions must be finite and lie in [-1, 1]")
        fraction = (value + 1.0) / 2.0
        if scale == "log":
            values[name] = math.exp(math.log(lower) + fraction * math.log(upper / lower))
        else:
            values[name] = lower + fraction * (upper - lower)
    if version == "v3":
        values["mos_multiplier"] = int(round(values["mos_multiplier"]))
        values["mos_width_um"] = round(values["mos_width_um"], 3)
        values["mos_length_um"] = round(values["mos_length_um"], 3)
    return ReceiverParameters(**values)


def _finite_metric(metrics: Mapping[str, object], name: str, default: float = 0.0) -> float:
    try:
        value = float(metrics.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def observation_from_evaluation(evaluation: ReceiverEvaluation) -> tuple[float, ...]:
    metrics = evaluation.metrics
    values: list[float] = []
    valid: list[float] = []
    for name in METRIC_OBSERVATION_NAMES:
        source_name = "dfe_locked_phase_eye_height_v" if name == "dfe_eye_height_v" else name
        raw = metrics.get(source_name)
        try:
            numeric = float(raw)
            present = math.isfinite(numeric)
        except (TypeError, ValueError):
            numeric, present = 0.0, False
        values.append(numeric if present else 0.0)
        valid.append(1.0 if present else 0.0)
    return tuple(values + valid)


def constraints_from_evaluation(evaluation: ReceiverEvaluation) -> tuple[float, ...]:
    metric = evaluation.metrics
    errors = _finite_metric(metric, "dfe_error_count", 1.0)
    margin = _finite_metric(metric, "dfe_min_margin_v", -1.0)
    height = _finite_metric(metric, "dfe_locked_phase_eye_height_v", -1.0)
    width = _finite_metric(metric, "dfe_eye_width_ui", -1.0)
    power = _finite_metric(metric, "ctle_power_w", 1.0)
    # Positive means satisfied; these values are useful to constrained-RL
    # algorithms without hiding exact engineering thresholds.
    return (
        1.0 if evaluation.success else -1.0,
        0.5 - errors,
        margin,
        height - 0.1,
        width - 0.4,
        0.015 - power,
    )


def reward_v1(evaluation: ReceiverEvaluation) -> float:
    """Versioned bounded engineering score; failures cannot beat successes."""
    if not evaluation.success:
        return -100.0
    metrics = evaluation.metrics
    height = _finite_metric(metrics, "dfe_locked_phase_eye_height_v")
    width = _finite_metric(metrics, "dfe_eye_width_ui")
    margin = _finite_metric(metrics, "dfe_min_margin_v")
    power = _finite_metric(metrics, "ctle_power_w", 0.015)
    peaking = _finite_metric(metrics, "peaking_db", 6.0)
    return float(max(-99.0, min(100.0,
        100.0 * height + 10.0 * width + 50.0 * margin
        - 1000.0 * power - abs(peaking - 6.0)
    )))


class ReceiverRLAdapter:
    def __init__(
        self,
        *,
        evaluator: Callable[..., ReceiverEvaluation] = evaluate_receiver,
        conditions: SimulationConditions = SimulationConditions(),
        fidelity: EvaluationFidelity = EvaluationFidelity.TRAINING,
        budget: RLBudget = RLBudget(),
        seed: int = 0,
        evaluator_kwargs: Mapping[str, object] | None = None,
        version: str = "v1",
        grids=None,
    ):
        from .design_schema import parameter_bounds
        self.version = version
        self.grids = grids
        self.evaluation_phase = "candidate"
        self.action_bounds = parameter_bounds(version)
        self.last_evaluation = None
        self.evaluator = evaluator
        self.conditions = conditions
        if fidelity < EvaluationFidelity.TRAINING:
            raise ValueError("RL adapter requires training fidelity or higher for channel/DFE observations")
        self.fidelity = fidelity
        self.budget = budget
        self.evaluator_kwargs = dict(evaluator_kwargs or {})
        self.seed(seed)
        self.evaluations = 0
        self.total_evaluations = 0

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)
        self.random = random.Random(self.seed_value)

    def reset(self, *, seed: int | None = None) -> tuple[tuple[float, ...], dict[str, object]]:
        if seed is not None:
            self.seed(seed)
        self.evaluations = 0
        return ((0.0,) * len(OBSERVATION_NAMES), {
            "seed": self.seed_value,
            "action_schema_version": 3 if self.version == "v3" else ACTION_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "reward_version": REWARD_VERSION,
        })

    def sample_action(self) -> tuple[float, ...]:
        return tuple(self.random.uniform(-1.0, 1.0) for _ in self.action_bounds)

    def step(self, action: Sequence[float]) -> RLStep:
        if self.total_evaluations >= self.budget.max_evaluations:
            raise RuntimeError(
                "RL evaluation budget exhausted; create a new adapter or increase its configured budget"
            )
        parameters = normalized_action_to_parameters(action, version=self.version)
        if self.grids is not None:
            from rl.parameter_grid import quantize_parameters
            parameters = quantize_parameters(parameters, self.grids)
        started = time.perf_counter()
        evaluation = self.evaluator(
            parameters, self.conditions, self.fidelity, **self.evaluator_kwargs,
        )
        elapsed_s = time.perf_counter() - started
        from analysis.run_graph import observe
        observe(evaluation, phase=self.evaluation_phase, elapsed_s=elapsed_s)
        self.last_evaluation = evaluation
        self.evaluations += 1
        self.total_evaluations += 1
        truncated = self.total_evaluations >= self.budget.max_evaluations
        return RLStep(
            observation_from_evaluation(evaluation),
            constraints_from_evaluation(evaluation),
            reward_v1(evaluation),
            False,
            truncated,
            {
                "evaluation_id": evaluation.evaluation_id,
                "evaluation_count": self.evaluations,
                "total_evaluation_count": self.total_evaluations,
                # Preserve the v1/v2 five-parameter event contract exactly.
                # Expanded MOS sizing uses a separately versioned adapter/schema.
                "parameters": {name: getattr(parameters, name) for name, *_ in self.action_bounds},
                "reward_version": REWARD_VERSION,
                "action_schema_version": 3 if self.version == "v3" else ACTION_SCHEMA_VERSION,
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "failure_stage": evaluation.failed_stage,
                "raw_metrics": dict(evaluation.metrics),
                "cache_hit": evaluation.cache_hit,
                "runtime_s": elapsed_s,
            },
        )
