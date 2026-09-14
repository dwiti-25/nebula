"""Opt-in, in-process evaluation cache for the RL training loop.

Runtime/simulation-efficiency scope only. Does NOT modify
simulator/rl_adapter.py or simulator/receiver.py -- ReceiverRLAdapter
already accepts any callable matching evaluate_receiver's call signature
via its `evaluator=` constructor parameter (the same extension point
synthetic_evaluate_receiver already uses), so caching is implemented
purely as a wrapper around that callable, entirely within rl/.

Cache key: every field of ReceiverParameters + every field of
SimulationConditions + fidelity + the evaluator_kwargs the wrapped
evaluator was called with. This covers every input evaluate_receiver's
own call signature exposes as capable of changing its result. The RL
"target" (TargetSpec) is deliberately excluded: evaluate_receiver's
signature never receives a target, so it cannot affect the simulator's
output -- including it would only reduce the hit rate without adding
safety. A cache is scoped to one wrapper instance (one training run);
do not share a single instance across runs configured with different
evaluator_kwargs, conditions policies, or evaluator callables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable
from threading import RLock
from concurrent.futures import Future

from simulator.config import SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters

Evaluator = Callable[..., ReceiverEvaluation]


def _kwargs_key(kwargs: dict) -> tuple:
    return tuple(sorted((name, repr(value)) for name, value in kwargs.items()))


def cache_key(
    parameters: ReceiverParameters,
    conditions: SimulationConditions,
    fidelity: EvaluationFidelity,
    **kwargs: object,
) -> tuple:
    return (
        tuple(sorted(asdict(parameters).items())),
        tuple(sorted(asdict(conditions).items())),
        int(fidelity),
        _kwargs_key(kwargs),
    )


@dataclass
class EvaluationCacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass
class CachedEvaluator:
    """Callable wrapper matching evaluate_receiver's call signature.

    Drop-in replacement for the `evaluator=` argument of
    simulator.rl_adapter.ReceiverRLAdapter -- no adapter/env changes
    required. Reuses ReceiverEvaluation's existing `cache_hit` field
    (already defined in simulator/receiver.py for exactly this purpose)
    rather than inventing new semantics.
    """

    evaluator: Evaluator
    stats: EvaluationCacheStats = field(default_factory=EvaluationCacheStats)
    _store: dict = field(default_factory=dict, repr=False)
    _lock: object = field(default_factory=RLock, repr=False)
    _pending: dict = field(default_factory=dict, repr=False)

    def __call__(
        self,
        parameters: ReceiverParameters,
        conditions: SimulationConditions = SimulationConditions(),
        fidelity: EvaluationFidelity = EvaluationFidelity.TRAINING,
        **kwargs: object,
    ) -> ReceiverEvaluation:
        key = cache_key(parameters, conditions, fidelity, **kwargs)
        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                self.stats.hits += 1
                return replace(cached, cache_hit=True)
            pending = self._pending.get(key)
            owner = pending is None
            if owner:
                pending = self._pending[key] = Future()
                self.stats.misses += 1
            else:
                self.stats.hits += 1
        if not owner:
            return replace(pending.result(), cache_hit=True)
        try:
            result = self.evaluator(parameters, conditions, fidelity, **kwargs)
            from simulator.receiver import _evaluation_is_cacheable
            with self._lock:
                if _evaluation_is_cacheable(result):
                    self._store[key] = result
                self._pending.pop(key)
                pending.set_result(result)
            return result
        except BaseException as error:
            with self._lock:
                self._pending.pop(key, None)
                pending.set_exception(error)
            raise

    def __len__(self) -> int:
        return len(self._store)

    def approx_memory_bytes(self) -> int:
        """Shallow sys.getsizeof over stored keys/values -- a bounded,
        reproducible proxy, not a precise deep-memory measurement."""
        import sys

        return sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self._store.items())


def make_cached_evaluator(evaluator: Evaluator) -> CachedEvaluator:
    return CachedEvaluator(evaluator=evaluator)
