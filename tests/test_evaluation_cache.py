"""Tests for rl/evaluation_cache.py -- the opt-in, in-process evaluator
cache used by the runtime/simulation-efficiency work. Synthetic-only,
SPICE-free.
"""

from __future__ import annotations

import unittest

from rl.evaluation_cache import CachedEvaluator, EvaluationCacheStats, cache_key, make_cached_evaluator
from rl.synthetic_benchmark import synthetic_evaluate_receiver_graded
from simulator.config import ProcessCorner, SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverParameters


class CacheKeyTests(unittest.TestCase):
    def test_identical_inputs_produce_identical_key(self):
        p = ReceiverParameters(rload_ohm=500.0)
        c = SimulationConditions()
        k1 = cache_key(p, c, EvaluationFidelity.TRAINING)
        k2 = cache_key(p, c, EvaluationFidelity.TRAINING)
        self.assertEqual(k1, k2)

    def test_different_parameters_produce_different_key(self):
        c = SimulationConditions()
        k1 = cache_key(ReceiverParameters(rload_ohm=500.0), c, EvaluationFidelity.TRAINING)
        k2 = cache_key(ReceiverParameters(rload_ohm=501.0), c, EvaluationFidelity.TRAINING)
        self.assertNotEqual(k1, k2)

    def test_different_conditions_produce_different_key(self):
        p = ReceiverParameters()
        k1 = cache_key(p, SimulationConditions(temperature_c=27.0), EvaluationFidelity.TRAINING)
        k2 = cache_key(p, SimulationConditions(temperature_c=85.0), EvaluationFidelity.TRAINING)
        self.assertNotEqual(k1, k2)

    def test_different_process_corner_produces_different_key(self):
        p = ReceiverParameters()
        k1 = cache_key(p, SimulationConditions(process_corner=ProcessCorner.TT), EvaluationFidelity.TRAINING)
        k2 = cache_key(p, SimulationConditions(process_corner=ProcessCorner.SS), EvaluationFidelity.TRAINING)
        self.assertNotEqual(k1, k2)

    def test_different_fidelity_produces_different_key(self):
        p, c = ReceiverParameters(), SimulationConditions()
        k1 = cache_key(p, c, EvaluationFidelity.TRAINING)
        k2 = cache_key(p, c, EvaluationFidelity.CANDIDATE)
        self.assertNotEqual(k1, k2)

    def test_different_kwargs_produce_different_key(self):
        p, c = ReceiverParameters(), SimulationConditions()
        k1 = cache_key(p, c, EvaluationFidelity.TRAINING, channel_path="a")
        k2 = cache_key(p, c, EvaluationFidelity.TRAINING, channel_path="b")
        self.assertNotEqual(k1, k2)


class CachedEvaluatorTests(unittest.TestCase):
    def test_second_identical_call_is_a_hit_and_returns_equal_metrics(self):
        cached = make_cached_evaluator(synthetic_evaluate_receiver_graded)
        p, c, f = ReceiverParameters(rload_ohm=500.0), SimulationConditions(), EvaluationFidelity.TRAINING
        first = cached(p, c, f)
        second = cached(p, c, f)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(first.success, second.success)
        self.assertEqual(cached.stats.hits, 1)
        self.assertEqual(cached.stats.misses, 1)

    def test_different_parameters_both_miss(self):
        cached = make_cached_evaluator(synthetic_evaluate_receiver_graded)
        c, f = SimulationConditions(), EvaluationFidelity.TRAINING
        cached(ReceiverParameters(rload_ohm=500.0), c, f)
        cached(ReceiverParameters(rload_ohm=600.0), c, f)
        self.assertEqual(cached.stats.hits, 0)
        self.assertEqual(cached.stats.misses, 2)

    def test_underlying_evaluator_called_exactly_once_per_unique_key(self):
        calls = []

        def counting_evaluator(parameters, conditions=SimulationConditions(),
                                fidelity=EvaluationFidelity.TRAINING, **_kwargs):
            calls.append(parameters)
            return synthetic_evaluate_receiver_graded(parameters, conditions, fidelity, **_kwargs)

        cached = make_cached_evaluator(counting_evaluator)
        p, c, f = ReceiverParameters(rload_ohm=500.0), SimulationConditions(), EvaluationFidelity.TRAINING
        for _ in range(5):
            cached(p, c, f)
        self.assertEqual(len(calls), 1)

    def test_does_not_cache_across_different_conditions(self):
        cached = make_cached_evaluator(synthetic_evaluate_receiver_graded)
        p, f = ReceiverParameters(rload_ohm=500.0), EvaluationFidelity.TRAINING
        cached(p, SimulationConditions(temperature_c=27.0), f)
        result = cached(p, SimulationConditions(temperature_c=85.0), f)
        self.assertFalse(result.cache_hit)
        self.assertEqual(cached.stats.misses, 2)

    def test_len_and_memory_estimate_are_non_negative_and_track_unique_entries(self):
        cached = make_cached_evaluator(synthetic_evaluate_receiver_graded)
        c, f = SimulationConditions(), EvaluationFidelity.TRAINING
        cached(ReceiverParameters(rload_ohm=500.0), c, f)
        cached(ReceiverParameters(rload_ohm=500.0), c, f)
        cached(ReceiverParameters(rload_ohm=600.0), c, f)
        self.assertEqual(len(cached), 2)
        self.assertGreater(cached.approx_memory_bytes(), 0)

    def test_hit_rate_computed_correctly(self):
        stats = EvaluationCacheStats(hits=3, misses=1)
        self.assertAlmostEqual(stats.hit_rate, 0.75)

    def test_hit_rate_is_zero_with_no_calls(self):
        self.assertEqual(EvaluationCacheStats().hit_rate, 0.0)

    def test_wrapped_evaluator_is_drop_in_compatible_with_adapter_call_convention(self):
        from simulator.rl_adapter import ReceiverRLAdapter, RLBudget

        cached = make_cached_evaluator(synthetic_evaluate_receiver_graded)
        adapter = ReceiverRLAdapter(evaluator=cached, budget=RLBudget(10))
        adapter.reset()
        step = adapter.step((0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertIsInstance(step.observation, tuple)


if __name__ == "__main__":
    unittest.main()
