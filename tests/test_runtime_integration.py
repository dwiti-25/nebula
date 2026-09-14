import math
import unittest
from dataclasses import replace
from unittest.mock import patch
from analysis.target_assessment import assess_target, final_violations
from analysis.final_specification import build_final_specification_report
from analysis.pvt_selection import PVTPointResult, summarize_pvt_results
from rl.target_spec import TargetSpec
from rl.evaluation_cache import make_cached_evaluator
from rl.synthetic_benchmark import synthetic_evaluate_receiver_graded
from simulator.receiver import ReceiverParameters, StageResult
from simulator.config import PVT_GRID
from simulator.runtime import time_budget, remaining_timeout, expired


class AccuracyTests(unittest.TestCase):
    def test_finalist_cli_preserves_target_and_channel(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from experiments.validate_finalist import main
        target = TargetSpec.from_existing_thresholds()
        metrics = {**target.as_dict(), "peaking_db": 6, "dfe_error_count": 0,
                   "hd3_db": -40, "input_referred_noise_vrms": 0.0005}
        good = replace(synthetic_evaluate_receiver_graded(ReceiverParameters()), success=True, metrics=metrics)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = root / "input.json"
            source.write_text(json.dumps({"selection": {"selected": {"parameters": {}}},
                "target": target.as_dict(), "channel": {"path": "explicit.s4p"}}))
            args = ["--input", str(source), "--output", str(root / "out.jsonl"), "--cache", str(root / "cache")]
            with patch("experiments.validate_finalist.evaluate_receiver", return_value=good) as evaluate:
                self.assertEqual(main(args), 0)
                self.assertEqual(evaluate.call_args.kwargs["channel_path"], "explicit.s4p")
            args[3] = str(root / "failed.jsonl")
            with patch("experiments.validate_finalist.evaluate_receiver", return_value=replace(good, metrics={})):
                self.assertEqual(main(args), 1)

    def test_parallel_checkpoint_resume_and_graph(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from experiments.train_autockt import main
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            common = ["train_autockt", "--rl-version", "v3", "--backend", "synthetic",
                      "--workers", "2", "--updates", "1", "--episodes-per-update", "2",
                      "--horizon", "1", "--max-evaluations", "16", "--evaluation-cache"]
            for index in (0, 1):
                args = common + ["--output", str(root / f"train{index}.jsonl"),
                    "--save-full-checkpoint", str(root / f"full{index}.pt"),
                    "--summary-output", str(root / f"summary{index}.json"),
                    "--graph-output", str(root / f"graph{index}.json")]
                if index:
                    args += ["--resume", str(root / "full0.pt")]
                with patch("sys.argv", args):
                    self.assertEqual(main(), 0)
                summary = json.loads((root / f"summary{index}.json").read_text())
                self.assertEqual(summary["workers"], 2)
                self.assertEqual(summary["completed_updates"], 1)
                self.assertLessEqual(summary["total_evaluations"], 16)

    def test_finalist_requires_all_fresh_measurements(self):
        target = TargetSpec.from_existing_thresholds()
        metrics = {**target.as_dict(), "peaking_db": 6, "dfe_error_count": 0,
                   "hd3_db": -40, "input_referred_noise_vrms": 0.0005}
        self.assertFalse(final_violations(metrics, target))
        for name in metrics:
            incomplete = dict(metrics)
            incomplete.pop(name)
            self.assertIn(name, final_violations(incomplete, target))

    def test_concurrent_identical_evaluations_share_one_call(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        entered, release = Event(), Event()
        def evaluate(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(5))
            return synthetic_evaluate_receiver_graded(*args, **kwargs)
        cached = make_cached_evaluator(evaluate)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(cached, ReceiverParameters())
            self.assertTrue(entered.wait(5))
            second = pool.submit(cached, ReceiverParameters())
            release.set()
            self.assertEqual(first.result().metrics, second.result().metrics)
        self.assertEqual(cached.stats.misses, 1)
        self.assertEqual(cached.stats.hits, 1)

    def test_invalid_measurements_cannot_qualify(self):
        target = TargetSpec.from_existing_thresholds()
        for value in (math.inf, -math.inf, math.nan, True, "1.2", None):
            with self.subTest(value=value):
                metrics = target.as_dict()
                metrics["dfe_locked_phase_eye_height_v"] = value
                assessment = assess_target(metrics, target, simulator_success=True)
                self.assertFalse(assessment.passed)

    def test_full_pvt_requires_unique_complete_conditions(self):
        points = [PVTPointResult(c.process_corner.value, c.supply_v, c.temperature_c, True, None) for c in PVT_GRID]
        def verdict(rows):
            report = build_final_specification_report(design_id="x", parameters={}, nominal_metrics={},
                nominal_source="test", pvt_result=summarize_pvt_results("x", rows))
            return report["rows"][-1]["verdict"]
        self.assertEqual(verdict(points), "PASS")
        self.assertEqual(verdict(points[:27]), "NOT CLAIMED")
        self.assertEqual(verdict([points[0]] * 60), "NOT CLAIMED")
        self.assertEqual(verdict(points + [replace(points[0], success=False)]), "FAIL")

    def test_retryable_result_is_retried(self):
        good = synthetic_evaluate_receiver_graded(ReceiverParameters())
        bad = replace(good, success=False, stages=(StageResult("transient", False, 1,
            failure_code="ngspice_timeout", retryable=True),))
        with patch("builtins.print"):
            evaluator = unittest.mock.Mock(side_effect=[bad, good])
            cached = make_cached_evaluator(evaluator)
            cached(ReceiverParameters())
            cached(ReceiverParameters())
            self.assertEqual(evaluator.call_count, 2)

    def test_deadline_is_scoped_and_caps_each_process(self):
        with patch("simulator.runtime.time.monotonic", return_value=100):
            with time_budget(5):
                self.assertEqual(remaining_timeout(30), 5)
                with patch("simulator.runtime.time.monotonic", return_value=106):
                    self.assertTrue(expired())
                    self.assertEqual(remaining_timeout(30), 0)
            self.assertFalse(expired())

    def test_parallel_budget_includes_resets_and_preserves_episodes(self):
        from rl.autockt_env import AutoCktReceiverEnv
        from rl.parameter_grid import build_parameter_grids, verified_initial_indices
        from rl.ppo_agent import PPOAgent
        from rl.autockt_state_v2 import STATE_DIM_V2
        from rl.parallel_rollout import collect_parallel
        from simulator.rl_adapter import ReceiverRLAdapter, RLBudget
        grids = build_parameter_grids()
        adapter = ReceiverRLAdapter(evaluator=synthetic_evaluate_receiver_graded, budget=RLBudget(7))
        env = AutoCktReceiverEnv(target_pool=(TargetSpec.from_hard_target(),),
            initial_indices=verified_initial_indices(grids), horizon=4, adapter=adapter,
            grids=grids, evaluate_on_reset=True, state_schema="v2", use_reward_v2=True)
        transitions, _, logs = collect_parallel(env, PPOAgent(STATE_DIM_V2, 5, seed=3), episodes=3)
        self.assertLessEqual(adapter.total_evaluations, 7)
        self.assertEqual(sum(len(log.steps) for log in logs), len(transitions))
        self.assertTrue(transitions)
        self.assertTrue(transitions[-1].terminated or transitions[-1].truncated)


if __name__ == "__main__":
    unittest.main()
