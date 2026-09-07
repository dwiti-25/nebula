"""SPICE-free tests for experiments/broader_generalization_check.py --
PPO model-improvement study, improvement #2. Uses a mocked evaluator
(ReceiverRLAdapter(evaluator=...)) throughout, matching the established
pattern in tests/test_autockt_rl.py -- no ngspice, no PDK.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from simulator.config import SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters
from simulator.rl_adapter import METRIC_OBSERVATION_NAMES

from experiments.broader_generalization_check import HELD_OUT_TARGETS, _main, hard, trivial
from rl.target_spec import EXISTING_THRESHOLDS


def _fake_evaluation(success: bool = True, *, metrics: dict | None = None) -> ReceiverEvaluation:
    values = {name: 0.0 for name in METRIC_OBSERVATION_NAMES}
    values["dfe_locked_phase_eye_height_v"] = values.pop("dfe_eye_height_v")
    values.update(metrics or {})
    return ReceiverEvaluation(
        success, ReceiverParameters(), SimulationConditions(), EvaluationFidelity.TRAINING,
        (), values, None if success else "dc", 0.0, "fake", {},
    )


class HeldOutTargetsTests(unittest.TestCase):
    def test_three_targets_defined(self):
        self.assertEqual(set(HELD_OUT_TARGETS), {"moderate_25pct", "moderate_75pct", "power_focused"})

    def test_moderate_targets_are_genuinely_between_trivial_and_hard(self):
        for name in ("moderate_25pct", "moderate_75pct"):
            target = HELD_OUT_TARGETS[name]
            for spec_name in ("dfe_locked_phase_eye_height_v", "dfe_eye_width_ui", "dfe_min_margin_v"):
                lo, hi = sorted((getattr(trivial, spec_name), getattr(hard, spec_name)))
                self.assertLessEqual(lo, getattr(target, spec_name))
                self.assertLessEqual(getattr(target, spec_name), hi)

    def test_none_of_the_new_targets_equal_the_existing_midpoint(self):
        existing_midpoint = {
            "dfe_locked_phase_eye_height_v": 0.45, "dfe_eye_width_ui": 0.5,
            "dfe_min_margin_v": 0.175, "ctle_power_w": 0.015,
        }
        for target in HELD_OUT_TARGETS.values():
            self.assertNotEqual(target.as_dict(), existing_midpoint)

    def test_power_focused_target_varies_power_unlike_every_prior_generalization_target(self):
        # trivial, hard, and their midpoint all share ctle_power_w == 0.015
        # -- no prior generalization check ever varied this dimension.
        target = HELD_OUT_TARGETS["power_focused"]
        self.assertNotEqual(target.ctle_power_w, EXISTING_THRESHOLDS["ctle_power_w"])
        self.assertLess(target.ctle_power_w, EXISTING_THRESHOLDS["ctle_power_w"])

    def test_power_focused_target_is_plausibly_achievable_not_extrapolated(self):
        # Design A's own real measured power (~0.00109 W, docs/autockt-mapping.md)
        # is below this target's 0.002 W ceiling -- not extrapolated beyond
        # demonstrated real-SPICE behavior.
        target = HELD_OUT_TARGETS["power_focused"]
        design_a_power_w = 0.0010850994
        self.assertGreater(target.ctle_power_w, design_a_power_w)


class MainCliTests(unittest.TestCase):
    def test_refuses_to_overwrite_existing_output(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.jsonl"
            output.write_text("already here")
            argv = ["broader_generalization_check.py", "--output", str(output)]
            with patch("sys.argv", argv):
                with self.assertRaises(FileExistsError):
                    _main()

    def test_runs_end_to_end_with_a_fake_checkpoint_and_mocked_evaluator(self):
        import torch
        from rl.ppo_agent import PPOAgent
        from rl.autockt_state import STATE_DIM
        from rl.parameter_grid import PARAMETER_NAMES

        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "fake_policy.pt"
            fresh_agent = PPOAgent(state_dim=STATE_DIM, num_heads=len(PARAMETER_NAMES), seed=1)
            torch.save(fresh_agent.policy.state_dict(), checkpoint_path)

            output = Path(tmp) / "out.jsonl"
            argv = [
                "broader_generalization_check.py", "--checkpoint", str(checkpoint_path),
                "--horizon", "1", "--episodes", "1", "--eval-seed", "1", "--agent-seed", "1",
                "--output", str(output),
            ]

            def fake_evaluator(parameters, conditions, fidelity, **kwargs):
                return _fake_evaluation(True, metrics={
                    "dfe_locked_phase_eye_height_v": 1.5, "dfe_eye_width_ui": 0.85,
                    "dfe_min_margin_v": 0.5, "ctle_power_w": 0.001,
                })

            with patch("experiments.broader_generalization_check.ReceiverRLAdapter") as mock_adapter_cls:
                from simulator.rl_adapter import ReceiverRLAdapter as RealAdapter
                mock_adapter_cls.side_effect = lambda **kw: RealAdapter(evaluator=fake_evaluator, **kw)
                with patch("sys.argv", argv):
                    rc = _main()

            self.assertEqual(rc, 0)
            self.assertTrue(output.is_file())
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            summary = rows[-1]
            self.assertEqual(summary["row_type"], "summary")
            self.assertEqual(set(summary["results"]), {"moderate_25pct", "moderate_75pct", "power_focused"})
            self.assertIn("existing_midpoint_generalization_reference", summary)


if __name__ == "__main__":
    unittest.main()
