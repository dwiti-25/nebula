"""V3 contracts, physical propagation, safe checkpoints and recorded graphs."""
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

import torch
from rl.parameter_grid import build_parameter_grids, verified_initial_indices
from rl.autockt_action import indices_to_normalized_action, apply_action
from rl.autockt_env import AutoCktReceiverEnv
from rl.autockt_v3 import STATE_DIM_V3, reward_v3
from rl.synthetic_v3 import synthetic_evaluate_receiver_v3
from rl.runtime_contract import runtime_contract, validate_inference_contract, grids_from_contract
from rl.ppo_agent import PPOAgent
from rl.checkpoint import export_inference_only, save_full_checkpoint, load_full_checkpoint, CheckpointIncompatibleError
from rl.target_spec import TargetSpec
from simulator.rl_adapter import ReceiverRLAdapter, RLBudget
from simulator.receiver import ReceiverParameters
from rl.evaluation_cache import make_cached_evaluator
from analysis.run_graph import RunGraph, ACTIVE, render_svg
from analysis.run_evidence import build_run_dashboard
from analysis.plot_renderer import render_chart

class V3IntegrationTests(unittest.TestCase):
    def make_env(self, budget=10):
        grids = build_parameter_grids(version="v3")
        adapter = ReceiverRLAdapter(version="v3", evaluator=synthetic_evaluate_receiver_v3, budget=RLBudget(budget))
        return AutoCktReceiverEnv(target_pool=(TargetSpec.from_existing_thresholds(),),
            initial_indices=verified_initial_indices(grids), horizon=2, grids=grids, adapter=adapter,
            state_schema="v3", evaluate_on_reset=True)

    def test_legacy_unchanged_v3_eight_integer_grid(self):
        self.assertEqual(len(build_parameter_grids()), 5)
        grids = build_parameter_grids(version="v3")
        self.assertEqual(len(grids), 8)
        self.assertEqual(grids["mos_multiplier"].values, tuple(range(1, 17)))
        indices = verified_initial_indices(grids)
        self.assertEqual(len(apply_action(indices, (1,) * 8, grids)), 8)
        env = self.make_env()
        state, _ = env.reset()
        self.assertEqual(len(state), STATE_DIM_V3)
        out = env.step((1,) * 8)
        self.assertEqual(out.info["parameters"]["mos_width_um"], 10.0)
        self.assertEqual(out.info["parameters"]["mos_multiplier"], 1)
        self.assertEqual(env.adapter.total_evaluations, 2)

    def test_mos_actions_reach_evaluator(self):
        env = self.make_env()
        env.reset()
        a = env.step((1,) * 8)
        env.reset()
        b = env.step((1,) * 5 + (2, 2, 2))
        for key in ("mos_width_um", "mos_length_um", "mos_multiplier"):
            self.assertNotEqual(a.info["parameters"][key], b.info["parameters"][key])
        self.assertNotEqual(a.info["metrics"], b.info["metrics"])

    def test_reward_rejects_missing_peaking_errors_and_simulation_failure(self):
        target = TargetSpec.from_existing_thresholds()
        metrics = dict(target.as_dict(), peaking_db=6, dfe_error_count=0)
        self.assertTrue(reward_v3(metrics, target, success=True, failure_stage=None).strict_target_pass)
        for key in metrics:
            missing = dict(metrics); del missing[key]
            self.assertFalse(reward_v3(missing, target, success=True, failure_stage=None).strict_target_pass)
        self.assertFalse(reward_v3(metrics, target, success=False, failure_stage="transient").strict_target_pass)
        metrics["peaking_db"] = 15
        self.assertLess(reward_v3(metrics, target, success=True, failure_stage=None).total, 0)

    def test_safe_export_and_contract_validation(self):
        grids = build_parameter_grids(version="v3")
        contract = runtime_contract("v3", grids, backend="synthetic")
        agent = PPOAgent(state_dim=STATE_DIM_V3, num_heads=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            export_inference_only(path, agent=agent, metadata=contract)
            payload = torch.load(path, weights_only=True)
            validate_inference_contract(payload["metadata"], contract)
            self.assertEqual(grids_from_contract(contract), grids)
            bad = dict(contract, action_deltas=[-1, 0, 1])
            with self.assertRaises(ValueError):
                validate_inference_contract(bad, contract)
            with self.assertRaises(ValueError):
                validate_inference_contract(None, contract)
            path = Path(tmp) / "full.pt"
            save_full_checkpoint(path, agent=agent, update_count=1, evaluation_count=3,
                state_schema="v3", reward_schema="v3", grid_points=21, grid_spacing="linear", target_ids=("a",),
                extra_hyperparameters={"runtime_contract": contract})
            torch.load(path, weights_only=True)  # no NumPy pickle allowlisting
            before = {k: v.clone() for k, v in agent.policy.state_dict().items()}
            with self.assertRaises(CheckpointIncompatibleError):
                load_full_checkpoint(path, agent=agent, state_schema="v3", reward_schema="v3", expected_contract=dict(contract, backend="real"))
            self.assertTrue(all(torch.equal(v, agent.policy.state_dict()[k]) for k, v in before.items()))
            load_full_checkpoint(path, agent=agent, state_schema="v3", reward_schema="v3", expected_contract=contract)

    def test_graph_records_cache_origin_and_renders_missing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.graph.json"
            graph = RunGraph({"backend": "synthetic", "untrusted": "<script>"}, path)
            token = ACTIVE.set(graph)
            try:
                env = self.make_env()
                env.adapter.evaluator = make_cached_evaluator(synthetic_evaluate_receiver_v3)
                env.reset(); env.step((1,) * 8)
                graph.finish({"selection": {"selected": None}})
            finally:
                ACTIVE.reset(token)
            data = json.loads(path.read_text())
            self.assertEqual(data["evaluation_requests"], 2)
            self.assertEqual(data["cache_hits"], 1)
            self.assertTrue(any(e["relation"] == "cache_reuse" for e in data["edges"]))
            self.assertNotIn("<script>", render_svg(data))
            dashboard = build_run_dashboard(Path(tmp) / "run.events.jsonl")
            self.assertTrue(dashboard["charts"])
            for chart in dashboard["charts"]:
                self.assertIn(b"<svg", render_chart(chart))
            self.assertIn(b"<svg", render_chart({"id": "missing", "type": "line", "series": [{"values": [1, None, 2]}]}))

    def test_pipeline_v3_policy_grid_restored_and_graph_json_serializable(self):
        from experiments.run_autockt_pipeline import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v3.pt"
            grids = build_parameter_grids(5, spacing="linear", version="v3")
            contract = runtime_contract("v3", grids, backend="synthetic")
            agent = PPOAgent(state_dim=STATE_DIM_V3, num_heads=8)
            export_inference_only(path, agent=agent, metadata=contract)
            result = run_pipeline(target=TargetSpec.from_existing_thresholds(), checkpoint_path=path,
                rl_version="v3", backend="synthetic", episodes=1, horizon=1)
            json.dumps(result)
            evaluations = [n for n in result["execution_graph"]["nodes"] if n["kind"] == "evaluation"]
            self.assertEqual(len(evaluations), 2)
            self.assertEqual(len(evaluations[0]["detail"]["parameters"]), 8)

    def test_ui_train_argv_and_v3_enabled(self):
        from experiments.web_ui import _build_argv, _validate_request, INDEX_HTML
        payload = dict(workflow="train", updates=1, target_mode="trivial", backend="real", rl_version="v3",
            channel_id="ieee802_reference", episodes=1, horizon=1, pvt_condition_set="none", trade_off_preference="balanced")
        self.assertEqual(_validate_request(payload), [])
        args = _build_argv(payload, output_path=Path("results/run.json"), schematic_path=Path("results/run.spice"))
        self.assertIn("experiments.train_autockt", args)
        self.assertIn("--graph-output", args)
        self.assertNotIn('value="v3" disabled', INDEX_HTML)

    def test_eye_diagram_renderer_writes_svg_png_and_source_data(self):
        from analysis.eye_diagram import render_eye_diagram
        capture = {
            "plot_semantics": {"ctle": "SPICE", "dfe": "behavioral samples"},
            "conditions": {"process_corner": "tt", "supply_v": 1.8, "temperature_c": 27},
            "channel": {"checksum": "abc"},
            "metrics": {"dfe_locked_phase_eye_height_v": 0.2, "dfe_eye_width_ui": 0.5,
                        "dfe_error_count": 0},
            "locked_phase_ui": 0.5,
            "ctle_traces": [{"phase_ui": [0, 1, 2], "voltage_v": [-0.2, 0.2, -0.2]}],
            "dfe_points": [{"phase_ui": 0.5, "voltage_v": -0.15, "expected_bit": 0},
                           {"phase_ui": 0.5, "voltage_v": 0.15, "expected_bit": 1}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = render_eye_diagram(capture, Path(tmp) / "eye.svg")
            self.assertEqual(result["status"], "produced")
            self.assertIn(b"<svg", Path(result["svg_path"]).read_bytes())
            self.assertTrue(Path(result["png_path"]).read_bytes().startswith(b"\x89PNG"))
            self.assertEqual(json.loads(Path(result["data_path"]).read_text())["locked_phase_ui"], 0.5)

    def test_training_summary_events_and_resume(self):
        from unittest.mock import patch
        from experiments.train_autockt import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = root / "full.pt"
            common = ["train_autockt", "--rl-version", "v3", "--backend", "synthetic", "--updates", "1",
                      "--episodes-per-update", "1", "--horizon", "1", "--max-evaluations", "6"]
            with patch("sys.argv", common + ["--output", str(root / "train.jsonl"), "--save-full-checkpoint", str(full),
                        "--summary-output", str(root / "summary.json"), "--graph-output", str(root / "custom.graph.json")]):
                self.assertEqual(main(), 0)
            before = torch.load(full, weights_only=True)
            with patch("sys.argv", common + ["--resume", str(full), "--output", str(root / "resume.jsonl"),
                        "--save-full-checkpoint", str(root / "resumed.pt")]):
                self.assertEqual(main(), 0)
            after = torch.load(root / "resumed.pt", weights_only=True)
            self.assertEqual(after["metadata"]["update_count"], 2)
            self.assertGreater(after["metadata"]["evaluation_count"], before["metadata"]["evaluation_count"])
            self.assertEqual(json.loads((root / "summary.json").read_text())["workflow"], "train")
            self.assertTrue((root / "custom.graph.json").is_file())
            events = [json.loads(line) for line in (root / "train.events.jsonl").read_text().splitlines()]
            self.assertTrue(any(e["event_type"] == "evaluation" for e in events))
            self.assertTrue(any(e["event_type"] == "update" for e in events))

if __name__ == "__main__":
    unittest.main()
