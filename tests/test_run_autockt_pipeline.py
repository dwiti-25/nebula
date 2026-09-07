"""SPICE-free orchestration tests for experiments/run_autockt_pipeline.py
-- Task 1 of the overnight chunk (sec 22). All tests use
backend='synthetic' (rl.synthetic_benchmark.synthetic_evaluate_receiver,
no ngspice, no PDK) -- this IS the pipeline's dry-run mode, exercised
through the real orchestration code, not a separate mock of it.
"""

from __future__ import annotations

import json
import io
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from rl.target_spec import TargetSpec

from experiments.run_autockt_pipeline import (
    PVT_CONDITION_SET_FIDELITY,
    PVT_CONDITION_SETS,
    PipelineCandidate,
    _main,
    _load_checkpoint_state,
    _pvt_result_from_selection,
    filter_nominal_feasible,
    generate_candidates,
    measure_hd3_and_noise,
    run_pipeline,
    select_final_design,
    validate_target,
)


class ValidateTargetTests(unittest.TestCase):
    def test_valid_target_has_no_problems(self):
        self.assertEqual(validate_target(TargetSpec.from_existing_thresholds()), [])
        self.assertEqual(validate_target(TargetSpec.from_hard_target()), [])

    def test_non_finite_value_is_a_problem(self):
        target = TargetSpec(
            dfe_locked_phase_eye_height_v=math.inf, dfe_eye_width_ui=0.4,
            dfe_min_margin_v=0.0, ctle_power_w=0.015,
        )
        problems = validate_target(target)
        self.assertTrue(any("not finite" in p for p in problems))

    def test_width_out_of_ui_range_is_a_problem(self):
        target = TargetSpec(
            dfe_locked_phase_eye_height_v=0.1, dfe_eye_width_ui=1.5,
            dfe_min_margin_v=0.0, ctle_power_w=0.015,
        )
        problems = validate_target(target)
        self.assertTrue(any("dfe_eye_width_ui" in p for p in problems))

    def test_nonpositive_power_is_a_problem(self):
        target = TargetSpec(
            dfe_locked_phase_eye_height_v=0.1, dfe_eye_width_ui=0.4,
            dfe_min_margin_v=0.0, ctle_power_w=0.0,
        )
        problems = validate_target(target)
        self.assertTrue(any("ctle_power_w" in p for p in problems))


class GenerateCandidatesTests(unittest.TestCase):
    def test_checkpoint_can_be_loaded_from_packaged_bytes(self):
        payload = io.BytesIO()
        torch.save({"weight": torch.tensor([1.0])}, payload)
        with patch.object(Path, "is_file", return_value=False), patch(
            "experiments.run_autockt_pipeline.read_packaged_bytes", return_value=payload.getvalue(),
        ):
            state = _load_checkpoint_state(Path("results/policy.pt"))
        self.assertTrue(torch.equal(state["weight"], torch.tensor([1.0])))

    def test_synthetic_backend_produces_the_requested_episode_count(self):
        candidates = generate_candidates(
            target=TargetSpec.from_existing_thresholds(), checkpoint_path=None,
            agent_seed=1, eval_seed=1, episodes=5, horizon=1, backend="synthetic",
            initial_indices_source="grid-center",
        )
        self.assertEqual(len(candidates), 5)
        for c in candidates:
            self.assertIsInstance(c, PipelineCandidate)
            self.assertEqual(set(c.parameters), {"rload_ohm", "rdeg_ohm", "cdeg_f", "itail_a", "dfe_tap_v"})
            self.assertGreaterEqual(c.steps, 1)

    def test_deterministic_given_the_same_seeds(self):
        kwargs = dict(
            target=TargetSpec.from_existing_thresholds(), checkpoint_path=None,
            agent_seed=7, eval_seed=7, episodes=4, horizon=1, backend="synthetic",
            initial_indices_source="grid-center",
        )
        a = generate_candidates(**kwargs)
        b = generate_candidates(**kwargs)
        self.assertEqual([c.parameters for c in a], [c.parameters for c in b])
        self.assertEqual([c.spec_satisfied for c in a], [c.spec_satisfied for c in b])

    def test_grid_center_init_reliably_finds_feasible_candidates_on_synthetic(self):
        # The synthetic benchmark's own "good region" is centered at
        # normalized 0.5 in every dimension -- the same point grid-center
        # initialization starts from -- so this combination should produce
        # a healthy feasible fraction, unlike the default 'verified'
        # (Random-Search-derived) starting point, which is unrelated to the
        # synthetic landscape's geometry.
        candidates = generate_candidates(
            target=TargetSpec.from_existing_thresholds(), checkpoint_path=None,
            agent_seed=1, eval_seed=1, episodes=10, horizon=1, backend="synthetic",
            initial_indices_source="grid-center",
        )
        feasible = filter_nominal_feasible(candidates)
        self.assertGreater(len(feasible), 0)


class FilterNominalFeasibleTests(unittest.TestCase):
    def test_keeps_only_strictly_target_satisfied_candidates(self):
        passing_metrics = {
            "dfe_locked_phase_eye_height_v": 0.1,
            "dfe_eye_width_ui": 0.4,
            "dfe_min_margin_v": 0.0,
            "ctle_power_w": 0.015,
        }
        candidates = [
            PipelineCandidate(0, {}, passing_metrics, 10.0, True, 1),
            PipelineCandidate(1, {}, {}, -1.0, False, 4, False),
        ]
        feasible = filter_nominal_feasible(candidates)
        self.assertEqual(len(feasible), 1)
        self.assertEqual(feasible[0].episode, 0)

    def test_autockt_terminal_tolerance_does_not_count_as_strict_pass(self):
        target = TargetSpec.from_hard_target()
        within_reward_tolerance = {
            "dfe_locked_phase_eye_height_v": 0.79,
            "dfe_eye_width_ui": 0.6,
            "dfe_min_margin_v": 0.35,
            "ctle_power_w": 0.015,
        }
        candidate = PipelineCandidate(0, {}, within_reward_tolerance, 10.0, True, 1)
        self.assertEqual(filter_nominal_feasible([candidate], target), [])

    def test_simulator_failure_never_counts_as_strict_pass(self):
        metrics = {
            "dfe_locked_phase_eye_height_v": 1.0,
            "dfe_eye_width_ui": 0.8,
            "dfe_min_margin_v": 0.5,
            "ctle_power_w": 0.001,
        }
        candidate = PipelineCandidate(0, {}, metrics, 10.0, True, 1, False)
        self.assertEqual(filter_nominal_feasible([candidate]), [])


class SelectFinalDesignTests(unittest.TestCase):
    def test_no_feasible_candidates_returns_none_selected_not_a_crash(self):
        selection = select_final_design([])
        self.assertIsNone(selection["selected"])
        self.assertIn("reason", selection)

    def test_nominal_only_selection_picks_a_design_and_labels_it(self):
        candidates = [
            PipelineCandidate(0, {"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                                  "itail_a": 1e-4, "dfe_tap_v": 0.0},
                               {"ctle_power_w": 0.001, "dfe_locked_phase_eye_height_v": 1.0,
                                "dfe_eye_width_ui": 0.5, "dfe_min_margin_v": 0.3},
                               10.0, True, 1),
        ]
        selection = select_final_design(candidates)
        self.assertIsNotNone(selection["selected"])
        self.assertEqual(selection["selection_basis"], "nominal-only (no PVT conditions supplied)")
        self.assertIsNone(selection["pvt"])

    def test_pvt_path_calls_run_pvt_evaluation_and_ranks_by_pass_rate(self):
        from simulator.config import ProcessCorner, SimulationConditions
        from simulator.receiver import EvaluationFidelity, ReceiverEvaluation

        candidates = [
            PipelineCandidate(0, {"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                                  "itail_a": 1e-4, "dfe_tap_v": 0.0}, {"ctle_power_w": 0.001}, 10.0, True, 1),
            PipelineCandidate(1, {"rload_ohm": 2000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                                  "itail_a": 1e-4, "dfe_tap_v": 0.0}, {"ctle_power_w": 0.001}, 10.0, True, 1),
        ]
        conditions = (SimulationConditions(ProcessCorner.TT, 27.0, 1.8),)

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            # episode-0 design (rload=1000) always passes; episode-1 (rload=2000) always fails
            success = parameters.rload_ohm == 1000.0
            return tuple(
                ReceiverEvaluation(success, parameters, c, fidelity, (), {},
                                    None if success else "transient", 0.0, "id", {})
                for c in conditions
            )

        with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
            selection = select_final_design(candidates, pvt_conditions=conditions)

        self.assertEqual(selection["selected"]["design_id"], "pipeline_ep0")
        self.assertEqual(selection["pvt"]["pass_rate"], 1.0)
        self.assertTrue(selection["pvt"]["met_minimum_pass_rate"])

    def test_pvt_selection_fails_closed_when_no_candidate_meets_minimum(self):
        from simulator.config import ProcessCorner, SimulationConditions
        from simulator.receiver import ReceiverEvaluation

        candidate = PipelineCandidate(
            0, {"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                "itail_a": 1e-4, "dfe_tap_v": 0.0},
            {"ctle_power_w": 0.001}, 10.0, True, 1,
        )
        conditions = (SimulationConditions(ProcessCorner.TT, 27.0, 1.8),)

        def fail_grid(parameters, *, conditions, fidelity):
            return tuple(ReceiverEvaluation(
                False, parameters, condition, fidelity, (), {}, "transient", 0.0, "id", {},
            ) for condition in conditions)

        with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fail_grid):
            selection = select_final_design([candidate], pvt_conditions=conditions)
        self.assertIsNone(selection["selected"])
        self.assertIn("no candidate met", selection["reason"])
        self.assertEqual(selection["best_available"]["pass_rate"], 0.0)

    def test_pvt_tie_is_broken_by_trade_off_preference(self):
        # Task 3 (NEXT IMPLEMENTATION CHUNK): select_final_design must
        # actually use analysis.pvt_selection.select_with_trade_off_
        # preference, not just rank_by_robustness, so a genuine PVT tie is
        # broken by the caller's stated preference rather than by
        # incidental list order.
        from simulator.config import ProcessCorner, SimulationConditions
        from simulator.receiver import EvaluationFidelity, ReceiverEvaluation

        candidates = [
            PipelineCandidate(0, {"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                                  "itail_a": 1e-4, "dfe_tap_v": 0.0},
                               {"ctle_power_w": 0.01, "dfe_locked_phase_eye_height_v": 1.0,
                                "dfe_eye_width_ui": 0.5, "dfe_min_margin_v": 0.3}, 10.0, True, 1),
            PipelineCandidate(1, {"rload_ohm": 2000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13,
                                  "itail_a": 1e-4, "dfe_tap_v": 0.0},
                               {"ctle_power_w": 0.001, "dfe_locked_phase_eye_height_v": 1.0,
                                "dfe_eye_width_ui": 0.5, "dfe_min_margin_v": 0.3}, 10.0, True, 1),
        ]
        conditions = (SimulationConditions(ProcessCorner.TT, 27.0, 1.8),)

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            # both designs pass at every condition -- a genuine tie.
            return tuple(
                ReceiverEvaluation(True, parameters, c, fidelity, (), {}, None, 0.0, "id", {})
                for c in conditions
            )

        with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
            lowest_power = select_final_design(
                candidates, pvt_conditions=conditions, trade_off_preference="lowest_power",
            )
            most_robust = select_final_design(
                candidates, pvt_conditions=conditions, trade_off_preference="most_robust",
            )

        # pipeline_ep1 has the lower ctle_power_w (0.001 vs 0.01).
        self.assertEqual(lowest_power["selected"]["design_id"], "pipeline_ep1")
        self.assertIn("lowest_power", lowest_power["selection_basis"])
        # 'most_robust' ignores trade-offs and returns the first by
        # robustness order among the tie -- pipeline_ep0.
        self.assertEqual(most_robust["selected"]["design_id"], "pipeline_ep0")


class PvtResultFlowsIntoFinalSpecificationTests(unittest.TestCase):
    """Integration requirement (NEXT IMPLEMENTATION CHUNK): the pipeline's
    own real PVT result (if any) must reach the final specification
    report's PVT row, not be silently dropped in favor of a fresh
    'NOT CLAIMED' -- and must not require re-running PVT a second time to
    get there.
    """

    def test_pvt_result_from_selection_reconstructs_the_summary(self):
        selection = {
            "selected": {"design_id": "x", "parameters": {}, "metrics": {}},
            "pvt": {
                "n_conditions": 4, "n_passing": 3, "pass_rate": 0.75,
                "met_minimum_pass_rate": False,
                "worst_case_conditions": [
                    {"corner": "ff", "vdd": 1.71, "temp_c": 125.0, "failed_stage": "transient"},
                ],
            },
        }
        result = _pvt_result_from_selection(selection)
        self.assertEqual(result.design_id, "x")
        self.assertEqual((result.n_conditions, result.n_passing, result.pass_rate), (4, 3, 0.75))
        self.assertEqual(len(result.worst_case_conditions), 1)
        self.assertEqual(result.worst_case_conditions[0].failed_stage, "transient")

    def test_no_pvt_selection_reconstructs_to_none(self):
        self.assertIsNone(_pvt_result_from_selection({"selected": {"design_id": "x"}, "pvt": None}))

    def test_run_pipeline_final_specification_reflects_real_pvt_result(self):
        from simulator.config import ProcessCorner, SimulationConditions
        from simulator.receiver import ReceiverEvaluation

        conditions = (SimulationConditions(ProcessCorner.TT, 27.0, 1.8),
                      SimulationConditions(ProcessCorner.FF, 125.0, 1.71))

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            metrics = {
                "dfe_locked_phase_eye_height_v": 1.0,
                "dfe_eye_width_ui": 0.8,
                "dfe_min_margin_v": 0.5,
                "ctle_power_w": 0.001,
            }
            return tuple(
                ReceiverEvaluation(True, parameters, c, fidelity, (), metrics, None, 0.0, "id", {})
                for c in conditions
            )

        with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
            result = run_pipeline(
                target=TargetSpec.from_existing_thresholds(), checkpoint_path=None,
                agent_seed=1, eval_seed=1, episodes=8, horizon=1, backend="synthetic",
                initial_indices_source="grid-center", pvt_conditions=conditions,
            )

        self.assertIsNotNone(result["selection"]["selected"])
        pvt_row = next(r for r in result["final_specification"]["rows"] if r["metric"] == "PVT (pass/total)")
        # must reflect the real 2-condition sweep, not "NOT CLAIMED".
        self.assertEqual(pvt_row["measured"], "2/2")
        self.assertEqual(pvt_row["verdict"], "PASS")


class MeasureHd3AndNoiseTests(unittest.TestCase):
    """FINAL AUDIT gaps A/B: HD3 and input-referred-noise refinement.
    simulator.receiver.evaluate_receiver is mocked -- SPICE-free -- to
    verify the wiring; the real measurement itself is exercised
    separately (see docs/FINAL_TECHNICAL_AUDIT.md for the one real-SPICE
    validation run against Design A).
    """

    PARAMS = {"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13, "itail_a": 1e-4, "dfe_tap_v": 0.0}

    def test_success_returns_hd3_and_noise_metrics(self):
        from simulator.receiver import EvaluationFidelity, ReceiverEvaluation, ReceiverParameters

        fake_metrics = {"hd3_db": -45.2, "input_referred_noise_vrms": 0.0008, "dfe_eye_width_ui": 0.8}

        def fake_evaluate_receiver(parameters, conditions, fidelity):
            self.assertEqual(fidelity, EvaluationFidelity.FINAL)
            return ReceiverEvaluation(True, parameters, conditions, fidelity, (), fake_metrics,
                                       None, 120.0, "id", {})

        with patch("experiments.run_autockt_pipeline.evaluate_receiver", side_effect=fake_evaluate_receiver):
            result = measure_hd3_and_noise(self.PARAMS)

        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"]["hd3_db"], -45.2)
        self.assertEqual(result["metrics"]["input_referred_noise_vrms"], 0.0008)
        self.assertIsNone(result["failed_stage"])

    def test_failure_reports_failed_stage_not_a_fabricated_value(self):
        from simulator.receiver import EvaluationFidelity, ReceiverEvaluation

        def fake_evaluate_receiver(parameters, conditions, fidelity):
            return ReceiverEvaluation(False, parameters, conditions, fidelity, (), {"dc_ok": 0.0},
                                       "hd3", 30.0, "id", {})

        with patch("experiments.run_autockt_pipeline.evaluate_receiver", side_effect=fake_evaluate_receiver):
            result = measure_hd3_and_noise(self.PARAMS)

        self.assertFalse(result["success"])
        self.assertEqual(result["failed_stage"], "hd3")
        self.assertNotIn("hd3_db", result["metrics"])


class RunPipelineHd3NoiseRefinementTests(unittest.TestCase):
    def _feasible_candidates(self):
        return [PipelineCandidate(
            0, {"rload_ohm": 1000.0, "rdeg_ohm": 1000.0, "cdeg_f": 5e-13, "itail_a": 1e-4, "dfe_tap_v": 0.0},
            {"ctle_power_w": 0.001, "dfe_locked_phase_eye_height_v": 1.0, "dfe_eye_width_ui": 0.5,
             "dfe_min_margin_v": 0.3}, 10.0, True, 1,
        )]

    def test_refinement_merges_hd3_noise_into_the_final_specification(self):
        from simulator.receiver import EvaluationFidelity, ReceiverEvaluation

        def fake_evaluate_receiver(parameters, conditions, fidelity):
            return ReceiverEvaluation(True, parameters, conditions, fidelity, (),
                                       {"hd3_db": -40.0, "input_referred_noise_vrms": 0.0005},
                                       None, 90.0, "id", {})

        with patch("experiments.run_autockt_pipeline.generate_candidates", return_value=self._feasible_candidates()):
            with patch("experiments.run_autockt_pipeline.evaluate_receiver", side_effect=fake_evaluate_receiver):
                result = run_pipeline(
                    target=TargetSpec.from_existing_thresholds(), checkpoint_path=None, backend="real",
                    measure_hd3_noise_flag=True,
                )

        self.assertTrue(result["hd3_noise_refinement"]["attempted"])
        self.assertTrue(result["hd3_noise_refinement"]["success"])
        hd3_row = next(r for r in result["final_specification"]["rows"] if r["metric"] == "HD3 (dB)")
        self.assertEqual(hd3_row["verdict"], "PASS")
        noise_row = next(r for r in result["final_specification"]["rows"]
                          if r["metric"] == "Input-referred noise (Vrms)")
        self.assertEqual(noise_row["verdict"], "PASS")

    def test_flag_off_by_default_leaves_hd3_noise_not_claimed(self):
        with patch("experiments.run_autockt_pipeline.generate_candidates", return_value=self._feasible_candidates()):
            result = run_pipeline(target=TargetSpec.from_existing_thresholds(), checkpoint_path=None, backend="real")

        self.assertNotIn("hd3_noise_refinement", result)
        hd3_row = next(r for r in result["final_specification"]["rows"] if r["metric"] == "HD3 (dB)")
        self.assertEqual(hd3_row["verdict"], "NOT CLAIMED")

    def test_flag_is_a_no_op_for_synthetic_backend(self):
        with patch("experiments.run_autockt_pipeline.generate_candidates", return_value=self._feasible_candidates()):
            result = run_pipeline(
                target=TargetSpec.from_existing_thresholds(), checkpoint_path=None, backend="synthetic",
                measure_hd3_noise_flag=True,
            )
        self.assertFalse(result["hd3_noise_refinement"]["attempted"])


class PvtConditionSetsAuditTests(unittest.TestCase):
    """FINAL AUDIT gap D: full PVT must be an explicit, non-default,
    non-automatic option that reuses (not duplicates) the authoritative
    27-point set experiments/pvt_sweep.py already defines.
    """

    def test_none_is_still_the_default_no_pvt_spent(self):
        self.assertIsNone(PVT_CONDITION_SETS["none"])

    def test_minimal27_is_present_and_has_27_conditions(self):
        self.assertIn("minimal27", PVT_CONDITION_SETS)
        self.assertEqual(len(PVT_CONDITION_SETS["minimal27"]), 27)

    def test_minimal27_is_the_same_object_as_pvt_sweeps_own_set(self):
        from experiments import pvt_sweep
        self.assertEqual(PVT_CONDITION_SETS["minimal27"], pvt_sweep.MINIMAL_27_CONDITIONS)

    def test_cli_default_pvt_condition_set_is_still_none(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = ["run_autockt_pipeline.py", "--backend", "synthetic", "--episodes", "1", "--output", str(output)]
            with patch("sys.argv", argv):
                _main()
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertIsNone(result["selection"]["pvt"])

    def test_smoke_maps_to_candidate_fidelity_not_final(self):
        # RUNTIME diagnosis: select_final_design's own default (FINAL) was
        # being silently inherited by "smoke", making a "small" 2-condition
        # set cost ~571s (measured) for one candidate. CANDIDATE runs the
        # identical stage set (verified against simulator/receiver.py) and
        # is sufficient for PVT pass/fail + trade-off selection, since
        # analysis.pvt_selection never reads evaluation.metrics.
        from simulator.receiver import EvaluationFidelity
        self.assertEqual(PVT_CONDITION_SET_FIDELITY["smoke"], EvaluationFidelity.CANDIDATE)

    def test_minimal27_still_maps_to_final_fidelity_unchanged(self):
        # The actual robustness proof must not be weakened.
        from simulator.receiver import EvaluationFidelity
        self.assertEqual(PVT_CONDITION_SET_FIDELITY["minimal27"], EvaluationFidelity.FINAL)

    def test_none_never_reaches_pvt_evaluation_so_has_no_fidelity_to_choose(self):
        # "none" skips select_final_design's PVT branch entirely (pvt_conditions
        # is None) -- there is no fidelity decision to make for it, and it must
        # not silently gain one.
        self.assertNotIn("none", PVT_CONDITION_SET_FIDELITY)

    def test_smoke_condition_definitions_are_exactly_unchanged(self):
        # Locks down the two smoke corners themselves -- only the FIDELITY
        # used to evaluate them may change, never the corner/temp/vdd values.
        from simulator.config import ProcessCorner, SimulationConditions
        nominal, stress = PVT_CONDITION_SETS["smoke"]
        self.assertEqual(nominal, SimulationConditions(ProcessCorner.TT, 27.0, 1.8))
        self.assertEqual(stress, SimulationConditions(ProcessCorner.FF, 125.0, 1.71))


class CLIIntegrationTests(unittest.TestCase):
    def test_full_synthetic_dry_run_produces_a_structured_result_with_schematic_and_spec(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            schematic = Path(tmp) / "schematic.spice"
            argv = [
                "run_autockt_pipeline.py", "--target-mode", "trivial", "--backend", "synthetic",
                "--episodes", "8", "--horizon", "1", "--initial-indices-source", "grid-center",
                "--agent-seed", "1", "--eval-seed", "1",
                "--output", str(output), "--export-schematic", str(schematic),
            ]
            with patch("sys.argv", argv):
                _main()

            self.assertTrue(output.is_file())
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["backend"], "synthetic")
            self.assertEqual(result["n_candidates_generated"], 8)
            self.assertIn("selection", result)
            if result["selection"]["selected"] is not None:
                self.assertTrue(schematic.is_file())
                self.assertIn("final_specification", result)
                self.assertIn("rows", result["final_specification"])

    def test_refuses_to_overwrite_existing_output(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            output.write_text("already here")
            argv = [
                "run_autockt_pipeline.py", "--backend", "synthetic", "--episodes", "1",
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(FileExistsError):
                    _main()

    def test_pvt_condition_set_flag_reaches_the_pvt_branch_via_the_cli(self):
        # Confirms the CLI-level gap (analysis.pvt_selection was previously
        # unreachable from _main()) is closed: --pvt-condition-set smoke
        # must cause select_final_design's PVT branch to run, and its
        # result must reach the final specification report's PVT row.
        from simulator.receiver import ReceiverEvaluation

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            return tuple(
                ReceiverEvaluation(True, parameters, c, fidelity, (), {}, None, 0.0, "id", {})
                for c in conditions
            )

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = [
                "run_autockt_pipeline.py", "--target-mode", "trivial", "--backend", "synthetic",
                "--episodes", "8", "--horizon", "1", "--initial-indices-source", "grid-center",
                "--agent-seed", "1", "--eval-seed", "1", "--pvt-condition-set", "smoke",
                "--output", str(output),
            ]
            with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
                with patch("sys.argv", argv):
                    _main()

            result = json.loads(output.read_text(encoding="utf-8"))
            selection = result["selection"]
            if selection["selected"] is not None:
                self.assertIsNotNone(selection["pvt"])
                self.assertEqual(selection["pvt"]["n_conditions"], 2)
                self.assertIn("PVT-ranked", selection["selection_basis"])
                pvt_row = next(r for r in result["final_specification"]["rows"]
                                if r["metric"] == "PVT (pass/total)")
                self.assertEqual(pvt_row["measured"], "2/2")

    def test_smoke_end_to_end_reaches_evaluate_pvt_grid_at_candidate_fidelity(self):
        # Proves the wiring, not just the static PVT_CONDITION_SET_FIDELITY
        # dict: --pvt-condition-set smoke must cause the REAL call into
        # analysis.pvt_selection.evaluate_pvt_grid to receive
        # EvaluationFidelity.CANDIDATE, and PVT selection must still produce
        # a normal pass/fail result using only the metrics CANDIDATE
        # fidelity provides (this fake evaluator returns no metrics at all,
        # matching that run_pvt_evaluation never reads evaluation.metrics).
        from simulator.receiver import EvaluationFidelity, ReceiverEvaluation

        seen_fidelities = []

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            seen_fidelities.append(fidelity)
            return tuple(
                ReceiverEvaluation(True, parameters, c, fidelity, (), {}, None, 0.0, "id", {})
                for c in conditions
            )

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = [
                "run_autockt_pipeline.py", "--target-mode", "trivial", "--backend", "synthetic",
                "--episodes", "8", "--horizon", "1", "--initial-indices-source", "grid-center",
                "--agent-seed", "1", "--eval-seed", "1", "--pvt-condition-set", "smoke",
                "--output", str(output),
            ]
            with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
                with patch("sys.argv", argv):
                    _main()

            result = json.loads(output.read_text(encoding="utf-8"))
            if result["selection"]["selected"] is not None:
                self.assertTrue(seen_fidelities, "expected evaluate_pvt_grid to be called at least once")
                self.assertTrue(all(f == EvaluationFidelity.CANDIDATE for f in seen_fidelities))
                self.assertEqual(result["selection"]["pvt"]["pass_rate"], 1.0)
                self.assertTrue(result["selection"]["pvt"]["met_minimum_pass_rate"])

    def test_minimal27_end_to_end_still_reaches_evaluate_pvt_grid_at_final_fidelity(self):
        from simulator.receiver import EvaluationFidelity, ReceiverEvaluation

        seen_fidelities = []

        def fake_evaluate_pvt_grid(parameters, *, conditions, fidelity):
            seen_fidelities.append(fidelity)
            return tuple(
                ReceiverEvaluation(True, parameters, c, fidelity, (), {}, None, 0.0, "id", {})
                for c in conditions
            )

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = [
                "run_autockt_pipeline.py", "--target-mode", "trivial", "--backend", "synthetic",
                "--episodes", "8", "--horizon", "1", "--initial-indices-source", "grid-center",
                "--agent-seed", "1", "--eval-seed", "1", "--pvt-condition-set", "minimal27",
                "--output", str(output),
            ]
            with patch("analysis.pvt_selection.evaluate_pvt_grid", side_effect=fake_evaluate_pvt_grid):
                with patch("sys.argv", argv):
                    _main()

            result = json.loads(output.read_text(encoding="utf-8"))
            if result["selection"]["selected"] is not None:
                self.assertTrue(seen_fidelities, "expected evaluate_pvt_grid to be called at least once")
                self.assertTrue(all(f == EvaluationFidelity.FINAL for f in seen_fidelities))

    def test_target_json_flag_builds_an_arbitrary_target(self):
        import json as _json
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            custom_target = {
                "dfe_locked_phase_eye_height_v": 0.2, "dfe_eye_width_ui": 0.5,
                "dfe_min_margin_v": 0.05, "ctle_power_w": 0.012,
            }
            argv = [
                "run_autockt_pipeline.py", "--backend", "synthetic", "--episodes", "1",
                "--horizon", "1", "--initial-indices-source", "grid-center",
                "--target-json", _json.dumps(custom_target),
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                _main()
            result = _json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["target"], custom_target)

    def test_target_json_missing_field_raises_clearly(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = [
                "run_autockt_pipeline.py", "--backend", "synthetic", "--episodes", "1",
                "--target-json", '{"dfe_locked_phase_eye_height_v": 0.2}',
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(ValueError):
                    _main()

    def test_default_pvt_condition_set_is_none_unchanged_behavior(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            argv = [
                "run_autockt_pipeline.py", "--backend", "synthetic", "--episodes", "1",
                "--horizon", "1", "--initial-indices-source", "grid-center",
                "--output", str(output),
            ]
            with patch("sys.argv", argv):
                _main()
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertIsNone(result["selection"]["pvt"])


if __name__ == "__main__":
    unittest.main()
