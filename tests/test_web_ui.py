"""SPICE-free tests for experiments/web_ui.py -- the competition-demo
local UI around experiments/run_autockt_pipeline.py.

Covers request validation, CLI argv construction (must match
run_autockt_pipeline.py's own flags exactly), and the full HTTP flow using
--backend synthetic (no ngspice, no PDK). The crash-handling path
(subprocess terminated by a signal) is tested by mocking subprocess.Popen
rather than by relying on the real, intermittent SIGSEGV documented in
docs/autockt-mapping.md sec 23 -- that failure mode is real but not
reliably reproducible on demand, so it is simulated here deterministically.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.client import HTTPResponse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from experiments import web_ui


class ValidateRequestTests(unittest.TestCase):
    def test_pvt_pattern_bits_validation_and_argv(self):
        for value in (128, 2048, None, "1024"):
            self.assertTrue(web_ui._validate_request(self._base(pvt_pattern_bits=value)))
        for bits in (512, 1024):
            payload = self._base(pvt_pattern_bits=bits)
            self.assertEqual(web_ui._validate_request(payload), [])
            argv = web_ui._build_argv(payload, output_path=Path("new.json"), schematic_path=Path("new.cir"))
            self.assertEqual(argv[argv.index("--pvt-pattern-bits") + 1], str(bits))

    def test_simulator_timeout_validation_and_argv(self):
        for value in (0, -1, float("nan"), float("inf"), True, 3601):
            self.assertTrue(web_ui._validate_request(self._base(simulator_timeout_seconds=value)))
        payload = self._base(simulator_timeout_seconds=720)
        self.assertEqual(web_ui._validate_request(payload), [])
        argv = web_ui._build_argv(payload, output_path=Path("new.json"), schematic_path=Path("new.cir"))
        self.assertEqual(argv[argv.index("--simulator-timeout-seconds") + 1], "720")

    def _base(self, **overrides):
        payload = {
            "target_mode": "trivial", "backend": "synthetic", "checkpoint": None,
            "episodes": 3, "horizon": 4, "pvt_condition_set": "none",
            "trade_off_preference": "most_robust",
        }
        payload.update(overrides)
        return payload

    def test_valid_trivial_synthetic_request_has_no_problems(self):
        self.assertEqual(web_ui._validate_request(self._base()), [])

    def test_bad_target_mode_is_rejected(self):
        problems = web_ui._validate_request(self._base(target_mode="nonsense"))
        self.assertTrue(any("target_mode" in p for p in problems))

    def test_custom_target_requires_all_four_fields(self):
        problems = web_ui._validate_request(self._base(target_mode="custom", target={"dfe_eye_width_ui": 0.5}))
        self.assertTrue(any("dfe_locked_phase_eye_height_v" in p for p in problems))

    def test_custom_target_with_all_fields_is_valid(self):
        target = {"dfe_locked_phase_eye_height_v": 0.1, "dfe_eye_width_ui": 0.4,
                  "dfe_min_margin_v": 0.0, "ctle_power_w": 0.015}
        self.assertEqual(web_ui._validate_request(self._base(target_mode="custom", target=target)), [])

    def test_real_backend_without_checkpoint_is_rejected(self):
        problems = web_ui._validate_request(self._base(backend="real", checkpoint=None))
        self.assertTrue(any("checkpoint" in p for p in problems))

    def test_nonexistent_checkpoint_is_rejected(self):
        problems = web_ui._validate_request(self._base(backend="real", checkpoint="results/does_not_exist.pt"))
        self.assertTrue(any("not found" in p for p in problems))

    def test_packaged_checkpoint_is_valid_without_loose_result_file(self):
        checkpoint = "results/autockt_mixed_target_confirmation_policy.pt"
        with patch.object(Path, "is_file", return_value=False), patch(
            "experiments.web_ui.packaged_file_exists", return_value=True,
        ):
            problems = web_ui._validate_request(self._base(backend="real", checkpoint=checkpoint))
        self.assertEqual(problems, [])

    def test_checkpoint_path_escaping_the_project_is_rejected(self):
        problems = web_ui._validate_request(self._base(backend="real", checkpoint="../../etc/passwd"))
        self.assertTrue(any("not found" in p for p in problems))

    def test_non_integer_episodes_is_rejected(self):
        problems = web_ui._validate_request(self._base(episodes="not-a-number"))
        self.assertTrue(any("episodes" in p for p in problems))

    def test_zero_episodes_is_rejected(self):
        problems = web_ui._validate_request(self._base(episodes=0))
        self.assertTrue(any("episodes" in p for p in problems))

    def test_bad_pvt_condition_set_is_rejected(self):
        problems = web_ui._validate_request(self._base(pvt_condition_set="invalid"))
        self.assertTrue(any("pvt_condition_set" in p for p in problems))

    def test_unknown_channel_is_rejected(self):
        problems = web_ui._validate_request(self._base(channel_id="not-a-channel"))
        self.assertTrue(any("channel_id" in p for p in problems))


class BuildArgvTests(unittest.TestCase):
    def test_reference_channel_adds_path_and_qualified_port_map(self):
        argv = web_ui._build_argv(
            {"target_mode": "trivial", "backend": "synthetic", "episodes": 1, "horizon": 1,
             "pvt_condition_set": "none", "trade_off_preference": "most_robust",
             "channel_id": "ieee802_reference"},
            output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"),
        )
        channel_index = argv.index("--channel")
        ports_index = argv.index("--channel-ports")
        self.assertEqual(argv[channel_index + 1], "channels/ieee802_ibm_20db_thru.s4p")
        self.assertEqual(argv[ports_index + 1:ports_index + 5], ["1", "3", "2", "4"])

    def test_trivial_preset_uses_target_mode_flag(self):
        argv = web_ui._build_argv(
            {"target_mode": "trivial", "backend": "synthetic", "episodes": 3, "horizon": 4,
             "pvt_condition_set": "none", "trade_off_preference": "most_robust"},
            output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"),
        )
        self.assertIn("--target-mode", argv)
        self.assertIn("trivial", argv)
        self.assertNotIn("--target-json", argv)

    def test_custom_target_uses_target_json_flag(self):
        target = {"dfe_locked_phase_eye_height_v": 0.2, "dfe_eye_width_ui": 0.5,
                  "dfe_min_margin_v": 0.05, "ctle_power_w": 0.012}
        argv = web_ui._build_argv(
            {"target_mode": "custom", "target": target, "backend": "synthetic", "episodes": 3, "horizon": 4,
             "pvt_condition_set": "none", "trade_off_preference": "most_robust"},
            output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"),
        )
        self.assertIn("--target-json", argv)
        self.assertNotIn("--target-mode", argv)
        json_arg = argv[argv.index("--target-json") + 1]
        self.assertEqual(json.loads(json_arg), target)

    def test_checkpoint_included_only_when_given(self):
        base = {"target_mode": "trivial", "backend": "synthetic", "episodes": 1, "horizon": 1,
                "pvt_condition_set": "none", "trade_off_preference": "most_robust"}
        argv_without = web_ui._build_argv(base, output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"))
        self.assertNotIn("--checkpoint", argv_without)

        argv_with = web_ui._build_argv(
            {**base, "checkpoint": "results/foo.pt"}, output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"),
        )
        self.assertIn("--checkpoint", argv_with)
        self.assertIn("results/foo.pt", argv_with)

    def test_module_invoked_is_the_existing_unmodified_pipeline_entry_point(self):
        argv = web_ui._build_argv(
            {"target_mode": "trivial", "backend": "synthetic", "episodes": 1, "horizon": 1,
             "pvt_condition_set": "none", "trade_off_preference": "most_robust"},
            output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"),
        )
        self.assertIn("experiments.run_autockt_pipeline", argv)
        self.assertIn("-m", argv)


class ExecuteRunCrashHandlingTests(unittest.TestCase):
    """Simulates the real, intermittent SIGSEGV documented in
    docs/autockt-mapping.md sec 23 deterministically, via a real short-
    lived subprocess that signals itself -- so this test does not depend
    on the crash actually occurring in the real pipeline, and runs with
    zero real SPICE. Uses real subprocesses (not a mocked
    subprocess.Popen) because _execute_run now reads proc.stdout/stderr
    via select.select, which requires real OS-backed pipe file
    descriptors that a MagicMock cannot provide.
    """

    def test_signal_terminated_subprocess_is_reported_clearly_not_as_a_hang(self):
        run_id = "testrun_crash_00000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["x"],
                "output_path": "/tmp/nope.json", "schematic_path": "/tmp/nope.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }
        argv = [
            sys.executable, "-c",
            "import os, signal, sys; print('partial stdout before crash'); "
            "sys.stdout.flush(); os.kill(os.getpid(), signal.SIGSEGV)",
        ]
        web_ui._execute_run(run_id, argv, Path("/tmp/nope.json"), Path("/tmp/nope.spice"))

        payload = web_ui._status_payload(run_id)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("signal 11", payload["error"])
        self.assertIn("SIGSEGV", payload["error"])
        self.assertIn("partial stdout before crash", payload["stdout_tail"])

    def test_nonzero_exit_without_signal_is_reported_as_a_normal_failure(self):
        run_id = "testrun_fail_000000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["x"],
                "output_path": "/tmp/nope2.json", "schematic_path": "/tmp/nope2.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }
        argv = [
            sys.executable, "-c",
            "import sys; sys.stderr.write('ValueError: invalid target specification'); sys.exit(1)",
        ]
        web_ui._execute_run(run_id, argv, Path("/tmp/nope2.json"), Path("/tmp/nope2.spice"))

        payload = web_ui._status_payload(run_id)
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("signal", payload["error"])
        self.assertIn("non-zero status (1)", payload["error"])
        self.assertIn("ValueError: invalid target specification", payload["stderr_tail"])


class CliArgsTests(unittest.TestCase):
    def test_default_host_and_port(self):
        args = web_ui._parse_args([])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8000)

    def test_default_host_is_localhost_only(self):
        # Requirement: the server must stay localhost-only unless the
        # operator explicitly overrides --host.
        self.assertEqual(web_ui.HOST, "127.0.0.1")
        self.assertNotEqual(web_ui.HOST, "0.0.0.0")

    def test_port_override_is_actually_used(self):
        args = web_ui._parse_args(["--port", "8001"])
        self.assertEqual(args.port, 8001)
        self.assertEqual(args.host, "127.0.0.1")  # unchanged when only --port is given

    def test_host_override(self):
        args = web_ui._parse_args(["--host", "0.0.0.0", "--port", "9999"])
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9999)

    def test_direct_script_entry_point_can_import_sibling_packages(self):
        proc = web_ui.subprocess.run(
            [web_ui.sys.executable, str(Path(web_ui.__file__).resolve()), "--help"],
            cwd=web_ui.PROJECT_ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("NEBULA local UI server", proc.stdout)


class ServerStartupAndReuseTests(unittest.TestCase):
    def test_build_server_binds_the_requested_host_and_port(self):
        server = web_ui.build_server("127.0.0.1", 0)
        try:
            self.assertEqual(server.server_address[0], "127.0.0.1")
            self.assertGreater(server.server_address[1], 0)
        finally:
            server.server_close()

    def test_server_class_has_reuse_address_enabled(self):
        self.assertTrue(web_ui.ReusableThreadingHTTPServer.allow_reuse_address)

    def test_port_can_be_rebound_immediately_after_close(self):
        # Simulates "I stopped the server and want to relaunch on the same
        # port right away" -- must not raise OSError: Address already in use.
        first = web_ui.build_server("127.0.0.1", 0)
        port = first.server_address[1]
        first.server_close()

        second = web_ui.build_server("127.0.0.1", port)
        try:
            self.assertEqual(second.server_address[1], port)
        finally:
            second.server_close()

    def test_main_prints_the_exact_browser_url_and_returns_without_blocking(self):
        # main() calls serve_forever(), which blocks -- patch it out so this
        # test only exercises argument parsing, server construction, and
        # the printed URL.
        with patch.object(web_ui.ReusableThreadingHTTPServer, "serve_forever", return_value=None):
            with patch("builtins.print") as mock_print:
                rc = web_ui.main(["--port", "0"])
        self.assertEqual(rc, 0)
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list)
        self.assertIn("http://127.0.0.1:0", printed)


class ManualTargetEntryUiIntactTests(unittest.TestCase):
    """Guards the requirement that structured manual target-spec input
    fields remain the primary interface (not replaced by anything
    natural-language/LLM-based), and that they cover exactly the fields
    TargetSpec actually supports -- no more, no less. TargetSpec (see
    rl/target_spec.py) has exactly 4 fields: eye height, eye width,
    margin, power. It has no peaking bounds -- peaking is a separate,
    fixed downstream check (analysis/final_specification.py, 3-12 dB),
    not a PPO target input -- so this UI correctly does not expose a
    peaking input field, and this test guards against one being added
    that wouldn't actually connect to anything.
    """

    def test_all_four_target_spec_fields_have_a_manual_input(self):
        for field_id in ("tEyeHeight", "tEyeWidth", "tMargin", "tPower"):
            self.assertIn(f'id="{field_id}"', web_ui.INDEX_HTML)

    def test_manual_inputs_are_plain_number_fields_not_free_text_or_llm(self):
        for field_id in ("tEyeHeight", "tEyeWidth", "tMargin", "tPower"):
            start = web_ui.INDEX_HTML.index(f'id="{field_id}"')
            snippet = web_ui.INDEX_HTML[max(0, start - 40):start + 60]
            self.assertIn('type="number"', snippet)

    def test_no_peaking_input_field_exists(self):
        # TargetSpec has no peaking bounds -- a peaking input would be
        # decorative and not wired to anything real.
        self.assertNotIn('id="tPeaking', web_ui.INDEX_HTML)

    def test_run_button_still_calls_the_existing_pipeline_api(self):
        self.assertIn("'/api/run'", web_ui.INDEX_HTML)
        self.assertIn("runBtn", web_ui.INDEX_HTML)

    def test_multiple_evidence_graphs_are_wired_to_the_dashboard_endpoint(self):
        self.assertIn("'/api/evidence'", web_ui.INDEX_HTML)
        self.assertIn("renderChart", web_ui.INDEX_HTML)
        self.assertIn("chartGrid", web_ui.INDEX_HTML)

    def test_historical_run_ui_renders_saved_results_and_pvt_values(self):
        self.assertIn("`/api/result/${runId}`", web_ui.INDEX_HTML)
        self.assertIn('id="pvtRows"', web_ui.INDEX_HTML)
        self.assertIn("m.dfe_locked_phase_eye_height_v", web_ui.INDEX_HTML)
        self.assertIn("m.input_referred_noise_vrms", web_ui.INDEX_HTML)


class SubprocessTimeoutTests(unittest.TestCase):
    """FINAL AUDIT gap E: bounded outer-process handling for a HUNG (not
    crashed) subprocess -- distinct from the SIGSEGV path above. Verifies
    the subprocess is killed and reaped, and the run is reported as a
    clean timeout, never retried automatically.
    """

    def test_hung_subprocess_is_killed_and_reported_as_a_timeout(self):
        run_id = "testrun_timeout_0000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["x"],
                "output_path": "/tmp/nope3.json", "schematic_path": "/tmp/nope3.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }
        argv = [
            sys.executable, "-c",
            "import sys, time; print('partial output before the hang was killed'); "
            "sys.stdout.flush(); time.sleep(30)",
        ]
        with patch.object(web_ui, "RUN_TIMEOUT_S", 0.3):
            web_ui._execute_run(run_id, argv, Path("/tmp/nope3.json"), Path("/tmp/nope3.spice"))

        payload = web_ui._status_payload(run_id)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("did not finish within", payload["error"])
        self.assertIn("hang", payload["error"])
        self.assertIn("partial output", payload["stdout_tail"])

    def test_normal_completion_never_calls_kill(self):
        run_id = "testrun_normal_00000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["x"],
                "output_path": "/tmp/nope4.json", "schematic_path": "/tmp/nope4.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }
        argv = [sys.executable, "-c", "import sys; sys.exit(1)"]
        with patch.object(subprocess.Popen, "kill") as mock_kill:
            web_ui._execute_run(run_id, argv, Path("/tmp/nope4.json"), Path("/tmp/nope4.spice"))
        mock_kill.assert_not_called()


class ParseProgressLineTests(unittest.TestCase):
    """Pure-function tests for _parse_progress_line -- no subprocess, no
    locking. Must never raise, regardless of input.
    """

    def test_valid_progress_json_is_parsed(self):
        line = json.dumps({"nebula_progress_event": "pvt_condition_start", "corner": "tt"})
        event = web_ui._parse_progress_line(line)
        self.assertEqual(event["nebula_progress_event"], "pvt_condition_start")
        self.assertEqual(event["corner"], "tt")

    def test_blank_line_is_not_a_progress_event(self):
        self.assertIsNone(web_ui._parse_progress_line(""))
        self.assertIsNone(web_ui._parse_progress_line("   \n"))

    def test_non_json_line_is_not_a_progress_event(self):
        self.assertIsNone(web_ui._parse_progress_line("ngspice: warning: convergence issue"))

    def test_json_without_the_marker_key_is_not_a_progress_event(self):
        # e.g. the pipeline's OWN final result JSON, or any other stray
        # JSON a subprocess might print -- must not be misread as progress.
        self.assertIsNone(web_ui._parse_progress_line(json.dumps({"target": {}, "backend": "real"})))

    def test_json_array_is_not_a_progress_event(self):
        self.assertIsNone(web_ui._parse_progress_line(json.dumps([1, 2, 3])))

    def test_truncated_json_does_not_raise(self):
        self.assertIsNone(web_ui._parse_progress_line('{"nebula_progress_event": "pvt_condition_st'))


class ApplyProgressEventTests(unittest.TestCase):
    """Pure-function tests for _apply_progress_event -- the state-mutation
    logic, independent of subprocess/locking plumbing.
    """

    def _entry(self):
        return {
            "pvt_conditions_total": None, "pvt_conditions_completed": None,
            "pvt_current_corner": None, "pvt_current_temperature_c": None, "pvt_current_supply_v": None,
        }

    def test_candidate_generation_complete_sets_total_from_feasible_times_conditions(self):
        # Requirement 3: total = number_of_feasible_candidates * len(pvt_conditions).
        entry = self._entry()
        web_ui._apply_progress_event(entry, {
            "nebula_progress_event": "candidate_generation_complete",
            "n_feasible": 3, "pvt_conditions_total": 6,  # 3 candidates x 2 smoke conditions
        })
        self.assertEqual(entry["pvt_conditions_total"], 6)
        self.assertEqual(entry["pvt_conditions_completed"], 0)

    def test_zero_total_leaves_progress_fields_unset(self):
        # pvt_conditions_total == 0 means "none" mode (or zero feasible
        # candidates) -- must not turn on PVT progress display at all.
        entry = self._entry()
        web_ui._apply_progress_event(entry, {
            "nebula_progress_event": "candidate_generation_complete",
            "n_feasible": 2, "pvt_conditions_total": 0,
        })
        self.assertIsNone(entry["pvt_conditions_total"])
        self.assertIsNone(entry["pvt_conditions_completed"])

    def test_start_event_sets_current_condition_but_does_not_advance_completed(self):
        # Requirement 4: starting a condition must NOT count as completed.
        entry = self._entry()
        entry["pvt_conditions_total"] = 2
        entry["pvt_conditions_completed"] = 0
        web_ui._apply_progress_event(entry, {
            "nebula_progress_event": "pvt_condition_start",
            "corner": "ff", "temperature_c": 125.0, "supply_v": 1.71,
        })
        self.assertEqual(entry["pvt_conditions_completed"], 0)
        self.assertEqual(entry["pvt_current_corner"], "ff")
        self.assertEqual(entry["pvt_current_temperature_c"], 125.0)
        self.assertEqual(entry["pvt_current_supply_v"], 1.71)

    def test_complete_event_advances_completed_count_by_exactly_one(self):
        entry = self._entry()
        entry["pvt_conditions_total"] = 2
        entry["pvt_conditions_completed"] = 0
        web_ui._apply_progress_event(entry, {
            "nebula_progress_event": "pvt_condition_complete",
            "corner": "tt", "temperature_c": 27.0, "supply_v": 1.8, "success": True, "failed_stage": None,
        })
        self.assertEqual(entry["pvt_conditions_completed"], 1)

    def test_complete_event_advances_count_on_definitive_failure_too(self):
        # "completed successfully OR failed definitively" both count.
        entry = self._entry()
        entry["pvt_conditions_total"] = 2
        entry["pvt_conditions_completed"] = 0
        web_ui._apply_progress_event(entry, {
            "nebula_progress_event": "pvt_condition_complete",
            "corner": "ff", "temperature_c": 125.0, "supply_v": 1.71, "success": False, "failed_stage": "transient",
        })
        self.assertEqual(entry["pvt_conditions_completed"], 1)

    def test_full_sequence_reaches_total_completed(self):
        # Requirement 7: "PVT evaluation: 0/2 -> 1/2 -> 2/2".
        entry = self._entry()
        events = [
            {"nebula_progress_event": "candidate_generation_complete", "n_feasible": 1, "pvt_conditions_total": 2},
            {"nebula_progress_event": "pvt_condition_start", "corner": "tt", "temperature_c": 27.0, "supply_v": 1.8},
            {"nebula_progress_event": "pvt_condition_complete", "corner": "tt", "temperature_c": 27.0,
             "supply_v": 1.8, "success": True, "failed_stage": None},
            {"nebula_progress_event": "pvt_condition_start", "corner": "ff", "temperature_c": 125.0, "supply_v": 1.71},
            {"nebula_progress_event": "pvt_condition_complete", "corner": "ff", "temperature_c": 125.0,
             "supply_v": 1.71, "success": True, "failed_stage": None},
        ]
        seen_completed = []
        for event in events:
            web_ui._apply_progress_event(entry, event)
            seen_completed.append(entry["pvt_conditions_completed"])
        self.assertEqual(seen_completed, [0, 0, 1, 1, 2])
        self.assertEqual(entry["pvt_conditions_total"], 2)

    def test_unrecognized_event_kind_is_ignored_not_an_error(self):
        entry = self._entry()
        web_ui._apply_progress_event(entry, {"nebula_progress_event": "something_new_and_unknown"})
        self.assertIsNone(entry["pvt_conditions_total"])  # unchanged, no exception


class StatusPayloadProgressFieldsTests(unittest.TestCase):
    """_status_payload's exposure of progress fields -- "when available"
    (requirement 1), and never for a "none" PVT run (requirement 6).
    """

    def _base_entry(self, run_id, **overrides):
        entry = {
            "run_id": run_id, "status": "running", "command": ["x"],
            "output_path": "/tmp/x.json", "schematic_path": "/tmp/x.spice",
            "started_at": 0.0, "finished_at": None, "returncode": None,
            "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            "pvt_condition_set": None,
            "pvt_conditions_total": None, "pvt_conditions_completed": None,
            "pvt_current_corner": None, "pvt_current_temperature_c": None, "pvt_current_supply_v": None,
        }
        entry.update(overrides)
        return entry

    def test_progress_fields_present_when_pvt_total_is_set(self):
        run_id = "testrun_progress_00000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = self._base_entry(
                run_id, pvt_conditions_total=2, pvt_conditions_completed=1,
                pvt_current_corner="ff", pvt_current_temperature_c=125.0, pvt_current_supply_v=1.71,
            )
        payload = web_ui._status_payload(run_id)
        self.assertEqual(payload["pvt_conditions_total"], 2)
        self.assertEqual(payload["pvt_conditions_completed"], 1)
        self.assertEqual(payload["pvt_progress_fraction"], 0.5)
        self.assertEqual(payload["pvt_current_corner"], "ff")

    def test_none_pvt_run_has_no_progress_fields_at_all(self):
        run_id = "testrun_noneprog_000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = self._base_entry(run_id, pvt_condition_set=None)
        payload = web_ui._status_payload(run_id)
        for key in ("pvt_conditions_total", "pvt_conditions_completed", "pvt_progress_fraction",
                    "pvt_current_corner", "pvt_long_running_full_sweep"):
            self.assertNotIn(key, payload)

    def test_entry_missing_progress_keys_entirely_does_not_break_status(self):
        # Backward-compat: an entry built without the new keys (e.g. by
        # older in-flight code) must not raise a KeyError.
        run_id = "testrun_legacy_0000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "running", "command": ["x"],
                "output_path": "/tmp/x.json", "schematic_path": "/tmp/x.spice",
                "started_at": 0.0, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }
        payload = web_ui._status_payload(run_id)
        self.assertNotIn("pvt_conditions_total", payload)

    def test_minimal27_gets_long_running_label(self):
        run_id = "testrun_full27_0000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = self._base_entry(
                run_id, pvt_condition_set="minimal27", pvt_conditions_total=27, pvt_conditions_completed=3,
            )
        payload = web_ui._status_payload(run_id)
        self.assertTrue(payload["pvt_long_running_full_sweep"])
        # No ETA anywhere -- never computed from elapsed time.
        self.assertNotIn("eta_s", payload)
        self.assertNotIn("estimated_completion", payload)

    def test_smoke_does_not_get_long_running_label(self):
        run_id = "testrun_smokeprog_00000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = self._base_entry(
                run_id, pvt_condition_set="smoke", pvt_conditions_total=2, pvt_conditions_completed=0,
            )
        payload = web_ui._status_payload(run_id)
        self.assertNotIn("pvt_long_running_full_sweep", payload)


class ExecuteRunProgressIntegrationTests(unittest.TestCase):
    """End-to-end: a real short-lived subprocess emitting the exact line
    format experiments/run_autockt_pipeline.py's progress emitter
    produces, read through the real _execute_run/_drain_subprocess_with_
    progress path (no mocking) -- proves the wiring, not just the pure
    helper functions in isolation.
    """

    def test_progress_lines_from_a_real_subprocess_update_status_live(self):
        run_id = "testrun_liveprog_0000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["x"],
                "output_path": "/tmp/nope5.json", "schematic_path": "/tmp/nope5.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
                "pvt_condition_set": "smoke",
                "pvt_conditions_total": None, "pvt_conditions_completed": None,
                "pvt_current_corner": None, "pvt_current_temperature_c": None, "pvt_current_supply_v": None,
            }
        script = (
            "import json, sys, time\n"
            "def emit(d):\n"
            "    print(json.dumps(d)); sys.stdout.flush()\n"
            "emit({'nebula_progress_event': 'candidate_generation_complete', "
            "'n_feasible': 1, 'pvt_conditions_total': 2})\n"
            "emit({'nebula_progress_event': 'pvt_condition_start', 'corner': 'tt', "
            "'temperature_c': 27.0, 'supply_v': 1.8})\n"
            "emit({'nebula_progress_event': 'pvt_condition_complete', 'corner': 'tt', "
            "'temperature_c': 27.0, 'supply_v': 1.8, 'success': True, 'failed_stage': None})\n"
            "emit({'nebula_progress_event': 'pvt_condition_start', 'corner': 'ff', "
            "'temperature_c': 125.0, 'supply_v': 1.71})\n"
            "print('ngspice: some unrelated stderr-ish stdout noise')\n"  # malformed/irrelevant line
            "emit({'nebula_progress_event': 'pvt_condition_complete', 'corner': 'ff', "
            "'temperature_c': 125.0, 'supply_v': 1.71, 'success': True, 'failed_stage': None})\n"
            "sys.exit(1)\n"
        )
        argv = [sys.executable, "-c", script]
        web_ui._execute_run(run_id, argv, Path("/tmp/nope5.json"), Path("/tmp/nope5.spice"))

        payload = web_ui._status_payload(run_id)
        self.assertEqual(payload["pvt_conditions_total"], 2)
        self.assertEqual(payload["pvt_conditions_completed"], 2)
        self.assertEqual(payload["pvt_progress_fraction"], 1.0)
        self.assertEqual(payload["pvt_current_corner"], "ff")
        # The interleaved non-JSON line must not have broken anything.
        self.assertIn("unrelated stderr-ish stdout noise", payload["stdout_tail"])

    def test_full_pvt_timeout_exceeds_a_measured_single_sweep(self):
        timeout_s = web_ui._estimate_timeout_s({"pvt_condition_set": "minimal27", "episodes": 1})
        self.assertGreater(timeout_s, 138.4 * 60)

    def test_full_pvt_timeout_scales_with_possible_candidates(self):
        one = web_ui._estimate_timeout_s({"pvt_condition_set": "minimal27", "episodes": 1})
        three = web_ui._estimate_timeout_s({"pvt_condition_set": "minimal27", "episodes": 3})
        self.assertGreater(three, one)

    def test_non_full_pvt_keeps_bounded_default(self):
        self.assertEqual(
            web_ui._estimate_timeout_s({"pvt_condition_set": "smoke", "episodes": 99}),
            web_ui.RUN_TIMEOUT_S,
        )


class PvtOptionsMatchPipelineTests(unittest.TestCase):
    """FINAL AUDIT gap D: web_ui.py deliberately keeps its own plain-string
    PVT_CONDITION_SETS tuple (not an import of run_autockt_pipeline, so
    this lightweight server never has to import torch/simulator/rl at
    startup) -- this test is the drift guard that relationship needs.
    """

    def test_web_ui_pvt_options_match_the_pipelines_own_condition_sets(self):
        from experiments.run_autockt_pipeline import PVT_CONDITION_SETS as pipeline_sets
        self.assertEqual(set(web_ui.PVT_CONDITION_SETS), set(pipeline_sets.keys()))

    def test_none_is_still_valid_and_first(self):
        self.assertEqual(web_ui.PVT_CONDITION_SETS[0], "none")

    def test_minimal27_is_exposed(self):
        self.assertIn("minimal27", web_ui.PVT_CONDITION_SETS)


class Hd3NoiseUiTests(unittest.TestCase):
    def test_checkbox_exists_and_is_wired_to_the_flag(self):
        self.assertIn('id="measureHd3Noise"', web_ui.INDEX_HTML)
        self.assertIn("measure_hd3_noise: $('measureHd3Noise').checked", web_ui.INDEX_HTML)

    def test_build_argv_passes_the_flag_only_when_requested(self):
        base = {"target_mode": "trivial", "backend": "real", "episodes": 1, "horizon": 1,
                "pvt_condition_set": "none", "trade_off_preference": "most_robust"}
        argv_off = web_ui._build_argv(base, output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"))
        self.assertNotIn("--measure-hd3-noise", argv_off)

        argv_on = web_ui._build_argv(
            {**base, "measure_hd3_noise": True}, output_path=Path("/tmp/x.json"), schematic_path=Path("/tmp/x.spice"),
        )
        self.assertIn("--measure-hd3-noise", argv_on)


class PvtLabelingHonestyTests(unittest.TestCase):
    """FINAL AUDIT gap D wording audit: no UI text may imply that
    nominal-only ("none") or "smoke" selection constitutes PVT robustness.
    """

    def test_none_option_explicitly_says_nominal_only(self):
        self.assertIn("NOMINAL-ONLY", web_ui.INDEX_HTML)

    def test_smoke_option_explicitly_disclaims_being_a_robustness_proof(self):
        idx = web_ui.INDEX_HTML.index('value="smoke"')
        snippet = web_ui.INDEX_HTML[idx:idx + 200]
        self.assertIn("not a robustness proof", snippet)

    def test_full_sweep_option_is_labeled_with_its_real_cost(self):
        idx = web_ui.INDEX_HTML.index('value="minimal27"')
        snippet = web_ui.INDEX_HTML[idx:idx + 200]
        self.assertIn("SLOW", snippet)
        self.assertIn("27-point", snippet)

    def test_hint_text_states_none_and_smoke_do_not_establish_robustness(self):
        self.assertIn("do NOT establish PVT robustness", web_ui.INDEX_HTML)


class HttpIntegrationTests(unittest.TestCase):
    """Full request/response cycle over real HTTP, using --backend synthetic
    (no SPICE) end to end -- exercises the same code path a browser would.
    """

    @classmethod
    def setUpClass(cls):
        cls.server = web_ui.build_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path: str) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as resp:  # type: HTTPResponse
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_index_page_serves_html(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/") as resp:
            body = resp.read().decode("utf-8")
        self.assertEqual(resp.status, 200)
        self.assertIn("Run NEBULA", body)

    def test_checkpoints_endpoint_lists_pt_files(self):
        status, data = self._get("/api/checkpoints")
        self.assertEqual(status, 200)
        self.assertIsInstance(data["checkpoints"], list)

    def test_evidence_endpoint_returns_multiple_graphs_and_provenance(self):
        status, data = self._get("/api/evidence")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(data["charts"]), 6)
        self.assertTrue(data["sources"])

    def test_invalid_request_returns_400_with_problems(self):
        status, data = self._post("/api/run", {"target_mode": "bogus"})
        self.assertEqual(status, 400)
        self.assertIn("problems", data)

    def test_unknown_run_id_returns_404(self):
        status, data = self._get("/api/status/" + "0" * 32)
        self.assertEqual(status, 404)

    def test_saved_historical_result_can_be_loaded(self):
        run_id = "07acbcab1cd346cca24eab31919496c0"
        result_path = web_ui.RUNS_DIR / f"{run_id}.json"
        if not result_path.is_file():
            self.skipTest("recorded 27-corner UI run is not present in this checkout")
        status, payload = self._get(f"/api/result/{run_id}")
        self.assertEqual(status, 200)
        self.assertEqual(payload["run_id"], run_id)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["result"]["pvt_pattern_bits"], 512)
        self.assertEqual(payload["result"]["selection"]["pvt"]["n_conditions"], 27)

    def test_unknown_historical_result_returns_404(self):
        status, data = self._get("/api/result/" + "0" * 32)
        self.assertEqual(status, 404)

    def test_malformed_historical_result_id_returns_400(self):
        status, data = self._get("/api/result/not-a-valid-id")
        self.assertEqual(status, 400)

    def test_malformed_run_id_returns_400(self):
        status, data = self._get("/api/status/not-a-valid-id")
        self.assertEqual(status, 400)

    def test_full_synthetic_run_completes_and_reports_a_final_specification(self):
        with TemporaryDirectory():
            status, data = self._post("/api/run", {
                "target_mode": "trivial", "backend": "synthetic", "checkpoint": None,
                "episodes": 8, "horizon": 1, "initial_indices_source": "grid-center",
                "pvt_condition_set": "none", "trade_off_preference": "most_robust",
            })
            self.assertEqual(status, 202)
            run_id = data["run_id"]

            deadline = threading.Event()
            result_payload = None
            for _ in range(60):
                s, payload = self._get(f"/api/status/{run_id}")
                if payload["status"] not in ("queued", "running"):
                    result_payload = payload
                    break
                deadline.wait(0.5)

            self.assertIsNotNone(result_payload, "synthetic run did not finish in time")
            self.assertEqual(result_payload["status"], "completed")
            self.assertIn("final_specification", result_payload["result"])
            self.assertIn("rows", result_payload["result"]["final_specification"])


if __name__ == "__main__":
    unittest.main()
