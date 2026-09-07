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
        problems = web_ui._validate_request(self._base(pvt_condition_set="full60"))
        self.assertTrue(any("pvt_condition_set" in p for p in problems))


class BuildArgvTests(unittest.TestCase):
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
    docs/autockt-mapping.md sec 23 deterministically, via a mocked
    subprocess.Popen -- so this test does not depend on the crash actually
    occurring, and runs with zero real SPICE.
    """

    def test_signal_terminated_subprocess_is_reported_clearly_not_as_a_hang(self):
        run_id = "testrun_crash_00000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["python", "-m", "x"],
                "output_path": "/tmp/nope.json", "schematic_path": "/tmp/nope.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("partial stdout before crash", "")
        mock_proc.returncode = -11  # SIGSEGV, as subprocess.Popen reports on POSIX
        with patch("experiments.web_ui.subprocess.Popen", return_value=mock_proc):
            web_ui._execute_run(run_id, ["python", "-m", "x"], Path("/tmp/nope.json"), Path("/tmp/nope.spice"))

        payload = web_ui._status_payload(run_id)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("signal 11", payload["error"])
        self.assertIn("SIGSEGV", payload["error"])

    def test_nonzero_exit_without_signal_is_reported_as_a_normal_failure(self):
        run_id = "testrun_fail_000000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["python", "-m", "x"],
                "output_path": "/tmp/nope2.json", "schematic_path": "/tmp/nope2.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "ValueError: invalid target specification")
        mock_proc.returncode = 1
        with patch("experiments.web_ui.subprocess.Popen", return_value=mock_proc):
            web_ui._execute_run(run_id, ["python", "-m", "x"], Path("/tmp/nope2.json"), Path("/tmp/nope2.spice"))

        payload = web_ui._status_payload(run_id)
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("signal", payload["error"])
        self.assertIn("non-zero status (1)", payload["error"])


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


class SubprocessTimeoutTests(unittest.TestCase):
    """FINAL AUDIT gap E: bounded outer-process handling for a HUNG (not
    crashed) subprocess -- distinct from the SIGSEGV path above. Verifies
    the subprocess is killed and reaped, and the run is reported as a
    clean timeout, never retried automatically.
    """

    def test_hung_subprocess_is_killed_and_reported_as_a_timeout(self):
        import subprocess as subprocess_module

        run_id = "testrun_timeout_0000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["python", "-m", "x"],
                "output_path": "/tmp/nope3.json", "schematic_path": "/tmp/nope3.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }

        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = [
            subprocess_module.TimeoutExpired(cmd="x", timeout=web_ui.RUN_TIMEOUT_S),
            ("partial output before the hang was killed", ""),
        ]
        mock_proc.returncode = -9  # SIGKILL, after proc.kill()

        with patch("experiments.web_ui.subprocess.Popen", return_value=mock_proc):
            web_ui._execute_run(run_id, ["python", "-m", "x"], Path("/tmp/nope3.json"), Path("/tmp/nope3.spice"))

        mock_proc.kill.assert_called_once()
        payload = web_ui._status_payload(run_id)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("did not finish within", payload["error"])
        self.assertIn("hang", payload["error"])
        self.assertIn("partial output", payload["stdout_tail"])

    def test_normal_completion_never_calls_kill(self):
        run_id = "testrun_normal_00000000000000"
        with web_ui._RUNS_LOCK:
            web_ui._RUNS[run_id] = {
                "run_id": run_id, "status": "queued", "command": ["python", "-m", "x"],
                "output_path": "/tmp/nope4.json", "schematic_path": "/tmp/nope4.spice",
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "result": None, "stdout_tail": "", "stderr_tail": "",
            }
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 1
        with patch("experiments.web_ui.subprocess.Popen", return_value=mock_proc):
            web_ui._execute_run(run_id, ["python", "-m", "x"], Path("/tmp/nope4.json"), Path("/tmp/nope4.spice"))
        mock_proc.kill.assert_not_called()

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
