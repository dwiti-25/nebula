"""SPICE-free tests for nebula.llm_wrapper -- the thin natural-language
interface around experiments/run_autockt_pipeline.py.

Nothing here invokes real ngspice or PPO training: subprocess calls are
mocked (matching tests/test_web_ui.py's own pattern for the same
pipeline), and the LLM call is exercised only through injected
success/failure paths -- no live network call, no real API key needed.
"""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from rl.target_spec import EXISTING_THRESHOLDS, SPEC_NAMES

from nebula.llm_providers import (
    AnthropicLLMProvider,
    MockLLMProvider,
    ProviderResult,
    get_provider,
)
from nebula.llm_wrapper import (
    build_pipeline_argv,
    format_report,
    summarize_result,
)
from nebula.target_parsing import DEFAULT_TARGET, parse_request_text


class ParseRequestTextTests(unittest.TestCase):
    def test_extracts_explicit_eye_width_and_power_with_unit_conversion(self):
        parsed = parse_request_text(
            "Design a low-power PCIe Gen-2 receiver with eye width above 0.4 UI and power below 15 mW."
        )
        self.assertEqual(parsed.fields["dfe_eye_width_ui"], 0.4)
        self.assertAlmostEqual(parsed.fields["ctle_power_w"], 0.015)
        self.assertIn("dfe_eye_width_ui", parsed.explicit_fields)
        self.assertIn("ctle_power_w", parsed.explicit_fields)

    def test_unmentioned_fields_fall_back_to_existing_repo_defaults(self):
        parsed = parse_request_text("Design a receiver with eye width above 0.4 UI.")
        self.assertEqual(parsed.fields["dfe_locked_phase_eye_height_v"], EXISTING_THRESHOLDS["dfe_locked_phase_eye_height_v"])
        self.assertEqual(parsed.fields["dfe_min_margin_v"], EXISTING_THRESHOLDS["dfe_min_margin_v"])
        self.assertNotIn("dfe_locked_phase_eye_height_v", parsed.explicit_fields)
        self.assertNotIn("dfe_min_margin_v", parsed.explicit_fields)

    def test_schema_matches_existing_target_spec_field_names(self):
        parsed = parse_request_text("anything")
        self.assertEqual(set(parsed.fields), set(SPEC_NAMES))

    def test_qualitative_only_language_does_not_fabricate_a_numeric_threshold(self):
        parsed = parse_request_text("Design a low-power receiver.")
        self.assertEqual(parsed.fields["ctle_power_w"], DEFAULT_TARGET["ctle_power_w"])
        self.assertNotIn("ctle_power_w", parsed.explicit_fields)
        self.assertTrue(any("low-power" in note for note in parsed.unquantified_notes))

    def test_power_without_a_unit_is_not_parsed(self):
        # "15" alone is ambiguous (15 W vs 15 mW) -- must not guess.
        parsed = parse_request_text("power below 15")
        self.assertNotIn("ctle_power_w", parsed.explicit_fields)

    def test_millivolt_margin_is_converted_to_volts(self):
        parsed = parse_request_text("margin above 50 mV")
        self.assertAlmostEqual(parsed.fields["dfe_min_margin_v"], 0.05)
        self.assertIn("dfe_min_margin_v", parsed.explicit_fields)


class MissingApiKeyFallbackTests(unittest.TestCase):
    def test_anthropic_provider_falls_back_to_mock_on_any_call_failure(self):
        provider = AnthropicLLMProvider(api_key=None)
        with patch.object(AnthropicLLMProvider, "_call_claude", side_effect=RuntimeError("no api key configured")):
            result = provider.parse_target("eye width above 0.4 UI")
        self.assertEqual(result.provider_used, "mock")
        self.assertIsNotNone(result.fallback_reason)
        self.assertIn("no api key configured", result.fallback_reason)
        # the fallback must still produce a usable, correctly-parsed target
        self.assertEqual(result.parsed.fields["dfe_eye_width_ui"], 0.4)

    def test_get_provider_mock_never_touches_the_llm_path(self):
        provider, requested = get_provider(provider_name="mock")
        self.assertIsInstance(provider, MockLLMProvider)
        self.assertEqual(requested, "mock")
        result = provider.parse_target("eye width above 0.4 UI")
        self.assertEqual(result.provider_used, "mock")
        self.assertIsNone(result.fallback_reason)

    def test_get_provider_defaults_to_anthropic_when_unset(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("NEBULA_LLM_PROVIDER", None)
            provider, requested = get_provider()
        self.assertEqual(requested, "anthropic")
        self.assertIsInstance(provider, AnthropicLLMProvider)

    def test_unknown_provider_name_raises(self):
        with self.assertRaises(ValueError):
            get_provider(provider_name="not-a-real-provider")


class BuildPipelineArgvTests(unittest.TestCase):
    def test_argv_carries_target_json_and_backend(self):
        argv = build_pipeline_argv(
            target={"dfe_locked_phase_eye_height_v": 0.1, "dfe_eye_width_ui": 0.4,
                    "dfe_min_margin_v": 0.0, "ctle_power_w": 0.015},
            output_path=Path("/tmp/x.json"), backend="synthetic", checkpoint=None,
            rl_version="v1", episodes=3, horizon=4, measure_hd3_noise=False,
            pvt_condition_set="none",
        )
        self.assertIn("experiments.run_autockt_pipeline", argv)
        self.assertIn("--backend", argv)
        self.assertEqual(argv[argv.index("--backend") + 1], "synthetic")
        target_json = argv[argv.index("--target-json") + 1]
        self.assertEqual(json.loads(target_json)["dfe_eye_width_ui"], 0.4)
        self.assertNotIn("--checkpoint", argv)
        self.assertNotIn("--measure-hd3-noise", argv)

    def test_checkpoint_and_measure_hd3_noise_are_passed_through_when_set(self):
        argv = build_pipeline_argv(
            target=dict(DEFAULT_TARGET), output_path=Path("/tmp/x.json"), backend="real",
            checkpoint="results/some_policy.pt", rl_version="v1", episodes=3, horizon=4,
            measure_hd3_noise=True, pvt_condition_set="none",
        )
        self.assertIn("--checkpoint", argv)
        self.assertEqual(argv[argv.index("--checkpoint") + 1], "results/some_policy.pt")
        self.assertIn("--measure-hd3-noise", argv)


def _fake_process(returncode=0, stderr=""):
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.returncode = returncode
    proc.stderr = stderr
    return proc


def _provider_result(explicit=("dfe_eye_width_ui", "ctle_power_w"), fallback_reason=None):
    from nebula.target_parsing import ParsedTarget

    fields = dict(DEFAULT_TARGET)
    fields["dfe_eye_width_ui"] = 0.4
    fields["ctle_power_w"] = 0.015
    return ProviderResult(
        parsed=ParsedTarget(fields=fields, explicit_fields=tuple(explicit)),
        provider_used="mock" if fallback_reason else "anthropic",
        fallback_reason=fallback_reason,
    )


# A realistic fixture matching analysis/final_specification.py's real,
# already-verified output shape (see build_final_specification_report).
_FEASIBLE_PIPELINE_RESULT = {
    "backend": "synthetic",
    "selection": {"reason": "selected"},
    "final_specification": {
        "design_id": "ep0",
        "parameters": {"rload_ohm": 2511.89, "rdeg_ohm": 446.68, "cdeg_f": 4.4668e-13,
                        "itail_a": 6.30957e-4, "dfe_tap_v": -0.019},
        "rows": [
            {"metric": "Eye width (UI)", "measured": "0.5373", "requirement": "> 0.4 UI", "verdict": "PASS", "source": "x"},
            {"metric": "Power (W)", "measured": "0.0147", "requirement": "0 < power < 0.015 W", "verdict": "PASS", "source": "x"},
            {"metric": "HD3 (dB)", "measured": None, "requirement": "< -30 dB", "verdict": "NOT CLAIMED", "source": "x"},
            {"metric": "Input-referred noise (Vrms)", "measured": None, "requirement": "< 0.0015 Vrms", "verdict": "NOT CLAIMED", "source": "x"},
            {"metric": "PVT (pass/total)", "measured": None, "requirement": "36-condition grid", "verdict": "NOT CLAIMED", "source": "x"},
        ],
    },
}

_INFEASIBLE_PIPELINE_RESULT = {
    "backend": "synthetic",
    "selection": {"reason": "no nominally feasible candidates"},
}


class MetricPropagationTests(unittest.TestCase):
    def test_measured_metrics_and_verdicts_are_copied_verbatim(self):
        report = summarize_result(
            request="Design a low-power receiver with eye width above 0.4 UI and power below 15 mW.",
            provider_result=_provider_result(), unquantified_notes=(), backend="synthetic",
            checkpoint="results/policy.pt", runtime_s=2.5, pipeline_argv=["python"],
            process=_fake_process(0), pipeline_result=_FEASIBLE_PIPELINE_RESULT,
        )
        self.assertEqual(report.selected_parameters["rload_ohm"], 2511.89)
        eye_width_row = next(r for r in report.spec_rows if r["metric"] == "Eye width (UI)")
        self.assertEqual(eye_width_row["measured"], "0.5373")
        self.assertEqual(eye_width_row["verdict"], "PASS")
        # identity, not recomputation, against the fixture's own object
        self.assertIs(report.spec_rows, _FEASIBLE_PIPELINE_RESULT["final_specification"]["rows"])

    def test_format_report_renders_the_propagated_rows(self):
        report = summarize_result(
            request="req", provider_result=_provider_result(), unquantified_notes=(),
            backend="synthetic", checkpoint="results/policy.pt", runtime_s=1.0,
            pipeline_argv=["python"], process=_fake_process(0), pipeline_result=_FEASIBLE_PIPELINE_RESULT,
        )
        text = format_report(report)
        self.assertIn("0.5373", text)
        self.assertIn("PASS", text)
        self.assertIn("rload_ohm", text)


class NotClaimedStatusTests(unittest.TestCase):
    def test_hd3_noise_pvt_report_not_claimed_when_unmeasured(self):
        report = summarize_result(
            request="req", provider_result=_provider_result(), unquantified_notes=(),
            backend="synthetic", checkpoint="results/policy.pt", runtime_s=1.0,
            pipeline_argv=["python"], process=_fake_process(0), pipeline_result=_FEASIBLE_PIPELINE_RESULT,
        )
        text = format_report(report)
        self.assertIn("HD3 status:   NOT CLAIMED", text)
        self.assertIn("Noise status: NOT CLAIMED", text)
        self.assertIn("PVT status:   NOT CLAIMED", text)

    def test_synthetic_backend_warns_metrics_are_not_real_measurements(self):
        report = summarize_result(
            request="req", provider_result=_provider_result(), unquantified_notes=(),
            backend="synthetic", checkpoint="results/policy.pt", runtime_s=1.0,
            pipeline_argv=["python"], process=_fake_process(0), pipeline_result=_FEASIBLE_PIPELINE_RESULT,
        )
        self.assertTrue(any("NOT real circuit measurements" in w for w in report.warnings))

    def test_missing_checkpoint_warns_policy_is_untrained(self):
        report = summarize_result(
            request="req", provider_result=_provider_result(), unquantified_notes=(),
            backend="synthetic", checkpoint=None, runtime_s=1.0,
            pipeline_argv=["python"], process=_fake_process(0), pipeline_result=_FEASIBLE_PIPELINE_RESULT,
        )
        self.assertTrue(any("UNTRAINED policy" in w for w in report.warnings))


class NoFabricatedResultsTests(unittest.TestCase):
    def test_no_selected_design_reports_no_parameters_and_no_metrics(self):
        report = summarize_result(
            request="req", provider_result=_provider_result(), unquantified_notes=(),
            backend="synthetic", checkpoint="results/policy.pt", runtime_s=1.0,
            pipeline_argv=["python"], process=_fake_process(0), pipeline_result=_INFEASIBLE_PIPELINE_RESULT,
        )
        self.assertIsNone(report.selected_parameters)
        self.assertEqual(report.spec_rows, [])
        self.assertEqual(report.selection_reason, "no nominally feasible candidates")
        self.assertTrue(any("no feasible design was selected" in w for w in report.warnings))

    def test_no_selected_design_report_text_does_not_print_a_parameter_table(self):
        report = summarize_result(
            request="req", provider_result=_provider_result(), unquantified_notes=(),
            backend="synthetic", checkpoint="results/policy.pt", runtime_s=1.0,
            pipeline_argv=["python"], process=_fake_process(0), pipeline_result=_INFEASIBLE_PIPELINE_RESULT,
        )
        text = format_report(report)
        self.assertIn("No feasible design selected", text)
        self.assertNotIn("Selected circuit parameters:", text)

    def test_missing_pipeline_output_does_not_crash_or_invent_a_result(self):
        report = summarize_result(
            request="req", provider_result=_provider_result(), unquantified_notes=(),
            backend="synthetic", checkpoint="results/policy.pt", runtime_s=1.0,
            pipeline_argv=["python"], process=_fake_process(1, stderr="boom"), pipeline_result=None,
        )
        self.assertIsNone(report.selected_parameters)
        self.assertEqual(report.spec_rows, [])
        self.assertTrue(any("exited with code 1" in w for w in report.warnings))

    def test_llm_fallback_is_disclosed_as_a_warning_not_hidden(self):
        report = summarize_result(
            request="req", provider_result=_provider_result(fallback_reason="AuthenticationError: invalid api key"),
            unquantified_notes=(), backend="synthetic", checkpoint="results/policy.pt", runtime_s=1.0,
            pipeline_argv=["python"], process=_fake_process(0), pipeline_result=_FEASIBLE_PIPELINE_RESULT,
        )
        self.assertTrue(any("fallback" in w.lower() for w in report.warnings))


if __name__ == "__main__":
    unittest.main()
