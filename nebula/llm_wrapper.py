"""CLI entry point: python -m nebula.llm_wrapper "<natural-language request>"

Thin orchestration only -- see nebula/__init__.py. This module never
imports simulator/, rl/ppo_agent.py, or any measurement/selection code; it
only imports rl/target_spec.py (pure dataclasses, see
nebula/target_parsing.py) and shells out to the existing, unmodified
`python -m experiments.run_autockt_pipeline` CLI for every actual
evaluation -- the same entry point experiments/web_ui.py already uses.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from nebula.llm_providers import ProviderResult, get_provider

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "results" / "llm_wrapper_runs"
DEFAULT_TIMEOUT_S = 1800.0  # generous but bounded; --backend real + PVT sweeps can be slow


# Substrings of known checkpoint-incompatibility errors raised by
# rl/runtime_contract.py::validate_inference_contract / grids_from_contract
# and experiments/run_autockt_pipeline.py::generate_candidates -- surfaced
# as a distinct, clear error rather than a raw stack-trace dump (Phase 2:
# "add a clear error if the checkpoint is missing or incompatible").
_CHECKPOINT_ERROR_MARKERS = (
    "requires a v3 metadata-bearing policy export",
    "Incompatible v3 checkpoint",
    "does not match the requested PPO version",
    "Checkpoint state schema does not match",
    "Invalid v3 checkpoint",
    "checkpoint not found",
    "state schema does not match",
)


@dataclass(frozen=True)
class WrapperReport:
    request: str
    provider_used: str
    fallback_reason: str | None
    requested_target: dict[str, float]
    explicit_fields: tuple[str, ...]
    unquantified_notes: tuple[str, ...]
    backend: str
    checkpoint_path: str | None
    rl_version: str
    runtime_s: float
    pipeline_argv: list[str]
    pipeline_exit_code: int
    selected_parameters: dict[str, float] | None
    spec_rows: list[dict[str, str | None]]
    selection_reason: str | None
    checkpoint_error: str | None = None
    warnings: list[str] = field(default_factory=list)


def build_pipeline_argv(
    *,
    target: dict[str, float],
    output_path: Path,
    backend: str,
    checkpoint: str | None,
    rl_version: str,
    episodes: int,
    horizon: int,
    measure_hd3_noise: bool,
    pvt_condition_set: str,
) -> list[str]:
    """Pure function -- the exact argv for the existing, unmodified
    experiments.run_autockt_pipeline CLI. No selection/measurement logic
    lives here; this only serializes already-decided options.
    """

    argv = [
        sys.executable, "-m", "experiments.run_autockt_pipeline",
        "--target-json", json.dumps(target),
        "--backend", backend,
        "--rl-version", rl_version,
        "--episodes", str(episodes),
        "--horizon", str(horizon),
        "--pvt-condition-set", pvt_condition_set,
        "--output", str(output_path),
    ]
    if checkpoint:
        argv += ["--checkpoint", checkpoint]
    if measure_hd3_noise:
        argv.append("--measure-hd3-noise")
    return argv


def run_pipeline_subprocess(argv: list[str], *, timeout_s: float, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)


def load_pipeline_result(output_path: Path) -> dict:
    return json.loads(output_path.read_text(encoding="utf-8"))


def summarize_result(
    *,
    request: str,
    provider_result: ProviderResult,
    unquantified_notes: tuple[str, ...],
    backend: str,
    checkpoint: str | None,
    rl_version: str,
    runtime_s: float,
    pipeline_argv: list[str],
    process: subprocess.CompletedProcess,
    pipeline_result: dict | None,
) -> WrapperReport:
    """Pure function over already-known data -- assembles the report from
    the pipeline's own output. Never computes or infers a metric value:
    every row in `spec_rows` is copied verbatim from
    analysis/final_specification.py's existing PASS/FAIL/NOT CLAIMED
    output (via pipeline_result["final_specification"]["rows"]).
    """

    warnings: list[str] = []
    if checkpoint is None:
        warnings.append(
            "no --checkpoint given: candidate generation used a freshly-initialized, "
            "UNTRAINED policy (undirected rollout), not a learned design search. "
            "Pass --checkpoint <path> to use an existing trained policy."
        )
    if backend == "synthetic":
        warnings.append(
            "backend=synthetic: all metrics below come from "
            "rl.synthetic_benchmark, a fast deterministic stand-in for PPO-training "
            "mechanics -- NOT real circuit measurements. Pass --backend real for "
            "actual SPICE-measured results (real SPICE evaluation is slow: "
            "roughly 15-65s per evaluation)."
        )
    if provider_result.fallback_reason:
        warnings.append(
            f"LLM provider unavailable, used deterministic parser fallback instead "
            f"({provider_result.fallback_reason})."
        )
    for note in unquantified_notes:
        warnings.append(note)

    checkpoint_error: str | None = None
    if process.returncode != 0:
        stderr_tail = process.stderr.strip()[-4000:]
        marker = next((m for m in _CHECKPOINT_ERROR_MARKERS if m in stderr_tail), None)
        if marker is not None:
            # Extract the specific ValueError line for a clean, non-stack-trace message.
            error_line = next(
                (line for line in reversed(stderr_tail.splitlines()) if marker in line), marker,
            )
            checkpoint_error = (
                f"checkpoint '{checkpoint}' is missing or incompatible with --rl-version {rl_version}: "
                f"{error_line.strip()}"
            )
            warnings.append(checkpoint_error)
        else:
            warnings.append(f"pipeline exited with code {process.returncode}: {stderr_tail[-2000:]}")

    selected_parameters: dict[str, float] | None = None
    spec_rows: list[dict[str, str | None]] = []
    selection_reason: str | None = None
    if pipeline_result is not None:
        selection = pipeline_result.get("selection") or {}
        selection_reason = selection.get("reason")
        final_spec = pipeline_result.get("final_specification")
        if final_spec is not None:
            selected_parameters = final_spec.get("parameters")
            spec_rows = final_spec.get("rows", [])
            for label, substring in (("PVT", "PVT"), ("HD3", "HD3"), ("noise", "noise")):
                if _status_for(spec_rows, substring).startswith("NOT CLAIMED"):
                    warnings.append(f"{label} was NOT CLAIMED for this run (not executed) -- not a pass.")
        else:
            warnings.append(
                "no feasible design was selected for this target -- "
                "no circuit parameters or measured metrics to report."
            )

    return WrapperReport(
        request=request,
        provider_used=provider_result.provider_used,
        fallback_reason=provider_result.fallback_reason,
        requested_target=provider_result.parsed.fields,
        explicit_fields=provider_result.parsed.explicit_fields,
        unquantified_notes=unquantified_notes,
        backend=backend,
        checkpoint_path=checkpoint,
        rl_version=rl_version,
        runtime_s=runtime_s,
        pipeline_argv=pipeline_argv,
        pipeline_exit_code=process.returncode,
        selected_parameters=selected_parameters,
        spec_rows=spec_rows,
        selection_reason=selection_reason,
        checkpoint_error=checkpoint_error,
        warnings=warnings,
    )


def _status_for(spec_rows: list[dict[str, str | None]], metric_substring: str) -> str:
    matches = [row for row in spec_rows if metric_substring.lower() in row["metric"].lower()]
    if not matches:
        return "NOT CLAIMED (not reported by this run)"
    verdicts = {row["verdict"] for row in matches}
    if len(verdicts) == 1:
        return verdicts.pop()
    return "/".join(sorted(verdicts))


def _measured_for(spec_rows: list[dict[str, str | None]], metric_prefix: str) -> str:
    row = next((r for r in spec_rows if r["metric"].lower().startswith(metric_prefix.lower())), None)
    if row is None:
        return "NOT CLAIMED"
    measured = row["measured"] if row["measured"] is not None else "n/a"
    return f"{measured} ({row['verdict']})"


def _overall_verdict(report: WrapperReport) -> str:
    if report.checkpoint_error:
        return "ERROR (checkpoint incompatible -- see warnings)"
    if report.selected_parameters is None:
        return "NO FEASIBLE DESIGN"
    if any(row["verdict"] == "FAIL" for row in report.spec_rows):
        return "FAIL"
    return "PASS"


def format_report(report: WrapperReport) -> str:
    lines = [
        "NEBULA natural-language wrapper -- orchestration/formatting layer only.",
        "This tool does NOT claim the LLM improves circuit quality; all circuit",
        "results below come unmodified from experiments/run_autockt_pipeline.py.",
        "",
        f"Request: {report.request}",
        f"Target-parsing provider: {report.provider_used}"
        + (f" (fallback: {report.fallback_reason})" if report.fallback_reason else ""),
        f"Requested target ({'explicit: ' + ', '.join(report.explicit_fields) if report.explicit_fields else 'all defaults'}):",
    ]
    for name, value in report.requested_target.items():
        marker = "*" if name in report.explicit_fields else " "
        lines.append(f"  {marker} {name:35s} = {value:.6g}")
    lines += [
        "",
        f"RL version: {report.rl_version}",
        f"Backend used: {report.backend}",
        f"Checkpoint: {report.checkpoint_path or '(none -- untrained policy, see warnings)'}",
        f"Runtime: {report.runtime_s:.2f}s",
        f"Pipeline exit code: {report.pipeline_exit_code}",
        "",
    ]
    if report.selected_parameters is not None:
        lines.append("Selected circuit parameters:")
        for name, value in report.selected_parameters.items():
            lines.append(f"  {name:15s} = {value:.6g}")
        lines.append("")
        lines.append(f"{'Metric':30s} {'Measured':15s} {'Requirement':45s} {'Verdict'}")
        for row in report.spec_rows:
            measured = row["measured"] if row["measured"] is not None else "n/a"
            lines.append(f"{row['metric']:30s} {measured:15s} {row['requirement']:45s} {row['verdict']}")
        lines.append("")
        lines.append(f"PVT status:   {_status_for(report.spec_rows, 'PVT')}")
        lines.append(f"HD3 status:   {_status_for(report.spec_rows, 'HD3')}")
        lines.append(f"Noise status: {_status_for(report.spec_rows, 'noise')}")
    else:
        lines.append(f"No feasible design selected. Reason: {report.selection_reason or 'unknown'}")

    if report.warnings:
        lines += ["", "Warnings:"]
        lines += [f"  - {w}" for w in report.warnings]

    lines += [
        "",
        "## NEBULA FINAL RESULT",
        "",
        f"Request: {report.request}",
        f"RL version: {report.rl_version}",
        f"Checkpoint: {report.checkpoint_path or 'none (untrained)'}",
        f"Backend: {report.backend}",
        f"Runtime: {report.runtime_s:.2f}s",
        "",
        f"Eye width:              {_measured_for(report.spec_rows, 'Eye width')}",
        f"Eye height:             {_measured_for(report.spec_rows, 'Eye height')}",
        f"Margin:                 {_measured_for(report.spec_rows, 'Margin')}",
        f"Power:                  {_measured_for(report.spec_rows, 'Power')}",
        f"Peaking:                {_measured_for(report.spec_rows, 'Peaking')}",
        f"HD3:                    {_measured_for(report.spec_rows, 'HD3')}",
        f"Input-referred noise:   {_measured_for(report.spec_rows, 'Input-referred noise')}",
        f"PVT:                    {_measured_for(report.spec_rows, 'PVT')}",
        "",
        f"Overall verdict: {_overall_verdict(report)}",
        f"Warnings: {len(report.warnings)}" + (" (see above)" if report.warnings else " (none)"),
    ]

    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nebula.llm_wrapper",
        description="Thin natural-language interface around experiments/run_autockt_pipeline.py. "
        "Does not improve circuit quality; only parses a request into the existing target "
        "schema and reports the existing pipeline's own results.",
    )
    parser.add_argument("request", help='e.g. "Design a low-power PCIe Gen-2 receiver with eye width above 0.4 UI and power below 15 mW."')
    parser.add_argument("--backend", choices=("synthetic", "real"), default="synthetic",
                        help="default synthetic (fast, no real SPICE cost). 'real' calls actual ngspice "
                        "and is slow (~15-65s/evaluation).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to an existing trained-policy checkpoint. Omitting this uses an "
                        "untrained policy -- see the printed warning.")
    parser.add_argument("--rl-version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--measure-hd3-noise", action="store_true", default=False,
                        help="only has an effect with --backend real; adds one more real-SPICE evaluation.")
    parser.add_argument("--pvt-condition-set", choices=("none", "smoke", "minimal27", "full36"), default="none",
                        help="default none (nominal-only, cheap). smoke/minimal27/full36 are real-SPICE-only "
                        "and can be very slow -- never selected automatically.")
    parser.add_argument("--llm-provider", choices=("anthropic", "mock"), default=None,
                        help="overrides NEBULA_LLM_PROVIDER for this call.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--output", type=Path, default=None,
                        help="where to write the pipeline's raw JSON result (default: a new file "
                        "under results/llm_wrapper_runs/, never overwritten).")
    parser.add_argument("--json", action="store_true", help="print the raw pipeline JSON instead of the report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    provider, _requested_name = get_provider(provider_name=args.llm_provider)
    provider_result = provider.parse_target(args.request)

    output_path = args.output or (DEFAULT_RUNS_DIR / f"{uuid.uuid4().hex}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_argv = build_pipeline_argv(
        target=provider_result.parsed.fields, output_path=output_path, backend=args.backend,
        checkpoint=args.checkpoint, rl_version=args.rl_version, episodes=args.episodes,
        horizon=args.horizon, measure_hd3_noise=args.measure_hd3_noise,
        pvt_condition_set=args.pvt_condition_set,
    )

    start = time.perf_counter()
    try:
        process = run_pipeline_subprocess(pipeline_argv, timeout_s=args.timeout, cwd=PROJECT_ROOT)
    except subprocess.TimeoutExpired:
        print(f"error: pipeline did not finish within {args.timeout:.0f}s (backend={args.backend})", file=sys.stderr)
        return 1
    runtime_s = time.perf_counter() - start

    pipeline_result: dict | None = None
    if output_path.is_file():
        try:
            pipeline_result = load_pipeline_result(output_path)
        except (json.JSONDecodeError, OSError):
            pipeline_result = None

    report = summarize_result(
        request=args.request, provider_result=provider_result,
        unquantified_notes=provider_result.parsed.unquantified_notes,
        backend=args.backend, checkpoint=args.checkpoint, rl_version=args.rl_version, runtime_s=runtime_s,
        pipeline_argv=pipeline_argv, process=process, pipeline_result=pipeline_result,
    )

    if args.json:
        print(json.dumps(pipeline_result if pipeline_result is not None else {"error": "no pipeline output"}, indent=2))
    else:
        print(format_report(report))

    return 0 if process.returncode == 0 else process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
