"""Task 2 (overnight chunk, sec 22): the single authoritative final-design
specification report. For a selected design, collects every official-brief
metric that is ACTUALLY measured somewhere in this project's real,
already-computed results -- never inventing a missing measurement.

Each row: metric, measured value, required threshold, PASS/FAIL/NOT CLAIMED,
and the exact source file/condition the number came from -- so every number
in the report is traceable back to a real SPICE evaluation, not asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from analysis.area_estimate import estimate_ctle_area
from analysis.pvt_selection import PVTRobustnessResult, load_pvt_results_from_jsonl
from analysis.target_assessment import assess_target
from rl.target_spec import TargetSpec


@dataclass(frozen=True)
class SpecRow:
    metric: str
    measured: Optional[str]
    requirement: str
    verdict: str  # "PASS", "FAIL", or "NOT CLAIMED"
    source: str


def _gate(value: Optional[float], *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> str:
    if value is None:
        return "NOT CLAIMED"
    if minimum is not None and value < minimum:
        return "FAIL"
    if maximum is not None and value > maximum:
        return "FAIL"
    return "PASS"


def build_final_specification_report(
    *,
    design_id: str,
    parameters: dict[str, float],
    nominal_metrics: dict[str, float],
    nominal_source: str,
    pvt_result: Optional[PVTRobustnessResult] = None,
    target: Optional[TargetSpec] = None,
) -> dict:
    """Pure function over already-known data -- no simulation. Callers
    supply the design's real parameters, its real nominal-condition metrics
    (with a source string identifying exactly where they came from), and
    optionally an already-computed PVTRobustnessResult (see
    analysis.pvt_selection -- never computed automatically here).
    """

    area = estimate_ctle_area()
    effective_target = target or TargetSpec.from_existing_thresholds()
    target_assessment = assess_target(nominal_metrics, effective_target, simulator_success=True)

    def metric(name: str) -> Optional[float]:
        return nominal_metrics.get(name)

    rows = [
        SpecRow(
            "Eye width (UI)", f"{metric('dfe_eye_width_ui'):.4g}" if metric("dfe_eye_width_ui") is not None else None,
            "> 0.4 UI", _gate(metric("dfe_eye_width_ui"), minimum=0.4), nominal_source,
        ),
        SpecRow(
            "Eye height (V)",
            f"{metric('dfe_locked_phase_eye_height_v'):.4g}" if metric("dfe_locked_phase_eye_height_v") is not None else None,
            "> 0.1 V (this repo's own EXISTING_THRESHOLDS value -- the official "
            "slide's exact mV figure was not independently re-verified against "
            "this number in this pass; see docs/autockt-mapping.md sec 21)",
            _gate(metric("dfe_locked_phase_eye_height_v"), minimum=0.1), nominal_source,
        ),
        SpecRow(
            "Margin (V)", f"{metric('dfe_min_margin_v'):.4g}" if metric("dfe_min_margin_v") is not None else None,
            "> 0 V", _gate(metric("dfe_min_margin_v"), minimum=0.0), nominal_source,
        ),
        SpecRow(
            "Power (W)", f"{metric('ctle_power_w'):.4g}" if metric("ctle_power_w") is not None else None,
            "< 0.015 W (15 mW)", _gate(metric("ctle_power_w"), maximum=0.015), nominal_source,
        ),
        SpecRow(
            "Peaking (dB)", f"{metric('peaking_db'):.4g}" if metric("peaking_db") is not None else None,
            "3-12 dB, ~1.25-2.5 GHz",
            _gate(metric("peaking_db"), minimum=3.0, maximum=12.0), nominal_source,
        ),
        SpecRow(
            "HD3 (dB)", f"{metric('hd3_db'):.4g}" if metric("hd3_db") is not None else None,
            "< -30 dB", _gate(metric("hd3_db"), maximum=-30.0), nominal_source,
        ),
        SpecRow(
            "Input-referred noise (Vrms)",
            f"{metric('input_referred_noise_vrms'):.4g}" if metric("input_referred_noise_vrms") is not None else None,
            "< 0.0015 Vrms (1.5 mV)",
            _gate(metric("input_referred_noise_vrms"), maximum=0.0015), nominal_source,
        ),
        SpecRow(
            "Transistor channel area (mm^2)",
            f"{area.transistor_channel_area_mm2:.4g}",
            "< 0.05 mm^2 -- PARTIAL measurement only (channel area, not total "
            "circuit area; resistor/capacitor/layout-overhead area are not "
            "modeled anywhere in this project -- see analysis/area_estimate.py)",
            "NOT CLAIMED", str(area.source_file),
        ),
    ]

    if pvt_result is not None:
        worst_case = [
            f"{p.process_corner}/{p.supply_v}V/{p.temperature_c}C" for p in pvt_result.worst_case_conditions
        ]
        pvt_verdict = "PASS" if pvt_result.pass_rate >= 1.0 else "PARTIAL"
        rows.append(SpecRow(
            "PVT (pass/total)", f"{pvt_result.n_passing}/{pvt_result.n_conditions}",
            "TT/SS/FF x VDD+/-5% x 0-125C", pvt_verdict,
            f"{pvt_result.n_conditions}-point sweep, design_id={pvt_result.design_id}"
            + (f", worst case: {', '.join(worst_case)}" if worst_case else ""),
        ))
    else:
        rows.append(SpecRow("PVT (pass/total)", None, "TT/SS/FF x VDD+/-5% x 0-125C", "NOT CLAIMED", "no PVT result supplied"))

    return {
        "design_id": design_id,
        "parameters": parameters,
        "requested_target": effective_target.as_dict(),
        "target_qualification": target_assessment.to_dict(),
        "rows": [
            {"metric": r.metric, "measured": r.measured, "requirement": r.requirement,
             "verdict": r.verdict, "source": r.source}
            for r in rows
        ],
    }


def format_report(report: dict) -> str:
    lines = [f"FINAL DESIGN: {report['design_id']}", "-" * 40]
    for name, value in report["parameters"].items():
        lines.append(f"{name:15s} = {value:.6g}")
    lines.append("")
    lines.append(f"{'Metric':30s} {'Measured':15s} {'Requirement':45s} {'Verdict'}")
    for row in report["rows"]:
        measured = row["measured"] if row["measured"] is not None else "n/a"
        lines.append(f"{row['metric']:30s} {measured:15s} {row['requirement']:45s} {row['verdict']}")
    return "\n".join(lines)


def _main() -> int:
    # Design A -- the only design in the catalog with a complete, real,
    # FINAL-fidelity PVT characterization (see docs/autockt-mapping.md
    # sec 21/22). Nominal metrics come from the exact TT/1.8V/27C point of
    # the 27/27 reproducibility re-run (results/design_a_pvt_minimal27_rerun.jsonl),
    # the current, most-verified record -- not the original 23/27 run.
    parameters = {
        "rload_ohm": 2342.472156058411, "rdeg_ohm": 822.3558626926603,
        "cdeg_f": 9.999375862792168e-13, "itail_a": 0.0006028331705063624,
        "dfe_tap_v": -0.011090823885148815,
    }
    rerun_path = Path("results/design_a_pvt_minimal27_rerun.jsonl")
    nominal_metrics = {}
    nominal_source = "results/design_a_pvt_minimal27_rerun.jsonl (no nominal point found)"
    if rerun_path.is_file():
        rows = [json.loads(line) for line in rerun_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        nominal_row = next(
            (r for r in rows if r["process_corner"] == "tt" and r["supply_v"] == 1.8 and r["temperature_c"] == 27.0),
            None,
        )
        if nominal_row:
            nominal_metrics = nominal_row["metrics"]
            nominal_source = f"{rerun_path} (tt/1.8V/27C point)"

    target = TargetSpec.from_existing_thresholds()
    pvt_result = (
        load_pvt_results_from_jsonl("design_a", rerun_path, target=target)
        if rerun_path.is_file() else None
    )

    report = build_final_specification_report(
        design_id="design_a", parameters=parameters, nominal_metrics=nominal_metrics,
        nominal_source=nominal_source, pvt_result=pvt_result, target=target,
    )
    output_path = Path("results/design_a_final_specification.json")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {output_path}")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_report(report))
    print(f"\nWritten: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
