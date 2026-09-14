"""Measured nominal CTLE capacitance sweep; sampled coverage, not continuous tuning proof."""
import argparse
from dataclasses import replace, asdict
import json
from pathlib import Path
import numpy as np
from simulator.receiver import ReceiverParameters, EvaluationFidelity, evaluate_receiver
from simulator.cache import EvaluationCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    data = json.loads(args.input.read_text())
    candidate = data["selection"]["selected"]
    if not candidate:
        parser.error("no selected candidate")
    base = ReceiverParameters(**candidate["parameters"])
    rows = []
    report = {"status": "running", "scope": "nominal CTLE, variable CDEG only", "parameters": asdict(base),
              "target_frequencies_hz": [1.25e9, 1.875e9, 2.5e9], "tolerance_fraction": 0.05,
              "continuous_tunability_proven": False, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = EvaluationCache(args.cache)
    for capacitance in np.geomspace(max(10e-15, base.cdeg_f / 4), min(10e-12, base.cdeg_f * 4), 25):
        parameters = replace(base, cdeg_f=float(capacitance))
        result = evaluate_receiver(parameters, fidelity=EvaluationFidelity.SCREENING, cache=cache)
        rows.append({"capacitance_f": float(capacitance), "evaluation": result.to_dict()})
        args.output.write_text(json.dumps(report, indent=2, default=str))
    report["targets"] = []
    for frequency in report["target_frequencies_hz"]:
        usable = [r for r in rows if r["evaluation"]["success"] and r["evaluation"]["metrics"].get("global_peak_frequency_hz")]
        best = min(usable, key=lambda r: abs(r["evaluation"]["metrics"]["global_peak_frequency_hz"] / frequency - 1)) if usable else None
        measured = best["evaluation"]["metrics"]["global_peak_frequency_hz"] if best else None
        report["targets"].append({"target_hz": frequency, "measured_hz": measured,
            "capacitance_f": best["capacitance_f"] if best else None,
            "within_tolerance": measured is not None and abs(measured / frequency - 1) <= 0.05})
    report["status"] = "complete"
    report["limitation"] = "Screening-fidelity nominal CTLE sweep only; no noise, HD3, receiver-eye or PVT qualification of tuning settings. Exact endpoints and continuous coverage remain unproven."
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["targets"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
