"""Produce a submission-facing dossier from recorded qualification evidence."""
import argparse
import json
from pathlib import Path


def build_report(root):
    root = Path(root)
    def read(name):
        path = root / "results" / name
        return json.loads(path.read_text()) if path.is_file() else {}
    qualification = read("v3_qualification_final_20260913.json")
    tuning = read("v3_tuning_20260913.json")
    comparison = read("v3_midpoint_comparison_20260913.json")
    lines = ["# v3 completion evidence — TT/SS/FF scope", "",
        "Required grid: TT/SS/FF × 1.71/1.80/1.89 V × 0/27/75/125 °C (36 conditions).",
        "Physical DFE is excluded. The CTLE remains transistor-level with an ideal tail source; DFE is behavioral.", "",
        "## Finalist qualification", "", f"Sequence status: **{qualification.get('status', 'not run')}**.", "",
        "| Design | Nominal 512 | Nominal 1024 | Corner passes / attempted | Status |",
        "|---|---|---|---|---|"]
    for candidate in qualification.get("candidates", []):
        nominal = candidate.get("nominal", [])
        label = lambda i: ("PASS" if nominal[i]["passed"] else "FAIL / incomplete") if len(nominal) > i else "not run"
        points = candidate.get("pvt", [])
        lines.append(f"| {candidate['design_id']} | {label(0)} | {label(1)} | {sum(p['passed'] for p in points)}/{len(points)} of 36 | {candidate['status']} |")
    for candidate in qualification.get("candidates", []):
        nominal = candidate.get("nominal", [])
        if nominal:
            metrics = nominal[-1]["evaluation"]["metrics"]
            lines.extend(["", f"{candidate['design_id']} nominal metrics:", ""])
            for key in ("hd3_db", "input_referred_noise_vrms", "dfe_locked_phase_eye_height_v", "dfe_eye_width_ui", "dfe_error_count", "simulated_bits"):
                lines.append(f"- {key}: {metrics.get(key, 'not measured')}")
            lines.append("")
        points = candidate.get("pvt", [])
        if points:
            lines.extend([f"### {candidate['design_id']} measured corner ranges", "",
                          "| Metric | Minimum | Maximum |", "|---|---|---|"])
            for key in ("hd3_db", "input_referred_noise_vrms", "dfe_locked_phase_eye_height_v", "dfe_eye_width_ui", "ctle_power_w", "dfe_error_count"):
                values = [p["evaluation"]["metrics"][key] for p in points if key in p["evaluation"]["metrics"]]
                if values:
                    lines.append(f"| {key} | {min(values):.6g} | {max(values):.6g} |")
            lines.append("")
    lines.extend(["", "Corners use 512-bit CANDIDATE fidelity including noise and 100mVpp HD3. Nominal FINAL uses 1024 bits and three HD3 amplitudes. Zero observed errors is not a BER guarantee.",
        "", "## Tuning", "", "| Target GHz | Measured GHz | Within ±5% |", "|---|---|---|"])
    for target in tuning.get("targets", []):
        value = target.get("measured_hz")
        lines.append(f"| {target['target_hz']/1e9:g} | {value/1e9 if value else 'unavailable'} | {target['within_tolerance']} |")
    lines.extend(["", "Sampled nominal AC sweep only. Continuous coverage and receiver/noise/HD3 compliance at tuning settings remain unproven.",
        "", "## Frozen-policy comparison", "", "| Method | Requests | Strict passes | Complete batches |", "|---|---|---|---|"])
    runs = comparison.get("runs", [])
    for method in dict.fromkeys(r["method"] for r in runs):
        subset = [r for r in runs if r["method"] == method]
        lines.append(f"| {method} | {sum(r['evaluations'] for r in subset)} | {sum(row['success'] for r in subset for row in r['rows'])} | {sum(r['complete'] for r in subset)}/3 |")
    lines.extend(["", "Three shared random starting points, midpoint target, 24 requests per method/seed, horizon 12. This evaluates one frozen trained policy, not three independently trained models. Pretraining cost excluded.",
        "", "## Outstanding submission limits", "",
        "- Eye-height threshold is the repository's 0.1 V convention; the original statement still needs confirmation.",
        "- Total receiver area and power are unproven: MOS channel area and CTLE power are partial accounting.",
        "- Synthetic regression channel is used here, not PCIe compliance evidence.",
        "- No jitter/mismatch sign-off, continuous tuning proof or guaranteed BER.",
        "- Failed or unfinished corner checks never qualify a finalist.",
        "- No model retraining was performed in this sequence.", "",
        "## Source artifacts", "",
        "- results/v3_qualification_final_20260913.json",
        "- results/v3_nominal_final_20260913.jsonl",
        "- results/v3_tuning_20260913.json",
        "- results/v3_midpoint_comparison_20260913.json",
        "- results/v3_qualification_20260913.json: interrupted 1024-bit-per-corner attempt; timeouts preserved, not treated as electrical failures.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-directory", type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(build_report(Path.cwd()))
    if args.export_directory:
        from experiments.export_final_schematic import render_final_schematic, BLOCK
        from simulator.receiver import ReceiverParameters
        report = json.loads(Path("results/v3_qualification_final_20260913.json").read_text())
        args.export_directory.mkdir(parents=True, exist_ok=True)
        # Only qualified designs are exported; rejected alternatives remain
        # evidence records, never advertised as accepted schematics.
        for candidate in report["candidates"]:
            if candidate["status"] != "qualified":
                continue
            ident = candidate["design_id"]
            text = render_final_schematic(ReceiverParameters(**candidate["parameters"]),
                achieved_metrics=candidate["nominal"][-1]["evaluation"]["metrics"],
                target_description=str(report["target"]),
                source_description="36/36 TT/SS/FF at 512 bits; nominal 1024 bits; behavioral DFE. See qualification JSON.")
            text = text.replace(BLOCK.as_posix(), "ctle.spice")
            with (args.export_directory / f"{ident}.cir").open("x", encoding="utf-8") as stream:
                stream.write(text)
            with (args.export_directory / f"{ident}.json").open("x", encoding="utf-8") as stream:
                json.dump(candidate, stream, indent=2)
        with (args.export_directory / "ctle.spice").open("x", encoding="utf-8") as stream:
            stream.write(BLOCK.read_text())


if __name__ == "__main__":
    main()
