"""Current v3 evidence only; never silently substitutes legacy benchmark data."""
import json
from pathlib import Path
from analysis.run_evidence import build_run_dashboard, load_events


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build_dashboard(root="."):
    root = Path(root)
    sources, charts = [], []
    summary = {"ppo_evaluations": 0, "ppo_logged_successes": 0,
               "pvt_passes": 0, "pvt_total": 0, "feasible_designs": 0}
    notes = ["v3 only. Training history is saved evidence, not newly retrained weights.",
             "Required PVT scope: 36 TT/SS/FF conditions; no robustness claim without completed validation."]
    candidates = [root / "results/v3_tt_1000_summary.json", *list((root / "results/web_ui_runs").glob("*.json"))]
    candidates = sorted((p for p in candidates if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        data = read_json(path)
        if data.get("workflow") != "train" or data.get("rl_version") != "v3" or not data.get("run_complete"):
            continue
        event_path = Path(str(data.get("events_output", "")).replace("\\", "/"))
        if not event_path.is_absolute():
            event_path = root / event_path
        if not event_path.is_file():
            continue
        events = load_events(event_path)
        charts.extend(build_run_dashboard(event_path)["charts"])
        sources.extend([str(path), str(event_path)])
        evaluations = [e for e in events if e.get("event_type") == "evaluation"]
        steps = [e for e in events if e.get("event_type") == "step"]
        summary.update(ppo_evaluations=len(evaluations), ppo_logged_successes=sum(bool(e.get("strict_pass")) for e in steps))
        points = [{"x": e["raw_metrics"]["ctle_power_w"] * 1000,
                  "y": e["raw_metrics"]["dfe_locked_phase_eye_height_v"], "label": ""}
                  for e in evaluations if e.get("failure_stage") is None and
                  all(k in e.get("raw_metrics", {}) for k in ("ctle_power_w", "dfe_locked_phase_eye_height_v"))]
        charts.append({"id": "v3_eye_power", "title": "v3 measured eye / power (saved training)", "type": "scatter",
                       "x_label": "power (mW)", "y_label": "eye height (V)", "points": points,
                       "note": f"Backend: {data.get('backend')}; nominal measurements, not FINAL qualification."})
        summary["feasible_designs"] = len(points)
        summary["checkpoint"] = data.get("policy_path")
        break
    comparison_path = root / "results/v3_ui_comparison_20260912.json"
    comparison = read_json(comparison_path)
    if comparison and str(summary.get("checkpoint", "")).replace("\\", "/") == str(comparison.get("checkpoint", "")).replace("\\", "/"):
        sources.append(str(comparison_path))
        runs = comparison.get("runs", [])
        methods = list(dict.fromkeys(run["method"] for run in runs))
        complete = len(runs) == 8 and all(run["complete"] for run in runs)
        notes.extend(comparison["limitations"])
        for key, title, unit in (("success_rate", "Nominal strict-pass rate", "fraction"),
                                 ("elapsed_s", "Evaluation batch runtime", "seconds"),
                                 ("evaluations", "Evaluation budget consumed", "requests")):
            values = [sum(r[key] for r in runs if r["method"] == m) / sum(r["method"] == m for r in runs) for m in methods]
            charts.append({"id": "v3_compare_" + key, "title": "v3 comparison — " + title, "type": "bar",
                           "categories": methods, "values": values, "x_label": "method", "y_label": unit,
                           "note": f"{'Complete' if complete else 'PARTIAL — not a completed comparison'}; two baseline seeds, deterministic PPO repeated; TRAINING fidelity."})
    else:
        notes.append("No matching new optimizer comparison for the displayed checkpoint.")
    inference_path = root / "results/v3_ui_inference_20260912.json"
    inference = read_json(inference_path)
    if inference and str(inference.get("checkpoint_path", "")).replace("\\", "/") == str(summary.get("checkpoint", "")).replace("\\", "/"):
        sources.append(str(inference_path))
        current = build_run_dashboard(inference_path.with_suffix(".events.jsonl"))
        for chart in current["charts"]:
            chart["id"] = "fresh_inference_" + chart["id"]
            chart["title"] = "Fresh v3 inference — " + chart["title"]
            charts.append(chart)
        notes.append(f"Fresh verified-start inference: {inference.get('n_nominally_feasible', 0)}/{inference.get('n_candidates_generated', 0)} nominally feasible episodes; not comparable to grid-center benchmark.")
    qualification_path = root / "results/v3_qualification_final_20260913.json"
    qualification = read_json(qualification_path)
    if qualification and str(qualification.get("source_checkpoint", "")).replace("\\", "/") == str(summary.get("checkpoint", "")).replace("\\", "/"):
        sources.append(str(qualification_path))
        summary["qualification_status"] = qualification["status"]
        summary["qualified_designs"] = qualification.get("qualified_designs", [])
        for index, candidate in enumerate(qualification.get("candidates", [])):
            points = candidate.get("pvt", [])
            if index == 0:
                summary["pvt_total"] = len(points)
                summary["pvt_passes"] = sum(bool(p["passed"]) for p in points)
            conditions = [(v, t) for v in (1.71, 1.8, 1.89) for t in (0, 27, 75, 125)]
            charts.append({"id": f"v3_pvt_{index}", "title": f"{candidate['design_id']} — TT/SS/FF qualification", "type": "heatmap",
                "rows": ["tt", "ss", "ff"], "conditions": conditions,
                "columns": [f"{v}V/{t}C" for v, t in conditions],
                "cells": [{"process": p["condition"]["process_corner"], "supply_v": p["condition"]["supply_v"],
                           "temperature_c": p["condition"]["temperature_c"], "passed": p["passed"]} for p in points],
                "note": f"{candidate['status']}; {len(points)}/36 conditions attempted at 512 bits. Nominal long test: 1024 bits. Gray = unmeasured; green = pass; red = fail."})
            notes.append(f"{candidate['design_id']}: {candidate['status']}; nominal stages: " +
                         ", ".join("pass" if n["passed"] else "fail" for n in candidate.get("nominal", [])))
    tuning_path = root / "results/v3_tuning_20260913.json"
    tuning = read_json(tuning_path)
    if tuning and tuning.get("parameters") == (inference.get("selection", {}).get("selected") or {}).get("parameters"):
        sources.append(str(tuning_path))
        rows = [r for r in tuning.get("rows", []) if "global_peak_frequency_hz" in r["evaluation"]["metrics"]]
        charts.append({"id": "v3_tuning", "title": "Nominal CDEG tuning sweep", "type": "line", "x_label": "CDEG (pF)", "y_label": "global peak (GHz)",
            "series": [{"name": "measured global peak", "x_values": [r["capacitance_f"] * 1e12 for r in rows],
                        "values": [r["evaluation"]["metrics"]["global_peak_frequency_hz"] / 1e9 for r in rows]}],
            "note": "Includes failing screening settings for diagnosis; sampled AC behavior only, not continuous or full-receiver qualification."})
    stronger_path = root / "results/v3_midpoint_comparison_20260913.json"
    stronger = read_json(stronger_path)
    if stronger and stronger.get("runs") and str(stronger.get("checkpoint", "")).replace("\\", "/") == str(summary.get("checkpoint", "")).replace("\\", "/"):
        sources.append(str(stronger_path))
        runs = stronger["runs"]
        methods = list(dict.fromkeys(r["method"] for r in runs))
        charts.append({"id": "v3_midpoint_comparison", "title": "Midpoint-target comparison — shared seeded starts", "type": "bar",
            "categories": methods, "values": [sum(r["success_rate"] for r in runs if r["method"] == m) / sum(r["method"] == m for r in runs) for m in methods],
            "x_label": "method", "y_label": "nominal strict-pass fraction",
            "note": f"{len(runs)}/12 method/seed batches recorded; 24 requests each; no training cost included. " +
                    ("Complete." if len(runs) == 12 and all(r['complete'] for r in runs) else "PARTIAL.")})
    for fidelity in ("screening", "training"):
        path = root / f"results/runtime_parallel_{fidelity}.json"
        data = read_json(path)
        if data:
            sources.append(str(path))
            charts.append({"id": "v3_runtime_" + fidelity, "title": f"Simulator throughput — {fidelity}", "type": "bar",
                "categories": [str(r["workers"]) + " workers" for r in data["runs"]],
                "values": [r["seconds"] for r in data["runs"]], "x_label": "workers", "y_label": "seconds",
                "note": "Fixed-design simulator benchmark, not policy accuracy. " + data["limitation"]})
    summary["chart_count"] = len(charts)
    return {"schema_version": 2, "charts": charts, "sources": sources, "limitations": notes, "summary": summary}
