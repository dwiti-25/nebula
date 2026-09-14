"""Deterministic, read-only aggregation for the NEBULA evidence dashboard.

The UI consumes this module's small JSON chart schema. Evidence is read from
``results/`` when extracted, or directly from the tracked Phase-2 artifact ZIP
on a clean checkout. No simulator is run and missing evidence stays explicit.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Optional
from analysis.artifact_store import read_packaged_bytes


def _read_text(root: Path, relative: str) -> Optional[str]:
    direct = root / relative
    if direct.is_file():
        return direct.read_text(encoding="utf-8")
    packaged = read_packaged_bytes(root, relative)
    return packaged.decode("utf-8") if packaged is not None else None


def _jsonl(root: Path, relative: str) -> list[dict[str, Any]]:
    text = _read_text(root, relative)
    return [json.loads(line) for line in text.splitlines() if line.strip()] if text else []


def _json(root: Path, relative: str) -> Optional[dict[str, Any]]:
    text = _read_text(root, relative)
    return json.loads(text) if text else None


def build_dashboard(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    charts: list[dict[str, Any]] = []
    limitations: list[str] = []
    sources: list[str] = []

    training_path = "results/autockt_mixed_target_confirmation.jsonl"
    training = [r for r in _jsonl(root, training_path) if isinstance(r.get("reward"), (int, float))]
    if training:
        sources.append(training_path)
        rewards = [float(r["reward"]) for r in training]
        running_success: list[float] = []
        success_count = 0
        for index, row in enumerate(training, 1):
            success_count += bool(row.get("spec_satisfied"))
            running_success.append(success_count / index)
        charts.extend([
            {
                "id": "ppo_reward", "title": "PPO reward by real evaluation", "type": "line",
                "x_label": "evaluation", "y_label": "reward",
                "series": [{"name": "reward", "values": rewards}],
            },
            {
                "id": "ppo_success", "title": "PPO cumulative target satisfaction", "type": "line",
                "x_label": "evaluation", "y_label": "success rate",
                "series": [{"name": "strict logged success rate", "values": running_success}],
            },
        ])
        failures = Counter(str(r.get("failure_stage") or "target_not_met") for r in training
                           if not r.get("spec_satisfied"))
        charts.append({
            "id": "failure_modes", "title": "PPO failure-stage distribution", "type": "bar",
            "x_label": "stage", "y_label": "count",
            "categories": sorted(failures), "values": [failures[k] for k in sorted(failures)],
        })
        limitations.append(
            "Historical PPO rows log reward and target satisfaction but not raw metrics; they cannot be "
            "retrospectively re-scored against a different target."
        )

    benchmark_path = "results/benchmark_report.json"
    benchmark = _json(root, benchmark_path)
    if benchmark:
        sources.append(benchmark_path)
        trial = benchmark.get("trials", {}).get("no_warm_start_headtohead_sec20", {})
        methods = trial.get("methods", {})
        categories: list[str] = []
        values: list[float] = []
        for method in ("random_search", "cem", "ppo"):
            row = methods.get(method, {})
            value = row.get("reported_strict_success_rate", row.get("uniform_success_rate"))
            if value is None and method == "ppo":  # schema-v1 archived report
                value = row.get("native_success_rate")
            if value is not None:
                categories.append(method)
                values.append(float(value))
        charts.append({
            "id": "optimizer_comparison", "title": "Budget-matched optimizer success", "type": "bar",
            "x_label": "optimizer", "y_label": "success rate", "categories": categories, "values": values,
            "note": "n=20, one seed. PPO uses its logged strict trivial-target result because raw metrics were not logged.",
        })
        limitations.append(str(trial.get("comparability_caveat", "")))

    generalization_path = "results/controlled_unseen_target_generalization.jsonl"
    summaries = [r for r in _jsonl(root, generalization_path) if r.get("row_type") == "summary"]
    if summaries:
        sources.append(generalization_path)
        summary = summaries[-1]
        charts.append({
            "id": "unseen_target", "title": "Unseen-target checkpoint comparison", "type": "bar",
            "x_label": "checkpoint", "y_label": "satisfaction rate",
            "categories": ["initial", "trained"],
            "values": [summary.get("initial_satisfaction_rate"), summary.get("final_satisfaction_rate")],
            "note": f"{summary.get('matched_episodes', 0)} paired episodes; one target and one evaluation seed.",
        })

    catalog_path = "results/feasible_design_catalog.jsonl"
    catalog = _jsonl(root, catalog_path)
    if catalog:
        sources.append(catalog_path)
        points = []
        for row in catalog:
            metrics = row.get("metrics", {})
            if "ctle_power_w" in metrics and "dfe_locked_phase_eye_height_v" in metrics:
                points.append({
                    "label": row.get("design_id", "design"),
                    "x": float(metrics["ctle_power_w"]) * 1000.0,
                    "y": float(metrics["dfe_locked_phase_eye_height_v"]),
                })
        charts.append({
            "id": "pareto", "title": "Feasible-design eye/power trade-off", "type": "scatter",
            "x_label": "CTLE power (mW)", "y_label": "eye height (V)", "points": points,
        })

    pvt_path = "results/design_a_pvt_minimal27_rerun.jsonl"
    pvt = _jsonl(root, pvt_path)
    if pvt:
        sources.append(pvt_path)
        process_order = ["ss", "tt", "ff"]
        conditions = sorted({(float(r["supply_v"]), float(r["temperature_c"])) for r in pvt})
        cells = []
        for row in pvt:
            cells.append({
                "process": row["process_corner"], "supply_v": float(row["supply_v"]),
                "temperature_c": float(row["temperature_c"]), "passed": bool(row.get("success")),
            })
        charts.append({
            "id": "pvt", "title": "Design A full PVT matrix", "type": "heatmap",
            "rows": process_order, "columns": [f"{v:g}V / {t:g}C" for v, t in conditions],
            "conditions": conditions, "cells": cells,
            "note": "Green means the archived simulator evaluation succeeded; strict target scoring is shown for new runs.",
        })

    return {
        "schema_version": 1,
        "charts": charts,
        "sources": list(dict.fromkeys(sources)),
        "limitations": [x for x in dict.fromkeys(limitations) if x],
        "summary": {
            "chart_count": len(charts),
            "ppo_evaluations": len(training),
            "ppo_logged_successes": sum(bool(r.get("spec_satisfied")) for r in training),
            "pvt_passes": sum(bool(r.get("success")) for r in pvt),
            "pvt_total": len(pvt),
            "feasible_designs": len(catalog),
        },
    }
