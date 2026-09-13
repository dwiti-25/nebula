"""Append-only PPO event persistence and deterministic per-run aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from analysis.target_assessment import assess_target
from rl.target_spec import TargetSpec


class JsonlEventWriter:
    def __init__(self, path: str | Path, *, metadata: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)

    def __call__(self, event: dict[str, Any]) -> None:
        enriched = {**self.metadata, **event}
        if event.get("event_type") == "step":
            target = TargetSpec(**event["target_values"])
            assessment = assess_target(
                event.get("raw_metrics", {}), target,
                simulator_success=event.get("failure_stage") is None,
            )
            enriched["strict_target_assessment"] = assessment.to_dict()
            enriched["strict_pass"] = assessment.passed and (event.get("strict_pass") is not False)
            enriched.setdefault("cache_hit", None)
            enriched.setdefault("evaluation_timing_s", None)
            enriched.setdefault("stage_timings_s", None)
        else:
            enriched.setdefault("approx_kl", None)
            enriched.setdefault("clip_fraction", None)
            enriched.setdefault("explained_variance", None)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, default=list) + "\n")
            handle.flush()


def load_events(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_run_dashboard(path: str | Path) -> dict[str, Any]:
    events = load_events(path)
    steps = [event for event in events if event.get("event_type") == "step"]
    updates = [event for event in events if event.get("event_type") == "update"]
    evaluations = [event for event in events if event.get("event_type") == "evaluation"]
    charts: list[dict[str, Any]] = []
    if steps:
        label = str(steps[0].get("configuration_id", "ppo"))
        success_count = 0
        success_rates = []
        for index, event in enumerate(steps, 1):
            success_count += bool(event.get("strict_pass"))
            success_rates.append(success_count / index)
        charts.extend([
            {"id": "run_reward", "title": f"{label} reward progression", "type": "line",
             "x_label": "evaluation requests (reset/cache included)", "y_label": "reward",
             "series": [{"name": label, "x_values": [event.get("evaluation_count", index) for index, event in enumerate(steps, 1)], "values": [event["reward_total"] for event in steps]}]},
            {"id": "run_strict_success", "title": f"{label} strict success", "type": "line",
             "x_label": "policy transitions", "y_label": "cumulative strict-pass rate",
             "series": [{"name": label, "values": success_rates}]},
        ])
        failures: dict[str, int] = {}
        for event in steps:
            stage = str(event.get("failure_stage") or "target_not_met")
            failures[stage] = failures.get(stage, 0) + (not bool(event.get("strict_pass")))
        charts.append({"id": "run_failure_stage", "title": f"{label} failure stages", "type": "bar",
                       "x_label": "stage", "y_label": "count", "categories": list(failures),
                       "values": list(failures.values())})
    if updates:
        charts.append({"id": "run_ppo_health", "title": "PPO optimization health", "type": "line",
                       "x_label": "update", "y_label": "value",
                       "series": [{"name": key, "values": [row.get(key) for row in updates]}
                                  for key in ("policy_loss", "value_loss", "entropy")]})
    if evaluations:
        for key, units in (("ctle_power_w", "W"), ("peaking_db", "dB"),
                           ("dfe_locked_phase_eye_height_v", "V"), ("dfe_eye_width_ui", "UI"),
                           ("input_referred_noise_vrms", "Vrms"), ("hd3_db", "dB")):
            values = [row.get("raw_metrics", {}).get(key) for row in evaluations]
            if any(value is not None for value in values):
                charts.append({"id": f"run_{key}", "title": key, "type": "line",
                    "x_label": "evaluation requests (reset/PVT/cache included)", "y_label": units,
                    "note": "Missing measurements are gaps; synthetic backend values are not SPICE evidence.",
                    "series": [{"name": key, "values": values}]})
        charts.append({"id": "run_timing", "title": "Observed evaluation request time", "type": "line",
            "x_label": "evaluation requests", "y_label": "seconds",
            "series": [{"name": "elapsed_s", "values": [row.get("elapsed_s") for row in evaluations]}]})
    return {"schema_version": 1, "events": len(events), "charts": charts,
            "metadata": events[0] if events else {},
            "limitations": ["Unavailable historical or backend timing fields remain null; none are inferred."]}
