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
            enriched["strict_pass"] = assessment.passed
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
    charts: list[dict[str, Any]] = []
    if steps:
        label = str(steps[0].get("configuration_id", "ppo"))
        charts.extend([
            {"id": "run_reward", "title": f"{label} reward progression", "type": "line",
             "x_label": "cumulative real evaluations", "y_label": "reward",
             "series": [{"name": label, "values": [event["reward_total"] for event in steps]}]},
            {"id": "run_strict_success", "title": f"{label} strict success", "type": "line",
             "x_label": "cumulative real evaluations", "y_label": "cumulative strict-pass rate",
             "series": [{"name": label, "values": [
                 sum(bool(row.get("strict_pass")) for row in steps[:index]) / index
                 for index in range(1, len(steps) + 1)
             ]}]},
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
                       "series": [{"name": key, "values": [row.get(key, 0) for row in updates]}
                                  for key in ("policy_loss", "value_loss", "entropy")]})
    return {"schema_version": 1, "events": len(events), "charts": charts,
            "metadata": events[0] if events else {},
            "limitations": ["Unavailable historical or backend timing fields remain null; none are inferred."]}

