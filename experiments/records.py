"""Incremental JSONL/CSV storage for simulation evaluations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from simulator.ctle import CTLEEvaluation


def flatten_evaluation(evaluation: CTLEEvaluation, run_id: int) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": run_id,
        "success": evaluation.success,
        "constraints_passed": evaluation.constraints_passed,
        "runtime_s": evaluation.runtime_s,
        "failure_stage": evaluation.failure_stage,
        "failure_code": evaluation.failure_code,
        "errors": " | ".join(evaluation.errors),
    }
    row.update(evaluation.parameters)
    row.update(evaluation.measurements)
    row.update(evaluation.metrics)
    return row


def append_jsonl(path: str | Path, row: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
