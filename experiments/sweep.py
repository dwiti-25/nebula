"""Grid and random sweep entry points for the CTLE evaluator."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import random
from typing import Iterable, Mapping

from experiments.records import append_jsonl, flatten_evaluation, write_csv
from simulator import Sky130Config, evaluate_ctle
from simulator.models import ParameterValue


DEFAULT_GRID = {
    "RLOAD": ("750", "1k", "1.25k"),
    "RDEG": ("500", "750", "1k"),
    "CDEG": ("0.25p", "0.5p", "0.75p"),
    "ITAIL_VAL": ("75u", "100u", "125u"),
}


def grid_parameters(grid: Mapping[str, Iterable[ParameterValue]]) -> Iterable[dict[str, ParameterValue]]:
    names = tuple(grid)
    for values in product(*(grid[name] for name in names)):
        yield dict(zip(names, values))


def random_parameters(count: int, seed: int = 0) -> Iterable[dict[str, float]]:
    generator = random.Random(seed)
    for _ in range(count):
        yield {
            "RLOAD": generator.uniform(500, 2000),
            "RDEG": generator.uniform(250, 2000),
            "CDEG": generator.uniform(0.1e-12, 1.5e-12),
            "ITAIL_VAL": generator.uniform(50e-6, 250e-6),
        }


def run_sweep(
    candidates: Iterable[Mapping[str, ParameterValue]],
    *,
    jsonl: str | Path,
    csv_path: str | Path | None = None,
    sky130: Sky130Config | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_id, parameters in enumerate(candidates, start=1):
        row = flatten_evaluation(evaluate_ctle(parameters, sky130=sky130), run_id)
        append_jsonl(jsonl, row)
        rows.append(row)
    if csv_path:
        write_csv(csv_path, rows)
    return rows


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run a reproducible SKY130 CTLE sweep")
    parser.add_argument("--mode", choices=("grid", "random"), default="grid")
    parser.add_argument("--count", type=int, default=20, help="random-search sample count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jsonl", type=Path, default=Path("results/ctle_sweep.jsonl"))
    parser.add_argument("--csv", type=Path, default=Path("results/ctle_sweep.csv"))
    args = parser.parse_args()
    candidates = grid_parameters(DEFAULT_GRID) if args.mode == "grid" else random_parameters(args.count, args.seed)
    rows = run_sweep(candidates, jsonl=args.jsonl, csv_path=args.csv)
    successes = sum(bool(row["success"]) for row in rows)
    print(json.dumps({"evaluations": len(rows), "successes": successes}, indent=2))
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(_main())
