"""Simple random-search baseline used before any RL implementation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.sweep import random_parameters, run_sweep


def score_row(row: dict[str, object], target_peaking_db: float = 6.0) -> float:
    if not row.get("success"):
        return -math.inf
    peaking = float(row["peaking_2p5ghz_db"])
    power = float(row["power_w"])
    return -abs(peaking - target_peaking_db) - 1000.0 * power


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run the baseline CTLE random search")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/random_search.jsonl"))
    args = parser.parse_args()
    rows = run_sweep(random_parameters(args.count, args.seed), jsonl=args.output)
    ranked = sorted(rows, key=score_row, reverse=True)
    best = ranked[0] if ranked and math.isfinite(score_row(ranked[0])) else None
    print(json.dumps({"evaluations": len(rows), "best": best}, indent=2))
    return 0 if best else 1


if __name__ == "__main__":
    raise SystemExit(_main())
