"""Task 3: multiple feasible designs / trade-offs. Collects every genuinely
SPICE-verified, uniform-criterion-feasible design already discovered by
this project (zero new SPICE), deduplicates them, and ranks them by
REAL measured trade-offs (never a label unsupported by measured metrics).

Sources, each independently verified against results/*.jsonl already on
disk (see CANDIDATE_SOURCES below for exactly which file/rows and why):
  - Design A (Random Search, seed=123, candidate_index 8)
  - "Design B" and a second, better step from the same run
    (results/rl_reward_directed_smoke.jsonl)
  - 7 surviving candidates from the R/C/Cdeg counterfactual sweep around
    Design A's mixed-target-run cousin (each differs from that baseline on
    exactly one parameter -- see docs/autockt-mapping.md sec 17)

Every candidate here has REAL, logged (not fabricated) metrics: eye
height/width, margin, power. Trade-off labels are assigned by directly
comparing these measured values across the pool, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rl.target_spec import TargetSpec

COMPARISON_TARGET = TargetSpec.from_existing_thresholds()


@dataclass(frozen=True)
class FeasibleDesign:
    design_id: str
    source_file: str
    source_description: str
    parameters: dict[str, float]
    metrics: dict[str, float]
    native_reward: float
    native_reward_scale: str  # "autockt_reward" (max 10.0) or "reward_v1" (different scale/semantics) --
    # sources use different reward functions; native_reward is NOT comparable
    # across differing native_reward_scale values. See uniform_reward below
    # for a value that IS comparable across the whole catalog.

    def parameter_key(self, significant_digits: int = 12) -> tuple:
        """Unit-safe key that does not erase pico/femto-scale parameters.

        Decimal-place rounding made every capacitance below 1e-6 become zero.
        Significant-digit formatting preserves scale while still tolerating
        insignificant float serialization noise.
        """

        return tuple(
            (name, format(float(self.parameters[name]), f".{significant_digits}g"))
            for name in sorted(self.parameters)
        )

    @property
    def uniform_reward(self) -> float:
        """autockt_reward recomputed from this design's own real metrics
        against COMPARISON_TARGET -- comparable across every catalog entry
        regardless of which reward function originally scored it. Every
        catalog entry is already uniform-criterion feasible by construction,
        so this is always exactly TERMINAL_BONUS (10.0) for anything in the
        catalog -- it confirms shared-criterion feasibility, not a ranking
        signal (use trade_off_labels / raw metrics for that).
        """
        from rl.autockt_reward import autockt_reward
        return autockt_reward(self.metrics, COMPARISON_TARGET, success=True)


def _design_a() -> FeasibleDesign:
    return FeasibleDesign(
        design_id="design_a",
        source_file="results/receiver_random_search_20_seed123.jsonl",
        source_description="uniform Random Search, seed=123, candidate_index 8 -- "
        "the project's original, most-verified feasible design (also PVT-tested, "
        "23/27, see docs/autockt-mapping.md sec 21)",
        parameters={
            "rload_ohm": 2342.472156058411, "rdeg_ohm": 822.3558626926603,
            "cdeg_f": 9.999375862792168e-13, "itail_a": 0.0006028331705063624,
            "dfe_tap_v": -0.011090823885148815,
        },
        metrics={
            "dfe_locked_phase_eye_height_v": 1.5215334399802445,
            "dfe_eye_width_ui": 0.8699999999999999,
            "dfe_min_margin_v": 0.5174474651666459,
            "ctle_power_w": 0.0010850994,
        },
        native_reward=10.0, native_reward_scale="autockt_reward",
    )


def load_reward_directed_smoke_designs(
    path: str | Path = "results/rl_reward_directed_smoke.jsonl",
) -> list[FeasibleDesign]:
    """results/rl_reward_directed_smoke.jsonl -- experiments/train_rl.py's
    reward-directed sequential search, 2 rows, both success=true. Metrics
    are recovered from the logged `constraints` tuple (the same
    simulator.rl_adapter.constraints_from_evaluation shape used throughout
    this project: [success, 0.5-errors, margin, height-0.1, width-0.4,
    0.015-power]), not fabricated.
    """

    import json

    resolved = Path(path)
    if not resolved.is_file():
        return []
    designs = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("success"):
            continue
        c = row["constraints"]
        metrics = {
            "dfe_min_margin_v": c[2],
            "dfe_locked_phase_eye_height_v": c[3] + 0.1,
            "dfe_eye_width_ui": c[4] + 0.4,
            "ctle_power_w": 0.015 - c[5],
        }
        designs.append(FeasibleDesign(
            design_id=f"reward_directed_smoke_ep{row['episode']}_step{row['step']}",
            source_file=str(resolved),
            source_description=f"experiments/train_rl.py reward-directed sequential search, "
            f"episode {row['episode']} step {row['step']}",
            parameters=dict(row["parameters"]),
            metrics=metrics,
            native_reward=float(row["reward"]), native_reward_scale=row.get("reward_version", "reward_v1"),
        ))
    return designs


def load_rc_counterfactual_survivors(
    path: str | Path = "results/rc_counterfactual_sweep_mixed_target.jsonl",
) -> list[FeasibleDesign]:
    """results/rc_counterfactual_sweep_mixed_target.jsonl -- the 18-point
    one-factor-at-a-time sweep (docs/autockt-mapping.md sec 17), evaluated
    via direct real-SPICE evaluate_receiver calls. Only rows with
    spec_satisfied=true are genuinely feasible designs (7 of 18); each
    differs from that sweep's baseline design on exactly one parameter, so
    each is a distinct, independently-verified feasible point, not a
    duplicate of Design A.
    """

    import json

    resolved = Path(path)
    if not resolved.is_file():
        return []
    designs = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("spec_satisfied"):
            continue
        designs.append(FeasibleDesign(
            design_id=f"rc_sweep_{row['parameter']}_offset{row['index_offset']:+d}",
            source_file=str(resolved),
            source_description=f"R/C/Cdeg counterfactual sweep: {row['parameter']} "
            f"offset {row['index_offset']:+d} grid steps from that sweep's baseline design",
            parameters=dict(row["parameters"]),
            metrics=dict(row["metrics"]),
            native_reward=float(row["autockt_reward"]), native_reward_scale="autockt_reward",
        ))
    return designs


def load_all_known_feasible_designs() -> list[FeasibleDesign]:
    """The full, currently-known pool: Design A plus every other
    independently-verified feasible design already on disk. No new SPICE.
    """

    designs = [_design_a()]
    designs.extend(load_reward_directed_smoke_designs())
    designs.extend(load_rc_counterfactual_survivors())
    return designs


def deduplicate(designs: list[FeasibleDesign], significant_digits: int = 12) -> list[FeasibleDesign]:
    """Drops designs whose parameters agree to the requested significant digits,
    earlier design in the list, keeping the first occurrence.
    """

    seen: set[tuple] = set()
    unique: list[FeasibleDesign] = []
    for design in designs:
        key = design.parameter_key(significant_digits)
        if key in seen:
            continue
        seen.add(key)
        unique.append(design)
    return unique


@dataclass(frozen=True)
class RankedDesign:
    design: FeasibleDesign
    trade_off_labels: tuple[str, ...]


def rank_by_measured_trade_offs(designs: list[FeasibleDesign]) -> list[RankedDesign]:
    """Assigns trade-off labels by DIRECTLY comparing each design's measured
    metrics against the rest of the pool -- a label is only ever assigned
    because a design's OWN measured value is the best (or tied-best) in the
    pool on that axis, never asserted independently of the numbers.
    """

    if not designs:
        return []

    def best(metric: str, minimize: bool) -> Optional[float]:
        values = [d.metrics[metric] for d in designs if metric in d.metrics]
        if not values:
            return None
        return min(values) if minimize else max(values)

    best_power = best("ctle_power_w", minimize=True)
    best_height = best("dfe_locked_phase_eye_height_v", minimize=False)
    best_width = best("dfe_eye_width_ui", minimize=False)
    best_margin = best("dfe_min_margin_v", minimize=False)

    ranked = []
    for design in designs:
        labels = []
        m = design.metrics
        if best_power is not None and "ctle_power_w" in m and m["ctle_power_w"] == best_power:
            labels.append("lowest_power")
        if (best_height is not None and "dfe_locked_phase_eye_height_v" in m
                and m["dfe_locked_phase_eye_height_v"] == best_height):
            labels.append("strongest_eye_height")
        if best_width is not None and "dfe_eye_width_ui" in m and m["dfe_eye_width_ui"] == best_width:
            labels.append("widest_eye")
        if best_margin is not None and "dfe_min_margin_v" in m and m["dfe_min_margin_v"] == best_margin:
            labels.append("largest_margin")
        if not labels:
            labels.append("balanced")
        ranked.append(RankedDesign(design=design, trade_off_labels=tuple(labels)))
    return ranked


def build_catalog() -> list[RankedDesign]:
    designs = deduplicate(load_all_known_feasible_designs())
    return rank_by_measured_trade_offs(designs)


def _main() -> int:
    import json

    catalog = build_catalog()
    rows = []
    for entry in catalog:
        rows.append({
            "design_id": entry.design.design_id,
            "source_file": entry.design.source_file,
            "source_description": entry.design.source_description,
            "parameters": entry.design.parameters,
            "metrics": entry.design.metrics,
            "native_reward": entry.design.native_reward,
            "native_reward_scale": entry.design.native_reward_scale,
            "uniform_reward": entry.design.uniform_reward,
            "trade_off_labels": list(entry.trade_off_labels),
        })
    output_path = Path("results/feasible_design_catalog.jsonl")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {output_path}")
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps({"output": str(output_path), "n_designs": len(rows)}, indent=2))
    for row in rows:
        print(f"{row['design_id']:35s} {row['trade_off_labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
