"""Task 4: PVT-aware candidate selection -- turns the PVT work (sec 21,
Task 1) into actual framework functionality rather than a one-off
experiment.

Conceptual flow implemented here:

    candidate designs (analysis.design_catalog)
          |
    nominal feasibility screening (already done -- catalog entries are
          |  already uniform-criterion feasible by construction)
          v
    PVT evaluation (simulator.receiver.evaluate_pvt_grid, reused unmodified;
          |          only run for candidates that reach this stage, and only
          |          against an explicitly given, small condition set --
          |          never automatically against the full 60-point grid for
          |          every candidate)
          v
    robustness score (pass rate, worst-case conditions)
          v
    rank candidates by robustness
          v
    select final design(s)

Does not call evaluate_pvt_grid at import time or on any candidate by
default -- callers explicitly choose which candidates and which condition
set to spend real SPICE on. Can also summarize ALREADY-COMPUTED PVT results
(e.g. results/design_a_pvt_minimal27.jsonl) without running anything new,
which is how this module is demonstrated below with zero additional SPICE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from simulator.config import SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverParameters, evaluate_pvt_grid

from analysis.design_catalog import FeasibleDesign, rank_by_measured_trade_offs
from analysis.target_assessment import TargetAssessment, assess_target
from rl.target_spec import TargetSpec


@dataclass(frozen=True)
class PVTPointResult:
    process_corner: str
    supply_v: float
    temperature_c: float
    success: bool
    failed_stage: Optional[str]
    simulator_success: Optional[bool] = None
    metrics: Optional[dict[str, float]] = None
    target_assessment: Optional[TargetAssessment] = None


@dataclass(frozen=True)
class PVTRobustnessResult:
    design_id: str
    n_conditions: int
    n_passing: int
    pass_rate: float
    worst_case_conditions: tuple[PVTPointResult, ...]  # the failing points, if any
    points: tuple[PVTPointResult, ...]


def summarize_pvt_results(design_id: str, points: list[PVTPointResult]) -> PVTRobustnessResult:
    """Pure aggregation -- no simulation. Turns a list of per-condition
    pass/fail results (from a real evaluate_pvt_grid run, or loaded from an
    existing results/*.jsonl file) into a robustness score.
    """

    n = len(points)
    passing = [p for p in points if p.success]
    failing = tuple(p for p in points if not p.success)
    return PVTRobustnessResult(
        design_id=design_id, n_conditions=n, n_passing=len(passing),
        pass_rate=(len(passing) / n) if n else 0.0,
        worst_case_conditions=failing, points=tuple(points),
    )


def load_pvt_results_from_jsonl(
    design_id: str, path: str | Path, *, target: Optional[TargetSpec] = None,
) -> PVTRobustnessResult:
    """Summarizes an ALREADY-COMPUTED PVT sweep (e.g.
    results/design_a_pvt_minimal27.jsonl, written by experiments/pvt_sweep.py)
    -- no new SPICE. This is how this module is demonstrated below.
    """

    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    points = []
    for row in rows:
        metrics = dict(row.get("metrics") or {})
        simulator_success = bool(row["success"])
        assessment = (
            assess_target(metrics, target, simulator_success=simulator_success)
            if target is not None else None
        )
        points.append(PVTPointResult(
            process_corner=row["process_corner"], supply_v=row["supply_v"],
            temperature_c=row["temperature_c"],
            success=assessment.passed if assessment is not None else simulator_success,
            failed_stage=row.get("failed_stage"), simulator_success=simulator_success,
            metrics=metrics, target_assessment=assessment,
        ))
    return summarize_pvt_results(design_id, points)


def run_pvt_evaluation(
    design: FeasibleDesign,
    conditions: tuple[SimulationConditions, ...],
    *,
    fidelity: EvaluationFidelity = EvaluationFidelity.FINAL,
    target: Optional[TargetSpec] = None,
    evaluator_kwargs: Mapping[str, Any] | None = None,
) -> PVTRobustnessResult:
    """Actually spends real SPICE: runs the EXISTING, unmodified
    evaluate_pvt_grid for one candidate design against the given condition
    set. Callers choose which candidates and which (small) condition set to
    spend budget on -- this function never expands the set itself.
    """

    parameters = ReceiverParameters(**design.parameters)
    evaluations = evaluate_pvt_grid(
        parameters, conditions=conditions, fidelity=fidelity,
        **dict(evaluator_kwargs or {}),
    )
    points = []
    for condition, evaluation in zip(conditions, evaluations):
        metrics = dict(evaluation.metrics)
        assessment = (
            assess_target(metrics, target, simulator_success=evaluation.success)
            if target is not None else None
        )
        points.append(PVTPointResult(
            process_corner=condition.process_corner.value, supply_v=condition.supply_v,
            temperature_c=condition.temperature_c,
            success=assessment.passed if assessment is not None else evaluation.success,
            failed_stage=evaluation.failed_stage, simulator_success=evaluation.success,
            metrics=metrics, target_assessment=assessment,
        ))
    return summarize_pvt_results(design.design_id, points)


def rank_by_robustness(results: list[PVTRobustnessResult]) -> list[PVTRobustnessResult]:
    """Highest pass_rate first; ties broken by n_conditions descending (a
    design tested against more conditions and still tying on pass_rate is
    the better-evidenced candidate, not a coin flip).
    """

    return sorted(results, key=lambda r: (-r.pass_rate, -r.n_conditions))


def select_final_designs(
    results: list[PVTRobustnessResult], *, minimum_pass_rate: float = 1.0, top_n: int = 1,
) -> list[PVTRobustnessResult]:
    """Select up to ``top_n`` qualified designs, failing closed.

    A below-threshold design is never returned as a final selection. Callers
    may display ``rank_by_robustness(results)[0]`` separately as the best
    available non-qualifying design.
    """

    ranked = rank_by_robustness(results)
    meeting_bar = [r for r in ranked if r.pass_rate >= minimum_pass_rate]
    return meeting_bar[:top_n]


# [NEBULA ADAPTATION] documented, non-arbitrary selection priority (not a
# weighted score): (1) nominal feasibility -- guaranteed by construction,
# callers only pass PVTRobustnessResults for designs that already survived
# nominal screening (analysis.design_catalog / pipeline filter_nominal_
# feasible); (2) PVT pass_rate, highest first; (3) robustness tie-break,
# n_conditions descending (more-evidenced ties beat less-evidenced ties);
# (4) secondary trade-off preference among any PVT ties, via the ORIGINAL
# design's own measured metrics (analysis.design_catalog.
# rank_by_measured_trade_offs) -- never applied before priorities 1-3.
TRADE_OFF_PREFERENCES = ("most_robust", "lowest_power", "strongest_eye_height", "widest_eye", "largest_margin", "balanced")


def select_with_trade_off_preference(
    pvt_results: list[PVTRobustnessResult],
    designs: list[FeasibleDesign],
    *,
    preference: str = "most_robust",
    minimum_pass_rate: float = 1.0,
) -> Optional[PVTRobustnessResult]:
    """Full 4-level, documented priority: nominal feasibility (guaranteed by
    the caller only passing already-feasible designs/results) -> PVT pass
    rate -> robustness tie-break -> secondary trade-off preference among
    any remaining ties. `preference='most_robust'` (default) stops after
    priority 3 (returns the single most-robust candidate, ignoring
    trade-offs). Any other value in TRADE_OFF_PREFERENCES additionally
    prefers, among designs SHARING THE TOP pass_rate, whichever one's own
    measured metrics carry that trade-off label
    (analysis.design_catalog.rank_by_measured_trade_offs) -- never
    overriding a strictly higher pass_rate. Returns None if `pvt_results`
    is empty.
    """

    if preference not in TRADE_OFF_PREFERENCES:
        raise ValueError(f"unknown preference {preference!r}; choose from {TRADE_OFF_PREFERENCES}")
    if not pvt_results:
        return None

    ranked = rank_by_robustness([
        result for result in pvt_results if result.pass_rate >= minimum_pass_rate
    ])
    if not ranked:
        return None
    if preference == "most_robust":
        return ranked[0]

    top_pass_rate = ranked[0].pass_rate
    tied = [r for r in ranked if r.pass_rate == top_pass_rate]
    if len(tied) == 1:
        return tied[0]

    by_id = {d.design_id: d for d in designs}
    tied_designs = [by_id[r.design_id] for r in tied if r.design_id in by_id]
    if not tied_designs:
        return tied[0]  # no metrics available to break the tie further -- first by robustness order

    trade_off_ranked = rank_by_measured_trade_offs(tied_designs)
    preferred = next((entry for entry in trade_off_ranked if preference in entry.trade_off_labels), None)
    chosen_design_id = (preferred.design if preferred else trade_off_ranked[0].design).design_id
    return next(r for r in tied if r.design_id == chosen_design_id)


def _main() -> int:
    # Demonstration using ALREADY-COMPUTED, real PVT data -- zero new SPICE.
    # Both the original first-run result AND the later reproducibility-
    # investigation re-run are shown -- the original is historical evidence
    # (preserved, not overwritten or hidden) that motivated the diagnosis
    # documented in docs/autockt-mapping.md sec 22 Task 1; the re-run is the
    # current, most-verified characterization (see that section for why the
    # two differ: simulation-level non-reproducibility at 4 points in the
    # first run, not a design defect -- Design A itself was never changed).
    for label, path in (
        ("design_a (original first run)", "results/design_a_pvt_minimal27.jsonl"),
        ("design_a (reproducibility re-run)", "results/design_a_pvt_minimal27_rerun.jsonl"),
    ):
        result = load_pvt_results_from_jsonl(label, path)
        print(json.dumps({
            "design_id": result.design_id,
            "source": path,
            "n_conditions": result.n_conditions,
            "n_passing": result.n_passing,
            "pass_rate": result.pass_rate,
            "worst_case_conditions": [
                {"corner": p.process_corner, "vdd": p.supply_v, "temp_c": p.temperature_c,
                 "failed_stage": p.failed_stage}
                for p in result.worst_case_conditions
            ],
        }, indent=2))
        selected = select_final_designs([result], minimum_pass_rate=1.0, top_n=1)
        print(json.dumps({
            "selected": selected[0].design_id, "pass_rate": selected[0].pass_rate,
            "met_minimum_pass_rate": selected[0].pass_rate >= 1.0,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
