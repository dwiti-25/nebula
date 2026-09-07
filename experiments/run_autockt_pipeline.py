"""Task 1 (overnight chunk, sec 22): the clean end-to-end automated
pipeline --

    TargetSpec -> validate -> generate candidates (existing PPO policy,
    deterministic rollout, real SPICE per step) -> nominal feasibility
    filter -> PVT-aware selection (optional) -> final schematic export ->
    final specification report

Every stage below is a thin orchestration wrapper around ALREADY-EXISTING,
unmodified components -- nothing here reimplements PPO, the environment,
the reward, the simulator, or PVT evaluation. See each function's docstring
for exactly which existing module it delegates to. No new PPO training
happens anywhere in this file: candidate generation always uses
`agent.act(..., deterministic=True)` (a read-only forward pass) against
either an explicitly-loaded, already-trained checkpoint, or (for dry-run /
synthetic-backend testing only) a freshly-constructed, untrained policy --
`agent.update()` is never called.
"""

from __future__ import annotations

import os

# FINAL AUDIT gap E: must be set before numpy is imported (transitively,
# by the `from simulator...` imports below) to take effect. This process
# links numpy against Apple's Accelerate framework (ACCELERATE_NEW_LAPACK,
# confirmed via numpy.show_config() on this checkout's arm64 build), a
# combination with documented threading/reentrancy crash reports on
# Apple Silicon -- and this SAME process also loads PyTorch (imported
# below), which runs its own separate thread pool. A real, captured
# faulthandler traceback from one of this session's real-SPICE SIGSEGV
# crashes showed the fault inside numpy/linalg/_linalg.py, called from
# simulator/waveform.py::hd3_db's numpy.linalg.cond/lstsq (used by the
# HD3 measurement stage, simulator/receiver.py::_run_hd3) -- exactly the
# kind of call Accelerate's threading has been reported to fault on when
# another thread pool is active in the same process. Pinning BLAS/LAPACK
# to a single thread removes that race without changing any simulator
# math, algorithm, or numeric result -- an environment/process-level
# mitigation, not a change to simulator/waveform.py itself. setdefault()
# so an operator's own explicit choice is never overridden. Not proven to
# eliminate the crash (the failure was already known to be rare/
# intermittent, so absence of further crashes cannot be proven from a
# small number of trials) -- see docs/FINAL_TECHNICAL_AUDIT.md sec 4.E.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from simulator.config import ProcessCorner, SimulationConditions
from simulator.receiver import EvaluationFidelity, ReceiverParameters, evaluate_receiver
from simulator.rl_adapter import ReceiverRLAdapter, RLBudget

from rl.autockt_env import AutoCktReceiverEnv
from rl.autockt_state import STATE_DIM
from rl.autockt_state_v2 import STATE_DIM_V2
from rl.evaluation_cache import make_cached_evaluator
from rl.parameter_grid import PARAMETER_NAMES, build_parameter_grids
from rl.ppo_agent import PPOAgent
from rl.synthetic_benchmark import synthetic_evaluate_receiver
from rl.target_spec import SPEC_NAMES, TargetSpec

from experiments import pvt_sweep
from experiments.export_final_schematic import render_final_schematic
from experiments.train_autockt import _resolve_initial_indices

from analysis.design_catalog import FeasibleDesign, rank_by_measured_trade_offs
from analysis.artifact_store import read_packaged_bytes
from analysis.final_specification import build_final_specification_report, format_report
from analysis.target_assessment import assess_target
from analysis.pvt_selection import (
    PVTPointResult,
    PVTRobustnessResult,
    run_pvt_evaluation,
    select_with_trade_off_preference,
)


# ---------------------------------------------------------------------------
# CLI-only: named PVT condition sets (mirrors experiments/pvt_sweep.py's
# --condition-set pattern). "smoke" is intentionally the smallest set that
# still exercises the PVT-aware SELECTION LOGIC (pass-rate ranking, tie-break,
# trade-off preference) with more than one real condition. "minimal27"
# reuses experiments/pvt_sweep.py's own MINIMAL_27_CONDITIONS (the SAME
# 27-point set behind docs/autockt-mapping.md sec 22's 27/27 result) --
# it is exposed here as an explicit, opt-in choice so full PVT robustness
# checking is reachable from this pipeline/UI, but "none" stays the
# default and nothing here ever selects or runs it automatically. At
# ~121.8 min historically for one design (sec 22), it is exponentially
# more expensive per nominally-feasible candidate than "smoke" and should
# be chosen deliberately, not by default.
# ---------------------------------------------------------------------------

PVT_CONDITION_SETS: dict[str, Optional[tuple[SimulationConditions, ...]]] = {
    "none": None,
    "smoke": (
        SimulationConditions(ProcessCorner.TT, 27.0, 1.8),   # nominal
        SimulationConditions(ProcessCorner.FF, 125.0, 1.71),  # sec 22's worst-case corner, now 27/27-confirmed
    ),
    "minimal27": pvt_sweep.MINIMAL_27_CONDITIONS,
}

# Runtime diagnosis (docs/autockt-mapping.md sec 24 / RUNTIME investigation):
# select_final_design's own fidelity default (EvaluationFidelity.FINAL) was
# being silently inherited by EVERY non-"none" PVT set, including "smoke" --
# whose whole design intent (see the --pvt-condition-set help text below) is
# to be the CHEAP, small-condition-count option. A direct diagnostic (one
# known-good candidate, both smoke conditions, real ngspice) measured
# ~281-290s per condition at FINAL fidelity, ~571s combined for just ONE
# candidate -- legitimate, successful computation (no hang, no crash, no
# convergence failure) that nonetheless looks indistinguishable from a hung
# UI run, since nothing surfaces stage-level progress.
#
# CANDIDATE fidelity runs the IDENTICAL stage set as FINAL (verified by
# reading simulator/receiver.py directly: dc/ac/ctle_transient/channel_
# diagnostics/noise/hd3/transient are all gated at <=CANDIDATE) -- FINAL
# only adds a 3-amplitude HD3 characterization sweep (characterize=True)
# instead of CANDIDATE's single amplitude, which changes HD3 METRIC VALUES
# only. analysis.pvt_selection.run_pvt_evaluation reads only
# evaluation.success/failed_stage (never .metrics) to build PVTPointResult,
# and select_with_trade_off_preference's tie-break reads the candidate's
# already-computed NOMINAL metrics (from generate_candidates), never
# anything produced by the PVT sweep itself -- so PVT feasibility and
# "most robust"/trade-off selection use zero information FINAL adds over
# CANDIDATE. CANDIDATE is fidelity-sufficient for this consumer, and roughly
# halves the HD3 stage's contribution (~70-86s -> ~23-29s per condition,
# confirmed by direct measurement of both fidelities on the same candidate).
#
# "minimal27" (the actual robustness proof) deliberately keeps FINAL --
# only "smoke" (explicitly a cheap pre-check, never itself a robustness
# claim) is downgraded. Explicit per-named-set mapping, not a change to
# select_final_design's own default (still FINAL for any caller that
# doesn't pass pvt_fidelity, e.g. existing tests/other call sites).
PVT_CONDITION_SET_FIDELITY: dict[str, EvaluationFidelity] = {
    "smoke": EvaluationFidelity.CANDIDATE,
    "minimal27": EvaluationFidelity.FINAL,
}


# ---------------------------------------------------------------------------
# Stage 1: target validation
# ---------------------------------------------------------------------------

def validate_target(target: TargetSpec) -> list[str]:
    """Lightweight, simulator-free validation reusing rl.target_spec.SPEC_NAMES.
    Returns a list of problems; empty means valid.
    """

    problems: list[str] = []
    for name in SPEC_NAMES:
        value = getattr(target, name)
        if not math.isfinite(value):
            problems.append(f"{name} is not finite: {value}")
    if not (0 < target.dfe_eye_width_ui <= 1):
        problems.append(f"dfe_eye_width_ui out of plausible (0,1] UI range: {target.dfe_eye_width_ui}")
    if target.ctle_power_w <= 0:
        problems.append(f"ctle_power_w must be positive: {target.ctle_power_w}")
    return problems


# ---------------------------------------------------------------------------
# Stage 2/3: candidate generation via the existing RL framework + SPICE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineCandidate:
    episode: int
    parameters: dict[str, float]
    metrics: dict[str, float]
    reward: float
    spec_satisfied: bool
    steps: int
    evaluation_success: bool = True


def _load_checkpoint_payload(checkpoint_path: Path) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """Load a policy state from a loose file or the immutable artifact ZIP."""

    if checkpoint_path.is_file():
        checkpoint_source: Any = checkpoint_path
    else:
        packaged = read_packaged_bytes(PROJECT_ROOT, checkpoint_path)
        if packaged is None:
            raise FileNotFoundError(f"checkpoint not found on disk or in artifact package: {checkpoint_path}")
        checkpoint_source = io.BytesIO(packaged)
    payload = torch.load(checkpoint_source, weights_only=True)
    if "policy_state_dict" in payload:
        return payload["policy_state_dict"], payload.get("metadata")
    return payload, None


def _load_checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    """Backward-compatible inference-only policy loader."""
    return _load_checkpoint_payload(checkpoint_path)[0]


def generate_candidates(
    *,
    target: TargetSpec,
    checkpoint_path: Optional[Path],
    agent_seed: int = 42,
    eval_seed: int = 42,
    episodes: int = 3,
    horizon: int = 4,
    backend: str = "real",
    grid_points: int = 21,
    grid_spacing: str = "log",
    initial_indices_source: str = "verified",
    randomize_initial_state: bool = True,
    max_evaluations: int = 1000,
    rl_version: str = "v1",
    use_evaluation_cache: bool = False,
) -> list[PipelineCandidate]:
    """Deterministic rollout of an existing policy against `target`, via the
    real, UNMODIFIED AutoCktReceiverEnv + PPOAgent + ReceiverRLAdapter.
    `checkpoint_path=None` uses a freshly-constructed, untrained policy
    (only sensible for backend='synthetic' dry-run testing -- an untrained
    policy against real SPICE has no learned behavior to demonstrate, see
    docs/autockt-mapping.md sec 20). `backend='synthetic'` uses
    rl.synthetic_benchmark.synthetic_evaluate_receiver (no SPICE, no PDK
    needed); `backend='real'` calls real ngspice via the unmodified
    evaluate_receiver, once per environment step.
    """

    grids = build_parameter_grids(grid_points, spacing=grid_spacing)
    initial_indices = _resolve_initial_indices(grids, initial_indices_source)

    if rl_version not in {"v1", "v2"}:
        raise ValueError("rl_version must be 'v1' or 'v2'")
    adapter_kwargs: dict[str, Any] = {}
    if backend == "synthetic":
        adapter_kwargs["evaluator"] = synthetic_evaluate_receiver
    if use_evaluation_cache:
        adapter_kwargs["evaluator"] = make_cached_evaluator(
            adapter_kwargs.get("evaluator", evaluate_receiver)
        )
    adapter = ReceiverRLAdapter(budget=RLBudget(max_evaluations), seed=eval_seed, **adapter_kwargs)

    v2 = rl_version == "v2"
    env = AutoCktReceiverEnv(
        target_pool=(target,), initial_indices=initial_indices, horizon=horizon,
        adapter=adapter, grids=grids, seed=eval_seed, randomize_initial_state=randomize_initial_state,
        evaluate_on_reset=v2, state_schema=rl_version,
        use_reward_v2=v2, max_total_evaluations=max_evaluations,
    )
    agent = PPOAgent(state_dim=STATE_DIM_V2 if v2 else STATE_DIM,
                     num_heads=len(PARAMETER_NAMES), seed=agent_seed)
    if checkpoint_path is not None:
        policy_state, metadata = _load_checkpoint_payload(checkpoint_path)
        if metadata and metadata.get("state_schema") != rl_version:
            raise ValueError(
                f"checkpoint uses state schema {metadata.get('state_schema')!r}, "
                f"but --rl-version={rl_version!r} was requested"
            )
        agent.policy.load_state_dict(policy_state)

    candidates: list[PipelineCandidate] = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = truncated = False
        episode_reward = 0.0
        steps = 0
        last_parameters: Optional[dict[str, float]] = None
        last_metrics: Optional[dict[str, float]] = None
        last_success = False
        while not done and not truncated:
            choices, _log_prob, _value = agent.act(state, deterministic=True)
            step_out = env.step(choices)
            episode_reward += step_out.reward
            state = step_out.state
            done, truncated = step_out.done, step_out.truncated
            steps += 1
            last_parameters = step_out.info["parameters"]
            last_metrics = step_out.info["metrics"]
            last_success = bool(step_out.info["success"])
        candidates.append(PipelineCandidate(
            episode=episode, parameters=last_parameters or {}, metrics=last_metrics or {},
            reward=episode_reward, spec_satisfied=done, steps=steps,
            evaluation_success=last_success,
        ))
    return candidates


# ---------------------------------------------------------------------------
# Stage 4: nominal feasibility filter
# ---------------------------------------------------------------------------

def filter_nominal_feasible(
    candidates: list[PipelineCandidate], target: Optional[TargetSpec] = None,
) -> list[PipelineCandidate]:
    """Filter using strict per-metric target satisfaction, not reward tolerance."""

    effective_target = target or TargetSpec.from_existing_thresholds()
    return [
        candidate for candidate in candidates
        if assess_target(
            candidate.metrics, effective_target,
            simulator_success=candidate.evaluation_success,
        ).passed
    ]


# ---------------------------------------------------------------------------
# Stage 5: PVT-aware selection (optional -- spends real SPICE only if a
# condition set is explicitly given)
# ---------------------------------------------------------------------------

def _candidate_to_design(candidate: PipelineCandidate) -> FeasibleDesign:
    return FeasibleDesign(
        design_id=f"pipeline_ep{candidate.episode}", source_file="pipeline (in-memory, this run)",
        source_description=f"generate_candidates episode {candidate.episode}, "
        f"{candidate.steps} step(s), reward={candidate.reward:.3f}",
        parameters=candidate.parameters, metrics=candidate.metrics,
        native_reward=candidate.reward, native_reward_scale="autockt_reward",
    )


def select_final_design(
    feasible_candidates: list[PipelineCandidate],
    *,
    pvt_conditions: Optional[tuple] = None,
    pvt_fidelity=None,
    minimum_pass_rate: float = 1.0,
    trade_off_preference: str = "most_robust",
    target: Optional[TargetSpec] = None,
) -> dict[str, Any]:
    """Priority, deterministic and documented (not an arbitrary weighted
    score): (1) nominal feasibility -- already guaranteed by only receiving
    `feasible_candidates`; (2) PVT pass rate, if `pvt_conditions` given
    (highest first); (3) robustness tie-break (n_conditions descending);
    (4) among any remaining PVT ties, `trade_off_preference` (one of
    `analysis.pvt_selection.TRADE_OFF_PREFERENCES`) via
    `analysis.pvt_selection.select_with_trade_off_preference` -- never
    overriding a strictly higher PVT pass rate. When `pvt_conditions` is
    None, falls back to nominal-only trade-off ranking (unchanged). PVT is
    evaluated (spending real SPICE) ONLY for candidates reaching this
    stage, and ONLY if the caller explicitly supplies `pvt_conditions` --
    never automatic.
    """

    designs = [_candidate_to_design(c) for c in feasible_candidates]
    if not designs:
        return {"selected": None, "reason": "no nominally feasible candidates", "pvt": None}

    if pvt_conditions is None:
        ranked = rank_by_measured_trade_offs(designs)
        best = ranked[0].design
        return {
            "selected": {"design_id": best.design_id, "parameters": best.parameters, "metrics": best.metrics},
            "trade_off_labels": list(ranked[0].trade_off_labels),
            "pvt": None,
            "selection_basis": "nominal-only (no PVT conditions supplied)",
        }

    fidelity = pvt_fidelity or EvaluationFidelity.FINAL
    pvt_results: list[PVTRobustnessResult] = [
        run_pvt_evaluation(d, pvt_conditions, fidelity=fidelity, target=target) for d in designs
    ]
    top = select_with_trade_off_preference(
        pvt_results, designs, preference=trade_off_preference, minimum_pass_rate=minimum_pass_rate,
    )
    if top is None:
        best_available = sorted(pvt_results, key=lambda result: (-result.pass_rate, -result.n_conditions))[0]
        return {
            "selected": None,
            "reason": f"no candidate met minimum PVT pass rate {minimum_pass_rate:.1%}",
            "best_available": {
                "design_id": best_available.design_id,
                "pass_rate": best_available.pass_rate,
                "n_passing": best_available.n_passing,
                "n_conditions": best_available.n_conditions,
            },
            "pvt": None,
            "selection_basis": "PVT qualification failed closed",
        }
    matching_design = next(d for d in designs if d.design_id == top.design_id)
    return {
        "selected": {
            "design_id": matching_design.design_id, "parameters": matching_design.parameters,
            "metrics": matching_design.metrics,
        },
        "pvt": {
            "n_conditions": top.n_conditions, "n_passing": top.n_passing, "pass_rate": top.pass_rate,
            "met_minimum_pass_rate": top.pass_rate >= minimum_pass_rate,
            "worst_case_conditions": [
                {"corner": p.process_corner, "vdd": p.supply_v, "temp_c": p.temperature_c,
                 "failed_stage": p.failed_stage}
                for p in top.worst_case_conditions
            ],
        },
        "selection_basis": (
            f"PVT-ranked across {len(pvt_conditions)} conditions, "
            f"trade_off_preference={trade_off_preference!r}"
        ),
    }


def _pvt_result_from_selection(selection: dict[str, Any]) -> Optional[PVTRobustnessResult]:
    """Reconstructs the PVTRobustnessResult select_final_design already
    computed (real SPICE, if pvt_conditions was given) from its
    JSON-serializable `selection["pvt"]` summary, so run_pipeline can feed
    the SAME result into build_final_specification_report instead of
    dropping it (or, worse, re-running PVT a second time to get an object
    back). Only `worst_case_conditions`/summary fields are reconstructed --
    `points` (the full per-condition list) is not part of the summary dict
    and is not needed by build_final_specification_report/format_report,
    which only read the summary fields.
    """

    pvt = selection.get("pvt")
    if pvt is None:
        return None
    return PVTRobustnessResult(
        design_id=selection["selected"]["design_id"],
        n_conditions=pvt["n_conditions"], n_passing=pvt["n_passing"], pass_rate=pvt["pass_rate"],
        worst_case_conditions=tuple(
            PVTPointResult(
                process_corner=w["corner"], supply_v=w["vdd"], temperature_c=w["temp_c"],
                success=False, failed_stage=w.get("failed_stage"),
            )
            for w in pvt["worst_case_conditions"]
        ),
        points=(),
    )


# ---------------------------------------------------------------------------
# Optional stage: HD3/input-referred-noise refinement (final audit gaps A/B)
#
# simulator/receiver.py already implements both measurements for real
# (_run_hd3: 100 mVpp differential at 100 MHz, three amplitudes at FINAL
# fidelity; _run_noise: 10 MHz-5 GHz integrated input-referred noise) --
# they are not missing infrastructure. They only run at
# fidelity >= EvaluationFidelity.CANDIDATE, but generate_candidates() uses
# TRAINING fidelity throughout (the cheaper level PPO's rollout needs), so
# a freshly-generated candidate's own metrics never include them. This
# stage runs ONE additional real-SPICE evaluation, at FINAL fidelity and
# nominal conditions, of the ALREADY-SELECTED design only -- never during
# search/optimization, never for rejected candidates -- to obtain the
# genuine values for the final specification report. Off by default (an
# explicit opt-in, since it is one more real-SPICE evaluation on top of
# whatever candidate generation already spent).
# ---------------------------------------------------------------------------

def measure_hd3_and_noise(
    parameters: dict[str, float], *, conditions: SimulationConditions = SimulationConditions(),
) -> dict[str, Any]:
    """Runs the existing, unmodified evaluate_receiver at FINAL fidelity for
    one design at nominal conditions -- the only fidelity level at which
    HD3 and input-referred noise are measured. Returns
    {"success", "metrics", "failed_stage"}; never fabricates a value --
    if this evaluation itself fails (a stricter FINAL-fidelity gate can
    reject a design that passed at TRAINING fidelity), the caller keeps
    reporting NOT CLAIMED for HD3/noise rather than inventing a number.
    """

    evaluation = evaluate_receiver(ReceiverParameters(**parameters), conditions, EvaluationFidelity.FINAL)
    return {"success": evaluation.success, "metrics": dict(evaluation.metrics), "failed_stage": evaluation.failed_stage}


# ---------------------------------------------------------------------------
# PVT progress reporting (experiments/web_ui.py "Add visible PVT progress"):
# small, purely additive stdout output so a subprocess caller can show real
# progress for a long PVT run instead of an opaque "running" state. Does
# NOT change select_final_design/PVT_CONDITION_SET_FIDELITY (commit
# 54eeb04) or any evaluation result: each single-condition call below is
# behaviorally identical to evaluate_pvt_grid's own per-condition loop
# (analysis.pvt_selection.run_pvt_evaluation never passes
# stop_on_failure=True, so there is no early-exit difference to preserve).
# ---------------------------------------------------------------------------

def _pvt_progress_evaluator(feasible_candidates: list[PipelineCandidate], real_evaluate_pvt_grid):
    """Wraps analysis.pvt_selection.evaluate_pvt_grid (same call signature)
    to emit one flush=True JSON line per (design, condition) pair -- start
    and complete -- to stdout. select_final_design calls run_pvt_evaluation
    once per design, in the same order as feasible_candidates, so a simple
    counter recovers which design_id (matching _candidate_to_design's own
    f"pipeline_ep{episode}" naming) each call belongs to.
    """
    total_designs = len(feasible_candidates)
    design_index = 0

    def wrapped(parameters, *, conditions, fidelity):
        nonlocal design_index
        design_id = f"pipeline_ep{feasible_candidates[design_index].episode}"
        total_conditions = len(conditions)
        results = []
        for condition_index, condition in enumerate(conditions):
            print(json.dumps({
                "nebula_progress_event": "pvt_condition_start",
                "design_id": design_id, "design_index": design_index, "total_designs": total_designs,
                "condition_index": condition_index, "total_conditions_per_design": total_conditions,
                "corner": condition.process_corner.value, "temperature_c": condition.temperature_c,
                "supply_v": condition.supply_v,
            }), flush=True)
            (evaluation,) = real_evaluate_pvt_grid(parameters, conditions=(condition,), fidelity=fidelity)
            results.append(evaluation)
            print(json.dumps({
                "nebula_progress_event": "pvt_condition_complete",
                "design_id": design_id, "design_index": design_index, "total_designs": total_designs,
                "condition_index": condition_index, "total_conditions_per_design": total_conditions,
                "corner": condition.process_corner.value, "temperature_c": condition.temperature_c,
                "supply_v": condition.supply_v,
                "success": evaluation.success, "failed_stage": evaluation.failed_stage,
            }), flush=True)
        design_index += 1
        return tuple(results)

    return wrapped


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    *,
    target: TargetSpec,
    checkpoint_path: Optional[Path],
    agent_seed: int = 42,
    eval_seed: int = 42,
    episodes: int = 3,
    horizon: int = 4,
    backend: str = "real",
    randomize_initial_state: bool = True,
    initial_indices_source: str = "verified",
    pvt_conditions: Optional[tuple] = None,
    pvt_fidelity: Optional[EvaluationFidelity] = None,
    trade_off_preference: str = "most_robust",
    measure_hd3_noise_flag: bool = False,
    export_schematic_to: Optional[Path] = None,
    rl_version: str = "v1",
    use_evaluation_cache: bool = False,
) -> dict[str, Any]:
    problems = validate_target(target)
    if problems:
        raise ValueError(f"invalid target specification: {problems}")

    candidates = generate_candidates(
        target=target, checkpoint_path=checkpoint_path, agent_seed=agent_seed, eval_seed=eval_seed,
        episodes=episodes, horizon=horizon, backend=backend, randomize_initial_state=randomize_initial_state,
        initial_indices_source=initial_indices_source,
        rl_version=rl_version, use_evaluation_cache=use_evaluation_cache,
    )
    feasible = filter_nominal_feasible(candidates, target)
    print(json.dumps({
        "nebula_progress_event": "candidate_generation_complete",
        "n_feasible": len(feasible),
        "pvt_conditions_total": (len(feasible) * len(pvt_conditions)) if pvt_conditions else 0,
    }), flush=True)

    if pvt_conditions is not None:
        import analysis.pvt_selection as _pvt_selection_module
        _real_evaluate_pvt_grid = _pvt_selection_module.evaluate_pvt_grid
        _pvt_selection_module.evaluate_pvt_grid = _pvt_progress_evaluator(feasible, _real_evaluate_pvt_grid)
        try:
            selection = select_final_design(
                feasible, pvt_conditions=pvt_conditions, pvt_fidelity=pvt_fidelity,
                trade_off_preference=trade_off_preference, target=target,
            )
        finally:
            _pvt_selection_module.evaluate_pvt_grid = _real_evaluate_pvt_grid
    else:
        selection = select_final_design(
            feasible, pvt_conditions=pvt_conditions, pvt_fidelity=pvt_fidelity,
            trade_off_preference=trade_off_preference, target=target,
        )

    result: dict[str, Any] = {
        "target": target.as_dict(),
        "backend": backend,
        "rl_version": rl_version,
        "state_schema": rl_version,
        "reward_schema": "reward_v2" if rl_version == "v2" else "autockt_reward_v1",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "n_candidates_generated": len(candidates),
        "n_nominally_feasible": len(feasible),
        "selection": selection,
        "candidate_assessments": [
            {
                "episode": candidate.episode,
                "autockt_terminal_success": candidate.spec_satisfied,
                "strict_target": assess_target(
                    candidate.metrics, target, simulator_success=candidate.evaluation_success,
                ).to_dict(),
            }
            for candidate in candidates
        ],
    }

    if selection["selected"] is not None and measure_hd3_noise_flag and backend == "real":
        refinement = measure_hd3_and_noise(selection["selected"]["parameters"])
        result["hd3_noise_refinement"] = {
            "attempted": True, "success": refinement["success"], "failed_stage": refinement["failed_stage"],
        }
        if refinement["success"]:
            # Merge -- keep the original TRAINING-fidelity metrics and add/
            # override with the richer FINAL-fidelity set (which includes
            # hd3_db/input_referred_noise_vrms, absent before this point).
            selection["selected"]["metrics"] = {**selection["selected"]["metrics"], **refinement["metrics"]}
    elif measure_hd3_noise_flag and backend != "real":
        result["hd3_noise_refinement"] = {
            "attempted": False, "success": None, "failed_stage": None,
            "note": "measure_hd3_noise_flag has no effect for backend='synthetic' -- "
                    "HD3/noise are real-ngspice-only measurements.",
        }

    if selection["selected"] is not None and export_schematic_to is not None:
        parameters = ReceiverParameters(**selection["selected"]["parameters"])
        schematic_text = render_final_schematic(
            parameters, achieved_metrics=selection["selected"]["metrics"],
            target_description=str(target.as_dict()),
            source_description=f"experiments.run_autockt_pipeline, backend={backend}",
        )
        export_schematic_to.parent.mkdir(parents=True, exist_ok=True)
        export_schematic_to.write_text(schematic_text, encoding="utf-8")
        result["schematic_path"] = str(export_schematic_to)

    if selection["selected"] is not None:
        source_note = f"pipeline run, backend={backend}, checkpoint={checkpoint_path}"
        if result.get("hd3_noise_refinement", {}).get("success"):
            source_note += " + FINAL-fidelity HD3/noise refinement (measure_hd3_and_noise)"
        report = build_final_specification_report(
            design_id=selection["selected"]["design_id"],
            parameters=selection["selected"]["parameters"],
            nominal_metrics=selection["selected"]["metrics"],
            nominal_source=source_note,
            pvt_result=_pvt_result_from_selection(selection),
            target=target,
        )
        result["final_specification"] = report

    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end NEBULA pipeline: target -> PPO -> SPICE -> final design.")
    parser.add_argument("--target-mode", choices=("trivial", "hard"), default="trivial")
    parser.add_argument(
        "--target-json", type=str, default=None,
        help="Optional JSON object with the 4 TargetSpec fields "
             f"({', '.join(SPEC_NAMES)}) to use an arbitrary target instead of "
             "--target-mode's two presets -- e.g. for a UI/judge-entered spec. "
             "Still just constructs rl.target_spec.TargetSpec, the same class "
             "--target-mode uses; no new target semantics.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--rl-version", choices=("v1", "v2"), default="v1",
                        help="Versioned PPO configuration; v1 remains the historical baseline.")
    parser.add_argument("--evaluation-cache", action="store_true",
                        help="Cache exact repeated evaluator calls (recommended for PPO v2).")
    parser.add_argument("--backend", choices=("real", "synthetic"), default="synthetic")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--randomize-initial-state", action="store_true", default=True)
    parser.add_argument("--initial-indices-source", choices=("verified", "grid-center"), default="verified")
    parser.add_argument("--pvt-condition-set", choices=tuple(PVT_CONDITION_SETS), default="none",
                         help="'none' (default, unchanged behavior): NOMINAL-ONLY selection -- the selected design "
                              "is NOT validated across process/voltage/temperature, only at nominal TT/1.8V/27C; "
                              "do not read a 'none' run as PVT-robust. 'smoke': 2 conditions (nominal TT + one "
                              "stress corner), evaluated at EvaluationFidelity.CANDIDATE (not FINAL -- CANDIDATE "
                              "runs the identical stage set and is sufficient for pass/fail feasibility and "
                              "trade-off selection, see PVT_CONDITION_SET_FIDELITY's comment) -- exercises the "
                              "PVT-aware selection pathway with real SPICE, still NOT a robustness proof. "
                              "'minimal27': the full 27-point TT/SS/FF x VDD+/-5% x "
                              "0-125C robustness sweep (experiments/pvt_sweep.py's own MINIMAL_27_CONDITIONS) -- "
                              "the only choice that constitutes an actual PVT robustness result; SLOW "
                              "(~121.8 min historically for one design, docs/autockt-mapping.md sec 22) and never "
                              "selected automatically.")
    parser.add_argument("--trade-off-preference", choices=("most_robust", "lowest_power", "strongest_eye_height",
                                                             "widest_eye", "largest_margin", "balanced"),
                         default="most_robust")
    parser.add_argument(
        "--measure-hd3-noise", action="store_true", default=False,
        help="After selection, run ONE additional real-SPICE evaluation (FINAL fidelity, nominal conditions) of "
             "the selected design to measure HD3 (100 mVpp differential) and 10 MHz-5 GHz input-referred noise -- "
             "both otherwise NOT CLAIMED, since candidate generation runs at the cheaper TRAINING fidelity that "
             "never reaches those stages (simulator/receiver.py::_run_hd3/_run_noise). No effect for "
             "--backend synthetic. Off by default (adds real SPICE time).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-schematic", type=Path, default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    if args.target_json is not None:
        spec_values = json.loads(args.target_json)
        missing = [name for name in SPEC_NAMES if name not in spec_values]
        if missing:
            raise ValueError(f"--target-json is missing required field(s): {missing}")
        target = TargetSpec(**{name: float(spec_values[name]) for name in SPEC_NAMES})
    else:
        target = TargetSpec.from_hard_target() if args.target_mode == "hard" else TargetSpec.from_existing_thresholds()

    result = run_pipeline(
        target=target, checkpoint_path=args.checkpoint, agent_seed=args.agent_seed, eval_seed=args.eval_seed,
        episodes=args.episodes, horizon=args.horizon, backend=args.backend,
        randomize_initial_state=args.randomize_initial_state,
        initial_indices_source=args.initial_indices_source,
        pvt_conditions=PVT_CONDITION_SETS[args.pvt_condition_set],
        pvt_fidelity=PVT_CONDITION_SET_FIDELITY.get(args.pvt_condition_set),
        trade_off_preference=args.trade_off_preference,
        measure_hd3_noise_flag=args.measure_hd3_noise,
        export_schematic_to=args.export_schematic,
        rl_version=args.rl_version, use_evaluation_cache=args.evaluation_cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "final_specification"}, indent=2))
    if "final_specification" in result:
        print(format_report(result["final_specification"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
