"""Task 5 (NEXT IMPLEMENTATION CHUNK): honest, consolidated RS/CEM/PPO
benchmark report.

Computes nothing new -- reuses the already-tested MethodSummary objects
from analysis/fair_comparison.py (which itself only reads existing
results/*.jsonl files) and adds, for each of the two trials this repo has
actually run, the explicit structural metadata the comparison needs to be
read honestly: evaluation budget, initialization, search space,
sequential-vs-one-shot nature, success criterion, wall-clock, and whether
the trial is genuinely comparable across methods. No new SPICE is run
here; if a field cannot be computed from an existing file it is reported
as unavailable, never fabricated.

Two trials:

  "warm_started_ppo_sec19" -- the FIRST cross-method comparison
    (docs/autockt-mapping.md sec 19). Random Search and CEM ran a fresh,
    unbiased n=20 budget each; PPO's contribution is instead the
    trivial-target-tagged SUBSET of a full mixed-target TRAINING run
    started from VERIFIED_INITIAL_PARAMETERS (itself derived from Random
    Search's own best result) -- explicitly marked non-comparable.

  "no_warm_start_headtohead_sec20" -- the deliberate follow-up
    (docs/autockt-mapping.md sec 20) that removes sec 19's own named
    confounds as far as possible without touching the locked PPO
    formulation: same n=20 budget, same centered/unbiased initial
    distribution, same seed=123, for all three methods. The closest
    structurally fair trial this repo has run -- still not perfectly
    comparable (see its own comparability_caveat).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from analysis.fair_comparison import MethodSummary, summarize_cem, summarize_ppo_subset, summarize_random_search
from rl.target_spec import TargetSpec

TRIVIAL_TARGET = TargetSpec.from_existing_thresholds()


@dataclass(frozen=True)
class BenchmarkTrial:
    trial_id: str
    description: str
    evaluation_budget: str
    initialization: str
    search_space: str
    sequential_vs_one_shot: str
    success_criterion: str
    comparable_across_methods: bool
    comparability_caveat: str
    methods: dict[str, MethodSummary]


def warm_started_trial() -> BenchmarkTrial:
    methods = {
        "random_search": summarize_random_search("results/receiver_random_search_20_seed123.jsonl"),
        "cem": summarize_cem(
            "results/cem_unbiased_seed123.jsonl", warm_started=False, complete=True, planned=20,
        ),
        "ppo": summarize_ppo_subset(
            "results/autockt_mixed_target_confirmation.jsonl", target=TRIVIAL_TARGET, label="trivial",
        ),
    }
    return BenchmarkTrial(
        trial_id="warm_started_ppo_sec19",
        description=(
            "First cross-method trial. Random Search and CEM ran a fresh, "
            "unbiased n=20 one-shot budget each; PPO's row is instead the "
            "trivial-target-tagged SUBSET of a full mixed-target TRAINING "
            "run started from VERIFIED_INITIAL_PARAMETERS (itself derived "
            "from Random Search's own best result)."
        ),
        evaluation_budget=(
            "Random Search / CEM: n=20 each, fixed in advance. "
            "PPO: 65 trivial-tagged steps out of a 95-evaluation, "
            "~87-minute shared-weight training run (budget not separable "
            "from the interleaved hard-target training)."
        ),
        initialization=(
            "Random Search: uniform over the full continuous [-1,1]^5 "
            "action space. CEM: Gaussian mean=0, std=2/sqrt(12) (unbiased). "
            "PPO: WARM-STARTED at VERIFIED_INITIAL_PARAMETERS, itself "
            "seeded from Random Search's own candidate_index 8 result -- "
            "not an independent starting point."
        ),
        search_space=(
            "Random Search/CEM: continuous [-1,1]^5. PPO: 21-point "
            "log-spaced discretized grid per parameter."
        ),
        sequential_vs_one_shot=(
            "Random Search/CEM: independent, i.i.d. one-shot draws. "
            "PPO: sequential, on-policy steps within a shared-weight "
            "training run, interleaved in real evaluation order with "
            "steps against a different (hard) target."
        ),
        success_criterion=(
            "autockt_reward >= 10.0 against TargetSpec.from_existing_thresholds() "
            "(the 'trivial' target), for every method."
        ),
        comparable_across_methods=False,
        comparability_caveat=(
            "PPO's number here is inflated by inheriting Random Search's "
            "own best-known starting point (confound), and its budget is "
            "not separable from a much larger multi-target training run. "
            "See the no_warm_start_headtohead_sec20 trial for the run that "
            "removes these confounds."
        ),
        methods=methods,
    )


def no_warm_start_headtohead_trial() -> BenchmarkTrial:
    methods = {
        "random_search": summarize_random_search("results/receiver_random_search_20_seed123.jsonl"),
        "cem": summarize_cem(
            "results/cem_graded_unbiased_seed123.jsonl", warm_started=False, complete=True, planned=20,
        ),
        "ppo": summarize_ppo_subset(
            "results/autockt_fair_headtohead_seed123.jsonl", target=TRIVIAL_TARGET, label="trivial",
        ),
    }
    return BenchmarkTrial(
        trial_id="no_warm_start_headtohead_sec20",
        description=(
            "Deliberate follow-up trial removing sec 19's own named "
            "confounds as far as possible without touching the locked PPO "
            "formulation: same n=20 budget, same centered/unbiased initial "
            "distribution, same seed=123, for all three methods."
        ),
        evaluation_budget=(
            "n=20 for all three methods, exactly (Random Search reused; "
            "CEM 4 iterations x population 5; PPO 4 updates x 5 episodes "
            "x horizon=1)."
        ),
        initialization=(
            "All three centered/unbiased, not warm-started: Random Search "
            "uniform over [-1,1]^5 (unchanged); CEM Gaussian mean=0, "
            "std=2/sqrt(12); PPO grid-center (index 10 of 21 per "
            "parameter, the true geometric midpoint of each bound, "
            "confirmed structurally NOT equal to VERIFIED_INITIAL_PARAMETERS)."
        ),
        search_space=(
            "Random Search/CEM: continuous [-1,1]^5. PPO: 21-point "
            "log-spaced discretized grid per parameter (unresolved "
            "structural difference, unchanged from sec 19)."
        ),
        sequential_vs_one_shot=(
            "Random Search/CEM: independent, i.i.d. one-shot draws. "
            "PPO: horizon=1 (exactly one action-then-evaluate step per "
            "episode) -- the closest structurally honest match to a "
            "one-shot budget achievable without changing PPO's actual "
            "step/episode mechanics; still not identical (each PPO point "
            "is grid-center perturbed by at most one randomized index "
            "plus one further +/-1/+2 index-delta action, not a free "
            "draw anywhere in the space)."
        ),
        success_criterion=(
            "autockt_reward >= 10.0 against TargetSpec.from_existing_thresholds() "
            "(the 'trivial' target), for every method -- unchanged from sec 19."
        ),
        comparable_across_methods=True,
        comparability_caveat=(
            "The closest fair trial this repo has run, but n=1 seed only "
            "(not a distribution) and the discretized-grid-vs-continuous-"
            "space and step-local-vs-global-draw structural differences "
            "remain unresolved. See docs/autockt-mapping.md sec 20 E for "
            "the full limitations list."
        ),
        methods=methods,
    )


CONCLUSION = (
    "Across both trials run this session, PPO shows no demonstrated "
    "practical advantage over Random Search on this specific circuit-"
    "sizing problem. The warm-started trial's apparently-higher PPO "
    "success rate is explained by starting from Random Search's own "
    "best-known point, not by independent search capability. The "
    "no-warm-start, budget-matched, same-seed trial found PPO and CEM "
    "both at 0/20 successes, against Random Search's 1/20 -- a result "
    "this repo's own analysis (docs/autockt-mapping.md sec 20 E) "
    "explicitly says lacks the statistical power (n=1 seed, ~5% base "
    "rate) to conclude anything about relative optimizer quality, and "
    "that primarily demonstrates PPO's reachability constraint under a "
    "short horizon and an unbiased start rather than a flaw in its "
    "learning algorithm. This module does not overturn or restate that "
    "conclusion differently -- it consolidates the same, "
    "already-verified numbers into one queryable, tested artifact."
)


def _method_summary_dict(m: MethodSummary) -> dict:
    # Prefer independently recomputed raw-metric evidence. Historical PPO
    # logs omitted raw metrics, but these target-filtered rows were natively
    # evaluated against this exact comparison target, so their logged
    # target outcome is usable with a clearly weaker evidence grade.
    if m.uniform_success_rate is not None:
        reported_strict_rate = m.uniform_success_rate
        evidence_grade = "raw_metrics_recomputed"
    elif m.method == "ppo" and "target-filtered subset" in m.source_file:
        reported_strict_rate = m.native_success_rate
        evidence_grade = "logged_same_target_outcome_raw_metrics_unavailable"
    else:
        reported_strict_rate = None
        evidence_grade = "unavailable"
    return {
        "source_file": m.source_file,
        "n_evaluations": m.n_evaluations,
        "native_success_rate": m.native_success_rate,
        "uniform_success_rate": m.uniform_success_rate,
        "reported_strict_success_rate": reported_strict_rate,
        "strict_success_evidence_grade": evidence_grade,
        "uniform_evaluations_to_first_success": m.uniform_evaluations_to_first_success,
        "best_uniform_reward": m.best_uniform_reward,
        "wall_clock_total_s": m.wall_clock_total_s,
        "wall_clock_available": m.wall_clock_available,
        "caveats": list(m.caveats),
    }


def _trial_dict(t: BenchmarkTrial) -> dict:
    return {
        "description": t.description,
        "evaluation_budget": t.evaluation_budget,
        "initialization": t.initialization,
        "search_space": t.search_space,
        "sequential_vs_one_shot": t.sequential_vs_one_shot,
        "success_criterion": t.success_criterion,
        "comparable_across_methods": t.comparable_across_methods,
        "comparability_caveat": t.comparability_caveat,
        "methods": {name: _method_summary_dict(m) for name, m in t.methods.items()},
    }


def build_benchmark_report() -> dict:
    trials = {
        "warm_started_ppo_sec19": warm_started_trial(),
        "no_warm_start_headtohead_sec20": no_warm_start_headtohead_trial(),
    }
    return {
        "schema_version": 2,
        "trials": {trial_id: _trial_dict(t) for trial_id, t in trials.items()},
        "conclusion": CONCLUSION,
    }


def _main() -> int:
    report = build_benchmark_report()
    output_path = Path("results/benchmark_report.json")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {output_path}")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
