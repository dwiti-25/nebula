# NEBULA Phase 7 completion report

This report records the ordered Phase 1–7 integration pass performed from
`origin/main` commit `649ce331ccf971b9f934d2aebadb6e5c754bd719`. It does not
replace the historical experimental record in the Phase-2 artifact package.

## Ordered implementation status

| Phase | Outcome | Evidence |
|---|---|---|
| 1. Reproducibility baseline | Complete | Artifact checksums are executable tests; NumPy and PyTorch are both declared runtime dependencies. Dashboard readers and checkpoint loading work from extracted results or directly from the tracked ZIP. |
| 2. Decision correctness | Complete | Nominal and PVT qualification now use independent, strict, per-metric target checks. Missing metrics and simulator failures fail closed. No below-threshold PVT candidate is promoted. |
| 3. Runtime hardening | Complete | The UI retains subprocess isolation and uses a configuration-aware watchdog. A full PVT run receives fixed overhead plus three hours per possible candidate instead of the invalid fixed two-hour limit. |
| 4. Metrics aggregation | Complete | `analysis/performance_dashboard.py` deterministically aggregates archived evidence and preserves provenance/limitations. It performs no simulation. |
| 5. Graphical wrapper | Complete | The existing local UI exposes seven SVG views: PPO reward, cumulative target satisfaction, failure-stage counts, optimizer success, unseen-target checkpoint comparison, feasible-design eye/power trade-off, and the 27-point PVT matrix. |
| 6. Final qualification | Complete | Final reports include the exact requested target and a machine-readable strict target qualification in addition to the fixed official specification rows. PVT selection remains target-aware. |
| 7. Optimizer comparison | Complete for existing evidence | The report distinguishes raw-metric recomputation from same-target logged outcomes. It does not manufacture missing PPO metrics or claim PPO superiority. |

## What the performance display means

The display is an evidence viewer, not a substitute for an experiment. Each
graph is calculated from one of five named result sources and the response
includes those source paths and known limitations. On the packaged evidence it
shows 95 PPO evaluations, 21 logged per-step target satisfactions, eight
feasible designs, and a 27/27 Design-A PVT rerun. The optimizer plot shows the
budget-matched, one-seed trial and carries its statistical caveat. The
unseen-target plot deliberately shows the trained policy's regression (90% to
70%), rather than presenting training-only improvement as generalization.

An LLM may later summarize this structured `/api/evidence` response, but it
must not calculate or alter verdicts. The deterministic assessment and chart
data remain the source of truth. The LLM integration stays assigned to
Teammate B so this phase does not create a competing wrapper implementation.

## Stage-2 items reviewed again

The receiver-wrapper decisions deferred these hardware/modeling changes:

| Deferred item | Nine-day disposition | Why |
|---|---|---|
| Physical tail/bias source | Post-Phase-7 | Changes topology and invalidates existing checkpoints/evidence. |
| Grouped MOS sizing variables | Post-Phase-7 | Changes action/state spaces and requires retraining plus new baselines. |
| Physical sampler and DFE | Post-Phase-7 | Major circuit implementation; current `dfe_tap_v` remains explicitly behavioral. |
| Sampler/DFE power and area | Blocked by physical implementation | Cannot be credibly reported before the blocks exist. |
| Load and package/pad capacitance calibration | Qualification follow-up | Needs an approved package/load reference and new simulations. |
| Final real `.s4p` channel selection | Highest-priority evidence follow-up | Current synthetic channel is only a deterministic regression input. |
| PRBS15, clock jitter, mismatch/Monte Carlo | Qualification follow-up | Valuable robustness evidence, but not safe to mix into the frozen nine-day demonstration. |
| Stable hierarchical device operating-point extraction | Engineering follow-up | Useful for diagnostics and physical validity checks. |
| Sampling phase/threshold as design variables | Research follow-up | Changes the optimization formulation and requires a fresh fairness review. |
| Full 60-corner PVT grid | Evidence follow-up | Current claim is specifically the documented minimal 27-point grid, not all 60 Stage-1 conditions. |

These items are not silently dropped. They are excluded from Phase 7 because
implementing them now would change the model, topology, or evidence base and
would exceed the remaining schedule. The final presentation must retain these
scope boundaries.

## Evidence limits and next work

Phase 7 completes the integration and evidence-display layer; it does not make
the model scientifically complete. The next highest-value work is:

1. Teammate B finishes the natural-language-to-validated-`TargetSpec` wrapper
   against the existing pipeline and structured evidence endpoint.
2. Run a multi-seed, budget-matched PPO/Random Search/CEM comparison with raw
   metrics and wall-clock recorded for every evaluation.
3. Select and independently validate the final real channel, then add PRBS15,
   jitter, mismatch/Monte Carlo, and the full 60-corner qualification in that
   order.
4. Only after the demonstration freeze, decide whether to change the action
   reachability/horizon or undertake the Stage-2 physical-circuit expansion.

Claims remain bounded: training-target learning is demonstrated; unseen-target
generalization and PPO superiority are not; total circuit area is not known;
and the 27/27 rerun does not erase the preserved historical 23/27 run.

## Final verification

- Full discovery: **419 tests**, all successful; 24 skipped because they are
  platform-specific or require loose historical fixtures/SKY130 configuration.
- Clean-package fast suite: **346 tests**, all successful; 23 fixture-dependent
  tests skipped and no errors.
- Artifact integrity: all **145 archived files** match the tracked SHA-256 ledger.
- Python bytecode compilation and `git diff --check`: successful.
- The Windows host exposes ngspice; the optional SKY130 integration test remains
  skipped because its model-library environment variable is not configured.
