# Checkpoint-compatible runtime integration

Existing policy weights remain usable. State dimensions, action heads, action
steps, parameter grids and reward formulas are unchanged by this integration.
Physical DFE/sampler and the teammate's nine-variable physical-bias policy are
excluded. Behavioral DFE remains. Numerical floating-point arithmetic is still
required; this work does not convert SPICE or PPO to integer arithmetic.

## Integrated changes

- Two-worker training (`--workers 2`), with main-thread policy sampling,
  independent episode environments, preallocated evaluation allowances and
  contiguous trajectories. Default remains one worker. Parallel training is
  not bit-for-bit equivalent to serial training; full resume validates the
  worker configuration. Existing serial checkpoints retain their contract.
- Exact-key caching coalesces concurrent duplicate requests. Retryable and
  internal failures are not retained. Shared run-graph writes are synchronized.
- `--max-evaluations` and `--search-seconds` bound search/training. Resets count
  against the evaluation allowance. Deadlines propagate into SPICE subprocess
  timeouts; Python/setup/optimizer overhead makes this a soft overall budget.
  Expensive validation remains outside the training/search deadline.
- Qualification rejects absent, non-finite and nonnumeric measurements, and
  does not use reward to determine acceptance. Failed/incomplete FINAL
  refinement clears selection, preventing export of a failed finalist.
- PVT results retain actual condition identities. A passing subset, duplicated
  conditions or summary-only historical record cannot prove full coverage.
  `full60` is available in CLI and UI; completing it remains expensive.
- Noise-bench common-mode resistor renamed to avoid bias-parameter collision.
  AC metrics separately expose `boost_2p5ghz_db` and `global_peak_boost_db`;
  existing reward features retain their meaning.
- Separate resumable finalist validation, repeatable real-batch benchmark,
  and simulator-free GitHub test workflow.

## Usage

Append `--workers 2 --evaluation-cache --max-evaluations 1000 --search-seconds 600`
to a new training command. Existing models need no retraining to benefit from
caching, qualification/reporting corrections, or new validation tools.
The same budget controls are exposed in the web UI. Workers affect training
only; inference still uses the existing serial candidate generator.

Validate an existing pipeline result independently:

```powershell
.venv\Scripts\python.exe -m experiments.validate_finalist --input results/my_run.json --output results/my_validation.jsonl --cache results/final_validation_cache --seconds 600
```

Add `--full60` for the full grid. Rerun with a **new output filename** and the
same cache to reuse completed content-addressed evaluations. Retryable failures
are reevaluated. The input pipeline's target, channel and port mapping are
preserved; plain parameter JSON uses the repository's default target/channel.
No passing selection is fabricated when validation is incomplete or fails.

## Local evidence (2026-09-12)

Two uncached, fixed-design evaluations per batch with NGSpice 47/KLU:

| Fidelity | Serial | Two workers | Reduction | Results |
|---|---:|---:|---:|---|
| SCREENING | 2.92 s | 2.22 s | 23.9% | Identical, both pass |
| TRAINING | 15.06 s | 9.70 s | 35.6% | Identical, both pass |

Raw reports: `results/runtime_parallel_screening.json` and
`results/runtime_parallel_training.json`. Reproduce using
`python -m experiments.benchmark_parallel_evaluation --fidelity TRAINING --output NEW_REPORT.json`.
This small fixed-order benchmark is not a general speedup guarantee, nor
evidence of better learned-policy accuracy or superiority over random/CEM.

No new full-60 validation, long-pattern eye, jitter sweep, tuning sweep or
physical-circuit validation is claimed. Existing early CTLE screening already
uses its short stimulus independently of FINAL settings and is preserved.
