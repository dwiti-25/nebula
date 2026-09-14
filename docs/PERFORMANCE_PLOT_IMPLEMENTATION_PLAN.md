# NEBULA performance-plot implementation plan

## Objective

Replace the browser's approximate handwritten SVG charts with reproducible,
publication-quality Matplotlib figures. Preserve the current deterministic
evidence aggregator as the numerical source of truth. Plotting must never
change an experiment, qualification verdict, or source record.

This plan deliberately separates responsibilities:

- **Team B owns RL event production and controlled RL evaluation records.**
- **The integration team owns statistical aggregation, rendering, storage,
  browser delivery, report export, tests, and scientific claim controls.**

## Research basis

The plot suite follows these primary sources:

1. [AutoCkt](https://arxiv.org/abs/2001.01808) reports mean reward against
   environment steps, sample efficiency in simulator calls, reached versus
   unreached target-space distributions, representative optimization
   trajectories, and schematic-versus-layout differences. These map directly
   to NEBULA's expensive SPICE-in-the-loop setting.
2. The original [PPO paper](https://arxiv.org/abs/1707.06347) evaluates return
   against environment interaction, motivating learning curves indexed by
   real evaluation count rather than only episode number or wall-clock time.
3. Agarwal et al., [Deep Reinforcement Learning at the Edge of the Statistical
   Precipice](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html),
   recommends interval estimates, interquartile means, performance profiles,
   and probability of improvement for few-run RL comparisons.
4. Jordan et al., [Evaluating the Performance of Reinforcement Learning
   Algorithms](https://proceedings.mlr.press/v119/jordan20a.html), motivates a
   comprehensive evaluation protocol instead of selecting whichever scalar
   metric makes an algorithm look strongest.
5. Eimer et al., [Hyperparameters in Reinforcement Learning and How To Tune
   Them](https://proceedings.mlr.press/v202/eimer23a.html), supports separating
   tuning seeds from final test seeds to avoid reporting seed-overfit gains.
6. [Dyna-style analog circuit RL](https://arxiv.org/abs/2011.07665) emphasizes
   simulator-call efficiency, supporting plots whose x-axis is real SPICE
   evaluations and not merely inexpensive optimizer updates.

## Non-negotiable plotting rules

- Raw observations are always available; smoothing is optional, separately
  labeled, and never replaces raw data.
- Training, validation, and held-out targets use distinct visual encodings.
- Every aggregate across stochastic runs includes the number of seeds and an
  uncertainty interval. A one-seed result is explicitly marked descriptive.
- The primary x-axis for sample efficiency is cumulative real simulator
  evaluations. Wall-clock is a separate plot because cache hits and failure
  stages have different costs.
- Exact engineering PASS comes from `analysis.target_assessment`, never from
  PPO terminal reward.
- Missing historical fields remain unavailable; no curve is reconstructed from
  documentation prose.
- Figures carry run IDs, source paths, target split, algorithm/configuration
  version, seed count, and generation commit in machine-readable metadata.

## Plot catalogue

### A. Core plots for every run

| ID | Figure | Axes/encoding | Current data | Owner |
|---|---|---|---|---|
| `run_reward` | Reward progression | cumulative real evaluations × reward; raw points plus optional rolling median | Available for historical PPO | Integration |
| `run_strict_success` | Strict success progression | cumulative evaluations × cumulative strict-pass rate; target split encoded | Partially available; historical rows use logged same-target outcomes | Team B emits strict predicate result; Integration renders |
| `run_metric_margins` | Specification convergence | step × normalized margin, one facet per height/width/margin/power; zero threshold line | Missing from older PPO logs | Team B data; Integration plot |
| `run_parameter_trajectory` | Design movement | step × normalized parameter value, five aligned facets | Partial historical coverage | Team B data; Integration plot |
| `run_failure_stage` | Failure evolution | evaluation window × failure-stage fraction, stacked area | Aggregate historical stages available; windowed events needed | Team B data; Integration plot |
| `run_action_health` | Action/reachability diagnostics | requested/applied delta, boundary-hit and no-change rate per parameter | Not available | Team B data; Integration plot |
| `run_runtime` | Simulator cost | evaluation × wall-clock seconds, colored by last completed/failed stage; cumulative line | Missing from most historical PPO rows | Team B records timing/event identity; Integration plot |
| `run_ppo_health` | PPO optimization diagnostics | update × policy loss/value loss/entropy/approx-KL/clip fraction, aligned facets | Not available | Team B data; Integration plot |

### B. Checkpoint and generalization plots

| ID | Figure | Purpose | Data requirement | Owner |
|---|---|---|---|---|
| `checkpoint_success` | Initial/intermediate/final strict success with bootstrap interval | Show whether training improves frozen-policy performance | Matched starts, fixed targets, evaluation-only episodes, multiple seeds | Team B produces records; Integration aggregates/renders |
| `checkpoint_efficiency` | Empirical CDF of evaluations to first strict success | Show probability of solving within a SPICE budget | Per-case evaluation count; unsolved cases right-censored/marked | Shared contract; Integration statistics |
| `target_coverage` | Reached/unreached target-space small multiples | AutoCkt-style generalization view without hiding a fourth dimension | Stable target IDs, split labels, target vector, strict result | Team B records; Integration renders |
| `generalization_gap` | Train/validation/held-out success and reward with intervals | Detect overfitting such as the existing 90%→70% unseen-target regression | Frozen split and independent evaluation seeds | Team B records; Integration renders |

### C. Algorithm-comparison plots

These figures are enabled only after PPO, Random Search, and CEM have matched
budgets, targets, starts where applicable, fidelity, and independent seeds.

| ID | Figure | Statistical treatment | Owner |
|---|---|---|---|
| `algorithm_sample_efficiency` | Strict success within budget | Mean/IQM curve versus real evaluations with stratified bootstrap 95% interval | Integration |
| `algorithm_performance_profile` | Fraction of runs exceeding normalized-score thresholds | Performance profile with bootstrap confidence band | Integration |
| `algorithm_probability_improvement` | Probability each method improves over baseline | Pairwise probability with 95% bootstrap interval and 0.5 reference | Integration |
| `algorithm_aggregate_interval` | IQM, median, mean and optimality gap | Point-and-interval plot; IQM is headline, not best-seed score | Integration |
| `algorithm_cost` | Quality versus simulator calls and wall-clock | Separate panels; cache hits visibly distinguished | Integration |

No inferential comparison is generated for the current one-seed historical
trial. It remains a descriptive bar chart with its existing caveat.

### D. Circuit-quality and qualification plots

| ID | Figure | Current status | Owner |
|---|---|---|---|
| `design_pareto` | Eye height/width/margin against power, Pareto membership labeled | Existing evidence sufficient | Integration |
| `pvt_heatmap` | Process × voltage/temperature strict PASS matrix | Existing 27-point evidence sufficient; historical heatmap must distinguish simulator success from target re-score | Integration |
| `metric_target_comparison` | Selected-design measurements with target/official limits | Existing evidence sufficient | Integration |
| `failure_distribution` | Failure counts by stage and algorithm | Existing data partially sufficient | Integration |

## Team B handoff: data only, no plotting code

Team B must add these fields to its already-planned versioned PPO event schema:

- `run_id`, configuration ID, algorithm/state/action/reward/checkpoint versions
- training/validation/held-out split and stable target ID
- seed roles: training seed, tuning seed, final-evaluation seed
- cumulative real evaluation count and cache-hit flag
- pre/post state, raw available metrics, validity mask, failure stage
- strict target assessment and its per-metric normalized margins
- requested action, applied delta, clipping and no-change indicators
- parameters before and after the action
- reward total and components
- episode return, termination/truncation reason, steps to success
- policy loss, value loss, entropy, approximate KL, clip fraction and explained variance per update
- evaluation and stage timings when available

Events must be append-only JSONL and flushed after every real evaluation. Team
B also delivers a schema document and a small deterministic fixture. Team B
does not import Matplotlib or modify the web UI, statistics, PVT, baselines, or
plotting code.

## Integration implementation phases

### Phase P1 — renderer foundation

1. Add `matplotlib` as a runtime dependency.
2. Add `analysis/plot_renderer.py` using `Figure` plus `FigureCanvasSVG` /
   `FigureCanvasAgg`; do not use global pyplot state in the threaded server.
3. Define an immutable `PlotRequest` containing plot ID, evidence hash, format,
   style version, dimensions and accessibility description.
4. Return SVG by default and PNG for dense raster exports.
5. Cache rendered bytes by evidence hash + plot style version.

Acceptance: concurrent rendering is deterministic, has labeled axes/units, and
does not mutate Matplotlib global state.

### Phase P2 — migrate the seven existing graphs

Migrate reward, cumulative success, failure stages, optimizer comparison,
unseen-target comparison, design Pareto and PVT. Add threshold lines and honest
one-seed annotations. Retain the current JSON endpoint as inspectable source
data.

Acceptance: every existing graph has a Matplotlib equivalent and a source link;
the old handwritten renderer can then be removed.

### Phase P3 — per-run rendering

1. Add `results/web_ui_runs/<run_id>.events.jsonl` and atomically generated
   `<run_id>.evidence.json`.
2. Add `/api/evidence/<run_id>` and `/api/plot/<run_id>/<plot_id>.svg`.
3. Render partial runs from flushed events, including failures/timeouts.
4. Refresh charts only when the evidence hash changes, not on every status poll.

Acceptance: synthetic success, strict failure, timeout and malformed-event
fixtures each produce truthful partial/final plots.

### Phase P4 — multi-seed statistical plots

Implement stratified bootstrap intervals, IQM, performance profiles,
probability of improvement and optimality gap. Require minimum metadata and
matched protocols before enabling comparison plots. Separate tuning and test
seeds.

Acceptance: validated against hand-computable fixtures and invariant to input
row order. One-seed inputs produce a descriptive warning, not a confidence
interval.

### Phase P5 — UI and export polish

Add a run/project evidence selector, accessible figure descriptions, SVG/PNG
download, report-safe dimensions, loading/error states, and a provenance drawer.
An LLM may summarize the structured evidence but cannot choose data, smooth
curves, calculate verdicts, or suppress negative plots.

Acceptance: responsive at desktop/mobile widths, readable in light/dark export,
and all plots render offline without a CDN.

## Verification suite

- Unit tests for aggregation, bootstrap intervals, censoring, normalization,
  threshold direction and missing data.
- Golden-image or perceptual checks for representative SVG/PNG output; avoid
  raw SVG byte equality because generated element IDs can vary.
- HTTP content-type, cache validator, invalid run/plot ID and path-traversal
  tests.
- Threaded-render stress test.
- Clean-checkout test reading evidence/checkpoint from the artifact ZIP.
- Regression assertion that generating plots does not call SPICE, PPO inference,
  PPO update, or target-selection logic.

## Recommended delivery order

Start P1 and P2 immediately using existing evidence while Team B implements the
event contract. Merge Team B's fixture at the P3 boundary. Do not wait for new
RL training to replace the existing seven charts. P4 becomes authoritative only
after controlled multi-seed records exist; until then the UI must retain the
current descriptive caveats.

## Integration status (2026-09-07)

- P1 complete: `analysis.plot_renderer` uses figure-local Matplotlib SVG/PNG canvases and an evidence-hash/style cache.
- P2 complete for all seven archived-evidence charts: the UI loads rendered endpoints and offers SVG/PNG downloads.
- P3 foundation complete: Teammate B's typed hook is connected to append-only, flushed JSONL; strict assessments and per-run evidence/plot endpoints are implemented.
- P4 statistical primitives complete: IQM, bootstrap intervals, performance profiles, and probability of improvement are available. Inferential UI plots remain disabled until matched independent multi-seed real records exist.
- P5 partially complete: responsive figures, accessible alternative text, downloads, provenance, and offline rendering are present. A richer multi-run selector and report bundle remain presentation polish.

No absent historical field is reconstructed. PPO v1/v2 raw rewards remain separate because their reward scales differ.
