# NEBULA RL Runtime / Simulation-Efficiency Study

Branch `rl-ppo-v2-study`, built on top of the completed PPO v2 study
(`docs/RL_PPO_V2_STUDY.md`), which this work does not modify or
invalidate. Scope: RL/environment/evaluation orchestration only. Does
not touch the DFE, simulator, PVT orchestration, RS/CEM, UI, LLM
wrapper, surrogate/BO, or another RL algorithm, and does not expand
the circuit topology.

## Hypothesis (defined before measuring)

NEBULA can reduce end-to-end design time primarily by reducing
unnecessary/duplicate expensive circuit evaluations and/or the number
of evaluations required to reach a useful/feasible design, while
preserving design quality and PPO v1 compatibility. Two objectives:
(A) wall-clock efficiency, (B) simulation/sample efficiency — B is the
one that matters for real deployment, since the final DFE may change
per-evaluation runtime; this study does not optimize for Python
execution time as an end in itself.

## Pre-checks (confirmed, not assumed)

- Branch: `rl-ppo-v2-study`. HEAD at the time of writing: `55dc37b`.
- `origin/main` baseline: `649ce33` ("Add Phase 2 experimental
  artifact package for Person B") — confirmed via `git log -1
  origin/main`.
- Person B's files (`simulator/rl_adapter.py`, `simulator/receiver.py`,
  `simulator/cache.py`) confirmed byte-identical to `origin/main` via
  `git diff --stat`: zero output, i.e. zero diff lines, for all three,
  both before this work started and again at the end.
- PPO v1 files (`rl/autockt_state.py`, `rl/ppo_agent.py`,
  `rl/parameter_grid.py`) confirmed byte-identical to `origin/main`:
  zero diff lines. `rl/autockt_reward.py` shows +70/−0 vs
  `origin/main` (pure addition from the earlier PPO v2 study; the
  original `autockt_reward` function itself is untouched).
- No checkpoints were overwritten; no large real-SPICE run was
  launched at any point in this work.

## Phase 1 — Profile (synthetic backend, cheapest reliable setup first)

Methodology: a read-only `cProfile` harness (zero modification to any
RL/env/reward/agent source) run over the exact `v4_reward_v2`
configuration (`evaluate_on_reset=True, state_schema="v2",
use_reward_v2=True, action_deltas=ACTION_DELTAS`), 5 updates × 5
episodes/update, horizon 4, synthetic evaluator
(`synthetic_evaluate_receiver_graded`).

Measured (one representative profiling run, 100 step events, 125
evaluator calls, 25 episodes, total wall-clock 0.386s):

| bottleneck | % of runtime | absolute time | eval count | optimization opportunity | risk to correctness |
|---|---|---|---|---|---|
| PPO `update()` (backward + optimizer step) | ~62% (0.240s of 0.386s cumulative) | 0.240s over 5 updates | n/a (not per-eval) | none identified — this is PyTorch's own backward pass, not our code | high if touched — is the learning algorithm itself |
| `torch.autograd` backward | ~39% (0.150s) | 0.150s over 40 backward calls | n/a | none — third-party | high |
| `PPOAgent.act()` (rollout action sampling) | ~30% (0.116s) | 0.116s over 130 calls | 125 | none identified at this scale | medium — touches policy sampling |
| `env.step()` Python-side overhead | ~5% (0.020s) | 0.020s over 100 calls, ~0.2ms/call | 100 | negligible in absolute terms | low |
| synthetic evaluator body itself | not separately visible above noise floor | ~microseconds/call | 125 | none — already the cheapest possible stand-in | n/a |
| duplicate parameter-vector evaluations | n/a (not a time bucket, a count) | — | **28/100 (28%)** | **evaluation caching** | low — see Phase 2 |

**Conclusion, stated explicitly rather than assumed**: in synthetic
mode, PPO's own `update()`/`backward()`/`act()` calls dominate total
wall-clock (roughly 60–100%+ combined across overlapping frames), and
`env.step()`'s own Python overhead and the synthetic evaluator body are
both small by comparison (~5% and sub-noise-floor respectively). This
is **not representative of real-SPICE deployment**: per this project's
own prior historical measurements (elsewhere in this repo, not
re-measured here per the instruction to use the cheapest reliable
setup first), a single real-SPICE evaluation costs roughly 15–65s at
TRAINING fidelity and up to ~300s at FINAL fidelity — 4–6 orders of
magnitude more than the ~0.2ms `env.step()` overhead measured above.
Under real deployment, evaluation **count**, not Python speed, is the
lever that matters. The one concrete, actionable count-level finding
from this profiling run — a 28% duplicate-evaluation rate — is the
basis for Phase 2.

## Phase 2 — Redundant evaluations

Investigated the listed redundancy patterns against the profiling
run's event stream. The dominant, measured pattern: **identical
parameter vectors evaluated more than once within the same run**
(28/100 steps in the Phase 1 run; 9.58% of calls in the larger Phase 7
20-seed study — see below for why the rate differs).

`simulator/cache.py::EvaluationCache` already exists (disk-based,
content-addressed, keyed by a provenance-aware `stable_fingerprint`)
and `simulator/receiver.py::evaluate_receiver` already accepts an
optional `cache=` parameter — but `simulator/rl_adapter.py`, confirmed
by reading (not just grepping) `ReceiverRLAdapter.__init__`/`.step()`
in full, never wires it through: `step()` calls
`self.evaluator(parameters, self.conditions, self.fidelity,
**self.evaluator_kwargs)` with no cache argument anywhere. This task
forbids modifying `simulator/rl_adapter.py` or `simulator/receiver.py`,
so the fix could not reuse that mechanism directly.

**Design implemented**: `rl/evaluation_cache.py`, a caching wrapper
built entirely in `rl/`, using `ReceiverRLAdapter`'s existing
`evaluator=` constructor parameter — the same extension point
`synthetic_evaluate_receiver` already uses. Zero changes to any
`simulator/` file. Opt-in: nothing is wired into
`AutoCktReceiverEnv`/`ReceiverRLAdapter` construction by default;
callers choose to wrap their evaluator or not.

**Exact deterministic cache key** (per the explicit instruction, "do
not assume caching is safe"): every field of `ReceiverParameters` +
every field of `SimulationConditions` (including `process_corner`) +
`fidelity` + the `evaluator_kwargs` the call was made with. This
covers everything `evaluate_receiver`'s own call signature exposes as
capable of changing its result. The RL **target** is deliberately
excluded: `evaluate_receiver` never receives a target argument at all
(target only affects reward computation, which happens *after* the
cached raw evaluation is returned), so including it would only reduce
the hit rate without adding safety — verified by reading
`evaluate_receiver`'s signature directly, not assumed.

Tested (14 tests, `tests/test_evaluation_cache.py`): identical inputs
→ identical key; any differing field (parameters, conditions,
process corner, fidelity, kwargs) → different key and a genuine cache
miss; repeated identical calls hit the cache and the underlying
evaluator is invoked exactly once; the wrapped callable is drop-in
compatible with `ReceiverRLAdapter`'s call convention.

**Measured hit rate and memory** (20-seed Phase 7 study, see below):
mean cache hit rate **9.58%** (min 0%, max 16.7% across seeds), mean
**54.25 unique cache entries** per run, mean **~6.5KB** approximate
shallow memory footprint per run (`sys.getsizeof`-based, a bounded
proxy, not a precise deep-memory measurement — disclosed as such).

## Phase 3 — Hierarchical fidelity usage

Investigated whether the RL loop underuses the simulator's
SCREENING → TRAINING → CANDIDATE → FINAL fidelity hierarchy, without
redesigning the simulator or adding HD3/noise to training state (both
explicitly forbidden).

Finding: `ReceiverRLAdapter.fidelity` is fixed for the adapter's
entire lifetime (constructor argument, default `TRAINING`), and
neither `experiments/rl_ppo_v2_ablation.py` nor
`experiments/rl_runtime_efficiency_study.py` overrides it — RL
training runs exclusively at `TRAINING` fidelity. Reading
`simulator/receiver.py` directly: the channel/DFE stage — which
produces every metric `reward_v2`/state-v2 depend on
(`dfe_locked_phase_eye_height_v`, `dfe_error_rate`, etc.) — is gated
behind `if fidelity >= EvaluationFidelity.TRAINING` (line 580); the
cheaper `SCREENING` level does not run it at all. `CANDIDATE`/`FINAL`
are reserved for downstream confirmation outside RL scope (PVT sweep,
final spec generation).

**Conclusion**: RL training already sits at the minimum fidelity
level capable of producing the metrics its own reward function
requires. Using `SCREENING` would silently degrade or remove the
reward signal (an algorithmic change, forbidden by Phase 6); using
`CANDIDATE`/`FINAL` would only add cost with no RL-training benefit.
No safe change was found here — this is a documented "investigated,
no action" outcome, not a gap.

## Phase 4 — Action/environment efficiency (instrumentation only, no action-space change)

Using the existing `on_event`/`PPOStepEvent` infrastructure (already
built, no new code needed to observe it), computed on the Phase 7
20-seed baseline runs (48 steps/run mean):

| metric | value |
|---|---|
| no-op actions (all-zero applied delta) | mean 1.5/run — **3.1% of steps** |
| directional reversals (a parameter's delta sign flips vs. the prior step, same episode) | mean 21.8/run — **45.4% of steps** |
| boundary-clipped actions | mean **57.9%** of steps |
| distance travelled (normalized-space, summed per run) | mean 12.7 |
| evaluations per reward improvement | mean 4.57 (all 20 seeds had ≥1 improvement) |

No-op actions are a small fraction of waste (3.1%). Boundary-clipping
and directional reversal are both large and worth flagging for a
**future, separately-scoped** action-space investigation — but this
task explicitly instructs "do not change the action space yet, first
instrument," and the high boundary-clip rate is partly explained by
the deliberately infeasible starting corner (grid index 0 on every
parameter, i.e. already at the lower boundary) used throughout the
PPO v2 study and reused here for identical-everything comparability.
No action-space change was made.

## Phase 5 — Safe optimizations implemented

Only one change met every listed bar (correct, measurable,
reproducible, PPO-v1-compatible, v4-compatible, DFE-independent,
reversible/versioned): the opt-in evaluation cache (Phase 2). It is
additive, off by default, contained entirely in a new `rl/` module,
and was proven not to be a hidden algorithmic change by the Phase 7
trajectory-equality check (below). No other change (no arbitrary
micro-optimizations) was made — profiling in Phase 1 did not surface
another candidate that both mattered and was safe within this task's
scope.

## Phase 6 — Frozen baseline confirmation

`experiments/rl_runtime_efficiency_study.py` imports
`VARIANTS["v4_reward_v2"]`, `UPDATES`, `EPISODES_PER_UPDATE`,
`HORIZON`, `TRAIN_TARGET`, `GRID_POINTS`, `GRID_SPACING`,
`_infeasible_corner_indices`, `_distance_travelled`, and
`_uniform_strict_pass` directly from `experiments/rl_ppo_v2_ablation.py`
rather than redefining them — "identical everything" is structural,
not a claim. PPO v1 files remain at zero diff (confirmed above). The
evaluation cache changes **when** the expensive evaluator body runs,
never **what** it returns; this is not an algorithmic change, and is
verified empirically (not just argued) in Phase 7.

## Phase 7 — Controlled runtime experiment (20 seeds, synthetic)

`experiments/rl_runtime_efficiency_study.py`: for each of 20 seeds,
runs baseline `v4_reward_v2` (uncached evaluator) and optimized
`v4_reward_v2` (same evaluator wrapped by `rl/evaluation_cache.py`)
under identical config/targets/starting state/seeds/network init/
horizon/budget/stopping criteria. A correctness assertion in
`run_pair()` requires the two runs' action and reward trajectories to
be bit-for-bit identical — this held for all 20 seeds
(`all_trajectories_identical: true`).

A measurement bug was caught and fixed before trusting any wall-clock
number: an unwarmed first in-process run measured ~5–6x slower
regardless of which configuration ran first (confirmed by swapping
run order twice) — a PyTorch/interpreter/CPU warm-up artifact, not a
caching effect. A single discarded warm-up pass before any timed run
removed it (both configurations then agreed within ~2% on the same
uncached evaluator run twice). Without this fix, the study would have
reported a fabricated ~79% synthetic wall-clock "speedup."

**Results** (`results/rl_runtime_efficiency_20seeds.jsonl`, gitignored
per project convention; real, measured, not projected):

| metric | value |
|---|---|
| n_seeds | 20 |
| all_trajectories_identical | true |
| total baseline expensive evaluator calls | 1200 |
| total optimized expensive evaluator calls | 1085 |
| **evaluation reduction** | **9.58%** |
| total baseline wall-clock (synthetic) | 1.192s |
| total optimized wall-clock (synthetic) | 1.176s |
| **wall-clock reduction (synthetic)** | **1.41%** |
| baseline mean strict success rate | 0.0 |
| optimized mean strict success rate | 0.0 (identical) |
| baseline mean reward | −4.0358 |
| optimized mean reward | −4.0358 (bit-identical) |
| mean cache hit rate | 9.58% (min 0%, max 16.7% across seeds) |
| mean cache unique entries | 54.25/run |
| mean cache approx. memory | ~6.5KB/run |

Design quality is unchanged (bit-identical reward and success rate
between configurations, by construction of the trajectory-equality
invariant, not by coincidence). The synthetic wall-clock reduction is
small, as Phase 1 predicted — the synthetic evaluator body is cheap,
so removing 9.58% of its calls barely moves total wall-clock when
PPO's own `update()`/`backward()` dominates. The evaluation-count
reduction is the number that transfers to real-SPICE deployment.

20 seeds was cheap enough to run in full (~1.2s per configuration,
~25s total including the two ablation-study prerequisite imports) —
no smaller pilot-then-propose step was needed for the synthetic study
itself.

## Phase 8 — Real-SPICE pilot (proposed, NOT executed)

Per explicit instruction, no real-SPICE run was launched. This is a
proposal only.

**Justification for proposing it**: the synthetic study establishes
(a) the caching mechanism is correctness-preserving (bit-identical
trajectories, all 20 seeds) and (b) it measurably reduces evaluation
count (9.58%) without any success-rate or reward regression. Neither
of these facts requires real SPICE to establish, and both are
prerequisites before spending real-SPICE budget on the same question.

**Proposed pilot specification** (to be run only on explicit
approval):

- Reuse `experiments/rl_runtime_efficiency_study.py` unchanged in
  structure, swapping `synthetic_evaluate_receiver_graded` for
  `simulator.receiver.evaluate_receiver` as the wrapped evaluator.
- Seeds: 3–5 (real-SPICE cost per call is 15–65s at `TRAINING`
  fidelity per historical figures; even 5 seeds × ~48 steps/run ×
  ~2 configs is on the order of several hours of wall-clock — a
  genuine budget decision, not a default to make unilaterally).
- Same frozen `v4_reward_v2` config, same infeasible-corner starting
  point, same target, same horizon/budget as the synthetic study.
- Same trajectory-equality assertion carried over unchanged — a real
  hardware/floating-point nondeterminism failure of that assertion
  (e.g. from ngspice solver noise) would itself be a valid and
  important finding, not a bug to suppress.
- Report the same metric set as Phase 7, plus real wall-clock time
  per evaluation (currently unmeasured directly by this study).
- **Explicitly labeled provisional**: real-SPICE per-evaluation
  runtime figures cited or measured anywhere in this study are
  provisional with respect to the final transistor-level DFE circuit,
  which is still under development by another team member.
- **Explicitly not evidence of PPO superiority**: this pilot, if run,
  validates runtime behavior only. It is not a comparison against
  Random Search or CEM, and must not be reported as one.

## Phase 9 — What this study does and does not claim

**Valid conclusions supported by the evidence above**:
- The RL layer can reduce the number of expensive evaluator calls by
  a measured 9.58% (synthetic, 20 seeds) via exact-key caching, with
  zero change to PPO's decisions, rewards, or success rate.
- Synthetic-mode wall-clock savings from this specific change are
  small (1.41%) because the synthetic evaluator body is cheap; the
  dominant synthetic-mode cost is PPO's own training math, not
  evaluation.
- RL training already uses the minimum simulator fidelity level
  capable of producing its own reward signal — no fidelity-hierarchy
  change was safe or available.

**Explicitly NOT claimed**:
- Not a claim that PPO is globally optimal.
- Not a claim that PPO beats Random Search or CEM (this study contains
  no RS/CEM comparison at all).
- Not a claim about final transistor-level circuit speed or quality.
- Not a claim about the final DFE's real-SPICE runtime — current
  real-SPICE runtime figures cited here are historical/provisional,
  not re-measured in this study.
- Not a claim that a runtime improvement implies a better circuit —
  design quality here is proven *unchanged* (bit-identical), not
  improved.

## Required deliverables

- `rl/evaluation_cache.py` — opt-in evaluator-level cache (Phase 2).
- `tests/test_evaluation_cache.py` — 14 tests.
- `experiments/rl_runtime_efficiency_study.py` — Phase 7 controlled
  comparison harness, reusable for the Phase 8 pilot.
- `tests/test_rl_runtime_efficiency_study.py` — 8 tests, including the
  bit-for-bit trajectory-equality correctness check.
- `results/rl_runtime_efficiency_20seeds.jsonl` — raw 20-seed results
  (gitignored per project convention; regenerable via
  `python -m experiments.rl_runtime_efficiency_study --seeds 20
  --output <path>`).
- This document.

**Additional required reporting**:
- Exact commit (this document's own commit, immediately following):
  see `git log -1`.
- `git status` at completion: only files listed above added/modified
  by this work; `experiments/train_rl.py` and `.claude/settings.json`
  carry pre-existing modifications from before this task began and
  were not touched by it.
- Files changed: `rl/evaluation_cache.py` (new),
  `tests/test_evaluation_cache.py` (new),
  `experiments/rl_runtime_efficiency_study.py` (new),
  `tests/test_rl_runtime_efficiency_study.py` (new),
  `tests/run_fast_suite.py` (2-line addition, registers the two new
  test modules), this document (new).
- Files explicitly NOT changed: every `simulator/` file, every
  PPO v1 file (`rl/autockt_state.py`, `rl/ppo_agent.py`,
  `rl/parameter_grid.py`), `rl/autockt_env.py`,
  `experiments/rl_ppo_v2_ablation.py`, `experiments/run_autockt_pipeline.py`,
  `experiments/web_ui.py`, `experiments/pvt_sweep.py`,
  `experiments/receiver_search.py`, `experiments/train_cem.py`.
- Test count: 430/430 passing (`python -m tests.run_fast_suite`), up
  from 408 at the start of the PPO v2 study and 422 before this
  runtime work.
- Real-SPICE: not run at any point in this task.
- Post-DFE-integration validity: the caching mechanism's correctness
  does not depend on any property of the DFE circuit — it caches
  whatever `evaluate_receiver` (or any drop-in evaluator) returns for
  a given exact input, unconditionally. It remains valid unchanged
  once the final DFE lands. Its *hit rate* may change (a different
  DFE could change how often the policy revisits identical parameter
  vectors), but that is a quantity to re-measure post-integration, not
  a correctness risk.

## Required verdict

```
RUNTIME STATUS:
- baseline runtime: 1.192s total wall-clock, synthetic, 20 seeds (60 evaluator calls/seed)
- optimized runtime: 1.176s total wall-clock, synthetic, 20 seeds (same 60 calls/seed, cached)
- runtime reduction: 1.41% (synthetic; small because the synthetic evaluator body is cheap and PPO's own training math dominates synthetic-mode wall-clock -- see Phase 1)
- baseline evaluations: 1200 expensive evaluator calls (20 seeds x 60/seed)
- optimized evaluations: 1085 expensive evaluator calls (20 seeds; mean hit rate 9.58%)
- evaluation reduction: 9.58%
- design-quality change: none (bit-identical reward and strict-success-rate between baseline and optimized, all 20 seeds; enforced by an explicit trajectory-equality assertion, not just observed)
- recommended optimization: adopt rl/evaluation_cache.py as opt-in for RL training runs (both synthetic ablations and any future real-SPICE training), since it is proven correctness-preserving and its evaluation-count reduction is the metric expected to matter most once real SPICE (15-300s/call) replaces the synthetic evaluator
- whether real-SPICE pilot is justified: yes, a SMALL pilot (3-5 seeds) per the Phase 8 specification above -- not yet run, pending approval and real-SPICE compute budget; must be labeled provisional pending the final DFE and must not be presented as evidence against Random Search or CEM
```
