# NEBULA RL Subsystem — PPO Model-Improvement Study (PPO v2)

Scope: RL subsystem only (PPO state/action/reward/env/checkpointing/
logging, synthetic ablations, frozen-policy evaluation). Simulator, PVT
orchestration, RS/CEM, the end-to-end pipeline, the UI, graphs/dashboard,
LLM wrapper, physical circuit expansion, and final project claims are
explicitly out of scope and untouched.

Branch: `rl-ppo-v2-study` (based on `origin/main@649ce33` plus two
already-committed, in-scope RL commits from the prior session). Commits:
`da39d9d`, `8912c1b`, `113f156`, `d05f8f5`, `1268186`, `b9b3235`.

---

## 1. Audit (before any code change)

**Current PPO architecture** (`rl/ppo_agent.py`): 2-layer (64,64) tanh
MLP, separate policy/value trunks, 5 independent Discrete(3) categorical
heads. Standard PPO-literature hyperparameters (γ=0.99, GAE λ=0.95,
clip=0.2, entropy=0.01) — labeled `[UNVERIFIED-FROM-SOURCE]` in the code
itself; AutoCkt's own Ray 0.6.3 config never pinned these.

**State** (`rl/autockt_state.py`): 13-dim = `[4 signed relative-errors
(achieved, goal), 4 signed relative-errors (goal, reference), 5
normalized param indices]`. `reset()` fed this with an all-zero metrics
dict — the "artificial zeros" problem this study's Required change 1
addresses.

**Action** (`rl/autockt_action.py`): 5×Discrete(3), index deltas
`{-1,0,+2}` on a 21-point grid per parameter.

**Reward** (`rl/autockt_reward.py`): sum of negative signed relative
errors when unsatisfied, flat terminal bonus 10.0 once ≥ -0.02. **On any
failure, a flat `FAILURE_REWARD=-1.0` regardless of stage or distance —
confirmed, not hypothesized: docs/autockt-mapping.md sec 20's fair PPO
trial had every one of 20 episodes fail at `dc`, so `mean_episode_reward`
was exactly -1.0 across all 4 updates, zero variance.**

**Target sampling** (`rl/target_spec.py`,
`experiments/train_autockt.py::_build_target_pools`): the reported
`--target-mode mixed` run trains on exactly 2 fixed targets and validates
on 1 fixed target (their arithmetic midpoint).

**Horizon**: CLI default 6; reported runs used 4 or 1 (1 chosen for
RS/CEM comparability, not because it was believed optimal for PPO).

**Training budget**: mixed-target run = 6 updates × 6 episodes/update, 4
PPO epochs, minibatch 16.

**Identified bottlenecks**, ranked by expected benefit / cost (this
analysis predates and motivated the current spec's own Required changes
1-4, confirmed still accurate):

| Rank | Bottleneck | Cost to fix |
|---|---|---|
| 1 | Reward-floor collapse on failure (zero gradient signal) | Zero SPICE cost — pure reward-function change |
| 2 | Fabricated-zero reset state | Zero-to-low SPICE cost — an env-level fix |
| 3 | No validity/failure-stage information in state | Zero SPICE cost — the adapter already computes it, just discarded |

---

## 2. Hypothesis

**Primary hypothesis under test**: *A corrected initial state plus
validity-aware dense reward improves PPO's ability to move from
non-feasible starting regions without sacrificing trained-target
performance.*

Tested via 5 incrementally-nested, independently-attributable variants
(Section 5) rather than changing reward+state+actions+network+targets+
hyperparameters simultaneously.

---

## 3. PPO v1 compatibility report

**Unmodified**: `rl/autockt_state.py` (`build_state`, `STATE_DIM`),
`rl/autockt_reward.py::autockt_reward` (the exact function every
historical training log was produced with), `rl/ppo_agent.py` (network
architecture, hyperparameters), `rl/parameter_grid.py::ACTION_DELTAS`
(the default). Every existing `AutoCktReceiverEnv(...)` call site that
omits the new parameters (`evaluate_on_reset`, `state_schema`,
`use_reward_v2`, `action_deltas`, `max_total_evaluations`) gets
byte-for-byte identical behavior — verified by test
(`tests/test_rl_ppo_v2.py::EnvV2OptInTests::test_default_behavior_completely_unchanged`
and others).

**One real bug fixed, not a formulation change**: `AutoCktReceiverEnv.step()`
previously ignored `RLStep.truncated` (the adapter's own budget-exhaustion
signal, already computed by `simulator/rl_adapter.py` but silently
discarded) — meaning a shared adapter budget exhausted mid-training could
raise an uncaught `RuntimeError` from `ReceiverRLAdapter.step()`'s own
guard. Now truncates cleanly. This is a latent correctness bug in the
existing v1 code path being closed, not a behavior change to reward,
state, or action semantics.

**Existing checkpoints**: `results/autockt_mixed_target_confirmation_policy.pt`
(a bare `policy.state_dict()`) remains loadable via the existing pipeline's
`torch.load(path, weights_only=True)` call sites, unchanged —
`rl/checkpoint.py::export_inference_only` produces the identical format.

---

## 4. PPO v2 configuration surface

All additive, all opt-in, all default to exact v1 behavior:

| Parameter | Default | Values | Required change |
|---|---|---|---|
| `AutoCktReceiverEnv(evaluate_on_reset=...)` | `False` | bool | 1 |
| `AutoCktReceiverEnv(state_schema=...)` | `"v1"` | `"v1"`, `"v2"` | 2 |
| `AutoCktReceiverEnv(use_reward_v2=...)` | `False` | bool | 3 |
| `AutoCktReceiverEnv(action_deltas=...)` | `ACTION_DELTAS` (-1,0,+2) | any 3-tuple | 7 (optional) |
| `AutoCktReceiverEnv(max_total_evaluations=...)` | adapter's own budget | int | 4 |
| `train_autockt.py --reward-mode` | `terminal` | `terminal`, `graded` | (prior session — `graded_autockt_reward`, a lighter-weight predecessor to reward_v2, still available independently) |

State schema v2 adds (to v1's 13 dims): 1 validity flag, 2 early-stage
margin dims (`ctle_power_w`, `peaking_db` — both genuinely available even
on a dc/ac-stage failure), 11-way one-hot failure-stage encoding.
`STATE_DIM_V2 = STATE_DIM_V1 + 1 + 2 + 11 = 27`.

Reward v2 (`rl/autockt_reward_v2.py::RewardResult`): `total`,
`autockt_terminal_success` (v1's -0.02 tolerance), `strict_target_pass`
(exact, zero-tolerance boundary), `components`, `constraint_margins`,
`available_metrics`, `failure_stage`. Never interprets a missing metric
as zero; a simulator failure is scored strictly below every valid
design (`NO_INFORMATION_FLOOR_V2 = -20.0`); a strict pass gets a bounded
overshoot bonus (deliberately differentiates near-feasible designs,
unlike v1's own tested "all successes = 10.0" behavior).

---

## 5. Checkpoint documentation

`rl/checkpoint.py` — two deliberately distinct formats:

- **`save_full_checkpoint`/`load_full_checkpoint`**: policy + value-net
  weights, optimizer state, Python/NumPy/PyTorch RNG states, update/
  evaluation counters, hyperparameters, parameter-grid definition, target
  IDs, state/action/reward schema versions, git identity (via
  `simulator.provenance.git_identity`, reused read-only). Loaded with
  `weights_only=False`. `load_full_checkpoint` **rejects** a
  state_dim/num_heads/state_schema/reward_schema mismatch
  (`CheckpointIncompatibleError`) before touching any weights.
- **`export_inference_only`**: a bare `policy.state_dict()`, identical
  to the existing pipeline's expected format, `weights_only=True`-loadable.

---

## 6. Event schema (Required change 6)

`rl/events.py::PPOStepEvent`/`PPOUpdateEvent` — `TypedDict`, schema-
versioned (`EVENT_SCHEMA_VERSION=1`), `event_type`-discriminated. Fields:
run ID; step/episode/update; target ID and values; pre/post state;
metric-validity mask; failure stage; action choice; requested vs. applied
delta with a boundary-clip flag; parameters before/after; raw metrics;
reward components; strict PASS; log probability; value estimate; policy/
value losses; entropy; termination/truncation reason; evaluation count.

`rl/trainer.py::collect_rollout`/`train` gain an optional `on_event` hook
(default `None`) that emits these — **schema and emission only; durable
storage/visualization are explicitly another team's responsibility.**

---

## 7. Test results

408/408 SPICE-free tests passing (`python -m tests.run_fast_suite`), up
from 378 at the start of this study. New test files: `tests/
test_rl_ppo_v2.py` (38 — state v2, reward v2 property tests, env
opt-ins, budget-aware truncation, action deltas), `tests/
test_rl_checkpoint.py` (11), `tests/test_rl_events.py` (9), `tests/
test_rl_ppo_v2_ablation.py` (6), plus 4 in `tests/test_synthetic_benchmark.py`.

**Property tests required by the spec, all present**
(`RewardV2PropertyTests` in `tests/test_rl_ppo_v2.py`): correct direction
(larger/smaller-is-better), monotonicity in distance from target,
exact-boundary behavior (strict pass at the literal threshold, fails just
below it), missing-metric handling (never interpreted as zero),
failure ordering (no-information < graded-transient-failure < success),
PPO v1 unchanged.

---

## 8. Synthetic ablation records (Required change 7)

20 seeds × 5 variants, `experiments/rl_ppo_v2_ablation.py`, results in
`results/rl_ppo_v2_ablation_20seeds.jsonl` (gitignored per convention,
reproducible via the committed code — see commit `b9b3235`).

**Configuration** (identical across all variants/seeds except the
variant's own opt-in flags): starting point = grid index 0 on every
parameter (a **genuinely infeasible corner**, not grid-center — see the
methodological note below), 3 updates × 4 episodes/update, horizon 4,
`synthetic_evaluate_receiver_graded` (verified deterministic, pure-Python,
no SPICE), training target = trivial, held-out target = hard, 5 held-out
episodes per seed.

**Methodological finding recorded, not hidden**: the first version of
this ablation started every episode at grid-center. Grid-center
coincides almost exactly with the synthetic landscape's own "good"
center (both are each parameter's own midpoint) — every variant
succeeded trivially in 1 evaluation regardless of configuration,
producing zero differentiation and silently defeating the entire
hypothesis under test. This is the *exact same mistake*
`docs/autockt-mapping.md` sec 22 already diagnosed for the real
`VERIFIED_INITIAL_PARAMETERS` starting point. Fixed by using a genuinely
infeasible corner instead — recorded here so it isn't repeated.

**Results** (uniform criteria — see §"uniform metrics" below):

| Variant | Strict success (20 seeds) | Loose success (mean) | Win/Tie/Loss vs. v1 (loose, per-seed) | Boundary-hit rate | Distance travelled |
|---|---:|---:|---|---:|---:|
| v1_baseline | 0/20 | 0.230 | — | 0.594 | 12.17 |
| v2_corrected_reset | 0/20 | 0.234 | 3W / 17T / 0L | 0.593 | 12.18 |
| v3_validity_state | 0/20 | 0.258 | 13W / 2T / 5L | 0.586 | 12.61 |
| v4_reward_v2 | 0/20 | 0.260 | 15W / 0T / 5L | 0.579 | 12.70 |
| v5_symmetric_actions | 0/20 | 0.006 | **0W / 0T / 20L** | 0.625 | 7.27 |

**Uniform metrics**: "strict success" and "loose success" are both
recomputed post-hoc via the SAME `reward_v2()` call for every variant,
regardless of what that variant actually trained with — never each
variant's own native criterion, matching this project's established
practice (`analysis/fair_comparison.py`).

**Honest interpretation, no overclaiming**:
- **Strict target satisfaction never occurred for any variant at this
  budget** (0/20 everywhere) — the budget (12 episodes) is genuinely too
  small to reach strict target satisfaction from a fully infeasible
  start, for any configuration. This is a real limitation of the study's
  scale, not evidence any variant is equally good/bad at the strict
  criterion.
- On the looser (simulator-level) criterion, **v2 never regresses**
  (3W/17T/0L) — the safest, smallest improvement.
- **v3 and v4 show a real, per-seed-consistent improvement** (13/20 and
  15/20 seeds respectively beat baseline), not just an aggregate-mean
  artifact — but neither is universal (5 losses each out of 20).
- **v5 (symmetric actions) is uniformly, consistently worse — 0/20
  seeds beat baseline.** The smaller maximum step size (`{-1,0,+1}` vs.
  `{-1,0,+2}`) plausibly explains this directly: `distance_travelled`
  dropped from ~12.2-12.7 to 7.27, meaning the policy simply cannot cover
  as much ground in the same step budget. This is a clear negative
  result — symmetric actions are **not** pursued further.
- **v4's raw `mean_reward` (-4.04) looks much worse than v1's (-0.89) —
  this is a reward-SCALE artifact, not a performance regression.**
  `reward_v2`'s `NO_INFORMATION_FLOOR_V2=-20.0` is far more negative than
  v1's flat `FAILURE_REWARD=-1.0`; the two rewards are not on comparable
  scales. The uniform strict/loose success-rate metrics above are the
  only valid cross-variant comparison for this study.
- **Held-out generalization was 0.0 for every variant** — a direct
  consequence of the strict-success floor effect above (training itself
  never reaches strict success, so zero-shot generalization to a harder
  held-out target cannot either, at this budget).

---

## 9. Recommended variant

**v4_reward_v2** (corrected reset + validity/failure state + reward_v2
stacked — literally the hypothesis under test) shows the most consistent
per-seed improvement over baseline (15/20 wins, 0 ties, 5 losses) among
the tested variants, and directly exercises the full mechanism this
study's hypothesis names. **v3_validity_state** is a close, slightly
more conservative second choice (13/20 wins) if the reward-scale
complexity of v4 is undesirable for a first real-SPICE trial.
**v5_symmetric_actions is explicitly NOT recommended** — a consistent,
20/20-seed negative result.

This is a **synthetic-only recommendation**. Per the spec's own gate,
promotion to real SPICE requires: improvement across multiple synthetic
seeds (✅ shown above), survival of warm-start removal (✅ — this
ablation never used a warm start; the infeasible-corner setup is the
no-warm-start case by construction), held-out performance not regressing
materially (⚠️ — held-out was 0.0 for every variant at this budget; this
criterion cannot yet be evaluated meaningfully and needs a larger-budget
follow-up before real-SPICE promotion), and the full event record being
available (✅ — `on_event`, Required change 6).

---

## 10. Proposed real-SPICE experiment (NOT launched — proposal only)

Per the explicit gate ("Do not launch a large real-SPICE training run
automatically"), this section is a **proposal**, submitted for separate
authorization, not an executed run.

- **Exact code commit**: `b9b3235` (this branch, `rl-ppo-v2-study`)
- **PPO configuration**: `v4_reward_v2` — `evaluate_on_reset=True,
  state_schema="v2", use_reward_v2=True, action_deltas=ACTION_DELTAS`
  (v1's own asymmetric deltas — v5 is excluded per its negative result)
- **Target IDs**: training = `TargetSpec.from_existing_thresholds()`
  (trivial); held-out = `TargetSpec.from_hard_target()` (hard) — both
  already real-SPICE-verified-achievable/discriminating targets, no new
  target invented
- **Starting-state distribution**: `initial_indices_source="grid-center"`,
  `randomize_initial_state=True` — the same no-warm-start setup as
  `docs/autockt-mapping.md` sec 20's fair trial, for direct comparability
  to the existing 0/20 real-SPICE result
- **Seeds**: a single seed first (matching this project's own established
  fallback convention, "otherwise 1 seed first"), e.g. seed=123
- **Evaluation budget**: n=20 real evaluations (matching sec 20's exact
  budget for direct comparability) — `--updates 4 --episodes-per-update 5
  --horizon 1` or a slightly longer horizon (2-4) given `evaluate_on_reset`
  now spends 1 extra real evaluation per episode (see next line)
- **Fidelity**: TRAINING (matching every other PPO training run to date)
- **Checkpoint evaluations**: 3 initial + 3 final (matching the existing
  `--checkpoint-eval-episodes 3` convention)
- **Expected runtime**: `evaluate_on_reset` adds exactly 1 real evaluation
  per episode (measured cost, not estimated — see
  `tests/test_rl_ppo_v2.py::EnvV2OptInTests::test_evaluate_on_reset_spends_exactly_one_evaluation`).
  For n=20 episodes at horizon 1 plus corrected reset: ~20 extra
  evaluations beyond sec 20's own 20, i.e. roughly double sec 20's real
  wall-clock (530.8s) — **estimated ~15-20 minutes**, not a large
  training run.

**Promotion criteria** (from the spec, restated as the actual bar this
proposal must clear before being called anything more than a pilot): a
single small real-SPICE run **remains a pilot, not evidence of
superiority**, regardless of outcome.

---

## 11. Honest limitations

1. **Strict target satisfaction was never reached by any synthetic
   variant at this study's budget** — the comparison rests entirely on
   the looser, simulator-level success criterion. A larger-budget
   synthetic follow-up (more updates/episodes, still SPICE-free and
   cheap) would be needed before claiming any variant reaches genuine
   target satisfaction reliably.
2. **Held-out generalization could not be meaningfully evaluated** in
   this study (0.0 for every variant, a floor effect inherited from #1).
3. **Synthetic results establish software behavior only** — they do not
   and cannot prove circuit improvement (per `rl/synthetic_benchmark.py`'s
   own module docstring, unchanged, and this document's own framing).
4. **v4's reward-scale difference from v1 makes raw reward values
   incomparable** across the v1-3 vs. v4-5 boundary — only the uniform
   strict/loose success-rate metrics are valid for cross-variant
   comparison.
5. **No real-SPICE PPO v2 run has been made** — Section 10 is a proposal
   only, per the explicit gate.
6. **This study does not claim PPO beats Random Search or CEM** — no
   evidence here bears on that question at all; the existing evidence
   (`docs/autockt-mapping.md` sec 19/20) is unchanged and remains the
   governing record.
7. `rl/autockt_reward_v2.py`'s peaking component references the AC
   gate's 3-12 dB band as a fixed reference, since `TargetSpec` has no
   peaking field — this is a disclosed, deliberate design choice
   (peaking is banded, not monotone), not an oversight, matching
   `rl/target_spec.py`'s own existing documented reasoning.
