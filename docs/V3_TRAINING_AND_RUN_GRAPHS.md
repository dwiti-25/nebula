# PPO v3: implementation and training handoff

Implemented 2026-09-11. This is the current implementation guide; the earlier
PPO_V3_AND_RUN_GRAPH_PLAN.md remains the design discussion. Manim is not used.

## What is connected

- Eight-head PPO training and saved-policy inference, available in the local UI.
- Ordered parameters: RLOAD, RDEG, CDEG, ITAIL, behavioral DFE tap, matched MOS
  width, length and integer multiplier. Both CTLE transistors share MOS geometry.
- Existing hierarchical SPICE evaluation, reference/synthetic channel selection,
  exact evaluation caching, nominal filtering, optional PVT and final noise/HD3,
  schematic export and final specification report all retain the chosen geometry/channel.
- Random Search and CEM accept `--rl-version v3`. CEM also accepts channel and
  grid settings. The generalization and matched-checkpoint tools accept v3 and
  `--channel` / `--channel-ports`.
- The UI supports **Generate and verify circuits** and **Train a new PPO checkpoint**.
  Training saves new policy/full-checkpoint files, discoverable in its checkpoint menu.
- Per-run execution graph: actual evaluations, reset calls, policy actions,
  stages, cache origins, failures/skipped stages, selection, verification and artifacts.
  Expand nodes in the UI or download JSON/SVG. The overview is limited to 180 nodes;
  the JSON retains the complete record. Graphs survive ordinary pipeline exceptions.
- Per-run plots: reward, strict success, PPO losses/entropy, measured power,
  peaking, eye height/width, noise/HD3 when available, and evaluation latency.
  Missing values are gaps. Historical plots remain separate from the current run.
- Selected real-SPICE designs now produce a two-panel eye artifact: a conventional
  transistor-level CTLE waveform eye and a behavioral one-tap DFE sampling-eye.
  SVG, PNG and the bounded source JSON are downloadable from the UI. This is one
  post-selection evaluation only; PPO training does not capture or render eyes.

## Frozen v3 contract

State dimension: **52**. Action heads: **8**, each with index deltas `[-1, 0, +2]`.
The state includes target errors/references, eight normalized indices, per-metric
values and validity, failure-stage identity, channel response at 1.25/2.5/5 GHz
and delay, partial MOS channel area and an explicit false total-area-valid flag.

Default grid resolution is 21, with legacy linear spacing unchanged. Exact arrays,
not just a point count, are saved. MOS W/L are rounded to 0.001 µm; W includes
the historical 10 µm anchor. M has exactly the integers 1–16. These are bounded
sizing grids, **not proof of manufacturability or DRC closure**.

Reward v3 requires successful evaluation, every target, peaking within 3–12 dB,
and zero measured decision errors for termination. A small regularizer of weight
0.02 favors smaller **partial MOS channel area**; it never establishes compliance
with the total-area budget. v1/v2 state dimensions and five-head policies are retained.

V3 inference exports carry metadata and load with `weights_only=True`. A bare
v1/v2 state dictionary cannot be used as v3. Inference restores the checkpoint's
exact physical grids. Selecting another channel is a generalization experiment;
the graph records both the training contract and actual evaluation contract.

Full checkpoints save policy/value networks, optimizer, Python/NumPy/Torch RNG,
environment target/reset RNG, update/evaluation counters, training settings,
exact grids, channel checksum/map and toolchain identity. Resume requires the
same runtime/training contract; `--updates` means additional updates and
`--max-evaluations` is the total budget including work before resume.
The in-memory cache is not serialized; resumed runs may spend more simulator
time on a repeated design. Training and frozen checkpoint evaluation have separate budgets.

## Start with a smoke run

From the repository directory, use its existing environment (do not reinstall
PyTorch for each session):

```powershell
.\.venv\Scripts\python.exe -m experiments.train_autockt `
  --rl-version v3 --backend synthetic --updates 1 `
  --episodes-per-update 2 --horizon 2 --max-evaluations 6 `
  --evaluation-cache --output results/v3_smoke_train.jsonl `
  --save-final-policy results/ppo_v3_smoke_policy.pt `
  --save-full-checkpoint results/ppo_v3_smoke_full.pt
```

Synthetic training is a software test, not circuit optimization evidence.
Its eight-dimensional algebraic landscape is explicitly separate from SPICE.

## Real training

Ensure `SKY130_MODEL_LIBRARY` and `NGSPICE_EXECUTABLE` point to working local
installations (see README setup). Start small, inspect failure stages, then
increase the budget; no convergence or performance improvement is assumed.

```powershell
.\.venv\Scripts\python.exe -m experiments.train_autockt `
  --rl-version v3 --backend real --seed 101 `
  --channel channels/ieee802_ibm_20db_thru.s4p --channel-ports 1 3 2 4 `
  --updates 10 --episodes-per-update 2 --horizon 4 --max-evaluations 100 `
  --evaluation-cache --randomize-initial-state `
  --output results/v3_seed101.jsonl `
  --save-final-policy results/ppo_v3_seed101_policy.pt `
  --save-full-checkpoint results/ppo_v3_seed101_full.pt
```

To resume, keep the same seed, target, horizon, grid and training settings; add
`--resume results/ppo_v3_seed101_full.pt`, increase the total evaluation budget,
and use new output filenames. Do not resume across implementation changes without
reviewing compatibility. Synthetic-to-real transfer is inference, not exact resume.

In the UI select v3, the saved policy and its channel, then generate circuits.
Use PVT/final HD3/noise afterward; training itself uses nominal TRAINING fidelity.
UI training uses preset targets; custom target-pool studies use the CLI.

`run_autockt_pipeline --eye-diagram-output results/my_eye.svg` requests the
selected-design eye artifacts. When FINAL HD3/noise refinement also succeeds,
the eye data are reused from that same 1,024-bit evaluation. Otherwise a separate
128-bit TRAINING-fidelity TT evaluation is made. A failed pre-transient stage is
reported as `not_produced`; missing waveform data are never fabricated.

## What this does not complete

- Long, multi-seed training and matched-budget evidence of improvement are still needed.
- The DFE is behavioral, not a transistor-level implemented DFE.
- Total layout area is unavailable: passive area, diffusion, routing and DFE area
  are not modeled. The partial MOS metric must not be called minimum total area.
- A reference-channel result is not PCI-SIG compliance certification.
- A 27-condition sweep is not the full original five-corner, 60-condition matrix.
- PVT verification is downstream of nominal PPO training; a learned PVT-robust
  policy is not established merely by enabling that verification.
- Surrogate-assisted and multi-actor RL remain separate research work, not PPO v3 features.

## Verification

`python -m unittest tests.test_v3_integration -q` exercises eight-parameter
propagation, strict rewards, safe checkpoint compatibility, cache provenance,
SVG/plot rendering, training/resume and UI command construction.
Use `python -m unittest discover -s tests -q` for the regression suite.

A short real-SPICE training smoke reached nominal v3 feasibility with W=15.357 µm,
L=0.15 µm and M=1 on the IEEE/IBM reference channel. This is integration evidence,
not a converged checkpoint or proof that all random starts will succeed.

Final validation: 582 regression tests ran successfully (558 passed, 24 skipped).
The eight v3 integration tests also passed in a separate final run. A live UI
training request completed and its JSON/SVG graph, per-run charts and saved policy
were served successfully. Local smoke artifacts are kept in `.local-validation/v3/`
and are not production-trained checkpoints.
