# Nebula

Nebula is an automated analog-design framework for a PCIe Gen-2 receiver
equalizer. This branch integrates the parameterized SKY130 CTLE with a robust,
RL-independent Python-to-ngspice evaluation pipeline.

The current optimization scope is the Stage 1 schematic model: a reusable
one-stage CTLE, a file-driven differential channel, and a behavioral one-tap
DFE. MOS dimensions and the physical DFE implementation are deliberately left
for Stage 2.

```text
parameters -> DC -> AC -> 32-bit CTLE diagnostic -> channel validation
           -> staged receiver transient -> noise/HD3 for candidates
           -> scalar metrics and structured failures
```

## Baseline circuit

| Parameter | Baseline | Automated range |
| --- | ---: | ---: |
| `RLOAD` | 1 kΩ | 100 Ω–10 kΩ |
| `RDEG` | 1 kΩ | 10 Ω–10 kΩ |
| `CDEG` | 0.5 pF | 10 fF–10 pF |
| `ITAIL_VAL` | 100 µA | 10 µA–1 mA |

MOS dimensions remain fixed at W=10 µm and L=0.15 µm. The original validated
netlists remain under `circuits/`; `circuits/ctle_automation.cir` is the portable
automation entry point.

## Code guide

### Repository layout

| Path | Purpose |
| --- | --- |
| `simulator/` | Python simulation wrapper, metrics, channel processing, DFE and result contracts |
| `circuits/blocks/` | Reusable SPICE subcircuits used by generated testbenches |
| `circuits/benches/` | Separate DC, AC, noise, HD3 and transient analysis netlists |
| `circuits/test/` | Small simulator fixtures that do not require the SKY130 PDK |
| `circuits/ctle_*.cir` | Original handoff and legacy regression netlists; these are not the primary receiver flow |
| `channels/` | Synthetic four-port Touchstone channel and its metadata |
| `experiments/` | Command-line evaluation, sweep, search and result-recording utilities |
| `tests/` | Unit, ngspice, synthetic-model and optional real-SKY130 tests |
| `docs/` | Approved Stage 1 decisions and deferred Stage 2 work |
| `results/` | Generated sweep/search results; most outputs are ignored by Git |

### Python modules

| File | What it does |
| --- | --- |
| `simulator/models.py` | Defines typed simulation requests, results, statuses and failure codes |
| `simulator/config.py` | Locates SKY130 and defines nominal conditions, load values and the 60-point PVT grid |
| `simulator/ngspice.py` | Renders parameters/templates, creates an isolated run directory, launches ngspice and returns structured results |
| `simulator/parser.py` | Parses scalar ngspice measurements and detects simulator errors |
| `simulator/waveform.py` | Parses `wrdata` files and calculates AC, noise and HD3 metrics |
| `simulator/channel.py` | Reads `.s4p` files, forms differential `Sdd21` and filters the NRZ waveform |
| `simulator/stimulus.py` | Generates deterministic PRBS7 NRZ data and terminated differential PWL sources |
| `simulator/receiver_metrics.py` | Selects sampling phase, applies the behavioral one-tap DFE and measures eye opening |
| `simulator/receiver.py` | Orchestrates hierarchical evaluation, constraints, early stopping, caching and PVT runs |
| `simulator/cache.py` | Stores scalar evaluations using content-derived identifiers |
| `simulator/provenance.py` | Hashes implementation/model inputs and records Git identity |
| `simulator/ctle.py` | Legacy CTLE-only evaluator retained for handoff regression |
| `simulator/metrics.py` | Legacy CTLE-only metric calculations |

### SPICE files

`circuits/blocks/ctle.spice` contains the reusable differential CTLE. It has
resistive loads, two fixed-size SKY130 NMOS devices, parallel source-degeneration
resistors/capacitors and an ideal Stage 1 tail-current source.

The analysis benches are deliberately separate:

| Bench | Output |
| --- | --- |
| `ctle_dc.cir` | Bias voltages, currents, output common mode and DC power |
| `ctle_ac.cir` | 1 MHz–20 GHz differential transfer magnitude and phase |
| `ctle_noise.cir` | Input/output-referred noise spectra from 10 MHz–5 GHz |
| `ctle_hd3.cir` | 100 MHz transient used to calculate third-harmonic distortion |
| `ctle_transient.cir` | Short channel-free signal/swing diagnostic |
| `receiver_transient.cir` | Channel-driven CTLE waveform exported for sampling, DFE and eye analysis |

The common-source CTLE is electrically inverting. The benches therefore use
`V(outn)-V(outp)` as the logical differential output so received bits have the
same polarity as transmitted bits.

### Evaluation flow

The primary API is `evaluate_receiver()` in `simulator/receiver.py`:

1. Validate `RLOAD`, `RDEG`, `CDEG`, tail current, DFE tap and simulation conditions.
2. Locate the SKY130 library and ngspice executable.
3. Hash the design, model, channel, implementation and tool configuration.
4. Return a cached deterministic result when an identical evaluation exists.
5. Run progressively more expensive stages, stopping on the first hard failure.
6. Parse temporary waveform files into scalar metrics and discard the files.
7. Return a `ReceiverEvaluation` containing stages, metrics, violations, warnings and provenance.

The fidelity hierarchy is intended to reduce design time: inexpensive DC/AC
checks reject invalid designs before long receiver simulations, and only final
candidates should reach full PVT validation.

### Parameters and conditions

Use `ReceiverParameters` for design variables:

```python
from simulator import ReceiverParameters

design = ReceiverParameters(
    rload_ohm=1000.0,
    rdeg_ohm=1000.0,
    cdeg_f=0.5e-12,
    itail_a=100e-6,
    dfe_tap_v=0.0,
)
```

Use `SimulationConditions` for environment and loading:

```python
from simulator import ProcessCorner, SimulationConditions

conditions = SimulationConditions(
    process_corner=ProcessCorner.TT,
    temperature_c=27.0,
    supply_v=1.8,
    output_load_f=20e-15,
)
```

### Understanding results

`ReceiverEvaluation.success` means every executed stage passed its hard
constraints. Important fields are:

- `failed_stage`: first stage that failed, or `null` on success;
- `metrics`: combined scalar circuit/channel/DFE results;
- `stages`: per-analysis results, warnings, violations and runtime;
- `evaluation_id`: content-derived cache identity;
- `provenance`: model, circuit, channel, ngspice and Git fingerprints;
- `cache_hit`: whether ngspice was skipped because the result was reused.

Execution failures such as timeouts are marked retryable and are not cached.
Constraint failures are deterministic and may be cached. Candidate/final eye
checks use the trained sampling phase and one contiguous opening around it.

## Requirements

- Python 3.10 or newer
- NumPy (`python -m pip install -r requirements.txt`)
- ngspice (tested with the command-line executable)
- A local SKY130A ngspice model installation

Use one pinned interpreter for installation and test execution.  This avoids a
common Windows failure where NumPy is installed for Python 3.12 but tests are
launched by an MSYS Python 3.10:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import sys, numpy; print(sys.executable); print(numpy.__version__)"
python -m unittest discover -v
```

Select ngspice when it is not already on `PATH`:

```powershell
$env:NGSPICE_EXECUTABLE = 'C:\path\to\ngspice_con.exe'
```

Point Nebula at the local SKY130 library. The PDK itself is deliberately not
committed:

```powershell
$env:SKY130_MODEL_LIBRARY = 'C:\path\to\sky130A\libs.tech\ngspice\sky130.lib.spice'
```

If SKY130 is not installed, use a pinned prebuilt release from the
[`ciel` PDK manager](https://github.com/fossi-foundation/ciel), which is the
package-manager route recommended by
[`open_pdks`](https://github.com/RTimothyEdwards/open_pdks).  List available
SKY130 releases, choose and record an exact commit, then enable it:

```powershell
python -m pip install ciel
ciel ls-remote --pdk-family=sky130
ciel enable --pdk-family=sky130 <commit-hash>
```

Do not silently switch PDK revisions between a baseline and its verification;
the receiver-search manifest records the selected model checksum.

On this Windows setup the complete PDK was already installed by `ciel` under
WSL. Windows ngspice could not reliably open the WSL/UNC include tree, so the
required `libs.tech/ngspice` and `libs.ref/sky130_fd_pr/spice` directories were
staged under the ignored `.local-pdk/sky130A` directory. For Stage 1, generate a
fast fixed-device selector from those official model files:

```powershell
python -m experiments.build_compact_sky130 `
    .local-pdk\sky130A\libs.tech\ngspice\sky130.lib.spice
$env:SKY130_MODEL_LIBRARY = (Resolve-Path `
    .local-pdk\sky130A\libs.tech\ngspice\sky130.nebula_nfet.lib.spice).Path
```

The compact selector includes the official `sky130_fd_pr__nfet_01v8` model and
mismatch parameters for `tt/ss/ff/sf/fs`; it is not a replacement for the full
PDK in Stage 2. Provenance recursively fingerprints every included model file.

The `$env:` assignments above apply only to the current PowerShell session.
To save the locations permanently for the current Windows user, run:

```powershell
[Environment]::SetEnvironmentVariable(
    "SKY130_MODEL_LIBRARY",
    "C:\path\to\sky130A\libs.tech\ngspice\sky130.lib.spice",
    "User"
)

[Environment]::SetEnvironmentVariable(
    "NGSPICE_EXECUTABLE",
    "C:\path\to\ngspice_con.exe",
    "User"
)
```

Open a new PowerShell window after saving the variables, then verify them with:

```powershell
$env:SKY130_MODEL_LIBRARY
$env:NGSPICE_EXECUTABLE
```

`SKY130_MODEL_LIBRARY` is only required for the real-SKY130 integration tests;
the lightweight tests run without a local PDK installation.

The wrapper also discovers the standard `.ciel/sky130A`, `.volare/sky130A`,
and `$PDK_ROOT/sky130A` layouts. This keeps project files free of user-specific
absolute paths on Windows, WSL, Linux, and macOS.

## Evaluate the Stage 1 receiver

Run a cached 128-bit training evaluation against the included deterministic
synthetic channel:

```powershell
python -m experiments.evaluate_receiver --fidelity training
```

Use a real four-port channel when one is available:

```powershell
python -m experiments.evaluate_receiver --channel C:\channels\board.s4p --fidelity candidate
```

Port order is explicit and one-based (`TXP TXN RXP RXN`).  The default is
`1 2 3 4`; override it only from the channel vendor's documentation:

```powershell
python -m experiments.evaluate_receiver --channel C:\channels\board.s4p `
  --channel-ports 1 3 2 4 --fidelity candidate
```

Increase the per-ngspice timeout for a slow final run when necessary:

```powershell
python -m experiments.evaluate_receiver --fidelity final --timeout 300
```

The same flow can be called directly from Python:

```python
from simulator import EvaluationFidelity, ReceiverParameters, evaluate_receiver

result = evaluate_receiver(
    ReceiverParameters(rload_ohm=1000, rdeg_ohm=1000,
                       cdeg_f=0.5e-12, itail_a=100e-6,
                       dfe_tap_v=0.0),
    fidelity=EvaluationFidelity.CANDIDATE,
)

print(result.success, result.failed_stage)
print(result.metrics)
```

Fidelity levels control cost:

| Level | Work performed |
| --- | --- |
| `screening` | DC, full AC, 32-bit channel-free CTLE diagnostic |
| `training` | screening plus `.s4p` diagnostics and 128-bit receiver/DFE run |
| `candidate` | training path at 512 bits plus noise and 100 mV HD3 |
| `final` | 1024-bit receiver run and HD3 at 50/100/200 mV |

Every stage stops immediately on a hard failure. Results retain scalar metrics,
checksums, git state, ngspice version/solver, PDK identity, and schema versions.
Full waveform files live only in isolated temporary directories. The optional
content-addressed cache is `.nebula-cache/` and is ignored by Git.

The full final PVT grid is available as `simulator.PVT_GRID`: five process
corners, three supplies, and four temperatures (60 points). No reduced PVT set
is silently assumed; pass an explicit subset to `evaluate_pvt_grid` after that
stress set is calibrated from project data.

## Evaluate one CTLE design

```python
from simulator import evaluate_ctle

evaluation = evaluate_ctle({
    "RLOAD": "1k",
    "RDEG": "1k",
    "CDEG": "0.5p",
    "ITAIL_VAL": "100u",
})

print(evaluation.to_dict())
```

The legacy `evaluate_ctle` API remains available for regression against the
original handoff netlists. New optimization and RL work should use
`evaluate_receiver` so it receives the hierarchical tests, 100 MHz peaking
reference, channel/DFE metrics, early stopping, and evaluation fingerprints.

## PVT condition modes (`experiments/run_autockt_pipeline.py --pvt-condition-set`)

Also selectable from the local web UI (`experiments/web_ui.py`). Three modes:

| Mode | Conditions | Fidelity | What it establishes |
|---|---|---|---|
| `none` (default) | none — nominal only (TT / 27 °C / 1.8 V) | n/a | Nothing about PVT robustness; the selected design is only validated at nominal conditions. |
| `smoke` | 2 — nominal (TT / 27 °C / 1.8 V) + one stress corner (FF / 125 °C / 1.71 V) | `EvaluationFidelity.CANDIDATE` | A quick PVT screening/debug check that exercises the PVT-aware selection pathway with real SPICE. **It does not establish full PVT robustness** — it is not a substitute for the full sweep. |
| full 27-point sweep (`minimal27`) | 27 — the TT/SS/FF × VDD±5% × 0–125 °C set from `experiments/pvt_sweep.py` | `EvaluationFidelity.FINAL` | The actual PVT robustness validation. Slow (tens of minutes to hours depending on how many candidates reach this stage) and never run automatically. |

`smoke` uses `CANDIDATE` fidelity rather than `FINAL` because PVT selection
(`analysis/pvt_selection.py`) only ever reads each evaluation's
success/failure outcome, never its metric values — `CANDIDATE` runs the
identical evaluation stages `FINAL` does, just without `FINAL`'s extra
HD3 characterization sweep, which PVT selection never uses anyway.

For any mode that spends real SPICE time on PVT (`smoke` or the full
sweep), the web UI shows live progress once a run is under way — completed
vs. total condition evaluations (e.g. "PVT evaluation: 1 / 2 conditions"),
and the current condition being evaluated, when that information is
available from the pipeline's own output. A `none` run never shows PVT
progress, since it never evaluates any PVT condition. No estimated
completion time is shown for the full sweep — only that it is a
long-running, full 27-point evaluation.

## Tests

```powershell
python -m unittest discover -v
```

Tests that require SKY130 are skipped clearly when `SKY130_MODEL_LIBRARY` is
not configured. The lightweight divider integration test still verifies the
installed ngspice executable independently of SKY130.

## Sweeps and baseline search

Run the small default grid:

```powershell
python -m experiments.sweep --mode grid
```

Run deterministic random sampling:

```powershell
python -m experiments.sweep --mode random --count 50 --seed 7
```

Run the pre-RL random-search baseline:

```powershell
python -m experiments.receiver_search --preflight-only `
    --model-library C:\path\to\sky130A\libs.tech\ngspice\sky130.lib.spice `
    --ngspice-executable C:\path\to\ngspice_con.exe `
    --channel C:\channels\qualified_receiver.s4p

python -m experiments.receiver_search --count 100 --seed 7 `
    --sampling-policy constraint_aware_v1 `
    --model-library C:\path\to\sky130A\libs.tech\ngspice\sky130.lib.spice `
    --ngspice-executable C:\path\to\ngspice_con.exe `
    --channel C:\channels\qualified_receiver.s4p
```

The receiver search writes each attempt durably to JSON Lines and writes a
sidecar manifest containing the seed, schema versions and channel checksum.
It also fingerprints the Python evaluator, SPICE benches/block, PDK, ngspice,
conditions and port map.  Use a clean Git commit for an official baseline;
dirty work is fingerprinted but is harder for another person to reproduce.
Rerunning the identical command resumes missing candidate indices; a changed
manifest is rejected.  Use 100--500 designs before RL.  This command is a
workflow, not evidence that a qualified baseline has been completed: do not
train until the output contains the requested number of unique receiver rows
using the intended SKY130 library and qualified real channel.

Local software/PDK baseline evidence from 2026-08-23 is preserved in `results/`
(ignored by Git because it contains machine-specific paths):

- full-space `uniform`, seed 7: 100/100 unique rows, 0 successes, with 78 DC,
  21 AC and 1 transient failure;
- `constraint_aware_v1`, seed 11: 100/100 unique rows, 14 successes, 84 AC and
  2 transient failures, zero DC failures, reward range -100 to 100;
- top-three replay with cache disabled: all evaluation IDs and metrics matched
  at `rtol=1e-6`, `atol=1e-9`.

Both runs use real SKY130 transistor models and the synthetic regression channel.
They satisfy the pre-RL software baseline requirement but are not real-channel
or PCIe qualification evidence.

The preflight command does not run circuit simulations.  It resolves and
fingerprints the PDK and ngspice executable, parses the selected port map,
applies the same channel gates used by the evaluator, and reports the fitted
delay relative to the phase-unwrapping ambiguity limit.  Treat a successful
preflight as an input-integrity check, not as independent RF qualification.

The CLI accepts `--timeout`, `--solver`, `--cache`, and `--no-cache` when the
defaults are unsuitable.  A completed or interrupted checkpoint can be
validated and summarized without launching ngspice:

```powershell
python -m experiments.receiver_search --summary-only `
    --output results/receiver_random_search.jsonl
```

Completed runs also write `receiver_random_search.jsonl.summary.json`, which
records completeness, missing indices, success/failure counts, reward bounds,
unique evaluation identities, and the ten highest-scoring candidates.  The
summary validates every row against the manifest before reporting results.

After the checkpoint is complete, independently re-simulate the five
highest-scoring candidates with cache disabled and compare every metric:

```powershell
python -m experiments.receiver_search --verify-top 5 `
    --output results/receiver_random_search.jsonl `
    --model-library C:\path\to\sky130A\libs.tech\ngspice\sky130.lib.spice `
    --ngspice-executable C:\path\to\ngspice_con.exe
```

Verification first checks the current model, channel, ngspice, initialization,
solver, and checksums against the stored manifest.  It then writes a
`.verification.json` report.  A changed evaluation identity, success status,
failed stage, missing metric, or metric outside `--metric-rtol` and
`--metric-atol` makes verification fail.

The versioned receiver score uses locked-phase DFE eye height, width, decision
margin, CTLE power and peaking. Raw metrics and evaluation identities are kept
so a later objective can be audited.

## RL adapter contract

`simulator.ReceiverRLAdapter` is the framework-independent boundary for later
Gymnasium/RL integration.  It provides five normalized actions in `[-1, 1]`,
fixed-shape finite observations with per-metric validity masks, signed
constraint margins, a versioned reward, deterministic action sampling and a
non-resettable lifetime evaluation budget. Candidate failures do not terminate
an episode or silently reset the budget.

```python
from simulator import RLBudget, ReceiverRLAdapter

adapter = ReceiverRLAdapter(seed=7, budget=RLBudget(max_evaluations=100))
observation, info = adapter.reset(seed=7)
step = adapter.step(adapter.sample_action())
print(step.reward, step.constraints, step.info["reward_version"])
```

The adapter does not make the behavioral DFE or synthetic channel physical.
Promotion to RL still requires a real-channel baseline, repeatability checks,
and independent channel qualification.

Channel filtering interpolates magnitude and unwrapped phase, which preserves
bulk delay for adequately sampled Touchstone data.  Sparse frequency grids can
still make phase unwrap/delay ambiguous (roughly beyond half the reciprocal of
the largest frequency spacing).  Reject or resample such vendor data with a
trusted RF tool; the built-in passivity/causality screens are not a replacement
for scikit-rf or VNA-model qualification.

## Baseline verification

The circuit handoff reports approximately:

| Metric | Reported value |
| --- | ---: |
| Gain at 1 MHz | -7.18 dB |
| Gain near 2.5 GHz | -1.47 dB |
| 2.5 GHz peaking | 5.7 dB |
| Gain at 100 GHz | 1.37 dB |

Run `tests/test_sky130_integration.py` with the SKY130 environment variable set
to compare the local installation against these values. Differences outside
the documented tolerances should be investigated before optimization results
are trusted.

## RL readiness: strengths, weaknesses and next steps

The main software correctness pass for an RL backend is implemented and covered
by deterministic golden tests. A 100-design full-space baseline has been run
with the real SKY130 device models and synthetic channel; real-channel baseline
and repeatability evidence remain. Synthetic channels validate software behavior,
not PCIe signoff behavior.

| Area | Readiness | Current state |
| --- | ---: | --- |
| Parameter validation | 8/10 | Invalid designs and malformed conditions are handled cleanly |
| Failure handling | 8/10 | Timeouts, convergence failures and internal errors become structured results |
| Deterministic single-worker evaluation | 7/10 | Suitable for nominal and synthetic-channel experiments |
| Real `.s4p` channel correctness | provisional | Port mapping, passivity, causality and delay alignment are gated; vendor/tool qualification remains |
| DFE measurement | tested | Phase lock uses known-bit DFE-corrected margin and deterministic post-warm-up error-count/margin gates |
| Parallel RL workers | tested locally | Per-key cache locking, atomic commits and thread/process stress tests are present |
| Evaluation speed | 4/10 | Candidate fidelity is too expensive for every RL step |
| Physical signoff coverage | 4/10 | Reflections, passivity, PVT, loading and parasitics need more work |

### Strengths

- Design parameters and simulation conditions are validated before ngspice is
  launched.
- Screening, training, candidate and final fidelity levels provide early
  stopping and control evaluation cost.
- Timeouts, convergence problems and unexpected internal failures have stable,
  machine-readable failure codes.
- Retryable failures are not cached, while deterministic results include
  implementation, channel, model and tool provenance.
- Candidate eye height is evaluated at the trained phase, and eye width is the
  contiguous opening around that phase.
- The one-tap DFE retains its learned state when moving from known-bit training
  to decision-directed operation.
- The evaluator has a scalar result contract that an RL environment can consume
  without depending on temporary waveform files.
- Synthetic, lightweight-ngspice and optional real-SKY130 tests cover the main
  execution paths.

### Implemented correctness gates and remaining qualification

1. **Channel latency is aligned in software.** Fitted bulk delay, integer-bit
   offset and fractional phase are recorded and tested from 0 through 20 UI.
2. **Sampling-phase training is DFE-aware.** Only the known training window is
   scored; measurement bits are excluded from phase selection.
3. **Bit correctness is a hard constraint.** Every receiver fidelity requires
   zero post-warm-up errors and positive minimum margin.
4. **AC shapes are classified.** Local peaks and bounded high-frequency shelves
   pass; an uncontrolled monotonic rise fails.
5. **The cache is parallel-safe on local filesystems.** Per-identity locks,
   unique temporary files and atomic first-valid-writer commits are stress-tested.
6. **The real-channel model remains an approximation.** Configurable port maps,
   passivity and band-limited causality screens are present, but `Sdd21` filtering
   still omits a full reflected/mode-converted termination solution.
7. **HD3 validates/resamples time output** and uses a conditioned multi-harmonic
   fit; real-PDK bench correlation remains required.
8. **DC rail headroom, transient average power and true rail excursions are hard gates.**
   A resistively loaded differential pair is allowed to let its inactive output
   return close to `VDD`; proximity alone is not classified as clipping.

The random-search runner supports two manifest-pinned policies. `uniform` is the
unbiased full action-space baseline. `constraint_aware_v1` still produces normalized
random actions, but rejects combinations whose first-order resistive-load bias cannot
meet DC headroom, whose degeneration time constant is far from the target band, or
whose DFE tap is excessive for the synthetic channel. The simulator remains the final
authority for every accepted action; this policy does not change the RL action bounds.

The bundled synthetic channel has zero phase and exists only to test software
deterministically. It must not be used as final PCIe channel evidence.

### Remaining qualification before RL training

The software suite covers analytic 0/0.3/0.7/0.99/1/5/20-UI delays,
phase-sensitive DFE behavior, AC shapes, HD3 timebase/harmonic behavior, cache
concurrency and the RL contract.  These are implementation tests, not a bit
error-rate qualification or a substitute for an independent RF oracle.

Before training, qualify the chosen real channel and port map with an external
RF tool, run the checkpointed 100--500-design receiver baseline on a clean
commit, repeat pinned candidates to establish numerical tolerance, and review
failure distributions/reward rankings.  Reserve final fidelity and the full
PVT grid for promoted designs.  Stage-level reuse across fidelity promotions is
a remaining performance enhancement, not a reward-correctness prerequisite.

## Project boundaries

This integration provides both the preserved PPO v1 baseline and the opt-in
PPO v2 configuration. Select them with `--rl-version v1|v2` in the pipeline
or training CLI, or with the version selector in the browser UI. Full Stage 2 work still
includes a physical bias/tail source, sampler and DFE, MOS sizing groups,
parasitics, and layout-grounded area/power. The complete decision and
deferred-work ledger is in `docs/stage1-decisions.md`.

## Phase 2 — Experimental Artifacts & Reproducibility

A complete, byte-for-byte experimental artifact package is available for
anyone reviewing the reported PPO/Random-Search/CEM/PVT results and the
current competition/demo pipeline:

- **Package**: [`artifacts/NEBULA_phase2_experimental_artifacts.zip`](artifacts/NEBULA_phase2_experimental_artifacts.zip)
- **Index**: [`artifacts/PHASE2_ARTIFACT_INDEX.md`](artifacts/PHASE2_ARTIFACT_INDEX.md) — per-file experiment, purpose, status, seed, and relationship to each reported result
- **Machine-readable manifest**: [`artifacts/PHASE2_ARTIFACT_MANIFEST.json`](artifacts/PHASE2_ARTIFACT_MANIFEST.json)
- **Checksums**: [`artifacts/PHASE2_SHA256SUMS.txt`](artifacts/PHASE2_SHA256SUMS.txt) — SHA-256 for every file in the ZIP

**What's included**: all PPO/Random Search/CEM logs (including the incomplete,
warm-started `cem_baseline_3x10.jsonl`, kept and explicitly labeled — never
treated as a fair benchmark), the mixed-target/single-target PPO learning
checkpoint, PVT original run, diagnosis, targeted reproduction check, and
final full rerun, exported schematics, channel inputs, circuit configs,
the feasible-design catalog, three real-SPICE SIGSEGV crash logs (with the
partial fault-handler trace that identified the likely root cause), the
successful real-SPICE UI run (`run_id ed91dd5617fa4c99b9539931c1c25ed6`),
and the complete current pipeline/UI source (`experiments/web_ui.py`,
`experiments/run_autockt_pipeline.py`, and every RL/simulator/analysis
module they depend on) plus their tests.

**PVT evidence, preserved as a sequence, not replaced**:

1. Original minimal-27 sweep — **23/27 PASS** (`results/design_a_pvt_minimal27.jsonl`, kept unmodified on disk)
2. Targeted diagnosis of the 4 failing conditions
3. Targeted reproduction check — **4/4 PASS**
4. Complete 27-point rerun — **27/27 PASS** (`results/design_a_pvt_minimal27_rerun.jsonl`, a separate file — the original 23/27 was never overwritten)

**Guarantees**:
- No historical experiment file was modified, regenerated, or deleted to build this package.
- Failed and incomplete runs are preserved, not hidden (three SIGSEGV crash logs, one incomplete/warm-started CEM run).
- Missing metadata (commands, seeds, versions) is marked `NOT RECORDED` in the index — never guessed or reconstructed and presented as exact.
- No new experiment was run to produce this package; it packages what already existed.

See `docs/FINAL_TECHNICAL_AUDIT.md` for the full narrative interpretation
of these results (what can and cannot be honestly claimed from them).

## Phase 7 — target-correct qualification and evidence dashboard

The local UI now includes seven source-traceable performance graphs and can
read both historical evidence and the trained PPO checkpoint directly from
the tracked Phase-2 ZIP on a clean checkout. Candidate filtering and PVT
selection use strict, independent per-metric target checks and fail closed;
AutoCkt's intentionally tolerant terminal reward is retained for learning but
is no longer treated as an engineering qualification predicate.

Run the UI with:

```text
python experiments/web_ui.py --port 8001
```

See [`docs/PHASE7_COMPLETION_REPORT.md`](docs/PHASE7_COMPLETION_REPORT.md)
for the preserved implementation order, performance-display semantics,
reviewed Stage-2 deferrals, and remaining qualification work. The
research-backed Matplotlib migration and performance-plot roadmap is in
[`docs/PERFORMANCE_PLOT_IMPLEMENTATION_PLAN.md`](docs/PERFORMANCE_PLOT_IMPLEMENTATION_PLAN.md).

PPO v1 and v2 are deliberately reported separately. Their reward scales are
not directly comparable, so strict target success, normalized target margins,
simulator evaluations, failures, and runtime are the common comparison axes.
New training runs emit append-only `.events.jsonl` records and can export both
inference-only and resumable checkpoints. Dashboard plots are rendered through
Matplotlib as downloadable SVG or PNG rather than browser-drawn approximations.
