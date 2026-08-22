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

Select ngspice when it is not already on `PATH`:

```powershell
$env:NGSPICE_EXECUTABLE = 'C:\path\to\ngspice_con.exe'
```

Point Nebula at the local SKY130 library. The PDK itself is deliberately not
committed:

```powershell
$env:SKY130_MODEL_LIBRARY = 'C:\path\to\sky130A\libs.tech\ngspice\sky130.lib.spice'
```

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
python -m experiments.search --count 100 --seed 7
```

Each attempt is recorded incrementally as JSON Lines so a later failure does
not discard earlier results. CSV is produced for completed sweeps. Generated
result files are ignored by Git.

The preliminary search score favors peaking near 6 dB and lower power. Raw
measurements are stored separately from that score so the objective can change
without rerunning ngspice.

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

The wrapper is suitable for early, single-worker RL prototyping with the
synthetic channel, but it is not yet reliable enough for large parallel
training or reward generation from arbitrary real channel files. Its current
overall readiness is approximately **5/10 as an RL backend**: the software
contract and failure handling are strong, while several measurement and
scalability issues could still teach an agent the wrong behavior.

| Area | Readiness | Current state |
| --- | ---: | --- |
| Parameter validation | 8/10 | Invalid designs and malformed conditions are handled cleanly |
| Failure handling | 8/10 | Timeouts, convergence failures and internal errors become structured results |
| Deterministic single-worker evaluation | 7/10 | Suitable for nominal and synthetic-channel experiments |
| Real `.s4p` channel correctness | 4/10 | Bulk channel delay is not aligned with transmitted bit indices |
| DFE measurement | 5/10 | DFE state is preserved, but phase selection is not DFE-aware |
| Parallel RL workers | 3/10 | Concurrent cache writes can collide |
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

### Weaknesses and correctness blockers

1. **Channel latency is not aligned.** A real S-parameter channel may delay the
   waveform by several unit intervals. Searching only for a fractional phase
   within one UI can compare received sample `n` with the wrong transmitted bit.
2. **Sampling-phase training is not DFE-aware.** The best phase should be chosen
   from the DFE-corrected training margin, not only the raw CTLE output.
3. **Bit correctness is not a hard candidate constraint.** Eye height and width
   can pass even if the deterministic waveform contains decision errors.
4. **AC peak classification is incomplete.** Selecting the maximum only inside
   the target band can incorrectly accept a monotonically rising response.
5. **The cache is not parallel-safe.** Multiple workers can attempt to replace
   the same temporary cache file, particularly on Windows.
6. **The real-channel model is an approximation.** It uses matched differential
   `Sdd21`; reflections, mode conversion, configurable port maps, passivity,
   causality and impedance interactions are not fully modeled.
7. **HD3 assumes effectively uniform transient samples.** Nonuniform output
   should be validated or resampled before FFT analysis.
8. **Some physical limits are informational only.** Output clipping, transient
   average power, headroom and area are not all enforced as hard constraints.

The bundled synthetic channel has zero phase and exists only to test software
deterministically. It must not be used as final PCIe channel evidence.

### Recommended implementation order

Complete a simulator-correctness and RL-contract pass before spending
significant compute on training:

1. Estimate bulk delay from unwrapped `Sdd21` phase, align received samples with
   transmitted bits, report channel latency, and discard invalid warm-up/tail
   regions.
2. Run training-mode DFE at each candidate sampling phase and select the phase
   with the greatest corrected decision margin.
3. Require zero post-warm-up errors and positive minimum decision margin for a
   deterministic candidate evaluation.
4. Define and test the required CTLE peak explicitly as a local peak, global
   peak, or controlled high-frequency shelf.
5. Make cache writes parallel-safe with unique temporary files and atomic
   replacement.
6. Make screening independent of the channel, then validate a requested channel
   before starting expensive noise, HD3 or receiver stages.
7. Add stage-level caching so a promoted design can reuse compatible DC, AC,
   noise and HD3 results.
8. Add an RL adapter with normalized actions, explicit observations and
   constraints, deterministic seeds, evaluation budgets and a separately
   versioned reward function.

Before RL training, create a golden validation suite covering 0, 0.3, 1, 5 and
20 UI channel delays, controlled precursor/postcursor ISI, monotonic and peaked
AC responses, DFE phase-sensitive signals, repeated deterministic evaluations
and concurrent cache writes. Then run a 100--500 design random-search baseline
before single-worker RL. Enable multiple RL workers only after cache and
reproducibility stress tests pass; reserve final fidelity and the full PVT grid
for promoted designs.

## Project boundaries

This integration deliberately stops before the reinforcement-learning agent.
The first four correctness blockers above should be resolved before trusting
the wrapper's reward signal for real-channel training. Full Stage 2 work still
includes a physical bias/tail source, sampler and DFE, MOS sizing groups,
parasitics, and layout-grounded area/power. The complete decision and
deferred-work ledger is in `docs/stage1-decisions.md`.
