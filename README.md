# Nebula

Nebula is an automated analog-design framework for a PCIe Gen-2 receiver
equalizer. This branch integrates the parameterized SKY130 CTLE with a robust,
RL-independent Python-to-ngspice evaluation pipeline.

The current optimization scope is the 1-stage CTLE only:

```text
parameters -> generated netlist -> ngspice -> scalar measurements
           -> validity checks -> derived metrics -> JSONL/CSV results
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

## Requirements

- Python 3.10 or newer
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

The evaluator returns raw gain and operating-point measurements, 2.5 GHz
peaking relative to 1 MHz, output common mode/mismatch, approximate DC power,
runtime, constraint status, and structured failure information.

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

## Project boundaries

This integration deliberately stops at a reliable CTLE evaluator, dataset
collection, grid/random sweeps, and a baseline search. No reinforcement-learning
agent is implemented yet. RL should consume this same evaluator only after the
baseline is reproducible and the parameter ranges and objective are agreed.
