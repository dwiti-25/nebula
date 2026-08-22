# Nebula Stage 1 implementation decisions

This document records the design decisions approved before implementing the
receiver wrapper. Stage 1 keeps the CTLE transistor-level and implements the
linear channel, sampler, and one-tap DFE behaviorally in Python.

## Circuit and analyses

- The CTLE is a reusable subcircuit exposing `RLOAD`, `RDEG`, `CDEG`, and
  `ITAIL_VAL`; MOS W/L remain fixed during Stage 1.
- DC, AC, noise, HD3, CTLE transient, and receiver transient analyses use
  separate benches and execute as a progressively more expensive hierarchy.
- Nominal conditions are TT, 1.8 V, 27 C, and 1.2 V input common mode.
- Required PVT values are TT/SS/FF/SF/FS, 1.71/1.80/1.89 V, and
  0/27/75/125 C. Routine evaluation is nominal; final validation uses the full
  60-condition grid.
- Each CTLE output has a nominal 20 fF next-stage load, with 10--50 fF
  validation.
- The integrated receiver uses a 100 ohm differential termination and a
  50 ohm Thevenin resistance per leg.

## Channel, stimulus, and receiver

- The wrapper accepts documented four-port Touchstone `.s4p` channels and
  includes a deterministic synthetic regression channel.
- Python converts the channel to a differential response and filters the NRZ
  waveform before generating receiver-side PWL sources.
- The nominal NRZ stimulus is 5 GT/s, 800 mVpp differential, 1.2 V common
  mode, 20 ps edges, and 2 ps maximum time step. PRBS7 is the primary pattern.
- The Stage 1 sampler and one-tap DFE run in Python. Sampling phase is learned
  from training bits, fixed for measurement bits, and uses a zero differential
  threshold. Training and decision-directed DFE modes are both supported.
- The common-source CTLE is electrically inverting. Analysis benches define
  logical receiver output as `V(outn)-V(outp)` so BER and eye polarity match
  the transmitted bit convention.
- The primary HD3 test is a 100 MHz, 100 mVpp differential sinusoid;
  50/100/200 mVpp characterization is supported.

## Reproducibility and performance

- ngspice receives an isolated wrapper-managed `.spiceinit`; KLU is used only
  when supported.
- External paths use typed placeholders. Circuit values use validated
  `.param` replacement.
- Results include content fingerprints for circuit, bench, channel, stimulus,
  initialization, conditions, metrics, PDK, and ngspice identity.
- Large transient waveforms are parsed immediately and discarded by default;
  diagnostics and final candidates may retain compressed artifacts.
- Analytical checks, DC, and AC reject bad candidates before expensive
  transient, noise, HD3, eye, or PVT work.

## Deferred Stage 2 and later evaluation

- Replace the ideal CTLE tail source with a physical bias/current source.
- Introduce grouped MOS sizing variables.
- Replace behavioral sampler and DFE with physical implementations.
- Complete sampler/DFE power and area accounting.
- Calibrate output load and package/pad input capacitance.
- Select the final real evaluation `.s4p` channel.
- Add PRBS15, clock jitter, mismatch/Monte Carlo, and stable hierarchical
  device operating-point extraction.
- Re-evaluate sampling phase and threshold as possible design variables.
