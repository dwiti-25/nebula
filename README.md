# Nebula

Automated analog receiver equalization framework for a PCIe Gen-2 receiver.

The project aims to develop a simulation-efficient reinforcement-learning-based
framework for optimizing a fixed-topology receiver consisting of a 1-stage CTLE
with source degeneration and a 1-tap DFE.

## Current Status

### Person A — CTLE / Circuit & SPICE

Completed:

- SKY130A + ngspice simulation environment
- SKY130 NMOS sanity test
- 1-stage differential CTLE
- Resistive loads
- Source degeneration using R || C
- Tail current source
- DC operating-point simulation
- AC frequency-response simulation
- Baseline circuit sizing
- Parameterized CTLE interface

### Current baseline

| Parameter | Value |
|---|---:|
| VDD | 1.8 V |
| NMOS W | 10 µm |
| NMOS L | 0.15 µm |
| RLOAD | 1 kΩ |
| RDEG | 1 kΩ |
| CDEG | 0.5 pF |
| ITAIL | 100 µA |

AC sweep:

```text
1 MHz → 100 GHz

