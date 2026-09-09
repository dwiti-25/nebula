# NEBULA next-phase status and implementation handoff

## Purpose

This file records the exact state of the project at the PPO v2 training
snapshot and defines the work required to reach the promised product: an
automated, simulation-efficient, transistor-level PCIe Gen-2 CTLE + 1-tap DFE
sizing flow with defensible area and PVT qualification.

Snapshot branch: `codex/v2-training-snapshot`

The repository is approximately **55-65% complete against the full promised
product**. The reusable software framework is approximately **75-85% complete**;
physical circuit completion and final experimental validation are approximately
**35-45% complete**. These are engineering estimates, not test-coverage values.

## Completed and preserved

- End-to-end target -> PPO candidate generation -> strict selection -> report
  and schematic/netlist export.
- Real ngspice integration using SKY130 device models.
- Hierarchical DC, AC, transient, channel, noise and HD3 evaluation with early
  rejection.
- AutoCkt-compatible PPO v1 preserved as a selectable historical baseline.
- PPO v2 corrected reset, validity-aware state, dense reward, inference-only
  checkpoints and resumable checkpoints.
- Evaluation caching, append-only event records and checkpoint metadata.
- Random Search and CEM implementations.
- PVT-aware selection and a preserved Design-A 27/27 TT/SS/FF evidence run.
- Strict per-metric target assessment, including the 15 mW power limit.
- Matplotlib SVG/PNG evidence plots and statistical primitives.
- Synthetic PPO v2 training checkpoints for seeds 101-105. Seed 101 improved
  from 0/20 to 20/20 synthetic checkpoint-evaluation successes. These models
  are mechanics evidence only and are not real-circuit evidence.

## Important limitations at this snapshot

- MOS width, length and multiplicity are fixed and are not optimization
  variables.
- The sampler/DFE remains behavioral rather than a complete physical
  transistor-level implementation.
- Total circuit area is not computable; only partial MOS channel area is known.
- The final PCIe `.s4p` channel has not been selected or qualified.
- The documented 27-point result covers TT/SS/FF, not the promised five process
  corners TT/SS/FF/SF/FS or the available full 60-condition grid.
- PPO has not been shown to outperform Random Search or CEM under matched,
  independent multi-seed real-SPICE evaluation.
- Controlled unseen-target evaluation did not establish reliable PPO
  generalization.
- Multi-actor/MA-Opt and surrogate-assisted actor-critic variants are not
  implemented.

## Current implementation scope

Implement everything below **except final `.s4p` channel integration**, which
is intentionally deferred until a documented channel file is supplied.

### 1. Expanded physical parameter schema

- Add grouped MOS width, length and multiplicity variables for matched CTLE
  devices.
- Preserve matching by exposing group variables, never independent dimensions
  for devices that must remain matched.
- Define explicit linear/log/discrete fabrication-feasible grids.
- Record units, bounds, scaling, grouping and schema version in checkpoints and
  run manifests.
- Keep the existing five-parameter schema runnable for historical reproduction.

### 2. Netlist and evaluator integration

- Parameterize the SKY130 CTLE subcircuit with the new grouped MOS variables.
- Propagate the variables through receiver parameters, netlist rendering,
  cache fingerprints, reports and schematic export.
- Add validation and fail-closed handling for invalid geometry/multiplicity.
- Keep staged simulation and early-rejection semantics unchanged.

### 3. Area model

- Calculate MOS active/channel area from grouped dimensions and multiplicity.
- Add passive area only when backed by documented PDK resistor/capacitor
  density or generated-layout geometry.
- Keep `total_area_computable=false` until MOS, passives, sampler, DFE and
  layout overhead are all represented.
- Never convert a partial estimate into a PASS for the 0.05 mm2 requirement.
- Expose component-level area contributions and provenance in JSON/UI reports.

### 4. Versioned RL contract

- Introduce a new expanded action/state/checkpoint schema; do not silently
  reinterpret PPO v1/v2 checkpoints.
- Add normalized geometry/area observations and validity masks.
- Add area margin to the dense reward only when the selected area mode is
  computable; strict qualification remains independent of reward.
- Reject incompatible checkpoints with a clear error.

### 5. Optimizers, pipeline and UI

- Give PPO, Random Search and CEM the identical ordered parameter schema,
  bounds and budget accounting.
- Expose legacy versus expanded sizing mode in CLI/UI.
- Display MOS group values and partial/complete area status.
- Add the missing visible training-run evidence selector.
- Continue to label synthetic results as non-circuit evidence.

### 6. Tests and smoke evidence

- Unit-test bounds, grid construction, matching, serialization and netlist
  substitution.
- Assert legacy checkpoints and results remain reproducible.
- Assert incompatible checkpoints fail clearly.
- Assert all optimizers consume the same schema.
- Add synthetic smoke training for the expanded state/action dimensions.
- Add a minimal real-ngspice geometry sensitivity test when the local PDK is
  available.

## Deferred `.s4p` channel work

When a candidate channel is supplied, separately verify port order, reference
impedance, frequency coverage, fixture/de-embedding status, differential
insertion loss at 2.5 GHz, passivity, causality, delay and mode conversion.
The final training campaign must use the qualified channel; connector-only
models may be used for integration testing but not called a complete PCIe link.

## Teammate training contract

Do not start final training until the expanded schema is merged and frozen.
The implementation handoff must provide:

- exact Git commit;
- state/action/reward/checkpoint schema versions;
- ordered parameters, groups, units, bounds and scaling;
- circuit and later channel checksums;
- target definitions and strict qualification rules;
- training/tuning/final-evaluation seed separation;
- episode, horizon and real-evaluation budgets;
- required policy/full-checkpoint and JSONL output names.

After handoff, the training teammate should run a synthetic compatibility
smoke, at least five independent training seeds, matched frozen-policy tests,
and budget-matched PPO/Random Search/CEM comparisons. Real-SPICE promotion is
allowed only after repeatable synthetic improvement, and final claims require
independent real-SPICE records.

## Definition of completion

The promised product is complete only when it produces transistor-level
parameter sets from target specifications, measures all required metrics,
computes defensible total area, validates all required PVT corners using the
qualified final channel, produces multiple verified trade-off designs, and
demonstrates simulation/time performance against matched alternative
optimizers. A functioning pipeline or a synthetic PPO success alone does not
satisfy this definition.
