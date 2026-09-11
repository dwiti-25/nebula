# PPO v3 integration and per-run execution graph plan

Status: implementation contract to apply **after** the incoming AI changes are
merged and audited. This document deliberately does not start checkpoint
training or reinterpret an existing v1/v2 checkpoint.

## 1. Objective and non-negotiable rules

PPO v3 will extend the corrected PPO v2 formulation from five to eight design
variables while keeping the circuit topology fixed. It will optimize the four
passive/bias values, the behavioral DFE tap, and the matched CTLE input-pair MOS
geometry/multiplicity. It will use the same evaluator, targets, real channel,
budgets and final qualification rules as Random Search and CEM.

The implementation must satisfy these rules:

1. v1 and v2 remain reproducible and their checkpoint shapes and semantics are
   never silently changed.
2. v3 receives new state, action, reward, checkpoint and event schema IDs.
3. One ordered parameter registry is the source of truth for every optimizer,
   report and UI. No optimizer may maintain a private parameter order.
4. The selected channel path, port map and checksum are mandatory run inputs
   for real training and evaluation and are saved in every full checkpoint and
   run manifest.
5. Synthetic runs remain labelled as software/mechanics evidence and cannot be
   presented as circuit results.
6. Partial MOS channel area may be optimized as an explicitly named proxy, but
   it can never satisfy the official total-area requirement. A strict area PASS
   is enabled only when `total_area_computable=true`.
7. Missing measurements remain missing/null with validity flags. They are never
   replaced by plausible-looking zeros.
8. Invalid parameters, incompatible schemas and changed input fingerprints fail
   closed with actionable errors.

Here, "no floating points" means no floating or disconnected features/code
paths. Physical continuous quantities necessarily remain floating-point
numbers; discrete multiplicity remains an integer throughout serialization,
action decoding and SPICE rendering.

## 2. PPO v3 parameter contract

The fixed ordered action schema will be:

| Index | Parameter | Unit | Type/spacing | Current validated range | Function |
|---:|---|---|---|---|---|
| 0 | `rload_ohm` | ohm | positive/log grid | 100–10,000 | CTLE load and gain/headroom |
| 1 | `rdeg_ohm` | ohm | positive/log grid | 10–10,000 | source-degeneration strength |
| 2 | `cdeg_f` | F | positive/log grid | 10 fF–10 pF | degeneration zero/peaking |
| 3 | `itail_a` | A | positive/log grid | 10 uA–1 mA | bias, gm and power |
| 4 | `dfe_tap_v` | V | signed/linear grid | -0.4–0.4 | one-tap post-cursor cancellation |
| 5 | `mos_width_um` | um | fabrication grid | 0.42–100 | matched input-pair width |
| 6 | `mos_length_um` | um | fabrication grid | 0.15–1.0 | matched input-pair length |
| 7 | `mos_multiplier` | count | integer/discrete | 1–16 | matched input-pair multiplicity |

Bounds will be imported from the validated receiver/parameter definition rather
than copied into each optimizer. The MOS variables represent one matched group:
both input devices always receive identical W/L/M values. No independent P/N
actions will exist.

The first implementation will preserve the existing AutoCkt-style independent
three-choice action head per parameter (`decrease`, `hold`, `increase`, with the
project's documented index delta semantics). `mos_multiplier` will move only on
integer indices. All grids, including exact values and ordering, will be stored
in full checkpoints and run manifests.

## 3. PPO v3 state

`state_schema=v3` will retain the useful v2 information and add the expanded
physical context. The fixed state layout will contain:

1. target specification values and signed normalized target errors;
2. validity flags for every measured metric;
3. normalized indices for all eight parameters;
4. one-hot evaluator stage/failure status;
5. genuinely available early-stage values/margins such as DC power and AC
   peaking;
6. channel context: loss at 1.25/2.5/5 GHz and normalized fitted delay;
7. MOS geometry context: normalized W, L, M and partial MOS channel-area proxy;
8. explicit area status flags: partial/computable and claimed/not-claimable.

Every field will have a stable ordered name list. `STATE_DIM_V3` will be derived
from that list and asserted against the produced vector. Unknown failure stages
or missing registry fields will raise rather than shift positional meanings.

## 4. Reward and termination

v3 will start from reward v2 because it distinguishes early simulator failure,
graded target distance and genuine success. Additions will be versioned as
`reward_v3`:

- retain eye height, eye width, DFE margin and power target terms;
- retain explicit penalties for invalid parameters and early-stage failures;
- incorporate peaking compliance rather than merely displaying peaking later;
- optionally use the partial MOS channel-area proxy as a small, separately
  logged tie-break/regularizer;
- use total area in strict reward/termination only after a complete,
  PDK-grounded area model exists;
- never allow a reward bonus to override strict feasibility.

Episode success and final design selection will remain based on a shared strict
target assessment, not a reward threshold. Reward components and the strict
assessment will both be emitted for every step so their relationship is
auditable.

## 5. Checkpoints and compatibility

Full v3 checkpoints will contain:

- checkpoint/state/action/reward/event schema versions;
- exact ordered parameter registry and all grid values;
- policy, value network and optimizer state;
- Python, NumPy and PyTorch RNG state;
- update/evaluation counters and hyperparameters;
- training/validation target IDs and seeds;
- backend and fidelity;
- channel path, SHA-256, port map and qualification summary;
- SKY130 dependency fingerprint and ngspice identity;
- implementation Git commit and dirty-tree status;
- cache policy and evaluation-budget definition.

Inference exports will include a small metadata envelope rather than relying on
tensor shapes alone. Load order is validate-all-metadata first, mutate model
weights second. A v1/v2 checkpoint selected with v3, or a v3 checkpoint paired
with a different parameter order/channel without an explicit override, must be
rejected before evaluation.

## 6. Required integration matrix

The work is incomplete until every row below is connected and tested.

| Feature | Required v3 behavior |
|---|---|
| Receiver/netlists | W/L/M reach every relevant bench and matched devices |
| Real channel | Same `.s4p`, port map and checksum used in training, inference, PVT and refinement |
| PPO trainer | `--rl-version v3`; resumable full and inference checkpoints |
| Main pipeline | Load v3 metadata, build eight-head agent, run channel-aware candidates |
| UI | Enable v3 only when implementation is complete; explain all eight parameters and checkpoint compatibility |
| Random Search | Consume identical registry/bounds and log identical budget units |
| CEM | Consume identical registry/bounds, including integer M handling |
| Cache | Key includes eight parameters, conditions, fidelity, channel, models and implementation identity |
| PVT | Preserve v3 parameters and channel across every condition; no fallback defaults |
| HD3/noise refinement | Re-evaluate selected v3 design with the same channel and full parameter set |
| Area | Display component contributions, proxy/computability status and provenance |
| Schematic export | Emit W/L/M plus passives/bias and achieved metrics |
| Generalization | Version-aware unseen-target evaluator supporting v1/v2/v3 |
| Events/plots | Eight-parameter trajectories, reward components, failures, timing, cache and strict success |
| Comparison | PPO v2/v3, Random Search and CEM under matched real-SPICE evaluation budgets |
| Documentation | Frozen teammate training command, seed table and artifact naming |

## 7. Per-run graph-based output

This is additional to the current specification table and performance charts.
It is an execution/provenance graph for the particular run, answering: **what
produced this result?**

### 7.1 Graph contents

The graph will be a directed acyclic graph with typed nodes:

- `run`: run ID, start/end time and overall status;
- `configuration`: PPO version, hyperparameters, seeds and budget;
- `target`: exact requested specifications;
- `checkpoint`: path, schema versions and checksum;
- `channel`: path, port map, checksum and qualification metrics;
- `toolchain`: Git commit, PDK/model and ngspice fingerprints;
- `candidate`: episode, step, eight parameters, reward and strict status;
- `evaluation`: evaluation ID, cache hit/miss, fidelity and wall time;
- `stage`: DC, AC, CTLE transient, channel, noise, HD3 and receiver transient;
- `selection`: feasibility filter, PVT rank and trade-off rule;
- `PVT condition`: corner, supply, temperature and result;
- `artifact`: final JSON, checkpoint, schematic and plot files.

Typed edges will include `configured_by`, `initialized_from`, `uses_channel`,
`generated`, `evaluated_as`, `passed_to`, `failed_at`, `selected_by`,
`qualified_at` and `produced`.

An abbreviated successful run should read visually as:

```text
Target + PPO v3 checkpoint + channel + toolchain
                         |
                         v
                Candidate trajectory
                         |
                         v
 DC -> AC -> CTLE transient -> channel -> noise -> HD3 -> receiver/DFE
                         |
                         v
              strict feasibility filter
                         |
                         v
                  optional PVT grid
                         |
                         v
        selected design -> schematic + report + plots
```

A failed run terminates its stage chain at the actual failed stage and retains
the failure code/violations; it must not draw downstream stages as completed.

### 7.2 Data and rendering

Add `analysis/run_graph.py` with a pure builder and deterministic renderer.
The canonical output is versioned JSON (`run_graph_schema_version=1`), embedded
under `execution_graph` in pipeline output and also saved as
`<run-id>.graph.json`. SVG is rendered from that JSON for the UI; PNG download
is optional through the existing image-rendering infrastructure.

The renderer will use a deterministic layered layout and existing project
facilities rather than requiring Graphviz. Node colors will encode measured
PASS, measured FAIL, NOT RUN, cached, synthetic and NOT CLAIMED. Color is never
the only signal; every node includes a text/icon/status label.

The UI will add a **Run generation graph** card with:

- overview and candidate-detail modes;
- click/keyboard expansion of node evidence;
- links from artifact nodes to existing raw JSON/schematic/plot endpoints;
- SVG and graph-JSON downloads;
- a visible legend and limitations block.

### 7.3 Truthfulness and size controls

- Graph nodes are generated only from recorded run events/results.
- Unknown historical values remain null/unknown; no timings or links are
  inferred.
- Each metric records origin: `measured`, `derived`, `target`, `metadata` or
  `not_claimed`.
- IDs/checksums connect graph nodes back to immutable evidence.
- Large training runs initially show aggregated episode/update nodes; a bounded
  candidate subgraph is loaded on demand so the browser is not given millions
  of DOM elements.
- Graph construction must not rerun SPICE or modify the experiment.

## 8. Implementation order after incoming changes

1. **Merge audit:** inventory incoming files, reconcile parameter and schema
   definitions, run the existing full suite, and record the clean merge commit.
2. **Registry/schema freeze:** implement the single v3 registry and freeze state,
   action, reward, event and checkpoint contracts before training.
3. **Evaluator closure:** prove all eight variables affect rendered netlists,
   fingerprints and measured outputs; prove matched-device invariants.
4. **Optimizer closure:** connect PPO, Random Search and CEM to the same registry,
   channel and budget accounting.
5. **Pipeline/UI closure:** enable v3, channel-aware inference, PVT, refinement,
   exports and result displays.
6. **Run graph:** emit canonical graph JSON, render it in the UI and add downloads.
7. **Verification:** synthetic schema smoke, real geometry sensitivity, channel
   smoke, cache replay, resume determinism and failure-path tests.
8. **Freeze for teammate:** publish commit, exact commands, seeds, budgets,
   expected filenames and checksums; only then begin checkpoint training.

## 9. Definition of done

PPO v3 is ready for training only when all of the following are true:

- no disabled or placeholder v3 UI option remains;
- no production v3 path imports the legacy five-parameter order implicitly;
- v1/v2 regression tests pass unchanged;
- incompatible checkpoints fail before weights or simulator state are mutated;
- PPO/Random Search/CEM report the identical eight-parameter contract;
- real channel identity is present in checkpoints, manifests and final output;
- a real ngspice smoke proves W, L and M reach the circuit and the channel/DFE
  stages complete;
- cache identity changes for every new parameter and channel input;
- PVT/refinement/schematic paths preserve all eight values;
- current tables and performance plots still render;
- run graph JSON validates, SVG renders, and failed/cached/skipped stages are
  represented truthfully;
- the full test suite passes on a clean commit;
- the teammate receives a frozen, reproducible training contract.

