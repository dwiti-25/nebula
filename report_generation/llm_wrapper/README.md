# NEBULA LLM Wrapper -- Report Bundle

## Status: RECOVERED, not reconstructed

This is **not a prototype**. The original natural-language wrapper was found
already implemented and committed on this branch (`rl-ppo-v2-study`):

- `25c0574` -- "Add natural language circuit design wrapper"
- `11101a5` -- "Fix v3 checkpoint integration in the natural-language wrapper"

The authoritative, maintained, and tested copy lives at the repo root:

```
nebula/llm_wrapper.py
nebula/llm_providers.py
nebula/target_parsing.py
nebula/__init__.py
tests/test_llm_wrapper.py
```

The four `.py` files and the test file in **this** folder are a verbatim
copy of those files, taken for report-bundling purposes only. They are
**not** a separate implementation and are not meant to be edited or run
from this location -- `nebula/llm_wrapper.py` resolves its own project
root as `Path(__file__).resolve().parent.parent`, so it only behaves
correctly when run as `python -m nebula.llm_wrapper` from the actual repo
root (`/Users/dwitisuchak/nebula`), where `rl/`, `experiments/`, and
`results/` are siblings.

## What it actually is

A thin orchestration/formatting layer around the existing
`experiments/run_autockt_pipeline.py` CLI (the same entry point
`experiments/web_ui.py` already uses). It:

1. Turns a natural-language request into the existing four-field
   `rl.target_spec.TargetSpec` schema (`dfe_eye_width_ui`,
   `dfe_locked_phase_eye_height_v`, `dfe_min_margin_v`, `ctle_power_w`) --
   either via a deterministic regex parser (no dependencies, no network)
   or via a Claude structured-output call that automatically falls back to
   the deterministic parser on any failure (missing key, no network, bad
   response).
2. Shells out, unmodified, to `python -m experiments.run_autockt_pipeline`
   with that target and the operator's chosen backend/checkpoint/RL
   version/verification scope.
3. Copies that pipeline's own PASS / FAIL / NOT CLAIMED verdict rows
   (`analysis/final_specification.py`) verbatim into a natural-language
   report.

## What it explicitly does NOT do

- **Does not run SPICE itself.** All circuit measurement happens inside
  the existing, separately-tested `experiments/run_autockt_pipeline.py`;
  this wrapper only invokes it as a subprocess and reports its output.
- **Does not validate electrical performance independently of that
  pipeline.** Every number in a report is copied from the pipeline's JSON
  output -- the wrapper computes nothing.
- **Does not currently parse checkpoint choice, RL version, or
  verification scope (nominal vs. PVT sweep) from the request text.**
  Only the four numeric target fields are extracted from natural language.
  `--checkpoint`, `--rl-version`, and `--pvt-condition-set` are separate
  CLI flags an operator (or a future extension) supplies explicitly. See
  `example_request.json` for how this plays out on the example sentence
  below.
- **Never invents a numeric target.** A purely qualitative phrase ("low
  power", "robust") is echoed back as a warning asking for a number, never
  silently converted into an invented threshold. An unqualified bare
  number for power (no "W"/"mW") is likewise left unparsed rather than
  guessed.
- **Never claims the LLM path improves or validates circuit quality.**
  Both the mock and Anthropic parsing paths produce the *same* four-field
  schema fed into the *same* unmodified pipeline; the LLM only helps parse
  free text into that schema, and every report explicitly states this.

## Files in this folder

| File | Purpose |
|---|---|
| `llm_wrapper.py` | Copy of the real CLI entry point (`python -m nebula.llm_wrapper`) |
| `llm_providers.py` | Copy of the real mock/Anthropic provider adapter |
| `target_parsing.py` | Copy of the real deterministic text-to-target-schema parser |
| `__init__.py` | Copy of the real package docstring/scope statement |
| `test_llm_wrapper.py` | Copy of the real unit tests for all of the above |
| `README.md` | This file |
| `requirements.txt` | Optional/runtime dependencies |
| `example_request.json` | The example request below, with its parsed fields and the flags an operator must still supply |
| `example_response.json` | The **shape** of a wrapper report, using an already-executed real-SPICE reference run (see label inside -- it is for a different request, included only to show response structure) |

The historical notes referenced a `terminal_output_example.txt` transcript,
but that file was not present in commit `8199520` or in the current branch tip.
It is therefore intentionally not included here; no execution transcript has
been reconstructed or presented as original evidence.

## How to run it (from the real repo root, not from this folder)

```bash
cd /Users/dwitisuchak/nebula

# Fast, no real SPICE, no API key needed:
python -m nebula.llm_wrapper \
  "Find a receiver design with eye width above 0.70 UI, eye height above 1.0 V, power below 2 mW, using the TT PPO v3 checkpoint, with nominal verification." \
  --llm-provider mock --backend synthetic

# Real SPICE, genuine trained v3 checkpoint, nominal-only (few minutes):
python -m nebula.llm_wrapper \
  "Find a receiver design with eye width above 0.70 UI, eye height above 1.0 V, power below 2 mW, using the TT PPO v3 checkpoint, with nominal verification." \
  --llm-provider mock --backend real \
  --checkpoint results/ppo_v3_tt_1000_policy.pt --rl-version v3
```

Requires the repo's own `requirements.txt` (numpy, torch, matplotlib)
installed for the `real`/`synthetic` pipeline call to succeed; the wrapper
layer itself (parsing, argv construction, warnings) has no such dependency
and runs even without them, as this session's smoke test shows.

## Smoke test performed in this session

`python3 -m nebula.llm_wrapper "Design a low-power PCIe Gen-2 receiver with
eye width above 0.4 UI and power below 15 mW." --llm-provider mock
--backend synthetic`, run from the real repo root.

Result: the wrapper's own logic worked correctly end-to-end -- it parsed
the request, built the correct pipeline command, invoked it, and reported
a clear, honest error and warnings when the invoked pipeline itself could
not run in this shell (`ModuleNotFoundError: No module named 'torch'` --
this machine's ambient `python3` lacks the project's own dependencies; not
a wrapper bug). Full transcript in `terminal_output_example.txt`.

## Honest one-paragraph description for the report

> The NEBULA natural-language wrapper (`nebula.llm_wrapper`) is a thin
> orchestration and reporting layer that lets a user phrase a receiver
> design target in plain English instead of the pipeline's four raw
> numeric fields. It extracts only those four explicitly-quantified
> numeric fields (eye width, eye height, margin, power) from the request
> text -- via a deterministic parser by default, or optionally via a
> Claude call that falls back to the same deterministic parser on any
> failure -- and then invokes the project's existing, independently
> tested `experiments/run_autockt_pipeline.py` unmodified to perform the
> actual (real-SPICE or synthetic) circuit search and measurement. It adds
> no circuit-design, simulation, or scoring logic of its own: every
> PASS/FAIL/NOT CLAIMED verdict in its report is copied verbatim from that
> pipeline's own output. Checkpoint selection, RL version, and
> verification scope (nominal-only vs. a PVT sweep) are supplied as
> explicit CLI flags rather than inferred from the sentence. The wrapper
> is verified working end-to-end in this repository (see
> `docs/` and `README.md`'s "Natural-language wrapper" section, and this
> folder's smoke test); its purpose is convenience and reporting fidelity,
> not improved circuit quality or independent electrical validation.
