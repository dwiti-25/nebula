"""Exports a FINAL, permanent, directly-simulatable SPICE schematic for a
given receiver design -- the deliverable this project's official brief
requires ("outputs the final schematic and resulting specs") that nothing
in the repository previously produced.

WHY THIS IS NEEDED (see docs/autockt-mapping.md sec 21 for the full audit):
every simulation this project has ever run builds its netlist inside a
`tempfile.TemporaryDirectory` (simulator/ngspice.py::run_simulation) that is
deleted the moment the simulation finishes. Every experiment's "final
output" -- Random Search's Design A, PPO's best_parameters, CEM's best
candidate -- has only ever been a JSON dict of 5 numbers. This script closes
that gap: given a design's parameters (and, optionally, its already-measured
metrics), it renders the actual CTLE circuit -- the same
circuits/blocks/ctle.spice subcircuit every evaluation already uses,
UNCHANGED -- with the design's final component values substituted in, plus
a header documenting achieved specs and provenance, and writes it to a
permanent file.

Reuses ONLY public, already-existing simulator interfaces, read-only:
`simulator.receiver.BLOCK` (the CTLE block file path),
`simulator.ngspice.parameterize_netlist` (the same substitution mechanism
every real evaluation already uses to set component values), and
`simulator.config.Sky130Config`/`ProcessCorner` for provenance. Does NOT
call evaluate_receiver, does NOT run ngspice, does NOT modify
simulator/receiver.py or simulator/rl_adapter.py.

Does not (yet) attempt to also render the full channel-driven
receiver-level test bench (circuits/benches/receiver_transient.cir) as a
single self-contained file -- that bench's NRZ stimulus is generated
dynamically per-evaluation by receiver.py-private code, not a static
template, and inlining it is out of scope for this pass. The exported
schematic documents which bench file was used to verify the design, and
notes the DFE tap value as a behavioral (non-netlist) design parameter --
see sec 21's finding H.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from simulator.config import ProcessCorner, Sky130Config
from simulator.ngspice import parameterize_netlist
from simulator.provenance import git_identity
from simulator.receiver import BENCHES, BLOCK, ReceiverParameters

from analysis.area_estimate import estimate_ctle_area

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_BENCH = BENCHES / "receiver_transient.cir"

# The four physical component parameters that DO appear in the CTLE
# netlist's own `.param` line (see circuits/blocks/ctle.spice). dfe_tap_v
# is deliberately excluded -- it never reaches any netlist (sec 21 finding
# H); ReceiverParameters.spice_parameters() confirms this independently.
NETLIST_PARAMETER_NAMES = ("RLOAD", "RDEG", "CDEG", "ITAIL_VAL", "MOS_W", "MOS_L", "MOS_M")


# [CORRECTNESS NOTE, confirmed by direct test before use] circuits/blocks/
# ctle.spice declares its defaults via `.subckt CTLE ... \n+ params: RLOAD=1k
# ...` continuation syntax, NOT standalone `.param NAME=value` statements --
# simulator.ngspice.parameterize_netlist's regex only matches the latter
# (confirmed: calling it directly on BLOCK's own source raises KeyError,
# "Parameter 'RLOAD' is not declared in the netlist"). Every real bench file
# (e.g. circuits/benches/ctle_ac.cir) instead declares its OWN standalone
# `.param RLOAD=1k RDEG=1k ...` line and instantiates XCTLE with explicit
# `RLOAD={RLOAD} ...` references -- substitution happens at the BENCH's
# `.param` line, not inside the subcircuit file itself. This wrapper
# reproduces that exact, already-proven mechanism (a `.param` line this
# module writes, `parameterize_netlist`d the same way every real evaluation
# already substitutes bench parameters) around an UNMODIFIED `.include` of
# the real ctle.spice block file.
_WRAPPER_TEMPLATE = """\
.param RLOAD=1k RDEG=1k CDEG=0.5p ITAIL_VAL=100u MOS_W=10 MOS_L=0.15 MOS_M=1
.include "{block_path}"

XCTLE inp inn outp outn vdd 0 CTLE
+ RLOAD={{RLOAD}} RDEG={{RDEG}} CDEG={{CDEG}} ITAIL_VAL={{ITAIL_VAL}} MOS_W={{MOS_W}} MOS_L={{MOS_L}} MOS_M={{MOS_M}}
.end
"""


def render_final_schematic(
    parameters: ReceiverParameters,
    *,
    process_corner: ProcessCorner = ProcessCorner.TT,
    achieved_metrics: Optional[dict[str, Any]] = None,
    target_description: Optional[str] = None,
    source_description: Optional[str] = None,
    sky130: Optional[Sky130Config] = None,
) -> str:
    """Builds the final schematic's full text. Read-only: does not modify
    circuits/blocks/ctle.spice -- that file is `.include`d unmodified by
    absolute path; only resolves the SKY130 model path (for the header
    comment).
    """

    wrapper_source = _WRAPPER_TEMPLATE.format(block_path=BLOCK.as_posix())
    substituted = parameterize_netlist(wrapper_source, {
        "RLOAD": parameters.rload_ohm,
        "RDEG": parameters.rdeg_ohm,
        "CDEG": parameters.cdeg_f,
        "ITAIL_VAL": parameters.itail_a,
        "MOS_W": parameters.mos_width_um,
        "MOS_L": parameters.mos_length_um,
        "MOS_M": parameters.mos_multiplier,
    })

    model_path: str
    try:
        model_path = str((sky130 or Sky130Config()).resolve_model_library())
    except (FileNotFoundError, OSError) as exc:
        model_path = f"<not resolved on this machine: {exc}>"

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        identity = git_identity(REPOSITORY_ROOT)
        commit = identity.get("git_commit", "unknown")
        dirty = identity.get("working_tree_dirty", "unknown")
    except Exception:
        commit, dirty = "unknown", "unknown"

    header_lines = [
        "* ============================================================",
        "* NEBULA final schematic export",
        f"* Generated: {generated_at}",
        f"* Repository commit: {commit} (working tree dirty: {dirty})",
        f"* SKY130 model library: {model_path}",
        f"* Nominal process corner this file documents: {process_corner.value}",
        "* ============================================================",
        "*",
        "* This is the ACTUAL CTLE circuit (circuits/blocks/ctle.spice,",
        "* unmodified, .include'd below by absolute path) with this design's",
        "* final component values set as this file's own .param defaults --",
        "* a real, syntactically valid SPICE netlist of the designed",
        "* circuit itself, not a parameter list. It is NOT a complete",
        "* testbench (no supply/bias sources, no input stimulus, no",
        "* analysis command) -- instantiate it with your own sources/",
        "* .control block, or see the verifying bench referenced below for",
        "* the exact bias/termination/load structure this design was",
        "* qualified against.",
        "*",
    ]

    if target_description:
        header_lines.append(f"* Target specification: {target_description}")
    if source_description:
        header_lines.append(f"* Design source: {source_description}")

    header_lines.append("*")
    header_lines.append("* Final component values:")
    header_lines.append(f"*   RLOAD (load resistor)         = {parameters.rload_ohm:.6g} ohm")
    header_lines.append(f"*   RDEG  (source degeneration R) = {parameters.rdeg_ohm:.6g} ohm")
    header_lines.append(f"*   CDEG  (source degeneration C) = {parameters.cdeg_f:.6g} F")
    header_lines.append(f"*   ITAIL (tail bias current)     = {parameters.itail_a:.6g} A")
    header_lines.append("*")
    header_lines.append(
        f"* DFE tap (BEHAVIORAL, not a netlist element -- decision-feedback"
    )
    header_lines.append(
        f"*   correction applied in Python, see docs/stage1-decisions.md):"
    )
    header_lines.append(f"*   dfe_tap_v = {parameters.dfe_tap_v:.6g} V")
    header_lines.append("*")

    area = estimate_ctle_area()
    header_lines.append(
        f"* Transistor channel area: {area.transistor_channel_area_um2:.4g} um^2 "
        f"({area.transistor_channel_area_mm2:.4g} mm^2) -- PARTIAL measurement only,"
    )
    header_lines.append(
        "*   NOT total circuit area (resistor/capacitor/layout-overhead area are not"
    )
    header_lines.append(
        "*   modeled anywhere in this project -- see analysis/area_estimate.py). Do NOT"
    )
    header_lines.append(
        "*   compare this number against the 0.05 mm^2 official budget as if complete."
    )
    header_lines.append("*")

    if achieved_metrics:
        header_lines.append("* Achieved specifications (as measured, not re-verified by this export):")
        for name, value in sorted(achieved_metrics.items()):
            header_lines.append(f"*   {name} = {value}")
        header_lines.append("*")

    header_lines.append(
        f"* Verified against test bench: {VERIFICATION_BENCH.relative_to(REPOSITORY_ROOT)}"
    )
    header_lines.append(
        "* (channel-driven full-receiver transient bench; its NRZ stimulus"
    )
    header_lines.append(
        "*  is synthesized per-run and is not reproduced in this static file --"
    )
    header_lines.append(
        "*  re-run the evaluation pipeline to reproduce the verifying transient.)"
    )
    header_lines.append("* ============================================================")
    header_lines.append("")

    return "\n".join(header_lines) + substituted


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a final, permanent SPICE schematic for a receiver design."
    )
    parser.add_argument(
        "--parameters-json", type=Path, required=True,
        help="JSON file containing a flat dict with keys rload_ohm, rdeg_ohm, cdeg_f, "
        "itail_a, dfe_tap_v (the same shape every results/*.jsonl row's 'parameters' "
        "field already has).",
    )
    parser.add_argument(
        "--metrics-json", type=Path, default=None,
        help="Optional JSON file with the design's already-measured metrics, to "
        "document in the schematic's header (not re-verified by this script).",
    )
    parser.add_argument("--process-corner", choices=("tt", "ss", "ff", "sf", "fs"), default="tt")
    parser.add_argument("--target-description", type=str, default=None)
    parser.add_argument("--source-description", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing {args.output}")

    parameters = ReceiverParameters(**json.loads(args.parameters_json.read_text(encoding="utf-8")))
    achieved_metrics = None
    if args.metrics_json is not None:
        achieved_metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))

    schematic = render_final_schematic(
        parameters,
        process_corner=ProcessCorner(args.process_corner),
        achieved_metrics=achieved_metrics,
        target_description=args.target_description,
        source_description=args.source_description,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(schematic, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "parameters": asdict(parameters)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
