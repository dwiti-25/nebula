"""Area measurement for the CTLE circuit -- Task 2 of the official-brief
audit (docs/autockt-mapping.md sec 21, finding D: "area is completely
unmeasured"). Read-only: parses the real, unmodified
circuits/blocks/ctle.spice (the same file experiments/export_final_schematic.py
already includes) rather than duplicating hardcoded literals, so this stays
correct if that file ever changes.

WHAT THIS DOES AND DOES NOT COMPUTE -- read before using this number
============================================================================
`circuits/blocks/ctle.spice` instantiates exactly two transistors
(XMP, XMN, both `sky130_fd_pr__nfet_01v8`) with FIXED, never-varying
dimensions (W=10, L=0.15 -- identical across every circuit file in this
repository, confirmed by direct grep across circuits/; no experiment in
this project has ever tuned W or L). Per SKY130's documented device-
subcircuit convention (W/L given directly, matching the PDK's own
150 nm minimum channel length exactly under a microns reading -- 0.15
meters would be physically nonsensical for a transistor channel), these
are interpreted as MICRONS. This module computes ONLY the raw transistor
CHANNEL area (2 x W x L) from those two fixed values.

This function DOES NOT, and per instruction MUST NOT pretend to, compute:
  - real transistor LAYOUT area (source/drain diffusion, contacts, guard
    rings, spacing -- typically several x to tens-of-x the raw channel
    area; no layout-margin data exists anywhere in this project)
  - resistor area (RLOAD, RDEG are ideal SPICE `R` elements in
    circuits/blocks/ctle.spice -- not a physical SKY130 resistor
    primitive; no sheet-resistance (ohm/sq) or per-ohm layout-width value
    exists anywhere in this project for them)
  - capacitor area (CDEG is an ideal SPICE `C` element -- not a physical
    SKY130 MiM/MoM capacitor primitive; no capacitance-per-area (fF/um^2)
    value exists anywhere in this project for it)

Given the design's total area is almost certainly dominated by the
resistors and capacitor (multi-kohm resistors and 10fF-10pF-range
capacitors typically require far more silicon area than a minimum-size
transistor pair), `transistor_channel_area_um2` is NOT a usable proxy for
total circuit area and must never be compared directly against the
official 0.05 mm^2 budget as if it were the whole answer. `AreaEstimate.
total_area_computable` is always False; the missing resistor/capacitor
area models are documented, not fabricated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from simulator.receiver import BLOCK, ReceiverParameters

_DEVICE_LINE_RE = re.compile(
    r"^\s*X(\w+)\s+.*?\bsky130_fd_pr__nfet_01v8\b.*?\bW\s*=\s*([0-9.eE+-]+)\s+L\s*=\s*([0-9.eE+-]+)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class TransistorDevice:
    name: str
    width_um: float
    length_um: float
    multiplier: int = 1

    @property
    def channel_area_um2(self) -> float:
        return self.width_um * self.length_um * self.multiplier


@dataclass(frozen=True)
class AreaEstimate:
    source_file: str
    devices: tuple[TransistorDevice, ...]
    transistor_channel_area_um2: float
    transistor_channel_area_mm2: float
    total_area_computable: bool = False
    missing_components: tuple[str, ...] = field(default_factory=lambda: (
        "resistor layout area (RLOAD, RDEG are ideal SPICE elements -- "
        "no sheet-resistance/ohm-per-square-micron model exists in this project)",
        "capacitor layout area (CDEG is an ideal SPICE element -- no "
        "capacitance-per-area (fF/um^2) model exists in this project)",
        "transistor layout overhead beyond raw channel area (diffusion, "
        "contacts, guard rings, spacing -- no layout-margin factor exists "
        "in this project)",
    ))
    notes: tuple[str, ...] = field(default_factory=lambda: (
        "W/L are interpreted as microns per SKY130's documented device-"
        "subcircuit convention (not independently re-verified against the "
        "locally cached PDK files in this pass); L=0.15 matching SKY130's "
        "documented 150nm minimum channel length is consistent with that "
        "reading and physically implausible under any other unit.",
        "transistor_channel_area_um2/mm2 is the channel area ONLY -- do "
        "NOT compare it directly against the 0.05 mm^2 official budget; "
        "see missing_components for what is not included.",
        "what a defensible total-area estimate would require: (1) SKY130 "
        "PDK per-unit resistor (ohm/square) and capacitor (fF/um^2) "
        "layout data for the specific device flavors this design would "
        "use -- not sourced here, since the locally cached PDK is outside "
        "this project's own access boundary and no such figures are "
        "otherwise recorded in this repository; (2) a stated layout-"
        "overhead/margin factor for the transistors (diffusion, contacts, "
        "guard rings, routing) -- none is defined anywhere in this "
        "project. Absent both, any total-area number would be invented, "
        "not measured -- so none is reported.",
    ))


def parse_transistor_devices(source: str) -> tuple[TransistorDevice, ...]:
    """Parses X-instantiation lines referencing sky130_fd_pr__nfet_01v8 for
    their W/L values, straight from the actual circuit text. Raises if none
    are found, rather than silently returning an empty/zero estimate.
    """

    devices = tuple(
        TransistorDevice(name=match.group(1), width_um=float(match.group(2)), length_um=float(match.group(3)))
        for match in _DEVICE_LINE_RE.finditer(source)
    )
    if not devices:
        raise ValueError("no sky130_fd_pr__nfet_01v8 device instantiations found in source")
    return devices


def estimate_ctle_area(
    block_path: str | Path = BLOCK,
    parameters: ReceiverParameters | None = None,
) -> AreaEstimate:
    """Computes the CTLE's transistor channel area from the real,
    unmodified circuit file (default: circuits/blocks/ctle.spice, the same
    file every real evaluation and experiments/export_final_schematic.py
    already use). See module docstring for exactly what this does and does
    not measure.
    """

    resolved = Path(block_path)
    source = resolved.read_text(encoding="utf-8")
    if "W={MOS_W}" in source and "L={MOS_L}" in source:
        sized = parameters or ReceiverParameters()
        devices = tuple(
            TransistorDevice(name=name, width_um=sized.mos_width_um,
                             length_um=sized.mos_length_um,
                             multiplier=int(sized.mos_multiplier))
            for name in ("MP", "MN")
        )
    else:
        devices = parse_transistor_devices(source)
    total_channel_area_um2 = sum(device.channel_area_um2 for device in devices)
    return AreaEstimate(
        source_file=str(resolved),
        devices=devices,
        transistor_channel_area_um2=total_channel_area_um2,
        transistor_channel_area_mm2=total_channel_area_um2 * 1e-6,
    )


def format_area_report(estimate: AreaEstimate) -> str:
    lines = [
        f"Transistor channel area (from {estimate.source_file}):",
        f"  {estimate.transistor_channel_area_um2:.4g} um^2 "
        f"({estimate.transistor_channel_area_mm2:.4g} mm^2)",
        "  Devices:",
    ]
    for device in estimate.devices:
        lines.append(
            f"    {device.name}: W={device.width_um}um L={device.length_um}um "
            f"M={device.multiplier} "
            f"-> {device.channel_area_um2:.4g} um^2"
        )
    lines.append("")
    lines.append("Official budget: area < 0.05 mm^2")
    lines.append(
        "Verdict: NOT ASSESSABLE -- transistor_channel_area_mm2 is a partial "
        "quantity, not total circuit area (see missing_components below); "
        "no PASS/FAIL claim is made against the 0.05 mm^2 budget."
    )
    lines.append("")
    lines.append("Missing to compute total area:")
    for item in estimate.missing_components:
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("Notes:")
    for note in estimate.notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_area_report(estimate_ctle_area()))
