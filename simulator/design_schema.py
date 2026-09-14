"""Ordered physical design contracts. Legacy policies retain their five heads."""

LEGACY_BOUNDS = (
    ("rload_ohm", 100.0, 10000.0, "log"),
    ("rdeg_ohm", 10.0, 10000.0, "log"),
    ("cdeg_f", 10e-15, 10e-12, "log"),
    ("itail_a", 10e-6, 1e-3, "log"),
    ("dfe_tap_v", -0.4, 0.4, "linear"),
)
V3_BOUNDS = LEGACY_BOUNDS + (
    ("mos_width_um", 0.42, 100.0, "linear"),
    ("mos_length_um", 0.15, 1.0, "linear"),
    ("mos_multiplier", 1, 16, "integer"),
)

def parameter_bounds(version="v1"):
    if version not in ("v1", "v2", "v3"):
        raise ValueError(f"Unknown design version: {version}")
    return V3_BOUNDS if version == "v3" else LEGACY_BOUNDS

def parameter_names(version="v1"):
    return tuple(row[0] for row in parameter_bounds(version))
