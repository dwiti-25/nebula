"""Natural-language request -> rl.target_spec.TargetSpec field schema.

Reuses the existing target schema verbatim (rl/target_spec.py::SPEC_NAMES,
EXISTING_THRESHOLDS) rather than inventing a new one. `rl/target_spec.py`
has no torch/simulator dependency (verified: `rl/__init__.py` is a
docstring only), so importing it here keeps this module import-light.

Two parsers share this module's schema and defaults:
  - the deterministic regex-based extractor below (`parse_request_text`),
    used directly by the mock/no-API-key path, and
  - nebula.llm_providers.AnthropicLLMProvider, which asks Claude for the
    same JSON shape via structured outputs and falls back to this parser
    on any failure.

Design choice, stated explicitly to avoid fabricating targets: a field is
only overridden from its existing-repo default when the request contains
an explicit NUMBER for that metric. Purely qualitative language ("low
power", "high speed") is recorded as an unquantified note, never
translated into an invented numeric threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rl.target_spec import EXISTING_THRESHOLDS, SPEC_NAMES, TargetSpec

# Base target: the existing repo's own always-achievable thresholds
# (rl/target_spec.py::TargetSpec.from_existing_thresholds -- no invented
# safety margin). Unmentioned fields fall back to these, never to a
# fabricated number.
DEFAULT_TARGET: dict[str, float] = dict(EXISTING_THRESHOLDS)

# metric_name -> (keyword pattern, unit-scale table). Unit scale maps a
# captured unit token (lowercased) to a multiplier that converts the
# captured number into the field's own SI unit (volts, watts, UI).
_FIELD_PATTERNS: dict[str, tuple[str, dict[str | None, float]]] = {
    "dfe_eye_width_ui": (
        r"eye[\s-]*width[^0-9+\-]{0,20}(-?\d+(?:\.\d+)?)\s*(ui)?",
        {None: 1.0, "ui": 1.0},
    ),
    "dfe_locked_phase_eye_height_v": (
        r"eye[\s-]*height[^0-9+\-]{0,20}(-?\d+(?:\.\d+)?)\s*(mv|v)?",
        {None: 1.0, "v": 1.0, "mv": 1e-3},
    ),
    "dfe_min_margin_v": (
        r"margin[^0-9+\-]{0,20}(-?\d+(?:\.\d+)?)\s*(mv|v)?",
        {None: 1.0, "v": 1.0, "mv": 1e-3},
    ),
    "ctle_power_w": (
        # unit is required for power -- "15" alone is ambiguous (W vs mW);
        # an unqualified number is left unparsed rather than guessed.
        r"power[^0-9+\-]{0,20}(-?\d+(?:\.\d+)?)\s*(mw|w)\b",
        {"w": 1.0, "mw": 1e-3},
    ),
}

_QUALITATIVE_TERMS = (
    "low power", "low-power", "high speed", "high-speed", "robust", "compact",
)


@dataclass(frozen=True)
class ParsedTarget:
    """Result of turning one natural-language request into a target dict.

    `fields`: the full 4-field target dict (rl.target_spec.SPEC_NAMES),
    every value either explicitly parsed from the request or copied from
    DEFAULT_TARGET -- `explicit_fields` records which is which.
    """

    fields: dict[str, float]
    explicit_fields: tuple[str, ...]
    unquantified_notes: tuple[str, ...] = field(default_factory=tuple)

    def as_target_spec(self) -> TargetSpec:
        return TargetSpec(**self.fields)


def parse_request_text(request: str) -> ParsedTarget:
    """Deterministic, dependency-free extraction -- the mock/no-API-key
    fallback, and the ground truth the LLM path is checked against.
    """

    lowered = request.lower()
    values = dict(DEFAULT_TARGET)
    explicit: list[str] = []

    for field_name, (pattern, unit_scale) in _FIELD_PATTERNS.items():
        match = re.search(pattern, lowered)
        if not match:
            continue
        number = float(match.group(1))
        unit = match.group(2) if match.lastindex and match.lastindex >= 2 else None
        if unit not in unit_scale:
            continue
        values[field_name] = number * unit_scale[unit]
        explicit.append(field_name)

    notes = tuple(
        f"qualitative preference '{term}' noted but not quantified -- "
        f"specify a numeric threshold to constrain it"
        for term in _QUALITATIVE_TERMS
        if term in lowered
    )

    return ParsedTarget(fields=values, explicit_fields=tuple(explicit), unquantified_notes=notes)


assert set(DEFAULT_TARGET) == set(SPEC_NAMES)  # schema stays in sync with rl/target_spec.py
