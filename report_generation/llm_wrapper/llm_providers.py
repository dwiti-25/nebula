"""LLM provider adapter -- isolates the (optional) Claude API call behind a
small interface, so the rest of nebula.llm_wrapper never talks to any LLM
SDK directly and works identically with no LLM configured at all.

Selected via environment variables:
  NEBULA_LLM_PROVIDER  "anthropic" (default) or "mock". "anthropic" still
                        falls back to the deterministic parser at runtime
                        (not just at startup) if the API is unavailable,
                        unconfigured, or errors -- see parse_target below.
  NEBULA_LLM_API_KEY    API key for the "anthropic" provider. If unset,
                        the standard Anthropic SDK credential resolution
                        (ANTHROPIC_API_KEY, `ant auth login`, etc.) is used
                        instead; if that also finds nothing, parsing falls
                        back to the deterministic parser.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

from rl.target_spec import SPEC_NAMES
from nebula.target_parsing import DEFAULT_TARGET, ParsedTarget, parse_request_text

PROVIDER_ENV_VAR = "NEBULA_LLM_PROVIDER"
API_KEY_ENV_VAR = "NEBULA_LLM_API_KEY"
DEFAULT_PROVIDER = "anthropic"
MODEL_ID = "claude-opus-5"

_TARGET_JSON_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": ["number", "null"]} for name in SPEC_NAMES},
    "required": list(SPEC_NAMES),
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "Extract circuit design target values from the user's request into exactly "
    "these four fields: dfe_locked_phase_eye_height_v (eye height in volts), "
    "dfe_eye_width_ui (eye width in unit intervals), dfe_min_margin_v (timing/"
    "voltage margin in volts), ctle_power_w (power in watts). "
    "Only fill in a field when the request states an explicit number for it "
    "(convert mV to V and mW to W). If a field is not explicitly quantified in "
    "the request, return null for it -- do not guess or invent a number for an "
    "unquantified or purely qualitative preference such as 'low power'."
)


@dataclass(frozen=True)
class ProviderResult:
    parsed: ParsedTarget
    provider_used: str  # "anthropic" or "mock"
    fallback_reason: str | None = None  # set only when provider_used == "mock" but "anthropic" was requested


class LLMProvider(Protocol):
    def parse_target(self, request: str) -> ProviderResult: ...


class MockLLMProvider:
    """Deterministic, dependency-free fallback -- no network, no API key.
    Always available; used directly when NEBULA_LLM_PROVIDER=mock, and as
    the automatic fallback for the anthropic provider.
    """

    def parse_target(self, request: str) -> ProviderResult:
        return ProviderResult(parsed=parse_request_text(request), provider_used="mock")


class AnthropicLLMProvider:
    """Calls Claude (structured outputs) to extract the same 4-field
    target schema the mock parser produces. Any failure (missing SDK,
    missing/invalid credentials, API error, malformed response) falls back
    to the deterministic parser rather than raising -- this provider never
    lets a natural-language interface become a hard dependency on a live
    LLM API key.
    """

    def __init__(self, *, api_key: str | None = None, model: str = MODEL_ID):
        self._api_key = api_key
        self._model = model

    def parse_target(self, request: str) -> ProviderResult:
        try:
            fields = self._call_claude(request)
        except Exception as exc:  # noqa: BLE001 -- any failure means "fall back", by design
            fallback = MockLLMProvider().parse_target(request)
            return ProviderResult(
                parsed=fallback.parsed, provider_used="mock",
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )

        values = dict(DEFAULT_TARGET)
        explicit: list[str] = []
        for name in SPEC_NAMES:
            value = fields.get(name)
            if value is None:
                continue
            try:
                values[name] = float(value)
            except (TypeError, ValueError):
                continue
            explicit.append(name)

        return ProviderResult(
            parsed=ParsedTarget(fields=values, explicit_fields=tuple(explicit)),
            provider_used="anthropic",
        )

    def _call_claude(self, request: str) -> dict[str, object]:
        import anthropic  # local import: only required when this path is actually used

        client = anthropic.Anthropic(api_key=self._api_key) if self._api_key else anthropic.Anthropic()
        response = client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": request}],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": _TARGET_JSON_SCHEMA},
            },
        )
        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text)


def get_provider(*, provider_name: str | None = None, api_key: str | None = None) -> tuple[LLMProvider, str]:
    """Resolves NEBULA_LLM_PROVIDER/NEBULA_LLM_API_KEY (or explicit
    overrides, e.g. from CLI flags) into a provider instance. Returns
    (provider, requested_provider_name) -- the caller uses the latter to
    distinguish "mock requested" from "anthropic requested but it may still
    fall back internally at call time".
    """

    name = (provider_name or os.environ.get(PROVIDER_ENV_VAR) or DEFAULT_PROVIDER).strip().lower()
    key = api_key if api_key is not None else os.environ.get(API_KEY_ENV_VAR)

    if name == "mock":
        return MockLLMProvider(), name
    if name == "anthropic":
        return AnthropicLLMProvider(api_key=key), name
    raise ValueError(f"unknown {PROVIDER_ENV_VAR}={name!r}; expected 'anthropic' or 'mock'")
