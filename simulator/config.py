"""Portable local configuration for SKY130-backed simulations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


SKY130_MODEL_ENV = "SKY130_MODEL_LIBRARY"


@dataclass(frozen=True)
class Sky130Config:
    model_library: str | Path | None = None

    def resolve_model_library(self) -> Path:
        requested = self.model_library or os.environ.get(SKY130_MODEL_ENV)
        if not requested:
            raise FileNotFoundError(
                f"Set {SKY130_MODEL_ENV} to the full path of sky130.lib.spice"
            )
        path = Path(requested).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SKY130 model library does not exist: {path}")
        return path.resolve()
