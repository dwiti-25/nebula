"""Small content-addressed JSON cache for scalar evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class EvaluationCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, evaluation_id: str) -> Path:
        return self.root / evaluation_id[:2] / f"{evaluation_id}.json"

    def get(self, evaluation_id: str) -> dict[str, object] | None:
        path = self._path(evaluation_id)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if value.get("evaluation_id") == evaluation_id else None

    def put(self, evaluation_id: str, value: Mapping[str, object]) -> Path:
        path = self._path(evaluation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        payload = {**value, "evaluation_id": evaluation_id}
        temporary.write_text(json.dumps(payload, sort_keys=True, default=str), encoding="utf-8")
        temporary.replace(path)
        return path
