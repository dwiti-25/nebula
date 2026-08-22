"""Content fingerprints and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fingerprint(values: Mapping[str, object]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(encoded)


def git_identity(root: str | Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout.strip())
        return {"git_commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "working_tree_dirty": None}
