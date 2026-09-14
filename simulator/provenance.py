"""Content fingerprints and reproducibility metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping


_SPICE_INCLUDE_RE = re.compile(
    r"(?im)^\s*\.inc(?:lude)?\s+(?:\"([^\"]+)\"|'([^']+)'|([^\s;]+))"
)
_SPICE_LIB_RE = re.compile(r'''(?im)^\s*\.lib\s+(?:"([^"]+)"|'([^']+)'|([^\s;]+))[ \t]+[^\s;]+''')


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


def spice_dependency_manifest(path: str | Path) -> dict[str, str]:
    """Hash a SPICE file and every literal relative ``.include`` dependency."""
    root = Path(path).expanduser().resolve()
    if not root.is_file():
        raise FileNotFoundError(f"SPICE model file does not exist: {root}")
    pending = [root]
    visited: set[Path] = set()
    dependencies: dict[str, str] = {}
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        text = current.read_text(encoding="utf-8", errors="replace")
        relative = os.path.relpath(current, root.parent).replace("\\", "/")
        dependencies[relative] = sha256_text(text)
        for match in (*_SPICE_INCLUDE_RE.finditer(text), *_SPICE_LIB_RE.finditer(text)):
            value = next(group for group in match.groups() if group is not None)
            if "{" in value or "$" in value:
                raise ValueError(f"dynamic SPICE include cannot be fingerprinted: {value}")
            included = (current.parent / value).resolve()
            if not included.is_file():
                raise FileNotFoundError(f"SPICE include does not exist: {included}")
            pending.append(included)
    return dict(sorted(dependencies.items()))


def spice_dependency_fingerprint(path: str | Path) -> str:
    """Return a content fingerprint for a SPICE file's complete include closure."""
    return stable_fingerprint(spice_dependency_manifest(path))


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
